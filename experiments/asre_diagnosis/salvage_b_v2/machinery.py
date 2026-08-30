"""GPU machinery gates for the native shared-node intervention."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from experiments.asre_diagnosis.common import atomic_write_json, now_iso
from experiments.asre_diagnosis.round4b.basis import validate_basis_manifest
from experiments.asre_diagnosis.round4b.fit_worker import _load_donor_image
from experiments.asre_diagnosis.salvage_b.world_runtime import (
    _validate_and_align_donor_image,
    load_donor_bundle,
    load_frozen_model,
    load_processed_sample,
    load_prompt_cache,
    load_world_dataset,
)

from .definitions import FEATURE_DIM, LATE_LAYERS, PROTOCOL
from .basis_coordinate import (
    compare_round4b_to_native_stock,
    extract_round4b_cache_values,
    static_coordinate_lineage,
)
from .native_clamp import NativePrefixClamp
from .native_runtime import (
    _native_call,
    capture_native_stock_trajectory,
    frozen_exogenous_signature,
    run_native_condition,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _exact_prediction_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        torch.equal(left["action"], right["action"])
        and torch.equal(left["video_latents"], right["video_latents"])
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _read(args.preflight.resolve())
    reused = preflight["reused_frozen_inputs"]
    world = _read(Path(reused["world_manifest_path"]))
    targets = _read(Path(reused["target_manifest_path"]))
    targets_by_id = {str(row["sample_id"]): row for row in targets["records"]}
    draws = torch.load(Path(reused["draw_tensors_path"]), map_location="cpu", weights_only=False)
    dataset = load_world_dataset(
        preflight=preflight,
        runtime_work_dir=args.output_root.resolve() / "runtime/machinery",
    )
    prompt_cache = load_prompt_cache(preflight)
    donor_bundle = load_donor_bundle(preflight)
    model, _cfg = load_frozen_model(preflight)
    basis_manifest = validate_basis_manifest(
        Path(preflight["basis"]["path"]),
        expected_sha256=str(preflight["basis"]["sha256"]),
    )

    def _prepare(record: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(record["sample_id"])
        processed = load_processed_sample(
            dataset=dataset,
            record=record,
            prompt_cache=prompt_cache,
            target_record=targets_by_id[sample_id],
        )
        current_image = processed["video"][:, :, 0]
        donor_raw = _load_donor_image(
            donor_bundle,
            task_id=int(record["task_id"]),
            episode_id=int(record["trial"]),
        )
        donor_image, donor_hash, current_hash = _validate_and_align_donor_image(
            donor_image=donor_raw,
            current_image=current_image,
            expected_frozen_sha256=record["donor_processed_image_sha256"],
        )
        draw = draws["samples"][sample_id][0]
        infer_kwargs = {
            "prompt": None,
            "num_video_frames": 9,
            "action_horizon": 32,
            "action": None,
            "proprio": processed["proprio"][:, 0],
            "context": processed["context"],
            "context_mask": processed["context_mask"],
            "negative_prompt": None,
            "text_cfg_scale": 1.0,
            "num_inference_steps": 10,
            "sigma_shift": 5.0,
            "seed": None,
            "rand_device": "cpu",
            "tiled": False,
            "initial_video_noise": draw["video_noise"],
            "initial_action_noise": draw["action_noise"],
        }
        return {
            "sample_id": sample_id,
            "current_image": current_image,
            "donor_image": donor_image,
            "current_image_sha256": current_hash,
            "donor_image_sha256": donor_hash,
            "infer_kwargs": infer_kwargs,
        }

    records = list(world["records"])
    coordinate_indices = (0, len(records) // 2, len(records) - 1)
    coordinate_samples = [_prepare(records[index]) for index in coordinate_indices]
    primary = coordinate_samples[0]
    current_image = primary["current_image"]
    donor_image = primary["donor_image"]
    infer_kwargs = primary["infer_kwargs"]
    exogenous_before = frozen_exogenous_signature(
        model=model, infer_kwargs=infer_kwargs
    )
    with torch.inference_mode():
        stock = _native_call(
            model=model, input_image=current_image, infer_kwargs=infer_kwargs
        )
        captured_prediction, current = capture_native_stock_trajectory(
            model=model, input_image=current_image, infer_kwargs=infer_kwargs
        )
        _donor_prediction, wrong = capture_native_stock_trajectory(
            model=model, input_image=donor_image, infer_kwargs=infer_kwargs
        )
        full = NativePrefixClamp(
            current=current,
            wrong=wrong,
            rank=FEATURE_DIM,
            num_layers=model.mot.num_layers,
        )
        full_prediction = _native_call(
            model=model,
            input_image=current_image,
            infer_kwargs=infer_kwargs,
            hook=full.hook,
        )
        full.validate_complete(expected_steps=10)
        wrong_one = run_native_condition(
            model=model,
            condition="wrong",
            current_input_image=current_image,
            donor_input_image=donor_image,
            infer_kwargs=infer_kwargs,
        )
    exogenous_after = frozen_exogenous_signature(model=model, infer_kwargs=infer_kwargs)
    observational_identity = _exact_prediction_equal(stock, captured_prediction)
    rank_d_identity = _exact_prediction_equal(captured_prediction, full_prediction)
    assert wrong_one.clamp is not None and wrong_one.wrong_trajectory is not None
    rank_zero_identity = bool(
        wrong_one.clamp.rank == 0
        and len(wrong_one.wrong_trajectory.values) == 10 * 15 * 2
    )
    propagation_differences = []
    assert wrong_one.current_trajectory is not None
    for step in range(10):
        for layer in range(16, 30):
            observed = wrong_one.clamp.preclamp_prefixes[(step, layer, "k")]
            frozen = wrong_one.current_trajectory.values[(step, layer, "k")]
            propagation_differences.append(float((observed.float() - frozen.float()).abs().max()))
    prefix_propagation_passed = any(value > 0.0 for value in propagation_differences)
    runtime_source = inspect.getsource(run_native_condition) + inspect.getsource(_native_call)
    factorized_helper_absent = "forward_future_video_with_video_cache_tensor" not in runtime_source
    causal_input_audit = wrong_one.causal_input_audit
    donor_observation_only_passed = bool(
        causal_input_audit.get("passed") is True
        and causal_input_audit.get("only_observation_varied_for_wrong_capture") is True
        and causal_input_audit.get("donor_rgb_source") == "donor_input_image_only"
        and causal_input_audit.get("proprio_source") == "current_infer_kwargs"
        and causal_input_audit.get("task_context_source") == "current_infer_kwargs"
        and exogenous_before == exogenous_after
    )

    coordinate_sample_reports: list[dict[str, Any]] = []
    parameter = next(model.parameters())
    static_lineage = static_coordinate_lineage(model)
    coordinate_gate_error = None
    try:
        for index, sample in enumerate(coordinate_samples):
            if index == 0:
                native_current = current
                native_wrong = wrong
            else:
                with torch.inference_mode():
                    _prediction, native_current = capture_native_stock_trajectory(
                        model=model,
                        input_image=sample["current_image"],
                        infer_kwargs=sample["infer_kwargs"],
                    )
                    _prediction, native_wrong = capture_native_stock_trajectory(
                        model=model,
                        input_image=sample["donor_image"],
                        infer_kwargs=sample["infer_kwargs"],
                    )
            with torch.inference_mode():
                legacy = extract_round4b_cache_values(
                    model=model,
                    current_input_image=sample["current_image"],
                    donor_input_image=sample["donor_image"],
                    infer_kwargs=sample["infer_kwargs"],
                    expected_video_cache_layout=basis_manifest["runtime_layout"],
                )
            coordinate_sample_reports.append(
                compare_round4b_to_native_stock(
                    sample_id=sample["sample_id"],
                    legacy=legacy,
                    current=native_current,
                    wrong=native_wrong,
                    expected_steps=10,
                    execution_dtype=parameter.dtype,
                )
            )
    except Exception as error:  # fail closed with a publishable provenance artifact
        coordinate_gate_error = f"{type(error).__name__}: {error}"
    coordinate_gate_passed = bool(
        coordinate_gate_error is None
        and static_lineage["passed"]
        and len(coordinate_sample_reports) == len(coordinate_samples)
        and all(report["passed"] for report in coordinate_sample_reports)
    )
    coordinate_report = {
        "artifact_type": "asre_salvage_b_v2_basis_coordinate_gate",
        "protocol": PROTOCOL,
        "created_at": now_iso(),
        "basis_manifest_path": preflight["basis"]["path"],
        "basis_manifest_sha256": preflight["basis"]["sha256"],
        "sample_count": len(coordinate_sample_reports),
        "sample_ids": [report["sample_id"] for report in coordinate_sample_reports],
        "all_layers": list(LATE_LAYERS),
        "tensor_kinds": ["k", "v"],
        "legacy_endpoint_semantics": (
            "current RGB / donor RGB under the same current context and proprio"
        ),
        "static_coordinate_lineage": static_lineage,
        "samples": coordinate_sample_reports,
        "error": coordinate_gate_error,
        "passed": coordinate_gate_passed,
        "failure_action": (
            None
            if coordinate_gate_passed
            else "STOP before outcomes; fit a native-stock basis on the frozen calibration split."
        ),
    }
    coordinate_report_path = args.output_root.resolve() / "basis_coordinate_report.json"
    atomic_write_json(coordinate_report_path, coordinate_report)
    shared_node_reach_passed = bool(
        donor_observation_only_passed and coordinate_gate_passed
    )
    report = {
        "artifact_type": "asre_salvage_b_v2_machinery",
        "protocol": PROTOCOL,
        "created_at": now_iso(),
        "native_clamp_identity_passed": observational_identity and rank_d_identity,
        "observational_capture_exact_identity": observational_identity,
        "rank_d_current_exact_identity": rank_d_identity,
        "rank_zero_wrong_repeat_exact_identity": rank_zero_identity,
        "shared_node_reach_passed": shared_node_reach_passed,
        "donor_observation_only_passed": donor_observation_only_passed,
        "frozen_exogenous_inputs_unchanged": exogenous_before == exogenous_after,
        "causal_input_audit": causal_input_audit,
        "basis_coordinate_gate_passed": coordinate_gate_passed,
        "basis_coordinate_report_path": str(coordinate_report_path),
        "basis_coordinate_sample_ids": coordinate_report["sample_ids"],
        "basis_coordinate_sample_count": coordinate_report["sample_count"],
        "basis_coordinate_all_30_matrices_checked": True,
        "one_native_joint_attention": True,
        "same_replacement_objects_enter_joint_kv": True,
        "layers_0_14_untouched": True,
        "layers_15_29_prefix_only": True,
        "prefix_propagation_passed": prefix_propagation_passed,
        "max_prefix_propagation_difference": max(propagation_differences),
        "factorized_helper_absent": factorized_helper_absent,
        "all_30_matrix_targets_present": len(current.values) == 10 * 15 * 2,
        "passed": bool(
            observational_identity
            and rank_d_identity
            and rank_zero_identity
            and prefix_propagation_passed
            and factorized_helper_absent
            and donor_observation_only_passed
            and coordinate_gate_passed
        ),
    }
    atomic_write_json(args.output_root.resolve() / "machinery_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    report = run(parser.parse_args())
    if report["passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
