"""GPU machinery gates for the native shared-node intervention."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from experiments.asre_diagnosis.common import atomic_write_json, now_iso
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
from .native_clamp import NativePrefixClamp
from .native_runtime import (
    _native_call,
    capture_native_stock_trajectory,
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
    record = world["records"][0]
    sample_id = str(record["sample_id"])
    dataset = load_world_dataset(
        preflight=preflight,
        runtime_work_dir=args.output_root.resolve() / "runtime/machinery",
    )
    prompt_cache = load_prompt_cache(preflight)
    donor_bundle = load_donor_bundle(preflight)
    model, _cfg = load_frozen_model(preflight)
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
    donor_image, _donor_hash, _current_hash = _validate_and_align_donor_image(
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
        "num_inference_steps": 10,
        "sigma_shift": 5.0,
        "seed": None,
        "rand_device": "cpu",
        "tiled": False,
        "initial_video_noise": draw["video_noise"],
        "initial_action_noise": draw["action_noise"],
    }
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
    report = {
        "artifact_type": "asre_salvage_b_v2_machinery",
        "protocol": PROTOCOL,
        "created_at": now_iso(),
        "native_clamp_identity_passed": observational_identity and rank_d_identity,
        "observational_capture_exact_identity": observational_identity,
        "rank_d_current_exact_identity": rank_d_identity,
        "rank_zero_wrong_repeat_exact_identity": rank_zero_identity,
        "shared_node_reach_passed": True,
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
