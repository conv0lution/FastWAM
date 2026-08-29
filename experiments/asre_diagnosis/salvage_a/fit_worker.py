"""Fit matched SVD/ActionAware shards or collect held-out Delta-Z Grams."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from hydra.utils import instantiate


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_A_PROTOCOL,
    atomic_write_json,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.g0.machinery_tests import _compose  # noqa: E402
from experiments.asre_diagnosis.salvage_a.action_sensitivity import (  # noqa: E402
    ACTION_SHAPE,
    EXPECTED_VJPS_PER_STATE,
    INTERPOLATION_LAMBDAS,
    PROBE_SEEDS,
    accumulate_covariance_,
    frozen_probe_records,
    rademacher_probe,
)
from experiments.asre_diagnosis.salvage_a.basis import (  # noqa: E402
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
)
from experiments.asre_diagnosis.salvage_a.donor import (  # noqa: E402
    SalvageADonorBundle,
    load_donor_image,
)
from experiments.libero.eval_libero_single import (  # noqa: E402
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


EARLY_DISABLED = tuple(range(15))
TASK_CONFIG = "libero_uncond_2cam224_1e-4"
EXPECTED_TOKENS = 98
EXPECTED_HEADS = 24
EXPECTED_HEAD_DIM = 128


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_model(checkpoint: Path) -> tuple[torch.nn.Module, Any]:
    cfg = _compose(TASK_CONFIG)
    cfg.model.load_text_encoder = False
    cfg.EVALUATION.text_encoder_device = None
    cfg.EVALUATION.compile_action_infer = False
    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=dtype, device="cuda:0")
    _load_model_checkpoint(model, str(checkpoint))
    model = model.to("cuda:0").eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Failed to freeze every model parameter.")
    return model, cfg


def _selection_ids(selection: Mapping[str, Any], phase: str) -> list[str]:
    key = "calibration_sample_ids" if phase == "fit" else "heldout_sample_ids"
    values = selection.get(key)
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ValueError(f"State-selection manifest is missing {key}.")
    if len(values) != 100 or len(set(values)) != 100:
        raise ValueError(f"{key} must contain exactly 100 unique selected states.")
    return list(values)


def _stability_half_by_sample(selection: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for half in ("a", "b"):
        values = selection.get(f"stability_half_{half}_sample_ids")
        if not isinstance(values, list) or len(values) != 50:
            raise ValueError(
                f"State-selection manifest requires 50 stability_half_{half}_sample_ids."
            )
        for identifier in values:
            if not isinstance(identifier, str) or identifier in result:
                raise ValueError("Stability-half state selections overlap or are malformed.")
            result[identifier] = half
    calibration = set(_selection_ids(selection, "fit"))
    if set(result) != calibration:
        raise ValueError("Stability halves do not partition calibration states exactly.")
    return result


def _load_sample(
    *, record: Mapping[str, Any], source_path: Path
) -> dict[str, Any]:
    sample_path = Path(str(record["sample_path"]))
    if not sample_path.is_absolute():
        sample_path = source_path.parent / sample_path
    payload = torch.load(sample_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("sample_id") != record.get("sample_id"):
        raise ValueError(f"Malformed state-bank sample: {sample_path}")
    return payload


def _validate_layout(layout: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(layout)
    if (
        observed.get("feature_dim") != EXPECTED_FEATURE_DIM
        or observed.get("action_visible_token_count") != EXPECTED_TOKENS
        or observed.get("action_visible_token_indices") != list(range(EXPECTED_TOKENS))
        or observed.get("num_heads") != EXPECTED_HEADS
        or observed.get("head_dim") != EXPECTED_HEAD_DIM
        or observed.get("num_layers") != 30
    ):
        raise ValueError(f"Unexpected Salvage-A cache layout: {observed}")
    # The exported Delta-Z layer shard is an analysis request, not a runtime
    # cache-layout property, and differs between the two registered lambdas.
    observed.pop("delta_layers", None)
    observed.pop("action_sensitive_layers", None)
    return observed


def _worker_gpu(index: int) -> tuple[int, tuple[int, ...]]:
    if index not in range(4):
        raise ValueError("worker-index must be 0..3.")
    physical = os.environ.get("ASRE_SALVAGE_A_PHYSICAL_GPU")
    if physical is None or os.environ.get("CUDA_VISIBLE_DEVICES") != physical:
        raise RuntimeError("Salvage-A worker requires an isolated auditable physical GPU.")
    if not torch.cuda.is_available():
        raise RuntimeError("Salvage-A fitting requires CUDA.")
    return int(physical), tuple(LATE_LAYERS[index::4])


def _inputs(
    args: argparse.Namespace,
) -> tuple[
    Path,
    dict[str, Any],
    list[str],
    dict[str, Mapping[str, Any]],
    SalvageADonorBundle,
]:
    split_path = args.split.resolve()
    split = _read(split_path)
    if split.get("protocol") != SALVAGE_A_PROTOCOL:
        raise ValueError("Incompatible Salvage-A split manifest.")
    selection_path = args.state_selection.resolve()
    selection = _read(selection_path)
    recorded_selection_split = selection.get(
        "split_manifest_sha256", selection.get("split_sha256")
    )
    if (
        selection.get("protocol") != SALVAGE_A_PROTOCOL
        or recorded_selection_split != sha256_file(split_path)
    ):
        raise ValueError("State selection does not bind the supplied split.")
    sample_ids = _selection_ids(selection, args.phase)
    source_path = Path(str(split["source_manifest_path"])).resolve()
    source_records = load_manifest(source_path)
    by_id = {str(record["sample_id"]): record for record in source_records}
    if any(identifier not in by_id for identifier in sample_ids):
        raise ValueError("Selected state is absent from the frozen state bank.")
    bundle = SalvageADonorBundle.load(
        mapping_path=args.donor_mapping.resolve(),
        observation_manifest_path=args.donor_manifest.resolve(),
        observation_root=args.donor_root.resolve(),
    )
    recorded_donor_split = bundle.mapping_payload.get(
        "split_manifest_sha256", bundle.mapping_payload.get("split_sha256")
    )
    if recorded_donor_split != sha256_file(split_path):
        raise ValueError("Donor mapping is not bound to the supplied split.")
    return source_path, selection, sample_ids, by_id, bundle


def _allocate(layers: tuple[int, ...]) -> dict[str, dict[int, torch.Tensor]]:
    return {
        kind: {
            layer: torch.zeros(
                (EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_DIM),
                device="cuda:0",
                dtype=torch.float32,
            )
            for layer in layers
        }
        for kind in ("k", "v")
    }


def _accumulate_delta_grams(
    targets: Mapping[str, Mapping[int, torch.Tensor]],
    deltas: Mapping[str, Mapping[int, torch.Tensor]],
    layers: tuple[int, ...],
) -> None:
    for kind in ("k", "v"):
        for layer in layers:
            values = deltas[kind][layer].reshape(-1, EXPECTED_FEATURE_DIM).float()
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"Non-finite Delta-Z at layer {layer} {kind}.")
            targets[kind][layer].addmm_(values.T, values)


def _run_fit(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    layers: tuple[int, ...],
    source_path: Path,
    selection: Mapping[str, Any],
    sample_ids: list[str],
    by_id: Mapping[str, Mapping[str, Any]],
    bundle: SalvageADonorBundle,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any], int, int]:
    grams = _allocate(layers)
    half_covariances = {half: _allocate(layers) for half in ("a", "b")}
    half_by_sample = _stability_half_by_sample(selection)
    probes = {
        interpolation_lambda: [
            rademacher_probe(seed).to(device="cuda:0")
            for seed in PROBE_SEEDS[interpolation_lambda]
        ]
        for interpolation_lambda in INTERPOLATION_LAMBDAS
    }
    layout: dict[str, Any] | None = None
    row_count = 0
    vjp_count = 0
    for position, identifier in enumerate(sample_ids, start=1):
        record = by_id[identifier]
        sample = _load_sample(record=record, source_path=source_path)
        infer_kwargs = dict(sample["infer_action_kwargs"])
        if (
            int(infer_kwargs.get("num_inference_steps", -1)) != 10
            or int(infer_kwargs.get("seed", -1)) != 42
            or str(infer_kwargs.get("rand_device")) != "cpu"
        ):
            raise ValueError("Selected state does not use frozen 10-step CPU-noise inference.")
        donor = load_donor_image(
            bundle,
            task_id=int(record["task_id"]),
            episode_id=int(record["episode_id"]),
        ).to(dtype=infer_kwargs["input_image"].dtype)
        for interpolation_lambda in INTERPOLATION_LAMBDAS:
            export_deltas = interpolation_lambda == INTERPOLATION_LAMBDAS[0]
            result = model.infer_action(
                **infer_kwargs,
                disabled_video_layers=EARLY_DISABLED,
                replacement_input_image=donor,
                replacement_video_layers=LATE_LAYERS,
                action_sensitive_interpolation_lambda=interpolation_lambda,
                action_sensitive_layers=LATE_LAYERS,
                action_sensitive_gradient_checkpointing=True,
                return_video_cache_deltas=export_deltas,
                video_cache_delta_layers=layers if export_deltas else None,
                compile_action_infer=False,
            )
            observed_layout = _validate_layout(result["video_cache_layout"])
            if layout is None:
                layout = observed_layout
            elif observed_layout != layout:
                raise ValueError("Runtime cache layout drifted across calibration states.")
            action = result["action"]
            if tuple(action.shape) != ACTION_SHAPE or not bool(
                torch.isfinite(action).all().item()
            ):
                raise ValueError("Differentiable raw normalized action is malformed.")
            leaves = result["action_sensitive_cache_tensors"]
            leaf_keys = [
                (kind, layer) for kind in ("k", "v") for layer in layers
            ]
            target_leaves = tuple(leaves[kind][layer] for kind, layer in leaf_keys)
            if not all(
                leaf.requires_grad
                and tuple(leaf.shape) == (1, EXPECTED_TOKENS, EXPECTED_FEATURE_DIM)
                for leaf in target_leaves
            ):
                raise ValueError("Injected cache leaves have wrong shape or grad state.")
            for probe_index, probe in enumerate(probes[interpolation_lambda]):
                scalar = torch.sum(action * probe)
                gradients = torch.autograd.grad(
                    scalar,
                    target_leaves,
                    retain_graph=probe_index + 1 < len(probes[interpolation_lambda]),
                    create_graph=False,
                    allow_unused=False,
                )
                for (kind, layer), gradient in zip(leaf_keys, gradients):
                    if not bool(torch.isfinite(gradient).all().item()) or not bool(
                        gradient.detach().abs().max().item() > 0.0
                    ):
                        raise ValueError(
                            f"Invalid VJP for {identifier}, lambda={interpolation_lambda}, "
                            f"layer={layer}, kind={kind}."
                        )
                    accumulate_covariance_(
                        half_covariances[half_by_sample[identifier]][kind][layer],
                        gradient,
                    )
                vjp_count += 1
            if export_deltas:
                _accumulate_delta_grams(grams, result["video_cache_deltas"], layers)
                row_count += EXPECTED_TOKENS
            # Drop every reference into the just-consumed ten-step graph before
            # constructing the next lambda graph.  Keeping the final scalar or
            # VJP tuple alive here can otherwise double the transient GPU peak.
            del result, action, leaves, target_leaves, scalar, gradients, gradient
        if position % 5 == 0:
            print(
                f"[Salvage-A fit shard {args.worker_index}] "
                f"{position}/{len(sample_ids)} states",
                flush=True,
            )
    if vjp_count != len(sample_ids) * EXPECTED_VJPS_PER_STATE:
        raise RuntimeError("The registered four-VJP-per-state schedule was not executed.")
    assert layout is not None
    matrices = {
        "gram": grams,
        "half_a": half_covariances["a"],
        "half_b": half_covariances["b"],
    }
    return matrices, layout, row_count, vjp_count


def _run_heldout(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    layers: tuple[int, ...],
    source_path: Path,
    sample_ids: list[str],
    by_id: Mapping[str, Mapping[str, Any]],
    bundle: SalvageADonorBundle,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any], int]:
    grams = _allocate(layers)
    layout: dict[str, Any] | None = None
    row_count = 0
    with torch.inference_mode():
        for position, identifier in enumerate(sample_ids, start=1):
            record = by_id[identifier]
            sample = _load_sample(record=record, source_path=source_path)
            infer_kwargs = dict(sample["infer_action_kwargs"])
            donor = load_donor_image(
                bundle,
                task_id=int(record["task_id"]),
                episode_id=int(record["episode_id"]),
            ).to(dtype=infer_kwargs["input_image"].dtype)
            result = model.infer_action(
                **infer_kwargs,
                disabled_video_layers=EARLY_DISABLED,
                replacement_input_image=donor,
                replacement_video_layers=LATE_LAYERS,
                return_video_cache_deltas=True,
                video_cache_delta_layers=layers,
                cache_only=True,
                compile_action_infer=False,
            )
            observed_layout = _validate_layout(result["video_cache_layout"])
            if layout is None:
                layout = observed_layout
            elif observed_layout != layout:
                raise ValueError("Runtime cache layout drifted across held-out states.")
            _accumulate_delta_grams(grams, result["video_cache_deltas"], layers)
            row_count += EXPECTED_TOKENS
            if position % 25 == 0:
                print(
                    f"[Salvage-A heldout shard {args.worker_index}] "
                    f"{position}/{len(sample_ids)} states",
                    flush=True,
                )
    assert layout is not None
    return grams, layout, row_count


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    physical_gpu, layers = _worker_gpu(args.worker_index)
    source_path, selection, sample_ids, by_id, bundle = _inputs(args)
    checkpoint = args.checkpoint.resolve()
    differentiable_path = args.differentiable_path_report.resolve()
    differentiable = _read(differentiable_path)
    expected_differentiable_links = {
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "split_sha256": sha256_file(args.split.resolve()),
        "state_selection_sha256": sha256_file(args.state_selection.resolve()),
        "donor_mapping_sha256": bundle.mapping_sha256,
        "donor_manifest_sha256": bundle.observation_manifest_sha256,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    if (
        differentiable.get("protocol") != SALVAGE_A_PROTOCOL
        or differentiable.get("passed") is not True
        or any(
            differentiable.get(key) != value
            for key, value in expected_differentiable_links.items()
        )
    ):
        raise ValueError(
            "Salvage-A fitting requires the matching passed differentiable-path gate."
        )
    set_global_seed(42, get_worker_init_fn=False)
    model, _cfg = _load_model(checkpoint)

    if args.phase == "fit":
        matrices, layout, row_count, vjp_count = _run_fit(
            args=args,
            model=model,
            layers=layers,
            source_path=source_path,
            selection=selection,
            sample_ids=sample_ids,
            by_id=by_id,
            bundle=bundle,
        )
    else:
        heldout, layout, row_count = _run_heldout(
            args=args,
            model=model,
            layers=layers,
            source_path=source_path,
            sample_ids=sample_ids,
            by_id=by_id,
            bundle=bundle,
        )
        matrices = {"gram": heldout}
        vjp_count = 0

    output_dir = args.output_dir.resolve()
    artifact_root = output_dir / f"{args.phase}_artifacts"
    artifacts: dict[str, dict[str, Any]] = {}
    for kind in ("k", "v"):
        for layer in layers:
            matrix = f"layer{layer:02d}_{kind}"
            path = artifact_root / f"{matrix}.pt"
            payload: dict[str, Any] = {
                "artifact_type": f"asre_salvage_a_{args.phase}_matrix",
                "schema_version": 1,
                "protocol": SALVAGE_A_PROTOCOL,
                "phase": args.phase,
                "layer": layer,
                "tensor_kind": kind,
                "sample_count": len(sample_ids),
                "rows": row_count,
                "split_sha256": sha256_file(args.split.resolve()),
                "state_selection_sha256": sha256_file(
                    args.state_selection.resolve()
                ),
                "gram": matrices["gram"][kind][layer].cpu(),
            }
            if args.phase == "fit":
                half_a = matrices["half_a"][kind][layer].cpu()
                half_b = matrices["half_b"][kind][layer].cpu()
                payload.update(
                    {
                        "covariance": half_a + half_b,
                        "half_a_covariance": half_a,
                        "half_b_covariance": half_b,
                        "vjp_count": vjp_count,
                        "interpolation_lambdas": list(INTERPOLATION_LAMBDAS),
                        "probe_records": frozen_probe_records(),
                    }
                )
            _atomic_torch_save(path, payload)
            artifacts[matrix] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "shape": [EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_DIM],
                "total_delta_z_energy": float(
                    torch.trace(matrices["gram"][kind][layer]).item()
                ),
                "total_action_sensitivity": (
                    float(
                        torch.trace(
                            matrices["half_a"][kind][layer]
                            + matrices["half_b"][kind][layer]
                        ).item()
                    )
                    if args.phase == "fit"
                    else None
                ),
            }

    report = {
        "artifact_type": f"asre_salvage_a_{args.phase}_shard_report",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "status": "complete",
        "phase": args.phase,
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "worker_index": args.worker_index,
        "physical_gpu": physical_gpu,
        "logical_device": "cuda:0",
        "no_ddp": True,
        "layers": list(layers),
        "sample_count": len(sample_ids),
        "sample_ids_sha256": sha256_json(sample_ids),
        "row_count_per_matrix": row_count,
        "vjp_count": vjp_count,
        "vjp_count_per_state": (
            EXPECTED_VJPS_PER_STATE if args.phase == "fit" else 0
        ),
        "probe_records": frozen_probe_records() if args.phase == "fit" else [],
        "split_path": str(args.split.resolve()),
        "split_sha256": sha256_file(args.split.resolve()),
        "state_selection_path": str(args.state_selection.resolve()),
        "state_selection_sha256": sha256_file(args.state_selection.resolve()),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "donor_mapping_sha256": bundle.mapping_sha256,
        "donor_manifest_sha256": bundle.observation_manifest_sha256,
        "differentiable_path_report_path": str(differentiable_path),
        "differentiable_path_report_sha256": sha256_file(differentiable_path),
        "runtime_layout": layout,
        "model_parameters_frozen": True,
        "full_action_denoising_steps": 10 if args.phase == "fit" else None,
        "gradient_checkpointing": args.phase == "fit",
        "artifacts": artifacts,
    }
    report_path = output_dir / f"{args.phase}_shard{args.worker_index}.json"
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "heldout"), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--state-selection", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--differentiable-path-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    report = run_worker(parser.parse_args())
    print(
        f"Salvage-A {report['phase']} shard {report['worker_index']} complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
