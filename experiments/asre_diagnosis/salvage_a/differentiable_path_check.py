"""Real-checkpoint equivalence and VJP gate for the exact Salvage-A objective."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


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
)
from experiments.asre_diagnosis.salvage_a.action_sensitivity import (  # noqa: E402
    ACTION_SHAPE,
    PROBE_SEEDS,
    probe_sha256,
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
from experiments.asre_diagnosis.salvage_a.fit_worker import (  # noqa: E402
    EARLY_DISABLED,
    EXPECTED_HEADS,
    EXPECTED_HEAD_DIM,
    EXPECTED_TOKENS,
    _load_model,
    _load_sample,
    _read,
    _validate_layout,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


ACTION_TOLERANCE = 2.0e-2


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left = left.detach().float().cpu()
    right = right.detach().float().cpu()
    difference = (left - right).abs()
    return {
        "shape_equal": list(left.shape) == list(right.shape),
        "max_raw_action_abs_diff": float(difference.max().item()),
        "mean_raw_action_abs_diff": float(difference.mean().item()),
        "allclose_atol_rtol_2e_2": bool(
            torch.allclose(
                left,
                right,
                atol=ACTION_TOLERANCE,
                rtol=ACTION_TOLERANCE,
            )
        ),
    }


def _model_parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Differentiable-path validation requires CUDA.")
    preflight_path = args.preflight.resolve()
    preflight = _read(preflight_path)
    if (
        preflight.get("protocol") != SALVAGE_A_PROTOCOL
        or preflight.get("status") != "compatible"
    ):
        raise ValueError("Salvage-A preflight did not pass.")
    split_path = args.split.resolve()
    split = _read(split_path)
    selection_path = args.state_selection.resolve()
    selection = _read(selection_path)
    recorded_split = selection.get(
        "split_manifest_sha256", selection.get("split_sha256")
    )
    if (
        split.get("protocol") != SALVAGE_A_PROTOCOL
        or recorded_split != sha256_file(split_path)
    ):
        raise ValueError("Differentiable check received incompatible split inputs.")
    sample_ids = selection.get("calibration_sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != 100:
        raise ValueError("Expected 100 frozen calibration states.")
    source_path = Path(str(split["source_manifest_path"])).resolve()
    records = {str(row["sample_id"]): row for row in load_manifest(source_path)}
    sample_id = str(sample_ids[0])
    record = records[sample_id]
    sample = _load_sample(record=record, source_path=source_path)
    infer_kwargs = dict(sample["infer_action_kwargs"])
    if int(infer_kwargs.get("num_inference_steps", -1)) != 10:
        raise ValueError("Equivalence check must use full 10-step denoising.")
    bundle = SalvageADonorBundle.load(
        mapping_path=args.donor_mapping.resolve(),
        observation_manifest_path=args.donor_manifest.resolve(),
        observation_root=args.donor_root.resolve(),
    )
    donor = load_donor_image(
        bundle,
        task_id=int(record["task_id"]),
        episode_id=int(record["episode_id"]),
    ).to(dtype=infer_kwargs["input_image"].dtype)
    set_global_seed(42, get_worker_init_fn=False)
    model, _cfg = _load_model(args.checkpoint.resolve())
    parameter_sha_before = _model_parameter_sha256(model)
    with torch.no_grad():
        current = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            compile_action_infer=False,
        )
        wrong = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor,
            replacement_video_layers=LATE_LAYERS,
            compile_action_infer=False,
        )
    endpoint_current_result = model.infer_action(
        **infer_kwargs,
        disabled_video_layers=EARLY_DISABLED,
        replacement_input_image=donor,
        replacement_video_layers=LATE_LAYERS,
        action_sensitive_interpolation_lambda=1.0,
        action_sensitive_layers=LATE_LAYERS,
        action_sensitive_gradient_checkpointing=True,
        compile_action_infer=False,
    )
    endpoint_current_action = endpoint_current_result["action"].detach().cpu()
    del endpoint_current_result
    torch.cuda.empty_cache()
    endpoint_wrong_result = model.infer_action(
        **infer_kwargs,
        disabled_video_layers=EARLY_DISABLED,
        replacement_input_image=donor,
        replacement_video_layers=LATE_LAYERS,
        action_sensitive_interpolation_lambda=0.0,
        action_sensitive_layers=LATE_LAYERS,
        action_sensitive_gradient_checkpointing=True,
        compile_action_infer=False,
    )
    endpoint_wrong_action = endpoint_wrong_result["action"].detach().cpu()
    del endpoint_wrong_result
    torch.cuda.empty_cache()
    midpoint = model.infer_action(
        **infer_kwargs,
        disabled_video_layers=EARLY_DISABLED,
        replacement_input_image=donor,
        replacement_video_layers=LATE_LAYERS,
        action_sensitive_interpolation_lambda=0.5,
        action_sensitive_layers=LATE_LAYERS,
        action_sensitive_gradient_checkpointing=True,
        compile_action_infer=False,
    )
    current_comparison = _comparison(endpoint_current_action, current["action"])
    wrong_comparison = _comparison(endpoint_wrong_action, wrong["action"])
    layout = _validate_layout(midpoint["video_cache_layout"])
    leaves = midpoint["action_sensitive_cache_tensors"]
    keys = [(kind, layer) for kind in ("k", "v") for layer in LATE_LAYERS]
    tensors = tuple(leaves[kind][layer] for kind, layer in keys)
    injected_leaves_valid = len(tensors) == 30 and all(
        tensor.requires_grad
        and tensor.is_leaf
        and tuple(tensor.shape) == (1, EXPECTED_TOKENS, EXPECTED_FEATURE_DIM)
        for tensor in tensors
    )
    probe = rademacher_probe(PROBE_SEEDS[0.5][0]).to(device="cuda:0")
    gradients = torch.autograd.grad(
        torch.sum(midpoint["action"] * probe),
        tensors,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    vjp_rows: list[dict[str, Any]] = []
    for (kind, layer), tensor, gradient in zip(keys, tensors, gradients):
        finite = bool(torch.isfinite(gradient).all().item())
        nonzero = bool(gradient.detach().abs().max().item() > 0.0)
        vjp_rows.append(
            {
                "matrix": f"layer{layer:02d}_{kind}",
                "layer": layer,
                "tensor_kind": kind.upper(),
                "leaf_shape": list(tensor.shape),
                "gradient_shape": list(gradient.shape),
                "leaf_requires_grad": tensor.requires_grad,
                "leaf_is_leaf": tensor.is_leaf,
                "finite": finite,
                "nonzero": nonzero,
                "l2_norm": float(torch.linalg.vector_norm(gradient.float()).item()),
            }
        )
    parameters_frozen = all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.parameters()
    )
    parameter_sha_after = _model_parameter_sha256(model)
    parameters_unchanged = parameter_sha_before == parameter_sha_after
    action_shapes_finite = all(
        tuple(action.shape) == ACTION_SHAPE
        and bool(torch.isfinite(action).all().item())
        for action in (
            current["action"],
            wrong["action"],
            endpoint_current_action,
            endpoint_wrong_action,
            midpoint["action"],
        )
    )
    vjp_passed = all(
        row["leaf_shape"] == [1, EXPECTED_TOKENS, EXPECTED_FEATURE_DIM]
        and row["gradient_shape"] == [1, EXPECTED_TOKENS, EXPECTED_FEATURE_DIM]
        and row["leaf_requires_grad"]
        and row["leaf_is_leaf"]
        and row["finite"]
        and row["nonzero"]
        for row in vjp_rows
    )
    passed = bool(
        current_comparison["allclose_atol_rtol_2e_2"]
        and wrong_comparison["allclose_atol_rtol_2e_2"]
        and action_shapes_finite
        and vjp_passed
        and injected_leaves_valid
        and parameters_frozen
        and parameters_unchanged
        and layout["action_visible_token_count"] == EXPECTED_TOKENS
        and layout["num_heads"] == EXPECTED_HEADS
        and layout["head_dim"] == EXPECTED_HEAD_DIM
    )
    report = {
        "artifact_type": "asre_salvage_a_differentiable_path_report",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "status": "passed" if passed else "failed",
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "sample_id": sample_id,
        "task_id": int(record["task_id"]),
        "episode_id": int(record["episode_id"]),
        "full_action_denoising_steps": 10,
        "raw_normalized_action_shape": list(ACTION_SHAPE),
        "compile_action_infer": False,
        "gradient_checkpointing": True,
        "current_endpoint_equivalence": current_comparison,
        "wrong_endpoint_equivalence": wrong_comparison,
        "action_shapes_finite": action_shapes_finite,
        "runtime_layout": layout,
        "probe": {
            "seed": PROBE_SEEDS[0.5][0],
            "sha256": probe_sha256(probe),
            "shape": list(probe.shape),
        },
        "vjp_rows": vjp_rows,
        "vjp_passed": vjp_passed,
        "injected_cache_tensors_are_exact_leaves": injected_leaves_valid,
        "model_parameters_frozen": parameters_frozen,
        "model_parameters_unchanged": parameters_unchanged,
        "model_parameter_sha256_before": parameter_sha_before,
        "model_parameter_sha256_after": parameter_sha_after,
        "only_injected_cache_tensors_require_grad": (
            injected_leaves_valid and parameters_frozen
        ),
        "preflight_sha256": sha256_file(preflight_path),
        "split_sha256": sha256_file(split_path),
        "state_selection_sha256": sha256_file(selection_path),
        "donor_mapping_sha256": bundle.mapping_sha256,
        "donor_manifest_sha256": bundle.observation_manifest_sha256,
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "passed": passed,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--state-selection", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    atomic_write_json(args.output.resolve(), report)
    if not report["passed"]:
        raise RuntimeError("Exact differentiable action path failed validation.")
    print(f"Salvage-A differentiable path passed: {args.output.resolve()}")


if __name__ == "__main__":
    main()
