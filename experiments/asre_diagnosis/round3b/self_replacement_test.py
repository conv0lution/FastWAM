"""Numerical identity gate for the Round-3B cache replacement machinery."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND3B_PROTOCOL,
    atomic_write_json,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round2.validate_state_bank import (  # noqa: E402
    _validate_sample_partition,
    validate_existing_artifacts,
)


ATOL = 1.0e-4
RTOL = 1.0e-4
DISABLED_LAYERS = tuple(range(15))
REPLACEMENT_LAYERS = tuple(range(15, 30))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify same-input video-cache replacement preserves actions."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--state-bank-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task-config",
        default="libero_uncond_2cam224_1e-4",
    )
    return parser.parse_args()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _compose(task_config: str):
    with initialize_config_dir(
        config_dir=str((project_root / "configs").resolve()), version_base="1.3"
    ):
        return compose(config_name="sim_libero.yaml", overrides=[f"task={task_config}"])


def _validate_audit(audit: Any) -> dict[str, Any]:
    if not isinstance(audit, Mapping):
        raise TypeError("Replacement path did not return video_cache_stats.")
    expected_top = {
        "schema_version": 1,
        "replacement_video_layers": list(REPLACEMENT_LAYERS),
        "disabled_video_layers": list(DISABLED_LAYERS),
        "has_replacement": True,
        "summarized_video_layers": list(REPLACEMENT_LAYERS),
    }
    mismatch = {
        key: {"observed": audit.get(key), "expected": value}
        for key, value in expected_top.items()
        if audit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Replacement cache audit mismatch: {json.dumps(mismatch)}")
    if audit.get("current_video_seq_len") != audit.get("replacement_video_seq_len"):
        raise ValueError("Current/replacement video sequence lengths differ.")
    if audit.get("current_video_tokens_per_frame") != audit.get(
        "replacement_video_tokens_per_frame"
    ):
        raise ValueError("Current/replacement video token counts differ.")
    layers = audit.get("layers")
    if not isinstance(layers, list) or len(layers) != 30:
        raise ValueError(f"Expected 30 layer audit records, got {type(layers)} / {len(layers or [])}.")
    max_cache_difference = 0.0
    for expected_layer, layer in enumerate(layers):
        if not isinstance(layer, Mapping) or int(layer.get("layer", -1)) != expected_layer:
            raise ValueError(f"Cache audit layer ordering mismatch at {expected_layer}.")
        expected_source = "replacement" if expected_layer in REPLACEMENT_LAYERS else "current"
        if layer.get("selected_source") != expected_source:
            raise ValueError(
                f"Layer {expected_layer} selected {layer.get('selected_source')!r}, "
                f"expected {expected_source!r}."
            )
        if expected_layer not in REPLACEMENT_LAYERS:
            if layer.get("summarized") is not False or any(
                layer.get(key) is not None
                for key in ("current", "replacement", "difference")
            ):
                raise ValueError(
                    f"Unselected layer {expected_layer} was unexpectedly summarized."
                )
            continue
        if layer.get("summarized") is not True:
            raise ValueError(f"Replacement layer {expected_layer} was not summarized.")
        current = layer.get("current")
        replacement = layer.get("replacement")
        difference = layer.get("difference")
        if not all(isinstance(value, Mapping) for value in (current, replacement, difference)):
            raise TypeError(f"Layer {expected_layer} cache audit is incomplete.")
        for tensor_name in ("k", "v"):
            current_stats = current.get(tensor_name)
            replacement_stats = replacement.get(tensor_name)
            difference_stats = difference.get(tensor_name)
            if not all(
                isinstance(value, Mapping)
                for value in (current_stats, replacement_stats, difference_stats)
            ):
                raise TypeError(f"Layer {expected_layer} {tensor_name} audit is incomplete.")
            for key in ("shape", "dtype", "device", "numel"):
                if current_stats.get(key) != replacement_stats.get(key):
                    raise ValueError(
                        f"Layer {expected_layer} {tensor_name} {key} differs between caches."
                    )
            if current_stats.get("finite") is not True or replacement_stats.get("finite") is not True:
                raise ValueError(f"Layer {expected_layer} {tensor_name} cache is non-finite.")
            value = float(difference_stats.get("max_abs", math.inf))
            if not math.isfinite(value):
                raise ValueError(f"Layer {expected_layer} {tensor_name} difference is non-finite.")
            max_cache_difference = max(max_cache_difference, value)
    if max_cache_difference > ATOL:
        raise ValueError(
            f"Same-image cache replacement differs by {max_cache_difference}, tolerance={ATOL}."
        )
    if float(audit.get("max_abs_current_replacement", math.inf)) > ATOL:
        raise ValueError("Top-level same-image cache difference exceeds tolerance.")
    return {
        "all_replacement_caches_exact_equal": bool(
            audit.get("all_replacement_caches_exact_equal", False)
        ),
        "max_abs_current_replacement": float(audit["max_abs_current_replacement"]),
        "current_video_seq_len": int(audit["current_video_seq_len"]),
        "replacement_video_seq_len": int(audit["replacement_video_seq_len"]),
        "current_video_tokens_per_frame": int(audit["current_video_tokens_per_frame"]),
        "replacement_video_tokens_per_frame": int(
            audit["replacement_video_tokens_per_frame"]
        ),
        "action_attention_mask_shape": audit.get("action_attention_mask_shape"),
        "num_layers_audited": len(layers),
        "max_layer_cache_absolute_difference": max_cache_difference,
        "layers": layers,
    }


def run_identity_test(
    *,
    checkpoint_path: Path,
    dataset_stats_path: Path,
    valid_manifest_path: Path,
    state_bank_dir: Path,
    task_config: str,
) -> dict[str, Any]:
    # Keep LIBERO-dependent imports lazy so the pure audit validator remains
    # unit-testable without a simulator checkout on PYTHONPATH.
    from experiments.libero.eval_libero_single import (
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _resolve_eval_device,
    )
    from fastwam.utils.pytorch_utils import set_global_seed

    checkpoint_path = checkpoint_path.expanduser().resolve()
    dataset_stats_path = dataset_stats_path.expanduser().resolve()
    valid_manifest_path = valid_manifest_path.expanduser().resolve()
    state_bank_dir = state_bank_dir.expanduser().resolve()
    valid = _read_json(valid_manifest_path, label="valid state manifest")
    source_manifest_path = (state_bank_dir / "manifest.jsonl").resolve()
    prompt_cache_path = Path(str(valid["prompt_context_cache_path"])).resolve()
    source_records = load_manifest(source_manifest_path)
    valid_ids = _validate_sample_partition(valid, source_records)
    validate_existing_artifacts(
        valid_manifest_path=valid_manifest_path,
        prompt_cache_path=prompt_cache_path,
        source_manifest_path=source_manifest_path,
        source_records=source_records,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        verify_checkpoint_hash=False,
    )
    if len(valid_ids) != 499:
        raise ValueError(f"Identity gate requires exactly 499 valid states, got {len(valid_ids)}.")
    record_by_id = {str(record["sample_id"]): record for record in source_records}
    selected_record = record_by_id[valid_ids[0]]
    sample_path = Path(str(selected_record["sample_path"]))
    if not sample_path.is_absolute():
        sample_path = state_bank_dir / sample_path
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    infer_kwargs = sample.get("infer_action_kwargs")
    if not isinstance(infer_kwargs, Mapping) or not torch.is_tensor(
        infer_kwargs.get("input_image")
    ):
        raise TypeError("Selected state-bank sample has malformed infer_action_kwargs.")

    cfg = _compose(task_config)
    cfg.model.load_text_encoder = False
    cfg.EVALUATION.text_encoder_device = None
    model_device = _resolve_eval_device(cfg)
    if str(model_device) != "cuda:0":
        raise ValueError(f"Identity gate requires logical cuda:0, got {model_device}.")
    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    set_global_seed(42, get_worker_init_fn=False)
    model = instantiate(cfg.model, model_dtype=dtype, device=model_device)
    _load_model_checkpoint(model, str(checkpoint_path))
    model = model.to(model_device).eval()
    base_kwargs = dict(infer_kwargs)
    with torch.no_grad():
        original = model.infer_action(
            **base_kwargs,
            disabled_video_layers=DISABLED_LAYERS,
            compile_action_infer=bool(cfg.EVALUATION.compile_action_infer),
        )
        replacement = model.infer_action(
            **base_kwargs,
            disabled_video_layers=DISABLED_LAYERS,
            replacement_input_image=infer_kwargs["input_image"].clone(),
            replacement_video_layers=REPLACEMENT_LAYERS,
            return_video_cache_stats=True,
            compile_action_infer=bool(cfg.EVALUATION.compile_action_infer),
        )
    if not isinstance(original, Mapping) or not isinstance(replacement, Mapping):
        raise TypeError("Fast-WAM identity calls did not return mappings.")
    action_a = original.get("action")
    action_b = replacement.get("action")
    if not torch.is_tensor(action_a) or not torch.is_tensor(action_b):
        raise TypeError("Fast-WAM identity calls did not return action tensors.")
    if tuple(action_a.shape) != tuple(action_b.shape) or tuple(action_a.shape[-2:]) != (32, 7):
        raise ValueError(
            f"Identity action shapes must match with trailing [32,7], got "
            f"{tuple(action_a.shape)} and {tuple(action_b.shape)}."
        )
    a = action_a.detach().float().cpu()
    b = action_b.detach().float().cpu()
    difference = (a - b).abs()
    action_max = float(difference.max().item())
    action_mean = float(difference.mean().item())
    allclose = bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL))
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    cache = _validate_audit(replacement.get("video_cache_stats"))
    passed = finite and allclose and action_max <= ATOL
    report = {
        "artifact_type": "asre_round3b_self_replacement_identity",
        "schema_version": 1,
        "protocol": ROUND3B_PROTOCOL,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "created_at": now_iso(),
        "git_commit_hash": git_commit(project_root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(valid["checkpoint_sha256"]),
        "dataset_stats_path": str(dataset_stats_path),
        "dataset_stats_sha256": str(valid["dataset_stats_sha256"]),
        "state_bank_manifest_path": str(source_manifest_path),
        "state_bank_manifest_sha256": str(valid["source_manifest_sha256"]),
        "valid_state_bank_manifest_path": str(valid_manifest_path),
        "valid_state_bank_manifest_sha256": sha256_file(valid_manifest_path),
        "sample_id": str(selected_record["sample_id"]),
        "sample_path": str(sample_path.resolve()),
        "disabled_video_layers": list(DISABLED_LAYERS),
        "replacement_video_layers": list(REPLACEMENT_LAYERS),
        "action_shape": list(a.shape),
        "action_finite": finite,
        "action_max_absolute_difference": action_max,
        "action_mean_absolute_difference": action_mean,
        "action_allclose": allclose,
        "max_raw_action_abs_diff": action_max,
        "mean_raw_action_abs_diff": action_mean,
        "torch_allclose": allclose,
        "atol": ATOL,
        "rtol": RTOL,
        "cache_audit": cache,
    }
    if not passed:
        raise RuntimeError(f"Self-replacement identity gate failed: {json.dumps(report)}")
    return report


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    expected_identity = {
        "artifact_type": "asre_round3b_self_replacement_identity",
        "schema_version": 1,
        "protocol": ROUND3B_PROTOCOL,
        "status": "passed",
        "passed": True,
        "git_commit_hash": git_commit(project_root),
        "checkpoint_path": str(args.checkpoint.expanduser().resolve()),
        "valid_state_bank_manifest_path": str(args.valid_manifest.expanduser().resolve()),
        "disabled_video_layers": list(DISABLED_LAYERS),
        "replacement_video_layers": list(REPLACEMENT_LAYERS),
        "atol": ATOL,
        "rtol": RTOL,
        "torch_allclose": True,
    }
    if output.exists():
        existing = _read_json(output, label="existing self-replacement report")
        mismatch = {
            key: {"existing": existing.get(key), "requested": value}
            for key, value in expected_identity.items()
            if existing.get(key) != value
        }
        if mismatch:
            raise FileExistsError(
                f"Refusing incompatible identity report: {json.dumps(mismatch)}"
            )
        print(json.dumps(existing, indent=2, sort_keys=True))
        print(f"Self-replacement gate already passed: {output}")
        return
    report = run_identity_test(
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        valid_manifest_path=args.valid_manifest,
        state_bank_dir=args.state_bank_dir,
        task_config=args.task_config,
    )
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Self-replacement identity gate passed: {output}")


if __name__ == "__main__":
    main()
