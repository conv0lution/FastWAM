"""Fixed-input numerical gates required before any G0 rollout."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (
    G0_PROTOCOL,
    atomic_write_json,
    build_g0_conditions,
    build_round2_conditions,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.g0.definitions import TASK_CONFIG
from experiments.asre_diagnosis.round2.validate_state_bank import (
    _validate_sample_partition,
    validate_existing_artifacts,
)
from experiments.asre_diagnosis.round3b.donor import tensor_sha256


ATOL = 1.0e-4
RTOL = 1.0e-4
EARLY_DISABLED = tuple(range(15))
LATE_REPLACEMENT = tuple(range(15, 30))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object.")
    return payload


def _comparison(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    left = a.detach().to(device="cpu", dtype=torch.float32)
    right = b.detach().to(device="cpu", dtype=torch.float32)
    if tuple(left.shape) != tuple(right.shape) or tuple(left.shape[-2:]) != (32, 7):
        raise ValueError(f"Expected aligned action tensors ending [32,7], got {left.shape}/{right.shape}.")
    difference = (left - right).abs()
    return {
        "action_shape": list(left.shape),
        "action_finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
        "max_raw_action_abs_diff": float(difference.max().item()),
        "mean_raw_action_abs_diff": float(difference.mean().item()),
        "torch_allclose": bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL)),
        "atol": ATOL,
        "rtol": RTOL,
    }


def _validate_cache_audit(
    audit: Any, *, expect_identical: bool
) -> dict[str, Any]:
    if not isinstance(audit, Mapping):
        raise TypeError("Replacement path did not return video_cache_stats.")
    expected_top = {
        "replacement_video_layers": list(LATE_REPLACEMENT),
        "disabled_video_layers": list(EARLY_DISABLED),
        "has_replacement": True,
        "summarized_video_layers": list(LATE_REPLACEMENT),
    }
    mismatch = {
        key: {"observed": audit.get(key), "expected": value}
        for key, value in expected_top.items()
        if audit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Replacement cache audit mismatch: {mismatch}.")
    for current_key, donor_key in (
        ("current_video_seq_len", "replacement_video_seq_len"),
        ("current_video_tokens_per_frame", "replacement_video_tokens_per_frame"),
    ):
        if audit.get(current_key) != audit.get(donor_key):
            raise ValueError(f"Matched-shape cache invariant failed: {current_key}/{donor_key}.")
    layers = audit.get("layers")
    if not isinstance(layers, list) or len(layers) != 30:
        raise ValueError("Replacement audit must contain exactly 30 layer records.")
    max_difference = 0.0
    all_exact = True
    for layer_id in LATE_REPLACEMENT:
        layer = layers[layer_id]
        if int(layer.get("layer", -1)) != layer_id or layer.get("selected_source") != "replacement":
            raise ValueError(f"Layer {layer_id} did not select replacement K/V.")
        if layer.get("summarized") is not True:
            raise ValueError(f"Layer {layer_id} lacks cache summary.")
        for tensor_name in ("k", "v"):
            current = layer["current"][tensor_name]
            replacement = layer["replacement"][tensor_name]
            difference = layer["difference"][tensor_name]
            for key in ("shape", "dtype", "device", "numel"):
                if current.get(key) != replacement.get(key):
                    raise ValueError(f"Layer {layer_id} {tensor_name} {key} mismatch.")
            if current.get("finite") is not True or replacement.get("finite") is not True:
                raise ValueError(f"Layer {layer_id} {tensor_name} contains NaN/Inf.")
            value = float(difference.get("max_abs", math.inf))
            if not math.isfinite(value):
                raise ValueError(f"Layer {layer_id} {tensor_name} difference is non-finite.")
            max_difference = max(max_difference, value)
            all_exact &= value == 0.0
    if expect_identical and max_difference > ATOL:
        raise ValueError(f"Self-replacement cache difference {max_difference} exceeds {ATOL}.")
    if not expect_identical and max_difference <= 0.0:
        raise ValueError("Wrong-scene donor unexpectedly produced identical video K/V.")
    return {
        "matched_shape": True,
        "dtype_device_match": True,
        "all_values_finite": True,
        "current_video_seq_len": int(audit["current_video_seq_len"]),
        "replacement_video_seq_len": int(audit["replacement_video_seq_len"]),
        "current_video_tokens_per_frame": int(audit["current_video_tokens_per_frame"]),
        "replacement_video_tokens_per_frame": int(
            audit["replacement_video_tokens_per_frame"]
        ),
        "max_layer_cache_absolute_difference": max_difference,
        "all_replacement_caches_exact_equal": all_exact,
    }


def _compose(task_config: str):
    with initialize_config_dir(
        config_dir=str((project_root / "configs").resolve()), version_base="1.3"
    ):
        return compose(config_name="sim_libero.yaml", overrides=[f"task={task_config}"])


def _load_state_inputs(
    state_bank_dir: Path, valid_manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    records = load_manifest(state_bank_dir / "manifest.jsonl")
    valid_ids = _validate_sample_partition(valid_manifest, records)
    by_id = {str(record["sample_id"]): record for record in records}
    recipient_id = valid_ids[0]
    recipient_record = by_id[recipient_id]
    recipient_path = Path(str(recipient_record["sample_path"]))
    if not recipient_path.is_absolute():
        recipient_path = state_bank_dir / recipient_path
    recipient = torch.load(recipient_path, map_location="cpu", weights_only=False)
    recipient_kwargs = recipient.get("infer_action_kwargs")
    if not isinstance(recipient_kwargs, Mapping) or not torch.is_tensor(
        recipient_kwargs.get("input_image")
    ):
        raise TypeError("Recipient state-bank sample has malformed infer_action_kwargs.")
    recipient_image = recipient_kwargs["input_image"]
    recipient_hash = tensor_sha256(recipient_image)
    for candidate_id in valid_ids[1:]:
        record = by_id[candidate_id]
        if (
            record.get("task_suite") != recipient_record.get("task_suite")
            or int(record.get("task_id", -1)) != int(recipient_record.get("task_id", -2))
        ):
            continue
        path = Path(str(record["sample_path"]))
        if not path.is_absolute():
            path = state_bank_dir / path
        sample = torch.load(path, map_location="cpu", weights_only=False)
        kwargs = sample.get("infer_action_kwargs")
        if not isinstance(kwargs, Mapping) or not torch.is_tensor(kwargs.get("input_image")):
            continue
        donor_image = kwargs["input_image"]
        if tuple(donor_image.shape) == tuple(recipient_image.shape) and tensor_sha256(
            donor_image
        ) != recipient_hash:
            return dict(recipient_kwargs), dict(kwargs), recipient_id, candidate_id
    raise ValueError("No distinct same-task matched-shape machinery donor was found.")


def run_machinery_tests(
    *,
    checkpoint: Path,
    dataset_stats: Path,
    valid_manifest_path: Path,
    state_bank_dir: Path,
    preflight_report_path: Path,
    task_config: str = TASK_CONFIG,
) -> dict[str, Any]:
    from experiments.libero.eval_libero_single import (
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _resolve_eval_device,
    )
    from fastwam.utils.pytorch_utils import set_global_seed

    paths = [checkpoint, dataset_stats, valid_manifest_path, preflight_report_path]
    resolved = [path.expanduser().resolve() for path in paths]
    if any(not path.is_file() for path in resolved):
        raise FileNotFoundError(f"Machinery input is unavailable: {resolved}.")
    checkpoint, dataset_stats, valid_manifest_path, preflight_report_path = resolved
    state_bank_dir = state_bank_dir.expanduser().resolve()
    preflight = _read_json(preflight_report_path, "G0 preflight")
    if preflight.get("status") != "compatible":
        raise ValueError("G0 preflight did not pass.")
    valid = _read_json(valid_manifest_path, "valid state manifest")
    source_manifest = state_bank_dir / "manifest.jsonl"
    records = load_manifest(source_manifest)
    valid_ids = _validate_sample_partition(valid, records)
    validate_existing_artifacts(
        valid_manifest_path=valid_manifest_path,
        prompt_cache_path=Path(str(valid["prompt_context_cache_path"])).resolve(),
        source_manifest_path=source_manifest,
        source_records=records,
        checkpoint_path=checkpoint,
        dataset_stats_path=dataset_stats,
        verify_checkpoint_hash=False,
    )
    if len(valid_ids) != 499:
        raise ValueError(f"Machinery gate expects 499 valid states, got {len(valid_ids)}.")
    recipient_kwargs, donor_kwargs, recipient_id, donor_id = _load_state_inputs(
        state_bank_dir, valid
    )

    cfg = _compose(task_config)
    cfg.model.load_text_encoder = False
    cfg.EVALUATION.text_encoder_device = None
    device = str(_resolve_eval_device(cfg))
    if torch.device(device).type != "cuda":
        raise RuntimeError("G0 machinery tests require an available CUDA device.")
    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    set_global_seed(42, get_worker_init_fn=False)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    _load_model_checkpoint(model, str(checkpoint))
    model = model.to(device).eval()
    compile_action = bool(cfg.EVALUATION.compile_action_infer)
    recipient_image = recipient_kwargs["input_image"]
    donor_image = donor_kwargs["input_image"]
    with torch.no_grad():
        normal = model.infer_action(**recipient_kwargs, compile_action_infer=compile_action)
        full_g0 = model.infer_action(
            **recipient_kwargs,
            disabled_video_layers=(),
            compile_action_infer=compile_action,
        )
        late_round2 = model.infer_action(
            **recipient_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            compile_action_infer=compile_action,
        )
        late_g0 = model.infer_action(
            **recipient_kwargs,
            disabled_video_layers=tuple(
                build_g0_conditions(30)[1].disabled_video_layers
            ),
            compile_action_infer=compile_action,
        )
        wrong = model.infer_action(
            **recipient_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor_image,
            replacement_video_layers=LATE_REPLACEMENT,
            return_video_cache_stats=True,
            compile_action_infer=compile_action,
        )
        self_replaced = model.infer_action(
            **recipient_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=recipient_image.clone(),
            replacement_video_layers=LATE_REPLACEMENT,
            return_video_cache_stats=True,
            compile_action_infer=compile_action,
        )

    if build_round2_conditions(30)[1].disabled_video_layers != EARLY_DISABLED:
        raise ValueError("Frozen keep_15_29 registration drifted.")
    controls = {
        "A_full_current_matches_normal": _comparison(normal["action"], full_g0["action"]),
        "B_late_current_matches_round2_keep_15_29": _comparison(
            late_round2["action"], late_g0["action"]
        ),
        "C_wrong_scene_replacement": {
            "action_shape": list(wrong["action"].shape),
            "action_finite": bool(torch.isfinite(wrong["action"]).all()),
            "recipient_image_sha256": tensor_sha256(recipient_image),
            "donor_image_sha256": tensor_sha256(donor_image),
            "images_differ": tensor_sha256(recipient_image) != tensor_sha256(donor_image),
            "cache_audit": _validate_cache_audit(
                wrong.get("video_cache_stats"), expect_identical=False
            ),
        },
        "D_self_replacement": {
            **_comparison(late_round2["action"], self_replaced["action"]),
            "cache_audit": _validate_cache_audit(
                self_replaced.get("video_cache_stats"), expect_identical=True
            ),
        },
    }
    passed = (
        controls["A_full_current_matches_normal"]["torch_allclose"]
        and controls["B_late_current_matches_round2_keep_15_29"]["torch_allclose"]
        and controls["C_wrong_scene_replacement"]["action_finite"]
        and controls["D_self_replacement"]["torch_allclose"]
    )
    report = {
        "artifact_type": "asre_g0_machinery_report",
        "schema_version": 1,
        "protocol": G0_PROTOCOL,
        "status": "passed" if passed else "failed",
        "passed": bool(passed),
        "created_at": now_iso(),
        "git_commit_hash": git_commit(project_root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": str(valid["checkpoint_sha256"]),
        "dataset_stats_path": str(dataset_stats),
        "dataset_stats_sha256": str(valid["dataset_stats_sha256"]),
        "valid_state_bank_manifest_path": str(valid_manifest_path),
        "valid_state_bank_manifest_sha256": sha256_file(valid_manifest_path),
        "preflight_report_path": str(preflight_report_path),
        "preflight_report_sha256": sha256_file(preflight_report_path),
        "recipient_sample_id": recipient_id,
        "donor_sample_id": donor_id,
        "controls": controls,
    }
    if not passed:
        raise RuntimeError(f"G0 machinery gate failed: {json.dumps(report)}")
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--state-bank-dir", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-config", default=TASK_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = run_machinery_tests(
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        valid_manifest_path=args.valid_manifest,
        state_bank_dir=args.state_bank_dir,
        preflight_report_path=args.preflight_report,
        task_config=args.task_config,
    )
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
