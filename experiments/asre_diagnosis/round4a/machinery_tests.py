"""Real-checkpoint endpoint, identity, token-mask, and head-mask gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from hydra.utils import instantiate


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4A_PROTOCOL,
    atomic_write_json,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.g0.machinery_tests import (  # noqa: E402
    ATOL,
    RTOL,
    _comparison,
    _compose,
)
from experiments.asre_diagnosis.round2.validate_state_bank import (  # noqa: E402
    _validate_sample_partition,
    validate_existing_artifacts,
)
from experiments.asre_diagnosis.round3b.donor import tensor_sha256  # noqa: E402
from experiments.asre_diagnosis.round3b.offline_donor import (  # noqa: E402
    load_offline_donor_manifest,
)
from experiments.asre_diagnosis.round4a.masks import (  # noqa: E402
    LATE_LAYERS,
    load_mask_spec,
    runtime_layout_from_cache_stats,
    write_or_verify_frozen_manifest,
    write_or_verify_axis_manifests,
)


EARLY_DISABLED = tuple(range(15))
TASK_CONFIG = "libero_uncond_2cam224_1e-4"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return payload


def _load_pair(
    *,
    state_bank_dir: Path,
    valid_manifest: Mapping[str, Any],
    offline_donor_mapping_path: Path,
) -> tuple[dict[str, Any], torch.Tensor, str, str, list[Mapping[str, Any]]]:
    records = load_manifest(state_bank_dir / "manifest.jsonl")
    valid_ids = _validate_sample_partition(valid_manifest, records)
    by_id = {str(record["sample_id"]): record for record in records}
    mapping = load_offline_donor_manifest(offline_donor_mapping_path)
    donor_by_recipient = {
        str(entry["recipient_sample_id"]): entry for entry in mapping["entries"]
    }
    recipient_id = valid_ids[0]
    donor_id = str(donor_by_recipient[recipient_id]["donor_sample_id"])

    def load(identifier: str) -> dict[str, Any]:
        record = by_id[identifier]
        path = Path(str(record["sample_path"]))
        if not path.is_absolute():
            path = state_bank_dir / path
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("sample_id")) != identifier:
            raise ValueError(f"State-bank sample identity mismatch: {identifier}.")
        return payload

    recipient = load(recipient_id)
    donor = load(donor_id)
    infer_kwargs = recipient.get("infer_action_kwargs")
    donor_kwargs = donor.get("infer_action_kwargs")
    if not isinstance(infer_kwargs, dict) or not isinstance(donor_kwargs, dict):
        raise TypeError("Machinery samples lack infer_action_kwargs.")
    recipient_image = infer_kwargs.get("input_image")
    donor_image = donor_kwargs.get("input_image")
    if not torch.is_tensor(recipient_image) or not torch.is_tensor(donor_image):
        raise TypeError("Machinery samples lack tensor input images.")
    if tuple(recipient_image.shape) != tuple(donor_image.shape):
        raise ValueError("Machinery donor image does not match recipient shape.")
    if tensor_sha256(recipient_image) == tensor_sha256(donor_image):
        raise ValueError("Machinery donor image is identical to its recipient.")
    return dict(infer_kwargs), donor_image, recipient_id, donor_id, records


def _reference_action(path: Path, sample_id: str) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as payload:
        sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
        if sample_ids.count(sample_id) != 1:
            raise ValueError(f"Reference actions do not contain {sample_id} exactly once.")
        action = np.asarray(payload["raw_actions"][sample_ids.index(sample_id)], dtype=np.float32)
    if action.shape != (32, 7) or not np.all(np.isfinite(action)):
        raise ValueError(f"Malformed reference action in {path}: {action.shape}.")
    return torch.from_numpy(action)


def _reference_cache_record(path: Path, sample_id: str) -> dict[str, Any]:
    matches = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if str(record.get("sample_id")) == sample_id:
                    matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"Reference cache stats do not identify {sample_id} exactly once.")
    return matches[0]


def _cache_endpoint_comparison(
    observed: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    if observed.get("replacement_video_layers") != list(LATE_LAYERS):
        raise ValueError("Observed wrong endpoint has the wrong replacement layers.")
    if reference.get("replacement_video_layers") != list(LATE_LAYERS):
        raise ValueError("Frozen Round-3B cache reference has the wrong layers.")
    compared_values = []
    for layer in LATE_LAYERS:
        observed_layer = observed["layers"][layer]
        reference_layer = reference["layers"][layer]
        if observed_layer["selected_source"] != "replacement":
            raise ValueError(f"Wrong endpoint did not select donor cache at layer {layer}.")
        for source in ("current", "replacement"):
            for kind in ("k", "v"):
                left = observed_layer[source][kind]
                right = reference_layer[source][kind]
                for key in ("shape", "dtype", "numel", "finite"):
                    if left[key] != right[key]:
                        raise ValueError(
                            f"Endpoint cache {key} drift at layer {layer} {source}/{kind}."
                        )
                for key in ("mean", "std", "rms"):
                    compared_values.append(abs(float(left[key]) - float(right[key])))
    max_abs = max(compared_values, default=math.inf)
    return {
        "late_layer_count": len(LATE_LAYERS),
        "shape_dtype_numel_finite_match": True,
        "max_summary_stat_absolute_difference": max_abs,
        "within_tolerance": max_abs <= ATOL,
        "atol": ATOL,
    }


def _validate_hybrid_audit(
    audit: Any,
    *,
    axis: str,
    expected_spec: Any,
) -> dict[str, Any]:
    if not isinstance(audit, Mapping):
        raise TypeError("Hybrid inference did not return video_cache_stats.")
    hybrid = audit.get("hybrid_video_cache")
    if not isinstance(hybrid, Mapping) or hybrid.get("mode") != axis:
        raise ValueError(f"Hybrid cache audit is absent or has the wrong axis: {axis}.")
    if hybrid.get("same_mask_for_k_and_v") is not True:
        raise ValueError("Hybrid cache audit did not certify one shared K/V mask.")
    if hybrid.get("shape_preserved") is not True:
        raise ValueError("Hybrid cache audit did not preserve geometry.")
    if hybrid.get("replacement_video_layers") != list(LATE_LAYERS):
        raise ValueError("Hybrid cache audit has wrong late-layer coverage.")
    layers = audit.get("layers")
    if not isinstance(layers, list) or len(layers) != 30:
        raise ValueError("Hybrid cache audit must contain 30 layer records.")
    for layer in LATE_LAYERS:
        if layers[layer].get("selected_source") != f"hybrid_{axis}":
            raise ValueError(f"Layer {layer} did not use hybrid_{axis} cache.")
        for source in ("current", "replacement"):
            for kind in ("k", "v"):
                if layers[layer][source][kind].get("finite") is not True:
                    raise ValueError(f"Nonfinite cache at layer {layer} {source}/{kind}.")
    if axis == "token":
        expected = list(expected_spec.retained_current_token_indices)
        if hybrid.get("retained_current_token_indices") != expected:
            raise ValueError("Token audit differs from the frozen manifest.")
        selected_count = int(hybrid["retained_current_token_count"])
        population = int(hybrid["action_visible_token_count"])
        if selected_count != (population + 1) // 2:
            raise ValueError("Token audit does not implement ceil(n/2).")
    else:
        expected = {
            str(layer): list(heads)
            for layer, heads in expected_spec.retained_current_heads_by_layer.items()
        }
        if hybrid.get("retained_current_heads_by_layer") != expected:
            raise ValueError("Head audit differs from the frozen per-layer manifest.")
        for entry in hybrid["layers"]:
            if int(entry["retained_current_head_count"]) != (
                int(hybrid["num_heads"]) + 1
            ) // 2:
                raise ValueError("Head audit does not implement ceil(n/2).")
    return {
        "axis": axis,
        "same_mask_for_k_and_v": True,
        "shape_preserved": True,
        "late_layers": list(LATE_LAYERS),
        "action_visible_token_count": int(hybrid["action_visible_token_count"]),
        "num_heads": hybrid.get("num_heads"),
        "head_dim": hybrid.get("head_dim"),
        "manifest_exact_match": True,
    }


def run_machinery_tests(
    *,
    checkpoint: Path,
    dataset_stats: Path,
    valid_manifest_path: Path,
    state_bank_dir: Path,
    offline_donor_mapping_path: Path,
    round3b_current_actions: Path,
    round3b_wrong_actions: Path,
    round3b_wrong_cache_stats: Path,
    preflight_report_path: Path,
    mask_manifest_path: Path,
    task_config: str = TASK_CONFIG,
) -> dict[str, Any]:
    from experiments.libero.eval_libero_single import (
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _resolve_eval_device,
    )
    from fastwam.utils.pytorch_utils import set_global_seed

    file_paths = [
        checkpoint,
        dataset_stats,
        valid_manifest_path,
        offline_donor_mapping_path,
        round3b_current_actions,
        round3b_wrong_actions,
        round3b_wrong_cache_stats,
        preflight_report_path,
    ]
    resolved = [path.expanduser().resolve() for path in file_paths]
    if any(not path.is_file() for path in resolved):
        raise FileNotFoundError(f"Round-4A machinery input is unavailable: {resolved}.")
    (
        checkpoint,
        dataset_stats,
        valid_manifest_path,
        offline_donor_mapping_path,
        round3b_current_actions,
        round3b_wrong_actions,
        round3b_wrong_cache_stats,
        preflight_report_path,
    ) = resolved
    state_bank_dir = state_bank_dir.expanduser().resolve()
    preflight = _read_json(preflight_report_path, "Round-4A preflight")
    if preflight.get("status") != "compatible" or preflight.get("protocol") != ROUND4A_PROTOCOL:
        raise ValueError("Round-4A preflight did not pass.")
    valid = _read_json(valid_manifest_path, "valid state manifest")
    records = load_manifest(state_bank_dir / "manifest.jsonl")
    valid_ids = _validate_sample_partition(valid, records)
    validate_existing_artifacts(
        valid_manifest_path=valid_manifest_path,
        prompt_cache_path=Path(str(valid["prompt_context_cache_path"])).resolve(),
        source_manifest_path=state_bank_dir / "manifest.jsonl",
        source_records=records,
        checkpoint_path=checkpoint,
        dataset_stats_path=dataset_stats,
        verify_checkpoint_hash=False,
    )
    if len(valid_ids) != 499:
        raise ValueError(f"Machinery gate requires 499 valid states, got {len(valid_ids)}.")
    infer_kwargs, donor_image, recipient_id, donor_id, _ = _load_pair(
        state_bank_dir=state_bank_dir,
        valid_manifest=valid,
        offline_donor_mapping_path=offline_donor_mapping_path,
    )

    cfg = _compose(task_config)
    cfg.model.load_text_encoder = False
    cfg.EVALUATION.text_encoder_device = None
    device = str(_resolve_eval_device(cfg))
    if torch.device(device).type != "cuda":
        raise RuntimeError("Round-4A machinery tests require CUDA.")
    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    set_global_seed(42, get_worker_init_fn=False)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    _load_model_checkpoint(model, str(checkpoint))
    model = model.to(device).eval()
    compile_action = bool(cfg.EVALUATION.compile_action_infer)
    recipient_image = infer_kwargs["input_image"]

    with torch.no_grad():
        current = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            return_video_cache_stats=True,
            compile_action_infer=compile_action,
        )
        wrong = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor_image,
            replacement_video_layers=LATE_LAYERS,
            return_video_cache_stats=True,
            compile_action_infer=compile_action,
        )

    layout = runtime_layout_from_cache_stats(wrong["video_cache_stats"])
    mask_digest = write_or_verify_frozen_manifest(mask_manifest_path, layout)
    axis_manifests = write_or_verify_axis_manifests(mask_manifest_path)
    token_spec = load_mask_spec(
        path=mask_manifest_path,
        expected_sha256=mask_digest,
        condition_name="token50_seed1",
    )
    head_spec = load_mask_spec(
        path=mask_manifest_path,
        expected_sha256=mask_digest,
        condition_name="head50_seed1",
    )
    visible_tokens = tuple(
        int(index) for index in layout["action_visible_token_indices"]
    )
    with torch.no_grad():
        all_current_endpoint = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor_image,
            replacement_video_layers=LATE_LAYERS,
            retained_current_video_token_indices=visible_tokens,
            expected_video_cache_layout=layout,
            return_video_cache_stats=True,
            compile_action_infer=compile_action,
        )
        all_wrong_endpoint = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor_image,
            replacement_video_layers=LATE_LAYERS,
            retained_current_video_token_indices=(),
            expected_video_cache_layout=layout,
            return_video_cache_stats=True,
            compile_action_infer=compile_action,
        )
        token_self = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=recipient_image.clone(),
            replacement_video_layers=LATE_LAYERS,
            return_video_cache_stats=True,
            compile_action_infer=compile_action,
            **token_spec.inference_kwargs(),
        )
        head_self = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=recipient_image.clone(),
            replacement_video_layers=LATE_LAYERS,
            return_video_cache_stats=True,
            compile_action_infer=compile_action,
            **head_spec.inference_kwargs(),
        )
        token_wrong = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor_image,
            replacement_video_layers=LATE_LAYERS,
            return_video_cache_stats=True,
            compile_action_infer=compile_action,
            **token_spec.inference_kwargs(),
        )
        head_wrong = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor_image,
            replacement_video_layers=LATE_LAYERS,
            return_video_cache_stats=True,
            compile_action_infer=compile_action,
            **head_spec.inference_kwargs(),
        )

    current_reference = _reference_action(round3b_current_actions, recipient_id)
    wrong_reference = _reference_action(round3b_wrong_actions, recipient_id)
    wrong_cache_reference = _reference_cache_record(
        round3b_wrong_cache_stats, recipient_id
    )
    current_endpoint_identity = _comparison(
        all_current_endpoint["action"], current_reference
    )
    current_endpoint_identity["hybrid_to_direct_current"] = _comparison(
        all_current_endpoint["action"], current["action"]
    )
    wrong_endpoint_action = _comparison(
        all_wrong_endpoint["action"], wrong_reference
    )
    wrong_endpoint_action["hybrid_to_direct_wrong"] = _comparison(
        all_wrong_endpoint["action"], wrong["action"]
    )
    controls = {
        "A_current_endpoint_identity": current_endpoint_identity,
        "B_wrong_endpoint_identity": {
            "action": wrong_endpoint_action,
            "cache": _cache_endpoint_comparison(
                wrong["video_cache_stats"], wrong_cache_reference
            ),
        },
        "C_arbitrary_mask_self_replacement_identity": {
            "token50_seed1": _comparison(current["action"], token_self["action"]),
            "head50_seed1": _comparison(current["action"], head_self["action"]),
            "token_cache_exact": bool(
                token_self["video_cache_stats"][
                    "all_replacement_caches_exact_equal"
                ]
            ),
            "head_cache_exact": bool(
                head_self["video_cache_stats"][
                    "all_replacement_caches_exact_equal"
                ]
            ),
        },
        "D_token_mask_integrity": _validate_hybrid_audit(
            token_wrong["video_cache_stats"], axis="token", expected_spec=token_spec
        ),
        "E_head_mask_integrity": _validate_hybrid_audit(
            head_wrong["video_cache_stats"], axis="head", expected_spec=head_spec
        ),
    }
    action_tensors = [
        current["action"],
        wrong["action"],
        all_current_endpoint["action"],
        all_wrong_endpoint["action"],
        token_self["action"],
        head_self["action"],
        token_wrong["action"],
        head_wrong["action"],
    ]
    all_actions_valid = all(
        tuple(action.shape) == (32, 7) and bool(torch.isfinite(action).all())
        for action in action_tensors
    )
    passed = bool(
        all_actions_valid
        and controls["A_current_endpoint_identity"]["torch_allclose"]
        and controls["A_current_endpoint_identity"]["hybrid_to_direct_current"][
            "torch_allclose"
        ]
        and controls["B_wrong_endpoint_identity"]["action"]["torch_allclose"]
        and controls["B_wrong_endpoint_identity"]["action"][
            "hybrid_to_direct_wrong"
        ]["torch_allclose"]
        and controls["B_wrong_endpoint_identity"]["cache"]["within_tolerance"]
        and controls["C_arbitrary_mask_self_replacement_identity"][
            "token50_seed1"
        ]["torch_allclose"]
        and controls["C_arbitrary_mask_self_replacement_identity"][
            "head50_seed1"
        ]["torch_allclose"]
        and controls["C_arbitrary_mask_self_replacement_identity"][
            "token_cache_exact"
        ]
        and controls["C_arbitrary_mask_self_replacement_identity"][
            "head_cache_exact"
        ]
    )
    report = {
        "artifact_type": "asre_round4a_machinery_report",
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
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
        "recipient_image_sha256": tensor_sha256(recipient_image),
        "donor_image_sha256": tensor_sha256(donor_image),
        "mask_manifest_path": str(mask_manifest_path.resolve()),
        "mask_manifest_sha256": mask_digest,
        "token_mask_manifest_path": axis_manifests["token"]["path"],
        "token_mask_manifest_sha256": axis_manifests["token"]["sha256"],
        "head_mask_manifest_path": axis_manifests["head"]["path"],
        "head_mask_manifest_sha256": axis_manifests["head"]["sha256"],
        "runtime_layout": layout,
        "all_action_outputs_shape_32x7_and_finite": all_actions_valid,
        "controls": controls,
    }
    if not passed:
        raise RuntimeError("Round-4A machinery gate failed: " + json.dumps(report))
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--state-bank-dir", type=Path, required=True)
    parser.add_argument("--offline-donor-mapping", type=Path, required=True)
    parser.add_argument("--round3b-current-actions", type=Path, required=True)
    parser.add_argument("--round3b-wrong-actions", type=Path, required=True)
    parser.add_argument("--round3b-wrong-cache-stats", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
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
        offline_donor_mapping_path=args.offline_donor_mapping,
        round3b_current_actions=args.round3b_current_actions,
        round3b_wrong_actions=args.round3b_wrong_actions,
        round3b_wrong_cache_stats=args.round3b_wrong_cache_stats,
        preflight_report_path=args.preflight_report,
        mask_manifest_path=args.mask_manifest.expanduser().resolve(),
        task_config=args.task_config,
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        existing = _read_json(output, "existing Round-4A machinery report")
        stable_report = {key: value for key, value in report.items() if key != "created_at"}
        stable_existing = {
            key: value for key, value in existing.items() if key != "created_at"
        }
        if stable_existing != stable_report:
            raise FileExistsError(
                f"Refusing to overwrite incompatible machinery report: {output}"
            )
    else:
        atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
