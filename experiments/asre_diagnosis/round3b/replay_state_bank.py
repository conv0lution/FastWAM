"""Replay one matched-cache condition over the immutable 499-state bank.

The original Round-3B path remains unchanged.  Round-4A reuses the same validated
state ordering and donor mapping while supplying a frozen token/head mask.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND3B_PROTOCOL,
    ROUND4A_PROTOCOL,
    atomic_write_json,
    build_round3b_conditions,
    build_round4a_conditions,
    build_run_metadata,
    get_num_model_layers,
    load_manifest,
    now_iso,
    resolve_condition,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.round2.metrics import (  # noqa: E402
    CONTINUOUS_ACTION_DIMENSIONS,
    compute_round2_metrics,
    extract_action_global_std,
)
from experiments.asre_diagnosis.round2.validate_state_bank import (  # noqa: E402
    _read_json,
    _resolve_path,
    _validate_sample_partition,
    validate_existing_artifacts,
)
from experiments.asre_diagnosis.round3b.offline_donor import (  # noqa: E402
    load_offline_donor_manifest,
    tensor_sha256,
)
from experiments.asre_diagnosis.round4a.masks import (  # noqa: E402
    Round4AMaskSpec,
    load_mask_manifest,
    load_mask_spec,
)
from experiments.libero.eval_libero_single import (  # noqa: E402
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _postprocess_action,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import (  # noqa: E402
    FastWAMProcessor,
)
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


SCALAR_METRICS = (
    "executed_prefix_norm_rms",
    "norm_rms_h0",
    "norm_rms_h0_h1",
    "full_chunk_norm_rms_0_31",
    "round1_raw_output_full_chunk_rms",
    "executed_prefix_cosine_similarity",
    "executed_prefix_gripper_flip_rate",
    "full_horizon_gripper_flip_rate",
    "translation_norm_rms",
    "rotation_norm_rms",
)
DIMENSION_METRIC = "executed_prefix_norm_rms_by_dimension"


def _require_clean_worktree() -> None:
    try:
        output = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                ".",
                ":(exclude)asre_results/round3b/**",
                ":(exclude)asre_results/round4a/**",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot inspect the Git worktree before replay: {exc}") from exc
    if output.strip():
        raise RuntimeError(
            "ASRE Round 3B offline replay requires a clean Git worktree. Commit the "
            "reviewed source implementation first; generated Round-3B artifacts are "
            "exempt and no changes are discarded automatically.\n"
            f"git status --porcelain:\n{output.rstrip()}"
        )


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    if not records:
        return math.nan
    return float(np.mean([float(record[key]) for record in records]))


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _atomic_write_actions(
    path: Path,
    *,
    sample_ids: Sequence[str],
    raw_actions: Sequence[np.ndarray],
    executed_actions: Sequence[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(raw_actions, dtype=np.float32)
    executed = np.asarray(executed_actions, dtype=np.float32)
    if raw.shape != executed.shape or raw.ndim != 3 or raw.shape[-1] != 7:
        raise ValueError(
            f"Round-3B saved actions must align as [N,H,7], got {raw.shape}/{executed.shape}."
        )
    if raw.shape[0] != len(sample_ids):
        raise ValueError("Round-3B action count does not match sample IDs.")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(executed)):
        raise ValueError("Refusing to save nonfinite Round-3B actions.")
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(
                handle,
                sample_ids=np.asarray(sample_ids, dtype=str),
                raw_actions=raw,
                executed_actions=executed,
            )
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _load_actions(path: Path) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
        raw = np.asarray(payload["raw_actions"], dtype=np.float32)
        executed = np.asarray(payload["executed_actions"], dtype=np.float32)
    if raw.shape != executed.shape or raw.ndim != 3 or raw.shape[-1] != 7:
        raise ValueError(f"Malformed replay action checkpoint: {path}.")
    if raw.shape[0] != len(sample_ids):
        raise ValueError(f"Replay action checkpoint identity mismatch: {path}.")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(executed)):
        raise ValueError(f"Replay action checkpoint contains NaN/Inf: {path}.")
    return sample_ids, list(raw), list(executed)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(f"JSONL record must be an object: {path}")
                records.append(payload)
    return records


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _strict_valid_records(
    *,
    valid_manifest: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    valid_ids = _validate_sample_partition(valid_manifest, source_records)
    by_id = {str(record["sample_id"]): record for record in source_records}
    return [by_id[identifier] for identifier in valid_ids]


def _load_donor_lookup(
    *,
    mapping_path: Path,
    valid_manifest_sha256: str,
    source_manifest_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    payload = load_offline_donor_manifest(mapping_path)
    if payload.get("valid_manifest_sha256") != valid_manifest_sha256:
        raise ValueError("Offline donor mapping valid-manifest SHA256 mismatch.")
    if payload.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("Offline donor mapping source-manifest SHA256 mismatch.")
    entries = payload["entries"]
    lookup = {str(entry["recipient_sample_id"]): entry for entry in entries}
    if len(lookup) != 499:
        raise ValueError("Offline donor mapping must have 499 unique recipients.")
    return lookup, payload


def _run_model(
    *,
    model: torch.nn.Module,
    cfg: DictConfig,
    infer_kwargs: Mapping[str, Any],
    disabled_video_layers: Sequence[int],
    replacement_video_layers: Sequence[int],
    replacement_input_image: torch.Tensor | None,
    hybrid_mask_spec: Round4AMaskSpec | None,
    return_video_cache_stats: bool,
) -> dict[str, Any]:
    call_kwargs = dict(infer_kwargs)
    call_kwargs.update(
        {
            "disabled_video_layers": tuple(int(value) for value in disabled_video_layers),
            "replacement_video_layers": tuple(
                int(value) for value in replacement_video_layers
            ),
            "replacement_input_image": replacement_input_image,
            "return_video_cache_stats": bool(return_video_cache_stats),
        }
    )
    if hybrid_mask_spec is not None:
        call_kwargs.update(hybrid_mask_spec.inference_kwargs())
    with torch.no_grad():
        prediction = model.infer_action(
            **call_kwargs,
            compile_action_infer=bool(cfg.EVALUATION.get("compile_action_infer", False)),
        )
    if not isinstance(prediction, Mapping) or not isinstance(
        prediction.get("action"), torch.Tensor
    ):
        raise TypeError("Fast-WAM infer_action did not return an action tensor mapping.")
    return dict(prediction)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="sim_libero.yaml")
def replay_state_bank(cfg: DictConfig) -> None:
    _require_clean_worktree()
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    diagnosis_cfg = cfg.ASRE_DIAGNOSIS
    if not bool(diagnosis_cfg.get("enabled", False)):
        raise ValueError("Round-3B offline replay requires ASRE_DIAGNOSIS.enabled=true.")
    protocol = str(diagnosis_cfg.get("protocol", ""))
    if protocol not in {ROUND3B_PROTOCOL, ROUND4A_PROTOCOL}:
        raise ValueError(
            "Matched-cache offline replay requires protocol "
            f"{ROUND3B_PROTOCOL!r} or {ROUND4A_PROTOCOL!r}."
        )
    round_label = "Round-4A" if protocol == ROUND4A_PROTOCOL else "Round-3B"
    if str(diagnosis_cfg.get("mode", "")) != "replace_video_kv":
        raise ValueError(f"{round_label} offline replay requires mode=replace_video_kv.")
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    state_bank_dir = _resolve_path(
        diagnosis_cfg.get("state_bank_dir"), label="ASRE_DIAGNOSIS.state_bank_dir"
    )
    output_root = _resolve_path(
        diagnosis_cfg.get("offline_output_dir"),
        label="ASRE_DIAGNOSIS.offline_output_dir",
    )
    valid_manifest_path = _resolve_path(
        diagnosis_cfg.get("valid_state_bank_manifest_path"),
        label="ASRE_DIAGNOSIS.valid_state_bank_manifest_path",
    )
    donor_mapping_path = _resolve_path(
        diagnosis_cfg.get("offline_donor_mapping_path"),
        label="ASRE_DIAGNOSIS.offline_donor_mapping_path",
    )
    valid_manifest = _read_json(valid_manifest_path)
    valid_manifest_sha256 = sha256_file(valid_manifest_path)
    prompt_cache_path = _resolve_path(
        valid_manifest.get("prompt_context_cache_path"),
        label="state_bank_valid_manifest.prompt_context_cache_path",
    )
    source_manifest_path = (state_bank_dir / "manifest.jsonl").resolve()
    source_records = load_manifest(source_manifest_path)
    checkpoint_path = _resolve_path(cfg.ckpt, label="ckpt")
    dataset_stats_path = _resolve_dataset_stats_path(cfg).resolve()
    trusted_parent_digest = os.environ.get(
        "ASRE_ROUND3B_TRUSTED_PREFLIGHT_MANIFEST_SHA256"
    )
    validate_existing_artifacts(
        valid_manifest_path=valid_manifest_path,
        prompt_cache_path=prompt_cache_path,
        source_manifest_path=source_manifest_path,
        source_records=source_records,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        verify_checkpoint_hash=(trusted_parent_digest != valid_manifest_sha256),
    )
    selected_records = _strict_valid_records(
        valid_manifest=valid_manifest, source_records=source_records
    )
    if len(selected_records) != 499:
        raise ValueError(
            f"{round_label} requires exactly 499 valid states, got {len(selected_records)}."
        )
    donor_lookup, donor_mapping = _load_donor_lookup(
        mapping_path=donor_mapping_path,
        valid_manifest_sha256=valid_manifest_sha256,
        source_manifest_sha256=sha256_file(source_manifest_path),
    )
    expected_ids = [str(record["sample_id"]) for record in selected_records]
    if set(donor_lookup) != set(expected_ids):
        raise ValueError("Offline donor mapping recipients differ from valid state set.")

    source_metadata = _read_json(state_bank_dir / "run_metadata.json")
    executed_prefix_length = int(diagnosis_cfg.get("executed_prefix_length", 10))
    if executed_prefix_length != 10 or executed_prefix_length != int(
        source_metadata.get("replan_steps", -1)
    ):
        raise ValueError(
            f"{round_label} executed-prefix length must equal replan interval 10."
        )
    source_compatibility = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_stats_path": str(dataset_stats_path),
        "num_model_layers": 30,
        "compile_action_infer": bool(cfg.EVALUATION.get("compile_action_infer", False)),
        "binarize_gripper": bool(cfg.EVALUATION.get("binarize_gripper", False)),
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
    }
    mismatches = {
        key: {"state_bank": source_metadata.get(key), "replay": value}
        for key, value in source_compatibility.items()
        if source_metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{round_label} replay configuration is incompatible with the state bank: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )

    cfg.model.load_text_encoder = False
    cfg.EVALUATION.text_encoder_device = None
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(checkpoint_path))
    model = model.to(model_device).eval()
    num_layers = get_num_model_layers(model)
    condition = resolve_condition(diagnosis_cfg, num_layers)
    condition_index = (
        build_round4a_conditions(num_layers).index(condition)
        if protocol == ROUND4A_PROTOCOL
        else build_round3b_conditions(num_layers).index(condition)
    )
    hybrid_mask_spec: Round4AMaskSpec | None = None
    mask_manifest_path: Path | None = None
    mask_manifest_sha256: str | None = None
    if protocol == ROUND4A_PROTOCOL:
        mask_manifest_path = _resolve_path(
            diagnosis_cfg.get("hybrid_mask_manifest_path"),
            label="ASRE_DIAGNOSIS.hybrid_mask_manifest_path",
        )
        mask_manifest_sha256 = str(
            diagnosis_cfg.get("hybrid_mask_manifest_sha256", "")
        )
        load_mask_manifest(
            path=mask_manifest_path, expected_sha256=mask_manifest_sha256
        )
        axis = diagnosis_cfg.get("hybrid_axis")
        if axis not in {None, "", "none", "null"}:
            hybrid_mask_spec = load_mask_spec(
                path=mask_manifest_path,
                expected_sha256=mask_manifest_sha256,
                condition_name=condition.name,
            )
            if hybrid_mask_spec.axis != str(axis):
                raise ValueError(f"{round_label} configured mask axis mismatch.")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_gpu_env = (
        "ASRE_ROUND4A_PHYSICAL_GPU"
        if protocol == ROUND4A_PROTOCOL
        else "ASRE_ROUND3B_PHYSICAL_GPU"
    )
    physical_gpu_value = os.environ.get(physical_gpu_env)
    if physical_gpu_value is None or not physical_gpu_value.isdigit():
        raise ValueError(
            f"{round_label} offline replay requires {physical_gpu_env} to "
            "record the assigned physical device."
        )
    physical_gpu = int(physical_gpu_value)
    if cuda_visible_devices != str(physical_gpu):
        raise ValueError(
            f"{round_label} offline GPU assignment disagrees with CUDA visibility; "
            f"condition={condition.name}, expected physical GPU {physical_gpu}, got "
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}."
        )
    if str(model_device) != "cuda:0":
        raise ValueError(f"{round_label} replay requires logical cuda:0, got {model_device}.")

    cfg.ASRE_DIAGNOSIS.condition_name = condition.name
    cfg.ASRE_DIAGNOSIS.enabled_video_retrieval_layers = list(
        condition.enabled_video_retrieval_layers(num_layers)
    )
    cfg.ASRE_DIAGNOSIS.disabled_video_layers = list(condition.disabled_video_layers)
    cfg.ASRE_DIAGNOSIS.replacement_video_layers = list(
        condition.replacement_video_layers
    )

    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    action_std = extract_action_global_std(dataset_stats)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    output_dir = output_root / condition.name
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Replay output is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    first_sample = torch.load(
        state_bank_dir / str(selected_records[0]["sample_path"]),
        map_location="cpu",
        weights_only=False,
    )
    first_infer_kwargs = first_sample["infer_action_kwargs"]
    action_horizon = int(first_infer_kwargs["action_horizon"])
    num_inference_steps = int(first_infer_kwargs["num_inference_steps"])
    task_ids = sorted({int(record["task_id"]) for record in selected_records})
    config_sha256 = sha256_json(OmegaConf.to_container(cfg, resolve=True))
    metadata = build_run_metadata(
        repo_root=PROJECT_ROOT,
        checkpoint=str(checkpoint_path),
        dataset_stats_path=str(dataset_stats_path),
        condition=condition,
        num_layers=num_layers,
        task_suite=str(first_sample["task_suite"]),
        task_ids=task_ids,
        seed=None if cfg.get("seed") is None else int(cfg.seed),
        num_trials=int(source_metadata["number_of_trials"]),
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        replan_steps=int(source_metadata["replan_steps"]),
        start_timestamp=now_iso(),
        condition_protocol=protocol,
        checkpoint_sha256=str(valid_manifest["checkpoint_sha256"]),
        dataset_stats_sha256=str(valid_manifest["dataset_stats_sha256"]),
        state_bank_manifest_path=str(source_manifest_path),
        state_bank_manifest_sha256=str(valid_manifest["source_manifest_sha256"]),
        valid_state_bank_manifest_path=str(valid_manifest_path),
        valid_state_bank_manifest_sha256=valid_manifest_sha256,
        prompt_context_cache_path=str(prompt_cache_path),
        prompt_context_cache_sha256=str(valid_manifest["prompt_context_cache_sha256"]),
        config_sha256=config_sha256,
    )
    metadata.update(
        {
            "artifact_type": (
                "asre_round4a_offline_state_bank_replay"
                if protocol == ROUND4A_PROTOCOL
                else "asre_round3b_offline_state_bank_replay"
            ),
            "status": "running",
            "output_dir": str(output_dir),
            "num_valid_samples": len(selected_records),
            "executed_prefix_length": executed_prefix_length,
            "action_global_std": action_std.astype(float).tolist(),
            "compile_action_infer": bool(
                cfg.EVALUATION.get("compile_action_infer", False)
            ),
            "binarize_gripper": bool(cfg.EVALUATION.get("binarize_gripper", False)),
            "sigma_shift": source_compatibility["sigma_shift"],
            "rand_device": source_compatibility["rand_device"],
            "text_conditioning_source": "stored_round1_state_bank_context",
            "condition_index": condition_index,
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": cuda_visible_devices,
            "model_device": str(model_device),
            "round3a_parent_tag": "ASRE-round3a-factorial",
            "round3a_parent_commit": "d36383c16d974ba1e5a750c088327a7a88baa8fb",
            "validated_round3a_run_commit": "92e842c79f5209b31c2653944cc0c1719e95eb9e",
            "offline_donor_mapping_path": str(donor_mapping_path),
            "offline_donor_mapping_sha256": sha256_file(donor_mapping_path),
            "offline_donor_mapping_rule": donor_mapping["mapping_rule"],
            "replacement_semantics": (
                "fixed donor image per saved donor state; current recipient cached text "
                "context and proprioception are retained when recomputing donor video K/V"
            ),
            "condition_config": condition.to_dict(),
            "hybrid_axis": diagnosis_cfg.get("hybrid_axis"),
            "hybrid_mask_seed": diagnosis_cfg.get("hybrid_mask_seed"),
            "hybrid_mask_manifest_path": (
                None if mask_manifest_path is None else str(mask_manifest_path)
            ),
            "hybrid_mask_manifest_sha256": mask_manifest_sha256,
        }
    )
    if protocol == ROUND4A_PROTOCOL:
        for key in (
            "preflight_report_path",
            "preflight_report_sha256",
            "machinery_report_path",
            "machinery_report_sha256",
            "round3b_parent_tag",
            "round3b_parent_commit",
            "g0_parent_tag",
            "g0_parent_commit",
            "g0_summary_path",
            "g0_summary_sha256",
            "token_mask_manifest_path",
            "token_mask_manifest_sha256",
            "head_mask_manifest_path",
            "head_mask_manifest_sha256",
        ):
            metadata[key] = diagnosis_cfg.get(key)
    metadata_path = output_dir / "run_metadata.json"
    records_path = output_dir / "per_sample.jsonl"
    actions_path = output_dir / "actions.npz"
    cache_stats_path = output_dir / "video_cache_stats.jsonl"
    records: list[dict[str, Any]] = []
    cache_stats_records: list[dict[str, Any]] = []
    raw_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    if metadata_path.exists():
        existing_metadata = _read_json(metadata_path)
        resume_expected = {
            "artifact_type": metadata["artifact_type"],
            "git_commit_hash": metadata["git_commit_hash"],
            "condition_protocol": protocol,
            "diagnosis_condition": condition.name,
            "valid_state_bank_manifest_sha256": valid_manifest_sha256,
            "offline_donor_mapping_sha256": sha256_file(donor_mapping_path),
            "hybrid_mask_manifest_sha256": mask_manifest_sha256,
            "num_valid_samples": len(selected_records),
        }
        mismatch = {
            key: {"existing": existing_metadata.get(key), "requested": value}
            for key, value in resume_expected.items()
            if existing_metadata.get(key) != value
        }
        if mismatch:
            raise FileExistsError(
                f"Refusing incompatible {round_label} replay resume: "
                f"{json.dumps(mismatch, sort_keys=True)}"
            )
        artifact_paths = (records_path, actions_path)
        if any(path.exists() for path in artifact_paths) and not all(
            path.is_file() for path in artifact_paths
        ):
            raise FileExistsError("Replay resume requires both records and actions checkpoints.")
        if all(path.is_file() for path in artifact_paths):
            records = _load_jsonl(records_path)
            saved_ids, raw_actions, executed_actions = _load_actions(actions_path)
            record_ids = [str(record.get("sample_id")) for record in records]
            expected_prefix = expected_ids[: len(records)]
            if saved_ids != record_ids or record_ids != expected_prefix:
                raise ValueError("Replay resume checkpoint is not the frozen state-order prefix.")
        if cache_stats_path.is_file():
            cache_stats_records = _load_jsonl(cache_stats_path)
        if str(existing_metadata.get("status")) == "complete":
            if len(records) != len(selected_records):
                raise ValueError("Completed replay metadata has an incomplete checkpoint.")
            print(f"{round_label} offline replay already complete: {output_dir}")
            return
        metadata["start_timestamp"] = existing_metadata.get(
            "start_timestamp", metadata["start_timestamp"]
        )
        metadata["resume_count"] = int(existing_metadata.get("resume_count", 0)) + 1
    else:
        unexpected = [path.name for path in output_dir.iterdir()]
        if unexpected:
            raise FileExistsError(
                f"Unidentified replay output without metadata: {output_dir}: {unexpected}"
            )
        metadata["resume_count"] = 0
    metadata["completed_samples"] = len(records)
    atomic_write_json(metadata_path, metadata)

    checkpoint_interval = 5
    for index, manifest_record in enumerate(selected_records, start=1):
        if index <= len(records):
            continue
        sample_id = str(manifest_record["sample_id"])
        sample = torch.load(
            state_bank_dir / str(manifest_record["sample_path"]),
            map_location="cpu",
            weights_only=False,
        )
        if str(sample.get("sample_id", "")) != sample_id:
            raise ValueError(f"Sample identity mismatch for {sample_id}.")
        infer_kwargs = sample.get("infer_action_kwargs")
        if not isinstance(infer_kwargs, dict):
            raise TypeError(f"Sample {sample_id} has no infer_action_kwargs object.")

        donor_entry = donor_lookup[sample_id]
        replacement_image: torch.Tensor | None = None
        donor_sample_id: str | None = None
        if condition.replacement_video_layers:
            donor_sample_id = str(donor_entry["donor_sample_id"])
            donor_record = next(
                record
                for record in source_records
                if str(record["sample_id"]) == donor_sample_id
            )
            donor_sample = torch.load(
                state_bank_dir / str(donor_record["sample_path"]),
                map_location="cpu",
                weights_only=False,
            )
            replacement_image = donor_sample["infer_action_kwargs"]["input_image"]
            if tensor_sha256(infer_kwargs["input_image"]) != str(
                donor_entry["recipient_image_sha256"]
            ):
                raise ValueError(f"Recipient image hash drift for {sample_id}.")
            if tensor_sha256(replacement_image) != str(donor_entry["donor_image_sha256"]):
                raise ValueError(f"Donor image hash drift for {sample_id}->{donor_sample_id}.")
            if tuple(replacement_image.shape) != tuple(infer_kwargs["input_image"].shape):
                raise ValueError(f"Donor image shape mismatch for {sample_id}.")

        prediction = _run_model(
            model=model,
            cfg=cfg,
            infer_kwargs=infer_kwargs,
            disabled_video_layers=condition.disabled_video_layers,
            replacement_video_layers=condition.replacement_video_layers,
            replacement_input_image=replacement_image,
            hybrid_mask_spec=hybrid_mask_spec,
            return_video_cache_stats=bool(condition.replacement_video_layers)
            and (protocol != ROUND4A_PROTOCOL or index == 1),
        )
        diagnosis_raw_tensor = prediction["action"]
        if not bool(torch.isfinite(diagnosis_raw_tensor).all().item()):
            raise ValueError(f"Nonfinite action for {sample_id}.")
        diagnosis_executed = _postprocess_action(diagnosis_raw_tensor, processor, cfg)
        baseline_raw_tensor = sample.get("baseline_raw_action")
        baseline_executed_tensor = sample.get("baseline_executed_action")
        if not isinstance(baseline_raw_tensor, torch.Tensor) or not isinstance(
            baseline_executed_tensor, torch.Tensor
        ):
            raise TypeError(f"Sample {sample_id} has no stored baseline action tensors.")
        diagnosis_raw = diagnosis_raw_tensor.detach().float().cpu().numpy()
        metrics = compute_round2_metrics(
            baseline_raw=baseline_raw_tensor.detach().float().cpu().numpy(),
            diagnosis_raw=diagnosis_raw,
            baseline_executed=baseline_executed_tensor.detach().float().cpu().numpy(),
            diagnosis_executed=diagnosis_executed,
            action_std=action_std,
            executed_prefix_length=executed_prefix_length,
        )
        record = {
            key: manifest_record[key]
            for key in (
                "sample_id",
                "task_suite",
                "task_id",
                "task_description",
                "episode_id",
                "replan_id",
                "environment_seed",
                "environment_step",
                "action_inference_seed",
            )
        }
        record.update(
            {
                "condition": condition.name,
                "enabled_video_retrieval_layers": list(
                    condition.enabled_video_retrieval_layers(num_layers)
                ),
                "disabled_video_layers": list(condition.disabled_video_layers),
                "replacement_video_layers": list(condition.replacement_video_layers),
                "donor_sample_id": donor_sample_id,
                "donor_episode_id": (
                    None if donor_sample_id is None else int(donor_entry["donor_episode_id"])
                ),
                "donor_pixel_mae": (
                    None if donor_sample_id is None else float(donor_entry["pixel_mae"])
                ),
                "action_dimension_names": list(CONTINUOUS_ACTION_DIMENSIONS),
                **metrics,
            }
        )
        records.append(record)
        raw_actions.append(diagnosis_raw)
        executed_actions.append(np.asarray(diagnosis_executed, dtype=np.float32))

        if condition.replacement_video_layers and (
            protocol != ROUND4A_PROTOCOL or index == 1
        ):
            cache_stats = prediction.get("video_cache_stats")
            if not isinstance(cache_stats, Mapping):
                raise TypeError(f"Missing video_cache_stats for {sample_id}.")
            if list(cache_stats.get("replacement_video_layers", [])) != list(
                condition.replacement_video_layers
            ):
                raise ValueError(f"Cache audit replacement-layer mismatch for {sample_id}.")
            layers = cache_stats.get("layers")
            if not isinstance(layers, list) or len(layers) != num_layers:
                raise ValueError(f"Cache audit must contain {num_layers} layers for {sample_id}.")
            for layer in range(15, 30):
                layer_stats = layers[layer]
                if int(layer_stats.get("layer", -1)) != layer:
                    raise ValueError(f"Cache audit layer ordering mismatch for {sample_id}.")
                expected_source = (
                    f"hybrid_{hybrid_mask_spec.axis}"
                    if hybrid_mask_spec is not None
                    else "replacement"
                )
                if str(layer_stats.get("selected_source")) != expected_source:
                    raise ValueError(f"Layer {layer} did not select replacement for {sample_id}.")
                for source in ("current", "replacement"):
                    for key in ("k", "v"):
                        stats = layer_stats[source][key]
                        if not bool(stats.get("finite", False)):
                            raise ValueError(
                                f"Nonfinite {source} {key} cache at layer {layer}, {sample_id}."
                            )
            if hybrid_mask_spec is not None:
                hybrid_audit = cache_stats.get("hybrid_video_cache")
                if not isinstance(hybrid_audit, Mapping):
                    raise TypeError(f"Missing hybrid cache audit for {sample_id}.")
                if str(hybrid_audit.get("mode")) != hybrid_mask_spec.axis:
                    raise ValueError(f"Hybrid cache audit axis mismatch for {sample_id}.")
            cache_stats_records.append(
                {
                    "sample_id": sample_id,
                    "donor_sample_id": donor_sample_id,
                    **dict(cache_stats),
                }
            )
        print(f"Replay {index}/{len(selected_records)} {condition.name}: {sample_id}")
        if index % checkpoint_interval == 0 or index == len(selected_records):
            _atomic_write_jsonl(records_path, records)
            _atomic_write_actions(
                actions_path,
                sample_ids=expected_ids[: len(records)],
                raw_actions=raw_actions,
                executed_actions=executed_actions,
            )
            if cache_stats_records:
                _atomic_write_jsonl(cache_stats_path, cache_stats_records)
            metadata["completed_samples"] = len(records)
            metadata["last_checkpoint_timestamp"] = now_iso()
            atomic_write_json(metadata_path, metadata)

    dimension_mean = np.mean(
        np.asarray([record[DIMENSION_METRIC] for record in records], dtype=np.float64),
        axis=0,
    )
    summary: dict[str, Any] = {
        "condition": condition.name,
        "enabled_video_retrieval_layers": json.dumps(
            list(condition.enabled_video_retrieval_layers(num_layers))
        ),
        "disabled_video_layers": json.dumps(list(condition.disabled_video_layers)),
        "replacement_video_layers": json.dumps(list(condition.replacement_video_layers)),
        "num_retrieval_layers": len(condition.enabled_video_retrieval_layers(num_layers)),
        "num_replacement_layers": len(condition.replacement_video_layers),
        "num_samples": len(records),
        "executed_prefix_length": executed_prefix_length,
    }
    summary.update({metric: _mean(records, metric) for metric in SCALAR_METRICS})
    summary[DIMENSION_METRIC] = json.dumps(dimension_mean.astype(float).tolist())

    _atomic_write_jsonl(records_path, records)
    _atomic_write_actions(
        actions_path,
        sample_ids=expected_ids,
        raw_actions=raw_actions,
        executed_actions=executed_actions,
    )
    if cache_stats_records:
        _atomic_write_jsonl(cache_stats_path, cache_stats_records)
    _write_summary_csv(output_dir / "summary.csv", summary)
    metadata.update(
        {
            "status": "complete",
            "end_timestamp": now_iso(),
            "num_samples": len(records),
            "actions_sha256": sha256_file(actions_path),
            "cache_stats_sha256": (
                sha256_file(cache_stats_path)
                if cache_stats_records
                else None
            ),
        }
    )
    atomic_write_json(metadata_path, metadata)
    print(json.dumps(summary, indent=2))
    print(f"{round_label} offline replay complete: {output_dir}")


if __name__ == "__main__":
    replay_state_bank()
