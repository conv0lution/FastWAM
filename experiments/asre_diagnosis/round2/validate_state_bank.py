"""Pre-register and validate the fixed sample set used by ASRE Round 2.

The first invocation replays every Round-1 state-bank sample with the unmodified
baseline path, applies the pre-registered deterministic QC rule, and writes two
immutable artifacts beside one another:

* ``state_bank_valid_manifest.json``: the ordered common sample population;
* ``prompt_context_cache.pt``: one exact DEFAULT_PROMPT context per task.

If both artifacts already exist, the command validates their paths, hashes,
sample partition, and tensor schema without loading the model or overwriting
either file.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# This is an offline-only command. A stale physical EGL index from an online
# launcher must not be interpreted relative to this process's visible devices.
os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

from experiments.asre_diagnosis.common import (  # noqa: E402
    atomic_write_json,
    get_num_model_layers,
    load_manifest,
    now_iso,
    sha256_file,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


QC_SCHEMA_VERSION = 1
PROMPT_CACHE_SCHEMA_VERSION = 1
EXPECTED_SOURCE_SAMPLES = 500
EXPECTED_ACTION_SHAPE = (32, 7)
RAW_MAX_ABSOLUTE_ERROR_TOLERANCE = 1e-6
QC_RULE: dict[str, Any] = {
    "raw_max_absolute_error_tolerance": RAW_MAX_ABSOLUTE_ERROR_TOLERANCE,
    "expected_action_shape": list(EXPECTED_ACTION_SHAPE),
    "require_raw_shape_match": True,
    "require_executed_shape_match": True,
    "require_finite_raw_actions": True,
    "require_finite_executed_actions": True,
    "require_exact_postprocessed_gripper": True,
}


def _resolve_path(value: Any, *, label: str) -> Path:
    if value is None:
        raise ValueError(f"Set {label} to an explicit path.")
    return Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _ordered_source_ids(source_records: Sequence[Mapping[str, Any]]) -> list[str]:
    identifiers = [str(record.get("sample_id", "")) for record in source_records]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every source-manifest record must have a nonempty sample_id.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("The source state-bank manifest contains duplicate sample IDs.")
    return identifiers


def _excluded_ids(excluded_samples: Any) -> list[str]:
    if not isinstance(excluded_samples, list):
        raise TypeError("excluded_samples must be a list of QC detail objects.")
    identifiers: list[str] = []
    for index, detail in enumerate(excluded_samples):
        if not isinstance(detail, dict):
            raise TypeError(f"excluded_samples[{index}] must be an object.")
        identifier = str(detail.get("sample_id", ""))
        if not identifier:
            raise ValueError(f"excluded_samples[{index}] has no sample_id.")
        reasons = detail.get("reasons")
        if not isinstance(reasons, list) or not reasons:
            raise ValueError(f"Excluded sample {identifier} must record at least one reason.")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("excluded_samples contains duplicate sample IDs.")
    return identifiers


def _validate_sample_partition(
    payload: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    source_ids = _ordered_source_ids(source_records)
    valid_ids_value = payload.get("valid_sample_ids")
    if not isinstance(valid_ids_value, list) or not valid_ids_value:
        raise ValueError("valid_sample_ids must be a nonempty ordered list.")
    valid_ids = [str(identifier) for identifier in valid_ids_value]
    if len(set(valid_ids)) != len(valid_ids):
        raise ValueError("valid_sample_ids contains duplicates.")
    excluded_ids = _excluded_ids(payload.get("excluded_samples"))
    if set(valid_ids).intersection(excluded_ids):
        raise ValueError("A sample cannot be both valid and excluded.")
    if set(valid_ids).union(excluded_ids) != set(source_ids):
        missing = sorted(set(source_ids) - set(valid_ids) - set(excluded_ids))
        extra = sorted((set(valid_ids) | set(excluded_ids)) - set(source_ids))
        raise ValueError(
            "QC sample partition does not exactly cover the source manifest: "
            f"missing={missing}, extra={extra}."
        )
    expected_valid_order = [identifier for identifier in source_ids if identifier in set(valid_ids)]
    expected_excluded_order = [
        identifier for identifier in source_ids if identifier in set(excluded_ids)
    ]
    if valid_ids != expected_valid_order:
        raise ValueError("valid_sample_ids do not preserve source-manifest ordering.")
    if excluded_ids != expected_excluded_order:
        raise ValueError("excluded_samples do not preserve source-manifest ordering.")
    return valid_ids


def evaluate_replay_qc(
    *,
    stored_raw: np.ndarray,
    replayed_raw: np.ndarray,
    stored_executed: np.ndarray,
    replayed_executed: np.ndarray,
    tolerance: float = RAW_MAX_ABSOLUTE_ERROR_TOLERANCE,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate one sample using only the pre-registered deterministic rules."""

    stored_raw = np.asarray(stored_raw)
    replayed_raw = np.asarray(replayed_raw)
    stored_executed = np.asarray(stored_executed)
    replayed_executed = np.asarray(replayed_executed)
    raw_shape_match = (
        stored_raw.shape == replayed_raw.shape == EXPECTED_ACTION_SHAPE
    )
    executed_shape_match = (
        stored_executed.shape == replayed_executed.shape == EXPECTED_ACTION_SHAPE
    )
    raw_finite = bool(np.all(np.isfinite(stored_raw))) and bool(
        np.all(np.isfinite(replayed_raw))
    )
    executed_finite = bool(np.all(np.isfinite(stored_executed))) and bool(
        np.all(np.isfinite(replayed_executed))
    )
    max_raw_absolute_error = (
        float(np.max(np.abs(replayed_raw.astype(np.float64) - stored_raw.astype(np.float64))))
        if raw_shape_match and raw_finite and stored_raw.size
        else None
    )
    gripper_exact = bool(
        executed_shape_match
        and executed_finite
        and stored_executed.ndim >= 1
        and stored_executed.shape[-1] >= 1
        and np.array_equal(stored_executed[..., -1], replayed_executed[..., -1])
    )

    reasons: list[str] = []
    if not raw_shape_match:
        reasons.append("raw_shape_mismatch")
    if not executed_shape_match:
        reasons.append("executed_shape_mismatch")
    if not raw_finite:
        reasons.append("nonfinite_raw_action")
    if not executed_finite:
        reasons.append("nonfinite_executed_action")
    if max_raw_absolute_error is None or max_raw_absolute_error > float(tolerance):
        reasons.append("raw_max_absolute_error_exceeds_tolerance")
    if not gripper_exact:
        reasons.append("postprocessed_gripper_mismatch")

    detail = {
        "reasons": reasons,
        "stored_raw_shape": list(stored_raw.shape),
        "replayed_raw_shape": list(replayed_raw.shape),
        "stored_executed_shape": list(stored_executed.shape),
        "replayed_executed_shape": list(replayed_executed.shape),
        "raw_actions_finite": raw_finite,
        "executed_actions_finite": executed_finite,
        "max_raw_absolute_error": max_raw_absolute_error,
        "postprocessed_gripper_exact": gripper_exact,
    }
    return not reasons, detail


def _validate_prompt_cache(
    cache_path: Path,
    *,
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Prompt cache must be a dict: {cache_path}")
    if int(payload.get("schema_version", -1)) != PROMPT_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported prompt cache schema in {cache_path}.")
    prompts = payload.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError(f"Prompt cache has no prompt records: {cache_path}")

    expected_tasks = {int(record["task_id"]) for record in source_records}
    observed_tasks: set[int] = set()
    for prompt, record in prompts.items():
        if not isinstance(prompt, str) or not isinstance(record, dict):
            raise TypeError("Prompt-cache keys must be strings and values must be objects.")
        task_id = int(record.get("task_id", -1))
        task_description = str(record.get("task_description", ""))
        if prompt != DEFAULT_PROMPT.format(task=task_description):
            raise ValueError(f"Prompt cache entry for task {task_id} is not DEFAULT_PROMPT.")
        context = record.get("context")
        context_mask = record.get("context_mask")
        if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
            raise TypeError(f"Prompt cache tensors are missing for task {task_id}.")
        if context.ndim != 3 or context.shape[0] != 1:
            raise ValueError(f"Invalid context shape for task {task_id}: {tuple(context.shape)}")
        if context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"Invalid context_mask shape for task {task_id}: {tuple(context_mask.shape)}"
            )
        if not bool(torch.isfinite(context).all()):
            raise ValueError(f"Prompt context contains NaN or Inf for task {task_id}.")
        if context_mask.dtype != torch.bool:
            raise TypeError(f"Prompt context_mask must be bool for task {task_id}.")
        if context.device.type != "cpu" or context_mask.device.type != "cpu":
            raise ValueError(f"Prompt cache tensors must reside on CPU for task {task_id}.")
        if task_id in observed_tasks:
            raise ValueError(f"Prompt cache contains more than one entry for task {task_id}.")
        observed_tasks.add(task_id)
    if observed_tasks != expected_tasks:
        raise ValueError(
            "Prompt-cache task IDs do not match the source manifest: "
            f"expected={sorted(expected_tasks)}, observed={sorted(observed_tasks)}."
        )
    return payload


def validate_existing_artifacts(
    *,
    valid_manifest_path: Path,
    prompt_cache_path: Path,
    source_manifest_path: Path,
    source_records: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    dataset_stats_path: Path,
    verify_checkpoint_hash: bool = True,
) -> dict[str, Any]:
    """Validate immutable QC artifacts against current source inputs."""

    if valid_manifest_path.exists() != prompt_cache_path.exists():
        raise FileExistsError(
            "QC artifacts are incomplete; refusing to overwrite either existing artifact: "
            f"manifest={valid_manifest_path.exists()}, cache={prompt_cache_path.exists()}."
        )
    if not valid_manifest_path.is_file() or not prompt_cache_path.is_file():
        raise FileNotFoundError("Both QC artifacts must exist for compatibility validation.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is unavailable: {checkpoint_path}")
    if not dataset_stats_path.is_file():
        raise FileNotFoundError(f"Dataset stats are unavailable: {dataset_stats_path}")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest is unavailable: {source_manifest_path}")
    payload = _read_json(valid_manifest_path)
    expected = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_stats_path": str(dataset_stats_path),
        "dataset_stats_sha256": sha256_file(dataset_stats_path),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "prompt_context_cache_path": str(prompt_cache_path),
        "prompt_context_cache_sha256": sha256_file(prompt_cache_path),
        "qc_rule": QC_RULE,
    }
    if verify_checkpoint_hash:
        expected["checkpoint_sha256"] = sha256_file(checkpoint_path)
    mismatches = {
        key: {"artifact": payload.get(key), "current": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if int(payload.get("schema_version", -1)) != QC_SCHEMA_VERSION:
        mismatches["schema_version"] = {
            "artifact": payload.get("schema_version"),
            "current": QC_SCHEMA_VERSION,
        }
    recorded_checkpoint_hash = payload.get("checkpoint_sha256")
    if not isinstance(recorded_checkpoint_hash, str) or len(recorded_checkpoint_hash) != 64:
        mismatches["checkpoint_sha256"] = {
            "artifact": recorded_checkpoint_hash,
            "current": "a 64-character SHA256 recorded by state-bank QC",
        }
    if mismatches:
        raise ValueError(
            "Existing state-bank QC artifacts are incompatible: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    _validate_sample_partition(payload, source_records)
    _validate_prompt_cache(prompt_cache_path, source_records=source_records)
    return payload


def _sample_arrays(sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = sample.get("baseline_raw_action")
    executed = sample.get("baseline_executed_action")
    if not isinstance(raw, torch.Tensor) or not isinstance(executed, torch.Tensor):
        raise TypeError("State-bank samples must contain tensor baseline actions.")
    return raw.detach().float().cpu().numpy(), executed.detach().float().cpu().numpy()


def _prompt_record_from_sample(sample: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    task_id = int(sample["task_id"])
    task_description = str(sample["task_description"])
    prompt = DEFAULT_PROMPT.format(task=task_description)
    infer_kwargs = sample.get("infer_action_kwargs")
    if not isinstance(infer_kwargs, dict):
        raise TypeError(f"Sample for task {task_id} has no infer_action_kwargs object.")
    context = infer_kwargs.get("context")
    context_mask = infer_kwargs.get("context_mask")
    if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
        raise TypeError(f"Sample for task {task_id} has no stored prompt context tensors.")
    return prompt, {
        "task_id": task_id,
        "task_description": task_description,
        "context": context.detach().cpu().clone(),
        "context_mask": context_mask.detach().to(device="cpu", dtype=torch.bool).clone(),
    }


def _merge_prompt_record(
    by_task: dict[int, tuple[str, dict[str, Any]]],
    sample: Mapping[str, Any],
) -> None:
    prompt, candidate = _prompt_record_from_sample(sample)
    task_id = int(candidate["task_id"])
    previous = by_task.get(task_id)
    if previous is None:
        by_task[task_id] = (prompt, candidate)
        return
    previous_prompt, previous_record = previous
    if (
        prompt != previous_prompt
        or candidate["task_description"] != previous_record["task_description"]
        or not torch.equal(candidate["context"], previous_record["context"])
        or not torch.equal(candidate["context_mask"], previous_record["context_mask"])
    ):
        raise ValueError(f"Stored DEFAULT_PROMPT context is inconsistent within task {task_id}.")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="sim_libero.yaml")
def validate_state_bank(cfg: DictConfig) -> None:
    # Keep simulator-dependent imports out of module import so the pure QC and
    # manifest validators remain usable in lightweight unit-test environments.
    from experiments.libero.eval_libero_single import (
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _postprocess_action,
        _resolve_dataset_stats_path,
        _resolve_eval_device,
        _run_prepared_action_inference,
    )
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    configured_tolerance = float(
        cfg.ASRE_DIAGNOSIS.get(
            "qc_max_abs_tolerance", RAW_MAX_ABSOLUTE_ERROR_TOLERANCE
        )
    )
    if configured_tolerance != RAW_MAX_ABSOLUTE_ERROR_TOLERANCE:
        raise ValueError(
            "ASRE Round-2 QC is pre-registered at qc_max_abs_tolerance=1e-6; "
            f"got {configured_tolerance}."
        )
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    state_bank_dir = _resolve_path(
        cfg.ASRE_DIAGNOSIS.get("state_bank_dir"),
        label="ASRE_DIAGNOSIS.state_bank_dir",
    )
    valid_manifest_path = _resolve_path(
        cfg.ASRE_DIAGNOSIS.get("valid_state_bank_manifest_path"),
        label="ASRE_DIAGNOSIS.valid_state_bank_manifest_path",
    )
    prompt_cache_value = cfg.EVALUATION.get("prompt_context_cache_path")
    prompt_cache_path = (
        valid_manifest_path.parent / "prompt_context_cache.pt"
        if prompt_cache_value is None
        else _resolve_path(prompt_cache_value, label="EVALUATION.prompt_context_cache_path")
    )
    source_manifest_path = (state_bank_dir / "manifest.jsonl").resolve()
    source_records = load_manifest(source_manifest_path)
    if len(source_records) != EXPECTED_SOURCE_SAMPLES:
        raise ValueError(
            f"Round-2 QC expects the original {EXPECTED_SOURCE_SAMPLES}-sample bank, "
            f"got {len(source_records)} from {source_manifest_path}."
        )
    _ordered_source_ids(source_records)

    checkpoint_path = _resolve_path(cfg.ckpt, label="ckpt")
    dataset_stats_path = _resolve_dataset_stats_path(cfg).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is unavailable: {checkpoint_path}")
    if not dataset_stats_path.is_file():
        raise FileNotFoundError(f"Dataset stats are unavailable: {dataset_stats_path}")
    source_metadata = _read_json(state_bank_dir / "run_metadata.json")
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
    source_mismatches = {
        key: {"state_bank": source_metadata.get(key), "requested": value}
        for key, value in source_compatibility.items()
        if source_metadata.get(key) != value
    }
    if source_mismatches:
        raise ValueError(
            "QC configuration is incompatible with the Round-1 state bank: "
            f"{json.dumps(source_mismatches, sort_keys=True)}"
        )

    if valid_manifest_path.exists() or prompt_cache_path.exists():
        payload = validate_existing_artifacts(
            valid_manifest_path=valid_manifest_path,
            prompt_cache_path=prompt_cache_path,
            source_manifest_path=source_manifest_path,
            source_records=source_records,
            checkpoint_path=checkpoint_path,
            dataset_stats_path=dataset_stats_path,
        )
        print(
            f"Validated existing immutable QC artifacts: "
            f"valid={len(payload['valid_sample_ids'])} "
            f"excluded={len(payload['excluded_samples'])}"
        )
        return

    cfg.model.load_text_encoder = False
    cfg.EVALUATION.text_encoder_device = None
    cfg.ASRE_DIAGNOSIS.enabled = False
    cfg.ASRE_DIAGNOSIS.protocol = "round1_drop_groups"
    cfg.ASRE_DIAGNOSIS.condition_name = "baseline"
    cfg.ASRE_DIAGNOSIS.condition_index = None
    cfg.ASRE_DIAGNOSIS.enabled_video_retrieval_layers = None
    cfg.ASRE_DIAGNOSIS.disabled_video_layers = []

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(checkpoint_path))
    model = model.to(model_device).eval()
    num_layers = get_num_model_layers(model)
    if num_layers != 30:
        raise ValueError(f"ASRE Round 2 requires 30 action layers, got {num_layers}.")

    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    valid_sample_ids: list[str] = []
    excluded_samples: list[dict[str, Any]] = []
    prompt_records_by_task: dict[int, tuple[str, dict[str, Any]]] = {}
    for index, manifest_record in enumerate(source_records, start=1):
        sample_id = str(manifest_record["sample_id"])
        sample_path = (state_bank_dir / str(manifest_record["sample_path"])).resolve()
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        if str(sample.get("sample_id", "")) != sample_id:
            raise ValueError(f"Sample identity mismatch for {sample_path}.")
        infer_kwargs = sample.get("infer_action_kwargs")
        if not isinstance(infer_kwargs, dict):
            raise TypeError(f"Sample {sample_id} has no infer_action_kwargs object.")
        replayed_raw_tensor, _ = _run_prepared_action_inference(
            model=model,
            cfg=cfg,
            infer_kwargs=infer_kwargs,
        )
        replayed_executed = _postprocess_action(replayed_raw_tensor, processor, cfg)
        stored_raw, stored_executed = _sample_arrays(sample)
        replayed_raw = replayed_raw_tensor.detach().float().cpu().numpy()
        passed, detail = evaluate_replay_qc(
            stored_raw=stored_raw,
            replayed_raw=replayed_raw,
            stored_executed=stored_executed,
            replayed_executed=replayed_executed,
            tolerance=configured_tolerance,
        )
        if passed:
            valid_sample_ids.append(sample_id)
            _merge_prompt_record(prompt_records_by_task, sample)
        else:
            excluded_samples.append(
                {
                    "sample_id": sample_id,
                    "sample_path": str(sample_path),
                    **detail,
                }
            )
        print(
            f"QC {index}/{len(source_records)} {sample_id}: "
            f"{'valid' if passed else 'excluded'} "
            f"max_raw_abs={detail['max_raw_absolute_error']}"
        )

    expected_tasks = {int(record["task_id"]) for record in source_records}
    if set(prompt_records_by_task) != expected_tasks:
        raise ValueError(
            "At least one valid state is required per task to construct the prompt cache: "
            f"valid_tasks={sorted(prompt_records_by_task)}, expected={sorted(expected_tasks)}."
        )
    prompt_payload = {
        "schema_version": PROMPT_CACHE_SCHEMA_VERSION,
        "artifact_type": "asre_round2_prompt_context_cache",
        "created_at": now_iso(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "prompts": {
            prompt: record
            for task_id, (prompt, record) in sorted(prompt_records_by_task.items())
        },
    }
    _atomic_torch_save(prompt_cache_path, prompt_payload)
    prompt_cache_sha256 = sha256_file(prompt_cache_path)

    valid_manifest_payload = {
        "schema_version": QC_SCHEMA_VERSION,
        "artifact_type": "asre_round2_valid_state_bank_manifest",
        "created_at": now_iso(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": prompt_payload["checkpoint_sha256"],
        "dataset_stats_path": str(dataset_stats_path),
        "dataset_stats_sha256": sha256_file(dataset_stats_path),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": prompt_payload["source_manifest_sha256"],
        "prompt_context_cache_path": str(prompt_cache_path),
        "prompt_context_cache_sha256": prompt_cache_sha256,
        "valid_sample_ids": valid_sample_ids,
        "excluded_samples": excluded_samples,
        "qc_rule": QC_RULE,
        "num_source_samples": len(source_records),
        "num_valid_samples": len(valid_sample_ids),
        "num_excluded_samples": len(excluded_samples),
    }
    _validate_sample_partition(valid_manifest_payload, source_records)
    _validate_prompt_cache(prompt_cache_path, source_records=source_records)
    atomic_write_json(valid_manifest_path, valid_manifest_payload)
    print(
        f"Wrote immutable Round-2 QC artifacts: valid={len(valid_sample_ids)} "
        f"excluded={len(excluded_samples)}\n"
        f"Manifest: {valid_manifest_path}\nPrompt cache: {prompt_cache_path}"
    )


if __name__ == "__main__":
    validate_state_bank()
