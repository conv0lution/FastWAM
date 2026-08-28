"""Create immutable CUDA-encoded, suite-specific prompt caches for ASRE G0."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
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
    git_commit,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.g0.definitions import (
    SUITE_ORDER,
    TASK_CONFIG,
    TASK_IDS,
    assert_output_scope,
)
from experiments.asre_diagnosis.round3b.donor import tensor_sha256
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


CACHE_SCHEMA_VERSION = 1
CACHE_ARTIFACT_TYPE = "asre_g0_suite_prompt_context_cache"
PROMPT_CONTEXT_STRATEGY = "suite_cuda_prompt_context_cache"
TEXT_CONDITIONING_SOURCE = "g0_suite_prompt_context_cache"
EXPECTED_CONTEXT_SHAPE = (1, 128, 4096)
EXPECTED_MASK_SHAPE = (1, 128)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _is_sha256(value: Any) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value)) is not None


def _expected_tasks_from_preflight(
    preflight: Mapping[str, Any], suite: str
) -> dict[int, str]:
    inventory = preflight.get("suite_inventory")
    if not isinstance(inventory, list):
        raise ValueError("G0 preflight has no suite inventory.")
    matches = [
        row
        for row in inventory
        if isinstance(row, Mapping) and row.get("suite_identifier") == suite
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("tasks"), list):
        raise ValueError(f"G0 preflight has no unique task inventory for {suite}.")
    tasks: dict[int, str] = {}
    for record in matches[0]["tasks"]:
        if not isinstance(record, Mapping):
            raise TypeError(f"Malformed G0 preflight task record for {suite}.")
        task_id = int(record.get("task_id", -1))
        language = str(record.get("task_language", ""))
        if task_id in tasks or not language:
            raise ValueError(f"Malformed/duplicate task {task_id} in {suite} preflight.")
        tasks[task_id] = language
    expected_ids = set(TASK_IDS)
    if set(tasks) != expected_ids:
        raise ValueError(
            f"G0 preflight task IDs differ for {suite}: "
            f"observed={sorted(tasks)}, expected={sorted(expected_ids)}."
        )
    return tasks


def _manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    prompts = payload.get("prompts")
    if not isinstance(prompts, Mapping):
        raise TypeError("G0 prompt cache has no prompt mapping.")
    records = []
    for prompt, record in prompts.items():
        if not isinstance(record, Mapping):
            raise TypeError(f"Malformed G0 prompt record for {prompt!r}.")
        records.append(
            {
                "task_id": int(record.get("task_id", -1)),
                "task_description": str(record.get("task_description", "")),
                "prompt": str(prompt),
                "context_shape": list(record.get("context_shape", [])),
                "context_dtype": str(record.get("context_dtype", "")),
                "context_sha256": str(record.get("context_sha256", "")),
                "context_mask_shape": list(record.get("context_mask_shape", [])),
                "context_mask_dtype": str(record.get("context_mask_dtype", "")),
                "context_mask_sha256": str(record.get("context_mask_sha256", "")),
            }
        )
    records.sort(key=lambda row: row["task_id"])
    return {
        "artifact_type": payload.get("artifact_type"),
        "schema_version": payload.get("schema_version"),
        "protocol": payload.get("protocol"),
        "strategy": payload.get("strategy"),
        "text_conditioning_source": payload.get("text_conditioning_source"),
        "git_commit_hash": payload.get("git_commit_hash"),
        "checkpoint_path": payload.get("checkpoint_path"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "task_suite": payload.get("task_suite"),
        "prompt_template": payload.get("prompt_template"),
        "encoding_device_type": payload.get("encoding_device_type"),
        "model_device": payload.get("model_device"),
        "text_encoder_device": payload.get("text_encoder_device"),
        "prompt_count": payload.get("prompt_count"),
        "records": records,
    }


def validate_prompt_context_cache(
    cache_path: Path,
    *,
    suite: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    expected_tasks: Mapping[int, str],
    expected_git_commit: str,
) -> dict[str, Any]:
    """Validate cache identity and every stored tensor, failing closed on drift."""

    path = cache_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"G0 prompt-context cache is unavailable: {path}.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"G0 prompt-context cache must contain a dict: {path}.")
    expected_top = {
        "artifact_type": CACHE_ARTIFACT_TYPE,
        "schema_version": CACHE_SCHEMA_VERSION,
        "protocol": G0_PROTOCOL,
        "strategy": PROMPT_CONTEXT_STRATEGY,
        "text_conditioning_source": TEXT_CONDITIONING_SOURCE,
        "git_commit_hash": expected_git_commit,
        "checkpoint_path": str(checkpoint.expanduser().resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "task_suite": suite,
        "prompt_template": DEFAULT_PROMPT,
        "encoding_device_type": "cuda",
        "model_device": "cuda:0",
        "text_encoder_device": "cuda:1",
        "prompt_count": len(TASK_IDS),
    }
    mismatches = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected_top.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"G0 prompt cache identity mismatch in {path}: {mismatches}.")
    prompts = payload.get("prompts")
    if not isinstance(prompts, dict) or len(prompts) != len(TASK_IDS):
        raise ValueError(f"G0 prompt cache must contain exactly 10 prompts: {path}.")

    observed_tasks: set[int] = set()
    for prompt, record in prompts.items():
        if not isinstance(prompt, str) or not isinstance(record, dict):
            raise TypeError(f"Malformed G0 prompt record in {path}.")
        task_id = int(record.get("task_id", -1))
        if task_id not in expected_tasks or task_id in observed_tasks:
            raise ValueError(f"Unexpected/duplicate task {task_id} in {path}.")
        task_description = str(record.get("task_description", ""))
        expected_description = str(expected_tasks[task_id])
        expected_prompt = DEFAULT_PROMPT.format(task=expected_description)
        if task_description != expected_description or prompt != expected_prompt:
            raise ValueError(f"Task text/prompt drift for {suite} task {task_id} in {path}.")
        context = record.get("context")
        context_mask = record.get("context_mask")
        if not torch.is_tensor(context) or not torch.is_tensor(context_mask):
            raise TypeError(f"Missing prompt tensors for {suite} task {task_id} in {path}.")
        if (
            tuple(context.shape) != EXPECTED_CONTEXT_SHAPE
            or context.dtype != torch.bfloat16
            or context.device.type != "cpu"
            or not bool(torch.isfinite(context.float()).all())
        ):
            raise ValueError(
                f"Invalid prompt context for {suite} task {task_id}: "
                f"shape={tuple(context.shape)}, dtype={context.dtype}, device={context.device}."
            )
        if (
            tuple(context_mask.shape) != EXPECTED_MASK_SHAPE
            or context_mask.dtype != torch.bool
            or context_mask.device.type != "cpu"
            or not bool(context_mask.all())
        ):
            raise ValueError(
                f"Invalid prompt mask for {suite} task {task_id}: "
                f"shape={tuple(context_mask.shape)}, dtype={context_mask.dtype}."
            )
        expected_record = {
            "context_shape": list(context.shape),
            "context_dtype": str(context.dtype),
            "context_sha256": tensor_sha256(context),
            "context_mask_shape": list(context_mask.shape),
            "context_mask_dtype": str(context_mask.dtype),
            "context_mask_sha256": tensor_sha256(context_mask),
        }
        record_mismatches = {
            key: {"observed": record.get(key), "expected": value}
            for key, value in expected_record.items()
            if record.get(key) != value
        }
        if record_mismatches:
            raise ValueError(
                f"G0 prompt tensor digest/shape drift for {suite} task {task_id}: "
                f"{record_mismatches}."
            )
        observed_tasks.add(task_id)
    if observed_tasks != set(TASK_IDS):
        raise ValueError(f"G0 prompt cache task population is incomplete: {path}.")

    expected_manifest = sha256_json(_manifest_payload(payload))
    if payload.get("prompt_context_manifest_sha256") != expected_manifest:
        raise ValueError(f"G0 prompt semantic manifest SHA256 mismatch: {path}.")
    if not _is_sha256(expected_manifest):
        raise ValueError(f"Invalid G0 prompt semantic digest: {expected_manifest!r}.")
    return payload


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _compose_model_config():
    with initialize_config_dir(
        config_dir=str((project_root / "configs").resolve()), version_base="1.3"
    ):
        return compose(config_name="sim_libero.yaml", overrides=[f"task={TASK_CONFIG}"])


def _require_two_dedicated_cuda_devices() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            "G0 prompt preparation requires exactly two visible CUDA devices "
            "(main model on cuda:0, T5 on cuda:1)."
        )
    insufficient: dict[int, int] = {}
    for device_index in range(2):
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        free_mib = int(free_bytes // 2**20)
        if free_mib < 22000:
            insufficient[device_index] = free_mib
    if insufficient:
        raise RuntimeError(
            "G0 prompt preparation needs two dedicated cards with at least 22000 MiB "
            f"free; logical-device free memory: {insufficient}."
        )


def _build_cache_payload(
    *,
    suite: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    task_descriptions: Mapping[int, str],
    prompt_tensors: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    commit: str,
) -> dict[str, Any]:
    prompts: dict[str, dict[str, Any]] = {}
    for task_id in TASK_IDS:
        task_description = str(task_descriptions[task_id])
        prompt = DEFAULT_PROMPT.format(task=task_description)
        context, context_mask = prompt_tensors[prompt]
        context = context.detach().to(device="cpu").contiguous().clone()
        context_mask = (
            context_mask.detach().to(device="cpu", dtype=torch.bool).contiguous().clone()
        )
        prompts[prompt] = {
            "task_id": task_id,
            "task_description": task_description,
            "context": context,
            "context_mask": context_mask,
            "context_shape": list(context.shape),
            "context_dtype": str(context.dtype),
            "context_sha256": tensor_sha256(context),
            "context_mask_shape": list(context_mask.shape),
            "context_mask_dtype": str(context_mask.dtype),
            "context_mask_sha256": tensor_sha256(context_mask),
        }
    payload: dict[str, Any] = {
        "artifact_type": CACHE_ARTIFACT_TYPE,
        "schema_version": CACHE_SCHEMA_VERSION,
        "protocol": G0_PROTOCOL,
        "strategy": PROMPT_CONTEXT_STRATEGY,
        "text_conditioning_source": TEXT_CONDITIONING_SOURCE,
        "created_at": now_iso(),
        "git_commit_hash": commit,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "task_suite": suite,
        "prompt_template": DEFAULT_PROMPT,
        "encoding_device_type": "cuda",
        "model_device": "cuda:0",
        "text_encoder_device": "cuda:1",
        "prompt_count": len(prompts),
        "prompts": prompts,
    }
    payload["prompt_context_manifest_sha256"] = sha256_json(
        _manifest_payload(payload)
    )
    return payload


@torch.no_grad()
def prepare_prompt_contexts(
    *, checkpoint: Path, preflight_report: Path, output_root: Path
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    preflight_report = preflight_report.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    assert_output_scope(output_root, project_root)
    if not checkpoint.is_file() or not preflight_report.is_file():
        raise FileNotFoundError(
            f"G0 prompt preparation inputs are unavailable: {checkpoint}, {preflight_report}."
        )
    preflight = _read_json(preflight_report, "G0 preflight report")
    commit = git_commit(project_root)
    expected_preflight = {
        "artifact_type": "asre_g0_preflight_report",
        "status": "compatible",
    }
    if any(preflight.get(key) != value for key, value in expected_preflight.items()):
        raise ValueError("G0 prompt preparation requires a compatible preflight report.")
    expected_output_root = output_root.parent.resolve()
    if (
        preflight.get("output_root") != str(expected_output_root)
        or output_root != expected_output_root / "prompt_contexts"
    ):
        raise ValueError(
            "G0 prompt cache output must be the prompt_contexts directory belonging "
            "to its preflight report."
        )
    if preflight.get("git", {}).get("head") != commit or preflight.get("git", {}).get(
        "allow_dirty"
    ):
        raise ValueError("G0 prompt preparation requires the current clean formal commit.")
    checkpoint_record = preflight.get("checkpoint", {})
    checkpoint_sha256 = str(checkpoint_record.get("sha256", ""))
    if (
        checkpoint_record.get("path") != str(checkpoint)
        or not _is_sha256(checkpoint_sha256)
    ):
        raise ValueError("G0 prompt preparation checkpoint differs from preflight.")

    expected_tasks = {
        suite: _expected_tasks_from_preflight(preflight, suite) for suite in SUITE_ORDER
    }
    cache_paths = {suite: output_root / f"{suite}.pt" for suite in SUITE_ORDER}
    pending = []
    for suite, path in cache_paths.items():
        if path.exists():
            validate_prompt_context_cache(
                path,
                suite=suite,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                expected_tasks=expected_tasks[suite],
                expected_git_commit=commit,
            )
            print(f"Validated existing G0 prompt cache: {path}", flush=True)
        else:
            pending.append(suite)
    if not pending:
        return {
            suite: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for suite, path in cache_paths.items()
        }

    _require_two_dedicated_cuda_devices()
    cfg = _compose_model_config()
    cfg.model.load_text_encoder = True
    cfg.EVALUATION.device = "cuda:0"
    cfg.EVALUATION.text_encoder_device = "cuda:1"
    model_dtype = torch.bfloat16
    model = instantiate(
        cfg.model,
        model_dtype=model_dtype,
        device="cuda:0",
        text_encoder_device="cuda:1",
    )
    model = model.eval()
    if getattr(model, "text_encoder", None) is None or getattr(
        model, "tokenizer", None
    ) is None:
        raise RuntimeError("G0 prompt preparation model did not load T5/tokenizer.")

    for suite in pending:
        encoded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for task_id in TASK_IDS:
            prompt = DEFAULT_PROMPT.format(task=expected_tasks[suite][task_id])
            context, context_mask = model.encode_prompt(prompt)
            encoded[prompt] = (context.detach().cpu(), context_mask.detach().cpu())
            del context, context_mask
        payload = _build_cache_payload(
            suite=suite,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            task_descriptions=expected_tasks[suite],
            prompt_tensors=encoded,
            commit=commit,
        )
        path = cache_paths[suite]
        _atomic_torch_save(path, payload)
        validate_prompt_context_cache(
            path,
            suite=suite,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            expected_tasks=expected_tasks[suite],
            expected_git_commit=commit,
        )
        print(f"Wrote immutable G0 CUDA prompt cache: {path}", flush=True)
        del encoded, payload

    return {
        suite: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for suite, path in cache_paths.items()
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = prepare_prompt_contexts(
        checkpoint=args.checkpoint,
        preflight_report=args.preflight_report,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
