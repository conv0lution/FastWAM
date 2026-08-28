"""Launch one frozen Round-4A wave on isolated GPUs (never DDP)."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4A_PROTOCOL,
    Round4ACondition,
    atomic_write_json,
    build_round4a_conditions,
    git_commit,
    now_iso,
    sha256_json,
)
from experiments.asre_diagnosis.g0.launch_four_gpu import (  # noqa: E402
    _child_environment,
    _launcher_lock,
)
from experiments.asre_diagnosis.round3a.launch_three_gpu import (  # noqa: E402
    DirectorySafetyError,
    _read_json,
    _recorded_live_pid,
    _resolve_python,
    _validate_action_array,
)
from experiments.asre_diagnosis.round4a.provenance import (  # noqa: E402
    Round4AProvenance,
    load_round4a_provenance,
)


TASK_SUITE = "libero_spatial"
TASK_CONFIG = "libero_uncond_2cam224_1e-4"
SEED = 42
ACTION_HORIZON = 32
INFERENCE_STEPS = 10
REPLAN_STEPS = 10
SMOKE_TASK_IDS = (0,)
FULL_TASK_IDS = tuple(range(10))
SMOKE_TRIALS = 2
FULL_TRIALS = 10
WAVE_CONDITIONS = {
    1: (0, 1, 2, 5),
    2: (3, 4, 6, 7),
}
SMOKE_CONDITIONS = {
    1: WAVE_CONDITIONS[1],
    2: (3, 6),
}


@dataclass(frozen=True)
class Runtime:
    mode: str
    wave: int
    task_ids: tuple[int, ...]
    num_trials: int
    condition_indices: tuple[int, ...]


@dataclass
class LiveChild:
    condition_index: int
    condition: Round4ACondition
    physical_gpu: int
    process: subprocess.Popen[Any]
    log_handle: Any
    status_path: Path
    status: dict[str, Any]


def resolve_runtime(mode: str, wave: int) -> Runtime:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported Round-4A mode: {mode}.")
    if wave not in WAVE_CONDITIONS:
        raise ValueError(f"Round-4A wave must be 1 or 2, got {wave}.")
    return Runtime(
        mode=mode,
        wave=wave,
        task_ids=SMOKE_TASK_IDS if mode == "smoke" else FULL_TASK_IDS,
        num_trials=SMOKE_TRIALS if mode == "smoke" else FULL_TRIALS,
        condition_indices=(
            SMOKE_CONDITIONS[wave] if mode == "smoke" else WAVE_CONDITIONS[wave]
        ),
    )


def _require_clean_worktree() -> None:
    output = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            ".",
            ":(exclude)asre_results/round4a/**",
        ],
        cwd=PROJECT_ROOT,
        text=True,
    )
    if output.strip():
        raise RuntimeError(
            "Round-4A launch requires a clean source worktree; commit code first.\n"
            + output
        )


def _gpu_inventory() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    records = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise RuntimeError(f"Unexpected nvidia-smi output: {line!r}")
        records.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "pci_bus_id": fields[3],
                "driver_version": fields[4],
                "memory_total_mib": int(fields[5]),
                "memory_free_mib_at_launch": int(fields[6]),
            }
        )
    if not records:
        raise RuntimeError("nvidia-smi returned no GPUs.")
    return records


def _validate_gpu_ids(values: Sequence[int], runtime: Runtime) -> tuple[int, ...]:
    gpu_ids = tuple(int(value) for value in values)
    if len(gpu_ids) != len(runtime.condition_indices):
        raise ValueError(
            f"{runtime.mode} wave {runtime.wave} requires "
            f"{len(runtime.condition_indices)} GPU IDs, got {gpu_ids}."
        )
    if any(value < 0 for value in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU IDs must be distinct and nonnegative: {gpu_ids}.")
    return gpu_ids


def condition_command(
    *,
    python_path: Path,
    condition_index: int,
    condition: Round4ACondition,
    condition_output: Path,
    runtime: Runtime,
    provenance: Round4AProvenance,
) -> list[str]:
    compact = lambda value: json.dumps(value, separators=(",", ":"))
    null_or_value = lambda value: "null" if value is None else str(value)
    enabled = list(condition.enabled_video_retrieval_layers(30))
    return [
        str(python_path),
        str(PROJECT_ROOT / "experiments" / "libero" / "eval_libero_single.py"),
        f"task={TASK_CONFIG}",
        f"ckpt={provenance.checkpoint_path}",
        "model.load_text_encoder=false",
        "gpu_id=0",
        f"seed={SEED}",
        "EVALUATION.device=cuda:0",
        "EVALUATION.text_encoder_device=null",
        f"EVALUATION.prompt_context_cache_path={provenance.prompt_context_cache_path}",
        f"EVALUATION.task_suite_name={TASK_SUITE}",
        f"EVALUATION.task_ids={compact(list(runtime.task_ids))}",
        f"EVALUATION.num_trials={runtime.num_trials}",
        f"EVALUATION.action_horizon={ACTION_HORIZON}",
        f"EVALUATION.num_inference_steps={INFERENCE_STEPS}",
        f"EVALUATION.replan_steps={REPLAN_STEPS}",
        f"EVALUATION.output_dir={condition_output}",
        "EVALUATION.visualize_future_video=false",
        f"EVALUATION.dataset_stats_path={provenance.dataset_stats_path}",
        "ASRE_DIAGNOSIS.enabled=true",
        "ASRE_DIAGNOSIS.mode=replace_video_kv",
        f"ASRE_DIAGNOSIS.protocol={ROUND4A_PROTOCOL}",
        f"ASRE_DIAGNOSIS.condition_index={condition_index}",
        f"ASRE_DIAGNOSIS.condition_name={condition.name}",
        f"ASRE_DIAGNOSIS.enabled_video_retrieval_layers={compact(enabled)}",
        f"ASRE_DIAGNOSIS.disabled_video_layers={compact(list(condition.disabled_video_layers))}",
        f"ASRE_DIAGNOSIS.replacement_video_layers={compact(list(condition.replacement_video_layers))}",
        f"ASRE_DIAGNOSIS.hybrid_axis={null_or_value(condition.hybrid_axis)}",
        f"ASRE_DIAGNOSIS.hybrid_mask_seed={null_or_value(condition.mask_seed)}",
        f"ASRE_DIAGNOSIS.hybrid_mask_manifest_path={provenance.mask_manifest_path}",
        f"ASRE_DIAGNOSIS.hybrid_mask_manifest_sha256={provenance.mask_manifest_sha256}",
        f"ASRE_DIAGNOSIS.token_mask_manifest_path={provenance.token_mask_manifest_path}",
        f"ASRE_DIAGNOSIS.token_mask_manifest_sha256={provenance.token_mask_manifest_sha256}",
        f"ASRE_DIAGNOSIS.head_mask_manifest_path={provenance.head_mask_manifest_path}",
        f"ASRE_DIAGNOSIS.head_mask_manifest_sha256={provenance.head_mask_manifest_sha256}",
        "ASRE_DIAGNOSIS.save_rollout_video=false",
        "ASRE_DIAGNOSIS.save_action_trace=true",
        f"ASRE_DIAGNOSIS.checkpoint_sha256={provenance.checkpoint_sha256}",
        f"ASRE_DIAGNOSIS.dataset_stats_sha256={provenance.dataset_stats_sha256}",
        f"ASRE_DIAGNOSIS.state_bank_manifest_path={provenance.source_manifest_path}",
        f"ASRE_DIAGNOSIS.state_bank_manifest_sha256={provenance.source_manifest_sha256}",
        f"ASRE_DIAGNOSIS.valid_state_bank_manifest_path={provenance.valid_manifest_path}",
        f"ASRE_DIAGNOSIS.valid_state_bank_manifest_sha256={provenance.valid_manifest_sha256}",
        f"ASRE_DIAGNOSIS.prompt_context_cache_sha256={provenance.prompt_context_cache_sha256}",
        f"ASRE_DIAGNOSIS.donor_mapping_path={provenance.online_donor_mapping_path}",
        f"ASRE_DIAGNOSIS.donor_mapping_sha256={provenance.online_donor_mapping_sha256}",
        f"ASRE_DIAGNOSIS.donor_observation_manifest_path={provenance.online_donor_manifest_path}",
        f"ASRE_DIAGNOSIS.donor_observation_manifest_sha256={provenance.online_donor_manifest_sha256}",
        f"ASRE_DIAGNOSIS.donor_observation_root={provenance.online_donor_root}",
        f"ASRE_DIAGNOSIS.preflight_report_path={provenance.preflight_report_path}",
        f"ASRE_DIAGNOSIS.preflight_report_sha256={provenance.preflight_report_sha256}",
        f"ASRE_DIAGNOSIS.machinery_report_path={provenance.machinery_report_path}",
        f"ASRE_DIAGNOSIS.machinery_report_sha256={provenance.machinery_report_sha256}",
        f"ASRE_DIAGNOSIS.round3b_parent_tag={provenance.round3b_parent_tag}",
        f"ASRE_DIAGNOSIS.round3b_parent_commit={provenance.round3b_parent_commit}",
        f"ASRE_DIAGNOSIS.g0_parent_tag={provenance.g0_parent_tag}",
        f"ASRE_DIAGNOSIS.g0_parent_commit={provenance.g0_parent_commit}",
        f"ASRE_DIAGNOSIS.g0_summary_path={provenance.g0_summary_path}",
        f"ASRE_DIAGNOSIS.g0_summary_sha256={provenance.g0_summary_sha256}",
    ]


def _expected_metadata(
    condition: Round4ACondition,
    runtime: Runtime,
    provenance: Round4AProvenance,
) -> dict[str, Any]:
    return {
        "git_commit_hash": provenance.git_commit_hash,
        "checkpoint_path": str(provenance.checkpoint_path),
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "dataset_stats_path": str(provenance.dataset_stats_path),
        "dataset_stats_sha256": provenance.dataset_stats_sha256,
        "diagnosis_condition": condition.name,
        "condition_protocol": ROUND4A_PROTOCOL,
        "enabled_video_retrieval_layers": list(
            condition.enabled_video_retrieval_layers(30)
        ),
        "disabled_video_layers": list(condition.disabled_video_layers),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "num_model_layers": 30,
        "task_suite": TASK_SUITE,
        "task_ids": list(runtime.task_ids),
        "seed": SEED,
        "number_of_trials": runtime.num_trials,
        "action_horizon": ACTION_HORIZON,
        "number_of_inference_steps": INFERENCE_STEPS,
        "replan_steps": REPLAN_STEPS,
        "prompt_context_cache_path": str(provenance.prompt_context_cache_path),
        "prompt_context_cache_sha256": provenance.prompt_context_cache_sha256,
        "hybrid_mask_manifest_path": str(provenance.mask_manifest_path),
        "hybrid_mask_manifest_sha256": provenance.mask_manifest_sha256,
        "token_mask_manifest_path": str(provenance.token_mask_manifest_path),
        "token_mask_manifest_sha256": provenance.token_mask_manifest_sha256,
        "head_mask_manifest_path": str(provenance.head_mask_manifest_path),
        "head_mask_manifest_sha256": provenance.head_mask_manifest_sha256,
        "preflight_report_sha256": provenance.preflight_report_sha256,
        "machinery_report_sha256": provenance.machinery_report_sha256,
        "round3b_parent_commit": provenance.round3b_parent_commit,
        "g0_parent_commit": provenance.g0_parent_commit,
        "g0_summary_sha256": provenance.g0_summary_sha256,
    }


def _mismatches(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }


def _validate_trace(
    path: Path,
    *,
    condition: Round4ACondition,
    task_id: int,
    episode: int,
    provenance: Round4AProvenance,
) -> None:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise DirectorySafetyError(f"Malformed trace object: {path}")
                records.append(payload)
    if not records:
        raise DirectorySafetyError(f"Empty action trace: {path}")
    for replan_id, record in enumerate(records):
        expected = {
            "task_suite": TASK_SUITE,
            "task_id": task_id,
            "episode_id": episode,
            "replan_id": replan_id,
            "diagnosis_condition": condition.name,
            "environment_step": 30 + replan_id * REPLAN_STEPS,
            "action_inference_seed": SEED,
            "replacement_video_layers": list(condition.replacement_video_layers),
            "hybrid_axis": condition.hybrid_axis,
            "hybrid_mask_seed": condition.mask_seed,
            "hybrid_mask_manifest_sha256": provenance.mask_manifest_sha256,
        }
        mismatch = _mismatches(record, expected)
        if mismatch:
            raise DirectorySafetyError(
                f"Round-4A trace identity mismatch in {path}: {json.dumps(mismatch)}"
            )
        if condition.replacement_video_layers:
            if int(record.get("donor_trial", -1)) != (episode + 1) % 10:
                raise DirectorySafetyError(f"Wrong donor identity in {path}.")
            if int(record.get("donor_replan_id", replan_id)) != replan_id:
                raise DirectorySafetyError(f"Wrong same-replan donor identity in {path}.")
        _validate_action_array(record.get("raw_action"), path=path, field="raw_action")
        _validate_action_array(
            record.get("executed_action"), path=path, field="executed_action"
        )


def inspect_condition(
    condition_dir: Path,
    *,
    condition: Round4ACondition,
    runtime: Runtime,
    provenance: Round4AProvenance,
) -> tuple[str, tuple[int, ...], str]:
    if not condition_dir.exists() or not any(condition_dir.iterdir()):
        return "missing", (), "directory absent or empty"
    metadata_path = condition_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return "partial", (), "run metadata not written yet"
    metadata = _read_json(metadata_path, label="Round-4A condition metadata")
    mismatch = _mismatches(metadata, _expected_metadata(condition, runtime, provenance))
    condition_config = metadata.get("condition_config", {})
    expected_config = {
        "hybrid_axis": condition.hybrid_axis,
        "hybrid_mask_seed": condition.mask_seed,
    }
    mismatch.update(
        {
            f"condition_config.{key}": value
            for key, value in _mismatches(condition_config, expected_config).items()
        }
    )
    if mismatch:
        raise DirectorySafetyError(
            "Incompatible Round-4A condition output: " + json.dumps(mismatch)
        )
    completed = []
    for task_id in runtime.task_ids:
        matches = list(condition_dir.glob(f"**/gpu*_task{task_id}_results.json"))
        if len(matches) > 1:
            raise DirectorySafetyError(f"Duplicate result for task {task_id}.")
        if not matches:
            continue
        result = _read_json(matches[0], label="Round-4A task result")
        result_expected = {
            "task_suite": TASK_SUITE,
            "task_id": task_id,
            "diagnosis_condition": condition.name,
            "condition_protocol": ROUND4A_PROTOCOL,
            "total_episodes": runtime.num_trials,
            "hybrid_axis": condition.hybrid_axis,
            "hybrid_mask_seed": condition.mask_seed,
            "hybrid_mask_manifest_sha256": provenance.mask_manifest_sha256,
        }
        result_mismatch = _mismatches(result, result_expected)
        if result_mismatch:
            raise DirectorySafetyError(
                f"Incompatible task result {matches[0]}: {json.dumps(result_mismatch)}"
            )
        successes = [int(value) for value in result.get("success_episodes", [])]
        failures = [int(value) for value in result.get("failure_episodes", [])]
        if (
            set(successes) & set(failures)
            or set(successes) | set(failures) != set(range(runtime.num_trials))
            or len(successes) != len(set(successes))
            or len(failures) != len(set(failures))
        ):
            raise DirectorySafetyError(f"Malformed paired outcomes: {matches[0]}")
        for episode in range(runtime.num_trials):
            trace_path = (
                condition_dir
                / TASK_SUITE
                / "action_traces"
                / f"task{task_id}_trial{episode}.jsonl"
            )
            if not trace_path.is_file():
                raise DirectorySafetyError(f"Missing action trace: {trace_path}")
            _validate_trace(
                trace_path,
                condition=condition,
                task_id=task_id,
                episode=episode,
                provenance=provenance,
            )
        completed.append(task_id)
    if list(condition_dir.rglob("*.mp4")):
        raise DirectorySafetyError(f"Rollout videos are forbidden in {condition_dir}.")
    all_complete = set(completed) == set(runtime.task_ids)
    status = str(metadata.get("status", ""))
    if status == "completed" and all_complete:
        return "complete", tuple(sorted(completed)), "all outputs validated"
    if status == "completed" and not all_complete:
        raise DirectorySafetyError("Metadata claims completion but task outputs are missing.")
    if status not in {"", "running", "failed", "interrupted"}:
        raise DirectorySafetyError(f"Unsupported metadata status: {status!r}.")
    return "partial", tuple(sorted(completed)), f"metadata={status}"


def _root_identity(
    *,
    runtime: Runtime,
    provenance: Round4AProvenance,
    python_path: Path,
    gpu_ids: Sequence[int],
    gpu_inventory: Sequence[Mapping[str, Any]],
    stagger_seconds: float,
) -> dict[str, Any]:
    conditions = build_round4a_conditions(30)
    payload = {
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "mode": runtime.mode,
        "wave": runtime.wave,
        "task_suite": TASK_SUITE,
        "task_config": TASK_CONFIG,
        "task_ids": list(runtime.task_ids),
        "num_trials": runtime.num_trials,
        "seed": SEED,
        "action_horizon": ACTION_HORIZON,
        "inference_steps": INFERENCE_STEPS,
        "replan_steps": REPLAN_STEPS,
        "no_ddp": True,
        "git_commit_hash": provenance.git_commit_hash,
        "python": str(python_path),
        "gpu_ids": list(gpu_ids),
        "launch_stagger_seconds": stagger_seconds,
        "condition_indices": list(runtime.condition_indices),
        "conditions": [
            {
                "slot": slot,
                "condition_index": index,
                "physical_gpu": gpu_ids[slot],
                **conditions[index].to_dict(),
            }
            for slot, index in enumerate(runtime.condition_indices)
        ],
        "provenance": provenance.identity_dict(),
        "gpu_inventory": list(gpu_inventory),
    }
    payload["identity_sha256"] = sha256_json(payload)
    return payload


def _validate_gate(
    path: Path | None,
    *,
    mode: str,
    wave: int,
    provenance: Round4AProvenance,
) -> None:
    if path is None:
        raise ValueError(f"A successful {mode} wave-{wave} summary is required.")
    summary = _read_json(path.resolve(), label=f"Round-4A {mode} wave-{wave} gate")
    expected = {
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "mode": mode,
        "wave": wave,
        "all_succeeded": True,
        "interrupted": False,
        "git_commit_hash": provenance.git_commit_hash,
        "provenance": provenance.identity_dict(),
    }
    mismatch = _mismatches(summary, expected)
    if mismatch:
        raise ValueError("Round-4A prerequisite gate failed: " + json.dumps(mismatch))


def _status_payload(
    *,
    runtime: Runtime,
    condition_index: int,
    condition: Round4ACondition,
    physical_gpu: int,
    output_dir: Path,
    previous: Mapping[str, Any] | None,
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    attempts = []
    if previous is not None and isinstance(previous.get("attempts"), list):
        attempts.extend(previous["attempts"])
    attempts.append(dict(attempt))
    return {
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "mode": runtime.mode,
        "wave": runtime.wave,
        "condition_index": condition_index,
        "condition": condition.name,
        "hybrid_axis": condition.hybrid_axis,
        "hybrid_mask_seed": condition.mask_seed,
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": str(physical_gpu),
        "logical_model_device": "cuda:0",
        "no_ddp": True,
        "output_dir": str(output_dir),
        "status": attempt["status"],
        "attempts": attempts,
    }


def _launch(
    *,
    runtime: Runtime,
    condition_index: int,
    condition: Round4ACondition,
    physical_gpu: int,
    output_root: Path,
    python_path: Path,
    provenance: Round4AProvenance,
    lock_fd: int,
) -> LiveChild:
    output_dir = output_root / condition.name
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "launcher_status.json"
    previous = _read_json(status_path, label="launcher status") if status_path.exists() else None
    if previous is not None:
        live_pid = _recorded_live_pid(previous)
        if live_pid is not None:
            raise DirectorySafetyError(
                f"Condition {condition.name} still has live PID {live_pid}."
            )
    attempts = [] if previous is None else previous.get("attempts", [])
    attempt_number = len(attempts) + 1 if isinstance(attempts, list) else 1
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{condition.name}.attempt{attempt_number:02d}.log"
    if log_path.exists():
        raise DirectorySafetyError(f"Refusing to overwrite launch log: {log_path}")
    command = condition_command(
        python_path=python_path,
        condition_index=condition_index,
        condition=condition,
        condition_output=output_dir,
        runtime=runtime,
        provenance=provenance,
    )
    attempt = {
        "attempt": attempt_number,
        "start_timestamp": now_iso(),
        "end_timestamp": None,
        "pid": None,
        "exit_status": None,
        "status": "launching",
        "log_path": str(log_path),
        "command": shlex.join(command),
    }
    status = _status_payload(
        runtime=runtime,
        condition_index=condition_index,
        condition=condition,
        physical_gpu=physical_gpu,
        output_dir=output_dir,
        previous=previous,
        attempt=attempt,
    )
    atomic_write_json(status_path, status)
    log_handle = log_path.open("x", encoding="utf-8")
    log_handle.write(
        f"[{now_iso()}] {condition.name} physical_gpu={physical_gpu} "
        "logical_cuda=0 no_DDP=true\n" + shlex.join(command) + "\n"
    )
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=_child_environment(physical_gpu),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        pass_fds=(lock_fd,),
    )
    status["attempts"][-1]["pid"] = process.pid
    status["attempts"][-1]["status"] = "running"
    status["status"] = "running"
    atomic_write_json(status_path, status)
    return LiveChild(
        condition_index,
        condition,
        physical_gpu,
        process,
        log_handle,
        status_path,
        status,
    )


def _finish(child: LiveChild, status: str) -> None:
    if not child.log_handle.closed:
        child.log_handle.close()
    attempt = child.status["attempts"][-1]
    attempt["exit_status"] = int(child.process.returncode or 0)
    attempt["end_timestamp"] = now_iso()
    attempt["status"] = status
    child.status["status"] = status
    atomic_write_json(child.status_path, child.status)


def _terminate(child: LiveChild) -> None:
    if child.process.poll() is None:
        try:
            os.killpg(child.process.pid, signal.SIGTERM)
            child.process.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if child.process.poll() is None:
                try:
                    os.killpg(child.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.process.wait(timeout=20)
    _finish(child, "interrupted")


def run_launcher(args: argparse.Namespace) -> Path:
    _require_clean_worktree()
    runtime = resolve_runtime(args.mode, args.wave)
    python_path = _resolve_python(args.python)
    provenance = load_round4a_provenance(
        preflight_report_path=args.preflight_report,
        machinery_report_path=args.machinery_report,
        mask_manifest_path=args.mask_manifest,
    )
    if runtime.mode == "full":
        _validate_gate(
            args.smoke_summary,
            mode="smoke",
            wave=runtime.wave,
            provenance=provenance,
        )
        if runtime.wave == 2:
            _validate_gate(
                args.wave1_full_summary,
                mode="full",
                wave=1,
                provenance=provenance,
            )
    elif args.smoke_summary is not None or args.wave1_full_summary is not None:
        raise ValueError("Prerequisite summaries are only valid in full mode.")
    gpu_ids = _validate_gpu_ids(args.gpu_ids, runtime)
    stagger = float(args.launch_stagger_seconds)
    if not math.isfinite(stagger) or stagger < 0:
        raise ValueError("Launch stagger must be finite and nonnegative.")
    inventory = _gpu_inventory()
    inventory_by_id = {int(record["index"]): record for record in inventory}
    if any(gpu not in inventory_by_id for gpu in gpu_ids):
        raise ValueError(f"Selected GPUs are absent from inventory: {gpu_ids}.")
    selected_inventory = [inventory_by_id[gpu] for gpu in gpu_ids]
    output_root = args.output_root.expanduser().resolve()
    identity = _root_identity(
        runtime=runtime,
        provenance=provenance,
        python_path=python_path,
        gpu_ids=gpu_ids,
        gpu_inventory=selected_inventory,
        stagger_seconds=stagger,
    )
    config_path = output_root / "launcher_config.json"
    if config_path.exists():
        if _read_json(config_path, label="launcher config") != identity:
            raise DirectorySafetyError(f"Output root belongs to another run: {output_root}")
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise DirectorySafetyError(
                f"Nonempty output root lacks launcher identity: {output_root}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_path, identity)
    allowed = {
        "launcher_config.json",
        "launcher_summary.json",
        "logs",
        *(build_round4a_conditions(30)[index].name for index in runtime.condition_indices),
    }
    unexpected = [path.name for path in output_root.iterdir() if path.name not in allowed]
    if unexpected:
        raise DirectorySafetyError(f"Unexpected launcher output entries: {unexpected}")

    conditions = build_round4a_conditions(30)
    initial_states = {}
    to_launch = []
    for slot, condition_index in enumerate(runtime.condition_indices):
        condition = conditions[condition_index]
        state, completed, detail = inspect_condition(
            output_root / condition.name,
            condition=condition,
            runtime=runtime,
            provenance=provenance,
        )
        initial_states[condition.name] = {
            "state": state,
            "completed_task_ids": list(completed),
            "detail": detail,
            "physical_gpu": gpu_ids[slot],
        }
        if state != "complete":
            to_launch.append((slot, condition_index, condition))

    start = now_iso()
    interrupted = False
    failure: str | None = None
    live: list[LiveChild] = []
    with _launcher_lock(output_root) as lock_fd:
        try:
            for launch_position, (slot, condition_index, condition) in enumerate(to_launch):
                live.append(
                    _launch(
                        runtime=runtime,
                        condition_index=condition_index,
                        condition=condition,
                        physical_gpu=gpu_ids[slot],
                        output_root=output_root,
                        python_path=python_path,
                        provenance=provenance,
                        lock_fd=lock_fd,
                    )
                )
                if launch_position + 1 < len(to_launch) and stagger:
                    time.sleep(stagger)
            pending = list(live)
            while pending:
                for child in list(pending):
                    returncode = child.process.poll()
                    if returncode is None:
                        continue
                    pending.remove(child)
                    _finish(child, "completed" if returncode == 0 else "failed")
                    if returncode != 0:
                        failure = (
                            f"{child.condition.name} exited {returncode}; inspect its log."
                        )
                        for other in pending:
                            _terminate(other)
                        pending.clear()
                        break
                if pending:
                    time.sleep(2)
        except KeyboardInterrupt:
            interrupted = True
            failure = "launcher interrupted"
            for child in live:
                if child.process.poll() is None:
                    _terminate(child)

    final_states = {}
    all_complete = True
    for slot, condition_index in enumerate(runtime.condition_indices):
        condition = conditions[condition_index]
        try:
            state, completed, detail = inspect_condition(
                output_root / condition.name,
                condition=condition,
                runtime=runtime,
                provenance=provenance,
            )
        except Exception as exc:
            state, completed, detail = "invalid", (), str(exc)
        final_states[condition.name] = {
            "state": state,
            "completed_task_ids": list(completed),
            "detail": detail,
            "physical_gpu": gpu_ids[slot],
        }
        all_complete &= state == "complete"
    summary_path = output_root / "launcher_summary.json"
    atomic_write_json(
        summary_path,
        {
            "schema_version": 1,
            "protocol": ROUND4A_PROTOCOL,
            "mode": runtime.mode,
            "wave": runtime.wave,
            "all_succeeded": bool(all_complete and not interrupted and failure is None),
            "interrupted": interrupted,
            "failure": failure,
            "output_root": str(output_root),
            "task_ids": list(runtime.task_ids),
            "num_trials": runtime.num_trials,
            "seed": SEED,
            "git_commit_hash": provenance.git_commit_hash,
            "gpu_ids": list(gpu_ids),
            "launch_stagger_seconds": stagger,
            "condition_indices": list(runtime.condition_indices),
            "expected_conditions": [
                conditions[index].name for index in runtime.condition_indices
            ],
            "provenance": provenance.identity_dict(),
            "initial_condition_states": initial_states,
            "condition_states": final_states,
            "start_timestamp": start,
            "end_timestamp": now_iso(),
            "no_ddp": True,
        },
    )
    if not all_complete or interrupted or failure is not None:
        raise RuntimeError(failure or f"Round-4A wave {runtime.wave} is incomplete.")
    return summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--wave", type=int, choices=(1, 2), required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--machinery-report", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--wave1-full-summary", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None)
    parser.add_argument("--launch-stagger-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime = resolve_runtime(args.mode, args.wave)
    if args.gpu_ids is None:
        args.gpu_ids = list(range(len(runtime.condition_indices)))
    summary = run_launcher(args)
    print(f"Round-4A {args.mode} wave {args.wave} complete: {summary}")


if __name__ == "__main__":
    main()
