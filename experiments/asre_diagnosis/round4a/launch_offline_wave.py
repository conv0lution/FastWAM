"""Launch one four-GPU Round-4A offline replay wave with checkpointed resume."""

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

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4A_PROTOCOL,
    Round4ACondition,
    atomic_write_json,
    build_round4a_conditions,
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
    _resolve_python,
)
from experiments.asre_diagnosis.round4a.launch_wave import (  # noqa: E402
    WAVE_CONDITIONS,
    _gpu_inventory,
    _mismatches,
)
from experiments.asre_diagnosis.round4a.provenance import (  # noqa: E402
    Round4AProvenance,
    load_round4a_provenance,
)


TASK_CONFIG = "libero_uncond_2cam224_1e-4"


@dataclass
class LiveChild:
    condition: Round4ACondition
    process: subprocess.Popen[Any]
    log_handle: Any
    status_path: Path
    status: dict[str, Any]


def offline_condition_command(
    *,
    python_path: Path,
    condition_index: int,
    condition: Round4ACondition,
    output_root: Path,
    provenance: Round4AProvenance,
) -> list[str]:
    compact = lambda value: json.dumps(value, separators=(",", ":"))
    scalar = lambda value: "null" if value is None else str(value)
    return [
        str(python_path),
        "-m",
        "experiments.asre_diagnosis.round4a.replay_state_bank",
        f"task={TASK_CONFIG}",
        f"ckpt={provenance.checkpoint_path}",
        "model.load_text_encoder=false",
        "gpu_id=0",
        "seed=42",
        "EVALUATION.device=cuda:0",
        "EVALUATION.text_encoder_device=null",
        f"EVALUATION.dataset_stats_path={provenance.dataset_stats_path}",
        "ASRE_DIAGNOSIS.enabled=true",
        "ASRE_DIAGNOSIS.mode=replace_video_kv",
        f"ASRE_DIAGNOSIS.protocol={ROUND4A_PROTOCOL}",
        f"ASRE_DIAGNOSIS.condition_index={condition_index}",
        f"ASRE_DIAGNOSIS.condition_name={condition.name}",
        f"ASRE_DIAGNOSIS.enabled_video_retrieval_layers={compact(list(condition.enabled_video_retrieval_layers(30)))}",
        f"ASRE_DIAGNOSIS.disabled_video_layers={compact(list(condition.disabled_video_layers))}",
        f"ASRE_DIAGNOSIS.replacement_video_layers={compact(list(condition.replacement_video_layers))}",
        f"ASRE_DIAGNOSIS.hybrid_axis={scalar(condition.hybrid_axis)}",
        f"ASRE_DIAGNOSIS.hybrid_mask_seed={scalar(condition.mask_seed)}",
        f"ASRE_DIAGNOSIS.hybrid_mask_manifest_path={provenance.mask_manifest_path}",
        f"ASRE_DIAGNOSIS.hybrid_mask_manifest_sha256={provenance.mask_manifest_sha256}",
        f"ASRE_DIAGNOSIS.token_mask_manifest_path={provenance.token_mask_manifest_path}",
        f"ASRE_DIAGNOSIS.token_mask_manifest_sha256={provenance.token_mask_manifest_sha256}",
        f"ASRE_DIAGNOSIS.head_mask_manifest_path={provenance.head_mask_manifest_path}",
        f"ASRE_DIAGNOSIS.head_mask_manifest_sha256={provenance.head_mask_manifest_sha256}",
        f"ASRE_DIAGNOSIS.state_bank_dir={provenance.state_bank_dir}",
        f"ASRE_DIAGNOSIS.offline_output_dir={output_root}",
        f"ASRE_DIAGNOSIS.valid_state_bank_manifest_path={provenance.valid_manifest_path}",
        f"ASRE_DIAGNOSIS.offline_donor_mapping_path={provenance.offline_donor_mapping_path}",
        "ASRE_DIAGNOSIS.executed_prefix_length=10",
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


def inspect_offline_condition(
    output_root: Path,
    *,
    condition: Round4ACondition,
    provenance: Round4AProvenance,
) -> tuple[str, int, str]:
    directory = output_root / condition.name
    if not directory.exists() or not any(directory.iterdir()):
        return "missing", 0, "directory absent or empty"
    metadata_path = directory / "run_metadata.json"
    if not metadata_path.is_file():
        raise DirectorySafetyError(f"Offline output lacks metadata: {directory}")
    metadata = _read_json(metadata_path, label="Round-4A offline metadata")
    expected = {
        "artifact_type": "asre_round4a_offline_state_bank_replay",
        "condition_protocol": ROUND4A_PROTOCOL,
        "diagnosis_condition": condition.name,
        "git_commit_hash": provenance.git_commit_hash,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "valid_state_bank_manifest_sha256": provenance.valid_manifest_sha256,
        "offline_donor_mapping_sha256": provenance.offline_donor_mapping_sha256,
        "hybrid_axis": condition.hybrid_axis,
        "hybrid_mask_seed": condition.mask_seed,
        "hybrid_mask_manifest_sha256": provenance.mask_manifest_sha256,
        "token_mask_manifest_sha256": provenance.token_mask_manifest_sha256,
        "head_mask_manifest_sha256": provenance.head_mask_manifest_sha256,
        "num_valid_samples": 499,
        "executed_prefix_length": 10,
    }
    mismatch = _mismatches(metadata, expected)
    if mismatch:
        raise DirectorySafetyError(
            "Incompatible offline replay metadata: " + json.dumps(mismatch)
        )
    completed = int(metadata.get("completed_samples", 0))
    records_path = directory / "per_sample.jsonl"
    actions_path = directory / "actions.npz"
    if records_path.is_file() != actions_path.is_file():
        raise DirectorySafetyError("Offline replay has an incomplete checkpoint pair.")
    if records_path.is_file():
        with records_path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        with np.load(actions_path, allow_pickle=False) as payload:
            sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
            raw = np.asarray(payload["raw_actions"])
            executed = np.asarray(payload["executed_actions"])
        record_ids = [str(record.get("sample_id")) for record in records]
        if sample_ids != record_ids or len(records) != completed:
            raise DirectorySafetyError("Offline checkpoint IDs/count disagree with metadata.")
        if (
            raw.shape != executed.shape
            or raw.shape != (completed, 32, 7)
            or not np.all(np.isfinite(raw))
            or not np.all(np.isfinite(executed))
        ):
            raise DirectorySafetyError("Offline checkpoint actions are malformed/nonfinite.")
        if any(record.get("condition") != condition.name for record in records):
            raise DirectorySafetyError("Offline per-sample condition identity drifted.")
    elif completed != 0:
        raise DirectorySafetyError("Offline metadata records progress without checkpoints.")
    if str(metadata.get("status")) == "complete":
        if completed != 499:
            raise DirectorySafetyError("Completed offline replay does not contain 499 states.")
        return "complete", completed, "499 ordered state outputs validated"
    if str(metadata.get("status")) not in {"running", "failed", "interrupted", ""}:
        raise DirectorySafetyError(f"Unsupported offline status: {metadata.get('status')!r}")
    return "partial", completed, "checkpoint is resumable"


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
    if not child.log_handle.closed:
        child.log_handle.close()
    attempt = child.status["attempts"][-1]
    attempt["exit_status"] = int(child.process.returncode or 1)
    attempt["end_timestamp"] = now_iso()
    attempt["status"] = "interrupted"
    child.status["status"] = "interrupted"
    atomic_write_json(child.status_path, child.status)


def _launch_child(
    *,
    wave: int,
    condition_index: int,
    condition: Round4ACondition,
    physical_gpu: int,
    output_root: Path,
    python_path: Path,
    provenance: Round4AProvenance,
    lock_fd: int,
) -> LiveChild:
    status_path = output_root / f"offline_wave{wave}_{condition.name}.status.json"
    previous = _read_json(status_path, label="offline launcher status") if status_path.exists() else None
    attempts = [] if previous is None else list(previous.get("attempts", []))
    attempt_number = len(attempts) + 1
    logs = output_root.parent / "logs" / f"offline_wave{wave}"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{condition.name}.attempt{attempt_number:02d}.log"
    command = offline_condition_command(
        python_path=python_path,
        condition_index=condition_index,
        condition=condition,
        output_root=output_root,
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
    attempts.append(attempt)
    status = {
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "kind": "offline",
        "wave": wave,
        "condition_index": condition_index,
        "condition": condition.name,
        "physical_gpu": physical_gpu,
        "no_ddp": True,
        "status": "launching",
        "attempts": attempts,
    }
    atomic_write_json(status_path, status)
    log_handle = log_path.open("x", encoding="utf-8")
    log_handle.write(shlex.join(command) + "\n")
    log_handle.flush()
    environment = _child_environment(physical_gpu)
    environment["ASRE_ROUND4A_PHYSICAL_GPU"] = str(physical_gpu)
    environment["ASRE_ROUND3B_TRUSTED_PREFLIGHT_MANIFEST_SHA256"] = (
        provenance.valid_manifest_sha256
    )
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        pass_fds=(lock_fd,),
    )
    status["attempts"][-1]["pid"] = process.pid
    status["attempts"][-1]["status"] = "running"
    status["status"] = "running"
    atomic_write_json(status_path, status)
    return LiveChild(condition, process, log_handle, status_path, status)


def run_launcher(args: argparse.Namespace) -> Path:
    if args.wave not in (1, 2):
        raise ValueError("Offline wave must be 1 or 2.")
    provenance = load_round4a_provenance(
        preflight_report_path=args.preflight_report,
        machinery_report_path=args.machinery_report,
        mask_manifest_path=args.mask_manifest,
    )
    if args.wave == 2:
        if args.wave1_summary is None:
            raise ValueError("Offline wave 2 requires --wave1-summary.")
        wave1 = _read_json(args.wave1_summary.resolve(), label="offline wave-1 summary")
        expected = {
            "schema_version": 1,
            "protocol": ROUND4A_PROTOCOL,
            "kind": "offline",
            "wave": 1,
            "all_succeeded": True,
            "provenance": provenance.identity_dict(),
        }
        mismatch = _mismatches(wave1, expected)
        if mismatch:
            raise ValueError("Offline wave-1 gate failed: " + json.dumps(mismatch))
    elif args.wave1_summary is not None:
        raise ValueError("--wave1-summary is only valid for wave 2.")
    python_path = _resolve_python(args.python)
    gpu_ids = tuple(int(value) for value in args.gpu_ids)
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or any(value < 0 for value in gpu_ids):
        raise ValueError(f"Offline wave requires four distinct GPU IDs: {gpu_ids}.")
    inventory = _gpu_inventory()
    inventory_by_id = {int(record["index"]): record for record in inventory}
    inventory_ids = set(inventory_by_id)
    if any(gpu not in inventory_ids for gpu in gpu_ids):
        raise ValueError(f"Selected offline GPUs are unavailable: {gpu_ids}.")
    stagger = float(args.launch_stagger_seconds)
    if not math.isfinite(stagger) or stagger < 0:
        raise ValueError("Offline launch stagger must be finite and nonnegative.")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = build_round4a_conditions(30)
    indices = WAVE_CONDITIONS[args.wave]
    identity = {
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "kind": "offline",
        "wave": args.wave,
        "condition_indices": list(indices),
        "conditions": [conditions[index].name for index in indices],
        "gpu_ids": list(gpu_ids),
        "gpu_inventory": [inventory_by_id[gpu] for gpu in gpu_ids],
        "launch_stagger_seconds": stagger,
        "git_commit_hash": provenance.git_commit_hash,
        "python": str(python_path),
        "provenance": provenance.identity_dict(),
        "no_ddp": True,
    }
    identity["identity_sha256"] = sha256_json(identity)
    config_path = output_root / f"offline_wave{args.wave}_launcher_config.json"
    if config_path.exists():
        if _read_json(config_path, label="offline wave config") != identity:
            raise DirectorySafetyError("Offline wave output identity changed.")
    else:
        atomic_write_json(config_path, identity)
    initial_states = {}
    launch = []
    for slot, index in enumerate(indices):
        condition = conditions[index]
        state, completed, detail = inspect_offline_condition(
            output_root, condition=condition, provenance=provenance
        )
        initial_states[condition.name] = {
            "state": state,
            "completed_samples": completed,
            "physical_gpu": gpu_ids[slot],
            "detail": detail,
        }
        if state != "complete":
            launch.append((slot, index, condition))
    children: list[LiveChild] = []
    failure = None
    interrupted = False
    start = now_iso()
    with _launcher_lock(output_root.parent / f"offline_wave{args.wave}") as lock_fd:
        try:
            for position, (slot, index, condition) in enumerate(launch):
                children.append(
                    _launch_child(
                        wave=args.wave,
                        condition_index=index,
                        condition=condition,
                        physical_gpu=gpu_ids[slot],
                        output_root=output_root,
                        python_path=python_path,
                        provenance=provenance,
                        lock_fd=lock_fd,
                    )
                )
                if position + 1 < len(launch) and stagger:
                    time.sleep(stagger)
            pending = list(children)
            while pending:
                for child in list(pending):
                    returncode = child.process.poll()
                    if returncode is None:
                        continue
                    pending.remove(child)
                    child.log_handle.close()
                    attempt = child.status["attempts"][-1]
                    attempt["exit_status"] = returncode
                    attempt["end_timestamp"] = now_iso()
                    attempt["status"] = "completed" if returncode == 0 else "failed"
                    child.status["status"] = attempt["status"]
                    atomic_write_json(child.status_path, child.status)
                    if returncode != 0:
                        failure = f"Offline {child.condition.name} exited {returncode}."
                        for other in pending:
                            _terminate(other)
                        pending.clear()
                        break
                if pending:
                    time.sleep(2)
        except KeyboardInterrupt:
            interrupted = True
            failure = "offline launcher interrupted"
            for child in children:
                if child.process.poll() is None:
                    _terminate(child)
    final_states = {}
    all_complete = True
    for slot, index in enumerate(indices):
        condition = conditions[index]
        try:
            state, completed, detail = inspect_offline_condition(
                output_root, condition=condition, provenance=provenance
            )
        except Exception as exc:
            state, completed, detail = "invalid", 0, str(exc)
        final_states[condition.name] = {
            "state": state,
            "completed_samples": completed,
            "physical_gpu": gpu_ids[slot],
            "detail": detail,
        }
        all_complete &= state == "complete"
    summary_path = output_root / f"offline_wave{args.wave}_launcher_summary.json"
    atomic_write_json(
        summary_path,
        {
            "schema_version": 1,
            "protocol": ROUND4A_PROTOCOL,
            "kind": "offline",
            "wave": args.wave,
            "all_succeeded": bool(all_complete and not interrupted and failure is None),
            "interrupted": interrupted,
            "failure": failure,
            "condition_indices": list(indices),
            "expected_conditions": [conditions[index].name for index in indices],
            "gpu_ids": list(gpu_ids),
            "git_commit_hash": provenance.git_commit_hash,
            "provenance": provenance.identity_dict(),
            "initial_condition_states": initial_states,
            "condition_states": final_states,
            "start_timestamp": start,
            "end_timestamp": now_iso(),
            "no_ddp": True,
        },
    )
    if not all_complete or interrupted or failure:
        raise RuntimeError(failure or f"Offline wave {args.wave} incomplete.")
    return summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, choices=(1, 2), required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--machinery-report", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wave1-summary", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-ids", nargs=4, type=int, default=(0, 1, 2, 3))
    parser.add_argument("--launch-stagger-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_launcher(args)
    print(f"Round-4A offline wave {args.wave} complete: {summary}")


if __name__ == "__main__":
    main()
