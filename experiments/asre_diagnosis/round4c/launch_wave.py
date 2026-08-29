"""Launch one isolated-GPU Round-4C online smoke/full wave."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4C_PROTOCOL,
    Round4BCondition,
    atomic_write_json,
    build_round4c_conditions,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.g0.launch_four_gpu import (  # noqa: E402
    _child_environment,
    _launcher_lock,
)
from experiments.asre_diagnosis.round4a.launch_wave import _gpu_inventory  # noqa: E402
from experiments.asre_diagnosis.round4c.definitions import (  # noqa: E402
    CUMULATIVE_ENERGY_COMMIT,
    REGISTERED_HELDOUT_ENERGY,
    ROUND4B_SOURCE_COMMIT,
    WAVES,
)


TASK_SUITE = "libero_spatial"
TASK_CONFIG = "libero_uncond_2cam224_1e-4"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _expected_condition(index: int):
    return build_round4c_conditions(30)[index]


def _validate_smoke_donor_assignments(
    assignments: Any, *, condition: Round4BCondition
) -> None:
    if not isinstance(assignments, list):
        raise ValueError(f"Smoke donor assignments drifted for {condition.name}.")

    # The current endpoint performs no late-layer replacement, so it must not
    # emit per-episode donor assignments. Replacement conditions must record
    # both frozen smoke-trial assignments and verify the recipient query image.
    if not condition.replacement_video_layers:
        if assignments:
            raise ValueError(f"Smoke donor assignments drifted for {condition.name}.")
        return

    if (
        len(assignments) != 2
        or [int(row["recipient_trial"]) for row in assignments] != [0, 1]
        or any(int(row["recipient_task_id"]) != 0 for row in assignments)
        or any(
            row.get("recipient_first_query_image_verified") is not True
            for row in assignments
        )
    ):
        raise ValueError(f"Smoke donor assignments drifted for {condition.name}.")


def _complete(
    root: Path, *, index: int, task_ids: tuple[int, ...], trials: int
) -> bool:
    condition = _expected_condition(index)
    metadata_path = root / condition.name / "run_metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = _read(metadata_path)
    config = metadata.get("condition_config", {})
    if not (
        metadata.get("status") == "completed"
        and metadata.get("condition_protocol") == ROUND4C_PROTOCOL
        and metadata.get("diagnosis_condition") == condition.name
        and metadata.get("git_commit_hash") == git_commit(PROJECT_ROOT)
        and metadata.get("task_suite") == TASK_SUITE
        and metadata.get("task_ids") == list(task_ids)
        and metadata.get("number_of_trials") == trials
        and metadata.get("seed") == 42
        and metadata.get("action_horizon") == 32
        and metadata.get("number_of_inference_steps") == 10
        and metadata.get("replan_steps") == 10
        and config.get("subspace_basis_kind") == condition.basis_kind
        and config.get("subspace_rank") == condition.subspace_rank
        and config.get("disabled_video_layers") == list(range(15))
        and config.get("replacement_video_layers")
        == list(condition.replacement_video_layers)
    ):
        return False
    files = sorted((root / condition.name / TASK_SUITE).glob("gpu*_task*_results.json"))
    if len(files) != len(task_ids):
        return False
    observed_tasks: set[int] = set()
    for path in files:
        result = _read(path)
        task_id = int(result.get("task_id", -1))
        successes = set(map(int, result.get("success_episodes", [])))
        failures = set(map(int, result.get("failure_episodes", [])))
        if (
            task_id not in task_ids
            or task_id in observed_tasks
            or successes & failures
            or successes | failures != set(range(trials))
        ):
            return False
        observed_tasks.add(task_id)
    return observed_tasks == set(task_ids)


def _validate_smoke(root: Path, *, index: int, basis_sha256: str, donor_sha256: str) -> None:
    condition = _expected_condition(index)
    directory = root / condition.name
    metadata = _read(directory / "run_metadata.json")
    if metadata.get("subspace_basis_manifest_sha256") != basis_sha256:
        raise ValueError(f"Smoke basis hash mismatch for {condition.name}.")
    if metadata.get("donor_mapping_sha256") != donor_sha256:
        raise ValueError(f"Smoke donor hash mismatch for {condition.name}.")
    result_paths = list((directory / TASK_SUITE).glob("gpu*_task0_results.json"))
    if len(result_paths) != 1:
        raise ValueError(f"Smoke task provenance is incomplete for {condition.name}.")
    result = _read(result_paths[0])
    assignments = result.get("donor_assignments", [])
    _validate_smoke_donor_assignments(assignments, condition=condition)
    traces = sorted((directory / TASK_SUITE / "action_traces").glob("task0_trial*.jsonl"))
    if len(traces) != 2:
        raise ValueError(f"Smoke action traces are incomplete for {condition.name}.")
    trace_rows = 0
    for path in traces:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            raw = np.asarray(row.get("raw_action"), dtype=np.float64)
            executed = np.asarray(row.get("executed_action"), dtype=np.float64)
            if (
                raw.shape != (32, 7)
                or executed.shape != (32, 7)
                or not np.isfinite(raw).all()
                or not np.isfinite(executed).all()
                or row.get("replacement_video_layers")
                != list(condition.replacement_video_layers)
                or row.get("subspace_basis_kind") != condition.basis_kind
                or row.get("subspace_rank") != condition.subspace_rank
                or row.get("subspace_basis_manifest_sha256") != basis_sha256
            ):
                raise ValueError(f"Invalid smoke action trace: {path}")
            trace_rows += 1
    if trace_rows == 0:
        raise ValueError(f"Smoke produced no action queries for {condition.name}.")


def _gate(path: Path | None, *, mode: str, wave: int) -> None:
    if path is None:
        raise ValueError(f"Round-4C {mode} wave {wave} gate is required.")
    value = _read(path.resolve())
    if not (
        value.get("protocol") == ROUND4C_PROTOCOL
        and value.get("mode") == mode
        and value.get("wave") == wave
        and value.get("all_succeeded") is True
    ):
        raise ValueError(f"Round-4C prerequisite gate failed: {path}")


def launch(args: argparse.Namespace) -> Path:
    if args.mode == "full":
        _gate(args.smoke_summary, mode="smoke", wave=args.wave)
        if args.wave == 2:
            _gate(args.wave1_full_summary, mode="full", wave=1)
    elif args.wave == 2:
        _gate(args.wave1_smoke_summary, mode="smoke", wave=1)
        _gate(args.wave1_full_summary, mode="full", wave=1)
    indices = WAVES[args.wave]
    task_ids = (0,) if args.mode == "smoke" else tuple(range(10))
    trials = 2 if args.mode == "smoke" else 10
    gpu_ids = tuple(args.gpu_ids)
    if len(gpu_ids) != len(indices) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Round-4C wave {args.wave} requires {len(indices)} distinct GPU IDs.")
    inventory = _gpu_inventory()
    available = {int(item["index"]) for item in inventory}
    if set(gpu_ids) - available:
        raise ValueError("Requested Round-4C GPU is unavailable.")
    preflight = _read(args.preflight.resolve())
    machinery = _read(args.machinery.resolve())
    if preflight.get("status") != "compatible" or machinery.get("passed") is not True:
        raise ValueError("Round-4C preflight/machinery gate is incomplete.")
    state = preflight["state_bank"]
    donors = preflight["donors"]
    paths = {
        "preflight_report": args.preflight.resolve(),
        "machinery_report": args.machinery.resolve(),
        "calibration_split_manifest": args.split.resolve(),
        "subspace_basis_manifest": args.basis_manifest.resolve(),
        "subspace_diagnostics": args.diagnostics.resolve(),
        "energy_analysis_manifest": args.energy_manifest.resolve(),
        "energy_candidate_ranks": args.energy_candidates.resolve(),
        "round4b_summary": args.round4b_summary.resolve(),
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    conditions = build_round4c_conditions(30)
    children = []
    failure = None
    start = now_iso()
    with _launcher_lock(output_root.parent / f"{args.mode}_wave{args.wave}") as lock_fd:
        for slot, index in enumerate(indices):
            condition = conditions[index]
            if _complete(output_root, index=index, task_ids=task_ids, trials=trials):
                if args.mode == "smoke":
                    _validate_smoke(
                        output_root,
                        index=index,
                        basis_sha256=sha256_file(args.basis_manifest.resolve()),
                        donor_sha256=donors["mapping_sha256"],
                    )
                continue
            condition_root = output_root / condition.name
            condition_root.mkdir(parents=True, exist_ok=True)
            attempt = 1
            log = logs / f"{condition.name}.attempt{attempt:02d}.log"
            while log.exists():
                attempt += 1
                log = logs / f"{condition.name}.attempt{attempt:02d}.log"
            compact = lambda value: json.dumps(value, separators=(",", ":"))
            scalar = lambda value: "null" if value is None else str(value)
            command = [
                str(args.python.resolve()),
                str(PROJECT_ROOT / "experiments/libero/eval_libero_single.py"),
                f"task={TASK_CONFIG}",
                f"ckpt={state['checkpoint_path']}",
                "model.load_text_encoder=false",
                "gpu_id=0",
                "seed=42",
                "EVALUATION.device=cuda:0",
                "EVALUATION.text_encoder_device=null",
                f"EVALUATION.prompt_context_cache_path={state['prompt_context_cache_path']}",
                f"EVALUATION.task_suite_name={TASK_SUITE}",
                f"EVALUATION.task_ids={compact(task_ids)}",
                f"EVALUATION.num_trials={trials}",
                "EVALUATION.action_horizon=32",
                "EVALUATION.num_inference_steps=10",
                "EVALUATION.replan_steps=10",
                f"EVALUATION.output_dir={condition_root}",
                "EVALUATION.visualize_future_video=false",
                f"EVALUATION.dataset_stats_path={state['dataset_stats_path']}",
                "ASRE_DIAGNOSIS.enabled=true",
                "ASRE_DIAGNOSIS.mode=replace_video_kv",
                f"ASRE_DIAGNOSIS.protocol={ROUND4C_PROTOCOL}",
                f"ASRE_DIAGNOSIS.condition_index={index}",
                f"ASRE_DIAGNOSIS.condition_name={condition.name}",
                f"ASRE_DIAGNOSIS.enabled_video_retrieval_layers={compact(condition.enabled_video_retrieval_layers(30))}",
                f"ASRE_DIAGNOSIS.disabled_video_layers={compact(condition.disabled_video_layers)}",
                f"ASRE_DIAGNOSIS.replacement_video_layers={compact(condition.replacement_video_layers)}",
                f"ASRE_DIAGNOSIS.subspace_basis_kind={scalar(condition.basis_kind)}",
                f"ASRE_DIAGNOSIS.subspace_rank={scalar(condition.subspace_rank)}",
                f"ASRE_DIAGNOSIS.checkpoint_sha256={state['checkpoint_sha256']}",
                f"ASRE_DIAGNOSIS.dataset_stats_sha256={state['dataset_stats_sha256']}",
                f"ASRE_DIAGNOSIS.state_bank_manifest_path={state['source_manifest_path']}",
                f"ASRE_DIAGNOSIS.state_bank_manifest_sha256={state['source_manifest_sha256']}",
                f"ASRE_DIAGNOSIS.valid_state_bank_manifest_path={state['valid_manifest_path']}",
                f"ASRE_DIAGNOSIS.valid_state_bank_manifest_sha256={state['valid_manifest_sha256']}",
                f"ASRE_DIAGNOSIS.prompt_context_cache_sha256={state['prompt_context_cache_sha256']}",
                f"ASRE_DIAGNOSIS.donor_mapping_path={donors['mapping_path']}",
                f"ASRE_DIAGNOSIS.donor_mapping_sha256={donors['mapping_sha256']}",
                f"ASRE_DIAGNOSIS.donor_observation_manifest_path={donors['observation_manifest_path']}",
                f"ASRE_DIAGNOSIS.donor_observation_manifest_sha256={donors['observation_manifest_sha256']}",
                f"ASRE_DIAGNOSIS.donor_observation_root={donors['observation_root']}",
                f"ASRE_DIAGNOSIS.round4b_source_commit={ROUND4B_SOURCE_COMMIT}",
                f"ASRE_DIAGNOSIS.cumulative_energy_analysis_commit={CUMULATIVE_ENERGY_COMMIT}",
                "ASRE_DIAGNOSIS.save_rollout_video=false",
                "ASRE_DIAGNOSIS.save_action_trace=true",
            ]
            for stem, path in paths.items():
                command.extend(
                    [
                        f"ASRE_DIAGNOSIS.{stem}_path={path}",
                        f"ASRE_DIAGNOSIS.{stem}_sha256={sha256_file(path)}",
                    ]
                )
            handle = log.open("x", encoding="utf-8")
            handle.write(shlex.join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=_child_environment(gpu_ids[slot]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock_fd,),
            )
            children.append((condition.name, process, handle, log))
            if args.launch_stagger_seconds and slot + 1 < len(indices):
                time.sleep(args.launch_stagger_seconds)
        pending = list(children)
        while pending:
            for child in list(pending):
                name, process, handle, log = child
                code = process.poll()
                if code is None:
                    continue
                pending.remove(child)
                handle.close()
                if code:
                    failure = f"Online {name} exited {code}; inspect {log}."
                    for _, other, other_handle, _ in pending:
                        if other.poll() is None:
                            os.killpg(other.pid, signal.SIGTERM)
                        other_handle.close()
                    pending.clear()
                    break
            if pending:
                time.sleep(2)
    states = {
        conditions[index].name: _complete(
            output_root, index=index, task_ids=task_ids, trials=trials
        )
        for index in indices
    }
    if failure is None and all(states.values()) and args.mode == "smoke":
        for index in indices:
            _validate_smoke(
                output_root,
                index=index,
                basis_sha256=sha256_file(args.basis_manifest.resolve()),
                donor_sha256=donors["mapping_sha256"],
            )
    summary = output_root / "launcher_summary.json"
    atomic_write_json(
        summary,
        {
            "artifact_type": "asre_round4c_online_wave_summary",
            "schema_version": 1,
            "protocol": ROUND4C_PROTOCOL,
            "mode": args.mode,
            "wave": args.wave,
            "all_succeeded": all(states.values()) and failure is None,
            "conditions": [conditions[index].name for index in indices],
            "condition_indices": list(indices),
            "gpu_ids": list(gpu_ids),
            "gpu_inventory": [item for item in inventory if int(item["index"]) in gpu_ids],
            "task_ids": list(task_ids),
            "num_trials": trials,
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "condition_complete": states,
            "registered_heldout_energy": {
                str(conditions[index].subspace_rank): REGISTERED_HELDOUT_ENERGY[
                    int(conditions[index].subspace_rank)
                ]
                for index in indices
                if conditions[index].subspace_rank is not None
            },
            "failure": failure,
            "start_timestamp": start,
            "end_timestamp": now_iso(),
            "no_ddp": True,
            "spare_gpus_used_for_unrelated_work": False,
        },
    )
    if failure or not all(states.values()):
        raise RuntimeError(failure or "Round-4C online wave incomplete.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--wave", type=int, choices=(1, 2), required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs="+", type=int, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--machinery", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--energy-manifest", type=Path, required=True)
    parser.add_argument("--energy-candidates", type=Path, required=True)
    parser.add_argument("--round4b-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--wave1-smoke-summary", type=Path)
    parser.add_argument("--wave1-full-summary", type=Path)
    parser.add_argument("--launch-stagger-seconds", type=float, default=30.0)
    path = launch(parser.parse_args())
    print(f"Round-4C online wave complete: {path}")


if __name__ == "__main__":
    main()
