"""Launch one isolated four-GPU online smoke/full wave for ASRE Salvage A."""

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
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_A_PROTOCOL,
    Round4BCondition,
    atomic_write_json,
    build_salvage_a_conditions,
    git_commit,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.g0.launch_four_gpu import (  # noqa: E402
    _child_environment,
    _launcher_lock,
)
from experiments.asre_diagnosis.round4a.launch_wave import _gpu_inventory  # noqa: E402
from experiments.asre_diagnosis.salvage_a.definitions import (  # noqa: E402
    ACTION_HORIZON,
    CONDITIONS,
    INFERENCE_STEPS,
    ONLINE_TRIALS_PER_TASK,
    REPLAN_STEPS,
    SEED,
    SMOKE_TRIALS,
    TASK_CONFIG,
    TASK_IDS,
    TASK_SUITE,
    WAVES,
    validate_frozen_condition_matrix,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _expected_condition(index: int) -> Round4BCondition:
    conditions = validate_frozen_condition_matrix()
    condition = conditions[index]
    assert isinstance(condition, Round4BCondition)
    return condition


def _split_trials(split: Mapping[str, Any]) -> dict[tuple[int, int], str]:
    records = split.get("per_task")
    if not isinstance(records, list) or len(records) != 10:
        raise ValueError("Salvage A split must define ten tasks.")
    result: dict[tuple[int, int], str] = {}
    for record in records:
        task_id = int(record["task_id"])
        for split_name, key in (
            ("calibration", "calibration_episode_ids"),
            ("heldout", "heldout_episode_ids"),
        ):
            trials = list(map(int, record[key]))
            if len(trials) != 5 or len(set(trials)) != 5:
                raise ValueError(f"Malformed Salvage A split for task {task_id}.")
            for trial in trials:
                pair = (task_id, trial)
                if pair in result:
                    raise ValueError(f"Duplicate split membership: {pair}.")
                result[pair] = split_name
    expected = {(task, trial) for task in TASK_IDS for trial in range(10)}
    if set(result) != expected:
        raise ValueError("Salvage A split does not cover task/trial pairs 0..9.")
    return result


def _donor_pairs(mapping: Mapping[str, Any]) -> dict[tuple[int, int], tuple[int, int]]:
    records = mapping.get("records")
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("Salvage A donor mapping must contain 100 records.")
    pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for row in records:
        recipient = (int(row["task_id"]), int(row["recipient_trial"]))
        donor = (
            int(row.get("donor_task_id", row["task_id"])),
            int(row["donor_trial"]),
        )
        if recipient in pairs:
            raise ValueError(f"Duplicate donor recipient: {recipient}.")
        pairs[recipient] = donor
    expected = {(task, trial) for task in TASK_IDS for trial in range(10)}
    if set(pairs) != expected:
        raise ValueError("Salvage A donor mapping does not cover all online recipients.")
    return pairs


def _validate_assignments(
    assignments: Any,
    *,
    condition: Round4BCondition,
    task_id: int,
    trials: int,
    split_by_trial: Mapping[tuple[int, int], str],
    donor_pairs: Mapping[tuple[int, int], tuple[int, int]],
) -> None:
    if not isinstance(assignments, list):
        raise ValueError(f"Donor assignments drifted for {condition.name}.")
    if not condition.replacement_video_layers:
        if assignments:
            raise ValueError(f"Donor assignments drifted for {condition.name}.")
        return
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in assignments:
        recipient_trial = int(row["recipient_trial"])
        recipient = (int(row["recipient_task_id"]), recipient_trial)
        donor = (int(row["donor_task_id"]), int(row["donor_trial"]))
        if (
            recipient[0] != task_id
            or recipient_trial in indexed
            or donor != donor_pairs.get(recipient)
            or recipient == donor
            or split_by_trial.get(recipient) != split_by_trial.get(donor)
            or row.get("recipient_first_query_image_verified") is not True
        ):
            raise ValueError(f"Donor assignments drifted for {condition.name}: {row}.")
        indexed[recipient_trial] = row
    if set(indexed) != set(range(trials)):
        raise ValueError(f"Donor assignments drifted for {condition.name}.")


def _metadata_identity(
    *,
    preflight: Path,
    machinery: Path,
    split: Path,
    state_selection: Path,
    differentiable_path_report: Path,
    basis_manifest: Path,
    diagnostics: Path,
    donor_mapping: Path,
    donor_manifest: Path,
    donor_root: Path,
    round4c_summary: Path,
) -> dict[str, Any]:
    payload = {
        "protocol": SALVAGE_A_PROTOCOL,
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "preflight_report_path": str(preflight),
        "preflight_report_sha256": sha256_file(preflight),
        "machinery_report_path": str(machinery),
        "machinery_report_sha256": sha256_file(machinery),
        "calibration_split_manifest_path": str(split),
        "calibration_split_manifest_sha256": sha256_file(split),
        "state_selection_manifest_path": str(state_selection),
        "state_selection_manifest_sha256": sha256_file(state_selection),
        "differentiable_path_report_path": str(differentiable_path_report),
        "differentiable_path_report_sha256": sha256_file(differentiable_path_report),
        "subspace_basis_manifest_path": str(basis_manifest),
        "subspace_basis_manifest_sha256": sha256_file(basis_manifest),
        "subspace_diagnostics_path": str(diagnostics),
        "subspace_diagnostics_sha256": sha256_file(diagnostics),
        "donor_mapping_path": str(donor_mapping),
        "donor_mapping_sha256": sha256_file(donor_mapping),
        "donor_observation_manifest_path": str(donor_manifest),
        "donor_observation_manifest_sha256": sha256_file(donor_manifest),
        "donor_observation_root": str(donor_root),
        "round4c_summary_path": str(round4c_summary),
        "round4c_summary_sha256": sha256_file(round4c_summary),
    }
    payload["identity_sha256"] = sha256_json(payload)
    return payload


def _complete(
    root: Path,
    *,
    index: int,
    task_ids: tuple[int, ...],
    trials: int,
    expected_identity: Mapping[str, Any],
    split_by_trial: Mapping[tuple[int, int], str],
    donor_pairs: Mapping[tuple[int, int], tuple[int, int]],
) -> bool:
    condition = _expected_condition(index)
    metadata_path = root / condition.name / "run_metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = _read(metadata_path)
    config = metadata.get("condition_config", {})
    identity_fields = {
        key: metadata.get(key)
        for key in expected_identity
        if key not in {"protocol", "identity_sha256"}
    }
    if not (
        metadata.get("status") == "completed"
        and metadata.get("condition_protocol") == SALVAGE_A_PROTOCOL
        and metadata.get("diagnosis_condition") == condition.name
        and metadata.get("task_suite") == TASK_SUITE
        and metadata.get("task_ids") == list(task_ids)
        and metadata.get("number_of_trials") == trials
        and metadata.get("seed") == SEED
        and metadata.get("action_horizon") == ACTION_HORIZON
        and metadata.get("number_of_inference_steps") == INFERENCE_STEPS
        and metadata.get("replan_steps") == REPLAN_STEPS
        and identity_fields
        == {
            key: value
            for key, value in expected_identity.items()
            if key not in {"protocol", "identity_sha256"}
        }
        and isinstance(config, dict)
        and config.get("condition_name") == condition.name
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
    try:
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
            _validate_assignments(
                result.get("donor_assignments", []),
                condition=condition,
                task_id=task_id,
                trials=trials,
                split_by_trial=split_by_trial,
                donor_pairs=donor_pairs,
            )
            observed_tasks.add(task_id)
    except (KeyError, TypeError, ValueError):
        return False
    return observed_tasks == set(task_ids)


def _validate_smoke(root: Path, *, index: int, basis_sha256: str) -> None:
    condition = _expected_condition(index)
    directory = root / condition.name
    metadata = _read(directory / "run_metadata.json")
    if metadata.get("subspace_basis_manifest_sha256") != basis_sha256:
        raise ValueError(f"Smoke basis hash mismatch for {condition.name}.")
    traces = sorted((directory / TASK_SUITE / "action_traces").glob("task0_trial*.jsonl"))
    if len(traces) != SMOKE_TRIALS:
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
                raise ValueError(f"Invalid Salvage A smoke action trace: {path}.")
            trace_rows += 1
    if trace_rows == 0:
        raise ValueError(f"Smoke produced no action queries for {condition.name}.")


def _gate(
    path: Path | None,
    *,
    mode: str,
    wave: int,
    shared_identity_sha256: str,
) -> None:
    if path is None:
        raise ValueError(f"Salvage A {mode} wave {wave} gate is required.")
    payload = _read(path.resolve())
    if not (
        payload.get("protocol") == SALVAGE_A_PROTOCOL
        and payload.get("mode") == mode
        and payload.get("wave") == wave
        and payload.get("all_succeeded") is True
        and payload.get("shared_input_identity_sha256") == shared_identity_sha256
        and payload.get("git_commit_hash") == git_commit(PROJECT_ROOT)
    ):
        raise ValueError(f"Salvage A prerequisite gate failed: {path}.")


def launch(args: argparse.Namespace) -> Path:
    validate_frozen_condition_matrix()
    preflight_path = args.preflight.resolve()
    machinery_path = args.machinery.resolve()
    split_path = args.split.resolve()
    state_selection_path = args.state_selection.resolve()
    differentiable_path = args.differentiable_path_report.resolve()
    basis_path = args.basis_manifest.resolve()
    diagnostics_path = args.diagnostics.resolve()
    donor_mapping_path = args.donor_mapping.resolve()
    donor_manifest_path = args.donor_manifest.resolve()
    round4c_summary_path = args.round4c_summary.resolve()
    shared_identity = _metadata_identity(
        preflight=preflight_path,
        machinery=machinery_path,
        split=split_path,
        state_selection=state_selection_path,
        differentiable_path_report=differentiable_path,
        basis_manifest=basis_path,
        diagnostics=diagnostics_path,
        donor_mapping=donor_mapping_path,
        donor_manifest=donor_manifest_path,
        donor_root=args.donor_root.resolve(),
        round4c_summary=round4c_summary_path,
    )
    if args.mode == "full":
        _gate(
            args.smoke_summary,
            mode="smoke",
            wave=args.wave,
            shared_identity_sha256=shared_identity["identity_sha256"],
        )
        if args.wave == 2:
            _gate(
                args.wave1_full_summary,
                mode="full",
                wave=1,
                shared_identity_sha256=shared_identity["identity_sha256"],
            )
    elif args.wave == 2:
        _gate(
            args.wave1_smoke_summary,
            mode="smoke",
            wave=1,
            shared_identity_sha256=shared_identity["identity_sha256"],
        )

    indices = WAVES[args.wave]
    task_ids = (0,) if args.mode == "smoke" else TASK_IDS
    trials = SMOKE_TRIALS if args.mode == "smoke" else ONLINE_TRIALS_PER_TASK
    gpu_ids = tuple(args.gpu_ids)
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError("Every Salvage A wave requires four distinct GPU IDs.")
    inventory = _gpu_inventory()
    available = {int(item["index"]) for item in inventory}
    if set(gpu_ids) - available:
        raise ValueError("Requested Salvage A GPU is unavailable.")
    preflight = _read(preflight_path)
    machinery = _read(machinery_path)
    if not (
        preflight.get("protocol") == SALVAGE_A_PROTOCOL
        and preflight.get("status") == "compatible"
        and machinery.get("protocol") == SALVAGE_A_PROTOCOL
        and machinery.get("passed") is True
    ):
        raise ValueError("Salvage A preflight/machinery gate is incomplete.")
    state = preflight["state_bank"]
    split_by_trial = _split_trials(_read(split_path))
    donor_pairs = _donor_pairs(_read(donor_mapping_path))
    for recipient, donor in donor_pairs.items():
        if recipient[0] != donor[0] or split_by_trial[recipient] != split_by_trial[donor]:
            raise ValueError(f"Donor mapping crosses task or split: {recipient} -> {donor}.")

    paths = {
        "preflight_report": preflight_path,
        "machinery_report": machinery_path,
        "calibration_split_manifest": split_path,
        "state_selection_manifest": state_selection_path,
        "differentiable_path_report": differentiable_path,
        "subspace_basis_manifest": basis_path,
        "subspace_diagnostics": diagnostics_path,
        "round4c_summary": round4c_summary_path,
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    conditions = build_salvage_a_conditions(30)
    children: list[tuple[str, subprocess.Popen[Any], Any, Path]] = []
    failure: str | None = None
    start = now_iso()
    with _launcher_lock(output_root.parent / f"{args.mode}_wave{args.wave}") as lock_fd:
        for slot, index in enumerate(indices):
            condition = conditions[index]
            if _complete(
                output_root,
                index=index,
                task_ids=task_ids,
                trials=trials,
                expected_identity=shared_identity,
                split_by_trial=split_by_trial,
                donor_pairs=donor_pairs,
            ):
                if args.mode == "smoke":
                    _validate_smoke(
                        output_root,
                        index=index,
                        basis_sha256=sha256_file(basis_path),
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
                f"seed={SEED}",
                "EVALUATION.device=cuda:0",
                "EVALUATION.text_encoder_device=null",
                f"EVALUATION.prompt_context_cache_path={state['prompt_context_cache_path']}",
                f"EVALUATION.task_suite_name={TASK_SUITE}",
                f"EVALUATION.task_ids={compact(task_ids)}",
                f"EVALUATION.num_trials={trials}",
                f"EVALUATION.action_horizon={ACTION_HORIZON}",
                f"EVALUATION.num_inference_steps={INFERENCE_STEPS}",
                f"EVALUATION.replan_steps={REPLAN_STEPS}",
                f"EVALUATION.output_dir={condition_root}",
                "EVALUATION.visualize_future_video=false",
                f"EVALUATION.dataset_stats_path={state['dataset_stats_path']}",
                "ASRE_DIAGNOSIS.enabled=true",
                "ASRE_DIAGNOSIS.mode=replace_video_kv",
                f"ASRE_DIAGNOSIS.protocol={SALVAGE_A_PROTOCOL}",
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
                f"ASRE_DIAGNOSIS.donor_mapping_path={donor_mapping_path}",
                f"ASRE_DIAGNOSIS.donor_mapping_sha256={sha256_file(donor_mapping_path)}",
                f"ASRE_DIAGNOSIS.donor_observation_manifest_path={donor_manifest_path}",
                f"ASRE_DIAGNOSIS.donor_observation_manifest_sha256={sha256_file(donor_manifest_path)}",
                f"ASRE_DIAGNOSIS.donor_observation_root={args.donor_root.resolve()}",
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
            output_root,
            index=index,
            task_ids=task_ids,
            trials=trials,
            expected_identity=shared_identity,
            split_by_trial=split_by_trial,
            donor_pairs=donor_pairs,
        )
        for index in indices
    }
    if failure is None and all(states.values()) and args.mode == "smoke":
        for index in indices:
            _validate_smoke(
                output_root,
                index=index,
                basis_sha256=sha256_file(basis_path),
            )
    summary = output_root / "launcher_summary.json"
    atomic_write_json(
        summary,
        {
            "artifact_type": "asre_salvage_a_online_wave_summary",
            "schema_version": 1,
            "protocol": SALVAGE_A_PROTOCOL,
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
            "shared_input_identity": shared_identity,
            "shared_input_identity_sha256": shared_identity["identity_sha256"],
            "condition_complete": states,
            "failure": failure,
            "start_timestamp": start,
            "end_timestamp": now_iso(),
            "no_ddp": True,
            "spare_gpus_used_for_unrelated_work": False,
            "later_stage_launched": False,
        },
    )
    if failure or not all(states.values()):
        raise RuntimeError(failure or "Salvage A online wave incomplete.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--wave", type=int, choices=(1, 2), required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--machinery", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--state-selection", type=Path, required=True)
    parser.add_argument("--differentiable-path-report", type=Path, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--round4c-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--wave1-smoke-summary", type=Path)
    parser.add_argument("--wave1-full-summary", type=Path)
    parser.add_argument("--launch-stagger-seconds", type=float, default=30.0)
    path = launch(parser.parse_args())
    print(f"Salvage A online wave complete: {path}")


if __name__ == "__main__":
    main()
