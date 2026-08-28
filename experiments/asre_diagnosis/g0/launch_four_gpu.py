"""Launch one G0 suite as four isolated, paired, non-DDP GPU workers."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (
    G0_PROTOCOL,
    DiagnosisCondition,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.g0.definitions import (
    CONDITIONS,
    NUM_LAYERS,
    NUM_STEPS_WAIT,
    ROUND3A_COMMIT,
    ROUND3A_TAG,
    ROUND3B_COMMIT,
    ROUND3B_TAG,
    SEED,
    SUITE_ORDER,
    TASK_CONFIG,
    RuntimeSpec,
    assert_output_scope,
    metadata_mismatches,
    runtime_for,
    validate_gpu_mapping,
)
from experiments.asre_diagnosis.round3b.donor import OnlineDonorBundle


ROOT_CONFIG = "launcher_config.json"
SUMMARY = "launcher_summary.json"
STATUS = "launcher_status.json"


class DirectorySafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Provenance:
    checkpoint: Path
    checkpoint_sha256: str
    dataset_stats: Path
    dataset_stats_sha256: str
    donor_mapping: Path
    donor_mapping_sha256: str
    donor_manifest: Path
    donor_manifest_sha256: str
    donor_root: Path
    preflight_report: Path
    preflight_report_sha256: str
    machinery_report: Path
    machinery_report_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class Inspection:
    state: str
    completed_task_ids: tuple[int, ...]
    detail: str


@dataclass
class LiveProcess:
    condition_index: int
    physical_gpu: int
    condition: DiagnosisCondition
    process: subprocess.Popen[Any]
    log_handle: Any
    log_path: Path
    status_path: Path
    status: dict[str, Any]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {resolved}.")
    return resolved


def _load_provenance(args: argparse.Namespace, suite: str) -> Provenance:
    worktree = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--",
            ".",
            ":(exclude)asre_results/g0_cross_suite/**",
        ],
        cwd=project_root,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if worktree:
        raise RuntimeError(
            "Formal G0 launch requires committed source and an otherwise clean worktree:\n"
            + worktree
        )
    checkpoint = _resolve_file(args.checkpoint, "checkpoint")
    dataset_stats = _resolve_file(args.dataset_stats, "dataset statistics")
    mapping = _resolve_file(args.donor_mapping, "donor mapping")
    manifest = _resolve_file(args.donor_manifest, "donor observation manifest")
    donor_root = args.donor_root.expanduser().resolve()
    if not donor_root.is_dir():
        raise FileNotFoundError(f"Donor root is unavailable: {donor_root}.")
    preflight_path = _resolve_file(args.preflight_report, "G0 preflight report")
    machinery_path = _resolve_file(args.machinery_report, "G0 machinery report")
    preflight = _read_json(preflight_path, "G0 preflight report")
    machinery = _read_json(machinery_path, "G0 machinery report")
    if preflight.get("status") != "compatible" or preflight.get("git", {}).get(
        "allow_dirty"
    ):
        raise ValueError("Formal G0 launch requires a clean, compatible preflight report.")
    if machinery.get("status") != "passed" or machinery.get("passed") is not True:
        raise ValueError("G0 machinery report is absent or failed.")
    if preflight.get("git", {}).get("head") != git_commit(project_root):
        raise ValueError("G0 preflight Git commit differs from the current checkout.")
    if machinery.get("git_commit_hash") != git_commit(project_root):
        raise ValueError("G0 machinery report Git commit differs from the current checkout.")
    expected_checkpoint = preflight.get("checkpoint", {})
    expected_stats = preflight.get("dataset_statistics", {})
    if expected_checkpoint.get("path") != str(checkpoint) or expected_stats.get(
        "path"
    ) != str(dataset_stats):
        raise ValueError("Launcher checkpoint/statistics paths differ from preflight.")
    checkpoint_digest = str(expected_checkpoint.get("sha256", ""))
    stats_digest = str(expected_stats.get("sha256", ""))
    if machinery.get("checkpoint_sha256") != checkpoint_digest or machinery.get(
        "dataset_stats_sha256"
    ) != stats_digest:
        raise ValueError("Machinery and preflight artifact hashes disagree.")
    bundle = OnlineDonorBundle.load(
        mapping_path=mapping,
        observation_manifest_path=manifest,
        observation_root=donor_root,
    )
    expected_bundle = {
        "task_suite": suite,
        "seed": SEED,
        "num_tasks": 10,
        "num_trials": 10,
    }
    mismatch = metadata_mismatches(bundle.mapping_payload, expected_bundle)
    if mismatch:
        raise ValueError(f"G0 donor bundle mismatch: {mismatch}.")
    return Provenance(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_digest,
        dataset_stats=dataset_stats,
        dataset_stats_sha256=stats_digest,
        donor_mapping=mapping,
        donor_mapping_sha256=bundle.mapping_sha256,
        donor_manifest=manifest,
        donor_manifest_sha256=bundle.observation_manifest_sha256,
        donor_root=donor_root,
        preflight_report=preflight_path,
        preflight_report_sha256=sha256_file(preflight_path),
        machinery_report=machinery_path,
        machinery_report_sha256=sha256_file(machinery_path),
    )


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot query physical GPUs: {exc}") from exc
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise RuntimeError(f"Unexpected nvidia-smi row: {line!r}.")
        rows.append(
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
    return rows


def _select_gpus(inventory: Sequence[Mapping[str, Any]], ids: Sequence[int]) -> list[dict[str, Any]]:
    by_id = {int(row["index"]): dict(row) for row in inventory}
    missing = [gpu_id for gpu_id in ids if gpu_id not in by_id]
    if missing:
        raise ValueError(f"Requested GPUs are unavailable: {missing}.")
    selected = [by_id[gpu_id] for gpu_id in ids]
    insufficient = {
        int(record["index"]): int(record["memory_free_mib_at_launch"])
        for record in selected
        if int(record["memory_free_mib_at_launch"]) < 22000
    }
    if insufficient:
        raise RuntimeError(
            "Selected G0 GPUs do not have the required 22000 MiB free at launch: "
            f"{insufficient}. Choose the four dedicated cards explicitly."
        )
    return selected


def _expected_metadata(
    condition: DiagnosisCondition, runtime: RuntimeSpec, provenance: Provenance
) -> dict[str, Any]:
    return {
        "git_commit_hash": git_commit(project_root),
        "checkpoint_path": str(provenance.checkpoint),
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "dataset_stats_path": str(provenance.dataset_stats),
        "dataset_stats_sha256": provenance.dataset_stats_sha256,
        "diagnosis_condition": condition.name,
        "condition_protocol": G0_PROTOCOL,
        "enabled_video_retrieval_layers": list(
            condition.enabled_video_retrieval_layers(NUM_LAYERS)
        ),
        "disabled_video_layers": list(condition.disabled_video_layers),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "num_model_layers": NUM_LAYERS,
        "task_suite": runtime.suite,
        "task_ids": list(runtime.task_ids),
        "seed": SEED,
        "number_of_trials": runtime.num_trials,
        "action_horizon": runtime.action_horizon,
        "number_of_inference_steps": runtime.inference_steps,
        "replan_steps": runtime.replan_steps,
        "compile_action_infer": True,
        "binarize_gripper": True,
        "text_conditioning_source": "model_text_encoder",
        "prompt_context_strategy": "suite_gpu_prewarm_then_text_encoder_release",
        "prompt_context_count": 10,
        "donor_mapping_path": str(provenance.donor_mapping),
        "donor_mapping_sha256": provenance.donor_mapping_sha256,
        "donor_observation_manifest_path": str(provenance.donor_manifest),
        "donor_observation_manifest_sha256": provenance.donor_manifest_sha256,
        "donor_observation_root": str(provenance.donor_root),
        "preflight_report_path": str(provenance.preflight_report),
        "preflight_report_sha256": provenance.preflight_report_sha256,
        "machinery_report_path": str(provenance.machinery_report),
        "machinery_report_sha256": provenance.machinery_report_sha256,
        "round3a_parent_tag": ROUND3A_TAG,
        "round3a_parent_commit": ROUND3A_COMMIT,
        "round3b_parent_tag": ROUND3B_TAG,
        "round3b_parent_commit": ROUND3B_COMMIT,
    }


def _validate_action_array(value: Any, path: Path, field: str) -> None:
    if not isinstance(value, list) or len(value) != 32:
        raise DirectorySafetyError(f"{field} in {path} must have shape [32,7].")
    for row in value:
        if not isinstance(row, list) or len(row) != 7 or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in row
        ):
            raise DirectorySafetyError(f"{field} in {path} contains malformed actions.")


def _validate_traces(
    condition_dir: Path,
    condition: DiagnosisCondition,
    runtime: RuntimeSpec,
    task_id: int,
) -> None:
    for trial in range(runtime.num_trials):
        path = (
            condition_dir
            / runtime.suite
            / "action_traces"
            / f"task{task_id}_trial{trial}.jsonl"
        )
        if not path.is_file():
            raise DirectorySafetyError(f"Missing G0 action trace: {path}.")
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise DirectorySafetyError(f"Empty G0 action trace: {path}.")
        for replan_id, record in enumerate(records):
            expected = {
                "task_suite": runtime.suite,
                "task_id": task_id,
                "diagnosis_condition": condition.name,
                "episode_id": trial,
                "replan_id": replan_id,
                "environment_step": NUM_STEPS_WAIT + replan_id * runtime.replan_steps,
                "action_inference_seed": SEED,
                "replacement_video_layers": list(condition.replacement_video_layers),
            }
            mismatch = metadata_mismatches(record, expected)
            if mismatch:
                raise DirectorySafetyError(f"Trace identity mismatch in {path}: {mismatch}.")
            if condition.name == "late_wrong_scene_15_29":
                if int(record.get("donor_task_id", -1)) != task_id or int(
                    record.get("donor_trial", -1)
                ) != (trial + 1) % 10:
                    raise DirectorySafetyError(f"Wrong donor identity in {path}.")
                if replan_id == 0 and record.get(
                    "current_input_image_sha256"
                ) == record.get("donor_image_sha256"):
                    raise DirectorySafetyError(f"Recipient/donor image collision in {path}.")
            _validate_action_array(record.get("raw_action"), path, "raw_action")
            _validate_action_array(record.get("executed_action"), path, "executed_action")


def _validate_result(
    path: Path, condition: DiagnosisCondition, runtime: RuntimeSpec
) -> int:
    result = _read_json(path, "task result")
    expected = {
        "task_suite": runtime.suite,
        "diagnosis_condition": condition.name,
        "condition_protocol": G0_PROTOCOL,
        "enabled_video_retrieval_layers": list(
            condition.enabled_video_retrieval_layers(NUM_LAYERS)
        ),
        "disabled_video_layers": list(condition.disabled_video_layers),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "total_episodes": runtime.num_trials,
    }
    mismatch = metadata_mismatches(result, expected)
    if mismatch:
        raise DirectorySafetyError(f"Task result mismatch in {path}: {mismatch}.")
    task_id = int(result.get("task_id", -1))
    successes = [int(item) for item in result.get("success_episodes", [])]
    failures = [int(item) for item in result.get("failure_episodes", [])]
    if (
        task_id not in runtime.task_ids
        or set(successes) & set(failures)
        or sorted(successes + failures) != list(range(runtime.num_trials))
        or int(result.get("successes", -1)) != len(successes)
    ):
        raise DirectorySafetyError(f"Unpaired/incomplete task outcomes in {path}.")
    return task_id


def _inspect_condition(
    condition_dir: Path,
    condition: DiagnosisCondition,
    runtime: RuntimeSpec,
    provenance: Provenance,
    physical_gpu: int,
) -> Inspection:
    if not condition_dir.exists() or not any(condition_dir.iterdir()):
        return Inspection("missing", (), "directory is absent or empty")
    status_path = condition_dir / STATUS
    if status_path.is_file():
        status = _read_json(status_path, "launcher status")
        mismatch = metadata_mismatches(
            status,
            {
                "protocol": G0_PROTOCOL,
                "suite": runtime.suite,
                "mode": runtime.mode,
                "condition": condition.name,
                "physical_gpu": physical_gpu,
            },
        )
        if mismatch:
            raise DirectorySafetyError(f"Incompatible launcher status: {mismatch}.")
        attempts = status.get("attempts", [])
        if status.get("state") in {"launching", "running"} and isinstance(
            attempts, list
        ) and attempts:
            pid = attempts[-1].get("pid")
            if isinstance(pid, int) and pid > 0:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    raise DirectorySafetyError(
                        f"Previous condition child may still be live (pid={pid})."
                    ) from exc
                else:
                    raise DirectorySafetyError(
                        f"Previous condition child is still live (pid={pid})."
                    )
    metadata_path = condition_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return Inspection("partial", (), "run metadata is not written yet")
    metadata = _read_json(metadata_path, "condition metadata")
    mismatch = metadata_mismatches(
        metadata, _expected_metadata(condition, runtime, provenance)
    )
    if mismatch:
        raise DirectorySafetyError(f"Incompatible condition metadata: {mismatch}.")
    prompt_digest = metadata.get("prompt_context_manifest_sha256")
    if (
        not isinstance(prompt_digest, str)
        or len(prompt_digest) != 64
        or any(character not in "0123456789abcdef" for character in prompt_digest)
    ):
        raise DirectorySafetyError(
            f"Invalid prompt-context manifest SHA256 in {metadata_path}."
        )
    completed: dict[int, Path] = {}
    for path in sorted((condition_dir / runtime.suite).glob("gpu*_task*_results.json")):
        task_id = _validate_result(path, condition, runtime)
        if task_id in completed:
            raise DirectorySafetyError(f"Duplicate task result for task {task_id}.")
        completed[task_id] = path
        _validate_traces(condition_dir, condition, runtime, task_id)
    completed_ids = tuple(sorted(completed))
    all_complete = set(completed_ids) == set(runtime.task_ids)
    status = str(metadata.get("status", ""))
    if status == "completed" and all_complete:
        return Inspection("complete", completed_ids, "all task/trial outputs validated")
    if status == "completed" and not all_complete:
        raise DirectorySafetyError("Metadata claims completion but task outputs are missing.")
    return Inspection("partial", completed_ids, f"metadata={status!r}")


def _condition_command(
    *,
    python: Path,
    condition_index: int,
    condition: DiagnosisCondition,
    output: Path,
    runtime: RuntimeSpec,
    provenance: Provenance,
) -> list[str]:
    compact = lambda value: json.dumps(value, separators=(",", ":"))
    return [
        str(python),
        str(project_root / "experiments/libero/eval_libero_single.py"),
        f"task={TASK_CONFIG}",
        f"ckpt={provenance.checkpoint}",
        "model.load_text_encoder=true",
        "gpu_id=0",
        f"seed={SEED}",
        "EVALUATION.device=cuda:0",
        "EVALUATION.text_encoder_device=cuda:0",
        "EVALUATION.prompt_context_cache_path=null",
        "EVALUATION.prewarm_suite_prompts_and_release_text_encoder=true",
        f"EVALUATION.task_suite_name={runtime.suite}",
        f"EVALUATION.task_ids={compact(list(runtime.task_ids))}",
        f"EVALUATION.num_trials={runtime.num_trials}",
        f"EVALUATION.action_horizon={runtime.action_horizon}",
        f"EVALUATION.num_inference_steps={runtime.inference_steps}",
        f"EVALUATION.replan_steps={runtime.replan_steps}",
        f"EVALUATION.output_dir={output}",
        "EVALUATION.visualize_future_video=false",
        f"EVALUATION.dataset_stats_path={provenance.dataset_stats}",
        "ASRE_DIAGNOSIS.enabled=true",
        "ASRE_DIAGNOSIS.mode=replace_video_kv",
        f"ASRE_DIAGNOSIS.protocol={G0_PROTOCOL}",
        f"ASRE_DIAGNOSIS.condition_index={condition_index}",
        f"ASRE_DIAGNOSIS.condition_name={condition.name}",
        "ASRE_DIAGNOSIS.enabled_video_retrieval_layers="
        + compact(list(condition.enabled_video_retrieval_layers(NUM_LAYERS))),
        "ASRE_DIAGNOSIS.disabled_video_layers="
        + compact(list(condition.disabled_video_layers)),
        "ASRE_DIAGNOSIS.replacement_video_layers="
        + compact(list(condition.replacement_video_layers)),
        "ASRE_DIAGNOSIS.save_rollout_video=false",
        f"ASRE_DIAGNOSIS.checkpoint_sha256={provenance.checkpoint_sha256}",
        f"ASRE_DIAGNOSIS.dataset_stats_sha256={provenance.dataset_stats_sha256}",
        f"ASRE_DIAGNOSIS.donor_mapping_path={provenance.donor_mapping}",
        f"ASRE_DIAGNOSIS.donor_mapping_sha256={provenance.donor_mapping_sha256}",
        f"ASRE_DIAGNOSIS.donor_observation_manifest_path={provenance.donor_manifest}",
        f"ASRE_DIAGNOSIS.donor_observation_manifest_sha256={provenance.donor_manifest_sha256}",
        f"ASRE_DIAGNOSIS.donor_observation_root={provenance.donor_root}",
        f"ASRE_DIAGNOSIS.preflight_report_path={provenance.preflight_report}",
        f"ASRE_DIAGNOSIS.preflight_report_sha256={provenance.preflight_report_sha256}",
        f"ASRE_DIAGNOSIS.machinery_report_path={provenance.machinery_report}",
        f"ASRE_DIAGNOSIS.machinery_report_sha256={provenance.machinery_report_sha256}",
        f"ASRE_DIAGNOSIS.round3a_parent_tag={ROUND3A_TAG}",
        f"ASRE_DIAGNOSIS.round3a_parent_commit={ROUND3A_COMMIT}",
        f"ASRE_DIAGNOSIS.round3b_parent_tag={ROUND3B_TAG}",
        f"ASRE_DIAGNOSIS.round3b_parent_commit={ROUND3B_COMMIT}",
    ]


def _child_environment(physical_gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    default_numba_cache = Path("/tmp/fastwam-g0-numba-cache")
    default_matplotlib_cache = Path("/tmp/fastwam-g0-matplotlib-cache")
    default_numba_cache.mkdir(parents=True, exist_ok=True)
    default_matplotlib_cache.mkdir(parents=True, exist_ok=True)
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "LIBERO_WORKER_MODE",
    ):
        environment.pop(key, None)
    libero_root = Path(
        environment.get("LIBERO_ROOT", str(project_root.parent / "LIBERO"))
    ).resolve()
    entries = [str(project_root / "src"), str(project_root), str(libero_root)]
    if environment.get("PYTHONPATH"):
        entries.append(environment["PYTHONPATH"])
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(physical_gpu),
            "PYTHONPATH": os.pathsep.join(entries),
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "NUMBA_CACHE_DIR": environment.get(
                "NUMBA_CACHE_DIR", str(default_numba_cache)
            ),
            "MPLCONFIGDIR": environment.get(
                "MPLCONFIGDIR", str(default_matplotlib_cache)
            ),
        }
    )
    return environment


@contextmanager
def _launcher_lock(output_root: Path) -> Iterator[int]:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    path = output_root.parent / f".{output_root.name}.launcher.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DirectorySafetyError(f"Another launcher owns {path}.") from exc
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "started_at": now_iso()}, handle)
        handle.flush()
        os.fsync(handle.fileno())
        yield handle.fileno()
    finally:
        handle.close()


def _prepare_root(output_root: Path, identity: Mapping[str, Any]) -> None:
    config = output_root / ROOT_CONFIG
    if output_root.exists() and not output_root.is_dir():
        raise DirectorySafetyError(f"Output root is not a directory: {output_root}.")
    if output_root.exists() and any(output_root.iterdir()):
        if not config.is_file():
            raise DirectorySafetyError(f"Non-empty output lacks {ROOT_CONFIG}: {output_root}.")
        existing = _read_json(config, "launcher config")
        if existing != dict(identity):
            raise DirectorySafetyError("Existing launcher config belongs to another run.")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config, identity)


def _launch(
    *,
    output_root: Path,
    python: Path,
    condition_index: int,
    physical_gpu: int,
    condition: DiagnosisCondition,
    runtime: RuntimeSpec,
    provenance: Provenance,
    lock_fd: int,
) -> LiveProcess:
    condition_dir = output_root / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    status_path = condition_dir / STATUS
    previous = _read_json(status_path, "launcher status") if status_path.exists() else {}
    attempts = list(previous.get("attempts", []))
    attempt = len(attempts) + 1
    logs = output_root / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"{condition.name}.attempt{attempt:02d}.log"
    command = _condition_command(
        python=python,
        condition_index=condition_index,
        condition=condition,
        output=condition_dir,
        runtime=runtime,
        provenance=provenance,
    )
    attempt_record = {
        "attempt": attempt,
        "started_at": now_iso(),
        "ended_at": None,
        "pid": None,
        "exit_status": None,
        "state": "launching",
        "log_path": str(log_path),
        "command": shlex.join(command),
    }
    status = {
        "schema_version": 1,
        "protocol": G0_PROTOCOL,
        "suite": runtime.suite,
        "mode": runtime.mode,
        "condition": condition.name,
        "condition_index": condition_index,
        "physical_gpu": physical_gpu,
        "logical_device": "cuda:0",
        "ddp": False,
        "attempts": attempts + [attempt_record],
        "state": "launching",
    }
    atomic_write_json(status_path, status)
    log_handle = log_path.open("x", encoding="utf-8")
    log_handle.write(shlex.join(command) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=_child_environment(physical_gpu),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        pass_fds=(lock_fd,),
    )
    attempt_record["pid"] = process.pid
    attempt_record["state"] = "running"
    status["attempts"][-1] = attempt_record
    status["state"] = "running"
    atomic_write_json(status_path, status)
    return LiveProcess(
        condition_index,
        physical_gpu,
        condition,
        process,
        log_handle,
        log_path,
        status_path,
        status,
    )


def _finish(live: LiveProcess, exit_status: int, state: str) -> None:
    if not live.log_handle.closed:
        live.log_handle.close()
    attempt = live.status["attempts"][-1]
    attempt.update(
        {"ended_at": now_iso(), "exit_status": int(exit_status), "state": state}
    )
    live.status["attempts"][-1] = attempt
    live.status["state"] = state
    atomic_write_json(live.status_path, live.status)


def _terminate(live: LiveProcess) -> None:
    if live.process.poll() is None:
        try:
            os.killpg(live.process.pid, signal.SIGTERM)
            live.process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(live.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            live.process.wait()
    _finish(live, int(live.process.returncode or -1), "terminated_due_to_peer_failure")


def _validate_smoke(
    path: Path | None,
    suite: str,
    gpu_ids: Sequence[int],
    provenance: Provenance,
) -> None:
    if path is None:
        raise ValueError("Full G0 launch requires --smoke-summary.")
    summary = _read_json(path.expanduser().resolve(), "smoke summary")
    expected = {
        "protocol": G0_PROTOCOL,
        "suite": suite,
        "mode": "smoke",
        "all_succeeded": True,
        "task_ids": [0],
        "num_trials": 2,
        "gpu_ids": list(gpu_ids),
    }
    mismatch = metadata_mismatches(summary, expected)
    if mismatch:
        raise ValueError(f"Smoke summary is incompatible: {mismatch}.")
    config_path = Path(str(summary.get("output_root", ""))).resolve() / ROOT_CONFIG
    if not config_path.is_file() or sha256_file(config_path) != summary.get(
        "launcher_config_sha256"
    ):
        raise ValueError("Smoke launcher config is unavailable or changed.")
    config = _read_json(config_path, "smoke launcher config")
    config_expected = {
        "protocol": G0_PROTOCOL,
        "suite": suite,
        "mode": "smoke",
        "git_commit_hash": git_commit(project_root),
        "gpu_ids": list(gpu_ids),
        "provenance": provenance.to_dict(),
    }
    config_mismatch = metadata_mismatches(config, config_expected)
    if config_mismatch:
        raise ValueError(f"Smoke launcher provenance is incompatible: {config_mismatch}.")


def launch_suite(
    *,
    suite: str,
    mode: str,
    output_root: Path,
    python: Path,
    gpu_ids: Sequence[int],
    provenance: Provenance,
    smoke_summary: Path | None = None,
    launch_stagger_seconds: float = 0.0,
) -> dict[str, Any]:
    runtime = runtime_for(suite, mode)
    ids = validate_gpu_mapping(gpu_ids)
    output_root = output_root.expanduser().resolve()
    assert_output_scope(output_root, project_root)
    python = _resolve_file(python, "Python executable")
    if mode == "full":
        _validate_smoke(smoke_summary, suite, ids, provenance)
    inventory = _gpu_inventory()
    selected = _select_gpus(inventory, ids)
    identity = {
        "artifact_type": "asre_g0_launcher_config",
        "schema_version": 1,
        "protocol": G0_PROTOCOL,
        "suite": suite,
        "mode": mode,
        "git_commit_hash": git_commit(project_root),
        "task_ids": list(runtime.task_ids),
        "num_trials": runtime.num_trials,
        "seed": SEED,
        "gpu_ids": list(ids),
        "gpu_inventory": [
            {
                key: value
                for key, value in record.items()
                if key != "memory_free_mib_at_launch"
            }
            for record in selected
        ],
        "python": str(python),
        "launch_stagger_seconds": float(launch_stagger_seconds),
        "provenance": provenance.to_dict(),
        "conditions": [
            {
                "condition_index": index,
                "condition": condition.name,
                "physical_gpu": ids[index],
                "enabled_video_retrieval_layers": list(
                    condition.enabled_video_retrieval_layers(NUM_LAYERS)
                ),
                "disabled_video_layers": list(condition.disabled_video_layers),
                "replacement_video_layers": list(condition.replacement_video_layers),
            }
            for index, condition in enumerate(CONDITIONS)
        ],
    }
    identity["identity_sha256"] = sha256_json(identity)
    started = now_iso()
    failure: str | None = None
    interrupted = False
    with _launcher_lock(output_root) as lock_fd:
        _prepare_root(output_root, identity)
        inspections = {
            condition.name: _inspect_condition(
                output_root / condition.name,
                condition,
                runtime,
                provenance,
                ids[index],
            )
            for index, condition in enumerate(CONDITIONS)
        }
        live: list[LiveProcess] = []
        try:
            for index, condition in enumerate(CONDITIONS):
                if inspections[condition.name].state == "complete":
                    print(f"Skipping verified-complete {suite}/{condition.name}.")
                    continue
                live.append(
                    _launch(
                        output_root=output_root,
                        python=python,
                        condition_index=index,
                        physical_gpu=ids[index],
                        condition=condition,
                        runtime=runtime,
                        provenance=provenance,
                        lock_fd=lock_fd,
                    )
                )
                if launch_stagger_seconds > 0 and index < len(CONDITIONS) - 1:
                    time.sleep(launch_stagger_seconds)
            while live:
                for child in list(live):
                    returncode = child.process.poll()
                    if returncode is None:
                        continue
                    live.remove(child)
                    if returncode != 0:
                        _finish(child, returncode, "failed")
                        failure = (
                            f"{suite}/{child.condition.name} exited {returncode}; "
                            f"see {child.log_path}."
                        )
                        break
                    inspected = _inspect_condition(
                        output_root / child.condition.name,
                        child.condition,
                        runtime,
                        provenance,
                        child.physical_gpu,
                    )
                    if inspected.state != "complete":
                        _finish(child, returncode, "failed_validation")
                        failure = f"{child.condition.name} exited 0 but is {inspected.state}."
                        break
                    _finish(child, returncode, "completed")
                if failure:
                    break
                if live:
                    time.sleep(5)
        except KeyboardInterrupt:
            interrupted = True
            failure = "launcher interrupted"
        finally:
            if failure:
                for child in live:
                    _terminate(child)

        states: dict[str, Any] = {}
        all_complete = True
        for condition in CONDITIONS:
            try:
                inspected = _inspect_condition(
                    output_root / condition.name,
                    condition,
                    runtime,
                    provenance,
                    ids[CONDITIONS.index(condition)],
                )
                states[condition.name] = {
                    "state": inspected.state,
                    "completed_task_ids": list(inspected.completed_task_ids),
                    "detail": inspected.detail,
                }
                all_complete &= inspected.state == "complete"
            except Exception as exc:
                states[condition.name] = {"state": "invalid", "detail": repr(exc)}
                all_complete = False
        prompt_context_digests: dict[str, Any] = {}
        shared_prompt_context_digest = None
        if all_complete:
            prompt_context_digests = {
                condition.name: _read_json(
                    output_root / condition.name / "run_metadata.json",
                    "condition metadata",
                ).get("prompt_context_manifest_sha256")
                for condition in CONDITIONS
            }
            unique_prompt_digests = set(prompt_context_digests.values())
            if len(prompt_context_digests) != len(CONDITIONS) or len(
                unique_prompt_digests
            ) != 1:
                all_complete = False
                failure = (
                    "G0 conditions produced different suite prompt contexts: "
                    f"{prompt_context_digests}."
                )
            else:
                shared_prompt_context_digest = next(iter(unique_prompt_digests))
        summary = {
            "artifact_type": "asre_g0_launcher_summary",
            "schema_version": 1,
            "protocol": G0_PROTOCOL,
            "suite": suite,
            "mode": mode,
            "started_at": started,
            "ended_at": now_iso(),
            "git_commit_hash": git_commit(project_root),
            "output_root": str(output_root),
            "task_ids": list(runtime.task_ids),
            "num_trials": runtime.num_trials,
            "seed": SEED,
            "gpu_ids": list(ids),
            "condition_states": states,
            "all_succeeded": bool(all_complete and failure is None),
            "interrupted": interrupted,
            "failure": failure,
            "prompt_context_manifest_sha256": shared_prompt_context_digest,
            "launcher_config_sha256": sha256_file(output_root / ROOT_CONFIG),
        }
        atomic_write_json(output_root / SUMMARY, summary)
    if not summary["all_succeeded"]:
        raise RuntimeError(f"G0 suite launch failed: {summary['failure'] or states}.")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITE_ORDER, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--machinery-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not math.isfinite(args.launch_stagger_seconds) or args.launch_stagger_seconds < 0:
        raise ValueError("--launch-stagger-seconds must be finite and nonnegative.")
    provenance = _load_provenance(args, args.suite)
    summary = launch_suite(
        suite=args.suite,
        mode=args.mode,
        output_root=args.output_root,
        python=args.python,
        gpu_ids=args.gpu_ids,
        provenance=provenance,
        smoke_summary=args.smoke_summary,
        launch_stagger_seconds=args.launch_stagger_seconds,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
