"""Launch the frozen ASRE Round-3B three-arm control on three isolated GPUs.

The launcher is fail-closed around provenance and output reuse.  It never uses
DDP: each condition owns one explicitly recorded physical GPU and sees it as
``cuda:0``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND3B_PROTOCOL,
    DiagnosisCondition,
    atomic_write_json,
    build_round3b_conditions,
    git_commit,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.round3a.launch_three_gpu import (  # noqa: E402
    ConditionInspection,
    DirectorySafetyError,
    RuntimeSpec,
    _child_environment,
    _exclusive_launcher_lock,
    _gpu_inventory,
    _metadata_mismatches,
    _read_json,
    _recorded_live_pid,
    _resolve_python,
    _validate_action_array,
)
from experiments.asre_diagnosis.round3b.preflight import (  # noqa: E402
    ROUND3A_FROZEN_COMMIT,
    ROUND3A_RUN_COMMIT,
    ROUND3A_TAG,
    _validate_donors,
)


NUM_LAYERS = 30
TASK_SUITE = "libero_spatial"
SEED = 42
FULL_TASK_IDS = tuple(range(10))
SMOKE_TASK_IDS = (0,)
FULL_TRIALS = 10
SMOKE_TRIALS = 2
ACTION_HORIZON = 32
REPLAN_STEPS = 10
INFERENCE_STEPS = 10
CONDITION_INDICES = (0, 1, 2)
DEFAULT_GPU_IDS = CONDITION_INDICES
ROOT_CONFIG_NAME = "launcher_config.json"
SUMMARY_NAME = "launcher_summary.json"
STATUS_NAME = "launcher_status.json"


@dataclass(frozen=True)
class Provenance:
    checkpoint_path: Path
    checkpoint_sha256: str
    dataset_stats_path: Path
    dataset_stats_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    valid_manifest_path: Path
    valid_manifest_sha256: str
    prompt_context_cache_path: Path
    prompt_context_cache_sha256: str
    donor_mapping_path: Path
    donor_mapping_sha256: str
    donor_observation_manifest_path: Path
    donor_observation_manifest_sha256: str
    donor_observation_root: Path
    preflight_report_path: Path
    preflight_report_sha256: str
    self_replacement_report_path: Path
    self_replacement_report_sha256: str
    round3a_tag: str
    round3a_frozen_commit: str
    round3a_run_commit: str
    valid_sample_count: int

    def identity_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch ASRE Round-3B matched-shape K/V controls without DDP."
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-observation-manifest", type=Path, required=True)
    parser.add_argument("--donor-observation-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--self-replacement-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument(
        "--task-config",
        choices=("libero_uncond_2cam224_1e-4",),
        default="libero_uncond_2cam224_1e-4",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpu-ids",
        nargs=3,
        type=int,
        default=DEFAULT_GPU_IDS,
        metavar=("GPU_FOR_CORRECT", "GPU_FOR_WRONG", "GPU_FOR_NO_VIDEO"),
        help=(
            "Three distinct physical GPU indices in condition order; defaults to "
            "0 1 2. Each child still sees only logical cuda:0."
        ),
    )
    return parser.parse_args()


def _validate_gpu_ids(values: Sequence[int]) -> tuple[int, int, int]:
    gpu_ids = tuple(int(value) for value in values)
    if len(gpu_ids) != len(CONDITION_INDICES):
        raise ValueError(f"Round-3B requires exactly three GPU IDs, got {gpu_ids}.")
    if any(value < 0 for value in gpu_ids):
        raise ValueError(f"GPU IDs must be nonnegative, got {gpu_ids}.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Round-3B requires three distinct physical GPUs, got {gpu_ids}.")
    return gpu_ids


def _resolve_path(value: str | Path, *, label: str, file: bool = True) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()
    valid = path.is_file() if file else path.is_dir()
    if not valid:
        kind = "file" if file else "directory"
        raise FileNotFoundError(f"{label} {kind} is unavailable: {path}")
    return path


def _load_provenance(args: argparse.Namespace) -> Provenance:
    valid_path = _resolve_path(args.valid_manifest, label="QC valid manifest")
    valid = _read_json(valid_path, label="QC valid manifest")
    base = valid_path.parent

    def declared_path(key: str) -> Path:
        value = valid.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"QC valid manifest lacks path field {key!r}.")
        path = Path(os.path.expanduser(os.path.expandvars(value)))
        return _resolve_path(path if path.is_absolute() else base / path, label=key)

    checkpoint_path = declared_path("checkpoint_path")
    dataset_stats_path = declared_path("dataset_stats_path")
    source_manifest_path = declared_path("source_manifest_path")
    prompt_cache_path = declared_path("prompt_context_cache_path")
    if checkpoint_path != Path(args.checkpoint).expanduser().resolve():
        raise ValueError("--checkpoint disagrees with the frozen valid manifest.")
    if dataset_stats_path != Path(args.dataset_stats_path).expanduser().resolve():
        raise ValueError("--dataset-stats-path disagrees with the frozen valid manifest.")

    donor_mapping_path = _resolve_path(args.donor_mapping, label="donor mapping")
    donor_manifest_path = _resolve_path(
        args.donor_observation_manifest, label="donor observation manifest"
    )
    donor_root = _resolve_path(
        args.donor_observation_root, label="donor observation root", file=False
    )
    donor_report = _validate_donors(
        donor_mapping_path, donor_manifest_path, donor_root
    )
    preflight_path = _resolve_path(args.preflight_report, label="preflight report")
    preflight = _read_json(preflight_path, label="Round-3B preflight report")
    expected_preflight = {
        "artifact_type": "asre_round3b_preflight_report",
        "schema_version": 1,
        "status": "compatible",
    }
    mismatch = _metadata_mismatches(preflight, expected_preflight)
    git_section = preflight.get("git")
    donors_section = preflight.get("donors")
    round3a_section = preflight.get("round3a")
    expected_git = {
        "round3a_tag": ROUND3A_TAG,
        "round3a_frozen_commit": ROUND3A_FROZEN_COMMIT,
        "round3a_run_commit": ROUND3A_RUN_COMMIT,
        "current_head": git_commit(project_root),
    }
    if not isinstance(git_section, Mapping):
        mismatch["git"] = {"existing": git_section, "requested": expected_git}
    else:
        mismatch.update(
            {
                f"git.{key}": value
                for key, value in _metadata_mismatches(git_section, expected_git).items()
            }
        )
    if not isinstance(donors_section, Mapping):
        mismatch["donors"] = {"existing": donors_section, "requested": donor_report}
    else:
        donor_expected = {
            key: donor_report[key]
            for key in (
                "donor_mapping_path",
                "donor_mapping_sha256",
                "donor_observation_manifest_path",
                "donor_observation_manifest_sha256",
                "donor_observation_root",
                "online_mapping_count",
                "online_observation_count",
                "mapping_rule",
            )
        }
        mismatch.update(
            {
                f"donors.{key}": value
                for key, value in _metadata_mismatches(
                    donors_section, donor_expected
                ).items()
            }
        )
    if not isinstance(round3a_section, Mapping):
        mismatch["round3a"] = {
            "existing": round3a_section,
            "requested": "validated frozen Round-3A section",
        }
    if mismatch:
        raise ValueError(
            "Preflight report is incompatible with this launch: "
            f"{json.dumps(mismatch, sort_keys=True)}"
        )

    identity_path = _resolve_path(
        args.self_replacement_report, label="self-replacement report"
    )
    identity = _read_json(identity_path, label="self-replacement report")
    identity_expected = {
        "artifact_type": "asre_round3b_self_replacement_identity",
        "schema_version": 1,
        "protocol": ROUND3B_PROTOCOL,
        "status": "passed",
        "passed": True,
        "git_commit_hash": git_commit(project_root),
        "checkpoint_path": str(checkpoint_path),
        "valid_state_bank_manifest_path": str(valid_path),
        "disabled_video_layers": list(range(15)),
        "replacement_video_layers": list(range(15, 30)),
        "action_finite": True,
        "action_allclose": True,
        "torch_allclose": True,
        "atol": 1.0e-4,
        "rtol": 1.0e-4,
    }
    mismatch = _metadata_mismatches(identity, identity_expected)
    if float(identity.get("action_max_absolute_difference", math.inf)) > 1.0e-4:
        mismatch["action_max_absolute_difference"] = {
            "existing": identity.get("action_max_absolute_difference"),
            "requested": "<= 1e-4",
        }
    cache_audit = identity.get("cache_audit")
    if not isinstance(cache_audit, Mapping) or float(
        cache_audit.get("max_abs_current_replacement", math.inf)
    ) > 1.0e-4:
        mismatch["cache_audit"] = {
            "existing": cache_audit,
            "requested": "same-input cache max_abs <= 1e-4",
        }
    if mismatch:
        raise ValueError(
            "Self-replacement machinery gate is incompatible or failed: "
            f"{json.dumps(mismatch, sort_keys=True)}"
        )

    valid_digest = sha256_file(valid_path)
    if isinstance(round3a_section, Mapping) and round3a_section.get(
        "valid_state_bank_manifest_sha256"
    ) != valid_digest:
        raise ValueError("Preflight report references a different valid state manifest.")
    valid_ids = valid.get("valid_sample_ids")
    if not isinstance(valid_ids, list) or len(valid_ids) != 499:
        raise ValueError("Round 3B requires exactly 499 frozen QC-valid states.")

    required_hashes = (
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "source_manifest_sha256",
        "prompt_context_cache_sha256",
    )
    for key in required_hashes:
        value = valid.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"QC valid manifest has invalid {key}: {value!r}")

    return Provenance(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=str(valid["checkpoint_sha256"]),
        dataset_stats_path=dataset_stats_path,
        dataset_stats_sha256=str(valid["dataset_stats_sha256"]),
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=str(valid["source_manifest_sha256"]),
        valid_manifest_path=valid_path,
        valid_manifest_sha256=valid_digest,
        prompt_context_cache_path=prompt_cache_path,
        prompt_context_cache_sha256=str(valid["prompt_context_cache_sha256"]),
        donor_mapping_path=donor_mapping_path,
        donor_mapping_sha256=str(donor_report["donor_mapping_sha256"]),
        donor_observation_manifest_path=donor_manifest_path,
        donor_observation_manifest_sha256=str(
            donor_report["donor_observation_manifest_sha256"]
        ),
        donor_observation_root=donor_root,
        preflight_report_path=preflight_path,
        preflight_report_sha256=sha256_file(preflight_path),
        self_replacement_report_path=identity_path,
        self_replacement_report_sha256=sha256_file(identity_path),
        round3a_tag=ROUND3A_TAG,
        round3a_frozen_commit=ROUND3A_FROZEN_COMMIT,
        round3a_run_commit=ROUND3A_RUN_COMMIT,
        valid_sample_count=len(valid_ids),
    )


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
            ],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot inspect Git worktree before Round-3B: {exc}") from exc
    if output.strip():
        raise RuntimeError(
            "ASRE Round 3B requires a clean code worktree so metadata identifies "
            "the exact implementation. Commit code first; generated Round-3B "
            f"artifacts are exempt.\n{output.rstrip()}"
        )


def _validate_output_scope(output_root: Path, provenance: Provenance) -> None:
    output_root = output_root.resolve()
    protected = {
        provenance.source_manifest_path.parent.resolve(),
        provenance.valid_manifest_path.parent.resolve(),
    }
    preflight = _read_json(provenance.preflight_report_path, label="preflight report")
    round3a = preflight["round3a"]
    protected.add(Path(round3a["round3a_aggregate_metadata_path"]).parents[1].resolve())
    for path in protected:
        if output_root == path or output_root.is_relative_to(path) or path.is_relative_to(output_root):
            raise DirectorySafetyError(
                f"Round-3B output overlaps a frozen Round-1/2/3A tree: {output_root}, {path}."
            )


def _resolve_runtime(task_config: str, mode: str) -> RuntimeSpec:
    from hydra import compose, initialize_config_dir

    config_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="sim_libero.yaml", overrides=[f"task={task_config}"])
    num_layers = int(cfg.model.action_dit_config.num_layers)
    if num_layers != NUM_LAYERS:
        raise ValueError(f"Round 3B requires 30 action layers, got {num_layers}.")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if cfg.EVALUATION.get("action_horizon") is None
        else int(cfg.EVALUATION.action_horizon)
    )
    inference_steps = int(cfg.EVALUATION.num_inference_steps)
    replan_steps = int(cfg.EVALUATION.replan_steps)
    if (action_horizon, inference_steps, replan_steps) != (
        ACTION_HORIZON,
        INFERENCE_STEPS,
        REPLAN_STEPS,
    ):
        raise ValueError("Task config disagrees with the frozen Round-3B runtime.")
    controls = {
        "compile_action_infer": bool(cfg.EVALUATION.compile_action_infer),
        "binarize_gripper": bool(cfg.EVALUATION.binarize_gripper),
        "sigma_shift": cfg.EVALUATION.sigma_shift,
        "rand_device": str(cfg.EVALUATION.rand_device),
        "num_steps_wait": int(cfg.EVALUATION.num_steps_wait),
    }
    expected = {
        "compile_action_infer": True,
        "binarize_gripper": True,
        "sigma_shift": None,
        "rand_device": "cpu",
        "num_steps_wait": 30,
    }
    if controls != expected:
        raise ValueError(f"Task config controls changed: {controls} != {expected}.")
    return RuntimeSpec(
        task_config=task_config,
        task_ids=SMOKE_TASK_IDS if mode == "smoke" else FULL_TASK_IDS,
        num_trials=SMOKE_TRIALS if mode == "smoke" else FULL_TRIALS,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        inference_steps=inference_steps,
    )


def _expected_metadata(
    condition: DiagnosisCondition,
    runtime: RuntimeSpec,
    provenance: Provenance,
) -> dict[str, Any]:
    return {
        "git_commit_hash": git_commit(project_root),
        "checkpoint_path": str(provenance.checkpoint_path),
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "dataset_stats_path": str(provenance.dataset_stats_path),
        "dataset_stats_sha256": provenance.dataset_stats_sha256,
        "diagnosis_condition": condition.name,
        "condition_protocol": ROUND3B_PROTOCOL,
        "enabled_video_retrieval_layers": list(
            condition.enabled_video_retrieval_layers(NUM_LAYERS)
        ),
        "disabled_video_layers": list(condition.disabled_video_layers),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "num_model_layers": NUM_LAYERS,
        "task_suite": TASK_SUITE,
        "task_ids": list(runtime.task_ids),
        "seed": SEED,
        "number_of_trials": runtime.num_trials,
        "action_horizon": runtime.action_horizon,
        "number_of_inference_steps": runtime.inference_steps,
        "replan_steps": runtime.replan_steps,
        "state_bank_manifest_path": str(provenance.source_manifest_path),
        "state_bank_manifest_sha256": provenance.source_manifest_sha256,
        "valid_state_bank_manifest_path": str(provenance.valid_manifest_path),
        "valid_state_bank_manifest_sha256": provenance.valid_manifest_sha256,
        "prompt_context_cache_path": str(provenance.prompt_context_cache_path),
        "prompt_context_cache_sha256": provenance.prompt_context_cache_sha256,
        "donor_mapping_path": str(provenance.donor_mapping_path),
        "donor_mapping_sha256": provenance.donor_mapping_sha256,
        "donor_observation_manifest_path": str(
            provenance.donor_observation_manifest_path
        ),
        "donor_observation_manifest_sha256": (
            provenance.donor_observation_manifest_sha256
        ),
        "donor_observation_root": str(provenance.donor_observation_root),
        "round3a_parent_tag": provenance.round3a_tag,
        "round3a_parent_commit": provenance.round3a_frozen_commit,
        "round3a_run_commit": provenance.round3a_run_commit,
        "self_replacement_report_path": str(
            provenance.self_replacement_report_path
        ),
        "self_replacement_report_sha256": (
            provenance.self_replacement_report_sha256
        ),
    }


def _validate_task_result(
    path: Path, *, condition: DiagnosisCondition, runtime: RuntimeSpec
) -> int:
    result = _read_json(path, label="task result")
    expected = {
        "task_suite": TASK_SUITE,
        "diagnosis_condition": condition.name,
        "condition_protocol": ROUND3B_PROTOCOL,
        "enabled_video_retrieval_layers": list(
            condition.enabled_video_retrieval_layers(NUM_LAYERS)
        ),
        "disabled_video_layers": list(condition.disabled_video_layers),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "total_episodes": runtime.num_trials,
    }
    mismatch = _metadata_mismatches(result, expected)
    if mismatch:
        raise DirectorySafetyError(
            f"Task result is incompatible: {path}: {json.dumps(mismatch, sort_keys=True)}"
        )
    task_id = int(result.get("task_id", -1))
    if task_id not in runtime.task_ids:
        raise DirectorySafetyError(f"Unexpected task_id={task_id} in {path}.")
    successes = [int(value) for value in result.get("success_episodes", [])]
    failures = [int(value) for value in result.get("failure_episodes", [])]
    success_set, failure_set = set(successes), set(failures)
    expected_episodes = set(range(runtime.num_trials))
    if (
        len(successes) != len(success_set)
        or len(failures) != len(failure_set)
        or success_set & failure_set
        or success_set | failure_set != expected_episodes
        or int(result.get("successes", -1)) != len(success_set)
    ):
        raise DirectorySafetyError(f"Inconsistent paired outcomes in {path}.")
    return task_id


def _validate_action_traces(
    condition_dir: Path,
    *,
    task_id: int,
    condition: DiagnosisCondition,
    runtime: RuntimeSpec,
) -> None:
    trace_dir = condition_dir / TASK_SUITE / "action_traces"
    for episode_id in range(runtime.num_trials):
        path = trace_dir / f"task{task_id}_trial{episode_id}.jsonl"
        if not path.is_file():
            raise DirectorySafetyError(f"Missing action trace: {path}")
        records: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, Mapping):
                        raise DirectorySafetyError(f"Invalid action trace object: {path}")
                    records.append(record)
        if not records:
            raise DirectorySafetyError(f"Empty action trace: {path}")
        for replan_id, record in enumerate(records):
            expected = {
                "task_suite": TASK_SUITE,
                "task_id": task_id,
                "episode_id": episode_id,
                "replan_id": replan_id,
                "diagnosis_condition": condition.name,
                "environment_step": 30 + replan_id * runtime.replan_steps,
                "action_inference_seed": SEED,
                "replacement_video_layers": list(condition.replacement_video_layers),
            }
            mismatch = _metadata_mismatches(record, expected)
            if mismatch:
                raise DirectorySafetyError(
                    f"Action trace identity mismatch in {path}: "
                    f"{json.dumps(mismatch, sort_keys=True)}"
                )
            if condition.name == "late_wrong_scene":
                donor_trial = record.get("donor_trial")
                if donor_trial != (episode_id + 1) % 10:
                    raise DirectorySafetyError(
                        f"Wrong donor_trial in {path}: {donor_trial}."
                    )
            _validate_action_array(record.get("raw_action"), path=path, field="raw_action")
            _validate_action_array(
                record.get("executed_action"), path=path, field="executed_action"
            )


def _inspect_condition_dir(
    condition_dir: Path,
    *,
    condition_index: int,
    physical_gpu: int,
    condition: DiagnosisCondition,
    runtime: RuntimeSpec,
    provenance: Provenance,
) -> ConditionInspection:
    if not condition_dir.exists() or (
        condition_dir.is_dir() and not any(condition_dir.iterdir())
    ):
        return ConditionInspection("missing", (), "directory is absent or empty")
    if not condition_dir.is_dir():
        raise DirectorySafetyError(f"Condition output is not a directory: {condition_dir}")
    status_path = condition_dir / STATUS_NAME
    status = _read_json(status_path, label="launcher status") if status_path.exists() else None
    if status is not None:
        mismatch = _metadata_mismatches(
            status,
            {
                "schema_version": 1,
                "protocol": ROUND3B_PROTOCOL,
                "condition": condition.name,
                "condition_index": condition_index,
                "physical_gpu": physical_gpu,
                "replacement_video_layers": list(condition.replacement_video_layers),
            },
        )
        if mismatch:
            raise DirectorySafetyError(f"Incompatible launcher status: {mismatch}")
        live_pid = _recorded_live_pid(status)
        if live_pid is not None:
            raise DirectorySafetyError(
                f"Previous child still appears live (pid={live_pid}): {condition_dir}"
            )
    metadata_path = condition_dir / "run_metadata.json"
    if not metadata_path.is_file():
        unexpected = [p.name for p in condition_dir.iterdir() if p.name != STATUS_NAME]
        if unexpected or status is None:
            raise DirectorySafetyError(
                f"Unidentifiable condition directory without metadata: {condition_dir}"
            )
        return ConditionInspection("partial", (), "launcher child has not written metadata")
    metadata = _read_json(metadata_path, label="condition run metadata")
    mismatch = _metadata_mismatches(
        metadata, _expected_metadata(condition, runtime, provenance)
    )
    if mismatch:
        raise DirectorySafetyError(
            f"Condition metadata is incompatible: {json.dumps(mismatch, sort_keys=True)}"
        )
    completed: dict[int, Path] = {}
    for path in sorted(condition_dir.glob("**/gpu*_task*_results.json")):
        task_id = _validate_task_result(path, condition=condition, runtime=runtime)
        if task_id in completed:
            raise DirectorySafetyError(f"Duplicate task result for task {task_id}.")
        completed[task_id] = path
        _validate_action_traces(
            condition_dir,
            task_id=task_id,
            condition=condition,
            runtime=runtime,
        )
    completed_ids = tuple(sorted(completed))
    all_complete = set(completed_ids) == set(runtime.task_ids)
    metadata_status = str(metadata.get("status", ""))
    if metadata_status == "completed" and not all_complete:
        raise DirectorySafetyError("Metadata claims completion with missing task results.")
    if metadata_status == "completed" and all_complete:
        return ConditionInspection("complete", completed_ids, "all task results validated")
    if metadata_status not in {"", "running", "failed", "interrupted"}:
        raise DirectorySafetyError(f"Unsupported metadata status {metadata_status!r}.")
    return ConditionInspection("partial", completed_ids, f"metadata={metadata_status}")


def _root_identity(
    *,
    mode: str,
    runtime: RuntimeSpec,
    provenance: Provenance,
    python_path: Path,
    gpu_inventory: Sequence[Mapping[str, Any]],
    gpu_ids: Sequence[int],
) -> dict[str, Any]:
    conditions = build_round3b_conditions(NUM_LAYERS)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": ROUND3B_PROTOCOL,
        "mode": mode,
        "task_config": runtime.task_config,
        "task_suite": TASK_SUITE,
        "task_ids": list(runtime.task_ids),
        "num_trials": runtime.num_trials,
        "seed": SEED,
        "action_horizon": runtime.action_horizon,
        "replan_steps": runtime.replan_steps,
        "inference_steps": runtime.inference_steps,
        "git_commit_hash": git_commit(project_root),
        "python": str(python_path),
        "gpu_ids": list(gpu_ids),
        "provenance": provenance.identity_dict(),
        "conditions": [
            {
                "condition_index": index,
                "condition": condition.name,
                "physical_gpu": gpu_ids[index],
                "enabled_video_retrieval_layers": list(
                    condition.enabled_video_retrieval_layers(NUM_LAYERS)
                ),
                "disabled_video_layers": list(condition.disabled_video_layers),
                "replacement_video_layers": list(condition.replacement_video_layers),
            }
            for index, condition in enumerate(conditions)
        ],
        "gpu_inventory": [
            {
                key: record[key]
                for key in (
                    "index",
                    "name",
                    "uuid",
                    "pci_bus_id",
                    "driver_version",
                    "memory_total_mib",
                )
            }
            for record in gpu_inventory
        ],
    }
    payload["identity_sha256"] = sha256_json(payload)
    return payload


def _prepare_output_root(output_root: Path, identity: Mapping[str, Any]) -> None:
    config = output_root / ROOT_CONFIG_NAME
    if output_root.exists() and not output_root.is_dir():
        raise DirectorySafetyError(f"Output root is not a directory: {output_root}")
    if config.exists():
        if _read_json(config, label="launcher root config") != identity:
            raise DirectorySafetyError(f"Output root belongs to another run: {output_root}")
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise DirectorySafetyError(f"Non-empty output lacks {ROOT_CONFIG_NAME}: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config, identity)


def _validate_root_layout(output_root: Path) -> None:
    allowed = {
        ROOT_CONFIG_NAME,
        SUMMARY_NAME,
        "logs",
        *(condition.name for condition in build_round3b_conditions(NUM_LAYERS)),
    }
    unexpected = sorted(path.name for path in output_root.iterdir() if path.name not in allowed)
    if unexpected:
        raise DirectorySafetyError(f"Unexpected output entries: {unexpected}")


def _condition_command(
    *,
    python_path: Path,
    condition_index: int,
    condition: DiagnosisCondition,
    condition_output: Path,
    runtime: RuntimeSpec,
    provenance: Provenance,
) -> list[str]:
    compact = lambda value: json.dumps(value, separators=(",", ":"))
    enabled = list(condition.enabled_video_retrieval_layers(NUM_LAYERS))
    return [
        str(python_path),
        str(project_root / "experiments" / "libero" / "eval_libero_single.py"),
        f"task={runtime.task_config}",
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
        f"EVALUATION.action_horizon={runtime.action_horizon}",
        f"EVALUATION.num_inference_steps={runtime.inference_steps}",
        f"EVALUATION.replan_steps={runtime.replan_steps}",
        f"EVALUATION.output_dir={condition_output}",
        "EVALUATION.visualize_future_video=false",
        f"EVALUATION.dataset_stats_path={provenance.dataset_stats_path}",
        "ASRE_DIAGNOSIS.enabled=true",
        "ASRE_DIAGNOSIS.mode=replace_video_kv",
        f"ASRE_DIAGNOSIS.protocol={ROUND3B_PROTOCOL}",
        f"ASRE_DIAGNOSIS.condition_index={condition_index}",
        f"ASRE_DIAGNOSIS.condition_name={condition.name}",
        f"ASRE_DIAGNOSIS.enabled_video_retrieval_layers={compact(enabled)}",
        f"ASRE_DIAGNOSIS.disabled_video_layers={compact(list(condition.disabled_video_layers))}",
        f"ASRE_DIAGNOSIS.replacement_video_layers={compact(list(condition.replacement_video_layers))}",
        "ASRE_DIAGNOSIS.save_rollout_video=false",
        f"ASRE_DIAGNOSIS.checkpoint_sha256={provenance.checkpoint_sha256}",
        f"ASRE_DIAGNOSIS.dataset_stats_sha256={provenance.dataset_stats_sha256}",
        f"ASRE_DIAGNOSIS.state_bank_manifest_path={provenance.source_manifest_path}",
        f"ASRE_DIAGNOSIS.state_bank_manifest_sha256={provenance.source_manifest_sha256}",
        f"ASRE_DIAGNOSIS.valid_state_bank_manifest_path={provenance.valid_manifest_path}",
        f"ASRE_DIAGNOSIS.valid_state_bank_manifest_sha256={provenance.valid_manifest_sha256}",
        f"ASRE_DIAGNOSIS.prompt_context_cache_sha256={provenance.prompt_context_cache_sha256}",
        f"ASRE_DIAGNOSIS.donor_mapping_path={provenance.donor_mapping_path}",
        f"ASRE_DIAGNOSIS.donor_mapping_sha256={provenance.donor_mapping_sha256}",
        f"ASRE_DIAGNOSIS.donor_observation_manifest_path={provenance.donor_observation_manifest_path}",
        f"ASRE_DIAGNOSIS.donor_observation_manifest_sha256={provenance.donor_observation_manifest_sha256}",
        f"ASRE_DIAGNOSIS.donor_observation_root={provenance.donor_observation_root}",
        f"ASRE_DIAGNOSIS.preflight_report_path={provenance.preflight_report_path}",
        f"ASRE_DIAGNOSIS.preflight_report_sha256={provenance.preflight_report_sha256}",
        f"ASRE_DIAGNOSIS.self_replacement_report_path={provenance.self_replacement_report_path}",
        f"ASRE_DIAGNOSIS.self_replacement_report_sha256={provenance.self_replacement_report_sha256}",
        f"ASRE_DIAGNOSIS.round3a_parent_tag={provenance.round3a_tag}",
        f"ASRE_DIAGNOSIS.round3a_parent_commit={provenance.round3a_frozen_commit}",
        f"ASRE_DIAGNOSIS.round3a_run_commit={provenance.round3a_run_commit}",
    ]


def _validate_smoke_gate(
    path: Path | None,
    provenance: Provenance,
    task_config: str,
    gpu_ids: Sequence[int],
) -> None:
    if path is None:
        raise ValueError("--mode full requires --smoke-summary.")
    summary_path = path.resolve()
    summary = _read_json(summary_path, label="Round-3B smoke summary")
    names = [condition.name for condition in build_round3b_conditions(NUM_LAYERS)]
    expected = {
        "schema_version": 1,
        "protocol": ROUND3B_PROTOCOL,
        "mode": "smoke",
        "all_succeeded": True,
        "interrupted": False,
        "task_config": task_config,
        "task_ids": list(SMOKE_TASK_IDS),
        "num_trials": SMOKE_TRIALS,
        "seed": SEED,
        "git_commit_hash": git_commit(project_root),
        "gpu_ids": list(gpu_ids),
        "provenance": provenance.identity_dict(),
        "expected_conditions": names,
    }
    mismatch = _metadata_mismatches(summary, expected)
    states = summary.get("condition_states")
    if not isinstance(states, Mapping) or any(
        not isinstance(states.get(name), Mapping)
        or states[name].get("state") != "complete"
        for name in names
    ):
        mismatch["condition_states"] = {
            "existing": states,
            "requested": "all smoke conditions complete",
        }
    if mismatch:
        raise ValueError(f"Smoke gate failed: {json.dumps(mismatch, sort_keys=True)}")
    smoke_root = Path(str(summary.get("output_root", ""))).resolve()
    runtime = RuntimeSpec(
        task_config=task_config,
        task_ids=SMOKE_TASK_IDS,
        num_trials=SMOKE_TRIALS,
        action_horizon=ACTION_HORIZON,
        replan_steps=REPLAN_STEPS,
        inference_steps=INFERENCE_STEPS,
    )
    for index, condition in enumerate(build_round3b_conditions(NUM_LAYERS)):
        inspected = _inspect_condition_dir(
            smoke_root / condition.name,
            condition_index=index,
            physical_gpu=gpu_ids[index],
            condition=condition,
            runtime=runtime,
            provenance=provenance,
        )
        if inspected.state != "complete":
            raise ValueError(f"Smoke condition is no longer complete: {condition.name}")


def _status_payload(
    condition_index: int,
    physical_gpu: int,
    condition: DiagnosisCondition,
    mode: str,
    output_dir: Path,
    gpu_record: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    attempts = []
    if previous is not None and isinstance(previous.get("attempts"), list):
        attempts.extend(previous["attempts"])
    attempts.append(dict(attempt))
    return {
        "schema_version": 1,
        "protocol": ROUND3B_PROTOCOL,
        "mode": mode,
        "condition": condition.name,
        "condition_index": condition_index,
        "enabled_video_retrieval_layers": list(
            condition.enabled_video_retrieval_layers(NUM_LAYERS)
        ),
        "disabled_video_layers": list(condition.disabled_video_layers),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "physical_gpu": physical_gpu,
        "gpu": dict(gpu_record),
        "cuda_visible_devices": str(physical_gpu),
        "model_device": "cuda:0",
        "mujoco_egl_device_id": str(physical_gpu),
        "output_dir": str(output_dir),
        "status": attempt["status"],
        "attempts": attempts,
    }


def _launch_condition(
    *,
    condition_index: int,
    physical_gpu: int,
    condition: DiagnosisCondition,
    output_root: Path,
    python_path: Path,
    runtime: RuntimeSpec,
    provenance: Provenance,
    mode: str,
    gpu_record: Mapping[str, Any],
    launcher_lock_fd: int,
) -> LiveProcess:
    condition_output = output_root / condition.name
    condition_output.mkdir(parents=True, exist_ok=True)
    status_path = condition_output / STATUS_NAME
    previous = _read_json(status_path, label="launcher status") if status_path.exists() else None
    attempts = previous.get("attempts", []) if isinstance(previous, Mapping) else []
    attempt_number = len(attempts) + 1 if isinstance(attempts, list) else 1
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{condition.name}.attempt{attempt_number:02d}.log"
    while log_path.exists():
        attempt_number += 1
        log_path = logs / f"{condition.name}.attempt{attempt_number:02d}.log"
    command = _condition_command(
        python_path=python_path,
        condition_index=condition_index,
        condition=condition,
        condition_output=condition_output,
        runtime=runtime,
        provenance=provenance,
    )
    attempt: dict[str, Any] = {
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
        condition_index,
        physical_gpu,
        condition,
        mode,
        condition_output,
        gpu_record,
        previous,
        attempt,
    )
    atomic_write_json(status_path, status)
    log_handle = log_path.open("x", encoding="utf-8")
    log_handle.write(
        f"[{now_iso()}] {condition.name} physical_gpu={physical_gpu} "
        "logical_device=cuda:0 no_DDP=true\n"
    )
    log_handle.write(shlex.join(command) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=_child_environment(physical_gpu),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        pass_fds=(launcher_lock_fd,),
    )
    attempt["pid"] = process.pid
    attempt["status"] = "running"
    status["attempts"][-1] = attempt
    status["status"] = "running"
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


def _finish(live: LiveProcess, exit_status: int, status: str | None = None) -> None:
    if not live.log_handle.closed:
        live.log_handle.close()
    state = status or ("completed" if exit_status == 0 else "failed")
    attempt = live.status["attempts"][-1]
    attempt["exit_status"] = int(exit_status)
    attempt["end_timestamp"] = now_iso()
    attempt["status"] = state
    live.status["status"] = state
    atomic_write_json(live.status_path, live.status)


def _terminate(live: LiveProcess, state: str) -> None:
    if live.process.poll() is None:
        try:
            os.killpg(live.process.pid, signal.SIGTERM)
            live.process.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if live.process.poll() is None:
                try:
                    os.killpg(live.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                live.process.wait(timeout=15)
    _finish(live, int(live.process.returncode or 1), state)


def _collect_states(
    output_root: Path,
    runtime: RuntimeSpec,
    provenance: Provenance,
    gpu_ids: Sequence[int],
) -> tuple[dict[str, Any], bool]:
    states: dict[str, Any] = {}
    all_complete = True
    for index, condition in enumerate(build_round3b_conditions(NUM_LAYERS)):
        try:
            inspected = _inspect_condition_dir(
                output_root / condition.name,
                condition_index=index,
                physical_gpu=gpu_ids[index],
                condition=condition,
                runtime=runtime,
                provenance=provenance,
            )
            states[condition.name] = {
                "state": inspected.state,
                "completed_task_ids": list(inspected.completed_task_ids),
                "detail": inspected.detail,
                "physical_gpu": gpu_ids[index],
            }
            all_complete &= inspected.state == "complete"
        except Exception as exc:
            states[condition.name] = {
                "state": "invalid",
                "completed_task_ids": [],
                "detail": str(exc),
                "physical_gpu": gpu_ids[index],
            }
            all_complete = False
    return states, all_complete


def _write_summary(
    output_root: Path,
    *,
    mode: str,
    runtime: RuntimeSpec,
    provenance: Provenance,
    states: Mapping[str, Any],
    all_succeeded: bool,
    interrupted: bool,
    start: str,
    gpu_ids: Sequence[int],
) -> Path:
    path = output_root / SUMMARY_NAME
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "protocol": ROUND3B_PROTOCOL,
            "mode": mode,
            "all_succeeded": all_succeeded,
            "interrupted": interrupted,
            "output_root": str(output_root),
            "task_config": runtime.task_config,
            "task_ids": list(runtime.task_ids),
            "num_trials": runtime.num_trials,
            "seed": SEED,
            "git_commit_hash": git_commit(project_root),
            "gpu_ids": list(gpu_ids),
            "provenance": provenance.identity_dict(),
            "expected_conditions": [
                condition.name for condition in build_round3b_conditions(NUM_LAYERS)
            ],
            "condition_states": dict(states),
            "start_timestamp": start,
            "end_timestamp": now_iso(),
        },
    )
    return path


def _main_locked(
    args: argparse.Namespace,
    *,
    output_root: Path,
    launcher_lock_fd: int,
    provenance: Provenance | None = None,
) -> None:
    start = now_iso()
    python_path = _resolve_python(args.python)
    provenance = _load_provenance(args) if provenance is None else provenance
    runtime = _resolve_runtime(args.task_config, args.mode)
    gpu_ids = _validate_gpu_ids(args.gpu_ids)
    if args.mode == "full":
        _validate_smoke_gate(
            args.smoke_summary, provenance, args.task_config, gpu_ids
        )
    elif args.smoke_summary is not None:
        raise ValueError("--smoke-summary is only valid for --mode full.")
    gpu_inventory = _gpu_inventory()
    inventory_by_index = {int(record["index"]): record for record in gpu_inventory}
    missing_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id not in inventory_by_index]
    if missing_gpu_ids:
        raise ValueError(
            f"Requested physical GPUs are absent from inventory: {missing_gpu_ids}."
        )
    identity = _root_identity(
        mode=args.mode,
        runtime=runtime,
        provenance=provenance,
        python_path=python_path,
        gpu_inventory=gpu_inventory,
        gpu_ids=gpu_ids,
    )
    _prepare_output_root(output_root, identity)
    _validate_root_layout(output_root)

    conditions = build_round3b_conditions(NUM_LAYERS)
    inspections = {
        index: _inspect_condition_dir(
            output_root / condition.name,
            condition_index=index,
            physical_gpu=gpu_ids[index],
            condition=condition,
            runtime=runtime,
            provenance=provenance,
        )
        for index, condition in enumerate(conditions)
    }
    if all(item.state == "complete" for item in inspections.values()):
        states, complete = _collect_states(output_root, runtime, provenance, gpu_ids)
        existing_summary = output_root / SUMMARY_NAME
        if existing_summary.exists():
            current = _read_json(existing_summary, label="launcher summary")
            if not current.get("all_succeeded") or current.get("condition_states") != states:
                raise DirectorySafetyError("Existing completed summary is incompatible.")
            print(f"All Round-3B conditions already complete: {existing_summary}")
            return
        print(_write_summary(
            output_root,
            mode=args.mode,
            runtime=runtime,
            provenance=provenance,
            states=states,
            all_succeeded=complete,
            interrupted=False,
            start=start,
            gpu_ids=gpu_ids,
        ))
        return

    live: list[LiveProcess] = []
    interrupted = False
    failure: BaseException | None = None
    try:
        for index, condition in enumerate(conditions):
            if inspections[index].state == "complete":
                print(f"Skipping completed {condition.name}.", flush=True)
                continue
            launched = _launch_condition(
                condition_index=index,
                physical_gpu=gpu_ids[index],
                condition=condition,
                output_root=output_root,
                python_path=python_path,
                runtime=runtime,
                provenance=provenance,
                mode=args.mode,
                gpu_record=inventory_by_index[gpu_ids[index]],
                launcher_lock_fd=launcher_lock_fd,
            )
            live.append(launched)
            print(
                f"Launched {condition.name} on physical GPU {gpu_ids[index]}: "
                f"{launched.log_path}",
                flush=True,
            )
        for process in live:
            code = process.process.wait()
            _finish(process, code)
            if code != 0:
                failure = RuntimeError(f"{process.condition.name} exited {code}.")
            else:
                inspected = _inspect_condition_dir(
                    output_root / process.condition.name,
                    condition_index=process.condition_index,
                    physical_gpu=process.physical_gpu,
                    condition=process.condition,
                    runtime=runtime,
                    provenance=provenance,
                )
                if inspected.state != "complete":
                    failure = DirectorySafetyError(
                        f"{process.condition.name} exited 0 but is {inspected.state}."
                    )
                    process.status["status"] = "validation_failed"
                    atomic_write_json(process.status_path, process.status)
    except KeyboardInterrupt as exc:
        interrupted = True
        failure = exc
    except BaseException as exc:
        failure = exc
    finally:
        for process in live:
            if process.process.poll() is None:
                _terminate(process, "interrupted" if interrupted else "terminated")
            elif not process.log_handle.closed:
                process.log_handle.close()

    states, complete = _collect_states(output_root, runtime, provenance, gpu_ids)
    summary = _write_summary(
        output_root,
        mode=args.mode,
        runtime=runtime,
        provenance=provenance,
        states=states,
        all_succeeded=complete and failure is None,
        interrupted=interrupted,
        start=start,
        gpu_ids=gpu_ids,
    )
    print(f"Launcher summary: {summary}")
    if interrupted:
        raise SystemExit(130)
    if failure is not None:
        raise failure
    if not complete:
        raise SystemExit(1)


def main() -> None:
    args = _parse_args()
    _require_clean_worktree()
    output_root = Path(args.output_root).expanduser().resolve()
    # Load provenance before creating the output root so protected scopes are known.
    provenance = _load_provenance(args)
    _validate_output_scope(output_root, provenance)
    with _exclusive_launcher_lock(output_root) as launcher_lock_fd:
        _main_locked(
            args,
            output_root=output_root,
            launcher_lock_fd=launcher_lock_fd,
            provenance=provenance,
        )


if __name__ == "__main__":
    main()
