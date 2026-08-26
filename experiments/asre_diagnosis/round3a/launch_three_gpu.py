"""Launch the pre-registered ASRE Round 3A online matrix on three GPUs.

Each condition owns exactly one physical GPU.  Text contexts are loaded from
the QC-approved Round-1 state-bank cache, so no T5 encoder is instantiated.
The launcher is intentionally fail-closed around provenance and result-dir
reuse: completed condition directories are inspected and left untouched,
partial directories resume only after immutable metadata validation, and
unidentifiable directories are rejected.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (
    ROUND2_PROTOCOL,
    ROUND3A_PROTOCOL,
    DiagnosisCondition,
    atomic_write_json,
    build_round3a_conditions,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.round2.validate_state_bank import (
    EXPECTED_SOURCE_SAMPLES,
    QC_RULE,
    QC_SCHEMA_VERSION,
    _validate_sample_partition,
    validate_existing_artifacts,
)


NUM_LAYERS = 30
TASK_SUITE = "libero_spatial"
SEED = 42
FULL_TASK_IDS = tuple(range(10))
SMOKE_TASK_IDS = (0,)
FULL_TRIALS = 10
SMOKE_TRIALS = 2
SMOKE_CONDITION_INDICES = (0, 1, 2)
ACTION_HORIZON = 32
REPLAN_STEPS = 10
INFERENCE_STEPS = 10
ROOT_CONFIG_NAME = "launcher_config.json"
SUMMARY_NAME = "launcher_summary.json"
STATUS_NAME = "launcher_status.json"


class DirectorySafetyError(RuntimeError):
    """Raised when an output directory cannot be safely identified/resumed."""


def _raise_keyboard_interrupt(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"launcher received signal {signum}")


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
    valid_sample_count: int
    qc_rule: Any
    round2_reference_metadata_path: Path
    round2_reference_metadata_sha256: str

    def identity_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "dataset_stats_path": str(self.dataset_stats_path),
            "dataset_stats_sha256": self.dataset_stats_sha256,
            "source_manifest_path": str(self.source_manifest_path),
            "source_manifest_sha256": self.source_manifest_sha256,
            "valid_manifest_path": str(self.valid_manifest_path),
            "valid_manifest_sha256": self.valid_manifest_sha256,
            "prompt_context_cache_path": str(self.prompt_context_cache_path),
            "prompt_context_cache_sha256": self.prompt_context_cache_sha256,
            "valid_sample_count": self.valid_sample_count,
            "qc_rule": self.qc_rule,
            "round2_reference_metadata_path": str(
                self.round2_reference_metadata_path
            ),
            "round2_reference_metadata_sha256": (
                self.round2_reference_metadata_sha256
            ),
        }


@dataclass(frozen=True)
class RuntimeSpec:
    task_config: str
    task_ids: tuple[int, ...]
    num_trials: int
    action_horizon: int
    replan_steps: int
    inference_steps: int


@dataclass(frozen=True)
class ConditionInspection:
    state: str
    completed_task_ids: tuple[int, ...]
    detail: str


@dataclass
class LiveProcess:
    condition_index: int
    condition: DiagnosisCondition
    process: subprocess.Popen
    log_handle: Any
    log_path: Path
    status_path: Path
    status: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the three missing ASRE Round-3A factorial cells without DDP."
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--round2-reference-metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument(
        "--task-config",
        choices=("libero_uncond_2cam224_1e-4",),
        default="libero_uncond_2cam224_1e-4",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve_artifact_path(value: Any, *, base_dir: Path, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Valid manifest field {key!r} must be a non-empty path string.")
    expanded = Path(os.path.expanduser(os.path.expandvars(value)))
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact declared by {key!r} is unavailable: {resolved}")
    return resolved


def _validate_sha256(value: Any, *, key: str) -> str:
    normalized = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"Valid manifest field {key!r} is not a SHA256 digest: {value!r}")
    return normalized


def _verify_artifact(path: Path, expected_sha256: str, *, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch for {path}: expected {expected_sha256}, got {actual}."
        )


def _validate_round2_reference_metadata(
    path_arg: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    dataset_stats_path: Path,
    dataset_stats_sha256: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    valid_manifest_path: Path,
    valid_manifest_sha256: str,
    prompt_context_cache_path: Path,
    prompt_context_cache_sha256: str,
) -> tuple[Path, str]:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_arg)))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen Round-2 reference metadata is unavailable: {path}")
    metadata = _read_json(path, label="frozen Round-2 reference metadata")
    expected = {
        "status": "completed",
        "condition_protocol": ROUND2_PROTOCOL,
        "diagnosis_condition": "keep_15_29",
        "enabled_video_retrieval_layers": list(range(15, 30)),
        "disabled_video_layers": list(range(15)),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_stats_path": str(dataset_stats_path),
        "dataset_stats_sha256": dataset_stats_sha256,
        "state_bank_manifest_path": str(source_manifest_path),
        "state_bank_manifest_sha256": source_manifest_sha256,
        "valid_state_bank_manifest_path": str(valid_manifest_path),
        "valid_state_bank_manifest_sha256": valid_manifest_sha256,
        "prompt_context_cache_path": str(prompt_context_cache_path),
        "prompt_context_cache_sha256": prompt_context_cache_sha256,
        "task_suite": TASK_SUITE,
        "task_ids": list(FULL_TASK_IDS),
        "seed": SEED,
        "number_of_trials": FULL_TRIALS,
        "action_horizon": ACTION_HORIZON,
        "number_of_inference_steps": INFERENCE_STEPS,
        "replan_steps": REPLAN_STEPS,
        "compile_action_infer": True,
        "binarize_gripper": True,
        "sigma_shift": None,
        "rand_device": "cpu",
        "text_conditioning_source": "round1_state_bank_prompt_context_cache",
    }
    mismatches = _metadata_mismatches(metadata, expected)
    if mismatches:
        raise ValueError(
            "Frozen Round-2 reference metadata is incompatible with Round 3A: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    if re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("git_commit_hash", ""))) is None:
        raise ValueError(f"Frozen Round-2 metadata has an invalid Git commit: {path}")
    return path, sha256_file(path)


def _load_provenance(
    valid_manifest_arg: Path,
    checkpoint_arg: str,
    dataset_stats_arg: str,
    round2_reference_metadata_arg: Path,
) -> Provenance:
    valid_manifest_path = Path(
        os.path.expanduser(os.path.expandvars(str(valid_manifest_arg)))
    ).resolve()
    if not valid_manifest_path.is_file():
        raise FileNotFoundError(f"QC valid manifest is unavailable: {valid_manifest_path}")
    payload = _read_json(valid_manifest_path, label="QC valid manifest")
    required = (
        "checkpoint_path",
        "checkpoint_sha256",
        "dataset_stats_path",
        "dataset_stats_sha256",
        "source_manifest_path",
        "source_manifest_sha256",
        "prompt_context_cache_path",
        "prompt_context_cache_sha256",
        "valid_sample_ids",
        "qc_rule",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"QC valid manifest is missing required fields: {missing}")

    base_dir = valid_manifest_path.parent
    checkpoint_path = _resolve_artifact_path(
        payload["checkpoint_path"], base_dir=base_dir, key="checkpoint_path"
    )
    dataset_stats_path = _resolve_artifact_path(
        payload["dataset_stats_path"], base_dir=base_dir, key="dataset_stats_path"
    )
    source_manifest_path = _resolve_artifact_path(
        payload["source_manifest_path"], base_dir=base_dir, key="source_manifest_path"
    )
    prompt_context_cache_path = _resolve_artifact_path(
        payload["prompt_context_cache_path"],
        base_dir=base_dir,
        key="prompt_context_cache_path",
    )

    checkpoint_cli = Path(
        os.path.expanduser(os.path.expandvars(str(checkpoint_arg)))
    ).resolve()
    stats_cli = Path(
        os.path.expanduser(os.path.expandvars(str(dataset_stats_arg)))
    ).resolve()
    if checkpoint_cli != checkpoint_path:
        raise ValueError(
            "--checkpoint disagrees with the QC valid manifest: "
            f"CLI={checkpoint_cli}, manifest={checkpoint_path}."
        )
    if stats_cli != dataset_stats_path:
        raise ValueError(
            "--dataset-stats-path disagrees with the QC valid manifest: "
            f"CLI={stats_cli}, manifest={dataset_stats_path}."
        )

    checkpoint_sha256 = _validate_sha256(
        payload["checkpoint_sha256"], key="checkpoint_sha256"
    )
    dataset_stats_sha256 = _validate_sha256(
        payload["dataset_stats_sha256"], key="dataset_stats_sha256"
    )
    source_manifest_sha256 = _validate_sha256(
        payload["source_manifest_sha256"], key="source_manifest_sha256"
    )
    prompt_context_cache_sha256 = _validate_sha256(
        payload["prompt_context_cache_sha256"], key="prompt_context_cache_sha256"
    )

    print("Validating immutable Round-3A artifact hashes...", flush=True)
    _verify_artifact(checkpoint_path, checkpoint_sha256, label="Checkpoint")
    _verify_artifact(dataset_stats_path, dataset_stats_sha256, label="Dataset stats")
    _verify_artifact(source_manifest_path, source_manifest_sha256, label="Source manifest")
    _verify_artifact(
        prompt_context_cache_path,
        prompt_context_cache_sha256,
        label="Prompt-context cache",
    )

    source_records = load_manifest(source_manifest_path)
    validate_existing_artifacts(
        valid_manifest_path=valid_manifest_path,
        prompt_cache_path=prompt_context_cache_path,
        source_manifest_path=source_manifest_path,
        source_records=source_records,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        # The 12-GB checkpoint was hashed immediately above.
        verify_checkpoint_hash=False,
    )
    valid_sample_ids = _validate_sample_partition(payload, source_records)
    manifest_expected = {
        "artifact_type": "asre_round2_valid_state_bank_manifest",
        "schema_version": QC_SCHEMA_VERSION,
        "num_source_samples": EXPECTED_SOURCE_SAMPLES,
        "num_valid_samples": EXPECTED_SOURCE_SAMPLES - 1,
        "num_excluded_samples": 1,
        "qc_rule": QC_RULE,
    }
    manifest_mismatches = _metadata_mismatches(payload, manifest_expected)
    if len(source_records) != EXPECTED_SOURCE_SAMPLES:
        manifest_mismatches["source_record_count"] = {
            "existing": len(source_records),
            "requested": EXPECTED_SOURCE_SAMPLES,
        }
    if len(valid_sample_ids) != EXPECTED_SOURCE_SAMPLES - 1:
        manifest_mismatches["valid_sample_count"] = {
            "existing": len(valid_sample_ids),
            "requested": EXPECTED_SOURCE_SAMPLES - 1,
        }
    if manifest_mismatches:
        raise ValueError(
            "Round 3A requires the exact Round-2 500-to-499 QC population: "
            f"{json.dumps(manifest_mismatches, sort_keys=True)}"
        )

    valid_manifest_sha256 = sha256_file(valid_manifest_path)
    reference_path, reference_sha256 = _validate_round2_reference_metadata(
        round2_reference_metadata_arg,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        dataset_stats_path=dataset_stats_path,
        dataset_stats_sha256=dataset_stats_sha256,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        valid_manifest_path=valid_manifest_path,
        valid_manifest_sha256=valid_manifest_sha256,
        prompt_context_cache_path=prompt_context_cache_path,
        prompt_context_cache_sha256=prompt_context_cache_sha256,
    )

    return Provenance(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        dataset_stats_path=dataset_stats_path,
        dataset_stats_sha256=dataset_stats_sha256,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        valid_manifest_path=valid_manifest_path,
        valid_manifest_sha256=valid_manifest_sha256,
        prompt_context_cache_path=prompt_context_cache_path,
        prompt_context_cache_sha256=prompt_context_cache_sha256,
        valid_sample_count=len(valid_sample_ids),
        qc_rule=payload["qc_rule"],
        round2_reference_metadata_path=reference_path,
        round2_reference_metadata_sha256=reference_sha256,
    )


def _resolve_python(value: str) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(value))
    candidate = shutil.which(expanded) if os.sep not in expanded else expanded
    if candidate is None:
        raise FileNotFoundError(f"Python executable is unavailable: {value}")
    path = Path(candidate).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Python executable is unavailable: {path}")
    return path


def _require_clean_worktree() -> None:
    """Fail before creating outputs when HEAD does not identify the run code."""

    try:
        output = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                ".",
                ":(exclude)asre_results/round3a/aggregate/**",
            ],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot inspect the Git worktree before launch: {exc}") from exc
    if output.strip():
        raise RuntimeError(
            "ASRE Round 3A requires a clean code worktree so run metadata identifies "
            "the exact code. Commit the Round-3A implementation first; only generated "
            "asre_results/round3a/aggregate artifacts are exempt. The launcher will "
            "not commit or discard changes automatically.\n"
            f"git status --porcelain:\n{output.rstrip()}"
        )


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _validate_round3a_output_scope(
    output_root: Path, *, valid_manifest_path: Path
) -> None:
    """Keep all Round-3A writes outside frozen Round-1/2 result trees."""

    output_root = output_root.resolve()
    valid_manifest_path = valid_manifest_path.resolve()
    if not valid_manifest_path.is_file():
        raise FileNotFoundError(f"QC valid manifest is unavailable: {valid_manifest_path}")
    manifest = _read_json(valid_manifest_path, label="QC valid manifest")
    source_manifest = _resolve_artifact_path(
        manifest.get("source_manifest_path"),
        base_dir=valid_manifest_path.parent,
        key="source_manifest_path",
    )
    round2_root = valid_manifest_path.parent.resolve()
    round1_root = source_manifest.parent.parent.resolve()
    protected = {round2_root, source_manifest.parent.resolve()}
    for name in ("offline", "online_smoke", "online_full", "aggregate"):
        candidate = (round1_root / name).resolve()
        if candidate.exists():
            protected.add(candidate)
    overlaps = sorted(str(path) for path in protected if _paths_overlap(output_root, path))
    if overlaps:
        raise DirectorySafetyError(
            "Round-3A output root overlaps frozen Round-1/2 results; refusing any "
            f"write: output={output_root}, protected={overlaps}."
        )


def _resolve_runtime(task_config: str, mode: str) -> RuntimeSpec:
    from hydra import compose, initialize_config_dir

    config_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="sim_libero.yaml", overrides=[f"task={task_config}"])

    num_layers = int(cfg.model.action_dit_config.num_layers)
    if num_layers != NUM_LAYERS:
        raise ValueError(
            f"ASRE Round 3A requires {NUM_LAYERS} action layers, task config exposes {num_layers}."
        )
    action_horizon_value = cfg.EVALUATION.get("action_horizon")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_value is None
        else int(action_horizon_value)
    )
    inference_value = cfg.EVALUATION.get("num_inference_steps")
    inference_steps = (
        int(cfg.get("eval_num_inference_steps", INFERENCE_STEPS))
        if inference_value is None
        else int(inference_value)
    )
    replan_steps = int(cfg.EVALUATION.get("replan_steps", REPLAN_STEPS))
    observed = (action_horizon, inference_steps, replan_steps)
    required = (ACTION_HORIZON, INFERENCE_STEPS, REPLAN_STEPS)
    if observed != required:
        raise ValueError(
            "Task config does not match the pre-registered Round-3A runtime "
            f"(action_horizon, inference_steps, replan_steps): {observed} != {required}."
        )
    runtime_controls = {
        "compile_action_infer": bool(cfg.EVALUATION.get("compile_action_infer", False)),
        "binarize_gripper": bool(cfg.EVALUATION.get("binarize_gripper", False)),
        "sigma_shift": cfg.EVALUATION.get("sigma_shift"),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
    }
    required_controls = {
        "compile_action_infer": True,
        "binarize_gripper": True,
        "sigma_shift": None,
        "rand_device": "cpu",
    }
    if runtime_controls != required_controls:
        raise ValueError(
            "Task config does not match the frozen Round-2 inference controls: "
            f"observed={runtime_controls}, required={required_controls}."
        )

    return RuntimeSpec(
        task_config=task_config,
        task_ids=SMOKE_TASK_IDS if mode == "smoke" else FULL_TASK_IDS,
        num_trials=SMOKE_TRIALS if mode == "smoke" else FULL_TRIALS,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        inference_steps=inference_steps,
    )


def _condition_indices(mode: str) -> tuple[int, ...]:
    return SMOKE_CONDITION_INDICES if mode == "smoke" else tuple(range(3))


def _expected_metadata(
    condition: DiagnosisCondition,
    runtime: RuntimeSpec,
    provenance: Provenance,
) -> dict[str, Any]:
    return {
        "git_commit_hash": git_commit(project_root),
        "checkpoint_path": str(provenance.checkpoint_path),
        "dataset_stats_path": str(provenance.dataset_stats_path),
        "diagnosis_condition": condition.name,
        "condition_protocol": ROUND3A_PROTOCOL,
        "enabled_video_retrieval_layers": list(
            condition.enabled_video_retrieval_layers(NUM_LAYERS)
        ),
        "disabled_video_layers": list(condition.disabled_video_layers),
        "num_model_layers": NUM_LAYERS,
        "task_suite": TASK_SUITE,
        "task_ids": list(runtime.task_ids),
        "seed": SEED,
        "number_of_trials": runtime.num_trials,
        "action_horizon": runtime.action_horizon,
        "number_of_inference_steps": runtime.inference_steps,
        "replan_steps": runtime.replan_steps,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "dataset_stats_sha256": provenance.dataset_stats_sha256,
        "state_bank_manifest_path": str(provenance.source_manifest_path),
        "state_bank_manifest_sha256": provenance.source_manifest_sha256,
        "valid_state_bank_manifest_path": str(provenance.valid_manifest_path),
        "valid_state_bank_manifest_sha256": provenance.valid_manifest_sha256,
        "prompt_context_cache_path": str(provenance.prompt_context_cache_path),
        "prompt_context_cache_sha256": provenance.prompt_context_cache_sha256,
    }


def _metadata_mismatches(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"existing": actual.get(key), "requested": value}
        for key, value in expected.items()
        if actual.get(key) != value
    }


def _recorded_live_pid(status_payload: Mapping[str, Any]) -> int | None:
    """Return a still-live PID from an unfinished launcher attempt, if any.

    A launcher killed with SIGKILL cannot clean up its children.  Treating that
    directory as immediately resumable could then put two model processes on
    the same physical GPU.  PID reuse is deliberately handled conservatively:
    if the recorded PID exists, a human must first establish that it is stale.
    """

    if status_payload.get("status") not in {"launching", "running"}:
        return None
    attempts = status_payload.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    latest = attempts[-1]
    if not isinstance(latest, dict) or latest.get("status") not in {
        "launching",
        "running",
    }:
        return None
    pid = latest.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        # A process that exists under a different owner is still unsafe to
        # ignore: the numeric PID may have been reused.
        return pid
    return pid


@contextmanager
def _exclusive_launcher_lock(output_root: Path) -> Iterator[int]:
    """Hold one non-blocking lock for the complete launcher lifecycle.

    The descriptor is also inherited by every evaluation child.  Consequently
    an uncatchable parent SIGKILL does not open a race for a second launcher
    while orphaned GPU jobs are still running.
    """

    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}.launcher.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "owner metadata unavailable"
            raise DirectorySafetyError(
                f"Another launcher or inherited evaluation child holds {lock_path}: "
                f"{owner}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "start_timestamp": now_iso(),
                "output_root": str(output_root),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield handle.fileno()
    finally:
        # Do not call LOCK_UN here.  Evaluation children inherit this same open
        # file description; explicitly unlocking it in the parent would also
        # drop their crash-safety lock.  Closing only the parent descriptor
        # keeps the lock until the final inheriting child exits.
        handle.close()


def _validate_task_result(
    path: Path,
    *,
    condition: DiagnosisCondition,
    runtime: RuntimeSpec,
) -> int:
    result = _read_json(path, label="task result")
    task_id = int(result.get("task_id", -1))
    expected_fields = {
        "task_suite": TASK_SUITE,
        "diagnosis_condition": condition.name,
        "condition_protocol": ROUND3A_PROTOCOL,
        "enabled_video_retrieval_layers": list(
            condition.enabled_video_retrieval_layers(NUM_LAYERS)
        ),
        "disabled_video_layers": list(condition.disabled_video_layers),
        "total_episodes": runtime.num_trials,
    }
    mismatches = _metadata_mismatches(result, expected_fields)
    if mismatches:
        raise DirectorySafetyError(
            f"Task result is incompatible: {path}: {json.dumps(mismatches, sort_keys=True)}"
        )
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
        raise DirectorySafetyError(f"Incomplete/inconsistent episode outcomes in {path}.")
    return task_id


def _validate_action_array(value: Any, *, path: Path, field: str) -> None:
    if not isinstance(value, list) or len(value) != ACTION_HORIZON:
        raise DirectorySafetyError(
            f"{field} in {path} must have shape ({ACTION_HORIZON}, 7)."
        )
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 7:
            raise DirectorySafetyError(
                f"{field}[{row_index}] in {path} must contain seven action values."
            )
        for column_index, item in enumerate(row):
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                raise DirectorySafetyError(
                    f"Non-finite/non-numeric {field}[{row_index}][{column_index}] "
                    f"in {path}: {item!r}."
                )


def _validate_task_action_traces(
    condition_dir: Path,
    *,
    task_id: int,
    condition: DiagnosisCondition,
    runtime: RuntimeSpec,
) -> None:
    """Validate every completed episode trace, including finite action tensors."""

    trace_dir = condition_dir / TASK_SUITE / "action_traces"
    for episode_id in range(runtime.num_trials):
        path = trace_dir / f"task{task_id}_trial{episode_id}.jsonl"
        if not path.is_file():
            raise DirectorySafetyError(
                f"Completed task {task_id} is missing action trace for trial "
                f"{episode_id}: {path}."
            )
        records: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError(f"line {line_number} is not an object")
                    records.append(record)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise DirectorySafetyError(f"Invalid action trace {path}: {exc}") from exc
        if not records:
            raise DirectorySafetyError(f"Action trace is empty: {path}.")
        for replan_id, record in enumerate(records):
            expected = {
                "task_suite": TASK_SUITE,
                "task_id": task_id,
                "diagnosis_condition": condition.name,
                "episode_id": episode_id,
                "replan_id": replan_id,
                "environment_step": 30 + replan_id * runtime.replan_steps,
                "action_inference_seed": SEED,
            }
            mismatches = _metadata_mismatches(record, expected)
            if mismatches:
                raise DirectorySafetyError(
                    f"Action-trace identity mismatch in {path}: "
                    f"{json.dumps(mismatches, sort_keys=True)}"
                )
            _validate_action_array(record.get("raw_action"), path=path, field="raw_action")
            _validate_action_array(
                record.get("executed_action"), path=path, field="executed_action"
            )


def _inspect_condition_dir(
    condition_dir: Path,
    *,
    condition: DiagnosisCondition,
    runtime: RuntimeSpec,
    provenance: Provenance,
) -> ConditionInspection:
    if not condition_dir.exists():
        return ConditionInspection("missing", (), "directory does not exist")
    if not condition_dir.is_dir():
        raise DirectorySafetyError(f"Condition output path is not a directory: {condition_dir}")
    entries = list(condition_dir.iterdir())
    if not entries:
        return ConditionInspection("missing", (), "directory is empty")

    status_path = condition_dir / STATUS_NAME
    status_payload = None
    if status_path.exists():
        status_payload = _read_json(status_path, label="launcher status")
        status_expected = {
            "condition": condition.name,
            "protocol": ROUND3A_PROTOCOL,
            "physical_gpu": build_round3a_conditions(NUM_LAYERS).index(condition),
        }
        mismatches = _metadata_mismatches(status_payload, status_expected)
        if mismatches:
            raise DirectorySafetyError(
                f"Launcher status is incompatible in {condition_dir}: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        live_pid = _recorded_live_pid(status_payload)
        if live_pid is not None:
            raise DirectorySafetyError(
                "Refusing to resume a condition whose previous launcher child "
                f"still appears alive (pid={live_pid}): {condition_dir}"
            )

    metadata_path = condition_dir / "run_metadata.json"
    if not metadata_path.exists():
        unexpected = [path.name for path in entries if path.name != STATUS_NAME]
        if unexpected:
            raise DirectorySafetyError(
                f"Non-empty condition directory has no run_metadata.json: {condition_dir}; "
                f"entries={sorted(unexpected)}"
            )
        if status_payload is None:
            raise DirectorySafetyError(
                f"Cannot identify non-empty condition directory: {condition_dir}"
            )
        if status_payload.get("status") == "completed":
            raise DirectorySafetyError(
                f"Launcher claims completion but run metadata is absent: {condition_dir}"
            )
        return ConditionInspection("partial", (), "launcher attempt exists; model did not start")

    metadata = _read_json(metadata_path, label="condition run metadata")
    expected_metadata = _expected_metadata(condition, runtime, provenance)
    mismatches = _metadata_mismatches(metadata, expected_metadata)
    if mismatches:
        raise DirectorySafetyError(
            f"Condition metadata is incompatible in {condition_dir}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )

    completed_by_task: dict[int, Path] = {}
    for result_path in sorted(condition_dir.glob("**/gpu*_task*_results.json")):
        task_id = _validate_task_result(
            result_path,
            condition=condition,
            runtime=runtime,
        )
        if task_id in completed_by_task:
            raise DirectorySafetyError(
                f"Duplicate results for task {task_id}: "
                f"{completed_by_task[task_id]} and {result_path}."
            )
        completed_by_task[task_id] = result_path
        _validate_task_action_traces(
            condition_dir,
            task_id=task_id,
            condition=condition,
            runtime=runtime,
        )

    completed_ids = tuple(sorted(completed_by_task))
    metadata_status = str(metadata.get("status", ""))
    all_tasks_complete = set(completed_ids) == set(runtime.task_ids)
    if metadata_status == "completed" and not all_tasks_complete:
        raise DirectorySafetyError(
            f"Metadata claims completion but task results are missing in {condition_dir}: "
            f"completed={list(completed_ids)}, expected={list(runtime.task_ids)}"
        )
    if status_payload is not None and status_payload.get("status") == "completed":
        if metadata_status != "completed" or not all_tasks_complete:
            raise DirectorySafetyError(
                f"Launcher status and run metadata disagree in {condition_dir}."
            )
    if metadata_status == "completed" and all_tasks_complete:
        return ConditionInspection("complete", completed_ids, "all task results validated")
    if metadata_status not in {"", "running", "failed", "interrupted"}:
        raise DirectorySafetyError(
            f"Unsupported run metadata status {metadata_status!r} in {metadata_path}."
        )
    return ConditionInspection(
        "partial",
        completed_ids,
        f"resumable metadata status={metadata_status or 'unset'}",
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
        raise RuntimeError(f"Cannot query the three physical GPUs with nvidia-smi: {exc}") from exc
    records: list[dict[str, Any]] = []
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
                "memory_free_mib_at_preflight": int(fields[6]),
            }
        )
    by_index = {record["index"]: record for record in records}
    missing = [index for index in range(3) if index not in by_index]
    if missing:
        raise RuntimeError(f"Round 3A requires physical GPUs 0..2; missing {missing}.")
    selected = [by_index[index] for index in range(3)]
    names = {record["name"] for record in selected}
    if len(names) != 1 or not all("A5000" in name for name in names):
        raise RuntimeError(
            "Round 3A is pre-registered for three identical A5000 GPUs; "
            f"observed models={sorted(names)}."
        )
    return selected


def _root_identity(
    *,
    mode: str,
    runtime: RuntimeSpec,
    provenance: Provenance,
    python_path: Path,
    conditions: Sequence[tuple[int, DiagnosisCondition]],
    gpu_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stable_gpu_inventory = [
        {
            key: record[key]
            for key in ("index", "name", "uuid", "pci_bus_id", "driver_version", "memory_total_mib")
        }
        for record in gpu_inventory
    ]
    payload = {
        "schema_version": 1,
        "protocol": ROUND3A_PROTOCOL,
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
        "provenance": provenance.identity_dict(),
        "conditions": [
            {
                "condition_index": index,
                "condition": condition.name,
                "physical_gpu": index,
                "enabled_video_retrieval_layers": list(
                    condition.enabled_video_retrieval_layers(NUM_LAYERS)
                ),
                "disabled_video_layers": list(condition.disabled_video_layers),
            }
            for index, condition in conditions
        ],
        "gpu_inventory": stable_gpu_inventory,
    }
    payload["identity_sha256"] = sha256_json(payload)
    return payload


def _prepare_output_root(output_root: Path, identity: Mapping[str, Any]) -> None:
    config_path = output_root / ROOT_CONFIG_NAME
    if output_root.exists() and not output_root.is_dir():
        raise DirectorySafetyError(f"Output root is not a directory: {output_root}")
    if config_path.exists():
        existing = _read_json(config_path, label="launcher root config")
        if existing != identity:
            raise DirectorySafetyError(
                f"Output root belongs to an incompatible run: {output_root}"
            )
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise DirectorySafetyError(
            f"Refusing non-empty output root without {ROOT_CONFIG_NAME}: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config_path, identity)


def _validate_root_layout(
    output_root: Path,
    conditions: Sequence[tuple[int, DiagnosisCondition]],
) -> None:
    allowed = {
        ROOT_CONFIG_NAME,
        SUMMARY_NAME,
        "logs",
        *(condition.name for _, condition in conditions),
    }
    unexpected = sorted(path.name for path in output_root.iterdir() if path.name not in allowed)
    if unexpected:
        raise DirectorySafetyError(
            f"Round-3A output root contains unexpected entries: {output_root}: {unexpected}"
        )


def _validate_smoke_gate(
    smoke_summary_path: Path | None,
    *,
    provenance: Provenance,
    task_config: str,
) -> None:
    if smoke_summary_path is None:
        raise ValueError("--mode full requires --smoke-summary.")
    path = smoke_summary_path.resolve()
    summary = _read_json(path, label="Round-3A smoke summary")
    expected_names = [
        build_round3a_conditions(NUM_LAYERS)[index].name
        for index in SMOKE_CONDITION_INDICES
    ]
    expected = {
        "mode": "smoke",
        "protocol": ROUND3A_PROTOCOL,
        "all_succeeded": True,
        "task_config": task_config,
        "task_ids": list(SMOKE_TASK_IDS),
        "num_trials": SMOKE_TRIALS,
        "seed": SEED,
        "expected_conditions": expected_names,
        "git_commit_hash": git_commit(project_root),
        "provenance": provenance.identity_dict(),
    }
    mismatches = _metadata_mismatches(summary, expected)
    if mismatches:
        raise ValueError(
            "Smoke summary is incompatible with the requested full run: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    states = summary.get("condition_states")
    if not isinstance(states, dict) or any(
        states.get(name, {}).get("state") != "complete" for name in expected_names
    ):
        raise ValueError(f"Smoke summary does not validate all required conditions: {path}")

    smoke_root_value = summary.get("output_root")
    if not isinstance(smoke_root_value, str):
        raise ValueError(f"Smoke summary lacks output_root: {path}")
    smoke_root = Path(smoke_root_value).resolve()
    smoke_runtime = RuntimeSpec(
        task_config=task_config,
        task_ids=SMOKE_TASK_IDS,
        num_trials=SMOKE_TRIALS,
        action_horizon=ACTION_HORIZON,
        replan_steps=REPLAN_STEPS,
        inference_steps=INFERENCE_STEPS,
    )
    conditions = build_round3a_conditions(NUM_LAYERS)
    for index in SMOKE_CONDITION_INDICES:
        inspection = _inspect_condition_dir(
            smoke_root / conditions[index].name,
            condition=conditions[index],
            runtime=smoke_runtime,
            provenance=provenance,
        )
        if inspection.state != "complete":
            raise ValueError(
                f"Smoke condition is no longer complete: {conditions[index].name}: "
                f"{inspection.detail}"
            )


def _condition_command(
    *,
    python_path: Path,
    condition_index: int,
    condition: DiagnosisCondition,
    condition_output: Path,
    runtime: RuntimeSpec,
    provenance: Provenance,
) -> list[str]:
    enabled = list(condition.enabled_video_retrieval_layers(NUM_LAYERS))
    disabled = list(condition.disabled_video_layers)
    eval_script = project_root / "experiments" / "libero" / "eval_libero_single.py"
    compact = lambda value: json.dumps(value, separators=(",", ":"))
    return [
        str(python_path),
        str(eval_script),
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
        "ASRE_DIAGNOSIS.mode=drop_video_kv",
        f"ASRE_DIAGNOSIS.protocol={ROUND3A_PROTOCOL}",
        f"ASRE_DIAGNOSIS.condition_index={condition_index}",
        f"ASRE_DIAGNOSIS.condition_name={condition.name}",
        f"ASRE_DIAGNOSIS.enabled_video_retrieval_layers={compact(enabled)}",
        f"ASRE_DIAGNOSIS.disabled_video_layers={compact(disabled)}",
        "ASRE_DIAGNOSIS.save_rollout_video=false",
        f"ASRE_DIAGNOSIS.checkpoint_sha256={provenance.checkpoint_sha256}",
        f"ASRE_DIAGNOSIS.dataset_stats_sha256={provenance.dataset_stats_sha256}",
        f"ASRE_DIAGNOSIS.state_bank_manifest_path={provenance.source_manifest_path}",
        f"ASRE_DIAGNOSIS.state_bank_manifest_sha256={provenance.source_manifest_sha256}",
        f"ASRE_DIAGNOSIS.valid_state_bank_manifest_path={provenance.valid_manifest_path}",
        f"ASRE_DIAGNOSIS.valid_state_bank_manifest_sha256={provenance.valid_manifest_sha256}",
        f"ASRE_DIAGNOSIS.prompt_context_cache_sha256={provenance.prompt_context_cache_sha256}",
    ]


def _child_environment(physical_gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    for variable in (
        "LIBERO_WORKER_MODE",
        "LIBERO_WORKER_ID",
        "LIBERO_WORKER_PENDING_FILE",
        "LIBERO_WORKER_LOCK_FILE",
        "LIBERO_WORKER_STATUS_DIR",
        "LIBERO_WORKER_FAILED_FILE",
        "LIBERO_WORKER_STOP_FILE",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(variable, None)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            # robosuite indexes eglQueryDevicesEXT globally, not relative to CVD.
            "MUJOCO_EGL_DEVICE_ID": str(physical_gpu),
        }
    )
    libero_root = Path(
        environment.get("LIBERO_ROOT", str(project_root.parent / "LIBERO"))
    ).resolve()
    python_entries = [
        str(project_root / "src"),
        str(project_root),
        str(libero_root),
    ]
    if environment.get("PYTHONPATH"):
        python_entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_entries)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["HYDRA_FULL_ERROR"] = "1"
    return environment


def _next_attempt(status_payload: Mapping[str, Any] | None, logs_dir: Path, name: str) -> int:
    attempts = [] if status_payload is None else status_payload.get("attempts", [])
    attempt = len(attempts) + 1 if isinstance(attempts, list) else 1
    while (logs_dir / f"{name}.attempt{attempt:02d}.log").exists():
        attempt += 1
    return attempt


def _launch_condition(
    *,
    condition_index: int,
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
    existing_status = (
        _read_json(status_path, label="launcher status") if status_path.exists() else None
    )
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    attempt_number = _next_attempt(existing_status, logs_dir, condition.name)
    log_path = logs_dir / f"{condition.name}.attempt{attempt_number:02d}.log"
    command = _condition_command(
        python_path=python_path,
        condition_index=condition_index,
        condition=condition,
        condition_output=condition_output,
        runtime=runtime,
        provenance=provenance,
    )
    attempts = []
    if existing_status is not None and isinstance(existing_status.get("attempts"), list):
        attempts = list(existing_status["attempts"])
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
    status: dict[str, Any] = {
        "schema_version": 1,
        "protocol": ROUND3A_PROTOCOL,
        "mode": mode,
        "condition": condition.name,
        "condition_index": condition_index,
        "enabled_video_retrieval_layers": list(
            condition.enabled_video_retrieval_layers(NUM_LAYERS)
        ),
        "disabled_video_layers": list(condition.disabled_video_layers),
        "physical_gpu": condition_index,
        "gpu": dict(gpu_record),
        "cuda_visible_devices": str(condition_index),
        "model_device": "cuda:0",
        "mujoco_egl_device_id": str(condition_index),
        "output_dir": str(condition_output),
        "status": "launching",
        "attempts": attempts,
    }
    atomic_write_json(status_path, status)

    log_handle = log_path.open("x", encoding="utf-8")
    log_handle.write(
        f"[{now_iso()}] condition={condition.name} physical_gpu={condition_index} "
        f"CUDA_VISIBLE_DEVICES={condition_index} model=cuda:0 "
        f"MUJOCO_EGL_DEVICE_ID={condition_index}\n"
    )
    log_handle.write(shlex.join(command) + "\n")
    log_handle.flush()
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=_child_environment(condition_index),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(launcher_lock_fd,),
        )
        attempt["pid"] = process.pid
        attempt["status"] = "running"
        status["status"] = "running"
        atomic_write_json(status_path, status)
        return LiveProcess(
            condition_index=condition_index,
            condition=condition,
            process=process,
            log_handle=log_handle,
            log_path=log_path,
            status_path=status_path,
            status=status,
        )
    except BaseException as exc:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)
        log_handle.close()
        attempt["status"] = "launch_failed"
        attempt["end_timestamp"] = now_iso()
        attempt["error"] = repr(exc)
        status["status"] = "failed"
        atomic_write_json(status_path, status)
        raise


def _finish_live_process(live: LiveProcess, exit_status: int) -> None:
    if not live.log_handle.closed:
        live.log_handle.close()
    attempt = live.status["attempts"][-1]
    attempt["exit_status"] = int(exit_status)
    attempt["end_timestamp"] = now_iso()
    attempt["status"] = "completed" if exit_status == 0 else "failed"
    live.status["status"] = attempt["status"]
    atomic_write_json(live.status_path, live.status)


def _mark_validation_failure(live: LiveProcess, error: BaseException) -> None:
    attempt = live.status["attempts"][-1]
    attempt["status"] = "validation_failed"
    attempt["validation_error"] = str(error)
    live.status["status"] = "validation_failed"
    atomic_write_json(live.status_path, live.status)


def _terminate_live_process(live: LiveProcess, *, reason: str) -> None:
    process = live.process
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=15)
    if not live.log_handle.closed:
        live.log_handle.close()
    attempt = live.status["attempts"][-1]
    attempt["exit_status"] = process.returncode
    attempt["end_timestamp"] = now_iso()
    attempt["status"] = reason
    live.status["status"] = reason
    atomic_write_json(live.status_path, live.status)


def _collect_final_states(
    *,
    output_root: Path,
    conditions: Sequence[tuple[int, DiagnosisCondition]],
    runtime: RuntimeSpec,
    provenance: Provenance,
) -> tuple[dict[str, dict[str, Any]], bool]:
    states: dict[str, dict[str, Any]] = {}
    all_complete = True
    for index, condition in conditions:
        try:
            inspection = _inspect_condition_dir(
                output_root / condition.name,
                condition=condition,
                runtime=runtime,
                provenance=provenance,
            )
            states[condition.name] = {
                "state": inspection.state,
                "completed_task_ids": list(inspection.completed_task_ids),
                "detail": inspection.detail,
                "physical_gpu": index,
            }
            all_complete = all_complete and inspection.state == "complete"
        except Exception as exc:
            states[condition.name] = {
                "state": "invalid",
                "completed_task_ids": [],
                "detail": str(exc),
                "physical_gpu": index,
            }
            all_complete = False
    return states, all_complete


def _write_summary(
    *,
    output_root: Path,
    mode: str,
    runtime: RuntimeSpec,
    provenance: Provenance,
    conditions: Sequence[tuple[int, DiagnosisCondition]],
    condition_states: Mapping[str, Any],
    all_succeeded: bool,
    start_timestamp: str,
    interrupted: bool,
) -> Path:
    summary = {
        "schema_version": 1,
        "protocol": ROUND3A_PROTOCOL,
        "mode": mode,
        "all_succeeded": bool(all_succeeded),
        "interrupted": bool(interrupted),
        "output_root": str(output_root),
        "task_config": runtime.task_config,
        "task_ids": list(runtime.task_ids),
        "num_trials": runtime.num_trials,
        "seed": SEED,
        "git_commit_hash": git_commit(project_root),
        "provenance": provenance.identity_dict(),
        "expected_conditions": [condition.name for _, condition in conditions],
        "condition_states": dict(condition_states),
        "start_timestamp": start_timestamp,
        "end_timestamp": now_iso(),
    }
    summary_path = output_root / SUMMARY_NAME
    atomic_write_json(summary_path, summary)
    return summary_path


def _validate_completed_summary(
    path: Path,
    *,
    mode: str,
    runtime: RuntimeSpec,
    provenance: Provenance,
    conditions: Sequence[tuple[int, DiagnosisCondition]],
) -> None:
    """Validate a completed launcher summary without rewriting it on a no-op rerun."""

    summary = _read_json(path, label="completed launcher summary")
    expected_names = [condition.name for _, condition in conditions]
    expected = {
        "schema_version": 1,
        "protocol": ROUND3A_PROTOCOL,
        "mode": mode,
        "all_succeeded": True,
        "interrupted": False,
        "task_config": runtime.task_config,
        "task_ids": list(runtime.task_ids),
        "num_trials": runtime.num_trials,
        "seed": SEED,
        "git_commit_hash": git_commit(project_root),
        "provenance": provenance.identity_dict(),
        "expected_conditions": expected_names,
    }
    mismatches = _metadata_mismatches(summary, expected)
    states = summary.get("condition_states")
    if not isinstance(states, dict) or set(states) != set(expected_names):
        mismatches["condition_states"] = {
            "existing": states,
            "requested": f"exact completed states for {expected_names}",
        }
    elif any(
        not isinstance(states[name], dict) or states[name].get("state") != "complete"
        for name in expected_names
    ):
        mismatches["condition_states"] = {
            "existing": states,
            "requested": "all states complete",
        }
    if mismatches:
        raise DirectorySafetyError(
            "Completed condition directories have an incompatible launcher summary; "
            f"refusing to rewrite it: {json.dumps(mismatches, sort_keys=True)}"
        )


def _main_locked(
    args: argparse.Namespace,
    *,
    output_root: Path,
    launcher_lock_fd: int,
) -> None:
    start_timestamp = now_iso()
    python_path = _resolve_python(args.python)
    provenance = _load_provenance(
        args.valid_manifest,
        args.checkpoint,
        args.dataset_stats_path,
        args.round2_reference_metadata,
    )
    runtime = _resolve_runtime(args.task_config, args.mode)
    all_conditions = build_round3a_conditions(NUM_LAYERS)
    condition_pairs = [
        (index, all_conditions[index]) for index in _condition_indices(args.mode)
    ]
    if args.mode == "full":
        _validate_smoke_gate(
            args.smoke_summary,
            provenance=provenance,
            task_config=args.task_config,
        )
    elif args.smoke_summary is not None:
        raise ValueError("--smoke-summary is only valid with --mode full.")

    gpu_inventory = _gpu_inventory()
    identity = _root_identity(
        mode=args.mode,
        runtime=runtime,
        provenance=provenance,
        python_path=python_path,
        conditions=condition_pairs,
        gpu_inventory=gpu_inventory,
    )
    _prepare_output_root(output_root, identity)
    _validate_root_layout(output_root, condition_pairs)

    preflight: dict[int, ConditionInspection] = {}
    for index, condition in condition_pairs:
        inspection = _inspect_condition_dir(
            output_root / condition.name,
            condition=condition,
            runtime=runtime,
            provenance=provenance,
        )
        preflight[index] = inspection
        if inspection.state == "complete":
            print(
                f"Skipping completed {condition.name}; condition directory remains untouched.",
                flush=True,
            )
        else:
            print(
                f"Preparing {condition.name} on physical GPU {index}: "
                f"{inspection.state} ({inspection.detail}).",
                flush=True,
            )

    if all(inspection.state == "complete" for inspection in preflight.values()):
        summary_path = output_root / SUMMARY_NAME
        if summary_path.exists():
            _validate_completed_summary(
                summary_path,
                mode=args.mode,
                runtime=runtime,
                provenance=provenance,
                conditions=condition_pairs,
            )
            print(
                f"All conditions and the existing summary are complete; no files changed: "
                f"{summary_path}",
                flush=True,
            )
            return
        condition_states, all_succeeded = _collect_final_states(
            output_root=output_root,
            conditions=condition_pairs,
            runtime=runtime,
            provenance=provenance,
        )
        summary_path = _write_summary(
            output_root=output_root,
            mode=args.mode,
            runtime=runtime,
            provenance=provenance,
            conditions=condition_pairs,
            condition_states=condition_states,
            all_succeeded=all_succeeded,
            start_timestamp=start_timestamp,
            interrupted=False,
        )
        print(f"Launcher summary: {summary_path}", flush=True)
        return

    live_processes: list[LiveProcess] = []
    interrupted = False
    launch_error: BaseException | None = None
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        for index, condition in condition_pairs:
            if preflight[index].state == "complete":
                continue
            live = _launch_condition(
                condition_index=index,
                condition=condition,
                output_root=output_root,
                python_path=python_path,
                runtime=runtime,
                provenance=provenance,
                mode=args.mode,
                gpu_record=gpu_inventory[index],
                launcher_lock_fd=launcher_lock_fd,
            )
            live_processes.append(live)
            print(
                f"Launched {condition.name} on physical GPU {index}; log={live.log_path}",
                flush=True,
            )

        for live in live_processes:
            exit_status = live.process.wait()
            _finish_live_process(live, exit_status)
            if exit_status == 0:
                try:
                    inspection = _inspect_condition_dir(
                        output_root / live.condition.name,
                        condition=live.condition,
                        runtime=runtime,
                        provenance=provenance,
                    )
                    if inspection.state != "complete":
                        raise DirectorySafetyError(
                            f"Process exited successfully but {live.condition.name} is "
                            f"{inspection.state}: {inspection.detail}"
                        )
                except BaseException as exc:
                    _mark_validation_failure(live, exc)
            print(
                f"{live.condition.name}: exit={exit_status} log={live.log_path}",
                flush=True,
            )
    except KeyboardInterrupt as exc:
        interrupted = True
        launch_error = exc
        print("Interrupt received; terminating all active Round-3A children.", file=sys.stderr)
        for live in live_processes:
            if live.process.poll() is None:
                _terminate_live_process(live, reason="interrupted")
            elif live.status.get("status") in {"launching", "running"}:
                _finish_live_process(live, int(live.process.returncode))
    except BaseException as exc:
        launch_error = exc
        print(f"Launcher error; terminating active children: {exc}", file=sys.stderr)
        for live in live_processes:
            if live.process.poll() is None:
                _terminate_live_process(live, reason="terminated")
            elif live.status.get("status") in {"launching", "running"}:
                _finish_live_process(live, int(live.process.returncode))
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        for live in live_processes:
            if not live.log_handle.closed:
                live.log_handle.close()

    condition_states, all_succeeded = _collect_final_states(
        output_root=output_root,
        conditions=condition_pairs,
        runtime=runtime,
        provenance=provenance,
    )
    all_succeeded = all_succeeded and launch_error is None
    summary_path = _write_summary(
        output_root=output_root,
        mode=args.mode,
        runtime=runtime,
        provenance=provenance,
        conditions=condition_pairs,
        condition_states=condition_states,
        all_succeeded=all_succeeded,
        start_timestamp=start_timestamp,
        interrupted=interrupted,
    )
    print(f"Launcher summary: {summary_path}", flush=True)

    if interrupted:
        raise SystemExit(130)
    if launch_error is not None:
        raise launch_error
    if not all_succeeded:
        raise SystemExit(1)


def main() -> None:
    args = _parse_args()
    _require_clean_worktree()
    output_root = Path(
        os.path.expanduser(os.path.expandvars(str(args.output_root)))
    ).resolve()
    _validate_round3a_output_scope(
        output_root, valid_manifest_path=args.valid_manifest
    )
    with _exclusive_launcher_lock(output_root) as launcher_lock_fd:
        _main_locked(
            args,
            output_root=output_root,
            launcher_lock_fd=launcher_lock_fd,
        )


if __name__ == "__main__":
    main()
