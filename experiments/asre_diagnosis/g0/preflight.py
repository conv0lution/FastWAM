"""Fail-closed preflight for the ASRE G0 cross-suite gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import atomic_write_json, now_iso, sha256_file
from experiments.asre_diagnosis.g0.definitions import (
    CHECKPOINT_NAME,
    NUM_STEPS_WAIT,
    NUM_TRIALS,
    PROTECTED_STAGE1_DIRS,
    ROUND3A_COMMIT,
    ROUND3A_TAG,
    ROUND3B_COMMIT,
    ROUND3B_TAG,
    SEED,
    SUITE_ORDER,
    TASK_CONFIG,
    TASK_IDS,
    assert_output_scope,
    condition_registration_payload,
)


REFERENCE_PATHS = (
    "asre_results/round2/aggregate/summary.csv",
    "asre_results/round2/aggregate/task_success_rates.csv",
    "asre_results/round3b/aggregate/online_condition_summary.csv",
    "asre_results/round3b/aggregate/task_success.csv",
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=project_root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _tag_commit(tag: str) -> str:
    return _git("rev-list", "-n", "1", tag)


def validate_git_state(*, allow_dirty: bool) -> dict[str, Any]:
    tags = sorted(
        tag for tag in _git("tag", "--list").splitlines() if "asre" in tag.lower()
    )
    expected = {ROUND3A_TAG: ROUND3A_COMMIT, ROUND3B_TAG: ROUND3B_COMMIT}
    observed = {tag: _tag_commit(tag) for tag in expected}
    if observed != expected:
        raise ValueError(f"Frozen ASRE tag mismatch: observed={observed}, expected={expected}.")
    status = _git("status", "--porcelain=v1", "--untracked-files=normal")
    protected_status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *PROTECTED_STAGE1_DIRS,
    )
    if protected_status:
        raise RuntimeError(
            "Frozen Round-1/2/3A/3B artifact trees are modified:\n" + protected_status
        )
    non_g0_status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
        "--",
        ".",
        ":(exclude)asre_results/g0_cross_suite/**",
    )
    if non_g0_status and not allow_dirty:
        raise RuntimeError(
            "Formal G0 requires committed source so the recorded Git commit is exact. "
            "Commit the implementation first or use --allow-dirty for development-only "
            "preflight.\n" + non_g0_status
        )
    return {
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "status": status,
        "non_g0_status": non_g0_status,
        "allow_dirty": bool(allow_dirty),
        "asre_tags": tags,
        "round3a_parent_tag": ROUND3A_TAG,
        "round3a_parent_commit": observed[ROUND3A_TAG],
        "round3b_parent_tag": ROUND3B_TAG,
        "round3b_parent_commit": observed[ROUND3B_TAG],
        "protected_stage1_status": protected_status,
    }


def _resolve_suite_inventory() -> list[dict[str, Any]]:
    import torch
    from libero.libero import benchmark, get_libero_path

    registered = benchmark.get_benchmark_dict()
    missing = [suite for suite in SUITE_ORDER if suite not in registered]
    if missing:
        raise ValueError(f"Repository does not register required G0 suites: {missing}.")
    rows: list[dict[str, Any]] = []
    for suite_name in SUITE_ORDER:
        suite = registered[suite_name]()
        if int(suite.n_tasks) != len(TASK_IDS):
            raise ValueError(
                f"G0 expects 10 tasks in {suite_name}, observed {suite.n_tasks}."
            )
        tasks: list[dict[str, Any]] = []
        for task_id in TASK_IDS:
            task = suite.get_task(task_id)
            state_path = (
                Path(get_libero_path("init_states"))
                / task.problem_folder
                / task.init_states_file
            ).resolve()
            states = torch.load(state_path, map_location="cpu", weights_only=False)
            if len(states) < NUM_TRIALS:
                raise ValueError(
                    f"{suite_name} task {task_id} has {len(states)} initial states; "
                    f"G0 requires {NUM_TRIALS}."
                )
            tasks.append(
                {
                    "task_id": task_id,
                    "task_name": str(task.name),
                    "task_language": str(task.language),
                    "problem_folder": str(task.problem_folder),
                    "init_states_path": str(state_path),
                    "available_initial_states": len(states),
                    "paired_trial_ids": list(range(NUM_TRIALS)),
                }
            )
        rows.append(
            {
                "suite_identifier": suite_name,
                "num_tasks": int(suite.n_tasks),
                "tasks": tasks,
            }
        )
    return rows


def _validate_file(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def run_preflight(
    *,
    checkpoint: Path,
    dataset_stats: Path,
    output_root: Path,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    assert_output_scope(output_root, project_root)
    if checkpoint.name != CHECKPOINT_NAME:
        raise ValueError(
            f"G0 requires checkpoint {CHECKPOINT_NAME!r}, got {checkpoint.name!r}."
        )
    references = []
    for relative in REFERENCE_PATHS:
        references.append(
            {"relative_path": relative, **_validate_file(project_root / relative, label=relative)}
        )
    report = {
        "artifact_type": "asre_g0_preflight_report",
        "schema_version": 1,
        "status": "compatible",
        "created_at": now_iso(),
        "git": validate_git_state(allow_dirty=allow_dirty),
        "checkpoint": _validate_file(checkpoint, label="Fast-WAM LIBERO checkpoint"),
        "dataset_statistics": _validate_file(
            dataset_stats, label="LIBERO dataset statistics"
        ),
        "suite_inventory": _resolve_suite_inventory(),
        "condition_registration": condition_registration_payload(),
        "protocol": {
            "task_config": TASK_CONFIG,
            "seed": SEED,
            "num_trials": NUM_TRIALS,
            "num_steps_wait": NUM_STEPS_WAIT,
            "suite_order": list(SUITE_ORDER),
            "suite_execution": "sequential",
            "condition_execution": "four_parallel_non_ddp_workers",
            "prompt_context_preparation": (
                "one_two_gpu_cuda_prestage_then_shared_suite_cache"
            ),
            "rollout_worker_load_text_encoder": False,
        },
        "frozen_spatial_reference_files": references,
        "protected_stage1_paths": list(PROTECTED_STAGE1_DIRS),
        "output_root": str(output_root.expanduser().resolve()),
    }
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = run_preflight(
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        output_root=args.output_root,
        allow_dirty=args.allow_dirty,
    )
    output = args.output.expanduser().resolve()
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
