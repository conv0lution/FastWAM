"""Drive the authorized Round-4B cumulative-energy follow-up and stop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4B_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    EXPECTED_FEATURE_DIM,
    MAX_RANK,
    validate_basis_manifest,
)
from experiments.asre_diagnosis.round4b.energy_curve_definitions import (  # noqa: E402
    COLLECT_STAGE,
    RECOVER_STAGE,
)


FOLLOWUP_START_COMMIT = "134ff98002e9da59e960cfd289dc31c9a3b15c2d"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _run(stage: str, command: Sequence[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    print(f"[Round4B Energy] Starting {stage}; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"Round-4B energy stage {stage} exited {process.returncode}; inspect {log_path}."
        )
    print(f"[Round4B Energy] Completed {stage}", flush=True)


def _preflight(round4b_root: Path, output: Path) -> dict[str, Any]:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", FOLLOWUP_START_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
    )
    dirty = _git(
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        ".",
        f":(exclude){output.relative_to(PROJECT_ROOT)}/**",
    )
    if dirty:
        raise RuntimeError("Formal energy analysis requires committed source code.\n" + dirty)
    old_preflight_path = round4b_root / "preflight_report.json"
    old_preflight = _read(old_preflight_path)
    split_path = round4b_root / "calibration/calibration_split_manifest.json"
    basis_path = round4b_root / "calibration/basis_manifest.json"
    diagnostics_path = round4b_root / "calibration/subspace_diagnostics.json"
    aggregate_path = round4b_root / "aggregate/round4b_summary.json"
    driver_path = round4b_root / "driver_status.json"
    split = _read(split_path)
    basis = validate_basis_manifest(basis_path)
    diagnostics = _read(diagnostics_path)
    aggregate = _read(aggregate_path)
    driver = _read(driver_path)
    if (
        split.get("holdout_sample_count") != 100
        or split.get("holdout_episode_clusters") != 20
        or basis.get("feature_dim") != EXPECTED_FEATURE_DIM
        or basis.get("max_rank") != MAX_RANK
        or diagnostics.get("status") != "complete"
        or diagnostics.get("basis_manifest_sha256") != sha256_file(basis_path)
        or aggregate.get("classification", {}).get("classification") != "STRONG"
        or driver.get("status") != "complete"
        or driver.get("later_stage_launched") is not False
    ):
        raise ValueError("Frozen Round-4B parent artifacts are incomplete or incompatible.")
    return {
        "artifact_type": "asre_round4b_energy_curve_preflight",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "status": "compatible",
        "created_at": now_iso(),
        "current_branch": _git("branch", "--show-current"),
        "starting_commit": FOLLOWUP_START_COMMIT,
        "analysis_commit": git_commit(PROJECT_ROOT),
        "worktree_clean_excluding_analysis_outputs": True,
        "round4b_root": str(round4b_root),
        "source_round4b_commit": old_preflight["git"]["current_head"],
        "feature_dim": EXPECTED_FEATURE_DIM,
        "frozen_prefix_rank": MAX_RANK,
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "basis_manifest_path": str(basis_path),
        "basis_manifest_sha256": sha256_file(basis_path),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "round4b_aggregate_path": str(aggregate_path),
        "round4b_aggregate_sha256": sha256_file(aggregate_path),
        "checkpoint_path": old_preflight["state_bank"]["checkpoint_path"],
        "checkpoint_sha256": old_preflight["state_bank"]["checkpoint_sha256"],
        "donor_mapping_path": old_preflight["donors"]["mapping_path"],
        "donor_mapping_sha256": old_preflight["donors"]["mapping_sha256"],
        "donor_manifest_path": old_preflight["donors"]["observation_manifest_path"],
        "donor_manifest_sha256": old_preflight["donors"]["observation_manifest_sha256"],
        "donor_root": old_preflight["donors"]["observation_root"],
        "heldout_state_count": 100,
        "heldout_episode_clusters": 20,
        "online_episodes_authorized": 0,
        "environment_rollouts_authorized": 0,
        "heldout_svd_refit_authorized": False,
        "cache_only_heldout_gram_pass_authorized": True,
        "complete_eigensystem_recovery_from_frozen_calibration_gram_authorized": True,
    }


def _reuse_stable_preflight(
    existing: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Accept a resume when only the intentionally dynamic timestamp differs."""

    recorded_timestamp = existing.get("created_at")
    if not isinstance(recorded_timestamp, str) or not recorded_timestamp:
        raise RuntimeError("Existing cumulative-energy preflight lacks created_at.")
    comparable = dict(expected)
    comparable["created_at"] = recorded_timestamp
    if existing != comparable:
        mismatches = {
            key: {"existing": existing.get(key), "expected": comparable.get(key)}
            for key in sorted(set(existing) | set(comparable))
            if existing.get(key) != comparable.get(key)
        }
        raise RuntimeError(
            "Refusing incompatible cumulative-energy resume: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return existing


def run(args: argparse.Namespace) -> Path:
    round4b_root = args.round4b_root.resolve()
    output = args.output_root.resolve()
    expected_parent = round4b_root / "energy_curve_analysis"
    if output != expected_parent:
        raise ValueError(f"Analysis output must be the isolated directory {expected_parent}.")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    preflight = _preflight(round4b_root, output)
    preflight_path = output / "preflight_report.json"
    if preflight_path.exists():
        preflight = _reuse_stable_preflight(_read(preflight_path), preflight)
    else:
        atomic_write_json(preflight_path, preflight)

    python = str(args.python.resolve())
    gpu_args = ["--gpu-ids", *(str(value) for value in args.gpu_ids)]
    basis = preflight["basis_manifest_path"]
    split = preflight["split_path"]
    diagnostics = preflight["diagnostics_path"]
    heldout_dir = output / "heldout_grams"
    coordinate_dir = output / "coordinate_energy"
    _run(
        COLLECT_STAGE,
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4b.launch_energy_curve_shards",
            "--stage",
            COLLECT_STAGE,
            "--python",
            python,
            *gpu_args,
            "--basis-manifest",
            basis,
            "--checkpoint",
            preflight["checkpoint_path"],
            "--split",
            split,
            "--donor-mapping",
            preflight["donor_mapping_path"],
            "--donor-manifest",
            preflight["donor_manifest_path"],
            "--donor-root",
            preflight["donor_root"],
            "--output-dir",
            str(heldout_dir),
            "--log-dir",
            str(logs / "shards"),
            "--launch-stagger-seconds",
            str(args.launch_stagger_seconds),
        ],
        logs,
    )
    _run(
        RECOVER_STAGE,
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4b.launch_energy_curve_shards",
            "--stage",
            RECOVER_STAGE,
            "--python",
            python,
            *gpu_args,
            "--basis-manifest",
            basis,
            "--fit-dir",
            str(round4b_root / "calibration"),
            "--heldout-gram-dir",
            str(heldout_dir),
            "--diagnostics",
            diagnostics,
            "--output-dir",
            str(coordinate_dir),
            "--log-dir",
            str(logs / "shards"),
            "--launch-stagger-seconds",
            str(args.launch_stagger_seconds),
        ],
        logs,
    )
    _run(
        "aggregate_and_plots",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4b.energy_curve_analysis",
            "--coordinate-dir",
            str(coordinate_dir),
            "--basis-manifest",
            basis,
            "--split",
            split,
            "--diagnostics",
            diagnostics,
            "--output-dir",
            str(output),
        ],
        logs,
    )
    report = output / "energy_curve_analysis_summary.md"
    atomic_write_json(
        output / "analysis_status.json",
        {
            "artifact_type": "asre_round4b_energy_curve_driver_status",
            "schema_version": 1,
            "protocol": ROUND4B_PROTOCOL,
            "status": "complete",
            "created_at": now_iso(),
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "current_branch": _git("branch", "--show-current"),
            "starting_commit": FOLLOWUP_START_COMMIT,
            "gpu_ids": list(args.gpu_ids),
            "cache_only_heldout_states": 100,
            "online_episodes": 0,
            "environment_rollouts": 0,
            "heldout_svd_refit": False,
            "later_experiment_launched": False,
            "final_report": str(report),
        },
    )
    print(f"Round-4B cumulative energy follow-up complete: {report}")
    print("No online or later-stage experiment was launched.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round4b-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--launch-stagger-seconds", type=float, default=30.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
