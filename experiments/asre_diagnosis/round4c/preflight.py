"""Fail-closed frozen-input and provenance gate for ASRE Stage-2 Round-4C."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4B_PROTOCOL,
    ROUND4C_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round3b.donor import OnlineDonorBundle  # noqa: E402
from experiments.asre_diagnosis.round4b.basis import validate_basis_manifest  # noqa: E402
from experiments.asre_diagnosis.round4c.definitions import (  # noqa: E402
    CUMULATIVE_ENERGY_COMMIT,
    EXPECTED_BASIS_SHA256,
    EXPECTED_CANDIDATES_SHA256,
    EXPECTED_DIAGNOSTICS_SHA256,
    EXPECTED_ENERGY_MANIFEST_SHA256,
    EXPECTED_ROUND4B_SUMMARY_SHA256,
    EXPECTED_SPLIT_SHA256,
    ROUND4A_COMMIT,
    ROUND4A_TAG,
    ROUND4B_SOURCE_COMMIT,
    validate_energy_candidates,
)


EXPECTED_CHECKPOINT = "libero_uncond_2cam224.pt"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _require_digest(path: Path, expected: str, label: str) -> str:
    path = path.resolve()
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"Frozen {label} SHA256 drifted: {observed} != {expected}.")
    return observed


def _validate_git(output_root: Path) -> dict[str, Any]:
    round4a = _git("rev-list", "-n", "1", ROUND4A_TAG)
    if round4a != ROUND4A_COMMIT:
        raise ValueError(f"Frozen Round-4A tag moved: {round4a} != {ROUND4A_COMMIT}.")
    for ancestor in (ROUND4A_COMMIT, ROUND4B_SOURCE_COMMIT, CUMULATIVE_ENERGY_COMMIT):
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
        )
    relative_output = output_root.resolve().relative_to(PROJECT_ROOT.resolve())
    dirty = _git(
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        ".",
        f":(exclude){relative_output}/**",
    )
    if dirty:
        raise RuntimeError(
            "Round-4C formal runs require committed reviewed source code.\n" + dirty
        )
    round4b_tags = _git("tag", "--points-at", ROUND4B_SOURCE_COMMIT).splitlines()
    return {
        "current_branch": _git("branch", "--show-current"),
        "current_head": git_commit(PROJECT_ROOT),
        "worktree_clean_excluding_round4c_outputs": True,
        "round4a_tag": ROUND4A_TAG,
        "round4a_commit": ROUND4A_COMMIT,
        "round4b_source_commit": ROUND4B_SOURCE_COMMIT,
        "round4b_tags_pointing_at_source_commit": round4b_tags,
        "round4b_tag_present": bool(round4b_tags),
        "cumulative_energy_analysis_commit": CUMULATIVE_ENERGY_COMMIT,
    }


def run_preflight(*, round4b_root: Path, output_root: Path) -> dict[str, Any]:
    round4b_root = round4b_root.resolve()
    output_root = output_root.resolve()
    paths = {
        "round4b_preflight": round4b_root / "preflight_report.json",
        "round4b_summary": round4b_root / "aggregate/round4b_summary.json",
        "split": round4b_root / "calibration/calibration_split_manifest.json",
        "basis": round4b_root / "calibration/basis_manifest.json",
        "diagnostics": round4b_root / "calibration/subspace_diagnostics.json",
        "energy_manifest": round4b_root
        / "energy_curve_analysis/energy_curve_analysis_manifest.json",
        "energy_candidates": round4b_root
        / "energy_curve_analysis/candidate_next_online_ranks.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Round-4C frozen input artifacts are missing: {missing}")
    git = _validate_git(output_root)
    _require_digest(paths["round4b_summary"], EXPECTED_ROUND4B_SUMMARY_SHA256, "Round-4B summary")
    _require_digest(paths["split"], EXPECTED_SPLIT_SHA256, "calibration split")
    _require_digest(paths["basis"], EXPECTED_BASIS_SHA256, "basis manifest")
    _require_digest(paths["diagnostics"], EXPECTED_DIAGNOSTICS_SHA256, "diagnostics")
    _require_digest(paths["energy_manifest"], EXPECTED_ENERGY_MANIFEST_SHA256, "energy manifest")
    _require_digest(paths["energy_candidates"], EXPECTED_CANDIDATES_SHA256, "candidate ranks")
    basis = validate_basis_manifest(
        paths["basis"], expected_sha256=EXPECTED_BASIS_SHA256, verify_files=True
    )
    old_preflight = _read(paths["round4b_preflight"])
    summary = _read(paths["round4b_summary"])
    energy = _read(paths["energy_manifest"])
    candidates = _read(paths["energy_candidates"])
    registered_energy = validate_energy_candidates(candidates)
    if (
        old_preflight.get("protocol") != ROUND4B_PROTOCOL
        or old_preflight.get("git", {}).get("current_head") != ROUND4B_SOURCE_COMMIT
        or summary.get("status") != "complete"
        or summary.get("classification", {}).get("classification") != "STRONG"
    ):
        raise ValueError("Frozen Round-4B source run is incompatible with Round-4C.")
    if (
        energy.get("status") != "complete"
        or energy.get("git_commit_hash") != CUMULATIVE_ENERGY_COMMIT
        or energy.get("feature_dim") != 3072
        or energy.get("matrix_count") != 30
        or energy.get("checks", {}).get("heldout_svd_refit") is not False
        or energy.get("checks", {}).get("online_episodes") != 0
        or energy.get("provenance", {}).get("basis_manifest_sha256")
        != EXPECTED_BASIS_SHA256
        or energy.get("provenance", {}).get("split_sha256") != EXPECTED_SPLIT_SHA256
    ):
        raise ValueError("Frozen cumulative-energy analysis is incompatible with Round-4C.")
    state = old_preflight["state_bank"]
    if Path(state["checkpoint_path"]).name != EXPECTED_CHECKPOINT:
        raise ValueError("Round-4C requires libero_uncond_2cam224.pt.")
    for stem in (
        "checkpoint",
        "dataset_stats",
        "prompt_context_cache",
        "source_manifest",
        "valid_manifest",
    ):
        _require_digest(
            Path(state[f"{stem}_path"]), state[f"{stem}_sha256"], f"state-bank {stem}"
        )
    donors = old_preflight["donors"]
    bundle = OnlineDonorBundle.load(
        mapping_path=Path(donors["mapping_path"]),
        observation_manifest_path=Path(donors["observation_manifest_path"]),
        observation_root=Path(donors["observation_root"]),
    )
    if len(bundle.mappings) != 100 or bundle.mapping_payload.get("seed") != 42:
        raise ValueError("Frozen Round-4C donor mapping is incompatible.")
    return {
        "artifact_type": "asre_round4c_preflight_report",
        "schema_version": 1,
        "protocol": ROUND4C_PROTOCOL,
        "status": "compatible",
        "created_at": now_iso(),
        "git": git,
        "scope": {
            "task_suite": "libero_spatial",
            "checkpoint_name": EXPECTED_CHECKPOINT,
            "conditions": ["current_all", "wrong_all", "svd_r36", "svd_r97", "svd_r170"],
            "online_episodes_per_condition": 100,
            "later_stage_authorized": False,
        },
        "state_bank": state,
        "donors": donors,
        "frozen_round4b": {
            "root": str(round4b_root),
            "source_commit": ROUND4B_SOURCE_COMMIT,
            "summary_path": str(paths["round4b_summary"]),
            "summary_sha256": EXPECTED_ROUND4B_SUMMARY_SHA256,
            "summary_classification": "STRONG",
            "basis_manifest_path": str(paths["basis"]),
            "basis_manifest_sha256": EXPECTED_BASIS_SHA256,
            "split_manifest_path": str(paths["split"]),
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "diagnostics_path": str(paths["diagnostics"]),
            "diagnostics_sha256": EXPECTED_DIAGNOSTICS_SHA256,
            "basis_runtime_layout": basis["runtime_layout"],
        },
        "cumulative_energy": {
            "analysis_commit": CUMULATIVE_ENERGY_COMMIT,
            "manifest_path": str(paths["energy_manifest"]),
            "manifest_sha256": EXPECTED_ENERGY_MANIFEST_SHA256,
            "candidate_ranks_path": str(paths["energy_candidates"]),
            "candidate_ranks_sha256": EXPECTED_CANDIDATES_SHA256,
            "registered_heldout_global_energy": {
                str(rank): energy_value for rank, energy_value in registered_energy.items()
            },
            "energy_label_source": "frozen held-out global absolute-energy-weighted curve",
            "heldout_svd_refit": False,
        },
        "previous_artifacts_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round4b-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_preflight(round4b_root=args.round4b_root, output_root=args.output_root)
    atomic_write_json(args.output.resolve(), report)
    print(f"Round-4C preflight compatible: {args.output.resolve()}")


if __name__ == "__main__":
    main()
