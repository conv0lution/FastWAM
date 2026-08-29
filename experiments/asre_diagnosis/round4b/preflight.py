"""Fail-closed parent/provenance gate for ASRE Stage-2 Round-4B."""

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
    atomic_write_json,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round2.validate_state_bank import (  # noqa: E402
    _validate_sample_partition,
    validate_existing_artifacts,
)
from experiments.asre_diagnosis.round3b.donor import OnlineDonorBundle  # noqa: E402


ROUND4A_TAG = "ASRE-round4a-token-head-sparsity"
ROUND4A_COMMIT = "094994e4e6f936c84fde3e1023ad998fabd17bcf"
EXPECTED_CHECKPOINT = "libero_uncond_2cam224.pt"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _validate_git() -> dict[str, Any]:
    parent = _git("rev-list", "-n", "1", ROUND4A_TAG)
    if parent != ROUND4A_COMMIT:
        raise ValueError(f"Frozen Round-4A tag moved: {parent} != {ROUND4A_COMMIT}.")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", ROUND4A_TAG, "HEAD"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
    )
    dirty = _git(
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        ".",
        ":(exclude)asre_results/round4b_subspace/**",
    )
    if dirty:
        raise RuntimeError(
            "Round-4B formal runs require committed reviewed source code.\n" + dirty
        )
    return {
        "current_head": git_commit(PROJECT_ROOT),
        "current_branch": _git("branch", "--show-current"),
        "worktree_clean_excluding_round4b_outputs": True,
        "round4a_parent_tag": ROUND4A_TAG,
        "round4a_parent_commit": ROUND4A_COMMIT,
    }


def run_preflight(
    *,
    valid_manifest_path: Path,
    online_donor_mapping_path: Path,
    online_donor_manifest_path: Path,
    online_donor_root: Path,
    round4a_summary_path: Path,
) -> dict[str, Any]:
    git = _validate_git()
    valid_manifest_path = valid_manifest_path.resolve()
    valid = _read(valid_manifest_path)
    source = Path(str(valid["source_manifest_path"])).resolve()
    checkpoint = Path(str(valid["checkpoint_path"])).resolve()
    stats = Path(str(valid["dataset_stats_path"])).resolve()
    prompt = Path(str(valid["prompt_context_cache_path"])).resolve()
    records = load_manifest(source)
    valid_ids = _validate_sample_partition(valid, records)
    if checkpoint.name != EXPECTED_CHECKPOINT or len(records) != 500 or len(valid_ids) != 499:
        raise ValueError("Round-4B requires the exact LIBERO-Spatial 500/499 state bank.")
    validate_existing_artifacts(
        valid_manifest_path=valid_manifest_path,
        prompt_cache_path=prompt,
        source_manifest_path=source,
        source_records=records,
        checkpoint_path=checkpoint,
        dataset_stats_path=stats,
        verify_checkpoint_hash=True,
    )
    bundle = OnlineDonorBundle.load(
        mapping_path=online_donor_mapping_path.resolve(),
        observation_manifest_path=online_donor_manifest_path.resolve(),
        observation_root=online_donor_root.resolve(),
    )
    if (
        bundle.mapping_payload.get("task_suite") != "libero_spatial"
        or bundle.mapping_payload.get("seed") != 42
        or len(bundle.mappings) != 100
    ):
        raise ValueError("Frozen outcome-independent donor bundle is incompatible.")
    round4a_summary_path = round4a_summary_path.resolve()
    round4a = _read(round4a_summary_path)
    analyses = round4a.get("axis_analysis", {})
    if (
        round4a.get("artifact_type") != "asre_round4a_aggregate"
        or analyses.get("token", {}).get("classification") != "WEAK"
        or analyses.get("head", {}).get("classification") != "WEAK"
        or round4a.get("recommendation", {}).get("next_step_category")
        != "subspace reconsideration"
    ):
        raise ValueError("Frozen Round-4A result does not authorize Round-4B.")
    return {
        "artifact_type": "asre_round4b_preflight_report",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "status": "compatible",
        "created_at": now_iso(),
        "git": git,
        "state_bank": {
            "valid_manifest_path": str(valid_manifest_path),
            "valid_manifest_sha256": sha256_file(valid_manifest_path),
            "source_manifest_path": str(source),
            "source_manifest_sha256": sha256_file(source),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "dataset_stats_path": str(stats),
            "dataset_stats_sha256": sha256_file(stats),
            "prompt_context_cache_path": str(prompt),
            "prompt_context_cache_sha256": sha256_file(prompt),
            "source_sample_count": 500,
            "valid_sample_count": 499,
        },
        "donors": {
            "mapping_path": str(bundle.mapping_path),
            "mapping_sha256": bundle.mapping_sha256,
            "observation_manifest_path": str(bundle.observation_manifest_path),
            "observation_manifest_sha256": bundle.observation_manifest_sha256,
            "observation_root": str(bundle.observation_root),
            "mapping_rule": bundle.mapping_payload["mapping_rule"],
            "outcome_independent": True,
            "fixed_first_policy_query_image": True,
        },
        "round4a": {
            "summary_path": str(round4a_summary_path),
            "summary_sha256": sha256_file(round4a_summary_path),
            "parent_tag": ROUND4A_TAG,
            "parent_commit": ROUND4A_COMMIT,
            "token_classification": "WEAK",
            "head_classification": "WEAK",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--online-donor-mapping", type=Path, required=True)
    parser.add_argument("--online-donor-manifest", type=Path, required=True)
    parser.add_argument("--online-donor-root", type=Path, required=True)
    parser.add_argument("--round4a-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_preflight(
        valid_manifest_path=args.valid_manifest,
        online_donor_mapping_path=args.online_donor_mapping,
        online_donor_manifest_path=args.online_donor_manifest,
        online_donor_root=args.online_donor_root,
        round4a_summary_path=args.round4a_summary,
    )
    atomic_write_json(args.output.resolve(), report)
    print(f"Round-4B preflight compatible: {args.output.resolve()}")


if __name__ == "__main__":
    main()
