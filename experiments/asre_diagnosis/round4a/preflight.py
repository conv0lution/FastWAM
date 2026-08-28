"""Fail-closed provenance gate for Stage-2 ASRE Round-4A."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND3B_PROTOCOL,
    ROUND4A_PROTOCOL,
    atomic_write_json,
    build_round4a_conditions,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round2.validate_state_bank import (  # noqa: E402
    _validate_sample_partition,
    validate_existing_artifacts,
)
from experiments.asre_diagnosis.round3b.offline_donor import (  # noqa: E402
    load_offline_donor_manifest,
)
from experiments.asre_diagnosis.round3b.preflight import (  # noqa: E402
    _validate_donors,
)


ROUND3B_TAG = "ASRE-round3b-kv-replacement"
ROUND3B_COMMIT = "fb771af7ec32dd8e8bd12b0eedea735e307771a1"
G0_TAG = "ASRE-g0-cross-suite"
G0_COMMIT = "874e5fee511c48c663cc03f0cbe273e030a328d7"
EXPECTED_CHECKPOINT_NAME = "libero_uncond_2cam224.pt"


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return payload


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Git command failed: git {' '.join(args)}\n{getattr(exc, 'output', '')}"
        ) from exc


def _require_frozen_git() -> dict[str, Any]:
    tags = {ROUND3B_TAG: ROUND3B_COMMIT, G0_TAG: G0_COMMIT}
    for tag, expected_commit in tags.items():
        observed = _git("rev-list", "-n", "1", tag)
        if observed != expected_commit:
            raise ValueError(f"Frozen tag {tag} moved: {observed} != {expected_commit}.")
        try:
            subprocess.check_call(
                ["git", "merge-base", "--is-ancestor", tag, "HEAD"],
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"Current HEAD does not descend from {tag}.") from exc
    worktree = _git(
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        ".",
        ":(exclude)asre_results/round4a/**",
    )
    if worktree:
        raise RuntimeError(
            "Round-4A formal runs require a clean source worktree. Commit the reviewed "
            f"implementation first.\n{worktree}"
        )
    prior_results = _git(
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        "asre_results",
        ":(exclude)asre_results/round4a/**",
    )
    if prior_results:
        raise RuntimeError(
            "Frozen pre-Round-4A result files are modified:\n" + prior_results
        )
    return {
        "current_head": git_commit(PROJECT_ROOT),
        "current_branch": _git("branch", "--show-current"),
        "worktree_clean_excluding_round4a_outputs": True,
        "round3b_parent_tag": ROUND3B_TAG,
        "round3b_parent_commit": ROUND3B_COMMIT,
        "g0_parent_tag": G0_TAG,
        "g0_parent_commit": G0_COMMIT,
    }


def _declared_file(payload: Mapping[str, Any], path_key: str, base: Path) -> Path:
    value = payload.get(path_key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Valid state manifest lacks {path_key!r}.")
    path = Path(os.path.expanduser(os.path.expandvars(value)))
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Declared artifact is unavailable: {path}")
    return path


def _validate_state_bank(valid_manifest_path: Path) -> dict[str, Any]:
    valid_manifest_path = valid_manifest_path.resolve()
    valid = _read_json(valid_manifest_path, label="QC-valid state manifest")
    source_path = _declared_file(valid, "source_manifest_path", valid_manifest_path.parent)
    checkpoint_path = _declared_file(valid, "checkpoint_path", valid_manifest_path.parent)
    dataset_stats_path = _declared_file(
        valid, "dataset_stats_path", valid_manifest_path.parent
    )
    prompt_cache_path = _declared_file(
        valid, "prompt_context_cache_path", valid_manifest_path.parent
    )
    if checkpoint_path.name != EXPECTED_CHECKPOINT_NAME:
        raise ValueError(
            "Round-4A discovery is restricted to checkpoint "
            f"{EXPECTED_CHECKPOINT_NAME!r}, got {checkpoint_path.name!r}."
        )
    source_records = load_manifest(source_path)
    valid_ids = _validate_sample_partition(valid, source_records)
    if len(source_records) != 500 or len(valid_ids) != 499:
        raise ValueError(
            f"Round-4A requires the exact 500-to-499 state bank, got "
            f"{len(source_records)} source/{len(valid_ids)} valid."
        )
    validate_existing_artifacts(
        valid_manifest_path=valid_manifest_path,
        prompt_cache_path=prompt_cache_path,
        source_manifest_path=source_path,
        source_records=source_records,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        verify_checkpoint_hash=True,
    )
    return {
        "valid_state_bank_manifest_path": str(valid_manifest_path),
        "valid_state_bank_manifest_sha256": sha256_file(valid_manifest_path),
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(valid["checkpoint_sha256"]),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "dataset_stats_path": str(dataset_stats_path),
        "dataset_stats_sha256": str(valid["dataset_stats_sha256"]),
        "prompt_context_cache_path": str(prompt_cache_path),
        "prompt_context_cache_sha256": str(valid["prompt_context_cache_sha256"]),
        "source_sample_count": len(source_records),
        "valid_sample_count": len(valid_ids),
        "valid_sample_order_sha256": __import__("hashlib").sha256(
            json.dumps(valid_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _validate_offline_donors(
    path: Path, *, state_bank: Mapping[str, Any]
) -> dict[str, Any]:
    path = path.resolve()
    payload = load_offline_donor_manifest(path)
    expected = {
        "artifact_type": "asre_round3b_offline_donor_mapping",
        "schema_version": 1,
        "num_pairs": 499,
        "same_task": True,
        "same_replan": True,
        "different_episode": True,
        "outcome_independent": True,
        "derangement_verified": True,
        "source_manifest_sha256": state_bank["source_manifest_sha256"],
        "valid_manifest_sha256": state_bank[
            "valid_state_bank_manifest_sha256"
        ],
    }
    mismatch = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 499:
        mismatch["entries"] = {
            "observed": None if not isinstance(entries, list) else len(entries),
            "expected": 499,
        }
    if mismatch:
        raise ValueError(
            "Frozen offline donor mapping is incompatible: "
            + json.dumps(mismatch, sort_keys=True)
        )
    return {
        "offline_donor_mapping_path": str(path),
        "offline_donor_mapping_sha256": sha256_file(path),
        "mapping_rule": str(payload["mapping_rule"]),
        "mapping_count": 499,
    }


def _validate_stage1_results(
    *, round3b_summary_path: Path, g0_summary_path: Path
) -> dict[str, Any]:
    round3b_summary_path = round3b_summary_path.resolve()
    g0_summary_path = g0_summary_path.resolve()
    round3b = _read_json(round3b_summary_path, label="Round-3B summary")
    decision = round3b.get("analysis", {}).get("decision", {})
    if (
        round3b.get("artifact_type") != "asre_round3b_aggregate"
        or round3b.get("condition_protocol") != ROUND3B_PROTOCOL
        or decision.get("classification") != "GO"
        or decision.get("label") != "strong_content_support"
    ):
        raise ValueError("Frozen Round-3B result does not authorize the content screen.")
    g0 = _read_json(g0_summary_path, label="G0 summary")
    expected_g0 = {
        "artifact_type": "asre_g0_cross_suite_summary",
        "schema_version": 1,
        "status": "complete",
        "overall_classification": "G0-STRONG",
        "stage2_round4a_launched": False,
    }
    mismatch = {
        key: {"observed": g0.get(key), "expected": value}
        for key, value in expected_g0.items()
        if g0.get(key) != value
    }
    if mismatch:
        raise ValueError("Frozen G0 gate is incompatible: " + json.dumps(mismatch))
    return {
        "round3b_summary_path": str(round3b_summary_path),
        "round3b_summary_sha256": sha256_file(round3b_summary_path),
        "round3b_decision": dict(decision),
        "g0_summary_path": str(g0_summary_path),
        "g0_summary_sha256": sha256_file(g0_summary_path),
        "g0_classification": "G0-STRONG",
        "g0_stage2_previously_launched": False,
    }


def run_preflight(
    *,
    valid_manifest_path: Path,
    online_donor_mapping_path: Path,
    online_donor_manifest_path: Path,
    online_donor_root: Path,
    offline_donor_mapping_path: Path,
    round3b_summary_path: Path,
    g0_summary_path: Path,
) -> dict[str, Any]:
    git = _require_frozen_git()
    state_bank = _validate_state_bank(valid_manifest_path)
    online_donors = _validate_donors(
        online_donor_mapping_path,
        online_donor_manifest_path,
        online_donor_root,
    )
    offline_donors = _validate_offline_donors(
        offline_donor_mapping_path, state_bank=state_bank
    )
    stage1 = _validate_stage1_results(
        round3b_summary_path=round3b_summary_path,
        g0_summary_path=g0_summary_path,
    )
    return {
        "artifact_type": "asre_round4a_preflight_report",
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "status": "compatible",
        "created_at": now_iso(),
        "git": git,
        "discovery_scope": {
            "task_suite": "libero_spatial",
            "task_ids": list(range(10)),
            "num_trials": 10,
            "seed": 42,
            "checkpoint_name": EXPECTED_CHECKPOINT_NAME,
            "forbidden_suites": ["libero_object", "libero_goal", "libero_10", "robotwin"],
        },
        "conditions": [
            {"condition_index": index, **condition.to_dict()}
            for index, condition in enumerate(build_round4a_conditions(30))
        ],
        "state_bank": state_bank,
        "online_donors": online_donors,
        "offline_donors": offline_donors,
        "stage1": stage1,
        "later_stage_launched": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--online-donor-mapping", type=Path, required=True)
    parser.add_argument("--online-donor-manifest", type=Path, required=True)
    parser.add_argument("--online-donor-root", type=Path, required=True)
    parser.add_argument("--offline-donor-mapping", type=Path, required=True)
    parser.add_argument("--round3b-summary", type=Path, required=True)
    parser.add_argument("--g0-summary", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_preflight(
        valid_manifest_path=args.valid_manifest,
        online_donor_mapping_path=args.online_donor_mapping,
        online_donor_manifest_path=args.online_donor_manifest,
        online_donor_root=args.online_donor_root,
        offline_donor_mapping_path=args.offline_donor_mapping,
        round3b_summary_path=args.round3b_summary,
        g0_summary_path=args.g0_summary,
    )
    output = args.output_report.resolve()
    if output.exists():
        existing = _read_json(output, label="existing Round-4A preflight")
        stable = {key: value for key, value in report.items() if key != "created_at"}
        existing_stable = {
            key: value for key, value in existing.items() if key != "created_at"
        }
        if existing_stable != stable:
            raise FileExistsError(
                f"Refusing to overwrite incompatible Round-4A preflight report: {output}"
            )
        report = existing
    else:
        atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Round-4A preflight passed; no frozen parent artifact was modified.")


if __name__ == "__main__":
    main()
