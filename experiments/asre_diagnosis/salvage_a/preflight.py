"""Fail-closed frozen-parent and source-data gate for ASRE Salvage A."""

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
    ROUND4C_PROTOCOL,
    SALVAGE_A_PROTOCOL,
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
from experiments.asre_diagnosis.round3b.donor import (  # noqa: E402
    validate_observation_manifest,
)


ROUND4C_ANALYSIS_COMMIT = "c18d58a153ad9f0afa2b5ac0194313642c3fb52b"
ROUND4C_SUMMARY_SHA256 = "b6108d134197c0825c4f8f2bcb11f87bab6645d3c714033cae5515d16b40f4d1"
VALID_STATE_MANIFEST_SHA256 = "a05ebb6b50d64eb022103269a0eef595e749fe8c16e49e996ba08669f44378cd"
DONOR_OBSERVATION_MANIFEST_SHA256 = "3a568b0a47e7cacf26914c881c859362d84743d4b5194017c5d126a0d85ba5bd"
EXPECTED_CHECKPOINT_NAME = "libero_uncond_2cam224.pt"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path.resolve())
    if observed != expected:
        raise ValueError(f"Frozen {label} SHA256 drifted: {observed} != {expected}.")
    return observed


def _validate_git(output_root: Path) -> dict[str, Any]:
    subprocess.check_call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ROUND4C_ANALYSIS_COMMIT,
            "HEAD",
        ],
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
            "Salvage-A formal runs require committed reviewed source code.\n" + dirty
        )
    return {
        "current_branch": _git("branch", "--show-current"),
        "current_head": git_commit(PROJECT_ROOT),
        "round4c_analysis_commit": ROUND4C_ANALYSIS_COMMIT,
        "round4c_is_ancestor": True,
        "worktree_clean_excluding_output_root": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    round4c_path = args.round4c_summary.resolve()
    valid_path = args.valid_manifest.resolve()
    donor_manifest_path = args.donor_manifest.resolve()
    _require_sha(round4c_path, ROUND4C_SUMMARY_SHA256, "Round-4C summary")
    _require_sha(valid_path, VALID_STATE_MANIFEST_SHA256, "valid state manifest")
    _require_sha(
        donor_manifest_path,
        DONOR_OBSERVATION_MANIFEST_SHA256,
        "donor observation manifest",
    )
    git = _validate_git(output_root)

    round4c = _read(round4c_path)
    if (
        round4c.get("artifact_type") != "asre_round4c_summary"
        or round4c.get("protocol") != ROUND4C_PROTOCOL
        or round4c.get("status") != "complete"
        or round4c.get("classification", {}).get("classification") != "WEAK"
        or round4c.get("stop_rule_applied") is not True
        or round4c.get("later_stage_launched") is not False
        or round4c.get("git_commit_hash") != ROUND4C_ANALYSIS_COMMIT
    ):
        raise ValueError("Frozen Round-4C result does not authorize Salvage A.")

    valid = _read(valid_path)
    source = Path(str(valid["source_manifest_path"])).resolve()
    checkpoint = Path(str(valid["checkpoint_path"])).resolve()
    dataset_stats = Path(str(valid["dataset_stats_path"])).resolve()
    prompt_cache = Path(str(valid["prompt_context_cache_path"])).resolve()
    records = load_manifest(source)
    valid_ids = _validate_sample_partition(valid, records)
    if (
        checkpoint.name != EXPECTED_CHECKPOINT_NAME
        or len(records) != 500
        or len(valid_ids) != 499
    ):
        raise ValueError("Salvage A requires the exact LIBERO-Spatial 500/499 state bank.")
    validate_existing_artifacts(
        valid_manifest_path=valid_path,
        prompt_cache_path=prompt_cache,
        source_manifest_path=source,
        source_records=records,
        checkpoint_path=checkpoint,
        dataset_stats_path=dataset_stats,
        verify_checkpoint_hash=True,
    )

    donor_payload = _read(donor_manifest_path)
    observations = validate_observation_manifest(donor_payload)
    donor_root = args.donor_root.resolve()
    if len(observations) != 100:
        raise ValueError("Salvage A requires all 100 frozen donor observations.")
    missing = []
    for record in observations.values():
        artifact = donor_root / str(record["artifact_relative_path"])
        if not artifact.is_file() or sha256_file(artifact) != record["artifact_sha256"]:
            missing.append(str(artifact))
    if missing:
        raise ValueError(f"Frozen donor observation artifacts drifted: {missing[:5]}")

    return {
        "artifact_type": "asre_salvage_a_preflight_report",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "status": "compatible",
        "created_at": now_iso(),
        "git": git,
        "round4c": {
            "summary_path": str(round4c_path),
            "summary_sha256": sha256_file(round4c_path),
            "analysis_commit": ROUND4C_ANALYSIS_COMMIT,
            "classification": "WEAK",
            "svd_success": {"36": 0.0, "97": 0.79, "170": 0.96},
        },
        "state_bank": {
            "valid_manifest_path": str(valid_path),
            "valid_manifest_sha256": sha256_file(valid_path),
            "source_manifest_path": str(source),
            "source_manifest_sha256": sha256_file(source),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "dataset_stats_path": str(dataset_stats),
            "dataset_stats_sha256": sha256_file(dataset_stats),
            "prompt_context_cache_path": str(prompt_cache),
            "prompt_context_cache_sha256": sha256_file(prompt_cache),
            "source_sample_count": 500,
            "valid_sample_count": 499,
        },
        "donor_observations": {
            "manifest_path": str(donor_manifest_path),
            "manifest_sha256": sha256_file(donor_manifest_path),
            "observation_root": str(donor_root),
            "observation_count": 100,
            "captured_before_salvage_outcomes": True,
        },
        "no_online_episode_launched": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round4c-summary", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    atomic_write_json(args.output.resolve(), report)
    print(f"Salvage-A preflight compatible: {args.output.resolve()}")


if __name__ == "__main__":
    main()
