"""Fail-fast integrity check for the five frozen Round-2 factorial cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND2_PROTOCOL,
    build_round2_conditions,
)
from experiments.asre_diagnosis.round3a.aggregate_factorial import (  # noqa: E402
    NUM_LAYERS,
    ONLINE_COMPATIBILITY_FIELDS,
    _load_offline_cell,
    _load_task_outcomes,
    _load_valid_manifest,
    _validate_launcher_summary,
    _validate_online_metadata,
)
from experiments.asre_diagnosis.round3a.factorial import (  # noqa: E402
    FACTORIAL_CELLS,
)
from experiments.asre_diagnosis.round3a.launch_three_gpu import (  # noqa: E402
    _require_clean_worktree,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all five frozen Round-2 factorial source cells."
    )
    parser.add_argument("--round2-online-root", type=Path, required=True)
    parser.add_argument("--round2-offline-root", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    return parser.parse_args()


def _compatibility_mismatches(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"reference_y111": reference.get(key), "candidate": candidate.get(key)}
        for key in ONLINE_COMPATIBILITY_FIELDS
        if candidate.get(key) != reference.get(key)
    }


def preflight_round2(
    *,
    online_root: Path,
    offline_root: Path,
    valid_manifest_path: Path,
) -> dict[str, Any]:
    online_root = online_root.resolve()
    offline_root = offline_root.resolve()
    valid_manifest_path = valid_manifest_path.resolve()
    if not online_root.is_dir() or not offline_root.is_dir():
        raise FileNotFoundError(
            f"Frozen Round-2 roots are unavailable: online={online_root}, "
            f"offline={offline_root}."
        )

    _payload, valid_ids, valid_digest = _load_valid_manifest(valid_manifest_path)
    _validate_launcher_summary(
        online_root,
        protocol=ROUND2_PROTOCOL,
        expected_conditions=[
            condition.name for condition in build_round2_conditions(NUM_LAYERS)
        ],
    )

    cells = tuple(cell for cell in FACTORIAL_CELLS if cell.protocol == ROUND2_PROTOCOL)
    metadata_by_code: dict[str, dict[str, Any]] = {}
    outcomes_by_code: dict[str, dict[tuple[str, int, int], int]] = {}
    descriptions_reference: dict[int, str] | None = None
    records_by_code: dict[str, list[dict[str, Any]]] = {}
    commit_hashes: set[str] = set()

    for cell in cells:
        condition_dir = online_root / cell.condition.name
        metadata = _validate_online_metadata(
            condition_dir / "run_metadata.json",
            cell=cell,
            valid_manifest_path=valid_manifest_path,
            valid_manifest_sha256=valid_digest,
        )
        outcomes, _by_task, descriptions, _paths = _load_task_outcomes(
            condition_dir, cell=cell
        )
        if descriptions_reference is None:
            descriptions_reference = descriptions
        elif descriptions != descriptions_reference:
            raise ValueError(
                f"Frozen Round-2 task descriptions differ for cell {cell.code}."
            )
        records, _summary, _provenance = _load_offline_cell(
            offline_root,
            cell=cell,
            expected_sample_ids=valid_ids,
            valid_manifest_path=valid_manifest_path,
            valid_manifest_sha256=valid_digest,
            online_metadata=metadata,
        )
        metadata_by_code[cell.code] = metadata
        outcomes_by_code[cell.code] = outcomes
        records_by_code[cell.code] = records
        commit_hashes.add(str(metadata["git_commit_hash"]))

    if len(commit_hashes) != 1:
        raise ValueError(
            f"Frozen Round-2 factorial cells span multiple commits: {sorted(commit_hashes)}"
        )
    reference_metadata = metadata_by_code["111"]
    reference_keys = set(outcomes_by_code["111"])
    for cell in cells:
        mismatches = _compatibility_mismatches(
            reference_metadata, metadata_by_code[cell.code]
        )
        if mismatches:
            raise ValueError(
                f"Frozen Round-2 provenance differs for cell {cell.code}: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        if set(outcomes_by_code[cell.code]) != reference_keys:
            raise ValueError(
                f"Frozen Round-2 paired online keys differ for cell {cell.code}."
            )

    for sample_index, sample_id in enumerate(valid_ids):
        reference = records_by_code["111"][sample_index]
        reference_identity = tuple(
            reference[key]
            for key in ("task_suite", "task_id", "episode_id", "replan_id")
        )
        for cell in cells:
            record = records_by_code[cell.code][sample_index]
            identity = tuple(
                record[key]
                for key in ("task_suite", "task_id", "episode_id", "replan_id")
            )
            if str(record["sample_id"]) != sample_id or identity != reference_identity:
                raise ValueError(
                    f"Frozen Round-2 offline pairing differs for cell {cell.code}, "
                    f"sample {sample_id}."
                )

    return {
        "status": "compatible",
        "round2_git_commit_hash": next(iter(commit_hashes)),
        "valid_manifest_sha256": valid_digest,
        "valid_sample_count": len(valid_ids),
        "cell_success_rates": {
            cell.code: sum(outcomes_by_code[cell.code].values()) / len(reference_keys)
            for cell in cells
        },
        "cell_conditions": {cell.code: cell.condition.name for cell in cells},
    }


def main() -> None:
    args = _parse_args()
    _require_clean_worktree()
    report = preflight_round2(
        online_root=args.round2_online_root,
        offline_root=args.round2_offline_root,
        valid_manifest_path=args.valid_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Frozen Round-2 five-cell factorial preflight passed; no source files changed.")


if __name__ == "__main__":
    main()
