"""Merge fit and strict held-out Round-4B subspace energy diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4B_PROTOCOL,
    atomic_write_json,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    LATE_LAYERS,
    RANKS,
    validate_basis_manifest,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def merge(*, basis_manifest: Path, holdout_dir: Path, output: Path) -> dict[str, Any]:
    basis_manifest = basis_manifest.resolve()
    basis = validate_basis_manifest(basis_manifest)
    reports = [_read(holdout_dir / f"holdout_shard{index}.json") for index in range(4)]
    if any(
        report.get("status") != "complete" or report.get("phase") != "holdout"
        for report in reports
    ):
        raise ValueError("Held-out Round-4B diagnostic shards are incomplete.")
    heldout: dict[str, Any] = {}
    for report in reports:
        for key, record in report["artifacts"].items():
            if key in heldout:
                raise ValueError(f"Duplicate held-out diagnostic: {key}")
            heldout[key] = record
    expected = {
        f"layer{layer:02d}_{kind}" for layer in LATE_LAYERS for kind in ("k", "v")
    }
    if set(heldout) != expected:
        raise ValueError("Held-out diagnostics do not cover every late-layer K/V pair.")
    rows = []
    for key in sorted(expected):
        fit = basis["fit_diagnostics"][key]
        hold = heldout[key]
        for rank in RANKS:
            rows.append(
                {
                    "matrix": key,
                    "rank": rank,
                    "fit_captured_fraction": fit["fit_captured_fraction"][str(rank)],
                    "holdout_captured_fraction": hold["captured_fraction"][str(rank)],
                    "generalization_gap": (
                        fit["fit_captured_fraction"][str(rank)]
                        - hold["captured_fraction"][str(rank)]
                    ),
                    "effective_rank": fit["effective_rank"],
                    "spectral_gap_ratio": fit["spectral_gap_ratio"][str(rank)],
                }
            )
    payload = {
        "artifact_type": "asre_round4b_subspace_diagnostics",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "status": "complete",
        "created_at": now_iso(),
        "basis_manifest_path": str(basis_manifest),
        "basis_manifest_sha256": sha256_file(basis_manifest),
        "split_path": basis["split_path"],
        "split_sha256": basis["split_sha256"],
        "fit_is_uncentered": True,
        "holdout_never_used_for_fitting": True,
        "ranks": list(RANKS),
        "rows": rows,
        "heldout_by_matrix": heldout,
        "fit_by_matrix": basis["fit_diagnostics"],
        "holdout_shard_reports": [
            {
                "path": str((holdout_dir / f"holdout_shard{index}.json").resolve()),
                "sha256": sha256_file(holdout_dir / f"holdout_shard{index}.json"),
            }
            for index in range(4)
        ],
    }
    atomic_write_json(output.resolve(), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(
        basis_manifest=args.basis_manifest,
        holdout_dir=args.holdout_dir.resolve(),
        output=args.output,
    )
    print(f"Round-4B diagnostics complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
