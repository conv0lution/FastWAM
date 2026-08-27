"""Freeze and audit the Round-3B 499-state offline donor mapping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import atomic_write_json, sha256_file  # noqa: E402
from experiments.asre_diagnosis.round3b.offline_donor import (  # noqa: E402
    build_offline_donor_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-bank-dir",
        type=Path,
        default=PROJECT_ROOT / "asre_results/state_bank",
    )
    parser.add_argument(
        "--valid-manifest",
        type=Path,
        default=PROJECT_ROOT / "asre_results/round2/state_bank_valid_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "asre_results/round3b/donors/offline_donor_mapping.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    state_bank_dir = args.state_bank_dir.resolve()
    payload = build_offline_donor_manifest(
        state_bank_dir=state_bank_dir,
        source_manifest_path=state_bank_dir / "manifest.jsonl",
        valid_manifest_path=args.valid_manifest.resolve(),
    )
    output = args.output.resolve()
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise FileExistsError(
                f"Refusing to overwrite incompatible offline donor mapping: {output}"
            )
        print(f"Reused identical frozen offline donor mapping: {output}")
    else:
        atomic_write_json(output, payload)
        print(f"Wrote frozen offline donor mapping: {output}")
    print(
        json.dumps(
            {
                "num_pairs": payload["num_pairs"],
                "mapping_sha256": sha256_file(output),
                "derangement_verified": payload["derangement_verified"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
