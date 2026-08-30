"""Publish a fail-closed technical stop without launching later stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.asre_diagnosis.common import atomic_write_json, now_iso

from .definitions import PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    aggregate = args.output_root.resolve() / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "asre_salvage_b_v2_technical_stop",
        "protocol": PROTOCOL,
        "status": "stopped",
        "classification": args.classification,
        "reason": args.reason,
        "created_at": now_iso(),
        "old_factorized_results_reused": False,
        "later_experiment_launched": False,
    }
    atomic_write_json(aggregate / "salvage_b_v2_summary.json", payload)
    text = (
        "# Fast-WAM ASRE Salvage B v2 — Technical Stop\n\n"
        f"**{args.classification}**\n\n{args.reason}\n\n"
        "The quarantined factorized Salvage-B results were not reused. "
        "No later ASRE experiment was launched.\n"
    )
    (aggregate / "result_summary_for_gpt.md").write_text(text, encoding="utf-8")
    (aggregate / "salvage_b_v2_summary.md").write_text(text, encoding="utf-8")
