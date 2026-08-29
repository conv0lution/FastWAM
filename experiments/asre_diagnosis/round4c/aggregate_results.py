"""Strict paired online aggregation for ASRE Stage-2 Round-4C."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4C_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round3b.outcome_statistics import (  # noqa: E402
    comparison_statistics,
)
from experiments.asre_diagnosis.round4c.classification import (  # noqa: E402
    classify_action_sufficiency,
)
from experiments.asre_diagnosis.round4c.definitions import (  # noqa: E402
    CONDITIONS,
    CUMULATIVE_ENERGY_COMMIT,
    DISPLAY,
    FEATURE_DIM,
    RANKS,
    REGISTERED_HELDOUT_ENERGY,
    REGISTERED_R256_HELDOUT_ENERGY,
    ROUND4B_SOURCE_COMMIT,
    WAVES,
    validate_energy_candidates,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value.rstrip() + "\n")
    os.replace(temporary, path)


def load_online(
    wave1: Path, wave2: Path
) -> tuple[dict[str, dict[tuple[str, int, int], int]], list[dict[str, Any]]]:
    for wave, root in ((1, wave1), (2, wave2)):
        summary = _read(root / "launcher_summary.json")
        if not (
            summary.get("protocol") == ROUND4C_PROTOCOL
            and summary.get("mode") == "full"
            and summary.get("wave") == wave
            and summary.get("all_succeeded") is True
        ):
            raise ValueError(f"Round-4C full wave {wave} is incomplete.")
    outcomes: dict[str, dict[tuple[str, int, int], int]] = {}
    task_rows: list[dict[str, Any]] = []
    shared_identity: dict[str, Any] | None = None
    identity_keys = (
        "git_commit_hash",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "state_bank_manifest_sha256",
        "valid_state_bank_manifest_sha256",
        "prompt_context_cache_sha256",
        "donor_mapping_sha256",
        "donor_observation_manifest_sha256",
        "preflight_report_sha256",
        "machinery_report_sha256",
        "calibration_split_manifest_sha256",
        "subspace_basis_manifest_sha256",
        "subspace_diagnostics_sha256",
        "energy_analysis_manifest_sha256",
        "energy_candidate_ranks_sha256",
        "round4b_summary_sha256",
        "round4b_source_commit",
        "cumulative_energy_analysis_commit",
    )
    for index, condition in enumerate(CONDITIONS):
        root = wave1 if index in WAVES[1] else wave2
        directory = root / condition
        metadata = _read(directory / "run_metadata.json")
        if not (
            metadata.get("status") == "completed"
            and metadata.get("condition_protocol") == ROUND4C_PROTOCOL
            and metadata.get("diagnosis_condition") == condition
            and metadata.get("task_ids") == list(range(10))
            and metadata.get("number_of_trials") == 10
            and metadata.get("seed") == 42
        ):
            raise ValueError(f"Round-4C online metadata mismatch: {condition}")
        identity = {key: metadata.get(key) for key in identity_keys}
        if any(value is None or str(value) == "" for value in identity.values()):
            raise ValueError(f"Round-4C provenance is incomplete: {condition}")
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError(f"Round-4C provenance differs across conditions: {condition}")
        expected_rank = None if not condition.startswith("svd_r") else int(
            condition.removeprefix("svd_r")
        )
        expected_kind = None if expected_rank is None else "svd"
        condition_config = metadata.get("condition_config")
        if (
            not isinstance(condition_config, dict)
            or condition_config.get("condition_name") != condition
            or condition_config.get("subspace_basis_kind") != expected_kind
            or condition_config.get("subspace_rank") != expected_rank
        ):
            raise ValueError(f"Round-4C rank provenance mismatch: {condition}")
        files = sorted((directory / "libero_spatial").glob("gpu*_task*_results.json"))
        if len(files) != 10:
            raise ValueError(f"{condition} lacks ten task result files.")
        values: dict[tuple[str, int, int], int] = {}
        for path in files:
            result = _read(path)
            task_id = int(result["task_id"])
            successes = set(map(int, result["success_episodes"]))
            failures = set(map(int, result["failure_episodes"]))
            if successes & failures or successes | failures != set(range(10)):
                raise ValueError(f"Malformed paired outcomes: {path}")
            for trial in range(10):
                values[("libero_spatial", task_id, trial)] = int(trial in successes)
            task_rows.append(
                {
                    "condition": condition,
                    "display_name": DISPLAY[condition],
                    "task_id": task_id,
                    "task_description": result["task_description"],
                    "successes": len(successes),
                    "trials": 10,
                    "success_rate": len(successes) / 10,
                }
            )
        outcomes[condition] = values
    keys = set(outcomes["current_all"])
    if len(keys) != 100 or any(set(values) != keys for values in outcomes.values()):
        raise ValueError("Round-4C outcomes are not paired over exactly 100 episodes.")
    return outcomes, task_rows


def analyze_online(
    outcomes: Mapping[str, Mapping[tuple[str, int, int], int]],
    task_rows: list[dict[str, Any]],
    *,
    round4b_r256_success: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    keys = sorted(outcomes["current_all"])
    arrays = {
        condition: np.asarray([outcomes[condition][key] for key in keys], dtype=np.int8)
        for condition in CONDITIONS
    }
    zero = np.zeros(len(keys), dtype=np.int8)
    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        rate = comparison_statistics(
            keys,
            zero,
            arrays[condition],
            bootstrap_samples=10_000,
            bootstrap_seed=4208,
            seed_label=f"rate-{condition}",
        )
        current = comparison_statistics(
            keys,
            arrays["current_all"],
            arrays[condition],
            bootstrap_samples=10_000,
            bootstrap_seed=4208,
            seed_label=f"current-{condition}",
        )
        wrong = comparison_statistics(
            keys,
            arrays["wrong_all"],
            arrays[condition],
            bootstrap_samples=10_000,
            bootstrap_seed=4208,
            seed_label=f"wrong-{condition}",
        )
        summary_rows.append(
            {
                "condition": condition,
                "display_name": DISPLAY[condition],
                "successes": int(arrays[condition].sum()),
                "episodes": len(keys),
                "success_rate": float(arrays[condition].mean()),
                "paired_ci_low": rate["paired_ci_low"],
                "paired_ci_high": rate["paired_ci_high"],
                "task_hierarchical_ci_low": rate["task_hierarchical_ci_low"],
                "task_hierarchical_ci_high": rate["task_hierarchical_ci_high"],
                "delta_vs_current": current["delta_success_rate"],
                "delta_vs_wrong": wrong["delta_success_rate"],
            }
        )
        for reference, stats in (("current_all", current), ("wrong_all", wrong)):
            contrast_rows.append(
                {
                    "comparison": f"{condition}_minus_{reference}",
                    "reference_condition": reference,
                    "target_condition": condition,
                    "primary_rank_contrast": condition.startswith("svd_r"),
                    **stats,
                }
            )
    rates = {row["condition"]: row["success_rate"] for row in summary_rows}
    task_success = {
        condition: {
            int(row["task_id"]): float(row["success_rate"])
            for row in task_rows
            if row["condition"] == condition
        }
        for condition in CONDITIONS
    }
    classification = classify_action_sufficiency(
        success_rates=rates,
        task_success=task_success,
        round4b_r256_success=round4b_r256_success,
    )
    for row in task_rows:
        task_id = int(row["task_id"])
        row["delta_vs_current"] = row["success_rate"] - task_success["current_all"][task_id]
        if row["condition"] in {"svd_r36", "svd_r97", "svd_r170"}:
            row["rank"] = int(row["condition"].removeprefix("svd_r"))
    monotonicity = []
    for task_id in range(10):
        values = [task_success[f"svd_r{rank}"][task_id] for rank in RANKS]
        monotonicity.append(
            {
                "task_id": task_id,
                "r36_success": values[0],
                "r97_success": values[1],
                "r170_success": values[2],
                "behavior_monotonic_nondecreasing": values[0] <= values[1] <= values[2],
            }
        )
    classification["task_rank_monotonicity"] = monotonicity
    classification["nonmonotonic_task_ids"] = [
        row["task_id"] for row in monotonicity if not row["behavior_monotonic_nondecreasing"]
    ]
    return summary_rows, contrast_rows, classification


def _round4b_reference(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if summary.get("status") != "complete":
        raise ValueError("Round-4B summary is incomplete.")
    rows = {row["condition"]: row for row in summary["online_condition_summary"]}
    required = {"current_all", "wrong_all", "svd_r256"}
    if not required.issubset(rows):
        raise ValueError("Round-4B summary lacks frozen endpoint/r256 references.")
    return {key: rows[key] for key in required}


def build_curve_and_saturation(
    online_rows: Sequence[Mapping[str, Any]],
    round4b: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observed_energy = validate_energy_candidates(candidates)
    rows = {row["condition"]: row for row in online_rows}
    points = [
        ("wrong_all", 0, 0.0, "Round 4C", True, rows["wrong_all"]),
        *[
            (
                f"svd_r{rank}",
                rank,
                observed_energy[rank],
                "Round 4C",
                True,
                rows[f"svd_r{rank}"],
            )
            for rank in RANKS
        ],
        (
            "svd_r256",
            256,
            REGISTERED_R256_HELDOUT_ENERGY,
            "Round 4B frozen reference",
            False,
            round4b["svd_r256"],
        ),
        ("current_all", FEATURE_DIM, 1.0, "Round 4C", True, rows["current_all"]),
    ]
    curve = [
        {
            "condition": condition,
            "rank": rank,
            "rank_fraction": rank / FEATURE_DIM,
            "heldout_delta_z_energy": energy,
            "successes": source.get("successes"),
            "episodes": source.get("episodes", 100),
            "success_rate": float(source["success_rate"]),
            "paired_ci_low": float(source["paired_ci_low"]),
            "paired_ci_high": float(source["paired_ci_high"]),
            "source_round": source_round,
            "newly_run_round4c": newly_run,
        }
        for condition, rank, energy, source_round, newly_run, source in points
    ]
    current = float(rows["current_all"]["success_rate"])
    saturation = []
    for point in curve:
        if point["condition"] not in {"wrong_all", "current_all"}:
            saturation.append(
                {
                    "condition": point["condition"],
                    "rank": point["rank"],
                    "heldout_delta_z_energy": point["heldout_delta_z_energy"],
                    "success_rate": point["success_rate"],
                    "action_retention_ratio": (
                        None if current <= 0.0 else point["success_rate"] / current
                    ),
                    "success_gap_current_minus_condition": current - point["success_rate"],
                    "descriptive_not_causal_fraction": True,
                    "source_round": point["source_round"],
                }
            )
    reproducibility = []
    for condition in ("current_all", "wrong_all"):
        old = float(round4b[condition]["success_rate"])
        new = float(rows[condition]["success_rate"])
        reproducibility.append(
            {
                "condition": condition,
                "round4b_success_rate": old,
                "round4c_success_rate": new,
                "round4c_minus_round4b": new - old,
            }
        )
    return curve, saturation, reproducibility


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Fast-WAM ASRE Stage 2 — Round 4C Final Report",
        "",
        "Round 4C is complete. No later experiment was launched.",
        "",
        "## Five-condition online result",
        "",
        "| Condition | Success | Paired 95% CI | Task-hierarchical 95% CI | Δ current | Δ wrong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["online_condition_summary"]:
        lines.append(
            f"| {row['display_name']} | {row['successes']}/100 ({row['success_rate']:.1%}) | "
            f"{row['paired_ci_low']:.1%}–{row['paired_ci_high']:.1%} | "
            f"{row['task_hierarchical_ci_low']:.1%}–{row['task_hierarchical_ci_high']:.1%} | "
            f"{row['delta_vs_current']:+.1%} | {row['delta_vs_wrong']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "Exact paired transition tables and McNemar tests are in `online_contrasts.csv`.",
            "",
            "## Energy versus closed-loop success",
            "",
            "| Energy | Rank | Success | Source |",
            "|---:|---:|---:|---|",
        ]
    )
    for row in payload["energy_success_curve"]:
        lines.append(
            f"| {row['heldout_delta_z_energy']:.2%} | {row['rank']} | "
            f"{row['success_rate']:.1%} | {row['source_round']} |"
        )
    lines.extend(
        [
            "",
            "The r256 point is a frozen Round-4B reference and was not pooled as a "
            "new Round-4C observation.",
            "",
            "## Action saturation",
            "",
            "| Condition | Action retention | Success gap vs current |",
            "|---|---:|---:|",
        ]
    )
    for row in payload["action_saturation"]:
        retention = row["action_retention_ratio"]
        retention_text = "undefined" if retention is None else f"{retention:.1%}"
        lines.append(
            f"| {row['condition']} | {retention_text} | "
            f"{row['success_gap_current_minus_condition']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "Action-retention ratios are descriptive saturation statistics, not causal fractions.",
            "",
            "## Current/wrong reproducibility versus Round 4B",
            "",
            "| Condition | Round 4B | Round 4C | Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["current_wrong_reproducibility"]:
        lines.append(
            f"| {row['condition']} | {row['round4b_success_rate']:.1%} | "
            f"{row['round4c_success_rate']:.1%} | {row['round4c_minus_round4b']:+.1%} |"
        )
    decision = payload["classification"]
    recommendation = payload["recommendation"]
    lines.extend(
        [
            "",
            "## Task-level behavior",
            "",
            f"Non-monotonic task IDs across r36→r97→r170: "
            f"`{decision['nonmonotonic_task_ids']}`.",
            "The complete task × condition matrix is in `task_success.csv`; task differences "
            "are reported descriptively without semantic stage labels.",
            "",
            "## Decision",
            "",
            f"**{decision['classification']}**",
            "",
            recommendation["interpretation"],
            "",
            f"Recommended next step: {recommendation['next_step']}",
            "",
            "## Frozen inputs and machinery",
            "",
            f"- Frozen Round-4B source commit: `{ROUND4B_SOURCE_COMMIT}`.",
            f"- Cumulative-energy analysis commit: `{CUMULATIVE_ENERGY_COMMIT}`.",
            f"- Frozen basis: `{payload['provenance']['basis_manifest_path']}` "
            f"(`{payload['provenance']['basis_manifest_sha256']}`).",
            f"- Frozen split: `{payload['provenance']['split_manifest_path']}` "
            f"(`{payload['provenance']['split_manifest_sha256']}`).",
            f"- Frozen donor mapping: `{payload['provenance']['donor_mapping_path']}` "
            f"(`{payload['provenance']['donor_mapping_sha256']}`).",
            f"- Machinery passed: `{payload['machinery']['passed']}`; exact rank prefixes, "
            "98 tokens, 24 heads, unchanged K/V shapes, finite `[32,7]` actions.",
            "- Frozen Round-4B SVD basis was reused without refitting.",
            "- Optional offline action replay was not run; closed-loop success is primary.",
            "",
            "## Interpretation limits",
            "",
            "Energy means Frobenius Delta-Z energy. It is not semantic information, an "
            "action-subspace label, a minimality result, a compute-reduction claim, or "
            "cross-suite generalization.",
            "",
            "## Stop rule",
            "",
            "No additional rank, cross-suite, RoboTwin, random-control, token/head/channel, "
            "supervised-subspace, semantic-probe, reconstruction, or later-stage experiment "
            "was launched.",
        ]
    )
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> Path:
    preflight = _read(args.preflight.resolve())
    machinery = _read(args.machinery.resolve())
    if preflight.get("status") != "compatible" or machinery.get("passed") is not True:
        raise ValueError("Round-4C preflight/machinery gate is incomplete.")
    expected_links = {
        "energy_manifest": (
            args.energy_manifest.resolve(),
            preflight["cumulative_energy"]["manifest_path"],
            preflight["cumulative_energy"]["manifest_sha256"],
        ),
        "energy_candidates": (
            args.energy_candidates.resolve(),
            preflight["cumulative_energy"]["candidate_ranks_path"],
            preflight["cumulative_energy"]["candidate_ranks_sha256"],
        ),
        "round4b_summary": (
            args.round4b_summary.resolve(),
            preflight["frozen_round4b"]["summary_path"],
            preflight["frozen_round4b"]["summary_sha256"],
        ),
    }
    for label, (path, expected_path, expected_sha256) in expected_links.items():
        if str(path) != expected_path or sha256_file(path) != expected_sha256:
            raise ValueError(f"Round-4C frozen {label} provenance drifted.")
    if machinery.get("preflight_report_sha256") != sha256_file(args.preflight.resolve()):
        raise ValueError("Round-4C machinery report belongs to another preflight.")
    round4b_summary = _read(args.round4b_summary.resolve())
    round4b = _round4b_reference(round4b_summary)
    outcomes, task_rows = load_online(args.online_wave1.resolve(), args.online_wave2.resolve())
    online_rows, contrast_rows, classification = analyze_online(
        outcomes,
        task_rows,
        round4b_r256_success=float(round4b["svd_r256"]["success_rate"]),
    )
    candidates = _read(args.energy_candidates.resolve())
    curve, saturation, reproducibility = build_curve_and_saturation(
        online_rows, round4b, candidates
    )
    decision = classification["classification"]
    recommendation = {
        "STRONG": {
            "interpretation": (
                "A compact dominant scene-specific feature subspace is sufficient for action "
                "behavior, and success saturates before reconstruction of the full scene-induced "
                "representation variation."
            ),
            "next_step": "consider a separately authorized cross-suite frozen-subspace confirmation",
        },
        "MODERATE": {
            "interpretation": (
                "Approximately 80% of scene-specific Delta-Z energy is required for near-baseline "
                "behavior; the separation from generic reconstruction is present but weaker."
            ),
            "next_step": "reassess Paper 1 strength before any expansion",
        },
        "WEAK": {
            "interpretation": (
                "Action behavior remains consistent with tracking reconstruction of the dominant "
                "scene-specific representation manifold; no post-hoc rank sweep is warranted."
            ),
            "next_step": "stop and reassess Paper 1",
        },
    }[decision]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "online_condition_summary.csv", online_rows)
    _write_csv(output / "online_contrasts.csv", contrast_rows)
    _write_csv(output / "task_success.csv", task_rows)
    _write_csv(output / "energy_success_curve.csv", curve)
    _write_csv(output / "action_saturation.csv", saturation)
    _write_csv(output / "current_wrong_reproducibility.csv", reproducibility)
    payload = {
        "artifact_type": "asre_round4c_aggregate",
        "schema_version": 1,
        "protocol": ROUND4C_PROTOCOL,
        "status": "complete",
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "online_condition_summary": online_rows,
        "online_contrasts": contrast_rows,
        "task_success": task_rows,
        "energy_success_curve": curve,
        "action_saturation": saturation,
        "current_wrong_reproducibility": reproducibility,
        "classification": classification,
        "recommendation": recommendation,
        "registered_ranks": list(RANKS),
        "registered_heldout_energy": {
            str(rank): REGISTERED_HELDOUT_ENERGY[rank] for rank in RANKS
        },
        "bootstrap": {
            "samples": 10_000,
            "seed": 4208,
            "paired_and_task_hierarchical": True,
        },
        "machinery": machinery,
        "optional_offline_action_analysis_run": False,
        "provenance": {
            "round4b_source_commit": ROUND4B_SOURCE_COMMIT,
            "cumulative_energy_analysis_commit": CUMULATIVE_ENERGY_COMMIT,
            "preflight_path": str(args.preflight.resolve()),
            "preflight_sha256": sha256_file(args.preflight.resolve()),
            "machinery_path": str(args.machinery.resolve()),
            "machinery_sha256": sha256_file(args.machinery.resolve()),
            "basis_manifest_path": preflight["frozen_round4b"]["basis_manifest_path"],
            "basis_manifest_sha256": preflight["frozen_round4b"]["basis_manifest_sha256"],
            "split_manifest_path": preflight["frozen_round4b"]["split_manifest_path"],
            "split_manifest_sha256": preflight["frozen_round4b"]["split_manifest_sha256"],
            "donor_mapping_path": preflight["donors"]["mapping_path"],
            "donor_mapping_sha256": preflight["donors"]["mapping_sha256"],
            "energy_manifest_path": args.energy_manifest.resolve().as_posix(),
            "energy_manifest_sha256": sha256_file(args.energy_manifest.resolve()),
            "energy_candidates_path": args.energy_candidates.resolve().as_posix(),
            "energy_candidates_sha256": sha256_file(args.energy_candidates.resolve()),
            "round4b_summary_path": args.round4b_summary.resolve().as_posix(),
            "round4b_summary_sha256": sha256_file(args.round4b_summary.resolve()),
            "online_wave1_summary_path": str(args.online_wave1.resolve() / "launcher_summary.json"),
            "online_wave1_summary_sha256": sha256_file(
                args.online_wave1.resolve() / "launcher_summary.json"
            ),
            "online_wave2_summary_path": str(args.online_wave2.resolve() / "launcher_summary.json"),
            "online_wave2_summary_sha256": sha256_file(
                args.online_wave2.resolve() / "launcher_summary.json"
            ),
        },
        "later_stage_launched": False,
        "stop_rule_applied": True,
    }
    json_path = output / "round4c_summary.json"
    atomic_write_json(json_path, payload)
    report = _markdown(payload)
    report_path = output / "round4c_summary.md"
    _write_text(report_path, report)
    _write_text(output / "result_summary_for_gpt.md", report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-wave1", type=Path, required=True)
    parser.add_argument("--online-wave2", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--machinery", type=Path, required=True)
    parser.add_argument("--energy-manifest", type=Path, required=True)
    parser.add_argument("--energy-candidates", type=Path, required=True)
    parser.add_argument("--round4b-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    path = aggregate(parser.parse_args())
    print(f"Round-4C aggregation complete: {path}")


if __name__ == "__main__":
    main()
