from __future__ import annotations

import pytest

from experiments.asre_diagnosis.round4b.aggregate_results import (
    CONDITIONS,
    analyze_energy_capture,
    analyze_online_by_split,
    split_online_keys,
)


def _split() -> dict:
    return {
        "per_task": [
            {
                "task_id": task_id,
                "fit_episode_ids": list(range(8)),
                "holdout_episode_ids": [8, 9],
            }
            for task_id in range(10)
        ]
    }


def test_online_outcomes_are_stratified_by_frozen_80_20_split() -> None:
    keys = [
        ("libero_spatial", task_id, trial)
        for task_id in range(10)
        for trial in range(10)
    ]
    outcomes = {}
    for condition in CONDITIONS:
        if condition == "current_all":
            outcomes[condition] = {key: int(key[2] != 9) for key in keys}
        elif condition.startswith("svd_r"):
            outcomes[condition] = {key: int(key[2] < 8) for key in keys}
        else:
            outcomes[condition] = {key: 0 for key in keys}

    rows, contrasts, task_rows = analyze_online_by_split(
        outcomes, _split(), bootstrap_samples=50
    )
    lookup = {(row["split"], row["condition"]): row for row in rows}
    assert lookup[("calibration", "current_all")]["successes"] == 80
    assert lookup[("heldout", "current_all")]["successes"] == 10
    assert lookup[("calibration", "svd_r256")]["success_rate"] == 1.0
    assert lookup[("heldout", "svd_r256")]["success_rate"] == 0.0
    assert lookup[("heldout", "svd_r256")]["heldout_minus_calibration"] == -1.0
    assert len(rows) == 16
    assert len(task_rows) == 160
    assert sum(row.get("primary_matched_rank_contrast", False) for row in contrasts) == 6


def test_online_split_rejects_non_exhaustive_partition() -> None:
    keys = [
        ("libero_spatial", task_id, trial)
        for task_id in range(10)
        for trial in range(10)
    ]
    malformed = _split()
    malformed["per_task"][0]["holdout_episode_ids"] = [8, 8]
    with pytest.raises(ValueError, match="Malformed 8/2"):
        split_online_keys(malformed, keys)


def test_energy_capture_reports_weighted_and_per_matrix_values() -> None:
    ranks = (256, 768, 1536)
    fit_by_matrix = {}
    heldout_by_matrix = {}
    diagnostic_rows = []
    for layer in range(15, 30):
        for kind in ("k", "v"):
            matrix = f"layer{layer:02d}_{kind}"
            fit_total = float(layer * (1 if kind == "k" else 2))
            heldout_total = fit_total / 2
            fit_fractions = {
                str(rank): 0.5 + index * 0.2 for index, rank in enumerate(ranks)
            }
            heldout_fractions = {
                str(rank): fraction - 0.05
                for rank, fraction in zip(ranks, fit_fractions.values())
            }
            fit_by_matrix[matrix] = {
                "fit_rows": 100,
                "fit_total_energy": fit_total,
                "fit_captured_fraction": fit_fractions,
            }
            heldout_by_matrix[matrix] = {
                "total_energy": heldout_total,
                "captured_fraction": heldout_fractions,
                "captured_energy": {
                    str(rank): heldout_fractions[str(rank)] * heldout_total
                    for rank in ranks
                },
            }
            for rank in ranks:
                diagnostic_rows.append(
                    {
                        "matrix": matrix,
                        "rank": rank,
                        "effective_rank": 10.0,
                        "spectral_gap_ratio": 1.0,
                    }
                )
    diagnostics = {
        "ranks": list(ranks),
        "fit_by_matrix": fit_by_matrix,
        "heldout_by_matrix": heldout_by_matrix,
        "rows": diagnostic_rows,
    }

    summary, details = analyze_energy_capture(diagnostics)
    lookup = {(row["rank"], row["scope"]): row for row in summary}
    assert len(summary) == 9
    assert len(details) == 90
    assert lookup[(256, "all")]["calibration_weighted_captured_fraction"] == pytest.approx(0.5)
    assert lookup[(256, "all")]["heldout_weighted_captured_fraction"] == pytest.approx(0.45)
    assert lookup[(256, "all")][
        "weighted_generalization_gap_calibration_minus_heldout"
    ] == pytest.approx(0.05)
    assert {row["tensor_kind"] for row in details} == {"K", "V"}
