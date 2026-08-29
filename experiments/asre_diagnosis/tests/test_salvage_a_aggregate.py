from __future__ import annotations

from pathlib import Path

import pytest

from experiments.asre_diagnosis.salvage_a.aggregate_results import (
    _write_csv,
    analyze_online,
    split_online_keys,
    validate_diagnostics,
)
from experiments.asre_diagnosis.salvage_a.definitions import CONDITIONS
from experiments.asre_diagnosis.salvage_a.plot_results import plot


def _split() -> dict:
    return {
        "per_task": [
            {
                "task_id": task,
                "calibration_episode_ids": [0, 1, 2, 3, 4],
                "heldout_episode_ids": [5, 6, 7, 8, 9],
            }
            for task in range(10)
        ]
    }


def _keys() -> list[tuple[str, int, int]]:
    return [
        ("libero_spatial", task, trial)
        for task in range(10)
        for trial in range(10)
    ]


def _outcomes() -> dict[str, dict[tuple[str, int, int], int]]:
    keys = _keys()
    outcomes = {condition: {key: 0 for key in keys} for condition in CONDITIONS}
    for key in keys:
        heldout = key[2] >= 5
        outcomes["current_all"][key] = 1
        outcomes["wrong_all"][key] = 0
        outcomes["svd_r36"][key] = 0
        outcomes["random_r36"][key] = 0
        outcomes["svd_r97"][key] = int(heldout and key[2] != 5)
        outcomes["random_r97"][key] = 0
        # Held-out AA36 is 80%; calibration AA36 is deliberately 0%.
        outcomes["actionaware_r36"][key] = int(heldout and key[2] != 5)
        outcomes["actionaware_r97"][key] = int(heldout)
    return outcomes


def _diagnostics() -> dict:
    summary = []
    matrices = []
    overlap = []
    stability = []
    for family_index, family in enumerate(("svd", "actionaware", "random")):
        for rank in (36, 97):
            energy = 0.2 + 0.1 * family_index + rank / 1000
            sensitivity = 0.3 + 0.1 * family_index + rank / 1000
            summary.append(
                {
                    "basis_family": family,
                    "rank": rank,
                    "heldout_delta_z_energy_captured": energy,
                    "calibration_action_sensitivity_captured": sensitivity,
                }
            )
            for layer in range(15, 30):
                for kind in ("K", "V"):
                    matrices.append(
                        {
                            "matrix": f"layer{layer:02d}_{kind.lower()}",
                            "layer": layer,
                            "tensor_kind": kind,
                            "basis_family": family,
                            "rank": rank,
                            "heldout_delta_z_energy_captured": energy,
                            "calibration_action_sensitivity_captured": sensitivity,
                        }
                    )
    for rank in (36, 97):
        for layer in range(15, 30):
            for kind in ("K", "V"):
                common = {
                    "matrix": f"layer{layer:02d}_{kind.lower()}",
                    "layer": layer,
                    "tensor_kind": kind,
                    "rank": rank,
                    "subspace_overlap": 0.5,
                }
                overlap.append(dict(common))
                stability.append(dict(common, principal_angle_median_degrees=30.0))
    return {
        "basis_diagnostics_summary": summary,
        "basis_diagnostics_by_matrix": matrices,
        "aa_svd_overlap": overlap,
        "aa_half_stability": stability,
    }


def test_online_analysis_uses_heldout_50_only_for_decision() -> None:
    rows, contrasts, task_rows, decision = analyze_online(
        _outcomes(), _split(), bootstrap_samples=50
    )
    assert len(rows) == 24
    assert len(contrasts) == 18
    assert len(task_rows) == 240
    assert sum(row["primary_salvage_claim"] for row in contrasts) == 6
    lookup = {(row["scope"], row["condition"]): row for row in rows}
    assert lookup[("heldout", "actionaware_r36")]["success_rate"] == 0.8
    assert lookup[("calibration", "actionaware_r36")]["success_rate"] == 0.0
    assert decision["classification"] == "STRONG"
    assert decision["classification_input_scope"] == "heldout"
    assert decision["descriptive_scopes_excluded"] == ["calibration", "all"]


def test_split_mapping_rejects_any_non_5_5_partition() -> None:
    partitions = split_online_keys(_split(), _keys())
    assert len(partitions["calibration"]) == 50
    assert len(partitions["heldout"]) == 50
    malformed = _split()
    malformed["per_task"][0]["heldout_episode_ids"] = [5, 6, 7, 8, 8]
    with pytest.raises(ValueError, match="Malformed 5/5"):
        split_online_keys(malformed, _keys())


def test_registered_diagnostic_shapes_are_fail_closed() -> None:
    summary, matrices, overlap, stability = validate_diagnostics(_diagnostics())
    assert len(summary) == 6
    assert len(matrices) == 180
    assert len(overlap) == 60
    assert len(stability) == 60
    malformed = _diagnostics()
    malformed["aa_svd_overlap"].pop()
    with pytest.raises(ValueError, match="60"):
        validate_diagnostics(malformed)


def test_salvage_a_figures_render_from_registered_tables(tmp_path: Path) -> None:
    rows, contrasts, tasks, _decision = analyze_online(
        _outcomes(), _split(), bootstrap_samples=20
    )
    summary, _matrices, _overlap, _stability = validate_diagnostics(_diagnostics())
    _write_csv(tmp_path / "online_condition_summary.csv", rows)
    _write_csv(tmp_path / "online_contrasts.csv", contrasts)
    _write_csv(tmp_path / "task_success.csv", tasks)
    _write_csv(tmp_path / "basis_diagnostics_summary.csv", summary)
    paths = plot(tmp_path)
    assert len(paths) == 5
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
