import json
from pathlib import Path

import numpy as np
import pytest

from experiments.asre_diagnosis.g0.aggregate_results import (
    analyze_suite,
    classify_findings,
    classify_overall,
    load_frozen_spatial,
    paired_bootstrap_ci,
    task_hierarchical_bootstrap_ci,
    validate_paired_alignment,
)
from experiments.asre_diagnosis.g0.definitions import CONDITION_ORDER
from experiments.asre_diagnosis.g0.plot_results import plot_all


def _synthetic_outcomes():
    suite = "libero_goal"
    outcomes = {condition: {} for condition in CONDITION_ORDER}
    descriptions = {task: f"task {task}" for task in range(10)}
    for task in range(10):
        for trial in range(10):
            key = (suite, task, trial)
            outcomes["full_current"][key] = 1
            outcomes["late_current_15_29"][key] = int(trial != 0)
            outcomes["early_current_00_19"][key] = int(trial < 2)
            outcomes["late_wrong_scene_15_29"][key] = 0
    return suite, outcomes, descriptions


def test_bootstrap_formulas_are_exact_for_constant_values() -> None:
    values = np.ones(100)
    assert paired_bootstrap_ci(values, samples=100, seed=1) == (1.0, 1.0)
    keys = [("suite", task, trial) for task in range(10) for trial in range(10)]
    assert task_hierarchical_bootstrap_ci(
        keys, values, samples=100, seed=1
    ) == (1.0, 1.0)


def test_paired_task_trial_alignment_and_contrast_formulas() -> None:
    suite, outcomes, descriptions = _synthetic_outcomes()
    analysis = analyze_suite(
        suite,
        outcomes,
        descriptions,
        bootstrap_samples=200,
        bootstrap_seed=7,
        provenance_label="test",
    )
    cells = {row["condition"]: row for row in analysis["condition_rows"]}
    contrasts = {row["contrast"]: row for row in analysis["contrast_rows"]}
    assert cells["full_current"]["success_rate"] == 1.0
    assert cells["late_current_15_29"]["success_rate"] == 0.9
    assert cells["early_current_00_19"]["success_rate"] == 0.2
    assert contrasts["delta_late"]["delta"] == -0.1
    assert contrasts["delta_early_vs_late"]["delta"] == -0.7
    assert contrasts["delta_wrong_vs_late"]["delta"] == -0.9
    assert sum(
        row["count"]
        for row in analysis["transition_rows"]
        if row["contrast"] == "delta_late"
    ) == 100
    outcomes["late_wrong_scene_15_29"].pop((suite, 0, 0))
    with pytest.raises(ValueError, match="Paired task/trial alignment failed"):
        validate_paired_alignment(outcomes)


def test_classification_thresholds() -> None:
    strong = classify_findings(
        full_rate=0.70,
        delta_late=-0.10,
        delta_early_vs_late=-0.20,
        delta_wrong_vs_late=-0.20,
        wrong_task_ci_high=-0.051,
        catastrophic_late_task_ids=[],
    )
    assert strong["finding_A"] == "strongly_consistent"
    assert strong["finding_B"] == "strongly_consistent"
    assert strong["finding_C"] == "strongly_consistent"
    assert strong["finding_C_task_hierarchical_support"] is True
    intermediate = classify_findings(
        full_rate=0.9,
        delta_late=-0.11,
        delta_early_vs_late=-0.10,
        delta_wrong_vs_late=-0.10,
        wrong_task_ci_high=0.0,
        catastrophic_late_task_ids=[],
    )
    assert intermediate["finding_B"] == "intermediate"
    assert intermediate["finding_C"] == "intermediate"
    low = classify_findings(
        full_rate=0.69,
        delta_late=0,
        delta_early_vs_late=0,
        delta_wrong_vs_late=0,
        wrong_task_ci_high=0,
        catastrophic_late_task_ids=[],
    )
    assert low["baseline_informative"] is False


def test_overall_classification_rules() -> None:
    strong = {
        "baseline_informative": True,
        "finding_A": "strongly_consistent",
        "finding_B": "strongly_consistent",
        "finding_C": "strongly_consistent",
    }
    assert classify_overall({"a": strong, "b": strong, "c": strong}) == "G0-STRONG"
    weak = {
        **strong,
        "finding_B": "weak_or_inconsistent",
        "finding_C": "weak_or_inconsistent",
    }
    assert classify_overall({"a": weak, "b": weak, "c": strong}) == "G0-WEAK / REASSESS"
    mixed = {**strong, "finding_B": "intermediate"}
    assert classify_overall({"a": strong, "b": mixed, "c": mixed}) == "G0-TASK-CONDITIONED"


def test_frozen_spatial_reference_import() -> None:
    outcomes, descriptions, provenance = load_frozen_spatial()
    assert set(outcomes) == set(CONDITION_ORDER)
    assert len(descriptions) == 10
    rates = {
        condition: np.mean(list(condition_outcomes.values()))
        for condition, condition_outcomes in outcomes.items()
    }
    assert rates == {
        "full_current": 0.97,
        "late_current_15_29": 0.95,
        "early_current_00_19": 0.12,
        "late_wrong_scene_15_29": 0.0,
    }
    assert provenance["label"] == "frozen prior result"


def test_all_four_preregistered_figures_are_generated(tmp_path) -> None:
    suite, outcomes, descriptions = _synthetic_outcomes()
    analysis = analyze_suite(
        suite,
        outcomes,
        descriptions,
        bootstrap_samples=20,
        bootstrap_seed=3,
        provenance_label="test",
    )
    summary = {
        "status": "complete",
        "suite_analyses": {
            name: analysis
            for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        },
    }
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    report = plot_all(summary_path=source, output_dir=tmp_path / "plots")
    assert len(report["figures"]) == 4
    assert all(Path(row["path"]).is_file() for row in report["figures"])
