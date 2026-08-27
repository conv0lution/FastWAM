from __future__ import annotations

import numpy as np
import pytest

from experiments.asre_diagnosis.round3b.statistics import (
    CONDITION_ORDER,
    analyze_outcomes,
    classify_decision,
    exact_mcnemar_p_value,
    paired_bootstrap_ci,
)


def _outcomes(correct: list[int], wrong: list[int], no_video: list[int]):
    keys = [("libero_spatial", index // 10, index % 10) for index in range(len(correct))]
    vectors = (correct, wrong, no_video)
    return {
        condition: dict(zip(keys, values))
        for condition, values in zip(CONDITION_ORDER, vectors)
    }


def test_analysis_preserves_direction_and_transitions() -> None:
    correct = [1] * 80 + [0] * 20
    wrong = [1] * 30 + [0] * 70
    no_video = [0] * 100
    result = analyze_outcomes(
        _outcomes(correct, wrong, no_video),
        bootstrap_samples=500,
        bootstrap_seed=7,
    )

    primary = result["comparisons"]["wrong_minus_correct"]
    assert result["success_rates"]["late_current_correct"] == pytest.approx(0.8)
    assert primary["delta_success_rate"] == pytest.approx(-0.5)
    assert primary["reference_success_to_target_failure"] == 50
    assert primary["reference_failure_to_target_success"] == 0
    assert result["content_gap_fraction"] == pytest.approx(0.625)
    assert result["decision"]["classification"] == "GO"


def test_decision_gate_has_all_three_outcomes() -> None:
    assert classify_decision(
        correct_success_rate=0.95,
        wrong_success_rate=0.60,
        no_video_success_rate=0.0,
        wrong_minus_correct_task_ci=(-0.50, -0.10),
    )["classification"] == "GO"
    assert classify_decision(
        correct_success_rate=0.95,
        wrong_success_rate=0.93,
        no_video_success_rate=0.0,
        wrong_minus_correct_task_ci=(-0.08, 0.04),
    )["classification"] == "STOP"
    assert classify_decision(
        correct_success_rate=0.95,
        wrong_success_rate=0.82,
        no_video_success_rate=0.0,
        wrong_minus_correct_task_ci=(-0.30, 0.03),
    )["classification"] == "VERIFY"


def test_exact_mcnemar_and_degenerate_bootstrap() -> None:
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(8, 0) == pytest.approx(2.0 / 256.0)
    assert paired_bootstrap_ci(np.zeros(10), samples=50, seed=0) == (0.0, 0.0)


def test_rejects_unpaired_or_nonbinary_outcomes() -> None:
    data = _outcomes([1, 0], [0, 0], [0, 0])
    data["late_wrong_scene"].pop(("libero_spatial", 0, 1))
    with pytest.raises(ValueError, match="keys differ"):
        analyze_outcomes(data, bootstrap_samples=10)

    data = _outcomes([1, 0], [2, 0], [0, 0])
    with pytest.raises(ValueError, match="binary"):
        analyze_outcomes(data, bootstrap_samples=10)
