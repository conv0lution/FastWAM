from __future__ import annotations

import numpy as np
import pytest

from experiments.asre_diagnosis.salvage_b.classification import (
    classify_functional_dissociation,
    classify_special_failure,
)
from experiments.asre_diagnosis.salvage_b.definitions import CONDITIONS
from experiments.asre_diagnosis.salvage_b.statistics import (
    action_recovery,
    analyze_world_losses,
    functional_dissociation,
    paired_bootstrap_ci,
    task_hierarchical_bootstrap_ci,
    world_endpoint_gate,
    world_recovery,
    world_recovery_statistics,
)


def _keys(n: int = 20) -> list[tuple[int, int, int]]:
    return [(index // 4, index // 2, index) for index in range(n)]


def _action_rates(*, r170: float = 0.96) -> dict[str, float]:
    return {
        "current_all": 0.95,
        "wrong_all": 0.0,
        "svd_r97": 0.79,
        "svd_r170": r170,
    }


def _world_losses(*, r170: float) -> dict[str, float]:
    return {
        "current_all": 1.0,
        "wrong_all": 3.0,
        "svd_r97": 2.0,
        "svd_r170": r170,
    }


def test_frozen_conditions_and_primary_order() -> None:
    assert CONDITIONS == ("current_all", "wrong_all", "svd_r97", "svd_r170")


def test_unclipped_recovery_formulas_and_zero_denominators() -> None:
    assert action_recovery(0.48, current_success=0.48, wrong_success=0.0) == 1.0
    assert action_recovery(0.60, current_success=0.50, wrong_success=0.0) == 1.2
    assert world_recovery(1.0, current_loss=1.0, wrong_loss=3.0) == 1.0
    assert world_recovery(3.0, current_loss=1.0, wrong_loss=3.0) == 0.0
    assert world_recovery(0.0, current_loss=1.0, wrong_loss=3.0) == 1.5
    assert world_recovery(4.0, current_loss=1.0, wrong_loss=3.0) == -0.5
    assert functional_dissociation(
        action_recovery_value=1.1, world_recovery_value=0.6
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="denominator"):
        action_recovery(0.5, current_success=0.5, wrong_success=0.5)
    with pytest.raises(ValueError, match="denominator"):
        world_recovery(1.0, current_loss=2.0, wrong_loss=2.0)


def test_bootstraps_are_seed_deterministic_and_hierarchical() -> None:
    values = np.linspace(-1.0, 1.0, 20)
    assert paired_bootstrap_ci(values, samples=200, seed=17) == paired_bootstrap_ci(
        values, samples=200, seed=17
    )
    first = task_hierarchical_bootstrap_ci(
        _keys(), values, samples=200, seed=19
    )
    second = task_hierarchical_bootstrap_ci(
        _keys(), values, samples=200, seed=19
    )
    assert first == second
    assert first[0] <= float(values.mean()) <= first[1]


def test_endpoint_gate_requires_positive_mean_and_paired_ci() -> None:
    current = np.ones(20)
    wrong = np.full(20, 2.0)
    passed = world_endpoint_gate(
        _keys(), current, wrong, bootstrap_samples=200, bootstrap_seed=7
    )
    assert passed["informative"] is True
    assert passed["paired_ci_low"] == 1.0
    assert passed["wrong_minus_current_median"] == 1.0

    failed = world_endpoint_gate(
        _keys(), current, current, bootstrap_samples=200, bootstrap_seed=7
    )
    assert failed["informative"] is False
    assert failed["classification"] == "WORLD-ENDPOINT-UNINFORMATIVE"

    # Median is descriptive; when a paired CI still supports degradation it
    # must not become an unregistered extra gate.
    sparse_wrong = current.copy()
    sparse_wrong[:9] = 20.0
    zero_median = world_endpoint_gate(
        _keys(), current, sparse_wrong, bootstrap_samples=200, bootstrap_seed=7
    )
    assert zero_median["wrong_minus_current_mean"] > 0.0
    assert zero_median["wrong_minus_current_median"] == 0.0
    assert zero_median["paired_ci_low"] > 0.0
    assert zero_median["informative"] is True


def test_world_recovery_bootstrap_is_joint_paired_and_unclipped() -> None:
    current = np.linspace(0.9, 1.1, 20)
    wrong = current + 2.0
    condition = current - 1.0
    result = world_recovery_statistics(
        _keys(),
        condition,
        current,
        wrong,
        bootstrap_samples=200,
        bootstrap_seed=31,
    )
    assert result["world_recovery"] == pytest.approx(1.5)
    assert result["paired_ci_low"] == pytest.approx(1.5)
    assert result["paired_ci_high"] == pytest.approx(1.5)
    assert result["unclipped"] is True


def test_analyze_world_losses_validates_pairing_and_reports_all_conditions() -> None:
    base = np.linspace(0.9, 1.1, 20)
    losses = {
        "current_all": base,
        "wrong_all": base + 2.0,
        "svd_r97": base + 1.0,
        "svd_r170": base + 0.5,
    }
    result = analyze_world_losses(
        _keys(), losses, bootstrap_samples=100, bootstrap_seed=5
    )
    assert result["endpoint_gate"]["informative"] is True
    assert set(result["conditions"]) == set(CONDITIONS)
    assert result["conditions"]["svd_r170"]["world_recovery"] == pytest.approx(
        0.75
    )
    with pytest.raises(ValueError, match="exactly"):
        analyze_world_losses(
            _keys(), {key: value for key, value in losses.items() if key != "svd_r97"}
        )
    with pytest.raises(ValueError, match="unique"):
        analyze_world_losses(
            [_keys()[0]] * 20, losses, bootstrap_samples=10
        )


def test_classification_strong_inclusive_boundaries() -> None:
    result = classify_functional_dissociation(
        action_success_rates=_action_rates(r170=0.95),
        world_mean_losses=_world_losses(r170=1.5),  # WR exactly 0.75
        primary_world_vs_current_paired_ci=(0.1, 0.9),
    )
    assert result["classification"] == "STRONG"
    assert result["recoveries"]["svd_r170"]["action_recovery"] == 1.0
    assert result["recoveries"]["svd_r170"]["world_recovery"] == 0.75


def test_classification_requires_ci_support_and_has_moderate_boundary() -> None:
    no_support = classify_functional_dissociation(
        action_success_rates=_action_rates(r170=0.95),
        world_mean_losses=_world_losses(r170=1.5),
        primary_world_vs_current_paired_ci=(0.0, 0.9),
    )
    assert no_support["classification"] == "WEAK"

    moderate = classify_functional_dissociation(
        action_success_rates=_action_rates(r170=0.85),  # exactly 10 pp below
        world_mean_losses=_world_losses(r170=1.3),  # WR exactly 0.85
        primary_world_vs_current_paired_ci=(0.05, 0.5),
    )
    assert moderate["classification"] == "MODERATE"


def test_classification_weak_near_current_and_tracking() -> None:
    near_current = classify_functional_dissociation(
        action_success_rates=_action_rates(),
        world_mean_losses=_world_losses(r170=1.2),  # WR = 0.90
        primary_world_vs_current_paired_ci=(0.01, 0.4),
    )
    assert near_current["classification"] == "WEAK"
    assert near_current["explicit_weak_world_near_current"] is True

    tracking = classify_functional_dissociation(
        action_success_rates=_action_rates(r170=0.76),  # AR=0.80
        world_mean_losses=_world_losses(r170=1.4),  # WR=0.80
        primary_world_vs_current_paired_ci=(0.01, 0.6),
    )
    assert tracking["classification"] == "WEAK"
    assert tracking["broad_action_world_tracking"] is True


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((False, False, False), "SHARED-INTERFACE-NOT-AVAILABLE"),
        ((True, False, False), "WORLD-METRIC-NOT-VALIDATABLE"),
        ((True, True, False), "WORLD-ENDPOINT-UNINFORMATIVE"),
        ((True, True, True), None),
    ],
)
def test_special_failure_precedence(
    flags: tuple[bool, bool, bool], expected: str | None
) -> None:
    assert classify_special_failure(
        shared_interface_available=flags[0],
        world_metric_validatable=flags[1],
        world_endpoint_informative=flags[2],
    ) == expected


def test_special_failure_short_circuits_without_numeric_inputs() -> None:
    result = classify_functional_dissociation(
        action_success_rates={},
        world_mean_losses={},
        primary_world_vs_current_paired_ci=(float("nan"), float("nan")),
        shared_interface_available=False,
    )
    assert result["classification"] == "SHARED-INTERFACE-NOT-AVAILABLE"
