"""Pre-registered Salvage-B functional-dissociation classification."""

from __future__ import annotations

import math
from typing import Any, Mapping

from experiments.asre_diagnosis.salvage_b.definitions import (
    BROAD_TRACKING_GAP_MAX,
    CONDITIONS,
    MODERATE_ACTION_CURRENT_GAP_MAX,
    MODERATE_WORLD_RECOVERY_MAX,
    PRIMARY_CONDITION,
    STRONG_ACTION_CURRENT_GAP_MAX,
    STRONG_ACTION_RECOVERY_MIN,
    STRONG_WORLD_RECOVERY_MAX,
    WEAK_WORLD_RECOVERY_MIN,
)
from experiments.asre_diagnosis.salvage_b.statistics import (
    action_recovery,
    functional_dissociation,
    world_recovery,
)


def _finite_mapping(
    values: Mapping[str, float], *, name: str, probability: bool = False
) -> dict[str, float]:
    if set(values) != set(CONDITIONS):
        raise ValueError(
            f"{name} must contain exactly {list(CONDITIONS)}, got {sorted(values)}."
        )
    result = {condition: float(values[condition]) for condition in CONDITIONS}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"{name} must contain only finite values.")
    if probability and any(value < 0.0 or value > 1.0 for value in result.values()):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    return result


def classify_special_failure(
    *,
    shared_interface_available: bool,
    world_metric_validatable: bool,
    world_endpoint_informative: bool,
) -> str | None:
    """Return the first protocol-ordered technical failure, if any."""

    flags = (
        shared_interface_available,
        world_metric_validatable,
        world_endpoint_informative,
    )
    if any(not isinstance(value, bool) for value in flags):
        raise ValueError("Special-failure inputs must be booleans.")
    if not shared_interface_available:
        return "SHARED-INTERFACE-NOT-AVAILABLE"
    if not world_metric_validatable:
        return "WORLD-METRIC-NOT-VALIDATABLE"
    if not world_endpoint_informative:
        return "WORLD-ENDPOINT-UNINFORMATIVE"
    return None


def classify_functional_dissociation(
    *,
    action_success_rates: Mapping[str, float],
    world_mean_losses: Mapping[str, float],
    primary_world_vs_current_paired_ci: tuple[float, float],
    shared_interface_available: bool = True,
    world_metric_validatable: bool = True,
    world_endpoint_informative: bool = True,
) -> dict[str, Any]:
    """Classify the primary r170 result and summarize both frozen ranks.

    ``primary_world_vs_current_paired_ci`` is the paired interval for
    ``L(r170) - L(current)``.  Its lower endpoint must be strictly positive to
    provide the protocol's credible-degradation support for STRONG/MODERATE.
    Point-estimate thresholds are inclusive; CI support is strict.
    """

    special = classify_special_failure(
        shared_interface_available=shared_interface_available,
        world_metric_validatable=world_metric_validatable,
        world_endpoint_informative=world_endpoint_informative,
    )
    if special is not None:
        return {
            "classification": special,
            "special_failure": True,
            "primary_condition": PRIMARY_CONDITION,
            "stop_unconditionally": True,
        }

    action = _finite_mapping(
        action_success_rates, name="action_success_rates", probability=True
    )
    world = _finite_mapping(world_mean_losses, name="world_mean_losses")
    ci_low, ci_high = (float(value) for value in primary_world_vs_current_paired_ci)
    if not math.isfinite(ci_low) or not math.isfinite(ci_high) or ci_low > ci_high:
        raise ValueError("Primary world-vs-current paired interval is invalid.")

    recoveries: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        action_value = action_recovery(
            action[condition],
            current_success=action["current_all"],
            wrong_success=action["wrong_all"],
        )
        world_value = world_recovery(
            world[condition],
            current_loss=world["current_all"],
            wrong_loss=world["wrong_all"],
        )
        recoveries[condition] = {
            "action_recovery": action_value,
            "world_recovery": world_value,
            "functional_dissociation": functional_dissociation(
                action_recovery_value=action_value,
                world_recovery_value=world_value,
            ),
        }

    primary = recoveries[PRIMARY_CONDITION]
    action_gap = abs(
        action[PRIMARY_CONDITION] - action["current_all"]
    )
    credible_world_degradation = ci_low > 0.0
    strong = (
        action_gap <= STRONG_ACTION_CURRENT_GAP_MAX
        and primary["action_recovery"] >= STRONG_ACTION_RECOVERY_MIN
        and primary["world_recovery"] <= STRONG_WORLD_RECOVERY_MAX
        and credible_world_degradation
    )
    moderate = (
        not strong
        and action_gap <= MODERATE_ACTION_CURRENT_GAP_MAX
        and primary["world_recovery"] <= MODERATE_WORLD_RECOVERY_MAX
        and credible_world_degradation
    )
    explicit_weak_near_current = (
        primary["world_recovery"] >= WEAK_WORLD_RECOVERY_MIN
    )
    broad_tracking = (
        abs(primary["functional_dissociation"]) <= BROAD_TRACKING_GAP_MAX
    )
    classification = "STRONG" if strong else "MODERATE" if moderate else "WEAK"
    return {
        "classification": classification,
        "special_failure": False,
        "primary_condition": PRIMARY_CONDITION,
        "recoveries": recoveries,
        "primary_action_success_absolute_gap_vs_current": action_gap,
        "primary_world_minus_current_paired_ci_low": ci_low,
        "primary_world_minus_current_paired_ci_high": ci_high,
        "credible_primary_world_degradation": credible_world_degradation,
        "strong_rule_satisfied": strong,
        "moderate_rule_satisfied": moderate,
        "explicit_weak_world_near_current": explicit_weak_near_current,
        "broad_action_world_tracking": broad_tracking,
        "conservative_weak_fallback": (
            classification == "WEAK"
            and not explicit_weak_near_current
            and not broad_tracking
        ),
        "thresholds": {
            "strong_action_current_absolute_gap_max": STRONG_ACTION_CURRENT_GAP_MAX,
            "strong_action_recovery_min": STRONG_ACTION_RECOVERY_MIN,
            "strong_world_recovery_max": STRONG_WORLD_RECOVERY_MAX,
            "moderate_action_current_absolute_gap_max": MODERATE_ACTION_CURRENT_GAP_MAX,
            "moderate_world_recovery_max": MODERATE_WORLD_RECOVERY_MAX,
            "weak_world_recovery_min": WEAK_WORLD_RECOVERY_MIN,
            "broad_tracking_gap_max": BROAD_TRACKING_GAP_MAX,
            "credible_world_degradation": (
                "paired CI for L(primary)-L(current) has lower endpoint > 0"
            ),
        },
        "recoveries_clipped": False,
        "stop_unconditionally": True,
    }
