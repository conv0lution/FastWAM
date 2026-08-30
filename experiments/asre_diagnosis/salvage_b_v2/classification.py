"""Registered v2 technical and scientific decision rules."""

from __future__ import annotations

import math
from typing import Any, Mapping

from experiments.asre_diagnosis.salvage_b.statistics import (
    action_recovery,
    functional_dissociation,
    world_recovery,
)

from .definitions import (
    CONDITIONS,
    MODERATE_ACTION_CURRENT_GAP_MAX,
    MODERATE_WORLD_RECOVERY_MAX,
    STRONG_ACTION_CURRENT_GAP_MAX,
    STRONG_ACTION_RECOVERY_MIN,
    STRONG_WORLD_RECOVERY_MAX,
)


def classify(
    *,
    machinery: Mapping[str, Any],
    action_success: Mapping[str, float],
    world_loss: Mapping[str, float],
    action_endpoint_valid: bool,
    world_endpoint_informative: bool,
    r170_world_minus_current_ci: tuple[float, float],
) -> dict[str, Any]:
    if machinery.get("native_clamp_identity_passed") is not True:
        return {"classification": "NATIVE-CLAMP-IDENTITY-FAILED", "stop": True}
    if machinery.get("shared_node_reach_passed") is not True:
        return {"classification": "SHARED-NODE-REACH-FAILED", "stop": True}
    if not action_endpoint_valid:
        return {"classification": "ENDPOINT-INVALID", "stop": True}
    if not world_endpoint_informative:
        return {"classification": "WORLD-ENDPOINT-UNINFORMATIVE", "stop": True}
    if set(action_success) != set(CONDITIONS) or set(world_loss) != set(CONDITIONS):
        raise ValueError("Scientific classification requires exactly four new v2 conditions.")
    if any(not math.isfinite(float(v)) for v in (*action_success.values(), *world_loss.values())):
        raise ValueError("Scientific classification inputs must be finite.")
    recoveries: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        ar = action_recovery(
            action_success[condition],
            current_success=action_success["current"],
            wrong_success=action_success["wrong"],
        )
        wr = world_recovery(
            world_loss[condition],
            current_loss=world_loss["current"],
            wrong_loss=world_loss["wrong"],
        )
        recoveries[condition] = {
            "action_recovery": ar,
            "world_recovery": wr,
            "functional_dissociation": functional_dissociation(
                action_recovery_value=ar,
                world_recovery_value=wr,
            ),
        }
    primary = recoveries["svd_r170"]
    action_gap = abs(action_success["svd_r170"] - action_success["current"])
    credible_world_degradation = float(r170_world_minus_current_ci[0]) > 0.0
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
    return {
        "classification": "STRONG" if strong else "MODERATE" if moderate else "WEAK",
        "stop": True,
        "recoveries": recoveries,
        "primary_action_absolute_gap_vs_current": action_gap,
        "credible_primary_world_degradation": credible_world_degradation,
        "r170_world_minus_current_paired_ci": list(r170_world_minus_current_ci),
        "strong_rule_satisfied": strong,
        "moderate_rule_satisfied": moderate,
        "recoveries_clipped": False,
    }
