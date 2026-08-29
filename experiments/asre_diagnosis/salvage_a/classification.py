"""Pre-registered held-out decision rule for the compact-ASRE salvage test."""

from __future__ import annotations

import math
from typing import Any, Mapping

from experiments.asre_diagnosis.salvage_a.definitions import CONDITIONS, PRIMARY_SCOPE


R36_CURRENT_ABSOLUTE_GAP_MAX = 0.10
R36_SVD_IMPROVEMENT_MIN = 0.30
R97_CURRENT_ABSOLUTE_GAP_MAX = 0.05
R97_SVD_IMPROVEMENT_MIN = 0.10
WEAK_R97_SVD_IMPROVEMENT_MAX = 0.05


def classify_salvage(
    *, success_rates: Mapping[str, float], analysis_scope: str
) -> dict[str, Any]:
    """Classify using held-out point estimates and no descriptive outcomes."""

    if analysis_scope != PRIMARY_SCOPE:
        raise ValueError("Salvage A classification is defined only on held-out episodes.")
    if set(success_rates) != set(CONDITIONS):
        raise ValueError("Salvage A classification requires exactly eight conditions.")
    rates = {condition: float(success_rates[condition]) for condition in CONDITIONS}
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in rates.values()):
        raise ValueError("Salvage A success rates must be finite probabilities.")

    current = rates["current_all"]
    aa36 = rates["actionaware_r36"]
    svd36 = rates["svd_r36"]
    aa97 = rates["actionaware_r97"]
    svd97 = rates["svd_r97"]
    current_gap = {36: current - aa36, 97: current - aa97}
    absolute_current_gap = {36: abs(current - aa36), 97: abs(current - aa97)}
    aa_minus_svd = {36: aa36 - svd36, 97: aa97 - svd97}

    strong_a = (
        absolute_current_gap[36] <= R36_CURRENT_ABSOLUTE_GAP_MAX
        and aa_minus_svd[36] >= R36_SVD_IMPROVEMENT_MIN
    )
    strong_b = (
        absolute_current_gap[97] <= R97_CURRENT_ABSOLUTE_GAP_MAX
        and aa_minus_svd[97] >= R97_SVD_IMPROVEMENT_MIN
    )
    moderate_r36 = (
        aa_minus_svd[36] >= R36_SVD_IMPROVEMENT_MIN
        and current_gap[36] > R36_CURRENT_ABSOLUTE_GAP_MAX
    )
    moderate_r97 = (
        aa_minus_svd[97] >= R97_SVD_IMPROVEMENT_MIN
        and current_gap[97] > R97_CURRENT_ABSOLUTE_GAP_MAX
    )
    moderate = not (strong_a or strong_b) and (moderate_r36 or moderate_r97)
    explicit_weak = (
        current_gap[36] > R36_CURRENT_ABSOLUTE_GAP_MAX
        and (
            aa_minus_svd[97] <= WEAK_R97_SVD_IMPROVEMENT_MAX
            or current_gap[97] > R97_CURRENT_ABSOLUTE_GAP_MAX
        )
    )
    classification = "STRONG" if strong_a or strong_b else "MODERATE" if moderate else "WEAK"
    compact_status = {
        "STRONG": "revived",
        "MODERATE": "partially_supported",
        "WEAK": "closed",
    }[classification]
    return {
        "classification": classification,
        "compact_asre_status": compact_status,
        "classification_input_scope": PRIMARY_SCOPE,
        "strong_rule_a_r36": strong_a,
        "strong_rule_b_r97": strong_b,
        "moderate_rule_r36": moderate_r36 and not (strong_a or strong_b),
        "moderate_rule_r97": moderate_r97 and not (strong_a or strong_b),
        "explicit_weak_close_rule": explicit_weak,
        "conservative_weak_fallback": classification == "WEAK" and not explicit_weak,
        "current_minus_actionaware": {
            "36": current_gap[36],
            "97": current_gap[97],
        },
        "absolute_current_gap": {
            "36": absolute_current_gap[36],
            "97": absolute_current_gap[97],
        },
        "actionaware_minus_svd": {
            "36": aa_minus_svd[36],
            "97": aa_minus_svd[97],
        },
        "thresholds": {
            "r36_current_absolute_gap_max": R36_CURRENT_ABSOLUTE_GAP_MAX,
            "r36_svd_improvement_min": R36_SVD_IMPROVEMENT_MIN,
            "r97_current_absolute_gap_max": R97_CURRENT_ABSOLUTE_GAP_MAX,
            "r97_svd_improvement_min": R97_SVD_IMPROVEMENT_MIN,
            "weak_r97_svd_improvement_max": WEAK_R97_SVD_IMPROVEMENT_MAX,
        },
        "uncertainty_used_as_decision_gate": False,
        "random_control_used_as_decision_gate": False,
        "task_level_veto_used": False,
    }
