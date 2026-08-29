"""Pre-specified Round-4C energy-controlled action-sufficiency decision rule."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from experiments.asre_diagnosis.round4c.definitions import CONDITIONS, RANKS


R36_CURRENT_ABSOLUTE_GAP_MAX = 0.10
R97_CURRENT_ABSOLUTE_GAP_MAX = 0.05
R170_MODERATE_LOSS_MIN = 0.05
R170_MODERATE_LOSS_MAX = 0.10
CATASTROPHIC_TASK_LOSS_MIN = 0.50
NEAR_WRONG_ABSOLUTE_GAP_MAX = 0.10
SEVERE_CURRENT_LOSS_MIN = 0.10


def classify_action_sufficiency(
    *,
    success_rates: Mapping[str, float],
    task_success: Mapping[str, Mapping[int, float]],
    round4b_r256_success: float,
) -> dict[str, Any]:
    if set(success_rates) != set(CONDITIONS) or set(task_success) != set(CONDITIONS):
        raise ValueError("Round-4C classification requires exactly five conditions.")
    values = {key: float(success_rates[key]) for key in CONDITIONS}
    if any(value < 0.0 or value > 1.0 for value in values.values()):
        raise ValueError("Round-4C success rates must be probabilities.")
    current = values["current_all"]
    wrong = values["wrong_all"]
    losses = {rank: current - values[f"svd_r{rank}"] for rank in RANKS}
    absolute_gaps = {rank: abs(losses[rank]) for rank in RANKS}
    catastrophic: list[dict[str, Any]] = []
    repeated: Counter[int] = Counter()
    for rank in RANKS:
        for task_id, current_rate in task_success["current_all"].items():
            loss = float(current_rate) - float(task_success[f"svd_r{rank}"][task_id])
            if loss >= CATASTROPHIC_TASK_LOSS_MIN:
                catastrophic.append({"rank": rank, "task_id": int(task_id), "loss": loss})
                repeated[int(task_id)] += 1
    repeated_collapse = any(count >= 2 for count in repeated.values())
    strong_a = absolute_gaps[36] <= R36_CURRENT_ABSOLUTE_GAP_MAX and not repeated_collapse
    strong_b = absolute_gaps[97] <= R97_CURRENT_ABSOLUTE_GAP_MAX and not repeated_collapse
    moderate = (
        not (strong_a or strong_b)
        and R170_MODERATE_LOSS_MIN <= losses[170] <= R170_MODERATE_LOSS_MAX
        and losses[97] > R97_CURRENT_ABSOLUTE_GAP_MAX
    )
    explicit_weak = (
        abs(values["svd_r36"] - wrong) <= NEAR_WRONG_ABSOLUTE_GAP_MAX
        and losses[97] >= SEVERE_CURRENT_LOSS_MIN
        and losses[170] > R170_MODERATE_LOSS_MAX
        and abs(current - float(round4b_r256_success)) <= R97_CURRENT_ABSOLUTE_GAP_MAX
    )
    classification = "STRONG" if strong_a or strong_b else "MODERATE" if moderate else "WEAK"
    return {
        "classification": classification,
        "strong_rule_a_r36_within_10pp": strong_a,
        "strong_rule_b_r97_within_5pp": strong_b,
        "moderate_rule_r170_5_to_10pp_and_r97_degraded": moderate,
        "explicit_weak_manifold_tracking_rule": explicit_weak,
        "conservative_weak_fallback": classification == "WEAK" and not explicit_weak,
        "current_minus_svd": {str(rank): losses[rank] for rank in RANKS},
        "absolute_current_gap": {str(rank): absolute_gaps[rank] for rank in RANKS},
        "catastrophic_task_events": catastrophic,
        "repeated_catastrophic_task_collapse": repeated_collapse,
        "thresholds": {
            "strong_r36_current_absolute_gap_max": R36_CURRENT_ABSOLUTE_GAP_MAX,
            "strong_r97_current_absolute_gap_max": R97_CURRENT_ABSOLUTE_GAP_MAX,
            "moderate_r170_current_loss_range": [
                R170_MODERATE_LOSS_MIN,
                R170_MODERATE_LOSS_MAX,
            ],
            "catastrophic_task_loss_min": CATASTROPHIC_TASK_LOSS_MIN,
            "repeated_catastrophic_definition": "same task loses >=50pp at >=2 ranks",
            "weak_r36_near_wrong_absolute_gap_max": NEAR_WRONG_ABSOLUTE_GAP_MAX,
            "weak_r97_severe_current_loss_min": SEVERE_CURRENT_LOSS_MIN,
        },
    }
