"""Frozen deterministic Round-4B Stage-2 decision rule."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


CURRENT_GAP_STRONG_MAX = 0.05
SVD_RANDOM_STRONG_MIN = 0.20
MODERATE_CURRENT_GAP_MIN_EXCLUSIVE = 0.05
MODERATE_CURRENT_GAP_MAX = 0.10
SUBSTANTIAL_ADVANTAGE_MIN = 0.10
GENERIC_CURRENT_GAP_MAX = 0.05
GENERIC_MATCHED_GAP_MAX = 0.05
SEVERE_DEGRADATION_MIN = 0.20
MEANINGFUL_ADVANTAGE_MIN = 0.10
CATASTROPHIC_TASK_LOSS_MIN = 0.50


def classify_subspace(
    *,
    success_rates: Mapping[str, float],
    task_success: Mapping[str, Mapping[int, float]],
) -> dict[str, Any]:
    required = {
        "current_all",
        "wrong_all",
        "svd_r256",
        "random_r256",
        "svd_r768",
        "random_r768",
        "svd_r1536",
        "random_r1536",
    }
    if set(success_rates) != required or set(task_success) != required:
        raise ValueError("Round-4B classification requires exactly eight conditions.")
    current = float(success_rates["current_all"])
    losses = {
        rank: current - float(success_rates[f"svd_r{rank}"])
        for rank in (256, 768, 1536)
    }
    advantages = {
        rank: float(success_rates[f"svd_r{rank}"])
        - float(success_rates[f"random_r{rank}"])
        for rank in (256, 768, 1536)
    }
    catastrophic = []
    repeated: Counter[int] = Counter()
    current_tasks = task_success["current_all"]
    for rank in (256, 768, 1536):
        for task_id, rate in task_success[f"svd_r{rank}"].items():
            loss = float(current_tasks[int(task_id)] - rate)
            if loss >= CATASTROPHIC_TASK_LOSS_MIN:
                catastrophic.append({"rank": rank, "task_id": int(task_id), "loss": loss})
                repeated[int(task_id)] += 1
    repeated_collapse = any(count >= 2 for count in repeated.values())
    strong = (
        abs(losses[768]) <= CURRENT_GAP_STRONG_MAX
        and advantages[768] >= SVD_RANDOM_STRONG_MIN
        and not repeated_collapse
    )
    moderate = (
        not strong
        and MODERATE_CURRENT_GAP_MIN_EXCLUSIVE < losses[1536]
        <= MODERATE_CURRENT_GAP_MAX
        and advantages[1536] >= SUBSTANTIAL_ADVANTAGE_MIN
        and (
            abs(losses[768]) > CURRENT_GAP_STRONG_MAX
            or advantages[768] < SVD_RANDOM_STRONG_MIN
        )
    )
    generic_ranks = [
        rank
        for rank in (256, 768, 1536)
        if abs(current - float(success_rates[f"svd_r{rank}"]))
        <= GENERIC_CURRENT_GAP_MAX
        and abs(current - float(success_rates[f"random_r{rank}"]))
        <= GENERIC_CURRENT_GAP_MAX
        and abs(advantages[rank]) <= GENERIC_MATCHED_GAP_MAX
    ]
    generic = not strong and not moderate and bool(generic_ranks)
    severe = losses[1536] >= SEVERE_DEGRADATION_MIN
    no_advantage = max(advantages.values()) < MEANINGFUL_ADVANTAGE_MIN
    lower_not_current = losses[256] > MODERATE_CURRENT_GAP_MAX and losses[768] > MODERATE_CURRENT_GAP_MAX
    weak_rule = severe or (no_advantage and lower_not_current)
    if strong:
        classification = "STRONG"
    elif moderate:
        classification = "MODERATE"
    elif generic:
        classification = "GENERIC"
    else:
        # Conservatively stop when no positive registered gate is met.  The
        # explicit weak-rule components remain separately visible below.
        classification = "WEAK"
    return {
        "classification": classification,
        "svd_current_loss": {str(key): value for key, value in losses.items()},
        "svd_minus_random": {str(key): value for key, value in advantages.items()},
        "catastrophic_task_events": catastrophic,
        "repeated_catastrophic_task_collapse": repeated_collapse,
        "generic_preservation_ranks": generic_ranks,
        "explicit_weak_rule_met": weak_rule,
        "conservative_weak_fallback": classification == "WEAK" and not weak_rule,
        "thresholds": {
            "strong_svd768_current_absolute_gap_max": CURRENT_GAP_STRONG_MAX,
            "strong_svd768_minus_random768_min": SVD_RANDOM_STRONG_MIN,
            "moderate_svd1536_current_loss_range": [
                MODERATE_CURRENT_GAP_MIN_EXCLUSIVE,
                MODERATE_CURRENT_GAP_MAX,
            ],
            "substantial_svd_random_advantage_min": SUBSTANTIAL_ADVANTAGE_MIN,
            "generic_both_current_gap_max": GENERIC_CURRENT_GAP_MAX,
            "generic_svd_random_absolute_gap_max": GENERIC_MATCHED_GAP_MAX,
            "severe_svd1536_degradation_min": SEVERE_DEGRADATION_MIN,
            "meaningful_svd_random_advantage_min": MEANINGFUL_ADVANTAGE_MIN,
            "catastrophic_task_loss_min": CATASTROPHIC_TASK_LOSS_MIN,
            "repeated_catastrophic_definition": "same task loses >=50pp at >=2 SVD ranks",
        },
    }
