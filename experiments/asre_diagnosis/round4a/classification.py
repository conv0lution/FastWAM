from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


STRONG_MEDIAN_GAP_MAX = 0.05
STRONG_MASK_GAP_MAX = 0.10
CATASTROPHIC_TASK_LOSS_MIN = 0.50
WEAK_RECOVERY_MAX = 0.20
WRONG_NEAR_MAX = 0.05


def classify_axis(
    *,
    axis: str,
    mask_success_rates: Sequence[float],
    current_success_rate: float,
    wrong_success_rate: float,
    mask_task_success: Sequence[Mapping[int, float]],
    current_task_success: Mapping[int, float],
) -> dict[str, Any]:
    axis = str(axis).lower()
    if axis not in {"token", "head"}:
        raise ValueError(f"Axis must be token or head, got {axis!r}.")
    rates = np.asarray(mask_success_rates, dtype=np.float64)
    if rates.shape != (3,) or not np.all(np.isfinite(rates)):
        raise ValueError("Axis classification requires exactly three finite mask rates.")
    if len(mask_task_success) != 3:
        raise ValueError("Axis classification requires three task-success mappings.")
    if not np.all((0.0 <= rates) & (rates <= 1.0)):
        raise ValueError("Mask success rates must be probabilities.")
    gap = float(current_success_rate - wrong_success_rate)
    recoveries = (
        np.full(3, np.nan)
        if abs(gap) <= 1e-15
        else (rates - float(wrong_success_rate)) / gap
    )
    catastrophic_events = []
    per_task_count: Counter[int] = Counter()
    expected_tasks = set(int(task) for task in current_task_success)
    for mask_index, task_rates in enumerate(mask_task_success, start=1):
        if set(int(task) for task in task_rates) != expected_tasks:
            raise ValueError("Axis task-success mappings do not align with current tasks.")
        for task_id in sorted(expected_tasks):
            loss = float(current_task_success[task_id] - task_rates[task_id])
            if loss >= CATASTROPHIC_TASK_LOSS_MIN:
                catastrophic_events.append(
                    {
                        "mask_index": mask_index,
                        "task_id": task_id,
                        "loss": loss,
                    }
                )
                per_task_count[task_id] += 1
    # "Repeated across masks" means the same task collapses for at least two
    # independently frozen masks. A single broadly bad mask is still reported,
    # but does not by itself satisfy the registered repeated-collapse criterion.
    repeated_catastrophic = any(count >= 2 for count in per_task_count.values())
    median_rate = float(np.median(rates))
    losses = float(current_success_rate) - rates
    strong = bool(
        median_rate >= float(current_success_rate) - STRONG_MEDIAN_GAP_MAX
        and float(losses.max()) <= STRONG_MASK_GAP_MAX
        and not repeated_catastrophic
    )
    near_wrong_count = int(np.sum(rates <= float(wrong_success_rate) + WRONG_NEAR_MAX))
    median_recovery = (
        None if np.all(np.isnan(recoveries)) else float(np.nanmedian(recoveries))
    )
    weak = bool(
        not strong
        and (
            (median_recovery is not None and median_recovery <= WEAK_RECOVERY_MAX)
            or near_wrong_count >= 2
            or (len(catastrophic_events) >= 6 and float(np.ptp(rates)) >= 0.20)
        )
    )
    classification = "STRONG" if strong else "WEAK" if weak else "INTERMEDIATE"
    return {
        "axis": axis,
        "classification": classification,
        "mask_success_rates": rates.astype(float).tolist(),
        "mean_success_rate": float(np.mean(rates)),
        "median_success_rate": median_rate,
        "min_success_rate": float(np.min(rates)),
        "max_success_rate": float(np.max(rates)),
        "range_success_rate": float(np.ptp(rates)),
        "variance_success_rate": float(np.var(rates)),
        "current_success_rate": float(current_success_rate),
        "wrong_success_rate": float(wrong_success_rate),
        "current_wrong_gap": gap,
        "retained_behavior_by_mask": (rates - float(wrong_success_rate)).astype(float).tolist(),
        "normalized_recovery_by_mask": [
            None if not np.isfinite(value) else float(value) for value in recoveries
        ],
        "median_normalized_recovery": median_recovery,
        "near_wrong_mask_count": near_wrong_count,
        "catastrophic_task_events": catastrophic_events,
        "repeated_catastrophic_task_collapse": repeated_catastrophic,
        "thresholds": {
            "strong_median_current_gap_max": STRONG_MEDIAN_GAP_MAX,
            "strong_individual_mask_current_gap_max": STRONG_MASK_GAP_MAX,
            "catastrophic_task_loss_min": CATASTROPHIC_TASK_LOSS_MIN,
            "repeated_catastrophic_definition": (
                "the same task is catastrophic in >=2 independently frozen masks"
            ),
            "weak_median_normalized_recovery_max": WEAK_RECOVERY_MAX,
            "wrong_near_gap_max": WRONG_NEAR_MAX,
        },
    }


def recommend_axis(
    token: Mapping[str, Any], head: Mapping[str, Any]
) -> dict[str, Any]:
    analyses = {"token": token, "head": head}
    strong = [axis for axis, result in analyses.items() if result["classification"] == "STRONG"]
    intermediate = [
        axis
        for axis, result in analyses.items()
        if result["classification"] == "INTERMEDIATE"
    ]
    if len(strong) == 1:
        axis = strong[0]
        return {
            "recommended_primary_axis": axis,
            "recommended_next_experiment": (
                f"Cross-suite replication of the identical frozen {axis}-50 rule"
            ),
            "next_step_category": "cross-suite 50% replication",
            "rationale": "Only one axis met every strong robustness criterion.",
            "auto_launch": False,
        }
    if len(strong) == 2:
        # Median behavior dominates, then robustness range, catastrophic-event count,
        # and finally token's simpler global positional interpretation.
        rank = sorted(
            strong,
            key=lambda axis: (
                -float(analyses[axis]["median_success_rate"]),
                float(analyses[axis]["range_success_rate"]),
                len(analyses[axis]["catastrophic_task_events"]),
                0 if axis == "token" else 1,
            ),
        )
        return {
            "recommended_primary_axis": rank[0],
            "recommended_next_experiment": (
                f"Cross-suite replication of the identical frozen {rank[0]}-50 rule"
            ),
            "next_step_category": "cross-suite 50% replication",
            "rationale": (
                "Both axes were strong; selection used median success, mask range, "
                "catastrophic task effects, then interpretability."
            ),
            "auto_launch": False,
        }
    if intermediate:
        axis = sorted(
            intermediate,
            key=lambda name: (
                -float(analyses[name]["median_success_rate"]),
                float(analyses[name]["range_success_rate"]),
            ),
        )[0]
        return {
            "recommended_primary_axis": axis,
            "recommended_next_experiment": (
                f"A pre-registered {axis}-75 diagnostic/control before any cross-suite claim"
            ),
            "next_step_category": "75% diagnostic",
            "rationale": "No axis was strong, but at least one showed intermediate recovery.",
            "auto_launch": False,
        }
    return {
        "recommended_primary_axis": None,
        "recommended_next_experiment": (
            "Reconsider the intervention subspace; do not reduce token/head budgets"
        ),
        "next_step_category": "subspace reconsideration",
        "rationale": "Both 50% axes were weak.",
        "auto_launch": False,
    }
