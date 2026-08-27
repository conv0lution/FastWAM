"""Paired outcome statistics and the pre-specified Round-3B decision gate."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np


EpisodeKey = tuple[str, int, int]

CONDITION_ORDER = (
    "late_current_correct",
    "late_wrong_scene",
    "late_no_video",
)

COMPARISON_ORDER = (
    "wrong_minus_correct",
    "no_video_minus_correct",
    "wrong_minus_no_video",
)

COMPARISON_DEFINITIONS = {
    "wrong_minus_correct": ("late_current_correct", "late_wrong_scene"),
    "no_video_minus_correct": ("late_current_correct", "late_no_video"),
    "wrong_minus_no_video": ("late_no_video", "late_wrong_scene"),
}


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative, got {result}.")
    return result


def _stable_seed(base_seed: int, *parts: object) -> int:
    label = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _percentile_interval(estimates: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval after jointly resampling paired episode differences."""

    samples = _positive_integer(samples, name="samples")
    seed = _nonnegative_integer(seed, name="seed")
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Paired bootstrap requires at least one difference.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Paired bootstrap differences must be finite.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    return _percentile_interval(values[indices].mean(axis=1))


def task_hierarchical_bootstrap_ci(
    keys: Sequence[EpisodeKey],
    differences: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Resample tasks and then paired trials within the sampled task."""

    samples = _positive_integer(samples, name="samples")
    seed = _nonnegative_integer(seed, name="seed")
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if len(keys) != values.size or values.size == 0:
        raise ValueError("Task bootstrap requires one value per nonempty episode key.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Task bootstrap differences must be finite.")

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(f"Invalid episode key at index {index}: {key!r}.")
        suite, task_id, _trial_id = key
        grouped[(str(suite), int(task_id))].append(index)
    group_indices = [np.asarray(grouped[key], dtype=int) for key in sorted(grouped)]
    if not group_indices:
        raise ValueError("Task bootstrap found no task groups.")

    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for bootstrap_index in range(samples):
        sampled_groups = rng.integers(0, len(group_indices), size=len(group_indices))
        sampled_task_means = np.empty(len(group_indices), dtype=np.float64)
        for output_index, group_index in enumerate(sampled_groups):
            members = group_indices[int(group_index)]
            selected = members[rng.integers(0, members.size, size=members.size)]
            sampled_task_means[output_index] = float(values[selected].mean())
        estimates[bootstrap_index] = float(sampled_task_means.mean())
    return _percentile_interval(estimates)


def exact_mcnemar_p_value(induced_failures: int, rescued_successes: int) -> float:
    """Two-sided exact paired McNemar/binomial p-value."""

    induced = _nonnegative_integer(induced_failures, name="induced_failures")
    rescued = _nonnegative_integer(rescued_successes, name="rescued_successes")
    discordant = induced + rescued
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(induced, rescued) + 1)
    )
    return float(min(1.0, 2.0 * tail / (2**discordant)))


def _normalize_outcomes(
    outcomes: Mapping[str, Mapping[EpisodeKey, int]],
) -> tuple[list[EpisodeKey], dict[str, np.ndarray]]:
    if set(outcomes) != set(CONDITION_ORDER):
        raise ValueError(
            "Round-3B outcomes must contain exactly "
            f"{list(CONDITION_ORDER)}, got {sorted(outcomes)}."
        )
    key_sets = {condition: set(outcomes[condition]) for condition in CONDITION_ORDER}
    reference_keys = key_sets[CONDITION_ORDER[0]]
    if not reference_keys:
        raise ValueError("Round-3B outcomes contain no paired episodes.")
    mismatches = {
        condition: {
            "missing": sorted(reference_keys - keys),
            "extra": sorted(keys - reference_keys),
        }
        for condition, keys in key_sets.items()
        if keys != reference_keys
    }
    if mismatches:
        raise ValueError(f"Round-3B paired episode keys differ: {mismatches}.")
    keys = sorted(reference_keys)
    matrices: dict[str, np.ndarray] = {}
    for condition in CONDITION_ORDER:
        values = np.asarray([outcomes[condition][key] for key in keys])
        if values.dtype == np.bool_:
            values = values.astype(np.int8)
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError(f"{condition} outcomes must be binary zero/one values.")
        matrices[condition] = values.astype(np.int8)
    return keys, matrices


def comparison_statistics(
    keys: Sequence[EpisodeKey],
    reference_values: np.ndarray,
    target_values: np.ndarray,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    seed_label: str,
) -> dict[str, Any]:
    reference = np.asarray(reference_values, dtype=np.int8).reshape(-1)
    target = np.asarray(target_values, dtype=np.int8).reshape(-1)
    if reference.shape != target.shape or reference.size != len(keys):
        raise ValueError("Comparison arrays must align one-to-one with episode keys.")
    if not np.all(np.isin(reference, [0, 1])) or not np.all(np.isin(target, [0, 1])):
        raise ValueError("Comparison outcomes must be binary.")
    differences = target.astype(np.float64) - reference.astype(np.float64)
    paired_low, paired_high = paired_bootstrap_ci(
        differences,
        samples=bootstrap_samples,
        seed=_stable_seed(bootstrap_seed, seed_label, "paired"),
    )
    task_low, task_high = task_hierarchical_bootstrap_ci(
        keys,
        differences,
        samples=bootstrap_samples,
        seed=_stable_seed(bootstrap_seed, seed_label, "task_hierarchical"),
    )
    transitions = Counter(zip(reference.astype(int), target.astype(int)))
    induced = int(transitions[(1, 0)])
    rescued = int(transitions[(0, 1)])
    return {
        "reference_success_rate": float(reference.mean()),
        "target_success_rate": float(target.mean()),
        "delta_success_rate": float(differences.mean()),
        "paired_ci_low": paired_low,
        "paired_ci_high": paired_high,
        "task_hierarchical_ci_low": task_low,
        "task_hierarchical_ci_high": task_high,
        "reference_success_to_target_success": int(transitions[(1, 1)]),
        "reference_success_to_target_failure": induced,
        "reference_failure_to_target_success": rescued,
        "reference_failure_to_target_failure": int(transitions[(0, 0)]),
        "induced_failures": induced,
        "rescued_successes": rescued,
        "discordant_pairs": induced + rescued,
        "mcnemar_exact_p_value": exact_mcnemar_p_value(induced, rescued),
        "paired_episodes": int(reference.size),
    }


def classify_decision(
    *,
    correct_success_rate: float,
    wrong_success_rate: float,
    no_video_success_rate: float,
    wrong_minus_correct_task_ci: tuple[float, float],
) -> dict[str, Any]:
    """Apply the frozen screening rule without inspecting any other experiment."""

    rates = np.asarray(
        [correct_success_rate, wrong_success_rate, no_video_success_rate],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(rates)) or np.any((rates < 0.0) | (rates > 1.0)):
        raise ValueError(f"Success rates must be finite probabilities, got {rates.tolist()}.")
    ci_low, ci_high = (float(value) for value in wrong_minus_correct_task_ci)
    if not math.isfinite(ci_low) or not math.isfinite(ci_high) or ci_low > ci_high:
        raise ValueError("wrong-minus-correct task interval is invalid.")

    content_loss = float(correct_success_rate - wrong_success_rate)
    correct_to_no_video_loss = float(correct_success_rate - no_video_success_rate)
    # The supplied interval is target-reference (wrong-correct). Negating and
    # reversing its endpoints yields the correct->wrong loss interval.
    task_loss_ci = (-ci_high, -ci_low)

    if content_loss >= 0.20 and task_loss_ci[0] > 0.05:
        classification = "GO"
        label = "strong_content_support"
        rationale = (
            "The observed correct-to-wrong loss is at least 20 percentage points, "
            "and the task-hierarchical interval excludes losses of 5 points or less."
        )
    elif content_loss <= 0.05 and correct_to_no_video_loss >= 0.20:
        classification = "STOP"
        label = "structural_confound_warning"
        rationale = (
            "Wrong-scene K/V remains within 5 percentage points of correct K/V while "
            "the same-run no-video anchor loses at least 20 points."
        )
    else:
        classification = "VERIFY"
        label = "ambiguous"
        rationale = (
            "The result does not satisfy both strong-content criteria and is not the "
            "pre-specified near-correct wrong-scene structural-warning pattern."
        )
    return {
        "classification": classification,
        "label": label,
        "correct_to_wrong_loss": content_loss,
        "correct_to_wrong_task_ci_low": float(task_loss_ci[0]),
        "correct_to_wrong_task_ci_high": float(task_loss_ci[1]),
        "correct_to_no_video_loss": correct_to_no_video_loss,
        "thresholds": {
            "go_observed_loss_minimum": 0.20,
            "go_task_ci_lower_bound_strictly_above": 0.05,
            "stop_wrong_near_correct_maximum_loss": 0.05,
            "stop_correct_to_no_video_minimum_loss": 0.20,
        },
        "rationale": rationale,
    }


def analyze_outcomes(
    outcomes: Mapping[str, Mapping[EpisodeKey, int]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Analyze all three paired conditions and apply the decision gate."""

    bootstrap_samples = _positive_integer(bootstrap_samples, name="bootstrap_samples")
    bootstrap_seed = _nonnegative_integer(bootstrap_seed, name="bootstrap_seed")
    keys, values = _normalize_outcomes(outcomes)
    comparisons: dict[str, dict[str, Any]] = {}
    for name in COMPARISON_ORDER:
        reference_name, target_name = COMPARISON_DEFINITIONS[name]
        comparisons[name] = {
            "comparison": name,
            "reference_condition": reference_name,
            "target_condition": target_name,
            **comparison_statistics(
                keys,
                values[reference_name],
                values[target_name],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                seed_label=name,
            ),
        }

    rates = {condition: float(values[condition].mean()) for condition in CONDITION_ORDER}
    denominator = rates["late_current_correct"] - rates["late_no_video"]
    content_gap_fraction = (
        None
        if math.isclose(denominator, 0.0, abs_tol=1e-15)
        else float(
            (rates["late_current_correct"] - rates["late_wrong_scene"])
            / denominator
        )
    )
    primary = comparisons["wrong_minus_correct"]
    decision = classify_decision(
        correct_success_rate=rates["late_current_correct"],
        wrong_success_rate=rates["late_wrong_scene"],
        no_video_success_rate=rates["late_no_video"],
        wrong_minus_correct_task_ci=(
            primary["task_hierarchical_ci_low"],
            primary["task_hierarchical_ci_high"],
        ),
    )
    return {
        "condition_order": list(CONDITION_ORDER),
        "paired_episode_count": len(keys),
        "success_rates": rates,
        "comparisons": comparisons,
        "content_gap_fraction": content_gap_fraction,
        "content_gap_fraction_interpretation": (
            "Descriptive normalization only; not a mediation fraction or variance explained."
        ),
        "decision": decision,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "paired_unit": "matched task/trial episode",
            "task_hierarchical_unit": "resample tasks, then matched trials within task",
            "interval_type": "pointwise percentile 95% CI",
        },
    }
