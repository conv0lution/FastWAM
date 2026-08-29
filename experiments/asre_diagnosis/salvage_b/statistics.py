"""Paired world-loss statistics for the final Salvage-B gate.

The public analysis function expects one native-loss value per evaluation
sample and condition.  If the native objective is evaluated with the four
frozen stochastic draws, callers must average those four draws *within each
sample* before calling :func:`analyze_world_losses`.  Every resample remains
paired across all four conditions.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.asre_diagnosis.salvage_b.definitions import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CONDITIONS,
)


# task, episode, evaluation-sample.  Object is intentional: manifests may use
# integer IDs or stable strings, while the hierarchy is the same.
WorldSampleKey = tuple[object, object, object]


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
    seed = _nonnegative_integer(base_seed, name="seed")
    label = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _percentile_interval(estimates: np.ndarray) -> tuple[float, float]:
    values = _finite_vector(estimates, name="bootstrap estimates")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _validate_keys(
    keys: Sequence[WorldSampleKey], *, expected_size: int
) -> list[WorldSampleKey]:
    if len(keys) != expected_size or expected_size == 0:
        raise ValueError("World keys must align one-to-one with a nonempty value vector.")
    normalized: list[WorldSampleKey] = []
    for index, key in enumerate(keys):
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(
                "Each world key must be a (task, episode, sample) tuple; "
                f"index {index} is {key!r}."
            )
        normalized.append(key)
    if len(set(normalized)) != len(normalized):
        raise ValueError("World sample keys must be unique.")
    return normalized


def paired_bootstrap_estimates(
    values: Sequence[float] | np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Return means from deterministic paired-sample resampling."""

    count = _positive_integer(samples, name="samples")
    seed = _nonnegative_integer(seed, name="seed")
    vector = _finite_vector(values, name="paired values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, vector.size, size=(count, vector.size))
    return vector[indices].mean(axis=1)


def paired_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile CI after resampling complete paired evaluation samples."""

    return _percentile_interval(
        paired_bootstrap_estimates(values, samples=samples, seed=seed)
    )


def _hierarchical_groups(
    keys: Sequence[WorldSampleKey], *, expected_size: int
) -> list[list[np.ndarray]]:
    normalized = _validate_keys(keys, expected_size=expected_size)
    grouped: dict[object, dict[object, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (task, episode, _sample) in enumerate(normalized):
        grouped[task][episode].append(index)

    # repr-based ordering supports mixed manifest identifier types and keeps
    # the fixed-seed result stable across Python dictionary insertion orders.
    result: list[list[np.ndarray]] = []
    for task in sorted(grouped, key=repr):
        episodes = grouped[task]
        result.append(
            [
                np.asarray(episodes[episode], dtype=np.int64)
                for episode in sorted(episodes, key=repr)
            ]
        )
    if not result:
        raise ValueError("Task-hierarchical bootstrap found no task groups.")
    return result


def task_hierarchical_bootstrap_indices(
    keys: Sequence[WorldSampleKey],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[np.ndarray]:
    """Resample task -> episode -> sample and return each replicate's indices.

    Tasks receive equal weight, episodes receive equal weight within a task,
    and samples receive equal weight within an episode.  To preserve this
    weighting for unbalanced hierarchies, callers should use
    :func:`task_hierarchical_bootstrap_estimates` rather than averaging these
    raw index arrays directly.  The indices are exposed for reproducibility
    audits and balanced test fixtures.
    """

    count = _positive_integer(samples, name="samples")
    seed = _nonnegative_integer(seed, name="seed")
    groups = _hierarchical_groups(keys, expected_size=len(keys))
    rng = np.random.default_rng(seed)
    replicates: list[np.ndarray] = []
    for _ in range(count):
        selected: list[int] = []
        for task_index in rng.integers(0, len(groups), size=len(groups)):
            episodes = groups[int(task_index)]
            for episode_index in rng.integers(
                0, len(episodes), size=len(episodes)
            ):
                members = episodes[int(episode_index)]
                selected.extend(
                    members[
                        rng.integers(0, members.size, size=members.size)
                    ].tolist()
                )
        replicates.append(np.asarray(selected, dtype=np.int64))
    return replicates


def task_hierarchical_bootstrap_estimates(
    keys: Sequence[WorldSampleKey],
    values: Sequence[float] | np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Bootstrap a mean under task -> episode -> sample clustering."""

    count = _positive_integer(samples, name="samples")
    seed = _nonnegative_integer(seed, name="seed")
    vector = _finite_vector(values, name="hierarchical values")
    groups = _hierarchical_groups(keys, expected_size=vector.size)
    rng = np.random.default_rng(seed)
    estimates = np.empty(count, dtype=np.float64)
    for replicate in range(count):
        task_means = np.empty(len(groups), dtype=np.float64)
        for task_output, task_index in enumerate(
            rng.integers(0, len(groups), size=len(groups))
        ):
            episodes = groups[int(task_index)]
            episode_means = np.empty(len(episodes), dtype=np.float64)
            for episode_output, episode_index in enumerate(
                rng.integers(0, len(episodes), size=len(episodes))
            ):
                members = episodes[int(episode_index)]
                selected = members[
                    rng.integers(0, members.size, size=members.size)
                ]
                episode_means[episode_output] = float(vector[selected].mean())
            task_means[task_output] = float(episode_means.mean())
        estimates[replicate] = float(task_means.mean())
    return estimates


def task_hierarchical_bootstrap_ci(
    keys: Sequence[WorldSampleKey],
    values: Sequence[float] | np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile CI under task -> episode -> evaluation-sample resampling."""

    return _percentile_interval(
        task_hierarchical_bootstrap_estimates(
            keys, values, samples=samples, seed=seed
        )
    )


def _safe_denominator(numerator_scale: float, denominator: float, *, name: str) -> float:
    scale = max(1.0, abs(float(numerator_scale)))
    if not math.isfinite(denominator) or abs(denominator) <= np.finfo(np.float64).eps * scale:
        raise ValueError(f"{name} denominator is zero or numerically degenerate.")
    return float(denominator)


def action_recovery(
    condition_success: float,
    *,
    current_success: float,
    wrong_success: float,
) -> float:
    """Unclipped normalized action recovery."""

    values = np.asarray(
        [condition_success, current_success, wrong_success], dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Action-recovery inputs must be finite.")
    denominator = _safe_denominator(
        max(abs(current_success), abs(wrong_success)),
        float(current_success) - float(wrong_success),
        name="ActionRecovery",
    )
    return float((float(condition_success) - float(wrong_success)) / denominator)


def world_recovery(
    condition_loss: float,
    *,
    current_loss: float,
    wrong_loss: float,
) -> float:
    """Unclipped normalized recovery for a lower-is-better world loss."""

    values = np.asarray([condition_loss, current_loss, wrong_loss], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("World-recovery inputs must be finite.")
    denominator = _safe_denominator(
        max(abs(current_loss), abs(wrong_loss)),
        float(wrong_loss) - float(current_loss),
        name="WorldRecovery",
    )
    return float(
        1.0 - (float(condition_loss) - float(current_loss)) / denominator
    )


def functional_dissociation(
    *, action_recovery_value: float, world_recovery_value: float
) -> float:
    """Return the unclipped descriptive action-minus-world recovery gap."""

    values = np.asarray(
        [action_recovery_value, world_recovery_value], dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Functional-dissociation inputs must be finite.")
    return float(action_recovery_value) - float(world_recovery_value)


def _joint_paired_mean_estimates(
    matrix: np.ndarray, *, samples: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, matrix.shape[1], size=(samples, matrix.shape[1]))
    return matrix[:, indices].mean(axis=2).T


def _joint_hierarchical_mean_estimates(
    keys: Sequence[WorldSampleKey],
    matrix: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    groups = _hierarchical_groups(keys, expected_size=matrix.shape[1])
    rng = np.random.default_rng(seed)
    estimates = np.empty((samples, matrix.shape[0]), dtype=np.float64)
    for replicate in range(samples):
        task_means = np.empty(
            (len(groups), matrix.shape[0]), dtype=np.float64
        )
        for task_output, task_index in enumerate(
            rng.integers(0, len(groups), size=len(groups))
        ):
            episodes = groups[int(task_index)]
            episode_means = np.empty(
                (len(episodes), matrix.shape[0]), dtype=np.float64
            )
            for episode_output, episode_index in enumerate(
                rng.integers(0, len(episodes), size=len(episodes))
            ):
                members = episodes[int(episode_index)]
                selected = members[
                    rng.integers(0, members.size, size=members.size)
                ]
                episode_means[episode_output] = matrix[:, selected].mean(axis=1)
            task_means[task_output] = episode_means.mean(axis=0)
        estimates[replicate] = task_means.mean(axis=0)
    return estimates


def _ratio_estimates(joint_means: np.ndarray) -> tuple[np.ndarray, int]:
    # Columns are condition, current, wrong.
    denominator = joint_means[:, 2] - joint_means[:, 1]
    scale = np.maximum(
        1.0, np.maximum(np.abs(joint_means[:, 1]), np.abs(joint_means[:, 2]))
    )
    valid = np.isfinite(denominator) & (
        np.abs(denominator) > np.finfo(np.float64).eps * scale
    )
    recoveries = 1.0 - (
        joint_means[valid, 0] - joint_means[valid, 1]
    ) / denominator[valid]
    if recoveries.size == 0:
        raise ValueError("Every bootstrap replicate has a degenerate world endpoint.")
    return recoveries, int((~valid).sum())


def world_recovery_statistics(
    keys: Sequence[WorldSampleKey],
    condition_losses: Sequence[float] | np.ndarray,
    current_losses: Sequence[float] | np.ndarray,
    wrong_losses: Sequence[float] | np.ndarray,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    seed_label: str = "world_recovery",
) -> dict[str, Any]:
    """Point estimate and paired/hierarchical CIs for WorldRecovery."""

    count = _positive_integer(bootstrap_samples, name="bootstrap_samples")
    condition = _finite_vector(condition_losses, name="condition losses")
    current = _finite_vector(current_losses, name="current losses")
    wrong = _finite_vector(wrong_losses, name="wrong losses")
    if condition.shape != current.shape or current.shape != wrong.shape:
        raise ValueError("World-recovery loss vectors must have identical shapes.")
    normalized_keys = _validate_keys(keys, expected_size=condition.size)
    point = world_recovery(
        float(condition.mean()),
        current_loss=float(current.mean()),
        wrong_loss=float(wrong.mean()),
    )
    matrix = np.stack([condition, current, wrong], axis=0)
    paired_means = _joint_paired_mean_estimates(
        matrix,
        samples=count,
        seed=_stable_seed(bootstrap_seed, seed_label, "paired"),
    )
    paired_values, paired_degenerate = _ratio_estimates(paired_means)
    hierarchical_means = _joint_hierarchical_mean_estimates(
        normalized_keys,
        matrix,
        samples=count,
        seed=_stable_seed(bootstrap_seed, seed_label, "hierarchical"),
    )
    hierarchical_values, hierarchical_degenerate = _ratio_estimates(
        hierarchical_means
    )
    paired_low, paired_high = _percentile_interval(paired_values)
    hierarchical_low, hierarchical_high = _percentile_interval(
        hierarchical_values
    )
    return {
        "world_recovery": point,
        "paired_ci_low": paired_low,
        "paired_ci_high": paired_high,
        "task_hierarchical_ci_low": hierarchical_low,
        "task_hierarchical_ci_high": hierarchical_high,
        "paired_valid_replicates": int(paired_values.size),
        "paired_degenerate_replicates": paired_degenerate,
        "task_hierarchical_valid_replicates": int(hierarchical_values.size),
        "task_hierarchical_degenerate_replicates": hierarchical_degenerate,
        "unclipped": True,
    }


def world_endpoint_gate(
    keys: Sequence[WorldSampleKey],
    current_losses: Sequence[float] | np.ndarray,
    wrong_losses: Sequence[float] | np.ndarray,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Apply the pre-projection Current-vs-Wrong informativeness gate.

    The degradation is ``wrong - current`` because the native metric is a
    lower-is-better loss.  The endpoint is informative only when the aggregate
    mean is positive and the paired 95% interval excludes zero on the positive
    side.  The median and hierarchical interval are descriptive and are not
    additional gates in the registered protocol.
    """

    current = _finite_vector(current_losses, name="current losses")
    wrong = _finite_vector(wrong_losses, name="wrong losses")
    if current.shape != wrong.shape:
        raise ValueError("Endpoint loss vectors must have identical shapes.")
    normalized_keys = _validate_keys(keys, expected_size=current.size)
    degradation = wrong - current
    paired_low, paired_high = paired_bootstrap_ci(
        degradation,
        samples=bootstrap_samples,
        seed=_stable_seed(bootstrap_seed, "world_endpoint", "paired"),
    )
    hierarchical_low, hierarchical_high = task_hierarchical_bootstrap_ci(
        normalized_keys,
        degradation,
        samples=bootstrap_samples,
        seed=_stable_seed(bootstrap_seed, "world_endpoint", "hierarchical"),
    )
    mean_degradation = float(degradation.mean())
    median_degradation = float(np.median(degradation))
    informative = (
        mean_degradation > 0.0
        and paired_low > 0.0
    )
    return {
        "classification": None if informative else "WORLD-ENDPOINT-UNINFORMATIVE",
        "informative": informative,
        "sample_count": int(current.size),
        "current_mean_loss": float(current.mean()),
        "current_median_loss": float(np.median(current)),
        "wrong_mean_loss": float(wrong.mean()),
        "wrong_median_loss": float(np.median(wrong)),
        "wrong_minus_current_mean": mean_degradation,
        "wrong_minus_current_median": median_degradation,
        "paired_ci_low": paired_low,
        "paired_ci_high": paired_high,
        "task_hierarchical_ci_low": hierarchical_low,
        "task_hierarchical_ci_high": hierarchical_high,
        "gate_requirements": {
            "mean_wrong_minus_current_strictly_positive": mean_degradation > 0.0,
            "median_wrong_minus_current_strictly_positive_descriptive": (
                median_degradation > 0.0
            ),
            "paired_ci_lower_strictly_positive": paired_low > 0.0,
        },
    }


def _difference_statistics(
    keys: Sequence[WorldSampleKey],
    differences: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    seed_label: str,
) -> dict[str, float]:
    paired_low, paired_high = paired_bootstrap_ci(
        differences,
        samples=bootstrap_samples,
        seed=_stable_seed(bootstrap_seed, seed_label, "paired"),
    )
    hierarchical_low, hierarchical_high = task_hierarchical_bootstrap_ci(
        keys,
        differences,
        samples=bootstrap_samples,
        seed=_stable_seed(bootstrap_seed, seed_label, "hierarchical"),
    )
    return {
        "mean": float(differences.mean()),
        "median": float(np.median(differences)),
        "paired_ci_low": paired_low,
        "paired_ci_high": paired_high,
        "task_hierarchical_ci_low": hierarchical_low,
        "task_hierarchical_ci_high": hierarchical_high,
    }


def analyze_world_losses(
    keys: Sequence[WorldSampleKey],
    losses_by_condition: Mapping[str, Sequence[float] | np.ndarray],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Analyze exactly four paired, sample-averaged Salvage-B loss vectors."""

    if set(losses_by_condition) != set(CONDITIONS):
        raise ValueError(
            "Salvage-B world losses must contain exactly "
            f"{list(CONDITIONS)}, got {sorted(losses_by_condition)}."
        )
    vectors = {
        condition: _finite_vector(
            losses_by_condition[condition], name=f"{condition} losses"
        )
        for condition in CONDITIONS
    }
    size = vectors[CONDITIONS[0]].size
    if any(vector.size != size for vector in vectors.values()):
        raise ValueError("All Salvage-B world-loss vectors must have equal length.")
    normalized_keys = _validate_keys(keys, expected_size=size)
    current = vectors["current_all"]
    wrong = vectors["wrong_all"]
    endpoint = world_endpoint_gate(
        normalized_keys,
        current,
        wrong,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    rows: dict[str, Any] = {}
    for condition in CONDITIONS:
        values = vectors[condition]
        versus_current = _difference_statistics(
            normalized_keys,
            values - current,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            seed_label=f"{condition}_minus_current",
        )
        versus_wrong = _difference_statistics(
            normalized_keys,
            values - wrong,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            seed_label=f"{condition}_minus_wrong",
        )
        recovery = world_recovery_statistics(
            normalized_keys,
            values,
            current,
            wrong,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            seed_label=condition,
        )
        rows[condition] = {
            "condition": condition,
            "sample_count": int(values.size),
            "mean_loss": float(values.mean()),
            "median_loss": float(np.median(values)),
            "delta_vs_current": versus_current,
            "delta_vs_wrong": versus_wrong,
            **recovery,
        }
    return {
        "conditions": rows,
        "endpoint_gate": endpoint,
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "paired_unit": "evaluation_sample_after_averaging_four_frozen_draws",
            "task_hierarchy": "task_then_episode_then_evaluation_sample",
        },
        "world_metric_direction": "lower_is_better",
    }
