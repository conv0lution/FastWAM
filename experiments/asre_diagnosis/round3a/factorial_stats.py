"""Paired statistics for the ASRE Round-3A late-half 2^3 factorial.

The eight columns are always ordered as ``000, 001, ..., 111``, where the
bits denote direct video-K/V retrieval in action-layer regions A=15--19,
B=20--24, and C=25--29.  Every resampling operation samples a complete
eight-cell outcome vector, preserving the task/trial pairing by construction.

Reported two-way interactions are averaged differences-in-differences and
the three-way interaction is a change in difference-in-differences.  These
scales follow the Round-3A specification; they are not the coefficients from
a +/-1-coded regression and the distinction is intentional.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np


EpisodeKey = tuple[str, int, int]

CELL_ORDER = (
    "000",
    "001",
    "010",
    "011",
    "100",
    "101",
    "110",
    "111",
)


@dataclass(frozen=True)
class EstimandDefinition:
    """A named linear contrast over the cells in :data:`CELL_ORDER`."""

    estimand: str
    kind: str
    formula: str
    coefficients: tuple[float, ...]
    factor: str | None = None
    context: str | None = None
    reference_cell: str | None = None
    target_cell: str | None = None

    def __post_init__(self) -> None:
        if len(self.coefficients) != len(CELL_ORDER):
            raise ValueError(
                f"{self.estimand} must define {len(CELL_ORDER)} coefficients."
            )


def _pair_coefficients(reference_cell: str, target_cell: str) -> tuple[float, ...]:
    coefficients = [0.0] * len(CELL_ORDER)
    coefficients[CELL_ORDER.index(reference_cell)] = -1.0
    coefficients[CELL_ORDER.index(target_cell)] = 1.0
    return tuple(coefficients)


# Context order follows the specification: the first named context factor
# changes fastest, followed by the second named context factor.
SIMPLE_EFFECTS = (
    EstimandDefinition(
        "simple_A_B0_C0",
        "simple_effect",
        "y100 - y000",
        _pair_coefficients("000", "100"),
        "A",
        "B=0,C=0",
        "000",
        "100",
    ),
    EstimandDefinition(
        "simple_A_B1_C0",
        "simple_effect",
        "y110 - y010",
        _pair_coefficients("010", "110"),
        "A",
        "B=1,C=0",
        "010",
        "110",
    ),
    EstimandDefinition(
        "simple_A_B0_C1",
        "simple_effect",
        "y101 - y001",
        _pair_coefficients("001", "101"),
        "A",
        "B=0,C=1",
        "001",
        "101",
    ),
    EstimandDefinition(
        "simple_A_B1_C1",
        "simple_effect",
        "y111 - y011",
        _pair_coefficients("011", "111"),
        "A",
        "B=1,C=1",
        "011",
        "111",
    ),
    EstimandDefinition(
        "simple_B_A0_C0",
        "simple_effect",
        "y010 - y000",
        _pair_coefficients("000", "010"),
        "B",
        "A=0,C=0",
        "000",
        "010",
    ),
    EstimandDefinition(
        "simple_B_A1_C0",
        "simple_effect",
        "y110 - y100",
        _pair_coefficients("100", "110"),
        "B",
        "A=1,C=0",
        "100",
        "110",
    ),
    EstimandDefinition(
        "simple_B_A0_C1",
        "simple_effect",
        "y011 - y001",
        _pair_coefficients("001", "011"),
        "B",
        "A=0,C=1",
        "001",
        "011",
    ),
    EstimandDefinition(
        "simple_B_A1_C1",
        "simple_effect",
        "y111 - y101",
        _pair_coefficients("101", "111"),
        "B",
        "A=1,C=1",
        "101",
        "111",
    ),
    EstimandDefinition(
        "simple_C_A0_B0",
        "simple_effect",
        "y001 - y000",
        _pair_coefficients("000", "001"),
        "C",
        "A=0,B=0",
        "000",
        "001",
    ),
    EstimandDefinition(
        "simple_C_A1_B0",
        "simple_effect",
        "y101 - y100",
        _pair_coefficients("100", "101"),
        "C",
        "A=1,B=0",
        "100",
        "101",
    ),
    EstimandDefinition(
        "simple_C_A0_B1",
        "simple_effect",
        "y011 - y010",
        _pair_coefficients("010", "011"),
        "C",
        "A=0,B=1",
        "010",
        "011",
    ),
    EstimandDefinition(
        "simple_C_A1_B1",
        "simple_effect",
        "y111 - y110",
        _pair_coefficients("110", "111"),
        "C",
        "A=1,B=1",
        "110",
        "111",
    ),
)

FACTORIAL_CONTRASTS = (
    EstimandDefinition(
        "main_A",
        "factorial_contrast",
        "0.25 * [(y100-y000) + (y101-y001) + (y110-y010) + (y111-y011)]",
        (-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25),
        "A",
    ),
    EstimandDefinition(
        "main_B",
        "factorial_contrast",
        "0.25 * [(y010-y000) + (y011-y001) + (y110-y100) + (y111-y101)]",
        (-0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25),
        "B",
    ),
    EstimandDefinition(
        "main_C",
        "factorial_contrast",
        "0.25 * [(y001-y000) + (y011-y010) + (y101-y100) + (y111-y110)]",
        (-0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25),
        "C",
    ),
    EstimandDefinition(
        "interaction_AB",
        "factorial_contrast",
        "0.5 * [(y110-y010-y100+y000) + (y111-y011-y101+y001)]",
        (0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5),
    ),
    EstimandDefinition(
        "interaction_AC",
        "factorial_contrast",
        "0.5 * [(y101-y001-y100+y000) + (y111-y011-y110+y010)]",
        (0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5),
    ),
    EstimandDefinition(
        "interaction_BC",
        "factorial_contrast",
        "0.5 * [(y011-y001-y010+y000) + (y111-y101-y110+y100)]",
        (0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5),
    ),
    EstimandDefinition(
        "interaction_ABC",
        "factorial_contrast",
        "(y111-y101-y011+y001) - (y110-y100-y010+y000)",
        (-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
    ),
)

SIMPLE_EFFECT_ORDER = tuple(definition.estimand for definition in SIMPLE_EFFECTS)
FACTORIAL_CONTRAST_ORDER = tuple(
    definition.estimand for definition in FACTORIAL_CONTRASTS
)


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def _integer_seed(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative, got {result}.")
    return result


def _normalize_inputs(
    keys: Sequence[EpisodeKey], outcomes: np.ndarray | Sequence[Sequence[int]]
) -> tuple[tuple[EpisodeKey, ...], np.ndarray]:
    """Validate and canonically order paired keys and an N-by-8 binary matrix."""

    raw_keys = list(keys)
    try:
        numeric = np.asarray(outcomes, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcomes must be a finite binary numeric matrix.") from exc
    if numeric.ndim != 2 or numeric.shape[1:] != (len(CELL_ORDER),):
        raise ValueError(
            f"outcomes must have shape (N, {len(CELL_ORDER)}) in CELL_ORDER; "
            f"got {numeric.shape}."
        )
    if numeric.shape[0] == 0:
        raise ValueError("At least one paired episode is required.")
    if len(raw_keys) != numeric.shape[0]:
        raise ValueError(
            f"Expected one key per outcome row, got {len(raw_keys)} keys and "
            f"{numeric.shape[0]} rows."
        )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("outcomes must contain only finite values.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError("outcomes must contain only binary values 0 or 1.")

    normalized: list[EpisodeKey] = []
    for row_index, key in enumerate(raw_keys):
        if not isinstance(key, (tuple, list)) or len(key) != 3:
            raise ValueError(
                f"Key at row {row_index} must be (suite, task_id, episode_id); "
                f"got {key!r}."
            )
        suite, task_id, episode_id = key
        if not isinstance(suite, str) or not suite:
            raise ValueError(f"Key at row {row_index} has an invalid suite: {suite!r}.")
        if (
            isinstance(task_id, (bool, np.bool_))
            or not isinstance(task_id, Integral)
            or isinstance(episode_id, (bool, np.bool_))
            or not isinstance(episode_id, Integral)
        ):
            raise ValueError(
                f"Key at row {row_index} must use integer task/episode IDs; got {key!r}."
            )
        normalized.append((suite, int(task_id), int(episode_id)))

    if len(set(normalized)) != len(normalized):
        duplicates = sorted(key for key, count in Counter(normalized).items() if count > 1)
        raise ValueError(f"Duplicate paired episode keys are not allowed: {duplicates!r}.")

    canonical_indices = sorted(range(len(normalized)), key=normalized.__getitem__)
    canonical_keys = tuple(normalized[index] for index in canonical_indices)
    canonical_outcomes = np.asarray(numeric[canonical_indices], dtype=np.float64)
    return canonical_keys, canonical_outcomes


def _coefficient_matrix(
    definitions: Sequence[EstimandDefinition],
) -> np.ndarray:
    return np.asarray([definition.coefficients for definition in definitions], dtype=float)


def _joint_paired_bootstrap_cell_means(
    outcomes: np.ndarray, *, samples: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    number_of_rows = outcomes.shape[0]
    estimates = np.empty((samples, len(CELL_ORDER)), dtype=np.float64)
    # Keep the temporary sampled tensor near 16 MB for large N.
    chunk_size = max(1, min(samples, 2_000_000 // (number_of_rows * len(CELL_ORDER))))
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(0, number_of_rows, size=(stop - start, number_of_rows))
        estimates[start:stop] = outcomes[indices].mean(axis=1)
    return estimates


def _joint_task_hierarchical_bootstrap_cell_means(
    keys: Sequence[EpisodeKey],
    outcomes: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    grouped_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row_index, (suite, task_id, _episode_id) in enumerate(keys):
        grouped_indices[(suite, task_id)].append(row_index)
    groups = [np.asarray(grouped_indices[key], dtype=int) for key in sorted(grouped_indices)]
    if not groups:
        raise ValueError("Task-hierarchical bootstrap requires at least one task.")

    rng = np.random.default_rng(seed)
    number_of_tasks = len(groups)
    estimates = np.empty((samples, len(CELL_ORDER)), dtype=np.float64)
    task_means = np.empty((number_of_tasks, len(CELL_ORDER)), dtype=np.float64)
    for bootstrap_index in range(samples):
        selected_groups = rng.integers(0, number_of_tasks, size=number_of_tasks)
        for output_index, selected_group in enumerate(selected_groups):
            members = groups[int(selected_group)]
            selected_members = members[
                rng.integers(0, members.size, size=members.size)
            ]
            task_means[output_index] = outcomes[selected_members].mean(axis=0)
        # Equal task weighting matches the validated Round-2 hierarchy.
        estimates[bootstrap_index] = task_means.mean(axis=0)
    return estimates


def _pointwise_percentile_intervals(estimates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.quantile(estimates, [0.025, 0.975], axis=0)
    return np.asarray(low, dtype=float), np.asarray(high, dtype=float)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values, preserving the input names."""

    checked: list[tuple[float, str]] = []
    for name, raw_value in p_values.items():
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid p-value for {name}: {raw_value!r}.")
        checked.append((value, str(name)))
    ordered = sorted(checked)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    family_size = len(ordered)
    for rank, (value, name) in enumerate(ordered):
        running_max = max(running_max, (family_size - rank) * value)
        adjusted[name] = float(min(1.0, running_max))
    return adjusted


def _sign_flip_p_values_from_outcomes(
    outcomes: np.ndarray,
    *,
    samples: int | None,
    seed: int,
) -> dict[str, float]:
    coefficients = _coefficient_matrix(FACTORIAL_CONTRASTS)
    scores = outcomes @ coefficients.T
    observed = np.abs(scores.mean(axis=0))
    number_of_rows = outcomes.shape[0]
    tolerance = 1e-15

    if samples is None:
        if number_of_rows > 20:
            raise ValueError(
                "Exact sign-flip enumeration is limited to at most 20 paired episodes."
            )
        total = 1 << number_of_rows
        exceedances = np.zeros(len(FACTORIAL_CONTRASTS), dtype=np.int64)
        shifts = np.arange(number_of_rows, dtype=np.uint64)
        chunk_size = 65_536
        for start in range(0, total, chunk_size):
            stop = min(total, start + chunk_size)
            labels = np.arange(start, stop, dtype=np.uint64)[:, None]
            bits = ((labels >> shifts) & 1).astype(np.int8)
            signs = bits * 2 - 1
            permuted = np.abs((signs.astype(float) @ scores) / number_of_rows)
            exceedances += np.sum(permuted >= observed - tolerance, axis=0)
        values = exceedances.astype(float) / float(total)
    else:
        samples = _positive_integer(samples, name="samples")
        rng = np.random.default_rng(seed)
        exceedances = np.zeros(len(FACTORIAL_CONTRASTS), dtype=np.int64)
        chunk_size = min(samples, 65_536)
        for start in range(0, samples, chunk_size):
            size = min(chunk_size, samples - start)
            bits = rng.integers(0, 2, size=(size, number_of_rows), dtype=np.int8)
            signs = bits * 2 - 1
            permuted = np.abs((signs.astype(float) @ scores) / number_of_rows)
            exceedances += np.sum(permuted >= observed - tolerance, axis=0)
        # The plus-one correction prevents zero Monte-Carlo p-values.
        values = (exceedances.astype(float) + 1.0) / (samples + 1.0)

    return {
        definition.estimand: float(value)
        for definition, value in zip(FACTORIAL_CONTRASTS, values)
    }


def joint_paired_sign_flip_p_values(
    keys: Sequence[EpisodeKey],
    outcomes: np.ndarray | Sequence[Sequence[int]],
    *,
    samples: int | None = 100_000,
    seed: int = 0,
) -> dict[str, float]:
    """Test the seven contrasts by jointly generated paired sign flips.

    ``samples=None`` requests exact enumeration and is restricted to at most
    20 paired episodes.  Monte-Carlo p-values use a plus-one correction.
    The test assumes sign-exchangeability of episode-level contrast scores;
    it is not a design-exact randomization test because retrieval schedules
    were not randomly assigned.
    """

    _keys, numeric = _normalize_inputs(keys, outcomes)
    seed = _integer_seed(seed, name="seed")
    return _sign_flip_p_values_from_outcomes(numeric, samples=samples, seed=seed)


def exact_mcnemar_p_value(induced_failures: int, rescued_successes: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""

    if (
        isinstance(induced_failures, (bool, np.bool_))
        or not isinstance(induced_failures, Integral)
        or isinstance(rescued_successes, (bool, np.bool_))
        or not isinstance(rescued_successes, Integral)
    ):
        raise ValueError("McNemar transition counts must be integers.")
    induced = int(induced_failures)
    rescued = int(rescued_successes)
    if induced < 0 or rescued < 0:
        raise ValueError("McNemar transition counts must be nonnegative.")
    discordant = induced + rescued
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(induced, rescued) + 1)
    )
    return float(min(1.0, 2.0 * tail / (2**discordant)))


def simple_effect_transitions(
    keys: Sequence[EpisodeKey],
    outcomes: np.ndarray | Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Return paired transitions for each of the 12 two-cell simple effects.

    McNemar values here are descriptive and unadjusted; they are not part of
    the pre-specified seven-contrast Holm family.
    """

    _keys, numeric = _normalize_inputs(keys, outcomes)
    rows: list[dict[str, Any]] = []
    for definition in SIMPLE_EFFECTS:
        assert definition.reference_cell is not None
        assert definition.target_cell is not None
        reference = numeric[:, CELL_ORDER.index(definition.reference_cell)].astype(int)
        target = numeric[:, CELL_ORDER.index(definition.target_cell)].astype(int)
        transitions = Counter(zip(reference.tolist(), target.tolist()))
        induced = int(transitions[(1, 0)])
        rescued = int(transitions[(0, 1)])
        rows.append(
            {
                "estimand": definition.estimand,
                "factor": definition.factor,
                "context": definition.context,
                "formula": definition.formula,
                "reference_cell": definition.reference_cell,
                "target_cell": definition.target_cell,
                "success_to_success": int(transitions[(1, 1)]),
                "success_to_failure": induced,
                "failure_to_success": rescued,
                "failure_to_failure": int(transitions[(0, 0)]),
                "induced_failures": induced,
                "rescued_successes": rescued,
                "discordant_pairs": induced + rescued,
                "mcnemar_exact_p_value": exact_mcnemar_p_value(induced, rescued),
                "multiplicity_status": "descriptive_unadjusted_outside_factorial_holm_family",
                "paired_episodes": int(numeric.shape[0]),
            }
        )
    return rows


def _result_row(
    definition: EstimandDefinition,
    *,
    effect: float,
    paired_low: float,
    paired_high: float,
    hierarchical_low: float,
    hierarchical_high: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "estimand": definition.estimand,
        "kind": definition.kind,
        "formula": definition.formula,
        "coefficients": [float(value) for value in definition.coefficients],
        "coefficient_by_cell": {
            cell: float(value)
            for cell, value in zip(CELL_ORDER, definition.coefficients)
        },
        "effect_scale": "success_probability",
        "effect": float(effect),
        "paired_ci_low": float(paired_low),
        "paired_ci_high": float(paired_high),
        "task_hierarchical_ci_low": float(hierarchical_low),
        "task_hierarchical_ci_high": float(hierarchical_high),
    }
    if definition.factor is not None:
        row["factor"] = definition.factor
    if definition.context is not None:
        row["context"] = definition.context
    if definition.reference_cell is not None:
        row["reference_cell"] = definition.reference_cell
    if definition.target_cell is not None:
        row["target_cell"] = definition.target_cell
    return row


def analyze_factorial(
    keys: Sequence[EpisodeKey],
    outcomes: np.ndarray | Sequence[Sequence[int]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    sign_flip_samples: int = 100_000,
    sign_flip_seed: int = 0,
) -> dict[str, Any]:
    """Estimate all Round-3A simple effects and factorial contrasts.

    Both bootstrap procedures jointly resample complete eight-cell vectors.
    Their percentile intervals are pointwise 95% intervals.  Only the seven
    pre-specified factorial contrasts receive sign-flip tests and a common
    seven-member Holm adjustment family.
    """

    canonical_keys, numeric = _normalize_inputs(keys, outcomes)
    bootstrap_samples = _positive_integer(
        bootstrap_samples, name="bootstrap_samples"
    )
    bootstrap_seed = _integer_seed(bootstrap_seed, name="bootstrap_seed")
    sign_flip_samples = _positive_integer(
        sign_flip_samples, name="sign_flip_samples"
    )
    sign_flip_seed = _integer_seed(sign_flip_seed, name="sign_flip_seed")

    definitions = SIMPLE_EFFECTS + FACTORIAL_CONTRASTS
    coefficient_matrix = _coefficient_matrix(definitions)
    cell_means = numeric.mean(axis=0)
    point_estimates = coefficient_matrix @ cell_means

    paired_cell_means = _joint_paired_bootstrap_cell_means(
        numeric, samples=bootstrap_samples, seed=bootstrap_seed
    )
    paired_estimates = paired_cell_means @ coefficient_matrix.T
    paired_low, paired_high = _pointwise_percentile_intervals(paired_estimates)

    hierarchical_cell_means = _joint_task_hierarchical_bootstrap_cell_means(
        canonical_keys,
        numeric,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    hierarchical_estimates = hierarchical_cell_means @ coefficient_matrix.T
    hierarchical_low, hierarchical_high = _pointwise_percentile_intervals(
        hierarchical_estimates
    )

    rows = [
        _result_row(
            definition,
            effect=point_estimates[index],
            paired_low=paired_low[index],
            paired_high=paired_high[index],
            hierarchical_low=hierarchical_low[index],
            hierarchical_high=hierarchical_high[index],
        )
        for index, definition in enumerate(definitions)
    ]
    number_of_simple_effects = len(SIMPLE_EFFECTS)
    simple_rows = rows[:number_of_simple_effects]
    contrast_rows = rows[number_of_simple_effects:]

    raw_p_values = _sign_flip_p_values_from_outcomes(
        numeric, samples=sign_flip_samples, seed=sign_flip_seed
    )
    holm_p_values = holm_adjust(raw_p_values)
    for row in contrast_rows:
        estimand = str(row["estimand"])
        row.update(
            {
                "raw_p_value": raw_p_values[estimand],
                "holm_p_value": holm_p_values[estimand],
                "test_method": "two_sided_joint_monte_carlo_paired_sign_flip",
                "test_assumption": "episode_contrast_score_sign_exchangeability",
                "holm_family": list(FACTORIAL_CONTRAST_ORDER),
                "holm_family_size": len(FACTORIAL_CONTRASTS),
            }
        )

    task_keys = {(suite, task_id) for suite, task_id, _episode_id in canonical_keys}
    return {
        "cell_order": list(CELL_ORDER),
        "cell_success_rates": {
            cell: float(value) for cell, value in zip(CELL_ORDER, cell_means)
        },
        "n_episodes": int(numeric.shape[0]),
        "n_tasks": len(task_keys),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "paired_bootstrap_unit": "complete_paired_episode_eight_cell_vector",
        "task_hierarchical_bootstrap": (
            "resample_tasks_then_paired_episode_eight_cell_vectors_within_task; "
            "equal_task_weighting"
        ),
        "ci_method": "percentile_95_percent",
        "ci_scope": "pointwise",
        "sign_flip_samples": sign_flip_samples,
        "sign_flip_seed": sign_flip_seed,
        "sign_flip_monte_carlo_plus_one_correction": True,
        "simple_effects": simple_rows,
        "factorial_contrasts": contrast_rows,
    }


__all__ = [
    "CELL_ORDER",
    "FACTORIAL_CONTRAST_ORDER",
    "FACTORIAL_CONTRASTS",
    "SIMPLE_EFFECT_ORDER",
    "SIMPLE_EFFECTS",
    "EstimandDefinition",
    "analyze_factorial",
    "exact_mcnemar_p_value",
    "holm_adjust",
    "joint_paired_sign_flip_p_values",
    "simple_effect_transitions",
]
