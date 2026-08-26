from __future__ import annotations

import unittest

import numpy as np

from experiments.asre_diagnosis.round3a.factorial_stats import (
    CELL_ORDER,
    FACTORIAL_CONTRAST_ORDER,
    FACTORIAL_CONTRASTS,
    SIMPLE_EFFECT_ORDER,
    analyze_factorial,
    exact_mcnemar_p_value,
    holm_adjust,
    joint_paired_sign_flip_p_values,
    simple_effect_transitions,
)


def _keys(number_of_tasks: int = 2, episodes_per_task: int = 2):
    return [
        ("libero_spatial", task_id, episode_id)
        for task_id in range(number_of_tasks)
        for episode_id in range(episodes_per_task)
    ]


def _small_analysis(keys, outcomes):
    return analyze_factorial(
        keys,
        outcomes,
        bootstrap_samples=200,
        bootstrap_seed=17,
        sign_flip_samples=500,
        sign_flip_seed=23,
    )


class Round3AFactorialStatsTest(unittest.TestCase):
    def test_fixed_orders_and_interaction_scales(self) -> None:
        self.assertEqual(CELL_ORDER, tuple(f"{index:03b}" for index in range(8)))
        self.assertEqual(
            FACTORIAL_CONTRAST_ORDER,
            (
                "main_A",
                "main_B",
                "main_C",
                "interaction_AB",
                "interaction_AC",
                "interaction_BC",
                "interaction_ABC",
            ),
        )
        self.assertEqual(
            SIMPLE_EFFECT_ORDER,
            (
                "simple_A_B0_C0",
                "simple_A_B1_C0",
                "simple_A_B0_C1",
                "simple_A_B1_C1",
                "simple_B_A0_C0",
                "simple_B_A1_C0",
                "simple_B_A0_C1",
                "simple_B_A1_C1",
                "simple_C_A0_B0",
                "simple_C_A1_B0",
                "simple_C_A0_B1",
                "simple_C_A1_B1",
            ),
        )
        coefficient_by_name = {
            definition.estimand: np.asarray(definition.coefficients)
            for definition in FACTORIAL_CONTRASTS
        }

        # Pure AB has a unit averaged difference-in-differences, not the
        # half-sized +/-1-regression coefficient.
        pure_ab = np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=float)
        self.assertAlmostEqual(coefficient_by_name["main_A"] @ pure_ab, 0.5)
        self.assertAlmostEqual(coefficient_by_name["main_B"] @ pure_ab, 0.5)
        self.assertAlmostEqual(coefficient_by_name["interaction_AB"] @ pure_ab, 1.0)
        self.assertAlmostEqual(coefficient_by_name["interaction_ABC"] @ pure_ab, 0.0)

        # A unit pure ABC cell verifies the specified change-in-DiD scale.
        pure_abc = np.asarray([0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
        for name in ("main_A", "main_B", "main_C"):
            self.assertAlmostEqual(coefficient_by_name[name] @ pure_abc, 0.25)
        for name in ("interaction_AB", "interaction_AC", "interaction_BC"):
            self.assertAlmostEqual(coefficient_by_name[name] @ pure_abc, 0.5)
        self.assertAlmostEqual(coefficient_by_name["interaction_ABC"] @ pure_abc, 1.0)

    def test_additive_cell_means_have_zero_interactions(self) -> None:
        # y = 0.25*A + 0.50*B + 0.25*C, represented by four binary rows.
        expected_means = np.asarray([0, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 1.0])
        outcomes = np.zeros((4, 8), dtype=int)
        for cell_index, count in enumerate((expected_means * 4).astype(int)):
            outcomes[4 - count :, cell_index] = 1
        result = _small_analysis(_keys(), outcomes)
        self.assertEqual(result["cell_success_rates"], dict(zip(CELL_ORDER, expected_means)))
        rows = {row["estimand"]: row for row in result["factorial_contrasts"]}
        self.assertAlmostEqual(rows["main_A"]["effect"], 0.25)
        self.assertAlmostEqual(rows["main_B"]["effect"], 0.50)
        self.assertAlmostEqual(rows["main_C"]["effect"], 0.25)
        for name in (
            "interaction_AB",
            "interaction_AC",
            "interaction_BC",
            "interaction_ABC",
        ):
            self.assertAlmostEqual(rows[name]["effect"], 0.0)

        simple = {row["estimand"]: row for row in result["simple_effects"]}
        self.assertAlmostEqual(simple["simple_B_A0_C1"]["effect"], 0.5)
        self.assertEqual(simple["simple_B_A0_C1"]["reference_cell"], "001")
        self.assertEqual(simple["simple_B_A0_C1"]["target_cell"], "011")

    def test_joint_bootstraps_preserve_complete_episode_vectors(self) -> None:
        row_values = np.asarray([0, 1, 0, 1], dtype=int)
        outcomes = np.repeat(row_values[:, None], len(CELL_ORDER), axis=1)
        result = _small_analysis(_keys(), outcomes)
        for row in result["simple_effects"] + result["factorial_contrasts"]:
            self.assertEqual(row["effect"], 0.0)
            self.assertEqual(row["paired_ci_low"], 0.0)
            self.assertEqual(row["paired_ci_high"], 0.0)
            self.assertEqual(row["task_hierarchical_ci_low"], 0.0)
            self.assertEqual(row["task_hierarchical_ci_high"], 0.0)
        for row in result["factorial_contrasts"]:
            self.assertEqual(row["raw_p_value"], 1.0)
            self.assertEqual(row["holm_p_value"], 1.0)

    def test_canonical_key_order_makes_seeded_output_permutation_invariant(self) -> None:
        keys = _keys(number_of_tasks=3, episodes_per_task=2)
        rng = np.random.default_rng(5)
        outcomes = rng.integers(0, 2, size=(len(keys), len(CELL_ORDER)))
        original = _small_analysis(keys, outcomes)
        permutation = np.asarray([4, 1, 5, 0, 3, 2])
        shuffled = _small_analysis(
            [keys[index] for index in permutation], outcomes[permutation]
        )
        self.assertEqual(original, shuffled)

    def test_exact_small_n_sign_flip_matches_enumeration(self) -> None:
        # Each episode has main-A score +1.  Of four sign assignments, two
        # attain |mean| >= 1, so the exact two-sided p-value is 1/2.
        outcomes = np.asarray(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ],
            dtype=int,
        )
        p_values = joint_paired_sign_flip_p_values(
            [("libero_spatial", 0, 0), ("libero_spatial", 0, 1)],
            outcomes,
            samples=None,
        )
        self.assertEqual(p_values["main_A"], 0.5)
        self.assertEqual(p_values["interaction_AB"], 1.0)

    def test_monte_carlo_sign_flip_is_reproducible_and_never_zero(self) -> None:
        keys = _keys()
        outcomes = np.asarray(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ],
            dtype=int,
        )
        first = joint_paired_sign_flip_p_values(keys, outcomes, samples=31, seed=9)
        second = joint_paired_sign_flip_p_values(keys, outcomes, samples=31, seed=9)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["main_A"], 1.0 / 32.0)

    def test_simple_effect_transition_orientation_and_mcnemar(self) -> None:
        outcomes = np.zeros((4, len(CELL_ORDER)), dtype=int)
        outcomes[:, CELL_ORDER.index("000")] = [1, 1, 0, 0]
        outcomes[:, CELL_ORDER.index("010")] = [1, 0, 1, 0]
        rows = {
            row["estimand"]: row
            for row in simple_effect_transitions(_keys(), outcomes)
        }
        target = rows["simple_B_A0_C0"]
        self.assertEqual(target["success_to_success"], 1)
        self.assertEqual(target["success_to_failure"], 1)
        self.assertEqual(target["failure_to_success"], 1)
        self.assertEqual(target["failure_to_failure"], 1)
        self.assertEqual(target["induced_failures"], 1)
        self.assertEqual(target["rescued_successes"], 1)
        self.assertEqual(target["mcnemar_exact_p_value"], 1.0)
        self.assertIn("outside_factorial_holm_family", target["multiplicity_status"])

    def test_holm_and_mcnemar_boundaries(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertEqual(exact_mcnemar_p_value(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p_value(3, 0), 0.25)
        with self.assertRaises(ValueError):
            holm_adjust({"bad": float("nan")})
        with self.assertRaises(ValueError):
            exact_mcnemar_p_value(-1, 0)

    def test_output_contract_and_theoretical_ranges(self) -> None:
        outcomes = np.zeros((4, len(CELL_ORDER)), dtype=int)
        outcomes[:, -1] = 1
        result = _small_analysis(_keys(), outcomes)
        self.assertEqual(len(result["simple_effects"]), 12)
        self.assertEqual(len(result["factorial_contrasts"]), 7)
        self.assertEqual(result["ci_scope"], "pointwise")
        self.assertEqual(result["n_episodes"], 4)
        self.assertEqual(result["n_tasks"], 2)
        for row in result["simple_effects"]:
            self.assertLessEqual(-1.0, row["effect"])
            self.assertLessEqual(row["effect"], 1.0)
        contrast_ranges = {
            "main_A": 1.0,
            "main_B": 1.0,
            "main_C": 1.0,
            "interaction_AB": 2.0,
            "interaction_AC": 2.0,
            "interaction_BC": 2.0,
            "interaction_ABC": 4.0,
        }
        for row in result["factorial_contrasts"]:
            bound = contrast_ranges[row["estimand"]]
            self.assertLessEqual(-bound, row["effect"])
            self.assertLessEqual(row["effect"], bound)
            self.assertEqual(row["holm_family_size"], 7)
            self.assertEqual(len(row["coefficients"]), 8)

    def test_strict_input_validation(self) -> None:
        valid_keys = _keys()
        valid = np.zeros((4, len(CELL_ORDER)), dtype=int)
        invalid_cases = (
            (valid_keys, np.zeros((4, 7))),
            (valid_keys[:-1], valid),
            (valid_keys, np.where(np.indices(valid.shape)[0] == 0, 2, valid)),
            (valid_keys, np.where(np.indices(valid.shape)[0] == 0, np.nan, valid)),
            ([valid_keys[0]] * 4, valid),
            (["not-a-key"] * 4, valid),
        )
        for keys, outcomes in invalid_cases:
            with self.subTest(keys=keys, shape=np.asarray(outcomes).shape):
                with self.assertRaises(ValueError):
                    _small_analysis(keys, outcomes)
        with self.assertRaises(ValueError):
            _small_analysis([], np.zeros((0, len(CELL_ORDER))))
        with self.assertRaises(ValueError):
            analyze_factorial(
                valid_keys,
                valid,
                bootstrap_samples=0,
                sign_flip_samples=10,
            )
        with self.assertRaises(ValueError):
            analyze_factorial(
                valid_keys,
                valid,
                bootstrap_samples=10,
                sign_flip_samples=0,
            )

    def test_exact_sign_flip_rejects_more_than_twenty_episodes(self) -> None:
        keys = [("libero_spatial", 0, index) for index in range(21)]
        outcomes = np.zeros((21, len(CELL_ORDER)), dtype=int)
        with self.assertRaisesRegex(ValueError, "at most 20"):
            joint_paired_sign_flip_p_values(keys, outcomes, samples=None)


if __name__ == "__main__":
    unittest.main()
