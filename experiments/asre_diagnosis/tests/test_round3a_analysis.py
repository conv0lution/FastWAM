from __future__ import annotations

import unittest

import numpy as np

from experiments.asre_diagnosis.round3a.aggregate_factorial import (
    _flatten_simple_rows,
    _parse_simple_context,
)
from experiments.asre_diagnosis.round3a.factorial_stats import (
    CELL_ORDER,
    analyze_factorial,
    simple_effect_transitions,
)


class Round3AAggregateContractTest(unittest.TestCase):
    def test_context_parser_accepts_only_two_fixed_factors(self) -> None:
        self.assertEqual(
            _parse_simple_context("B=0,C=1", factor="A"), {"B": 0, "C": 1}
        )
        with self.assertRaises(ValueError):
            _parse_simple_context("A=0,C=1", factor="A")
        with self.assertRaises(ValueError):
            _parse_simple_context("B=0,B=1", factor="A")
        with self.assertRaises(ValueError):
            _parse_simple_context("B=2,C=1", factor="A")

    def test_stats_rows_flatten_into_csv_contract(self) -> None:
        keys = [
            ("libero_spatial", task_id, episode_id)
            for task_id in range(2)
            for episode_id in range(2)
        ]
        outcomes = np.zeros((len(keys), len(CELL_ORDER)), dtype=int)
        outcomes[:, CELL_ORDER.index("000")] = [1, 1, 0, 0]
        outcomes[:, CELL_ORDER.index("010")] = [1, 0, 1, 0]
        analysis = analyze_factorial(
            keys,
            outcomes,
            bootstrap_samples=50,
            bootstrap_seed=3,
            sign_flip_samples=50,
            sign_flip_seed=5,
        )
        transitions = simple_effect_transitions(keys, outcomes)

        simple_rows, transition_rows = _flatten_simple_rows(analysis, transitions)

        self.assertEqual(len(simple_rows), 12)
        self.assertEqual(len(transition_rows), 12)
        simple = {row["effect_id"]: row for row in simple_rows}
        paired = {row["effect_id"]: row for row in transition_rows}
        target = simple["simple_B_A0_C0"]
        self.assertEqual(target["context_A"], 0)
        self.assertEqual(target["context_B"], "")
        self.assertEqual(target["context_C"], 0)
        transition = paired["simple_B_A0_C0"]
        self.assertEqual(transition["reference_success_to_target_success"], 1)
        self.assertEqual(transition["reference_success_to_target_failure"], 1)
        self.assertEqual(transition["reference_failure_to_target_success"], 1)
        self.assertEqual(transition["reference_failure_to_target_failure"], 1)


if __name__ == "__main__":
    unittest.main()
