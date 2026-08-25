from __future__ import annotations

import unittest

import numpy as np

from experiments.asre_diagnosis.metrics import compute_action_deviation_metrics


class ActionDeviationMetricsTest(unittest.TestCase):
    def test_requested_metrics_and_per_horizon_values(self) -> None:
        baseline_raw = np.asarray([[1.0, 2.0, 0.2], [3.0, 4.0, 0.8]])
        diagnosis_raw = np.asarray([[2.0, 2.0, 0.4], [3.0, 2.0, 0.7]])
        baseline_executed = np.asarray([[1.0, 2.0, -1.0], [3.0, 4.0, 1.0]])
        diagnosis_executed = np.asarray([[2.0, 2.0, 1.0], [3.0, 2.0, 1.0]])

        metrics = compute_action_deviation_metrics(
            baseline_raw,
            diagnosis_raw,
            baseline_executed,
            diagnosis_executed,
        )

        self.assertAlmostEqual(metrics["continuous_action_mae"], 0.75)
        self.assertAlmostEqual(
            metrics["normalized_continuous_action_l2"],
            np.sqrt(1.25),
        )
        self.assertAlmostEqual(metrics["raw_gripper_difference"], 0.15)
        self.assertAlmostEqual(metrics["post_binarization_gripper_flip_rate"], 0.5)
        np.testing.assert_allclose(
            metrics["per_action_horizon_l2_deviation"],
            [np.sqrt(0.5), np.sqrt(2.0)],
        )

    def test_identical_actions_have_unit_cosine_and_zero_deviation(self) -> None:
        action = np.asarray([[0.5, -0.25, 1.0]])
        metrics = compute_action_deviation_metrics(action, action, action, action)
        self.assertEqual(metrics["normalized_continuous_action_l2"], 0.0)
        self.assertAlmostEqual(metrics["continuous_action_cosine_similarity"], 1.0)
        self.assertEqual(metrics["post_binarization_gripper_flip_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
