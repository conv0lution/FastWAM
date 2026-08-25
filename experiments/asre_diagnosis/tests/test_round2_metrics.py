from __future__ import annotations

import unittest

import numpy as np

from experiments.asre_diagnosis.round2.metrics import compute_round2_metrics


class Round2MetricsTest(unittest.TestCase):
    def test_primary_metric_uses_only_executed_prefix_and_dataset_std(self) -> None:
        baseline = np.zeros((12, 7), dtype=np.float64)
        diagnosis = baseline.copy()
        diagnosis[:10, 0] = 2.0
        diagnosis[10:, 0] = 200.0
        action_std = np.asarray([2.0, 1, 1, 1, 1, 1, 1], dtype=np.float64)

        metrics = compute_round2_metrics(
            baseline,
            diagnosis,
            baseline,
            diagnosis,
            action_std,
            executed_prefix_length=10,
            eps=0.0,
        )

        self.assertAlmostEqual(metrics["executed_prefix_norm_rms"], np.sqrt(1.0 / 6.0))
        self.assertGreater(
            metrics["full_chunk_norm_rms_0_31"],
            metrics["executed_prefix_norm_rms"],
        )
        self.assertAlmostEqual(
            metrics["executed_prefix_norm_rms_by_dimension"][0], 1.0
        )
        np.testing.assert_allclose(
            metrics["executed_prefix_norm_rms_by_dimension"][1:], 0.0
        )

    def test_gripper_prefix_and_full_horizon_are_separate(self) -> None:
        raw = np.zeros((12, 7), dtype=np.float64)
        baseline_executed = raw.copy()
        diagnosis_executed = raw.copy()
        diagnosis_executed[10:, -1] = 1.0
        metrics = compute_round2_metrics(
            raw,
            raw,
            baseline_executed,
            diagnosis_executed,
            np.ones(7),
            executed_prefix_length=10,
        )
        self.assertEqual(metrics["executed_prefix_gripper_flip_rate"], 0.0)
        self.assertAlmostEqual(metrics["full_horizon_gripper_flip_rate"], 2.0 / 12.0)

    def test_translation_and_rotation_groups_are_explicit(self) -> None:
        baseline = np.zeros((10, 7), dtype=np.float64)
        diagnosis = baseline.copy()
        diagnosis[:, :3] = 1.0
        metrics = compute_round2_metrics(
            baseline,
            diagnosis,
            baseline,
            diagnosis,
            np.ones(7),
        )
        self.assertAlmostEqual(metrics["translation_norm_rms"], 1.0)
        self.assertEqual(metrics["rotation_norm_rms"], 0.0)


if __name__ == "__main__":
    unittest.main()
