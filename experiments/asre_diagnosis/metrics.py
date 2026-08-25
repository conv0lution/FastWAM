from __future__ import annotations

from typing import Any

import numpy as np


def safe_cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_flat = lhs.reshape(-1).astype(np.float64)
    rhs_flat = rhs.reshape(-1).astype(np.float64)
    denominator = float(np.linalg.norm(lhs_flat) * np.linalg.norm(rhs_flat))
    if denominator == 0.0:
        return 1.0 if np.array_equal(lhs_flat, rhs_flat) else 0.0
    return float(np.dot(lhs_flat, rhs_flat) / denominator)


def compute_action_deviation_metrics(
    baseline_raw: np.ndarray,
    diagnosis_raw: np.ndarray,
    baseline_executed: np.ndarray,
    diagnosis_executed: np.ndarray,
) -> dict[str, Any]:
    if baseline_raw.shape != diagnosis_raw.shape:
        raise ValueError(
            f"Action shape mismatch: baseline={baseline_raw.shape}, diagnosis={diagnosis_raw.shape}."
        )
    if baseline_executed.shape != diagnosis_executed.shape:
        raise ValueError(
            "Executed action shape mismatch: "
            f"baseline={baseline_executed.shape}, diagnosis={diagnosis_executed.shape}."
        )
    continuous_baseline = baseline_executed[..., :-1]
    continuous_diagnosis = diagnosis_executed[..., :-1]
    continuous_difference = continuous_diagnosis - continuous_baseline
    normalized_difference = diagnosis_raw[..., :-1] - baseline_raw[..., :-1]
    per_horizon_l2 = np.sqrt(np.mean(np.square(continuous_difference), axis=-1))
    baseline_gripper = baseline_executed[..., -1]
    diagnosis_gripper = diagnosis_executed[..., -1]
    return {
        "continuous_action_mae": float(np.mean(np.abs(continuous_difference))),
        # Fast-WAM raw outputs are already normalized per dimension using dataset stats.
        "normalized_continuous_action_l2": float(
            np.sqrt(np.mean(np.square(normalized_difference)))
        ),
        "continuous_action_cosine_similarity": safe_cosine(
            continuous_baseline,
            continuous_diagnosis,
        ),
        "raw_gripper_difference": float(
            np.mean(np.abs(diagnosis_raw[..., -1] - baseline_raw[..., -1]))
        ),
        "post_binarization_gripper_flip_rate": float(
            np.mean(diagnosis_gripper != baseline_gripper)
        ),
        "per_action_horizon_l2_deviation": per_horizon_l2.astype(float).tolist(),
    }
