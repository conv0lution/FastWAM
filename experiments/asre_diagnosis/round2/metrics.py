"""Offline action-sensitivity metrics for the ASRE Round-2 protocol."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from experiments.asre_diagnosis.metrics import safe_cosine


CONTINUOUS_ACTION_DIMENSIONS = (
    "delta_position_x",
    "delta_position_y",
    "delta_position_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
)
ACTION_DIMENSION_LABEL_SOURCE = (
    "configs/data/libero_2cam.yaml defines eef_pose(6)+gripper(1); LIBERO uses "
    "robosuite OSC_POSE, whose first three controls are delta position and whose "
    "next three controls are delta orientation in axis-angle form"
)


def extract_action_global_std(dataset_stats: Mapping[str, Any]) -> np.ndarray:
    try:
        values = dataset_stats["action"]["default"]["global_std"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "Dataset stats must contain action.default.global_std for Round-2 metrics."
        ) from exc
    action_std = np.asarray(values, dtype=np.float64)
    if action_std.ndim != 1 or action_std.size < 7:
        raise ValueError(
            "action.default.global_std must be a one-dimensional vector with at least "
            f"seven entries, got shape {action_std.shape}."
        )
    if not np.all(np.isfinite(action_std)) or np.any(action_std <= 0):
        raise ValueError("action.default.global_std must contain finite positive values.")
    return action_std


def _validate_action_pair(lhs: np.ndarray, rhs: np.ndarray, label: str) -> None:
    if lhs.shape != rhs.shape:
        raise ValueError(f"{label} action shape mismatch: {lhs.shape} != {rhs.shape}.")
    if lhs.ndim != 2 or lhs.shape[1] < 7:
        raise ValueError(f"{label} actions must have shape [H, >=7], got {lhs.shape}.")
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise ValueError(f"{label} actions contain NaN or Inf.")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def compute_round2_metrics(
    baseline_raw: np.ndarray,
    diagnosis_raw: np.ndarray,
    baseline_executed: np.ndarray,
    diagnosis_executed: np.ndarray,
    action_std: np.ndarray,
    *,
    executed_prefix_length: int = 10,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute the pre-registered Round-2 fixed-state metrics.

    The primary metrics use denormalized physical continuous actions divided by
    the dataset-wide action standard deviation.  The separately named Round-1
    compatibility metric remains in the model's min/max-normalized raw space.
    """

    baseline_raw = np.asarray(baseline_raw)
    diagnosis_raw = np.asarray(diagnosis_raw)
    baseline_executed = np.asarray(baseline_executed)
    diagnosis_executed = np.asarray(diagnosis_executed)
    action_std = np.asarray(action_std, dtype=np.float64)
    _validate_action_pair(baseline_raw, diagnosis_raw, "raw")
    _validate_action_pair(baseline_executed, diagnosis_executed, "executed")
    if baseline_raw.shape != baseline_executed.shape:
        raise ValueError(
            "Raw and executed action shapes must agree, got "
            f"{baseline_raw.shape} and {baseline_executed.shape}."
        )
    if executed_prefix_length <= 0 or executed_prefix_length > baseline_raw.shape[0]:
        raise ValueError(
            f"executed_prefix_length must be in [1, {baseline_raw.shape[0]}], "
            f"got {executed_prefix_length}."
        )
    if action_std.ndim != 1 or action_std.size < baseline_raw.shape[1]:
        raise ValueError(
            f"action_std must cover all {baseline_raw.shape[1]} action dimensions, "
            f"got shape {action_std.shape}."
        )
    continuous_std = action_std[:6] + float(eps)
    if not np.all(np.isfinite(continuous_std)) or np.any(continuous_std <= 0):
        raise ValueError("Continuous action standard deviations must be finite and positive.")

    continuous_difference = (
        diagnosis_executed[:, :6].astype(np.float64)
        - baseline_executed[:, :6].astype(np.float64)
    )
    normalized_difference = continuous_difference / continuous_std[None, :]
    prefix = normalized_difference[:executed_prefix_length]
    raw_normalized_difference = (
        diagnosis_raw[:, :6].astype(np.float64) - baseline_raw[:, :6].astype(np.float64)
    )
    baseline_gripper = baseline_executed[:, -1]
    diagnosis_gripper = diagnosis_executed[:, -1]
    per_dimension = np.sqrt(np.mean(np.square(prefix), axis=0))
    per_horizon = np.sqrt(np.mean(np.square(normalized_difference), axis=1))

    return {
        "executed_prefix_length": int(executed_prefix_length),
        "executed_prefix_norm_rms": _rms(prefix),
        "norm_rms_h0": _rms(normalized_difference[:1]),
        "norm_rms_h0_h1": _rms(normalized_difference[:2]),
        "full_chunk_norm_rms_0_31": _rms(normalized_difference),
        "round1_raw_output_full_chunk_rms": _rms(raw_normalized_difference),
        "executed_prefix_cosine_similarity": safe_cosine(
            baseline_executed[:executed_prefix_length, :6],
            diagnosis_executed[:executed_prefix_length, :6],
        ),
        "executed_prefix_gripper_flip_rate": float(
            np.mean(
                diagnosis_gripper[:executed_prefix_length]
                != baseline_gripper[:executed_prefix_length]
            )
        ),
        "full_horizon_gripper_flip_rate": float(
            np.mean(diagnosis_gripper != baseline_gripper)
        ),
        "executed_prefix_norm_rms_by_dimension": per_dimension.astype(float).tolist(),
        "translation_norm_rms": _rms(prefix[:, :3]),
        "rotation_norm_rms": _rms(prefix[:, 3:6]),
        "per_horizon_norm_rms": per_horizon.astype(float).tolist(),
    }
