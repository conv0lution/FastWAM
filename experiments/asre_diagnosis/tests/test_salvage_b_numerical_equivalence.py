from __future__ import annotations

import pytest
import torch

from experiments.asre_diagnosis.salvage_b import numerical_equivalence as numeq


def test_tensor_error_metrics_reports_registered_quantities() -> None:
    reference = torch.tensor([3.0, 4.0])
    candidate = torch.tensor([0.0, 4.0])
    result = numeq.tensor_error_metrics(reference, candidate)

    assert result["shape_equal"] is True
    assert result["finite"] is True
    assert result["max_abs_error"] == pytest.approx(3.0)
    assert result["rms_absolute_error"] == pytest.approx((9.0 / 2.0) ** 0.5)
    assert result["reference_output_norm"] == pytest.approx(5.0)
    assert result["candidate_output_norm"] == pytest.approx(4.0)
    assert result["relative_norm_error"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("bf16", "fp32", "structural", "jumps", "expected"),
    [
        (8.0e-3, 5.0e-4, True, [], numeq.CONFIRMED),
        (8.0e-3, 3.0e-3, True, [], numeq.UNCERTAIN),
        (8.0e-3, 6.0e-3, True, [], numeq.POSSIBLE_MISMATCH),
        (8.0e-3, 5.0e-4, False, [], numeq.POSSIBLE_MISMATCH),
        (
            8.0e-3,
            5.0e-4,
            True,
            [{"layer": 17, "jump_factor": 12.0}],
            numeq.POSSIBLE_MISMATCH,
        ),
    ],
)
def test_numerical_classification_is_fail_closed(
    bf16: float,
    fp32: float,
    structural: bool,
    jumps: list[dict[str, float]],
    expected: str,
) -> None:
    result = numeq.classify_numerical_equivalence(
        bf16_prediction_relative_rmse=bf16,
        fp32_prediction_relative_rmse=fp32,
        structural_passed=structural,
        fp32_sudden_jumps=jumps,
    )
    assert result["classification"] == expected


def test_mask_audit_compares_relations_not_raw_shape() -> None:
    prefix = 98
    future = 196
    action = 32
    video = torch.ones((prefix + future, prefix + future), dtype=torch.bool)
    video[:prefix, prefix:] = False
    full = torch.zeros(
        (prefix + future + action, prefix + future + action), dtype=torch.bool
    )
    full[: prefix + future, : prefix + future] = video
    full[prefix + future :, :] = True

    result = numeq._mask_audit(
        {"full_joint_mask": full, "video_mask": video}
    )
    assert result["passed"] is True
    assert result["no_extra_visible_token"] is True
    assert result["shapes"]["stock_joint"] == [326, 326]
    assert result["shapes"]["factorized_future"] == [196, 294]
    assert result["allowed_entries"][
        "stock_action_columns_visible_to_video_queries"
    ] == 0

    full[100, -1] = True
    failed = numeq._mask_audit(
        {"full_joint_mask": full, "video_mask": video}
    )
    assert failed["passed"] is False


def test_sudden_jump_requires_order_of_magnitude_and_meaningful_size() -> None:
    rows = []
    for layer, value in enumerate((1.0e-5, 2.0e-5, 6.0e-4)):
        rows.append(
            {
                "dtype": "torch.float32",
                "layer": layer,
                "tensor": "future_token_hidden_output",
                "relative_rmse": value,
            }
        )
    jumps = numeq._sudden_jumps(rows, "torch.float32")
    assert jumps == [
        {
            "tensor": "future_token_hidden_output",
            "previous_layer": 1,
            "layer": 2,
            "previous_relative_rmse": 2.0e-5,
            "relative_rmse": 6.0e-4,
            "jump_factor": pytest.approx(30.0),
        }
    ]
