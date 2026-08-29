from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from experiments.asre_diagnosis.salvage_b import machinery_tests


def test_factorization_budget_is_derived_from_execution_dtype() -> None:
    bf16 = machinery_tests._factorization_relative_rmse_budget(torch.bfloat16)
    assert bf16["machine_epsilon"] == pytest.approx(0.0078125)
    assert bf16["relative_rmse_max"] == pytest.approx(0.015625)
    assert bf16["outcome_independent"] is True

    fp32 = machinery_tests._factorization_relative_rmse_budget(torch.float32)
    assert fp32["relative_rmse_max"] == pytest.approx(
        machinery_tests.FACTORIZATION_RELATIVE_RMSE_FLOOR
    )

    observed_bf16_partition = {
        "passed": False,
        "shape_equal": True,
        "finite": True,
        "torch_allclose": False,
        "relative_rmse": 0.0082,
    }
    assert machinery_tests._factorization_comparison_passes(
        observed_bf16_partition,
        relative_rmse_max=bf16["relative_rmse_max"],
    )
    assert not machinery_tests._factorization_comparison_passes(
        {**observed_bf16_partition, "relative_rmse": 0.02},
        relative_rmse_max=bf16["relative_rmse_max"],
    )


def test_manual_native_metric_is_terminal_future_latent_mse() -> None:
    target = torch.tensor(
        [[[[[99.0]], [[1.0]], [[3.0]]], [[[99.0]], [[2.0]], [[4.0]]]]]
    )
    prediction = torch.tensor(
        [[[[[0.0]], [[1.0]]], [[[1.0]], [[2.0]]]]]
    )
    prepared = SimpleNamespace(input_latents=target)

    result = machinery_tests._manual_reconstruction_metric(
        prepared=prepared, prediction=prediction
    )
    expected = torch.nn.functional.mse_loss(
        prediction.float(), target[:, :, 1:].float(), reduction="mean"
    ).item()

    assert result == {
        "native_world_loss": expected,
        "future_latent_mse": expected,
    }


def test_machinery_report_json_is_last_completion_sentinel(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "machinery_report.json"
    output_md = tmp_path / "machinery_report.md"
    report = {
        "passed": True,
        "phase_b_authorized": True,
        "checks": {"example": {"passed": True}},
    }
    original_atomic_json = machinery_tests.atomic_write_json

    def fail_json(*_args, **_kwargs):
        raise RuntimeError("simulated interruption before JSON publication")

    monkeypatch.setattr(machinery_tests, "atomic_write_json", fail_json)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        machinery_tests._publish_report_bundle(
            output=output, output_md=output_md, report=report
        )
    assert output_md.is_file()
    assert not output.exists()

    monkeypatch.setattr(
        machinery_tests, "atomic_write_json", original_atomic_json
    )
    machinery_tests._publish_report_bundle(
        output=output, output_md=output_md, report=report
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert "Status: **PASSED**" in output_md.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="completed"):
        machinery_tests._publish_report_bundle(
            output=output, output_md=output_md, report=report
        )
