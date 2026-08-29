from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from experiments.asre_diagnosis.salvage_b import machinery_tests


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
