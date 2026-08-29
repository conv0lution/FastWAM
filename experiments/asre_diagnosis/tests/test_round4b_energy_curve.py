from __future__ import annotations

import numpy as np
import pytest

from experiments.asre_diagnosis.round4b.basis import EXPECTED_FEATURE_DIM
from experiments.asre_diagnosis.round4b.energy_curve_analysis import (
    _curves,
    candidate_ranks,
    cumulative_curve,
    minimum_rank,
    validate_curve,
)
from experiments.asre_diagnosis.round4b.run_energy_curve_analysis import (
    _reuse_stable_preflight,
)


def test_cumulative_curve_covers_rank_zero_through_feature_dim() -> None:
    energy = np.arange(EXPECTED_FEATURE_DIM, 0, -1, dtype=np.float64)
    curve = cumulative_curve(energy, float(energy.sum()))
    validate_curve(curve, label="test", rank_d_tolerance=1e-12)
    assert curve.shape == (EXPECTED_FEATURE_DIM + 1,)
    assert curve[0] == 0.0
    assert curve[-1] == pytest.approx(1.0)
    rank = minimum_rank(curve, 0.5)
    assert curve[rank] >= 0.5
    assert curve[rank - 1] < 0.5


def test_global_curves_are_absolute_energy_weighted_not_matrix_means() -> None:
    matrices = {}
    for layer in range(15, 30):
        for kind in ("k", "v"):
            matrix = f"layer{layer:02d}_{kind}"
            weight = 100.0 if matrix == "layer15_k" else 1.0
            calibration = np.zeros(EXPECTED_FEATURE_DIM, dtype=np.float64)
            heldout = np.zeros(EXPECTED_FEATURE_DIM, dtype=np.float64)
            calibration[0 if weight == 100.0 else 1] = weight
            heldout[0 if weight == 100.0 else 1] = weight
            matrices[matrix] = {
                "tensor_kind": kind,
                "calibration_coordinate_energy": calibration,
                "heldout_coordinate_energy": heldout,
                "calibration_total_energy": weight,
                "heldout_total_energy": weight,
            }
    matrix_curves, global_curves = _curves(matrices)
    assert len(matrix_curves) == 30
    expected_rank_one = 100.0 / 129.0
    assert global_curves["calibration"]["all"][1] == pytest.approx(expected_rank_one)
    assert global_curves["heldout"]["all"][2] == pytest.approx(1.0)


def test_candidate_file_payload_has_exact_three_energy_defined_ranks() -> None:
    ranks = np.arange(EXPECTED_FEATURE_DIM + 1, dtype=np.float64)
    linear = ranks / EXPECTED_FEATURE_DIM
    curves = {
        "calibration": {"all": linear, "K": linear, "V": linear},
        "heldout": {"all": linear, "K": linear, "V": linear},
    }
    payload = candidate_ranks(curves)
    assert set(payload) == {"candidate_r50", "candidate_r70", "candidate_r80"}
    assert payload["candidate_r50"]["rank"] == 1536
    assert payload["candidate_r70"]["rank"] == 2151
    assert payload["candidate_r80"]["rank"] == 2458
    assert all("online_success" not in value for value in payload.values())


def test_curve_checks_fail_closed_on_nonmonotonic_or_incomplete_data() -> None:
    curve = np.linspace(0.0, 1.0, EXPECTED_FEATURE_DIM + 1)
    broken = curve.copy()
    broken[100] = broken[99] - 0.1
    with pytest.raises(ValueError, match="not monotonic"):
        validate_curve(broken, label="broken", rank_d_tolerance=1e-6)
    with pytest.raises(ValueError, match="length 3072"):
        cumulative_curve(np.ones(10), 10.0)


def test_resume_preflight_ignores_only_the_dynamic_creation_timestamp() -> None:
    existing = {
        "artifact_type": "preflight",
        "created_at": "2026-08-29T01:00:00+08:00",
        "analysis_commit": "abc",
        "split_sha256": "1" * 64,
    }
    regenerated = dict(existing, created_at="2026-08-29T02:00:00+08:00")
    assert _reuse_stable_preflight(existing, regenerated) == existing
    changed = dict(regenerated, split_sha256="2" * 64)
    with pytest.raises(RuntimeError, match="split_sha256"):
        _reuse_stable_preflight(existing, changed)
