from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.asre_diagnosis.common import SALVAGE_A_PROTOCOL, sha256_file
from experiments.asre_diagnosis.salvage_a import basis as basis_module
from experiments.asre_diagnosis.salvage_a.basis import BASIS_KINDS
from experiments.asre_diagnosis.salvage_a.finalize_bases import (
    _principal_angle_summary,
    _subspace_overlap,
    _weighted_summary,
)


def _small_manifest(tmp_path: Path) -> tuple[Path, dict]:
    artifacts = {family: {"1": {}} for family in BASIS_KINDS}
    identity = torch.eye(6, dtype=torch.float32)[:, :3]
    for family in BASIS_KINDS:
        for tensor_kind in ("k", "v"):
            path = tmp_path / f"{family}_{tensor_kind}.pt"
            torch.save({"basis": identity.clone()}, path)
            artifacts[family]["1"][tensor_kind] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "shape": [6, 3],
                "dtype": "torch.float32",
            }
    payload = {
        "artifact_type": "asre_salvage_a_basis_manifest",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "feature_dim": 6,
        "ranks": [2, 3],
        "max_rank": 3,
        "late_layers": [1],
        "k_v_fitted_separately": True,
        "uncentered_svd": True,
        "actionaware_full_calibration_basis": True,
        "differentiable_path_gate_passed": True,
        "differentiable_path_report_sha256": "d" * 64,
        "random_basis_nested": True,
        "random_prefix_reused_pre_outcome": True,
        "runtime_layout": {
            "num_layers": 30,
            "video_seq_len": 98,
            "action_visible_token_count": 98,
            "action_visible_token_indices": list(range(98)),
            "feature_dim": 6,
            "num_heads": 24,
            "head_dim": 128,
        },
        "artifacts": artifacts,
    }
    path = tmp_path / "basis_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_salvage_a_three_family_runtime_loader_uses_nested_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(basis_module, "EXPECTED_FEATURE_DIM", 6)
    monkeypatch.setattr(basis_module, "LATE_LAYERS", (1,))
    monkeypatch.setattr(basis_module, "RANKS", (2, 3))
    monkeypatch.setattr(basis_module, "MAX_RANK", 3)
    basis_module._RUNTIME_CACHE.clear()
    path, _payload = _small_manifest(tmp_path)
    digest = sha256_file(path)
    for family in BASIS_KINDS:
        rank2 = basis_module.load_runtime_basis(
            path, digest, family, 2, "cpu", torch.float32
        )
        rank3 = basis_module.load_runtime_basis(
            path, digest, family, 3, "cpu", torch.float32
        )
        assert rank2.basis_kind == family
        assert rank2.inference_kwargs()["feature_projection_rank"] == 2
        torch.testing.assert_close(
            rank2.bases_by_layer[1]["k"],
            rank3.bases_by_layer[1]["k"][:, :2],
        )
    with pytest.raises(ValueError, match="rank must be one of"):
        basis_module.load_runtime_basis(
            path, digest, "actionaware", 1, "cpu", torch.float32
        )


def test_salvage_a_manifest_rejects_missing_basis_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(basis_module, "EXPECTED_FEATURE_DIM", 6)
    monkeypatch.setattr(basis_module, "LATE_LAYERS", (1,))
    monkeypatch.setattr(basis_module, "RANKS", (2, 3))
    monkeypatch.setattr(basis_module, "MAX_RANK", 3)
    path, payload = _small_manifest(tmp_path)
    del payload["artifacts"]["random"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly svd/actionaware/random"):
        basis_module.validate_basis_manifest(path)


def test_salvage_a_overlap_and_weighted_diagnostics_are_explicit() -> None:
    identity = torch.eye(4)
    orthogonal_a = identity[:, :2]
    orthogonal_b = identity[:, 2:]
    assert _subspace_overlap(orthogonal_a, orthogonal_a) == pytest.approx(1.0)
    assert _subspace_overlap(orthogonal_a, orthogonal_b) == pytest.approx(0.0)
    angles = _principal_angle_summary(orthogonal_a, orthogonal_b)
    assert angles["principal_angle_max_degrees"] == pytest.approx(90.0)

    rows = []
    for family in BASIS_KINDS:
        for rank in (36, 97):
            for kind, heldout_total, heldout_captured in (
                ("K", 1.0, 0.5),
                ("V", 3.0, 0.75),
            ):
                rows.append(
                    {
                        "family": family,
                        "rank": rank,
                        "tensor_kind": kind,
                        "heldout_total_delta_z_energy": heldout_total,
                        "heldout_captured_delta_z_energy": heldout_captured,
                        "calibration_total_action_sensitivity": heldout_total,
                        "calibration_captured_action_sensitivity": heldout_captured,
                        "aa_svd_overlap": 0.25,
                        "aa_half_stability_overlap": 0.75,
                    }
                )
    summary = _weighted_summary(rows)
    assert len(summary) == 18
    all_row = next(
        row
        for row in summary
        if row["family"] == "actionaware"
        and row["rank"] == 36
        and row["scope"] == "all"
    )
    assert all_row["basis_family"] == "actionaware"
    assert all_row["heldout_delta_z_energy_captured"] == pytest.approx(1.25 / 4.0)
    assert all_row["calibration_action_sensitivity_captured"] == pytest.approx(
        1.25 / 4.0
    )
