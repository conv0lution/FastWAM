from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = REPO_ROOT.parent / "LIBERO"
if LIBERO_ROOT.is_dir() and str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from experiments.asre_diagnosis.common import SALVAGE_A_PROTOCOL, sha256_file
from experiments.asre_diagnosis.tests.test_round3b_model_replacement import (
    _TinyFastWAM,
    _infer_fastwam_kwargs,
)
from experiments.libero import eval_libero_single as evaluator


def _cfg(diagnosis: dict[str, object]) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 42,
            "EVALUATION": {
                "task_suite_name": "libero_spatial",
                "num_trials": 10,
                "visualize_future_video": False,
                "compile_action_infer": False,
            },
            "ASRE_DIAGNOSIS": diagnosis,
        }
    )


def test_salvage_a_loads_split_local_donor_bundle(tmp_path: Path) -> None:
    mapping = tmp_path / "donor_mapping.json"
    manifest = tmp_path / "donor_observation_manifest.json"
    observation_root = tmp_path / "observations"
    mapping.write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    observation_root.mkdir()
    fake_bundle = SimpleNamespace(
        mapping_payload={
            "task_suite": "libero_spatial",
            "seed": 42,
            "num_trials": 10,
            "mapping_rule": "split_local_deterministic_derangement",
        },
        mappings={(0, trial): (0, (trial + 1) % 10) for trial in range(10)},
    )
    cfg = _cfg(
        {
            "enabled": True,
            "protocol": SALVAGE_A_PROTOCOL,
            "donor_mapping_path": str(mapping),
            "donor_mapping_sha256": "a" * 64,
            "donor_observation_manifest_path": str(manifest),
            "donor_observation_manifest_sha256": "b" * 64,
            "donor_observation_root": str(observation_root),
        }
    )

    with patch.object(
        evaluator.SalvageADonorBundle, "load", return_value=fake_bundle
    ) as loader:
        assert evaluator._load_donor_bundle(cfg, task_ids=[0]) is fake_bundle

    assert loader.call_args.kwargs["mapping_path"] == mapping.resolve()
    assert loader.call_args.kwargs["observation_manifest_path"] == manifest.resolve()
    assert loader.call_args.kwargs["observation_root"] == observation_root.resolve()


def test_salvage_a_actionaware_basis_resolves_on_model_device(tmp_path: Path) -> None:
    basis_manifest = tmp_path / "basis_manifest.json"
    basis_manifest.write_text("{}\n", encoding="utf-8")
    cfg = _cfg(
        {
            "protocol": SALVAGE_A_PROTOCOL,
            "subspace_basis_manifest_path": str(basis_manifest),
            "subspace_basis_manifest_sha256": "c" * 64,
            "subspace_basis_kind": "actionaware",
            "subspace_rank": 36,
        }
    )
    model = nn.Linear(2, 2, dtype=torch.float64)
    sentinel = object()

    with patch.object(
        evaluator, "validate_salvage_a_basis_manifest"
    ) as validator, patch.object(
        evaluator, "load_salvage_a_runtime_basis", return_value=sentinel
    ) as loader:
        assert evaluator._load_round4b_basis_spec(cfg, model) is sentinel

    validator.assert_called_once_with(
        basis_manifest.resolve(), expected_sha256="c" * 64, verify_files=False
    )
    assert loader.call_args.kwargs == {
        "manifest_path": basis_manifest.resolve(),
        "expected_sha256": "c" * 64,
        "basis_kind": "actionaware",
        "rank": 36,
        "device": model.weight.device,
        "dtype": model.weight.dtype,
    }


def _provenance_cfg(tmp_path: Path) -> OmegaConf:
    diagnosis: dict[str, object] = {"protocol": SALVAGE_A_PROTOCOL}
    for stem in evaluator._SALVAGE_A_PROVENANCE_STEMS:
        path = tmp_path / f"{stem}.json"
        path.write_text(f'{{"artifact": "{stem}"}}\n', encoding="utf-8")
        diagnosis[f"{stem}_path"] = str(path)
        diagnosis[f"{stem}_sha256"] = sha256_file(path)
    return _cfg(diagnosis)


def test_salvage_a_provenance_is_hashed_normalized_and_resume_locked(
    tmp_path: Path,
) -> None:
    cfg = _provenance_cfg(tmp_path)
    with patch.object(evaluator, "validate_salvage_a_basis_manifest") as validator:
        payload = evaluator._resolve_salvage_a_run_provenance(cfg)

    assert set(payload) == set(evaluator._SALVAGE_A_PROVENANCE_KEYS)
    assert all(Path(payload[key]).is_absolute() for key in payload if key.endswith("_path"))
    validator.assert_called_once_with(
        Path(payload["subspace_basis_manifest_path"]),
        expected_sha256=payload["subspace_basis_manifest_sha256"],
        verify_files=False,
    )
    assert set(evaluator._salvage_a_resume_keys()) == {
        "subspace_basis_kind",
        "subspace_rank",
        *evaluator._SALVAGE_A_PROVENANCE_KEYS,
    }
    per_task = evaluator._salvage_a_result_metadata(cfg.ASRE_DIAGNOSIS)
    assert per_task["differentiable_path_report_sha256"] == payload[
        "differentiable_path_report_sha256"
    ]
    assert per_task["state_selection_manifest_sha256"] == payload[
        "state_selection_manifest_sha256"
    ]

    cfg.ASRE_DIAGNOSIS.machinery_report_sha256 = "0" * 64
    with pytest.raises(ValueError, match="artifact SHA256 mismatch"):
        evaluator._resolve_salvage_a_run_provenance(cfg)


class _ProjectionAwareRuntimeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.received: dict[str, object] | None = None

    def infer_action(
        self,
        *,
        disabled_video_layers=(),
        replacement_input_image=None,
        replacement_video_layers=(),
        feature_projection_bases_by_layer=None,
        feature_projection_rank=None,
        expected_video_cache_layout=None,
        compile_action_infer=False,
        **kwargs,
    ):
        self.received = {
            "disabled_video_layers": tuple(disabled_video_layers),
            "replacement_input_image": replacement_input_image,
            "replacement_video_layers": tuple(replacement_video_layers),
            "feature_projection_bases_by_layer": feature_projection_bases_by_layer,
            "feature_projection_rank": feature_projection_rank,
            "expected_video_cache_layout": expected_video_cache_layout,
            "compile_action_infer": compile_action_infer,
            "kwargs": kwargs,
        }
        return {"action": torch.zeros((1, 32, 7))}


def test_salvage_a_runtime_forwards_actionaware_projection() -> None:
    cfg = _cfg(
        {
            "enabled": True,
            "protocol": SALVAGE_A_PROTOCOL,
            "disabled_video_layers": list(range(15)),
            "replacement_video_layers": list(range(15, 30)),
        }
    )
    model = _ProjectionAwareRuntimeModel()
    donor = torch.ones((1, 3, 2, 2))
    bases = {15: {"k": torch.ones((2, 1)), "v": torch.ones((2, 1))}}
    basis_spec = SimpleNamespace(
        inference_kwargs=lambda: {
            "feature_projection_bases_by_layer": bases,
            "feature_projection_rank": 36,
            "expected_video_cache_layout": {"video_seq_len": 98},
        }
    )

    with patch.object(evaluator, "_load_round4b_basis_spec", return_value=basis_spec):
        action, future = evaluator._run_prepared_action_inference(
            model, cfg, {}, replacement_input_image=donor
        )

    assert list(action.shape) == [1, 32, 7]
    assert future is None
    assert model.received is not None
    assert model.received["feature_projection_bases_by_layer"] is bases
    assert model.received["feature_projection_rank"] == 36
    assert model.received["replacement_video_layers"] == tuple(range(15, 30))
    assert model.received["replacement_input_image"] is donor


class _DifferentiableTinyFastWAM(_TinyFastWAM):
    def __init__(self) -> None:
        super().__init__()
        self.frozen_gain = nn.Parameter(torch.tensor(0.01), requires_grad=False)

    def _denoise_action_with_video_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        disabled_video_layers: tuple[int, ...] = (),
    ) -> torch.Tensor:
        del timestep_action, context, context_mask, action_attention_mask
        enabled = [
            cache
            for layer, cache in enumerate(video_cache_k)
            if layer not in disabled_video_layers
        ] + [
            cache
            for layer, cache in enumerate(video_cache_v)
            if layer not in disabled_video_layers
        ]
        cache_signal = torch.stack([cache.mean() for cache in enabled]).sum()
        return latents_action * 0.05 + self.frozen_gain * cache_signal


def test_full_ten_step_action_sensitive_path_is_equivalent_and_differentiable() -> None:
    current = torch.zeros((1, 3, 16, 16))
    wrong = torch.ones_like(current)
    baseline_model = _DifferentiableTinyFastWAM()
    gradient_model = _DifferentiableTinyFastWAM()
    kwargs = _infer_fastwam_kwargs(current)
    kwargs.update(num_inference_steps=10, seed=42)

    baseline = baseline_model.infer_action(**kwargs)
    differentiable = gradient_model.infer_action(
        **kwargs,
        replacement_input_image=wrong,
        replacement_video_layers=(0, 1, 2),
        action_sensitive_interpolation_lambda=1.0,
        action_sensitive_layers=(0, 1, 2),
    )

    torch.testing.assert_close(
        differentiable["action"].detach(), baseline["action"], atol=1.0e-7, rtol=0
    )
    assert differentiable["action"].requires_grad
    assert differentiable["action_sensitive_inference"]["num_inference_steps"] == 10
    leaves = [
        differentiable["action_sensitive_cache_tensors"][kind][layer]
        for kind in ("k", "v")
        for layer in (0, 1, 2)
    ]
    gradients = torch.autograd.grad(differentiable["action"].sum(), leaves)
    assert all(leaf.is_leaf and leaf.requires_grad for leaf in leaves)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    assert all(not parameter.requires_grad for parameter in gradient_model.parameters())
    assert all(parameter.grad is None for parameter in gradient_model.parameters())


def test_sim_libero_declares_salvage_a_provenance_fields() -> None:
    config = OmegaConf.load(REPO_ROOT / "configs" / "sim_libero.yaml")
    for key in evaluator._SALVAGE_A_PROVENANCE_KEYS:
        assert key in config.ASRE_DIAGNOSIS
        assert config.ASRE_DIAGNOSIS.get(key) is None
