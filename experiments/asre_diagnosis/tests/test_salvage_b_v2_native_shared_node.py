from __future__ import annotations

import inspect

import torch

from experiments.asre_diagnosis.common import build_salvage_b_v2_conditions
from experiments.asre_diagnosis.salvage_b_v2.classification import classify
from experiments.asre_diagnosis.salvage_b_v2.definitions import FEATURE_DIM, LATE_LAYERS
from experiments.asre_diagnosis.salvage_b_v2.native_clamp import (
    FrozenPrefixTrajectory,
    NativePrefixClamp,
)
from experiments.asre_diagnosis.salvage_b_v2.native_runtime import _native_call
from fastwam.models.wan22.mot import MoT


def _trajectory(value: float, *, steps: int = 1) -> FrozenPrefixTrajectory:
    trajectory = FrozenPrefixTrajectory()
    for step in range(steps):
        for layer in range(30):
            tensor = torch.full((1, 98, FEATURE_DIM), value + layer / 100.0)
            trajectory.capture_hook(layer, tensor, tensor + 1.0, 98)
    trajectory.validate_complete(expected_steps=steps)
    return trajectory


def test_frozen_capture_is_observational_and_complete() -> None:
    capture = FrozenPrefixTrajectory()
    for layer in range(30):
        key = torch.randn(1, 98, FEATURE_DIM)
        value = torch.randn_like(key)
        returned_k, returned_v = capture.capture_hook(layer, key, value, 98)
        assert returned_k is key
        assert returned_v is value
    capture.validate_complete(expected_steps=1)
    assert len(capture.values) == 30


def test_rank_endpoints_and_prefix_only_clamp() -> None:
    current = _trajectory(4.0)
    wrong = _trajectory(-3.0)
    for rank, target in ((0, wrong), (FEATURE_DIM, current)):
        clamp = NativePrefixClamp(current=current, wrong=wrong, rank=rank)
        for layer in range(30):
            key = torch.randn(1, 103, FEATURE_DIM)
            value = torch.randn_like(key)
            original_k = key.clone()
            original_v = value.clone()
            out_k, out_v = clamp.hook(layer, key, value, 98)
            if layer < 15:
                assert out_k is key and out_v is value
            else:
                torch.testing.assert_close(out_k[:, :98], target.values[(0, layer, "k")])
                torch.testing.assert_close(out_v[:, :98], target.values[(0, layer, "v")])
                torch.testing.assert_close(out_k[:, 98:], original_k[:, 98:])
                torch.testing.assert_close(out_v[:, 98:], original_v[:, 98:])
        clamp.validate_complete(expected_steps=1)


def test_projected_target_uses_independent_frozen_k_v_bases() -> None:
    current = _trajectory(2.0)
    wrong = _trajectory(0.0)
    bases = {}
    for layer in LATE_LAYERS:
        k_basis = torch.zeros(FEATURE_DIM, 1)
        v_basis = torch.zeros(FEATURE_DIM, 1)
        k_basis[0, 0] = 1
        v_basis[1, 0] = 1
        bases[layer] = {"k": k_basis, "v": v_basis}
    clamp = NativePrefixClamp(current=current, wrong=wrong, rank=1, bases_by_layer=bases)
    for layer in range(30):
        key = torch.zeros(1, 100, FEATURE_DIM)
        value = torch.zeros_like(key)
        out_k, out_v = clamp.hook(layer, key, value, 98)
        if layer in LATE_LAYERS:
            torch.testing.assert_close(
                out_k[:, :98, 0], current.values[(0, layer, "k")][:, :, 0]
            )
            torch.testing.assert_close(
                out_k[:, :98, 1:], wrong.values[(0, layer, "k")][:, :, 1:]
            )
            torch.testing.assert_close(
                out_v[:, :98, 1], current.values[(0, layer, "v")][:, :, 1]
            )
    clamp.validate_complete(expected_steps=1)


def test_registered_conditions_touch_no_early_layer() -> None:
    conditions = build_salvage_b_v2_conditions(30)
    assert [condition.name for condition in conditions] == [
        "current",
        "wrong",
        "svd_r97",
        "svd_r170",
    ]
    assert all(condition.disabled_video_layers == () for condition in conditions)
    assert conditions[0].replacement_video_layers == ()
    assert all(c.replacement_video_layers == LATE_LAYERS for c in conditions[1:])


def test_runtime_cannot_call_factorized_future_helper() -> None:
    source = inspect.getsource(_native_call)
    assert "infer_joint" in source
    assert "forward_future_video_with_video_cache_tensor" not in source


def test_classification_rules_and_technical_priority() -> None:
    machinery = {
        "native_clamp_identity_passed": True,
        "shared_node_reach_passed": True,
    }
    result = classify(
        machinery=machinery,
        action_success={"current": 0.95, "wrong": 0.0, "svd_r97": 0.79, "svd_r170": 0.95},
        world_loss={"current": 0.4, "wrong": 0.8, "svd_r97": 0.65, "svd_r170": 0.6},
        action_endpoint_valid=True,
        world_endpoint_informative=True,
        r170_world_minus_current_ci=(0.1, 0.3),
    )
    assert result["classification"] == "STRONG"
    failed = classify(
        machinery={"native_clamp_identity_passed": False, "shared_node_reach_passed": True},
        action_success={},
        world_loss={},
        action_endpoint_valid=True,
        world_endpoint_informative=True,
        r170_world_minus_current_ci=(0.1, 0.3),
    )
    assert failed["classification"] == "NATIVE-CLAMP-IDENTITY-FAILED"


def test_mot_hook_is_inside_single_native_joint_layer() -> None:
    source = inspect.getsource(MoT._forward_joint_layer)
    assert source.count("flash_attention(") == 1
    assert source.index("native_prefix_kv_hook(") < source.index("flash_attention(")
    assert "torch.cat([q_video, q_action]" in source
    assert "torch.cat([k_video, k_action]" in source
    assert "torch.cat([v_video, v_action]" in source
