from __future__ import annotations

import pytest
import torch

from fastwam.models.wan22.video_cache_replacement import (
    action_visible_video_token_indices,
    mix_replacement_video_cache,
)


def _cache(*, layers: int = 3, tokens: int = 4, width: int = 8):
    current_k = [
        torch.arange(tokens * width, dtype=torch.float32).reshape(1, tokens, width)
        + layer * 100
        for layer in range(layers)
    ]
    current_v = [tensor + 0.25 for tensor in current_k]
    donor_k = [tensor + 1_000 for tensor in current_k]
    donor_v = [tensor + 1_000 for tensor in current_v]
    return current_k, current_v, donor_k, donor_v


def test_action_visible_positions_come_from_runtime_attention_mask() -> None:
    mask = torch.tensor(
        [
            [True, False, False, False, True],
            [False, False, True, False, True],
        ]
    )
    assert action_visible_video_token_indices(mask, video_seq_len=4) == (0, 2)


def test_token_mask_changes_only_unretained_visible_positions_and_uses_same_kv_mask() -> None:
    current_k, current_v, donor_k, donor_v = _cache()
    mixed_k, mixed_v, audit = mix_replacement_video_cache(
        current_cache_k=current_k,
        current_cache_v=current_v,
        replacement_cache_k=donor_k,
        replacement_cache_v=donor_v,
        replacement_video_layers=(1, 2),
        action_visible_token_indices=(0, 1, 2),
        retained_current_token_indices=(0, 2),
        num_layers=3,
    )

    assert mixed_k[0] is current_k[0]
    assert mixed_v[0] is current_v[0]
    for layer in (1, 2):
        torch.testing.assert_close(mixed_k[layer][:, 0], current_k[layer][:, 0])
        torch.testing.assert_close(mixed_k[layer][:, 1], donor_k[layer][:, 1])
        torch.testing.assert_close(mixed_k[layer][:, 2], current_k[layer][:, 2])
        # Token 3 is not action-visible and must never be intervened on.
        torch.testing.assert_close(mixed_k[layer][:, 3], current_k[layer][:, 3])
        torch.testing.assert_close(mixed_v[layer][:, 1], donor_v[layer][:, 1])
        torch.testing.assert_close(mixed_v[layer][:, 3], current_v[layer][:, 3])
    assert audit["mode"] == "token"
    assert audit["same_mask_for_k_and_v"] is True
    assert audit["replacement_token_indices"] == [1]


def test_head_mask_is_independent_per_layer_and_preserves_geometry() -> None:
    current_k, current_v, donor_k, donor_v = _cache()
    retained = {1: (0, 2), 2: (1, 3)}
    mixed_k, mixed_v, audit = mix_replacement_video_cache(
        current_cache_k=current_k,
        current_cache_v=current_v,
        replacement_cache_k=donor_k,
        replacement_cache_v=donor_v,
        replacement_video_layers=(1, 2),
        action_visible_token_indices=(0, 1, 2),
        retained_current_heads_by_layer=retained,
        num_heads=4,
        head_dim=2,
        num_layers=3,
    )

    for layer, retained_heads in retained.items():
        mixed_k_view = mixed_k[layer].reshape(1, 4, 4, 2)
        mixed_v_view = mixed_v[layer].reshape(1, 4, 4, 2)
        current_k_view = current_k[layer].reshape(1, 4, 4, 2)
        current_v_view = current_v[layer].reshape(1, 4, 4, 2)
        donor_k_view = donor_k[layer].reshape(1, 4, 4, 2)
        donor_v_view = donor_v[layer].reshape(1, 4, 4, 2)
        for head in range(4):
            expected_k = current_k_view if head in retained_heads else donor_k_view
            expected_v = current_v_view if head in retained_heads else donor_v_view
            torch.testing.assert_close(
                mixed_k_view[:, :3, head], expected_k[:, :3, head]
            )
            torch.testing.assert_close(
                mixed_v_view[:, :3, head], expected_v[:, :3, head]
            )
        # Non-visible positions stay current for every head.
        torch.testing.assert_close(mixed_k_view[:, 3], current_k_view[:, 3])
        torch.testing.assert_close(mixed_v_view[:, 3], current_v_view[:, 3])
        assert mixed_k[layer].shape == current_k[layer].shape
        assert mixed_k[layer].dtype == current_k[layer].dtype
        assert mixed_k[layer].device == current_k[layer].device
    assert audit["mode"] == "head"
    assert audit["num_heads"] == 4
    assert audit["head_dim"] == 2


@pytest.mark.parametrize("mode", ["token", "head"])
def test_arbitrary_mask_self_replacement_is_bit_exact(mode: str) -> None:
    current_k, current_v, _, _ = _cache()
    kwargs = (
        {"retained_current_token_indices": (0, 2)}
        if mode == "token"
        else {
            "retained_current_heads_by_layer": {1: (0, 2), 2: (1, 3)},
            "num_heads": 4,
            "head_dim": 2,
        }
    )
    mixed_k, mixed_v, _ = mix_replacement_video_cache(
        current_cache_k=current_k,
        current_cache_v=current_v,
        replacement_cache_k=[tensor.clone() for tensor in current_k],
        replacement_cache_v=[tensor.clone() for tensor in current_v],
        replacement_video_layers=(1, 2),
        action_visible_token_indices=(0, 1, 2, 3),
        num_layers=3,
        **kwargs,
    )
    for layer in range(3):
        assert torch.equal(mixed_k[layer], current_k[layer])
        assert torch.equal(mixed_v[layer], current_v[layer])


def test_token_endpoints_reproduce_current_and_wrong_cache_exactly() -> None:
    current_k, current_v, donor_k, donor_v = _cache()
    for retained, expected_k, expected_v in (
        ((0, 1, 2, 3), current_k, current_v),
        ((), donor_k, donor_v),
    ):
        mixed_k, mixed_v, _ = mix_replacement_video_cache(
            current_cache_k=current_k,
            current_cache_v=current_v,
            replacement_cache_k=donor_k,
            replacement_cache_v=donor_v,
            replacement_video_layers=(0, 1, 2),
            action_visible_token_indices=(0, 1, 2, 3),
            retained_current_token_indices=retained,
            num_layers=3,
        )
        for layer in range(3):
            assert torch.equal(mixed_k[layer], expected_k[layer])
            assert torch.equal(mixed_v[layer], expected_v[layer])


def test_head_endpoints_reproduce_current_and_wrong_cache_exactly() -> None:
    current_k, current_v, donor_k, donor_v = _cache()
    for retained_heads, expected_k, expected_v in (
        ((0, 1, 2, 3), current_k, current_v),
        ((), donor_k, donor_v),
    ):
        mixed_k, mixed_v, _ = mix_replacement_video_cache(
            current_cache_k=current_k,
            current_cache_v=current_v,
            replacement_cache_k=donor_k,
            replacement_cache_v=donor_v,
            replacement_video_layers=(0, 1, 2),
            action_visible_token_indices=(0, 1, 2, 3),
            retained_current_heads_by_layer={
                layer: retained_heads for layer in range(3)
            },
            num_heads=4,
            head_dim=2,
            num_layers=3,
        )
        for layer in range(3):
            assert torch.equal(mixed_k[layer], expected_k[layer])
            assert torch.equal(mixed_v[layer], expected_v[layer])


def test_head_mask_must_cover_exactly_the_replacement_layers() -> None:
    current_k, current_v, donor_k, donor_v = _cache()
    with pytest.raises(ValueError, match="exactly every replacement layer"):
        mix_replacement_video_cache(
            current_cache_k=current_k,
            current_cache_v=current_v,
            replacement_cache_k=donor_k,
            replacement_cache_v=donor_v,
            replacement_video_layers=(1, 2),
            action_visible_token_indices=(0, 1),
            retained_current_heads_by_layer={1: (0, 1)},
            num_heads=4,
            head_dim=2,
            num_layers=3,
        )
