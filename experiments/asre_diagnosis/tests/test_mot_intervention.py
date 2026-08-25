from __future__ import annotations

import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from fastwam.models.wan22 import mot as mot_module
from fastwam.models.wan22.mot import MoT


class _FakeMoT(MoT):
    def __init__(self, num_layers: int):
        nn.Module.__init__(self)
        self.mixtures = {"action": types.SimpleNamespace(blocks=[object()] * num_layers)}
        self.num_layers = num_layers
        self.num_heads = 1

    def _build_expert_attention_io(self, expert, block, x, freqs, t_mod):
        del expert, block, freqs, t_mod
        return x, x + 1.0, x + 2.0, x, x, x, x, x, False

    def _apply_expert_post_block_tensor(
        self,
        block,
        residual_x,
        mixed_attn_out,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        context,
        context_mask,
    ):
        del block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, context, context_mask
        return mixed_attn_out


class MoTInterventionTest(unittest.TestCase):
    def test_disabled_layer_removes_video_kv_and_slices_mask(self) -> None:
        model = _FakeMoT(num_layers=2)
        calls = []

        def fake_attention(q, k, v, num_heads, ctx_mask):
            del num_heads
            calls.append((q.shape[1], k.shape[1], v.shape[1], tuple(ctx_mask.shape)))
            return q

        action_tokens = torch.zeros((1, 2, 1))
        video_cache = [torch.zeros((1, 3, 1)) for _ in range(2)]
        attention_mask = torch.ones((2, 5), dtype=torch.bool)
        original_attention_mask = attention_mask.clone()
        with mock.patch.object(mot_module, "flash_attention", side_effect=fake_attention):
            model.forward_action_with_video_cache_tensor(
                action_tokens=action_tokens,
                action_freqs=torch.empty(0),
                action_t_mod=torch.empty(0),
                action_context=torch.empty(0),
                action_context_mask=torch.empty(0),
                video_cache_k=video_cache,
                video_cache_v=video_cache,
                action_attention_mask=attention_mask,
                disabled_video_layers=(1,),
            )

        self.assertEqual(calls[0], (2, 5, 5, (2, 5)))
        self.assertEqual(calls[1], (2, 2, 2, (2, 2)))
        self.assertTrue(torch.equal(attention_mask, original_attention_mask))

    def test_empty_disabled_layers_preserve_original_attention_shapes(self) -> None:
        model = _FakeMoT(num_layers=2)
        key_lengths = []

        def fake_attention(q, k, v, num_heads, ctx_mask):
            del v, num_heads, ctx_mask
            key_lengths.append(k.shape[1])
            return q

        action_tokens = torch.zeros((1, 2, 1))
        video_cache = [torch.zeros((1, 3, 1)) for _ in range(2)]
        with mock.patch.object(mot_module, "flash_attention", side_effect=fake_attention):
            model.forward_action_with_video_cache_tensor(
                action_tokens=action_tokens,
                action_freqs=torch.empty(0),
                action_t_mod=torch.empty(0),
                action_context=torch.empty(0),
                action_context_mask=torch.empty(0),
                video_cache_k=video_cache,
                video_cache_v=video_cache,
                action_attention_mask=torch.ones((2, 5), dtype=torch.bool),
                disabled_video_layers=(),
            )
        self.assertEqual(key_lengths, [5, 5])

    def test_disabled_layer_tuple_is_torch_compile_compatible(self) -> None:
        model = _FakeMoT(num_layers=2)

        def fake_attention(q, k, v, num_heads, ctx_mask):
            del k, v, num_heads, ctx_mask
            return q

        action_tokens = torch.zeros((1, 2, 1))
        video_cache = [torch.zeros((1, 3, 1)) for _ in range(2)]
        with mock.patch.object(mot_module, "flash_attention", new=fake_attention):
            compiled = torch.compile(
                model.forward_action_with_video_cache_tensor,
                backend="eager",
                fullgraph=True,
            )
            output = compiled(
                action_tokens=action_tokens,
                action_freqs=torch.empty(0),
                action_t_mod=torch.empty(0),
                action_context=torch.empty(0),
                action_context_mask=torch.empty(0),
                video_cache_k=video_cache,
                video_cache_v=video_cache,
                action_attention_mask=torch.ones((2, 5), dtype=torch.bool),
                disabled_video_layers=(1,),
            )
        self.assertEqual(tuple(output.shape), (1, 2, 1))


if __name__ == "__main__":
    unittest.main()
