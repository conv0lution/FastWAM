from __future__ import annotations

import torch
import torch.nn as nn

from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import DiTBlock


class _TinyExpert(nn.Module):
    def __init__(self, *, layers: int, hidden_dim: int, heads: int, head_dim: int):
        super().__init__()
        self.num_heads = heads
        self.attn_head_dim = head_dim
        self.use_gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=head_dim,
                    num_heads=heads,
                    ffn_dim=hidden_dim * 2,
                )
                for _ in range(layers)
            ]
        )


def _fixture():
    torch.manual_seed(20260829)
    batch, prefix, future, action, hidden = 1, 2, 3, 2, 8
    heads, head_dim, layers, context_len = 2, 4, 3, 4
    mot = MoT(
        {
            "video": _TinyExpert(
                layers=layers,
                hidden_dim=hidden,
                heads=heads,
                head_dim=head_dim,
            ),
            "action": _TinyExpert(
                layers=layers,
                hidden_dim=hidden,
                heads=heads,
                head_dim=head_dim,
            ),
        }
    ).eval()

    video_len = prefix + future
    total_len = video_len + action
    video_tokens = torch.randn(batch, video_len, hidden)
    action_tokens = torch.randn(batch, action, hidden)
    video_t_mod = torch.randn(batch, video_len, 6, hidden) * 0.1
    action_t_mod = torch.randn(batch, action, 6, hidden) * 0.1
    video_freqs = torch.ones(video_len, 1, head_dim // 2, dtype=torch.complex128)
    action_freqs = torch.ones(action, 1, head_dim // 2, dtype=torch.complex128)
    video_context = torch.randn(batch, context_len, hidden)
    action_context = torch.randn(batch, context_len, hidden)
    video_context_mask = torch.ones(batch, video_len, context_len, dtype=torch.bool)
    action_context_mask = torch.ones(batch, action, context_len, dtype=torch.bool)

    # Exact first_frame_causal MoT layout: prefix queries cannot see future;
    # future queries see all video; action sees prefix and action only.
    attention_mask = torch.zeros(total_len, total_len, dtype=torch.bool)
    attention_mask[:prefix, :prefix] = True
    attention_mask[prefix:video_len, :video_len] = True
    attention_mask[video_len:, :prefix] = True
    attention_mask[video_len:, video_len:] = True
    return {
        "mot": mot,
        "prefix": prefix,
        "video_len": video_len,
        "video_tokens": video_tokens,
        "action_tokens": action_tokens,
        "video_t_mod": video_t_mod,
        "action_t_mod": action_t_mod,
        "video_freqs": video_freqs,
        "action_freqs": action_freqs,
        "video_context": video_context,
        "action_context": action_context,
        "video_context_mask": video_context_mask,
        "action_context_mask": action_context_mask,
        "attention_mask": attention_mask,
    }


@torch.no_grad()
def test_current_prefix_factorization_matches_stock_joint_future_tokens() -> None:
    state = _fixture()
    mot = state["mot"]
    prefix = state["prefix"]
    video_len = state["video_len"]

    stock_video, _ = mot.forward_joint_core(
        video_tokens=state["video_tokens"],
        action_tokens=state["action_tokens"],
        video_freqs=state["video_freqs"],
        action_freqs=state["action_freqs"],
        video_t_mod=state["video_t_mod"],
        action_t_mod=state["action_t_mod"],
        video_context=state["video_context"],
        video_context_mask=state["video_context_mask"],
        action_context=state["action_context"],
        action_context_mask=state["action_context_mask"],
        attention_mask=state["attention_mask"],
    )
    cache_k, cache_v = mot.prefill_video_cache_tensor(
        video_tokens=state["video_tokens"][:, :prefix],
        video_freqs=state["video_freqs"][:prefix],
        video_t_mod=state["video_t_mod"][:, :prefix],
        video_context=state["video_context"],
        video_context_mask=state["video_context_mask"][:, :prefix],
        video_attention_mask=state["attention_mask"][:prefix, :prefix],
    )
    # Runtime uses a constant-zero carrier prefix and discards those tokens;
    # the external cache still comes from the true current prefix above.  With
    # patch_t=1 the future input tokens are unchanged by that placeholder.
    factorized_carrier_tokens = state["video_tokens"].clone()
    factorized_carrier_tokens[:, :prefix] = 0
    assert not torch.equal(
        factorized_carrier_tokens[:, :prefix],
        state["video_tokens"][:, :prefix],
    )
    assert torch.equal(
        factorized_carrier_tokens[:, prefix:video_len],
        state["video_tokens"][:, prefix:video_len],
    )
    factorized_future = mot.forward_future_video_with_video_cache_tensor(
        future_video_tokens=factorized_carrier_tokens[:, prefix:video_len],
        future_video_freqs=state["video_freqs"][prefix:video_len],
        future_video_t_mod=state["video_t_mod"][:, prefix:video_len],
        future_video_context=state["video_context"],
        future_video_context_mask=state["video_context_mask"][:, prefix:video_len],
        video_cache_k=cache_k,
        video_cache_v=cache_v,
        future_video_attention_mask=state["attention_mask"][
            prefix:video_len, :video_len
        ],
    )

    torch.testing.assert_close(
        factorized_future,
        stock_video[:, prefix:video_len],
        atol=2e-6,
        rtol=2e-6,
    )


@torch.no_grad()
def test_disabled_prefix_layers_remove_the_only_prefix_pathway() -> None:
    state = _fixture()
    mot = state["mot"]
    prefix = state["prefix"]
    video_len = state["video_len"]
    cache_k, cache_v = mot.prefill_video_cache_tensor(
        video_tokens=state["video_tokens"][:, :prefix],
        video_freqs=state["video_freqs"][:prefix],
        video_t_mod=state["video_t_mod"][:, :prefix],
        video_context=state["video_context"],
        video_context_mask=state["video_context_mask"][:, :prefix],
        video_attention_mask=state["attention_mask"][:prefix, :prefix],
    )
    common = {
        "future_video_tokens": state["video_tokens"][:, prefix:video_len],
        "future_video_freqs": state["video_freqs"][prefix:video_len],
        "future_video_t_mod": state["video_t_mod"][:, prefix:video_len],
        "future_video_context": state["video_context"],
        "future_video_context_mask": state["video_context_mask"][
            :, prefix:video_len
        ],
        "future_video_attention_mask": state["attention_mask"][
            prefix:video_len, :video_len
        ],
    }
    altered_k = [tensor + 100.0 for tensor in cache_k]
    altered_v = [tensor - 100.0 for tensor in cache_v]

    enabled = mot.forward_future_video_with_video_cache_tensor(
        **common,
        video_cache_k=cache_k,
        video_cache_v=cache_v,
    )
    altered_enabled = mot.forward_future_video_with_video_cache_tensor(
        **common,
        video_cache_k=altered_k,
        video_cache_v=altered_v,
    )
    disabled = mot.forward_future_video_with_video_cache_tensor(
        **common,
        video_cache_k=cache_k,
        video_cache_v=cache_v,
        disabled_video_prefix_layers=tuple(range(mot.num_layers)),
    )
    altered_disabled = mot.forward_future_video_with_video_cache_tensor(
        **common,
        video_cache_k=altered_k,
        video_cache_v=altered_v,
        disabled_video_prefix_layers=tuple(range(mot.num_layers)),
    )

    assert not torch.allclose(enabled, altered_enabled)
    torch.testing.assert_close(disabled, altered_disabled, atol=0.0, rtol=0.0)
