from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from experiments.asre_diagnosis.salvage_b.world_runtime import (
    NATIVE_WORLD_INFERENCE_SHIFT,
    NATIVE_WORLD_INFERENCE_STEPS,
    PreparedWorldSample,
    native_world_loss,
)


class _Scheduler:
    def __init__(self) -> None:
        self.calls = 0

    def build_inference_schedule(
        self, *, num_inference_steps, device, dtype, shift_override
    ):
        assert num_inference_steps == NATIVE_WORLD_INFERENCE_STEPS
        assert shift_override == NATIVE_WORLD_INFERENCE_SHIFT
        timesteps = torch.linspace(1000.0, 100.0, num_inference_steps, device=device)
        deltas = torch.full((num_inference_steps,), -0.1, device=device)
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    def step(self, model_output, delta, sample):
        self.calls += 1
        return sample + model_output * delta


class _VideoExpert:
    @staticmethod
    def prepare(*, x, timestep, context, context_mask, action, fuse_vae_embedding_in_latents):
        assert tuple(x.shape) == (1, 48, 3, 14, 28)
        assert action is None and fuse_vae_embedding_in_latents is True
        assert int(torch.count_nonzero(x[:, :, 0]).item()) == 0
        # The runtime checks the native 3 x 98 token layout.  Values are not
        # relevant to this target-is-scoring-only contract fixture.
        tokens = torch.zeros((1, 294, 1), device=x.device, dtype=x.dtype)
        t = torch.zeros_like(tokens)
        t_mod = torch.zeros((1, 294, 6, 1), device=x.device, dtype=x.dtype)
        expanded_mask = context_mask[:, None, :].expand(-1, 294, -1)
        freqs = torch.zeros((294, 1), device=x.device, dtype=x.dtype)
        return tokens, t, t_mod, context, expanded_mask, freqs, 3, 7, 14, 98

    @staticmethod
    def build_video_to_video_mask(*, video_seq_len, video_tokens_per_frame, device):
        assert (video_seq_len, video_tokens_per_frame) == (294, 98)
        return torch.ones((294, 294), device=device, dtype=torch.bool)

    @staticmethod
    def post(tokens, timestep, f, h, w):
        assert tuple(tokens.shape) == (1, 196, 1)
        assert (f, h, w) == (2, 7, 14)
        return torch.zeros((1, 48, 2, 14, 28), device=tokens.device, dtype=tokens.dtype)


class _Mot:
    @staticmethod
    def forward_future_video_with_video_cache_tensor(**kwargs):
        assert kwargs["disabled_video_prefix_layers"] == tuple(range(15))
        assert tuple(kwargs["future_video_tokens"].shape) == (1, 196, 1)
        return kwargs["future_video_tokens"]


class _Model:
    def __init__(self) -> None:
        self.infer_video_scheduler = _Scheduler()
        self.video_expert = _VideoExpert()
        self.mot = _Mot()


def _prepared(target: torch.Tensor) -> PreparedWorldSample:
    return PreparedWorldSample(
        sample_id="sample-0",
        task_id=0,
        episode_id=0,
        trial=0,
        inputs={
            "context": torch.zeros((1, 2, 1)),
            "context_mask": torch.ones((1, 2), dtype=torch.bool),
        },
        input_latents=target,
        current_frame_latent=torch.full((1, 48, 1, 14, 28), 123.0),
        current_cache_k=[],
        current_cache_v=[],
        wrong_cache_k=[],
        wrong_cache_v=[],
        prefix_context=torch.zeros((1, 2, 1)),
        prefix_context_mask=torch.ones((1, 98, 2), dtype=torch.bool),
        prefix_tokens=98,
        current_image_sha256="a" * 64,
        donor_image_sha256="b" * 64,
        target_latent_sha256="c" * 64,
    )


def test_native_world_inference_is_independent_of_future_scoring_target() -> None:
    noise = torch.full((48, 2, 14, 28), 0.25)
    first_target = torch.zeros((1, 48, 3, 14, 28))
    second_target = first_target.clone()
    second_target[:, :, 1:] = 2.0
    model = _Model()

    first = native_world_loss(
        model=model,
        prepared=_prepared(first_target),
        video_cache_k=[],
        video_cache_v=[],
        video_noise=noise,
    )
    second = native_world_loss(
        model=model,
        prepared=replace(_prepared(first_target), input_latents=second_target),
        video_cache_k=[],
        video_cache_v=[],
        video_noise=noise,
    )

    torch.testing.assert_close(first["prediction"], second["prediction"], rtol=0, atol=0)
    torch.testing.assert_close(first["prediction"], noise.unsqueeze(0), rtol=0, atol=0)
    assert first["native_world_loss"] == pytest.approx(0.25**2)
    assert second["native_world_loss"] == pytest.approx((2.0 - 0.25) ** 2)
    assert first["native_world_loss"] == first["future_latent_mse"]
    assert first["inference_steps"] == NATIVE_WORLD_INFERENCE_STEPS
    assert first["inference_shift"] == NATIVE_WORLD_INFERENCE_SHIFT
    assert model.infer_video_scheduler.calls == 2 * NATIVE_WORLD_INFERENCE_STEPS
