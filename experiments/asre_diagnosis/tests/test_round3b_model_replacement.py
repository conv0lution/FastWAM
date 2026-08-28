from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.video_cache_replacement import (
    build_video_cache_stats as _build_video_cache_stats,
    normalize_video_layer_indices as _normalize_video_layer_indices,
    select_replacement_video_cache as _select_replacement_video_cache,
)
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


class _TinyVideoExpert(nn.Module):
    patch_size = (1, 1, 1)
    fuse_vae_embedding_in_latents = False
    video_attention_mask_mode = "first_frame_causal"

    @staticmethod
    def build_video_to_video_mask(
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        del video_tokens_per_frame
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    @staticmethod
    def prepare(
        *,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action,
        fuse_vae_embedding_in_latents: bool,
    ):
        del action, fuse_vae_embedding_in_latents
        tokens = x.flatten(2).transpose(1, 2)
        token_count = tokens.shape[1]
        t_mod = timestep.reshape(1, 1, 1).expand(1, token_count, 1)
        freqs = torch.zeros((token_count, 1), dtype=x.dtype, device=x.device)
        return (
            tokens,
            timestep,
            t_mod,
            context,
            context_mask,
            freqs,
            1,
            1,
            1,
            token_count,
        )


class _TinyActionExpert(nn.Module):
    action_dim = 1


class _TinyMoT(nn.Module):
    num_layers = 3
    num_heads = 1
    attn_head_dim = 1

    def __init__(self) -> None:
        super().__init__()
        self.prefill_inputs: list[torch.Tensor] = []

    def prefill_video_cache_tensor(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        video_attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        del (
            video_freqs,
            video_t_mod,
            video_context,
            video_context_mask,
            video_attention_mask,
        )
        self.prefill_inputs.append(video_tokens.detach().clone())
        cache_k = [video_tokens + float(layer) for layer in range(self.num_layers)]
        cache_v = [video_tokens + float(layer) + 0.5 for layer in range(self.num_layers)]
        return cache_k, cache_v


class _TinyFastWAM(FastWAM):
    """The actual model class selected by libero_uncond_2cam224_1e-4."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.proprio_dim = None
        self.proprio_encoder = None
        self.video_expert = _TinyVideoExpert()
        self.action_expert = _TinyActionExpert()
        self.mot = _TinyMoT()
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(shift=1.0)
        self.action_cache_inputs: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []

    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled: bool = False,
    ) -> torch.Tensor:
        del tiled
        return input_image.mean().reshape(1, 1, 1, 1, 1)

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
        del (
            timestep_action,
            context,
            context_mask,
            action_attention_mask,
            disabled_video_layers,
        )
        self.action_cache_inputs.append((list(video_cache_k), list(video_cache_v)))
        return torch.zeros_like(latents_action)


def _infer_kwargs(input_image: torch.Tensor) -> dict:
    return {
        "prompt": None,
        "input_image": input_image,
        "action_horizon": 2,
        "num_video_frames": 1,
        "context": torch.zeros((1, 1, 1), dtype=torch.float32),
        "context_mask": torch.ones((1, 1), dtype=torch.bool),
        "num_inference_steps": 1,
        "seed": 7,
    }


def _infer_fastwam_kwargs(input_image: torch.Tensor) -> dict:
    kwargs = _infer_kwargs(input_image)
    kwargs.pop("num_video_frames")
    return kwargs


class Round3BLayerValidationTest(unittest.TestCase):
    def test_layer_indices_are_sorted_and_strict(self) -> None:
        self.assertEqual(
            _normalize_video_layer_indices(
                [2, 0], argument_name="replacement_video_layers", num_layers=3
            ),
            (0, 2),
        )
        for invalid, error in (([True], TypeError), ([1, 1], ValueError), ([3], ValueError)):
            with self.subTest(invalid=invalid), self.assertRaises(error):
                _normalize_video_layer_indices(
                    invalid,
                    argument_name="replacement_video_layers",
                    num_layers=3,
                )

    def test_cache_selection_uses_replacement_only_on_requested_layers(self) -> None:
        current_k = [torch.full((1, 2, 1), float(i)) for i in range(3)]
        current_v = [tensor + 0.25 for tensor in current_k]
        replacement_k = [tensor + 10.0 for tensor in current_k]
        replacement_v = [tensor + 10.0 for tensor in current_v]

        selected_k, selected_v = _select_replacement_video_cache(
            current_cache_k=current_k,
            current_cache_v=current_v,
            replacement_cache_k=replacement_k,
            replacement_cache_v=replacement_v,
            replacement_video_layers=(1, 2),
            num_layers=3,
        )

        self.assertIs(selected_k[0], current_k[0])
        self.assertIs(selected_v[0], current_v[0])
        self.assertIs(selected_k[1], replacement_k[1])
        self.assertIs(selected_v[2], replacement_v[2])

    def test_cache_selection_rejects_non_matching_shape_and_dtype(self) -> None:
        current_k = [torch.zeros((1, 2, 1))]
        current_v = [torch.zeros((1, 2, 1))]
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            _select_replacement_video_cache(
                current_cache_k=current_k,
                current_cache_v=current_v,
                replacement_cache_k=[torch.zeros((1, 3, 1))],
                replacement_cache_v=[torch.zeros((1, 2, 1))],
                replacement_video_layers=(0,),
                num_layers=1,
            )
        with self.assertRaisesRegex(TypeError, "dtype mismatch"):
            _select_replacement_video_cache(
                current_cache_k=current_k,
                current_cache_v=current_v,
                replacement_cache_k=[torch.zeros((1, 2, 1), dtype=torch.float64)],
                replacement_cache_v=[torch.zeros((1, 2, 1), dtype=torch.float64)],
                replacement_video_layers=(0,),
                num_layers=1,
            )


class Round3BCacheStatsTest(unittest.TestCase):
    def test_self_replacement_stats_are_exact(self) -> None:
        current_k = [torch.tensor([[[1.0], [2.0]]])]
        current_v = [torch.tensor([[[3.0], [4.0]]])]
        stats = _build_video_cache_stats(
            current_cache_k=current_k,
            current_cache_v=current_v,
            replacement_cache_k=[current_k[0].clone()],
            replacement_cache_v=[current_v[0].clone()],
            replacement_video_layers=(0,),
            num_layers=1,
        )
        self.assertTrue(stats["all_replacement_caches_exact_equal"])
        self.assertEqual(stats["max_abs_current_replacement"], 0.0)
        self.assertEqual(stats["layers"][0]["selected_source"], "replacement")
        self.assertEqual(stats["layers"][0]["current"]["k"]["shape"], [1, 2, 1])


class Round3BRealCheckpointModelPathTest(unittest.TestCase):
    def test_default_path_preserves_fastwam_output_and_single_prefill(self) -> None:
        model = _TinyFastWAM()
        output = model.infer_action(
            **_infer_fastwam_kwargs(torch.zeros((1, 3, 16, 16)))
        )

        self.assertEqual(set(output), {"action"})
        self.assertEqual(len(model.mot.prefill_inputs), 1)
        self.assertEqual(len(model.action_cache_inputs), 1)

    def test_fastwam_replacement_selects_only_requested_layers(self) -> None:
        model = _TinyFastWAM()
        output = model.infer_action(
            **_infer_fastwam_kwargs(torch.zeros((1, 3, 16, 16))),
            replacement_input_image=torch.ones((1, 3, 16, 16)),
            replacement_video_layers=(1, 2),
            return_video_cache_stats=True,
        )

        self.assertEqual(len(model.mot.prefill_inputs), 2)
        selected_k, selected_v = model.action_cache_inputs[0]
        self.assertEqual(float(selected_k[0].item()), 0.0)
        self.assertEqual(float(selected_k[1].item()), 2.0)
        self.assertEqual(float(selected_v[2].item()), 3.5)
        audit = output["video_cache_stats"]
        self.assertEqual(audit["replacement_video_layers"], [1, 2])
        self.assertEqual(audit["current_video_seq_len"], audit["replacement_video_seq_len"])
        self.assertEqual(
            [entry["selected_source"] for entry in audit["layers"]],
            ["current", "replacement", "replacement"],
        )

    def test_fastwam_same_image_replacement_is_exact(self) -> None:
        image = torch.rand((1, 3, 16, 16), generator=torch.Generator().manual_seed(9))
        baseline = _TinyFastWAM().infer_action(**_infer_fastwam_kwargs(image.clone()))
        replacement = _TinyFastWAM().infer_action(
            **_infer_fastwam_kwargs(image.clone()),
            replacement_input_image=image.clone(),
            replacement_video_layers=(0, 1, 2),
            return_video_cache_stats=True,
        )

        torch.testing.assert_close(baseline["action"], replacement["action"], atol=0, rtol=0)
        self.assertTrue(
            replacement["video_cache_stats"]["all_replacement_caches_exact_equal"]
        )
        self.assertEqual(
            replacement["video_cache_stats"]["max_abs_current_replacement"], 0.0
        )

    def test_fastwam_rejects_missing_or_overlapping_replacement(self) -> None:
        model = _TinyFastWAM()
        kwargs = _infer_fastwam_kwargs(torch.zeros((1, 3, 16, 16)))
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            model.infer_action(**kwargs, replacement_video_layers=(1,))
        with self.assertRaisesRegex(ValueError, "overlapping layers"):
            model.infer_action(
                **kwargs,
                disabled_video_layers=(1,),
                replacement_input_image=torch.ones((1, 3, 16, 16)),
                replacement_video_layers=(1,),
            )
        with self.assertRaisesRegex(ValueError, "image shape"):
            model.infer_action(
                **kwargs,
                replacement_input_image=torch.zeros((1, 3, 16, 32)),
                replacement_video_layers=(1,),
            )
        with self.assertRaisesRegex(TypeError, "image dtype"):
            model.infer_action(
                **kwargs,
                replacement_input_image=torch.zeros(
                    (1, 3, 16, 16), dtype=torch.float64
                ),
                replacement_video_layers=(1,),
            )
        with self.assertRaisesRegex(ValueError, "same device"):
            model.infer_action(
                **kwargs,
                replacement_input_image=torch.empty(
                    (1, 3, 16, 16), device="meta"
                ),
                replacement_video_layers=(1,),
            )


if __name__ == "__main__":
    unittest.main()
