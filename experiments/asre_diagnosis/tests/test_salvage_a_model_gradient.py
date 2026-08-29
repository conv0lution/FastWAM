from __future__ import annotations

import unittest

import torch

from experiments.asre_diagnosis.tests.test_round3b_model_replacement import (
    _TinyFastWAM,
    _infer_fastwam_kwargs,
)


class _CacheDependentTinyFastWAM(_TinyFastWAM):
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
        signal = latents_action.new_zeros(())
        for layer, (cache_k, cache_v) in enumerate(zip(video_cache_k, video_cache_v)):
            if layer not in disabled_video_layers:
                signal = signal + cache_k.mean() + 0.5 * cache_v.mean()
        return latents_action * 0.125 + signal * 0.01


class SalvageAExactGradientPathTest(unittest.TestCase):
    def _run(self, interpolation_lambda: float, *, checkpoint: bool):
        model = _CacheDependentTinyFastWAM().eval()
        model.requires_grad_(False)
        kwargs = _infer_fastwam_kwargs(torch.zeros((1, 3, 16, 16)))
        return model.infer_action(
            **kwargs,
            disabled_video_layers=(0,),
            replacement_input_image=torch.ones((1, 3, 16, 16)),
            replacement_video_layers=(1, 2),
            action_sensitive_interpolation_lambda=interpolation_lambda,
            action_sensitive_layers=(1, 2),
            action_sensitive_gradient_checkpointing=checkpoint,
            return_video_cache_deltas=True,
            video_cache_delta_layers=(1, 2),
            compile_action_infer=False,
        )

    def test_endpoints_match_validated_standard_paths(self) -> None:
        kwargs = _infer_fastwam_kwargs(torch.zeros((1, 3, 16, 16)))
        current_model = _CacheDependentTinyFastWAM().eval()
        current = current_model.infer_action(**kwargs, disabled_video_layers=(0,))
        wrong_model = _CacheDependentTinyFastWAM().eval()
        wrong = wrong_model.infer_action(
            **kwargs,
            disabled_video_layers=(0,),
            replacement_input_image=torch.ones((1, 3, 16, 16)),
            replacement_video_layers=(1, 2),
        )

        endpoint_current = self._run(1.0, checkpoint=False)
        endpoint_wrong = self._run(0.0, checkpoint=False)
        torch.testing.assert_close(endpoint_current["action"], current["action"])
        torch.testing.assert_close(endpoint_wrong["action"], wrong["action"])
        self.assertTrue(endpoint_current["action"].requires_grad)
        self.assertEqual(endpoint_current["video_cache_layout"]["feature_dim"], 1)

    def test_full_path_returns_finite_nonzero_cache_vjps(self) -> None:
        output = self._run(0.5, checkpoint=True)
        leaves = output["action_sensitive_cache_tensors"]
        ordered = tuple(
            leaves[kind][layer] for kind in ("k", "v") for layer in (1, 2)
        )
        gradients = torch.autograd.grad(output["action"].sum(), ordered)
        self.assertEqual(len(gradients), 4)
        for leaf, gradient in zip(ordered, gradients):
            self.assertTrue(leaf.requires_grad)
            self.assertEqual(tuple(gradient.shape), (1, 1, 1))
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().max()), 0.0)
        self.assertEqual(output["action_sensitive_inference"]["num_inference_steps"], 1)
        self.assertTrue(
            output["action_sensitive_inference"]["gradient_checkpointing"]
        )

    def test_checkpointed_and_direct_objectives_are_equivalent(self) -> None:
        direct = self._run(0.5, checkpoint=False)
        checkpointed = self._run(0.5, checkpoint=True)
        torch.testing.assert_close(direct["action"], checkpointed["action"])

    def test_action_sensitive_mode_fails_closed_on_bad_flags(self) -> None:
        model = _CacheDependentTinyFastWAM().eval()
        kwargs = _infer_fastwam_kwargs(torch.zeros((1, 3, 16, 16)))
        with self.assertRaisesRegex(ValueError, "both interpolation lambda and layers"):
            model.infer_action(
                **kwargs,
                replacement_input_image=torch.ones((1, 3, 16, 16)),
                replacement_video_layers=(1, 2),
                action_sensitive_interpolation_lambda=0.5,
            )
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            model.infer_action(
                **kwargs,
                replacement_input_image=torch.ones((1, 3, 16, 16)),
                replacement_video_layers=(1, 2),
                action_sensitive_interpolation_lambda=0.5,
                action_sensitive_layers=(2,),
            )


if __name__ == "__main__":
    unittest.main()
