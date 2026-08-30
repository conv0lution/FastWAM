"""Run Salvage-B-v2 conditions through the unmodified native joint graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .definitions import FEATURE_DIM, RANK_BY_CONDITION
from .native_clamp import FrozenPrefixTrajectory, NativePrefixClamp


@dataclass
class NativeConditionResult:
    prediction: dict[str, Any]
    clamp: NativePrefixClamp | None
    current_trajectory: FrozenPrefixTrajectory | None
    wrong_trajectory: FrozenPrefixTrajectory | None


def _native_call(
    *,
    model,
    input_image: torch.Tensor,
    infer_kwargs: Mapping[str, Any],
    hook=None,
) -> dict[str, Any]:
    kwargs = dict(infer_kwargs)
    kwargs.update(
        {
            "input_image": input_image,
            "test_action_with_infer_action": False,
            "compile_action_infer": False,
            "native_prefix_kv_hook": hook,
            "decode_video": False,
        }
    )
    return model.infer_joint(**kwargs)


def capture_native_stock_trajectory(
    *,
    model,
    input_image: torch.Tensor,
    infer_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], FrozenPrefixTrajectory]:
    trajectory = FrozenPrefixTrajectory(num_layers=int(model.mot.num_layers))
    prediction = _native_call(
        model=model,
        input_image=input_image,
        infer_kwargs=infer_kwargs,
        hook=trajectory.capture_hook,
    )
    steps = int(infer_kwargs["num_inference_steps"])
    trajectory.validate_complete(expected_steps=steps)
    return prediction, trajectory


def run_native_condition(
    *,
    model,
    condition: str,
    current_input_image: torch.Tensor,
    donor_input_image: torch.Tensor | None,
    infer_kwargs: Mapping[str, Any],
    bases_by_layer: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
) -> NativeConditionResult:
    """Evaluate exactly one condition with frozen exogenous per-step clamps."""

    if condition not in RANK_BY_CONDITION:
        raise ValueError(f"Unknown Salvage-B-v2 condition: {condition!r}.")
    if condition == "current":
        prediction = _native_call(
            model=model,
            input_image=current_input_image,
            infer_kwargs=infer_kwargs,
        )
        return NativeConditionResult(prediction, None, None, None)
    if donor_input_image is None:
        raise ValueError(f"Condition {condition!r} requires a frozen donor image.")

    _current_prediction, current = capture_native_stock_trajectory(
        model=model,
        input_image=current_input_image,
        infer_kwargs=infer_kwargs,
    )
    _wrong_prediction, wrong = capture_native_stock_trajectory(
        model=model,
        input_image=donor_input_image,
        infer_kwargs=infer_kwargs,
    )
    rank = int(RANK_BY_CONDITION[condition])
    clamp = NativePrefixClamp(
        current=current,
        wrong=wrong,
        rank=rank,
        bases_by_layer=bases_by_layer,
        num_layers=int(model.mot.num_layers),
    )
    prediction = _native_call(
        model=model,
        input_image=current_input_image,
        infer_kwargs=infer_kwargs,
        hook=clamp.hook,
    )
    clamp.validate_complete(expected_steps=int(infer_kwargs["num_inference_steps"]))
    return NativeConditionResult(prediction, clamp, current, wrong)


def endpoint_rank(condition: str) -> int:
    rank = int(RANK_BY_CONDITION[condition])
    if rank not in (0, 97, 170, FEATURE_DIM):
        raise AssertionError("The v2 protocol admitted an unregistered rank.")
    return rank
