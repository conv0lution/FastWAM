"""Run Salvage-B-v2 conditions through the unmodified native joint graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from experiments.asre_diagnosis.common import sha256_json
from experiments.asre_diagnosis.round3b.donor import tensor_sha256

from .definitions import FEATURE_DIM, RANK_BY_CONDITION
from .native_clamp import FrozenPrefixTrajectory, NativePrefixClamp


FROZEN_EXOGENOUS_FIELDS = (
    "prompt",
    "num_video_frames",
    "action_horizon",
    "action",
    "proprio",
    "context",
    "context_mask",
    "negative_prompt",
    "text_cfg_scale",
    "num_inference_steps",
    "sigma_shift",
    "seed",
    "rand_device",
    "tiled",
    "initial_video_noise",
    "initial_action_noise",
)

_OPTIONAL_EXOGENOUS_DEFAULTS = {
    "action": None,
    "negative_prompt": None,
    "text_cfg_scale": 1.0,
    "initial_video_noise": None,
    "initial_action_noise": None,
}


def _fingerprint_value(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().to(device="cpu").contiguous()
        return {
            "type": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": tensor_sha256(tensor),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "A frozen native-joint exogenous input has an unsupported type: "
        f"{type(value).__name__}."
    )


def frozen_exogenous_signature(*, model, infer_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Hash every non-observation input shared by current/donor/projected calls."""

    missing = [
        field
        for field in FROZEN_EXOGENOUS_FIELDS
        if field not in infer_kwargs and field not in _OPTIONAL_EXOGENOUS_DEFAULTS
    ]
    if missing:
        raise ValueError(
            "Native shared-node inference is missing frozen exogenous fields: "
            f"{missing}."
        )
    fields = {
        field: _fingerprint_value(
            infer_kwargs.get(field, _OPTIONAL_EXOGENOUS_DEFAULTS.get(field))
        )
        for field in FROZEN_EXOGENOUS_FIELDS
    }
    if (
        fields["seed"] is None
        and (
            fields["initial_video_noise"] is None
            or fields["initial_action_noise"] is None
        )
    ):
        raise ValueError(
            "Native shared-node inference must freeze stochastic inputs using either "
            "an explicit seed or explicit initial video/action noise tensors."
        )
    def _scheduler(scheduler: Any) -> dict[str, Any]:
        configuration = {
            name: _fingerprint_value(getattr(scheduler, name))
            for name in (
                "num_train_timesteps",
                "shift",
                "eps",
                "_y_min",
                "_weight_norm_const",
            )
            if hasattr(scheduler, name)
        }
        return {
            "class": type(scheduler).__qualname__,
            "object_id": id(scheduler),
            "configuration": configuration,
            "configuration_sha256": sha256_json(configuration),
        }

    scheduler_objects = {
        "video": _scheduler(model.infer_video_scheduler),
        "action": _scheduler(model.infer_action_scheduler),
    }
    return {
        "fields": fields,
        "fields_sha256": sha256_json(fields),
        "scheduler_objects": scheduler_objects,
    }


def _assert_frozen_exogenous_inputs(
    *,
    expected: Mapping[str, Any],
    model,
    infer_kwargs: Mapping[str, Any],
    call_role: str,
) -> None:
    observed = frozen_exogenous_signature(model=model, infer_kwargs=infer_kwargs)
    if observed != expected:
        raise RuntimeError(
            "Salvage-B-v2 donor/current/projected calls changed a non-observation "
            f"input during {call_role}: expected={expected}, observed={observed}."
        )


@dataclass
class NativeConditionResult:
    prediction: dict[str, Any]
    clamp: NativePrefixClamp | None
    current_trajectory: FrozenPrefixTrajectory | None
    wrong_trajectory: FrozenPrefixTrajectory | None
    causal_input_audit: dict[str, Any]


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
    expected_exogenous = frozen_exogenous_signature(
        model=model, infer_kwargs=infer_kwargs
    )
    current_image_sha256 = tensor_sha256(
        current_input_image.detach().to(device="cpu").contiguous()
    )
    donor_image_sha256 = (
        None
        if donor_input_image is None
        else tensor_sha256(donor_input_image.detach().to(device="cpu").contiguous())
    )

    def _audit(call_role: str) -> None:
        _assert_frozen_exogenous_inputs(
            expected=expected_exogenous,
            model=model,
            infer_kwargs=infer_kwargs,
            call_role=call_role,
        )

    def _audit_payload(call_roles: list[str]) -> dict[str, Any]:
        return {
            "passed": True,
            "only_observation_varied_for_wrong_capture": condition == "current"
            or donor_input_image is not None,
            "donor_rgb_source": "donor_input_image_only",
            "task_context_source": "current_infer_kwargs",
            "proprio_source": "current_infer_kwargs",
            "noise_source": "same_frozen_infer_kwargs",
            "scheduler_source": "same_model_objects_and_schedule_arguments",
            "frozen_exogenous_fields": list(FROZEN_EXOGENOUS_FIELDS),
            "frozen_exogenous_fields_sha256": expected_exogenous["fields_sha256"],
            "scheduler_objects": expected_exogenous["scheduler_objects"],
            "current_image_sha256": current_image_sha256,
            "donor_image_sha256": donor_image_sha256,
            "call_roles": call_roles,
        }

    if condition == "current":
        _audit("before_current_scientific_call")
        prediction = _native_call(
            model=model,
            input_image=current_input_image,
            infer_kwargs=infer_kwargs,
        )
        _audit("after_current_scientific_call")
        return NativeConditionResult(
            prediction,
            None,
            None,
            None,
            _audit_payload(["current_scientific"]),
        )
    if donor_input_image is None:
        raise ValueError(f"Condition {condition!r} requires a frozen donor image.")

    _audit("before_current_capture")
    _current_prediction, current = capture_native_stock_trajectory(
        model=model,
        input_image=current_input_image,
        infer_kwargs=infer_kwargs,
    )
    _audit("after_current_capture")
    # The donor call receives only a different ``input_image`` argument.  It is
    # deliberately impossible to supply donor proprio/context through this API:
    # every non-image input comes from the same current-recipient infer_kwargs.
    _audit("before_donor_rgb_only_capture")
    _wrong_prediction, wrong = capture_native_stock_trajectory(
        model=model,
        input_image=donor_input_image,
        infer_kwargs=infer_kwargs,
    )
    _audit("after_donor_rgb_only_capture")
    rank = int(RANK_BY_CONDITION[condition])
    clamp = NativePrefixClamp(
        current=current,
        wrong=wrong,
        rank=rank,
        bases_by_layer=bases_by_layer,
        num_layers=int(model.mot.num_layers),
    )
    _audit("before_intervened_current_call")
    prediction = _native_call(
        model=model,
        input_image=current_input_image,
        infer_kwargs=infer_kwargs,
        hook=clamp.hook,
    )
    _audit("after_intervened_current_call")
    clamp.validate_complete(expected_steps=int(infer_kwargs["num_inference_steps"]))
    return NativeConditionResult(
        prediction,
        clamp,
        current,
        wrong,
        _audit_payload(
            ["current_capture", "donor_rgb_only_capture", "intervened_current"]
        ),
    )


def endpoint_rank(condition: str) -> int:
    rank = int(RANK_BY_CONDITION[condition])
    if rank not in (0, 97, 170, FEATURE_DIM):
        raise AssertionError("The v2 protocol admitted an unregistered rank.")
    return rank
