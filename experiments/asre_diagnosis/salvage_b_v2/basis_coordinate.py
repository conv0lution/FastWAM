"""Round-4B-cache versus native-stock prefix-K/V coordinate gate.

The Round-4B basis was fitted from the action-side single-frame cache path.
This module compares that exact legacy extraction with the first-frame prefix
rows produced inside the stock joint graph.  It is a provenance gate, not a
scientific outcome and not an output-equivalence tolerance.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Mapping

import torch

from .definitions import LATE_LAYERS
from .native_clamp import FrozenPrefixTrajectory


COORDINATE_NUMERICAL_EPS_MULTIPLIER = 2.0
COORDINATE_MIN_COSINE_SIMILARITY = 0.999


def extract_round4b_cache_values(
    *,
    model,
    current_input_image: torch.Tensor,
    donor_input_image: torch.Tensor,
    infer_kwargs: Mapping[str, Any],
    expected_video_cache_layout: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the exact cache-only path used by Round 4B and return K/V values."""

    result = model.infer_action(
        prompt=infer_kwargs["prompt"],
        input_image=current_input_image,
        action_horizon=int(infer_kwargs["action_horizon"]),
        proprio=infer_kwargs["proprio"],
        context=infer_kwargs["context"],
        context_mask=infer_kwargs["context_mask"],
        negative_prompt=infer_kwargs["negative_prompt"],
        text_cfg_scale=float(infer_kwargs["text_cfg_scale"]),
        num_inference_steps=int(infer_kwargs["num_inference_steps"]),
        sigma_shift=float(infer_kwargs["sigma_shift"]),
        seed=infer_kwargs["seed"],
        rand_device=str(infer_kwargs["rand_device"]),
        tiled=bool(infer_kwargs["tiled"]),
        compile_action_infer=False,
        disabled_video_layers=tuple(range(15)),
        replacement_input_image=donor_input_image,
        replacement_video_layers=LATE_LAYERS,
        expected_video_cache_layout=expected_video_cache_layout,
        return_video_cache_deltas=True,
        return_video_cache_values=True,
        video_cache_delta_layers=LATE_LAYERS,
        cache_only=True,
    )
    values = result.get("video_cache_values")
    if not isinstance(values, dict) or set(values) != {"current", "wrong"}:
        raise RuntimeError("Round-4B cache-only extraction did not return both endpoints.")
    return {
        "layout": dict(result["video_cache_layout"]),
        "values": {
            endpoint: {
                kind: {
                    int(layer): tensor.detach().to(device="cpu").clone().contiguous()
                    for layer, tensor in by_layer.items()
                }
                for kind, by_layer in by_kind.items()
            }
            for endpoint, by_kind in values.items()
        },
    }


def _metrics(reference: torch.Tensor, observed: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float()
    observed_f = observed.float()
    difference = observed_f - reference_f
    reference_rms = float(reference_f.square().mean().sqrt().item())
    difference_rms = float(difference.square().mean().sqrt().item())
    denominator = max(reference_rms, torch.finfo(torch.float32).tiny)
    reference_flat = reference_f.reshape(-1)
    observed_flat = observed_f.reshape(-1)
    norm_product = float(reference_flat.norm().item() * observed_flat.norm().item())
    cosine = (
        1.0
        if norm_product == 0.0 and bool(torch.equal(reference_f, observed_f))
        else float(torch.dot(reference_flat, observed_flat).item() / norm_product)
    )
    return {
        "reference_rms": reference_rms,
        "difference_rms": difference_rms,
        "relative_rmse": difference_rms / denominator,
        "cosine_similarity": cosine,
        "max_absolute_difference": float(difference.abs().max().item()),
    }


def compare_round4b_to_native_stock(
    *,
    sample_id: str,
    legacy: Mapping[str, Any],
    current: FrozenPrefixTrajectory,
    wrong: FrozenPrefixTrajectory,
    expected_steps: int,
    execution_dtype: torch.dtype,
) -> dict[str, Any]:
    """Compare all 30 matrices for both endpoints and their fitted delta."""

    if not execution_dtype.is_floating_point:
        raise TypeError("Native coordinate comparison requires a floating dtype.")
    epsilon = float(torch.finfo(execution_dtype).eps)
    relative_budget = COORDINATE_NUMERICAL_EPS_MULTIPLIER * epsilon
    rows: list[dict[str, Any]] = []
    for step in range(expected_steps):
        for layer in LATE_LAYERS:
            for kind in ("k", "v"):
                legacy_current = legacy["values"]["current"][kind][layer]
                legacy_wrong = legacy["values"]["wrong"][kind][layer]
                native_current = current.values[(step, layer, kind)]
                native_wrong = wrong.values[(step, layer, kind)]
                for endpoint, reference, observed in (
                    ("current", legacy_current, native_current),
                    ("wrong", legacy_wrong, native_wrong),
                    (
                        "delta_current_minus_wrong",
                        legacy_current - legacy_wrong,
                        native_current - native_wrong,
                    ),
                ):
                    row = {
                        "sample_id": sample_id,
                        "step": step,
                        "layer": layer,
                        "tensor_kind": kind,
                        "endpoint": endpoint,
                    }
                    row.update(_metrics(reference, observed))
                    rows.append(row)
    endpoint_rows = [
        row for row in rows if row["endpoint"] in {"current", "wrong"}
    ]
    delta_rows = [
        row for row in rows if row["endpoint"] == "delta_current_minus_wrong"
    ]
    max_endpoint_relative_rmse = max(
        float(row["relative_rmse"]) for row in endpoint_rows
    )
    min_endpoint_cosine = min(
        float(row["cosine_similarity"]) for row in endpoint_rows
    )
    return {
        "sample_id": sample_id,
        "execution_dtype": str(execution_dtype),
        "execution_dtype_epsilon": epsilon,
        "relative_rmse_budget": relative_budget,
        "minimum_cosine_similarity": COORDINATE_MIN_COSINE_SIMILARITY,
        "endpoint_comparison_count": len(endpoint_rows),
        "delta_comparison_count": len(delta_rows),
        "max_endpoint_relative_rmse": max_endpoint_relative_rmse,
        "min_endpoint_cosine_similarity": min_endpoint_cosine,
        "max_delta_relative_rmse_descriptive": max(
            float(row["relative_rmse"]) for row in delta_rows
        ),
        "passed": bool(
            math.isfinite(max_endpoint_relative_rmse)
            and math.isfinite(min_endpoint_cosine)
            and max_endpoint_relative_rmse <= relative_budget
            and min_endpoint_cosine >= COORDINATE_MIN_COSINE_SIMILARITY
        ),
        "rows": rows,
    }


def static_coordinate_lineage(model) -> dict[str, Any]:
    """Prove both paths use the same video blocks and Q/K/V builder/order."""

    legacy_source = inspect.getsource(model.mot.prefill_video_cache_tensor)
    native_source = inspect.getsource(model.mot._forward_joint_layer)
    legacy_builder_calls = legacy_source.count("_build_expert_attention_io(")
    native_builder_calls = native_source.count("_build_expert_attention_io(")
    passed = bool(
        legacy_builder_calls == 1
        and native_builder_calls == 2
        and 'expert = self.mixtures["video"]' in legacy_source
        and 'video_expert = self.mixtures["video"]' in native_source
        and "k_video" in native_source
        and "v_video" in native_source
    )
    return {
        "passed": passed,
        "same_model_instance": True,
        "same_video_block_object_ids": [
            id(block) for block in model.mot.mixtures["video"].blocks
        ],
        "legacy_qkv_builder_calls": legacy_builder_calls,
        "native_qkv_builder_calls": native_builder_calls,
        "coordinate_order": "native flattened heads: 24 x 128 = 3072",
        "legacy_extractor": "FastWAM.infer_action -> MoT.prefill_video_cache_tensor",
        "native_extractor": "FastWAM.infer_joint -> MoT._forward_joint_layer",
    }

