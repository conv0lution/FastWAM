"""Small, model-independent helpers for matched-shape video-cache replacement."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch


def normalize_video_layer_indices(
    layers: Optional[Sequence[int]],
    *,
    argument_name: str,
    num_layers: int,
) -> tuple[int, ...]:
    if layers is None:
        return ()
    requested = list(layers)
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in requested):
        raise TypeError(f"`{argument_name}` must contain only integer layer indices.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"`{argument_name}` must not contain duplicate indices.")
    invalid = [layer for layer in requested if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"`{argument_name}` contains out-of-range indices {invalid}; "
            f"valid range is [0, {num_layers - 1}]."
        )
    return tuple(sorted(requested))


def validate_matching_video_cache(
    *,
    current_cache_k: Sequence[torch.Tensor],
    current_cache_v: Sequence[torch.Tensor],
    replacement_cache_k: Sequence[torch.Tensor],
    replacement_cache_v: Sequence[torch.Tensor],
    num_layers: int,
) -> None:
    groups = {
        "current K": current_cache_k,
        "current V": current_cache_v,
        "replacement K": replacement_cache_k,
        "replacement V": replacement_cache_v,
    }
    for label, cache in groups.items():
        if len(cache) != num_layers:
            raise ValueError(
                f"Expected {num_layers} per-layer tensors in {label} cache, got {len(cache)}."
            )
    for layer in range(num_layers):
        current_k = current_cache_k[layer]
        current_v = current_cache_v[layer]
        replacement_k = replacement_cache_k[layer]
        replacement_v = replacement_cache_v[layer]
        for label, tensor in (
            ("current K", current_k),
            ("current V", current_v),
            ("replacement K", replacement_k),
            ("replacement V", replacement_v),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{label} cache at layer {layer} must be a tensor, "
                    f"got {type(tensor).__name__}."
                )
        for kind, current, replacement in (
            ("K", current_k, replacement_k),
            ("V", current_v, replacement_v),
        ):
            if replacement.shape != current.shape:
                raise ValueError(
                    f"Replacement {kind} cache shape mismatch at layer {layer}: "
                    f"current={tuple(current.shape)}, replacement={tuple(replacement.shape)}."
                )
            if replacement.dtype != current.dtype:
                raise TypeError(
                    f"Replacement {kind} cache dtype mismatch at layer {layer}: "
                    f"current={current.dtype}, replacement={replacement.dtype}."
                )
            if replacement.device != current.device:
                raise ValueError(
                    f"Replacement {kind} cache device mismatch at layer {layer}: "
                    f"current={current.device}, replacement={replacement.device}."
                )
        for source, cache_k, cache_v in (
            ("Current", current_k, current_v),
            ("Replacement", replacement_k, replacement_v),
        ):
            if cache_v.shape != cache_k.shape:
                raise ValueError(
                    f"{source} K/V cache shape mismatch at layer {layer}: "
                    f"K={tuple(cache_k.shape)}, V={tuple(cache_v.shape)}."
                )
            if cache_v.dtype != cache_k.dtype:
                raise TypeError(
                    f"{source} K/V cache dtype mismatch at layer {layer}: "
                    f"K={cache_k.dtype}, V={cache_v.dtype}."
                )
            if cache_v.device != cache_k.device:
                raise ValueError(
                    f"{source} K/V cache device mismatch at layer {layer}: "
                    f"K={cache_k.device}, V={cache_v.device}."
                )


def select_replacement_video_cache(
    *,
    current_cache_k: Sequence[torch.Tensor],
    current_cache_v: Sequence[torch.Tensor],
    replacement_cache_k: Sequence[torch.Tensor],
    replacement_cache_v: Sequence[torch.Tensor],
    replacement_video_layers: Sequence[int],
    num_layers: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    validate_matching_video_cache(
        current_cache_k=current_cache_k,
        current_cache_v=current_cache_v,
        replacement_cache_k=replacement_cache_k,
        replacement_cache_v=replacement_cache_v,
        num_layers=num_layers,
    )
    selected = set(replacement_video_layers)
    return (
        [
            replacement_cache_k[layer]
            if layer in selected
            else current_cache_k[layer]
            for layer in range(num_layers)
        ],
        [
            replacement_cache_v[layer]
            if layer in selected
            else current_cache_v[layer]
            for layer in range(num_layers)
        ],
    )


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().to(dtype=torch.float32)
    if values.numel() == 0:
        raise ValueError("Cannot summarize an empty video-cache tensor.")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "finite": bool(torch.isfinite(values).all().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "rms": float(values.square().mean().sqrt().item()),
    }


def _difference_stats(
    current: torch.Tensor, replacement: torch.Tensor
) -> dict[str, Any]:
    delta = current.detach().to(dtype=torch.float32) - replacement.detach().to(
        dtype=torch.float32
    )
    absolute = delta.abs()
    return {
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "rms": float(delta.square().mean().sqrt().item()),
        "exact_equal": bool(torch.equal(current, replacement)),
    }


def build_video_cache_stats(
    *,
    current_cache_k: Sequence[torch.Tensor],
    current_cache_v: Sequence[torch.Tensor],
    replacement_cache_k: Optional[Sequence[torch.Tensor]],
    replacement_cache_v: Optional[Sequence[torch.Tensor]],
    replacement_video_layers: Sequence[int],
    num_layers: int,
    summary_video_layers: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    if len(current_cache_k) != num_layers or len(current_cache_v) != num_layers:
        raise ValueError(
            "Current video cache does not match the model layer count: "
            f"K={len(current_cache_k)}, V={len(current_cache_v)}, expected={num_layers}."
        )
    has_replacement = replacement_cache_k is not None or replacement_cache_v is not None
    if has_replacement:
        if replacement_cache_k is None or replacement_cache_v is None:
            raise ValueError("Replacement K and V caches must be provided together.")
        validate_matching_video_cache(
            current_cache_k=current_cache_k,
            current_cache_v=current_cache_v,
            replacement_cache_k=replacement_cache_k,
            replacement_cache_v=replacement_cache_v,
            num_layers=num_layers,
        )
    replacement_set = set(replacement_video_layers)
    summary_set = (
        set(range(num_layers))
        if summary_video_layers is None
        else set(summary_video_layers)
    )
    invalid = sorted(layer for layer in summary_set if layer < 0 or layer >= num_layers)
    if invalid:
        raise ValueError(f"Cache-summary layers are out of range: {invalid}.")

    layers: list[dict[str, Any]] = []
    global_max = 0.0
    all_exact = True
    for layer in range(num_layers):
        entry: dict[str, Any] = {
            "layer": layer,
            "selected_source": "replacement" if layer in replacement_set else "current",
            "summarized": layer in summary_set,
            "current": None,
            "replacement": None,
            "difference": None,
        }
        if layer in summary_set:
            entry["current"] = {
                "k": _tensor_stats(current_cache_k[layer]),
                "v": _tensor_stats(current_cache_v[layer]),
            }
        if (
            layer in summary_set
            and replacement_cache_k is not None
            and replacement_cache_v is not None
        ):
            difference_k = _difference_stats(
                current_cache_k[layer], replacement_cache_k[layer]
            )
            difference_v = _difference_stats(
                current_cache_v[layer], replacement_cache_v[layer]
            )
            entry["replacement"] = {
                "k": _tensor_stats(replacement_cache_k[layer]),
                "v": _tensor_stats(replacement_cache_v[layer]),
            }
            entry["difference"] = {"k": difference_k, "v": difference_v}
            global_max = max(
                global_max,
                float(difference_k["max_abs"]),
                float(difference_v["max_abs"]),
            )
            all_exact = (
                all_exact
                and bool(difference_k["exact_equal"])
                and bool(difference_v["exact_equal"])
            )
        layers.append(entry)
    return {
        "schema_version": 1,
        "replacement_video_layers": list(replacement_video_layers),
        "summarized_video_layers": sorted(summary_set),
        "has_replacement": has_replacement,
        "all_replacement_caches_exact_equal": all_exact if has_replacement else None,
        "max_abs_current_replacement": global_max if has_replacement else None,
        "layers": layers,
    }
