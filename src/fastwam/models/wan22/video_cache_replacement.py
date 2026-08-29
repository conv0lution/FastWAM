"""Small, model-independent helpers for matched-shape video-cache replacement."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

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


def project_replacement_video_cache(
    *,
    current_cache_k: Sequence[torch.Tensor],
    current_cache_v: Sequence[torch.Tensor],
    replacement_cache_k: Sequence[torch.Tensor],
    replacement_cache_v: Sequence[torch.Tensor],
    replacement_video_layers: Sequence[int],
    action_visible_token_indices: Sequence[int],
    feature_bases_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
    projection_rank: int,
    num_layers: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    """Project current-minus-donor cache deltas into frozen feature subspaces.

    K and V use independent right-feature bases.  Only action-visible video rows
    are changed; cache geometry, token count, and head packing remain untouched.
    """
    validate_matching_video_cache(
        current_cache_k=current_cache_k,
        current_cache_v=current_cache_v,
        replacement_cache_k=replacement_cache_k,
        replacement_cache_v=replacement_cache_v,
        num_layers=num_layers,
    )
    layers = normalize_video_layer_indices(
        replacement_video_layers,
        argument_name="replacement_video_layers",
        num_layers=num_layers,
    )
    if not layers:
        raise ValueError("Feature projection requires non-empty replacement layers.")
    if isinstance(projection_rank, bool) or not isinstance(projection_rank, int):
        raise TypeError("`projection_rank` must be an integer.")
    representative = current_cache_k[layers[0]]
    if representative.ndim != 3:
        raise ValueError(
            "Video-cache tensors must have shape [batch,tokens,feature], got "
            f"{tuple(representative.shape)}."
        )
    video_seq_len = int(representative.shape[1])
    feature_dim = int(representative.shape[2])
    if projection_rank < 0 or projection_rank > feature_dim:
        raise ValueError(
            f"projection_rank must be in [0,{feature_dim}], got {projection_rank}."
        )
    visible = _normalize_component_indices(
        action_visible_token_indices,
        argument_name="action_visible_token_indices",
        upper_bound=video_seq_len,
    )
    if not visible:
        raise ValueError("No action-visible video tokens were found at runtime.")
    if set(feature_bases_by_layer) != set(layers):
        raise ValueError(
            "Feature bases must cover exactly every replacement layer; "
            f"configured={sorted(feature_bases_by_layer)}, expected={list(layers)}."
        )

    visible_index = torch.tensor(visible, device=representative.device, dtype=torch.long)
    projected_k = list(current_cache_k)
    projected_v = list(current_cache_v)
    layer_audits: list[dict[str, Any]] = []
    for layer in layers:
        pair = feature_bases_by_layer[layer]
        if set(pair) != {"k", "v"}:
            raise ValueError(f"Layer {layer} feature bases must contain exactly K and V.")
        kind_audit: dict[str, Any] = {}
        for kind, current, donor, destination in (
            ("k", current_cache_k[layer], replacement_cache_k[layer], projected_k),
            ("v", current_cache_v[layer], replacement_cache_v[layer], projected_v),
        ):
            basis = pair[kind]
            if not isinstance(basis, torch.Tensor) or basis.ndim != 2:
                raise TypeError(f"Layer {layer} {kind.upper()} basis must be a matrix.")
            if tuple(basis.shape) != (feature_dim, projection_rank):
                raise ValueError(
                    f"Layer {layer} {kind.upper()} basis shape mismatch: "
                    f"{tuple(basis.shape)} != {(feature_dim, projection_rank)}."
                )
            if basis.device != current.device:
                raise ValueError(
                    f"Layer {layer} {kind.upper()} basis device mismatch: "
                    f"{basis.device} != {current.device}."
                )
            if not bool(torch.isfinite(basis).all().item()):
                raise ValueError(f"Layer {layer} {kind.upper()} basis is non-finite.")
            result = current.clone()
            donor_rows = donor.index_select(1, visible_index)
            if projection_rank == 0:
                projected_rows = donor_rows
            elif projection_rank == feature_dim:
                # Preserve the registered full-rank endpoint exactly instead of
                # adding avoidable mixed-precision roundoff from B @ B.T.
                projected_rows = current.index_select(1, visible_index)
            else:
                current_rows = current.index_select(1, visible_index)
                compute_dtype = basis.dtype
                delta = (current_rows - donor_rows).to(dtype=compute_dtype)
                projected_delta = torch.matmul(torch.matmul(delta, basis), basis.T)
                projected_rows = donor_rows + projected_delta.to(dtype=donor_rows.dtype)
            result.index_copy_(1, visible_index, projected_rows)
            destination[layer] = result
            kind_audit[kind] = {
                "basis_shape": list(basis.shape),
                "basis_dtype": str(basis.dtype),
                "finite": True,
            }
        layer_audits.append({"layer": layer, **kind_audit})
    return projected_k, projected_v, {
        "schema_version": 1,
        "mode": "feature_subspace",
        "replacement_video_layers": list(layers),
        "projection_rank": projection_rank,
        "feature_dim": feature_dim,
        "video_seq_len": video_seq_len,
        "action_visible_token_indices": list(visible),
        "action_visible_token_count": len(visible),
        "non_action_visible_token_count": video_seq_len - len(visible),
        "k_v_bases_independent": True,
        "tokens_modified": False,
        "heads_modified": False,
        "shape_preserved": True,
        "layers": layer_audits,
    }


def action_visible_video_token_indices(
    action_attention_mask: torch.Tensor,
    *,
    video_seq_len: int,
) -> tuple[int, ...]:
    """Return video positions visible to at least one action query.

    The attention mask is the runtime source of truth.  This deliberately avoids
    assumptions about image resolution, camera count, or token packing.
    """
    if not isinstance(action_attention_mask, torch.Tensor):
        raise TypeError("`action_attention_mask` must be a tensor.")
    if action_attention_mask.ndim < 2:
        raise ValueError(
            "`action_attention_mask` must have at least query and key dimensions."
        )
    if video_seq_len <= 0:
        raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}.")
    if int(action_attention_mask.shape[-1]) < video_seq_len:
        raise ValueError(
            "Action attention mask has fewer key positions than the video sequence: "
            f"mask={tuple(action_attention_mask.shape)}, video_seq_len={video_seq_len}."
        )
    visible = action_attention_mask[..., :video_seq_len].to(dtype=torch.bool)
    reduce_dims = tuple(range(visible.ndim - 1))
    visible = visible.any(dim=reduce_dims)
    return tuple(
        int(index)
        for index in torch.nonzero(visible, as_tuple=False).flatten().cpu().tolist()
    )


def _normalize_component_indices(
    indices: Sequence[int],
    *,
    argument_name: str,
    upper_bound: int,
) -> tuple[int, ...]:
    requested = list(indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in requested):
        raise TypeError(f"`{argument_name}` must contain only integer indices.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"`{argument_name}` must not contain duplicate indices.")
    invalid = [index for index in requested if index < 0 or index >= upper_bound]
    if invalid:
        raise ValueError(
            f"`{argument_name}` contains out-of-range indices {invalid}; "
            f"valid range is [0, {upper_bound - 1}]."
        )
    return tuple(sorted(requested))


def mix_replacement_video_cache(
    *,
    current_cache_k: Sequence[torch.Tensor],
    current_cache_v: Sequence[torch.Tensor],
    replacement_cache_k: Sequence[torch.Tensor],
    replacement_cache_v: Sequence[torch.Tensor],
    replacement_video_layers: Sequence[int],
    action_visible_token_indices: Sequence[int],
    retained_current_token_indices: Optional[Sequence[int]] = None,
    retained_current_heads_by_layer: Optional[Mapping[int, Sequence[int]]] = None,
    num_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    num_layers: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    """Mix current and donor K/V without changing cache or attention geometry.

    Exactly one granularity is selected: a single runtime-token mask shared by all
    replacement layers, or one head mask per replacement layer.  Positions that no
    action query can attend remain current and therefore are never intervened on.
    """
    validate_matching_video_cache(
        current_cache_k=current_cache_k,
        current_cache_v=current_cache_v,
        replacement_cache_k=replacement_cache_k,
        replacement_cache_v=replacement_cache_v,
        num_layers=num_layers,
    )
    replacement_layers = normalize_video_layer_indices(
        replacement_video_layers,
        argument_name="replacement_video_layers",
        num_layers=num_layers,
    )
    token_mode = retained_current_token_indices is not None
    head_mode = retained_current_heads_by_layer is not None
    if token_mode == head_mode:
        raise ValueError(
            "Provide exactly one of `retained_current_token_indices` or "
            "`retained_current_heads_by_layer`."
        )
    if not replacement_layers:
        raise ValueError("Hybrid cache mixing requires non-empty replacement layers.")

    representative = current_cache_k[replacement_layers[0]]
    if representative.ndim != 3:
        raise ValueError(
            "Video-cache tensors must have runtime shape [batch, tokens, heads*head_dim], "
            f"got {tuple(representative.shape)}."
        )
    video_seq_len = int(representative.shape[1])
    visible = _normalize_component_indices(
        action_visible_token_indices,
        argument_name="action_visible_token_indices",
        upper_bound=video_seq_len,
    )
    if not visible:
        raise ValueError("No action-visible video tokens were found at runtime.")
    visible_set = set(visible)

    mixed_k = list(current_cache_k)
    mixed_v = list(current_cache_v)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "mode": "token" if token_mode else "head",
        "replacement_video_layers": list(replacement_layers),
        "video_seq_len": video_seq_len,
        "action_visible_token_indices": list(visible),
        "action_visible_token_count": len(visible),
        "non_action_visible_token_count": video_seq_len - len(visible),
        "same_mask_for_k_and_v": True,
        "shape_preserved": True,
        "layers": [],
    }

    if token_mode:
        assert retained_current_token_indices is not None
        retained = _normalize_component_indices(
            retained_current_token_indices,
            argument_name="retained_current_token_indices",
            upper_bound=video_seq_len,
        )
        outside_visible = sorted(set(retained) - visible_set)
        if outside_visible:
            raise ValueError(
                "Retained-current token mask contains positions not visible to action "
                f"queries: {outside_visible}."
            )
        replaced = tuple(index for index in visible if index not in set(retained))
        replace_index = torch.tensor(replaced, device=representative.device, dtype=torch.long)
        for layer in replacement_layers:
            for source, destination, donor in (
                (current_cache_k[layer], mixed_k, replacement_cache_k[layer]),
                (current_cache_v[layer], mixed_v, replacement_cache_v[layer]),
            ):
                result = source.clone()
                if replaced:
                    result.index_copy_(1, replace_index, donor.index_select(1, replace_index))
                destination[layer] = result
            audit["layers"].append(
                {
                    "layer": layer,
                    "retained_current_token_count": len(retained),
                    "replacement_token_count": len(replaced),
                }
            )
        audit.update(
            {
                "retained_current_token_indices": list(retained),
                "retained_current_token_count": len(retained),
                "replacement_token_indices": list(replaced),
                "replacement_token_count": len(replaced),
                "num_heads": None,
                "head_dim": None,
            }
        )
    else:
        assert retained_current_heads_by_layer is not None
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            raise ValueError(f"`num_heads` must be a positive integer, got {num_heads!r}.")
        if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
            raise ValueError(f"`head_dim` must be a positive integer, got {head_dim!r}.")
        expected_width = num_heads * head_dim
        configured_layers = set(retained_current_heads_by_layer)
        if configured_layers != set(replacement_layers):
            raise ValueError(
                "Head mask must define exactly every replacement layer; "
                f"configured={sorted(configured_layers)}, expected={list(replacement_layers)}."
            )
        visible_mask = torch.zeros(
            video_seq_len, device=representative.device, dtype=torch.bool
        )
        visible_mask[list(visible)] = True
        normalized_heads: dict[str, list[int]] = {}
        for layer in replacement_layers:
            retained_heads = _normalize_component_indices(
                retained_current_heads_by_layer[layer],
                argument_name=f"retained_current_heads_by_layer[{layer}]",
                upper_bound=num_heads,
            )
            replacement_heads = tuple(
                head for head in range(num_heads) if head not in set(retained_heads)
            )
            head_replacement_mask = torch.zeros(
                num_heads, device=representative.device, dtype=torch.bool
            )
            head_replacement_mask[list(replacement_heads)] = True
            replacement_mask = (
                visible_mask.view(1, video_seq_len, 1, 1)
                & head_replacement_mask.view(1, 1, num_heads, 1)
            )
            for source, destination, donor in (
                (current_cache_k[layer], mixed_k, replacement_cache_k[layer]),
                (current_cache_v[layer], mixed_v, replacement_cache_v[layer]),
            ):
                if int(source.shape[-1]) != expected_width:
                    raise ValueError(
                        f"Layer {layer} cache width {source.shape[-1]} does not equal "
                        f"num_heads*head_dim={num_heads}*{head_dim}={expected_width}."
                    )
                current_view = source.reshape(*source.shape[:-1], num_heads, head_dim)
                donor_view = donor.reshape(*donor.shape[:-1], num_heads, head_dim)
                destination[layer] = torch.where(
                    replacement_mask, donor_view, current_view
                ).reshape_as(source)
            normalized_heads[str(layer)] = list(retained_heads)
            audit["layers"].append(
                {
                    "layer": layer,
                    "retained_current_head_indices": list(retained_heads),
                    "retained_current_head_count": len(retained_heads),
                    "replacement_head_indices": list(replacement_heads),
                    "replacement_head_count": len(replacement_heads),
                }
            )
        audit.update(
            {
                "retained_current_heads_by_layer": normalized_heads,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "retained_current_token_indices": None,
            }
        )
    return mixed_k, mixed_v, audit


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
