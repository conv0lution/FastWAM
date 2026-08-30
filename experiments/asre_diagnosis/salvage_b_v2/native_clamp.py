"""Frozen exogenous clamps for the native Fast-WAM joint K/V nodes.

This module never evaluates attention.  It is called from the single stock
``MoT._forward_joint_layer`` immediately after native video K/V construction
and immediately before the stock concatenated joint-attention call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from .definitions import FEATURE_DIM, LATE_LAYERS, PREFIX_TOKENS


PrefixKey = tuple[int, int, str]


@dataclass
class FrozenPrefixTrajectory:
    """CPU-owned native stock prefix values indexed by step/layer/kind."""

    values: dict[PrefixKey, torch.Tensor] = field(default_factory=dict)
    num_layers: int = 30
    prefix_tokens: int = PREFIX_TOKENS
    _expected_layer: int = 0
    _step: int = 0

    def capture_hook(
        self,
        layer: int,
        k_video: torch.Tensor,
        v_video: torch.Tensor,
        prefix_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer != self._expected_layer:
            raise RuntimeError(
                "Native joint-layer order drifted while capturing a frozen clamp: "
                f"expected {self._expected_layer}, observed {layer}."
            )
        if prefix_seq_len != self.prefix_tokens:
            raise ValueError(
                f"Native prefix length drifted: {prefix_seq_len} != {self.prefix_tokens}."
            )
        if k_video.ndim != 3 or v_video.shape != k_video.shape:
            raise ValueError("Native joint K/V must be matching [B,S,D] tensors.")
        if int(k_video.shape[-1]) != FEATURE_DIM:
            raise ValueError(
                f"Native feature dimension drifted: {k_video.shape[-1]} != {FEATURE_DIM}."
            )
        if layer in LATE_LAYERS:
            self.values[(self._step, layer, "k")] = (
                k_video[:, :prefix_seq_len].detach().to("cpu").clone().contiguous()
            )
            self.values[(self._step, layer, "v")] = (
                v_video[:, :prefix_seq_len].detach().to("cpu").clone().contiguous()
            )
        self._expected_layer = (layer + 1) % self.num_layers
        if self._expected_layer == 0:
            self._step += 1
        # Capturing is observational: return the exact same objects.
        return k_video, v_video

    def validate_complete(self, *, expected_steps: int) -> None:
        expected = {
            (step, layer, kind)
            for step in range(expected_steps)
            for layer in LATE_LAYERS
            for kind in ("k", "v")
        }
        if set(self.values) != expected:
            missing = sorted(expected - set(self.values))[:8]
            extra = sorted(set(self.values) - expected)[:8]
            raise ValueError(
                "Frozen native prefix trajectory is incomplete: "
                f"missing={missing}, extra={extra}."
            )
        if self._expected_layer != 0 or self._step != expected_steps:
            raise ValueError("Frozen native prefix trajectory ended mid-layer stack.")


@dataclass
class NativePrefixClamp:
    """Apply Wrong + frozen-basis projection(Current-Wrong) at every late layer."""

    current: FrozenPrefixTrajectory
    wrong: FrozenPrefixTrajectory
    rank: int
    bases_by_layer: Mapping[int, Mapping[str, torch.Tensor]] | None = None
    num_layers: int = 30
    prefix_tokens: int = PREFIX_TOKENS
    _expected_layer: int = 0
    _step: int = 0
    preclamp_prefixes: dict[PrefixKey, torch.Tensor] = field(default_factory=dict)
    replacement_object_ids: dict[tuple[int, int], tuple[int, int]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.rank < 0 or self.rank > FEATURE_DIM:
            raise ValueError(f"Clamp rank must be in [0,{FEATURE_DIM}], got {self.rank}.")
        if self.rank not in (0, FEATURE_DIM):
            if self.bases_by_layer is None or set(self.bases_by_layer) != set(LATE_LAYERS):
                raise ValueError("Projected native clamps require all 15 frozen K/V bases.")

    def _target(
        self,
        *,
        layer: int,
        kind: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (self._step, layer, kind)
        current = self.current.values[key].to(device=device, dtype=dtype)
        wrong = self.wrong.values[key].to(device=device, dtype=dtype)
        if self.rank == 0:
            return wrong
        if self.rank == FEATURE_DIM:
            return current
        assert self.bases_by_layer is not None
        basis = self.bases_by_layer[layer][kind].to(device=device, dtype=dtype)
        if tuple(basis.shape) != (FEATURE_DIM, self.rank):
            raise ValueError(
                f"Layer {layer} {kind.upper()} basis shape drifted: {tuple(basis.shape)}."
            )
        delta = current - wrong
        return wrong + torch.matmul(torch.matmul(delta, basis), basis.T)

    def hook(
        self,
        layer: int,
        k_video: torch.Tensor,
        v_video: torch.Tensor,
        prefix_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer != self._expected_layer:
            raise RuntimeError(
                "Native clamp layer order drifted: "
                f"expected {self._expected_layer}, observed {layer}."
            )
        if prefix_seq_len != self.prefix_tokens:
            raise ValueError("Native clamp reached an unexpected prefix length.")
        if layer in LATE_LAYERS:
            for kind, source in (("k", k_video), ("v", v_video)):
                self.preclamp_prefixes[(self._step, layer, kind)] = (
                    source[:, :prefix_seq_len].detach().to("cpu").clone()
                )
            target_k = self._target(
                layer=layer, kind="k", device=k_video.device, dtype=k_video.dtype
            )
            target_v = self._target(
                layer=layer, kind="v", device=v_video.device, dtype=v_video.dtype
            )
            # Prefix only. Future rows retain the values produced by this exact
            # native intervened forward pass.
            k_video = torch.cat((target_k, k_video[:, prefix_seq_len:]), dim=1)
            v_video = torch.cat((target_v, v_video[:, prefix_seq_len:]), dim=1)
            self.replacement_object_ids[(self._step, layer)] = (
                id(k_video),
                id(v_video),
            )
        self._expected_layer = (layer + 1) % self.num_layers
        if self._expected_layer == 0:
            self._step += 1
        return k_video, v_video

    def validate_complete(self, *, expected_steps: int) -> None:
        if self._step != expected_steps or self._expected_layer != 0:
            raise ValueError("Native clamp did not traverse the complete joint stack.")
        expected_replacements = expected_steps * len(LATE_LAYERS)
        if len(self.replacement_object_ids) != expected_replacements:
            raise ValueError(
                "Native clamp did not replace every registered late-layer node."
            )
