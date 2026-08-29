"""Frozen numerical definitions for the Salvage-A action-sensitive estimator."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch


ACTION_SHAPE = (32, 7)
ACTION_DIM = 224
INTERPOLATION_LAMBDAS = (0.5, 1.0)
PROBE_SEEDS = {
    0.5: (845_050_001, 845_050_002),
    1.0: (845_100_001, 845_100_002),
}
PROBES_PER_LAMBDA = 2
EXPECTED_VJPS_PER_STATE = 4


def rademacher_probe(seed: int) -> torch.Tensor:
    """Return one deterministic CPU float32 Rademacher action probe."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bits = torch.randint(0, 2, ACTION_SHAPE, generator=generator)
    return bits.to(torch.float32).mul_(2.0).sub_(1.0)


def probe_sha256(probe: torch.Tensor) -> str:
    value = probe.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def frozen_probe_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for interpolation_lambda in INTERPOLATION_LAMBDAS:
        for probe_index, seed in enumerate(PROBE_SEEDS[interpolation_lambda]):
            probe = rademacher_probe(seed)
            records.append(
                {
                    "interpolation_lambda": interpolation_lambda,
                    "probe_index": probe_index,
                    "seed": seed,
                    "shape": list(probe.shape),
                    "sha256": probe_sha256(probe),
                }
            )
    return records


def covariance_contribution(gradient: torch.Tensor) -> torch.Tensor:
    """Compute G^T G after summing Hutchinson rows over token positions."""

    if gradient.ndim != 3 or gradient.shape[0] != 1:
        raise ValueError(f"Expected VJP shape [1,T,D], got {list(gradient.shape)}.")
    values = gradient.detach().reshape(-1, gradient.shape[-1]).float()
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("Action-sensitive VJP contains non-finite values.")
    return values.T @ values


def accumulate_covariance_(target: torch.Tensor, gradient: torch.Tensor) -> None:
    """Accumulate one probe's feature covariance without retaining its graph."""

    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("Covariance accumulator must be square.")
    values = gradient.detach().reshape(-1, gradient.shape[-1]).float()
    if values.shape[1] != target.shape[0]:
        raise ValueError(
            f"VJP feature dimension {values.shape[1]} != {target.shape[0]}."
        )
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("Action-sensitive VJP contains non-finite values.")
    target.addmm_(values.T, values)


def validate_probe_registry(records: Iterable[dict[str, object]]) -> None:
    observed = list(records)
    expected = frozen_probe_records()
    if observed != expected:
        raise ValueError("Action-sensitive probe registry disagrees with frozen seeds.")
