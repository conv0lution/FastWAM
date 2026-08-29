"""Frozen Round-4C definitions and energy-label validation."""

from __future__ import annotations

from typing import Any, Mapping


FEATURE_DIM = 3072
RANKS = (36, 97, 170)
REGISTERED_HELDOUT_ENERGY = {36: 0.5048, 97: 0.7003, 170: 0.8005}
REGISTERED_R256_HELDOUT_ENERGY = 0.8617280991735757
ENERGY_TOLERANCE = 5e-4
ROUND4B_SOURCE_COMMIT = "e108783571105b0bf6b5f02d53fd8b357514779f"
CUMULATIVE_ENERGY_COMMIT = "b533e29df03464860bf4edda1e74e1ecbe319174"
ROUND4A_TAG = "ASRE-round4a-token-head-sparsity"
ROUND4A_COMMIT = "094994e4e6f936c84fde3e1023ad998fabd17bcf"
EXPECTED_BASIS_SHA256 = "1a3fcd8ed44e012abbfdf8da119b82e76baf01755a3a8472eb470845c7122d02"
EXPECTED_SPLIT_SHA256 = "38a6cb79e3006fd8b1bacc24691df6a79751d323fdff8cc3b5912601f217024d"
EXPECTED_DIAGNOSTICS_SHA256 = "f7026749f0b44b3533f976ef28a3cdf43d114f511b7296ca7a85942e80cbd2bb"
EXPECTED_ROUND4B_SUMMARY_SHA256 = "b8c1d51dce82ed74f4d6d8300ff33f682f1b2df571aa5d3f8df1527d0758df56"
EXPECTED_ENERGY_MANIFEST_SHA256 = "a4b83c2fae9d76ac927a2db11a2232131f264dd2a0b2d009a77a853d5bda3429"
EXPECTED_CANDIDATES_SHA256 = "dc6825a02ff5347a0d0140132a93075a96af0ec48b1809a6de8ff9f1cfbf4e7d"
CONDITIONS = ("current_all", "wrong_all", "svd_r36", "svd_r97", "svd_r170")
DISPLAY = {
    "current_all": "Current (100% energy)",
    "wrong_all": "Wrong (0% energy)",
    "svd_r36": "SVD-36 (50.48%)",
    "svd_r97": "SVD-97 (70.03%)",
    "svd_r170": "SVD-170 (80.05%)",
}
WAVES = {1: (0, 1, 2, 3), 2: (4,)}


def validate_energy_candidates(payload: Mapping[str, Any]) -> dict[int, float]:
    """Fail closed unless the frozen candidate file reproduces registered ranks."""
    expected_keys = ("candidate_r50", "candidate_r70", "candidate_r80")
    if set(payload) != set(expected_keys):
        raise ValueError("Round-4C candidate artifact must define exactly r50/r70/r80.")
    observed: dict[int, float] = {}
    for key, rank in zip(expected_keys, RANKS):
        record = payload[key]
        if not isinstance(record, Mapping) or int(record.get("rank", -1)) != rank:
            raise ValueError(f"Round-4C frozen candidate rank drifted for {key}.")
        energy = float(record.get("heldout_global_energy", float("nan")))
        if abs(energy - REGISTERED_HELDOUT_ENERGY[rank]) > ENERGY_TOLERANCE:
            raise ValueError(
                f"Round-4C registered held-out energy drifted at rank {rank}: {energy}."
            )
        if abs(float(record.get("rank_fraction", -1.0)) - rank / FEATURE_DIM) > 1e-12:
            raise ValueError(f"Round-4C rank fraction drifted at rank {rank}.")
        observed[rank] = energy
    return observed
