"""Round-4B basis artifact definitions, validation, and runtime loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments.asre_diagnosis.common import ROUND4B_PROTOCOL, sha256_file


LATE_LAYERS = tuple(range(15, 30))
EXPECTED_FEATURE_DIM = 3072
RANKS = (256, 768, 1536)
ROUND4C_SVD_RANKS = (36, 97, 170)
MAX_RANK = max(RANKS)
RANDOM_SEED = 4205


def stable_matrix_seed(kind: str, layer: int, tensor_kind: str) -> int:
    digest = hashlib.sha256(
        f"round4b-basis\0{RANDOM_SEED}\0{kind}\0{layer}\0{tensor_kind}".encode(
            "ascii"
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def validate_basis_manifest(
    path: Path, *, expected_sha256: str | None = None, verify_files: bool = True
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Round-4B basis manifest SHA256 mismatch.")
    payload = _read_json(path)
    expected = {
        "artifact_type": "asre_round4b_basis_manifest",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "ranks": list(RANKS),
        "max_rank": MAX_RANK,
        "late_layers": list(LATE_LAYERS),
        "k_v_fitted_separately": True,
        "uncentered_svd": True,
        "random_basis_nested": True,
        "random_seed": RANDOM_SEED,
    }
    mismatch = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Round-4B basis manifest mismatch: {mismatch}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"svd", "random"}:
        raise ValueError("Round-4B manifest must define SVD and random artifacts.")
    layout = payload.get("runtime_layout")
    if not isinstance(layout, dict):
        raise ValueError("Round-4B basis manifest lacks runtime cache layout.")
    required_layout = {
        "num_layers": 30,
        "video_seq_len": 98,
        "action_visible_token_count": 98,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "num_heads": 24,
        "head_dim": 128,
    }
    if any(layout.get(key) != value for key, value in required_layout.items()):
        raise ValueError(f"Unexpected Round-4B runtime layout: {layout}")
    if layout.get("action_visible_token_indices") != list(range(98)):
        raise ValueError("Round-4B requires exactly action-visible token indices 0..97.")
    for basis_kind in ("svd", "random"):
        by_layer = artifacts[basis_kind]
        if set(map(int, by_layer)) != set(LATE_LAYERS):
            raise ValueError(f"Incomplete {basis_kind} layer coverage.")
        for layer in LATE_LAYERS:
            pair = by_layer[str(layer)]
            if set(pair) != {"k", "v"}:
                raise ValueError(f"Incomplete basis pair: {basis_kind}/layer{layer}.")
            for tensor_kind in ("k", "v"):
                record = pair[tensor_kind]
                if record.get("shape") != [EXPECTED_FEATURE_DIM, MAX_RANK]:
                    raise ValueError("Round-4B basis artifact shape mismatch.")
                artifact = Path(str(record["path"])).resolve()
                if verify_files:
                    if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
                        raise ValueError(f"Basis artifact unavailable or drifted: {artifact}")
    return payload


@dataclass(frozen=True)
class RuntimeBasisSpec:
    basis_kind: str
    rank: int
    feature_dim: int
    bases_by_layer: dict[int, dict[str, torch.Tensor]]
    manifest_path: Path
    manifest_sha256: str
    expected_video_cache_layout: dict[str, Any]

    def inference_kwargs(self) -> dict[str, Any]:
        return {
            "feature_projection_bases_by_layer": self.bases_by_layer,
            "feature_projection_rank": self.rank,
            "expected_video_cache_layout": self.expected_video_cache_layout,
        }


_RUNTIME_CACHE: dict[tuple[str, str, str, int, str, str], RuntimeBasisSpec] = {}


def load_runtime_basis(
    *,
    manifest_path: Path,
    expected_sha256: str,
    basis_kind: str,
    rank: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> RuntimeBasisSpec:
    if basis_kind not in {"svd", "random"}:
        raise ValueError(f"Unknown Round-4B basis kind: {basis_kind!r}.")
    allowed_ranks = RANKS if basis_kind == "random" else (*ROUND4C_SVD_RANKS, *RANKS)
    if rank not in allowed_ranks:
        raise ValueError(
            f"Frozen {basis_kind.upper()} basis rank must be one of "
            f"{allowed_ranks}, got {rank}."
        )
    path = manifest_path.expanduser().resolve()
    key = (str(path), expected_sha256, basis_kind, rank, str(device), str(dtype))
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]
    manifest = validate_basis_manifest(path, expected_sha256=expected_sha256)
    bases: dict[int, dict[str, torch.Tensor]] = {}
    for layer in LATE_LAYERS:
        pair: dict[str, torch.Tensor] = {}
        for tensor_kind in ("k", "v"):
            record = manifest["artifacts"][basis_kind][str(layer)][tensor_kind]
            artifact = Path(str(record["path"])).resolve()
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            tensor = payload.get("basis") if isinstance(payload, Mapping) else None
            if not torch.is_tensor(tensor) or list(tensor.shape) != record["shape"]:
                raise ValueError(f"Malformed basis tensor: {artifact}")
            pair[tensor_kind] = tensor[:, :rank].to(
                device=device, dtype=dtype, non_blocking=True
            ).contiguous()
        bases[layer] = pair
    spec = RuntimeBasisSpec(
        basis_kind=basis_kind,
        rank=rank,
        feature_dim=EXPECTED_FEATURE_DIM,
        bases_by_layer=bases,
        manifest_path=path,
        manifest_sha256=expected_sha256,
        expected_video_cache_layout={
            key: value
            for key, value in manifest["runtime_layout"].items()
            if key != "feature_dim"
        },
    )
    _RUNTIME_CACHE[key] = spec
    return spec
