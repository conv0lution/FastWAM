"""Frozen Salvage-A basis definitions, validation, and runtime loading.

All three basis families deliberately share one artifact layout.  The online
evaluator therefore cannot accidentally apply a different reconstruction path
to the matched SVD, ActionAware, and preregistered random controls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments.asre_diagnosis.common import SALVAGE_A_PROTOCOL, sha256_file


EXPECTED_FEATURE_DIM = 3072
LATE_LAYERS = tuple(range(15, 30))
RANKS = (36, 97)
MAX_RANK = max(RANKS)
BASIS_KINDS = ("svd", "actionaware", "random")
TENSOR_KINDS = ("k", "v")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _validate_runtime_layout(layout: Any) -> dict[str, Any]:
    if not isinstance(layout, dict):
        raise ValueError("Salvage-A basis manifest lacks runtime cache layout.")
    required = {
        "num_layers": 30,
        "action_visible_token_count": 98,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "num_heads": 24,
        "head_dim": 128,
    }
    mismatch = {
        key: {"observed": layout.get(key), "expected": value}
        for key, value in required.items()
        if layout.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Unexpected Salvage-A runtime layout: {mismatch}")
    if layout.get("action_visible_token_indices") != list(range(98)):
        raise ValueError("Salvage-A requires action-visible token indices 0..97.")
    video_seq_len = layout.get("video_seq_len")
    if video_seq_len is not None and int(video_seq_len) != 98:
        raise ValueError("Salvage-A runtime video sequence length must be 98.")
    return dict(layout)


def validate_basis_payload(
    payload: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Validate the registered three-family, 30-matrix basis inventory."""

    expected = {
        "artifact_type": "asre_salvage_a_basis_manifest",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "ranks": list(RANKS),
        "max_rank": MAX_RANK,
        "late_layers": list(LATE_LAYERS),
        "k_v_fitted_separately": True,
        "uncentered_svd": True,
        "actionaware_full_calibration_basis": True,
        "differentiable_path_gate_passed": True,
        "random_basis_nested": True,
        "random_prefix_reused_pre_outcome": True,
    }
    mismatch = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Salvage-A basis manifest mismatch: {mismatch}")
    differentiable_sha = payload.get("differentiable_path_report_sha256")
    if (
        not isinstance(differentiable_sha, str)
        or len(differentiable_sha) != 64
        or any(character not in "0123456789abcdef" for character in differentiable_sha)
    ):
        raise ValueError("Salvage-A basis manifest lacks a valid differentiable-path SHA256.")
    _validate_runtime_layout(payload.get("runtime_layout"))

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(BASIS_KINDS):
        raise ValueError(
            "Salvage-A basis manifest must contain exactly svd/actionaware/random."
        )
    for basis_kind in BASIS_KINDS:
        by_layer = artifacts[basis_kind]
        if not isinstance(by_layer, dict) or set(map(int, by_layer)) != set(
            LATE_LAYERS
        ):
            raise ValueError(f"Incomplete Salvage-A {basis_kind} layer coverage.")
        for layer in LATE_LAYERS:
            pair = by_layer.get(str(layer))
            if not isinstance(pair, dict) or set(pair) != set(TENSOR_KINDS):
                raise ValueError(
                    f"Incomplete Salvage-A basis pair: {basis_kind}/layer{layer}."
                )
            for tensor_kind in TENSOR_KINDS:
                record = pair[tensor_kind]
                if not isinstance(record, dict):
                    raise ValueError("Salvage-A basis record must be a JSON object.")
                if record.get("shape") != [EXPECTED_FEATURE_DIM, MAX_RANK]:
                    raise ValueError(
                        f"Salvage-A basis shape mismatch: "
                        f"{basis_kind}/layer{layer}/{tensor_kind}."
                    )
                if record.get("dtype") != "torch.float32":
                    raise ValueError("Salvage-A frozen bases must use torch.float32.")
                artifact = Path(str(record.get("path", ""))).expanduser().resolve()
                if verify_files and (
                    not artifact.is_file()
                    or sha256_file(artifact) != record.get("sha256")
                ):
                    raise ValueError(
                        f"Basis artifact unavailable or drifted: {artifact}"
                    )
    return dict(payload)


def validate_basis_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Salvage-A basis manifest SHA256 mismatch.")
    return validate_basis_payload(_read_json(path), verify_files=verify_files)


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


_RUNTIME_CACHE: dict[
    tuple[str, str, str, int, str, str], RuntimeBasisSpec
] = {}


def load_runtime_basis(
    manifest_path: Path,
    expected_sha256: str,
    basis_kind: str,
    rank: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> RuntimeBasisSpec:
    """Load one registered nested prefix for online reconstruction."""

    if basis_kind not in BASIS_KINDS:
        raise ValueError(f"Unknown Salvage-A basis kind: {basis_kind!r}.")
    if int(rank) not in RANKS:
        raise ValueError(f"Salvage-A basis rank must be one of {RANKS}, got {rank}.")
    path = manifest_path.expanduser().resolve()
    key = (str(path), expected_sha256, basis_kind, int(rank), str(device), str(dtype))
    cached = _RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached
    manifest = validate_basis_manifest(path, expected_sha256=expected_sha256)
    bases: dict[int, dict[str, torch.Tensor]] = {}
    for layer in LATE_LAYERS:
        pair: dict[str, torch.Tensor] = {}
        for tensor_kind in TENSOR_KINDS:
            record = manifest["artifacts"][basis_kind][str(layer)][tensor_kind]
            artifact = Path(str(record["path"])).expanduser().resolve()
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            tensor = payload.get("basis") if isinstance(payload, Mapping) else None
            if (
                not torch.is_tensor(tensor)
                or list(tensor.shape) != [EXPECTED_FEATURE_DIM, MAX_RANK]
                or tensor.dtype != torch.float32
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise ValueError(f"Malformed Salvage-A basis tensor: {artifact}")
            pair[tensor_kind] = tensor[:, : int(rank)].to(
                device=device, dtype=dtype, non_blocking=True
            ).contiguous()
        bases[layer] = pair
    expected_layout = {
        key: value
        for key, value in manifest["runtime_layout"].items()
        if key != "feature_dim"
    }
    spec = RuntimeBasisSpec(
        basis_kind=basis_kind,
        rank=int(rank),
        feature_dim=EXPECTED_FEATURE_DIM,
        bases_by_layer=bases,
        manifest_path=path,
        manifest_sha256=expected_sha256,
        expected_video_cache_layout=expected_layout,
    )
    _RUNTIME_CACHE[key] = spec
    return spec
