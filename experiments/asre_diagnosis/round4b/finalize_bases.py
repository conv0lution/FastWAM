"""Diagonalize Round-4B Gram matrices and freeze nested SVD/random bases."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4B_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
    MAX_RANK,
    RANDOM_SEED,
    RANKS,
    stable_matrix_seed,
    validate_basis_manifest,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _orthogonality_error(basis: torch.Tensor) -> float:
    identity = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
    return float((basis.T @ basis - identity).abs().max().item())


def _effective_rank(eigenvalues: torch.Tensor) -> float:
    values = eigenvalues.clamp_min(0)
    total = values.sum()
    if float(total.item()) <= 0:
        return 0.0
    probabilities = values / total
    entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()
    return float(entropy.exp().item())


def finalize(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Round-4B eigendecomposition requires CUDA.")
    output = args.output.resolve()
    if output.exists():
        validate_basis_manifest(output)
        return output
    fit_dir = args.fit_dir.resolve()
    split = args.split.resolve()
    split_sha = sha256_file(split)
    reports = [_read(fit_dir / f"fit_shard{index}.json") for index in range(4)]
    if any(
        report.get("status") != "complete"
        or report.get("phase") != "fit"
        or report.get("split_sha256") != split_sha
        for report in reports
    ):
        raise ValueError("Round-4B fit shards are missing or incompatible.")
    covered = [layer for report in reports for layer in report["layers"]]
    if sorted(covered) != list(LATE_LAYERS) or len(set(covered)) != len(covered):
        raise ValueError("Round-4B fit shards do not partition late layers exactly once.")
    layout_keys = (
        "num_layers",
        "video_seq_len",
        "action_visible_token_indices",
        "action_visible_token_count",
        "feature_dim",
        "num_heads",
        "head_dim",
        "tokens_per_frame",
        "video_grid_size",
        "input_image_shape",
    )
    runtime_layout = {
        key: reports[0]["runtime_layout"][key] for key in layout_keys
    }
    if any(
        {key: report["runtime_layout"][key] for key in layout_keys} != runtime_layout
        for report in reports[1:]
    ):
        raise ValueError("Runtime cache layout differs across fit shards.")
    artifact_root = output.parent / "artifacts"
    artifacts: dict[str, dict[str, dict[str, Any]]] = {
        "svd": {},
        "random": {},
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    report_for_layer = {
        int(layer): report for report in reports for layer in report["layers"]
    }
    for layer in LATE_LAYERS:
        artifacts["svd"][str(layer)] = {}
        artifacts["random"][str(layer)] = {}
        for kind in ("k", "v"):
            gram_path = fit_dir / "grams" / f"layer{layer:02d}_{kind}.pt"
            recorded = report_for_layer[layer]["artifacts"][f"layer{layer:02d}_{kind}"]
            if Path(recorded["path"]).resolve() != gram_path.resolve() or sha256_file(
                gram_path
            ) != recorded["sha256"]:
                raise ValueError(f"Gram artifact drifted: {gram_path}")
            gram_payload = torch.load(gram_path, map_location="cpu", weights_only=False)
            gram = gram_payload["gram"]
            if list(gram.shape) != [EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_DIM]:
                raise ValueError(f"Malformed Gram matrix: {gram_path}")
            gram = gram.to(device="cuda:0", dtype=torch.float32)
            gram = (gram + gram.T) * 0.5
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            order = torch.arange(EXPECTED_FEATURE_DIM - 1, -1, -1, device=gram.device)
            eigenvalues = eigenvalues.index_select(0, order).clamp_min(0)
            basis = eigenvectors.index_select(1, order[:MAX_RANK]).contiguous()
            singular_values = eigenvalues.sqrt()
            total = float(eigenvalues.sum().item())
            captured = {
                str(rank): float(eigenvalues[:rank].sum().item() / total)
                for rank in RANKS
            }
            gaps = {
                str(rank): (
                    None
                    if float(eigenvalues[rank].item()) <= 0
                    else float(eigenvalues[rank - 1].item() / eigenvalues[rank].item())
                )
                for rank in RANKS
            }
            svd_path = artifact_root / "svd" / f"layer{layer:02d}_{kind}.pt"
            _atomic_save(
                svd_path,
                {
                    "artifact_type": "asre_round4b_svd_basis",
                    "schema_version": 1,
                    "protocol": ROUND4B_PROTOCOL,
                    "layer": layer,
                    "tensor_kind": kind,
                    "feature_dim": EXPECTED_FEATURE_DIM,
                    "max_rank": MAX_RANK,
                    "uncentered": True,
                    "split_sha256": split_sha,
                    "basis": basis.cpu(),
                    "singular_values": singular_values.cpu(),
                },
            )
            artifacts["svd"][str(layer)][kind] = {
                "path": str(svd_path),
                "sha256": sha256_file(svd_path),
                "shape": [EXPECTED_FEATURE_DIM, MAX_RANK],
                "dtype": "torch.float32",
            }
            diagnostics[f"layer{layer:02d}_{kind}"] = {
                "fit_rows": int(gram_payload["rows"]),
                "fit_total_energy": total,
                "fit_captured_fraction": captured,
                "effective_rank": _effective_rank(eigenvalues),
                "spectral_gap_ratio": gaps,
                "largest_singular_values": singular_values[:64].cpu().tolist(),
                "smallest_singular_value": float(singular_values[-1].item()),
                "svd_basis_orthogonality_max_abs": _orthogonality_error(basis),
            }
            del gram, eigenvalues, eigenvectors, basis, singular_values
            torch.cuda.empty_cache()

            seed = stable_matrix_seed("random", layer, kind)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            gaussian = torch.randn(
                EXPECTED_FEATURE_DIM, MAX_RANK, generator=generator, dtype=torch.float32
            ).to("cuda:0")
            random_basis, _ = torch.linalg.qr(gaussian, mode="reduced")
            random_path = artifact_root / "random" / f"layer{layer:02d}_{kind}.pt"
            _atomic_save(
                random_path,
                {
                    "artifact_type": "asre_round4b_random_orthogonal_basis",
                    "schema_version": 1,
                    "protocol": ROUND4B_PROTOCOL,
                    "layer": layer,
                    "tensor_kind": kind,
                    "feature_dim": EXPECTED_FEATURE_DIM,
                    "max_rank": MAX_RANK,
                    "seed": seed,
                    "nested_prefix_ranks": list(RANKS),
                    "basis": random_basis.cpu(),
                },
            )
            artifacts["random"][str(layer)][kind] = {
                "path": str(random_path),
                "sha256": sha256_file(random_path),
                "shape": [EXPECTED_FEATURE_DIM, MAX_RANK],
                "dtype": "torch.float32",
                "seed": seed,
                "orthogonality_max_abs": _orthogonality_error(random_basis),
            }
            del gaussian, random_basis
            torch.cuda.empty_cache()
            print(f"[Round4B finalize] layer {layer} {kind.upper()} complete", flush=True)
    payload = {
        "artifact_type": "asre_round4b_basis_manifest",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "feature_dim": EXPECTED_FEATURE_DIM,
        "ranks": list(RANKS),
        "max_rank": MAX_RANK,
        "late_layers": list(LATE_LAYERS),
        "k_v_fitted_separately": True,
        "uncentered_svd": True,
        "random_basis_nested": True,
        "random_seed": RANDOM_SEED,
        "runtime_layout": runtime_layout,
        "split_path": str(split),
        "split_sha256": split_sha,
        "fit_shard_reports": [
            {
                "path": str(fit_dir / f"fit_shard{index}.json"),
                "sha256": sha256_file(fit_dir / f"fit_shard{index}.json"),
            }
            for index in range(4)
        ],
        "fit_diagnostics": diagnostics,
        "artifacts": artifacts,
    }
    atomic_write_json(output, payload)
    validate_basis_manifest(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    path = finalize(parser.parse_args())
    print(f"Frozen Round-4B basis manifest: {path}")


if __name__ == "__main__":
    main()
