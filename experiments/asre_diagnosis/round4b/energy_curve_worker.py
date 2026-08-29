"""Collect frozen held-out Grams and recover complete Round-4B energy coordinates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4B_PROTOCOL,
    atomic_write_json,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.round3b.donor import OnlineDonorBundle  # noqa: E402
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
    MAX_RANK,
    RANKS,
    validate_basis_manifest,
)
from experiments.asre_diagnosis.round4b.energy_curve_definitions import (  # noqa: E402
    COLLECT_STAGE,
    RECOVER_STAGE,
)
from experiments.asre_diagnosis.round4b.fit_worker import (  # noqa: E402
    EARLY_DISABLED,
    _atomic_torch_save,
    _load_donor_image,
    _load_model,
    _read,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


def _physical_gpu() -> int:
    physical = os.environ.get("ASRE_ROUND4B_ENERGY_PHYSICAL_GPU")
    if physical is None or os.environ.get("CUDA_VISIBLE_DEVICES") != physical:
        raise RuntimeError("Energy worker requires an auditable isolated physical GPU.")
    if not torch.cuda.is_available():
        raise RuntimeError("Energy worker requires CUDA.")
    return int(physical)


def _validate_worker_index(index: int) -> tuple[int, ...]:
    if index not in range(4):
        raise ValueError("worker-index must be 0..3.")
    return LATE_LAYERS[index::4]


def collect_heldout_grams(args: argparse.Namespace) -> dict[str, Any]:
    """Run cache-only inference on frozen held-out states and persist ΔZ Grams."""

    physical_gpu = _physical_gpu()
    layers = _validate_worker_index(args.worker_index)
    split_path = args.split.resolve()
    split = _read(split_path)
    if split.get("protocol") != ROUND4B_PROTOCOL:
        raise ValueError("Incompatible Round-4B split manifest.")
    sample_ids = list(split["holdout_sample_ids"])
    if len(sample_ids) != 100 or int(split.get("holdout_episode_clusters", -1)) != 20:
        raise ValueError("Energy collection requires the frozen 100-state/20-episode holdout.")
    source_path = Path(str(split["source_manifest_path"])).resolve()
    records = load_manifest(source_path)
    by_id = {str(record["sample_id"]): record for record in records}
    if any(identifier not in by_id for identifier in sample_ids):
        raise ValueError("Held-out sample is absent from the frozen source state bank.")
    basis_path = args.basis_manifest.resolve()
    basis = validate_basis_manifest(basis_path)
    if basis["split_sha256"] != sha256_file(split_path):
        raise ValueError("Basis/split provenance mismatch.")
    bundle = OnlineDonorBundle.load(
        mapping_path=args.donor_mapping.resolve(),
        observation_manifest_path=args.donor_manifest.resolve(),
        observation_root=args.donor_root.resolve(),
    )
    checkpoint = args.checkpoint.resolve()
    set_global_seed(42, get_worker_init_fn=False)
    model, _cfg = _load_model(checkpoint)
    grams = {
        kind: {
            layer: torch.zeros(
                (EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_DIM),
                device="cuda:0",
                dtype=torch.float32,
            )
            for layer in layers
        }
        for kind in ("k", "v")
    }
    layout: dict[str, Any] | None = None
    row_count = 0
    state_root = source_path.parent
    with torch.inference_mode():
        for position, identifier in enumerate(sample_ids, start=1):
            record = by_id[identifier]
            sample_path = Path(str(record["sample_path"]))
            if not sample_path.is_absolute():
                sample_path = state_root / sample_path
            sample = torch.load(sample_path, map_location="cpu", weights_only=False)
            infer_kwargs = dict(sample["infer_action_kwargs"])
            donor = _load_donor_image(
                bundle,
                task_id=int(record["task_id"]),
                episode_id=int(record["episode_id"]),
            ).to(dtype=infer_kwargs["input_image"].dtype)
            result = model.infer_action(
                **infer_kwargs,
                disabled_video_layers=EARLY_DISABLED,
                replacement_input_image=donor,
                replacement_video_layers=LATE_LAYERS,
                return_video_cache_deltas=True,
                video_cache_delta_layers=layers,
                cache_only=True,
                compile_action_infer=False,
            )
            observed_layout = dict(result["video_cache_layout"])
            if (
                observed_layout["feature_dim"] != EXPECTED_FEATURE_DIM
                or observed_layout["action_visible_token_count"] != 98
                or observed_layout["num_heads"] != 24
                or observed_layout["head_dim"] != 128
            ):
                raise ValueError(f"Unexpected runtime cache layout: {observed_layout}")
            if layout is None:
                layout = observed_layout
            elif observed_layout != layout:
                raise ValueError("Runtime cache layout drifted during held-out collection.")
            deltas = result["video_cache_deltas"]
            row_count += int(observed_layout["action_visible_token_count"])
            for kind in ("k", "v"):
                for layer in layers:
                    values = deltas[kind][layer].reshape(-1, EXPECTED_FEATURE_DIM).float()
                    if not bool(torch.isfinite(values).all().item()):
                        raise ValueError(f"Nonfinite held-out ΔZ for {identifier}.")
                    grams[kind][layer].addmm_(values.T, values)
            del result, deltas
            if position % 25 == 0:
                print(
                    f"[Round4B energy collect shard {args.worker_index}] "
                    f"{position}/{len(sample_ids)} states",
                    flush=True,
                )
    output_dir = args.output_dir.resolve()
    gram_root = output_dir / "grams"
    artifacts: dict[str, dict[str, Any]] = {}
    for kind in ("k", "v"):
        for layer in layers:
            matrix = f"layer{layer:02d}_{kind}"
            path = gram_root / f"{matrix}.pt"
            gram = grams[kind][layer].cpu()
            _atomic_torch_save(
                path,
                {
                    "artifact_type": "asre_round4b_heldout_delta_z_gram",
                    "schema_version": 1,
                    "protocol": ROUND4B_PROTOCOL,
                    "layer": layer,
                    "tensor_kind": kind,
                    "rows": row_count,
                    "sample_count": len(sample_ids),
                    "split_sha256": sha256_file(split_path),
                    "basis_manifest_sha256": sha256_file(basis_path),
                    "gram": gram,
                },
            )
            artifacts[matrix] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "shape": [EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_DIM],
                "total_energy": float(torch.trace(gram).item()),
            }
    report = {
        "artifact_type": "asre_round4b_energy_curve_heldout_gram_shard",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "stage": COLLECT_STAGE,
        "status": "complete",
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "worker_index": args.worker_index,
        "physical_gpu": physical_gpu,
        "logical_device": "cuda:0",
        "layers": list(layers),
        "sample_count": len(sample_ids),
        "sample_ids_sha256": sha256_json(sample_ids),
        "row_count_per_matrix": row_count,
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "basis_manifest_path": str(basis_path),
        "basis_manifest_sha256": sha256_file(basis_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "donor_mapping_sha256": bundle.mapping_sha256,
        "donor_manifest_sha256": bundle.observation_manifest_sha256,
        "runtime_layout": layout,
        "cache_only": True,
        "online_episodes": 0,
        "environment_rollouts": 0,
        "heldout_svd_refit": False,
        "artifacts": artifacts,
    }
    atomic_write_json(output_dir / f"{COLLECT_STAGE}_shard{args.worker_index}.json", report)
    return report


def _load_recorded_tensor(path: Path, *, expected_sha256: str, key: str) -> torch.Tensor:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Artifact digest drifted: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload.get(key) if isinstance(payload, dict) else None
    if not torch.is_tensor(tensor):
        raise ValueError(f"Malformed tensor artifact: {path}")
    return tensor


def recover_coordinate_energy(args: argparse.Namespace) -> dict[str, Any]:
    """Recover a full frozen-calibration eigensystem and score held-out Grams."""

    physical_gpu = _physical_gpu()
    layers = _validate_worker_index(args.worker_index)
    basis_path = args.basis_manifest.resolve()
    basis = validate_basis_manifest(basis_path, verify_files=False)
    diagnostics_path = args.diagnostics.resolve()
    diagnostics = _read(diagnostics_path)
    if (
        diagnostics.get("status") != "complete"
        or diagnostics.get("basis_manifest_sha256") != sha256_file(basis_path)
    ):
        raise ValueError("Registered diagnostics do not match the frozen basis.")
    collect_report_path = (
        args.heldout_gram_dir.resolve()
        / f"{COLLECT_STAGE}_shard{args.worker_index}.json"
    )
    collect_report = _read(collect_report_path)
    if (
        collect_report.get("stage") != COLLECT_STAGE
        or collect_report.get("status") != "complete"
        or collect_report.get("layers") != list(layers)
        or collect_report.get("basis_manifest_sha256") != sha256_file(basis_path)
    ):
        raise ValueError("Held-out Gram shard is incomplete or incompatible.")
    fit_reports = {
        index: _read(args.fit_dir.resolve() / f"fit_shard{index}.json")
        for index in range(4)
    }
    report_for_layer = {
        int(layer): report
        for report in fit_reports.values()
        for layer in report["layers"]
    }
    output_dir = args.output_dir.resolve()
    artifacts: dict[str, dict[str, Any]] = {}
    registered = diagnostics["heldout_by_matrix"]
    for layer in layers:
        for kind in ("k", "v"):
            matrix = f"layer{layer:02d}_{kind}"
            fit_record = report_for_layer[layer]["artifacts"][matrix]
            fit_path = Path(str(fit_record["path"])).resolve()
            calibration_gram = _load_recorded_tensor(
                fit_path, expected_sha256=fit_record["sha256"], key="gram"
            )
            heldout_record = collect_report["artifacts"][matrix]
            heldout_path = Path(str(heldout_record["path"])).resolve()
            heldout_gram = _load_recorded_tensor(
                heldout_path, expected_sha256=heldout_record["sha256"], key="gram"
            )
            svd_record = basis["artifacts"]["svd"][str(layer)][kind]
            svd_path = Path(str(svd_record["path"])).resolve()
            if sha256_file(svd_path) != svd_record["sha256"]:
                raise ValueError(f"Frozen SVD artifact drifted: {svd_path}")
            svd_payload = torch.load(svd_path, map_location="cpu", weights_only=False)
            frozen_basis = svd_payload["basis"]
            singular_values = svd_payload["singular_values"]
            if (
                list(calibration_gram.shape) != [EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_DIM]
                or heldout_gram.shape != calibration_gram.shape
                or list(frozen_basis.shape) != [EXPECTED_FEATURE_DIM, MAX_RANK]
                or list(singular_values.shape) != [EXPECTED_FEATURE_DIM]
            ):
                raise ValueError(f"Malformed energy input for {matrix}.")

            calibration_gram = calibration_gram.to("cuda:0", dtype=torch.float32)
            heldout_gram = heldout_gram.to("cuda:0", dtype=torch.float32)
            calibration_gram = (calibration_gram + calibration_gram.T) * 0.5
            heldout_gram = (heldout_gram + heldout_gram.T) * 0.5
            frozen_basis = frozen_basis.to("cuda:0", dtype=torch.float32)

            # Preserve the originally registered prefix exactly for ranks <=1536.
            heldout_top = (
                frozen_basis * (heldout_gram @ frozen_basis)
            ).sum(dim=0).clamp_min(0)

            # Recover only the missing ordered complement from the original frozen
            # calibration Gram. No held-out quantity enters this eigensystem.
            _eigenvalues, eigenvectors = torch.linalg.eigh(calibration_gram)
            q_top, _ = torch.linalg.qr(frozen_basis, mode="reduced")
            tail_seed = eigenvectors[:, : EXPECTED_FEATURE_DIM - MAX_RANK].flip(1)
            tail_seed = tail_seed - q_top @ (q_top.T @ tail_seed)
            q_tail, _ = torch.linalg.qr(tail_seed, mode="reduced")
            restricted = q_tail.T @ calibration_gram @ q_tail
            restricted = (restricted + restricted.T) * 0.5
            tail_eigenvalues, tail_rotation = torch.linalg.eigh(restricted)
            tail_basis = q_tail @ tail_rotation.flip(1)
            heldout_tail = (
                tail_basis * (heldout_gram @ tail_basis)
            ).sum(dim=0).clamp_min(0)

            calibration_energy = singular_values.double().square()
            heldout_energy = torch.cat((heldout_top, heldout_tail)).double().cpu()
            heldout_total = float(torch.trace(heldout_gram).item())
            calibration_total = float(calibration_energy.sum().item())
            frozen_orthogonality = float(
                (
                    frozen_basis.T @ frozen_basis
                    - torch.eye(MAX_RANK, device="cuda:0", dtype=torch.float32)
                )
                .abs()
                .max()
                .item()
            )
            cross_orthogonality = float(
                (q_top.T @ tail_basis).abs().max().item()
            )
            tail_spectrum = tail_eigenvalues.flip(0).clamp_min(0).double().cpu()
            stored_tail = calibration_energy[MAX_RANK:]
            tail_spectrum_relative_l1 = float(
                (tail_spectrum - stored_tail).abs().sum().item()
                / max(float(stored_tail.sum().item()), 1e-30)
            )
            registered_errors = {}
            for rank in RANKS:
                observed = float(heldout_energy[:rank].sum().item() / heldout_total)
                expected = float(registered[matrix]["captured_fraction"][str(rank)])
                registered_errors[str(rank)] = {
                    "observed": observed,
                    "expected": expected,
                    "absolute_error": abs(observed - expected),
                }
            registered_total = float(registered[matrix]["total_energy"])
            total_relative_error = abs(heldout_total - registered_total) / registered_total
            rank_d_fraction = float(heldout_energy.sum().item() / heldout_total)
            artifact_path = output_dir / "coordinates" / f"{matrix}.pt"
            _atomic_torch_save(
                artifact_path,
                {
                    "artifact_type": "asre_round4b_complete_delta_z_coordinate_energy",
                    "schema_version": 1,
                    "protocol": ROUND4B_PROTOCOL,
                    "matrix": matrix,
                    "layer": layer,
                    "tensor_kind": kind,
                    "feature_dim": EXPECTED_FEATURE_DIM,
                    "frozen_prefix_rank": MAX_RANK,
                    "calibration_coordinate_energy": calibration_energy,
                    "heldout_coordinate_energy": heldout_energy,
                    "calibration_total_energy": calibration_total,
                    "heldout_total_energy": heldout_total,
                    "basis_manifest_sha256": sha256_file(basis_path),
                    "diagnostics_sha256": sha256_file(diagnostics_path),
                    "heldout_gram_sha256": heldout_record["sha256"],
                    "no_heldout_refit": True,
                    "registered_anchor_errors": registered_errors,
                    "registered_total_relative_error": total_relative_error,
                    "rank_d_heldout_fraction": rank_d_fraction,
                    "frozen_basis_orthogonality_max_abs": frozen_orthogonality,
                    "recovered_tail_cross_orthogonality_max_abs": cross_orthogonality,
                    "recovered_tail_spectrum_relative_l1_error": tail_spectrum_relative_l1,
                },
            )
            artifacts[matrix] = {
                "path": str(artifact_path),
                "sha256": sha256_file(artifact_path),
                "shape": [EXPECTED_FEATURE_DIM],
                "registered_anchor_errors": registered_errors,
                "registered_total_relative_error": total_relative_error,
                "rank_d_heldout_fraction": rank_d_fraction,
                "frozen_basis_orthogonality_max_abs": frozen_orthogonality,
                "recovered_tail_cross_orthogonality_max_abs": cross_orthogonality,
                "recovered_tail_spectrum_relative_l1_error": tail_spectrum_relative_l1,
            }
            del (
                calibration_gram,
                heldout_gram,
                frozen_basis,
                heldout_top,
                eigenvectors,
                q_top,
                tail_seed,
                q_tail,
                restricted,
                tail_eigenvalues,
                tail_rotation,
                tail_basis,
                heldout_tail,
            )
            torch.cuda.empty_cache()
            print(f"[Round4B energy recover] {matrix} complete", flush=True)
    report = {
        "artifact_type": "asre_round4b_energy_curve_coordinate_shard",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "stage": RECOVER_STAGE,
        "status": "complete",
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "worker_index": args.worker_index,
        "physical_gpu": physical_gpu,
        "logical_device": "cuda:0",
        "layers": list(layers),
        "basis_manifest_path": str(basis_path),
        "basis_manifest_sha256": sha256_file(basis_path),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "heldout_gram_report_path": str(collect_report_path),
        "heldout_gram_report_sha256": sha256_file(collect_report_path),
        "full_eigensystem_recovered_from_frozen_calibration_gram": True,
        "heldout_svd_refit": False,
        "online_episodes": 0,
        "environment_rollouts": 0,
        "artifacts": artifacts,
    }
    atomic_write_json(output_dir / f"{RECOVER_STAGE}_shard{args.worker_index}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(COLLECT_STAGE, RECOVER_STAGE), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--donor-mapping", type=Path)
    parser.add_argument("--donor-manifest", type=Path)
    parser.add_argument("--donor-root", type=Path)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--heldout-gram-dir", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    if args.stage == COLLECT_STAGE:
        required = (
            args.checkpoint,
            args.split,
            args.donor_mapping,
            args.donor_manifest,
            args.donor_root,
        )
        if any(value is None for value in required):
            parser.error("collect_heldout_grams requires checkpoint/split/donor inputs")
        report = collect_heldout_grams(args)
    else:
        if any(
            value is None
            for value in (args.fit_dir, args.heldout_gram_dir, args.diagnostics)
        ):
            parser.error("recover_coordinate_energy requires fit/heldout/diagnostics inputs")
        report = recover_coordinate_energy(args)
    print(
        f"Round-4B energy {report['stage']} shard {report['worker_index']} complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
