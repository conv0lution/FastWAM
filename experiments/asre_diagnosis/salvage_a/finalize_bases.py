"""Freeze matched SVD/ActionAware/random bases and offline diagnostics.

The fit and held-out workers only accumulate sufficient statistics.  This
module is the single point that diagonalizes those statistics, imports the
pre-outcome Round-4B random prefix, writes runtime artifacts, and computes the
registered held-out diagnostics without fitting anything on held-out data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_A_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    validate_basis_manifest as validate_round4b_basis_manifest,
)
from experiments.asre_diagnosis.salvage_a.action_sensitivity import (  # noqa: E402
    INTERPOLATION_LAMBDAS,
    frozen_probe_records,
    validate_probe_registry,
)
from experiments.asre_diagnosis.salvage_a.basis import (  # noqa: E402
    BASIS_KINDS,
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
    MAX_RANK,
    RANKS,
    TENSOR_KINDS,
    validate_basis_manifest,
)
from experiments.asre_diagnosis.salvage_a.split import (  # noqa: E402
    validate_split_manifest,
)
from experiments.asre_diagnosis.salvage_a.state_selection import (  # noqa: E402
    validate_state_selection_manifest,
)


DEFAULT_RANDOM_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "asre_results"
    / "round4b_subspace"
    / "calibration"
    / "basis_manifest.json"
)
DEFAULT_RANDOM_SOURCE_MANIFEST_SHA256 = (
    "1a3fcd8ed44e012abbfdf8da119b82e76baf01755a3a8472eb470845c7122d02"
)
EXPECTED_FIT_STATES = 100
EXPECTED_HELDOUT_STATES = 100
EXPECTED_ROWS = 100 * 98
EXPECTED_VJPS = 100 * 4
PSD_RELATIVE_TOLERANCE = 5e-5
ORTHOGONALITY_TOLERANCE = 5e-4


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _orthogonality_error(basis: torch.Tensor) -> float:
    identity = torch.eye(
        basis.shape[1], device=basis.device, dtype=basis.dtype
    )
    return float((basis.T @ basis - identity).abs().max().item())


def _effective_rank(eigenvalues: torch.Tensor) -> float:
    values = eigenvalues.clamp_min(0)
    total = values.sum()
    if float(total.item()) <= 0:
        return 0.0
    probabilities = values / total
    positive = probabilities[probabilities > 0]
    return float((-(positive * positive.log()).sum()).exp().item())


def _eigendecompose_psd(
    matrix: torch.Tensor, *, label: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Symmetrize, PSD-check, and return descending eigenpairs."""

    if list(matrix.shape) != [EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_DIM]:
        raise ValueError(f"Malformed {label} shape: {list(matrix.shape)}")
    if matrix.dtype != torch.float32 or not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f"{label} must be finite float32.")
    value = matrix.to(device=device, dtype=torch.float32)
    symmetry_error = float((value - value.T).abs().max().item())
    value = (value + value.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    largest = max(float(eigenvalues[-1].item()), 1.0)
    minimum = float(eigenvalues[0].item())
    tolerance = PSD_RELATIVE_TOLERANCE * largest
    if minimum < -tolerance:
        raise ValueError(
            f"{label} is not PSD within tolerance: min={minimum}, tol={tolerance}."
        )
    order = torch.arange(
        EXPECTED_FEATURE_DIM - 1, -1, -1, device=eigenvalues.device
    )
    descending = eigenvalues.index_select(0, order).clamp_min(0)
    vectors = eigenvectors.index_select(1, order).contiguous()
    diagnostics = {
        "symmetry_max_abs_before_symmetrization": symmetry_error,
        "minimum_raw_eigenvalue": minimum,
        "psd_absolute_tolerance": tolerance,
        "largest_eigenvalue": float(descending[0].item()),
        "total_trace": float(descending.sum().item()),
        "effective_rank": _effective_rank(descending),
    }
    if diagnostics["total_trace"] <= 0:
        raise ValueError(f"{label} has zero total trace.")
    return descending, vectors, diagnostics


def _projection_capture(basis: torch.Tensor, matrix: torch.Tensor) -> tuple[float, float]:
    total = float(torch.trace(matrix).item())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Projection diagnostic matrix has non-positive trace.")
    raw_captured = float((basis * (matrix @ basis)).sum().item())
    # Tiny negative/above-one excursions are possible after float32 accumulation.
    captured = min(total, max(0.0, raw_captured))
    fraction = captured / total
    return captured, fraction


def _subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] <= 0:
        raise ValueError("Subspace overlap requires equal nonempty 2-D bases.")
    rank = left.shape[1]
    value = float((left.T @ right).square().sum().item() / rank)
    return min(1.0, max(0.0, value))


def _principal_angle_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    singular_values = torch.linalg.svdvals(left.T @ right).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(singular_values))
    return {
        "principal_angle_mean_degrees": float(angles.mean().item()),
        "principal_angle_median_degrees": float(angles.median().item()),
        "principal_angle_max_degrees": float(angles.max().item()),
        "minimum_canonical_correlation": float(singular_values.min().item()),
    }


def _report_paths(directory: Path, phase: str) -> list[Path]:
    return [directory / f"{phase}_shard{index}.json" for index in range(4)]


def _validate_reports(
    *,
    directory: Path,
    phase: str,
    split_sha256: str,
    state_selection_sha256: str,
    expected_sample_ids_sha256: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    paths = _report_paths(directory, phase)
    reports = [_read_json(path) for path in paths]
    expected_count = EXPECTED_FIT_STATES if phase == "fit" else EXPECTED_HELDOUT_STATES
    expected_vjps = EXPECTED_VJPS if phase == "fit" else 0
    common_identities = {
        (
            report.get("git_commit_hash"),
            report.get("checkpoint_sha256"),
            report.get("donor_mapping_sha256"),
            report.get("donor_manifest_sha256"),
            report.get("differentiable_path_report_sha256"),
        )
        for report in reports
    }
    if len(common_identities) != 1:
        raise ValueError(f"{phase} shard provenance differs across workers.")
    if {int(report.get("worker_index", -1)) for report in reports} != set(range(4)):
        raise ValueError(f"{phase} shard reports do not cover workers 0..3.")
    for report in reports:
        if (
            report.get("artifact_type")
            != f"asre_salvage_a_{phase}_shard_report"
            or report.get("schema_version") != 1
            or report.get("protocol") != SALVAGE_A_PROTOCOL
            or report.get("status") != "complete"
            or report.get("phase") != phase
            or report.get("split_sha256") != split_sha256
            or report.get("state_selection_sha256") != state_selection_sha256
            or report.get("sample_count") != expected_count
            or report.get("sample_ids_sha256") != expected_sample_ids_sha256
            or report.get("row_count_per_matrix") != EXPECTED_ROWS
            or report.get("vjp_count") != expected_vjps
        ):
            raise ValueError(f"Incompatible Salvage-A {phase} shard report.")
        if phase == "fit":
            validate_probe_registry(report.get("probe_records", ()))
            if (
                report.get("vjp_count_per_state") != 4
                or report.get("full_action_denoising_steps") != 10
                or report.get("model_parameters_frozen") is not True
            ):
                raise ValueError("Fit shard did not use the registered action path.")
    covered = [int(layer) for report in reports for layer in report.get("layers", ())]
    if sorted(covered) != list(LATE_LAYERS) or len(set(covered)) != len(covered):
        raise ValueError(f"{phase} shards do not partition layers 15..29 exactly once.")
    layouts = [report.get("runtime_layout") for report in reports]
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError(f"Runtime layout differs across {phase} shards.")
    report_for_layer = {
        int(layer): report for report in reports for layer in report["layers"]
    }
    return reports, report_for_layer, dict(layouts[0])


def _load_matrix_artifact(
    *, report: Mapping[str, Any], matrix_name: str, phase: str
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    record = report.get("artifacts", {}).get(matrix_name)
    if not isinstance(record, dict):
        raise ValueError(f"Missing {phase} artifact record for {matrix_name}.")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{phase} matrix artifact drifted: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Malformed {phase} matrix artifact: {path}")
    if (
        payload.get("artifact_type") != f"asre_salvage_a_{phase}_matrix"
        or payload.get("protocol") != SALVAGE_A_PROTOCOL
        or payload.get("phase") != phase
    ):
        raise ValueError(f"Incompatible {phase} matrix artifact: {path}")
    return record, path, payload


def _source_random_prefix(
    source_manifest: Mapping[str, Any], *, layer: int, tensor_kind: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    record = source_manifest["artifacts"]["random"][str(layer)][tensor_kind]
    path = Path(str(record["path"])).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    basis = payload.get("basis") if isinstance(payload, Mapping) else None
    if (
        not torch.is_tensor(basis)
        or basis.ndim != 2
        or basis.shape[0] != EXPECTED_FEATURE_DIM
        or basis.shape[1] < MAX_RANK
        or basis.dtype != torch.float32
        or not bool(torch.isfinite(basis).all().item())
    ):
        raise ValueError(f"Malformed preregistered random basis: {path}")
    prefix = basis[:, :MAX_RANK].contiguous()
    return prefix, {
        "source_artifact_path": str(path),
        "source_artifact_sha256": str(record["sha256"]),
        "source_artifact_shape": list(record["shape"]),
        "source_prefix_columns": [0, MAX_RANK - 1],
        "source_prefix_tensor_sha256": _tensor_sha256(prefix),
        "source_seed": record.get("seed"),
    }


def _weighted_summary(
    matrix_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in BASIS_KINDS:
        for rank in RANKS:
            family_rows = [
                row
                for row in matrix_rows
                if row["family"] == family and int(row["rank"]) == rank
            ]
            for scope in ("all", "K", "V"):
                selected = (
                    family_rows
                    if scope == "all"
                    else [row for row in family_rows if row["tensor_kind"] == scope]
                )
                if not selected:
                    raise AssertionError("Empty Salvage-A diagnostic summary scope.")
                heldout_total = sum(float(row["heldout_total_delta_z_energy"]) for row in selected)
                heldout_captured = sum(
                    float(row["heldout_captured_delta_z_energy"]) for row in selected
                )
                action_total = sum(
                    float(row["calibration_total_action_sensitivity"]) for row in selected
                )
                action_captured = sum(
                    float(row["calibration_captured_action_sensitivity"])
                    for row in selected
                )
                stability_weighted = sum(
                    float(row["aa_half_stability_overlap"])
                    * float(row["calibration_total_action_sensitivity"])
                    for row in selected
                ) / action_total
                overlap_weighted = sum(
                    float(row["aa_svd_overlap"])
                    * float(row["calibration_total_action_sensitivity"])
                    for row in selected
                ) / action_total
                rows.append(
                    {
                        "family": family,
                        "basis_family": family,
                        "rank": rank,
                        "scope": scope,
                        "matrix_count": len(selected),
                        "heldout_total_delta_z_energy": heldout_total,
                        "heldout_captured_delta_z_energy": heldout_captured,
                        "heldout_delta_z_energy_captured": heldout_captured
                        / heldout_total,
                        "calibration_total_action_sensitivity": action_total,
                        "calibration_captured_action_sensitivity": action_captured,
                        "calibration_action_sensitivity_captured": action_captured
                        / action_total,
                        "aa_svd_overlap": overlap_weighted,
                        "aa_half_stability_overlap": stability_weighted,
                    }
                )
    return rows


def finalize(args: argparse.Namespace) -> tuple[Path, Path]:
    output = args.output.expanduser().resolve()
    diagnostics_output = (
        output.parent / "subspace_diagnostics.json"
        if args.diagnostics_output is None
        else args.diagnostics_output.expanduser().resolve()
    )
    split_path = args.split.expanduser().resolve()
    split = validate_split_manifest(split_path)
    split_sha = sha256_file(split_path)
    state_selection_path = args.state_selection.expanduser().resolve()
    selection = validate_state_selection_manifest(
        state_selection_path, split_manifest=split
    )
    state_selection_sha = sha256_file(state_selection_path)

    fit_dir = args.fit_dir.expanduser().resolve()
    heldout_dir = args.heldout_dir.expanduser().resolve()
    fit_reports, fit_by_layer, fit_layout = _validate_reports(
        directory=fit_dir,
        phase="fit",
        split_sha256=split_sha,
        state_selection_sha256=state_selection_sha,
        expected_sample_ids_sha256=sha256_json(selection["calibration_sample_ids"]),
    )
    heldout_reports, heldout_by_layer, heldout_layout = _validate_reports(
        directory=heldout_dir,
        phase="heldout",
        split_sha256=split_sha,
        state_selection_sha256=state_selection_sha,
        expected_sample_ids_sha256=sha256_json(selection["heldout_sample_ids"]),
    )
    identity_keys = (
        "git_commit_hash",
        "checkpoint_sha256",
        "donor_mapping_sha256",
        "donor_manifest_sha256",
        "differentiable_path_report_sha256",
    )
    if any(
        fit_reports[0].get(key) != heldout_reports[0].get(key)
        for key in identity_keys
    ):
        raise ValueError("Fit and held-out shards used different frozen inputs.")
    if fit_reports[0].get("git_commit_hash") != git_commit(PROJECT_ROOT):
        raise ValueError("Fit shards were produced by a different source commit.")
    if fit_layout != heldout_layout:
        raise ValueError("Fit and held-out runtime cache layouts differ.")

    source_path = args.random_source_manifest.expanduser().resolve()
    source_sha = args.random_source_sha256
    source_manifest = validate_round4b_basis_manifest(
        source_path, expected_sha256=source_sha
    )
    if output.exists():
        manifest = validate_basis_manifest(output)
        if not diagnostics_output.is_file():
            raise FileNotFoundError(
                "Basis manifest exists but its subspace diagnostics are missing."
            )
        diagnostics = _read_json(diagnostics_output)
        expected_fit_reports = [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in _report_paths(fit_dir, "fit")
        ]
        expected_heldout_reports = [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in _report_paths(heldout_dir, "heldout")
        ]
        expected_manifest_links = {
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "split_sha256": split_sha,
            "state_selection_sha256": state_selection_sha,
            "differentiable_path_report_sha256": fit_reports[0][
                "differentiable_path_report_sha256"
            ],
            "donor_mapping_sha256": fit_reports[0]["donor_mapping_sha256"],
            "donor_observation_manifest_sha256": fit_reports[0][
                "donor_manifest_sha256"
            ],
            "checkpoint_sha256": fit_reports[0]["checkpoint_sha256"],
            "random_source_manifest_sha256": source_sha,
            "subspace_diagnostics_sha256": sha256_file(diagnostics_output),
            "fit_shard_reports": expected_fit_reports,
            "heldout_shard_reports": expected_heldout_reports,
        }
        expected_diagnostics_links = {
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "split_sha256": split_sha,
            "state_selection_sha256": state_selection_sha,
            "differentiable_path_report_sha256": fit_reports[0][
                "differentiable_path_report_sha256"
            ],
            "heldout_used_for_fitting": False,
        }
        manifest_mismatch = {
            key: {"observed": manifest.get(key), "expected": value}
            for key, value in expected_manifest_links.items()
            if manifest.get(key) != value
        }
        diagnostics_mismatch = {
            key: {"observed": diagnostics.get(key), "expected": value}
            for key, value in expected_diagnostics_links.items()
            if diagnostics.get(key) != value
        }
        if manifest_mismatch or diagnostics_mismatch:
            raise ValueError(
                "Refusing incompatible standalone Salvage-A basis resume: "
                f"manifest={manifest_mismatch}, diagnostics={diagnostics_mismatch}."
            )
        return output, diagnostics_output
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Salvage-A basis finalization requested CUDA, but none is available.")

    artifact_root = output.parent / "basis_artifacts"
    artifacts: dict[str, dict[str, dict[str, Any]]] = {
        family: {} for family in BASIS_KINDS
    }
    matrix_rows: list[dict[str, Any]] = []
    matrix_diagnostics: dict[str, Any] = {}

    for layer in LATE_LAYERS:
        for family in BASIS_KINDS:
            artifacts[family][str(layer)] = {}
        for tensor_kind in TENSOR_KINDS:
            matrix_name = f"layer{layer:02d}_{tensor_kind}"
            _, fit_path, fit_payload = _load_matrix_artifact(
                report=fit_by_layer[layer], matrix_name=matrix_name, phase="fit"
            )
            _, heldout_path, heldout_payload = _load_matrix_artifact(
                report=heldout_by_layer[layer],
                matrix_name=matrix_name,
                phase="heldout",
            )
            if (
                fit_payload.get("layer") != layer
                or fit_payload.get("tensor_kind") != tensor_kind
                or fit_payload.get("sample_count") != EXPECTED_FIT_STATES
                or fit_payload.get("rows") != EXPECTED_ROWS
                or fit_payload.get("vjp_count") != EXPECTED_VJPS
                or fit_payload.get("interpolation_lambdas")
                != list(INTERPOLATION_LAMBDAS)
            ):
                raise ValueError(f"Fit matrix metadata drifted: {fit_path}")
            validate_probe_registry(fit_payload.get("probe_records", ()))
            if (
                heldout_payload.get("layer") != layer
                or heldout_payload.get("tensor_kind") != tensor_kind
                or heldout_payload.get("sample_count") != EXPECTED_HELDOUT_STATES
                or heldout_payload.get("rows") != EXPECTED_ROWS
            ):
                raise ValueError(f"Held-out matrix metadata drifted: {heldout_path}")

            gram = fit_payload.get("gram")
            covariance = fit_payload.get("covariance")
            half_a_covariance = fit_payload.get("half_a_covariance")
            half_b_covariance = fit_payload.get("half_b_covariance")
            heldout_gram = heldout_payload.get("gram")
            if not all(
                torch.is_tensor(value)
                for value in (
                    gram,
                    covariance,
                    half_a_covariance,
                    half_b_covariance,
                    heldout_gram,
                )
            ):
                raise ValueError(f"Incomplete sufficient statistics for {matrix_name}.")
            if not torch.allclose(
                covariance,
                half_a_covariance + half_b_covariance,
                rtol=1e-5,
                atol=1e-4,
            ):
                raise ValueError(f"Stability covariances do not sum for {matrix_name}.")

            svd_values, svd_vectors, svd_psd = _eigendecompose_psd(
                gram, label=f"{matrix_name} calibration Gram", device=device
            )
            action_values, action_vectors, action_psd = _eigendecompose_psd(
                covariance,
                label=f"{matrix_name} action covariance",
                device=device,
            )
            half_a_values, half_a_vectors, half_a_psd = _eigendecompose_psd(
                half_a_covariance,
                label=f"{matrix_name} half-A action covariance",
                device=device,
            )
            half_b_values, half_b_vectors, half_b_psd = _eigendecompose_psd(
                half_b_covariance,
                label=f"{matrix_name} half-B action covariance",
                device=device,
            )
            heldout_matrix = ((heldout_gram + heldout_gram.T) * 0.5).to(
                device=device, dtype=torch.float32
            )
            calibration_action = ((covariance + covariance.T) * 0.5).to(
                device=device, dtype=torch.float32
            )
            svd_basis = svd_vectors[:, :MAX_RANK].contiguous()
            action_basis = action_vectors[:, :MAX_RANK].contiguous()
            random_cpu, random_source = _source_random_prefix(
                source_manifest, layer=layer, tensor_kind=tensor_kind
            )
            random_basis = random_cpu.to(device=device).contiguous()
            family_bases = {
                "svd": svd_basis,
                "actionaware": action_basis,
                "random": random_basis,
            }

            orthogonality: dict[str, float] = {}
            for family, basis in family_bases.items():
                error = _orthogonality_error(basis)
                orthogonality[family] = error
                if error > ORTHOGONALITY_TOLERANCE:
                    raise ValueError(
                        f"{family} basis is not orthonormal for {matrix_name}: {error}."
                    )
                path = artifact_root / family / f"{matrix_name}.pt"
                artifact_payload: dict[str, Any] = {
                    "artifact_type": f"asre_salvage_a_{family}_basis",
                    "schema_version": 1,
                    "protocol": SALVAGE_A_PROTOCOL,
                    "layer": layer,
                    "tensor_kind": tensor_kind,
                    "feature_dim": EXPECTED_FEATURE_DIM,
                    "max_rank": MAX_RANK,
                    "nested_prefix_ranks": list(RANKS),
                    "basis": basis.detach().cpu(),
                    "split_sha256": split_sha,
                    "state_selection_sha256": state_selection_sha,
                }
                if family == "svd":
                    artifact_payload.update(
                        {
                            "uncentered": True,
                            "eigenvalues": svd_values.detach().cpu(),
                            "singular_values": svd_values.sqrt().detach().cpu(),
                        }
                    )
                elif family == "actionaware":
                    artifact_payload.update(
                        {
                            "covariance_estimator": "deterministic_hutchinson_vjp",
                            "interpolation_lambdas": list(INTERPOLATION_LAMBDAS),
                            "probe_records": frozen_probe_records(),
                            "eigenvalues": action_values.detach().cpu(),
                        }
                    )
                else:
                    artifact_payload.update(random_source)
                    artifact_payload["reused_before_online_outcomes"] = True
                _atomic_torch_save(path, artifact_payload)
                record: dict[str, Any] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "shape": [EXPECTED_FEATURE_DIM, MAX_RANK],
                    "dtype": "torch.float32",
                    "orthogonality_max_abs": error,
                    "nested_prefix_ranks": list(RANKS),
                }
                if family == "random":
                    record.update(random_source)
                artifacts[family][str(layer)][tensor_kind] = record

            per_rank_geometry: dict[str, Any] = {}
            for rank in RANKS:
                aa_prefix = action_basis[:, :rank]
                svd_prefix = svd_basis[:, :rank]
                half_a_prefix = half_a_vectors[:, :rank]
                half_b_prefix = half_b_vectors[:, :rank]
                aa_svd_overlap = _subspace_overlap(aa_prefix, svd_prefix)
                stability_overlap = _subspace_overlap(half_a_prefix, half_b_prefix)
                angle_summary = _principal_angle_summary(
                    half_a_prefix, half_b_prefix
                )
                per_rank_geometry[str(rank)] = {
                    "aa_svd_overlap": aa_svd_overlap,
                    "aa_half_stability_overlap": stability_overlap,
                    **angle_summary,
                }
                for family, full_basis in family_bases.items():
                    prefix = full_basis[:, :rank]
                    heldout_captured, heldout_fraction = _projection_capture(
                        prefix, heldout_matrix
                    )
                    action_captured, action_fraction = _projection_capture(
                        prefix, calibration_action
                    )
                    matrix_rows.append(
                        {
                            "matrix": matrix_name,
                            "layer": layer,
                            "tensor_kind": tensor_kind.upper(),
                            "family": family,
                            "basis_family": family,
                            "rank": rank,
                            "heldout_total_delta_z_energy": float(
                                torch.trace(heldout_matrix).item()
                            ),
                            "heldout_captured_delta_z_energy": heldout_captured,
                            "heldout_delta_z_energy_captured": heldout_fraction,
                            "calibration_total_action_sensitivity": float(
                                torch.trace(calibration_action).item()
                            ),
                            "calibration_captured_action_sensitivity": action_captured,
                            "calibration_action_sensitivity_captured": action_fraction,
                            "aa_svd_overlap": aa_svd_overlap,
                            "aa_half_stability_overlap": stability_overlap,
                            **angle_summary,
                        }
                    )
            matrix_diagnostics[matrix_name] = {
                "fit_artifact_path": str(fit_path),
                "fit_artifact_sha256": sha256_file(fit_path),
                "heldout_artifact_path": str(heldout_path),
                "heldout_artifact_sha256": sha256_file(heldout_path),
                "svd_psd": svd_psd,
                "actionaware_psd": action_psd,
                "half_a_psd": half_a_psd,
                "half_b_psd": half_b_psd,
                "basis_orthogonality_max_abs": orthogonality,
                "rank_diagnostics": per_rank_geometry,
            }
            del (
                gram,
                covariance,
                half_a_covariance,
                half_b_covariance,
                heldout_gram,
                heldout_matrix,
                calibration_action,
                svd_values,
                svd_vectors,
                action_values,
                action_vectors,
                half_a_values,
                half_a_vectors,
                half_b_values,
                half_b_vectors,
                svd_basis,
                action_basis,
                random_basis,
                random_cpu,
                family_bases,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[Salvage-A finalize] {matrix_name} complete", flush=True)

    summary_rows = _weighted_summary(matrix_rows)
    actionaware_rows = [
        row for row in matrix_rows if row["family"] == "actionaware"
    ]
    aa_svd_overlap_rows = [
        {
            "matrix": row["matrix"],
            "layer": row["layer"],
            "tensor_kind": row["tensor_kind"],
            "rank": row["rank"],
            "subspace_overlap": row["aa_svd_overlap"],
        }
        for row in actionaware_rows
    ]
    aa_half_stability_rows = [
        {
            "matrix": row["matrix"],
            "layer": row["layer"],
            "tensor_kind": row["tensor_kind"],
            "rank": row["rank"],
            "subspace_overlap": row["aa_half_stability_overlap"],
            "principal_angle_mean_degrees": row[
                "principal_angle_mean_degrees"
            ],
            "principal_angle_median_degrees": row[
                "principal_angle_median_degrees"
            ],
            "principal_angle_max_degrees": row[
                "principal_angle_max_degrees"
            ],
            "minimum_canonical_correlation": row[
                "minimum_canonical_correlation"
            ],
        }
        for row in actionaware_rows
    ]
    if len(aa_svd_overlap_rows) != 60 or len(aa_half_stability_rows) != 60:
        raise AssertionError("Expected exactly 30 matrices x 2 ranks of AA geometry.")
    primary_summary_rows = [
        row for row in summary_rows if row["scope"] == "all"
    ]
    if len(primary_summary_rows) != 6:
        raise AssertionError("Expected exactly three basis families x two ranks.")
    diagnostics_payload = {
        "artifact_type": "asre_salvage_a_subspace_diagnostics",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "offline_only": True,
        "heldout_used_for_fitting": False,
        "weighted_summary_definition": (
            "absolute trace energy/sensitivity weighting across included matrices"
        ),
        "split_path": str(split_path),
        "split_sha256": split_sha,
        "state_selection_path": str(state_selection_path),
        "state_selection_sha256": state_selection_sha,
        "differentiable_path_report_sha256": fit_reports[0][
            "differentiable_path_report_sha256"
        ],
        "ranks": list(RANKS),
        "summary_rows": summary_rows,
        "matrix_rows": matrix_rows,
        "basis_diagnostics_summary": primary_summary_rows,
        "basis_diagnostics_by_matrix": matrix_rows,
        "aa_svd_overlap": aa_svd_overlap_rows,
        "aa_half_stability": aa_half_stability_rows,
        "matrix_diagnostics": matrix_diagnostics,
    }
    atomic_write_json(diagnostics_output, diagnostics_payload)

    manifest_payload = {
        "artifact_type": "asre_salvage_a_basis_manifest",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "feature_dim": EXPECTED_FEATURE_DIM,
        "ranks": list(RANKS),
        "max_rank": MAX_RANK,
        "late_layers": list(LATE_LAYERS),
        "k_v_fitted_separately": True,
        "uncentered_svd": True,
        "matched_calibration_states_for_svd_and_actionaware": True,
        "actionaware_full_calibration_basis": True,
        "actionaware_estimator": "deterministic_hutchinson_vjp",
        "interpolation_lambdas": list(INTERPOLATION_LAMBDAS),
        "probe_records": frozen_probe_records(),
        "random_basis_nested": True,
        "random_prefix_reused_pre_outcome": True,
        "random_source_manifest_path": str(source_path),
        "random_source_manifest_sha256": source_sha,
        "runtime_layout": fit_layout,
        "split_path": str(split_path),
        "split_sha256": split_sha,
        "state_selection_path": str(state_selection_path),
        "state_selection_sha256": state_selection_sha,
        "differentiable_path_gate_passed": True,
        "differentiable_path_report_sha256": fit_reports[0][
            "differentiable_path_report_sha256"
        ],
        "donor_mapping_sha256": fit_reports[0]["donor_mapping_sha256"],
        "donor_observation_manifest_sha256": fit_reports[0][
            "donor_manifest_sha256"
        ],
        "checkpoint_sha256": fit_reports[0]["checkpoint_sha256"],
        "fit_shard_reports": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in _report_paths(fit_dir, "fit")
        ],
        "heldout_shard_reports": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in _report_paths(heldout_dir, "heldout")
        ],
        "subspace_diagnostics_path": str(diagnostics_output),
        "subspace_diagnostics_sha256": sha256_file(diagnostics_output),
        "artifacts": artifacts,
    }
    atomic_write_json(output, manifest_payload)
    validate_basis_manifest(output)
    return output, diagnostics_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--state-selection", type=Path, required=True)
    parser.add_argument(
        "--random-source-manifest",
        type=Path,
        default=DEFAULT_RANDOM_SOURCE_MANIFEST,
    )
    parser.add_argument(
        "--random-source-sha256",
        default=DEFAULT_RANDOM_SOURCE_MANIFEST_SHA256,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path)
    manifest, diagnostics = finalize(parser.parse_args())
    print(f"Frozen Salvage-A basis manifest: {manifest}")
    print(f"Frozen Salvage-A subspace diagnostics: {diagnostics}")


if __name__ == "__main__":
    main()
