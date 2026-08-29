"""Real-checkpoint endpoint, basis, covariance, and provenance gates for Salvage A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_A_PROTOCOL,
    atomic_write_json,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.salvage_a.basis import (  # noqa: E402
    BASIS_KINDS,
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
    RANKS,
    TENSOR_KINDS,
    load_runtime_basis,
    validate_basis_manifest,
)
from experiments.asre_diagnosis.salvage_a.donor import (  # noqa: E402
    SalvageADonorBundle,
    load_donor_image,
)
from experiments.asre_diagnosis.salvage_a.fit_worker import (  # noqa: E402
    EARLY_DISABLED,
    _load_model,
    _load_sample,
    _read,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


ACTION_TOLERANCE = 2.0e-2
ORTHOGONALITY_TOLERANCE = 5.0e-4


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left = left.detach().float().cpu()
    right = right.detach().float().cpu()
    difference = (left - right).abs()
    return {
        "max_raw_action_abs_diff": float(difference.max().item()),
        "mean_raw_action_abs_diff": float(difference.mean().item()),
        "torch_allclose": bool(
            torch.allclose(
                left, right, atol=ACTION_TOLERANCE, rtol=ACTION_TOLERANCE
            )
        ),
        "exact_equal": bool(torch.equal(left, right)),
    }


def _load_basis_tensor(record: Mapping[str, Any]) -> torch.Tensor:
    path = Path(str(record["path"])).resolve()
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Basis artifact drifted: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload.get("basis") if isinstance(payload, Mapping) else None
    if (
        not torch.is_tensor(tensor)
        or list(tensor.shape) != [EXPECTED_FEATURE_DIM, max(RANKS)]
        or tensor.dtype != torch.float32
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise ValueError(f"Malformed basis artifact: {path}")
    return tensor


def _validate_offline_geometry(
    manifest: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> dict[str, Any]:
    basis_rows: list[dict[str, Any]] = []
    all_orthonormal = True
    all_prefixes_nested = True
    for family in BASIS_KINDS:
        for layer in LATE_LAYERS:
            for kind in TENSOR_KINDS:
                tensor = _load_basis_tensor(
                    manifest["artifacts"][family][str(layer)][kind]
                )
                gram = tensor.T @ tensor
                identity = torch.eye(max(RANKS), dtype=torch.float32)
                error = float((gram - identity).abs().max().item())
                nested = bool(torch.equal(tensor[:, :36], tensor[:, :97][:, :36]))
                all_orthonormal = all_orthonormal and error <= ORTHOGONALITY_TOLERANCE
                all_prefixes_nested = all_prefixes_nested and nested
                basis_rows.append(
                    {
                        "matrix": f"layer{layer:02d}_{kind}",
                        "basis_family": family,
                        "orthogonality_max_abs": error,
                        "r36_exact_prefix_of_r97": nested,
                    }
                )
    matrix_checks = diagnostics.get("matrix_diagnostics")
    if not isinstance(matrix_checks, dict) or len(matrix_checks) != 30:
        raise ValueError("Offline diagnostics must contain exactly 30 matrices.")
    covariance_psd = True
    covariance_symmetric = True
    for record in matrix_checks.values():
        for stem in ("svd_psd", "actionaware_psd", "half_a_psd", "half_b_psd"):
            check = record.get(stem)
            if not isinstance(check, dict):
                raise ValueError(f"Missing {stem} machinery diagnostic.")
            covariance_psd = covariance_psd and float(
                check["minimum_raw_eigenvalue"]
            ) >= -float(check["psd_absolute_tolerance"])
            covariance_symmetric = covariance_symmetric and float(
                check["symmetry_max_abs_before_symmetrization"]
            ) <= max(1e-3, float(check["largest_eigenvalue"]) * 1e-5)
    summary = diagnostics.get(
        "basis_diagnostics_summary", diagnostics.get("summary_rows")
    )
    by_matrix = diagnostics.get(
        "basis_diagnostics_by_matrix", diagnostics.get("matrix_rows")
    )
    diagnostics_complete = bool(
        isinstance(summary, list)
        and len(summary) in (6, 18)
        and isinstance(by_matrix, list)
        and len(by_matrix) == 180
    )
    return {
        "basis_rows": basis_rows,
        "all_bases_orthonormal": all_orthonormal,
        "all_rank_prefixes_nested": all_prefixes_nested,
        "covariances_psd_within_tolerance": covariance_psd,
        "covariances_symmetric_within_tolerance": covariance_symmetric,
        "offline_diagnostics_complete": diagnostics_complete,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Salvage-A machinery tests require CUDA.")
    preflight_path = args.preflight.resolve()
    differentiable_path = args.differentiable_path.resolve()
    preflight = _read(preflight_path)
    differentiable = _read(differentiable_path)
    if (
        preflight.get("protocol") != SALVAGE_A_PROTOCOL
        or preflight.get("status") != "compatible"
        or differentiable.get("protocol") != SALVAGE_A_PROTOCOL
        or differentiable.get("passed") is not True
    ):
        raise ValueError("Preflight or differentiable-path gate did not pass.")
    current_commit = git_commit(PROJECT_ROOT)
    if (
        preflight.get("git", {}).get("current_head") != current_commit
        or differentiable.get("git_commit_hash") != current_commit
    ):
        raise ValueError("Machinery inputs were produced by a different source commit.")

    basis_path = args.basis_manifest.resolve()
    manifest = validate_basis_manifest(basis_path, verify_files=True)
    diagnostics_path = args.diagnostics.resolve()
    diagnostics = _read(diagnostics_path)
    differentiable_sha = sha256_file(differentiable_path)
    if (
        diagnostics.get("protocol") != SALVAGE_A_PROTOCOL
        or diagnostics.get("heldout_used_for_fitting") is not False
        or manifest.get("differentiable_path_gate_passed") is not True
        or manifest.get("differentiable_path_report_sha256") != differentiable_sha
        or diagnostics.get("differentiable_path_report_sha256")
        != differentiable_sha
        or manifest.get("subspace_diagnostics_sha256")
        != sha256_file(diagnostics_path)
        or manifest.get("git_commit_hash") != current_commit
        or diagnostics.get("git_commit_hash") != current_commit
    ):
        raise ValueError("Basis/diagnostics provenance is incompatible.")
    geometry = _validate_offline_geometry(manifest, diagnostics)

    split_path = args.split.resolve()
    split = _read(split_path)
    selection_path = args.state_selection.resolve()
    selection = _read(selection_path)
    if (
        manifest.get("split_sha256") != sha256_file(split_path)
        or manifest.get("state_selection_sha256") != sha256_file(selection_path)
        or differentiable.get("split_sha256") != sha256_file(split_path)
        or differentiable.get("state_selection_sha256")
        != sha256_file(selection_path)
    ):
        raise ValueError("Basis does not bind the frozen split/state selection.")
    source_path = Path(str(split["source_manifest_path"])).resolve()
    by_id = {str(row["sample_id"]): row for row in load_manifest(source_path)}
    sample_id = str(selection["heldout_sample_ids"][0])
    record = by_id[sample_id]
    sample = _load_sample(record=record, source_path=source_path)
    infer_kwargs = dict(sample["infer_action_kwargs"])
    bundle = SalvageADonorBundle.load(
        mapping_path=args.donor_mapping.resolve(),
        observation_manifest_path=args.donor_manifest.resolve(),
        observation_root=args.donor_root.resolve(),
    )
    if (
        manifest.get("donor_mapping_sha256") != bundle.mapping_sha256
        or manifest.get("donor_observation_manifest_sha256")
        != bundle.observation_manifest_sha256
        or differentiable.get("donor_mapping_sha256") != bundle.mapping_sha256
        or differentiable.get("donor_manifest_sha256")
        != bundle.observation_manifest_sha256
        or manifest.get("checkpoint_sha256")
        != sha256_file(args.checkpoint.resolve())
        or differentiable.get("checkpoint_sha256")
        != sha256_file(args.checkpoint.resolve())
    ):
        raise ValueError("Basis/action gate does not bind donor or checkpoint inputs.")
    donor = load_donor_image(
        bundle,
        task_id=int(record["task_id"]),
        episode_id=int(record["episode_id"]),
    ).to(dtype=infer_kwargs["input_image"].dtype)
    set_global_seed(42, get_worker_init_fn=False)
    model, _cfg = _load_model(args.checkpoint.resolve())
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    runtime_specs = {
        (family, rank): load_runtime_basis(
            manifest_path=basis_path,
            expected_sha256=sha256_file(basis_path),
            basis_kind=family,
            rank=rank,
            device=device,
            dtype=dtype,
        )
        for family in BASIS_KINDS
        for rank in RANKS
    }
    with torch.no_grad():
        current = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            compile_action_infer=False,
        )
        wrong = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor,
            replacement_video_layers=LATE_LAYERS,
            compile_action_infer=False,
        )
        self_replacement = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=infer_kwargs["input_image"].clone(),
            replacement_video_layers=LATE_LAYERS,
            return_video_cache_stats=True,
            compile_action_infer=False,
        )
        zero = torch.empty((EXPECTED_FEATURE_DIM, 0), device=device, dtype=dtype)
        zero_bases = {
            layer: {"k": zero, "v": zero} for layer in LATE_LAYERS
        }
        identity = torch.eye(EXPECTED_FEATURE_DIM, device=device, dtype=dtype)
        full_bases = {
            layer: {"k": identity, "v": identity} for layer in LATE_LAYERS
        }
        rank0 = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor,
            replacement_video_layers=LATE_LAYERS,
            feature_projection_bases_by_layer=zero_bases,
            feature_projection_rank=0,
            expected_video_cache_layout=runtime_specs[("svd", 36)].expected_video_cache_layout,
            return_video_cache_stats=True,
            compile_action_infer=False,
        )
        rankd = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor,
            replacement_video_layers=LATE_LAYERS,
            feature_projection_bases_by_layer=full_bases,
            feature_projection_rank=EXPECTED_FEATURE_DIM,
            expected_video_cache_layout=runtime_specs[("svd", 36)].expected_video_cache_layout,
            return_video_cache_stats=True,
            compile_action_infer=False,
        )
        projected = {
            (family, rank): model.infer_action(
                **infer_kwargs,
                disabled_video_layers=EARLY_DISABLED,
                replacement_input_image=donor,
                replacement_video_layers=LATE_LAYERS,
                return_video_cache_stats=True,
                compile_action_infer=False,
                **runtime_specs[(family, rank)].inference_kwargs(),
            )
            for family in BASIS_KINDS
            for rank in RANKS
        }

    rank0_comparison = _comparison(rank0["action"], wrong["action"])
    rankd_comparison = _comparison(rankd["action"], current["action"])
    self_comparison = _comparison(self_replacement["action"], current["action"])
    outputs = [
        current["action"],
        wrong["action"],
        self_replacement["action"],
        rank0["action"],
        rankd["action"],
        *(value["action"] for value in projected.values()),
    ]
    finite_actions = all(
        tuple(action.shape) == (32, 7)
        and bool(torch.isfinite(action).all().item())
        for action in outputs
    )
    projection_audits: list[dict[str, Any]] = []
    audits_passed = True
    for (family, rank), output in projected.items():
        audit = output["video_cache_stats"]["hybrid_video_cache"]
        passed = bool(
            audit.get("mode") == "feature_projection"
            and audit.get("projection_rank") == rank
            and audit.get("replacement_video_layers") == list(LATE_LAYERS)
            and audit.get("action_visible_token_count") == 98
            and audit.get("shape_preserved") is True
            and audit.get("tokens_modified") is False
            and audit.get("heads_modified") is False
            and audit.get("k_v_bases_independent") is True
        )
        audits_passed = audits_passed and passed
        projection_audits.append(
            {"basis_family": family, "rank": rank, "passed": passed, "audit": audit}
        )
    parameters_frozen = all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.parameters()
    )
    passed = bool(
        differentiable["passed"]
        and geometry["all_bases_orthonormal"]
        and geometry["all_rank_prefixes_nested"]
        and geometry["covariances_psd_within_tolerance"]
        and geometry["covariances_symmetric_within_tolerance"]
        and geometry["offline_diagnostics_complete"]
        and rank0_comparison["torch_allclose"]
        and rankd_comparison["torch_allclose"]
        and self_comparison["exact_equal"]
        and finite_actions
        and audits_passed
        and parameters_frozen
    )
    return {
        "artifact_type": "asre_salvage_a_machinery_report",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "status": "passed" if passed else "failed",
        "created_at": now_iso(),
        "git_commit_hash": current_commit,
        "sample_id": sample_id,
        "rank0_equals_wrong": rank0_comparison,
        "rankd_equals_current": rankd_comparison,
        "self_replacement_equals_current": self_comparison,
        "finite_raw_action_shape_32x7": finite_actions,
        "projection_audits": projection_audits,
        "projection_audits_passed": audits_passed,
        "model_parameters_frozen_and_without_gradients": parameters_frozen,
        **geometry,
        "preflight_report_sha256": sha256_file(preflight_path),
        "differentiable_path_report_sha256": sha256_file(differentiable_path),
        "split_manifest_sha256": sha256_file(split_path),
        "state_selection_manifest_sha256": sha256_file(selection_path),
        "donor_mapping_sha256": bundle.mapping_sha256,
        "donor_observation_manifest_sha256": bundle.observation_manifest_sha256,
        "basis_manifest_sha256": sha256_file(basis_path),
        "subspace_diagnostics_sha256": sha256_file(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--differentiable-path", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--state-selection", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    atomic_write_json(args.output.resolve(), report)
    if not report["passed"]:
        raise RuntimeError("Salvage-A machinery tests failed.")
    print(f"Salvage-A machinery passed: {args.output.resolve()}")


if __name__ == "__main__":
    main()
