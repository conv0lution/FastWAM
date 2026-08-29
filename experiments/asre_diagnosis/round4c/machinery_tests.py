"""Real-checkpoint endpoint, rank-prefix, and layout gates for Round-4C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4C_PROTOCOL,
    atomic_write_json,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.g0.machinery_tests import _comparison, _compose  # noqa: E402
from experiments.asre_diagnosis.round3b.donor import OnlineDonorBundle  # noqa: E402
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
    load_runtime_basis,
    validate_basis_manifest,
)
from experiments.asre_diagnosis.round4b.fit_worker import _load_donor_image  # noqa: E402
from experiments.asre_diagnosis.round4c.definitions import RANKS  # noqa: E402
from experiments.libero.eval_libero_single import (  # noqa: E402
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


EARLY_DISABLED = tuple(range(15))
TASK_CONFIG = "libero_uncond_2cam224_1e-4"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Round-4C machinery tests require CUDA.")
    preflight_path = args.preflight.resolve()
    preflight = _read(preflight_path)
    if preflight.get("protocol") != ROUND4C_PROTOCOL or preflight.get("status") != "compatible":
        raise ValueError("Round-4C preflight did not pass.")
    basis_path = args.basis_manifest.resolve()
    manifest = validate_basis_manifest(
        basis_path,
        expected_sha256=preflight["frozen_round4b"]["basis_manifest_sha256"],
        verify_files=True,
    )
    split = _read(args.split.resolve())
    source = Path(str(split["source_manifest_path"])).resolve()
    records = {row["sample_id"]: row for row in load_manifest(source)}
    sample_id = split["holdout_sample_ids"][0]
    record = records[sample_id]
    sample_path = Path(str(record["sample_path"]))
    if not sample_path.is_absolute():
        sample_path = source.parent / sample_path
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    infer_kwargs = dict(sample["infer_action_kwargs"])
    bundle = OnlineDonorBundle.load(
        mapping_path=args.donor_mapping.resolve(),
        observation_manifest_path=args.donor_manifest.resolve(),
        observation_root=args.donor_root.resolve(),
    )
    donor = _load_donor_image(
        bundle, task_id=int(record["task_id"]), episode_id=int(record["episode_id"])
    ).to(dtype=infer_kwargs["input_image"].dtype)
    cfg = _compose(TASK_CONFIG)
    cfg.model.load_text_encoder = False
    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    set_global_seed(42, get_worker_init_fn=False)
    model = instantiate(cfg.model, model_dtype=dtype, device="cuda:0")
    _load_model_checkpoint(model, str(args.checkpoint.resolve()))
    model = model.to("cuda:0").eval()
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
            return_video_cache_stats=True,
            compile_action_infer=False,
        )
    stats = wrong["video_cache_stats"]
    layout = {
        "video_seq_len": stats["current_video_seq_len"],
        "action_visible_token_indices": stats["action_visible_video_token_indices"],
        "tokens_per_frame": stats["current_video_tokens_per_frame"],
        "num_layers": 30,
        "num_heads": stats["num_heads"],
        "head_dim": stats["head_dim"],
        "video_grid_size": stats["current_video_grid_size"],
        "input_image_shape": stats["current_input_image_shape"],
    }
    feature_dim = int(stats["layers"][15]["current"]["k"]["shape"][-1])
    expected_layout = manifest["runtime_layout"]
    if (
        feature_dim != EXPECTED_FEATURE_DIM
        or layout["video_seq_len"] != 98
        or layout["action_visible_token_indices"] != list(range(98))
        or layout["num_heads"] != 24
        or layout["head_dim"] != 128
        or expected_layout["feature_dim"] != feature_dim
    ):
        raise ValueError(f"Round-4C runtime layout mismatch: {layout}, D={feature_dim}.")
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    specs = {
        rank: load_runtime_basis(
            manifest_path=basis_path,
            expected_sha256=sha256_file(basis_path),
            basis_kind="svd",
            rank=rank,
            device=device,
            dtype=model_dtype,
        )
        for rank in RANKS
    }
    frozen_r256 = load_runtime_basis(
        manifest_path=basis_path,
        expected_sha256=sha256_file(basis_path),
        basis_kind="svd",
        rank=256,
        device=device,
        dtype=model_dtype,
    )
    prefix_rows = []
    prefixes_exact = True
    for layer in LATE_LAYERS:
        for kind in ("k", "v"):
            b36 = specs[36].bases_by_layer[layer][kind]
            b97 = specs[97].bases_by_layer[layer][kind]
            b170 = specs[170].bases_by_layer[layer][kind]
            b256 = frozen_r256.bases_by_layer[layer][kind]
            exact = bool(
                torch.equal(b36, b97[:, :36])
                and torch.equal(b97, b170[:, :97])
                and torch.equal(b170, b256[:, :170])
            )
            prefixes_exact = prefixes_exact and exact
            prefix_rows.append(
                {
                    "layer": layer,
                    "tensor_kind": kind.upper(),
                    "r36_subset_r97_subset_r170_subset_frozen_r256": exact,
                    "shapes": {"36": list(b36.shape), "97": list(b97.shape), "170": list(b170.shape)},
                }
            )
    zero = torch.empty((feature_dim, 0), device=device, dtype=model_dtype)
    identity = torch.eye(feature_dim, device=device, dtype=model_dtype)
    zero_bases = {layer: {"k": zero, "v": zero} for layer in LATE_LAYERS}
    full_bases = {layer: {"k": identity, "v": identity} for layer in LATE_LAYERS}
    projected: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        rank0 = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor,
            replacement_video_layers=LATE_LAYERS,
            feature_projection_bases_by_layer=zero_bases,
            feature_projection_rank=0,
            expected_video_cache_layout=layout,
            return_video_cache_stats=True,
            compile_action_infer=False,
        )
        rankd = model.infer_action(
            **infer_kwargs,
            disabled_video_layers=EARLY_DISABLED,
            replacement_input_image=donor,
            replacement_video_layers=LATE_LAYERS,
            feature_projection_bases_by_layer=full_bases,
            feature_projection_rank=feature_dim,
            expected_video_cache_layout=layout,
            return_video_cache_stats=True,
            compile_action_infer=False,
        )
        for rank, spec in specs.items():
            projected[rank] = model.infer_action(
                **infer_kwargs,
                disabled_video_layers=EARLY_DISABLED,
                replacement_input_image=donor,
                replacement_video_layers=LATE_LAYERS,
                return_video_cache_stats=True,
                compile_action_infer=False,
                **spec.inference_kwargs(),
            )
    rank0_comparison = _comparison(rank0["action"], wrong["action"])
    rankd_comparison = _comparison(rankd["action"], current["action"])
    output_actions = [
        current["action"],
        wrong["action"],
        rank0["action"],
        rankd["action"],
    ]
    output_actions.extend(projected[rank]["action"] for rank in RANKS)
    finite_shape = all(
        tuple(action.shape) == (32, 7) and bool(torch.isfinite(action).all().item())
        for action in output_actions
    )
    audits = {
        str(rank): projected[rank]["video_cache_stats"]["hybrid_video_cache"]
        for rank in RANKS
    }
    audit_passed = all(
        audit["projection_rank"] == rank
        and audit["replacement_video_layers"] == list(LATE_LAYERS)
        and audit["action_visible_token_count"] == 98
        and audit["tokens_modified"] is False
        and audit["heads_modified"] is False
        and audit["shape_preserved"] is True
        and audit["k_v_bases_independent"] is True
        for rank, audit in ((rank, audits[str(rank)]) for rank in RANKS)
    )
    passed = bool(
        prefixes_exact
        and rank0_comparison["torch_allclose"]
        and rankd_comparison["max_raw_action_abs_diff"] <= 2e-2
        and finite_shape
        and audit_passed
    )
    report = {
        "artifact_type": "asre_round4c_machinery_report",
        "schema_version": 1,
        "protocol": ROUND4C_PROTOCOL,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "sample_id": sample_id,
        "runtime_layout": {**layout, "feature_dim": feature_dim},
        "rank0_wrong_identity": rank0_comparison,
        "rankD_current_identity": rankd_comparison,
        "rankD_action_tolerance": 2e-2,
        "rank_prefix_integrity": {"passed": prefixes_exact, "rows": prefix_rows},
        "projected_condition_audits": audits,
        "all_action_outputs_shape_32x7_and_finite": finite_shape,
        "no_token_or_head_modification": audit_passed,
        "frozen_basis_reused_without_refit": True,
        "preflight_report_path": str(preflight_path),
        "preflight_report_sha256": sha256_file(preflight_path),
        "split_manifest_path": str(args.split.resolve()),
        "split_manifest_sha256": sha256_file(args.split.resolve()),
        "basis_manifest_path": str(basis_path),
        "basis_manifest_sha256": sha256_file(basis_path),
    }
    if not passed:
        raise RuntimeError(f"Round-4C machinery controls failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    atomic_write_json(args.output.resolve(), report)
    print(f"Round-4C machinery passed: {args.output.resolve()}")


if __name__ == "__main__":
    main()
