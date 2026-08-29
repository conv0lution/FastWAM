"""Fit one layer shard or score held-out energy for Round-4B."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from hydra.utils import instantiate


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
from experiments.asre_diagnosis.g0.machinery_tests import _compose  # noqa: E402
from experiments.asre_diagnosis.round3b.donor import (  # noqa: E402
    OnlineDonorBundle,
    tensor_is_finite,
    tensor_sha256,
)
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
    RANKS,
    validate_basis_manifest,
)
from experiments.libero.eval_libero_single import (  # noqa: E402
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


EARLY_DISABLED = tuple(range(15))
TASK_CONFIG = "libero_uncond_2cam224_1e-4"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
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


def _load_donor_image(
    bundle: OnlineDonorBundle, *, task_id: int, episode_id: int
) -> torch.Tensor:
    mapping = bundle.mappings[(task_id, episode_id)]
    donor_key = (task_id, int(mapping["donor_trial"]))
    observation = bundle.observations[donor_key]
    path = (bundle.observation_root / observation["artifact_relative_path"]).resolve()
    if sha256_file(path) != observation["artifact_sha256"]:
        raise ValueError(f"Donor artifact drifted: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    image = payload.get("input_image") if isinstance(payload, Mapping) else None
    if not torch.is_tensor(image) or not tensor_is_finite(image):
        raise ValueError(f"Malformed donor image: {path}")
    if tensor_sha256(image) != observation["processed_image_sha256"]:
        raise ValueError(f"Donor image identity mismatch: {path}")
    return image


def _load_model(checkpoint: Path) -> tuple[torch.nn.Module, Any]:
    cfg = _compose(TASK_CONFIG)
    cfg.model.load_text_encoder = False
    cfg.EVALUATION.text_encoder_device = None
    cfg.EVALUATION.compile_action_infer = False
    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=dtype, device="cuda:0")
    _load_model_checkpoint(model, str(checkpoint))
    return model.to("cuda:0").eval(), cfg


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Round-4B fitting requires CUDA.")
    physical = os.environ.get("ASRE_ROUND4B_PHYSICAL_GPU")
    if physical is None or os.environ.get("CUDA_VISIBLE_DEVICES") != physical:
        raise RuntimeError("Round-4B worker requires an auditable isolated physical GPU.")
    if args.worker_index not in range(4):
        raise ValueError("worker-index must be 0..3.")
    layers = LATE_LAYERS[args.worker_index :: 4]
    split_path = args.split.resolve()
    split = _read(split_path)
    if split.get("protocol") != ROUND4B_PROTOCOL:
        raise ValueError("Incompatible Round-4B split manifest.")
    sample_ids = list(
        split["fit_sample_ids"]
        if args.phase == "fit"
        else split["holdout_sample_ids"]
    )
    source_path = Path(str(split["source_manifest_path"])).resolve()
    records = load_manifest(source_path)
    by_id = {str(record["sample_id"]): record for record in records}
    if any(identifier not in by_id for identifier in sample_ids):
        raise ValueError("Split contains sample IDs absent from the source state bank.")
    bundle = OnlineDonorBundle.load(
        mapping_path=args.donor_mapping.resolve(),
        observation_manifest_path=args.donor_manifest.resolve(),
        observation_root=args.donor_root.resolve(),
    )
    checkpoint = args.checkpoint.resolve()
    set_global_seed(42, get_worker_init_fn=False)
    model, _cfg = _load_model(checkpoint)

    grams: dict[str, dict[int, torch.Tensor]] = {}
    totals: dict[str, dict[int, float]] = {}
    captured: dict[str, dict[int, dict[int, float]]] = {}
    bases: dict[str, dict[int, torch.Tensor]] = {}
    if args.phase == "fit":
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
    else:
        if args.basis_manifest is None:
            raise ValueError("Holdout scoring requires --basis-manifest.")
        manifest = validate_basis_manifest(args.basis_manifest.resolve())
        for kind in ("k", "v"):
            totals[kind] = {layer: 0.0 for layer in layers}
            captured[kind] = {
                layer: {rank: 0.0 for rank in RANKS} for layer in layers
            }
            bases[kind] = {}
            for layer in layers:
                record = manifest["artifacts"]["svd"][str(layer)][kind]
                artifact = torch.load(record["path"], map_location="cpu", weights_only=False)
                bases[kind][layer] = artifact["basis"].to(
                    device="cuda:0", dtype=torch.float32
                )

    layout: dict[str, Any] | None = None
    row_count = 0
    state_root = source_path.parent
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
            raise ValueError("Runtime cache layout drifted during calibration.")
        deltas = result["video_cache_deltas"]
        row_count += int(observed_layout["action_visible_token_count"])
        for kind in ("k", "v"):
            for layer in layers:
                values = deltas[kind][layer].reshape(-1, EXPECTED_FEATURE_DIM).float()
                if not bool(torch.isfinite(values).all().item()):
                    raise ValueError(f"Nonfinite cache delta for {identifier}.")
                if args.phase == "fit":
                    grams[kind][layer].addmm_(values.T, values)
                else:
                    totals[kind][layer] += float(values.square().sum().item())
                    coefficients = values @ bases[kind][layer]
                    for rank in RANKS:
                        captured[kind][layer][rank] += float(
                            coefficients[:, :rank].square().sum().item()
                        )
        del result, deltas
        if position % 25 == 0:
            print(
                f"[Round4B {args.phase} shard {args.worker_index}] "
                f"{position}/{len(sample_ids)} states",
                flush=True,
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    if args.phase == "fit":
        gram_root = output_dir / "grams"
        for kind in ("k", "v"):
            for layer in layers:
                path = gram_root / f"layer{layer:02d}_{kind}.pt"
                _atomic_torch_save(
                    path,
                    {
                        "artifact_type": "asre_round4b_uncentered_gram",
                        "protocol": ROUND4B_PROTOCOL,
                        "layer": layer,
                        "tensor_kind": kind,
                        "rows": row_count,
                        "sample_count": len(sample_ids),
                        "split_sha256": sha256_file(split_path),
                        "gram": grams[kind][layer].cpu(),
                    },
                )
                artifacts[f"layer{layer:02d}_{kind}"] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "shape": [EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_DIM],
                }
    else:
        for kind in ("k", "v"):
            for layer in layers:
                total = totals[kind][layer]
                artifacts[f"layer{layer:02d}_{kind}"] = {
                    "total_energy": total,
                    "captured_energy": {
                        str(rank): captured[kind][layer][rank] for rank in RANKS
                    },
                    "captured_fraction": {
                        str(rank): captured[kind][layer][rank] / total
                        for rank in RANKS
                    },
                }
    report = {
        "artifact_type": f"asre_round4b_{args.phase}_shard_report",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "status": "complete",
        "created_at": now_iso(),
        "phase": args.phase,
        "worker_index": args.worker_index,
        "physical_gpu": int(physical),
        "logical_device": "cuda:0",
        "no_ddp": True,
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "layers": list(layers),
        "sample_count": len(sample_ids),
        "sample_ids_sha256": sha256_json(sample_ids),
        "row_count_per_matrix": row_count,
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "basis_manifest_sha256": (
            None
            if args.basis_manifest is None
            else sha256_file(args.basis_manifest.resolve())
        ),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "donor_mapping_sha256": bundle.mapping_sha256,
        "donor_manifest_sha256": bundle.observation_manifest_sha256,
        "runtime_layout": layout,
        "artifacts": artifacts,
    }
    report_path = output_dir / f"{args.phase}_shard{args.worker_index}.json"
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "holdout"), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--basis-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    report = run_worker(parser.parse_args())
    print(
        f"Round-4B {report['phase']} shard {report['worker_index']} complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
