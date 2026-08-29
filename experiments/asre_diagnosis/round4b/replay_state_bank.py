"""Replay only the frozen 20-cluster Round-4B holdout partition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from hydra.utils import instantiate


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4B_PROTOCOL,
    atomic_write_json,
    build_round4b_conditions,
    git_commit,
    load_manifest,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.g0.machinery_tests import _compose  # noqa: E402
from experiments.asre_diagnosis.round3b.donor import OnlineDonorBundle  # noqa: E402
from experiments.asre_diagnosis.round3b.replay_state_bank import (  # noqa: E402
    _atomic_write_actions,
    _atomic_write_jsonl,
    _load_actions,
    _load_jsonl,
)
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    load_runtime_basis,
    validate_basis_manifest,
)
from experiments.asre_diagnosis.round4b.fit_worker import _load_donor_image  # noqa: E402
from experiments.libero.eval_libero_single import (  # noqa: E402
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _postprocess_action,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import (  # noqa: E402
    FastWAMProcessor,
)
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


TASK_CONFIG = "libero_uncond_2cam224_1e-4"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def replay(args: argparse.Namespace) -> None:
    conditions = build_round4b_conditions(30)
    condition = conditions[args.condition_index]
    if condition.name != args.condition_name:
        raise ValueError("Round-4B condition index/name mismatch.")
    physical = os.environ.get("ASRE_ROUND4B_PHYSICAL_GPU")
    if physical is None or os.environ.get("CUDA_VISIBLE_DEVICES") != physical:
        raise RuntimeError("Round-4B offline replay requires isolated GPU provenance.")
    split_path = args.split.resolve()
    split = _read(split_path)
    sample_ids = list(split["holdout_sample_ids"])
    source_path = Path(str(split["source_manifest_path"])).resolve()
    by_id = {record["sample_id"]: record for record in load_manifest(source_path)}
    bundle = OnlineDonorBundle.load(
        mapping_path=args.donor_mapping.resolve(),
        observation_manifest_path=args.donor_manifest.resolve(),
        observation_root=args.donor_root.resolve(),
    )
    basis_path = args.basis_manifest.resolve()
    basis_sha = sha256_file(basis_path)
    validate_basis_manifest(basis_path, expected_sha256=basis_sha, verify_files=False)
    cfg = _compose(TASK_CONFIG)
    cfg.model.load_text_encoder = False
    cfg.EVALUATION.text_encoder_device = None
    cfg.EVALUATION.compile_action_infer = False
    stats_path = args.dataset_stats.resolve()
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    set_global_seed(42, get_worker_init_fn=False)
    model = instantiate(cfg.model, model_dtype=dtype, device="cuda:0")
    _load_model_checkpoint(model, str(args.checkpoint.resolve()))
    model = model.to("cuda:0").eval()
    basis_spec = None
    if condition.basis_kind is not None:
        basis_spec = load_runtime_basis(
            manifest_path=basis_path,
            expected_sha256=basis_sha,
            basis_kind=condition.basis_kind,
            rank=int(condition.subspace_rank),
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
    output_dir = args.output_root.resolve() / condition.name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    records_path = output_dir / "per_sample.jsonl"
    actions_path = output_dir / "actions.npz"
    cache_path = output_dir / "video_cache_stats.jsonl"
    metadata = {
        "artifact_type": "asre_round4b_offline_holdout_replay",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "status": "running",
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "condition_index": args.condition_index,
        "diagnosis_condition": condition.name,
        "condition_config": condition.to_dict(),
        "subspace_basis_kind": condition.basis_kind,
        "subspace_rank": condition.subspace_rank,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "dataset_stats_path": str(stats_path),
        "dataset_stats_sha256": sha256_file(stats_path),
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "basis_manifest_path": str(basis_path),
        "basis_manifest_sha256": basis_sha,
        "donor_mapping_path": str(bundle.mapping_path),
        "donor_mapping_sha256": bundle.mapping_sha256,
        "donor_manifest_path": str(bundle.observation_manifest_path),
        "donor_manifest_sha256": bundle.observation_manifest_sha256,
        "donor_semantics": "same-task next-trial fixed first-policy-query image",
        "num_holdout_episode_clusters": 20,
        "num_holdout_samples": len(sample_ids),
        "physical_gpu": int(physical),
        "logical_device": "cuda:0",
        "no_ddp": True,
        "start_timestamp": now_iso(),
        "end_timestamp": None,
        "completed_samples": 0,
    }
    records: list[dict[str, Any]] = []
    raw_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    cache_records: list[dict[str, Any]] = []
    if metadata_path.exists():
        existing = _read(metadata_path)
        for key in (
            "protocol",
            "git_commit_hash",
            "diagnosis_condition",
            "checkpoint_sha256",
            "split_manifest_sha256",
            "basis_manifest_sha256",
            "donor_mapping_sha256",
        ):
            if existing.get(key) != metadata.get(key):
                raise FileExistsError(f"Incompatible offline resume field: {key}.")
        if records_path.exists() and actions_path.exists():
            records = _load_jsonl(records_path)
            saved_ids, raw_actions, executed_actions = _load_actions(actions_path)
            if saved_ids != sample_ids[: len(saved_ids)] or saved_ids != [
                record["sample_id"] for record in records
            ]:
                raise ValueError("Offline resume is not an exact holdout-order prefix.")
        if cache_path.exists():
            cache_records = _load_jsonl(cache_path)
        if existing.get("status") == "complete":
            if len(records) != len(sample_ids):
                raise ValueError("Completed offline output is incomplete.")
            print(f"Round-4B offline already complete: {condition.name}")
            return
        metadata["start_timestamp"] = existing.get("start_timestamp")
    elif any(output_dir.iterdir()):
        raise FileExistsError(f"Unidentified offline output: {output_dir}")
    metadata["completed_samples"] = len(records)
    atomic_write_json(metadata_path, metadata)

    for index, sample_id in enumerate(sample_ids, start=1):
        if index <= len(records):
            continue
        manifest_record = by_id[sample_id]
        sample_path = Path(str(manifest_record["sample_path"]))
        if not sample_path.is_absolute():
            sample_path = source_path.parent / sample_path
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        infer_kwargs = dict(sample["infer_action_kwargs"])
        donor = _load_donor_image(
            bundle,
            task_id=int(manifest_record["task_id"]),
            episode_id=int(manifest_record["episode_id"]),
        ).to(dtype=infer_kwargs["input_image"].dtype)
        call: dict[str, Any] = {
            **infer_kwargs,
            "disabled_video_layers": condition.disabled_video_layers,
            "compile_action_infer": False,
        }
        if condition.replacement_video_layers:
            call.update(
                {
                    "replacement_input_image": donor,
                    "replacement_video_layers": condition.replacement_video_layers,
                    "return_video_cache_stats": index == 1,
                }
            )
        if basis_spec is not None:
            call.update(basis_spec.inference_kwargs())
        prediction = model.infer_action(**call)
        raw = prediction["action"]
        if tuple(raw.shape) != (32, 7) or not bool(torch.isfinite(raw).all().item()):
            raise ValueError(f"Malformed action for {sample_id}.")
        executed = _postprocess_action(raw, processor, cfg)
        raw_actions.append(raw.float().cpu().numpy())
        executed_actions.append(np.asarray(executed, dtype=np.float32))
        records.append(
            {
                "sample_id": sample_id,
                "task_suite": manifest_record["task_suite"],
                "task_id": int(manifest_record["task_id"]),
                "episode_id": int(manifest_record["episode_id"]),
                "replan_id": int(manifest_record["replan_id"]),
                "condition": condition.name,
                "basis_kind": condition.basis_kind,
                "rank": condition.subspace_rank,
                "donor_trial": int(
                    bundle.mappings[
                        (int(manifest_record["task_id"]), int(manifest_record["episode_id"]))
                    ]["donor_trial"]
                ),
            }
        )
        if index == 1 and condition.replacement_video_layers:
            cache_records.append(
                {"sample_id": sample_id, **prediction["video_cache_stats"]}
            )
        if index % 5 == 0 or index == len(sample_ids):
            _atomic_write_jsonl(records_path, records)
            _atomic_write_actions(
                actions_path,
                sample_ids=[record["sample_id"] for record in records],
                raw_actions=raw_actions,
                executed_actions=executed_actions,
            )
            if cache_records:
                _atomic_write_jsonl(cache_path, cache_records)
            metadata["completed_samples"] = len(records)
            atomic_write_json(metadata_path, metadata)
            print(
                f"[Round4B offline {condition.name}] {len(records)}/{len(sample_ids)}",
                flush=True,
            )
    metadata["status"] = "complete"
    metadata["end_timestamp"] = now_iso()
    atomic_write_json(metadata_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-index", type=int, choices=range(8), required=True)
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
