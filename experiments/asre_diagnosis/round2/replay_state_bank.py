"""Replay one ASRE Round-2 keep schedule over the immutable valid state set."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND2_PROTOCOL,
    atomic_write_json,
    build_run_metadata,
    get_num_model_layers,
    load_manifest,
    now_iso,
    resolve_condition,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.round2.metrics import (  # noqa: E402
    CONTINUOUS_ACTION_DIMENSIONS,
    compute_round2_metrics,
    extract_action_global_std,
)
from experiments.asre_diagnosis.round2.validate_state_bank import (  # noqa: E402
    _read_json,
    _resolve_path,
    _validate_sample_partition,
    validate_existing_artifacts,
)
from experiments.libero.eval_libero_single import (  # noqa: E402
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _postprocess_action,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _run_prepared_action_inference,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import (  # noqa: E402
    FastWAMProcessor,
)
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


SCALAR_METRICS = (
    "executed_prefix_norm_rms",
    "norm_rms_h0",
    "norm_rms_h0_h1",
    "full_chunk_norm_rms_0_31",
    "round1_raw_output_full_chunk_rms",
    "executed_prefix_cosine_similarity",
    "executed_prefix_gripper_flip_rate",
    "full_horizon_gripper_flip_rate",
    "translation_norm_rms",
    "rotation_norm_rms",
)
DIMENSION_METRIC = "executed_prefix_norm_rms_by_dimension"


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    if not records:
        return math.nan
    return float(np.mean([float(record[key]) for record in records]))


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
            handle.flush()
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)


def _strict_valid_records(
    *,
    valid_manifest: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    valid_ids = _validate_sample_partition(valid_manifest, source_records)
    by_id = {str(record["sample_id"]): record for record in source_records}
    return [by_id[identifier] for identifier in valid_ids]


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="sim_libero.yaml")
def replay_state_bank(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    if not bool(cfg.ASRE_DIAGNOSIS.get("enabled", False)):
        raise ValueError("Round-2 offline replay requires ASRE_DIAGNOSIS.enabled=true.")
    if str(cfg.ASRE_DIAGNOSIS.get("protocol", "")) != ROUND2_PROTOCOL:
        raise ValueError(
            f"Round-2 offline replay requires ASRE_DIAGNOSIS.protocol={ROUND2_PROTOCOL}."
        )
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    state_bank_dir = _resolve_path(
        cfg.ASRE_DIAGNOSIS.get("state_bank_dir"),
        label="ASRE_DIAGNOSIS.state_bank_dir",
    )
    output_root = _resolve_path(
        cfg.ASRE_DIAGNOSIS.get("offline_output_dir"),
        label="ASRE_DIAGNOSIS.offline_output_dir",
    )
    valid_manifest_path = _resolve_path(
        cfg.ASRE_DIAGNOSIS.get("valid_state_bank_manifest_path"),
        label="ASRE_DIAGNOSIS.valid_state_bank_manifest_path",
    )
    valid_manifest = _read_json(valid_manifest_path)
    prompt_cache_path = _resolve_path(
        valid_manifest.get("prompt_context_cache_path"),
        label="state_bank_valid_manifest.prompt_context_cache_path",
    )
    source_manifest_path = (state_bank_dir / "manifest.jsonl").resolve()
    source_records = load_manifest(source_manifest_path)
    checkpoint_path = _resolve_path(cfg.ckpt, label="ckpt")
    dataset_stats_path = _resolve_dataset_stats_path(cfg).resolve()
    validate_existing_artifacts(
        valid_manifest_path=valid_manifest_path,
        prompt_cache_path=prompt_cache_path,
        source_manifest_path=source_manifest_path,
        source_records=source_records,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        # The immutable QC step already hashed the multi-GB checkpoint. Eight
        # concurrent condition replays verify its absolute path and trust that
        # recorded digest instead of redundantly reading the checkpoint 8x.
        verify_checkpoint_hash=False,
    )
    selected_records = _strict_valid_records(
        valid_manifest=valid_manifest,
        source_records=source_records,
    )
    if not selected_records:
        raise ValueError("The Round-2 valid state-bank manifest contains no samples.")

    source_metadata = _read_json(state_bank_dir / "run_metadata.json")
    executed_prefix_length = int(cfg.ASRE_DIAGNOSIS.get("executed_prefix_length", 10))
    if executed_prefix_length != 10 or executed_prefix_length != int(
        source_metadata.get("replan_steps", -1)
    ):
        raise ValueError(
            "ASRE Round-2 executed_prefix_length must equal the pre-registered "
            f"replan interval of 10; got {executed_prefix_length}."
        )
    source_compatibility = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_stats_path": str(dataset_stats_path),
        "num_model_layers": 30,
        "compile_action_infer": bool(cfg.EVALUATION.get("compile_action_infer", False)),
        "binarize_gripper": bool(cfg.EVALUATION.get("binarize_gripper", False)),
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
    }
    source_mismatches = {
        key: {"state_bank": source_metadata.get(key), "replay": value}
        for key, value in source_compatibility.items()
        if source_metadata.get(key) != value
    }
    if source_mismatches:
        raise ValueError(
            "Round-2 replay configuration is incompatible with the source state bank: "
            f"{json.dumps(source_mismatches, sort_keys=True)}"
        )

    cfg.model.load_text_encoder = False
    cfg.EVALUATION.text_encoder_device = None
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(checkpoint_path))
    model = model.to(model_device).eval()
    num_layers = get_num_model_layers(model)
    condition = resolve_condition(cfg.ASRE_DIAGNOSIS, num_layers)
    cfg.ASRE_DIAGNOSIS.condition_name = condition.name
    cfg.ASRE_DIAGNOSIS.enabled_video_retrieval_layers = list(
        condition.enabled_video_retrieval_layers(num_layers)
    )
    cfg.ASRE_DIAGNOSIS.disabled_video_layers = list(condition.disabled_video_layers)

    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    action_std = extract_action_global_std(dataset_stats)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    output_dir = output_root / condition.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite completed or partial Round-2 replay output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    first_sample_path = state_bank_dir / str(selected_records[0]["sample_path"])
    first_sample = torch.load(first_sample_path, map_location="cpu", weights_only=False)
    first_infer_kwargs = first_sample["infer_action_kwargs"]
    action_horizon = int(first_infer_kwargs["action_horizon"])
    num_inference_steps = int(first_infer_kwargs["num_inference_steps"])
    task_ids = sorted({int(record["task_id"]) for record in selected_records})
    valid_manifest_sha256 = sha256_file(valid_manifest_path)
    config_sha256 = sha256_json(OmegaConf.to_container(cfg, resolve=True))
    metadata = build_run_metadata(
        repo_root=project_root,
        checkpoint=str(checkpoint_path),
        dataset_stats_path=str(dataset_stats_path),
        condition=condition,
        num_layers=num_layers,
        task_suite=str(first_sample["task_suite"]),
        task_ids=task_ids,
        seed=None if cfg.get("seed") is None else int(cfg.seed),
        num_trials=int(source_metadata["number_of_trials"]),
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        replan_steps=int(source_metadata["replan_steps"]),
        start_timestamp=now_iso(),
        condition_protocol=ROUND2_PROTOCOL,
        checkpoint_sha256=str(valid_manifest["checkpoint_sha256"]),
        dataset_stats_sha256=str(valid_manifest["dataset_stats_sha256"]),
        state_bank_manifest_path=str(source_manifest_path),
        state_bank_manifest_sha256=str(valid_manifest["source_manifest_sha256"]),
        valid_state_bank_manifest_path=str(valid_manifest_path),
        valid_state_bank_manifest_sha256=valid_manifest_sha256,
        prompt_context_cache_path=str(prompt_cache_path),
        prompt_context_cache_sha256=str(valid_manifest["prompt_context_cache_sha256"]),
        config_sha256=config_sha256,
    )
    metadata.update(
        {
            "artifact_type": "asre_round2_offline_state_bank_replay",
            "status": "running",
            "output_dir": str(output_dir),
            "num_valid_samples": len(selected_records),
            "executed_prefix_length": executed_prefix_length,
            "action_global_std": action_std.astype(float).tolist(),
            "compile_action_infer": bool(
                cfg.EVALUATION.get("compile_action_infer", False)
            ),
            "binarize_gripper": bool(cfg.EVALUATION.get("binarize_gripper", False)),
            "sigma_shift": (
                None
                if cfg.EVALUATION.get("sigma_shift") is None
                else float(cfg.EVALUATION.get("sigma_shift"))
            ),
            "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
            "text_conditioning_source": "stored_round1_state_bank_context",
            "condition_config": {
                "name": condition.name,
                "mode": str(cfg.ASRE_DIAGNOSIS.get("mode", "drop_video_kv")),
                "protocol": ROUND2_PROTOCOL,
                "enabled_video_retrieval_layers": list(
                    condition.enabled_video_retrieval_layers(num_layers)
                ),
                "disabled_video_layers": list(condition.disabled_video_layers),
            },
        }
    )
    atomic_write_json(output_dir / "run_metadata.json", metadata)

    records: list[dict[str, Any]] = []
    for index, manifest_record in enumerate(selected_records, start=1):
        sample_id = str(manifest_record["sample_id"])
        sample = torch.load(
            state_bank_dir / str(manifest_record["sample_path"]),
            map_location="cpu",
            weights_only=False,
        )
        if str(sample.get("sample_id", "")) != sample_id:
            raise ValueError(f"Sample identity mismatch for {sample_id}.")
        infer_kwargs = sample.get("infer_action_kwargs")
        if not isinstance(infer_kwargs, dict):
            raise TypeError(f"Sample {sample_id} has no infer_action_kwargs object.")
        diagnosis_raw_tensor, _ = _run_prepared_action_inference(
            model=model,
            cfg=cfg,
            infer_kwargs=infer_kwargs,
        )
        diagnosis_executed = _postprocess_action(diagnosis_raw_tensor, processor, cfg)
        baseline_raw_tensor = sample.get("baseline_raw_action")
        baseline_executed_tensor = sample.get("baseline_executed_action")
        if not isinstance(baseline_raw_tensor, torch.Tensor) or not isinstance(
            baseline_executed_tensor, torch.Tensor
        ):
            raise TypeError(f"Sample {sample_id} has no stored baseline action tensors.")
        metrics = compute_round2_metrics(
            baseline_raw=baseline_raw_tensor.detach().float().cpu().numpy(),
            diagnosis_raw=diagnosis_raw_tensor.detach().float().cpu().numpy(),
            baseline_executed=baseline_executed_tensor.detach().float().cpu().numpy(),
            diagnosis_executed=diagnosis_executed,
            action_std=action_std,
            executed_prefix_length=executed_prefix_length,
        )
        record = {
            key: manifest_record[key]
            for key in (
                "sample_id",
                "task_suite",
                "task_id",
                "task_description",
                "episode_id",
                "replan_id",
                "environment_seed",
                "environment_step",
                "action_inference_seed",
            )
        }
        record.update(
            {
                "condition": condition.name,
                "enabled_video_retrieval_layers": list(
                    condition.enabled_video_retrieval_layers(num_layers)
                ),
                "disabled_video_layers": list(condition.disabled_video_layers),
                "action_dimension_names": list(CONTINUOUS_ACTION_DIMENSIONS),
                **metrics,
            }
        )
        records.append(record)
        print(f"Replay {index}/{len(selected_records)} {condition.name}: {sample_id}")

    dimension_mean = np.mean(
        np.asarray([record[DIMENSION_METRIC] for record in records], dtype=np.float64),
        axis=0,
    )
    summary: dict[str, Any] = {
        "condition": condition.name,
        "enabled_video_retrieval_layers": json.dumps(
            list(condition.enabled_video_retrieval_layers(num_layers))
        ),
        "disabled_video_layers": json.dumps(list(condition.disabled_video_layers)),
        "num_retrieval_layers": len(condition.enabled_video_retrieval_layers(num_layers)),
        "num_samples": len(records),
        "executed_prefix_length": executed_prefix_length,
    }
    summary.update({metric: _mean(records, metric) for metric in SCALAR_METRICS})
    summary[DIMENSION_METRIC] = json.dumps(dimension_mean.astype(float).tolist())

    _atomic_write_jsonl(output_dir / "per_sample.jsonl", records)
    _write_summary_csv(output_dir / "summary.csv", summary)
    metadata["status"] = "complete"
    metadata["end_timestamp"] = now_iso()
    metadata["num_samples"] = len(records)
    atomic_write_json(output_dir / "run_metadata.json", metadata)
    print(json.dumps(summary, indent=2))
    print(f"Round-2 offline replay complete: {output_dir}")


if __name__ == "__main__":
    replay_state_bank()
