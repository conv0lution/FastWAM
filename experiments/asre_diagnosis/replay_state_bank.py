"""Replay one video-layer diagnosis condition over a fixed LIBERO state bank."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Offline replay never creates a simulator or EGL context. An EGL device left
# over from a previous online run can nevertheless make robosuite fail during
# import when each replay process exposes a different single GPU.
os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _postprocess_action,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _run_prepared_action_inference,
)
from experiments.asre_diagnosis.common import (
    atomic_write_json,
    build_run_metadata,
    get_num_model_layers,
    load_manifest,
    now_iso,
    resolve_condition,
)
from experiments.asre_diagnosis.metrics import compute_action_deviation_metrics
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records])) if records else math.nan


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def replay_state_bank(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    state_bank_value = cfg.ASRE_DIAGNOSIS.get("state_bank_dir")
    output_value = cfg.ASRE_DIAGNOSIS.get("offline_output_dir")
    if state_bank_value is None or output_value is None:
        raise ValueError(
            "Set ASRE_DIAGNOSIS.state_bank_dir and ASRE_DIAGNOSIS.offline_output_dir."
        )
    if not bool(cfg.ASRE_DIAGNOSIS.get("enabled", False)):
        raise ValueError("Offline replay requires ASRE_DIAGNOSIS.enabled=true.")
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    state_bank_dir = Path(os.path.expanduser(os.path.expandvars(str(state_bank_value)))).resolve()
    manifest = load_manifest(state_bank_dir / "manifest.jsonl")
    if not manifest:
        raise ValueError(f"State bank manifest is empty: {state_bank_dir / 'manifest.jsonl'}")
    with (state_bank_dir / "run_metadata.json").open("r", encoding="utf-8") as handle:
        state_bank_metadata = json.load(handle)

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    # Samples already contain context/context_mask, so T5 is unused here.
    cfg.model.load_text_encoder = False
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    num_layers = get_num_model_layers(model)
    condition = resolve_condition(cfg.ASRE_DIAGNOSIS, num_layers)
    cfg.ASRE_DIAGNOSIS.condition_name = condition.name
    cfg.ASRE_DIAGNOSIS.disabled_video_layers = list(condition.disabled_video_layers)

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    expected_checkpoint = str(
        Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt)))).resolve()
    )
    expected_stats = str(dataset_stats_path.resolve())
    compatibility_checks = {
        "checkpoint_path": expected_checkpoint,
        "dataset_stats_path": expected_stats,
        "num_model_layers": num_layers,
        "compile_action_infer": bool(cfg.EVALUATION.get("compile_action_infer", False)),
        "binarize_gripper": bool(cfg.EVALUATION.get("binarize_gripper", False)),
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
    }
    mismatches = {
        key: {"state_bank": state_bank_metadata.get(key), "replay": expected}
        for key, expected in compatibility_checks.items()
        if state_bank_metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Offline replay configuration does not exactly match the state bank: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    output_root = Path(os.path.expanduser(os.path.expandvars(str(output_value)))).resolve()
    output_dir = output_root / condition.name
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample.jsonl"
    if per_sample_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed/partial replay output: {per_sample_path}"
        )

    first_sample = torch.load(state_bank_dir / manifest[0]["sample_path"], weights_only=False)
    action_horizon = int(first_sample["infer_action_kwargs"]["action_horizon"])
    num_inference_steps = int(first_sample["infer_action_kwargs"]["num_inference_steps"])
    task_ids = sorted({int(record["task_id"]) for record in manifest})
    run_start = now_iso()
    metadata = build_run_metadata(
        repo_root=project_root,
        checkpoint=str(cfg.ckpt),
        dataset_stats_path=str(dataset_stats_path),
        condition=condition,
        num_layers=num_layers,
        task_suite=str(first_sample["task_suite"]),
        task_ids=task_ids,
        seed=None if cfg.get("seed") is None else int(cfg.seed),
        num_trials=int(state_bank_metadata["number_of_trials"]),
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        replan_steps=int(state_bank_metadata["replan_steps"]),
        start_timestamp=run_start,
    )
    metadata["artifact_type"] = "offline_state_bank_replay"
    metadata["state_bank_dir"] = str(state_bank_dir)
    metadata.update(
        {
            "compile_action_infer": compatibility_checks["compile_action_infer"],
            "binarize_gripper": compatibility_checks["binarize_gripper"],
            "sigma_shift": compatibility_checks["sigma_shift"],
            "rand_device": compatibility_checks["rand_device"],
        }
    )
    atomic_write_json(output_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))

    records: list[dict[str, Any]] = []
    with per_sample_path.open("w", encoding="utf-8") as output_handle:
        for manifest_record in manifest:
            sample = torch.load(
                state_bank_dir / manifest_record["sample_path"],
                weights_only=False,
            )
            infer_kwargs = sample["infer_action_kwargs"]
            diagnosis_raw_tensor, _ = _run_prepared_action_inference(
                model=model,
                cfg=cfg,
                infer_kwargs=infer_kwargs,
            )
            diagnosis_executed = _postprocess_action(diagnosis_raw_tensor, processor, cfg)
            baseline_raw = sample["baseline_raw_action"].float().numpy()
            diagnosis_raw = diagnosis_raw_tensor.detach().float().cpu().numpy()
            baseline_executed = sample["baseline_executed_action"].float().numpy()
            metrics = compute_action_deviation_metrics(
                baseline_raw=baseline_raw,
                diagnosis_raw=diagnosis_raw,
                baseline_executed=baseline_executed,
                diagnosis_executed=diagnosis_executed,
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
                    "disabled_video_layers": list(condition.disabled_video_layers),
                    **metrics,
                }
            )
            output_handle.write(json.dumps(record, sort_keys=True) + "\n")
            output_handle.flush()
            records.append(record)
            print(f"replayed {condition.name}: {record['sample_id']}")

    summary = {
        "condition": condition.name,
        "disabled_video_layers": json.dumps(list(condition.disabled_video_layers)),
        "num_samples": len(records),
        "offline_continuous_action_mae": _mean(records, "continuous_action_mae"),
        "offline_normalized_l2": _mean(records, "normalized_continuous_action_l2"),
        "offline_cosine_similarity": _mean(records, "continuous_action_cosine_similarity"),
        "raw_gripper_difference": _mean(records, "raw_gripper_difference"),
        "gripper_flip_rate": _mean(records, "post_binarization_gripper_flip_rate"),
    }
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    metadata["end_timestamp"] = now_iso()
    metadata["num_samples"] = len(records)
    atomic_write_json(output_dir / "run_metadata.json", metadata)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    replay_state_bank()
