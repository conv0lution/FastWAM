"""Replay one Round-3B condition over the immutable 499-state bank."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND3B_PROTOCOL,
    atomic_write_json,
    build_round3b_conditions,
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
from experiments.asre_diagnosis.round3b.offline_donor import (  # noqa: E402
    load_offline_donor_manifest,
    tensor_sha256,
)
from experiments.libero.eval_libero_single import (  # noqa: E402
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _postprocess_action,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
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


def _require_clean_worktree() -> None:
    try:
        output = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                ".",
                ":(exclude)asre_results/round3b/**",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot inspect the Git worktree before replay: {exc}") from exc
    if output.strip():
        raise RuntimeError(
            "ASRE Round 3B offline replay requires a clean Git worktree. Commit the "
            "reviewed source implementation first; generated Round-3B artifacts are "
            "exempt and no changes are discarded automatically.\n"
            f"git status --porcelain:\n{output.rstrip()}"
        )


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
        temporary = Path(handle.name)
        try:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _atomic_write_actions(
    path: Path,
    *,
    sample_ids: Sequence[str],
    raw_actions: Sequence[np.ndarray],
    executed_actions: Sequence[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(raw_actions, dtype=np.float32)
    executed = np.asarray(executed_actions, dtype=np.float32)
    if raw.shape != executed.shape or raw.ndim != 3 or raw.shape[-1] != 7:
        raise ValueError(
            f"Round-3B saved actions must align as [N,H,7], got {raw.shape}/{executed.shape}."
        )
    if raw.shape[0] != len(sample_ids):
        raise ValueError("Round-3B action count does not match sample IDs.")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(executed)):
        raise ValueError("Refusing to save nonfinite Round-3B actions.")
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(
                handle,
                sample_ids=np.asarray(sample_ids, dtype=str),
                raw_actions=raw,
                executed_actions=executed,
            )
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


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
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _strict_valid_records(
    *,
    valid_manifest: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    valid_ids = _validate_sample_partition(valid_manifest, source_records)
    by_id = {str(record["sample_id"]): record for record in source_records}
    return [by_id[identifier] for identifier in valid_ids]


def _load_donor_lookup(
    *,
    mapping_path: Path,
    valid_manifest_sha256: str,
    source_manifest_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    payload = load_offline_donor_manifest(mapping_path)
    if payload.get("valid_manifest_sha256") != valid_manifest_sha256:
        raise ValueError("Offline donor mapping valid-manifest SHA256 mismatch.")
    if payload.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("Offline donor mapping source-manifest SHA256 mismatch.")
    entries = payload["entries"]
    lookup = {str(entry["recipient_sample_id"]): entry for entry in entries}
    if len(lookup) != 499:
        raise ValueError("Offline donor mapping must have 499 unique recipients.")
    return lookup, payload


def _run_model(
    *,
    model: torch.nn.Module,
    cfg: DictConfig,
    infer_kwargs: Mapping[str, Any],
    disabled_video_layers: Sequence[int],
    replacement_video_layers: Sequence[int],
    replacement_input_image: torch.Tensor | None,
    return_video_cache_stats: bool,
) -> dict[str, Any]:
    call_kwargs = dict(infer_kwargs)
    call_kwargs.update(
        {
            "disabled_video_layers": tuple(int(value) for value in disabled_video_layers),
            "replacement_video_layers": tuple(
                int(value) for value in replacement_video_layers
            ),
            "replacement_input_image": replacement_input_image,
            "return_video_cache_stats": bool(return_video_cache_stats),
        }
    )
    with torch.no_grad():
        prediction = model.infer_action(
            **call_kwargs,
            compile_action_infer=bool(cfg.EVALUATION.get("compile_action_infer", False)),
        )
    if not isinstance(prediction, Mapping) or not isinstance(
        prediction.get("action"), torch.Tensor
    ):
        raise TypeError("Fast-WAM infer_action did not return an action tensor mapping.")
    return dict(prediction)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="sim_libero.yaml")
def replay_state_bank(cfg: DictConfig) -> None:
    _require_clean_worktree()
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    diagnosis_cfg = cfg.ASRE_DIAGNOSIS
    if not bool(diagnosis_cfg.get("enabled", False)):
        raise ValueError("Round-3B offline replay requires ASRE_DIAGNOSIS.enabled=true.")
    if str(diagnosis_cfg.get("protocol", "")) != ROUND3B_PROTOCOL:
        raise ValueError(
            f"Round-3B offline replay requires ASRE_DIAGNOSIS.protocol={ROUND3B_PROTOCOL}."
        )
    if str(diagnosis_cfg.get("mode", "")) != "replace_video_kv":
        raise ValueError("Round-3B offline replay requires mode=replace_video_kv.")
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    state_bank_dir = _resolve_path(
        diagnosis_cfg.get("state_bank_dir"), label="ASRE_DIAGNOSIS.state_bank_dir"
    )
    output_root = _resolve_path(
        diagnosis_cfg.get("offline_output_dir"),
        label="ASRE_DIAGNOSIS.offline_output_dir",
    )
    valid_manifest_path = _resolve_path(
        diagnosis_cfg.get("valid_state_bank_manifest_path"),
        label="ASRE_DIAGNOSIS.valid_state_bank_manifest_path",
    )
    donor_mapping_path = _resolve_path(
        diagnosis_cfg.get("offline_donor_mapping_path"),
        label="ASRE_DIAGNOSIS.offline_donor_mapping_path",
    )
    valid_manifest = _read_json(valid_manifest_path)
    valid_manifest_sha256 = sha256_file(valid_manifest_path)
    prompt_cache_path = _resolve_path(
        valid_manifest.get("prompt_context_cache_path"),
        label="state_bank_valid_manifest.prompt_context_cache_path",
    )
    source_manifest_path = (state_bank_dir / "manifest.jsonl").resolve()
    source_records = load_manifest(source_manifest_path)
    checkpoint_path = _resolve_path(cfg.ckpt, label="ckpt")
    dataset_stats_path = _resolve_dataset_stats_path(cfg).resolve()
    trusted_parent_digest = os.environ.get(
        "ASRE_ROUND3B_TRUSTED_PREFLIGHT_MANIFEST_SHA256"
    )
    validate_existing_artifacts(
        valid_manifest_path=valid_manifest_path,
        prompt_cache_path=prompt_cache_path,
        source_manifest_path=source_manifest_path,
        source_records=source_records,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        verify_checkpoint_hash=(trusted_parent_digest != valid_manifest_sha256),
    )
    selected_records = _strict_valid_records(
        valid_manifest=valid_manifest, source_records=source_records
    )
    if len(selected_records) != 499:
        raise ValueError(f"Round-3B requires exactly 499 valid states, got {len(selected_records)}.")
    donor_lookup, donor_mapping = _load_donor_lookup(
        mapping_path=donor_mapping_path,
        valid_manifest_sha256=valid_manifest_sha256,
        source_manifest_sha256=sha256_file(source_manifest_path),
    )
    expected_ids = [str(record["sample_id"]) for record in selected_records]
    if set(donor_lookup) != set(expected_ids):
        raise ValueError("Offline donor mapping recipients differ from valid state set.")

    source_metadata = _read_json(state_bank_dir / "run_metadata.json")
    executed_prefix_length = int(diagnosis_cfg.get("executed_prefix_length", 10))
    if executed_prefix_length != 10 or executed_prefix_length != int(
        source_metadata.get("replan_steps", -1)
    ):
        raise ValueError("Round-3B executed-prefix length must equal replan interval 10.")
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
    mismatches = {
        key: {"state_bank": source_metadata.get(key), "replay": value}
        for key, value in source_compatibility.items()
        if source_metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Round-3B replay configuration is incompatible with the state bank: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )

    cfg.model.load_text_encoder = False
    cfg.EVALUATION.text_encoder_device = None
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(checkpoint_path))
    model = model.to(model_device).eval()
    num_layers = get_num_model_layers(model)
    condition = resolve_condition(diagnosis_cfg, num_layers)
    condition_index = build_round3b_conditions(num_layers).index(condition)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices != str(condition_index):
        raise ValueError(
            "Round-3B offline GPU mapping is condition 0/1/2 -> physical GPU 0/1/2; "
            f"condition={condition.name}, expected {condition_index}, got "
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}."
        )
    if str(model_device) != "cuda:0":
        raise ValueError(f"Round-3B replay requires logical cuda:0, got {model_device}.")

    cfg.ASRE_DIAGNOSIS.condition_name = condition.name
    cfg.ASRE_DIAGNOSIS.enabled_video_retrieval_layers = list(
        condition.enabled_video_retrieval_layers(num_layers)
    )
    cfg.ASRE_DIAGNOSIS.disabled_video_layers = list(condition.disabled_video_layers)
    cfg.ASRE_DIAGNOSIS.replacement_video_layers = list(
        condition.replacement_video_layers
    )

    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    action_std = extract_action_global_std(dataset_stats)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    output_dir = output_root / condition.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Round-3B replay output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    first_sample = torch.load(
        state_bank_dir / str(selected_records[0]["sample_path"]),
        map_location="cpu",
        weights_only=False,
    )
    first_infer_kwargs = first_sample["infer_action_kwargs"]
    action_horizon = int(first_infer_kwargs["action_horizon"])
    num_inference_steps = int(first_infer_kwargs["num_inference_steps"])
    task_ids = sorted({int(record["task_id"]) for record in selected_records})
    config_sha256 = sha256_json(OmegaConf.to_container(cfg, resolve=True))
    metadata = build_run_metadata(
        repo_root=PROJECT_ROOT,
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
        condition_protocol=ROUND3B_PROTOCOL,
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
            "artifact_type": "asre_round3b_offline_state_bank_replay",
            "status": "running",
            "output_dir": str(output_dir),
            "num_valid_samples": len(selected_records),
            "executed_prefix_length": executed_prefix_length,
            "action_global_std": action_std.astype(float).tolist(),
            "compile_action_infer": bool(
                cfg.EVALUATION.get("compile_action_infer", False)
            ),
            "binarize_gripper": bool(cfg.EVALUATION.get("binarize_gripper", False)),
            "sigma_shift": source_compatibility["sigma_shift"],
            "rand_device": source_compatibility["rand_device"],
            "text_conditioning_source": "stored_round1_state_bank_context",
            "condition_index": condition_index,
            "physical_gpu": condition_index,
            "cuda_visible_devices": cuda_visible_devices,
            "model_device": str(model_device),
            "round3a_parent_tag": "ASRE-round3a-factorial",
            "round3a_parent_commit": "d36383c16d974ba1e5a750c088327a7a88baa8fb",
            "validated_round3a_run_commit": "92e842c79f5209b31c2653944cc0c1719e95eb9e",
            "offline_donor_mapping_path": str(donor_mapping_path),
            "offline_donor_mapping_sha256": sha256_file(donor_mapping_path),
            "offline_donor_mapping_rule": donor_mapping["mapping_rule"],
            "replacement_semantics": (
                "fixed donor image per saved donor state; current recipient cached text "
                "context and proprioception are retained when recomputing donor video K/V"
            ),
            "condition_config": condition.to_dict(),
        }
    )
    atomic_write_json(output_dir / "run_metadata.json", metadata)

    records: list[dict[str, Any]] = []
    cache_stats_records: list[dict[str, Any]] = []
    raw_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
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

        donor_entry = donor_lookup[sample_id]
        replacement_image: torch.Tensor | None = None
        donor_sample_id: str | None = None
        if condition.replacement_video_layers:
            donor_sample_id = str(donor_entry["donor_sample_id"])
            donor_record = next(
                record
                for record in source_records
                if str(record["sample_id"]) == donor_sample_id
            )
            donor_sample = torch.load(
                state_bank_dir / str(donor_record["sample_path"]),
                map_location="cpu",
                weights_only=False,
            )
            replacement_image = donor_sample["infer_action_kwargs"]["input_image"]
            if tensor_sha256(infer_kwargs["input_image"]) != str(
                donor_entry["recipient_image_sha256"]
            ):
                raise ValueError(f"Recipient image hash drift for {sample_id}.")
            if tensor_sha256(replacement_image) != str(donor_entry["donor_image_sha256"]):
                raise ValueError(f"Donor image hash drift for {sample_id}->{donor_sample_id}.")
            if tuple(replacement_image.shape) != tuple(infer_kwargs["input_image"].shape):
                raise ValueError(f"Donor image shape mismatch for {sample_id}.")

        prediction = _run_model(
            model=model,
            cfg=cfg,
            infer_kwargs=infer_kwargs,
            disabled_video_layers=condition.disabled_video_layers,
            replacement_video_layers=condition.replacement_video_layers,
            replacement_input_image=replacement_image,
            return_video_cache_stats=bool(condition.replacement_video_layers),
        )
        diagnosis_raw_tensor = prediction["action"]
        if not bool(torch.isfinite(diagnosis_raw_tensor).all().item()):
            raise ValueError(f"Nonfinite action for {sample_id}.")
        diagnosis_executed = _postprocess_action(diagnosis_raw_tensor, processor, cfg)
        baseline_raw_tensor = sample.get("baseline_raw_action")
        baseline_executed_tensor = sample.get("baseline_executed_action")
        if not isinstance(baseline_raw_tensor, torch.Tensor) or not isinstance(
            baseline_executed_tensor, torch.Tensor
        ):
            raise TypeError(f"Sample {sample_id} has no stored baseline action tensors.")
        diagnosis_raw = diagnosis_raw_tensor.detach().float().cpu().numpy()
        metrics = compute_round2_metrics(
            baseline_raw=baseline_raw_tensor.detach().float().cpu().numpy(),
            diagnosis_raw=diagnosis_raw,
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
                "replacement_video_layers": list(condition.replacement_video_layers),
                "donor_sample_id": donor_sample_id,
                "donor_episode_id": (
                    None if donor_sample_id is None else int(donor_entry["donor_episode_id"])
                ),
                "donor_pixel_mae": (
                    None if donor_sample_id is None else float(donor_entry["pixel_mae"])
                ),
                "action_dimension_names": list(CONTINUOUS_ACTION_DIMENSIONS),
                **metrics,
            }
        )
        records.append(record)
        raw_actions.append(diagnosis_raw)
        executed_actions.append(np.asarray(diagnosis_executed, dtype=np.float32))

        if condition.replacement_video_layers:
            cache_stats = prediction.get("video_cache_stats")
            if not isinstance(cache_stats, Mapping):
                raise TypeError(f"Missing video_cache_stats for {sample_id}.")
            if list(cache_stats.get("replacement_video_layers", [])) != list(
                condition.replacement_video_layers
            ):
                raise ValueError(f"Cache audit replacement-layer mismatch for {sample_id}.")
            layers = cache_stats.get("layers")
            if not isinstance(layers, list) or len(layers) != num_layers:
                raise ValueError(f"Cache audit must contain {num_layers} layers for {sample_id}.")
            for layer in range(15, 30):
                layer_stats = layers[layer]
                if int(layer_stats.get("layer", -1)) != layer:
                    raise ValueError(f"Cache audit layer ordering mismatch for {sample_id}.")
                if str(layer_stats.get("selected_source")) != "replacement":
                    raise ValueError(f"Layer {layer} did not select replacement for {sample_id}.")
                for source in ("current", "replacement"):
                    for key in ("k", "v"):
                        stats = layer_stats[source][key]
                        if not bool(stats.get("finite", False)):
                            raise ValueError(
                                f"Nonfinite {source} {key} cache at layer {layer}, {sample_id}."
                            )
            cache_stats_records.append(
                {
                    "sample_id": sample_id,
                    "donor_sample_id": donor_sample_id,
                    **dict(cache_stats),
                }
            )
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
        "replacement_video_layers": json.dumps(list(condition.replacement_video_layers)),
        "num_retrieval_layers": len(condition.enabled_video_retrieval_layers(num_layers)),
        "num_replacement_layers": len(condition.replacement_video_layers),
        "num_samples": len(records),
        "executed_prefix_length": executed_prefix_length,
    }
    summary.update({metric: _mean(records, metric) for metric in SCALAR_METRICS})
    summary[DIMENSION_METRIC] = json.dumps(dimension_mean.astype(float).tolist())

    _atomic_write_jsonl(output_dir / "per_sample.jsonl", records)
    _atomic_write_actions(
        output_dir / "actions.npz",
        sample_ids=expected_ids,
        raw_actions=raw_actions,
        executed_actions=executed_actions,
    )
    if cache_stats_records:
        _atomic_write_jsonl(output_dir / "video_cache_stats.jsonl", cache_stats_records)
    _write_summary_csv(output_dir / "summary.csv", summary)
    metadata.update(
        {
            "status": "complete",
            "end_timestamp": now_iso(),
            "num_samples": len(records),
            "actions_sha256": sha256_file(output_dir / "actions.npz"),
            "cache_stats_sha256": (
                sha256_file(output_dir / "video_cache_stats.jsonl")
                if cache_stats_records
                else None
            ),
        }
    )
    atomic_write_json(output_dir / "run_metadata.json", metadata)
    print(json.dumps(summary, indent=2))
    print(f"Round-3B offline replay complete: {output_dir}")


if __name__ == "__main__":
    replay_state_bank()
