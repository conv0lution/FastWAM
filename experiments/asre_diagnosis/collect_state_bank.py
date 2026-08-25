"""Collect model-ready fixed states from baseline LIBERO rollouts."""

from __future__ import annotations

import json
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

from experiments.libero.action_ensembler import ActionEnsembler
from experiments.libero.eval_libero_single import (
    _get_max_steps,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _place_text_encoder,
    _postprocess_action,
    _prepare_action_inference,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _run_prepared_action_inference,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
)
from experiments.asre_diagnosis.common import (
    DiagnosisCondition,
    atomic_write_json,
    build_run_metadata,
    get_num_model_layers,
    now_iso,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark, get_libero_path


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return value


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def collect_state_bank(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    state_bank_value = cfg.ASRE_DIAGNOSIS.get("state_bank_dir")
    if state_bank_value is None:
        raise ValueError("Set ASRE_DIAGNOSIS.state_bank_dir to a dedicated output directory.")
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(state_bank_value)))).resolve()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    metadata_path = output_dir / "run_metadata.json"
    existing_sample_ids: set[str] = set()
    if manifest_path.exists():
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Cannot resume state bank without metadata: {metadata_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing_sample_ids.add(str(json.loads(line)["sample_id"]))

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(
        cfg.model,
        model_dtype=model_dtype,
        device=model_device,
        text_encoder_device=cfg.EVALUATION.get("text_encoder_device"),
    )
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    _place_text_encoder(model, cfg.EVALUATION.get("text_encoder_device"))
    num_layers = get_num_model_layers(model)

    # Collection is always an unablated baseline rollout.
    cfg.ASRE_DIAGNOSIS.enabled = False
    cfg.ASRE_DIAGNOSIS.condition_name = "baseline"
    cfg.ASRE_DIAGNOSIS.condition_index = None
    cfg.ASRE_DIAGNOSIS.disabled_video_layers = []
    cfg.ASRE_DIAGNOSIS.save_rollout_video = False

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_value = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_value is None
        else int(action_horizon_value)
    )
    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])
    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    configured_task_ids = cfg.EVALUATION.get("task_ids", None)
    if configured_task_ids is None:
        task_ids = list(range(int(task_suite.n_tasks)))
    else:
        task_ids = [int(task_id) for task_id in configured_task_ids]
    max_states = int(cfg.ASRE_DIAGNOSIS.get("max_states_per_episode", 5))
    if max_states <= 0:
        raise ValueError("ASRE_DIAGNOSIS.max_states_per_episode must be positive.")

    run_start = now_iso()
    num_inference_steps_value = cfg.EVALUATION.get("num_inference_steps", None)
    num_inference_steps = (
        int(cfg.get("eval_num_inference_steps", 20))
        if num_inference_steps_value is None
        else int(num_inference_steps_value)
    )
    metadata = build_run_metadata(
        repo_root=project_root,
        checkpoint=str(cfg.ckpt),
        dataset_stats_path=str(dataset_stats_path),
        condition=DiagnosisCondition("baseline", ()),
        num_layers=num_layers,
        task_suite=suite_name,
        task_ids=task_ids,
        seed=None if cfg.get("seed") is None else int(cfg.seed),
        num_trials=int(cfg.EVALUATION.num_trials),
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        replan_steps=int(cfg.EVALUATION.get("replan_steps", 5)),
        start_timestamp=run_start,
    )
    metadata.update(
        {
            "artifact_type": "fixed_state_bank",
            "max_states_per_episode": max_states,
            "manifest": "manifest.jsonl",
            "compile_action_infer": bool(cfg.EVALUATION.get("compile_action_infer", False)),
            "binarize_gripper": bool(cfg.EVALUATION.get("binarize_gripper", False)),
            "sigma_shift": (
                None
                if cfg.EVALUATION.get("sigma_shift") is None
                else float(cfg.EVALUATION.get("sigma_shift"))
            ),
            "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        }
    )
    if manifest_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
        resume_keys = (
            "checkpoint_path",
            "dataset_stats_path",
            "num_model_layers",
            "task_suite",
            "task_ids",
            "seed",
            "number_of_trials",
            "action_horizon",
            "number_of_inference_steps",
            "replan_steps",
            "max_states_per_episode",
            "compile_action_infer",
            "binarize_gripper",
            "sigma_shift",
            "rand_device",
        )
        mismatches = {
            key: {"existing": existing_metadata.get(key), "requested": metadata.get(key)}
            for key in resume_keys
            if existing_metadata.get(key) != metadata.get(key)
        }
        if mismatches:
            raise ValueError(
                "Refusing to mix incompatible samples in an existing state bank: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        metadata["start_timestamp"] = existing_metadata.get(
            "start_timestamp", metadata["start_timestamp"]
        )
    atomic_write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    total_new_samples = 0

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states_path = (
            Path(get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file
        )
        initial_states = torch.load(initial_states_path, weights_only=False)
        num_trials = int(cfg.EVALUATION.num_trials)
        while len(initial_states) < num_trials:
            initial_states.extend(initial_states[: num_trials - len(initial_states)])
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))

        for episode_id in range(num_trials):
            env.reset()
            obs = env.set_init_state(initial_states[episode_id])
            pending_actions: list[list[float]] = []
            if use_action_ensembler:
                ensembler = ActionEnsembler()
                ensembler.reset()
            t = 0
            replan_id = -1
            done = False
            while t < _get_max_steps(suite_name) + num_steps_wait:
                if t < num_steps_wait:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    t += 1
                    continue

                if not pending_actions:
                    replan_id += 1
                    infer_kwargs, _ = _prepare_action_inference(
                        obs=obs,
                        task_description=task_description,
                        model=model,
                        processor=processor,
                        cfg=cfg,
                        action_horizon=action_horizon,
                        input_w=input_w,
                        input_h=input_h,
                        model_device=model_device,
                    )
                    baseline_raw_action, _ = _run_prepared_action_inference(
                        model=model,
                        cfg=cfg,
                        infer_kwargs=infer_kwargs,
                    )
                    action_chunk = _postprocess_action(baseline_raw_action, processor, cfg)

                    if replan_id < max_states:
                        sample_id = f"task{task_id:02d}_episode{episode_id:02d}_replan{replan_id:03d}"
                        relative_path = Path("samples") / f"{sample_id}.pt"
                        if sample_id not in existing_sample_ids:
                            payload = {
                                "schema_version": 1,
                                "sample_id": sample_id,
                                "task_suite": suite_name,
                                "task_id": int(task_id),
                                "task_description": task_description,
                                "episode_id": int(episode_id),
                                "replan_id": int(replan_id),
                                "environment_seed": None if cfg.get("seed") is None else int(cfg.seed),
                                "environment_step": int(t),
                                "action_inference_seed": infer_kwargs["seed"],
                                "infer_action_kwargs": _cpu_clone(infer_kwargs),
                                "baseline_raw_action": baseline_raw_action.detach().to(
                                    device="cpu", dtype=torch.float32
                                ),
                                "baseline_executed_action": torch.from_numpy(action_chunk.copy()),
                            }
                            torch.save(payload, output_dir / relative_path)
                            manifest_record = {
                                "sample_id": sample_id,
                                "sample_path": relative_path.as_posix(),
                                "task_suite": suite_name,
                                "task_id": int(task_id),
                                "task_description": task_description,
                                "episode_id": int(episode_id),
                                "replan_id": int(replan_id),
                                "environment_seed": payload["environment_seed"],
                                "environment_step": int(t),
                                "action_inference_seed": infer_kwargs["seed"],
                            }
                            _append_jsonl(manifest_path, manifest_record)
                            existing_sample_ids.add(sample_id)
                            total_new_samples += 1

                    if use_action_ensembler:
                        ensembler.add_actions(action_chunk, t)
                        pending_actions = [
                            ensembler.get_action(timestamp).tolist()
                            for timestamp in range(t, t + replan_steps)
                        ]
                    else:
                        pending_actions = action_chunk[:replan_steps].tolist()

                obs, _, done, _ = env.step(pending_actions.pop(0))
                if done:
                    break
                t += 1
            print(
                f"collected task={task_id} episode={episode_id} "
                f"states={min(replan_id + 1, max_states)} success={bool(done)}"
            )

        close_fn = getattr(env, "close", None)
        if close_fn is not None:
            close_fn()

    metadata["end_timestamp"] = now_iso()
    metadata["new_samples_written"] = total_new_samples
    metadata["total_manifest_samples"] = len(existing_sample_ids)
    atomic_write_json(metadata_path, metadata)
    print(f"State bank ready: {output_dir} ({len(existing_sample_ids)} samples)")


if __name__ == "__main__":
    collect_state_bank()
