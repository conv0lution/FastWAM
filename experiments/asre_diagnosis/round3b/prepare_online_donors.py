"""Capture and freeze first-query donor observations for ASRE Round 3B."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (  # noqa: E402
    git_commit,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.round3b.donor import (  # noqa: E402
    DONOR_MAPPING_NAME,
    DONOR_OBSERVATION_MANIFEST_NAME,
    DONOR_SCHEMA_VERSION,
    OnlineDonorBundle,
    atomic_torch_save,
    build_donor_mapping_payload,
    canonical_sha256,
    donor_artifact_relative_path,
    task_text_sha256,
    tensor_is_finite,
    tensor_sha256,
    write_frozen_json,
)
from experiments.libero.eval_libero_single import _obs_to_model_input  # noqa: E402
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
)
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402


TASK_SUITE = "libero_spatial"
TASK_IDS = tuple(range(10))
NUM_TRIALS = 10
NUM_STEPS_WAIT = 30
SEED = 42
MODEL_IMAGE_DTYPE = torch.bfloat16


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the frozen same-task, next-trial donor observation bundle "
            "for ASRE Round 3B."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument(
        "--task-config",
        default="libero_uncond_2cam224_1e-4",
        help="Hydra task config used by the frozen Round-2/Round-3A runs.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-steps-wait", type=int, default=NUM_STEPS_WAIT)
    return parser.parse_args()


def _compose_config(task_config: str):
    config_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="sim_libero.yaml", overrides=[f"task={task_config}"])


def _load_initial_states(task: Any, num_trials: int) -> list[Any]:
    path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    payload = torch.load(path, weights_only=False)
    states = list(payload)
    if len(states) < num_trials:
        raise ValueError(
            f"Task {task.name!r} exposes only {len(states)} initial states; "
            f"Round 3B requires {num_trials}."
        )
    return states[:num_trials]


def _capture_first_query_image(
    *,
    env: Any,
    initial_state: Any,
    cfg: Any,
    processor: Any,
    input_w: int,
    input_h: int,
    num_steps_wait: int,
) -> torch.Tensor:
    env.reset()
    obs = env.set_init_state(initial_state)
    for step in range(num_steps_wait):
        obs, _reward, done, _info = env.step(get_libero_dummy_action())
        if done:
            raise RuntimeError(
                "Donor environment terminated during the initial dummy-action wait: "
                f"step={step + 1}/{num_steps_wait}."
            )
    image, _proprio, _raw_images = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device="cpu",
        dtype=MODEL_IMAGE_DTYPE,
    )
    image = image.detach().to(device="cpu").contiguous()
    if not tensor_is_finite(image):
        raise ValueError("Captured donor model-ready image contains NaN or Inf.")
    return image


def _validate_existing_bundle(
    *,
    output_root: Path,
    dataset_stats_path: Path,
    task_config: str,
    seed: int,
    num_steps_wait: int,
) -> tuple[Path, Path]:
    manifest_path = output_root / DONOR_OBSERVATION_MANIFEST_NAME
    mapping_path = output_root / DONOR_MAPPING_NAME
    bundle = OnlineDonorBundle.load(
        mapping_path=mapping_path,
        observation_manifest_path=manifest_path,
        observation_root=output_root,
    )
    expected = {
        "task_suite": TASK_SUITE,
        "seed": seed,
        "num_tasks": len(TASK_IDS),
        "num_trials": NUM_TRIALS,
        "num_steps_wait": num_steps_wait,
        "task_config": task_config,
        "dataset_stats_sha256": sha256_file(dataset_stats_path),
    }
    mismatches = {
        key: {
            "frozen": bundle.observation_payload.get(key),
            "requested": value,
        }
        for key, value in expected.items()
        if bundle.observation_payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Existing donor bundle is incompatible with this request: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    # Validate every frozen artifact's bytes, identity, image hash, shape and dtype.
    observed_shapes: set[tuple[int, ...]] = set()
    observed_dtypes: set[str] = set()
    for key, observation in bundle.observations.items():
        relative = Path(str(observation["artifact_relative_path"]))
        artifact = (output_root / relative).resolve()
        if not artifact.is_relative_to(output_root):
            raise ValueError(f"Frozen donor artifact escapes output root: {artifact}")
        if sha256_file(artifact) != observation["artifact_sha256"]:
            raise ValueError(f"Frozen donor artifact SHA256 mismatch: {artifact}")
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        image = payload.get("input_image") if isinstance(payload, dict) else None
        if not torch.is_tensor(image):
            raise ValueError(f"Malformed frozen donor artifact: {artifact}")
        if tensor_sha256(image) != observation["processed_image_sha256"]:
            raise ValueError(f"Frozen donor image SHA256 mismatch: {artifact}")
        if list(image.shape) != observation["processed_image_shape"]:
            raise ValueError(f"Frozen donor image shape mismatch: {artifact}")
        if str(image.dtype) != observation["processed_image_dtype"]:
            raise ValueError(f"Frozen donor image dtype mismatch: {artifact}")
        observed_shapes.add(tuple(int(dim) for dim in image.shape))
        observed_dtypes.add(str(image.dtype))
        if key != (int(payload["task_id"]), int(payload["source_trial"])):
            raise ValueError(f"Frozen donor artifact identity mismatch: {artifact}")
        if not tensor_is_finite(image):
            raise ValueError(f"Frozen donor image contains NaN or Inf: {artifact}")
    declared_shape = bundle.observation_payload.get("model_ready_image_shape")
    declared_dtype = bundle.observation_payload.get("model_ready_image_dtype")
    if observed_shapes != {tuple(int(dim) for dim in declared_shape or ())}:
        raise ValueError(
            "Frozen donor top-level image shape disagrees with its artifacts: "
            f"declared={declared_shape}, observed={sorted(observed_shapes)}."
        )
    if observed_dtypes != {str(declared_dtype)}:
        raise ValueError(
            "Frozen donor top-level image dtype disagrees with its artifacts: "
            f"declared={declared_dtype!r}, observed={sorted(observed_dtypes)}."
        )
    return manifest_path, mapping_path


def prepare_online_donors(
    *,
    output_root: Path,
    dataset_stats_path: Path,
    task_config: str,
    seed: int = SEED,
    num_steps_wait: int = NUM_STEPS_WAIT,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    dataset_stats_path = dataset_stats_path.expanduser().resolve()
    if not dataset_stats_path.is_file():
        raise FileNotFoundError(f"Dataset statistics are unavailable: {dataset_stats_path}")
    if seed != SEED:
        raise ValueError(f"Round 3B freezes the environment seed at {SEED}, got {seed}.")
    if num_steps_wait != NUM_STEPS_WAIT:
        raise ValueError(
            f"Round 3B freezes the initial wait at {NUM_STEPS_WAIT} dummy steps, "
            f"got {num_steps_wait}."
        )
    manifest_path = output_root / DONOR_OBSERVATION_MANIFEST_NAME
    mapping_path = output_root / DONOR_MAPPING_NAME
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"Donor output root is not a directory: {output_root}")
        if manifest_path.is_file() and mapping_path.is_file():
            manifest_path, mapping_path = _validate_existing_bundle(
                output_root=output_root,
                dataset_stats_path=dataset_stats_path,
                task_config=task_config,
                seed=seed,
                num_steps_wait=num_steps_wait,
            )
            return {
                "status": "already_frozen_and_valid",
                "donor_observation_root": str(output_root),
                "donor_observation_manifest_path": str(manifest_path),
                "donor_observation_manifest_sha256": sha256_file(manifest_path),
                "donor_mapping_path": str(mapping_path),
                "donor_mapping_sha256": sha256_file(mapping_path),
            }
        raise FileExistsError(
            "Refusing to merge donor artifacts into a pre-existing, unfrozen directory: "
            f"{output_root}. Move it aside and rerun."
        )

    cfg = _compose_config(task_config)
    if str(cfg.EVALUATION.task_suite_name) != TASK_SUITE:
        raise ValueError(
            f"Task config resolves to {cfg.EVALUATION.task_suite_name!r}; "
            f"Round 3B requires {TASK_SUITE!r}."
        )
    if int(cfg.EVALUATION.get("num_steps_wait", -1)) != num_steps_wait:
        raise ValueError(
            "Task config initial wait differs from the donor capture protocol: "
            f"{cfg.EVALUATION.get('num_steps_wait')} != {num_steps_wait}."
        )
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    video_size = list(cfg.data.train.video_size)
    if len(video_size) != 2:
        raise ValueError(f"Expected data.train.video_size=[H,W], got {video_size!r}.")
    input_h, input_w = int(video_size[0]), int(video_size[1])
    set_global_seed(seed, get_worker_init_fn=False)

    images_by_key: dict[tuple[int, int], torch.Tensor] = {}
    state_hashes: dict[tuple[int, int], str] = {}
    text_by_task: dict[int, str] = {}
    task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    for task_id in TASK_IDS:
        task = task_suite.get_task(task_id)
        initial_states = _load_initial_states(task, NUM_TRIALS)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        if str(task_description) != str(task.language):
            raise ValueError(f"Task text mismatch for LIBERO task {task_id}.")
        text_by_task[task_id] = str(task_description)
        try:
            for trial, initial_state in enumerate(initial_states):
                image = _capture_first_query_image(
                    env=env,
                    initial_state=initial_state,
                    cfg=cfg,
                    processor=processor,
                    input_w=input_w,
                    input_h=input_h,
                    num_steps_wait=num_steps_wait,
                )
                images_by_key[(task_id, trial)] = image
                state_hashes[(task_id, trial)] = canonical_sha256(initial_state)
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

    model_ready_shapes = {tuple(image.shape) for image in images_by_key.values()}
    model_ready_dtypes = {str(image.dtype) for image in images_by_key.values()}
    if len(model_ready_shapes) != 1 or len(model_ready_dtypes) != 1:
        raise ValueError(
            "Captured donor model inputs do not share one tensor structure: "
            f"shapes={sorted(model_ready_shapes)}, dtypes={sorted(model_ready_dtypes)}."
        )
    model_ready_shape = list(next(iter(model_ready_shapes)))
    model_ready_dtype = next(iter(model_ready_dtypes))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent)
    )
    staging_root = temporary_parent / "bundle"
    staging_root.mkdir()
    try:
        records: list[dict[str, Any]] = []
        for task_id, trial in sorted(images_by_key):
            image = images_by_key[(task_id, trial)]
            relative_path = donor_artifact_relative_path(task_id, trial)
            staging_artifact = staging_root / relative_path
            final_artifact = output_root / relative_path
            image_hash = tensor_sha256(image)
            payload = {
                "schema_version": DONOR_SCHEMA_VERSION,
                "task_suite": TASK_SUITE,
                "task_id": task_id,
                "source_trial": trial,
                "first_policy_query_environment_step": num_steps_wait,
                "task_description": text_by_task[task_id],
                "task_text_sha256": task_text_sha256(text_by_task[task_id]),
                "initial_state_sha256": state_hashes[(task_id, trial)],
                "processed_image_sha256": image_hash,
                "input_image": image,
            }
            atomic_torch_save(staging_artifact, payload)
            records.append(
                {
                    "task_id": task_id,
                    "source_task_id": task_id,
                    "source_trial": trial,
                    "first_policy_query_environment_step": num_steps_wait,
                    "task_description": text_by_task[task_id],
                    "task_text_sha256": task_text_sha256(text_by_task[task_id]),
                    "initial_state_sha256": state_hashes[(task_id, trial)],
                    "processed_image_sha256": image_hash,
                    "processed_image_shape": list(image.shape),
                    "processed_image_dtype": str(image.dtype),
                    "processed_image_finite": tensor_is_finite(image),
                    "processed_image_min": float(image.float().min().item()),
                    "processed_image_max": float(image.float().max().item()),
                    "artifact_relative_path": str(relative_path),
                    "artifact_absolute_path": str(final_artifact),
                    "artifact_sha256": sha256_file(staging_artifact),
                }
            )
        manifest_payload = {
            "schema_version": DONOR_SCHEMA_VERSION,
            "created_at": now_iso(),
            "git_commit_hash": git_commit(project_root),
            "task_suite": TASK_SUITE,
            "task_ids": list(TASK_IDS),
            "seed": seed,
            "num_tasks": len(TASK_IDS),
            "num_trials": NUM_TRIALS,
            "num_steps_wait": num_steps_wait,
            "first_policy_query_environment_step": num_steps_wait,
            "task_config": task_config,
            "task_config_sha256": sha256_json(
                {
                    "task_config": task_config,
                    "data_train": OmegaConf.to_container(
                        cfg.data.train, resolve=True
                    ),
                    "num_steps_wait": int(cfg.EVALUATION.num_steps_wait),
                }
            ),
            "dataset_stats_path": str(dataset_stats_path),
            "dataset_stats_sha256": sha256_file(dataset_stats_path),
            "model_ready_image_dtype": model_ready_dtype,
            # ``_obs_to_model_input`` performs the configured multi-camera
            # concatenation, so the frozen tensor width is not necessarily the
            # per-camera ``video_size`` width (two LIBERO cameras yield 448).
            "model_ready_image_shape": model_ready_shape,
            "camera_concatenation": str(
                cfg.data.train.get("concat_multi_camera", "horizontal")
            ),
            "records": records,
        }
        staging_manifest = staging_root / DONOR_OBSERVATION_MANIFEST_NAME
        write_frozen_json(staging_manifest, manifest_payload)
        manifest_digest = sha256_file(staging_manifest)
        mapping_payload = build_donor_mapping_payload(
            manifest_payload,
            observation_manifest_path=output_root / DONOR_OBSERVATION_MANIFEST_NAME,
            observation_manifest_sha256=manifest_digest,
            images_by_key=images_by_key,
        )
        staging_mapping = staging_root / DONOR_MAPPING_NAME
        write_frozen_json(staging_mapping, mapping_payload)
        os.replace(staging_root, output_root)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)

    return {
        "status": "created_and_frozen",
        "donor_observation_root": str(output_root),
        "donor_observation_manifest_path": str(manifest_path),
        "donor_observation_manifest_sha256": sha256_file(manifest_path),
        "donor_mapping_path": str(mapping_path),
        "donor_mapping_sha256": sha256_file(mapping_path),
        "record_count": len(images_by_key),
    }


def main() -> None:
    args = _parse_args()
    report = prepare_online_donors(
        output_root=args.output_root,
        dataset_stats_path=args.dataset_stats,
        task_config=str(args.task_config),
        seed=int(args.seed),
        num_steps_wait=int(args.num_steps_wait),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
