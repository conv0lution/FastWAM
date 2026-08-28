"""Capture one frozen first-policy-query donor bundle for each G0 suite."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import git_commit, now_iso, sha256_file, sha256_json
from experiments.asre_diagnosis.g0.definitions import (
    NUM_STEPS_WAIT,
    NUM_TRIALS,
    SEED,
    SUITE_ORDER,
    TASK_CONFIG,
    TASK_IDS,
    assert_output_scope,
)
from experiments.asre_diagnosis.round3b.donor import (
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
from experiments.asre_diagnosis.round3b.prepare_online_donors import (
    MODEL_IMAGE_DTYPE,
    _capture_first_query_image,
    _load_initial_states,
)
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark


def _compose_config(task_config: str, suite: str):
    with initialize_config_dir(
        config_dir=str((project_root / "configs").resolve()), version_base="1.3"
    ):
        cfg = compose(config_name="sim_libero.yaml", overrides=[f"task={task_config}"])
    cfg.EVALUATION.task_suite_name = suite
    return cfg


def _validate_existing(
    *,
    root: Path,
    suite: str,
    dataset_stats: Path,
    task_config: str,
) -> dict[str, Any]:
    manifest_path = root / DONOR_OBSERVATION_MANIFEST_NAME
    mapping_path = root / DONOR_MAPPING_NAME
    bundle = OnlineDonorBundle.load(
        mapping_path=mapping_path,
        observation_manifest_path=manifest_path,
        observation_root=root,
    )
    expected = {
        "task_suite": suite,
        "task_ids": list(TASK_IDS),
        "seed": SEED,
        "num_tasks": len(TASK_IDS),
        "num_trials": NUM_TRIALS,
        "num_steps_wait": NUM_STEPS_WAIT,
        "task_config": task_config,
        "dataset_stats_sha256": sha256_file(dataset_stats),
    }
    mismatches = {
        key: {"observed": bundle.observation_payload.get(key), "expected": value}
        for key, value in expected.items()
        if bundle.observation_payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Existing G0 donor bundle is incompatible: {mismatches}.")
    for observation in bundle.observations.values():
        artifact = (root / str(observation["artifact_relative_path"])).resolve()
        if not artifact.is_relative_to(root):
            raise ValueError(f"Donor artifact escapes bundle root: {artifact}.")
        if sha256_file(artifact) != observation["artifact_sha256"]:
            raise ValueError(f"Donor artifact SHA256 mismatch: {artifact}.")
    return {
        "status": "already_frozen_and_valid",
        "suite": suite,
        "donor_observation_root": str(root),
        "donor_observation_manifest_path": str(manifest_path),
        "donor_observation_manifest_sha256": sha256_file(manifest_path),
        "donor_mapping_path": str(mapping_path),
        "donor_mapping_sha256": sha256_file(mapping_path),
        "record_count": len(bundle.observations),
    }


def prepare_suite_donors(
    *,
    suite: str,
    output_root: Path,
    dataset_stats: Path,
    task_config: str = TASK_CONFIG,
) -> dict[str, Any]:
    if suite not in SUITE_ORDER:
        raise ValueError(f"Unsupported G0 suite {suite!r}; expected {list(SUITE_ORDER)}.")
    root = output_root.expanduser().resolve()
    assert_output_scope(root, project_root)
    stats = dataset_stats.expanduser().resolve()
    if not stats.is_file():
        raise FileNotFoundError(f"Dataset statistics are unavailable: {stats}.")
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"Donor output root is not a directory: {root}.")
        if (root / DONOR_MAPPING_NAME).is_file() and (
            root / DONOR_OBSERVATION_MANIFEST_NAME
        ).is_file():
            return _validate_existing(
                root=root,
                suite=suite,
                dataset_stats=stats,
                task_config=task_config,
            )
        raise FileExistsError(
            f"Refusing to merge into pre-existing unfrozen donor directory: {root}."
        )

    cfg = _compose_config(task_config, suite)
    if int(cfg.EVALUATION.num_steps_wait) != NUM_STEPS_WAIT:
        raise ValueError("Configured initial dummy-step wait differs from G0 protocol.")
    dataset_statistics = load_dataset_stats_from_json(str(stats))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_statistics)
    input_h, input_w = (int(value) for value in cfg.data.train.video_size)
    set_global_seed(SEED, get_worker_init_fn=False)

    images: dict[tuple[int, int], torch.Tensor] = {}
    state_hashes: dict[tuple[int, int], str] = {}
    text_by_task: dict[int, str] = {}
    suite_object = benchmark.get_benchmark_dict()[suite]()
    if int(suite_object.n_tasks) != len(TASK_IDS):
        raise ValueError(f"{suite} does not expose exactly 10 tasks.")
    for task_id in TASK_IDS:
        task = suite_object.get_task(task_id)
        initial_states = _load_initial_states(task, NUM_TRIALS)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)
        if str(task_description) != str(task.language):
            raise ValueError(f"Task language drift for {suite} task {task_id}.")
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
                    num_steps_wait=NUM_STEPS_WAIT,
                )
                images[(task_id, trial)] = image
                state_hashes[(task_id, trial)] = canonical_sha256(initial_state)
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

    shapes = {tuple(image.shape) for image in images.values()}
    dtypes = {str(image.dtype) for image in images.values()}
    if len(shapes) != 1 or dtypes != {str(MODEL_IMAGE_DTYPE)}:
        raise ValueError(f"G0 donor tensor structure drift: shapes={shapes}, dtypes={dtypes}.")

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging.", dir=root.parent)
    )
    staging = temporary_parent / "bundle"
    staging.mkdir()
    try:
        records: list[dict[str, Any]] = []
        for task_id, trial in sorted(images):
            image = images[(task_id, trial)]
            relative = donor_artifact_relative_path(task_id, trial)
            staging_artifact = staging / relative
            final_artifact = root / relative
            image_digest = tensor_sha256(image)
            artifact_payload = {
                "schema_version": DONOR_SCHEMA_VERSION,
                "task_suite": suite,
                "task_id": task_id,
                "source_trial": trial,
                "first_policy_query_environment_step": NUM_STEPS_WAIT,
                "task_description": text_by_task[task_id],
                "task_text_sha256": task_text_sha256(text_by_task[task_id]),
                "initial_state_sha256": state_hashes[(task_id, trial)],
                "processed_image_sha256": image_digest,
                "input_image": image,
            }
            atomic_torch_save(staging_artifact, artifact_payload)
            records.append(
                {
                    "task_id": task_id,
                    "source_task_id": task_id,
                    "source_trial": trial,
                    "first_policy_query_environment_step": NUM_STEPS_WAIT,
                    "task_description": text_by_task[task_id],
                    "task_text_sha256": task_text_sha256(text_by_task[task_id]),
                    "initial_state_sha256": state_hashes[(task_id, trial)],
                    "processed_image_sha256": image_digest,
                    "processed_image_shape": list(image.shape),
                    "processed_image_dtype": str(image.dtype),
                    "processed_image_finite": tensor_is_finite(image),
                    "processed_image_min": float(image.float().min().item()),
                    "processed_image_max": float(image.float().max().item()),
                    "artifact_relative_path": str(relative),
                    "artifact_absolute_path": str(final_artifact),
                    "artifact_sha256": sha256_file(staging_artifact),
                }
            )
        manifest = {
            "schema_version": DONOR_SCHEMA_VERSION,
            "created_at": now_iso(),
            "git_commit_hash": git_commit(project_root),
            "task_suite": suite,
            "task_ids": list(TASK_IDS),
            "seed": SEED,
            "num_tasks": len(TASK_IDS),
            "num_trials": NUM_TRIALS,
            "num_steps_wait": NUM_STEPS_WAIT,
            "first_policy_query_environment_step": NUM_STEPS_WAIT,
            "task_config": task_config,
            "task_config_sha256": sha256_json(
                {
                    "task_config": task_config,
                    "data_train": OmegaConf.to_container(cfg.data.train, resolve=True),
                    "num_steps_wait": NUM_STEPS_WAIT,
                    "task_suite": suite,
                }
            ),
            "dataset_stats_path": str(stats),
            "dataset_stats_sha256": sha256_file(stats),
            "model_ready_image_dtype": next(iter(dtypes)),
            "model_ready_image_shape": list(next(iter(shapes))),
            "camera_concatenation": str(
                cfg.data.train.get("concat_multi_camera", "horizontal")
            ),
            "records": records,
        }
        staging_manifest = staging / DONOR_OBSERVATION_MANIFEST_NAME
        write_frozen_json(staging_manifest, manifest)
        mapping = build_donor_mapping_payload(
            manifest,
            observation_manifest_path=root / DONOR_OBSERVATION_MANIFEST_NAME,
            observation_manifest_sha256=sha256_file(staging_manifest),
            images_by_key=images,
        )
        write_frozen_json(staging / DONOR_MAPPING_NAME, mapping)
        os.replace(staging, root)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)

    return _validate_existing(
        root=root, suite=suite, dataset_stats=stats, task_config=task_config
    ) | {"status": "created_and_frozen"}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITE_ORDER, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--task-config", default=TASK_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = prepare_suite_donors(
        suite=args.suite,
        output_root=args.output_root,
        dataset_stats=args.dataset_stats,
        task_config=args.task_config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
