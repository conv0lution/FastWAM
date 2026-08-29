"""Freeze the official LIBERO-Spatial world clips and four stochastic draws."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASKS = tuple(range(10))
SAMPLES_PER_TASK = 10
VIDEO_FRAME_OFFSETS = tuple(range(0, 33, 4))
ACTION_FRAME_OFFSETS = tuple(range(32))
DRAWS_PER_SAMPLE = 4
SELECTION_SEED = 4211
DRAW_SEED = 4212
# Only the two future latent frames are initialized from noise.  The real
# causal-VAE future target is scoring-only and never enters the predictor.
VIDEO_NOISE_SHAPE = (48, 2, 14, 28)
ACTION_NOISE_SHAPE = (32, 7)
VIDEO_INFERENCE_STEPS = 10
VIDEO_INFERENCE_SHIFT = 5.0
NATIVE_WORLD_METRIC = "pure_noise_native_future_latent_reconstruction_mse"
PROMPT_PREFIX = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: "
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _episodes(dataset_root: Path) -> pd.DataFrame:
    files = sorted((dataset_root / "meta/episodes").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {dataset_root}.")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    required = {
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "tasks",
        "length",
        "stats/task_index/min",
        "videos/observation.images.image/chunk_index",
        "videos/observation.images.image/file_index",
        "videos/observation.images.image/from_timestamp",
        "videos/observation.images.image/to_timestamp",
        "videos/observation.images.wrist_image/chunk_index",
        "videos/observation.images.wrist_image/file_index",
        "videos/observation.images.wrist_image/from_timestamp",
        "videos/observation.images.wrist_image/to_timestamp",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"LIBERO-Spatial episode metadata lacks {sorted(required-frame.columns)}.")
    return frame.sort_values("episode_index").reset_index(drop=True)


def _task_id(row: Mapping[str, Any]) -> int:
    value = row["stats/task_index/min"]
    if isinstance(value, (list, tuple)):
        value = value[0]
    elif hasattr(value, "tolist"):
        converted = value.tolist()
        value = converted[0] if isinstance(converted, list) else converted
    return int(value)


def _task_description(row: Mapping[str, Any]) -> str:
    values = row["tasks"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        raise ValueError(f"Expected exactly one task description, got {values!r}.")
    return str(values[0])


def _load_official_task_identities(
    preflight: Mapping[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Resolve dataset task descriptions to the frozen online suite IDs.

    LeRobot's internal ``task_index`` ordering is not the LIBERO suite ordering
    used by the frozen prompt cache, donor mapping, and Round-4C action results.
    The prompt cache is already a hash-frozen input and carries both identities,
    so it is the authoritative bridge between those namespaces.
    """

    path = Path(str(preflight["state"]["prompt_context_cache_path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen prompt-context cache: {path}")
    if _sha256_file(path) != str(
        preflight["state"]["prompt_context_cache_sha256"]
    ):
        raise ValueError("Frozen prompt-context cache drifted before task resolution.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prompts = payload.get("prompts") if isinstance(payload, Mapping) else None
    if not isinstance(prompts, Mapping):
        raise ValueError(f"Malformed frozen prompt-context cache: {path}")

    by_description: dict[str, int] = {}
    by_task_id: dict[int, str] = {}
    for prompt, raw_entry in prompts.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Frozen prompt-context cache contains a malformed entry.")
        task_id = int(raw_entry.get("task_id", -1))
        description = str(raw_entry.get("task_description", ""))
        if task_id not in TASKS or not description:
            raise ValueError(
                "Frozen prompt-context cache has an invalid task identity: "
                f"task_id={task_id}, description={description!r}."
            )
        if str(prompt) != f"{PROMPT_PREFIX}{description}":
            raise ValueError(
                "Frozen prompt-context cache prompt/description identity drifted."
            )
        if description in by_description and by_description[description] != task_id:
            raise ValueError(
                f"Task description maps to multiple suite IDs: {description!r}."
            )
        if task_id in by_task_id and by_task_id[task_id] != description:
            raise ValueError(f"Suite task {task_id} maps to multiple descriptions.")
        by_description[description] = task_id
        by_task_id[task_id] = description

    if set(by_task_id) != set(TASKS) or len(by_description) != len(TASKS):
        raise ValueError(
            "Frozen prompt-context cache must define exactly one identity for each "
            "LIBERO-Spatial suite task 0..9."
        )
    identities = [
        {"task_id": task_id, "task_description": by_task_id[task_id]}
        for task_id in TASKS
    ]
    return by_description, identities


def _select_records(
    *,
    dataset_root: Path,
    donor_mapping: Mapping[str, Any],
    donor_manifest: Mapping[str, Any],
    official_task_id_by_description: Mapping[str, int],
) -> list[dict[str, Any]]:
    episodes = _episodes(dataset_root)
    dataset_description_by_id: dict[int, str] = {}
    dataset_id_by_official_task: dict[int, int] = {}
    for _, row in episodes.iterrows():
        payload = row.to_dict()
        dataset_task_id = _task_id(payload)
        task_description = _task_description(payload)
        official_task_id = official_task_id_by_description.get(task_description)
        if official_task_id is None:
            raise ValueError(
                "Official LeRobot trajectory has a task description absent from "
                f"the frozen prompt cache: {task_description!r}."
            )
        if dataset_task_id not in TASKS or int(official_task_id) not in TASKS:
            raise ValueError("World task identity falls outside the registered 0..9 set.")
        if (
            dataset_task_id in dataset_description_by_id
            and dataset_description_by_id[dataset_task_id] != task_description
        ):
            raise ValueError(
                f"LeRobot task_index {dataset_task_id} maps to multiple descriptions."
            )
        if (
            int(official_task_id) in dataset_id_by_official_task
            and dataset_id_by_official_task[int(official_task_id)] != dataset_task_id
        ):
            raise ValueError(
                f"LIBERO suite task {official_task_id} maps to multiple LeRobot IDs."
            )
        dataset_description_by_id[dataset_task_id] = task_description
        dataset_id_by_official_task[int(official_task_id)] = dataset_task_id
    if (
        set(dataset_description_by_id) != set(TASKS)
        or set(dataset_id_by_official_task) != set(TASKS)
    ):
        raise ValueError(
            "LeRobot and frozen LIBERO suite task identities must form a total "
            "bijection over task IDs 0..9."
        )
    mappings = {
        (int(row["task_id"]), int(row["recipient_trial"])): row
        for row in donor_mapping["records"]
    }
    observations = {
        (int(row["task_id"]), int(row["source_trial"])): row
        for row in donor_manifest["records"]
    }
    records: list[dict[str, Any]] = []
    for task_id in TASKS:
        candidates = []
        for _, row in episodes.iterrows():
            payload = row.to_dict()
            task_description = _task_description(payload)
            resolved_task_id = official_task_id_by_description.get(task_description)
            if int(resolved_task_id) != task_id:
                continue
            dataset_task_id = _task_id(payload)
            length = int(payload["length"])
            if length < 33:
                continue
            episode_id = int(payload["episode_index"])
            # Preserve the originally frozen LeRobot sampling rule in its own
            # namespace.  Relabeling to suite IDs must not silently change the
            # selected source episodes or offsets.
            order = _stable_seed(
                "salvage-b-world-episode",
                SELECTION_SEED,
                dataset_task_id,
                episode_id,
            )
            candidates.append((order, payload))
        candidates.sort(key=lambda item: (item[0], int(item[1]["episode_index"])))
        if len(candidates) < SAMPLES_PER_TASK:
            raise ValueError(f"Task {task_id} has only {len(candidates)} valid 33-frame clips.")
        for trial, (_order, row) in enumerate(candidates[:SAMPLES_PER_TASK]):
            episode_id = int(row["episode_index"])
            length = int(row["length"])
            valid_start_count = length - 32
            offset = _stable_seed(
                "salvage-b-world-offset",
                SELECTION_SEED,
                _task_id(row),
                episode_id,
            ) % valid_start_count
            episode_from = int(row["dataset_from_index"])
            episode_to = int(row["dataset_to_index"])
            if episode_to - episode_from != length:
                raise ValueError(f"Episode length/index mismatch for episode {episode_id}.")
            dataset_index = episode_from + int(offset)
            task_description = _task_description(row)
            dataset_task_id = _task_id(row)
            mapping = mappings[(task_id, trial)]
            donor_trial = int(mapping["donor_trial"])
            donor = observations[(task_id, donor_trial)]
            if (
                int(donor.get("task_id", -1)) != task_id
                or str(donor.get("task_description")) != task_description
            ):
                raise ValueError(
                    "Frozen donor semantic identity does not match the selected world "
                    f"task {task_id}: {donor.get('task_description')!r} != "
                    f"{task_description!r}."
                )
            sample_id = (
                f"libero_spatial_task{task_id:02d}_trial{trial:02d}_"
                f"episode{episode_id:06d}_offset{int(offset):04d}"
            )
            data_chunk = int(row["data/chunk_index"])
            data_file = int(row["data/file_index"])
            camera_sources = {}
            for camera in (
                "observation.images.image",
                "observation.images.wrist_image",
            ):
                chunk = int(row[f"videos/{camera}/chunk_index"])
                file_index = int(row[f"videos/{camera}/file_index"])
                camera_sources[camera] = {
                    "relative_path": f"videos/{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4",
                    "from_timestamp": float(row[f"videos/{camera}/from_timestamp"]),
                    "to_timestamp": float(row[f"videos/{camera}/to_timestamp"]),
                }
            records.append(
                {
                    "sample_id": sample_id,
                    "task_id": task_id,
                    "dataset_task_id": dataset_task_id,
                    "trial": trial,
                    "task_description": task_description,
                    "episode_id": episode_id,
                    "episode_length": length,
                    "episode_dataset_from_index": episode_from,
                    "episode_dataset_to_index": episode_to,
                    "clip_start_offset": int(offset),
                    "dataset_index": dataset_index,
                    "source_data_relative_path": (
                        f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet"
                    ),
                    "source_video_artifacts": camera_sources,
                    "video_frame_offsets": list(VIDEO_FRAME_OFFSETS),
                    "video_dataset_indices": [dataset_index + x for x in VIDEO_FRAME_OFFSETS],
                    "action_frame_offsets": list(ACTION_FRAME_OFFSETS),
                    "action_dataset_indices": [dataset_index + x for x in ACTION_FRAME_OFFSETS],
                    "contains_padding": False,
                    "donor_task_id": task_id,
                    "donor_trial": donor_trial,
                    "donor_mapping_rule": donor_mapping["mapping_rule"],
                    "donor_processed_image_sha256": donor["processed_image_sha256"],
                    "donor_artifact_relative_path": donor["artifact_relative_path"],
                    "donor_artifact_sha256": donor["artifact_sha256"],
                }
            )
    if len(records) != len(TASKS) * SAMPLES_PER_TASK:
        raise AssertionError("World manifest must contain exactly 100 samples.")
    return records


def _draws(records: list[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tensors: dict[str, list[dict[str, Any]]] = {}
    manifest: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        sample_draws: list[dict[str, Any]] = []
        for draw_id in range(DRAWS_PER_SAMPLE):
            seeds = {
                name: _stable_seed("salvage-b-draw", DRAW_SEED, sample_id, draw_id, name)
                for name in (
                    "video_noise",
                    "action_noise",
                    "action_uniform",
                )
            }
            video_generator = torch.Generator(device="cpu").manual_seed(seeds["video_noise"])
            action_generator = torch.Generator(device="cpu").manual_seed(seeds["action_noise"])
            action_u_generator = torch.Generator(device="cpu").manual_seed(seeds["action_uniform"])
            video_noise = torch.randn(VIDEO_NOISE_SHAPE, generator=video_generator)
            action_noise = torch.randn(ACTION_NOISE_SHAPE, generator=action_generator)
            action_u = torch.rand((), generator=action_u_generator, dtype=torch.float32)
            # Action noise is frozen only for the one-draw shared-consumer
            # machinery audit; the video pathway cannot attend action tokens.
            action_sigma = action_u
            sample_draws.append(
                {
                    "draw_id": draw_id,
                    "video_noise": video_noise,
                    "action_noise": action_noise,
                    "action_timestep": action_sigma * 1000.0,
                }
            )
            manifest.append(
                {
                    "sample_id": sample_id,
                    "draw_id": draw_id,
                    "seeds": seeds,
                    "video_noise_sha256": _tensor_sha256(video_noise),
                    "action_uniform_u": float(action_u.item()),
                    "action_timestep": float((action_sigma * 1000.0).item()),
                    "action_noise_sha256": _tensor_sha256(action_noise),
                }
            )
        tensors[sample_id] = sample_draws
    return {
        "artifact_type": "asre_salvage_b_fixed_stochastic_tensors",
        "schema_version": 2,
        "draws_per_sample": DRAWS_PER_SAMPLE,
        "video_noise_shape": list(VIDEO_NOISE_SHAPE),
        "action_noise_shape": list(ACTION_NOISE_SHAPE),
        "dtype": "torch.float32",
        "native_world_metric": NATIVE_WORLD_METRIC,
        "schedulers": {
            "video": {
                "name": "WanContinuousFlowMatchScheduler",
                "num_train_timesteps": 1000,
                "inference_steps": VIDEO_INFERENCE_STEPS,
                "inference_shift": VIDEO_INFERENCE_SHIFT,
                "schedule_rule": "native build_inference_schedule from sigma=1 to 0",
                "initial_state": "pure Gaussian future-latent noise",
            },
            "action": {
                "name": "WanContinuousFlowMatchScheduler",
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "sigma_formula": "u",
            },
        },
        "samples": tensors,
    }, manifest


def build(args: argparse.Namespace) -> None:
    outputs = (
        args.world_manifest.resolve(),
        args.stochastic_manifest.resolve(),
        args.draw_tensors.resolve(),
    )
    parents = {path.parent for path in outputs}
    if len(parents) != 1:
        raise ValueError("Frozen world bundle artifacts must share one directory.")
    bundle_root = next(iter(parents))
    bundle_marker = bundle_root / "world_bundle_complete.json"
    if bundle_marker.exists():
        raise FileExistsError(
            "Refusing to overwrite a completed Salvage-B world bundle: "
            f"{bundle_marker}. Resume through the fail-closed driver or use a new output root."
        )
    preflight_path = args.preflight.resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "compatible":
        raise ValueError("Salvage-B preflight must pass before freezing world inputs.")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if preflight.get("git_commit_hash") != git_commit:
        raise ValueError("Preflight/source commit drifted before world manifest freeze.")
    dataset_root = args.dataset_root.expanduser().resolve()
    donor_mapping_path = args.donor_mapping.resolve()
    donor_manifest_path = args.donor_manifest.resolve()
    expected_paths = {
        "dataset_root": (
            dataset_root,
            Path(str(preflight["world_data"]["dataset_root"])).resolve(),
        ),
        "donor_mapping": (
            donor_mapping_path,
            Path(str(preflight["donors"]["mapping_path"])).resolve(),
        ),
        "donor_manifest": (
            donor_manifest_path,
            Path(str(preflight["donors"]["manifest_path"])).resolve(),
        ),
    }
    path_drift = {
        key: (str(observed), str(expected))
        for key, (observed, expected) in expected_paths.items()
        if observed != expected
    }
    if path_drift:
        raise ValueError(f"World-manifest CLI inputs differ from preflight: {path_drift}")
    required = [
        dataset_root / "meta/info.json",
        dataset_root / "meta/tasks.parquet",
        donor_mapping_path,
        donor_manifest_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen world inputs: {missing}")
    if (
        _sha256_file(donor_mapping_path) != preflight["donors"]["mapping_sha256"]
        or _sha256_file(donor_manifest_path) != preflight["donors"]["manifest_sha256"]
        or _sha256_file(dataset_root / "meta/info.json")
        != preflight["world_data"]["info_sha256"]
        or _sha256_file(dataset_root / "meta/tasks.parquet")
        != preflight["world_data"]["tasks_sha256"]
    ):
        raise ValueError("World-manifest inputs drifted from their preflight hashes.")
    donor_mapping = json.loads(donor_mapping_path.read_text(encoding="utf-8"))
    donor_manifest = json.loads(donor_manifest_path.read_text(encoding="utf-8"))
    official_task_id_by_description, official_task_identities = (
        _load_official_task_identities(preflight)
    )
    records = _select_records(
        dataset_root=dataset_root,
        donor_mapping=donor_mapping,
        donor_manifest=donor_manifest,
        official_task_id_by_description=official_task_id_by_description,
    )
    dataset_task_identities = [
        {
            "task_id": task_id,
            "dataset_task_id": int(
                next(record for record in records if record["task_id"] == task_id)[
                    "dataset_task_id"
                ]
            ),
            "task_description": next(
                record for record in records if record["task_id"] == task_id
            )["task_description"],
        }
        for task_id in TASKS
    ]
    basis_split_path = Path(str(preflight["basis"]["split_path"])).resolve()
    if (
        not basis_split_path.is_file()
        or _sha256_file(basis_split_path) != preflight["basis"]["split_sha256"]
    ):
        raise ValueError("Frozen basis-fit split drifted before world manifest freeze.")
    basis_split = json.loads(basis_split_path.read_text(encoding="utf-8"))
    basis_fit_ids = {str(value) for value in basis_split.get("fit_sample_ids", [])}
    world_ids = {str(record["sample_id"]) for record in records}
    source_identifier_overlap = sorted(basis_fit_ids & world_ids)
    if source_identifier_overlap:
        raise ValueError(
            "Primary world evaluation overlaps frozen basis-fit sample identifiers: "
            f"{source_identifier_overlap}"
        )
    draw_payload, draw_records = _draws(records)
    metadata_files = [
        dataset_root / "meta/info.json",
        dataset_root / "meta/tasks.parquet",
        *sorted((dataset_root / "meta/episodes").glob("*/*.parquet")),
    ]
    target_source_files = sorted(
        {
            dataset_root / str(record["source_data_relative_path"])
            for record in records
        }
        | {
            dataset_root / str(camera["relative_path"])
            for record in records
            for camera in record["source_video_artifacts"].values()
        }
    )
    missing_target_sources = [str(path) for path in target_source_files if not path.is_file()]
    if missing_target_sources:
        raise FileNotFoundError(
            f"Selected world clips reference missing source artifacts: {missing_target_sources}"
        )
    bundle_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".world_bundle.", dir=bundle_root)
    )
    staged_draw = temporary_root / args.draw_tensors.name
    staged_world = temporary_root / args.world_manifest.name
    staged_stochastic = temporary_root / args.stochastic_manifest.name
    try:
        _atomic_torch(staged_draw, draw_payload)
        draw_tensor_sha = _sha256_file(staged_draw)
        world_manifest = {
            "artifact_type": "asre_salvage_b_world_evaluation_manifest",
            "schema_version": 2,
            "status": "frozen_before_metrics",
            "git_commit_hash": git_commit,
            "preflight_report_path": str(preflight_path),
            "preflight_report_sha256": _sha256_file(preflight_path),
            "dataset_root": str(dataset_root),
            "dataset_name": "official_LIBERO-Spatial_LeRobot_v3_no_noops",
            "dataset_checkpoint_split_status": (
                "official dataset exposes no independent test split; this set is "
                "source-provenance-disjoint from the online basis-fit state bank"
            ),
            "basis_fit_namespace": "online_libero_spatial_state_bank",
            "world_evaluation_namespace": "official_lerobot_v30_trajectory",
            "source_identifier_overlap_with_basis_fit": source_identifier_overlap,
            "overlap_check_scope": (
                "source namespaces and exact artifacts; not a cross-dataset semantic "
                "episode-identity proof"
            ),
            "basis_fit_sample_identifier_count": len(basis_fit_ids),
            "basis_fit_source_manifest_path": preflight["state"]["source_manifest_path"],
            "basis_fit_source_manifest_sha256": preflight["state"][
                "source_manifest_sha256"
            ],
            "basis_fit_split_manifest_path": preflight["basis"]["split_path"],
            "basis_fit_split_manifest_sha256": preflight["basis"]["split_sha256"],
            "basis_fit_overlap_evidence": (
                "basis fitting used closed-loop online state-bank artifacts; primary "
                "world evaluation uses separately versioned official LeRobot-v3 "
                "trajectory artifacts, all frozen by exact path/hash before outcomes"
            ),
            "selection_seed": SELECTION_SEED,
            "selection_rule": (
                "per task hash-order valid episodes, take first 10; one hash-selected "
                "unpadded 33-action-frame/9-video-frame clip per episode; source "
                "episode and offset seeds use LeRobot dataset_task_id"
            ),
            "task_ids": list(TASKS),
            "task_identity_source": {
                "namespace": "frozen_LIBERO-Spatial_prompt_context_cache",
                "path": preflight["state"]["prompt_context_cache_path"],
                "sha256": preflight["state"]["prompt_context_cache_sha256"],
                "reason": (
                    "LeRobot internal task_index ordering differs from the online "
                    "LIBERO suite task_id ordering used by donors and action results"
                ),
                "identities": official_task_identities,
                "dataset_to_suite_identities": dataset_task_identities,
            },
            "samples_per_task": SAMPLES_PER_TASK,
            "sample_count": len(records),
            "records": records,
            "metadata_files": [
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "size": path.stat().st_size,
                }
                for path in metadata_files
            ],
            "target_source_files": [
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "size": path.stat().st_size,
                }
                for path in target_source_files
            ],
            "target_source_files_frozen_before_metrics": True,
            "donor_mapping_path": str(donor_mapping_path),
            "donor_mapping_sha256": _sha256_file(donor_mapping_path),
            "donor_manifest_path": str(donor_manifest_path),
            "donor_manifest_sha256": _sha256_file(donor_manifest_path),
            "donor_semantics": (
                "exact frozen Round-4C within-task trial derangement and donor "
                "observation pool; recipient text/proprio/target remain unchanged"
            ),
            "outcome_based_selection": False,
            "future_targets_available": True,
            "contains_padding": False,
            "native_world_metric": {
                "name": NATIVE_WORLD_METRIC,
                "direction": "lower_is_better",
                "initialization": "pure Gaussian noise in the two future latent frames",
                "conditioning": (
                    "frozen text/proprio plus the selected external first-frame K/V cache; "
                    "the carrier prefix is a discarded constant-zero placeholder"
                ),
                "inference_steps": VIDEO_INFERENCE_STEPS,
                "inference_shift": VIDEO_INFERENCE_SHIFT,
                "target": "causal-VAE latent frames 1:3 from the paired real 9-frame clip",
                "target_usage": "scoring_only_after_inference",
                "reduction": "mean squared error over batch/channel/time/height/width",
                "decoded_visual_metrics_primary": False,
            },
            "draw_tensor_path": str(args.draw_tensors.resolve()),
            "draw_tensor_sha256": draw_tensor_sha,
        }
        stochastic_manifest = {
            "artifact_type": "asre_salvage_b_stochastic_manifest",
            "schema_version": 2,
            "status": "frozen_before_metrics",
            "git_commit_hash": git_commit,
            "preflight_report_path": str(preflight_path),
            "preflight_report_sha256": _sha256_file(preflight_path),
            "draw_seed": DRAW_SEED,
            "native_world_metric": NATIVE_WORLD_METRIC,
            "video_inference_steps": VIDEO_INFERENCE_STEPS,
            "video_inference_shift": VIDEO_INFERENCE_SHIFT,
            "draws_per_sample": DRAWS_PER_SAMPLE,
            "draw_justification": (
                "four paired pure-noise starts reduce diffusion-trajectory Monte Carlo "
                "variance while preserving a fixed pre-outcome compute budget"
            ),
            "sample_count": len(records),
            "records": draw_records,
            "draw_tensor_path": str(args.draw_tensors.resolve()),
            "draw_tensor_sha256": draw_tensor_sha,
            "pairing_rule": (
                "same real target, pure-noise initialization, native inference "
                "schedule, text/proprio, and scheduler state across all conditions"
            ),
            "world_manifest_path": str(args.world_manifest.resolve()),
        }
        _atomic_json(staged_world, world_manifest)
        stochastic_manifest["world_manifest_sha256"] = _sha256_file(staged_world)
        _atomic_json(staged_stochastic, stochastic_manifest)
        for staged, destination in (
            (staged_draw, args.draw_tensors.resolve()),
            (staged_world, args.world_manifest.resolve()),
            (staged_stochastic, args.stochastic_manifest.resolve()),
        ):
            os.replace(staged, destination)
        _atomic_json(
            bundle_marker,
            {
                "artifact_type": "asre_salvage_b_world_bundle_completion",
                "schema_version": 2,
                "status": "complete",
                "git_commit_hash": git_commit,
                "preflight_report_sha256": _sha256_file(preflight_path),
                "artifacts": {
                    path.name: {"path": str(path), "sha256": _sha256_file(path)}
                    for path in outputs
                },
            },
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    print(f"Frozen Salvage-B world manifest: {args.world_manifest.resolve()}")
    print(f"Frozen Salvage-B draw manifest: {args.stochastic_manifest.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--world-manifest", type=Path, required=True)
    parser.add_argument("--stochastic-manifest", type=Path, required=True)
    parser.add_argument("--draw-tensors", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
