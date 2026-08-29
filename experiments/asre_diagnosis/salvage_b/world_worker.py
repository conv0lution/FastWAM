"""Evaluate one isolated Salvage-B native-world-loss sample shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_B_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.salvage_b.definitions import (  # noqa: E402
    ENDPOINT_CONDITIONS,
    PROJECTED_CONDITIONS,
    RANK_BY_CONDITION,
    WORLD_DRAWS_PER_SAMPLE,
)
from experiments.asre_diagnosis.salvage_b.world_runtime import (  # noqa: E402
    EARLY_DISABLED,
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
    NATIVE_WORLD_INFERENCE_SHIFT,
    NATIVE_WORLD_INFERENCE_STEPS,
    PREFIX_TOKENS,
    condition_cache,
    load_donor_bundle,
    load_frozen_model,
    load_processed_sample,
    load_prompt_cache,
    load_rank_specs,
    load_world_dataset,
    native_world_loss,
    prepare_world_sample,
    read_json,
    require_clean_source,
    validate_architecture_source_inventory,
)
from experiments.asre_diagnosis.salvage_b.world_manifest import (  # noqa: E402
    ACTION_NOISE_SHAPE,
    NATIVE_WORLD_METRIC,
    VIDEO_NOISE_SHAPE,
)
from experiments.asre_diagnosis.round3b.donor import tensor_sha256  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


PHASES = ("endpoint", "projected")
WORKERS = 4
SAMPLES = 100
SAMPLES_PER_WORKER = 25
ROWS_PER_WORKER = 200


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def phase_conditions(phase: str) -> tuple[str, str]:
    if phase == "endpoint":
        return ENDPOINT_CONDITIONS
    if phase == "projected":
        return PROJECTED_CONDITIONS
    raise ValueError(f"Unsupported Salvage-B world phase: {phase!r}.")


def worker_records(records: Sequence[Mapping[str, Any]], worker_index: int) -> list[dict[str, Any]]:
    if worker_index not in range(WORKERS):
        raise ValueError("worker-index must be 0..3.")
    if len(records) != SAMPLES:
        raise ValueError(f"Salvage-B world manifest must contain {SAMPLES} records.")
    result = [dict(record) for record in records[worker_index::WORKERS]]
    if len(result) != SAMPLES_PER_WORKER:
        raise ValueError("World records do not form four exact 25-sample shards.")
    return result


def _frozen_tensor_sha256(tensor: torch.Tensor) -> str:
    """Match the pre-metric tensor hash frozen by ``world_manifest.py``."""

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _validate_unique_world_records(records: Sequence[Mapping[str, Any]]) -> None:
    sample_ids = [str(record.get("sample_id")) for record in records]
    if len(sample_ids) != SAMPLES or len(set(sample_ids)) != SAMPLES:
        raise ValueError("World sample IDs must contain exactly 100 unique values.")
    task_trials = [
        (int(record.get("task_id", -1)), int(record.get("trial", -1)))
        for record in records
    ]
    if set(task_trials) != {(task, trial) for task in range(10) for trial in range(10)}:
        raise ValueError("World manifest must contain every task/trial pair exactly once.")
    if any(
        bool(record.get("contains_padding", True))
        or int(record.get("episode_id", -1)) < 0
        or int(record.get("dataset_index", -1)) < 0
        for record in records
    ):
        raise ValueError("World records contain padding or malformed source indices.")


def _validate_target_records(
    *, world_records: Sequence[Mapping[str, Any]], target: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    target_records = target.get("records")
    if not isinstance(target_records, list) or len(target_records) != SAMPLES:
        raise ValueError("Processed target manifest must contain exactly 100 records.")
    by_id: dict[str, dict[str, Any]] = {}
    world_by_id = {str(record["sample_id"]): record for record in world_records}
    for raw in target_records:
        record = dict(raw)
        sample_id = str(record.get("sample_id"))
        if sample_id in by_id or sample_id not in world_by_id:
            raise ValueError("Processed target IDs are duplicate or absent from world manifest.")
        world = world_by_id[sample_id]
        if any(
            int(record.get(key, -1)) != int(world[key])
            for key in ("task_id", "episode_id", "trial", "dataset_index")
        ):
            raise ValueError(f"Processed target identity drifted for {sample_id}.")
        hashes = record.get("processed_tensor_sha256")
        if not isinstance(hashes, Mapping) or set(hashes) != {
            "video",
            "action",
            "proprio",
            "image_is_pad",
            "action_is_pad",
            "prompt_sha256",
        }:
            raise ValueError(f"Processed target hashes are malformed for {sample_id}.")
        if any(not _is_sha256(value) for value in hashes.values()):
            raise ValueError(f"Processed target hash width drifted for {sample_id}.")
        if not _is_sha256(record.get("current_image_sha256")):
            raise ValueError(f"Processed current-image hash is malformed for {sample_id}.")
        if (
            record.get("video_shape") != [3, 9, 224, 448]
            or record.get("action_shape") != [32, 7]
            or record.get("proprio_shape") != [32, 8]
            or bool(record.get("contains_padding", True))
        ):
            raise ValueError(f"Processed target tensor layout drifted for {sample_id}.")
        by_id[sample_id] = record
    if set(by_id) != set(world_by_id):
        raise ValueError("Processed targets do not pair one-to-one with world samples.")
    return by_id


def _validate_stochastic_records(
    *, sample_ids: set[str], stochastic: Mapping[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    records = stochastic.get("records")
    if not isinstance(records, list) or len(records) != SAMPLES * WORLD_DRAWS_PER_SAMPLE:
        raise ValueError("Stochastic manifest must contain exactly 400 sample/draw records.")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        sample_id = str(record.get("sample_id"))
        draw_id = int(record.get("draw_id", -1))
        key = (sample_id, draw_id)
        if (
            sample_id not in sample_ids
            or draw_id not in range(WORLD_DRAWS_PER_SAMPLE)
            or key in result
        ):
            raise ValueError("Stochastic sample/draw pairing is malformed.")
        for name in ("video_noise_sha256", "action_noise_sha256"):
            value = record.get(name)
            if not _is_sha256(value):
                raise ValueError(f"Malformed frozen stochastic hash: {key}/{name}.")
        action_timestep = float(record.get("action_timestep", math.nan))
        if not math.isfinite(action_timestep) or not 0.0 <= action_timestep <= 1000.0:
            raise ValueError(f"Malformed frozen action timestep: {key}.")
        if "video_timestep" in record:
            raise ValueError(
                f"Native-inference draw unexpectedly freezes a video timestep: {key}."
            )
        result[key] = record
    expected = {
        (sample_id, draw_id)
        for sample_id in sample_ids
        for draw_id in range(WORLD_DRAWS_PER_SAMPLE)
    }
    if set(result) != expected:
        raise ValueError("Stochastic manifest does not cover every sample/draw pair.")
    return result


def validate_endpoint_gate_payload(
    payload: Mapping[str, Any],
    *,
    world_manifest_sha256: str,
    stochastic_manifest_sha256: str,
    target_manifest_sha256: str,
    machinery_sha256: str,
    git_commit_hash: str,
) -> None:
    """Fail closed on the frozen endpoint authorization consumed by Phase B."""

    expected_provenance = {
        "world_manifest_sha256": world_manifest_sha256,
        "stochastic_manifest_sha256": stochastic_manifest_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "machinery_sha256": machinery_sha256,
        "git_commit_hash": git_commit_hash,
    }
    if (
        payload.get("artifact_type") != "asre_salvage_b_world_endpoint_gate"
        or int(payload.get("schema_version", -1)) != 2
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
        or payload.get("classification") is not None
        or int(payload.get("sample_count", -1)) != SAMPLES
        or payload.get("loss_direction") != "lower_is_better"
        or payload.get("native_world_metric") != NATIVE_WORLD_METRIC
        or payload.get("inference_steps") != NATIVE_WORLD_INFERENCE_STEPS
        or payload.get("inference_shift") != NATIVE_WORLD_INFERENCE_SHIFT
        or any(payload.get(key) != value for key, value in expected_provenance.items())
    ):
        raise ValueError("Projected endpoint gate is failed or belongs to another run.")

    shards = payload.get("endpoint_shards")
    if not isinstance(shards, list) or len(shards) != WORKERS:
        raise ValueError("Projected endpoint gate lacks exactly four frozen shard hashes.")
    by_worker: dict[int, Mapping[str, Any]] = {}
    for raw in shards:
        if not isinstance(raw, Mapping):
            raise ValueError("Projected endpoint gate contains a malformed shard record.")
        worker_index = int(raw.get("worker_index", -1))
        if worker_index not in range(WORKERS) or worker_index in by_worker:
            raise ValueError("Projected endpoint gate shard indices are malformed.")
        if int(raw.get("record_count", -1)) != ROWS_PER_WORKER:
            raise ValueError("Projected endpoint gate shard record counts drifted.")
        for name in ("metadata_sha256", "rows_sha256"):
            value = raw.get(name)
            if not _is_sha256(value):
                raise ValueError(f"Projected endpoint gate lacks a valid {name}.")
        for path_name, sha_name in (
            ("metadata_path", "metadata_sha256"),
            ("rows_path", "rows_sha256"),
        ):
            path = Path(str(raw.get(path_name, ""))).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != raw[sha_name]:
                raise ValueError(
                    f"Projected endpoint gate referenced {path_name} changed: {path}."
                )
        by_worker[worker_index] = raw
    if set(by_worker) != set(range(WORKERS)):
        raise ValueError("Projected endpoint gate does not cover workers 0..3.")
    for path_name, sha_name in (
        ("draw_rows_path", "draw_rows_sha256"),
        ("sample_rows_path", "sample_rows_sha256"),
        ("sample_identity_rows_path", "sample_identity_rows_sha256"),
    ):
        value = payload.get(sha_name)
        path = Path(str(payload.get(path_name, ""))).expanduser().resolve()
        if not _is_sha256(value) or not path.is_file() or sha256_file(path) != value:
            raise ValueError(
                f"Projected endpoint gate lacks an unchanged aggregate {path_name}."
            )

    identities = payload.get("sample_identity")
    if not isinstance(identities, list) or len(identities) != SAMPLES:
        raise ValueError("Projected endpoint gate lacks exactly 100 sample identities.")
    identity_sha = payload.get("sample_identity_sha256")
    if not _is_sha256(identity_sha) or identity_sha != sha256_json(identities):
        raise ValueError("Projected endpoint sample-identity digest is malformed.")
    by_sample: dict[str, Mapping[str, Any]] = {}
    for raw in identities:
        if not isinstance(raw, Mapping):
            raise ValueError("Projected endpoint sample identity is malformed.")
        sample_id = str(raw.get("sample_id", ""))
        if not sample_id or sample_id in by_sample:
            raise ValueError("Projected endpoint sample identities are duplicate.")
        if any(
            not _is_sha256(raw.get(name))
            for name in (
                "target_sha256",
                "target_latent_sha256",
                "current_image_sha256",
                "current_frame_latent_sha256",
                "donor_image_sha256",
            )
        ):
            raise ValueError(f"Projected endpoint identity hashes drifted for {sample_id}.")
        if any(
            int(raw.get(name, -1)) < 0
            for name in ("task_id", "episode_id", "trial_index")
        ):
            raise ValueError(f"Projected endpoint identity indices drifted for {sample_id}.")
        by_sample[sample_id] = raw


def _endpoint_identity_by_sample(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    identities = payload.get("sample_identity")
    if not isinstance(identities, list):
        raise ValueError("Endpoint gate sample identities are unavailable.")
    return {str(row["sample_id"]): dict(row) for row in identities}


def _validate_draw_payload(
    *,
    draw_payload: Mapping[str, Any],
    stochastic_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if (
        draw_payload.get("artifact_type")
        != "asre_salvage_b_fixed_stochastic_tensors"
        or int(draw_payload.get("schema_version", -1)) != 2
        or int(draw_payload.get("draws_per_sample", -1)) != WORLD_DRAWS_PER_SAMPLE
        or draw_payload.get("video_noise_shape") != list(VIDEO_NOISE_SHAPE)
        or draw_payload.get("action_noise_shape") != list(ACTION_NOISE_SHAPE)
        or draw_payload.get("dtype") != "torch.float32"
        or draw_payload.get("native_world_metric") != NATIVE_WORLD_METRIC
        or draw_payload.get("schedulers", {}).get("video", {}).get(
            "num_train_timesteps"
        )
        != 1000
        or draw_payload.get("schedulers", {}).get("video", {}).get("inference_steps")
        != NATIVE_WORLD_INFERENCE_STEPS
        or draw_payload.get("schedulers", {}).get("video", {}).get("inference_shift")
        != NATIVE_WORLD_INFERENCE_SHIFT
        or draw_payload.get("schedulers", {}).get("action", {}).get("shift") != 1.0
    ):
        raise ValueError("Frozen stochastic tensor header drifted.")
    samples = draw_payload.get("samples")
    if not isinstance(samples, dict) or set(samples) != {
        key[0] for key in stochastic_by_key
    }:
        raise ValueError("Frozen draw tensor sample IDs drifted.")
    result: dict[str, list[dict[str, Any]]] = {}
    for sample_id, raw_draws in samples.items():
        if not isinstance(raw_draws, list) or len(raw_draws) != WORLD_DRAWS_PER_SAMPLE:
            raise ValueError(f"Frozen draw count drifted for {sample_id}.")
        by_draw: dict[int, dict[str, Any]] = {}
        for raw in raw_draws:
            draw = dict(raw)
            draw_id = int(draw.get("draw_id", -1))
            key = (str(sample_id), draw_id)
            if draw_id not in range(WORLD_DRAWS_PER_SAMPLE) or draw_id in by_draw:
                raise ValueError(f"Frozen draw IDs drifted for {sample_id}.")
            video_noise = draw.get("video_noise")
            action_noise = draw.get("action_noise")
            action_timestep = draw.get("action_timestep")
            if (
                "video_timestep" in draw
                or
                not torch.is_tensor(video_noise)
                or tuple(video_noise.shape) != VIDEO_NOISE_SHAPE
                or video_noise.dtype != torch.float32
                or not bool(torch.isfinite(video_noise).all())
                or not torch.is_tensor(action_noise)
                or tuple(action_noise.shape) != ACTION_NOISE_SHAPE
                or action_noise.dtype != torch.float32
                or not bool(torch.isfinite(action_noise).all())
                or not torch.is_tensor(action_timestep)
                or action_timestep.numel() != 1
                or not bool(torch.isfinite(action_timestep).all())
            ):
                raise ValueError(f"Frozen draw tensor layout drifted for {key}.")
            manifest = stochastic_by_key[key]
            if (
                _frozen_tensor_sha256(video_noise) != manifest["video_noise_sha256"]
                or _frozen_tensor_sha256(action_noise) != manifest["action_noise_sha256"]
                or float(action_timestep.item()) != float(manifest["action_timestep"])
            ):
                raise ValueError(f"Frozen draw tensors disagree with manifest for {key}.")
            by_draw[draw_id] = draw
        result[str(sample_id)] = [by_draw[index] for index in range(WORLD_DRAWS_PER_SAMPLE)]
    return result


def load_frozen_contract(
    args: argparse.Namespace, *, load_draw_tensors: bool
) -> dict[str, Any]:
    paths = {
        "preflight": _require_file(args.preflight, "preflight report"),
        "world": _require_file(args.world_manifest, "world manifest"),
        "stochastic": _require_file(args.stochastic_manifest, "stochastic manifest"),
        "draws": _require_file(args.draw_tensors, "draw tensor artifact"),
        "targets": _require_file(args.target_manifest, "processed target manifest"),
        "machinery": _require_file(args.machinery, "machinery report"),
        "launcher": _require_file(args.launcher_config, "launcher configuration"),
    }
    payloads = {
        key: read_json(path)
        for key, path in paths.items()
        if key not in {"draws"}
    }
    shas = {key: sha256_file(path) for key, path in paths.items()}
    commit = git_commit(PROJECT_ROOT)
    preflight = payloads["preflight"]
    world = payloads["world"]
    stochastic = payloads["stochastic"]
    target = payloads["targets"]
    machinery = payloads["machinery"]
    launcher = payloads["launcher"]
    conditions = phase_conditions(args.phase)

    if (
        preflight.get("artifact_type") != "asre_salvage_b_preflight_report"
        or preflight.get("protocol") != SALVAGE_B_PROTOCOL
        or preflight.get("status") != "compatible"
        or preflight.get("git_commit_hash") != commit
        or preflight.get("scope", {}).get("conditions")
        != list(ENDPOINT_CONDITIONS + PROJECTED_CONDITIONS)
    ):
        raise ValueError("Incompatible Salvage-B preflight report.")
    if (
        world.get("artifact_type") != "asre_salvage_b_world_evaluation_manifest"
        or int(world.get("schema_version", -1)) != 2
        or world.get("status") != "frozen_before_metrics"
        or world.get("git_commit_hash") != commit
        or int(world.get("sample_count", -1)) != SAMPLES
        or world.get("preflight_report_sha256") != shas["preflight"]
        or Path(str(world.get("draw_tensor_path", ""))).resolve() != paths["draws"]
        or world.get("draw_tensor_sha256") != shas["draws"]
        or bool(world.get("outcome_based_selection", True))
        or bool(world.get("contains_padding", True))
        or not bool(world.get("target_source_files_frozen_before_metrics", False))
        or world.get("native_world_metric", {}).get("name") != NATIVE_WORLD_METRIC
        or world.get("native_world_metric", {}).get("direction") != "lower_is_better"
        or world.get("native_world_metric", {}).get("inference_steps")
        != NATIVE_WORLD_INFERENCE_STEPS
        or world.get("native_world_metric", {}).get("inference_shift")
        != NATIVE_WORLD_INFERENCE_SHIFT
        or world.get("native_world_metric", {}).get("target_usage")
        != "scoring_only_after_inference"
    ):
        raise ValueError("Incompatible frozen world manifest.")
    records = world.get("records")
    if not isinstance(records, list):
        raise ValueError("World manifest records are malformed.")
    _validate_unique_world_records(records)
    if (
        stochastic.get("artifact_type") != "asre_salvage_b_stochastic_manifest"
        or int(stochastic.get("schema_version", -1)) != 2
        or stochastic.get("status") != "frozen_before_metrics"
        or stochastic.get("git_commit_hash") != commit
        or int(stochastic.get("sample_count", -1)) != SAMPLES
        or int(stochastic.get("draws_per_sample", -1)) != WORLD_DRAWS_PER_SAMPLE
        or stochastic.get("preflight_report_sha256") != shas["preflight"]
        or stochastic.get("world_manifest_sha256") != shas["world"]
        or stochastic.get("draw_tensor_sha256") != shas["draws"]
        or Path(str(stochastic.get("draw_tensor_path", ""))).resolve() != paths["draws"]
        or stochastic.get("native_world_metric") != NATIVE_WORLD_METRIC
        or int(stochastic.get("video_inference_steps", -1))
        != NATIVE_WORLD_INFERENCE_STEPS
        or float(stochastic.get("video_inference_shift", math.nan))
        != NATIVE_WORLD_INFERENCE_SHIFT
        or stochastic.get("pairing_rule")
        != (
            "same real target, pure-noise initialization, native inference "
            "schedule, text/proprio, and scheduler state across all conditions"
        )
    ):
        raise ValueError("Incompatible frozen stochastic manifest.")
    stochastic_by_key = _validate_stochastic_records(
        sample_ids={str(record["sample_id"]) for record in records},
        stochastic=stochastic,
    )
    if (
        target.get("artifact_type")
        != "asre_salvage_b_processed_world_target_manifest"
        or int(target.get("schema_version", -1)) != 2
        or target.get("protocol") != SALVAGE_B_PROTOCOL
        or target.get("status") != "frozen_before_gpu_metrics"
        or target.get("git_commit_hash") != commit
        or int(target.get("sample_count", -1)) != SAMPLES
        or target.get("preflight_report_sha256") != shas["preflight"]
        or target.get("world_manifest_sha256") != shas["world"]
        or bool(target.get("model_outcomes_inspected", True))
        or bool(target.get("gpu_metric_executed", True))
    ):
        raise ValueError("Incompatible processed world-target manifest.")
    targets_by_id = _validate_target_records(world_records=records, target=target)
    expected_machinery = {
        "git_commit_hash": commit,
        "preflight_report_sha256": shas["preflight"],
        "world_manifest_sha256": shas["world"],
        "stochastic_manifest_sha256": shas["stochastic"],
        "draw_tensors_sha256": shas["draws"],
        "processed_targets_sha256": shas["targets"],
    }
    if (
        machinery.get("status") != "passed"
        or machinery.get("passed") is not True
        or any(machinery.get(key) != value for key, value in expected_machinery.items())
    ):
        raise ValueError("Salvage-B world workers require the matching passed machinery gate.")
    expected_identity_hash = launcher.get("identity_sha256")
    canonical_launcher = dict(launcher)
    canonical_launcher.pop("identity_sha256", None)
    if (
        not isinstance(expected_identity_hash, str)
        or expected_identity_hash != sha256_json(canonical_launcher)
        or int(launcher.get("schema_version", -1)) != 2
        or launcher.get("protocol") != SALVAGE_B_PROTOCOL
        or launcher.get("phase") != args.phase
        or launcher.get("git_commit_hash") != commit
        or launcher.get("conditions") != list(conditions)
        or launcher.get("no_ddp") is not True
        or launcher.get("preflight_report_sha256") != shas["preflight"]
        or launcher.get("world_manifest_sha256") != shas["world"]
        or launcher.get("stochastic_manifest_sha256") != shas["stochastic"]
        or launcher.get("draw_tensors_sha256") != shas["draws"]
        or launcher.get("target_manifest_sha256") != shas["targets"]
        or launcher.get("machinery_sha256") != shas["machinery"]
        or launcher.get("native_metric") != NATIVE_WORLD_METRIC
        or launcher.get("inference_steps") != NATIVE_WORLD_INFERENCE_STEPS
        or launcher.get("inference_shift") != NATIVE_WORLD_INFERENCE_SHIFT
    ):
        raise ValueError("Worker launcher configuration is incompatible.")

    endpoint_gate = None
    endpoint_identity_by_sample = None
    if args.phase == "endpoint":
        if args.endpoint_gate is not None or launcher.get("endpoint_gate_sha256") is not None:
            raise ValueError("Endpoint phase must not consume a projected-phase endpoint gate.")
    else:
        if args.endpoint_gate is None:
            raise ValueError("Projected phase requires --endpoint-gate.")
        gate_path = _require_file(args.endpoint_gate, "passed world endpoint gate")
        gate_sha = sha256_file(gate_path)
        endpoint_gate = read_json(gate_path)
        validate_endpoint_gate_payload(
            endpoint_gate,
            world_manifest_sha256=shas["world"],
            stochastic_manifest_sha256=shas["stochastic"],
            target_manifest_sha256=shas["targets"],
            machinery_sha256=shas["machinery"],
            git_commit_hash=commit,
        )
        endpoint_identity_by_sample = _endpoint_identity_by_sample(endpoint_gate)
        if set(endpoint_identity_by_sample) != {
            str(record["sample_id"]) for record in records
        }:
            raise ValueError("Projected endpoint identities do not cover the world manifest.")
        if launcher.get("endpoint_gate_sha256") != gate_sha:
            raise ValueError("Projected phase endpoint gate hash is mismatched.")
        paths["endpoint_gate"] = gate_path
        shas["endpoint_gate"] = gate_sha

    draw_payload = None
    draws_by_sample = None
    if load_draw_tensors:
        draw_payload = torch.load(paths["draws"], map_location="cpu", weights_only=False)
        if not isinstance(draw_payload, Mapping):
            raise ValueError("Frozen draw tensor artifact is malformed.")
        draws_by_sample = _validate_draw_payload(
            draw_payload=draw_payload,
            stochastic_by_key=stochastic_by_key,
        )
        if sha256_file(paths["draws"]) != shas["draws"]:
            raise ValueError("Frozen draw tensor artifact changed while it was validated.")
    return {
        "paths": paths,
        "payloads": payloads,
        "shas": shas,
        "commit": commit,
        "conditions": conditions,
        "records": [dict(record) for record in records],
        "targets_by_id": targets_by_id,
        "stochastic_by_key": stochastic_by_key,
        "draw_payload": draw_payload,
        "draws_by_sample": draws_by_sample,
        "endpoint_gate": endpoint_gate,
        "endpoint_identity_by_sample": endpoint_identity_by_sample,
    }


def _guard_frozen_source(contract: Mapping[str, Any]) -> None:
    """Fail closed on both Git state and the ignored Phase-A source audit."""

    preflight = contract.get("payloads", {}).get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("Worker contract lacks a frozen preflight payload.")
    raw_output_root = preflight.get("output_root")
    if not isinstance(raw_output_root, str) or not raw_output_root:
        raise ValueError("Worker preflight lacks its frozen output root.")
    expected_commit = str(contract.get("commit", ""))
    require_clean_source(
        expected_commit,
        output_root=Path(raw_output_root).expanduser().resolve(),
    )
    validate_architecture_source_inventory(
        preflight,
        expected_commit=expected_commit,
    )


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed shard row {path}:{line_number}.") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Shard row is not an object: {path}:{line_number}.")
            rows.append(value)
    return rows


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    worker_index: int,
    expected_sample_ids: Sequence[str],
    conditions: Sequence[str],
    provenance: Mapping[str, str],
) -> None:
    if len(rows) != ROWS_PER_WORKER:
        raise ValueError(f"Shard must contain exactly {ROWS_PER_WORKER} metric rows.")
    expected = {
        (sample_id, draw_id, condition)
        for sample_id in expected_sample_ids
        for draw_id in range(WORLD_DRAWS_PER_SAMPLE)
        for condition in conditions
    }
    observed: set[tuple[str, int, str]] = set()
    paired: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    sample_identity: dict[str, tuple[Any, ...]] = {}
    required_provenance = {
        "world_manifest_sha256": provenance["world_manifest_sha256"],
        "stochastic_manifest_sha256": provenance["stochastic_manifest_sha256"],
        "target_manifest_sha256": provenance["target_manifest_sha256"],
        "machinery_sha256": provenance["machinery_sha256"],
        "git_commit": provenance["git_commit"],
    }
    for row in rows:
        sample_id = str(row.get("sample_id"))
        draw_id = int(row.get("draw_id", -1))
        condition = str(row.get("condition"))
        key = (sample_id, draw_id, condition)
        if key in observed or key not in expected:
            raise ValueError(f"Duplicate or unexpected metric row: {key}.")
        observed.add(key)
        if row.get("phase") != phase or int(row.get("worker_index", -1)) != worker_index:
            raise ValueError(f"Metric row phase/worker drifted: {key}.")
        if row.get("schema_version") != 2 or row.get("protocol") != SALVAGE_B_PROTOCOL:
            raise ValueError(f"Metric row schema/protocol drifted: {key}.")
        if any(row.get(name) != value for name, value in required_provenance.items()):
            raise ValueError(f"Metric row provenance drifted: {key}.")
        if int(row.get("rank", -1)) != RANK_BY_CONDITION[condition]:
            raise ValueError(f"Metric row rank drifted: {key}.")
        try:
            native = float(row.get("native_world_loss", math.nan))
            latent = float(row.get("future_latent_mse", math.nan))
            inference_steps = int(row.get("inference_steps", -1))
            inference_shift = float(row.get("inference_shift", math.nan))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric row contains invalid numeric values: {key}.") from exc
        if (
            not math.isfinite(native)
            or native < 0.0
            or not math.isfinite(latent)
            or latent < 0.0
            or inference_steps != NATIVE_WORLD_INFERENCE_STEPS
            or not math.isclose(
                inference_shift,
                NATIVE_WORLD_INFERENCE_SHIFT,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(native, latent, rel_tol=1e-7, abs_tol=1e-9)
        ):
            raise ValueError(f"Metric row contains invalid numeric values: {key}.")
        for name in (
            "target_sha256",
            "target_latent_sha256",
            "video_noise_sha256",
            "current_image_sha256",
            "current_frame_latent_sha256",
            "donor_image_sha256",
        ):
            value = row.get(name)
            if not _is_sha256(value):
                raise ValueError(f"Metric row {name} is malformed: {key}.")
        if (
            row.get("prediction_shape") != [1, 48, 2, 14, 28]
            or row.get("target_shape") != [1, 48, 2, 14, 28]
            or row.get("future_token_shape") != [1, 196, EXPECTED_FEATURE_DIM]
        ):
            raise ValueError(f"Metric row native-inference shapes drifted: {key}.")
        identity = (
            int(row.get("task_id", -1)),
            int(row.get("episode_id", -1)),
            int(row.get("trial_index", -1)),
            str(row["target_sha256"]),
            str(row["target_latent_sha256"]),
            str(row["current_image_sha256"]),
            str(row["current_frame_latent_sha256"]),
            str(row["donor_image_sha256"]),
        )
        if sample_id in sample_identity and sample_identity[sample_id] != identity:
            raise ValueError(f"Sample identity changed across rows: {sample_id}.")
        sample_identity[sample_id] = identity
        paired.setdefault((sample_id, draw_id), []).append(row)
    if observed != expected or set(sample_identity) != set(expected_sample_ids):
        raise ValueError("Metric rows do not form the exact registered Cartesian product.")
    for pair, values in paired.items():
        if len(values) != len(conditions):
            raise ValueError(f"Incomplete condition pairing for {pair}.")
        reference = values[0]
        for other in values[1:]:
            for name in (
                "target_sha256",
                "target_latent_sha256",
                "current_image_sha256",
                "current_frame_latent_sha256",
                "donor_image_sha256",
                "video_noise_sha256",
                "inference_steps",
                "inference_shift",
            ):
                if other[name] != reference[name]:
                    raise ValueError(f"Frozen pairing drifted for {pair}/{name}.")


def shard_paths(output_dir: Path, phase: str, worker_index: int) -> tuple[Path, Path]:
    root = output_dir.resolve() / phase / f"worker_{worker_index:02d}"
    return root / "metadata.json", root / "rows.jsonl"


def validate_completed_shard(
    *,
    output_dir: Path,
    phase: str,
    worker_index: int,
    expected_sample_ids: Sequence[str],
    conditions: Sequence[str],
    provenance: Mapping[str, str],
    launcher_config_sha256: str,
) -> bool:
    metadata_path, rows_path = shard_paths(output_dir, phase, worker_index)
    if not metadata_path.exists() and not rows_path.exists():
        if metadata_path.parent.exists():
            raise RuntimeError(
                f"Refusing incomplete Salvage-B shard resume: {metadata_path.parent}."
            )
        return False
    if not metadata_path.is_file() or not rows_path.is_file():
        raise RuntimeError(
            f"Refusing partial Salvage-B shard resume: {metadata_path.parent}."
        )
    metadata = read_json(metadata_path)
    expected_metadata = {
        "schema_version": 2,
        "status": "completed",
        "phase": phase,
        "worker_index": worker_index,
        "record_count": ROWS_PER_WORKER,
        "sample_count": SAMPLES_PER_WORKER,
        "conditions": list(conditions),
        "draws_per_sample": WORLD_DRAWS_PER_SAMPLE,
        "world_manifest_sha256": provenance["world_manifest_sha256"],
        "stochastic_manifest_sha256": provenance["stochastic_manifest_sha256"],
        "target_manifest_sha256": provenance["target_manifest_sha256"],
        "machinery_sha256": provenance["machinery_sha256"],
        "git_commit": provenance["git_commit"],
        "launcher_config_sha256": launcher_config_sha256,
        "rows_sha256": sha256_file(rows_path),
        "sample_ids_sha256": sha256_json(list(expected_sample_ids)),
        "native_metric": NATIVE_WORLD_METRIC,
        "inference_steps": NATIVE_WORLD_INFERENCE_STEPS,
        "inference_shift": NATIVE_WORLD_INFERENCE_SHIFT,
        "no_ddp": True,
    }
    mismatch = {
        key: {"observed": metadata.get(key), "expected": value}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Refusing incompatible Salvage-B shard resume: {mismatch}.")
    rows = _read_rows(rows_path)
    validate_rows(
        rows,
        phase=phase,
        worker_index=worker_index,
        expected_sample_ids=expected_sample_ids,
        conditions=conditions,
        provenance=provenance,
    )
    return True


def _isolated_gpu(worker_index: int) -> int:
    if worker_index not in range(WORKERS):
        raise ValueError("worker-index must be 0..3.")
    physical = os.environ.get("ASRE_SALVAGE_B_PHYSICAL_GPU")
    if physical is None or os.environ.get("CUDA_VISIBLE_DEVICES") != physical:
        raise RuntimeError("Salvage-B worker requires an auditable isolated physical GPU.")
    if any(os.environ.get(key) is not None for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise RuntimeError("Salvage-B world evaluation forbids distributed/DDP state.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Salvage-B worker requires exactly one visible CUDA device.")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("Salvage-B world evaluation must not initialize torch.distributed.")
    return int(physical)


def _validate_cache(
    cache_k: Sequence[torch.Tensor], cache_v: Sequence[torch.Tensor], *, condition: str
) -> None:
    if len(cache_k) != 30 or len(cache_v) != 30:
        raise ValueError(f"Condition {condition} cache does not contain 30 layers.")
    for layer, (key, value) in enumerate(zip(cache_k, cache_v)):
        if (
            tuple(key.shape) != (1, PREFIX_TOKENS, EXPECTED_FEATURE_DIM)
            or tuple(value.shape) != (1, PREFIX_TOKENS, EXPECTED_FEATURE_DIM)
            or key.device != value.device
            or key.dtype != value.dtype
            or not bool(torch.isfinite(key).all())
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Condition {condition} cache layout drifted at layer {layer}.")


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.phase not in PHASES:
        raise ValueError(f"Unsupported world phase: {args.phase}.")
    physical_gpu = _isolated_gpu(args.worker_index)
    contract = load_frozen_contract(args, load_draw_tensors=True)
    # This guard deliberately precedes the completed-shard fast return: an old
    # result must not make a dirty or source-drifted formal invocation succeed.
    _guard_frozen_source(contract)
    configured_gpus = contract["payloads"]["launcher"].get("gpu_ids")
    if (
        not isinstance(configured_gpus, list)
        or len(configured_gpus) != WORKERS
        or int(configured_gpus[args.worker_index]) != physical_gpu
    ):
        raise ValueError("Worker physical GPU differs from frozen launcher mapping.")
    conditions = contract["conditions"]
    records = worker_records(contract["records"], args.worker_index)
    sample_ids = [str(record["sample_id"]) for record in records]
    provenance = {
        "world_manifest_sha256": contract["shas"]["world"],
        "stochastic_manifest_sha256": contract["shas"]["stochastic"],
        "target_manifest_sha256": contract["shas"]["targets"],
        "machinery_sha256": contract["shas"]["machinery"],
        "git_commit": contract["commit"],
    }
    if validate_completed_shard(
        output_dir=args.output_dir,
        phase=args.phase,
        worker_index=args.worker_index,
        expected_sample_ids=sample_ids,
        conditions=conditions,
        provenance=provenance,
        launcher_config_sha256=contract["shas"]["launcher"],
    ):
        metadata_path, _ = shard_paths(args.output_dir, args.phase, args.worker_index)
        return read_json(metadata_path)

    set_global_seed(42, get_worker_init_fn=False)
    dataset = load_world_dataset(
        preflight=contract["payloads"]["preflight"],
        runtime_work_dir=args.runtime_work_dir.resolve()
        / args.phase
        / f"worker_{args.worker_index:02d}",
    )
    prompt_cache = load_prompt_cache(contract["payloads"]["preflight"])
    donor_bundle = load_donor_bundle(contract["payloads"]["preflight"])
    model, _cfg = load_frozen_model(contract["payloads"]["preflight"])
    if model.mot.num_layers != 30 or model.mot.num_heads != 24 or model.mot.attn_head_dim != 128:
        raise ValueError("Frozen Fast-WAM shared-cache architecture drifted.")
    rank_specs = (
        load_rank_specs(model=model, preflight=contract["payloads"]["preflight"])
        if args.phase == "projected"
        else {}
    )

    rows: list[dict[str, Any]] = []
    start = now_iso()
    with torch.inference_mode():
        for position, record in enumerate(records, start=1):
            sample_id = str(record["sample_id"])
            target_record = contract["targets_by_id"][sample_id]
            processed = load_processed_sample(
                dataset=dataset,
                record=record,
                prompt_cache=prompt_cache,
                target_record=target_record,
            )
            prepared = prepare_world_sample(
                model=model,
                processed=processed,
                record=record,
                donor_bundle=donor_bundle,
            )
            current_frame_latent_sha256 = tensor_sha256(
                prepared.current_frame_latent.detach().cpu()
            )
            sample_identity = {
                "sample_id": sample_id,
                "task_id": int(record["task_id"]),
                "episode_id": int(record["episode_id"]),
                "trial_index": int(record["trial"]),
                "target_sha256": str(
                    target_record["processed_tensor_sha256"]["video"]
                ),
                "target_latent_sha256": prepared.target_latent_sha256,
                "current_image_sha256": prepared.current_image_sha256,
                "current_frame_latent_sha256": current_frame_latent_sha256,
                "donor_image_sha256": prepared.donor_image_sha256,
            }
            if prepared.current_image_sha256 != str(target_record["current_image_sha256"]):
                raise ValueError(f"Frozen current image drifted for {sample_id}.")
            if prepared.donor_image_sha256 != str(record["donor_processed_image_sha256"]):
                raise ValueError(f"Frozen donor image drifted for {sample_id}.")
            if args.phase == "projected":
                endpoint_identity = contract["endpoint_identity_by_sample"].get(sample_id)
                if endpoint_identity != sample_identity:
                    raise ValueError(
                        f"Projected sample identity differs from endpoint phase: {sample_id}."
                    )
            selected: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
            for condition in conditions:
                cache_k, cache_v, audit = condition_cache(
                    prepared=prepared,
                    condition=condition,
                    rank_specs=rank_specs,
                )
                _validate_cache(cache_k, cache_v, condition=condition)
                if condition.startswith("svd_r"):
                    rank = RANK_BY_CONDITION[condition]
                    if (
                        not isinstance(audit, Mapping)
                        or audit.get("mode") != "feature_subspace"
                        or audit.get("projection_rank") != rank
                        or audit.get("feature_dim") != EXPECTED_FEATURE_DIM
                        or audit.get("video_seq_len") != PREFIX_TOKENS
                        or audit.get("action_visible_token_indices")
                        != list(range(PREFIX_TOKENS))
                        or audit.get("replacement_video_layers") != list(LATE_LAYERS)
                        or audit.get("shape_preserved") is not True
                        or audit.get("tokens_modified") is not False
                        or audit.get("heads_modified") is not False
                    ):
                        raise ValueError(f"Projection audit drifted for {condition}.")
                elif audit is not None:
                    raise ValueError(f"Endpoint condition unexpectedly emitted projection audit: {condition}.")
                selected[condition] = (cache_k, cache_v)

            target_sha = str(target_record["processed_tensor_sha256"]["video"])
            draws = contract["draws_by_sample"][sample_id]
            for draw in draws:
                draw_id = int(draw["draw_id"])
                stochastic = contract["stochastic_by_key"][(sample_id, draw_id)]
                noise_sha = str(stochastic["video_noise_sha256"])
                for condition in conditions:
                    cache_k, cache_v = selected[condition]
                    result = native_world_loss(
                        model=model,
                        prepared=prepared,
                        video_cache_k=cache_k,
                        video_cache_v=cache_v,
                        video_noise=draw["video_noise"],
                        disabled_video_prefix_layers=EARLY_DISABLED,
                    )
                    numeric = (
                        float(result["native_world_loss"]),
                        float(result["future_latent_mse"]),
                    )
                    if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
                        raise ValueError(f"Non-finite native world metric for {sample_id}.")
                    if (
                        not math.isclose(numeric[0], numeric[1], rel_tol=1e-7, abs_tol=1e-9)
                        or int(result["inference_steps"])
                        != NATIVE_WORLD_INFERENCE_STEPS
                        or float(result["inference_shift"])
                        != NATIVE_WORLD_INFERENCE_SHIFT
                        or result["prediction_shape"] != [1, 48, 2, 14, 28]
                        or result["target_shape"] != [1, 48, 2, 14, 28]
                        or result["future_token_shape"] != [1, 196, 3072]
                    ):
                        raise ValueError("Native future prediction shape drifted.")
                    rows.append(
                        {
                            "schema_version": 2,
                            "protocol": SALVAGE_B_PROTOCOL,
                            "phase": args.phase,
                            "worker_index": args.worker_index,
                            "sample_id": sample_id,
                            "task_id": int(record["task_id"]),
                            "episode_id": int(record["episode_id"]),
                            "trial_index": int(record["trial"]),
                            "draw_id": draw_id,
                            "condition": condition,
                            "rank": RANK_BY_CONDITION[condition],
                            "native_world_loss": numeric[0],
                            "future_latent_mse": numeric[1],
                            "inference_steps": int(result["inference_steps"]),
                            "inference_shift": float(result["inference_shift"]),
                            "target_sha256": target_sha,
                            "target_latent_sha256": prepared.target_latent_sha256,
                            "video_noise_sha256": noise_sha,
                            "current_image_sha256": prepared.current_image_sha256,
                            "current_frame_latent_sha256": current_frame_latent_sha256,
                            "donor_image_sha256": prepared.donor_image_sha256,
                            "world_manifest_sha256": provenance["world_manifest_sha256"],
                            "stochastic_manifest_sha256": provenance[
                                "stochastic_manifest_sha256"
                            ],
                            "target_manifest_sha256": provenance["target_manifest_sha256"],
                            "machinery_sha256": provenance["machinery_sha256"],
                            "git_commit": provenance["git_commit"],
                            "checkpoint_sha256": contract["payloads"]["preflight"][
                                "state"
                            ]["checkpoint_sha256"],
                            "basis_manifest_sha256": contract["payloads"]["preflight"][
                                "basis"
                            ]["sha256"],
                            "basis_split_sha256": contract["payloads"]["preflight"][
                                "basis"
                            ]["split_sha256"],
                            "donor_mapping_sha256": contract["payloads"]["preflight"][
                                "donors"
                            ]["mapping_sha256"],
                            "donor_manifest_sha256": contract["payloads"]["preflight"][
                                "donors"
                            ]["manifest_sha256"],
                            "early_prefix_layers_disabled": list(EARLY_DISABLED),
                            "late_shared_cache_layers": list(LATE_LAYERS),
                            "prediction_shape": result["prediction_shape"],
                            "target_shape": result["target_shape"],
                            "future_token_shape": result["future_token_shape"],
                        }
                    )
                    del result
            del processed, prepared, selected
            if position % 5 == 0:
                print(
                    f"[SalvageB {args.phase} shard {args.worker_index}] "
                    f"{position}/{len(records)} samples",
                    flush=True,
                )

    validate_rows(
        rows,
        phase=args.phase,
        worker_index=args.worker_index,
        expected_sample_ids=sample_ids,
        conditions=conditions,
        provenance=provenance,
    )
    # Detect any frozen artifact mutation during the potentially long GPU pass.
    for key in ("preflight", "world", "stochastic", "draws", "targets", "machinery", "launcher"):
        if sha256_file(contract["paths"][key]) != contract["shas"][key]:
            raise ValueError(f"Frozen Salvage-B artifact changed during worker run: {key}.")
    if args.phase == "projected":
        if sha256_file(contract["paths"]["endpoint_gate"]) != contract["shas"]["endpoint_gate"]:
            raise ValueError("Passed endpoint gate changed during projected worker run.")
        validate_endpoint_gate_payload(
            contract["endpoint_gate"],
            world_manifest_sha256=contract["shas"]["world"],
            stochastic_manifest_sha256=contract["shas"]["stochastic"],
            target_manifest_sha256=contract["shas"]["targets"],
            machinery_sha256=contract["shas"]["machinery"],
            git_commit_hash=contract["commit"],
        )

    # Recheck after the long GPU pass and before creating or atomically
    # publishing a shard.  The architecture audit is under the ignored output
    # root, so it is independently authenticated by _guard_frozen_source.
    _guard_frozen_source(contract)

    metadata_path, rows_path = shard_paths(args.output_dir, args.phase, args.worker_index)
    shard_root = metadata_path.parent
    if shard_root.exists():
        raise RuntimeError(f"Refusing to overwrite a Salvage-B shard: {shard_root}.")
    shard_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            dir=shard_root.parent,
            prefix=f".{shard_root.name}.",
        )
    )
    temporary_rows = temporary_root / "rows.jsonl"
    temporary_metadata = temporary_root / "metadata.json"
    _atomic_write_jsonl(temporary_rows, rows)
    metadata = {
        "artifact_type": "asre_salvage_b_world_metric_shard",
        "schema_version": 2,
        "protocol": SALVAGE_B_PROTOCOL,
        "status": "completed",
        "phase": args.phase,
        "worker_index": args.worker_index,
        "physical_gpu": physical_gpu,
        "logical_device": "cuda:0",
        "record_count": len(rows),
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "sample_ids_sha256": sha256_json(sample_ids),
        "conditions": list(conditions),
        "draws_per_sample": WORLD_DRAWS_PER_SAMPLE,
        "world_manifest_sha256": provenance["world_manifest_sha256"],
        "stochastic_manifest_sha256": provenance["stochastic_manifest_sha256"],
        "target_manifest_sha256": provenance["target_manifest_sha256"],
        "machinery_sha256": provenance["machinery_sha256"],
        "git_commit": provenance["git_commit"],
        "preflight_report_sha256": contract["shas"]["preflight"],
        "draw_tensors_sha256": contract["shas"]["draws"],
        "checkpoint_sha256": contract["payloads"]["preflight"]["state"][
            "checkpoint_sha256"
        ],
        "basis_manifest_sha256": contract["payloads"]["preflight"]["basis"][
            "sha256"
        ],
        "basis_split_sha256": contract["payloads"]["preflight"]["basis"][
            "split_sha256"
        ],
        "donor_mapping_sha256": contract["payloads"]["preflight"]["donors"][
            "mapping_sha256"
        ],
        "donor_manifest_sha256": contract["payloads"]["preflight"]["donors"][
            "manifest_sha256"
        ],
        "launcher_config_sha256": contract["shas"]["launcher"],
        "endpoint_gate_sha256": contract["shas"].get("endpoint_gate"),
        "rows_path": str(rows_path),
        "rows_sha256": sha256_file(temporary_rows),
        "native_metric": NATIVE_WORLD_METRIC,
        "metric_direction": "lower_is_better",
        "inference_steps": NATIVE_WORLD_INFERENCE_STEPS,
        "inference_shift": NATIVE_WORLD_INFERENCE_SHIFT,
        "initial_state": "pure_gaussian_future_latent_noise",
        "target_usage": "scoring_only_after_inference",
        "fixed_draws": True,
        "paired_conditions_within_worker": True,
        "action_draws_consumed_by_world": False,
        "online_episodes": 0,
        "environment_rollouts": 0,
        "action_rerun": False,
        "heldout_svd_refit": False,
        "no_ddp": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "start_timestamp": start,
        "end_timestamp": now_iso(),
    }
    try:
        atomic_write_json(temporary_metadata, metadata)
        # Directory rename publishes metadata and rows as one completion unit,
        # so a killed worker cannot leave a half-visible resumable shard.
        os.replace(temporary_root, shard_root)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
    if not validate_completed_shard(
        output_dir=args.output_dir,
        phase=args.phase,
        worker_index=args.worker_index,
        expected_sample_ids=sample_ids,
        conditions=conditions,
        provenance=provenance,
        launcher_config_sha256=contract["shas"]["launcher"],
    ):
        raise AssertionError("Atomic shard validation unexpectedly returned false.")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--world-manifest", type=Path, required=True)
    parser.add_argument("--stochastic-manifest", type=Path, required=True)
    parser.add_argument("--draw-tensors", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--machinery", type=Path, required=True)
    parser.add_argument("--launcher-config", type=Path, required=True)
    parser.add_argument("--endpoint-gate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-work-dir", type=Path, required=True)
    metadata = run_worker(parser.parse_args())
    print(
        f"Salvage-B {metadata['phase']} shard {metadata['worker_index']} complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
