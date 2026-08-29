"""Strictly merge one Salvage-B world-evaluation phase on CPU.

Each phase is produced by exactly four independent GPU workers.  This module
does not run a model: it fail-closes on shard completeness, frozen pairing,
and provenance before emitting sample-averaged native-loss artifacts.  The
endpoint phase alone applies the registered Current-vs-Wrong gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.asre_diagnosis.common import (
    SALVAGE_B_PROTOCOL,
    atomic_write_json,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.salvage_b.definitions import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    ENDPOINT_CONDITIONS,
    PROJECTED_CONDITIONS,
    RANK_BY_CONDITION,
    WORLD_DRAWS_PER_SAMPLE,
)
from experiments.asre_diagnosis.salvage_b.world_manifest import (
    NATIVE_WORLD_METRIC,
    VIDEO_INFERENCE_SHIFT,
    VIDEO_INFERENCE_STEPS,
)
from experiments.asre_diagnosis.salvage_b.statistics import (
    paired_bootstrap_ci,
    task_hierarchical_bootstrap_ci,
)


PHASE_CONDITIONS = {
    "endpoint": ENDPOINT_CONDITIONS,
    "projected": PROJECTED_CONDITIONS,
}
WORKER_COUNT = 4
SAMPLE_COUNT = 100
SAMPLES_PER_WORKER = SAMPLE_COUNT // WORKER_COUNT

ROW_FIELDS = (
    "schema_version",
    "protocol",
    "phase",
    "worker_index",
    "sample_id",
    "task_id",
    "episode_id",
    "trial_index",
    "draw_id",
    "condition",
    "rank",
    "native_world_loss",
    "future_latent_mse",
    "inference_steps",
    "inference_shift",
    "target_sha256",
    "target_latent_sha256",
    "current_image_sha256",
    "current_frame_latent_sha256",
    "donor_image_sha256",
    "video_noise_sha256",
    "prediction_shape",
    "target_shape",
    "future_token_shape",
    "world_manifest_sha256",
    "stochastic_manifest_sha256",
    "target_manifest_sha256",
    "machinery_sha256",
    "git_commit",
)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"Blank JSONL record at {path}:{line_number}.")
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}.") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}.")
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _metadata_field(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    provenance = metadata.get("provenance")
    if isinstance(provenance, Mapping) and key in provenance:
        return provenance[key]
    if key == "git_commit" and "git_commit_hash" in metadata:
        return metadata["git_commit_hash"]
    if key == "git_commit" and isinstance(provenance, Mapping):
        return provenance.get("git_commit_hash")
    return None


def _finite_number(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, got bool.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "nonnegative " if nonnegative else ""
        raise ValueError(f"{label} must be a finite {qualifier}number, got {value!r}.")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got bool.")
    result = int(value)
    if float(value) != float(result):
        raise ValueError(f"{label} must be an integer, got {value!r}.")
    return result


def _validate_frozen_inputs(
    *,
    world_manifest_path: Path,
    stochastic_manifest_path: Path,
    target_manifest_path: Path,
    machinery_path: Path,
    preflight_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    world = _read_json(world_manifest_path)
    stochastic = _read_json(stochastic_manifest_path)
    targets = _read_json(target_manifest_path)
    machinery = _read_json(machinery_path)
    preflight = _read_json(preflight_path)
    world_metric = world.get("native_world_metric")
    if not (
        world.get("artifact_type") == "asre_salvage_b_world_evaluation_manifest"
        and world.get("schema_version") == 2
        and world.get("status") == "frozen_before_metrics"
        and world.get("sample_count") == SAMPLE_COUNT
        and isinstance(world_metric, Mapping)
        and world_metric.get("name") == NATIVE_WORLD_METRIC
        and world_metric.get("direction") == "lower_is_better"
        and world_metric.get("inference_steps") == VIDEO_INFERENCE_STEPS
        and world_metric.get("inference_shift") == VIDEO_INFERENCE_SHIFT
        and world_metric.get("target_usage") == "scoring_only_after_inference"
        and stochastic.get("artifact_type") == "asre_salvage_b_stochastic_manifest"
        and stochastic.get("schema_version") == 2
        and stochastic.get("status") == "frozen_before_metrics"
        and stochastic.get("sample_count") == SAMPLE_COUNT
        and stochastic.get("draws_per_sample") == WORLD_DRAWS_PER_SAMPLE
        and stochastic.get("native_world_metric") == NATIVE_WORLD_METRIC
        and stochastic.get("video_inference_steps") == VIDEO_INFERENCE_STEPS
        and stochastic.get("video_inference_shift") == VIDEO_INFERENCE_SHIFT
        and targets.get("artifact_type")
        == "asre_salvage_b_processed_world_target_manifest"
        and targets.get("schema_version") == 2
        and targets.get("status") == "frozen_before_gpu_metrics"
        and targets.get("sample_count") == SAMPLE_COUNT
        and preflight.get("artifact_type") == "asre_salvage_b_preflight_report"
        and preflight.get("status") == "compatible"
    ):
        raise ValueError("A frozen Salvage-B input manifest is incomplete or incompatible.")
    if machinery.get("passed") is not True:
        raise ValueError("Shared-interface runtime machinery did not pass.")
    commits = {
        str(world.get("git_commit_hash")),
        str(stochastic.get("git_commit_hash")),
        str(targets.get("git_commit_hash")),
        str(preflight.get("git_commit_hash")),
    }
    if len(commits) != 1 or "None" in commits or "" in commits:
        raise ValueError("Frozen Salvage-B input commits do not match.")
    hashes = {
        "world_manifest_sha256": sha256_file(world_manifest_path),
        "stochastic_manifest_sha256": sha256_file(stochastic_manifest_path),
        "target_manifest_sha256": sha256_file(target_manifest_path),
        "machinery_sha256": sha256_file(machinery_path),
        "preflight_report_sha256": sha256_file(preflight_path),
        "git_commit": next(iter(commits)),
    }
    if stochastic.get("world_manifest_sha256") != hashes["world_manifest_sha256"]:
        raise ValueError("Stochastic/world manifest identity drifted.")
    if targets.get("world_manifest_sha256") != hashes["world_manifest_sha256"]:
        raise ValueError("Processed-target/world manifest identity drifted.")
    if (
        world.get("preflight_report_sha256") != hashes["preflight_report_sha256"]
        or stochastic.get("preflight_report_sha256")
        != hashes["preflight_report_sha256"]
        or targets.get("preflight_report_sha256")
        != hashes["preflight_report_sha256"]
    ):
        raise ValueError("Frozen input/preflight identity drifted.")
    return world, stochastic, targets, machinery, hashes


def _manifest_indices(
    world: Mapping[str, Any],
    stochastic: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> tuple[
    list[str],
    dict[str, Mapping[str, Any]],
    dict[tuple[str, int], Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    world_records = world.get("records")
    stochastic_records = stochastic.get("records")
    target_records = targets.get("records")
    if not all(isinstance(value, list) for value in (world_records, stochastic_records, target_records)):
        raise TypeError("Frozen Salvage-B record collections must be lists.")
    assert isinstance(world_records, list)
    assert isinstance(stochastic_records, list)
    assert isinstance(target_records, list)
    world_by_id = {str(row["sample_id"]): row for row in world_records}
    target_by_id = {str(row["sample_id"]): row for row in target_records}
    stochastic_by_key = {
        (str(row["sample_id"]), int(row["draw_id"])): row
        for row in stochastic_records
    }
    ordered_ids = [str(row["sample_id"]) for row in world_records]
    if (
        len(ordered_ids) != SAMPLE_COUNT
        or len(world_by_id) != SAMPLE_COUNT
        or len(target_by_id) != SAMPLE_COUNT
        or set(target_by_id) != set(world_by_id)
        or len(stochastic_by_key) != SAMPLE_COUNT * WORLD_DRAWS_PER_SAMPLE
    ):
        raise ValueError("Frozen world/target/stochastic sample identities are incomplete.")
    for sample_id in ordered_ids:
        world_row = world_by_id[sample_id]
        target_row = target_by_id[sample_id]
        for field in ("task_id", "episode_id", "trial"):
            if int(world_row[field]) != int(target_row[field]):
                raise ValueError(f"Target/world {field} mismatch for {sample_id}.")
        if not _is_sha256(world_row.get("donor_processed_image_sha256")):
            raise ValueError(f"Frozen donor image hash is malformed for {sample_id}.")
        if not _is_sha256(target_row.get("current_image_sha256")):
            raise ValueError(f"Frozen current image hash is malformed for {sample_id}.")
        for draw_id in range(WORLD_DRAWS_PER_SAMPLE):
            if (sample_id, draw_id) not in stochastic_by_key:
                raise ValueError(f"Missing frozen draw {draw_id} for {sample_id}.")
            draw = stochastic_by_key[(sample_id, draw_id)]
            if "video_timestep" in draw or not _is_sha256(
                draw.get("video_noise_sha256")
            ):
                raise ValueError(f"Frozen native-inference draw drifted for {sample_id}.")
    return ordered_ids, world_by_id, stochastic_by_key, target_by_id


def _validate_metadata(
    *,
    metadata: Mapping[str, Any],
    phase: str,
    worker_index: int,
    conditions: tuple[str, ...],
    hashes: Mapping[str, str],
) -> None:
    expected_records = SAMPLES_PER_WORKER * len(conditions) * WORLD_DRAWS_PER_SAMPLE
    if not (
        metadata.get("schema_version") == 2
        and metadata.get("status") == "completed"
        and metadata.get("phase") == phase
        and metadata.get("worker_index") == worker_index
        and metadata.get("sample_count") == SAMPLES_PER_WORKER
        and metadata.get("record_count") == expected_records
        and metadata.get("draws_per_sample") == WORLD_DRAWS_PER_SAMPLE
        and metadata.get("conditions") == list(conditions)
        and metadata.get("native_metric") == NATIVE_WORLD_METRIC
        and metadata.get("inference_steps") == VIDEO_INFERENCE_STEPS
        and metadata.get("inference_shift") == VIDEO_INFERENCE_SHIFT
        and metadata.get("initial_state") == "pure_gaussian_future_latent_noise"
        and metadata.get("target_usage") == "scoring_only_after_inference"
        and metadata.get("no_ddp") is True
    ):
        raise ValueError(f"Malformed {phase} metadata for worker {worker_index:02d}.")
    for key in (
        "world_manifest_sha256",
        "stochastic_manifest_sha256",
        "target_manifest_sha256",
        "machinery_sha256",
        "git_commit",
    ):
        if _metadata_field(metadata, key) != hashes[key]:
            raise ValueError(
                f"Worker {worker_index:02d} metadata provenance drifted at {key}."
            )


def _validate_row(
    *,
    row: Mapping[str, Any],
    phase: str,
    worker_index: int,
    conditions: tuple[str, ...],
    hashes: Mapping[str, str],
    expected_sample_ids: set[str],
    world_by_id: Mapping[str, Mapping[str, Any]],
    stochastic_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    target_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    missing = [field for field in ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Worker row lacks required fields: {missing}.")
    if row["schema_version"] != 2 or row["protocol"] != SALVAGE_B_PROTOCOL:
        raise ValueError("Worker row schema/protocol drifted.")
    if row["phase"] != phase or _integer(row["worker_index"], label="worker_index") != worker_index:
        raise ValueError("Worker row phase/index drifted.")
    sample_id = str(row["sample_id"])
    if sample_id not in expected_sample_ids or sample_id not in world_by_id:
        raise ValueError(f"Worker {worker_index:02d} emitted an unassigned sample: {sample_id}.")
    manifest = world_by_id[sample_id]
    task_id = _integer(row["task_id"], label="task_id")
    episode_id = _integer(row["episode_id"], label="episode_id")
    trial = _integer(row["trial_index"], label="trial_index")
    if (
        task_id != int(manifest["task_id"])
        or episode_id != int(manifest["episode_id"])
        or trial != int(manifest["trial"])
    ):
        raise ValueError(f"Worker row manifest identity drifted for {sample_id}.")
    draw_id = _integer(row["draw_id"], label="draw_id")
    if draw_id not in range(WORLD_DRAWS_PER_SAMPLE):
        raise ValueError(f"Invalid draw_id for {sample_id}: {draw_id}.")
    condition = str(row["condition"])
    if condition not in conditions:
        raise ValueError(f"Unexpected {phase} condition: {condition}.")
    rank = _integer(row["rank"], label="rank")
    if rank != RANK_BY_CONDITION[condition]:
        raise ValueError(f"Frozen intervention rank drifted for {sample_id}/{condition}.")
    frozen_draw = stochastic_by_key[(sample_id, draw_id)]
    if str(row["video_noise_sha256"]) != str(frozen_draw["video_noise_sha256"]):
        raise ValueError(f"Frozen video noise drifted for {sample_id}/draw {draw_id}.")
    inference_steps = _integer(row["inference_steps"], label="inference_steps")
    inference_shift = _finite_number(row["inference_shift"], label="inference_shift")
    if (
        inference_steps != VIDEO_INFERENCE_STEPS
        or not math.isclose(
            inference_shift, VIDEO_INFERENCE_SHIFT, rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise ValueError(f"Native inference schedule drifted for {sample_id}/draw {draw_id}.")
    for key in (
        "world_manifest_sha256",
        "stochastic_manifest_sha256",
        "target_manifest_sha256",
        "machinery_sha256",
        "git_commit",
    ):
        if str(row[key]) != hashes[key]:
            raise ValueError(f"Worker row provenance drifted at {key}.")
    native = _finite_number(
        row["native_world_loss"], label="native_world_loss", nonnegative=True
    )
    latent = _finite_number(
        row["future_latent_mse"], label="future_latent_mse", nonnegative=True
    )
    if not math.isclose(native, latent, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("Native world loss must equal terminal future-latent MSE.")
    target_sha = str(row["target_sha256"])
    processed_hashes = target_by_id[sample_id].get("processed_tensor_sha256")
    expected_target_sha = (
        processed_hashes.get("video") if isinstance(processed_hashes, Mapping) else None
    )
    if (
        not _is_sha256(expected_target_sha)
        or target_sha != expected_target_sha
    ):
        raise ValueError(f"Frozen processed video target hash drifted for {sample_id}.")
    identity_hashes = {
        name: str(row[name])
        for name in (
            "target_latent_sha256",
            "current_image_sha256",
            "current_frame_latent_sha256",
            "donor_image_sha256",
        )
    }
    if any(not _is_sha256(value) for value in identity_hashes.values()):
        raise ValueError(f"World sample identity hash is malformed for {sample_id}.")
    if identity_hashes["current_image_sha256"] != str(
        target_by_id[sample_id]["current_image_sha256"]
    ):
        raise ValueError(f"Frozen current-image hash drifted for {sample_id}.")
    if identity_hashes["donor_image_sha256"] != str(
        manifest["donor_processed_image_sha256"]
    ):
        raise ValueError(f"Frozen donor-image hash drifted for {sample_id}.")
    if (
        row["prediction_shape"] != [1, 48, 2, 14, 28]
        or row["target_shape"] != [1, 48, 2, 14, 28]
        or row["future_token_shape"] != [1, 196, 3072]
    ):
        raise ValueError(f"Native world-output shape drifted for {sample_id}.")
    return {
        "phase": phase,
        "worker_index": worker_index,
        "sample_id": sample_id,
        "task_id": task_id,
        "episode_id": episode_id,
        "trial_index": trial,
        "draw_id": draw_id,
        "condition": condition,
        "rank": rank,
        "native_world_loss": native,
        "future_latent_mse": latent,
        "inference_steps": inference_steps,
        "inference_shift": inference_shift,
        "target_sha256": target_sha,
        **identity_hashes,
        "video_noise_sha256": str(row["video_noise_sha256"]),
        "prediction_shape": row["prediction_shape"],
        "target_shape": row["target_shape"],
        "future_token_shape": row["future_token_shape"],
        "world_manifest_sha256": str(row["world_manifest_sha256"]),
        "stochastic_manifest_sha256": str(row["stochastic_manifest_sha256"]),
        "target_manifest_sha256": str(row["target_manifest_sha256"]),
        "machinery_sha256": str(row["machinery_sha256"]),
        "git_commit": str(row["git_commit"]),
    }


def _pairing_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    ordered_ids: Sequence[str],
    conditions: tuple[str, ...],
) -> None:
    keys = [
        (str(row["sample_id"]), int(row["draw_id"]), str(row["condition"]))
        for row in rows
    ]
    expected = {
        (sample_id, draw_id, condition)
        for sample_id in ordered_ids
        for draw_id in range(WORLD_DRAWS_PER_SAMPLE)
        for condition in conditions
    }
    counts = Counter(keys)
    duplicates = [key for key, count in counts.items() if count != 1]
    if set(keys) != expected or duplicates:
        raise ValueError(
            "World shards are not exactly paired over sample × draw × condition; "
            f"missing={len(expected-set(keys))}, extra={len(set(keys)-expected)}, "
            f"duplicates={len(duplicates)}."
        )
    paired: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        paired[(str(row["sample_id"]), int(row["draw_id"]))].append(row)
    for key, group in paired.items():
        invariant_fields = (
            "task_id",
            "episode_id",
            "trial_index",
            "inference_steps",
            "inference_shift",
            "target_sha256",
            "target_latent_sha256",
            "current_image_sha256",
            "current_frame_latent_sha256",
            "donor_image_sha256",
            "video_noise_sha256",
        )
        for field in invariant_fields:
            values = {row[field] for row in group}
            if len(values) != 1:
                raise ValueError(f"Condition pairing drifted at {key}/{field}: {values}.")


def _sample_rows(
    rows: Sequence[Mapping[str, Any]], *, ordered_ids: Sequence[str], conditions: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sample_id"]), str(row["condition"]))].append(row)
    output: list[dict[str, Any]] = []
    for sample_id in ordered_ids:
        for condition in conditions:
            values = sorted(grouped[(sample_id, condition)], key=lambda row: int(row["draw_id"]))
            if len(values) != WORLD_DRAWS_PER_SAMPLE:
                raise ValueError(f"Incomplete sample-averaged cell: {sample_id}/{condition}.")
            for field in (
                "task_id",
                "episode_id",
                "trial_index",
                "rank",
                "inference_steps",
                "inference_shift",
                "target_sha256",
                "target_latent_sha256",
                "current_image_sha256",
                "current_frame_latent_sha256",
                "donor_image_sha256",
            ):
                if len({row[field] for row in values}) != 1:
                    raise ValueError(
                        f"Sample identity changed across draws: {sample_id}/{condition}/{field}."
                    )
            output.append(
                {
                    "sample_id": sample_id,
                    "task_id": int(values[0]["task_id"]),
                    "episode_id": int(values[0]["episode_id"]),
                    "trial_index": int(values[0]["trial_index"]),
                    "condition": condition,
                    "rank": int(values[0]["rank"]),
                    "draw_count": WORLD_DRAWS_PER_SAMPLE,
                    "mean_native_world_loss": float(
                        np.mean([float(row["native_world_loss"]) for row in values])
                    ),
                    "median_native_world_loss": float(
                        np.median([float(row["native_world_loss"]) for row in values])
                    ),
                    "mean_future_latent_mse": float(
                        np.mean([float(row["future_latent_mse"]) for row in values])
                    ),
                    "inference_steps": int(values[0]["inference_steps"]),
                    "inference_shift": float(values[0]["inference_shift"]),
                    "target_sha256": str(values[0]["target_sha256"]),
                    "target_latent_sha256": str(values[0]["target_latent_sha256"]),
                    "current_image_sha256": str(values[0]["current_image_sha256"]),
                    "current_frame_latent_sha256": str(
                        values[0]["current_frame_latent_sha256"]
                    ),
                    "donor_image_sha256": str(values[0]["donor_image_sha256"]),
                }
            )
    return output


def _sample_identity_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    ordered_ids: Sequence[str],
    conditions: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_sample[str(row["sample_id"])].append(row)
    fields = (
        "task_id",
        "episode_id",
        "trial_index",
        "target_sha256",
        "target_latent_sha256",
        "current_image_sha256",
        "current_frame_latent_sha256",
        "donor_image_sha256",
    )
    result: list[dict[str, Any]] = []
    for sample_id in ordered_ids:
        rows = by_sample[sample_id]
        if len(rows) != len(conditions):
            raise ValueError(f"Sample identity lacks all phase conditions: {sample_id}.")
        identity = {field: rows[0][field] for field in fields}
        for row in rows[1:]:
            if any(row[field] != identity[field] for field in fields):
                raise ValueError(f"Sample identity changed across conditions: {sample_id}.")
        result.append({"sample_id": sample_id, **identity})
    return result


def _condition_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    conditions: tuple[str, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    result = []
    for offset, condition in enumerate(conditions):
        values = np.asarray(
            [
                float(row["mean_native_world_loss"])
                for row in sample_rows
                if row["condition"] == condition
            ],
            dtype=np.float64,
        )
        low, high = paired_bootstrap_ci(
            values, samples=bootstrap_samples, seed=bootstrap_seed + offset
        )
        result.append(
            {
                "condition": condition,
                "samples": int(values.size),
                "draws_per_sample": WORLD_DRAWS_PER_SAMPLE,
                "mean_native_world_loss": float(values.mean()),
                "median_native_world_loss": float(np.median(values)),
                "mean_future_latent_mse": float(values.mean()),
                "paired_sample_bootstrap_mean_ci_low": low,
                "paired_sample_bootstrap_mean_ci_high": high,
            }
        )
    return result


def _endpoint_gate(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_cell = {
        (str(row["sample_id"]), str(row["condition"])): float(
            row["mean_native_world_loss"]
        )
        for row in sample_rows
    }
    current_rows = [row for row in sample_rows if row["condition"] == "current_all"]
    keys = [
        (int(row["task_id"]), int(row["episode_id"]), str(row["sample_id"]))
        for row in current_rows
    ]
    degradation = np.asarray(
        [
            by_cell[(str(row["sample_id"]), "wrong_all")]
            - by_cell[(str(row["sample_id"]), "current_all")]
            for row in current_rows
        ],
        dtype=np.float64,
    )
    paired_low, paired_high = paired_bootstrap_ci(
        degradation, samples=bootstrap_samples, seed=bootstrap_seed
    )
    hierarchical_low, hierarchical_high = task_hierarchical_bootstrap_ci(
        keys, degradation, samples=bootstrap_samples, seed=bootstrap_seed + 1
    )
    mean = float(degradation.mean())
    median = float(np.median(degradation))
    passed = mean > 0.0 and paired_low > 0.0
    return {
        "artifact_type": "asre_salvage_b_world_endpoint_gate",
        "schema_version": 2,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "classification": None if passed else "WORLD-ENDPOINT-UNINFORMATIVE",
        "loss_direction": "lower_is_better",
        "native_world_metric": NATIVE_WORLD_METRIC,
        "inference_steps": VIDEO_INFERENCE_STEPS,
        "inference_shift": VIDEO_INFERENCE_SHIFT,
        "degradation_definition": "native_loss(wrong_all)-native_loss(current_all)",
        "sample_count": int(degradation.size),
        "mean_wrong_minus_current": mean,
        "median_wrong_minus_current": median,
        "paired_bootstrap_ci_low": paired_low,
        "paired_bootstrap_ci_high": paired_high,
        "task_hierarchical_bootstrap_ci_low": hierarchical_low,
        "task_hierarchical_bootstrap_ci_high": hierarchical_high,
        "gate": {
            "mean_strictly_positive": mean > 0.0,
            "paired_ci_lower_strictly_positive": paired_low > 0.0,
            "median_descriptive_only": True,
            "hierarchical_ci_descriptive_only": True,
        },
    }


def aggregate_phase(
    *,
    phase: str,
    phase_root: Path,
    world_manifest_path: Path,
    stochastic_manifest_path: Path,
    target_manifest_path: Path,
    machinery_path: Path,
    preflight_path: Path,
    output_dir: Path,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if phase not in PHASE_CONDITIONS:
        raise ValueError(f"Unknown Salvage-B world phase: {phase}.")
    conditions = PHASE_CONDITIONS[phase]
    paths = [
        world_manifest_path,
        stochastic_manifest_path,
        target_manifest_path,
        machinery_path,
        preflight_path,
    ]
    if any(not Path(path).resolve().is_file() for path in paths):
        raise FileNotFoundError("A required Salvage-B aggregation input is absent.")
    phase_root = phase_root.resolve()
    output_dir = output_dir.resolve()
    world, stochastic, targets, _machinery, hashes = _validate_frozen_inputs(
        world_manifest_path=world_manifest_path.resolve(),
        stochastic_manifest_path=stochastic_manifest_path.resolve(),
        target_manifest_path=target_manifest_path.resolve(),
        machinery_path=machinery_path.resolve(),
        preflight_path=preflight_path.resolve(),
    )
    ordered_ids, world_by_id, stochastic_by_key, target_by_id = _manifest_indices(
        world, stochastic, targets
    )
    worker_dirs = sorted(
        path.name for path in phase_root.glob("worker_[0-9][0-9]") if path.is_dir()
    )
    expected_dirs = [f"worker_{index:02d}" for index in range(WORKER_COUNT)]
    if worker_dirs != expected_dirs:
        raise ValueError(
            f"{phase} must contain exactly four worker shards: {worker_dirs}."
        )
    launcher_config_path = phase_root / "launcher_config.json"
    if not launcher_config_path.is_file():
        raise FileNotFoundError(f"Missing immutable phase launcher config: {launcher_config_path}")
    launcher_config = _read_json(launcher_config_path)
    launcher_config_sha256 = sha256_file(launcher_config_path)
    canonical_launcher = dict(launcher_config)
    launcher_identity_sha256 = canonical_launcher.pop("identity_sha256", None)
    if (
        launcher_config.get("schema_version") != 2
        or launcher_config.get("phase") != phase
        or launcher_config.get("conditions") != list(conditions)
        or launcher_config.get("native_metric") != NATIVE_WORLD_METRIC
        or launcher_config.get("inference_steps") != VIDEO_INFERENCE_STEPS
        or launcher_config.get("inference_shift") != VIDEO_INFERENCE_SHIFT
        or launcher_config.get("world_manifest_sha256")
        != hashes["world_manifest_sha256"]
        or launcher_config.get("stochastic_manifest_sha256")
        != hashes["stochastic_manifest_sha256"]
        or launcher_config.get("target_manifest_sha256")
        != hashes["target_manifest_sha256"]
        or launcher_config.get("machinery_sha256") != hashes["machinery_sha256"]
        or launcher_config.get("git_commit_hash") != hashes["git_commit"]
        or launcher_identity_sha256 != sha256_json(canonical_launcher)
    ):
        raise ValueError("Immutable phase launcher config is incompatible.")

    all_rows: list[dict[str, Any]] = []
    shard_artifacts: list[dict[str, Any]] = []
    for worker_index in range(WORKER_COUNT):
        directory = phase_root / f"worker_{worker_index:02d}"
        metadata_path = directory / "metadata.json"
        rows_path = directory / "rows.jsonl"
        if not metadata_path.is_file() or not rows_path.is_file():
            raise FileNotFoundError(f"Incomplete shard directory: {directory}")
        metadata = _read_json(metadata_path)
        _validate_metadata(
            metadata=metadata,
            phase=phase,
            worker_index=worker_index,
            conditions=conditions,
            hashes=hashes,
        )
        raw_rows = _read_jsonl(rows_path)
        expected_sample_ids = {
            sample_id
            for position, sample_id in enumerate(ordered_ids)
            if position % WORKER_COUNT == worker_index
        }
        if len(expected_sample_ids) != SAMPLES_PER_WORKER:
            raise AssertionError("Frozen modulo-four assignment is imbalanced.")
        rows = [
            _validate_row(
                row=row,
                phase=phase,
                worker_index=worker_index,
                conditions=conditions,
                hashes=hashes,
                expected_sample_ids=expected_sample_ids,
                world_by_id=world_by_id,
                stochastic_by_key=stochastic_by_key,
                target_by_id=target_by_id,
            )
            for row in raw_rows
        ]
        if len(rows) != int(metadata["record_count"]):
            raise ValueError(f"Worker {worker_index:02d} row count disagrees with metadata.")
        if metadata.get("rows_sha256") != sha256_file(rows_path):
            raise ValueError(f"Worker {worker_index:02d} rows hash disagrees with metadata.")
        if metadata.get("sample_ids_sha256") != sha256_json(
            [
                sample_id
                for position, sample_id in enumerate(ordered_ids)
                if position % WORKER_COUNT == worker_index
            ]
        ):
            raise ValueError(f"Worker {worker_index:02d} sample-ID hash drifted.")
        launcher_hash = metadata.get("launcher_config_sha256")
        if launcher_hash != launcher_config_sha256:
            raise ValueError(f"Worker {worker_index:02d} launcher hash is mismatched.")
        endpoint_gate_sha = metadata.get("endpoint_gate_sha256")
        if (phase == "endpoint" and endpoint_gate_sha is not None) or (
            phase == "projected" and not _is_sha256(endpoint_gate_sha)
        ):
            raise ValueError(f"Worker {worker_index:02d} endpoint-gate binding drifted.")
        if {str(row["sample_id"]) for row in rows} != expected_sample_ids:
            raise ValueError(f"Worker {worker_index:02d} sample assignment is incomplete.")
        all_rows.extend(rows)
        shard_artifacts.append(
            {
                "worker_index": worker_index,
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "rows_path": str(rows_path),
                "rows_sha256": sha256_file(rows_path),
                "record_count": len(rows),
            }
        )
    expected_rows = SAMPLE_COUNT * WORLD_DRAWS_PER_SAMPLE * len(conditions)
    if len(all_rows) != expected_rows:
        raise ValueError(f"{phase} has {len(all_rows)} rows, expected {expected_rows}.")
    _pairing_audit(all_rows, ordered_ids=ordered_ids, conditions=conditions)
    all_rows.sort(
        key=lambda row: (
            ordered_ids.index(str(row["sample_id"])),
            conditions.index(str(row["condition"])),
            int(row["draw_id"]),
        )
    )
    sample_rows = _sample_rows(all_rows, ordered_ids=ordered_ids, conditions=conditions)
    sample_identity = _sample_identity_rows(
        sample_rows,
        ordered_ids=ordered_ids,
        conditions=conditions,
    )
    sample_identity_sha256 = sha256_json(sample_identity)
    condition_rows = _condition_rows(
        sample_rows,
        conditions=conditions,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_csv = output_dir / "world_draw_losses.csv"
    sample_csv = output_dir / "world_sample_losses.csv"
    sample_identity_csv = output_dir / "world_sample_identity.csv"
    condition_csv = output_dir / "world_condition_summary.csv"
    _write_csv(draw_csv, all_rows)
    _write_csv(sample_csv, sample_rows)
    _write_csv(sample_identity_csv, sample_identity)
    _write_csv(condition_csv, condition_rows)
    frozen_paths = {
        "world_manifest_sha256": world_manifest_path.resolve(),
        "stochastic_manifest_sha256": stochastic_manifest_path.resolve(),
        "target_manifest_sha256": target_manifest_path.resolve(),
        "machinery_sha256": machinery_path.resolve(),
        "preflight_report_sha256": preflight_path.resolve(),
    }
    for sha_name, path in frozen_paths.items():
        if sha256_file(path) != hashes[sha_name]:
            raise ValueError(f"Frozen aggregation input changed during merge: {path}.")
    endpoint = None
    if phase == "endpoint":
        endpoint = _endpoint_gate(
            sample_rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        endpoint.update(
            {
                "world_manifest_sha256": hashes["world_manifest_sha256"],
                "stochastic_manifest_sha256": hashes["stochastic_manifest_sha256"],
                "target_manifest_sha256": hashes["target_manifest_sha256"],
                "machinery_sha256": hashes["machinery_sha256"],
                "git_commit_hash": hashes["git_commit"],
                "endpoint_shards": shard_artifacts,
                "draw_rows_path": str(draw_csv),
                "draw_rows_sha256": sha256_file(draw_csv),
                "sample_rows_path": str(sample_csv),
                "sample_rows_sha256": sha256_file(sample_csv),
                "sample_identity": sample_identity,
                "sample_identity_sha256": sample_identity_sha256,
                "sample_identity_rows_path": str(sample_identity_csv),
                "sample_identity_rows_sha256": sha256_file(sample_identity_csv),
            }
        )
        atomic_write_json(output_dir / "world_endpoint_gate.json", endpoint)
    passed = endpoint is None or bool(endpoint["passed"])
    summary = {
        "artifact_type": "asre_salvage_b_world_phase_aggregate",
        "schema_version": 2,
        "protocol": SALVAGE_B_PROTOCOL,
        "phase": phase,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "classification": None if passed else "WORLD-ENDPOINT-UNINFORMATIVE",
        "created_at": now_iso(),
        "conditions": list(conditions),
        "sample_count": SAMPLE_COUNT,
        "draws_per_sample": WORLD_DRAWS_PER_SAMPLE,
        "draw_record_count": len(all_rows),
        "pairing_complete": True,
        "native_world_metric": NATIVE_WORLD_METRIC,
        "metric_direction": "lower_is_better",
        "inference_steps": VIDEO_INFERENCE_STEPS,
        "inference_shift": VIDEO_INFERENCE_SHIFT,
        "initial_state": "pure_gaussian_future_latent_noise",
        "target_usage": "scoring_only_after_inference",
        "worker_count": WORKER_COUNT,
        "launcher_config_path": str(launcher_config_path),
        "launcher_config_sha256": launcher_config_sha256,
        "endpoint_gate_sha256": launcher_config.get("endpoint_gate_sha256"),
        "world_manifest_sha256": hashes["world_manifest_sha256"],
        "stochastic_manifest_sha256": hashes["stochastic_manifest_sha256"],
        "target_manifest_sha256": hashes["target_manifest_sha256"],
        "machinery_sha256": hashes["machinery_sha256"],
        "preflight_report_sha256": hashes["preflight_report_sha256"],
        "git_commit_hash": hashes["git_commit"],
        "shards": shard_artifacts,
        "draw_rows_path": str(draw_csv),
        "draw_rows_sha256": sha256_file(draw_csv),
        "sample_rows_path": str(sample_csv),
        "sample_rows_sha256": sha256_file(sample_csv),
        "sample_identity": sample_identity,
        "sample_identity_sha256": sample_identity_sha256,
        "sample_identity_rows_path": str(sample_identity_csv),
        "sample_identity_rows_sha256": sha256_file(sample_identity_csv),
        "condition_summary_path": str(condition_csv),
        "condition_summary_sha256": sha256_file(condition_csv),
        "sample_results": sample_rows,
        "condition_summary": condition_rows,
        "endpoint_gate": endpoint,
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "unit": "paired evaluation sample after averaging four frozen draws",
            "task_hierarchy": "task then episode then evaluation sample",
        },
        "later_stage_launched": False,
    }
    summary_path = output_dir / f"{phase}_world_summary.json"
    atomic_write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASE_CONDITIONS), required=True)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--world-manifest", type=Path, required=True)
    parser.add_argument("--stochastic-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--machinery", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    summary = aggregate_phase(
        phase=args.phase,
        phase_root=args.phase_root,
        world_manifest_path=args.world_manifest,
        stochastic_manifest_path=args.stochastic_manifest,
        target_manifest_path=args.target_manifest,
        machinery_path=args.machinery,
        preflight_path=args.preflight,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        f"Salvage-B {args.phase} world aggregate: {summary['status']} "
        f"({args.output_dir.resolve()})"
    )


if __name__ == "__main__":
    main()
