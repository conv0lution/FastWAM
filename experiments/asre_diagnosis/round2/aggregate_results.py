"""Strict aggregation for the pre-registered ASRE Round-2 experiment.

This module intentionally does not fall back to intersections of sample or
episode identifiers.  A condition is comparable only when all eight online
runs contain the same 100 paired episodes and all eight offline replays contain
the same validated 499 state-bank samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND2_PROTOCOL,
    atomic_write_json,
    build_conditions,
    build_round2_conditions,
    git_commit,
    now_iso,
)
from experiments.asre_diagnosis.round2.metrics import (  # noqa: E402
    ACTION_DIMENSION_LABEL_SOURCE,
    CONTINUOUS_ACTION_DIMENSIONS,
)


NUM_LAYERS = 30
BASELINE = "baseline_round2"
ACTION_DIMENSION_NAMES = CONTINUOUS_ACTION_DIMENSIONS
SCALAR_OFFLINE_METRICS = (
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
DIMENSION_OFFLINE_METRIC = "executed_prefix_norm_rms_by_dimension"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly aggregate all eight ASRE Round-2 conditions."
    )
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--round1-summary",
        type=Path,
        help=(
            "Optional Round-1 aggregate/summary.csv. When supplied, aggregation "
            "also writes a joint Round-1/Round-2 table and baseline-repeat check."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--expected-online-episodes", type=int, default=100)
    parser.add_argument(
        "--expected-offline-samples",
        type=int,
        default=None,
        help=(
            "Optional assertion. The authoritative sample count and ordering are "
            "always read from the immutable QC valid manifest."
        ),
    )
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--expected-trials-per-task", type=int, default=10)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--candidate-max-loss", type=float, default=0.05)
    parser.add_argument("--catastrophic-baseline-rate", type=float, default=0.80)
    parser.add_argument("--catastrophic-task-delta", type=float, default=-0.50)
    parser.add_argument("--strong-degradation-loss", type=float, default=0.20)
    return parser.parse_args()


def _condition_definitions() -> tuple[list[str], dict[str, dict[str, Any]]]:
    conditions = build_round2_conditions(NUM_LAYERS)
    order = [condition.name for condition in conditions]
    definitions: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        enabled = list(condition.enabled_video_retrieval_layers(NUM_LAYERS))
        disabled = list(condition.disabled_video_layers)
        definitions[condition.name] = {
            "enabled_video_retrieval_layers": enabled,
            "disabled_video_layers": disabled,
            "enabled_layer_ranges": _format_layer_ranges(enabled),
            "num_retrieval_layers": len(enabled),
        }
    return order, definitions


def _load_round1_summary(path: Path) -> list[dict[str, Any]]:
    """Load the completed Round-1 table used for joint mechanistic interpretation."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Round-1 aggregate summary is unavailable: {resolved}")
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    expected_conditions = build_conditions(NUM_LAYERS)
    expected_order = [condition.name for condition in expected_conditions]
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        condition = str(row.get("condition", ""))
        if not condition or condition in by_name:
            raise ValueError(f"Invalid or duplicate Round-1 condition {condition!r} in {resolved}.")
        by_name[condition] = row
    if set(by_name) != set(expected_order):
        raise ValueError(
            "Round-1 summary conditions are incomplete: "
            f"missing={sorted(set(expected_order) - set(by_name))}, "
            f"unexpected={sorted(set(by_name) - set(expected_order))}."
        )
    validated: list[dict[str, Any]] = []
    for condition in expected_conditions:
        row = by_name[condition.name]
        try:
            disabled = json.loads(str(row["disabled_video_layers"]))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid disabled_video_layers for {condition.name} in {resolved}."
            ) from exc
        if [int(value) for value in disabled] != list(condition.disabled_video_layers):
            raise ValueError(f"Round-1 disabled-layer mismatch for {condition.name}.")
        validated.append(
            {
                **row,
                "online_success_rate": _finite_float(
                    row.get("online_success_rate"),
                    context=f"Round-1 {condition.name}.online_success_rate",
                ),
                "delta_success_rate": _finite_float(
                    row.get("delta_success_rate"),
                    context=f"Round-1 {condition.name}.delta_success_rate",
                ),
                "offline_normalized_l2": _finite_float(
                    row.get("offline_normalized_l2"),
                    context=f"Round-1 {condition.name}.offline_normalized_l2",
                ),
                "disabled_video_layers": list(condition.disabled_video_layers),
                "enabled_video_retrieval_layers": list(
                    condition.enabled_video_retrieval_layers(NUM_LAYERS)
                ),
            }
        )
    return validated


def _build_joint_round_table(
    round1_rows: Sequence[Mapping[str, Any]],
    round2_rows: Sequence[Mapping[str, Any]],
    *,
    round2_offline_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in round1_rows:
        enabled = [int(value) for value in record["enabled_video_retrieval_layers"]]
        rows.append(
            {
                "round": "round1_necessity",
                "condition": record["condition"],
                "enabled_layer_ranges": _format_layer_ranges(enabled),
                "num_retrieval_layers": len(enabled),
                "online_success_rate": record["online_success_rate"],
                "delta_success_rate_within_round": record["delta_success_rate"],
                "round1_raw_output_full_chunk_rms": record["offline_normalized_l2"],
                "offline_sample_population": 500,
                "offline_population_note": "original Round-1 bank",
            }
        )
    for record in round2_rows:
        rows.append(
            {
                "round": "round2_sufficiency",
                "condition": record["condition"],
                "enabled_layer_ranges": record["enabled_layer_ranges"],
                "num_retrieval_layers": record["num_retrieval_layers"],
                "online_success_rate": record["online_success_rate"],
                "delta_success_rate_within_round": record["delta_success_rate"],
                "round1_raw_output_full_chunk_rms": record[
                    "round1_raw_output_full_chunk_rms"
                ],
                "offline_sample_population": round2_offline_samples,
                "offline_population_note": "fixed QC-valid Round-1 subset",
            }
        )
    return rows


def _format_layer_ranges(layers: Sequence[int]) -> str:
    if not layers:
        return "none"
    ranges: list[str] = []
    start = previous = int(layers[0])
    for layer_value in layers[1:]:
        layer = int(layer_value)
        if layer == previous + 1:
            previous = layer
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = layer
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer columns for empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            records.append(value)
    return records


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_set(paths: Sequence[Path], *, relative_to: Path) -> str:
    """Hash both relative names and bytes for a deterministic set of artifacts."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.resolve().relative_to(relative_to.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _finite_float(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric {context}, got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"Expected finite {context}, got {result!r}.")
    return result


def _stable_seed(base_seed: int, *parts: object) -> int:
    label = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _percentile_interval(estimates: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Paired bootstrap requires at least one pair.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    return _percentile_interval(values[indices].mean(axis=1))


def task_hierarchical_bootstrap_ci(
    keys: Sequence[tuple[str, int, int]],
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if len(keys) != values.size or values.size == 0:
        raise ValueError("Task-hierarchical bootstrap requires one value per nonempty key.")
    grouped_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (suite, task_id, _episode_id) in enumerate(keys):
        grouped_indices[(suite, int(task_id))].append(index)
    groups = [np.asarray(grouped_indices[key], dtype=int) for key in sorted(grouped_indices)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for bootstrap_index in range(samples):
        selected_groups = rng.integers(0, len(groups), size=len(groups))
        task_means = np.empty(len(groups), dtype=np.float64)
        for output_index, selected_group in enumerate(selected_groups):
            member_indices = groups[int(selected_group)]
            selected_members = member_indices[
                rng.integers(0, member_indices.size, size=member_indices.size)
            ]
            task_means[output_index] = float(values[selected_members].mean())
        estimates[bootstrap_index] = float(task_means.mean())
    return _percentile_interval(estimates)


def episode_cluster_bootstrap_ci(
    values: Sequence[float],
    cluster_ids: Sequence[tuple[str, int, int]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(cluster_ids) != numeric.size or numeric.size == 0:
        raise ValueError("Cluster bootstrap requires one cluster ID per nonempty value.")
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for cluster, value in zip(cluster_ids, numeric):
        grouped[cluster].append(float(value))
    ordered = [np.asarray(grouped[key], dtype=np.float64) for key in sorted(grouped)]
    cluster_sums = np.asarray([group.sum() for group in ordered], dtype=np.float64)
    cluster_counts = np.asarray([group.size for group in ordered], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ordered), size=(samples, len(ordered)))
    estimates = cluster_sums[indices].sum(axis=1) / cluster_counts[indices].sum(axis=1)
    return _percentile_interval(estimates)


def exact_mcnemar_p_value(induced_failures: int, rescued_successes: int) -> float:
    b = int(induced_failures)
    c = int(rescued_successes)
    if b < 0 or c < 0:
        raise ValueError("McNemar transition counts must be nonnegative.")
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(b, c) + 1))
    return float(min(1.0, 2.0 * tail / (2**discordant)))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted((float(value), key) for key, value in p_values.items())
    adjusted: dict[str, float] = {}
    running_max = 0.0
    number = len(ordered)
    for rank, (value, key) in enumerate(ordered):
        running_max = max(running_max, (number - rank) * value)
        adjusted[key] = float(min(1.0, running_max))
    return adjusted


def _discover_condition_metadata(
    online_root: Path,
    expected_order: Sequence[str],
    definitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    for path in sorted(online_root.glob("*/run_metadata.json")):
        metadata = _read_json(path)
        condition = str(metadata.get("diagnosis_condition", ""))
        if not condition:
            raise ValueError(f"Missing diagnosis_condition in {path}.")
        if condition in discovered:
            raise ValueError(f"Duplicate online metadata for {condition!r}.")
        metadata["_condition_dir"] = str(path.parent.resolve())
        metadata["_metadata_path"] = str(path.resolve())
        discovered[condition] = metadata
    expected = set(expected_order)
    actual = set(discovered)
    if actual != expected:
        raise ValueError(
            "Online conditions do not exactly match Round 2: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )
    for condition in expected_order:
        metadata = discovered[condition]
        expected_disabled = list(definitions[condition]["disabled_video_layers"])
        actual_disabled = [int(value) for value in metadata.get("disabled_video_layers", [])]
        if actual_disabled != expected_disabled:
            raise ValueError(
                f"Online metadata for {condition} has disabled layers {actual_disabled}, "
                f"expected {expected_disabled}."
            )
        if "enabled_video_retrieval_layers" in metadata:
            expected_enabled = list(definitions[condition]["enabled_video_retrieval_layers"])
            actual_enabled = [
                int(value) for value in metadata["enabled_video_retrieval_layers"]
            ]
            if actual_enabled != expected_enabled:
                raise ValueError(
                    f"Online metadata for {condition} has enabled layers {actual_enabled}, "
                    f"expected {expected_enabled}."
                )
    return discovered


def _validate_online_metadata_compatibility(
    metadata: Mapping[str, Mapping[str, Any]],
    condition_order: Sequence[str],
) -> None:
    reference = metadata[BASELINE]
    compatibility_fields = (
        "git_commit_hash",
        "checkpoint_path",
        "checkpoint_sha256",
        "dataset_stats_path",
        "dataset_stats_sha256",
        "state_bank_manifest_path",
        "state_bank_manifest_sha256",
        "valid_state_bank_manifest_path",
        "valid_state_bank_manifest_sha256",
        "prompt_context_cache_path",
        "prompt_context_cache_sha256",
        "task_suite",
        "task_ids",
        "seed",
        "number_of_trials",
        "action_horizon",
        "number_of_inference_steps",
        "replan_steps",
        "compile_action_infer",
        "binarize_gripper",
        "sigma_shift",
        "rand_device",
        "text_conditioning_source",
        "prompt_template",
        "environment_seed",
        "action_inference_seed",
        "action_noise_seed",
        "gpu_model",
        "torch_version",
        "cuda_version",
    )
    required_fields = set(compatibility_fields) | {
        "condition_protocol",
        "enabled_video_retrieval_layers",
        "disabled_video_layers",
        "condition_config",
        "config_sha256",
        "start_timestamp",
        "end_timestamp",
        "status",
    }
    # `sigma_shift=None` is the official Fast-WAM configuration and is a
    # meaningful scheduler setting, not missing provenance.  The key must
    # still be present, and compatibility below requires the same value for
    # every condition.
    nullable_required_fields = {"sigma_shift"}
    for condition in condition_order:
        record = metadata[condition]
        missing = sorted(
            key
            for key in required_fields
            if key not in record
            or (record[key] is None and key not in nullable_required_fields)
        )
        if missing:
            raise ValueError(f"Online metadata for {condition} is missing {missing}.")
        if record["condition_protocol"] != ROUND2_PROTOCOL:
            raise ValueError(
                f"Online metadata for {condition} has protocol "
                f"{record['condition_protocol']!r}, expected {ROUND2_PROTOCOL!r}."
            )
        if record["status"] != "completed" or not str(record["end_timestamp"]).strip():
            raise ValueError(f"Online condition {condition} is not completed.")
        config_sha256 = str(record["config_sha256"])
        if re.fullmatch(r"[0-9a-f]{64}", config_sha256.lower()) is None:
            raise ValueError(f"Online metadata for {condition} has invalid config_sha256.")
    for condition in condition_order[1:]:
        record = metadata[condition]
        mismatches = {
            key: {"baseline": reference.get(key), "condition": record.get(key)}
            for key in compatibility_fields
            if reference.get(key) != record.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Online metadata for {condition} is incompatible with {BASELINE}: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )


def _load_authoritative_valid_sample_ids(
    online_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], Path, str]:
    """Load and authenticate the QC population referenced by the online run."""

    reference = online_metadata[BASELINE]
    path_value = reference.get("valid_state_bank_manifest_path")
    digest_value = str(reference.get("valid_state_bank_manifest_sha256", "")).lower()
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Baseline metadata has no valid_state_bank_manifest_path.")
    if re.fullmatch(r"[0-9a-f]{64}", digest_value) is None:
        raise ValueError("Baseline metadata has no valid valid-state manifest SHA256.")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"The immutable QC valid manifest is unavailable: {path}")
    observed_digest = _sha256_path(path)
    if observed_digest != digest_value:
        raise ValueError(
            f"QC valid-manifest SHA256 mismatch: expected {digest_value}, "
            f"observed {observed_digest}."
        )
    payload = _read_json(path)
    values = payload.get("valid_sample_ids")
    if not isinstance(values, list) or not values:
        raise ValueError(f"QC valid manifest has no valid_sample_ids: {path}")
    sample_ids = [str(value) for value in values]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"QC valid manifest has empty or duplicate sample IDs: {path}")
    return sample_ids, path, digest_value


def _load_expected_action_global_std(
    online_metadata: Mapping[str, Mapping[str, Any]],
) -> list[float]:
    reference = online_metadata[BASELINE]
    path = Path(str(reference["dataset_stats_path"])).resolve()
    expected_digest = str(reference["dataset_stats_sha256"]).lower()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset-stats artifact is unavailable: {path}")
    observed_digest = _sha256_path(path)
    if observed_digest != expected_digest:
        raise ValueError(
            f"Dataset-stats SHA256 mismatch: expected {expected_digest}, "
            f"observed {observed_digest}."
        )
    payload = _read_json(path)
    try:
        values = payload["action"]["default"]["global_std"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Dataset stats lack action.default.global_std: {path}") from exc
    numeric = np.asarray(values, dtype=np.float32).astype(np.float64)
    if numeric.ndim != 1 or numeric.size < 7 or not np.all(np.isfinite(numeric)):
        raise ValueError(f"Invalid action.default.global_std in {path}.")
    return numeric.astype(float).tolist()


EpisodeKey = tuple[str, int, int]


def _load_online_outcomes(
    metadata: Mapping[str, Mapping[str, Any]],
    condition_order: Sequence[str],
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    expected_episodes: int,
    expected_tasks: int,
    expected_trials_per_task: int,
    expected_suite: str,
) -> tuple[
    dict[str, dict[EpisodeKey, int]],
    dict[str, dict[int, list[int]]],
    dict[int, str],
]:
    outcomes: dict[str, dict[EpisodeKey, int]] = {}
    task_outcomes: dict[str, dict[int, list[int]]] = {}
    baseline_descriptions: dict[int, str] = {}
    for condition in condition_order:
        condition_dir = Path(str(metadata[condition]["_condition_dir"]))
        result_paths = sorted(condition_dir.glob("**/gpu*_task*_results.json"))
        if len(result_paths) != expected_tasks:
            raise ValueError(
                f"{condition} has {len(result_paths)} task result files; expected {expected_tasks}."
            )
        condition_outcomes: dict[EpisodeKey, int] = {}
        by_task: dict[int, list[int]] = {}
        seen_tasks: set[int] = set()
        for path in result_paths:
            result = _read_json(path)
            suite = str(result.get("task_suite", ""))
            if suite != expected_suite:
                raise ValueError(f"Unexpected task suite {suite!r} in {path}; expected {expected_suite!r}.")
            task_id = int(result["task_id"])
            if task_id in seen_tasks:
                raise ValueError(f"Duplicate task {task_id} for {condition}: {path}.")
            seen_tasks.add(task_id)
            total = int(result["total_episodes"])
            if total != expected_trials_per_task:
                raise ValueError(
                    f"{path} has {total} episodes; expected {expected_trials_per_task}."
                )
            successes = {int(value) for value in result.get("success_episodes", [])}
            failures = {int(value) for value in result.get("failure_episodes", [])}
            expected_ids = set(range(total))
            if successes & failures or successes | failures != expected_ids:
                raise ValueError(f"Incomplete or inconsistent episode IDs in {path}.")
            listed_successes = int(result.get("successes", len(successes)))
            if listed_successes != len(successes):
                raise ValueError(f"Success count disagrees with success_episodes in {path}.")
            diagnosis_condition = result.get("diagnosis_condition")
            if diagnosis_condition is not None and str(diagnosis_condition) != condition:
                raise ValueError(f"Condition label mismatch in {path}: {diagnosis_condition!r}.")
            if "disabled_video_layers" in result:
                actual_disabled = [int(value) for value in result["disabled_video_layers"]]
                if actual_disabled != list(definitions[condition]["disabled_video_layers"]):
                    raise ValueError(f"Disabled layer mismatch in {path}.")
            description = str(result.get("task_description", "")).strip()
            if not description:
                raise ValueError(f"Missing task_description in {path}.")
            if condition == BASELINE:
                baseline_descriptions[task_id] = description
            elif task_id in baseline_descriptions and baseline_descriptions[task_id] != description:
                raise ValueError(f"Task description mismatch for task {task_id} in {path}.")
            task_values = [int(episode_id in successes) for episode_id in range(total)]
            by_task[task_id] = task_values
            for episode_id, value in enumerate(task_values):
                key = (suite, task_id, episode_id)
                if key in condition_outcomes:
                    raise ValueError(f"Duplicate online episode key {key} for {condition}.")
                condition_outcomes[key] = value
        if seen_tasks != set(range(expected_tasks)):
            raise ValueError(
                f"{condition} task IDs are {sorted(seen_tasks)}, expected 0..{expected_tasks - 1}."
            )
        if len(condition_outcomes) != expected_episodes:
            raise ValueError(
                f"{condition} has {len(condition_outcomes)} online episodes; "
                f"expected {expected_episodes}."
            )
        outcomes[condition] = condition_outcomes
        task_outcomes[condition] = by_task

    baseline_keys = set(outcomes[BASELINE])
    for condition in condition_order[1:]:
        condition_keys = set(outcomes[condition])
        if condition_keys != baseline_keys:
            raise ValueError(
                f"Online episode keys for {condition} do not exactly match {BASELINE}: "
                f"missing={sorted(baseline_keys - condition_keys)}, "
                f"extra={sorted(condition_keys - baseline_keys)}."
            )
    for condition in condition_order[1:]:
        for task_id, description in baseline_descriptions.items():
            # Description equality was checked when baseline was discovered first in fixed order.
            if task_id not in task_outcomes[condition]:
                raise AssertionError(f"Missing validated task {task_id} for {condition}.")
    return outcomes, task_outcomes, baseline_descriptions


def _parse_dimension_vector(value: Any, *, context: str) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON dimension vector for {context}.") from exc
    if not isinstance(value, list) or len(value) != len(ACTION_DIMENSION_NAMES):
        raise ValueError(
            f"{context} must contain {len(ACTION_DIMENSION_NAMES)} values, got {value!r}."
        )
    return [_finite_float(item, context=f"{context}[{index}]") for index, item in enumerate(value)]


def _load_offline_records(
    offline_root: Path,
    condition_order: Sequence[str],
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    expected_sample_ids: Sequence[str],
    valid_manifest_path: Path,
    valid_manifest_sha256: str,
    online_metadata: Mapping[str, Mapping[str, Any]],
    expected_action_global_std: Sequence[float],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    expected_ids = [str(identifier) for identifier in expected_sample_ids]
    expected_samples = len(expected_ids)
    unexpected = {
        path.parent.name
        for path in offline_root.glob("*/per_sample.jsonl")
        if path.parent.name not in set(condition_order)
    }
    if unexpected:
        raise ValueError(f"Unexpected offline conditions: {sorted(unexpected)}.")
    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    baseline_ids: set[str] | None = None
    for condition in condition_order:
        condition_dir = offline_root / condition
        per_sample_path = condition_dir / "per_sample.jsonl"
        summary_path = condition_dir / "summary.csv"
        metadata_path = condition_dir / "run_metadata.json"
        if (
            not per_sample_path.is_file()
            or not summary_path.is_file()
            or not metadata_path.is_file()
        ):
            raise FileNotFoundError(
                f"Offline condition {condition} requires {per_sample_path}, "
                f"{summary_path}, and {metadata_path}."
            )
        run_metadata = _read_json(metadata_path)
        expected_metadata = {
            "status": "complete",
            "condition_protocol": ROUND2_PROTOCOL,
            "diagnosis_condition": condition,
            "enabled_video_retrieval_layers": definitions[condition][
                "enabled_video_retrieval_layers"
            ],
            "disabled_video_layers": definitions[condition]["disabled_video_layers"],
            "valid_state_bank_manifest_path": str(valid_manifest_path),
            "valid_state_bank_manifest_sha256": valid_manifest_sha256,
            "num_valid_samples": expected_samples,
            "num_samples": expected_samples,
        }
        shared_provenance_fields = (
            "git_commit_hash",
            "checkpoint_path",
            "checkpoint_sha256",
            "dataset_stats_path",
            "dataset_stats_sha256",
            "state_bank_manifest_path",
            "state_bank_manifest_sha256",
            "valid_state_bank_manifest_path",
            "valid_state_bank_manifest_sha256",
            "prompt_context_cache_path",
            "prompt_context_cache_sha256",
            "action_horizon",
            "number_of_inference_steps",
            "replan_steps",
            "num_model_layers",
            "task_suite",
            "task_ids",
            "seed",
            "number_of_trials",
            "compile_action_infer",
            "binarize_gripper",
            "sigma_shift",
            "rand_device",
            "torch_version",
            "cuda_version",
        )
        expected_metadata.update(
            {
                key: online_metadata[condition].get(key)
                for key in shared_provenance_fields
            }
        )
        metadata_mismatches = {
            key: {"observed": run_metadata.get(key), "expected": value}
            for key, value in expected_metadata.items()
            if run_metadata.get(key) != value
        }
        if metadata_mismatches:
            raise ValueError(
                f"Offline metadata for {condition} is incompatible: "
                f"{json.dumps(metadata_mismatches, sort_keys=True)}"
            )
        observed_action_std = np.asarray(
            run_metadata.get("action_global_std"), dtype=np.float64
        )
        if not np.array_equal(
            observed_action_std,
            np.asarray(expected_action_global_std, dtype=np.float64),
        ):
            raise ValueError(
                f"Offline action_global_std for {condition} disagrees with the "
                "authenticated dataset-stats artifact."
            )
        records = _read_jsonl(per_sample_path)
        if len(records) != expected_samples:
            raise ValueError(
                f"{condition} has {len(records)} offline samples; expected {expected_samples}."
            )
        ids: list[str] = []
        for index, record in enumerate(records):
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"Missing sample_id in {per_sample_path} record {index}.")
            ids.append(sample_id)
            if str(record.get("condition", "")) != condition:
                raise ValueError(f"Condition label mismatch for {sample_id} in {per_sample_path}.")
            if "disabled_video_layers" in record:
                actual_disabled = [int(value) for value in record["disabled_video_layers"]]
                if actual_disabled != list(definitions[condition]["disabled_video_layers"]):
                    raise ValueError(
                        f"Disabled layer mismatch for {sample_id} in {per_sample_path}."
                    )
            if "enabled_video_retrieval_layers" in record:
                actual_enabled = [
                    int(value) for value in record["enabled_video_retrieval_layers"]
                ]
                if actual_enabled != list(
                    definitions[condition]["enabled_video_retrieval_layers"]
                ):
                    raise ValueError(
                        f"Enabled layer mismatch for {sample_id} in {per_sample_path}."
                    )
            for identity_key in ("task_suite", "task_id", "episode_id", "replan_id"):
                if identity_key not in record:
                    raise ValueError(f"Missing {identity_key} for {sample_id} in {per_sample_path}.")
            if int(record.get("executed_prefix_length", -1)) != 10:
                raise ValueError(
                    f"Invalid executed_prefix_length for {sample_id}: "
                    f"{record.get('executed_prefix_length')!r}."
                )
            if record.get("action_dimension_names") != list(ACTION_DIMENSION_NAMES):
                raise ValueError(f"Action-dimension labels mismatch for {sample_id}.")
            for metric in SCALAR_OFFLINE_METRICS:
                record[metric] = _finite_float(record.get(metric), context=f"{sample_id}.{metric}")
            record[DIMENSION_OFFLINE_METRIC] = _parse_dimension_vector(
                record.get(DIMENSION_OFFLINE_METRIC),
                context=f"{sample_id}.{DIMENSION_OFFLINE_METRIC}",
            )
        if len(set(ids)) != len(ids):
            duplicate_ids = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
            raise ValueError(f"Duplicate offline sample IDs for {condition}: {duplicate_ids}.")
        if ids != expected_ids:
            raise ValueError(
                f"Offline sample ordering for {condition} does not exactly match the "
                "immutable QC valid manifest."
            )
        condition_ids = set(ids)
        if baseline_ids is None:
            baseline_ids = condition_ids
        elif condition_ids != baseline_ids:
            raise ValueError(
                f"Offline sample IDs for {condition} do not exactly match {BASELINE}: "
                f"missing={sorted(baseline_ids - condition_ids)}, "
                f"extra={sorted(condition_ids - baseline_ids)}."
            )
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
        if len(summary_rows) != 1:
            raise ValueError(f"Expected exactly one summary row in {summary_path}.")
        summary = dict(summary_rows[0])
        if str(summary.get("condition", "")) != condition:
            raise ValueError(f"Summary condition mismatch in {summary_path}.")
        if int(summary.get("num_samples", -1)) != expected_samples:
            raise ValueError(f"Summary sample count mismatch in {summary_path}.")
        if int(summary.get("executed_prefix_length", -1)) != 10:
            raise ValueError(f"Summary executed-prefix length mismatch in {summary_path}.")
        for metric in SCALAR_OFFLINE_METRICS:
            reported = _finite_float(summary.get(metric), context=f"{summary_path}.{metric}")
            recomputed = float(np.mean([record[metric] for record in records]))
            if not math.isclose(reported, recomputed, rel_tol=1e-7, abs_tol=1e-12):
                raise ValueError(
                    f"Offline summary mismatch for {condition}.{metric}: "
                    f"reported={reported}, recomputed={recomputed}."
                )
            summary[metric] = reported
        reported_dimensions = _parse_dimension_vector(
            summary.get(DIMENSION_OFFLINE_METRIC),
            context=f"{summary_path}.{DIMENSION_OFFLINE_METRIC}",
        )
        recomputed_dimensions = np.mean(
            np.asarray([record[DIMENSION_OFFLINE_METRIC] for record in records], dtype=float),
            axis=0,
        )
        if not np.allclose(reported_dimensions, recomputed_dimensions, rtol=1e-7, atol=1e-12):
            raise ValueError(f"Offline summary dimension vector mismatch for {condition}.")
        summary[DIMENSION_OFFLINE_METRIC] = [float(value) for value in reported_dimensions]
        records_by_condition[condition] = records
        summaries[condition] = summary
    return records_by_condition, summaries


def _comparison_statistics(
    reference: Mapping[EpisodeKey, int],
    target: Mapping[EpisodeKey, int],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    seed_label: str,
) -> dict[str, Any]:
    if set(reference) != set(target):
        raise ValueError(f"Comparison {seed_label} received nonidentical paired episode keys.")
    keys = sorted(reference)
    reference_values = np.asarray([reference[key] for key in keys], dtype=int)
    target_values = np.asarray([target[key] for key in keys], dtype=int)
    differences = target_values.astype(float) - reference_values.astype(float)
    paired_low, paired_high = paired_bootstrap_ci(
        differences,
        samples=bootstrap_samples,
        seed=_stable_seed(bootstrap_seed, seed_label, "paired"),
    )
    hierarchical_low, hierarchical_high = task_hierarchical_bootstrap_ci(
        keys,
        differences,
        samples=bootstrap_samples,
        seed=_stable_seed(bootstrap_seed, seed_label, "hierarchical"),
    )
    transitions = Counter(
        (int(reference_value), int(target_value))
        for reference_value, target_value in zip(reference_values, target_values)
    )
    induced = transitions[(1, 0)]
    rescued = transitions[(0, 1)]
    return {
        "reference_success_rate": float(reference_values.mean()),
        "target_success_rate": float(target_values.mean()),
        "paired_delta_success_rate": float(differences.mean()),
        "paired_delta_ci_low": paired_low,
        "paired_delta_ci_high": paired_high,
        "task_hierarchical_delta_ci_low": hierarchical_low,
        "task_hierarchical_delta_ci_high": hierarchical_high,
        "reference_success_to_target_success": transitions[(1, 1)],
        "reference_success_to_target_failure": induced,
        "reference_failure_to_target_success": rescued,
        "reference_failure_to_target_failure": transitions[(0, 0)],
        "induced_failures": induced,
        "rescued_successes": rescued,
        "discordant_pairs": induced + rescued,
        "mcnemar_exact_p_value": exact_mcnemar_p_value(induced, rescued),
        "paired_episodes": len(keys),
    }


def _cluster_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(record["task_suite"]),
        int(record["task_id"]),
        int(record["episode_id"]),
    )


def _build_task_tables(
    task_outcomes: Mapping[str, Mapping[int, list[int]]],
    task_descriptions: Mapping[int, str],
    condition_order: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for task_id in sorted(task_descriptions):
        baseline_values = np.asarray(task_outcomes[BASELINE][task_id], dtype=int)
        baseline_rate = float(baseline_values.mean())
        wide = {
            "task_id": task_id,
            "task_description": task_descriptions[task_id],
        }
        for condition in condition_order:
            values = np.asarray(task_outcomes[condition][task_id], dtype=int)
            rate = float(values.mean())
            delta = rate - baseline_rate
            transitions = Counter(
                (int(reference), int(target))
                for reference, target in zip(baseline_values, values)
            )
            wide[condition] = rate
            long_rows.append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    "task_description": task_descriptions[task_id],
                    "successes": int(values.sum()),
                    "total_episodes": int(values.size),
                    "success_rate": rate,
                    "baseline_success_rate": baseline_rate,
                    "delta_success_rate": delta,
                    "induced_failures": transitions[(1, 0)],
                    "rescued_successes": transitions[(0, 1)],
                }
            )
        wide_rows.append(wide)
    return long_rows, wide_rows


def _catastrophic_flags(
    task_long_rows: Sequence[Mapping[str, Any]],
    condition_order: Sequence[str],
    *,
    baseline_rate_threshold: float,
    delta_threshold: float,
) -> tuple[dict[str, bool], dict[str, list[int]]]:
    flags = {condition: False for condition in condition_order}
    task_ids = {condition: [] for condition in condition_order}
    for row in task_long_rows:
        condition = str(row["condition"])
        if condition == BASELINE:
            continue
        catastrophic = (
            float(row["baseline_success_rate"]) >= baseline_rate_threshold - 1e-12
            and float(row["delta_success_rate"]) <= delta_threshold + 1e-12
        )
        if catastrophic:
            flags[condition] = True
            task_ids[condition].append(int(row["task_id"]))
    return flags, task_ids


def _build_offline_dimension_rows(
    offline_records: Mapping[str, Sequence[Mapping[str, Any]]],
    condition_order: Sequence[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in condition_order:
        records = offline_records[condition]
        clusters = [_cluster_key(record) for record in records]
        vectors = np.asarray(
            [record[DIMENSION_OFFLINE_METRIC] for record in records], dtype=np.float64
        )
        components: list[tuple[str, str, str, np.ndarray]] = []
        for dimension_index, dimension_name in enumerate(ACTION_DIMENSION_NAMES):
            semantic_group = "translation" if dimension_index < 3 else "rotation"
            components.append(
                ("dimension", str(dimension_index), dimension_name, vectors[:, dimension_index])
            )
        components.extend(
            (
                (
                    "group",
                    "translation",
                    "translation",
                    np.asarray([record["translation_norm_rms"] for record in records]),
                ),
                (
                    "group",
                    "rotation",
                    "rotation",
                    np.asarray([record["rotation_norm_rms"] for record in records]),
                ),
            )
        )
        for component_type, component_index, component_name, values in components:
            semantic_group = (
                component_name
                if component_type == "group"
                else ("translation" if int(component_index) < 3 else "rotation")
            )
            low, high = episode_cluster_bootstrap_ci(
                values,
                clusters,
                samples=bootstrap_samples,
                seed=_stable_seed(
                    bootstrap_seed, condition, "offline_dimension", component_index
                ),
            )
            rows.append(
                {
                    "condition": condition,
                    "component_type": component_type,
                    "component_index": component_index,
                    "component_name": component_name,
                    "semantic_group": semantic_group,
                    "metric_name": "executed_prefix_norm_rms",
                    "mean_value": float(np.mean(values)),
                    "mean_norm_rms": float(np.mean(values)),
                    "episode_cluster_ci_low": low,
                    "episode_cluster_ci_high": high,
                    "num_samples": len(values),
                    "num_episode_clusters": len(set(clusters)),
                }
            )
        gripper_values = np.asarray(
            [record["executed_prefix_gripper_flip_rate"] for record in records],
            dtype=np.float64,
        )
        gripper_low, gripper_high = episode_cluster_bootstrap_ci(
            gripper_values,
            clusters,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, condition, "offline_dimension", "gripper"),
        )
        rows.append(
            {
                "condition": condition,
                "component_type": "dimension",
                "component_index": "6",
                "component_name": "gripper",
                "semantic_group": "gripper",
                "metric_name": "executed_prefix_gripper_flip_rate",
                "mean_value": float(np.mean(gripper_values)),
                "mean_norm_rms": None,
                "episode_cluster_ci_low": gripper_low,
                "episode_cluster_ci_high": gripper_high,
                "num_samples": len(gripper_values),
                "num_episode_clusters": len(set(clusters)),
            }
        )
    return rows


def _build_replan_rows(
    offline_records: Mapping[str, Sequence[Mapping[str, Any]]],
    condition_order: Sequence[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in condition_order:
        by_replan: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for record in offline_records[condition]:
            by_replan[int(record["replan_id"])].append(record)
        if set(by_replan) != set(range(5)):
            raise ValueError(
                f"{condition} has replan indices {sorted(by_replan)}, expected 0..4."
            )
        for replan_id in range(5):
            records = by_replan[replan_id]
            values = [float(record["executed_prefix_norm_rms"]) for record in records]
            clusters = [_cluster_key(record) for record in records]
            low, high = episode_cluster_bootstrap_ci(
                values,
                clusters,
                samples=bootstrap_samples,
                seed=_stable_seed(bootstrap_seed, condition, "replan", replan_id),
            )
            rows.append(
                {
                    "condition": condition,
                    "replan_id": replan_id,
                    "executed_prefix_norm_rms": float(np.mean(values)),
                    "episode_cluster_ci_low": low,
                    "episode_cluster_ci_high": high,
                    "num_samples": len(values),
                    "num_episode_clusters": len(set(clusters)),
                }
            )
    return rows


def _build_planned_comparisons(
    outcomes: Mapping[str, Mapping[EpisodeKey, int]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    baseline_holm: Mapping[str, float],
) -> list[dict[str, Any]]:
    specifications = (
        ("A", "early_access_collectively_unnecessary", BASELINE, "keep_15_29"),
        ("B1", "early_half_only_vs_baseline", BASELINE, "keep_00_14"),
        ("B2", "through_layer_19_vs_baseline", BASELINE, "keep_00_19"),
        (
            "B3",
            "contribution_of_layers_15_19_under_early_only_retrieval",
            "keep_00_14",
            "keep_00_19",
        ),
        ("C", "contribution_of_layers_15_19", "keep_15_29", "keep_20_29"),
        ("D", "final_window_sufficiency", BASELINE, "keep_25_29"),
        ("E1", "middle_window_only_vs_baseline", BASELINE, "keep_15_19"),
        ("E2", "final_window_only_vs_baseline", BASELINE, "keep_25_29"),
        (
            "E3",
            "combined_critical_windows_vs_baseline",
            BASELINE,
            "keep_15_19_25_29",
        ),
        (
            "E4",
            "combined_critical_windows_vs_middle_window_only",
            "keep_15_19",
            "keep_15_19_25_29",
        ),
        (
            "E5",
            "combined_critical_windows_vs_final_window_only",
            "keep_25_29",
            "keep_15_19_25_29",
        ),
    )
    rows: list[dict[str, Any]] = []
    for comparison_id, question, reference_name, target_name in specifications:
        stats = _comparison_statistics(
            outcomes[reference_name],
            outcomes[target_name],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            seed_label=f"{reference_name}->{target_name}",
        )
        rows.append(
            {
                "comparison_id": comparison_id,
                "question": question,
                "reference_condition": reference_name,
                "target_condition": target_name,
                **stats,
                "mcnemar_holm_p_value_for_baseline_family": (
                    baseline_holm.get(target_name) if reference_name == BASELINE else None
                ),
            }
        )
    return rows


def _diagnostic_summary(
    summary_by_condition: Mapping[str, Mapping[str, Any]],
    catastrophic_task_ids: Mapping[str, Sequence[int]],
    *,
    candidate_max_loss: float,
    catastrophic_baseline_rate: float,
    catastrophic_task_delta: float,
    strong_degradation_loss: float,
) -> dict[str, Any]:
    losses = {
        condition: -float(record["delta_success_rate"])
        for condition, record in summary_by_condition.items()
    }
    candidates = {
        condition: bool(record.get("candidate_action_sufficient", False))
        for condition, record in summary_by_condition.items()
        if condition != BASELINE
    }
    strong = {
        condition: losses[condition] >= strong_degradation_loss - 1e-12
        for condition in losses
        if condition != BASELINE
    }
    patterns = {
        "A_late_half_candidate": candidates["keep_15_29"],
        "B_early_only_strongly_degraded": (
            strong["keep_00_14"] and strong["keep_00_19"]
        ),
        "C_critical_windows_complementary_candidate": (
            strong["keep_15_19"]
            and strong["keep_25_29"]
            and candidates["keep_15_19_25_29"]
        ),
        "D_final_five_candidate": candidates["keep_25_29"],
        "E_all_keep_schedules_strongly_degraded": all(strong.values()),
    }
    interpretations = {
        "A_late_half_candidate": (
            "Early direct visual retrieval is jointly dispensable in this "
            "LIBERO-Spatial screening run when later retrieval remains available."
        ),
        "B_early_only_strongly_degraded": (
            "Visual information acquired only early in the action stream is "
            "insufficient here; later direct visual re-grounding is required."
        ),
        "C_critical_windows_complementary_candidate": (
            "The 15-19 and 25-29 retrieval windows show a candidate complementary, "
            "jointly sufficient pattern."
        ),
        "D_final_five_candidate": (
            "Direct visual retrieval may be highly sparse: the final five layers "
            "meet the Round-2 candidate-sufficiency screening rule."
        ),
        "E_all_keep_schedules_strongly_degraded": (
            "No sparse keep-only schedule is supported; the Round-1 necessity profile "
            "does not by itself establish a sparse sufficient schedule."
        ),
    }
    supported = [name for name, value in patterns.items() if value]
    return {
        "screening_not_formal_non_inferiority": True,
        "thresholds": {
            "candidate_max_success_loss": candidate_max_loss,
            "catastrophic_task_baseline_rate": catastrophic_baseline_rate,
            "catastrophic_task_delta": catastrophic_task_delta,
            "strong_degradation_success_loss": strong_degradation_loss,
        },
        "candidate_action_sufficient_schedules": [
            condition for condition, value in candidates.items() if value
        ],
        "candidate_flags": candidates,
        "success_losses": losses,
        "catastrophic_task_ids": {
            condition: list(task_ids)
            for condition, task_ids in catastrophic_task_ids.items()
            if condition != BASELINE
        },
        "pattern_flags": patterns,
        "supported_pattern_flags": supported,
        "supported_interpretations": [
            {"pattern": name, "interpretation": interpretations[name]}
            for name in supported
        ],
        "overall_status": "mixed_or_inconclusive" if not supported else "pattern_flags_detected",
        "language_guardrail": (
            "Positive deltas are treated as no detected loss, not as evidence of improvement."
        ),
    }


def _diagnostic_markdown(payload: Mapping[str, Any]) -> str:
    candidates = list(payload["candidate_action_sufficient_schedules"])
    lines = [
        "# ASRE Round-2 machine diagnostic",
        "",
        "This is a 100-episode screening analysis, not a formal non-inferiority study.",
        "",
        "## Candidate schedules",
        "",
    ]
    if candidates:
        lines.extend(f"- `{condition}`" for condition in candidates)
    else:
        lines.append("- None met the pre-registered screening rule.")
    lines.extend(
        [
            "",
            "| Condition | Success loss | Candidate | Catastrophic task IDs |",
            "| --- | ---: | :---: | --- |",
        ]
    )
    candidate_flags = payload["candidate_flags"]
    catastrophic_task_ids = payload["catastrophic_task_ids"]
    for condition, loss in payload["success_losses"].items():
        if condition == BASELINE:
            continue
        task_ids = catastrophic_task_ids.get(condition, [])
        lines.append(
            f"| `{condition}` | {100.0 * float(loss):+.1f} pp | "
            f"{'yes' if candidate_flags[condition] else 'no'} | "
            f"{', '.join(str(task_id) for task_id in task_ids) if task_ids else 'none'} |"
        )
    lines.extend(["", "## Pattern flags", ""])
    for name, supported in payload["pattern_flags"].items():
        lines.append(f"- `{name}`: {'true' if supported else 'false'}")
    lines.extend(["", "## Supported interpretations", ""])
    supported_interpretations = payload["supported_interpretations"]
    if supported_interpretations:
        for record in supported_interpretations:
            lines.append(
                f"- `{record['pattern']}`: {record['interpretation']}"
            )
    else:
        lines.append("- No pre-specified pattern was decisively flagged by the screening rules.")
    baseline_reproduction = payload.get("round1_round2_baseline_reproduction")
    if isinstance(baseline_reproduction, dict):
        lines.extend(
            [
                "",
                "## Round-1 / Round-2 baseline repeat",
                "",
                f"- Round-1 baseline SR: {100.0 * float(baseline_reproduction['round1_baseline_success_rate']):.1f}%",
                f"- Round-2 baseline SR: {100.0 * float(baseline_reproduction['round2_baseline_success_rate']):.1f}%",
                f"- Round-2 minus Round-1: {100.0 * float(baseline_reproduction['round2_minus_round1_success_rate']):+.1f} pp",
                "- Aggregate rate exact match: "
                f"{'yes' if baseline_reproduction['aggregate_success_rate_exact_match'] else 'no'}",
                f"- Scope: {baseline_reproduction['scope_note']}",
            ]
        )
    lines.extend(
        [
            "",
            "A positive observed delta is reported only as no detected loss; it is not interpreted as an improvement.",
            "The protocol stop rule applies: do not add layer schedules before joint Round-1/Round-2 interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_records_for_json(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["enabled_video_retrieval_layers"] = json.loads(
            str(record["enabled_video_retrieval_layers"])
        )
        record["disabled_video_layers"] = json.loads(str(record["disabled_video_layers"]))
        task_ids = record.get("catastrophic_task_ids")
        record["catastrophic_task_ids"] = [] if task_ids in (None, "") else json.loads(str(task_ids))
        records.append(record)
    return records


def main() -> None:
    args = _parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    if args.expected_online_episodes != args.expected_tasks * args.expected_trials_per_task:
        raise ValueError(
            "--expected-online-episodes must equal --expected-tasks times "
            "--expected-trials-per-task."
        )
    condition_order, definitions = _condition_definitions()
    online_root = args.online_root.resolve()
    offline_root = args.offline_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Aggregate output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Refusing to modify a non-empty Round-2 aggregate directory. "
            f"Use a new output directory or explicitly remove a known partial run: {output_dir}"
        )
    round1_summary_path = (
        None if args.round1_summary is None else args.round1_summary.resolve()
    )
    round1_rows = (
        None
        if round1_summary_path is None
        else _load_round1_summary(round1_summary_path)
    )

    online_metadata = _discover_condition_metadata(online_root, condition_order, definitions)
    _validate_online_metadata_compatibility(online_metadata, condition_order)
    valid_sample_ids, valid_manifest_path, valid_manifest_sha256 = (
        _load_authoritative_valid_sample_ids(online_metadata)
    )
    expected_action_global_std = _load_expected_action_global_std(online_metadata)
    if (
        args.expected_offline_samples is not None
        and args.expected_offline_samples != len(valid_sample_ids)
    ):
        raise ValueError(
            "--expected-offline-samples disagrees with the immutable QC valid manifest: "
            f"{args.expected_offline_samples} != {len(valid_sample_ids)}."
        )
    expected_offline_samples = len(valid_sample_ids)
    outcomes, task_outcomes, task_descriptions = _load_online_outcomes(
        online_metadata,
        condition_order,
        definitions,
        expected_episodes=args.expected_online_episodes,
        expected_tasks=args.expected_tasks,
        expected_trials_per_task=args.expected_trials_per_task,
        expected_suite=args.task_suite,
    )
    offline_records, offline_summaries = _load_offline_records(
        offline_root,
        condition_order,
        definitions,
        expected_sample_ids=valid_sample_ids,
        valid_manifest_path=valid_manifest_path,
        valid_manifest_sha256=valid_manifest_sha256,
        online_metadata=online_metadata,
        expected_action_global_std=expected_action_global_std,
    )

    baseline_stats_by_condition: dict[str, dict[str, Any]] = {}
    raw_mcnemar: dict[str, float] = {}
    for condition in condition_order:
        stats = _comparison_statistics(
            outcomes[BASELINE],
            outcomes[condition],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            seed_label=f"{BASELINE}->{condition}",
        )
        baseline_stats_by_condition[condition] = stats
        if condition != BASELINE:
            raw_mcnemar[condition] = float(stats["mcnemar_exact_p_value"])
    adjusted_mcnemar = holm_adjust(raw_mcnemar)

    task_long_rows, task_wide_rows = _build_task_tables(
        task_outcomes, task_descriptions, condition_order
    )
    catastrophic_flags, catastrophic_task_ids = _catastrophic_flags(
        task_long_rows,
        condition_order,
        baseline_rate_threshold=args.catastrophic_baseline_rate,
        delta_threshold=args.catastrophic_task_delta,
    )

    summary_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    baseline_rate = baseline_stats_by_condition[BASELINE]["target_success_rate"]
    for condition in condition_order:
        definition = definitions[condition]
        offline = offline_summaries[condition]
        stats = baseline_stats_by_condition[condition]
        records = offline_records[condition]
        cluster_low, cluster_high = episode_cluster_bootstrap_ci(
            [record["executed_prefix_norm_rms"] for record in records],
            [_cluster_key(record) for record in records],
            samples=args.bootstrap_samples,
            seed=_stable_seed(args.bootstrap_seed, condition, "offline_overall"),
        )
        delta = float(stats["target_success_rate"] - baseline_rate)
        candidate: bool | None = None
        if condition != BASELINE:
            candidate = bool(
                -delta <= args.candidate_max_loss + 1e-12
                and not catastrophic_flags[condition]
            )
        summary_rows.append(
            {
                "condition": condition,
                "enabled_layer_ranges": definition["enabled_layer_ranges"],
                "enabled_video_retrieval_layers": json.dumps(
                    definition["enabled_video_retrieval_layers"]
                ),
                "disabled_video_layers": json.dumps(definition["disabled_video_layers"]),
                "num_retrieval_layers": definition["num_retrieval_layers"],
                "executed_prefix_norm_rms": offline["executed_prefix_norm_rms"],
                "executed_prefix_norm_rms_cluster_ci_low": cluster_low,
                "executed_prefix_norm_rms_cluster_ci_high": cluster_high,
                "norm_rms_h0": offline["norm_rms_h0"],
                "norm_rms_h0_h1": offline["norm_rms_h0_h1"],
                "full_chunk_norm_rms_0_31": offline["full_chunk_norm_rms_0_31"],
                "round1_raw_output_full_chunk_rms": offline[
                    "round1_raw_output_full_chunk_rms"
                ],
                "executed_prefix_cosine_similarity": offline[
                    "executed_prefix_cosine_similarity"
                ],
                "executed_prefix_gripper_flip_rate": offline[
                    "executed_prefix_gripper_flip_rate"
                ],
                "full_horizon_gripper_flip_rate": offline[
                    "full_horizon_gripper_flip_rate"
                ],
                "translation_norm_rms": offline["translation_norm_rms"],
                "rotation_norm_rms": offline["rotation_norm_rms"],
                "online_success_rate": stats["target_success_rate"],
                "delta_success_rate": delta,
                "paired_delta_success_rate": stats["paired_delta_success_rate"],
                "paired_delta_ci_low": stats["paired_delta_ci_low"],
                "paired_delta_ci_high": stats["paired_delta_ci_high"],
                "task_hierarchical_delta_ci_low": stats[
                    "task_hierarchical_delta_ci_low"
                ],
                "task_hierarchical_delta_ci_high": stats[
                    "task_hierarchical_delta_ci_high"
                ],
                "mcnemar_exact_p_value": (
                    None if condition == BASELINE else stats["mcnemar_exact_p_value"]
                ),
                "mcnemar_holm_p_value": adjusted_mcnemar.get(condition),
                "candidate_action_sufficient": candidate,
                "catastrophic_task_failure": (
                    None if condition == BASELINE else catastrophic_flags[condition]
                ),
                "catastrophic_task_ids": (
                    "" if condition == BASELINE else json.dumps(catastrophic_task_ids[condition])
                ),
                "total_episodes": stats["paired_episodes"],
                "offline_samples": len(records),
            }
        )
        transition_rows.append(
            {
                "condition": condition,
                "baseline_success_to_intervention_success": stats[
                    "reference_success_to_target_success"
                ],
                "baseline_success_to_intervention_failure": stats[
                    "reference_success_to_target_failure"
                ],
                "baseline_failure_to_intervention_success": stats[
                    "reference_failure_to_target_success"
                ],
                "baseline_failure_to_intervention_failure": stats[
                    "reference_failure_to_target_failure"
                ],
                "induced_failures": stats["induced_failures"],
                "rescued_successes": stats["rescued_successes"],
                "discordant_pairs": stats["discordant_pairs"],
                "mcnemar_exact_p_value": (
                    None if condition == BASELINE else stats["mcnemar_exact_p_value"]
                ),
                "mcnemar_holm_p_value": adjusted_mcnemar.get(condition),
                "paired_episodes": stats["paired_episodes"],
            }
        )

    summary_by_condition = {row["condition"]: row for row in summary_rows}
    dimension_rows = _build_offline_dimension_rows(
        offline_records,
        condition_order,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    replan_rows = _build_replan_rows(
        offline_records,
        condition_order,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    planned_rows = _build_planned_comparisons(
        outcomes,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        baseline_holm=adjusted_mcnemar,
    )
    diagnostic = _diagnostic_summary(
        summary_by_condition,
        catastrophic_task_ids,
        candidate_max_loss=args.candidate_max_loss,
        catastrophic_baseline_rate=args.catastrophic_baseline_rate,
        catastrophic_task_delta=args.catastrophic_task_delta,
        strong_degradation_loss=args.strong_degradation_loss,
    )
    joint_rows: list[dict[str, Any]] | None = None
    if round1_rows is not None:
        joint_rows = _build_joint_round_table(
            round1_rows,
            summary_rows,
            round2_offline_samples=expected_offline_samples,
        )
        round1_baseline = next(
            record for record in round1_rows if record["condition"] == "baseline"
        )
        round1_rate = float(round1_baseline["online_success_rate"])
        round2_rate = float(summary_by_condition[BASELINE]["online_success_rate"])
        diagnostic["round1_round2_baseline_reproduction"] = {
            "round1_baseline_success_rate": round1_rate,
            "round2_baseline_success_rate": round2_rate,
            "round2_minus_round1_success_rate": round2_rate - round1_rate,
            "aggregate_success_rate_exact_match": math.isclose(
                round1_rate, round2_rate, rel_tol=0.0, abs_tol=1e-12
            ),
            "scope_note": (
                "This checks aggregate success-rate reproducibility; the Round-1 "
                "summary alone does not establish episode-level outcome identity."
            ),
        }
        diagnostic["joint_round1_round2_table"] = "round1_round2_joint_summary.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "paired_transitions.csv", transition_rows)
    _write_csv(output_dir / "task_success_rates.csv", task_wide_rows)
    _write_csv(output_dir / "task_success_delta.csv", task_long_rows)
    _write_csv(output_dir / "offline_action_dimensions.csv", dimension_rows)
    _write_csv(output_dir / "replan_stage_metrics.csv", replan_rows)
    _write_csv(output_dir / "planned_comparisons.csv", planned_rows)
    if joint_rows is not None:
        _write_csv(output_dir / "round1_round2_joint_summary.csv", joint_rows)
    atomic_write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "baseline_condition": BASELINE,
            "condition_order": condition_order,
            "records": _summary_records_for_json(summary_rows),
        },
    )
    atomic_write_json(output_dir / "diagnostic_summary.json", diagnostic)
    (output_dir / "diagnostic_summary.md").write_text(
        _diagnostic_markdown(diagnostic), encoding="utf-8"
    )
    input_artifacts: dict[str, dict[str, str]] = {}
    reproducibility_fields = (
        "git_commit_hash",
        "checkpoint_path",
        "checkpoint_sha256",
        "dataset_stats_path",
        "dataset_stats_sha256",
        "state_bank_manifest_path",
        "state_bank_manifest_sha256",
        "valid_state_bank_manifest_path",
        "valid_state_bank_manifest_sha256",
        "seed",
        "task_suite",
        "task_ids",
        "number_of_trials",
        "action_horizon",
        "replan_steps",
        "number_of_inference_steps",
    )
    reproducibility_metadata: dict[str, dict[str, Any]] = {}
    for condition in condition_order:
        condition_dir = Path(str(online_metadata[condition]["_condition_dir"]))
        online_task_paths = sorted(condition_dir.glob("**/gpu*_task*_results.json"))
        metadata_path = Path(str(online_metadata[condition]["_metadata_path"]))
        offline_condition_dir = offline_root / condition
        input_artifacts[condition] = {
            "online_run_metadata_sha256": _sha256_path(metadata_path),
            "online_task_results_file_set_sha256": _sha256_file_set(
                online_task_paths, relative_to=condition_dir
            ),
            "offline_per_sample_sha256": _sha256_path(
                offline_condition_dir / "per_sample.jsonl"
            ),
            "offline_summary_sha256": _sha256_path(offline_condition_dir / "summary.csv"),
        }
        reproducibility_metadata[condition] = {
            key: online_metadata[condition].get(key) for key in reproducibility_fields
        }
    atomic_write_json(
        output_dir / "aggregate_metadata.json",
        {
            "schema_version": 1,
            "status": "complete",
            "generated_at": now_iso(),
            "analysis_git_commit_hash": git_commit(project_root),
            "online_root": str(online_root),
            "offline_root": str(offline_root),
            "strict_episode_and_sample_identity_validation": True,
            "condition_order": condition_order,
            "condition_definitions": definitions,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "paired_bootstrap_unit": "paired_episode",
            "task_hierarchical_bootstrap": "resample tasks, then paired episodes within task",
            "offline_bootstrap_unit": "episode cluster preserving saved replan states",
            "expected_online_episodes_per_condition": args.expected_online_episodes,
            "expected_offline_samples_per_condition": expected_offline_samples,
            "valid_state_bank_manifest_path": str(valid_manifest_path),
            "valid_state_bank_manifest_sha256": valid_manifest_sha256,
            "round1_summary_path": (
                None if round1_summary_path is None else str(round1_summary_path)
            ),
            "round1_summary_sha256": (
                None if round1_summary_path is None else _sha256_path(round1_summary_path)
            ),
            "expected_tasks": args.expected_tasks,
            "expected_trials_per_task": args.expected_trials_per_task,
            "task_suite": args.task_suite,
            "holm_family": [condition for condition in condition_order if condition != BASELINE],
            "candidate_rule": {
                "maximum_success_loss": args.candidate_max_loss,
                "catastrophic_task_baseline_rate": args.catastrophic_baseline_rate,
                "catastrophic_task_delta": args.catastrophic_task_delta,
            },
            "strong_degradation_success_loss": args.strong_degradation_loss,
            "action_dimension_names": list(ACTION_DIMENSION_NAMES),
            "action_dimension_label_source": ACTION_DIMENSION_LABEL_SOURCE,
            "online_metadata_paths": {
                condition: online_metadata[condition]["_metadata_path"]
                for condition in condition_order
            },
            "input_artifact_sha256": input_artifacts,
            "online_reproducibility_metadata": reproducibility_metadata,
        },
    )
    print(f"Aggregated all 8 ASRE Round-2 conditions into {output_dir}")


if __name__ == "__main__":
    main()
