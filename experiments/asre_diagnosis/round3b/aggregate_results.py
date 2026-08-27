"""Strict aggregation for the three-arm ASRE Round-3B causal control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND3B_PROTOCOL,
    atomic_write_json,
    build_round3b_conditions,
    sha256_file,
)
from experiments.asre_diagnosis.round2.metrics import (  # noqa: E402
    compute_round2_metrics,
)
from experiments.asre_diagnosis.round3b.outcome_statistics import (  # noqa: E402
    COMPARISON_ORDER,
    CONDITION_ORDER,
    analyze_outcomes,
    paired_bootstrap_ci,
    task_hierarchical_bootstrap_ci,
)


DISPLAY_NAMES = {
    "late_current_correct": "Correct Current K/V",
    "late_wrong_scene": "Wrong Same-Task Scene K/V",
    "late_no_video": "No Video K/V",
}

SCALAR_OFFLINE_METRICS = (
    "executed_prefix_norm_rms",
    "full_chunk_norm_rms_0_31",
    "translation_norm_rms",
    "rotation_norm_rms",
    "executed_prefix_cosine_similarity",
    "executed_prefix_gripper_flip_rate",
    "full_horizon_gripper_flip_rate",
)

EXPECTED_PARENT_TAG = "ASRE-round3a-factorial"
EXPECTED_PARENT_COMMIT = "d36383c16d974ba1e5a750c088327a7a88baa8fb"
EXPECTED_RUN_COMMIT = "92e842c79f5209b31c2653944cc0c1719e95eb9e"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online-root",
        type=Path,
        default=PROJECT_ROOT / "asre_results/round3b/online_full",
    )
    parser.add_argument(
        "--offline-root",
        type=Path,
        default=PROJECT_ROOT / "asre_results/round3b/offline",
    )
    parser.add_argument(
        "--valid-manifest",
        type=Path,
        default=PROJECT_ROOT / "asre_results/round2/state_bank_valid_manifest.json",
    )
    parser.add_argument(
        "--offline-donor-mapping",
        type=Path,
        default=PROJECT_ROOT / "asre_results/round3b/donors/offline_donor_mapping.json",
    )
    parser.add_argument(
        "--online-donor-mapping",
        type=Path,
        default=PROJECT_ROOT / "asre_results/round3b/donors/donor_mapping.json",
    )
    parser.add_argument(
        "--online-donor-manifest",
        type=Path,
        default=PROJECT_ROOT / "asre_results/round3b/donors/donor_observation_manifest.json",
    )
    parser.add_argument(
        "--self-replacement-result",
        type=Path,
        default=PROJECT_ROOT / "asre_results/round3b/self_replacement/result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "asre_results/round3b/aggregate",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required Round-3B artifact is missing: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}.")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
                if not isinstance(record, dict):
                    raise TypeError(f"Expected JSON object at {path}:{line_number}.")
                records.append(record)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required Round-3B artifact is missing: {path}") from exc
    return records


def _finite(value: Any, *, context: str) -> float:
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}.")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
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
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
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
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _expected_condition_configs() -> dict[str, dict[str, Any]]:
    return {condition.name: condition.to_dict() for condition in build_round3b_conditions(30)}


def _validate_online_metadata(
    metadata: Mapping[str, Any],
    *,
    condition: str,
    donor_mapping_sha256: str,
    donor_manifest_sha256: str,
) -> None:
    expected = _expected_condition_configs()[condition]
    checks = {
        "diagnosis_condition": condition,
        "condition_protocol": ROUND3B_PROTOCOL,
        "num_model_layers": 30,
        "number_of_trials": 10,
        "seed": 42,
        "action_horizon": 32,
        "number_of_inference_steps": 10,
        "replan_steps": 10,
        "disabled_video_layers": expected["disabled_video_layers"],
        "replacement_video_layers": expected["replacement_video_layers"],
    }
    mismatches = {
        key: {"expected": expected_value, "observed": metadata.get(key)}
        for key, expected_value in checks.items()
        if metadata.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"Online metadata mismatch for {condition}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    if str(metadata.get("status")) not in {"complete", "completed"}:
        raise ValueError(f"Online condition {condition} is not complete.")
    # All three arms load the same frozen donor bundle, even though only the
    # wrong-scene arm consumes its image.  Requiring the hashes in every arm
    # makes cross-condition provenance explicit and fail-closed.
    provenance = {
        "donor_mapping_sha256": donor_mapping_sha256,
        "donor_observation_manifest_sha256": donor_manifest_sha256,
    }
    bad = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in provenance.items()
        if metadata.get(key) != value
    }
    if bad:
        raise ValueError(
            f"Online donor provenance mismatch for {condition}: "
            f"{json.dumps(bad, sort_keys=True)}"
        )


def load_online_results(
    *,
    online_root: Path,
    donor_mapping_path: Path,
    donor_manifest_path: Path,
) -> tuple[
    dict[str, dict[tuple[str, int, int], int]],
    dict[str, dict[str, Any]],
    dict[int, str],
]:
    donor_mapping_sha256 = sha256_file(donor_mapping_path)
    donor_manifest_sha256 = sha256_file(donor_manifest_path)
    launcher_summary = _read_json(online_root / "launcher_summary.json")
    launcher_expected = {
        "schema_version": 1,
        "protocol": ROUND3B_PROTOCOL,
        "mode": "full",
        "all_succeeded": True,
        "interrupted": False,
        "task_ids": list(range(10)),
        "num_trials": 10,
        "seed": 42,
        "expected_conditions": list(CONDITION_ORDER),
    }
    launcher_bad = {
        key: {"expected": value, "observed": launcher_summary.get(key)}
        for key, value in launcher_expected.items()
        if launcher_summary.get(key) != value
    }
    launcher_states = launcher_summary.get("condition_states")
    if not isinstance(launcher_states, Mapping) or any(
        not isinstance(launcher_states.get(condition), Mapping)
        or launcher_states[condition].get("state") != "complete"
        or launcher_states[condition].get("completed_task_ids") != list(range(10))
        for condition in CONDITION_ORDER
    ):
        launcher_bad["condition_states"] = {
            "expected": "all three conditions complete for tasks 0-9",
            "observed": launcher_states,
        }
    if launcher_bad:
        raise ValueError(
            "Round-3B full launcher summary is incomplete or incompatible: "
            f"{json.dumps(launcher_bad, sort_keys=True)}"
        )
    outcomes: dict[str, dict[tuple[str, int, int], int]] = {}
    metadata_by_condition: dict[str, dict[str, Any]] = {}
    task_descriptions: dict[int, str] = {}

    for condition in CONDITION_ORDER:
        condition_dir = (online_root / condition).resolve()
        metadata = _read_json(condition_dir / "run_metadata.json")
        _validate_online_metadata(
            metadata,
            condition=condition,
            donor_mapping_sha256=donor_mapping_sha256,
            donor_manifest_sha256=donor_manifest_sha256,
        )
        task_files = sorted((condition_dir / "libero_spatial").glob("*_task*_results.json"))
        if len(task_files) != 10:
            raise ValueError(
                f"Expected 10 task result files for {condition}, got {len(task_files)}."
            )
        condition_outcomes: dict[tuple[str, int, int], int] = {}
        seen_tasks: set[int] = set()
        for task_file in task_files:
            payload = _read_json(task_file)
            if str(payload.get("diagnosis_condition")) != condition:
                raise ValueError(f"Task condition mismatch in {task_file}.")
            if str(payload.get("condition_protocol")) != ROUND3B_PROTOCOL:
                raise ValueError(f"Task protocol mismatch in {task_file}.")
            task_id = int(payload.get("task_id", -1))
            if task_id in seen_tasks or task_id not in range(10):
                raise ValueError(f"Duplicate/invalid task ID {task_id} for {condition}.")
            seen_tasks.add(task_id)
            if int(payload.get("total_episodes", -1)) != 10:
                raise ValueError(f"Task {task_id} for {condition} did not run 10 trials.")
            successes = [int(value) for value in payload.get("success_episodes", [])]
            failures = [int(value) for value in payload.get("failure_episodes", [])]
            if set(successes) & set(failures) or sorted(successes + failures) != list(range(10)):
                raise ValueError(f"Invalid trial partition in {task_file}.")
            if int(payload.get("successes", -1)) != len(successes):
                raise ValueError(f"Success count mismatch in {task_file}.")
            suite = str(payload.get("task_suite"))
            if suite != "libero_spatial":
                raise ValueError(f"Unexpected task suite {suite!r} in {task_file}.")
            description = str(payload.get("task_description", ""))
            if not description:
                raise ValueError(f"Missing task description in {task_file}.")
            if task_id in task_descriptions and task_descriptions[task_id] != description:
                raise ValueError(f"Task description drift for task {task_id}.")
            task_descriptions[task_id] = description
            for trial_id in successes:
                condition_outcomes[(suite, task_id, trial_id)] = 1
            for trial_id in failures:
                condition_outcomes[(suite, task_id, trial_id)] = 0
        if seen_tasks != set(range(10)) or len(condition_outcomes) != 100:
            raise ValueError(f"Online condition {condition} does not cover 100 paired episodes.")
        outcomes[condition] = condition_outcomes
        metadata_by_condition[condition] = metadata

    paired_sets = {condition: set(values) for condition, values in outcomes.items()}
    if any(keys != paired_sets[CONDITION_ORDER[0]] for keys in paired_sets.values()):
        raise ValueError("Online task/trial keys are not exactly paired across conditions.")
    invariant_fields = (
        "git_commit_hash",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "state_bank_manifest_sha256",
        "valid_state_bank_manifest_sha256",
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
    )
    reference = metadata_by_condition[CONDITION_ORDER[0]]
    for condition in CONDITION_ORDER[1:]:
        mismatches = {
            field: {
                "reference": reference.get(field),
                "observed": metadata_by_condition[condition].get(field),
            }
            for field in invariant_fields
            if metadata_by_condition[condition].get(field) != reference.get(field)
        }
        if mismatches:
            raise ValueError(
                f"Cross-condition online provenance drift for {condition}: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
    if launcher_summary.get("git_commit_hash") != reference.get("git_commit_hash"):
        raise ValueError("Launcher summary Git commit differs from condition metadata.")
    return outcomes, metadata_by_condition, task_descriptions


def _cell_intervals(
    outcomes: Mapping[str, Mapping[tuple[str, int, int], int]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    keys = sorted(outcomes[CONDITION_ORDER[0]])
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        values = np.asarray([outcomes[condition][key] for key in keys], dtype=np.float64)
        paired_low, paired_high = paired_bootstrap_ci(
            values,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, condition, "cell_paired"),
        )
        task_low, task_high = task_hierarchical_bootstrap_ci(
            keys,
            values,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, condition, "cell_task"),
        )
        rows.append(
            {
                "condition": condition,
                "display_name": DISPLAY_NAMES[condition],
                "successes": int(values.sum()),
                "episodes": int(values.size),
                "success_rate": float(values.mean()),
                "paired_ci_low": paired_low,
                "paired_ci_high": paired_high,
                "task_hierarchical_ci_low": task_low,
                "task_hierarchical_ci_high": task_high,
            }
        )
    return rows


def _task_rows(
    outcomes: Mapping[str, Mapping[tuple[str, int, int], int]],
    task_descriptions: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in range(10):
        row: dict[str, Any] = {
            "task_id": task_id,
            "task_description": task_descriptions[task_id],
        }
        for condition in CONDITION_ORDER:
            values = [
                outcomes[condition][("libero_spatial", task_id, trial_id)]
                for trial_id in range(10)
            ]
            row[condition] = float(np.mean(values))
        row["wrong_minus_correct"] = (
            row["late_wrong_scene"] - row["late_current_correct"]
        )
        row["no_video_minus_correct"] = (
            row["late_no_video"] - row["late_current_correct"]
        )
        row["wrong_minus_no_video"] = (
            row["late_wrong_scene"] - row["late_no_video"]
        )
        rows.append(row)
    return rows


def _comparison_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(analysis["comparisons"][name]) for name in COMPARISON_ORDER]


def _transition_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in COMPARISON_ORDER:
        comparison = analysis["comparisons"][name]
        rows.extend(
            [
                {
                    "comparison": name,
                    "reference_outcome": reference,
                    "target_outcome": target,
                    "count": int(comparison[field]),
                }
                for reference, target, field in (
                    (1, 1, "reference_success_to_target_success"),
                    (1, 0, "reference_success_to_target_failure"),
                    (0, 1, "reference_failure_to_target_success"),
                    (0, 0, "reference_failure_to_target_failure"),
                )
            ]
        )
    return rows


def _load_valid_ids(path: Path) -> tuple[list[str], dict[str, Any], str]:
    payload = _read_json(path)
    valid_ids = [str(value) for value in payload.get("valid_sample_ids", [])]
    if len(valid_ids) != 499 or len(set(valid_ids)) != 499:
        raise ValueError("Valid state manifest must contain exactly 499 unique IDs.")
    return valid_ids, payload, sha256_file(path)


def _load_actions(path: Path, expected_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            ids = [str(value) for value in payload["sample_ids"].tolist()]
            raw = np.asarray(payload["raw_actions"], dtype=np.float32)
            executed = np.asarray(payload["executed_actions"], dtype=np.float32)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing Round-3B action artifact: {path}") from exc
    if ids != list(expected_ids):
        raise ValueError(f"Action sample ordering does not match valid manifest: {path}.")
    if raw.shape != (499, 32, 7) or executed.shape != raw.shape:
        raise ValueError(f"Unexpected saved action shapes in {path}: {raw.shape}/{executed.shape}.")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(executed)):
        raise ValueError(f"Nonfinite saved actions in {path}.")
    return raw, executed


def _cluster_bootstrap_ci(
    values: Sequence[float],
    cluster_ids: Sequence[tuple[str, int, int]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.size == 0 or numeric.size != len(cluster_ids):
        raise ValueError("Cluster bootstrap values/IDs must be nonempty and aligned.")
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for cluster, value in zip(cluster_ids, numeric):
        grouped[cluster].append(float(value))
    groups = [np.asarray(grouped[key], dtype=np.float64) for key in sorted(grouped)]
    sums = np.asarray([group.sum() for group in groups])
    counts = np.asarray([group.size for group in groups])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(samples, len(groups)))
    estimates = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def load_offline_results(
    *,
    offline_root: Path,
    valid_manifest_path: Path,
    offline_donor_mapping_path: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    valid_ids, valid_manifest, valid_manifest_sha256 = _load_valid_ids(valid_manifest_path)
    mapping = _read_json(offline_donor_mapping_path)
    mapping_sha256 = sha256_file(offline_donor_mapping_path)
    if mapping.get("valid_manifest_sha256") != valid_manifest_sha256:
        raise ValueError("Offline donor mapping does not match the valid manifest.")
    if int(mapping.get("num_pairs", -1)) != 499:
        raise ValueError("Offline donor mapping does not contain 499 pairs.")

    expected_configs = _expected_condition_configs()
    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    metadata_by_condition: dict[str, dict[str, Any]] = {}
    raw_by_condition: dict[str, np.ndarray] = {}
    executed_by_condition: dict[str, np.ndarray] = {}
    condition_rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        condition_dir = offline_root / condition
        metadata = _read_json(condition_dir / "run_metadata.json")
        expected_config = expected_configs[condition]
        checks = {
            "artifact_type": "asre_round3b_offline_state_bank_replay",
            "status": "complete",
            "diagnosis_condition": condition,
            "condition_protocol": ROUND3B_PROTOCOL,
            "num_model_layers": 30,
            "num_samples": 499,
            "valid_state_bank_manifest_sha256": valid_manifest_sha256,
            "offline_donor_mapping_sha256": mapping_sha256,
            "disabled_video_layers": expected_config["disabled_video_layers"],
            "replacement_video_layers": expected_config["replacement_video_layers"],
            "round3a_parent_tag": EXPECTED_PARENT_TAG,
            "round3a_parent_commit": EXPECTED_PARENT_COMMIT,
            "validated_round3a_run_commit": EXPECTED_RUN_COMMIT,
        }
        bad = {
            key: {"expected": value, "observed": metadata.get(key)}
            for key, value in checks.items()
            if metadata.get(key) != value
        }
        if bad:
            raise ValueError(
                f"Offline metadata mismatch for {condition}: {json.dumps(bad, sort_keys=True)}"
            )
        records = _read_jsonl(condition_dir / "per_sample.jsonl")
        ids = [str(record.get("sample_id", "")) for record in records]
        if ids != valid_ids:
            raise ValueError(f"Offline sample ordering mismatch for {condition}.")
        for record in records:
            if str(record.get("condition")) != condition:
                raise ValueError(f"Offline per-sample condition mismatch for {condition}.")
            for metric in SCALAR_OFFLINE_METRICS:
                _finite(record.get(metric), context=f"{condition}.{record['sample_id']}.{metric}")
        raw, executed = _load_actions(condition_dir / "actions.npz", valid_ids)
        if sha256_file(condition_dir / "actions.npz") != metadata.get("actions_sha256"):
            raise ValueError(f"Saved action hash mismatch for {condition}.")
        records_by_condition[condition] = records
        metadata_by_condition[condition] = metadata
        raw_by_condition[condition] = raw
        executed_by_condition[condition] = executed
        row: dict[str, Any] = {
            "condition": condition,
            "display_name": DISPLAY_NAMES[condition],
            "num_samples": len(records),
        }
        for metric in SCALAR_OFFLINE_METRICS:
            row[metric] = float(np.mean([float(record[metric]) for record in records]))
        condition_rows.append(row)

    reference_metadata = metadata_by_condition[CONDITION_ORDER[0]]
    for condition in CONDITION_ORDER[1:]:
        for field in (
            "git_commit_hash",
            "checkpoint_sha256",
            "dataset_stats_sha256",
            "state_bank_manifest_sha256",
            "valid_state_bank_manifest_sha256",
            "prompt_context_cache_sha256",
            "action_global_std",
            "compile_action_infer",
            "binarize_gripper",
            "sigma_shift",
            "rand_device",
        ):
            if metadata_by_condition[condition].get(field) != reference_metadata.get(field):
                raise ValueError(f"Offline cross-condition provenance drift: {condition}.{field}.")

    action_std = np.asarray(reference_metadata["action_global_std"], dtype=np.float64)
    correct_records = records_by_condition["late_current_correct"]
    wrong_cross_records: list[dict[str, Any]] = []
    for index, source_record in enumerate(correct_records):
        metrics = compute_round2_metrics(
            baseline_raw=raw_by_condition["late_current_correct"][index],
            diagnosis_raw=raw_by_condition["late_wrong_scene"][index],
            baseline_executed=executed_by_condition["late_current_correct"][index],
            diagnosis_executed=executed_by_condition["late_wrong_scene"][index],
            action_std=action_std,
            executed_prefix_length=10,
        )
        wrong_cross_records.append(
            {
                "sample_id": source_record["sample_id"],
                "task_suite": source_record["task_suite"],
                "task_id": int(source_record["task_id"]),
                "episode_id": int(source_record["episode_id"]),
                "replan_id": int(source_record["replan_id"]),
                **metrics,
            }
        )
    cluster_ids = [
        (str(record["task_suite"]), int(record["task_id"]), int(record["episode_id"]))
        for record in wrong_cross_records
    ]
    wrong_cross_rows: list[dict[str, Any]] = []
    for metric in SCALAR_OFFLINE_METRICS:
        values = [float(record[metric]) for record in wrong_cross_records]
        low, high = _cluster_bootstrap_ci(
            values,
            cluster_ids,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, "wrong_vs_correct", metric),
        )
        wrong_cross_rows.append(
            {
                "comparison": "wrong_vs_correct",
                "metric": metric,
                "mean": float(np.mean(values)),
                "episode_cluster_ci_low": low,
                "episode_cluster_ci_high": high,
                "num_states": len(values),
                "num_episode_clusters": len(set(cluster_ids)),
            }
        )

    replan_rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        for replan_id in range(5):
            subset = [
                record
                for record in records_by_condition[condition]
                if int(record["replan_id"]) == replan_id
            ]
            replan_rows.append(
                {
                    "series": condition,
                    "replan_id": replan_id,
                    "num_states": len(subset),
                    **{
                        metric: float(np.mean([float(record[metric]) for record in subset]))
                        for metric in SCALAR_OFFLINE_METRICS
                    },
                }
            )
    for replan_id in range(5):
        subset = [
            record
            for record in wrong_cross_records
            if int(record["replan_id"]) == replan_id
        ]
        replan_rows.append(
            {
                "series": "wrong_vs_correct",
                "replan_id": replan_id,
                "num_states": len(subset),
                **{
                    metric: float(np.mean([float(record[metric]) for record in subset]))
                    for metric in SCALAR_OFFLINE_METRICS
                },
            }
        )

    kv_rows = summarize_kv_stats(
        _read_jsonl(offline_root / "late_wrong_scene/video_cache_stats.jsonl"),
        expected_sample_ids=valid_ids,
    )
    return (
        condition_rows,
        wrong_cross_rows,
        wrong_cross_records,
        replan_rows,
        kv_rows,
        metadata_by_condition,
    )


def summarize_kv_stats(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_sample_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if len(records) != len(expected_sample_ids):
        raise ValueError(
            f"Expected {len(expected_sample_ids)} cache-audit records, got {len(records)}."
        )
    if [str(record.get("sample_id", "")) for record in records] != list(
        expected_sample_ids
    ):
        raise ValueError("Cache-audit ordering does not match the valid state manifest.")
    rows: list[dict[str, Any]] = []
    for layer in range(15, 30):
        for tensor_name in ("k", "v"):
            current_stats: list[Mapping[str, Any]] = []
            replacement_stats: list[Mapping[str, Any]] = []
            for record in records:
                layers = record.get("layers")
                if not isinstance(layers, list) or len(layers) != 30:
                    raise ValueError("Each cache audit must contain exactly 30 layer records.")
                layer_record = layers[layer]
                if int(layer_record.get("layer", -1)) != layer:
                    raise ValueError("Cache-audit layer order mismatch.")
                if layer_record.get("selected_source") != "replacement":
                    raise ValueError(f"Layer {layer} is not replacement-selected.")
                current_stats.append(layer_record["current"][tensor_name])
                replacement_stats.append(layer_record["replacement"][tensor_name])
            current_shapes = {tuple(value["shape"]) for value in current_stats}
            replacement_shapes = {tuple(value["shape"]) for value in replacement_stats}
            dtypes = {
                str(value["dtype"]) for value in current_stats + replacement_stats
            }
            if len(current_shapes) != 1 or current_shapes != replacement_shapes:
                raise ValueError(f"K/V shape drift at layer {layer} {tensor_name}.")
            if len(dtypes) != 1:
                raise ValueError(f"K/V dtype drift at layer {layer} {tensor_name}.")
            if not all(bool(value.get("finite", False)) for value in current_stats):
                raise ValueError(f"Nonfinite current cache at layer {layer} {tensor_name}.")
            if not all(bool(value.get("finite", False)) for value in replacement_stats):
                raise ValueError(f"Nonfinite donor cache at layer {layer} {tensor_name}.")
            current_mean = float(np.mean([_finite(value["mean"], context="cache mean") for value in current_stats]))
            donor_mean = float(np.mean([_finite(value["mean"], context="cache mean") for value in replacement_stats]))
            current_std = float(np.mean([_finite(value["std"], context="cache std") for value in current_stats]))
            donor_std = float(np.mean([_finite(value["std"], context="cache std") for value in replacement_stats]))
            current_rms = float(np.mean([_finite(value["rms"], context="cache rms") for value in current_stats]))
            donor_rms = float(np.mean([_finite(value["rms"], context="cache rms") for value in replacement_stats]))
            rows.append(
                {
                    "layer": layer,
                    "tensor": tensor_name.upper(),
                    "shape": json.dumps(list(next(iter(current_shapes)))),
                    "dtype": next(iter(dtypes)),
                    "num_samples": len(records),
                    "current_mean": current_mean,
                    "donor_mean": donor_mean,
                    "donor_current_mean_ratio": (
                        None if math.isclose(current_mean, 0.0, abs_tol=1e-15) else donor_mean / current_mean
                    ),
                    "current_std": current_std,
                    "donor_std": donor_std,
                    "donor_current_std_ratio": donor_std / current_std,
                    "current_rms": current_rms,
                    "donor_rms": donor_rms,
                    "donor_current_rms_ratio": donor_rms / current_rms,
                }
            )
    return rows


def _format_rate(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_pp(value: float) -> str:
    return f"{100.0 * float(value):+.1f} pp"


def _summary_markdown(payload: Mapping[str, Any], *, for_gpt: bool) -> str:
    cells = payload["online_condition_summary"]
    comparisons = payload["online_comparisons"]
    tasks = payload["task_success"]
    offline = payload["offline_condition_metrics"]
    cross = {row["metric"]: row for row in payload["offline_wrong_vs_correct"]}
    decision = payload["analysis"]["decision"]
    primary = next(row for row in comparisons if row["comparison"] == "wrong_minus_correct")
    title = (
        "# Fast-WAM ASRE Round 3B: Matched-Shape K/V Replacement Result Summary"
        if for_gpt
        else "# ASRE Round 3B summary"
    )
    lines = [
        title,
        "",
        "## Executive result",
        "",
        (
            f"The pre-specified decision is **{decision['classification']}** "
            f"(`{decision['label']}`). {decision['rationale']}"
        ),
        "",
        (
            "The primary wrong-minus-correct contrast is "
            f"{_format_pp(primary['delta_success_rate'])}, with paired 95% CI "
            f"[{_format_pp(primary['paired_ci_low'])}, "
            f"{_format_pp(primary['paired_ci_high'])}] and task-hierarchical 95% CI "
            f"[{_format_pp(primary['task_hierarchical_ci_low'])}, "
            f"{_format_pp(primary['task_hierarchical_ci_high'])}]."
        ),
        "",
        "## Online success",
        "",
        "| Condition | Success | Paired 95% CI | Task-hierarchical 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['display_name']} | {row['successes']}/{row['episodes']} "
            f"({_format_rate(row['success_rate'])}) | "
            f"[{_format_rate(row['paired_ci_low'])}, {_format_rate(row['paired_ci_high'])}] | "
            f"[{_format_rate(row['task_hierarchical_ci_low'])}, "
            f"{_format_rate(row['task_hierarchical_ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "| Contrast (target - reference) | Delta | Paired 95% CI | Task 95% CI | Induced | Rescued | McNemar p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| `{row['comparison']}` | {_format_pp(row['delta_success_rate'])} | "
            f"[{_format_pp(row['paired_ci_low'])}, {_format_pp(row['paired_ci_high'])}] | "
            f"[{_format_pp(row['task_hierarchical_ci_low'])}, "
            f"{_format_pp(row['task_hierarchical_ci_high'])}] | "
            f"{row['induced_failures']} | {row['rescued_successes']} | "
            f"{row['mcnemar_exact_p_value']:.4g} |"
        )
    fraction = payload["analysis"]["content_gap_fraction"]
    lines.extend(
        [
            "",
            "The descriptive `content_gap_fraction` is "
            + ("undefined" if fraction is None else f"{float(fraction):.3f}")
            + ". It is not a mediation fraction or variance-explained estimate.",
            "",
            "## Per-task outcomes",
            "",
            "| Task | Correct | Wrong scene | No video | Wrong - correct |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tasks:
        lines.append(
            f"| {row['task_id']} | {_format_rate(row['late_current_correct'])} | "
            f"{_format_rate(row['late_wrong_scene'])} | "
            f"{_format_rate(row['late_no_video'])} | "
            f"{_format_pp(row['wrong_minus_correct'])} |"
        )
    lines.extend(
        [
            "",
            "## Offline fixed-state action sensitivity",
            "",
            "| Condition | Prefix RMS | Full RMS | Translation RMS | Rotation RMS | Cosine | Grip flip |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in offline:
        lines.append(
            f"| {row['display_name']} | {row['executed_prefix_norm_rms']:.4f} | "
            f"{row['full_chunk_norm_rms_0_31']:.4f} | {row['translation_norm_rms']:.4f} | "
            f"{row['rotation_norm_rms']:.4f} | "
            f"{row['executed_prefix_cosine_similarity']:.4f} | "
            f"{100.0 * row['executed_prefix_gripper_flip_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "The direct wrong-vs-correct executed-prefix RMS is "
            f"{cross['executed_prefix_norm_rms']['mean']:.4f} "
            f"(episode-clustered 95% CI "
            f"[{cross['executed_prefix_norm_rms']['episode_cluster_ci_low']:.4f}, "
            f"{cross['executed_prefix_norm_rms']['episode_cluster_ci_high']:.4f}]).",
            "",
            "## Machinery and donor controls",
            "",
            f"- Self replacement passed: `{payload['self_replacement']['passed']}`; "
            f"max raw-action absolute difference "
            f"`{payload['self_replacement']['max_raw_action_abs_diff']}`.",
            "- Online donors use `(recipient_trial + 1) mod 10` within each task and a fixed "
            "first-policy-query visual observation after the evaluator's 30 dummy steps.",
            "- Offline donors are cyclic derangements within the same task and saved replan "
            "index, always from a different episode.",
            "- Donor K/V is recomputed at every recipient replan using the fixed donor image "
            "plus current recipient text context and proprioception; this isolates scene "
            "content, not temporal freshness.",
            "- Layer-wise current/donor K/V scale summaries for layers 15-29 are in "
            "`kv_scale_sanity.csv`; shapes, dtypes and finiteness were strictly checked.",
            "",
            "## Provenance and scope",
            "",
            f"- Round-3A frozen parent: `{EXPECTED_PARENT_TAG}` / `{EXPECTED_PARENT_COMMIT}`.",
            f"- Validated Round-3A run code: `{EXPECTED_RUN_COMMIT}`.",
            f"- Round-3B source commit: `{payload['provenance']['git_commit_hash']}`.",
            f"- Checkpoint SHA256: `{payload['provenance']['checkpoint_sha256']}`.",
            f"- Dataset statistics SHA256: `{payload['provenance']['dataset_stats_sha256']}`.",
            f"- Valid state manifest SHA256: `{payload['provenance']['valid_state_bank_manifest_sha256']}`.",
            f"- Prompt cache SHA256: `{payload['provenance']['prompt_context_cache_sha256']}`.",
            f"- Online donor mapping SHA256: `{payload['provenance']['donor_mapping_sha256']}`.",
            f"- Donor observation manifest SHA256: `{payload['provenance']['donor_observation_manifest_sha256']}`.",
            "- One checkpoint, one seed and LIBERO-Spatial only; 10 trials per task.",
            "- Fixed wrong-scene content does not isolate temporal freshness.",
            "- The Round-3B stop rule was applied; no stale/cross-task/key-only/value-only, "
            "patching, sparsity, pruning, additional-suite or additional-checkpoint experiment "
            "is part of this run.",
        ]
    )
    if for_gpt:
        lines.extend(
            [
                "",
                "## Questions for GPT discussion",
                "",
                "1. Is the GO/VERIFY/STOP interpretation appropriately calibrated to the "
                "paired and task-hierarchical uncertainty?",
                "2. How strongly can the result revise the causal interpretation of the "
                "Rounds 1-3A deletion experiments?",
                "3. If classification is VERIFY, which single matched-shape control should be "
                "pre-registered next without broadening into sparsity experiments?",
            ]
        )
    return "\n".join(lines) + "\n"


def aggregate(
    *,
    online_root: Path,
    offline_root: Path,
    valid_manifest_path: Path,
    offline_donor_mapping_path: Path,
    online_donor_mapping_path: Path,
    online_donor_manifest_path: Path,
    self_replacement_result_path: Path,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if bootstrap_samples <= 0 or bootstrap_seed < 0:
        raise ValueError("Bootstrap samples must be positive and seed nonnegative.")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes, online_metadata, task_descriptions = load_online_results(
        online_root=online_root.resolve(),
        donor_mapping_path=online_donor_mapping_path.resolve(),
        donor_manifest_path=online_donor_manifest_path.resolve(),
    )
    analysis = analyze_outcomes(
        outcomes,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    cell_rows = _cell_intervals(
        outcomes,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    task_rows = _task_rows(outcomes, task_descriptions)
    comparison_rows = _comparison_rows(analysis)
    transition_rows = _transition_rows(analysis)
    (
        offline_rows,
        wrong_cross_rows,
        wrong_cross_records,
        replan_rows,
        kv_rows,
        offline_metadata,
    ) = load_offline_results(
        offline_root=offline_root.resolve(),
        valid_manifest_path=valid_manifest_path.resolve(),
        offline_donor_mapping_path=offline_donor_mapping_path.resolve(),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    self_replacement = _read_json(self_replacement_result_path.resolve())
    if not bool(self_replacement.get("passed", False)):
        raise ValueError("Self-replacement machinery control did not pass.")
    if not bool(self_replacement.get("torch_allclose", False)):
        raise ValueError("Self-replacement raw actions are not allclose at the frozen tolerance.")
    if _finite(
        self_replacement.get("max_raw_action_abs_diff"), context="self max action diff"
    ) > 1e-4:
        raise ValueError("Self-replacement max action difference exceeds 1e-4.")

    reference_metadata = online_metadata["late_current_correct"]
    provenance = {
        key: reference_metadata.get(key)
        for key in (
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
            "gpu_model",
            "torch_version",
            "cuda_version",
        )
    }
    provenance.update(
        {
            "round3a_parent_tag": EXPECTED_PARENT_TAG,
            "round3a_parent_commit": EXPECTED_PARENT_COMMIT,
            "validated_round3a_run_commit": EXPECTED_RUN_COMMIT,
            "donor_mapping_path": str(online_donor_mapping_path.resolve()),
            "donor_mapping_sha256": sha256_file(online_donor_mapping_path.resolve()),
            "donor_observation_manifest_path": str(
                online_donor_manifest_path.resolve()
            ),
            "donor_observation_manifest_sha256": sha256_file(
                online_donor_manifest_path.resolve()
            ),
            "offline_donor_mapping_path": str(offline_donor_mapping_path.resolve()),
            "offline_donor_mapping_sha256": sha256_file(
                offline_donor_mapping_path.resolve()
            ),
            "self_replacement_result_path": str(self_replacement_result_path.resolve()),
            "self_replacement_result_sha256": sha256_file(
                self_replacement_result_path.resolve()
            ),
        }
    )
    payload: dict[str, Any] = {
        "artifact_type": "asre_round3b_aggregate",
        "schema_version": 1,
        "condition_protocol": ROUND3B_PROTOCOL,
        "condition_order": list(CONDITION_ORDER),
        "condition_display_names": DISPLAY_NAMES,
        "analysis": analysis,
        "online_condition_summary": cell_rows,
        "online_comparisons": comparison_rows,
        "paired_transitions": transition_rows,
        "task_success": task_rows,
        "offline_condition_metrics": offline_rows,
        "offline_wrong_vs_correct": wrong_cross_rows,
        "offline_replan_stage_metrics": replan_rows,
        "kv_scale_sanity": kv_rows,
        "self_replacement": self_replacement,
        "provenance": provenance,
        "stop_rule": {
            "applied": True,
            "later_stage_experiment_launched": False,
            "prohibited_followups": [
                "stale_kv",
                "cross_task_kv",
                "key_only_replacement",
                "value_only_replacement",
                "residual_patching",
                "head_token_channel_sparsity",
                "pruning",
                "additional_suites_or_checkpoints",
            ],
        },
    }

    _write_csv(output_dir / "online_condition_summary.csv", cell_rows)
    _write_csv(output_dir / "online_contrasts.csv", comparison_rows)
    _write_csv(output_dir / "paired_transitions.csv", transition_rows)
    _write_csv(output_dir / "task_success.csv", task_rows)
    _write_csv(output_dir / "offline_condition_metrics.csv", offline_rows)
    _write_csv(output_dir / "offline_wrong_vs_correct.csv", wrong_cross_rows)
    _write_csv(output_dir / "offline_wrong_vs_correct_per_sample.csv", wrong_cross_records)
    _write_csv(output_dir / "offline_replan_stage_metrics.csv", replan_rows)
    _write_csv(output_dir / "kv_scale_sanity.csv", kv_rows)
    atomic_write_json(output_dir / "round3b_summary.json", payload)
    _write_text(output_dir / "round3b_summary.md", _summary_markdown(payload, for_gpt=False))
    _write_text(
        output_dir / "result_summary_for_gpt.md",
        _summary_markdown(payload, for_gpt=True),
    )
    aggregate_metadata = {
        "artifact_type": "asre_round3b_aggregate_metadata",
        "schema_version": 1,
        "condition_protocol": ROUND3B_PROTOCOL,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "input_roots": {
            "online": str(online_root.resolve()),
            "offline": str(offline_root.resolve()),
        },
        "provenance": provenance,
        "input_run_commits": {
            "online": {
                condition: metadata.get("git_commit_hash")
                for condition, metadata in online_metadata.items()
            },
            "offline": {
                condition: metadata.get("git_commit_hash")
                for condition, metadata in offline_metadata.items()
            },
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "aggregate_metadata.json"
        },
    }
    atomic_write_json(output_dir / "aggregate_metadata.json", aggregate_metadata)
    return payload


def main() -> None:
    args = _parse_args()
    payload = aggregate(
        online_root=args.online_root,
        offline_root=args.offline_root,
        valid_manifest_path=args.valid_manifest,
        offline_donor_mapping_path=args.offline_donor_mapping,
        online_donor_mapping_path=args.online_donor_mapping,
        online_donor_manifest_path=args.online_donor_manifest,
        self_replacement_result_path=args.self_replacement_result,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(payload["analysis"], indent=2, sort_keys=True))
    print(f"Round-3B aggregate complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
