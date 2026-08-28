"""Strict paired online/offline aggregation for ASRE Stage-2 Round-4A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4A_PROTOCOL,
    atomic_write_json,
    build_round4a_conditions,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round2.metrics import (  # noqa: E402
    CONTINUOUS_ACTION_DIMENSIONS,
    compute_round2_metrics,
)
from experiments.asre_diagnosis.round3b.outcome_statistics import (  # noqa: E402
    comparison_statistics,
)
from experiments.asre_diagnosis.round4a.classification import (  # noqa: E402
    classify_axis,
    recommend_axis,
)
from experiments.asre_diagnosis.round4a.launch_wave import (  # noqa: E402
    FULL_TASK_IDS,
    FULL_TRIALS,
    WAVE_CONDITIONS,
)
from experiments.asre_diagnosis.round4a.masks import (  # noqa: E402
    load_mask_manifest,
)
from experiments.asre_diagnosis.round4a.provenance import (  # noqa: E402
    Round4AProvenance,
    load_round4a_provenance,
)


CONDITIONS = tuple(condition.name for condition in build_round4a_conditions(30))
HEAD_CONDITIONS = ("head50_seed1", "head50_seed2", "head50_seed3")
TOKEN_CONDITIONS = ("token50_seed1", "token50_seed2", "token50_seed3")
HYBRID_CONDITIONS = HEAD_CONDITIONS + TOKEN_CONDITIONS
DISPLAY_NAMES = {
    "current_all": "Current",
    "wrong_all": "Wrong",
    "head50_seed1": "H50-1",
    "head50_seed2": "H50-2",
    "head50_seed3": "H50-3",
    "token50_seed1": "T50-1",
    "token50_seed2": "T50-2",
    "token50_seed3": "T50-3",
}
SCALAR_METRICS = (
    "executed_prefix_norm_rms",
    "full_chunk_norm_rms_0_31",
    "executed_prefix_cosine_similarity",
    "translation_norm_rms",
    "rotation_norm_rms",
    "executed_prefix_gripper_flip_rate",
    "full_horizon_gripper_flip_rate",
)
DIMENSION_METRICS = tuple(
    f"dimension_rms_{name}" for name in CONTINUOUS_ACTION_DIMENSIONS
)
OFFLINE_METRICS = SCALAR_METRICS + DIMENSION_METRICS


def validate_paired_outcome_keys(
    outcomes: Mapping[str, Mapping[tuple[str, int, int], int]],
) -> list[tuple[str, int, int]]:
    if set(outcomes) != set(CONDITIONS):
        raise ValueError(
            f"Round-4A outcomes require exactly {list(CONDITIONS)}, got {sorted(outcomes)}."
        )
    reference = set(outcomes["current_all"])
    if len(reference) != 100:
        raise ValueError(f"Round-4A requires 100 current endpoint keys, got {len(reference)}.")
    mismatch = {
        condition: {
            "missing": sorted(reference - set(values)),
            "extra": sorted(set(values) - reference),
        }
        for condition, values in outcomes.items()
        if set(values) != reference
    }
    if mismatch:
        raise ValueError(f"Online task/trial keys are not exactly paired: {mismatch}.")
    for condition, values in outcomes.items():
        if any(value not in {0, 1, False, True} for value in values.values()):
            raise ValueError(f"{condition} contains non-binary online outcomes.")
    return sorted(reference)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(f"Expected object at {path}:{line_number}.")
                records.append(payload)
    return records


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    os.replace(temporary, path)


def _write_text(path: Path, content: str) -> None:
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
        handle.write(content)
        if not content.endswith("\n"):
            handle.write("\n")
        handle.flush()
    os.replace(temporary, path)


def _stable_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _cluster_ci(
    values: Sequence[float],
    clusters: Sequence[tuple[str, int, int]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.size == 0 or numeric.size != len(clusters):
        raise ValueError("Episode-cluster bootstrap requires aligned values.")
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for cluster, value in zip(clusters, numeric):
        grouped[cluster].append(float(value))
    groups = [np.asarray(grouped[key]) for key in sorted(grouped)]
    sums = np.asarray([group.sum() for group in groups])
    counts = np.asarray([group.size for group in groups])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(samples, len(groups)))
    estimates = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def _condition_root(
    condition: str, wave1_root: Path, wave2_root: Path
) -> Path:
    index = CONDITIONS.index(condition)
    return (wave1_root if index in WAVE_CONDITIONS[1] else wave2_root) / condition


def load_online(
    *,
    wave1_root: Path,
    wave2_root: Path,
    provenance: Round4AProvenance,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[
    dict[str, dict[tuple[str, int, int], int]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, str],
]:
    conditions = build_round4a_conditions(30)
    for wave, root in ((1, wave1_root), (2, wave2_root)):
        summary = _read_json(root / "launcher_summary.json")
        expected = {
            "schema_version": 1,
            "protocol": ROUND4A_PROTOCOL,
            "mode": "full",
            "wave": wave,
            "all_succeeded": True,
            "interrupted": False,
            "task_ids": list(FULL_TASK_IDS),
            "num_trials": FULL_TRIALS,
            "git_commit_hash": provenance.git_commit_hash,
            "provenance": provenance.identity_dict(),
        }
        bad = {
            key: {"observed": summary.get(key), "expected": value}
            for key, value in expected.items()
            if summary.get(key) != value
        }
        if bad:
            raise ValueError(f"Online wave {wave} summary mismatch: {json.dumps(bad)}")

    outcomes: dict[str, dict[tuple[str, int, int], int]] = {}
    task_descriptions: dict[int, str] = {}
    task_rows: list[dict[str, Any]] = []
    for condition_index, condition_config in enumerate(conditions):
        condition = condition_config.name
        root = _condition_root(condition, wave1_root, wave2_root)
        metadata = _read_json(root / "run_metadata.json")
        expected_metadata = {
            "status": "completed",
            "condition_protocol": ROUND4A_PROTOCOL,
            "diagnosis_condition": condition,
            "git_commit_hash": provenance.git_commit_hash,
            "checkpoint_sha256": provenance.checkpoint_sha256,
            "dataset_stats_sha256": provenance.dataset_stats_sha256,
            "valid_state_bank_manifest_sha256": provenance.valid_manifest_sha256,
            "prompt_context_cache_sha256": provenance.prompt_context_cache_sha256,
            "donor_mapping_sha256": provenance.online_donor_mapping_sha256,
            "hybrid_mask_manifest_sha256": provenance.mask_manifest_sha256,
            "token_mask_manifest_sha256": provenance.token_mask_manifest_sha256,
            "head_mask_manifest_sha256": provenance.head_mask_manifest_sha256,
            "number_of_trials": 10,
            "task_ids": list(range(10)),
            "action_horizon": 32,
            "number_of_inference_steps": 10,
            "replan_steps": 10,
        }
        bad = {
            key: {"observed": metadata.get(key), "expected": value}
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if bad:
            raise ValueError(f"Online metadata mismatch for {condition}: {json.dumps(bad)}")
        expected_condition_config = {
            "hybrid_axis": condition_config.hybrid_axis,
            "hybrid_mask_seed": condition_config.mask_seed,
        }
        observed_condition_config = metadata.get("condition_config", {})
        config_bad = {
            key: {
                "observed": observed_condition_config.get(key),
                "expected": value,
            }
            for key, value in expected_condition_config.items()
            if observed_condition_config.get(key) != value
        }
        if config_bad:
            raise ValueError(
                f"Online condition metadata mismatch for {condition}: "
                f"{json.dumps(config_bad)}"
            )
        files = sorted((root / "libero_spatial").glob("gpu*_task*_results.json"))
        if len(files) != 10:
            raise ValueError(f"{condition} must have ten task result files.")
        condition_outcomes = {}
        for path in files:
            result = _read_json(path)
            task_id = int(result["task_id"])
            description = str(result["task_description"])
            if task_id in task_descriptions and task_descriptions[task_id] != description:
                raise ValueError(f"Task description drift for task {task_id}.")
            task_descriptions[task_id] = description
            successes = set(int(value) for value in result["success_episodes"])
            failures = set(int(value) for value in result["failure_episodes"])
            if successes & failures or successes | failures != set(range(10)):
                raise ValueError(f"Paired trial coverage failure: {path}")
            for trial in range(10):
                condition_outcomes[("libero_spatial", task_id, trial)] = int(
                    trial in successes
                )
            task_rows.append(
                {
                    "condition": condition,
                    "display_name": DISPLAY_NAMES[condition],
                    "axis": condition_config.hybrid_axis or "endpoint",
                    "mask_seed": condition_config.mask_seed,
                    "task_id": task_id,
                    "task_description": description,
                    "successes": len(successes),
                    "trials": 10,
                    "success_rate": len(successes) / 10.0,
                }
            )
        if len(condition_outcomes) != 100:
            raise ValueError(f"{condition} does not contain 100 paired outcomes.")
        outcomes[condition] = condition_outcomes

    keys = validate_paired_outcome_keys(outcomes)
    episode_keys = [(suite, task, trial) for suite, task, trial in keys]
    arrays = {
        condition: np.asarray([outcomes[condition][key] for key in keys], dtype=np.int8)
        for condition in CONDITIONS
    }
    zero = np.zeros(len(keys), dtype=np.int8)
    condition_rows = []
    contrast_rows = []
    transition_rows = []
    for condition in CONDITIONS:
        rate_stats = comparison_statistics(
            episode_keys,
            zero,
            arrays[condition],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            seed_label=f"rate_{condition}",
        )
        current_stats = comparison_statistics(
            episode_keys,
            arrays["current_all"],
            arrays[condition],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            seed_label=f"current_to_{condition}",
        )
        wrong_stats = comparison_statistics(
            episode_keys,
            arrays["wrong_all"],
            arrays[condition],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            seed_label=f"wrong_to_{condition}",
        )
        condition_rows.append(
            {
                "condition": condition,
                "display_name": DISPLAY_NAMES[condition],
                "successes": int(arrays[condition].sum()),
                "trials": len(keys),
                "success_rate": float(arrays[condition].mean()),
                "paired_bootstrap_ci_low": rate_stats["paired_ci_low"],
                "paired_bootstrap_ci_high": rate_stats["paired_ci_high"],
                "task_hierarchical_ci_low": rate_stats["task_hierarchical_ci_low"],
                "task_hierarchical_ci_high": rate_stats["task_hierarchical_ci_high"],
                "delta_vs_current": current_stats["delta_success_rate"],
                "retained_behavior_vs_wrong": wrong_stats["delta_success_rate"],
            }
        )
        for reference, stats in (("current_all", current_stats), ("wrong_all", wrong_stats)):
            row = {
                "comparison": f"{condition}_minus_{reference}",
                "reference_condition": reference,
                "target_condition": condition,
                **stats,
            }
            contrast_rows.append(row)
            transition_rows.append(
                {
                    key: row[key]
                    for key in (
                        "comparison",
                        "reference_condition",
                        "target_condition",
                        "reference_success_to_target_success",
                        "reference_success_to_target_failure",
                        "reference_failure_to_target_success",
                        "reference_failure_to_target_failure",
                        "induced_failures",
                        "rescued_successes",
                        "discordant_pairs",
                        "mcnemar_exact_p_value",
                    )
                }
            )
    current_task = {
        int(row["task_id"]): float(row["success_rate"])
        for row in task_rows
        if row["condition"] == "current_all"
    }
    for row in task_rows:
        row["delta_vs_current"] = float(row["success_rate"]) - current_task[
            int(row["task_id"])
        ]
    return (
        outcomes,
        condition_rows,
        contrast_rows,
        transition_rows,
        task_rows,
        task_descriptions,
    )


def _load_actions(path: Path, expected_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        ids = [str(value) for value in payload["sample_ids"].tolist()]
        raw = np.asarray(payload["raw_actions"], dtype=np.float32)
        executed = np.asarray(payload["executed_actions"], dtype=np.float32)
    if ids != list(expected_ids) or raw.shape != (499, 32, 7) or executed.shape != raw.shape:
        raise ValueError(f"Offline action artifact is incompatible: {path}")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(executed)):
        raise ValueError(f"Offline actions contain NaN/Inf: {path}")
    return raw, executed


def load_offline(
    *,
    offline_root: Path,
    provenance: Round4AProvenance,
    online_outcomes: Mapping[str, Mapping[tuple[str, int, int], int]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    valid = _read_json(provenance.valid_manifest_path)
    valid_ids = [str(value) for value in valid["valid_sample_ids"]]
    if len(valid_ids) != 499 or len(set(valid_ids)) != 499:
        raise ValueError("Offline valid state identity must contain 499 ordered IDs.")
    records_by_condition = {}
    raw_by_condition = {}
    executed_by_condition = {}
    metadata_by_condition = {}
    condition_specs = {
        condition.name: condition for condition in build_round4a_conditions(30)
    }
    for condition in CONDITIONS:
        directory = offline_root / condition
        metadata = _read_json(directory / "run_metadata.json")
        condition_spec = condition_specs[condition]
        expected = {
            "artifact_type": "asre_round4a_offline_state_bank_replay",
            "status": "complete",
            "condition_protocol": ROUND4A_PROTOCOL,
            "diagnosis_condition": condition,
            "git_commit_hash": provenance.git_commit_hash,
            "valid_state_bank_manifest_sha256": provenance.valid_manifest_sha256,
            "offline_donor_mapping_sha256": provenance.offline_donor_mapping_sha256,
            "hybrid_mask_manifest_sha256": provenance.mask_manifest_sha256,
            "token_mask_manifest_sha256": provenance.token_mask_manifest_sha256,
            "head_mask_manifest_sha256": provenance.head_mask_manifest_sha256,
            "hybrid_axis": condition_spec.hybrid_axis,
            "hybrid_mask_seed": condition_spec.mask_seed,
            "num_samples": 499,
            "completed_samples": 499,
        }
        bad = {
            key: {"observed": metadata.get(key), "expected": value}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if bad:
            raise ValueError(f"Offline metadata mismatch for {condition}: {json.dumps(bad)}")
        records = _read_jsonl(directory / "per_sample.jsonl")
        if [str(record["sample_id"]) for record in records] != valid_ids:
            raise ValueError(f"Offline state ordering mismatch for {condition}.")
        raw, executed = _load_actions(directory / "actions.npz", valid_ids)
        if sha256_file(directory / "actions.npz") != metadata["actions_sha256"]:
            raise ValueError(f"Offline action digest mismatch for {condition}.")
        records_by_condition[condition] = records
        raw_by_condition[condition] = raw
        executed_by_condition[condition] = executed
        metadata_by_condition[condition] = metadata
    action_std = np.asarray(
        metadata_by_condition["current_all"]["action_global_std"], dtype=np.float64
    )
    for condition in CONDITIONS[1:]:
        if metadata_by_condition[condition]["action_global_std"] != metadata_by_condition[
            "current_all"
        ]["action_global_std"]:
            raise ValueError("Offline action normalization drifted across conditions.")

    per_state_rows = []
    metric_rows = []
    replan_rows = []
    episode_condition_deviation: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for condition in CONDITIONS:
        for reference in ("current_all", "wrong_all"):
            local_rows = []
            for index, source_record in enumerate(records_by_condition[condition]):
                metrics = compute_round2_metrics(
                    baseline_raw=raw_by_condition[reference][index],
                    diagnosis_raw=raw_by_condition[condition][index],
                    baseline_executed=executed_by_condition[reference][index],
                    diagnosis_executed=executed_by_condition[condition][index],
                    action_std=action_std,
                    executed_prefix_length=10,
                )
                row = {
                    "condition": condition,
                    "reference": reference,
                    "sample_id": source_record["sample_id"],
                    "task_suite": source_record["task_suite"],
                    "task_id": int(source_record["task_id"]),
                    "task_description": source_record["task_description"],
                    "episode_id": int(source_record["episode_id"]),
                    "replan_id": int(source_record["replan_id"]),
                    **{
                        metric: float(metrics[metric]) for metric in SCALAR_METRICS
                    },
                    **{
                        f"dimension_rms_{name}": float(value)
                        for name, value in zip(
                            CONTINUOUS_ACTION_DIMENSIONS,
                            metrics["executed_prefix_norm_rms_by_dimension"],
                        )
                    },
                }
                local_rows.append(row)
                per_state_rows.append(row)
                if reference == "current_all" and condition in HYBRID_CONDITIONS:
                    episode_condition_deviation[
                        (condition, int(row["task_id"]), int(row["episode_id"]))
                    ].append(float(row["executed_prefix_norm_rms"]))
            clusters = [
                (str(row["task_suite"]), int(row["task_id"]), int(row["episode_id"]))
                for row in local_rows
            ]
            for metric in OFFLINE_METRICS:
                values = [float(row[metric]) for row in local_rows]
                low, high = _cluster_ci(
                    values,
                    clusters,
                    samples=bootstrap_samples,
                    seed=_stable_seed(bootstrap_seed, condition, reference, metric),
                )
                metric_rows.append(
                    {
                        "condition": condition,
                        "reference": reference,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "episode_cluster_ci_low": low,
                        "episode_cluster_ci_high": high,
                        "num_states": len(values),
                        "num_episode_clusters": len(set(clusters)),
                    }
                )
            for replan_id in range(5):
                subset = [row for row in local_rows if int(row["replan_id"]) == replan_id]
                replan_rows.append(
                    {
                        "condition": condition,
                        "reference": reference,
                        "replan_id": replan_id,
                        "num_states": len(subset),
                        **{
                            metric: float(np.mean([row[metric] for row in subset]))
                            for metric in OFFLINE_METRICS
                        },
                    }
                )

    counterexamples = []
    condition_scatter = []
    success_by_condition = {
        condition: float(np.mean(list(online_outcomes[condition].values())))
        for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        metric = next(
            row
            for row in metric_rows
            if row["condition"] == condition
            and row["reference"] == "current_all"
            and row["metric"] == "executed_prefix_norm_rms"
        )
        condition_scatter.append(
            {
                "condition": condition,
                "display_name": DISPLAY_NAMES[condition],
                "offline_executed_prefix_norm_rms_vs_current": metric["mean"],
                "online_success_rate": success_by_condition[condition],
            }
        )
    for condition in HYBRID_CONDITIONS:
        episodes = []
        for (candidate, task_id, episode_id), values in episode_condition_deviation.items():
            if candidate != condition:
                continue
            key = ("libero_spatial", task_id, episode_id)
            episodes.append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "offline_mean_executed_prefix_norm_rms": float(np.mean(values)),
                    "online_success": int(online_outcomes[condition][key]),
                }
            )
        successes = [episode for episode in episodes if episode["online_success"] == 1]
        failures = [episode for episode in episodes if episode["online_success"] == 0]
        pairs = []
        for success in successes:
            for failure in failures:
                gap = abs(
                    success["offline_mean_executed_prefix_norm_rms"]
                    - failure["offline_mean_executed_prefix_norm_rms"]
                )
                pairs.append((gap, success, failure))
        for rank, (gap, success, failure) in enumerate(sorted(pairs, key=lambda item: item[0])[:5], start=1):
            counterexamples.append(
                {
                    "condition": condition,
                    "rank_within_condition": rank,
                    "offline_deviation_absolute_gap": gap,
                    "success_task_id": success["task_id"],
                    "success_episode_id": success["episode_id"],
                    "success_offline_deviation": success[
                        "offline_mean_executed_prefix_norm_rms"
                    ],
                    "failure_task_id": failure["task_id"],
                    "failure_episode_id": failure["episode_id"],
                    "failure_offline_deviation": failure[
                        "offline_mean_executed_prefix_norm_rms"
                    ],
                    "online_success_difference": 1,
                    "interpretation": (
                        "Similar offline deviation, different online behavior; "
                        "offline distance is not a behavioral surrogate."
                    ),
                }
            )
    if not counterexamples:
        counterexamples.append(
            {
                "condition": "none",
                "rank_within_condition": None,
                "offline_deviation_absolute_gap": None,
                "interpretation": (
                    "No within-condition success/failure pair existed; offline metrics "
                    "remain descriptive and are not treated as behavioral surrogates."
                ),
            }
        )
    return metric_rows, per_state_rows, replan_rows, counterexamples, condition_scatter


def _axis_analysis(
    condition_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rates = {str(row["condition"]): float(row["success_rate"]) for row in condition_rows}
    task_rates = {
        condition: {
            int(row["task_id"]): float(row["success_rate"])
            for row in task_rows
            if row["condition"] == condition
        }
        for condition in CONDITIONS
    }
    token = classify_axis(
        axis="token",
        mask_success_rates=[rates[name] for name in TOKEN_CONDITIONS],
        current_success_rate=rates["current_all"],
        wrong_success_rate=rates["wrong_all"],
        mask_task_success=[task_rates[name] for name in TOKEN_CONDITIONS],
        current_task_success=task_rates["current_all"],
    )
    head = classify_axis(
        axis="head",
        mask_success_rates=[rates[name] for name in HEAD_CONDITIONS],
        current_success_rate=rates["current_all"],
        wrong_success_rate=rates["wrong_all"],
        mask_task_success=[task_rates[name] for name in HEAD_CONDITIONS],
        current_task_success=task_rates["current_all"],
    )
    recommendation = recommend_axis(token, head)
    robustness_rows = []
    for analysis in (head, token):
        robustness_rows.append(
            {
                key: analysis[key]
                for key in (
                    "axis",
                    "classification",
                    "mean_success_rate",
                    "median_success_rate",
                    "min_success_rate",
                    "max_success_rate",
                    "range_success_rate",
                    "variance_success_rate",
                    "current_success_rate",
                    "wrong_success_rate",
                    "current_wrong_gap",
                    "median_normalized_recovery",
                    "near_wrong_mask_count",
                    "repeated_catastrophic_task_collapse",
                )
            }
            | {"catastrophic_event_count": len(analysis["catastrophic_task_events"])}
        )
    return token, head, recommendation, robustness_rows


def _rate(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _report(payload: Mapping[str, Any]) -> str:
    online = payload["online_condition_summary"]
    token = payload["axis_analysis"]["token"]
    head = payload["axis_analysis"]["head"]
    recommendation = payload["recommendation"]
    machinery = payload["machinery"]
    current_identity = machinery["controls"]["A_current_endpoint_identity"]
    wrong_identity = machinery["controls"]["B_wrong_endpoint_identity"]["action"]
    self_identity = machinery["controls"][
        "C_arbitrary_mask_self_replacement_identity"
    ]
    layout = payload["mask_manifest"]["runtime_layout"]
    masks = payload["mask_manifest"]["conditions"]
    lines = [
        "# Fast-WAM ASRE Stage 2 — Round 4A Final Report",
        "",
        "Round 4A is complete and aggregation has stopped. Offline distances below are descriptive, not behavioral surrogates.",
        "",
        "## 1. Exact source files changed",
        "",
        *[f"- `{path}`" for path in payload["source_files_changed"]],
        "",
        "## 2. Current Git state",
        "",
        f"- Branch: `{payload['git_state']['current_branch']}`",
        f"- Commit: `{payload['provenance']['git_commit_hash']}`",
        f"- Round-3B parent: `{payload['provenance']['round3b_parent_tag']}` / `{payload['provenance']['round3b_parent_commit']}`",
        f"- G0 parent: `{payload['provenance']['g0_parent_tag']}` / `{payload['provenance']['g0_parent_commit']}`",
        f"- Formal source clean: `{payload['formal_source_clean']}`",
        "",
        "## 3. Core Fast-WAM model change",
        "",
        "The optional `infer_action` cache-mixing arguments were added because matched token/head K/V composition must occur after real cache construction. Calls that omit them retain the prior path, covered by regression tests.",
        "",
        "## 4. Machinery identity tests",
        "",
        f"- Status: `{machinery['status']}`; all finite `[32,7]`: `{machinery['all_action_outputs_shape_32x7_and_finite']}`",
        f"- Current endpoint: max `{current_identity['max_raw_action_abs_diff']:.8g}`, mean `{current_identity['mean_raw_action_abs_diff']:.8g}`, allclose `{current_identity['torch_allclose']}`.",
        f"- Wrong endpoint: max `{wrong_identity['max_raw_action_abs_diff']:.8g}`, mean `{wrong_identity['mean_raw_action_abs_diff']:.8g}`, allclose `{wrong_identity['torch_allclose']}`.",
        f"- Token self-replacement: max `{self_identity['token50_seed1']['max_raw_action_abs_diff']:.8g}`, mean `{self_identity['token50_seed1']['mean_raw_action_abs_diff']:.8g}`, allclose `{self_identity['token50_seed1']['torch_allclose']}`.",
        f"- Head self-replacement: max `{self_identity['head50_seed1']['max_raw_action_abs_diff']:.8g}`, mean `{self_identity['head50_seed1']['mean_raw_action_abs_diff']:.8g}`, allclose `{self_identity['head50_seed1']['torch_allclose']}`.",
        f"- Token/head K/V integrity gates: `{machinery['controls']['D_token_mask_integrity']['manifest_exact_match']}` / `{machinery['controls']['E_head_mask_integrity']['manifest_exact_match']}`.",
        "",
        "## 5. Runtime K/V representation shapes",
        "",
        f"- Late-layer cache shapes: `{json.dumps(layout['cache_shapes_by_layer'], sort_keys=True)}`",
        "",
        "## 6. Action-visible token definition and count",
        "",
        "A token is action-visible iff at least one runtime action query has an enabled attention-mask entry for that video key position.",
        f"Count: `{layout['action_visible_token_count']}`; indices: `{layout['action_visible_token_indices']}`.",
        f"View stratification: `{layout['view_stratified']}` — {layout['view_stratification_reason']}",
        "",
        "## 7. Runtime heads",
        "",
        f"- `num_heads={layout['num_heads']}`, `head_dim={layout['head_dim']}`.",
        "",
        "## 8. Exact frozen masks",
        "",
        f"- Combined: `{payload['provenance']['mask_manifest_path']}` (`{payload['provenance']['mask_manifest_sha256']}`)",
        f"- Token: `{payload['provenance']['token_mask_manifest_path']}` (`{payload['provenance']['token_mask_manifest_sha256']}`)",
        f"- Head: `{payload['provenance']['head_mask_manifest_path']}` (`{payload['provenance']['head_mask_manifest_sha256']}`)",
        f"- Seeds: `{payload['mask_manifest']['mask_seeds']}`; rounding: `{payload['mask_manifest']['rounding_rule']}`.",
        *[
            f"- `{condition}` retained token indices: `{masks[condition]['retained_current_token_indices']}`; per-view quota: `{masks[condition]['retained_current_token_quota_by_view']}`."
            for condition in TOKEN_CONDITIONS
        ],
        *[
            f"- `{condition}` retained heads by layer: `{json.dumps(masks[condition]['retained_current_heads_by_layer'], sort_keys=True)}`."
            for condition in HEAD_CONDITIONS
        ],
        "",
        "## 9. Smoke tests",
        "",
        *[
            f"- Wave {wave}: all succeeded `{summary['all_succeeded']}`, conditions `{summary['expected_conditions']}`."
            for wave, summary in payload["smoke_summaries"].items()
        ],
        "",
        "## 10. Full eight-condition online success",
        "",
        "| Condition | Success | Paired 95% CI | Task-hierarchical 95% CI | Δ current |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row['condition']} | {_rate(row['success_rate'])} | {_rate(row['paired_bootstrap_ci_low'])}–{_rate(row['paired_bootstrap_ci_high'])} | {_rate(row['task_hierarchical_ci_low'])}–{_rate(row['task_hierarchical_ci_high'])} | {100*row['delta_vs_current']:+.1f} pp |"
            for row in online
        ],
        "",
        "## 11. Paired and task-hierarchical uncertainty",
        "",
        "All contrasts use identical task/trial keys. The artifact `online_contrasts.csv` contains paired percentile CIs, task-hierarchical CIs, transition cells, and exact McNemar p-values.",
        "",
        "## 12. Task-level outcomes",
        "",
        "See `task_success.csv` and Figure C. Catastrophic-task events are explicitly stored under each axis analysis without semantic storytelling.",
        "",
        "## 13. Offline metrics",
        "",
        "Each condition is compared with both current and wrong for executed-prefix normalized RMS, full-chunk RMS, cosine, translation/rotation RMS, gripper flips, per-dimension RMS, and replan summaries. See `offline_metric_summary.csv`, `offline_per_state.csv`, and `offline_replan_summary.csv`.",
        "",
        "## 14. Token-50 robustness",
        "",
        f"- Rates: `{[_rate(value) for value in token['mask_success_rates']]}`; median `{_rate(token['median_success_rate'])}`; range `{100*token['range_success_rate']:.1f} pp`; classification `{token['classification']}`.",
        "",
        "## 15. Head-50 robustness",
        "",
        f"- Rates: `{[_rate(value) for value in head['mask_success_rates']]}`; median `{_rate(head['median_success_rate'])}`; range `{100*head['range_success_rate']:.1f} pp`; classification `{head['classification']}`.",
        "",
        "## 16. TOKEN classification",
        "",
        f"**{token['classification']}**",
        "",
        "## 17. HEAD classification",
        "",
        f"**{head['classification']}**",
        "",
        "## 18. Recommended primary axis",
        "",
        f"`{recommendation['recommended_primary_axis']}` — {recommendation['rationale']}",
        "",
        "## 19. Recommended next experiment",
        "",
        recommendation["recommended_next_experiment"] + ". It was not launched.",
        "",
        "## 20. Next-step category",
        "",
        f"**{recommendation['next_step_category']}**",
        "",
        "## 21. Artifact paths",
        "",
        *[f"- `{path}`" for path in payload["artifact_paths"]],
        "",
        "## 22. Stop-rule confirmation",
        "",
        "No token25/token75/head25/head75, ranked mask, feature/channel, subspace, cross-suite, RoboTwin, or additional-checkpoint experiment was launched.",
    ]
    if token["classification"] == "WEAK" and head["classification"] == "WEAK":
        lines.extend(
            [
                "",
                "Neither Token nor Head shows meaningful 50% sparsity under the registered criteria.",
            ]
        )
    return "\n".join(lines) + "\n"


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    provenance = load_round4a_provenance(
        preflight_report_path=args.preflight_report,
        machinery_report_path=args.machinery_report,
        mask_manifest_path=args.mask_manifest,
    )
    mask_manifest = load_mask_manifest(
        path=provenance.mask_manifest_path,
        expected_sha256=provenance.mask_manifest_sha256,
    )
    preflight = _read_json(provenance.preflight_report_path)
    machinery = _read_json(provenance.machinery_report_path)
    outcomes, condition_rows, contrast_rows, transition_rows, task_rows, _ = load_online(
        wave1_root=args.online_wave1_root.resolve(),
        wave2_root=args.online_wave2_root.resolve(),
        provenance=provenance,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    metric_rows, per_state_rows, replan_rows, counterexamples, scatter_rows = load_offline(
        offline_root=args.offline_root.resolve(),
        provenance=provenance,
        online_outcomes=outcomes,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    token, head, recommendation, robustness_rows = _axis_analysis(
        condition_rows, task_rows
    )
    smoke_summaries = {
        "1": _read_json(args.smoke_wave1_summary.resolve()),
        "2": _read_json(args.smoke_wave2_summary.resolve()),
    }
    for wave, summary in smoke_summaries.items():
        if not summary.get("all_succeeded") or summary.get("protocol") != ROUND4A_PROTOCOL:
            raise ValueError(f"Smoke wave {wave} did not pass.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_files = {
        "online_condition_summary.csv": condition_rows,
        "online_contrasts.csv": contrast_rows,
        "paired_transitions.csv": transition_rows,
        "task_success.csv": task_rows,
        "axis_robustness.csv": robustness_rows,
        "offline_metric_summary.csv": metric_rows,
        "offline_per_state.csv": per_state_rows,
        "offline_replan_summary.csv": replan_rows,
        "offline_online_counterexamples.csv": counterexamples,
        "offline_online_condition_summary.csv": scatter_rows,
    }
    for name, rows in output_files.items():
        _write_csv(output / name, rows)

    source_files = [
        "configs/sim_libero.yaml",
        "src/fastwam/models/wan22/fastwam.py",
        "src/fastwam/models/wan22/video_cache_replacement.py",
        "experiments/libero/eval_libero_single.py",
        "experiments/asre_diagnosis/common.py",
        "experiments/asre_diagnosis/round3b/replay_state_bank.py",
        "experiments/asre_diagnosis/tests/test_round3b_model_replacement.py",
        "experiments/asre_diagnosis/round4a/run_full_experiment.sh",
        *[
            str(path.relative_to(PROJECT_ROOT))
            for path in sorted(
                (PROJECT_ROOT / "experiments/asre_diagnosis/round4a").glob("*.py")
            )
        ],
        *[
            str(path.relative_to(PROJECT_ROOT))
            for path in sorted(
                (PROJECT_ROOT / "experiments/asre_diagnosis/tests").glob(
                    "test_round4a*.py"
                )
            )
        ],
    ]
    artifact_paths = [
        str(provenance.preflight_report_path),
        str(provenance.machinery_report_path),
        str(provenance.mask_manifest_path),
        str(provenance.token_mask_manifest_path),
        str(provenance.head_mask_manifest_path),
        str(args.smoke_wave1_summary.resolve()),
        str(args.smoke_wave2_summary.resolve()),
        str(args.online_wave1_root.resolve()),
        str(args.online_wave2_root.resolve()),
        str(args.offline_root.resolve()),
        str(output),
    ]
    payload = {
        "artifact_type": "asre_round4a_summary",
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "status": "complete",
        "created_at": now_iso(),
        "provenance": provenance.identity_dict(),
        "git_state": preflight["git"],
        "formal_source_clean": True,
        "source_files_changed": source_files,
        "mask_manifest": mask_manifest,
        "machinery": machinery,
        "smoke_summaries": smoke_summaries,
        "online_condition_summary": condition_rows,
        "online_contrasts": contrast_rows,
        "paired_transitions": transition_rows,
        "task_success": task_rows,
        "axis_analysis": {"token": token, "head": head},
        "recommendation": recommendation,
        "offline_metric_summary": metric_rows,
        "offline_replan_summary": replan_rows,
        "offline_online_counterexamples": counterexamples,
        "offline_online_condition_summary": scatter_rows,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "interval": "percentile 95%",
            "paired_unit": "matched task/trial",
            "task_hierarchical_unit": "tasks then matched trials",
            "offline_cluster_unit": "task/episode across saved replans",
        },
        "artifact_paths": artifact_paths,
        "later_stage_launched": False,
        "stop_rule_applied": True,
    }
    atomic_write_json(output / "round4a_summary.json", payload)
    report = _report(payload)
    _write_text(output / "round4a_summary.md", report)
    _write_text(output / "result_summary_for_gpt.md", report)
    aggregate_metadata = {
        "artifact_type": "asre_round4a_aggregate_metadata",
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "status": "complete",
        "created_at": now_iso(),
        "source_git_commit": provenance.git_commit_hash,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "inputs": provenance.identity_dict(),
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
        "later_stage_launched": False,
    }
    atomic_write_json(output / "aggregate_metadata.json", aggregate_metadata)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--machinery-report", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--smoke-wave1-summary", type=Path, required=True)
    parser.add_argument("--smoke-wave2-summary", type=Path, required=True)
    parser.add_argument("--online-wave1-root", type=Path, required=True)
    parser.add_argument("--online-wave2-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=4104)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = aggregate(args)
    print(json.dumps(payload["recommendation"], indent=2, sort_keys=True))
    print(f"Round-4A aggregate complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
