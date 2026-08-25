"""Aggregate offline deviations and paired online LIBERO outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import atomic_write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def _write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _condition_metadata(online_root: Path) -> dict[str, dict[str, Any]]:
    metadata = {}
    for path in online_root.glob("*/run_metadata.json"):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        condition = str(record["diagnosis_condition"])
        record["condition_dir"] = str(path.parent)
        metadata[condition] = record
    if not metadata:
        raise FileNotFoundError(f"No condition run_metadata.json files under {online_root}.")
    return metadata


def _load_online_episodes(
    condition_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[tuple[str, int, int], int]], dict[str, dict[int, list[int]]]]:
    episodes: dict[str, dict[tuple[str, int, int], int]] = defaultdict(dict)
    task_outcomes: dict[str, dict[int, list[int]]] = defaultdict(dict)
    for condition, metadata in condition_metadata.items():
        condition_dir = Path(metadata["condition_dir"])
        for result_path in condition_dir.glob("**/gpu*_task*_results.json"):
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            suite = str(result["task_suite"])
            task_id = int(result["task_id"])
            total = int(result["total_episodes"])
            successes = {int(value) for value in result.get("success_episodes", [])}
            failures = {int(value) for value in result.get("failure_episodes", [])}
            if successes & failures or successes | failures != set(range(total)):
                raise ValueError(f"Incomplete/inconsistent episode IDs in {result_path}.")
            outcomes = [int(episode_id in successes) for episode_id in range(total)]
            task_outcomes[condition][task_id] = outcomes
            for episode_id, outcome in enumerate(outcomes):
                episodes[condition][(suite, task_id, episode_id)] = outcome
    return episodes, task_outcomes


def _paired_bootstrap_ci(
    baseline: np.ndarray,
    diagnosis: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if baseline.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    differences = diagnosis.astype(np.float64) - baseline.astype(np.float64)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        bootstrap_indices = rng.integers(0, differences.size, size=differences.size)
        estimates[index] = float(np.mean(differences[bootstrap_indices]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _load_offline(offline_root: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in offline_root.glob("*/summary.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one summary row in {path}.")
        records[str(rows[0]["condition"])] = rows[0]
    return records


def main() -> None:
    args = _parse_args()
    condition_metadata = _condition_metadata(args.online_root.resolve())
    if "baseline" not in condition_metadata:
        raise ValueError("Online results must include the baseline condition.")
    episodes, task_outcomes = _load_online_episodes(condition_metadata)
    offline = _load_offline(args.offline_root.resolve())
    baseline_episodes = episodes["baseline"]
    if not baseline_episodes:
        raise ValueError("Baseline has no completed episodes.")

    summary_records: list[dict[str, Any]] = []
    transition_records: list[dict[str, Any]] = []
    task_delta_records: list[dict[str, Any]] = []
    condition_order = sorted(
        condition_metadata,
        key=lambda name: (
            len(condition_metadata[name].get("disabled_video_layers", [])),
            condition_metadata[name].get("disabled_video_layers", [10**9])[0]
            if condition_metadata[name].get("disabled_video_layers")
            else -1,
        ),
    )
    baseline_rate = float(np.mean(list(baseline_episodes.values())))

    for condition_index, condition in enumerate(condition_order):
        condition_episodes = episodes.get(condition, {})
        paired_keys = sorted(set(baseline_episodes) & set(condition_episodes))
        baseline_paired = np.asarray([baseline_episodes[key] for key in paired_keys], dtype=int)
        condition_paired = np.asarray([condition_episodes[key] for key in paired_keys], dtype=int)
        success_rate = float(np.mean(list(condition_episodes.values()))) if condition_episodes else float("nan")
        paired_delta = (
            float(np.mean(condition_paired - baseline_paired)) if paired_keys else float("nan")
        )
        ci_low, ci_high = _paired_bootstrap_ci(
            baseline_paired,
            condition_paired,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + condition_index,
        )
        transitions = Counter(
            (int(baseline_value), int(condition_value))
            for baseline_value, condition_value in zip(baseline_paired, condition_paired)
        )
        transition_records.append(
            {
                "condition": condition,
                "baseline_success_to_ablation_success": transitions[(1, 1)],
                "baseline_success_to_ablation_failure": transitions[(1, 0)],
                "baseline_failure_to_ablation_success": transitions[(0, 1)],
                "baseline_failure_to_ablation_failure": transitions[(0, 0)],
                "paired_episodes": len(paired_keys),
            }
        )

        offline_record = offline.get(condition, {})
        summary_records.append(
            {
                "condition": condition,
                "disabled_video_layers": json.dumps(
                    condition_metadata[condition].get("disabled_video_layers", [])
                ),
                "offline_normalized_l2": offline_record.get("offline_normalized_l2", ""),
                "offline_cosine_similarity": offline_record.get("offline_cosine_similarity", ""),
                "gripper_flip_rate": offline_record.get("gripper_flip_rate", ""),
                "online_success_rate": success_rate,
                "delta_success_rate": success_rate - baseline_rate,
                "paired_delta_success_rate": paired_delta,
                "paired_delta_ci_low": ci_low,
                "paired_delta_ci_high": ci_high,
                "total_episodes": len(condition_episodes),
                "paired_episodes": len(paired_keys),
            }
        )

        all_task_ids = sorted(set(task_outcomes["baseline"]) | set(task_outcomes[condition]))
        for task_id in all_task_ids:
            baseline_task = task_outcomes["baseline"].get(task_id, [])
            condition_task = task_outcomes[condition].get(task_id, [])
            baseline_task_rate = float(np.mean(baseline_task)) if baseline_task else float("nan")
            condition_task_rate = float(np.mean(condition_task)) if condition_task else float("nan")
            task_delta_records.append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    "baseline_success_rate": baseline_task_rate,
                    "condition_success_rate": condition_task_rate,
                    "delta_success_rate": condition_task_rate - baseline_task_rate,
                    "baseline_episodes": len(baseline_task),
                    "condition_episodes": len(condition_task),
                }
            )

    output_dir = args.output_dir.resolve()
    _write_csv(output_dir / "summary.csv", summary_records, list(summary_records[0]))
    _write_csv(
        output_dir / "paired_transitions.csv",
        transition_records,
        list(transition_records[0]),
    )
    _write_csv(
        output_dir / "task_success_delta.csv",
        task_delta_records,
        list(task_delta_records[0]),
    )
    atomic_write_json(
        output_dir / "aggregate_metadata.json",
        {
            "online_root": str(args.online_root.resolve()),
            "offline_root": str(args.offline_root.resolve()),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "conditions": condition_order,
        },
    )
    print(f"Aggregated {len(condition_order)} conditions into {output_dir}")


if __name__ == "__main__":
    main()
