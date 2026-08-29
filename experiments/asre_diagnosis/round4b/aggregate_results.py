"""Strict paired online/held-out aggregation for ASRE Stage-2 Round-4B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    ROUND4B_PROTOCOL,
    atomic_write_json,
    build_round4b_conditions,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round2.metrics import (  # noqa: E402
    CONTINUOUS_ACTION_DIMENSIONS,
    compute_round2_metrics,
    extract_action_global_std,
)
from experiments.asre_diagnosis.round3b.outcome_statistics import (  # noqa: E402
    comparison_statistics,
)
from experiments.asre_diagnosis.round4b.classification import (  # noqa: E402
    classify_subspace,
)
from experiments.asre_diagnosis.round4b.launch_wave import WAVES  # noqa: E402


CONDITIONS = tuple(item.name for item in build_round4b_conditions(30))
DISPLAY = {
    "current_all": "Current (D)",
    "wrong_all": "Wrong (0)",
    "svd_r256": "SVD-256",
    "random_r256": "Random-256",
    "svd_r768": "SVD-768",
    "random_r768": "Random-768",
    "svd_r1536": "SVD-1536",
    "random_r1536": "Random-1536",
}
METRICS = (
    "executed_prefix_norm_rms",
    "full_chunk_norm_rms_0_31",
    "executed_prefix_cosine_similarity",
    "translation_norm_rms",
    "rotation_norm_rms",
    "executed_prefix_gripper_flip_rate",
    "full_horizon_gripper_flip_rate",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value.rstrip() + "\n")
    os.replace(temporary, path)


def _stable_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()
    return (seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _cluster_ci(
    values: Sequence[float], clusters: Sequence[tuple[int, int]], *, seed: int
) -> tuple[float, float]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for cluster, value in zip(clusters, values):
        grouped[cluster].append(float(value))
    groups = [np.asarray(grouped[key], dtype=np.float64) for key in sorted(grouped)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(10_000)
    for index in range(estimates.size):
        selected = rng.integers(0, len(groups), size=len(groups))
        chosen = [groups[item] for item in selected]
        estimates[index] = sum(float(x.sum()) for x in chosen) / sum(len(x) for x in chosen)
    return tuple(map(float, np.quantile(estimates, (0.025, 0.975))))


def load_online(
    wave1: Path, wave2: Path
) -> tuple[dict[str, dict[tuple[str, int, int], int]], list[dict[str, Any]]]:
    for wave, root in ((1, wave1), (2, wave2)):
        summary = _read(root / "launcher_summary.json")
        if not (
            summary.get("protocol") == ROUND4B_PROTOCOL
            and summary.get("mode") == "full"
            and summary.get("wave") == wave
            and summary.get("all_succeeded") is True
        ):
            raise ValueError(f"Round-4B full wave {wave} is incomplete.")
    outcomes: dict[str, dict[tuple[str, int, int], int]] = {}
    task_rows = []
    for index, condition in enumerate(CONDITIONS):
        root = wave1 if index in WAVES[1] else wave2
        directory = root / condition
        metadata = _read(directory / "run_metadata.json")
        if not (
            metadata.get("status") == "completed"
            and metadata.get("condition_protocol") == ROUND4B_PROTOCOL
            and metadata.get("diagnosis_condition") == condition
            and metadata.get("task_ids") == list(range(10))
            and metadata.get("number_of_trials") == 10
        ):
            raise ValueError(f"Online metadata mismatch: {condition}")
        files = sorted((directory / "libero_spatial").glob("gpu*_task*_results.json"))
        if len(files) != 10:
            raise ValueError(f"{condition} lacks ten task result files.")
        values = {}
        for path in files:
            result = _read(path)
            task_id = int(result["task_id"])
            successes = set(map(int, result["success_episodes"]))
            failures = set(map(int, result["failure_episodes"]))
            if successes & failures or successes | failures != set(range(10)):
                raise ValueError(f"Malformed paired outcomes: {path}")
            for trial in range(10):
                values[("libero_spatial", task_id, trial)] = int(trial in successes)
            task_rows.append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    "task_description": result["task_description"],
                    "successes": len(successes),
                    "trials": 10,
                    "success_rate": len(successes) / 10,
                }
            )
        outcomes[condition] = values
    keys = set(outcomes["current_all"])
    if len(keys) != 100 or any(set(value) != keys for value in outcomes.values()):
        raise ValueError("Round-4B online outcomes are not exactly paired over 100 episodes.")
    return outcomes, task_rows


def analyze_online(
    outcomes: Mapping[str, Mapping[tuple[str, int, int], int]],
    task_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    keys = sorted(outcomes["current_all"])
    arrays = {
        condition: np.asarray([outcomes[condition][key] for key in keys], dtype=np.int8)
        for condition in CONDITIONS
    }
    zero = np.zeros(100, dtype=np.int8)
    summary_rows = []
    contrast_rows = []
    for condition in CONDITIONS:
        rate = comparison_statistics(
            keys,
            zero,
            arrays[condition],
            bootstrap_samples=10_000,
            bootstrap_seed=4206,
            seed_label=f"rate-{condition}",
        )
        current = comparison_statistics(
            keys,
            arrays["current_all"],
            arrays[condition],
            bootstrap_samples=10_000,
            bootstrap_seed=4206,
            seed_label=f"current-{condition}",
        )
        wrong = comparison_statistics(
            keys,
            arrays["wrong_all"],
            arrays[condition],
            bootstrap_samples=10_000,
            bootstrap_seed=4206,
            seed_label=f"wrong-{condition}",
        )
        summary_rows.append(
            {
                "condition": condition,
                "display_name": DISPLAY[condition],
                "successes": int(arrays[condition].sum()),
                "episodes": 100,
                "success_rate": float(arrays[condition].mean()),
                "paired_ci_low": rate["paired_ci_low"],
                "paired_ci_high": rate["paired_ci_high"],
                "task_hierarchical_ci_low": rate["task_hierarchical_ci_low"],
                "task_hierarchical_ci_high": rate["task_hierarchical_ci_high"],
                "delta_vs_current": current["delta_success_rate"],
                "delta_vs_wrong": wrong["delta_success_rate"],
            }
        )
        for reference, stats in (("current_all", current), ("wrong_all", wrong)):
            contrast_rows.append(
                {
                    "comparison": f"{condition}_minus_{reference}",
                    "reference_condition": reference,
                    "target_condition": condition,
                    **stats,
                }
            )
    for rank in (256, 768, 1536):
        reference = f"random_r{rank}"
        target = f"svd_r{rank}"
        stats = comparison_statistics(
            keys,
            arrays[reference],
            arrays[target],
            bootstrap_samples=10_000,
            bootstrap_seed=4206,
            seed_label=f"svd-random-{rank}",
        )
        contrast_rows.append(
            {
                "comparison": f"svd_minus_random_r{rank}",
                "reference_condition": reference,
                "target_condition": target,
                "primary_matched_rank_contrast": True,
                **stats,
            }
        )
    rates = {row["condition"]: row["success_rate"] for row in summary_rows}
    task_success = {
        condition: {
            int(row["task_id"]): float(row["success_rate"])
            for row in task_rows
            if row["condition"] == condition
        }
        for condition in CONDITIONS
    }
    classification = classify_subspace(success_rates=rates, task_success=task_success)
    current_task = task_success["current_all"]
    for row in task_rows:
        row["delta_vs_current"] = row["success_rate"] - current_task[row["task_id"]]
    return summary_rows, contrast_rows, classification


def _load_actions(path: Path, ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        observed = list(map(str, data["sample_ids"].tolist()))
        raw = np.asarray(data["raw_actions"], dtype=np.float32)
        executed = np.asarray(data["executed_actions"], dtype=np.float32)
    if observed != list(ids) or raw.shape != (len(ids), 32, 7) or executed.shape != raw.shape:
        raise ValueError(f"Malformed offline actions: {path}")
    return raw, executed


def analyze_offline(
    *, offline_root: Path, split_path: Path, dataset_stats_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    split = _read(split_path)
    ids = list(split["holdout_sample_ids"])
    action_std = extract_action_global_std(_read(dataset_stats_path))
    records = {}
    raw = {}
    executed = {}
    for condition in CONDITIONS:
        directory = offline_root / condition
        metadata = _read(directory / "run_metadata.json")
        if not (
            metadata.get("status") == "complete"
            and metadata.get("protocol") == ROUND4B_PROTOCOL
            and metadata.get("completed_samples") == len(ids)
            and metadata.get("split_manifest_sha256") == sha256_file(split_path)
        ):
            raise ValueError(f"Offline metadata mismatch: {condition}")
        records[condition] = _read_jsonl(directory / "per_sample.jsonl")
        if [row["sample_id"] for row in records[condition]] != ids:
            raise ValueError(f"Offline ordering mismatch: {condition}")
        raw[condition], executed[condition] = _load_actions(directory / "actions.npz", ids)
    per_state = []
    summary = []
    replan = []
    for condition in CONDITIONS:
        for reference in ("current_all", "wrong_all"):
            local = []
            for index, source in enumerate(records[condition]):
                metrics = compute_round2_metrics(
                    baseline_raw=raw[reference][index],
                    diagnosis_raw=raw[condition][index],
                    baseline_executed=executed[reference][index],
                    diagnosis_executed=executed[condition][index],
                    action_std=action_std,
                    executed_prefix_length=10,
                )
                row = {
                    "condition": condition,
                    "reference": reference,
                    "sample_id": source["sample_id"],
                    "task_id": source["task_id"],
                    "episode_id": source["episode_id"],
                    "replan_id": source["replan_id"],
                    **{metric: float(metrics[metric]) for metric in METRICS},
                }
                for name, value in zip(
                    CONTINUOUS_ACTION_DIMENSIONS,
                    metrics["executed_prefix_norm_rms_by_dimension"],
                ):
                    row[f"dimension_rms_{name}"] = float(value)
                local.append(row)
                per_state.append(row)
            clusters = [(row["task_id"], row["episode_id"]) for row in local]
            metric_names = list(METRICS) + [
                f"dimension_rms_{name}" for name in CONTINUOUS_ACTION_DIMENSIONS
            ]
            for metric in metric_names:
                values = [row[metric] for row in local]
                low, high = _cluster_ci(
                    values,
                    clusters,
                    seed=_stable_seed(4207, condition, reference, metric),
                )
                summary.append(
                    {
                        "condition": condition,
                        "reference": reference,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "episode_cluster_ci_low": low,
                        "episode_cluster_ci_high": high,
                        "states": len(values),
                        "episode_clusters": len(set(clusters)),
                    }
                )
            for replan_id in sorted({row["replan_id"] for row in local}):
                selected = [row for row in local if row["replan_id"] == replan_id]
                replan.append(
                    {
                        "condition": condition,
                        "reference": reference,
                        "replan_id": replan_id,
                        "states": len(selected),
                        **{
                            f"mean_{metric}": float(np.mean([row[metric] for row in selected]))
                            for metric in METRICS
                        },
                    }
                )
    return per_state, summary, replan


def _markdown(payload: Mapping[str, Any]) -> str:
    rows = payload["online_condition_summary"]
    classification = payload["classification"]
    lines = [
        "# Fast-WAM ASRE Stage 2 — Round 4B Final Report",
        "",
        "Round 4B is complete. No later experiment was launched.",
        "",
        "## Online paired success",
        "",
        "| Condition | Success | Paired 95% CI | Δ current |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['success_rate']:.1%} | "
            f"{row['paired_ci_low']:.1%}–{row['paired_ci_high']:.1%} | "
            f"{row['delta_vs_current']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{classification['classification']}**",
            "",
            f"SVD-minus-random: `{classification['svd_minus_random']}`.",
            "",
            "## Calibration and controls",
            "",
            f"- Split: `{payload['provenance']['split_manifest_path']}` "
            f"(`{payload['provenance']['split_manifest_sha256']}`).",
            f"- Basis manifest: `{payload['provenance']['basis_manifest_path']}` "
            f"(`{payload['provenance']['basis_manifest_sha256']}`).",
            f"- Machinery passed: `{payload['machinery']['passed']}`.",
            "- Offline analysis used only the 20 frozen held-out episode clusters.",
            "- Offline distances are descriptive and are not behavioral surrogates.",
            "",
            "## Stop rule",
            "",
            "No cross-suite, alternate-checkpoint, additional-rank, token/head, or later-stage "
            "experiment was launched.",
        ]
    )
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> Path:
    outcomes, task_rows = load_online(args.online_wave1.resolve(), args.online_wave2.resolve())
    online_rows, contrasts, classification = analyze_online(outcomes, task_rows)
    per_state, offline_rows, replan_rows = analyze_offline(
        offline_root=args.offline_root.resolve(),
        split_path=args.split.resolve(),
        dataset_stats_path=args.dataset_stats.resolve(),
    )
    diagnostics = _read(args.diagnostics.resolve())
    machinery = _read(args.machinery.resolve())
    if machinery.get("passed") is not True or diagnostics.get("status") != "complete":
        raise ValueError("Round-4B machinery/diagnostics gate is incomplete.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "online_condition_summary.csv", online_rows)
    _write_csv(output / "online_contrasts.csv", contrasts)
    _write_csv(output / "task_success.csv", task_rows)
    _write_csv(output / "offline_per_state.csv", per_state)
    _write_csv(output / "offline_metric_summary.csv", offline_rows)
    _write_csv(output / "offline_replan_summary.csv", replan_rows)
    _write_csv(output / "subspace_diagnostics.csv", diagnostics["rows"])
    payload = {
        "artifact_type": "asre_round4b_aggregate",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "status": "complete",
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "online_condition_summary": online_rows,
        "online_contrasts": contrasts,
        "task_success": task_rows,
        "classification": classification,
        "offline_metric_summary": offline_rows,
        "offline_replan_summary": replan_rows,
        "subspace_diagnostics": diagnostics,
        "machinery": machinery,
        "bootstrap": {
            "samples": 10_000,
            "online_seed": 4206,
            "offline_seed": 4207,
            "paired_and_task_hierarchical": True,
        },
        "provenance": {
            "round4a_parent_tag": "ASRE-round4a-token-head-sparsity",
            "round4a_parent_commit": "094994e4e6f936c84fde3e1023ad998fabd17bcf",
            "round4a_summary_path": str(args.round4a_summary.resolve()),
            "round4a_summary_sha256": sha256_file(args.round4a_summary.resolve()),
            "preflight_path": str(args.preflight.resolve()),
            "preflight_sha256": sha256_file(args.preflight.resolve()),
            "split_manifest_path": str(args.split.resolve()),
            "split_manifest_sha256": sha256_file(args.split.resolve()),
            "basis_manifest_path": str(args.basis_manifest.resolve()),
            "basis_manifest_sha256": sha256_file(args.basis_manifest.resolve()),
            "diagnostics_path": str(args.diagnostics.resolve()),
            "diagnostics_sha256": sha256_file(args.diagnostics.resolve()),
            "machinery_path": str(args.machinery.resolve()),
            "machinery_sha256": sha256_file(args.machinery.resolve()),
        },
        "later_stage_launched": False,
        "stop_rule_applied": True,
    }
    json_path = output / "round4b_summary.json"
    atomic_write_json(json_path, payload)
    report = _markdown(payload)
    report_path = output / "round4b_summary.md"
    _write_text(report_path, report)
    _write_text(output / "result_summary_for_gpt.md", report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-wave1", type=Path, required=True)
    parser.add_argument("--online-wave2", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--machinery", type=Path, required=True)
    parser.add_argument("--round4a-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    path = aggregate(parser.parse_args())
    print(f"Round-4B aggregation complete: {path}")


if __name__ == "__main__":
    main()
