"""Aggregate paired G0 outcomes, uncertainty, classifications, and provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (
    G0_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.g0.definitions import (
    CONDITION_ORDER,
    PRIMARY_CONTRASTS,
    REFERENCE_SUITE,
    SUITE_ORDER,
    TASK_IDS,
    assert_output_scope,
    metadata_mismatches,
)


Key = tuple[str, int, int]
OutcomeMap = dict[str, dict[Key, int]]
SPATIAL_SOURCES = {
    "full_current": ("asre_results/round2/online_full/baseline_round2", "baseline_round2"),
    "late_current_15_29": (
        "asre_results/round2/online_full/keep_15_29",
        "keep_15_29",
    ),
    "early_current_00_19": (
        "asre_results/round2/online_full/keep_00_19",
        "keep_00_19",
    ),
    "late_wrong_scene_15_29": (
        "asre_results/round3b/online_full/late_wrong_scene",
        "late_wrong_scene",
    ),
}
SPATIAL_LATE_ROUND3B = (
    "asre_results/round3b/online_full/late_current_correct",
    "late_current_correct",
)


def _read_json(path: Path, label: str = "JSON") -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def validate_paired_alignment(outcomes: Mapping[str, Mapping[Key, int]]) -> list[Key]:
    missing = [condition for condition in CONDITION_ORDER if condition not in outcomes]
    if missing:
        raise ValueError(f"Missing G0 conditions for paired alignment: {missing}.")
    reference = set(outcomes[CONDITION_ORDER[0]])
    if not reference:
        raise ValueError("G0 paired outcomes are empty.")
    mismatched = {
        condition: {
            "missing": sorted(reference - set(outcomes[condition])),
            "extra": sorted(set(outcomes[condition]) - reference),
        }
        for condition in CONDITION_ORDER[1:]
        if set(outcomes[condition]) != reference
    }
    if mismatched:
        raise ValueError(f"Paired task/trial alignment failed: {mismatched}.")
    return sorted(reference)


def paired_bootstrap_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or samples <= 0:
        raise ValueError("Paired bootstrap requires non-empty 1D values and samples > 0.")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    batch = min(1000, samples)
    for start in range(0, samples, batch):
        stop = min(start + batch, samples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        estimates[start:stop] = array[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def task_hierarchical_bootstrap_ci(
    keys: Sequence[Key],
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if len(keys) != len(values) or not keys:
        raise ValueError("Task-hierarchical bootstrap inputs must be aligned and non-empty.")
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for key, value in zip(keys, values):
        grouped[(key[0], key[1])].append(float(value))
    task_keys = sorted(grouped)
    groups = [np.asarray(grouped[key], dtype=np.float64) for key in task_keys]
    if any(group.size == 0 for group in groups):
        raise ValueError("Task-hierarchical bootstrap encountered an empty task.")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected = rng.integers(0, len(groups), size=len(groups))
        task_means = []
        for task_index in selected:
            group = groups[int(task_index)]
            trial_indices = rng.integers(0, group.size, size=group.size)
            task_means.append(float(group[trial_indices].mean()))
        estimates[sample] = float(np.mean(task_means))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def cross_suite_hierarchical_bootstrap_ci(
    keys: Sequence[Key],
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if len(keys) != len(values) or not keys:
        raise ValueError("Cross-suite bootstrap inputs must be aligned and non-empty.")
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, value in zip(keys, values):
        grouped[key[0]][key[1]].append(float(value))
    suites = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected_suites = rng.integers(0, len(suites), size=len(suites))
        suite_means = []
        for suite_index in selected_suites:
            tasks = grouped[suites[int(suite_index)]]
            task_ids = sorted(tasks)
            selected_tasks = rng.integers(0, len(task_ids), size=len(task_ids))
            task_means = []
            for task_index in selected_tasks:
                trials = np.asarray(tasks[task_ids[int(task_index)]], dtype=np.float64)
                trial_indices = rng.integers(0, trials.size, size=trials.size)
                task_means.append(float(trials[trial_indices].mean()))
            suite_means.append(float(np.mean(task_means)))
        estimates[sample] = float(np.mean(suite_means))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _load_condition_results(
    root: Path,
    *,
    suite: str,
    recorded_condition: str,
    expected_protocol: str | None,
) -> tuple[dict[Key, int], dict[int, str]]:
    task_files = sorted((root / suite).glob("gpu*_task*_results.json"))
    if len(task_files) != 10:
        raise ValueError(f"Expected 10 task result files in {root}, got {len(task_files)}.")
    outcomes: dict[Key, int] = {}
    descriptions: dict[int, str] = {}
    seen_tasks: set[int] = set()
    for path in task_files:
        payload = _read_json(path, "task result")
        expected = {
            "task_suite": suite,
            "diagnosis_condition": recorded_condition,
            "total_episodes": 10,
        }
        if expected_protocol is not None:
            expected["condition_protocol"] = expected_protocol
        mismatch = metadata_mismatches(payload, expected)
        if mismatch:
            raise ValueError(f"Task result mismatch in {path}: {mismatch}.")
        task_id = int(payload.get("task_id", -1))
        if task_id not in TASK_IDS or task_id in seen_tasks:
            raise ValueError(f"Duplicate/invalid task ID {task_id} in {root}.")
        seen_tasks.add(task_id)
        successes = {int(value) for value in payload.get("success_episodes", [])}
        failures = {int(value) for value in payload.get("failure_episodes", [])}
        if successes & failures or successes | failures != set(range(10)):
            raise ValueError(f"Task/trial outcomes are not a complete partition in {path}.")
        description = str(payload.get("task_description", ""))
        if not description:
            raise ValueError(f"Missing task description in {path}.")
        descriptions[task_id] = description
        for trial in range(10):
            outcomes[(suite, task_id, trial)] = int(trial in successes)
    return outcomes, descriptions


def load_new_suite(
    g0_root: Path, suite: str
) -> tuple[OutcomeMap, dict[int, str], dict[str, Any]]:
    full_root = g0_root / suite / "full"
    summary = _read_json(full_root / "launcher_summary.json", "launcher summary")
    expected_summary = {
        "protocol": G0_PROTOCOL,
        "suite": suite,
        "mode": "full",
        "all_succeeded": True,
        "task_ids": list(TASK_IDS),
        "num_trials": 10,
    }
    mismatch = metadata_mismatches(summary, expected_summary)
    if mismatch:
        raise ValueError(f"Incomplete G0 launcher summary for {suite}: {mismatch}.")
    outcomes: OutcomeMap = {}
    descriptions: dict[int, str] | None = None
    metadata_by_condition: dict[str, Any] = {}
    for condition in CONDITION_ORDER:
        condition_root = full_root / condition
        condition_outcomes, condition_descriptions = _load_condition_results(
            condition_root,
            suite=suite,
            recorded_condition=condition,
            expected_protocol=G0_PROTOCOL,
        )
        outcomes[condition] = condition_outcomes
        if descriptions is None:
            descriptions = condition_descriptions
        elif condition_descriptions != descriptions:
            raise ValueError(f"Task description drift across conditions in {suite}.")
        metadata = _read_json(condition_root / "run_metadata.json", "run metadata")
        if metadata.get("status") != "completed":
            raise ValueError(f"Condition metadata is incomplete: {condition_root}.")
        metadata_by_condition[condition] = metadata
    validate_paired_alignment(outcomes)
    invariant = (
        "git_commit_hash",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "task_suite",
        "task_ids",
        "seed",
        "number_of_trials",
        "action_horizon",
        "number_of_inference_steps",
        "replan_steps",
        "compile_action_infer",
        "binarize_gripper",
        "donor_mapping_sha256",
        "donor_observation_manifest_sha256",
        "text_conditioning_source",
    )
    reference = metadata_by_condition[CONDITION_ORDER[0]]
    for condition in CONDITION_ORDER[1:]:
        drift = {
            field: {
                "reference": reference.get(field),
                "observed": metadata_by_condition[condition].get(field),
            }
            for field in invariant
            if metadata_by_condition[condition].get(field) != reference.get(field)
        }
        if drift:
            raise ValueError(f"Cross-condition provenance drift in {suite}: {drift}.")
    assert descriptions is not None
    return outcomes, descriptions, metadata_by_condition


def load_frozen_spatial() -> tuple[OutcomeMap, dict[int, str], dict[str, Any]]:
    outcomes: OutcomeMap = {}
    descriptions: dict[int, str] | None = None
    provenance: dict[str, Any] = {"label": "frozen prior result", "sources": {}}
    for alias, (relative, recorded) in SPATIAL_SOURCES.items():
        root = (project_root / relative).resolve()
        condition_outcomes, condition_descriptions = _load_condition_results(
            root,
            suite=REFERENCE_SUITE,
            recorded_condition=recorded,
            expected_protocol=None,
        )
        outcomes[alias] = condition_outcomes
        if descriptions is None:
            descriptions = condition_descriptions
        elif descriptions != condition_descriptions:
            raise ValueError("Frozen Spatial task descriptions drift across Stage-1 rounds.")
        metadata_path = root / "run_metadata.json"
        provenance["sources"][alias] = {
            "root": str(root),
            "run_metadata_path": str(metadata_path),
            "run_metadata_sha256": sha256_file(metadata_path),
            "recorded_condition": recorded,
        }
    late3b_root = (project_root / SPATIAL_LATE_ROUND3B[0]).resolve()
    late3b, _ = _load_condition_results(
        late3b_root,
        suite=REFERENCE_SUITE,
        recorded_condition=SPATIAL_LATE_ROUND3B[1],
        expected_protocol=None,
    )
    if late3b != outcomes["late_current_15_29"]:
        raise ValueError(
            "Frozen Round-2 keep_15_29 and Round-3B late_current outcomes differ; "
            "refusing to mix historical contrasts."
        )
    provenance["sources"]["late_current_round3b_crosscheck"] = {
        "root": str(late3b_root),
        "run_metadata_sha256": sha256_file(late3b_root / "run_metadata.json"),
        "outcomes_exactly_match_round2": True,
    }
    assert descriptions is not None
    return outcomes, descriptions, provenance


def classify_findings(
    *,
    full_rate: float,
    delta_late: float,
    delta_early_vs_late: float,
    delta_wrong_vs_late: float,
    wrong_task_ci_high: float,
    catastrophic_late_task_ids: Sequence[int],
) -> dict[str, Any]:
    informative = full_rate >= 0.70
    if not informative:
        return {
            "baseline_informative": False,
            "finding_A": "not_interpretable_low_baseline",
            "finding_B": "not_interpretable_low_baseline",
            "finding_C": "not_interpretable_low_baseline",
            "finding_C_task_hierarchical_support": False,
        }
    repeated_catastrophic = len(catastrophic_late_task_ids) >= 2
    if delta_late >= -0.10 and not repeated_catastrophic:
        finding_a = "strongly_consistent"
    elif delta_late >= -0.20:
        finding_a = "intermediate_or_heterogeneous"
    else:
        finding_a = "weak_or_inconsistent"
    if delta_early_vs_late <= -0.20:
        finding_b = "strongly_consistent"
    elif delta_early_vs_late <= -0.10:
        finding_b = "intermediate"
    else:
        finding_b = "weak_or_inconsistent"
    if delta_wrong_vs_late <= -0.20:
        finding_c = "strongly_consistent"
    elif delta_wrong_vs_late < -0.05:
        finding_c = "intermediate"
    else:
        finding_c = "weak_or_inconsistent"
    return {
        "baseline_informative": True,
        "finding_A": finding_a,
        "finding_A_repeated_catastrophic_task_collapse": repeated_catastrophic,
        "finding_B": finding_b,
        "finding_C": finding_c,
        "finding_C_task_hierarchical_support": wrong_task_ci_high < -0.05,
    }


def classify_overall(suite_classifications: Mapping[str, Mapping[str, Any]]) -> str:
    informative = [
        value for value in suite_classifications.values() if value["baseline_informative"]
    ]
    complete_patterns = [
        value
        for value in informative
        if all(
            value[key] == "strongly_consistent"
            for key in ("finding_A", "finding_B", "finding_C")
        )
    ]
    if len(informative) >= 2 and len(complete_patterns) >= 2:
        return "G0-STRONG"
    key_failures = [
        value
        for value in informative
        if value["finding_B"] == "weak_or_inconsistent"
        or value["finding_C"] == "weak_or_inconsistent"
    ]
    if not informative or len(key_failures) > len(informative) / 2:
        return "G0-WEAK / REASSESS"
    return "G0-TASK-CONDITIONED"


def scientific_interpretation(
    classification: str, suite_classifications: Mapping[str, Mapping[str, Any]]
) -> str:
    if classification == "G0-STRONG":
        return (
            "Stage-1 representation-usage structure is not limited to LIBERO-Spatial "
            "and is reasonably stable across multiple task distributions under the "
            "same checkpoint. Proceeding to Stage 2 Round 4A is scientifically supported, "
            "but this G0 pipeline does not launch it."
        )
    if classification == "G0-TASK-CONDITIONED":
        return (
            "Action-sufficient representation usage is task-distribution dependent. "
            "Stage 2 should be reformulated around task-conditioned action sufficiency; "
            "Round 4A must not be launched automatically."
        )
    informative = sum(
        bool(value["baseline_informative"])
        for value in suite_classifications.values()
    )
    if informative == 0:
        return (
            "The checkpoint's full-current baselines are uninformative on all new suites. "
            "This is not evidence against ASRE, but G0 cannot support generalization and the "
            "project must reassess the checkpoint/protocol before token or head sparsity."
        )
    return (
        "Most baseline-informative suites fail to reproduce key Stage-1 structure. "
        "The current claims are substantially LIBERO-Spatial specific; do not proceed "
        "directly to token, head, or subspace sparsity."
    )


def analyze_suite(
    suite: str,
    outcomes: OutcomeMap,
    descriptions: Mapping[int, str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    provenance_label: str,
) -> dict[str, Any]:
    keys = validate_paired_alignment(outcomes)
    condition_rows = []
    per_task_rows = []
    for condition in CONDITION_ORDER:
        values = np.asarray([outcomes[condition][key] for key in keys], dtype=np.float64)
        pair_low, pair_high = paired_bootstrap_ci(
            values,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, suite, condition, "cell_paired"),
        )
        task_low, task_high = task_hierarchical_bootstrap_ci(
            keys,
            values,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, suite, condition, "cell_task"),
        )
        condition_rows.append(
            {
                "suite": suite,
                "provenance": provenance_label,
                "condition": condition,
                "successes": int(values.sum()),
                "episodes": int(values.size),
                "success_rate": float(values.mean()),
                "paired_ci_low": pair_low,
                "paired_ci_high": pair_high,
                "task_hierarchical_ci_low": task_low,
                "task_hierarchical_ci_high": task_high,
            }
        )
    by_condition = {row["condition"]: row for row in condition_rows}
    contrast_rows = []
    transition_rows = []
    for name, (reference, target) in PRIMARY_CONTRASTS.items():
        reference_values = np.asarray(
            [outcomes[reference][key] for key in keys], dtype=np.float64
        )
        target_values = np.asarray([outcomes[target][key] for key in keys], dtype=np.float64)
        delta = target_values - reference_values
        pair_low, pair_high = paired_bootstrap_ci(
            delta,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, suite, name, "contrast_paired"),
        )
        task_low, task_high = task_hierarchical_bootstrap_ci(
            keys,
            delta,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, suite, name, "contrast_task"),
        )
        counts = {
            f"reference_{r}_target_{t}": int(
                np.sum((reference_values == r) & (target_values == t))
            )
            for r in (0, 1)
            for t in (0, 1)
        }
        contrast_rows.append(
            {
                "suite": suite,
                "contrast": name,
                "reference_condition": reference,
                "target_condition": target,
                "reference_success_rate": float(reference_values.mean()),
                "target_success_rate": float(target_values.mean()),
                "delta": float(delta.mean()),
                "paired_ci_low": pair_low,
                "paired_ci_high": pair_high,
                "task_hierarchical_ci_low": task_low,
                "task_hierarchical_ci_high": task_high,
                **counts,
            }
        )
        for reference_value in (1, 0):
            for target_value in (1, 0):
                transition_rows.append(
                    {
                        "suite": suite,
                        "contrast": name,
                        "reference_outcome": reference_value,
                        "target_outcome": target_value,
                        "count": int(
                            np.sum(
                                (reference_values == reference_value)
                                & (target_values == target_value)
                            )
                        ),
                    }
                )
    contrasts = {row["contrast"]: row for row in contrast_rows}
    catastrophic_late_tasks = []
    for task_id in TASK_IDS:
        row: dict[str, Any] = {
            "suite": suite,
            "task_id": task_id,
            "task_description": descriptions[task_id],
        }
        for condition in CONDITION_ORDER:
            values = [outcomes[condition][(suite, task_id, trial)] for trial in range(10)]
            row[condition] = float(np.mean(values))
        row["delta_late"] = row["late_current_15_29"] - row["full_current"]
        row["delta_early_vs_late"] = (
            row["early_current_00_19"] - row["late_current_15_29"]
        )
        row["delta_wrong_vs_late"] = (
            row["late_wrong_scene_15_29"] - row["late_current_15_29"]
        )
        if row["full_current"] >= 0.70 and row["late_current_15_29"] <= 0.20:
            catastrophic_late_tasks.append(task_id)
        per_task_rows.append(row)
    classification = classify_findings(
        full_rate=by_condition["full_current"]["success_rate"],
        delta_late=contrasts["delta_late"]["delta"],
        delta_early_vs_late=contrasts["delta_early_vs_late"]["delta"],
        delta_wrong_vs_late=contrasts["delta_wrong_vs_late"]["delta"],
        wrong_task_ci_high=contrasts["delta_wrong_vs_late"][
            "task_hierarchical_ci_high"
        ],
        catastrophic_late_task_ids=catastrophic_late_tasks,
    )
    classification["catastrophic_late_task_ids"] = catastrophic_late_tasks
    return {
        "suite": suite,
        "provenance": provenance_label,
        "condition_rows": condition_rows,
        "contrast_rows": contrast_rows,
        "transition_rows": transition_rows,
        "per_task_rows": per_task_rows,
        "classification": classification,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    analyses = summary["suite_analyses"]
    lines = [
        "# Fast-WAM ASRE G0 — Cross-Suite Generalization Gate",
        "",
        f"Final classification: **{summary['overall_classification']}**",
        "",
        summary["scientific_interpretation"],
        "",
        "| Suite | Provenance | Full | Late 15-29 | Early 0-19 | Wrong Late | Delta Late | Early-vs-Late | Wrong-vs-Late |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for suite in (REFERENCE_SUITE, *SUITE_ORDER):
        analysis = analyses[suite]
        cells = {row["condition"]: row for row in analysis["condition_rows"]}
        contrasts = {row["contrast"]: row for row in analysis["contrast_rows"]}
        lines.append(
            "| {suite} | {provenance} | {full:.1%} | {late:.1%} | {early:.1%} | "
            "{wrong:.1%} | {dl:+.1%} | {de:+.1%} | {dw:+.1%} |".format(
                suite=suite,
                provenance=analysis["provenance"],
                full=cells["full_current"]["success_rate"],
                late=cells["late_current_15_29"]["success_rate"],
                early=cells["early_current_00_19"]["success_rate"],
                wrong=cells["late_wrong_scene_15_29"]["success_rate"],
                dl=contrasts["delta_late"]["delta"],
                de=contrasts["delta_early_vs_late"]["delta"],
                dw=contrasts["delta_wrong_vs_late"]["delta"],
            )
        )
    lines.extend(["", "## New-suite classifications", ""])
    for suite in SUITE_ORDER:
        classification = analyses[suite]["classification"]
        lines.append(
            f"- `{suite}`: baseline_informative={classification['baseline_informative']}; "
            f"A={classification['finding_A']}; B={classification['finding_B']}; "
            f"C={classification['finding_C']}."
        )
    lines.extend(
        [
            "",
            "## Primary paired contrasts",
            "",
            "| Suite | Contrast | Delta | Paired 95% CI | Task-hierarchical 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for suite in (REFERENCE_SUITE, *SUITE_ORDER):
        for row in analyses[suite]["contrast_rows"]:
            lines.append(
                "| {suite} | {contrast} | {delta:+.1%} | [{pl:+.1%}, {ph:+.1%}] | "
                "[{tl:+.1%}, {th:+.1%}] |".format(
                    suite=suite,
                    contrast=row["contrast"],
                    delta=row["delta"],
                    pl=row["paired_ci_low"],
                    ph=row["paired_ci_high"],
                    tl=row["task_hierarchical_ci_low"],
                    th=row["task_hierarchical_ci_high"],
                )
            )
    lines.extend(
        [
            "",
            "## New-suite task-level success",
            "",
            "| Suite | Task | Full | Late | Early | Wrong Late |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for suite in SUITE_ORDER:
        for row in analyses[suite]["per_task_rows"]:
            lines.append(
                "| {suite} | {task} | {full:.0%} | {late:.0%} | {early:.0%} | "
                "{wrong:.0%} |".format(
                    suite=suite,
                    task=row["task_id"],
                    full=row["full_current"],
                    late=row["late_current_15_29"],
                    early=row["early_current_00_19"],
                    wrong=row["late_wrong_scene_15_29"],
                )
            )
    lines.extend(
        [
            "",
            "Spatial is a frozen Stage-1 reference. Object, Goal, and LIBERO-10 are new G0 runs.",
            "Stage 2 Round 4A was not launched by this pipeline.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(
    *,
    g0_root: Path,
    output_dir: Path,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20260828,
) -> dict[str, Any]:
    g0_root = g0_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    assert_output_scope(g0_root, project_root)
    assert_output_scope(output_dir, project_root)
    all_outcomes: dict[str, OutcomeMap] = {}
    analyses: dict[str, Any] = {}
    spatial_outcomes, spatial_descriptions, spatial_provenance = load_frozen_spatial()
    all_outcomes[REFERENCE_SUITE] = spatial_outcomes
    analyses[REFERENCE_SUITE] = analyze_suite(
        REFERENCE_SUITE,
        spatial_outcomes,
        spatial_descriptions,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        provenance_label="frozen prior result",
    )
    provenance_by_suite: dict[str, Any] = {REFERENCE_SUITE: spatial_provenance}
    for suite in SUITE_ORDER:
        outcomes, descriptions, metadata = load_new_suite(g0_root, suite)
        all_outcomes[suite] = outcomes
        analyses[suite] = analyze_suite(
            suite,
            outcomes,
            descriptions,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            provenance_label="new G0 run",
        )
        provenance_by_suite[suite] = {
            condition: {
                key: value
                for key, value in condition_metadata.items()
                if key
                in {
                    "git_commit_hash",
                    "checkpoint_sha256",
                    "dataset_stats_sha256",
                    "donor_mapping_sha256",
                    "donor_observation_manifest_sha256",
                    "torch_version",
                    "cuda_version",
                    "start_timestamp",
                    "end_timestamp",
                }
            }
            for condition, condition_metadata in metadata.items()
        }
    suite_classifications = {
        suite: analyses[suite]["classification"] for suite in SUITE_ORDER
    }
    overall = classify_overall(suite_classifications)
    interpretation = scientific_interpretation(overall, suite_classifications)

    pooled_rows = []
    pooled_keys = sorted(
        key for suite in SUITE_ORDER for key in all_outcomes[suite][CONDITION_ORDER[0]]
    )
    for condition in CONDITION_ORDER:
        values = [
            all_outcomes[key[0]][condition][key]
            for key in pooled_keys
        ]
        low, high = cross_suite_hierarchical_bootstrap_ci(
            pooled_keys,
            values,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, condition, "cross_suite"),
        )
        pooled_rows.append(
            {
                "statistic": condition,
                "estimate": float(np.mean(values)),
                "suite_task_trial_ci_low": low,
                "suite_task_trial_ci_high": high,
                "descriptive_only": True,
            }
        )
    for contrast, (reference, target) in PRIMARY_CONTRASTS.items():
        values = [
            all_outcomes[key[0]][target][key]
            - all_outcomes[key[0]][reference][key]
            for key in pooled_keys
        ]
        low, high = cross_suite_hierarchical_bootstrap_ci(
            pooled_keys,
            values,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, contrast, "cross_suite"),
        )
        pooled_rows.append(
            {
                "statistic": contrast,
                "estimate": float(np.mean(values)),
                "suite_task_trial_ci_low": low,
                "suite_task_trial_ci_high": high,
                "descriptive_only": True,
            }
        )

    summary = {
        "artifact_type": "asre_g0_cross_suite_summary",
        "schema_version": 1,
        "status": "complete",
        "created_at": now_iso(),
        "analysis_git_commit_hash": git_commit(project_root),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "suite_analyses": analyses,
        "suite_provenance": provenance_by_suite,
        "cross_suite_descriptive_rows": pooled_rows,
        "overall_classification": overall,
        "scientific_interpretation": interpretation,
        "baseline_informative_suite_count": sum(
            bool(value["baseline_informative"])
            for value in suite_classifications.values()
        ),
        "stage2_round4a_launched": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    all_condition_rows = [
        row
        for suite in (REFERENCE_SUITE, *SUITE_ORDER)
        for row in analyses[suite]["condition_rows"]
    ]
    all_contrast_rows = [
        row
        for suite in (REFERENCE_SUITE, *SUITE_ORDER)
        for row in analyses[suite]["contrast_rows"]
    ]
    all_transition_rows = [
        row
        for suite in (REFERENCE_SUITE, *SUITE_ORDER)
        for row in analyses[suite]["transition_rows"]
    ]
    all_task_rows = [
        row
        for suite in (REFERENCE_SUITE, *SUITE_ORDER)
        for row in analyses[suite]["per_task_rows"]
    ]
    _write_csv(output_dir / "condition_summary.csv", all_condition_rows)
    _write_csv(output_dir / "primary_contrasts.csv", all_contrast_rows)
    _write_csv(output_dir / "paired_transitions.csv", all_transition_rows)
    _write_csv(output_dir / "task_success.csv", all_task_rows)
    _write_csv(output_dir / "cross_suite_descriptive.csv", pooled_rows)
    atomic_write_json(output_dir / "g0_summary.json", summary)
    markdown = _summary_markdown(summary)
    (output_dir / "g0_summary.md").write_text(markdown, encoding="utf-8")
    (output_dir / "result_summary_for_gpt.md").write_text(
        markdown, encoding="utf-8"
    )
    metadata = {
        "artifact_type": "asre_g0_aggregate_metadata",
        "schema_version": 1,
        "status": "complete",
        "created_at": now_iso(),
        "summary_sha256": sha256_file(output_dir / "g0_summary.json"),
        "condition_summary_sha256": sha256_file(output_dir / "condition_summary.csv"),
        "primary_contrasts_sha256": sha256_file(output_dir / "primary_contrasts.csv"),
        "task_success_sha256": sha256_file(output_dir / "task_success.csv"),
        "result_summary_for_gpt_sha256": sha256_file(
            output_dir / "result_summary_for_gpt.md"
        ),
    }
    atomic_write_json(output_dir / "aggregate_metadata.json", metadata)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = aggregate(
        g0_root=args.g0_root,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"overall_classification": summary["overall_classification"]}, indent=2))


if __name__ == "__main__":
    main()
