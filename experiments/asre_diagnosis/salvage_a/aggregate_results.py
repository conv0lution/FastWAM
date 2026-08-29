"""Strict held-out-primary aggregation for ASRE Salvage A."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_A_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round3b.outcome_statistics import (  # noqa: E402
    comparison_statistics,
)
from experiments.asre_diagnosis.salvage_a.classification import (  # noqa: E402
    classify_salvage,
)
from experiments.asre_diagnosis.salvage_a.definitions import (  # noqa: E402
    ACTION_HORIZON,
    ANALYSIS_SCOPES,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CONDITIONS,
    DISPLAY,
    INFERENCE_STEPS,
    ONLINE_TRIALS_PER_TASK,
    PRIMARY_COMPARISONS,
    PRIMARY_SCOPE,
    REPLAN_STEPS,
    SEED,
    TASK_IDS,
    TASK_SUITE,
    WAVES,
    condition_family_rank,
)


EpisodeKey = tuple[str, int, int]

ONLINE_IDENTITY_KEYS = (
    "git_commit_hash",
    "checkpoint_sha256",
    "dataset_stats_sha256",
    "state_bank_manifest_sha256",
    "valid_state_bank_manifest_sha256",
    "prompt_context_cache_sha256",
    "donor_mapping_path",
    "donor_mapping_sha256",
    "donor_observation_manifest_path",
    "donor_observation_manifest_sha256",
    "donor_observation_root",
    "preflight_report_path",
    "preflight_report_sha256",
    "machinery_report_path",
    "machinery_report_sha256",
    "calibration_split_manifest_path",
    "calibration_split_manifest_sha256",
    "state_selection_manifest_path",
    "state_selection_manifest_sha256",
    "differentiable_path_report_path",
    "differentiable_path_report_sha256",
    "subspace_basis_manifest_path",
    "subspace_basis_manifest_sha256",
    "subspace_diagnostics_path",
    "subspace_diagnostics_sha256",
    "round4c_summary_path",
    "round4c_summary_sha256",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _frozen_donor_pairs(
    mapping: Mapping[str, Any],
) -> dict[tuple[int, int], tuple[int, int]]:
    records = mapping.get("records")
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("Salvage A donor mapping must contain exactly 100 records.")
    pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for row in records:
        recipient = (int(row["task_id"]), int(row["recipient_trial"]))
        donor = (int(row.get("donor_task_id", row["task_id"])), int(row["donor_trial"]))
        if recipient in pairs or recipient == donor:
            raise ValueError(f"Malformed frozen donor pair: {recipient} -> {donor}.")
        pairs[recipient] = donor
    expected = {(task, trial) for task in TASK_IDS for trial in range(10)}
    if set(pairs) != expected:
        raise ValueError("Frozen donor mapping does not cover all registered episodes.")
    return pairs


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


def split_online_keys(
    split: Mapping[str, Any], available_keys: Sequence[EpisodeKey]
) -> dict[str, list[EpisodeKey]]:
    """Map the frozen 5/5 task split onto all paired online episodes."""

    records = split.get("per_task")
    if not isinstance(records, list) or len(records) != 10:
        raise ValueError("Salvage A split must contain ten per-task partitions.")
    partitions: dict[str, list[EpisodeKey]] = {"calibration": [], "heldout": []}
    observed_tasks: set[int] = set()
    for record in records:
        task_id = int(record["task_id"])
        if task_id in observed_tasks:
            raise ValueError(f"Duplicate task in Salvage A split: {task_id}.")
        observed_tasks.add(task_id)
        calibration = list(map(int, record["calibration_episode_ids"]))
        heldout = list(map(int, record["heldout_episode_ids"]))
        if (
            len(calibration) != 5
            or len(heldout) != 5
            or len(set(calibration)) != 5
            or len(set(heldout)) != 5
            or set(calibration) & set(heldout)
            or set(calibration) | set(heldout) != set(range(10))
        ):
            raise ValueError(f"Malformed 5/5 episode split for task {task_id}.")
        partitions["calibration"].extend(
            (TASK_SUITE, task_id, trial) for trial in sorted(calibration)
        )
        partitions["heldout"].extend(
            (TASK_SUITE, task_id, trial) for trial in sorted(heldout)
        )
    if observed_tasks != set(TASK_IDS):
        raise ValueError("Salvage A split task IDs must be exactly 0..9.")
    calibration_set = set(partitions["calibration"])
    heldout_set = set(partitions["heldout"])
    available = set(available_keys)
    if (
        len(calibration_set) != 50
        or len(heldout_set) != 50
        or calibration_set & heldout_set
        or calibration_set | heldout_set != available
    ):
        raise ValueError("Frozen 50/50 split does not cover the paired online episodes.")
    return partitions


def _validate_donor_assignments(
    assignments: Any,
    *,
    condition: str,
    task_id: int,
    trials: int,
    partitions: Mapping[str, Sequence[EpisodeKey]],
    donor_pairs: Mapping[tuple[int, int], tuple[int, int]] | None = None,
) -> None:
    if not isinstance(assignments, list):
        raise ValueError(f"Malformed donor assignments for {condition}, task {task_id}.")
    if condition == "current_all":
        if assignments:
            raise ValueError("Current endpoint must not emit donor assignments.")
        return
    calibration = {key[2] for key in partitions["calibration"] if key[1] == task_id}
    heldout = {key[2] for key in partitions["heldout"] if key[1] == task_id}
    by_recipient: dict[int, Mapping[str, Any]] = {}
    for row in assignments:
        recipient = int(row["recipient_trial"])
        donor = int(row["donor_trial"])
        recipient_partition = "calibration" if recipient in calibration else "heldout"
        donor_partition = "calibration" if donor in calibration else "heldout"
        if recipient in by_recipient:
            raise ValueError(f"Duplicate donor assignment for {condition}, task {task_id}.")
        if (
            int(row["recipient_task_id"]) != task_id
            or int(row["donor_task_id"]) != task_id
            or recipient == donor
            or row.get("recipient_first_query_image_verified") is not True
            or row.get("recipient_partition") != recipient_partition
            or row.get("donor_partition") != donor_partition
            or (
                donor_pairs is not None
                and (int(row["donor_task_id"]), donor)
                != donor_pairs.get((task_id, recipient))
            )
            or not (
                (recipient in calibration and donor in calibration)
                or (recipient in heldout and donor in heldout)
            )
        ):
            raise ValueError(f"Donor assignment violates the frozen split: {row}.")
        by_recipient[recipient] = row
    if set(by_recipient) != set(range(trials)):
        raise ValueError(f"Donor assignments do not cover trials 0..{trials - 1}.")


def load_online(
    wave1: Path,
    wave2: Path,
    split: Mapping[str, Any],
    *,
    expected_hashes: Mapping[str, str] | None = None,
    donor_pairs: Mapping[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[
    dict[str, dict[EpisodeKey, int]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Load eight strictly paired conditions and validate shared provenance."""

    summaries: dict[int, dict[str, Any]] = {}
    for wave, root in ((1, wave1), (2, wave2)):
        summary = _read(root / "launcher_summary.json")
        expected_conditions = [CONDITIONS[index] for index in WAVES[wave]]
        if not (
            summary.get("protocol") == SALVAGE_A_PROTOCOL
            and summary.get("mode") == "full"
            and summary.get("wave") == wave
            and summary.get("all_succeeded") is True
            and summary.get("conditions") == expected_conditions
            and summary.get("condition_indices") == list(WAVES[wave])
        ):
            raise ValueError(f"Salvage A full wave {wave} is incomplete or drifted.")
        summaries[wave] = summary

    outcomes: dict[str, dict[EpisodeKey, int]] = {}
    task_rows: list[dict[str, Any]] = []
    shared_identity: dict[str, Any] | None = None
    partitions: dict[str, list[EpisodeKey]] | None = None
    for index, condition in enumerate(CONDITIONS):
        root = wave1 if index in WAVES[1] else wave2
        directory = root / condition
        metadata = _read(directory / "run_metadata.json")
        family, rank = condition_family_rank(condition)
        config = metadata.get("condition_config")
        if not (
            metadata.get("status") == "completed"
            and metadata.get("condition_protocol") == SALVAGE_A_PROTOCOL
            and metadata.get("diagnosis_condition") == condition
            and metadata.get("task_suite") == TASK_SUITE
            and metadata.get("task_ids") == list(TASK_IDS)
            and metadata.get("number_of_trials") == ONLINE_TRIALS_PER_TASK
            and metadata.get("seed") == SEED
            and metadata.get("action_horizon") == ACTION_HORIZON
            and metadata.get("number_of_inference_steps") == INFERENCE_STEPS
            and metadata.get("replan_steps") == REPLAN_STEPS
            and isinstance(config, dict)
            and config.get("condition_name") == condition
            and config.get("subspace_basis_kind") == family
            and config.get("subspace_rank") == rank
            and config.get("disabled_video_layers") == list(range(15))
            and config.get("replacement_video_layers")
            == ([] if condition == "current_all" else list(range(15, 30)))
        ):
            raise ValueError(f"Salvage A online metadata mismatch: {condition}.")
        identity = {key: metadata.get(key) for key in ONLINE_IDENTITY_KEYS}
        if any(value is None or str(value) == "" for value in identity.values()):
            missing = [key for key, value in identity.items() if value is None or str(value) == ""]
            raise ValueError(f"Salvage A provenance is incomplete for {condition}: {missing}.")
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError(f"Salvage A provenance differs across conditions: {condition}.")
        if expected_hashes:
            mismatch = {
                key: {"observed": identity.get(key), "expected": value}
                for key, value in expected_hashes.items()
                if identity.get(key) != value
            }
            if mismatch:
                raise ValueError(f"Salvage A online provenance drifted: {mismatch}.")

        files = sorted((directory / TASK_SUITE).glob("gpu*_task*_results.json"))
        if len(files) != len(TASK_IDS):
            raise ValueError(f"{condition} lacks ten task result files.")
        values: dict[EpisodeKey, int] = {}
        observed_tasks: set[int] = set()
        for path in files:
            result = _read(path)
            task_id = int(result["task_id"])
            successes = set(map(int, result["success_episodes"]))
            failures = set(map(int, result["failure_episodes"]))
            if (
                task_id not in TASK_IDS
                or task_id in observed_tasks
                or successes & failures
                or successes | failures != set(range(ONLINE_TRIALS_PER_TASK))
            ):
                raise ValueError(f"Malformed paired outcomes: {path}.")
            result_identity = {
                key: result.get(key)
                for key in (
                    "subspace_basis_kind",
                    "subspace_rank",
                    "preflight_report_sha256",
                    "machinery_report_sha256",
                    "calibration_split_manifest_sha256",
                    "state_selection_manifest_sha256",
                    "differentiable_path_report_sha256",
                    "subspace_basis_manifest_sha256",
                    "subspace_diagnostics_sha256",
                    "round4c_summary_sha256",
                )
            }
            if result_identity != {
                key: metadata.get(key) for key in result_identity
            }:
                raise ValueError(f"Task-result provenance drifted: {path}.")
            observed_tasks.add(task_id)
            if partitions is None:
                all_keys = [
                    (TASK_SUITE, task, trial)
                    for task in TASK_IDS
                    for trial in range(ONLINE_TRIALS_PER_TASK)
                ]
                partitions = split_online_keys(split, all_keys)
            _validate_donor_assignments(
                result.get("donor_assignments", []),
                condition=condition,
                task_id=task_id,
                trials=ONLINE_TRIALS_PER_TASK,
                partitions=partitions,
                donor_pairs=donor_pairs,
            )
            for trial in range(ONLINE_TRIALS_PER_TASK):
                values[(TASK_SUITE, task_id, trial)] = int(trial in successes)
            task_rows.append(
                {
                    "scope": "all",
                    "condition": condition,
                    "display_name": DISPLAY[condition],
                    "task_id": task_id,
                    "task_description": result["task_description"],
                    "successes": len(successes),
                    "trials": ONLINE_TRIALS_PER_TASK,
                    "success_rate": len(successes) / ONLINE_TRIALS_PER_TASK,
                }
            )
        outcomes[condition] = values

    keys = set(outcomes["current_all"])
    if len(keys) != 100 or any(set(values) != keys for values in outcomes.values()):
        raise ValueError("Salvage A outcomes are not paired over exactly 100 episodes.")
    assert shared_identity is not None
    online_commit = str(shared_identity["git_commit_hash"])
    if any(summary.get("git_commit_hash") != online_commit for summary in summaries.values()):
        raise ValueError("Launcher summaries disagree with the online execution commit.")
    return outcomes, task_rows, {
        "shared_identity": shared_identity,
        "online_execution_git_commit": online_commit,
        "wave_summaries": summaries,
    }


def analyze_online(
    outcomes: Mapping[str, Mapping[EpisodeKey, int]],
    split: Mapping[str, Any],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Compute held-out primary and calibration/all descriptive statistics."""

    all_keys = sorted(outcomes["current_all"])
    partitions = split_online_keys(split, all_keys)
    scope_keys = {
        "heldout": partitions["heldout"],
        "calibration": partitions["calibration"],
        "all": all_keys,
    }
    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    rates: dict[tuple[str, str], float] = {}
    for scope in ANALYSIS_SCOPES:
        keys = scope_keys[scope]
        arrays = {
            condition: np.asarray([outcomes[condition][key] for key in keys], dtype=np.int8)
            for condition in CONDITIONS
        }
        zero = np.zeros(len(keys), dtype=np.int8)
        for condition in CONDITIONS:
            rate_stats = comparison_statistics(
                keys,
                zero,
                arrays[condition],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=BOOTSTRAP_SEED,
                seed_label=f"{scope}-rate-{condition}",
            )
            current_stats = comparison_statistics(
                keys,
                arrays["current_all"],
                arrays[condition],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=BOOTSTRAP_SEED,
                seed_label=f"{scope}-current-{condition}",
            )
            wrong_stats = comparison_statistics(
                keys,
                arrays["wrong_all"],
                arrays[condition],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=BOOTSTRAP_SEED,
                seed_label=f"{scope}-wrong-{condition}",
            )
            value = float(arrays[condition].mean())
            rates[(scope, condition)] = value
            family, rank = condition_family_rank(condition)
            summary_rows.append(
                {
                    "scope": scope,
                    "scope_role": "primary" if scope == PRIMARY_SCOPE else "descriptive",
                    "condition": condition,
                    "display_name": DISPLAY[condition],
                    "basis_family": family,
                    "rank": rank,
                    "successes": int(arrays[condition].sum()),
                    "episodes": len(keys),
                    "success_rate": value,
                    "paired_ci_low": rate_stats["paired_ci_low"],
                    "paired_ci_high": rate_stats["paired_ci_high"],
                    "task_hierarchical_ci_low": rate_stats["task_hierarchical_ci_low"],
                    "task_hierarchical_ci_high": rate_stats["task_hierarchical_ci_high"],
                    "delta_vs_current": current_stats["delta_success_rate"],
                    "delta_vs_wrong": wrong_stats["delta_success_rate"],
                }
            )
            for task_id in TASK_IDS:
                task_values = [
                    outcomes[condition][key] for key in keys if key[1] == task_id
                ]
                task_rows.append(
                    {
                        "scope": scope,
                        "scope_role": "primary" if scope == PRIMARY_SCOPE else "descriptive",
                        "condition": condition,
                        "display_name": DISPLAY[condition],
                        "task_id": task_id,
                        "successes": int(sum(task_values)),
                        "trials": len(task_values),
                        "success_rate": float(np.mean(task_values)),
                    }
                )
        for label, reference, target, rank, reference_kind in PRIMARY_COMPARISONS:
            stats = comparison_statistics(
                keys,
                arrays[reference],
                arrays[target],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=BOOTSTRAP_SEED,
                seed_label=f"{scope}-{label}",
            )
            contrast_rows.append(
                {
                    "scope": scope,
                    "scope_role": "primary" if scope == PRIMARY_SCOPE else "descriptive",
                    "comparison": label,
                    "reference_condition": reference,
                    "target_condition": target,
                    "rank": rank,
                    "reference_kind": reference_kind,
                    "primary_salvage_claim": scope == PRIMARY_SCOPE,
                    **stats,
                }
            )
    heldout_rates = {
        condition: rates[(PRIMARY_SCOPE, condition)] for condition in CONDITIONS
    }
    classification = classify_salvage(
        success_rates=heldout_rates, analysis_scope=PRIMARY_SCOPE
    )
    classification["descriptive_scopes_excluded"] = ["calibration", "all"]
    return summary_rows, contrast_rows, task_rows, classification


def validate_diagnostics(
    diagnostics: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Validate the registered 30-matrix offline diagnostic tables."""

    names = (
        "basis_diagnostics_summary",
        "basis_diagnostics_by_matrix",
        "aa_svd_overlap",
        "aa_half_stability",
    )
    values = [diagnostics.get(name) for name in names]
    if any(not isinstance(value, list) for value in values):
        raise ValueError(f"Salvage A diagnostics must define lists {names}.")
    summary, matrices, overlap, stability = values
    assert isinstance(summary, list)
    assert isinstance(matrices, list)
    assert isinstance(overlap, list)
    assert isinstance(stability, list)
    expected_family_rank = {
        (family, rank) for family in ("svd", "actionaware", "random") for rank in (36, 97)
    }
    if (
        {(str(row["basis_family"]), int(row["rank"])) for row in summary}
        != expected_family_rank
        or len(summary) != 6
    ):
        raise ValueError("Diagnostic summary must cover exactly three families at two ranks.")
    for row in summary:
        for key in (
            "heldout_delta_z_energy_captured",
            "calibration_action_sensitivity_captured",
        ):
            value = float(row[key])
            if not np.isfinite(value) or value < -1e-8 or value > 1.0 + 1e-6:
                raise ValueError(f"Invalid diagnostic fraction {key}: {row}.")
    expected_matrices = {
        f"layer{layer:02d}_{kind}" for layer in range(15, 30) for kind in ("k", "v")
    }
    matrix_keys = {
        (str(row["basis_family"]), int(row["rank"]), str(row["matrix"]))
        for row in matrices
    }
    expected_matrix_keys = {
        (family, rank, matrix)
        for family, rank in expected_family_rank
        for matrix in expected_matrices
    }
    if len(matrices) != 180 or matrix_keys != expected_matrix_keys:
        raise ValueError("Per-matrix diagnostics must contain 30 matrices x 6 bases.")
    if len(overlap) != 60 or len(stability) != 60:
        raise ValueError(
            "Overlap and stability tables must each contain 60 rows "
            "(30 matrices x 2 ranks)."
        )
    if any("subspace_overlap" not in row for row in (*overlap, *stability)):
        raise ValueError("Overlap/stability rows must report subspace_overlap.")
    for label, rows in (("overlap", overlap), ("stability", stability)):
        keys = {(int(row["rank"]), str(row["matrix"])) for row in rows}
        expected = {(rank, matrix) for rank in (36, 97) for matrix in expected_matrices}
        if keys != expected:
            raise ValueError(f"{label} rows do not cover every registered matrix/rank.")
        if any(
            not np.isfinite(float(row["subspace_overlap"]))
            or float(row["subspace_overlap"]) < -1e-8
            or float(row["subspace_overlap"]) > 1.0 + 1e-6
            for row in rows
        ):
            raise ValueError(f"Invalid {label} subspace overlap.")
    return (
        [dict(row) for row in summary],
        [dict(row) for row in matrices],
        [dict(row) for row in overlap],
        [dict(row) for row in stability],
    )


def _geometry_summary(
    summary: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Condense preregistered AA overlap/stability diagnostics by rank."""

    aa = {
        int(row["rank"]): row
        for row in summary
        if str(row["basis_family"]) == "actionaware"
    }
    rows: list[dict[str, Any]] = []
    for rank in (36, 97):
        selected = [row for row in stability if int(row["rank"]) == rank]
        if rank not in aa or len(selected) != 30:
            raise ValueError("Incomplete ActionAware geometry summary inputs.")
        angles = [
            float(row["principal_angle_median_degrees"])
            for row in selected
            if row.get("principal_angle_median_degrees") is not None
        ]
        rows.append(
            {
                "rank": rank,
                "weighted_aa_svd_overlap": float(aa[rank]["aa_svd_overlap"]),
                "weighted_aa_half_stability_overlap": float(
                    aa[rank]["aa_half_stability_overlap"]
                ),
                "median_matrix_principal_angle_median_degrees": (
                    float(np.median(angles)) if len(angles) == 30 else None
                ),
            }
        )
    return rows


def _markdown(payload: Mapping[str, Any]) -> str:
    rows = payload["online_condition_summary"]
    lookup = {(row["scope"], row["condition"]): row for row in rows}
    lines = [
        "# Fast-WAM ASRE Salvage A — Final Report",
        "",
        "Salvage A is complete. No later experiment was launched.",
        "",
        "## Primary held-out 50 episodes",
        "",
        "The classification below uses only five frozen held-out trials per task.",
        "",
        "| Condition | Success | Paired 95% CI | Task-hierarchical 95% CI | Δ current |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = lookup[("heldout", condition)]
        lines.append(
            f"| {row['display_name']} | {row['successes']}/50 ({row['success_rate']:.1%}) | "
            f"{row['paired_ci_low']:.1%}–{row['paired_ci_high']:.1%} | "
            f"{row['task_hierarchical_ci_low']:.1%}–"
            f"{row['task_hierarchical_ci_high']:.1%} | {row['delta_vs_current']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "### Registered primary contrasts",
            "",
            "| Comparison | Δ success | Paired 95% CI | Task-hierarchical 95% CI | "
            "Transitions 00/01/10/11 | McNemar p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["online_contrasts"]:
        if row["scope"] != PRIMARY_SCOPE:
            continue
        lines.append(
            f"| {row['comparison']} | {row['delta_success_rate']:+.1%} | "
            f"{row['paired_ci_low']:+.1%}–{row['paired_ci_high']:+.1%} | "
            f"{row['task_hierarchical_ci_low']:+.1%}–"
            f"{row['task_hierarchical_ci_high']:+.1%} | "
            f"{row['reference_failure_to_target_failure']}/"
            f"{row['reference_failure_to_target_success']}/"
            f"{row['reference_success_to_target_failure']}/"
            f"{row['reference_success_to_target_success']} | "
            f"{row['mcnemar_exact_p_value']:.4g} |"
        )
    for scope, title, episodes in (
        ("calibration", "Calibration 50 (descriptive only)", 50),
        ("all", "All 100 (descriptive only)", 100),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Condition | Success | Paired 95% CI |",
                "|---|---:|---:|",
            ]
        )
        for condition in CONDITIONS:
            row = lookup[(scope, condition)]
            lines.append(
                f"| {row['display_name']} | {row['successes']}/{episodes} "
                f"({row['success_rate']:.1%}) | "
                f"{row['paired_ci_low']:.1%}–{row['paired_ci_high']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## Offline basis diagnostics",
            "",
            "| Family | Rank | Held-out ΔZ energy | Calibration action sensitivity |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        payload["basis_diagnostics_summary"],
        key=lambda item: (int(item["rank"]), str(item["basis_family"])),
    ):
        lines.append(
            f"| {row['basis_family']} | {int(row['rank'])} | "
            f"{float(row['heldout_delta_z_energy_captured']):.2%} | "
            f"{float(row['calibration_action_sensitivity_captured']):.2%} |"
        )
    lines.extend(
        [
            "",
            "### ActionAware geometry and stability",
            "",
            "| Rank | Weighted AA–SVD overlap | Weighted half-split AA overlap | "
            "Median matrix principal angle |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in payload["basis_geometry_summary"]:
        angle = row["median_matrix_principal_angle_median_degrees"]
        angle_text = "n/a" if angle is None else f"{float(angle):.2f}°"
        lines.append(
            f"| {int(row['rank'])} | {float(row['weighted_aa_svd_overlap']):.3f} | "
            f"{float(row['weighted_aa_half_stability_overlap']):.3f} | {angle_text} |"
        )
    differentiable = payload["differentiable_path"]
    current_equivalence = differentiable["current_endpoint_equivalence"]
    wrong_equivalence = differentiable["wrong_endpoint_equivalence"]
    lines.extend(
        [
            "",
            "## Frozen construction and machinery",
            "",
            "- The task-stratified split contains 50 calibration and 50 held-out episode "
            "clusters (five trials per task in each partition); all replans remain cluster-local.",
            "- The split-local donor mapping is a deterministic within-task derangement; "
            "calibration and held-out recipients never cross partitions.",
            "- Matched SVD uses uncentered calibration ΔZ rows from the exact same 100 "
            "earliest/latest calibration states as ActionAware.",
            "- ActionAware uses the normalized raw `[32,7]` action, full 10-step "
            "differentiable denoising, λ={0.5,1.0}, and two frozen Rademacher probes per λ/state.",
            "- K and V are fitted separately for every layer 15–29; ranks 36/97 are nested "
            "orthonormal prefixes. Random uses the frozen pre-outcome nested QR prefix.",
            f"- Differentiable-path gate passed: `{differentiable['passed']}`; current/wrong "
            f"maximum raw-action differences were "
            f"`{float(current_equivalence['max_raw_action_abs_diff']):.6g}` / "
            f"`{float(wrong_equivalence['max_raw_action_abs_diff']):.6g}`.",
            f"- VJP finite/nonzero/shape gate passed: `{differentiable['vjp_passed']}`; "
            f"model frozen/unchanged: `{differentiable['model_parameters_frozen']}` / "
            f"`{differentiable['model_parameters_unchanged']}`.",
            f"- Endpoint, projection, PSD, orthogonality, nesting, shape, and self-replacement "
            f"machinery passed: `{payload['machinery']['passed']}`.",
            "- Full per-matrix energy, sensitivity, overlap, stability, and principal-angle "
            "tables are emitted beside this report; task-level outcomes are in `task_success.csv`.",
        ]
    )
    decision = payload["classification"]
    interpretation = {
        "STRONG": (
            "Action-sensitive geometry materially outperforms matched-rank variance geometry; "
            "the compact ASRE hypothesis is revived."
        ),
        "MODERATE": (
            "Action-aware geometry is useful but compact sufficiency remains incomplete; "
            "the compact ASRE hypothesis is only partially supported."
        ),
        "WEAK": (
            "Direct action sensitivity does not reveal a substantially smaller control-sufficient "
            "subspace; the compact ASRE search is closed."
        ),
    }[decision["classification"]]
    provenance = payload["provenance"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision['classification']}**",
            "",
            interpretation,
            "",
            f"Compact ASRE status: **{decision['compact_asre_status']}**.",
            "",
            "The decision used held-out point estimates only. Calibration/all-100 outcomes, "
            "uncertainty intervals, random controls, and task-level patterns did not alter the gate.",
            "",
            "## Provenance and controls",
            "",
            f"- Online execution commit: `{provenance['online_execution_git_commit']}`.",
            f"- Analysis commit: `{provenance['analysis_git_commit']}`.",
            f"- Frozen split: `{provenance['split_manifest_path']}` "
            f"(`{provenance['split_manifest_sha256']}`).",
            f"- Split-local donor mapping: `{provenance['donor_mapping_path']}` "
            f"(`{provenance['donor_mapping_sha256']}`).",
            f"- Frozen donor observation manifest: "
            f"`{provenance['donor_observation_manifest_path']}` "
            f"(`{provenance['donor_observation_manifest_sha256']}`).",
            f"- Basis manifest: `{provenance['basis_manifest_path']}` "
            f"(`{provenance['basis_manifest_sha256']}`).",
            f"- Machinery passed: `{payload['machinery']['passed']}`.",
            "- Formal preflight required a clean worktree outside the output root; the source "
            "file inventory is fixed by the execution commit above.",
            "- SVD and ActionAware bases used the same frozen calibration clusters/state selection.",
            "- Held-out states were not used to construct bases, choose ranks, probes, or lambdas.",
            "",
            "## Interpretation limits",
            "",
            "ActionAware directions are not semantic labels, causal minimality, architecture-wide "
            "universality, or an efficiency claim.",
            "",
            "## Stop rule",
            "",
            "No additional rank, cross-suite, RoboTwin, alternative gradient objective, "
            "world-prediction experiment, semantic probe, or later-stage experiment was launched.",
        ]
    )
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> Path:
    preflight_path = args.preflight.resolve()
    machinery_path = args.machinery.resolve()
    split_path = args.split.resolve()
    selection_path = args.state_selection.resolve()
    differentiable_path = args.differentiable_path_report.resolve()
    basis_path = args.basis_manifest.resolve()
    diagnostics_path = args.diagnostics.resolve()
    donor_mapping_path = args.donor_mapping.resolve()
    donor_manifest_path = args.donor_manifest.resolve()
    donor_root = args.donor_root.resolve()
    round4c_path = args.round4c_summary.resolve()
    preflight = _read(preflight_path)
    machinery = _read(machinery_path)
    split = _read(split_path)
    selection = _read(selection_path)
    differentiable = _read(differentiable_path)
    basis = _read(basis_path)
    diagnostics = _read(diagnostics_path)
    donor_mapping = _read(donor_mapping_path)
    round4c = _read(round4c_path)
    if not (
        preflight.get("protocol") == SALVAGE_A_PROTOCOL
        and preflight.get("status") == "compatible"
        and machinery.get("protocol") == SALVAGE_A_PROTOCOL
        and machinery.get("passed") is True
        and split.get("protocol") == SALVAGE_A_PROTOCOL
        and selection.get("protocol") == SALVAGE_A_PROTOCOL
        and differentiable.get("protocol") == SALVAGE_A_PROTOCOL
        and differentiable.get("passed") is True
        and basis.get("protocol") == SALVAGE_A_PROTOCOL
        and diagnostics.get("protocol") == SALVAGE_A_PROTOCOL
        and donor_mapping.get("protocol") == SALVAGE_A_PROTOCOL
        and round4c.get("status") == "complete"
    ):
        raise ValueError("Salvage A prerequisite reports are incomplete or incompatible.")
    expected_machine_links = {
        "preflight_report_sha256": sha256_file(preflight_path),
        "split_manifest_sha256": sha256_file(split_path),
        "state_selection_manifest_sha256": sha256_file(selection_path),
        "differentiable_path_report_sha256": sha256_file(differentiable_path),
        "basis_manifest_sha256": sha256_file(basis_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
    }
    machine_mismatch = {
        key: {"observed": machinery.get(key), "expected": value}
        for key, value in expected_machine_links.items()
        if machinery.get(key) != value
    }
    if machine_mismatch:
        raise ValueError(f"Salvage A machinery provenance drifted: {machine_mismatch}.")
    if basis.get("split_sha256") != sha256_file(split_path):
        raise ValueError("Salvage A basis belongs to another calibration split.")
    if basis.get("subspace_diagnostics_sha256") != sha256_file(diagnostics_path):
        raise ValueError("Salvage A basis belongs to another diagnostics artifact.")
    if (
        diagnostics.get("split_sha256") != sha256_file(split_path)
        or diagnostics.get("state_selection_sha256") != sha256_file(selection_path)
        or diagnostics.get("heldout_used_for_fitting") is not False
    ):
        raise ValueError("Salvage A diagnostics violate the frozen split/holdout contract.")
    if (
        donor_mapping.get("split_manifest_sha256") != sha256_file(split_path)
        or donor_mapping.get("donor_observation_manifest_sha256")
        != sha256_file(donor_manifest_path)
        or donor_mapping.get("split_local") is not True
        or donor_mapping.get("same_task") is not True
        or donor_mapping.get("derangement_verified") is not True
        or not donor_root.is_dir()
    ):
        raise ValueError("Salvage A donor bundle violates the frozen split contract.")
    if (
        preflight.get("round4c", {}).get("summary_sha256") != sha256_file(round4c_path)
        or selection.get("split_manifest_sha256") != sha256_file(split_path)
        or differentiable.get("split_sha256") != sha256_file(split_path)
    ):
        raise ValueError("Salvage A frozen input provenance is inconsistent.")
    expected_online_hashes = {
        "preflight_report_sha256": sha256_file(preflight_path),
        "machinery_report_sha256": sha256_file(machinery_path),
        "calibration_split_manifest_sha256": sha256_file(split_path),
        "state_selection_manifest_sha256": sha256_file(selection_path),
        "differentiable_path_report_sha256": sha256_file(differentiable_path),
        "subspace_basis_manifest_sha256": sha256_file(basis_path),
        "subspace_diagnostics_sha256": sha256_file(diagnostics_path),
        "round4c_summary_sha256": sha256_file(round4c_path),
    }
    outcomes, _raw_tasks, online_provenance = load_online(
        args.online_wave1.resolve(),
        args.online_wave2.resolve(),
        split,
        expected_hashes=expected_online_hashes,
        donor_pairs=_frozen_donor_pairs(donor_mapping),
    )
    online_rows, contrast_rows, task_rows, classification = analyze_online(outcomes, split)
    summary_diag, matrix_diag, overlap, stability = validate_diagnostics(diagnostics)
    geometry_summary = _geometry_summary(summary_diag, stability)
    weighted_scope_rows = diagnostics.get("summary_rows")
    if not isinstance(weighted_scope_rows, list) or len(weighted_scope_rows) != 18:
        raise ValueError("Diagnostics must include 18 weighted all/K/V summary rows.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "online_condition_summary.csv", online_rows)
    _write_csv(output / "online_contrasts.csv", contrast_rows)
    _write_csv(output / "task_success.csv", task_rows)
    _write_csv(output / "basis_diagnostics_summary.csv", summary_diag)
    _write_csv(output / "basis_diagnostics_by_matrix.csv", matrix_diag)
    _write_csv(output / "aa_svd_overlap.csv", overlap)
    _write_csv(output / "aa_half_stability.csv", stability)
    _write_csv(output / "basis_geometry_summary.csv", geometry_summary)
    _write_csv(output / "basis_diagnostics_weighted_by_scope.csv", weighted_scope_rows)
    shared = online_provenance["shared_identity"]
    payload = {
        "artifact_type": "asre_salvage_a_aggregate",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "status": "complete",
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "online_condition_summary": online_rows,
        "online_contrasts": contrast_rows,
        "task_success": task_rows,
        "basis_diagnostics_summary": summary_diag,
        "basis_diagnostics_by_matrix": matrix_diag,
        "aa_svd_overlap": overlap,
        "aa_half_stability": stability,
        "basis_geometry_summary": geometry_summary,
        "basis_diagnostics_weighted_by_scope": weighted_scope_rows,
        "classification": classification,
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "paired_and_task_hierarchical": True,
            "decision_scope": PRIMARY_SCOPE,
        },
        "machinery": machinery,
        "differentiable_path": differentiable,
        "provenance": {
            "online_execution_git_commit": online_provenance[
                "online_execution_git_commit"
            ],
            "analysis_git_commit": git_commit(PROJECT_ROOT),
            "preflight_path": str(preflight_path),
            "preflight_sha256": sha256_file(preflight_path),
            "machinery_path": str(machinery_path),
            "machinery_sha256": sha256_file(machinery_path),
            "split_manifest_path": str(split_path),
            "split_manifest_sha256": sha256_file(split_path),
            "state_selection_manifest_path": str(selection_path),
            "state_selection_manifest_sha256": sha256_file(selection_path),
            "differentiable_path_report_path": str(differentiable_path),
            "differentiable_path_report_sha256": sha256_file(differentiable_path),
            "donor_mapping_path": str(donor_mapping_path),
            "donor_mapping_sha256": shared["donor_mapping_sha256"],
            "donor_observation_manifest_path": str(donor_manifest_path),
            "donor_observation_manifest_sha256": sha256_file(donor_manifest_path),
            "donor_observation_root": str(donor_root),
            "basis_manifest_path": str(basis_path),
            "basis_manifest_sha256": sha256_file(basis_path),
            "diagnostics_path": str(diagnostics_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "round4c_summary_path": str(round4c_path),
            "round4c_summary_sha256": sha256_file(round4c_path),
            "online_wave1_summary_path": str(
                args.online_wave1.resolve() / "launcher_summary.json"
            ),
            "online_wave1_summary_sha256": sha256_file(
                args.online_wave1.resolve() / "launcher_summary.json"
            ),
            "online_wave2_summary_path": str(
                args.online_wave2.resolve() / "launcher_summary.json"
            ),
            "online_wave2_summary_sha256": sha256_file(
                args.online_wave2.resolve() / "launcher_summary.json"
            ),
        },
        "classification_used_heldout_50_only": True,
        "calibration_and_all100_descriptive_only": True,
        "later_stage_launched": False,
        "stop_rule_applied": True,
    }
    json_path = output / "salvage_a_summary.json"
    atomic_write_json(json_path, payload)
    report = _markdown(payload)
    report_path = output / "salvage_a_summary.md"
    _write_text(report_path, report)
    _write_text(output / "result_summary_for_gpt.md", report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-wave1", type=Path, required=True)
    parser.add_argument("--online-wave2", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--machinery", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--state-selection", type=Path, required=True)
    parser.add_argument("--differentiable-path-report", type=Path, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--round4c-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    path = aggregate(parser.parse_args())
    print(f"Salvage A aggregation complete: {path}")


if __name__ == "__main__":
    main()
