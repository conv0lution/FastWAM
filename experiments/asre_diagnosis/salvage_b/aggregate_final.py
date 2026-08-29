"""Build the final Salvage-B world-vs-action dissociation report on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.asre_diagnosis.common import (
    SALVAGE_B_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.salvage_b.classification import (
    classify_functional_dissociation,
)
from experiments.asre_diagnosis.salvage_b.definitions import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CONDITIONS,
    PRIMARY_CONDITION,
    PROJECTED_CONDITIONS,
    SPECIAL_FAILURE_CLASSIFICATIONS,
)
from experiments.asre_diagnosis.salvage_b.statistics import (
    analyze_world_losses,
    functional_dissociation,
    paired_bootstrap_ci,
    task_hierarchical_bootstrap_ci,
)


DISPLAY = {
    "current_all": "Current",
    "wrong_all": "Wrong",
    "svd_r97": "SVD-97",
    "svd_r170": "SVD-170",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_COMPLETION_FILENAME = "salvage_b_completion.json"


def _payload_commit(payload: Mapping[str, Any]) -> str | None:
    """Return a source commit from one of the registered artifact layouts."""

    for key in ("git_commit_hash", "git_commit"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    for parent, key in (
        ("git", "head"),
        ("git", "current_head"),
        ("identity", "git_commit_hash"),
        ("provenance", "git_commit_hash"),
    ):
        nested = payload.get(parent)
        if isinstance(nested, Mapping):
            value = nested.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _payload_branch(payload: Mapping[str, Any]) -> str | None:
    for parent, key in (("git", "branch"), ("provenance", "git_branch")):
        nested = payload.get(parent)
        if isinstance(nested, Mapping):
            value = nested.get(key)
            if isinstance(value, str) and value:
                return value
    value = payload.get("git_branch")
    return value if isinstance(value, str) and value else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text.rstrip() + "\n")
    os.replace(temporary, path)


def _completion_artifact(
    *,
    output_dir: Path,
    summary_path: Path,
    summary: Mapping[str, Any],
    git_commit_hash: str,
    figure_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the immutable completion sentinel after every final file exists."""

    status = str(summary.get("status"))
    if status not in {"complete", "stopped"}:
        raise ValueError(f"Final summary is not terminal: {status!r}.")
    classification = summary.get("classification")
    if not isinstance(classification, Mapping):
        raise ValueError("Final summary lacks a classification object.")
    classification_name = classification.get("classification")
    if not isinstance(classification_name, str) or not classification_name:
        raise ValueError("Final summary classification is malformed.")
    if _payload_commit(summary) != git_commit_hash:
        raise ValueError("Final summary/source commit disagrees before publication.")

    report_paths = {
        "salvage_b_summary_md": output_dir / "salvage_b_summary.md",
        "result_summary_for_gpt": output_dir / "result_summary_for_gpt.md",
    }
    missing = [str(path) for path in report_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final Markdown report bundle is incomplete: {missing}")

    figure_path = output_dir / "plots/figure_manifest.json"
    figure_record: dict[str, Any] | None = None
    figures: list[dict[str, Any]] = []
    if status == "complete":
        if figure_manifest is None or not figure_path.is_file():
            raise FileNotFoundError("Completed Salvage-B result lacks its figure manifest.")
        persisted = _read_json(figure_path)
        if persisted != dict(figure_manifest) or summary.get("figures") != persisted:
            raise ValueError("Summary and persisted figure manifest disagree.")
        raw_figures = persisted.get("figures")
        if not isinstance(raw_figures, list) or len(raw_figures) != 4:
            raise ValueError("Completed Salvage-B result must contain exactly four figures.")
        for raw in raw_figures:
            if not isinstance(raw, Mapping):
                raise ValueError("Malformed final figure record.")
            path = Path(str(raw.get("path", ""))).resolve()
            digest = raw.get("sha256")
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"Final figure is absent or drifted: {path}")
            figures.append({"name": raw.get("name"), "path": str(path), "sha256": digest})
        figure_record = {
            "path": str(figure_path),
            "sha256": sha256_file(figure_path),
        }
    elif figure_manifest is not None or summary.get("figures") not in ([], None):
        raise ValueError("A technical-stop report must not claim a figure bundle.")

    return {
        "artifact_type": "asre_salvage_b_final_completion",
        "schema_version": 1,
        "protocol": SALVAGE_B_PROTOCOL,
        "status": status,
        "published_at": now_iso(),
        "git_commit_hash": git_commit_hash,
        "classification": classification_name,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "reports": [
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in report_paths.items()
        ],
        "figure_manifest": figure_record,
        "figures": figures,
        "publication_complete": True,
        "later_stage_launched": False,
        "salvage_c_exists": False,
    }


def _publish_completion(
    *,
    output_dir: Path,
    summary_path: Path,
    summary: Mapping[str, Any],
    git_commit_hash: str,
    figure_manifest: Mapping[str, Any] | None,
) -> Path:
    """Publish the one terminal marker last; partial reports have no marker."""

    completion_path = output_dir / FINAL_COMPLETION_FILENAME
    completion = _completion_artifact(
        output_dir=output_dir,
        summary_path=summary_path,
        summary=summary,
        git_commit_hash=git_commit_hash,
        figure_manifest=figure_manifest,
    )
    atomic_write_json(completion_path, completion)
    return completion_path


def _quantile(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite bootstrap recovery replicates remain.")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _recovery_from_means(means: np.ndarray) -> np.ndarray:
    """Columns are condition/current/wrong; works for loss or success."""

    condition = means[:, 0]
    current = means[:, 1]
    wrong = means[:, 2]
    denominator = wrong - current
    scale = np.maximum(1.0, np.maximum(np.abs(wrong), np.abs(current)))
    valid = np.abs(denominator) > np.finfo(np.float64).eps * scale
    result = np.full(condition.shape, np.nan, dtype=np.float64)
    result[valid] = (wrong[valid] - condition[valid]) / denominator[valid]
    return result


def _paired_recovery_draws(
    condition: np.ndarray,
    current: np.ndarray,
    wrong: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    matrix = np.stack([condition, current, wrong], axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, condition.size, size=(samples, condition.size))
    means = matrix[:, indices].mean(axis=2).T
    return _recovery_from_means(means)


def _hierarchical_recovery_draws(
    keys: Sequence[tuple[object, object, object]],
    condition: np.ndarray,
    current: np.ndarray,
    wrong: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    matrix = np.stack([condition, current, wrong], axis=0)
    grouped: dict[object, dict[object, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, key in enumerate(keys):
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(f"Invalid hierarchy key: {key!r}.")
        grouped[key[0]][key[1]].append(index)
    groups = [
        [
            np.asarray(grouped[task][episode], dtype=np.int64)
            for episode in sorted(grouped[task], key=repr)
        ]
        for task in sorted(grouped, key=repr)
    ]
    if not groups:
        raise ValueError("No task groups for recovery bootstrap.")
    rng = np.random.default_rng(seed)
    means = np.empty((samples, 3), dtype=np.float64)
    for replicate in range(samples):
        task_means = np.empty((len(groups), 3), dtype=np.float64)
        for task_output, task_index in enumerate(
            rng.integers(0, len(groups), size=len(groups))
        ):
            episodes = groups[int(task_index)]
            episode_means = np.empty((len(episodes), 3), dtype=np.float64)
            for episode_output, episode_index in enumerate(
                rng.integers(0, len(episodes), size=len(episodes))
            ):
                members = episodes[int(episode_index)]
                selected = members[
                    rng.integers(0, members.size, size=members.size)
                ]
                episode_means[episode_output] = matrix[:, selected].mean(axis=1)
            task_means[task_output] = episode_means.mean(axis=0)
        means[replicate] = task_means.mean(axis=0)
    return _recovery_from_means(means)


def _phase_summary(
    path: Path,
    *,
    phase: str,
    conditions: tuple[str, ...],
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    summary = _read_json(path)
    if not (
        summary.get("artifact_type") == "asre_salvage_b_world_phase_aggregate"
        and summary.get("schema_version") == 2
        and summary.get("protocol") == SALVAGE_B_PROTOCOL
        and summary.get("phase") == phase
        and summary.get("status") == "passed"
        and summary.get("passed") is True
        and summary.get("classification") is None
        and summary.get("conditions") == list(conditions)
        and summary.get("sample_count") == 100
        and summary.get("draws_per_sample") == 4
        and summary.get("pairing_complete") is True
        and summary.get("worker_count") == 4
    ):
        raise ValueError(f"Incompatible Salvage-B {phase} aggregate.")
    for key, expected in expected_hashes.items():
        if summary.get(key) != expected:
            raise ValueError(f"{phase} aggregate provenance drifted at {key}.")
    sample_identity_rows = summary.get("sample_identity")
    sample_identity = summary.get("sample_identity_sha256")
    if not (
        isinstance(sample_identity_rows, list)
        and len(sample_identity_rows) == 100
        and isinstance(sample_identity, str)
        and len(sample_identity) == 64
        and all(character in "0123456789abcdef" for character in sample_identity.lower())
        and sha256_json(sample_identity_rows) == sample_identity
    ):
        raise ValueError(f"{phase} aggregate lacks a valid frozen sample identity hash.")
    for key in (
        "draw_rows",
        "sample_rows",
        "sample_identity_rows",
        "condition_summary",
    ):
        file_path = Path(str(summary[f"{key}_path"])).resolve()
        if not file_path.is_file() or sha256_file(file_path) != summary[f"{key}_sha256"]:
            raise ValueError(f"{phase} aggregate artifact drifted: {file_path}")
    if phase == "endpoint":
        gate = summary.get("endpoint_gate", {})
        if not (
            gate.get("status") == "passed"
            and gate.get("passed") is True
            and gate.get("classification") is None
            and gate.get("gate", {}).get("median_descriptive_only") is True
        ):
            raise ValueError("Projected outcomes cannot follow an uninformative endpoint.")
    return summary


def _load_action_outcomes(
    *, preflight: Mapping[str, Any]
) -> tuple[
    dict[str, np.ndarray],
    list[tuple[int, int, int]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    frozen = preflight.get("frozen_action", {})
    summary_path = Path(str(frozen.get("summary_path"))).resolve()
    if (
        not summary_path.is_file()
        or sha256_file(summary_path) != frozen.get("summary_sha256")
    ):
        raise ValueError("Frozen Round-4C action summary drifted.")
    round4c = _read_json(summary_path)
    rows = {
        str(row["condition"]): row for row in round4c["online_condition_summary"]
    }
    if not set(CONDITIONS).issubset(rows):
        raise ValueError("Round-4C action summary lacks a Salvage-B condition.")
    artifacts = frozen.get("results", {}).get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(CONDITIONS):
        raise ValueError("Frozen action artifact inventory is incomplete.")
    outcome_maps: dict[str, dict[tuple[int, int, int], int]] = {}
    provenance: list[dict[str, Any]] = []
    for artifact in artifacts:
        condition = str(artifact["condition"])
        if condition not in CONDITIONS or condition in outcome_maps:
            raise ValueError(f"Unexpected frozen action condition: {condition}.")
        values: dict[tuple[int, int, int], int] = {}
        metadata_path = Path(str(artifact.get("metadata_path", ""))).resolve()
        if (
            not metadata_path.is_file()
            or sha256_file(metadata_path) != artifact.get("metadata_sha256")
        ):
            raise ValueError(
                f"Frozen Round-4C action metadata drifted: {condition}: {metadata_path}"
            )
        task_files = artifact.get("task_results")
        if not isinstance(task_files, list) or len(task_files) != 10:
            raise ValueError(f"Frozen action task inventory is incomplete: {condition}.")
        for item in task_files:
            path = Path(str(item["path"])).resolve()
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError(f"Frozen Round-4C action result drifted: {path}")
            result = _read_json(path)
            task_id = int(result["task_id"])
            success = set(map(int, result["success_episodes"]))
            failure = set(map(int, result["failure_episodes"]))
            if success & failure or success | failure != set(range(10)):
                raise ValueError(f"Malformed frozen action outcomes: {path}")
            for trial in range(10):
                values[(task_id, trial, trial)] = int(trial in success)
        if len(values) != 100 or sum(values.values()) != int(rows[condition]["successes"]):
            raise ValueError(f"Frozen action files disagree with summary: {condition}.")
        outcome_maps[condition] = values
        provenance.append(dict(artifact))
    reference_keys = sorted(outcome_maps["current_all"])
    if any(set(values) != set(reference_keys) for values in outcome_maps.values()):
        raise ValueError("Round-4C action episodes are not paired across conditions.")
    arrays = {
        condition: np.asarray(
            [outcome_maps[condition][key] for key in reference_keys], dtype=np.float64
        )
        for condition in CONDITIONS
    }
    action_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        source = rows[condition]
        action_rows.append(
            {
                "condition": condition,
                "display_name": DISPLAY[condition],
                "successes": int(source["successes"]),
                "episodes": int(source["episodes"]),
                "success_rate": float(source["success_rate"]),
                "paired_ci_low": float(source["paired_ci_low"]),
                "paired_ci_high": float(source["paired_ci_high"]),
                "task_hierarchical_ci_low": float(
                    source["task_hierarchical_ci_low"]
                ),
                "task_hierarchical_ci_high": float(
                    source["task_hierarchical_ci_high"]
                ),
                "source": "frozen Round 4C; no action episode rerun",
            }
        )
    return arrays, reference_keys, action_rows, {
        "summary_path": str(summary_path),
        "summary_sha256": frozen["summary_sha256"],
        "execution_commit": preflight["git"]["round4c_execution_commit"],
        "artifacts": provenance,
        "rerun": False,
    }


def _world_arrays(
    endpoint: Mapping[str, Any], projected: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], list[tuple[int, int, str]]]:
    combined = list(endpoint["sample_results"]) + list(projected["sample_results"])
    by_cell = {
        (str(row["sample_id"]), str(row["condition"])): row for row in combined
    }
    current_rows = [
        row for row in endpoint["sample_results"] if row["condition"] == "current_all"
    ]
    if len(current_rows) != 100 or len(by_cell) != 400:
        raise ValueError("Combined world sample aggregate is not exactly 100 × 4.")
    sample_ids = [str(row["sample_id"]) for row in current_rows]
    if len(set(sample_ids)) != 100:
        raise ValueError("Combined world sample IDs are duplicated.")
    arrays = {}
    for condition in CONDITIONS:
        if any((sample_id, condition) not in by_cell for sample_id in sample_ids):
            raise ValueError(f"World condition is not paired: {condition}.")
        arrays[condition] = np.asarray(
            [
                float(by_cell[(sample_id, condition)]["mean_native_world_loss"])
                for sample_id in sample_ids
            ],
            dtype=np.float64,
        )
    keys = [
        (
            int(row["task_id"]),
            int(row["episode_id"]),
            str(row["sample_id"]),
        )
        for row in current_rows
    ]
    return arrays, keys


def _loss_rows(
    *,
    analysis: Mapping[str, Any],
    world_arrays: Mapping[str, np.ndarray],
    world_keys: Sequence[tuple[int, int, str]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for offset, condition in enumerate(CONDITIONS):
        source = analysis["conditions"][condition]
        values = world_arrays[condition]
        mean_low, mean_high = paired_bootstrap_ci(
            values, samples=bootstrap_samples, seed=bootstrap_seed + 100 + offset
        )
        hierarchical_low, hierarchical_high = task_hierarchical_bootstrap_ci(
            world_keys,
            values,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 200 + offset,
        )
        rows.append(
            {
                "condition": condition,
                "display_name": DISPLAY[condition],
                "samples": int(source["sample_count"]),
                "draws_per_sample": 4,
                "mean_native_world_loss": float(source["mean_loss"]),
                "median_native_world_loss": float(source["median_loss"]),
                "mean_loss_paired_ci_low": mean_low,
                "mean_loss_paired_ci_high": mean_high,
                "mean_loss_task_hierarchical_ci_low": hierarchical_low,
                "mean_loss_task_hierarchical_ci_high": hierarchical_high,
                "delta_vs_current": float(source["delta_vs_current"]["mean"]),
                "delta_vs_current_paired_ci_low": float(
                    source["delta_vs_current"]["paired_ci_low"]
                ),
                "delta_vs_current_paired_ci_high": float(
                    source["delta_vs_current"]["paired_ci_high"]
                ),
                "delta_vs_current_task_hierarchical_ci_low": float(
                    source["delta_vs_current"]["task_hierarchical_ci_low"]
                ),
                "delta_vs_current_task_hierarchical_ci_high": float(
                    source["delta_vs_current"]["task_hierarchical_ci_high"]
                ),
                "delta_vs_wrong": float(source["delta_vs_wrong"]["mean"]),
                "delta_vs_wrong_paired_ci_low": float(
                    source["delta_vs_wrong"]["paired_ci_low"]
                ),
                "delta_vs_wrong_paired_ci_high": float(
                    source["delta_vs_wrong"]["paired_ci_high"]
                ),
                "delta_vs_wrong_task_hierarchical_ci_low": float(
                    source["delta_vs_wrong"]["task_hierarchical_ci_low"]
                ),
                "delta_vs_wrong_task_hierarchical_ci_high": float(
                    source["delta_vs_wrong"]["task_hierarchical_ci_high"]
                ),
            }
        )
    return rows


def _recovery_rows(
    *,
    action_arrays: Mapping[str, np.ndarray],
    action_keys: Sequence[tuple[int, int, int]],
    world_arrays: Mapping[str, np.ndarray],
    world_keys: Sequence[tuple[int, int, str]],
    analysis: Mapping[str, Any],
    classification: Mapping[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for offset, condition in enumerate(CONDITIONS):
        paired_action = _paired_recovery_draws(
            action_arrays[condition],
            action_arrays["current_all"],
            action_arrays["wrong_all"],
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1000 + offset,
        )
        hierarchical_action = _hierarchical_recovery_draws(
            action_keys,
            action_arrays[condition],
            action_arrays["current_all"],
            action_arrays["wrong_all"],
            samples=bootstrap_samples,
            seed=bootstrap_seed + 2000 + offset,
        )
        paired_world = _paired_recovery_draws(
            world_arrays[condition],
            world_arrays["current_all"],
            world_arrays["wrong_all"],
            samples=bootstrap_samples,
            seed=bootstrap_seed + 3000 + offset,
        )
        hierarchical_world = _hierarchical_recovery_draws(
            world_keys,
            world_arrays[condition],
            world_arrays["current_all"],
            world_arrays["wrong_all"],
            samples=bootstrap_samples,
            seed=bootstrap_seed + 4000 + offset,
        )
        ar_low, ar_high = _quantile(paired_action)
        ar_h_low, ar_h_high = _quantile(hierarchical_action)
        wr_low, wr_high = _quantile(paired_world)
        wr_h_low, wr_h_high = _quantile(hierarchical_world)
        # Action and world samples are different frozen episode sets.  Their
        # cross-functional gap is therefore bootstrapped independently rather
        # than falsely paired at the sample level.
        fd_paired = paired_action - paired_world
        fd_hierarchical = hierarchical_action - hierarchical_world
        fd_low, fd_high = _quantile(fd_paired)
        fd_h_low, fd_h_high = _quantile(fd_hierarchical)
        point = classification["recoveries"][condition]
        registered_world = analysis["conditions"][condition]
        if not math.isclose(
            float(point["world_recovery"]),
            float(registered_world["world_recovery"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Classification/world-recovery point estimates disagree.")
        rows.append(
            {
                "condition": condition,
                "display_name": DISPLAY[condition],
                "action_recovery": float(point["action_recovery"]),
                "action_recovery_paired_ci_low": ar_low,
                "action_recovery_paired_ci_high": ar_high,
                "action_recovery_task_hierarchical_ci_low": ar_h_low,
                "action_recovery_task_hierarchical_ci_high": ar_h_high,
                "world_recovery": float(point["world_recovery"]),
                "world_recovery_paired_ci_low": wr_low,
                "world_recovery_paired_ci_high": wr_high,
                "world_recovery_task_hierarchical_ci_low": wr_h_low,
                "world_recovery_task_hierarchical_ci_high": wr_h_high,
                "functional_dissociation": functional_dissociation(
                    action_recovery_value=float(point["action_recovery"]),
                    world_recovery_value=float(point["world_recovery"]),
                ),
                "functional_dissociation_independent_ci_low": fd_low,
                "functional_dissociation_independent_ci_high": fd_high,
                "functional_dissociation_task_hierarchical_independent_ci_low": fd_h_low,
                "functional_dissociation_task_hierarchical_independent_ci_high": fd_h_high,
                "recoveries_clipped": False,
                "cross_function_sample_pairing": False,
            }
        )
    return rows


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _number(value: float) -> str:
    return f"{float(value):.6g}"


def _artifact_entry(
    name: str,
    path_value: str | Path,
    *,
    expected_sha256: str | None = None,
    kind: str = "file",
) -> dict[str, Any]:
    path = Path(path_value).resolve()
    exists = path.is_dir() if kind == "directory" else path.is_file()
    if not exists:
        raise FileNotFoundError(f"Registered artifact path is absent: {name}: {path}")
    digest = None
    if kind == "file":
        digest = sha256_file(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"Registered artifact hash drifted: {name}: {path}")
    return {
        "name": name,
        "path": str(path),
        "sha256": digest,
        "kind": kind,
        "publication_state": "verified_existing",
    }


def _publication_entry(name: str, path: Path) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path.resolve()),
        "sha256": None,
        "kind": "file",
        "publication_state": "written_by_final_publication_and_bound_by_completion",
    }


def _normal_artifact_inventory(
    *,
    output_dir: Path,
    preflight: Mapping[str, Any],
    world_manifest: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    projected: Mapping[str, Any],
    action_provenance: Mapping[str, Any],
    direct_paths: Mapping[str, Path],
    csv_paths: Mapping[str, Path],
    figures: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries = [
        _artifact_entry("Phase-A architecture audit", direct_paths["architecture"]),
        _artifact_entry(
            "Phase-A architecture audit Markdown",
            direct_paths["architecture"].with_suffix(".md"),
        ),
        _artifact_entry("Salvage-B preflight", direct_paths["preflight"]),
        _artifact_entry("shared-interface machinery", direct_paths["machinery"]),
        _artifact_entry(
            "shared-interface machinery Markdown",
            direct_paths["machinery"].with_suffix(".md"),
        ),
        _artifact_entry("world-evaluation manifest", direct_paths["world_manifest"]),
        _artifact_entry(
            "frozen world bundle completion sentinel",
            direct_paths["world_manifest"].parent / "world_bundle_complete.json",
        ),
        _artifact_entry("stochastic manifest", direct_paths["stochastic_manifest"]),
        _artifact_entry(
            "fixed draw tensors",
            str(world_manifest["draw_tensor_path"]),
            expected_sha256=str(world_manifest["draw_tensor_sha256"]),
        ),
        _artifact_entry("processed target manifest", direct_paths["target_manifest"]),
        _artifact_entry("endpoint aggregate", direct_paths["endpoint"]),
        _artifact_entry("projected aggregate", direct_paths["projected"]),
        _artifact_entry(
            "frozen Round-4C action summary",
            str(action_provenance["summary_path"]),
            expected_sha256=str(action_provenance["summary_sha256"]),
        ),
        _artifact_entry(
            "checkpoint",
            str(preflight["state"]["checkpoint_path"]),
            expected_sha256=str(preflight["state"]["checkpoint_sha256"]),
        ),
        _artifact_entry(
            "valid state-bank manifest",
            str(preflight["state"]["valid_manifest_path"]),
            expected_sha256=str(preflight["state"]["valid_manifest_sha256"]),
        ),
        _artifact_entry(
            "state-bank source manifest",
            str(preflight["state"]["source_manifest_path"]),
            expected_sha256=str(preflight["state"]["source_manifest_sha256"]),
        ),
        _artifact_entry(
            "dataset statistics",
            str(preflight["state"]["dataset_stats_path"]),
            expected_sha256=str(preflight["state"]["dataset_stats_sha256"]),
        ),
        _artifact_entry(
            "prompt context cache",
            str(preflight["state"]["prompt_context_cache_path"]),
            expected_sha256=str(preflight["state"]["prompt_context_cache_sha256"]),
        ),
        _artifact_entry(
            "frozen Round-4B SVD basis manifest",
            str(preflight["basis"]["path"]),
            expected_sha256=str(preflight["basis"]["sha256"]),
        ),
        _artifact_entry(
            "frozen Round-4B calibration split",
            str(preflight["basis"]["split_path"]),
            expected_sha256=str(preflight["basis"]["split_sha256"]),
        ),
        _artifact_entry(
            "frozen donor mapping",
            str(preflight["donors"]["mapping_path"]),
            expected_sha256=str(preflight["donors"]["mapping_sha256"]),
        ),
        _artifact_entry(
            "frozen donor observation manifest",
            str(preflight["donors"]["manifest_path"]),
            expected_sha256=str(preflight["donors"]["manifest_sha256"]),
        ),
        _artifact_entry(
            "frozen donor root", str(preflight["donors"]["root"]), kind="directory"
        ),
        _artifact_entry(
            "frozen Round-4C result root",
            str(preflight["frozen_action"]["round4c_root"]),
            kind="directory",
        ),
        _artifact_entry(
            "Salvage-A final summary",
            str(preflight["salvage_a"]["summary_path"]),
            expected_sha256=str(preflight["salvage_a"]["summary_sha256"]),
        ),
        _artifact_entry(
            "official world dataset root",
            str(preflight["world_data"]["dataset_root"]),
            kind="directory",
        ),
        _artifact_entry(
            "official world dataset info metadata",
            str(preflight["world_data"]["info_path"]),
            expected_sha256=str(preflight["world_data"]["info_sha256"]),
        ),
        _artifact_entry(
            "official world dataset tasks metadata",
            str(preflight["world_data"]["tasks_path"]),
            expected_sha256=str(preflight["world_data"]["tasks_sha256"]),
        ),
    ]
    for phase_name, phase_summary in (("endpoint", endpoint), ("projected", projected)):
        entries.append(
            _artifact_entry(
                f"{phase_name} launcher configuration",
                str(phase_summary["launcher_config_path"]),
                expected_sha256=str(phase_summary["launcher_config_sha256"]),
            )
        )
        for shard in phase_summary["shards"]:
            worker = int(shard["worker_index"])
            for key in ("metadata", "rows"):
                entries.append(
                    _artifact_entry(
                        f"{phase_name} worker {worker:02d} {key}",
                        str(shard[f"{key}_path"]),
                        expected_sha256=str(shard[f"{key}_sha256"]),
                    )
                )
        for key in (
            "draw_rows",
            "sample_rows",
            "sample_identity_rows",
            "condition_summary",
        ):
            entries.append(
                _artifact_entry(
                    f"{phase_name} {key.replace('_', ' ')}",
                    str(phase_summary[f"{key}_path"]),
                    expected_sha256=str(phase_summary[f"{key}_sha256"]),
                )
            )
    entries.append(
        _artifact_entry(
            "standalone world endpoint gate",
            direct_paths["endpoint"].parent / "world_endpoint_gate.json",
        )
    )
    for artifact in action_provenance["artifacts"]:
        condition = str(artifact["condition"])
        entries.append(
            _artifact_entry(
                f"Round-4C {condition} run metadata",
                str(artifact["metadata_path"]),
                expected_sha256=str(artifact["metadata_sha256"]),
            )
        )
        for index, item in enumerate(artifact["task_results"]):
            entries.append(
                _artifact_entry(
                    f"Round-4C {condition} task result {index:02d}",
                    str(item["path"]),
                    expected_sha256=str(item["sha256"]),
                )
            )
    for name, path in csv_paths.items():
        entries.append(_artifact_entry(name, path))
    figure_manifest = output_dir / "plots/figure_manifest.json"
    entries.append(_artifact_entry("figure manifest", figure_manifest))
    for figure in figures["figures"]:
        entries.append(
            _artifact_entry(
                str(figure["name"]),
                str(figure["path"]),
                expected_sha256=str(figure["sha256"]),
            )
        )
    entries.extend(
        [
            _publication_entry(
                "driver status (finalized after aggregate)",
                output_dir.parent / "driver_status.json",
            ),
            _publication_entry("final JSON summary", output_dir / "salvage_b_summary.json"),
            _publication_entry("final human Markdown", output_dir / "salvage_b_summary.md"),
            _publication_entry("final GPT Markdown", output_dir / "result_summary_for_gpt.md"),
            _publication_entry(
                "final completion sentinel", output_dir / FINAL_COMPLETION_FILENAME
            ),
        ]
    )
    return entries


def _normal_markdown(summary: Mapping[str, Any]) -> str:
    endpoint = summary["world_endpoint_gate"]
    classification = summary["classification"]["classification"]
    factorization = summary["phase_a"].get("factorization_equivalence", {})
    numerical_policy = factorization.get("numerical_equivalence_policy", {})
    token_equivalence = factorization.get("token_equivalence", {})
    prediction_equivalence = factorization.get("prediction_equivalence", {})
    if all(
        isinstance(value, (int, float))
        for value in (
            token_equivalence.get("relative_rmse"),
            prediction_equivalence.get("relative_rmse"),
            numerical_policy.get("relative_rmse_max"),
        )
    ):
        factorization_note = (
            "The mathematically equivalent masked-SDPA factorization was validated "
            "under an execution-dtype-derived numerical budget: token relative RMSE "
            f"`{100.0 * float(token_equivalence['relative_rmse']):.4f}%`, prediction "
            f"relative RMSE `{100.0 * float(prediction_equivalence['relative_rmse']):.4f}%`, "
            f"budget `{100.0 * float(numerical_policy['relative_rmse_max']):.4f}%` "
            f"for `{numerical_policy.get('dtype')}`. Pointwise allclose is descriptive, "
            "not the gate, because the stock and factorized calls use different fully "
            "masked sequence extents."
        )
    else:
        factorization_note = (
            "Runtime machinery validated the mathematically equivalent masked-SDPA "
            "factorization under its execution-dtype-derived numerical budget."
        )
    lines = [
        "# Fast-WAM ASRE Salvage B — Final Report",
        "",
        "Salvage B is complete. No later ASRE experiment was launched.",
        "",
        "## Phase A — shared causal interface",
        "",
        "**Preferred Path A passed.** The exact shared representation is the late "
        "first-frame video K/V prefix `Z={(K_l,V_l): l=15..29}`, with each tensor "
        "shaped `[1,98,3072]` (24 heads × 128). The same selected cache objects feed "
        "both action and future-video consumers. Layers 0–14 disable the prefix for "
        "both functions, and the future-only factorization removes an untouched raw-RGB "
        "prefix bypass.",
        "",
        "| Role | Path | Class.function | Line |",
        "|---|---|---|---:|",
    ]
    for role, location in summary["phase_a"]["source_locations"].items():
        class_name = location.get("class")
        function_name = str(location.get("function", ""))
        qualified = (
            f"{class_name}.{function_name}" if class_name else function_name
        )
        lines.append(
            f"| `{role}` | `{location.get('path')}` | `{qualified}` | "
            f"{location.get('line')} |"
        )
    lines.extend(
        [
            "",
            "Runtime machinery passed stock-equivalence, same-object reach, intervention "
            "reach to both consumers, no-bypass, rank-0/rank-D endpoints, and r97/r170 "
            "shape/key/head identity.",
            factorization_note,
            "",
            "## Projection/basis provenance",
            "",
            "The exact intervention is `Z_wrong + (Z_current-Z_wrong) B_r B_r^T`. "
            "`B_r` is the frozen nested prefix of the registered Round-4B SVD basis; "
            "ranks 97 and 170 were used with no refit or other basis fitting in "
            "Salvage B.",
            "",
            f"- Basis: `{summary['provenance']['basis_path']}` "
            f"(`{summary['provenance']['basis_sha256']}`).",
            f"- Split: `{summary['provenance']['split_path']}` "
            f"(`{summary['provenance']['split_sha256']}`).",
            "- Basis family: frozen Round-4B SVD; `refit=False`.",
            "",
            "## Native world objective and data",
            "",
            "The primary metric is pure-noise native 10-step future-latent reconstruction "
            "MSE (lower is better). Each inference starts from a frozen future-latent "
            "Gaussian draw; the real future latent is used only as the terminal scoring "
            "target. Exactly four pre-frozen draws are averaged within each sample.",
            "",
            "The world set contains 100 deterministic official LIBERO-Spatial clips: ten "
            "tasks × ten distinct episodes. The local official dataset exposes no checkpoint "
            "test split; these clips are held out from Round-4B basis fitting and were frozen "
            "before metric computation.",
            "",
            "The world manifest establishes separation at the source-namespace and exact-"
            "artifact provenance level: `"
            f"{summary['world_dataset']['source_separation']['basis_fit_namespace']}` versus `"
            f"{summary['world_dataset']['source_separation']['world_evaluation_namespace']}`. "
            "It does **not** establish a semantic episode/seed crosswalk across those two "
            "dataset namespaces.",
            "",
            "## Current-vs-Wrong endpoint gate",
            "",
            f"Mean native degradation `Wrong − Current` was **{_number(endpoint['mean_wrong_minus_current'])}** "
            f"with paired 95% CI [{_number(endpoint['paired_bootstrap_ci_low'])}, "
            f"{_number(endpoint['paired_bootstrap_ci_high'])}]. The median was "
            f"{_number(endpoint['median_wrong_minus_current'])} (descriptive only). "
            "The endpoint gate passed because the mean and paired-CI lower endpoint were "
            "strictly positive.",
            "",
            "## Frozen action results",
            "",
            "No action episode was rerun. These are the strictly identity-validated Round-4C outcomes.",
            "",
            "| Condition | Success | Paired 95% CI | Task-hierarchical 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["action_condition_summary"]:
        lines.append(
            f"| {row['display_name']} | {row['successes']}/{row['episodes']} "
            f"({_pct(row['success_rate'])}) | {_pct(row['paired_ci_low'])}–"
            f"{_pct(row['paired_ci_high'])} | {_pct(row['task_hierarchical_ci_low'])}–"
            f"{_pct(row['task_hierarchical_ci_high'])} |"
        )
    lines.extend(
        [
            "",
            "## Native world results",
            "",
            "Each inferential cell is `point [paired 95% CI; task-hierarchical 95% CI]`.",
            "",
            "| Condition | Mean loss | Median loss (descriptive) | Δ vs Current | Δ vs Wrong |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["world_condition_summary"]:
        lines.append(
            f"| {row['display_name']} | {_number(row['mean_native_world_loss'])} "
            f"[paired {_number(row['mean_loss_paired_ci_low'])}, "
            f"{_number(row['mean_loss_paired_ci_high'])}; task-hier "
            f"{_number(row['mean_loss_task_hierarchical_ci_low'])}, "
            f"{_number(row['mean_loss_task_hierarchical_ci_high'])}] | "
            f"{_number(row['median_native_world_loss'])} | {_number(row['delta_vs_current'])} "
            f"[paired {_number(row['delta_vs_current_paired_ci_low'])}, "
            f"{_number(row['delta_vs_current_paired_ci_high'])}; task-hier "
            f"{_number(row['delta_vs_current_task_hierarchical_ci_low'])}, "
            f"{_number(row['delta_vs_current_task_hierarchical_ci_high'])}] | "
            f"{_number(row['delta_vs_wrong'])} "
            f"[paired {_number(row['delta_vs_wrong_paired_ci_low'])}, "
            f"{_number(row['delta_vs_wrong_paired_ci_high'])}; task-hier "
            f"{_number(row['delta_vs_wrong_task_hierarchical_ci_low'])}, "
            f"{_number(row['delta_vs_wrong_task_hierarchical_ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Unclipped functional recovery",
            "",
            "Each recovery cell is `point [paired 95% CI; task-hierarchical 95% CI]`. "
            "FunctionalDissociation uses independent cross-dataset intervals.",
            "",
            "| Condition | ActionRecovery | WorldRecovery | FunctionalDissociation (independent cross-dataset) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["functional_recovery"]:
        lines.append(
            f"| {row['display_name']} | {row['action_recovery']:.3f} "
            f"[paired {row['action_recovery_paired_ci_low']:.3f}, "
            f"{row['action_recovery_paired_ci_high']:.3f}; task-hier "
            f"{row['action_recovery_task_hierarchical_ci_low']:.3f}, "
            f"{row['action_recovery_task_hierarchical_ci_high']:.3f}] | "
            f"{row['world_recovery']:.3f} [paired "
            f"{row['world_recovery_paired_ci_low']:.3f}, "
            f"{row['world_recovery_paired_ci_high']:.3f}; task-hier "
            f"{row['world_recovery_task_hierarchical_ci_low']:.3f}, "
            f"{row['world_recovery_task_hierarchical_ci_high']:.3f}] | "
            f"{row['functional_dissociation']:+.3f} "
            f"[independent {row['functional_dissociation_independent_ci_low']:+.3f}, "
            f"{row['functional_dissociation_independent_ci_high']:+.3f}; "
            f"task-hier independent "
            f"{row['functional_dissociation_task_hierarchical_independent_ci_low']:+.3f}, "
            f"{row['functional_dissociation_task_hierarchical_independent_ci_high']:+.3f}] |"
        )
    interpretation = summary["scientific_interpretation"]
    recommendation = summary["recommendation"]
    lines.extend(
        [
            "",
            "Action and world evaluations use different frozen episode sets; their "
            "FunctionalDissociation interval is therefore an independent cross-dataset "
            "bootstrap, not a falsely paired sample interval. Paired and task-hierarchical "
            "world contrasts are shown above and retained in the machine-readable artifacts.",
            "",
            "## Decision",
            "",
            f"**{classification}**",
            "",
            interpretation,
            "",
            f"Recommendation: **{recommendation}**",
            "",
            "These normalized recoveries are descriptive and unclipped. They are not "
            "information percentages, semantic labels, minimality claims, universal WAM "
            "claims, or efficiency claims.",
            "",
            "## Provenance and artifacts",
            "",
        ]
    )
    provenance = summary["provenance"]
    for key in ("git_branch", "git_commit_hash"):
        lines.append(f"- `{key}`: `{provenance[key]}`")
    hardware = summary.get("runtime_hardware", {})
    lines.extend(
        [
            f"- `pytorch`: `{hardware.get('torch_version')}`",
            f"- `cuda`: `{hardware.get('torch_cuda_version')}`",
            f"- `visible_gpu`: `{hardware.get('device_0')}`",
            f"- `machinery_timestamp`: `{summary.get('machinery_timestamp')}`",
            f"- `final_aggregate_timestamp`: `{summary.get('created_at')}`",
        ]
    )
    lines.extend(
        [
            "",
            "### All artifact paths",
            "",
            "| Artifact | Path | SHA256 / publication binding |",
            "|---|---|---|",
        ]
    )
    for artifact in summary["artifact_inventory"]:
        digest = artifact.get("sha256") or artifact["publication_state"]
        lines.append(
            f"| {artifact['name']} | `{artifact['path']}` | `{digest}` |"
        )
    lines.extend(
        [
            "",
            "## Stop rule",
            "",
            "Stop unconditionally. No Salvage C, additional rank, metric, representation "
            "axis, WAM, RoboTwin run, semantic probe, action rerun, or later ASRE experiment "
            "was launched.",
        ]
    )
    return "\n".join(lines)


def _recommendation(classification: str) -> tuple[str, str]:
    if classification == "STRONG":
        return (
            "Under a shared causal intervention on Fast-WAM's world representation, "
            "closed-loop action behavior remains intact despite a substantial loss in "
            "the representation's native world-prediction function.",
            "Paper 1 revived under functional dissociation",
        )
    if classification == "MODERATE":
        return (
            "World and action requirements differ measurably under the shared intervention, "
            "but the separation is moderate and should not be expanded into a stronger claim.",
            "Paper 1 should be frozen/closed",
        )
    return (
        "The representation information sufficient for action is not clearly distinguishable "
        "from that sufficient for native world prediction under the tested intervention.",
        "Paper 1 should be frozen/closed",
    )


def aggregate_final(
    *,
    endpoint_summary_path: Path,
    projected_summary_path: Path,
    preflight_path: Path,
    architecture_audit_path: Path,
    machinery_path: Path,
    world_manifest_path: Path,
    stochastic_manifest_path: Path,
    target_manifest_path: Path,
    output_dir: Path,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    completion_path = output_dir / FINAL_COMPLETION_FILENAME
    if completion_path.exists():
        raise FileExistsError(
            "Refusing to overwrite a published Salvage-B final bundle: "
            f"{completion_path}"
        )
    paths = (
        endpoint_summary_path,
        projected_summary_path,
        preflight_path,
        architecture_audit_path,
        machinery_path,
        world_manifest_path,
        stochastic_manifest_path,
        target_manifest_path,
    )
    if any(not path.resolve().is_file() for path in paths):
        raise FileNotFoundError("A required final Salvage-B artifact is absent.")
    preflight = _read_json(preflight_path.resolve())
    architecture = _read_json(architecture_audit_path.resolve())
    machinery = _read_json(machinery_path.resolve())
    world_manifest = _read_json(world_manifest_path.resolve())
    if not (
        preflight.get("status") == "compatible"
        and architecture.get("static_audit_passed") is True
        and architecture.get("path") == "preferred_path_a"
        and machinery.get("passed") is True
    ):
        raise ValueError("Preferred Path A did not pass all static/runtime gates.")
    basis_fit_namespace = world_manifest.get("basis_fit_namespace")
    world_evaluation_namespace = world_manifest.get("world_evaluation_namespace")
    source_overlap = world_manifest.get("source_identifier_overlap_with_basis_fit")
    overlap_scope = world_manifest.get("overlap_check_scope")
    if not (
        isinstance(basis_fit_namespace, str)
        and basis_fit_namespace
        and isinstance(world_evaluation_namespace, str)
        and world_evaluation_namespace
        and basis_fit_namespace != world_evaluation_namespace
        and source_overlap == []
        and isinstance(overlap_scope, str)
        and overlap_scope
    ):
        raise ValueError(
            "Frozen world manifest lacks the registered namespace/artifact-separation "
            "evidence or incorrectly claims overlapping source identifiers."
        )
    hashes = {
        "world_manifest_sha256": sha256_file(world_manifest_path.resolve()),
        "stochastic_manifest_sha256": sha256_file(stochastic_manifest_path.resolve()),
        "target_manifest_sha256": sha256_file(target_manifest_path.resolve()),
        "machinery_sha256": sha256_file(machinery_path.resolve()),
        "preflight_report_sha256": sha256_file(preflight_path.resolve()),
        "git_commit_hash": str(preflight["git_commit_hash"]),
    }
    endpoint = _phase_summary(
        endpoint_summary_path.resolve(),
        phase="endpoint",
        conditions=("current_all", "wrong_all"),
        expected_hashes=hashes,
    )
    projected = _phase_summary(
        projected_summary_path.resolve(),
        phase="projected",
        conditions=PROJECTED_CONDITIONS,
        expected_hashes=hashes,
    )
    if endpoint["sample_identity_sha256"] != projected["sample_identity_sha256"]:
        raise ValueError(
            "Endpoint/projected aggregates do not describe the same frozen 100-sample "
            "evaluation identity."
        )
    action_arrays, action_keys, action_rows, action_provenance = _load_action_outcomes(
        preflight=preflight
    )
    world_arrays, world_keys = _world_arrays(endpoint, projected)
    analysis = analyze_world_losses(
        world_keys,
        world_arrays,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    action_rates = {
        row["condition"]: float(row["success_rate"]) for row in action_rows
    }
    world_means = {
        condition: float(world_arrays[condition].mean()) for condition in CONDITIONS
    }
    primary_ci = analysis["conditions"][PRIMARY_CONDITION]["delta_vs_current"]
    classification = classify_functional_dissociation(
        action_success_rates=action_rates,
        world_mean_losses=world_means,
        primary_world_vs_current_paired_ci=(
            float(primary_ci["paired_ci_low"]),
            float(primary_ci["paired_ci_high"]),
        ),
        shared_interface_available=True,
        world_metric_validatable=True,
        world_endpoint_informative=True,
    )
    world_rows = _loss_rows(
        analysis=analysis,
        world_arrays=world_arrays,
        world_keys=world_keys,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    recovery_rows = _recovery_rows(
        action_arrays=action_arrays,
        action_keys=action_keys,
        world_arrays=world_arrays,
        world_keys=world_keys,
        analysis=analysis,
        classification=classification,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    action_csv = output_dir / "action_condition_summary.csv"
    world_csv = output_dir / "world_condition_summary.csv"
    recovery_csv = output_dir / "functional_recovery.csv"
    _write_csv(action_csv, action_rows)
    _write_csv(world_csv, world_rows)
    _write_csv(recovery_csv, recovery_rows)
    interpretation, recommendation = _recommendation(classification["classification"])
    summary: dict[str, Any] = {
        "artifact_type": "asre_salvage_b_final_aggregate",
        "schema_version": 1,
        "protocol": SALVAGE_B_PROTOCOL,
        "status": "complete",
        "created_at": now_iso(),
        "path": "preferred_path_a",
        "phase_a": {
            "status": "passed",
            "shared_interface": architecture["shared_interface"],
            "tensor_flow": architecture["tensor_flow"],
            "eligibility_criteria": architecture["eligibility_criteria"],
            "source_locations": architecture["source_locations"],
            "native_training_objective": architecture["native_training_objective"],
            "runtime_machinery_passed": True,
            "factorization_equivalence": machinery.get("checks", {}).get(
                "stock_joint_vs_factorized_first_pure_noise_step", {}
            ),
        },
        "native_world_metric": architecture["native_metric"],
        "world_dataset": {
            "sample_count": 100,
            "tasks": 10,
            "episodes_per_task": 10,
            "draws_per_sample": 4,
            "sample_identity_sha256": endpoint["sample_identity_sha256"],
            "split_note": preflight["world_data"]["official_split_note"],
            "held_out_from_basis_fitting": True,
            "source_separation": {
                "basis_fit_namespace": basis_fit_namespace,
                "world_evaluation_namespace": world_evaluation_namespace,
                "source_identifier_overlap_with_basis_fit": source_overlap,
                "source_namespace_and_exact_artifact_provenance_disjoint": True,
                "overlap_check_scope": overlap_scope,
                "basis_fit_overlap_evidence": world_manifest.get(
                    "basis_fit_overlap_evidence"
                ),
                "semantic_episode_or_seed_crosswalk_established": False,
                "limitation": (
                    "The registered check establishes source-namespace and exact-"
                    "artifact provenance separation only; it is not a semantic "
                    "episode/seed crosswalk across datasets."
                ),
            },
        },
        "world_endpoint_gate": endpoint["endpoint_gate"],
        "action_condition_summary": action_rows,
        "world_condition_summary": world_rows,
        "functional_recovery": recovery_rows,
        "classification": classification,
        "scientific_interpretation": interpretation,
        "recommendation": recommendation,
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "world": "paired samples plus task->episode->sample hierarchy",
            "action": "paired frozen Round-4C episodes plus task->trial hierarchy",
            "functional_gap": "independent cross-dataset bootstrap; not sample-paired",
        },
        "runtime_hardware": machinery.get("hardware", {}),
        "machinery_timestamp": machinery.get("created_at"),
        "provenance": {
            "git_branch": preflight["git"]["branch"],
            "git_commit_hash": hashes["git_commit_hash"],
            "architecture_audit_path": str(architecture_audit_path.resolve()),
            "architecture_audit_sha256": sha256_file(architecture_audit_path.resolve()),
            "preflight_path": str(preflight_path.resolve()),
            "preflight_sha256": hashes["preflight_report_sha256"],
            "machinery_path": str(machinery_path.resolve()),
            "machinery_sha256": hashes["machinery_sha256"],
            "checkpoint_path": preflight["state"]["checkpoint_path"],
            "checkpoint_sha256": preflight["state"]["checkpoint_sha256"],
            "basis_path": preflight["basis"]["path"],
            "basis_sha256": preflight["basis"]["sha256"],
            "split_path": preflight["basis"]["split_path"],
            "split_sha256": preflight["basis"]["split_sha256"],
            "donor_mapping_path": preflight["donors"]["mapping_path"],
            "donor_mapping_sha256": preflight["donors"]["mapping_sha256"],
            "donor_manifest_path": preflight["donors"]["manifest_path"],
            "donor_manifest_sha256": preflight["donors"]["manifest_sha256"],
            "world_manifest_path": str(world_manifest_path.resolve()),
            "world_manifest_sha256": hashes["world_manifest_sha256"],
            "stochastic_manifest_path": str(stochastic_manifest_path.resolve()),
            "stochastic_manifest_sha256": hashes["stochastic_manifest_sha256"],
            "target_manifest_path": str(target_manifest_path.resolve()),
            "target_manifest_sha256": hashes["target_manifest_sha256"],
            "endpoint_summary_path": str(endpoint_summary_path.resolve()),
            "endpoint_summary_sha256": sha256_file(endpoint_summary_path.resolve()),
            "projected_summary_path": str(projected_summary_path.resolve()),
            "projected_summary_sha256": sha256_file(projected_summary_path.resolve()),
            "round4c_action_summary_path": action_provenance["summary_path"],
            "round4c_action_summary_sha256": action_provenance["summary_sha256"],
            "round4c_action_execution_commit": action_provenance["execution_commit"],
            "action_rerun": False,
            "action_artifacts": action_provenance["artifacts"],
            "action_condition_summary_path": str(action_csv),
            "action_condition_summary_sha256": sha256_file(action_csv),
            "world_condition_summary_path": str(world_csv),
            "world_condition_summary_sha256": sha256_file(world_csv),
            "functional_recovery_path": str(recovery_csv),
            "functional_recovery_sha256": sha256_file(recovery_csv),
        },
        "recoveries_clipped": False,
        "later_stage_launched": False,
        "stop_rule_applied": True,
        "salvage_c_exists": False,
    }
    # Build every companion before publishing a terminal summary/sentinel.  A
    # crash during plotting therefore leaves only recoverable partial files.
    summary_path = output_dir / "salvage_b_summary.json"
    # Import lazily so technical-stop reporting never initializes matplotlib.
    from experiments.asre_diagnosis.salvage_b.plots import generate_plots

    figures = generate_plots(summary, output_dir=output_dir)
    summary["figures"] = figures
    summary["artifact_inventory"] = _normal_artifact_inventory(
        output_dir=output_dir,
        preflight=preflight,
        world_manifest=world_manifest,
        endpoint=endpoint,
        projected=projected,
        action_provenance=action_provenance,
        direct_paths={
            "architecture": architecture_audit_path.resolve(),
            "preflight": preflight_path.resolve(),
            "machinery": machinery_path.resolve(),
            "world_manifest": world_manifest_path.resolve(),
            "stochastic_manifest": stochastic_manifest_path.resolve(),
            "target_manifest": target_manifest_path.resolve(),
            "endpoint": endpoint_summary_path.resolve(),
            "projected": projected_summary_path.resolve(),
        },
        csv_paths={
            "final action condition CSV": action_csv,
            "final world condition CSV": world_csv,
            "final functional recovery CSV": recovery_csv,
        },
        figures=figures,
    )
    atomic_write_json(summary_path, summary)
    markdown = _normal_markdown(summary)
    _write_text(output_dir / "salvage_b_summary.md", markdown)
    _write_text(output_dir / "result_summary_for_gpt.md", markdown)
    _publish_completion(
        output_dir=output_dir,
        summary_path=summary_path,
        summary=summary,
        git_commit_hash=hashes["git_commit_hash"],
        figure_manifest=figures,
    )
    return summary


def _validate_special_provenance_chain(
    *,
    classification: str,
    payloads: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
    frozen_commit: str,
) -> tuple[str, ...]:
    """Fail closed unless a technical-stop label has its exact causal evidence chain."""

    architecture = payloads.get("architecture_audit")
    if architecture is None:
        raise ValueError(
            f"{classification} requires a Phase-A architecture audit artifact."
        )
    if not (
        architecture.get("artifact_type")
        == "asre_salvage_b_phase_a_architecture_audit"
        and _payload_commit(architecture) == frozen_commit
    ):
        raise ValueError("Special-report architecture audit identity is incompatible.")

    static_passed = architecture.get("static_audit_passed")
    if classification == "SHARED-INTERFACE-NOT-AVAILABLE" and static_passed is False:
        required = ("architecture_audit",)
        if not (
            architecture.get("status") == "shared_interface_not_available"
            and architecture.get("path") is None
        ):
            raise ValueError(
                "Static SHARED-INTERFACE-NOT-AVAILABLE evidence is semantically "
                "inconsistent."
            )
    else:
        if not (
            static_passed is True
            and architecture.get("status")
            == "preferred_path_a_pending_runtime_machinery"
            and architecture.get("path") == "preferred_path_a"
        ):
            raise ValueError(
                f"{classification} requires a statically valid Preferred Path A audit."
            )
        if classification in {
            "SHARED-INTERFACE-NOT-AVAILABLE",
            "WORLD-METRIC-NOT-VALIDATABLE",
        }:
            required = ("architecture_audit", "preflight", "machinery")
        else:
            required = (
                "architecture_audit",
                "preflight",
                "machinery",
                "endpoint_summary",
            )

    provided = tuple(payloads)
    missing = [label for label in required if label not in payloads]
    unexpected = [label for label in provided if label not in required]
    if missing or unexpected:
        raise ValueError(
            f"{classification} provenance chain is not exact: "
            f"missing={missing}, unexpected={unexpected}."
        )
    for label in required:
        if _payload_commit(payloads[label]) != frozen_commit:
            raise ValueError(
                f"Special-report {label} is not bound to commit {frozen_commit}."
            )

    if required == ("architecture_audit",):
        return required

    preflight = payloads["preflight"]
    preflight_path = paths["preflight"]
    architecture_path = paths["architecture_audit"]
    phase_a = preflight.get("phase_a")
    if not (
        preflight.get("artifact_type") == "asre_salvage_b_preflight_report"
        and preflight.get("protocol") == SALVAGE_B_PROTOCOL
        and preflight.get("status") == "compatible"
        and isinstance(phase_a, Mapping)
        and Path(str(phase_a.get("architecture_audit_path", ""))).resolve()
        == architecture_path
        and phase_a.get("architecture_audit_sha256")
        == sha256_file(architecture_path)
    ):
        raise ValueError(
            "Special-report preflight is not a compatible, hash-bound child of the "
            "architecture audit."
        )

    machinery = payloads["machinery"]
    machinery_preflight = machinery.get("inputs", {}).get("preflight", {})
    if not (
        machinery.get("artifact_type")
        == "asre_salvage_b_shared_interface_machinery_report"
        and machinery.get("protocol") == SALVAGE_B_PROTOCOL
        and machinery.get("preflight_report_sha256") == sha256_file(preflight_path)
        and isinstance(machinery_preflight, Mapping)
        and Path(str(machinery_preflight.get("path", ""))).resolve()
        == preflight_path
        and machinery_preflight.get("sha256") == sha256_file(preflight_path)
    ):
        raise ValueError(
            "Special-report machinery is not hash/path-bound to the compatible preflight."
        )

    if classification in {
        "SHARED-INTERFACE-NOT-AVAILABLE",
        "WORLD-METRIC-NOT-VALIDATABLE",
    }:
        if not (
            machinery.get("status") == "failed"
            and machinery.get("passed") is False
            and machinery.get("phase_b_authorized") is False
            and machinery.get("recommended_special_classification") == classification
        ):
            raise ValueError(
                f"Failed machinery does not semantically authorize {classification}."
            )
        return required

    if not (
        machinery.get("status") == "passed"
        and machinery.get("passed") is True
        and machinery.get("phase_b_authorized") is True
        and machinery.get("recommended_special_classification") is None
    ):
        raise ValueError(
            "WORLD-ENDPOINT-UNINFORMATIVE requires passed shared-interface machinery."
        )

    endpoint = payloads["endpoint_summary"]
    endpoint_gate = endpoint.get("endpoint_gate")
    if not (
        endpoint.get("artifact_type")
        == "asre_salvage_b_world_phase_aggregate"
        and endpoint.get("protocol") == SALVAGE_B_PROTOCOL
        and endpoint.get("phase") == "endpoint"
        and endpoint.get("conditions") == ["current_all", "wrong_all"]
        and endpoint.get("status") == "failed"
        and endpoint.get("passed") is False
        and endpoint.get("classification") == classification
        and endpoint.get("preflight_report_sha256") == sha256_file(preflight_path)
        and endpoint.get("machinery_sha256") == sha256_file(paths["machinery"])
        and isinstance(endpoint_gate, Mapping)
        and endpoint_gate.get("status") == "failed"
        and endpoint_gate.get("passed") is False
        and endpoint_gate.get("classification") == classification
        and endpoint_gate.get("machinery_sha256")
        == sha256_file(paths["machinery"])
        and _payload_commit(endpoint_gate) == frozen_commit
    ):
        raise ValueError(
            "Endpoint aggregate does not provide a closed, failed "
            "WORLD-ENDPOINT-UNINFORMATIVE chain."
        )
    return required


FINAL_REPORT_ITEM_TITLES = (
    "Phase-A architecture audit",
    "Exact shared representation/interface",
    "Evidence that both action and world prediction consume it",
    "Preferred Path A or Fallback Path B",
    "Exact native world objective",
    "World-evaluation dataset/split",
    "Current-vs-Wrong world endpoint check",
    "Projection/basis provenance",
    "Action results",
    "Native world results",
    "ActionRecovery table",
    "WorldRecovery table",
    "FunctionalDissociation table",
    "Paired/hierarchical uncertainty",
    "Figures",
    "Final classification",
    "Exact scientific interpretation",
    "Explicit recommendation",
    "All artifact paths",
    "Confirmation that NO later ASRE experiment was launched",
)


def _special_artifact_inventory(
    *,
    output_dir: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        name: str,
        path_value: Any,
        digest: Any = None,
        *,
        kind: str = "file",
    ) -> None:
        if not isinstance(path_value, (str, Path)) or not str(path_value):
            return
        resolved = str(Path(path_value).resolve())
        if resolved in seen:
            return
        entry = _artifact_entry(
            name,
            resolved,
            expected_sha256=str(digest) if isinstance(digest, str) else None,
            kind=kind,
        )
        entries.append(entry)
        seen.add(resolved)

    for label, path in paths.items():
        add(label.replace("_", " "), path, sha256_file(path))
        companion = path.with_suffix(".md")
        if companion.is_file():
            add(f"{label.replace('_', ' ')} Markdown", companion)

    preflight = payloads.get("preflight", {})
    state = preflight.get("state", {})
    basis = preflight.get("basis", {})
    donors = preflight.get("donors", {})
    world_data = preflight.get("world_data", {})
    frozen_action = preflight.get("frozen_action", {})
    if isinstance(state, Mapping):
        add("checkpoint", state.get("checkpoint_path"), state.get("checkpoint_sha256"))
        for label, stem in (
            ("valid state-bank manifest", "valid_manifest"),
            ("state-bank source manifest", "source_manifest"),
            ("dataset statistics", "dataset_stats"),
            ("prompt context cache", "prompt_context_cache"),
        ):
            add(label, state.get(f"{stem}_path"), state.get(f"{stem}_sha256"))
    if isinstance(basis, Mapping):
        add("frozen Round-4B SVD basis", basis.get("path"), basis.get("sha256"))
        add(
            "frozen Round-4B calibration split",
            basis.get("split_path"),
            basis.get("split_sha256"),
        )
    if isinstance(donors, Mapping):
        add("frozen donor mapping", donors.get("mapping_path"), donors.get("mapping_sha256"))
        add(
            "frozen donor observation manifest",
            donors.get("manifest_path"),
            donors.get("manifest_sha256"),
        )
        add("frozen donor root", donors.get("root"), kind="directory")
    if isinstance(world_data, Mapping):
        add("official world dataset root", world_data.get("dataset_root"), kind="directory")
        add("official dataset info", world_data.get("info_path"), world_data.get("info_sha256"))
        add("official dataset tasks", world_data.get("tasks_path"), world_data.get("tasks_sha256"))
    if isinstance(frozen_action, Mapping):
        add(
            "frozen Round-4C result root",
            frozen_action.get("round4c_root"),
            kind="directory",
        )
        add(
            "frozen Round-4C action summary",
            frozen_action.get("summary_path"),
            frozen_action.get("summary_sha256"),
        )
        action_artifacts = frozen_action.get("results", {}).get("artifacts", [])
        if isinstance(action_artifacts, list):
            for artifact in action_artifacts:
                if not isinstance(artifact, Mapping):
                    continue
                condition = artifact.get("condition", "unknown")
                add(
                    f"Round-4C {condition} metadata",
                    artifact.get("metadata_path"),
                    artifact.get("metadata_sha256"),
                )
                for index, item in enumerate(artifact.get("task_results", [])):
                    if isinstance(item, Mapping):
                        add(
                            f"Round-4C {condition} task result {index:02d}",
                            item.get("path"),
                            item.get("sha256"),
                        )

    salvage_a = preflight.get("salvage_a", {})
    if isinstance(salvage_a, Mapping):
        add(
            "Salvage-A final summary",
            salvage_a.get("summary_path"),
            salvage_a.get("summary_sha256"),
        )

    machinery = payloads.get("machinery", {})
    machinery_inputs = machinery.get("inputs", {})
    if isinstance(machinery_inputs, Mapping):
        for name, item in machinery_inputs.items():
            if isinstance(item, Mapping):
                add(f"machinery input {name}", item.get("path"), item.get("sha256"))
        world_input = machinery_inputs.get("world_manifest")
        if isinstance(world_input, Mapping) and world_input.get("path"):
            bundle = Path(str(world_input["path"])).resolve().parent / "world_bundle_complete.json"
            if bundle.is_file():
                add("frozen world bundle completion sentinel", bundle)
    endpoint = payloads.get("endpoint_summary", {})
    for key, value in endpoint.items():
        if key.endswith("_path"):
            add(
                f"endpoint {key.removesuffix('_path').replace('_', ' ')}",
                value,
                endpoint.get(f"{key.removesuffix('_path')}_sha256"),
            )
    endpoint_shards = endpoint.get("shards", [])
    if isinstance(endpoint_shards, list):
        for shard in endpoint_shards:
            if not isinstance(shard, Mapping):
                continue
            worker = shard.get("worker_index", "unknown")
            for key in ("metadata", "rows"):
                add(
                    f"endpoint worker {worker} {key}",
                    shard.get(f"{key}_path"),
                    shard.get(f"{key}_sha256"),
                )
    if "endpoint_summary" in paths:
        gate = paths["endpoint_summary"].parent / "world_endpoint_gate.json"
        if gate.is_file():
            add("standalone world endpoint gate", gate)

    for name, filename in (
        ("final JSON summary", "salvage_b_summary.json"),
        ("final human Markdown", "salvage_b_summary.md"),
        ("final GPT Markdown", "result_summary_for_gpt.md"),
        ("final completion sentinel", FINAL_COMPLETION_FILENAME),
    ):
        entries.append(_publication_entry(name, output_dir / filename))
    entries.append(
        _publication_entry(
            "driver status (finalized after aggregate)",
            output_dir.parent / "driver_status.json",
        )
    )
    return entries


def _special_markdown(summary: Mapping[str, Any], *, phase_note: str) -> str:
    classification = summary["classification"]["classification"]
    phase_a = summary.get("phase_a") or {}
    frozen = summary.get("frozen_inputs") or {}
    endpoint = summary.get("world_endpoint_gate")
    lines = ["# Fast-WAM ASRE Salvage B — Final Report", ""]

    def section(number: int, body: Sequence[str]) -> None:
        lines.extend([f"## {number}. {FINAL_REPORT_ITEM_TITLES[number - 1]}", ""])
        lines.extend(body)
        lines.append("")

    section(
        1,
        [
            f"Status: `{phase_a.get('status', 'N/A')}`.",
            f"Static audit passed: `{phase_a.get('static_audit_passed', 'N/A')}`.",
            f"Source locations: `{json.dumps(phase_a.get('source_locations'), sort_keys=True)}`.",
        ],
    )
    shared = (
        phase_a.get("shared_interface")
        if phase_a.get("static_audit_passed") is True
        else None
    )
    section(
        2,
        [
            f"`{json.dumps(shared, sort_keys=True)}`"
            if shared
            else "N/A — the technical stop did not establish an eligible shared interface."
        ],
    )
    machinery_status = summary.get("machinery_status")
    section(
        3,
        [
            "Runtime machinery passed the shared-consumer gates before the endpoint stop."
            if machinery_status == "passed"
            else "N/A — runtime machinery did not validate both consumers; see the failed machinery artifact."
        ],
    )
    section(
        4,
        [
            f"Architecture path: `{phase_a.get('path')}`. {phase_note}",
            "Fallback Path B was not used.",
        ],
    )
    native_metric = phase_a.get("native_metric")
    section(
        5,
        [
            f"`{json.dumps(native_metric, sort_keys=True)}`"
            if native_metric
            else "N/A — no valid native world objective reached evaluation."
        ],
    )
    world_data = summary.get("world_dataset_or_split")
    section(
        6,
        [
            f"`{json.dumps(world_data, sort_keys=True)}`"
            if world_data
            else "N/A — world dataset/split evaluation was not reached by this technical stop."
        ],
    )
    section(
        7,
        [
            f"`{json.dumps(endpoint, sort_keys=True)}`"
            if endpoint
            else "N/A — the Current/Wrong endpoint was not run."
        ],
    )
    basis_path = frozen.get("basis_path")
    section(
        8,
        [
            (
                "Formula: `Z_wrong + (Z_current-Z_wrong) B_r B_r^T`; frozen "
                f"Round-4B SVD basis `{basis_path}` (`{frozen.get('basis_sha256')}`), "
                "with no refit."
            )
            if basis_path
            else "N/A — projection/basis provenance was not frozen before this stop."
        ],
    )
    action_path = frozen.get("frozen_action_summary_path")
    frozen_action_results = summary.get("frozen_action_results") or {}
    action_successes = frozen_action_results.get("successes")
    episodes_per_condition = frozen_action_results.get("episodes_per_condition")
    action_body = []
    if action_path and isinstance(action_successes, Mapping):
        action_body.extend(
            [
                f"Frozen Round-4C summary: `{action_path}`; no action episode was rerun.",
                "| Condition | Frozen successes / episodes |",
                "|---|---:|",
            ]
        )
        for condition in CONDITIONS:
            action_body.append(
                f"| {DISPLAY[condition]} | {action_successes.get(condition, 'N/A')} / "
                f"{episodes_per_condition if episodes_per_condition is not None else 'N/A'} |"
            )
    elif action_path:
        action_body.append(
            f"Frozen Round-4C summary: `{action_path}`; outcome counts were not "
            "present in this technical-stop evidence chain, and no action episode was rerun."
        )
    else:
        action_body.append(
            "N/A — frozen action results were not loaded; no action episode was run."
        )
    section(
        9,
        action_body,
    )
    endpoint_results = summary.get("endpoint_condition_summary")
    section(
        10,
        [
            "Only the uninformative Current/Wrong endpoint exists; projected native-world "
            f"results were not run. Endpoint values: `{json.dumps(endpoint_results, sort_keys=True)}`."
            if endpoint and endpoint_results
            else "Only the uninformative Current/Wrong endpoint artifact exists; projected native-world results were not run."
            if endpoint
            else "N/A — native world evaluation did not produce a valid result."
        ],
    )
    section(11, ["N/A — full valid Phase B was not completed, so ActionRecovery was not computed."])
    section(12, ["N/A — full valid Phase B was not completed, so WorldRecovery was not computed."])
    section(13, ["N/A — full valid Phase B was not completed, so FunctionalDissociation was not computed."])
    section(
        14,
        [
            (
                "Endpoint paired CI: "
                f"[{endpoint.get('paired_bootstrap_ci_low')}, "
                f"{endpoint.get('paired_bootstrap_ci_high')}]; task-hierarchical CI: "
                f"[{endpoint.get('task_hierarchical_bootstrap_ci_low')}, "
                f"{endpoint.get('task_hierarchical_bootstrap_ci_high')}]."
            )
            if endpoint
            else "N/A — no valid paired/hierarchical result uncertainty was produced."
        ],
    )
    section(15, ["N/A — technical-stop reports do not generate Phase-B figures."])
    section(16, [f"**{classification}**"])
    section(
        17,
        [
            str(summary["reason"]),
            "This is a technical non-result, not evidence for or against functional dissociation.",
        ],
    )
    section(18, ["**Paper 1 should be frozen/closed.**"])
    lines.extend(
        [
            f"## 19. {FINAL_REPORT_ITEM_TITLES[18]}",
            "",
            "| Artifact | Path | SHA256 / publication binding |",
            "|---|---|---|",
        ]
    )
    for artifact in summary["artifact_inventory"]:
        digest = artifact.get("sha256") or artifact["publication_state"]
        lines.append(f"| {artifact['name']} | `{artifact['path']}` | `{digest}` |")
    lines.append("")
    section(
        20,
        [
            "Confirmed: **NO later ASRE experiment was launched.** Stop unconditionally; "
            "there is No Salvage C."
        ],
    )
    return "\n".join(lines)


def write_special_report(
    *,
    classification: str,
    reason: str,
    output_dir: Path,
    architecture_audit_path: Path | None = None,
    preflight_path: Path | None = None,
    machinery_path: Path | None = None,
    endpoint_summary_path: Path | None = None,
    git_commit_hash: str | None = None,
) -> dict[str, Any]:
    if classification not in SPECIAL_FAILURE_CLASSIFICATIONS:
        raise ValueError(f"Not a Salvage-B special failure: {classification}.")
    output_dir = output_dir.resolve()
    completion_path = output_dir / FINAL_COMPLETION_FILENAME
    if completion_path.exists():
        raise FileExistsError(
            "Refusing to overwrite a published Salvage-B technical-stop bundle: "
            f"{completion_path}"
        )
    artifacts: dict[str, Any] = {}
    artifact_payloads: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, Path] = {}
    artifact_commits: set[str] = set()
    for label, path in (
        ("architecture_audit", architecture_audit_path),
        ("preflight", preflight_path),
        ("machinery", machinery_path),
        ("endpoint_summary", endpoint_summary_path),
    ):
        if path is not None:
            resolved = path.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            artifacts[f"{label}_path"] = str(resolved)
            artifacts[f"{label}_sha256"] = sha256_file(resolved)
            payload = _read_json(resolved)
            artifact_payloads[label] = payload
            artifact_paths[label] = resolved
            recorded = _payload_commit(payload)
            if recorded is not None:
                artifact_commits.add(recorded)
    frozen_commit = git_commit_hash or git_commit(PROJECT_ROOT)
    if len(frozen_commit) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in frozen_commit
    ):
        raise ValueError(f"Malformed special-report source commit: {frozen_commit!r}.")
    if artifact_commits and artifact_commits != {frozen_commit}:
        raise ValueError(
            "Special-report artifacts do not share the frozen source commit: "
            f"artifacts={sorted(artifact_commits)}, expected={frozen_commit}."
        )
    artifact_branches = {
        branch
        for payload in artifact_payloads.values()
        if (branch := _payload_branch(payload)) is not None
    }
    if len(artifact_branches) > 1:
        raise ValueError(
            "Special-report artifacts disagree on the source branch: "
            f"{sorted(artifact_branches)}."
        )
    current_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
    ).strip()
    frozen_branch = next(iter(artifact_branches), current_branch or "DETACHED")
    validated_chain = _validate_special_provenance_chain(
        classification=classification,
        payloads=artifact_payloads,
        paths=artifact_paths,
        frozen_commit=frozen_commit,
    )
    artifacts["git_commit_hash"] = frozen_commit
    artifacts["git_branch"] = frozen_branch
    artifacts["validated_chain"] = list(validated_chain)

    architecture = artifact_payloads.get("architecture_audit")
    preflight = artifact_payloads.get("preflight")
    machinery = artifact_payloads.get("machinery")
    endpoint = artifact_payloads.get("endpoint_summary")
    phase_a = None
    if architecture is not None:
        phase_a = {
            "status": architecture.get("status"),
            "path": architecture.get("path"),
            "static_audit_passed": architecture.get("static_audit_passed"),
            "shared_interface": architecture.get("shared_interface"),
            "tensor_flow": architecture.get("tensor_flow"),
            "eligibility_criteria": architecture.get("eligibility_criteria"),
            "source_locations": architecture.get("source_locations"),
            "native_metric": architecture.get("native_metric"),
            "native_training_objective": architecture.get(
                "native_training_objective"
            ),
        }
    frozen_inputs = None
    if preflight is not None:
        frozen_inputs = {
            "checkpoint_path": preflight.get("state", {}).get("checkpoint_path"),
            "checkpoint_sha256": preflight.get("state", {}).get("checkpoint_sha256"),
            "basis_path": preflight.get("basis", {}).get("path"),
            "basis_sha256": preflight.get("basis", {}).get("sha256"),
            "split_path": preflight.get("basis", {}).get("split_path"),
            "split_sha256": preflight.get("basis", {}).get("split_sha256"),
            "donor_mapping_path": preflight.get("donors", {}).get("mapping_path"),
            "donor_mapping_sha256": preflight.get("donors", {}).get(
                "mapping_sha256"
            ),
            "donor_manifest_path": preflight.get("donors", {}).get("manifest_path"),
            "donor_manifest_sha256": preflight.get("donors", {}).get(
                "manifest_sha256"
            ),
            "frozen_action_summary_path": preflight.get("frozen_action", {}).get(
                "summary_path"
            ),
            "frozen_action_summary_sha256": preflight.get("frozen_action", {}).get(
                "summary_sha256"
            ),
            "action_rerun": False,
        }
    summary = {
        "artifact_type": "asre_salvage_b_final_aggregate",
        "schema_version": 1,
        "protocol": SALVAGE_B_PROTOCOL,
        "status": "stopped",
        "created_at": now_iso(),
        "classification": {
            "classification": classification,
            "special_failure": True,
            "primary_condition": PRIMARY_CONDITION,
            "stop_unconditionally": True,
        },
        "reason": str(reason),
        "provenance": artifacts,
        "phase_a": phase_a,
        "frozen_inputs": frozen_inputs,
        "frozen_action_results": (
            preflight.get("frozen_action", {}).get("results") if preflight else None
        ),
        "world_dataset_or_split": preflight.get("world_data") if preflight else None,
        "runtime_hardware": machinery.get("hardware", {}) if machinery else {},
        "machinery_status": machinery.get("status") if machinery else None,
        "machinery_timestamp": machinery.get("created_at") if machinery else None,
        "world_endpoint_gate": endpoint.get("endpoint_gate") if endpoint else None,
        "endpoint_condition_summary": (
            endpoint.get("condition_summary") if endpoint else None
        ),
        "phase_b_projected_conditions_run": (
            classification not in {
                "SHARED-INTERFACE-NOT-AVAILABLE",
                "WORLD-METRIC-NOT-VALIDATABLE",
                "WORLD-ENDPOINT-UNINFORMATIVE",
            }
        ),
        "figures": [],
        "action_rerun": False,
        "online_episodes": 0,
        "environment_rollouts": 0,
        "later_stage_launched": False,
        "stop_rule_applied": True,
        "salvage_c_exists": False,
    }
    summary["artifact_inventory"] = _special_artifact_inventory(
        output_dir=output_dir,
        payloads=artifact_payloads,
        paths=artifact_paths,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "salvage_b_summary.json"
    phase_note = {
        "SHARED-INTERFACE-NOT-AVAILABLE": "Phase B was not launched.",
        "WORLD-METRIC-NOT-VALIDATABLE": "World metric evaluation was not launched.",
        "WORLD-ENDPOINT-UNINFORMATIVE": (
            "Only Current/Wrong endpoint evaluation ran; projected r97/r170 did not launch."
        ),
    }[classification]
    atomic_write_json(summary_path, summary)
    markdown = _special_markdown(summary, phase_note=phase_note)
    _write_text(output_dir / "salvage_b_summary.md", markdown)
    _write_text(output_dir / "result_summary_for_gpt.md", markdown)
    _publish_completion(
        output_dir=output_dir,
        summary_path=summary_path,
        summary=summary,
        git_commit_hash=frozen_commit,
        figure_manifest=None,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-summary", type=Path)
    parser.add_argument("--projected-summary", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--architecture-audit", type=Path)
    parser.add_argument("--machinery", type=Path)
    parser.add_argument("--world-manifest", type=Path)
    parser.add_argument("--stochastic-manifest", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--special-classification", choices=SPECIAL_FAILURE_CLASSIFICATIONS)
    parser.add_argument("--special-reason")
    parser.add_argument("--git-commit-hash")
    args = parser.parse_args()
    if args.special_classification is not None:
        if args.special_classification not in SPECIAL_FAILURE_CLASSIFICATIONS:
            parser.error("--special-classification must name a technical failure label.")
        summary = write_special_report(
            classification=args.special_classification,
            reason=args.special_reason or "The registered technical gate did not pass.",
            output_dir=args.output_dir,
            architecture_audit_path=args.architecture_audit,
            preflight_path=args.preflight,
            machinery_path=args.machinery,
            endpoint_summary_path=args.endpoint_summary,
            git_commit_hash=args.git_commit_hash,
        )
    else:
        required = {
            "endpoint_summary": args.endpoint_summary,
            "projected_summary": args.projected_summary,
            "preflight": args.preflight,
            "architecture_audit": args.architecture_audit,
            "machinery": args.machinery,
            "world_manifest": args.world_manifest,
            "stochastic_manifest": args.stochastic_manifest,
            "target_manifest": args.target_manifest,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"Normal final aggregation lacks arguments: {missing}.")
        summary = aggregate_final(
            endpoint_summary_path=args.endpoint_summary,
            projected_summary_path=args.projected_summary,
            preflight_path=args.preflight,
            architecture_audit_path=args.architecture_audit,
            machinery_path=args.machinery,
            world_manifest_path=args.world_manifest,
            stochastic_manifest_path=args.stochastic_manifest,
            target_manifest_path=args.target_manifest,
            output_dir=args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    print(f"Salvage-B final classification: {summary['classification']['classification']}")


if __name__ == "__main__":
    main()
