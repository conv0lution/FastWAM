"""Aggregate the entirely new native action/world v2 outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.asre_diagnosis.common import (
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.salvage_b.statistics import (
    analyze_world_losses,
    paired_bootstrap_ci,
    task_hierarchical_bootstrap_ci,
    task_hierarchical_bootstrap_indices,
)

from .classification import classify
from .definitions import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, CONDITIONS, PROTOCOL


OLD_NAME = {
    "current": "current_all",
    "wrong": "wrong_all",
    "svd_r97": "svd_r97",
    "svd_r170": "svd_r170",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _action_vectors(root: Path) -> tuple[list[tuple[int, int, str]], dict[str, np.ndarray]]:
    rows: dict[str, dict[tuple[int, int], float]] = {}
    descriptions: dict[int, str] = {}
    for condition in CONDITIONS:
        condition_rows: dict[tuple[int, int], float] = {}
        paths = sorted((root / "action_full" / condition / "libero_spatial").glob("gpu*_task*_results.json"))
        if len(paths) != 10:
            raise ValueError(f"Condition {condition} lacks ten task result files.")
        for path in paths:
            payload = _read(path)
            task = int(payload["task_id"])
            descriptions[task] = str(payload["task_description"])
            success = set(map(int, payload["success_episodes"]))
            failure = set(map(int, payload["failure_episodes"]))
            if success & failure or success | failure != set(range(10)):
                raise ValueError(f"Incomplete paired action outcomes: {path}")
            for trial in range(10):
                condition_rows[(task, trial)] = float(trial in success)
        rows[condition] = condition_rows
    identities = sorted(rows["current"])
    if any(sorted(rows[condition]) != identities for condition in CONDITIONS):
        raise ValueError("Action episode identities are not paired across conditions.")
    keys = [(task, trial, descriptions[task]) for task, trial in identities]
    return keys, {
        condition: np.asarray([rows[condition][identity] for identity in identities])
        for condition in CONDITIONS
    }


def _world_vectors(root: Path) -> tuple[list[tuple[int, int, str]], dict[str, np.ndarray]]:
    rows_by_condition: dict[str, dict[str, list[dict[str, Any]]]] = {}
    identity: dict[str, tuple[int, int, str]] = {}
    for condition in CONDITIONS:
        payload = _read(root / "world" / condition / "rows.json")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in payload["rows"]:
            if row.get("native_joint_graph") is not True or row.get("factorized_helper_used") is not False:
                raise ValueError("A world row did not use the native joint graph exclusively.")
            sample_id = str(row["sample_id"])
            grouped[sample_id].append(row)
            identity[sample_id] = (int(row["task_id"]), int(row["episode_id"]), sample_id)
        if len(grouped) != 100 or any(len(draws) != 4 for draws in grouped.values()):
            raise ValueError(f"World condition {condition} is not 100 samples x four draws.")
        rows_by_condition[condition] = grouped
    sample_ids = sorted(rows_by_condition["current"])
    if any(sorted(rows_by_condition[c]) != sample_ids for c in CONDITIONS):
        raise ValueError("World samples are not exactly paired across conditions.")
    return [identity[sample] for sample in sample_ids], {
        condition: np.asarray(
            [np.mean([float(row["loss"]) for row in rows_by_condition[condition][sample]]) for sample in sample_ids],
            dtype=np.float64,
        )
        for condition in CONDITIONS
    }


def _interval(keys, values: np.ndarray, label: str) -> dict[str, float]:
    paired = paired_bootstrap_ci(values, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED + sum(map(ord, label)))
    hierarchical = task_hierarchical_bootstrap_ci(
        keys, values, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED + 1000 + sum(map(ord, label))
    )
    return {
        "point": float(values.mean()),
        "paired_low": paired[0],
        "paired_high": paired[1],
        "task_hierarchical_low": hierarchical[0],
        "task_hierarchical_high": hierarchical[1],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _recovery_bootstrap(
    *,
    keys,
    condition: np.ndarray,
    current: np.ndarray,
    wrong: np.ndarray,
    world: bool,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    def ratio(means: np.ndarray) -> np.ndarray:
        c, cur, bad = means[:, 0], means[:, 1], means[:, 2]
        denominator = bad - cur if world else cur - bad
        valid = np.abs(denominator) > np.finfo(np.float64).eps
        result = (
            1.0 - (c[valid] - cur[valid]) / denominator[valid]
            if world
            else (c[valid] - bad[valid]) / denominator[valid]
        )
        return result

    matrix = np.stack((condition, current, wrong), axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(condition), size=(BOOTSTRAP_SAMPLES, len(condition)))
    paired = ratio(matrix[indices].mean(axis=1))
    hierarchical_indices = task_hierarchical_bootstrap_indices(
        keys, samples=BOOTSTRAP_SAMPLES, seed=seed + 1
    )
    hierarchical = ratio(
        np.asarray([matrix[index].mean(axis=0) for index in hierarchical_indices])
    )
    point_means = matrix.mean(axis=0)
    point = float(ratio(point_means[None, :])[0])
    return (
        {
            "point": point,
            "paired_low": float(np.percentile(paired, 2.5)),
            "paired_high": float(np.percentile(paired, 97.5)),
            "task_hierarchical_low": float(np.percentile(hierarchical, 2.5)),
            "task_hierarchical_high": float(np.percentile(hierarchical, 97.5)),
            "unclipped": True,
        },
        paired,
        hierarchical,
    )


def _plots(aggregate: Path, action: dict[str, float], world: dict[str, float], recoveries: dict[str, Any]) -> list[Path]:
    import matplotlib.pyplot as plt

    plot_dir = aggregate / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    specs = (
        ("figure_A_action_success.png", "Action success", action, "Success"),
        ("figure_B_native_world_loss.png", "Native future loss", world, "MSE"),
        (
            "figure_C_action_recovery.png",
            "Action recovery",
            {c: recoveries[c]["action_recovery"] for c in CONDITIONS},
            "Recovery",
        ),
        (
            "figure_D_world_recovery.png",
            "World recovery",
            {c: recoveries[c]["world_recovery"] for c in CONDITIONS},
            "Recovery",
        ),
        (
            "figure_E_functional_dissociation.png",
            "Functional dissociation",
            {c: recoveries[c]["functional_dissociation"] for c in CONDITIONS},
            "Action - world recovery",
        ),
    )
    for name, title, values, ylabel in specs:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(CONDITIONS)), [values[c] for c in CONDITIONS])
        ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        path = plot_dir / name
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    preflight = _read(root / "preflight_report.json")
    machinery = _read(root / "machinery_report.json")
    action_keys, action_vectors = _action_vectors(root)
    world_keys, world_vectors = _world_vectors(root)
    action_rates = {condition: float(values.mean()) for condition, values in action_vectors.items()}
    world_means = {condition: float(values.mean()) for condition, values in world_vectors.items()}
    action_rows = []
    for condition in CONDITIONS:
        ci = _interval(action_keys, action_vectors[condition], f"action_{condition}")
        action_rows.append({"condition": condition, "successes": int(action_vectors[condition].sum()), "episodes": 100, **ci})
    endpoint_delta = action_vectors["current"] - action_vectors["wrong"]
    endpoint_ci = paired_bootstrap_ci(endpoint_delta, samples=BOOTSTRAP_SAMPLES, seed=42001)
    action_endpoint_valid = action_rates["current"] >= 0.5 and endpoint_ci[0] > 0.0

    old_world = {OLD_NAME[c]: world_vectors[c] for c in CONDITIONS}
    old_world_keys = [(task, episode, sample) for task, episode, sample in world_keys]
    world_analysis = analyze_world_losses(old_world_keys, old_world)
    world_endpoint_informative = bool(world_analysis["endpoint_gate"]["informative"])
    r170_world_ci = (
        float(world_analysis["conditions"]["svd_r170"]["delta_vs_current"]["paired_ci_low"]),
        float(world_analysis["conditions"]["svd_r170"]["delta_vs_current"]["paired_ci_high"]),
    )
    decision = classify(
        machinery=machinery,
        action_success=action_rates,
        world_loss=world_means,
        action_endpoint_valid=action_endpoint_valid,
        world_endpoint_informative=world_endpoint_informative,
        r170_world_minus_current_ci=r170_world_ci,
    )
    recoveries = decision.get("recoveries", {})
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    _write_csv(aggregate / "action_condition_summary.csv", action_rows)
    world_rows = [
        {
            "condition": condition,
            "mean_loss": world_means[condition],
            "delta_vs_current": world_means[condition] - world_means["current"],
            "delta_vs_wrong": world_means[condition] - world_means["wrong"],
        }
        for condition in CONDITIONS
    ]
    _write_csv(aggregate / "world_condition_summary.csv", world_rows)
    recovery_rows = []
    recovery_statistics: dict[str, Any] = {}
    if recoveries:
        for index, condition in enumerate(CONDITIONS):
            action_stats, action_paired, action_hierarchical = _recovery_bootstrap(
                keys=action_keys,
                condition=action_vectors[condition],
                current=action_vectors["current"],
                wrong=action_vectors["wrong"],
                world=False,
                seed=51000 + index * 10,
            )
            world_stats, world_paired, world_hierarchical = _recovery_bootstrap(
                keys=world_keys,
                condition=world_vectors[condition],
                current=world_vectors["current"],
                wrong=world_vectors["wrong"],
                world=True,
                seed=61000 + index * 10,
            )
            paired_dissociation = action_paired[: min(len(action_paired), len(world_paired))] - world_paired[: min(len(action_paired), len(world_paired))]
            hierarchical_dissociation = action_hierarchical[: min(len(action_hierarchical), len(world_hierarchical))] - world_hierarchical[: min(len(action_hierarchical), len(world_hierarchical))]
            dissociation_stats = {
                "point": recoveries[condition]["functional_dissociation"],
                "independent_low": float(np.percentile(paired_dissociation, 2.5)),
                "independent_high": float(np.percentile(paired_dissociation, 97.5)),
                "task_hierarchical_independent_low": float(np.percentile(hierarchical_dissociation, 2.5)),
                "task_hierarchical_independent_high": float(np.percentile(hierarchical_dissociation, 97.5)),
            }
            recovery_statistics[condition] = {
                "action": action_stats,
                "world": world_stats,
                "functional_dissociation": dissociation_stats,
            }
            recovery_rows.append(
                {
                    "condition": condition,
                    "action_recovery": action_stats["point"],
                    "action_paired_ci_low": action_stats["paired_low"],
                    "action_paired_ci_high": action_stats["paired_high"],
                    "action_task_hierarchical_ci_low": action_stats["task_hierarchical_low"],
                    "action_task_hierarchical_ci_high": action_stats["task_hierarchical_high"],
                    "world_recovery": world_stats["point"],
                    "world_paired_ci_low": world_stats["paired_low"],
                    "world_paired_ci_high": world_stats["paired_high"],
                    "world_task_hierarchical_ci_low": world_stats["task_hierarchical_low"],
                    "world_task_hierarchical_ci_high": world_stats["task_hierarchical_high"],
                    "functional_dissociation": dissociation_stats["point"],
                    "dissociation_independent_ci_low": dissociation_stats["independent_low"],
                    "dissociation_independent_ci_high": dissociation_stats["independent_high"],
                    "dissociation_task_hierarchical_independent_ci_low": dissociation_stats["task_hierarchical_independent_low"],
                    "dissociation_task_hierarchical_independent_ci_high": dissociation_stats["task_hierarchical_independent_high"],
                }
            )
    if recovery_rows:
        _write_csv(aggregate / "functional_recovery.csv", recovery_rows)
        figure_paths = _plots(aggregate, action_rates, world_means, recoveries)
    else:
        figure_paths = []
    task_rows = []
    for task in range(10):
        action_index = [index for index, key in enumerate(action_keys) if key[0] == task]
        world_index = [index for index, key in enumerate(world_keys) if key[0] == task]
        for condition in CONDITIONS:
            task_rows.append(
                {
                    "task_id": task,
                    "condition": condition,
                    "action_success": float(action_vectors[condition][action_index].mean()),
                    "native_world_mean_loss": float(world_vectors[condition][world_index].mean()),
                }
            )
    _write_csv(aggregate / "task_condition_summary.csv", task_rows)
    test_evidence = {}
    for name in ("unit_tests", "full_asre_tests"):
        path = root / "logs" / f"{name}.log"
        if path.is_file():
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if "passed" in line]
            test_evidence[name] = lines[-1] if lines else "completed"
    summary = {
        "artifact_type": "asre_salvage_b_v2_final_summary",
        "protocol": PROTOCOL,
        "status": "complete",
        "created_at": now_iso(),
        "git_commit_hash": git_commit(Path(__file__).resolve().parents[3]),
        "classification": decision["classification"],
        "decision": decision,
        "machinery": machinery,
        "action_success": action_rates,
        "action_endpoint_valid": action_endpoint_valid,
        "action_endpoint_paired_ci": list(endpoint_ci),
        "world_mean_loss": world_means,
        "world_analysis": world_analysis,
        "world_endpoint_informative": world_endpoint_informative,
        "recovery_statistics": recovery_statistics,
        "old_factorized_results_reused": False,
        "native_joint_graph_only": True,
        "stop_unconditionally": True,
        "figures": [str(path) for path in figure_paths],
        "test_evidence": test_evidence,
        "provenance": {
            "preflight_path": str(root / "preflight_report.json"),
            "preflight_sha256": sha256_file(root / "preflight_report.json"),
            "basis_manifest_path": preflight["basis"]["path"],
            "basis_manifest_sha256": preflight["basis"]["sha256"],
            "basis_coordinate_report_path": machinery[
                "basis_coordinate_report_path"
            ],
            "basis_coordinate_report_sha256": sha256_file(
                Path(machinery["basis_coordinate_report_path"])
            ),
            "donor_mapping_path": preflight["donors"]["mapping_path"],
            "donor_mapping_sha256": preflight["donors"]["mapping_sha256"],
            "world_manifest_path": preflight["reused_frozen_inputs"]["world_manifest_path"],
            "world_manifest_sha256": preflight["reused_frozen_inputs"]["world_manifest_sha256"],
            "draw_tensors_path": preflight["reused_frozen_inputs"]["draw_tensors_path"],
            "draw_tensors_sha256": preflight["reused_frozen_inputs"]["draw_tensors_sha256"],
        },
    }
    json_path = aggregate / "salvage_b_v2_summary.json"
    atomic_write_json(json_path, summary)
    lines = [
        "# Fast-WAM ASRE Salvage B v2 — Final Report",
        "",
        "All action and world outcomes below were newly generated through the same native stock joint graph.",
        "The quarantined factorized Salvage-B outcomes were not reused.",
        "",
        "## New action outcomes",
        "",
        "| Condition | Success | Paired 95% CI | Task-hierarchical 95% CI |",
        "|---|---:|---:|---:|",
    ]
    action_by_condition = {row["condition"]: row for row in action_rows}
    for condition in CONDITIONS:
        row = action_by_condition[condition]
        lines.append(
            f"| {condition} | {action_rates[condition]:.1%} | "
            f"{row['paired_low']:.1%}–{row['paired_high']:.1%} | "
            f"{row['task_hierarchical_low']:.1%}–{row['task_hierarchical_high']:.1%} |"
        )
    lines.extend(("", "## New native-world outcomes", "", "| Condition | Mean loss | Δ vs Current (paired 95% CI) |", "|---|---:|---:|"))
    for condition in CONDITIONS:
        old_row = world_analysis["conditions"][OLD_NAME[condition]]["delta_vs_current"]
        lines.append(
            f"| {condition} | {world_means[condition]:.6f} | "
            f"{old_row['mean']:+.6f} [{old_row['paired_ci_low']:+.6f}, "
            f"{old_row['paired_ci_high']:+.6f}] |"
        )
    if recoveries:
        lines.extend(("", "## Functional recovery", "", "Each recovery is `point [paired 95% CI; task-hierarchical 95% CI]`. Dissociation uses independent action/world resampling because the episode sets differ.", "", "| Condition | ActionRecovery | WorldRecovery | Dissociation |", "|---|---:|---:|---:|"))
        for condition in CONDITIONS:
            row = recovery_statistics[condition]
            action = row["action"]
            world = row["world"]
            dissociation = row["functional_dissociation"]
            lines.append(
                f"| {condition} | {action['point']:.3f} [{action['paired_low']:.3f}, {action['paired_high']:.3f}; {action['task_hierarchical_low']:.3f}, {action['task_hierarchical_high']:.3f}] | "
                f"{world['point']:.3f} [{world['paired_low']:.3f}, {world['paired_high']:.3f}; {world['task_hierarchical_low']:.3f}, {world['task_hierarchical_high']:.3f}] | "
                f"{dissociation['point']:+.3f} [{dissociation['independent_low']:+.3f}, {dissociation['independent_high']:+.3f}; {dissociation['task_hierarchical_independent_low']:+.3f}, {dissociation['task_hierarchical_independent_high']:+.3f}] |"
            )
    interpretation = {
        "STRONG": "Functional dissociation is supported: the same native shared-node intervention preserves action while substantially reducing native world-function recovery.",
        "MODERATE": "Functional dissociation receives limited support under the registered moderate thresholds.",
        "WEAK": "Functional dissociation is not supported strongly enough; functional ASRE is closed.",
    }.get(decision["classification"], "The run ended in a registered technical classification.")
    lines.extend(
        (
            "",
            "## Decision",
            "",
            f"**{decision['classification']}**",
            "",
            interpretation,
            "",
            "## Native causal implementation",
            "",
            "The only graph used was `FastWAM.infer_joint -> _joint_denoise_core -> MoT.forward_joint_core -> MoT._forward_joint_layer`. The intervention is at `src/fastwam/models/wan22/mot.py::_forward_joint_layer`, after native video K/V construction and before its one concatenated joint attention.",
            "",
            "The intervention was applied only to native prefix K/V rows `0:98` at layers 15–29. Layers 0–14, all future K/V rows, action K/V, masks, timesteps, contexts, scheduler state, and original heads were left native. The prohibited factorized future-only helper was never called.",
            "",
            "At every denoising step, uninterrupted Current and Donor native trajectories supplied frozen exogenous K/V targets. The intervened Current pass propagated prefix hidden states normally and was re-clamped at each selected layer. Rank 0 was the exact Donor target and rank D the exact Current target.",
            "",
            "## Controls and provenance",
            "",
            f"- Machinery passed: `{machinery.get('passed')}`; exact stock/capture identity: `{machinery.get('observational_capture_exact_identity')}`; exact rank-D identity: `{machinery.get('rank_d_current_exact_identity')}`.",
            f"- Donor-observation-only gate passed: `{machinery.get('donor_observation_only_passed')}`. Donor capture changed RGB only; prompt/context, proprio, noise, scheduler objects, and schedule arguments came from the current-recipient call.",
            f"- Round-4B/native-stock basis-coordinate gate passed: `{machinery.get('basis_coordinate_gate_passed')}` across `{machinery.get('basis_coordinate_sample_count')}` frozen states and all 30 late K/V matrices; detailed artifact: `{machinery.get('basis_coordinate_report_path')}`.",
            f"- Frozen basis: `{preflight['basis']['path']}` (`{preflight['basis']['sha256']}`). No refit was performed.",
            f"- Frozen donor mapping: `{preflight['donors']['mapping_path']}` (`{preflight['donors']['mapping_sha256']}`).",
            "- The official 100-sample world identities, processed targets, and four stochastic starts were reused as pre-outcome inputs only. Their old factorized losses were not read.",
            f"- Tests: `{test_evidence}`.",
            "",
            "## Artifact locations",
            "",
            f"- Final JSON: `{json_path}`",
            f"- Action table: `{aggregate / 'action_condition_summary.csv'}`",
            f"- World table: `{aggregate / 'world_condition_summary.csv'}`",
            f"- Recovery table: `{aggregate / 'functional_recovery.csv'}`",
            f"- Task table: `{aggregate / 'task_condition_summary.csv'}`",
            f"- Figures: `{aggregate / 'plots'}`",
            "",
            "Stop unconditionally. No later ASRE experiment was launched.",
        )
    )
    markdown = "\n".join(lines) + "\n"
    for name in ("salvage_b_v2_summary.md", "result_summary_for_gpt.md"):
        (aggregate / name).write_text(markdown, encoding="utf-8")
    completion = {
        "status": "complete",
        "classification": decision["classification"],
        "summary_path": str(json_path),
        "summary_sha256": sha256_file(json_path),
    }
    atomic_write_json(aggregate / "salvage_b_v2_completion.json", completion)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
