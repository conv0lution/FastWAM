"""Generate five main ASRE Round-2 figures and the exploratory replan plot."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import build_round2_conditions  # noqa: E402


NUM_LAYERS = 30
BASELINE = "baseline_round2"
DISPLAY_NAMES = {
    "baseline_round2": "baseline",
    "keep_15_29": "keep 15–29",
    "keep_20_29": "keep 20–29",
    "keep_25_29": "keep 25–29",
    "keep_00_14": "keep 0–14",
    "keep_00_19": "keep 0–19",
    "keep_15_19": "keep 15–19",
    "keep_15_19_25_29": "keep 15–19 + 25–29",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all five main ASRE Round-2 figures plus replan-stage analysis."
    )
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(record: Mapping[str, str], key: str) -> float:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or nonnumeric {key!r} in aggregate record {record}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key!r} in aggregate record {record}.")
    return value


def _ordered_summary(
    rows: Sequence[dict[str, str]], condition_order: Sequence[str]
) -> list[dict[str, str]]:
    by_condition: dict[str, dict[str, str]] = {}
    for row in rows:
        condition = row.get("condition", "")
        if condition in by_condition:
            raise ValueError(f"Duplicate summary row for {condition!r}.")
        by_condition[condition] = row
    expected = set(condition_order)
    actual = set(by_condition)
    if actual != expected:
        raise ValueError(
            "Summary conditions do not exactly match ASRE Round 2: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )
    return [by_condition[condition] for condition in condition_order]


def _validate_task_rows(
    rows: Sequence[dict[str, str]], condition_order: Sequence[str]
) -> tuple[list[int], dict[int, str], dict[tuple[str, int], dict[str, str]]]:
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    descriptions: dict[int, str] = {}
    for row in rows:
        condition = str(row.get("condition", ""))
        if condition not in condition_order:
            raise ValueError(f"Unexpected condition {condition!r} in task table.")
        task_id = int(row["task_id"])
        key = (condition, task_id)
        if key in lookup:
            raise ValueError(f"Duplicate task table row {key}.")
        lookup[key] = row
        description = str(row.get("task_description", "")).strip()
        if not description:
            raise ValueError(f"Missing task description for {key}.")
        if task_id in descriptions and descriptions[task_id] != description:
            raise ValueError(f"Inconsistent descriptions for task {task_id}.")
        descriptions[task_id] = description
    task_ids = sorted(descriptions)
    expected_keys = {
        (condition, task_id) for condition in condition_order for task_id in task_ids
    }
    if set(lookup) != expected_keys:
        raise ValueError(
            "Task table is incomplete: "
            f"missing={sorted(expected_keys - set(lookup))}, "
            f"unexpected={sorted(set(lookup) - expected_keys)}."
        )
    return task_ids, descriptions, lookup


def _validate_replan_rows(
    rows: Sequence[dict[str, str]], condition_order: Sequence[str]
) -> dict[tuple[str, int], dict[str, str]]:
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        condition = str(row.get("condition", ""))
        if condition not in condition_order:
            raise ValueError(f"Unexpected condition {condition!r} in replan table.")
        replan_id = int(row["replan_id"])
        key = (condition, replan_id)
        if key in lookup:
            raise ValueError(f"Duplicate replan table row {key}.")
        lookup[key] = row
    expected = {
        (condition, replan_id)
        for condition in condition_order
        for replan_id in range(5)
    }
    if set(lookup) != expected:
        raise ValueError(
            "Replan table is incomplete: "
            f"missing={sorted(expected - set(lookup))}, "
            f"unexpected={sorted(set(lookup) - expected)}."
        )
    return lookup


def _short_description(task_id: int, description: str, limit: int = 66) -> str:
    clean = " ".join(description.split())
    if len(clean) > limit:
        clean = clean[: limit - 1].rstrip() + "…"
    return f"{task_id}: {clean}"


def main() -> None:
    args = _parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError(
            "Round-2 plotting requires matplotlib in the analysis environment."
        ) from exc

    conditions = build_round2_conditions(NUM_LAYERS)
    condition_order = [condition.name for condition in conditions]
    intervention_order = condition_order[1:]
    aggregate_dir = args.aggregate_dir.resolve()
    output_dir = (args.output_dir or aggregate_dir / "plots").resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Plot output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Refusing to modify a non-empty Round-2 plot directory. "
            f"Use a new output directory or explicitly remove a known partial run: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _ordered_summary(
        _read_csv(aggregate_dir / "summary.csv"), condition_order
    )
    summary = {row["condition"]: row for row in summary_rows}
    task_ids, task_descriptions, task_lookup = _validate_task_rows(
        _read_csv(aggregate_dir / "task_success_delta.csv"), condition_order
    )
    replan_lookup = _validate_replan_rows(
        _read_csv(aggregate_dir / "replan_stage_metrics.csv"), condition_order
    )

    # Figure 1: paired behavioral effect, with both requested uncertainty views.
    x = np.arange(len(intervention_order))
    delta = np.asarray(
        [_finite(summary[condition], "delta_success_rate") for condition in intervention_order]
    ) * 100.0
    paired_low = np.asarray(
        [_finite(summary[condition], "paired_delta_ci_low") for condition in intervention_order]
    ) * 100.0
    paired_high = np.asarray(
        [_finite(summary[condition], "paired_delta_ci_high") for condition in intervention_order]
    ) * 100.0
    hierarchical_low = np.asarray(
        [
            _finite(summary[condition], "task_hierarchical_delta_ci_low")
            for condition in intervention_order
        ]
    ) * 100.0
    hierarchical_high = np.asarray(
        [
            _finite(summary[condition], "task_hierarchical_delta_ci_high")
            for condition in intervention_order
        ]
    ) * 100.0

    fig, axis = plt.subplots(figsize=(11.5, 5.4))
    axis.vlines(
        x,
        hierarchical_low,
        hierarchical_high,
        color="#4C78A8",
        linewidth=5.0,
        alpha=0.35,
        zorder=1,
    )
    axis.vlines(
        x,
        paired_low,
        paired_high,
        color="#1F4E79",
        linewidth=1.5,
        zorder=2,
    )
    axis.scatter(x, delta, color="#1F4E79", s=36, zorder=3)
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.axhline(-5.0, color="#D95F02", linewidth=1.0, linestyle="--", alpha=0.8)
    axis.set_xticks(x, [DISPLAY_NAMES[name] for name in intervention_order], rotation=28, ha="right")
    axis.set_ylabel("Δ success rate vs Round-2 baseline (percentage points)")
    axis.set_xlabel("Enabled video-retrieval schedule")
    axis.set_title("Retrieval schedule vs behavioral performance")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(
        handles=[
            Line2D([0], [0], color="#1F4E79", marker="o", label="Paired bootstrap 95% CI"),
            Line2D([0], [0], color="#4C78A8", linewidth=5, alpha=0.35, label="Task-hierarchical 95% CI"),
            Line2D([0], [0], color="#D95F02", linestyle="--", label="−5 pp screening threshold"),
        ],
        loc="best",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "figure_1_retrieval_schedule_vs_behavioral_performance.png",
        dpi=args.dpi,
    )
    plt.close(fig)

    # Figure 2: executed-prefix sensitivity, clustered by baseline episode.
    offline = np.asarray(
        [
            _finite(summary[condition], "executed_prefix_norm_rms")
            for condition in intervention_order
        ]
    )
    offline_low = np.asarray(
        [
            _finite(summary[condition], "executed_prefix_norm_rms_cluster_ci_low")
            for condition in intervention_order
        ]
    )
    offline_high = np.asarray(
        [
            _finite(summary[condition], "executed_prefix_norm_rms_cluster_ci_high")
            for condition in intervention_order
        ]
    )
    fig, axis = plt.subplots(figsize=(11.5, 5.2))
    axis.vlines(
        x,
        offline_low,
        offline_high,
        color="#6A3D9A",
        linewidth=1.5,
    )
    axis.scatter(x, offline, color="#6A3D9A", s=36, zorder=2)
    baseline_offline = _finite(summary[BASELINE], "executed_prefix_norm_rms")
    axis.axhline(
        baseline_offline,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label="Round-2 baseline replay",
    )
    axis.set_xticks(x, [DISPLAY_NAMES[name] for name in intervention_order], rotation=28, ha="right")
    axis.set_ylabel("Executed-prefix normalized RMS (actions 0–9)")
    axis.set_xlabel("Enabled video-retrieval schedule")
    axis.set_title("Retrieval schedule vs executed-action sensitivity")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(
        output_dir / "figure_2_retrieval_schedule_vs_executed_action_sensitivity.png",
        dpi=args.dpi,
    )
    plt.close(fig)

    # Figure 3: direct visual-retrieval access schematic.
    enabled_matrix = np.zeros((len(conditions), NUM_LAYERS), dtype=int)
    for row_index, condition in enumerate(conditions):
        enabled_matrix[
            row_index, list(condition.enabled_video_retrieval_layers(NUM_LAYERS))
        ] = 1
    fig, axis = plt.subplots(figsize=(14.2, 4.8))
    image = axis.imshow(
        enabled_matrix,
        cmap=ListedColormap(["#ECECEC", "#2166AC"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    del image
    axis.set_xticks(np.arange(NUM_LAYERS), [str(layer) for layer in range(NUM_LAYERS)], fontsize=7)
    axis.set_yticks(
        np.arange(len(condition_order)),
        [DISPLAY_NAMES[name] for name in condition_order],
    )
    axis.set_xlabel("Action layer")
    axis.set_ylabel("Retrieval schedule")
    axis.set_title("Enabled direct video K/V retrieval by action layer")
    axis.set_xticks(np.arange(-0.5, NUM_LAYERS, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(condition_order), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.55)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.legend(
        handles=[
            Line2D([0], [0], marker="s", linestyle="none", color="#2166AC", markersize=9, label="video K/V enabled"),
            Line2D([0], [0], marker="s", linestyle="none", color="#ECECEC", markeredgecolor="#999999", markersize=9, label="video K/V disabled"),
        ],
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "figure_3_enabled_retrieval_depth_schematic.png", dpi=args.dpi
    )
    plt.close(fig)

    # Figure 4: task-wise paired delta. Baseline is retained as a zero reference column.
    heatmap = np.asarray(
        [
            [
                _finite(task_lookup[(condition, task_id)], "delta_success_rate") * 100.0
                for condition in condition_order
            ]
            for task_id in task_ids
        ],
        dtype=float,
    )
    max_abs = max(10.0, float(np.max(np.abs(heatmap))))
    fig, axis = plt.subplots(
        figsize=(13.5, max(6.0, 0.72 * len(task_ids) + 2.0))
    )
    image = axis.imshow(
        heatmap,
        cmap="RdBu",
        vmin=-max_abs,
        vmax=max_abs,
        aspect="auto",
    )
    axis.set_xticks(
        np.arange(len(condition_order)),
        [DISPLAY_NAMES[name] for name in condition_order],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(task_ids)),
        [_short_description(task_id, task_descriptions[task_id]) for task_id in task_ids],
        fontsize=8,
    )
    for row_index in range(heatmap.shape[0]):
        for column_index in range(heatmap.shape[1]):
            value = heatmap[row_index, column_index]
            foreground = "white" if abs(value) > 0.55 * max_abs else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:+.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color=foreground,
            )
    axis.set_xlabel("Enabled video-retrieval schedule")
    axis.set_ylabel("LIBERO-Spatial task")
    axis.set_title("Task × retrieval-schedule success-rate delta")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Δ success rate vs Round-2 baseline (percentage points)")
    fig.tight_layout()
    fig.savefig(
        output_dir / "figure_4_task_by_retrieval_schedule_heatmap.png", dpi=args.dpi
    )
    plt.close(fig)

    # Figure 5: the pre-specified complementarity view.
    complementarity_order = (
        BASELINE,
        "keep_15_19",
        "keep_25_29",
        "keep_15_19_25_29",
    )
    complementarity_x = np.arange(len(complementarity_order))
    complementarity_delta = np.asarray(
        [_finite(summary[name], "delta_success_rate") for name in complementarity_order]
    ) * 100.0
    complementarity_low = np.asarray(
        [
            _finite(summary[name], "task_hierarchical_delta_ci_low")
            for name in complementarity_order
        ]
    ) * 100.0
    complementarity_high = np.asarray(
        [
            _finite(summary[name], "task_hierarchical_delta_ci_high")
            for name in complementarity_order
        ]
    ) * 100.0
    absolute_rates = np.asarray(
        [_finite(summary[name], "online_success_rate") for name in complementarity_order]
    ) * 100.0
    fig, axis = plt.subplots(figsize=(8.8, 5.3))
    axis.plot(
        complementarity_x,
        complementarity_delta,
        "o-",
        color="#238B45",
        linewidth=1.4,
        markersize=7,
    )
    axis.vlines(
        complementarity_x,
        complementarity_low,
        complementarity_high,
        color="#238B45",
        linewidth=1.5,
    )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.axhline(-5.0, color="#D95F02", linewidth=1.0, linestyle="--", alpha=0.8)
    for position, delta_value, rate in zip(
        complementarity_x, complementarity_delta, absolute_rates
    ):
        axis.annotate(
            f"SR={rate:.0f}%",
            (position, delta_value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axis.set_xticks(
        complementarity_x,
        [DISPLAY_NAMES[name] for name in complementarity_order],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("Δ success rate vs Round-2 baseline (percentage points)")
    axis.set_xlabel("Critical-window retrieval schedule")
    axis.set_title("Critical-window complementarity")
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(
        output_dir / "figure_5_critical_window_complementarity.png", dpi=args.dpi
    )
    plt.close(fig)

    # Exploratory replan-stage view. All states come from baseline visitation.
    replans = np.arange(5)
    colors = plt.get_cmap("tab10").colors
    fig, axis = plt.subplots(figsize=(10.2, 6.0))
    for condition_index, condition in enumerate(condition_order):
        rows = [replan_lookup[(condition, replan_id)] for replan_id in replans]
        centers = np.asarray(
            [_finite(row, "executed_prefix_norm_rms") for row in rows]
        )
        lows = np.asarray(
            [_finite(row, "episode_cluster_ci_low") for row in rows]
        )
        highs = np.asarray(
            [_finite(row, "episode_cluster_ci_high") for row in rows]
        )
        color = "black" if condition == BASELINE else colors[condition_index % len(colors)]
        axis.plot(
            replans,
            centers,
            marker="o",
            linewidth=2.1 if condition == BASELINE else 1.35,
            linestyle="--" if condition == BASELINE else "-",
            color=color,
            label=DISPLAY_NAMES[condition],
        )
        axis.fill_between(replans, lows, highs, color=color, alpha=0.10)
    axis.set_xticks(replans)
    axis.set_xlabel("Saved policy-query / replan index")
    axis.set_ylabel("Executed-prefix normalized RMS (actions 0–9)")
    axis.set_title("Exploratory replan-stage sensitivity (baseline visitation states)")
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(
        output_dir / "exploratory_replan_stage_action_deviation.png", dpi=args.dpi
    )
    plt.close(fig)

    print(f"Wrote five main figures and one exploratory replan figure to {output_dir}")


if __name__ == "__main__":
    main()
