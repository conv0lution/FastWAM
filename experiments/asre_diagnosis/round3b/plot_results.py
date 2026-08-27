"""Generate the five pre-specified ASRE Round-3B causal-control figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COLORS = ("#2E6F9E", "#D9822B", "#777777")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Required plot input is empty: {path}.")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}.")
    return payload


def _number(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite plot value {key}={value}.")
    return value


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_success(cells: Sequence[Mapping[str, Any]], path: Path) -> None:
    labels = [str(row["display_name"]).replace(" K/V", "\nK/V") for row in cells]
    rates = np.asarray([_number(row, "success_rate") for row in cells]) * 100.0
    paired_low = np.asarray([_number(row, "paired_ci_low") for row in cells]) * 100.0
    paired_high = np.asarray([_number(row, "paired_ci_high") for row in cells]) * 100.0
    task_low = np.asarray(
        [_number(row, "task_hierarchical_ci_low") for row in cells]
    ) * 100.0
    task_high = np.asarray(
        [_number(row, "task_hierarchical_ci_high") for row in cells]
    ) * 100.0
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x, rates, width=0.62, color=COLORS, edgecolor="black", linewidth=0.7)
    ax.errorbar(
        x,
        rates,
        yerr=np.vstack([rates - task_low, task_high - rates]),
        fmt="none",
        ecolor="#222222",
        elinewidth=1.2,
        capsize=6,
        label="Task-hierarchical 95% CI",
    )
    ax.errorbar(
        x,
        rates,
        yerr=np.vstack([rates - paired_low, paired_high - rates]),
        fmt="o",
        color="white",
        markeredgecolor="#111111",
        ecolor="#111111",
        elinewidth=2.0,
        capsize=3,
        label="Paired-episode 95% CI",
    )
    for position, rate in zip(x, rates):
        ax.text(position, rate + 2.0, f"{rate:.0f}%", ha="center", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylabel("LIBERO-Spatial success (%)")
    ax.set_ylim(0, max(105.0, float(task_high.max()) + 8.0))
    ax.set_title("Round 3B: correct content vs matched-shape wrong content")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_task_deltas(tasks: Sequence[Mapping[str, Any]], path: Path) -> None:
    ordered = sorted(tasks, key=lambda row: int(row["task_id"]))
    task_ids = [int(row["task_id"]) for row in ordered]
    deltas = np.asarray([_number(row, "wrong_minus_correct") for row in ordered]) * 100.0
    colors = ["#B5403C" if value < 0 else "#3B7A57" for value in deltas]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(task_ids, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.axhline(-20.0, color="#555555", linestyle="--", linewidth=1.0, label="20 pp loss")
    ax.set_xticks(task_ids)
    ax.set_xlabel("LIBERO-Spatial task")
    ax.set_ylabel("Wrong scene - correct success (pp)")
    ax.set_title("Per-task effect of replacing current scene content")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_primary_transitions(summary: Mapping[str, Any], path: Path) -> None:
    comparison = summary["analysis"]["comparisons"]["wrong_minus_correct"]
    matrix = np.asarray(
        [
            [
                int(comparison["reference_failure_to_target_failure"]),
                int(comparison["reference_failure_to_target_success"]),
            ],
            [
                int(comparison["reference_success_to_target_failure"]),
                int(comparison["reference_success_to_target_success"]),
            ],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=16,
                color="white" if value > matrix.max() / 2 else "black",
            )
    ax.set_xticks([0, 1], ["Wrong failure", "Wrong success"])
    ax.set_yticks([0, 1], ["Correct failure", "Correct success"])
    ax.set_xlabel("Wrong same-task scene K/V")
    ax.set_ylabel("Correct current K/V")
    ax.set_title("Paired episode transitions")
    fig.colorbar(image, ax=ax, label="Episode count", shrink=0.82)
    _save(fig, path)


def plot_offline_metrics(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    metrics = (
        ("executed_prefix_norm_rms", "Prefix RMS"),
        ("full_chunk_norm_rms_0_31", "Full-chunk RMS"),
        ("translation_norm_rms", "Translation RMS"),
        ("rotation_norm_rms", "Rotation RMS"),
        ("executed_prefix_cosine_similarity", "Prefix cosine"),
        ("executed_prefix_gripper_flip_rate", "Grip flip rate"),
    )
    labels = [str(row["display_name"]).split(" K/V")[0] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.7))
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = [_number(row, metric) for row in rows]
        ax.bar(x, values, color=COLORS, edgecolor="black", linewidth=0.45)
        ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Offline fixed-state action metrics (historical baseline reference)")
    fig.tight_layout()
    _save(fig, path)


def plot_kv_sanity(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharex=True)
    for ax, tensor_name in zip(axes, ("K", "V")):
        subset = sorted(
            (row for row in rows if str(row["tensor"]) == tensor_name),
            key=lambda row: int(row["layer"]),
        )
        if len(subset) != 15:
            raise ValueError(f"Expected 15 K/V rows for {tensor_name}, got {len(subset)}.")
        layers = [int(row["layer"]) for row in subset]
        current = [_number(row, "current_rms") for row in subset]
        donor = [_number(row, "donor_rms") for row in subset]
        ax.plot(layers, current, marker="o", color=COLORS[0], label="Current")
        ax.plot(layers, donor, marker="s", color=COLORS[1], label="Donor")
        ax.set_title(f"{tensor_name} RMS")
        ax.set_xlabel("Action layer")
        ax.set_xticks([15, 19, 24, 29])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean cache RMS across 499 states")
    axes[1].legend(frameon=False)
    fig.suptitle("Matched-shape donor/current K/V scale sanity")
    fig.tight_layout()
    _save(fig, path)


def generate_plots(aggregate_dir: Path) -> list[Path]:
    aggregate_dir = aggregate_dir.resolve()
    plot_dir = aggregate_dir / "plots"
    cells = _read_csv(aggregate_dir / "online_condition_summary.csv")
    tasks = _read_csv(aggregate_dir / "task_success.csv")
    offline = _read_csv(aggregate_dir / "offline_condition_metrics.csv")
    kv = _read_csv(aggregate_dir / "kv_scale_sanity.csv")
    summary = _read_json(aggregate_dir / "round3b_summary.json")
    outputs = [
        plot_dir / "figure_A_online_success.png",
        plot_dir / "figure_B_per_task_correct_to_wrong_delta.png",
        plot_dir / "figure_C_primary_paired_transitions.png",
        plot_dir / "figure_D_offline_action_metrics.png",
        plot_dir / "figure_E_kv_scale_sanity.png",
    ]
    plot_success(cells, outputs[0])
    plot_task_deltas(tasks, outputs[1])
    plot_primary_transitions(summary, outputs[2])
    plot_offline_metrics(offline, outputs[3])
    plot_kv_sanity(kv, outputs[4])
    return outputs


def main() -> None:
    args = _parse_args()
    outputs = generate_plots(args.aggregate_dir)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
