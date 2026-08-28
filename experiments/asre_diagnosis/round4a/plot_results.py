"""Generate only the five pre-registered Round-4A scientific figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4A_PROTOCOL,
    atomic_write_json,
    now_iso,
    sha256_file,
)


ORDER = (
    "current_all",
    "wrong_all",
    "head50_seed1",
    "head50_seed2",
    "head50_seed3",
    "token50_seed1",
    "token50_seed2",
    "token50_seed3",
)
LABELS = ("Current", "Wrong", "H50-1", "H50-2", "H50-3", "T50-1", "T50-2", "T50-3")
COLORS = ("#2f6f4e", "#9d3c3c", "#5b6fb0", "#5b6fb0", "#5b6fb0", "#d38b2d", "#d38b2d", "#d38b2d")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_a(rows: list[dict[str, Any]], path: Path) -> None:
    by_name = {row["condition"]: row for row in rows}
    rates = np.asarray([float(by_name[name]["success_rate"]) for name in ORDER])
    low = np.asarray([float(by_name[name]["paired_bootstrap_ci_low"]) for name in ORDER])
    high = np.asarray([float(by_name[name]["paired_bootstrap_ci_high"]) for name in ORDER])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(ORDER))
    ax.bar(x, rates, color=COLORS, edgecolor="black", linewidth=0.6)
    ax.errorbar(x, rates, yerr=np.vstack((rates - low, high - rates)), fmt="none", color="black", capsize=3)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Online success rate")
    ax.set_title("Figure A — Eight-condition online success")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def figure_b(summary: dict[str, Any], path: Path) -> None:
    token = summary["axis_analysis"]["token"]
    head = summary["axis_analysis"]["head"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for x, analysis, color, label in (
        (0, head, "#5b6fb0", "Head-50"),
        (1, token, "#d38b2d", "Token-50"),
    ):
        values = np.asarray(analysis["mask_success_rates"])
        ax.scatter(np.full(3, x) + np.asarray((-0.06, 0, 0.06)), values, s=65, color=color, edgecolor="black", label=label)
        ax.plot([x - 0.16, x + 0.16], [np.median(values)] * 2, color="black", linewidth=2)
        ax.vlines(x, values.min(), values.max(), color=color, linewidth=3, alpha=0.55)
        ax.text(x, values.max() + 0.045, analysis["classification"], ha="center", weight="bold")
    ax.axhline(head["current_success_rate"], color="#2f6f4e", linestyle="--", label="Current")
    ax.axhline(head["wrong_success_rate"], color="#9d3c3c", linestyle=":", label="Wrong")
    ax.set_xticks((0, 1), ("Head-50", "Token-50"))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Success rate across frozen masks")
    ax.set_title("Figure B — Token-vs-Head robustness")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def figure_c(rows: list[dict[str, str]], path: Path) -> None:
    matrix = np.full((10, len(ORDER)), np.nan)
    for row in rows:
        matrix[int(row["task_id"]), ORDER.index(row["condition"])] = float(row["success_rate"])
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Task-condition heatmap is incomplete.")
    fig, ax = plt.subplots(figsize=(11, 5.6))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for task in range(10):
        for condition in range(len(ORDER)):
            ax.text(condition, task, f"{matrix[task, condition]:.1f}", ha="center", va="center", color="white" if matrix[task, condition] < 0.6 else "black", fontsize=8)
    ax.set_xticks(range(len(ORDER)), LABELS)
    ax.set_yticks(range(10), [f"Task {task}" for task in range(10)])
    ax.set_title("Figure C — Task × condition success")
    fig.colorbar(image, ax=ax, label="Success rate")
    _save(fig, path)


def figure_d(rows: list[dict[str, str]], path: Path) -> None:
    by_name = {row["condition"]: row for row in rows}
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for name, label, color in zip(ORDER, LABELS, COLORS):
        row = by_name[name]
        x = float(row["offline_executed_prefix_norm_rms_vs_current"])
        y = float(row["online_success_rate"])
        ax.scatter(x, y, s=75, color=color, edgecolor="black")
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Offline executed-prefix normalized RMS vs current")
    ax.set_ylabel("Online success rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Figure D — Offline action deviation vs online success")
    ax.text(0.02, 0.02, "Descriptive only; not a behavioral surrogate", transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, path)


def figure_e(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Figure E — Matched-content token/head cache masking")
    ax.add_patch(Rectangle((0.4, 2.55), 2.4, 0.8, facecolor="#b7d8c5", edgecolor="black"))
    ax.text(1.6, 2.95, "Current-scene K/V\nidentical geometry", ha="center", va="center")
    ax.add_patch(Rectangle((0.4, 0.65), 2.4, 0.8, facecolor="#e3b3b3", edgecolor="black"))
    ax.text(1.6, 1.05, "Wrong-scene K/V\nsame task/replan mapping", ha="center", va="center")
    ax.annotate("", xy=(4.0, 2.5), xytext=(2.8, 2.95), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(4.0, 1.5), xytext=(2.8, 1.05), arrowprops={"arrowstyle": "->"})
    for row, label in ((2.35, "M=1: retain current"), (1.35, "M=0: use wrong")):
        for index in range(8):
            color = "#6aaa80" if (index % 2 == 0) else "#c76565"
            ax.add_patch(Rectangle((4.0 + 0.35 * index, row), 0.3, 0.55, facecolor=color, edgecolor="white"))
        ax.text(5.4, row - 0.18, label, ha="center", va="top", fontsize=9)
    ax.text(5.4, 3.35, "Token positions or per-layer heads\n50% deterministic frozen mask", ha="center")
    ax.annotate("", xy=(7.5, 2.0), xytext=(6.9, 2.0), arrowprops={"arrowstyle": "->"})
    ax.add_patch(Rectangle((7.5, 1.25), 2.0, 1.5, facecolor="#d7d7ea", edgecolor="black"))
    ax.text(8.5, 2.0, "Mixed K/V\nZ = M·current\n+ (1−M)·wrong", ha="center", va="center")
    ax.text(5.4, 0.3, "Shapes, key count, attention mask, and softmax geometry remain constant", ha="center")
    _save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate = args.aggregate_dir.resolve()
    summary = _read_json(aggregate / "round4a_summary.json")
    if summary.get("protocol") != ROUND4A_PROTOCOL or summary.get("status") != "complete":
        raise ValueError("Round-4A aggregate is not complete.")
    plots = aggregate / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    outputs = {
        "figure_A_eight_condition_online_success.png": lambda path: figure_a(summary["online_condition_summary"], path),
        "figure_B_token_vs_head_robustness.png": lambda path: figure_b(summary, path),
        "figure_C_task_condition_heatmap.png": lambda path: figure_c(_read_csv(aggregate / "task_success.csv"), path),
        "figure_D_offline_vs_online.png": lambda path: figure_d(_read_csv(aggregate / "offline_online_condition_summary.csv"), path),
        "figure_E_matched_content_masking_schematic.png": figure_e,
    }
    for name, build in outputs.items():
        build(plots / name)
    manifest = {
        "artifact_type": "asre_round4a_figure_manifest",
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "created_at": now_iso(),
        "speed_or_flops_plots_generated": False,
        "figures": {
            name: {"path": str(plots / name), "sha256": sha256_file(plots / name)}
            for name in outputs
        },
    }
    atomic_write_json(plots / "figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
