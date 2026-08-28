"""Generate the four pre-registered, scientifically useful G0 figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import atomic_write_json, now_iso, sha256_file
from experiments.asre_diagnosis.g0.definitions import (
    CONDITION_ORDER,
    PRIMARY_CONTRASTS,
    REFERENCE_SUITE,
    SUITE_ORDER,
)


DISPLAY_SUITES = {
    "libero_spatial": "Spatial\n(frozen)",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}
DISPLAY_CONDITIONS = {
    "full_current": "Full current",
    "late_current_15_29": "Late current 15–29",
    "early_current_00_19": "Early current 0–19",
    "late_wrong_scene_15_29": "Wrong-scene late",
}
DISPLAY_CONTRASTS = {
    "delta_late": "Late − Full",
    "delta_early_vs_late": "Early − Late",
    "delta_wrong_vs_late": "Wrong − Late",
}
COLORS = ("#3b5b92", "#3f8f83", "#d18b47", "#b65357")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError(f"G0 summary is incomplete: {path}.")
    return payload


def plot_all(*, summary_path: Path, output_dir: Path) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    summary = _load(summary_path)
    analyses = summary["suite_analyses"]
    suites = (REFERENCE_SUITE, *SUITE_ORDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9})

    # Figure A: grouped success rates.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(suites), dtype=float)
    width = 0.19
    for index, condition in enumerate(CONDITION_ORDER):
        rates = [
            next(
                row["success_rate"]
                for row in analyses[suite]["condition_rows"]
                if row["condition"] == condition
            )
            for suite in suites
        ]
        ax.bar(
            x + (index - 1.5) * width,
            rates,
            width,
            color=COLORS[index],
            label=DISPLAY_CONDITIONS[condition],
        )
    ax.set_xticks(x, [DISPLAY_SUITES[suite] for suite in suites])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("G0 cross-suite retrieval dependence")
    ax.legend(ncol=2, frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_a = output_dir / "figure_A_grouped_success_rates.png"
    fig.savefig(figure_a)
    plt.close(fig)

    # Figure B: primary deltas with task-hierarchical intervals.
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), sharey=True)
    for axis, contrast in zip(axes, PRIMARY_CONTRASTS):
        rows = [
            next(
                row
                for row in analyses[suite]["contrast_rows"]
                if row["contrast"] == contrast
            )
            for suite in suites
        ]
        estimates = np.asarray([row["delta"] for row in rows])
        low = np.asarray([row["task_hierarchical_ci_low"] for row in rows])
        high = np.asarray([row["task_hierarchical_ci_high"] for row in rows])
        axis.errorbar(
            np.arange(len(suites)),
            estimates,
            yerr=np.vstack([estimates - low, high - estimates]),
            fmt="o",
            color="#283747",
            capsize=3,
        )
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_xticks(
            np.arange(len(suites)),
            [DISPLAY_SUITES[suite] for suite in suites],
            rotation=25,
            ha="right",
        )
        axis.set_title(DISPLAY_CONTRASTS[contrast])
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Success-rate difference")
    fig.suptitle("Primary G0 contrasts (task-hierarchical 95% CI)")
    fig.tight_layout()
    figure_b = output_dir / "figure_B_primary_deltas.png"
    fig.savefig(figure_b)
    plt.close(fig)

    # Figure C: task-level rates for only the three new suites.
    rows = [
        row
        for suite in SUITE_ORDER
        for row in analyses[suite]["per_task_rows"]
    ]
    matrix = np.asarray(
        [[row[condition] for condition in CONDITION_ORDER] for row in rows],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    labels = [f"{row['suite'].replace('libero_', '')}:T{row['task_id']}" for row in rows]
    ax.set_yticks(np.arange(len(rows)), labels)
    ax.set_xticks(
        np.arange(len(CONDITION_ORDER)),
        [DISPLAY_CONDITIONS[name] for name in CONDITION_ORDER],
        rotation=20,
        ha="right",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if value < 0.55 else "black",
                fontsize=7,
            )
    ax.set_title("New-suite task-level success rates")
    fig.colorbar(image, ax=ax, label="Success rate", shrink=0.8)
    fig.tight_layout()
    figure_c = output_dir / "figure_C_new_suite_task_heatmap.png"
    fig.savefig(figure_c)
    plt.close(fig)

    # Figure D: simple intervention schematic, not a measured sparsity figure.
    fig, axes = plt.subplots(4, 1, figsize=(10, 3.9), sharex=True)
    for axis, condition in zip(axes, CONDITION_ORDER):
        for layer in range(30):
            if condition == "late_wrong_scene_15_29" and layer >= 15:
                color = "#b65357"
            elif (
                condition == "full_current"
                or condition == "late_current_15_29" and layer >= 15
                or condition == "early_current_00_19" and layer <= 19
            ):
                color = "#3f8f83"
            else:
                color = "#d9d9d9"
            axis.add_patch(plt.Rectangle((layer, 0), 1, 1, color=color, ec="white", lw=0.25))
        axis.set_xlim(0, 30)
        axis.set_ylim(0, 1)
        axis.set_yticks([0.5], [DISPLAY_CONDITIONS[condition]])
        axis.tick_params(axis="y", length=0)
    axes[-1].set_xticks(np.arange(0, 31, 5))
    axes[-1].set_xlabel("Action-transformer layer")
    fig.suptitle("G0 retrieval schedules")
    fig.tight_layout()
    figure_d = output_dir / "figure_D_retrieval_schedule_schematic.png"
    fig.savefig(figure_d)
    plt.close(fig)

    paths = [figure_a, figure_b, figure_c, figure_d]
    report = {
        "artifact_type": "asre_g0_figure_manifest",
        "schema_version": 1,
        "created_at": now_iso(),
        "source_summary_path": str(summary_path),
        "source_summary_sha256": sha256_file(summary_path),
        "figures": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
    }
    atomic_write_json(output_dir / "figure_manifest.json", report)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = plot_all(summary_path=args.summary, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
