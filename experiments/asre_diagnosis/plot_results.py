"""Generate the four requested ASRE diagnosis figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = _parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plot_results.py requires matplotlib in the analysis environment; "
            "no training/evaluation dependency is added by this experiment."
        ) from exc

    aggregate_dir = args.aggregate_dir.resolve()
    output_dir = (args.output_dir or aggregate_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(aggregate_dir / "summary.csv")
    task_delta = _read_csv(aggregate_dir / "task_success_delta.csv")
    labels = [record["condition"] for record in summary]
    x = np.arange(len(labels))

    def values(key: str) -> np.ndarray:
        return np.asarray(
            [float(record[key]) if record[key] != "" else np.nan for record in summary],
            dtype=float,
        )

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(x, values("offline_normalized_l2"), marker="o")
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("Normalized continuous-action L2")
    axis.set_xlabel("Video K/V layer group removed")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "A_layer_group_vs_action_deviation.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4.8))
    success = values("online_success_rate")
    delta = values("delta_success_rate")
    axis.plot(x, success, marker="o", label="Success rate")
    axis.plot(x, delta, marker="s", label="Delta vs baseline")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("Rate")
    axis.set_xlabel("Video K/V layer group removed")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "B_layer_group_vs_success_rate.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 5.2))
    deviation = values("offline_normalized_l2")
    degradation = -delta
    axis.scatter(deviation, degradation)
    for label, x_value, y_value in zip(labels, deviation, degradation):
        if np.isfinite(x_value) and np.isfinite(y_value):
            axis.annotate(label, (x_value, y_value), fontsize=8, xytext=(4, 3), textcoords="offset points")
    axis.set_xlabel("Offline normalized continuous-action L2")
    axis.set_ylabel("Online success-rate degradation")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "C_offline_deviation_vs_success_degradation.png", dpi=180)
    plt.close(fig)

    task_ids = sorted({int(record["task_id"]) for record in task_delta})
    lookup = {
        (record["condition"], int(record["task_id"])): float(record["delta_success_rate"])
        for record in task_delta
    }
    heatmap = np.asarray(
        [[lookup.get((condition, task_id), np.nan) for condition in labels] for task_id in task_ids],
        dtype=float,
    )
    fig, axis = plt.subplots(figsize=(max(9, len(labels) * 1.15), max(5, len(task_ids) * 0.55)))
    image = axis.imshow(heatmap, cmap="RdBu", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(task_ids)), [str(task_id) for task_id in task_ids])
    axis.set_xlabel("Video K/V layer group removed")
    axis.set_ylabel("LIBERO task ID")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Success-rate delta vs baseline")
    fig.tight_layout()
    fig.savefig(output_dir / "D_task_by_layer_group_success_delta_heatmap.png", dpi=180)
    plt.close(fig)
    print(f"Wrote four plots to {output_dir}")


if __name__ == "__main__":
    main()
