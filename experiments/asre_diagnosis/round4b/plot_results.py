"""Generate the five registered Round-4B figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.asre_diagnosis.common import atomic_write_json, now_iso


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot(aggregate: Path) -> list[Path]:
    aggregate = aggregate.resolve()
    output = aggregate / "plots"
    output.mkdir(parents=True, exist_ok=True)
    online = {row["condition"]: row for row in _csv(aggregate / "online_condition_summary.csv")}
    contrasts = _csv(aggregate / "online_contrasts.csv")
    tasks = _csv(aggregate / "task_success.csv")
    diagnostics = _csv(aggregate / "subspace_diagnostics.csv")
    offline = _csv(aggregate / "offline_metric_summary.csv")
    paths = [
        output / "figure_A_online_rank_curves.png",
        output / "figure_B_svd_minus_random_contrasts.png",
        output / "figure_C_fit_holdout_energy.png",
        output / "figure_D_task_condition_heatmap.png",
        output / "figure_E_offline_action_deviation.png",
    ]

    ranks = np.asarray([0, 256, 768, 1536, 3072])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, color in (("svd", "#2369bd"), ("random", "#d97924")):
        conditions = ["wrong_all"] + [f"{kind}_r{rank}" for rank in ranks[1:-1]] + ["current_all"]
        values = np.asarray([float(online[name]["success_rate"]) for name in conditions])
        low = np.asarray([float(online[name]["paired_ci_low"]) for name in conditions])
        high = np.asarray([float(online[name]["paired_ci_high"]) for name in conditions])
        ax.plot(ranks, values, marker="o", label=kind.upper(), color=color)
        ax.fill_between(ranks, low, high, alpha=0.15, color=color)
    ax.set(xlabel="Feature-subspace rank", ylabel="Online success", ylim=(-0.03, 1.03))
    ax.legend(frameon=False)
    ax.set_title("Round 4B behavior curves with shared endpoints")
    _save(fig, paths[0])

    primary = [row for row in contrasts if row["comparison"].startswith("svd_minus_random")]
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    x = np.arange(3)
    values = np.asarray([float(row["delta_success_rate"]) for row in primary])
    low = np.asarray([float(row["paired_ci_low"]) for row in primary])
    high = np.asarray([float(row["paired_ci_high"]) for row in primary])
    ax.errorbar(x, values, yerr=[values - low, high - values], fmt="o", capsize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ["256", "768", "1536"])
    ax.set(xlabel="Matched rank", ylabel="SVD − random success", title="Primary paired contrasts")
    _save(fig, paths[1])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split_name, key, style in (
        ("Fit", "fit_captured_fraction", "-"),
        ("Holdout", "holdout_captured_fraction", "--"),
    ):
        means = []
        for rank in (256, 768, 1536):
            means.append(
                np.mean([float(row[key]) for row in diagnostics if int(row["rank"]) == rank])
            )
        ax.plot((256, 768, 1536), means, marker="o", linestyle=style, label=split_name)
    ax.set(xlabel="Rank", ylabel="Captured delta energy", ylim=(0, 1.02), title="Uncentered SVD energy generalization")
    ax.legend(frameon=False)
    _save(fig, paths[2])

    matrix = np.zeros((len(CONDITIONS := [
        "current_all", "wrong_all", "svd_r256", "random_r256", "svd_r768",
        "random_r768", "svd_r1536", "random_r1536"
    ]), 10))
    for row in tasks:
        matrix[CONDITIONS.index(row["condition"]), int(row["task_id"])] = float(row["success_rate"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(CONDITIONS)), CONDITIONS)
    ax.set_xticks(range(10), range(10))
    ax.set(xlabel="Task ID", title="Task-level online success")
    fig.colorbar(image, ax=ax, label="Success")
    _save(fig, paths[3])

    selected = [
        row
        for row in offline
        if row["reference"] == "current_all"
        and row["metric"] == "executed_prefix_norm_rms"
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(selected))
    values = np.asarray([float(row["mean"]) for row in selected])
    low = np.asarray([float(row["episode_cluster_ci_low"]) for row in selected])
    high = np.asarray([float(row["episode_cluster_ci_high"]) for row in selected])
    ax.errorbar(x, values, yerr=[values - low, high - values], fmt="o", capsize=3)
    ax.set_xticks(x, [row["condition"] for row in selected], rotation=30, ha="right")
    ax.set(ylabel="Executed-prefix normalized RMS vs current", title="Held-out fixed-state action deviation")
    _save(fig, paths[4])
    atomic_write_json(
        output / "figure_manifest.json",
        {
            "artifact_type": "asre_round4b_figure_manifest",
            "schema_version": 1,
            "created_at": now_iso(),
            "figures": [str(path) for path in paths],
            "efficiency_claims_or_flops": False,
        },
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    paths = plot(parser.parse_args().aggregate_dir)
    print(f"Generated {len(paths)} Round-4B figures.")


if __name__ == "__main__":
    main()
