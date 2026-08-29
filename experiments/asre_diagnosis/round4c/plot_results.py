"""Generate the four registered Round-4C figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.asre_diagnosis.common import atomic_write_json, now_iso
from experiments.asre_diagnosis.round4c.definitions import CONDITIONS


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
    curve = _csv(aggregate / "energy_success_curve.csv")
    tasks = _csv(aggregate / "task_success.csv")
    saturation = _csv(aggregate / "action_saturation.csv")
    paths = [
        output / "figure_A_energy_vs_online_success.png",
        output / "figure_B_rank_fraction_vs_online_success.png",
        output / "figure_C_task_condition_heatmap.png",
        output / "figure_D_success_gap_vs_energy.png",
    ]
    energy = np.asarray([float(row["heldout_delta_z_energy"]) for row in curve])
    rank_fraction = np.asarray([float(row["rank_fraction"]) for row in curve])
    success = np.asarray([float(row["success_rate"]) for row in curve])
    low = np.asarray([float(row["paired_ci_low"]) for row in curve])
    high = np.asarray([float(row["paired_ci_high"]) for row in curve])
    new = np.asarray([row["newly_run_round4c"].lower() == "true" for row in curve])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(energy, success, color="#2369bd", linewidth=1.4, alpha=0.75)
    ax.errorbar(
        energy[new],
        success[new],
        yerr=np.vstack((success[new] - low[new], high[new] - success[new])),
        fmt="o",
        color="#2369bd",
        capsize=4,
        label="Round 4C (new)",
    )
    ax.errorbar(
        energy[~new],
        success[~new],
        yerr=np.vstack((success[~new] - low[~new], high[~new] - success[~new])),
        fmt="o",
        markerfacecolor="white",
        markeredgecolor="#d97924",
        color="#d97924",
        capsize=4,
        label="Round 4B frozen reference",
    )
    for row, x, y in zip(curve, energy, success):
        ax.annotate(row["condition"], (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(
        xlabel="Retained held-out Delta-Z energy",
        ylabel="Closed-loop success",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        title="Energy-controlled action sufficiency",
    )
    ax.legend(frameon=False)
    _save(fig, paths[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(rank_fraction, success, marker="o", color="#2369bd")
    ax.fill_between(rank_fraction, low, high, color="#2369bd", alpha=0.15)
    for row, x, y in zip(curve, rank_fraction, success):
        ax.annotate(row["condition"], (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(
        xlabel="Retained feature-rank fraction",
        ylabel="Closed-loop success",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        title="Rank fraction versus behavior",
    )
    _save(fig, paths[1])

    matrix = np.zeros((len(CONDITIONS), 10), dtype=np.float64)
    for row in tasks:
        matrix[CONDITIONS.index(row["condition"]), int(row["task_id"])] = float(
            row["success_rate"]
        )
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(CONDITIONS)), CONDITIONS)
    ax.set_xticks(range(10), range(10))
    ax.set(xlabel="Task ID", title="Task-level online success")
    fig.colorbar(image, ax=ax, label="Success")
    _save(fig, paths[2])

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    sat_energy = np.asarray([float(row["heldout_delta_z_energy"]) for row in saturation])
    gaps = np.asarray(
        [float(row["success_gap_current_minus_condition"]) for row in saturation]
    )
    ax.plot(sat_energy, gaps, marker="o", color="#7a3e9d")
    ax.axhline(0.0, color="black", linewidth=0.8)
    for row, x, y in zip(saturation, sat_energy, gaps):
        ax.annotate(row["condition"], (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(
        xlabel="Retained held-out Delta-Z energy",
        ylabel="Current success minus condition success",
        title="Behavioral gap to current endpoint",
    )
    _save(fig, paths[3])

    atomic_write_json(
        output / "figure_manifest.json",
        {
            "artifact_type": "asre_round4c_figure_manifest",
            "schema_version": 1,
            "created_at": now_iso(),
            "figures": [str(path) for path in paths],
            "round4b_reference_visually_distinguished": True,
            "offline_figure_generated": False,
            "efficiency_claims_or_flops": False,
        },
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    paths = plot(parser.parse_args().aggregate_dir)
    print(f"Generated {len(paths)} Round-4C figures.")


if __name__ == "__main__":
    main()
