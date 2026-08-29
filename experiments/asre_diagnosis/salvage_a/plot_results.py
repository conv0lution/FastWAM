"""Generate the registered result figures for ASRE Salvage A."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.asre_diagnosis.common import atomic_write_json, now_iso
from experiments.asre_diagnosis.salvage_a.definitions import CONDITIONS, DISPLAY


FAMILY_COLORS = {
    "svd": "#377eb8",
    "actionaware": "#e41a1c",
    "random": "#999999",
}


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
    online = _csv(aggregate / "online_condition_summary.csv")
    contrasts = _csv(aggregate / "online_contrasts.csv")
    tasks = _csv(aggregate / "task_success.csv")
    diagnostics = _csv(aggregate / "basis_diagnostics_summary.csv")
    heldout = {row["condition"]: row for row in online if row["scope"] == "heldout"}
    if set(heldout) != set(CONDITIONS):
        raise ValueError("Held-out plot input must contain exactly eight conditions.")
    paths = [
        output / "figure_A_heldout_primary_conditions.png",
        output / "figure_B_heldout_matched_rank_families.png",
        output / "figure_C_scope_comparison.png",
        output / "figure_D_heldout_task_heatmap.png",
        output / "figure_E_energy_vs_action_sensitivity.png",
    ]

    rates = np.asarray([float(heldout[name]["success_rate"]) for name in CONDITIONS])
    low = np.asarray([float(heldout[name]["paired_ci_low"]) for name in CONDITIONS])
    high = np.asarray([float(heldout[name]["paired_ci_high"]) for name in CONDITIONS])
    colors = [
        "#2b2b2b" if name == "current_all" else "#d9d9d9" if name == "wrong_all" else
        FAMILY_COLORS[name.split("_r", 1)[0]]
        for name in CONDITIONS
    ]
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    x = np.arange(len(CONDITIONS))
    ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(
        x,
        rates,
        yerr=np.vstack((rates - low, high - rates)),
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.set_xticks(x, [DISPLAY[name] for name in CONDITIONS], rotation=25, ha="right")
    ax.set(
        ylabel="Held-out closed-loop success",
        ylim=(0.0, 1.05),
        title="Salvage A primary held-out outcomes (n=50 per condition)",
    )
    _save(fig, paths[0])

    matched = [
        name
        for rank in (36, 97)
        for name in (f"svd_r{rank}", f"actionaware_r{rank}", f"random_r{rank}")
    ]
    matched_rates = np.asarray([float(heldout[name]["success_rate"]) for name in matched])
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    positions = np.asarray([0, 1, 2, 4, 5, 6], dtype=float)
    ax.bar(
        positions,
        matched_rates,
        color=[FAMILY_COLORS[name.split("_r", 1)[0]] for name in matched],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(float(heldout["current_all"]["success_rate"]), color="black", linestyle="--")
    ax.set_xticks(positions, [DISPLAY[name] for name in matched], rotation=20, ha="right")
    ax.set(
        ylabel="Held-out success",
        ylim=(0.0, 1.05),
        title="Matched-rank variance, action-sensitive, and random subspaces",
    )
    _save(fig, paths[1])

    scope_lookup = {
        (row["scope"], row["condition"]): float(row["success_rate"]) for row in online
    }
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    width = 0.25
    x = np.arange(len(CONDITIONS), dtype=float)
    for offset, scope, label, color in (
        (-width, "heldout", "Held-out 50 (primary)", "#d62728"),
        (0.0, "calibration", "Calibration 50", "#1f77b4"),
        (width, "all", "All 100", "#7f7f7f"),
    ):
        values = [scope_lookup[(scope, condition)] for condition in CONDITIONS]
        ax.bar(x + offset, values, width=width, label=label, color=color)
    ax.set_xticks(x, [DISPLAY[name] for name in CONDITIONS], rotation=25, ha="right")
    ax.set(ylabel="Success", ylim=(0.0, 1.05), title="Primary and descriptive online scopes")
    ax.legend(frameon=False)
    _save(fig, paths[2])

    matrix = np.zeros((len(CONDITIONS), 10), dtype=np.float64)
    seen: set[tuple[str, int]] = set()
    for row in tasks:
        if row["scope"] != "heldout":
            continue
        key = (row["condition"], int(row["task_id"]))
        if key in seen:
            raise ValueError(f"Duplicate held-out task row: {key}.")
        seen.add(key)
        matrix[CONDITIONS.index(row["condition"]), key[1]] = float(row["success_rate"])
    if len(seen) != 80:
        raise ValueError("Held-out task heatmap requires eight conditions x ten tasks.")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(CONDITIONS)), [DISPLAY[name] for name in CONDITIONS])
    ax.set_xticks(range(10), range(10))
    ax.set(xlabel="Task ID", title="Held-out task-level success (five trials/task)")
    fig.colorbar(image, ax=ax, label="Success")
    _save(fig, paths[3])

    if len(diagnostics) != 6:
        raise ValueError("Offline geometry plot requires six family/rank summary rows.")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    markers = {36: "o", 97: "s"}
    for row in diagnostics:
        family = row["basis_family"]
        rank = int(row["rank"])
        energy = float(row["heldout_delta_z_energy_captured"])
        sensitivity = float(row["calibration_action_sensitivity_captured"])
        ax.scatter(
            energy,
            sensitivity,
            marker=markers[rank],
            s=70,
            color=FAMILY_COLORS[family],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(f"{family}-{rank}", (energy, sensitivity), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set(
        xlabel="Held-out ΔZ energy captured",
        ylabel="Calibration action sensitivity captured",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        title="Representation energy versus action-sensitive geometry",
    )
    _save(fig, paths[4])

    primary_contrasts = [
        row for row in contrasts if row["scope"] == "heldout" and row["primary_salvage_claim"].lower() == "true"
    ]
    if len(primary_contrasts) != 6:
        raise ValueError("Figure inputs do not contain six registered held-out contrasts.")
    atomic_write_json(
        output / "figure_manifest.json",
        {
            "artifact_type": "asre_salvage_a_figure_manifest",
            "schema_version": 1,
            "created_at": now_iso(),
            "figures": [str(path) for path in paths],
            "primary_scope": "heldout",
            "calibration_and_all100_descriptive_only": True,
            "semantic_or_efficiency_claims": False,
        },
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    paths = plot(parser.parse_args().aggregate_dir)
    print(f"Generated {len(paths)} Salvage A figures.")


if __name__ == "__main__":
    main()
