"""Generate the four registered Salvage-B functional-dissociation figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_B_PROTOCOL,
    atomic_write_json,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.salvage_b.definitions import CONDITIONS  # noqa: E402
from experiments.asre_diagnosis.salvage_b.world_manifest import (  # noqa: E402
    NATIVE_WORLD_METRIC,
)


DISPLAY = {
    "current_all": "Current",
    "wrong_all": "Wrong",
    "svd_r97": "SVD-97",
    "svd_r170": "SVD-170",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _ordered(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_condition = {str(row["condition"]): row for row in rows}
    if set(by_condition) != set(CONDITIONS):
        raise ValueError("Plot input does not contain exactly four Salvage-B conditions.")
    return [by_condition[condition] for condition in CONDITIONS]


def generate_plots(
    summary: Mapping[str, Any], *, output_dir: Path
) -> dict[str, Any]:
    """Plot one validated normal Phase-B summary; special failures have no figures."""

    if (
        summary.get("protocol") != SALVAGE_B_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("classification", {}).get("classification")
        not in {"STRONG", "MODERATE", "WEAK"}
    ):
        raise ValueError("Registered figures require a complete valid Phase-B summary.")
    recovery = _ordered(list(summary["functional_recovery"]))
    world = _ordered(list(summary["world_condition_summary"]))
    action = _ordered(list(summary["action_condition_summary"]))
    output = output_dir.resolve() / "plots"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "figure_A": output / "figure_A_action_vs_world_recovery.png",
        "figure_B": output / "figure_B_native_world_loss.png",
        "figure_C": output / "figure_C_action_success.png",
        "figure_D": output / "figure_D_world_vs_action_recovery_scatter.png",
    }
    x = np.arange(len(CONDITIONS), dtype=np.float64)
    labels = [DISPLAY[condition] for condition in CONDITIONS]

    # Figure A -- the registered primary visualization.
    action_recovery = np.asarray(
        [float(row["action_recovery"]) for row in recovery], dtype=np.float64
    )
    world_recovery = np.asarray(
        [float(row["world_recovery"]) for row in recovery], dtype=np.float64
    )
    ar_low = np.asarray(
        [float(row["action_recovery_paired_ci_low"]) for row in recovery]
    )
    ar_high = np.asarray(
        [float(row["action_recovery_paired_ci_high"]) for row in recovery]
    )
    wr_low = np.asarray(
        [float(row["world_recovery_paired_ci_low"]) for row in recovery]
    )
    wr_high = np.asarray(
        [float(row["world_recovery_paired_ci_high"]) for row in recovery]
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.errorbar(
        x - 0.04,
        action_recovery,
        yerr=np.maximum(
            0.0, np.vstack((action_recovery - ar_low, ar_high - action_recovery))
        ),
        marker="o",
        capsize=4,
        color="#2267b4",
        label="ActionRecovery",
    )
    ax.errorbar(
        x + 0.04,
        world_recovery,
        yerr=np.maximum(
            0.0, np.vstack((world_recovery - wr_low, wr_high - world_recovery))
        ),
        marker="s",
        capsize=4,
        color="#d26a24",
        label="WorldRecovery",
    )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x, labels)
    ax.set(
        ylabel="Unclipped normalized recovery",
        title="Shared-intervention functional recovery",
    )
    ax.legend(frameon=False)
    _save(fig, paths["figure_A"])

    # Figure B -- native loss, never a proxy visual score.
    loss = np.asarray([float(row["mean_native_world_loss"]) for row in world])
    loss_low = np.asarray(
        [float(row["mean_loss_paired_ci_low"]) for row in world]
    )
    loss_high = np.asarray(
        [float(row["mean_loss_paired_ci_high"]) for row in world]
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colors = ["#3578b9", "#9a9a9a", "#e09542", "#c56531"]
    ax.bar(x, loss, color=colors, alpha=0.9)
    ax.errorbar(
        x,
        loss,
        yerr=np.maximum(0.0, np.vstack((loss - loss_low, loss_high - loss))),
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.set_xticks(x, labels)
    ax.set(
        ylabel="Terminal future-latent MSE (lower is better)",
        title="Pure-noise 10-step native world inference",
    )
    _save(fig, paths["figure_B"])

    # Figure C -- frozen Round-4C behavior, visibly not a new online run.
    success = np.asarray([float(row["success_rate"]) for row in action])
    success_low = np.asarray([float(row["paired_ci_low"]) for row in action])
    success_high = np.asarray([float(row["paired_ci_high"]) for row in action])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(x, success, color=colors, alpha=0.9)
    ax.errorbar(
        x,
        success,
        yerr=np.maximum(
            0.0, np.vstack((success - success_low, success_high - success))
        ),
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.set_xticks(x, labels)
    ax.set(
        ylabel="Closed-loop action success",
        ylim=(0.0, 1.06),
        title="Frozen Round-4C action outcomes (not rerun)",
    )
    _save(fig, paths["figure_C"])

    # Figure D -- x is WorldRecovery and y is ActionRecovery, as registered.
    fig, ax = plt.subplots(figsize=(6.2, 5.7))
    lower = min(-0.05, float(world_recovery.min()), float(action_recovery.min()))
    upper = max(1.05, float(world_recovery.max()), float(action_recovery.max()))
    margin = 0.05 * max(1.0, upper - lower)
    limits = (lower - margin, upper + margin)
    ax.plot(limits, limits, color="black", linewidth=1.0, linestyle="--", label="y = x")
    for index, condition in enumerate(CONDITIONS):
        ax.scatter(
            world_recovery[index],
            action_recovery[index],
            s=60,
            color=colors[index],
            zorder=3,
        )
        ax.annotate(
            DISPLAY[condition],
            (world_recovery[index], action_recovery[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set(
        xlabel="WorldRecovery",
        ylabel="ActionRecovery",
        xlim=limits,
        ylim=limits,
        title="World-vs-action functional recovery",
        aspect="equal",
    )
    ax.legend(frameon=False)
    _save(fig, paths["figure_D"])

    manifest = {
        "artifact_type": "asre_salvage_b_figure_manifest",
        "schema_version": 2,
        "protocol": SALVAGE_B_PROTOCOL,
        "created_at": now_iso(),
        "figures": [
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        ],
        "primary_figure": "figure_A",
        "native_world_metric_used": True,
        "native_world_metric": NATIVE_WORLD_METRIC,
        "action_results_rerun": False,
        "recoveries_clipped": False,
        "later_experiment_launched": False,
    }
    atomic_write_json(output / "figure_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    manifest = generate_plots(summary, output_dir=args.output_dir)
    print(f"Generated {len(manifest['figures'])} Salvage-B figures.")


if __name__ == "__main__":
    main()
