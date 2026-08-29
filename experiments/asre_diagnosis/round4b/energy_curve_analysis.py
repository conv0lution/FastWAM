"""Aggregate and plot complete frozen Round-4B cumulative ΔZ energy curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4B_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round4b.aggregate_results import (  # noqa: E402
    _read,
    _write_csv,
    _write_text,
)
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
    RANKS,
)
from experiments.asre_diagnosis.round4b.energy_curve_definitions import RECOVER_STAGE  # noqa: E402


GLOBAL_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99)
MATRIX_THRESHOLDS = (0.50, 0.70, 0.80, 0.85, 0.90, 0.95)
TRANSFER_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90)
ANCHOR_TOLERANCE = 5e-4
RANK_D_TOLERANCE = 5e-4


def minimum_rank(curve: np.ndarray, threshold: float) -> int:
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size != EXPECTED_FEATURE_DIM + 1:
        raise ValueError("Energy curve must contain ranks 0..3072.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"Invalid energy threshold: {threshold}")
    index = int(np.searchsorted(values, threshold, side="left"))
    if index >= values.size:
        raise ValueError(f"Curve never reaches threshold {threshold}.")
    return index


def cumulative_curve(coordinate_energy: np.ndarray, total_energy: float) -> np.ndarray:
    energy = np.asarray(coordinate_energy, dtype=np.float64)
    if energy.shape != (EXPECTED_FEATURE_DIM,) or not np.all(np.isfinite(energy)):
        raise ValueError("Coordinate-energy vector must be finite with length 3072.")
    if np.min(energy) < -1e-8 or not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("Coordinate and total energies must be nonnegative/positive.")
    return np.concatenate(([0.0], np.cumsum(np.maximum(energy, 0.0)) / total_energy))


def validate_curve(curve: np.ndarray, *, label: str, rank_d_tolerance: float) -> None:
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (EXPECTED_FEATURE_DIM + 1,):
        raise ValueError(f"{label} does not cover every rank 0..3072.")
    if values[0] != 0.0 or np.min(np.diff(values)) < -1e-12:
        raise ValueError(f"{label} is not monotonic from exact rank-zero energy.")
    if abs(float(values[-1]) - 1.0) > rank_d_tolerance:
        raise ValueError(f"{label} rank-D energy is {values[-1]}, not approximately one.")


def _load_coordinates(
    coordinate_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    matrices: dict[str, dict[str, Any]] = {}
    reports = []
    for index in range(4):
        report_path = coordinate_dir / f"{RECOVER_STAGE}_shard{index}.json"
        report = _read(report_path)
        if (
            report.get("status") != "complete"
            or report.get("stage") != RECOVER_STAGE
            or report.get("layers") != list(LATE_LAYERS[index::4])
            or report.get("heldout_svd_refit") is not False
        ):
            raise ValueError(f"Incomplete coordinate-energy shard: {report_path}")
        reports.append(
            {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
                "git_commit_hash": report["git_commit_hash"],
            }
        )
        for matrix, record in report["artifacts"].items():
            if matrix in matrices:
                raise ValueError(f"Duplicate coordinate-energy matrix: {matrix}")
            path = Path(str(record["path"])).resolve()
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"Coordinate-energy artifact drifted: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("matrix") != matrix or payload.get("no_heldout_refit") is not True:
                raise ValueError(f"Malformed coordinate-energy artifact: {path}")
            matrices[matrix] = payload
    expected = {
        f"layer{layer:02d}_{kind}" for layer in LATE_LAYERS for kind in ("k", "v")
    }
    if set(matrices) != expected:
        raise ValueError("Coordinate energies do not cover exactly 15 layers × K/V.")
    return matrices, reports


def _curves(
    matrices: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    matrix_curves: dict[str, dict[str, np.ndarray]] = {}
    global_energy = {
        split: {
            scope: np.zeros(EXPECTED_FEATURE_DIM, dtype=np.float64)
            for scope in ("all", "K", "V")
        }
        for split in ("calibration", "heldout")
    }
    global_totals = {
        split: {scope: 0.0 for scope in ("all", "K", "V")}
        for split in ("calibration", "heldout")
    }
    for matrix, payload in matrices.items():
        kind = str(payload["tensor_kind"]).upper()
        calibration = np.asarray(payload["calibration_coordinate_energy"], dtype=np.float64)
        heldout = np.asarray(payload["heldout_coordinate_energy"], dtype=np.float64)
        calibration_total = float(payload["calibration_total_energy"])
        heldout_total = float(payload["heldout_total_energy"])
        matrix_curves[matrix] = {
            "calibration": cumulative_curve(calibration, calibration_total),
            "heldout": cumulative_curve(heldout, heldout_total),
        }
        for split, energy, total in (
            ("calibration", calibration, calibration_total),
            ("heldout", heldout, heldout_total),
        ):
            for scope in ("all", kind):
                global_energy[split][scope] += energy
                global_totals[split][scope] += total
    global_curves = {
        split: {
            scope: cumulative_curve(global_energy[split][scope], global_totals[split][scope])
            for scope in ("all", "K", "V")
        }
        for split in ("calibration", "heldout")
    }
    for matrix, by_split in matrix_curves.items():
        for split, curve in by_split.items():
            validate_curve(
                curve,
                label=f"{matrix}/{split}",
                rank_d_tolerance=RANK_D_TOLERANCE,
            )
    for split, by_scope in global_curves.items():
        for scope, curve in by_scope.items():
            validate_curve(
                curve,
                label=f"global/{split}/{scope}",
                rank_d_tolerance=RANK_D_TOLERANCE,
            )
    return matrix_curves, global_curves


def _threshold_rows(global_curves: Mapping[str, Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for threshold in GLOBAL_THRESHOLDS:
        for scope in ("all", "K", "V"):
            calibration_rank = minimum_rank(global_curves["calibration"][scope], threshold)
            heldout_rank = minimum_rank(global_curves["heldout"][scope], threshold)
            rows.append(
                {
                    "threshold": threshold,
                    "scope": scope,
                    "calibration_rank": calibration_rank,
                    "heldout_rank": heldout_rank,
                    "calibration_rank_fraction": calibration_rank / EXPECTED_FEATURE_DIM,
                    "heldout_rank_fraction": heldout_rank / EXPECTED_FEATURE_DIM,
                    "calibration_energy_at_selected_rank": float(
                        global_curves["calibration"][scope][calibration_rank]
                    ),
                    "heldout_energy_at_selected_rank": float(
                        global_curves["heldout"][scope][heldout_rank]
                    ),
                }
            )
    return rows


def _per_matrix_rows(
    matrix_curves: Mapping[str, Mapping[str, np.ndarray]],
    diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = diagnostics["fit_by_matrix"]
    rows = []
    for matrix in sorted(matrix_curves):
        layer = int(matrix[5:7])
        kind = matrix[-1].upper()
        for split in ("calibration", "heldout"):
            curve = matrix_curves[matrix][split]
            row: dict[str, Any] = {
                "split": split,
                "matrix": matrix,
                "layer": layer,
                "tensor_kind": kind,
                "feature_dim": EXPECTED_FEATURE_DIM,
                "effective_rank": source[matrix]["effective_rank"],
                "spectral_gap_r256": source[matrix]["spectral_gap_ratio"]["256"],
                "spectral_gap_r768": source[matrix]["spectral_gap_ratio"]["768"],
                "spectral_gap_r1536": source[matrix]["spectral_gap_ratio"]["1536"],
            }
            for threshold in MATRIX_THRESHOLDS:
                label = int(round(threshold * 100))
                rank = minimum_rank(curve, threshold)
                row[f"r{label}"] = rank
                row[f"r{label}_fraction"] = rank / EXPECTED_FEATURE_DIM
            rows.append(row)
    return rows


def _layer_rows(per_matrix_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["split"]), int(row["layer"]), str(row["tensor_kind"])): row
        for row in per_matrix_rows
    }
    rows = []
    for split in ("calibration", "heldout"):
        for layer in LATE_LAYERS:
            k = lookup[(split, layer, "K")]
            v = lookup[(split, layer, "V")]
            row: dict[str, Any] = {
                "split": split,
                "layer": layer,
                "K_effective_rank": k["effective_rank"],
                "V_effective_rank": v["effective_rank"],
                "K_spectral_gap_r256": k["spectral_gap_r256"],
                "V_spectral_gap_r256": v["spectral_gap_r256"],
                "K_spectral_gap_r768": k["spectral_gap_r768"],
                "V_spectral_gap_r768": v["spectral_gap_r768"],
                "K_spectral_gap_r1536": k["spectral_gap_r1536"],
                "V_spectral_gap_r1536": v["spectral_gap_r1536"],
            }
            for label in (50, 70, 80):
                row[f"K_r{label}"] = k[f"r{label}"]
                row[f"V_r{label}"] = v[f"r{label}"]
                row[f"median_KV_r{label}"] = float(
                    np.median([k[f"r{label}"], v[f"r{label}"]])
                )
            rows.append(row)
    return rows


def _trend_summary(layer_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    x = np.asarray(list(LATE_LAYERS), dtype=np.float64)
    for split in ("calibration", "heldout"):
        selected = {int(row["layer"]): row for row in layer_rows if row["split"] == split}
        for threshold in (50, 70, 80):
            for kind in ("K", "V"):
                values = np.asarray(
                    [selected[layer][f"{kind}_r{threshold}"] for layer in LATE_LAYERS],
                    dtype=np.float64,
                )
                slope = float(np.polyfit(x, values, deg=1)[0])
                rows.append(
                    {
                        "split": split,
                        "threshold": threshold / 100,
                        "tensor_kind": kind,
                        "median_rank": float(np.median(values)),
                        "minimum_rank": int(values.min()),
                        "maximum_rank": int(values.max()),
                        "linear_rank_change_per_layer": slope,
                        "depth_direction": (
                            "larger"
                            if slope > 0
                            else "smaller"
                            if slope < 0
                            else "flat"
                        ),
                    }
                )
    return rows


def _transfer_rows(global_curves: Mapping[str, Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    calibration = global_curves["calibration"]["all"]
    heldout = global_curves["heldout"]["all"]
    for threshold in TRANSFER_THRESHOLDS:
        rank = minimum_rank(calibration, threshold)
        rows.append(
            {
                "calibration_threshold": threshold,
                "calibration_selected_rank": rank,
                "rank_fraction": rank / EXPECTED_FEATURE_DIM,
                "calibration_capture": float(calibration[rank]),
                "heldout_capture_at_calibration_rank": float(heldout[rank]),
                "calibration_minus_heldout": float(calibration[rank] - heldout[rank]),
            }
        )
    return rows


def candidate_ranks(global_curves: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    candidates = {}
    for label, threshold in (("candidate_r50", 0.50), ("candidate_r70", 0.70), ("candidate_r80", 0.80)):
        rank = minimum_rank(global_curves["heldout"]["all"], threshold)
        candidates[label] = {
            "rank": rank,
            "rank_fraction": rank / EXPECTED_FEATURE_DIM,
            "target_heldout_energy": threshold,
            "calibration_global_energy": float(global_curves["calibration"]["all"][rank]),
            "heldout_global_energy": float(global_curves["heldout"]["all"][rank]),
            "heldout_K_energy": float(global_curves["heldout"]["K"][rank]),
            "heldout_V_energy": float(global_curves["heldout"]["V"][rank]),
        }
    return candidates


def _plots(
    output: Path,
    global_curves: Mapping[str, Mapping[str, np.ndarray]],
    layer_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ranks = np.arange(EXPECTED_FEATURE_DIM + 1)
    fractions = ranks / EXPECTED_FEATURE_DIM
    paths = [
        figure_dir / "figure_A_global_cumulative_energy.png",
        figure_dir / "figure_B_heldout_K_vs_V.png",
        figure_dir / "figure_C_layer_threshold_ranks.png",
        figure_dir / "figure_D_calibration_to_heldout_transfer.png",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions, global_curves["calibration"]["all"], label="Calibration")
    ax.plot(fractions, global_curves["heldout"]["all"], label="Held-out", linestyle="--")
    for rank in RANKS:
        ax.axvline(rank / EXPECTED_FEATURE_DIM, color="grey", alpha=0.45, linewidth=0.8)
        ax.text(rank / EXPECTED_FEATURE_DIM, 0.04, str(rank), rotation=90, va="bottom", ha="right")
    ax.set(xlabel="Rank / 3072", ylabel="Weighted retained ΔZ energy", ylim=(0, 1.01))
    ax.set_title("Global cumulative ΔZ energy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths[0], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions, global_curves["heldout"]["K"], label="Held-out K")
    ax.plot(fractions, global_curves["heldout"]["V"], label="Held-out V")
    ax.set(xlabel="Rank / 3072", ylabel="Weighted retained ΔZ energy", ylim=(0, 1.01))
    ax.set_title("Held-out K versus V cumulative energy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths[1], dpi=180)
    plt.close(fig)

    heldout_layers = [row for row in layer_rows if row["split"] == "heldout"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, threshold in zip(axes, (50, 70, 80)):
        layers = [int(row["layer"]) for row in heldout_layers]
        ax.plot(layers, [row[f"K_r{threshold}"] for row in heldout_layers], marker="o", label="K")
        ax.plot(layers, [row[f"V_r{threshold}"] for row in heldout_layers], marker="o", label="V")
        ax.set(xlabel="Layer", title=f"{threshold}% energy")
    axes[0].set_ylabel("Minimum held-out rank")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths[2], dpi=180)
    plt.close(fig)

    thresholds = np.asarray([row["calibration_threshold"] for row in transfer_rows])
    calibration = np.asarray([row["calibration_capture"] for row in transfer_rows])
    heldout = np.asarray([row["heldout_capture_at_calibration_rank"] for row in transfer_rows])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(thresholds, calibration, marker="o", label="Calibration")
    ax.plot(thresholds, heldout, marker="o", label="Held-out at calibration rank")
    ax.plot((0.45, 0.95), (0.45, 0.95), color="grey", linestyle=":", linewidth=0.8)
    ax.set(xlabel="Calibration target", ylabel="Retained energy", ylim=(0.45, 0.95))
    ax.set_title("Calibration-selected rank transfer")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths[3], dpi=180)
    plt.close(fig)
    return paths


def _markdown(
    *,
    threshold_rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Any],
    transfer_rows: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
    anchor_checks: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> str:
    lookup = {(row["threshold"], row["scope"]): row for row in threshold_rows}
    lines = [
        "# Round 4B Follow-up — Complete Cumulative ΔZ Energy",
        "",
        "This analysis used the frozen calibration split and donor mapping. It reran no online "
        "episode and fitted no SVD on held-out data.",
        "",
        "## Global threshold ranks",
        "",
        "| Threshold | Calibration all | Held-out all | Calibration K/V | Held-out K/V |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threshold in GLOBAL_THRESHOLDS:
        all_row = lookup[(threshold, "all")]
        k_row = lookup[(threshold, "K")]
        v_row = lookup[(threshold, "V")]
        lines.append(
            f"| {threshold:.0%} | {all_row['calibration_rank']} | {all_row['heldout_rank']} | "
            f"{k_row['calibration_rank']} / {v_row['calibration_rank']} | "
            f"{k_row['heldout_rank']} / {v_row['heldout_rank']} |"
        )
    lines.extend(
        [
            "",
            "## Candidate ranks for a future online experiment",
            "",
            "These ranks are selected only from held-out ΔZ energy; no online result exists at them.",
            "",
            "| Candidate | Rank | Rank fraction | Calibration all | Held-out all | Held-out K | Held-out V |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("candidate_r50", "candidate_r70", "candidate_r80"):
        item = candidates[name]
        lines.append(
            f"| {name} | {item['rank']} | {item['rank_fraction']:.2%} | "
            f"{item['calibration_global_energy']:.2%} | {item['heldout_global_energy']:.2%} | "
            f"{item['heldout_K_energy']:.2%} | {item['heldout_V_energy']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Calibration-to-held-out transfer",
            "",
            "| Calibration target | Selected rank | Calibration capture | Held-out capture | Gap |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in transfer_rows:
        lines.append(
            f"| {row['calibration_threshold']:.0%} | {row['calibration_selected_rank']} | "
            f"{row['calibration_capture']:.2%} | "
            f"{row['heldout_capture_at_calibration_rank']:.2%} | "
            f"{row['calibration_minus_heldout']:+.2%} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive layer trends",
            "",
            "Ranks describe representation geometry only; no semantic stage is inferred.",
            "",
            "| Split | Threshold | K/V | Median | Min–max | Linear change per layer |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in trend_rows:
        lines.append(
            f"| {row['split']} | {row['threshold']:.0%} | {row['tensor_kind']} | "
            f"{row['median_rank']:.0f} | {row['minimum_rank']}–{row['maximum_rank']} | "
            f"{row['linear_rank_change_per_layer']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Registered-anchor consistency",
            "",
            "| Rank | Calibration reconstructed | Calibration registered | Held-out reconstructed | Held-out registered |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in anchor_checks:
        lines.append(
            f"| {row['rank']} | {row['calibration_reconstructed']:.6f} | "
            f"{row['calibration_registered']:.6f} | {row['heldout_reconstructed']:.6f} | "
            f"{row['heldout_registered']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Provenance and stop rule",
            "",
            f"- Feature dimension: `{EXPECTED_FEATURE_DIM}`.",
            f"- Frozen basis manifest: `{provenance['basis_manifest_path']}`.",
            f"- Frozen split: `{provenance['split_path']}`.",
            "- Held-out Grams came from one authorized cache-only pass over 100 frozen states.",
            "- The complete eigensystem was recovered only from the original calibration Grams.",
            "- No held-out SVD fitting, online episode, environment rollout, or later experiment was launched.",
            "",
            "The energy curves alone do not establish action-specific sufficiency. Future online "
            "success at the lower-energy candidates is required.",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> Path:
    coordinate_dir = args.coordinate_dir.resolve()
    matrices, coordinate_reports = _load_coordinates(coordinate_dir)
    diagnostics_path = args.diagnostics.resolve()
    diagnostics = _read(diagnostics_path)
    basis_path = args.basis_manifest.resolve()
    split_path = args.split.resolve()
    if (
        diagnostics.get("status") != "complete"
        or diagnostics.get("basis_manifest_sha256") != sha256_file(basis_path)
        or diagnostics.get("split_sha256") != sha256_file(split_path)
    ):
        raise ValueError("Frozen diagnostics/basis/split provenance is inconsistent.")
    matrix_curves, global_curves = _curves(matrices)
    threshold_rows = _threshold_rows(global_curves)
    per_matrix_rows = _per_matrix_rows(matrix_curves, diagnostics)
    layer_rows = _layer_rows(per_matrix_rows)
    trend_rows = _trend_summary(layer_rows)
    transfer_rows = _transfer_rows(global_curves)
    candidates = candidate_ranks(global_curves)

    anchor_checks = []
    for rank in RANKS:
        calibration_registered = sum(
            float(record["fit_captured_fraction"][str(rank)])
            * float(record["fit_total_energy"])
            for record in diagnostics["fit_by_matrix"].values()
        ) / sum(
            float(record["fit_total_energy"])
            for record in diagnostics["fit_by_matrix"].values()
        )
        heldout_registered = sum(
            float(record["captured_energy"][str(rank)])
            for record in diagnostics["heldout_by_matrix"].values()
        ) / sum(
            float(record["total_energy"])
            for record in diagnostics["heldout_by_matrix"].values()
        )
        calibration_reconstructed = float(global_curves["calibration"]["all"][rank])
        heldout_reconstructed = float(global_curves["heldout"]["all"][rank])
        if (
            abs(calibration_reconstructed - calibration_registered) > ANCHOR_TOLERANCE
            or abs(heldout_reconstructed - heldout_registered) > ANCHOR_TOLERANCE
        ):
            raise ValueError(f"Registered rank-{rank} energy failed reconstruction.")
        anchor_checks.append(
            {
                "rank": rank,
                "calibration_reconstructed": calibration_reconstructed,
                "calibration_registered": calibration_registered,
                "calibration_absolute_error": abs(calibration_reconstructed - calibration_registered),
                "heldout_reconstructed": heldout_reconstructed,
                "heldout_registered": heldout_registered,
                "heldout_absolute_error": abs(heldout_reconstructed - heldout_registered),
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    global_rows = []
    for rank in range(EXPECTED_FEATURE_DIM + 1):
        global_rows.append(
            {
                "rank": rank,
                "rank_fraction": rank / EXPECTED_FEATURE_DIM,
                "calibration_all": float(global_curves["calibration"]["all"][rank]),
                "heldout_all": float(global_curves["heldout"]["all"][rank]),
                "calibration_K": float(global_curves["calibration"]["K"][rank]),
                "heldout_K": float(global_curves["heldout"]["K"][rank]),
                "calibration_V": float(global_curves["calibration"]["V"][rank]),
                "heldout_V": float(global_curves["heldout"]["V"][rank]),
            }
        )
    _write_csv(output / "cumulative_energy_global.csv", global_rows)
    _write_csv(output / "energy_threshold_rank_summary.csv", threshold_rows)
    _write_csv(output / "per_matrix_threshold_ranks.csv", per_matrix_rows)
    _write_csv(output / "layer_threshold_rank_summary.csv", layer_rows)
    _write_csv(output / "layer_threshold_rank_trends.csv", trend_rows)
    _write_csv(output / "calibration_rank_holdout_transfer.csv", transfer_rows)
    atomic_write_json(output / "candidate_next_online_ranks.json", candidates)
    figure_paths = _plots(output, global_curves, layer_rows, transfer_rows)
    provenance = {
        "basis_manifest_path": str(basis_path),
        "basis_manifest_sha256": sha256_file(basis_path),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
    }
    report = _markdown(
        threshold_rows=threshold_rows,
        candidates=candidates,
        transfer_rows=transfer_rows,
        trend_rows=trend_rows,
        anchor_checks=anchor_checks,
        provenance=provenance,
    )
    report_path = output / "energy_curve_analysis_summary.md"
    _write_text(report_path, report)
    atomic_write_json(
        output / "energy_curve_analysis_manifest.json",
        {
            "artifact_type": "asre_round4b_complete_cumulative_energy_analysis",
            "schema_version": 1,
            "protocol": ROUND4B_PROTOCOL,
            "status": "complete",
            "created_at": now_iso(),
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "feature_dim": EXPECTED_FEATURE_DIM,
            "matrix_count": len(matrices),
            "ranks": [0, EXPECTED_FEATURE_DIM],
            "coordinate_reports": coordinate_reports,
            "provenance": provenance,
            "checks": {
                "exact_matrix_count": len(matrices) == 30,
                "complete_rank_range": True,
                "monotonic": True,
                "rank_zero_exact": True,
                "rank_d_approximately_one": True,
                "registered_anchor_tolerance": ANCHOR_TOLERANCE,
                "registered_anchor_checks": anchor_checks,
                "heldout_svd_refit": False,
                "online_episodes": 0,
                "environment_rollouts": 0,
                "later_experiment_launched": False,
            },
            "outputs": {
                "global_curve": str(output / "cumulative_energy_global.csv"),
                "threshold_summary": str(output / "energy_threshold_rank_summary.csv"),
                "per_matrix_thresholds": str(output / "per_matrix_threshold_ranks.csv"),
                "transfer": str(output / "calibration_rank_holdout_transfer.csv"),
                "candidates": str(output / "candidate_next_online_ranks.json"),
                "summary": str(report_path),
                "figures": [str(path) for path in figure_paths],
            },
            "stop_rule_applied": True,
        },
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate-dir", type=Path, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    path = analyze(parser.parse_args())
    print(f"Round-4B cumulative energy analysis complete: {path}")


if __name__ == "__main__":
    main()
