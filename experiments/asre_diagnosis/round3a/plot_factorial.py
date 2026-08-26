"""Generate the five pre-registered ASRE Round-3A factorial figures.

The plotting stage is deliberately strict.  It consumes only the compact
factorial aggregate tables supplied through ``--aggregate-dir`` and validates
the complete 2^3 design before creating the output directory.  In particular,
it never discovers or rewrites Round-1 or Round-2 result directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


NUM_LAYERS = 30
FACTORS = ("A", "B", "C")
CELL_CODES = tuple(f"{value:03b}" for value in range(8))
CELL_COLUMNS = tuple(f"y{code}" for code in CELL_CODES)
INTERACTION_IDS = (
    "main_A",
    "main_B",
    "main_C",
    "interaction_AB",
    "interaction_AC",
    "interaction_BC",
    "interaction_ABC",
)
PLOT_FILENAMES = (
    "figure_A_complete_factorial_cell_plot.png",
    "figure_B_contextual_contribution_of_B.png",
    "figure_C_pairwise_interaction_summary.png",
    "figure_D_retrieval_schedule_schematic.png",
    "figure_E_task_level_factorial_heatmap.png",
)

CELL_REQUIRED_COLUMNS = {
    "cell_code",
    "A",
    "B",
    "C",
    "online_success_rate",
    "paired_ci_low",
    "paired_ci_high",
    "task_hierarchical_ci_low",
    "task_hierarchical_ci_high",
    "enabled_video_retrieval_layers",
}
SIMPLE_REQUIRED_COLUMNS = {
    "effect_id",
    "factor",
    "context_A",
    "context_B",
    "context_C",
    "effect",
    "paired_ci_low",
    "paired_ci_high",
    "task_hierarchical_ci_low",
    "task_hierarchical_ci_high",
}
INTERACTION_REQUIRED_COLUMNS = {
    "contrast_id",
    "effect",
    "paired_ci_low",
    "paired_ci_high",
    "task_hierarchical_ci_low",
    "task_hierarchical_ci_high",
}
TASK_REQUIRED_COLUMNS = {
    "task_id",
    "task_description",
    *CELL_COLUMNS,
}


@dataclass(frozen=True)
class Estimate:
    point: float
    paired_low: float
    paired_high: float
    hierarchical_low: float
    hierarchical_high: float


@dataclass(frozen=True)
class CellRecord:
    code: str
    a: int
    b: int
    c: int
    enabled_layers: tuple[int, ...]
    estimate: Estimate


@dataclass(frozen=True)
class SimpleEffectRecord:
    effect_id: str
    factor: str
    context: tuple[int | None, int | None, int | None]
    estimate: Estimate


@dataclass(frozen=True)
class TaskRecord:
    task_id: int
    description: str
    rates: tuple[float, ...]


@dataclass(frozen=True)
class PlotInputs:
    cells: tuple[CellRecord, ...]
    simple_effects: tuple[SimpleEffectRecord, ...]
    interactions: Mapping[str, Estimate]
    tasks: tuple[TaskRecord, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the complete ASRE Round-3A 2^3 factorial analysis."
    )
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _read_csv(
    path: Path,
    *,
    required_columns: set[str],
    label: str,
) -> list[dict[str, str]]:
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked {label} input: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} table is unavailable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"{label} has no CSV header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{label} has duplicate CSV columns: {path}")
        missing = sorted(required_columns - set(fieldnames))
        if missing:
            raise ValueError(f"{label} is missing required columns {missing}: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{label} contains no data rows: {path}")
    if any(None in row for row in rows):
        raise ValueError(f"{label} contains rows wider than its CSV header: {path}")
    return rows


def _text(row: Mapping[str, str], key: str, *, context: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing {key!r} in {context}.")
    return value


def _integer(row: Mapping[str, str], key: str, *, context: str) -> int:
    value = str(row.get(key, "")).strip()
    if re.fullmatch(r"-?(0|[1-9][0-9]*)", value) is None:
        raise ValueError(f"Expected integer {key!r} in {context}, got {value!r}.")
    return int(value)


def _binary(row: Mapping[str, str], key: str, *, context: str) -> int:
    value = _integer(row, key, context=context)
    if value not in (0, 1):
        raise ValueError(f"Expected binary {key!r} in {context}, got {value}.")
    return value


def _finite(row: Mapping[str, str], key: str, *, context: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected numeric {key!r} in {context}, got {row.get(key)!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"Expected finite {key!r} in {context}, got {value!r}.")
    return value


def _estimate(row: Mapping[str, str], *, context: str) -> Estimate:
    estimate = Estimate(
        point=_finite(row, "effect" if "effect" in row else "online_success_rate", context=context),
        paired_low=_finite(row, "paired_ci_low", context=context),
        paired_high=_finite(row, "paired_ci_high", context=context),
        hierarchical_low=_finite(
            row, "task_hierarchical_ci_low", context=context
        ),
        hierarchical_high=_finite(
            row, "task_hierarchical_ci_high", context=context
        ),
    )
    if estimate.paired_low > estimate.paired_high:
        raise ValueError(f"Reversed paired confidence interval in {context}.")
    if estimate.hierarchical_low > estimate.hierarchical_high:
        raise ValueError(f"Reversed task-hierarchical confidence interval in {context}.")
    return estimate


def _enabled_layers(value: str, *, context: str) -> tuple[int, ...]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid enabled_video_retrieval_layers JSON in {context}."
        ) from exc
    if not isinstance(parsed, list) or any(
        isinstance(layer, bool) or not isinstance(layer, int) for layer in parsed
    ):
        raise ValueError(
            f"enabled_video_retrieval_layers must be a JSON integer list in {context}."
        )
    if parsed != sorted(set(parsed)):
        raise ValueError(
            f"enabled_video_retrieval_layers must be sorted and unique in {context}."
        )
    if any(layer < 0 or layer >= NUM_LAYERS for layer in parsed):
        raise ValueError(f"Out-of-range enabled layer in {context}: {parsed}.")
    return tuple(parsed)


def _expected_enabled_layers(a: int, b: int, c: int) -> tuple[int, ...]:
    enabled: list[int] = []
    if a:
        enabled.extend(range(15, 20))
    if b:
        enabled.extend(range(20, 25))
    if c:
        enabled.extend(range(25, 30))
    return tuple(enabled)


def _load_cells(path: Path) -> tuple[CellRecord, ...]:
    rows = _read_csv(
        path,
        required_columns=CELL_REQUIRED_COLUMNS,
        label="factorial cell",
    )
    by_code: dict[str, CellRecord] = {}
    for row_index, row in enumerate(rows, start=2):
        context = f"{path}:{row_index}"
        code = _text(row, "cell_code", context=context)
        if code not in CELL_CODES:
            raise ValueError(f"Unexpected factorial cell code {code!r} in {context}.")
        if code in by_code:
            raise ValueError(f"Duplicate factorial cell {code!r} in {path}.")
        a = _binary(row, "A", context=context)
        b = _binary(row, "B", context=context)
        c = _binary(row, "C", context=context)
        if code != f"{a}{b}{c}":
            raise ValueError(
                f"Factor columns {(a, b, c)} disagree with cell_code={code!r} in {context}."
            )
        enabled = _enabled_layers(
            row["enabled_video_retrieval_layers"], context=context
        )
        expected_enabled = _expected_enabled_layers(a, b, c)
        if enabled != expected_enabled:
            raise ValueError(
                f"Cell {code} enables {list(enabled)}, expected {list(expected_enabled)}; "
                "layers 0-14 must remain disabled in every factorial cell."
            )
        estimate = _estimate(row, context=context)
        for field_name, value in (
            ("online_success_rate", estimate.point),
            ("paired_ci_low", estimate.paired_low),
            ("paired_ci_high", estimate.paired_high),
            ("task_hierarchical_ci_low", estimate.hierarchical_low),
            ("task_hierarchical_ci_high", estimate.hierarchical_high),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    f"Cell probability {field_name}={value} is outside [0, 1] in {context}."
                )
        by_code[code] = CellRecord(code, a, b, c, enabled, estimate)
    if set(by_code) != set(CELL_CODES):
        raise ValueError(
            "Factorial cell table must contain exactly 000..111: "
            f"missing={sorted(set(CELL_CODES) - set(by_code))}, "
            f"unexpected={sorted(set(by_code) - set(CELL_CODES))}."
        )
    return tuple(by_code[code] for code in CELL_CODES)


def _context_value(
    row: Mapping[str, str], factor: str, manipulated_factor: str, *, context: str
) -> int | None:
    key = f"context_{factor}"
    raw = str(row.get(key, "")).strip()
    if factor == manipulated_factor:
        if raw:
            raise ValueError(
                f"{key} must be blank when factor {factor} is manipulated in {context}."
            )
        return None
    if raw not in {"0", "1"}:
        raise ValueError(f"{key} must be 0 or 1 in {context}, got {raw!r}.")
    return int(raw)


def _load_simple_effects(path: Path) -> tuple[SimpleEffectRecord, ...]:
    rows = _read_csv(
        path,
        required_columns=SIMPLE_REQUIRED_COLUMNS,
        label="factorial simple-effect",
    )
    records: list[SimpleEffectRecord] = []
    seen_ids: set[str] = set()
    seen_contexts: set[tuple[str, tuple[int | None, int | None, int | None]]] = set()
    for row_index, row in enumerate(rows, start=2):
        context_label = f"{path}:{row_index}"
        effect_id = _text(row, "effect_id", context=context_label)
        if effect_id in seen_ids:
            raise ValueError(f"Duplicate simple-effect ID {effect_id!r} in {path}.")
        factor = _text(row, "factor", context=context_label).upper()
        if factor not in FACTORS:
            raise ValueError(f"Unexpected manipulated factor {factor!r} in {context_label}.")
        context = tuple(
            _context_value(row, name, factor, context=context_label)
            for name in FACTORS
        )
        context_key = (factor, context)
        if context_key in seen_contexts:
            raise ValueError(
                f"Duplicate simple effect for factor {factor}, context {context} in {path}."
            )
        estimate = _estimate(row, context=context_label)
        records.append(SimpleEffectRecord(effect_id, factor, context, estimate))
        seen_ids.add(effect_id)
        seen_contexts.add(context_key)

    expected_contexts = {
        "A": {(None, b, c) for b in (0, 1) for c in (0, 1)},
        "B": {(a, None, c) for a in (0, 1) for c in (0, 1)},
        "C": {(a, b, None) for a in (0, 1) for b in (0, 1)},
    }
    for factor in FACTORS:
        observed = {record.context for record in records if record.factor == factor}
        if observed != expected_contexts[factor]:
            raise ValueError(
                f"Simple effects for factor {factor} are incomplete: "
                f"missing={sorted(expected_contexts[factor] - observed, key=str)}, "
                f"unexpected={sorted(observed - expected_contexts[factor], key=str)}."
            )
    if len(records) != 12:
        raise ValueError(f"Expected exactly 12 factorial simple effects, got {len(records)}.")
    return tuple(records)


def _load_interactions(path: Path) -> dict[str, Estimate]:
    rows = _read_csv(
        path,
        required_columns=INTERACTION_REQUIRED_COLUMNS,
        label="factorial contrast",
    )
    by_id: dict[str, Estimate] = {}
    for row_index, row in enumerate(rows, start=2):
        context = f"{path}:{row_index}"
        contrast_id = _text(row, "contrast_id", context=context)
        if contrast_id not in INTERACTION_IDS:
            raise ValueError(f"Unexpected factorial contrast {contrast_id!r} in {context}.")
        if contrast_id in by_id:
            raise ValueError(f"Duplicate factorial contrast {contrast_id!r} in {path}.")
        by_id[contrast_id] = _estimate(row, context=context)
    if set(by_id) != set(INTERACTION_IDS):
        raise ValueError(
            "Factorial contrast table must contain the three main, three pairwise, "
            "and one three-way contrast: "
            f"missing={sorted(set(INTERACTION_IDS) - set(by_id))}, "
            f"unexpected={sorted(set(by_id) - set(INTERACTION_IDS))}."
        )
    return by_id


def _load_tasks(path: Path) -> tuple[TaskRecord, ...]:
    rows = _read_csv(
        path,
        required_columns=TASK_REQUIRED_COLUMNS,
        label="factorial task-success",
    )
    by_id: dict[int, TaskRecord] = {}
    for row_index, row in enumerate(rows, start=2):
        context = f"{path}:{row_index}"
        task_id = _integer(row, "task_id", context=context)
        if task_id in by_id:
            raise ValueError(f"Duplicate task_id={task_id} in {path}.")
        description = _text(row, "task_description", context=context)
        rates = tuple(_finite(row, column, context=context) for column in CELL_COLUMNS)
        if any(rate < 0.0 or rate > 1.0 for rate in rates):
            raise ValueError(f"Task success rates must lie in [0, 1] in {context}.")
        by_id[task_id] = TaskRecord(task_id, description, rates)
    expected_ids = set(range(10))
    if set(by_id) != expected_ids:
        raise ValueError(
            "Factorial task table must contain LIBERO-Spatial task IDs 0..9: "
            f"missing={sorted(expected_ids - set(by_id))}, "
            f"unexpected={sorted(set(by_id) - expected_ids)}."
        )
    return tuple(by_id[task_id] for task_id in range(10))


def _load_plot_inputs(aggregate_dir: Path) -> PlotInputs:
    resolved = aggregate_dir.resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"Aggregate directory is unavailable: {resolved}")
    return PlotInputs(
        cells=_load_cells(resolved / "factorial_cells.csv"),
        simple_effects=_load_simple_effects(
            resolved / "factorial_simple_effects.csv"
        ),
        interactions=_load_interactions(
            resolved / "factorial_interactions.csv"
        ),
        tasks=_load_tasks(resolved / "factorial_task_success.csv"),
    )


def _short_description(task: TaskRecord, limit: int = 64) -> str:
    clean = " ".join(task.description.split())
    if len(clean) > limit:
        clean = clean[: limit - 1].rstrip() + "…"
    return f"{task.task_id}: {clean}"


def _uncertainty_plot(
    axis,
    x: np.ndarray,
    estimates: Sequence[Estimate],
    *,
    scale: float = 100.0,
    point_color: str = "#1F4E79",
) -> None:
    points = np.asarray([estimate.point for estimate in estimates]) * scale
    paired_low = np.asarray([estimate.paired_low for estimate in estimates]) * scale
    paired_high = np.asarray([estimate.paired_high for estimate in estimates]) * scale
    hierarchical_low = np.asarray(
        [estimate.hierarchical_low for estimate in estimates]
    ) * scale
    hierarchical_high = np.asarray(
        [estimate.hierarchical_high for estimate in estimates]
    ) * scale
    axis.vlines(
        x,
        hierarchical_low,
        hierarchical_high,
        color="#4C78A8",
        linewidth=5.0,
        alpha=0.34,
        zorder=1,
    )
    axis.vlines(
        x,
        paired_low,
        paired_high,
        color=point_color,
        linewidth=1.5,
        zorder=2,
    )
    axis.scatter(x, points, color=point_color, s=38, zorder=3)


def _plot_all(inputs: PlotInputs, output_dir: Path, *, dpi: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError(
            "Round-3A plotting requires matplotlib in the analysis environment."
        ) from exc

    uncertainty_legend = [
        Line2D(
            [0],
            [0],
            color="#1F4E79",
            marker="o",
            label="Paired bootstrap 95% CI",
        ),
        Line2D(
            [0],
            [0],
            color="#4C78A8",
            linewidth=5,
            alpha=0.34,
            label="Task-hierarchical 95% CI",
        ),
    ]
    interaction_uncertainty_legend = [
        Line2D(
            [0],
            [0],
            color="#6A3D9A",
            marker="o",
            label="Paired bootstrap 95% CI",
        ),
        uncertainty_legend[1],
    ]

    # Figure A: all cells in factorial order, never ordered by performance.
    x = np.arange(len(inputs.cells))
    fig, axis = plt.subplots(figsize=(10.8, 5.4))
    _uncertainty_plot(axis, x, [cell.estimate for cell in inputs.cells])
    axis.set_xticks(
        x,
        [f"{cell.code}\nA{cell.a} B{cell.b} C{cell.c}" for cell in inputs.cells],
    )
    axis.set_ylim(-3.0, 103.0)
    axis.set_xlabel("Late-half retrieval factorial cell")
    axis.set_ylabel("LIBERO-Spatial success rate (%)")
    axis.set_title("Complete 2³ factorial cell performance")
    axis.axvline(3.5, color="#777777", linewidth=0.8, linestyle=":")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(handles=uncertainty_legend, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / PLOT_FILENAMES[0], dpi=dpi)
    plt.close(fig)

    # Figure B: the four pre-specified contextual effects of B.
    b_lookup = {
        (record.context[0], record.context[2]): record
        for record in inputs.simple_effects
        if record.factor == "B"
    }
    b_order = ((0, 0), (1, 0), (0, 1), (1, 1))
    b_records = [b_lookup[context] for context in b_order]
    x = np.arange(4)
    fig, axis = plt.subplots(figsize=(8.8, 5.3))
    _uncertainty_plot(axis, x, [record.estimate for record in b_records])
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(
        x,
        [f"B | A={a}, C={c}" for a, c in b_order],
        rotation=12,
        ha="right",
    )
    axis.set_xlabel("Fixed retrieval context")
    axis.set_ylabel("Simple conditional effect of B (percentage points)")
    axis.set_title("Contextual contribution of region B")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(handles=uncertainty_legend, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / PLOT_FILENAMES[1], dpi=dpi)
    plt.close(fig)

    # Figure C: pairwise interactions, with the three-way contrast separated.
    contrast_order = (
        "interaction_AB",
        "interaction_AC",
        "interaction_BC",
        "interaction_ABC",
    )
    contrast_labels = ("A×B", "A×C", "B×C", "A×B×C")
    contrast_x = np.asarray([0.0, 1.0, 2.0, 3.35])
    fig, axis = plt.subplots(figsize=(8.7, 5.3))
    _uncertainty_plot(
        axis,
        contrast_x,
        [inputs.interactions[name] for name in contrast_order],
        point_color="#6A3D9A",
    )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.axvline(2.68, color="#777777", linewidth=0.8, linestyle=":")
    axis.set_xticks(contrast_x, contrast_labels)
    axis.set_ylabel("Factorial interaction contrast (percentage points)")
    axis.set_xlabel("Probability-scale factorial contrast")
    axis.set_title("Pairwise and three-way interaction summary")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(handles=interaction_uncertainty_legend, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / PLOT_FILENAMES[2], dpi=dpi)
    plt.close(fig)

    # Figure D: exact enabled-layer schedules. Validation guarantees 0-14 are off.
    enabled_matrix = np.zeros((len(inputs.cells), NUM_LAYERS), dtype=int)
    for row_index, cell in enumerate(inputs.cells):
        enabled_matrix[row_index, list(cell.enabled_layers)] = 1
    fig, axis = plt.subplots(figsize=(14.2, 4.9))
    axis.imshow(
        enabled_matrix,
        cmap=ListedColormap(["#ECECEC", "#2166AC"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    axis.set_xticks(
        np.arange(NUM_LAYERS),
        [str(layer) for layer in range(NUM_LAYERS)],
        fontsize=7,
    )
    axis.set_yticks(np.arange(8), [cell.code for cell in inputs.cells])
    axis.set_xlabel("Action layer")
    axis.set_ylabel("Factorial cell (ABC)")
    axis.set_title("Direct video K/V retrieval schedules (layers 0–14 fixed OFF)")
    for boundary in (14.5, 19.5, 24.5):
        axis.axvline(boundary, color="#555555", linewidth=1.0)
    axis.set_xticks(np.arange(-0.5, NUM_LAYERS, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(inputs.cells), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.55)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                color="#2166AC",
                markersize=9,
                label="video K/V enabled",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                color="#ECECEC",
                markeredgecolor="#999999",
                markersize=9,
                label="video K/V disabled",
            ),
        ],
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(output_dir / PLOT_FILENAMES[3], dpi=dpi)
    plt.close(fig)

    # Figure E: task-wise success deltas relative to the full late-half cell y111.
    heatmap = np.asarray(
        [
            [(rate - task.rates[-1]) * 100.0 for rate in task.rates]
            for task in inputs.tasks
        ],
        dtype=float,
    )
    max_abs = max(10.0, float(np.max(np.abs(heatmap))))
    fig, axis = plt.subplots(figsize=(12.2, 8.2))
    image = axis.imshow(
        heatmap,
        cmap="RdBu",
        vmin=-max_abs,
        vmax=max_abs,
        interpolation="nearest",
        aspect="auto",
    )
    axis.set_xticks(np.arange(8), CELL_CODES)
    axis.set_yticks(
        np.arange(10),
        [_short_description(task) for task in inputs.tasks],
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
    axis.set_xlabel("Factorial cell (ABC)")
    axis.set_ylabel("LIBERO-Spatial task")
    axis.set_title("Task-level success delta relative to y111")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Success-rate delta vs y111 (percentage points)")
    fig.tight_layout()
    fig.savefig(output_dir / PLOT_FILENAMES[4], dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    aggregate_dir = args.aggregate_dir.resolve()
    inputs = _load_plot_inputs(aggregate_dir)

    output_dir = aggregate_dir / "plots"
    if output_dir.is_symlink():
        raise ValueError(f"Refusing symlinked plot output directory: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Plot output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Refusing to modify a non-empty Round-3A plot directory: "
            f"{output_dir}"
        )
    if output_dir.exists():
        output_dir.rmdir()
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".round3a_plots.", dir=aggregate_dir)
    )
    try:
        _plot_all(inputs, temporary_dir, dpi=args.dpi)
        produced = {path.name for path in temporary_dir.iterdir() if path.is_file()}
        if produced != set(PLOT_FILENAMES):
            raise RuntimeError(
                "Round-3A plot output set is incomplete: "
                f"missing={sorted(set(PLOT_FILENAMES) - produced)}, "
                f"unexpected={sorted(produced - set(PLOT_FILENAMES))}."
            )
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    print(f"Wrote five ASRE Round-3A factorial figures to {output_dir}")


if __name__ == "__main__":
    main()
