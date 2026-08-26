from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.asre_diagnosis.round3a.plot_factorial import (
    CELL_CODES,
    INTERACTION_IDS,
    PLOT_FILENAMES,
    _load_plot_inputs,
    main as plot_main,
)


CELL_RATES = {
    "000": 0.00,
    "001": 0.00,
    "010": 0.12,
    "011": 0.49,
    "100": 0.08,
    "101": 0.86,
    "110": 0.44,
    "111": 0.95,
}


def _enabled_layers(code: str) -> list[int]:
    enabled: list[int] = []
    if code[0] == "1":
        enabled.extend(range(15, 20))
    if code[1] == "1":
        enabled.extend(range(20, 25))
    if code[2] == "1":
        enabled.extend(range(25, 30))
    return enabled


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise AssertionError("Synthetic CSV helper requires at least one row.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _interval_fields(point: float) -> dict[str, float]:
    return {
        "paired_ci_low": point - 0.04,
        "paired_ci_high": point + 0.04,
        "task_hierarchical_ci_low": point - 0.07,
        "task_hierarchical_ci_high": point + 0.07,
    }


def _probability_interval_fields(point: float) -> dict[str, float]:
    return {
        "paired_ci_low": max(0.0, point - 0.03),
        "paired_ci_high": min(1.0, point + 0.03),
        "task_hierarchical_ci_low": max(0.0, point - 0.05),
        "task_hierarchical_ci_high": min(1.0, point + 0.05),
    }


def _cell(code: str) -> float:
    return CELL_RATES[code]


def _factorial_contrasts() -> dict[str, float]:
    main_a = sum(
        _cell(f"1{b}{c}") - _cell(f"0{b}{c}")
        for b in (0, 1)
        for c in (0, 1)
    ) / 4.0
    main_b = sum(
        _cell(f"{a}1{c}") - _cell(f"{a}0{c}")
        for a in (0, 1)
        for c in (0, 1)
    ) / 4.0
    main_c = sum(
        _cell(f"{a}{b}1") - _cell(f"{a}{b}0")
        for a in (0, 1)
        for b in (0, 1)
    ) / 4.0
    interaction_ab = sum(
        _cell(f"11{c}")
        - _cell(f"01{c}")
        - _cell(f"10{c}")
        + _cell(f"00{c}")
        for c in (0, 1)
    ) / 2.0
    interaction_ac = sum(
        _cell(f"1{b}1")
        - _cell(f"0{b}1")
        - _cell(f"1{b}0")
        + _cell(f"0{b}0")
        for b in (0, 1)
    ) / 2.0
    interaction_bc = sum(
        _cell(f"{a}11")
        - _cell(f"{a}01")
        - _cell(f"{a}10")
        + _cell(f"{a}00")
        for a in (0, 1)
    ) / 2.0
    interaction_abc = (
        _cell("111")
        - _cell("011")
        - _cell("101")
        + _cell("001")
        - _cell("110")
        + _cell("010")
        + _cell("100")
        - _cell("000")
    )
    return {
        "main_A": main_a,
        "main_B": main_b,
        "main_C": main_c,
        "interaction_AB": interaction_ab,
        "interaction_AC": interaction_ac,
        "interaction_BC": interaction_bc,
        "interaction_ABC": interaction_abc,
    }


def _write_synthetic_aggregate(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    cell_rows: list[dict[str, object]] = []
    for code in CELL_CODES:
        rate = CELL_RATES[code]
        cell_rows.append(
            {
                "cell_code": code,
                "A": int(code[0]),
                "B": int(code[1]),
                "C": int(code[2]),
                "online_success_rate": rate,
                **_probability_interval_fields(rate),
                "enabled_video_retrieval_layers": json.dumps(
                    _enabled_layers(code), separators=(",", ":")
                ),
            }
        )
    _write_csv(root / "factorial_cells.csv", cell_rows)

    simple_rows: list[dict[str, object]] = []
    for factor in ("A", "B", "C"):
        manipulated_index = {"A": 0, "B": 1, "C": 2}[factor]
        fixed_indices = [index for index in range(3) if index != manipulated_index]
        for first in (0, 1):
            for second in (0, 1):
                low_bits = [0, 0, 0]
                high_bits = [0, 0, 0]
                low_bits[fixed_indices[0]] = high_bits[fixed_indices[0]] = first
                low_bits[fixed_indices[1]] = high_bits[fixed_indices[1]] = second
                high_bits[manipulated_index] = 1
                low_code = "".join(str(value) for value in low_bits)
                high_code = "".join(str(value) for value in high_bits)
                effect = CELL_RATES[high_code] - CELL_RATES[low_code]
                contexts: dict[str, object] = {}
                for index, name in enumerate(("A", "B", "C")):
                    contexts[f"context_{name}"] = (
                        "" if index == manipulated_index else low_bits[index]
                    )
                simple_rows.append(
                    {
                        "effect_id": f"effect_{factor}_{low_code}_to_{high_code}",
                        "factor": factor,
                        **contexts,
                        "effect": effect,
                        **_interval_fields(effect),
                    }
                )
    _write_csv(root / "factorial_simple_effects.csv", simple_rows)

    contrast_values = _factorial_contrasts()
    self_order_check = tuple(contrast_values)
    if self_order_check != INTERACTION_IDS:
        raise AssertionError(self_order_check)
    interaction_rows = [
        {
            "contrast_id": contrast_id,
            "effect": effect,
            **_interval_fields(effect),
        }
        for contrast_id, effect in contrast_values.items()
    ]
    _write_csv(root / "factorial_interactions.csv", interaction_rows)

    task_rows: list[dict[str, object]] = []
    for task_id in range(10):
        offset = ((task_id % 3) - 1) * 0.02
        task_rows.append(
            {
                "task_id": task_id,
                "task_description": f"synthetic LIBERO-Spatial task {task_id}",
                **{
                    f"y{code}": min(1.0, max(0.0, CELL_RATES[code] + offset))
                    for code in CELL_CODES
                },
            }
        )
    _write_csv(root / "factorial_task_success.csv", task_rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Round3APlotTest(unittest.TestCase):
    def test_full_runner_checks_the_plotter_output_names(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "round3a"
            / "run_full_experiment.sh"
        ).read_text(encoding="utf-8")
        for filename in PLOT_FILENAMES:
            self.assertIn(filename, runner)

    def test_synthetic_tables_generate_exactly_five_headless_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "round3a" / "aggregate"
            _write_synthetic_aggregate(aggregate)
            matplotlib_cache = Path(directory) / "matplotlib"
            with mock.patch.dict(
                "os.environ", {"MPLCONFIGDIR": str(matplotlib_cache)}, clear=False
            ), mock.patch.object(
                sys,
                "argv",
                [
                    "plot_factorial.py",
                    "--aggregate-dir",
                    str(aggregate),
                    "--dpi",
                    "35",
                ],
            ):
                plot_main()

            plots = aggregate / "plots"
            observed = {path.name for path in plots.iterdir() if path.is_file()}
            self.assertEqual(observed, set(PLOT_FILENAMES))
            for filename in PLOT_FILENAMES:
                self.assertGreater((plots / filename).stat().st_size, 100)

    def test_missing_factorial_cell_is_rejected_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate"
            _write_synthetic_aggregate(aggregate)
            rows = _read_rows(aggregate / "factorial_cells.csv")[:-1]
            _write_csv(aggregate / "factorial_cells.csv", rows)

            with self.assertRaisesRegex(ValueError, "exactly 000..111"):
                _load_plot_inputs(aggregate)
            self.assertFalse((aggregate / "plots").exists())

    def test_plot_failure_leaves_no_partial_public_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate"
            _write_synthetic_aggregate(aggregate)
            with mock.patch.object(
                sys,
                "argv",
                ["plot_factorial.py", "--aggregate-dir", str(aggregate)],
            ), mock.patch(
                "experiments.asre_diagnosis.round3a.plot_factorial._plot_all",
                side_effect=RuntimeError("synthetic plot failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic plot failure"):
                    plot_main()
            self.assertFalse((aggregate / "plots").exists())
            self.assertFalse(
                any(path.name.startswith(".round3a_plots.") for path in aggregate.iterdir())
            )

    def test_early_layer_enabled_in_any_cell_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate"
            _write_synthetic_aggregate(aggregate)
            rows = _read_rows(aggregate / "factorial_cells.csv")
            rows[0]["enabled_video_retrieval_layers"] = "[0]"
            _write_csv(aggregate / "factorial_cells.csv", rows)

            with self.assertRaisesRegex(ValueError, "layers 0-14"):
                _load_plot_inputs(aggregate)

    def test_duplicate_simple_effect_context_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate"
            _write_synthetic_aggregate(aggregate)
            path = aggregate / "factorial_simple_effects.csv"
            rows = _read_rows(path)
            b_rows = [index for index, row in enumerate(rows) if row["factor"] == "B"]
            rows[b_rows[1]]["context_A"] = rows[b_rows[0]]["context_A"]
            rows[b_rows[1]]["context_C"] = rows[b_rows[0]]["context_C"]
            _write_csv(path, rows)

            with self.assertRaisesRegex(ValueError, "Duplicate simple effect"):
                _load_plot_inputs(aggregate)

    def test_missing_task_cell_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate"
            _write_synthetic_aggregate(aggregate)
            path = aggregate / "factorial_task_success.csv"
            rows = _read_rows(path)
            rows_without_y111 = [
                {key: value for key, value in row.items() if key != "y111"}
                for row in rows
            ]
            _write_csv(path, rows_without_y111)

            with self.assertRaisesRegex(ValueError, "missing required columns.*y111"):
                _load_plot_inputs(aggregate)


if __name__ == "__main__":
    unittest.main()
