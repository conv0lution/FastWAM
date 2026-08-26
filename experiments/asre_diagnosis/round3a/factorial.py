"""Immutable mapping between ASRE conditions and the late-half 2^3 cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.asre_diagnosis.common import (
    ROUND2_PROTOCOL,
    ROUND3A_PROTOCOL,
    DiagnosisCondition,
    build_round2_conditions,
    build_round3a_conditions,
)


NUM_LAYERS = 30
CELL_ORDER = ("000", "001", "010", "011", "100", "101", "110", "111")


@dataclass(frozen=True)
class FactorialCell:
    code: str
    condition: DiagnosisCondition
    protocol: str

    @property
    def a(self) -> int:
        return int(self.code[0])

    @property
    def b(self) -> int:
        return int(self.code[1])

    @property
    def c(self) -> int:
        return int(self.code[2])

    @property
    def enabled_layers(self) -> tuple[int, ...]:
        return self.condition.enabled_video_retrieval_layers(NUM_LAYERS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_code": self.code,
            "A": self.a,
            "B": self.b,
            "C": self.c,
            "condition": self.condition.name,
            "protocol": self.protocol,
            "enabled_video_retrieval_layers": list(self.enabled_layers),
            "disabled_video_layers": list(self.condition.disabled_video_layers),
            "num_retrieval_layers": len(self.enabled_layers),
        }


def build_factorial_cells() -> tuple[FactorialCell, ...]:
    """Return all cells in binary order without changing either source protocol."""
    round2 = {condition.name: condition for condition in build_round2_conditions(NUM_LAYERS)}
    round3a = {condition.name: condition for condition in build_round3a_conditions(NUM_LAYERS)}
    definitions = (
        ("000", round3a["keep_none_late"], ROUND3A_PROTOCOL),
        ("001", round2["keep_25_29"], ROUND2_PROTOCOL),
        ("010", round3a["keep_20_24"], ROUND3A_PROTOCOL),
        ("011", round2["keep_20_29"], ROUND2_PROTOCOL),
        ("100", round2["keep_15_19"], ROUND2_PROTOCOL),
        ("101", round2["keep_15_19_25_29"], ROUND2_PROTOCOL),
        ("110", round3a["keep_15_24"], ROUND3A_PROTOCOL),
        ("111", round2["keep_15_29"], ROUND2_PROTOCOL),
    )
    cells = tuple(
        FactorialCell(code=code, condition=condition, protocol=protocol)
        for code, condition, protocol in definitions
    )
    _validate_cells(cells)
    return cells


def _validate_cells(cells: tuple[FactorialCell, ...]) -> None:
    if tuple(cell.code for cell in cells) != CELL_ORDER:
        raise AssertionError("Late-half factorial cells are not in canonical binary order.")
    if len({cell.condition.name for cell in cells}) != len(CELL_ORDER):
        raise AssertionError("Each factorial cell must map to a distinct condition.")
    for cell in cells:
        expected = (
            tuple(range(15, 20)) if cell.a else ()
        ) + (
            tuple(range(20, 25)) if cell.b else ()
        ) + (
            tuple(range(25, 30)) if cell.c else ()
        )
        if cell.enabled_layers != expected:
            raise AssertionError(
                f"Cell {cell.code} maps to enabled layers {cell.enabled_layers}, "
                f"expected {expected}."
            )


FACTORIAL_CELLS = build_factorial_cells()
CELL_BY_CODE = {cell.code: cell for cell in FACTORIAL_CELLS}
CELL_BY_CONDITION = {cell.condition.name: cell for cell in FACTORIAL_CELLS}

