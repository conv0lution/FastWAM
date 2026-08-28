"""Frozen registration and pure validation helpers for ASRE G0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.asre_diagnosis.common import (
    G0_PROTOCOL,
    DiagnosisCondition,
    build_g0_conditions,
)


NUM_LAYERS = 30
SEED = 42
NUM_TRIALS = 10
SMOKE_TRIALS = 2
TASK_IDS = tuple(range(10))
SMOKE_TASK_IDS = (0,)
ACTION_HORIZON = 32
INFERENCE_STEPS = 10
REPLAN_STEPS = 10
NUM_STEPS_WAIT = 30
TASK_CONFIG = "libero_uncond_2cam224_1e-4"
CHECKPOINT_NAME = "libero_uncond_2cam224.pt"

SUITE_ORDER = ("libero_object", "libero_goal", "libero_10")
REFERENCE_SUITE = "libero_spatial"
CONDITIONS = tuple(build_g0_conditions(NUM_LAYERS))
CONDITION_ORDER = tuple(condition.name for condition in CONDITIONS)
GPU_BY_CONDITION = {name: index for index, name in enumerate(CONDITION_ORDER)}

ROUND3A_TAG = "ASRE-round3a-factorial"
ROUND3A_COMMIT = "d36383c16d974ba1e5a750c088327a7a88baa8fb"
ROUND3B_TAG = "ASRE-round3b-kv-replacement"
ROUND3B_COMMIT = "fb771af7ec32dd8e8bd12b0eedea735e307771a1"

PROTECTED_STAGE1_DIRS = (
    "asre_results/aggregate",
    "asre_results/logs",
    "asre_results/offline",
    "asre_results/online_full",
    "asre_results/online_smoke",
    "asre_results/result_summary_for_gpt.md",
    "asre_results/state_bank",
    "asre_results/round2",
    "asre_results/round3a",
    "asre_results/round3b",
)

PRIMARY_CONTRASTS = {
    "delta_late": ("full_current", "late_current_15_29"),
    "delta_early_vs_late": ("late_current_15_29", "early_current_00_19"),
    "delta_wrong_vs_late": (
        "late_current_15_29",
        "late_wrong_scene_15_29",
    ),
}


@dataclass(frozen=True)
class RuntimeSpec:
    suite: str
    mode: str
    task_ids: tuple[int, ...]
    num_trials: int
    action_horizon: int = ACTION_HORIZON
    inference_steps: int = INFERENCE_STEPS
    replan_steps: int = REPLAN_STEPS


def condition_by_name(name: str) -> DiagnosisCondition:
    by_name = {condition.name: condition for condition in CONDITIONS}
    try:
        return by_name[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown G0 condition {name!r}; expected one of {list(by_name)}."
        ) from exc


def validate_gpu_mapping(values: Sequence[int]) -> tuple[int, int, int, int]:
    gpu_ids = tuple(int(value) for value in values)
    if len(gpu_ids) != 4:
        raise ValueError(f"G0 requires exactly four GPU IDs, got {gpu_ids}.")
    if any(value < 0 for value in gpu_ids):
        raise ValueError(f"GPU IDs must be nonnegative, got {gpu_ids}.")
    if len(set(gpu_ids)) != 4:
        raise ValueError(f"G0 requires four distinct physical GPUs, got {gpu_ids}.")
    return gpu_ids  # type: ignore[return-value]


def runtime_for(suite: str, mode: str) -> RuntimeSpec:
    if suite not in SUITE_ORDER:
        raise ValueError(f"Unsupported G0 suite {suite!r}; expected {list(SUITE_ORDER)}.")
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported G0 mode {mode!r}.")
    return RuntimeSpec(
        suite=suite,
        mode=mode,
        task_ids=SMOKE_TASK_IDS if mode == "smoke" else TASK_IDS,
        num_trials=SMOKE_TRIALS if mode == "smoke" else NUM_TRIALS,
    )


def assert_output_scope(output_root: Path, project_root: Path) -> None:
    root = output_root.expanduser().resolve()
    expected = (project_root / "asre_results" / "g0_cross_suite").resolve()
    if root != expected and not root.is_relative_to(expected):
        raise ValueError(f"G0 output must be within {expected}, got {root}.")
    for relative in PROTECTED_STAGE1_DIRS:
        protected = (project_root / relative).resolve()
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError(f"G0 output overlaps frozen Stage-1 artifacts: {protected}.")


def metadata_mismatches(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"observed": observed.get(key), "expected": value}
        for key, value in expected.items()
        if observed.get(key) != value
    }


def validate_resume_metadata(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    mismatches = metadata_mismatches(observed, expected)
    if mismatches:
        raise ValueError(f"Incompatible G0 resume metadata: {mismatches}.")


def condition_registration_payload() -> dict[str, Any]:
    return {
        "protocol": G0_PROTOCOL,
        "num_layers": NUM_LAYERS,
        "suite_order": list(SUITE_ORDER),
        "conditions": [
            {
                **condition.to_dict(),
                "enabled_video_retrieval_layers": list(
                    condition.enabled_video_retrieval_layers(NUM_LAYERS)
                ),
                "condition_slot": GPU_BY_CONDITION[condition.name],
            }
            for condition in CONDITIONS
        ],
    }
