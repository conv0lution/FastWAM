"""Frozen condition, execution, and analysis definitions for ASRE Salvage A."""

from __future__ import annotations

from typing import Any

from experiments.asre_diagnosis.common import build_salvage_a_conditions


FEATURE_DIM = 3072
RANKS = (36, 97)
CONDITIONS = (
    "current_all",
    "wrong_all",
    "svd_r36",
    "actionaware_r36",
    "random_r36",
    "svd_r97",
    "actionaware_r97",
    "random_r97",
)
DISPLAY = {
    "current_all": "Current",
    "wrong_all": "Wrong",
    "svd_r36": "SVD-36",
    "actionaware_r36": "ActionAware-36",
    "random_r36": "Random-36",
    "svd_r97": "SVD-97",
    "actionaware_r97": "ActionAware-97",
    "random_r97": "Random-97",
}
WAVES = {1: (0, 1, 2, 3), 2: (4, 5, 6, 7)}
ANALYSIS_SCOPES = ("heldout", "calibration", "all")
PRIMARY_SCOPE = "heldout"
TASK_SUITE = "libero_spatial"
TASK_CONFIG = "libero_uncond_2cam224_1e-4"
TASK_IDS = tuple(range(10))
ONLINE_TRIALS_PER_TASK = 10
SMOKE_TRIALS = 2
SEED = 42
ACTION_HORIZON = 32
INFERENCE_STEPS = 10
REPLAN_STEPS = 10
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 4210


PRIMARY_COMPARISONS = (
    ("actionaware_r36_minus_svd_r36", "svd_r36", "actionaware_r36", 36, "svd"),
    (
        "actionaware_r36_minus_random_r36",
        "random_r36",
        "actionaware_r36",
        36,
        "random",
    ),
    (
        "actionaware_r36_minus_current_all",
        "current_all",
        "actionaware_r36",
        36,
        "current",
    ),
    ("actionaware_r97_minus_svd_r97", "svd_r97", "actionaware_r97", 97, "svd"),
    (
        "actionaware_r97_minus_random_r97",
        "random_r97",
        "actionaware_r97",
        97,
        "random",
    ),
    (
        "actionaware_r97_minus_current_all",
        "current_all",
        "actionaware_r97",
        97,
        "current",
    ),
)


PROVENANCE_STEMS = (
    "preflight_report",
    "machinery_report",
    "calibration_split_manifest",
    "state_selection_manifest",
    "differentiable_path_report",
    "subspace_basis_manifest",
    "subspace_diagnostics",
    "round4c_summary",
)


def condition_family_rank(condition: str) -> tuple[str | None, int | None]:
    """Return the frozen basis family/rank encoded by a condition name."""

    if condition in {"current_all", "wrong_all"}:
        return None, None
    for family in ("svd", "actionaware", "random"):
        prefix = f"{family}_r"
        if condition.startswith(prefix):
            rank = int(condition.removeprefix(prefix))
            if rank not in RANKS:
                break
            return family, rank
    raise ValueError(f"Unknown Salvage A condition: {condition!r}.")


def validate_frozen_condition_matrix() -> tuple[Any, ...]:
    """Fail closed if shared condition construction drifts from the registration."""

    conditions = tuple(build_salvage_a_conditions(30))
    if tuple(condition.name for condition in conditions) != CONDITIONS:
        raise ValueError("Shared Salvage A condition order drifted.")
    for condition in conditions:
        family, rank = condition_family_rank(condition.name)
        if condition.basis_kind != family or condition.subspace_rank != rank:
            raise ValueError(f"Shared Salvage A basis definition drifted: {condition.name}.")
        expected_replacement = () if condition.name == "current_all" else tuple(range(15, 30))
        if (
            condition.disabled_video_layers != tuple(range(15))
            or condition.replacement_video_layers != expected_replacement
        ):
            raise ValueError(f"Shared Salvage A layer schedule drifted: {condition.name}.")
    return conditions
