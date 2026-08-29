from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch


ROUND1_PROTOCOL = "round1_drop_groups"
ROUND2_PROTOCOL = "round2_keep_schedules"
ROUND3A_PROTOCOL = "round3a_late_factorial"
ROUND3B_PROTOCOL = "round3b_matched_kv_replacement"
G0_PROTOCOL = "g0_cross_suite_generalization"
ROUND4A_PROTOCOL = "round4a_matched_content_axis_screen"
ROUND4B_PROTOCOL = "round4b_low_rank_feature_subspace"
ROUND4C_PROTOCOL = "round4c_energy_controlled_action_sufficiency"
SALVAGE_A_PROTOCOL = "salvage_a_action_sensitive_subspace"
SALVAGE_B_PROTOCOL = "salvage_b_world_action_functional_dissociation"


@dataclass(frozen=True)
class DiagnosisCondition:
    name: str
    disabled_video_layers: tuple[int, ...]
    replacement_video_layers: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["disabled_video_layers"] = list(self.disabled_video_layers)
        payload["replacement_video_layers"] = list(self.replacement_video_layers)
        return payload

    def enabled_video_retrieval_layers(self, num_layers: int) -> tuple[int, ...]:
        disabled = set(self.disabled_video_layers)
        return tuple(layer for layer in range(num_layers) if layer not in disabled)


@dataclass(frozen=True)
class Round4ACondition(DiagnosisCondition):
    hybrid_axis: Optional[str] = None
    mask_seed: Optional[int] = None


@dataclass(frozen=True)
class Round4BCondition(DiagnosisCondition):
    basis_kind: Optional[str] = None
    subspace_rank: Optional[int] = None


def build_conditions(num_layers: int) -> list[DiagnosisCondition]:
    """Build baseline, six contiguous groups, and drop-all dynamically."""
    if num_layers < 6:
        raise ValueError(
            f"`num_layers` must be at least 6 to form six non-empty groups, got {num_layers}."
        )

    quotient, remainder = divmod(num_layers, 6)
    groups: list[tuple[int, ...]] = []
    start = 0
    for group_idx in range(6):
        size = quotient + (1 if group_idx < remainder else 0)
        stop = start + size
        if size > 0:
            groups.append(tuple(range(start, stop)))
        else:
            groups.append(())
        start = stop

    conditions = [DiagnosisCondition("baseline", ())]
    for group_idx, layers in enumerate(groups):
        if layers:
            name = f"drop_{layers[0]:02d}_{layers[-1]:02d}"
        else:
            name = f"drop_empty_group_{group_idx}"
        conditions.append(DiagnosisCondition(name, layers))
    conditions.append(DiagnosisCondition("drop_all", tuple(range(num_layers))))
    return conditions


def build_round2_conditions(num_layers: int) -> list[DiagnosisCondition]:
    """Build the pre-registered ASRE Round-2 keep schedules for a 30-layer model."""
    if num_layers != 30:
        raise ValueError(
            "ASRE Round 2 is pre-registered for exactly 30 action layers; "
            f"the selected model exposes {num_layers}."
        )

    enabled_schedules = (
        ("baseline_round2", tuple(range(0, 30))),
        ("keep_15_29", tuple(range(15, 30))),
        ("keep_20_29", tuple(range(20, 30))),
        ("keep_25_29", tuple(range(25, 30))),
        ("keep_00_14", tuple(range(0, 15))),
        ("keep_00_19", tuple(range(0, 20))),
        ("keep_15_19", tuple(range(15, 20))),
        ("keep_15_19_25_29", tuple(range(15, 20)) + tuple(range(25, 30))),
    )
    return [
        DiagnosisCondition(
            name=name,
            disabled_video_layers=enabled_to_disabled_layers(enabled, num_layers),
        )
        for name, enabled in enabled_schedules
    ]


def build_round3a_conditions(num_layers: int) -> list[DiagnosisCondition]:
    """Build the three missing cells in the pre-registered late-half factorial."""
    if num_layers != 30:
        raise ValueError(
            "ASRE Round 3A is pre-registered for exactly 30 action layers; "
            f"the selected model exposes {num_layers}."
        )

    enabled_schedules = (
        ("keep_none_late", ()),
        ("keep_20_24", tuple(range(20, 25))),
        ("keep_15_24", tuple(range(15, 25))),
    )
    return [
        DiagnosisCondition(
            name=name,
            disabled_video_layers=enabled_to_disabled_layers(enabled, num_layers),
        )
        for name, enabled in enabled_schedules
    ]


def build_round3b_conditions(num_layers: int) -> list[DiagnosisCondition]:
    """Build the frozen three-arm matched-shape K/V replacement control."""
    if num_layers != 30:
        raise ValueError(
            "ASRE Round 3B is pre-registered for exactly 30 action layers; "
            f"the selected model exposes {num_layers}."
        )

    early_layers = tuple(range(15))
    late_layers = tuple(range(15, 30))
    return [
        DiagnosisCondition("late_current_correct", early_layers),
        DiagnosisCondition(
            "late_wrong_scene",
            early_layers,
            replacement_video_layers=late_layers,
        ),
        DiagnosisCondition("late_no_video", tuple(range(30))),
    ]


def build_g0_conditions(num_layers: int) -> list[DiagnosisCondition]:
    """Build the four pre-registered G0 cross-suite generalization conditions."""
    if num_layers != 30:
        raise ValueError(
            "ASRE G0 is pre-registered for exactly 30 action layers; "
            f"the selected model exposes {num_layers}."
        )

    early_half = tuple(range(0, 15))
    late_third = tuple(range(20, 30))
    late_half = tuple(range(15, 30))
    return [
        DiagnosisCondition("full_current", ()),
        DiagnosisCondition("late_current_15_29", early_half),
        DiagnosisCondition("early_current_00_19", late_third),
        DiagnosisCondition(
            "late_wrong_scene_15_29",
            early_half,
            replacement_video_layers=late_half,
        ),
    ]


def build_round4a_conditions(num_layers: int) -> list[Round4ACondition]:
    """Build the frozen eight-arm Stage-2 Round-4A screening matrix."""
    if num_layers != 30:
        raise ValueError(
            "ASRE Round 4A is pre-registered for exactly 30 action layers; "
            f"the selected model exposes {num_layers}."
        )
    early_layers = tuple(range(0, 15))
    late_layers = tuple(range(15, 30))
    conditions = [
        Round4ACondition("current_all", early_layers),
        Round4ACondition(
            "wrong_all", early_layers, replacement_video_layers=late_layers
        ),
    ]
    conditions.extend(
        Round4ACondition(
            f"head50_seed{seed}",
            early_layers,
            replacement_video_layers=late_layers,
            hybrid_axis="head",
            mask_seed=seed,
        )
        for seed in (1, 2, 3)
    )
    conditions.extend(
        Round4ACondition(
            f"token50_seed{seed}",
            early_layers,
            replacement_video_layers=late_layers,
            hybrid_axis="token",
            mask_seed=seed,
        )
        for seed in (1, 2, 3)
    )
    return conditions


def build_round4b_conditions(num_layers: int) -> list[Round4BCondition]:
    """Build the frozen eight-arm Stage-2 Round-4B subspace matrix."""
    if num_layers != 30:
        raise ValueError(
            "ASRE Round 4B is pre-registered for exactly 30 action layers; "
            f"the selected model exposes {num_layers}."
        )
    early = tuple(range(15))
    late = tuple(range(15, 30))
    return [
        Round4BCondition("current_all", early),
        Round4BCondition("wrong_all", early, replacement_video_layers=late),
        Round4BCondition("svd_r256", early, late, "svd", 256),
        Round4BCondition("random_r256", early, late, "random", 256),
        Round4BCondition("svd_r768", early, late, "svd", 768),
        Round4BCondition("random_r768", early, late, "random", 768),
        Round4BCondition("svd_r1536", early, late, "svd", 1536),
        Round4BCondition("random_r1536", early, late, "random", 1536),
    ]


def build_round4c_conditions(num_layers: int) -> list[Round4BCondition]:
    """Build the frozen five-arm Stage-2 Round-4C energy-sufficiency matrix."""
    if num_layers != 30:
        raise ValueError(
            "ASRE Round 4C is pre-registered for exactly 30 action layers; "
            f"the selected model exposes {num_layers}."
        )
    early = tuple(range(15))
    late = tuple(range(15, 30))
    return [
        Round4BCondition("current_all", early),
        Round4BCondition("wrong_all", early, replacement_video_layers=late),
        Round4BCondition("svd_r36", early, late, "svd", 36),
        Round4BCondition("svd_r97", early, late, "svd", 97),
        Round4BCondition("svd_r170", early, late, "svd", 170),
    ]


def build_salvage_a_conditions(num_layers: int) -> list[Round4BCondition]:
    """Build the frozen eight-arm action-sensitive salvage matrix."""
    if num_layers != 30:
        raise ValueError(
            "ASRE Salvage A is pre-registered for exactly 30 action layers; "
            f"the selected model exposes {num_layers}."
        )
    early = tuple(range(15))
    late = tuple(range(15, 30))
    return [
        Round4BCondition("current_all", early),
        Round4BCondition("wrong_all", early, replacement_video_layers=late),
        Round4BCondition("svd_r36", early, late, "svd", 36),
        Round4BCondition("actionaware_r36", early, late, "actionaware", 36),
        Round4BCondition("random_r36", early, late, "random", 36),
        Round4BCondition("svd_r97", early, late, "svd", 97),
        Round4BCondition("actionaware_r97", early, late, "actionaware", 97),
        Round4BCondition("random_r97", early, late, "random", 97),
    ]


def get_num_model_layers(model: torch.nn.Module) -> int:
    mot = getattr(model, "mot", None)
    num_layers = getattr(mot, "num_layers", None)
    if num_layers is None:
        raise ValueError(f"{type(model).__name__} does not expose model.mot.num_layers.")
    return int(num_layers)


def validate_disabled_layers(
    disabled_video_layers: Optional[Sequence[int]],
    num_layers: int,
) -> tuple[int, ...]:
    if disabled_video_layers is None:
        return ()
    requested = list(disabled_video_layers)
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in requested):
        raise TypeError("`disabled_video_layers` must contain only integer layer indices.")
    if len(set(requested)) != len(requested):
        raise ValueError("`disabled_video_layers` must not contain duplicate indices.")
    invalid = [layer for layer in requested if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"Invalid disabled video layers {invalid}; valid range is [0, {num_layers - 1}]."
        )
    return tuple(sorted(requested))


def validate_enabled_layers(
    enabled_video_retrieval_layers: Optional[Sequence[int]],
    num_layers: int,
) -> Optional[tuple[int, ...]]:
    """Validate a human-facing keep schedule without treating null as keep-none."""
    if enabled_video_retrieval_layers is None:
        return None
    requested = list(enabled_video_retrieval_layers)
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in requested):
        raise TypeError(
            "`enabled_video_retrieval_layers` must contain only integer layer indices."
        )
    if len(set(requested)) != len(requested):
        raise ValueError("`enabled_video_retrieval_layers` must not contain duplicates.")
    invalid = [layer for layer in requested if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"Invalid enabled video retrieval layers {invalid}; valid range is "
            f"[0, {num_layers - 1}]."
        )
    return tuple(sorted(requested))


def validate_replacement_layers(
    replacement_video_layers: Optional[Sequence[int]],
    num_layers: int,
) -> tuple[int, ...]:
    """Validate layer indices whose current video cache is replaced by a donor."""
    if replacement_video_layers is None:
        return ()
    requested = list(replacement_video_layers)
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in requested):
        raise TypeError("`replacement_video_layers` must contain only integer indices.")
    if len(set(requested)) != len(requested):
        raise ValueError("`replacement_video_layers` must not contain duplicate indices.")
    invalid = [layer for layer in requested if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"Invalid replacement video layers {invalid}; valid range is "
            f"[0, {num_layers - 1}]."
        )
    return tuple(sorted(requested))


def enabled_to_disabled_layers(
    enabled_video_retrieval_layers: Sequence[int],
    num_layers: int,
) -> tuple[int, ...]:
    enabled = validate_enabled_layers(enabled_video_retrieval_layers, num_layers)
    assert enabled is not None
    enabled_set = set(enabled)
    return tuple(layer for layer in range(num_layers) if layer not in enabled_set)


def resolve_condition(diagnosis_cfg: Mapping[str, Any], num_layers: int) -> DiagnosisCondition:
    enabled = bool(diagnosis_cfg.get("enabled", False))
    mode = str(diagnosis_cfg.get("mode", "drop_video_kv"))
    if mode not in {"drop_video_kv", "replace_video_kv"}:
        raise ValueError(
            f"Unsupported ASRE_DIAGNOSIS.mode={mode!r}; expected 'drop_video_kv' "
            "or 'replace_video_kv'."
        )

    explicit = validate_disabled_layers(
        diagnosis_cfg.get("disabled_video_layers", ()),
        num_layers,
    )
    enabled_layers = validate_enabled_layers(
        diagnosis_cfg.get("enabled_video_retrieval_layers"),
        num_layers,
    )
    replacement_layers = validate_replacement_layers(
        diagnosis_cfg.get("replacement_video_layers", ()),
        num_layers,
    )
    compiled_disabled = (
        None
        if enabled_layers is None
        else enabled_to_disabled_layers(enabled_layers, num_layers)
    )
    if compiled_disabled is not None and explicit and explicit != compiled_disabled:
        raise ValueError(
            "enabled_video_retrieval_layers and disabled_video_layers are not exact "
            f"complements: enabled={list(enabled_layers)}, disabled={list(explicit)}."
        )
    if compiled_disabled is not None:
        explicit = compiled_disabled
    overlap = sorted(set(explicit) & set(replacement_layers))
    if overlap:
        raise ValueError(
            "replacement_video_layers must be disjoint from disabled_video_layers; "
            f"overlap={overlap}."
        )
    if not enabled:
        if explicit or enabled_layers is not None or replacement_layers:
            raise ValueError(
                "ASRE_DIAGNOSIS.enabled=false requires both disabled_video_layers=[] "
                "and replacement_video_layers=[], with "
                "enabled_video_retrieval_layers=null; "
                "otherwise the requested intervention would be silently ignored."
            )
        return DiagnosisCondition("baseline", ())

    protocol = str(diagnosis_cfg.get("protocol", ROUND1_PROTOCOL))
    if protocol not in {
        ROUND1_PROTOCOL,
        ROUND2_PROTOCOL,
        ROUND3A_PROTOCOL,
        ROUND3B_PROTOCOL,
        G0_PROTOCOL,
        ROUND4A_PROTOCOL,
        ROUND4B_PROTOCOL,
        ROUND4C_PROTOCOL,
        SALVAGE_A_PROTOCOL,
    }:
        raise ValueError(
            f"Unsupported ASRE_DIAGNOSIS.protocol={protocol!r}; expected "
            f"one of {ROUND1_PROTOCOL!r}, {ROUND2_PROTOCOL!r}, or "
            f"{ROUND3A_PROTOCOL!r}, {ROUND3B_PROTOCOL!r}, {G0_PROTOCOL!r}, "
            f"or {ROUND4A_PROTOCOL!r}, {ROUND4B_PROTOCOL!r}, "
            f"{ROUND4C_PROTOCOL!r}, {SALVAGE_A_PROTOCOL!r}."
        )

    replacement_protocols = {
        ROUND3B_PROTOCOL,
        G0_PROTOCOL,
        ROUND4A_PROTOCOL,
        ROUND4B_PROTOCOL,
        ROUND4C_PROTOCOL,
        SALVAGE_A_PROTOCOL,
    }
    if protocol in replacement_protocols and mode != "replace_video_kv":
        raise ValueError(
            f"ASRE protocol {protocol!r} requires "
            "ASRE_DIAGNOSIS.mode='replace_video_kv'."
        )
    if protocol not in replacement_protocols and mode != "drop_video_kv":
        raise ValueError(
            f"{protocol} requires ASRE_DIAGNOSIS.mode='drop_video_kv'."
        )

    if protocol in {
        ROUND2_PROTOCOL,
        ROUND3A_PROTOCOL,
        ROUND3B_PROTOCOL,
        G0_PROTOCOL,
        ROUND4A_PROTOCOL,
        ROUND4B_PROTOCOL,
        ROUND4C_PROTOCOL,
        SALVAGE_A_PROTOCOL,
    }:
        is_round2 = protocol == ROUND2_PROTOCOL
        is_round3a = protocol == ROUND3A_PROTOCOL
        is_round3b = protocol == ROUND3B_PROTOCOL
        is_round4a = protocol == ROUND4A_PROTOCOL
        is_round4b = protocol == ROUND4B_PROTOCOL
        is_round4c = protocol == ROUND4C_PROTOCOL
        is_salvage_a = protocol == SALVAGE_A_PROTOCOL
        round_label = (
            "Round-2"
            if is_round2
            else "Round-3A"
            if is_round3a
            else "Round-3B"
            if is_round3b
            else "Round-4A"
            if is_round4a
            else "Round-4B"
            if is_round4b
            else "Round-4C"
            if is_round4c
            else "Salvage-A"
            if is_salvage_a
            else "G0"
        )
        conditions = (
            build_round2_conditions(num_layers)
            if is_round2
            else build_round3a_conditions(num_layers)
            if is_round3a
            else build_round3b_conditions(num_layers)
            if is_round3b
            else build_round4a_conditions(num_layers)
            if is_round4a
            else build_round4b_conditions(num_layers)
            if is_round4b
            else build_round4c_conditions(num_layers)
            if is_round4c
            else build_salvage_a_conditions(num_layers)
            if is_salvage_a
            else build_g0_conditions(num_layers)
        )
        condition_index = diagnosis_cfg.get("condition_index")
        if condition_index is not None:
            condition_index = int(condition_index)
            if condition_index < 0 or condition_index >= len(conditions):
                raise ValueError(
                    f"{round_label} condition_index must be in "
                    f"[0, {len(conditions) - 1}], "
                    f"got {condition_index}."
                )
            selected = conditions[condition_index]
        else:
            condition_name = str(diagnosis_cfg.get("condition_name", ""))
            by_name = {condition.name: condition for condition in conditions}
            if condition_name not in by_name:
                raise ValueError(
                    f"Unknown ASRE {round_label} condition {condition_name!r}; "
                    "expected one of "
                    f"{list(by_name)}."
                )
            selected = by_name[condition_name]

        selected_enabled = selected.enabled_video_retrieval_layers(num_layers)
        if enabled_layers is None:
            raise ValueError(
                f"ASRE {round_label} conditions require an explicit "
                "enabled_video_retrieval_layers keep schedule."
            )
        if enabled_layers != selected_enabled:
            raise ValueError(
                f"Configured enabled layers {list(enabled_layers)} disagree with "
                f"{selected.name}: {list(selected_enabled)}."
            )
        if explicit != selected.disabled_video_layers:
            raise ValueError(
                f"Configured disabled layers {list(explicit)} disagree with "
                f"{selected.name}: {list(selected.disabled_video_layers)}."
            )
        if replacement_layers != selected.replacement_video_layers:
            raise ValueError(
                f"Configured replacement layers {list(replacement_layers)} disagree "
                f"with {selected.name}: {list(selected.replacement_video_layers)}."
            )
        if is_round4a:
            assert isinstance(selected, Round4ACondition)
            configured_axis = diagnosis_cfg.get("hybrid_axis")
            if configured_axis in {"", "none", "null"}:
                configured_axis = None
            if configured_axis != selected.hybrid_axis:
                raise ValueError(
                    f"Configured hybrid_axis={configured_axis!r} disagrees with "
                    f"{selected.name}: {selected.hybrid_axis!r}."
                )
            configured_seed = diagnosis_cfg.get("hybrid_mask_seed")
            if configured_seed is not None:
                configured_seed = int(configured_seed)
            if configured_seed != selected.mask_seed:
                raise ValueError(
                    f"Configured hybrid_mask_seed={configured_seed!r} disagrees with "
                    f"{selected.name}: {selected.mask_seed!r}."
                )
        if is_round4b or is_round4c or is_salvage_a:
            assert isinstance(selected, Round4BCondition)
            configured_kind = diagnosis_cfg.get("subspace_basis_kind")
            if configured_kind in {"", "none", "null"}:
                configured_kind = None
            if configured_kind != selected.basis_kind:
                raise ValueError(
                    f"Configured subspace_basis_kind={configured_kind!r} disagrees "
                    f"with {selected.name}: {selected.basis_kind!r}."
                )
            configured_rank = diagnosis_cfg.get("subspace_rank")
            if configured_rank is not None:
                configured_rank = int(configured_rank)
            if configured_rank != selected.subspace_rank:
                raise ValueError(
                    f"Configured subspace_rank={configured_rank!r} disagrees with "
                    f"{selected.name}: {selected.subspace_rank!r}."
                )
        return selected

    if replacement_layers:
        raise ValueError("Round-1 conditions do not support replacement_video_layers.")

    conditions = build_conditions(num_layers)
    condition_index = diagnosis_cfg.get("condition_index")
    if condition_index is not None:
        if enabled_layers is not None:
            raise ValueError(
                "condition_index selects the Round-1 drop matrix and cannot be combined "
                "with enabled_video_retrieval_layers."
            )
        condition_index = int(condition_index)
        if condition_index < 0 or condition_index >= len(conditions):
            raise ValueError(
                f"condition_index must be in [0, {len(conditions) - 1}], got {condition_index}."
            )
        selected = conditions[condition_index]
        if explicit and explicit != selected.disabled_video_layers:
            raise ValueError(
                f"Explicit disabled layers {list(explicit)} disagree with {selected.name}: "
                f"{list(selected.disabled_video_layers)}."
            )
        return selected

    condition_name = str(diagnosis_cfg.get("condition_name", "baseline"))
    by_name = {condition.name: condition for condition in conditions}
    if condition_name in by_name:
        selected = by_name[condition_name]
        if explicit and explicit != selected.disabled_video_layers:
            raise ValueError(
                f"Explicit disabled layers {list(explicit)} disagree with {selected.name}: "
                f"{list(selected.disabled_video_layers)}."
            )
        return selected
    return DiagnosisCondition(condition_name, explicit)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def gpu_model_name() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(torch.cuda.current_device())


def build_run_metadata(
    *,
    repo_root: Path,
    checkpoint: str,
    dataset_stats_path: Optional[str],
    condition: DiagnosisCondition,
    num_layers: int,
    task_suite: str,
    task_ids: Sequence[int],
    seed: Optional[int],
    num_trials: int,
    action_horizon: int,
    num_inference_steps: int,
    replan_steps: int,
    start_timestamp: str,
    condition_protocol: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    dataset_stats_sha256: Optional[str] = None,
    state_bank_manifest_path: Optional[str] = None,
    state_bank_manifest_sha256: Optional[str] = None,
    valid_state_bank_manifest_path: Optional[str] = None,
    valid_state_bank_manifest_sha256: Optional[str] = None,
    prompt_context_cache_path: Optional[str] = None,
    prompt_context_cache_sha256: Optional[str] = None,
    config_sha256: Optional[str] = None,
) -> dict[str, Any]:
    metadata = {
        "git_commit_hash": git_commit(repo_root),
        "checkpoint_path": str(
            Path(os.path.expanduser(os.path.expandvars(checkpoint))).resolve()
        ),
        "checkpoint_name": Path(checkpoint).name,
        "dataset_stats_path": dataset_stats_path,
        "diagnosis_condition": condition.name,
        "condition_protocol": condition_protocol,
        "enabled_video_retrieval_layers": list(
            condition.enabled_video_retrieval_layers(num_layers)
        ),
        "disabled_video_layers": list(condition.disabled_video_layers),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "num_model_layers": int(num_layers),
        "task_suite": task_suite,
        "task_ids": [int(task_id) for task_id in task_ids],
        "seed": None if seed is None else int(seed),
        "number_of_trials": int(num_trials),
        "action_horizon": int(action_horizon),
        "number_of_inference_steps": int(num_inference_steps),
        "replan_steps": int(replan_steps),
        "gpu_model": gpu_model_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "start_timestamp": start_timestamp,
        "end_timestamp": None,
    }
    optional_fields = {
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_stats_sha256": dataset_stats_sha256,
        "state_bank_manifest_path": state_bank_manifest_path,
        "state_bank_manifest_sha256": state_bank_manifest_sha256,
        "valid_state_bank_manifest_path": valid_state_bank_manifest_path,
        "valid_state_bank_manifest_sha256": valid_state_bank_manifest_sha256,
        "prompt_context_cache_path": prompt_context_cache_path,
        "prompt_context_cache_sha256": prompt_context_cache_sha256,
        "config_sha256": config_sha256,
    }
    metadata.update({key: value for key, value in optional_fields.items() if value is not None})
    return metadata


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records
