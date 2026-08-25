from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class DiagnosisCondition:
    name: str
    disabled_video_layers: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["disabled_video_layers"] = list(self.disabled_video_layers)
        return payload


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


def resolve_condition(diagnosis_cfg: Mapping[str, Any], num_layers: int) -> DiagnosisCondition:
    enabled = bool(diagnosis_cfg.get("enabled", False))
    mode = str(diagnosis_cfg.get("mode", "drop_video_kv"))
    if mode != "drop_video_kv":
        raise ValueError(f"Unsupported ASRE_DIAGNOSIS.mode={mode!r}; expected 'drop_video_kv'.")

    explicit = validate_disabled_layers(
        diagnosis_cfg.get("disabled_video_layers", ()),
        num_layers,
    )
    if not enabled:
        if explicit:
            raise ValueError(
                "ASRE_DIAGNOSIS.enabled=false requires disabled_video_layers=[]; "
                "otherwise the requested intervention would be silently ignored."
            )
        return DiagnosisCondition("baseline", ())

    conditions = build_conditions(num_layers)
    condition_index = diagnosis_cfg.get("condition_index")
    if condition_index is not None:
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
) -> dict[str, Any]:
    return {
        "git_commit_hash": git_commit(repo_root),
        "checkpoint_path": str(
            Path(os.path.expanduser(os.path.expandvars(checkpoint))).resolve()
        ),
        "checkpoint_name": Path(checkpoint).name,
        "dataset_stats_path": dataset_stats_path,
        "diagnosis_condition": condition.name,
        "disabled_video_layers": list(condition.disabled_video_layers),
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
