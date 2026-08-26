"""Aggregate the complete paired ASRE late-half 2^3 factorial.

Five cells are read in place from the frozen Round-2 result tree and three
cells are read from the independent Round-3A tree.  No source artifact is
copied, rewritten, or silently intersected with another condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND2_PROTOCOL,
    ROUND3A_PROTOCOL,
    atomic_write_json,
    build_round2_conditions,
    build_round3a_conditions,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round2.aggregate_results import (  # noqa: E402
    episode_cluster_bootstrap_ci,
)
from experiments.asre_diagnosis.round2.metrics import (  # noqa: E402
    ACTION_DIMENSION_LABEL_SOURCE,
    CONTINUOUS_ACTION_DIMENSIONS,
)
from experiments.asre_diagnosis.round3a.factorial import (  # noqa: E402
    CELL_ORDER,
    FACTORIAL_CELLS,
    FactorialCell,
)
from experiments.asre_diagnosis.round3a.factorial_stats import (  # noqa: E402
    analyze_factorial,
    simple_effect_transitions,
)


TASK_SUITE = "libero_spatial"
EXPECTED_TASKS = 10
EXPECTED_TRIALS = 10
EXPECTED_EPISODES = EXPECTED_TASKS * EXPECTED_TRIALS
EXPECTED_OFFLINE_SAMPLES = 499
EXPECTED_REPLAN_STAGES = tuple(range(5))
NUM_LAYERS = 30

SCALAR_OFFLINE_METRICS = (
    "executed_prefix_norm_rms",
    "norm_rms_h0",
    "norm_rms_h0_h1",
    "full_chunk_norm_rms_0_31",
    "round1_raw_output_full_chunk_rms",
    "executed_prefix_cosine_similarity",
    "executed_prefix_gripper_flip_rate",
    "full_horizon_gripper_flip_rate",
    "translation_norm_rms",
    "rotation_norm_rms",
)
DIMENSION_OFFLINE_METRIC = "executed_prefix_norm_rms_by_dimension"
EpisodeKey = tuple[str, int, int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the complete ASRE Round-3A late-half factorial."
    )
    parser.add_argument("--round2-online-root", type=Path, required=True)
    parser.add_argument("--round2-offline-root", type=Path, required=True)
    parser.add_argument("--round3a-online-root", type=Path, required=True)
    parser.add_argument("--round3a-offline-root", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--sign-flip-samples", type=int, default=100_000)
    parser.add_argument("--sign-flip-seed", type=int, default=0)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {path}.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Record {line_number} is not an object.")
                records.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Cannot read JSONL records at {path}: {exc}") from exc
    return records


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer CSV columns for empty output {path}.")
    columns = list(fieldnames or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _sha256_file_set(paths: Sequence[Path], *, relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((candidate.resolve() for candidate in paths), key=str):
        try:
            relative = path.relative_to(relative_to.resolve())
        except ValueError as exc:
            raise ValueError(f"Hash input {path} is outside {relative_to}.") from exc
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _finite_float(value: Any, *, context: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not numeric: {value!r}.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{context} is not finite: {value!r}.")
    return numeric


def _stable_seed(base_seed: int, *parts: object) -> int:
    label = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _json_for_csv(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _parse_simple_context(value: Any, *, factor: str) -> dict[str, int]:
    """Parse the canonical ``A=0,B=1`` context emitted by factorial_stats."""

    manipulated = str(factor).upper()
    if manipulated not in {"A", "B", "C"}:
        raise ValueError(f"Invalid manipulated factor for a simple effect: {factor!r}.")
    if not isinstance(value, str):
        raise TypeError(f"Simple-effect context must be a string, got {value!r}.")

    context: dict[str, int] = {}
    for component in value.split(","):
        match = re.fullmatch(r"([ABC])=([01])", component.strip())
        if match is None:
            raise ValueError(f"Invalid simple-effect context component: {component!r}.")
        name, bit = match.groups()
        if name in context:
            raise ValueError(f"Duplicate factor {name!r} in simple-effect context {value!r}.")
        context[name] = int(bit)

    expected = {"A", "B", "C"} - {manipulated}
    if set(context) != expected:
        raise ValueError(
            f"Simple-effect context {value!r} for factor {manipulated} must fix "
            f"exactly {sorted(expected)}."
        )
    return context


def _format_layer_ranges(layers: Sequence[int]) -> str:
    values = sorted(int(value) for value in layers)
    if not values:
        return "none"
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _cell_roots(
    cell: FactorialCell,
    *,
    round2_online_root: Path,
    round2_offline_root: Path,
    round3a_online_root: Path,
    round3a_offline_root: Path,
) -> tuple[Path, Path]:
    if cell.protocol == ROUND2_PROTOCOL:
        return round2_online_root, round2_offline_root
    if cell.protocol == ROUND3A_PROTOCOL:
        return round3a_online_root, round3a_offline_root
    raise AssertionError(f"Unsupported factorial source protocol {cell.protocol!r}.")


def _validate_launcher_summary(
    root: Path,
    *,
    protocol: str,
    expected_conditions: Sequence[str],
) -> dict[str, Any]:
    path = root / "launcher_summary.json"
    payload = _read_json(path)
    expected = {
        "protocol": protocol,
        "mode": "full",
        "all_succeeded": True,
        "interrupted": False,
        "task_ids": list(range(EXPECTED_TASKS)),
        "num_trials": EXPECTED_TRIALS,
        "seed": 42,
        "expected_conditions": list(expected_conditions),
    }
    mismatches = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    states = payload.get("condition_states")
    if not isinstance(states, dict) or any(
        not isinstance(states.get(name), dict)
        or states[name].get("state") != "complete"
        for name in expected_conditions
    ):
        mismatches["condition_states"] = {
            "observed": states,
            "expected": "all expected conditions complete",
        }
    if mismatches:
        raise ValueError(
            f"Full launcher summary is incompatible at {path}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    return payload


def _validate_online_metadata(
    path: Path,
    *,
    cell: FactorialCell,
    valid_manifest_path: Path,
    valid_manifest_sha256: str,
) -> dict[str, Any]:
    metadata = _read_json(path)
    required = {
        "git_commit_hash",
        "checkpoint_path",
        "checkpoint_sha256",
        "dataset_stats_path",
        "dataset_stats_sha256",
        "diagnosis_condition",
        "condition_protocol",
        "enabled_video_retrieval_layers",
        "disabled_video_layers",
        "num_model_layers",
        "task_suite",
        "task_ids",
        "seed",
        "number_of_trials",
        "action_horizon",
        "number_of_inference_steps",
        "replan_steps",
        "checkpoint_sha256",
        "state_bank_manifest_path",
        "state_bank_manifest_sha256",
        "valid_state_bank_manifest_path",
        "valid_state_bank_manifest_sha256",
        "prompt_context_cache_path",
        "prompt_context_cache_sha256",
        "config_sha256",
        "compile_action_infer",
        "binarize_gripper",
        "sigma_shift",
        "rand_device",
        "text_conditioning_source",
        "prompt_template",
        "environment_seed",
        "action_inference_seed",
        "action_noise_seed",
        "gpu_model",
        "torch_version",
        "cuda_version",
        "condition_config",
        "start_timestamp",
        "end_timestamp",
        "status",
    }
    nullable = {"sigma_shift"}
    missing = sorted(
        key
        for key in required
        if key not in metadata or (metadata[key] is None and key not in nullable)
    )
    if missing:
        raise ValueError(f"Online metadata for cell {cell.code} is missing {missing}: {path}")
    expected = {
        "diagnosis_condition": cell.condition.name,
        "condition_protocol": cell.protocol,
        "enabled_video_retrieval_layers": list(cell.enabled_layers),
        "disabled_video_layers": list(cell.condition.disabled_video_layers),
        "num_model_layers": NUM_LAYERS,
        "task_suite": TASK_SUITE,
        "task_ids": list(range(EXPECTED_TASKS)),
        "seed": 42,
        "number_of_trials": EXPECTED_TRIALS,
        "action_horizon": 32,
        "number_of_inference_steps": 10,
        "replan_steps": 10,
        "valid_state_bank_manifest_path": str(valid_manifest_path),
        "valid_state_bank_manifest_sha256": valid_manifest_sha256,
        "status": "completed",
    }
    mismatches = {
        key: {"observed": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Online metadata mismatch for cell {cell.code}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    for hash_key in (
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "state_bank_manifest_sha256",
        "valid_state_bank_manifest_sha256",
        "prompt_context_cache_sha256",
        "config_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(metadata[hash_key]).lower()) is None:
            raise ValueError(f"Invalid {hash_key} for cell {cell.code}.")
    if re.fullmatch(r"[0-9a-f]{40}", str(metadata["git_commit_hash"]).lower()) is None:
        raise ValueError(f"Invalid git_commit_hash for cell {cell.code}.")
    if not str(metadata["end_timestamp"]).strip():
        raise ValueError(f"Online metadata for cell {cell.code} has no end timestamp.")
    condition_config = metadata["condition_config"]
    if not isinstance(condition_config, dict):
        raise TypeError(f"condition_config for cell {cell.code} must be an object.")
    expected_config = {
        "protocol": cell.protocol,
        "condition_name": cell.condition.name,
        "enabled_video_retrieval_layers": list(cell.enabled_layers),
        "disabled_video_layers": list(cell.condition.disabled_video_layers),
    }
    config_mismatches = {
        key: {"observed": condition_config.get(key), "expected": value}
        for key, value in expected_config.items()
        if condition_config.get(key) != value
    }
    if config_mismatches:
        raise ValueError(
            f"Online condition_config mismatch for cell {cell.code}: "
            f"{json.dumps(config_mismatches, sort_keys=True)}"
        )
    return metadata


ONLINE_COMPATIBILITY_FIELDS = (
    "checkpoint_path",
    "checkpoint_sha256",
    "dataset_stats_path",
    "dataset_stats_sha256",
    "state_bank_manifest_path",
    "state_bank_manifest_sha256",
    "valid_state_bank_manifest_path",
    "valid_state_bank_manifest_sha256",
    "prompt_context_cache_path",
    "prompt_context_cache_sha256",
    "num_model_layers",
    "task_suite",
    "task_ids",
    "seed",
    "number_of_trials",
    "action_horizon",
    "number_of_inference_steps",
    "replan_steps",
    "compile_action_infer",
    "binarize_gripper",
    "sigma_shift",
    "rand_device",
    "text_conditioning_source",
    "prompt_template",
    "environment_seed",
    "action_inference_seed",
    "action_noise_seed",
    "gpu_model",
    "torch_version",
    "cuda_version",
)


def _validate_cross_cell_online(metadata_by_code: Mapping[str, Mapping[str, Any]]) -> None:
    reference = metadata_by_code["111"]
    for code in CELL_ORDER:
        metadata = metadata_by_code[code]
        mismatches = {
            key: {"reference_y111": reference.get(key), "cell": metadata.get(key)}
            for key in ONLINE_COMPATIBILITY_FIELDS
            if metadata.get(key) != reference.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Online provenance/runtime for cell {code} is incompatible with y111: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
    for protocol in (ROUND2_PROTOCOL, ROUND3A_PROTOCOL):
        commits = {
            str(metadata_by_code[cell.code]["git_commit_hash"])
            for cell in FACTORIAL_CELLS
            if cell.protocol == protocol
        }
        if len(commits) != 1:
            raise ValueError(
                f"Cells from protocol {protocol!r} have inconsistent Git commits: "
                f"{sorted(commits)}"
            )


def _load_task_outcomes(
    condition_dir: Path,
    *,
    cell: FactorialCell,
) -> tuple[dict[EpisodeKey, int], dict[int, list[int]], dict[int, str], list[Path]]:
    paths = sorted(condition_dir.glob("**/gpu*_task*_results.json"))
    if len(paths) != EXPECTED_TASKS:
        raise ValueError(
            f"Cell {cell.code} has {len(paths)} task result files; "
            f"expected {EXPECTED_TASKS}."
        )
    outcomes: dict[EpisodeKey, int] = {}
    by_task: dict[int, list[int]] = {}
    descriptions: dict[int, str] = {}
    for path in paths:
        result = _read_json(path)
        expected = {
            "task_suite": TASK_SUITE,
            "diagnosis_condition": cell.condition.name,
            "condition_protocol": cell.protocol,
            "enabled_video_retrieval_layers": list(cell.enabled_layers),
            "disabled_video_layers": list(cell.condition.disabled_video_layers),
            "total_episodes": EXPECTED_TRIALS,
        }
        mismatches = {
            key: {"observed": result.get(key), "expected": value}
            for key, value in expected.items()
            if result.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Task result mismatch for cell {cell.code} at {path}: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        task_id = int(result.get("task_id", -1))
        if task_id not in range(EXPECTED_TASKS) or task_id in by_task:
            raise ValueError(f"Invalid or duplicate task_id={task_id} for cell {cell.code}.")
        successes = [int(value) for value in result.get("success_episodes", [])]
        failures = [int(value) for value in result.get("failure_episodes", [])]
        success_set, failure_set = set(successes), set(failures)
        expected_trials = set(range(EXPECTED_TRIALS))
        if (
            len(successes) != len(success_set)
            or len(failures) != len(failure_set)
            or success_set & failure_set
            or success_set | failure_set != expected_trials
            or int(result.get("successes", -1)) != len(success_set)
        ):
            raise ValueError(f"Incomplete paired trial outcomes in {path}.")
        description = str(result.get("task_description", "")).strip()
        if not description:
            raise ValueError(f"Missing task description in {path}.")
        values = [int(trial_id in success_set) for trial_id in range(EXPECTED_TRIALS)]
        by_task[task_id] = values
        descriptions[task_id] = description
        for trial_id, outcome in enumerate(values):
            key = (TASK_SUITE, task_id, trial_id)
            if key in outcomes:
                raise ValueError(f"Duplicate paired episode key {key} for cell {cell.code}.")
            outcomes[key] = outcome
    if set(by_task) != set(range(EXPECTED_TASKS)) or len(outcomes) != EXPECTED_EPISODES:
        raise ValueError(f"Cell {cell.code} does not contain the complete task/trial grid.")
    return outcomes, by_task, descriptions, paths


def load_online_factorial(
    *,
    round2_online_root: Path,
    round3a_online_root: Path,
    valid_manifest_path: Path,
    valid_manifest_sha256: str,
) -> tuple[
    list[EpisodeKey],
    np.ndarray,
    dict[str, dict[int, list[int]]],
    dict[int, str],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    round2_expected = [condition.name for condition in build_round2_conditions(NUM_LAYERS)]
    round3a_expected = [condition.name for condition in build_round3a_conditions(NUM_LAYERS)]
    _validate_launcher_summary(
        round2_online_root,
        protocol=ROUND2_PROTOCOL,
        expected_conditions=round2_expected,
    )
    _validate_launcher_summary(
        round3a_online_root,
        protocol=ROUND3A_PROTOCOL,
        expected_conditions=round3a_expected,
    )

    outcomes_by_code: dict[str, dict[EpisodeKey, int]] = {}
    task_outcomes_by_code: dict[str, dict[int, list[int]]] = {}
    descriptions_reference: dict[int, str] | None = None
    metadata_by_code: dict[str, dict[str, Any]] = {}
    provenance_by_code: dict[str, dict[str, Any]] = {}
    for cell in FACTORIAL_CELLS:
        online_root = (
            round2_online_root if cell.protocol == ROUND2_PROTOCOL else round3a_online_root
        )
        condition_dir = online_root / cell.condition.name
        metadata_path = condition_dir / "run_metadata.json"
        metadata = _validate_online_metadata(
            metadata_path,
            cell=cell,
            valid_manifest_path=valid_manifest_path,
            valid_manifest_sha256=valid_manifest_sha256,
        )
        outcomes, by_task, descriptions, result_paths = _load_task_outcomes(
            condition_dir,
            cell=cell,
        )
        if descriptions_reference is None:
            descriptions_reference = descriptions
        elif descriptions != descriptions_reference:
            raise ValueError(f"Task descriptions differ for factorial cell {cell.code}.")
        outcomes_by_code[cell.code] = outcomes
        task_outcomes_by_code[cell.code] = by_task
        metadata_by_code[cell.code] = metadata
        provenance_by_code[cell.code] = {
            "online_root": str(online_root),
            "online_condition_dir": str(condition_dir.resolve()),
            "online_run_metadata_path": str(metadata_path.resolve()),
            "online_run_metadata_sha256": sha256_file(metadata_path),
            "online_task_result_file_set_sha256": _sha256_file_set(
                result_paths,
                relative_to=condition_dir,
            ),
            "online_git_commit_hash": metadata["git_commit_hash"],
        }
    _validate_cross_cell_online(metadata_by_code)

    reference_keys = set(outcomes_by_code["111"])
    for code in CELL_ORDER:
        keys = set(outcomes_by_code[code])
        if keys != reference_keys:
            raise ValueError(
                f"Online paired keys for cell {code} do not exactly match y111: "
                f"missing={sorted(reference_keys - keys)}, extra={sorted(keys - reference_keys)}"
            )
    ordered_keys = sorted(reference_keys)
    outcome_matrix = np.column_stack(
        [np.asarray([outcomes_by_code[code][key] for key in ordered_keys], dtype=np.int8)
         for code in CELL_ORDER]
    )
    return (
        ordered_keys,
        outcome_matrix,
        task_outcomes_by_code,
        descriptions_reference or {},
        metadata_by_code,
        provenance_by_code,
    )


def _load_valid_manifest(path: Path) -> tuple[dict[str, Any], list[str], str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Round-2 QC-valid manifest is unavailable: {resolved}")
    digest = sha256_file(resolved)
    payload = _read_json(resolved)
    values = payload.get("valid_sample_ids")
    if not isinstance(values, list):
        raise TypeError("QC-valid manifest valid_sample_ids must be a list.")
    sample_ids = [str(value) for value in values]
    if (
        len(sample_ids) != EXPECTED_OFFLINE_SAMPLES
        or len(sample_ids) != len(set(sample_ids))
        or any(not value for value in sample_ids)
    ):
        raise ValueError(
            f"QC-valid manifest must contain exactly {EXPECTED_OFFLINE_SAMPLES} unique IDs."
        )
    return payload, sample_ids, digest


OFFLINE_ONLINE_COMPATIBILITY_FIELDS = (
    "git_commit_hash",
    "checkpoint_path",
    "checkpoint_sha256",
    "dataset_stats_path",
    "dataset_stats_sha256",
    "state_bank_manifest_path",
    "state_bank_manifest_sha256",
    "valid_state_bank_manifest_path",
    "valid_state_bank_manifest_sha256",
    "prompt_context_cache_path",
    "prompt_context_cache_sha256",
    "num_model_layers",
    "task_suite",
    "task_ids",
    "seed",
    "number_of_trials",
    "action_horizon",
    "number_of_inference_steps",
    "replan_steps",
    "compile_action_infer",
    "binarize_gripper",
    "sigma_shift",
    "rand_device",
    "gpu_model",
    "torch_version",
    "cuda_version",
)


def _parse_dimension_vector(value: Any, *, context: str) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON vector for {context}.") from exc
    if not isinstance(value, list) or len(value) != len(CONTINUOUS_ACTION_DIMENSIONS):
        raise ValueError(
            f"{context} must contain {len(CONTINUOUS_ACTION_DIMENSIONS)} values."
        )
    return [_finite_float(item, context=f"{context}[{index}]") for index, item in enumerate(value)]


def _load_offline_cell(
    offline_root: Path,
    *,
    cell: FactorialCell,
    expected_sample_ids: Sequence[str],
    valid_manifest_path: Path,
    valid_manifest_sha256: str,
    online_metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    condition_dir = offline_root / cell.condition.name
    metadata_path = condition_dir / "run_metadata.json"
    summary_path = condition_dir / "summary.csv"
    per_sample_path = condition_dir / "per_sample.jsonl"
    for path in (metadata_path, summary_path, per_sample_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing offline artifact for cell {cell.code}: {path}")
    metadata = _read_json(metadata_path)
    expected = {
        "status": "complete",
        "condition_protocol": cell.protocol,
        "diagnosis_condition": cell.condition.name,
        "enabled_video_retrieval_layers": list(cell.enabled_layers),
        "disabled_video_layers": list(cell.condition.disabled_video_layers),
        "valid_state_bank_manifest_path": str(valid_manifest_path),
        "valid_state_bank_manifest_sha256": valid_manifest_sha256,
        "num_valid_samples": len(expected_sample_ids),
        "num_samples": len(expected_sample_ids),
        "executed_prefix_length": 10,
    }
    if cell.protocol == ROUND3A_PROTOCOL:
        physical_gpu = build_round3a_conditions(NUM_LAYERS).index(cell.condition)
        expected.update(
            {
                "condition_index": physical_gpu,
                "physical_gpu": physical_gpu,
                "cuda_visible_devices": str(physical_gpu),
                "model_device": "cuda:0",
            }
        )
    expected.update(
        {key: online_metadata.get(key) for key in OFFLINE_ONLINE_COMPATIBILITY_FIELDS}
    )
    mismatches = {
        key: {"observed": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Offline metadata mismatch for cell {cell.code}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    records = _read_jsonl(per_sample_path)
    if len(records) != len(expected_sample_ids):
        raise ValueError(
            f"Cell {cell.code} has {len(records)} offline samples; "
            f"expected {len(expected_sample_ids)}."
        )
    observed_ids: list[str] = []
    for index, record in enumerate(records):
        sample_id = str(record.get("sample_id", ""))
        observed_ids.append(sample_id)
        expected_record = {
            "condition": cell.condition.name,
            "enabled_video_retrieval_layers": list(cell.enabled_layers),
            "disabled_video_layers": list(cell.condition.disabled_video_layers),
            "executed_prefix_length": 10,
            "action_dimension_names": list(CONTINUOUS_ACTION_DIMENSIONS),
        }
        record_mismatches = {
            key: {"observed": record.get(key), "expected": value}
            for key, value in expected_record.items()
            if record.get(key) != value
        }
        if record_mismatches:
            raise ValueError(
                f"Offline record mismatch for cell {cell.code}, sample {sample_id or index}: "
                f"{json.dumps(record_mismatches, sort_keys=True)}"
            )
        for identity in ("task_suite", "task_id", "episode_id", "replan_id"):
            if identity not in record:
                raise ValueError(f"Offline sample {sample_id} is missing {identity}.")
        for metric in SCALAR_OFFLINE_METRICS:
            record[metric] = _finite_float(
                record.get(metric), context=f"{cell.code}.{sample_id}.{metric}"
            )
        record[DIMENSION_OFFLINE_METRIC] = _parse_dimension_vector(
            record.get(DIMENSION_OFFLINE_METRIC),
            context=f"{cell.code}.{sample_id}.{DIMENSION_OFFLINE_METRIC}",
        )
    if observed_ids != list(expected_sample_ids):
        raise ValueError(
            f"Offline sample ordering for cell {cell.code} does not exactly match "
            "the immutable valid manifest."
        )
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Offline summary must contain one row: {summary_path}")
    summary: dict[str, Any] = dict(rows[0])
    summary_expected = {
        "condition": cell.condition.name,
        "num_samples": str(len(records)),
        "executed_prefix_length": "10",
    }
    for key, value in summary_expected.items():
        if str(summary.get(key)) != value:
            raise ValueError(f"Offline summary mismatch for {cell.code}.{key}.")
    for metric in SCALAR_OFFLINE_METRICS:
        reported = _finite_float(summary.get(metric), context=f"{summary_path}.{metric}")
        recomputed = float(np.mean([record[metric] for record in records]))
        if not math.isclose(reported, recomputed, rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError(
                f"Offline summary mismatch for cell {cell.code}.{metric}: "
                f"reported={reported}, recomputed={recomputed}."
            )
        summary[metric] = reported
    dimensions = _parse_dimension_vector(
        summary.get(DIMENSION_OFFLINE_METRIC),
        context=f"{summary_path}.{DIMENSION_OFFLINE_METRIC}",
    )
    recomputed_dimensions = np.mean(
        np.asarray([record[DIMENSION_OFFLINE_METRIC] for record in records], dtype=float),
        axis=0,
    )
    if not np.allclose(dimensions, recomputed_dimensions, rtol=1e-7, atol=1e-12):
        raise ValueError(f"Offline action-dimension summary mismatch for cell {cell.code}.")
    summary[DIMENSION_OFFLINE_METRIC] = dimensions
    provenance = {
        "offline_root": str(offline_root),
        "offline_condition_dir": str(condition_dir.resolve()),
        "offline_run_metadata_path": str(metadata_path.resolve()),
        "offline_run_metadata_sha256": sha256_file(metadata_path),
        "offline_summary_path": str(summary_path.resolve()),
        "offline_summary_sha256": sha256_file(summary_path),
        "offline_per_sample_path": str(per_sample_path.resolve()),
        "offline_per_sample_sha256": sha256_file(per_sample_path),
        "offline_git_commit_hash": metadata["git_commit_hash"],
    }
    return records, summary, provenance


def load_offline_factorial(
    *,
    round2_offline_root: Path,
    round3a_offline_root: Path,
    expected_sample_ids: Sequence[str],
    valid_manifest_path: Path,
    valid_manifest_sha256: str,
    online_metadata_by_code: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    records_by_code: dict[str, list[dict[str, Any]]] = {}
    summaries_by_code: dict[str, dict[str, Any]] = {}
    provenance_by_code: dict[str, dict[str, Any]] = {}
    for cell in FACTORIAL_CELLS:
        offline_root = (
            round2_offline_root if cell.protocol == ROUND2_PROTOCOL else round3a_offline_root
        )
        records, summary, provenance = _load_offline_cell(
            offline_root,
            cell=cell,
            expected_sample_ids=expected_sample_ids,
            valid_manifest_path=valid_manifest_path,
            valid_manifest_sha256=valid_manifest_sha256,
            online_metadata=online_metadata_by_code[cell.code],
        )
        records_by_code[cell.code] = records
        summaries_by_code[cell.code] = summary
        provenance_by_code[cell.code] = provenance
    for sample_index, expected_id in enumerate(expected_sample_ids):
        reference = records_by_code["111"][sample_index]
        reference_identity = tuple(
            reference[key] for key in ("task_suite", "task_id", "episode_id", "replan_id")
        )
        for code in CELL_ORDER:
            record = records_by_code[code][sample_index]
            identity = tuple(
                record[key] for key in ("task_suite", "task_id", "episode_id", "replan_id")
            )
            if str(record["sample_id"]) != expected_id or identity != reference_identity:
                raise ValueError(
                    f"Offline paired identity mismatch for cell {code}, sample {expected_id}."
                )
    return records_by_code, summaries_by_code, provenance_by_code


def _joint_cell_bootstrap_intervals(
    keys: Sequence[EpisodeKey],
    outcomes: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(outcomes, dtype=np.float64)
    if values.shape != (len(keys), len(CELL_ORDER)) or not keys:
        raise ValueError("Joint cell bootstrap requires one complete eight-cell row per key.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), size=(samples, len(keys)))
    paired_estimates = values[indices].mean(axis=1)
    paired_low, paired_high = np.quantile(paired_estimates, [0.025, 0.975], axis=0)

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (suite, task_id, _trial_id) in enumerate(keys):
        grouped[(suite, int(task_id))].append(index)
    groups = [np.asarray(grouped[key], dtype=int) for key in sorted(grouped)]
    hierarchical = np.empty((samples, len(CELL_ORDER)), dtype=np.float64)
    for bootstrap_index in range(samples):
        selected_groups = rng.integers(0, len(groups), size=len(groups))
        task_means = np.empty((len(groups), len(CELL_ORDER)), dtype=np.float64)
        for output_index, group_index in enumerate(selected_groups):
            members = groups[int(group_index)]
            selected = members[rng.integers(0, members.size, size=members.size)]
            task_means[output_index] = values[selected].mean(axis=0)
        hierarchical[bootstrap_index] = task_means.mean(axis=0)
    hierarchical_low, hierarchical_high = np.quantile(
        hierarchical, [0.025, 0.975], axis=0
    )
    return paired_low, paired_high, hierarchical_low, hierarchical_high


def _offline_rows(
    records_by_code: Mapping[str, Sequence[Mapping[str, Any]]],
    summaries_by_code: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    replan_rows: list[dict[str, Any]] = []
    for cell in FACTORIAL_CELLS:
        records = list(records_by_code[cell.code])
        clusters = [
            (str(record["task_suite"]), int(record["task_id"]), int(record["episode_id"]))
            for record in records
        ]
        primary_values = [float(record["executed_prefix_norm_rms"]) for record in records]
        ci_low, ci_high = episode_cluster_bootstrap_ci(
            primary_values,
            clusters,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, "offline", cell.code),
        )
        summary = summaries_by_code[cell.code]
        row: dict[str, Any] = {
            "cell_code": cell.code,
            "A": cell.a,
            "B": cell.b,
            "C": cell.c,
            "condition": cell.condition.name,
            "num_samples": len(records),
            "num_episode_clusters": len(set(clusters)),
            "executed_prefix_norm_rms_cluster_ci_low": ci_low,
            "executed_prefix_norm_rms_cluster_ci_high": ci_high,
        }
        row.update({metric: float(summary[metric]) for metric in SCALAR_OFFLINE_METRICS})
        row[DIMENSION_OFFLINE_METRIC] = _json_for_csv(summary[DIMENSION_OFFLINE_METRIC])
        rows.append(row)

        by_replan: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            by_replan[int(record["replan_id"])].append(record)
        if set(by_replan) != set(EXPECTED_REPLAN_STAGES):
            raise ValueError(
                f"Cell {cell.code} has replan stages {sorted(by_replan)}, "
                f"expected {list(EXPECTED_REPLAN_STAGES)}."
            )
        for replan_id in EXPECTED_REPLAN_STAGES:
            stage_records = by_replan[replan_id]
            stage_values = [
                float(record["executed_prefix_norm_rms"]) for record in stage_records
            ]
            stage_clusters = [
                (
                    str(record["task_suite"]),
                    int(record["task_id"]),
                    int(record["episode_id"]),
                )
                for record in stage_records
            ]
            stage_low, stage_high = episode_cluster_bootstrap_ci(
                stage_values,
                stage_clusters,
                samples=bootstrap_samples,
                seed=_stable_seed(bootstrap_seed, "replan", cell.code, replan_id),
            )
            replan_rows.append(
                {
                    "cell_code": cell.code,
                    "A": cell.a,
                    "B": cell.b,
                    "C": cell.c,
                    "condition": cell.condition.name,
                    "replan_id": replan_id,
                    "executed_prefix_norm_rms": float(np.mean(stage_values)),
                    "episode_cluster_ci_low": stage_low,
                    "episode_cluster_ci_high": stage_high,
                    "num_samples": len(stage_values),
                    "num_episode_clusters": len(set(stage_clusters)),
                }
            )
    return rows, replan_rows


def _flatten_simple_rows(
    analysis: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transition_by_id = {str(row["estimand"]): row for row in transitions}
    simple_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for source in analysis["simple_effects"]:
        estimand = str(source["estimand"])
        context = _parse_simple_context(
            source.get("context"), factor=str(source["factor"])
        )
        transition = dict(transition_by_id[estimand])
        row = {
            "effect_id": estimand,
            "factor": source["factor"],
            "context_A": context.get("A", ""),
            "context_B": context.get("B", ""),
            "context_C": context.get("C", ""),
            "reference_cell": source["reference_cell"],
            "target_cell": source["target_cell"],
            "formula": source["formula"],
            "coefficients": _json_for_csv(source["coefficients"]),
            "coefficient_by_cell": _json_for_csv(source["coefficient_by_cell"]),
            "effect_scale": "success_probability_difference",
            "effect": source["effect"],
            "effect_pp": 100.0 * float(source["effect"]),
            "paired_ci_low": source["paired_ci_low"],
            "paired_ci_high": source["paired_ci_high"],
            "task_hierarchical_ci_low": source["task_hierarchical_ci_low"],
            "task_hierarchical_ci_high": source["task_hierarchical_ci_high"],
            "ci_method": "joint paired bootstrap; joint task-hierarchical paired bootstrap",
            "ci_scope": "pointwise",
            "bootstrap_samples": analysis["bootstrap_samples"],
            "bootstrap_seed": analysis["bootstrap_seed"],
            "mcnemar_exact_p_value": transition["mcnemar_exact_p_value"],
        }
        simple_rows.append(row)
        transition_rows.append(
            {
                "effect_id": estimand,
                "factor": source["factor"],
                "context_A": context.get("A", ""),
                "context_B": context.get("B", ""),
                "context_C": context.get("C", ""),
                "reference_cell": source["reference_cell"],
                "target_cell": source["target_cell"],
                "reference_success_to_target_success": transition[
                    "success_to_success"
                ],
                "reference_success_to_target_failure": transition[
                    "success_to_failure"
                ],
                "reference_failure_to_target_success": transition[
                    "failure_to_success"
                ],
                "reference_failure_to_target_failure": transition[
                    "failure_to_failure"
                ],
                "induced_failures": transition["induced_failures"],
                "rescued_successes": transition["rescued_successes"],
                "discordant_pairs": transition["discordant_pairs"],
                "mcnemar_exact_p_value": transition["mcnemar_exact_p_value"],
                "multiplicity_note": "descriptive simple-effect p; not in seven-contrast Holm family",
                "paired_episodes": analysis["n_episodes"],
            }
        )
    return simple_rows, transition_rows


def _flatten_contrast_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in analysis["factorial_contrasts"]:
        rows.append(
            {
                "contrast_id": source["estimand"],
                "contrast_kind": source["kind"],
                "formula": source["formula"],
                "coefficients": _json_for_csv(source["coefficients"]),
                "coefficient_by_cell": _json_for_csv(source["coefficient_by_cell"]),
                "effect_scale": "average_probability_effect_or_difference_of_differences",
                "effect": source["effect"],
                "effect_pp": 100.0 * float(source["effect"]),
                "paired_ci_low": source["paired_ci_low"],
                "paired_ci_high": source["paired_ci_high"],
                "task_hierarchical_ci_low": source["task_hierarchical_ci_low"],
                "task_hierarchical_ci_high": source["task_hierarchical_ci_high"],
                "ci_method": "joint paired bootstrap; joint task-hierarchical paired bootstrap",
                "ci_scope": "pointwise",
                "bootstrap_samples": analysis["bootstrap_samples"],
                "bootstrap_seed": analysis["bootstrap_seed"],
                "raw_p_value": source["raw_p_value"],
                "holm_p_value": source["holm_p_value"],
                "holm_family_size": 7,
                "test_method": source["test_method"],
                "test_assumption": source["test_assumption"],
                "sign_flip_samples": analysis["sign_flip_samples"],
                "sign_flip_seed": analysis["sign_flip_seed"],
            }
        )
    return rows


def _task_rows(
    task_outcomes_by_code: Mapping[str, Mapping[int, Sequence[int]]],
    descriptions: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in range(EXPECTED_TASKS):
        row: dict[str, Any] = {
            "task_id": task_id,
            "task_description": descriptions[task_id],
        }
        reference_rate = float(np.mean(task_outcomes_by_code["111"][task_id]))
        for code in CELL_ORDER:
            rate = float(np.mean(task_outcomes_by_code[code][task_id]))
            row[f"y{code}"] = rate
            row[f"delta_y{code}_vs_y111"] = rate - reference_rate
        rows.append(row)
    return rows


def _evidence_from_intervals(row: Mapping[str, Any]) -> str:
    paired = (float(row["paired_ci_low"]), float(row["paired_ci_high"]))
    hierarchical = (
        float(row["task_hierarchical_ci_low"]),
        float(row["task_hierarchical_ci_high"]),
    )
    if paired[0] > 0 and hierarchical[0] > 0:
        return "positive; both pointwise 95% CIs exclude zero"
    if paired[1] < 0 and hierarchical[1] < 0:
        return "negative; both pointwise 95% CIs exclude zero"
    return "uncertain; at least one pointwise 95% CI includes zero"


def _scientific_questions(
    simple_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    simple = {str(row["effect_id"]): row for row in simple_rows}
    contrasts = {str(row["contrast_id"]): row for row in contrast_rows}
    b_ids = (
        "simple_B_A0_C0",
        "simple_B_A1_C0",
        "simple_B_A0_C1",
        "simple_B_A1_C1",
    )
    b_rows = [simple[identifier] for identifier in b_ids]
    b_effects = [float(row["effect_pp"]) for row in b_rows]
    b_interactions = [
        contrasts[identifier]
        for identifier in ("interaction_AB", "interaction_BC", "interaction_ABC")
    ]
    context_supported = any(
        _evidence_from_intervals(row).startswith(("positive", "negative"))
        for row in b_interactions
    )
    ac = contrasts["interaction_AC"]
    abc = contrasts["interaction_ABC"]
    return [
        {
            "question": "Q1 Does B alone perform above no retrieval?",
            "estimand": b_ids[0],
            "answer": f"Observed effect {b_effects[0]:+.1f} pp; {_evidence_from_intervals(b_rows[0])}.",
        },
        {
            "question": "Q2 Does adding B to A help when C is absent?",
            "estimand": b_ids[1],
            "answer": f"Observed effect {b_effects[1]:+.1f} pp; {_evidence_from_intervals(b_rows[1])}.",
        },
        {
            "question": "Q3 Does adding B to C help when A is absent?",
            "estimand": b_ids[2],
            "answer": f"Observed effect {b_effects[2]:+.1f} pp; {_evidence_from_intervals(b_rows[2])}.",
        },
        {
            "question": "Q4 Does adding B to A+C help?",
            "estimand": b_ids[3],
            "answer": f"Observed effect {b_effects[3]:+.1f} pp; {_evidence_from_intervals(b_rows[3])}.",
        },
        {
            "question": "Q5 Is B strongly context-dependent?",
            "estimand": "B simple-effect range and B-related interactions",
            "answer": (
                f"B simple effects range from {min(b_effects):+.1f} to {max(b_effects):+.1f} pp. "
                + (
                    "At least one B-related interaction has both pointwise CIs excluding zero."
                    if context_supported
                    else "No B-related interaction has both pointwise CIs excluding zero."
                )
            ),
        },
        {
            "question": "Q6 Is averaged A+C complementarity visible?",
            "estimand": "interaction_AC",
            "answer": (
                f"A×C={float(ac['effect_pp']):+.1f} pp; {_evidence_from_intervals(ac)}. "
                "This is an averaged factorial interaction, not an established semantic mechanism."
            ),
        },
        {
            "question": "Q7 Is there evidence for an A×B×C interaction?",
            "estimand": "interaction_ABC",
            "answer": (
                f"A×B×C={float(abc['effect_pp']):+.1f} pp; {_evidence_from_intervals(abc)}; "
                f"Holm p={float(abc['holm_p_value']):.6g}."
            ),
        },
    ]


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# ASRE Round 3A: complete late-half 2^3 factorial",
        "",
        "This summary combines five immutable Round-2 cells with the three new",
        "Round-3A cells. Effects are on the success-probability scale; percentage-",
        "point values are probability effects multiplied by 100.",
        "",
        "## Observed",
        "",
        "| Cell | A | B | C | Condition | Success | Executed-prefix RMS |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["cells"]:
        lines.append(
            f"| {row['cell_code']} | {row['A']} | {row['B']} | {row['C']} | "
            f"`{row['condition']}` | {100.0 * float(row['online_success_rate']):.1f}% | "
            f"{float(row['executed_prefix_norm_rms']):.4f} |"
        )
    lines.extend(["", "### Contextual effects", ""])
    for row in payload["simple_effects"]:
        lines.append(
            f"- `{row['effect_id']}`: {float(row['effect_pp']):+.1f} pp; paired 95% CI "
            f"[{100.0 * float(row['paired_ci_low']):+.1f}, "
            f"{100.0 * float(row['paired_ci_high']):+.1f}] pp; task-hierarchical "
            f"[{100.0 * float(row['task_hierarchical_ci_low']):+.1f}, "
            f"{100.0 * float(row['task_hierarchical_ci_high']):+.1f}] pp."
        )
    lines.extend(["", "### Seven factorial contrasts", ""])
    for row in payload["factorial_contrasts"]:
        lines.append(
            f"- `{row['contrast_id']}`: {float(row['effect_pp']):+.1f} pp; paired 95% CI "
            f"[{100.0 * float(row['paired_ci_low']):+.1f}, "
            f"{100.0 * float(row['paired_ci_high']):+.1f}] pp; task-hierarchical "
            f"[{100.0 * float(row['task_hierarchical_ci_low']):+.1f}, "
            f"{100.0 * float(row['task_hierarchical_ci_high']):+.1f}] pp; "
            f"raw sign-flip p={float(row['raw_p_value']):.6g}, "
            f"Holm p={float(row['holm_p_value']):.6g}."
        )
    lines.extend(["", "### Pre-specified scientific questions", ""])
    for item in payload["scientific_questions"]:
        lines.append(f"- **{item['question']}** {item['answer']}")
    lines.extend(
        [
            "",
            "## Supported",
            "",
            "Only directions for which both the paired and task-hierarchical pointwise",
            "95% intervals exclude zero are described above as supported. Holm correction",
            "applies only to the seven main/pairwise/three-way factorial contrasts; the",
            "12 McNemar simple-effect p-values are descriptive and unadjusted.",
            "",
            "The interaction contrasts summarize experimental dependence on retained",
            "retrieval context. They are not a complete causal model of neural computation.",
            "",
            "## Hypothesis",
            "",
            "Terms such as bridging, grounding, refinement, or staged retrieval remain",
            "functional hypotheses. Removing K/V also changes the attention key set and",
            "softmax normalization. Matched-shape K/V replacement or residual patching would",
            "be required to separate visual-content dependence from structural attention",
            "effects, and those experiments are outside the Round-3A stop rule.",
            "",
            "## Statistical scope",
            "",
            f"- Paired episodes: {payload['analysis']['n_episodes']}",
            f"- Tasks: {payload['analysis']['n_tasks']}",
            f"- Joint bootstrap replicates: {payload['analysis']['bootstrap_samples']}",
            "- Bootstrap intervals are pointwise, not simultaneous.",
            "- Factorial p-values use a Monte-Carlo paired sign-flip test whose",
            "  interpretation depends on sign-exchangeability of episode-level contrast",
            "  scores; this is not claimed to be a design-exact randomized-treatment test.",
            "- Results remain one checkpoint, one seed, and LIBERO-Spatial only.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(value)
            if not value.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def aggregate(
    *,
    round2_online_root: Path,
    round2_offline_root: Path,
    round3a_online_root: Path,
    round3a_offline_root: Path,
    valid_manifest_path: Path,
    output_dir: Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    sign_flip_samples: int = 100_000,
    sign_flip_seed: int = 0,
) -> Path:
    input_dirs = [
        round2_online_root.resolve(),
        round2_offline_root.resolve(),
        round3a_online_root.resolve(),
        round3a_offline_root.resolve(),
    ]
    for path in input_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Required factorial input directory is unavailable: {path}")
    output_dir = output_dir.resolve()
    round2_root = round2_online_root.resolve().parent
    if (
        round2_offline_root.resolve().parent != round2_root
        or valid_manifest_path.resolve().parent != round2_root
    ):
        raise ValueError(
            "Round-2 online, offline, and valid-manifest inputs must share one "
            "frozen Round-2 root."
        )
    safety_manifest = _read_json(valid_manifest_path.resolve())
    source_manifest_value = safety_manifest.get("source_manifest_path")
    if not isinstance(source_manifest_value, str) or not source_manifest_value.strip():
        raise ValueError("QC-valid manifest has no source_manifest_path for safety checks.")
    source_manifest_path = Path(
        os.path.expanduser(os.path.expandvars(source_manifest_value))
    )
    if not source_manifest_path.is_absolute():
        source_manifest_path = valid_manifest_path.resolve().parent / source_manifest_path
    round1_root = source_manifest_path.resolve().parent.parent
    protected_roots = {round2_root}
    for name in ("state_bank", "offline", "online_smoke", "online_full", "aggregate"):
        candidate = (round1_root / name).resolve()
        if candidate.exists():
            protected_roots.add(candidate)
    if any(
        output_dir == path
        or output_dir.is_relative_to(path)
        or path.is_relative_to(output_dir)
        for path in protected_roots
    ):
        raise ValueError(
            "Factorial output directory overlaps frozen Round-1/2 results: "
            f"{output_dir}."
        )
    if output_dir in input_dirs or any(output_dir.is_relative_to(path) for path in input_dirs):
        raise ValueError("Factorial output directory must not overlap any source result tree.")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Round-3A aggregate directory: {output_dir}"
        )
    if bootstrap_samples <= 0 or sign_flip_samples <= 0:
        raise ValueError("Bootstrap and sign-flip sample counts must be positive.")

    valid_manifest, valid_sample_ids, valid_manifest_sha256 = _load_valid_manifest(
        valid_manifest_path
    )
    valid_manifest_path = valid_manifest_path.resolve()
    (
        online_keys,
        outcome_matrix,
        task_outcomes_by_code,
        task_descriptions,
        online_metadata_by_code,
        online_provenance_by_code,
    ) = load_online_factorial(
        round2_online_root=round2_online_root.resolve(),
        round3a_online_root=round3a_online_root.resolve(),
        valid_manifest_path=valid_manifest_path,
        valid_manifest_sha256=valid_manifest_sha256,
    )
    records_by_code, summaries_by_code, offline_provenance_by_code = load_offline_factorial(
        round2_offline_root=round2_offline_root.resolve(),
        round3a_offline_root=round3a_offline_root.resolve(),
        expected_sample_ids=valid_sample_ids,
        valid_manifest_path=valid_manifest_path,
        valid_manifest_sha256=valid_manifest_sha256,
        online_metadata_by_code=online_metadata_by_code,
    )

    analysis = analyze_factorial(
        online_keys,
        outcome_matrix,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        sign_flip_samples=sign_flip_samples,
        sign_flip_seed=sign_flip_seed,
    )
    transitions = simple_effect_transitions(online_keys, outcome_matrix)
    paired_low, paired_high, hierarchical_low, hierarchical_high = (
        _joint_cell_bootstrap_intervals(
            online_keys,
            outcome_matrix,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    )
    offline_rows, replan_rows = _offline_rows(
        records_by_code,
        summaries_by_code,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    offline_by_code = {row["cell_code"]: row for row in offline_rows}

    cell_rows: list[dict[str, Any]] = []
    cell_json: list[dict[str, Any]] = []
    for index, cell in enumerate(FACTORIAL_CELLS):
        online_rate = float(outcome_matrix[:, index].mean())
        provenance = {
            **online_provenance_by_code[cell.code],
            **offline_provenance_by_code[cell.code],
            "checkpoint_path": online_metadata_by_code[cell.code]["checkpoint_path"],
            "checkpoint_sha256": online_metadata_by_code[cell.code]["checkpoint_sha256"],
            "dataset_stats_path": online_metadata_by_code[cell.code]["dataset_stats_path"],
            "dataset_stats_sha256": online_metadata_by_code[cell.code][
                "dataset_stats_sha256"
            ],
            "valid_state_bank_manifest_path": str(valid_manifest_path),
            "valid_state_bank_manifest_sha256": valid_manifest_sha256,
            "prompt_context_cache_sha256": online_metadata_by_code[cell.code][
                "prompt_context_cache_sha256"
            ],
        }
        row = {
            **cell.to_dict(),
            "enabled_layer_ranges": _format_layer_ranges(cell.enabled_layers),
            "online_success_rate": online_rate,
            "paired_ci_low": float(paired_low[index]),
            "paired_ci_high": float(paired_high[index]),
            "task_hierarchical_ci_low": float(hierarchical_low[index]),
            "task_hierarchical_ci_high": float(hierarchical_high[index]),
            "per_task_success_rate": _json_for_csv(
                {
                    str(task_id): float(np.mean(task_outcomes_by_code[cell.code][task_id]))
                    for task_id in range(EXPECTED_TASKS)
                }
            ),
            **{
                metric: offline_by_code[cell.code][metric]
                for metric in SCALAR_OFFLINE_METRICS
            },
            "executed_prefix_norm_rms_cluster_ci_low": offline_by_code[cell.code][
                "executed_prefix_norm_rms_cluster_ci_low"
            ],
            "executed_prefix_norm_rms_cluster_ci_high": offline_by_code[cell.code][
                "executed_prefix_norm_rms_cluster_ci_high"
            ],
            "offline_samples": len(records_by_code[cell.code]),
            "provenance": _json_for_csv(provenance),
        }
        cell_rows.append(row)
        json_row = dict(row)
        json_row["per_task_success_rate"] = json.loads(row["per_task_success_rate"])
        json_row["provenance"] = provenance
        cell_json.append(json_row)

    simple_rows, transition_rows = _flatten_simple_rows(analysis, transitions)
    contrast_rows = _flatten_contrast_rows(analysis)
    task_rows = _task_rows(task_outcomes_by_code, task_descriptions)
    questions = _scientific_questions(simple_rows, contrast_rows)
    summary_payload = {
        "schema_version": 1,
        "artifact_type": "asre_round3a_complete_late_half_factorial",
        "status": "complete",
        "created_at": now_iso(),
        "analysis_git_commit_hash": git_commit(project_root),
        "cell_order": list(CELL_ORDER),
        "cells": cell_json,
        "simple_effects": [
            {
                **row,
                "coefficients": json.loads(str(row["coefficients"])),
                "coefficient_by_cell": json.loads(
                    str(row["coefficient_by_cell"])
                ),
            }
            for row in simple_rows
        ],
        "factorial_contrasts": [
            {
                **row,
                "coefficients": json.loads(str(row["coefficients"])),
                "coefficient_by_cell": json.loads(
                    str(row["coefficient_by_cell"])
                ),
            }
            for row in contrast_rows
        ],
        "scientific_questions": questions,
        "analysis": {
            key: value
            for key, value in analysis.items()
            if key not in {"simple_effects", "factorial_contrasts"}
        },
        "statistical_scope": {
            "effect_scale": "success probability; pp values multiply effects by 100",
            "bootstrap_pairing": "complete eight-cell outcome vector resampled jointly",
            "task_hierarchical_pairing": (
                "tasks resampled, then paired episode rows resampled jointly within task"
            ),
            "ci_scope": "pointwise",
            "holm_family": [row["contrast_id"] for row in contrast_rows],
            "simple_mcnemar_multiplicity": "descriptive and unadjusted",
        },
        "provenance": {
            "round2_online_root": str(round2_online_root.resolve()),
            "round2_offline_root": str(round2_offline_root.resolve()),
            "round3a_online_root": str(round3a_online_root.resolve()),
            "round3a_offline_root": str(round3a_offline_root.resolve()),
            "valid_state_bank_manifest_path": str(valid_manifest_path),
            "valid_state_bank_manifest_sha256": valid_manifest_sha256,
            "valid_sample_count": len(valid_sample_ids),
            "valid_manifest_checkpoint_sha256": valid_manifest.get("checkpoint_sha256"),
            "valid_manifest_dataset_stats_sha256": valid_manifest.get(
                "dataset_stats_sha256"
            ),
            "valid_manifest_source_manifest_sha256": valid_manifest.get(
                "source_manifest_sha256"
            ),
            "valid_manifest_prompt_context_cache_sha256": valid_manifest.get(
                "prompt_context_cache_sha256"
            ),
        },
        "stop_rule": (
            "Round 3A completes the layer-level factorial. No Round 3B experiment "
            "is authorized by this analysis."
        ),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _write_csv(temporary_dir / "factorial_cells.csv", cell_rows)
        atomic_write_json(temporary_dir / "factorial_cells.json", {"cells": cell_json})
        _write_csv(temporary_dir / "factorial_simple_effects.csv", simple_rows)
        _write_csv(temporary_dir / "factorial_interactions.csv", contrast_rows)
        _write_csv(temporary_dir / "factorial_task_success.csv", task_rows)
        _write_csv(temporary_dir / "factorial_offline_metrics.csv", offline_rows)
        _write_csv(temporary_dir / "factorial_replan_stage_metrics.csv", replan_rows)
        _write_csv(temporary_dir / "factorial_paired_transitions.csv", transition_rows)
        atomic_write_json(temporary_dir / "factorial_summary.json", summary_payload)
        _write_text(temporary_dir / "factorial_summary.md", _summary_markdown(summary_payload))
        aggregate_metadata = {
            "schema_version": 1,
            "artifact_type": "asre_round3a_factorial_aggregate_metadata",
            "status": "complete",
            "analysis_git_commit_hash": git_commit(project_root),
            "created_at": now_iso(),
            "round2_online_root": str(round2_online_root.resolve()),
            "round2_offline_root": str(round2_offline_root.resolve()),
            "round3a_online_root": str(round3a_online_root.resolve()),
            "round3a_offline_root": str(round3a_offline_root.resolve()),
            "valid_state_bank_manifest_path": str(valid_manifest_path),
            "valid_state_bank_manifest_sha256": valid_manifest_sha256,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "sign_flip_samples": sign_flip_samples,
            "sign_flip_seed": sign_flip_seed,
            "output_files": sorted(
                [
                    path.name
                    for path in temporary_dir.iterdir()
                    if path.is_file()
                ]
                + ["aggregate_metadata.json"]
            ),
        }
        atomic_write_json(temporary_dir / "aggregate_metadata.json", aggregate_metadata)
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return output_dir


def main() -> None:
    args = _parse_args()
    output = aggregate(
        round2_online_root=args.round2_online_root,
        round2_offline_root=args.round2_offline_root,
        round3a_online_root=args.round3a_online_root,
        round3a_offline_root=args.round3a_offline_root,
        valid_manifest_path=args.valid_manifest,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        sign_flip_samples=args.sign_flip_samples,
        sign_flip_seed=args.sign_flip_seed,
    )
    print(f"Complete ASRE late-half factorial aggregated into {output}")


if __name__ == "__main__":
    main()
