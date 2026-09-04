#!/usr/bin/env python3
"""Aggregate the fixed ``move_block_reveal_can`` experiment outputs.

This script intentionally knows only the directory layout of this one task.  It
reports observations from the two native calibrations, the admission decision,
and (when admitted) the three frozen task conditions.  A REJECT decision is a
valid calibration-only terminal result, not an aggregation error.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


TASK = "move_block_reveal_can"
CONDITIONS = ("visible", "missing", "oracle_reveal")
FORMAL_FREEZE_NAME = "MINIBENCH_TASK_FREEZE_V0_3_11"
PILOT_FIELDS = (
    "seed",
    "success",
    "block_moved_by_policy",
    "block_moved_step",
    "max_block_displacement_from_handoff_m",
    "max_block_rotation_from_handoff_rad",
    "oracle_applied",
    "can_ever_visible",
    "can_became_visible_after_handoff",
    "first_visible_step",
    "final_target_pixels",
    "success_diagnostics",
    "video_path",
)
PILOT_VALIDATION_FIELDS = (
    "instruction",
    "initial_success",
    "pre_intervention_state",
    "handoff_state",
    "handoff_target_pixels",
)
PAIRED_STATE_ATOL = 1e-4

CALIBRATION_SPECS = (
    (
        "place_can_basket",
        Path("fastwam/calibration/place_can_basket/place_can_basket/native/episodes.jsonl"),
        Path("metadata/place_can_basket_calibration_seeds.json"),
    ),
    (
        "move_block_reveal_can_block_calibration",
        Path(
            "fastwam/calibration/move_block_reveal_can_block_calibration/"
            "move_block_reveal_can_block_calibration/native/episodes.jsonl"
        ),
        Path("metadata/move_block_reveal_can_block_calibration_seeds.json"),
    ),
)

FREEZE_RELATIVE_PATH = Path(f"metadata/{FORMAL_FREEZE_NAME}.json")
SEED_MANIFEST_RELATIVE_PATH = Path("metadata/paired_seeds.json")
ADMISSION_RELATIVE_PATH = Path("metadata/admission.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate only the move_block_reveal_can formal experiment."
    )
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="Formal experiment root containing the fastwam/ directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("admission", "aggregate"),
        default="aggregate",
    )
    return parser.parse_args()


def _input_paths(run_root: Path) -> dict[str, Path]:
    paths = {
        "freeze": run_root / FREEZE_RELATIVE_PATH,
        "seed_manifest": run_root / SEED_MANIFEST_RELATIVE_PATH,
        "admission": run_root / ADMISSION_RELATIVE_PATH,
    }
    paths.update(
        {
            f"calibration/{name}": run_root / relative_path
            for name, relative_path, _ in CALIBRATION_SPECS
        }
    )
    paths.update(
        {
            f"calibration_manifest/{name}": run_root / manifest_path
            for name, _, manifest_path in CALIBRATION_SPECS
        }
    )
    paths.update(
        {
            f"pilot/{condition}": (
                run_root
                / "fastwam"
                / "pilot"
                / condition
                / TASK
                / condition
                / "episodes.jsonl"
            )
            for condition in CONDITIONS
        }
    )
    return paths


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}")
    return payload


def _validate_seed_list(value: Any, *, label: str, path: Path) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty seed list in {path}")
    seeds: list[int] = []
    for index, seed in enumerate(value):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(
                f"{label}[{index}] must be an integer seed in {path}, got {seed!r}"
            )
        seeds.append(seed)
    duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if duplicates:
        raise ValueError(f"Duplicate seeds in {label} at {path}: {duplicates}")
    return seeds


def _load_frozen_protocol(
    freeze_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    freeze = _read_json_object(freeze_path)
    manifest = _read_json_object(manifest_path)

    if freeze.get("freeze_name") != FORMAL_FREEZE_NAME:
        raise ValueError(
            f"Wrong or missing freeze_name in {freeze_path}: {freeze.get('freeze_name')!r}"
        )
    for label, payload, path in (
        ("freeze", freeze, freeze_path),
        ("seed manifest", manifest, manifest_path),
    ):
        if payload.get("task") != TASK:
            raise ValueError(
                f"Wrong task in {label} {path}: expected {TASK!r}, "
                f"got {payload.get('task')!r}"
            )

    freeze_config = freeze.get("task_config")
    manifest_config = manifest.get("task_config")
    if not isinstance(freeze_config, str) or not freeze_config:
        raise ValueError(f"Missing task_config in freeze {freeze_path}")
    if manifest_config != freeze_config:
        raise ValueError(
            f"task_config mismatch between freeze and seed manifest: "
            f"{freeze_config!r} != {manifest_config!r}"
        )

    freeze_seeds = _validate_seed_list(
        freeze.get("seeds"), label="freeze seeds", path=freeze_path
    )
    manifest_seeds = _validate_seed_list(
        manifest.get("seeds"), label="manifest seeds", path=manifest_path
    )
    if freeze_seeds != manifest_seeds:
        raise ValueError(
            "Seed list mismatch between freeze and seed manifest: "
            f"freeze={freeze_seeds}, manifest={manifest_seeds}"
        )
    return freeze, manifest, manifest_seeds


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}, "
                    f"got {type(record).__name__}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"No episode records found in {path}")
    return records


def _record_seed(record: dict[str, Any], path: Path, index: int) -> int:
    if "seed" not in record:
        raise ValueError(f"Missing 'seed' in {path} episode record {index}")
    seed = record["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"Expected integer 'seed' in {path} episode record {index}, got {seed!r}"
        )
    return seed


def _validate_unique_record_seeds(
    records: list[dict[str, Any]], path: Path, *, label: str
) -> list[int]:
    seeds = [_record_seed(record, path, index) for index, record in enumerate(records)]
    duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if duplicates:
        raise ValueError(f"Duplicate episode seeds in {label} at {path}: {duplicates}")
    return seeds


def _require_boolean(record: dict[str, Any], field: str, path: Path, index: int) -> bool:
    if field not in record:
        raise ValueError(f"Missing {field!r} in {path} episode record {index}")
    value = record[field]
    if not isinstance(value, bool):
        raise ValueError(
            f"Expected boolean {field!r} in {path} episode record {index}, "
            f"got {value!r}"
        )
    return value


def _summary(records: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    successes = sum(
        _require_boolean(record, "success", path, index)
        for index, record in enumerate(records)
    )
    total = len(records)
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total,
    }


def _calibration_result(
    path: Path,
    name: str,
    manifest_path: Path,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    manifest = _read_json_object(manifest_path)
    if manifest.get("task") != name:
        raise ValueError(
            f"Calibration seed manifest task mismatch in {manifest_path}: "
            f"expected {name!r}, got {manifest.get('task')!r}"
        )
    if manifest.get("task_config") != "demo_clean":
        raise ValueError(f"Calibration seed manifest must use demo_clean: {manifest_path}")
    expected_seeds = _validate_seed_list(
        manifest.get("seeds"), label=f"{name} calibration seeds", path=manifest_path
    )
    records = _read_jsonl(path)
    record_seeds = _validate_unique_record_seeds(
        records, path, label=f"native calibration {name}"
    )
    if record_seeds != expected_seeds:
        raise ValueError(
            f"Strict seed sequence mismatch for native calibration {name} at {path}: "
            f"expected={expected_seeds}, actual={record_seeds}"
        )

    evaluation_plan = freeze.get("evaluation_plan_frozen_before_fastwam")
    if not isinstance(evaluation_plan, dict):
        raise ValueError("Freeze is missing evaluation_plan_frozen_before_fastwam")
    plan_key = (
        "native_downstream_calibration"
        if name == "place_can_basket"
        else "exact_block_reveal_calibration"
    )
    plan = evaluation_plan.get(plan_key)
    if not isinstance(plan, dict):
        raise ValueError(f"Freeze is missing calibration plan {plan_key!r}")
    if plan.get("task") != name or plan.get("seeds") != expected_seeds:
        raise ValueError(
            f"Frozen calibration plan does not match {manifest_path}: plan={plan!r}"
        )
    if plan.get("episodes") != len(expected_seeds):
        raise ValueError(f"Frozen calibration episode count is wrong for {name}")

    result = _summary(records, path)
    result["seeds"] = expected_seeds
    result["seed_manifest"] = str(manifest_path)
    result["episodes"] = [
        {
            "seed": record["seed"],
            "success": record["success"],
            "video_path": record.get("video_path"),
        }
        for record in records
    ]
    for index, record in enumerate(records):
        if record.get("task") != name:
            raise ValueError(
                f"Task mismatch for calibration {name} at {path} record {index}: "
                f"{record.get('task')!r}"
            )
        video_path = record.get("video_path")
        if (
            not isinstance(video_path, str)
            or not Path(video_path).is_file()
            or Path(video_path).stat().st_size <= 0
        ):
            raise FileNotFoundError(
                f"Calibration video missing or empty for {name} record {index}: "
                f"{video_path!r}"
            )
    return result


def _load_admission(
    path: Path,
    calibration: dict[str, dict[str, Any]],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    admission = _read_json_object(path)
    decision = admission.get("decision")
    if decision not in {"GO", "REJECT"}:
        raise ValueError(f"Invalid admission decision in {path}: {decision!r}")

    evaluation_plan = freeze.get("evaluation_plan_frozen_before_fastwam")
    if not isinstance(evaluation_plan, dict):
        raise ValueError("Freeze is missing evaluation_plan_frozen_before_fastwam")
    plan_keys = {
        "place_can_basket": "native_downstream_calibration",
        "move_block_reveal_can_block_calibration": "exact_block_reveal_calibration",
    }

    computed_go = True
    for name, plan_key in plan_keys.items():
        item = admission.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"Admission is missing {name!r} in {path}")
        rate = item.get("rate")
        threshold = item.get("threshold")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError(f"Invalid admission rate for {name} in {path}: {rate!r}")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                f"Invalid admission threshold for {name} in {path}: {threshold!r}"
            )
        frozen_plan = evaluation_plan.get(plan_key)
        if not isinstance(frozen_plan, dict):
            raise ValueError(f"Freeze is missing calibration plan {plan_key!r}")
        frozen_threshold = frozen_plan.get("admission_threshold")
        if (
            isinstance(frozen_threshold, bool)
            or not isinstance(frozen_threshold, (int, float))
            or abs(float(threshold) - float(frozen_threshold)) > 1e-12
        ):
            raise ValueError(
                f"Admission threshold does not match frozen plan for {name}: "
                f"admission={threshold}, frozen={frozen_threshold}"
            )
        observed_rate = calibration[name]["rate"]
        if abs(float(rate) - observed_rate) > 1e-12:
            raise ValueError(
                f"Admission/calibration rate mismatch for {name}: "
                f"admission={rate}, episodes={observed_rate}"
            )
        computed_go = computed_go and float(rate) >= float(threshold)

    expected_decision = "GO" if computed_go else "REJECT"
    if decision != expected_decision:
        raise ValueError(
            f"Admission decision mismatch in {path}: "
            f"recorded={decision}, recomputed={expected_decision}"
        )
    return admission


def _pilot_result(
    path: Path, expected_condition: str, manifest_seeds: list[int]
) -> dict[str, Any]:
    records = _read_jsonl(path)
    record_seeds = _validate_unique_record_seeds(
        records, path, label=f"pilot condition {expected_condition}"
    )
    if record_seeds != manifest_seeds:
        raise ValueError(
            f"Strict seed sequence mismatch for pilot condition {expected_condition} "
            f"at {path}: expected={manifest_seeds}, actual={record_seeds}"
        )
    episodes: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        missing_fields = [
            field
            for field in (*PILOT_FIELDS, *PILOT_VALIDATION_FIELDS)
            if field not in record
        ]
        if missing_fields:
            raise ValueError(
                f"Missing required fields {missing_fields} in {path} episode record {index}"
            )
        _require_boolean(record, "success", path, index)
        logged_condition = record.get("condition")
        if logged_condition != expected_condition:
            raise ValueError(
                f"Condition mismatch in {path} episode record {index}: "
                f"expected {expected_condition!r}, got {logged_condition!r}"
            )
        logged_task = record.get("task")
        if logged_task != TASK:
            raise ValueError(
                f"Task mismatch in {path} episode record {index}: "
                f"expected {TASK!r}, got {logged_task!r}"
            )
        video_path = record.get("video_path")
        if (
            not isinstance(video_path, str)
            or not Path(video_path).is_file()
            or Path(video_path).stat().st_size <= 0
        ):
            raise FileNotFoundError(
                f"Pilot video missing or empty in {path} episode record {index}: "
                f"{video_path!r}"
            )
        episodes.append({field: record[field] for field in PILOT_FIELDS})

    result = _summary(records, path)
    first_success = next((episode for episode in episodes if episode["success"]), None)
    first_failure = next((episode for episode in episodes if not episode["success"]), None)
    result.update(
        {
            "episodes": episodes,
            "typical_episodes": {
                "first_success": first_success,
                "first_failure": first_failure,
            },
        }
    )
    return result


def _max_state_delta(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf")
        return max(
            (_max_state_delta(left[key], right[key]) for key in left), default=0.0
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max(
            (_max_state_delta(lhs, rhs) for lhs, rhs in zip(left, right)),
            default=0.0,
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _without_block(snapshot: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(snapshot))
    result.pop("occluder_pose", None)
    result.pop("occluder_velocity", None)
    result["scene_actor_states"] = [
        actor
        for actor in result.get("scene_actor_states", [])
        if actor.get("name") != "minibench_occluder"
    ]
    return result


def _validate_paired_pilot(paths: dict[str, Path], seeds: list[int]) -> None:
    by_condition = {
        condition: _read_jsonl(paths[f"pilot/{condition}"])
        for condition in CONDITIONS
    }
    for index, seed in enumerate(seeds):
        rows = {condition: by_condition[condition][index] for condition in CONDITIONS}
        if any(int(row.get("seed", -1)) != seed for row in rows.values()):
            raise ValueError(f"Paired pilot seed mismatch at index {index}")
        if any(row.get("instruction") != "Put the can into the basket." for row in rows.values()):
            raise ValueError(f"Instruction mismatch in paired pilot seed {seed}")
        if any(bool(row.get("initial_success")) for row in rows.values()):
            raise ValueError(f"Pilot starts in success state for seed {seed}")
        if not (
            rows["oracle_reveal"].get("oracle_applied") is True
            and rows["visible"].get("oracle_applied") is False
            and rows["missing"].get("oracle_applied") is False
        ):
            raise ValueError(f"Oracle intervention flags are invalid for seed {seed}")

        pre = [rows[condition]["pre_intervention_state"] for condition in CONDITIONS]
        if any(_max_state_delta(pre[0], state) > PAIRED_STATE_ATOL for state in pre[1:]):
            raise ValueError(f"Full pre-intervention state mismatch for seed {seed}")
        visible_handoff = rows["visible"]["handoff_state"]
        oracle_handoff = rows["oracle_reveal"]["handoff_state"]
        if _max_state_delta(visible_handoff, oracle_handoff) > PAIRED_STATE_ATOL:
            raise ValueError(f"Visible/Oracle full handoff mismatch for seed {seed}")
        missing_handoff = rows["missing"]["handoff_state"]
        if (
            _max_state_delta(
                _without_block(visible_handoff), _without_block(missing_handoff)
            )
            > PAIRED_STATE_ATOL
        ):
            raise ValueError(f"Visible/Missing non-block handoff mismatch for seed {seed}")

        missing_pixels = rows["missing"]["handoff_target_pixels"]
        visible_pixels = rows["visible"]["handoff_target_pixels"]
        oracle_pixels = rows["oracle_reveal"]["handoff_target_pixels"]
        if any(int(count) != 0 for count in missing_pixels.values()):
            raise ValueError(f"Missing handoff exposes can for seed {seed}: {missing_pixels}")
        if not any(int(count) >= 32 for count in visible_pixels.values()):
            raise ValueError(f"Visible handoff hides can for seed {seed}: {visible_pixels}")
        if not any(int(count) >= 32 for count in oracle_pixels.values()):
            raise ValueError(f"Oracle handoff hides can for seed {seed}: {oracle_pixels}")


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    fieldnames = [
        "row_type",
        "group",
        "item",
        "condition",
        "successes",
        "total",
        "rate",
        *PILOT_FIELDS,
        "typical",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for name, item in result["calibration"].items():
            writer.writerow(
                {
                    "row_type": "summary",
                    "group": "calibration",
                    "item": name,
                    "condition": "native",
                    "successes": item["successes"],
                    "total": item["total"],
                    "rate": item["rate"],
                }
            )

        writer.writerow(
            {
                "row_type": "decision",
                "group": "admission",
                "item": "native_calibration_gate",
                "condition": result["admission"]["decision"],
                "typical": _json_cell(result["admission"]),
            }
        )

        for condition, item in result["pilot"].items():
            writer.writerow(
                {
                    "row_type": "summary",
                    "group": "pilot",
                    "item": TASK,
                    "condition": condition,
                    "successes": item["successes"],
                    "total": item["total"],
                    "rate": item["rate"],
                }
            )
            typical_success = item["typical_episodes"]["first_success"]
            typical_failure = item["typical_episodes"]["first_failure"]
            for episode in item["episodes"]:
                typical: list[str] = []
                if episode is typical_success:
                    typical.append("first_success")
                if episode is typical_failure:
                    typical.append("first_failure")
                row = {
                    "row_type": "episode",
                    "group": "pilot",
                    "item": TASK,
                    "condition": condition,
                    "successes": item["successes"],
                    "total": item["total"],
                    "rate": item["rate"],
                    "typical": ",".join(typical),
                }
                row.update({field: _json_cell(episode[field]) for field in PILOT_FIELDS})
                writer.writerow(row)


def _md_cell(value: Any) -> str:
    text = _json_cell(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _summary_rows(result: dict[str, Any]) -> Iterable[tuple[str, str, str, dict[str, Any]]]:
    for name, item in result["calibration"].items():
        yield "calibration", name, "native", item
    for condition, item in result["pilot"].items():
        yield "pilot", TASK, condition, item


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {TASK} aggregate results",
        "",
        "本报告只汇总原始实验记录，不应用通过门槛，也不推断结果是否符合预期。",
        "",
        f"- Run root: `{result['run_root']}`",
        f"- Freeze: `{result['frozen_protocol']['freeze_name']}`",
        f"- Task config: `{result['frozen_protocol']['task_config']}`",
        f"- Aggregate mode: `{result['aggregate_mode']}`",
        f"- Admission decision: `{result['admission']['decision']}`",
        "- Paired seed manifest: `"
        + _json_cell(result["frozen_protocol"]["seeds"])
        + "`",
        "",
        "## Summary",
        "",
        "| Group | Item | Condition | Successes | Total | Rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for group, item_name, condition, item in _summary_rows(result):
        lines.append(
            f"| {group} | {item_name} | {condition} | "
            f"{item['successes']} | {item['total']} | {item['rate']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Admission",
            "",
            "```json",
            json.dumps(result["admission"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Pilot episodes",
            "",
        ]
    )
    if not result["pilot"]:
        lines.extend(
            [
                "未运行：native calibration 的 admission 决定为 `REJECT`，",
                "因此按冻结协议停止在 calibration-only 结果。",
                "",
            ]
        )
    else:
        episode_headers = [*PILOT_FIELDS]
        for condition in CONDITIONS:
            item = result["pilot"][condition]
            lines.extend(
                [
                    f"### {condition}",
                    "",
                    "| " + " | ".join(episode_headers) + " |",
                    "|" + "|".join("---" for _ in episode_headers) + "|",
                ]
            )
            for episode in item["episodes"]:
                lines.append(
                    "| "
                    + " | ".join(_md_cell(episode[field]) for field in episode_headers)
                    + " |"
                )

            typical = item["typical_episodes"]
            lines.extend(
                [
                    "",
                    "Typical episodes (first occurrence in JSONL order):",
                    "",
                    "- First success: "
                    + (
                        "`null`"
                        if typical["first_success"] is None
                        else "`" + _json_cell(typical["first_success"]) + "`"
                    ),
                    "- First failure: "
                    + (
                        "`null`"
                        if typical["first_failure"] is None
                        else "`" + _json_cell(typical["first_failure"]) + "`"
                    ),
                    "",
                ]
            )

    lines.extend(["## Input files", ""])
    for name, source in result["sources"].items():
        lines.append(f"- `{name}`: `{source}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_admission(run_root: Path) -> Path:
    """Validate frozen calibration JSONL and atomically derive the gate result."""
    run_root = run_root.expanduser().resolve()
    paths = _input_paths(run_root)
    freeze, _, _ = _load_frozen_protocol(
        paths["freeze"], paths["seed_manifest"]
    )
    calibration = {
        name: _calibration_result(
            paths[f"calibration/{name}"],
            name,
            paths[f"calibration_manifest/{name}"],
            freeze,
        )
        for name, _, _ in CALIBRATION_SPECS
    }
    plan = freeze["evaluation_plan_frozen_before_fastwam"]
    plan_keys = {
        "place_can_basket": "native_downstream_calibration",
        "move_block_reveal_can_block_calibration": "exact_block_reveal_calibration",
    }
    payload = {}
    go = True
    for name, plan_key in plan_keys.items():
        threshold = float(plan[plan_key]["admission_threshold"])
        rate = float(calibration[name]["rate"])
        payload[name] = {
            "rate": rate,
            "threshold": threshold,
            "successes": int(calibration[name]["successes"]),
            "total": int(calibration[name]["total"]),
            "seeds": calibration[name]["seeds"],
        }
        go = go and rate >= threshold
    payload["decision"] = "GO" if go else "REJECT"
    path = paths["admission"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    # Re-read with the same strict validator used by aggregation and the pilot gate.
    _load_admission(path, calibration, freeze)
    return path


def aggregate(run_root: Path) -> Path:
    run_root = run_root.expanduser().resolve()
    paths = _input_paths(run_root)
    pilot_keys = {f"pilot/{condition}" for condition in CONDITIONS}
    required_paths = {
        name: path for name, path in paths.items() if name not in pilot_keys
    }
    missing_required = [path for path in required_paths.values() if not path.is_file()]
    if missing_required:
        formatted = "\n".join(f"  - {path}" for path in missing_required)
        raise FileNotFoundError(
            "Required move_block_reveal_can episode logs are missing:\n" + formatted
        )

    freeze, manifest, manifest_seeds = _load_frozen_protocol(
        paths["freeze"], paths["seed_manifest"]
    )
    calibration = {
        name: _calibration_result(
            paths[f"calibration/{name}"],
            name,
            paths[f"calibration_manifest/{name}"],
            freeze,
        )
        for name, _, _ in CALIBRATION_SPECS
    }
    admission = _load_admission(paths["admission"], calibration, freeze)
    present_pilot_conditions = [
        condition for condition in CONDITIONS if paths[f"pilot/{condition}"].is_file()
    ]
    if present_pilot_conditions and len(present_pilot_conditions) != len(CONDITIONS):
        missing_conditions = sorted(set(CONDITIONS) - set(present_pilot_conditions))
        raise FileNotFoundError(
            "Partial pilot output is not aggregatable: "
            f"present={present_pilot_conditions}, missing={missing_conditions}"
        )
    if present_pilot_conditions and admission["decision"] != "GO":
        raise ValueError(
            "Pilot outputs exist even though the frozen native calibration admission "
            "decision is REJECT"
        )
    if not present_pilot_conditions and admission["decision"] == "GO":
        formatted = "\n".join(
            f"  - {paths[f'pilot/{condition}']}" for condition in CONDITIONS
        )
        raise FileNotFoundError(
            "Admission is GO, so all frozen pilot outputs are required:\n" + formatted
        )

    if present_pilot_conditions:
        _validate_paired_pilot(paths, manifest_seeds)

    pilot = (
        {
            condition: _pilot_result(
                paths[f"pilot/{condition}"], condition, manifest_seeds
            )
            for condition in CONDITIONS
        }
        if present_pilot_conditions
        else {}
    )
    loaded_sources = dict(required_paths)
    loaded_sources.update(
        {
            f"pilot/{condition}": paths[f"pilot/{condition}"]
            for condition in present_pilot_conditions
        }
    )
    result = {
        "task": TASK,
        "run_root": str(run_root),
        "aggregate_mode": "full" if pilot else "calibration_only",
        "sources": {name: str(path) for name, path in loaded_sources.items()},
        "frozen_protocol": {
            "freeze_name": freeze["freeze_name"],
            "task_config": freeze["task_config"],
            "seeds": manifest_seeds,
            "manifest_protocol": manifest.get("protocol"),
        },
        "calibration": calibration,
        "admission": admission,
        "pilot": pilot,
    }

    output_dir = run_root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "results.csv", result)
    _write_report(output_dir / "report.md", result)
    return output_dir


def main() -> None:
    args = parse_args()
    if args.mode == "admission":
        path = write_admission(args.run_root)
        print(f"Wrote validated admission decision to {path}")
    else:
        output_dir = aggregate(args.run_root)
        print(f"Wrote aggregate results to {output_dir}")


if __name__ == "__main__":
    main()
