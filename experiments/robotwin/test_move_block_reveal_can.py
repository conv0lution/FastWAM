#!/usr/bin/env python3
"""Simulator-only validation and freeze tool for move_block_reveal_can.

This is intentionally task-specific.  It never loads FastWAM and never searches
for favorable seeds or geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import yaml
from PIL import Image


CONDITIONS = ("visible", "missing", "oracle_reveal")
FORMAL_FREEZE_NAME = "MINIBENCH_TASK_FREEZE_V0_3_11"
CALIBRATION_MANIFEST_NAMES = {
    "place_can_basket": "place_can_basket_calibration_seeds.json",
    "move_block_reveal_can_block_calibration": (
        "move_block_reveal_can_block_calibration_seeds.json"
    ),
}
# Physics solver ordering produces ~2e-5 residual robot-qvel differences even
# when qpos and camera/actor poses match.  This tolerance is fixed before model
# evaluation and remains far below any commanded motion.
STATE_ATOL = 1e-4
MIN_REPEATABILITY_ATTEMPTS = 2


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("reset", "expert", "calibration_expert", "freeze"),
    )
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=project_root / "third_party" / "RoboTwin",
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=Path(__file__).with_name("move_block_reveal_can_seeds.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in payload["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"Manifest must contain unique seeds: {path}")
    if payload.get("task") != "move_block_reveal_can":
        raise ValueError(f"Wrong task in seed manifest: {payload}")
    return payload, seeds


def load_calibration_manifests(project_root: Path):
    result = {}
    manifest_root = project_root / "experiments" / "robotwin"
    for task, filename in CALIBRATION_MANIFEST_NAMES.items():
        path = manifest_root / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        seeds = [int(seed) for seed in payload["seeds"]]
        if payload.get("task") != task:
            raise ValueError(
                f"Wrong task in calibration seed manifest {path}: {payload.get('task')!r}"
            )
        if payload.get("task_config") != "demo_clean":
            raise ValueError(f"Wrong task config in calibration seed manifest {path}")
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError(f"Calibration manifest must contain unique seeds: {path}")
        result[task] = {
            "path": path.resolve(),
            "payload": payload,
            "seeds": seeds,
        }
    return result


def install_robotwin(robotwin_root: Path):
    robotwin_root = robotwin_root.resolve()
    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    module = importlib.import_module("envs.move_block_reveal_can")
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(robotwin_root):
        raise RuntimeError(f"Task import escaped source of truth: {module_path}")
    return getattr(module, "move_block_reveal_can"), module_path


def import_task_class(robotwin_root: Path, task_name: str):
    module = importlib.import_module(f"envs.{task_name}")
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(robotwin_root):
        raise RuntimeError(f"Task import escaped source of truth: {module_path}")
    return getattr(module, task_name), module_path


def load_task_args(robotwin_root: Path):
    config_root = robotwin_root / "task_config"
    task_args = yaml.safe_load((config_root / "demo_clean.yml").read_text(encoding="utf-8"))
    embodiments = yaml.safe_load(
        (config_root / "_embodiment_config.yml").read_text(encoding="utf-8")
    )
    embodiment = task_args["embodiment"]
    if embodiment != ["aloha-agilex"]:
        raise ValueError(f"Expected Aloha dual arm, got {embodiment}")
    robot_path = (robotwin_root / embodiments[embodiment[0]]["file_path"]).resolve()
    robot_config = yaml.safe_load((robot_path / "config.yml").read_text(encoding="utf-8"))
    task_args.update(
        {
            "task_name": "move_block_reveal_can",
            "task_config": "demo_clean",
            "left_robot_file": str(robot_path),
            "right_robot_file": str(robot_path),
            "left_embodiment_config": deepcopy(robot_config),
            "right_embodiment_config": deepcopy(robot_config),
            "dual_arm_embodied": True,
            "eval_mode": True,
            "render_freq": 0,
            "eval_video_log": False,
        }
    )
    return task_args


def build(task_class, task_args, condition: str, seed: int, episode: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    task = task_class()
    task.setup_demo(
        now_ep_num=episode,
        seed=seed,
        is_test=True,
        minibench_condition=condition,
        **task_args,
    )
    return task


def capture(task, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    task._update_render()
    task.cameras.update_picture()
    rgb = task.cameras.get_rgb()
    actors = {"can": task.can, "basket": task.basket, "block": task.occluder}
    cameras = task._camera_map()
    actor_pixels = {name: {} for name in actors}

    for camera_name, camera in cameras.items():
        image = np.asarray(rgb[camera_name]["rgb"], dtype=np.uint8)
        labels = np.asarray(camera.get_picture("Segmentation"))[..., 1].astype(np.uint32)
        iio.imwrite(directory / f"{camera_name}_rgb.png", image)
        for actor_name, actor in actors.items():
            actor_id = int(actor.actor.get_per_scene_id())
            mask = labels == actor_id
            actor_pixels[actor_name][camera_name] = int(np.count_nonzero(mask))
            iio.imwrite(
                directory / f"{camera_name}_{actor_name}_mask.png",
                np.where(mask, 255, 0).astype(np.uint8),
            )

    head = np.asarray(
        Image.fromarray(np.asarray(rgb["head_camera"]["rgb"], dtype=np.uint8)).resize(
            (320, 256), Image.Resampling.BILINEAR
        )
    )
    left = np.asarray(
        Image.fromarray(np.asarray(rgb["left_camera"]["rgb"], dtype=np.uint8)).resize(
            (160, 128), Image.Resampling.BILINEAR
        )
    )
    right = np.asarray(
        Image.fromarray(np.asarray(rgb["right_camera"]["rgb"], dtype=np.uint8)).resize(
            (160, 128), Image.Resampling.BILINEAR
        )
    )
    iio.imwrite(
        directory / "fastwam_three_camera_rgb.png",
        np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0),
    )
    return actor_pixels


def max_delta(left: Any, right: Any, path="state"):
    mismatches = []
    maximum = 0.0
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf"), [f"{path}: keys differ"]
        for key in sorted(left):
            delta, children = max_delta(left[key], right[key], f"{path}.{key}")
            maximum = max(maximum, delta)
            mismatches.extend(children)
        return maximum, mismatches
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf"), [f"{path}: lengths differ"]
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            delta, children = max_delta(lhs, rhs, f"{path}[{index}]")
            maximum = max(maximum, delta)
            mismatches.extend(children)
        return maximum, mismatches
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        delta = abs(float(left) - float(right))
        if delta > STATE_ATOL:
            mismatches.append(f"{path}: {left} != {right} (abs={delta})")
        return delta, mismatches
    if left != right:
        return float("inf"), [f"{path}: {left!r} != {right!r}"]
    return 0.0, []


def without_block(snapshot):
    result = deepcopy(snapshot)
    result.pop("occluder_pose", None)
    result.pop("occluder_velocity", None)
    result["scene_actor_states"] = [
        actor
        for actor in result.get("scene_actor_states", [])
        if actor.get("name") != "minibench_occluder"
    ]
    return result


def success_bookkeeping_snapshot(task):
    """Small task-state snapshot used to prove check_success is read-only."""
    return {
        "physics_steps_after_handoff": int(
            task.minibench_physics_steps_after_handoff
        ),
        "success_streak_physics_steps": int(
            task.minibench_success_streak_physics_steps
        ),
        "max_success_streak_physics_steps": int(
            task.minibench_max_success_streak_physics_steps
        ),
        "policy_success_step": task.minibench_policy_success_step,
        "expert_success_streak": int(task.minibench_expert_success_streak),
        "eval_success": bool(task.eval_success),
        "take_action_count": int(task.take_action_cnt),
    }


def block_contact_partners(task):
    partners = set()
    block_name = task.occluder.get_name()
    for contact in task.scene.get_contacts():
        left = contact.bodies[0].entity.name
        right = contact.bodies[1].entity.name
        if left == block_name:
            partners.add(right)
        elif right == block_name:
            partners.add(left)
    return sorted(partners)


def occluder_handoff_metrics(task):
    actual = task.occluder.get_pose()
    expected = (
        task.occluder_occluding_pose
        if task.minibench_condition == "missing"
        else task.occluder_revealed_pose
    )
    # Use the task's own snapshot so this remains independent of component
    # ordering and records exactly the state used by the paired-state audit.
    velocity = task._state_snapshot()["occluder_velocity"]
    pose_error = float(np.linalg.norm(np.asarray(actual.p) - np.asarray(expected.p)))
    quat_error = task._quaternion_angle(actual.q, expected.q)
    up_z = float(actual.to_transformation_matrix()[2, 2])
    return {
        "position_error_m": pose_error,
        "rotation_error_rad": quat_error,
        "up_z": up_z,
        "linear_speed_mps": float(np.linalg.norm(velocity["linear"])),
        "angular_speed_radps": float(np.linalg.norm(velocity["angular"])),
    }


def run_reset(task_class, task_args, seeds, output_dir, source_fingerprint):
    records = []
    failures = []
    for episode, seed in enumerate(seeds):
        by_condition = {}
        for condition in CONDITIONS:
            task = None
            try:
                task = build(task_class, task_args, condition, seed, episode)
                pre_pixels = capture(
                    task,
                    output_dir / "screenshots" / f"seed_{seed}" / condition / "pre_intervention",
                )
                task.prepare_policy_handoff()
                handoff_obs = task.get_obs()
                del handoff_obs
                success_bookkeeping_before_queries = success_bookkeeping_snapshot(task)
                repeated_success_queries = [
                    bool(task.check_success()) for _ in range(100)
                ]
                success_bookkeeping_after_queries = success_bookkeeping_snapshot(task)
                handoff_pixels = capture(
                    task,
                    output_dir / "screenshots" / f"seed_{seed}" / condition / "policy_handoff",
                )
                by_condition[condition] = {
                    "record": task.get_minibench_record(refresh_visibility=False),
                    "pre_actor_pixels": pre_pixels,
                    "handoff_actor_pixels": handoff_pixels,
                    "block_contact_partners_at_handoff": block_contact_partners(task),
                    "occluder_handoff_metrics": occluder_handoff_metrics(task),
                    "robot_link_names": sorted(
                        {
                            link.get_name()
                            for link in (
                                task.robot.left_entity.get_links()
                                + task.robot.right_entity.get_links()
                            )
                        }
                    ),
                    "block_contacts_can_at_handoff": bool(
                        task.check_actors_contact("minibench_occluder", "071_can")
                    ),
                    "repeated_success_queries": repeated_success_queries,
                    "success_bookkeeping_before_queries": (
                        success_bookkeeping_before_queries
                    ),
                    "success_bookkeeping_after_queries": (
                        success_bookkeeping_after_queries
                    ),
                }
            except Exception as exc:
                by_condition[condition] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            finally:
                if task is not None:
                    task.close_env()

        checks = {}
        if all("error" not in by_condition[name] for name in CONDITIONS):
            visible = by_condition["visible"]
            missing = by_condition["missing"]
            oracle = by_condition["oracle_reveal"]
            visible_record = visible["record"]
            missing_record = missing["record"]
            oracle_record = oracle["record"]

            nonblock_vm_delta, nonblock_vm_mismatch = max_delta(
                without_block(visible_record["pre_intervention_state"]),
                without_block(missing_record["pre_intervention_state"]),
            )
            nonblock_vo_delta, nonblock_vo_mismatch = max_delta(
                without_block(visible_record["pre_intervention_state"]),
                without_block(oracle_record["pre_intervention_state"]),
            )
            missing_oracle_delta, missing_oracle_mismatch = max_delta(
                missing_record["pre_intervention_state"],
                oracle_record["pre_intervention_state"],
            )
            visible_oracle_delta, visible_oracle_mismatch = max_delta(
                visible_record["handoff_state"], oracle_record["handoff_state"]
            )
            visible_missing_handoff_delta, visible_missing_handoff_mismatch = max_delta(
                without_block(visible_record["handoff_state"]),
                without_block(missing_record["handoff_state"]),
            )
            full_pre_vm_delta, full_pre_vm_mismatch = max_delta(
                visible_record["pre_intervention_state"],
                missing_record["pre_intervention_state"],
            )
            threshold = task_class.VISIBLE_PIXEL_THRESHOLD

            checks = {
                "all_conditions_reset": True,
                "all_initial_success_false": all(
                    not by_condition[name]["record"]["initial_success"] for name in CONDITIONS
                ),
                "no_block_can_contact": all(
                    not by_condition[name]["block_contacts_can_at_handoff"] for name in CONDITIONS
                ),
                "block_contacts_only_table_at_handoff": all(
                    set(by_condition[name]["block_contact_partners_at_handoff"])
                    <= {"table"}
                    for name in CONDITIONS
                ),
                "block_contacts_no_robot_links_at_handoff": all(
                    not (
                        set(by_condition[name]["block_contact_partners_at_handoff"])
                        & set(by_condition[name]["robot_link_names"])
                    )
                    for name in CONDITIONS
                ),
                "block_canonical_stable_upright_at_handoff": all(
                    by_condition[name]["occluder_handoff_metrics"]["position_error_m"]
                    <= 0.005
                    and by_condition[name]["occluder_handoff_metrics"]["rotation_error_rad"]
                    <= np.deg2rad(3.0)
                    and by_condition[name]["occluder_handoff_metrics"]["up_z"] >= 0.98
                    and by_condition[name]["occluder_handoff_metrics"]["linear_speed_mps"]
                    <= 0.01
                    and by_condition[name]["occluder_handoff_metrics"]["angular_speed_radps"]
                    <= 0.1
                    for name in CONDITIONS
                ),
                "missing_can_zero_all_cameras": all(
                    count == 0
                    for count in missing["handoff_actor_pixels"]["can"].values()
                ),
                "oracle_pre_can_zero_all_cameras": all(
                    count == 0 for count in oracle["pre_actor_pixels"]["can"].values()
                ),
                "visible_can_visible": any(
                    count >= threshold
                    for count in visible["handoff_actor_pixels"]["can"].values()
                ),
                "oracle_can_visible": any(
                    count >= threshold
                    for count in oracle["handoff_actor_pixels"]["can"].values()
                ),
                "basket_visible_all_conditions": all(
                    by_condition[name]["handoff_actor_pixels"]["basket"]["head_camera"]
                    >= threshold
                    for name in CONDITIONS
                ),
                "block_visible_all_conditions": all(
                    any(
                        count >= threshold
                        for count in by_condition[name]["handoff_actor_pixels"]["block"].values()
                    )
                    for name in CONDITIONS
                ),
                "visible_missing_nonblock_state_equal": nonblock_vm_delta <= STATE_ATOL,
                "all_conditions_full_pre_state_equal": full_pre_vm_delta <= STATE_ATOL
                and missing_oracle_delta <= STATE_ATOL,
                "visible_missing_nonblock_handoff_state_equal": (
                    visible_missing_handoff_delta <= STATE_ATOL
                ),
                "visible_oracle_nonblock_pre_state_equal": nonblock_vo_delta <= STATE_ATOL,
                "missing_oracle_full_pre_state_equal": missing_oracle_delta <= STATE_ATOL,
                "visible_oracle_full_handoff_state_equal": visible_oracle_delta <= STATE_ATOL,
                "oracle_only_intervention_applied": (
                    not visible_record["oracle_applied"]
                    and not missing_record["oracle_applied"]
                    and bool(oracle_record["oracle_applied"])
                ),
                "all_handoff_success_false": all(
                    not by_condition[name]["record"]["success"] for name in CONDITIONS
                ),
                "success_query_is_pure": all(
                    by_condition[name]["success_bookkeeping_before_queries"]
                    == by_condition[name]["success_bookkeeping_after_queries"]
                    and len(set(by_condition[name]["repeated_success_queries"])) == 1
                    for name in CONDITIONS
                ),
            }
            diagnostics = {
                "visible_missing_nonblock_max_delta": nonblock_vm_delta,
                "visible_missing_nonblock_mismatches": nonblock_vm_mismatch,
                "visible_oracle_nonblock_pre_max_delta": nonblock_vo_delta,
                "visible_oracle_nonblock_pre_mismatches": nonblock_vo_mismatch,
                "missing_oracle_pre_max_delta": missing_oracle_delta,
                "missing_oracle_pre_mismatches": missing_oracle_mismatch,
                "visible_oracle_handoff_max_delta": visible_oracle_delta,
                "visible_oracle_handoff_mismatches": visible_oracle_mismatch,
                "visible_missing_handoff_nonblock_max_delta": visible_missing_handoff_delta,
                "visible_missing_handoff_nonblock_mismatches": visible_missing_handoff_mismatch,
                "visible_missing_full_pre_max_delta": full_pre_vm_delta,
                "visible_missing_full_pre_mismatches": full_pre_vm_mismatch,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                failures.append(f"seed {seed}: {', '.join(failed)}")
        else:
            diagnostics = {}
            failures.append(f"seed {seed}: reset error")
        records.append(
            {
                "seed": seed,
                "conditions": by_condition,
                "checks": checks,
                "diagnostics": diagnostics,
            }
        )

    screenshot_files = sorted((output_dir / "screenshots").rglob("*.png"), key=str)
    expected_screenshot_count = len(seeds) * len(CONDITIONS) * 2 * 13
    if len(screenshot_files) != expected_screenshot_count:
        failures.append(
            "screenshot count mismatch: "
            f"actual={len(screenshot_files)} expected={expected_screenshot_count}"
        )
    screenshot_hashes = {
        str(path.resolve()): sha256(path) for path in screenshot_files
    }
    screenshot_manifest_path = output_dir / "screenshots.sha256"
    screenshot_manifest_path.write_text(
        "".join(
            f"{digest}  {path}\n" for path, digest in screenshot_hashes.items()
        ),
        encoding="utf-8",
    )
    payload = {
        "mode": "reset",
        "task": "move_block_reveal_can",
        "task_config": "demo_clean",
        "state_atol": STATE_ATOL,
        "seeds": seeds,
        "source_fingerprint": source_fingerprint,
        "passed": not failures,
        "failures": failures,
        "screenshot_count": len(screenshot_files),
        "screenshot_sha256": screenshot_hashes,
        "screenshot_manifest": str(screenshot_manifest_path.resolve()),
        "records": records,
    }
    write_json(output_dir / "reset_audit.json", payload)
    if failures:
        raise RuntimeError("Reset audit failed: " + "; ".join(failures))
    return payload


def run_expert(task_class, task_args, seeds, output_dir, source_fingerprint):
    records = []
    failures = []
    cells_without_witness = []
    max_attempts = int(getattr(task_class, "SCRIPTED_EXPERT_MAX_ATTEMPTS", 1))
    if max_attempts < 1:
        raise ValueError("SCRIPTED_EXPERT_MAX_ATTEMPTS must be positive")
    for episode, seed in enumerate(seeds):
        for condition in CONDITIONS:
            attempts = []
            witness = None
            for attempt_index in range(max_attempts):
                task = None
                try:
                    # Every attempt reconstructs the exact same fixed-seed scene.
                    # No failed seed is substituted and no task parameter changes.
                    task = build(task_class, task_args, condition, seed, episode)
                    pre_pose = task._pose_list(task.occluder.get_pose())
                    task.prepare_policy_handoff()
                    handoff_pose = task._pose_list(task.occluder.get_pose())
                    task.play_once()
                    final_counts = task.get_target_pixel_counts()
                    success_bookkeeping_before_queries = (
                        success_bookkeeping_snapshot(task)
                    )
                    repeated_success_queries = [
                        bool(task.check_success()) for _ in range(100)
                    ]
                    success_bookkeeping_after_queries = (
                        success_bookkeeping_snapshot(task)
                    )
                    record = task.get_minibench_record(refresh_visibility=False)
                    native_parent_completion_stages = [
                        stage
                        for stage in record.get("expert_trace", [])
                        if stage.get("stage") == "can_native_parent_complete"
                    ]
                    audited_expert_path = bool(
                        record.get("native_parent_expert_completed") is True
                        and len(native_parent_completion_stages) == 1
                    )
                    query_is_pure = bool(
                        success_bookkeeping_before_queries
                        == success_bookkeeping_after_queries
                        and len(set(repeated_success_queries)) == 1
                    )
                    final_success = bool(repeated_success_queries[0])
                    expert_stability_verified = bool(
                        success_bookkeeping_before_queries[
                            "success_streak_physics_steps"
                        ]
                        >= task.SUCCESS_STABLE_PHYSICS_STEPS
                    )
                    passed = bool(
                        task.plan_success
                        and final_success
                        and query_is_pure
                        and expert_stability_verified
                        and audited_expert_path
                    )
                    attempt_row = {
                        "attempt_index": attempt_index,
                        "seed": seed,
                        "condition": condition,
                        "passed": passed,
                        "plan_success": bool(task.plan_success),
                        "final_success": final_success,
                        "success_query_is_pure": query_is_pure,
                        "success_bookkeeping_before_queries": (
                            success_bookkeeping_before_queries
                        ),
                        "success_bookkeeping_after_queries": (
                            success_bookkeeping_after_queries
                        ),
                        "expert_stability_verified": expert_stability_verified,
                        "audited_expert_path": audited_expert_path,
                        "block_pose_pre_intervention": pre_pose,
                        "block_pose_at_handoff": handoff_pose,
                        "block_pose_final": task._pose_list(task.occluder.get_pose()),
                        "final_target_pixels": final_counts,
                        "record": record,
                    }
                except Exception as exc:
                    debug_state = None
                    if task is not None:
                        try:
                            debug_state = {
                                "plan_success": bool(task.plan_success),
                                "block_pose_final": task._pose_list(
                                    task.occluder.get_pose()
                                ),
                                "target_pixels_final": task.get_target_pixel_counts(),
                                "minibench_record": task.get_minibench_record(
                                    refresh_visibility=False
                                ),
                            }
                        except Exception as debug_exc:
                            debug_state = {
                                "capture_error": f"{type(debug_exc).__name__}: {debug_exc}"
                            }
                    attempt_row = {
                        "attempt_index": attempt_index,
                        "seed": seed,
                        "condition": condition,
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "debug_state": debug_state,
                    }
                finally:
                    if task is not None:
                        task.close_env()
                attempts.append(attempt_row)
                if attempt_row["passed"] and witness is None:
                    witness = attempt_row
                if witness is not None and len(attempts) >= MIN_REPEATABILITY_ATTEMPTS:
                    break

            attempt_state_deltas = []
            repeatability_passed = True
            reference_record = None
            for attempt in attempts:
                attempt_record = attempt.get("record")
                if attempt_record is None:
                    attempt_record = (attempt.get("debug_state") or {}).get(
                        "minibench_record"
                    )
                if not attempt_record:
                    repeatability_passed = False
                    attempt_state_deltas.append(
                        {
                            "attempt_index": attempt["attempt_index"],
                            "error": "missing minibench record for repeatability audit",
                        }
                    )
                    continue
                if reference_record is None:
                    reference_record = attempt_record
                    continue
                pre_delta, pre_mismatches = max_delta(
                    reference_record["pre_intervention_state"],
                    attempt_record["pre_intervention_state"],
                    path="repeat.pre_intervention_state",
                )
                handoff_delta, handoff_mismatches = max_delta(
                    reference_record["handoff_state"],
                    attempt_record["handoff_state"],
                    path="repeat.handoff_state",
                )
                attempt_state_deltas.append(
                    {
                        "attempt_index": attempt["attempt_index"],
                        "pre_state_max_delta": pre_delta,
                        "pre_state_mismatches": pre_mismatches,
                        "handoff_state_max_delta": handoff_delta,
                        "handoff_state_mismatches": handoff_mismatches,
                    }
                )
                repeatability_passed = bool(
                    repeatability_passed
                    and pre_delta <= STATE_ATOL
                    and handoff_delta <= STATE_ATOL
                )

            selected = witness if witness is not None else attempts[-1]
            row = dict(selected)
            row.update(
                {
                    "passed": witness is not None and repeatability_passed,
                    "attempt_count": len(attempts),
                    "max_attempts": max_attempts,
                    "minimum_repeatability_attempts": MIN_REPEATABILITY_ATTEMPTS,
                    "repeatability_comparison_count": len(attempt_state_deltas),
                    "repeatability_passed": repeatability_passed,
                    "attempt_state_deltas": attempt_state_deltas,
                    "witness_attempt_index": (
                        int(witness["attempt_index"]) if witness is not None else None
                    ),
                    "attempts": attempts,
                }
            )
            if witness is None:
                cells_without_witness.append(
                    f"seed {seed} condition {condition}: no complete expert witness "
                    f"in {max_attempts} fixed-seed attempts"
                )
            elif not repeatability_passed:
                failures.append(
                    f"seed {seed} condition {condition}: same-seed attempt state "
                    "repeatability audit failed"
                )
            records.append(row)

    all_attempts = [attempt for row in records for attempt in row["attempts"]]
    first_attempt_successes = sum(
        int(bool(row["attempts"][0]["passed"])) for row in records
    )
    condition_witness_coverage = {
        condition: sum(
            int(bool(row["passed"]))
            for row in records
            if row["condition"] == condition
        )
        for condition in CONDITIONS
    }
    for condition, witness_count in condition_witness_coverage.items():
        if witness_count < 1:
            failures.append(
                f"condition {condition}: no complete scene-solvability witness"
            )
    payload = {
        "mode": "expert",
        "task": "move_block_reveal_can",
        "task_config": "demo_clean",
        "seeds": seeds,
        "source_fingerprint": source_fingerprint,
        "expert_attempt_protocol": {
            "max_attempts_per_seed_condition": max_attempts,
            "fresh_scene_each_attempt": True,
            "seed_substitution": False,
            "task_parameter_changes_between_attempts": False,
            "pre_and_handoff_state_repeatability_asserted": True,
            "minimum_repeatability_attempts": MIN_REPEATABILITY_ATTEMPTS,
            "pass_semantics": (
                "at least one complete scene-solvability witness per condition, "
                "with every fixed-state reconstruction passing repeatability checks; "
                "per-cell and per-attempt outcomes remain raw diagnostics"
            ),
        },
        "passed": not failures,
        "successes": sum(int(row["passed"]) for row in records),
        "total": len(records),
        "first_attempt_successes": first_attempt_successes,
        "first_attempt_total": len(records),
        "all_attempt_successes": sum(
            int(bool(attempt["passed"])) for attempt in all_attempts
        ),
        "all_attempt_total": len(all_attempts),
        "condition_witness_coverage": condition_witness_coverage,
        "cells_without_witness": cells_without_witness,
        "failures": failures,
        "records": records,
    }
    write_json(output_dir / "expert_audit.json", payload)
    if failures:
        raise RuntimeError("Expert audit failed: " + "; ".join(failures))
    return payload


def run_calibration_expert(
    project_root, robotwin_root, task_args, output_dir, source_fingerprint
):
    manifests = load_calibration_manifests(project_root)
    evidence_path = (
        project_root
        / "experiments"
        / "robotwin"
        / "move_block_reveal_can_expert_solvability_audit.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    scene_paths = {
        "place_can_basket_source": robotwin_root / "envs" / "place_can_basket.py",
        "move_block_reveal_can_source": robotwin_root
        / "envs"
        / "move_block_reveal_can.py",
        "block_calibration_source": robotwin_root
        / "envs"
        / "move_block_reveal_can_block_calibration.py",
        "demo_clean_config": robotwin_root / "task_config" / "demo_clean.yml",
        "place_can_seed_manifest": manifests["place_can_basket"]["path"],
        "block_seed_manifest": manifests[
            "move_block_reveal_can_block_calibration"
        ]["path"],
    }
    current_scene_hashes = {
        name: sha256(path) for name, path in scene_paths.items()
    }
    evidence_failures = []
    if evidence.get("calibration_scene_defining_sha256") != current_scene_hashes:
        evidence_failures.append(
            "calibration scene-defining hashes do not match the pinned "
            "cross-run solvability evidence"
        )
    for run_name in (
        "v0_3_10_calibration_expert",
        "v0_3_11_calibration_expert_raw_172000",
    ):
        raw = evidence.get("raw_runs", {}).get(run_name, {})
        raw_path = Path(str(raw.get("path", "")))
        if (
            not raw_path.is_file()
            or not raw.get("sha256")
            or sha256(raw_path) != raw.get("sha256")
        ):
            evidence_failures.append(
                f"missing or hash-mismatched raw calibration evidence: {run_name}"
            )
    cross_run_witnesses = evidence.get(
        "cross_run_calibration_scene_solvability_witnesses", {}
    )
    expected_cross_run_pairs = {
        (task_name, int(seed))
        for task_name, manifest in manifests.items()
        for seed in manifest["seeds"]
    }
    actual_cross_run_pairs = {
        (str(task_name), int(seed))
        for task_name, task_witnesses in cross_run_witnesses.items()
        for seed, witness in task_witnesses.items()
        if str(witness).strip()
    }
    if actual_cross_run_pairs != expected_cross_run_pairs:
        evidence_failures.append(
            "cross-run calibration scene-solvability coverage is incomplete or extra"
        )

    records = []
    raw_failures = []
    for task_name, manifest in manifests.items():
        task_class, module_path = import_task_class(robotwin_root, task_name)
        named_args = deepcopy(task_args)
        named_args["task_name"] = task_name
        # One non-adaptive attempt is retained for every fixed calibration
        # scene in this run. Scene solvability is established by the pinned
        # cross-run witness audit, rather than rerunning until a desired result
        # appears.
        max_attempts = 1
        for episode, seed in enumerate(manifest["seeds"]):
            attempts = []
            witness = None
            for attempt_index in range(max_attempts):
                task = None
                try:
                    condition = (
                        "missing"
                        if task_name == "move_block_reveal_can_block_calibration"
                        else "visible"
                    )
                    task = build(task_class, named_args, condition, seed, episode)
                    handoff = getattr(task, "prepare_policy_handoff", None)
                    if handoff is not None:
                        handoff()
                    task.play_once()
                    passed = bool(task.plan_success and task.check_success())
                    row = {
                        "attempt_index": attempt_index,
                        "passed": passed,
                        "plan_success": bool(task.plan_success),
                        "final_success": bool(task.check_success()),
                    }
                    if hasattr(task, "get_minibench_record"):
                        row["record"] = task.get_minibench_record(
                            refresh_visibility=False
                        )
                except Exception as exc:
                    row = {
                        "attempt_index": attempt_index,
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                finally:
                    if task is not None:
                        task.close_env()
                attempts.append(row)
                if row["passed"]:
                    witness = row
                    break
            record = {
                "task": task_name,
                "task_module": str(module_path),
                "seed": seed,
                "passed": witness is not None,
                "current_run_witness": witness is not None,
                "cross_run_witness": cross_run_witnesses.get(task_name, {}).get(
                    str(seed)
                ),
                "scene_solvability_witness": bool(
                    cross_run_witnesses.get(task_name, {}).get(str(seed))
                ),
                "witness_attempt_index": (
                    int(witness["attempt_index"]) if witness is not None else None
                ),
                "max_attempts": max_attempts,
                "attempts": attempts,
            }
            records.append(record)
            if witness is None:
                raw_failures.append(
                    f"{task_name} seed {seed}: no scripted expert witness in "
                    f"{max_attempts} fixed-seed attempts"
                )
    scene_solvability_successes = sum(
        int(bool(row["scene_solvability_witness"])) for row in records
    )
    if scene_solvability_successes != len(records):
        evidence_failures.append(
            "not every current fixed calibration scene has a pinned solvability witness"
        )
    payload = {
        "mode": "calibration_expert",
        "task_config": "demo_clean",
        "source_fingerprint": source_fingerprint,
        "attempt_protocol": {
            "attempts_per_fixed_scene": 1,
            "adaptive_retry": False,
            "seed_substitution": False,
            "raw_outcomes_are_not_the_freeze_gate": True,
        },
        "scene_defining_sha256": current_scene_hashes,
        "cross_run_evidence_path": str(evidence_path.resolve()),
        "cross_run_evidence_sha256": sha256(evidence_path),
        "manifests": {
            name: {
                "path": str(value["path"]),
                "seeds": value["seeds"],
            }
            for name, value in manifests.items()
        },
        "passed": not evidence_failures,
        "successes": sum(int(row["passed"]) for row in records),
        "total": len(records),
        "scene_solvability_successes": scene_solvability_successes,
        "scene_solvability_total": len(records),
        "raw_failures": raw_failures,
        "failures": evidence_failures,
        "records": records,
    }
    write_json(output_dir / "calibration_expert_audit.json", payload)
    if evidence_failures:
        raise RuntimeError(
            "Calibration expert evidence audit failed: "
            + "; ".join(evidence_failures)
        )
    return payload


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_files(root: Path, *, suffixes=None, excluded_dir_names=()):
    root = root.resolve()
    result = []
    excluded = set(excluded_dir_names)
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if name not in excluded and name != "__pycache__"
        )
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if suffixes is None or path.suffix.lower() in suffixes:
                result.append(path.resolve())
    return result


def runtime_asset_files(robotwin_root: Path):
    roots = (
        robotwin_root / "assets" / "objects" / "071_can",
        robotwin_root / "assets" / "objects" / "110_basket",
        robotwin_root / "assets" / "embodiments" / "aloha-agilex",
    )
    files = [path for root in roots for path in _tree_files(root)]
    if not files:
        raise RuntimeError("Required can/basket/Aloha runtime assets are missing")
    return sorted(set(files), key=str)


def simulator_source_files(project_root, robotwin_root):
    # Hash the complete small RoboTwin source/config/description tree so a
    # transitive helper change cannot be paired with stale reset/expert audits.
    files = _tree_files(
        robotwin_root,
        suffixes={".py", ".json", ".yml", ".yaml"},
        excluded_dir_names={".git", "assets", "policy", "results", "result"},
    )
    files.extend(
        [
            project_root / "configs" / "sim_robotwin.yaml",
            project_root / "experiments" / "robotwin" / "eval_robotwin_single.py",
            project_root / "experiments" / "robotwin" / "fastwam_policy" / "__init__.py",
            project_root / "experiments" / "robotwin" / "fastwam_policy" / "deploy_policy.py",
            project_root / "experiments" / "robotwin" / "fastwam_policy" / "deploy_policy.yml",
            project_root / "experiments" / "robotwin" / "move_block_reveal_can_seeds.json",
            project_root
            / "experiments"
            / "robotwin"
            / CALIBRATION_MANIFEST_NAMES["place_can_basket"],
            project_root
            / "experiments"
            / "robotwin"
            / CALIBRATION_MANIFEST_NAMES[
                "move_block_reveal_can_block_calibration"
            ],
            project_root
            / "experiments"
            / "robotwin"
            / "move_block_reveal_can_insertion_geometry_audit.json",
            project_root
            / "experiments"
            / "robotwin"
            / "move_block_reveal_can_transfer_geometry_audit.json",
            project_root
            / "experiments"
            / "robotwin"
            / "move_block_reveal_can_native_parent_expert_audit.json",
            project_root
            / "experiments"
            / "robotwin"
            / "move_block_reveal_can_success_predicate_audit.json",
            project_root
            / "experiments"
            / "robotwin"
            / "move_block_reveal_can_expert_solvability_audit.json",
            project_root
            / "experiments"
            / "robotwin"
            / "MINIBENCH_TASK_FREEZE_V0_3_11.json",
            project_root
            / "experiments"
            / "robotwin"
            / "MINIBENCH_TASK_FREEZE_V0_3_11.sha256",
            project_root / "experiments" / "robotwin" / "test_move_block_reveal_can.py",
            project_root
            / "experiments"
            / "robotwin"
            / "summarize_move_block_reveal_can.py",
            project_root / "scripts" / "run_move_block_reveal_can.sh",
        ]
    )
    return sorted(set(path.resolve() for path in files), key=str)


def simulator_source_fingerprint(project_root, robotwin_root):
    return {
        str(path): sha256(path)
        for path in (
            simulator_source_files(project_root, robotwin_root)
            + runtime_asset_files(robotwin_root)
        )
    }


def run_freeze(project_root, robotwin_root, task_class, manifest, output_dir):
    reset_path = output_dir / "simulator" / "reset_audit.json"
    expert_path = output_dir / "simulator" / "expert_audit.json"
    calibration_expert_path = (
        output_dir / "simulator" / "calibration_expert_audit.json"
    )
    provenance_path = output_dir / "metadata" / "simulator_provenance.json"
    expected_seeds = [int(seed) for seed in manifest["seeds"]]
    calibration_manifests = load_calibration_manifests(project_root)
    current_source_fingerprint = simulator_source_fingerprint(project_root, robotwin_root)
    for path, expected_mode in ((reset_path, "reset"), (expert_path, "expert")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if not result.get("passed"):
            raise RuntimeError(f"Cannot freeze: simulator audit did not pass: {path}")
        expected_fields = {
            "mode": expected_mode,
            "task": "move_block_reveal_can",
            "task_config": "demo_clean",
            "seeds": expected_seeds,
            "source_fingerprint": current_source_fingerprint,
        }
        mismatches = {
            key: {"expected": value, "actual": result.get(key)}
            for key, value in expected_fields.items()
            if result.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Cannot freeze stale/mismatched simulator audit {path}: {mismatches}"
            )
    reset_result = json.loads(reset_path.read_text(encoding="utf-8"))
    reset_records = reset_result.get("records", [])
    reset_record_seeds = [int(row.get("seed")) for row in reset_records]
    reset_screenshot_paths = sorted(
        (output_dir / "simulator" / "screenshots").rglob("*.png"), key=str
    )
    current_screenshot_hashes = {
        str(path.resolve()): sha256(path) for path in reset_screenshot_paths
    }
    if (
        len(reset_records) != len(expected_seeds)
        or reset_record_seeds != expected_seeds
        or len(reset_record_seeds) != len(set(reset_record_seeds))
        or any(
            set(row.get("conditions", {})) != set(CONDITIONS)
            or any(
                "error" in row["conditions"][condition]
                for condition in CONDITIONS
            )
            or not row.get("checks")
            or not all(bool(value) for value in row["checks"].values())
            for row in reset_records
        )
        or reset_result.get("screenshot_sha256") != current_screenshot_hashes
        or int(reset_result.get("screenshot_count", -1))
        != len(current_screenshot_hashes)
        or len(current_screenshot_hashes)
        != len(expected_seeds) * len(CONDITIONS) * 2 * 13
    ):
        raise RuntimeError(
            "Cannot freeze incomplete/stale reset audit or screenshots: "
            f"record_seeds={reset_record_seeds}, expected={expected_seeds}, "
            f"screenshot_count={len(current_screenshot_hashes)}"
        )
    expert_result = json.loads(expert_path.read_text(encoding="utf-8"))
    expected_expert_total = len(expected_seeds) * len(CONDITIONS)
    expected_pairs = {
        (int(seed), condition) for seed in expected_seeds for condition in CONDITIONS
    }
    actual_pairs = [
        (int(row["seed"]), str(row["condition"]))
        for row in expert_result.get("records", [])
    ]
    witness_rows = [
        row for row in expert_result.get("records", []) if row.get("passed")
    ]
    computed_condition_coverage = {
        condition: sum(
            int(bool(row.get("passed")))
            for row in expert_result.get("records", [])
            if row.get("condition") == condition
        )
        for condition in CONDITIONS
    }
    if (
        expert_result.get("passed") is not True
        or expert_result.get("total") != expected_expert_total
        or expert_result.get("successes") != len(witness_rows)
        or len(actual_pairs) != len(set(actual_pairs))
        or set(actual_pairs) != expected_pairs
        or expert_result.get("condition_witness_coverage")
        != computed_condition_coverage
        or any(value < 1 for value in computed_condition_coverage.values())
        or any(
            int(row.get("repeatability_comparison_count", 0)) < 1
            or not row.get("repeatability_passed")
            for row in expert_result.get("records", [])
        )
        or any(
            row.get("witness_attempt_index") is None
            or not row.get("success_query_is_pure")
            or not row.get("expert_stability_verified")
            or not row.get("audited_expert_path")
            or row.get("record", {}).get("native_parent_expert_completed") is not True
            or (
                row.get("condition") == "missing"
                and float(
                    row.get("record", {}).get(
                        "expert_reveal_max_can_displacement_m", float("inf")
                    )
                )
                > task_class.EXPERT_REVEAL_MAX_CAN_DISPLACEMENT_M
            )
            or (
                row.get("condition") == "missing"
                and float(
                    row.get("record", {}).get(
                        "expert_reveal_max_can_rotation_rad", float("inf")
                    )
                )
                > task_class.EXPERT_REVEAL_MAX_CAN_ROTATION_RAD
            )
            for row in witness_rows
        )
    ):
        raise RuntimeError(
            "Cannot freeze incomplete expert audit: "
            f"successes={expert_result.get('successes')} total={expert_result.get('total')} "
            f"expected_cells={expected_expert_total} "
            f"condition_coverage={computed_condition_coverage} "
            f"actual_pairs={actual_pairs}"
        )
    expected_attempt_protocol = {
        "max_attempts_per_seed_condition": int(
            getattr(task_class, "SCRIPTED_EXPERT_MAX_ATTEMPTS", 1)
        ),
        "fresh_scene_each_attempt": True,
        "seed_substitution": False,
        "task_parameter_changes_between_attempts": False,
        "pre_and_handoff_state_repeatability_asserted": True,
        "minimum_repeatability_attempts": MIN_REPEATABILITY_ATTEMPTS,
        "pass_semantics": (
            "at least one complete scene-solvability witness per condition, "
            "with every fixed-state reconstruction passing repeatability checks; "
            "per-cell and per-attempt outcomes remain raw diagnostics"
        ),
    }
    if expert_result.get("expert_attempt_protocol") != expected_attempt_protocol:
        raise RuntimeError(
            "Cannot freeze mismatched expert attempt protocol: "
            f"expected={expected_attempt_protocol} "
            f"actual={expert_result.get('expert_attempt_protocol')}"
        )

    calibration_expert_result = json.loads(
        calibration_expert_path.read_text(encoding="utf-8")
    )
    expected_calibration_pairs = {
        (task_name, int(seed))
        for task_name, calibration_manifest in calibration_manifests.items()
        for seed in calibration_manifest["seeds"]
    }
    actual_calibration_pairs = [
        (str(row.get("task")), int(row.get("seed")))
        for row in calibration_expert_result.get("records", [])
    ]
    expected_manifest_summary = {
        name: {
            "path": str(value["path"]),
            "seeds": value["seeds"],
        }
        for name, value in calibration_manifests.items()
    }
    calibration_evidence_path = (
        project_root
        / "experiments"
        / "robotwin"
        / "move_block_reveal_can_expert_solvability_audit.json"
    ).resolve()
    expected_calibration_attempt_protocol = {
        "attempts_per_fixed_scene": 1,
        "adaptive_retry": False,
        "seed_substitution": False,
        "raw_outcomes_are_not_the_freeze_gate": True,
    }
    raw_calibration_successes = sum(
        int(bool(row.get("passed")))
        for row in calibration_expert_result.get("records", [])
    )
    if (
        not calibration_expert_result.get("passed")
        or calibration_expert_result.get("mode") != "calibration_expert"
        or calibration_expert_result.get("task_config") != "demo_clean"
        or calibration_expert_result.get("source_fingerprint")
        != current_source_fingerprint
        or calibration_expert_result.get("manifests") != expected_manifest_summary
        or len(actual_calibration_pairs) != len(set(actual_calibration_pairs))
        or set(actual_calibration_pairs) != expected_calibration_pairs
        or calibration_expert_result.get("attempt_protocol")
        != expected_calibration_attempt_protocol
        or calibration_expert_result.get("cross_run_evidence_path")
        != str(calibration_evidence_path)
        or calibration_expert_result.get("cross_run_evidence_sha256")
        != sha256(calibration_evidence_path)
        or calibration_expert_result.get("successes")
        != raw_calibration_successes
        or calibration_expert_result.get("total")
        != len(expected_calibration_pairs)
        or calibration_expert_result.get("scene_solvability_successes")
        != len(expected_calibration_pairs)
        or calibration_expert_result.get("scene_solvability_total")
        != len(expected_calibration_pairs)
        or any(
            not row.get("scene_solvability_witness")
            or not str(row.get("cross_run_witness", "")).strip()
            or int(row.get("max_attempts", -1)) != 1
            or len(row.get("attempts", [])) != 1
            or bool(row.get("passed")) != bool(row.get("current_run_witness"))
            or (
                row.get("passed")
                and row.get("witness_attempt_index") is None
            )
            for row in calibration_expert_result.get("records", [])
        )
    ):
        raise RuntimeError(
            "Cannot freeze incomplete/stale calibration expert audit: "
            f"actual_pairs={actual_calibration_pairs}, "
            f"expected_pairs={sorted(expected_calibration_pairs)}"
        )

    source_files = simulator_source_files(project_root, robotwin_root)
    asset_files = runtime_asset_files(robotwin_root)
    source_hashes = {str(path.resolve()): sha256(path) for path in source_files}
    asset_hashes = {str(path.resolve()): sha256(path) for path in asset_files}
    hashes = {
        **source_hashes,
        **asset_hashes,
        **current_screenshot_hashes,
        str((output_dir / "simulator" / "screenshots.sha256").resolve()): sha256(
            output_dir / "simulator" / "screenshots.sha256"
        ),
    }
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=project_root, text=True
    )
    asset_target = (robotwin_root / "assets").resolve()
    donor_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=asset_target.parent, text=True
    ).strip()

    payload = {
        "freeze_name": FORMAL_FREEZE_NAME,
        "source_of_truth_robotwin": str(robotwin_root),
        "asset_donor": str(asset_target),
        "asset_donor_git_head": donor_head,
        "fastwam_git_head": git_head,
        "fastwam_git_status_at_freeze": git_status.splitlines(),
        "source_sha256": source_hashes,
        "runtime_asset_sha256": asset_hashes,
        "task": "move_block_reveal_can",
        "task_config": "demo_clean",
        "instruction": "Put the can into the basket.",
        "seeds": manifest["seeds"],
        "can_sampling": {
            "left_x": [-0.25, -0.20],
            "right_x": [0.20, 0.25],
            "y": [0.0, 0.1],
            "models": [0, 1, 2, 3, 5, 6],
        },
        "basket_sampling": {
            "left_arm_case_x": [0.02, 0.02],
            "right_arm_case_x": [-0.02, -0.02],
            "y": [-0.08, -0.05],
            "models": [0, 1],
        },
        "occluder": {
            "implementation": "one dynamic rigid actor with procedural collision and visual geometry for both body and grasp cap",
            "opaque_body_half_size_m": task_class.OCCLUDER_HALF_SIZE.tolist(),
            "opaque_body_full_size_m": (2.0 * task_class.OCCLUDER_HALF_SIZE).tolist(),
            "grasp_cap_half_size_m": task_class.OCCLUDER_HANDLE_HALF_SIZE.tolist(),
            "grasp_cap_full_size_m": (
                2.0 * task_class.OCCLUDER_HANDLE_HALF_SIZE
            ).tolist(),
            "grasp_cap_center_z_m": task_class.OCCLUDER_HANDLE_CENTER_Z,
            "overall_axis_aligned_full_size_m": [0.11, 0.06, 0.23],
            "mass_kg": 0.04,
            "color_rgb": [0.12, 0.35, 0.85],
            "occluding_pose": "[can_x, can_y - 0.080, table_top + half_height]",
            "revealed_pose": "[sign(can_x) * 0.300, 0.250, table_top + half_height]",
            "selection_basis": (
                "V0 opaque body was the simulator-only geometry that yielded zero can "
                "pixels in all three cameras; V0/V0.1 contact diagnostics showed that "
                "body was not graspable, while V0.2 showed the exact native 0.06 m "
                "block was too narrow to occlude. The rigid grasp cap uses that native "
                "0.06 m cross-section. No FastWAM result was available."
            ),
        },
        "conditions": {
            "shared_base": (
                "all three conditions are constructed and settled with the block at the "
                "occluding pose, producing an identical pre-intervention state"
            ),
            "visible": (
                "before the first policy observation, block pose/velocity is canonicalized "
                "to the revealed pose; the policy's initial observation sees the can"
            ),
            "missing": "block remains at the occluding pose; no intervention",
            "oracle_reveal": (
                "starts exactly as missing; before the first policy observation only the "
                "block pose/velocity is changed by the same operation used for visible"
            ),
        },
        "visibility": {
            "detector": "SAPIEN actor-ID segmentation",
            "missing": "can pixel count equals zero in head, left wrist, and right wrist",
            "visible_oracle": "can has at least 32 pixels in any one of the three policy cameras",
            "basket": "at least 32 head-camera pixels",
            "block": "at least 32 pixels in at least one of the three policy cameras",
        },
        "episode_tracking": {
            "block_translation_threshold_m": task_class.BLOCK_MOVED_THRESHOLD_M,
            "block_rotation_threshold_rad": task_class.BLOCK_ROTATED_THRESHOLD_RAD,
            "motion_sampling": "after every controlled physics step",
            "visibility_sampling": "after every qpos action and requested observation",
        },
        "paired_state_absolute_tolerance": STATE_ATOL,
        "success": (
            "the exact native place_can_basket.check_success predicate remains true for "
            "25 actual controlled physics steps: basket and can are each lifted >0.02 m "
            "from their initial heights, basket local y-axis dot world z >0.5, can/basket "
            "center L1 distance <0.15 m, can is not in contact with the table, and can "
            "contacts the basket. The block is ignored and check_success is a "
            "side-effect-free read. Full-AABB, release, and relative-motion measurements "
            "are supplemental diagnostics only"
        ),
        "success_thresholds": {
            "native_basket_lift_m": 0.02,
            "native_can_lift_m": 0.02,
            "native_basket_axis_dot_world_z": 0.5,
            "native_can_basket_center_l1_distance_m": 0.15,
            "native_can_not_contact_table": True,
            "native_can_contacts_basket": True,
            "success_consecutive_physics_steps": task_class.SUCCESS_STABLE_PHYSICS_STEPS,
            "success_consecutive_duration_s": task_class.SUCCESS_STABLE_PHYSICS_STEPS
            / 250.0,
            "expert_settle_max_physics_steps": task_class.EXPERT_SUCCESS_SETTLE_MAX_STEPS,
        },
        "scripted_expert": {
            "visible_and_oracle": "exact RoboTwin place_can_basket.play_once parent expert",
            "missing_reveal": "AABB-derived vertical clearance, then outward-X and then rearward-Y transfer to the fixed revealed pose",
            "reveal_vertical_clearance_m": task_class.EXPERT_REVEAL_VERTICAL_CLEARANCE_M,
            "reveal_clearance_tolerance_m": task_class.EXPERT_REVEAL_CLEARANCE_TOLERANCE_M,
            "reveal_lift_command_buffer_m": task_class.EXPERT_REVEAL_LIFT_COMMAND_BUFFER_M,
            "meaningful_contact_separation_threshold_m": task_class.EXPERT_REVEAL_CONTACT_SEPARATION_THRESHOLD_M,
            "meaningful_contact_impulse_threshold_ns": task_class.EXPERT_REVEAL_CONTACT_IMPULSE_THRESHOLD_NS,
            "reveal_max_can_displacement_m": task_class.EXPERT_REVEAL_MAX_CAN_DISPLACEMENT_M,
            "reveal_max_can_rotation_rad": task_class.EXPERT_REVEAL_MAX_CAN_ROTATION_RAD,
            "downstream_implementation": "place_can_basket.play_once(self)",
            "max_attempts_per_seed_condition": task_class.SCRIPTED_EXPERT_MAX_ATTEMPTS,
            "minimum_repeatability_attempts": MIN_REPEATABILITY_ATTEMPTS,
            "seed_substitution": False,
        },
        "horizon": 1000,
        "evaluation_plan_frozen_before_fastwam": {
            "native_downstream_calibration": {
                "task": "place_can_basket",
                "episodes": len(calibration_manifests["place_can_basket"]["seeds"]),
                "seeds": calibration_manifests["place_can_basket"]["seeds"],
                "seed_manifest": str(
                    calibration_manifests["place_can_basket"]["path"]
                ),
                "admission_threshold": 0.80,
            },
            "exact_block_reveal_calibration": {
                "task": "move_block_reveal_can_block_calibration",
                "episodes": len(
                    calibration_manifests[
                        "move_block_reveal_can_block_calibration"
                    ]["seeds"]
                ),
                "seeds": calibration_manifests[
                    "move_block_reveal_can_block_calibration"
                ]["seeds"],
                "seed_manifest": str(
                    calibration_manifests[
                        "move_block_reveal_can_block_calibration"
                    ]["path"]
                ),
                "admission_threshold": 0.70,
                "evidence_boundary": (
                    "same frozen Missing scene, compound block, and reveal success; "
                    "the instruction explicitly requests the block reveal and the can-to-"
                    "basket downstream is not scored"
                ),
            },
            "admission_rule": (
                "run frozen native calibrations once; run three-condition pilot only if "
                "both thresholds pass; never alter task geometry from policy results"
            ),
            "pilot_episodes_per_condition": len(manifest["seeds"]),
        },
        "gpu_isolation": {
            "simulator_physical_index": os.environ.get("MINIBENCH_SIM_GPU_INDEX"),
            "policy_physical_index": os.environ.get("MINIBENCH_POLICY_GPU_INDEX"),
            "simulator_gpu_uuid": os.environ.get("MINIBENCH_SIM_GPU_UUID"),
            "policy_gpu_uuid": os.environ.get("MINIBENCH_POLICY_GPU_UUID"),
            "render_device": os.environ.get("ROBOTWIN_RENDER_DEVICE"),
            "mechanism": (
                "bubblewrap device namespace exposes only selected NVIDIA and matching "
                "DRI nodes; SAPIEN RenderSystem is pinned to render_device"
            ),
        },
        "simulator_reset_audit": str(reset_path),
        "scripted_expert_audit": str(expert_path),
        "calibration_scripted_expert_audit": str(calibration_expert_path),
    }
    freeze_path = output_dir / "metadata" / f"{FORMAL_FREEZE_NAME}.json"
    write_json(freeze_path, payload)
    hashes.update(
        {
            str(freeze_path.resolve()): sha256(freeze_path),
            str(reset_path.resolve()): sha256(reset_path),
            str(expert_path.resolve()): sha256(expert_path),
            str(calibration_expert_path.resolve()): sha256(
                calibration_expert_path
            ),
            str(provenance_path.resolve()): sha256(provenance_path),
        }
    )
    checksums = output_dir / "metadata" / f"{FORMAL_FREEZE_NAME}.sha256"
    checksums.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in hashes.items()),
        encoding="utf-8",
    )
    return payload


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    robotwin_root = args.robotwin_root.expanduser().resolve()
    expected_root = (project_root / "third_party" / "RoboTwin").resolve()
    if robotwin_root != expected_root:
        raise RuntimeError(f"Source-of-truth violation: {robotwin_root} != {expected_root}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, seeds = load_manifest(args.seed_manifest.expanduser().resolve())
    task_class, module_path = install_robotwin(robotwin_root)
    source_fingerprint = simulator_source_fingerprint(project_root, robotwin_root)
    write_json(
        output_dir / "metadata" / "simulator_provenance.json",
        {
            "robotwin_root": str(robotwin_root),
            "task_module": str(module_path),
            "seed_manifest": str(args.seed_manifest.expanduser().resolve()),
            "source_fingerprint": source_fingerprint,
        },
    )
    task_args = load_task_args(robotwin_root)

    if args.mode == "reset":
        result = run_reset(
            task_class,
            task_args,
            seeds,
            output_dir / "simulator",
            source_fingerprint,
        )
    elif args.mode == "expert":
        result = run_expert(
            task_class,
            task_args,
            seeds,
            output_dir / "simulator",
            source_fingerprint,
        )
    elif args.mode == "calibration_expert":
        result = run_calibration_expert(
            project_root,
            robotwin_root,
            task_args,
            output_dir / "simulator",
            source_fingerprint,
        )
    else:
        result = run_freeze(project_root, robotwin_root, task_class, manifest, output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
