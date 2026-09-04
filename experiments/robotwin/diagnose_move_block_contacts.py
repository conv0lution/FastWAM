#!/usr/bin/env python3
"""Task-specific contact-point diagnosis for the frozen MiniBench block."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np

from test_move_block_reveal_can import (
    build,
    install_robotwin,
    load_task_args,
    write_json,
)


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=project_root / "third_party" / "RoboTwin",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    robotwin_root = args.robotwin_root.resolve()
    expected_root = (project_root / "third_party" / "RoboTwin").resolve()
    if robotwin_root != expected_root:
        raise RuntimeError(f"Source-of-truth violation: {robotwin_root} != {expected_root}")
    output = args.output.resolve()
    task_class, module_path = install_robotwin(robotwin_root)
    task_args = load_task_args(robotwin_root)
    rows = []

    # 4300000 is a left-arm scene and 4300002 is a right-arm scene.  Testing
    # every contact on both sides isolates planner reachability from aperture.
    for episode, seed in enumerate((4300000, 4300002)):
        for contact_id in range(4):
            task = None
            try:
                task = build(task_class, task_args, "missing", seed, episode)
                task.prepare_policy_handoff()
                arm_tag = task.arm_tag
                initial_pose = np.asarray(task.occluder.get_pose().p, dtype=float)
                grasp_ok = bool(
                    task.move(
                        task.grasp_actor(
                            task.occluder,
                            arm_tag=arm_tag,
                            pre_grasp_dis=0.08,
                            grasp_dis=0.0,
                            contact_point_id=contact_id,
                        )
                    )
                )
                after_grasp_pose = np.asarray(task.occluder.get_pose().p, dtype=float)
                prefix = "fl" if str(arm_tag) == "left" else "fr"
                finger_contacts = {
                    f"{prefix}_link7": bool(
                        task.check_actors_contact("minibench_occluder", f"{prefix}_link7")
                    ),
                    f"{prefix}_link8": bool(
                        task.check_actors_contact("minibench_occluder", f"{prefix}_link8")
                    ),
                }
                lift_ok = False
                if grasp_ok:
                    lift_ok = bool(task.move(task.move_by_displacement(arm_tag=arm_tag, z=0.08)))
                final_pose = np.asarray(task.occluder.get_pose().p, dtype=float)
                rows.append(
                    {
                        "seed": seed,
                        "arm": str(arm_tag),
                        "contact_id": contact_id,
                        "grasp_ok": grasp_ok,
                        "lift_ok": lift_ok,
                        "plan_success": bool(task.plan_success),
                        "finger_contacts_after_close": finger_contacts,
                        "initial_xyz": initial_pose.tolist(),
                        "after_grasp_xyz": after_grasp_pose.tolist(),
                        "final_xyz": final_pose.tolist(),
                        "delta_after_grasp_xyz": (after_grasp_pose - initial_pose).tolist(),
                        "delta_final_xyz": (final_pose - initial_pose).tolist(),
                        "lift_followed": bool(final_pose[2] - initial_pose[2] >= 0.04),
                    }
                )
            except Exception as exc:
                row = {
                    "seed": seed,
                    "contact_id": contact_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                if task is not None:
                    row["arm"] = str(task.arm_tag)
                    row["plan_success"] = bool(task.plan_success)
                    row["final_xyz"] = np.asarray(
                        task.occluder.get_pose().p, dtype=float
                    ).tolist()
                rows.append(row)
            finally:
                if task is not None:
                    task.close_env()

    write_json(
        output,
        {
            "task": "move_block_reveal_can",
            "condition": "missing",
            "task_module": str(module_path),
            "geometry_changed": False,
            "rows": rows,
        },
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
