"""Explicit block-reveal calibration for ``move_block_reveal_can``.

This task uses the frozen Missing scene and exact compound occluder.  It asks
only for the reveal action and never scores the downstream can placement.
"""

import numpy as np

from .move_block_reveal_can import move_block_reveal_can


class move_block_reveal_can_block_calibration(move_block_reveal_can):
    def setup_demo(self, is_test=False, **kwargs):
        requested = str(kwargs.get("minibench_condition", "missing")).lower()
        if requested != "missing":
            raise ValueError(
                "block calibration is defined only on the frozen missing condition"
            )
        kwargs["minibench_condition"] = "missing"
        super().setup_demo(is_test=is_test, **kwargs)

    def _success_diagnostics(self):
        if not getattr(self, "minibench_handoff_prepared", False):
            checks = {
                "block_translated_aside": False,
                "can_currently_visible": False,
                "visibility_sample_matches_current_block_pose": False,
                "block_released": False,
                "block_not_contacting_can": False,
                "block_linear_speed_stable": False,
                "block_angular_speed_stable": False,
            }
            return {"checks": checks}

        displacement = float(
            np.linalg.norm(
                np.asarray(self.occluder.get_pose().p, dtype=np.float64)
                - self.minibench_handoff_occluder_xyz
            )
        )
        linear, angular = self._rigid_velocity(self.occluder)
        visibility_pose_matches = bool(
            self.minibench_visibility_sample_occluder_xyz is not None
            and np.linalg.norm(
                np.asarray(self.occluder.get_pose().p, dtype=np.float64)
                - self.minibench_visibility_sample_occluder_xyz
            )
            <= self.VISIBILITY_POSE_TOLERANCE_M
            and self._quaternion_angle(
                self.occluder.get_pose().q,
                self.minibench_visibility_sample_occluder_q,
            )
            <= self.VISIBILITY_ROTATION_TOLERANCE_RAD
        )
        gripper_contacts = [
            name
            for name in self.robot.gripper_name
            if self.check_actors_contact(self.occluder.get_name(), name)
        ]
        block_can_contact = self._actor_contact_diagnostics(
            self.occluder.get_name(), self.can_name
        )
        checks = {
            "block_translated_aside": bool(
                displacement >= self.BLOCK_MOVED_THRESHOLD_M
            ),
            "can_currently_visible": bool(self.minibench_current_target_visible),
            "visibility_sample_matches_current_block_pose": visibility_pose_matches,
            "block_released": not bool(gripper_contacts),
            "block_not_contacting_can": not block_can_contact[
                "meaningful_contact"
            ],
            "block_linear_speed_stable": bool(
                np.linalg.norm(linear) < self.SUCCESS_REL_LINEAR_SPEED_MPS
            ),
            "block_angular_speed_stable": bool(
                np.linalg.norm(angular) < self.SUCCESS_REL_ANGULAR_SPEED_RADPS
            ),
        }
        return {
            "checks": checks,
            "block_displacement_from_handoff_m": displacement,
            "block_linear_speed_mps": float(np.linalg.norm(linear)),
            "block_angular_speed_radps": float(np.linalg.norm(angular)),
            "block_gripper_contacts": gripper_contacts,
            "block_can_contact_diagnostics": block_can_contact,
        }

    def play_once(self):
        self.minibench_expert_active = True
        try:
            self.prepare_policy_handoff()
            self._expert_reveal()
            for _ in range(self.EXPERT_SUCCESS_SETTLE_MAX_STEPS):
                self._step_physics()
                if self.check_success():
                    break
            self.minibench_expert_success_streak = int(
                self.minibench_success_streak_physics_steps
            )
            if not self.check_success():
                raise RuntimeError(
                    "scripted block calibration did not remain successful for "
                    f"{self.SUCCESS_STABLE_PHYSICS_STEPS} physics steps"
                )
            self.info["info"] = {}
            return self.info
        finally:
            self.minibench_expert_active = False
