"""A paired occlusion variant of RoboTwin's native ``place_can_basket`` task.

The task deliberately keeps all sampling in the native parent.  The only
condition-dependent state is the pose of one procedural, dynamic block.
"""

from copy import deepcopy

import numpy as np
import sapien
import sapien.physx as sapienp

from .place_can_basket import place_can_basket
from .utils import Actor, ArmTag, UnStableError


class move_block_reveal_can(place_can_basket):
    CONDITIONS = frozenset({"visible", "missing", "oracle_reveal"})
    CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")

    # The opaque body is the smallest tested V0 body that fully hid the can in
    # all three policy cameras.  A native-sized procedural grasp cap is rigidly
    # attached at its top: V0/V0.1 proved the body alone was too wide for the
    # gripper, while V0.2 proved RoboTwin's narrow native block did not occlude
    # the can.  Both choices were made from simulator-only evidence.
    OCCLUDER_HALF_SIZE = np.asarray([0.055, 0.025, 0.090], dtype=np.float64)
    OCCLUDER_HANDLE_HALF_SIZE = np.asarray([0.030, 0.030, 0.025], dtype=np.float64)
    OCCLUDER_HANDLE_CENTER_Z = 0.115
    OCCLUDER_LOCAL_AABB_MIN = np.asarray([-0.055, -0.030, -0.090], dtype=np.float64)
    OCCLUDER_LOCAL_AABB_MAX = np.asarray([0.055, 0.030, 0.140], dtype=np.float64)
    OCCLUDER_Y_OFFSET = 0.080
    # The revealed block is placed in the same-side rear workspace, away from
    # the robot's reset wrist volume and from the can/basket manipulation path.
    REVEALED_X_ABS = 0.300
    REVEALED_Y = 0.250
    BLOCK_MOVED_THRESHOLD_M = 0.080
    BLOCK_ROTATED_THRESHOLD_RAD = np.deg2rad(15.0)
    VISIBLE_PIXEL_THRESHOLD = 32
    SUCCESS_LATERAL_MARGIN_M = 0.002
    SUCCESS_VERTICAL_TOLERANCE_M = 0.002
    SUCCESS_REL_LINEAR_SPEED_MPS = 0.05
    SUCCESS_REL_ANGULAR_SPEED_RADPS = 1.0
    # ``check_success`` is a pure query. Both policy and expert success require
    # this many actual physics-step hook updates, so repeated log queries can
    # never turn one unchanged state into a success.
    SUCCESS_STABLE_PHYSICS_STEPS = 25
    EXPERT_SUCCESS_SETTLE_MAX_STEPS = 500
    SCRIPTED_EXPERT_MAX_ATTEMPTS = 3
    EXPERT_REVEAL_CONTACT_POINT_ID = [0, 1, 2, 3]
    EXPERT_REVEAL_VERTICAL_CLEARANCE_M = 0.05
    EXPERT_REVEAL_CLEARANCE_TOLERANCE_M = 0.002
    # The commanded lift includes a fixed execution margin, while the
    # independent world-AABB assertion below continues to enforce the same
    # 5 cm clearance (with its existing 2 mm numerical tolerance).  This is a
    # scripted-expert trajectory margin; it does not change task geometry.
    EXPERT_REVEAL_LIFT_COMMAND_BUFFER_M = 0.010
    # PhysX exposes speculative contact-offset pairs with positive separation
    # and zero impulse.  Such pairs are proximity candidates, not a collision.
    # A reveal contact is physically meaningful only if a point is touching /
    # penetrating (signed separation <= 0) or carries an applied impulse.
    EXPERT_REVEAL_CONTACT_SEPARATION_THRESHOLD_M = 0.0
    EXPERT_REVEAL_CONTACT_IMPULSE_THRESHOLD_NS = 0.0
    EXPERT_REVEAL_MAX_CAN_DISPLACEMENT_M = 0.005
    EXPERT_REVEAL_MAX_CAN_ROTATION_RAD = np.deg2rad(3.0)
    VISIBILITY_POSE_TOLERANCE_M = 0.005
    VISIBILITY_ROTATION_TOLERANCE_RAD = np.deg2rad(3.0)

    def _create_occluder(self, pose):
        """Build both procedural shapes before adding their single rigid actor."""
        entity = sapien.Entity()
        entity.set_name("minibench_occluder")
        entity.set_pose(pose)

        rigid = sapienp.PhysxRigidDynamicComponent()
        body_collision = sapienp.PhysxCollisionShapeBox(
            half_size=self.OCCLUDER_HALF_SIZE,
            material=self.scene.default_physical_material,
        )
        rigid.attach(body_collision)
        cap_pose = sapien.Pose([0.0, 0.0, self.OCCLUDER_HANDLE_CENTER_Z])
        cap_collision = sapienp.PhysxCollisionShapeBox(
            half_size=self.OCCLUDER_HANDLE_HALF_SIZE,
            material=self.scene.default_physical_material,
        )
        cap_collision.set_local_pose(cap_pose)
        rigid.attach(cap_collision)

        material = sapien.render.RenderMaterial(base_color=[0.12, 0.35, 0.85, 1.0])
        render = sapien.render.RenderBodyComponent()
        render.attach(sapien.render.RenderShapeBox(self.OCCLUDER_HALF_SIZE, material))
        cap_visual = sapien.render.RenderShapeBox(
            self.OCCLUDER_HANDLE_HALF_SIZE, material
        )
        cap_visual.set_local_pose(cap_pose)
        render.attach(cap_visual)

        entity.add_component(rigid)
        entity.add_component(render)
        self.scene.add_entity(entity)
        entity.set_pose(pose)

        # Reuse the four native ``boxtype=long`` approach orientations, but
        # center them on the 6 cm grasp cap rather than the 11 cm opaque body.
        z_normalized = self.OCCLUDER_HANDLE_CENTER_Z / float(
            self.OCCLUDER_HALF_SIZE[2]
        )
        data = {
            "center": [0.0, 0.0, 0.0],
            "extents": self.OCCLUDER_HALF_SIZE.tolist(),
            "scale": self.OCCLUDER_HALF_SIZE.tolist(),
            "target_pose": [np.eye(4).tolist()],
            "contact_points_pose": [
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, z_normalized], [0, 0, 0, 1]],
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, z_normalized], [0, 0, 0, 1]],
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, z_normalized], [0, 0, 0, 1]],
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, z_normalized], [0, 0, 0, 1]],
            ],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [],
            "contact_points_description": [],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True],
            "target_point_description": [],
        }
        return Actor(entity, data)

    def setup_demo(self, is_test=False, **kwargs):
        condition = str(kwargs.get("minibench_condition", "visible")).strip().lower()
        if condition not in self.CONDITIONS:
            raise ValueError(
                f"Unsupported minibench_condition={condition!r}; "
                f"expected one of {sorted(self.CONDITIONS)}"
            )

        self.minibench_condition = condition
        self.minibench_seed = int(kwargs.get("seed", 0))
        self.minibench_handoff_prepared = False
        self.minibench_oracle_applied = False
        self.minibench_pre_intervention_state = None
        self.minibench_handoff_state = None
        self.minibench_handoff_visibility = None
        self.minibench_handoff_occluder_xyz = None
        self.minibench_handoff_occluder_q = None
        self.minibench_block_moved_by_policy = False
        self.minibench_block_moved_by_expert = False
        self.minibench_block_moved_step = None
        self.minibench_expert_active = False
        self.minibench_first_visible_step = None
        self.minibench_ever_visible = False
        self.minibench_current_target_visible = False
        self.minibench_visibility_sample_occluder_xyz = None
        self.minibench_visibility_sample_occluder_q = None
        self.minibench_max_block_displacement_m = 0.0
        self.minibench_max_block_rotation_rad = 0.0
        self.minibench_physics_steps_after_handoff = 0
        self.minibench_success_streak_physics_steps = 0
        self.minibench_max_success_streak_physics_steps = 0
        self.minibench_policy_success_step = None
        self.minibench_expert_success_streak = 0
        self.minibench_expert_trace = []
        self.minibench_expert_reveal_max_can_displacement_m = 0.0
        self.minibench_expert_reveal_max_can_rotation_rad = 0.0
        self.minibench_expert_reveal_active = False
        self.minibench_expert_reveal_stage = None
        self.minibench_expert_reveal_can_start_xyz = None
        self.minibench_expert_reveal_can_start_q = None
        self.minibench_expert_reveal_clearance_required = False
        self.minibench_expert_reveal_violation = None
        self.minibench_expert_reveal_block_can_raw_pair_steps = 0
        self.minibench_expert_reveal_block_can_meaningful_steps = 0
        self.minibench_expert_reveal_min_block_can_separation_m = None
        self.minibench_expert_reveal_max_block_can_impulse_ns = 0.0
        self.minibench_native_parent_expert_completed = False

        super().setup_demo(is_test=is_test, **kwargs)

        up_z = float(self.occluder.get_pose().to_transformation_matrix()[2, 2])
        if up_z < 0.98:
            raise UnStableError(
                "minibench occluder tipped before handoff "
                f"(up_z={up_z:.6f}, seed={self.minibench_seed}, condition={condition})"
            )
        self._update_render()
        self.minibench_pre_intervention_state = self._state_snapshot()
        self.minibench_initial_success = bool(self._instant_success())

    def load_actors(self):
        # Native can/basket model and pose sampling, arm choice, and masses
        # remain in the parent task.  This subclass supplies its audited expert.
        super().load_actors()

        can_pose = self.can.get_pose()
        side = -1.0 if can_pose.p[0] < 0 else 1.0
        table_top = 0.74 + self.table_z_bias
        half = self.OCCLUDER_HALF_SIZE

        occluding_xyz = np.asarray(
            [
                float(can_pose.p[0]),
                float(can_pose.p[1] - self.OCCLUDER_Y_OFFSET),
                float(table_top + half[2]),
            ],
            dtype=np.float64,
        )
        revealed_xyz = np.asarray(
            [side * self.REVEALED_X_ABS, self.REVEALED_Y, table_top + half[2]],
            dtype=np.float64,
        )
        self.occluder_occluding_pose = sapien.Pose(occluding_xyz, [1, 0, 0, 0])
        self.occluder_revealed_pose = sapien.Pose(revealed_xyz, [1, 0, 0, 0])

        # Every condition is constructed from the same physical base state.
        # Visible and Oracle are moved to the same revealed pose only at the
        # policy-handoff intervention boundary.
        initial_pose = self.occluder_occluding_pose
        self.occluder = self._create_occluder(initial_pose)
        self.occluder.set_mass(0.04)

        # Add both possible block footprints in every condition.  This keeps
        # randomized clutter sampling paired instead of making the prohibit
        # list condition-dependent.
        hx, hy = float(half[0] + 0.04), float(half[1] + 0.04)
        for pose in (self.occluder_occluding_pose, self.occluder_revealed_pose):
            x, y = float(pose.p[0]), float(pose.p[1])
            self.prohibited_area.append([x - hx, y - hy, x + hx, y + hy])

    @staticmethod
    def _pose_list(pose):
        return np.concatenate([np.asarray(pose.p), np.asarray(pose.q)]).astype(float).tolist()

    def _camera_map(self):
        cameras = {
            "left_camera": self.cameras.left_camera,
            "right_camera": self.cameras.right_camera,
        }
        cameras.update(dict(zip(self.cameras.static_camera_name, self.cameras.static_camera_list)))
        return {name: cameras[name] for name in self.CAMERA_NAMES}

    def _camera_matrices(self):
        config = self.cameras.get_config()
        return {
            name: np.asarray(config[name]["cam2world_gl"], dtype=np.float64).round(9).tolist()
            for name in self.CAMERA_NAMES
        }

    def _state_snapshot(self):
        def entity_velocity(entity):
            rigid = entity.find_component_by_type(sapienp.PhysxRigidDynamicComponent)
            if rigid is None:
                return {"linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, 0.0]}
            return {
                "linear": np.asarray(rigid.get_linear_velocity(), dtype=np.float64).round(9).tolist(),
                "angular": np.asarray(rigid.get_angular_velocity(), dtype=np.float64).round(9).tolist(),
            }

        scene_actor_states = []
        for index, entity in enumerate(self.scene.get_all_actors()):
            scene_actor_states.append(
                {
                    "index": index,
                    "name": entity.get_name(),
                    "pose": self._pose_list(entity.get_pose()),
                    "velocity": entity_velocity(entity),
                }
            )

        return {
            "seed": self.minibench_seed,
            "can_model_id": int(self.can_id),
            "basket_model_id": int(self.basket_id),
            "arm_tag": str(self.arm_tag),
            "can_pose": self._pose_list(self.can.get_pose()),
            "basket_pose": self._pose_list(self.basket.get_pose()),
            "occluder_pose": self._pose_list(self.occluder.get_pose()),
            "can_velocity": entity_velocity(self.can.actor),
            "basket_velocity": entity_velocity(self.basket.actor),
            "occluder_velocity": entity_velocity(self.occluder.actor),
            "scene_actor_states": scene_actor_states,
            "robot_qpos": np.asarray(
                self.robot.get_left_arm_jointState() + self.robot.get_right_arm_jointState(),
                dtype=np.float64,
            ).round(9).tolist(),
            "robot_qvel": {
                "left": np.asarray(self.robot.left_entity.get_qvel(), dtype=np.float64)
                .round(9)
                .tolist(),
                "right": np.asarray(self.robot.right_entity.get_qvel(), dtype=np.float64)
                .round(9)
                .tolist(),
            },
            "camera_cam2world": self._camera_matrices(),
            "table_z_bias": float(self.table_z_bias),
            "wall_texture": self.wall_texture,
            "table_texture": self.table_texture,
            "domain_randomization": {
                "random_background": bool(self.random_background),
                "cluttered_table": bool(self.cluttered_table),
                "random_head_camera_dis": float(self.random_head_camera_dis),
                "random_table_height": float(self.random_table_height),
                "random_light": bool(self.random_light),
            },
        }

    def _teleport_occluder(self, target_pose):
        self.occluder.actor.set_pose(target_pose)
        rigid = self.occluder.actor.find_component_by_type(sapienp.PhysxRigidDynamicComponent)
        if rigid is not None:
            rigid.set_linear_velocity([0.0, 0.0, 0.0])
            rigid.set_angular_velocity([0.0, 0.0, 0.0])

    def prepare_policy_handoff(self):
        """Apply the sole Oracle intervention before the first policy observation."""
        if self.minibench_handoff_prepared:
            return self.get_minibench_record(refresh_visibility=False)

        # Visible is re-set to the same canonical pose so its handoff matches
        # Oracle after the latter is moved from the Missing initial state.
        if self.minibench_condition in {"visible", "oracle_reveal"}:
            self._teleport_occluder(self.occluder_revealed_pose)
        if self.minibench_condition == "oracle_reveal":
            self.minibench_oracle_applied = True

        # Advance every condition equally.  Oracle changes only the block pose.
        for _ in range(5):
            self.scene.step()
        self._update_render()
        self.cameras.update_picture()

        self.minibench_handoff_prepared = True
        self.minibench_handoff_state = self._state_snapshot()
        self.minibench_handoff_occluder_xyz = np.asarray(
            self.occluder.get_pose().p, dtype=np.float64
        )
        self.minibench_handoff_occluder_q = np.asarray(
            self.occluder.get_pose().q, dtype=np.float64
        )
        self.minibench_handoff_visibility = self._read_actor_pixel_counts(self.can)
        visible = self._target_visible(self.minibench_handoff_visibility)
        self.minibench_current_target_visible = visible
        self.minibench_visibility_sample_occluder_xyz = np.asarray(
            self.occluder.get_pose().p, dtype=np.float64
        ).copy()
        self.minibench_visibility_sample_occluder_q = np.asarray(
            self.occluder.get_pose().q, dtype=np.float64
        ).copy()
        self.minibench_ever_visible = visible
        if visible:
            self.minibench_first_visible_step = -1
        return self.get_minibench_record(refresh_visibility=False)

    def _read_actor_pixel_counts(self, actor):
        actor_id = int(actor.actor.get_per_scene_id())
        return {
            name: int(
                np.count_nonzero(
                    np.asarray(camera.get_picture("Segmentation"))[..., 1].astype(np.uint32)
                    == actor_id
                )
            )
            for name, camera in self._camera_map().items()
        }

    def get_target_pixel_counts(self):
        self._update_render()
        self.cameras.update_picture()
        return self._read_actor_pixel_counts(self.can)

    def get_actor_pixel_counts(self):
        self._update_render()
        self.cameras.update_picture()
        return {
            "can": self._read_actor_pixel_counts(self.can),
            "basket": self._read_actor_pixel_counts(self.basket),
            "block": self._read_actor_pixel_counts(self.occluder),
        }

    def _target_visible(self, counts):
        return bool(any(count >= self.VISIBLE_PIXEL_THRESHOLD for count in counts.values()))

    @staticmethod
    def _quaternion_angle(left, right):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        left /= np.linalg.norm(left)
        right /= np.linalg.norm(right)
        return float(2.0 * np.arccos(np.clip(abs(np.dot(left, right)), 0.0, 1.0)))

    def _update_episode_tracking(self, counts=None):
        if not self.minibench_handoff_prepared:
            return
        displacement = float(
            np.linalg.norm(
                np.asarray(self.occluder.get_pose().p, dtype=np.float64)
                - self.minibench_handoff_occluder_xyz
            )
        )
        self.minibench_max_block_displacement_m = max(
            self.minibench_max_block_displacement_m, displacement
        )
        rotation = self._quaternion_angle(
            self.occluder.get_pose().q,
            self.minibench_handoff_occluder_q,
        )
        self.minibench_max_block_rotation_rad = max(
            self.minibench_max_block_rotation_rad, rotation
        )
        if (
            displacement >= self.BLOCK_MOVED_THRESHOLD_M
            or rotation >= self.BLOCK_ROTATED_THRESHOLD_RAD
        ):
            if self.minibench_expert_active:
                self.minibench_block_moved_by_expert = True
            elif not self.minibench_block_moved_by_policy:
                self.minibench_block_moved_by_policy = True
                self.minibench_block_moved_step = int(self.take_action_cnt)

        if counts is not None:
            self.minibench_current_target_visible = self._target_visible(counts)
            self.minibench_visibility_sample_occluder_xyz = np.asarray(
                self.occluder.get_pose().p, dtype=np.float64
            ).copy()
            self.minibench_visibility_sample_occluder_q = np.asarray(
                self.occluder.get_pose().q, dtype=np.float64
            ).copy()
            if self.minibench_current_target_visible:
                self.minibench_ever_visible = True
                if self.minibench_first_visible_step is None:
                    self.minibench_first_visible_step = int(self.take_action_cnt)

    def get_obs(self):
        if not self.minibench_handoff_prepared:
            self.prepare_policy_handoff()
        observation = super().get_obs()
        counts = self._read_actor_pixel_counts(self.can)
        self._update_episode_tracking(counts)
        return observation

    def take_action(self, *args, **kwargs):
        result = super().take_action(*args, **kwargs)
        if self.minibench_handoff_prepared:
            counts = self.get_target_pixel_counts()
            self._update_episode_tracking(counts)
        return result

    @staticmethod
    def _actor_world_bounds(actor):
        pose_matrix = actor.get_pose().to_transformation_matrix()
        scale = np.asarray(actor.config["scale"], dtype=np.float64)
        center = np.asarray(actor.config["center"], dtype=np.float64) * scale
        half = np.asarray(actor.config["extents"], dtype=np.float64) * scale / 2.0
        center_world = pose_matrix[:3, :3] @ center + pose_matrix[:3, 3]
        half_world = np.abs(pose_matrix[:3, :3]) @ half
        return center_world - half_world, center_world + half_world

    def _occluder_world_bounds(self):
        pose_matrix = self.occluder.get_pose().to_transformation_matrix()
        local_center = (
            self.OCCLUDER_LOCAL_AABB_MIN + self.OCCLUDER_LOCAL_AABB_MAX
        ) / 2.0
        local_half = (
            self.OCCLUDER_LOCAL_AABB_MAX - self.OCCLUDER_LOCAL_AABB_MIN
        ) / 2.0
        center_world = (
            pose_matrix[:3, :3] @ local_center + pose_matrix[:3, 3]
        )
        half_world = np.abs(pose_matrix[:3, :3]) @ local_half
        return center_world - half_world, center_world + half_world

    def _record_expert_reveal_violation(self, stage, **details):
        payload = {"stage": str(stage), **deepcopy(details)}
        if self.minibench_expert_reveal_violation is None:
            self.minibench_expert_reveal_violation = payload
        return payload

    def _actor_contact_diagnostics(self, actor1, actor2):
        """Distinguish a PhysX contact-offset pair from physical contact."""
        points = []
        raw_pair_reported = False
        meaningful_contact = False
        for contact in self.scene.get_contacts():
            names = (
                contact.bodies[0].entity.name,
                contact.bodies[1].entity.name,
            )
            if not (
                (names[0] == actor1 and names[1] == actor2)
                or (names[0] == actor2 and names[1] == actor1)
            ):
                continue
            raw_pair_reported = True
            for point in contact.points:
                separation = float(point.separation)
                impulse_norm = float(
                    np.linalg.norm(np.asarray(point.impulse, dtype=np.float64))
                )
                point_meaningful = bool(
                    separation
                    <= self.EXPERT_REVEAL_CONTACT_SEPARATION_THRESHOLD_M
                    or impulse_norm > self.EXPERT_REVEAL_CONTACT_IMPULSE_THRESHOLD_NS
                )
                meaningful_contact = meaningful_contact or point_meaningful
                points.append(
                    {
                        "separation_m": separation,
                        "impulse_norm_ns": impulse_norm,
                        "meaningful": point_meaningful,
                    }
                )
        separations = [point["separation_m"] for point in points]
        impulses = [point["impulse_norm_ns"] for point in points]
        return {
            "raw_pair_reported": bool(raw_pair_reported),
            "point_count": len(points),
            "min_separation_m": min(separations) if separations else None,
            "max_impulse_norm_ns": max(impulses) if impulses else 0.0,
            "meaningful_contact": bool(meaningful_contact),
        }

    def _assert_expert_reveal_can_untouched(self, start_xyz, stage):
        displacement = float(
            np.linalg.norm(
                np.asarray(self.can.get_pose().p, dtype=np.float64) - start_xyz
            )
        )
        self.minibench_expert_reveal_max_can_displacement_m = max(
            self.minibench_expert_reveal_max_can_displacement_m, displacement
        )
        rotation = self._quaternion_angle(
            self.can.get_pose().q,
            self.minibench_expert_reveal_can_start_q,
        )
        self.minibench_expert_reveal_max_can_rotation_rad = max(
            self.minibench_expert_reveal_max_can_rotation_rad, rotation
        )
        block_contact_diagnostics = self._actor_contact_diagnostics(
            self.occluder.get_name(), self.can_name
        )
        if block_contact_diagnostics["raw_pair_reported"]:
            self.minibench_expert_reveal_block_can_raw_pair_steps += 1
        if block_contact_diagnostics["meaningful_contact"]:
            self.minibench_expert_reveal_block_can_meaningful_steps += 1
        separation = block_contact_diagnostics["min_separation_m"]
        if separation is not None:
            previous = self.minibench_expert_reveal_min_block_can_separation_m
            self.minibench_expert_reveal_min_block_can_separation_m = (
                separation if previous is None else min(previous, separation)
            )
        self.minibench_expert_reveal_max_block_can_impulse_ns = max(
            self.minibench_expert_reveal_max_block_can_impulse_ns,
            block_contact_diagnostics["max_impulse_norm_ns"],
        )
        gripper_contact_diagnostics = {
            name: self._actor_contact_diagnostics(self.can_name, name)
            for name in self.robot.gripper_name
        }
        gripper_contacts = [
            name
            for name, diagnostics in gripper_contact_diagnostics.items()
            if diagnostics["meaningful_contact"]
        ]
        if (
            displacement > self.EXPERT_REVEAL_MAX_CAN_DISPLACEMENT_M
            or rotation > self.EXPERT_REVEAL_MAX_CAN_ROTATION_RAD
            or block_contact_diagnostics["meaningful_contact"]
            or gripper_contacts
        ):
            violation = self._record_expert_reveal_violation(
                stage,
                kind="can_interference",
                displacement_m=displacement,
                rotation_rad=rotation,
                block_contact_pair_reported=block_contact_diagnostics[
                    "raw_pair_reported"
                ],
                block_meaningful_contact=block_contact_diagnostics[
                    "meaningful_contact"
                ],
                block_contact_min_separation_m=block_contact_diagnostics[
                    "min_separation_m"
                ],
                block_contact_max_impulse_norm_ns=block_contact_diagnostics[
                    "max_impulse_norm_ns"
                ],
                can_gripper_contacts=gripper_contacts,
                can_gripper_contact_diagnostics=gripper_contact_diagnostics,
            )
            raise RuntimeError(
                "scripted reveal interfered with the can at "
                f"{stage}: {violation}"
            )

    def _assert_expert_reveal_clearance(self, stage):
        block_min, _ = self._occluder_world_bounds()
        _, can_max = self._actor_world_bounds(self.can)
        actual_clearance = float(block_min[2] - can_max[2])
        required = float(self.EXPERT_REVEAL_VERTICAL_CLEARANCE_M)
        if actual_clearance < required - self.EXPERT_REVEAL_CLEARANCE_TOLERANCE_M:
            violation = self._record_expert_reveal_violation(
                stage,
                kind="insufficient_world_aabb_clearance",
                actual_clearance_m=actual_clearance,
                required_clearance_m=required,
                tolerance_m=float(self.EXPERT_REVEAL_CLEARANCE_TOLERANCE_M),
            )
            raise RuntimeError(
                "scripted reveal lost world-AABB clearance over the can at "
                f"{stage}: {violation}"
            )

    def _after_physics_step(self):
        """Track persistence and expert noninterference once per real physics step."""
        if not getattr(self, "minibench_handoff_prepared", False):
            return
        self.minibench_physics_steps_after_handoff += 1
        self._update_episode_tracking(counts=None)

        if self._instant_success():
            self.minibench_success_streak_physics_steps += 1
        else:
            self.minibench_success_streak_physics_steps = 0
        self.minibench_max_success_streak_physics_steps = max(
            self.minibench_max_success_streak_physics_steps,
            self.minibench_success_streak_physics_steps,
        )
        if (
            not self.minibench_expert_active
            and self.minibench_policy_success_step is None
            and self.minibench_success_streak_physics_steps
            >= self.SUCCESS_STABLE_PHYSICS_STEPS
        ):
            self.minibench_policy_success_step = int(self.take_action_cnt)

        if self.minibench_expert_reveal_active:
            stage = self.minibench_expert_reveal_stage or "expert_reveal_physics_step"
            self._assert_expert_reveal_can_untouched(
                self.minibench_expert_reveal_can_start_xyz,
                stage,
            )
            if self.minibench_expert_reveal_clearance_required:
                self._assert_expert_reveal_clearance(stage)
    def _step_physics(self, count=1):
        for _ in range(int(count)):
            self.scene.step()
            self._after_physics_step()

    def _expert_reveal(self):
        arm_tag = ArmTag(str(self.arm_tag))
        contact_id = self.EXPERT_REVEAL_CONTACT_POINT_ID
        can_start_xyz = np.asarray(self.can.get_pose().p, dtype=np.float64).copy()
        can_start_q = np.asarray(self.can.get_pose().q, dtype=np.float64).copy()
        self.minibench_expert_reveal_active = True
        self.minibench_expert_reveal_can_start_xyz = can_start_xyz
        self.minibench_expert_reveal_can_start_q = can_start_q
        try:
            self.minibench_expert_reveal_stage = "block_grasp"
            self._trace_expert_stage("block_pre_grasp")
            if not self.move(
                self.grasp_actor(
                    self.occluder,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.08,
                    grasp_dis=0.0,
                    contact_point_id=contact_id,
                )
            ):
                raise RuntimeError("scripted expert failed to grasp the occluder")
            self._trace_expert_stage("block_post_close")
            self._assert_expert_reveal_can_untouched(can_start_xyz, "block_post_close")

            block_min, _ = self._occluder_world_bounds()
            _, can_max = self._actor_world_bounds(self.can)
            required_lift = max(
                0.0,
                float(
                    can_max[2]
                    + self.EXPERT_REVEAL_VERTICAL_CLEARANCE_M
                    - block_min[2]
                ),
            )
            commanded_lift = (
                required_lift + self.EXPERT_REVEAL_LIFT_COMMAND_BUFFER_M
            )
            self.minibench_expert_reveal_stage = "block_clearance_lift"
            if not self.move(
                self.move_by_displacement(arm_tag=arm_tag, z=commanded_lift)
            ):
                raise RuntimeError("scripted expert failed to lift the occluder")
            self._trace_expert_stage(
                "block_post_lift",
                required_lift_m=required_lift,
                commanded_lift_m=commanded_lift,
                command_buffer_m=self.EXPERT_REVEAL_LIFT_COMMAND_BUFFER_M,
            )
            self._assert_expert_reveal_can_untouched(can_start_xyz, "block_post_lift")
            self._assert_expert_reveal_clearance("block_post_lift")

            # Cross the can's Y coordinate only after the compound block's real
            # world AABB clears the can's real metadata-derived world AABB.
            self.minibench_expert_reveal_clearance_required = True
            self.minibench_expert_reveal_stage = "block_lateral_translate"
            displacement_x = float(
                self.occluder_revealed_pose.p[0] - self.occluder.get_pose().p[0]
            )
            if not self.move(
                self.move_by_displacement(arm_tag=arm_tag, x=displacement_x)
            ):
                raise RuntimeError(
                    "scripted expert failed the lateral occluder translation"
                )
            self._trace_expert_stage("block_post_lateral_translate")
            self._assert_expert_reveal_can_untouched(
                can_start_xyz, "block_post_lateral_translate"
            )
            self._assert_expert_reveal_clearance("block_post_lateral_translate")

            self.minibench_expert_reveal_stage = "block_rearward_translate"
            displacement_y = float(
                self.occluder_revealed_pose.p[1] - self.occluder.get_pose().p[1]
            )
            if not self.move(
                self.move_by_displacement(arm_tag=arm_tag, y=displacement_y)
            ):
                raise RuntimeError(
                    "scripted expert failed the rearward occluder translation"
                )
            self._trace_expert_stage("block_post_rearward_translate")
            self._assert_expert_reveal_can_untouched(
                can_start_xyz, "block_post_rearward_translate"
            )
            self._assert_expert_reveal_clearance("block_post_rearward_translate")
            self.minibench_expert_reveal_clearance_required = False

            translated_xy = np.asarray(
                self.occluder.get_pose().p[:2], dtype=np.float64
            )
            revealed_xy = np.asarray(
                self.occluder_revealed_pose.p[:2], dtype=np.float64
            )
            if np.linalg.norm(translated_xy - revealed_xy) > 0.08:
                raise RuntimeError(
                    "scripted expert planned a translation but the occluder did not "
                    f"follow: actual_xy={translated_xy.tolist()} "
                    f"target_xy={revealed_xy.tolist()}"
                )

            self.minibench_expert_reveal_stage = "block_release"
            if not self.move(self.open_gripper(arm_tag=arm_tag)):
                raise RuntimeError("scripted expert failed to release the occluder")
            self._trace_expert_stage("block_post_release")
            self._assert_expert_reveal_can_untouched(can_start_xyz, "block_post_release")
            # The arm is already at the high transfer pose.  An additional
            # +10 cm retreat can exceed Aloha's workspace after release; return
            # directly to the native home pose instead.
            self.minibench_expert_reveal_stage = "block_arm_home"
            if not self.move(self.back_to_origin(arm_tag=arm_tag)):
                raise RuntimeError("scripted expert failed to return the reveal arm home")
            self._assert_expert_reveal_can_untouched(can_start_xyz, "block_arm_home")

            # Let the released dynamic block settle; the per-step guard remains
            # active, so transient contact or can motion cannot be hidden.
            self.minibench_expert_reveal_stage = "block_settle"
            self._step_physics(2 * self.SUCCESS_STABLE_PHYSICS_STEPS)
            counts = self.get_target_pixel_counts()
            self._update_episode_tracking(counts)
            self._trace_expert_stage("block_post_settle")
            self._assert_expert_reveal_can_untouched(can_start_xyz, "block_post_settle")
            if not self._target_visible(counts):
                raise RuntimeError(
                    f"scripted expert moved block but can is still hidden: {counts}"
                )
        finally:
            self.minibench_expert_reveal_active = False
            self.minibench_expert_reveal_clearance_required = False
            self.minibench_expert_reveal_stage = None

    def _expert_place_can(self):
        """Execute RoboTwin's native ``place_can_basket`` expert unchanged."""
        self._trace_expert_stage("can_native_parent_start")
        place_can_basket.play_once(self)
        if not self.plan_success:
            raise RuntimeError(
                "native place_can_basket scripted expert did not complete"
            )
        self.minibench_native_parent_expert_completed = True
        self._trace_expert_stage("can_native_parent_complete")
    def play_once(self):
        self.minibench_expert_active = True
        try:
            self.prepare_policy_handoff()
            if self.minibench_condition == "missing":
                self._expert_reveal()
            self._expert_place_can()
            # The task hook advances the streak exactly once per physics step;
            # check_success remains a pure read for policy and logging.
            for _ in range(self.EXPERT_SUCCESS_SETTLE_MAX_STEPS):
                self._step_physics()
                if self.check_success():
                    break
            self.minibench_expert_success_streak = int(
                self.minibench_success_streak_physics_steps
            )
            if not self.check_success():
                raise RuntimeError(
                    "scripted expert final can placement did not remain valid for "
                    f"{self.SUCCESS_STABLE_PHYSICS_STEPS} consecutive physics steps"
                )
            self._trace_expert_stage("final_settle")
            # The benchmark language is intentionally invariant and contains no
            # reveal hint or object-ID placeholder.
            self.info["info"] = {}
            return self.info
        finally:
            self.minibench_expert_active = False

    @staticmethod
    def _rigid_velocity(actor):
        rigid = actor.actor.find_component_by_type(sapienp.PhysxRigidDynamicComponent)
        if rigid is None:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        return (
            np.asarray(rigid.get_linear_velocity(), dtype=np.float64),
            np.asarray(rigid.get_angular_velocity(), dtype=np.float64),
        )

    def _trace_expert_stage(self, stage, **details):
        """Capture physical execution evidence without mutating success state."""
        can_linear, can_angular = self._rigid_velocity(self.can)
        basket_linear, basket_angular = self._rigid_velocity(self.basket)
        block_linear, block_angular = self._rigid_velocity(self.occluder)
        self.minibench_expert_trace.append(
            {
                "stage": str(stage),
                "details": deepcopy(details),
                "plan_success": bool(self.plan_success),
                "can_pose": self._pose_list(self.can.get_pose()),
                "basket_pose": self._pose_list(self.basket.get_pose()),
                "block_pose": self._pose_list(self.occluder.get_pose()),
                "can_velocity": {
                    "linear": can_linear.astype(float).tolist(),
                    "angular": can_angular.astype(float).tolist(),
                },
                "basket_velocity": {
                    "linear": basket_linear.astype(float).tolist(),
                    "angular": basket_angular.astype(float).tolist(),
                },
                "block_velocity": {
                    "linear": block_linear.astype(float).tolist(),
                    "angular": block_angular.astype(float).tolist(),
                },
                "left_ee_pose": np.asarray(
                    self.robot.get_left_ee_pose(), dtype=np.float64
                ).tolist(),
                "right_ee_pose": np.asarray(
                    self.robot.get_right_ee_pose(), dtype=np.float64
                ).tolist(),
                "robot_qpos": np.asarray(
                    self.robot.get_left_arm_jointState()
                    + self.robot.get_right_arm_jointState(),
                    dtype=np.float64,
                ).tolist(),
                "left_gripper_open": bool(self.is_left_gripper_open()),
                "right_gripper_open": bool(self.is_right_gripper_open()),
                "contacts": {
                    "can_basket": bool(
                        self.check_actors_contact(self.can_name, self.basket_name)
                    ),
                    "can_table": bool(self.check_actors_contact(self.can_name, "table")),
                    "can_grippers": [
                        name
                        for name in self.robot.gripper_name
                        if self.check_actors_contact(self.can_name, name)
                    ],
                    "block_grippers": [
                        name
                        for name in self.robot.gripper_name
                        if self.check_actors_contact(self.occluder.get_name(), name)
                    ],
                },
                "success_diagnostics": self._success_diagnostics(),
            }
        )

    def _success_diagnostics(self):
        """Return the native parent gate plus non-gating geometric diagnostics."""
        can_tm = self.can.get_pose().to_transformation_matrix()
        basket_tm = self.basket.get_pose().to_transformation_matrix()
        can_scale = np.asarray(self.can.config["scale"], dtype=np.float64)
        basket_scale = np.asarray(self.basket.config["scale"], dtype=np.float64)
        can_center_actor = np.asarray(self.can.config["center"], dtype=np.float64) * can_scale
        basket_center_local = (
            np.asarray(self.basket.config["center"], dtype=np.float64) * basket_scale
        )
        can_half_actor = (
            np.asarray(self.can.config["extents"], dtype=np.float64) * can_scale / 2.0
        )
        basket_half_local = (
            np.asarray(self.basket.config["extents"], dtype=np.float64)
            * basket_scale
            / 2.0
        )

        can_center_world = can_tm[:3, :3] @ can_center_actor + can_tm[:3, 3]
        can_center_local = basket_tm[:3, :3].T @ (
            can_center_world - basket_tm[:3, 3]
        )
        can_rotation_local = basket_tm[:3, :3].T @ can_tm[:3, :3]
        can_half_local = np.abs(can_rotation_local) @ can_half_actor

        lateral_margin = self.SUCCESS_LATERAL_MARGIN_M
        lateral_inside = all(
            abs(can_center_local[axis] - basket_center_local[axis])
            + can_half_local[axis]
            <= basket_half_local[axis] - lateral_margin
            for axis in (0, 2)
        )
        basket_bottom = basket_center_local[1] - basket_half_local[1]
        can_bottom = can_center_local[1] - can_half_local[1]
        placement_center_ceiling = min(
            float(np.asarray(matrix, dtype=np.float64)[1, 3] * basket_scale[1])
            for matrix in self.basket.config["functional_matrix"]
        )
        vertical_inside = bool(
            can_center_local[1] >= basket_bottom - self.SUCCESS_VERTICAL_TOLERANCE_M
            and can_center_local[1]
            <= placement_center_ceiling + self.SUCCESS_VERTICAL_TOLERANCE_M
        )

        can_linear, can_angular = self._rigid_velocity(self.can)
        basket_linear, basket_angular = self._rigid_velocity(self.basket)
        basket_linear_at_can = basket_linear + np.cross(
            basket_angular,
            can_center_world - basket_tm[:3, 3],
        )
        relative_linear_speed = float(np.linalg.norm(can_linear - basket_linear_at_can))
        relative_angular_speed = float(np.linalg.norm(can_angular - basket_angular))
        no_gripper_contact = all(
            not self.check_actors_contact(self.can_name, gripper_name)
            for gripper_name in self.robot.gripper_name
        )
        basket_axis = basket_tm[:3, :3] @ np.asarray([0.0, 1.0, 0.0])
        can_arm_open = (
            self.is_left_gripper_open()
            if self.arm_tag == "left"
            else self.is_right_gripper_open()
        )
        can_not_on_table = bool(not self.check_actors_contact(self.can_name, "table"))
        supplemental_checks = {
            "basket_upright": bool(np.dot(basket_axis, [0, 0, 1]) > 0.5),
            "can_aabb_laterally_inside": bool(lateral_inside),
            "can_center_vertically_inside": bool(vertical_inside),
            "can_contacts_basket": bool(
                self.check_actors_contact(self.can_name, self.basket_name)
            ),
            "can_arm_open": bool(can_arm_open),
            "can_not_touching_any_gripper": bool(no_gripper_contact),
            "relative_linear_speed_stable": bool(
                relative_linear_speed < self.SUCCESS_REL_LINEAR_SPEED_MPS
            ),
            "relative_angular_speed_stable": bool(
                relative_angular_speed < self.SUCCESS_REL_ANGULAR_SPEED_RADPS
            ),
        }
        return {
            # The task requirement is the native can -> basket objective.  Keep
            # its predicate authoritative instead of silently replacing it with
            # a stricter geometric surrogate.  The 25-step persistence wrapper
            # remains in ``check_success`` below.
            "checks": {
                "native_place_can_basket_success": bool(
                    place_can_basket.check_success(self)
                )
            },
            "supplemental_non_gating_checks": supplemental_checks,
            "native_parent_basket_lifted_diagnostic": bool(
                self.basket.get_pose().p[2] - self.start_height > 0.02
            ),
            "native_parent_can_arm_open_diagnostic": bool(can_arm_open),
            "can_table_contact_pair_reported_diagnostic": not can_not_on_table,
            "can_center_in_basket_frame_m": can_center_local.round(9).tolist(),
            "can_projected_half_extent_in_basket_frame_m": can_half_local.round(9).tolist(),
            "basket_center_local_m": basket_center_local.round(9).tolist(),
            "basket_half_extent_local_m": basket_half_local.round(9).tolist(),
            "basket_bottom_local_m": float(basket_bottom),
            "basket_placement_center_ceiling_local_m": float(
                placement_center_ceiling
            ),
            "can_bottom_local_m": float(can_bottom),
            "relative_linear_speed_mps": relative_linear_speed,
            "relative_angular_speed_radps": relative_angular_speed,
        }

    def _instant_success(self):
        diagnostics = self._success_diagnostics()
        return bool(all(diagnostics["checks"].values()))

    def check_success(self):
        """Pure persistent can-in-basket query; the block is intentionally ignored."""
        return bool(
            self.minibench_success_streak_physics_steps
            >= self.SUCCESS_STABLE_PHYSICS_STEPS
            and self._instant_success()
        )

    def get_minibench_record(self, refresh_visibility=True):
        final_counts = None
        if refresh_visibility and self.minibench_handoff_prepared:
            final_counts = self.get_target_pixel_counts()
            self._update_episode_tracking(final_counts)
        current_pose = self._pose_list(self.occluder.get_pose())
        success = bool(self.check_success())
        success_diagnostics = self._success_diagnostics()
        return deepcopy(
            {
                "task": self.task_name,
                "condition": self.minibench_condition,
                "seed": self.minibench_seed,
                "pre_intervention_state": self.minibench_pre_intervention_state,
                "handoff_state": self.minibench_handoff_state,
                "handoff_target_pixels": self.minibench_handoff_visibility,
                "handoff_prepared": self.minibench_handoff_prepared,
                "oracle_applied": self.minibench_oracle_applied,
                "occluder_pose_current": current_pose,
                "block_moved_by_policy": self.minibench_block_moved_by_policy,
                "block_moved_by_expert": self.minibench_block_moved_by_expert,
                "block_moved_step": self.minibench_block_moved_step,
                "max_block_displacement_from_handoff_m": self.minibench_max_block_displacement_m,
                "max_block_rotation_from_handoff_rad": self.minibench_max_block_rotation_rad,
                "can_ever_visible": self.minibench_ever_visible,
                "can_currently_visible": self.minibench_current_target_visible,
                "visibility_sample_occluder_pose": (
                    None
                    if self.minibench_visibility_sample_occluder_xyz is None
                    else np.concatenate(
                        [
                            self.minibench_visibility_sample_occluder_xyz,
                            self.minibench_visibility_sample_occluder_q,
                        ]
                    )
                    .astype(float)
                    .tolist()
                ),
                "can_became_visible_after_handoff": bool(
                    self.minibench_first_visible_step is not None
                    and self.minibench_first_visible_step >= 0
                ),
                "first_visible_step": self.minibench_first_visible_step,
                "final_target_pixels": final_counts,
                "initial_success": self.minibench_initial_success,
                "success_diagnostics": success_diagnostics,
                "instant_success": bool(self._instant_success()),
                "physics_steps_after_handoff": int(
                    self.minibench_physics_steps_after_handoff
                ),
                "success_streak_physics_steps": int(
                    self.minibench_success_streak_physics_steps
                ),
                "max_success_streak_physics_steps": int(
                    self.minibench_max_success_streak_physics_steps
                ),
                "expert_verified_success_streak_physics_steps": int(
                    self.minibench_expert_success_streak
                ),
                "expert_reveal_max_can_displacement_m": float(
                    self.minibench_expert_reveal_max_can_displacement_m
                ),
                "expert_reveal_max_can_rotation_rad": float(
                    self.minibench_expert_reveal_max_can_rotation_rad
                ),
                "expert_reveal_block_can_raw_pair_steps": int(
                    self.minibench_expert_reveal_block_can_raw_pair_steps
                ),
                "expert_reveal_block_can_meaningful_steps": int(
                    self.minibench_expert_reveal_block_can_meaningful_steps
                ),
                "expert_reveal_min_block_can_separation_m": (
                    None
                    if self.minibench_expert_reveal_min_block_can_separation_m is None
                    else float(
                        self.minibench_expert_reveal_min_block_can_separation_m
                    )
                ),
                "expert_reveal_max_block_can_impulse_ns": float(
                    self.minibench_expert_reveal_max_block_can_impulse_ns
                ),
                "native_parent_expert_completed": bool(
                    self.minibench_native_parent_expert_completed
                ),
                "expert_reveal_violation": deepcopy(
                    self.minibench_expert_reveal_violation
                ),
                "success": success,
                "success_step": (
                    self.minibench_policy_success_step if success else None
                ),
                "plan_success": bool(self.plan_success),
                "expert_trace": deepcopy(self.minibench_expert_trace),
            }
        )
