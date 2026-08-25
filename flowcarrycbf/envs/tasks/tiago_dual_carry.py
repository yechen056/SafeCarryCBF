"""Pure-contact dual-arm payload carrying task for the Tiago Dual Holo."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import numpy as np
import torch
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from isaacsim.core.prims import SingleRigidPrim

from flowcarrycbf.robots.handlers.TiagoDualHandler import TiagoDualHandler
from flowcarrycbf.envs.tasks.base.rl_task import RLTask
from flowcarrycbf.envs.tasks.utils.handled_box import (
    create_handled_box,
    create_physics_material,
    create_table,
)
from flowcarrycbf.utils.files import get_usd_path


class CarryPhase(str, Enum):
    SETTLE = "SETTLE"
    ROTATE = "ROTATE"
    TRANSLATE = "TRANSLATE"
    HOLD = "HOLD"
    POLICY = "POLICY"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _as_numpy(value: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


class TiagoDualCarryTask(RLTask):
    """Carry one compound handled box from A to B without grasp/release logic."""

    # Calibrated with the bundled Pinocchio model.  Both grippers are symmetric,
    # their opening axes are nearly vertical and their fingers point inward.
    HOLD_ARM_CONFIGURATION = np.array(
        [0.6093, 0.8440, 0.8962, 1.8250, 0.3477, -0.9432, 0.8746],
        dtype=np.float32,
    )

    STATE_SIZE = 60
    ACTION_SIZE = 17

    def __init__(self, name, sim_config, env) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._env = env
        self._device = self._cfg["sim_device"]
        self._num_envs = int(self._task_cfg["env"]["numEnvs"])
        if self._num_envs != 1:
            raise ValueError("TiagoDualCarry currently supports exactly one environment")
        self._env_spacing = float(self._task_cfg["env"]["envSpacing"])
        self._max_episode_length = int(self._task_cfg["env"]["horizon"])
        self._num_observations = self.STATE_SIZE
        self._num_actions = self.ACTION_SIZE
        self._use_torso = bool(self._task_cfg["env"]["use_torso"])

        env_cfg = self._task_cfg["env"]
        self._physics_dt = float(self._task_cfg["sim"]["dt"])
        self._control_dt = self._physics_dt * int(env_cfg["controlFrequencyInv"])
        self._goal_xy = np.asarray(env_cfg["goal_xy"], dtype=np.float32)
        self._goal_yaw = float(env_cfg["goal_yaw"])
        self._goal_pos_thresh = float(env_cfg["goal_pos_thresh"])
        self._goal_yaw_thresh = float(env_cfg["goal_yaw_thresh"])
        self._hold_steps = max(1, int(round(float(env_cfg["hold_seconds"]) / self._control_dt)))
        self._hold_test_steps = max(
            1, int(round(float(env_cfg.get("hold_test_seconds", 10.0)) / self._control_dt))
        )
        self._required_hold_steps = self._hold_steps
        self._active_goal_xy = self._goal_xy.copy()
        self._active_goal_yaw = self._goal_yaw
        self._baseline_scenario = "full"

        self._max_linear_velocity = float(env_cfg["max_base_linear_velocity"])
        self._max_angular_velocity = float(env_cfg["max_base_angular_velocity"])
        self._max_linear_acceleration = float(env_cfg["max_base_linear_acceleration"])
        self._max_angular_acceleration = float(env_cfg["max_base_angular_acceleration"])
        self._max_arm_delta = float(env_cfg["max_arm_delta"])

        self._payload_dimensions = np.asarray(env_cfg["payload_dimensions"], dtype=np.float32)
        self._payload_nominal_mass = float(env_cfg["payload_mass"])
        self._payload_local_position = np.asarray(env_cfg["payload_local_position"], dtype=np.float32)
        self._handle_radius = float(env_cfg["handle_radius"])
        self._handle_length = float(env_cfg["handle_length"])
        self._handle_clearance = float(env_cfg["handle_clearance"])
        self._gripper_tool_outward_offset = float(
            env_cfg.get("gripper_tool_outward_offset", 0.04)
        )
        self._randomize_default = bool(env_cfg.get("randomize", True))

        self._pose_jitter = float(env_cfg["payload_pose_jitter"])
        self._angle_jitter = float(env_cfg["payload_angle_jitter"])
        self._mass_jitter = float(env_cfg["payload_mass_jitter"])
        self._friction_jitter = float(env_cfg["friction_jitter"])
        self._base_position_jitter = float(env_cfg["base_position_jitter"])
        self._base_yaw_jitter = float(env_cfg["base_yaw_jitter"])

        usd_path = (get_usd_path() / "tiago_dual_holobase_zed/tiago_dual_holobase_zed.usd").as_posix()
        self.tiago_handler = TiagoDualHandler(
            use_torso=self._use_torso,
            sim_config=self._sim_config,
            num_envs=self._num_envs,
            device=self._device,
            usd_path=usd_path,
            intrinsics=[env_cfg["fx"], env_cfg["fy"], env_cfg["cx"], env_cfg["cy"]],
        )
        hold = torch.as_tensor(self.HOLD_ARM_CONFIGURATION, device=self._device)
        self.tiago_handler.arm_left_start = hold.clone()
        self.tiago_handler.arm_right_start = hold.clone()
        self.tiago_handler.gripper_left_start = torch.full((2,), 0.035, device=self._device)
        self.tiago_handler.gripper_right_start = torch.full((2,), 0.035, device=self._device)
        self._nominal_grasp_quaternions: dict[str, np.ndarray] = {}
        for side, solver in (
            ("left", self.tiago_handler._ik_solver_left_arm),
            ("right", self.tiago_handler._ik_solver_right_arm),
        ):
            _, quaternion = solver.solve_fk_tiago(self.HOLD_ARM_CONFIGURATION)
            self._nominal_grasp_quaternions[side] = quaternion

        self._rng = np.random.default_rng(0)
        self._seed = 0
        self._randomize_episode = self._randomize_default
        self._phase = CarryPhase.SETTLE
        self._failure_reason = ""
        self._initialization_failed = False
        self._terminated = False
        self._truncated = False
        self._success = False
        self._hold_counter = 0
        self._previous_goal_distance = float(np.linalg.norm(self._goal_xy))
        self._base_targets = np.zeros(3, dtype=np.float32)
        self._base_command = np.zeros(3, dtype=np.float32)
        self._left_arm_targets = self.HOLD_ARM_CONFIGURATION.copy()
        self._right_arm_targets = self.HOLD_ARM_CONFIGURATION.copy()
        self._last_action = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        self._last_contact = {
            "left": False,
            "right": False,
            "left_force": 0.0,
            "right_force": 0.0,
            "left_box_contact": False,
            "right_box_contact": False,
            "left_grip_bar_contact": False,
            "right_grip_bar_contact": False,
            "classification_fallback": False,
        }
        self._last_payload_state: dict[str, np.ndarray | float] = {}
        self._episode_randomization: dict[str, Any] = {}
        self._last_goal_error = (float(np.linalg.norm(self._goal_xy)), math.pi)
        self._table_contact = False
        self._active_contact_pairs: dict[tuple[str, str], float] = {}
        self._contact_report_sub = None

        super().__init__(name, env)

    @property
    def phase(self) -> str:
        return self._phase.value

    def set_seed(self, seed: int) -> None:
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def set_episode_randomization(self, enabled: bool) -> None:
        self._randomize_episode = bool(enabled)

    def set_baseline_scenario(self, scenario: str) -> None:
        if scenario not in {"full", "hold", "rotate"}:
            raise ValueError(f"unsupported carry scenario: {scenario}")
        self._baseline_scenario = scenario

    def set_up_scene(self, scene) -> None:
        stage = self._env._world.stage
        self.tiago_handler.get_robot()

        table_material_path = "/World/Physics_Materials/CarryTable"
        create_physics_material(stage, table_material_path, 0.9, 0.8)
        env_root = self.tiago_handler.default_zero_env_path
        create_table(stage, env_root + "/A_table", (1.25, 0.0), table_material_path)
        create_table(stage, env_root + "/B_table", (-4.25, 0.0), table_material_path)

        self._payload_path = env_root + "/payload"
        self._payload_paths = create_handled_box(
            stage,
            self._payload_path,
            dimensions=self._payload_dimensions,
            mass=self._payload_nominal_mass,
            handle_radius=self._handle_radius,
            handle_length=self._handle_length,
            handle_clearance=self._handle_clearance,
        )
        self._payload_mass_api = UsdPhysics.MassAPI.Get(stage, self._payload_path)
        self._payload_physx_api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, self._payload_path)
        self._payload_material_api = UsdPhysics.MaterialAPI.Get(
            stage, self._payload_path + "/PhysicsMaterial"
        )

        self._enable_gripper_ccd()
        self._setup_contact_reporting()
        super().set_up_scene(scene, replicate_physics=False)

        self._robots = self.tiago_handler.create_articulation_view()
        scene.add(self._robots)
        self._payload = SingleRigidPrim(self._payload_path, name="carry_payload")
        scene.add(self._payload)
        self.set_initial_camera_params(camera_position=[4.5, 5.5, 3.0], camera_target=[-1.5, 0.0, 0.8])

    def _enable_gripper_ccd(self) -> None:
        stage = self._env._world.stage
        robot_root = self.tiago_handler.default_zero_env_path + "/TiagoDualHolo"
        for side in ("left", "right"):
            for finger in ("left", "right"):
                path = robot_root + f"/gripper_{side}_{finger}_finger_link"
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path))
                api.CreateEnableCCDAttr(True)
                api.CreateSolverPositionIterationCountAttr(8)
                api.CreateSolverVelocityIterationCountAttr(2)

    def _setup_contact_reporting(self) -> None:
        stage = self._env._world.stage
        report = PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(self._payload_path)
        )
        report.CreateThresholdAttr(0.0)
        self._contact_report_sub = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact_report_event
            )
        )

    def _on_contact_report_event(self, contact_headers, contact_data) -> None:
        for header in contact_headers:
            collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            if "/payload/" not in collider0 and "/payload/" not in collider1:
                continue
            pair = tuple(sorted((collider0, collider1)))
            if header.type == ContactEventType.CONTACT_LOST:
                self._active_contact_pairs.pop(pair, None)
                continue
            impulse = 0.0
            begin = int(header.contact_data_offset)
            end = begin + int(header.num_contact_data)
            for index in range(begin, end):
                value = contact_data[index].impulse
                impulse += math.sqrt(sum(float(component) ** 2 for component in value))
            self._active_contact_pairs[pair] = impulse / self._physics_dt

    def post_reset(self) -> None:
        self.tiago_handler.post_reset(generate_camera=bool(self._task_cfg["sim"]["enable_cameras"]))

    def _payload_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, orientation = self._payload.get_world_pose()
        return _as_numpy(position).reshape(-1)[:3], _as_numpy(orientation).reshape(-1)[:4]

    def _payload_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        linear = _as_numpy(self._payload.get_linear_velocity()).reshape(-1)[:3]
        angular = _as_numpy(self._payload.get_angular_velocity()).reshape(-1)[:3]
        return linear, angular

    def _set_payload_velocities(self, linear: torch.Tensor | np.ndarray, angular: torch.Tensor | np.ndarray) -> None:
        """Set both payload velocity components through the GPU-safe view API."""
        linear_value = _as_numpy(linear, dtype=np.float32).reshape(3)
        angular_value = _as_numpy(angular, dtype=np.float32).reshape(3)
        velocities = torch.as_tensor(
            np.concatenate((linear_value, angular_value), axis=0).reshape(1, 6),
            dtype=torch.float32,
            device=self._device,
        )
        self._payload._rigid_prim_view.set_velocities(velocities)

    def _payload_local_pose(self, position: np.ndarray) -> np.ndarray:
        base = self._get_base_position()
        c, s = math.cos(base[2]), math.sin(base[2])
        world_delta = position[:2] - base[:2]
        local_xy = np.array([c * world_delta[0] + s * world_delta[1], -s * world_delta[0] + c * world_delta[1]])
        return np.array([local_xy[0], local_xy[1], position[2]], dtype=np.float32)

    def _get_base_position(self) -> np.ndarray:
        position, _ = self.tiago_handler.get_base_dof_values()
        return _as_numpy(position).reshape(-1)[:3]

    def _base_is_stopped(self) -> bool:
        return float(np.linalg.norm(self._base_command)) < 0.03

    def _allow_payload_slip_success(self) -> bool:
        """Whether a stable but ungrasped payload may still finish the task."""
        return False

    def _requires_both_contacts_for_success(self) -> bool:
        return True

    def _get_contact_state(self) -> dict[str, float | bool]:
        result: dict[str, float | bool] = {
            "left": False,
            "right": False,
            "left_force": 0.0,
            "right_force": 0.0,
            "left_box_contact": False,
            "right_box_contact": False,
            "left_grip_bar_contact": False,
            "right_grip_bar_contact": False,
            "classification_fallback": False,
        }
        for pair, force in self._active_contact_pairs.items():
            for gripper_side in ("left", "right"):
                if not any(f"/gripper_{gripper_side}_" in path for path in pair):
                    continue
                same_handle = f"/payload/{gripper_side}_handle/"
                if any(same_handle in path for path in pair):
                    result[gripper_side] = True
                    result[gripper_side + "_force"] = (
                        float(result[gripper_side + "_force"]) + force
                    )
                    if any(
                        f"/payload/{gripper_side}_handle/grip_bar" in path
                        for path in pair
                    ):
                        result[gripper_side + "_grip_bar_contact"] = True
                elif any("/payload/" in path for path in pair):
                    result[gripper_side + "_box_contact"] = True
        return result

    def _payload_touches_environment(self) -> bool:
        for pair in self._active_contact_pairs:
            if not any("/payload/" in path for path in pair):
                continue
            if any("ground" in path.lower() or "_table/" in path.lower() for path in pair):
                return True
        return False

    def get_observations(self):
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        base_pos, base_vel = self.tiago_handler.get_base_dof_values()
        torso_pos, torso_vel = self.tiago_handler.get_torso_dof_values()
        left_pos, right_pos = self.tiago_handler.get_arms_dof_pos()
        left_vel, right_vel = self.tiago_handler.get_arms_dof_vel()
        left_gripper = self.tiago_handler.get_gripper_left_positions()
        right_gripper = self.tiago_handler.get_gripper_right_positions()
        payload_pos, payload_quat = self._payload_pose()
        payload_linear, payload_angular = self._payload_velocities()
        contact = self._get_contact_state()

        base_np = _as_numpy(base_pos).reshape(-1)[:3]
        dx_world = self._active_goal_xy - base_np[:2]
        c, s = math.cos(base_np[2]), math.sin(base_np[2])
        goal_relative = np.array(
            [
                c * dx_world[0] + s * dx_world[1],
                -s * dx_world[0] + c * dx_world[1],
                _wrap_angle(self._active_goal_yaw - base_np[2]),
            ],
            dtype=np.float32,
        )

        state = np.concatenate(
            [
                base_np,
                _as_numpy(base_vel).reshape(-1)[:3],
                _as_numpy(torso_pos).reshape(-1)[:1],
                _as_numpy(torso_vel).reshape(-1)[:1],
                _as_numpy(left_pos).reshape(-1)[:7],
                _as_numpy(right_pos).reshape(-1)[:7],
                _as_numpy(left_vel).reshape(-1)[:7],
                _as_numpy(right_vel).reshape(-1)[:7],
                _as_numpy(left_gripper).reshape(-1)[:2],
                _as_numpy(right_gripper).reshape(-1)[:2],
                payload_pos,
                payload_quat,
                payload_linear,
                payload_angular,
                goal_relative,
                np.array(
                    [
                        float(contact["left"]),
                        float(contact["right"]),
                        float(contact["left_force"]),
                        float(contact["right_force"]),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        if state.shape != (self.STATE_SIZE,):
            raise RuntimeError(f"carry state has shape {state.shape}, expected {(self.STATE_SIZE,)}")
        self.obs_buf[0] = torch.as_tensor(state, device=self._device)
        self._last_contact = contact
        return self.obs_buf

    def get_rgbd_observation(self) -> tuple[np.ndarray, np.ndarray]:
        height = int(self._task_cfg["env"]["camera_height"])
        width = int(self._task_cfg["env"]["camera_width"])
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width, 1), dtype=np.float32)
        try:
            rgba = self.tiago_handler.head_camera.get_rgba()
            if rgba is not None:
                rgba = _as_numpy(rgba, dtype=np.uint8)
                if rgba.shape[:2] == (height, width):
                    rgb = rgba[..., :3].copy()
            frame = self.tiago_handler.head_camera.get_current_frame()
            raw_depth = frame.get("distance_to_image_plane") if frame else None
            if raw_depth is not None:
                raw_depth = _as_numpy(raw_depth)
                if raw_depth.shape == (height, width):
                    raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
                    depth[..., 0] = raw_depth
        except (AttributeError, RuntimeError, KeyError):
            pass
        return rgb, depth

    def _rate_limit(self, current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        delta = desired - current
        linear_limit = self._max_linear_acceleration * self._control_dt
        angular_limit = self._max_angular_acceleration * self._control_dt
        delta[:2] = np.clip(delta[:2], -linear_limit, linear_limit)
        delta[2] = np.clip(delta[2], -angular_limit, angular_limit)
        return current + delta

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        if self._terminated or self._truncated:
            return

        action = np.clip(_as_numpy(actions).reshape(-1), -1.0, 1.0)
        if action.shape != (self.ACTION_SIZE,):
            raise ValueError(f"expected {(self.ACTION_SIZE,)} action, got {action.shape}")
        desired_command = np.array(
            [
                action[0] * self._max_linear_velocity,
                action[1] * self._max_linear_velocity,
                action[2] * self._max_angular_velocity,
            ],
            dtype=np.float32,
        )
        self._base_command = self._rate_limit(self._base_command, desired_command)

        yaw = float(self._base_targets[2])
        c, s = math.cos(yaw), math.sin(yaw)
        world_vx = c * self._base_command[0] - s * self._base_command[1]
        world_vy = s * self._base_command[0] + c * self._base_command[1]
        self._base_targets[0] += world_vx * self._control_dt
        self._base_targets[1] += world_vy * self._control_dt
        self._base_targets[2] = _wrap_angle(self._base_targets[2] + self._base_command[2] * self._control_dt)

        self._left_arm_targets += action[3:10] * self._max_arm_delta
        self._right_arm_targets += action[10:17] * self._max_arm_delta
        lower_left = _as_numpy(self.tiago_handler.arm_left_dof_lower)
        upper_left = _as_numpy(self.tiago_handler.arm_left_dof_upper)
        lower_right = _as_numpy(self.tiago_handler.arm_right_dof_lower)
        upper_right = _as_numpy(self.tiago_handler.arm_right_dof_upper)
        self._left_arm_targets = np.clip(self._left_arm_targets, lower_left, upper_left)
        self._right_arm_targets = np.clip(self._right_arm_targets, lower_right, upper_right)

        self.tiago_handler.set_base_velocity_targets(
            torch.as_tensor(
                [[world_vx, world_vy, self._base_command[2]]],
                dtype=torch.float32,
                device=self._device,
            )
        )
        self.tiago_handler.set_arm_position_targets(
            torch.as_tensor(self._left_arm_targets, device=self._device).unsqueeze(0),
            torch.as_tensor(self._right_arm_targets, device=self._device).unsqueeze(0),
        )
        closed = torch.zeros((1, 2), device=self._device)
        self.tiago_handler.set_gripper_left_position_targets(closed)
        self.tiago_handler.set_gripper_right_position_targets(closed)
        self._last_action = action.astype(np.float32)

    def _set_payload_randomization(self) -> tuple[float, float, float]:
        if self._randomize_episode:
            mass_scale = 1.0 + self._rng.uniform(-self._mass_jitter, self._mass_jitter)
            friction_scale = 1.0 + self._rng.uniform(-self._friction_jitter, self._friction_jitter)
            pose_scale = 1.0
        else:
            mass_scale = friction_scale = pose_scale = 1.0
        self._payload_mass_api.GetMassAttr().Set(self._payload_nominal_mass * mass_scale)
        self._payload_material_api.GetStaticFrictionAttr().Set(1.4 * friction_scale)
        self._payload_material_api.GetDynamicFrictionAttr().Set(1.2 * friction_scale)
        return mass_scale, friction_scale, pose_scale

    def _solve_reset_grasp_ik(
        self,
        payload_position: np.ndarray,
        payload_rotation: Rotation,
        base_xy: np.ndarray,
        base_yaw: float,
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """Solve both arms for the two perturbed grip-bar poses in robot frame."""

        base_rotation = Rotation.from_euler("z", base_yaw)
        payload_position_robot = base_rotation.inv().apply(
            payload_position - np.array([base_xy[0], base_xy[1], 0.0])
        )
        payload_rotation_robot = base_rotation.inv() * payload_rotation
        grip_y = 0.5 * float(self._payload_dimensions[1]) + self._handle_clearance
        solutions: dict[str, np.ndarray] = {}
        all_success = True
        for side, sign, solver in (
            ("left", 1.0, self.tiago_handler._ik_solver_left_arm),
            ("right", -1.0, self.tiago_handler._ik_solver_right_arm),
        ):
            target_position = payload_position_robot + payload_rotation_robot.apply(
                np.array(
                    [
                        0.0,
                        sign * (grip_y + self._gripper_tool_outward_offset),
                        0.0,
                    ]
                )
            )
            nominal_quaternion = self._nominal_grasp_quaternions[side]
            nominal_rotation = Rotation.from_quat(nominal_quaternion[[1, 2, 3, 0]])
            target_rotation = payload_rotation_robot * nominal_rotation
            quaternion_xyzw = target_rotation.as_quat()
            target_quaternion = quaternion_xyzw[[3, 0, 1, 2]]
            success, solution = solver.solve_ik_pos_tiago(
                target_position,
                target_quaternion,
                curr_joints=self.HOLD_ARM_CONFIGURATION.copy(),
                n_trials=2,
                dt=0.05,
                pos_threshold=0.002,
                angle_threshold=math.radians(1.0),
            )
            all_success = all_success and bool(success)
            solutions[side] = np.asarray(solution, dtype=np.float32)
        return all_success, solutions["left"], solutions["right"]

    def reset_idx(self, env_ids) -> None:
        indices = env_ids.to(dtype=torch.int32)
        self.tiago_handler.reset(indices, randomize=False)
        if getattr(self, "_zero_robot_velocity_on_reset", False):
            # Isaac's articulation reset restores positions but can preserve
            # residual joint/base velocities from the preceding episode.
            # V2's pure-contact grasp is sensitive to that impulse during the
            # first lateral translation; clear it before the payload is
            # reattached.  The flag is opt-in so V1 reset behavior is intact.
            default_state = self.tiago_handler.robots.get_joints_default_state()
            zero_velocity = torch.zeros(
                (len(indices), default_state.positions.shape[-1]),
                dtype=torch.float32,
                device=self._device,
            )
            self.tiago_handler.robots.set_joint_velocities(
                zero_velocity,
                indices=indices,
            )

        # Break every previous payload contact before clearing the event cache.
        # PhysX otherwise keeps persistent pairs alive without another FOUND
        # event when two consecutive resets happen to use similar poses.
        self._payload_physx_api.GetDisableGravityAttr().Set(True)
        self._payload.set_world_pose(
            position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        zero_velocity = torch.zeros(3, device=self._device)
        self._set_payload_velocities(zero_velocity, zero_velocity)
        for _ in range(10):
            self._env._world.step(render=False)
        self._active_contact_pairs.clear()

        if self._randomize_episode:
            base_xy = self._rng.uniform(-self._base_position_jitter, self._base_position_jitter, size=2)
            base_yaw = float(self._rng.uniform(-self._base_yaw_jitter, self._base_yaw_jitter))
            payload_jitter = self._rng.uniform(-self._pose_jitter, self._pose_jitter, size=3)
            angle_jitter = self._rng.uniform(-self._angle_jitter, self._angle_jitter, size=3)
        else:
            override_xy = getattr(self, "_initial_xy_override", None)
            base_xy = np.asarray(override_xy, dtype=np.float64).reshape(2) if override_xy is not None else np.zeros(2)
            base_yaw = float(getattr(self, "_initial_yaw_override", 0.0))
            payload_jitter = np.zeros(3)
            angle_jitter = np.zeros(3)

        self._base_targets = np.array([base_xy[0], base_xy[1], base_yaw], dtype=np.float32)
        if self._baseline_scenario == "hold":
            self._active_goal_xy = self._base_targets[:2].copy()
            self._active_goal_yaw = float(self._base_targets[2])
            self._required_hold_steps = self._hold_test_steps
        elif self._baseline_scenario == "rotate":
            self._active_goal_xy = self._base_targets[:2].copy()
            self._active_goal_yaw = self._goal_yaw
            self._required_hold_steps = self._hold_steps
        else:
            self._active_goal_xy = self._goal_xy.copy()
            self._active_goal_yaw = self._goal_yaw
            self._required_hold_steps = self._hold_steps
        self._base_command.fill(0.0)
        self.tiago_handler.set_base_positions(
            torch.as_tensor(self._base_targets, device=self._device).unsqueeze(0)
        )
        self.tiago_handler.set_base_velocity_targets(
            torch.zeros((1, 3), dtype=torch.float32, device=self._device)
        )
        c, s = math.cos(base_yaw), math.sin(base_yaw)
        local = self._payload_local_position.copy()
        payload_position = np.array(
            [
                base_xy[0] + c * local[0] - s * local[1],
                base_xy[1] + s * local[0] + c * local[1],
                local[2],
            ],
            dtype=np.float32,
        )
        payload_position += payload_jitter.astype(np.float32)
        payload_rotation = Rotation.from_euler("z", base_yaw) * Rotation.from_euler("xyz", angle_jitter)
        payload_quat_xyzw = payload_rotation.as_quat()
        payload_quat = payload_quat_xyzw[[3, 0, 1, 2]].astype(np.float32)
        ik_success, self._left_arm_targets, self._right_arm_targets = self._solve_reset_grasp_ik(
            payload_position, payload_rotation, base_xy, base_yaw
        )
        left_targets = torch.as_tensor(
            self._left_arm_targets, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        right_targets = torch.as_tensor(
            self._right_arm_targets, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        self.tiago_handler.set_arm_positions(left_targets, right_targets)
        self.tiago_handler.set_arm_position_targets(left_targets, right_targets)
        self._payload.set_world_pose(position=payload_position, orientation=payload_quat)
        zero_velocity = torch.zeros(3, dtype=torch.float32, device=self._device)
        self._set_payload_velocities(zero_velocity, zero_velocity)
        mass_scale, friction_scale, _ = self._set_payload_randomization()
        self._episode_randomization = {
            "base_xy": base_xy.astype(float).tolist(),
            "base_yaw": float(base_yaw),
            "payload_position_jitter": payload_jitter.astype(float).tolist(),
            "payload_angle_jitter": angle_jitter.astype(float).tolist(),
            "payload_mass_scale": float(mass_scale),
            "payload_friction_scale": float(friction_scale),
        }

        self._payload_physx_api.GetDisableGravityAttr().Set(True)
        closed = torch.zeros((1, 2), device=self._device)
        self.tiago_handler.set_gripper_left_position_targets(closed)
        self.tiago_handler.set_gripper_right_position_targets(closed)
        for index in range(60):
            self._env._world.step(render=self._env._run_sim_rendering and index == 59)
        self._payload_physx_api.GetDisableGravityAttr().Set(False)
        contact_window = []
        for index in range(60):
            self._env._world.step(render=self._env._run_sim_rendering and index == 59)
            if index >= 57:
                contact_window.append(self._get_contact_state())

        contact = contact_window[-1]
        for side in ("left", "right"):
            contact[side] = any(bool(frame[side]) for frame in contact_window)
            contact[side + "_grip_bar_contact"] = any(
                bool(frame[side + "_grip_bar_contact"]) for frame in contact_window
            )
            contact[side + "_box_contact"] = any(
                bool(frame[side + "_box_contact"]) for frame in contact_window
            )
            contact[side + "_force"] = max(
                float(frame[side + "_force"]) for frame in contact_window
            )
        settled_position, _ = self._payload_pose()
        settled_local = self._payload_local_pose(settled_position)
        slip = float(np.linalg.norm(settled_local - self._payload_local_position))
        self._initialization_failed = (
            not ik_success
            or not (bool(contact["left"]) and bool(contact["right"]))
            or slip > 0.12
        )
        if not ik_success:
            self._failure_reason = "initial_ik_failed"
        elif self._initialization_failed:
            self._failure_reason = "initial_contact_failed"
        else:
            self._failure_reason = ""
        if self._initialization_failed:
            self._phase = CarryPhase.FAILURE
        elif self._baseline_scenario == "hold":
            self._phase = CarryPhase.HOLD
        else:
            self._phase = CarryPhase.ROTATE
        self._terminated = False
        self._truncated = False
        self._success = False
        self._hold_counter = 0
        self._previous_goal_distance = float(
            np.linalg.norm(self._active_goal_xy - self._base_targets[:2])
        )
        self._last_action.fill(0.0)
        self._last_contact = contact
        self._table_contact = False
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.rew_buf[env_ids] = 0.0
        self.extras = {}

    def _base_collides_with_table(self, base: np.ndarray) -> bool:
        half_top = np.array([0.4, 0.5], dtype=np.float32)
        robot_radius = 0.42
        for center in (np.array([1.25, 0.0]), np.array([-4.25, 0.0])):
            if np.all(np.abs(base[:2] - center) < half_top + robot_radius):
                return True
        return False

    def calculate_metrics(self) -> None:
        base = self._get_base_position()
        goal_distance = float(np.linalg.norm(self._active_goal_xy - base[:2]))
        yaw_error = abs(_wrap_angle(self._active_goal_yaw - base[2]))
        payload_position, payload_quat = self._payload_pose()
        payload_linear, payload_angular = self._payload_velocities()
        payload_local = self._payload_local_pose(payload_position)
        payload_slip = float(np.linalg.norm(payload_local - self._payload_local_position))
        payload_rotation = Rotation.from_quat(payload_quat[[1, 2, 3, 0]])
        payload_tilt = float(np.linalg.norm(payload_rotation.as_euler("xyz")[:2]))
        contact = self._get_contact_state()
        self._table_contact = self._payload_touches_environment()

        failure_reason = ""
        if self._initialization_failed:
            failure_reason = self._failure_reason or "initial_contact_failed"
        elif payload_position[2] < 0.45:
            failure_reason = "payload_dropped"
        elif payload_slip > 0.22 and not self._allow_payload_slip_success():
            failure_reason = "payload_left_grippers"
        elif self._table_contact:
            failure_reason = "payload_touched_environment"
        elif self._base_collides_with_table(base):
            failure_reason = "base_table_collision"

        stopped = self._base_is_stopped()
        at_goal = goal_distance <= self._goal_pos_thresh and yaw_error <= self._goal_yaw_thresh
        both_contacts = bool(contact["left"]) and bool(contact["right"])
        contact_requirement = both_contacts or not self._requires_both_contacts_for_success()
        if not failure_reason and at_goal and stopped and contact_requirement:
            self._hold_counter += 1
        else:
            self._hold_counter = 0

        if failure_reason:
            self._failure_reason = failure_reason
            self._terminated = True
            self._success = False
            self._phase = CarryPhase.FAILURE
        elif self._hold_counter >= self._required_hold_steps:
            self._terminated = True
            self._success = True
            self._phase = CarryPhase.SUCCESS

        progress_reward = 5.0 * (self._previous_goal_distance - goal_distance)
        action_penalty = 0.001 * float(np.square(self._last_action).sum())
        reward = progress_reward - action_penalty
        if self._success:
            reward += 10.0
        elif self._terminated:
            reward -= 10.0
        self.rew_buf[0] = reward
        self._previous_goal_distance = goal_distance
        self._last_goal_error = (goal_distance, yaw_error)
        self._last_contact = contact
        self._last_payload_state = {
            "position": payload_position,
            "orientation": payload_quat,
            "linear_velocity": payload_linear,
            "angular_velocity": payload_angular,
            "height": float(payload_position[2]),
            "tilt": payload_tilt,
            "slip": payload_slip,
        }
        self.extras = self.get_info()

    def is_done(self) -> None:
        if int(self.progress_buf[0]) >= self._max_episode_length and not self._terminated:
            self._truncated = True
            self._failure_reason = "timeout"
        self.reset_buf[0] = int(self._terminated or self._truncated)

    def get_info(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "scenario": self._baseline_scenario,
            "seed": self._seed,
            "success": bool(self._success),
            "failure_reason": self._failure_reason,
            "goal_distance": float(self._last_goal_error[0]),
            "goal_yaw_error": float(self._last_goal_error[1]),
            "left_handle_contact": bool(self._last_contact["left"]),
            "right_handle_contact": bool(self._last_contact["right"]),
            "left_box_contact": bool(self._last_contact["left_box_contact"]),
            "right_box_contact": bool(self._last_contact["right_box_contact"]),
            "left_grip_bar_contact": bool(
                self._last_contact["left_grip_bar_contact"]
            ),
            "right_grip_bar_contact": bool(
                self._last_contact["right_grip_bar_contact"]
            ),
            "contact_classification_fallback": bool(
                self._last_contact["classification_fallback"]
            ),
            "left_contact_force": float(self._last_contact["left_force"]),
            "right_contact_force": float(self._last_contact["right_force"]),
            "payload_height": float(self._last_payload_state.get("height", self._payload_local_position[2])),
            "payload_tilt": float(self._last_payload_state.get("tilt", 0.0)),
            "payload_slip": float(self._last_payload_state.get("slip", 0.0)),
            "payload_environment_contact": bool(self._table_contact),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "step": int(self.progress_buf[0]),
            "randomization": dict(self._episode_randomization),
        }

    def baseline_action(self) -> np.ndarray:
        action = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        if self._terminated or self._truncated or self._initialization_failed:
            return action
        base = self._get_base_position()
        if self._phase == CarryPhase.POLICY:
            self._phase = CarryPhase.ROTATE

        if self._phase == CarryPhase.ROTATE:
            error = _wrap_angle(self._active_goal_yaw - base[2])
            desired_w = math.copysign(
                min(self._max_angular_velocity, math.sqrt(max(0.0, 2.0 * self._max_angular_acceleration * abs(error)))),
                error,
            )
            action[2] = desired_w / self._max_angular_velocity
            if abs(error) < math.radians(1.5) and abs(self._base_command[2]) < 0.04:
                self._phase = CarryPhase.TRANSLATE
        elif self._phase == CarryPhase.TRANSLATE:
            delta_world = self._active_goal_xy - base[:2]
            distance = float(np.linalg.norm(delta_world))
            c, s = math.cos(base[2]), math.sin(base[2])
            local = np.array(
                [c * delta_world[0] + s * delta_world[1], -s * delta_world[0] + c * delta_world[1]],
                dtype=np.float32,
            )
            desired_speed = min(
                self._max_linear_velocity,
                math.sqrt(max(0.0, 2.0 * self._max_linear_acceleration * distance)),
            )
            if distance > 1.0e-5:
                desired_local = desired_speed * local / distance
                action[0:2] = desired_local / self._max_linear_velocity
            yaw_error = _wrap_angle(self._active_goal_yaw - base[2])
            action[2] = np.clip(1.5 * yaw_error / self._max_angular_velocity, -1.0, 1.0)
            if distance < 0.08:
                self._phase = CarryPhase.HOLD
        return np.clip(action, -1.0, 1.0)

    def close(self) -> None:
        self._contact_report_sub = None
