"""Perception-driven, whole-body safe carry V2 Isaac Sim task."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from omni.physx.bindings._physx import ContactEventType
from omni.isaac.sensor import Camera
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom
from scipy.spatial.transform import Rotation
from isaacsim.core.prims import SingleXFormPrim

from flowcarrycbf.policies.flowcarry_cbf.control.kinematics import FullBodyKinematics
from flowcarrycbf.policies.flowcarry_cbf.control.motion import precompute_motion_schedule
from flowcarrycbf.policies.flowcarry_cbf.control.scene import (
    effective_arm_avoidance,
    fixed_obstacle_models,
    phase3_quick_obstacle_models,
    QUICK_TEMPLATE_ALIASES,
    QUICK_TEMPLATE_IDS,
    random_obstacle_models,
    phase3_recording_camera_poses,
    stage2_camera_poses,
    stage1_obstacle_models,
    uses_external_cameras,
)
from flowcarrycbf.policies.flowcarry_cbf.control.schema import (
    ACCELERATION,
    CUBE_TYPE,
    HALF_EXTENTS,
    INSTANCE_COLORS,
    MAX_OBSTACLES,
    OBSTACLE_FEATURES,
    POSITION,
    SPHERE_TYPE,
    TYPE_INDEX,
    UNCERTAINTY_INDEX,
    VALID_INDEX,
    VELOCITY,
)
from flowcarrycbf.policies.flowcarry_cbf.eval_scenarios import (
    EVAL_TEMPLATE_IDS,
    canonical_template_id,
    eval_obstacle_models,
)
from flowcarrycbf.envs.tasks.tiago_dual_carry import TiagoDualCarryTask, _as_numpy
from flowcarrycbf.envs.tasks.utils.handled_box import create_physics_material
from flowcarrycbf.envs.tasks.utils.safe_obstacles_v2 import (
    V2ObstacleSlot,
    configure_v2_obstacle_slot,
    create_v2_obstacle_slot,
)
from flowcarrycbf.envs.tasks.utils.phase3_visuals import create_phase3_visuals


class FlowCarryCBFTask(TiagoDualCarryTask):
    """V2 task with randomized 3-D obstacles and truth isolated to logging."""

    def _base_is_stopped(self) -> bool:
        _, velocity = self.tiago_handler.get_base_dof_values()
        measured = _as_numpy(velocity).reshape(-1)[:3]
        return float(np.linalg.norm(measured[:2])) < 0.03 and abs(float(measured[2])) < 0.04

    def _allow_payload_slip_success(self) -> bool:
        # V2 separates task completion from grasp-quality scoring.  A payload
        # that remains supported and does not touch the environment may finish
        # without both handle contacts; the evaluator applies the grasp penalty.
        return True

    def _requires_both_contacts_for_success(self) -> bool:
        return False

    def __init__(self, name, sim_config, env) -> None:
        cfg = sim_config.task_config["env"]
        if int(cfg.get("max_obstacles", MAX_OBSTACLES)) != MAX_OBSTACLES:
            raise ValueError("V2 tensor schema requires exactly five obstacle slots")
        self._v2_scenario = str(cfg.get("obstacle_scenario", "fixed"))
        self._v2_template_id = str(cfg.get("obstacle_template_id", "")) or None
        self._v2_eval_template_id: str | None = None
        self._phase3_visuals = bool(cfg.get("phase3_visuals", False))
        self._enable_eval_video_cameras = bool(
            cfg.get("enable_eval_video_cameras", False)
        )
        self._v2_slots: list[V2ObstacleSlot] = []
        self._v2_prims: list[SingleXFormPrim] = []
        self._v2_models: list[dict[str, Any]] = []
        self._v2_collision_pairs: set[tuple[str, str]] = set()
        self._v2_collision = False
        self._v2_collision_entity = ""
        self._v2_table_collision_pairs: set[tuple[str, str]] = set()
        self._v2_table_collision = False
        self._v2_table_collision_entity = ""
        self._v2_table_collision_name = ""
        self._v2_elapsed = 0.0
        self._v2_schedule_index = 0
        self._v2_motion_schedules: dict[int, np.ndarray] = {}
        self._v2_minimum_clearance = float("inf")
        self._v2_minimum_table_clearance = float("inf")
        self._v2_minimum_table_clearance_entity = ""
        self._v2_current_table_clearance = float("inf")
        self._v2_current_table_clearance_entity = ""
        self._v2_below_8cm_frames = 0
        self._v2_payload_max_offset = 0.0
        self._v2_payload_recovery_position: np.ndarray | None = None
        self._v2_continuous_handle_contact = True
        self._v2_handle_contact_loss_frames = 0
        self._v2_layout_resamples = 0
        self._v2_cached_truth_obstacles: np.ndarray | None = None
        self._v2_lightweight_metrics = False
        self._v2_suppress_info_refresh = False
        self._v2_safety_metrics: dict[str, Any] = {}
        self._zero_robot_velocity_on_reset = True
        self._fixed_timeout_steps = int(round(float(cfg.get("fixed_timeout_seconds", 30.0)) / (float(sim_config.task_config["sim"]["dt"]) * int(cfg["controlFrequencyInv"]))))
        self._stage1_timeout_steps = int(round(float(cfg.get("stage1_timeout_seconds", 45.0)) / (float(sim_config.task_config["sim"]["dt"]) * int(cfg["controlFrequencyInv"]))))
        self._stage2_timeout_steps = int(round(float(cfg.get("stage2_timeout_seconds", 45.0)) / (float(sim_config.task_config["sim"]["dt"]) * int(cfg["controlFrequencyInv"]))))
        self._v2_kinematics = FullBodyKinematics(
            dt=float(sim_config.task_config["sim"]["dt"]) * int(cfg["controlFrequencyInv"]),
            max_arm_delta=float(cfg["max_arm_delta"]),
        )
        super().__init__(name, sim_config, env)

    def set_obstacle_scenario(self, scenario: str) -> None:
        if scenario not in {"stage1", "stage2", "fixed", "random", "phase3_quick"}:
            raise ValueError(f"unsupported V2 obstacle scenario: {scenario}")
        self._v2_scenario = scenario

    def set_obstacle_template(self, template_id: str | None) -> None:
        if template_id in {None, "", "none"}:
            self._v2_template_id = None
            return
        template_id = QUICK_TEMPLATE_ALIASES.get(str(template_id), str(template_id))
        if template_id not in QUICK_TEMPLATE_IDS:
            raise ValueError(f"unsupported Phase 3 quick template: {template_id}")
        self._v2_template_id = template_id

    def set_eval_obstacle_template(self, template_id: str | None) -> None:
        """Enable an eval-only template without extending collection choices."""

        if template_id in {None, "", "none"}:
            self._v2_eval_template_id = None
            return
        template_id = canonical_template_id(str(template_id))
        if template_id not in EVAL_TEMPLATE_IDS:
            raise ValueError(f"unsupported adaptive eval template: {template_id}")
        self._v2_eval_template_id = template_id

    def set_safety_metrics(self, **metrics: Any) -> None:
        self._v2_safety_metrics.update(metrics)

    def set_collection_mode(self, lightweight: bool) -> None:
        self._v2_lightweight_metrics = bool(lightweight)

    def _truth_obstacles(self) -> np.ndarray:
        if self._v2_cached_truth_obstacles is None:
            self._v2_cached_truth_obstacles = self.get_ground_truth_obstacles()
        return self._v2_cached_truth_obstacles

    @staticmethod
    def _camera_extrinsics(camera: Camera) -> np.ndarray:
        """Return the measured world-from-camera transform in USD optical axes."""

        position, quaternion = camera.get_world_pose(camera_axes="usd")
        # GPU PhysX returns camera poses as CUDA tensors; normalize them before
        # entering the NumPy-based tracker/extrinsics API.
        position = _as_numpy(position, dtype=np.float64).reshape(3)
        w, x, y, z = _as_numpy(quaternion, dtype=np.float64).reshape(4)
        rotation = np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        return transform

    def get_rgbd_extrinsics(self) -> np.ndarray:
        """Return the primary camera transform for the existing V2 API."""

        return self._camera_extrinsics(self._v2_cameras[0])

    def get_rgbd_extrinsics_all(self) -> list[np.ndarray]:
        cameras = self._v2_cameras if uses_external_cameras(self._v2_scenario) else self._v2_cameras[:1]
        return [self._camera_extrinsics(camera) for camera in cameras]

    def _read_rgbd_camera(self, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        height = int(self._task_cfg["env"]["camera_height"])
        width = int(self._task_cfg["env"]["camera_width"])
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width, 1), dtype=np.float32)
        try:
            rgba = camera.get_rgba()
            if rgba is not None:
                value = _as_numpy(rgba, dtype=np.uint8)
                if value.shape[:2] == (height, width):
                    rgb = value[..., :3].copy()
            frame = camera.get_current_frame()
            raw_depth = frame.get("distance_to_image_plane") if frame else None
            if raw_depth is not None:
                value = _as_numpy(raw_depth)
                if value.shape == (height, width):
                    depth[..., 0] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        except (AttributeError, RuntimeError, KeyError):
            pass
        return rgb, depth

    def get_rgbd_observations(self) -> list[tuple[np.ndarray, np.ndarray]]:
        cameras = self._v2_cameras if uses_external_cameras(self._v2_scenario) else self._v2_cameras[:1]
        return [self._read_rgbd_camera(camera) for camera in cameras]

    def get_rgb_observations(self) -> list[np.ndarray]:
        """Read Phase 3 training RGB without requesting depth annotators."""

        cameras = self._v2_cameras if uses_external_cameras(self._v2_scenario) else self._v2_cameras[:1]
        height = int(self._task_cfg["env"]["camera_height"])
        width = int(self._task_cfg["env"]["camera_width"])
        result = []
        for camera in cameras:
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            try:
                rgba = camera.get_rgba()
                if rgba is not None:
                    value = _as_numpy(rgba, dtype=np.uint8)
                    if value.shape[:2] == (height, width):
                        rgb = value[..., :3].copy()
            except (AttributeError, RuntimeError, KeyError):
                pass
            result.append(rgb)
        return result

    def get_eval_video_rgb_observations(self) -> list[np.ndarray]:
        """Read the two Phase 3 recording cameras without exposing them to FM."""

        if not self._enable_eval_video_cameras:
            return []
        height = int(self._task_cfg["env"].get("eval_camera_height", 720))
        width = int(self._task_cfg["env"].get("eval_camera_width", 1280))
        result = []
        for camera in self._eval_video_cameras:
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            try:
                rgba = camera.get_rgba()
                if rgba is not None:
                    value = _as_numpy(rgba, dtype=np.uint8)
                    if value.shape[:2] == (height, width):
                        rgb = value[..., :3].copy()
            except (AttributeError, RuntimeError, KeyError):
                pass
            result.append(rgb)
        return result

    def post_reset(self) -> None:
        # V2 uses a calibrated external RGB-D camera. V1 keeps the stock ZED.
        self.tiago_handler.post_reset(generate_camera=False)
        cfg = self._task_cfg["env"]
        self._v2_cameras = []
        self._eval_video_cameras = []
        if not bool(self._task_cfg["sim"].get("enable_cameras", True)):
            return
        for index in range(2):
            camera = Camera(
                prim_path=f"{self.tiago_handler.default_zero_env_path}/v2_rgbd_camera_{index}",
                name=f"safe_carry_v2_rgbd_camera_{index}",
                frequency=120,
                resolution=(int(cfg["camera_width"]), int(cfg["camera_height"])),
            )
            camera.initialize()
            if bool(cfg.get("enable_depth_cameras", True)):
                camera.add_distance_to_image_plane_to_frame()
            camera.set_clipping_range(0.05, 8.0)
            self.tiago_handler._set_cam_intrinsics(
                camera,
                float(cfg["fx"]),
                float(cfg["fy"]),
                float(cfg["cx"]),
                float(cfg["cy"]),
            )
            self._v2_cameras.append(camera)
        self._v2_camera = self._v2_cameras[0]
        self.tiago_handler.head_camera = self._v2_camera
        if self._enable_eval_video_cameras:
            width = int(cfg.get("eval_camera_width", 1280))
            height = int(cfg.get("eval_camera_height", 720))
            for index, name in enumerate(("first_person", "overview")):
                camera = Camera(
                    prim_path=(
                        f"{self.tiago_handler.default_zero_env_path}/"
                        f"phase3_{name}_camera"
                    ),
                    name=f"phase3_{name}_camera",
                    frequency=120,
                    resolution=(width, height),
                )
                camera.initialize()
                camera.set_clipping_range(0.05, 50.0)
                self.tiago_handler._set_cam_intrinsics(
                    camera,
                    700.0,
                    700.0,
                    width / 2.0,
                    height / 2.0,
                )
                self._eval_video_cameras.append(camera)
        self.sync_v2_camera()

    def sync_v2_camera(self) -> None:
        if not hasattr(self, "_v2_cameras"):
            return
        base = self._get_base_position()
        if uses_external_cameras(self._v2_scenario):
            camera_poses = stage2_camera_poses(base)
        else:
            # Preserve the existing V2 camera behavior outside the stage-two
            # bootstrap controller.
            camera_yaw = 0.0 if float(base[0]) < -2.80 else float(base[2])
            c, s = math.cos(camera_yaw), math.sin(camera_yaw)
            world_from_base = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            if float(base[0]) < -2.80:
                base_from_camera = np.asarray(
                    [[0.0, 0.8660254, -0.5], [-1.0, 0.0, 0.0], [0.0, 0.5, 0.8660254]],
                    dtype=np.float64,
                )
            else:
                base_from_camera = np.asarray(
                    [[0.0, -0.8660254, 0.5], [1.0, 0.0, 0.0], [0.0, 0.5, 0.8660254]],
                    dtype=np.float64,
                )
            rotation = world_from_base @ base_from_camera
            position = np.asarray([base[0], base[1], 0.0]) + world_from_base @ np.asarray([0.10, 0.0, 3.00])
            camera_poses = ((position, rotation),) * len(self._v2_cameras)
        for camera, (position, rotation) in zip(self._v2_cameras, camera_poses):
            quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
            quaternion_wxyz = np.roll(quaternion_xyzw, 1)
            camera.set_world_pose(
                position=position.astype(np.float32),
                orientation=quaternion_wxyz.astype(np.float32),
                camera_axes="usd",
            )
        if self._enable_eval_video_cameras:
            for camera, (position, rotation) in zip(
                self._eval_video_cameras, phase3_recording_camera_poses(base)
            ):
                quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
                camera.set_world_pose(
                    position=position.astype(np.float32),
                    orientation=np.roll(quaternion_xyzw, 1).astype(np.float32),
                    camera_axes="usd",
                )

    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)
        stage = self._env._world.stage
        if self._phase3_visuals:
            create_phase3_visuals(stage)
            root = self.tiago_handler.default_zero_env_path
            for table in ("A_table", "B_table"):
                for part in ("top", "leg_0", "leg_1", "leg_2", "leg_3"):
                    prim = stage.GetPrimAtPath(f"{root}/{table}/{part}")
                    if prim.IsValid():
                        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        material_path = "/World/Physics_Materials/SafeCarryV2Obstacle"
        create_physics_material(stage, material_path, 0.8, 0.7)
        root = self.tiago_handler.default_zero_env_path
        for index in range(MAX_OBSTACLES):
            slot = create_v2_obstacle_slot(
                stage,
                f"{root}/v2_obstacle_{index}",
                material_path,
                INSTANCE_COLORS[index],
            )
            self._v2_slots.append(slot)
            prim = SingleXFormPrim(slot.root, name=f"v2_obstacle_{index}")
            self._v2_prims.append(prim)
            scene.add(prim)

    def _on_contact_report_event(self, contact_headers, contact_data) -> None:
        super()._on_contact_report_event(contact_headers, contact_data)
        for header in contact_headers:
            pair = tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.collider0)), str(PhysicsSchemaTools.intToSdfPath(header.collider1)))))
            table_path = next((path for path in pair if "/A_table/" in path or "/B_table/" in path), "")
            if table_path:
                other = next((path for path in pair if path != table_path), "")
                if "/TiagoDualHolo/" in other or "/payload/" in other:
                    if header.type == ContactEventType.CONTACT_LOST:
                        self._v2_table_collision_pairs.discard(pair)
                    else:
                        self._v2_table_collision_pairs.add(pair)
                        self._v2_table_collision = True
                        self._v2_table_collision_name = "A_table" if "/A_table/" in table_path else "B_table"
                        if "/payload/" in other:
                            self._v2_table_collision_entity = "payload"
                        elif "/gripper_" in other:
                            self._v2_table_collision_entity = "gripper"
                        elif "/arm_" in other:
                            self._v2_table_collision_entity = "arm"
                        elif "/torso_" in other:
                            self._v2_table_collision_entity = "torso"
                        else:
                            self._v2_table_collision_entity = "base_or_body"
            if not any("/v2_obstacle_" in path for path in pair):
                continue
            other = next((path for path in pair if "/v2_obstacle_" not in path), "")
            if "/TiagoDualHolo/" not in other and "/payload/" not in other:
                continue
            if header.type == ContactEventType.CONTACT_LOST:
                self._v2_collision_pairs.discard(pair)
                continue
            self._v2_collision_pairs.add(pair)
            self._v2_collision = True
            if "/payload/" in other:
                self._v2_collision_entity = "payload"
            elif "/arm_" in other or "/gripper_" in other:
                self._v2_collision_entity = "arm_or_gripper"
            elif "/torso_" in other:
                self._v2_collision_entity = "torso"
            else:
                self._v2_collision_entity = "base_or_body"

    def _deactivate_v2_obstacles(self) -> None:
        if not self._v2_slots:
            return
        stage = self._env._world.stage
        for index, slot in enumerate(self._v2_slots):
            configure_v2_obstacle_slot(stage, slot, "inactive", (0.0, 0.0, 0.0))
            self._v2_prims[index].set_world_pose(
                position=np.asarray([0.0, 0.0, -10.0], dtype=np.float32),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

    def _fixed_models(self) -> list[dict[str, Any]]:
        return fixed_obstacle_models(self._rng)

    def _stage1_models(self) -> list[dict[str, Any]]:
        return stage1_obstacle_models()

    def _random_models(self) -> list[dict[str, Any]]:
        return random_obstacle_models(self._rng)

    def _phase3_quick_models(self) -> list[dict[str, Any]]:
        if self._v2_eval_template_id is not None:
            return eval_obstacle_models(self._rng, self._v2_eval_template_id)
        return phase3_quick_obstacle_models(self._rng, self._v2_template_id)

    def _layout_valid(self, models: list[dict[str, Any]]) -> bool:
        bearings: list[tuple[float, float, float]] = []
        for model in models:
            x, y, z = map(float, model["position"])
            forward = -x
            if forward <= 0.5 or abs(y) > 0.72 * forward:
                return False
            bearings.append((math.atan2(y, forward), forward, z))
        for first_index, first in enumerate(models):
            for second in models[first_index + 1:]:
                separation = np.abs(np.asarray(first["position"]) - np.asarray(second["position"]))
                required = np.asarray(first["half_extents"]) + np.asarray(second["half_extents"]) + 0.05
                if np.all(separation < required):
                    return False
        for first_index, first in enumerate(bearings):
            for second in bearings[first_index + 1:]:
                if abs(first[0] - second[0]) < 0.07 and abs(first[1] - second[1]) < 0.45 and abs(first[2] - second[2]) < 0.35:
                    return False
        static = [model for model in models if model["shape"] == "cube"]
        # Preserve a base-width passage on at least one side of every static x slice.
        for first_index, first in enumerate(static):
            for second in static[first_index + 1:]:
                if abs(float(first["position"][0] - second["position"][0])) < 0.55:
                    gap = abs(float(first["position"][1] - second["position"][1])) - float(first["half_extents"][1] + second["half_extents"][1])
                    if gap < 0.95 and np.sign(first["position"][1]) != np.sign(second["position"][1]):
                        return False
        return True

    def _configure_v2_obstacles(self) -> None:
        self._v2_layout_resamples = 0
        if self._v2_scenario == "stage1":
            models = self._stage1_models()
        elif self._v2_scenario == "stage2":
            models = self._fixed_models()
        elif self._v2_scenario == "fixed":
            models = self._fixed_models()
        elif self._v2_scenario == "random":
            for attempt in range(200):
                models = self._random_models()
                if self._layout_valid(models):
                    self._v2_layout_resamples = attempt
                    break
            else:
                raise RuntimeError("could not sample a feasible V2 obstacle layout")
        else:
            models = self._phase3_quick_models()
            if self._v2_eval_template_id is None and not self._layout_valid(models):
                raise RuntimeError("could not sample a feasible Phase 3 quick obstacle layout")
        self._v2_models = models
        self._v2_schedule_index = 0
        self._v2_motion_schedules = {}
        if self._v2_scenario in {"random", "phase3_quick"}:
            schedule_steps = int(self._max_episode_length) * int(self.control_frequency_inv)
            for index, model in enumerate(models):
                if model["motion"] is not None:
                    self._v2_motion_schedules[index] = precompute_motion_schedule(
                        model["motion"], steps=schedule_steps, dt=float(self._physics_dt)
                    )
        stage = self._env._world.stage
        for index, slot in enumerate(self._v2_slots):
            if index >= len(models):
                configure_v2_obstacle_slot(stage, slot, "inactive", (0.0, 0.0, 0.0))
                continue
            model = models[index]
            configure_v2_obstacle_slot(
                stage,
                slot,
                model["shape"],
                model["half_extents"],
                model.get("color"),
            )
            self._v2_prims[index].set_world_pose(
                position=model["position"],
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

    def advance_dynamic_obstacles(self, dt: float) -> None:
        self._v2_cached_truth_obstacles = None
        self._v2_elapsed += float(dt)
        base = self._get_base_position()
        if self._v2_scenario in {"random", "phase3_quick"}:
            self._v2_schedule_index += 1
        for index, model in enumerate(self._v2_models):
            motion = model["motion"]
            if motion is None:
                continue
            if self._v2_scenario in {"random", "phase3_quick"}:
                schedule = self._v2_motion_schedules[index]
                sample = schedule[min(self._v2_schedule_index, len(schedule) - 1)]
                model["position"][:] = sample[0:3]
                model["velocity"] = np.asarray(sample[3:6], dtype=np.float32)
                model["acceleration"] = np.asarray(sample[6:9], dtype=np.float32)
            else:
                radius = float(model["half_extents"][0])
                reaction_distance = 0.50 + 0.25 + radius + 0.10
                guarded = abs(float(model["position"][0]) - float(base[0])) <= reaction_distance
                retarget_guard = (float(base[1]), 0.72) if guarded else None
                sample = motion.step(dt, retarget_guard=retarget_guard)
                model["position"][1] = sample.position
                model["velocity"] = np.asarray([0.0, sample.velocity, 0.0], dtype=np.float32)
                model["acceleration"] = np.asarray([0.0, sample.acceleration, 0.0], dtype=np.float32)
            self._v2_prims[index].set_world_pose(position=model["position"], orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def get_oracle_obstacle_schedule(self, control_steps: int | None = None) -> np.ndarray:
        """Return world-frame future obstacles for the privileged collector only."""

        steps = int(self._max_episode_length if control_steps is None else control_steps)
        result = np.zeros((steps + 1, MAX_OBSTACLES, OBSTACLE_FEATURES), dtype=np.float32)
        stride = int(self.control_frequency_inv)
        for obstacle_index, model in enumerate(self._v2_models):
            result[:, obstacle_index, POSITION] = np.asarray(model["position"], dtype=np.float32)
            result[:, obstacle_index, HALF_EXTENTS] = np.asarray(model["half_extents"], dtype=np.float32)
            result[:, obstacle_index, TYPE_INDEX] = SPHERE_TYPE if model["shape"] == "sphere" else CUBE_TYPE
            result[:, obstacle_index, VALID_INDEX] = 1.0
            schedule = self._v2_motion_schedules.get(obstacle_index)
            if schedule is None:
                continue
            indices = np.minimum(np.arange(steps + 1) * stride, len(schedule) - 1)
            samples = schedule[indices]
            result[:, obstacle_index, POSITION] = samples[:, 0:3]
            result[:, obstacle_index, VELOCITY] = samples[:, 3:6]
            result[:, obstacle_index, ACCELERATION] = samples[:, 6:9]
        return result

    def get_ground_truth_obstacles(self) -> np.ndarray:
        result = np.zeros((MAX_OBSTACLES, OBSTACLE_FEATURES), dtype=np.float32)
        base = self._get_base_position()
        c, s = math.cos(float(base[2])), math.sin(float(base[2]))
        rotation = np.asarray([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        for index, model in enumerate(self._v2_models):
            relative = np.asarray(model["position"], dtype=np.float32) - np.asarray([base[0], base[1], 0.0], dtype=np.float32)
            result[index, POSITION] = rotation @ relative
            result[index, VELOCITY] = rotation @ np.asarray(model.get("velocity", np.zeros(3)), dtype=np.float32)
            result[index, ACCELERATION] = rotation @ np.asarray(model.get("acceleration", np.zeros(3)), dtype=np.float32)
            result[index, HALF_EXTENTS] = model["half_extents"]
            result[index, TYPE_INDEX] = SPHERE_TYPE if model["shape"] == "sphere" else CUBE_TYPE
            result[index, UNCERTAINTY_INDEX] = 0.0
            result[index, VALID_INDEX] = 1.0
        return result

    @staticmethod
    def _point_shape_clearance(point: np.ndarray, radius: float, obstacle: np.ndarray) -> float:
        if obstacle[TYPE_INDEX] < 0.5:
            return float(np.linalg.norm(point - obstacle[POSITION]) - radius - obstacle[HALF_EXTENTS][0])
        delta = np.abs(point - obstacle[POSITION]) - obstacle[HALF_EXTENTS]
        outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
        inside = min(float(np.max(delta)), 0.0)
        return outside + inside - radius

    def _truth_clearance(self) -> float:
        truth = self._truth_obstacles()
        dynamic_obstacle_count = len(truth)
        labels = [
            f"{model['shape']}_{index}" for index, model in enumerate(self._v2_models)
        ]
        labels.extend(
            f"inactive_{index}" for index in range(len(labels), len(truth))
        )
        state = self.obs_buf[0].detach().cpu().numpy()
        if self._v2_scenario in {"random", "phase3_quick"}:
            base = np.asarray(state[:3], dtype=np.float32)
            c, s = math.cos(float(base[2])), math.sin(float(base[2]))
            rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
            # Match the five actual PhysX colliders created by create_table:
            # one 0.8 x 1.0 x 0.06 tabletop and four 0.06 x 0.06 legs.
            # The Eval CBF still receives the conservative whole-table AABB;
            # this decomposition is only for collection-time clearance
            # reporting, so empty space under the tabletop is not classified
            # as occupied table volume.
            table_centers = np.asarray([[1.25, 0.0], [-4.25, 0.0]], dtype=np.float32)
            table_parts: list[np.ndarray] = []
            table_labels: list[str] = []
            for table_index, center_xy in enumerate(table_centers):
                parts = [
                    (np.asarray([0.0, 0.0, 0.72], dtype=np.float32),
                     np.asarray([0.40, 0.50, 0.03], dtype=np.float32), "top"),
                ]
                for leg_index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                    parts.append((
                        np.asarray([sx * 0.31, sy * 0.41, 0.345], dtype=np.float32),
                        np.asarray([0.03, 0.03, 0.345], dtype=np.float32),
                        f"leg_{leg_index}",
                    ))
                for local_offset, half_extents, part_name in parts:
                    obstacle = np.zeros(OBSTACLE_FEATURES, dtype=np.float32)
                    world_xy = center_xy + local_offset[:2]
                    obstacle[:2] = rotation @ (world_xy - base[:2])
                    obstacle[2] = local_offset[2]
                    obstacle[9:11] = np.abs(rotation) @ half_extents[:2]
                    obstacle[11] = half_extents[2]
                    obstacle[TYPE_INDEX] = CUBE_TYPE
                    obstacle[VALID_INDEX] = 1.0
                    table_parts.append(obstacle)
                    table_name = "A_table" if table_index == 0 else "B_table"
                    table_labels.append(f"{table_name}/{part_name}")
            truth = np.concatenate((truth, np.asarray(table_parts, dtype=np.float32)), axis=0)
            labels.extend(table_labels)
        rollout = self._v2_kinematics.rollout(state, np.zeros((1, self.ACTION_SIZE), dtype=np.float32))
        minimum = float("inf")
        entity = ""
        table_minimum = float("inf")
        table_entity = ""
        for obstacle_index, obstacle in enumerate(truth[:dynamic_obstacle_count]):
            if obstacle[VALID_INDEX] < 0.5:
                continue
            clearance = self._v2_kinematics.step_minimum_clearance(rollout, 0, obstacle)
            if clearance < minimum:
                minimum = clearance
                entity = labels[obstacle_index] if obstacle_index < len(labels) else f"obstacle_{obstacle_index}"
        for obstacle_index, obstacle in enumerate(truth[dynamic_obstacle_count:]):
            if obstacle[VALID_INDEX] < 0.5:
                continue
            clearance = self._v2_kinematics.step_minimum_clearance(rollout, 0, obstacle)
            if clearance < table_minimum:
                table_minimum = clearance
                label_index = dynamic_obstacle_count + obstacle_index
                table_entity = labels[label_index] if label_index < len(labels) else f"table_{obstacle_index}"
        # The table distance is diagnostic only during Phase 3 collection.
        # Table failure is determined by the real PhysX contact report.
        self._v2_current_table_clearance = table_minimum
        self._v2_current_table_clearance_entity = table_entity
        self._v2_current_clearance_entity = entity
        return minimum

    def reset_idx(self, env_ids) -> None:
        self._deactivate_v2_obstacles()
        self._v2_models = []
        self._v2_collision_pairs.clear()
        self._v2_collision = False
        self._v2_collision_entity = ""
        self._v2_table_collision_pairs.clear()
        self._v2_table_collision = False
        self._v2_table_collision_entity = ""
        self._v2_table_collision_name = ""
        self._v2_elapsed = 0.0
        # Fixed dynamic V2 starts at the center acquisition pose with zero
        # yaw.  The RGB-D camera observes both moving spheres here before the
        # base enters the side corridor; starting directly in that corridor
        # leaves only the ground cubes inside the calibrated field of view.
        self._initial_yaw_override = 0.0
        if self._v2_scenario == "stage2":
            self._initial_xy_override = np.zeros(2, dtype=np.float32)
        elif self._v2_scenario == "phase3_quick":
            self._initial_xy_override = np.zeros(2, dtype=np.float32)
        else:
            self._initial_xy_override = None
        # A fixed layout must also have a fixed calibrated grasp state.  Keep
        # the obstacle motion seed-randomized, but do not inject the generic
        # base/payload/mass/friction jitter into the pure-contact reset.  That
        # jitter makes consecutive episodes start with different handle
        # preload and can drop one handle during the first translation.
        randomize_episode = self._randomize_episode
        if self._v2_scenario in {"stage2", "phase3_quick"}:
            self._randomize_episode = False
        try:
            super().reset_idx(env_ids)
        finally:
            self._randomize_episode = randomize_episode
        self._initial_yaw_override = 0.0
        self._initial_xy_override = None
        self._configure_v2_obstacles()
        self._v2_minimum_clearance = float("inf")
        self._v2_minimum_clearance_entity = ""
        self._v2_minimum_table_clearance = float("inf")
        self._v2_minimum_table_clearance_entity = ""
        self._v2_current_table_clearance = float("inf")
        self._v2_current_table_clearance_entity = ""
        self._v2_current_clearance_entity = ""
        self._v2_below_8cm_frames = 0
        self._v2_payload_max_offset = 0.0
        self._v2_payload_recovery_position = None
        self._v2_cached_truth_obstacles = None
        self._v2_continuous_handle_contact = True
        self._v2_handle_contact_loss_frames = 0
        self._v2_arm_avoidance_frames = 0
        self._v2_effective_arm_avoidance = False
        self._v2_safety_metrics = {
            "qp_status": "not_run",
            "emergency_stops": 0,
            "planning_latency_ms": 0.0,
            "filter_latency_ms": 0.0,
            "cbf_intervention": 0.0,
            "stage2_phase": "not_run",
            "selected_lateral_target": 0.0,
            "selected_arm_region": "nominal",
            "arm_target_ready": 0,
        }

    def calculate_metrics(self) -> None:
        started = time.perf_counter()
        self._v2_suppress_info_refresh = True
        try:
            super().calculate_metrics()
        finally:
            self._v2_suppress_info_refresh = False
        if not (bool(self._last_contact["left"]) and bool(self._last_contact["right"])):
            self._v2_continuous_handle_contact = False
            self._v2_handle_contact_loss_frames += 1
        local_payload = np.asarray(self._last_payload_state.get("position", np.zeros(3)), dtype=np.float32)
        if np.any(local_payload):
            local_payload = self._payload_local_pose(local_payload)
            if self._v2_payload_recovery_position is None:
                # PhysX settles the freshly closed grippers by a small amount
                # during reset. Use that measured stable pose as the episode's
                # standard carry reference rather than the pre-contact ideal
                # pose, while keeping the recovery gate at 5 cm.
                self._v2_payload_recovery_position = local_payload.copy()
            offset = float(
                np.linalg.norm(
                    local_payload[1:3] - self._v2_payload_recovery_position[1:3]
                )
            )
            self._v2_payload_max_offset = max(self._v2_payload_max_offset, offset)
            effective_arm_motion = self._v2_scenario == "stage2" and effective_arm_avoidance(
                str(self._v2_safety_metrics.get("selected_arm_region", "nominal")),
                bool(self._v2_safety_metrics.get("arm_target_ready", 0)),
                offset,
            )
            if effective_arm_motion:
                self._v2_arm_avoidance_frames += 1
                self._v2_effective_arm_avoidance = True
        clearance = self._truth_clearance()
        if clearance < self._v2_minimum_clearance:
            self._v2_minimum_clearance = clearance
            self._v2_minimum_clearance_entity = self._v2_current_clearance_entity
        if self._v2_current_table_clearance < self._v2_minimum_table_clearance:
            self._v2_minimum_table_clearance = self._v2_current_table_clearance
            self._v2_minimum_table_clearance_entity = self._v2_current_table_clearance_entity
        hard_clearance_threshold = (
            0.08 if self._v2_scenario == "phase3_quick"
            else 0.06 if self._phase3_visuals else 0.08
        )
        if clearance < hard_clearance_threshold:
            self._v2_below_8cm_frames += 1
        if self._v2_collision:
            self._failure_reason = "obstacle_collision"
            self._terminated = True
            self._truncated = False
            self._success = False
            self.rew_buf[0] = -10.0
        elif self._v2_table_collision:
            self._failure_reason = f"{self._v2_table_collision_entity}_table_collision"
            self._terminated = True
            self._truncated = False
            self._success = False
            self.rew_buf[0] = -10.0
        self._v2_safety_metrics["metrics_latency_ms"] = 1000.0 * (time.perf_counter() - started)
        if not self._v2_lightweight_metrics:
            self.extras = self.get_info()

    def is_done(self) -> None:
        if self._v2_scenario == "stage1":
            maximum = self._stage1_timeout_steps
        elif self._v2_scenario == "stage2":
            maximum = self._stage2_timeout_steps
        elif self._v2_scenario == "fixed":
            maximum = self._fixed_timeout_steps
        else:
            maximum = self._max_episode_length
        if int(self.progress_buf[0]) >= maximum and not self._terminated:
            self._truncated = True
            self._failure_reason = "timeout"
        self.reset_buf[0] = int(self._terminated or self._truncated)

    def get_info(self) -> dict[str, Any]:
        if self._v2_suppress_info_refresh:
            return super().get_info()
        info = super().get_info()
        info.update({
            "obstacle_scenario": self._v2_scenario,
            "template_id": self._v2_eval_template_id or self._v2_template_id or "",
            "ground_truth_obstacles": self._truth_obstacles().tolist(),
            "obstacle_collision": bool(self._v2_collision),
            "obstacle_collision_entity": self._v2_collision_entity,
            "table_collision": bool(self._v2_table_collision),
            "table_collision_entity": self._v2_table_collision_entity,
            "table_collision_name": self._v2_table_collision_name,
            "minimum_true_clearance": float(self._v2_minimum_clearance),
            "minimum_clearance_entity": self._v2_minimum_clearance_entity,
            "minimum_table_clearance": float(self._v2_minimum_table_clearance),
            "minimum_table_clearance_entity": self._v2_minimum_table_clearance_entity,
            "clearance_below_8cm_frames": int(self._v2_below_8cm_frames),
            # Quick collection reports the 8 cm count but deliberately has no
            # clearance-based rejection gate; collision contacts remain hard.
            "clearance_below_hard_limit_frames": 0,
            "hard_clearance_threshold_m": 0.0 if self._v2_scenario == "phase3_quick" else (
                0.06 if self._phase3_visuals else 0.08
            ),
            "payload_max_avoidance_offset": float(self._v2_payload_max_offset),
            "arm_avoidance_frames": int(self._v2_arm_avoidance_frames),
            "effective_arm_avoidance": bool(self._v2_effective_arm_avoidance),
            "continuous_handle_contact": bool(self._v2_continuous_handle_contact),
            "handle_contact_loss_frames": int(self._v2_handle_contact_loss_frames),
            "base_stopped": bool(self._base_is_stopped()),
            "layout_resamples": int(self._v2_layout_resamples),
            **self._v2_safety_metrics,
        })
        return info


# Compatibility for code written against the pre-release task name.
TiagoDualSafeCarryV2Task = FlowCarryCBFTask
