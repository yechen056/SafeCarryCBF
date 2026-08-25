"""Gymnasium wrapper for perception-only Tiago Dual Safe Carry V2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf

from flowcarrycbf.policies.flowcarry_cbf.control.perception import RGBDObstacleTracker
from flowcarrycbf.policies.flowcarry_cbf.control.schema import MAX_OBSTACLES, OBSTACLE_FEATURES
from flowcarrycbf.policies.flowcarry_cbf.control.scene import uses_external_cameras


ACTIVE_PERCEPTION_YAW_THRESHOLD = np.deg2rad(12.0)


def active_perception_acquiring(state: np.ndarray) -> bool:
    value = np.asarray(state, dtype=np.float32).reshape(60)
    # Keep the legacy acquisition contract for the central entry strip, but
    # never reset tracks in the certified side corridor or at the return
    # waypoint where the camera direction is explicitly scheduled.
    return (
        abs(float(value[55])) > ACTIVE_PERCEPTION_YAW_THRESHOLD
        and abs(float(value[1])) < 0.8
        and float(value[0]) > -2.8
    )


def perception_sweep_blocks_planning(scenario: str, state: np.ndarray) -> bool:
    """Only robot-mounted camera modes need the legacy turn-time sweep gate."""

    return not uses_external_cameras(scenario) and active_perception_acquiring(state)


class FlowCarryCBFEnv(gym.Env):
    """The policy observation is estimated from RGB-D; truth is info-only."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        headless: bool = True,
        render: bool = True,
        randomize: bool = True,
        obstacle_scenario: str = "fixed",
        lightweight_info: bool = False,
        phase3_visuals: bool = False,
        enable_eval_video_cameras: bool = False,
        adaptive_eval: bool = False,
        observation_mode: str = "perception",
        worker_threads: int = 4,
        physics_device: str = "cpu",
        config_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if observation_mode not in {"perception", "truth", "record"}:
            raise ValueError("observation_mode must be perception, truth or record")
        if physics_device not in {"cpu", "gpu"}:
            raise ValueError("physics_device must be cpu or gpu")
        if int(worker_threads) <= 0:
            raise ValueError("worker_threads must be positive")
        self._observation_mode = observation_mode
        self._capture_cameras = observation_mode != "truth"
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "config/FlowCarryCBF.yaml"
        task_config = OmegaConf.load(Path(config_path))
        task_config.env.randomize = bool(randomize)
        task_config.env.obstacle_scenario = obstacle_scenario
        task_config.env.phase3_visuals = bool(phase3_visuals)
        task_config.env.enable_eval_video_cameras = bool(enable_eval_video_cameras)
        task_config.env.enable_depth_cameras = observation_mode == "perception"
        task_config.sim.enable_cameras = self._capture_cameras
        task_config.sim.physx.worker_thread_count = int(worker_threads)
        use_cuda = physics_device == "gpu"
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("GPU PhysX requested but CUDA is unavailable")
        task_config.sim.use_gpu_pipeline = use_cuda
        task_config.sim.physx.use_gpu = use_cuda
        root_config = OmegaConf.create({
            "task_name": "FlowCarryCBF", "physics_engine": "physx",
            "headless": bool(headless), "render": self._capture_cameras, "test": True, "seed": 0,
            "pipeline": "cuda" if use_cuda else "cpu",
            "sim_device": "cuda:0" if use_cuda else "cpu", "device_id": 0,
            "rl_device": "cuda:0" if use_cuda else "cpu", "num_threads": int(worker_threads), "solver_type": 1, "task": task_config,
        })
        from flowcarrycbf.envs.isaac_env_mushroom import IsaacEnvMushroom
        from flowcarrycbf.utils.task_util import initialize_task

        self._backend = IsaacEnvMushroom(headless=bool(headless), render=self._capture_cameras, sim_app_cfg_path="")
        self.task = initialize_task(OmegaConf.to_container(root_config, resolve=True), self._backend)
        if hasattr(self.task, "set_collection_mode"):
            self.task.set_collection_mode(bool(lightweight_info))
        self.task.set_episode_randomization(bool(randomize))
        self.task.set_baseline_scenario("full")
        self.task.set_obstacle_scenario(obstacle_scenario)
        self._tracker = RGBDObstacleTracker(
            float(task_config.env.fx), float(task_config.env.fy),
            float(task_config.env.cx), float(task_config.env.cy),
            sphere_radius_bounds=(0.08, 0.28) if adaptive_eval else (0.08, 0.18),
            cube_size_bounds=(0.30, 0.60) if adaptive_eval else (0.30, 0.55),
            maximum_sphere_span=0.58 if adaptive_eval else 0.30,
        )
        self.render_mode = "rgb_array" if headless else "human"
        self._closed = False
        self._lightweight_info = bool(lightweight_info)
        self._last_observation: dict[str, np.ndarray] | None = None
        self._last_rgb_views: list[np.ndarray] = []
        height, width = int(task_config.env.camera_height), int(task_config.env.camera_width)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(60,), dtype=np.float32),
            "obstacles": spaces.Box(-np.inf, np.inf, shape=(MAX_OBSTACLES, OBSTACLE_FEATURES), dtype=np.float32),
            "rgb": spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8),
            "depth": spaces.Box(0.0, np.inf, shape=(height, width, 1), dtype=np.float32),
        })

    def _make_observation(self, state: np.ndarray) -> dict[str, np.ndarray]:
        height = int(self.observation_space["rgb"].shape[0])
        width = int(self.observation_space["rgb"].shape[1])
        state = np.asarray(state, dtype=np.float32).reshape(60)
        if self._observation_mode == "truth":
            obstacles = self.task.get_ground_truth_obstacles()
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            depth = np.zeros((height, width, 1), dtype=np.float32)
            self._last_rgb_views = []
        elif self._observation_mode == "record":
            self._last_rgb_views = self.task.get_rgb_observations()
            rgb = self._last_rgb_views[0]
            depth = np.zeros((height, width, 1), dtype=np.float32)
            obstacles = self.task.get_ground_truth_obstacles()
        else:
            rgbd_views = self.task.get_rgbd_observations()
            transforms = self.task.get_rgbd_extrinsics_all()
            rgb, depth = rgbd_views[0]
            self._last_rgb_views = [np.asarray(view_rgb, dtype=np.uint8) for view_rgb, _ in rgbd_views]
        # During the initial in-place camera sweep, detections at the edge of
        # the view are deliberately treated as acquisition candidates. Start
        # temporal tracking only after the goal corridor is facing the camera;
        # this avoids declaring a freshly acquired sphere "lost" mid-sweep.
            acquiring = perception_sweep_blocks_planning(self.task._v2_scenario, state)
            if acquiring:
                self._tracker.reset(np.full(MAX_OBSTACLES, -1.0, dtype=np.float32))
                obstacles = self._tracker.observation()
            else:
                tracker_views = [
                    (view_rgb, view_depth, transform)
                    for (view_rgb, view_depth), transform in zip(rgbd_views, transforms)
                ]
                obstacles = self._tracker.update_views(
                    tracker_views, state[:3], dt=float(self.task._control_dt)
                )
        observation = {
            "state": state,
            "obstacles": obstacles,
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "depth": np.asarray(depth, dtype=np.float32),
        }
        self._last_observation = observation
        return observation

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        seed = 0 if seed is None else int(seed)
        self.task.set_seed(seed)
        if options:
            if "randomize" in options:
                self.task.set_episode_randomization(bool(options["randomize"]))
            if "obstacle_scenario" in options:
                self.task.set_obstacle_scenario(str(options["obstacle_scenario"]))
            if "obstacle_template_id" in options:
                self.task.set_obstacle_template(options["obstacle_template_id"])
            if "eval_obstacle_template_id" in options:
                self.task.set_eval_obstacle_template(
                    options["eval_obstacle_template_id"]
                )
        self._tracker.reset()
        self.task.reset()
        if self._capture_cameras:
            self.task.sync_v2_camera()
        # RTX camera annotators need a few render-only frames after USD reset.
        # No physics or moving-obstacle time advances during this warm-up.
        if self._capture_cameras:
            for _ in range(3):
                self._backend._world.render()
        state = self.task.get_observations()[0].detach().cpu().numpy()
        observation = self._make_observation(state)
        info = self._info_for_step(force_full=not self._lightweight_info)
        return observation, info

    def _minimal_info(self) -> dict[str, Any]:
        task = self.task
        contact = getattr(task, "_last_contact", {})
        return {
            "success": bool(getattr(task, "_success", False)),
            "failure_reason": str(getattr(task, "_failure_reason", "")),
            "left_handle_contact": bool(contact.get("left", False)),
            "right_handle_contact": bool(contact.get("right", False)),
            "continuous_handle_contact": bool(getattr(task, "_v2_continuous_handle_contact", True)),
            "obstacle_collision": bool(getattr(task, "_v2_collision", False)),
            "obstacle_collision_entity": str(getattr(task, "_v2_collision_entity", "")),
            "table_collision": bool(getattr(task, "_v2_table_collision", False)),
            "table_collision_entity": str(getattr(task, "_v2_table_collision_entity", "")),
            "table_collision_name": str(getattr(task, "_v2_table_collision_name", "")),
            "clearance_below_8cm_frames": int(getattr(task, "_v2_below_8cm_frames", 0)),
            "clearance_below_hard_limit_frames": 0 if str(getattr(task, "_v2_scenario", "")) == "phase3_quick" else int(getattr(task, "_v2_below_8cm_frames", 0)),
            "hard_clearance_threshold_m": (
                0.0 if str(getattr(task, "_v2_scenario", "")) == "phase3_quick"
                else 0.06 if bool(getattr(task, "_phase3_visuals", False)) else 0.08
            ),
            "minimum_true_clearance": float(getattr(task, "_v2_minimum_clearance", float("inf"))),
            "minimum_clearance_entity": str(getattr(task, "_v2_minimum_clearance_entity", "")),
            "minimum_table_clearance": float(getattr(task, "_v2_minimum_table_clearance", float("inf"))),
            "minimum_table_clearance_entity": str(getattr(task, "_v2_minimum_table_clearance_entity", "")),
            "terminated": bool(getattr(task, "_terminated", False)),
            "truncated": bool(getattr(task, "_truncated", False)),
        }

    def _info_for_step(self, *, force_full: bool = False) -> dict[str, Any]:
        if self._lightweight_info and not force_full:
            return self._minimal_info()
        info = self.task.get_info()
        if not self._lightweight_info:
            info.update(self._perception_info())
        return info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}, got {action.shape}")
        tensor = torch.as_tensor(np.clip(action, -1.0, 1.0), dtype=torch.float32, device=self._backend._device).unsqueeze(0)
        self.task.pre_physics_step(tensor)
        substeps = int(self.task.control_frequency_inv)
        for index in range(substeps):
            self.task.advance_dynamic_obstacles(float(self.task._physics_dt))
            # Set the camera immediately before the render-producing step.
            # The next control frame corrects the substep-sized pose lag.
            last_substep = index == substeps - 1
            if last_substep and self._capture_cameras:
                self.task.sync_v2_camera()
            self._backend._world.step(render=last_substep and self._capture_cameras)
            self._backend.sim_frame_count += 1
        observations, rewards, _, _ = self.task.post_physics_step()
        observation = self._make_observation(observations[0].detach().cpu().numpy())
        terminated = bool(getattr(self.task, "_terminated", False))
        truncated = bool(getattr(self.task, "_truncated", False))
        info = self._info_for_step(force_full=terminated or truncated)
        return observation, float(rewards[0].detach().cpu()), bool(info["terminated"]), bool(info["truncated"]), info

    def _perception_info(self) -> dict[str, Any]:
        return {
            "estimated_tracks": self._tracker.observation().tolist(),
            "dynamic_track_missing_seconds": float(self._tracker.dynamic_missing_time),
            "accepts_new_plan": bool(self._tracker.accepts_new_plan),
            "track_missing_times": self._tracker.track_missing_times,
            "detection_pixel_counts": self._tracker.detection_pixel_counts,
            "detection_pixel_counts_by_camera": self._tracker.detection_pixel_counts_by_camera,
            "rgbd_camera_count": len(self._tracker.last_view_detections),
            "estimated_ground_heights": list(self._tracker.last_ground_heights),
        }

    def set_safety_metrics(self, **metrics: Any) -> None:
        self.task.set_safety_metrics(**metrics)

    @property
    def rgb_views(self) -> list[np.ndarray]:
        return [frame.copy() for frame in self._last_rgb_views]

    @property
    def eval_video_views(self) -> list[np.ndarray]:
        """Independent first-person and overview cameras for recording only."""
        return [
            np.asarray(frame, dtype=np.uint8).copy()
            for frame in self.task.get_eval_video_rgb_observations()
        ]

    def get_oracle_obstacle_schedule(self, control_steps: int | None = None) -> np.ndarray:
        return self.task.get_oracle_obstacle_schedule(control_steps)

    @property
    def perception_accepts_new_plan(self) -> bool:
        if (
            self._last_observation is not None
            and perception_sweep_blocks_planning(
                self.task._v2_scenario,
                self._last_observation["state"],
            )
        ):
            return False
        return self._tracker.accepts_new_plan

    @property
    def dynamic_track_missing_time(self) -> float:
        return self._tracker.dynamic_missing_time

    @property
    def track_missing_times(self) -> list[float]:
        return self._tracker.track_missing_times

    def render(self):
        if self.render_mode == "rgb_array" and self._last_observation is not None:
            return self._last_observation["rgb"]
        self._backend._world.render()
        return None

    def close(self) -> None:
        if not self._closed:
            self.task.close()
            self._backend.close()
            self._closed = True


# Compatibility for code written against the pre-release environment name.
TiagoDualSafeCarryV2Env = FlowCarryCBFEnv
