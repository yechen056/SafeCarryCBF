"""Closed-loop RGB Flow policy with an eval-only predictive CBF-QP."""

from __future__ import annotations

import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from flowcarrycbf.policies.flowcarry_cbf.control.kinematics import FullBodyKinematics
from .adaptive_cbf import (
    PredictiveWholeBodyCBFQP,
    cbf_strategy_for_template,
)
from .model import RGBFlowTrajectoryModel
from .schema import TRAIN_CAMERA_HEIGHT, TRAIN_CAMERA_WIDTH, controlled_positions


def _wrap(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


class RGBFlowCarryPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        seed: int = 0,
        prediction_steps: int = 8,
        safety_mode: str = "cbf",
    ) -> None:
        if safety_mode not in {"flow", "cbf"}:
            raise ValueError("safety_mode must be flow or cbf")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.model, payload = RGBFlowTrajectoryModel.load_checkpoint(checkpoint, self.device)
        self.normalization = payload.get("normalization", {})
        self.prediction_steps = int(prediction_steps)
        self.safety_mode = str(safety_mode)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.kinematics = FullBodyKinematics()
        self.safety_strategy = "hard"
        self.safety = PredictiveWholeBodyCBFQP(
            self.kinematics, strategy=self.safety_strategy,
        )
        self.previous_action = np.zeros(17, dtype=np.float32)
        self.metrics: dict[str, float | int | str] = {}

    def reset(self, seed: int = 0, *, template_id: str | None = None) -> None:
        strategy = cbf_strategy_for_template(template_id)
        self.safety_strategy = strategy
        self.safety = PredictiveWholeBodyCBFQP(
            self.kinematics, strategy=self.safety_strategy,
        )
        self.generator.manual_seed(int(seed))
        self.previous_action.fill(0.0)
        self.safety.reset()
        self.metrics = {}

    @staticmethod
    def _as_images(views: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        images = []
        for rgb, _ in views:
            image = np.asarray(rgb, dtype=np.uint8)
            if image.shape[:2] != (TRAIN_CAMERA_HEIGHT, TRAIN_CAMERA_WIDTH):
                image = cv2.resize(
                    image,
                    (TRAIN_CAMERA_WIDTH, TRAIN_CAMERA_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
            images.append(image)
        return np.stack(images, axis=0)

    def _targets_to_actions(self, targets: np.ndarray, state: np.ndarray) -> np.ndarray:
        current = controlled_positions(state)
        actions = np.zeros((len(targets), 17), dtype=np.float32)
        for index, target in enumerate(np.asarray(targets, dtype=np.float32)):
            delta = target - current
            c, s = math.cos(float(current[2])), math.sin(float(current[2]))
            local = np.asarray([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]], dtype=np.float32)
            actions[index, :2] = local / (self.kinematics.dt * self.kinematics.base_limits[:2])
            actions[index, 2] = _wrap(float(delta[2])) / (self.kinematics.dt * self.kinematics.base_limits[2])
            actions[index, 3:] = delta[3:] / self.kinematics.max_arm_delta
            actions[index] = np.clip(actions[index], -1.0, 1.0)
            current = target
        return actions

    def _denormalize(self, values: np.ndarray) -> np.ndarray:
        scale = np.asarray(self.normalization["actions_scale"], dtype=np.float32)
        offset = np.asarray(self.normalization["actions_offset"], dtype=np.float32)
        return (values - offset) / scale

    def act(self, observation: dict, views: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        state = np.asarray(observation["state"], dtype=np.float32).reshape(60)
        proprio = controlled_positions(state)
        proprio_scale = np.asarray(self.normalization["proprio_scale"], dtype=np.float32)
        proprio_offset = np.asarray(self.normalization["proprio_offset"], dtype=np.float32)
        images = torch.from_numpy(self._as_images(views)[None]).to(self.device)
        condition = torch.from_numpy((proprio * proprio_scale + proprio_offset)[None]).to(self.device)
        planning_started = time.perf_counter()
        with torch.no_grad():
            predicted = self.model.sample(images, condition, generator=self.generator, prediction_steps=self.prediction_steps)[0].cpu().numpy()
        nominal = self._targets_to_actions(self._denormalize(predicted), state)
        planning_latency_ms = (time.perf_counter() - planning_started) * 1000.0
        if self.safety_mode == "flow":
            action = np.asarray(nominal[0], dtype=np.float32)
            self.previous_action = action.copy()
            self.metrics = {
                "action_source": "flow",
                "cbf_strategy": self.safety_strategy,
                "cbf_status": "not_run",
                "cbf_risk": 0.0,
                "cbf_intervention": 0.0,
                "hard_minimum_clearance": float("nan"),
                "soft_minimum_clearance": float("nan"),
                "threatened_obstacle": "none",
                "threatened_robot_part": "none",
                "threat_mode": "none",
                "avoidance_side": 0.0,
                "base_lateral_escape_m": 0.0,
                "arm_task_escape_m": 0.0,
                "cbf_base_correction_rms": 0.0,
                "cbf_arm_correction_rms": 0.0,
                "action_correction_rms": 0.0,
                "grasp_consistent": 1,
                "arm_escape_within_limit": 1,
                "joint_limits_satisfied": 1,
                "self_collision_free": 1,
                "planning_latency_ms": planning_latency_ms,
                "filter_latency_ms": 0.0,
                "first_risk_step": -1,
                "first_intervention_step": -1,
                "predicted_clearance_margin": float("nan"),
                "escape_mode": "none",
                "certificate_failure_reason": "",
                "recovery_type": "none",
            }
            return np.clip(action, -1.0, 1.0)
        filtering_started = time.perf_counter()
        observed = np.asarray(observation["obstacles"], dtype=np.float32)
        shield = self.safety
        result = shield.project(
            nominal, state, observed, previous_action=self.previous_action,
        )
        filter_latency_ms = (time.perf_counter() - filtering_started) * 1000.0
        action = np.asarray(result.actions[0], dtype=np.float32)
        correction = action - np.asarray(nominal[0], dtype=np.float32)
        self.previous_action = action.copy()
        self.metrics = {
            "action_source": str(result.action_source),
            "cbf_strategy": self.safety_strategy,
            "cbf_status": str(result.status),
            "cbf_risk": float(result.risk),
            "cbf_intervention": float(result.intervention_rms),
            "hard_minimum_clearance": float(result.hard_minimum_clearance),
            "soft_minimum_clearance": float(result.soft_minimum_clearance),
            "threatened_obstacle": str(result.threatened_obstacle),
            "threatened_robot_part": str(result.threatened_robot_part),
            "threat_mode": str(getattr(shield, "last_threat_mode", "unclassified")),
            "approach_speed": float(getattr(shield, "last_approach_speed", 0.0)),
            "time_to_collision": float(
                getattr(shield, "last_time_to_collision", float("inf"))
            ),
            "avoidance_side": float(result.avoidance_side),
            "base_lateral_escape_m": float(result.base_lateral_escape_m),
            "arm_task_escape_m": float(result.arm_task_escape_m),
            "cbf_base_correction_rms": float(
                np.sqrt(np.mean(np.square(correction[:3])))
            ),
            "cbf_arm_correction_rms": float(
                np.sqrt(np.mean(np.square(correction[3:])))
            ),
            "action_correction_rms": float(
                np.sqrt(np.mean(np.square(correction)))
            ),
            "grasp_consistent": int(result.grasp_consistent),
            "arm_escape_within_limit": int(result.arm_escape_within_limit),
            "joint_limits_satisfied": int(result.joint_limits_satisfied),
            "self_collision_free": int(result.self_collision_free),
            "cbf_feasible": int(result.feasible),
            "planning_latency_ms": planning_latency_ms,
            "filter_latency_ms": filter_latency_ms,
            "first_risk_step": 0 if result.risk > 0.0 else -1,
            "first_intervention_step": 0 if result.intervention_rms > 0.0 else -1,
            "predicted_clearance_margin": float(result.hard_minimum_clearance),
            "escape_mode": str(result.action_source),
            "certificate_failure_reason": "" if result.feasible else str(result.status),
            "recovery_type": str(result.action_source) if str(result.action_source).startswith("recovery") else "none",
        }
        return np.clip(action, -1.0, 1.0)
