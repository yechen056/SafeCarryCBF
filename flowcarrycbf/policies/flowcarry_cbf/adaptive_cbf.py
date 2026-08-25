"""Receding-horizon whole-body time-varying CBF-QP for RGB evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import osqp
import scipy.sparse as sparse

from flowcarrycbf.policies.flowcarry_cbf.control.kinematics import (
    BodyCapsule,
    BodyOBB,
    BodySphere,
    FullBodyKinematics,
    Rollout,
)
from flowcarrycbf.policies.flowcarry_cbf.control.schema import (
    ACCELERATION,
    CUBE_TYPE,
    HALF_EXTENTS,
    OBSTACLE_FEATURES,
    POSITION,
    TYPE_INDEX,
    UNCERTAINTY_INDEX,
    VALID_INDEX,
    VELOCITY,
)


TABLE_CENTERS = np.asarray(
    [[1.25, 0.0, 0.0], [-4.25, 0.0, 0.0]], dtype=np.float32,
)
BASE_TABLE_COLLISION_RADIUS = 0.42


@dataclass(frozen=True)
class CBFStrategy:
    """Difficulty-specific orchestration for the shared CBF-QP kernel."""

    name: str
    soft_trigger: float
    release_clearance: float
    maximum_arm_escape: float
    maximum_base_escape: float
    release_frames: int
    zero_risk_stall_frames: int
    stall_release_horizon: int
    lock_recovery_side: bool


CBF_STRATEGIES: dict[str, CBFStrategy] = {
    "easy": CBFStrategy(
        name="easy",
        soft_trigger=0.12,
        release_clearance=0.16,
        maximum_arm_escape=0.10,
        maximum_base_escape=0.30,
        release_frames=2,
        zero_risk_stall_frames=2,
        stall_release_horizon=2,
        lock_recovery_side=True,
    ),
    "stress": CBFStrategy(
        name="stress",
        soft_trigger=0.16,
        release_clearance=0.20,
        maximum_arm_escape=0.14,
        maximum_base_escape=0.65,
        release_frames=3,
        zero_risk_stall_frames=4,
        stall_release_horizon=3,
        lock_recovery_side=True,
    ),
    # The hard strategy is the controller that predates difficulty routing.
    "hard": CBFStrategy(
        name="hard",
        soft_trigger=0.20,
        release_clearance=0.24,
        maximum_arm_escape=0.18,
        maximum_base_escape=1.25,
        release_frames=5,
        zero_risk_stall_frames=8,
        stall_release_horizon=1,
        lock_recovery_side=False,
    ),
}


def get_cbf_strategy(strategy: str | CBFStrategy) -> CBFStrategy:
    if isinstance(strategy, CBFStrategy):
        return strategy
    name = str(strategy).lower()
    if name not in CBF_STRATEGIES:
        raise ValueError(
            f"unknown CBF strategy {strategy!r}; expected one of "
            f"{', '.join(CBF_STRATEGIES)}"
        )
    return CBF_STRATEGIES[name]


def cbf_strategy_for_template(template_id: str | None) -> str:
    """Choose a CBF strategy from an evaluation task identifier."""

    value = str(template_id or "").lower()
    if value.endswith("_easy"):
        return "easy"
    if value.endswith("_stress"):
        return "stress"
    return "hard"


@dataclass(frozen=True)
class AdaptiveCBFResult:
    actions: np.ndarray
    feasible: bool
    status: str
    risk: float
    hard_minimum_clearance: float
    soft_minimum_clearance: float
    intervention_rms: float
    action_source: str
    threatened_obstacle: str
    threatened_robot_part: str
    avoidance_side: float
    base_lateral_escape_m: float
    arm_task_escape_m: float
    grasp_consistent: bool
    arm_escape_within_limit: bool
    joint_limits_satisfied: bool
    self_collision_free: bool
    latency_ms: float
    first_risk_step: int = -1
    first_intervention_step: int = -1
    predicted_clearance_margin: float = float("inf")
    escape_mode: str = "none"
    certificate_failure_reason: str = ""
    recovery_type: str = "none"


@dataclass(frozen=True)
class _ClearanceWitness:
    clearance: float
    obstacle_index: int
    obstacle_label: str
    robot_part: str


def _wrap(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


class PredictiveWholeBodyCBFQP:
    """Short-horizon CBF-QP that minimally modifies a Flow trajectory.

    Obstacles enter through the perception observation in the current base
    frame. They are transformed once into the world frame; all rollout
    geometry and time-varying obstacle predictions are then compared there.
    """

    def __init__(
        self,
        kinematics: FullBodyKinematics,
        *,
        hard_steps: int = 3,
        soft_steps: int = 24,
        soft_trigger: float | None = None,
        release_clearance: float | None = None,
        barrier_gain: float = 2.0,
        maximum_rounds: int = 2,
        maximum_arm_escape: float | None = None,
        certificate_horizon: int = 16,
        qp_horizon: int = 3,
        strategy: str | CBFStrategy = "hard",
    ) -> None:
        selected_strategy = get_cbf_strategy(strategy)
        self.kinematics = kinematics
        self.strategy = selected_strategy
        self.hard_steps = int(hard_steps)
        self.soft_steps = int(soft_steps)
        self.soft_trigger = float(
            selected_strategy.soft_trigger if soft_trigger is None else soft_trigger
        )
        self.release_clearance = float(
            selected_strategy.release_clearance
            if release_clearance is None else release_clearance
        )
        self.barrier_gain = float(barrier_gain)
        self.maximum_rounds = int(maximum_rounds)
        self.maximum_arm_escape = float(
            selected_strategy.maximum_arm_escape
            if maximum_arm_escape is None else maximum_arm_escape
        )
        self.certificate_horizon = int(certificate_horizon)
        self.qp_horizon = int(qp_horizon)
        self._warm_start: np.ndarray | None = None
        self._previous_safe_action = np.zeros(17, dtype=np.float32)
        self._avoidance_side = 0.0
        self._release_frames = 0
        self._avoidance_origin_xy: np.ndarray | None = None
        self._avoidance_axis_world: np.ndarray | None = None
        self._avoidance_hand_centers: np.ndarray | None = None
        self._avoidance_obstacle_index: int | None = None
        self._bimanual_detour_active = False
        self._source_table_retreat_active = False
        self._source_table_release_frames = 0
        self._rotation_complete = False
        self._zero_risk_grasp_stall_frames = 0

    def reset(self) -> None:
        self._warm_start = None
        self._previous_safe_action.fill(0.0)
        self._avoidance_side = 0.0
        self._release_frames = 0
        self._avoidance_origin_xy = None
        self._avoidance_axis_world = None
        self._avoidance_hand_centers = None
        self._avoidance_obstacle_index = None
        self._bimanual_detour_active = False
        self._source_table_retreat_active = False
        self._source_table_release_frames = 0
        self._rotation_complete = False
        self._zero_risk_grasp_stall_frames = 0

    def _update_zero_risk_grasp_stall(
        self,
        risk: float,
        nominal_certificate: tuple[
            bool, float, bool, bool, bool, bool, _ClearanceWitness,
        ],
        *,
        release_allowed: bool,
    ) -> bool:
        stalled = bool(
            risk <= 0.0
            and not nominal_certificate[2]
            and all(nominal_certificate[3:6])
            and release_allowed
        )
        self._zero_risk_grasp_stall_frames = (
            self._zero_risk_grasp_stall_frames + 1 if stalled else 0
        )
        return (
            self._zero_risk_grasp_stall_frames
            >= self.strategy.zero_risk_stall_frames
        )

    @staticmethod
    def _rotation(base: np.ndarray) -> np.ndarray:
        c, s = math.cos(float(base[2])), math.sin(float(base[2]))
        return np.asarray([[c, -s], [s, c]], dtype=np.float32)

    def _obstacles_to_world(
        self, obstacles: np.ndarray, base: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        values = np.asarray(obstacles, dtype=np.float32).reshape(
            -1, OBSTACLE_FEATURES,
        ).copy()
        rotation = self._rotation(base)
        valid = values[:, VALID_INDEX] > 0.5
        values[valid, :2] = values[valid, :2] @ rotation.T + base[:2]
        values[valid, 3:5] = values[valid, 3:5] @ rotation.T
        values[valid, 6:8] = values[valid, 6:8] @ rotation.T
        labels = [f"perceived_obstacle_{index}" for index in range(len(values))]
        return values, labels

    @staticmethod
    def _world_table_obstacles() -> tuple[np.ndarray, list[str]]:
        parts: list[np.ndarray] = []
        labels: list[str] = []
        for table_index, center in enumerate(TABLE_CENTERS):
            geometry = [
                (
                    np.asarray([0.0, 0.0, 0.72], dtype=np.float32),
                    np.asarray([0.40, 0.50, 0.03], dtype=np.float32),
                    "top",
                ),
            ]
            geometry.extend(
                (
                    np.asarray(
                        [side_x * 0.31, side_y * 0.41, 0.345],
                        dtype=np.float32,
                    ),
                    np.asarray([0.03, 0.03, 0.345], dtype=np.float32),
                    f"leg_{leg_index}",
                )
                for leg_index, (side_x, side_y) in enumerate(
                    ((-1, -1), (-1, 1), (1, -1), (1, 1))
                )
            )
            for offset, half_extents, part_name in geometry:
                obstacle = np.zeros(OBSTACLE_FEATURES, dtype=np.float32)
                obstacle[POSITION] = center + offset
                obstacle[HALF_EXTENTS] = half_extents
                obstacle[TYPE_INDEX] = CUBE_TYPE
                obstacle[VALID_INDEX] = 1.0
                parts.append(obstacle)
                labels.append(f"table_{table_index}_{part_name}")
        return np.asarray(parts, dtype=np.float32), labels

    def _rollout_in_world(self, state: np.ndarray, actions: np.ndarray) -> Rollout:
        local = self.kinematics.rollout(state, actions)
        base = np.asarray(state, dtype=np.float32)[:3]
        rotation = self._rotation(base)

        def point(value: np.ndarray) -> np.ndarray:
            result = np.asarray(value, dtype=np.float32).copy()
            result[:2] = rotation @ result[:2] + base[:2]
            return result

        spheres = tuple(
            tuple(BodySphere(point(item.center), item.radius, item.name) for item in step)
            for step in local.spheres
        )
        capsules = tuple(
            tuple(
                BodyCapsule(point(item.start), point(item.end), item.radius, item.name)
                for item in step
            )
            for step in local.capsules
        )
        obbs = tuple(
            tuple(
                BodyOBB(
                    point(item.center), item.half_extents.copy(),
                    _wrap(item.yaw + float(base[2])), item.name,
                )
                for item in step
            )
            for step in local.obbs
        )
        return Rollout(
            local.base, local.left_q, local.right_q,
            spheres, capsules, obbs,
        )

    def _current_world_rollout(self, state: np.ndarray) -> Rollout:
        value = np.asarray(state, dtype=np.float32).reshape(60)
        base = value[:3]
        local_spheres = self.kinematics._spheres(
            base, base, value[8:15], value[15:22],
        )
        rotation = self._rotation(base)

        def point(local: np.ndarray) -> np.ndarray:
            result = np.asarray(local, dtype=np.float32).copy()
            result[:2] = rotation @ result[:2] + base[:2]
            return result

        spheres = tuple(
            BodySphere(point(item.center), item.radius, item.name)
            for item in local_spheres
        )
        capsules = self.kinematics._capsules(spheres)
        payload = BodyOBB(
            point(local_spheres[-1].center),
            self.kinematics.payload_half_extents,
            float(base[2]),
            "payload_obb",
        )
        return Rollout(
            base[None].copy(), value[8:15][None].copy(), value[15:22][None].copy(),
            (spheres,), (capsules,), ((payload,),),
        )

    def _configuration_rollout(
        self, state: np.ndarray, actions: np.ndarray,
    ) -> np.ndarray:
        """Roll out the 17-D robot configuration without collision geometry."""

        value = np.asarray(state, dtype=np.float32).reshape(60)
        sequence = np.asarray(actions, dtype=np.float32).reshape(-1, 17)
        base = value[:3].copy()
        left = value[8:15].copy()
        right = value[15:22].copy()
        yaw = float(base[2])
        c, s = math.cos(yaw), math.sin(yaw)
        world_velocity = value[3:6]
        command = np.asarray(
            [
                c * world_velocity[0] + s * world_velocity[1],
                -s * world_velocity[0] + c * world_velocity[1],
                world_velocity[2],
            ],
            dtype=np.float32,
        )
        configurations: list[np.ndarray] = []
        for action in sequence:
            desired = action[:3] * self.kinematics.base_limits
            command += np.clip(
                desired - command,
                -self.kinematics.base_acceleration_step,
                self.kinematics.base_acceleration_step,
            )
            yaw = float(base[2])
            c, s = math.cos(yaw), math.sin(yaw)
            base[0] += (
                c * command[0] - s * command[1]
            ) * self.kinematics.dt
            base[1] += (
                s * command[0] + c * command[1]
            ) * self.kinematics.dt
            base[2] = _wrap(
                float(base[2]) + float(command[2]) * self.kinematics.dt
            )
            left += action[3:10] * self.kinematics.max_arm_delta
            right += action[10:17] * self.kinematics.max_arm_delta
            configurations.append(np.concatenate((base, left, right)))
        return np.asarray(configurations, dtype=np.float32)

    def _world_geometry_from_configuration(
        self, configuration: np.ndarray,
    ) -> Rollout:
        value = np.asarray(configuration, dtype=np.float32).reshape(17)
        base, left, right = value[:3], value[3:10], value[10:17]
        # A zero reference frame makes FullBodyKinematics return world centers.
        spheres = self.kinematics._spheres(
            base, np.zeros(3, dtype=np.float32), left, right,
        )
        capsules = self.kinematics._capsules(spheres)
        payload = BodyOBB(
            spheres[-1].center,
            self.kinematics.payload_half_extents,
            float(base[2]),
            "payload_obb",
        )
        return Rollout(
            base[None].copy(), left[None].copy(), right[None].copy(),
            (spheres,), (capsules,), ((payload,),),
        )

    def _clearance_action_gradients(
        self,
        state: np.ndarray,
        center: np.ndarray,
        obstacles: np.ndarray,
        labels: list[str],
        active_keys: list[tuple[int, int]],
        base_map: dict[tuple[int, int], float],
    ) -> np.ndarray:
        """Chain distance/configuration and configuration/action Jacobians."""

        steps = len(center)
        dimension = steps * 17
        epsilon = 1.0e-3
        center_flat = center.reshape(-1).astype(np.float64)
        configurations = self._configuration_rollout(state, center)
        configuration_jacobians = np.zeros(
            (steps, 17, dimension), dtype=np.float64,
        )
        for variable in range(dimension):
            perturbed = center_flat.copy()
            perturbed[variable] += epsilon
            changed = self._configuration_rollout(
                state, perturbed.reshape(steps, 17),
            )
            difference = changed.astype(np.float64) - configurations
            difference[:, 2] = np.asarray(
                [_wrap(value) for value in difference[:, 2]],
                dtype=np.float64,
            )
            configuration_jacobians[:, :, variable] = difference / epsilon

        by_step: dict[int, list[int]] = {}
        for step, obstacle_index in active_keys:
            by_step.setdefault(step, []).append(obstacle_index)
        row_by_key = {key: index for index, key in enumerate(active_keys)}
        gradients = np.zeros((len(active_keys), dimension), dtype=np.float64)
        for step, obstacle_indices in by_step.items():
            configuration_gradient = np.zeros(
                (len(obstacle_indices), 17), dtype=np.float64,
            )
            for component in range(17):
                changed = configurations[step].copy()
                changed[component] += epsilon
                if component == 2:
                    changed[component] = _wrap(changed[component])
                geometry = self._world_geometry_from_configuration(changed)
                future_time = (step + 1) * self.kinematics.dt
                for local_index, obstacle_index in enumerate(obstacle_indices):
                    predicted = self._predict_obstacle(
                        obstacles[obstacle_index], future_time,
                    )
                    witness = self._step_witness(
                        geometry,
                        0,
                        predicted,
                        obstacle_index,
                        labels[obstacle_index],
                    )
                    key = (step, obstacle_index)
                    configuration_gradient[local_index, component] = (
                        witness.clearance - base_map[key]
                    ) / epsilon
            for local_index, obstacle_index in enumerate(obstacle_indices):
                key = (step, obstacle_index)
                gradients[row_by_key[key]] = (
                    configuration_gradient[local_index]
                    @ configuration_jacobians[step]
                )
        return gradients

    def _predict_obstacle(self, obstacle: np.ndarray, future_time: float) -> np.ndarray:
        """Predict a conservative reachable obstacle envelope.

        The tracker uncertainty is a one-sigma positional estimate.  We
        propagate it with velocity/acceleration uncertainty and inflate the
        obstacle geometry, so the witness is certified against a reachable
        set rather than a single point estimate.
        """
        result = np.asarray(obstacle, dtype=np.float32).copy()
        t = float(max(0.0, future_time))
        result[POSITION] = (
            obstacle[POSITION]
            + obstacle[VELOCITY] * t
            + 0.5 * obstacle[ACCELERATION] * t**2
        )
        uncertainty = max(0.0, float(obstacle[UNCERTAINTY_INDEX]))
        # Bound propagation grows with time and is capped to avoid numerical
        # domination when a detector briefly reports a stale track.
        dynamic = float(np.linalg.norm(obstacle[VELOCITY]) + np.linalg.norm(obstacle[ACCELERATION])) > 1.0e-5
        growth = min(0.35, uncertainty + (0.015 * t + 0.004 * t * t if dynamic else 0.0))
        result[HALF_EXTENTS] = np.maximum(
            result[HALF_EXTENTS] + growth,
            np.full(3, 0.02, dtype=np.float32),
        )
        return result

    def _step_witness(
        self,
        rollout: Rollout,
        step: int,
        obstacle: np.ndarray,
        obstacle_index: int,
        obstacle_label: str,
    ) -> _ClearanceWitness:
        # Evaluate every sphere and every sampled capsule point in two NumPy
        # batches.  This is algebraically identical to the former Python
        # loops (including the conservative half-spacing correction), but a
        # risky QP frame otherwise makes tens of thousands of tiny norm calls.
        spheres = [
            sphere for sphere in rollout.spheres[step]
            if sphere.name != "payload"
        ]
        sphere_centers = np.asarray(
            [sphere.center for sphere in spheres], dtype=np.float64,
        )
        sphere_radii = np.asarray(
            [sphere.radius for sphere in spheres], dtype=np.float64,
        )
        obstacle_center = np.asarray(obstacle[POSITION], dtype=np.float64)
        if obstacle[TYPE_INDEX] < 0.5:
            sphere_values = (
                np.linalg.norm(sphere_centers - obstacle_center, axis=1)
                - sphere_radii - float(obstacle[HALF_EXTENTS][0])
            )
        else:
            half_extents = np.asarray(obstacle[HALF_EXTENTS], dtype=np.float64)
            sphere_delta = np.abs(sphere_centers - obstacle_center) - half_extents
            sphere_values = (
                np.linalg.norm(np.maximum(sphere_delta, 0.0), axis=1)
                + np.minimum(np.max(sphere_delta, axis=1), 0.0)
                - sphere_radii
            )
            if obstacle_label.endswith("_top"):
                base_index = next(
                    (index for index, sphere in enumerate(spheres) if sphere.name == "base"),
                    None,
                )
                if base_index is not None:
                    # Match the task's exact 2-D tabletop keep-out for base.
                    delta = (
                        np.abs(sphere_centers[base_index, :2] - obstacle_center[:2])
                        - half_extents[:2]
                        - BASE_TABLE_COLLISION_RADIUS
                    )
                    sphere_values[base_index] = (
                        np.linalg.norm(np.maximum(delta, 0.0))
                        + min(float(np.max(delta)), 0.0)
                    )

        capsules = rollout.capsules[step]
        starts = np.asarray(
            [capsule.start for capsule in capsules], dtype=np.float64,
        )
        ends = np.asarray(
            [capsule.end for capsule in capsules], dtype=np.float64,
        )
        capsule_radii = np.asarray(
            [capsule.radius for capsule in capsules], dtype=np.float64,
        )
        fractions = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        capsule_points = (
            starts[:, None, :]
            + fractions[None, :, None] * (ends - starts)[:, None, :]
        )
        if obstacle[TYPE_INDEX] < 0.5:
            capsule_samples = (
                np.linalg.norm(
                    capsule_points - obstacle_center[None, None, :], axis=2,
                )
                - capsule_radii[:, None]
                - float(obstacle[HALF_EXTENTS][0])
            )
        else:
            capsule_delta = (
                np.abs(capsule_points - obstacle_center[None, None, :])
                - half_extents[None, None, :]
            )
            capsule_samples = (
                np.linalg.norm(np.maximum(capsule_delta, 0.0), axis=2)
                + np.minimum(np.max(capsule_delta, axis=2), 0.0)
                - capsule_radii[:, None]
            )
        capsule_values = (
            np.min(capsule_samples, axis=1)
            - 0.125 * np.linalg.norm(ends - starts, axis=1)
        )
        obb_values = np.asarray([
            self.kinematics.obb_obstacle_clearance(obb, obstacle)
            for obb in rollout.obbs[step]
        ], dtype=np.float64)
        values = np.concatenate((sphere_values, capsule_values, obb_values))
        names = (
            [sphere.name for sphere in spheres]
            + [capsule.name for capsule in capsules]
            + [obb.name for obb in rollout.obbs[step]]
        )
        witness_index = int(np.argmin(values))
        clearance, part = float(values[witness_index]), names[witness_index]
        return _ClearanceWitness(
            float(clearance), obstacle_index, obstacle_label, part,
        )

    def _clearance_map(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        obstacles: np.ndarray,
        labels: list[str],
        steps: int,
    ) -> tuple[dict[tuple[int, int], float], _ClearanceWitness]:
        rollout = self._rollout_in_world(state, actions[:steps])
        result: dict[tuple[int, int], float] = {}
        witnesses: list[_ClearanceWitness] = []
        for step in range(min(steps, len(rollout.spheres))):
            future_time = (step + 1) * self.kinematics.dt
            for obstacle_index, obstacle in enumerate(obstacles):
                if obstacle[VALID_INDEX] < 0.5:
                    continue
                predicted = self._predict_obstacle(obstacle, future_time)
                witness = self._step_witness(
                    rollout, step, predicted, obstacle_index, labels[obstacle_index],
                )
                result[(step, obstacle_index)] = witness.clearance
                witnesses.append(witness)
        default = _ClearanceWitness(float("inf"), -1, "none", "none")
        return result, min(witnesses, key=lambda item: item.clearance, default=default)

    def _current_clearances(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
        labels: list[str],
    ) -> dict[int, float]:
        rollout = self._current_world_rollout(state)
        result: dict[int, float] = {}
        for obstacle_index, obstacle in enumerate(obstacles):
            if obstacle[VALID_INDEX] < 0.5:
                continue
            result[obstacle_index] = self._step_witness(
                rollout, 0, obstacle, obstacle_index, labels[obstacle_index],
            ).clearance
        return result

    def _barrier_clearance_satisfied(
        self,
        clearance_map: dict[tuple[int, int], float],
        current: dict[int, float],
        steps: int,
    ) -> bool:
        """Allow only barrier-certified recovery from an estimated overlap."""

        decay = max(0.0, 1.0 - self.barrier_gain * self.kinematics.dt)
        for obstacle_index, previous in current.items():
            for step in range(int(steps)):
                value = clearance_map.get((step, obstacle_index), float("inf"))
                required = (
                    -1.0e-3
                    if previous >= -1.0e-3
                    else decay * previous - 1.0e-3
                )
                if value < required:
                    return False
                previous = value
        return True

    def _update_source_table_retreat(
        self,
        clearance_ok: bool,
        current: dict[int, float],
    ) -> bool:
        """Latch source-table departure until the base is genuinely clear."""

        if not clearance_ok:
            self._source_table_retreat_active = True
            self._source_table_release_frames = 0
        elif self._source_table_retreat_active:
            minimum = min(current.values(), default=float("inf"))
            if minimum > self.release_clearance:
                self._source_table_release_frames += 1
                if self._source_table_release_frames >= self.strategy.release_frames:
                    self._source_table_retreat_active = False
                    self._source_table_release_frames = 0
            else:
                self._source_table_release_frames = 0
        return bool(self._source_table_retreat_active)

    def _certificate(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        obstacles: np.ndarray,
        labels: list[str],
    ) -> tuple[bool, float, bool, bool, bool, bool, _ClearanceWitness]:
        sequence = np.asarray(actions, dtype=np.float32)[
            : min(len(actions), self.certificate_horizon)
        ]
        clearance_map, witness = self._clearance_map(
            state, sequence, obstacles, labels, len(sequence),
        )
        minimum = min(clearance_map.values(), default=float("inf"))
        local_rollout = self.kinematics.rollout(state, sequence)
        grasp, arm_escape = self._grasp_and_escape_consistent(
            state, local_rollout,
        )
        joints = self.kinematics.joint_limits_satisfied(local_rollout)
        self_collision = self.kinematics.self_collision_free(local_rollout)
        current = self._current_clearances(state, obstacles, labels)
        clearance_ok = self._barrier_clearance_satisfied(
            clearance_map, current, len(sequence),
        )
        feasible = bool(
            clearance_ok and grasp and arm_escape and joints and self_collision
        )
        return (
            feasible,
            float(minimum),
            bool(grasp),
            bool(arm_escape),
            bool(joints),
            bool(self_collision),
            witness,
        )

    def _soft_risk(
        self,
        state: np.ndarray,
        nominal: np.ndarray,
        obstacles: np.ndarray,
        labels: list[str],
    ) -> tuple[float, float, _ClearanceWitness]:
        steps = min(self.soft_steps, len(nominal))
        clearance_map, witness = self._clearance_map(
            state, nominal, obstacles, labels, steps,
        )
        minimum = min(clearance_map.values(), default=float("inf"))
        uncertainty = 0.0
        if witness.obstacle_index >= 0:
            uncertainty = min(
                float(obstacles[witness.obstacle_index, UNCERTAINTY_INDEX]),
                0.05,
            )
        effective = minimum - uncertainty
        raw = np.clip(
            (self.soft_trigger - effective) / max(self.soft_trigger, 1.0e-6),
            0.0,
            1.0,
        )
        risk = float(raw * raw * (3.0 - 2.0 * raw))
        return risk, float(minimum), witness

    def _update_avoidance_side(
        self,
        risk: float,
        soft_clearance: float,
        witness: _ClearanceWitness,
        perceived_world: np.ndarray,
        state: np.ndarray,
    ) -> float:
        dynamic_obstacle = (
            witness.obstacle_index >= 0
            and witness.obstacle_index < len(perceived_world)
        )
        if risk > 0.0 and dynamic_obstacle:
            # A second moving obstacle needs a fresh escape origin and
            # lateral budget. Reusing the first obstacle's origin can leave
            # the filter with no remaining motion even though a new safe
            # homotopy exists.
            if self._avoidance_obstacle_index != witness.obstacle_index:
                self._avoidance_side = 0.0
                self._avoidance_origin_xy = None
                self._avoidance_axis_world = None
                self._avoidance_hand_centers = None
                self._bimanual_detour_active = False
                self._avoidance_obstacle_index = witness.obstacle_index
            if self._avoidance_side == 0.0:
                obstacle = perceived_world[witness.obstacle_index]
                rotation = self._rotation(np.asarray(state)[:3])
                local = rotation.T @ (obstacle[:2] - np.asarray(state)[:2])
                self._avoidance_side = -1.0 if local[1] >= 0.0 else 1.0
                self._avoidance_origin_xy = np.asarray(state, dtype=np.float32)[:2].copy()
                self._avoidance_axis_world = (
                    rotation[:, 1] * self._avoidance_side
                ).astype(np.float32)
                self._avoidance_hand_centers = self._hand_centers(state)
            self._release_frames = 0
        elif soft_clearance > self.release_clearance:
            self._release_frames += 1
            if self._release_frames >= self.strategy.release_frames:
                self._avoidance_side = 0.0
                self._release_frames = 0
                self._avoidance_origin_xy = None
                self._avoidance_axis_world = None
                self._avoidance_hand_centers = None
                self._avoidance_obstacle_index = None
                self._bimanual_detour_active = False
        else:
            self._release_frames = 0
        return float(self._avoidance_side)

    def _hand_centers(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=np.float32).reshape(60)
        spheres = self.kinematics._spheres(
            value[:3], value[:3], value[8:15], value[15:22],
        )
        return np.stack((spheres[-3].center, spheres[-2].center), axis=0)

    def _escape_progress(self, state: np.ndarray) -> tuple[float, float]:
        base_offset = 0.0
        if (
            self._avoidance_origin_xy is not None
            and self._avoidance_axis_world is not None
        ):
            delta = np.asarray(state, dtype=np.float32)[:2] - self._avoidance_origin_xy
            base_offset = max(0.0, float(delta @ self._avoidance_axis_world))
        arm_offset = 0.0
        if self._avoidance_hand_centers is not None:
            displacement = self._hand_centers(state) - self._avoidance_hand_centers
            arm_offset = float(np.max(np.linalg.norm(displacement, axis=1)))
        return base_offset, arm_offset

    def _grasp_and_escape_consistent(
        self,
        state: np.ndarray,
        rollout: Rollout,
    ) -> tuple[bool, bool]:
        """Certify the rigid bimanual relation and cumulative arm excursion."""

        reference = self.kinematics.grasp_vector(state)
        grasp_ok = True
        escape_ok = True
        for base, left, right in zip(
            rollout.base, rollout.left_q, rollout.right_q,
        ):
            spheres = self.kinematics._spheres(base, base, left, right)
            handle_vector = spheres[-2].center - spheres[-3].center
            if float(np.linalg.norm(handle_vector - reference)) > 0.012:
                grasp_ok = False
            if self._avoidance_hand_centers is not None:
                hands = np.stack(
                    (spheres[-3].center, spheres[-2].center), axis=0,
                )
                excursion = np.linalg.norm(
                    hands - self._avoidance_hand_centers, axis=1,
                )
                if float(np.max(excursion)) > self.maximum_arm_escape + 1.0e-3:
                    escape_ok = False
        return grasp_ok, escape_ok

    def _bimanual_escape_action(
        self, state: np.ndarray, side: float, risk: float,
        *, elevate: bool = False,
    ) -> np.ndarray:
        if side == 0.0 or risk <= 0.0:
            return np.zeros(14, dtype=np.float32)
        value = np.asarray(state, dtype=np.float32).reshape(60)
        base, left, right = value[:3], value[8:15], value[15:22]
        nominal = self.kinematics._spheres(base, base, left, right)
        nominal_hands = np.concatenate((nominal[-3].center, nominal[-2].center))
        jacobian = np.zeros((6, 14), dtype=np.float64)
        epsilon = 1.0e-3
        for component in range(14):
            perturbed_left, perturbed_right = left.copy(), right.copy()
            if component < 7:
                perturbed_left[component] += epsilon
            else:
                perturbed_right[component - 7] += epsilon
            geometry = self.kinematics._spheres(
                base, base, perturbed_left, perturbed_right,
            )
            hands = np.concatenate((geometry[-3].center, geometry[-2].center))
            jacobian[:, component] = (hands - nominal_hands) / epsilon
        # Track a geometry-bounded task-space detour instead of spreading the
        # remaining excursion uniformly over every future certificate step.
        # The old formulation produced only 2--3 cm of real hand motion
        # because the soft arm objective lost against the cheaper base
        # correction.  Feedback to a fixed target makes the modulation
        # persistent across replans while the certificate still bounds it.
        current_hands = nominal_hands.reshape(2, 3)
        origins = (
            current_hands
            if self._avoidance_hand_centers is None
            else np.asarray(self._avoidance_hand_centers, dtype=np.float64)
        )
        target = origins.copy()
        detour_distance = max(0.08, self.maximum_arm_escape - 0.02)
        target[:, 1] += float(side) * detour_distance
        if elevate:
            target[:, 2] += 0.06
        error = target - current_hands
        error_norm = np.linalg.norm(error, axis=1)
        for hand_index, norm in enumerate(error_norm):
            if norm > 0.010:
                error[hand_index] *= 0.010 / norm
        displacement = error.reshape(6)
        damping = 2.5e-3
        joint_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(6),
            displacement,
        )
        action = joint_delta / max(float(self.kinematics.max_arm_delta), 1.0e-6)
        return np.clip(
            action,
            -float(self.kinematics.arm_action_limit),
            float(self.kinematics.arm_action_limit),
        ).astype(np.float32)

    def _escape_reference(
        self,
        nominal: np.ndarray,
        state: np.ndarray,
        risk: float,
        side: float,
        *,
        use_base: bool = True,
        use_arms: bool = True,
        brake: bool = True,
        elevate_arms: bool = False,
    ) -> np.ndarray:
        result = np.asarray(nominal, dtype=np.float32).copy()
        if brake:
            result[:, 0] *= max(0.0, 1.0 - 0.85 * float(risk))
        if use_base and side != 0.0:
            base_offset, _ = self._escape_progress(state)
            # Respect the strategy-specific lateral escape corridor so a
            # second obstacle can be handled without unbounded base drift.
            remaining = max(
                0.0, self.strategy.maximum_base_escape - base_offset,
            )
            maximum_addition = remaining / max(
                max(1, len(result)) * self.kinematics.dt
                * float(self.kinematics.base_limits[1]),
                1.0e-6,
            )
            rotation = self._rotation(np.asarray(state, dtype=np.float32)[:3])
            world_axis = self._avoidance_axis_world
            if world_axis is None:
                world_axis = rotation[:, 1] * float(side)
            local_axis = rotation.T @ world_axis
            addition = local_axis * min(float(risk), maximum_addition)
            result[:, :2] = np.clip(result[:, :2] + addition[None], -1.0, 1.0)
        if not use_arms:
            result[:, 3:] = 0.0
        if use_arms:
            # Recompute one feedback command on every real control frame,
            # then taper it inside the certificate rollout at the estimated
            # number of steps required to reach the target.  This retains
            # closed-loop convergence without rebuilding fourteen finite-
            # difference Jacobians at every predicted step.
            arm = self._bimanual_escape_action(
                state, side, risk, elevate=elevate_arms,
            )
            current_hands = self._hand_centers(state)
            origins = (
                current_hands
                if self._avoidance_hand_centers is None
                else self._avoidance_hand_centers
            )
            target = np.asarray(origins, dtype=np.float32).copy()
            detour_distance = max(0.08, self.maximum_arm_escape - 0.02)
            target[:, 1] += float(side) * detour_distance
            if elevate_arms:
                target[:, 2] += 0.06
            remaining = float(np.max(np.linalg.norm(target - current_hands, axis=1)))
            active_steps = min(
                len(result), max(1, int(math.ceil(remaining / 0.009))),
            )
            for step in range(active_steps):
                # The task-space detour owns the arm command while active.
                # Adding it to Flow lets an opposing nominal arm velocity
                # cancel the feedback and caused real excursions to stall at
                # 1--3 cm even though the isolated rollout reached 10 cm.
                # Base commands remain untouched so Flow keeps advancing.
                result[step, 3:] = arm
        return result

    def _rate_limit_arm_actions(
        self,
        actions: np.ndarray,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        """Apply the same arm slew limit used by the QP to fallback actions."""

        result = np.asarray(actions, dtype=np.float32).copy()
        previous_arm = np.asarray(previous_action, dtype=np.float32).reshape(17)[3:].copy()
        maximum_step = float(self.kinematics.arm_acceleration_step)
        for step in range(len(result)):
            result[step, 3:] = np.clip(
                result[step, 3:],
                previous_arm - maximum_step,
                previous_arm + maximum_step,
            )
            previous_arm = result[step, 3:].copy()
        return result

    def _forward_progress(
        self,
        nominal: np.ndarray,
        candidate: np.ndarray,
    ) -> float:
        """Return candidate effort along Flow's intended planar direction."""

        nominal_effort = np.sum(np.asarray(nominal)[:, :2], axis=0)
        norm = float(np.linalg.norm(nominal_effort))
        if norm <= 1.0e-6:
            return 0.0
        direction = nominal_effort / norm
        candidate_effort = np.sum(np.asarray(candidate)[:, :2], axis=0)
        return float(candidate_effort @ direction)

    def _variable_bounds(
        self,
        state: np.ndarray,
        previous_action: np.ndarray,
        steps: int,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float], list[float]]:
        dimension = steps * 17
        lower = np.full(dimension, -1.0, dtype=np.float64)
        upper = np.full(dimension, 1.0, dtype=np.float64)
        for step in range(steps):
            lower[step * 17 + 3 : (step + 1) * 17] = -self.kinematics.arm_action_limit
            upper[step * 17 + 3 : (step + 1) * 17] = self.kinematics.arm_action_limit
        rows: list[np.ndarray] = []
        row_lower: list[float] = []
        row_upper: list[float] = []
        previous = np.asarray(previous_action, dtype=np.float64).reshape(17)
        for step in range(steps):
            for arm_component in range(14):
                index = step * 17 + 3 + arm_component
                row = np.zeros(dimension, dtype=np.float64)
                row[index] = 1.0
                if step == 0:
                    reference = previous[3 + arm_component]
                else:
                    row[(step - 1) * 17 + 3 + arm_component] = -1.0
                    reference = 0.0
                rows.append(row)
                row_lower.append(float(reference - self.kinematics.arm_acceleration_step))
                row_upper.append(float(reference + self.kinematics.arm_acceleration_step))
        q_values = np.concatenate((np.asarray(state)[8:15], np.asarray(state)[15:22]))
        q_lower = np.concatenate((self.kinematics.arm_lower, self.kinematics.arm_lower))
        q_upper = np.concatenate((self.kinematics.arm_upper, self.kinematics.arm_upper))
        for step in range(steps):
            for arm_component in range(14):
                row = np.zeros(dimension, dtype=np.float64)
                for previous_step in range(step + 1):
                    row[previous_step * 17 + 3 + arm_component] = self.kinematics.max_arm_delta
                rows.append(row)
                row_lower.append(float(q_lower[arm_component] - q_values[arm_component]))
                row_upper.append(float(q_upper[arm_component] - q_values[arm_component]))
        return lower, upper, rows, row_lower, row_upper

    def _solve_round(
        self,
        state: np.ndarray,
        center: np.ndarray,
        target: np.ndarray,
        obstacles: np.ndarray,
        labels: list[str],
        previous_action: np.ndarray,
        active_obstacles: set[int],
    ) -> tuple[np.ndarray | None, str]:
        steps = len(center)
        dimension = steps * 17
        center_flat = center.reshape(-1).astype(np.float64)
        base_map, _ = self._clearance_map(
            state, center, obstacles, labels, steps,
        )
        active_keys = [
            key for key in sorted(base_map)
            if key[1] in active_obstacles
        ]
        if not active_keys:
            return center.copy(), "no_active_constraints"
        gradients = self._clearance_action_gradients(
            state, center, obstacles, labels, active_keys, base_map,
        )
        rows: list[np.ndarray] = []
        row_lower: list[float] = []
        row_upper: list[float] = []
        current = self._current_clearances(state, obstacles, labels)
        key_to_row = {key: index for index, key in enumerate(active_keys)}
        decay = max(0.0, 1.0 - self.barrier_gain * self.kinematics.dt)
        for key in active_keys:
            step, obstacle_index = key
            gradient = gradients[key_to_row[key]]
            value = base_map[key]
            current_value = current.get(obstacle_index, float("inf"))
            hard_floor = min(0.0, current_value) if np.isfinite(current_value) else 0.0
            rows.append(gradient)
            row_lower.append(float(hard_floor - value + gradient @ center_flat))
            row_upper.append(float("inf"))
            if step == 0:
                previous_value = current.get(obstacle_index, float("inf"))
                rate_gradient = gradient
            else:
                previous_key = (step - 1, obstacle_index)
                if previous_key not in key_to_row:
                    continue
                previous_value = base_map[previous_key]
                rate_gradient = gradient - decay * gradients[key_to_row[previous_key]]
            # For a positive-clearance trajectory, the per-step geometric
            # floor is the active safety condition. Enforcing a monotonic
            # barrier-rate bound there can make a valid lateral escape
            # infeasible merely because a moving obstacle approaches. Keep
            # the rate constraint for estimated overlap recovery, where it
            # is needed to certify exit from penetration.
            if np.isfinite(previous_value) and previous_value <= 0.0:
                rate_value = value - decay * previous_value
                rows.append(rate_gradient)
                row_lower.append(float(-rate_value + rate_gradient @ center_flat))
                row_upper.append(float("inf"))
        lower, upper, limit_rows, limit_lower, limit_upper = self._variable_bounds(
            state, previous_action, steps,
        )
        rows.extend(limit_rows)
        row_lower.extend(limit_lower)
        row_upper.extend(limit_upper)
        identity = sparse.eye(dimension, format="csc")
        constraint = sparse.vstack(
            [sparse.csc_matrix(np.asarray(rows)), identity], format="csc",
        )
        constraint_lower = np.concatenate((np.asarray(row_lower), lower))
        constraint_upper = np.concatenate((np.asarray(row_upper), upper))
        weights = np.tile(
            np.asarray([1.5, 0.8, 1.5] + [3.0] * 14, dtype=np.float64),
            steps,
        )
        hessian = sparse.diags(weights, format="csc")
        linear = -weights * target.reshape(-1)
        smooth_weight = 0.80
        difference = np.zeros((dimension, dimension), dtype=np.float64)
        difference_target = np.zeros(dimension, dtype=np.float64)
        for step in range(steps):
            for component in range(17):
                row = step * 17 + component
                difference[row, row] = 1.0
                if step == 0:
                    difference_target[row] = previous_action[component]
                else:
                    difference[row, (step - 1) * 17 + component] = -1.0
        hessian = hessian + smooth_weight * sparse.csc_matrix(
            difference.T @ difference,
        )
        linear += -smooth_weight * difference.T @ difference_target
        problem = osqp.OSQP()
        problem.setup(
            P=sparse.triu(2.0 * hessian, format="csc"),
            q=2.0 * linear,
            A=constraint,
            l=constraint_lower,
            u=constraint_upper,
            verbose=False,
            warm_start=True,
            polish=True,
            eps_abs=1.0e-4,
            eps_rel=1.0e-4,
            max_iter=500,
        )
        if self._warm_start is not None and len(self._warm_start) == dimension:
            problem.warm_start(x=self._warm_start)
        solved = problem.solve()
        if solved.x is None or solved.info.status_val not in (1, 2):
            return None, f"qp_{str(solved.info.status).lower().replace(' ', '_')}"
        self._warm_start = np.asarray(solved.x, dtype=np.float64).copy()
        return np.asarray(solved.x, dtype=np.float32).reshape(steps, 17), "qp_solved"

    def _recovery(
        self,
        state: np.ndarray,
        nominal: np.ndarray,
        obstacles: np.ndarray,
        labels: list[str],
        risk: float,
        side: float,
        previous_action: np.ndarray,
        *,
        allow_arm_recovery: bool = True,
        dynamic_obstacle_index: int | None = None,
    ) -> tuple[np.ndarray, str, tuple[bool, float, bool, bool, bool, bool, _ClearanceWitness]]:
        steps = min(self.certificate_horizon, len(nominal))
        base = np.asarray(nominal[:steps], dtype=np.float32)
        candidates: list[tuple[str, np.ndarray]] = []
        # A previously certified near-zero action is safe for one horizon,
        # but repeatedly selecting it can freeze the robot until an
        # approaching obstacle finally enters that horizon.  For a witnessed
        # dynamic arm threat, add a geometry-derived lateral escape that
        # retains Flow's forward component.  Projecting the away vector onto
        # the non-reversing half-plane prevents a sphere in front of the robot
        # from turning this recovery into prolonged backing up.
        if (
            dynamic_obstacle_index is not None
            and 0 <= dynamic_obstacle_index < len(obstacles)
        ):
            obstacle = obstacles[dynamic_obstacle_index]
            away_world = np.asarray(state[:2], dtype=np.float64) - np.asarray(
                obstacle[POSITION][:2], dtype=np.float64,
            )
            away_norm = float(np.linalg.norm(away_world))
            if away_norm > 1.0e-6:
                rotation = self._rotation(np.asarray(state, dtype=np.float32)[:3])
                away_unit = away_world / away_norm
                forward_world = rotation[:, 0]
                reverse_component = min(
                    0.0, float(np.dot(away_unit, forward_world)),
                )
                non_reversing_world = (
                    away_unit - reverse_component * forward_world
                )
                escape_norm = float(np.linalg.norm(non_reversing_world))
                if escape_norm <= 1.0e-6:
                    world_axis = self._avoidance_axis_world
                    if world_axis is None:
                        world_axis = rotation[:, 1] * (side if side else 1.0)
                    non_reversing_world = np.asarray(
                        world_axis, dtype=np.float64,
                    )
                    escape_norm = float(np.linalg.norm(non_reversing_world))
                escape_local = rotation.T @ (
                    non_reversing_world / max(escape_norm, 1.0e-6)
                )
                # Numerical roundoff must not introduce a reverse component.
                escape_local[0] = max(0.0, float(escape_local[0]))
                lateral = base.copy()
                lateral[:, :2] = np.clip(
                    lateral[:, :2] + escape_local[None], -1.0, 1.0,
                )
                lateral[:, 3:] = 0.0
                candidates.append((
                    "certified_dynamic_lateral_escape",
                    lateral,
                ))
        sides = [float(side)] if side else []
        if not (self.strategy.lock_recovery_side and sides):
            sides.extend(value for value in (-1.0, 1.0) if value not in sides)
        for candidate_side in sides:
            candidates.append((
                f"recovery_base_side_{'left' if candidate_side < 0 else 'right'}",
                self._escape_reference(
                    base, state, max(risk, 0.5), candidate_side,
                    use_base=True, use_arms=False, brake=True,
                ),
            ))
            if allow_arm_recovery:
                for mode, use_base, elevate in (
                    ("joint", True, False),
                    ("arms_over", False, True),
                    ("joint_over", True, True),
                    ("arms", False, False),
                ):
                    candidates.append((
                        f"recovery_{mode}_side_{'left' if candidate_side < 0 else 'right'}",
                        self._escape_reference(
                            base, state, max(risk, 0.5), candidate_side,
                            use_base=use_base, use_arms=True, brake=True,
                            elevate_arms=elevate,
                        ),
                    ))
        braked = base.copy()
        braked[:, :2] = 0.0
        if not allow_arm_recovery:
            braked[:, 3:] = 0.0
        candidates.append(("recovery_brake", braked))
        previous_safe = np.repeat(self._previous_safe_action[None], steps, axis=0)
        if not allow_arm_recovery:
            previous_safe[:, 3:] = 0.0
        candidates.append((
            "previous_safe_action",
            previous_safe,
        ))
        candidates.append(("safety_hold", np.zeros((steps, 17), dtype=np.float32)))
        candidates = [
            (source, self._rate_limit_arm_actions(candidate, previous_action))
            for source, candidate in candidates
        ]
        last_certificate = self._certificate(
            state, candidates[-1][1], obstacles, labels,
        )
        ranked = sorted(
            candidates,
            key=lambda item: (
                item[0] == "certified_dynamic_lateral_escape",
                self._forward_progress(base, item[1]),
                -float(np.mean(np.square(item[1] - base))),
            ),
            reverse=True,
        )
        for source, candidate in ranked:
            certificate = self._certificate(state, candidate, obstacles, labels)
            if certificate[0]:
                return candidate, source, certificate
            last_certificate = certificate
        return candidates[-1][1], "safety_hold_uncertified", last_certificate

    def project(
        self,
        nominal: np.ndarray,
        state: np.ndarray,
        perceived_obstacles: np.ndarray,
        *,
        previous_action: np.ndarray | None = None,
    ) -> AdaptiveCBFResult:
        started = time.perf_counter()
        nominal = np.asarray(nominal, dtype=np.float32)
        if nominal.ndim != 2 or nominal.shape[1] != 17:
            raise ValueError("adaptive CBF expects [horizon,17] nominal actions")
        steps = min(self.qp_horizon, len(nominal))
        certificate_steps = min(self.certificate_horizon, len(nominal))
        previous = (
            self._previous_safe_action.copy()
            if previous_action is None else np.asarray(previous_action, dtype=np.float32).reshape(17)
        )
        state = np.asarray(state, dtype=np.float32).reshape(60)
        perceived_world, perceived_labels = self._obstacles_to_world(
            perceived_obstacles, state[:3],
        )
        table_world, table_labels = self._world_table_obstacles()
        all_obstacles = np.concatenate((perceived_world, table_world), axis=0)
        all_labels = perceived_labels + table_labels
        risk, soft_clearance, soft_witness = self._soft_risk(
            state, nominal, all_obstacles, all_labels,
        )
        source_table_world = table_world[:5]
        source_table_labels = table_labels[:5]
        table_risk, table_soft_clearance, table_witness = self._soft_risk(
            state, nominal, source_table_world, source_table_labels,
        )
        yaw_error = abs(_wrap(math.pi - float(state[2])))
        if yaw_error <= math.radians(8.0):
            self._rotation_complete = True
        side = 0.0
        if self._rotation_complete:
            side = self._update_avoidance_side(
                risk, soft_clearance, soft_witness, perceived_world, state,
            )
        nominal_certificate = self._certificate(
            state, nominal[:certificate_steps], all_obstacles, all_labels,
        )
        source_table_map, _ = self._clearance_map(
            state, nominal[:certificate_steps], source_table_world,
            source_table_labels, certificate_steps,
        )
        source_table_current = self._current_clearances(
            state, source_table_world, source_table_labels,
        )
        source_table_clearance_ok = self._barrier_clearance_satisfied(
            source_table_map, source_table_current, certificate_steps,
        )
        source_table_retreat_active = self._update_source_table_retreat(
            source_table_clearance_ok, source_table_current,
        )
        if not self._rotation_complete:
            if nominal_certificate[0]:
                result_actions = nominal.copy()
                source, status = "flow_rotation", "rotation_passthrough"
                certificate = nominal_certificate
            else:
                result_actions = nominal.copy()
                result_actions[0] = 0.0
                source, status = "rotation_safety_hold", "rotation_overlap_predicted"
                certificate = self._certificate(
                    state, result_actions[:certificate_steps], all_obstacles, all_labels,
                )
            action = result_actions[0]
            if certificate[0]:
                self._previous_safe_action = action.copy()
            base_escape, arm_escape = self._escape_progress(state)
            return AdaptiveCBFResult(
                result_actions, bool(certificate[0]), status, risk,
                certificate[1], soft_clearance,
                float(np.sqrt(np.mean(np.square(action - nominal[0])))),
                source, certificate[6].obstacle_label, certificate[6].robot_part,
                side, base_escape, arm_escape,
                certificate[2], certificate[3], certificate[4], certificate[5],
                1000.0 * (time.perf_counter() - started),
            )
        if risk <= 0.0 and nominal_certificate[0]:
            self._zero_risk_grasp_stall_frames = 0
            self._previous_safe_action = nominal[0].copy()
            base_escape, arm_escape = self._escape_progress(state)
            return AdaptiveCBFResult(
                nominal.copy(), True, "inactive_safe", 0.0,
                nominal_certificate[1], soft_clearance, 0.0,
                "flow", nominal_certificate[6].obstacle_label,
                nominal_certificate[6].robot_part, side, base_escape, arm_escape,
                nominal_certificate[2], nominal_certificate[3],
                nominal_certificate[4], nominal_certificate[5],
                1000.0 * (time.perf_counter() - started),
            )
        if self._update_zero_risk_grasp_stall(
            risk,
            nominal_certificate,
            release_allowed=bool(
                not source_table_retreat_active
                and not self._bimanual_detour_active
                and side == 0.0
            ),
        ):
            # Release a genuine deadlock only after it persists for half a
            # certificate horizon.  Replanning after one immediately
            # certified Flow step preserves the proactive positive-risk CBF
            # path while short zero-risk glitches remain fully filtered.
            immediate_certificate = self._certificate(
                state,
                nominal[:self.strategy.stall_release_horizon],
                all_obstacles,
                all_labels,
            )
            if immediate_certificate[0]:
                self._previous_safe_action = nominal[0].copy()
                base_escape, arm_escape = self._escape_progress(state)
                return AdaptiveCBFResult(
                    nominal.copy(), True, "zero_risk_grasp_stall_release", risk,
                    immediate_certificate[1], soft_clearance, 0.0,
                    "flow_zero_risk_stall_release",
                    immediate_certificate[6].obstacle_label,
                    immediate_certificate[6].robot_part,
                    side, base_escape, arm_escape,
                    immediate_certificate[2], immediate_certificate[3],
                    immediate_certificate[4], immediate_certificate[5],
                    1000.0 * (time.perf_counter() - started),
                )
        # The analytic retreat is for clearing the source table.  Applying it
        # to the destination table makes a safe delivery impossible by
        # continually pushing the base away from the goal; that table remains
        # covered by the ordinary predictive certificate and QP.
        table_threat = (
            table_witness.obstacle_index >= 0
            and source_table_retreat_active
        )
        if table_threat:
            # A short receding-horizon QP can repeatedly certify motion into
            # a table until the base reaches a state from which no lateral
            # escape remains.  For static table geometry, the globally safe
            # direction is available analytically: move the base directly
            # away from the witnessed table part while leaving Flow's arms
            # untouched.  This is geometry-derived rather than a tuned
            # left/right recovery distance.
            obstacle = source_table_world[0]
            away_world = state[:2] - obstacle[POSITION][:2]
            norm = float(np.linalg.norm(away_world))
            if norm > 1.0e-6:
                rotation = self._rotation(state[:3])
                away_local = rotation.T @ (away_world / norm)
                retreat = nominal[:certificate_steps].copy()
                retreat[:, :2] = np.clip(away_local[None], -1.0, 1.0)
                retreat[:, 3:] = 0.0
                retreat = self._rate_limit_arm_actions(retreat, previous)
                retreat_certificate = self._certificate(
                    state, retreat, all_obstacles, all_labels,
                )
                if retreat_certificate[0]:
                    result_actions = nominal.copy()
                    result_actions[:certificate_steps] = retreat
                    action = result_actions[0]
                    self._previous_safe_action = action.copy()
                    base_escape, arm_escape = self._escape_progress(state)
                    return AdaptiveCBFResult(
                        result_actions, True, "certified_table_retreat",
                        max(risk, table_risk), retreat_certificate[1],
                        min(soft_clearance, table_soft_clearance),
                        float(np.sqrt(np.mean(np.square(action - nominal[0])))),
                        "certified_table_retreat",
                        retreat_certificate[6].obstacle_label,
                        retreat_certificate[6].robot_part,
                        side, base_escape, arm_escape,
                        retreat_certificate[2], retreat_certificate[3],
                        retreat_certificate[4], retreat_certificate[5],
                        1000.0 * (time.perf_counter() - started),
                    )
        dynamic_threat = (
            soft_witness.obstacle_index >= 0
            and soft_witness.obstacle_index < len(perceived_world)
            and float(np.linalg.norm(
                perceived_world[soft_witness.obstacle_index, 3:9]
            )) > 1.0e-5
        )
        arm_threat = any(
            token in soft_witness.robot_part
            for token in ("arm", "gripper", "payload")
        )
        continuing_detour = (
            self._bimanual_detour_active
            and side != 0.0
            and self._avoidance_obstacle_index is not None
            and self._avoidance_hand_centers is not None
        )
        starting_detour = (
            dynamic_threat and arm_threat and side != 0.0 and risk >= 0.10
        )
        if starting_detour or continuing_detour:
            if starting_detour:
                self._bimanual_detour_active = True
            # Recompute the damped-Jacobian command from measured state on
            # every real control frame.  This is the closed-loop part of the
            # detour; caching the first command is only valid for a rollout,
            # not after the simulator has applied a different nominal action.
            arm_command = self._bimanual_escape_action(
                state, side, max(risk, 0.5), elevate=False,
            )
            detour = nominal[:certificate_steps].copy()
            # Own the arms for the full certificate while the threat latch is
            # active.  Once the target is reached ``arm_command`` becomes a
            # near-zero hold.  Releasing after one step let Flow pull the
            # hands back before the crossing sphere had actually passed.
            detour[:, 3:] = arm_command[None]
            detour = self._rate_limit_arm_actions(detour, previous)
            detour_certificate = self._certificate(
                state, detour, all_obstacles, all_labels,
            )
            if detour_certificate[0]:
                result_actions = nominal.copy()
                result_actions[:certificate_steps] = detour
                action = result_actions[0]
                self._previous_safe_action = action.copy()
                base_escape, arm_escape = self._escape_progress(state)
                return AdaptiveCBFResult(
                    result_actions, True, "certified_bimanual_detour", risk,
                    detour_certificate[1], soft_clearance,
                    float(np.sqrt(np.mean(np.square(action - nominal[0])))),
                    "certified_bimanual_detour",
                    detour_certificate[6].obstacle_label,
                    detour_certificate[6].robot_part,
                    side, base_escape, arm_escape,
                    detour_certificate[2], detour_certificate[3],
                    detour_certificate[4], detour_certificate[5],
                    1000.0 * (time.perf_counter() - started),
                )
        allow_arm_recovery = bool(dynamic_threat and arm_threat)
        target = self._escape_reference(
            nominal[:steps], state, risk, side,
            use_arms=allow_arm_recovery,
            elevate_arms=bool(risk >= 0.35),
        )
        active_obstacles: set[int] = set()
        hard_map, _ = self._clearance_map(
            state, nominal[:steps], all_obstacles, all_labels, steps,
        )
        for (step, obstacle_index), clearance in hard_map.items():
            del step
            table = obstacle_index >= len(perceived_world)
            if clearance <= (0.08 if table else self.soft_trigger) or clearance < 0.0:
                active_obstacles.add(obstacle_index)
        if soft_witness.obstacle_index >= 0:
            active_obstacles.add(soft_witness.obstacle_index)
        center = nominal[:steps].copy()
        solved_actions: np.ndarray | None = None
        status = "qp_not_run"
        certificate = nominal_certificate
        for _round in range(self.maximum_rounds):
            solved_actions, status = self._solve_round(
                state, center, target, all_obstacles, all_labels,
                previous, active_obstacles,
            )
            if solved_actions is None:
                break
            candidate = nominal.copy()
            candidate[:steps] = solved_actions
            certificate = self._certificate(
                state, candidate[:certificate_steps], all_obstacles, all_labels,
            )
            if certificate[0]:
                break
            center = solved_actions
            solved_actions = None
            status = "qp_postcheck_retry"
        if solved_actions is not None and certificate[0]:
            result_actions = nominal.copy()
            result_actions[:steps] = solved_actions
            source = "adaptive_cbf_qp"
        else:
            recovered, source, certificate = self._recovery(
                state, nominal, all_obstacles, all_labels, risk, side,
                previous, allow_arm_recovery=allow_arm_recovery,
                dynamic_obstacle_index=(
                    soft_witness.obstacle_index
                    if dynamic_threat and arm_threat and risk >= 0.10
                    else None
                ),
            )
            result_actions = nominal.copy()
            result_actions[:len(recovered)] = recovered
            status = f"{status}_{source}"
        action = result_actions[0]
        if certificate[0]:
            self._previous_safe_action = action.copy()
        base_escape, arm_escape = self._escape_progress(state)
        return AdaptiveCBFResult(
            result_actions,
            bool(certificate[0]),
            status,
            risk,
            certificate[1],
            soft_clearance,
            float(np.sqrt(np.mean(np.square(action - nominal[0])))),
            source,
            certificate[6].obstacle_label,
            certificate[6].robot_part,
            side,
            base_escape,
            arm_escape,
            certificate[2],
            certificate[3],
            certificate[4],
            certificate[5],
            1000.0 * (time.perf_counter() - started),
        )
