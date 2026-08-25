"""Privileged Phase 3 demonstration expert without CBF-QP."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np

from flowcarrycbf.policies.flowcarry_cbf.control.expert import offset_region
from flowcarrycbf.policies.flowcarry_cbf.control.kinematics import FullBodyKinematics, Rollout
from flowcarrycbf.policies.flowcarry_cbf.control.schema import (
    ACCELERATION,
    CUBE_TYPE,
    HALF_EXTENTS,
    OBSTACLE_FEATURES,
    POSITION,
    TYPE_INDEX,
    VALID_INDEX,
    VELOCITY,
)
from flowcarrycbf.envs.tasks.utils.pinoc_utils import PinTiagoIKSolver


TABLE_CENTERS = np.asarray([[1.25, 0.0, 0.375], [-4.25, 0.0, 0.375]], dtype=np.float32)
TABLE_HALF_EXTENTS = np.asarray([0.40, 0.50, 0.375], dtype=np.float32)
# Keep the same 5x5 candidate count as V2, with a slightly wider local
# displacement envelope for the reduced Phase 3 obstacle sizes.
PHASE3_OFFSET_GRID = (-0.40, -0.20, -0.10, 0.0, 0.10, 0.20, 0.40)
PLANNING_CLEARANCE_TOLERANCE = 0.02
# The geometric rollout is intentionally cheaper than PhysX and can differ by
# a few centimetres at contact transitions.  Keep the public 2 cm planning
# tolerance, but require an additional execution buffer for accepted oracle
# branches so replay remains above the independent 6 cm hard gate.
EXECUTION_CLEARANCE_BUFFER = 0.04


def _wrap(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def fixed_table_obstacles(reference_base: np.ndarray) -> np.ndarray:
    """Return conservative table AABBs in the current base frame."""

    base = np.asarray(reference_base, dtype=np.float32).reshape(3)
    c, s = math.cos(float(base[2])), math.sin(float(base[2]))
    rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    result = np.zeros((2, OBSTACLE_FEATURES), dtype=np.float32)
    for index, center in enumerate(TABLE_CENTERS):
        result[index, :2] = rotation @ (center[:2] - base[:2])
        result[index, 2] = center[2]
        # A world-axis box becomes an oriented box in the base frame.  The
        # enclosing AABB is conservative and supported by the existing model.
        result[index, 9:11] = np.abs(rotation) @ TABLE_HALF_EXTENTS[:2]
        result[index, 11] = TABLE_HALF_EXTENTS[2]
        result[index, TYPE_INDEX] = CUBE_TYPE
        result[index, VALID_INDEX] = 1.0
    return result


def fixed_table_collision_obstacles(reference_base: np.ndarray) -> np.ndarray:
    """Return the actual tabletop and four leg colliders for both tables."""

    base = np.asarray(reference_base, dtype=np.float32).reshape(3)
    c, s = math.cos(float(base[2])), math.sin(float(base[2]))
    rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    parts: list[np.ndarray] = []
    for center in TABLE_CENTERS:
        geometry = [
            (
                np.asarray([0.0, 0.0, 0.72], dtype=np.float32),
                np.asarray([0.40, 0.50, 0.03], dtype=np.float32),
            ),
        ]
        geometry.extend(
            (
                np.asarray([sx * 0.31, sy * 0.41, 0.345], dtype=np.float32),
                np.asarray([0.03, 0.03, 0.345], dtype=np.float32),
            )
            for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1))
        )
        for offset, half_extents in geometry:
            obstacle = np.zeros(OBSTACLE_FEATURES, dtype=np.float32)
            obstacle[:2] = rotation @ (center[:2] + offset[:2] - base[:2])
            obstacle[2] = offset[2]
            obstacle[9:11] = np.abs(rotation) @ half_extents[:2]
            obstacle[11] = half_extents[2]
            obstacle[TYPE_INDEX] = CUBE_TYPE
            obstacle[VALID_INDEX] = 1.0
            parts.append(obstacle)
    return np.asarray(parts, dtype=np.float32)


def world_obstacles_to_local(values: np.ndarray, reference_base: np.ndarray) -> np.ndarray:
    """Transform a time-indexed world obstacle schedule into one base frame."""

    result = np.asarray(values, dtype=np.float32).copy()
    base = np.asarray(reference_base, dtype=np.float32).reshape(3)
    c, s = math.cos(float(base[2])), math.sin(float(base[2]))
    rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    result[..., :2] = (result[..., :2] - base[:2]) @ rotation.T
    result[..., 3:5] = result[..., 3:5] @ rotation.T
    result[..., 6:8] = result[..., 6:8] @ rotation.T
    static = result[..., TYPE_INDEX] > 0.5
    if np.any(static):
        half_xy = result[..., 9:11].copy()
        result[..., 9:11] = half_xy @ np.abs(rotation).T
    return result


def time_indexed_minimum_clearance(
    kinematics: FullBodyKinematics,
    rollout: Rollout,
    obstacles: np.ndarray,
    *,
    safety_distance: float = 0.10,
) -> float:
    """Direct signed-distance check against exact per-step obstacle states."""

    values = np.asarray(obstacles, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != OBSTACLE_FEATURES:
        raise ValueError("obstacle schedule must have shape [time, obstacles, 15]")
    limit = min(len(rollout.spheres), len(values))
    minimum = float("inf")
    for step in range(limit):
        for obstacle in values[step]:
            if obstacle[VALID_INDEX] < 0.5:
                continue
            minimum = min(
                minimum,
                kinematics.step_minimum_clearance(rollout, step, obstacle)
                - float(safety_distance),
            )
    return minimum


def planning_clearance_satisfied(clearance: float) -> bool:
    """Accept at most 2 cm of numerical error around the planning margin."""

    return bool(float(clearance) >= -PLANNING_CLEARANCE_TOLERANCE - 1.0e-9)


def execution_clearance_satisfied(clearance: float) -> bool:
    """Require a positive geometric margin before committing a replay plan."""

    return bool(float(clearance) >= EXECUTION_CLEARANCE_BUFFER - 1.0e-9)


@dataclass(frozen=True)
class OracleAction:
    action: np.ndarray
    region: str
    clearance: float
    phase: str
    wait_until_step: int
    candidate_count: int
    latency_ms: float


@dataclass(frozen=True)
class OraclePlan:
    actions: np.ndarray
    phases: tuple[str, ...]
    clearance: float
    route: str
    arm_region: str
    wait_steps: int
    candidate_count: int
    latency_ms: float


class Phase3OracleTrajectoryExpert:
    """Privileged local-response expert without CBF-QP.

    Static cubes select the macroscopic base route.  Dynamic spheres are only
    considered in a short receding horizon and normally change coordinated
    arm motion or waiting, never the complete base route.
    """

    # One-second receding horizon at the 10 Hz control rate.  Dynamic balls
    # may influence the local arm response only inside this window.
    HORIZON = 10
    ARM_TRANSITION_STEPS = 10
    WAIT_SEARCH_HORIZON = 30
    QUICK_WAIT_SEARCH_HORIZON = 10
    QUICK_ARM_OFFSETS = (
        (0.10, 0.10), (0.10, -0.10), (-0.10, 0.10), (-0.10, -0.10),
        (0.0, 0.10), (0.0, -0.10), (0.10, 0.0), (-0.10, 0.0),
        (0.20, 0.20), (0.20, -0.20), (-0.20, 0.20), (-0.20, -0.20),
    )
    # Quick joint templates need a visible base correction. The command is
    # still checked against the full-body collision model before acceptance.
    QUICK_EMERGENCY_LATERAL_ACTIONS = (0.80, -0.80)
    QUICK_BASE_SHIFT_STEPS = 14
    QUICK_BASE_RETURN_STEPS = 12
    QUICK_CANDIDATE_BUDGET = 30
    # The Isaac replay gate remains hard at 8 cm.  The lightweight oracle
    # Keep the planned clearance at the agreed 10 cm; replay still enforces
    # the independent hard 8 cm failure gate.
    SAFETY_DISTANCE = 0.10
    # Match the accepted /50 action style: only a short retreat before the
    # initial turn. The independent PhysX replay still enforces the 8 cm gate.
    DEPARTURE_X = -0.10
    DEPARTURE_TOLERANCE = 0.001
    CUBE_TRIGGER_CLEARANCE = 0.30
    CUBE_APPROACH_SPEED = 0.20
    GOAL = np.asarray([-3.0, 0.0], dtype=np.float32)
    LATERAL_LANE = 0.90
    EMERGENCY_LATERAL_ACTIONS = (0.10, -0.10, 0.20, -0.20, 0.30, -0.30)

    def __init__(self, *, dt: float = 0.1, profile: str = "legacy") -> None:
        if profile not in {"legacy", "quick"}:
            raise ValueError("profile must be legacy or quick")
        self.profile = str(profile)
        self.kinematics = FullBodyKinematics(dt=dt)
        self._offset_targets = self._build_offset_targets()
        self._ordered_offsets = sorted(
            self._offset_targets,
            key=lambda value: (abs(value[0]) + abs(value[1]), abs(value[1]), value),
        )
        self.reset()

    def reset(self) -> None:
        self._waypoint_index = 0
        self._waypoints: list[np.ndarray] = []
        self._episode_nominal_left: np.ndarray | None = None
        self._episode_nominal_right: np.ndarray | None = None
        self._last_phase = ""
        self._cached_wait_until = -1
        self._turn_yaw = math.pi - 1.0e-3
        self._quick_turn_initialized = False
        self._side_shift_complete = False
        self._cube: np.ndarray | None = None
        self._cube_route_active = False
        self._cube_side_x = 0.0
        self._quick_recovery_offset: tuple[float, float] | None = None
        self._quick_recovery_lateral = 0.0
        self._quick_recovery_brake = False
        self._quick_recovery_phase = ""
        self._quick_recovery_age = 0
        self._quick_arm_response_required = False
        self._quick_arm_response_done = False
        self._quick_joint_response_required = False
        self._quick_base_response_done = False

    def _clone_for_branch(self) -> "Phase3OracleTrajectoryExpert":
        """Clone mutable planner state without rebuilding the IK tables.

        Branches only mutate route cursors, cached wait state and nominal arm
        references.  Kinematics and the precomputed offset targets are
        read-only during planning, so sharing them avoids a costly deepcopy of
        Pinocchio/IK objects at every beam expansion.
        """

        branch = object.__new__(type(self))
        branch.__dict__ = self.__dict__.copy()
        branch._waypoints = [np.asarray(value, dtype=np.float32).copy() for value in self._waypoints]
        branch._episode_nominal_left = (
            None if self._episode_nominal_left is None
            else np.asarray(self._episode_nominal_left, dtype=np.float32).copy()
        )
        branch._episode_nominal_right = (
            None if self._episode_nominal_right is None
            else np.asarray(self._episode_nominal_right, dtype=np.float32).copy()
        )
        branch._cube = None if self._cube is None else np.asarray(self._cube, dtype=np.float32).copy()
        branch._ordered_offsets = tuple(self._ordered_offsets)
        return branch

    def _build_offset_targets(self) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
        solvers = {
            side: PinTiagoIKSolver(move_group=f"arm_{side}", max_rot_vel=100.0)
            for side in ("left", "right")
        }
        targets: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        for offset_y in PHASE3_OFFSET_GRID:
            for offset_z in PHASE3_OFFSET_GRID:
                offset = (float(offset_y), float(offset_z))
                values = []
                for side in ("left", "right"):
                    if offset == (0.0, 0.0):
                        values.append(self.kinematics.HOLD.copy())
                        continue
                    solver = solvers[side]
                    position, quaternion = solver.solve_fk_tiago(self.kinematics.HOLD.copy())
                    success, solution = solver.solve_ik_pos_tiago(
                        np.asarray(position).copy() + np.asarray([0.0, offset_y, offset_z]),
                        quaternion,
                        curr_joints=self.kinematics.HOLD.copy(),
                        n_trials=2,
                        dt=0.05,
                        pos_threshold=0.003,
                        angle_threshold=np.deg2rad(1.5),
                    )
                    values.append(np.asarray(solution if success else self.kinematics.HOLD, dtype=np.float32))
                targets[offset] = (values[0], values[1])
        return targets

    def _candidate_offsets(self) -> tuple[tuple[float, float], ...]:
        if self.profile != "quick":
            return tuple(self._ordered_offsets)
        return tuple(
            offset for offset in self.QUICK_ARM_OFFSETS
            if offset in self._offset_targets
        )

    def _initialize_waypoints(self, world_schedule: np.ndarray) -> None:
        cubes = [
            obstacle
            for obstacle in np.asarray(world_schedule[0], dtype=np.float32)
            if obstacle[VALID_INDEX] > 0.5 and obstacle[TYPE_INDEX] > 0.5
        ]
        self._waypoints = []
        self._cube = None
        self._cube_route_active = False
        if cubes:
            cube = min(cubes, key=lambda item: abs(float(item[0]) + 1.6))
            self._cube = np.asarray(cube, dtype=np.float32).copy()
            self._cube_route_active = False
            side = -1.0 if float(cube[1]) >= 0.0 else 1.0
            lateral = side * self.LATERAL_LANE
            self._turn_yaw = math.copysign(math.pi - 1.0e-3, -lateral)
            self._waypoints.extend(
                [
                    np.asarray([float(cube[0]) + 0.62, lateral], dtype=np.float32),
                    np.asarray([float(cube[0]) - 0.62, lateral], dtype=np.float32),
                    np.asarray([min(float(cube[0]) - 0.78, -2.55), 0.0], dtype=np.float32),
                ]
            )
        else:
            # Dynamic spheres never select a global base lane.  The nominal
            # center route is deliberately retained so CBF-QP has meaningful
            # local work to do at evaluation time.
            lane = 0.0
            self._turn_yaw = math.pi - 1.0e-3
            self._waypoints.extend(
                [
                    np.asarray([-1.35, lane], dtype=np.float32),
                    np.asarray([-2.45, lane], dtype=np.float32),
                    np.asarray([-2.70, 0.0], dtype=np.float32),
                ]
            )
        self._waypoints.append(self.GOAL.copy())

    def _activate_cube_route(self, base: np.ndarray) -> None:
        """Lock the bypass at the near-obstacle x instead of at reset."""

        if self._cube is None:
            return
        side = -1.0 if float(self._cube[1]) >= 0.0 else 1.0
        lateral = side * self.LATERAL_LANE
        self._cube_route_active = True
        self._cube_side_x = float(base[0])
        self._side_shift_complete = False
        self._waypoint_index = 0
        self._waypoints = [
            np.asarray([float(self._cube[0]) + 0.62, lateral], dtype=np.float32),
            np.asarray([float(self._cube[0]) - 0.62, lateral], dtype=np.float32),
            np.asarray([min(float(self._cube[0]) - 0.78, -2.55), 0.0], dtype=np.float32),
            self.GOAL.copy(),
        ]

    def _waypoints_for_lane(self, world_schedule: np.ndarray, lane: float) -> list[np.ndarray]:
        cubes = [
            obstacle
            for obstacle in np.asarray(world_schedule[0], dtype=np.float32)
            if obstacle[VALID_INDEX] > 0.5 and obstacle[TYPE_INDEX] > 0.5
        ]
        if cubes:
            cube = min(cubes, key=lambda item: abs(float(item[0]) + 1.6))
            side = -1.0 if float(cube[1]) >= 0.0 else 1.0
            lane = side * self.LATERAL_LANE
            return [
                np.asarray([float(cube[0]) + 0.62, lane], dtype=np.float32),
                np.asarray([float(cube[0]) - 0.62, lane], dtype=np.float32),
                np.asarray([min(float(cube[0]) - 0.78, -2.55), 0.0], dtype=np.float32),
                self.GOAL.copy(),
            ]
        return [
            np.asarray([-1.35, lane], dtype=np.float32),
            np.asarray([-2.45, lane], dtype=np.float32),
            np.asarray([-2.70, 0.0], dtype=np.float32),
            self.GOAL.copy(),
        ]

    def _base_trajectory(
        self,
        state: np.ndarray,
        waypoints: list[np.ndarray],
        wait_steps: int,
        max_steps: int,
    ) -> tuple[np.ndarray, list[str]]:
        base = np.asarray(state[:3], dtype=np.float32).copy()
        yaw = float(base[2])
        c, s = math.cos(yaw), math.sin(yaw)
        world_velocity = np.asarray(state[3:6], dtype=np.float32)
        command = np.asarray(
            [
                c * world_velocity[0] + s * world_velocity[1],
                -s * world_velocity[0] + c * world_velocity[1],
                world_velocity[2],
            ],
            dtype=np.float32,
        )
        actions: list[np.ndarray] = []
        phases: list[str] = []

        def append(action: np.ndarray, phase: str) -> None:
            nonlocal base, command
            value = np.asarray(action, dtype=np.float32).copy()
            actions.append(value)
            phases.append(phase)
            desired = value[:3] * self.kinematics.base_limits
            command += np.clip(
                desired - command,
                -self.kinematics.base_acceleration_step,
                self.kinematics.base_acceleration_step,
            )
            yaw_value = float(base[2])
            cosine, sine = math.cos(yaw_value), math.sin(yaw_value)
            base[0] += (cosine * command[0] - sine * command[1]) * self.kinematics.dt
            base[1] += (sine * command[0] + cosine * command[1]) * self.kinematics.dt
            base[2] = _wrap(float(base[2] + command[2] * self.kinematics.dt))

        def command_to(target: np.ndarray, target_yaw: float, phase: str) -> np.ndarray:
            pseudo = np.asarray(state, dtype=np.float32).copy()
            pseudo[:3] = base
            return self._base_action(pseudo, target, target_yaw, phase)

        for _ in range(int(wait_steps)):
            append(np.zeros(17, dtype=np.float32), "wait_at_start")
        if getattr(self, "profile", "legacy") == "quick":
            # Quick collection starts with an in-place base turn. Translation
            # remains exactly zero until the 180-degree heading is reached.
            while (
                abs(_wrap(self._turn_yaw - float(base[2]))) > np.deg2rad(2.0)
                and len(actions) < max_steps
            ):
                action = np.zeros(17, dtype=np.float32)
                action[:3] = command_to(base[:2], self._turn_yaw, "rotate_clear")
                action[:2] = 0.0
                append(action, "rotate_clear")
        else:
            while base[0] > self.DEPARTURE_X + self.DEPARTURE_TOLERANCE and len(actions) < max_steps:
                action = np.zeros(17, dtype=np.float32)
                action[:3] = command_to(np.asarray([self.DEPARTURE_X, 0.0]), 0.0, "depart_table")
                append(action, "depart_table")
        entry_lane = float(waypoints[0][1]) if waypoints else 0.0
        turn_yaw = (
            math.copysign(math.pi - 1.0e-3, -entry_lane)
            if abs(entry_lane) > 0.20 else math.pi - 1.0e-3
        )
        if abs(entry_lane) > 0.20:
            side_target = np.asarray([self.DEPARTURE_X, entry_lane], dtype=np.float32)
            while np.linalg.norm(base[:2] - side_target) > 0.06 and len(actions) < max_steps:
                action = np.zeros(17, dtype=np.float32)
                # Translate while still facing the entry direction.  Rotating
                # first sweeps the payload into a cube near the nominal lane.
                action[:3] = command_to(side_target, float(state[2]), "preturn_side_shift")
                append(action, "preturn_side_shift")
        while abs(_wrap(turn_yaw - float(base[2]))) > np.deg2rad(5.0) and len(actions) < max_steps:
            action = np.zeros(17, dtype=np.float32)
            action[:3] = command_to(base[:2], turn_yaw, "rotate_clear")
            append(action, "rotate_clear")
        for waypoint_index, waypoint in enumerate(waypoints):
            final = waypoint_index == len(waypoints) - 1
            tolerance = 0.04 if final else 0.10
            phase = "approach_goal" if final else "traverse"
            while np.linalg.norm(base[:2] - waypoint) > tolerance and len(actions) < max_steps:
                action = np.zeros(17, dtype=np.float32)
                action[:3] = command_to(waypoint, turn_yaw, phase)
                append(action, phase)
        for _ in range(12):
            if len(actions) >= max_steps:
                break
            append(np.zeros(17, dtype=np.float32), "goal_hold")
        return np.asarray(actions, dtype=np.float32), phases

    def _apply_arm_region(
        self,
        state: np.ndarray,
        base_actions: np.ndarray,
        phases: list[str],
        offset: tuple[float, float],
    ) -> np.ndarray:
        actions = np.asarray(base_actions, dtype=np.float32).copy()
        left = np.asarray(state[8:15], dtype=np.float32).copy()
        right = np.asarray(state[15:22], dtype=np.float32).copy()
        raw_left, raw_right = self._offset_targets[offset]
        target_left = left + (raw_left - self.kinematics.HOLD)
        target_right = right + (raw_right - self.kinematics.HOLD)
        nominal_left, nominal_right = left.copy(), right.copy()
        for index, phase in enumerate(phases):
            desired_left, desired_right = (
                (target_left, target_right)
                if phase == "traverse"
                else (nominal_left, nominal_right)
            )
            left_action = np.clip(
                (desired_left - left) / self.kinematics.max_arm_delta,
                -self.kinematics.arm_action_limit,
                self.kinematics.arm_action_limit,
            )
            right_action = np.clip(
                (desired_right - right) / self.kinematics.max_arm_delta,
                -self.kinematics.arm_action_limit,
                self.kinematics.arm_action_limit,
            )
            actions[index, 3:10] = left_action
            actions[index, 10:17] = right_action
            left += left_action * self.kinematics.max_arm_delta
            right += right_action * self.kinematics.max_arm_delta
        return actions

    def _branch_candidates(
        self,
        state: np.ndarray,
        world_schedule: np.ndarray,
        current_step: int,
    ) -> list[tuple["Phase3OracleTrajectoryExpert", OracleAction]]:
        """Return a bounded set of locally different receding-horizon choices.

        The normal route evaluates one clone. Only a dynamic-risk decision is
        expanded, with several deterministic offset orderings. Reordering the
        existing finite candidate set is enough to expose alternative arm,
        braking, wait, and emergency choices without adding a continuous
        optimizer or changing the nominal route.
        """

        preferred_offsets = (
            (0.40, 0.40), (0.40, -0.40), (-0.40, 0.40), (-0.40, -0.40),
            (0.40, 0.0), (-0.40, 0.0), (0.0, 0.40), (0.0, -0.40),
            (0.20, 0.20), (0.20, -0.20), (-0.20, 0.20), (-0.20, -0.20),
        )
        candidates: list[tuple[Phase3OracleTrajectoryExpert, OracleAction]] = []
        seen: set[tuple[str, bytes]] = set()

        def evaluate(order: tuple[tuple[float, float], ...] | None) -> OracleAction:
            branch = self._clone_for_branch()
            if order is not None:
                branch._ordered_offsets = tuple(
                    list(order) + [value for value in self._ordered_offsets if value not in order]
                )
            decision = branch.act(state, world_schedule, current_step)
            key = (str(decision.phase), np.asarray(decision.action, dtype=np.float32).tobytes())
            if key not in seen:
                seen.add(key)
                candidates.append((branch, decision))
            return decision

        baseline = evaluate(None)
        dynamic = (
            baseline.region not in {"nominal"}
            or baseline.phase in {"invalid_robot_candidate", "no_safe_local_window"}
            or str(baseline.phase).startswith("dynamic_")
        )
        if not dynamic:
            return candidates
        if self.profile == "quick":
            # ``act`` already evaluated the bounded local recovery set. A
            # second pass with reordered offsets multiplies every expensive
            # whole-body check without adding a meaningful route alternative.
            return candidates
        for offset in preferred_offsets:
            if offset not in self._ordered_offsets:
                continue
            evaluate((offset,))
        return candidates

    def plan(
        self,
        state: np.ndarray,
        world_schedule: np.ndarray,
        *,
        max_steps: int = 600,
        progress_callback: Callable[[int, int, int], None] | None = None,
        required_arm_response: bool = False,
        required_joint_response: bool = False,
    ) -> OraclePlan | None:
        """Build a local-response trajectory with bounded backtracking.

        The base route is generated from static geometry only. Dynamic sphere
        responses are expanded only around a detected local risk. A small beam
        of cloned oracle states prevents a locally safe arm offset from making
        the remaining trajectory impossible, while preserving the nominal
        center route whenever no local risk is present.
        """

        started = time.perf_counter()
        initial = np.asarray(state, dtype=np.float32).reshape(60).copy()
        schedule = np.asarray(world_schedule, dtype=np.float32)
        cubes = bool(np.any((schedule[0, :, VALID_INDEX] > 0.5) & (schedule[0, :, TYPE_INDEX] > 0.5)))
        self.reset()
        self._quick_arm_response_required = bool(
            self.profile == "quick" and required_arm_response
        )
        self._quick_arm_response_done = False
        self._quick_joint_response_required = bool(
            self.profile == "quick" and required_joint_response
        )
        self._quick_base_response_done = False
        simulated = initial.copy()
        actions: list[np.ndarray] = []
        phases: list[str] = []
        arm_regions: set[str] = set()
        wait_steps = 0
        candidate_count = 0
        reached_goal = False
        self._last_plan_failure = ""
        self._last_plan_failure_state = None
        self._last_plan_failure_obstacles = None
        self._last_plan_actions = np.empty((0, 17), dtype=np.float32)
        self._last_plan_phases: tuple[str, ...] = ()

        def advance(value: np.ndarray, action: np.ndarray) -> np.ndarray:
            rollout = self.kinematics.rollout(value, np.asarray(action, dtype=np.float32)[None])
            result = value.copy()
            result[:3] = rollout.base[-1]
            result[8:15] = rollout.left_q[-1]
            result[15:22] = rollout.right_q[-1]
            # Keep the base velocity state used by PhysX-style acceleration
            # limiting. Resetting it to zero every control step makes the
            # offline route crawl and changes the old demonstration timing.
            yaw = float(value[2])
            c, s = math.cos(yaw), math.sin(yaw)
            local_velocity = np.asarray(
                [c * value[3] + s * value[4], -s * value[3] + c * value[4], value[5]],
                dtype=np.float32,
            )
            desired = np.asarray(action[:3], dtype=np.float32) * self.kinematics.base_limits
            local_velocity += np.clip(
                desired - local_velocity,
                -self.kinematics.base_acceleration_step,
                self.kinematics.base_acceleration_step,
            )
            c, s = math.cos(float(result[2])), math.sin(float(result[2]))
            result[3:6] = np.asarray(
                [c * local_velocity[0] - s * local_velocity[1],
                 s * local_velocity[0] + c * local_velocity[1],
                 local_velocity[2]],
                dtype=np.float32,
            )
            return result

        beam_width = 1 if getattr(self, "profile", "legacy") == "quick" else 8
        max_planning_steps = min(int(max_steps), len(schedule) - 1)
        initial_experts: list[Phase3OracleTrajectoryExpert] = [self]
        if getattr(self, "profile", "legacy") == "quick":
            # Quick uses one deterministic global route. Dynamic obstacles are
            # handled only by the local recovery set in ``act``.
            self._initialize_waypoints(schedule)
            initial_experts = [self]
        beam: list[dict[str, Any]] = [
            {
                "expert": expert,
                "state": simulated.copy(),
                "actions": [],
                "phases": [],
                "wait_steps": 0,
                "arm_regions": set(),
                "candidate_count": 0,
                "min_clearance": float("inf"),
                "emergency_steps": 0,
            }
            for expert in initial_experts
        ]
        goal_node: dict[str, Any] | None = None

        for step in range(max_planning_steps):
            if progress_callback is not None:
                progress_callback(step, max_planning_steps, len(beam))
            expanded: list[dict[str, Any]] = []
            for node in beam:
                candidates = node["expert"]._branch_candidates(
                    node["state"], schedule, step,
                )
                for branch_expert, decision in candidates:
                    if decision.phase in {"invalid_robot_candidate", "no_safe_local_window"}:
                        continue
                    action = np.asarray(decision.action, dtype=np.float32)
                    next_state = advance(node["state"], action)
                    next_actions = [*node["actions"], action]
                    next_phases = [*node["phases"], decision.phase]
                    next_regions = set(node["arm_regions"])
                    if decision.region == "wait":
                        next_wait_steps = int(node["wait_steps"]) + 1
                    else:
                        next_wait_steps = int(node["wait_steps"])
                    if decision.region not in {"nominal", "emergency_base", "wait"}:
                        next_regions.add(decision.region)
                    next_node = {
                        "expert": branch_expert,
                        "state": next_state,
                        "actions": next_actions,
                        "phases": next_phases,
                        "wait_steps": next_wait_steps,
                        "arm_regions": next_regions,
                        "candidate_count": int(node["candidate_count"]) + int(decision.candidate_count),
                        "min_clearance": min(float(node["min_clearance"]), float(decision.clearance)),
                        "emergency_steps": int(node["emergency_steps"]) + int(decision.region == "emergency_base"),
                    }
                    goal_tolerance = 0.12 if self.profile == "quick" else 0.06
                    if (
                        np.linalg.norm(next_state[:2] - self.GOAL) <= goal_tolerance
                        and abs(_wrap(math.pi - float(next_state[2]))) <= np.deg2rad(8.0)
                    ):
                        goal_node = next_node
                        break
                    expanded.append(next_node)
                if goal_node is not None:
                    break
            if goal_node is not None:
                break
            if not expanded:
                break
            expanded.sort(
                key=lambda node: (
                    int(node["emergency_steps"]),
                    int(node["wait_steps"]),
                    -float(node["min_clearance"]),
                    len(node["arm_regions"]),
                    len(node["actions"]),
                )
            )
            beam = expanded[:beam_width]

        if goal_node is not None:
            actions = list(goal_node["actions"])
            phases = list(goal_node["phases"])
            simulated = np.asarray(goal_node["state"], dtype=np.float32).copy()
            wait_steps = int(goal_node["wait_steps"])
            arm_regions = set(goal_node["arm_regions"])
            candidate_count = int(goal_node["candidate_count"])
            goal_hold_steps = 30 if self.profile == "quick" else 12
            for _ in range(goal_hold_steps):
                if len(actions) >= max_steps:
                    break
                hold = np.zeros(17, dtype=np.float32)
                actions.append(hold)
                phases.append("goal_hold")
                simulated = advance(simulated, hold)
            reached_goal = True
        else:
            failed = beam[0] if beam else {
                "state": initial.copy(), "actions": [], "candidate_count": 0,
            }
            simulated = np.asarray(failed["state"], dtype=np.float32).copy()
            actions = list(failed.get("actions", []))
            phases = list(failed.get("phases", []))
            candidate_count = int(failed.get("candidate_count", 0))
            self._last_plan_failure = (
                f"beam_no_complete_safe_trajectory:step_{len(actions)}:"
                f"base=({float(simulated[0]):.3f},{float(simulated[1]):.3f},"
                f"{float(simulated[2]):.3f})"
            )
            self._last_plan_failure_state = simulated.copy()
            self._last_plan_failure_obstacles = schedule[min(len(actions), len(schedule) - 1)].copy()

        if not reached_goal or not actions:
            self._last_plan_actions = np.asarray(actions, dtype=np.float32).reshape(-1, 17)
            self._last_plan_phases = tuple(phases)
            if not self._last_plan_failure:
                self._last_plan_failure = "goal_not_reached"
            self._last_plan_failure_state = simulated.copy()
            self._last_plan_failure_obstacles = schedule[min(len(actions), len(schedule) - 1)].copy()
            return None
        values = np.asarray(actions, dtype=np.float32)
        table_a_safety = np.asarray(
            [
                0.08 if phase == "rotate_clear" else self.SAFETY_DISTANCE
                for phase in phases
            ],
            dtype=np.float32,
        )
        valid, clearance, invariant_valid = self._check(
            initial,
            values,
            schedule[1 : 1 + len(values)],
            table_a_safety_distance=table_a_safety,
            include_tables=False,
        )
        if not valid and not self._candidate_collision_free(clearance, invariant_valid):
            self._last_plan_actions = values.copy()
            self._last_plan_phases = tuple(phases)
            self._last_plan_failure = f"replay_clearance:{clearance:.4f}"
            return None
        if not arm_regions:
            arm_region = "nominal"
        elif len(arm_regions) == 1:
            arm_region = next(iter(arm_regions))
        else:
            arm_region = "mixed"
        return OraclePlan(
            actions=values,
            phases=tuple(phases),
            clearance=float(clearance),
            route="cube_bypass" if cubes else "nominal",
            arm_region=arm_region,
            wait_steps=wait_steps,
            candidate_count=candidate_count,
            latency_ms=1000.0 * (time.perf_counter() - started),
        )

    def nominal_debug_plan(
        self,
        state: np.ndarray,
        world_schedule: np.ndarray,
        *,
        max_steps: int = 600,
    ) -> np.ndarray:
        """Return the static-route/nominal-arm trajectory for diagnostics.

        This is intentionally uncertified and is only used by the optional
        prescreen video debug path when the local oracle rejects a seed.
        """
        self.reset()
        state = np.asarray(state, dtype=np.float32).reshape(60)
        schedule = np.asarray(world_schedule, dtype=np.float32)
        actions, phases = self._base_trajectory(
            state, self._waypoints_for_lane(schedule, 0.0), 0, max_steps
        )
        return self._apply_arm_region(state, actions, phases, (0.0, 0.0))

    def _phase_and_target(self, state: np.ndarray, world_schedule: np.ndarray) -> tuple[str, np.ndarray, float]:
        base = np.asarray(state[:3], dtype=np.float32)
        if not self._waypoints:
            self._initialize_waypoints(world_schedule)
        if getattr(self, "profile", "legacy") == "quick" and not self._quick_turn_initialized:
            # Match the proven /50 reset protocol: the fixed zero pose turns
            # toward +pi through the negative-yaw half-circle.
            self._quick_turn_initialized = True
        if getattr(self, "profile", "legacy") != "quick" and base[0] > self.DEPARTURE_X + self.DEPARTURE_TOLERANCE:
            return "depart_table", np.asarray([self.DEPARTURE_X, 0.0], dtype=np.float32), 0.0
        # Departure is always followed by the initial in-place turn.  Static
        # cube approach happens only after the robot faces the transport
        # direction; otherwise the base keeps reversing toward the cube and
        # turns far too late.
        if not self._cube_route_active:
            initial_yaw_error = _wrap(self._turn_yaw - float(base[2]))
            if abs(initial_yaw_error) > np.deg2rad(5.0):
                return "rotate_clear", base[:2].copy(), self._turn_yaw
        if self._cube is not None and not self._cube_route_active:
            center_distance = float(np.linalg.norm(base[:2] - self._cube[:2]))
            cube_clearance = center_distance - float(np.max(self._cube[9:11]))
            if cube_clearance > self.CUBE_TRIGGER_CLEARANCE:
                return "cube_approach", np.asarray([float(self._cube[0]), 0.0], dtype=np.float32), 0.0
            self._activate_cube_route(base)
        entry_lane = float(self._waypoints[0][1]) if self._waypoints else 0.0
        yaw_error = _wrap(self._turn_yaw - float(base[2]))
        if abs(entry_lane) > 0.20:
            side_x = self._cube_side_x if self._cube_route_active else self.DEPARTURE_X
            side_target = np.asarray([side_x, entry_lane], dtype=np.float32)
            # The discrete base controller can settle a few centimetres past
            # the target while retaining the required cube-side clearance.
            if not self._side_shift_complete and np.linalg.norm(base[:2] - side_target) > 0.12:
                return "preturn_side_shift", side_target, float(state[2])
            self._side_shift_complete = True
        if abs(yaw_error) > np.deg2rad(5.0):
            return "rotate_clear", base[:2].copy(), self._turn_yaw
        while self._waypoint_index < len(self._waypoints) - 1:
            if np.linalg.norm(self._waypoints[self._waypoint_index] - base[:2]) > 0.12:
                break
            self._waypoint_index += 1
        target = self._waypoints[min(self._waypoint_index, len(self._waypoints) - 1)]
        return ("approach_goal" if self._waypoint_index == len(self._waypoints) - 1 else "traverse"), target, self._turn_yaw

    def _base_action(self, state: np.ndarray, target_xy: np.ndarray, target_yaw: float, phase: str) -> np.ndarray:
        base = np.asarray(state[:3], dtype=np.float32)
        result = np.zeros(3, dtype=np.float32)
        delta = np.asarray(target_xy, dtype=np.float32) - base[:2]
        translation_tolerance = (
            self.DEPARTURE_TOLERANCE if phase == "depart_table" else 0.04
        )
        if phase != "rotate_clear" and np.linalg.norm(delta) > translation_tolerance:
            speed_limit = self.CUBE_APPROACH_SPEED if phase == "cube_approach" else 0.20
            world_velocity = np.clip(0.9 * delta, -speed_limit, speed_limit)
            c, s = math.cos(float(base[2])), math.sin(float(base[2]))
            result[:2] = np.asarray(
                [c * world_velocity[0] + s * world_velocity[1], -s * world_velocity[0] + c * world_velocity[1]],
                dtype=np.float32,
            ) / self.kinematics.base_limits[:2]
        yaw_error = _wrap(target_yaw - float(base[2]))
        if phase == "rotate_clear" or abs(yaw_error) < np.deg2rad(20.0):
            result[2] = np.clip(yaw_error / 0.8 / self.kinematics.base_limits[2], -0.80, 0.80)
        return np.clip(result, -0.80, 0.80)

    def _sequence(
        self,
        state: np.ndarray,
        base_action: np.ndarray,
        offset: tuple[float, float],
        lateral_delta: float,
        *,
        brake_base: bool = False,
    ) -> np.ndarray:
        result = np.zeros((self.HORIZON, 17), dtype=np.float32)
        if not brake_base:
            result[:, :3] = base_action
        result[:, 1] = np.clip(result[:, 1] + lateral_delta, -0.80, 0.80)
        if self._episode_nominal_left is None or self._episode_nominal_right is None:
            self._episode_nominal_left = np.asarray(state[8:15], dtype=np.float32).copy()
            self._episode_nominal_right = np.asarray(state[15:22], dtype=np.float32).copy()
        raw_left, raw_right = self._offset_targets[offset]
        left_target = self._episode_nominal_left + (raw_left - self.kinematics.HOLD)
        right_target = self._episode_nominal_right + (raw_right - self.kinematics.HOLD)
        transition_steps = min(self.ARM_TRANSITION_STEPS, self.HORIZON)
        left = np.clip(
            (left_target - state[8:15])
            / (float(transition_steps) * self.kinematics.max_arm_delta),
            -0.70,
            0.70,
        )
        right = np.clip(
            (right_target - state[15:22])
            / (float(transition_steps) * self.kinematics.max_arm_delta),
            -0.70,
            0.70,
        )
        result[:transition_steps, 3:10] = left
        result[:transition_steps, 10:17] = right
        return result

    def _check(
        self,
        state: np.ndarray,
        sequence: np.ndarray,
        future_world: np.ndarray,
        *,
        table_a_safety_distance: float | np.ndarray | None = None,
        include_tables: bool = False,
    ) -> tuple[bool, float, bool]:
        rollout = self.kinematics.rollout(state, sequence)
        local = world_obstacles_to_local(future_world[: len(sequence)], state[:3])
        dynamic_clearance = time_indexed_minimum_clearance(
            self.kinematics, rollout, local, safety_distance=self.SAFETY_DISTANCE
        )
        clearance = dynamic_clearance
        if include_tables:
            # Certify the turn against the actual table volumes in every
            # rollout frame. The table AABBs are conservative, while the
            # whole-body rollout still checks the payload, arms and grippers.
            table_local = np.repeat(
                fixed_table_collision_obstacles(state[:3])[None], len(sequence), axis=0
            )
            table_safety = 0.08
            if table_a_safety_distance is not None:
                table_safety = max(
                    0.0,
                    min(0.08, float(np.min(np.asarray(table_a_safety_distance)))),
                )
            table_clearance = time_indexed_minimum_clearance(
                self.kinematics, rollout, table_local, safety_distance=table_safety
            )
            clearance = min(clearance, table_clearance)
        del table_a_safety_distance
        invariant_valid = bool(
            self.kinematics.grasp_consistent(
                rollout, reference_vector=self.kinematics.grasp_vector(state)
            )
            and self.kinematics.joint_limits_satisfied(rollout)
            and self.kinematics.self_collision_free(rollout)
        )
        return bool(clearance >= 0.0 and invariant_valid), float(clearance), invariant_valid

    def _candidate_collision_free(self, clearance: float, invariant_valid: bool) -> bool:
        """Quick treats the 10 cm margin as ranking, not an acceptance gate."""
        if getattr(self, "profile", "legacy") != "quick":
            return bool(invariant_valid and execution_clearance_satisfied(clearance))
        return bool(invariant_valid and float(clearance) + self.SAFETY_DISTANCE >= 0.0)

    def _quick_arm_response_window(
        self, state: np.ndarray, schedule: np.ndarray, current_step: int,
    ) -> bool:
        """Return whether an arm-template sphere is locally actionable.

        Template metadata is intentionally not part of the obstacle feature
        tensor.  The geometric window below is the bridge between the fixed
        scene template and the local planner: a sphere must be beside the arm
        corridor and close enough in x to affect the next one-second rollout.
        """

        if not getattr(self, "_quick_arm_response_required", False):
            return False
        if getattr(self, "_quick_arm_response_done", False):
            return False
        values = np.asarray(schedule, dtype=np.float32)
        start = max(0, int(current_step))
        stop = min(len(values), start + self.HORIZON + 1)
        base = np.asarray(state[:2], dtype=np.float32)
        for frame in values[start:stop]:
            for obstacle in frame:
                if obstacle[VALID_INDEX] < 0.5 or obstacle[TYPE_INDEX] > 0.5:
                    continue
                x_delta = abs(float(obstacle[0] - base[0]))
                y_delta = abs(float(obstacle[1] - base[1]))
                z_value = float(obstacle[2])
                # The arm event occupies the first longitudinal slot.  The
                # joint template's second sphere is a base-only event and
                # must not consume the arm response trigger.
                if (
                    float(obstacle[0]) < -1.70
                    and x_delta <= 0.95
                    and 0.45 <= y_delta <= 1.15
                    and 0.65 <= z_value <= 1.35
                ):
                    return True
        return False

    def _quick_candidate_budget(self) -> int:
        """Reserve enough bounded search for the joint base correction."""

        if getattr(self, "_quick_joint_response_required", False):
            return self.QUICK_CANDIDATE_BUDGET + 30
        return self.QUICK_CANDIDATE_BUDGET

    def _quick_base_response_window(
        self, state: np.ndarray, schedule: np.ndarray, current_step: int,
    ) -> bool:
        """Detect the second, base-only sphere in the joint template."""

        if not getattr(self, "_quick_joint_response_required", False):
            return False
        if getattr(self, "_quick_base_response_done", False):
            return False
        # The joint template's intended order is base correction followed by
        # the arm event. Once the arm event has completed, the same moving
        # sphere must not be interpreted as a second base event.
        if getattr(self, "_quick_arm_response_done", False):
            return False
        values = np.asarray(schedule, dtype=np.float32)
        start = max(0, int(current_step))
        # The base slot is earlier than the arm event, but with the reduced
        # longitudinal spacing it can enter the local horizon around step
        # 130-160. Keep the window open until the robot has passed that slot;
        # _quick_base_response_done prevents repeated corrections.
        if start >= 190:
            return False
        stop = min(len(values), start + self.HORIZON + 1)
        base = np.asarray(state[:2], dtype=np.float32)
        for frame in values[start:stop]:
            for obstacle in frame:
                if obstacle[VALID_INDEX] < 0.5 or obstacle[TYPE_INDEX] > 0.5:
                    continue
                if (
                    float(obstacle[0]) > -1.70
                    and abs(float(obstacle[0] - base[0])) <= 0.95
                    and 0.45 <= abs(float(obstacle[1] - base[1])) <= 0.95
                    and 0.20 <= float(obstacle[2]) <= 1.70
                ):
                    return True
        return False

    def _earliest_safe_step(
        self,
        state: np.ndarray,
        sequence: np.ndarray,
        schedule: np.ndarray,
        current_step: int,
        table_a_safety_distance: float,
    ) -> int:
        # Waiting is local.  A full-schedule scan lets a distant sphere
        # determine the whole robot route and creates implausible early
        # avoidance.
        wait_horizon = (
            self.QUICK_WAIT_SEARCH_HORIZON
            if getattr(self, "profile", "legacy") == "quick"
            else self.WAIT_SEARCH_HORIZON
        )
        maximum = min(
            len(schedule) - len(sequence),
            current_step + wait_horizon,
        )
        for start in range(current_step + 1, maximum + 1):
            valid, clearance, invariant_valid = self._check(
                state, sequence, schedule[start : start + len(sequence)],
                table_a_safety_distance=table_a_safety_distance,
            )
            if valid or self._candidate_collision_free(clearance, invariant_valid):
                return start
        return -1

    def _safe_to_wait(
        self,
        state: np.ndarray,
        schedule: np.ndarray,
        current_step: int,
        wait_until: int,
        table_a_safety_distance: float,
    ) -> bool:
        wait_steps = int(wait_until - current_step)
        if wait_steps <= 0:
            return True
        hold = np.zeros((wait_steps, 17), dtype=np.float32)
        future = schedule[current_step + 1 : current_step + 1 + wait_steps]
        valid, clearance, invariant_valid = self._check(
            state, hold, future,
            table_a_safety_distance=table_a_safety_distance,
        )
        return bool(
                valid or self._candidate_collision_free(clearance, invariant_valid)
        )

    def act(self, state: np.ndarray, world_schedule: np.ndarray, current_step: int) -> OracleAction:
        started = time.perf_counter()
        state = np.asarray(state, dtype=np.float32).reshape(60)
        schedule = np.asarray(world_schedule, dtype=np.float32)
        phase, target_xy, target_yaw = self._phase_and_target(state, schedule)
        table_a_safety_distance = (
            0.0
            if getattr(self, "profile", "legacy") == "quick" and phase == "rotate_clear"
            else 0.08 if phase == "rotate_clear" else self.SAFETY_DISTANCE
        )
        if self._last_phase in {"depart_table", "rotate_clear"} and phase not in {
            "depart_table", "rotate_clear",
        }:
            # Lock the stable post-turn grasp once. Contact dynamics settle the
            # real joints slightly during departure, so reset-frame targets are
            # not the correct restoration pose for the corridor.
            self._episode_nominal_left = state[8:15].copy()
            self._episode_nominal_right = state[15:22].copy()
        self._last_phase = phase
        if current_step < self._cached_wait_until:
            return OracleAction(
                np.zeros(17, dtype=np.float32), "wait", float("nan"),
                "wait_for_dynamic", self._cached_wait_until, 0,
                1000.0 * (time.perf_counter() - started),
            )
        self._cached_wait_until = -1
        base_action = self._base_action(state, target_xy, target_yaw, phase)
        force_arm_response = bool(
            getattr(self, "profile", "legacy") == "quick"
            and phase != "rotate_clear"
            and self._quick_arm_response_window(state, schedule, current_step)
        )
        force_base_response = bool(
            getattr(self, "profile", "legacy") == "quick"
            and phase != "rotate_clear"
            and self._quick_base_response_window(state, schedule, current_step)
        )
        future = schedule[current_step + 1 : current_step + 1 + self.HORIZON]
        if len(future) < self.HORIZON:
            future = np.concatenate((future, np.repeat(schedule[-1:], self.HORIZON - len(future), axis=0)))
        if self.profile == "quick" and self._quick_recovery_offset is not None and phase != "rotate_clear":
            recovery_age = int(self._quick_recovery_age)
            if (
                self._quick_recovery_phase == "dynamic_base_avoidance"
                and recovery_age >= self.QUICK_BASE_SHIFT_STEPS
            ):
                self._quick_recovery_phase = "dynamic_base_return"
                self._quick_recovery_lateral = 0.0
                self._quick_recovery_age = 0
                recovery_age = 0
            elif (
                self._quick_recovery_phase == "dynamic_base_return"
                and recovery_age >= self.QUICK_BASE_RETURN_STEPS
            ):
                self._quick_recovery_offset = None
                self._quick_recovery_lateral = 0.0
                self._quick_recovery_phase = ""
                self._quick_recovery_age = 0
        if self.profile == "quick" and self._quick_recovery_offset is not None and phase != "rotate_clear":
            recovery_age = int(self._quick_recovery_age)
            recovery = self._sequence(
                state,
                base_action,
                self._quick_recovery_offset,
                self._quick_recovery_lateral,
                # Braking is a short arm-transition action.  Holding a
                # braking candidate forever would freeze the base before the
                # goal, even after the arms have reached their offset.
                brake_base=self._quick_recovery_brake and recovery_age < self.ARM_TRANSITION_STEPS,
            )
            valid, clearance, invariant_valid = self._check(
                state, recovery, future,
                table_a_safety_distance=table_a_safety_distance,
            )
            if self._candidate_collision_free(clearance, invariant_valid):
                self._quick_recovery_age = recovery_age + 1
                return OracleAction(
                    recovery[0], offset_region(*self._quick_recovery_offset),
                    clearance, self._quick_recovery_phase, current_step, 1,
                    1000.0 * (time.perf_counter() - started),
                )
            self._quick_recovery_offset = None
            self._quick_recovery_lateral = 0.0
            self._quick_recovery_brake = False
            self._quick_recovery_phase = ""
            self._quick_recovery_age = 0
        candidate_count = 0
        best_unsafe: tuple[float, np.ndarray, tuple[float, float]] | None = None
        best_tolerated: tuple[
            float, np.ndarray, tuple[float, float], str
        ] | None = None
        # Certify the nominal action first.  Arm motion is only considered when
        # this nominal 1-second window actually violates the 10 cm plan margin.
        nominal = self._sequence(state, base_action, (0.0, 0.0), 0.0)
        certification_sequence = nominal
        certification_future = future
        if getattr(self, "profile", "legacy") == "quick" and phase == "rotate_clear":
            # Rotation is certified receding-horizon, one committed yaw step at
            # a time. A full 1 s turn envelope would reject safe intermediate
            # headings because it treats the swept arm envelope as occupied.
            certification_sequence = nominal[:1]
            certification_future = future[:1]
        nominal_valid, nominal_clearance, nominal_invariant = self._check(
            state, certification_sequence, certification_future,
            table_a_safety_distance=table_a_safety_distance,
            # This approximation rejects the collision-free x=0 turn already
            # demonstrated by /50. PhysX table contacts remain a hard gate.
            include_tables=False,
        )
        candidate_count += 1
        if getattr(self, "profile", "legacy") == "quick" and phase == "rotate_clear":
            # The start protocol is deliberately single-mode: no arm offset,
            # lateral correction, braking translation or reverse action is
            # allowed before the certified 180-degree turn completes.
            if nominal_valid or (
                self._candidate_collision_free(nominal_clearance, nominal_invariant)
            ):
                return OracleAction(
                    nominal[0], "nominal", nominal_clearance, phase,
                    current_step, candidate_count,
                    1000.0 * (time.perf_counter() - started),
                )
            return OracleAction(
                np.zeros(17, dtype=np.float32), "wait", nominal_clearance,
                "no_safe_local_window", len(schedule) - 1, candidate_count,
                1000.0 * (time.perf_counter() - started),
            )
        if (
            not nominal_valid
            and phase == "cube_approach"
            and self._cube is not None
            and not self._cube_route_active
        ):
            # The approach candidate is certified over the next second. If
            # that window first becomes unsafe, switch to the cube side route
            # at this exact state instead of waiting one more control step.
            self._activate_cube_route(state[:3])
            phase, target_xy, target_yaw = self._phase_and_target(state, schedule)
            table_a_safety_distance = (
                0.0
                if getattr(self, "profile", "legacy") == "quick" and phase == "rotate_clear"
                else 0.08 if phase == "rotate_clear" else self.SAFETY_DISTANCE
            )
            base_action = self._base_action(state, target_xy, target_yaw, phase)
            nominal = self._sequence(state, base_action, (0.0, 0.0), 0.0)
            nominal_valid, nominal_clearance, nominal_invariant = self._check(
                state, nominal, future,
                table_a_safety_distance=table_a_safety_distance,
            )
            candidate_count += 1
        if force_base_response:
            # Joint templates must expose a real base correction even when a
            # nominal rollout remains collision-free. Try the smallest local
            # lateral actions first, keeping the arms nominal for this event.
            # Consume the event before evaluating candidates so beam clones do
            # not repeatedly re-trigger the same moving sphere window.
            for lateral in self.QUICK_EMERGENCY_LATERAL_ACTIONS:
                sequence = self._sequence(state, base_action, (0.0, 0.0), lateral)
                valid, clearance, invariant_valid = self._check(
                    state, sequence, future,
                    table_a_safety_distance=table_a_safety_distance,
                )
                candidate_count += 1
                if self._candidate_collision_free(clearance, invariant_valid):
                    # Consume the event only after a collision-free candidate
                    # is actually certified. Failed candidates must remain
                    # retryable as the moving sphere enters the local window.
                    self._quick_base_response_done = True
                    self._quick_recovery_offset = (0.0, 0.0)
                    self._quick_recovery_lateral = float(lateral)
                    self._quick_recovery_brake = False
                    self._quick_recovery_phase = "dynamic_base_avoidance"
                    self._quick_recovery_age = 0
                    return OracleAction(
                        sequence[0], "emergency_base", clearance,
                        "dynamic_emergency_base", current_step,
                        candidate_count,
                        1000.0 * (time.perf_counter() - started),
                    )
        if nominal_valid and not force_arm_response and not force_base_response:
            if self.profile == "quick":
                self._quick_recovery_offset = None
            return OracleAction(
                nominal[0], "nominal", nominal_clearance, phase,
                current_step, candidate_count,
                1000.0 * (time.perf_counter() - started),
            )
        if (
            nominal_invariant and nominal_clearance >= 0.0
            and not force_arm_response and not force_base_response
        ):
            if self.profile == "quick":
                self._quick_recovery_offset = None
            return OracleAction(
                nominal[0], "nominal", nominal_clearance, phase,
                current_step, candidate_count,
                1000.0 * (time.perf_counter() - started),
            )
        # Quick keeps the 10 cm value as a preference. If nominal is already
        # collision-free but below that planning preference, retain it as a
        # fallback while checking whether a local recovery improves margin.
        if (
            self.profile == "quick"
            and not force_arm_response
            and not force_base_response
            and self._candidate_collision_free(
            nominal_clearance, nominal_invariant
            )
        ):
            best_tolerated = (
                nominal_clearance, nominal, (0.0, 0.0), phase,
            )
        # The nominal window is below 10 cm; now and only now search local arm
        # offsets. There is no alternate global base lane here.
        for offset in self._candidate_offsets():
            if offset == (0.0, 0.0):
                continue
            if self.profile == "quick" and candidate_count >= self._quick_candidate_budget():
                break
            sequence = self._sequence(state, base_action, offset, 0.0)
            valid, clearance, invariant_valid = self._check(
                state, sequence, future,
                table_a_safety_distance=table_a_safety_distance,
            )
            candidate_count += 1
            if valid:
                self._cached_wait_until = -1
                region = offset_region(*offset)
                if self.profile == "quick":
                    if force_arm_response:
                        self._quick_arm_response_done = True
                    self._quick_recovery_offset = offset
                    self._quick_recovery_lateral = 0.0
                    self._quick_recovery_brake = False
                    self._quick_recovery_phase = "dynamic_arm_avoidance"
                    self._quick_recovery_age = 0
                return OracleAction(
                    sequence[0], region, clearance,
                    "dynamic_arm_avoidance" if region != "nominal" else phase,
                    current_step, candidate_count,
                    1000.0 * (time.perf_counter() - started),
                )
            if self._candidate_collision_free(clearance, invariant_valid):
                if best_tolerated is None or clearance > best_tolerated[0]:
                    best_tolerated = (
                        clearance, sequence, offset, "dynamic_arm_avoidance",
                    )
            if invariant_valid and (best_unsafe is None or clearance > best_unsafe[0]):
                best_unsafe = (clearance, sequence, offset)
        # A large dynamic sphere can remain in the arm workspace throughout
        # the 2 s reaction horizon. Keep the response local by braking the
        # base while moving both arms to the smallest safe coordinated offset.
        # The base route itself is unchanged and resumes after the sphere
        # clears the short horizon.
        for offset in self._candidate_offsets():
            if offset == (0.0, 0.0):
                continue
            if self.profile == "quick" and candidate_count >= self._quick_candidate_budget():
                break
            sequence = self._sequence(
                state, base_action, offset, 0.0, brake_base=True,
            )
            valid, clearance, invariant_valid = self._check(
                state, sequence, future,
                table_a_safety_distance=table_a_safety_distance,
            )
            candidate_count += 1
            if valid:
                self._cached_wait_until = -1
                region = offset_region(*offset)
                if self.profile == "quick":
                    if force_arm_response:
                        self._quick_arm_response_done = True
                    self._quick_recovery_offset = offset
                    self._quick_recovery_lateral = 0.0
                    self._quick_recovery_brake = True
                    self._quick_recovery_phase = "dynamic_arm_avoidance_braking"
                    self._quick_recovery_age = 0
                return OracleAction(
                    sequence[0], region, clearance,
                    "dynamic_arm_avoidance_braking", current_step,
                    candidate_count, 1000.0 * (time.perf_counter() - started),
                )
            if self._candidate_collision_free(clearance, invariant_valid):
                if best_tolerated is None or clearance > best_tolerated[0]:
                    best_tolerated = (
                        clearance,
                        sequence,
                        offset,
                        "dynamic_arm_avoidance_braking",
                    )
            if invariant_valid and (best_unsafe is None or clearance > best_unsafe[0]):
                best_unsafe = (clearance, sequence, offset)

        if best_tolerated is not None:
            clearance, sequence, offset, avoidance_phase = best_tolerated
            self._cached_wait_until = -1
            if self.profile == "quick" and offset != (0.0, 0.0):
                if force_arm_response:
                    self._quick_arm_response_done = True
                self._quick_recovery_offset = offset
                self._quick_recovery_lateral = 0.0
                self._quick_recovery_brake = avoidance_phase.endswith("braking")
                self._quick_recovery_phase = avoidance_phase
                self._quick_recovery_age = 0
            return OracleAction(
                sequence[0], offset_region(*offset), clearance,
                avoidance_phase, current_step, candidate_count,
                1000.0 * (time.perf_counter() - started),
            )

        if best_unsafe is None:
            return OracleAction(
                np.zeros(17, dtype=np.float32), "wait", float("-inf"),
                "invalid_robot_candidate", len(schedule) - 1, candidate_count,
                1000.0 * (time.perf_counter() - started),
            )

        wait_until = self._earliest_safe_step(
            state, best_unsafe[1], schedule, current_step,
            table_a_safety_distance,
        )
        if wait_until >= 0 and self._safe_to_wait(
            state, schedule, current_step, wait_until,
            table_a_safety_distance,
        ):
            self._cached_wait_until = wait_until
            if self.profile == "quick":
                self._quick_recovery_offset = None
            wait = np.zeros(17, dtype=np.float32)
            return OracleAction(
                wait, "wait", best_unsafe[0], "wait_for_dynamic", wait_until,
                candidate_count, 1000.0 * (time.perf_counter() - started),
            )

        # Only after arm offsets and a wait of up to 3 s fail may the base make a
        # small local emergency correction.  The next receding-horizon call
        # returns to the static-cube route.
        best_emergency_tolerated: tuple[
            float, np.ndarray, tuple[float, float], float
        ] | None = None
        lateral_actions = (
            self.QUICK_EMERGENCY_LATERAL_ACTIONS
            if getattr(self, "profile", "legacy") == "quick"
            else self.EMERGENCY_LATERAL_ACTIONS
        )
        for lateral in lateral_actions:
            for offset in self._candidate_offsets():
                if self.profile == "quick" and candidate_count >= self._quick_candidate_budget():
                    break
                sequence = self._sequence(state, base_action, offset, lateral)
                valid, clearance, invariant_valid = self._check(
                    state, sequence, future,
                    table_a_safety_distance=table_a_safety_distance,
                )
                candidate_count += 1
                if valid:
                    self._last_act_debug = {"best_clearance": float(clearance), "offset": offset, "lateral": float(lateral)}
                    if self.profile == "quick":
                        self._quick_recovery_offset = offset
                        self._quick_recovery_lateral = float(lateral)
                        self._quick_recovery_brake = False
                        self._quick_recovery_phase = "dynamic_emergency_base"
                        self._quick_recovery_age = 0
                    return OracleAction(
                        sequence[0], "emergency_base", clearance,
                        "dynamic_emergency_base", current_step,
                        candidate_count, 1000.0 * (time.perf_counter() - started),
                    )
                if self._candidate_collision_free(clearance, invariant_valid):
                    if (
                        best_emergency_tolerated is None
                        or clearance > best_emergency_tolerated[0]
                    ):
                        best_emergency_tolerated = (
                            clearance, sequence, offset, float(lateral),
                        )
                if invariant_valid and (best_unsafe is None or clearance > best_unsafe[0]):
                    best_unsafe = (clearance, sequence, offset)
        if best_emergency_tolerated is not None:
            clearance, sequence, offset, lateral = best_emergency_tolerated
            self._last_act_debug = {
                "best_clearance": float(clearance),
                "offset": offset,
                "lateral": lateral,
            }
            if self.profile == "quick":
                self._quick_recovery_offset = offset
                self._quick_recovery_lateral = lateral
                self._quick_recovery_brake = False
                self._quick_recovery_phase = "dynamic_emergency_base"
                self._quick_recovery_age = 0
            return OracleAction(
                sequence[0], "emergency_base", clearance,
                "dynamic_emergency_base", current_step,
                candidate_count, 1000.0 * (time.perf_counter() - started),
            )
        self._last_act_debug = {
            "best_clearance": float(best_unsafe[0]),
            "offset": best_unsafe[2],
            "lateral": 0.0,
        }
        return OracleAction(
            np.zeros(17, dtype=np.float32), "wait", best_unsafe[0],
            "no_safe_local_window", len(schedule) - 1, candidate_count,
            1000.0 * (time.perf_counter() - started),
        )
