"""Model-based closed-loop direct-action expert for V2 demonstrations."""

from __future__ import annotations

import numpy as np

from flowcarrycbf.envs.tasks.utils.pinoc_utils import PinTiagoIKSolver
from .kinematics import FullBodyKinematics
from .schema import ACTION_DIMENSION, CONDITION_DIMENSION, HORIZON, TYPE_INDEX, VALID_INDEX
from .safety import WholeBodyCBFQP


REGIONS = ("nominal", "up", "down", "left", "right")
OFFSET_GRID = (-0.15, -0.075, 0.0, 0.075, 0.15)
REGION_OFFSETS = {"nominal": (0.0, 0.0), "up": (0.0, 0.15), "down": (0.0, -0.15), "left": (-0.15, 0.0), "right": (0.15, 0.0)}


def offset_region(offset_y: float, offset_z: float) -> str:
    if abs(offset_y) < 1.0e-8 and abs(offset_z) < 1.0e-8:
        return "nominal"
    if abs(offset_z) >= abs(offset_y):
        return "up" if offset_z > 0.0 else "down"
    return "right" if offset_y > 0.0 else "left"


class BimanualExpertPlanner:
    """Uses current tracks only; it never consumes future obstacle waypoints."""

    STATIC_ROUTE_OFFSET = 0.90
    STATIC_ROUTE_LONGITUDINAL_BUFFER = 0.80

    def __init__(self, *, dt: float = 0.1, horizon: int = HORIZON) -> None:
        self.horizon = int(horizon)
        self.kinematics = FullBodyKinematics(dt=dt)
        self.safety = WholeBodyCBFQP(self.kinematics)
        self.last_region = "nominal"
        self.last_offset = (0.0, 0.0)
        self._offset_targets = self._build_offset_targets()
        self._special_offset_targets = self._build_special_offset_targets()
        self._region_targets = {
            region: self._offset_targets[offset]
            for region, offset in REGION_OFFSETS.items()
        }
        self.reset()

    def reset(self) -> None:
        self.last_region = "nominal"
        self.last_offset = (0.0, 0.0)
        self._global_seed_initialized = False
        self._region_seed_offsets = {}

    def _build_offset_targets(self) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
        solvers = {
            "left": PinTiagoIKSolver(move_group="arm_left", max_rot_vel=100.0),
            "right": PinTiagoIKSolver(move_group="arm_right", max_rot_vel=100.0),
        }
        targets: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        for offset_y in OFFSET_GRID:
            for offset_z in OFFSET_GRID:
                offset = (offset_y, offset_z)
                if offset == (0.0, 0.0):
                    targets[offset] = (
                        self.kinematics.HOLD.copy(),
                        self.kinematics.HOLD.copy(),
                    )
                    continue
                values = []
                for side in ("left", "right"):
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

    def _build_special_offset_targets(self) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
        """Exact IK targets for calibrated intermediate safety offsets."""
        targets: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        for offset_z in (-0.11, -0.12, -0.13):
            values = []
            for side in ("left", "right"):
                solver = PinTiagoIKSolver(move_group=f"arm_{side}", max_rot_vel=100.0)
                position, quaternion = solver.solve_fk_tiago(self.kinematics.HOLD.copy())
                success, solution = solver.solve_ik_pos_tiago(
                    np.asarray(position).copy() + np.asarray([0.0, 0.0, offset_z]),
                    quaternion,
                    curr_joints=self.kinematics.HOLD.copy(),
                    n_trials=2,
                    dt=0.05,
                    pos_threshold=0.003,
                    angle_threshold=np.deg2rad(1.5),
                )
                values.append(np.asarray(solution if success else self.kinematics.HOLD, dtype=np.float32))
            targets[(0.0, offset_z)] = (values[0], values[1])
        return targets

    @staticmethod
    def _state_goal(state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float32).reshape(60)[53:56]

    @classmethod
    def _base_translation_goal(
        cls,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        """Return an opposite-side waypoint when a cube blocks the direct route."""

        goal = np.asarray(goal, dtype=np.float32).reshape(3)
        distance = float(np.linalg.norm(goal[:2]))
        if distance < 1.0e-6:
            return goal[:2].copy()
        direction = goal[:2] / distance
        lateral_axis = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        cubes = []
        for obstacle in np.asarray(obstacles, dtype=np.float32).reshape(-1, 15):
            if obstacle[VALID_INDEX] < 0.5 or obstacle[TYPE_INDEX] < 0.5:
                continue
            longitudinal = float(np.dot(obstacle[:2], direction))
            if 0.10 < longitudinal < distance + cls.STATIC_ROUTE_LONGITUDINAL_BUFFER:
                cubes.append((longitudinal, obstacle))
        if not cubes:
            return goal[:2].copy()
        longitudinal, cube = min(cubes, key=lambda item: item[0])
        cube_lateral = float(np.dot(cube[:2], lateral_axis))
        route_side = -1.0 if cube_lateral >= 0.0 else 1.0
        route_distance = min(
            distance,
            longitudinal + cls.STATIC_ROUTE_LONGITUDINAL_BUFFER,
        )
        return (
            direction * route_distance
            + lateral_axis * route_side * cls.STATIC_ROUTE_OFFSET
        ).astype(np.float32)

    def _sequence_for_offset(self, state: np.ndarray, obstacles: np.ndarray, offset: tuple[float, float]) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(60)
        goal = self._state_goal(state)
        translation_goal = self._base_translation_goal(goal, obstacles)
        actions = np.zeros((self.horizon, ACTION_DIMENSION), dtype=np.float32)
        angular_limit = float(self.kinematics.base_limits[2])
        turn_in_place = abs(float(goal[2])) > np.deg2rad(6.0)
        rotation_steps = min(
            14,
            max(1, int(np.ceil(abs(float(goal[2])) / max(angular_limit * 0.1, 1.0e-6)))),
        )
        for index in range(min(22, self.horizon)):
            remaining = max(0.25, float(self.horizon - index) * 0.1)
            if turn_in_place and index < rotation_steps:
                rotation_time = max(rotation_steps * 0.1, 0.1)
                actions[index, 2] = np.clip(goal[2] / rotation_time / angular_limit, -0.85, 0.85)
            else:
                actions[index, 0] = np.clip(translation_goal[0] / remaining / float(self.kinematics.base_limits[0]), -0.85, 0.85)
                actions[index, 1] = np.clip(translation_goal[1] / remaining / float(self.kinematics.base_limits[1]), -0.65, 0.65)
                actions[index, 2] = np.clip(goal[2] / remaining / angular_limit, -0.85, 0.85)
        left_target, right_target = self._offset_targets[offset]
        # Returning to the nominal carry pose is a physical grasp transition,
        # not just another avoidance seed. Use twice the settling horizon so a
        # pure-contact payload is not pulled sideways while the base advances.
        settle_horizon = 40.0 if offset == (0.0, 0.0) else 10.0
        left_command = np.clip((left_target - state[8:15]) / (settle_horizon * self.kinematics.max_arm_delta), -1.0, 1.0)
        right_command = np.clip((right_target - state[15:22]) / (settle_horizon * self.kinematics.max_arm_delta), -1.0, 1.0)
        # Move to the selected bimanual target during the first second, then
        # hold it for the rest of the horizon. Returning to HOLD halfway
        # through every rolling plan makes the physical payload oscillate while
        # a sphere is still in its sweep plane and can break pure-contact
        # grasping. Once the plane is passed, ``plan`` selects the nominal
        # offset again and restores the standard carry posture through the same
        # rate-limited command.
        actions[:10, 3:10] = left_command
        actions[:10, 10:17] = right_command
        return np.clip(actions, -1.0, 1.0)

    def _sequence(self, state: np.ndarray, obstacles: np.ndarray, region: str) -> np.ndarray:
        return self._sequence_for_offset(state, obstacles, REGION_OFFSETS[region])

    def _ranked_grid(self, state: np.ndarray, obstacles: np.ndarray, region: str | None = None):
        ranked = []
        for offset in self._offset_targets:
            candidate_region = offset_region(*offset)
            if region is not None and candidate_region != region:
                continue
            sequence = self._sequence_for_offset(state, obstacles, offset)
            rollout = self.kinematics.rollout(state, sequence[: self.safety.certified_steps])
            clearance = self.kinematics.minimum_clearance(rollout, obstacles, certified_steps=10)
            ranked.append((clearance, abs(offset[0]) + abs(offset[1]), candidate_region, offset, sequence))
        # Once the 10 cm+uncertainty margin is satisfied, prefer the smallest
        # payload displacement. This avoids unnecessary diagonal grid corners
        # whose combined Y/Z offset approaches the inherited V1 slip limit.
        ranked.sort(
            key=lambda item: (
                item[0] >= 0.0,
                -item[1] if item[0] >= 0.0 else item[0],
                item[0],
            ),
            reverse=True,
        )
        return ranked

    @staticmethod
    def _dynamic_vertical_region(state: np.ndarray, obstacles: np.ndarray) -> str | None:
        """Hold a vertical escape until the nearest sphere plane is passed."""
        state = np.asarray(state, dtype=np.float32).reshape(60)
        obstacles = np.asarray(obstacles, dtype=np.float32).reshape(-1, 15)
        dynamic = [
            obstacle for obstacle in obstacles
            if obstacle[14] > 0.5 and obstacle[12] < 0.5 and 0.10 <= float(obstacle[0]) <= 1.50
        ]
        if not dynamic:
            return None
        nearest = min(dynamic, key=lambda obstacle: float(obstacle[0]))
        payload_height = float(state[42]) if float(state[42]) > 0.45 else 0.9076
        return "up" if float(nearest[2]) <= payload_height else "down"

    def plan_candidates(self, state: np.ndarray, obstacles: np.ndarray) -> list[tuple[str, np.ndarray, object]]:
        candidates = []
        for region in REGIONS:
            selected = None
            for _, _, _, offset, sequence in self._ranked_grid(state, obstacles, region):
                projection = self.safety.project(sequence, state, obstacles)
                selected = (region, projection.actions, projection)
                if projection.feasible:
                    self.last_offset = offset
                    break
            assert selected is not None
            candidates.append(selected)
        feasible = [item for item in candidates if item[2].feasible]
        if feasible:
            feasible.sort(key=lambda item: (-item[2].minimum_clearance, REGIONS.index(item[0])))
            self.last_region = feasible[0][0]
        return candidates

    def plan_region(self, state: np.ndarray, obstacles: np.ndarray, region: str) -> tuple[np.ndarray, str, object]:
        if region not in REGIONS:
            raise ValueError(f"unknown expert region: {region}")
        if region in self._region_seed_offsets:
            offset = self._region_seed_offsets[region]
            sequence = self._sequence_for_offset(state, obstacles, offset)
            projection = self.safety.project(sequence, state, obstacles)
            if projection.feasible:
                self.last_region = region
                self.last_offset = offset
                return projection.actions, region, projection
        selected = None
        for _, _, _, offset, sequence in self._ranked_grid(state, obstacles, region):
            projection = self.safety.project(sequence, state, obstacles)
            selected = (projection.actions, region, projection)
            if projection.feasible:
                self.last_region = region
                self.last_offset = offset
                self._region_seed_offsets[region] = offset
                return projection.actions, region, projection
        assert selected is not None
        return selected

    def plan(self, state: np.ndarray, obstacles: np.ndarray) -> tuple[np.ndarray, str, object]:
        vertical_region = self._dynamic_vertical_region(state, obstacles)
        if vertical_region is not None:
            # Do not switch to a lateral/opposite-height seed while the robot
            # still overlaps a dynamic sphere's longitudinal plane. If this
            # mode is temporarily infeasible, the policy must retreat using a
            # same-mode emergency sequence rather than lower an arm into the
            # sphere's return path.
            selected = self.plan_region(state, obstacles, vertical_region)
            if selected[2].feasible:
                return selected
            # The longitudinal trigger is intentionally conservative. If a
            # vertical IK seed is unavailable while the sphere is already
            # laterally separated, search all regions before stopping.
            self._global_seed_initialized = False
        # A vertical escape is only held while a dynamic sphere overlaps the
        # carried body's longitudinal plane. Re-enter candidate search after
        # that plane is passed so the nominal seed can restore both arms.
        if self.last_region in {"up", "down"} and self.last_offset != (0.0, 0.0):
            self._global_seed_initialized = False
        if self._global_seed_initialized:
            sequence = self._sequence_for_offset(state, obstacles, self.last_offset)
            projection = self.safety.project(sequence, state, obstacles)
            if projection.feasible:
                return projection.actions, self.last_region, projection
        selected = None
        selected_offset = (0.0, 0.0)
        # Raw FK clearance ranks all 25 seeds cheaply. Sparse convexification
        # is reserved for the leading candidates; running OSQP on every known
        # unsafe seed dominates collection time without adding coverage.
        for _, _, region, offset, sequence in self._ranked_grid(state, obstacles)[:6]:
            projection = self.safety.project(sequence, state, obstacles)
            selected = (projection.actions, region, projection)
            selected_offset = offset
            if projection.feasible:
                break
        assert selected is not None
        self.last_region = selected[1]
        self.last_offset = selected_offset
        if selected[2].feasible:
            self._global_seed_initialized = True
        return selected


def expert_condition(state: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    result = np.concatenate((np.asarray(state, dtype=np.float32).reshape(60), np.asarray(obstacles, dtype=np.float32).reshape(-1)))
    if result.shape != (CONDITION_DIMENSION,):
        raise ValueError(f"expected condition shape {(CONDITION_DIMENSION,)}, got {result.shape}")
    return result
