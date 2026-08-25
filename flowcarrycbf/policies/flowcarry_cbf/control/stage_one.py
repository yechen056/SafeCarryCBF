"""Deterministic static-obstacle carry controller for V2 phase one.

This controller is intentionally independent of FM, RGB-D tracking, and the
dynamic-obstacle safety policy.  It provides the smallest useful expert gate:
keep the calibrated bimanual grasp unchanged and drive through a conservative
known corridor around the two fixed cubes.
"""

from __future__ import annotations

import math

import numpy as np


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


class StageOneStaticExpert:
    """Waypoint controller for the static-only fixed layout.

    Waypoints are expressed in world coordinates and anchored to the episode's
    initial base pose.  The lower corridor is deliberately wide: the robot
    leaves the cube's footprint before translating along X, then returns to the
    goal line only after the payload and base have cleared the obstacle slice.
    """

    ACTION_DIMENSION = 17
    MAX_LINEAR_VELOCITY = 0.25
    MAX_ANGULAR_VELOCITY = 0.60
    MAX_LINEAR_ACCELERATION = 0.15
    WAYPOINT_TOLERANCE = 0.08

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._waypoints: list[np.ndarray] = []
        self._waypoint_index = 0
        self._initialized = False

    @property
    def waypoint_index(self) -> int:
        return self._waypoint_index

    def _initialize(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(60)
        start = state[:3].copy()
        # The static cubes occupy y=-1.075..-0.625 on the lower side.  A
        # centerline of -1.75 m leaves a deliberate margin for the calibrated
        # base collision spheres while the robot crosses the complete X slice.
        side_y = float(start[1] - 1.75)
        self._waypoints = [
            np.asarray([start[0], side_y, 0.0], dtype=np.float32),
            np.asarray([-2.70, side_y, 0.0], dtype=np.float32),
            np.asarray([-3.00, side_y, 0.0], dtype=np.float32),
            np.asarray([-3.00, 0.0, 0.0], dtype=np.float32),
            np.asarray([-3.00, 0.0, math.pi], dtype=np.float32),
        ]
        self._waypoint_index = 0
        self._initialized = True

    def act(self, state: np.ndarray) -> tuple[np.ndarray, str]:
        state = np.asarray(state, dtype=np.float32).reshape(60)
        if not self._initialized:
            self._initialize(state)

        action = np.zeros(self.ACTION_DIMENSION, dtype=np.float32)
        if self._waypoint_index >= len(self._waypoints):
            return action, "stage1_complete"

        target = self._waypoints[self._waypoint_index]
        base = state[:3]
        delta = target[:2] - base[:2]
        distance = float(np.linalg.norm(delta))
        yaw_error = _wrap_angle(float(target[2]) - float(base[2]))
        if distance <= self.WAYPOINT_TOLERANCE and abs(yaw_error) <= math.radians(2.0):
            self._waypoint_index += 1
            if self._waypoint_index >= len(self._waypoints):
                return action, "stage1_complete"
            target = self._waypoints[self._waypoint_index]
            delta = target[:2] - base[:2]
            distance = float(np.linalg.norm(delta))
            yaw_error = _wrap_angle(float(target[2]) - float(base[2]))

        if distance > self.WAYPOINT_TOLERANCE:
            speed = min(
                self.MAX_LINEAR_VELOCITY,
                math.sqrt(2.0 * self.MAX_LINEAR_ACCELERATION * distance),
            )
            c, s = math.cos(float(base[2])), math.sin(float(base[2]))
            local_delta = np.asarray(
                [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]],
                dtype=np.float32,
            )
            action[:2] = (speed / self.MAX_LINEAR_VELOCITY) * local_delta / max(distance, 1.0e-6)
        else:
            action[2] = np.clip(
                yaw_error / self.MAX_ANGULAR_VELOCITY,
                -1.0,
                1.0,
            )
        # Arm deltas remain zero: the reset IK establishes the calibrated
        # bimanual handle contacts and the task keeps both grippers closed.
        return np.clip(action, -1.0, 1.0), f"stage1_waypoint_{self._waypoint_index}"


class StageTwoDynamicExpert:
    """Forward-only, current-track whole-body controller for phase two."""

    ACTION_DIMENSION = 17
    HORIZON = 10
    CERTIFIED_STEPS = 10
    TARGET_X = -3.0
    TARGET_Y = 0.0
    TARGET_YAW = math.pi
    LATERAL_TARGETS = (-0.10, 0.0, 0.10)
    REGIONS = ("nominal", "up", "down", "left", "right")
    ARM_READY_TOLERANCE = 0.08
    LINEAR_STOP_THRESHOLD = 0.03
    ANGULAR_STOP_THRESHOLD = 0.04
    YAW_READY_THRESHOLD = math.radians(1.5)
    MAX_FORWARD_SPEED = 0.25
    MAX_ANGULAR_SPEED = 0.60
    MAX_LINEAR_ACCELERATION = 0.12
    MAX_ANGULAR_ACCELERATION = 0.40
    PASS_BEHIND_X = -0.55
    SWITCH_CLEARANCE_MARGIN = 0.03
    SWITCH_CONFIRM_FRAMES = 3
    MINIMUM_REGION_DWELL_FRAMES = 20
    ARM_SETTLE_TIME_CONSTANT = 12.0
    CUBE_SLICE_MIN_X = -2.425
    CUBE_SLICE_MAX_X = -0.975
    CUBE_SLICE_Y_LIMIT = 0.10
    OUTSIDE_Y_LIMIT = 0.30

    def __init__(self, arm_targets, kinematics, safety) -> None:
        missing = set(self.REGIONS) - set(arm_targets)
        if missing:
            raise ValueError(f"stage-two arm targets missing regions: {sorted(missing)}")
        self.arm_targets = {
            region: (
                np.asarray(arm_targets[region][0], dtype=np.float32).reshape(7),
                np.asarray(arm_targets[region][1], dtype=np.float32).reshape(7),
            )
            for region in self.REGIONS
        }
        self.kinematics = kinematics
        self.safety = safety
        self.max_arm_delta = float(kinematics.max_arm_delta)
        self.reset()

    def reset(self) -> None:
        self.phase = "ROTATE"
        self._episode_arm_targets = None
        self._active_region = "nominal"
        self._desired_region = "nominal"
        self._lateral_target = 0.0
        self._desired_lateral = 0.0
        self._challenger = None
        self._challenger_frames = 0
        self._control_frame = 0
        self._region_activation_frame = 0
        self._seen_dynamic_slots: set[int] = set()
        self._passed_dynamic_slots: set[int] = set()
        self._certified_tail = None
        self._restored = False
        self._preferred_region = "nominal"
        self._emergency_stops = 0
        self.last_metrics = {}

    def _capture_episode_arm_targets(self, state: np.ndarray) -> None:
        """Anchor all stage-two targets to the grasp that PhysX actually settled."""
        if self._episode_arm_targets is not None:
            return
        nominal_left = np.asarray(state[8:15], dtype=np.float32).copy()
        nominal_right = np.asarray(state[15:22], dtype=np.float32).copy()
        canonical_left, canonical_right = self.arm_targets["nominal"]
        self._episode_arm_targets = {
            region: (
                nominal_left + targets[0] - canonical_left,
                nominal_right + targets[1] - canonical_right,
            )
            for region, targets in self.arm_targets.items()
        }

    def _arm_target(self, region: str) -> tuple[np.ndarray, np.ndarray]:
        targets = self._episode_arm_targets or self.arm_targets
        return targets[region]

    @property
    def active_region(self) -> str:
        return self._active_region

    @property
    def lateral_target(self) -> float:
        return self._lateral_target

    @property
    def all_dynamic_passed(self) -> bool:
        return (
            len(self._seen_dynamic_slots) >= 2
            and self._seen_dynamic_slots.issubset(self._passed_dynamic_slots)
        )

    @classmethod
    def candidate_keys(cls) -> tuple[tuple[float, str], ...]:
        return tuple((lateral, region) for lateral in cls.LATERAL_TARGETS for region in cls.REGIONS)

    @staticmethod
    def _dynamic_slots(obstacles: np.ndarray) -> list[int]:
        value = np.asarray(obstacles, dtype=np.float32).reshape(-1, 15)
        return [
            index
            for index, obstacle in enumerate(value)
            if obstacle[14] > 0.5 and obstacle[12] < 0.5
        ]

    def _update_dynamic_progress(
        self,
        state: np.ndarray,
        observed_obstacles: np.ndarray,
        missing_times: np.ndarray,
    ) -> None:
        yaw_ready = abs(_wrap_angle(self.TARGET_YAW - float(state[2]))) <= math.radians(15.0)
        if not yaw_ready:
            return
        for slot in self._dynamic_slots(observed_obstacles):
            if slot >= len(missing_times) or float(missing_times[slot]) > 0.2:
                continue
            self._seen_dynamic_slots.add(slot)
            if float(observed_obstacles[slot, 0]) <= self.PASS_BEHIND_X:
                self._passed_dynamic_slots.add(slot)

    def _tracks_ready(self, observed_obstacles: np.ndarray, missing_times: np.ndarray) -> bool:
        slots = self._dynamic_slots(observed_obstacles)
        return len(slots) >= 2 and all(
            slot < len(missing_times) and float(missing_times[slot]) <= 0.2
            for slot in slots[:2]
        )

    def _unpassed_track_age(self, missing_times: np.ndarray, fallback: float) -> float:
        relevant = self._seen_dynamic_slots - self._passed_dynamic_slots
        values = [
            float(missing_times[slot])
            for slot in relevant
            if slot < len(missing_times)
        ]
        return max(values, default=float(fallback) if relevant else 0.0)

    def _update_preferred_region(
        self,
        observed_obstacles: np.ndarray,
        missing_times: np.ndarray,
        payload_height: float,
    ) -> None:
        candidates = []
        for slot in self._dynamic_slots(observed_obstacles):
            if slot in self._passed_dynamic_slots:
                continue
            if slot >= len(missing_times) or float(missing_times[slot]) > 0.2:
                continue
            obstacle = observed_obstacles[slot]
            if float(obstacle[0]) <= self.PASS_BEHIND_X:
                continue
            candidates.append((float(obstacle[0]), float(obstacle[2])))
        if candidates:
            _, height = min(candidates)
            self._preferred_region = "up" if height <= float(payload_height) else "down"

    @staticmethod
    def _base_stopped(state: np.ndarray) -> bool:
        return (
            float(np.linalg.norm(state[3:5])) < StageTwoDynamicExpert.LINEAR_STOP_THRESHOLD
            and abs(float(state[5])) < StageTwoDynamicExpert.ANGULAR_STOP_THRESHOLD
        )

    def _arm_ready(self, state: np.ndarray, region: str) -> bool:
        left_target, right_target = self._arm_target(region)
        return max(
            float(np.max(np.abs(left_target - state[8:15]))),
            float(np.max(np.abs(right_target - state[15:22]))),
        ) <= self.ARM_READY_TOLERANCE

    def _arm_sequence(self, state: np.ndarray, region: str) -> np.ndarray:
        result = np.zeros((self.HORIZON, 14), dtype=np.float32)
        left = np.asarray(state[8:15], dtype=np.float32).copy()
        right = np.asarray(state[15:22], dtype=np.float32).copy()
        left_target, right_target = self._arm_target(region)
        limit = float(self.kinematics.arm_action_limit)
        settle_steps = self.ARM_SETTLE_TIME_CONSTANT
        for step in range(self.HORIZON):
            left_action = np.clip(
                (left_target - left) / (settle_steps * self.max_arm_delta),
                -limit,
                limit,
            )
            right_action = np.clip(
                (right_target - right) / (settle_steps * self.max_arm_delta),
                -limit,
                limit,
            )
            result[step] = np.concatenate((left_action, right_action))
            left += left_action * self.max_arm_delta
            right += right_action * self.max_arm_delta
        return result

    def _base_command(self, state: np.ndarray, lateral_target: float) -> np.ndarray:
        delta = np.asarray(
            [self.TARGET_X - float(state[0]), lateral_target - float(state[1])],
            dtype=np.float32,
        )
        distance = float(np.linalg.norm(delta))
        result = np.zeros(3, dtype=np.float32)
        if distance > 1.0e-6:
            speed = min(
                self.MAX_FORWARD_SPEED,
                math.sqrt(max(0.0, 2.0 * self.MAX_LINEAR_ACCELERATION * distance)),
            )
            world_velocity = speed * delta / distance
            c, s = math.cos(float(state[2])), math.sin(float(state[2]))
            local_velocity = np.asarray(
                [c * world_velocity[0] + s * world_velocity[1], -s * world_velocity[0] + c * world_velocity[1]],
                dtype=np.float32,
            )
            result[0] = max(0.0, float(local_velocity[0]) / float(self.kinematics.base_limits[0]))
            result[1] = float(local_velocity[1]) / float(self.kinematics.base_limits[1])
        yaw_error = _wrap_angle(self.TARGET_YAW - float(state[2]))
        result[2] = np.clip(1.5 * yaw_error / float(self.kinematics.base_limits[2]), -0.5, 0.5)
        return np.clip(result, -1.0, 1.0)

    def _candidate_sequence(self, state: np.ndarray, lateral_target: float, region: str) -> np.ndarray:
        sequence = np.zeros((self.HORIZON, self.ACTION_DIMENSION), dtype=np.float32)
        sequence[:, 3:] = self._arm_sequence(state, region)
        arm_transition = region != self._active_region or not self._arm_ready(state, region)
        if self.phase == "TRANSLATE" and not arm_transition:
            sequence[:, :3] = self._base_command(state, lateral_target)
        return sequence

    def _certified_current_hold(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
        previous_action: np.ndarray,
    ):
        sequence = np.zeros((self.HORIZON, self.ACTION_DIMENSION), dtype=np.float32)
        rollout = self.kinematics.rollout(state, sequence[: self.CERTIFIED_STEPS])
        clearance = self.kinematics.minimum_clearance(
            rollout,
            obstacles,
            certified_steps=self.CERTIFIED_STEPS,
        )
        if clearance < -1.0e-4:
            return None
        if not (
            self.kinematics.grasp_consistent(
                rollout,
                reference_vector=self.kinematics.grasp_vector(state),
            )
            and self.kinematics.joint_limits_satisfied(rollout)
            and self.kinematics.self_collision_free(rollout)
            and self._corridor_valid(rollout.base)
        ):
            return None
        projection = self.safety.project(
            sequence,
            state,
            obstacles,
            previous_action=previous_action,
        )
        return projection if projection.feasible else None

    def _certified_slow_current(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
        previous_action: np.ndarray,
    ):
        for scale in (0.50, 0.25):
            sequence = self._candidate_sequence(state, self._lateral_target, self._active_region)
            sequence[:, :3] *= scale
            rollout = self.kinematics.rollout(state, sequence[: self.CERTIFIED_STEPS])
            clearance = self.kinematics.minimum_clearance(
                rollout,
                obstacles,
                certified_steps=self.CERTIFIED_STEPS,
            )
            if clearance < -1.0e-4 or not (
                self.kinematics.grasp_consistent(
                    rollout,
                    reference_vector=self.kinematics.grasp_vector(state),
                )
                and self.kinematics.joint_limits_satisfied(rollout)
                and self.kinematics.self_collision_free(rollout)
                and self._corridor_valid(rollout.base)
            ):
                continue
            projection = self.safety.project(
                sequence,
                state,
                obstacles,
                previous_action=previous_action,
            )
            if projection.feasible and float(projection.actions[0, 0]) >= -1.0e-5:
                return projection
        return None

    @classmethod
    def _corridor_valid(cls, base_path: np.ndarray) -> bool:
        for x, y in np.asarray(base_path, dtype=np.float32)[:, :2]:
            limit = (
                cls.CUBE_SLICE_Y_LIMIT
                if cls.CUBE_SLICE_MIN_X <= float(x) <= cls.CUBE_SLICE_MAX_X
                else cls.OUTSIDE_Y_LIMIT
            )
            if abs(float(y)) > limit + 5.0e-3:
                return False
        return True

    @classmethod
    def _goal_progress(cls, state: np.ndarray, final_base: np.ndarray) -> float:
        initial = float(np.linalg.norm(np.asarray([cls.TARGET_X, cls.TARGET_Y]) - state[:2]))
        final = float(np.linalg.norm(np.asarray([cls.TARGET_X, cls.TARGET_Y]) - final_base[:2]))
        return initial - final

    def _project_candidates(self, state: np.ndarray, obstacles: np.ndarray, previous_action: np.ndarray) -> list[dict]:
        candidates = []
        self._candidate_rejections = {"grasp": 0, "joint": 0, "self_collision": 0, "corridor": 0}
        rollout_cache: dict[bytes, tuple[object, float, bool, bool, bool, bool]] = {}
        grasp_reference = self.kinematics.grasp_vector(state)
        for lateral_target, region in self.candidate_keys():
            nominal = self._candidate_sequence(state, lateral_target, region)
            # During an arm transition all three lateral targets produce the
            # same stopped-base action sequence. Reuse that FK result while
            # retaining all 15 logical candidates and their metrics.
            cache_key = np.ascontiguousarray(nominal[: self.CERTIFIED_STEPS]).tobytes()
            cached = rollout_cache.get(cache_key)
            if cached is None:
                raw_rollout = self.kinematics.rollout(state, nominal[: self.CERTIFIED_STEPS])
                raw_clearance = self.kinematics.minimum_clearance(
                    raw_rollout,
                    obstacles,
                    certified_steps=self.CERTIFIED_STEPS,
                )
                grasp_valid = self.kinematics.grasp_consistent(
                    raw_rollout,
                    reference_vector=grasp_reference,
                )
                joint_valid = self.kinematics.joint_limits_satisfied(raw_rollout)
                self_valid = self.kinematics.self_collision_free(raw_rollout)
                corridor_valid = self._corridor_valid(raw_rollout.base)
                cached = (
                    raw_rollout,
                    float(raw_clearance),
                    grasp_valid,
                    joint_valid,
                    self_valid,
                    corridor_valid,
                )
                rollout_cache[cache_key] = cached
            raw_rollout, raw_clearance, grasp_valid, joint_valid, self_valid, corridor_valid = cached
            if not grasp_valid:
                self._candidate_rejections["grasp"] += 1
            if not joint_valid:
                self._candidate_rejections["joint"] += 1
            if not self_valid:
                self._candidate_rejections["self_collision"] += 1
            if not corridor_valid:
                self._candidate_rejections["corridor"] += 1
            if not (grasp_valid and joint_valid and self_valid and corridor_valid):
                continue
            candidates.append({
                "key": (float(lateral_target), region),
                "lateral": float(lateral_target),
                "region": region,
                "sequence": nominal,
                "projection": None,
                "clearance": float(raw_clearance),
                "raw_clearance": float(raw_clearance),
                "progress": self._goal_progress(state, raw_rollout.base[-1]),
            })
        return candidates

    def _maximum_margin_emergency(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
        previous_action: np.ndarray,
        *,
        arm_regions: tuple[str, ...] | None = None,
    ):
        """Return the safest kinematically valid next action when 1 s is infeasible."""

        seeds: list[tuple[np.ndarray, str]] = []
        stopped = self._base_stopped(state)
        if stopped:
            for region in self.REGIONS if arm_regions is None else arm_regions:
                sequence = np.zeros((self.HORIZON, self.ACTION_DIMENSION), dtype=np.float32)
                sequence[:, 3:] = self._arm_sequence(state, region)
                seeds.append((sequence, region))
        else:
            for forward, lateral in (
                (0.0, 0.0),
                (0.0, -0.45),
                (0.0, 0.45),
                (0.50, -0.45),
                (0.50, 0.0),
                (0.50, 0.45),
                (1.0, -0.45),
                (1.0, 0.0),
                (1.0, 0.45),
            ):
                sequence = np.zeros((self.HORIZON, self.ACTION_DIMENSION), dtype=np.float32)
                sequence[:, :3] = [forward, lateral, self._base_command(state, self._lateral_target)[2]]
                seeds.append((sequence, self._active_region))

        valid: list[tuple[float, np.ndarray, str, object]] = []
        certified: list[tuple[float, np.ndarray, str, object]] = []
        seen: set[bytes] = set()
        grasp_reference = self.kinematics.grasp_vector(state)
        for nominal, region in seeds:
            limited = self.safety._enforce_kinematic_limits(nominal, state, previous_action)
            limited[:, 0] = np.maximum(limited[:, 0], 0.0)
            marker = np.round(limited, 5).tobytes()
            if marker in seen:
                continue
            seen.add(marker)
            rollout = self.kinematics.rollout(state, limited)
            if not (
                self.kinematics.grasp_consistent(rollout, reference_vector=grasp_reference)
                and self.kinematics.joint_limits_satisfied(rollout)
                and self.kinematics.self_collision_free(rollout)
                and self._corridor_valid(rollout.base)
            ):
                continue
            margin = self.kinematics.minimum_clearance(
                rollout, obstacles, certified_steps=self.CERTIFIED_STEPS
            )
            first = limited[0]
            shield = self.safety.shield(first, state, obstacles, previous_action=previous_action)
            item = (float(margin), first, region, shield)
            valid.append(item)
            if shield.feasible:
                projected = np.asarray(shield.actions, dtype=np.float32)
                projected_sequence = limited.copy()
                projected_sequence[0] = projected
                projected_rollout = self.kinematics.rollout(state, projected_sequence)
                projected_margin = self.kinematics.minimum_clearance(
                    projected_rollout, obstacles, certified_steps=self.CERTIFIED_STEPS
                )
                certified.append((float(projected_margin), projected, region, shield))
        choices = certified if certified else valid
        if not choices:
            retained = self._active_region if arm_regions is None else arm_regions[0]
            return np.zeros(self.ACTION_DIMENSION, dtype=np.float32), None, retained
        _, action, region, projection = max(choices, key=lambda item: item[0])
        return np.asarray(action, dtype=np.float32), projection, region

    def _certify_candidate(
        self,
        selected: dict,
        candidates: list[dict],
        state: np.ndarray,
        obstacles: np.ndarray,
        previous_action: np.ndarray,
    ) -> dict | None:
        alternatives = [selected]
        alternatives.extend(
            candidate
            for candidate in sorted(
                candidates,
                key=lambda item: (item["clearance"], item["progress"]),
                reverse=True,
            )
            if candidate["key"] != selected["key"]
        )
        for candidate in alternatives:
            # A negative raw clearance means that the candidate violates a
            # hard reachable-set constraint. Trying several large numerical
            # QPs in that state cannot produce a certified action in time;
            # wait for a raw-feasible mode and use OSQP only as its exact
            # kinematic projection.
            if candidate["clearance"] < -1.0e-4:
                continue
            projection = self.safety.project(
                candidate["sequence"],
                state,
                obstacles,
                previous_action=previous_action,
            )
            if not projection.feasible:
                continue
            projected = np.asarray(projection.actions, dtype=np.float32)
            if np.any(projected[: self.CERTIFIED_STEPS, 0] < -1.0e-5):
                continue
            rollout = self.kinematics.rollout(state, projected[: self.CERTIFIED_STEPS])
            if not self._corridor_valid(rollout.base):
                continue
            result = dict(candidate)
            result["sequence"] = projected
            result["projection"] = projection
            result["clearance"] = float(projection.minimum_clearance)
            result["progress"] = self._goal_progress(state, rollout.base[-1])
            return result
        return None

    def _select_candidate(self, candidates: list[dict]) -> dict | None:
        if not candidates:
            self._challenger = None
            self._challenger_frames = 0
            return None
        current_key = (float(self._lateral_target), self._active_region)
        current = next((candidate for candidate in candidates if candidate["key"] == current_key), None)
        best = max(candidates, key=lambda item: (item["clearance"], item["progress"], -abs(item["lateral"])))
        if current is None:
            self._challenger = None
            self._challenger_frames = 0
            return best
        if current["clearance"] < -1.0e-4:
            self._challenger = None
            self._challenger_frames = 0
            preferred = [
                candidate
                for candidate in candidates
                if candidate["region"] == self._preferred_region
                and candidate["clearance"] >= -1.0e-4
            ]
            return max(
                preferred,
                key=lambda item: (item["clearance"], item["progress"], -abs(item["lateral"])),
            ) if preferred else best
        if self._active_region != "nominal":
            # Once an avoidance posture is physically established, use its
            # safe crossing window instead of spending that window chasing a
            # marginally clearer posture. A truly unsafe hold is handled
            # before this method and may still trigger an immediate switch.
            self._challenger = None
            self._challenger_frames = 0
            return current
        if (
            best["key"] != current_key
            and self._control_frame - self._region_activation_frame < self.MINIMUM_REGION_DWELL_FRAMES
        ):
            self._challenger = None
            self._challenger_frames = 0
            return current
        preferred = [candidate for candidate in candidates if candidate["region"] == self._preferred_region]
        challenger = max(
            preferred,
            key=lambda item: (item["clearance"], item["progress"], -abs(item["lateral"])),
        ) if preferred else best
        if challenger["key"] == current_key or challenger["clearance"] < current["clearance"] + self.SWITCH_CLEARANCE_MARGIN:
            self._challenger = None
            self._challenger_frames = 0
            return current
        if self._challenger == challenger["key"]:
            self._challenger_frames += 1
        else:
            self._challenger = challenger["key"]
            self._challenger_frames = 1
        if self._challenger_frames >= self.SWITCH_CONFIRM_FRAMES:
            self._challenger = None
            self._challenger_frames = 0
            return challenger
        return current

    def _rotation_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(self.ACTION_DIMENSION, dtype=np.float32)
        error = _wrap_angle(self.TARGET_YAW - float(state[2]))
        desired = math.copysign(
            min(
                self.MAX_ANGULAR_SPEED,
                math.sqrt(max(0.0, 2.0 * self.MAX_ANGULAR_ACCELERATION * abs(error))),
            ),
            error,
        )
        action[2] = desired / float(self.kinematics.base_limits[2])
        return action

    def _shield_action(self, action: np.ndarray, state: np.ndarray, obstacles: np.ndarray, previous_action: np.ndarray):
        result = self.safety.shield(
            np.asarray(action, dtype=np.float32),
            state,
            obstacles,
            previous_action=previous_action,
        )
        value = np.asarray(result.actions, dtype=np.float32) if result.feasible else np.zeros(self.ACTION_DIMENSION, dtype=np.float32)
        if value[0] < -1.0e-5:
            value[:] = 0.0
        rollout = self.kinematics.rollout(state, value.reshape(1, -1))
        if not self._corridor_valid(rollout.base):
            value[:] = 0.0
        return value, result

    def _finish(self, action: np.ndarray, status: str, *, candidate_count: int = 0, feasible_count: int = 0, projection=None, new_plan: bool = False) -> tuple[np.ndarray, str]:
        self.last_metrics = {
            "stage2_phase": self.phase,
            "selected_lateral_target": float(self._lateral_target),
            "selected_arm_region": self._active_region,
            "desired_arm_region": self._desired_region,
            "arm_target_ready": int(self._arm_ready(self._last_state, self._active_region)),
            "candidate_count": int(candidate_count),
            "feasible_candidate_count": int(feasible_count),
            "minimum_clearance": float(getattr(projection, "minimum_clearance", float("inf"))),
            "filter_latency_ms": float(getattr(projection, "latency_ms", 0.0)),
            "cbf_intervention": float(getattr(projection, "intervention_rms", 0.0)),
            "emergency_stops": int(self._emergency_stops),
            "new_plan": int(new_plan),
            "dynamic_tracks_seen": int(len(self._seen_dynamic_slots)),
            "dynamic_tracks_passed": int(len(self._passed_dynamic_slots)),
            "candidate_rejections": dict(getattr(self, "_candidate_rejections", {})),
        }
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0), status

    def act(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
        *,
        observed_obstacles: np.ndarray | None = None,
        tracks_fresh: bool = True,
        dynamic_missing_time: float = 0.0,
        track_missing_times: np.ndarray | None = None,
        previous_action: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str]:
        state = np.asarray(state, dtype=np.float32).reshape(60)
        self._control_frame += 1
        self._last_state = state
        self._capture_episode_arm_targets(state)
        obstacles = np.asarray(obstacles, dtype=np.float32).reshape(-1, 15)
        observed = obstacles if observed_obstacles is None else np.asarray(observed_obstacles, dtype=np.float32).reshape(-1, 15)
        missing_times = np.asarray(
            [] if track_missing_times is None else track_missing_times,
            dtype=np.float32,
        ).reshape(-1)
        previous = np.zeros(self.ACTION_DIMENSION, dtype=np.float32) if previous_action is None else np.asarray(previous_action, dtype=np.float32).reshape(self.ACTION_DIMENSION)
        self._update_dynamic_progress(state, observed, missing_times)
        self._update_preferred_region(observed, missing_times, float(state[42]))

        if self.phase == "ROTATE":
            error = abs(_wrap_angle(self.TARGET_YAW - float(state[2])))
            if error <= self.YAW_READY_THRESHOLD and abs(float(state[5])) < self.ANGULAR_STOP_THRESHOLD:
                self.phase = "ACQUIRE"
                return self._finish(np.zeros(self.ACTION_DIMENSION, dtype=np.float32), "stage2_acquire")
            action, shield = self._shield_action(self._rotation_action(state), state, obstacles, previous)
            action[:2] = 0.0
            return self._finish(action, "stage2_rotate", projection=shield)

        if self.phase == "ACQUIRE":
            if tracks_fresh and self._tracks_ready(observed, missing_times):
                self.phase = "TRANSLATE"
            else:
                return self._finish(np.zeros(self.ACTION_DIMENSION, dtype=np.float32), "stage2_acquire")

        relevant_missing = self._unpassed_track_age(missing_times, dynamic_missing_time)
        if not self.all_dynamic_passed and (not tracks_fresh or relevant_missing > 0.2):
            if relevant_missing <= 0.5 and self._certified_tail is not None:
                shifted = np.vstack((self._certified_tail[1:], self._certified_tail[-1:]))
                recertified = self.safety.project(shifted, state, obstacles, previous_action=previous)
                if recertified.feasible and np.all(np.asarray(recertified.actions)[: self.CERTIFIED_STEPS, 0] >= -1.0e-5):
                    self._certified_tail = np.asarray(recertified.actions, dtype=np.float32)
                    action, shield = self._shield_action(self._certified_tail[0], state, obstacles, previous)
                    return self._finish(action, "stage2_recertified_track_tail", projection=shield)
            self._emergency_stops += 1
            return self._finish(np.zeros(self.ACTION_DIMENSION, dtype=np.float32), "stage2_track_lost_stop")

        if self.all_dynamic_passed and not self._restored:
            self._desired_region = "nominal"
            self._desired_lateral = 0.0
            if self._active_region == "nominal" and self._arm_ready(state, "nominal"):
                self._restored = True
                self._lateral_target = 0.0
            else:
                self.phase = "RESTORE"

        if self.phase == "RESTORE":
            if not self._base_stopped(state):
                return self._finish(np.zeros(self.ACTION_DIMENSION, dtype=np.float32), "stage2_restore_brake")
            if self._arm_ready(state, "nominal"):
                self._active_region = "nominal"
                self._desired_region = "nominal"
                self._lateral_target = 0.0
                self._restored = True
                self.phase = "TRANSLATE"
            else:
                sequence = self._candidate_sequence(state, 0.0, "nominal")
                raw_rollout = self.kinematics.rollout(state, sequence[: self.CERTIFIED_STEPS])
                raw_clearance = self.kinematics.minimum_clearance(
                    raw_rollout,
                    obstacles,
                    certified_steps=self.CERTIFIED_STEPS,
                )
                projection = (
                    self.safety.project(sequence, state, obstacles, previous_action=previous)
                    if raw_clearance >= -1.0e-4
                    else None
                )
                if projection is None or not projection.feasible:
                    self._emergency_stops += 1
                    return self._finish(np.zeros(self.ACTION_DIMENSION, dtype=np.float32), "stage2_restore_uncertified", projection=projection)
                action, shield = self._shield_action(projection.actions[0], state, obstacles, previous)
                return self._finish(action, "stage2_restore", projection=shield)

        goal_distance = float(np.linalg.norm(state[53:55]))
        yaw_error = abs(float(state[55]))
        if goal_distance <= 0.08 and yaw_error <= math.radians(2.0):
            self.phase = "HOLD"
        if self.phase == "HOLD":
            return self._finish(np.zeros(self.ACTION_DIMENSION, dtype=np.float32), "stage2_hold")

        if self.phase == "ARM_SETTLE":
            if not self._base_stopped(state):
                return self._finish(
                    np.zeros(self.ACTION_DIMENSION, dtype=np.float32),
                    "stage2_arm_brake",
                    candidate_count=15,
                )
            if self._arm_ready(state, self._desired_region):
                self._active_region = self._desired_region
                self._lateral_target = self._desired_lateral
                self._region_activation_frame = self._control_frame
                self._challenger = None
                self._challenger_frames = 0
                self.phase = "TRANSLATE"
            else:
                sequence = self._candidate_sequence(state, self._desired_lateral, self._desired_region)
                raw_rollout = self.kinematics.rollout(state, sequence[: self.CERTIFIED_STEPS])
                raw_clearance = self.kinematics.minimum_clearance(
                    raw_rollout,
                    obstacles,
                    certified_steps=self.CERTIFIED_STEPS,
                )
                projection = (
                    self.safety.project(sequence, state, obstacles, previous_action=previous)
                    if raw_clearance >= -1.0e-4
                    else None
                )
                if projection is None or not projection.feasible:
                    self._emergency_stops += 1
                    emergency, shield, region = self._maximum_margin_emergency(
                        state,
                        obstacles,
                        previous,
                        arm_regions=(self._desired_region,),
                    )
                    if shield is None or not shield.feasible:
                        # A moving obstacle can invalidate a transition after
                        # it starts, and an intermediate joint state can also
                        # make the remaining direct interpolation invalid.
                        # Re-certify all five targets so the controller can
                        # safely return to its established mode instead of
                        # remaining trapped halfway between two arm poses.
                        fallback, fallback_shield, fallback_region = self._maximum_margin_emergency(
                            state,
                            obstacles,
                            previous,
                            arm_regions=self.REGIONS,
                        )
                        if fallback_shield is not None and fallback_shield.feasible:
                            emergency = fallback
                            shield = fallback_shield
                            region = fallback_region
                    self._desired_region = region
                    status = (
                        "stage2_arm_emergency_certified_step"
                        if shield is not None and shield.feasible
                        else "stage2_arm_emergency_max_margin"
                    )
                    return self._finish(
                        emergency,
                        status,
                        candidate_count=15,
                        feasible_count=0,
                        projection=shield,
                    )
                action, shield = self._shield_action(projection.actions[0], state, obstacles, previous)
                return self._finish(
                    action,
                    "stage2_arm_settle",
                    candidate_count=15,
                    feasible_count=1,
                    projection=shield,
                    new_plan=True,
                )

        candidates = self._project_candidates(state, obstacles, previous)
        raw_feasible_count = sum(candidate["clearance"] >= -1.0e-4 for candidate in candidates)
        current_key = (float(self._lateral_target), self._active_region)
        current = next((candidate for candidate in candidates if candidate["key"] == current_key), None)
        selected_raw = None
        selected = None
        if current is None or current["clearance"] < -1.0e-4:
            # A safe stationary hold must not mask a feasible arm escape.  The
            # phase-two contract explicitly allows an immediate mode change
            # when the current mode is infeasible; ARM_SETTLE will brake the
            # base before either arm starts moving.
            selected_raw = self._select_candidate(candidates)
            if selected_raw is not None and selected_raw["key"] != current_key:
                selected = self._certify_candidate(
                    selected_raw, candidates, state, obstacles, previous
                )
                if selected is not None and str(selected["region"]) != self._active_region:
                    self._desired_region = str(selected["region"])
                    self._desired_lateral = float(selected["lateral"])
                    self.phase = "ARM_SETTLE"
                    return self._finish(
                        np.zeros(self.ACTION_DIMENSION, dtype=np.float32),
                        "stage2_arm_brake",
                        candidate_count=15,
                        feasible_count=raw_feasible_count,
                    )
            slow = None if current is None else self._certified_slow_current(state, obstacles, previous)
            if selected is None and slow is not None:
                self._challenger = None
                self._challenger_frames = 0
                self._certified_tail = np.asarray(slow.actions, dtype=np.float32)
                action, shield = self._shield_action(self._certified_tail[0], state, obstacles, previous)
                return self._finish(
                    action,
                    "stage2_slow_current_mode",
                    candidate_count=15,
                    feasible_count=raw_feasible_count,
                    projection=shield,
                    new_plan=True,
                )
            hold = None if selected is not None else self._certified_current_hold(state, obstacles, previous)
            if hold is not None:
                self._challenger = None
                self._challenger_frames = 0
                self._certified_tail = np.asarray(hold.actions, dtype=np.float32)
                action, shield = self._shield_action(self._certified_tail[0], state, obstacles, previous)
                return self._finish(
                    action,
                    "stage2_wait_current_mode",
                    candidate_count=15,
                    feasible_count=raw_feasible_count,
                    projection=shield,
                    new_plan=True,
                )
        if selected is None:
            selected_raw = self._select_candidate(candidates)
            selected = (
                None
                if selected_raw is None
                else self._certify_candidate(selected_raw, candidates, state, obstacles, previous)
            )
        if selected is None:
            self._emergency_stops += 1
            emergency, shield, region = self._maximum_margin_emergency(
                state, obstacles, previous
            )
            if region != self._active_region and self._base_stopped(state):
                self._desired_region = region
                self._desired_lateral = self._lateral_target
                self.phase = "ARM_SETTLE"
            status = (
                "stage2_emergency_certified_step"
                if shield is not None and shield.feasible
                else "stage2_emergency_max_margin"
            )
            return self._finish(
                emergency,
                status,
                candidate_count=15,
                feasible_count=0,
                projection=shield,
            )

        selected_region = str(selected["region"])
        selected_lateral = float(selected["lateral"])
        if selected_region != self._active_region:
            self._desired_region = selected_region
            self._desired_lateral = selected_lateral
            self.phase = "ARM_SETTLE"
            return self._finish(
                np.zeros(self.ACTION_DIMENSION, dtype=np.float32),
                "stage2_arm_brake",
                candidate_count=15,
                feasible_count=raw_feasible_count,
            )

        self._lateral_target = selected_lateral
        self._desired_lateral = selected_lateral
        self._desired_region = self._active_region
        self._certified_tail = np.asarray(selected["sequence"], dtype=np.float32)
        action, shield = self._shield_action(self._certified_tail[0], state, obstacles, previous)
        return self._finish(
            action,
            "stage2_translate",
            candidate_count=15,
            feasible_count=raw_feasible_count,
            projection=shield,
            new_plan=True,
        )
