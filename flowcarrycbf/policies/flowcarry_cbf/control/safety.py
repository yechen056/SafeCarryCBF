"""Sparse OSQP sequence projection and one-step whole-body shield."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import scipy.sparse as sparse
import osqp

from .kinematics import FullBodyKinematics


@dataclass(frozen=True)
class ActionProjection:
    actions: np.ndarray
    feasible: bool
    status: str
    intervention_rms: float
    minimum_clearance: float
    certified: bool
    latency_ms: float
    clearance_satisfied: bool
    grasp_consistent: bool
    joint_limits_satisfied: bool
    self_collision_free: bool


class WholeBodyCBFQP:
    def __init__(self, kinematics: FullBodyKinematics, *, safety_distance: float = 0.10, certified_steps: int = 10, maximum_rounds: int = 3) -> None:
        self.kinematics = kinematics
        self.safety_distance = float(safety_distance)
        self.certified_steps = int(certified_steps)
        self.maximum_rounds = int(maximum_rounds)

    def _certificate(
        self,
        rollout,
        obstacles: np.ndarray,
        grasp_reference: np.ndarray,
        steps: int,
    ) -> tuple[float, bool, bool, bool, bool]:
        minimum = self.kinematics.minimum_clearance(
            rollout,
            obstacles,
            certified_steps=steps,
            safety_distance=self.safety_distance,
        )
        clearance = bool(minimum >= -1.0e-4)
        grasp = bool(
            self.kinematics.grasp_consistent(
                rollout, reference_vector=grasp_reference,
            )
        )
        joints = bool(self.kinematics.joint_limits_satisfied(rollout))
        self_collision = bool(self.kinematics.self_collision_free(rollout))
        return minimum, clearance, grasp, joints, self_collision

    @staticmethod
    def _projection(
        actions: np.ndarray,
        feasible: bool,
        status: str,
        intervention_rms: float,
        minimum_clearance: float,
        latency_ms: float,
        certificate: tuple[bool, bool, bool, bool],
    ) -> ActionProjection:
        return ActionProjection(
            actions=actions,
            feasible=feasible,
            status=status,
            intervention_rms=intervention_rms,
            minimum_clearance=minimum_clearance,
            certified=feasible,
            latency_ms=latency_ms,
            clearance_satisfied=certificate[0],
            grasp_consistent=certificate[1],
            joint_limits_satisfied=certificate[2],
            self_collision_free=certificate[3],
        )

    def _enforce_kinematic_limits(
        self,
        actions: np.ndarray,
        state: np.ndarray,
        previous_action: np.ndarray | None = None,
    ) -> np.ndarray:
        result = np.asarray(actions, dtype=np.float32).copy()
        left = np.asarray(state, dtype=np.float32)[8:15].copy()
        right = np.asarray(state, dtype=np.float32)[15:22].copy()
        previous = np.zeros(17, dtype=np.float32) if previous_action is None else np.asarray(previous_action, dtype=np.float32).reshape(17)
        previous_left = np.clip(previous[3:10], -1.0, 1.0)
        previous_right = np.clip(previous[10:17], -1.0, 1.0)
        for index in range(len(result)):
            arm_step = float(self.kinematics.arm_acceleration_step)
            arm_limit = float(self.kinematics.arm_action_limit)
            result[index, 3:10] = np.clip(result[index, 3:10], previous_left - arm_step, previous_left + arm_step)
            result[index, 10:17] = np.clip(result[index, 10:17], previous_right - arm_step, previous_right + arm_step)
            result[index, 3:10] = np.clip(result[index, 3:10], -arm_limit, arm_limit)
            result[index, 10:17] = np.clip(result[index, 10:17], -arm_limit, arm_limit)
            result[index, 3:10] = np.clip(result[index, 3:10], (self.kinematics.arm_lower - left) / self.kinematics.max_arm_delta, (self.kinematics.arm_upper - left) / self.kinematics.max_arm_delta)
            result[index, 10:17] = np.clip(result[index, 10:17], (self.kinematics.arm_lower - right) / self.kinematics.max_arm_delta, (self.kinematics.arm_upper - right) / self.kinematics.max_arm_delta)
            left += result[index, 3:10] * self.kinematics.max_arm_delta
            right += result[index, 10:17] * self.kinematics.max_arm_delta
            previous_left = result[index, 3:10].copy(); previous_right = result[index, 10:17].copy()
        return np.clip(result, -1.0, 1.0)

    def _clearance(self, actions: np.ndarray, state: np.ndarray, obstacles: np.ndarray, step: int) -> float:
        rollout = self.kinematics.rollout(state, actions[: step + 1])
        if step >= len(rollout.spheres):
            return float("inf")
        minimum = float("inf")
        for obstacle in np.asarray(obstacles, dtype=np.float32):
            if obstacle[14] < 0.5:
                continue
            future_time = (step + 1) * self.kinematics.dt
            current = self.kinematics.reachable_obstacle(obstacle, future_time)
            required = self.safety_distance + float(obstacle[13])
            minimum = min(
                minimum,
                self.kinematics.step_minimum_clearance(rollout, step, current) - required,
            )
        return minimum

    def project(
        self,
        nominal: np.ndarray,
        state: np.ndarray,
        obstacles: np.ndarray,
        *,
        previous_action: np.ndarray | None = None,
    ) -> ActionProjection:
        started = time.perf_counter()
        nominal = np.asarray(nominal, dtype=np.float32)
        actions = self._enforce_kinematic_limits(nominal, state, previous_action)
        if actions.ndim != 2 or actions.shape[1] != 17:
            raise ValueError("V2 safety projection expects [horizon,17]")
        initial_rollout = self.kinematics.rollout(state, actions)
        grasp_reference = self.kinematics.grasp_vector(state)
        steps = min(self.certified_steps, len(actions))
        initial, clearance, grasp, joints, self_collision = self._certificate(
            initial_rollout, obstacles, grasp_reference, steps,
        )
        if clearance and grasp and joints and self_collision:
            intervention = float(np.sqrt(np.mean(np.square(actions - nominal))))
            return self._projection(
                actions, True,
                "already_safe" if intervention < 1.0e-7 else "limits_applied",
                intervention, initial, 1000.0 * (time.perf_counter() - started),
                (clearance, grasp, joints, self_collision),
            )
        dimension = steps * 17
        for _round in range(self.maximum_rounds):
            rows, lower = [], []
            nominal_rollout = self.kinematics.rollout(state, actions)
            for step in range(steps):
                value = self._clearance(actions, state, obstacles, step)
                if not np.isfinite(value) or value >= -1.0e-4:
                    continue
                gradient = np.zeros(dimension, dtype=np.float64)
                active_components = range(17)
                for control in (step,):
                    for component in active_components:
                        index = control * 17 + component
                        perturbed = actions.copy(); perturbed[control, component] += 1.0e-3
                        gradient[index] = (self._clearance(perturbed, state, obstacles, step) - value) / 1.0e-3
                if np.linalg.norm(gradient) < 1.0e-8:
                    continue
                rows.append(gradient); lower.append(-value)
            if not rows:
                break
            matrix_rows = list(rows)
            lower_bounds = list(lower)
            for index in range(dimension):
                row = np.zeros(dimension); row[index] = 1.0; matrix_rows.append(row); lower_bounds.append(-1.0 - float(actions[index // 17, index % 17]))
                matrix_rows.append(-row); lower_bounds.append(float(actions[index // 17, index % 17]) - 1.0)
            problem = osqp.OSQP()
            problem.setup(P=sparse.eye(dimension, format="csc"), q=np.zeros(dimension), A=sparse.csc_matrix(np.asarray(matrix_rows)), l=np.asarray(lower_bounds), u=np.full(len(matrix_rows), np.inf), verbose=False, eps_abs=1.0e-4, eps_rel=1.0e-4, max_iter=400)
            result = problem.solve()
            if result.x is None or result.info.status_val not in (1, 2):
                minimum, clearance, grasp, joints, self_collision = self._certificate(
                    nominal_rollout, obstacles, grasp_reference, steps,
                )
                return self._projection(
                    actions, False, "infeasible",
                    float(np.sqrt(np.mean(np.square(actions - nominal)))),
                    minimum, 1000.0 * (time.perf_counter() - started),
                    (clearance, grasp, joints, self_collision),
                )
            # OSQP only optimizes the hard-certified prefix.  Preserve the
            # remaining horizon for candidate ranking and receding-horizon
            # replanning instead of attempting to broadcast a prefix update
            # over all 32 actions.
            actions[:steps] += np.asarray(result.x, dtype=np.float32).reshape(steps, 17)
            actions = self._enforce_kinematic_limits(actions, state, previous_action)
            if self._clearance(actions, state, obstacles, steps - 1) >= -1.0e-4:
                break
        rollout = self.kinematics.rollout(state, actions)
        minimum, clearance, grasp, joints, self_collision = self._certificate(
            rollout, obstacles, grasp_reference, steps,
        )
        feasible = bool(clearance and grasp and joints and self_collision)
        return self._projection(
            actions, feasible, "solved" if feasible else "uncertified",
            float(np.sqrt(np.mean(np.square(actions - nominal)))),
            minimum, 1000.0 * (time.perf_counter() - started),
            (clearance, grasp, joints, self_collision),
        )

    def shield(
        self,
        action: np.ndarray,
        state: np.ndarray,
        obstacles: np.ndarray,
        *,
        previous_action: np.ndarray | None = None,
    ) -> ActionProjection:
        # The execution shield certifies only the action that will be applied
        # now.  Future steps are certified by ``project`` on the full planned
        # sequence; repeating the first action here can reject a valid
        # bimanual motion merely because that action is not safe for ten
        # consecutive control cycles.
        sequence = np.asarray(action, dtype=np.float32).reshape(1, 17)
        result = self.project(sequence, state, obstacles, previous_action=previous_action)
        return ActionProjection(
            actions=result.actions[0],
            feasible=result.feasible,
            status=result.status,
            intervention_rms=result.intervention_rms,
            minimum_clearance=result.minimum_clearance,
            certified=result.certified,
            latency_ms=result.latency_ms,
            clearance_satisfied=result.clearance_satisfied,
            grasp_consistent=result.grasp_consistent,
            joint_limits_satisfied=result.joint_limits_satisfied,
            self_collision_free=result.self_collision_free,
        )
