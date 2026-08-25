"""Lightweight full-body collision envelope used by V2 rollouts and QP."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pinocchio as pin

from flowcarrycbf.utils.files import get_urdf_path

from .motion import DEFAULT_ACCELERATION_LIMIT


@dataclass(frozen=True)
class BodySphere:
    center: np.ndarray
    radius: float
    name: str


@dataclass(frozen=True)
class BodyCapsule:
    start: np.ndarray
    end: np.ndarray
    radius: float
    name: str


@dataclass(frozen=True)
class BodyOBB:
    center: np.ndarray
    half_extents: np.ndarray
    yaw: float
    name: str


@dataclass(frozen=True)
class Rollout:
    base: np.ndarray
    left_q: np.ndarray
    right_q: np.ndarray
    spheres: tuple[tuple[BodySphere, ...], ...]
    capsules: tuple[tuple[BodyCapsule, ...], ...]
    obbs: tuple[tuple[BodyOBB, ...], ...]


class FullBodyKinematics:
    """Full Pinocchio model with calibrated sphere/capsule collision geometry.

    The production Isaac task supplies the exact articulated contact result;
    this model is the low-latency optimization envelope and includes base,
    torso, both seven-joint chains, grippers, and payload.
    """

    HOLD = np.asarray([0.6093, 0.8440, 0.8962, 1.8250, 0.3477, -0.9432, 0.8746], dtype=np.float32)

    def __init__(self, dt: float = 0.1, max_arm_delta: float = 0.02) -> None:
        self.dt = float(dt)
        self.max_arm_delta = float(max_arm_delta)
        self.payload_half_extents = np.asarray([0.12, 0.18, 0.06], dtype=np.float32)
        self.base_limits = np.asarray([0.25, 0.25, 0.60], dtype=np.float32)
        self.base_acceleration_step = np.asarray([0.015, 0.015, 0.040], dtype=np.float32)
        # Keep target changes smooth for pure PhysX contact while allowing a
        # complete avoidance-region transition on the dynamic-obstacle time
        # scale. This is at most 0.014 rad per 0.1 s control frame.
        self.arm_action_limit = 0.70
        self.arm_acceleration_step = 0.25
        self.arm_lower = np.asarray([-1.1780972, -1.1780972, -0.7853982, -0.3926991, -2.0943952, -1.4137167, -2.0943952], dtype=np.float32)
        self.arm_upper = np.asarray([1.5707963, 1.5707963, 3.9269908, 2.3561945, 2.0943952, 1.4137167, 2.0943952], dtype=np.float32)
        urdf = get_urdf_path() / "tiago_dual_holobase.urdf"
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self._neutral = pin.neutral(self.model)
        self._frame_ids = {
            name: self.model.getFrameId(name)
            for name in (
                "base_footprint", "torso_lift_link",
                *[f"arm_left_{index}_link" for index in range(1, 8)],
                *[f"arm_right_{index}_link" for index in range(1, 8)],
                "gripper_left_grasping_frame", "gripper_right_grasping_frame",
            )
        }
        nominal = self._spheres(np.zeros(3), np.zeros(3), self.HOLD, self.HOLD)
        self._nominal_handle_vector = nominal[-2].center - nominal[-3].center

    def _pin_configuration(self, base: np.ndarray, left_q: np.ndarray, right_q: np.ndarray) -> np.ndarray:
        q = self._neutral.copy()
        q[0], q[1] = float(base[0]), float(base[1])
        q[2], q[3] = math.cos(float(base[2])), math.sin(float(base[2]))
        q[26] = 0.25
        q[27:34] = left_q
        q[36:43] = right_q
        return q

    def _spheres(self, base: np.ndarray, reference_base: np.ndarray, left_q: np.ndarray, right_q: np.ndarray) -> tuple[BodySphere, ...]:
        pin.framesForwardKinematics(self.model, self.data, self._pin_configuration(base, left_q, right_q))
        c0, s0 = math.cos(float(reference_base[2])), math.sin(float(reference_base[2]))
        rotation = np.asarray([[c0, s0, 0.0], [-s0, c0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        origin = np.asarray([reference_base[0], reference_base[1], 0.0], dtype=np.float64)

        def local(name: str) -> np.ndarray:
            world = self.data.oMf[self._frame_ids[name]].translation.copy()
            return (rotation @ (world - origin)).astype(np.float32)

        left_hand = local("gripper_left_grasping_frame")
        right_hand = local("gripper_right_grasping_frame")
        spheres: list[BodySphere] = [
            BodySphere(local("base_footprint") + np.asarray([0.0, 0.0, 0.38]), 0.42, "base"),
            BodySphere(local("torso_lift_link"), 0.24, "torso"),
        ]
        arm_radii = (0.13, 0.12, 0.11, 0.105, 0.10, 0.09, 0.085)
        for side in ("left", "right"):
            for index, radius in enumerate(arm_radii, start=1):
                spheres.append(BodySphere(local(f"arm_{side}_{index}_link"), radius, f"{side}_arm_{index}"))
        spheres.extend((
            BodySphere(left_hand, 0.09, "left_gripper"),
            BodySphere(right_hand, 0.09, "right_gripper"),
            BodySphere(0.5 * (left_hand + right_hand), 0.23, "payload"),
        ))
        return tuple(spheres)

    @staticmethod
    def _capsules(spheres: tuple[BodySphere, ...]) -> tuple[BodyCapsule, ...]:
        by_name = {sphere.name: sphere for sphere in spheres}
        result = [
            BodyCapsule(
                by_name["base"].center + np.asarray([0.0, 0.0, 0.25], dtype=np.float32),
                by_name["torso"].center,
                0.18,
                "body_capsule",
            )
        ]
        for side in ("left", "right"):
            names = [f"{side}_arm_{index}" for index in range(1, 8)] + [f"{side}_gripper"]
            for first_name, second_name in zip(names[:-1], names[1:]):
                first, second = by_name[first_name], by_name[second_name]
                result.append(
                    BodyCapsule(
                        first.center,
                        second.center,
                        min(first.radius, second.radius),
                        f"{first_name}_to_{second_name}",
                    )
                )
        return tuple(result)

    def rollout(self, state: np.ndarray, actions: np.ndarray) -> Rollout:
        state = np.asarray(state, dtype=np.float32).reshape(60)
        actions = np.asarray(actions, dtype=np.float32)
        base = state[:3].copy()
        initial_base = base.copy()
        left_q = state[8:15].copy()
        right_q = state[15:22].copy()
        bases, lefts, rights, spheres, capsules, obbs = [], [], [], [], [], []
        yaw = float(base[2])
        c, s = math.cos(yaw), math.sin(yaw)
        world_velocity = state[3:6]
        command = np.asarray(
            [
                c * world_velocity[0] + s * world_velocity[1],
                -s * world_velocity[0] + c * world_velocity[1],
                world_velocity[2],
            ],
            dtype=np.float32,
        )
        for action in actions:
            desired = action[:3] * self.base_limits
            command += np.clip(desired - command, -self.base_acceleration_step, self.base_acceleration_step)
            yaw = float(base[2])
            c, s = math.cos(yaw), math.sin(yaw)
            base[0] += (c * command[0] - s * command[1]) * self.dt
            base[1] += (s * command[0] + c * command[1]) * self.dt
            base[2] = float((base[2] + command[2] * self.dt + math.pi) % (2.0 * math.pi) - math.pi)
            left_q = left_q + action[3:10] * self.max_arm_delta
            right_q = right_q + action[10:17] * self.max_arm_delta
            bases.append(base.copy()); lefts.append(left_q.copy()); rights.append(right_q.copy())
            step_spheres = self._spheres(base, initial_base, left_q, right_q)
            spheres.append(step_spheres)
            capsules.append(self._capsules(step_spheres))
            obbs.append((
                BodyOBB(
                    step_spheres[-1].center,
                    self.payload_half_extents,
                    float(base[2] - initial_base[2]),
                    "payload_obb",
                ),
            ))
        return Rollout(
            np.asarray(bases), np.asarray(lefts), np.asarray(rights),
            tuple(spheres), tuple(capsules), tuple(obbs),
        )

    @staticmethod
    def obstacle_clearance(point: np.ndarray, radius: float, obstacle: np.ndarray) -> float:
        if obstacle[12] < 0.5:
            return float(np.linalg.norm(np.asarray(point) - obstacle[0:3]) - radius - obstacle[9])
        delta = np.abs(np.asarray(point) - obstacle[0:3]) - obstacle[9:12]
        outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
        inside = min(float(np.max(delta)), 0.0)
        return outside + inside - float(radius)

    def capsule_obstacle_clearance(self, capsule: BodyCapsule, obstacle: np.ndarray) -> float:
        points = np.linspace(capsule.start, capsule.end, 5)
        sampled = min(
            self.obstacle_clearance(point, capsule.radius, obstacle)
            for point in points
        )
        # Signed distance is 1-Lipschitz. Subtracting half the sample spacing
        # makes this a conservative lower bound for the complete segment.
        spacing = float(np.linalg.norm(capsule.end - capsule.start)) / 4.0
        return sampled - 0.5 * spacing

    def obb_obstacle_clearance(self, obb: BodyOBB, obstacle: np.ndarray) -> float:
        c, s = math.cos(obb.yaw), math.sin(obb.yaw)
        rotation = np.asarray([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        local_center = rotation @ (np.asarray(obstacle[:3]) - obb.center)
        if obstacle[12] < 0.5:
            delta = np.abs(local_center) - obb.half_extents
            signed = float(np.linalg.norm(np.maximum(delta, 0.0)) + min(float(np.max(delta)), 0.0))
            return signed - float(obstacle[9])
        if abs(obb.yaw) < 1.0e-5:
            delta = np.abs(local_center) - obb.half_extents - obstacle[9:12]
            return float(np.linalg.norm(np.maximum(delta, 0.0)) + min(float(np.max(delta)), 0.0))
        # The enclosing sphere is conservative for a yawed OBB against an
        # axis-aligned cube in the current base frame.
        return self.obstacle_clearance(obb.center, float(np.linalg.norm(obb.half_extents)), obstacle)

    def step_minimum_clearance(self, rollout: Rollout, index: int, obstacle: np.ndarray) -> float:
        spheres = [sphere for sphere in rollout.spheres[index] if sphere.name != "payload"]
        sphere_centers = np.asarray([sphere.center for sphere in spheres], dtype=np.float64)
        sphere_radii = np.asarray([sphere.radius for sphere in spheres], dtype=np.float64)
        capsules = rollout.capsules[index]
        starts = np.asarray([capsule.start for capsule in capsules], dtype=np.float64)
        ends = np.asarray([capsule.end for capsule in capsules], dtype=np.float64)
        capsule_radii = np.asarray([capsule.radius for capsule in capsules], dtype=np.float64)
        fractions = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        capsule_points = starts[:, None, :] + fractions[None, :, None] * (ends - starts)[:, None, :]
        if obstacle[12] < 0.5:
            sphere_values = (
                np.linalg.norm(sphere_centers - obstacle[:3], axis=1)
                - sphere_radii - float(obstacle[9])
            )
            capsule_samples = (
                np.linalg.norm(capsule_points - obstacle[None, None, :3], axis=2)
                - capsule_radii[:, None] - float(obstacle[9])
            )
        else:
            sphere_delta = np.abs(sphere_centers - obstacle[:3]) - obstacle[9:12]
            sphere_values = (
                np.linalg.norm(np.maximum(sphere_delta, 0.0), axis=1)
                + np.minimum(np.max(sphere_delta, axis=1), 0.0)
                - sphere_radii
            )
            capsule_delta = np.abs(capsule_points - obstacle[None, None, :3]) - obstacle[None, None, 9:12]
            capsule_samples = (
                np.linalg.norm(np.maximum(capsule_delta, 0.0), axis=2)
                + np.minimum(np.max(capsule_delta, axis=2), 0.0)
                - capsule_radii[:, None]
            )
        spacing_correction = 0.125 * np.linalg.norm(ends - starts, axis=1)
        capsule_values = np.min(capsule_samples, axis=1) - spacing_correction
        obb_values = [
            self.obb_obstacle_clearance(obb, obstacle)
            for obb in rollout.obbs[index]
        ]
        return min(
            float(np.min(sphere_values)),
            float(np.min(capsule_values)),
            min(obb_values, default=float("inf")),
        )

    def minimum_clearance(
        self,
        rollout: Rollout,
        obstacles: np.ndarray,
        *,
        certified_steps: int | None = None,
        safety_distance: float = 0.10,
    ) -> float:
        minimum = float("inf")
        limit = len(rollout.spheres) if certified_steps is None else min(len(rollout.spheres), int(certified_steps))
        obstacles = np.asarray(obstacles, dtype=np.float32).reshape(-1, 15)
        for index in range(limit):
            for obstacle in obstacles:
                if obstacle[14] < 0.5:
                    continue
                future_time = (index + 1) * self.dt
                current = self.reachable_obstacle(obstacle, future_time)
                required_uncertainty = float(safety_distance) + float(obstacle[13])
                minimum = min(
                    minimum,
                    self.step_minimum_clearance(rollout, index, current) - required_uncertainty,
                )
        return minimum

    @staticmethod
    def reachable_obstacle(obstacle: np.ndarray, future_time: float) -> np.ndarray:
        """Predict a track and conservatively bound its known lateral motion."""
        current = np.asarray(obstacle, dtype=np.float32).copy()
        future_time = float(future_time)
        current[:3] = (
            obstacle[:3]
            + obstacle[3:6] * future_time
            + 0.5 * obstacle[6:9] * future_time**2
        )
        if obstacle[12] < 0.5:
            radius = float(obstacle[9])
            lateral_reach = 0.5 * DEFAULT_ACCELERATION_LIMIT * future_time**2
            # Stage-two kinematic balls have fixed X/Z and randomly changing
            # world-Y targets. Represent the swept sphere as an AABB so the
            # acceleration bound expands only the physically reachable axis.
            current[9:12] = [radius, radius + lateral_reach, radius]
            current[12] = 1.0
        return current

    def grasp_vector(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(60)
        base = state[:3]
        spheres = self._spheres(base, base, state[8:15], state[15:22])
        return spheres[-2].center - spheres[-3].center

    def grasp_consistent(
        self,
        rollout: Rollout,
        tolerance: float = 0.04,
        reference_vector: np.ndarray | None = None,
    ) -> bool:
        reference = self._nominal_handle_vector if reference_vector is None else np.asarray(reference_vector, dtype=np.float32).reshape(3)
        for spheres in rollout.spheres:
            handle_vector = spheres[-2].center - spheres[-3].center
            if abs(np.linalg.norm(handle_vector) - np.linalg.norm(reference)) > tolerance:
                return False
            if abs(float(handle_vector[2] - reference[2])) > 0.025:
                return False
        return True

    def joint_limits_satisfied(self, rollout: Rollout, tolerance: float = 1.0e-5) -> bool:
        return bool(
            np.all(rollout.left_q >= self.arm_lower - tolerance)
            and np.all(rollout.left_q <= self.arm_upper + tolerance)
            and np.all(rollout.right_q >= self.arm_lower - tolerance)
            and np.all(rollout.right_q <= self.arm_upper + tolerance)
        )

    @staticmethod
    def self_collision_free(rollout: Rollout, margin: float = 0.0) -> bool:
        for spheres in rollout.spheres:
            left = [sphere for sphere in spheres if sphere.name.startswith("left_")]
            right = [sphere for sphere in spheres if sphere.name.startswith("right_")]
            body = [sphere for sphere in spheres if sphere.name in {"base", "torso"}]
            payload = next(sphere for sphere in spheres if sphere.name == "payload")

            def separated(first: BodySphere, second: BodySphere) -> bool:
                return bool(
                    np.linalg.norm(first.center - second.center)
                    >= first.radius + second.radius + margin
                )

            for first in left:
                for second in right:
                    if not separated(first, second):
                        return False
            for full_chain in (left, right):
                chain = full_chain[:-1]
                # Consecutive joint envelopes intentionally overlap. Pairs
                # separated by at least three links must remain disjoint.
                for first_index, first in enumerate(chain):
                    for second in chain[first_index + 3:]:
                        if not separated(first, second):
                            return False
                # The shoulder envelope is attached to the torso; all later
                # links and the gripper must clear both body envelopes.
                for first in full_chain[1:]:
                    for second in body:
                        if not separated(first, second):
                            return False
            for first in (*body, *left[:-1], *right[:-1]):
                if not separated(first, payload):
                    return False
        return True
