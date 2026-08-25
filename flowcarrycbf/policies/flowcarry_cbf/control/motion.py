"""Seeded kinematic obstacle motion processes."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np


DEFAULT_ACCELERATION_LIMIT = 0.15


@dataclass(frozen=True)
class MotionSample:
    position: float
    velocity: float
    acceleration: float
    target: float
    segment: int


@dataclass(frozen=True)
class VectorMotionSample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    target: np.ndarray
    segment: int


class ConstantSpeedWaypointMotion3D:
    """Seeded 3-D waypoint motion with constant Euclidean speed.

    Waypoints are sampled inside the supplied axis-aligned bounds. Reaching a
    waypoint immediately selects the next one, so the obstacle never waits or
    slows in response to the robot.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        initial: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        speed: float = 0.30,
        minimum_target_distance: float = 0.15,
    ) -> None:
        self.rng = rng
        self.lower = np.asarray(lower, dtype=np.float64).reshape(3)
        self.upper = np.asarray(upper, dtype=np.float64).reshape(3)
        self.position = np.asarray(initial, dtype=np.float64).reshape(3).copy()
        self.speed = float(speed)
        self.minimum_target_distance = float(minimum_target_distance)
        if self.speed <= 0.0:
            raise ValueError("speed must be positive")
        if np.any(self.upper <= self.lower):
            raise ValueError("every upper bound must exceed its lower bound")
        if np.any(self.position < self.lower) or np.any(self.position > self.upper):
            raise ValueError("initial position must be inside the motion bounds")
        if np.linalg.norm(self.upper - self.lower) < self.minimum_target_distance:
            raise ValueError("motion bounds are smaller than the target separation")
        self._segment = 0
        self.target = self.position.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)
        self._start_segment()

    def _sample_target(self) -> np.ndarray:
        for _ in range(200):
            target = self.rng.uniform(self.lower, self.upper)
            if np.linalg.norm(target - self.position) >= self.minimum_target_distance:
                return target
        corners = np.asarray([
            [x, y, z]
            for x in (self.lower[0], self.upper[0])
            for y in (self.lower[1], self.upper[1])
            for z in (self.lower[2], self.upper[2])
        ], dtype=np.float64)
        distances = np.linalg.norm(corners - self.position, axis=1)
        return corners[int(np.argmax(distances))]

    def _start_segment(self) -> None:
        self.target = np.asarray(self._sample_target(), dtype=np.float64)
        direction = self.target - self.position
        self.velocity = self.speed * direction / np.linalg.norm(direction)
        self._segment += 1

    def step(self, dt: float) -> VectorMotionSample:
        remaining_distance = self.speed * float(dt)
        if remaining_distance < 0.0:
            raise ValueError("dt must be non-negative")
        previous_velocity = self.velocity.copy()
        while remaining_distance > 1.0e-12:
            delta = self.target - self.position
            distance = float(np.linalg.norm(delta))
            if distance <= remaining_distance + 1.0e-12:
                self.position = self.target.copy()
                remaining_distance -= distance
                self._start_segment()
                continue
            self.position += delta * (remaining_distance / distance)
            self.velocity = self.speed * delta / distance
            remaining_distance = 0.0
        self.acceleration = (
            (self.velocity - previous_velocity) / float(dt)
            if dt > 0.0
            else np.zeros(3, dtype=np.float64)
        )
        return VectorMotionSample(
            self.position.copy(),
            self.velocity.copy(),
            self.acceleration.copy(),
            self.target.copy(),
            self._segment,
        )


class RandomWaypointMotion:
    """One-dimensional random waypoint process connected by quintic segments.

    A segment may be interrupted after a randomized fraction of its duration.
    Position, velocity, and acceleration remain continuous because each new
    quintic starts from the current state.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        initial: float | None = None,
        lower: float = -0.90,
        upper: float = 0.90,
        speed_range: tuple[float, float] = (0.10, 0.30),
        acceleration_limit: float = DEFAULT_ACCELERATION_LIMIT,
        dwell_range: tuple[float, float] = (0.0, 0.25),
        minimum_target_distance: float = 0.35,
        interrupt_probability_per_second: float = 0.20,
    ) -> None:
        if upper - lower < minimum_target_distance:
            raise ValueError("waypoint interval is smaller than the target separation")
        self.rng = rng
        self.lower = float(lower)
        self.upper = float(upper)
        self.speed_range = tuple(float(x) for x in speed_range)
        self.acceleration_limit = float(acceleration_limit)
        self.dwell_range = tuple(float(x) for x in dwell_range)
        self.minimum_target_distance = float(minimum_target_distance)
        self.interrupt_probability_per_second = float(interrupt_probability_per_second)
        self.position = float(rng.uniform(lower, upper) if initial is None else initial)
        self.velocity = 0.0
        self.acceleration = 0.0
        self.target = self.position
        self._coefficients = np.zeros(6, dtype=np.float64)
        self._elapsed = 0.0
        self._duration = 0.0
        self._dwell = 0.0
        self._segment = 0
        if not self._start_segment():
            raise RuntimeError("could not initialize random waypoint motion")

    def _sample_target(self, retarget_guard: tuple[float, float] | None = None) -> float | None:
        if retarget_guard is not None:
            center, radius = map(float, retarget_guard)
            relative = self.position - center
            direction = np.sign(relative)
            if abs(relative) < 1.0e-6:
                direction = np.sign(self.velocity)
            if direction == 0.0:
                direction = -1.0 if self.rng.random() < 0.5 else 1.0
            if direction < 0.0:
                guarded_lower, guarded_upper = self.lower, min(self.upper, center - radius)
            else:
                guarded_lower, guarded_upper = max(self.lower, center + radius), self.upper
            if guarded_lower <= guarded_upper:
                for _ in range(100):
                    target = float(self.rng.uniform(guarded_lower, guarded_upper))
                    if abs(target - self.position) >= self.minimum_target_distance:
                        return target
                endpoint = guarded_lower if direction < 0.0 else guarded_upper
                if abs(endpoint - self.position) >= self.minimum_target_distance:
                    return float(endpoint)
                # Near a workspace edge the remaining outward distance can be
                # smaller than the normal 0.35 m segment separation. Continue
                # to the edge instead of freezing forever inside the robot's
                # certified crossing envelope.
                outward_delta = (endpoint - self.position) * direction
                if outward_delta > 1.0e-4:
                    return float(endpoint)
            return None
        for _ in range(100):
            target = float(self.rng.uniform(self.lower, self.upper))
            if abs(target - self.position) >= self.minimum_target_distance:
                return target
        return self.upper if self.position < 0.5 * (self.lower + self.upper) else self.lower

    @staticmethod
    def _quintic_coefficients(
        p0: float, v0: float, a0: float, p1: float, duration: float
    ) -> np.ndarray:
        t = float(duration)
        c0, c1, c2 = p0, v0, 0.5 * a0
        matrix = np.asarray(
            [[t**3, t**4, t**5], [3*t**2, 4*t**3, 5*t**4], [6*t, 12*t**2, 20*t**3]],
            dtype=np.float64,
        )
        rhs = np.asarray(
            [p1 - c0 - c1 * t - c2 * t**2, -c1 - 2*c2*t, -2*c2],
            dtype=np.float64,
        )
        return np.asarray([c0, c1, c2, *np.linalg.solve(matrix, rhs)], dtype=np.float64)

    def _segment_extrema(self, coefficients: np.ndarray, duration: float) -> tuple[float, float]:
        times = np.linspace(0.0, duration, 129)
        c = coefficients
        velocities = c[1] + 2*c[2]*times + 3*c[3]*times**2 + 4*c[4]*times**3 + 5*c[5]*times**4
        accelerations = 2*c[2] + 6*c[3]*times + 12*c[4]*times**2 + 20*c[5]*times**3
        return float(np.max(np.abs(velocities))), float(np.max(np.abs(accelerations)))

    def _start_segment(self, retarget_guard: tuple[float, float] | None = None) -> bool:
        target = self._sample_target(retarget_guard)
        if target is None:
            return False
        desired_speed = float(self.rng.uniform(*self.speed_range))
        nominal_duration = max(0.75, 1.875 * abs(target - self.position) / desired_speed)
        duration_candidates = np.unique(
            np.concatenate(
                (
                    np.linspace(0.50, 5.0, 80),
                    np.geomspace(max(0.50, nominal_duration), 20.0, 40)
                    if nominal_duration < 20.0
                    else np.asarray([20.0]),
                )
            )
        )
        feasible: list[tuple[float, np.ndarray]] = []
        for duration in duration_candidates:
            coefficients = self._quintic_coefficients(
                self.position, self.velocity, self.acceleration, target, float(duration)
            )
            maximum_speed, maximum_acceleration = self._segment_extrema(coefficients, float(duration))
            times = np.linspace(0.0, float(duration), 129)
            c = coefficients
            positions = c[0] + c[1]*times + c[2]*times**2 + c[3]*times**3 + c[4]*times**4 + c[5]*times**5
            if (
                maximum_speed <= self.speed_range[1] + 1.0e-5
                and maximum_acceleration <= self.acceleration_limit + 1.0e-5
                and positions.min() >= self.lower - 1.0e-4
                and positions.max() <= self.upper + 1.0e-4
            ):
                feasible.append((float(duration), coefficients))
        if not feasible:
            return False
        duration, coefficients = min(feasible, key=lambda item: abs(item[0] - nominal_duration))
        self.target = target
        self._coefficients = coefficients
        self._duration = float(duration)
        self._elapsed = 0.0
        self._dwell = float(self.rng.uniform(*self.dwell_range))
        self._segment += 1
        return True

    def _target_respects_guard(self, retarget_guard: tuple[float, float]) -> bool:
        center, radius = map(float, retarget_guard)
        relative = self.position - center
        direction = np.sign(relative)
        if abs(relative) < 1.0e-6:
            direction = np.sign(self.velocity)
        if direction == 0.0:
            return False
        boundary = center + direction * radius
        return bool(
            self.target <= boundary + 1.0e-6
            if direction < 0.0
            else self.target >= boundary - 1.0e-6
        )

    def _evaluate(self, time_value: float) -> None:
        t = min(max(float(time_value), 0.0), self._duration)
        c = self._coefficients
        self.position = float(c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5)
        self.velocity = float(c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4)
        self.acceleration = float(2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3)

    def step(self, dt: float, *, retarget_guard: tuple[float, float] | None = None) -> MotionSample:
        # Entering the robot's certified reaction envelope must also redirect
        # an already-active inward segment. Merely constraining the following
        # waypoint can leave several seconds of unavoidable inward motion.
        # _start_segment preserves the current position, velocity and
        # acceleration in the new quintic, so this does not introduce a jump.
        if retarget_guard is not None and not self._target_respects_guard(retarget_guard):
            self._start_segment(retarget_guard)
        remaining = max(0.0, float(dt))
        while remaining > 1.0e-12:
            if self._elapsed >= self._duration:
                self.position = self.target
                self.velocity = 0.0
                self.acceleration = 0.0
                dwell_step = min(remaining, self._dwell)
                self._dwell -= dwell_step
                remaining -= dwell_step
                if self._dwell <= 1.0e-12:
                    if not self._start_segment(retarget_guard):
                        if retarget_guard is not None:
                            self._dwell = max(remaining, 0.1)
                            continue
                        raise RuntimeError("could not connect a stopped random waypoint segment")
                continue
            step = min(remaining, self._duration - self._elapsed)
            self._elapsed += step
            remaining -= step
            self._evaluate(self._elapsed)
            if (
                self._elapsed > 0.25 * self._duration
                and self.rng.random() < self.interrupt_probability_per_second * step
                and abs(self.target - self.position) >= 0.5 * self.minimum_target_distance
            ):
                self._start_segment(retarget_guard)
        return MotionSample(
            self.position, self.velocity, self.acceleration, self.target, self._segment
        )


class LateralCrossingMotion3D(ConstantSpeedWaypointMotion3D):
    """Seeded, repeatable lateral crossing between two y-side waypoints."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        initial: np.ndarray,
        crossing_waypoints: np.ndarray,
        speed: float = 0.20,
    ) -> None:
        points = np.asarray(crossing_waypoints, dtype=np.float64).reshape(-1, 3)
        if len(points) < 2:
            raise ValueError("lateral crossing needs at least two waypoints")
        self._crossing_waypoints = points.copy()
        # The initial point is already occupied; the first segment crosses to
        # the opposite side instead of creating a zero-length segment.
        self._crossing_index = 1 % len(points)
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        upper = np.maximum(upper, lower + 1.0e-6)
        super().__init__(
            rng,
            initial=np.asarray(initial, dtype=np.float64),
            lower=lower,
            upper=upper,
            speed=speed,
            minimum_target_distance=0.18,
        )

    def _sample_target(self) -> np.ndarray:
        target = self._crossing_waypoints[self._crossing_index]
        self._crossing_index = (self._crossing_index + 1) % len(self._crossing_waypoints)
        return target.copy()


def precompute_motion_schedule(
    motion: RandomWaypointMotion | ConstantSpeedWaypointMotion3D,
    *,
    steps: int,
    dt: float,
) -> np.ndarray:
    """Return an independent, seeded motion schedule without mutating ``motion``.

    Scalar schedules contain position, velocity, acceleration, target and
    segment. Three-dimensional schedules contain three columns for each vector
    followed by target XYZ and segment. The first row is the reset state; later
    rows are sampled after each physics step. No robot state is an input.
    """

    if steps < 0:
        raise ValueError("steps must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    clone = deepcopy(motion)
    if isinstance(clone, ConstantSpeedWaypointMotion3D):
        result = np.empty((int(steps) + 1, 13), dtype=np.float32)
        result[0] = np.concatenate((
            clone.position,
            clone.velocity,
            clone.acceleration,
            clone.target,
            np.asarray([clone._segment], dtype=np.float64),
        ))
        for index in range(1, len(result)):
            sample = clone.step(float(dt))
            result[index] = np.concatenate((
                sample.position,
                sample.velocity,
                sample.acceleration,
                sample.target,
                np.asarray([sample.segment], dtype=np.float64),
            ))
        return result
    result = np.empty((int(steps) + 1, 5), dtype=np.float32)
    result[0] = (
        clone.position,
        clone.velocity,
        clone.acceleration,
        clone.target,
        clone._segment,
    )
    for index in range(1, len(result)):
        sample = clone.step(float(dt))
        result[index] = (
            sample.position,
            sample.velocity,
            sample.acceleration,
            sample.target,
            sample.segment,
        )
    return result
