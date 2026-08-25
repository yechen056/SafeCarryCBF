"""Evaluation-only obstacle templates for paired Flow/CBF experiments."""

from __future__ import annotations

import colorsys
import numpy as np

from flowcarrycbf.policies.flowcarry_cbf.control.motion import (
    ConstantSpeedWaypointMotion3D,
    VectorMotionSample,
)
from flowcarrycbf.policies.flowcarry_cbf.control.scene import phase3_quick_obstacle_models
from flowcarrycbf.policies.flowcarry_cbf.control.schema import INSTANCE_COLORS


EVAL_BASE_TEMPLATE_IDS = (
    "dual_arm_easy",
    "dual_joint_easy",
    "cube_arm_easy",
    "cube_joint_easy",
)
EVAL_STRESS_TEMPLATE_IDS = tuple(
    f"{template.removesuffix('_easy')}_stress"
    for template in EVAL_BASE_TEMPLATE_IDS
)
EVAL_HARD_TEMPLATE_IDS = tuple(
    f"{template.removesuffix('_easy')}_hard"
    for template in EVAL_BASE_TEMPLATE_IDS
)
EVAL_TEMPLATE_IDS = (
    EVAL_BASE_TEMPLATE_IDS + EVAL_STRESS_TEMPLATE_IDS + EVAL_HARD_TEMPLATE_IDS
)

HARD_Y_LIMIT = 1.20

# Historical IDs remain readable for old manifests and saved evaluation plans.
LEGACY_TEMPLATE_ID_ALIASES = {
    **{
        old: new for old, new in zip(
            ("2ball_arm", "2ball_joint", "1ball_cube_arm", "1ball_cube_joint"),
            EVAL_BASE_TEMPLATE_IDS,
        )
    },
    **{
        old: new for old, new in zip(
            ("2ball_arm_stress", "2ball_joint_stress", "1ball_cube_arm_stress", "1ball_cube_joint_stress"),
            EVAL_STRESS_TEMPLATE_IDS,
        )
    },
}
def canonical_template_id(template_id: str) -> str:
    """Return the current compact ID while accepting historical names."""

    value = str(template_id)
    return LEGACY_TEMPLATE_ID_ALIASES.get(value, value)


def _quick_template_id(template_id: str) -> str:
    """Map an evaluation ID to the unchanged Phase 3 quick geometry ID."""

    value = canonical_template_id(template_id)
    base = value.removesuffix("_hard").removesuffix("_stress")
    if not base.endswith("_easy"):
        base = f"{base}_easy"
    return {
        "dual_arm_easy": "dual_arm_easy",
        "dual_joint_easy": "dual_joint_easy",
        "cube_arm_easy": "cube_arm_easy",
        "cube_joint_easy": "cube_joint_easy",
    }[base]


class _OneWayCrossingMotion3D(ConstantSpeedWaypointMotion3D):
    """A single robot-independent constant-speed crossing for evaluation."""

    def __init__(
        self,
        initial: np.ndarray,
        velocity: np.ndarray,
        *,
        launch_delay: float = 0.0,
        stop_at: np.ndarray | None = None,
    ) -> None:
        self.position = np.asarray(initial, dtype=np.float64).reshape(3).copy()
        self._travel_velocity = np.asarray(
            velocity, dtype=np.float64,
        ).reshape(3).copy()
        self._stop_at = (
            None
            if stop_at is None
            else np.asarray(stop_at, dtype=np.float64).reshape(3).copy()
        )
        if self._stop_at is not None:
            travel_squared = float(self._travel_velocity @ self._travel_velocity)
            if travel_squared <= 1.0e-12:
                raise ValueError("a bounded crossing requires non-zero velocity")
            if float((self._stop_at - self.position) @ self._travel_velocity) < 0.0:
                raise ValueError("crossing stop must lie along the travel direction")
        self._launch_delay = max(0.0, float(launch_delay))
        self._stopped = False
        self.velocity = (
            np.zeros(3, dtype=np.float64)
            if self._launch_delay > 0.0 else self._travel_velocity.copy()
        )
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.speed = float(np.linalg.norm(self._travel_velocity))
        self.target = (
            self.position + 1000.0 * self._travel_velocity
            if self._stop_at is None else self._stop_at.copy()
        )
        self._segment = 1

    def step(self, dt: float) -> VectorMotionSample:
        if float(dt) < 0.0:
            raise ValueError("dt must be non-negative")
        remaining = float(dt)
        previous_velocity = self.velocity.copy()
        if self._launch_delay > 0.0:
            waiting = min(remaining, self._launch_delay)
            self._launch_delay -= waiting
            remaining -= waiting
            if self._launch_delay <= 1.0e-12:
                self.velocity = self._travel_velocity.copy()
        if not self._stopped and remaining > 0.0:
            if self._stop_at is None:
                self.position += self.velocity * remaining
            else:
                speed_squared = float(self._travel_velocity @ self._travel_velocity)
                time_to_stop = float(
                    (self._stop_at - self.position) @ self._travel_velocity
                    / speed_squared
                )
                if time_to_stop <= remaining + 1.0e-12:
                    self.position = self._stop_at.copy()
                    self.velocity = np.zeros(3, dtype=np.float64)
                    self._stopped = True
                    self._segment = 2
                else:
                    self.position += self.velocity * remaining
        self.acceleration = (
            (self.velocity - previous_velocity) / float(dt)
            if float(dt) > 0.0 else np.zeros(3, dtype=np.float64)
        )
        return VectorMotionSample(
            self.position.copy(),
            self.velocity.copy(),
            self.acceleration.copy(),
            self.target.copy(),
            self._segment,
        )


class _DoubleCrossingMotion3D(ConstantSpeedWaypointMotion3D):
    """A smooth, finite out-and-back crossing that is harder than stress.

    Stress contains one constant-speed pass.  Hard uses the same peak speed
    and a no-smaller sphere, but the primary threat returns once.  Stopping
    after the return avoids the indefinitely repeating, eventually
    unrecoverable encounters of the former sine schedule.
    """

    def __init__(
        self,
        *,
        x: float,
        z: float,
        amplitude: float,
        peak_speed: float,
        crossing_time: float,
        direction: float,
        dwell: float = 12.0,
        post_return_loiter_inner: float | None = None,
    ) -> None:
        self._x = float(x)
        self._z = float(z)
        self._amplitude = float(amplitude)
        self._duration = (
            np.pi * self._amplitude / max(float(peak_speed), 1.0e-6)
        )
        self._launch_delay = max(
            0.0, float(crossing_time) - 0.5 * self._duration,
        )
        self._dwell = max(0.0, float(dwell))
        self._direction = 1.0 if float(direction) >= 0.0 else -1.0
        self._post_return_loiter_inner = (
            None
            if post_return_loiter_inner is None
            else min(float(post_return_loiter_inner), self._amplitude - 0.02)
        )
        if self._post_return_loiter_inner is not None:
            self._loiter_center = 0.5 * (
                self._amplitude + self._post_return_loiter_inner
            )
            self._loiter_amplitude = 0.5 * (
                self._amplitude - self._post_return_loiter_inner
            )
            crossing_acceleration = (
                self._amplitude * (np.pi / self._duration) ** 2
            )
            self._loiter_omega = np.sqrt(
                crossing_acceleration / max(self._loiter_amplitude, 1.0e-6)
            )
        self._time = 0.0
        self._segment = 0
        self.speed = float(peak_speed)
        self.position, self.velocity, self.acceleration = self._sample(0.0)
        self.target = np.asarray([
            self._x, self._direction * self._amplitude, self._z,
        ], dtype=np.float64)

    def _sample(self, time_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        elapsed = float(time_value) - self._launch_delay
        amplitude, direction = self._amplitude, self._direction
        duration = self._duration
        if elapsed <= 0.0:
            y, velocity, acceleration = -direction * amplitude, 0.0, 0.0
            segment = 0
        elif elapsed < duration:
            phase = np.pi * elapsed / duration
            y = -direction * amplitude * np.cos(phase)
            velocity = direction * amplitude * np.pi / duration * np.sin(phase)
            acceleration = direction * amplitude * (np.pi / duration) ** 2 * np.cos(phase)
            segment = 1
        elif elapsed < duration + self._dwell:
            y, velocity, acceleration = direction * amplitude, 0.0, 0.0
            segment = 2
        elif elapsed < 2.0 * duration + self._dwell:
            phase = np.pi * (elapsed - duration - self._dwell) / duration
            y = direction * amplitude * np.cos(phase)
            velocity = -direction * amplitude * np.pi / duration * np.sin(phase)
            acceleration = -direction * amplitude * (np.pi / duration) ** 2 * np.cos(phase)
            segment = 3
        elif self._post_return_loiter_inner is None:
            y, velocity, acceleration = -direction * amplitude, 0.0, 0.0
            segment = 4
        else:
            phase = self._loiter_omega * (
                elapsed - 2.0 * duration - self._dwell
            )
            start_side = -direction
            magnitude = (
                self._loiter_center
                + self._loiter_amplitude * np.cos(phase)
            )
            y = start_side * magnitude
            velocity = (
                -start_side * self._loiter_amplitude
                * self._loiter_omega * np.sin(phase)
            )
            acceleration = (
                -start_side * self._loiter_amplitude
                * self._loiter_omega**2 * np.cos(phase)
            )
            segment = 4
        self._segment = segment
        return (
            np.asarray([self._x, y, self._z], dtype=np.float64),
            np.asarray([0.0, velocity, 0.0], dtype=np.float64),
            np.asarray([0.0, acceleration, 0.0], dtype=np.float64),
        )

    def step(self, dt: float) -> VectorMotionSample:
        if float(dt) < 0.0:
            raise ValueError("dt must be non-negative")
        self._time += float(dt)
        self.position, self.velocity, self.acceleration = self._sample(self._time)
        if self._segment <= 2:
            target_y = self._direction * self._amplitude
        elif self._segment == 3 or self._post_return_loiter_inner is None:
            target_y = -self._direction * self._amplitude
        else:
            start_side = -self._direction
            outward = self.velocity[1] * start_side >= 0.0
            magnitude = (
                self._amplitude if outward
                else float(self._post_return_loiter_inner)
            )
            target_y = start_side * magnitude
        self.target = np.asarray([self._x, target_y, self._z], dtype=np.float64)
        return VectorMotionSample(
            self.position.copy(), self.velocity.copy(), self.acceleration.copy(),
            self.target.copy(), self._segment,
        )


class _SafeSideOscillationMotion3D(ConstantSpeedWaypointMotion3D):
    """Continuously move a visual distractor inside one safe outer lane."""

    def __init__(
        self,
        *,
        x: float,
        z: float,
        inner_y: float = 1.02,
        outer_y: float = 1.18,
        peak_speed: float = 0.08,
    ) -> None:
        self._x = float(x)
        self._z = float(z)
        self._center = 0.5 * (float(inner_y) + float(outer_y))
        self._amplitude = 0.5 * (float(outer_y) - float(inner_y))
        if self._amplitude <= 0.0:
            raise ValueError("safe-side oscillation requires inner_y < outer_y")
        self._omega = float(peak_speed) / self._amplitude
        self._time = 0.0
        self._segment = 1
        self.speed = float(peak_speed)
        self.position, self.velocity, self.acceleration = self._sample(0.0)
        self.target = np.asarray([self._x, float(outer_y), self._z], dtype=np.float64)

    def _sample(self, time_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase = self._omega * float(time_value)
        return (
            np.asarray([
                self._x,
                self._center + self._amplitude * np.sin(phase),
                self._z,
            ], dtype=np.float64),
            np.asarray([
                0.0,
                self._amplitude * self._omega * np.cos(phase),
                0.0,
            ], dtype=np.float64),
            np.asarray([
                0.0,
                -self._amplitude * self._omega**2 * np.sin(phase),
                0.0,
            ], dtype=np.float64),
        )

    def step(self, dt: float) -> VectorMotionSample:
        if float(dt) < 0.0:
            raise ValueError("dt must be non-negative")
        previous_sign = np.sign(float(self.velocity[1]))
        self._time += float(dt)
        self.position, self.velocity, self.acceleration = self._sample(self._time)
        current_sign = np.sign(float(self.velocity[1]))
        if previous_sign != 0.0 and current_sign != 0.0 and current_sign != previous_sign:
            self._segment += 1
        target_y = self._center + (
            self._amplitude if self.velocity[1] >= 0.0 else -self._amplitude
        )
        self.target = np.asarray([self._x, target_y, self._z], dtype=np.float64)
        return VectorMotionSample(
            self.position.copy(), self.velocity.copy(), self.acceleration.copy(),
            self.target.copy(), self._segment,
        )


def base_template_id(template_id: str) -> str:
    value = canonical_template_id(template_id)
    value = value.removesuffix("_hard")
    value = value.removesuffix("_stress")
    return value if value.endswith("_easy") else f"{value}_easy"


def is_stress_template(template_id: str) -> bool:
    return canonical_template_id(template_id).removesuffix("_hard").endswith("_stress")


def is_hard_template(template_id: str) -> bool:
    return canonical_template_id(template_id).endswith("_hard")


def _trackable_color(
    rng: np.random.Generator, palette_index: int,
) -> np.ndarray:
    brightness = float(rng.uniform(0.88, 1.0))
    return np.clip(
        INSTANCE_COLORS[int(palette_index) % len(INSTANCE_COLORS)] * brightness,
        0.0,
        1.0,
    ).astype(np.float32)


def _hard_color(
    rng: np.random.Generator, palette_index: int,
) -> np.ndarray:
    base = INSTANCE_COLORS[int(palette_index) % len(INSTANCE_COLORS)]
    hue, saturation, _value = colorsys.rgb_to_hsv(*[float(value) for value in base])
    hue = (hue + float(rng.uniform(-4.0, 4.0)) / 360.0) % 1.0
    value = float(rng.uniform(0.78, 1.0))
    return np.asarray(colorsys.hsv_to_rgb(hue, saturation, value), dtype=np.float32)


def _hard_arm_motion(
    rng: np.random.Generator,
    model: dict,
    *,
    stress: bool,
    z: float,
    crossing_time: float,
    amplitude: float,
) -> _DoubleCrossingMotion3D:
    del stress
    peak_speed = 0.20
    direction = -1.0 if float(rng.random()) < 0.5 else 1.0
    x = float(np.asarray(model["position"])[0] + rng.uniform(-0.08, 0.08))
    z_value = float(z + rng.uniform(-0.08, 0.08))
    return _DoubleCrossingMotion3D(
        x=x,
        z=z_value,
        amplitude=amplitude,
        peak_speed=peak_speed,
        crossing_time=crossing_time,
        direction=direction,
    )


def _replace_with_hard_ball(
    model: dict,
    rng: np.random.Generator,
    *,
    stress: bool,
    z: float,
    crossing_time: float,
    palette_index: int,
    arm_target: bool,
) -> dict:
    result = dict(model)
    # Match stress's seeded size distribution exactly.  Difficulty comes
    # from the additional return pass, not from making the instantaneous
    # geometry larger than the stress contract.
    radius = float(rng.choice((0.18, 0.22, 0.25)))
    motion = _hard_arm_motion(
        rng, model, stress=stress, z=z, crossing_time=crossing_time,
        amplitude=0.95 + radius,
    )
    result["position"] = motion.position.astype(np.float32)
    result["half_extents"] = np.full(3, radius, dtype=np.float32)
    result["motion"] = motion
    result["color"] = _hard_color(rng, palette_index)
    result["eval_hard"] = True
    result["eval_hard_arm_target"] = bool(arm_target)
    result["motion_y_bounds"] = (-0.95 - radius, 0.95 + radius)
    result["crossing_speed"] = float(motion.speed)
    result["crossing_time"] = float(crossing_time)
    return result


def _keep_x_separation(models: list[dict]) -> None:
    first_x = float(np.asarray(models[0]["position"])[0])
    second_x = float(np.asarray(models[1]["position"])[0])
    separation = abs(first_x - second_x)
    if separation >= 1.0:
        return
    direction = 1.0 if second_x >= first_x else -1.0
    shift = direction * (1.0 - separation + 1.0e-3)
    models[1]["position"][0] += shift
    motion = models[1].get("motion")
    if motion is not None and hasattr(motion, "_x"):
        motion._x += shift
        motion.position[0] += shift
        motion.target[0] += shift


def _hard_models(
    rng: np.random.Generator,
    source_template: str,
    models: list[dict],
) -> list[dict]:
    base = base_template_id(source_template)
    stress = is_stress_template(source_template)
    crossing_time = 16.5 if stress else 18.0
    models[0] = _replace_with_hard_ball(
        models[0], rng, stress=stress, z=1.02,
        crossing_time=crossing_time, palette_index=0, arm_target=True,
    )
    if base.startswith("dual"):
        secondary_arm_target = base == "dual_arm_easy"
        if secondary_arm_target:
            # Keep the secondary sphere visibly moving in the far-side lane
            # without restoring the former simultaneous two-arm trap.
            models[1] = _replace_with_outbound_distractor(
                models[1], rng, palette_index=1, oscillating=True,
            )
        else:
            # Complete the lower crossing, return it to its safe start side,
            # then keep it moving there instead of leaving a static blocker.
            models[1] = _replace_with_crossing(
                models[1], rng, z=0.45, crossing_time=13.0,
                palette_index=1, return_loiter=True,
            )
        models[1]["eval_hard"] = True
        models[1]["eval_hard_arm_target"] = False
    else:
        # Match the upper end of stress geometry; the returning sphere makes
        # the schedule strictly harder than stress without enlarging the cube.
        half_extent = 0.28
        models[1]["half_extents"] = np.full(3, half_extent, dtype=np.float32)
        models[1]["position"][2] = half_extent
        models[1]["color"] = _hard_color(rng, 1)
        models[1]["eval_hard"] = True
    _keep_x_separation(models)
    return models


def _crossing_motion(
    rng: np.random.Generator,
    *,
    x: float,
    z: float,
    radius: float,
    desired_crossing_time: float,
    bounded: bool = False,
    return_loiter: bool = False,
) -> ConstantSpeedWaypointMotion3D:
    direction = -1.0 if float(rng.random()) < 0.5 else 1.0
    crossing_time = max(
        0.5,
        float(desired_crossing_time) + float(rng.uniform(-0.35, 0.35)),
    )
    velocity = np.asarray([0.0, direction * 0.20, 0.0], dtype=np.float64)
    edge = 0.95 + float(radius)
    initial = np.asarray(
        [x, -direction * edge, z], dtype=np.float64,
    )
    launch_delay = max(0.0, crossing_time - edge / 0.20)
    if return_loiter:
        return _DoubleCrossingMotion3D(
            x=x,
            z=z,
            amplitude=edge,
            peak_speed=0.20,
            crossing_time=crossing_time,
            direction=direction,
            dwell=0.0,
            post_return_loiter_inner=1.02,
        )
    return _OneWayCrossingMotion3D(
        initial,
        velocity,
        launch_delay=launch_delay,
        stop_at=(
            np.asarray([x, direction * edge, z], dtype=np.float64)
            if bounded else None
        ),
    )


def _replace_with_crossing(
    model: dict,
    rng: np.random.Generator,
    *,
    z: float,
    crossing_time: float,
    palette_index: int,
    bounded: bool = False,
    return_loiter: bool = False,
) -> dict:
    result = dict(model)
    radius = float(rng.choice((0.18, 0.22, 0.25)))
    # Preserve the collection layout invariant. Stress comes from the larger
    # moving sphere and crossing phase, not from compressing the x spacing.
    x = float(np.asarray(model["position"])[0])
    motion = _crossing_motion(
        rng,
        x=x,
        z=float(z + rng.uniform(-0.04, 0.04)),
        radius=radius,
        desired_crossing_time=crossing_time,
        bounded=bounded,
        return_loiter=return_loiter,
    )
    result["position"] = motion.position.astype(np.float32)
    result["half_extents"] = np.full(3, radius, dtype=np.float32)
    result["motion"] = motion
    result["color"] = _trackable_color(rng, palette_index)
    result["eval_crossing"] = True
    result["eval_cross_return_loiter"] = bool(return_loiter)
    result["crossing_speed"] = 0.20
    result["crossing_time"] = float(crossing_time)
    return result


def _replace_with_outbound_distractor(
    model: dict,
    rng: np.random.Generator,
    *,
    palette_index: int,
    oscillating: bool = False,
) -> dict:
    result = dict(model)
    radius = float(np.asarray(model["half_extents"])[0])
    position = np.asarray(model["position"], dtype=np.float64).copy()
    # Quick transport uses the -y corridor; keep this non-target sphere on
    # the opposite side while preserving independent dynamic motion.
    if oscillating:
        motion = _SafeSideOscillationMotion3D(
            x=float(position[0]), z=float(position[2]),
        )
    else:
        position[1] = 1.0 + radius
        velocity = np.asarray([0.0, 0.20, 0.0], dtype=np.float64)
        motion = _OneWayCrossingMotion3D(position, velocity)
    result["position"] = motion.position.astype(np.float32)
    result["motion"] = motion
    result["color"] = _trackable_color(rng, palette_index)
    result["eval_outbound_distractor"] = True
    result["eval_safe_side_oscillation"] = bool(oscillating)
    result["crossing_speed"] = float(motion.speed)
    return result


def eval_obstacle_models(
    rng: np.random.Generator,
    template_id: str,
) -> list[dict]:
    """Return a deterministic base or stress model without altering collection."""

    template = canonical_template_id(template_id)
    if template not in EVAL_TEMPLATE_IDS:
        raise ValueError(f"unsupported adaptive eval template: {template}")
    source_template = template.removesuffix("_hard")
    base = base_template_id(source_template)
    models = phase3_quick_obstacle_models(rng, _quick_template_id(base))
    for index, model in enumerate(models):
        model["eval_template_id"] = template
        model["color"] = _trackable_color(rng, index)
    if is_hard_template(template):
        models = _hard_models(rng, source_template, models)
        for index, model in enumerate(models):
            model["eval_template_id"] = template
            model["template_id"] = template
            if not model.get("eval_hard"):
                model["color"] = _hard_color(rng, index)
        return models
    if not is_stress_template(template):
        # The collection generator currently gives both cube variants the
        # same geometry. In eval, distinguish the joint variant by aligning
        # its sphere event with the still-offset cube-bypass interval.
        if base == "cube_joint_easy":
            models[0]["eval_event_window"] = "cube_bypass_overlap"
        elif base == "cube_arm_easy":
            models[0]["eval_event_window"] = "after_cube_bypass"
        return models

    if base == "dual_arm_easy":
        models[0] = _replace_with_crossing(
            models[0], rng, z=1.02, crossing_time=18.0, palette_index=0,
            bounded=True,
        )
        models[1] = _replace_with_outbound_distractor(
            models[1], rng, palette_index=1,
        )
    elif base == "dual_joint_easy":
        models[0] = _replace_with_crossing(
            models[0], rng, z=1.02, crossing_time=18.0, palette_index=0,
            bounded=True,
        )
        models[1] = _replace_with_crossing(
            models[1], rng, z=0.45, crossing_time=13.0, palette_index=1,
            bounded=True,
        )
    elif base == "cube_arm_easy":
        models[0] = _replace_with_crossing(
            models[0], rng, z=1.02, crossing_time=18.5, palette_index=0,
            bounded=True,
        )
        models[1]["half_extents"] = np.full(
            3, float(rng.uniform(0.25, 0.28)), dtype=np.float32,
        )
        models[1]["position"][2] = models[1]["half_extents"][2]
    elif base == "cube_joint_easy":
        models[0] = _replace_with_crossing(
            models[0], rng, z=0.92, crossing_time=16.0, palette_index=0,
            bounded=True,
        )
        models[0]["eval_event_window"] = "cube_bypass_overlap"
        models[1]["half_extents"] = np.full(
            3, float(rng.uniform(0.25, 0.28)), dtype=np.float32,
        )
        models[1]["position"][2] = models[1]["half_extents"][2]
    for index, model in enumerate(models):
        model["eval_template_id"] = template
        model["template_id"] = template
        model["color"] = _trackable_color(rng, index)
    return models


def sample_eval_cases(
    *,
    selection_seed: int,
    runs: int = 1,
) -> list[tuple[int, str]]:
    """Return the deterministic final12 cases in stable template order."""

    generator = np.random.default_rng(int(selection_seed))

    def canonical_seed(template: str, count: int) -> list[int]:
        start = 0 if str(template).startswith("dual") else 100000
        values = generator.choice(25, size=int(count), replace=False)
        return [start + int(value) for value in values]

    if int(runs) <= 0:
        raise ValueError("runs must be positive")
    if int(runs) > 25:
        raise ValueError("runs cannot exceed 25")
    return [
        (seed, template)
        for template in EVAL_TEMPLATE_IDS
        for seed in canonical_seed(template, int(runs))
    ]
