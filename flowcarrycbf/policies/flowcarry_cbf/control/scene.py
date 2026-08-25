"""Pure sampling helpers shared by the V2 Isaac task and unit tests."""

from __future__ import annotations

import numpy as np

from .motion import (
    ConstantSpeedWaypointMotion3D,
    RandomWaypointMotion,
)


def _look_at_camera_pose(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a world pose for an Isaac USD camera looking at ``target``."""

    position = np.asarray(position, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    view = target - position
    view /= np.linalg.norm(view)
    right = np.cross(view, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, view)
    # USD optical axes are +X right, +Y up and -Z forward.
    rotation = np.column_stack((right, camera_up, -view))
    return position, rotation


def stage2_camera_poses(base_pose: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Fixed opposing RGB-D views covering the complete stage-two corridor."""

    # Keep the parameter for the shared task API, but deliberately do not move
    # either camera with the robot. Opposing lateral views remove the permanent
    # self-occlusion that a carried box can create in any single side view.
    np.asarray(base_pose, dtype=np.float64).reshape(3)
    target = np.asarray([-1.50, 0.0, 0.85], dtype=np.float64)
    return tuple(
        _look_at_camera_pose(np.asarray([0.75, side * 2.50, 4.50]), target)
        for side in (-1.0, 1.0)
    )


def stage2_camera_pose(base_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the primary stage-two camera pose for compatibility."""

    return stage2_camera_poses(base_pose)[0]


def phase3_recording_camera_poses(
    base_pose: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return the robot-following and fixed overview RGB recording poses."""

    base = np.asarray(base_pose, dtype=np.float64).reshape(3)
    c, s = np.cos(base[2]), np.sin(base[2])
    forward = np.asarray([c, s, 0.0], dtype=np.float64)
    first_person_position = np.asarray(
        [base[0] + 0.08 * c, base[1] + 0.08 * s, 1.58], dtype=np.float64
    )
    # Looking over the payload from head height keeps the carried box in the
    # lower frame while retaining the complete obstacle corridor ahead.
    first_person_target = (
        np.asarray([base[0], base[1], 0.78], dtype=np.float64)
        + 2.20 * forward
    )
    first_person = _look_at_camera_pose(first_person_position, first_person_target)
    overview = _look_at_camera_pose(
        np.asarray([2.80, 4.80, 3.80], dtype=np.float64),
        np.asarray([-1.50, 0.0, 0.72], dtype=np.float64),
    )
    return first_person, overview


def effective_arm_avoidance(region: str, arm_ready: bool, payload_offset: float) -> bool:
    """Require both commanded joint convergence and measured payload motion."""

    return str(region) != "nominal" and bool(arm_ready) and float(payload_offset) >= 0.08


def uses_external_cameras(scenario: str) -> bool:
    """Return whether a scenario uses the fixed opposing RGB-D cameras."""

    return str(scenario) in {"stage2", "random", "phase3_quick"}


def sample_obstacle_counts(rng: np.random.Generator) -> tuple[int, int, int]:
    """Sample the legacy V2 random-layout obstacle counts."""

    dynamic = int(rng.integers(2, 4))
    static = int(rng.integers(0, 2))
    return dynamic + static, dynamic, static


def fixed_obstacle_models(rng: np.random.Generator) -> list[dict]:
    result = []
    for x, z in ((-1.35, 0.78), (-2.25, 1.05)):
        motion = RandomWaypointMotion(rng, initial=float(rng.uniform(-0.8, 0.8)))
        result.append({
            "shape": "sphere",
            "position": np.asarray([x, motion.position, z], dtype=np.float32),
            "half_extents": np.full(3, 0.12, dtype=np.float32),
            "motion": motion,
        })
    for y in (-0.85, 0.85):
        result.append({
            "shape": "cube",
            "position": np.asarray([-1.70, y, 0.225], dtype=np.float32),
            "half_extents": np.full(3, 0.225, dtype=np.float32),
            "motion": None,
        })
    return result


def stage1_obstacle_models() -> list[dict]:
    """Return the deterministic static-only layout used by phase one.

    Phase one deliberately removes dynamic-obstacle and perception-planning
    variables.  The geometry remains the two fixed landing cubes from V2 so
    the carry controller must demonstrate a real side passage.
    """

    return [
        {
            "shape": "cube",
            "position": np.asarray([-1.70, y, 0.225], dtype=np.float32),
            "half_extents": np.full(3, 0.225, dtype=np.float32),
            "motion": None,
        }
        for y in (-0.85, 0.85)
    ]


def random_obstacle_models(rng: np.random.Generator) -> list[dict]:
    _, dynamic_count, static_count = sample_obstacle_counts(rng)
    result: list[dict] = []
    route_edges = np.linspace(-2.50, -1.10, dynamic_count + 1)
    strata = rng.permutation(dynamic_count)
    sphere_palette = (
        (0.65, 0.02, 0.02),
        (0.02, 0.45, 0.04),
        (0.02, 0.08, 0.65),
    )
    for sphere_index in range(dynamic_count):
        stratum = int(strata[sphere_index])
        x = float(rng.uniform(route_edges[stratum] + 0.05, route_edges[stratum + 1] - 0.05))
        radius = float(rng.uniform(0.10, 0.16))
        initial = np.asarray(
            [x, rng.uniform(-0.75, 0.75), rng.uniform(0.72, 1.12)],
            dtype=np.float64,
        )
        half_ranges = np.asarray([0.12, 0.30, 0.12], dtype=np.float64)
        motion = ConstantSpeedWaypointMotion3D(
            rng,
            initial=initial,
            lower=initial - half_ranges,
            upper=initial + half_ranges,
            speed=0.30,
            minimum_target_distance=0.18,
        )
        result.append({
            "shape": "sphere",
            "position": motion.position.astype(np.float32),
            "half_extents": np.full(3, radius, dtype=np.float32),
            "motion": motion,
            "color": np.asarray(
                sphere_palette[sphere_index % len(sphere_palette)], dtype=np.float32,
            ),
        })
    for _ in range(static_count):
        half = 0.225
        position = np.asarray(
            [rng.uniform(-2.10, -1.10), rng.uniform(-0.20, 0.20), half],
            dtype=np.float32,
        )
        result.append({
            "shape": "cube",
            "position": position,
            "half_extents": np.full(3, half, dtype=np.float32),
            "motion": None,
            "color": np.asarray(
                ((0.78, 0.08, 0.32), (0.03, 0.03, 0.03), (0.62, 0.42, 0.03))[
                    int(rng.integers(0, 3))
                ], dtype=np.float32,
            ),
        })
    return result


QUICK_TEMPLATE_IDS = (
    "dual_arm_easy", "dual_joint_easy",
    "cube_arm_easy", "cube_joint_easy",
)
QUICK_TEMPLATE_ALIASES = {
    "2ball_arm": "dual_arm_easy",
    "2ball_joint": "dual_joint_easy",
    "1ball_cube_arm": "cube_arm_easy",
    "1ball_cube_joint": "cube_joint_easy",
}
QUICK_TEMPLATE_TOPOLOGY = {
    template: ("2ball_no_cube" if template.startswith("dual_") else "1ball_cube")
    for template in QUICK_TEMPLATE_IDS
}


def phase3_quick_obstacle_models(
    rng: np.random.Generator, template_id: str | None = None,
) -> list[dict]:
    """Sample the fast two-topology Phase 3 collection scene.

    The longitudinal slots are deliberately separated by more than one metre.
    This keeps the two obstacle events independently solvable while retaining
    both a dynamic sphere response and a fixed-cube base bypass signal.
    """

    sphere_palette = (
        (0.65, 0.02, 0.02),
        (0.02, 0.45, 0.04),
        (0.02, 0.08, 0.65),
    )
    if template_id is not None:
        template_id = QUICK_TEMPLATE_ALIASES.get(str(template_id), str(template_id))
        if template_id not in QUICK_TEMPLATE_IDS:
            raise ValueError(f"unsupported Phase 3 quick template: {template_id}")
    # ``None`` retains the pre-template random sampler for focused regression
    # tests. Collection passes one of the four explicit templates.
    selected_template = template_id
    if selected_template is None:
        selected_template = "cube_arm_easy" if float(rng.random()) >= 0.50 else "dual_arm_easy"
    has_cube = selected_template.startswith("cube_")
    first_x = float(rng.uniform(-2.55, -2.40))
    second_x = float(rng.uniform(-1.35, -1.20))
    if second_x - first_x < 1.0:
        second_x = first_x + 1.0
    interaction_side = -1.0 if float(rng.random()) < 0.5 else 1.0
    if has_cube:
        # The base bypass is fixed on -y. Keep the dynamic arm event on the
        # opposite side so the two signals do not occupy the same corridor.
        interaction_side = 1.0

    def make_sphere(x: float, interaction: bool, color_index: int, mode: str = "") -> dict:
        radius = (
            0.15
            if mode in {"distractor", "base", "crossing"}
            else float(rng.choice((0.15, 0.20, 0.25)))
        )
        # Keep all quick spheres in the original tabletop-height band. The
        # template distinction is lateral timing, not vertical flight.
        z = float(rng.uniform(0.95, 1.20))
        if mode == "base":
            # Keep the joint template's base event close enough to require a
            # lateral correction, but leave a real side passage for recovery.
            lower, upper = ((-0.74, -0.68) if interaction_side < 0.0 else (0.68, 0.74))
        elif mode == "distractor":
            lower, upper = ((0.72, 0.82) if interaction_side < 0.0 else (-0.82, -0.72))
        elif interaction:
            # Keep the arm-triggering sphere in the reachable side corridor,
            # outside the nominal payload envelope.  The old near-center band
            # produced genuine arm collisions before a local offset could
            # take effect, so no amount of seed resampling could make it a
            # reliable prescreen template.
            lower, upper = ((-0.94, -0.84) if interaction_side < 0.0 else (0.84, 0.94))
        else:
            lower, upper = ((0.52, 0.82) if interaction_side < 0.0 else (-0.82, -0.52))
        initial = np.asarray([x, rng.uniform(lower, upper), z], dtype=np.float64)
        half_ranges = np.asarray(
            [0.10, 0.05, 0.10]
            if mode in {"distractor"}
            else [0.10, 0.04, 0.10] if mode == "base"
            else [0.10, 0.25, 0.10],
            dtype=np.float64,
        )
        motion = ConstantSpeedWaypointMotion3D(
            rng,
            initial=initial,
            lower=initial - half_ranges,
            upper=initial + half_ranges,
            speed=0.20,
            minimum_target_distance=0.18,
        )
        return {
            "shape": "sphere",
            "position": motion.position.astype(np.float32),
            "half_extents": np.full(3, radius, dtype=np.float32),
            "motion": motion,
            "interaction_candidate": bool(interaction),
            "color": np.asarray(sphere_palette[color_index], dtype=np.float32),
        }

    def tagged(model: dict, behavior: str) -> dict:
        model["template_id"] = selected_template
        model["template_variant"] = behavior
        return model

    if has_cube:
        # The cube is always on +y; the Oracle bypass is always on -y.
        sphere = make_sphere(
            first_x, True,
            int(rng.integers(0, len(sphere_palette))),
            mode="arm",
        )
        cube = {
            "shape": "cube",
            "position": np.asarray([second_x, 0.18, 0.25], dtype=np.float32),
            "half_extents": np.full(3, 0.25, dtype=np.float32),
            "motion": None,
            "color": np.asarray((0.78, 0.08, 0.32), dtype=np.float32),
        }
        return [
            tagged(sphere, "arm" if sphere["interaction_candidate"] else "base"),
            tagged(cube, "base"),
        ]

    first = make_sphere(
        first_x,
        True,
        int(rng.integers(0, len(sphere_palette))),
        mode="arm",
    )
    second_mode = (
        "base" if selected_template == "dual_joint_easy"
        else "distractor"
    )
    second = make_sphere(
        second_x, False,
        int(rng.integers(0, len(sphere_palette))),
        mode=second_mode,
    )
    return [
        tagged(first, "arm"),
        tagged(second, "base" if selected_template == "dual_joint_easy" else "distractor"),
    ]
