"""Procedural USD assets used by the Tiago dual-arm carry task.

The payload is deliberately authored as one rigid body with several child
colliders.  The U handles are not separate bodies and no joints or hidden
constraints are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade


@dataclass(frozen=True)
class HandledBoxPaths:
    root: str
    body: str
    left_grip: str
    right_grip: str
    collision_paths: tuple[str, ...]


def _set_display_color(geom, color: Sequence[float]) -> None:
    geom.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])


def _bind_physics_material(stage, prim, material_path: str) -> None:
    material = UsdShade.Material.Get(stage, material_path)
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material, materialPurpose="physics")


def create_physics_material(
    stage,
    path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float = 0.0,
):
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(static_friction))
    api.CreateDynamicFrictionAttr(float(dynamic_friction))
    api.CreateRestitutionAttr(float(restitution))
    return material


def _configure_collider(stage, prim, material_path: str) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)
    physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx.CreateContactOffsetAttr(0.003)
    physx.CreateRestOffsetAttr(0.0)
    _bind_physics_material(stage, prim, material_path)


def _create_cube(
    stage,
    path: str,
    translation: Sequence[float],
    scale: Sequence[float],
    color: Sequence[float],
    material_path: str,
):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(Gf.Vec3d(*[float(v) for v in translation]))
    xform.SetScale(Gf.Vec3f(*[float(v) for v in scale]))
    _set_display_color(cube, color)
    _configure_collider(stage, cube.GetPrim(), material_path)
    return cube


def _create_cylinder(
    stage,
    path: str,
    translation: Sequence[float],
    axis: str,
    radius: float,
    height: float,
    color: Sequence[float],
    material_path: str,
):
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(axis.upper())
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    UsdGeom.XformCommonAPI(cylinder).SetTranslate(Gf.Vec3d(*[float(v) for v in translation]))
    _set_display_color(cylinder, color)
    _configure_collider(stage, cylinder.GetPrim(), material_path)
    return cylinder


def create_handled_box(
    stage,
    root_path: str,
    dimensions: Sequence[float] = (0.24, 0.36, 0.12),
    mass: float = 0.20,
    handle_radius: float = 0.0125,
    handle_length: float = 0.09,
    handle_clearance: float = 0.064,
    static_friction: float = 1.4,
    dynamic_friction: float = 1.2,
) -> HandledBoxPaths:
    """Create a box with symmetric U-shaped handles as one compound body.

    Coordinates are local to ``root_path``: X points forward, Y left and Z up.
    The outer grip bars run along X, allowing the Tiago finger opening axes to
    be vertical while the finger links point inward toward the payload.
    """

    dims = np.asarray(dimensions, dtype=np.float64)
    if dims.shape != (3,) or np.any(dims <= 0.0):
        raise ValueError(f"dimensions must contain three positive values, got {dimensions}")
    if handle_length <= 2.0 * handle_radius:
        raise ValueError("handle_length must be larger than the handle diameter")

    root = UsdGeom.Xform.Define(stage, root_path)
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_api.CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(float(mass))
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    physx_body.CreateEnableCCDAttr(True)
    physx_body.CreateDisableGravityAttr(False)
    physx_body.CreateSolverPositionIterationCountAttr(8)
    physx_body.CreateSolverVelocityIterationCountAttr(2)
    physx_body.CreateMaxDepenetrationVelocityAttr(2.0)

    material_path = root_path + "/PhysicsMaterial"
    create_physics_material(stage, material_path, static_friction, dynamic_friction)

    body_path = root_path + "/body"
    _create_cube(
        stage,
        body_path,
        translation=(0.0, 0.0, 0.0),
        scale=dims,
        color=(0.48, 0.22, 0.06),
        material_path=material_path,
    )

    half_y = 0.5 * dims[1]
    half_grip = 0.5 * handle_length
    outer_y = half_y + handle_clearance
    support_y = half_y + 0.5 * handle_clearance
    collision_paths = [body_path]
    grip_paths: dict[str, str] = {}

    for side_name, sign in (("left", 1.0), ("right", -1.0)):
        handle_root = root_path + f"/{side_name}_handle"
        UsdGeom.Xform.Define(stage, handle_root)
        grip_path = handle_root + "/grip_bar"
        _create_cylinder(
            stage,
            grip_path,
            translation=(0.0, sign * outer_y, 0.0),
            axis="X",
            radius=handle_radius,
            height=handle_length,
            color=(0.12, 0.12, 0.12),
            material_path=material_path,
        )
        collision_paths.append(grip_path)
        grip_paths[side_name] = grip_path

        for support_name, x in (("front_support", half_grip), ("rear_support", -half_grip)):
            support_path = handle_root + "/" + support_name
            _create_cylinder(
                stage,
                support_path,
                translation=(x, sign * support_y, 0.0),
                axis="Y",
                radius=handle_radius,
                height=handle_clearance,
                color=(0.12, 0.12, 0.12),
                material_path=material_path,
            )
            collision_paths.append(support_path)

    return HandledBoxPaths(
        root=root_path,
        body=body_path,
        left_grip=grip_paths["left"],
        right_grip=grip_paths["right"],
        collision_paths=tuple(collision_paths),
    )


def create_static_box(
    stage,
    path: str,
    position: Sequence[float],
    dimensions: Sequence[float],
    color: Sequence[float],
    material_path: str,
):
    """Create a world-space static cuboid collider."""

    return _create_cube(stage, path, position, dimensions, color, material_path)


def create_table(
    stage,
    root_path: str,
    center_xy: Sequence[float],
    material_path: str,
    top_dimensions: Sequence[float] = (0.8, 1.0, 0.06),
    height: float = 0.75,
) -> tuple[str, ...]:
    """Create a fixed table from five primitive colliders."""

    x, y = [float(v) for v in center_xy]
    top = np.asarray(top_dimensions, dtype=np.float64)
    top_z = float(height) - 0.5 * top[2]
    paths = []
    paths.append(
        str(
            create_static_box(
                stage,
                root_path + "/top",
                (x, y, top_z),
                top,
                (0.72, 0.52, 0.30),
                material_path,
            ).GetPath()
        )
    )
    leg_size = np.array([0.06, 0.06, height - top[2]], dtype=np.float64)
    leg_z = 0.5 * leg_size[2]
    inset = 0.09
    for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        leg_x = x + sx * (0.5 * top[0] - inset)
        leg_y = y + sy * (0.5 * top[1] - inset)
        paths.append(
            str(
                create_static_box(
                    stage,
                    root_path + f"/leg_{index}",
                    (leg_x, leg_y, leg_z),
                    leg_size,
                    (0.18, 0.14, 0.10),
                    material_path,
                ).GetPath()
            )
        )
    return tuple(paths)
