"""Fixed-topology sphere/cube obstacle slots for Safe Carry V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade


@dataclass(frozen=True)
class V2ObstacleSlot:
    root: str
    sphere: str
    cube: str


def create_v2_obstacle_slot(stage, root_path: str, material_path: str, color: Sequence[float]) -> V2ObstacleSlot:
    root = UsdGeom.Xform.Define(stage, root_path)
    rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(20.0)
    PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0.0)
    material = UsdShade.Material.Get(stage, material_path)
    # Bind a real render material in addition to the physics-only material.
    # Some Isaac renderer versions ignore displayColor when a prim also has a
    # material binding, which would make RGB HSV instance extraction blind.
    visual_path = root_path + "/VisualMaterial"
    visual = UsdShade.Material.Define(stage, visual_path)
    shader = UsdShade.Shader.Define(stage, visual_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    sphere_path = root_path + "/sphere"
    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.CreateRadiusAttr(0.12)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
    sphere_collision = UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    sphere_collision.CreateCollisionEnabledAttr(False)
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material, materialPurpose="physics")
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(visual)

    cube_path = root_path + "/cube"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(0.45, 0.45, 0.45))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
    cube_collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube_collision.CreateCollisionEnabledAttr(False)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose="physics")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(visual)
    return V2ObstacleSlot(root=root_path, sphere=sphere_path, cube=cube_path)


def configure_v2_obstacle_slot(
    stage,
    slot: V2ObstacleSlot,
    shape: str,
    half_extents: Sequence[float],
    color: Sequence[float] | None = None,
) -> None:
    sphere = UsdGeom.Sphere.Get(stage, slot.sphere)
    cube = UsdGeom.Cube.Get(stage, slot.cube)
    if color is not None:
        value = Gf.Vec3f(*map(float, color))
        sphere.CreateDisplayColorAttr([value])
        cube.CreateDisplayColorAttr([value])
        shader = UsdShade.Shader.Get(stage, slot.root + "/VisualMaterial/PreviewSurface")
        if shader:
            shader.GetInput("diffuseColor").Set(value)
    sphere_active = shape == "sphere"
    cube_active = shape == "cube"
    UsdPhysics.CollisionAPI.Get(stage, slot.sphere).GetCollisionEnabledAttr().Set(sphere_active)
    UsdPhysics.CollisionAPI.Get(stage, slot.cube).GetCollisionEnabledAttr().Set(cube_active)
    sphere.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if sphere_active else UsdGeom.Tokens.invisible)
    cube.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if cube_active else UsdGeom.Tokens.invisible)
    if sphere_active:
        sphere.GetRadiusAttr().Set(float(half_extents[0]))
    elif cube_active:
        scale = Gf.Vec3f(*[2.0 * float(value) for value in half_extents])
        operations = cube.GetOrderedXformOps()
        if operations:
            operations[0].Set(scale)
        else:
            cube.AddScaleOp().Set(scale)
    elif shape != "inactive":
        raise ValueError(f"unsupported V2 obstacle shape: {shape}")
