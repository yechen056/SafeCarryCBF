"""Visual-only scene styling for the Phase 3 RGB pipeline."""

from __future__ import annotations

from typing import Sequence

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


PHASE3_BACKGROUND_COLOR = (0.46, 0.48, 0.50)
PHASE3_TILE_COLOR = PHASE3_BACKGROUND_COLOR
PHASE3_GRID_SPACING = 0.5
PHASE3_MAJOR_GRID_SPACING = 2.0


def _preview_material(stage, path: str, color: Sequence[float]) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*map(float, color))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _is_major_line(value: float) -> bool:
    quotient = value / PHASE3_MAJOR_GRID_SPACING
    return abs(quotient - round(quotient)) < 1e-6


def create_phase3_visuals(stage) -> None:
    """Add a gray sky and collision-free metric floor grid to ``stage``.

    The physical ground plane remains responsible for contact. The overlay is
    one flat mesh made of disconnected tile quads, so it adds no collider and
    changes rendered ground depth by at most one millimeter.
    """

    root = "/World/Phase3Visuals"
    UsdGeom.Xform.Define(stage, root)

    dome = UsdLux.DomeLight.Define(stage, root + "/GrayBackground")
    dome.CreateColorAttr().Set(Gf.Vec3f(*PHASE3_BACKGROUND_COLOR))
    dome.CreateIntensityAttr().Set(180.0)
    dome.CreateExposureAttr().Set(0.0)

    material = _preview_material(stage, root + "/TileMaterial", PHASE3_TILE_COLOR)
    mesh = UsdGeom.Mesh.Define(stage, root + "/MetricGridTiles")
    points: list[Gf.Vec3f] = []
    counts: list[int] = []
    indices: list[int] = []
    grid_min = -12.0
    grid_max = 12.0
    cells = int(round((grid_max - grid_min) / PHASE3_GRID_SPACING))
    regular_half_width = 0.010
    major_half_width = 0.025

    for x_index in range(cells):
        x0 = grid_min + x_index * PHASE3_GRID_SPACING
        x1 = x0 + PHASE3_GRID_SPACING
        left = major_half_width if _is_major_line(x0) else regular_half_width
        right = major_half_width if _is_major_line(x1) else regular_half_width
        for y_index in range(cells):
            y0 = grid_min + y_index * PHASE3_GRID_SPACING
            y1 = y0 + PHASE3_GRID_SPACING
            bottom = major_half_width if _is_major_line(y0) else regular_half_width
            top = major_half_width if _is_major_line(y1) else regular_half_width
            first = len(points)
            points.extend(
                (
                    Gf.Vec3f(x0 + left, y0 + bottom, 0.001),
                    Gf.Vec3f(x1 - right, y0 + bottom, 0.001),
                    Gf.Vec3f(x1 - right, y1 - top, 0.001),
                    Gf.Vec3f(x0 + left, y1 - top, 0.001),
                )
            )
            counts.append(4)
            indices.extend((first, first + 1, first + 2, first + 3))

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
