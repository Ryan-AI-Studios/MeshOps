"""Algebra-mode parametric M4 mounting bracket template.

Geometry only — MeshOps harness exports STEP/STL.
Defaults: spacing 40 mm, wall 3, thickness 4, hole Ø 4.2.
"""

from __future__ import annotations

from meshops.design.models import BracketParams

TEMPLATE_ID = "bracket_m4"


def render_source(params: BracketParams | None = None) -> str:
    """Return build123d algebra source assigning top-level ``result``."""
    p = params if params is not None else BracketParams()
    length_x, width_y, thickness = p.plate_extents_mm()
    radius = p.hole_diameter_mm / 2.0
    half_span = p.hole_spacing_mm / 2.0
    # Cylinder taller than plate to guarantee through-cut
    cyl_h = thickness + 2.0

    return f"""\
# meshops template: {TEMPLATE_ID}
# units: mm
# params: hole_spacing_mm={p.hole_spacing_mm}, wall_mm={p.wall_mm},
#         thickness_mm={p.thickness_mm}, hole_diameter_mm={p.hole_diameter_mm}
# expected plate extents (X,Y,Z) mm: ({length_x}, {width_y}, {thickness})
# hole centers at x=±{half_span} (spacing={p.hole_spacing_mm} mm)
from build123d import Box, Cylinder, Pos

_length_x = {length_x}
_width_y = {width_y}
_thickness = {thickness}
_radius = {radius}
_half_span = {half_span}
_cyl_h = {cyl_h}

plate = Box(_length_x, _width_y, _thickness)
h1 = Pos(-_half_span, 0, 0) * Cylinder(radius=_radius, height=_cyl_h)
h2 = Pos(_half_span, 0, 0) * Cylinder(radius=_radius, height=_cyl_h)
result = plate - h1 - h2
"""


def expected_dimensions(params: BracketParams | None = None) -> dict[str, float]:
    """Golden expected dimensions for DoD-10 (±0.1 mm class)."""
    p = params if params is not None else BracketParams()
    length_x, width_y, thickness = p.plate_extents_mm()
    return {
        "extent_x_mm": length_x,
        "extent_y_mm": width_y,
        "extent_z_mm": thickness,
        "hole_spacing_mm": p.hole_spacing_mm,
        "wall_mm": p.wall_mm,
        "thickness_mm": p.thickness_mm,
        "hole_diameter_mm": p.hole_diameter_mm,
    }
