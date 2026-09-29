"""Track 0126 — soft new-id glute Y/Z placement (no 0106/0092/0068/0053/0070/0036 retune).

Does not weaken 0106 T* / 0068 seat / 0092 plate / 0036 outer asserts.
Authoring only — not mesh or print success.
"""

from __future__ import annotations

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    GLUTE_RX_LAT_FLOOR_FRAC_HIP_HW,
    GLUTE_SEAT_BEYOND_REF_Y,
    GLUTE_SEAT_Y_FLOOR_M,
    GLUTE_SEAT_Z_DROP_FRAC_H,
    HIP_SOFT_RY_FRAC_RX,
    HIP_SOFT_RZ_FRAC_RX,
    HIP_SOFT_Z_DROP_FRAC_H,
    PELVIS_OVAL_RY_FRAC_HALF_HIP,
    THIGH_DIST_SHAFT_SCALE,
    TORSO_HIP_Y_REAR_BIAS_FRAC_RY,
    TORSO_OVAL_RY_HIP_FRAC,
    build_blockout_recipe,
)
from meshops.proportion.models import LandmarkXYZ
from meshops.proportion.skeleton import build_blockout_skeleton
from test_proportion_torso_anti_tire_plus import (
    _product_class_report,
    _product_flags,
    _template,
)


def _lm(
    lid: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=lid, x_m=x_m, y_m=y_m, z_m=z_m)


def _emit(report, **flag_overrides: object):
    skel = build_blockout_skeleton(report)
    return build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(**flag_overrides),  # type: ignore[arg-type]
    )


def test_e1_measured_glute_y_survives_dual_lock() -> None:
    """E1: measured glute_bottom_l Y → emitted glute center[1] tracks measured.

    B32/B33: measured value must survive seat write and dual lock max(cys).
    """
    report = _product_class_report()
    measured_y = 0.18
    report.landmarks_xyz["glute_bottom_l"] = _lm(
        "glute_bottom_l", x_m=-0.13, y_m=measured_y, z_m=0.84
    )
    pkg = _emit(report)
    glute_l = next(p for p in pkg.parts if p.name == "RECIPE_glute_soft_l")
    assert glute_l.center is not None
    assert float(glute_l.center[1]) == pytest.approx(measured_y, abs=1e-6)
    assert any("measured glute y=" in m for m in pkg.messages)


def test_e2_absent_ids_prior_path() -> None:
    """E2: absent glute_bottom/top_seam ids → glute cy matches 0068 prior path."""
    report = _product_class_report()
    pkg = _emit(report)
    glute_l = next(p for p in pkg.parts if p.name == "RECIPE_glute_soft_l")
    assert glute_l.center is not None
    prior_y = float(glute_l.center[1])
    # 0068 path: floor 0.045 then beyond-ref may raise a couple mm (product ~0.0467).
    assert prior_y >= GLUTE_SEAT_Y_FLOOR_M - 1e-6
    assert prior_y == pytest.approx(GLUTE_SEAT_Y_FLOOR_M, abs=5e-3)
    assert prior_y != pytest.approx(0.18, abs=1e-3)
    assert not any("measured glute y=" in m for m in pkg.messages)


def test_e3_const_hold() -> None:
    """E3: 0106/0092/0068/0053/0070/0036 hold — B34 hip ry is _FRAC."""
    assert HIP_SOFT_RY_FRAC_RX == 0.62
    assert HIP_SOFT_RZ_FRAC_RX == 1.00
    assert HIP_SOFT_Z_DROP_FRAC_H == 0.022
    assert TORSO_OVAL_RY_HIP_FRAC == 0.64
    assert TORSO_HIP_Y_REAR_BIAS_FRAC_RY == 0.33
    assert GLUTE_SEAT_Y_FLOOR_M == 0.045
    assert GLUTE_SEAT_BEYOND_REF_Y == 0.035
    assert GLUTE_SEAT_Z_DROP_FRAC_H == 0.035
    assert PELVIS_OVAL_RY_FRAC_HALF_HIP == 0.60
    assert THIGH_DIST_SHAFT_SCALE == 0.72
    assert GLUTE_RX_LAT_FLOOR_FRAC_HIP_HW == 0.40


def test_e4_front_only_asis_no_glute_y() -> None:
    """E4: front-only measured asis → no Y write on glute."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["asis_l"] = _lm("asis_l", x_m=-0.10, z_m=0.95)
    moved = _emit(report)
    base_g = next(p for p in baseline.parts if p.name == "RECIPE_glute_soft_l")
    new_g = next(p for p in moved.parts if p.name == "RECIPE_glute_soft_l")
    assert base_g.center is not None and new_g.center is not None
    assert float(new_g.center[1]) == pytest.approx(float(base_g.center[1]), abs=1e-9)


def test_e5_have_ids_do_not_change_seat() -> None:
    """E5: hip_front/glute_back/greater_trochanter finite must not change seat emit."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["hip_front"] = _lm("hip_front", x_m=0.0, y_m=-0.20, z_m=1.00)
    report.landmarks_xyz["glute_back"] = _lm("glute_back", x_m=0.0, y_m=0.20, z_m=0.84)
    report.landmarks_xyz["greater_trochanter"] = _lm(
        "greater_trochanter", x_m=-0.22, y_m=0.01, z_m=0.86
    )
    moved = _emit(report)
    assert GLUTE_SEAT_Y_FLOOR_M == 0.045
    assert HIP_SOFT_RY_FRAC_RX == 0.62
    base_g = next(p for p in baseline.parts if p.name == "RECIPE_glute_soft_l")
    new_g = next(p for p in moved.parts if p.name == "RECIPE_glute_soft_l")
    assert base_g.center is not None and new_g.center is not None
    assert float(new_g.center[1]) == pytest.approx(float(base_g.center[1]), abs=1e-6)
    base_h = next(p for p in baseline.parts if p.name == "RECIPE_hip_soft_l")
    new_h = next(p for p in moved.parts if p.name == "RECIPE_hip_soft_l")
    assert base_h.center is not None and new_h.center is not None
    assert float(new_h.center[1]) == pytest.approx(float(base_h.center[1]), abs=1e-6)


def test_e6_measured_glute_outer_x_does_not_move_center_x() -> None:
    """E6: measured glute_outer_l X must not change glute center[0] vs 0036 align."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["glute_outer_l"] = _lm("glute_outer_l", x_m=-0.40, y_m=0.05, z_m=0.84)
    moved = _emit(report)
    base_g = next(p for p in baseline.parts if p.name == "RECIPE_glute_soft_l")
    new_g = next(p for p in moved.parts if p.name == "RECIPE_glute_soft_l")
    assert base_g.center is not None and new_g.center is not None
    assert float(new_g.center[0]) == pytest.approx(float(base_g.center[0]), abs=1e-9)


def test_e7_one_side_measured_survives_dual_lock_max() -> None:
    """E7: one-side measured Y smaller than contralateral prior survives dual lock."""
    report = _product_class_report()
    baseline = _emit(report)
    base_r = next(p for p in baseline.parts if p.name == "RECIPE_glute_soft_r")
    assert base_r.center is not None
    prior_r = float(base_r.center[1])
    measured_y = 0.02
    assert measured_y < prior_r
    report.landmarks_xyz["glute_bottom_l"] = _lm(
        "glute_bottom_l", x_m=-0.13, y_m=measured_y, z_m=0.84
    )
    pkg = _emit(report)
    glute_l = next(p for p in pkg.parts if p.name == "RECIPE_glute_soft_l")
    glute_r = next(p for p in pkg.parts if p.name == "RECIPE_glute_soft_r")
    assert glute_l.center is not None and glute_r.center is not None
    assert float(glute_l.center[1]) == pytest.approx(measured_y, abs=1e-6)
    assert float(glute_r.center[1]) == pytest.approx(prior_r, abs=1e-6)


def test_e_measured_glute_z_after_dual_lock() -> None:
    """Measured glute_bottom_l Z survives dual lock (B33)."""
    report = _product_class_report()
    measured_z = 0.90
    report.landmarks_xyz["glute_bottom_l"] = _lm(
        "glute_bottom_l", x_m=-0.13, y_m=0.18, z_m=measured_z
    )
    pkg = _emit(report)
    glute_l = next(p for p in pkg.parts if p.name == "RECIPE_glute_soft_l")
    assert glute_l.center is not None
    assert float(glute_l.center[2]) == pytest.approx(measured_z, abs=1e-6)
    assert any("measured glute z=" in m for m in pkg.messages)


def test_e_top_seam_y_both_sides() -> None:
    """Measured glute_top_seam Y overlays both glute_soft sides after dual lock."""
    report = _product_class_report()
    measured_y = 0.11
    report.landmarks_xyz["glute_top_seam"] = _lm(
        "glute_top_seam", x_m=0.0, y_m=measured_y, z_m=0.84
    )
    pkg = _emit(report)
    for side in ("l", "r"):
        glute = next(p for p in pkg.parts if p.name == f"RECIPE_glute_soft_{side}")
        assert glute.center is not None
        assert float(glute.center[1]) == pytest.approx(measured_y, abs=1e-6)
    assert any("glute_top_seam" in m for m in pkg.messages)


def test_e_mcp_catalog_50() -> None:
    assert len(TOOL_NAMES) == 54
