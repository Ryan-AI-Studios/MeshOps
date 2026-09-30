"""Track 0124 — soft placement consume (no 0102 scale consume).

Does not weaken 0102 T4/T5. Authoring only — not mesh or print success.
"""

from __future__ import annotations

import pytest

import meshops.proportion.face_recipe as face_recipe_mod
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.face_recipe import (
    EYE_RADIUS_FRAC_H,
    EYE_RY_FRAC_R,
    EYE_RZ_FRAC_R,
    FEATURE_FACE_Y_FRAC_RY,
    NOSE_RY_FRAC_H,
    NOSE_TIP_Y_FRAC_RY,
    build_face_parts,
)
from meshops.proportion.models import LandmarkXYZ
from test_proportion_head_face_hierarchy import (
    _full_torso_report,
    _product_class_bounds,
)


def _lm(
    lid: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=lid, x_m=x_m, y_m=y_m, z_m=z_m)


def test_e1_measured_eyes_half_ipd() -> None:
    """E1: measured eye_l/eye_r X → emitted |cx| matches half-IPD (not 2*eye_r)."""
    bounds = _product_class_bounds()
    report = _full_torso_report()
    half = 0.033
    report.landmarks_xyz["eye_l"] = _lm("eye_l", x_m=-half, z_m=1.60)
    report.landmarks_xyz["eye_r"] = _lm("eye_r", x_m=half, z_m=1.60)
    msgs: list[str] = []
    parts = build_face_parts(report, bounds, face=True, messages=msgs)
    eye_l = next(p for p in parts if p.name == "RECIPE_eye_soft_l")
    eye_r = next(p for p in parts if p.name == "RECIPE_eye_soft_r")
    assert eye_l.center is not None and eye_r.center is not None
    assert abs(float(eye_l.center[0])) == pytest.approx(half, abs=1e-6)
    assert abs(float(eye_r.center[0])) == pytest.approx(half, abs=1e-6)
    assert any("measured eye half-sep=" in m for m in msgs)


def test_e2_absent_eyes_loomis_sep() -> None:
    """E2: absent eyes → cx == ±2*EYE_RADIUS_FRAC_H*H (Loomis)."""
    bounds = _product_class_bounds()
    report = _full_torso_report()
    msgs: list[str] = []
    parts = build_face_parts(report, bounds, face=True, messages=msgs)
    eye_l = next(p for p in parts if p.name == "RECIPE_eye_soft_l")
    expected = 2.0 * EYE_RADIUS_FRAC_H * bounds.H
    assert eye_l.center is not None
    assert abs(float(eye_l.center[0])) == pytest.approx(expected, abs=1e-6)
    assert any("Loomis eye half-sep=" in m for m in msgs)


def test_e3_0102_scale_hold() -> None:
    """E3 / T4: 0102 + nose frac hold — do not consume measured into scale consts."""
    assert EYE_RY_FRAC_R == 0.62
    assert EYE_RADIUS_FRAC_H == 0.11
    assert EYE_RZ_FRAC_R == 0.58
    assert face_recipe_mod._LIP_Z_FRAC == 0.28
    assert NOSE_RY_FRAC_H == 0.055
    assert NOSE_TIP_Y_FRAC_RY == 0.98


def test_t1_measured_nose_tip_y_is_center_plus_ry() -> None:
    """T1/0131: measured nose_tip Y is the tip; center = y_m + nose.ry_m."""
    bounds = _product_class_bounds()
    report = _full_torso_report()
    tip_y = -0.08
    report.landmarks_xyz["nose_tip"] = _lm("nose_tip", y_m=tip_y, z_m=1.57)
    parts = build_face_parts(report, bounds, face=True, messages=[])
    nose = next(p for p in parts if p.name == "RECIPE_nose_soft")
    assert nose.center is not None and nose.ry_m is not None
    assert float(nose.ry_m) == pytest.approx(NOSE_RY_FRAC_H * bounds.H, abs=1e-9)
    assert float(nose.center[1]) == pytest.approx(tip_y + float(nose.ry_m), abs=1e-9)
    assert float(nose.center[1]) - float(nose.ry_m) == pytest.approx(tip_y, abs=1e-9)


def test_e4_no_dual_lip_product() -> None:
    """E4: no dual RECIPE_lip_soft_upper in product emit."""
    bounds = _product_class_bounds()
    parts = build_face_parts(_full_torso_report(), bounds, face=True, messages=[])
    names = {p.name for p in parts}
    assert "RECIPE_lip_soft" in names
    assert "RECIPE_lip_soft_upper" not in names
    assert "RECIPE_lip_soft_lower" not in names


def test_e_left_y_consumed_when_finite() -> None:
    """Left-finite y_m moves feature Y off the Loomis plane."""
    bounds = _product_class_bounds()
    report = _full_torso_report()
    report.landmarks_xyz["eye_l"] = _lm("eye_l", x_m=-0.033, y_m=-0.05, z_m=1.62)
    report.landmarks_xyz["eye_r"] = _lm("eye_r", x_m=0.033, y_m=-0.05, z_m=1.62)
    parts = build_face_parts(report, bounds, face=True, messages=[])
    eye = next(p for p in parts if p.name == "RECIPE_eye_soft_l")
    assert eye.center is not None
    assert float(eye.center[1]) == pytest.approx(-0.05, abs=1e-6)


def test_e5_front_only_y_stays_feature_plane() -> None:
    """E5: front-only measured → X/Z may move; Y stays feature-plane (no invent)."""
    bounds = _product_class_bounds()
    report = _full_torso_report()
    report.landmarks_xyz["eye_l"] = _lm("eye_l", x_m=-0.033, z_m=1.62)
    report.landmarks_xyz["eye_r"] = _lm("eye_r", x_m=0.033, z_m=1.62)
    parts = build_face_parts(report, bounds, face=True, messages=[])
    eye = next(p for p in parts if p.name == "RECIPE_eye_soft_l")
    assert eye.center is not None
    feature_y = bounds.y - FEATURE_FACE_Y_FRAC_RY * bounds.ry
    assert float(eye.center[1]) == pytest.approx(feature_y, abs=2e-4)
    assert abs(float(eye.center[0])) == pytest.approx(0.033, abs=1e-6)
    assert float(eye.center[2]) == pytest.approx(1.62, abs=2e-4)


def test_e_mcp_catalog_48() -> None:
    assert len(TOOL_NAMES) == 56
