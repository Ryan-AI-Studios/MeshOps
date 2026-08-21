"""Track 0102 — face orbital / lip shelf defaults (after 0085 leftover goggle).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 47 stay. Not photoreal / MediaPipe / chin_soft / 0107 / 0113 / 0119.
Does not reopen 0078 jaw, 0085 radius/RZ/pitch, 0058 FEATURE_Y / nose / brow.
"""

from __future__ import annotations

import math

import pytest

import meshops.proportion.face_recipe as face_recipe_mod
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    HEAD_PITCH_DEG,
    NECK_FORWARD_TILT_DEG,
    NECK_NAPE_SETBACK_M,
    RECIPE_SCHEMA_VERSION,
    _rotate_yz_about_x,
)
from meshops.proportion.face_recipe import (
    CHEEK_RY_FRAC_HEAD_RY,
    CHEEK_RZ_FRAC_H,
    CHEEK_Z_MIX,
    EYE_RADIUS_FRAC_H,
    EYE_RX_FRAC_R,
    EYE_RY_FRAC_R,
    EYE_RZ_FRAC_R,
    FEATURE_FACE_Y_FRAC_RY,
    JAW_RY_FRAC_HEAD_RY,
    JAW_RZ_FRAC_H,
    JAW_X_BULGE_ALLOW_M,
    LIP_RX_FRAC_H,
    LIP_RY_FRAC_H,
    LIP_RZ_FRAC_H,
    build_face_parts,
)
from test_proportion_head_face_hierarchy import (
    _full_torso_report,
    _head_h,
    _product_class_bounds,
    _product_pkg,
)

_DEFAULTS_PREFIX = "face orbital lip defaults:"
_HIER_PREFIX = "head face hierarchy:"
_SAUSAGE_RY = 0.035
_SAUSAGE_RZ = 0.025
EPS = 1e-6


def test_t0_const_freezes() -> None:
    """T0: 0102 eye_ry/lip_z/lip mass/cheek mix; keep 0085 radius/RZ/pitch + 0078 jaw."""
    assert EYE_RY_FRAC_R == 0.62
    assert face_recipe_mod._LIP_Z_FRAC == 0.28
    assert LIP_RY_FRAC_H == 0.028
    assert LIP_RZ_FRAC_H == 0.020
    assert CHEEK_Z_MIX == 0.30
    assert CHEEK_RZ_FRAC_H == 0.045
    assert CHEEK_RY_FRAC_HEAD_RY == 0.14
    assert EYE_RADIUS_FRAC_H == 0.11
    assert EYE_RX_FRAC_R == 1.00
    assert EYE_RZ_FRAC_R == 0.58
    assert LIP_RX_FRAC_H == 0.10
    assert FEATURE_FACE_Y_FRAC_RY == 0.90
    assert JAW_RY_FRAC_HEAD_RY == 0.42
    assert JAW_RZ_FRAC_H == 0.13
    assert HEAD_PITCH_DEG == 6.0
    assert NECK_FORWARD_TILT_DEG == 12.0
    assert 0.55 <= EYE_RY_FRAC_R <= 0.70
    assert 0.26 <= face_recipe_mod._LIP_Z_FRAC <= 0.30
    assert 0.026 <= LIP_RY_FRAC_H <= 0.032
    assert 0.018 <= LIP_RZ_FRAC_H <= 0.022
    assert 0.24 <= CHEEK_Z_MIX <= 0.34
    assert 0.040 <= CHEEK_RZ_FRAC_H <= 0.050
    assert 0.12 <= CHEEK_RY_FRAC_HEAD_RY <= 0.16


def test_t1_invert_0085_leftover() -> None:
    """T1: flatten/raise vs leftover 0085 goggle 0.95 + pin lip 0.24/0.022/0.016 + mix 0.50."""
    assert EYE_RY_FRAC_R < 0.95
    assert face_recipe_mod._LIP_Z_FRAC > 0.24
    assert LIP_RY_FRAC_H > 0.022
    assert LIP_RZ_FRAC_H > 0.016
    assert CHEEK_Z_MIX < 0.50
    assert CHEEK_RZ_FRAC_H < 0.06
    assert CHEEK_RY_FRAC_HEAD_RY < 0.22


def test_t2_product_class_eye_shelf() -> None:
    """T2: product-class emit — eye.ry == 0.62*eye.rx (rx still 1.00*0.11*H); ry < rx."""
    pkg = _product_pkg()
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    eye = next(p for p in pkg.parts if p.name == "RECIPE_eye_soft_l")
    assert eye.rx_m is not None and eye.ry_m is not None
    assert float(eye.ry_m) == pytest.approx(0.62 * float(eye.rx_m), abs=1e-4)
    assert float(eye.ry_m) < float(eye.rx_m)
    h = _head_h(head)
    assert float(eye.rx_m) == pytest.approx(1.00 * 0.11 * h, abs=1e-4)


def test_t3_lip_shelf_below_sausage() -> None:
    """T3: lip ry/rz/rx from H; stay below 0078 sausage ceilings."""
    bounds = _product_class_bounds()
    parts = build_face_parts(_full_torso_report(), bounds, face=True, messages=[])
    lip = next(p for p in parts if p.name == "RECIPE_lip_soft")
    h = bounds.H
    assert lip.ry_m is not None and lip.rz_m is not None and lip.rx_m is not None
    assert float(lip.ry_m) == pytest.approx(0.028 * h, abs=1e-6)
    assert float(lip.rz_m) == pytest.approx(0.020 * h, abs=1e-6)
    assert float(lip.rx_m) == pytest.approx(0.10 * h, abs=1e-6)
    assert float(lip.ry_m) < _SAUSAGE_RY * h
    assert float(lip.rz_m) < _SAUSAGE_RZ * h


def test_t4_unpitched_lip_z_shelf() -> None:
    """T4: unpitched lip z = z_chin + 0.28*H; still below nose_base. Do not pin 1.55."""
    bounds = _product_class_bounds()
    parts = build_face_parts(_full_torso_report(), bounds, face=True, messages=[])
    lip = next(p for p in parts if p.name == "RECIPE_lip_soft")
    assert lip.center is not None
    h = bounds.H
    z_chin = bounds.z_chin
    assert float(lip.center[2]) == pytest.approx(z_chin + 0.28 * h, abs=2e-3)
    assert float(lip.center[2]) < z_chin + 0.33 * h
    assert float(lip.center[2]) > z_chin + 0.24 * h


def test_t5_unpitched_cheek_below_orbit() -> None:
    """T5: invert pitch like 0058 cheek_center; cheek_top ≤ eye_bottom + 2 mm."""
    pkg = _product_pkg()
    eye = next(p for p in pkg.parts if p.name == "RECIPE_eye_soft_l")
    cheek = next(p for p in pkg.parts if p.name == "RECIPE_cheek_soft_l")
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    assert eye.center is not None and cheek.center is not None
    assert eye.rz_m is not None and cheek.rz_m is not None
    assert head.center is not None and head.rz_m is not None
    assert neck.p1 is not None
    pivot = [float(neck.p1[0]), float(neck.p1[1]), float(neck.p1[2])]
    th = -math.radians(HEAD_PITCH_DEG)
    eye_pre = _rotate_yz_about_x(list(eye.center), pivot, th)
    cheek_pre = _rotate_yz_about_x(list(cheek.center), pivot, th)
    head_pre = _rotate_yz_about_x(list(head.center), pivot, th)
    h = 2.0 * float(head.rz_m)
    z_chin = float(head_pre[2]) - float(head.rz_m)
    eye_z = z_chin + 0.50 * h
    nose_base_z = z_chin + 0.33 * h
    expected_z = CHEEK_Z_MIX * eye_z + (1.0 - CHEEK_Z_MIX) * nose_base_z
    assert float(cheek_pre[2]) == pytest.approx(expected_z, abs=2e-3)
    eye_bottom = float(eye_pre[2]) - float(eye.rz_m)
    cheek_top = float(cheek_pre[2]) + float(cheek.rz_m)
    assert cheek_top <= eye_bottom + 0.002


def test_t6_outer_x_inside_head() -> None:
    """T6: 0085 B1 outer-X math unchanged — flatten RY does not grow X."""
    pkg = _product_pkg()
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    eye = next(p for p in pkg.parts if p.name == "RECIPE_eye_soft_l")
    assert head.rx_m is not None and eye.center is not None and eye.rx_m is not None
    assert abs(float(eye.center[0])) + float(eye.rx_m) < float(head.rx_m)


def test_t6b_sibling_messages() -> None:
    """T6b: 0085 hierarchy still interpolates; new defaults sibling once."""
    pkg = _product_pkg()
    hier = [m for m in pkg.messages if _HIER_PREFIX in m]
    assert len(hier) == 1
    assert f"r={EYE_RADIUS_FRAC_H}" in hier[0]
    assert f"rz={EYE_RZ_FRAC_R}" in hier[0]
    assert f"lip_z={face_recipe_mod._LIP_Z_FRAC}" in hier[0]
    assert f"pitch={HEAD_PITCH_DEG}" in hier[0]
    hits = [m for m in pkg.messages if _DEFAULTS_PREFIX in m]
    assert len(hits) == 1
    line = hits[0]
    assert f"eye_ry={EYE_RY_FRAC_R}" in line
    assert f"lip_z={face_recipe_mod._LIP_Z_FRAC}" in line
    assert f"lip_ry={LIP_RY_FRAC_H}" in line
    assert f"cheek_mix={CHEEK_Z_MIX}" in line
    cheek_i = next(i for i, m in enumerate(pkg.messages) if "cheek soft pads present" in m)
    def_i = next(i for i, m in enumerate(pkg.messages) if _DEFAULTS_PREFIX in m)
    assert def_i > cheek_i


def test_t7_missing_face_skips_features() -> None:
    """T7: missing --face → no eye/lip/cheek (existing skip)."""
    pkg = _product_pkg(face=False)
    names = {p.name for p in pkg.parts}
    assert not any("eye_soft" in n for n in names)
    assert "RECIPE_lip_soft" not in names
    assert not any("cheek_soft" in n for n in names)


def test_t8_n_parts_schema_mcp() -> None:
    """T8: n_parts 131; schema 1.4.0; MCP 47."""
    pkg = _product_pkg()
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert pkg.schema_version == "1.4.0"
    assert len(TOOL_NAMES) == 47


def test_t9_public_exports_no_new_name() -> None:
    """T9: existing public names stay; _LIP_Z_FRAC stays private."""
    for name in (
        "EYE_RY_FRAC_R",
        "LIP_RY_FRAC_H",
        "LIP_RZ_FRAC_H",
        "CHEEK_Z_MIX",
        "CHEEK_RZ_FRAC_H",
        "CHEEK_RY_FRAC_HEAD_RY",
    ):
        assert name in face_recipe_mod.__all__
    assert "_LIP_Z_FRAC" not in face_recipe_mod.__all__


def test_t10_jaw_0078_hold() -> None:
    """T10: 0078 jaw ry/rz/bulge hold; FEATURE_Y 0.90."""
    assert JAW_RY_FRAC_HEAD_RY == 0.42
    assert JAW_RZ_FRAC_H == 0.13
    assert FEATURE_FACE_Y_FRAC_RY == 0.90
    bounds = _product_class_bounds()
    msgs: list[str] = []
    parts = build_face_parts(_full_torso_report(), bounds, face=True, messages=msgs)
    jaw = next(p for p in parts if p.name == "RECIPE_jaw")
    assert jaw.center is not None and jaw.rz_m is not None
    assert float(jaw.center[2]) - float(jaw.rz_m) == pytest.approx(bounds.z_chin, abs=0.002)
    bulge_line = next(m for m in msgs if "jaw_vs_head_x_bulge_m=" in m)
    bulge = float(bulge_line.split("jaw_vs_head_x_bulge_m=", 1)[1].split()[0])
    assert bulge <= JAW_X_BULGE_ALLOW_M + EPS


def test_t11_0085_0050_0086_hold() -> None:
    """T11: 0085 radius/RZ/pitch; 0050 12°; 0086 nape 0.018."""
    assert EYE_RADIUS_FRAC_H == 0.11
    assert EYE_RZ_FRAC_R == 0.58
    assert HEAD_PITCH_DEG == 6.0
    assert NECK_FORWARD_TILT_DEG == 12.0
    assert NECK_NAPE_SETBACK_M == 0.018


def test_t12_compact_culls_eye_lip() -> None:
    """T12: compact still culls eye_soft/lip_soft (0082 B4); jaw stays. Do not un-cull."""
    pkg = _product_pkg(soft_density="compact")
    names = {p.name for p in pkg.parts}
    assert not any("eye_soft" in n for n in names)
    assert not any("lip_soft" in n for n in names)
    assert "RECIPE_jaw" in names


def test_t13_no_unnamed_axis_gt_one() -> None:
    """T13: 0119 analog — named EYE_RY < 1; RX == 1.00; RZ < 1. No unnamed axis >1."""
    assert EYE_RY_FRAC_R < 1.0
    assert EYE_RX_FRAC_R == 1.00
    assert EYE_RZ_FRAC_R == 0.58
    assert EYE_RZ_FRAC_R < 1.0


def test_t14_both_eyes_lip_cheeks() -> None:
    """T14: both eyes + lip + both cheeks; eye.ry < eye.rx; cheek.z < eye.z."""
    pkg = _product_pkg()
    names = {p.name for p in pkg.parts}
    assert "RECIPE_eye_soft_l" in names
    assert "RECIPE_eye_soft_r" in names
    assert "RECIPE_lip_soft" in names
    assert "RECIPE_cheek_soft_l" in names
    assert "RECIPE_cheek_soft_r" in names
    eye = next(p for p in pkg.parts if p.name == "RECIPE_eye_soft_l")
    cheek = next(p for p in pkg.parts if p.name == "RECIPE_cheek_soft_l")
    assert eye.rx_m is not None and eye.ry_m is not None
    assert eye.center is not None and cheek.center is not None
    assert float(eye.ry_m) < float(eye.rx_m)
    assert float(cheek.center[2]) < float(eye.center[2])
