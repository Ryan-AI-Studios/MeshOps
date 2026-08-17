"""Track 0088 — hand digit taper plus (stronger tip + pinky hierarchy).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / skeleton 1.0.0 / MCP 46 stay. Not mesh/print success.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    RECIPE_SCHEMA_VERSION,
    build_blockout_recipe,
)
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.extremity_recipe import (
    _FINGER_DIGIT_L_SCALE,
    _FINGER_DIGIT_R_SCALE,
    _FINGER_MIN_CENTER_SPACING_VS_R,
    _FINGER_NAMES,
    _FINGER_R_CAP_VS_HALF_W,
    _FINGER_R_FLOOR_M,
    _FINGER_R_FRAC_PALM,
    _FINGER_R_SCALES_SEG,
    _FINGER_SEG_FRACS_HAND,
    _FINGER_SPLAY_FRAC_HALF_W,
    _MITTEN_R_FRAC_PALM,
    _PALM_PAD_RY_FRAC_TH,
    _PALM_THICKNESS_FRAC_HAND,
    _PALM_WIDTH_FRAC_HAND,
    _THUMB_DISTAL_L_SCALE,
    _THUMB_DISTAL_R_SCALE,
    _THUMB_PALM_PITCH,
    _THUMB_R_SCALE_VS_FINGER,
    _fingertip_y_trusted,
    finger_primary_axis,
)
from meshops.proportion.models import (
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import (
    ELBOW_HANG_T,
    SKELETON_SCHEMA_VERSION,
    _arm_forward_y,
    _elbow_hang_y,
    build_blockout_skeleton,
)


def _lm(
    id_: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=id_, x_m=x_m, y_m=y_m, z_m=z_m)


def _band(
    band_id: str,
    *,
    y_mid: float = 0.0,
    depth_m: float | None = None,
    depth_frac: float = 0.06,
) -> DepthBand:
    return DepthBand(
        band_id=band_id,
        depth_px=20.0,
        depth_frac=depth_frac,
        depth_m=depth_m,
        y_front=(y_mid + depth_frac / 2.0),
        y_back=(y_mid - depth_frac / 2.0),
        y_mid=y_mid,
        z_frac=None,
        confidence=0.8,
        sources=["left"],
        orientation_swapped=False,
    )


def _diam(band_id: str, *, half_width_m: float = 0.05) -> DiameterMeasure:
    return DiameterMeasure(
        band_id=band_id,
        view="front",
        width_px=40.0,
        width_eucl_px=40.0,
        theta_deg=90.0,
        width_frac=0.1,
        width_m=half_width_m * 2.0,
        half_width_m=half_width_m,
        mid_x_px=100.0,
        mid_y_px=200.0,
    )


def _report(
    lms: dict[str, LandmarkXYZ] | None = None,
    *,
    height_m: float | None = 1.72,
    depth_bands: list[DepthBand] | None = None,
    diameters: list[DiameterMeasure] | None = None,
) -> ProportionReport:
    return ProportionReport(
        schema_version="1.2.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms if lms is not None else {},
        depth_bands=list(depth_bands or []),
        diameters=list(diameters or []),
        quality=QualityFlags(),
    )


def _product_class_report(*, height_m: float = 1.72) -> ProportionReport:
    h = height_m
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=1.72),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.38),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=None, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=None, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=None, z_m=0.90),
        "hand_l": _lm("hand_l", x_m=-0.33, y_m=None, z_m=0.85),
        "hand_r": _lm("hand_r", x_m=0.33, y_m=None, z_m=0.85),
        "fingertip_l": _lm("fingertip_l", x_m=-0.36, y_m=None, z_m=0.72),
        "fingertip_r": _lm("fingertip_r", x_m=0.36, y_m=None, z_m=0.72),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.1303, z_m=1.25),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
        "heel_l": _lm("heel_l", x_m=-0.10, y_m=0.06, z_m=0.02),
        "heel_r": _lm("heel_r", x_m=0.10, y_m=0.06, z_m=0.02),
        "toe_l": _lm("toe_l", x_m=-0.10, y_m=-0.12, z_m=0.02),
        "toe_r": _lm("toe_r", x_m=0.10, y_m=-0.12, z_m=0.02),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.0438),
        _diam("upper_arm_r", half_width_m=0.0438),
        _diam("forearm_l", half_width_m=0.0350),
        _diam("forearm_r", half_width_m=0.0350),
        _diam("thigh_l", half_width_m=0.0613),
        _diam("thigh_r", half_width_m=0.0613),
        _diam("calf_l", half_width_m=0.0438),
        _diam("calf_r", half_width_m=0.0438),
        _diam("ank_foot_l", half_width_m=0.0263),
        _diam("ank_foot_r", half_width_m=0.0263),
    ]
    bands = [
        _band("chest", y_mid=0.0, depth_m=0.2606),
        _band("breast", depth_m=0.18),
        _band("hip", depth_m=0.26),
        _band("glute", depth_m=0.22),
    ]
    return _report(lms, height_m=h, depth_bands=bands, diameters=diams)


def _product_flags(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "limbs": True,
        "torso": "ovals",
        "glute": "two_spheres",
        "nofuse": True,
        "face": True,
        "hair": "short",
        "hands": True,
        "feet": True,
        "fingers": "full",
        "toes": "full",
        "profile": load_anatomy_profile("torso_limb_f_athletic_v1"),
    }
    base.update(overrides)
    return base


def _report_with_extremities(*, height_m: float = 1.72) -> ProportionReport:
    """Copy of 0079 helper: torso + wrist/hand/tip + ankle/heel/toe."""
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "hip_l": _lm("hip_l", x_m=-0.14, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.14, y_m=0.0, z_m=0.95),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.08, z_m=1.25),
        "wrist_l": _lm("wrist_l", x_m=-0.45, y_m=0.0, z_m=0.95),
        "wrist_r": _lm("wrist_r", x_m=0.45, y_m=0.0, z_m=0.95),
        "hand_l": _lm("hand_l", x_m=-0.48, y_m=0.0, z_m=0.88),
        "hand_r": _lm("hand_r", x_m=0.48, y_m=0.0, z_m=0.88),
        "fingertip_l": _lm("fingertip_l", x_m=-0.50, y_m=0.0, z_m=0.72),
        "fingertip_r": _lm("fingertip_r", x_m=0.50, y_m=0.0, z_m=0.72),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.02, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.02, z_m=0.08),
        "heel_l": _lm("heel_l", x_m=-0.10, y_m=0.06, z_m=0.02),
        "heel_r": _lm("heel_r", x_m=0.10, y_m=0.06, z_m=0.02),
        "toe_l": _lm("toe_l", x_m=-0.10, y_m=-0.12, z_m=0.02),
        "toe_r": _lm("toe_r", x_m=0.10, y_m=-0.12, z_m=0.02),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.05),
        _diam("upper_arm_r", half_width_m=0.05),
        _diam("thigh_l", half_width_m=0.05),
        _diam("thigh_r", half_width_m=0.05),
        _diam("ank_foot_l", half_width_m=0.035),
        _diam("ank_foot_r", half_width_m=0.035),
    ]
    bands = [
        _band("chest", depth_m=0.24),
        _band("hip", depth_m=0.26),
    ]
    return _report(lms, height_m=height_m, depth_bands=bands, diameters=diams)


def _rake_report(*, wrist_y: float = -0.0586, tip_y: float | None = None) -> ProportionReport:
    lms = {
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=wrist_y, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=wrist_y, z_m=0.90),
        "fingertip_l": _lm("fingertip_l", x_m=-0.36, y_m=tip_y, z_m=0.72),
        "fingertip_r": _lm("fingertip_r", x_m=0.36, y_m=tip_y, z_m=0.72),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
    }
    return _report(lms)


def _seg_length(part: object) -> float:
    p0 = getattr(part, "p0", None)
    p1 = getattr(part, "p1", None)
    assert p0 is not None and p1 is not None
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    dz = float(p1[2]) - float(p0[2])
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _require_radius_m(part: object) -> float:
    radius_m = getattr(part, "radius_m", None)
    assert radius_m is not None
    return float(radius_m)


def _capsule_center(part: object) -> list[float]:
    p0 = getattr(part, "p0", None)
    p1 = getattr(part, "p1", None)
    assert p0 is not None and p1 is not None
    return [
        0.5 * (float(p0[0]) + float(p1[0])),
        0.5 * (float(p0[1]) + float(p1[1])),
        0.5 * (float(p0[2]) + float(p1[2])),
    ]


def _hand_len_from_lms(lms: dict[str, LandmarkXYZ], side: str = "l") -> float:
    w = lms[f"wrist_{side}"]
    t = lms[f"fingertip_{side}"]
    assert w.x_m is not None and w.z_m is not None
    assert t.x_m is not None and t.z_m is not None
    wy = float(w.y_m) if w.y_m is not None else 0.0
    ty = float(t.y_m) if t.y_m is not None else 0.0
    dx = float(t.x_m) - float(w.x_m)
    dy = ty - wy
    dz = float(t.z_m) - float(w.z_m)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _finger(pkg: Any, fname: str, si: int, side: str) -> Any:
    return next(p for p in pkg.parts if p.name == f"RECIPE_finger_{fname}_{si}_{side}")


def test_t0_0088_constants() -> None:
    """T0: 0088 r/seg/digit/thumb freezes + B6 CAP/thumb-r0 fence."""
    assert _FINGER_R_SCALES_SEG == (1.00, 0.86, 0.72)
    assert 0.70 <= _FINGER_R_SCALES_SEG[2] <= 0.74
    assert 0.84 <= _FINGER_R_SCALES_SEG[1] <= 0.88
    assert _FINGER_R_SCALES_SEG[0] == 1.00
    assert _FINGER_SEG_FRACS_HAND == (0.27, 0.18, 0.10)
    assert abs(sum(_FINGER_SEG_FRACS_HAND) - 0.55) < 1e-12
    assert 0.09 <= _FINGER_SEG_FRACS_HAND[2] <= 0.11
    assert _FINGER_DIGIT_L_SCALE["pinky"] == pytest.approx(0.80)
    assert _FINGER_DIGIT_R_SCALE["pinky"] == pytest.approx(0.78)
    assert _FINGER_DIGIT_L_SCALE["index"] == pytest.approx(0.96)
    assert _FINGER_DIGIT_L_SCALE["ring"] == pytest.approx(0.96)
    assert _FINGER_DIGIT_R_SCALE["index"] == pytest.approx(0.94)
    assert _FINGER_DIGIT_R_SCALE["ring"] == pytest.approx(0.96)
    assert 0.78 <= _FINGER_DIGIT_L_SCALE["pinky"] <= 0.84
    assert 0.74 <= _FINGER_DIGIT_R_SCALE["pinky"] <= 0.82
    assert pytest.approx(0.72) == _THUMB_DISTAL_L_SCALE
    assert pytest.approx(0.80) == _THUMB_DISTAL_R_SCALE
    assert pytest.approx(0.16) == _FINGER_R_FRAC_PALM
    assert pytest.approx(0.55) == _FINGER_R_CAP_VS_HALF_W
    assert pytest.approx(1.25) == _THUMB_R_SCALE_VS_FINGER
    assert pytest.approx(1.95) == _FINGER_SPLAY_FRAC_HALF_W
    assert pytest.approx(0.36) == _PALM_THICKNESS_FRAC_HAND
    assert pytest.approx(0.78) == _PALM_PAD_RY_FRAC_TH
    assert pytest.approx(-0.55) == _THUMB_PALM_PITCH
    assert pytest.approx(0.006) == _FINGER_R_FLOOR_M
    import meshops.proportion.extremity_recipe as ext

    assert not hasattr(ext, "_FINGER_DISTAL_R_SCALE")


def test_t1_middle_r_ratios() -> None:
    """T1: product-like middle r2/r0 == 0.72; r1/r0 == 0.86; r2 < r1 < r0."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        r0 = _require_radius_m(_finger(pkg, "middle", 0, side))
        r1 = _require_radius_m(_finger(pkg, "middle", 1, side))
        r2 = _require_radius_m(_finger(pkg, "middle", 2, side))
        assert r2 < r1 < r0
        assert r2 / r0 == pytest.approx(_FINGER_R_SCALES_SEG[2], abs=1e-9)
        assert r1 / r0 == pytest.approx(_FINGER_R_SCALES_SEG[1], abs=1e-9)


def test_t2_middle_seg_lengths() -> None:
    """T2: middle L follows segs 0.27/0.18/0.10 * hand_len; PP > MP > DP."""
    report = _report_with_extremities()
    hand_len = _hand_len_from_lms(report.landmarks_xyz)
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        lengths = [_seg_length(_finger(pkg, "middle", si, side)) for si in range(3)]
        assert lengths[0] > lengths[1] > lengths[2]
        for si, frac in enumerate(_FINGER_SEG_FRACS_HAND):
            assert lengths[si] == pytest.approx(frac * hand_len, abs=1e-9)


def test_t3_four_digit_anti_stick() -> None:
    """T3: all four digits distal L/r < 1.50 and < 0.75 * prox L/r."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        for fname in _FINGER_NAMES:
            prox = _finger(pkg, fname, 0, side)
            dist = _finger(pkg, fname, 2, side)
            l0, r0 = _seg_length(prox), _require_radius_m(prox)
            l2, r2 = _seg_length(dist), _require_radius_m(dist)
            assert r0 > 0.0 and r2 > 0.0
            lr_dist = l2 / r2
            lr_prox = l0 / r0
            assert lr_dist < 1.50, f"{fname}_{side} lr_dist={lr_dist}"
            assert lr_dist < 0.75 * lr_prox, (
                f"{fname}_{side} lr_dist={lr_dist} vs 0.75*lr_prox={0.75 * lr_prox}"
            )


def test_t4_pinky_hierarchy() -> None:
    """T4: pinky total L < middle; pinky L ~ 0.80x; pinky r0 ~ 0.78x."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        mid_l = sum(_seg_length(_finger(pkg, "middle", si, side)) for si in range(3))
        pinky_l = sum(_seg_length(_finger(pkg, "pinky", si, side)) for si in range(3))
        mid_r0 = _require_radius_m(_finger(pkg, "middle", 0, side))
        pinky_r0 = _require_radius_m(_finger(pkg, "pinky", 0, side))
        assert pinky_l < mid_l
        assert pinky_l == pytest.approx(_FINGER_DIGIT_L_SCALE["pinky"] * mid_l, abs=1e-9)
        assert pinky_r0 == pytest.approx(_FINGER_DIGIT_R_SCALE["pinky"] * mid_r0, abs=1e-9)


def test_t5_thumb_distal_and_r0() -> None:
    """T5: thumb L1 == 0.72*L0; r1 == 0.80*r0; r0 == 1.25*fr (not ratio-only)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        t0 = next(p for p in pkg.parts if p.name == f"RECIPE_thumb_soft_0_{side}")
        t1 = next(p for p in pkg.parts if p.name == f"RECIPE_thumb_soft_1_{side}")
        l0, l1 = _seg_length(t0), _seg_length(t1)
        r0, r1 = _require_radius_m(t0), _require_radius_m(t1)
        fr = _require_radius_m(_finger(pkg, "middle", 0, side))
        assert l1 == pytest.approx(_THUMB_DISTAL_L_SCALE * l0, abs=1e-9)
        assert r1 == pytest.approx(_THUMB_DISTAL_R_SCALE * r0, abs=1e-9)
        assert r0 == pytest.approx(_THUMB_R_SCALE_VS_FINGER * fr, abs=1e-9)


def test_t6_c1_no_flatten() -> None:
    """T6 C1: no n[1] > 0 flatten; trusted tip stays wrist->tip; hang axis[1] <= 0."""
    src = inspect.getsource(finger_primary_axis)
    assert "if n[1] > 0" not in src
    assert "if n[1] > 0.0" not in src
    wrist = [0.0, -0.0586, 1.0]
    tip = [0.1, -0.20, 0.7]
    assert _fingertip_y_trusted(wrist[1], tip[1]) is True
    axis = finger_primary_axis(wrist, tip, hand_len=0.3)
    raw = (tip[0] - wrist[0], tip[1] - wrist[1], tip[2] - wrist[2])
    n = math.sqrt(raw[0] ** 2 + raw[1] ** 2 + raw[2] ** 2)
    expected = (raw[0] / n, raw[1] / n, raw[2] / n)
    assert axis[0] == pytest.approx(expected[0], abs=1e-6)
    assert axis[1] == pytest.approx(expected[1], abs=1e-6)
    assert axis[2] == pytest.approx(expected[2], abs=1e-6)
    hang = finger_primary_axis(wrist, None, hand_len=0.12)
    assert hang[1] <= 0.0
    assert hang[2] < 0.0


def test_t7_messages_const_driven() -> None:
    """T7: segs/r_scales/digit_L pinky/thumb_distal const-driven; mitten silent."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    bulk = [m for m in pkg.messages if "hand bulk: full digits" in m]
    taper = [m for m in pkg.messages if "hand taper:" in m]
    assert len(bulk) == 1
    assert len(taper) == 1
    assert f"segs={_FINGER_SEG_FRACS_HAND}" in bulk[0]
    assert "anti-stick" in bulk[0]
    assert f"r_scales={_FINGER_R_SCALES_SEG}" in taper[0]
    assert "digit_L=" in taper[0]
    assert f"'pinky': {_FINGER_DIGIT_L_SCALE['pinky']}" in taper[0]
    assert f"thumb_distal={_THUMB_DISTAL_L_SCALE}/{_THUMB_DISTAL_R_SCALE}" in taper[0]
    assert "anti-sausage" in taper[0]
    pkg_mitt = build_blockout_recipe(report, limbs=False, hands=True, fingers="mitten")
    assert not any("hand bulk: full digits" in m for m in pkg_mitt.messages)
    assert not any("hand taper:" in m for m in pkg_mitt.messages)


def test_t8_hang_palm_fence() -> None:
    """T8: untrusted tip -> palm Y == wrist Y; hang axis=hang pitch=-0.55."""
    wrist_y = -0.0586
    report = _rake_report(wrist_y=wrist_y, tip_y=None)
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert palm.center is not None
        assert float(palm.center[1]) == pytest.approx(wrist_y, abs=1e-6)
    hang = [m for m in pkg.messages if "hand hang:" in m]
    assert hang
    assert any("axis=hang" in m and "pitch=-0.55" in m for m in hang)


def test_t9_surface_n_parts() -> None:
    """T9: n_parts 131; schema 1.4.0; MCP 46; C_palm_ellipsoid pass."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_palm_ellipsoid"].status == "pass", by_id["C_palm_ellipsoid"].message
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert SKELETON_SCHEMA_VERSION == "1.0.0"
    assert len(TOOL_NAMES) == 46


def test_t10_product_composition_const_driven() -> None:
    """T10: fr = r_frac * palm_w; r1/r2 follow scales; L2/r2 < 1.50.

    H=1.72 product expect comment-only: r ~ 0.01200 / 0.01032 / 0.00864,
    L/r tip ~ 1.400. Do not hard-assert those literals (0094 T10 lesson).
    """
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    for side in ("l", "r"):
        p0 = _finger(pkg, "middle", 0, side)
        p1 = _finger(pkg, "middle", 1, side)
        p2 = _finger(pkg, "middle", 2, side)
        l0 = _seg_length(p0)
        l2 = _seg_length(p2)
        r0 = _require_radius_m(p0)
        r1 = _require_radius_m(p1)
        r2 = _require_radius_m(p2)
        hand_len = l0 / _FINGER_SEG_FRACS_HAND[0]
        palm_w = _PALM_WIDTH_FRAC_HAND * hand_len
        half_w = palm_w / 2.0
        fr = min(
            max(_FINGER_R_FRAC_PALM * palm_w, _FINGER_R_FLOOR_M),
            _FINGER_R_CAP_VS_HALF_W * half_w,
        )
        assert r0 == pytest.approx(fr * _FINGER_R_SCALES_SEG[0], abs=1e-9)
        assert r1 == pytest.approx(fr * _FINGER_R_SCALES_SEG[1], abs=1e-9)
        assert r2 == pytest.approx(fr * _FINGER_R_SCALES_SEG[2], abs=1e-9)
        assert l2 / r2 < 1.50


def test_t11_b16_floor_bind() -> None:
    """T11: short-hand B16 floor still >= 0.006 (pinky distal 0.72*0.78)."""
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "hip_l": _lm("hip_l", x_m=-0.14, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.14, y_m=0.0, z_m=0.95),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "wrist_l": _lm("wrist_l", x_m=-0.45, y_m=0.0, z_m=0.95),
        "wrist_r": _lm("wrist_r", x_m=0.45, y_m=0.0, z_m=0.95),
        "hand_l": _lm("hand_l", x_m=-0.46, y_m=0.0, z_m=0.93),
        "hand_r": _lm("hand_r", x_m=0.46, y_m=0.0, z_m=0.93),
        "fingertip_l": _lm("fingertip_l", x_m=-0.47, y_m=0.0, z_m=0.91),
        "fingertip_r": _lm("fingertip_r", x_m=0.47, y_m=0.0, z_m=0.91),
    }
    report = _report(lms)
    hand_len = _hand_len_from_lms(report.landmarks_xyz)
    assert 0.035 < hand_len < 0.060
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        pinky_2 = _finger(pkg, "pinky", 2, side)
        assert _require_radius_m(pinky_2) >= _FINGER_R_FLOOR_M - 1e-12
        for si in range(2):
            thumb = next(p for p in pkg.parts if p.name == f"RECIPE_thumb_soft_{si}_{side}")
            assert _require_radius_m(thumb) >= _FINGER_R_FLOOR_M - 1e-12


def test_t12_sides_equalize() -> None:
    """T12: L/R middle r match; L/R pinky r match."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for fname in ("middle", "pinky"):
        for si in range(3):
            rl = _require_radius_m(_finger(pkg, fname, si, "l"))
            rr = _require_radius_m(_finger(pkg, fname, si, "r"))
            assert rl == pytest.approx(rr, abs=1e-9), f"{fname}_{si} L={rl} R={rr}"


def test_t13_mitten_untouched() -> None:
    """T13: mitten r frac 0.72; no extra finger capsules."""
    assert pytest.approx(0.72) == _MITTEN_R_FRAC_PALM
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="mitten")
    names = {p.name for p in pkg.parts}
    assert "RECIPE_finger_mitten_l" in names
    assert "RECIPE_finger_mitten_r" in names
    for fname in _FINGER_NAMES:
        for si in range(3):
            assert f"RECIPE_finger_{fname}_{si}_l" not in names
            assert f"RECIPE_finger_{fname}_{si}_r" not in names


def test_t14_0087_elbow_hang_fence() -> None:
    """T14: product-class elbow y == hang T 0.50; wrist distal; palm y == wrist."""
    h = 1.72
    half = 0.1303
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    by_id = {j.id: j for j in skel.joints}
    expected = _arm_forward_y(0.0, half_depth=half, height_m=h, chest_front_y=-half)
    hang = _elbow_hang_y(0.0, expected, t=ELBOW_HANG_T)
    assert pytest.approx(0.50) == ELBOW_HANG_T
    assert by_id["elbow_l"].y_m == pytest.approx(hang, abs=1e-4)
    assert by_id["elbow_r"].y_m == pytest.approx(hang, abs=1e-4)
    assert by_id["wrist_l"].y_m == pytest.approx(expected, abs=1e-4)
    assert by_id["wrist_r"].y_m == pytest.approx(expected, abs=1e-4)
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert palm.center is not None
        wrist_y = by_id[f"wrist_{side}"].y_m
        assert wrist_y is not None
        assert float(palm.center[1]) == pytest.approx(float(wrist_y), abs=1e-6)


def test_t15_groove_vs_base_fr() -> None:
    """T15: adjacent finger_0 spacing >= 2x base fr (not post-scale pinky r)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        segs = [_finger(pkg, fname, 0, side) for fname in _FINGER_NAMES]
        fr = _require_radius_m(_finger(pkg, "middle", 0, side))
        xs = [_capsule_center(p)[0] for p in segs]
        min_dx = min(abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
        assert min_dx >= _FINGER_MIN_CENTER_SPACING_VS_R * fr - 1e-6, (
            f"side={side} min_dx={min_dx} base_fr={fr}"
        )
