"""Track 0084 — Hand hang pitch (no back-sloped digits).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY / SKELETON_HONESTY).
Schema 1.4.0 / skeleton 1.0.0 / MCP 46 stay.
"""

from __future__ import annotations

import math

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    RECIPE_SCHEMA_VERSION,
    build_blockout_recipe,
)
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.extremity_recipe import (
    _FINGER_R_FRAC_PALM,
    _FINGER_R_SCALES_SEG,
    _FINGER_SEG_FRACS_HAND,
    _FINGER_SPLAY_FRAC_HALF_W,
    _FINGERTIP_PLANE_MARGIN_M,
    _THUMB_PALM_PITCH,
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
    SKELETON_SCHEMA_VERSION,
    _arm_forward_y,
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


def _rake_report(*, wrist_y: float = -0.0586, tip_y: float | None = None) -> ProportionReport:
    """No-skel wrists at distal prior with plane-class (or given) fingertips."""
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


def _by_id(pkg):  # type: ignore[no-untyped-def]
    return {j.id: j for j in pkg.joints}


def test_t0_const() -> None:
    """T0: thumb pitch -0.55; helper + 0.02 margin; finger_primary_axis importable."""
    assert pytest.approx(-0.55) == _THUMB_PALM_PITCH
    assert pytest.approx(0.02) == _FINGERTIP_PLANE_MARGIN_M
    assert callable(_fingertip_y_trusted)
    assert callable(finger_primary_axis)
    assert not _fingertip_y_trusted(-0.0586, None)
    assert not _fingertip_y_trusted(-0.0586, 0.0)
    assert not _fingertip_y_trusted(-0.0586, -0.02)
    assert not _fingertip_y_trusted(None, -0.20)
    assert _fingertip_y_trusted(-0.0586, -0.20)
    assert not _fingertip_y_trusted(-0.30, -0.20)  # rear rake (tip +Y of wrist)


def test_t1_plane_tip_hangs() -> None:
    """T1: wrist Y=-0.0586, tip Y=None/0 → hang; axis[1]<=0; |z|>|y|; helper False."""
    wrist = [0.0, -0.0586, 1.0]
    for tip in (None, [0.1, 0.0, 0.7]):
        tip_y = None if tip is None else tip[1]
        assert _fingertip_y_trusted(wrist[1], tip_y) is False
        axis = finger_primary_axis(wrist, tip, hand_len=0.22)
        assert axis[1] <= 0.0
        assert axis[2] < 0.0
        assert abs(axis[2]) > abs(axis[1])


def test_t2_all_digit_p1_le_palm() -> None:
    """T2: every finger/thumb capsule p1.y <= palm.cy (L/R)."""
    report = _rake_report()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    fingers = ("index", "middle", "ring", "pinky")
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert palm.center is not None
        palm_cy = float(palm.center[1])
        for fname in fingers:
            for si in range(3):
                part = next(p for p in pkg.parts if p.name == f"RECIPE_finger_{fname}_{si}_{side}")
                assert part.p1 is not None
                assert float(part.p1[1]) <= palm_cy + 1e-6, (
                    f"{part.name} p1.y={part.p1[1]} palm={palm_cy}"
                )
        for si in range(2):
            thumb = next(p for p in pkg.parts if p.name == f"RECIPE_thumb_soft_{si}_{side}")
            assert thumb.p1 is not None
            assert float(thumb.p1[1]) <= palm_cy + 1e-6, (
                f"{thumb.name} p1.y={thumb.p1[1]} palm={palm_cy}"
            )


def test_t3_thumb_dir_y_negative() -> None:
    """T3: thumb dir.y < 0 (face -Y pitch)."""
    report = _rake_report()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        thumb = next(p for p in pkg.parts if p.name == f"RECIPE_thumb_soft_0_{side}")
        assert thumb.p0 is not None and thumb.p1 is not None
        dy = float(thumb.p1[1]) - float(thumb.p0[1])
        assert dy < 0.0, f"side={side} thumb dir.y={dy}"


def test_t4_mitten_center_le_palm() -> None:
    """T4: mitten center y <= palm.cy."""
    report = _rake_report()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="mitten")
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        mitt = next(p for p in pkg.parts if p.name == f"RECIPE_finger_mitten_{side}")
        assert palm.center is not None and mitt.center is not None
        assert float(mitt.center[1]) <= float(palm.center[1]) + 1e-6


def test_t5_palm_follows_wrist_not_halfway() -> None:
    """T5: no-skel plane-class tip → palm cy = wrist y (not mid toward 0)."""
    wrist_y = -0.0586
    report = _rake_report(wrist_y=wrist_y, tip_y=None)
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert palm.center is not None
        assert float(palm.center[1]) == pytest.approx(wrist_y, abs=1e-6)
        # Not halfway to invented tip Y=0
        assert abs(float(palm.center[1]) - (wrist_y / 2.0)) > 0.02


def test_t6_trusted_tip_still_wrist_tip() -> None:
    """T6: tip y=-0.20 stays wrist→tip (0029 B7 trusted path)."""
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


def test_t7_0079_fence() -> None:
    """T7: 0079 middle r0>r1>r2; segs sum 0.55; fr 0.16; splay 1.95."""
    assert _FINGER_SEG_FRACS_HAND == (0.25, 0.18, 0.12)
    assert abs(sum(_FINGER_SEG_FRACS_HAND) - 0.55) < 1e-12
    assert _FINGER_R_SCALES_SEG == (1.00, 0.90, 0.82)
    assert pytest.approx(0.16) == _FINGER_R_FRAC_PALM
    assert pytest.approx(1.95) == _FINGER_SPLAY_FRAC_HALF_W
    report = _rake_report()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        r0 = next(p for p in pkg.parts if p.name == f"RECIPE_finger_middle_0_{side}")
        r1 = next(p for p in pkg.parts if p.name == f"RECIPE_finger_middle_1_{side}")
        r2 = next(p for p in pkg.parts if p.name == f"RECIPE_finger_middle_2_{side}")
        assert r0.radius_m is not None and r1.radius_m is not None and r2.radius_m is not None
        assert float(r0.radius_m) > float(r1.radius_m) > float(r2.radius_m)


def test_t8_0083_distal_fence() -> None:
    """T8: product-like skeleton elbow/wrist still distal prior class."""
    h = 1.72
    half = 0.1303
    report = _product_class_report()
    pkg = build_blockout_skeleton(report)
    j = _by_id(pkg)
    expected = _arm_forward_y(0.0, half_depth=half, height_m=h, chest_front_y=-half)
    assert j["elbow_l"].y_m == pytest.approx(expected, abs=1e-4)
    assert j["wrist_l"].y_m == pytest.approx(expected, abs=1e-4)
    assert j["elbow_r"].y_m == pytest.approx(expected, abs=1e-4)
    assert j["wrist_r"].y_m == pytest.approx(expected, abs=1e-4)


def test_t9_hang_message() -> None:
    """T9: message includes 'hand hang:' and 'anti-rake'."""
    report = _rake_report()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    hang = [m for m in pkg.messages if "hand hang:" in m]
    assert hang, pkg.messages
    assert any("anti-rake" in m for m in hang)
    assert any("pitch=-0.55" in m for m in hang)
    assert any("axis=hang" in m for m in hang)


def test_t10_palm_ellipsoid_n_parts() -> None:
    """T10: C_palm_ellipsoid pass; product-class n_parts 129 (hair=short)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_palm_ellipsoid"].status == "pass", by_id["C_palm_ellipsoid"].message
    assert len(pkg.parts) == 129
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert SKELETON_SCHEMA_VERSION == "1.0.0"
    # Dual-layer: skeleton hand Y + B4 both land palm on wrist (not halfway to invent-0).
    for side in ("l", "r"):
        wrist = next(j for j in skel.joints if j.id == f"wrist_{side}")
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert wrist.y_m is not None and palm.center is not None
        assert float(palm.center[1]) == pytest.approx(float(wrist.y_m), abs=1e-6)


def test_t11_skeleton_hand_y_follows_wrist() -> None:
    """T11: plane-class fingertip → hand.y_m == wrist.y_m (XZ may still move)."""
    report = _product_class_report()
    pkg = build_blockout_skeleton(report)
    j = _by_id(pkg)
    for side in ("l", "r"):
        wrist = j[f"wrist_{side}"]
        hand = j[f"hand_{side}"]
        assert wrist.y_m is not None and hand.y_m is not None
        assert float(hand.y_m) == pytest.approx(float(wrist.y_m), abs=1e-6)
        # XZ still interpolates toward the tip (not a clone of wrist)
        assert hand.x_m is not None and wrist.x_m is not None
        assert abs(float(hand.x_m) - float(wrist.x_m)) > 1e-4
    assert any("fingertip untrusted" in m for m in pkg.messages)
