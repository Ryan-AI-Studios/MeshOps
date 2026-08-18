"""Track 0093 — mid-back / waist rear integrate (past 0.032 + lat 0.48 + z_below 0.035).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 46 stay. Not mesh/print success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    MID_BACK_BELOW_SCAP_M,
    MID_BACK_LAT_FRAC,
    MID_BACK_REAR_PAST_M,
    MID_BACK_RX_MIN_FRAC_H,
    MID_BACK_RY_FRAC_RX,
    MID_BACK_RY_MIN_FRAC_H,
    MID_BACK_RZ_FRAC_RX,
    MID_BACK_Z_BELOW_WAIST_M,
    MID_BACK_Z_DROP_FRAC_H,
    RECIPE_SCHEMA_VERSION,
    TORSO_CHEST_Y_REAR_BIAS_FRAC_RY,
    TORSO_HIP_Y_REAR_BIAS_FRAC_RY,
    TORSO_OVAL_OVERLAP_FLOOR_M,
    TORSO_OVAL_RY_CHEST_FRAC,
    TORSO_OVAL_RY_HIP_FRAC,
    TORSO_OVAL_RY_WAIST_FRAC,
    TORSO_OVAL_RZ_CHEST_FRAC,
    TORSO_OVAL_RZ_GROW_CAP_M,
    TORSO_OVAL_RZ_HIP_FRAC,
    TORSO_OVAL_RZ_WAIST_FRAC,
    TORSO_OVAL_Z_NORM_CHEST,
    TORSO_OVAL_Z_NORM_HIP,
    TORSO_OVAL_Z_NORM_WAIST,
    TORSO_WAIST_PINCH_TAPER_GATE,
    TORSO_WAIST_RX_MAX_FRAC_CHEST,
    TORSO_WAIST_Y_REAR_BIAS_FRAC_RY,
    RecipePart,
    _apply_mid_back_plane,
    _build_torso_ovals,
    _ResolvedMetrics,
    build_blockout_recipe,
)
from meshops.proportion.body_template import (
    AppliedConstants,
    TemplateAppliedPackage,
)
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.models import (
    CrossSection,
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import build_blockout_skeleton

_PRODUCT_NOFUSE = Path("work/rogue-v3/blockout/product_0093up/nofuse")
_PRODUCT_0092_NOFUSE = Path("work/rogue-v3/blockout/product_0092up/nofuse")
_CHEST = "RECIPE_torso_oval_chest"
_WAIST = "RECIPE_torso_oval_waist"
_HIP = "RECIPE_torso_oval_hip"

_PRODUCT_HALF_CHEST = 0.1303030303030303
_PRODUCT_HALF_HIP = 0.139


def _lm(
    id_: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=id_, x_m=x_m, y_m=y_m, z_m=z_m)


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


def _band(
    band_id: str,
    *,
    depth_m: float = 0.24,
    z_frac: float = 0.72,
    y_mid: float = 0.0,
) -> DepthBand:
    return DepthBand(
        band_id=band_id,
        depth_px=50.0,
        depth_frac=0.12,
        depth_m=depth_m,
        y_front=0.1,
        y_back=-0.1,
        y_mid=y_mid,
        z_frac=z_frac,
    )


def _part(
    name: str,
    *,
    kind: str = "ellipsoid",
    role: str = "mid_back_soft",
    center: list[float] | None = None,
    rx_m: float | None = None,
    ry_m: float | None = None,
    rz_m: float | None = None,
) -> RecipePart:
    kwargs: dict[str, Any] = {
        "name": name,
        "role": role,
        "kind": kind,
        "center": center,
        "rx_m": rx_m,
        "ry_m": ry_m,
        "rz_m": rz_m,
    }
    clean = {k: v for k, v in kwargs.items() if v is not None or k in ("name", "role", "kind")}
    return RecipePart.model_validate(clean)


def _empty_metrics(
    *,
    height_m: float = 1.72,
    shoulder_hw: float | None = 0.2575,
    shoulder_z: float | None = 1.3802,
    chest_half_depth: float | None = 0.12,
) -> _ResolvedMetrics:
    m = _ResolvedMetrics()
    m.height_m = height_m
    m.shoulder_hw = shoulder_hw
    m.shoulder_z = shoulder_z
    m.chest_half_depth = chest_half_depth
    return m


def _product_like_mid_backs(
    *,
    rx: float = 0.0654,
    ry: float = 0.0241,
    rz: float = 0.0800,
    cy: float = 0.0100,
    cz: float = 1.1411,
    cx: float = 0.0900,
) -> list[RecipePart]:
    return [
        _part(
            "RECIPE_mid_back_soft_l",
            role="mid_back_soft",
            center=[-cx, cy, cz],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
        ),
        _part(
            "RECIPE_mid_back_soft_r",
            role="mid_back_soft",
            center=[cx, cy, cz],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
        ),
    ]


def _waist_oval(
    *,
    cy: float = 0.031741818,
    ry: float = 0.075575758,
    cz: float = 1.14112,
    rx: float = 0.1772,
    rz: float = 0.1065,
) -> RecipePart:
    return _part(
        "RECIPE_torso_oval_waist",
        role="torso",
        center=[0.0, cy, cz],
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
    )


def _hip_oval(
    *,
    cy: float = 0.02935,
    ry: float = 0.08895,
    cz: float = 0.9881,
    rx: float = 0.2018,
    rz: float = 0.1165,
) -> RecipePart:
    return _part(
        "RECIPE_torso_oval_hip",
        role="torso",
        center=[0.0, cy, cz],
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
    )


def _product_like_scaps(
    *,
    rx: float = 0.0688,
    ry: float = 0.0289,
    rz: float = 0.0791,
    cy: float = 0.1248,
    cz: float = 1.2856,
    cx: float = 0.1159,
) -> list[RecipePart]:
    return [
        _part(
            "RECIPE_scap_soft_l",
            role="scap_soft",
            center=[-cx, cy, cz],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
        ),
        _part(
            "RECIPE_scap_soft_r",
            role="scap_soft",
            center=[cx, cy, cz],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
        ),
    ]


def _empty_report() -> ProportionReport:
    return ProportionReport(
        schema_version="1.0.0",
        height_m=1.72,
        landmarks_xyz={},
        diameters=[],
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
        _band("chest", depth_m=0.2606, y_mid=0.0),
        _band("breast", depth_m=0.18),
        _band("hip", depth_m=0.26),
        _band("glute", depth_m=0.22),
    ]
    return ProportionReport(
        schema_version="1.2.0",
        height_m=h,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        depth_bands=bands,
        diameters=diams,
        cross_sections=[
            CrossSection(
                level_id="bust",
                z_frac=0.72,
                rx_frac=0.10,
                ry_frac=0.08,
                sources=["test"],
            ),
        ],
        quality=QualityFlags(),
    )


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


def _template(*, taper: float = 0.22) -> TemplateAppliedPackage:
    constants = AppliedConstants(
        breast_mode="dual_tilted",
        glute_mode_default="two_spheres",
        torso_mode_default="ovals",
        torso_waist_taper=taper,
    )
    return TemplateAppliedPackage(
        template_id="female_adult_athletic",
        sex="female",  # type: ignore[arg-type]
        archetype="adult_athletic",
        source_report="mem",
        height_m=1.72,
        constants=constants,
    )


def _metrics_for_ovals(
    *,
    chest_y: float | None = 0.0,
    shoulder_hw: float = 0.20,
    hip_hw: float = 0.14,
    half_chest: float = _PRODUCT_HALF_CHEST,
    half_hip: float = _PRODUCT_HALF_HIP,
    shoulder_z: float = 1.38,
    hip_z: float = 0.95,
    height_m: float = 1.72,
    chest_z: float | None = None,
) -> _ResolvedMetrics:
    m = _ResolvedMetrics()
    m.shoulder_hw = shoulder_hw
    m.hip_hw = hip_hw
    m.chest_half_depth = half_chest
    m.hip_half_depth = half_hip
    m.shoulder_z = shoulder_z
    m.hip_z = hip_z
    m.chest_z = chest_z
    m.chest_y = chest_y
    m.height_m = height_m
    return m


def _apply_product_like_mid_back() -> tuple[list[RecipePart], list[str]]:
    parts = [*_product_like_mid_backs(), _waist_oval(), _hip_oval(), *_product_like_scaps()]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    return parts, msgs


def test_t0_public_freezes_and_fences() -> None:
    """T0: past 0.032 + lat 0.48 + z_below 0.035; neighbor fences unchanged."""
    assert MID_BACK_REAR_PAST_M == 0.032
    assert MID_BACK_LAT_FRAC == 0.48
    assert MID_BACK_Z_BELOW_WAIST_M == 0.035
    assert 0.028 <= MID_BACK_REAR_PAST_M <= 0.036
    assert 0.44 <= MID_BACK_LAT_FRAC <= 0.50
    assert 0.025 <= MID_BACK_Z_BELOW_WAIST_M <= 0.036
    assert MID_BACK_BELOW_SCAP_M == 0.008
    assert MID_BACK_RY_FRAC_RX == 0.38
    assert MID_BACK_RZ_FRAC_RX == 1.30
    assert MID_BACK_RX_MIN_FRAC_H == 0.038
    assert MID_BACK_RY_MIN_FRAC_H == 0.014
    assert MID_BACK_Z_DROP_FRAC_H == 0.14
    assert TORSO_WAIST_Y_REAR_BIAS_FRAC_RY == 0.42
    assert TORSO_OVAL_RY_HIP_FRAC == 0.64
    assert TORSO_HIP_Y_REAR_BIAS_FRAC_RY == 0.33
    assert TORSO_OVAL_RY_CHEST_FRAC == 0.72
    assert TORSO_CHEST_Y_REAR_BIAS_FRAC_RY == 0.51
    assert TORSO_OVAL_Z_NORM_CHEST == 0.18
    assert TORSO_OVAL_Z_NORM_WAIST == 0.50
    assert TORSO_OVAL_Z_NORM_HIP == 0.82
    assert TORSO_OVAL_OVERLAP_FLOOR_M == 0.070
    assert TORSO_OVAL_RZ_CHEST_FRAC == 0.28
    assert TORSO_OVAL_RZ_WAIST_FRAC == 0.16
    assert TORSO_OVAL_RZ_HIP_FRAC == 0.24
    assert TORSO_OVAL_RZ_GROW_CAP_M == 0.030
    assert TORSO_WAIST_RX_MAX_FRAC_CHEST == 0.80
    assert TORSO_OVAL_RY_WAIST_FRAC == 0.58
    assert TORSO_WAIST_PINCH_TAPER_GATE == 0.10


def test_t1_outer_past_and_z_below() -> None:
    """T1: product-like outer == waist_rear + 0.032 (pre-cape); z == waist_z - 0.035."""
    waist = _waist_oval()
    assert waist.center is not None and waist.ry_m is not None
    waist_rear = float(waist.center[1]) + float(waist.ry_m)
    waist_z = float(waist.center[2])
    parts, _msgs = _apply_product_like_mid_back()
    mbs = [p for p in parts if p.role == "mid_back_soft"]
    assert len(mbs) == 2
    for mb in mbs:
        assert mb.center is not None and mb.ry_m is not None
        outer = float(mb.center[1]) + float(mb.ry_m)
        assert outer == pytest.approx(waist_rear + MID_BACK_REAR_PAST_M, abs=1e-9)
        assert float(mb.center[2]) == pytest.approx(waist_z - MID_BACK_Z_BELOW_WAIST_M, abs=1e-9)


def test_t2_lat_frac_on_shoulder_hw() -> None:
    """T2: |cx| == lat * shoulder_hw with lat 0.48."""
    sh = 0.2575
    parts = [*_product_like_mid_backs(), _waist_oval(), *_product_like_scaps()]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(shoulder_hw=sh), msgs)
    expected = MID_BACK_LAT_FRAC * sh
    for p in parts:
        if p.role != "mid_back_soft":
            continue
        assert p.center is not None
        assert abs(float(p.center[0])) == pytest.approx(expected, abs=1e-9)


def test_t3_cape_still_caps_outer() -> None:
    """T3: even if past would exceed, outer <= scap_outer - 0.008 (0074 T7b no-abs)."""
    scaps = _product_like_scaps(cy=0.02, ry=0.01)  # outer = 0.03
    scap_outer = 0.03
    waist = _waist_oval(cy=0.05, ry=0.05)  # rear 0.10 → would cape
    parts = [*_product_like_mid_backs(rx=0.10, ry=0.05), waist, *scaps]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    for p in parts:
        if p.role != "mid_back_soft":
            continue
        assert p.center is not None and p.ry_m is not None
        cy = float(p.center[1])
        ry = float(p.ry_m)
        outer = cy + ry
        assert outer <= scap_outer - MID_BACK_BELOW_SCAP_M + 1e-6
        assert outer != abs(cy) + ry or cy >= 0.0


def test_t4_order_scap_mid_hip() -> None:
    """T4: scap_outer > mid_back_outer > hip_rear on product-like (B15)."""
    parts, _msgs = _apply_product_like_mid_back()
    hip = next(p for p in parts if p.name == _HIP)
    assert hip.center is not None and hip.ry_m is not None
    hip_rear = float(hip.center[1]) + float(hip.ry_m)
    scaps = [p for p in parts if p.role == "scap_soft"]
    mbs = [p for p in parts if p.role == "mid_back_soft"]
    scap_outers = [float(s.center[1]) + float(s.ry_m or 0.0) for s in scaps if s.center is not None]
    mb_outers = [float(m.center[1]) + float(m.ry_m or 0.0) for m in mbs if m.center is not None]
    scap_outer = sum(scap_outers) / float(len(scap_outers))
    mb_outer = sum(mb_outers) / float(len(mb_outers))
    assert scap_outer > mb_outer + 1e-9
    assert mb_outer > hip_rear + 1e-9
    assert mb_outer <= scap_outer - MID_BACK_BELOW_SCAP_M + 1e-9


def test_t5_no_waist_fallback_z_and_cy() -> None:
    """T5: no-waist z uses shoulder drop; cy == max(|pre_y|, 0.90*ry) (AI2 F4)."""
    pre_y = 0.0100
    parts = [*_product_like_mid_backs(cy=pre_y), *_product_like_scaps()]
    msgs: list[str] = []
    m = _empty_metrics(height_m=1.72, shoulder_z=1.3802)
    _apply_mid_back_plane(parts, _empty_report(), m, msgs)
    expected_z = 1.3802 - MID_BACK_Z_DROP_FRAC_H * 1.72
    for p in parts:
        if p.role != "mid_back_soft":
            continue
        assert p.center is not None and p.ry_m is not None
        assert float(p.center[2]) == pytest.approx(expected_z, abs=1e-9)
        ry = float(p.ry_m)
        assert float(p.center[1]) == pytest.approx(max(abs(pre_y), 0.90 * ry), abs=1e-9)


def test_t6_plate_axes_unchanged() -> None:
    """T6: plate axes stay ry/rx 0.38, rz/rx 1.30, rx >= 0.038*H."""
    parts, _msgs = _apply_product_like_mid_back()
    for p in parts:
        if p.role != "mid_back_soft":
            continue
        assert p.rx_m is not None and p.ry_m is not None and p.rz_m is not None
        rx = float(p.rx_m)
        assert rx >= MID_BACK_RX_MIN_FRAC_H * 1.72 - 1e-12
        assert float(p.ry_m) / rx == pytest.approx(MID_BACK_RY_FRAC_RX, rel=1e-6)
        assert float(p.rz_m) / rx == pytest.approx(MID_BACK_RZ_FRAC_RX, rel=1e-6)


def test_t7_integrate_and_0074_messages() -> None:
    """T7: sibling integrate line + 0074 mid_back_plane_past_m= still present."""
    _parts, msgs = _apply_product_like_mid_back()
    blob = " ".join(msgs)
    assert any(m == "mid_back_plane_applied: true" for m in msgs)
    assert f"mid_back_plane_past_m={MID_BACK_REAR_PAST_M}" in msgs
    assert f"mid_back_plane_lat_frac={MID_BACK_LAT_FRAC}" in msgs
    integ = [m for m in msgs if m.startswith("torso mid-back integrate:")]
    assert len(integ) == 1
    assert "past=" in integ[0]
    assert "lat=" in integ[0]
    assert "z_below=" in integ[0]
    assert "outer=" in integ[0]
    assert "z=" in integ[0]
    assert f"past={MID_BACK_REAR_PAST_M}" in integ[0]
    assert f"lat={MID_BACK_LAT_FRAC}" in integ[0]
    assert f"z_below={MID_BACK_Z_BELOW_WAIST_M}" in integ[0]
    assert "mid_back_soft_l:" in blob
    assert "outer_rear=" in blob


def test_t8_cluster_fence_dual_mid_back_scap_glute_iliac() -> None:
    """T8: dual mid_back_soft; scap pair; glute pair; iliac 0."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(taper=0.22),
        **_product_flags(),  # type: ignore[arg-type]
    )
    mbs = [p for p in pkg.parts if p.role == "mid_back_soft"]
    assert {p.name for p in mbs} == {"RECIPE_mid_back_soft_l", "RECIPE_mid_back_soft_r"}
    scaps = [p for p in pkg.parts if p.role == "scap_soft"]
    assert len(scaps) == 2
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) == 2
    iliac = [p for p in pkg.parts if p.role == "iliac_soft"]
    assert len(iliac) == 0


def test_t9_product_n_parts_131_schema_mcp() -> None:
    """T9: n_parts 131; schema 1.4.0; MCP 46; product-path C_thigh_outer / C_glute_outer."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(taper=0.22),
        **_product_flags(),  # type: ignore[arg-type]
    )
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert len(TOOL_NAMES) == 46
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert "C_glute_outer" in by_id
    assert "C_palm_ellipsoid" in by_id
    assert by_id["C_palm_ellipsoid"].status == "pass"
    assert by_id["C_glute_outer"].status == "pass"
    cons_path = _PRODUCT_NOFUSE / "constraints_report.json"
    if cons_path.is_file():
        cons = json.loads(cons_path.read_text(encoding="utf-8"))
        rule_by = {r["id"]: r["status"] for r in cons.get("rules", [])}
        assert rule_by.get("C_thigh_outer") == "pass"
        assert rule_by.get("C_glute_outer") == "pass"


def test_t10_hip_poles_unchanged() -> None:
    """T10: 0092 hip poles hold on product-like (rear ~ 0.1183; front ~ -0.0596)."""
    m = _metrics_for_ovals(chest_y=0.0)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    hip = by[_HIP]
    assert hip.center is not None and hip.ry_m is not None
    ry = float(hip.ry_m)
    cy = float(hip.center[1])
    front = cy - ry
    rear = cy + ry
    expected_ry = _PRODUCT_HALF_HIP * TORSO_OVAL_RY_HIP_FRAC
    expected_cy = 0.0 + TORSO_HIP_Y_REAR_BIAS_FRAC_RY * expected_ry
    assert ry == pytest.approx(expected_ry, abs=1e-9)
    assert front == pytest.approx(expected_cy - expected_ry, abs=1e-9)
    assert rear == pytest.approx(expected_cy + expected_ry, abs=1e-9)
    assert rear == pytest.approx(0.1183, abs=2e-4)
    assert front == pytest.approx(-0.0596, abs=2e-4)


def test_t11_waist_bias_and_rear_hold() -> None:
    """T11: waist bias still 0.42; waist rear holds vs 0.42 formula (abs Δ ≤ 1e-6)."""
    assert TORSO_WAIST_Y_REAR_BIAS_FRAC_RY == 0.42
    m = _metrics_for_ovals(chest_y=0.0)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    waist = by[_WAIST]
    assert waist.center is not None and waist.ry_m is not None
    ry = float(waist.ry_m)
    cy = float(waist.center[1])
    rear = cy + ry
    expected_cy = 0.0 + TORSO_WAIST_Y_REAR_BIAS_FRAC_RY * ry
    expected_rear = expected_cy + ry
    assert cy == pytest.approx(expected_cy, abs=1e-9)
    assert abs(rear - expected_rear) <= 1e-6


def test_t12_all_exports_z_below() -> None:
    """T12: __all__ exports MID_BACK_Z_BELOW_WAIST_M and still past/lat."""
    from meshops.proportion import blockout_recipe as br

    names = set(br.__all__)
    assert "MID_BACK_Z_BELOW_WAIST_M" in names
    assert "MID_BACK_REAR_PAST_M" in names
    assert "MID_BACK_LAT_FRAC" in names
    assert br.__all__.index("MID_BACK_Z_BELOW_WAIST_M") < br.__all__.index("MID_BACK_Z_DROP_FRAC_H")
    assert br.__all__.index("MID_BACK_RZ_FRAC_RX") < br.__all__.index("MID_BACK_Z_BELOW_WAIST_M")


def test_t13_dual_lr_equalize_after_z_below() -> None:
    """T13: dual L/R still equalize axes + |cx| + cy + z after z_below."""
    parts = [
        _part(
            "RECIPE_mid_back_soft_l",
            role="mid_back_soft",
            center=[-0.08, 0.01, 1.10],
            rx_m=0.050,
            ry_m=0.018,
            rz_m=0.060,
        ),
        _part(
            "RECIPE_mid_back_soft_r",
            role="mid_back_soft",
            center=[0.12, 0.02, 1.20],
            rx_m=0.070,
            ry_m=0.028,
            rz_m=0.090,
        ),
        _waist_oval(),
        *_product_like_scaps(),
    ]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    ml = next(p for p in parts if p.name == "RECIPE_mid_back_soft_l")
    mr = next(p for p in parts if p.name == "RECIPE_mid_back_soft_r")
    assert ml.center is not None and mr.center is not None
    assert float(ml.rx_m or 0.0) == pytest.approx(float(mr.rx_m or 0.0), abs=1e-12)
    assert float(ml.ry_m or 0.0) == pytest.approx(float(mr.ry_m or 0.0), abs=1e-12)
    assert float(ml.rz_m or 0.0) == pytest.approx(float(mr.rz_m or 0.0), abs=1e-12)
    assert abs(float(ml.center[0])) == pytest.approx(abs(float(mr.center[0])), abs=1e-12)
    assert float(ml.center[1]) == pytest.approx(float(mr.center[1]), abs=1e-12)
    assert float(ml.center[2]) == pytest.approx(float(mr.center[2]), abs=1e-12)
    waist = next(p for p in parts if p.name == _WAIST)
    assert waist.center is not None
    assert float(ml.center[2]) == pytest.approx(
        float(waist.center[2]) - MID_BACK_Z_BELOW_WAIST_M, abs=1e-9
    )


def test_t14_waist_z_norm_unchanged() -> None:
    """T14: waist z_norm still 0.50; waist oval z unchanged vs 0092up; mid_back moved."""
    assert TORSO_OVAL_Z_NORM_WAIST == 0.50
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(taper=0.22),
        **_product_flags(),  # type: ignore[arg-type]
    )
    by = {p.name: p for p in pkg.parts}
    waist = by[_WAIST]
    assert waist.center is not None
    waist_z = float(waist.center[2])
    mbs = [p for p in pkg.parts if p.role == "mid_back_soft"]
    assert len(mbs) == 2
    for mb in mbs:
        assert mb.center is not None
        assert float(mb.center[2]) == pytest.approx(waist_z - MID_BACK_Z_BELOW_WAIST_M, abs=2e-3)
        assert float(mb.center[2]) < waist_z - 1e-9
    old = _PRODUCT_0092_NOFUSE / "blockout_recipe.json"
    new = _PRODUCT_NOFUSE / "blockout_recipe.json"
    if old.is_file() and new.is_file():
        old_data = json.loads(old.read_text(encoding="utf-8"))
        new_data = json.loads(new.read_text(encoding="utf-8"))
        old_parts = {p["name"]: p for p in old_data["parts"]}
        new_parts = {p["name"]: p for p in new_data["parts"]}
        assert float(new_parts[_WAIST]["center"][2]) == pytest.approx(
            float(old_parts[_WAIST]["center"][2]), abs=1e-6
        )


def test_t15_waist_poles_documented() -> None:
    """T15: product-like waist rx/front/rear + mid_back.z < waist.z (B20)."""
    parts, _msgs = _apply_product_like_mid_back()
    waist = next(p for p in parts if p.name == _WAIST)
    assert waist.center is not None and waist.ry_m is not None and waist.rx_m is not None
    cy = float(waist.center[1])
    ry = float(waist.ry_m)
    front = cy - ry
    rear = cy + ry
    assert float(waist.rx_m) == pytest.approx(0.1772, abs=2e-3)
    assert front == pytest.approx(-0.0438, abs=2e-3)
    assert rear == pytest.approx(0.1073, abs=2e-3)
    for mb in parts:
        if mb.role != "mid_back_soft":
            continue
        assert mb.center is not None
        assert float(mb.center[2]) < float(waist.center[2]) - 1e-9
    recipe_path = _PRODUCT_NOFUSE / "blockout_recipe.json"
    if recipe_path.is_file():
        data = json.loads(recipe_path.read_text(encoding="utf-8"))
        pby = {p["name"]: p for p in data["parts"]}
        w = pby[_WAIST]
        w_cy = float(w["center"][1])
        w_ry = float(w["ry_m"])
        assert float(w["rx_m"]) == pytest.approx(0.1772, abs=2e-3)
        assert w_cy - w_ry == pytest.approx(-0.0438, abs=2e-3)
        assert w_cy + w_ry == pytest.approx(0.1073, abs=2e-3)
        mb_z = float(pby["RECIPE_mid_back_soft_l"]["center"][2])
        assert mb_z < float(w["center"][2]) - 1e-9


def test_t16_outer_x_overhangs_waist_rx() -> None:
    """T16: product-like outer_x = |cx|+rx > waist.rx at named 0.48 (B21; D7 owns shelf)."""
    parts, _msgs = _apply_product_like_mid_back()
    waist = next(p for p in parts if p.name == _WAIST)
    assert waist.rx_m is not None
    waist_rx = float(waist.rx_m)
    mbs = [p for p in parts if p.role == "mid_back_soft"]
    for mb in mbs:
        assert mb.center is not None and mb.rx_m is not None
        outer_x = abs(float(mb.center[0])) + float(mb.rx_m)
        assert outer_x > waist_rx + 1e-9
    recipe_path = _PRODUCT_NOFUSE / "blockout_recipe.json"
    if recipe_path.is_file():
        data = json.loads(recipe_path.read_text(encoding="utf-8"))
        pby = {p["name"]: p for p in data["parts"]}
        w_rx = float(pby[_WAIST]["rx_m"])
        mb = pby["RECIPE_mid_back_soft_l"]
        outer_x = abs(float(mb["center"][0])) + float(mb["rx_m"])
        assert outer_x > w_rx + 1e-9
        assert outer_x == pytest.approx(0.189, abs=0.005)
