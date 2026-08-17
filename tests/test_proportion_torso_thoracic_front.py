"""Track 0090 — torso thoracic front plane (chest ry 0.72 + rear bias 0.51).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 46 stay. Not mesh/print success.
"""

from __future__ import annotations

import math

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    MID_BACK_Z_BELOW_WAIST_M,
    PELVIS_OVAL_RY_FRAC_HALF_HIP,
    RECIPE_SCHEMA_VERSION,
    SCAP_REAR_PAST_M,
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

_CHEST = "RECIPE_torso_oval_chest"
_WAIST = "RECIPE_torso_oval_waist"
_HIP = "RECIPE_torso_oval_hip"

# 0065 leftover (not public law) — T2/T3 relative poles only.
_OLD_RY_CHEST_FRAC = 0.85
_OLD_CHEST_REAR_BIAS = 0.28

# Product-like halves from live 0089up (H=1.72, F athletic).
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


def _full_torso_report(
    *,
    height_m: float = 1.72,
    shoulder_x: float = 0.20,
    hip_x: float = 0.14,
    hip_z: float = 0.95,
    shoulder_z: float = 1.38,
    chest_depth_m: float = 0.24,
    hip_depth_m: float = 0.26,
    chest_mid_y: float | None = None,
) -> ProportionReport:
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-shoulder_x, y_m=0.0, z_m=shoulder_z),
        "shoulder_r": _lm("shoulder_r", x_m=shoulder_x, y_m=0.0, z_m=shoulder_z),
        "hip_l": _lm("hip_l", x_m=-hip_x, y_m=0.0, z_m=hip_z),
        "hip_r": _lm("hip_r", x_m=hip_x, y_m=0.0, z_m=hip_z),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
    }
    if chest_mid_y is not None:
        lms["chest_mid"] = _lm("chest_mid", x_m=0.0, y_m=chest_mid_y, z_m=1.25)
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=[
            _diam("bust", half_width_m=0.16),
            _diam("waist", half_width_m=0.13),
            _diam("neck", half_width_m=0.05),
        ],
        depth_bands=[
            _band("chest", depth_m=chest_depth_m, z_frac=0.72, y_mid=0.0),
            _band("hip", depth_m=hip_depth_m, z_frac=0.55),
        ],
        cross_sections=[],
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


def _chest_poles(
    *,
    half: float,
    ry_frac: float,
    bias: float,
    y_mid: float = 0.0,
) -> tuple[float, float, float]:
    ry = half * ry_frac
    cy = y_mid + bias * ry
    return ry, cy - ry, cy + ry


def _clavicle_med_y(part: RecipePart) -> float:
    assert part.p0 is not None and part.p1 is not None
    p0 = [float(x) for x in part.p0]
    p1 = [float(x) for x in part.p1]
    med = p0 if abs(p0[0]) < abs(p1[0]) else p1
    return float(med[1])


def test_t0_public_freezes_and_fences() -> None:
    """T0: chest ry 0.72 + bias 0.51; 0089/0073/0065 fences unchanged."""
    assert TORSO_OVAL_RY_CHEST_FRAC == 0.72
    assert TORSO_CHEST_Y_REAR_BIAS_FRAC_RY == 0.51
    assert 0.70 <= TORSO_OVAL_RY_CHEST_FRAC <= 0.76
    assert 0.46 <= TORSO_CHEST_Y_REAR_BIAS_FRAC_RY <= 0.56
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
    assert TORSO_OVAL_RY_HIP_FRAC == 0.64
    assert TORSO_WAIST_PINCH_TAPER_GATE == 0.10
    assert TORSO_WAIST_Y_REAR_BIAS_FRAC_RY == 0.42
    assert TORSO_HIP_Y_REAR_BIAS_FRAC_RY == 0.33


def test_t1_product_like_ry_and_full3d_cy() -> None:
    """T1: product-like ry = half*0.72; full3d cy = y_mid + 0.51*ry."""
    m = _metrics_for_ovals(chest_y=0.0)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    chest = by[_CHEST]
    assert chest.center is not None and chest.ry_m is not None
    ry = float(chest.ry_m)
    assert ry == pytest.approx(_PRODUCT_HALF_CHEST * TORSO_OVAL_RY_CHEST_FRAC, abs=1e-9)
    assert float(chest.center[1]) == pytest.approx(
        0.0 + TORSO_CHEST_Y_REAR_BIAS_FRAC_RY * ry, abs=1e-9
    )
    assert chest.placement == "full3d"


def test_t2_front_less_proud_than_old_085_028() -> None:
    """T2: chest_front less proud than 0.85/0.28 (Δ ≥ 0.025 on product-like half)."""
    m = _metrics_for_ovals(chest_y=0.0)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    chest = by[_CHEST]
    assert chest.center is not None and chest.ry_m is not None
    front = float(chest.center[1]) - float(chest.ry_m)
    _old_ry, old_front, _old_rear = _chest_poles(
        half=_PRODUCT_HALF_CHEST,
        ry_frac=_OLD_RY_CHEST_FRAC,
        bias=_OLD_CHEST_REAR_BIAS,
    )
    assert front - old_front >= 0.025 - 1e-9


def test_t3_rear_holds_vs_old_085_028() -> None:
    """T3: chest_rear holds vs 0.85/0.28 (abs Δ ≤ 0.002 on product-like half)."""
    m = _metrics_for_ovals(chest_y=0.0)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    chest = by[_CHEST]
    assert chest.center is not None and chest.ry_m is not None
    rear = float(chest.center[1]) + float(chest.ry_m)
    _old_ry, _old_front, old_rear = _chest_poles(
        half=_PRODUCT_HALF_CHEST,
        ry_frac=_OLD_RY_CHEST_FRAC,
        bias=_OLD_CHEST_REAR_BIAS,
    )
    assert abs(rear - old_rear) <= 0.002 + 1e-12


def test_t4_ry_over_rz_plate_and_chest_gt_waist() -> None:
    """T4: ry_chest/rz_chest < 0.80 on product-like; still ry_chest > ry_waist."""
    m = _metrics_for_ovals(chest_y=0.0)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    ry_c = float(by[_CHEST].ry_m or 0.0)
    rz_c = float(by[_CHEST].rz_m or 0.0)
    ry_w = float(by[_WAIST].ry_m or 0.0)
    assert rz_c > 1e-9
    assert ry_c / rz_c < 0.80
    assert ry_c > ry_w + 1e-9


def test_t5_pinch_cap_when_taper_gate_on() -> None:
    """T5: 0065 pinch rx_w/rx_c ≤ 0.80 when taper gate on."""
    report = _full_torso_report(shoulder_x=0.25, hip_x=0.24, chest_mid_y=0.0)
    pkg = build_blockout_recipe(
        report,
        limbs=False,
        torso="ovals",
        template_applied=_template(taper=0.14),
    )
    by = {p.name: p for p in pkg.parts}
    rx_c = float(by[_CHEST].rx_m or 0.0)
    rx_w = float(by[_WAIST].rx_m or 0.0)
    assert rx_w <= TORSO_WAIST_RX_MAX_FRAC_CHEST * rx_c + 1e-9


def test_t6_front_plane_no_rear_bias_still_072() -> None:
    """T6: front_plane cy stays mid; no rear bias; ry still 0.72."""
    msgs: list[str] = []
    m = _metrics_for_ovals(chest_y=None)
    parts = _build_torso_ovals(m, msgs, taper=0.14)
    by = {p.name: p for p in parts}
    chest = by[_CHEST]
    assert chest.center is not None and chest.ry_m is not None
    assert chest.placement == "front_plane"
    assert float(chest.center[1]) == pytest.approx(0.0, abs=1e-9)
    assert float(chest.ry_m) == pytest.approx(
        _PRODUCT_HALF_CHEST * TORSO_OVAL_RY_CHEST_FRAC, abs=1e-9
    )


def test_t7_thoracic_and_pinch_messages() -> None:
    """T7: sibling thoracic line + pinch still has chest_front_y=."""
    report = _full_torso_report(chest_mid_y=0.0)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    thoracic = [m for m in pkg.messages if m.startswith("torso thoracic front:")]
    pinch = [m for m in pkg.messages if m.startswith("torso front pinch:")]
    assert len(thoracic) == 1
    assert "ry_frac=" in thoracic[0]
    assert "bias=" in thoracic[0]
    assert f"ry_frac={TORSO_OVAL_RY_CHEST_FRAC}" in thoracic[0]
    assert f"bias={TORSO_CHEST_Y_REAR_BIAS_FRAC_RY}" in thoracic[0]
    assert "front=" in thoracic[0]
    assert "rear=" in thoracic[0]
    assert len(pinch) == 1
    assert "chest_front_y=" in pinch[0]
    by = {p.name: p for p in pkg.parts}
    chest = by[_CHEST]
    assert chest.center is not None and chest.ry_m is not None
    front = float(chest.center[1]) - float(chest.ry_m)
    rear = float(chest.center[1]) + float(chest.ry_m)
    assert f"front={front:.4f}" in thoracic[0]
    assert f"rear={rear:.4f}" in thoracic[0]


def test_t8_breast_still_proud_of_new_front() -> None:
    """T8: breast center still more -Y than chest_front (do not freeze breast Y)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    tpl = _template(taper=0.22)
    tpl = tpl.model_copy(
        update={
            "constants": tpl.constants.model_copy(update={"breast_y_m": -0.10}),
        }
    )
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=tpl,
        **_product_flags(),  # type: ignore[arg-type]
    )
    by = {p.name: p for p in pkg.parts}
    chest = by[_CHEST]
    assert chest.center is not None and chest.ry_m is not None
    chest_front = float(chest.center[1]) - float(chest.ry_m)
    breasts = [p for p in pkg.parts if p.role == "breast_soft"]
    assert breasts
    for b in breasts:
        assert b.center is not None
        assert float(b.center[1]) < chest_front - 1e-9


def test_t9_product_n_parts_131_schema_mcp() -> None:
    """T9: n_parts 131 via hair=short + profile; schema 1.4.0; MCP 46."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert len(TOOL_NAMES) == 46
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert "C_palm_ellipsoid" in by_id
    assert by_id["C_palm_ellipsoid"].status == "pass"


def test_t10_waist_hip_bias_and_mid_back_z() -> None:
    """T10: 0074 waist bias 0.42; 0092 hip bias 0.33; mid_back z ~ waist - z_below."""
    assert TORSO_WAIST_Y_REAR_BIAS_FRAC_RY == 0.42
    assert TORSO_HIP_Y_REAR_BIAS_FRAC_RY == 0.33
    assert TORSO_WAIST_Y_REAR_BIAS_FRAC_RY != TORSO_CHEST_Y_REAR_BIAS_FRAC_RY
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    by = {p.name: p for p in pkg.parts}
    waist = by[_WAIST]
    assert waist.center is not None
    waist_z = float(waist.center[2])
    mbs = [p for p in pkg.parts if p.role == "mid_back_soft"]
    assert len(mbs) == 2
    for mb in mbs:
        assert mb.center is not None
        assert float(mb.center[2]) == pytest.approx(waist_z - MID_BACK_Z_BELOW_WAIST_M, abs=2e-3)


def test_t11_scap_follows_chest_rear_plus_past() -> None:
    """T11: scap cy+ry ≈ chest_rear + 0.012 (existing 0066 law)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    by = {p.name: p for p in pkg.parts}
    chest = by[_CHEST]
    assert chest.center is not None and chest.ry_m is not None
    chest_rear = float(chest.center[1]) + float(chest.ry_m)
    scaps = [p for p in pkg.parts if p.role == "scap_soft"]
    assert scaps
    for s in scaps:
        assert s.center is not None and s.ry_m is not None
        outer = float(s.center[1]) + float(s.ry_m)
        assert outer == pytest.approx(chest_rear + SCAP_REAR_PAST_M, abs=2e-3)


def test_t12_all_exports_chest_ry_and_bias() -> None:
    """T12: __all__ still exports chest ry + rear-bias consts."""
    from meshops.proportion import blockout_recipe as br

    names = set(br.__all__)
    assert "TORSO_OVAL_RY_CHEST_FRAC" in names
    assert "TORSO_CHEST_Y_REAR_BIAS_FRAC_RY" in names


def test_t13_clavicle_med_not_in_front_of_new_plate() -> None:
    """T13: clavicle med Y ≥ chest_front (not in front of new plate)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    by = {p.name: p for p in pkg.parts}
    chest = by[_CHEST]
    assert chest.center is not None and chest.ry_m is not None
    chest_front = float(chest.center[1]) - float(chest.ry_m)
    clavs = [p for p in pkg.parts if p.role == "clavicle"]
    assert clavs
    for clav in clavs:
        med_y = _clavicle_med_y(clav)
        assert math.isfinite(med_y)
        # B15: oval binds (not recessed behind plate; not in front of it).
        assert med_y == pytest.approx(chest_front, abs=2e-3)


def test_t14_product_like_chest_deeper_than_hip() -> None:
    """T14: product-like halves restore ry_c > ry_h; still chest>waist and hip>pelvis."""
    m = _metrics_for_ovals(
        chest_y=0.0,
        half_chest=_PRODUCT_HALF_CHEST,
        half_hip=_PRODUCT_HALF_HIP,
    )
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    ry_c = float(by[_CHEST].ry_m or 0.0)
    ry_w = float(by[_WAIST].ry_m or 0.0)
    ry_h = float(by[_HIP].ry_m or 0.0)
    ry_p = _PRODUCT_HALF_HIP * PELVIS_OVAL_RY_FRAC_HALF_HIP
    assert ry_c > ry_h
    assert ry_c > ry_w + 1e-9
    assert ry_h > ry_p + 1e-9
