"""Track 0092 — hip hierarchy polish (hip ry 0.64 + rear bias 0.33).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 46 stay. Not mesh/print success.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    MID_BACK_Z_BELOW_WAIST_M,
    PELVIS_OVAL_RY_FRAC_HALF_HIP,
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

_PRODUCT_NOFUSE = Path("work/rogue-v3/blockout/product_0092up/nofuse")
_CHEST = "RECIPE_torso_oval_chest"
_WAIST = "RECIPE_torso_oval_waist"
_HIP = "RECIPE_torso_oval_hip"
_PELVIS = "RECIPE_pelvis_oval"

# 0073 leftover (not public law) — T2/T3 relative poles only.
_OLD_RY_HIP_FRAC = 0.70
_OLD_HIP_REAR_BIAS = 0.22

# Product-like halves from live 0090up (H=1.72, F athletic).
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


def _hip_poles(
    *,
    half: float,
    ry_frac: float,
    bias: float,
    y_mid: float = 0.0,
) -> tuple[float, float, float]:
    ry = half * ry_frac
    cy = y_mid + bias * ry
    return ry, cy - ry, cy + ry


def test_t0_public_freezes_and_fences() -> None:
    """T0: hip ry 0.64 + bias 0.33; 0090/0089/0073/0065 fences unchanged."""
    assert TORSO_OVAL_RY_HIP_FRAC == 0.64
    assert TORSO_HIP_Y_REAR_BIAS_FRAC_RY == 0.33
    assert 0.61 <= TORSO_OVAL_RY_HIP_FRAC <= 0.68
    assert 0.28 <= TORSO_HIP_Y_REAR_BIAS_FRAC_RY <= 0.38
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
    assert TORSO_WAIST_Y_REAR_BIAS_FRAC_RY == 0.42


def test_t1_product_like_ry_and_full3d_cy() -> None:
    """T1: product-like ry = half*0.64; full3d cy = y_mid + 0.33*ry."""
    m = _metrics_for_ovals(chest_y=0.0)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    hip = by[_HIP]
    assert hip.center is not None and hip.ry_m is not None
    ry = float(hip.ry_m)
    assert ry == pytest.approx(_PRODUCT_HALF_HIP * TORSO_OVAL_RY_HIP_FRAC, abs=1e-9)
    assert float(hip.center[1]) == pytest.approx(0.0 + TORSO_HIP_Y_REAR_BIAS_FRAC_RY * ry, abs=1e-9)
    assert hip.placement == "full3d"


def test_t2_front_less_proud_than_old_070_022() -> None:
    """T2: hip_front less proud than 0.70/0.22 (Δ ≥ 0.010 on product-like half)."""
    m = _metrics_for_ovals(chest_y=0.0)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    hip = by[_HIP]
    assert hip.center is not None and hip.ry_m is not None
    front = float(hip.center[1]) - float(hip.ry_m)
    _old_ry, old_front, _old_rear = _hip_poles(
        half=_PRODUCT_HALF_HIP,
        ry_frac=_OLD_RY_HIP_FRAC,
        bias=_OLD_HIP_REAR_BIAS,
    )
    assert front - old_front >= 0.010 - 1e-9


def test_t3_rear_holds_and_pelvis_kiss_ok() -> None:
    """T3: hip_rear holds vs 0.70/0.22; kiss-ok vs pelvis rear."""
    m = _metrics_for_ovals(chest_y=0.0)
    m.hip_y = 0.0347  # product-like pelvis cy so kiss is ~0.2 mm, not 35 mm
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    hip = by[_HIP]
    assert hip.center is not None and hip.ry_m is not None
    rear = float(hip.center[1]) + float(hip.ry_m)
    _old_ry, _old_front, old_rear = _hip_poles(
        half=_PRODUCT_HALF_HIP,
        ry_frac=_OLD_RY_HIP_FRAC,
        bias=_OLD_HIP_REAR_BIAS,
    )
    assert abs(rear - old_rear) <= 0.002 + 1e-12
    pelvis = by[_PELVIS]
    assert pelvis.center is not None and pelvis.ry_m is not None
    pelvis_rear = float(pelvis.center[1]) + float(pelvis.ry_m)
    assert rear >= pelvis_rear - 1e-3
    assert abs(rear - pelvis_rear) <= 0.002 + 1e-12


def test_t4_product_like_ry_order() -> None:
    """T4: product-like halves: ry_c > ry_h > ry_p and ry_c > ry_w."""
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
    assert ry_c > ry_h + 1e-9
    assert ry_h > ry_p + 1e-9
    assert ry_c > ry_w + 1e-9


def test_t5_ry_over_rz_less_ball() -> None:
    """T5: ry_hip/rz_hip < 0.80 on product-like (was ~0.835). Not < 0.70."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(taper=0.22),
        **_product_flags(),  # type: ignore[arg-type]
    )
    by = {p.name: p for p in pkg.parts}
    hip = by[_HIP]
    ry_h = float(hip.ry_m or 0.0)
    rz_h = float(hip.rz_m or 0.0)
    assert rz_h > 1e-9
    assert ry_h / rz_h < 0.80


def test_t6_front_plane_no_rear_bias_still_064() -> None:
    """T6: front_plane cy stays mid; no rear bias; ry still 0.64."""
    msgs: list[str] = []
    m = _metrics_for_ovals(chest_y=None)
    parts = _build_torso_ovals(m, msgs, taper=0.14)
    by = {p.name: p for p in parts}
    hip = by[_HIP]
    assert hip.center is not None and hip.ry_m is not None
    assert hip.placement == "front_plane"
    assert float(hip.center[1]) == pytest.approx(0.0, abs=1e-9)
    assert TORSO_OVAL_RY_HIP_FRAC == 0.64
    assert float(hip.ry_m) == pytest.approx(_PRODUCT_HALF_HIP * TORSO_OVAL_RY_HIP_FRAC, abs=1e-9)


def test_t7_hip_hierarchy_and_mid_back_messages() -> None:
    """T7: sibling hip-hierarchy line + 0074 hip_rear_bias= still present."""
    report = _full_torso_report(chest_mid_y=0.0)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    hip_msg = [m for m in pkg.messages if m.startswith("torso hip hierarchy:")]
    mid = [m for m in pkg.messages if m.startswith("torso mid-back:")]
    thoracic = [m for m in pkg.messages if m.startswith("torso thoracic front:")]
    assert len(hip_msg) == 1
    assert "ry_frac=" in hip_msg[0]
    assert "bias=" in hip_msg[0]
    assert f"ry_frac={TORSO_OVAL_RY_HIP_FRAC}" in hip_msg[0]
    assert f"bias={TORSO_HIP_Y_REAR_BIAS_FRAC_RY}" in hip_msg[0]
    assert "front=" in hip_msg[0]
    assert "rear=" in hip_msg[0]
    assert len(mid) == 1
    assert "hip_rear_bias=" in mid[0]
    assert len(thoracic) == 1
    by = {p.name: p for p in pkg.parts}
    hip = by[_HIP]
    assert hip.center is not None and hip.ry_m is not None
    front = float(hip.center[1]) - float(hip.ry_m)
    rear = float(hip.center[1]) + float(hip.ry_m)
    assert f"front={front:.4f}" in hip_msg[0]
    assert f"rear={rear:.4f}" in hip_msg[0]


def test_t8_cluster_fence_iliac_hip_soft_glute() -> None:
    """T8: iliac 0; hip_soft L/R present; glute pair present."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(taper=0.22),
        **_product_flags(),  # type: ignore[arg-type]
    )
    iliac = [p for p in pkg.parts if p.role == "iliac_soft"]
    assert len(iliac) == 0
    assert not any(p.name.startswith("RECIPE_iliac_soft_") for p in pkg.parts)
    by = {p.name: p for p in pkg.parts}
    assert "RECIPE_hip_soft_l" in by
    assert "RECIPE_hip_soft_r" in by
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) == 2


def test_t9_product_n_parts_131_schema_mcp() -> None:
    """T9: n_parts 131 via hair=short + profile; schema 1.4.0; MCP 46."""
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
    assert len(TOOL_NAMES) == 47
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert "C_thigh_outer" in by_id
    assert "C_glute_outer" in by_id
    assert "C_palm_ellipsoid" in by_id
    assert by_id["C_palm_ellipsoid"].status == "pass"
    assert by_id["C_glute_outer"].status == "pass"
    # Plan 1D: fixture C_thigh_outer ΔX is not chased. Product JSON is B18 SoT.
    cons_path = _PRODUCT_NOFUSE / "constraints_report.json"
    if cons_path.is_file():
        cons = json.loads(cons_path.read_text(encoding="utf-8"))
        rule_by = {r["id"]: r["status"] for r in cons.get("rules", [])}
        assert rule_by.get("C_thigh_outer") == "pass"
        assert rule_by.get("C_glute_outer") == "pass"


def test_t10_chest_poles_unchanged() -> None:
    """T10: 0090 chest poles hold 0.72/0.51 formula on product-like."""
    m = _metrics_for_ovals(chest_y=0.0)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    chest = by[_CHEST]
    assert chest.center is not None and chest.ry_m is not None
    ry = float(chest.ry_m)
    cy = float(chest.center[1])
    front = cy - ry
    rear = cy + ry
    expected_ry = _PRODUCT_HALF_CHEST * TORSO_OVAL_RY_CHEST_FRAC
    expected_cy = 0.0 + TORSO_CHEST_Y_REAR_BIAS_FRAC_RY * expected_ry
    assert ry == pytest.approx(expected_ry, abs=1e-9)
    assert front == pytest.approx(expected_cy - expected_ry, abs=1e-9)
    assert rear == pytest.approx(expected_cy + expected_ry, abs=1e-9)


def test_t11_mid_back_z_and_waist_bias() -> None:
    """T11: 0093 mid_back z ~ waist - 0.035; waist bias still 0.42 (not 0.33)."""
    assert TORSO_WAIST_Y_REAR_BIAS_FRAC_RY == 0.42
    assert TORSO_WAIST_Y_REAR_BIAS_FRAC_RY != TORSO_HIP_Y_REAR_BIAS_FRAC_RY
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


def test_t12_all_exports_hip_ry_and_bias() -> None:
    """T12: __all__ still exports hip ry + rear-bias consts."""
    from meshops.proportion import blockout_recipe as br

    names = set(br.__all__)
    assert "TORSO_OVAL_RY_HIP_FRAC" in names
    assert "TORSO_HIP_Y_REAR_BIAS_FRAC_RY" in names


def test_t13_hip_soft_past_cap() -> None:
    """T13: 0069 past-cap still holds (|hip_soft outer| > |thigh cap|)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(taper=0.22),
        **_product_flags(),  # type: ignore[arg-type]
    )
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by[f"RECIPE_hip_soft_{side}"]
        thigh = by[f"RECIPE_limb_thigh_{side}"]
        assert soft.center is not None and soft.rx_m is not None
        assert thigh.p0 is not None and thigh.radius_m is not None
        soft_outer = abs(float(soft.center[0])) + float(soft.rx_m)
        thigh_cap = abs(float(thigh.p0[0])) + float(thigh.radius_m)
        assert soft_outer > thigh_cap - 1e-4


def test_t14_form_fixture_ry_order() -> None:
    """T14: form-fixture halves 0.12/0.13 restore ry_c > ry_h > ry_p."""
    half_chest = 0.12
    half_hip = 0.13
    m = _metrics_for_ovals(chest_y=0.0, half_chest=half_chest, half_hip=half_hip)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    ry_c = float(by[_CHEST].ry_m or 0.0)
    ry_h = float(by[_HIP].ry_m or 0.0)
    ry_p = half_hip * PELVIS_OVAL_RY_FRAC_HALF_HIP
    assert ry_c > ry_h + 1e-9
    assert ry_h > ry_p + 1e-9


def test_t15_glute_front_still_anterior_of_hip_plate() -> None:
    """T15: document glute front ~-0.076 more anterior than hip plate (0068)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(taper=0.22),
        **_product_flags(),  # type: ignore[arg-type]
    )
    by = {p.name: p for p in pkg.parts}
    hip = by[_HIP]
    assert hip.center is not None and hip.ry_m is not None
    hip_front = float(hip.center[1]) - float(hip.ry_m)
    # B20 product pin (0068 cy=0.045, ry=0.1212). Fixture meters may differ;
    # document that the new hip plate sits behind that reach. Do not retune
    # glute or fail if live fixture glute is still proud.
    expected_glute_front = -0.076
    assert hip_front > expected_glute_front + 1e-9
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) == 2
    for g in glutes:
        assert g.center is not None and g.ry_m is not None
        glute_front = float(g.center[1]) - float(g.ry_m)
        assert math.isfinite(glute_front)
        # 0068 seat still +Y; do not assert recede toward the hip plate.
        assert float(g.center[1]) >= 0.0
    # B18 product SoT: live recipe documents 0068 reach still anterior of hip plate.
    recipe_path = _PRODUCT_NOFUSE / "blockout_recipe.json"
    if recipe_path.is_file():
        data = json.loads(recipe_path.read_text(encoding="utf-8"))
        parts = {p["name"]: p for p in data["parts"]}
        hip_p = parts[_HIP]
        glute_p = parts["RECIPE_glute_soft_l"]
        p_hip_front = float(hip_p["center"][1]) - float(hip_p["ry_m"])
        p_glute_front = float(glute_p["center"][1]) - float(glute_p["ry_m"])
        assert p_glute_front == pytest.approx(-0.076, abs=0.003)
        assert p_glute_front < p_hip_front - 1e-9
