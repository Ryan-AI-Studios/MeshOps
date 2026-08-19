"""Track 0089 — torso continuous silhouette (z_norm pull + overlap 0.070).

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
    RECIPE_SCHEMA_VERSION,
    TORSO_OVAL_OVERLAP_FLOOR_M,
    TORSO_OVAL_RY_HIP_FRAC,
    TORSO_OVAL_RZ_CHEST_FRAC,
    TORSO_OVAL_RZ_GROW_CAP_M,
    TORSO_OVAL_RZ_HIP_FRAC,
    TORSO_OVAL_RZ_WAIST_FRAC,
    TORSO_OVAL_Z_NORM_CHEST,
    TORSO_OVAL_Z_NORM_HIP,
    TORSO_OVAL_Z_NORM_WAIST,
    TORSO_WAIST_RX_MAX_FRAC_CHEST,
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

# 0073 even-thirds (not public law) — T1 relative pull baseline only.
_OLD_Z_NORM_CHEST = 0.15
_OLD_Z_NORM_WAIST = 0.50
_OLD_Z_NORM_HIP = 0.85


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
    half_chest: float = 0.12,
    half_hip: float = 0.13,
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


def _span_from_metrics(m: _ResolvedMetrics) -> float:
    z_candidates = [z for z in (m.shoulder_z, m.chest_z) if z is not None]
    assert z_candidates and m.hip_z is not None
    return max(z_candidates) - m.hip_z


def _z_top_from_metrics(m: _ResolvedMetrics) -> float:
    z_candidates = [z for z in (m.shoulder_z, m.chest_z) if z is not None]
    assert z_candidates
    return max(z_candidates)


def _planned_rz(span: float) -> dict[str, float]:
    return {
        _CHEST: max(0.025, span * TORSO_OVAL_RZ_CHEST_FRAC),
        _WAIST: max(0.025, span * TORSO_OVAL_RZ_WAIST_FRAC),
        _HIP: max(0.025, span * TORSO_OVAL_RZ_HIP_FRAC),
    }


def _pair_overlaps(by: dict[str, RecipePart]) -> tuple[float, float]:
    c = by[_CHEST]
    w = by[_WAIST]
    h = by[_HIP]
    assert c.center is not None and w.center is not None and h.center is not None
    assert c.rz_m is not None and w.rz_m is not None and h.rz_m is not None
    ov_cw = float(c.rz_m) + float(w.rz_m) - abs(float(c.center[2]) - float(w.center[2]))
    ov_wh = float(w.rz_m) + float(h.rz_m) - abs(float(w.center[2]) - float(h.center[2]))
    return ov_cw, ov_wh


def test_t0_public_z_norm_and_overlap_floor() -> None:
    """T0: named z_norm 0.18/0.50/0.82; rz fracs + grow cap unchanged; overlap 0.070."""
    assert TORSO_OVAL_Z_NORM_CHEST == 0.18
    assert TORSO_OVAL_Z_NORM_WAIST == 0.50
    assert TORSO_OVAL_Z_NORM_HIP == 0.82
    assert 0.16 <= TORSO_OVAL_Z_NORM_CHEST <= 0.22
    assert TORSO_OVAL_Z_NORM_WAIST == 0.50
    assert 0.78 <= TORSO_OVAL_Z_NORM_HIP <= 0.84
    assert TORSO_OVAL_RZ_CHEST_FRAC == 0.28
    assert TORSO_OVAL_RZ_WAIST_FRAC == 0.16
    assert TORSO_OVAL_RZ_HIP_FRAC == 0.24
    assert TORSO_OVAL_RZ_GROW_CAP_M == 0.030
    assert TORSO_OVAL_OVERLAP_FLOOR_M == 0.070
    assert 0.060 <= TORSO_OVAL_OVERLAP_FLOOR_M <= 0.080


def test_t1_layer_z_pull_relative_to_even_thirds() -> None:
    """T1: chest z < 0.15-class; hip z > 0.85-class; waist Δz < 2 mm."""
    m = _metrics_for_ovals(chest_y=0.0)
    msgs: list[str] = []
    parts = _build_torso_ovals(m, msgs, taper=0.22)
    by = {p.name: p for p in parts}
    z_top = _z_top_from_metrics(m)
    span = _span_from_metrics(m)
    old_c = z_top - _OLD_Z_NORM_CHEST * span
    old_w = z_top - _OLD_Z_NORM_WAIST * span
    old_h = z_top - _OLD_Z_NORM_HIP * span
    c_c, c_w, c_h = by[_CHEST].center, by[_WAIST].center, by[_HIP].center
    assert c_c is not None and c_w is not None and c_h is not None
    z_c = float(c_c[2])
    z_w = float(c_w[2])
    z_h = float(c_h[2])
    assert z_c < old_c - 1e-9
    assert z_h > old_h + 1e-9
    assert abs(z_w - old_w) < 0.002
    assert z_c == pytest.approx(z_top - TORSO_OVAL_Z_NORM_CHEST * span, abs=1e-9)
    assert z_w == pytest.approx(z_top - TORSO_OVAL_Z_NORM_WAIST * span, abs=1e-9)
    assert z_h == pytest.approx(z_top - TORSO_OVAL_Z_NORM_HIP * span, abs=1e-9)


def test_t2_pairwise_overlap_floor() -> None:
    """T2: pairwise overlap ≥ 0.070 after B2 grow."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    ov_cw, ov_wh = _pair_overlaps(by)
    assert ov_cw >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9
    assert ov_wh >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9


def test_t3_rz_order_not_equal_triad() -> None:
    """T3: not all-three rz equal; chest > hip > waist (waist thinnest)."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    rz_c = float(by[_CHEST].rz_m or 0.0)
    rz_w = float(by[_WAIST].rz_m or 0.0)
    rz_h = float(by[_HIP].rz_m or 0.0)
    eps = 1e-6
    assert not (abs(rz_c - rz_w) < eps and abs(rz_w - rz_h) < eps)
    assert rz_w < rz_h - eps
    assert rz_h < rz_c - eps


def test_t4_pinch_cap_when_taper_gate_on() -> None:
    """T4: 0065 pinch rx_w/rx_c ≤ 0.80 when taper gate on."""
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


def test_t5_hip_ry_frac_and_pelvis_order() -> None:
    """T5: hip ry frac 0.64; ry_hip > ry_pelvis."""
    half_hip = 0.13
    report = _full_torso_report(hip_depth_m=0.26)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    ry_h = float(by[_HIP].ry_m or 0.0)
    ry_p = float(by["RECIPE_pelvis_oval"].ry_m or 0.0)
    assert ry_h == pytest.approx(half_hip * TORSO_OVAL_RY_HIP_FRAC, abs=1e-9)
    assert TORSO_OVAL_RY_HIP_FRAC == 0.64
    assert ry_h > ry_p + 1e-9


def test_t6_rear_bias_full3d_and_front_plane_mid() -> None:
    """T6: full3d waist/hip cy > chest_y; front_plane cy mid."""
    report = _full_torso_report(chest_mid_y=0.0)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    chest_y = 0.0
    w_c, h_c = by[_WAIST].center, by[_HIP].center
    assert w_c is not None and h_c is not None
    assert float(w_c[1]) > chest_y + 1e-9
    assert float(h_c[1]) > chest_y + 1e-9

    msgs: list[str] = []
    m = _metrics_for_ovals(chest_y=None)
    parts = _build_torso_ovals(m, msgs, taper=0.14)
    by_fp = {p.name: p for p in parts}
    for name in (_CHEST, _WAIST, _HIP):
        fp_c = by_fp[name].center
        assert fp_c is not None
        assert by_fp[name].placement == "front_plane"
        assert float(fp_c[1]) == pytest.approx(0.0, abs=1e-9)


def test_t7_form_message_includes_z_norm() -> None:
    """T7: form silhouette line includes z_norm= 0.18/0.50/0.82."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    form = [m for m in pkg.messages if m.startswith("torso form silhouette:")]
    assert len(form) == 1
    msg = form[0]
    assert "z_norm=" in msg
    assert "0.18/0.50/0.82" in msg
    want = (
        f"z_norm=c/w/h={TORSO_OVAL_Z_NORM_CHEST:.2f}/"
        f"{TORSO_OVAL_Z_NORM_WAIST:.2f}/{TORSO_OVAL_Z_NORM_HIP:.2f}"
    )
    assert want in msg


def test_t8_front_plane_still_b1_b3() -> None:
    """T8: front_plane still applies z_norm + overlap grow; no rear bias."""
    msgs: list[str] = []
    m = _metrics_for_ovals(chest_y=None)
    parts = _build_torso_ovals(m, msgs, taper=0.14)
    by = {p.name: p for p in parts}
    z_top = _z_top_from_metrics(m)
    span = _span_from_metrics(m)
    chest_c = by[_CHEST].center
    assert chest_c is not None
    assert float(chest_c[2]) == pytest.approx(z_top - TORSO_OVAL_Z_NORM_CHEST * span, abs=1e-9)
    assert float(chest_c[1]) == pytest.approx(0.0, abs=1e-9)
    ov_cw, ov_wh = _pair_overlaps(by)
    assert ov_cw >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9
    assert ov_wh >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9


def test_t9_product_n_parts_131_schema_mcp() -> None:
    """T9: n_parts 131 via hair=short + profile; schema 1.4.0; MCP 46."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert len(TOOL_NAMES) == 47
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert "C_palm_ellipsoid" in by_id
    assert by_id["C_palm_ellipsoid"].status == "pass"


def test_t10_grow_respects_cap_never_shrink() -> None:
    """T10: 0073 grow still respects cap 0.030; never shrink."""
    shoulder_z = 1.30
    hip_z = 0.95
    span = shoulder_z - hip_z
    msgs: list[str] = []
    m = _metrics_for_ovals(shoulder_z=shoulder_z, hip_z=hip_z, chest_y=0.0)
    parts = _build_torso_ovals(m, msgs, taper=0.14)
    by = {p.name: p for p in parts}
    planned = _planned_rz(span)
    for name in (_CHEST, _WAIST, _HIP):
        grown = float(by[name].rz_m or 0.0) - planned[name]
        assert grown >= -1e-12, f"{name} shrunk"
        assert grown <= TORSO_OVAL_RZ_GROW_CAP_M + 1e-9


def test_t11_mid_back_follows_waist_hang_applied() -> None:
    """T11: mid_back z ~ waist - z_below; breast hang applied (do not pin z=1.228)."""
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
    breasts = [p for p in pkg.parts if p.role == "breast_soft"]
    assert breasts
    assert any("breast_hang_z_applied: true" in m for m in pkg.messages)
    chest = by[_CHEST]
    chest_c = chest.center
    assert chest_c is not None
    chest_z = float(chest_c[2])
    ref_msgs = [m for m in pkg.messages if m.startswith("breast_hang_z_chest_ref_m=")]
    assert ref_msgs, "B15 hang band must emit chest-ref (follows oval z)"
    ref_z = float(ref_msgs[0].split("=", 1)[1])
    assert ref_z == pytest.approx(chest_z, abs=2e-3)
    for b in breasts:
        assert b.center is not None
        assert math.isfinite(float(b.center[2]))


def test_t12_all_exports_z_norm_consts() -> None:
    """T12: __all__ exports the three TORSO_OVAL_Z_NORM_* consts."""
    from meshops.proportion import blockout_recipe as br

    names = set(br.__all__)
    assert "TORSO_OVAL_Z_NORM_CHEST" in names
    assert "TORSO_OVAL_Z_NORM_WAIST" in names
    assert "TORSO_OVAL_Z_NORM_HIP" in names
