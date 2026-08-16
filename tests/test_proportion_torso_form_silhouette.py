"""Track 0073 — torso form silhouette anti-stack (asymmetric rz + overlap grow; authoring only)."""

from __future__ import annotations

import math

import pytest

from meshops.proportion.blockout_recipe import (
    TORSO_CHEST_Y_REAR_BIAS_FRAC_RY,
    TORSO_OVAL_OVERLAP_FLOOR_M,
    TORSO_OVAL_RY_CHEST_FRAC,
    TORSO_OVAL_RY_HIP_FRAC,
    TORSO_OVAL_RY_WAIST_FRAC,
    TORSO_OVAL_RZ_CHEST_FRAC,
    TORSO_OVAL_RZ_FLOOR_M,
    TORSO_OVAL_RZ_GROW_CAP_M,
    TORSO_OVAL_RZ_HIP_FRAC,
    TORSO_OVAL_RZ_SPAN_FRAC,
    TORSO_OVAL_RZ_WAIST_FRAC,
    TORSO_OVAL_Z_NORM_CHEST,
    TORSO_OVAL_Z_NORM_HIP,
    TORSO_OVAL_Z_NORM_WAIST,
    TORSO_WAIST_PINCH_TAPER_GATE,
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
from meshops.proportion.connection_metrics import connection_gap_metrics
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.models import (
    CrossSection,
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

_CHEST = "RECIPE_torso_oval_chest"
_WAIST = "RECIPE_torso_oval_waist"
_HIP = "RECIPE_torso_oval_hip"


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
    with_soft_cs: bool = False,
    chest_mid_y: float | None = None,
) -> ProportionReport:
    """Minimal report so torso ovals emit (shoulder/hip hw + chest/hip depth)."""
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
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
    ]
    bands = [
        _band("chest", depth_m=chest_depth_m, z_frac=0.72, y_mid=0.0),
        _band("hip", depth_m=hip_depth_m, z_frac=0.55),
    ]
    cs: list[CrossSection] = []
    if with_soft_cs:
        cs = [
            CrossSection(
                level_id="bust",
                z_frac=0.72,
                rx_frac=0.10,
                ry_frac=0.08,
                sources=["test"],
            ),
            CrossSection(
                level_id="glute",
                z_frac=0.50,
                rx_frac=0.11,
                ry_frac=0.09,
                sources=["test"],
            ),
        ]
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=bands,
        cross_sections=cs,
        quality=QualityFlags(),
    )


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


def _planned_rz(span: float) -> dict[str, float]:
    return {
        _CHEST: max(TORSO_OVAL_RZ_FLOOR_M, span * TORSO_OVAL_RZ_CHEST_FRAC),
        _WAIST: max(TORSO_OVAL_RZ_FLOOR_M, span * TORSO_OVAL_RZ_WAIST_FRAC),
        _HIP: max(TORSO_OVAL_RZ_FLOOR_M, span * TORSO_OVAL_RZ_HIP_FRAC),
    }


def _pair_overlaps(by: dict[str, RecipePart]) -> tuple[float, float]:
    c = by[_CHEST]
    w = by[_WAIST]
    h = by[_HIP]
    assert c.center is not None and w.center is not None and h.center is not None
    assert c.rz_m is not None and w.rz_m is not None and h.rz_m is not None
    z_c = float(c.center[2])
    z_w = float(w.center[2])
    z_h = float(h.center[2])
    rz_c = float(c.rz_m)
    rz_w = float(w.rz_m)
    rz_h = float(h.rz_m)
    ov_cw = rz_c + rz_w - abs(z_c - z_w)
    ov_wh = rz_w + rz_h - abs(z_w - z_h)
    return ov_cw, ov_wh


# ---------------------------------------------------------------------------
# T0-T13
# ---------------------------------------------------------------------------


def test_t0_public_freezes_exported_in_bands() -> None:
    """T0: public freezes exported; within §0 bands; GROW_CAP=0.030; OVERLAP=0.070."""
    assert 0.26 <= TORSO_OVAL_RZ_CHEST_FRAC <= 0.32
    assert 0.14 <= TORSO_OVAL_RZ_WAIST_FRAC <= 0.18
    assert 0.22 <= TORSO_OVAL_RZ_HIP_FRAC <= 0.28
    assert 0.060 <= TORSO_OVAL_OVERLAP_FLOOR_M <= 0.080
    assert 0.025 <= TORSO_OVAL_RZ_GROW_CAP_M <= 0.035
    assert 0.65 <= TORSO_OVAL_RY_HIP_FRAC <= 0.75
    # exact defaults
    assert TORSO_OVAL_RZ_CHEST_FRAC == 0.28
    assert TORSO_OVAL_RZ_WAIST_FRAC == 0.16
    assert TORSO_OVAL_RZ_HIP_FRAC == 0.24
    assert TORSO_OVAL_OVERLAP_FLOOR_M == 0.070
    assert TORSO_OVAL_RZ_GROW_CAP_M == 0.030
    assert TORSO_OVAL_RY_HIP_FRAC == 0.70
    assert TORSO_OVAL_RZ_FLOOR_M == 0.025
    # B4 legacy fence symbol only
    assert TORSO_OVAL_RZ_SPAN_FRAC == 0.22
    # 0065 keep
    assert TORSO_OVAL_RY_CHEST_FRAC == 0.85
    assert TORSO_OVAL_RY_WAIST_FRAC == 0.58
    assert TORSO_WAIST_RX_MAX_FRAC_CHEST == 0.80
    assert TORSO_WAIST_PINCH_TAPER_GATE == 0.10
    assert TORSO_CHEST_Y_REAR_BIAS_FRAC_RY == 0.28


def test_t1_rz_at_least_planned_b1() -> None:
    """T1: each rz >= B1 planned (max floor, span*frac); may exceed via B2 only."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    # z_top from shoulder 1.38 / chest_front 1.25 / band z_frac 0.72*1.72
    shoulder_z = 1.38
    hip_z = 0.95
    chest_z = 0.72 * 1.72
    span = max(shoulder_z, chest_z) - hip_z
    planned = _planned_rz(span)
    for name, p_rz in planned.items():
        rz = float(by[name].rz_m or 0.0)
        assert rz >= p_rz - 1e-9
        # grow never exceeds planned + GROW_CAP
        assert rz <= p_rz + TORSO_OVAL_RZ_GROW_CAP_M + 1e-9


def test_t2_pairwise_overlap_floor() -> None:
    """T2: pairwise overlap ≥ TORSO_OVAL_OVERLAP_FLOOR_M after B2."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    ov_cw, ov_wh = _pair_overlaps(by)
    assert ov_cw >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9
    assert ov_wh >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9


def test_t3_not_equal_tire_triad_rz() -> None:
    """T3: not all three rz equal within 1e-6 (anti equal-tire triad)."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    rz_c = float(by[_CHEST].rz_m or 0.0)
    rz_w = float(by[_WAIST].rz_m or 0.0)
    rz_h = float(by[_HIP].rz_m or 0.0)
    eps = 1e-6
    assert not (abs(rz_c - rz_w) < eps and abs(rz_w - rz_h) < eps)
    # Stronger: waist ≠ chest (B2 may near-equal waist/hip)
    assert abs(rz_w - rz_c) > eps


def test_t4_ry_magnitudes_from_fracs() -> None:
    """T4: ry magnitudes — chest 0.85; waist 0.58; hip 0.70 of half_*."""
    half_chest = 0.12
    half_hip = 0.13
    report = _full_torso_report(chest_depth_m=0.24, hip_depth_m=0.26)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    assert float(by[_CHEST].ry_m or 0.0) == pytest.approx(
        half_chest * TORSO_OVAL_RY_CHEST_FRAC, abs=1e-9
    )
    assert float(by[_WAIST].ry_m or 0.0) == pytest.approx(
        half_chest * TORSO_OVAL_RY_WAIST_FRAC, abs=1e-9
    )
    assert float(by[_HIP].ry_m or 0.0) == pytest.approx(half_hip * TORSO_OVAL_RY_HIP_FRAC, abs=1e-9)


def test_t5_ry_hip_order_vs_chest_pelvis() -> None:
    """T5: ry_hip > ry_pelvis; ry_hip < ry_chest."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    ry_c = float(by[_CHEST].ry_m or 0.0)
    ry_h = float(by[_HIP].ry_m or 0.0)
    ry_p = float(by["RECIPE_pelvis_oval"].ry_m or 0.0)
    eps = 1e-9
    assert ry_h > ry_p + eps
    assert ry_h < ry_c - eps


def test_t6_front_pinch_and_chest_rear_bias() -> None:
    """T6: 0065 rx cap + full3d chest cy bias still hold."""
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
    chest = by[_CHEST]
    assert chest.center is not None and chest.ry_m is not None
    ry = float(chest.ry_m)
    assert chest.center[1] == pytest.approx(TORSO_CHEST_Y_REAR_BIAS_FRAC_RY * ry, abs=1e-9)
    assert chest.placement == "full3d"


def test_t7_front_plane_no_bias_still_b1_b2() -> None:
    """T7: front_plane — no chest bias and B1+B2 still apply (B15)."""
    msgs: list[str] = []
    m = _metrics_for_ovals(chest_y=None)
    parts = _build_torso_ovals(m, msgs, taper=0.14)
    by = {p.name: p for p in parts}
    chest = by[_CHEST]
    assert chest.center is not None
    assert chest.placement == "front_plane"
    assert chest.center[1] == pytest.approx(0.0, abs=1e-9)
    span = _span_from_metrics(m)
    planned = _planned_rz(span)
    for name, p_rz in planned.items():
        rz = float(by[name].rz_m or 0.0)
        assert rz >= p_rz - 1e-9
    ov_cw, ov_wh = _pair_overlaps(by)
    assert ov_cw >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9
    assert ov_wh >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9
    assert any(msg.startswith("torso form silhouette:") for msg in msgs)


def test_t8_form_silhouette_message_format() -> None:
    """T8: messages — form silhouette line exact prefix + :.4f fields present."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    form = [m for m in pkg.messages if m.startswith("torso form silhouette:")]
    assert len(form) == 1
    msg = form[0]
    by = {p.name: p for p in pkg.parts}
    rz_c = float(by[_CHEST].rz_m or 0.0)
    rz_w = float(by[_WAIST].rz_m or 0.0)
    rz_h = float(by[_HIP].rz_m or 0.0)
    ov_cw, ov_wh = _pair_overlaps(by)
    assert f"rz=c/w/h={rz_c:.4f}/{rz_w:.4f}/{rz_h:.4f}" in msg
    assert f"overlap_cw={ov_cw:.4f}" in msg
    assert f"overlap_wh={ov_wh:.4f}" in msg
    # depth taper + front pinch still present
    assert any(m.startswith("torso depth taper:") for m in pkg.messages)
    assert any(m.startswith("torso front pinch:") for m in pkg.messages)


def test_t9_trap_path_unchanged() -> None:
    """T9: trap path unchanged (default torso emits trap, not ovals)."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False)  # default trap
    names = {p.name for p in pkg.parts}
    assert "RECIPE_torso_trap" in names
    assert _CHEST not in names
    assert not any(m.startswith("torso form silhouette:") for m in pkg.messages)
    assert not any(m.startswith("torso front pinch:") for m in pkg.messages)


def test_t10_connection_gap_and_constraints() -> None:
    """T10: connection gap finite; constraints validate on product-like recipe."""
    report = _full_torso_report(with_soft_cs=True, chest_mid_y=0.0)
    report = report.model_copy(
        update={
            "landmarks_xyz": {
                **report.landmarks_xyz,
                "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.13, z_m=1.25),
            }
        }
    )
    pkg = build_blockout_recipe(report, limbs=True, torso="ovals")
    gaps = connection_gap_metrics(pkg)
    for key, val in gaps.items():
        assert math.isfinite(val), f"{key} not finite: {val}"
    assert "shoulder_l" in gaps and "shoulder_r" in gaps
    assert gaps["shoulder_l"] < 1e8
    assert gaps["shoulder_r"] < 1e8
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    axial = by_id.get("C_axial_depth_plane")
    assert axial is not None
    assert axial.status == "pass", axial.message


def test_t11_b2_preference_waist_grows_first() -> None:
    """T11: B2 preference - force shortfall -> waist grows first; grown <= GROW_CAP.

    Geometry: span <= ~0.375 m so pre-grow planned-rz minus layer Δz_norm < floor.
    """
    # span = 0.35 m -> pre-grow shortfall forces grow
    shoulder_z = 1.30
    hip_z = 0.95
    span = shoulder_z - hip_z
    assert span <= 0.375 + 1e-9
    dz_cw = TORSO_OVAL_Z_NORM_WAIST - TORSO_OVAL_Z_NORM_CHEST
    dz_wh = TORSO_OVAL_Z_NORM_HIP - TORSO_OVAL_Z_NORM_WAIST
    pre_ov_cw = (TORSO_OVAL_RZ_CHEST_FRAC + TORSO_OVAL_RZ_WAIST_FRAC - dz_cw) * span
    pre_ov_wh = (TORSO_OVAL_RZ_WAIST_FRAC + TORSO_OVAL_RZ_HIP_FRAC - dz_wh) * span
    assert pre_ov_cw < TORSO_OVAL_OVERLAP_FLOOR_M
    assert pre_ov_wh < TORSO_OVAL_OVERLAP_FLOOR_M

    msgs: list[str] = []
    m = _metrics_for_ovals(shoulder_z=shoulder_z, hip_z=hip_z, chest_y=0.0)
    parts = _build_torso_ovals(m, msgs, taper=0.14)
    by = {p.name: p for p in parts}
    planned = _planned_rz(span)
    grown = {name: float(by[name].rz_m or 0.0) - planned[name] for name in (_CHEST, _WAIST, _HIP)}
    # Waist grows first (preference); chest must not grow before waist exhausted.
    assert grown[_WAIST] > 1e-9
    assert grown[_WAIST] <= TORSO_OVAL_RZ_GROW_CAP_M + 1e-9
    # Prefer waist: either hip/chest ungrown while waist took first delta, or
    # waist at cap before others if shortfall required more.
    if grown[_CHEST] > 1e-9:
        assert grown[_WAIST] >= TORSO_OVAL_RZ_GROW_CAP_M - 1e-9
    if grown[_HIP] > 1e-9:
        # hip only after waist at cap on the short (wh) pair
        assert (
            grown[_WAIST]
            >= min(
                TORSO_OVAL_RZ_GROW_CAP_M,
                TORSO_OVAL_OVERLAP_FLOOR_M - pre_ov_wh,
            )
            - 1e-6
        )
    for name, g in grown.items():
        assert g >= -1e-12, f"{name} shrunk"
        assert g <= TORSO_OVAL_RZ_GROW_CAP_M + 1e-9
    ov_cw, ov_wh = _pair_overlaps(by)
    assert ov_cw >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9
    assert ov_wh >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9


def test_t12_anti_equal_triad_rx_ry_rz() -> None:
    """T12: anti-regression — not equal triad rx/ry; not equal triad rz."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(
        report,
        limbs=False,
        torso="ovals",
        template_applied=_template(taper=0.22),
    )
    by = {p.name: p for p in pkg.parts}
    names = (_CHEST, _WAIST, _HIP)
    rx = [float(by[n].rx_m or 0.0) for n in names]
    ry = [float(by[n].ry_m or 0.0) for n in names]
    rz = [float(by[n].rz_m or 0.0) for n in names]
    eps = 1e-9
    assert not (abs(rx[0] - rx[1]) < eps and abs(rx[1] - rx[2]) < eps)
    assert not (abs(ry[0] - ry[1]) < eps and abs(ry[1] - ry[2]) < eps)
    assert not (abs(rz[0] - rz[1]) < eps and abs(rz[1] - rz[2]) < eps)


def test_t13_floor_and_early_return_skip() -> None:
    """T13: tiny span still rz ≥ 0.025; early-return skip paths return empty."""
    # Tiny span → floor binds on B1
    msgs: list[str] = []
    m = _metrics_for_ovals(shoulder_z=1.00, hip_z=0.99, height_m=1.72)
    parts = _build_torso_ovals(m, msgs, taper=0.14)
    by = {p.name: p for p in parts}
    for name in (_CHEST, _WAIST, _HIP):
        assert float(by[name].rz_m or 0.0) >= TORSO_OVAL_RZ_FLOOR_M - 1e-9

    # Early-return: missing shoulder/hip hw
    msgs2: list[str] = []
    m2 = _ResolvedMetrics()
    m2.hip_z = 0.95
    empty = _build_torso_ovals(m2, msgs2, taper=0.14)
    assert empty == []
    assert any("skipped" in msg for msg in msgs2)

    # Early-return: missing hip_z
    msgs3: list[str] = []
    m3 = _ResolvedMetrics()
    m3.shoulder_hw = 0.20
    m3.hip_hw = 0.14
    empty3 = _build_torso_ovals(m3, msgs3, taper=0.14)
    assert empty3 == []
    assert any("hip_z" in msg for msg in msgs3)
