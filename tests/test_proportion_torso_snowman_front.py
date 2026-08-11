"""Track 0065 - torso snowman front (waist rx pinch + chest flatten; authoring only)."""

from __future__ import annotations

import math

import pytest

from meshops.proportion.blockout_recipe import (
    TORSO_CHEST_Y_REAR_BIAS_FRAC_RY,
    TORSO_OVAL_RY_CHEST_FRAC,
    TORSO_OVAL_RY_HIP_FRAC,
    TORSO_OVAL_RY_WAIST_FRAC,
    TORSO_OVAL_RZ_FLOOR_M,
    TORSO_OVAL_RZ_SPAN_FRAC,
    TORSO_WAIST_PINCH_TAPER_GATE,
    TORSO_WAIST_RX_MAX_FRAC_CHEST,
    _build_torso_ovals,
    _ResolvedMetrics,
    _waist_width_at,
    build_blockout_recipe,
)
from meshops.proportion.body_template import (
    AppliedConstants,
    TemplateAppliedPackage,
    load_body_template,
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


def _template(
    *,
    taper: float,
    template_id: str = "female_adult_athletic",
    sex: str = "female",
) -> TemplateAppliedPackage:
    constants = AppliedConstants(
        breast_mode="dual_tilted" if sex == "female" else "none",
        glute_mode_default="two_spheres" if sex == "female" else "oval",
        torso_mode_default="ovals",
        torso_waist_taper=taper,
    )
    return TemplateAppliedPackage(
        template_id=template_id,
        sex=sex,  # type: ignore[arg-type]
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
) -> _ResolvedMetrics:
    m = _ResolvedMetrics()
    m.shoulder_hw = shoulder_hw
    m.hip_hw = hip_hw
    m.chest_half_depth = half_chest
    m.hip_half_depth = half_hip
    m.shoulder_z = shoulder_z
    m.hip_z = hip_z
    m.chest_y = chest_y
    m.height_m = height_m
    return m


# ---------------------------------------------------------------------------
# T0-T13
# ---------------------------------------------------------------------------


def test_t0_public_freezes_exported_in_bands() -> None:
    """T0: public freezes exported; values within plan open bands."""
    assert 0.76 <= TORSO_WAIST_RX_MAX_FRAC_CHEST <= 0.84
    assert 0.08 <= TORSO_WAIST_PINCH_TAPER_GATE <= 0.12
    assert 0.52 <= TORSO_OVAL_RY_WAIST_FRAC <= 0.64
    assert 0.80 <= TORSO_OVAL_RY_CHEST_FRAC <= 0.90
    assert 0.20 <= TORSO_CHEST_Y_REAR_BIAS_FRAC_RY <= 0.35
    assert TORSO_OVAL_RY_HIP_FRAC == 0.70  # 0073 retarget (was 0.80)
    assert TORSO_OVAL_RZ_SPAN_FRAC == 0.22  # B4 fence symbol only
    assert TORSO_OVAL_RZ_FLOOR_M == 0.025
    # exact freeze defaults
    assert TORSO_WAIST_RX_MAX_FRAC_CHEST == 0.80
    assert TORSO_WAIST_PINCH_TAPER_GATE == 0.10
    assert TORSO_OVAL_RY_CHEST_FRAC == 0.85
    assert TORSO_OVAL_RY_WAIST_FRAC == 0.58
    assert TORSO_CHEST_Y_REAR_BIAS_FRAC_RY == 0.28


def test_t1_taper_ge_gate_waist_rx_capped() -> None:
    """T1: taper >= 0.10 -> hard max *binds* (raw > 0.80x chest, emit = cap)."""
    # Geometry where sin-pinch alone leaves waist > 0.80x post-taper chest.
    report = _full_torso_report(shoulder_x=0.25, hip_x=0.24)
    tpl = _template(taper=0.14)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals", template_applied=tpl)
    by = {p.name: p for p in pkg.parts}
    rx_c = float(by["RECIPE_torso_oval_chest"].rx_m or 0.0)
    rx_w = float(by["RECIPE_torso_oval_waist"].rx_m or 0.0)
    raw_c = _waist_width_at(0.15, 0.25, 0.24, 0.14)
    raw_w = _waist_width_at(0.50, 0.25, 0.24, 0.14)
    assert raw_w > TORSO_WAIST_RX_MAX_FRAC_CHEST * raw_c + 1e-6
    assert rx_c == pytest.approx(raw_c, abs=1e-9)
    assert rx_w == pytest.approx(TORSO_WAIST_RX_MAX_FRAC_CHEST * rx_c, abs=1e-9)
    assert rx_w <= TORSO_WAIST_RX_MAX_FRAC_CHEST * rx_c + 1e-9


def test_t2_male_taper_skips_hard_max() -> None:
    """T2: taper 0.05 -> hard max NOT applied (ratio may exceed 0.80)."""
    report = _full_torso_report(shoulder_x=0.20, hip_x=0.14)
    tpl = _template(taper=0.05, template_id="male_adult_athletic", sex="male")
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals", template_applied=tpl)
    by = {p.name: p for p in pkg.parts}
    rx_c = float(by["RECIPE_torso_oval_chest"].rx_m or 0.0)
    rx_w = float(by["RECIPE_torso_oval_waist"].rx_m or 0.0)
    # Geometry with taper=0.05 yields raw ratio > 0.80 when hard max off.
    raw_c = _waist_width_at(0.15, 0.20, 0.14, 0.05)
    raw_w = _waist_width_at(0.50, 0.20, 0.14, 0.05)
    assert raw_w / raw_c > TORSO_WAIST_RX_MAX_FRAC_CHEST + 1e-6
    assert rx_c == pytest.approx(raw_c, abs=1e-9)
    assert rx_w == pytest.approx(raw_w, abs=1e-9)
    assert rx_w > TORSO_WAIST_RX_MAX_FRAC_CHEST * rx_c + 1e-9


def test_t3_ry_magnitudes_from_fracs() -> None:
    """T3: ry chest half*0.85; waist half*0.58; hip half*0.70."""
    half_chest = 0.12
    half_hip = 0.13
    report = _full_torso_report(chest_depth_m=0.24, hip_depth_m=0.26)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    assert float(by["RECIPE_torso_oval_chest"].ry_m or 0.0) == pytest.approx(
        half_chest * TORSO_OVAL_RY_CHEST_FRAC, abs=1e-9
    )
    assert float(by["RECIPE_torso_oval_waist"].ry_m or 0.0) == pytest.approx(
        half_chest * TORSO_OVAL_RY_WAIST_FRAC, abs=1e-9
    )
    assert float(by["RECIPE_torso_oval_hip"].ry_m or 0.0) == pytest.approx(
        half_hip * TORSO_OVAL_RY_HIP_FRAC, abs=1e-9
    )


def test_t4_full3d_chest_rear_bias() -> None:
    """T4: full3d - chest cy = y_mid + 0.28*ry; waist/hip cy = y_mid."""
    y_mid = 0.0
    report = _full_torso_report(chest_mid_y=y_mid)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    chest = by["RECIPE_torso_oval_chest"]
    waist = by["RECIPE_torso_oval_waist"]
    hip = by["RECIPE_torso_oval_hip"]
    assert chest.center is not None and chest.ry_m is not None
    ry = float(chest.ry_m)
    assert chest.center[1] == pytest.approx(y_mid + TORSO_CHEST_Y_REAR_BIAS_FRAC_RY * ry, abs=1e-9)
    assert waist.center is not None
    assert hip.center is not None
    assert waist.center[1] == pytest.approx(y_mid, abs=1e-9)
    assert hip.center[1] == pytest.approx(y_mid, abs=1e-9)
    assert chest.placement == "full3d"


def test_t4b_front_plane_no_b5_bias() -> None:
    """T4b: front_plane (chest_y None) - chest cy stays y_mid / 0 - no B5 bias."""
    msgs: list[str] = []
    m = _metrics_for_ovals(chest_y=None)
    parts = _build_torso_ovals(m, msgs, taper=0.14)
    by = {p.name: p for p in parts}
    chest = by["RECIPE_torso_oval_chest"]
    assert chest.center is not None
    assert chest.placement == "front_plane"
    assert chest.center[1] == pytest.approx(0.0, abs=1e-9)
    # Not the biased rear position
    ry = float(chest.ry_m or 0.0)
    biased = TORSO_CHEST_Y_REAR_BIAS_FRAC_RY * ry
    assert abs(chest.center[1] - biased) > 1e-4


def test_t5_ry_order_and_hip_gt_pelvis() -> None:
    """T5: ry order + ry_hip > ry_pelvis."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    ry_c = float(by["RECIPE_torso_oval_chest"].ry_m or 0.0)
    ry_w = float(by["RECIPE_torso_oval_waist"].ry_m or 0.0)
    ry_h = float(by["RECIPE_torso_oval_hip"].ry_m or 0.0)
    ry_p = float(by["RECIPE_pelvis_oval"].ry_m or 0.0)
    eps = 1e-9
    assert ry_w < ry_c - eps
    assert ry_h >= ry_w - eps
    assert ry_h > ry_p + eps


def test_t6_rz_span_and_layer_overlap() -> None:
    """T6: per-layer rz ≥ planned fracs; pairwise overlap ≥ OVERLAP_FLOOR (0073)."""
    from meshops.proportion.blockout_recipe import (
        TORSO_OVAL_OVERLAP_FLOOR_M,
        TORSO_OVAL_RZ_CHEST_FRAC,
        TORSO_OVAL_RZ_HIP_FRAC,
        TORSO_OVAL_RZ_WAIST_FRAC,
    )

    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    names = (
        "RECIPE_torso_oval_chest",
        "RECIPE_torso_oval_waist",
        "RECIPE_torso_oval_hip",
    )
    # span from report geometry
    shoulder_z = 1.38
    hip_z = 0.95
    chest_z = 0.72 * 1.72
    z_top = max(shoulder_z, chest_z)
    span = z_top - hip_z
    planned = {
        "RECIPE_torso_oval_chest": max(TORSO_OVAL_RZ_FLOOR_M, span * TORSO_OVAL_RZ_CHEST_FRAC),
        "RECIPE_torso_oval_waist": max(TORSO_OVAL_RZ_FLOOR_M, span * TORSO_OVAL_RZ_WAIST_FRAC),
        "RECIPE_torso_oval_hip": max(TORSO_OVAL_RZ_FLOOR_M, span * TORSO_OVAL_RZ_HIP_FRAC),
    }
    for name in names:
        p = by[name]
        assert p.rz_m is not None
        assert float(p.rz_m) >= planned[name] - 1e-9
    # Pairwise vertical overlap floor (0073 B2)
    for i in range(len(names) - 1):
        a = by[names[i]]
        b = by[names[i + 1]]
        assert a.center is not None and b.center is not None
        assert a.rz_m is not None and b.rz_m is not None
        ov = float(a.rz_m) + float(b.rz_m) - abs(float(a.center[2]) - float(b.center[2]))
        assert ov >= TORSO_OVAL_OVERLAP_FLOOR_M - 1e-9
    # Legacy SPAN_FRAC remains fence symbol only (not equal-triad emit)
    assert TORSO_OVAL_RZ_SPAN_FRAC == 0.22


def test_t7_dual_breasts_proud_of_chest_front() -> None:
    """T7: dual breasts - breast cy < chest front y (proud of plate)."""
    report = _full_torso_report(with_soft_cs=True)
    # Product-class breast prior: hang in front of flattened chest plate (B9).
    tpl = _template(taper=0.22)
    tpl = tpl.model_copy(
        update={
            "constants": tpl.constants.model_copy(
                update={"breast_y_m": -0.10, "breast_tilt_x_deg": 20.0}
            )
        }
    )
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals", template_applied=tpl)
    by = {p.name: p for p in pkg.parts}
    chest = by["RECIPE_torso_oval_chest"]
    assert chest.center is not None and chest.ry_m is not None
    chest_front_y = float(chest.center[1]) - float(chest.ry_m)
    breasts = [p for p in pkg.parts if p.role == "breast_soft" and p.center]
    assert len(breasts) >= 2
    for b in breasts:
        assert b.center is not None
        assert b.center[1] < chest_front_y + 1e-9


def test_t8_messages_depth_taper_and_front_pinch() -> None:
    """T8: messages - depth taper + front pinch with front_y AND rear_y."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    depth = [m for m in pkg.messages if m.startswith("torso depth taper:")]
    pinch = [m for m in pkg.messages if m.startswith("torso front pinch:")]
    assert len(depth) == 1
    assert "anti-snowman" in depth[0]
    assert len(pinch) == 1
    assert "waist_rx/chest_rx=" in pinch[0]
    assert "chest_front_y=" in pinch[0]
    assert "chest_rear_y=" in pinch[0]
    by = {p.name: p for p in pkg.parts}
    chest = by["RECIPE_torso_oval_chest"]
    assert chest.center is not None and chest.ry_m is not None
    front_y = float(chest.center[1]) - float(chest.ry_m)
    rear_y = float(chest.center[1]) + float(chest.ry_m)
    assert f"chest_front_y={front_y:.4f}" in pinch[0]
    assert f"chest_rear_y={rear_y:.4f}" in pinch[0]


def test_t9_trap_path_unchanged() -> None:
    """T9: trap path unchanged (default torso emits trap, not ovals)."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False)  # default trap
    names = {p.name for p in pkg.parts}
    assert "RECIPE_torso_trap" in names
    assert "RECIPE_torso_oval_chest" not in names
    assert not any(m.startswith("torso front pinch:") for m in pkg.messages)
    assert not any(m.startswith("torso depth taper:") for m in pkg.messages)


def test_t10_connection_gap_finite_and_constraints_smoke() -> None:
    """T10: connection gap finite; constraints smoke green (axial)."""
    # chest_front well forward of mid so head tip / neck tilt stay axial-pass (0032/0050).
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
    # Shoulder gaps: finite numeric inventory (not missing sentinel 1e9).
    assert "shoulder_l" in gaps and "shoulder_r" in gaps
    assert gaps["shoulder_l"] < 1e8, f"shoulder_l missing/sentinel: {gaps['shoulder_l']}"
    assert gaps["shoulder_r"] < 1e8, f"shoulder_r missing/sentinel: {gaps['shoulder_r']}"
    # Snapshot magnitudes for B10 inventory (chest rear-bias may shift parent Y-span).
    assert gaps["shoulder_l"] == pytest.approx(gaps["shoulder_r"], abs=1e-6)
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    axial = by_id.get("C_axial_depth_plane")
    assert axial is not None
    assert axial.status == "pass", axial.message


def test_t11_female_template_torso_waist_taper() -> None:
    """T11: female template torso_waist_taper == 0.22."""
    doc = load_body_template("female_adult_athletic")
    assert doc.torso_waist_taper == pytest.approx(0.22)
    # male fence
    male = load_body_template("male_adult_athletic")
    assert male.torso_waist_taper == pytest.approx(0.05)


def test_t12_anti_equal_triad_rx_ry() -> None:
    """T12: anti-regression - not equal triad rx / ry."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(
        report,
        limbs=False,
        torso="ovals",
        template_applied=_template(taper=0.22),
    )
    by = {p.name: p for p in pkg.parts}
    rx = [
        float(by[n].rx_m or 0.0)
        for n in (
            "RECIPE_torso_oval_chest",
            "RECIPE_torso_oval_waist",
            "RECIPE_torso_oval_hip",
        )
    ]
    ry = [
        float(by[n].ry_m or 0.0)
        for n in (
            "RECIPE_torso_oval_chest",
            "RECIPE_torso_oval_waist",
            "RECIPE_torso_oval_hip",
        )
    ]
    eps = 1e-9
    assert not (abs(rx[0] - rx[1]) < eps and abs(rx[1] - rx[2]) < eps)
    assert not (abs(ry[0] - ry[1]) < eps and abs(ry[1] - ry[2]) < eps)


def test_t13_two_pass_cap_uses_post_taper_chest_rx() -> None:
    """T13: B1 denominator is post-taper chest rx (not pre-taper baseline)."""
    # Near-equal shoulder/hip so waist raw stays above 0.80*post-chest at high taper
    # (default 0.20/0.14 at taper=0.22 already pinches below the hard max).
    ws, wh = 0.25, 0.24
    taper = 0.22
    assert taper >= TORSO_WAIST_PINCH_TAPER_GATE
    rx_c_post = _waist_width_at(0.15, ws, wh, taper)
    rx_c_pre = _waist_width_at(0.15, ws, wh, 0.0)
    rx_w_raw = _waist_width_at(0.50, ws, wh, taper)
    expected = min(rx_w_raw, TORSO_WAIST_RX_MAX_FRAC_CHEST * rx_c_post)
    wrong = min(rx_w_raw, TORSO_WAIST_RX_MAX_FRAC_CHEST * rx_c_pre)
    # Cap must bind, and two-pass must disagree with pre-taper denominator.
    assert rx_w_raw > TORSO_WAIST_RX_MAX_FRAC_CHEST * rx_c_post + 1e-9
    assert expected != pytest.approx(wrong, abs=1e-9)
    assert rx_c_post < rx_c_pre - 1e-9

    report = _full_torso_report(shoulder_x=ws, hip_x=wh)
    pkg = build_blockout_recipe(
        report,
        limbs=False,
        torso="ovals",
        template_applied=_template(taper=taper),
    )
    by = {p.name: p for p in pkg.parts}
    rx_c = float(by["RECIPE_torso_oval_chest"].rx_m or 0.0)
    rx_w = float(by["RECIPE_torso_oval_waist"].rx_m or 0.0)
    assert rx_c == pytest.approx(rx_c_post, abs=1e-9)
    assert rx_w == pytest.approx(expected, abs=1e-9)
    assert rx_w != pytest.approx(wrong, abs=1e-9)
    assert rx_w == pytest.approx(TORSO_WAIST_RX_MAX_FRAC_CHEST * rx_c, abs=1e-9)
