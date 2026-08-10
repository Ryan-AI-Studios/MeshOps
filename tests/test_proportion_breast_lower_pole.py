"""Track 0067 — breast_soft athletic lower-pole teardrop + sternum gap (authoring RECIPE)."""

from __future__ import annotations

import math

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    BREAST_ATHLETIC_RX_MAX_FRAC_H,
    BREAST_ATTACH_Y_SCALE,
    BREAST_HANG_Z_MIN_DROP_FRAC_RZ,
    BREAST_STERNUM_CLEARANCE_M,
    BREAST_TEAR_RY_FRAC_RX,
    BREAST_TEAR_RZ_FRAC_RX,
    BREAST_X_SHOULDER_FLOOR_FRAC,
    BREAST_X_SHOULDER_MAX_FRAC,
    RECIPE_SCHEMA_VERSION,
    BlockoutRecipePackage,
    RecipePart,
    _apply_breast_lower_pole_athletic,
    _breast_sternum_soft_half,
    _ResolvedMetrics,
    build_blockout_recipe,
)
from meshops.proportion.models import (
    CrossSection,
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
    SoftSpacing,
)
from meshops.proportion.skeleton import BlockoutSkeleton, SkeletonBone, SkeletonJoint


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


def _base_lms(
    *,
    height_m: float = 1.72,
    shoulder_hw: float = 0.20,
    extra: dict[str, LandmarkXYZ] | None = None,
) -> dict[str, LandmarkXYZ]:
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-shoulder_hw, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=shoulder_hw, y_m=0.0, z_m=1.38),
        "hip_l": _lm("hip_l", x_m=-0.14, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.14, y_m=0.0, z_m=0.95),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
    }
    if extra:
        lms.update(extra)
    return lms


def _report_soft_cs(
    *,
    height_m: float = 1.72,
    shoulder_hw: float = 0.20,
    soft_spacing: SoftSpacing | None = None,
    extra_lms: dict[str, LandmarkXYZ] | None = None,
    bust_hw: float = 0.16,
) -> ProportionReport:
    """Base CS path dual breast_soft (no profile)."""
    lms = _base_lms(height_m=height_m, shoulder_hw=shoulder_hw, extra=extra_lms)
    diams = [
        _diam("bust", half_width_m=bust_hw),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
    ]
    bands = [
        DepthBand(
            band_id="chest",
            depth_px=50.0,
            depth_frac=0.12,
            depth_m=0.24,
            y_front=0.1,
            y_back=-0.1,
            y_mid=0.0,
            z_frac=0.72,
        ),
        DepthBand(
            band_id="hip",
            depth_px=55.0,
            depth_frac=0.13,
            depth_m=0.26,
            y_front=0.1,
            y_back=-0.1,
            y_mid=0.0,
            z_frac=0.55,
        ),
    ]
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=bands,
        soft_spacing=soft_spacing,
        cross_sections=[
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
        ],
        quality=QualityFlags(),
    )


def _rich_report(
    *,
    height_m: float = 1.72,
    shoulder_hw: float = 0.20,
    soft_spacing: SoftSpacing | None = None,
    bust_hw: float = 0.16,
    extra_lms: dict[str, LandmarkXYZ] | None = None,
) -> ProportionReport:
    """Profile path report (female dual breasts)."""
    lms = _base_lms(height_m=height_m, shoulder_hw=shoulder_hw, extra=extra_lms)
    lms["elbow_l"] = _lm("elbow_l", x_m=-0.28, y_m=-0.05, z_m=1.10)
    lms["elbow_r"] = _lm("elbow_r", x_m=0.28, y_m=-0.05, z_m=1.10)
    diams = [
        _diam("bust", half_width_m=bust_hw),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.05),
        _diam("upper_arm_r", half_width_m=0.05),
    ]
    bands = [
        DepthBand(
            band_id="chest",
            depth_px=50.0,
            depth_frac=0.12,
            depth_m=0.24,
            y_front=0.1,
            y_back=-0.1,
            y_mid=0.0,
            z_frac=0.72,
        ),
        DepthBand(
            band_id="breast",
            depth_px=40.0,
            depth_frac=0.10,
            depth_m=0.18,
            y_front=0.08,
            y_back=-0.05,
            y_mid=0.0,
            z_frac=0.70,
        ),
        DepthBand(
            band_id="hip",
            depth_px=55.0,
            depth_frac=0.13,
            depth_m=0.26,
            y_front=0.1,
            y_back=-0.1,
            y_mid=0.0,
            z_frac=0.55,
        ),
    ]
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=bands,
        soft_spacing=soft_spacing,
        quality=QualityFlags(),
    )


def _j(
    id_: str,
    *,
    x: float,
    y: float,
    z: float,
    side: str = "none",
    parent: str | None = None,
) -> SkeletonJoint:
    return SkeletonJoint(
        id=id_,
        parent=parent,
        side=side,  # type: ignore[arg-type]
        x_m=x,
        y_m=y,
        z_m=z,
        source="estimated",
    )


def _skeleton_with_arms(*, spine_high_z: float = 1.27) -> BlockoutSkeleton:
    return BlockoutSkeleton(
        schema_version="1.0.0",
        honesty="proportion_blockout_skeleton_not_mesh_or_print_success",
        joints=[
            _j("root", x=0.0, y=0.0, z=0.0),
            _j("pelvis", x=0.0, y=0.0, z=0.95, parent="root"),
            _j("spine_high", x=0.0, y=0.0, z=spine_high_z, parent="pelvis"),
            _j("neck_base", x=0.0, y=0.0, z=1.42, parent="spine_high"),
            _j("shoulder_l", x=-0.20, y=0.0, z=1.38, side="l", parent="spine_high"),
            _j("shoulder_r", x=0.20, y=0.0, z=1.38, side="r", parent="spine_high"),
            _j("elbow_l", x=-0.28, y=-0.05, z=1.10, side="l", parent="shoulder_l"),
            _j("elbow_r", x=0.28, y=-0.05, z=1.10, side="r", parent="shoulder_r"),
        ],
        bones=[
            SkeletonBone(id="spine", joint_a="pelvis", joint_b="spine_high", length_m=0.3),
            SkeletonBone(id="upper_arm_l", joint_a="shoulder_l", joint_b="elbow_l", length_m=0.3),
            SkeletonBone(id="upper_arm_r", joint_a="shoulder_r", joint_b="elbow_r", length_m=0.3),
        ],
    )


def _dual_breast_parts(
    *,
    center_z: float = 1.27,
    rz: float = 0.08,
    rx: float = 0.09,
    ry: float = 0.07,
    center_y: float = -0.06,
    offset_x: float = 0.06,
) -> list[RecipePart]:
    return [
        RecipePart(
            name="RECIPE_breast_soft_l",
            role="breast_soft",
            kind="ellipsoid",
            center=[-offset_x, center_y, center_z],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
            placement="full3d",
            label="RECIPE_breast_soft_l",
        ),
        RecipePart(
            name="RECIPE_breast_soft_r",
            role="breast_soft",
            kind="ellipsoid",
            center=[offset_x, center_y, center_z],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
            placement="full3d",
            label="RECIPE_breast_soft_r",
        ),
    ]


def _empty_metrics(
    *,
    height_m: float = 1.72,
    shoulder_hw: float | None = 0.20,
    chest_z: float | None = 1.31,
) -> _ResolvedMetrics:
    m = _ResolvedMetrics()
    m.height_m = height_m
    m.shoulder_hw = shoulder_hw
    m.chest_z = chest_z
    return m


def _msg_value(messages: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for m in messages:
        if m.startswith(prefix):
            return m[len(prefix) :]
        if m == key:
            return ""
    return None


def _contact_gap(breasts: list[RecipePart]) -> float:
    xs = sorted(float(p.center[0]) for p in breasts if p.center is not None)
    assert len(xs) == 2
    rxs = [float(p.rx_m) for p in breasts if p.rx_m is not None]
    assert len(rxs) == 2
    mean_rx = sum(rxs) / 2.0
    return (xs[1] - xs[0]) - 2.0 * mean_rx


def _f_athletic_pkg(
    *,
    soft_spacing: SoftSpacing | None = None,
    breast_tilt_deg: float | None = None,
    bust_hw: float = 0.16,
) -> BlockoutRecipePackage:
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    return build_blockout_recipe(
        _rich_report(soft_spacing=soft_spacing, bust_hw=bust_hw),
        limbs=False,
        profile=profile,
        skeleton=_skeleton_with_arms(),
        breast_tilt_deg=breast_tilt_deg,
    )


# ---------------------------------------------------------------------------
# T0 constants
# ---------------------------------------------------------------------------


def test_t0_constants_freeze() -> None:
    assert BREAST_ATHLETIC_RX_MAX_FRAC_H == 0.042
    assert BREAST_TEAR_RY_FRAC_RX == 0.78
    assert BREAST_TEAR_RZ_FRAC_RX == 1.05
    assert BREAST_STERNUM_CLEARANCE_M == 0.010
    assert BREAST_X_SHOULDER_FLOOR_FRAC == 0.25
    assert BREAST_X_SHOULDER_MAX_FRAC == 0.45
    assert BREAST_ATTACH_Y_SCALE == 1.0


# ---------------------------------------------------------------------------
# T1-T6 F athletic product-like
# ---------------------------------------------------------------------------


def test_t1_teardrop_ratios() -> None:
    """T1: F athletic recipe ry/rx≈0.78, rz/rx≈1.05."""
    pkg = _f_athletic_pkg()
    breasts = [p for p in pkg.parts if p.role == "breast_soft"]
    assert len(breasts) == 2
    for b in breasts:
        assert b.rx_m is not None and b.ry_m is not None and b.rz_m is not None
        rx = float(b.rx_m)
        assert float(b.ry_m) / rx == pytest.approx(BREAST_TEAR_RY_FRAC_RX, rel=1e-6)
        assert float(b.rz_m) / rx == pytest.approx(BREAST_TEAR_RZ_FRAC_RX, rel=1e-6)


def test_t2_athletic_rx_cap() -> None:
    """T2: rx <= 0.042*H + eps after post-pass."""
    h = 1.72
    pkg = _f_athletic_pkg(bust_hw=0.18)  # profile can emit large rx
    breasts = [p for p in pkg.parts if p.role == "breast_soft"]
    assert len(breasts) == 2
    cap = BREAST_ATHLETIC_RX_MAX_FRAC_H * h
    for b in breasts:
        assert b.rx_m is not None
        assert float(b.rx_m) <= cap + 1e-9


def test_t3_soft_half_zero_clearance_floor() -> None:
    """T3: soft_half=0 -> contact_gap >= 0.020 - eps (clearance floor)."""
    # Wide shoulders so floor/cap do not thrash medial_half gap.
    parts = _dual_breast_parts(rx=0.05, ry=0.04, rz=0.05, offset_x=0.04)
    report = _report_soft_cs(soft_spacing=None, shoulder_hw=0.30)
    msgs: list[str] = []
    _apply_breast_lower_pole_athletic(parts, report, _empty_metrics(shoulder_hw=0.30), None, msgs)
    assert any(m == "breast_lower_pole_athletic_applied: true" for m in msgs)
    gap = _contact_gap(parts)
    assert gap + 1e-9 >= 2.0 * BREAST_STERNUM_CLEARANCE_M - 1e-9
    soft_s = _msg_value(msgs, "breast_sternum_soft_half_m")
    assert soft_s is not None
    assert float(soft_s) == pytest.approx(0.0, abs=1e-12)


def test_t3b_product_like_soft_half_gap() -> None:
    """T3b: product-like soft_half≈0.0145 → contact_gap ≈ 0.029 when floor/cap idle."""
    soft_gap = 0.029  # template intermammary class
    parts = _dual_breast_parts(rx=0.072, ry=0.06, rz=0.07, offset_x=0.05)
    report = _report_soft_cs(
        soft_spacing=SoftSpacing(intermammary_gap_m=soft_gap),
        shoulder_hw=0.30,
    )
    msgs: list[str] = []
    _apply_breast_lower_pole_athletic(parts, report, _empty_metrics(shoulder_hw=0.30), None, msgs)
    gap = _contact_gap(parts)
    assert gap == pytest.approx(soft_gap, abs=1e-6)
    medial_s = _msg_value(msgs, "breast_sternum_medial_half_m")
    assert medial_s is not None
    assert float(medial_s) == pytest.approx(soft_gap / 2.0, abs=1e-9)


def test_t4_hang_still_applies() -> None:
    """T4: hang applied true; drop >= 0.40*rz (D2 min) on unclamped path.

    F athletic + torso ovals can soft-clamp hang (0049); base CS has no waist oval
    so pure B1 drop meets D2 on post-0067 rz.
    """
    pkg_f = _f_athletic_pkg()
    assert any(m == "breast_hang_z_applied: true" for m in pkg_f.messages)
    assert any(p.role == "breast_soft" for p in pkg_f.parts)

    pkg = build_blockout_recipe(_report_soft_cs(), limbs=False)
    breasts = [p for p in pkg.parts if p.role == "breast_soft" and p.center]
    assert len(breasts) == 2
    assert any(m == "breast_hang_z_applied: true" for m in pkg.messages)
    anchor_s = _msg_value(pkg.messages, "breast_hang_z_anchor_m")
    assert anchor_s is not None
    anchor = float(anchor_s)
    for b in breasts:
        assert b.center is not None and b.rz_m is not None
        drop = anchor - b.center[2]
        assert drop + 1e-9 >= BREAST_HANG_Z_MIN_DROP_FRAC_RZ * float(b.rz_m)


def test_t5_tilt_with_lower_pole() -> None:
    """T5: tilt 20 → rotation [20,0,0]."""
    pkg = _f_athletic_pkg(breast_tilt_deg=20.0)
    breasts = [p for p in pkg.parts if p.role == "breast_soft"]
    assert len(breasts) >= 2
    assert any(m == "breast_tilt_applied: true" for m in pkg.messages)
    for b in breasts:
        assert b.rotation_euler_deg == pytest.approx([20.0, 0.0, 0.0])


def test_t6_dual_lr_axes_and_z_equal() -> None:
    """T6: L/R axes equal; Z equal after hang."""
    pkg = _f_athletic_pkg()
    breasts = [p for p in pkg.parts if p.role == "breast_soft" and p.center]
    assert len(breasts) == 2
    a, b = breasts[0], breasts[1]
    assert a.rx_m is not None and b.rx_m is not None
    assert a.ry_m is not None and b.ry_m is not None
    assert a.rz_m is not None and b.rz_m is not None
    assert float(a.rx_m) == pytest.approx(float(b.rx_m), abs=1e-12)
    assert float(a.ry_m) == pytest.approx(float(b.ry_m), abs=1e-12)
    assert float(a.rz_m) == pytest.approx(float(b.rz_m), abs=1e-12)
    zs = [p.center[2] for p in breasts if p.center is not None]
    assert zs[0] == pytest.approx(zs[1], abs=1e-12)


# ---------------------------------------------------------------------------
# T7-T11 male / pec / base / messages
# ---------------------------------------------------------------------------


def test_t7_male_pec_skip() -> None:
    """T7: male pec — lower_pole applied false; no breast_soft."""
    profile = load_anatomy_profile("torso_limb_m_athletic_v1")
    pkg = build_blockout_recipe(_rich_report(), limbs=False, profile=profile, glute="two_spheres")
    assert not any(p.role == "breast_soft" for p in pkg.parts)
    assert any(m == "breast_lower_pole_athletic_applied: false" for m in pkg.messages)
    assert any(p.role == "pec_soft" for p in pkg.parts)


def test_t8_larger_measured_gap() -> None:
    """T8: larger measured intermammary → larger center/contact gap."""
    wide_sh = 0.30
    pkg_s = build_blockout_recipe(
        _report_soft_cs(
            soft_spacing=SoftSpacing(intermammary_gap_m=0.04),
            shoulder_hw=wide_sh,
        ),
        limbs=False,
    )
    pkg_l = build_blockout_recipe(
        _report_soft_cs(
            soft_spacing=SoftSpacing(intermammary_gap_m=0.12),
            shoulder_hw=wide_sh,
        ),
        limbs=False,
    )
    b_s = [p for p in pkg_s.parts if p.role == "breast_soft"]
    b_l = [p for p in pkg_l.parts if p.role == "breast_soft"]
    assert _contact_gap(b_s) < _contact_gap(b_l)


def test_t9_pec_soft_untouched() -> None:
    """T9: pec_soft axes/center unchanged when dual breasts also present (direct call)."""
    parts = _dual_breast_parts(rx=0.09)
    pec = RecipePart(
        name="RECIPE_pec_soft_l",
        role="pec_soft",
        kind="ellipsoid",
        center=[-0.05, -0.04, 1.30],
        rx_m=0.05,
        ry_m=0.04,
        rz_m=0.05,
        placement="full3d",
        label="RECIPE_pec_soft_l",
    )
    parts.append(pec)
    assert pec.center is not None
    pre = (list(pec.center), pec.rx_m, pec.ry_m, pec.rz_m)
    msgs: list[str] = []
    _apply_breast_lower_pole_athletic(parts, _report_soft_cs(), _empty_metrics(), None, msgs)
    pec_after = next(p for p in parts if p.role == "pec_soft")
    assert pec_after.center == pre[0]
    assert pec_after.rx_m == pre[1]
    assert pec_after.ry_m == pre[2]
    assert pec_after.rz_m == pre[3]


def test_t10_base_cs_tear_and_sternum() -> None:
    """T10: base CS duals (no profile) get tear + sternum."""
    pkg = build_blockout_recipe(_report_soft_cs(), limbs=False)
    breasts = [p for p in pkg.parts if p.role == "breast_soft"]
    assert len(breasts) == 2
    assert any(m == "breast_lower_pole_athletic_applied: true" for m in pkg.messages)
    for b in breasts:
        assert b.rx_m is not None and b.ry_m is not None and b.rz_m is not None
        rx = float(b.rx_m)
        assert float(b.ry_m) / rx == pytest.approx(BREAST_TEAR_RY_FRAC_RX, rel=1e-6)
        assert float(b.rz_m) / rx == pytest.approx(BREAST_TEAR_RZ_FRAC_RX, rel=1e-6)
    gap = _contact_gap(breasts)
    assert gap + 1e-9 >= 2.0 * BREAST_STERNUM_CLEARANCE_M - 1e-6


def test_t11_messages_when_applied() -> None:
    """T11: athletic / tear / sternum messages present when applied."""
    pkg = build_blockout_recipe(_report_soft_cs(), limbs=False)
    assert any(m == "breast_lower_pole_athletic_applied: true" for m in pkg.messages)
    assert _msg_value(pkg.messages, "breast_athletic_scale_s") is not None
    assert _msg_value(pkg.messages, "breast_tear_ry_frac_rx") == str(BREAST_TEAR_RY_FRAC_RX)
    assert _msg_value(pkg.messages, "breast_tear_rz_frac_rx") == str(BREAST_TEAR_RZ_FRAC_RX)
    assert _msg_value(pkg.messages, "breast_sternum_soft_half_m") is not None
    assert _msg_value(pkg.messages, "breast_sternum_medial_half_m") is not None
    assert _msg_value(pkg.messages, "breast_sternum_gap_m") is not None


# ---------------------------------------------------------------------------
# T12-T15 Y, shoulder floor, schema, hang isolation
# ---------------------------------------------------------------------------


def test_t12_center_y_unchanged_at_attach_1() -> None:
    """T12: BREAST_ATTACH_Y_SCALE=1.0 → center y unchanged by lower-pole post-pass."""
    y0 = -0.088
    parts = _dual_breast_parts(center_y=y0, rx=0.09)
    msgs: list[str] = []
    _apply_breast_lower_pole_athletic(
        parts, _report_soft_cs(), _empty_metrics(shoulder_hw=0.30), None, msgs
    )
    assert BREAST_ATTACH_Y_SCALE == 1.0
    for p in parts:
        assert p.center is not None
        assert p.center[1] == pytest.approx(y0, abs=1e-12)
        assert p.center[1] < 0.0  # still front -Y


def test_t13_shoulder_floor_binds() -> None:
    """T13: synthetic small rx+medial under floor → offset = shoulder*0.25."""
    # soft_half=0, medial=0.010; rx=0.02 → rx+medial=0.030; floor at sh*0.25=0.05
    parts = _dual_breast_parts(rx=0.02, ry=0.02, rz=0.02, offset_x=0.03)
    report = _report_soft_cs(soft_spacing=None, shoulder_hw=0.20)
    msgs: list[str] = []
    m = _empty_metrics(height_m=1.72, shoulder_hw=0.20)
    _apply_breast_lower_pole_athletic(parts, report, m, None, msgs)
    # B1: mean_rx=0.02 << cap 0.072 → no scale; tear keeps rx=0.02
    expected_offset = 0.20 * BREAST_X_SHOULDER_FLOOR_FRAC
    for p in parts:
        assert p.center is not None
        assert abs(p.center[0]) == pytest.approx(expected_offset, abs=1e-9)


def test_t14_schema_and_mcp_catalog() -> None:
    """T14: schema 1.4.0; MCP tool count 46."""
    from meshops.mcp import TOOL_NAMES

    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    pkg = build_blockout_recipe(_report_soft_cs(), limbs=False, breast_tilt_deg=20.0)
    assert pkg.schema_version == "1.4.0"
    assert len(TOOL_NAMES) == 46


def test_t15_hang_suite_smoke_still_works() -> None:
    """T15: hang still drops dual base-path breasts (hang suite isolation)."""
    pkg = build_blockout_recipe(_report_soft_cs(), limbs=False)
    breasts = [p for p in pkg.parts if p.role == "breast_soft" and p.center]
    assert len(breasts) == 2
    assert any(m == "breast_hang_z_applied: true" for m in pkg.messages)
    for b in breasts:
        assert b.center is not None and b.rz_m is not None
        # post-tear rz; hang uses that rz for drop floor
        assert math.isfinite(b.center[2])


def test_soft_half_ladder_measured_over_template() -> None:
    """Helper: measured soft_spacing beats template gap."""

    class _C:
        intermammary_gap_m = 0.08
        intermammary_gap_frac = 0.2

    class _T:
        constants = _C()

    report = _report_soft_cs(soft_spacing=SoftSpacing(intermammary_gap_m=0.04))
    half = _breast_sternum_soft_half(report, _empty_metrics(), _T())  # type: ignore[arg-type]
    assert half == pytest.approx(0.02)


def test_soft_half_else_zero_not_shoulder() -> None:
    """Helper else branch is 0.0 (not shoulder*0.18)."""
    report = _report_soft_cs(soft_spacing=None)
    half = _breast_sternum_soft_half(report, _empty_metrics(shoulder_hw=0.20), None)
    assert half == 0.0
