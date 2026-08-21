"""Track 0076 - heel / ankle visual proportion freezes (T0-T12).

Anti-ball ank (ry/rz shrink) + heel rear seat bias while keeping
0056 contact gap ≥ +0.005. Product half_width ~ 0.0263 for frac-win.
"""

from __future__ import annotations

import pytest

from meshops.proportion.blockout_recipe import RecipePart, build_blockout_recipe
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.extremity_recipe import (
    ANK_RY_FLOOR_M,
    ANK_RY_FRAC_HALF_W,
    ANK_RZ_FLOOR_M,
    ANK_RZ_FRAC_HALF_W,
    ANK_RZ_MAX_FRAC_ANK_Z,
    ANK_RZ_MIN_VS_CALF_B,
    HEEL_CONTACT_OVERLAP_TARGET_M,
    HEEL_REAR_OVERHANG_M,
    HEEL_REAR_Y_BIAS_FRAC_DEPTH,
    HEEL_RY_MIN_FRAC_DEPTH,
    SOLE_RZ_FLOOR_M,
    SOLE_RZ_FRAC_OF_THICKNESS,
    SOLE_THICKNESS_FRAC_H,
    TOE_R_FLOOR_M,
    TOE_R_FRAC_HALF_W,
    TOE_TIP_MAX_PAST_BALL_FRAC,
    TOE_TIP_MAX_PAST_BALL_M,
    TOE_TIP_PAD_SCALE,
)
from meshops.proportion.models import (
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

# Product-class ankle half-width — same as 0056 suite.
PRODUCT_HW_M: float = 0.0263
# 0098-class full-figure hw (never-shrink; stays above 0.16*stature length).
_PRODUCT_HW_0098_M: float = 0.04237
PRODUCT_H_M: float = 1.72
PRODUCT_ANK_Z_M: float = 0.1314
EPS: float = 1e-6


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
) -> DepthBand:
    return DepthBand(
        band_id=band_id,
        depth_px=50.0,
        depth_frac=0.12,
        depth_m=depth_m,
        y_front=0.1,
        y_back=-0.1,
        y_mid=0.0,
        z_frac=z_frac,
    )


def _product_feet_report(
    *,
    height_m: float = PRODUCT_H_M,
    half_width_m: float = PRODUCT_HW_M,
    ank_z: float = PRODUCT_ANK_Z_M,
    heel_y: float = 0.06,
    toe_y: float = -0.12,
) -> ProportionReport:
    """Product-like L/R feet: product hw, heel +Y, toe -Y, realistic ank_z."""
    lms: dict[str, LandmarkXYZ] = {
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
    }
    for side, sx in (("l", -0.10), ("r", 0.10)):
        lms[f"ankle_{side}"] = _lm(f"ankle_{side}", x_m=sx, y_m=0.02, z_m=ank_z)
        lms[f"heel_{side}"] = _lm(f"heel_{side}", x_m=sx, y_m=heel_y, z_m=0.02)
        lms[f"toe_{side}"] = _lm(f"toe_{side}", x_m=sx, y_m=toe_y, z_m=0.02)
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=[
            _diam("bust", half_width_m=0.16),
            _diam("waist", half_width_m=0.13),
            _diam("neck", half_width_m=0.05),
            _diam("ank_foot_l", half_width_m=half_width_m),
            _diam("ank_foot_r", half_width_m=half_width_m),
        ],
        depth_bands=[
            _band("chest", depth_m=0.24, z_frac=0.72),
            _band("hip", depth_m=0.26, z_frac=0.55),
        ],
        quality=QualityFlags(),
    )


def _ank(pkg_parts: list, side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_ank_foot_{side}")


def _heel(pkg_parts: list, side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_heel_{side}")


def _plate(pkg_parts: list, side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_foot_plate_{side}")


def _contact_overlap(pkg_parts: list, side: str = "l") -> float:
    ank = _ank(pkg_parts, side)
    heel = _heel(pkg_parts, side)
    assert ank.center is not None and ank.rz_m is not None
    assert heel.center is not None and heel.rz_m is not None
    ank_bottom = float(ank.center[2]) - float(ank.rz_m)
    heel_top = float(heel.center[2]) + float(heel.rz_m)
    return heel_top - ank_bottom


def _half_depth(pkg_parts: list, side: str = "l") -> float:
    plate = _plate(pkg_parts, side)
    assert plate.half_depth_m is not None
    return float(plate.half_depth_m)


# ---------------------------------------------------------------------------
# T0 public freezes
# ---------------------------------------------------------------------------


def test_t0_public_freezes_0076() -> None:
    """T0: B1-B3 exacts + contact/heel_ry/overhang fences (0097 invert)."""
    assert pytest.approx(0.78) == ANK_RY_FRAC_HALF_W
    assert pytest.approx(0.030) == ANK_RY_FLOOR_M
    assert pytest.approx(1.80) == ANK_RZ_FRAC_HALF_W
    assert pytest.approx(0.044) == ANK_RZ_FLOOR_M
    assert pytest.approx(0.14) == HEEL_REAR_Y_BIAS_FRAC_DEPTH
    # Bands (open retune) — 0108 AP flatten after 0097 leftover sphere
    assert 0.72 <= ANK_RY_FRAC_HALF_W <= 0.86
    assert 0.028 <= ANK_RY_FLOOR_M <= 0.033
    assert 1.70 <= ANK_RZ_FRAC_HALF_W <= 1.90
    assert 0.042 <= ANK_RZ_FLOOR_M <= 0.046
    assert 0.12 <= HEEL_REAR_Y_BIAS_FRAC_DEPTH <= 0.16
    # Unchanged fences
    assert pytest.approx(0.005) == HEEL_CONTACT_OVERLAP_TARGET_M
    assert pytest.approx(0.30) == HEEL_RY_MIN_FRAC_DEPTH
    assert pytest.approx(0.012) == HEEL_REAR_OVERHANG_M
    assert pytest.approx(1.35) == ANK_RZ_MIN_VS_CALF_B
    assert pytest.approx(0.60) == ANK_RZ_MAX_FRAC_ANK_Z


# ---------------------------------------------------------------------------
# T1-T6 product composition
# ---------------------------------------------------------------------------


def test_t1_product_contact_gap_ge_target() -> None:
    """T1: product-class contact gap ≥ +0.005 after full foot emit."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    gap = _contact_overlap(pkg.parts)
    assert gap >= HEEL_CONTACT_OVERLAP_TARGET_M - EPS, (
        f"overlap={gap} < target={HEEL_CONTACT_OVERLAP_TARGET_M}"
    )


def test_t2_product_ank_ry_rx_anti_ball_anti_pea() -> None:
    """T2: product ank_ry/rx AP flatten (0.72-0.86, 0108) + pea floor still holds."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    assert ank.rx_m is not None and ank.ry_m is not None
    ratio = float(ank.ry_m) / float(ank.rx_m)
    assert 0.72 <= ratio <= 0.86, f"ank_ry/rx={ratio}"


def test_t3_product_ank_ry_frac_wins_floor() -> None:
    """T3: 0.78*0098-hw > 0.030 and emitted ank_ry approx frac path (thin-hw floors)."""
    assert ANK_RY_FRAC_HALF_W * _PRODUCT_HW_0098_M > ANK_RY_FLOOR_M
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    assert ank.ry_m is not None
    assert ank.rx_m is not None
    # 0098 hw stays above width floor; ry uses emitted ank.rx * 0.78 (0108)
    frac_ry = ANK_RY_FRAC_HALF_W * float(ank.rx_m)
    assert float(ank.ry_m) == pytest.approx(frac_ry, abs=1e-5)
    assert float(ank.rx_m) == pytest.approx(0.04237, abs=1e-4)
    assert float(ank.ry_m) == pytest.approx(0.04237 * ANK_RY_FRAC_HALF_W, abs=1e-4)
    assert float(ank.ry_m) >= ANK_RY_FLOOR_M - 1e-9


def test_t4_product_ank_rz_rx_le_190() -> None:
    """T4: product ank_rz/rx ≤ 1.90."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    assert ank.rx_m is not None and ank.rz_m is not None
    ratio = float(ank.rz_m) / float(ank.rx_m)
    assert ratio <= 1.90 + EPS, f"ank_rz/rx={ratio}"


def test_t5_product_heel_dy_ge_008_half_depth() -> None:
    """T5: product heel_y - ank_y >= 0.08 * half_depth."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    heel = _heel(pkg.parts)
    assert ank.center is not None and heel.center is not None
    dy = float(heel.center[1]) - float(ank.center[1])
    hd = _half_depth(pkg.parts)
    assert dy >= 0.08 * hd - EPS, f"dy={dy} < 0.08*hd={0.08 * hd}"


def test_t6_heel_rear_tip_within_overhang() -> None:
    """T6: heel rear tip ≤ plate_rear + overhang."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    heel = _heel(pkg.parts)
    plate = _plate(pkg.parts)
    assert heel.center is not None and heel.ry_m is not None
    assert plate.center is not None and plate.half_depth_m is not None
    rear_tip = float(heel.center[1]) + float(heel.ry_m)
    plate_rear = float(plate.center[1]) + float(plate.half_depth_m)
    assert rear_tip <= plate_rear + HEEL_REAR_OVERHANG_M + EPS, (
        f"rear_tip={rear_tip} > plate_rear+overhang={plate_rear + HEEL_REAR_OVERHANG_M}"
    )


# ---------------------------------------------------------------------------
# T7 sole / toe fence smoke
# ---------------------------------------------------------------------------


def test_t7_sole_toe_freezes_unchanged_smoke() -> None:
    """T7: SOLE_*/TOE_* freezes still present; sole_rz floor still binds path."""
    assert pytest.approx(0.025) == SOLE_THICKNESS_FRAC_H
    assert pytest.approx(0.70) == SOLE_RZ_FRAC_OF_THICKNESS
    assert pytest.approx(0.016) == SOLE_RZ_FLOOR_M
    assert pytest.approx(0.36) == TOE_R_FRAC_HALF_W
    assert pytest.approx(0.009) == TOE_R_FLOOR_M
    assert pytest.approx(0.028) == TOE_TIP_MAX_PAST_BALL_M
    assert pytest.approx(0.12) == TOE_TIP_MAX_PAST_BALL_FRAC
    assert pytest.approx(0.78) == TOE_TIP_PAD_SCALE
    report = _product_feet_report(height_m=1.72)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    assert plate.rz_m is not None
    expect_rz = max(
        SOLE_THICKNESS_FRAC_H * 1.72 * SOLE_RZ_FRAC_OF_THICKNESS,
        SOLE_RZ_FLOOR_M,
    )
    assert float(plate.rz_m) == pytest.approx(expect_rz, abs=1e-4)


# ---------------------------------------------------------------------------
# T8 messages
# ---------------------------------------------------------------------------


def test_t8_messages_ank_contact_and_heel_ank_proportion() -> None:
    """T8: ank contact mass AND separate heel/ank proportion with dy=."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    joined = "\n".join(pkg.messages)
    assert "ank contact mass" in joined
    assert "heel/ank proportion" in joined
    assert "dy=" in joined
    assert any(
        "heel/ank proportion" in m and "dy=" in m and m.startswith("foot_") for m in pkg.messages
    )
    # Separate lines — proportion line is not the heel proportion ry= line
    prop_msgs = [m for m in pkg.messages if "heel/ank proportion" in m]
    assert prop_msgs
    assert all("heel proportion ry=" not in m for m in prop_msgs)


# ---------------------------------------------------------------------------
# T9 constraints
# ---------------------------------------------------------------------------


def test_t9_constraints_foot_stack_green() -> None:
    """T9: C_heel_reaches / C_foot_width / C_ankle_over_heel pass."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    for rule_id in (
        "C_heel_reaches_ank_foot",
        "C_foot_width",
        "C_ankle_over_heel",
    ):
        assert rule_id in by_id, f"missing rule {rule_id}"
        assert by_id[rule_id].status == "pass", (
            f"{rule_id} status={by_id[rule_id].status}: {by_id[rule_id].message}"
        )


# ---------------------------------------------------------------------------
# T10 ank_rz ceiling
# ---------------------------------------------------------------------------


def test_t10_ank_rz_ceiling_binds_synth() -> None:
    """T10: hw=0.035, ank_z=0.08 -> ceiling binds (0.60*ank_z)."""
    hw = 0.035
    ank_z = 0.08
    report = _product_feet_report(half_width_m=hw, ank_z=ank_z)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    ank = _ank(pkg.parts)
    assert ank.rz_m is not None
    ceil = ANK_RZ_MAX_FRAC_ANK_Z * ank_z
    frac_path = ANK_RZ_FRAC_HALF_W * hw  # 1.80*0.035 without ceiling
    assert frac_path > ceil  # ceiling must be the binding knob
    assert float(ank.rz_m) <= ceil + EPS
    assert float(ank.rz_m) == pytest.approx(ceil, abs=1e-5)


# ---------------------------------------------------------------------------
# T11 cap-lose contact
# ---------------------------------------------------------------------------


def test_t11_cap_lose_still_contact_ge_target() -> None:
    """T11: still_need/cap-lose path keeps contact ≥ +0.005 (or non-float).

    H=None so 0080 width floors skip and tiny-hw ank_rz stays floor-class.
    """
    report = _product_feet_report(half_width_m=0.02, ank_z=0.50)
    report = report.model_copy(update={"height_m": None})
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    gap = _contact_overlap(pkg.parts)
    joined = "\n".join(pkg.messages)
    assert "cap lose" in joined, (
        f"expected still_need cap-lose message; gap={gap}; msgs={pkg.messages}"
    )
    assert gap >= HEEL_CONTACT_OVERLAP_TARGET_M - EPS or gap >= -EPS, (
        f"gap={gap} must not float under heel on cap-lose"
    )


# ---------------------------------------------------------------------------
# T12 honesty package
# ---------------------------------------------------------------------------


def test_t12_schema_honesty_no_dual_radius() -> None:
    """T12: schema 1.4.0; no dual_radius field; heel/ank roles only for those parts."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    assert pkg.schema_version == "1.4.0"
    assert "dual_radius" not in RecipePart.model_fields
    for p in pkg.parts:
        dumped = p.model_dump()
        assert "dual_radius" not in dumped
        if p.name.startswith("RECIPE_heel_"):
            assert p.role == "heel"
        if p.name.startswith("RECIPE_ank_foot_"):
            assert p.role == "ankle_bridge"
