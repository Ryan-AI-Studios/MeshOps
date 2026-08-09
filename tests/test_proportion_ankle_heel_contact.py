"""Track 0056 - ankle / heel contact mass freezes (T0-T14).

Product half_width ~ 0.0263 must be used for frac-win asserts so dead-frac
regressions are not masked by synthetic ank_foot hw=0.035.
"""

from __future__ import annotations

import pytest

from meshops.proportion.blockout_recipe import RecipePart, build_blockout_recipe
from meshops.proportion.constraints import HEEL_REACH_GAP_TOL_M, validate_constraints
from meshops.proportion.extremity_recipe import (
    ANK_RY_FLOOR_M,
    ANK_RY_FRAC_HALF_W,
    ANK_RZ_FLOOR_M,
    ANK_RZ_FRAC_HALF_W,
    ANK_RZ_MAX_FRAC_ANK_Z,
    ANK_RZ_MIN_VS_CALF_B,
    FOOT_LEN_MIN_VS_ANK_HW,
    FOOT_LEN_MIN_VS_CALF_DIAM,
    FOOT_LEN_VISUAL_MIN_FRAC_H,
    HEEL_CONTACT_OVERLAP_TARGET_M,
    HEEL_REAR_Y_BIAS_FRAC_DEPTH,
    HEEL_RY_MIN_FRAC_DEPTH,
    HEEL_RZ_CAP_FRAC_ANK,
    HEEL_Z_FRAC_ANK,
    SOLE_RZ_FLOOR_M,
    SOLE_RZ_FRAC_OF_THICKNESS,
    SOLE_THICKNESS_FRAC_H,
    TOE_R_FRAC_HALF_W,
    build_foot_parts,
)
from meshops.proportion.models import (
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

# Product-class ankle half-width — must not use 0.035 synthetic default for T1-T4/T9.
PRODUCT_HW_M: float = 0.0263
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


# ---------------------------------------------------------------------------
# T0 exports + retune bands
# ---------------------------------------------------------------------------


def test_t0_exports_named_freezes_in_retune_bands() -> None:
    """T0: all ANK_* + HEEL_CONTACT_* exported and in plan §0 retune bands."""
    assert 1.40 <= ANK_RY_FRAC_HALF_W <= 1.55
    assert 0.034 <= ANK_RY_FLOOR_M <= 0.040
    assert 1.85 <= ANK_RZ_FRAC_HALF_W <= 2.15
    assert 0.045 <= ANK_RZ_FLOOR_M <= 0.055
    assert 1.20 <= ANK_RZ_MIN_VS_CALF_B <= 1.50
    assert 0.55 <= ANK_RZ_MAX_FRAC_ANK_Z <= 0.65
    assert 0.003 <= HEEL_CONTACT_OVERLAP_TARGET_M <= 0.010
    # Exact defaults from plan freezes
    assert pytest.approx(1.45) == ANK_RY_FRAC_HALF_W
    assert pytest.approx(0.036) == ANK_RY_FLOOR_M
    assert pytest.approx(2.00) == ANK_RZ_FRAC_HALF_W
    assert pytest.approx(0.048) == ANK_RZ_FLOOR_M
    assert pytest.approx(1.35) == ANK_RZ_MIN_VS_CALF_B
    assert pytest.approx(0.60) == ANK_RZ_MAX_FRAC_ANK_Z
    assert pytest.approx(0.005) == HEEL_CONTACT_OVERLAP_TARGET_M
    # Validate tol fence (constraints only — not emit budget)
    assert pytest.approx(0.015) == HEEL_REACH_GAP_TOL_M


# ---------------------------------------------------------------------------
# T1-T5 product contact mass + constraints
# ---------------------------------------------------------------------------


def test_t1_product_heel_ank_overlap() -> None:
    """T1: product-like heel_top - ank_bottom >= overlap target."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    gap = _contact_overlap(pkg.parts)
    assert gap >= HEEL_CONTACT_OVERLAP_TARGET_M - EPS, (
        f"overlap={gap} < target={HEEL_CONTACT_OVERLAP_TARGET_M}"
    )


def test_t2_product_ank_rz_frac_wins() -> None:
    """T2: product ank_rz >= frac*hw (frac wins over floor)."""
    report = _product_feet_report(half_width_m=PRODUCT_HW_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    assert ank.rz_m is not None
    frac_rz = ANK_RZ_FRAC_HALF_W * PRODUCT_HW_M
    assert float(ank.rz_m) >= frac_rz - EPS
    assert frac_rz > ANK_RZ_FLOOR_M
    assert float(ank.rz_m) > ANK_RZ_FLOOR_M + EPS or abs(float(ank.rz_m) - frac_rz) < 1e-5


def test_t3_product_ank_ry_frac_wins() -> None:
    """T3: product ank_ry >= frac*hw (frac wins — AI2 P2-3)."""
    report = _product_feet_report(half_width_m=PRODUCT_HW_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    assert ank.ry_m is not None
    frac_ry = ANK_RY_FRAC_HALF_W * PRODUCT_HW_M
    assert float(ank.ry_m) >= frac_ry - EPS
    assert frac_ry > ANK_RY_FLOOR_M
    assert float(ank.ry_m) > ANK_RY_FLOOR_M + EPS or abs(float(ank.ry_m) - frac_ry) < 1e-5


def test_t4_ank_rx_equals_half_width() -> None:
    """T4: ank_rx == half_width (C_foot_width)."""
    report = _product_feet_report(half_width_m=PRODUCT_HW_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    assert ank.rx_m is not None
    assert float(ank.rx_m) == pytest.approx(PRODUCT_HW_M, abs=1e-6)


def test_t5_constraints_foot_stack_green() -> None:
    """T5: C_heel_reaches / C_foot_width / C_ankle_over_heel pass."""
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
# T6-T7 fence 0054 / 0044 constants
# ---------------------------------------------------------------------------


def test_t6_0054_sole_toe_fence() -> None:
    """T6: SOLE_* / TOE_R_FRAC still at shipped 0054 values; plate.rz matches law."""
    assert pytest.approx(0.025) == SOLE_THICKNESS_FRAC_H
    assert pytest.approx(0.70) == SOLE_RZ_FRAC_OF_THICKNESS
    assert pytest.approx(0.016) == SOLE_RZ_FLOOR_M
    assert pytest.approx(0.36) == TOE_R_FRAC_HALF_W
    report = _product_feet_report(height_m=1.72)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    assert plate.rz_m is not None
    expect_rz = max(
        SOLE_THICKNESS_FRAC_H * 1.72 * SOLE_RZ_FRAC_OF_THICKNESS,
        SOLE_RZ_FLOOR_M,
    )
    assert float(plate.rz_m) == pytest.approx(expect_rz, abs=1e-4)


def test_t7_0044_length_heel_fence() -> None:
    """T7: FOOT_LEN_* / HEEL_REAR_Y_* / HEEL_Z_FRAC / HEEL_RZ_CAP / HEEL_RY_MIN unchanged."""
    assert pytest.approx(0.12) == FOOT_LEN_VISUAL_MIN_FRAC_H
    assert pytest.approx(4.8) == FOOT_LEN_MIN_VS_ANK_HW
    assert pytest.approx(1.55) == FOOT_LEN_MIN_VS_CALF_DIAM
    assert pytest.approx(0.12) == HEEL_REAR_Y_BIAS_FRAC_DEPTH
    assert pytest.approx(0.42) == HEEL_Z_FRAC_ANK
    assert pytest.approx(0.48) == HEEL_RZ_CAP_FRAC_ANK
    assert pytest.approx(0.42) == HEEL_RY_MIN_FRAC_DEPTH


# ---------------------------------------------------------------------------
# T8 messages
# ---------------------------------------------------------------------------


def test_t8_messages_include_ank_contact() -> None:
    """T8: messages include 'ank contact'."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    joined = "\n".join(pkg.messages)
    assert "ank contact" in joined
    assert any("ank contact mass" in m and m.startswith("foot_") for m in pkg.messages)


# ---------------------------------------------------------------------------
# T9 / T9b product frac-over-floor invariants
# ---------------------------------------------------------------------------


def test_t9_rz_frac_wins_on_product_hw() -> None:
    """T9: ANK_RZ_FRAC_HALF_W * 0.0263 > ANK_RZ_FLOOR_M."""
    assert ANK_RZ_FRAC_HALF_W * PRODUCT_HW_M > ANK_RZ_FLOOR_M


def test_t9b_ry_frac_wins_on_product_hw() -> None:
    """T9b: ANK_RY_FRAC_HALF_W * 0.0263 > ANK_RY_FLOOR_M (AI2 P2-3)."""
    assert ANK_RY_FRAC_HALF_W * PRODUCT_HW_M > ANK_RY_FLOOR_M


# ---------------------------------------------------------------------------
# T10 floor path (tiny hw)
# ---------------------------------------------------------------------------


def test_t10_ank_rz_floor_binds_when_hw_tiny() -> None:
    """T10: tiny hw -> ANK_RZ_FLOOR_M binds (before ceiling)."""
    tiny_hw = 0.015
    assert ANK_RZ_FRAC_HALF_W * tiny_hw < ANK_RZ_FLOOR_M
    report = _product_feet_report(half_width_m=tiny_hw, ank_z=0.15)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    ank = _ank(pkg.parts)
    assert ank.rz_m is not None
    # Floor binds; ceiling at ank_z=0.15 is 0.09 > floor
    assert float(ank.rz_m) == pytest.approx(ANK_RZ_FLOOR_M, abs=1e-5)


# ---------------------------------------------------------------------------
# T11 / T11b contact emit paths
# ---------------------------------------------------------------------------


def test_t11_no_cap_path_overlap() -> None:
    """T11: product no-cap path overlap >= target."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    gap = _contact_overlap(pkg.parts)
    assert gap >= HEEL_CONTACT_OVERLAP_TARGET_M - EPS
    # Cap-lose message should not appear on product path
    assert not any("cap lose" in m for m in pkg.messages)


def test_t11b_cap_lose_never_minus_tol_float() -> None:
    """T11b: force still_need/cap-lose path; never ≈ -0.015 float (AI2 P2-1).

    Tall ank_z + floor-stuck ank_rz (tiny hw) makes reach_need > rz_cap, then
    still_need after clamp exceeds rz_cap and emits 'heel_rz cap lose for contact'.
    """
    # reach_need > rz_cap when ank_z large and ank_rz ~ floor (tiny hw)
    report = _product_feet_report(half_width_m=0.02, ank_z=0.50)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    gap = _contact_overlap(pkg.parts)
    joined = "\n".join(pkg.messages)
    assert "cap lose" in joined, (
        f"expected still_need cap-lose message; gap={gap}; msgs={pkg.messages}"
    )
    # Cap-lose still keeps contact target (or at least non-float)
    assert gap >= HEEL_CONTACT_OVERLAP_TARGET_M - EPS or gap >= -EPS, (
        f"gap={gap} must not float under heel on cap-lose"
    )
    # Never reintroduce validate-tol float edge
    assert abs(gap - (-HEEL_REACH_GAP_TOL_M)) > 1e-3, (
        f"gap={gap} ≈ -HEEL_REACH_GAP_TOL (emit must not budget validate tol)"
    )
    assert gap > -HEEL_REACH_GAP_TOL_M + 1e-3


# ---------------------------------------------------------------------------
# T12 ank_rz ceiling (AI2 P2-2)
# ---------------------------------------------------------------------------


def test_t12_ank_rz_ceiling_binds_synth() -> None:
    """T12: hw=0.035, ank_z=0.08 -> ceiling binds (not 2.0*0.035 balloon)."""
    hw = 0.035
    ank_z = 0.08
    report = _product_feet_report(half_width_m=hw, ank_z=ank_z)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    ank = _ank(pkg.parts)
    assert ank.rz_m is not None
    ceil = ANK_RZ_MAX_FRAC_ANK_Z * ank_z
    frac_path = ANK_RZ_FRAC_HALF_W * hw  # 0.070 without ceiling
    assert frac_path > ceil  # ceiling must be the binding knob
    assert float(ank.rz_m) <= ceil + EPS
    assert float(ank.rz_m) == pytest.approx(ceil, abs=1e-5)


# ---------------------------------------------------------------------------
# T13 calf floor bind
# ---------------------------------------------------------------------------


def test_t13_calf_floor_binds() -> None:
    """T13: large RECIPE_calf_b r forces ANK_RZ_MIN_VS_CALF_B floor (AI2 P3-1)."""
    report = _product_feet_report(half_width_m=PRODUCT_HW_M, ank_z=PRODUCT_ANK_Z_M)
    # calf_r * 1.35 must beat frac (0.0526) and stay under ceiling (0.60*ank_z≈0.0788)
    calf_r = 0.050
    calf_floor = calf_r * ANK_RZ_MIN_VS_CALF_B  # 0.0675
    assert calf_floor > ANK_RZ_FRAC_HALF_W * PRODUCT_HW_M
    assert calf_floor < ANK_RZ_MAX_FRAC_ANK_Z * PRODUCT_ANK_Z_M
    calf = RecipePart(
        name="RECIPE_calf_b_l",
        role="limb_segment",
        kind="ellipsoid",
        center=[-0.10, 0.02, PRODUCT_ANK_Z_M],
        rx_m=calf_r,
        ry_m=calf_r,
        rz_m=calf_r,
    )
    msgs: list[str] = []
    parts = build_foot_parts(
        report,
        toes="wedge",
        messages=msgs,
        existing_parts=[calf],
    )
    ank = _ank(parts)
    assert ank.rz_m is not None
    assert float(ank.rz_m) == pytest.approx(calf_floor, abs=1e-5)


# ---------------------------------------------------------------------------
# T14 product ceiling non-binding
# ---------------------------------------------------------------------------


def test_t14_product_ceiling_non_binding() -> None:
    """T14: product ank_rz < ANK_RZ_MAX * ank_z (ceiling non-binding)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    assert ank.center is not None and ank.rz_m is not None
    ank_z = float(ank.center[2])
    assert float(ank.rz_m) < ANK_RZ_MAX_FRAC_ANK_Z * ank_z - EPS
