"""Track 0072 - foot heel / toe proportion freezes (T0-T12).

Product half_width ~ 0.0263 (PRODUCT_HW_M) for composition asserts.
True freezes only for 0054 SOLE_*/TOE_R_* and 0056 ANK_*/HEEL contact/Z/rz.
"""

from __future__ import annotations

import pytest

from meshops.proportion.blockout_recipe import RecipePart, build_blockout_recipe
from meshops.proportion.constraints import TOE_FORWARD_EPS_M, validate_constraints
from meshops.proportion.extremity_recipe import (
    ANK_RY_FLOOR_M,
    ANK_RY_FRAC_HALF_W,
    ANK_RZ_FLOOR_M,
    ANK_RZ_FRAC_HALF_W,
    ANK_RZ_MAX_FRAC_ANK_Z,
    ANK_RZ_MIN_VS_CALF_B,
    BALL_SOFT_RY_FRAC_HALF_DEPTH,
    FOOT_LEN_VISUAL_MIN_FRAC_H,
    HEEL_CONTACT_OVERLAP_TARGET_M,
    HEEL_REAR_OVERHANG_M,
    HEEL_REAR_Y_BIAS_FRAC_DEPTH,
    HEEL_RY_MAX_FRAC_HALF_DEPTH,
    HEEL_RY_MIN_FRAC_DEPTH,
    HEEL_RY_MIN_VS_RZ_FRAC,
    HEEL_RZ_CAP_FRAC_ANK,
    HEEL_Z_FRAC_ANK,
    SOLE_RZ_FLOOR_M,
    SOLE_RZ_FRAC_OF_THICKNESS,
    SOLE_THICKNESS_FRAC_H,
    TOE_BASE_NEST_FRAC,
    TOE_BIG_SCALE,
    TOE_FULL_LEN_FRAC,
    TOE_R_CAP_FRAC_HALF_W,
    TOE_R_FLOOR_M,
    TOE_R_FRAC_HALF_W,
    TOE_SPLAY_FRAC_HALF_W,
    TOE_TIP_MAX_PAST_FRAC,
    TOE_TIP_MAX_PAST_M,
    TOE_TIP_PAST_FRAC,
    apply_foot_length_visual_floor,
)
from meshops.proportion.models import (
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

PRODUCT_HW_M: float = 0.0263
PRODUCT_H_M: float = 1.72
PRODUCT_ANK_Z_M: float = 0.1314
EPS: float = 1e-4


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


def _heel(pkg_parts: list[RecipePart], side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_heel_{side}")


def _ank(pkg_parts: list[RecipePart], side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_ank_foot_{side}")


def _plate(pkg_parts: list[RecipePart], side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_foot_plate_{side}")


def _toe(pkg_parts: list[RecipePart], i: int, side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_toe_{i}_{side}")


def _toe_wedge(pkg_parts: list[RecipePart], side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_toe_soft_{side}")


# ---------------------------------------------------------------------------
# T0 public freezes
# ---------------------------------------------------------------------------


def test_t0_public_freezes_export_expected_values() -> None:
    """T0: public freezes export expected 0072 values."""
    assert pytest.approx(0.30) == HEEL_RY_MIN_FRAC_DEPTH
    assert pytest.approx(0.70) == HEEL_RY_MIN_VS_RZ_FRAC
    assert pytest.approx(0.06) == HEEL_REAR_Y_BIAS_FRAC_DEPTH
    assert pytest.approx(0.012) == HEEL_REAR_OVERHANG_M
    assert pytest.approx(0.34) == HEEL_RY_MAX_FRAC_HALF_DEPTH
    assert pytest.approx(0.16) == TOE_FULL_LEN_FRAC
    assert pytest.approx(0.35) == TOE_BASE_NEST_FRAC
    assert pytest.approx(0.90) == TOE_TIP_PAST_FRAC
    assert pytest.approx(0.038) == TOE_TIP_MAX_PAST_M
    assert pytest.approx(0.15) == TOE_TIP_MAX_PAST_FRAC
    assert pytest.approx(0.32) == BALL_SOFT_RY_FRAC_HALF_DEPTH
    assert pytest.approx(0.13) == FOOT_LEN_VISUAL_MIN_FRAC_H


# ---------------------------------------------------------------------------
# T1 / T1b heel ry composition (B1 + B1b)
# ---------------------------------------------------------------------------


def test_t1_winning_heel_ry_le_max_frac_half_depth() -> None:
    """T1: winning heel_ry <= HEEL_RY_MAX_FRAC_HALF_DEPTH x half_depth."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    heel = _heel(pkg.parts)
    assert plate.half_depth_m is not None or plate.ry_m is not None
    assert heel.ry_m is not None
    half_depth = float(plate.half_depth_m or plate.ry_m or 0.0)
    max_ry = HEEL_RY_MAX_FRAC_HALF_DEPTH * half_depth
    assert float(heel.ry_m) <= max_ry + EPS, f"heel_ry={heel.ry_m} > max={max_ry} (hd={half_depth})"


def test_t1b_heel_ry_not_approx_038_half_depth() -> None:
    """T1b: regression - heel_ry must NOT be ~ 0.38xhalf_depth (leftover B1b)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    heel = _heel(pkg.parts)
    assert heel.ry_m is not None
    half_depth = float(plate.half_depth_m or plate.ry_m or 0.0)
    banned = 0.38 * half_depth
    # If 0.38 were still in max(), product path would hit ~banned
    assert float(heel.ry_m) != pytest.approx(banned, abs=1e-4), (
        f"heel_ry={heel.ry_m} ~ 0.38xhd={banned} - leftover half_depth*0.38"
    )
    # And winning term should be B1 floor 0.30xhd (product path)
    expect_b1 = HEEL_RY_MIN_FRAC_DEPTH * half_depth
    assert float(heel.ry_m) == pytest.approx(expect_b1, abs=1e-4)


# ---------------------------------------------------------------------------
# T2 heel rear tip clamp
# ---------------------------------------------------------------------------


def test_t2_heel_rear_tip_within_overhang() -> None:
    """T2: heel_y + heel_ry <= plate_rear + HEEL_REAR_OVERHANG_M + eps."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    heel = _heel(pkg.parts)
    assert plate.center is not None and heel.center is not None
    assert heel.ry_m is not None
    half_depth = float(plate.half_depth_m or plate.ry_m or 0.0)
    plate_rear = float(plate.center[1]) + half_depth
    rear_tip = float(heel.center[1]) + float(heel.ry_m)
    assert rear_tip <= plate_rear + HEEL_REAR_OVERHANG_M + EPS, (
        f"rear_tip={rear_tip} > plate_rear+overhang={plate_rear + HEEL_REAR_OVERHANG_M}"
    )


# ---------------------------------------------------------------------------
# T3 / T4 toe nest composition
# ---------------------------------------------------------------------------


def test_t3_toe_tip_past_within_dual_budget() -> None:
    """T3: plate_front - tip_y <= min(TOE_TIP_MAX_PAST_M, TOE_TIP_MAX_PAST_FRACxfoot_len)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    assert plate.center is not None
    half_depth = float(plate.half_depth_m or plate.ry_m or 0.0)
    foot_len = 2.0 * half_depth
    plate_front = float(plate.center[1]) - half_depth
    budget = min(TOE_TIP_MAX_PAST_M, TOE_TIP_MAX_PAST_FRAC * foot_len)
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        assert toe.p1 is not None
        tip_past = plate_front - float(toe.p1[1])
        assert tip_past <= budget + EPS, f"toe_{i} tip_past={tip_past} > budget={budget}"


def test_t4_toe_base_nests_into_plate() -> None:
    """T4: base_y >= plate_front + TOE_BASE_NEST_FRACxtoe_len - eps."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    assert plate.center is not None
    half_depth = float(plate.half_depth_m or plate.ry_m or 0.0)
    foot_len = 2.0 * half_depth
    toe_len = TOE_FULL_LEN_FRAC * foot_len
    plate_front = float(plate.center[1]) - half_depth
    nest_min = plate_front + TOE_BASE_NEST_FRAC * toe_len
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        assert toe.p0 is not None
        assert float(toe.p0[1]) >= nest_min - EPS, (
            f"toe_{i} base_y={toe.p0[1]} < nest_min={nest_min}"
        )


# ---------------------------------------------------------------------------
# T5 / T5b constraints + snapshot metrics
# ---------------------------------------------------------------------------


def test_t5_foot_constraints_pass_complete_feet() -> None:
    """T5: foot C_* pass via validate_constraints (complete feet)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    for rule_id in (
        "C_toe_forward_of_heel",
        "C_heel_reaches_ank_foot",
        "C_toe_sole_z",
        "C_foot_width",
        "C_ankle_over_heel",
    ):
        assert rule_id in by_id, f"missing rule {rule_id}"
        assert by_id[rule_id].status == "pass", (
            f"{rule_id} status={by_id[rule_id].status}: {by_id[rule_id].message}"
        )


def test_t5b_snapshot_metrics_c_foot_width_and_toe_sole_z() -> None:
    """T5b: snapshot metrics for C_foot_width / C_toe_sole_z (numeric keys)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}

    width = by_id["C_foot_width"]
    assert width.status == "pass", width.message
    w_metrics = width.metrics or {}
    assert (
        "delta_l" in w_metrics
        or "delta_r" in w_metrics
        or any(k.startswith("delta_") for k in w_metrics)
    ), f"C_foot_width metrics missing delta_*: {w_metrics}"
    for k, v in w_metrics.items():
        if k.startswith("delta_"):
            assert float(v) <= 0.015 + 1e-6, f"{k}={v} exceeds width tol"

    sole = by_id["C_toe_sole_z"]
    assert sole.status == "pass", sole.message
    s_metrics = sole.metrics or {}
    # Prefer structured keys; fall back to message tokens when schema varies
    if s_metrics:
        assert any(
            "z" in k.lower() or "toe" in k.lower() or "plate" in k.lower() for k in s_metrics
        ), f"C_toe_sole_z metrics empty of z/toe/plate keys: {s_metrics}"
    else:
        assert "toe" in sole.message.lower() or "plate" in sole.message.lower()


# ---------------------------------------------------------------------------
# T6 recipe contact (not C_* alone)
# ---------------------------------------------------------------------------


def test_t6_recipe_contact_heel_top_minus_ank_bottom() -> None:
    """T6: heel_top - ank_bottom >= +0.005 (real guard, not C_* tol)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    heel = _heel(pkg.parts)
    assert ank.center is not None and ank.rz_m is not None
    assert heel.center is not None and heel.rz_m is not None
    ank_bottom = float(ank.center[2]) - float(ank.rz_m)
    heel_top = float(heel.center[2]) + float(heel.rz_m)
    gap = heel_top - ank_bottom
    assert gap >= HEEL_CONTACT_OVERLAP_TARGET_M - 1e-6, (
        f"contact gap={gap} < target={HEEL_CONTACT_OVERLAP_TARGET_M}"
    )


# ---------------------------------------------------------------------------
# T7 / T8 fence pins (true freezes only)
# ---------------------------------------------------------------------------


def test_t7_0054_sole_toe_r_freezes_unchanged() -> None:
    """T7: 0054 SOLE_* / TOE_R_* freezes unchanged (not TOE_FULL_LEN)."""
    assert pytest.approx(0.025) == SOLE_THICKNESS_FRAC_H
    assert pytest.approx(0.70) == SOLE_RZ_FRAC_OF_THICKNESS
    assert pytest.approx(0.016) == SOLE_RZ_FLOOR_M
    assert pytest.approx(0.36) == TOE_R_FRAC_HALF_W
    assert pytest.approx(0.009) == TOE_R_FLOOR_M
    assert pytest.approx(0.45) == TOE_R_CAP_FRAC_HALF_W
    assert pytest.approx(1.20) == TOE_BIG_SCALE
    assert pytest.approx(1.25) == TOE_SPLAY_FRAC_HALF_W


def test_t8_0056_ank_heel_contact_freezes_unchanged() -> None:
    """T8: 0056 ANK_* / HEEL_CONTACT / HEEL_Z_FRAC / HEEL_RZ_CAP unchanged."""
    assert pytest.approx(1.45) == ANK_RY_FRAC_HALF_W
    assert pytest.approx(0.036) == ANK_RY_FLOOR_M
    assert pytest.approx(2.00) == ANK_RZ_FRAC_HALF_W
    assert pytest.approx(0.048) == ANK_RZ_FLOOR_M
    assert pytest.approx(1.35) == ANK_RZ_MIN_VS_CALF_B
    assert pytest.approx(0.60) == ANK_RZ_MAX_FRAC_ANK_Z
    assert pytest.approx(0.005) == HEEL_CONTACT_OVERLAP_TARGET_M
    assert pytest.approx(0.42) == HEEL_Z_FRAC_ANK
    assert pytest.approx(0.48) == HEEL_RZ_CAP_FRAC_ANK


# ---------------------------------------------------------------------------
# T9 length floor max-only
# ---------------------------------------------------------------------------


def test_t9_length_floor_max_only_never_shrinks() -> None:
    """T9: FOOT_LEN_VISUAL_MIN_FRAC_H=0.13 max-only; long measured never shrinks."""
    assert pytest.approx(0.13) == FOOT_LEN_VISUAL_MIN_FRAC_H
    floor = FOOT_LEN_VISUAL_MIN_FRAC_H * PRODUCT_H_M
    # Short measured lifts to floor
    short = apply_foot_length_visual_floor(
        0.10,
        height_m=PRODUCT_H_M,
        half_width=PRODUCT_HW_M,
        calf_distal_r=None,
        messages=[],
        side="l",
    )
    assert short >= floor - 1e-6
    # Long measured never shrinks
    long_in = 0.30
    long_out = apply_foot_length_visual_floor(
        long_in,
        height_m=PRODUCT_H_M,
        half_width=PRODUCT_HW_M,
        calf_distal_r=None,
        messages=[],
        side="l",
    )
    assert long_out >= long_in - 1e-9


# ---------------------------------------------------------------------------
# T10 wedge still emits tip past front
# ---------------------------------------------------------------------------


def test_t10_wedge_still_emits_tip_past_front() -> None:
    """T10: wedge still emits; tip (center - ry bulk) past plate front."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    plate = _plate(pkg.parts)
    wedge = _toe_wedge(pkg.parts)
    assert plate.center is not None and wedge.center is not None
    assert wedge.ry_m is not None
    half_depth = float(plate.half_depth_m or plate.ry_m or 0.0)
    plate_front = float(plate.center[1]) - half_depth
    # Wedge center should be past front (more -Y)
    assert float(wedge.center[1]) < plate_front + EPS, (
        f"wedge center y={wedge.center[1]} not past plate front {plate_front}"
    )


# ---------------------------------------------------------------------------
# T11 digit tip forward of heel
# ---------------------------------------------------------------------------


def test_t11_each_digit_tip_forward_of_heel() -> None:
    """T11: each digit tip p1[1] < heel Y - TOE_FORWARD_EPS (tip, not center)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    heel = _heel(pkg.parts)
    assert heel.center is not None
    heel_y = float(heel.center[1])
    threshold = heel_y - TOE_FORWARD_EPS_M
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        assert toe.p1 is not None
        assert float(toe.p1[1]) < threshold + 1e-9, (
            f"toe_{i} tip y={toe.p1[1]} not < heel_y-eps={threshold}"
        )


# ---------------------------------------------------------------------------
# T12 messages numeric heel proportion + toe nest
# ---------------------------------------------------------------------------


def test_t12_messages_heel_proportion_and_toe_nest_numeric() -> None:
    """T12: messages contain heel proportion + toe nest AND numeric rear_tip/tip_past."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    joined = "\n".join(pkg.messages)
    assert "heel proportion" in joined
    assert "toe nest" in joined
    assert "rear_tip=" in joined
    assert "tip_past=" in joined
    # Numeric values present (not empty after =)
    heel_msgs = [m for m in pkg.messages if "heel proportion" in m]
    nest_msgs = [m for m in pkg.messages if "toe nest" in m]
    assert heel_msgs
    assert nest_msgs
    for m in heel_msgs:
        assert "ry=" in m and "rear_tip=" in m
        # extract a float after rear_tip=
        idx = m.index("rear_tip=") + len("rear_tip=")
        token = m[idx:].split()[0]
        float(token.rstrip(")"))
    for m in nest_msgs:
        assert "tip_past=" in m and "base_y=" in m
        idx = m.index("tip_past=") + len("tip_past=")
        token = m[idx:].split()[0]
        float(token.rstrip(")"))
