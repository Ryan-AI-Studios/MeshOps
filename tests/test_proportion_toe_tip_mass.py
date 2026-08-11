"""Track 0075 - toe tip mass / ball-relative nest freezes (T0-T12).

Product half_width ~ 0.0263 (PRODUCT_HW_M) for composition asserts.
Fences 0054 SOLE_*/TOE_R_*, 0072 heel, 0056 ANK/contact freezes.
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
    TOE_BALL_NEST_FRAC,
    TOE_BASE_NEST_FRAC,
    TOE_BIG_SCALE,
    TOE_FULL_LEN_FRAC,
    TOE_R_CAP_FRAC_HALF_W,
    TOE_R_FLOOR_M,
    TOE_R_FRAC_HALF_W,
    TOE_SPLAY_FRAC_HALF_W,
    TOE_TIP_MAX_PAST_BALL_FRAC,
    TOE_TIP_MAX_PAST_BALL_M,
    TOE_TIP_MAX_PAST_FRAC,
    TOE_TIP_MAX_PAST_M,
    TOE_TIP_PAD_SCALE,
    TOE_TIP_PAST_FRAC,
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


def _ball(pkg_parts: list[RecipePart], side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_ball_soft_{side}")


def _toe(pkg_parts: list[RecipePart], i: int, side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_toe_{i}_{side}")


def _toe_tip(pkg_parts: list[RecipePart], i: int, side: str = "l") -> RecipePart:
    """AI2 P3-8 helper for RECIPE_toe_tip_{i}_{side}."""
    return next(p for p in pkg_parts if p.name == f"RECIPE_toe_tip_{i}_{side}")


def _toe_wedge(pkg_parts: list[RecipePart], side: str = "l") -> RecipePart:
    return next(p for p in pkg_parts if p.name == f"RECIPE_toe_soft_{side}")


# ---------------------------------------------------------------------------
# T0 public freezes
# ---------------------------------------------------------------------------


def test_t0_public_freezes() -> None:
    """T0: 0075 freezes + retuned plate tip freezes."""
    assert pytest.approx(0.40) == TOE_BALL_NEST_FRAC
    assert pytest.approx(0.028) == TOE_TIP_MAX_PAST_BALL_M
    assert pytest.approx(0.12) == TOE_TIP_MAX_PAST_BALL_FRAC
    assert pytest.approx(1.15) == TOE_TIP_PAD_SCALE
    assert pytest.approx(0.55) == TOE_TIP_PAST_FRAC
    assert pytest.approx(0.024) == TOE_TIP_MAX_PAST_M
    assert pytest.approx(0.12) == TOE_TIP_MAX_PAST_FRAC
    assert pytest.approx(0.35) == TOE_BASE_NEST_FRAC
    assert pytest.approx(0.16) == TOE_FULL_LEN_FRAC


# ---------------------------------------------------------------------------
# T1 product base nest — ball wins
# ---------------------------------------------------------------------------


def test_t1_product_base_nest_ball_wins() -> None:
    """T1: product-class base_y >= ball_front + TOE_BALL_NEST_FRAC * toe_len - eps."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    ball = _ball(pkg.parts)
    assert plate.center is not None and ball.center is not None
    assert ball.ry_m is not None
    half_depth = float(plate.half_depth_m or plate.ry_m or 0.0)
    foot_len = 2.0 * half_depth
    toe_len = TOE_FULL_LEN_FRAC * foot_len
    ball_front = float(ball.center[1]) - float(ball.ry_m)
    ball_nest = ball_front + TOE_BALL_NEST_FRAC * toe_len
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        assert toe.p0 is not None
        assert float(toe.p0[1]) >= ball_nest - EPS, (
            f"toe_{i} base_y={toe.p0[1]} < ball_nest={ball_nest}"
        )


# ---------------------------------------------------------------------------
# T2 / T3 tip past ball (+ plate secondary)
# ---------------------------------------------------------------------------


def test_t2_tip_past_ball_budget() -> None:
    """T2: ball_front - tip_y <= min(BALL_M, BALL_FRAC * foot_len) + eps."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    ball = _ball(pkg.parts)
    assert plate.center is not None and ball.center is not None
    assert ball.ry_m is not None
    half_depth = float(plate.half_depth_m or plate.ry_m or 0.0)
    foot_len = 2.0 * half_depth
    ball_front = float(ball.center[1]) - float(ball.ry_m)
    ball_budget = min(TOE_TIP_MAX_PAST_BALL_M, TOE_TIP_MAX_PAST_BALL_FRAC * foot_len)
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        assert toe.p1 is not None
        tip_past_ball = ball_front - float(toe.p1[1])
        assert tip_past_ball <= ball_budget + EPS, (
            f"toe_{i} tip_past_ball={tip_past_ball} > budget={ball_budget}"
        )


def test_t3_ball_primary_plate_secondary() -> None:
    """T3: primary ball gate; secondary plate (negative tip_past_plate OK)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    ball = _ball(pkg.parts)
    assert plate.center is not None and ball.center is not None
    assert ball.ry_m is not None
    half_depth = float(plate.half_depth_m or plate.ry_m or 0.0)
    foot_len = 2.0 * half_depth
    plate_front = float(plate.center[1]) - half_depth
    ball_front = float(ball.center[1]) - float(ball.ry_m)
    ball_budget = min(TOE_TIP_MAX_PAST_BALL_M, TOE_TIP_MAX_PAST_BALL_FRAC * foot_len)
    plate_budget = min(TOE_TIP_MAX_PAST_M, TOE_TIP_MAX_PAST_FRAC * foot_len)
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        assert toe.p1 is not None
        tip_y = float(toe.p1[1])
        assert ball_front - tip_y <= ball_budget + EPS
        assert plate_front - tip_y <= plate_budget + EPS


# ---------------------------------------------------------------------------
# T4 tip pads present + scale hierarchy
# ---------------------------------------------------------------------------


def test_t4_tip_pads_present_and_scale() -> None:
    """T4: tip pads present; r_pad = min(1.15*r_i, cap); big pad >= mid pad."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    # half_width from plate top_half_width or product default
    half_width = float(plate.top_half_width_m or PRODUCT_HW_M)
    r_cap = TOE_R_CAP_FRAC_HALF_W * half_width
    for side in ("l", "r"):
        for i in range(1, 6):
            tip = _toe_tip(pkg.parts, i, side=side)
            toe = _toe(pkg.parts, i, side=side)
            assert tip.kind == "ellipsoid"
            assert tip.role == "toe_soft"
            assert tip.rx_m is not None and toe.radius_m is not None
            r_pad = float(tip.rx_m)
            r_i = float(toe.radius_m)
            expect = min(TOE_TIP_PAD_SCALE * r_i, r_cap)
            assert r_pad == pytest.approx(expect, abs=1e-6), (
                f"toe_tip_{i}_{side} r={r_pad} != expect={expect} (r_i={r_i} cap={r_cap})"
            )
            # Mid toes (uncapped path): pad is at least 1.12 * digit r
            if i != 1:
                assert r_pad >= 1.12 * r_i - EPS, (
                    f"toe_tip_{i}_{side} r={r_pad} < 1.12*r_i={1.12 * r_i}"
                )
        big_pad = float(_toe_tip(pkg.parts, 1, side=side).rx_m or 0.0)
        mid_pad = float(_toe_tip(pkg.parts, 3, side=side).rx_m or 0.0)
        assert big_pad >= mid_pad - EPS, f"big pad {big_pad} < mid pad {mid_pad}"


# ---------------------------------------------------------------------------
# T5 / T5b constraints
# ---------------------------------------------------------------------------


def test_t5_foot_constraints_pass() -> None:
    """T5: foot C_* pass complete feet (tip pads as toe mass — B16 intentional)."""
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


def test_t5b_snapshot_c_toe_forward_and_sole_z() -> None:
    """T5b: C_toe_* metrics use tip-pad geometry (B16 min-Y SoT)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}

    # Tip pads share tip_y / tip_z across digits; min-Y SoT is a tip pad (B16).
    pad_l = _toe_tip(pkg.parts, 3, side="l")
    pad_r = _toe_tip(pkg.parts, 3, side="r")
    assert pad_l.center is not None and pad_r.center is not None
    expect_y_l = float(pad_l.center[1])
    expect_y_r = float(pad_r.center[1])
    expect_z_l = float(pad_l.center[2])
    expect_z_r = float(pad_r.center[2])

    forward = by_id["C_toe_forward_of_heel"]
    assert forward.status == "pass", forward.message
    f_metrics = forward.metrics or {}
    assert "toe_y_l" in f_metrics and "toe_y_r" in f_metrics, f_metrics
    assert float(f_metrics["toe_y_l"]) == pytest.approx(expect_y_l, abs=1e-6)
    assert float(f_metrics["toe_y_r"]) == pytest.approx(expect_y_r, abs=1e-6)
    assert float(f_metrics["toe_y_l"]) < float(f_metrics["heel_y_l"]) - EPS

    sole = by_id["C_toe_sole_z"]
    assert sole.status == "pass", sole.message
    s_metrics = sole.metrics or {}
    assert "toe_z_l" in s_metrics and "toe_z_r" in s_metrics, s_metrics
    assert float(s_metrics["toe_z_l"]) == pytest.approx(expect_z_l, abs=1e-6)
    assert float(s_metrics["toe_z_r"]) == pytest.approx(expect_z_r, abs=1e-6)


# ---------------------------------------------------------------------------
# T6 contact fence
# ---------------------------------------------------------------------------


def test_t6_contact_gap_ge_target() -> None:
    """T6: recipe contact heel_top - ank_bottom >= +0.005."""
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
# T7 / T8 fence pins
# ---------------------------------------------------------------------------


def test_t7_0054_sole_toe_r_freezes_unchanged() -> None:
    """T7: 0054 SOLE_* / TOE_R_* freezes unchanged."""
    assert pytest.approx(0.025) == SOLE_THICKNESS_FRAC_H
    assert pytest.approx(0.70) == SOLE_RZ_FRAC_OF_THICKNESS
    assert pytest.approx(0.016) == SOLE_RZ_FLOOR_M
    assert pytest.approx(0.36) == TOE_R_FRAC_HALF_W
    assert pytest.approx(0.009) == TOE_R_FLOOR_M
    assert pytest.approx(0.45) == TOE_R_CAP_FRAC_HALF_W
    assert pytest.approx(1.20) == TOE_BIG_SCALE
    assert pytest.approx(1.25) == TOE_SPLAY_FRAC_HALF_W


def test_t8_0072_heel_and_0056_ank_freezes_unchanged() -> None:
    """T8: 0072 heel fence + 0076 ank freezes + contact fence."""
    assert pytest.approx(0.30) == HEEL_RY_MIN_FRAC_DEPTH
    assert pytest.approx(0.70) == HEEL_RY_MIN_VS_RZ_FRAC
    assert pytest.approx(0.10) == HEEL_REAR_Y_BIAS_FRAC_DEPTH  # 0076 B3 (was 0.06)
    assert pytest.approx(0.012) == HEEL_REAR_OVERHANG_M
    assert pytest.approx(0.34) == HEEL_RY_MAX_FRAC_HALF_DEPTH
    assert pytest.approx(0.32) == BALL_SOFT_RY_FRAC_HALF_DEPTH
    assert pytest.approx(0.13) == FOOT_LEN_VISUAL_MIN_FRAC_H
    assert pytest.approx(1.22) == ANK_RY_FRAC_HALF_W  # 0076 B1 (was 1.45)
    assert pytest.approx(0.030) == ANK_RY_FLOOR_M  # 0076 B1 (was 0.036)
    assert pytest.approx(1.80) == ANK_RZ_FRAC_HALF_W  # 0076 B2 (was 2.00)
    assert pytest.approx(0.044) == ANK_RZ_FLOOR_M  # 0076 B2 (was 0.048)
    assert pytest.approx(1.35) == ANK_RZ_MIN_VS_CALF_B
    assert pytest.approx(0.60) == ANK_RZ_MAX_FRAC_ANK_Z
    assert pytest.approx(0.005) == HEEL_CONTACT_OVERLAP_TARGET_M
    assert pytest.approx(0.42) == HEEL_Z_FRAC_ANK
    assert pytest.approx(0.48) == HEEL_RZ_CAP_FRAC_ANK


# ---------------------------------------------------------------------------
# T9 length freeze
# ---------------------------------------------------------------------------


def test_t9_toe_full_len_frac_still_016() -> None:
    """T9: TOE_FULL_LEN_FRAC still 0.16 (0072 B5)."""
    assert pytest.approx(0.16) == TOE_FULL_LEN_FRAC


# ---------------------------------------------------------------------------
# T10 wedge path — no tip pads
# ---------------------------------------------------------------------------


def test_t10_wedge_no_tip_pads() -> None:
    """T10: wedge path emits toe_soft wedge; no RECIPE_toe_tip_*."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    names = {p.name for p in pkg.parts}
    assert not any(n.startswith("RECIPE_toe_tip_") for n in names)
    wedge = _toe_wedge(pkg.parts)
    assert wedge.role == "toe_soft"
    assert wedge.kind == "ellipsoid"
    # Explicit: no digit capsules RECIPE_toe_{1..5}_{side}
    for i in range(1, 6):
        assert f"RECIPE_toe_{i}_l" not in names
        assert f"RECIPE_toe_{i}_r" not in names


# ---------------------------------------------------------------------------
# T11 stick gates on capsule tip AND tip-pad center
# ---------------------------------------------------------------------------


def test_t11_capsule_and_pad_y_stick_gates() -> None:
    """T11: capsule p1.y and tip-pad center y within ball budget; tip forward of heel."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    ball = _ball(pkg.parts)
    heel = _heel(pkg.parts)
    assert plate.center is not None and ball.center is not None and heel.center is not None
    assert ball.ry_m is not None
    half_depth = float(plate.half_depth_m or plate.ry_m or 0.0)
    foot_len = 2.0 * half_depth
    ball_front = float(ball.center[1]) - float(ball.ry_m)
    ball_budget = min(TOE_TIP_MAX_PAST_BALL_M, TOE_TIP_MAX_PAST_BALL_FRAC * foot_len)
    heel_y = float(heel.center[1])
    threshold = heel_y - TOE_FORWARD_EPS_M
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        pad = _toe_tip(pkg.parts, i)
        assert toe.p0 is not None and toe.p1 is not None and pad.center is not None
        tip_y = float(toe.p1[1])
        pad_y = float(pad.center[1])
        assert tip_y < float(toe.p0[1]) - 1e-6  # orientation
        assert ball_front - tip_y <= ball_budget + EPS
        assert ball_front - pad_y <= ball_budget + EPS
        assert tip_y < threshold + 1e-9
        assert pad_y < threshold + 1e-9
        # Pad colocated with capsule tip Y
        assert pad_y == pytest.approx(tip_y, abs=1e-6)


# ---------------------------------------------------------------------------
# T12 message tokens
# ---------------------------------------------------------------------------


def test_t12_message_tokens() -> None:
    """T12: toe tip mass + tip_past_ball/plate + base_y + ball_front."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    joined = "\n".join(pkg.messages)
    assert "toe tip mass" in joined
    assert "tip_past_ball=" in joined
    assert "tip_past_plate=" in joined
    assert "base_y=" in joined
    assert "ball_front=" in joined
    # Must not use bare tip_past= alone as the nest message token
    mass_msgs = [m for m in pkg.messages if "toe tip mass" in m]
    assert mass_msgs
    for m in mass_msgs:
        assert "tip_past_ball=" in m
        assert "tip_past_plate=" in m
        # bare tip_past= (without _ball/_plate) should not appear
        assert "tip_past=" not in m.replace("tip_past_ball=", "").replace("tip_past_plate=", "")
        for key in ("tip_past_ball=", "tip_past_plate=", "base_y=", "ball_front="):
            idx = m.index(key) + len(key)
            token = m[idx:].split()[0]
            float(token.rstrip(")"))
