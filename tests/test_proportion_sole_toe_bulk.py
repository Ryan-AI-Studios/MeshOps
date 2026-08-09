"""Track 0054 - sole thickness + full-toe bulk freezes (T1-T12).

Product half_width ~ 0.0263 must be used for T3/T3b/T12 so dead-frac
regressions are not masked by synthetic ank_foot hw=0.035.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from meshops.proportion.blockout_recipe import build_blockout_recipe
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.extremity_recipe import (
    FOOT_LEN_MIN_VS_ANK_HW,
    FOOT_LEN_MIN_VS_CALF_DIAM,
    FOOT_LEN_VISUAL_MIN_FRAC_H,
    SOLE_RZ_FLOOR_M,
    SOLE_RZ_FRAC_OF_THICKNESS,
    SOLE_THICKNESS_FRAC_H,
    TOE_BIG_SCALE,
    TOE_FULL_LEN_FRAC,
    TOE_MIN_CENTER_SPACING_VS_R,
    TOE_R_CAP_FRAC_HALF_W,
    TOE_R_FLOOR_M,
    TOE_R_FRAC_HALF_W,
    TOE_SPLAY_FRAC_HALF_W,
    TOE_WEDGE_RZ_FRAC_SOLE,
    apply_foot_length_visual_floor,
)
from meshops.proportion.models import (
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

# Product-class ankle half-width (AI2 P2-2) - must not use 0.035 synthetic default.
PRODUCT_HW_M: float = 0.0263
PRODUCT_H_M: float = 1.72
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
    ank_z: float = 0.13,
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


def _plate(pkg_parts: list, side: str = "l"):
    return next(p for p in pkg_parts if p.name == f"RECIPE_foot_plate_{side}")


def _toe(pkg_parts: list, i: int, side: str = "l"):
    return next(p for p in pkg_parts if p.name == f"RECIPE_toe_{i}_{side}")


# ---------------------------------------------------------------------------
# T1-T2 sole thickness / z_top law
# ---------------------------------------------------------------------------


def test_t1_sole_rz_and_z_top_floors_h_1_72() -> None:
    """T1: H=1.72 -> plate.rz_m >= 0.026; plate.z_top_m >= 0.052."""
    report = _product_feet_report(height_m=1.72)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    assert plate.rz_m is not None and plate.z_top_m is not None
    assert float(plate.rz_m) >= 0.026 - EPS
    assert float(plate.z_top_m) >= 0.052 - EPS
    # Product-class expected: sole_rz ~ 0.0301
    expect_rz = max(
        SOLE_THICKNESS_FRAC_H * 1.72 * SOLE_RZ_FRAC_OF_THICKNESS,
        SOLE_RZ_FLOOR_M,
    )
    assert float(plate.rz_m) == pytest.approx(expect_rz, abs=1e-4)


def test_t2_z_top_is_2x_rz() -> None:
    """T2: z_top_m == 2*rz_m (B4 law)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    plate = _plate(pkg.parts)
    assert plate.rz_m is not None and plate.z_top_m is not None
    assert float(plate.z_top_m) == pytest.approx(2.0 * float(plate.rz_m), abs=1e-6)


# ---------------------------------------------------------------------------
# T3-T4 full toe bulk (product hw)
# ---------------------------------------------------------------------------


def test_t3_full_toe_mid_radius_frac_wins_product_hw() -> None:
    """T3: product hw~0.0263 mid toe radius >= frac*hw (frac must win over floor)."""
    report = _product_feet_report(half_width_m=PRODUCT_HW_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    mid = _toe(pkg.parts, 3)
    assert mid.radius_m is not None
    r = float(mid.radius_m)
    frac_r = TOE_R_FRAC_HALF_W * PRODUCT_HW_M
    assert r >= frac_r - EPS, f"mid r={r} < frac path {frac_r}"
    assert r >= TOE_R_FLOOR_M - EPS
    # Frac must win: deliverable is not floor-only
    assert frac_r > TOE_R_FLOOR_M
    assert r > TOE_R_FLOOR_M + 1e-6 or abs(r - frac_r) < 1e-5


def test_t3b_constant_frac_wins_on_product_hw() -> None:
    """T3b: TOE_R_FRAC_HALF_W * 0.0263 > TOE_R_FLOOR_M (AI2 P2-1 invariant)."""
    assert TOE_R_FRAC_HALF_W * PRODUCT_HW_M > TOE_R_FLOOR_M


def test_t3c_dead_frac_detector() -> None:
    """T3c: frac band >= 0.35; if frac were 0.30 product would be floor-bound."""
    assert TOE_R_FRAC_HALF_W >= 0.35
    # Live default 0.36 must beat floor on product hw
    assert TOE_R_FRAC_HALF_W * PRODUCT_HW_M > TOE_R_FLOOR_M
    # Dead-frac regression: 0.30 * product hw < floor -> would stick to floor
    dead = 0.30 * PRODUCT_HW_M
    assert dead < TOE_R_FLOOR_M, "dead-frac detector assumes 0.30*hw is floor-bound"


def test_t4_big_toe_scale() -> None:
    """T4: toe_1 radius >= toe_3 * TOE_BIG_SCALE - eps (named freeze)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    r1 = float(_toe(pkg.parts, 1).radius_m or 0.0)
    r3 = float(_toe(pkg.parts, 3).radius_m or 0.0)
    # Big toe may be cap-limited; still >= mid * scale when under cap
    cap = TOE_R_CAP_FRAC_HALF_W * PRODUCT_HW_M
    expected = min(r3 * TOE_BIG_SCALE, cap)
    assert r1 >= expected - EPS
    assert r1 >= r3 * TOE_BIG_SCALE - EPS or abs(r1 - cap) < EPS


# ---------------------------------------------------------------------------
# T5 wedge tracks sole
# ---------------------------------------------------------------------------


def test_t5_wedge_toe_rz_tracks_sole() -> None:
    """T5: toe_soft.rz_m >= plate.rz * TOE_WEDGE_RZ_FRAC_SOLE - eps."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    plate = _plate(pkg.parts)
    toe = next(p for p in pkg.parts if p.name == "RECIPE_toe_soft_l")
    assert plate.rz_m is not None and toe.rz_m is not None
    assert float(toe.rz_m) >= float(plate.rz_m) * TOE_WEDGE_RZ_FRAC_SOLE - EPS


# ---------------------------------------------------------------------------
# T6 arch/ball in sole band
# ---------------------------------------------------------------------------


def test_t6_arch_ball_centers_in_sole_band() -> None:
    """T6: arch/ball centers z <= sole_top roughly; not tower above plate."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    by_name = {p.name: p for p in pkg.parts}
    plate = by_name["RECIPE_foot_plate_l"]
    arch = by_name["RECIPE_arch_soft_l"]
    ball = by_name["RECIPE_ball_soft_l"]
    assert plate.center is not None and plate.rz_m is not None
    sole_top = float(plate.center[2]) + float(plate.rz_m)
    assert arch.center is not None and ball.center is not None
    assert float(arch.center[2]) <= sole_top + EPS
    assert float(ball.center[2]) <= sole_top + EPS
    # Centers + rz should not sit absurdly above sole_top
    assert (
        float(arch.center[2]) + float(arch.rz_m or 0.0) <= sole_top + float(plate.rz_m) * 1.5 + 0.02
    )
    assert (
        float(ball.center[2]) + float(ball.rz_m or 0.0) <= sole_top + float(plate.rz_m) * 1.5 + 0.02
    )


# ---------------------------------------------------------------------------
# T7 constraints green
# ---------------------------------------------------------------------------


def test_t7_constraints_full_toes_green() -> None:
    """T7: C_toe_forward / C_heel_reaches / C_toe_sole_z / foot width path pass."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    for rule_id in (
        "C_toe_forward_of_heel",
        "C_heel_reaches_ank_foot",
        "C_toe_sole_z",
        "C_foot_width",
    ):
        assert rule_id in by_id, f"missing rule {rule_id}"
        assert by_id[rule_id].status == "pass", (
            f"{rule_id} status={by_id[rule_id].status}: {by_id[rule_id].message}"
        )


# ---------------------------------------------------------------------------
# T8 0044 length freezes still importable
# ---------------------------------------------------------------------------


def test_t8_foot_len_freezes_importable_and_floor() -> None:
    """T8: FOOT_LEN_* freezes importable; length floor still applies."""
    assert pytest.approx(0.13) == FOOT_LEN_VISUAL_MIN_FRAC_H
    assert FOOT_LEN_MIN_VS_ANK_HW > 0
    assert FOOT_LEN_MIN_VS_CALF_DIAM > 0
    msgs: list[str] = []
    out = apply_foot_length_visual_floor(
        0.10,
        height_m=1.72,
        half_width=PRODUCT_HW_M,
        calf_distal_r=None,
        messages=msgs,
        side="l",
    )
    expect = FOOT_LEN_VISUAL_MIN_FRAC_H * 1.72
    assert out >= expect - EPS


# ---------------------------------------------------------------------------
# T9 full toe past plate front (-Y)
# ---------------------------------------------------------------------------


def test_t9_full_toe_tip_past_plate_front() -> None:
    """T9: full toe p1.y (tip) < plate front y (past -Y)."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    assert plate.center is not None
    half_d = float(plate.half_depth_m or plate.ry_m or 0.0)
    plate_front = float(plate.center[1]) - half_d
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        assert toe.p1 is not None
        assert float(toe.p1[1]) < plate_front + EPS, (
            f"toe_{i} tip y={toe.p1[1]} not past plate front {plate_front}"
        )


# ---------------------------------------------------------------------------
# T10 messages
# ---------------------------------------------------------------------------


def test_t10_messages_sole_thickness_and_toe_bulk() -> None:
    """T10: messages include sole thickness + toe bulk; no hand bulk: collision."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    joined = "\n".join(pkg.messages)
    assert "sole thickness" in joined
    assert "toe bulk" in joined
    # Foot path must not use hand bulk exact-count prefix
    foot_msgs = [m for m in pkg.messages if m.startswith("foot_")]
    assert foot_msgs, "expected foot_* diagnostic messages"
    for m in foot_msgs:
        assert not m.startswith("hand bulk:")
        assert "hand bulk:" not in m
    assert any("sole thickness" in m and m.startswith("foot_") for m in pkg.messages)
    assert any("toe bulk" in m and m.startswith("foot_") for m in pkg.messages)


# ---------------------------------------------------------------------------
# T11 sole rz floor binds at low H
# ---------------------------------------------------------------------------


def test_t11_sole_rz_floor_binds_at_h_0_8() -> None:
    """T11: H=0.8 -> sole_rz == SOLE_RZ_FLOOR_M (thickness*0.70=0.014 < 0.016)."""
    # thickness = 0.025 * 0.8 = 0.02; *0.70 = 0.014 < 0.016 -> floor binds
    assert SOLE_THICKNESS_FRAC_H * 0.8 * SOLE_RZ_FRAC_OF_THICKNESS < SOLE_RZ_FLOOR_M
    report = _product_feet_report(height_m=0.8, ank_z=0.10)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    plate = _plate(pkg.parts)
    assert plate.rz_m is not None
    assert float(plate.rz_m) == pytest.approx(SOLE_RZ_FLOOR_M, abs=1e-6)


# ---------------------------------------------------------------------------
# T12 soft center spacing (B15)
# ---------------------------------------------------------------------------


def test_t12_adjacent_toe_center_spacing() -> None:
    """T12: product hw adjacent full-toe center spacing >= 1.0 * base_r."""
    report = _product_feet_report(half_width_m=PRODUCT_HW_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    base_r = min(
        max(TOE_R_FRAC_HALF_W * PRODUCT_HW_M, TOE_R_FLOOR_M),
        TOE_R_CAP_FRAC_HALF_W * PRODUCT_HW_M,
    )
    min_gap = TOE_MIN_CENTER_SPACING_VS_R * base_r
    # Use p0 (base) X positions - same Y for all digits in emit
    xs: list[float] = []
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        assert toe.p0 is not None
        xs.append(float(toe.p0[0]))
    xs_sorted = sorted(xs)
    for a, b in pairwise(xs_sorted):
        gap = abs(b - a)
        assert gap >= min_gap - EPS, f"adj gap {gap} < min {min_gap} (base_r={base_r})"


def test_exports_named_freezes() -> None:
    """Sanity: 0054 freezes are public and in expected bands."""
    assert 0.022 <= SOLE_THICKNESS_FRAC_H <= 0.028
    assert 0.62 <= SOLE_RZ_FRAC_OF_THICKNESS <= 0.78
    assert 0.014 <= SOLE_RZ_FLOOR_M <= 0.020
    assert 0.35 <= TOE_R_FRAC_HALF_W <= 0.40
    assert pytest.approx(0.16) == TOE_FULL_LEN_FRAC
    assert pytest.approx(1.25) == TOE_SPLAY_FRAC_HALF_W
    assert pytest.approx(1.20) == TOE_BIG_SCALE
    assert pytest.approx(0.85) == TOE_WEDGE_RZ_FRAC_SOLE
    assert pytest.approx(1.0) == TOE_MIN_CENTER_SPACING_VS_R
