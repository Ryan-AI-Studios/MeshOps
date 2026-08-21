"""Track 0108 — foot sphere stack polish (ank column + wedge toes).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 47 stay. Not mesh/print success. Not boots / 0098 reopen.
Does not reopen 0072 heel_ry, 0076 rz 1.80, 0056 contact, 0054 sole, TOE_R 0.36.
"""

from __future__ import annotations

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion import extremity_recipe as ext
from meshops.proportion.blockout_recipe import (
    RECIPE_SCHEMA_VERSION,
    build_blockout_recipe,
)
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.extremity_recipe import (
    _BALL_SOFT_R_FRAC_FOOT,
    ANK_RY_FLOOR_M,
    ANK_RY_FRAC_HALF_W,
    ANK_RZ_FRAC_HALF_W,
    ARCH_SOFT_RY_FRAC_HALF_DEPTH,
    BALL_SOFT_RY_FRAC_HALF_DEPTH,
    FOOT_HW_MIN_FRAC_LEN,
    FOOT_LEN_MIN_VS_CALF_DIAM,
    FOOT_LEN_VISUAL_MIN_FRAC_H,
    HEEL_CONTACT_OVERLAP_TARGET_M,
    HEEL_REAR_OVERHANG_M,
    HEEL_REAR_Y_BIAS_FRAC_DEPTH,
    HEEL_RY_MIN_FRAC_DEPTH,
    TOE_BALL_NEST_FRAC,
    TOE_FULL_LEN_FRAC,
    TOE_R_CAP_FRAC_HALF_W,
    TOE_R_FRAC_HALF_W,
    TOE_TIP_MAX_PAST_BALL_FRAC,
    TOE_TIP_MAX_PAST_BALL_M,
    TOE_TIP_PAD_RY_FRAC,
    TOE_TIP_PAD_SCALE,
)
from meshops.proportion.skeleton import build_blockout_skeleton
from test_proportion_foot_stack_hierarchy import (
    _PRODUCT_HW_0098_M,
    _THIN_HW_M,
    _ank,
    _arch,
    _ball,
    _contact_overlap,
    _heel,
    _plate,
    _product_class_report,
    _product_feet_report,
    _product_flags,
    _product_pkg,
    _template,
    _toe,
    _toe_tip,
)

_LIVE_REAR_PAST_0106 = 0.0138
_REAR_PAST_MIN_M = 0.012
_EXPECT_FOOT_LEN = 0.2648
EPS = 1e-6


def test_t0_const_freezes() -> None:
    """T0: 0108 ank/arch/ball/nest/tip + new tip_ry; keep 0098/0072/0076/0056/0054."""
    assert ANK_RY_FRAC_HALF_W == 0.78
    assert ARCH_SOFT_RY_FRAC_HALF_DEPTH == 0.18
    assert BALL_SOFT_RY_FRAC_HALF_DEPTH == 0.16
    assert _BALL_SOFT_R_FRAC_FOOT == 0.06
    assert TOE_BALL_NEST_FRAC == 0.52
    assert TOE_TIP_PAD_SCALE == 0.78
    assert TOE_TIP_PAD_RY_FRAC == 0.55
    assert HEEL_REAR_Y_BIAS_FRAC_DEPTH == 0.14
    assert ANK_RZ_FRAC_HALF_W == 1.80
    assert HEEL_RY_MIN_FRAC_DEPTH == 0.30
    assert HEEL_REAR_OVERHANG_M == 0.012
    assert HEEL_CONTACT_OVERLAP_TARGET_M == 0.005
    assert TOE_R_FRAC_HALF_W == 0.36
    assert FOOT_LEN_VISUAL_MIN_FRAC_H == 0.150
    assert FOOT_LEN_MIN_VS_CALF_DIAM == 4.2
    assert FOOT_HW_MIN_FRAC_LEN == 0.16
    assert ANK_RY_FLOOR_M == 0.030


def test_t1_invert_0097_leftover() -> None:
    """T1: flatten/recede/nest vs leftover 0097 1.00/0.26/0.24/0.40/1.00 isotropic tips."""
    assert ANK_RY_FRAC_HALF_W < 1.00
    assert ARCH_SOFT_RY_FRAC_HALF_DEPTH < 0.26
    assert BALL_SOFT_RY_FRAC_HALF_DEPTH < 0.24
    assert TOE_BALL_NEST_FRAC > 0.40
    assert TOE_TIP_PAD_SCALE < 1.00
    assert TOE_TIP_PAD_RY_FRAC < 1.00


def test_t2_product_hw_ank_ry_frac_wins() -> None:
    """T2: 0098-class hw 0.04237 — ank.ry == ank.rx * 0.78 and above pea floor."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        ank = _ank(pkg.parts, side)
        assert ank.rx_m is not None and ank.ry_m is not None
        assert float(ank.ry_m) == pytest.approx(float(ank.rx_m) * 0.78, abs=1e-4)
        assert float(ank.ry_m) > ANK_RY_FLOOR_M


def test_t3_rear_past_grows_after_ank_shrink() -> None:
    """T3: rear_past >= 0.012 and greater than live 0.0138-eps. Do not pin 0.023."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        ank = _ank(pkg.parts, side)
        heel = _heel(pkg.parts, side)
        assert ank.center is not None and heel.center is not None
        assert ank.ry_m is not None and heel.ry_m is not None
        heel_rear = float(heel.center[1]) + float(heel.ry_m)
        ank_rear = float(ank.center[1]) + float(ank.ry_m)
        rear_past = heel_rear - ank_rear
        assert rear_past >= _REAR_PAST_MIN_M, f"{side} rear_past={rear_past}"
        assert rear_past > _LIVE_REAR_PAST_0106 - EPS, f"{side} rear_past={rear_past}"


def test_t4_arch_ball_frac_wins() -> None:
    """T4: arch.ry == 0.18*hd; ball.ry == max(0.06*fl*1.1, 0.16*hd) and frac wins."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        plate = _plate(pkg.parts, side)
        arch = _arch(pkg.parts, side)
        ball = _ball(pkg.parts, side)
        assert plate.ry_m is not None and plate.half_depth_m is not None
        assert arch.ry_m is not None and ball.ry_m is not None
        hd = float(plate.half_depth_m)
        fl = 2.0 * hd
        fl_term = _BALL_SOFT_R_FRAC_FOOT * fl * 1.1
        frac_term = BALL_SOFT_RY_FRAC_HALF_DEPTH * hd
        assert fl_term < frac_term
        assert float(arch.ry_m) == pytest.approx(0.18 * hd, abs=2e-3)
        expect_ball = max(fl_term, frac_term)
        assert float(ball.ry_m) == pytest.approx(expect_ball, abs=2e-3)
        assert float(ball.ry_m) == pytest.approx(0.16 * hd, abs=2e-3)


def test_t5_nest_and_tip_past_budget() -> None:
    """T5: nest base_y >= ball_front+0.52*toe_len; inversion; tip_past_ball budget."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    ball = _ball(pkg.parts)
    assert plate.half_depth_m is not None
    assert ball.center is not None and ball.ry_m is not None
    fl = 2.0 * float(plate.half_depth_m)
    toe_len = TOE_FULL_LEN_FRAC * fl
    ball_front = float(ball.center[1]) - float(ball.ry_m)
    nest_y = ball_front + 0.52 * toe_len
    budget = min(TOE_TIP_MAX_PAST_BALL_M, TOE_TIP_MAX_PAST_BALL_FRAC * fl)
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        tip = _toe_tip(pkg.parts, i)
        assert toe.p0 is not None and toe.p1 is not None
        assert tip.center is not None
        base_y = float(toe.p0[1])
        tip_y = float(tip.center[1])
        assert base_y >= nest_y - EPS, f"toe_{i} base_y={base_y} nest={nest_y}"
        assert tip_y <= base_y - 1e-6
        tip_past = ball_front - tip_y
        assert tip_past <= budget + EPS


def test_t6_tip_pad_ry_flatten_cap_retained() -> None:
    """T6: r_pad == min(0.78*r_i, cap) (cap retained, product hw does not bind); ry=0.55*r."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    assert plate.top_half_width_m is not None or plate.rx_m is not None
    hw = float(plate.top_half_width_m or plate.rx_m or 0.0)
    cap = TOE_R_CAP_FRAC_HALF_W * hw
    rxs: list[float] = []
    for i in range(1, 6):
        toe = _toe(pkg.parts, i)
        tip = _toe_tip(pkg.parts, i)
        assert toe.radius_m is not None
        assert tip.rx_m is not None and tip.ry_m is not None
        r_i = float(toe.radius_m)
        raw = 0.78 * r_i
        expect = min(raw, cap)
        assert float(tip.rx_m) == pytest.approx(expect, abs=1e-6)
        assert raw < cap  # product hw does not bind
        assert float(tip.ry_m) == pytest.approx(0.55 * float(tip.rx_m), abs=1e-6)
        assert float(tip.ry_m) < float(tip.rx_m)
        rxs.append(float(tip.rx_m))
    assert rxs[0] >= rxs[2]  # big >= mid


def test_t6b_hierarchy_and_polish_messages() -> None:
    """T6b: hierarchy sibling interpolates ank/arch/ball/tip; polish sibling once."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    hier = [m for m in pkg.messages if m.startswith("foot stack hierarchy:")]
    assert len(hier) == 1
    line = hier[0]
    assert f"ank_ry={ANK_RY_FRAC_HALF_W}" in line
    assert f"arch={ARCH_SOFT_RY_FRAC_HALF_DEPTH}" in line
    assert f"ball={BALL_SOFT_RY_FRAC_HALF_DEPTH}" in line
    assert f"tip={TOE_TIP_PAD_SCALE}" in line
    polish = [m for m in pkg.messages if m.startswith("foot sphere stack polish:")]
    assert len(polish) == 1
    pl = polish[0]
    assert f"ank_ry={ANK_RY_FRAC_HALF_W}" in pl
    assert f"nest={TOE_BALL_NEST_FRAC}" in pl
    assert f"tip={TOE_TIP_PAD_SCALE}" in pl
    assert f"tip_ry={TOE_TIP_PAD_RY_FRAC}" in pl
    assert pl.count("ank_ry=") == 1
    assert pl.count("nest=") == 1
    assert pl.count("tip_ry=") == 1
    assert pl.replace("tip_ry=", "").count("tip=") == 1


def test_t7_missing_feet_flag_skips() -> None:
    """T7: missing feet flag → no foot parts (existing skip)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(feet=False),  # type: ignore[arg-type]
    )
    names = [p.name for p in pkg.parts]
    assert not any(n.startswith("RECIPE_foot_plate_") for n in names)
    assert not any(n.startswith("RECIPE_ank_foot_") for n in names)
    assert not any(n.startswith("RECIPE_heel_") for n in names)
    assert not any(n.startswith("RECIPE_arch_soft_") for n in names)
    assert not any(n.startswith("RECIPE_ball_soft_") for n in names)
    assert not any(n.startswith("RECIPE_toe_") for n in names)


def test_t8_n_parts_schema_mcp47() -> None:
    """T8: n_parts 131 via 0097-style product flags; schema 1.4.0; MCP 47."""
    pkg = _product_pkg()
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert pkg.schema_version == "1.4.0"
    assert len(TOOL_NAMES) == 47


def test_t9_all_exports_tip_ry() -> None:
    """T9: __all__ exports TOE_TIP_PAD_RY_FRAC + existing ank/arch/ball/nest/tip."""
    names = set(ext.__all__)
    assert "TOE_TIP_PAD_RY_FRAC" in names
    assert "ANK_RY_FRAC_HALF_W" in names
    assert "ARCH_SOFT_RY_FRAC_HALF_DEPTH" in names
    assert "BALL_SOFT_RY_FRAC_HALF_DEPTH" in names
    assert "TOE_BALL_NEST_FRAC" in names
    assert "TOE_TIP_PAD_SCALE" in names
    assert "_BALL_SOFT_R_FRAC_FOOT" not in names


def test_t10_foot_constraints_and_contact() -> None:
    """T10: C_* foot stack pass; contact >= +0.005."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(),  # type: ignore[arg-type]
    )
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    for rule_id in (
        "C_toe_forward_of_heel",
        "C_heel_reaches_ank_foot",
        "C_toe_sole_z",
        "C_foot_width",
        "C_ankle_over_heel",
    ):
        assert rule_id in by_id, f"missing {rule_id}"
        assert by_id[rule_id].status == "pass", (
            f"{rule_id} status={by_id[rule_id].status}: {by_id[rule_id].message}"
        )
    feet = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    feet_pkg = build_blockout_recipe(feet, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        gap = _contact_overlap(feet_pkg.parts, side)
        assert gap >= HEEL_CONTACT_OVERLAP_TARGET_M - EPS, f"{side} gap={gap}"


def test_t11_0098_len_hw_hold() -> None:
    """T11: 0098 len/hw hold on product-class (foot_len≈0.2648 / hw≈0.04237)."""
    pkg = _product_pkg()
    for side in ("l", "r"):
        plate = _plate(pkg.parts, side)
        assert plate.half_depth_m is not None
        foot_len = 2.0 * float(plate.half_depth_m)
        assert foot_len == pytest.approx(_EXPECT_FOOT_LEN, abs=1e-4)
        assert float(plate.top_half_width_m or 0.0) == pytest.approx(_PRODUCT_HW_0098_M, abs=1e-4)


def test_t12_0072_heel_ry_0076_rz_hold() -> None:
    """T12: 0072 heel_ry still >=0.30*hd; 0076 rz still 1.80*hw."""
    assert HEEL_RY_MIN_FRAC_DEPTH == 0.30
    assert ANK_RZ_FRAC_HALF_W == 1.80
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        plate = _plate(pkg.parts, side)
        heel = _heel(pkg.parts, side)
        ank = _ank(pkg.parts, side)
        assert plate.half_depth_m is not None and heel.ry_m is not None
        hd = float(plate.half_depth_m)
        assert float(heel.ry_m) >= 0.30 * hd - 2e-3
        assert ank.rx_m is not None and ank.rz_m is not None
        assert float(ank.rz_m) == pytest.approx(1.80 * float(ank.rx_m), abs=2e-3)


def test_t13_compact_culls_tip_and_arch() -> None:
    """T13: compact: 0 toe_tip / 0 arch_soft; still plate/heel/ank/ball/five toe_*."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    names = [p.name for p in pkg.parts]
    assert not any(n.startswith("RECIPE_toe_tip_") for n in names)
    assert not any(n.startswith("RECIPE_arch_soft_") for n in names)
    for side in ("l", "r"):
        assert f"RECIPE_foot_plate_{side}" in names
        assert f"RECIPE_heel_{side}" in names
        assert f"RECIPE_ank_foot_{side}" in names
        assert f"RECIPE_ball_soft_{side}" in names
        for i in range(1, 6):
            assert f"RECIPE_toe_{i}_{side}" in names


def test_t14_thin_hw_floor_binds() -> None:
    """T14: thin-hw 0.0263: 0.78*0.0263 < 0.030 floor-binds; product path frac-wins."""
    assert 0.78 * _THIN_HW_M < ANK_RY_FLOOR_M
    report = _product_feet_report(height_m=None, half_width_m=_THIN_HW_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    assert ank.ry_m is not None
    assert float(ank.ry_m) == pytest.approx(ANK_RY_FLOOR_M, abs=1e-6)
    prod = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    prod_pkg = build_blockout_recipe(prod, limbs=False, feet=True, toes="full")
    prod_ank = _ank(prod_pkg.parts)
    assert prod_ank.rx_m is not None and prod_ank.ry_m is not None
    assert float(prod_ank.ry_m) == pytest.approx(float(prod_ank.rx_m) * 0.78, abs=1e-4)
    assert float(prod_ank.ry_m) > ANK_RY_FLOOR_M


def test_t15_ank_rearward_heel_past_flat_tips() -> None:
    """T15: ank front rearward of plate center; heel_rear-ank_rear>=0.012; tip.ry<rx."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        plate = _plate(pkg.parts, side)
        ank = _ank(pkg.parts, side)
        heel = _heel(pkg.parts, side)
        assert plate.center is not None and ank.center is not None
        assert ank.ry_m is not None and heel.center is not None and heel.ry_m is not None
        ank_front = float(ank.center[1]) - float(ank.ry_m)
        assert ank_front > float(plate.center[1])
        heel_rear = float(heel.center[1]) + float(heel.ry_m)
        ank_rear = float(ank.center[1]) + float(ank.ry_m)
        assert heel_rear - ank_rear >= _REAR_PAST_MIN_M
        for i in range(1, 6):
            tip = _toe_tip(pkg.parts, i, side)
            assert tip.rx_m is not None and tip.ry_m is not None
            assert float(tip.ry_m) < float(tip.rx_m)
