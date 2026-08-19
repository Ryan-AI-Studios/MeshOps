"""Track 0097 — foot stack hierarchy (sole owns stack after 0080).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 46 stay. Not mesh/print success. Not boots / 0098 scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    CALF_DIST_END_SCALE,
    RECIPE_SCHEMA_VERSION,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.extremity_recipe import (
    _BALL_SOFT_R_FRAC_FOOT,
    ANK_RY_FLOOR_M,
    ANK_RY_FRAC_HALF_W,
    ANK_RZ_FRAC_HALF_W,
    ARCH_SOFT_RY_FRAC_HALF_DEPTH,
    BALL_SOFT_RY_FRAC_HALF_DEPTH,
    FOOT_LEN_MIN_VS_CALF_DIAM,
    HEEL_CONTACT_OVERLAP_TARGET_M,
    HEEL_REAR_OVERHANG_M,
    HEEL_REAR_Y_BIAS_FRAC_DEPTH,
    HEEL_RY_MIN_FRAC_DEPTH,
    TOE_TIP_MAX_PAST_BALL_FRAC,
    TOE_TIP_MAX_PAST_BALL_M,
    TOE_TIP_PAD_SCALE,
)
from meshops.proportion.models import (
    CrossSection,
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import build_blockout_skeleton

_PRODUCT_NOFUSE = Path("work/rogue-v3/blockout/product_0097up/nofuse")
_MID_R = 0.0613
_CALF_HW = 0.04379
_PRODUCT_HW_0098_M = 0.04237
_THIN_HW_M = 0.0263
_EXPECT_FOOT_LEN = 0.2648
_EXPECT_SOLE_RZ = 0.0301
_EXPECT_CALF_B = 0.03153
_EXPECT_KNEE_RX = 0.04767
_REAR_PAST_MIN_M = 0.012
_REAR_PAST_FAIL_M = 0.008
EPS = 1e-6


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


def _product_feet_report(
    *,
    height_m: float | None = 1.72,
    half_width_m: float = _PRODUCT_HW_0098_M,
    ank_z: float = 0.1314,
    heel_y: float = 0.06,
    toe_y: float = -0.12,
) -> ProportionReport:
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
        _diam("thigh_l", half_width_m=_MID_R),
        _diam("thigh_r", half_width_m=_MID_R),
        _diam("calf_l", half_width_m=_CALF_HW),
        _diam("calf_r", half_width_m=_CALF_HW),
        _diam("ank_foot_l", half_width_m=_THIN_HW_M),
        _diam("ank_foot_r", half_width_m=_THIN_HW_M),
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


def _template(*, taper: float = 0.22, thigh_tilt_deg: float = 10.0) -> TemplateAppliedPackage:
    constants = AppliedConstants(
        breast_mode="dual_tilted",
        glute_mode_default="two_spheres",
        torso_mode_default="ovals",
        torso_waist_taper=taper,
        thigh_tilt_deg=thigh_tilt_deg,
    )
    return TemplateAppliedPackage(
        template_id="female_adult_athletic",
        sex="female",  # type: ignore[arg-type]
        archetype="adult_athletic",
        source_report="mem",
        height_m=1.72,
        constants=constants,
    )


def _ank(pkg_parts: list, side: str = "l"):
    return next(p for p in pkg_parts if p.name == f"RECIPE_ank_foot_{side}")


def _heel(pkg_parts: list, side: str = "l"):
    return next(p for p in pkg_parts if p.name == f"RECIPE_heel_{side}")


def _plate(pkg_parts: list, side: str = "l"):
    return next(p for p in pkg_parts if p.name == f"RECIPE_foot_plate_{side}")


def _arch(pkg_parts: list, side: str = "l"):
    return next(p for p in pkg_parts if p.name == f"RECIPE_arch_soft_{side}")


def _ball(pkg_parts: list, side: str = "l"):
    return next(p for p in pkg_parts if p.name == f"RECIPE_ball_soft_{side}")


def _toe(pkg_parts: list, i: int, side: str = "l"):
    return next(p for p in pkg_parts if p.name == f"RECIPE_toe_{i}_{side}")


def _toe_tip(pkg_parts: list, i: int, side: str = "l"):
    return next(p for p in pkg_parts if p.name == f"RECIPE_toe_tip_{i}_{side}")


def _contact_overlap(pkg_parts: list, side: str = "l") -> float:
    ank = _ank(pkg_parts, side)
    heel = _heel(pkg_parts, side)
    assert ank.center is not None and ank.rz_m is not None
    assert heel.center is not None and heel.rz_m is not None
    ank_bottom = float(ank.center[2]) - float(ank.rz_m)
    heel_top = float(heel.center[2]) + float(heel.rz_m)
    return heel_top - ank_bottom


def _product_pkg():
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    return build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(),  # type: ignore[arg-type]
    )


def test_t0_const_freezes() -> None:
    """T0: B1-B5 + B4b bands; hold 0076 floors / rz / 0072 heel_ry / overhang / contact."""
    assert ANK_RY_FRAC_HALF_W == 1.00
    assert HEEL_REAR_Y_BIAS_FRAC_DEPTH == 0.14
    assert ARCH_SOFT_RY_FRAC_HALF_DEPTH == 0.26
    assert BALL_SOFT_RY_FRAC_HALF_DEPTH == 0.24
    assert _BALL_SOFT_R_FRAC_FOOT == 0.10
    assert TOE_TIP_PAD_SCALE == 1.00
    assert ANK_RY_FLOOR_M == 0.030
    assert ANK_RZ_FRAC_HALF_W == 1.80
    assert HEEL_RY_MIN_FRAC_DEPTH == 0.30
    assert HEEL_REAR_OVERHANG_M == 0.012
    assert HEEL_CONTACT_OVERLAP_TARGET_M == 0.005
    assert 0.90 <= ANK_RY_FRAC_HALF_W <= 1.08
    assert 0.12 <= HEEL_REAR_Y_BIAS_FRAC_DEPTH <= 0.16
    assert 0.22 <= ARCH_SOFT_RY_FRAC_HALF_DEPTH <= 0.30
    assert 0.20 <= BALL_SOFT_RY_FRAC_HALF_DEPTH <= 0.28
    assert 0.08 <= _BALL_SOFT_R_FRAC_FOOT <= 0.11
    assert ANK_RY_FRAC_HALF_W < 1.15


def test_t1_product_class_parts_present() -> None:
    """T1: plate / heel / ank / arch / ball / five toes / five tips; no new names; n=131."""
    pkg = _product_pkg()
    by_name = {p.name for p in pkg.parts}
    for side in ("l", "r"):
        assert f"RECIPE_foot_plate_{side}" in by_name
        assert f"RECIPE_heel_{side}" in by_name
        assert f"RECIPE_ank_foot_{side}" in by_name
        assert f"RECIPE_arch_soft_{side}" in by_name
        assert f"RECIPE_ball_soft_{side}" in by_name
        for i in range(1, 6):
            assert f"RECIPE_toe_{i}_{side}" in by_name
            assert f"RECIPE_toe_tip_{i}_{side}" in by_name
    joined = " ".join(by_name).lower()
    assert "calcaneus" not in joined
    assert "achilles" not in joined
    assert "shoe" not in joined
    assert len(pkg.parts) == 131


def test_t2_product_hw_ank_ry_equals_rx() -> None:
    """T2: 0098-class hw 0.04237 — ank.ry == ank.rx * 1.00 and above pea floor."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        ank = _ank(pkg.parts, side)
        assert ank.rx_m is not None and ank.ry_m is not None
        assert float(ank.ry_m) == pytest.approx(float(ank.rx_m) * ANK_RY_FRAC_HALF_W, abs=1e-4)
        assert float(ank.ry_m) > ANK_RY_FLOOR_M


def test_t3_heel_rear_past_and_dy() -> None:
    """T3 B6/B20: heel_rear - ank_rear >= 0.012; dy approx bias*plate.ry after clamp."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        ank = _ank(pkg.parts, side)
        heel = _heel(pkg.parts, side)
        plate = _plate(pkg.parts, side)
        assert ank.center is not None and heel.center is not None
        assert ank.ry_m is not None and heel.ry_m is not None
        assert plate.ry_m is not None
        heel_rear = float(heel.center[1]) + float(heel.ry_m)
        ank_rear = float(ank.center[1]) + float(ank.ry_m)
        rear_past = heel_rear - ank_rear
        assert rear_past >= _REAR_PAST_MIN_M, f"{side} rear_past={rear_past}"
        assert rear_past >= _REAR_PAST_FAIL_M
        dy = float(heel.center[1]) - float(ank.center[1])
        expect_dy = HEEL_REAR_Y_BIAS_FRAC_DEPTH * float(plate.ry_m)
        assert dy == pytest.approx(expect_dy, abs=2e-3)


def test_t4_arch_ball_frac_wins() -> None:
    """T4: arch.ry == 0.26*hd; B4b fl_term < 0.24*hd so ball.ry is frac; rz sole-class."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        plate = _plate(pkg.parts, side)
        arch = _arch(pkg.parts, side)
        ball = _ball(pkg.parts, side)
        assert plate.ry_m is not None and plate.rz_m is not None
        assert arch.ry_m is not None and ball.ry_m is not None
        hd = float(plate.ry_m)
        fl = 2.0 * hd
        fl_term = _BALL_SOFT_R_FRAC_FOOT * fl * 1.1
        frac_term = BALL_SOFT_RY_FRAC_HALF_DEPTH * hd
        assert fl_term < frac_term
        assert float(arch.ry_m) == pytest.approx(ARCH_SOFT_RY_FRAC_HALF_DEPTH * hd, abs=2e-3)
        assert float(ball.ry_m) == pytest.approx(frac_term, abs=2e-3)
        sole_rz = float(plate.rz_m)
        assert float(arch.rz_m or 0.0) <= sole_rz * 1.25 + EPS
        assert float(ball.rz_m or 0.0) <= sole_rz * 1.25 + EPS


def test_t5_tip_pad_equals_digit() -> None:
    """T5: tip r == digit r at scale 1.00; 0075 tip_past_ball fence holds."""
    report = _product_feet_report(half_width_m=_PRODUCT_HW_0098_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    plate = _plate(pkg.parts)
    ball = _ball(pkg.parts)
    assert plate.ry_m is not None and ball.center is not None and ball.ry_m is not None
    fl = 2.0 * float(plate.ry_m)
    ball_front = float(ball.center[1]) - float(ball.ry_m)
    budget = min(TOE_TIP_MAX_PAST_BALL_M, TOE_TIP_MAX_PAST_BALL_FRAC * fl)
    for side in ("l", "r"):
        for i in range(1, 6):
            toe = _toe(pkg.parts, i, side)
            tip = _toe_tip(pkg.parts, i, side)
            assert toe.radius_m is not None and tip.rx_m is not None
            assert float(tip.rx_m) == pytest.approx(
                TOE_TIP_PAD_SCALE * float(toe.radius_m), abs=1e-6
            )
            assert tip.center is not None
            tip_past = ball_front - float(tip.center[1])
            assert tip_past <= budget + EPS


def test_t6_sibling_after_both() -> None:
    """T6: per-side heel/ank proportion + exactly one const-driven sibling after both."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    msgs = pkg.messages
    foot_l = [i for i, m in enumerate(msgs) if m.startswith("foot_l: heel/ank proportion")]
    foot_r = [i for i, m in enumerate(msgs) if m.startswith("foot_r: heel/ank proportion")]
    sib = [i for i, m in enumerate(msgs) if m.startswith("foot stack hierarchy:")]
    assert len(foot_l) == 1
    assert len(foot_r) == 1
    assert len(sib) == 1
    line = msgs[sib[0]]
    assert f"ank_ry={ANK_RY_FRAC_HALF_W}" in line
    assert f"bias={HEEL_REAR_Y_BIAS_FRAC_DEPTH}" in line
    assert f"arch={ARCH_SOFT_RY_FRAC_HALF_DEPTH}" in line
    assert f"ball={BALL_SOFT_RY_FRAC_HALF_DEPTH}" in line
    assert f"tip={TOE_TIP_PAD_SCALE}" in line
    assert sib[0] > foot_l[0]
    assert sib[0] > foot_r[0]


def test_t7_n_parts_schema_mcp() -> None:
    """T7: n_parts 131 via product flags; schema 1.4.0; MCP 46."""
    pkg = _product_pkg()
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert len(TOOL_NAMES) == 47


def test_t8_product_path_constraints() -> None:
    """T8: product-path foot C_* pass (same flags as 0096 T8)."""
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
    cons_path = _PRODUCT_NOFUSE / "constraints_report.json"
    if cons_path.is_file():
        cons = json.loads(cons_path.read_text(encoding="utf-8"))
        rule_by = {r["id"]: r["status"] for r in cons.get("rules", [])}
        for rule_id in (
            "C_toe_forward_of_heel",
            "C_heel_reaches_ank_foot",
            "C_toe_sole_z",
            "C_foot_width",
            "C_ankle_over_heel",
        ):
            if rule_id in rule_by:
                assert rule_by[rule_id] == "pass"


def test_t9_invert_not_0076_0075() -> None:
    """T9: invert leftover 0076 1.22/0.10 and 0075 pad 1.15 / ball 0.14."""
    assert ANK_RY_FRAC_HALF_W != 1.22
    assert HEEL_REAR_Y_BIAS_FRAC_DEPTH != 0.10
    assert TOE_TIP_PAD_SCALE != 1.15
    assert _BALL_SOFT_R_FRAC_FOOT != 0.14
    from meshops.proportion import extremity_recipe as ext

    assert hasattr(ext, "ARCH_SOFT_RY_FRAC_HALF_DEPTH")


def test_t10_fence_0080_0072_0054_0095() -> None:
    """T10: 0080 len/hw, 0054 sole, 0072 heel_ry, 0080 calf_b, 0095 knee hold."""
    pkg = _product_pkg()
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        plate = by_name[f"RECIPE_foot_plate_{side}"]
        heel = by_name[f"RECIPE_heel_{side}"]
        calf_b = by_name[f"RECIPE_calf_b_{side}"]
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        assert plate.half_depth_m is not None
        foot_len = 2.0 * float(plate.half_depth_m)
        assert foot_len == pytest.approx(_EXPECT_FOOT_LEN, abs=1e-4)
        assert float(plate.top_half_width_m or 0.0) == pytest.approx(_PRODUCT_HW_0098_M, abs=1e-4)
        assert float(plate.rz_m or 0.0) == pytest.approx(_EXPECT_SOLE_RZ, abs=1e-4)
        hd = float(plate.half_depth_m)
        assert float(heel.ry_m or 0.0) == pytest.approx(HEEL_RY_MIN_FRAC_DEPTH * hd, abs=2e-3)
        assert float(calf_b.rx_m or 0.0) == pytest.approx(_EXPECT_CALF_B, abs=1e-4)
        assert float(calf_b.rx_m or 0.0) == pytest.approx(_CALF_HW * CALF_DIST_END_SCALE, abs=1e-4)
        expect_len = FOOT_LEN_MIN_VS_CALF_DIAM * (2.0 * float(calf_b.rx_m or 0.0))
        assert expect_len == pytest.approx(_EXPECT_FOOT_LEN, abs=1e-3)
        assert float(knee.rx_m or 0.0) == pytest.approx(_EXPECT_KNEE_RX, abs=2e-4)
    assert any("calf_diam" in m for m in pkg.messages)


def test_t11_contact_gap() -> None:
    """T11: contact gap >= 0.005 after emit recompute."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        gap = _contact_overlap(pkg.parts, side)
        assert gap >= HEEL_CONTACT_OVERLAP_TARGET_M - EPS, f"{side} gap={gap}"


def test_t12_all_exports_new_consts() -> None:
    """T12: __all__ exports ank / bias / arch / ball / tip."""
    from meshops.proportion import extremity_recipe as ext

    assert "ANK_RY_FRAC_HALF_W" in ext.__all__
    assert "HEEL_REAR_Y_BIAS_FRAC_DEPTH" in ext.__all__
    assert "ARCH_SOFT_RY_FRAC_HALF_DEPTH" in ext.__all__
    assert "BALL_SOFT_RY_FRAC_HALF_DEPTH" in ext.__all__
    assert "TOE_TIP_PAD_SCALE" in ext.__all__


def test_t13_thin_hw_floor_binds() -> None:
    """T13 B13: thin-hw 0.0263 floor-binds (not a product-path fail)."""
    report = _product_feet_report(height_m=None, half_width_m=_THIN_HW_M)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    ank = _ank(pkg.parts)
    assert ank.ry_m is not None
    assert float(ank.ry_m) == pytest.approx(ANK_RY_FLOOR_M, abs=1e-6)


def test_t14_dual_lr_and_join_ready_overhang() -> None:
    """T14: L/R ank/heel/arch radii equalize; join_ready does not stretch past overhang."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full", join_ready=True)
    by_name = {p.name: p for p in pkg.parts}
    for attr in ("rx_m", "ry_m", "rz_m"):
        for stem in ("RECIPE_ank_foot", "RECIPE_heel", "RECIPE_arch_soft"):
            left = getattr(by_name[f"{stem}_l"], attr)
            right = getattr(by_name[f"{stem}_r"], attr)
            assert float(left or 0.0) == pytest.approx(float(right or 0.0), abs=1e-12)
    for side in ("l", "r"):
        heel = _heel(pkg.parts, side)
        plate = _plate(pkg.parts, side)
        assert heel.center is not None and heel.ry_m is not None
        assert plate.center is not None and plate.half_depth_m is not None
        rear_tip = float(heel.center[1]) + float(heel.ry_m)
        plate_rear = float(plate.center[1]) + float(plate.half_depth_m)
        assert rear_tip <= plate_rear + HEEL_REAR_OVERHANG_M + EPS
