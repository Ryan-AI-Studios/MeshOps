"""Track 0098 — foot full-figure scale plus (stature 0.150 + calf 4.2).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 46 stay. Not mesh/print success. Not boots / 0097 reopen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    RECIPE_SCHEMA_VERSION,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.extremity_recipe import (
    _BALL_SOFT_R_FRAC_FOOT,
    ANK_RY_FRAC_HALF_W,
    ARCH_SOFT_RY_FRAC_HALF_DEPTH,
    BALL_SOFT_RY_FRAC_HALF_DEPTH,
    FOOT_HW_MIN_FRAC_H,
    FOOT_HW_MIN_FRAC_LEN,
    FOOT_HW_MIN_VS_CALF_R,
    FOOT_LEN_MIN_VS_CALF_DIAM,
    FOOT_LEN_VISUAL_MAX_FRAC_H,
    FOOT_LEN_VISUAL_MIN_FRAC_H,
    HEEL_CONTACT_OVERLAP_TARGET_M,
    HEEL_REAR_OVERHANG_M,
    HEEL_REAR_Y_BIAS_FRAC_DEPTH,
    HEEL_RY_MIN_FRAC_DEPTH,
    TOE_TIP_PAD_SCALE,
    apply_foot_half_width_visual_floor,
    apply_foot_length_visual_floor,
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

_PRODUCT_NOFUSE = Path("work/rogue-v3/blockout/product_0098up/nofuse")
_MID_R = 0.0613
_CALF_HW = 0.04379
_PRODUCT_HW_M = 0.04237
_THIN_HW_M = 0.0263
_EXPECT_FOOT_LEN = 0.2648
_EXPECT_CALF_B = 0.03153
_EXPECT_CALF_CYL = 0.05167
_EXPECT_KNEE_RX = 0.04767
_REAR_PAST_MIN_M = 0.012
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
    half_width_m: float = _PRODUCT_HW_M,
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


def _ball(pkg_parts: list, side: str = "l"):
    return next(p for p in pkg_parts if p.name == f"RECIPE_ball_soft_{side}")


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
    """T0: B1/B2 lift; hold B3-B5 + B20 hw-frac; invert leftover 0080 0.145/4.0."""
    assert FOOT_LEN_VISUAL_MIN_FRAC_H == 0.150
    assert FOOT_LEN_MIN_VS_CALF_DIAM == 4.2
    assert FOOT_LEN_VISUAL_MAX_FRAC_H == 0.155
    assert FOOT_HW_MIN_FRAC_LEN == 0.16
    assert FOOT_HW_MIN_VS_CALF_R == 1.20
    assert FOOT_HW_MIN_FRAC_H == 0.022
    assert FOOT_LEN_VISUAL_MIN_FRAC_H != 0.145
    assert FOOT_LEN_MIN_VS_CALF_DIAM != 4.0
    assert 0.148 <= FOOT_LEN_VISUAL_MIN_FRAC_H <= 0.152
    assert 4.1 <= FOOT_LEN_MIN_VS_CALF_DIAM <= 4.25
    assert FOOT_HW_MIN_FRAC_LEN < 0.17


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
    assert "boot" not in joined
    assert "shoe_last" not in joined
    assert "shoe" not in joined
    assert len(pkg.parts) == 131


def test_t2_product_len_hw_vs_calf() -> None:
    """T2: 2*plate.hd == 4.2*2*calf_b.rx; len/H in [0.150, 0.155]; hw == 0.16*len."""
    pkg = _product_pkg()
    by_name = {p.name: p for p in pkg.parts}
    h = 1.72
    for side in ("l", "r"):
        plate = by_name[f"RECIPE_foot_plate_{side}"]
        calf_b = by_name[f"RECIPE_calf_b_{side}"]
        assert plate.half_depth_m is not None and calf_b.rx_m is not None
        foot_len = 2.0 * float(plate.half_depth_m)
        expect_len = FOOT_LEN_MIN_VS_CALF_DIAM * (2.0 * float(calf_b.rx_m))
        assert foot_len == pytest.approx(expect_len, abs=1e-4)
        assert 0.150 <= foot_len / h <= 0.155
        hw = float(plate.top_half_width_m or 0.0)
        assert hw == pytest.approx(FOOT_HW_MIN_FRAC_LEN * foot_len, abs=1e-4)
        assert foot_len == pytest.approx(_EXPECT_FOOT_LEN, abs=1e-3)
        assert hw == pytest.approx(_PRODUCT_HW_M, abs=1e-4)


def test_t3_hierarchy_rear_past_holds() -> None:
    """T3 B7: heel_rear - ank_rear >= 0.012; ank.ry/rx == 1.00. Do not assert y_ov."""
    pkg = _product_pkg()
    for side in ("l", "r"):
        ank = _ank(pkg.parts, side)
        heel = _heel(pkg.parts, side)
        assert ank.center is not None and heel.center is not None
        assert ank.ry_m is not None and heel.ry_m is not None
        assert ank.rx_m is not None
        heel_rear = float(heel.center[1]) + float(heel.ry_m)
        ank_rear = float(ank.center[1]) + float(ank.ry_m)
        assert heel_rear - ank_rear >= _REAR_PAST_MIN_M
        assert float(ank.ry_m) / float(ank.rx_m) == pytest.approx(1.00, abs=1e-4)


def test_t4_unit_calf_diam_floor() -> None:
    """T4: apply_foot_length_visual_floor short + calf_r=0.08 → 4.2*2*0.08; source calf_diam."""
    msgs: list[str] = []
    out = apply_foot_length_visual_floor(
        0.10,
        height_m=None,
        half_width=0.02,
        calf_distal_r=0.08,
        messages=msgs,
        side="l",
    )
    expect = FOOT_LEN_MIN_VS_CALF_DIAM * (2.0 * 0.08)
    assert out == pytest.approx(expect, abs=1e-9)
    assert out != pytest.approx(0.64, abs=1e-9)
    assert any("calf_diam" in m for m in msgs)


def test_t5_unit_stature_and_never_shrink() -> None:
    """T5: short + H=1.72 + no calf → 0.150*H; long 0.28 never shrinks (B13)."""
    msgs: list[str] = []
    short = apply_foot_length_visual_floor(
        0.10,
        height_m=1.72,
        half_width=0.02,
        calf_distal_r=None,
        messages=msgs,
        side="l",
    )
    assert short == pytest.approx(FOOT_LEN_VISUAL_MIN_FRAC_H * 1.72, abs=1e-9)
    long = apply_foot_length_visual_floor(
        0.28,
        height_m=1.72,
        half_width=0.02,
        calf_distal_r=None,
        messages=[],
        side="l",
    )
    assert long == pytest.approx(0.28, abs=1e-9)


def test_t6_unit_b15_cap() -> None:
    """T6: calf_r=0.08 + H=1.72 → raw 4.2*diam capped at 0.155*H; length visual floor msg."""
    msgs: list[str] = []
    out = apply_foot_length_visual_floor(
        0.10,
        height_m=1.72,
        half_width=0.02,
        calf_distal_r=0.08,
        messages=msgs,
        side="l",
    )
    raw = FOOT_LEN_MIN_VS_CALF_DIAM * (2.0 * 0.08)
    cap = FOOT_LEN_VISUAL_MAX_FRAC_H * 1.72
    assert raw > cap
    assert out == pytest.approx(cap, abs=1e-9)
    assert any("length visual floor" in m for m in msgs)


def test_t7_unit_width_foot_len_wins() -> None:
    """T7: hw=0.02, foot_len=0.25, calf_r=0.03, H=1.72 → 0.16*0.25; wide 0.06 not shrunk."""
    msgs: list[str] = []
    out = apply_foot_half_width_visual_floor(
        0.02,
        foot_len=0.25,
        height_m=1.72,
        calf_distal_r=0.03,
        messages=msgs,
        side="l",
    )
    expect = FOOT_HW_MIN_FRAC_LEN * 0.25
    assert out == pytest.approx(expect, abs=1e-9)
    assert any("width visual floor" in m for m in msgs)
    assert any("foot_len" in m for m in msgs)
    wide = apply_foot_half_width_visual_floor(
        0.06,
        foot_len=0.25,
        height_m=1.72,
        calf_distal_r=0.03,
        messages=[],
        side="l",
    )
    assert wide == pytest.approx(0.06, abs=1e-9)


def test_t8_sibling_after_both_length_lines() -> None:
    """T8: per-side length visual floor + exactly one const-driven foot scale plus after both."""
    pkg = _product_pkg()
    msgs = pkg.messages
    foot_l = [i for i, m in enumerate(msgs) if m.startswith("foot_l: length visual floor")]
    foot_r = [i for i, m in enumerate(msgs) if m.startswith("foot_r: length visual floor")]
    sib = [i for i, m in enumerate(msgs) if m.startswith("foot scale plus:")]
    assert len(foot_l) == 1
    assert len(foot_r) == 1
    assert len(sib) == 1
    line = msgs[sib[0]]
    assert f"len_h={FOOT_LEN_VISUAL_MIN_FRAC_H}" in line
    assert f"calf={FOOT_LEN_MIN_VS_CALF_DIAM}" in line
    assert f"hw={FOOT_HW_MIN_FRAC_LEN}" in line
    assert f"cap={FOOT_LEN_VISUAL_MAX_FRAC_H}" in line
    assert sib[0] > foot_l[0]
    assert sib[0] > foot_r[0]
    hier = [i for i, m in enumerate(msgs) if m.startswith("foot stack hierarchy:")]
    assert len(hier) == 1


def test_t9_n_parts_schema_mcp() -> None:
    """T9: n_parts 131; schema 1.4.0; MCP 46."""
    pkg = _product_pkg()
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert len(TOOL_NAMES) == 46


def test_t10_product_path_constraints() -> None:
    """T10: product-path foot C_* pass (same flags as 0097 T8)."""
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


def test_t11_fence_0097_0072_0056() -> None:
    """T11: 0097 hierarchy + 0072 heel_ry/overhang + 0056 contact; ball 0.24*hd wins max()."""
    assert ANK_RY_FRAC_HALF_W == 1.00
    assert HEEL_REAR_Y_BIAS_FRAC_DEPTH == 0.14
    assert ARCH_SOFT_RY_FRAC_HALF_DEPTH == 0.26
    assert BALL_SOFT_RY_FRAC_HALF_DEPTH == 0.24
    assert _BALL_SOFT_R_FRAC_FOOT == 0.10
    assert TOE_TIP_PAD_SCALE == 1.00
    assert HEEL_RY_MIN_FRAC_DEPTH == 0.30
    assert HEEL_REAR_OVERHANG_M == 0.012
    assert HEEL_CONTACT_OVERLAP_TARGET_M == 0.005
    pkg = _product_pkg()
    plate = _plate(pkg.parts)
    assert plate.half_depth_m is not None
    hd = float(plate.half_depth_m)
    fl = 2.0 * hd
    fl_term = _BALL_SOFT_R_FRAC_FOOT * fl * 1.1
    frac_term = BALL_SOFT_RY_FRAC_HALF_DEPTH * hd
    assert fl_term < frac_term
    ball = _ball(pkg.parts)
    assert ball.ry_m is not None
    assert float(ball.ry_m) == pytest.approx(frac_term, abs=2e-3)


def test_t12_fence_0096_calf_b_source() -> None:
    """T12: calf_b 0.03153 / calf_cyl 0.05167 / knee 0.04767; floor source calf_diam not belly."""
    pkg = _product_pkg()
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        calf_b = by_name[f"RECIPE_calf_b_{side}"]
        calf_cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        assert float(calf_b.rx_m or 0.0) == pytest.approx(_EXPECT_CALF_B, abs=1e-4)
        assert float(calf_cyl.radius_m or 0.0) == pytest.approx(_EXPECT_CALF_CYL, abs=1e-4)
        assert float(knee.rx_m or 0.0) == pytest.approx(_EXPECT_KNEE_RX, abs=2e-4)
    assert any("calf_diam" in m for m in pkg.messages)
    assert not any("belly" in m and "length visual floor" in m for m in pkg.messages)


def test_t13_contact_gap() -> None:
    """T13: contact gap >= 0.005 after emit recompute."""
    report = _product_feet_report()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    for side in ("l", "r"):
        gap = _contact_overlap(pkg.parts, side)
        assert gap >= HEEL_CONTACT_OVERLAP_TARGET_M - EPS, f"{side} gap={gap}"


def test_t14_dual_lr_and_all_exports() -> None:
    """T14: L/R length/hw equalize; __all__ already exports 0080 floor names."""
    pkg = _product_pkg()
    by_name = {p.name: p for p in pkg.parts}
    for attr in ("half_depth_m", "top_half_width_m", "ry_m"):
        left = getattr(by_name["RECIPE_foot_plate_l"], attr)
        right = getattr(by_name["RECIPE_foot_plate_r"], attr)
        assert float(left or 0.0) == pytest.approx(float(right or 0.0), abs=1e-12)
    from meshops.proportion import extremity_recipe as ext

    assert "FOOT_LEN_VISUAL_MIN_FRAC_H" in ext.__all__
    assert "FOOT_LEN_MIN_VS_CALF_DIAM" in ext.__all__
    assert "FOOT_LEN_VISUAL_MAX_FRAC_H" in ext.__all__
    assert "FOOT_HW_MIN_FRAC_LEN" in ext.__all__
    assert "FOOT_HW_MIN_VS_CALF_R" in ext.__all__
    assert "FOOT_HW_MIN_FRAC_H" in ext.__all__
