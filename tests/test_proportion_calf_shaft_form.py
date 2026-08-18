"""Track 0096 — calf shaft form (two-seg belly + distal shank).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 46 stay. Not mesh/print success.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    CALF_BELLY_LAT_FRAC,
    CALF_BELLY_REAR_FRAC,
    CALF_BELLY_SCALE,
    CALF_DIST_END_SCALE,
    CALF_DIST_SHAFT_SCALE,
    CALF_PROX_END_SCALE,
    CALF_SPLIT_T,
    HIP_SOFT_RX_SCALE,
    KNEE_SOFT_RZ_FRAC,
    RECIPE_SCHEMA_VERSION,
    THIGH_DIST_SHAFT_SCALE,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.constraints import classify_part_name, validate_constraints
from meshops.proportion.extremity_recipe import (
    FOOT_LEN_MIN_VS_CALF_DIAM,
    _calf_distal_r_from_parts,
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

_PRODUCT_NOFUSE = Path("work/rogue-v3/blockout/product_0096up/nofuse")
_MID_R = 0.0613
_CALF_HW = 0.04379
_EXPECT_CYL_R = 0.05167
_EXPECT_TAPER_R = 0.03853
_EXPECT_KNEE_RX = 0.04767
_EXPECT_KNEE_RZ = 0.05482


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


def _limb_mass_report(
    *,
    height_m: float = 1.72,
    thigh_hw: float = _MID_R,
    calf_hw: float = _CALF_HW,
    arm_hw: float = 0.04,
    include_feet_lms: bool = False,
    knee_y: float | None = 0.04,
) -> ProportionReport:
    lms = {
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=1.72),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=0.0, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=0.0, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=0.0, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=0.0, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=knee_y, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=knee_y, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
    }
    if include_feet_lms:
        lms["heel_l"] = _lm("heel_l", x_m=-0.10, y_m=0.06, z_m=0.02)
        lms["heel_r"] = _lm("heel_r", x_m=0.10, y_m=0.06, z_m=0.02)
        lms["toe_l"] = _lm("toe_l", x_m=-0.10, y_m=-0.12, z_m=0.02)
        lms["toe_r"] = _lm("toe_r", x_m=0.10, y_m=-0.12, z_m=0.02)
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=arm_hw),
        _diam("upper_arm_r", half_width_m=arm_hw),
        _diam("forearm_l", half_width_m=arm_hw),
        _diam("forearm_r", half_width_m=arm_hw),
        _diam("thigh_l", half_width_m=thigh_hw),
        _diam("thigh_r", half_width_m=thigh_hw),
        _diam("calf_l", half_width_m=calf_hw),
        _diam("calf_r", half_width_m=calf_hw),
    ]
    if include_feet_lms:
        diams.append(_diam("ank_foot_l", half_width_m=0.035))
        diams.append(_diam("ank_foot_r", half_width_m=0.035))
    return ProportionReport(
        schema_version="1.0.0",
        height_m=height_m,
        landmarks_xyz=lms,
        diameters=diams,
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
        _diam("ank_foot_l", half_width_m=0.0263),
        _diam("ank_foot_r", half_width_m=0.0263),
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


def test_t0_const_freezes() -> None:
    """T0: belly 1.18 / dist shaft 0.88 / split 0.42 / lean 0.30/0.42; hold ends."""
    assert CALF_BELLY_SCALE == 1.18
    assert CALF_DIST_SHAFT_SCALE == 0.88
    assert CALF_SPLIT_T == 0.42
    assert CALF_BELLY_LAT_FRAC == 0.30
    assert CALF_BELLY_REAR_FRAC == 0.42
    assert CALF_PROX_END_SCALE == 0.88
    assert CALF_DIST_END_SCALE == 0.72
    assert 1.14 <= CALF_BELLY_SCALE <= 1.22
    assert 0.82 <= CALF_DIST_SHAFT_SCALE <= 0.92
    assert 0.38 <= CALF_SPLIT_T <= 0.48
    assert CALF_DIST_SHAFT_SCALE < CALF_BELLY_SCALE
    assert CALF_DIST_END_SCALE < CALF_DIST_SHAFT_SCALE


def test_t1_taper_parts_present() -> None:
    """T1: both RECIPE_calf_taper_dist_{l,r} capsules; cyl stays; no limb_calf/gastroc."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        assert taper.kind == "capsule"
        assert cyl.kind == "capsule"
        assert f"RECIPE_limb_calf_{side}" not in by_name
    assert not any("limb_calf" in p.name.lower() for p in pkg.parts)
    assert not any("gastroc" in p.name.lower() for p in pkg.parts)
    assert not any("soleus" in p.name.lower() for p in pkg.parts)


def test_t2_product_like_radii() -> None:
    """T2: mid 0.04379 — cyl=1.18x / taper=0.88x / a=0.88x / b=0.72x."""
    mid = _CALF_HW
    report = _limb_mass_report(calf_hw=mid)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        a = by_name[f"RECIPE_calf_a_{side}"]
        b = by_name[f"RECIPE_calf_b_{side}"]
        assert float(cyl.radius_m) == pytest.approx(mid * CALF_BELLY_SCALE, abs=1e-9)  # type: ignore[arg-type]
        assert float(taper.radius_m) == pytest.approx(  # type: ignore[arg-type]
            mid * CALF_DIST_SHAFT_SCALE, abs=1e-9
        )
        assert float(a.rx_m) == pytest.approx(mid * CALF_PROX_END_SCALE, abs=1e-9)  # type: ignore[arg-type]
        assert float(b.rx_m) == pytest.approx(mid * CALF_DIST_END_SCALE, abs=1e-9)  # type: ignore[arg-type]


def test_t3_classifier_unknown_and_no_dup() -> None:
    """T3: calf_taper → unknown; cyl → calf; _a proximal; _b distal; C_no_dup pass."""
    assert classify_part_name("RECIPE_calf_taper_dist_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_calf_taper_dist_r") == ("unknown", "r")
    assert classify_part_name("RECIPE_calf_cyl_l") == ("calf", "l")
    assert classify_part_name("RECIPE_calf_a_l") == ("calf_proximal", "l")
    assert classify_part_name("RECIPE_calf_b_l") == ("calf_distal", "l")
    report = _limb_mass_report(include_feet_lms=True)
    pkg = build_blockout_recipe(report, limbs=True, feet=True)
    result = validate_constraints(pkg)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_no_dup_limb"].status == "pass", by_id["C_no_dup_limb"].message


def test_t4_split_lerp() -> None:
    """T4: cyl.p1 == lerp(offset p0, ankle, SPLIT); taper.p0 == cyl.p1; taper.p1 == b."""
    report = _limb_mass_report(calf_hw=_CALF_HW, knee_y=0.04)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    t = CALF_SPLIT_T
    for side in ("l", "r"):
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        b = by_name[f"RECIPE_calf_b_{side}"]
        assert cyl.p0 is not None and cyl.p1 is not None
        assert taper.p0 is not None and taper.p1 is not None
        assert b.center is not None
        mid = [
            float(cyl.p0[0]) + t * (float(b.center[0]) - float(cyl.p0[0])),
            float(cyl.p0[1]) + t * (float(b.center[1]) - float(cyl.p0[1])),
            float(cyl.p0[2]) + t * (float(b.center[2]) - float(cyl.p0[2])),
        ]
        assert float(cyl.p1[0]) == pytest.approx(mid[0], abs=1e-9)
        assert float(cyl.p1[1]) == pytest.approx(mid[1], abs=1e-9)
        assert float(cyl.p1[2]) == pytest.approx(mid[2], abs=1e-9)
        assert float(taper.p0[0]) == pytest.approx(float(cyl.p1[0]), abs=1e-9)
        assert float(taper.p0[1]) == pytest.approx(float(cyl.p1[1]), abs=1e-9)
        assert float(taper.p0[2]) == pytest.approx(float(cyl.p1[2]), abs=1e-9)
        assert float(taper.p1[0]) == pytest.approx(float(b.center[0]), abs=1e-6)
        assert float(taper.p1[1]) == pytest.approx(float(b.center[1]), abs=1e-6)
        assert float(taper.p1[2]) == pytest.approx(float(b.center[2]), abs=1e-6)


def test_t5_p0_only_lean() -> None:
    """T5: cyl.p0 lat/rear x cyl.r vs calf_a; taper.p1 has no extra lean vs calf_b."""
    report = _limb_mass_report(calf_hw=_CALF_HW, knee_y=0.04)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        a = by_name[f"RECIPE_calf_a_{side}"]
        b = by_name[f"RECIPE_calf_b_{side}"]
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        assert a.center is not None and b.center is not None
        assert cyl.p0 is not None and taper.p1 is not None and cyl.radius_m is not None
        sign = 1.0 if side == "r" else -1.0
        cyl_r = float(cyl.radius_m)
        dx = sign * CALF_BELLY_LAT_FRAC * cyl_r
        dy = CALF_BELLY_REAR_FRAC * cyl_r
        assert float(cyl.p0[0]) == pytest.approx(float(a.center[0]) + dx, abs=1e-6)
        assert float(cyl.p0[1]) == pytest.approx(float(a.center[1]) + dy, abs=1e-6)
        assert float(taper.p1[0]) == pytest.approx(float(b.center[0]), abs=1e-6)
        assert float(taper.p1[1]) == pytest.approx(float(b.center[1]), abs=1e-6)
        assert float(taper.p1[2]) == pytest.approx(float(b.center[2]), abs=1e-6)


def test_t6_sibling_after_both() -> None:
    """T6: both belly/taper lines + exactly one const-driven sibling after both."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    msgs = pkg.messages
    calf_l = [i for i, m in enumerate(msgs) if m.startswith("calf_l: belly/taper")]
    calf_r = [i for i, m in enumerate(msgs) if m.startswith("calf_r: belly/taper")]
    sib = [i for i, m in enumerate(msgs) if m.startswith("calf shaft form:")]
    assert len(calf_l) == 1
    assert len(calf_r) == 1
    assert len(sib) == 1
    line = msgs[sib[0]]
    assert f"belly={CALF_BELLY_SCALE}" in line
    assert f"dist_shaft={CALF_DIST_SHAFT_SCALE}" in line
    assert f"split={CALF_SPLIT_T}" in line
    assert f"lat={CALF_BELLY_LAT_FRAC}" in line
    assert f"rear={CALF_BELLY_REAR_FRAC}" in line
    assert sib[0] > calf_l[0]
    assert sib[0] > calf_r[0]


def test_t7_n_parts_schema_mcp() -> None:
    """T7: n_parts 131 via product flags; schema 1.4.0; MCP 46."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(),  # type: ignore[arg-type]
    )
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert len(TOOL_NAMES) == 46


def test_t8_product_path_constraints() -> None:
    """T8: product-path C_no_dup_limb + C_calf_slant + C_thigh_outer (if-file)."""
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
    assert by_id["C_no_dup_limb"].status == "pass", by_id["C_no_dup_limb"].message
    assert by_id["C_calf_slant"].status == "pass", by_id["C_calf_slant"].message
    cons_path = _PRODUCT_NOFUSE / "constraints_report.json"
    if cons_path.is_file():
        cons = json.loads(cons_path.read_text(encoding="utf-8"))
        rule_by = {r["id"]: r["status"] for r in cons.get("rules", [])}
        assert rule_by.get("C_thigh_outer") == "pass"
        assert rule_by.get("C_no_dup_limb") == "pass"
        assert rule_by.get("C_calf_slant") == "pass"


def test_t9_invert_not_0045_0071() -> None:
    """T9: invert — no longer 0045 1.08 / 0071 0.22/0.28 one-cyl pipe."""
    assert CALF_BELLY_SCALE != 1.08
    assert CALF_BELLY_LAT_FRAC != 0.22
    assert CALF_BELLY_REAR_FRAC != 0.28
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    assert any(p.name.startswith("RECIPE_calf_taper_dist_") for p in pkg.parts)


def test_t10_product_like_meters() -> None:
    """T10 B20: product-like cyl ≈ 0.05167; taper ≈ 0.03853 (not old 0.04729 pipe)."""
    report = _limb_mass_report(calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        assert float(cyl.radius_m) == pytest.approx(_EXPECT_CYL_R, abs=2e-4)  # type: ignore[arg-type]
        assert float(taper.radius_m) == pytest.approx(_EXPECT_TAPER_R, abs=2e-4)  # type: ignore[arg-type]
        assert float(cyl.radius_m) != pytest.approx(0.04729, abs=2e-4)  # type: ignore[arg-type]


def test_t11_fence_knee_thigh_hip_elbow_foot() -> None:
    """T11: 0095 knee / 0094 thigh / hip_soft / elbow / 0080 calf_b foot_len hold."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(),  # type: ignore[arg-type]
    )
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        hip = by_name[f"RECIPE_hip_soft_{side}"]
        elbow = by_name[f"RECIPE_elbow_soft_{side}"]
        calf_b = by_name[f"RECIPE_calf_b_{side}"]
        assert float(knee.rx_m) == pytest.approx(_EXPECT_KNEE_RX, abs=2e-4)  # type: ignore[arg-type]
        assert float(knee.rz_m) == pytest.approx(_EXPECT_KNEE_RZ, abs=2e-4)  # type: ignore[arg-type]
        assert float(knee.rz_m) == pytest.approx(  # type: ignore[arg-type]
            float(knee.rx_m) * KNEE_SOFT_RZ_FRAC,  # type: ignore[arg-type]
            abs=1e-4,
        )
        assert float(dist.radius_m) == pytest.approx(0.04414, abs=1e-4)  # type: ignore[arg-type]
        assert THIGH_DIST_SHAFT_SCALE == 0.72
        assert float(hip.rx_m) == pytest.approx(0.0705, abs=1e-4)  # type: ignore[arg-type]
        assert float(hip.rx_m) == pytest.approx(  # type: ignore[arg-type]
            _MID_R * HIP_SOFT_RX_SCALE, abs=1e-4
        )
        assert float(elbow.rx_m) == pytest.approx(0.0470, abs=1e-4)  # type: ignore[arg-type]
        assert float(calf_b.rx_m) == pytest.approx(_CALF_HW * CALF_DIST_END_SCALE, abs=1e-4)  # type: ignore[arg-type]
        from_parts = _calf_distal_r_from_parts(pkg.parts, side)
        assert from_parts == pytest.approx(float(calf_b.rx_m), abs=1e-9)  # type: ignore[arg-type]
        expect_len = FOOT_LEN_MIN_VS_CALF_DIAM * (2.0 * float(calf_b.rx_m))  # type: ignore[arg-type]
        plate = by_name[f"RECIPE_foot_plate_{side}"]
        assert plate.half_depth_m is not None
        assert (2.0 * float(plate.half_depth_m)) == pytest.approx(expect_len, abs=1e-3)
    assert any("calf_diam" in m for m in pkg.messages)


def test_t12_all_exports_new_consts() -> None:
    """T12: __all__ exports belly / dist shaft / split / lat / rear."""
    from meshops.proportion import blockout_recipe as br

    assert "CALF_BELLY_SCALE" in br.__all__
    assert "CALF_DIST_SHAFT_SCALE" in br.__all__
    assert "CALF_SPLIT_T" in br.__all__
    assert "CALF_BELLY_LAT_FRAC" in br.__all__
    assert "CALF_BELLY_REAR_FRAC" in br.__all__


def test_t13_b6_writes_taper_not_cyl() -> None:
    """T13: B6 + feet — b.y == ank.y == taper.p1[1]; cyl.p1 is mid; p0 rear intact."""
    report = _limb_mass_report(include_feet_lms=True, calf_hw=_CALF_HW, knee_y=0.04)
    pkg = build_blockout_recipe(report, limbs=True, feet=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        ank = by_name[f"RECIPE_ank_foot_{side}"]
        b = by_name[f"RECIPE_calf_b_{side}"]
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        a = by_name[f"RECIPE_calf_a_{side}"]
        assert ank.center is not None and b.center is not None
        assert cyl.p0 is not None and cyl.p1 is not None
        assert taper.p1 is not None and a.center is not None and cyl.radius_m is not None
        ay = float(ank.center[1])
        assert float(b.center[1]) == pytest.approx(ay, abs=1e-6)
        assert float(taper.p1[1]) == pytest.approx(ay, abs=1e-6)
        assert abs(float(cyl.p1[1]) - ay) > 1e-3
        dy = CALF_BELLY_REAR_FRAC * float(cyl.radius_m)
        assert float(cyl.p0[1]) == pytest.approx(float(a.center[1]) + dy, abs=1e-6)
        assert any(f"calf_{side}: distal/taper p1 Y synced to ank_foot" in m for m in pkg.messages)


def test_t14_adduction_recomputes_mid() -> None:
    """T14: 10° adduction moves cyl.p0; mid recomputed; taper.p1 XZ stays ankle-class."""
    report = _limb_mass_report(calf_hw=_CALF_HW, knee_y=0.04)
    pkg0 = build_blockout_recipe(report, limbs=True)
    pkg = build_blockout_recipe(
        report,
        limbs=True,
        template_applied=_template(thigh_tilt_deg=10.0),
    )
    by0 = {p.name: p for p in pkg0.parts}
    by_name = {p.name: p for p in pkg.parts}
    t = CALF_SPLIT_T
    for side in ("l", "r"):
        cyl0 = by0[f"RECIPE_calf_cyl_{side}"]
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        taper0 = by0[f"RECIPE_calf_taper_dist_{side}"]
        ankle = report.landmarks_xyz[f"ankle_{side}"]
        assert cyl.p0 is not None and cyl.p1 is not None
        assert cyl0.p0 is not None and taper.p0 is not None and taper.p1 is not None
        assert taper0.p1 is not None
        assert abs(float(cyl.p0[0]) - float(cyl0.p0[0])) > 1e-4
        mid = [
            float(cyl.p0[0]) + t * (float(taper.p1[0]) - float(cyl.p0[0])),
            float(cyl.p0[1]) + t * (float(taper.p1[1]) - float(cyl.p0[1])),
            float(cyl.p0[2]) + t * (float(taper.p1[2]) - float(cyl.p0[2])),
        ]
        assert float(cyl.p1[0]) == pytest.approx(mid[0], abs=1e-9)
        assert float(cyl.p1[1]) == pytest.approx(mid[1], abs=1e-9)
        assert float(cyl.p1[2]) == pytest.approx(mid[2], abs=1e-9)
        assert float(taper.p0[0]) == pytest.approx(float(cyl.p1[0]), abs=1e-9)
        assert float(taper.p0[1]) == pytest.approx(float(cyl.p1[1]), abs=1e-9)
        assert float(taper.p1[0]) == pytest.approx(float(taper0.p1[0]), abs=1e-6)
        assert float(taper.p1[2]) == pytest.approx(float(taper0.p1[2]), abs=1e-6)
        assert float(taper.p1[0]) == pytest.approx(float(ankle.x_m), abs=1e-4)  # type: ignore[arg-type]
        assert float(taper.p1[2]) == pytest.approx(float(ankle.z_m), abs=1e-4)  # type: ignore[arg-type]


def test_t15_dual_lr_and_join_ready_order() -> None:
    """T15: dual L/R calf radii equalize; join_ready b < taper < cyl."""
    report = _limb_mass_report(include_feet_lms=True, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True, feet=True, join_ready=True)
    by_name = {p.name: p for p in pkg.parts}
    left = by_name["RECIPE_calf_cyl_l"]
    right = by_name["RECIPE_calf_cyl_r"]
    assert float(left.radius_m) == pytest.approx(float(right.radius_m), abs=1e-12)  # type: ignore[arg-type]
    tl = by_name["RECIPE_calf_taper_dist_l"]
    tr = by_name["RECIPE_calf_taper_dist_r"]
    assert float(tl.radius_m) == pytest.approx(float(tr.radius_m), abs=1e-12)  # type: ignore[arg-type]
    for side in ("l", "r"):
        b = by_name[f"RECIPE_calf_b_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        assert float(b.rx_m) < float(taper.radius_m) < float(cyl.radius_m)  # type: ignore[arg-type]
