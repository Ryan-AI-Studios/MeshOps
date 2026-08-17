"""Track 0095 — knee bead soften (1.18/0.90/0.95 → 1.08/0.82/1.15 tall sleeve).

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
    CALF_PROX_END_SCALE,
    HIP_SOFT_RX_SCALE,
    KNEE_SOFT_FRAC,
    KNEE_SOFT_MAX_VS_THIGH_PROX,
    KNEE_SOFT_MIN_FRAC_H,
    KNEE_SOFT_OUTER_FRAC_RX,
    KNEE_SOFT_REAR_FRAC_RY,
    KNEE_SOFT_RY_FRAC,
    KNEE_SOFT_RZ_FRAC,
    RECIPE_SCHEMA_VERSION,
    THIGH_DIST_SHAFT_SCALE,
    THIGH_PROX_SHAFT_SCALE,
    THIGH_SPLIT_T,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.constraints import classify_part_name, validate_constraints
from meshops.proportion.models import (
    CrossSection,
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import build_blockout_skeleton

_PRODUCT_NOFUSE = Path("work/rogue-v3/blockout/product_0095up/nofuse")
_MID_R = 0.0613
_CALF_HW = 0.0438
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
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
    }
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
    """T0: 1.08 / 0.82 / 1.15 + bands; hold floors; 0094 dist 0.72."""
    assert KNEE_SOFT_FRAC == 1.08
    assert KNEE_SOFT_RY_FRAC == 0.82
    assert KNEE_SOFT_RZ_FRAC == 1.15
    assert 1.05 <= KNEE_SOFT_FRAC <= 1.12
    assert 1.10 <= KNEE_SOFT_RZ_FRAC <= 1.22
    assert KNEE_SOFT_RZ_FRAC > 1.0
    assert KNEE_SOFT_RY_FRAC < 0.90
    assert KNEE_SOFT_MIN_FRAC_H == 0.018
    assert KNEE_SOFT_OUTER_FRAC_RX == 0.06
    assert KNEE_SOFT_REAR_FRAC_RY == 0.10
    assert KNEE_SOFT_MAX_VS_THIGH_PROX == 1.25
    assert THIGH_DIST_SHAFT_SCALE == 0.72


def test_t1_both_knees_still_present() -> None:
    """T1: both RECIPE_knee_soft_{l,r}; ellipsoid; no condyle / joint_sleeve."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        assert knee.kind == "ellipsoid"
    names = [p.name.lower() for p in pkg.parts]
    assert not any("condyle" in n or "joint_sleeve" in n for n in names)


def test_t2_product_like_aniso() -> None:
    """T2: product-like dist 0.04414: rx == dist*1.08; rz/rx == 1.15; ry/rx == 0.82."""
    report = _limb_mass_report(thigh_hw=_MID_R, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        dist_r = float(dist.radius_m)  # type: ignore[arg-type]
        assert dist_r == pytest.approx(0.04414, abs=2e-5)
        assert knee.rx_m is not None and knee.ry_m is not None and knee.rz_m is not None
        rx, ry, rz = float(knee.rx_m), float(knee.ry_m), float(knee.rz_m)
        assert rx == pytest.approx(dist_r * KNEE_SOFT_FRAC, abs=1e-9)
        assert rz / rx == pytest.approx(KNEE_SOFT_RZ_FRAC, abs=1e-9)
        assert ry / rx == pytest.approx(KNEE_SOFT_RY_FRAC, abs=1e-9)


def test_t3_classifier_unchanged() -> None:
    """T3: knee_soft → unknown; C_no_dup_limb still classifies."""
    assert classify_part_name("RECIPE_knee_soft_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_knee_soft_r") == ("unknown", "r")
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
    assert "C_no_dup_limb" in by_id


def test_t4_seam_still_max_dist_calf_a() -> None:
    """T4: seam still max(dist, calf_a); knee.rx == FRAC * seam (not * prox)."""
    report = _limb_mass_report(thigh_hw=_MID_R, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        calf_a = by_name[f"RECIPE_calf_a_{side}"]
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        seam = max(float(dist.radius_m), float(calf_a.rx_m))  # type: ignore[arg-type]
        assert knee.rx_m is not None
        assert float(knee.rx_m) == pytest.approx(KNEE_SOFT_FRAC * seam, abs=1e-9)
        assert float(knee.rx_m) != pytest.approx(
            KNEE_SOFT_FRAC * float(prox.radius_m),  # type: ignore[arg-type]
            abs=1e-4,
        )


def test_t5_clamp_base_then_aniso() -> None:
    """T5: clamp-base-then-aniso — rx <= 1.25x prox; ry/rz from clamped base."""
    report = _limb_mass_report(thigh_hw=0.04, calf_hw=0.08)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    height_m = 1.72
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        calf_a = by_name[f"RECIPE_calf_a_{side}"]
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        seam = max(float(dist.radius_m), float(calf_a.rx_m))  # type: ignore[arg-type]
        base = max(KNEE_SOFT_FRAC * seam, KNEE_SOFT_MIN_FRAC_H * height_m)
        cap = KNEE_SOFT_MAX_VS_THIGH_PROX * float(prox.radius_m)  # type: ignore[arg-type]
        assert base > cap
        base = min(base, cap)
        assert knee.rx_m is not None and knee.ry_m is not None and knee.rz_m is not None
        assert float(knee.rx_m) == pytest.approx(base, abs=1e-9)
        assert float(knee.rx_m) <= cap + 1e-12
        assert float(knee.ry_m) == pytest.approx(base * KNEE_SOFT_RY_FRAC, abs=1e-9)
        assert float(knee.rz_m) == pytest.approx(base * KNEE_SOFT_RZ_FRAC, abs=1e-9)


def test_t6_sibling_after_both_knees() -> None:
    """T6: both knee_soft_ lines + exactly one const-driven sibling after both."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    msgs = pkg.messages
    knee_l = [i for i, m in enumerate(msgs) if m.startswith("knee_soft_l: rx=")]
    knee_r = [i for i, m in enumerate(msgs) if m.startswith("knee_soft_r: rx=")]
    sib = [i for i, m in enumerate(msgs) if m.startswith("knee bead soften:")]
    assert len(knee_l) == 1
    assert len(knee_r) == 1
    assert len(sib) == 1
    line = msgs[sib[0]]
    assert f"frac={KNEE_SOFT_FRAC}" in line
    assert f"ry={KNEE_SOFT_RY_FRAC}" in line
    assert f"rz={KNEE_SOFT_RZ_FRAC}" in line
    assert sib[0] > knee_l[0]
    assert sib[0] > knee_r[0]


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
    """T8: product-path C_no_dup_limb + C_calf_slant; C_thigh_outer if-file."""
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
    assert "C_no_dup_limb" in by_id
    assert by_id["C_no_dup_limb"].status == "pass", by_id["C_no_dup_limb"].message
    assert "C_calf_slant" in by_id
    assert by_id["C_calf_slant"].status == "pass", by_id["C_calf_slant"].message
    cons_path = _PRODUCT_NOFUSE / "constraints_report.json"
    if cons_path.is_file():
        cons = json.loads(cons_path.read_text(encoding="utf-8"))
        rule_by = {r["id"]: r["status"] for r in cons.get("rules", [])}
        assert rule_by.get("C_thigh_outer") == "pass"
        assert rule_by.get("C_no_dup_limb") == "pass"
        assert rule_by.get("C_calf_slant") == "pass"


def test_t9_invert_not_0081() -> None:
    """T9: invert — no longer 0081 1.18 / 0.90 / 0.95."""
    assert KNEE_SOFT_FRAC != 1.18
    assert KNEE_SOFT_RZ_FRAC != 0.95
    assert KNEE_SOFT_RY_FRAC != 0.90


def test_t10_product_like_meters() -> None:
    """T10 B20: product-like knee.rx ≈ 0.04767; rz ≈ 0.05482 (not 0.05208)."""
    report = _limb_mass_report(thigh_hw=_MID_R, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        assert knee.rx_m is not None and knee.rz_m is not None
        assert float(knee.rx_m) == pytest.approx(_EXPECT_KNEE_RX, abs=2e-4)
        assert float(knee.rz_m) == pytest.approx(_EXPECT_KNEE_RZ, abs=2e-4)
        assert float(knee.rx_m) != pytest.approx(0.05208, abs=2e-4)


def test_t11_fence_thigh_calf_hip_elbow() -> None:
    """T11: thigh / calf_a / hip_soft / elbow hold vs 0094up (±1e-4)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(),  # type: ignore[arg-type]
    )
    by_name = {p.name: p for p in pkg.parts}
    assert THIGH_PROX_SHAFT_SCALE == 1.00
    assert THIGH_DIST_SHAFT_SCALE == 0.72
    assert THIGH_SPLIT_T == 0.50
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        calf_a = by_name[f"RECIPE_calf_a_{side}"]
        hip = by_name[f"RECIPE_hip_soft_{side}"]
        elbow = by_name[f"RECIPE_elbow_soft_{side}"]
        assert float(prox.radius_m) == pytest.approx(0.0613, abs=1e-4)  # type: ignore[arg-type]
        assert float(dist.radius_m) == pytest.approx(0.0441, abs=1e-4)  # type: ignore[arg-type]
        ratio = float(dist.radius_m) / float(prox.radius_m)  # type: ignore[arg-type]
        assert ratio == pytest.approx(0.72, abs=1e-6)
        assert float(calf_a.rx_m) == pytest.approx(0.0385, abs=1e-4)  # type: ignore[arg-type]
        assert float(hip.rx_m) == pytest.approx(0.0705, abs=1e-4)  # type: ignore[arg-type]
        assert float(elbow.rx_m) == pytest.approx(0.0470, abs=1e-4)  # type: ignore[arg-type]
        assert float(hip.rx_m) == pytest.approx(  # type: ignore[arg-type]
            float(prox.radius_m) * HIP_SOFT_RX_SCALE,  # type: ignore[arg-type]
            abs=1e-4,
        )
        assert float(calf_a.rx_m) == pytest.approx(  # type: ignore[arg-type]
            _CALF_HW * CALF_PROX_END_SCALE, abs=1e-4
        )


def test_t12_all_still_exports_knee_consts() -> None:
    """T12: __all__ still exports KNEE_SOFT_FRAC / RY / RZ."""
    from meshops.proportion import blockout_recipe as br

    assert "KNEE_SOFT_FRAC" in br.__all__
    assert "KNEE_SOFT_RY_FRAC" in br.__all__
    assert "KNEE_SOFT_RZ_FRAC" in br.__all__


def test_t13_tall_sleeve_law() -> None:
    """T13: tall-sleeve law rz ≥ rx ≥ ry on product-like emit."""
    report = _limb_mass_report(thigh_hw=_MID_R, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        assert knee.rx_m is not None and knee.ry_m is not None and knee.rz_m is not None
        rx, ry, rz = float(knee.rx_m), float(knee.ry_m), float(knee.rz_m)
        assert rz >= rx - 1e-12
        assert rx >= ry - 1e-12


def test_t14_knee_owns_step() -> None:
    """T14: knee still > dist and > calf_a (owns the step)."""
    report = _limb_mass_report(thigh_hw=_MID_R, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        calf_a = by_name[f"RECIPE_calf_a_{side}"]
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        assert knee.rx_m is not None
        assert float(knee.rx_m) > float(dist.radius_m)  # type: ignore[arg-type]
        assert float(knee.rx_m) > float(calf_a.rx_m)  # type: ignore[arg-type]


def test_t15_dual_lr_knee_equalize() -> None:
    """T15: dual L/R knee radii equalize."""
    report = _limb_mass_report(thigh_hw=_MID_R, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    left = by_name["RECIPE_knee_soft_l"]
    right = by_name["RECIPE_knee_soft_r"]
    assert float(left.rx_m) == pytest.approx(float(right.rx_m), abs=1e-12)  # type: ignore[arg-type]
    assert float(left.ry_m) == pytest.approx(float(right.ry_m), abs=1e-12)  # type: ignore[arg-type]
    assert float(left.rz_m) == pytest.approx(float(right.rz_m), abs=1e-12)  # type: ignore[arg-type]
