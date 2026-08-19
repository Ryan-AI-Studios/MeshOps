"""Track 0094 — thigh distal taper plus (dist scale 0.80 → 0.72).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 46 stay. Not mesh/print success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    _LIMB_DIST_SOFT_BANDS,
    CALF_PROX_END_SCALE,
    HIP_SOFT_RX_SCALE,
    KNEE_SOFT_FRAC,
    RECIPE_SCHEMA_VERSION,
    THIGH_ADDUCTION_MAX_MEDIAL_M,
    THIGH_DIST_SHAFT_SCALE,
    THIGH_PROX_SHAFT_SCALE,
    THIGH_SPLIT_T,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.connection_metrics import _hip_pair
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

_PRODUCT_NOFUSE = Path("work/rogue-v3/blockout/product_0094up/nofuse")
_MID_R = 0.0613
_CALF_HW = 0.0438


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


def _part(
    name: str,
    *,
    kind: str = "capsule",
    role: str = "limb_segment",
    center: list[float] | None = None,
    rx_m: float | None = None,
    ry_m: float | None = None,
    rz_m: float | None = None,
    radius_m: float | None = None,
    p0: list[float] | None = None,
    p1: list[float] | None = None,
) -> Any:
    from meshops.proportion.blockout_recipe import RecipePart

    kwargs: dict[str, Any] = {
        "name": name,
        "role": role,
        "kind": kind,
        "center": center,
        "rx_m": rx_m,
        "ry_m": ry_m,
        "rz_m": rz_m,
        "radius_m": radius_m,
        "p0": p0,
        "p1": p1,
    }
    clean = {k: v for k, v in kwargs.items() if v is not None or k in ("name", "role", "kind")}
    return RecipePart.model_validate(clean)


def test_t0_const_freezes() -> None:
    """T0: dist 0.72 / prox 1.00 / split 0.50 / band 0.68-0.74; no thigh dist_soft."""
    assert THIGH_DIST_SHAFT_SCALE == 0.72
    assert THIGH_PROX_SHAFT_SCALE == 1.00
    assert THIGH_SPLIT_T == 0.50
    assert 0.68 <= THIGH_DIST_SHAFT_SCALE <= 0.74
    assert THIGH_DIST_SHAFT_SCALE < THIGH_PROX_SHAFT_SCALE
    assert "thigh_l" not in _LIMB_DIST_SOFT_BANDS
    assert "thigh_r" not in _LIMB_DIST_SOFT_BANDS


def test_t1_both_segments_still_present() -> None:
    """T1: both RECIPE_limb_thigh_* + RECIPE_thigh_taper_dist_* (l+r); dist < prox."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        assert prox.kind == "capsule"
        assert dist.kind == "capsule"
        assert prox.radius_m is not None
        assert dist.radius_m is not None
        assert float(dist.radius_m) < float(prox.radius_m)


def test_t2_product_like_ratio() -> None:
    """T2: product-like mid 0.0613: dist == 0.0613*0.72; ratio == 0.72."""
    report = _limb_mass_report(thigh_hw=_MID_R)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        assert float(dist.radius_m) == pytest.approx(  # type: ignore[arg-type]
            _MID_R * THIGH_DIST_SHAFT_SCALE, abs=1e-9
        )
        assert float(dist.radius_m) == pytest.approx(_MID_R * 0.72, abs=1e-9)  # type: ignore[arg-type]
        ratio = float(dist.radius_m) / float(prox.radius_m)  # type: ignore[arg-type]
        assert ratio == pytest.approx(0.72, abs=1e-6)


def test_t3_classifier_unchanged() -> None:
    """T3: thigh_taper_dist → unknown; limb_thigh → thigh."""
    assert classify_part_name("RECIPE_thigh_taper_dist_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_thigh_taper_dist_r") == ("unknown", "r")
    assert classify_part_name("RECIPE_limb_thigh_l") == ("thigh", "l")
    assert classify_part_name("RECIPE_limb_thigh_r") == ("thigh", "r")


def test_t4_no_dist_soft_thigh() -> None:
    """T4: no dist_soft thigh names (0045 B13)."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    soft_names = [p.name for p in pkg.parts if "dist_soft" in p.name.lower()]
    assert not any("thigh" in n for n in soft_names)
    by_name = {p.name: p for p in pkg.parts}
    assert "RECIPE_dist_soft_thigh_l" not in by_name
    assert "RECIPE_dist_soft_thigh_r" not in by_name


def test_t5_adduction_holds() -> None:
    """T5: adduction still 10 deg / medial 0.030 on product-like."""
    assert THIGH_ADDUCTION_MAX_MEDIAL_M == 0.030
    report = _limb_mass_report()
    pkg = build_blockout_recipe(
        report,
        limbs=True,
        template_applied=_template(thigh_tilt_deg=10.0),  # type: ignore[arg-type]
    )
    assert any("adduction_tilt_deg=10.0" in m for m in pkg.messages)
    assert any("medial_shift_m=" in m for m in pkg.messages)
    for m in pkg.messages:
        if "medial_shift_m=" not in m:
            continue
        token = m.split("medial_shift_m=", 1)[1].split()[0]
        assert float(token) <= THIGH_ADDUCTION_MAX_MEDIAL_M + 1e-6


def test_t6_split_mid_still_half() -> None:
    """T6: split mid still t=0.50 (p0 + 0.5*(p1-p0))."""
    assert THIGH_SPLIT_T == 0.50
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        assert prox.p0 is not None and prox.p1 is not None
        assert dist.p0 is not None and dist.p1 is not None
        for i in range(3):
            expected = float(prox.p0[i]) + 0.5 * (float(dist.p1[i]) - float(prox.p0[i]))
            assert float(prox.p1[i]) == pytest.approx(expected, abs=1e-6)
            assert float(dist.p0[i]) == pytest.approx(float(prox.p1[i]), abs=1e-6)


def test_t7_sibling_after_both_shaft_taper() -> None:
    """T7: both shaft_taper lines + exactly one sibling after both sides."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    msgs = pkg.messages
    shaft_l = [i for i, m in enumerate(msgs) if m.startswith("thigh_l: shaft_taper")]
    shaft_r = [i for i, m in enumerate(msgs) if m.startswith("thigh_r: shaft_taper")]
    sib = [i for i, m in enumerate(msgs) if m.startswith("thigh distal taper plus:")]
    assert len(shaft_l) == 1
    assert len(shaft_r) == 1
    assert len(sib) == 1
    assert "dist_scale=" in msgs[sib[0]]
    assert f"dist_scale={THIGH_DIST_SHAFT_SCALE}" in msgs[sib[0]]
    assert sib[0] > shaft_l[0]
    assert sib[0] > shaft_r[0]


def test_t8_n_parts_schema_mcp() -> None:
    """T8: n_parts 131 via product flags; schema 1.4.0; MCP 46."""
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
    assert len(TOOL_NAMES) == 47


def test_t9_product_path_constraints() -> None:
    """T9: product-path C_thigh_outer + C_no_dup_limb pass (0093 T9 flags)."""
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
    cons_path = _PRODUCT_NOFUSE / "constraints_report.json"
    if cons_path.is_file():
        cons = json.loads(cons_path.read_text(encoding="utf-8"))
        rule_by = {r["id"]: r["status"] for r in cons.get("rules", [])}
        assert rule_by.get("C_thigh_outer") == "pass"
        assert rule_by.get("C_no_dup_limb") == "pass"


def test_t10_knee_cascade() -> None:
    """T10 B20: product-like knee.rx == dist_r * KNEE_SOFT_FRAC (expect ~0.04767)."""
    report = _limb_mass_report(thigh_hw=_MID_R, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        dist_r = float(dist.radius_m)  # type: ignore[arg-type]
        assert knee.rx_m is not None
        assert float(knee.rx_m) == pytest.approx(dist_r * KNEE_SOFT_FRAC, abs=1e-5)
        assert float(knee.rx_m) == pytest.approx(0.04767, abs=2e-4)


def test_t11_calf_and_hip_soft_hold() -> None:
    """T11: calf_a rx / hip_soft rx hold vs 0093up (±1e-4)."""
    report = _limb_mass_report(thigh_hw=_MID_R, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        hip = by_name[f"RECIPE_hip_soft_{side}"]
        calf_a = by_name[f"RECIPE_calf_a_{side}"]
        assert hip.rx_m is not None and calf_a.rx_m is not None
        assert float(hip.rx_m) == pytest.approx(_MID_R * HIP_SOFT_RX_SCALE, abs=1e-4)
        assert float(hip.rx_m) == pytest.approx(0.0705, abs=1e-4)
        assert float(calf_a.rx_m) == pytest.approx(_CALF_HW * CALF_PROX_END_SCALE, abs=1e-4)
        assert float(calf_a.rx_m) == pytest.approx(0.0385, abs=1e-4)


def test_t12_all_still_exports_dist_scale() -> None:
    """T12: __all__ still exports THIGH_DIST_SHAFT_SCALE."""
    from meshops.proportion import blockout_recipe as br

    assert "THIGH_DIST_SHAFT_SCALE" in br.__all__


def test_t13_hip_pair_excludes_taper() -> None:
    """T13: 0070 B10 hip_pair fallback still excludes thigh_taper."""
    thigh = _part(
        "RECIPE_custom_thigh_l",
        radius_m=0.06,
        p0=[0.12, 0.0, 0.95],
        p1=[0.12, 0.0, 0.725],
    )
    decoy = _part(
        "RECIPE_thigh_taper_dist_l",
        radius_m=0.044,
        p0=[0.12, 0.0, 0.725],
        p1=[0.12, 0.0, 0.50],
    )
    pelvis = _part(
        "RECIPE_pelvis_oval",
        role="pelvis",
        kind="ellipsoid",
        center=[0.0, 0.0, 0.90],
        rx_m=0.12,
        ry_m=0.08,
        rz_m=0.06,
    )
    parts = [decoy, thigh, pelvis]
    by_name = {p.name: p for p in parts}
    pair = _hip_pair(parts, by_name, "l")
    assert pair is not None
    child, parent = pair
    assert child.name == "RECIPE_custom_thigh_l"
    assert "thigh_taper" not in child.name
    assert parent.name == "RECIPE_pelvis_oval"


def test_t14_invert_not_080() -> None:
    """T14: invert — THIGH_DIST_SHAFT_SCALE is no longer 0.80."""
    assert THIGH_DIST_SHAFT_SCALE != 0.80
    assert THIGH_DIST_SHAFT_SCALE == 0.72


def test_t15_dual_lr_dist_equalize() -> None:
    """T15: dual L/R dist radii equalize."""
    report = _limb_mass_report(thigh_hw=_MID_R)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    left = float(by_name["RECIPE_thigh_taper_dist_l"].radius_m)  # type: ignore[arg-type]
    right = float(by_name["RECIPE_thigh_taper_dist_r"].radius_m)  # type: ignore[arg-type]
    assert left == pytest.approx(right, abs=1e-12)
    assert left == pytest.approx(_MID_R * THIGH_DIST_SHAFT_SCALE, abs=1e-9)
