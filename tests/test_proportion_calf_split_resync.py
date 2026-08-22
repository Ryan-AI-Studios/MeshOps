"""Track 0100 - calf split resync after B6 ankle Y (mid + cyl.placement).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Not mesh/print success. Schema 1.4.0 / MCP 46 stay.

Helpers copied from test_proportion_calf_shaft_form.py (per-module; do not import).
"""

from __future__ import annotations

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
    RECIPE_SCHEMA_VERSION,
    RecipePart,
    _sync_calf_distal_to_ankle,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.models import (
    CrossSection,
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import build_blockout_skeleton

_MID_R = 0.0613
_CALF_HW = 0.04379


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


def _lerp(p0: list[float], dest: list[float], t: float) -> list[float]:
    return [
        float(p0[0]) + t * (dest[0] - float(p0[0])),
        float(p0[1]) + t * (dest[1] - float(p0[1])),
        float(p0[2]) + t * (dest[2] - float(p0[2])),
    ]


def _t1_parts(*, side: str = "l") -> list[RecipePart]:
    """Bugbot path: emit mid from old ankle Y; ank.center Y ≠ emit ankle Y.

    XYZ (plan T1 table): cyl.p0 (0.1, 0.00, 0.50); emit ankle / taper.p1
    (0.1, 0.00, 0.10); emit mid T=0.42 (0.1, 0.00, 0.332); ank.center
    (0.1, 0.08, 0.10); post-fix mid Y = 0.00 + 0.42*0.08 = 0.0336.
    """
    sx = 0.1 if side == "r" else -0.1
    emit_mid = [sx, 0.00, 0.332]
    return [
        RecipePart(
            name=f"RECIPE_calf_cyl_{side}",
            role="limb_segment",
            kind="capsule",
            p0=[sx, 0.00, 0.50],
            p1=list(emit_mid),
            radius_m=0.04,
            placement="front_plane",
        ),
        RecipePart(
            name=f"RECIPE_calf_taper_dist_{side}",
            role="limb_segment",
            kind="capsule",
            p0=list(emit_mid),
            p1=[sx, 0.00, 0.10],
            radius_m=0.035,
            placement="front_plane",
        ),
        RecipePart(
            name=f"RECIPE_calf_b_{side}",
            role="limb_segment",
            kind="ellipsoid",
            center=[sx, 0.00, 0.10],
            rx_m=0.03,
            ry_m=0.03,
            rz_m=0.03,
            placement="front_plane",
        ),
        RecipePart(
            name=f"RECIPE_ank_foot_{side}",
            role="ankle_bridge",
            kind="ellipsoid",
            center=[sx, 0.08, 0.10],
            rx_m=0.03,
            ry_m=0.03,
            rz_m=0.03,
            placement="full3d",
        ),
    ]


def test_t0_const_freezes() -> None:
    """T0: split 0.42 / belly 1.18 / dist 0.80 / lat 0.30 / rear 0.42 / a 0.88 / b 0.72."""
    assert CALF_SPLIT_T == 0.42
    assert CALF_BELLY_SCALE == 1.18
    assert CALF_DIST_SHAFT_SCALE == 0.80
    assert CALF_BELLY_LAT_FRAC == 0.30
    assert CALF_BELLY_REAR_FRAC == 0.42
    assert CALF_PROX_END_SCALE == 0.88
    assert CALF_DIST_END_SCALE == 0.72
    assert 0.38 <= CALF_SPLIT_T <= 0.48
    assert CALF_DIST_SHAFT_SCALE < CALF_BELLY_SCALE
    assert CALF_DIST_END_SCALE < CALF_DIST_SHAFT_SCALE


def test_t1_helper_resyncs_mid_after_b6() -> None:
    """T1: direct B6 - mid = lerp(cyl.p0, new taper.p1, T); p0 unchanged.

    Fails on live L3090-3094 (taper.p1 Y only; emit mid Y stays 0.00).
    Post-fix mid Y = 0.0336; Z stays 0.332. Do not mix ankle Z as Y.
    """
    for side in ("l", "r"):
        parts = _t1_parts(side=side)
        by0 = {p.name: p for p in parts}
        cyl0 = by0[f"RECIPE_calf_cyl_{side}"]
        assert cyl0.p0 is not None and cyl0.p1 is not None
        p0_emit = [float(cyl0.p0[0]), float(cyl0.p0[1]), float(cyl0.p0[2])]
        messages: list[str] = []
        _sync_calf_distal_to_ankle(parts, messages)
        by_name = {p.name: p for p in parts}
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        b = by_name[f"RECIPE_calf_b_{side}"]
        ank = by_name[f"RECIPE_ank_foot_{side}"]
        assert cyl.p0 is not None and cyl.p1 is not None
        assert taper.p0 is not None and taper.p1 is not None
        assert b.center is not None and ank.center is not None
        ay = float(ank.center[1])
        assert ay == pytest.approx(0.08, abs=1e-12)
        assert float(taper.p1[1]) == pytest.approx(ay, abs=1e-9)
        assert float(b.center[1]) == pytest.approx(ay, abs=1e-9)
        dest = [float(taper.p1[0]), float(taper.p1[1]), float(taper.p1[2])]
        mid = _lerp(list(cyl.p0), dest, CALF_SPLIT_T)
        assert float(cyl.p1[0]) == pytest.approx(mid[0], abs=1e-9)
        assert float(cyl.p1[1]) == pytest.approx(mid[1], abs=1e-9)
        assert float(cyl.p1[2]) == pytest.approx(mid[2], abs=1e-9)
        assert float(taper.p0[0]) == pytest.approx(mid[0], abs=1e-9)
        assert float(taper.p0[1]) == pytest.approx(mid[1], abs=1e-9)
        assert float(taper.p0[2]) == pytest.approx(mid[2], abs=1e-9)
        assert float(cyl.p1[1]) == pytest.approx(0.0336, abs=1e-9)
        assert float(cyl.p1[2]) == pytest.approx(0.332, abs=1e-9)
        assert float(cyl.p0[0]) == pytest.approx(p0_emit[0], abs=1e-12)
        assert float(cyl.p0[1]) == pytest.approx(p0_emit[1], abs=1e-12)
        assert float(cyl.p0[2]) == pytest.approx(p0_emit[2], abs=1e-12)
        assert abs(float(cyl.p1[1]) - ay) > 1e-3
        assert any(f"calf_{side}: distal/taper p1 Y synced to ank_foot" in m for m in messages)


def test_t2_taper_path_upgrades_cyl_placement() -> None:
    """T2: taper-path front_plane cyl -> full3d after B6 (product leftover)."""
    for side in ("l", "r"):
        parts = _t1_parts(side=side)
        by0 = {p.name: p for p in parts}
        assert by0[f"RECIPE_calf_cyl_{side}"].placement == "front_plane"
        assert by0[f"RECIPE_calf_taper_dist_{side}"].placement == "front_plane"
        messages: list[str] = []
        _sync_calf_distal_to_ankle(parts, messages)
        by_name = {p.name: p for p in parts}
        assert by_name[f"RECIPE_calf_cyl_{side}"].placement == "full3d"
        assert by_name[f"RECIPE_calf_taper_dist_{side}"].placement == "full3d"
        assert by_name[f"RECIPE_calf_b_{side}"].placement == "full3d"


def test_t3_recipe_no_tilt_mid_lerp() -> None:
    """T3: limbs+feet, no template tilt - taper.p1 Y == ank; mid == lerp."""
    report = _limb_mass_report(include_feet_lms=True, knee_y=0.04)
    pkg = build_blockout_recipe(report, limbs=True, feet=True)
    by_name = {p.name: p for p in pkg.parts}
    t = CALF_SPLIT_T
    for side in ("l", "r"):
        ank = by_name[f"RECIPE_ank_foot_{side}"]
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        assert ank.center is not None
        assert cyl.p0 is not None and cyl.p1 is not None
        assert taper.p0 is not None and taper.p1 is not None
        ay = float(ank.center[1])
        assert float(taper.p1[1]) == pytest.approx(ay, abs=1e-6)
        dest = [float(taper.p1[0]), float(taper.p1[1]), float(taper.p1[2])]
        mid = _lerp(list(cyl.p0), dest, t)
        assert float(cyl.p1[0]) == pytest.approx(mid[0], abs=1e-9)
        assert float(cyl.p1[1]) == pytest.approx(mid[1], abs=1e-9)
        assert float(cyl.p1[2]) == pytest.approx(mid[2], abs=1e-9)
        assert float(taper.p0[0]) == pytest.approx(float(cyl.p1[0]), abs=1e-9)
        assert float(taper.p0[1]) == pytest.approx(float(cyl.p1[1]), abs=1e-9)
        assert abs(float(cyl.p1[1]) - ay) > 1e-3


def test_t4_legacy_cyl_only_still_writes_p1() -> None:
    """T4: cyl-only fixture - cyl.p1 Y == ay + full3d; distal/cyl; no taper invented."""
    parts = [
        RecipePart(
            name="RECIPE_calf_b_l",
            role="limb_segment",
            kind="ellipsoid",
            center=[0.1, 0.0, 0.15],
            rx_m=0.038,
            ry_m=0.038,
            rz_m=0.038,
            placement="front_plane",
        ),
        RecipePart(
            name="RECIPE_calf_cyl_l",
            role="limb_segment",
            kind="capsule",
            p0=[0.1, 0.0, 0.45],
            p1=[0.1, 0.0, 0.15],
            radius_m=0.04,
            placement="front_plane",
        ),
        RecipePart(
            name="RECIPE_ank_foot_l",
            role="ankle_bridge",
            kind="ellipsoid",
            center=[0.1, 0.07, 0.08],
            rx_m=0.03,
            ry_m=0.03,
            rz_m=0.03,
            placement="full3d",
        ),
    ]
    messages: list[str] = []
    _sync_calf_distal_to_ankle(parts, messages)
    by_name = {p.name: p for p in parts}
    assert float(by_name["RECIPE_calf_b_l"].center[1]) == pytest.approx(0.07)  # type: ignore[index]
    assert float(by_name["RECIPE_calf_cyl_l"].p1[1]) == pytest.approx(0.07)  # type: ignore[index]
    assert by_name["RECIPE_calf_b_l"].placement == "full3d"
    assert by_name["RECIPE_calf_cyl_l"].placement == "full3d"
    assert any("distal/cyl p1 Y synced to ank_foot" in m for m in messages)
    assert not any(p.name.startswith("RECIPE_calf_taper_dist_") for p in parts)


def test_t5_adduction_still_recomputes_mid() -> None:
    """T5: 10° adduction still locks mid = lerp(cyl.p0, taper.p1, T) (B17)."""
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
        assert cyl.p0 is not None and cyl.p1 is not None
        assert cyl0.p0 is not None and taper.p0 is not None and taper.p1 is not None
        assert abs(float(cyl.p0[0]) - float(cyl0.p0[0])) > 1e-4
        dest = [float(taper.p1[0]), float(taper.p1[1]), float(taper.p1[2])]
        mid = _lerp(list(cyl.p0), dest, t)
        assert float(cyl.p1[0]) == pytest.approx(mid[0], abs=1e-9)
        assert float(cyl.p1[1]) == pytest.approx(mid[1], abs=1e-9)
        assert float(cyl.p1[2]) == pytest.approx(mid[2], abs=1e-9)
        assert float(taper.p0[0]) == pytest.approx(float(cyl.p1[0]), abs=1e-9)
        assert float(taper.p0[1]) == pytest.approx(float(cyl.p1[1]), abs=1e-9)


def test_t6_schema_catalog_n_parts() -> None:
    """T6: recipe 1.4.0; MCP 46; product-class n_parts 131."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(),  # type: ignore[arg-type]
    )
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert len(TOOL_NAMES) == 47
    assert len(pkg.parts) == 131


def test_t7_all_hold_no_helper_export() -> None:
    """T7: CALF_SPLIT_T still in __all__; no _calf_split_mid / resync public name."""
    from meshops.proportion import blockout_recipe as br

    assert "CALF_SPLIT_T" in br.__all__
    assert "_calf_split_mid" not in br.__all__
    assert not any("resync" in name.lower() for name in br.__all__)
    assert not any(name.startswith("_calf_split") for name in br.__all__)


def test_t8_p0_lean_intact_after_t1() -> None:
    """T8: after T1, cyl.p0 Y still emit p0; taper.p1 has no extra lat/rear vs calf_b."""
    for side in ("l", "r"):
        parts = _t1_parts(side=side)
        by0 = {p.name: p for p in parts}
        cyl0 = by0[f"RECIPE_calf_cyl_{side}"]
        assert cyl0.p0 is not None
        p0_y = float(cyl0.p0[1])
        messages: list[str] = []
        _sync_calf_distal_to_ankle(parts, messages)
        by_name = {p.name: p for p in parts}
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        b = by_name[f"RECIPE_calf_b_{side}"]
        assert cyl.p0 is not None and taper.p1 is not None and b.center is not None
        assert float(cyl.p0[1]) == pytest.approx(p0_y, abs=1e-12)
        assert float(taper.p1[0]) == pytest.approx(float(b.center[0]), abs=1e-9)
        assert float(taper.p1[1]) == pytest.approx(float(b.center[1]), abs=1e-9)
        assert float(taper.p1[2]) == pytest.approx(float(b.center[2]), abs=1e-9)
