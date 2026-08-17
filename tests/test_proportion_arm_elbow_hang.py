"""Track 0087 — Arm elbow hang Y (T=0.50 lerp glenoid→wrist; C1 plane root).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY / SKELETON_HONESTY).
Schema 1.4.0 / skeleton 1.0.0 / MCP 46 stay. Not mesh/print success.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    ELBOW_SOFT_MAX_SCALE,
    ELBOW_SOFT_MIN_FRAC_H,
    ELBOW_SOFT_RY_FRAC,
    ELBOW_SOFT_RZ_FRAC,
    ELBOW_SOFT_SCALE,
    FA_DIST_SHAFT_SCALE,
    FA_PROX_SHAFT_SCALE,
    FA_SPLIT_T,
    RECIPE_SCHEMA_VERSION,
    UA_DIST_SHAFT_SCALE,
    UA_PROX_SHAFT_SCALE,
    UA_SPLIT_T,
    _noskel_arm_endpoint_ys,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.models import (
    CrossSection,
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import (
    ARM_FORWARD_OF_HALF_DEPTH_FRAC,
    ELBOW_HANG_T,
    GLENOID_ANTERIOR_FRAC,
    SKELETON_SCHEMA_VERSION,
    _arm_forward_y,
    _elbow_hang_y,
    build_blockout_skeleton,
)

_PRODUCT_NOFUSE = Path("work/rogue-v3/blockout/product_0087up/nofuse")
_HALF = 0.1303
_EXPECT_WRIST_Y = -0.058635  # 0.45 * 0.1303
_EXPECT_ELBOW_Y = -0.0293175  # lerp(0, wrist, 0.50)


def _lm(
    id_: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=id_, x_m=x_m, y_m=y_m, z_m=z_m)


def _band(
    band_id: str,
    *,
    y_mid: float = 0.0,
    depth_m: float | None = None,
    depth_frac: float = 0.06,
) -> DepthBand:
    return DepthBand(
        band_id=band_id,
        depth_px=20.0,
        depth_frac=depth_frac,
        depth_m=depth_m,
        y_front=(y_mid + depth_frac / 2.0),
        y_back=(y_mid - depth_frac / 2.0),
        y_mid=y_mid,
        z_frac=None,
        confidence=0.8,
        sources=["left"],
        orientation_swapped=False,
    )


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


def _report(
    lms: dict[str, LandmarkXYZ] | None = None,
    *,
    height_m: float | None = 1.72,
    depth_bands: list[DepthBand] | None = None,
    diameters: list[DiameterMeasure] | None = None,
) -> ProportionReport:
    return ProportionReport(
        schema_version="1.2.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms if lms is not None else {},
        depth_bands=list(depth_bands or []),
        diameters=list(diameters or []),
        quality=QualityFlags(),
    )


def _by_id(pkg):  # type: ignore[no-untyped-def]
    return {j.id: j for j in pkg.joints}


def _arm_lms(
    *,
    chest_y: float = 0.0,
    half_depth: float | None = 0.13,
    shoulder_y: float | None = None,
    with_front: bool = True,
    elbow_xyz: bool = True,
) -> dict[str, LandmarkXYZ]:
    lms: dict[str, LandmarkXYZ] = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=shoulder_y, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=shoulder_y, z_m=1.40),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
        "wrist_r": _lm("wrist_r", x_m=0.32, y_m=None, z_m=0.85),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=chest_y, z_m=1.25),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
    }
    if elbow_xyz:
        lms["elbow_l"] = _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10)
        lms["elbow_r"] = _lm("elbow_r", x_m=0.28, y_m=None, z_m=1.10)
    if with_front and half_depth is not None:
        lms["chest_front"] = _lm("chest_front", x_m=0.0, y_m=chest_y - half_depth, z_m=1.25)
    return lms


def _limb_diams() -> list[DiameterMeasure]:
    return [
        _diam("upper_arm_l", half_width_m=0.05),
        _diam("upper_arm_r", half_width_m=0.05),
        _diam("forearm_l"),
        _diam("forearm_r"),
        _diam("thigh_l"),
        _diam("thigh_r"),
        _diam("calf_l"),
        _diam("calf_r"),
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
    ]


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
        _diam("thigh_l", half_width_m=0.0613),
        _diam("thigh_r", half_width_m=0.0613),
        _diam("calf_l", half_width_m=0.0438),
        _diam("calf_r", half_width_m=0.0438),
        _diam("ank_foot_l", half_width_m=0.0263),
        _diam("ank_foot_r", half_width_m=0.0263),
    ]
    bands = [
        _band("chest", y_mid=0.0, depth_m=0.2606),
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
    """T0: ELBOW_HANG_T 0.50 in band; ARM_FORWARD 0.45; glenoid frac 0; elbow mass hold."""
    assert ELBOW_HANG_T == 0.50
    assert 0.45 <= ELBOW_HANG_T <= 0.55
    assert ARM_FORWARD_OF_HALF_DEPTH_FRAC == 0.45
    assert GLENOID_ANTERIOR_FRAC == 0.0
    assert ELBOW_SOFT_SCALE == 1.22
    assert ELBOW_SOFT_RY_FRAC == 0.90
    assert ELBOW_SOFT_RZ_FRAC == 0.78
    assert ELBOW_SOFT_MAX_SCALE == 1.28
    assert ELBOW_SOFT_MIN_FRAC_H == 0.016


def test_t1_product_like_lerp() -> None:
    """T1: elbow == lerp(sh, wr, T); wrist == distal; elbow != wrist."""
    h = 1.72
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    j = _by_id(pkg)
    sh = j["shoulder_l"].y_m
    wr = j["wrist_l"].y_m
    el = j["elbow_l"].y_m
    assert sh is not None and wr is not None and el is not None
    expected_wr = _arm_forward_y(float(sh), half_depth=half, height_m=h, chest_front_y=-half)
    assert wr == pytest.approx(expected_wr, abs=1e-6)
    assert el == pytest.approx(_elbow_hang_y(float(sh), float(wr)), abs=1e-6)
    assert el != pytest.approx(wr, abs=1e-6)


def test_t2_recipe_fa_slants() -> None:
    """T2: recipe-with-skeleton FA p0 == elbow; FA p1 == wrist; |ΔY| >= 0.025."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(),  # type: ignore[arg-type]
    )
    j = _by_id(skel)
    el_y = j["elbow_l"].y_m
    wr_y = j["wrist_l"].y_m
    assert el_y is not None and wr_y is not None
    fa = next(p for p in pkg.parts if p.name == "RECIPE_limb_forearm_l")
    fa_dist = next(p for p in pkg.parts if p.name == "RECIPE_arm_taper_dist_fa_l")
    assert fa.p0 is not None and fa_dist.p1 is not None
    assert fa.p0[1] == pytest.approx(el_y, abs=1e-6)
    assert fa_dist.p1[1] == pytest.approx(wr_y, abs=1e-6)
    assert abs(float(fa.p0[1]) - float(fa_dist.p1[1])) >= 0.025
    msgs = pkg.messages
    sib = [i for i, m in enumerate(msgs) if m.startswith("elbow hang:")]
    el_l = [i for i, m in enumerate(msgs) if m.startswith("elbow_soft_l:")]
    el_r = [i for i, m in enumerate(msgs) if m.startswith("elbow_soft_r:")]
    assert len(sib) == 1
    assert f"t={ELBOW_HANG_T}" in msgs[sib[0]]
    assert sib[0] > el_l[0]
    assert sib[0] > el_r[0]


def test_t3_c1_band_glenoid_roots_distal() -> None:
    """T3 C1: no chest_mid; glenoid from chest band y_mid; wrist prior roots at glenoid Y."""
    h = 1.72
    half = 0.13
    y_mid = 0.04
    glenoid_y = y_mid * h
    lms = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.28, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
        "wrist_r": _lm("wrist_r", x_m=0.32, y_m=None, z_m=0.85),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=glenoid_y - half, z_m=1.25),
    }
    bands = [_band("chest", y_mid=y_mid, depth_m=half * 2.0)]
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=bands))
    j = _by_id(pkg)
    sh = j["shoulder_l"].y_m
    wr = j["wrist_l"].y_m
    assert sh is not None and wr is not None
    assert sh == pytest.approx(glenoid_y, abs=1e-6)
    expected_wr = _arm_forward_y(
        float(sh), half_depth=half, height_m=h, chest_front_y=glenoid_y - half
    )
    assert wr == pytest.approx(expected_wr, abs=1e-6)
    assert wr == pytest.approx(float(sh) - ARM_FORWARD_OF_HALF_DEPTH_FRAC * half, abs=1e-6)
    frozen_zero = _arm_forward_y(0.0, half_depth=half, height_m=h, chest_front_y=glenoid_y - half)
    assert wr != pytest.approx(frozen_zero, abs=1e-4)
    el = j["elbow_l"].y_m
    assert el is not None
    assert el == pytest.approx(_elbow_hang_y(float(sh), float(wr)), abs=1e-6)


def test_t4_off_plane_inherit() -> None:
    """T4: off-plane landmark shoulder -0.09 -> el/wr inherit; no hang add-on."""
    h = 1.72
    measured = -0.09
    lms = _arm_lms(chest_y=0.0, half_depth=0.13, shoulder_y=measured)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    j = _by_id(pkg)
    assert j["shoulder_l"].y_m == pytest.approx(measured)
    assert j["elbow_l"].y_m == pytest.approx(measured)
    assert j["wrist_l"].y_m == pytest.approx(measured)
    assert any("elbow_l" in m and "inherited" in m and "(depth)" in m for m in pkg.messages)
    assert not any("elbow_l" in m and "elbow hang" in m for m in pkg.messages)


def test_t5_empty_xyz_mid() -> None:
    """T5: empty-xyz elbow still mid(sh, wr); equals T=0.50 hang."""
    h = 1.72
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half, elbow_xyz=False)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    j = _by_id(pkg)
    sh = j["shoulder_l"].y_m
    wr = j["wrist_l"].y_m
    el = j["elbow_l"].y_m
    assert sh is not None and wr is not None and el is not None
    mid = 0.5 * (float(sh) + float(wr))
    assert el == pytest.approx(mid, abs=1e-6)
    assert el == pytest.approx(_elbow_hang_y(float(sh), float(wr)), abs=1e-6)
    assert any("elbow_l" in m and "estimated mid shoulder->wrist" in m for m in pkg.messages)
    assert not any("elbow_l" in m and "elbow hang" in m for m in pkg.messages)


def test_t6_hang_messages() -> None:
    """T6: wrist still distal; elbow hang t=const; no arm-forward on shoulder_*."""
    h = 1.72
    lms = _arm_lms(chest_y=0.0, half_depth=0.13)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    el_msgs = [m for m in pkg.messages if "elbow_l" in m]
    wr_msgs = [m for m in pkg.messages if "wrist_l" in m]
    assert any("hang" in m and f"t={ELBOW_HANG_T}" in m for m in el_msgs), el_msgs
    assert any("distal" in m for m in wr_msgs), wr_msgs
    assert not any("shoulder_l" in m and "arm forward prior" in m for m in pkg.messages)
    assert not any("shoulder_r" in m and "arm forward prior" in m for m in pkg.messages)


def test_t7_noskel_both_null_hang() -> None:
    """T7: no-skel both-null: UA p0=plane p1=lerp; FA p0=lerp p1=prior."""
    y_plane = 0.0
    y_prior = -0.0586
    hang = _elbow_hang_y(y_plane, y_prior)
    ua0, ua1 = _noskel_arm_endpoint_ys("upper_arm_l", None, y_plane=y_plane, y_prior=y_prior)
    assert ua0 == pytest.approx(y_plane)
    assert ua1 == pytest.approx(hang)
    fa0, fa1 = _noskel_arm_endpoint_ys("forearm_l", None, y_plane=y_plane, y_prior=y_prior)
    assert fa0 == pytest.approx(hang)
    assert fa1 == pytest.approx(y_prior)
    # Helper-only landmark-p0 still hangs p1 from that p0 (not production).
    y0, y1 = _noskel_arm_endpoint_ys("upper_arm_l", 0.04, y_plane=y_plane, y_prior=y_prior)
    assert y0 == pytest.approx(0.04)
    assert y1 == pytest.approx(_elbow_hang_y(0.04, y_prior))


def test_t8_c2_mixed_null_front_plane() -> None:
    """T8 C2: mixed-null UA still front_plane (0051 T5b cousin). No hang assert."""
    measured_sh_y = -0.04
    lms = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=measured_sh_y, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.40),
        "elbow_r": _lm("elbow_r", x_m=0.28, y_m=0.0, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=0.0, z_m=0.85),
        "wrist_r": _lm("wrist_r", x_m=0.32, y_m=0.0, z_m=0.85),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.0, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.0, z_m=0.50),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
    }
    report = _report(lms, diameters=_limb_diams(), depth_bands=[_band("chest", depth_m=0.26)])
    pkg = build_blockout_recipe(report, limbs=True)
    ua = next(p for p in pkg.parts if p.name == "RECIPE_limb_upper_arm_l")
    assert ua.placement == "front_plane"
    assert ua.p0 is not None and ua.p1 is not None
    assert ua.p0[1] == pytest.approx(measured_sh_y)
    assert ua.p1[1] == pytest.approx(measured_sh_y)
    assert not any("upper_arm_l" in m and "elbow hang" in m for m in pkg.messages)


def test_t9_n_parts_schema_mcp() -> None:
    """T9: n_parts 129 via product flags; schema 1.4.0 / skeleton 1.0.0; MCP 46."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(),  # type: ignore[arg-type]
    )
    assert len(pkg.parts) == 129
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert SKELETON_SCHEMA_VERSION == "1.0.0"
    assert len(TOOL_NAMES) == 46


def test_t10_product_like_meters() -> None:
    """T10: product-like elbow y ~ -0.02932; wrist ~ -0.05864 (half=0.1303)."""
    report = _product_class_report()
    pkg = build_blockout_skeleton(report)
    j = _by_id(pkg)
    assert j["shoulder_l"].y_m == pytest.approx(0.0, abs=1e-6)
    assert j["wrist_l"].y_m == pytest.approx(_EXPECT_WRIST_Y, abs=2e-4)
    assert j["elbow_l"].y_m == pytest.approx(_EXPECT_ELBOW_Y, abs=2e-4)
    assert j["elbow_l"].y_m != pytest.approx(j["wrist_l"].y_m, abs=1e-4)


def test_t11_fence_mass_and_palm() -> None:
    """T11: ELBOW_SOFT_* / UA/FA shaft scales hold; palm Y == wrist Y."""
    assert ELBOW_SOFT_SCALE == 1.22
    assert ELBOW_SOFT_RY_FRAC == 0.90
    assert ELBOW_SOFT_RZ_FRAC == 0.78
    assert UA_PROX_SHAFT_SCALE == 1.00
    assert UA_DIST_SHAFT_SCALE == 0.88
    assert UA_SPLIT_T == 0.50
    assert FA_PROX_SHAFT_SCALE == 1.00
    assert FA_DIST_SHAFT_SCALE == 0.78
    assert FA_SPLIT_T == 0.50
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(),  # type: ignore[arg-type]
    )
    j = _by_id(skel)
    wr = j["wrist_l"].y_m
    palm = next(p for p in pkg.parts if p.name == "RECIPE_palm_l")
    assert palm.center is not None and wr is not None
    assert palm.center[1] == pytest.approx(wr, abs=1e-6)
    elbow = next(p for p in pkg.parts if p.name == "RECIPE_elbow_soft_l")
    assert elbow.rx_m is not None
    assert float(elbow.rx_m) == pytest.approx(0.0470, abs=1e-4)


def test_t12_all_exports() -> None:
    """T12: skeleton __all__ exports ELBOW_HANG_T and _elbow_hang_y."""
    from meshops.proportion import skeleton as skel_mod

    assert "ELBOW_HANG_T" in skel_mod.__all__
    assert "_elbow_hang_y" in skel_mod.__all__


def test_t13_delt_clav_fence() -> None:
    """T13: delt cy still ≈ glenoid 0; clav lat still glenoid."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(),  # type: ignore[arg-type]
    )
    sh_y = next(j.y_m for j in skel.joints if j.id == "shoulder_l")
    delt = next(p for p in pkg.parts if p.name == "RECIPE_deltoid_soft_l")
    assert delt.center is not None
    assert delt.center[1] == pytest.approx(sh_y, abs=1e-6)
    assert delt.center[1] == pytest.approx(0.0, abs=1e-6)
    clav = next(p for p in pkg.parts if p.name == "RECIPE_clavicle_l")
    assert clav.p0 is not None and clav.p1 is not None
    lat = max((clav.p0, clav.p1), key=lambda e: abs(float(e[0])))
    assert float(lat[1]) == pytest.approx(sh_y, abs=0.02)


def test_t14_both_sides_equalize() -> None:
    """T14: L/R elbow y match; L/R wrist y match."""
    report = _product_class_report()
    pkg = build_blockout_skeleton(report)
    j = _by_id(pkg)
    assert j["elbow_l"].y_m == pytest.approx(j["elbow_r"].y_m, abs=1e-9)
    assert j["wrist_l"].y_m == pytest.approx(j["wrist_r"].y_m, abs=1e-9)
    assert j["shoulder_l"].y_m == pytest.approx(j["shoulder_r"].y_m, abs=1e-9)


def test_t15_constraints_no_dup_limb() -> None:
    """T15: product-path C_no_dup_limb pass (same flags as 0083/0095)."""
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
        assert rule_by.get("C_no_dup_limb") == "pass"
