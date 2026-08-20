"""Track 0083 — Shoulder protraction fix (glenoid Y split from 0051 arm-forward).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY / SKELETON_HONESTY).
Schema 1.4.0 / skeleton 1.0.0 / MCP 46 stay.
"""

from __future__ import annotations

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    DELT_DISTAL_BURY_T,
    DELT_RY_FRAC,
    DELT_RZ_FRAC,
    GLENOID_ANTERIOR_FRAC,
    RECIPE_SCHEMA_VERSION,
    _noskel_arm_endpoint_ys,
    build_blockout_recipe,
)
from meshops.proportion.models import (
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import (
    ARM_FORWARD_OF_HALF_DEPTH_FRAC,
    ELBOW_HANG_T,
    SKELETON_SCHEMA_VERSION,
    _arm_forward_y,
    _elbow_hang_y,
    build_blockout_skeleton,
)


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
    return _report(lms, height_m=h, depth_bands=bands, diameters=diams)


def _product_flags(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "limbs": True,
        "torso": "ovals",
        "glute": "two_spheres",
        "nofuse": True,
        "face": True,
        "hands": True,
        "feet": True,
        "fingers": "full",
        "toes": "full",
        "profile": load_anatomy_profile("torso_limb_f_athletic_v1"),
    }
    base.update(overrides)
    return base


def test_t0_const() -> None:
    """T0: ARM_FORWARD stay 0.45; GLENOID_ANTERIOR_FRAC exported as 0.0."""
    assert ARM_FORWARD_OF_HALF_DEPTH_FRAC == 0.45
    assert GLENOID_ANTERIOR_FRAC == 0.0


def test_t_glenoid_plane() -> None:
    """T_glenoid_plane: product-like chest_mid + half-depth → shoulder Y ≈ 0."""
    h = 1.72
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    j = _by_id(pkg)
    assert j["shoulder_l"].y_m == pytest.approx(0.0, abs=1e-6)
    assert j["shoulder_r"].y_m == pytest.approx(0.0, abs=1e-6)


def test_t_distal_prior() -> None:
    """T_distal_prior (0087): wrist ≈ _arm_forward_y(0); elbow = lerp(0, wrist, T)."""
    h = 1.72
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    j = _by_id(pkg)
    expected = _arm_forward_y(0.0, half_depth=half, height_m=h, chest_front_y=-half)
    hang = _elbow_hang_y(0.0, expected, t=ELBOW_HANG_T)
    assert j["wrist_l"].y_m == pytest.approx(expected, abs=1e-6)
    assert j["wrist_r"].y_m == pytest.approx(expected, abs=1e-6)
    assert j["elbow_l"].y_m == pytest.approx(hang, abs=1e-6)
    assert j["elbow_r"].y_m == pytest.approx(hang, abs=1e-6)


def test_t_no_inherit_zero() -> None:
    """T_no_inherit_zero: wrist message is distal; not inherit glenoid 0."""
    h = 1.72
    lms = _arm_lms(chest_y=0.0, half_depth=0.13)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    wr = [m for m in pkg.messages if "wrist_l" in m]
    assert any("distal" in m for m in wr), wr
    assert not any("inherited" in m and "shoulder_l" in m for m in wr)


def test_t_delt_follows_glenoid() -> None:
    """T_delt_follows_glenoid: delt cy ≈ shoulder Y ≈ 0."""
    h = 1.72
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half)
    report = _report(
        lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)], diameters=_limb_diams()
    )
    skel = build_blockout_skeleton(report)
    sh_y = next(j.y_m for j in skel.joints if j.id == "shoulder_l")
    pkg = build_blockout_recipe(report, limbs=True, skeleton=skel)
    del_l = next(p for p in pkg.parts if p.name == "RECIPE_deltoid_soft_l")
    assert del_l.center is not None
    assert del_l.center[1] == pytest.approx(sh_y, abs=1e-6)
    assert del_l.center[1] == pytest.approx(0.0, abs=1e-6)


def test_t_delt_not_past_chest_front() -> None:
    """T_delt_not_past_chest_front: delt front behind chest oval front."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    delt = next(p for p in pkg.parts if p.name == "RECIPE_deltoid_soft_l")
    chest = next(p for p in pkg.parts if p.name == "RECIPE_torso_oval_chest")
    assert delt.center is not None and chest.center is not None
    assert delt.ry_m is not None and chest.ry_m is not None
    delt_front = float(delt.center[1]) - abs(float(delt.ry_m))
    chest_front = float(chest.center[1]) - abs(float(chest.ry_m))
    assert delt_front >= chest_front - 1e-3


def test_t_gap_shrinks() -> None:
    """T_gap_shrinks: |delt.cy - chest.cy| < 0.050 (was ~0.090)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    delt = next(p for p in pkg.parts if p.name == "RECIPE_deltoid_soft_l")
    chest = next(p for p in pkg.parts if p.name == "RECIPE_torso_oval_chest")
    assert delt.center is not None and chest.center is not None
    assert abs(float(delt.center[1]) - float(chest.center[1])) < 0.050


def test_t_bridge_p1_glenoid() -> None:
    """T_bridge_p1_glenoid: bridge p1 Y ≈ shoulder Y."""
    h = 1.72
    lms = _arm_lms(chest_y=0.0, half_depth=0.13)
    report = _report(
        lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)], diameters=_limb_diams()
    )
    skel = build_blockout_skeleton(report)
    sh_y = next(j.y_m for j in skel.joints if j.id == "shoulder_l")
    pkg = build_blockout_recipe(report, limbs=True, skeleton=skel)
    br = next(p for p in pkg.parts if p.name == "RECIPE_shoulder_bridge_l")
    assert br.p1 is not None
    assert br.p1[1] == pytest.approx(sh_y, abs=1e-6)


def test_t_ua_slant() -> None:
    """T_ua_slant (0087): UA + FA both slant ≥ 0.025 after hang (was ≥ 0.03 UA-only)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    ua = next(p for p in pkg.parts if p.name == "RECIPE_limb_upper_arm_l")
    dist = next(p for p in pkg.parts if p.name == "RECIPE_arm_taper_dist_ua_l")
    fa = next(p for p in pkg.parts if p.name == "RECIPE_limb_forearm_l")
    fa_dist = next(p for p in pkg.parts if p.name == "RECIPE_arm_taper_dist_fa_l")
    assert ua.p0 is not None and dist.p1 is not None
    assert fa.p0 is not None and fa_dist.p1 is not None
    sh_y = next(j.y_m for j in skel.joints if j.id == "shoulder_l")
    el_y = next(j.y_m for j in skel.joints if j.id == "elbow_l")
    assert ua.p0[1] == pytest.approx(sh_y, abs=1e-6)
    assert dist.p1[1] == pytest.approx(el_y, abs=1e-6)
    assert abs(float(ua.p0[1]) - float(dist.p1[1])) >= 0.025
    assert abs(float(fa.p0[1]) - float(fa_dist.p1[1])) >= 0.025


def test_t_landmark_wins() -> None:
    """T_landmark_wins: measured shoulder -0.09 stays; el/wr inherit (off-plane)."""
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


def test_t_plane_class_zero_lm() -> None:
    """T_plane_class_zero_lm (0087): landmark Y=0 → wrist distal; elbow hang lerp."""
    h = 1.72
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half, shoulder_y=0.0)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    j = _by_id(pkg)
    expected = _arm_forward_y(0.0, half_depth=half, height_m=h, chest_front_y=-half)
    hang = _elbow_hang_y(0.0, expected, t=ELBOW_HANG_T)
    assert j["shoulder_l"].y_m == pytest.approx(0.0)
    assert j["wrist_l"].y_m == pytest.approx(expected, abs=1e-6)
    assert j["elbow_l"].y_m == pytest.approx(hang, abs=1e-6)


def test_t_fence_0060() -> None:
    """T_fence_0060: 0103 happened — bury/axes 0.36 / 0.62 / 1.08."""
    assert DELT_DISTAL_BURY_T == 0.36
    assert DELT_RY_FRAC == 0.62
    assert DELT_RZ_FRAC == 1.08


def test_t_clav_lat() -> None:
    """T_clav_lat: med still shelf; lat ~ glenoid (not -0.080)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    sh_y = next(j.y_m for j in skel.joints if j.id == "shoulder_l")
    clav = next(p for p in pkg.parts if p.name == "RECIPE_clavicle_l")
    assert clav.p0 is not None and clav.p1 is not None
    ends = [clav.p0, clav.p1]
    lat = max(ends, key=lambda e: abs(float(e[0])))
    med = min(ends, key=lambda e: abs(float(e[0])))
    chest = next(p for p in pkg.parts if p.name == "RECIPE_torso_oval_chest")
    assert chest.center is not None and chest.ry_m is not None
    shelf = float(chest.center[1]) - abs(float(chest.ry_m))
    assert float(lat[1]) == pytest.approx(sh_y, abs=0.02)
    assert float(med[1]) <= shelf + 1e-3
    assert abs(float(lat[1]) - shelf) > 0.02


def test_t_elbow_mid_ok() -> None:
    """T_elbow_mid_ok: empty-xyz elbow mid(sh, wr) accepted (B22)."""
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
    assert any("elbow_l" in m and "estimated mid shoulder->wrist" in m for m in pkg.messages)


def test_t_n_parts() -> None:
    """T_n_parts: no new RECIPE names; class 130 (0096 +2 shafts)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    assert len(pkg.parts) == 130


def test_t_schema() -> None:
    """T_schema: recipe 1.4.0; skeleton 1.0.0."""
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert SKELETON_SCHEMA_VERSION == "1.0.0"


def test_t_b7_noskel_landmark_p0() -> None:
    """B7 (0087): no-skel UA p0 landmark/plane; p1 hang lerp; FA p0 hang, p1 prior."""
    y_plane = 0.08
    y_prior = -0.0586
    y0, y1 = _noskel_arm_endpoint_ys("upper_arm_l", 0.04, y_plane=y_plane, y_prior=y_prior)
    assert y0 == pytest.approx(0.04)
    assert y1 == pytest.approx(_elbow_hang_y(0.04, y_prior, t=ELBOW_HANG_T))
    y0b, y1b = _noskel_arm_endpoint_ys("upper_arm_r", None, y_plane=y_plane, y_prior=y_prior)
    assert y0b == pytest.approx(y_plane)
    assert y1b == pytest.approx(_elbow_hang_y(y_plane, y_prior, t=ELBOW_HANG_T))
    fa0, fa1 = _noskel_arm_endpoint_ys("forearm_l", None, y_plane=y_plane, y_prior=y_prior)
    assert fa0 == pytest.approx(_elbow_hang_y(y_plane, y_prior, t=ELBOW_HANG_T))
    assert fa1 == pytest.approx(y_prior)
