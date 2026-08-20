"""Track 0103 — deltoid anti-Michelin plus (tall cap + 2x distal bury).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 47 stay. Not mesh/print success.
Does not reopen 0046 scale 1.35, 0060 outer 0.08 / X clamp, 0083 Y=0,
0061 girdle, 0063 bicep, 0105 torso, or 0104 curl.
"""

from __future__ import annotations

import math

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    COMPACT_CULL_ROLES,
    DELT_ARM_RADIUS_SCALE,
    DELT_DISTAL_BURY_T,
    DELT_OUTER_X_FRAC,
    DELT_RY_FRAC,
    DELT_RZ_FRAC,
    RECIPE_SCHEMA_VERSION,
    _apply_deltoid_socket_bury,
    build_blockout_recipe,
)
from meshops.proportion.body_template import (
    AppliedConstants,
    TemplateAppliedPackage,
)
from meshops.proportion.skeleton import build_blockout_skeleton
from test_proportion_deltoid_socket import (
    _limb_mass_report,
    _lm,
    _part,
    _t_xz_along_ua,
)
from test_proportion_torso_anti_tire_plus import (
    _product_class_report,
    _product_flags,
)

_PREV_BURY_T = 0.18
_LIVE_DELT_Z = 1.3584


def _template(*, taper: float = 0.22) -> TemplateAppliedPackage:
    constants = AppliedConstants(
        breast_mode="dual_tilted",
        glute_mode_default="two_spheres",
        torso_mode_default="ovals",
        torso_waist_taper=taper,
        thigh_tilt_deg=10.0,
        breast_tilt_x_deg=20.0,
        intermammary_gap_frac=0.18,
        intermammary_gap_m=0.029,
        breast_y_m=-0.10,
    )
    return TemplateAppliedPackage(
        template_id="female_adult_athletic",
        sex="female",  # type: ignore[arg-type]
        archetype="adult_athletic",
        source_report="mem",
        height_m=1.72,
        constants=constants,
    )


def _product_pkg(**flag_overrides: object):
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    return build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(**flag_overrides),  # type: ignore[arg-type]
    )


def test_t0_const_freezes() -> None:
    """T0: 0103 DELT axes + bury; keep 0046 scale and 0060 outer."""
    assert DELT_RY_FRAC == 0.62
    assert DELT_RZ_FRAC == 1.08
    assert DELT_DISTAL_BURY_T == 0.36
    assert DELT_ARM_RADIUS_SCALE == 1.35
    assert DELT_OUTER_X_FRAC == 0.08


def test_t1_invert_0060_squat() -> None:
    """T1: RZ_FRAC > 1.0 > RY_FRAC (invert 0060 squat). BURY_T == 2 * 0.18."""
    assert DELT_RZ_FRAC > 1.0
    assert DELT_RY_FRAC < 1.0
    assert DELT_RZ_FRAC > DELT_RY_FRAC
    assert DELT_DISTAL_BURY_T == 2.0 * _PREV_BURY_T


def test_t2_base_axes_anisotropy() -> None:
    """T2: base path limbs=False: ry==rx*0.62, rz==rx*1.08."""
    arm_hw = 0.04
    report = _limb_mass_report(arm_hw=arm_hw)
    pkg = build_blockout_recipe(report, limbs=False)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    for d in delts:
        assert d.rx_m is not None and d.ry_m is not None and d.rz_m is not None
        rx = float(d.rx_m)
        assert d.ry_m == pytest.approx(rx * 0.62, abs=1e-9)
        assert d.rz_m == pytest.approx(rx * 1.08, abs=1e-9)


def test_t3_product_class_shelf() -> None:
    """T3: soft_outer - ua_outer >= 0.012 m (0060 T3 hold)."""
    arm_hw = 0.0438
    report = _limb_mass_report(arm_hw=arm_hw, shoulder_x=0.2575)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by[f"RECIPE_deltoid_soft_{side}"]
        ua = by[f"RECIPE_limb_upper_arm_{side}"]
        assert soft.center is not None and soft.rx_m is not None
        assert ua.p0 is not None and ua.radius_m is not None
        soft_outer = abs(float(soft.center[0])) + float(soft.rx_m)
        ua_outer = abs(float(ua.p0[0])) + float(ua.radius_m)
        assert soft_outer - ua_outer >= 0.012 - 1e-6


def test_t4_medial_socket() -> None:
    """T4: soft_medial <= ua_medial; overlap >= 2 mm (0060 T4 hold)."""
    arm_hw = 0.0438
    report = _limb_mass_report(arm_hw=arm_hw, shoulder_x=0.2575)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by[f"RECIPE_deltoid_soft_{side}"]
        ua = by[f"RECIPE_limb_upper_arm_{side}"]
        assert soft.center is not None and soft.rx_m is not None
        assert ua.p0 is not None and ua.radius_m is not None
        soft_medial = abs(float(soft.center[0])) - float(soft.rx_m)
        ua_medial = abs(float(ua.p0[0])) - float(ua.radius_m)
        assert soft_medial <= ua_medial + 1e-4
        assert ua_medial - soft_medial >= 0.002 - 1e-6


def test_t5_t_xz_floor_and_extra_distal() -> None:
    """T5: t_xz >= 0.22 only (do not pin 0.29). Extra distal vs t=0.18 >= 15 mm."""
    report = _limb_mass_report(arm_hw=0.0438, shoulder_x=0.2575)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by[f"RECIPE_deltoid_soft_{side}"]
        ua = by[f"RECIPE_limb_upper_arm_{side}"]
        assert soft.center is not None and ua.p0 is not None and ua.p1 is not None
        t_xz = _t_xz_along_ua(soft.center, ua.p0, ua.p1)
        assert t_xz >= 0.22 - 1e-6
        vz = abs(float(ua.p1[2]) - float(ua.p0[2]))
        extra_z = (DELT_DISTAL_BURY_T - _PREV_BURY_T) * vz
        assert extra_z >= 0.015 - 1e-6


def test_t6_bury_message_t_and_axes() -> None:
    """T6: bury message contains t=0.36 + ry= + rz=; not skip."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    for side in ("l", "r"):
        msgs = [m for m in pkg.messages if f"deltoid_{side}: socket bury" in m]
        assert msgs, f"missing bury message for {side}: {pkg.messages}"
        assert not any("skipped" in m for m in msgs)
        assert any("t=0.36" in m for m in msgs)
        assert any("ry=" in m for m in msgs)
        assert any("rz=" in m for m in msgs)


def test_t6b_y_fence_measured_elbow() -> None:
    """T6b: shoulder Y=0, elbow Y=-0.05 -> post-bury center[1] stays shoulder Y."""
    shoulder_y = 0.0
    report = _limb_mass_report(
        arm_hw=0.0438,
        shoulder_x=0.20,
        shoulder_y=shoulder_y,
        elbow_y=-0.05,
        wrist_y=-0.05,
    )
    skel = build_blockout_skeleton(report)
    sh_y = next(j.y_m for j in skel.joints if j.id == "shoulder_l")
    assert sh_y is not None
    pkg = build_blockout_recipe(report, limbs=True, skeleton=skel)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by[f"RECIPE_deltoid_soft_{side}"]
        assert soft.center is not None
        assert soft.center[1] == pytest.approx(float(sh_y), abs=1e-6)
        assert abs(float(soft.center[1]) - shoulder_y) < 1e-6


def test_t7_missing_ua_skip_message() -> None:
    """T7: missing UA -> skip message; axes still 0.62/1.08; centers unchanged."""
    parts = [
        _part(
            "RECIPE_deltoid_soft_l",
            center=[-0.26, 0.0, 1.38],
            rx_m=0.059,
            ry_m=0.059 * 0.62,
            rz_m=0.059 * 1.08,
        ),
        _part(
            "RECIPE_deltoid_soft_r",
            center=[0.26, 0.0, 1.38],
            rx_m=0.059,
            ry_m=0.059 * 0.62,
            rz_m=0.059 * 1.08,
        ),
    ]
    messages: list[str] = []
    _apply_deltoid_socket_bury(parts, messages)
    assert any("deltoid_l: socket bury skipped (missing UA)" in m for m in messages)
    assert any("deltoid_r: socket bury skipped (missing UA)" in m for m in messages)
    for p in parts:
        assert p.rx_m is not None and p.ry_m is not None and p.rz_m is not None
        assert float(p.ry_m) == pytest.approx(float(p.rx_m) * 0.62, abs=1e-9)
        assert float(p.rz_m) == pytest.approx(float(p.rx_m) * 1.08, abs=1e-9)
        assert p.center is not None
        assert abs(abs(float(p.center[0])) - 0.26) < 1e-9


def test_t8_tall_cap_product_class() -> None:
    """T8: rz_m > rx_m > ry_m on product-class emit (tall cap, not squat ball)."""
    report = _limb_mass_report(arm_hw=0.0438, shoulder_x=0.2575)
    pkg = build_blockout_recipe(report, limbs=True)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    for d in delts:
        assert d.rx_m is not None and d.ry_m is not None and d.rz_m is not None
        assert float(d.rz_m) > float(d.rx_m) > float(d.ry_m)


def test_t9_product_n_parts_131_schema_mcp47() -> None:
    """T9: n_parts 131 via 0060-style product flags + profile; schema 1.4.0; MCP 47."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert pkg.schema_version == "1.4.0"
    assert len(TOOL_NAMES) == 47


def test_t10_all_already_exports_delt_consts() -> None:
    """T10: __all__ already exports DELT_RY_FRAC / DELT_RZ_FRAC / DELT_DISTAL_BURY_T."""
    from meshops.proportion import blockout_recipe as br

    names = set(br.__all__)
    assert "DELT_RY_FRAC" in names
    assert "DELT_RZ_FRAC" in names
    assert "DELT_DISTAL_BURY_T" in names


def test_t11_b25_breast_front_vs_delt() -> None:
    """T11: 0118-class B25 breast front vs delt front >= 0.10 m (ry flatten)."""
    pkg = _product_pkg()
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    breasts = [p for p in pkg.parts if p.role == "breast_soft"]
    assert delts and breasts
    for b in breasts:
        assert b.center is not None and b.ry_m is not None
        breast_front = float(b.center[1]) - float(b.ry_m)
        for d in delts:
            assert d.center is not None and d.ry_m is not None
            delt_front = float(d.center[1]) - float(d.ry_m)
            assert (delt_front - breast_front) >= 0.10


def test_t12_front_plane_no_skeleton_axes() -> None:
    """T12: front_plane / no skeleton: still B1-B3 axes; Y not invented by bury."""
    report = _limb_mass_report(arm_hw=0.04, shoulder_x=0.20)
    report.landmarks_xyz["shoulder_l"] = _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.38)
    report.landmarks_xyz["shoulder_r"] = _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.38)
    pkg = build_blockout_recipe(report, limbs=True)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    for d in delts:
        assert d.rx_m is not None and d.ry_m is not None and d.rz_m is not None
        assert float(d.ry_m) == pytest.approx(float(d.rx_m) * 0.62, abs=1e-6)
        assert float(d.rz_m) == pytest.approx(float(d.rx_m) * 1.08, abs=1e-6)
        assert d.center is not None
        # Front-plane Y is the torso plane (0), not an invented elbow hang.
        assert d.center[1] == pytest.approx(0.0, abs=1e-6)
        assert d.placement in ("front_plane", "full3d")


def test_t13_m_profile_same_fracs() -> None:
    """T13: M profile torso_limb_m_athletic_v1: same 1.35 + 0.62/1.08."""
    arm_hw = 0.04
    report = _limb_mass_report(arm_hw=arm_hw)
    profile = load_anatomy_profile("torso_limb_m_athletic_v1")
    pkg = build_blockout_recipe(report, limbs=True, profile=profile)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    expected = arm_hw * 1.35
    for d in delts:
        assert d.rx_m is not None
        assert float(d.rx_m) >= expected - 1e-6
        assert d.ry_m is not None and d.rz_m is not None
        assert float(d.ry_m) == pytest.approx(float(d.rx_m) * 0.62, abs=1e-6)
        assert float(d.rz_m) == pytest.approx(float(d.rx_m) * 1.08, abs=1e-6)


def test_t14_compact_still_emits_deltoid_soft() -> None:
    """T14: compact soft_density still emits both deltoid_soft (not culled)."""
    assert "deltoid_soft" not in COMPACT_CULL_ROLES
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    names = {p.name for p in delts}
    assert "RECIPE_deltoid_soft_l" in names
    assert "RECIPE_deltoid_soft_r" in names


def test_t15_bury_math_t036_x_clamp_zero_xz() -> None:
    """T15: bury uses t=0.36; X clamp drops outward splay; zero-XZ still skips."""
    assert DELT_DISTAL_BURY_T == 0.36
    parts = [
        _part(
            "RECIPE_deltoid_soft_l",
            center=[-0.20, 0.0, 1.38],
            rx_m=0.05,
            ry_m=0.031,
            rz_m=0.054,
        ),
        _part(
            "RECIPE_limb_upper_arm_l",
            kind="capsule",
            role="limb_segment",
            p0=[-0.20, 0.0, 1.38],
            p1=[-0.15, -0.05, 1.10],
            radius_m=0.04,
        ),
    ]
    messages: list[str] = []
    y_before = float(parts[0].center[1])  # type: ignore[index]
    _apply_deltoid_socket_bury(parts, messages)
    c = parts[0].center
    assert c is not None
    t = 0.36
    vx = 0.05
    vz = 1.10 - 1.38
    assert c[0] == pytest.approx(-0.20 + t * vx, abs=1e-9)
    assert c[1] == pytest.approx(y_before, abs=1e-12)
    assert c[2] == pytest.approx(1.38 + t * vz, abs=1e-9)
    assert any("socket bury t=0.36" in m for m in messages)

    parts_splay = [
        _part(
            "RECIPE_deltoid_soft_r",
            center=[0.262, 0.0, 1.38],
            rx_m=0.059,
            ry_m=0.0366,
            rz_m=0.0637,
        ),
        _part(
            "RECIPE_limb_upper_arm_r",
            kind="capsule",
            role="limb_segment",
            p0=[0.2575, 0.0, 1.38],
            p1=[0.3275, 0.0, 1.26],
            radius_m=0.0438,
        ),
    ]
    cx_before = float(parts_splay[0].center[0])  # type: ignore[index]
    msgs_splay: list[str] = []
    _apply_deltoid_socket_bury(parts_splay, msgs_splay)
    cs = parts_splay[0].center
    assert cs is not None
    assert cs[0] == pytest.approx(cx_before, abs=1e-12)
    assert cs[2] < 1.38 - 1e-6
    assert abs(float(cs[2]) - (1.38 + t * (1.26 - 1.38))) < 1e-9

    parts_z = [
        _part(
            "RECIPE_deltoid_soft_r",
            center=[0.20, 0.0, 1.38],
            rx_m=0.05,
            ry_m=0.031,
            rz_m=0.054,
        ),
        _part(
            "RECIPE_limb_upper_arm_r",
            kind="capsule",
            role="limb_segment",
            p0=[0.20, 0.0, 1.38],
            p1=[0.20, -0.05, 1.38],
            radius_m=0.04,
        ),
    ]
    msgs2: list[str] = []
    _apply_deltoid_socket_bury(parts_z, msgs2)
    assert any("zero UA XZ length" in m for m in msgs2)
    assert math.isclose(float(parts_z[0].center[0]), 0.20)  # type: ignore[index]
    assert _LIVE_DELT_Z == 1.3584
