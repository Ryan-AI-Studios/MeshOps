"""Track 0106 — hip_soft hierarchy plus (trochanter cap + extra Z drop).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 47 stay. Not mesh/print success.
Does not reopen 0069 rx 1.15 / Y 0.12 / joint X, 0077 iliac skip,
0092 hip plate, 0068 glute seat, 0103 delt, or 0105 torso.
"""

from __future__ import annotations

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    COMPACT_CULL_ROLES,
    HIP_SOFT_CENTER,
    HIP_SOFT_RX_SCALE,
    HIP_SOFT_RY_FRAC_RX,
    HIP_SOFT_RZ_FRAC_RX,
    HIP_SOFT_Y_REAR_FRAC_RX,
    HIP_SOFT_Z_DROP_FRAC_H,
    RECIPE_SCHEMA_VERSION,
    TORSO_HIP_Y_REAR_BIAS_FRAC_RY,
    TORSO_OVAL_RY_HIP_FRAC,
    BlockoutRecipePackage,
    _append_all_hip_softs,
    build_blockout_recipe,
)
from meshops.proportion.constraints import classify_part_name, validate_constraints
from meshops.proportion.skeleton import build_blockout_skeleton
from test_proportion_hip_soft_cluster import _limb_mass_report, _part
from test_proportion_torso_anti_tire_plus import _product_class_report, _product_flags

_PREV_Z_DROP = 0.010
_PREV_RZ_FRAC = 0.70


def test_t0_const_freezes() -> None:
    """T0: 0106 HIP_SOFT axes + Z drop; keep 0069 scale / Y rear / joint center."""
    assert HIP_SOFT_RY_FRAC_RX == 0.62
    assert HIP_SOFT_RZ_FRAC_RX == 1.00
    assert HIP_SOFT_Z_DROP_FRAC_H == 0.022
    assert HIP_SOFT_RX_SCALE == 1.15
    assert HIP_SOFT_Y_REAR_FRAC_RX == 0.12
    assert HIP_SOFT_CENTER == "hip_joint"


def test_t1_invert_0069_squat() -> None:
    """T1: RZ_FRAC == 1.0 > RY_FRAC; RZ_FRAC > 0.70; Z_DROP == 2.2 * 0.010."""
    assert HIP_SOFT_RZ_FRAC_RX == 1.0
    assert HIP_SOFT_RZ_FRAC_RX > HIP_SOFT_RY_FRAC_RX
    assert HIP_SOFT_RZ_FRAC_RX > _PREV_RZ_FRAC
    assert abs(HIP_SOFT_Z_DROP_FRAC_H - 2.2 * _PREV_Z_DROP) < 1e-12


def test_t2_base_axes_anisotropy() -> None:
    """T2: limbs=True: ry==rx*0.62, rz==rx*1.00."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        assert soft.rx_m is not None and soft.ry_m is not None and soft.rz_m is not None
        rx = float(soft.rx_m)
        assert soft.ry_m == pytest.approx(rx * 0.62, abs=1e-9)
        assert soft.rz_m == pytest.approx(rx * 1.00, abs=1e-9)


def test_t3_past_cap_hold() -> None:
    """T3: past-cap hold: soft_outer > thigh_cap - 1e-4 (0069 T3)."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        thigh = by_name[f"RECIPE_limb_thigh_{side}"]
        assert soft.center is not None and soft.rx_m is not None
        assert thigh.p0 is not None and thigh.radius_m is not None
        soft_outer = abs(float(soft.center[0])) + float(soft.rx_m)
        thigh_cap = abs(float(thigh.p0[0])) + float(thigh.radius_m)
        assert soft_outer > thigh_cap - 1e-4


def test_t4_joint_x_and_z_below_p0() -> None:
    """T4: Joint X: center[0] == thigh.p0[0]; Z < p0.z when H known."""
    report = _limb_mass_report(height_m=1.72)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        thigh = by_name[f"RECIPE_limb_thigh_{side}"]
        assert soft.center is not None and thigh.p0 is not None
        assert float(soft.center[0]) == pytest.approx(float(thigh.p0[0]), abs=1e-9)
        assert float(soft.center[2]) < float(thigh.p0[2])


def test_t5_extra_drop_vs_0010() -> None:
    """T5: extra drop vs 0.010: (0.022-0.010)*H >= 0.015 on H=1.72. Do not pin z 0.8642."""
    h = 1.72
    extra = (HIP_SOFT_Z_DROP_FRAC_H - _PREV_Z_DROP) * h
    assert extra >= 0.015 - 1e-6
    report = _limb_mass_report(height_m=h)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        thigh = by_name[f"RECIPE_limb_thigh_{side}"]
        assert soft.center is not None and thigh.p0 is not None
        old_z = float(thigh.p0[2]) - _PREV_Z_DROP * h
        new_z = float(thigh.p0[2]) - HIP_SOFT_Z_DROP_FRAC_H * h
        assert float(soft.center[2]) == pytest.approx(new_z, abs=1e-9)
        assert old_z - float(soft.center[2]) >= 0.015 - 1e-6


def test_t6_message_rx_ry_rz_past_cap() -> None:
    """T6: message contains hip_soft_{side}: rx= + past_cap= + new ry/rz interpolated."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    for side in ("l", "r"):
        msgs = [m for m in pkg.messages if m.startswith(f"hip_soft_{side}:")]
        assert msgs, f"missing hip_soft_{side} message: {pkg.messages}"
        m = msgs[0]
        assert f"hip_soft_{side}: rx=" in m
        assert "past_cap=" in m
        assert "ry=" in m
        assert "rz=" in m
        assert "skipped" not in m


def test_t6b_h_missing_no_z_drop() -> None:
    """T6b: H-missing: no Z drop (center Z == p0.z) — 0069 T5 hold."""
    report = _limb_mass_report(height_m=None)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    assert "RECIPE_hip_soft_l" in by_name
    assert "RECIPE_hip_soft_r" in by_name
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        thigh = by_name[f"RECIPE_limb_thigh_{side}"]
        assert soft.center is not None and thigh.p0 is not None
        assert float(soft.center[2]) == pytest.approx(float(thigh.p0[2]), abs=1e-9)


def test_t7_missing_thigh_skip() -> None:
    """T7: missing thigh -> skip message; no hip_soft part."""
    parts = [
        _part(
            "RECIPE_pelvis_oval",
            role="pelvis",
            kind="ellipsoid",
            center=[0.0, 0.0, 0.90],
            rx_m=0.12,
            ry_m=0.08,
            rz_m=0.06,
        ),
    ]
    messages: list[str] = []
    _append_all_hip_softs(parts, height_m=1.72, messages=messages)
    assert not any(p.name.startswith("RECIPE_hip_soft_") for p in parts)
    assert any("hip_soft_l: skipped (no limb_thigh p0/r)" in m for m in messages)
    assert any("hip_soft_r: skipped (no limb_thigh p0/r)" in m for m in messages)


def test_t8_lateral_vertical_oval() -> None:
    """T8: rx_m >= rz_m >= ry_m (lateral + vertical oval, not AP ball). rz == rx named."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        assert soft.rx_m is not None and soft.ry_m is not None and soft.rz_m is not None
        rx = float(soft.rx_m)
        ry = float(soft.ry_m)
        rz = float(soft.rz_m)
        assert rx >= rz - 1e-12
        assert rz >= ry - 1e-12
        assert ry < rx
        assert rz == pytest.approx(rx, abs=1e-9)


def test_t9_product_n_parts_131_schema_mcp47() -> None:
    """T9: n_parts 131 via 0092-style product flags + profile; schema 1.4.0; MCP 47."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert pkg.schema_version == "1.4.0"
    assert len(TOOL_NAMES) == 47


def test_t10_all_already_exports_hip_soft_consts() -> None:
    """T10: __all__ already exports the five HIP_SOFT_* (no new names)."""
    from meshops.proportion import blockout_recipe as br

    names = set(br.__all__)
    assert "HIP_SOFT_RX_SCALE" in names
    assert "HIP_SOFT_RY_FRAC_RX" in names
    assert "HIP_SOFT_RZ_FRAC_RX" in names
    assert "HIP_SOFT_Z_DROP_FRAC_H" in names
    assert "HIP_SOFT_Y_REAR_FRAC_RX" in names
    assert "HIP_SOFT_CENTER" in names


def test_t11_iliac_skip_0077() -> None:
    """T11: iliac 0; skip message 0077 still present."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True, glute="two_spheres")
    iliac = [p for p in pkg.parts if p.role == "iliac_soft"]
    assert len(iliac) == 0
    assert not any(p.name.startswith("RECIPE_iliac_soft_") for p in pkg.parts)
    assert any("iliac_soft skipped: 0077" in m for m in pkg.messages)


def test_t12_glute_pair_c_outer_classifier() -> None:
    """T12: glute pair present; C_glute_outer + C_thigh_outer pass; hip_soft unknown."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True, glute="two_spheres")
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) == 2
    assert classify_part_name("RECIPE_hip_soft_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_hip_soft_r") == ("unknown", "r")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_glute_outer"].status == "pass", by_id["C_glute_outer"].message

    # C_thigh_outer: 0069 T9 aligned synthetic. hip_soft is unknown and must not fail it.
    hip = [-0.12, 0.0, 0.95]
    knee = [-0.12, 0.0, 0.50]
    mid = [0.5 * (hip[0] + knee[0]), 0.5 * (hip[1] + knee[1]), 0.5 * (hip[2] + knee[2])]
    r = 0.06
    chain_outer = mid[0] - r
    hip_half = 0.03
    hip_cx = chain_outer + hip_half
    syn = BlockoutRecipePackage(
        parts=[
            _part(
                "RECIPE_hip_bridge_l",
                role="hip_bridge",
                kind="ellipsoid",
                center=[hip_cx, 0.03, 0.95],
                rx_m=hip_half,
                ry_m=0.03,
                rz_m=0.03,
            ),
            _part("RECIPE_limb_thigh_l", radius_m=r, p0=list(hip), p1=list(mid)),
            _part(
                "RECIPE_thigh_taper_dist_l",
                radius_m=r * 0.8,
                p0=list(mid),
                p1=list(knee),
            ),
            _part(
                "RECIPE_hip_soft_l",
                kind="ellipsoid",
                center=list(hip),
                rx_m=r * HIP_SOFT_RX_SCALE,
                ry_m=r * HIP_SOFT_RX_SCALE * HIP_SOFT_RY_FRAC_RX,
                rz_m=r * HIP_SOFT_RX_SCALE * HIP_SOFT_RZ_FRAC_RX,
            ),
        ],
        counts={"parts": 4},
    )
    syn_result = validate_constraints(syn)
    syn_by = {rule.id: rule for rule in syn_result.rules}
    assert syn_by["C_thigh_outer"].status == "pass", syn_by["C_thigh_outer"].message


def test_t13_compact_still_emits_hip_soft() -> None:
    """T13: compact soft_density still emits both hip_soft (not culled)."""
    assert "limb_segment" not in COMPACT_CULL_ROLES
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    softs = [p for p in pkg.parts if p.name.startswith("RECIPE_hip_soft_")]
    assert len(softs) == 2
    names = {p.name for p in softs}
    assert "RECIPE_hip_soft_l" in names
    assert "RECIPE_hip_soft_r" in names


def test_t14_hip_oval_poles_0092_fence() -> None:
    """T14: 0092 hip oval poles hold (front/rear formula) — fence, do not retune."""
    assert TORSO_OVAL_RY_HIP_FRAC == 0.64
    assert TORSO_HIP_Y_REAR_BIAS_FRAC_RY == 0.33
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    hip = next(p for p in pkg.parts if p.name == "RECIPE_torso_oval_hip")
    assert hip.center is not None and hip.ry_m is not None
    cy = float(hip.center[1])
    ry = float(hip.ry_m)
    assert cy == pytest.approx(TORSO_HIP_Y_REAR_BIAS_FRAC_RY * ry, abs=1e-6)
    front = cy - ry
    rear = cy + ry
    assert front < 0.0
    assert rear > 0.0


def test_t15_y_rear_does_not_steal_glute_seat() -> None:
    """T15: Y rear still 0.12*rx; hip_soft rear < glute rear - 0.04 m (does not steal seat)."""
    assert HIP_SOFT_Y_REAR_FRAC_RX == 0.12
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        glute = by_name[f"RECIPE_glute_soft_{side}"]
        thigh = by_name[f"RECIPE_limb_thigh_{side}"]
        assert soft.center is not None and soft.rx_m is not None and soft.ry_m is not None
        assert glute.center is not None and glute.ry_m is not None
        assert thigh.p0 is not None
        expected_cy = float(thigh.p0[1]) + HIP_SOFT_Y_REAR_FRAC_RX * float(soft.rx_m)
        assert float(soft.center[1]) == pytest.approx(expected_cy, abs=1e-6)
        soft_rear = float(soft.center[1]) + float(soft.ry_m)
        glute_rear = float(glute.center[1]) + float(glute.ry_m)
        assert soft_rear < glute_rear - 0.04
