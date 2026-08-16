"""Track 0069 - Hip soft cluster simplify: anisotropic trochanter at joint (T0-T15)."""

from __future__ import annotations

from typing import Any

import pytest

from meshops.proportion.blockout_recipe import (
    GLUTE_SEAT_BEYOND_REF_Y,
    GLUTE_SEAT_Z_DROP_FRAC_H,
    HIP_SOFT_RX_SCALE,
    HIP_SOFT_RY_FRAC_RX,
    HIP_SOFT_RZ_FRAC_RX,
    HIP_SOFT_Y_REAR_FRAC_RX,
    HIP_SOFT_Z_DROP_FRAC_H,
    THIGH_DIST_SHAFT_SCALE,
    THIGH_PROX_SHAFT_SCALE,
    THIGH_PROX_SOFT_SCALE,
    THIGH_SPLIT_T,
    BlockoutRecipePackage,
    build_blockout_recipe,
)
from meshops.proportion.connection_metrics import _hip_pair
from meshops.proportion.constraints import (
    classify_part_name,
    validate_constraints,
)
from meshops.proportion.models import (
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)


def _lm(
    id_: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=id_, x_m=x_m, y_m=y_m, z_m=z_m)


def _diam(
    band_id: str,
    *,
    half_width_m: float | None = 0.05,
) -> DiameterMeasure:
    w = half_width_m * 2.0 if half_width_m is not None else 0.1
    return DiameterMeasure(
        band_id=band_id,
        view="front",
        width_px=40.0,
        width_eucl_px=40.0,
        theta_deg=90.0,
        width_frac=0.1,
        width_m=w,
        half_width_m=half_width_m,
        mid_x_px=100.0,
        mid_y_px=200.0,
    )


def _limb_mass_report(
    *,
    height_m: float | None = 1.72,
    thigh_hw: float = 0.0613,
    calf_hw: float = 0.05,
    arm_hw: float = 0.04,
    hip_x: float = 0.2224,
) -> ProportionReport:
    """Synthetic full-limb report; product-like mid thigh / hip joint spacing."""
    hx = abs(hip_x)
    lms = {
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "hip_l": _lm("hip_l", x_m=-hx, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=hx, y_m=0.0, z_m=0.95),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=1.72),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=0.0, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=0.0, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=0.0, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=0.0, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-hx, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=hx, y_m=0.04, z_m=0.50),
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


# ---------------------------------------------------------------------------
# T0-T15
# ---------------------------------------------------------------------------


def test_t0_classifier_hip_soft_unknown() -> None:
    """T0: classifier RECIPE_hip_soft_{l,r} → (unknown, side)."""
    assert classify_part_name("RECIPE_hip_soft_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_hip_soft_r") == ("unknown", "r")


def test_t0b_legacy_prox_soft_unknown() -> None:
    """T0b: legacy RECIPE_prox_soft_thigh_* still unknown."""
    assert classify_part_name("RECIPE_prox_soft_thigh_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_prox_soft_thigh_r") == ("unknown", "r")


def test_t1_hip_soft_present_no_prox_soft() -> None:
    """T1: product-like limbs → hip_soft present; no prox_soft_thigh."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    assert "RECIPE_hip_soft_l" in by_name
    assert "RECIPE_hip_soft_r" in by_name
    assert "RECIPE_prox_soft_thigh_l" not in by_name
    assert "RECIPE_prox_soft_thigh_r" not in by_name
    soft_names = [p.name for p in pkg.parts if "prox_soft" in p.name.lower()]
    assert soft_names == []


def test_t2_anisotropic_axes() -> None:
    """T2: ry < rx and rz < rx (de-sphere)."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        assert soft.rx_m is not None and soft.ry_m is not None and soft.rz_m is not None
        assert float(soft.ry_m) < float(soft.rx_m)
        assert float(soft.rz_m) < float(soft.rx_m)
        assert float(soft.ry_m) == pytest.approx(float(soft.rx_m) * HIP_SOFT_RY_FRAC_RX, abs=1e-9)
        assert float(soft.rz_m) == pytest.approx(float(soft.rx_m) * HIP_SOFT_RZ_FRAC_RX, abs=1e-9)


def test_t3_past_cap_visibility() -> None:
    """T3: |soft outer| > |thigh p0.x| + thigh.r - 1e-4 (AI1 P2-1)."""
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


def test_t4_joint_anchor_center() -> None:
    """T4: center X == thigh.p0[0]; center Z ≤ hip_z when H known."""
    report = _limb_mass_report(height_m=1.72)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        thigh = by_name[f"RECIPE_limb_thigh_{side}"]
        assert soft.center is not None and thigh.p0 is not None
        assert float(soft.center[0]) == pytest.approx(float(thigh.p0[0]), abs=1e-9)
        assert float(soft.center[2]) <= float(thigh.p0[2]) + 1e-9


def test_t5_z_drop_and_h_missing() -> None:
    """T5: Z drop never raises; H-missing still emits."""
    h = 1.72
    report = _limb_mass_report(height_m=h)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        thigh = by_name[f"RECIPE_limb_thigh_{side}"]
        assert soft.center is not None and thigh.p0 is not None
        expected_z = float(thigh.p0[2]) - HIP_SOFT_Z_DROP_FRAC_H * h
        assert float(soft.center[2]) == pytest.approx(expected_z, abs=1e-9)
        assert float(soft.center[2]) < float(thigh.p0[2])

    report_no_h = _limb_mass_report(height_m=None)
    pkg_no_h = build_blockout_recipe(report_no_h, limbs=True)
    by_no = {p.name: p for p in pkg_no_h.parts}
    assert "RECIPE_hip_soft_l" in by_no
    assert "RECIPE_hip_soft_r" in by_no
    for side in ("l", "r"):
        soft = by_no[f"RECIPE_hip_soft_{side}"]
        thigh = by_no[f"RECIPE_limb_thigh_{side}"]
        assert soft.center is not None and thigh.p0 is not None
        # No H → no Z drop; center Z == hip joint Z
        assert float(soft.center[2]) == pytest.approx(float(thigh.p0[2]), abs=1e-9)


def test_t6_dual_sides() -> None:
    """T6: dual sides L/R; names side-tagged."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    softs = [p for p in pkg.parts if p.name.startswith("RECIPE_hip_soft_")]
    assert len(softs) == 2
    names = {p.name for p in softs}
    assert names == {"RECIPE_hip_soft_l", "RECIPE_hip_soft_r"}
    by = {p.name: p for p in softs}
    assert by["RECIPE_hip_soft_l"].center is not None
    assert by["RECIPE_hip_soft_r"].center is not None
    assert float(by["RECIPE_hip_soft_l"].center[0]) < 0
    assert float(by["RECIPE_hip_soft_r"].center[0]) > 0


def test_t7_hip_pair_excludes_hip_soft() -> None:
    """T7: _hip_pair excludes hip_soft decoy."""
    thigh = _part(
        "RECIPE_custom_thigh_l",
        kind="capsule",
        radius_m=0.06,
        p0=[0.12, 0.0, 0.95],
        p1=[0.12, 0.0, 0.50],
    )
    decoy = _part(
        "RECIPE_hip_soft_l",
        kind="ellipsoid",
        center=[0.12, 0.0, 0.95],
        rx_m=0.07,
        ry_m=0.06,
        rz_m=0.05,
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
    assert parent.name == "RECIPE_pelvis_oval"
    assert "hip_soft" not in child.name


def test_t8_no_dup_with_hip_soft() -> None:
    """T8: C_no_dup_limb green with hip_soft + taper_dist + limb_thigh."""
    parts = [
        _part(
            "RECIPE_limb_thigh_l",
            radius_m=0.0613,
            p0=[0.12, 0.0, 0.95],
            p1=[0.12, 0.0, 0.725],
        ),
        _part(
            "RECIPE_thigh_taper_dist_l",
            radius_m=0.0490,
            p0=[0.12, 0.0, 0.725],
            p1=[0.12, 0.0, 0.50],
        ),
        _part(
            "RECIPE_hip_soft_l",
            kind="ellipsoid",
            center=[0.12, 0.0, 0.95],
            rx_m=0.0705,
            ry_m=0.0620,
            rz_m=0.0493,
        ),
        _part(
            "RECIPE_knee_soft_l",
            kind="ellipsoid",
            center=[0.12, 0.0, 0.50],
            rx_m=0.0337,
            ry_m=0.0337,
            rz_m=0.0337,
        ),
        _part(
            "RECIPE_calf_a_l",
            kind="ellipsoid",
            center=[0.12, 0.0, 0.50],
            rx_m=0.0385,
            ry_m=0.0385,
            rz_m=0.0385,
        ),
        _part(
            "RECIPE_calf_cyl_l",
            radius_m=0.0473,
            p0=[0.12, 0.0, 0.50],
            p1=[0.12, 0.04, 0.12],
        ),
        _part(
            "RECIPE_calf_b_l",
            kind="ellipsoid",
            center=[0.12, 0.04, 0.12],
            rx_m=0.0315,
            ry_m=0.0315,
            rz_m=0.0315,
        ),
    ]
    pkg = BlockoutRecipePackage(parts=parts, counts={"parts": len(parts)})
    result = validate_constraints(pkg)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_no_dup_limb"].status == "pass", by_id["C_no_dup_limb"].message


def test_t9_thigh_outer_green() -> None:
    """T9: C_thigh_outer green with hip_soft present (bridge/thigh SoT; soft unknown)."""
    from meshops.proportion.constraints import OUTER_X_TOL_M

    # Pre-aligned bridge outer ≈ thigh chain mid outer (same spirit as 0070 T9).
    hip = [-0.12, 0.0, 0.95]
    knee = [-0.12, 0.0, 0.50]
    mid = [0.5 * (hip[0] + knee[0]), 0.5 * (hip[1] + knee[1]), 0.5 * (hip[2] + knee[2])]
    r = 0.06
    chain_mid_x = mid[0]
    chain_outer = chain_mid_x - r  # left outer
    hip_half = 0.03
    hip_cx = chain_outer + hip_half
    parts = [
        _part(
            "RECIPE_hip_bridge_l",
            role="hip_bridge",
            kind="ellipsoid",
            center=[hip_cx, 0.03, 0.95],
            rx_m=hip_half,
            ry_m=0.03,
            rz_m=0.03,
        ),
        _part(
            "RECIPE_limb_thigh_l",
            radius_m=r,
            p0=list(hip),
            p1=list(mid),
        ),
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
    ]
    pkg = BlockoutRecipePackage(parts=parts, counts={"parts": len(parts)})
    result = validate_constraints(pkg)
    by_id = {r.id: r for r in result.rules}
    assert "C_thigh_outer" in by_id
    assert by_id["C_thigh_outer"].status == "pass", by_id["C_thigh_outer"].message
    assert by_id["C_thigh_outer"].metrics is not None
    assert float(by_id["C_thigh_outer"].metrics["delta_l"]) <= OUTER_X_TOL_M + 1e-9
    # hip_soft does not classify as thigh / break free-set
    assert classify_part_name("RECIPE_hip_soft_l") == ("unknown", "l")


def test_t10_glute_outer_align_0036() -> None:
    """T10: 0036 glute outer still aligns to hip_bridge."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True, glute="two_spheres")
    assert any("outer X aligned to hip_bridge" in m for m in pkg.messages)
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_glute_outer"].status == "pass", by_id["C_glute_outer"].message


def test_t11_message_rx_and_past_cap() -> None:
    """T11: message contains hip_soft_{side}: rx= and past_cap=."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    for side in ("l", "r"):
        msgs = [m for m in pkg.messages if m.startswith(f"hip_soft_{side}:")]
        assert msgs, f"missing hip_soft_{side} message"
        m = msgs[0]
        assert f"hip_soft_{side}: rx=" in m
        assert "past_cap=" in m
        assert "outer=" in m
        assert "thigh_cap=" in m


def test_t12_rx_ge_mid_r() -> None:
    """T12: rx ≥ mid_r (scale ≥1.0); soft not buried."""
    mid_r = 0.0613
    report = _limb_mass_report(thigh_hw=mid_r)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        thigh = by_name[f"RECIPE_limb_thigh_{side}"]
        assert soft.rx_m is not None and thigh.radius_m is not None
        assert float(soft.rx_m) >= float(thigh.radius_m) - 1e-9
        assert float(soft.rx_m) == pytest.approx(mid_r * HIP_SOFT_RX_SCALE, abs=1e-5)


def test_t13_0070_taper_fences() -> None:
    """T13: 0070 fences — taper_dist + shaft scales unchanged."""
    mid_r = 0.0613
    report = _limb_mass_report(thigh_hw=mid_r)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    assert THIGH_PROX_SHAFT_SCALE == 1.0
    assert THIGH_DIST_SHAFT_SCALE == 0.72
    assert THIGH_SPLIT_T == 0.5
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        assert float(prox.radius_m) == pytest.approx(  # type: ignore[arg-type]
            mid_r * THIGH_PROX_SHAFT_SCALE, abs=1e-9
        )
        assert float(dist.radius_m) == pytest.approx(  # type: ignore[arg-type]
            mid_r * THIGH_DIST_SHAFT_SCALE, abs=1e-9
        )


def test_t14_0068_seat_freezes() -> None:
    """T14: 0068 seat freezes untouched (import smoke + glute path still runs)."""
    assert GLUTE_SEAT_Z_DROP_FRAC_H == 0.035
    assert GLUTE_SEAT_BEYOND_REF_Y == 0.035
    # Fence: legacy prox soft const still importable (not used for product emit)
    assert THIGH_PROX_SOFT_SCALE == 1.18
    assert HIP_SOFT_RX_SCALE == 1.15
    assert HIP_SOFT_Y_REAR_FRAC_RX == 0.12
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True, glute="two_spheres")
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) >= 1
    assert any("glute" in m.lower() or "seat" in m.lower() for m in pkg.messages) or glutes


def test_t15_no_bridge_outer_clamp_bury() -> None:
    """T15: soft outer may exceed bridge outer (honest; not a fail)."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        bridge = by_name.get(f"RECIPE_hip_bridge_{side}")
        assert soft.center is not None and soft.rx_m is not None
        soft_outer = abs(float(soft.center[0])) + float(soft.rx_m)
        if bridge is not None and bridge.p0 is not None and bridge.radius_m is not None:
            # bridge outer ≈ max |p0.x|,|p1.x| + r (cylinder along X typically)
            xs = [float(bridge.p0[0])]
            if bridge.p1 is not None:
                xs.append(float(bridge.p1[0]))
            bridge_outer = max(abs(x) for x in xs) + float(bridge.radius_m)
            # Joint model: soft may (and typically does) exceed bridge — not a fail
            assert soft_outer > bridge_outer - 1e-3
