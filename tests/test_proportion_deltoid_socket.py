"""Track 0060 - Deltoid socket shape (anti-Michelin axes + X-Z distal bury)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    DELT_ARM_RADIUS_SCALE,
    DELT_DISTAL_BURY_T,
    DELT_OUTER_X_FRAC,
    DELT_RY_FRAC,
    DELT_RZ_FRAC,
    RecipePart,
    _apply_deltoid_socket_bury,
    build_blockout_recipe,
)
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
from meshops.proportion.skeleton import build_blockout_skeleton


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
    height_m: float = 1.72,
    arm_hw: float = 0.04,
    thigh_hw: float = 0.06,
    calf_hw: float = 0.05,
    shoulder_x: float = 0.20,
    shoulder_y: float = 0.0,
    elbow_y: float | None = 0.0,
    wrist_y: float | None = 0.0,
) -> ProportionReport:
    """Synthetic full-limb report for 0060 deltoid socket tests."""
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "shoulder_l": _lm("shoulder_l", x_m=-shoulder_x, y_m=shoulder_y, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=shoulder_x, y_m=shoulder_y, z_m=1.38),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=height_m),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=elbow_y, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=elbow_y, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=wrist_y, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=wrist_y, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
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
    kind: str = "ellipsoid",
    role: str = "deltoid_soft",
    center: list[float] | None = None,
    rx_m: float | None = None,
    ry_m: float | None = None,
    rz_m: float | None = None,
    radius_m: float | None = None,
    p0: list[float] | None = None,
    p1: list[float] | None = None,
) -> RecipePart:
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


def _t_xz_along_ua(center: list[float], p0: list[float], p1: list[float]) -> float:
    """Dimensionless XZ shaft parameter of center along UA p0→p1."""
    vx = float(p1[0]) - float(p0[0])
    vz = float(p1[2]) - float(p0[2])
    length2 = vx * vx + vz * vz
    if length2 < 1e-12:
        return 0.0
    dx = float(center[0]) - float(p0[0])
    dz = float(center[2]) - float(p0[2])
    return (dx * vx + dz * vz) / length2


# ---------------------------------------------------------------------------
# T0-T12 + T6b / T7b
# ---------------------------------------------------------------------------


def test_t0_const_freezes() -> None:
    """T0: DELT constant freezes (0103 retarget: 0.62 / 1.08 / t=0.36)."""
    assert DELT_ARM_RADIUS_SCALE == 1.35
    assert DELT_RY_FRAC == 0.62
    assert DELT_RZ_FRAC == 1.08
    assert DELT_OUTER_X_FRAC == 0.08
    assert DELT_DISTAL_BURY_T == 0.36


def test_t1_profile_rx_keeps_bulk() -> None:
    """T1: profile path rx >= arm_hw * 1.35 - eps (0046 bulk law)."""
    arm_hw = 0.04
    report = _limb_mass_report(arm_hw=arm_hw)
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(report, limbs=True, profile=profile)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    expected = arm_hw * DELT_ARM_RADIUS_SCALE
    for d in delts:
        assert d.rx_m is not None
        assert float(d.rx_m) >= expected - 1e-6


def test_t2_base_axes_anisotropy() -> None:
    """T2: base path limbs=False: ry/rz follow DELT_* fracs (0103 const-driven)."""
    arm_hw = 0.04
    report = _limb_mass_report(arm_hw=arm_hw)
    pkg = build_blockout_recipe(report, limbs=False)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    for d in delts:
        assert d.rx_m is not None and d.ry_m is not None and d.rz_m is not None
        rx = float(d.rx_m)
        assert d.ry_m == pytest.approx(rx * DELT_RY_FRAC, abs=1e-9)
        assert d.rz_m == pytest.approx(rx * DELT_RZ_FRAC, abs=1e-9)


def test_t3_product_class_shelf() -> None:
    """T3: soft_outer - ua_outer >= 0.012 m (outer+bulk shelf; product-class arm_hw)."""
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


def test_t3b_outer_frac_pre_bury_geometry() -> None:
    """P3-3 close: reverse-out bury - residual outer offset ~ OUTER*rx (not 0.25)."""
    arm_hw = 0.0438
    report = _limb_mass_report(arm_hw=arm_hw, shoulder_x=0.2575)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by[f"RECIPE_deltoid_soft_{side}"]
        ua = by[f"RECIPE_limb_upper_arm_{side}"]
        assert soft.center is not None and soft.rx_m is not None
        assert ua.p0 is not None and ua.p1 is not None
        sign = 1.0 if side == "r" else -1.0
        # Undo X bury to recover pre-bury center X (Z bury does not affect outer).
        vx = float(ua.p1[0]) - float(ua.p0[0])
        cx_pre = float(soft.center[0]) - DELT_DISTAL_BURY_T * vx
        joint_x = float(ua.p0[0])
        outer_delta = sign * (cx_pre - joint_x)
        expected = DELT_OUTER_X_FRAC * float(soft.rx_m)
        assert outer_delta == pytest.approx(expected, abs=2e-3)
        assert outer_delta < 0.20 * float(soft.rx_m)  # not retired 0.25 float


def test_t4_medial_socket_only() -> None:
    """T4: soft_medial ≤ ua_medial + eps — no retired 0.25 outer OR.

    Also require measurable medial overlap (≥2 mm product-class) so OUTER=0.25
    float without bury cannot satisfy T4 alone.
    """
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
        # Medial-only assert (AI2 P2-2): no retired |cx-joint| < 0.25*rx OR.
        assert soft_medial <= ua_medial + 1e-4
        # Stricter socket bury DoD: soft edge reaches into UA medial by ≥2 mm.
        assert ua_medial - soft_medial >= 0.002 - 1e-6


def test_t5_distal_xz_projection() -> None:
    """T5: XZ shaft parameter of center along UA p0→p1 ≥ 0.12 when UA present."""
    report = _limb_mass_report(arm_hw=0.0438, shoulder_x=0.2575)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by[f"RECIPE_deltoid_soft_{side}"]
        ua = by[f"RECIPE_limb_upper_arm_{side}"]
        assert soft.center is not None and ua.p0 is not None and ua.p1 is not None
        t_xz = _t_xz_along_ua(soft.center, ua.p0, ua.p1)
        assert t_xz >= 0.12 - 1e-6


def test_t6_bury_message() -> None:
    """T6: message contains socket bury + rx= when bury applied."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    for side in ("l", "r"):
        msgs = [m for m in pkg.messages if f"deltoid_{side}: socket bury" in m]
        assert msgs, f"missing bury message for {side}: {pkg.messages}"
        assert any("rx=" in m for m in msgs)


def test_t6b_y_fence_measured_elbow() -> None:
    """T6b: shoulder Y=0, elbow Y=-0.05 -> post-bury delt center[1] ~ shoulder Y."""
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
    assert sh_y == pytest.approx(shoulder_y, abs=1e-9)
    pkg = build_blockout_recipe(report, limbs=True, skeleton=skel)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by[f"RECIPE_deltoid_soft_{side}"]
        assert soft.center is not None
        # Full 3D bury would pull Y toward elbow (-0.05); X-Z bury keeps shoulder Y.
        assert soft.center[1] == pytest.approx(float(sh_y), abs=1e-6)
        assert abs(float(soft.center[1]) - shoulder_y) < 1e-6


def test_t7_missing_ua_skip_message() -> None:
    """T7: delt present, UA missing → skip message + no crash; axes still anisotropic."""
    parts = [
        _part(
            "RECIPE_deltoid_soft_l",
            center=[-0.26, 0.0, 1.38],
            rx_m=0.059,
            ry_m=0.059 * DELT_RY_FRAC,
            rz_m=0.059 * DELT_RZ_FRAC,
        ),
        _part(
            "RECIPE_deltoid_soft_r",
            center=[0.26, 0.0, 1.38],
            rx_m=0.059,
            ry_m=0.059 * DELT_RY_FRAC,
            rz_m=0.059 * DELT_RZ_FRAC,
        ),
    ]
    messages: list[str] = []
    _apply_deltoid_socket_bury(parts, messages)
    assert any("deltoid_l: socket bury skipped (missing UA)" in m for m in messages)
    assert any("deltoid_r: socket bury skipped (missing UA)" in m for m in messages)
    for p in parts:
        assert p.rx_m is not None and p.ry_m is not None and p.rz_m is not None
        assert float(p.ry_m) == pytest.approx(float(p.rx_m) * DELT_RY_FRAC, abs=1e-9)
        assert float(p.rz_m) == pytest.approx(float(p.rx_m) * DELT_RZ_FRAC, abs=1e-9)
        # Centers unchanged without UA
        assert p.center is not None
        assert abs(abs(float(p.center[0])) - 0.26) < 1e-9


def test_t7b_no_delts_no_skip_spam() -> None:
    """T7b: no delts → no 'socket bury skipped' messages."""
    parts = [
        _part(
            "RECIPE_limb_upper_arm_l",
            kind="capsule",
            role="limb_segment",
            p0=[-0.20, 0.0, 1.38],
            p1=[-0.22, 0.0, 1.24],
            radius_m=0.04,
        ),
        _part(
            "RECIPE_limb_upper_arm_r",
            kind="capsule",
            role="limb_segment",
            p0=[0.20, 0.0, 1.38],
            p1=[0.22, 0.0, 1.24],
            radius_m=0.04,
        ),
    ]
    messages: list[str] = []
    _apply_deltoid_socket_bury(parts, messages)
    assert not any("socket bury skipped" in m for m in messages)
    assert messages == []


def test_t8_no_shoulder_bridge_radius_grow() -> None:
    """T8: shoulder_bridge r is emit-formula only — not grown by 0060 bury."""
    arm_hw = 0.0438
    height_m = 1.72
    report = _limb_mass_report(arm_hw=arm_hw, height_m=height_m)
    pkg = build_blockout_recipe(report, limbs=True)
    expected_r = min(0.55 * arm_hw, 0.04 * height_m)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        bridge = by[f"RECIPE_shoulder_bridge_{side}"]
        assert bridge.radius_m is not None
        assert float(bridge.radius_m) == pytest.approx(expected_r, abs=1e-9)
        # Not ballooned toward deltoid bulk
        assert float(bridge.radius_m) < arm_hw * DELT_ARM_RADIUS_SCALE


def test_t9_join_ready_still_runs() -> None:
    """T9: join_ready=True still runs; deltoid present; bury success (not skip)."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True, join_ready=True)
    assert pkg.join_ready is True
    assert any("join_ready=true" in m for m in pkg.messages)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    # Require success form (t= + rx=), not "socket bury skipped".
    assert any("socket bury t=" in m and "rx=" in m for m in pkg.messages)


def test_t10_classifier_and_no_dup() -> None:
    """T10: classify deltoid_soft; C_no_dup_limb green."""
    assert classify_part_name("RECIPE_deltoid_soft_l") == ("deltoid", "l")
    assert classify_part_name("RECIPE_deltoid_soft_r") == ("deltoid", "r")
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    result = validate_constraints(pkg)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_no_dup_limb"].status == "pass", by_id["C_no_dup_limb"].message


def test_t11_t2_constants_export_smoke() -> None:
    """T11: import/smoke that t2 constants used (ry/rz fracs on base path)."""
    arm_hw = 0.04
    report = _limb_mass_report(arm_hw=arm_hw)
    pkg = build_blockout_recipe(report, limbs=False)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    for d in delts:
        assert d.rx_m is not None and d.ry_m is not None and d.rz_m is not None
        assert d.ry_m == pytest.approx(float(d.rx_m) * DELT_RY_FRAC, abs=1e-9)
        assert d.rz_m == pytest.approx(float(d.rx_m) * DELT_RZ_FRAC, abs=1e-9)


def test_t12_m_profile_scale() -> None:
    """T12: M profile scale still DELT_ARM_RADIUS_SCALE."""
    arm_hw = 0.04
    report = _limb_mass_report(arm_hw=arm_hw)
    profile = load_anatomy_profile("torso_limb_m_athletic_v1")
    pkg = build_blockout_recipe(report, limbs=True, profile=profile)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    expected = arm_hw * DELT_ARM_RADIUS_SCALE
    for d in delts:
        assert d.rx_m is not None
        assert float(d.rx_m) >= expected - 1e-6
        assert d.ry_m is not None and d.rz_m is not None
        assert float(d.ry_m) == pytest.approx(float(d.rx_m) * DELT_RY_FRAC, abs=1e-6)
        assert float(d.rz_m) == pytest.approx(float(d.rx_m) * DELT_RZ_FRAC, abs=1e-6)


def test_t5_bury_applies_distal_shift_math() -> None:
    """Unit: bury shifts X/Z by t*v (medial-ward X) and leaves Y."""
    # Left: p1 more medial (x less negative) so X bury is kept (lateral-non-increasing).
    parts = [
        _part(
            "RECIPE_deltoid_soft_l",
            center=[-0.20, 0.0, 1.38],
            rx_m=0.05,
            ry_m=0.036,
            rz_m=0.039,
        ),
        _part(
            "RECIPE_limb_upper_arm_l",
            kind="capsule",
            role="limb_segment",
            p0=[-0.20, 0.0, 1.38],
            p1=[-0.15, -0.05, 1.10],  # medial-ward + distal
            radius_m=0.04,
        ),
    ]
    messages: list[str] = []
    y_before = float(parts[0].center[1])  # type: ignore[index]
    _apply_deltoid_socket_bury(parts, messages)
    c = parts[0].center
    assert c is not None
    vx = 0.05
    vz = 1.10 - 1.38
    t = DELT_DISTAL_BURY_T
    assert c[0] == pytest.approx(-0.20 + t * vx, abs=1e-9)
    assert c[1] == pytest.approx(y_before, abs=1e-12)
    assert c[2] == pytest.approx(1.38 + t * vz, abs=1e-9)
    assert any("socket bury t=" in m for m in messages)
    # Pure-depth (zero XZ) skip path
    parts_z = [
        _part(
            "RECIPE_deltoid_soft_r",
            center=[0.20, 0.0, 1.38],
            rx_m=0.05,
            ry_m=0.036,
            rz_m=0.039,
        ),
        _part(
            "RECIPE_limb_upper_arm_r",
            kind="capsule",
            role="limb_segment",
            p0=[0.20, 0.0, 1.38],
            p1=[0.20, -0.05, 1.38],  # pure Y delta → zero XZ
            radius_m=0.04,
        ),
    ]
    msgs2: list[str] = []
    _apply_deltoid_socket_bury(parts_z, msgs2)
    assert any("zero UA XZ length" in m for m in msgs2)
    assert math.isclose(float(parts_z[0].center[0]), 0.20)  # type: ignore[index]


def test_t5b_lateral_splay_x_bury_clamped() -> None:
    """Product-class: outward UA splay must not increase |cx| via bury X."""
    parts = [
        _part(
            "RECIPE_deltoid_soft_r",
            center=[0.262, 0.0, 1.38],
            rx_m=0.059,
            ry_m=0.042,
            rz_m=0.046,
        ),
        _part(
            "RECIPE_limb_upper_arm_r",
            kind="capsule",
            role="limb_segment",
            p0=[0.2575, 0.0, 1.38],
            p1=[0.3275, 0.0, 1.26],  # outward splay + distal
            radius_m=0.0438,
        ),
    ]
    cx_before = float(parts[0].center[0])  # type: ignore[index]
    messages: list[str] = []
    _apply_deltoid_socket_bury(parts, messages)
    c = parts[0].center
    assert c is not None
    # X clamped (no further lateral); Z still buries
    assert c[0] == pytest.approx(cx_before, abs=1e-12)
    assert c[2] < 1.38 - 1e-6
    assert any("socket bury t=" in m for m in messages)


def test_t7c_nonfinite_ua_skips() -> None:
    """P1 close: NaN / short / inf UA endpoints skip without poisoning center."""
    nan = float("nan")
    cases: list[tuple[list[float], list[float]]] = [
        ([0.2, 0.0, 1.38], [nan, 0.0, 1.10]),
        ([0.2, 0.0, 1.38], [float("inf"), 0.0, 1.10]),
        ([0.2, 0.0], [0.25, 0.0, 1.10]),  # short p0
    ]
    for p0, p1 in cases:
        parts = [
            _part(
                "RECIPE_deltoid_soft_r",
                center=[0.26, 0.0, 1.38],
                rx_m=0.05,
                ry_m=0.036,
                rz_m=0.039,
            ),
            _part(
                "RECIPE_limb_upper_arm_r",
                kind="capsule",
                role="limb_segment",
                p0=p0,
                p1=p1,
                radius_m=0.04,
            ),
        ]
        messages: list[str] = []
        _apply_deltoid_socket_bury(parts, messages)
        c = parts[0].center
        assert c is not None
        assert c[0] == pytest.approx(0.26, abs=1e-12)
        assert math.isfinite(float(c[0])) and math.isfinite(float(c[2]))
        assert any("socket bury skipped" in m for m in messages)
