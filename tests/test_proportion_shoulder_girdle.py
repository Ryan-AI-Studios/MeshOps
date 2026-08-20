"""Track 0061 - Shoulder girdle softs (clavicle ridge + trap nape floors)."""

from __future__ import annotations

from typing import Any

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    CLAVICLE_LATERAL_INSET_FRAC,
    CLAVICLE_MEDIAL_Z_DROP_FRAC_H,
    CLAVICLE_RADIUS_FRAC_H,
    DELT_RY_FRAC,
    NECK_NAPE_CLEARANCE_M,
    TRAP_LAT_FRAC,
    TRAP_NAPE_Z_BIAS_FRAC_H,
    TRAP_RX_FLOOR_FRAC_H,
    TRAP_RY_FLOOR_FRAC_H,
    TRAP_RZ_FLOOR_FRAC_H,
    TRAP_Y_BACK_FRAC_RY,
    TRAP_Y_NEAR_ZERO,
    RecipePart,
    _apply_shoulder_girdle_softs,
    _chest_front_y_for_girdle,
    _neck_upper_z,
    build_blockout_recipe,
)
from meshops.proportion.models import (
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import BlockoutSkeleton, SkeletonBone, SkeletonJoint


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


def _girdle_report(
    *,
    height_m: float = 1.72,
    arm_hw: float = 0.04,
    shoulder_x: float = 0.20,
    shoulder_y: float = -0.04,
    chest_front_y: float = -0.08,
    shoulder_z: float = 1.38,
) -> ProportionReport:
    """Synthetic report with chest_front, shoulders, neck, limbs for 0061."""
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "shoulder_l": _lm("shoulder_l", x_m=-shoulder_x, y_m=shoulder_y, z_m=shoulder_z),
        "shoulder_r": _lm("shoulder_r", x_m=shoulder_x, y_m=shoulder_y, z_m=shoulder_z),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=height_m),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=shoulder_y, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=shoulder_y, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=shoulder_y, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=shoulder_y, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=chest_front_y, z_m=1.25),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=arm_hw),
        _diam("upper_arm_r", half_width_m=arm_hw),
        _diam("forearm_l", half_width_m=arm_hw),
        _diam("forearm_r", half_width_m=arm_hw),
        _diam("thigh_l", half_width_m=0.06),
        _diam("thigh_r", half_width_m=0.06),
        _diam("calf_l", half_width_m=0.05),
        _diam("calf_r", half_width_m=0.05),
    ]
    return ProportionReport(
        schema_version="1.0.0",
        height_m=height_m,
        landmarks_xyz=lms,
        diameters=diams,
        quality=QualityFlags(),
    )


def _skeleton_with_arms(
    *,
    shoulder_x: float = 0.20,
    shoulder_y: float = -0.04,
    shoulder_z: float = 1.38,
    neck_z: float = 1.42,
    spine_z: float = 1.25,
) -> BlockoutSkeleton:
    def j(
        id_: str,
        *,
        x: float,
        y: float,
        z: float,
        side: str = "none",
        parent: str | None = None,
    ) -> SkeletonJoint:
        return SkeletonJoint(
            id=id_,
            parent=parent,
            side=side,  # type: ignore[arg-type]
            x_m=x,
            y_m=y,
            z_m=z,
            source="estimated",
        )

    joints = [
        j("root", x=0.0, y=0.0, z=0.0),
        j("pelvis", x=0.0, y=0.0, z=0.95, parent="root"),
        j("spine_high", x=0.0, y=0.0, z=spine_z, parent="pelvis"),
        j("neck_base", x=0.0, y=0.0, z=neck_z, parent="spine_high"),
        j(
            "shoulder_l",
            x=-shoulder_x,
            y=shoulder_y,
            z=shoulder_z,
            side="l",
            parent="spine_high",
        ),
        j(
            "shoulder_r",
            x=shoulder_x,
            y=shoulder_y,
            z=shoulder_z,
            side="r",
            parent="spine_high",
        ),
        j("elbow_l", x=-0.28, y=shoulder_y, z=1.10, side="l", parent="shoulder_l"),
        j("elbow_r", x=0.28, y=shoulder_y, z=1.10, side="r", parent="shoulder_r"),
    ]
    bones = [
        SkeletonBone(id="spine", joint_a="pelvis", joint_b="spine_high", length_m=0.3),
        SkeletonBone(id="upper_arm_l", joint_a="shoulder_l", joint_b="elbow_l", length_m=0.3),
        SkeletonBone(id="upper_arm_r", joint_a="shoulder_r", joint_b="elbow_r", length_m=0.3),
    ]
    return BlockoutSkeleton(
        schema_version="1.0.0",
        honesty="proportion_blockout_skeleton_not_mesh_or_print_success",
        joints=joints,
        bones=bones,
        messages=[],
    )


def _part(
    name: str,
    *,
    kind: str = "ellipsoid",
    role: str = "trap_soft",
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


def _build_profile_pkg(
    *,
    limbs: bool = True,
    shoulder_x: float = 0.20,
    shoulder_y: float = -0.04,
    chest_front_y: float = -0.08,
    height_m: float = 1.72,
):
    report = _girdle_report(
        height_m=height_m,
        shoulder_x=shoulder_x,
        shoulder_y=shoulder_y,
        chest_front_y=chest_front_y,
    )
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    skel = _skeleton_with_arms(
        shoulder_x=shoulder_x,
        shoulder_y=shoulder_y,
    )
    return build_blockout_recipe(
        report,
        limbs=limbs,
        profile=profile,
        skeleton=skel,
        torso="ovals",
    )


class _MetricsStub:
    def __init__(
        self,
        *,
        height_m: float | None = 1.72,
        shoulder_hw: float | None = 0.20,
        chest_y: float | None = 0.0,
        chest_half_depth: float | None = 0.08,
    ) -> None:
        self.height_m = height_m
        self.shoulder_hw = shoulder_hw
        self.chest_y = chest_y
        self.chest_half_depth = chest_half_depth


# ---------------------------------------------------------------------------
# T0-T13
# ---------------------------------------------------------------------------


def test_t0_const_freezes() -> None:
    """T0: 0061 constant freezes including TRAP_LAT_FRAC."""
    assert CLAVICLE_RADIUS_FRAC_H == 0.012
    assert CLAVICLE_MEDIAL_Z_DROP_FRAC_H == 0.025
    assert CLAVICLE_LATERAL_INSET_FRAC == 0.06
    assert TRAP_RX_FLOOR_FRAC_H == 0.042
    assert TRAP_RY_FLOOR_FRAC_H == 0.022
    assert TRAP_RZ_FLOOR_FRAC_H == 0.038
    assert TRAP_LAT_FRAC == 0.55
    assert TRAP_NAPE_Z_BIAS_FRAC_H == 0.010
    assert TRAP_Y_NEAR_ZERO == 1e-4
    assert TRAP_Y_BACK_FRAC_RY == 0.4
    assert NECK_NAPE_CLEARANCE_M == 0.005


def test_t1_clavicle_radius_floor() -> None:
    """T1: clavicle r >= 0.012 * H with profile."""
    h = 1.72
    pkg = _build_profile_pkg(height_m=h)
    floor = CLAVICLE_RADIUS_FRAC_H * h
    clavs = [p for p in pkg.parts if p.role == "clavicle"]
    assert len(clavs) == 2
    for c in clavs:
        assert c.radius_m is not None
        assert float(c.radius_m) >= floor - 1e-9


def test_t2_both_sides_parent_joint() -> None:
    """T2: both sides clavicle; parent_joint shoulder_l/r side-correct."""
    pkg = _build_profile_pkg()
    by = {p.name: p for p in pkg.parts}
    assert "RECIPE_clavicle_l" in by and "RECIPE_clavicle_r" in by
    assert by["RECIPE_clavicle_l"].parent_joint == "shoulder_l"
    assert by["RECIPE_clavicle_r"].parent_joint == "shoulder_r"


def test_t3_asymmetric_front_shelf() -> None:
    """T3: medial on shelf; lateral deepen-front only (not lat==med forced).

    0090: `_chest_front_y_for_girdle` is max(landmark, oval_front). Live oval
    front after the thoracic plate is less proud than landmark -0.08, so the
    oval binds (B15 cascade). Lat still follows glenoid/emit (0083 B21).
    """
    lm_shelf = -0.08
    shoulder_y = -0.04  # lateral starts behind landmark (less front)
    pkg = _build_profile_pkg(shoulder_y=shoulder_y, chest_front_y=lm_shelf)
    chest = next(p for p in pkg.parts if p.name == "RECIPE_torso_oval_chest")
    assert chest.center is not None and chest.ry_m is not None
    oval_front = float(chest.center[1]) - float(chest.ry_m)
    expected_shelf = max(lm_shelf, oval_front)
    for side in ("l", "r"):
        clav = next(p for p in pkg.parts if p.name == f"RECIPE_clavicle_{side}")
        assert clav.p0 is not None and clav.p1 is not None
        ends = [clav.p0, clav.p1]
        lat = max(ends, key=lambda e: abs(float(e[0])))
        med = min(ends, key=lambda e: abs(float(e[0])))
        assert float(med[1]) <= expected_shelf + 1e-4
        assert float(med[1]) >= oval_front - 1e-4
        # 0083 B21: lat follows glenoid/emit — not forced to shelf
        assert float(lat[1]) == pytest.approx(shoulder_y, abs=1e-3)
        assert abs(float(lat[1]) - float(med[1])) > 1e-4


def test_t3b_lat_already_front_unchanged() -> None:
    """T3b: when pre-lat already front of shelf, lat.y unchanged (ridge depth)."""
    shelf = -0.05
    lat_y = -0.08  # more front than shelf
    med_y = 0.0
    parts = [
        _part(
            "RECIPE_clavicle_r",
            kind="capsule",
            role="clavicle",
            p0=[0.20, lat_y, 1.38],
            p1=[0.0, med_y, 1.27],
            radius_m=0.01,
        ),
        _part(
            "RECIPE_trap_soft_r",
            role="trap_soft",
            center=[0.10, 0.01, 1.38],
            rx_m=0.03,
            ry_m=0.01,
            rz_m=0.02,
        ),
    ]
    report = _girdle_report(chest_front_y=shelf, shoulder_y=lat_y)
    m = _MetricsStub(height_m=1.72, shoulder_hw=0.20)
    messages: list[str] = []
    _apply_shoulder_girdle_softs(parts, report, m, messages)  # type: ignore[arg-type]
    clav = parts[0]
    assert clav.p0 is not None and clav.p1 is not None
    ends = [clav.p0, clav.p1]
    lat = max(ends, key=lambda e: abs(float(e[0])))
    med = min(ends, key=lambda e: abs(float(e[0])))
    assert float(lat[1]) == pytest.approx(lat_y, abs=1e-9)
    assert float(med[1]) <= shelf + 1e-4
    assert float(lat[1]) < float(med[1])  # ridge: lat more front than med


def test_t3c_landmark_past_oval_shelf_is_oval() -> None:
    """T3c: landmark more front than oval → shelf = oval front (anti-overshoot)."""
    # Oval front = center.y - ry = 0.03 - 0.11 = -0.08
    oval_front = 0.03 - 0.11
    lm_y = -0.1303  # ~5 cm more front than oval (product left-only landmark class)
    parts = [
        _part(
            "RECIPE_torso_oval_chest",
            role="torso",
            center=[0.0, 0.03, 1.25],
            rx_m=0.16,
            ry_m=0.11,
            rz_m=0.08,
        ),
        _part(
            "RECIPE_clavicle_r",
            kind="capsule",
            role="clavicle",
            p0=[0.20, -0.04, 1.38],
            p1=[0.0, 0.0, 1.27],
            radius_m=0.01,
        ),
    ]
    report = _girdle_report(chest_front_y=lm_y, shoulder_y=-0.04)
    m = _MetricsStub(height_m=1.72, shoulder_hw=0.20)
    shelf = _chest_front_y_for_girdle(report, m, parts)  # type: ignore[arg-type]
    assert shelf == pytest.approx(oval_front, abs=1e-9)
    assert shelf is not None
    assert shelf > lm_y  # less front than overshoot landmark
    messages: list[str] = []
    _apply_shoulder_girdle_softs(parts, report, m, messages)  # type: ignore[arg-type]
    clav = parts[1]
    assert clav.p0 is not None and clav.p1 is not None
    ends = [clav.p0, clav.p1]
    lat = max(ends, key=lambda e: abs(float(e[0])))
    med = min(ends, key=lambda e: abs(float(e[0])))
    # Med deepens only to oval front — not landmark -0.1303. Lat keeps emit Y (B21).
    assert float(med[1]) == pytest.approx(oval_front, abs=1e-4)
    assert float(lat[1]) == pytest.approx(-0.04, abs=1e-4)
    assert float(med[1]) > lm_y + 1e-3


def test_t4_medial_x_and_z_drop() -> None:
    """T4: medial |x| ~ 0; Z drop ~ 0.025 * H (not full spine dive)."""
    h = 1.72
    pkg = _build_profile_pkg(height_m=h)
    for side in ("l", "r"):
        clav = next(p for p in pkg.parts if p.name == f"RECIPE_clavicle_{side}")
        assert clav.p0 is not None and clav.p1 is not None
        ends = [clav.p0, clav.p1]
        med = min(ends, key=lambda e: abs(float(e[0])))
        lat = max(ends, key=lambda e: abs(float(e[0])))
        assert abs(float(med[0])) < 1e-6
        expected_z = float(lat[2]) - CLAVICLE_MEDIAL_Z_DROP_FRAC_H * h
        assert float(med[2]) == pytest.approx(expected_z, abs=2e-3)
        # Not full dive to spine_high ~1.25
        assert float(med[2]) > 1.30


def test_t5_trap_axis_floors() -> None:
    """T5: trap axes >= floors * H."""
    h = 1.72
    pkg = _build_profile_pkg(height_m=h)
    traps = [p for p in pkg.parts if p.role == "trap_soft"]
    assert len(traps) == 2
    for t in traps:
        assert t.rx_m is not None and t.ry_m is not None and t.rz_m is not None
        assert float(t.rx_m) >= TRAP_RX_FLOOR_FRAC_H * h - 1e-9
        assert float(t.ry_m) >= TRAP_RY_FLOOR_FRAC_H * h - 1e-9
        assert float(t.rz_m) >= TRAP_RZ_FLOOR_FRAC_H * h - 1e-9


def test_t6_trap_lat_frac_l_ne_r() -> None:
    """T6: trap |cx| ~ 0.55 * |sh_x|; L != R."""
    sh_x = 0.20
    pkg = _build_profile_pkg(shoulder_x=sh_x)
    trap_l = next(p for p in pkg.parts if p.name == "RECIPE_trap_soft_l")
    trap_r = next(p for p in pkg.parts if p.name == "RECIPE_trap_soft_r")
    assert trap_l.center is not None and trap_r.center is not None
    expected = TRAP_LAT_FRAC * sh_x
    assert abs(float(trap_l.center[0])) == pytest.approx(expected, abs=1e-5)
    assert abs(float(trap_r.center[0])) == pytest.approx(expected, abs=1e-5)
    assert float(trap_l.center[0]) != float(trap_r.center[0])
    assert float(trap_l.center[0]) < 0.0 < float(trap_r.center[0])


def test_t6b_trap_cx_in_shoulder_range() -> None:
    """T6b: 0 <= |cx| <= |shoulder_x|."""
    sh_x = 0.2575
    pkg = _build_profile_pkg(shoulder_x=sh_x)
    for side in ("l", "r"):
        trap = next(p for p in pkg.parts if p.name == f"RECIPE_trap_soft_{side}")
        assert trap.center is not None
        cx = abs(float(trap.center[0]))
        assert 0.0 <= cx <= sh_x + 1e-6


def test_t7_trap_nape_z_clamp() -> None:
    """T7: trap Z raised by nape bias; not above neck upper - clearance."""
    h = 1.72
    pkg = _build_profile_pkg(height_m=h)
    neck_z = _neck_upper_z(pkg.parts)
    for side in ("l", "r"):
        trap = next(p for p in pkg.parts if p.name == f"RECIPE_trap_soft_{side}")
        assert trap.center is not None
        # Raised above bare shoulder mid (1.38) by nape bias when clamp allows
        assert float(trap.center[2]) >= 1.38 - 1e-3
        if neck_z is not None:
            assert float(trap.center[2]) <= float(neck_z) - NECK_NAPE_CLEARANCE_M + 1e-6


def test_t8_trap_y_back_positive() -> None:
    """T8: trap cy > 0; ry-fallback re-derives to floored ry * 0.4 (unit path)."""
    h = 1.72
    pkg = _build_profile_pkg(height_m=h, shoulder_y=0.0)
    for side in ("l", "r"):
        trap = next(p for p in pkg.parts if p.name == f"RECIPE_trap_soft_{side}")
        assert trap.center is not None and trap.ry_m is not None
        assert float(trap.center[1]) > 0.0

    # Unit: pre-floor y was ry-fallback (old_ry * 0.4) → refresh after floor.
    old_ry = 0.012 * h  # thin pre-floor
    pre_y = abs(old_ry) * TRAP_Y_BACK_FRAC_RY
    parts = [
        _part(
            "RECIPE_trap_soft_l",
            role="trap_soft",
            center=[-0.10, pre_y, 1.38],
            rx_m=0.035 * h,
            ry_m=old_ry,
            rz_m=0.028 * h,
        ),
        _part(
            "RECIPE_trap_soft_r",
            role="trap_soft",
            center=[0.10, pre_y, 1.38],
            rx_m=0.035 * h,
            ry_m=old_ry,
            rz_m=0.028 * h,
        ),
    ]
    report = _girdle_report(height_m=h, shoulder_y=0.0)
    m = _MetricsStub(height_m=h, shoulder_hw=0.20)
    messages: list[str] = []
    _apply_shoulder_girdle_softs(parts, report, m, messages)  # type: ignore[arg-type]
    for t in parts:
        assert t.center is not None and t.ry_m is not None
        assert float(t.ry_m) >= TRAP_RY_FLOOR_FRAC_H * h - 1e-9
        expected_y = abs(float(t.ry_m)) * TRAP_Y_BACK_FRAC_RY
        assert float(t.center[1]) == pytest.approx(expected_y, abs=1e-5)
        assert float(t.center[1]) > pre_y  # grew with floored ry


def test_t9_scap_soft_axes_unchanged() -> None:
    """T9: scap_soft axes/center unchanged by girdle helper."""
    scap_l = _part(
        "RECIPE_scap_soft_l",
        role="scap_soft",
        center=[-0.08, 0.04, 1.30],
        rx_m=0.0688,
        ry_m=0.0258,
        rz_m=0.0774,
    )
    scap_r = _part(
        "RECIPE_scap_soft_r",
        role="scap_soft",
        center=[0.08, 0.04, 1.30],
        rx_m=0.0688,
        ry_m=0.0258,
        rz_m=0.0774,
    )
    clav = _part(
        "RECIPE_clavicle_r",
        kind="capsule",
        role="clavicle",
        p0=[0.20, -0.04, 1.38],
        p1=[0.0, 0.0, 1.27],
        radius_m=0.01,
    )
    trap = _part(
        "RECIPE_trap_soft_r",
        role="trap_soft",
        center=[0.10, 0.01, 1.38],
        rx_m=0.03,
        ry_m=0.01,
        rz_m=0.02,
    )
    parts = [scap_l, scap_r, clav, trap]
    before = [
        (list(s.center) if s.center else None, s.rx_m, s.ry_m, s.rz_m) for s in (scap_l, scap_r)
    ]
    report = _girdle_report()
    m = _MetricsStub(height_m=1.72)
    messages: list[str] = []
    _apply_shoulder_girdle_softs(parts, report, m, messages)  # type: ignore[arg-type]
    after = [
        (list(s.center) if s.center else None, s.rx_m, s.ry_m, s.rz_m) for s in (scap_l, scap_r)
    ]
    assert before == after


def test_t10_delt_fence_0060() -> None:
    """T10: 0060 fence — delt ry/rx follows DELT_RY_FRAC; bury messages when UA present."""
    pkg = _build_profile_pkg(limbs=True)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    for d in delts:
        assert d.rx_m is not None and d.ry_m is not None
        assert float(d.ry_m) / float(d.rx_m) == pytest.approx(DELT_RY_FRAC, abs=1e-6)
    bury_msgs = [m for m in pkg.messages if "socket bury" in m]
    assert len(bury_msgs) >= 2


def test_t11_trap_lat_matches_frac() -> None:
    """T11: re-assert TRAP_LAT_FRAC * |sh_x| (anatomy_profile retarget cousin)."""
    sh_x = 0.20
    pkg = _build_profile_pkg(shoulder_x=sh_x)
    for side, sign in (("l", -1.0), ("r", 1.0)):
        trap = next(p for p in pkg.parts if p.name == f"RECIPE_trap_soft_{side}")
        assert trap.center is not None
        assert float(trap.center[0]) == pytest.approx(sign * TRAP_LAT_FRAC * sh_x, abs=1e-5)


def test_t12_quiet_skip_no_profile() -> None:
    """T12: limbs-only no profile — no clavicle/trap; no girdle crash/spam."""
    report = _girdle_report()
    pkg = build_blockout_recipe(report, limbs=True, profile=None)
    roles = {p.role for p in pkg.parts}
    assert "clavicle" not in roles
    assert "trap_soft" not in roles
    assert not any("radius floor" in m for m in pkg.messages)
    assert not any(m.startswith("trap_soft_") and "floors" in m for m in pkg.messages)


def test_t13_zero_length_and_missing_capsule_skip() -> None:
    """T13: zero-length / missing capsule skip does not crash; restores ends+radius."""
    # H=None → no Z drop; coincident ends stay zero-length after mutate → restore.
    parts = [
        _part(
            "RECIPE_clavicle_l",
            kind="capsule",
            role="clavicle",
            p0=[0.0, -0.08, 1.38],
            p1=[0.0, -0.08, 1.38],
            radius_m=0.005,
        ),
        _part(
            "RECIPE_clavicle_r",
            kind="ellipsoid",  # wrong kind
            role="clavicle",
            center=[0.1, 0.0, 1.38],
            rx_m=0.01,
            ry_m=0.01,
            rz_m=0.01,
        ),
    ]
    report = _girdle_report()
    m = _MetricsStub(height_m=None)
    messages: list[str] = []
    _apply_shoulder_girdle_softs(parts, report, m, messages)  # type: ignore[arg-type]
    clav_l = parts[0]
    assert clav_l.p0 is not None and clav_l.p1 is not None
    assert clav_l.p0 == [0.0, -0.08, 1.38]
    assert clav_l.p1 == [0.0, -0.08, 1.38]
    assert any("zero length" in msg for msg in messages)
    assert any("not capsule" in msg for msg in messages)

    # Tiny finite H: floor/Z-drop near zero so coincident ends stay zero-length;
    # restore must include radius (Codex P3 — floor applied before length check).
    parts_h = [
        _part(
            "RECIPE_clavicle_l",
            kind="capsule",
            role="clavicle",
            p0=[0.0, -0.08, 1.38],
            p1=[0.0, -0.08, 1.38],
            radius_m=0.005,
        ),
    ]
    m_h = _MetricsStub(height_m=1e-20)
    messages_h: list[str] = []
    _apply_shoulder_girdle_softs(parts_h, report, m_h, messages_h)  # type: ignore[arg-type]
    assert parts_h[0].radius_m == pytest.approx(0.005, abs=1e-12)
    assert parts_h[0].p0 == [0.0, -0.08, 1.38]
    assert parts_h[0].p1 == [0.0, -0.08, 1.38]
    assert any("zero length" in msg for msg in messages_h)


def test_t14_clavicle_front_shelf_axial_exempt() -> None:
    """T14: 0061 front-shelf clavicle is axial-exempt; C_axial_depth_plane passes."""
    from meshops.proportion.constraints import validate_constraints

    report = _girdle_report(chest_front_y=-0.13, shoulder_y=-0.04)
    # chest_mid at 0 so shelf clavicle would fail without exempt
    report.landmarks_xyz["chest_mid"] = _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25)
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(report, limbs=True, profile=profile, torso="ovals")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    axial = by_id["C_axial_depth_plane"]
    assert axial.status == "pass", axial.message
    assert axial.metrics is not None
    assert axial.metrics.get("RECIPE_clavicle_l_axial_exempt") is True
    assert axial.metrics.get("RECIPE_clavicle_r_axial_exempt") is True
    # Synthetic report may fail unrelated hard rules (e.g. C_thigh_outer);
    # 0061 gate is axial pass + clavicle exempt metrics.


def test_helpers_neck_and_shelf() -> None:
    """Helper smoke: neck upper from RECIPE_neck.p1; shelf from chest_front."""
    parts = [
        _part(
            "RECIPE_neck",
            kind="cylinder",
            role="neck",
            p0=[0.0, 0.0, 1.40],
            p1=[0.0, -0.02, 1.55],
            radius_m=0.04,
        ),
        _part(
            "RECIPE_torso_oval_chest",
            role="torso",
            center=[0.0, 0.03, 1.25],
            rx_m=0.16,
            ry_m=0.11,
            rz_m=0.08,
        ),
    ]
    assert _neck_upper_z(parts) == pytest.approx(1.55, abs=1e-9)
    report = _girdle_report(chest_front_y=-0.0797)
    m = _MetricsStub()
    shelf = _chest_front_y_for_girdle(report, m, parts)  # type: ignore[arg-type]
    assert shelf == pytest.approx(-0.0797, abs=1e-9)
    # Without landmark: oval front = center.y - ry
    report2 = _girdle_report()
    report2.landmarks_xyz.pop("chest_front", None)
    shelf2 = _chest_front_y_for_girdle(report2, m, parts)  # type: ignore[arg-type]
    assert shelf2 == pytest.approx(0.03 - 0.11, abs=1e-9)


def test_messages_optional_substrings() -> None:
    """Optional soft: girdle messages mention radius floor / floors."""
    pkg = _build_profile_pkg()
    joined = "\n".join(pkg.messages)
    assert "radius floor" in joined
    assert "floors" in joined
