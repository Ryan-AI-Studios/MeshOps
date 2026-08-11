"""Track 0066 — torso back scap plane (scap_soft plate + rear past chest oval)."""

from __future__ import annotations

from typing import Any

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    SCAP_LAT_FRAC,
    SCAP_REAR_PAST_M,
    SCAP_RX_MIN_FRAC_H,
    SCAP_RY_FRAC_RX,
    SCAP_RY_MIN_FRAC_H,
    SCAP_RZ_FRAC_RX,
    SCAP_Z_DROP_FRAC_H,
    RecipePart,
    _apply_scap_plane,
    _ResolvedMetrics,
    build_blockout_recipe,
)
from meshops.proportion.constraints import (
    _AXIAL_EXEMPT_NAME_TOKENS,
    _axial_name_exempt,
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


def _part(
    name: str,
    *,
    kind: str = "ellipsoid",
    role: str = "scap_soft",
    center: list[float] | None = None,
    rx_m: float | None = None,
    ry_m: float | None = None,
    rz_m: float | None = None,
) -> RecipePart:
    kwargs: dict[str, Any] = {
        "name": name,
        "role": role,
        "kind": kind,
        "center": center,
        "rx_m": rx_m,
        "ry_m": ry_m,
        "rz_m": rz_m,
    }
    clean = {k: v for k, v in kwargs.items() if v is not None or k in ("name", "role", "kind")}
    return RecipePart.model_validate(clean)


def _empty_metrics(
    *,
    height_m: float = 1.72,
    shoulder_hw: float | None = 0.2575,
    shoulder_z: float | None = 1.3802,
    chest_half_depth: float | None = 0.12,
) -> _ResolvedMetrics:
    m = _ResolvedMetrics()
    m.height_m = height_m
    m.shoulder_hw = shoulder_hw
    m.shoulder_z = shoulder_z
    m.chest_half_depth = chest_half_depth
    return m


def _product_like_scaps(
    *,
    rx: float = 0.0688,
    ry: float = 0.0258,
    rz: float = 0.0774,
    cy: float = 0.0103,
    cz: float = 1.2699,
    cx: float = 0.1159,
) -> list[RecipePart]:
    return [
        _part(
            "RECIPE_scap_soft_l",
            role="scap_soft",
            center=[-cx, cy, cz],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
        ),
        _part(
            "RECIPE_scap_soft_r",
            role="scap_soft",
            center=[cx, cy, cz],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
        ),
    ]


def _chest_oval(
    *,
    cy: float = 0.0310,
    ry: float = 0.1108,
    cz: float = 1.308,
    rx: float = 0.236,
    rz: float = 0.105,
) -> RecipePart:
    return _part(
        "RECIPE_torso_oval_chest",
        role="torso",
        center=[0.0, cy, cz],
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
    )


def _rich_report(
    *,
    height_m: float = 1.72,
    shoulder_hw: float = 0.2575,
    shoulder_z: float = 1.3802,
) -> ProportionReport:
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "shoulder_l": _lm("shoulder_l", x_m=-shoulder_hw, y_m=0.0, z_m=shoulder_z),
        "shoulder_r": _lm("shoulder_r", x_m=shoulder_hw, y_m=0.0, z_m=shoulder_z),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=height_m),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=-0.05, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.28, y_m=-0.05, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=0.0, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=0.0, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.08, z_m=1.25),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.05),
        _diam("upper_arm_r", half_width_m=0.05),
        _diam("forearm_l", half_width_m=0.04),
        _diam("forearm_r", half_width_m=0.04),
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
    shoulder_hw: float = 0.2575,
    shoulder_z: float = 1.3802,
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
        j("spine_high", x=0.0, y=0.0, z=1.25, parent="pelvis"),
        j("neck_base", x=0.0, y=0.0, z=1.42, parent="spine_high"),
        j(
            "shoulder_l",
            x=-shoulder_hw,
            y=0.0,
            z=shoulder_z,
            side="l",
            parent="spine_high",
        ),
        j(
            "shoulder_r",
            x=shoulder_hw,
            y=0.0,
            z=shoulder_z,
            side="r",
            parent="spine_high",
        ),
        j("elbow_l", x=-0.28, y=0.0, z=1.10, side="l", parent="shoulder_l"),
        j("elbow_r", x=0.28, y=0.0, z=1.10, side="r", parent="shoulder_r"),
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


def _empty_report() -> ProportionReport:
    return ProportionReport(
        schema_version="1.0.0",
        height_m=1.72,
        landmarks_xyz={},
        diameters=[],
        quality=QualityFlags(),
    )


# ---------------------------------------------------------------------------
# T0 constants
# ---------------------------------------------------------------------------


def test_t0_constants_freeze() -> None:
    """T0: 0066 named constant freezes (P2-2 floor 0.040 not 0.048)."""
    assert SCAP_RX_MIN_FRAC_H == 0.040
    assert SCAP_RY_FRAC_RX == 0.42
    assert SCAP_RY_MIN_FRAC_H == 0.016
    assert SCAP_RZ_FRAC_RX == 1.15
    assert SCAP_REAR_PAST_M == 0.012
    assert SCAP_LAT_FRAC == 0.45
    assert SCAP_Z_DROP_FRAC_H == 0.055


# ---------------------------------------------------------------------------
# T1-T6 unit helper
# ---------------------------------------------------------------------------


def test_t1_rear_outer_past_chest() -> None:
    """T1: outer rear cy+ry >= chest_rear + past - eps."""
    chest = _chest_oval(cy=0.0310, ry=0.1108)
    chest_rear = 0.0310 + 0.1108
    parts = [*_product_like_scaps(), chest]
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(), msgs)
    scaps = [p for p in parts if p.role == "scap_soft"]
    assert len(scaps) == 2
    for s in scaps:
        assert s.center is not None and s.ry_m is not None
        outer = float(s.center[1]) + float(s.ry_m)
        assert outer + 1e-9 >= chest_rear + SCAP_REAR_PAST_M - 1e-6


def test_t2_plate_ratios() -> None:
    """T2: ry/rx ≈ 0.42; rz/rx ≈ 1.15."""
    parts = [*_product_like_scaps(), _chest_oval()]
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(), msgs)
    for s in parts:
        if s.role != "scap_soft":
            continue
        assert s.rx_m is not None and s.ry_m is not None and s.rz_m is not None
        rx = float(s.rx_m)
        assert float(s.ry_m) / rx == pytest.approx(SCAP_RY_FRAC_RX, rel=1e-6)
        assert float(s.rz_m) / rx == pytest.approx(SCAP_RZ_FRAC_RX, rel=1e-6)


def test_t3_dual_lr_equal_axes_mirror_x() -> None:
    """T3: dual L/R equal axes; mirror X."""
    parts = [*_product_like_scaps(), _chest_oval()]
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(), msgs)
    sl = next(p for p in parts if p.name == "RECIPE_scap_soft_l")
    sr = next(p for p in parts if p.name == "RECIPE_scap_soft_r")
    assert sl.center is not None and sr.center is not None
    assert float(sl.rx_m or 0.0) == pytest.approx(float(sr.rx_m or 0.0), abs=1e-12)
    assert float(sl.ry_m or 0.0) == pytest.approx(float(sr.ry_m or 0.0), abs=1e-12)
    assert float(sl.rz_m or 0.0) == pytest.approx(float(sr.rz_m or 0.0), abs=1e-12)
    assert float(sl.center[0]) == pytest.approx(-float(sr.center[0]), abs=1e-12)
    assert float(sl.center[1]) == pytest.approx(float(sr.center[1]), abs=1e-12)
    assert float(sl.center[2]) == pytest.approx(float(sr.center[2]), abs=1e-12)
    assert float(sl.center[0]) < 0.0 < float(sr.center[0])


def test_t4_lateral_frac() -> None:
    """T4: |cx| = 0.45 * shoulder_hw."""
    sh = 0.2575
    parts = [*_product_like_scaps(), _chest_oval()]
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(shoulder_hw=sh), msgs)
    expected = SCAP_LAT_FRAC * sh
    for s in parts:
        if s.role != "scap_soft":
            continue
        assert s.center is not None
        assert abs(float(s.center[0])) == pytest.approx(expected, abs=1e-9)


def test_t5_z_drop_when_shoulder_z() -> None:
    """T5: Z = shoulder_z - 0.055*H when shoulder_z present."""
    h = 1.72
    sz = 1.3802
    parts = [*_product_like_scaps(), _chest_oval()]
    msgs: list[str] = []
    _apply_scap_plane(
        parts,
        _empty_report(),
        _empty_metrics(height_m=h, shoulder_z=sz),
        msgs,
    )
    expected_z = sz - SCAP_Z_DROP_FRAC_H * h
    for s in parts:
        if s.role != "scap_soft":
            continue
        assert s.center is not None
        assert float(s.center[2]) == pytest.approx(expected_z, abs=1e-9)


def test_t6_fence_trap_chest_delt() -> None:
    """T6: trap / chest oval / delt unchanged by scap helper."""
    trap = _part(
        "RECIPE_trap_soft_r",
        role="trap_soft",
        center=[0.142, 0.029, 1.397],
        rx_m=0.072,
        ry_m=0.038,
        rz_m=0.065,
    )
    delt = _part(
        "RECIPE_deltoid_soft_r",
        role="deltoid_soft",
        center=[0.22, -0.04, 1.38],
        rx_m=0.05,
        ry_m=0.036,
        rz_m=0.039,
    )
    chest = _chest_oval()
    scaps = _product_like_scaps()
    parts = [*scaps, trap, delt, chest]
    pre = {
        p.name: (list(p.center) if p.center else None, p.rx_m, p.ry_m, p.rz_m)
        for p in (trap, delt, chest)
    }
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(), msgs)
    for name, snap in pre.items():
        p = next(x for x in parts if x.name == name)
        assert (list(p.center) if p.center else None, p.rx_m, p.ry_m, p.rz_m) == snap


# ---------------------------------------------------------------------------
# T7-T10 profile / fallback / messages / role gate
# ---------------------------------------------------------------------------


def test_t7_male_and_female_athletic_apply() -> None:
    """T7: male + female athletic profiles both apply scap plane."""
    for pid in ("torso_limb_f_athletic_v1", "torso_limb_m_athletic_v1"):
        profile = load_anatomy_profile(pid)
        pkg = build_blockout_recipe(
            _rich_report(),
            limbs=True,
            profile=profile,
            skeleton=_skeleton_with_arms(),
            torso="ovals",
        )
        scaps = [p for p in pkg.parts if p.role == "scap_soft"]
        assert len(scaps) == 2, pid
        assert any("scap_plane_applied" in m for m in pkg.messages), pid
        for s in scaps:
            assert s.center is not None and s.rx_m is not None and s.ry_m is not None
            assert float(s.center[1]) > 0.05  # not buried y_back 0.4*ry


def test_t8_no_chest_oval_fallback_cy() -> None:
    """T8: no chest oval -> cy > 0.4*ry (P3-5 weaker fallback still lifts)."""
    # Buried emit Y like product y_back: cy = 0.4 * ry_pre
    pre_ry = 0.0258
    parts = _product_like_scaps(cy=0.4 * pre_ry, ry=pre_ry)
    msgs: list[str] = []
    m = _empty_metrics(chest_half_depth=0.12)
    _apply_scap_plane(parts, _empty_report(), m, msgs)
    for s in parts:
        assert s.center is not None and s.ry_m is not None
        cy = float(s.center[1])
        ry = float(s.ry_m)
        assert cy > 0.4 * ry


def test_t9_messages_include_keys() -> None:
    """T9: messages include scap_plane / past / chest_rear."""
    parts = [*_product_like_scaps(), _chest_oval()]
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(), msgs)
    blob = " ".join(msgs)
    assert "scap_plane" in blob
    assert "past" in blob or "past_m" in blob
    assert "chest_rear" in blob
    assert "lat_frac" in blob
    assert "z_drop" in blob
    assert any(m == "scap_plane_applied: true" for m in msgs)


def test_t10_role_gate_breast_glute_ignored() -> None:
    """T10: breast/glute ignored by role gate."""
    breast = _part(
        "RECIPE_breast_soft_l",
        role="breast_soft",
        center=[-0.06, -0.05, 1.25],
        rx_m=0.07,
        ry_m=0.06,
        rz_m=0.07,
    )
    glute = _part(
        "RECIPE_glute_soft_l",
        role="glute_soft",
        center=[-0.08, 0.04, 0.90],
        rx_m=0.07,
        ry_m=0.09,
        rz_m=0.07,
    )
    parts = [*_product_like_scaps(), breast, glute, _chest_oval()]
    pre_b = (list(breast.center) if breast.center else None, breast.rx_m, breast.ry_m, breast.rz_m)
    pre_g = (list(glute.center) if glute.center else None, glute.rx_m, glute.ry_m, glute.rz_m)
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(), msgs)
    b = next(p for p in parts if p.role == "breast_soft")
    g = next(p for p in parts if p.role == "glute_soft")
    assert (list(b.center) if b.center else None, b.rx_m, b.ry_m, b.rz_m) == pre_b
    assert (list(g.center) if g.center else None, g.rx_m, g.ry_m, g.rz_m) == pre_g


# ---------------------------------------------------------------------------
# T11-T14 integration / equalize / B15 / pack floor
# ---------------------------------------------------------------------------


def test_t11_integration_synthetic_build() -> None:
    """T11: synthetic build_blockout_recipe with torso ovals + F profile."""
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(
        _rich_report(),
        limbs=True,
        profile=profile,
        skeleton=_skeleton_with_arms(),
        torso="ovals",
    )
    chest = next((p for p in pkg.parts if p.name == "RECIPE_torso_oval_chest"), None)
    assert chest is not None and chest.center is not None and chest.ry_m is not None
    chest_rear = float(chest.center[1]) + float(chest.ry_m)
    scaps = [p for p in pkg.parts if p.role == "scap_soft"]
    assert len(scaps) == 2
    for s in scaps:
        assert s.center is not None and s.ry_m is not None and s.rx_m is not None
        outer = float(s.center[1]) + float(s.ry_m)
        assert outer + 1e-9 >= chest_rear + SCAP_REAR_PAST_M - 2e-3
        rx = float(s.rx_m)
        assert float(s.ry_m) / rx == pytest.approx(SCAP_RY_FRAC_RX, rel=1e-4)
        assert float(s.rz_m or 0.0) / rx == pytest.approx(SCAP_RZ_FRAC_RX, rel=1e-4)


def test_t12_b5_equalize_after_asymmetric_pre() -> None:
    """T12: B5 equalize after asymmetric pre axes/centers."""
    parts = [
        _part(
            "RECIPE_scap_soft_l",
            role="scap_soft",
            center=[-0.10, 0.01, 1.25],
            rx_m=0.060,
            ry_m=0.020,
            rz_m=0.070,
        ),
        _part(
            "RECIPE_scap_soft_r",
            role="scap_soft",
            center=[0.14, 0.02, 1.30],
            rx_m=0.080,
            ry_m=0.030,
            rz_m=0.090,
        ),
        _chest_oval(),
    ]
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(), msgs)
    sl = next(p for p in parts if p.name == "RECIPE_scap_soft_l")
    sr = next(p for p in parts if p.name == "RECIPE_scap_soft_r")
    assert sl.center is not None and sr.center is not None
    assert float(sl.rx_m or 0.0) == pytest.approx(float(sr.rx_m or 0.0), abs=1e-12)
    assert float(sl.ry_m or 0.0) == pytest.approx(float(sr.ry_m or 0.0), abs=1e-12)
    assert float(sl.rz_m or 0.0) == pytest.approx(float(sr.rz_m or 0.0), abs=1e-12)
    assert abs(float(sl.center[0])) == pytest.approx(abs(float(sr.center[0])), abs=1e-12)
    assert float(sl.center[1]) == pytest.approx(float(sr.center[1]), abs=1e-12)
    assert float(sl.center[2]) == pytest.approx(float(sr.center[2]), abs=1e-12)


def test_t13_b15_scap_axial_exempt() -> None:
    """T13: B15 scap name is axial-exempt via constraints helper."""
    assert "scap" in _AXIAL_EXEMPT_NAME_TOKENS
    assert _axial_name_exempt("RECIPE_scap_soft_l") is True
    assert _axial_name_exempt("RECIPE_scap_soft_r") is True
    assert _axial_name_exempt("RECIPE_torso_oval_chest") is False


def test_t14_b1_floor_pack_rx_unchanged() -> None:
    """T14: B1 floor 0.040 — pack-class rx 0.0688 unchanged (not forced to 0.0826)."""
    h = 1.72
    pack_rx = 0.0688
    # Rejected draft floor 0.048 would force max(0.0688, 0.08256)=0.08256
    assert SCAP_RX_MIN_FRAC_H * h == pytest.approx(0.0688, abs=1e-9)
    assert SCAP_RX_MIN_FRAC_H < 0.048
    parts = [*_product_like_scaps(rx=pack_rx), _chest_oval()]
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(height_m=h), msgs)
    for s in parts:
        if s.role != "scap_soft":
            continue
        assert s.rx_m is not None
        assert float(s.rx_m) == pytest.approx(pack_rx, abs=1e-9)
        assert float(s.rx_m) < 0.0820  # never draft 0.048 floor growth


def test_quiet_skip_no_scap() -> None:
    """Quiet skip when no scap_soft present."""
    parts = [
        _part(
            "RECIPE_trap_soft_r",
            role="trap_soft",
            center=[0.1, 0.02, 1.4],
            rx_m=0.05,
            ry_m=0.03,
            rz_m=0.04,
        )
    ]
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(), msgs)
    assert msgs == []
    assert parts[0].center == [0.1, 0.02, 1.4]
