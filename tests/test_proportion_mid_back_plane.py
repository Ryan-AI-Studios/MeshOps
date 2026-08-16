"""Track 0074 — torso mid-back plane (mid_back_soft + waist/hip rear bias)."""

from __future__ import annotations

from typing import Any

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    MID_BACK_BELOW_SCAP_M,
    MID_BACK_LAT_FRAC,
    MID_BACK_REAR_PAST_M,
    MID_BACK_RX_MIN_FRAC_H,
    MID_BACK_RY_FRAC_RX,
    MID_BACK_RY_MIN_FRAC_H,
    MID_BACK_RZ_FRAC_RX,
    MID_BACK_Z_DROP_FRAC_H,
    RECIPE_HONESTY,
    SCAP_REAR_PAST_M,
    TORSO_HIP_Y_REAR_BIAS_FRAC_RY,
    TORSO_WAIST_Y_REAR_BIAS_FRAC_RY,
    RecipePart,
    _apply_mid_back_plane,
    _apply_scap_plane,
    _build_torso_ovals,
    _ResolvedMetrics,
    build_blockout_recipe,
)
from meshops.proportion.constraints import (
    _AXIAL_EXEMPT_NAME_TOKENS,
    _axial_name_exempt,
    classify_part_name,
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
    role: str = "mid_back_soft",
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


def _product_like_mid_backs(
    *,
    rx: float = 0.0654,
    ry: float = 0.0241,
    rz: float = 0.0800,  # below B3 rz floor so ratio law wins (like scap T2)
    cy: float = 0.0100,
    cz: float = 1.1411,
    cx: float = 0.0900,
) -> list[RecipePart]:
    return [
        _part(
            "RECIPE_mid_back_soft_l",
            role="mid_back_soft",
            center=[-cx, cy, cz],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
        ),
        _part(
            "RECIPE_mid_back_soft_r",
            role="mid_back_soft",
            center=[cx, cy, cz],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
        ),
    ]


def _waist_oval(
    *,
    cy: float = 0.0318,
    ry: float = 0.0756,
    cz: float = 1.1411,
    rx: float = 0.1816,
    rz: float = 0.1065,
) -> RecipePart:
    return _part(
        "RECIPE_torso_oval_waist",
        role="torso",
        center=[0.0, cy, cz],
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
    )


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


def _product_like_scaps(
    *,
    rx: float = 0.0688,
    ry: float = 0.0289,
    rz: float = 0.0791,
    cy: float = 0.1249,
    cz: float = 1.2856,
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
        j("spine_mid", x=0.0, y=0.0, z=1.14, parent="pelvis"),
        j("spine_high", x=0.0, y=0.0, z=1.25, parent="spine_mid"),
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


def _metrics_for_ovals(*, chest_y: float | None = 0.0) -> _ResolvedMetrics:
    m = _ResolvedMetrics()
    m.height_m = 1.72
    m.shoulder_hw = 0.25
    m.hip_hw = 0.20
    m.shoulder_z = 1.38
    m.hip_z = 0.95
    m.chest_half_depth = 0.12
    m.hip_half_depth = 0.13
    m.chest_y = chest_y
    m.hip_y = chest_y if chest_y is not None else None
    return m


# ---------------------------------------------------------------------------
# T0 constants
# ---------------------------------------------------------------------------


def test_t0_constants_freeze() -> None:
    """T0: 0074 named constant freezes match plan B2-B8 / B17."""
    assert MID_BACK_REAR_PAST_M == 0.022
    assert MID_BACK_RX_MIN_FRAC_H == 0.038
    assert MID_BACK_RY_FRAC_RX == 0.38
    assert MID_BACK_RY_MIN_FRAC_H == 0.014
    assert MID_BACK_RZ_FRAC_RX == 1.30
    assert MID_BACK_LAT_FRAC == 0.38
    assert MID_BACK_Z_DROP_FRAC_H == 0.14
    assert MID_BACK_BELOW_SCAP_M == 0.008
    assert TORSO_WAIST_Y_REAR_BIAS_FRAC_RY == 0.42
    assert TORSO_HIP_Y_REAR_BIAS_FRAC_RY == 0.33


# ---------------------------------------------------------------------------
# T1 dual emit (F+M athletic profiles)
# ---------------------------------------------------------------------------


def test_t1_profiles_emit_dual_mid_back() -> None:
    """T1: F+M athletic profiles → dual RECIPE_mid_back_soft_{l,r} role mid_back_soft."""
    for pid in ("torso_limb_f_athletic_v1", "torso_limb_m_athletic_v1"):
        profile = load_anatomy_profile(pid)
        pkg = build_blockout_recipe(
            _rich_report(),
            limbs=True,
            profile=profile,
            skeleton=_skeleton_with_arms(),
            torso="ovals",
        )
        mbs = [p for p in pkg.parts if p.role == "mid_back_soft"]
        assert len(mbs) == 2, pid
        names = {p.name for p in mbs}
        assert names == {"RECIPE_mid_back_soft_l", "RECIPE_mid_back_soft_r"}, pid
        for p in mbs:
            assert p.role == "mid_back_soft"
            assert p.kind == "ellipsoid"


# ---------------------------------------------------------------------------
# T2-T7 unit helper path
# ---------------------------------------------------------------------------


def test_t2_outer_rear_past_waist() -> None:
    """T2: outer_rear >= waist_rear + past - eps (post-B7 waist)."""
    waist = _waist_oval(cy=0.0318, ry=0.0756)
    waist_rear = 0.0318 + 0.0756
    parts = [*_product_like_mid_backs(), waist, *_product_like_scaps()]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    mbs = [p for p in parts if p.role == "mid_back_soft"]
    assert len(mbs) == 2
    for mb in mbs:
        assert mb.center is not None and mb.ry_m is not None
        outer = float(mb.center[1]) + float(mb.ry_m)
        assert outer + 1e-9 >= waist_rear + MID_BACK_REAR_PAST_M - 1e-6


def test_t3_plate_ratios() -> None:
    """T3: ry/rx ≈ 0.38; rz/rx ≈ 1.30."""
    parts = [*_product_like_mid_backs(), _waist_oval(), *_product_like_scaps()]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    for p in parts:
        if p.role != "mid_back_soft":
            continue
        assert p.rx_m is not None and p.ry_m is not None and p.rz_m is not None
        rx = float(p.rx_m)
        assert float(p.ry_m) / rx == pytest.approx(MID_BACK_RY_FRAC_RX, rel=1e-6)
        assert float(p.rz_m) / rx == pytest.approx(MID_BACK_RZ_FRAC_RX, rel=1e-6)


def test_t4_lateral_frac() -> None:
    """T4: |cx| = shoulder_hw * 0.38."""
    sh = 0.2575
    parts = [*_product_like_mid_backs(), _waist_oval(), *_product_like_scaps()]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(shoulder_hw=sh), msgs)
    expected = MID_BACK_LAT_FRAC * sh
    for p in parts:
        if p.role != "mid_back_soft":
            continue
        assert p.center is not None
        assert abs(float(p.center[0])) == pytest.approx(expected, abs=1e-9)


def test_t5_z_from_waist_oval() -> None:
    """T5: z ≈ waist oval z when waist present (name SoT)."""
    waist_z = 1.1411
    parts = [
        *_product_like_mid_backs(cz=1.00),
        _waist_oval(cz=waist_z),
        *_product_like_scaps(),
    ]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    for p in parts:
        if p.role != "mid_back_soft":
            continue
        assert p.center is not None
        assert float(p.center[2]) == pytest.approx(waist_z, abs=1e-9)


def test_t6_dual_lr_equal() -> None:
    """T6: L/R equal axes and |cx|/cy/z."""
    parts = [
        _part(
            "RECIPE_mid_back_soft_l",
            role="mid_back_soft",
            center=[-0.08, 0.01, 1.10],
            rx_m=0.050,
            ry_m=0.018,
            rz_m=0.060,
        ),
        _part(
            "RECIPE_mid_back_soft_r",
            role="mid_back_soft",
            center=[0.12, 0.02, 1.20],
            rx_m=0.070,
            ry_m=0.028,
            rz_m=0.090,
        ),
        _waist_oval(),
        *_product_like_scaps(),
    ]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    ml = next(p for p in parts if p.name == "RECIPE_mid_back_soft_l")
    mr = next(p for p in parts if p.name == "RECIPE_mid_back_soft_r")
    assert ml.center is not None and mr.center is not None
    assert float(ml.rx_m or 0.0) == pytest.approx(float(mr.rx_m or 0.0), abs=1e-12)
    assert float(ml.ry_m or 0.0) == pytest.approx(float(mr.ry_m or 0.0), abs=1e-12)
    assert float(ml.rz_m or 0.0) == pytest.approx(float(mr.rz_m or 0.0), abs=1e-12)
    assert abs(float(ml.center[0])) == pytest.approx(abs(float(mr.center[0])), abs=1e-12)
    assert float(ml.center[1]) == pytest.approx(float(mr.center[1]), abs=1e-12)
    assert float(ml.center[2]) == pytest.approx(float(mr.center[2]), abs=1e-12)
    assert float(ml.center[0]) < 0.0 < float(mr.center[0])


def test_t7_anti_cape_vs_scap() -> None:
    """T7: outer_mb <= scap_outer - BELOW_SCAP + eps when scap present."""
    # Force mid_back that would cape past scap unless B17 pulls.
    scaps = _product_like_scaps(cy=0.10, ry=0.03)  # outer = 0.13
    scap_outer = 0.10 + 0.03
    waist = _waist_oval(cy=0.08, ry=0.08)  # rear 0.16 → would push outer high
    parts = [*_product_like_mid_backs(rx=0.08, ry=0.04), waist, *scaps]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    for p in parts:
        if p.role != "mid_back_soft":
            continue
        assert p.center is not None and p.ry_m is not None
        outer = float(p.center[1]) + float(p.ry_m)
        assert outer <= scap_outer - MID_BACK_BELOW_SCAP_M + 1e-6


def test_t7b_anti_cape_no_abs_reexpand() -> None:
    """T7b: anti-cape must not abs(cy) — negative pull would re-expand past cap."""
    # Tiny scap outer + large ry → cy = outer_cap - ry can go negative.
    # Old abs(cy) would flip positive and break outer <= scap_outer - margin.
    scaps = _product_like_scaps(cy=0.02, ry=0.01)  # outer = 0.03
    scap_outer = 0.03
    waist = _waist_oval(cy=0.05, ry=0.05)  # rear 0.10
    parts = [*_product_like_mid_backs(rx=0.10, ry=0.05), waist, *scaps]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    for p in parts:
        if p.role != "mid_back_soft":
            continue
        assert p.center is not None and p.ry_m is not None
        cy = float(p.center[1])
        ry = float(p.ry_m)
        outer = cy + ry
        assert outer <= scap_outer - MID_BACK_BELOW_SCAP_M + 1e-6
        # If abs had been applied, outer would be |cy|+ry and exceed cap.
        assert outer != abs(cy) + ry or cy >= 0.0


# ---------------------------------------------------------------------------
# T8-T11 integration / fence / messages
# ---------------------------------------------------------------------------


def test_t8_no_profile_no_mid_back() -> None:
    """T8: no profile → no mid_back parts; no hard fail."""
    pkg = build_blockout_recipe(
        _rich_report(),
        limbs=True,
        skeleton=_skeleton_with_arms(),
        torso="ovals",
    )
    mbs = [p for p in pkg.parts if p.role == "mid_back_soft" or "mid_back" in (p.name or "")]
    assert mbs == []
    assert not any("mid_back_plane_applied" in m for m in pkg.messages)


def test_t9_scap_still_past_chest_after_mid_back() -> None:
    """T9: scap outer still past chest (0066) after mid_back apply."""
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(
        _rich_report(),
        limbs=True,
        profile=profile,
        skeleton=_skeleton_with_arms(),
        torso="ovals",
    )
    chest = next(p for p in pkg.parts if p.name == "RECIPE_torso_oval_chest")
    assert chest.center is not None and chest.ry_m is not None
    chest_rear = float(chest.center[1]) + float(chest.ry_m)
    scaps = [p for p in pkg.parts if p.role == "scap_soft"]
    assert len(scaps) == 2
    for s in scaps:
        assert s.center is not None and s.ry_m is not None
        outer = float(s.center[1]) + float(s.ry_m)
        assert outer + 1e-9 >= chest_rear + SCAP_REAR_PAST_M - 2e-3


def test_t10_waist_hip_rear_bias_full3d_and_front_plane() -> None:
    """T10: full3d waist/hip cy = y + bias*ry; front_plane no bias."""
    # full3d path via metrics with chest_y set
    msgs_f: list[str] = []
    m_f = _metrics_for_ovals(chest_y=0.0)
    parts_f = _build_torso_ovals(m_f, msgs_f, taper=0.14)
    by_f = {p.name: p for p in parts_f}
    waist_f = by_f["RECIPE_torso_oval_waist"]
    hip_f = by_f["RECIPE_torso_oval_hip"]
    assert waist_f.center is not None and waist_f.ry_m is not None
    assert hip_f.center is not None and hip_f.ry_m is not None
    y = 0.0
    assert waist_f.center[1] == pytest.approx(
        y + TORSO_WAIST_Y_REAR_BIAS_FRAC_RY * float(waist_f.ry_m), abs=1e-9
    )
    assert hip_f.center[1] == pytest.approx(
        y + TORSO_HIP_Y_REAR_BIAS_FRAC_RY * float(hip_f.ry_m), abs=1e-9
    )
    assert any("torso mid-back:" in m for m in msgs_f)

    # front_plane: no chest_y → placement front_plane, no waist/hip bias
    msgs_p: list[str] = []
    m_p = _metrics_for_ovals(chest_y=None)
    parts_p = _build_torso_ovals(m_p, msgs_p, taper=0.14)
    by_p = {p.name: p for p in parts_p}
    waist_p = by_p["RECIPE_torso_oval_waist"]
    hip_p = by_p["RECIPE_torso_oval_hip"]
    assert waist_p.center is not None and hip_p.center is not None
    assert waist_p.placement == "front_plane"
    assert hip_p.placement == "front_plane"
    assert waist_p.center[1] == pytest.approx(0.0, abs=1e-9)
    assert hip_p.center[1] == pytest.approx(0.0, abs=1e-9)


def test_t11_messages_include_keys() -> None:
    """T11: messages contain mid_back_plane_applied + past/lat (B14)."""
    parts = [*_product_like_mid_backs(), _waist_oval(), *_product_like_scaps()]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    blob = " ".join(msgs)
    assert any(m == "mid_back_plane_applied: true" for m in msgs)
    assert f"mid_back_plane_past_m={MID_BACK_REAR_PAST_M}" in msgs
    assert f"mid_back_plane_lat_frac={MID_BACK_LAT_FRAC}" in msgs
    assert "mid_back_soft_l:" in blob
    assert "mid_back_soft_r:" in blob
    assert "outer_rear=" in blob


# ---------------------------------------------------------------------------
# T12-T13 exports / axial / honesty
# ---------------------------------------------------------------------------


def test_t12_all_exports_and_honesty() -> None:
    """T12: __all__ B16 exports + honesty has no photoreal/cape claims."""
    import meshops.proportion.blockout_recipe as br

    exports = set(br.__all__)
    required = {
        "MID_BACK_REAR_PAST_M",
        "MID_BACK_RX_MIN_FRAC_H",
        "MID_BACK_RY_FRAC_RX",
        "MID_BACK_RY_MIN_FRAC_H",
        "MID_BACK_RZ_FRAC_RX",
        "MID_BACK_LAT_FRAC",
        "MID_BACK_Z_DROP_FRAC_H",
        "MID_BACK_BELOW_SCAP_M",
        "TORSO_WAIST_Y_REAR_BIAS_FRAC_RY",
        "TORSO_HIP_Y_REAR_BIAS_FRAC_RY",
        "_apply_mid_back_plane",
    }
    missing = required - exports
    assert not missing, f"__all__ missing: {sorted(missing)}"
    # ASCII alpha: MID_BACK_* after MIDLINE_X_TOL_M
    mid_idx = br.__all__.index("MIDLINE_X_TOL_M")
    first_mb = br.__all__.index("MID_BACK_BELOW_SCAP_M")
    assert first_mb > mid_idx
    assert br.__all__.index("TORSO_HIP_Y_REAR_BIAS_FRAC_RY") > br.__all__.index(
        "TORSO_CHEST_Y_REAR_BIAS_FRAC_RY"
    )
    assert br.__all__.index("TORSO_WAIST_Y_REAR_BIAS_FRAC_RY") > br.__all__.index(
        "TORSO_WAIST_RX_MAX_FRAC_CHEST"
    )
    honesty = RECIPE_HONESTY.lower()
    assert "photoreal" not in honesty
    assert "cape" not in honesty
    assert "spine groove" not in honesty


def test_t13_mid_back_axial_exempt() -> None:
    """T13: _axial_name_exempt(RECIPE_mid_back_soft_l) is True."""
    assert "mid_back" in _AXIAL_EXEMPT_NAME_TOKENS
    assert _axial_name_exempt("RECIPE_mid_back_soft_l") is True
    assert _axial_name_exempt("RECIPE_mid_back_soft_r") is True
    assert _axial_name_exempt("RECIPE_torso_oval_waist") is False
    role, _side = classify_part_name("RECIPE_mid_back_soft_l")
    assert role == "torso"


def test_quiet_skip_no_mid_back() -> None:
    """Quiet skip when no mid_back_soft present."""
    parts = [*_product_like_scaps(), _waist_oval()]
    msgs: list[str] = []
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    assert msgs == []


def test_fence_scap_helper_then_mid_back() -> None:
    """Order fence: scap plane then mid_back; scap outer still past chest."""
    chest = _chest_oval()
    chest_rear = float(chest.center[1]) + float(chest.ry_m)  # type: ignore[index]
    parts = [
        *_product_like_scaps(cy=0.01, ry=0.02),
        *_product_like_mid_backs(),
        chest,
        _waist_oval(),
    ]
    msgs: list[str] = []
    _apply_scap_plane(parts, _empty_report(), _empty_metrics(), msgs)
    _apply_mid_back_plane(parts, _empty_report(), _empty_metrics(), msgs)
    for s in parts:
        if s.role != "scap_soft":
            continue
        assert s.center is not None and s.ry_m is not None
        outer = float(s.center[1]) + float(s.ry_m)
        assert outer + 1e-9 >= chest_rear + SCAP_REAR_PAST_M - 1e-6
