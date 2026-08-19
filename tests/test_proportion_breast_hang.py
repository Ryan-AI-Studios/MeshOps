"""Track 0049 — breast_soft vertical hang Z (authoring RECIPE only)."""

from __future__ import annotations

import math

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    BREAST_HANG_Z_DROP_FRAC_RZ,
    BREAST_HANG_Z_MIN_DROP_FRAC_RZ,
    RECIPE_SCHEMA_VERSION,
    RecipePart,
    _apply_breast_hang_z,
    _ResolvedMetrics,
    build_blockout_recipe,
)
from meshops.proportion.models import (
    CrossSection,
    DepthBand,
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


def _base_lms(
    *,
    height_m: float = 1.72,
    extra: dict[str, LandmarkXYZ] | None = None,
) -> dict[str, LandmarkXYZ]:
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "hip_l": _lm("hip_l", x_m=-0.14, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.14, y_m=0.0, z_m=0.95),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
    }
    if extra:
        lms.update(extra)
    return lms


def _report_soft_cs(
    *,
    height_m: float = 1.72,
    extra_lms: dict[str, LandmarkXYZ] | None = None,
) -> ProportionReport:
    """Base CS path dual breast_soft (no profile)."""
    lms = _base_lms(height_m=height_m, extra=extra_lms)
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
    ]
    bands = [
        DepthBand(
            band_id="chest",
            depth_px=50.0,
            depth_frac=0.12,
            depth_m=0.24,
            y_front=0.1,
            y_back=-0.1,
            y_mid=0.0,
            z_frac=0.72,
        ),
        DepthBand(
            band_id="hip",
            depth_px=55.0,
            depth_frac=0.13,
            depth_m=0.26,
            y_front=0.1,
            y_back=-0.1,
            y_mid=0.0,
            z_frac=0.55,
        ),
    ]
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=bands,
        cross_sections=[
            CrossSection(
                level_id="bust",
                z_frac=0.72,
                rx_frac=0.10,
                ry_frac=0.08,
                sources=["test"],
            ),
            CrossSection(
                level_id="glute",
                z_frac=0.50,
                rx_frac=0.11,
                ry_frac=0.09,
                sources=["test"],
            ),
        ],
        quality=QualityFlags(),
    )


def _rich_report(
    *,
    height_m: float = 1.72,
    extra_lms: dict[str, LandmarkXYZ] | None = None,
) -> ProportionReport:
    """Profile path report (female dual breasts)."""
    lms = _base_lms(height_m=height_m, extra=extra_lms)
    lms["elbow_l"] = _lm("elbow_l", x_m=-0.28, y_m=-0.05, z_m=1.10)
    lms["elbow_r"] = _lm("elbow_r", x_m=0.28, y_m=-0.05, z_m=1.10)
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.05),
        _diam("upper_arm_r", half_width_m=0.05),
    ]
    bands = [
        DepthBand(
            band_id="chest",
            depth_px=50.0,
            depth_frac=0.12,
            depth_m=0.24,
            y_front=0.1,
            y_back=-0.1,
            y_mid=0.0,
            z_frac=0.72,
        ),
        DepthBand(
            band_id="breast",
            depth_px=40.0,
            depth_frac=0.10,
            depth_m=0.18,
            y_front=0.08,
            y_back=-0.05,
            y_mid=0.0,
            z_frac=0.70,
        ),
        DepthBand(
            band_id="hip",
            depth_px=55.0,
            depth_frac=0.13,
            depth_m=0.26,
            y_front=0.1,
            y_back=-0.1,
            y_mid=0.0,
            z_frac=0.55,
        ),
    ]
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=bands,
        quality=QualityFlags(),
    )


def _j(
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


def _skeleton_with_arms(*, spine_high_z: float = 1.27) -> BlockoutSkeleton:
    return BlockoutSkeleton(
        schema_version="1.0.0",
        honesty="proportion_blockout_skeleton_not_mesh_or_print_success",
        joints=[
            _j("root", x=0.0, y=0.0, z=0.0),
            _j("pelvis", x=0.0, y=0.0, z=0.95, parent="root"),
            _j("spine_high", x=0.0, y=0.0, z=spine_high_z, parent="pelvis"),
            _j("neck_base", x=0.0, y=0.0, z=1.42, parent="spine_high"),
            _j("shoulder_l", x=-0.20, y=0.0, z=1.38, side="l", parent="spine_high"),
            _j("shoulder_r", x=0.20, y=0.0, z=1.38, side="r", parent="spine_high"),
            _j("elbow_l", x=-0.28, y=-0.05, z=1.10, side="l", parent="shoulder_l"),
            _j("elbow_r", x=0.28, y=-0.05, z=1.10, side="r", parent="shoulder_r"),
        ],
        bones=[
            SkeletonBone(id="spine", joint_a="pelvis", joint_b="spine_high", length_m=0.3),
            SkeletonBone(id="upper_arm_l", joint_a="shoulder_l", joint_b="elbow_l", length_m=0.3),
            SkeletonBone(id="upper_arm_r", joint_a="shoulder_r", joint_b="elbow_r", length_m=0.3),
        ],
    )


def _dual_breast_parts(
    *,
    center_z: float = 1.27,
    rz: float = 0.08,
    rx: float = 0.06,
    ry: float = 0.05,
) -> list[RecipePart]:
    return [
        RecipePart(
            name="RECIPE_breast_soft_l",
            role="breast_soft",
            kind="ellipsoid",
            center=[-0.08, -0.06, center_z],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
            placement="full3d",
            label="RECIPE_breast_soft_l",
        ),
        RecipePart(
            name="RECIPE_breast_soft_r",
            role="breast_soft",
            kind="ellipsoid",
            center=[0.08, -0.06, center_z],
            rx_m=rx,
            ry_m=ry,
            rz_m=rz,
            placement="full3d",
            label="RECIPE_breast_soft_r",
        ),
    ]


def _empty_metrics(*, height_m: float = 1.72, chest_z: float | None = 1.31) -> _ResolvedMetrics:
    m = _ResolvedMetrics()
    m.height_m = height_m
    m.chest_z = chest_z
    return m


def _msg_value(messages: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for m in messages:
        if m.startswith(prefix):
            return m[len(prefix) :]
        if m == key:
            return ""
    return None


# ---------------------------------------------------------------------------
# T0-T5 product/profile path
# ---------------------------------------------------------------------------


def test_t0_pre_hang_upper_clavicle_class() -> None:
    """T0: pre-hang style upper = anchor+rz (clavicle class); post drops."""
    pre_z = 1.27
    rz = 0.0806
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    upper_pre = pre_z + rz
    # product_0036up class: upper near shoulder ~1.35
    assert upper_pre == pytest.approx(1.3506, abs=1e-3)

    report = _report_soft_cs()
    msgs: list[str] = []
    _apply_breast_hang_z(parts, report, _empty_metrics(chest_z=1.3085), msgs)

    for p in parts:
        assert p.center is not None
        assert p.center[2] < pre_z
        upper_post = p.center[2] + rz
        assert upper_post < upper_pre


def test_t1_min_hang_drop_vs_pre_anchor() -> None:
    """T1/D2: final center <= pre_anchor - MIN_DROP*rz when hung (no waist clamp)."""
    # Pure unit without torso waist oval — B1 always meets D2.
    pre_anchor = 1.27
    rz = 0.08
    parts = _dual_breast_parts(center_z=pre_anchor, rz=rz)
    msgs: list[str] = []
    _apply_breast_hang_z(parts, _report_soft_cs(), _empty_metrics(), msgs)
    assert any(m == "breast_hang_z_applied: true" for m in msgs)
    for p in parts:
        assert p.center is not None
        assert p.center[2] <= pre_anchor - BREAST_HANG_Z_MIN_DROP_FRAC_RZ * rz + 1e-9
        # pure B1: drop == 0.55*rz
        assert p.center[2] == pytest.approx(pre_anchor - BREAST_HANG_Z_DROP_FRAC_RZ * rz, abs=1e-9)

    # Base CS path (no profile) also meets D2 — no waist oval unless torso=ovals.
    report = _report_soft_cs()
    pkg = build_blockout_recipe(report, limbs=False)
    breasts = [p for p in pkg.parts if p.role == "breast_soft" and p.center]
    assert len(breasts) == 2
    anchor_s = _msg_value(pkg.messages, "breast_hang_z_anchor_m")
    assert anchor_s is not None
    anchor = float(anchor_s)
    for b in breasts:
        assert b.center is not None and b.rz_m is not None
        assert b.center[2] <= anchor - BREAST_HANG_Z_MIN_DROP_FRAC_RZ * float(b.rz_m) + 1e-9


def test_t1b_product_like_ovals_hang_meets_d2() -> None:
    """T1b/P2-1: product_0036up-class breast+waist ovals clear pure B1 (D2/D3).

    Product pre-0049 numbers: breast z≈1.269857, rz≈0.080570, waist z≈1.141120
    → B1 clears waist floor with ~4 mm margin (no soft-clamp).
    """
    pre_z = 1.269857
    rz = 0.080570
    waist_z = 1.141120
    chest_z = 1.3085
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    parts.extend(
        [
            RecipePart(
                name="RECIPE_torso_oval_chest",
                role="torso",
                kind="ellipsoid",
                center=[0.0, 0.0, chest_z],
                rx_m=0.15,
                ry_m=0.10,
                rz_m=0.08,
                placement="full3d",
                label="RECIPE_torso_oval_chest",
            ),
            RecipePart(
                name="RECIPE_torso_oval_waist",
                role="torso",
                kind="ellipsoid",
                center=[0.0, 0.0, waist_z],
                rx_m=0.12,
                ry_m=0.08,
                rz_m=0.06,
                placement="full3d",
                label="RECIPE_torso_oval_waist",
            ),
        ]
    )
    msgs: list[str] = []
    _apply_breast_hang_z(parts, _report_soft_cs(), _empty_metrics(chest_z=chest_z), msgs)
    assert any(m == "breast_hang_z_applied: true" for m in msgs)
    assert not any(m.startswith("breast_hang_z_reason=") for m in msgs)
    assert any(m == "breast_hang_z_source=frac_rz" for m in msgs)
    b1 = pre_z - BREAST_HANG_Z_DROP_FRAC_RZ * rz
    breasts = [p for p in parts if p.role == "breast_soft"]
    for p in breasts:
        assert p.center is not None
        assert p.center[2] == pytest.approx(b1, abs=1e-9)
        # D2: center <= anchor - 0.40*rz
        assert p.center[2] <= pre_z - BREAST_HANG_Z_MIN_DROP_FRAC_RZ * rz + 1e-9
        # D3: center+rz <= anchor + 0.45*rz + eps
        assert p.center[2] + rz <= pre_z + 0.45 * rz + 1e-6


def test_t1c_waist_soft_clamp_emits_reason() -> None:
    """T1c/P2-2: high waist soft-clamps — reason + residual drop > 0."""
    pre_z = 1.27
    rz = 0.08
    # waist_z in (1.158, 1.19) → clamp raises above B1 but leaves drop < 0.40*rz
    waist_z = 1.17
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    parts.append(
        RecipePart(
            name="RECIPE_torso_oval_waist",
            role="torso",
            kind="ellipsoid",
            center=[0.0, 0.0, waist_z],
            rx_m=0.12,
            ry_m=0.08,
            rz_m=0.06,
            placement="full3d",
            label="RECIPE_torso_oval_waist",
        )
    )
    msgs: list[str] = []
    _apply_breast_hang_z(parts, _report_soft_cs(), _empty_metrics(), msgs)
    assert any(m == "breast_hang_z_applied: true" for m in msgs)
    assert any(m == "breast_hang_z_reason=clamped_floor_soft" for m in msgs)
    drop_s = _msg_value(msgs, "breast_hang_z_drop_m")
    assert drop_s is not None
    drop = float(drop_s)
    assert drop > 0.0
    assert drop + 1e-9 < BREAST_HANG_Z_MIN_DROP_FRAC_RZ * rz
    waist_floor = waist_z + rz
    breasts = [p for p in parts if p.role == "breast_soft"]
    for p in breasts:
        assert p.center is not None
        assert p.center[2] == pytest.approx(waist_floor, abs=1e-9)


def test_t2_upper_pole_vs_pre_anchor() -> None:
    """T2/D3: center+rz <= pre_anchor + 0.45xrz + eps."""
    pre_z = 1.27
    rz = 0.08
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    msgs: list[str] = []
    _apply_breast_hang_z(parts, _report_soft_cs(), _empty_metrics(), msgs)
    for p in parts:
        assert p.center is not None
        upper = p.center[2] + rz
        assert upper <= pre_z + 0.45 * rz + 1e-6
        # B1 pure: upper == pre + (1-0.55)*rz
        assert upper == pytest.approx(pre_z + (1.0 - BREAST_HANG_Z_DROP_FRAC_RZ) * rz, abs=1e-9)


def test_t3_lower_pole_hang_and_chest_ref() -> None:
    """T3/D4: lower pole min drop; soft chest_ref when chest above pre."""
    pre_z = 1.27
    rz = 0.08
    chest_ref = 1.31  # >= pre + 0.02
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    # oval chest for chest_ref path
    parts.append(
        RecipePart(
            name="RECIPE_torso_oval_chest",
            role="torso",
            kind="ellipsoid",
            center=[0.0, 0.0, chest_ref],
            rx_m=0.15,
            ry_m=0.10,
            rz_m=0.08,
            placement="full3d",
            label="RECIPE_torso_oval_chest",
        )
    )
    msgs: list[str] = []
    _apply_breast_hang_z(parts, _report_soft_cs(), _empty_metrics(chest_z=chest_ref), msgs)
    breasts = [p for p in parts if p.role == "breast_soft"]
    for b in breasts:
        assert b.center is not None
        lower = b.center[2] - rz
        assert lower <= pre_z - rz - BREAST_HANG_Z_MIN_DROP_FRAC_RZ * rz + 1e-9
        assert lower < chest_ref - 0.05


def test_t4_tilt_fence_with_hang() -> None:
    """T4/D5: hang + tilt 20 → breast_tilt_applied true + rotation [20,0,0]."""
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(
        _rich_report(),
        limbs=False,
        profile=profile,
        skeleton=_skeleton_with_arms(),
        breast_tilt_deg=20.0,
    )
    breasts = [p for p in pkg.parts if p.role == "breast_soft"]
    assert len(breasts) >= 2
    assert any(m == "breast_tilt_applied: true" for m in pkg.messages)
    assert any(m == "breast_hang_z_applied: true" for m in pkg.messages)
    for b in breasts:
        assert b.rotation_euler_deg == pytest.approx([20.0, 0.0, 0.0])


def test_t5_dual_lr_same_center_z() -> None:
    """T5: L/R breast center Z equal after hang."""
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(
        _rich_report(),
        limbs=False,
        profile=profile,
        skeleton=_skeleton_with_arms(),
    )
    breasts = [p for p in pkg.parts if p.role == "breast_soft" and p.center]
    assert len(breasts) == 2
    zs = [p.center[2] for p in breasts if p.center is not None]
    assert zs[0] == pytest.approx(zs[1], abs=1e-12)


# ---------------------------------------------------------------------------
# T6-T7 pec + base path
# ---------------------------------------------------------------------------


def test_t6_male_pec_one_applied_false() -> None:
    """T6/B11: male pec — single breast_hang_z_applied: false; no Z thrash."""
    profile = load_anatomy_profile("torso_limb_m_athletic_v1")
    report = _rich_report()
    pkg = build_blockout_recipe(report, limbs=False, profile=profile, glute="two_spheres")
    assert not any(p.role == "breast_soft" for p in pkg.parts)
    hang_msgs = [m for m in pkg.messages if m.startswith("breast_hang_z")]
    assert hang_msgs == ["breast_hang_z_applied: false"]
    # no drop/anchor/reason side-channels on pec-only path
    assert not any(m.startswith("breast_hang_z_drop") for m in pkg.messages)
    assert not any(m.startswith("breast_hang_z_anchor") for m in pkg.messages)
    assert not any(m.startswith("breast_hang_z_reason") for m in pkg.messages)
    pecs = [p for p in pkg.parts if p.role == "pec_soft" and p.center]
    assert len(pecs) >= 1
    # pecs stay near spine_high / chest (no hang drop of 0.55*rz class forced)
    pec_zs = [p.center[2] for p in pecs if p.center is not None]
    for z in pec_zs:
        assert z > 1.0
    # dual pecs (if present) share Z — no L/R hang thrash
    if len(pec_zs) >= 2:
        assert max(pec_zs) - min(pec_zs) < 1e-9


def test_t7_base_path_dual_drop() -> None:
    """T7/B9: CS/band dual breasts (no profile) drop >= B1 vs pre anchor."""
    report = _report_soft_cs()
    pkg = build_blockout_recipe(report, limbs=False)
    breasts = [p for p in pkg.parts if p.role == "breast_soft" and p.center]
    assert len(breasts) == 2
    assert any(m == "breast_hang_z_applied: true" for m in pkg.messages)
    anchor_s = _msg_value(pkg.messages, "breast_hang_z_anchor_m")
    assert anchor_s is not None
    anchor = float(anchor_s)
    for b in breasts:
        assert b.center is not None and b.rz_m is not None
        assert b.center[2] <= anchor - BREAST_HANG_Z_MIN_DROP_FRAC_RZ * float(b.rz_m) + 1e-9
        drop = anchor - b.center[2]
        assert drop >= BREAST_HANG_Z_DROP_FRAC_RZ * float(b.rz_m) - 1e-9


# ---------------------------------------------------------------------------
# T8-T10 measured deepen-only ladder
# ---------------------------------------------------------------------------


def test_t8_measured_deep_enough_source_breast_lower() -> None:
    """T8: measured lower deep enough → source=breast_lower."""
    pre_z = 1.27
    rz = 0.08
    b1 = pre_z - BREAST_HANG_Z_DROP_FRAC_RZ * rz
    # measured center = z_lower + rz must be <= b1 → z_lower <= b1 - rz
    z_lower = b1 - rz - 0.01  # deeper than B1
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    report = _report_soft_cs(
        extra_lms={"breast_lower": _lm("breast_lower", x_m=0.0, y_m=-0.05, z_m=z_lower)}
    )
    m = _empty_metrics(chest_z=1.31)
    msgs: list[str] = []
    _apply_breast_hang_z(parts, report, m, msgs)
    assert any(m == "breast_hang_z_source=breast_lower" for m in msgs)
    for p in parts:
        assert p.center is not None
        assert p.center[2] == pytest.approx(z_lower + rz, abs=1e-9)
        # axis-aligned lower ≈ measured
        assert (p.center[2] - rz) == pytest.approx(z_lower, abs=1e-9)


def test_t9_measured_shallow_uses_frac() -> None:
    """T9/P2-1: shallow measured still B1; reason measured_shallow_using_frac."""
    pre_z = 1.27
    rz = 0.08
    b1 = pre_z - BREAST_HANG_Z_DROP_FRAC_RZ * rz
    # measured center would be above b1 (shallower hang)
    z_lower = pre_z - 0.01  # almost at pre → measured center ≈ pre + rz - 0.01 >> b1
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    report = _report_soft_cs(
        extra_lms={"breast_lower": _lm("breast_lower", x_m=0.0, y_m=-0.05, z_m=z_lower)}
    )
    msgs: list[str] = []
    _apply_breast_hang_z(parts, report, _empty_metrics(chest_z=1.31), msgs)
    assert any(m == "breast_hang_z_source=frac_rz" for m in msgs)
    assert any(m == "breast_hang_z_reason=measured_shallow_using_frac" for m in msgs)
    for p in parts:
        assert p.center is not None
        assert p.center[2] == pytest.approx(b1, abs=1e-9)
        # D2 still holds
        assert p.center[2] <= pre_z - BREAST_HANG_Z_MIN_DROP_FRAC_RZ * rz + 1e-9


def test_t9b_waist_clamp_preserves_b2_reason() -> None:
    """P2-3: waist clamp must not overwrite prior B2 reason."""
    pre_z = 1.27
    rz = 0.08
    # shallow measured → measured_shallow_using_frac first
    z_lower = pre_z - 0.01
    # high waist that would set clamped_floor_soft if reason were free
    waist_z = 1.17
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    parts.append(
        RecipePart(
            name="RECIPE_torso_oval_waist",
            role="torso",
            kind="ellipsoid",
            center=[0.0, 0.0, waist_z],
            rx_m=0.12,
            ry_m=0.08,
            rz_m=0.06,
            placement="full3d",
            label="RECIPE_torso_oval_waist",
        )
    )
    report = _report_soft_cs(
        extra_lms={"breast_lower": _lm("breast_lower", x_m=0.0, y_m=-0.05, z_m=z_lower)}
    )
    msgs: list[str] = []
    _apply_breast_hang_z(parts, report, _empty_metrics(chest_z=1.31), msgs)
    assert any(m == "breast_hang_z_reason=measured_shallow_using_frac" for m in msgs)
    assert not any("clamped_floor" in m for m in msgs)
    # geometry still soft-clamped to waist floor
    waist_floor = waist_z + rz
    for p in parts:
        if p.role == "breast_soft":
            assert p.center is not None
            assert p.center[2] == pytest.approx(waist_floor, abs=1e-9)


def test_t10_lower_out_of_band_uses_frac() -> None:
    """T10: low out-of-band lower → frac_rz + reason lower_out_of_band."""
    pre_z = 1.27
    rz = 0.08
    chest_ref = 1.31
    h = 1.72
    # far below band: chest_ref - 0.12*H
    z_lower = chest_ref - 0.12 * h - 0.05
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    parts.append(
        RecipePart(
            name="RECIPE_torso_oval_chest",
            role="torso",
            kind="ellipsoid",
            center=[0.0, 0.0, chest_ref],
            rx_m=0.15,
            ry_m=0.10,
            rz_m=0.08,
            placement="full3d",
            label="RECIPE_torso_oval_chest",
        )
    )
    report = _report_soft_cs(
        extra_lms={"breast_lower": _lm("breast_lower", x_m=0.0, y_m=-0.05, z_m=z_lower)}
    )
    msgs: list[str] = []
    _apply_breast_hang_z(parts, report, _empty_metrics(height_m=h, chest_z=chest_ref), msgs)
    assert any(m == "breast_hang_z_source=frac_rz" for m in msgs)
    assert any(m == "breast_hang_z_reason=lower_out_of_band" for m in msgs)
    b1 = pre_z - BREAST_HANG_Z_DROP_FRAC_RZ * rz
    breasts = [p for p in parts if p.role == "breast_soft"]
    for p in breasts:
        assert p.center is not None
        assert p.center[2] == pytest.approx(b1, abs=1e-9)


def test_t10b_high_out_of_band_lower_uses_frac() -> None:
    """T10b/P3-4: z_lower > chest_ref + 0.02*H → lower_out_of_band."""
    pre_z = 1.27
    rz = 0.08
    chest_ref = 1.31
    h = 1.72
    hi = chest_ref + 0.02 * h
    z_lower = hi + 0.02  # above high band
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    parts.append(
        RecipePart(
            name="RECIPE_torso_oval_chest",
            role="torso",
            kind="ellipsoid",
            center=[0.0, 0.0, chest_ref],
            rx_m=0.15,
            ry_m=0.10,
            rz_m=0.08,
            placement="full3d",
            label="RECIPE_torso_oval_chest",
        )
    )
    report = _report_soft_cs(
        extra_lms={"breast_lower": _lm("breast_lower", x_m=0.0, y_m=-0.05, z_m=z_lower)}
    )
    msgs: list[str] = []
    _apply_breast_hang_z(parts, report, _empty_metrics(height_m=h, chest_z=chest_ref), msgs)
    assert any(m == "breast_hang_z_source=frac_rz" for m in msgs)
    assert any(m == "breast_hang_z_reason=lower_out_of_band" for m in msgs)
    b1 = pre_z - BREAST_HANG_Z_DROP_FRAC_RZ * rz
    breasts = [p for p in parts if p.role == "breast_soft"]
    for p in breasts:
        assert p.center is not None
        assert p.center[2] == pytest.approx(b1, abs=1e-9)


def test_t10c_crotch_clamp_emits_reason() -> None:
    """T10c/P3-5: crotch floor raises center → reason clamped_crotch when free."""
    pre_z = 1.27
    rz = 0.08
    b1 = pre_z - BREAST_HANG_Z_DROP_FRAC_RZ * rz  # 1.226
    crotch_z = 1.24  # between b1 and pre → raises without wiping hang
    parts = _dual_breast_parts(center_z=pre_z, rz=rz)
    report = _report_soft_cs(
        extra_lms={"crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=crotch_z)}
    )
    msgs: list[str] = []
    _apply_breast_hang_z(parts, report, _empty_metrics(), msgs)
    assert any(m == "breast_hang_z_applied: true" for m in msgs)
    assert any(m == "breast_hang_z_reason=clamped_crotch" for m in msgs)
    for p in parts:
        assert p.center is not None
        assert p.center[2] == pytest.approx(crotch_z, abs=1e-9)
        assert p.center[2] > b1


# ---------------------------------------------------------------------------
# T11-T13 safety + schema + messages
# ---------------------------------------------------------------------------


def test_t11_zero_rz_and_missing_skip_safe() -> None:
    """T11: zero rz / missing breast skip safe — applied false only."""
    # empty parts
    msgs: list[str] = []
    _apply_breast_hang_z([], _report_soft_cs(), _empty_metrics(), msgs)
    assert msgs == ["breast_hang_z_applied: false"]

    # zero rz breast not gated
    msgs2: list[str] = []
    parts = [
        RecipePart(
            name="RECIPE_breast_soft_l",
            role="breast_soft",
            kind="ellipsoid",
            center=[-0.08, -0.06, 1.27],
            rx_m=0.06,
            ry_m=0.05,
            rz_m=0.0,
            placement="full3d",
            label="RECIPE_breast_soft_l",
        )
    ]
    _apply_breast_hang_z(parts, _report_soft_cs(), _empty_metrics(), msgs2)
    assert msgs2 == ["breast_hang_z_applied: false"]
    assert parts[0].center is not None
    assert parts[0].center[2] == pytest.approx(1.27)

    # pec_soft never hung
    msgs3: list[str] = []
    pecs = [
        RecipePart(
            name="RECIPE_pec_soft_l",
            role="pec_soft",
            kind="ellipsoid",
            center=[-0.08, -0.04, 1.27],
            rx_m=0.05,
            ry_m=0.04,
            rz_m=0.05,
            placement="full3d",
            label="RECIPE_pec_soft_l",
        )
    ]
    pre = pecs[0].center[2] if pecs[0].center else None
    _apply_breast_hang_z(pecs, _report_soft_cs(), _empty_metrics(), msgs3)
    assert msgs3 == ["breast_hang_z_applied: false"]
    assert pecs[0].center is not None
    assert pecs[0].center[2] == pre


def test_t12_schema_stays_1_4_0() -> None:
    """T12: schema write stays 1.4.0 (no bump).

    MCP catalog stay 46 is suite-covered in test_mcp_server / face / extremity tracks;
    assert TOOL_NAMES freeze here for local visibility only.
    """
    from meshops.mcp import TOOL_NAMES

    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    pkg = build_blockout_recipe(_report_soft_cs(), limbs=False, breast_tilt_deg=20.0)
    assert pkg.schema_version == "1.4.0"
    assert len(TOOL_NAMES) == 47


def test_t13_messages_drop_anchor_chest_ref() -> None:
    """T13: applied true → drop_m + anchor_m; chest_ref when oval chest present."""
    report = _report_soft_cs()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    assert any(m == "breast_hang_z_applied: true" for m in pkg.messages)
    drop_s = _msg_value(pkg.messages, "breast_hang_z_drop_m")
    anchor_s = _msg_value(pkg.messages, "breast_hang_z_anchor_m")
    assert drop_s is not None
    assert anchor_s is not None
    assert float(drop_s) > 0.0
    assert math.isfinite(float(anchor_s))
    # oval chest present → chest_ref message
    assert any(p.name == "RECIPE_torso_oval_chest" for p in pkg.parts)
    chest_s = _msg_value(pkg.messages, "breast_hang_z_chest_ref_m")
    assert chest_s is not None
    assert math.isfinite(float(chest_s))
    assert any(m.startswith("breast_hang_z_source=") for m in pkg.messages)


def test_constant_b1_frac() -> None:
    """B1 freeze: named constants for hang floor + D2 min drop."""
    assert BREAST_HANG_Z_DROP_FRAC_RZ == 0.55
    assert BREAST_HANG_Z_MIN_DROP_FRAC_RZ == 0.40
