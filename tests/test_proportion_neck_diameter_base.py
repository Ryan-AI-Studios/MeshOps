"""Track 0059 — neck diameter ceiling, base soft ellipsoid, SCM thicken."""

from __future__ import annotations

import math

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    NECK_BASE_RX_FRAC_R,
    NECK_BASE_RY_FRAC_R,
    NECK_BASE_RZ_FRAC_R,
    NECK_BASE_Z_BURY_FRAC_RZ,
    NECK_FORWARD_TILT_DEG,
    NECK_R_MAX_FRAC_HEAD_RX,
    RECIPE_SCHEMA_VERSION,
    SCM_R_CAP_M,
    SCM_R_FLOOR_M,
    SCM_R_FRAC_NECK_R,
    RecipePart,
    _apply_neck_diameter_base,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.models import (
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


def _depth_band(
    band_id: str,
    *,
    depth_m: float = 0.24,
    z_frac: float = 0.72,
    y_mid: float = 0.0,
) -> DepthBand:
    return DepthBand(
        band_id=band_id,
        depth_px=50.0,
        depth_frac=0.12,
        depth_m=depth_m,
        y_front=0.1,
        y_back=-0.1,
        y_mid=y_mid,
        z_frac=z_frac,
    )


def _base_lms(
    *,
    height_m: float = 1.72,
    chin_z: float = 1.50,
    chin_y: float | None = -0.02,
    shoulder_z: float = 1.38,
    chest_mid_y: float = 0.0,
    chest_front_y: float = -0.13,
    extra: dict[str, LandmarkXYZ] | None = None,
) -> dict[str, LandmarkXYZ]:
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=chin_y, z_m=chin_z),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=shoulder_z),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=shoulder_z),
        "hip_l": _lm("hip_l", x_m=-0.14, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.14, y_m=0.0, z_m=0.95),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=chin_z + 0.18),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=chest_mid_y, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=chest_front_y, z_m=1.25),
    }
    _ = height_m
    if extra:
        lms.update(extra)
    return lms


def _report(
    *,
    height_m: float = 1.72,
    chin_z: float = 1.50,
    chin_y: float | None = -0.02,
    shoulder_z: float = 1.38,
    neck_hw: float = 0.05,
    chest_mid_y: float = 0.0,
    chest_front_y: float = -0.13,
    include_chin: bool = True,
    extra_lms: dict[str, LandmarkXYZ] | None = None,
    rich: bool = False,
) -> ProportionReport:
    lms = _base_lms(
        height_m=height_m,
        chin_z=chin_z,
        chin_y=chin_y,
        shoulder_z=shoulder_z,
        chest_mid_y=chest_mid_y,
        chest_front_y=chest_front_y,
        extra=extra_lms,
    )
    if not include_chin:
        lms.pop("chin", None)
    if rich:
        lms.update(
            {
                "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
                "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
                "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=height_m),
                "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=-0.05, z_m=1.10),
                "elbow_r": _lm("elbow_r", x_m=0.28, y_m=-0.05, z_m=1.10),
                "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=0.0, z_m=0.90),
                "wrist_r": _lm("wrist_r", x_m=0.30, y_m=0.0, z_m=0.90),
                "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
                "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
                "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
                "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
            }
        )
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=neck_hw),
    ]
    if rich:
        diams.extend(
            [
                _diam("upper_arm_l", half_width_m=0.05),
                _diam("upper_arm_r", half_width_m=0.05),
                _diam("forearm_l", half_width_m=0.04),
                _diam("forearm_r", half_width_m=0.04),
                _diam("thigh_l", half_width_m=0.06),
                _diam("thigh_r", half_width_m=0.06),
                _diam("calf_l", half_width_m=0.05),
                _diam("calf_r", half_width_m=0.05),
            ]
        )
    bands = [
        _depth_band("chest", depth_m=0.24, z_frac=0.72, y_mid=0.0),
        _depth_band("hip", depth_m=0.26, z_frac=0.55),
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


def _skeleton_with_arms(
    *,
    shoulder_hw: float = 0.20,
    shoulder_z: float = 1.38,
) -> BlockoutSkeleton:
    joints = [
        _j("root", x=0.0, y=0.0, z=0.0),
        _j("pelvis", x=0.0, y=0.0, z=0.95, parent="root"),
        _j("spine_high", x=0.0, y=0.0, z=1.25, parent="pelvis"),
        _j("neck_base", x=0.0, y=0.0, z=1.42, parent="spine_high"),
        _j("shoulder_l", x=-shoulder_hw, y=0.0, z=shoulder_z, side="l", parent="spine_high"),
        _j("shoulder_r", x=shoulder_hw, y=0.0, z=shoulder_z, side="r", parent="spine_high"),
        _j("elbow_l", x=-0.28, y=0.0, z=1.10, side="l", parent="shoulder_l"),
        _j("elbow_r", x=0.28, y=0.0, z=1.10, side="r", parent="shoulder_r"),
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


def _female_template(*, neck_thickness_scale: float = 0.72675) -> TemplateAppliedPackage:
    constants = AppliedConstants(
        breast_mode="dual_tilted",
        glute_mode_default="oval",
        torso_mode_default="trap",
        neck_thickness_scale=neck_thickness_scale,
        torso_waist_taper=0.14,
    )
    return TemplateAppliedPackage(
        template_id="female_adult_athletic",
        sex="female",
        archetype="adult_athletic",
        source_report="mem",
        height_m=1.72,
        constants=constants,
    )


def _neck_axis_tilt_deg(neck: RecipePart) -> float:
    assert neck.p0 is not None and neck.p1 is not None
    dx = float(neck.p1[0]) - float(neck.p0[0])
    dy = float(neck.p1[1]) - float(neck.p0[1])
    dz = float(neck.p1[2]) - float(neck.p0[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    assert length > 0.0
    cos_a = max(-1.0, min(1.0, dz / length))
    return math.degrees(math.acos(cos_a))


# ---------------------------------------------------------------------------
# T0 constants
# ---------------------------------------------------------------------------


def test_t0_constants_freeze() -> None:
    """T0: ceiling 0.40; base/SCM fracs in plan bands."""
    assert NECK_R_MAX_FRAC_HEAD_RX == 0.40
    assert 1.15 <= NECK_BASE_RX_FRAC_R <= 1.30
    assert NECK_BASE_RX_FRAC_R == 1.25
    assert 0.80 <= NECK_BASE_RY_FRAC_R <= 1.00
    assert NECK_BASE_RY_FRAC_R == 0.90
    assert 0.45 <= NECK_BASE_RZ_FRAC_R <= 0.65
    assert NECK_BASE_RZ_FRAC_R == 0.55
    assert 0.20 <= NECK_BASE_Z_BURY_FRAC_RZ <= 0.40
    assert NECK_BASE_Z_BURY_FRAC_RZ == 0.30
    assert 0.32 <= SCM_R_FRAC_NECK_R <= 0.45
    assert SCM_R_FRAC_NECK_R == 0.38
    assert 0.006 <= SCM_R_FLOOR_M <= 0.010
    assert SCM_R_FLOOR_M == 0.008
    assert 0.015 <= SCM_R_CAP_M <= 0.022
    assert SCM_R_CAP_M == 0.018
    assert NECK_FORWARD_TILT_DEG == 12.0
    assert RECIPE_SCHEMA_VERSION == "1.4.0"


# ---------------------------------------------------------------------------
# T1-T2 diameter ceiling
# ---------------------------------------------------------------------------


def test_t1_fat_neck_no_template_clamps() -> None:
    """T1: fat neck no-template r <= 0.40*head.rx; clamp message."""
    pkg = build_blockout_recipe(_report(neck_hw=0.12), limbs=False)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert neck.radius_m is not None and head.rx_m is not None
    cap = NECK_R_MAX_FRAC_HEAD_RX * float(head.rx_m)
    assert float(neck.radius_m) <= cap + 1e-9
    assert any(m.startswith("neck_radius_clamped_head_frac=") for m in pkg.messages)
    assert f"neck_radius_clamped_head_frac={NECK_R_MAX_FRAC_HEAD_RX}" in pkg.messages


def test_t2_product_like_female_clamps() -> None:
    """T2: product-like female (scale + fat pre) r <= 0.40*head.rx."""
    # pre-scale neck_hw ≈ product class 0.056 → after scale ~0.0407; fixture head.rx
    # ≈0.0917 → cap@0.40 ≈0.0367 so ceiling bites (product head.rx smaller still clamps).
    tpl = _female_template(neck_thickness_scale=0.72675)
    pkg = build_blockout_recipe(
        _report(neck_hw=0.056),
        limbs=False,
        template_applied=tpl,
    )
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert neck.radius_m is not None and head.rx_m is not None
    cap = NECK_R_MAX_FRAC_HEAD_RX * float(head.rx_m)
    assert float(neck.radius_m) <= cap + 1e-9
    assert float(neck.radius_m) / float(head.rx_m) <= NECK_R_MAX_FRAC_HEAD_RX + 1e-9
    assert any(m.startswith("neck_radius_clamped_head_frac=") for m in pkg.messages)


# ---------------------------------------------------------------------------
# T3-T4 base soft
# ---------------------------------------------------------------------------


def test_t3_base_soft_present_axes() -> None:
    """T3: RECIPE_neck_base_soft role neck ellipsoid; axes ~ fracs*r."""
    pkg = build_blockout_recipe(_report(), limbs=False)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    base = next(p for p in pkg.parts if p.name == "RECIPE_neck_base_soft")
    assert neck.radius_m is not None
    r = float(neck.radius_m)
    assert base.role == "neck"
    assert base.kind == "ellipsoid"
    assert base.rx_m is not None and base.ry_m is not None and base.rz_m is not None
    assert float(base.rx_m) == pytest.approx(NECK_BASE_RX_FRAC_R * r, abs=1e-9)
    assert float(base.ry_m) == pytest.approx(NECK_BASE_RY_FRAC_R * r, abs=1e-9)
    assert float(base.rz_m) == pytest.approx(NECK_BASE_RZ_FRAC_R * r, abs=1e-9)


def test_t4_base_z_bury_and_y() -> None:
    """T4: base center[2] < neck.p0[2]; cy == p0[1]."""
    pkg = build_blockout_recipe(_report(), limbs=False)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    base = next(p for p in pkg.parts if p.name == "RECIPE_neck_base_soft")
    assert neck.p0 is not None and base.center is not None and base.rz_m is not None
    assert float(base.center[2]) < float(neck.p0[2])
    assert float(base.center[1]) == pytest.approx(float(neck.p0[1]), abs=1e-9)
    assert float(base.center[0]) == pytest.approx(0.0, abs=1e-9)
    expected_z = float(neck.p0[2]) - NECK_BASE_Z_BURY_FRAC_RZ * float(base.rz_m)
    assert float(base.center[2]) == pytest.approx(expected_z, abs=1e-9)


def test_t3_unit_idempotent_update() -> None:
    """Unit: existing base soft is updated (not skipped)."""
    parts = [
        RecipePart(
            name="RECIPE_neck",
            role="neck",
            kind="cylinder",
            p0=[0.0, 0.0, 1.38],
            p1=[0.0, -0.027, 1.51],
            radius_m=0.035,
            placement="full3d",
            label="RECIPE_neck",
        ),
        RecipePart(
            name="RECIPE_neck_base_soft",
            role="neck",
            kind="ellipsoid",
            center=[0.0, 0.0, 1.0],
            rx_m=0.01,
            ry_m=0.01,
            rz_m=0.01,
            placement="full3d",
            label="RECIPE_neck_base_soft",
        ),
    ]
    msgs: list[str] = []
    _apply_neck_diameter_base(parts, msgs)
    bases = [p for p in parts if p.name == "RECIPE_neck_base_soft"]
    assert len(bases) == 1
    base = bases[0]
    r = 0.035
    assert base.center is not None
    assert float(base.rx_m or 0.0) == pytest.approx(NECK_BASE_RX_FRAC_R * r, abs=1e-9)
    assert float(base.center[2]) == pytest.approx(
        1.38 - NECK_BASE_Z_BURY_FRAC_RZ * NECK_BASE_RZ_FRAC_R * r,
        abs=1e-9,
    )


# ---------------------------------------------------------------------------
# T5-T6 SCM
# ---------------------------------------------------------------------------


def test_t5_face_scm_thickened() -> None:
    """T5: face path SCM r in [floor,cap]; ~0.38*neck.r; dual L/R equal."""
    pkg = build_blockout_recipe(_report(), limbs=False, face=True)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    scms = [p for p in pkg.parts if "sternomastoid" in p.name.lower()]
    assert len(scms) == 2
    assert neck.radius_m is not None
    r_neck = float(neck.radius_m)
    expected = min(SCM_R_CAP_M, max(SCM_R_FLOOR_M, SCM_R_FRAC_NECK_R * r_neck))
    radii = []
    for scm in scms:
        assert scm.radius_m is not None
        r = float(scm.radius_m)
        radii.append(r)
        assert SCM_R_FLOOR_M - 1e-9 <= r <= SCM_R_CAP_M + 1e-9
        assert r == pytest.approx(expected, abs=1e-9)
        assert r == pytest.approx(SCM_R_FRAC_NECK_R * r_neck, abs=1e-4)
    assert radii[0] == pytest.approx(radii[1], abs=1e-12)
    assert any(m.startswith("scm_radius_scaled: true") for m in pkg.messages)


def test_t6_face_false_no_scm_base_present() -> None:
    """T6: face=False → no SCM; base soft still present."""
    pkg = build_blockout_recipe(_report(), limbs=False, face=False)
    assert not any("sternomastoid" in p.name.lower() for p in pkg.parts)
    assert any(p.name == "RECIPE_neck_base_soft" for p in pkg.parts)
    assert not any(m.startswith("scm_radius_scaled:") for m in pkg.messages)
    assert any(m.startswith("neck_base_soft_applied: true") for m in pkg.messages)


# ---------------------------------------------------------------------------
# T7 no neck
# ---------------------------------------------------------------------------


def test_t7_no_neck_no_base_no_crash() -> None:
    """T7: no neck → no base; no crash."""
    pkg = build_blockout_recipe(_report(include_chin=False), limbs=False)
    assert not any(p.name == "RECIPE_neck" for p in pkg.parts)
    assert not any(p.name == "RECIPE_neck_base_soft" for p in pkg.parts)
    assert not any(m.startswith("neck_base_soft_applied:") for m in pkg.messages)
    assert not any(m.startswith("scm_radius_scaled:") for m in pkg.messages)

    # Unit path with only head / empty SCM: quiet skip
    parts = [
        RecipePart(
            name="RECIPE_head",
            role="head",
            kind="ellipsoid",
            center=[0.0, 0.0, 1.59],
            rx_m=0.09,
            ry_m=0.10,
            rz_m=0.11,
            placement="full3d",
            label="RECIPE_head",
        )
    ]
    msgs: list[str] = []
    _apply_neck_diameter_base(parts, msgs)
    assert msgs == []
    assert len(parts) == 1


# ---------------------------------------------------------------------------
# T8-T9 0050 fences
# ---------------------------------------------------------------------------


def test_t8_tilt_regression() -> None:
    """T8: axis 8-15 deg; tip -Y."""
    pkg = build_blockout_recipe(_report(chest_mid_y=0.0), limbs=False)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    tilt = _neck_axis_tilt_deg(neck)
    assert 8.0 <= tilt <= 15.0
    assert tilt == pytest.approx(NECK_FORWARD_TILT_DEG, abs=0.2)
    assert neck.p0 is not None and neck.p1 is not None
    assert float(neck.p1[1]) < float(neck.p0[1])


def test_t9_giraffe_and_thickness_scale() -> None:
    """T9: giraffe clamp + thickness_scale still apply under 0059."""
    h = 1.72
    shoulder_z = 1.20
    chin_z = shoulder_z + 0.50
    pkg = build_blockout_recipe(
        _report(height_m=h, chin_z=chin_z, shoulder_z=shoulder_z),
        limbs=False,
    )
    cap = 0.20 * h
    assert pkg.metrics.neck_len_m == pytest.approx(cap)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    assert neck.p0 is not None and neck.p1 is not None
    assert math.dist(neck.p0, neck.p1) == pytest.approx(cap, abs=1e-6)
    assert any("giraffe" in m for m in pkg.messages)

    scale = 0.72675
    tpl = _female_template(neck_thickness_scale=scale)
    pkg2 = build_blockout_recipe(
        _report(neck_hw=0.04),
        limbs=False,
        template_applied=tpl,
    )
    assert any(f"neck_thickness_scale={scale}" in m for m in pkg2.messages)
    neck2 = next(p for p in pkg2.parts if p.name == "RECIPE_neck")
    assert neck2.radius_m is not None
    assert float(neck2.radius_m) == pytest.approx(0.04 * scale, abs=1e-6)


# ---------------------------------------------------------------------------
# T10 axial
# ---------------------------------------------------------------------------


def test_t10_axial_depth_plane_pass() -> None:
    """T10: mid=0 front~-0.13 product-like -> C_axial_depth_plane pass."""
    report = _report(chest_mid_y=0.0, chest_front_y=-0.13)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    axial = by_id["C_axial_depth_plane"]
    assert axial.status == "pass", axial.message


# ---------------------------------------------------------------------------
# T11 fence trap/scap
# ---------------------------------------------------------------------------


def test_t11_fence_trap_scap_emit() -> None:
    """T11: trap/scap names still emit on full product-like path."""
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(
        _report(rich=True),
        limbs=True,
        profile=profile,
        skeleton=_skeleton_with_arms(),
        torso="ovals",
    )
    trap_names = [p.name for p in pkg.parts if p.role == "trap_soft" or "trap_soft" in p.name]
    scap_names = [p.name for p in pkg.parts if p.role == "scap_soft" or "scap_soft" in p.name]
    assert trap_names, "trap_soft should still emit"
    assert scap_names, "scap_soft should still emit"
    # Base soft also present when neck exists
    assert any(p.name == "RECIPE_neck_base_soft" for p in pkg.parts)


# ---------------------------------------------------------------------------
# T12 messages
# ---------------------------------------------------------------------------


def test_t12_messages_b8_key_value() -> None:
    """T12: B8 key=value messages when applied."""
    pkg = build_blockout_recipe(_report(), limbs=False, face=True)
    msgs = pkg.messages
    base_msgs = [m for m in msgs if m.startswith("neck_base_soft_applied: true")]
    assert len(base_msgs) == 1
    # rx=… ry=… rz=… z=… four decimal places
    blob = base_msgs[0]
    assert "rx=" in blob and "ry=" in blob and "rz=" in blob and "z=" in blob
    scm_msgs = [m for m in msgs if m.startswith("scm_radius_scaled: true")]
    assert len(scm_msgs) == 1
    assert "r=" in scm_msgs[0]
    assert f"frac={SCM_R_FRAC_NECK_R}" in scm_msgs[0]
    # Default neck_hw=0.05 no-template clamps under 0.40 ceiling
    assert f"neck_radius_clamped_head_frac={NECK_R_MAX_FRAC_HEAD_RX}" in msgs
