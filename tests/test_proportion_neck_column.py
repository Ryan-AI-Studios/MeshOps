"""Track 0050 — neck column forward tilt / head co-move / radius ceiling."""

from __future__ import annotations

import math

import pytest

from meshops.proportion.blockout_recipe import (
    HEAD_PITCH_DEG,
    NECK_FORWARD_TILT_DEG,
    NECK_R_MAX_FRAC_HEAD_RX,
    RECIPE_SCHEMA_VERSION,
    RecipePart,
    _apply_neck_column_priors,
    _rotate_yz_about_x,
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
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=neck_hw),
    ]
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


def _head_skeleton() -> BlockoutSkeleton:
    return BlockoutSkeleton(
        schema_version="1.0.0",
        honesty="proportion_blockout_skeleton_not_mesh_or_print_success",
        joints=[
            _j("root", x=0.0, y=0.0, z=0.0),
            _j("pelvis", x=0.0, y=0.0, z=0.95, parent="root"),
            _j("spine_high", x=0.0, y=0.0, z=1.25, parent="pelvis"),
            _j("neck_base", x=0.0, y=0.0, z=1.40, parent="spine_high"),
            _j("neck_top", x=0.0, y=0.0, z=1.48, parent="neck_base"),
            _j("chin", x=0.0, y=-0.02, z=1.50, parent="neck_top"),
            _j("head", x=0.0, y=-0.01, z=1.59, parent="chin"),
            _j("crown", x=0.0, y=-0.01, z=1.68, parent="head"),
            _j("shoulder_l", x=-0.20, y=0.0, z=1.38, side="l", parent="spine_high"),
            _j("shoulder_r", x=0.20, y=0.0, z=1.38, side="r", parent="spine_high"),
        ],
        bones=[
            SkeletonBone(id="spine", joint_a="pelvis", joint_b="spine_high", length_m=0.3),
            SkeletonBone(id="neck", joint_a="neck_base", joint_b="neck_top", length_m=0.08),
            SkeletonBone(id="head_bone", joint_a="neck_top", joint_b="head", length_m=0.11),
        ],
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
    # Angle from +Z toward -Y (axis in YZ plane)
    cos_a = max(-1.0, min(1.0, dz / length))
    return math.degrees(math.acos(cos_a))


def _dy_tip_from_length(length: float) -> float:
    return -float(length) * math.sin(math.radians(NECK_FORWARD_TILT_DEG))


# ---------------------------------------------------------------------------
# T0-T13
# ---------------------------------------------------------------------------


def test_neck_column__t0_unit_helper_vertical_baseline() -> None:
    """T0: pure unit pass on pre-vertical neck gets 12° tilt."""
    parts = [
        RecipePart(
            name="RECIPE_neck",
            role="neck",
            kind="cylinder",
            p0=[0.0, 0.0, 1.38],
            p1=[0.0, 0.0, 1.50],
            radius_m=0.05,
            placement="full3d",
            label="RECIPE_neck",
        ),
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
        ),
    ]
    messages: list[str] = []
    _apply_neck_column_priors(parts, messages)
    neck = parts[0]
    assert _neck_axis_tilt_deg(neck) == pytest.approx(NECK_FORWARD_TILT_DEG, abs=0.05)
    assert "neck_column_tilt_applied: true" in messages


def test_neck_column__t1_axis_tilt_about_12deg() -> None:
    """T1: axis angle from +Z ∈ [8°, 15°]; ≈12° when neck present."""
    pkg = build_blockout_recipe(_report(), limbs=False)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    tilt = _neck_axis_tilt_deg(neck)
    assert 8.0 <= tilt <= 15.0
    assert tilt == pytest.approx(NECK_FORWARD_TILT_DEG, abs=0.2)


def test_neck_column__t2_tip_leans_negative_y() -> None:
    """T2: p1[1] < p0[1] (tip -Y)."""
    pkg = build_blockout_recipe(_report(chest_mid_y=0.0), limbs=False)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    assert neck.p0 is not None and neck.p1 is not None
    assert neck.p1[1] < neck.p0[1]


def test_neck_column__t3_head_shift_preserve() -> None:
    """T3: head Y = pre + dy_tip; neck tip = y0 + dy_tip (shift, not absolute)."""
    report = _report(chin_y=-0.02)
    # Pre-head Y from chin; chest_y from mid=0 → neck base y0=0
    pkg = build_blockout_recipe(report, limbs=False)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert neck.p0 is not None and neck.p1 is not None and head.center is not None
    y0 = float(neck.p0[1])
    length = math.dist(neck.p0, neck.p1)
    dy_tip = _dy_tip_from_length(length)
    pre_head_y = -0.02  # chin y preserved into head
    assert float(neck.p1[1]) == pytest.approx(y0 + dy_tip, abs=1e-6)
    # 0085: 0050 shift still holds in pre-pitch space
    pivot = [float(neck.p1[0]), float(neck.p1[1]), float(neck.p1[2])]
    head_pre = _rotate_yz_about_x(list(head.center), pivot, -math.radians(HEAD_PITCH_DEG))
    assert float(head_pre[1]) == pytest.approx(pre_head_y + dy_tip, abs=1e-5)
    # Absolute head≈tip should NOT hold when chin offset present
    assert float(head.center[1]) != pytest.approx(float(neck.p1[1]), abs=1e-3)


def test_neck_column__t3b_chin_y_null_head_near_tip() -> None:
    """T3b optional: chin y null → head Y ≈ neck tip (both on axial mid + dy)."""
    report = _report(chin_y=None, chest_mid_y=0.0)
    lms = dict(report.landmarks_xyz)
    lms["cranial_vertex"] = _lm("cranial_vertex", x_m=0.0, y_m=None, z_m=1.68)
    report = report.model_copy(update={"landmarks_xyz": lms})
    pkg = build_blockout_recipe(report, limbs=False)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert neck.p1 is not None and head.center is not None
    pivot = [float(neck.p1[0]), float(neck.p1[1]), float(neck.p1[2])]
    head_pre = _rotate_yz_about_x(list(head.center), pivot, -math.radians(HEAD_PITCH_DEG))
    assert float(head_pre[1]) == pytest.approx(float(neck.p1[1]), abs=1e-5)


def test_neck_column__t4_face_soft_comoves() -> None:
    """T4: face soft (ear or jaw) co-moves by same dy_tip when face=True."""
    report = _report(chin_y=-0.02)
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert neck.p0 is not None and neck.p1 is not None and head.center is not None
    length = math.dist(neck.p0, neck.p1)
    dy_tip = _dy_tip_from_length(length)
    pre_head_y = -0.02
    pivot = [float(neck.p1[0]), float(neck.p1[1]), float(neck.p1[2])]
    th = -math.radians(HEAD_PITCH_DEG)
    head_pre = _rotate_yz_about_x(list(head.center), pivot, th)
    assert float(head_pre[1]) == pytest.approx(pre_head_y + dy_tip, abs=1e-5)
    # ear_soft sits at bounds.y (same pre as head); after co-move matches head Y
    ear = next((p for p in pkg.parts if "ear_soft" in p.name.lower()), None)
    jaw = next((p for p in pkg.parts if p.name == "RECIPE_jaw"), None)
    soft = ear or jaw
    assert soft is not None and soft.center is not None
    if ear is not None and ear.center is not None:
        ear_pre = _rotate_yz_about_x(list(ear.center), pivot, th)
        assert float(ear_pre[1]) == pytest.approx(pre_head_y + dy_tip, abs=1e-5)
        assert float(ear.center[1]) == pytest.approx(float(head.center[1]), abs=1e-5)
    else:
        # Jaw is face_y + JAW_Y_BIAS_FRAC_RY*ry (0.08) then + dy_tip; still moved faceward with head
        assert float(soft.center[1]) < pre_head_y - 1e-4
        assert abs(float(soft.center[1]) - float(head.center[1])) < 0.05


def test_neck_column__t5_neckline_not_shifted() -> None:
    """T5: neckline Y not shifted by full tip (collar stays on chest plane)."""
    report = _report(chest_mid_y=0.0)
    pkg = build_blockout_recipe(report, limbs=False, neckline="crew")
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    assert neck.p0 is not None and neck.p1 is not None
    dy_tip = float(neck.p1[1]) - float(neck.p0[1])
    assert dy_tip < 0.0
    necklines = [p for p in pkg.parts if p.role == "neckline" or "neckline" in p.name.lower()]
    assert necklines
    for nl in necklines:
        if nl.center is not None:
            # Not co-moved by full tip (still near chest mid / base plane)
            assert (
                abs(float(nl.center[1]) - dy_tip) > 1e-3
                or abs(float(nl.center[1])) < abs(dy_tip) * 0.25
            )
            assert float(nl.center[1]) == pytest.approx(0.0, abs=5e-2)
        if nl.p0 is not None:
            assert float(nl.p0[1]) == pytest.approx(0.0, abs=5e-2)


def test_neck_column__t5b_scm_base_fixed_tip_moves() -> None:
    """T5b: SCM p0 Y fixed; SCM p1 Y += dy_tip (strict pre/post unit)."""
    # Unit construction so pre tip Y is known (product build alone cannot re-read pre).
    parts = [
        RecipePart(
            name="RECIPE_neck",
            role="neck",
            kind="cylinder",
            p0=[0.0, 0.0, 1.38],
            p1=[0.0, 0.0, 1.51],
            radius_m=0.04,
            placement="full3d",
            label="RECIPE_neck",
        ),
        RecipePart(
            name="RECIPE_sternomastoid_soft_l",
            role="sternomastoid_soft",
            kind="capsule",
            p0=[-0.03, 0.0, 1.38],
            p1=[-0.05, 0.0, 1.50],
            radius_m=0.01,
            placement="full3d",
            label="RECIPE_sternomastoid_soft_l",
        ),
        RecipePart(
            name="RECIPE_sternomastoid_soft_r",
            role="sternomastoid_soft",
            kind="capsule",
            p0=[0.03, 0.0, 1.38],
            p1=[0.05, 0.0, 1.50],
            radius_m=0.01,
            placement="full3d",
            label="RECIPE_sternomastoid_soft_r",
        ),
    ]
    pre_scm = [
        (list(p.p0) if p.p0 else None, list(p.p1) if p.p1 else None)
        for p in parts
        if "sternomastoid" in p.name.lower()
    ]
    messages: list[str] = []
    _apply_neck_column_priors(parts, messages)
    neck = parts[0]
    assert neck.p0 is not None and neck.p1 is not None
    dy_tip = float(neck.p1[1]) - float(neck.p0[1])
    assert dy_tip < 0.0
    scms = [p for p in parts if "sternomastoid" in p.name.lower()]
    assert len(scms) == 2
    for scm, (pre_p0, pre_p1) in zip(scms, pre_scm, strict=True):
        assert scm.p0 is not None and scm.p1 is not None
        assert pre_p0 is not None and pre_p1 is not None
        # Base fixed (P2-2)
        assert float(scm.p0[0]) == pytest.approx(float(pre_p0[0]), abs=1e-9)
        assert float(scm.p0[1]) == pytest.approx(float(pre_p0[1]), abs=1e-9)
        assert float(scm.p0[2]) == pytest.approx(float(pre_p0[2]), abs=1e-9)
        # Tip Y only += dy_tip
        assert float(scm.p1[0]) == pytest.approx(float(pre_p1[0]), abs=1e-9)
        assert float(scm.p1[1]) == pytest.approx(float(pre_p1[1]) + dy_tip, abs=1e-9)
        assert float(scm.p1[2]) == pytest.approx(float(pre_p1[2]), abs=1e-9)


def test_neck_column__t6_radius_ceiling_fat_neck() -> None:
    """T6: fat neck no-template -> r <= 0.40*head.rx."""
    # Large measured neck half-width without template
    pkg = build_blockout_recipe(_report(neck_hw=0.12), limbs=False)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert neck.radius_m is not None and head.rx_m is not None
    cap = NECK_R_MAX_FRAC_HEAD_RX * float(head.rx_m)
    assert float(neck.radius_m) <= cap + 1e-9
    assert any(m.startswith("neck_radius_clamped_head_frac=") for m in pkg.messages)


def test_neck_column__t7_female_thin_no_clamp_message() -> None:
    """T7: female thin path ratio <0.40 → no clamp message (neck_hw=0.04 headroom)."""
    tpl = _female_template(neck_thickness_scale=0.72675)
    pkg = build_blockout_recipe(_report(neck_hw=0.04), limbs=False, template_applied=tpl)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert neck.radius_m is not None and head.rx_m is not None
    ratio = float(neck.radius_m) / float(head.rx_m)
    assert ratio < NECK_R_MAX_FRAC_HEAD_RX
    assert not any(m.startswith("neck_radius_clamped_head_frac=") for m in pkg.messages)
    assert any("neck_thickness_scale=" in m for m in pkg.messages)


def test_neck_column__t8_giraffe_still_clamps() -> None:
    """T8: giraffe still clamps raw 0.5 @ H=1.72; message preserved."""
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
    assert any("0.500" in m and "clamped" in m and "giraffe" in m for m in pkg.messages)
    assert "neck_column_tilt_applied: true" in pkg.messages


def test_neck_column__t9_thickness_scale_before_ceiling() -> None:
    """T9: thickness_scale multiplies pre-ceiling r (neck_hw=0.04 clear headroom)."""
    scale = 0.72675
    tpl = _female_template(neck_thickness_scale=scale)
    raw_hw = 0.04
    pkg = build_blockout_recipe(_report(neck_hw=raw_hw), limbs=False, template_applied=tpl)
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert neck.radius_m is not None and head.rx_m is not None
    expected_pre_ceiling = raw_hw * scale
    cap = NECK_R_MAX_FRAC_HEAD_RX * float(head.rx_m)
    expected = min(expected_pre_ceiling, cap)
    assert float(neck.radius_m) == pytest.approx(expected, abs=1e-6)
    # Scale applied (message) and thin enough that ceiling does not bite
    assert expected_pre_ceiling <= cap + 1e-9
    assert any(f"neck_thickness_scale={scale}" in m for m in pkg.messages)


def test_neck_column__t10_axial_depth_plane_pass() -> None:
    """T10: mid=0 front=-0.13 + default tilt → C_axial_depth_plane pass."""
    report = _report(chest_mid_y=0.0, chest_front_y=-0.13)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    axial = by_id["C_axial_depth_plane"]
    assert axial.status == "pass", axial.message


def test_neck_column__t11_no_neck_single_false() -> None:
    """T11: no neck → single applied:false (no second reason)."""
    pkg = build_blockout_recipe(_report(include_chin=False), limbs=False)
    assert not any(p.name == "RECIPE_neck" for p in pkg.parts)
    flags = [m for m in pkg.messages if m.startswith("neck_column_tilt_applied:")]
    assert flags == ["neck_column_tilt_applied: false"]
    assert not any("neck_column_tilt_reason=" in m for m in pkg.messages)


def test_neck_column__t12_schema_1_4_0() -> None:
    """T12: schema stays 1.4.0."""
    pkg = build_blockout_recipe(_report(), limbs=False)
    assert pkg.schema_version == "1.4.0"
    assert RECIPE_SCHEMA_VERSION == "1.4.0"


def test_neck_column__t13_messages_when_applied() -> None:
    """T13: tilt_deg, tip_dy_m, head_comove when applied."""
    pkg = build_blockout_recipe(_report(), limbs=False)
    msgs = pkg.messages
    assert f"neck_forward_tilt_deg={NECK_FORWARD_TILT_DEG}" in msgs
    assert "neck_column_tilt_applied: true" in msgs
    assert any(m.startswith("neck_column_tip_dy_m=") for m in msgs)
    assert "neck_column_head_comove: true" in msgs
    tip_msg = next(m for m in msgs if m.startswith("neck_column_tip_dy_m="))
    dy = float(tip_msg.split("=", 1)[1])
    assert dy < 0.0


def test_neck_column__unit_nonpositive_length_reason() -> None:
    """L≤0 path: applied false + nonpositive_length reason."""
    parts = [
        RecipePart(
            name="RECIPE_neck",
            role="neck",
            kind="cylinder",
            p0=[0.0, 0.0, 1.38],
            p1=[0.0, 0.0, 1.38],
            radius_m=0.05,
            placement="full3d",
            label="RECIPE_neck",
        )
    ]
    messages: list[str] = []
    _apply_neck_column_priors(parts, messages)
    assert "neck_column_tilt_applied: false" in messages
    assert "neck_column_tilt_reason=nonpositive_length" in messages
