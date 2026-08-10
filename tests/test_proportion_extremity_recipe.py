"""Track 0029 - hand / foot / digit RECIPE kit (offline; no Blender)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.blockout_recipe import (
    RECIPE_SCHEMA_VERSION,
    RecipePart,
    build_blockout_recipe,
    load_blockout_recipe,
    write_blockout_recipe,
)
from meshops.proportion.constraints import (
    FOOT_WIDTH_TOL_M,
    HEEL_REACH_GAP_TOL_M,
    classify_part_name,
    validate_constraints,
)
from meshops.proportion.extremity_recipe import (
    FOOT_LEN_BASE_FRAC_H,
    FOOT_LEN_MIN_VS_CALF_DIAM,
    FOOT_LEN_VISUAL_MIN_FRAC_H,
    HEEL_RZ_CAP_FRAC_ANK,
    _assert_ank_foot_name,
    apply_foot_length_visual_floor,
    build_foot_parts,
    finger_primary_axis,
)
from meshops.proportion.models import (
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import BlockoutSkeleton, SkeletonBone, SkeletonJoint

runner = CliRunner()


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


def _depth_band(
    band_id: str,
    *,
    depth_m: float = 0.22,
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


def _full_torso_report(
    *,
    height_m: float = 1.72,
    chin_z: float = 1.50,
    shoulder_z: float = 1.38,
    hip_z: float = 0.95,
    shoulder_x: float = 0.20,
    hip_x: float = 0.14,
    head_unit_frac: float = 1.0 / 7.5,
    extra_lms: dict[str, LandmarkXYZ] | None = None,
    extra_diams: list[DiameterMeasure] | None = None,
) -> ProportionReport:
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=chin_z),
        "shoulder_l": _lm("shoulder_l", x_m=-shoulder_x, y_m=0.0, z_m=shoulder_z),
        "shoulder_r": _lm("shoulder_r", x_m=shoulder_x, y_m=0.0, z_m=shoulder_z),
        "hip_l": _lm("hip_l", x_m=-hip_x, y_m=0.0, z_m=hip_z),
        "hip_r": _lm("hip_r", x_m=hip_x, y_m=0.0, z_m=hip_z),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=chin_z + 0.18),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.08, z_m=1.25),
    }
    if extra_lms:
        lms.update(extra_lms)

    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.05),
        _diam("upper_arm_r", half_width_m=0.05),
        _diam("thigh_l", half_width_m=0.05),
        _diam("thigh_r", half_width_m=0.05),
    ]
    if extra_diams:
        diams.extend(extra_diams)
    bands = [
        _depth_band("chest", depth_m=0.24, z_frac=0.72),
        _depth_band("hip", depth_m=0.26, z_frac=0.55),
    ]
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=head_unit_frac,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=bands,
        quality=QualityFlags(),
    )


def _extremity_lms(*, height_m: float = 1.72) -> dict[str, LandmarkXYZ]:
    """Wrist/hand/fingertip + ankle/heel/toe L/R with frame signs (toes -Y, heels +Y)."""
    ank_z = 0.08
    heel_y = 0.06
    toe_y = -0.12
    wrist_z = 0.95
    tip_z = 0.72
    return {
        "wrist_l": _lm("wrist_l", x_m=-0.45, y_m=0.0, z_m=wrist_z),
        "wrist_r": _lm("wrist_r", x_m=0.45, y_m=0.0, z_m=wrist_z),
        "hand_l": _lm("hand_l", x_m=-0.48, y_m=0.0, z_m=0.88),
        "hand_r": _lm("hand_r", x_m=0.48, y_m=0.0, z_m=0.88),
        "fingertip_l": _lm("fingertip_l", x_m=-0.50, y_m=0.0, z_m=tip_z),
        "fingertip_r": _lm("fingertip_r", x_m=0.50, y_m=0.0, z_m=tip_z),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.02, z_m=ank_z),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.02, z_m=ank_z),
        "heel_l": _lm("heel_l", x_m=-0.10, y_m=heel_y, z_m=0.02),
        "heel_r": _lm("heel_r", x_m=0.10, y_m=heel_y, z_m=0.02),
        "toe_l": _lm("toe_l", x_m=-0.10, y_m=toe_y, z_m=0.02),
        "toe_r": _lm("toe_r", x_m=0.10, y_m=toe_y, z_m=0.02),
    }


def _report_with_extremities(**kwargs: object) -> ProportionReport:
    extra = _extremity_lms()
    return _full_torso_report(
        extra_lms=extra,
        extra_diams=[
            _diam("ank_foot_l", half_width_m=0.035),
            _diam("ank_foot_r", half_width_m=0.035),
        ],
        **kwargs,  # type: ignore[arg-type]
    )


def _skeleton_extremities() -> BlockoutSkeleton:
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
        j("wrist_l", x=-0.45, y=0.0, z=0.95, side="l"),
        j("wrist_r", x=0.45, y=0.0, z=0.95, side="r"),
        j("hand_l", x=-0.48, y=0.0, z=0.88, side="l", parent="wrist_l"),
        j("hand_r", x=0.48, y=0.0, z=0.88, side="r", parent="wrist_r"),
        j("ankle_l", x=-0.10, y=0.02, z=0.08, side="l"),
        j("ankle_r", x=0.10, y=0.02, z=0.08, side="r"),
        j("heel_l", x=-0.10, y=0.06, z=0.02, side="l", parent="ankle_l"),
        j("heel_r", x=0.10, y=0.06, z=0.02, side="r", parent="ankle_r"),
        j("toe_l", x=-0.10, y=-0.12, z=0.02, side="l", parent="ankle_l"),
        j("toe_r", x=0.10, y=-0.12, z=0.02, side="r", parent="ankle_r"),
    ]
    bones = [
        SkeletonBone(id="hand_l", joint_a="wrist_l", joint_b="hand_l", length_m=0.08),
        SkeletonBone(id="hand_r", joint_a="wrist_r", joint_b="hand_r", length_m=0.08),
        SkeletonBone(id="foot_l", joint_a="heel_l", joint_b="toe_l", length_m=0.18),
        SkeletonBone(id="foot_r", joint_a="heel_r", joint_b="toe_r", length_m=0.18),
    ]
    return BlockoutSkeleton(
        schema_version="1.0.0",
        honesty="proportion_blockout_skeleton_not_mesh_or_print_success",
        joints=joints,
        bones=bones,
        messages=[],
    )


def _template_applied(height_m: float = 1.72, foot_len_scale: float = 1.4):
    from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
    from meshops.proportion.honesty import TEMPLATE_HONESTY

    return TemplateAppliedPackage(
        schema_version="1.0.0",
        honesty=TEMPLATE_HONESTY,
        template_id="female_adult_athletic",
        sex="female",
        archetype="adult_athletic",
        source_report="test",
        height_m=height_m,
        head_unit_m=height_m / 7.5,
        constants=AppliedConstants(
            breast_mode="dual_tilted",
            glute_mode_default="oval",
            torso_mode_default="trap",
            foot_len_scale=foot_len_scale,
            ank_foot_r_m=0.035,
            ank_foot_r_frac=0.035 / height_m,
        ),
    )


# ---------------------------------------------------------------------------
# R7 cases
# ---------------------------------------------------------------------------


def test_ext__default_no_extremity_roles() -> None:
    """B6 / R7: without flags, no palm/foot_plate/ank_foot roles."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False)
    roles = {p.role for p in pkg.parts}
    extremity_roles = {
        "palm",
        "finger_soft",
        "thumb_soft",
        "foot_plate",
        "heel",
        "ankle_bridge",
        "toe_soft",
        "ball_soft",
    }
    assert roles.isdisjoint(extremity_roles)
    assert pkg.schema_version == RECIPE_SCHEMA_VERSION == "1.4.0"


def test_ext__feet_wedge_roles_and_rear_third() -> None:
    """R7: --feet wedge -> L+R plate/heel/ank_foot/toe_soft; ank Y in rear third."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    by_role = pkg.counts.get("by_role") or {}
    assert by_role.get("foot_plate", 0) == 2
    assert by_role.get("heel", 0) == 2
    assert by_role.get("ankle_bridge", 0) == 2
    assert by_role.get("toe_soft", 0) == 2
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        plate = by_name[f"RECIPE_foot_plate_{side}"]
        ank = by_name[f"RECIPE_ank_foot_{side}"]
        assert plate.center is not None and ank.center is not None
        plate_y = float(plate.center[1])
        half_d = float(plate.half_depth_m or 0.0)
        rear0 = plate_y + (1.0 / 3.0) * half_d
        rear1 = plate_y + half_d
        ay = float(ank.center[1])
        assert rear0 - 1e-6 <= ay <= rear1 + 1e-6, f"ank Y {ay} not in [{rear0},{rear1}]"
        assert "ank_foot" in ank.name


def test_ext__foot_width_matches_ankle() -> None:
    """R7: plate width ≈ ankle width within FOOT_WIDTH_TOL."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, feet=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        plate = by_name[f"RECIPE_foot_plate_{side}"]
        ank = by_name[f"RECIPE_ank_foot_{side}"]
        pw = 2.0 * float(plate.top_half_width_m or 0.0)
        aw = 2.0 * float(ank.rx_m or 0.0)
        assert abs(pw - aw) <= FOOT_WIDTH_TOL_M + 1e-9


def test_ext__frame_toe_y_lt_heel_y() -> None:
    """R7: toe center Y < heel center Y (toes -Y, heels +Y)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        heel = by_name[f"RECIPE_heel_{side}"]
        toe = by_name[f"RECIPE_toe_soft_{side}"]
        assert heel.center is not None and toe.center is not None
        assert float(toe.center[1]) < float(heel.center[1])


def test_ext__toe_wedge_in_front_of_plate() -> None:
    """Toe mass sits past plate front edge (-Y), not on top of the foot plate."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        plate = by_name[f"RECIPE_foot_plate_{side}"]
        toe = by_name[f"RECIPE_toe_soft_{side}"]
        assert plate.center is not None and toe.center is not None
        half_d = float(plate.half_depth_m or 0.0)
        plate_front = float(plate.center[1]) - half_d
        # Center at or past front edge (bulk ahead of plate)
        assert float(toe.center[1]) <= plate_front + 1e-6, (
            f"toe Y {toe.center[1]} still over plate (front edge {plate_front})"
        )
        # Flat sole-class Z (not a tall ball perched on the plate)
        z_top = float(plate.z_top_m or 0.03)
        assert float(toe.center[2]) <= z_top + 0.04


def test_ext__foot_z_floor() -> None:
    """R7: plate bottom ≈ 0; sole may be box or organic ellipsoid."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, feet=True)
    for p in pkg.parts:
        if p.role == "foot_plate":
            assert p.z_bottom_m is not None
            assert abs(float(p.z_bottom_m)) < 1e-6
            assert p.z_top_m is not None and float(p.z_top_m) > 0
            assert p.center is not None
            assert float(p.center[2]) >= 0.0
            # Organic sole ellipsoid: center near sole thickness; box: mid of z span
            assert float(p.center[2]) <= float(p.z_top_m) + 1e-6
            assert p.kind in ("box", "ellipsoid")
            if p.kind == "ellipsoid":
                assert p.rx_m is not None and p.ry_m is not None and p.rz_m is not None
                assert p.half_depth_m is not None  # constraint rear-third needs this


def test_ext__toe_heel_z_not_ankle_clone() -> None:
    """Estimated heel/toe often inherit ankle Z — toe stays sole; heel pad reaches; ank high."""
    ank_z = 0.131
    extra = _extremity_lms()
    # Clone ankle height onto heel/toe (skeleton estimated pattern that floated toe_soft).
    for side in ("l", "r"):
        sx = -0.10 if side == "l" else 0.10
        extra[f"ankle_{side}"] = _lm(f"ankle_{side}", x_m=sx, y_m=0.02, z_m=ank_z)
        extra[f"heel_{side}"] = _lm(f"heel_{side}", x_m=sx, y_m=0.06, z_m=ank_z)
        extra[f"toe_{side}"] = _lm(f"toe_{side}", x_m=sx, y_m=-0.12, z_m=ank_z)
    report = _full_torso_report(
        extra_lms=extra,
        extra_diams=[
            _diam("ank_foot_l", half_width_m=0.035),
            _diam("ank_foot_r", half_width_m=0.035),
        ],
    )
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        plate = by_name[f"RECIPE_foot_plate_{side}"]
        heel = by_name[f"RECIPE_heel_{side}"]
        toe = by_name[f"RECIPE_toe_soft_{side}"]
        ank = by_name[f"RECIPE_ank_foot_{side}"]
        assert plate.center is not None and heel.center is not None
        assert toe.center is not None and ank.center is not None
        z_top = float(plate.z_top_m or 0.03)
        hz = float(heel.center[2])
        hrz = float(heel.rz_m or 0.0)
        az = float(ank.center[2])
        arz = float(ank.rz_m or 0.0)
        # Toe flat on sole — never mid-shin ankle clone
        assert float(toe.center[2]) < z_top + 0.05
        assert float(toe.center[2]) < ank_z * 0.5
        # Heel is a rear pad that reaches ank_foot (0044) — not ankle-clone sphere
        assert hz < az  # center below ankle joint
        heel_top = hz + hrz
        ank_bottom = az - arz
        assert heel_top >= ank_bottom - HEEL_REACH_GAP_TOL_M - 1e-6
        # Ankle bridge stays at real ankle height
        assert az == pytest.approx(ank_z, abs=1e-6)
        # Toe forward of heel (-Y)
        assert float(toe.center[1]) < float(heel.center[1])


def test_ext__palm_is_ellipsoid_not_box() -> None:
    """Palm must be ellipsoid (world-axis box read as cube+stick)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="mitten")
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        mitt = next(p for p in pkg.parts if p.name == f"RECIPE_finger_mitten_{side}")
        assert palm.kind == "ellipsoid"
        assert mitt.kind == "ellipsoid"
        assert palm.rx_m is not None and palm.rx_m > 0.015
        assert mitt.rx_m is not None and mitt.rx_m > 0.01


def test_ext__foot_plate_not_box() -> None:
    """Organic sole: foot_plate is ellipsoid, not square box."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    for p in pkg.parts:
        if p.role == "foot_plate":
            assert p.kind == "ellipsoid"
            assert p.ry_m is not None and p.rx_m is not None
            assert float(p.ry_m) > float(p.rx_m)  # longer heel->toe than width


def test_ext__arch_ball_embedded_in_sole() -> None:
    """Arch/ball centers stay in sole Z band — not perched above the foot."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        plate = by_name[f"RECIPE_foot_plate_{side}"]
        arch = by_name[f"RECIPE_arch_soft_{side}"]
        ball = by_name[f"RECIPE_ball_soft_{side}"]
        assert plate.center is not None and arch.center is not None and ball.center is not None
        sole_top = float(plate.center[2]) + float(plate.rz_m or 0.0)
        # Centers at or below sole top (pads live in the sole, not stacked on it)
        assert float(arch.center[2]) <= sole_top + 1e-6
        assert float(ball.center[2]) <= sole_top + 1e-6
        # Flat-ish pads: rz not much larger than sole thickness
        assert float(arch.rz_m or 0) <= sole_top * 1.6
        assert float(ball.rz_m or 0) <= sole_top * 1.6


def test_ext__mitten_radius_vs_palm() -> None:
    """T3 / B2: fat mitten cross-section r >= 0.70 x palm half-width (product pin 0.72).

    Asserts mitt.rx_m (product mitt_r on X in both hang and tip-directed branches),
    not max(rx,ry,rz) — length half-extent must not mask a thin stick.
    """
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="mitten")
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        mitt = next(p for p in pkg.parts if p.name == f"RECIPE_finger_mitten_{side}")
        assert palm.kind == "ellipsoid"
        assert mitt.kind == "ellipsoid"
        assert palm.rx_m is not None
        assert mitt.rx_m is not None
        palm_half_w = float(palm.rx_m)
        # mitt_r is always assigned to mitt_rx (hang: rx=mitt_r; tip-dir: rx=mitt_r)
        assert float(mitt.rx_m) >= 0.70 * palm_half_w


def test_ext__finger_direction_with_and_without_tip() -> None:
    """R7: with fingertip axis aligns wrist->tip; without primary -Z."""
    wrist = [0.0, 0.0, 1.0]
    tip = [0.1, -0.2, 0.7]
    axis = finger_primary_axis(wrist, tip, hand_len=0.3)
    raw = (tip[0] - wrist[0], tip[1] - wrist[1], tip[2] - wrist[2])
    n = (raw[0] ** 2 + raw[1] ** 2 + raw[2] ** 2) ** 0.5
    expected = (raw[0] / n, raw[1] / n, raw[2] / n)
    assert axis[0] == pytest.approx(expected[0], abs=1e-6)
    assert axis[1] == pytest.approx(expected[1], abs=1e-6)
    assert axis[2] == pytest.approx(expected[2], abs=1e-6)

    axis_no = finger_primary_axis(wrist, None, hand_len=0.3)
    # Primary -Z: |z| dominates over |y|
    assert abs(axis_no[2]) > abs(axis_no[1])
    assert axis_no[2] < 0
    assert abs(axis_no[0]) < 0.2


def test_ext__hands_mitten_recipe_only() -> None:
    """R7: --hands mitten -> palm + mitten per side; RECIPE_* only."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="mitten")
    by_role = pkg.counts.get("by_role") or {}
    assert by_role.get("palm", 0) == 2
    assert by_role.get("finger_soft", 0) == 2
    names = {p.name for p in pkg.parts if p.role in ("palm", "finger_soft")}
    assert "RECIPE_palm_l" in names and "RECIPE_palm_r" in names
    assert "RECIPE_finger_mitten_l" in names and "RECIPE_finger_mitten_r" in names
    for p in pkg.parts:
        assert p.name.startswith("RECIPE_")
        assert p.label.startswith("RECIPE_")
        assert not p.name.startswith("HAND_")
        assert not p.name.startswith("FOOT_")
        assert not p.name.startswith("DIGIT_")
    # Mitten is bulk ellipsoid (not thin stick capsule)
    mitt = next(p for p in pkg.parts if p.name == "RECIPE_finger_mitten_l")
    assert mitt.kind == "ellipsoid"


def test_ext__fingers_full_count() -> None:
    """R7: fingers full -> >= palm + 4x3 finger capsules per side (+ thumb 2)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    by_role = pkg.counts.get("by_role") or {}
    assert by_role.get("palm", 0) == 2
    # 4 fingers x 3 = 12 finger_soft per side -> 24; thumb is thumb_soft
    assert by_role.get("finger_soft", 0) >= 24
    assert by_role.get("thumb_soft", 0) >= 4  # 2 per side
    names = {p.name for p in pkg.parts}
    assert "RECIPE_finger_index_0_l" in names
    assert "RECIPE_finger_pinky_2_r" in names
    assert "RECIPE_thumb_soft_0_l" in names


def test_ext__classifier_b5() -> None:
    cases = [
        ("RECIPE_ank_foot_l", "ankle_bridge"),
        ("RECIPE_foot_plate_r", "foot_plate"),
        ("RECIPE_heel_l", "heel"),
        ("RECIPE_toe_soft_l", "unknown"),
        ("RECIPE_ball_soft_r", "unknown"),
        ("RECIPE_palm_l", "unknown"),
        ("RECIPE_finger_mitten_l", "unknown"),
        ("RECIPE_finger_index_0_r", "unknown"),
        ("RECIPE_thumb_soft_1_l", "unknown"),
        ("RECIPE_toe_1_l", "unknown"),
        ("RECIPE_toe_5_r", "unknown"),
    ]
    for name, expected in cases:
        role, _side = classify_part_name(name)
        assert role == expected, f"{name} -> {role}, expected {expected}"


def test_ext__ank_foot_assert_raises() -> None:
    with pytest.raises(ValueError, match="ank_foot"):
        _assert_ank_foot_name("RECIPE_ankle_l", "ankle_bridge")
    # ok path
    _assert_ank_foot_name("RECIPE_ank_foot_l", "ankle_bridge")


def test_ext__schema_write_1_4_load_1_2(tmp_path: Path) -> None:
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, feet=True)
    assert pkg.schema_version == "1.4.0"
    paths = write_blockout_recipe(tmp_path / "out14", pkg, format="json", force=True)
    loaded = load_blockout_recipe(paths[0])
    assert loaded.schema_version == "1.4.0"

    # Load 1.2.0 face recipe still OK
    face_pkg = build_blockout_recipe(report, limbs=False, face=True)
    data = face_pkg.model_dump(mode="json")
    data["schema_version"] = "1.2.0"
    p12 = tmp_path / "face_1_2.json"
    p12.write_text(json.dumps(data, indent=2), encoding="utf-8")
    loaded12 = load_blockout_recipe(p12)
    assert loaded12.schema_version == "1.2.0"


def test_ext__hand_prefix_forbidden() -> None:
    with pytest.raises(ValueError, match="RECIPE_"):
        RecipePart(
            name="HAND_palm_l",
            role="palm",
            kind="box",
            center=[0.0, 0.0, 0.9],
            top_half_width_m=0.04,
            bottom_half_width_m=0.04,
            half_depth_m=0.02,
            z_bottom_m=0.85,
            z_top_m=0.95,
        )


def test_ext__mcp_schema_and_tool_count() -> None:
    """B13: catalog 46 (0043); properties hands/feet/fingers/toes."""
    import asyncio

    pytest.importorskip("mcp")
    from mcp import Client

    from meshops.mcp import TOOL_NAMES
    from meshops.mcp.server import build_server

    assert len(TOOL_NAMES) == 46

    async def _body() -> None:
        server = build_server()
        async with Client(server) as client:
            listed = await client.list_tools()
            names = {t.name for t in listed.tools}
            assert len(names) == 46
            assert names >= TOOL_NAMES
            tool = next(t for t in listed.tools if t.name == "mesh_proportion_blockout_recipe")
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            assert schema is not None
            props = schema.get("properties") or {}
            assert "hands" in props
            assert "feet" in props
            assert "fingers" in props
            assert "toes" in props

    asyncio.run(_body())


def test_ext__validate_constraints_complete_feet() -> None:
    """R7 / B16 / 0042: complete feet wedge -> ankle/width + foot-stack connectivity pass."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    for rid in ("C_ankle_over_heel", "C_foot_width"):
        rule = by_id[rid]
        assert rule.status in ("pass", "skip"), f"{rid}: {rule.status} {rule.message}"
    for rid in (
        "C_toe_forward_of_heel",
        "C_heel_reaches_ank_foot",
        "C_toe_sole_z",
    ):
        assert by_id[rid].status == "pass", f"{rid}: {by_id[rid].status} {by_id[rid].message}"


def test_ext__template_foot_len() -> None:
    """R7: template foot len ≈ foot_len_scale/7.5 * H."""
    height = 1.72
    scale = 1.4
    # Ankle only (no heel/toe) so ladder uses template rung
    lms = {
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.0, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.0, z_m=0.08),
    }
    report = _full_torso_report(height_m=height, extra_lms=lms)
    tpl = _template_applied(height_m=height, foot_len_scale=scale)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="none", template_applied=tpl)
    expected = scale * FOOT_LEN_BASE_FRAC_H * height
    for p in pkg.parts:
        if p.role == "foot_plate":
            half = float(p.half_depth_m or 0.0)
            assert half == pytest.approx(expected / 2.0, rel=0.05)


def test_ext__invalid_fingers_toes_cli() -> None:
    r = runner.invoke(
        app,
        [
            "proportion",
            "blockout-recipe",
            "--report",
            "nonexistent.json",
            "--out",
            "out",
            "--fingers",
            "claws",
        ],
    )
    assert r.exit_code != 0
    r2 = runner.invoke(
        app,
        [
            "proportion",
            "blockout-recipe",
            "--report",
            "nonexistent.json",
            "--out",
            "out",
            "--toes",
            "hooves",
        ],
    )
    assert r2.exit_code != 0


def test_ext__mcp_invalid_enums() -> None:
    from meshops.mcp.tools import mesh_proportion_blockout_recipe

    with pytest.raises(ValueError, match="fingers"):
        mesh_proportion_blockout_recipe(
            Path("."),
            report="r.json",
            out="o",
            fingers="claws",
        )
    with pytest.raises(ValueError, match="toes"):
        mesh_proportion_blockout_recipe(
            Path("."),
            report="r.json",
            out="o",
            toes="hooves",
        )


def test_ext__skip_hand_without_wrist() -> None:
    report = _full_torso_report()  # no wrist/hand
    pkg = build_blockout_recipe(report, limbs=False, hands=True)
    assert not any(p.role == "palm" for p in pkg.parts)
    assert any("hand_" in m and "skipped" in m for m in pkg.messages)


def test_ext__skip_foot_without_joints() -> None:
    report = _full_torso_report()  # no ankle/heel/toe
    pkg = build_blockout_recipe(report, limbs=False, feet=True)
    assert not any(p.role == "foot_plate" for p in pkg.parts)
    assert any("foot_" in m and "skipped" in m for m in pkg.messages)


def test_ext__toes_full_ball_and_beads() -> None:
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="full")
    by_role = pkg.counts.get("by_role") or {}
    # arch_soft + ball_soft per side (organic foot masses)
    assert by_role.get("ball_soft", 0) == 4
    # 5 toes x 2 sides + no wedge
    assert by_role.get("toe_soft", 0) == 10
    names = {p.name for p in pkg.parts}
    assert "RECIPE_ball_soft_l" in names
    assert "RECIPE_arch_soft_l" in names
    assert "RECIPE_toe_1_l" in names
    assert "RECIPE_toe_5_r" in names
    # Soft toe names must not contain "foot"
    for n in names:
        if "toe_" in n or "ball_soft" in n or "arch_soft" in n:
            assert "foot" not in n.lower() or "ank_foot" in n.lower()


def test_ext__skeleton_parent_joint() -> None:
    report = _report_with_extremities()
    skel = _skeleton_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, feet=True, skeleton=skel)
    by_name = {p.name: p for p in pkg.parts}
    assert by_name["RECIPE_palm_l"].parent_joint in ("hand_l", "wrist_l")
    assert by_name["RECIPE_ank_foot_r"].parent_joint == "ankle_r"
    assert by_name["RECIPE_heel_l"].parent_joint in ("heel_l", "ankle_l")


# ---------------------------------------------------------------------------
# 0044 — Foot visual mass (T1-T8)
# ---------------------------------------------------------------------------


def _report_foot_span(
    *,
    heel_y: float,
    toe_y: float,
    height_m: float = 1.72,
    ank_z: float = 0.13,
    half_width_m: float = 0.028,
) -> ProportionReport:
    """Synthetic L/R feet with deliberate heel<->toe Y span (measured length)."""
    extra: dict[str, LandmarkXYZ] = {}
    for side, sx in (("l", -0.10), ("r", 0.10)):
        extra[f"ankle_{side}"] = _lm(f"ankle_{side}", x_m=sx, y_m=0.0, z_m=ank_z)
        extra[f"heel_{side}"] = _lm(f"heel_{side}", x_m=sx, y_m=heel_y, z_m=0.02)
        extra[f"toe_{side}"] = _lm(f"toe_{side}", x_m=sx, y_m=toe_y, z_m=0.02)
    return _full_torso_report(
        height_m=height_m,
        extra_lms=extra,
        extra_diams=[
            _diam("ank_foot_l", half_width_m=half_width_m),
            _diam("ank_foot_r", half_width_m=half_width_m),
        ],
    )


def test_ext__t1_short_measured_plate_ry_visual_floor() -> None:
    """T1: short measured foot + H uses FOOT_LEN_VISUAL_MIN_FRAC_H half floor."""
    # heel_y - toe_y = 0.10
    report = _report_foot_span(heel_y=0.05, toe_y=-0.05, height_m=1.72)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    floor_half = 0.5 * FOOT_LEN_VISUAL_MIN_FRAC_H * 1.72
    for p in pkg.parts:
        if p.role == "foot_plate":
            ry = float(p.ry_m or 0.0)
            half_d = float(p.half_depth_m or 0.0)
            assert ry >= floor_half - 1e-6, f"ry={ry} < floor_half={floor_half}"
            assert half_d >= floor_half - 1e-6


def test_ext__t2_long_measured_not_shrunk() -> None:
    """T2: long measured (0.28 m) -> half_depth not shrunk below measured/2."""
    # heel_y - toe_y = 0.28
    report = _report_foot_span(heel_y=0.14, toe_y=-0.14, height_m=1.72)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    measured_half = 0.14
    for p in pkg.parts:
        if p.role == "foot_plate":
            half_d = float(p.half_depth_m or 0.0)
            ry = float(p.ry_m or 0.0)
            assert half_d >= measured_half - 1e-6
            assert ry >= measured_half - 1e-6


def test_ext__t3_apply_foot_length_visual_floor_calf() -> None:
    """T3: unit floor calf_r=0.08, H=None, hw=0.02 -> >=1.55x0.16; source calf_diam."""
    msgs: list[str] = []
    out = apply_foot_length_visual_floor(
        0.10,
        height_m=None,
        half_width=0.02,
        calf_distal_r=0.08,
        messages=msgs,
        side="l",
    )
    expect = FOOT_LEN_MIN_VS_CALF_DIAM * (2.0 * 0.08)
    assert out >= expect - 1e-9
    assert out == pytest.approx(expect, abs=1e-9)
    assert any("calf_diam" in m for m in msgs)
    assert any("0.1000->" in m or "length visual floor" in m for m in msgs)


def test_ext__t4_heel_rear_pad_reach_and_cap() -> None:
    """T4: heel.y >= ank.y; heel.z < ank.z; heel_rz <= cap; C_heel_reaches pass."""
    report = _report_foot_span(heel_y=0.06, toe_y=-0.12, height_m=1.72, ank_z=0.13)
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        heel = by_name[f"RECIPE_heel_{side}"]
        ank = by_name[f"RECIPE_ank_foot_{side}"]
        assert heel.center is not None and ank.center is not None
        assert float(heel.center[1]) >= float(ank.center[1]) - 1e-9
        assert float(heel.center[2]) < float(ank.center[2])
        hrz = float(heel.rz_m or 0.0)
        az = float(ank.center[2])
        assert hrz <= HEEL_RZ_CAP_FRAC_ANK * az + 1e-6
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_heel_reaches_ank_foot"].status == "pass", by_id[
        "C_heel_reaches_ank_foot"
    ].message


def test_ext__t5_heel_not_tower_class() -> None:
    """T5: product-like emit heel_rz <= 0.48xank_z and < old tower class ~0.068."""
    report = _report_foot_span(
        heel_y=0.06,
        toe_y=-0.08,  # short-ish measured like product
        height_m=1.72,
        ank_z=0.13,
        half_width_m=0.028,
    )
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        heel = by_name[f"RECIPE_heel_{side}"]
        ank = by_name[f"RECIPE_ank_foot_{side}"]
        assert heel.center is not None and ank.center is not None
        hrz = float(heel.rz_m or 0.0)
        az = float(ank.center[2])
        assert hrz <= HEEL_RZ_CAP_FRAC_ANK * az + 1e-6
        assert hrz < 0.068 - 1e-6, f"heel_rz={hrz} still tower-class"


def test_ext__t6_organic_sole_arch_ball_toe_topology() -> None:
    """T6: ellipsoid plate; z_top≈2xsole_rz; arch always; ball when toes; wedge past front."""
    report = _report_with_extremities()
    pkg_w = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    by_name = {p.name: p for p in pkg_w.parts}
    for side in ("l", "r"):
        plate = by_name[f"RECIPE_foot_plate_{side}"]
        arch = by_name[f"RECIPE_arch_soft_{side}"]
        ball = by_name[f"RECIPE_ball_soft_{side}"]
        toe = by_name[f"RECIPE_toe_soft_{side}"]
        assert plate.kind == "ellipsoid"
        assert plate.rz_m is not None and plate.z_top_m is not None
        assert float(plate.z_top_m) == pytest.approx(2.0 * float(plate.rz_m), abs=1e-6)
        sole_top = float(plate.center[2]) + float(plate.rz_m)  # type: ignore[index]
        assert float(arch.center[2]) <= sole_top + 1e-6  # type: ignore[index]
        assert float(ball.center[2]) <= sole_top + 1e-6  # type: ignore[index]
        half_d = float(plate.half_depth_m or 0.0)
        plate_front = float(plate.center[1]) - half_d  # type: ignore[index]
        assert float(toe.center[1]) <= plate_front + 1e-6  # type: ignore[index]

    # toes=none: arch present, no forefoot ball_soft name
    pkg_n = build_blockout_recipe(report, limbs=False, feet=True, toes="none")
    names_n = {p.name for p in pkg_n.parts}
    assert "RECIPE_arch_soft_l" in names_n
    assert "RECIPE_arch_soft_r" in names_n
    assert "RECIPE_ball_soft_l" not in names_n
    assert "RECIPE_ball_soft_r" not in names_n


def test_ext__t7_constraints_complete_feet_0044() -> None:
    """T7: validate_constraints complete feet — C_foot_width + ankle + 0042 foot rules."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, feet=True, toes="wedge")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    for rid in ("C_ankle_over_heel", "C_foot_width"):
        rule = by_id[rid]
        assert rule.status in ("pass", "skip"), f"{rid}: {rule.status} {rule.message}"
    for rid in (
        "C_toe_forward_of_heel",
        "C_heel_reaches_ank_foot",
        "C_toe_sole_z",
    ):
        assert by_id[rid].status == "pass", f"{rid}: {by_id[rid].status} {by_id[rid].message}"


def test_ext__t8_mcp_catalog_stays_46() -> None:
    """T8: MCP catalog stays 46 (no new tool)."""
    from meshops.mcp import TOOL_NAMES

    assert len(TOOL_NAMES) == 46


def test_ext__build_foot_parts_existing_parts_calf_floor() -> None:
    """existing_parts calf_b feeds visual floor (B5 wiring)."""
    report = _report_foot_span(heel_y=0.04, toe_y=-0.04, height_m=1.72, half_width_m=0.02)
    # Strip height so stature floor does not dominate; calf term should win
    report = report.model_copy(update={"height_m": None})
    calf = RecipePart(
        name="RECIPE_calf_b_l",
        role="limb_segment",
        kind="ellipsoid",
        center=[-0.1, 0.0, 0.2],
        rx_m=0.08,
        ry_m=0.08,
        rz_m=0.08,
    )
    msgs: list[str] = []
    parts = build_foot_parts(
        report,
        toes="none",
        messages=msgs,
        existing_parts=[calf],
        height_m=None,
    )
    left_plates = [p for p in parts if p.name == "RECIPE_foot_plate_l"]
    assert left_plates
    half = float(left_plates[0].half_depth_m or 0.0)
    assert half >= 0.5 * FOOT_LEN_MIN_VS_CALF_DIAM * 0.16 - 1e-6
    assert any("calf_diam" in m for m in msgs)


# ---------------------------------------------------------------------------
# 0048 — Hand bulk priors (T4-T14)
# ---------------------------------------------------------------------------


def _hand_len_from_lms(lms: dict[str, LandmarkXYZ], side: str = "l") -> float:
    """Measure wrist→fingertip length in meters (0048 synthetic ≈ 0.235)."""
    w = lms[f"wrist_{side}"]
    t = lms[f"fingertip_{side}"]
    assert w.x_m is not None and w.z_m is not None
    assert t.x_m is not None and t.z_m is not None
    wy = float(w.y_m) if w.y_m is not None else 0.0
    ty = float(t.y_m) if t.y_m is not None else 0.0
    dx = float(t.x_m) - float(w.x_m)
    dy = ty - wy
    dz = float(t.z_m) - float(w.z_m)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _capsule_center(part: RecipePart) -> list[float]:
    """Midpoint of capsule endpoints; asserts p0/p1 present."""
    assert part.p0 is not None and part.p1 is not None
    p0 = part.p0
    p1 = part.p1
    return [
        0.5 * (float(p0[0]) + float(p1[0])),
        0.5 * (float(p0[1]) + float(p1[1])),
        0.5 * (float(p0[2]) + float(p1[2])),
    ]


def _require_radius_m(part: RecipePart) -> float:
    assert part.radius_m is not None
    return float(part.radius_m)


def test_ext__t4_full_finger_bulk() -> None:
    """T4: full finger radius_m == 0.16*palm_w when uncapped; always >= 0.14*palm_w."""
    report = _report_with_extremities()
    hand_len = _hand_len_from_lms(report.landmarks_xyz)
    palm_w = 0.62 * hand_len
    half_w = palm_w / 2.0
    uncapped = 0.16 * palm_w
    cap = 0.55 * half_w
    expected = min(max(uncapped, 0.006), cap)
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        idx = next(p for p in pkg.parts if p.name == f"RECIPE_finger_index_0_{side}")
        fr = _require_radius_m(idx)
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert palm.rx_m is not None
        palm_w_meas = 2.0 * float(palm.rx_m)
        assert fr == pytest.approx(expected, abs=1e-9)
        assert fr == pytest.approx(0.16 * palm_w_meas, abs=1e-9)
        assert fr >= 0.14 * palm_w_meas - 1e-12


def test_ext__t5_palm_pad_exact() -> None:
    """T5: palm.rx_m == max(0.31*hand_len, 0.02); pad axis == max(0.36*hand_len*0.78, 0.012)."""
    report = _report_with_extremities()
    hand_len = _hand_len_from_lms(report.landmarks_xyz)
    expect_rx = max(0.31 * hand_len, 0.02)
    expect_pad = max(0.36 * hand_len * 0.78, 0.012)
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert palm.kind == "ellipsoid"
        assert palm.rx_m is not None and palm.ry_m is not None and palm.rz_m is not None
        assert float(palm.rx_m) == pytest.approx(expect_rx, abs=1e-9)
        # A-pose hang (primary -Z): pad is ry
        assert float(palm.ry_m) == pytest.approx(expect_pad, abs=1e-9)


def test_ext__t6_thumb_thicker_than_index() -> None:
    """T6: thumb_soft_0.radius_m >= 1.20 x index_0.radius_m."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        thumb = next(p for p in pkg.parts if p.name == f"RECIPE_thumb_soft_0_{side}")
        idx = next(p for p in pkg.parts if p.name == f"RECIPE_finger_index_0_{side}")
        assert _require_radius_m(thumb) >= 1.20 * _require_radius_m(idx) - 1e-12


def test_ext__t7_thumb_lateral_base() -> None:
    """T7: abs(thumb.p0.x - palm.center[0]) >= 0.85 x palm.rx_m."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        thumb = next(p for p in pkg.parts if p.name == f"RECIPE_thumb_soft_0_{side}")
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert thumb.p0 is not None and palm.center is not None and palm.rx_m is not None
        lat = abs(float(thumb.p0[0]) - float(palm.center[0]))
        assert lat >= 0.85 * float(palm.rx_m) - 1e-9


def test_ext__t8_recipe_prefix_only_full_hands() -> None:
    """T8: full hands emit RECIPE_* only (no HAND_/DIGIT_ prefixes)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for p in pkg.parts:
        assert p.name.startswith("RECIPE_")
        assert p.label.startswith("RECIPE_")
        assert not p.name.startswith("HAND_")
        assert not p.name.startswith("DIGIT_")


def test_ext__t9_c_palm_ellipsoid_pass() -> None:
    """T9: C_palm_ellipsoid pass when palms present."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    assert any(p.role == "palm" for p in pkg.parts)
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_palm_ellipsoid"].status == "pass", by_id["C_palm_ellipsoid"].message


def test_ext__t10_finger_tier_counts() -> None:
    """T10: none=0 digits; mitten=1 finger_soft/side; full already covered by count test."""
    report = _report_with_extremities()
    pkg_none = build_blockout_recipe(report, limbs=False, hands=True, fingers="none")
    by_none = pkg_none.counts.get("by_role") or {}
    assert by_none.get("palm", 0) == 2
    assert by_none.get("finger_soft", 0) == 0
    assert by_none.get("thumb_soft", 0) == 0

    pkg_mitt = build_blockout_recipe(report, limbs=False, hands=True, fingers="mitten")
    by_mitt = pkg_mitt.counts.get("by_role") or {}
    assert by_mitt.get("finger_soft", 0) == 2  # one mitten per side
    assert by_mitt.get("thumb_soft", 0) == 0
    names = {p.name for p in pkg_mitt.parts}
    assert "RECIPE_finger_mitten_l" in names
    assert "RECIPE_finger_mitten_r" in names


def test_ext__t11_finger_r_floor() -> None:
    """T11: short hand forces fr floor 0.006 m."""
    # hand_len in (~0.035, ~0.060) so floor binds and cap does not
    extra = _extremity_lms()
    for side, sx in (("l", -0.45), ("r", 0.45)):
        extra[f"wrist_{side}"] = _lm(f"wrist_{side}", x_m=sx, y_m=0.0, z_m=0.95)
        extra[f"hand_{side}"] = _lm(
            f"hand_{side}", x_m=sx - 0.01 if side == "l" else sx + 0.01, y_m=0.0, z_m=0.93
        )
        extra[f"fingertip_{side}"] = _lm(
            f"fingertip_{side}",
            x_m=sx - 0.02 if side == "l" else sx + 0.02,
            y_m=0.0,
            z_m=0.91,  # ~0.045 m wrist→tip
        )
    report = _full_torso_report(
        extra_lms=extra,
        extra_diams=[
            _diam("ank_foot_l", half_width_m=0.035),
            _diam("ank_foot_r", half_width_m=0.035),
        ],
    )
    hand_len = _hand_len_from_lms(report.landmarks_xyz)
    assert 0.035 < hand_len < 0.060
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        idx = next(p for p in pkg.parts if p.name == f"RECIPE_finger_index_0_{side}")
        assert _require_radius_m(idx) >= 0.006 - 1e-12
        assert _require_radius_m(idx) == pytest.approx(0.006, abs=1e-9)


def test_ext__t12_inter_finger_groove_spacing() -> None:
    """T12: min adjacent same-side finger_0 center |Δx| >= spacing_vs_r * fr - 1e-6."""
    from meshops.proportion.extremity_recipe import _FINGER_MIN_CENTER_SPACING_VS_R

    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    finger_names = ("index", "middle", "ring", "pinky")
    for side in ("l", "r"):
        segs = []
        for fname in finger_names:
            part = next(p for p in pkg.parts if p.name == f"RECIPE_finger_{fname}_0_{side}")
            segs.append(part)
        fr = _require_radius_m(segs[0])
        xs = [_capsule_center(p)[0] for p in segs]
        min_dx = min(abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
        assert min_dx >= _FINGER_MIN_CENTER_SPACING_VS_R * fr - 1e-6, (
            f"side={side} min_dx={min_dx} fr={fr}"
        )


def test_ext__t13_thumb_palm_plane_pitch() -> None:
    """T13: thumb_dir from p0→p1 has |dy| > |dx|*0.3 OR abs(dir.y) >= 0.25."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        thumb = next(p for p in pkg.parts if p.name == f"RECIPE_thumb_soft_0_{side}")
        assert thumb.p0 is not None and thumb.p1 is not None
        dx = float(thumb.p1[0]) - float(thumb.p0[0])
        dy = float(thumb.p1[1]) - float(thumb.p0[1])
        dz = float(thumb.p1[2]) - float(thumb.p0[2])
        n = (dx * dx + dy * dy + dz * dz) ** 0.5
        assert n > 1e-12
        dir_x, dir_y = dx / n, dy / n
        assert abs(dir_y) > abs(dir_x) * 0.3 or abs(dir_y) >= 0.25, (
            f"side={side} dir=({dir_x:.4f},{dir_y:.4f}) lacks palm pitch"
        )


def test_ext__t14_thumb_clear_of_fingers() -> None:
    """T14: 3D dist(thumb_0 center, nearest finger_0 center) >= 0.75*(thumb_r+fr)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    finger_names = ("index", "middle", "ring", "pinky")
    for side in ("l", "r"):
        thumb = next(p for p in pkg.parts if p.name == f"RECIPE_thumb_soft_0_{side}")
        thumb_c = _capsule_center(thumb)
        thumb_r = _require_radius_m(thumb)
        fr = None
        min_d = float("inf")
        for fname in finger_names:
            fp = next(p for p in pkg.parts if p.name == f"RECIPE_finger_{fname}_0_{side}")
            if fr is None:
                fr = _require_radius_m(fp)
            fc = _capsule_center(fp)
            d = (
                (thumb_c[0] - fc[0]) ** 2 + (thumb_c[1] - fc[1]) ** 2 + (thumb_c[2] - fc[2]) ** 2
            ) ** 0.5
            min_d = min(min_d, d)
        assert fr is not None
        assert min_d >= 0.75 * (thumb_r + fr) - 1e-9, (
            f"side={side} min_d={min_d} need {0.75 * (thumb_r + fr)}"
        )


def test_ext__t_b11_bulk_message_full() -> None:
    """B11 / T19: hand bulk message once when fingers=full and parts non-empty."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    bulk_msgs = [m for m in pkg.messages if "hand bulk: full digits" in m]
    assert len(bulk_msgs) == 1
    assert "r_frac=0.16" in bulk_msgs[0]
    assert "palm_w=0.62" in bulk_msgs[0]
    assert "splay=1.95" in bulk_msgs[0]
    assert "palm_th=0.36" in bulk_msgs[0]
    assert "pad_ry=0.78" in bulk_msgs[0]
    assert "segs=(0.24, 0.18, 0.13)" in bulk_msgs[0]
    assert "dist_r=1.05" in bulk_msgs[0]
    assert "anti-stick" in bulk_msgs[0]

    pkg_mitt = build_blockout_recipe(report, limbs=False, hands=True, fingers="mitten")
    assert not any("hand bulk: full digits" in m for m in pkg_mitt.messages)


# ---------------------------------------------------------------------------
# 0064 — Palm pad + phalanx taper (T0, T5b, T15-T17)
# ---------------------------------------------------------------------------


def _seg_length(part: RecipePart) -> float:
    """Capsule p0→p1 length in meters."""
    assert part.p0 is not None and part.p1 is not None
    p0 = part.p0
    p1 = part.p1
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    dz = float(p1[2]) - float(p0[2])
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def test_ext__t0_0064_constants() -> None:
    """T0: 0064 freezes for palm pad + digit taper constants."""
    from meshops.proportion.extremity_recipe import (
        _FINGER_DISTAL_R_SCALE,
        _FINGER_SEG_FRACS_HAND,
        _PALM_PAD_RY_FRAC_TH,
        _PALM_THICKNESS_FRAC_HAND,
    )

    assert _PALM_THICKNESS_FRAC_HAND == 0.36
    assert _PALM_PAD_RY_FRAC_TH == 0.78
    assert _FINGER_SEG_FRACS_HAND == (0.24, 0.18, 0.13)
    assert abs(sum(_FINGER_SEG_FRACS_HAND) - 0.55) < 1e-12
    assert _FINGER_DISTAL_R_SCALE == 1.05


def test_ext__t5b_tip_directed_palm_pad() -> None:
    """T5b: tip-directed axis (abs(Y)>abs(Z)) puts pad on rz with 0.36*0.78 law."""
    extra = _extremity_lms()
    # Wrist→tip primarily -Y so finger_primary_axis is tip-directed (not A-pose -Z)
    for side, sx in (("l", -0.45), ("r", 0.45)):
        extra[f"wrist_{side}"] = _lm(f"wrist_{side}", x_m=sx, y_m=0.0, z_m=0.95)
        extra[f"hand_{side}"] = _lm(f"hand_{side}", x_m=sx, y_m=-0.08, z_m=0.95)
        extra[f"fingertip_{side}"] = _lm(f"fingertip_{side}", x_m=sx, y_m=-0.23, z_m=0.95)
    report = _full_torso_report(
        extra_lms=extra,
        extra_diams=[
            _diam("ank_foot_l", half_width_m=0.035),
            _diam("ank_foot_r", half_width_m=0.035),
        ],
    )
    hand_len = _hand_len_from_lms(report.landmarks_xyz)
    expect_pad = max(0.36 * hand_len * 0.78, 0.012)
    # Confirm axis branch: |dy| >> |dz|
    for side in ("l", "r"):
        w = report.landmarks_xyz[f"wrist_{side}"]
        t = report.landmarks_xyz[f"fingertip_{side}"]
        assert abs(float(t.y_m or 0.0) - float(w.y_m or 0.0)) > abs(
            float(t.z_m or 0.0) - float(w.z_m or 0.0)
        )
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert palm.rz_m is not None
        assert float(palm.rz_m) == pytest.approx(expect_pad, abs=1e-9)


def test_ext__t15_middle_phalanx_length_taper() -> None:
    """T15: middle finger L_dist < L_mid < L_prox (PP > MP > DP) both sides."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        p0 = next(p for p in pkg.parts if p.name == f"RECIPE_finger_middle_0_{side}")
        p1 = next(p for p in pkg.parts if p.name == f"RECIPE_finger_middle_1_{side}")
        p2 = next(p for p in pkg.parts if p.name == f"RECIPE_finger_middle_2_{side}")
        l0, l1, l2 = _seg_length(p0), _seg_length(p1), _seg_length(p2)
        assert l2 < l1 < l0, f"side={side} L=({l0:.6f},{l1:.6f},{l2:.6f})"


def test_ext__t16_distal_radius_scale() -> None:
    """T16: distal middle r == 1.05 * prox r (both sides)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        prox = next(p for p in pkg.parts if p.name == f"RECIPE_finger_middle_0_{side}")
        dist = next(p for p in pkg.parts if p.name == f"RECIPE_finger_middle_2_{side}")
        r0 = _require_radius_m(prox)
        r2 = _require_radius_m(dist)
        assert r2 == pytest.approx(1.05 * r0, abs=1e-9), f"side={side} r0={r0} r2={r2}"


def test_ext__t17_distal_lr_vs_prox_lr() -> None:
    """T17: distal L/r < 1.50 and < 0.75 * prox L/r (anti-stick dual assert)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        prox = next(p for p in pkg.parts if p.name == f"RECIPE_finger_middle_0_{side}")
        dist = next(p for p in pkg.parts if p.name == f"RECIPE_finger_middle_2_{side}")
        l0, r0 = _seg_length(prox), _require_radius_m(prox)
        l2, r2 = _seg_length(dist), _require_radius_m(dist)
        assert r0 > 0.0 and r2 > 0.0
        lr_dist = l2 / r2
        lr_prox = l0 / r0
        assert lr_dist < 1.50, f"side={side} lr_dist={lr_dist}"
        assert lr_dist < 0.75 * lr_prox, (
            f"side={side} lr_dist={lr_dist} vs 0.75*lr_prox={0.75 * lr_prox}"
        )
