"""Track 0058 - face feature softs (eye/nose/lip/brow/cheek; offline)."""

from __future__ import annotations

import pytest

import meshops.proportion.face_recipe as face_recipe_mod
from meshops.proportion.blockout_recipe import (
    _NECK_HEAD_ATTACHED_TOKENS,
    build_blockout_recipe,
)
from meshops.proportion.constraints import classify_part_name, validate_constraints
from meshops.proportion.face_recipe import (
    BROW_HALF_LEN_FRAC_EYE_R,
    BROW_R_FRAC_H,
    CHEEK_RX_FRAC_HEAD_RX,
    CHEEK_RY_FRAC_HEAD_RY,
    CHEEK_RZ_FRAC_H,
    CHEEK_X_FRAC_HEAD_RX,
    CHEEK_Y_BIAS_FRAC_RY,
    CHEEK_Z_MIX,
    EYE_RX_FRAC_R,
    EYE_RY_FRAC_R,
    EYE_RZ_FRAC_R,
    FEATURE_FACE_Y_FRAC_RY,
    JAW_RX_FRAC_HEAD_RX,
    JAW_X_BULGE_ALLOW_M,
    JAW_Y_BIAS_FRAC_RY,
    LIP_RX_FRAC_H,
    LIP_RY_FRAC_H,
    LIP_RZ_FRAC_H,
    NOSE_RX_FRAC_H,
    NOSE_RY_FRAC_H,
    NOSE_RZ_FRAC_H,
    NOSE_TIP_Y_FRAC_RY,
)
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
    include_chin: bool = True,
    head_unit_frac: float = 1.0 / 7.5,
    extra_lms: dict[str, LandmarkXYZ] | None = None,
) -> ProportionReport:
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
    }
    if include_chin:
        lms["chin"] = _lm("chin", x_m=0.0, y_m=-0.02, z_m=chin_z)
    lms["shoulder_l"] = _lm("shoulder_l", x_m=-shoulder_x, y_m=0.0, z_m=shoulder_z)
    lms["shoulder_r"] = _lm("shoulder_r", x_m=shoulder_x, y_m=0.0, z_m=shoulder_z)
    lms["hip_l"] = _lm("hip_l", x_m=-hip_x, y_m=0.0, z_m=hip_z)
    lms["hip_r"] = _lm("hip_r", x_m=hip_x, y_m=0.0, z_m=hip_z)
    lms["cranial_vertex"] = _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=chin_z + 0.18)
    lms["crotch_pubic"] = _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86)
    lms["chest_mid"] = _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25)
    lms["chest_front"] = _lm("chest_front", x_m=0.0, y_m=-0.08, z_m=1.25)
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
        _diam("head", half_width_m=0.08),
    ]
    bands = [
        _depth_band("chest", depth_m=0.24, z_frac=0.72),
        _depth_band("hip", depth_m=0.26, z_frac=0.55),
        _depth_band("cranial", depth_m=0.20, z_frac=0.92, y_mid=-0.01),
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


def _head_skeleton() -> BlockoutSkeleton:
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
        j("neck_base", x=0.0, y=0.0, z=1.40, parent="spine_high"),
        j("neck_top", x=0.0, y=0.0, z=1.48, parent="neck_base"),
        j("chin", x=0.0, y=-0.02, z=1.50, parent="neck_top"),
        j("head", x=0.0, y=-0.01, z=1.59, parent="chin"),
        j("crown", x=0.0, y=-0.01, z=1.68, parent="head"),
    ]
    bones = [
        SkeletonBone(id="spine", joint_a="pelvis", joint_b="spine_high", length_m=0.3),
        SkeletonBone(id="neck", joint_a="neck_base", joint_b="neck_top", length_m=0.08),
        SkeletonBone(id="head_bone", joint_a="neck_top", joint_b="head", length_m=0.11),
    ]
    return BlockoutSkeleton(
        schema_version="1.0.0",
        honesty="proportion_blockout_skeleton_not_mesh_or_print_success",
        joints=joints,
        bones=bones,
        messages=[],
    )


_PUBLIC_FEATURE_CONSTS: tuple[str, ...] = (
    "EYE_RX_FRAC_R",
    "EYE_RY_FRAC_R",
    "EYE_RZ_FRAC_R",
    "FEATURE_FACE_Y_FRAC_RY",
    "NOSE_RX_FRAC_H",
    "NOSE_RY_FRAC_H",
    "NOSE_RZ_FRAC_H",
    "NOSE_TIP_Y_FRAC_RY",
    "LIP_RX_FRAC_H",
    "LIP_RY_FRAC_H",
    "LIP_RZ_FRAC_H",
    "BROW_R_FRAC_H",
    "BROW_HALF_LEN_FRAC_EYE_R",
    "CHEEK_RX_FRAC_HEAD_RX",
    "CHEEK_RY_FRAC_HEAD_RY",
    "CHEEK_RZ_FRAC_H",
    "CHEEK_X_FRAC_HEAD_RX",
    "CHEEK_Z_MIX",
    "CHEEK_Y_BIAS_FRAC_RY",
)


def test_feature_softs__t0_constants_and_all() -> None:
    """T0: Constants match freezes; all public names in face_recipe.__all__."""
    assert pytest.approx(1.00) == EYE_RX_FRAC_R
    assert pytest.approx(0.95) == EYE_RY_FRAC_R  # B15 D7 left pad (was 0.85)
    assert pytest.approx(0.45) == EYE_RZ_FRAC_R
    assert pytest.approx(0.90) == FEATURE_FACE_Y_FRAC_RY  # D7 near-surface plane
    assert pytest.approx(0.045) == NOSE_RX_FRAC_H
    assert pytest.approx(0.055) == NOSE_RY_FRAC_H
    assert pytest.approx(0.040) == NOSE_RZ_FRAC_H
    assert pytest.approx(0.98) == NOSE_TIP_Y_FRAC_RY  # surface-readable tip (was 0.15 embed)
    assert pytest.approx(0.12) == LIP_RX_FRAC_H
    assert pytest.approx(0.035) == LIP_RY_FRAC_H
    assert pytest.approx(0.025) == LIP_RZ_FRAC_H
    assert pytest.approx(0.028) == BROW_R_FRAC_H
    assert pytest.approx(1.1) == BROW_HALF_LEN_FRAC_EYE_R
    assert pytest.approx(0.28) == CHEEK_RX_FRAC_HEAD_RX
    assert pytest.approx(0.22) == CHEEK_RY_FRAC_HEAD_RY
    assert pytest.approx(0.06) == CHEEK_RZ_FRAC_H
    assert pytest.approx(0.55) == CHEEK_X_FRAC_HEAD_RX
    assert pytest.approx(0.50) == CHEEK_Z_MIX
    assert pytest.approx(0.05) == CHEEK_Y_BIAS_FRAC_RY
    for name in _PUBLIC_FEATURE_CONSTS:
        assert name in face_recipe_mod.__all__, f"{name} missing from __all__"


def test_feature_softs__t1_eye_product_like_h() -> None:
    """T1: Eye product-like H: ry ≥ rz; ry/rz ≥ 1.2."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    eyes = [p for p in pkg.parts if p.role == "eye_soft"]
    assert len(eyes) == 2
    for eye in eyes:
        assert eye.ry_m is not None and eye.rz_m is not None
        assert float(eye.ry_m) >= float(eye.rz_m)
        assert float(eye.ry_m) / float(eye.rz_m) >= 1.2


def test_feature_softs__t2_nose_ellipsoid() -> None:
    """T2: Nose kind ellipsoid; rx_m ≥ 0.040 * H; front surface tip Y spirit."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    nose = next(p for p in pkg.parts if p.name == "RECIPE_nose_soft")
    assert nose.kind == "ellipsoid"
    assert nose.rx_m is not None and nose.ry_m is not None and nose.center is not None
    assert head.rz_m is not None and head.center is not None and head.ry_m is not None
    # H ≈ 2 * head.rz for full3d ellipsoid head
    h = 2.0 * float(head.rz_m)
    assert float(nose.rx_m) >= 0.040 * h - 1e-9
    expected_tip_y = float(head.center[1]) - NOSE_TIP_Y_FRAC_RY * float(head.ry_m)
    assert float(nose.center[1]) - float(nose.ry_m) == pytest.approx(expected_tip_y, abs=1e-6)


def test_feature_softs__t3_lip_brow_floors() -> None:
    """T3: Lip ry/rz floors; brow capsule r floor."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert head.rz_m is not None
    h = 2.0 * float(head.rz_m)
    lip = next(p for p in pkg.parts if p.name == "RECIPE_lip_soft")
    assert lip.ry_m is not None and lip.rz_m is not None
    assert float(lip.ry_m) == pytest.approx(LIP_RY_FRAC_H * h, abs=1e-6)
    assert float(lip.rz_m) == pytest.approx(LIP_RZ_FRAC_H * h, abs=1e-6)
    brows = [p for p in pkg.parts if p.role == "brow_soft"]
    assert len(brows) == 2
    for brow in brows:
        assert brow.kind == "capsule"
        assert brow.radius_m is not None
        assert float(brow.radius_m) == pytest.approx(BROW_R_FRAC_H * h, abs=1e-6)


def test_feature_softs__t4_cheek_present() -> None:
    """T4: cheek_soft L/R present with face=True; exactly two; roles correct."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    cheeks = [p for p in pkg.parts if "cheek_soft" in p.name.lower()]
    assert len(cheeks) == 2
    names = {p.name for p in cheeks}
    assert names == {"RECIPE_cheek_soft_l", "RECIPE_cheek_soft_r"}
    assert all(p.role == "cheek_soft" for p in cheeks)
    assert all(p.kind == "ellipsoid" for p in cheeks)
    # without face: none
    bare = build_blockout_recipe(report, limbs=False, face=False)
    assert not any("cheek_soft" in p.name.lower() for p in bare.parts)


def test_feature_softs__t5_classify_and_axial() -> None:
    """T5: classify RECIPE_cheek_soft_l → head; axial non-fail with face."""
    role, _side = classify_part_name("RECIPE_cheek_soft_l")
    assert role == "head"
    role_r, _ = classify_part_name("RECIPE_cheek_soft_r")
    assert role_r == "head"
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    axial = by_id["C_axial_depth_plane"]
    assert axial.status != "fail", axial.message
    assert "cheek_soft" not in (axial.message or "")


def test_feature_softs__t6_jaw_freezes_untouched() -> None:
    """T6: JAW_* freezes 0.74 / 0.08 / 0.006 unchanged."""
    assert pytest.approx(0.74) == JAW_RX_FRAC_HEAD_RX
    assert pytest.approx(0.08) == JAW_Y_BIAS_FRAC_RY
    assert pytest.approx(0.006) == JAW_X_BULGE_ALLOW_M


def test_feature_softs__t7_loomis_z_private() -> None:
    """T7: Loomis Z private fracs still 0.50/0.67/0.33/0.20."""
    assert pytest.approx(0.50) == face_recipe_mod._EYE_Z_FRAC
    assert pytest.approx(0.67) == face_recipe_mod._BROW_Z_FRAC
    assert pytest.approx(0.33) == face_recipe_mod._NOSE_BASE_Z_FRAC
    assert pytest.approx(0.20) == face_recipe_mod._LIP_Z_FRAC


def test_feature_softs__t8_messages_feature_softs() -> None:
    """T8: messages include feature soft lines."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    blob = " ".join(pkg.messages)
    assert "eye soft axes" in blob
    assert "nose soft ellipsoid" in blob
    assert "lip soft axes" in blob
    assert "brow soft capsule" in blob
    assert "cheek soft pads present" in blob


def test_feature_softs__t9_neck_head_attached_token() -> None:
    """T9: _NECK_HEAD_ATTACHED_TOKENS contains cheek_soft (0050 co-move)."""
    assert "cheek_soft" in _NECK_HEAD_ATTACHED_TOKENS


def test_feature_softs__cheek_center_geometry() -> None:
    """Sanity: cheek center uses freezes vs head/eye/nose Loomis Z."""
    report = _full_torso_report()
    skel = _head_skeleton()
    pkg = build_blockout_recipe(report, limbs=False, face=True, skeleton=skel)
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    eye = next(p for p in pkg.parts if p.name == "RECIPE_eye_soft_l")
    cheek_l = next(p for p in pkg.parts if p.name == "RECIPE_cheek_soft_l")
    cheek_r = next(p for p in pkg.parts if p.name == "RECIPE_cheek_soft_r")
    assert head.center is not None and head.rx_m is not None and head.ry_m is not None
    assert head.rz_m is not None and eye.center is not None and cheek_l.center is not None
    assert cheek_r.center is not None
    rx = float(head.rx_m)
    ry = float(head.ry_m)
    h = 2.0 * float(head.rz_m)
    feature_face_y = float(head.center[1]) - FEATURE_FACE_Y_FRAC_RY * ry
    # z_chin from head: z_c - rz
    z_chin = float(head.center[2]) - float(head.rz_m)
    eye_z = z_chin + 0.50 * h
    nose_base_z = z_chin + 0.33 * h
    expected_z = CHEEK_Z_MIX * eye_z + (1.0 - CHEEK_Z_MIX) * nose_base_z
    expected_y = feature_face_y + CHEEK_Y_BIAS_FRAC_RY * ry
    assert cheek_l.center[0] == pytest.approx(-CHEEK_X_FRAC_HEAD_RX * rx, abs=1e-6)
    assert cheek_r.center[0] == pytest.approx(CHEEK_X_FRAC_HEAD_RX * rx, abs=1e-6)
    assert cheek_l.center[1] == pytest.approx(expected_y, abs=1e-6)
    assert cheek_l.center[2] == pytest.approx(expected_z, abs=1e-6)
    assert cheek_l.rx_m == pytest.approx(CHEEK_RX_FRAC_HEAD_RX * rx, abs=1e-6)
    assert cheek_l.ry_m == pytest.approx(CHEEK_RY_FRAC_HEAD_RY * ry, abs=1e-6)
    assert cheek_l.rz_m == pytest.approx(CHEEK_RZ_FRAC_H * h, abs=1e-6)
