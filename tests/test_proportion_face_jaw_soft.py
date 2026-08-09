"""Track 0055 - face jaw soft mass ellipsoid (offline; no Blender / no network)."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest

import meshops.proportion.face_recipe as face_recipe_mod
from meshops.proportion.blockout_recipe import (
    RecipePart,
    build_blockout_recipe,
    emit_bpy_script,
    load_blockout_recipe,
)
from meshops.proportion.constraints import classify_part_name, validate_constraints
from meshops.proportion.face_recipe import (
    JAW_RX_FRAC_HEAD_RX,
    JAW_RY_FRAC_HEAD_RY,
    JAW_RZ_FRAC_H,
    JAW_X_BULGE_ALLOW_M,
    JAW_Y_BIAS_FRAC_RY,
    JAW_Z_CENTER_FRAC_H,
    HeadBounds,
    build_face_parts,
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


def _synthetic_bounds(
    *,
    z_chin: float = 1.50,
    h: float = 0.18,
    rx: float = 0.08,
    ry: float = 0.09,
    rz: float | None = None,
    y: float = -0.02,
) -> HeadBounds:
    z_top = z_chin + h
    z_c = (z_chin + z_top) / 2.0
    return HeadBounds(
        z_chin=z_chin,
        z_top=z_top,
        z_c=z_c,
        H=h,
        rx=rx,
        ry=ry,
        rz=float(rz) if rz is not None else h / 2.0,
        y=y,
        placement="full3d",
        has_y=True,
        top_from_landmark=True,
    )


def _jaw_part(parts: list[RecipePart]) -> RecipePart:
    jaw = next((p for p in parts if p.name == "RECIPE_jaw"), None)
    assert jaw is not None, "RECIPE_jaw missing"
    return jaw


# ---------------------------------------------------------------------------
# T0-T14
# ---------------------------------------------------------------------------


def test_jaw_soft__t0_constants_exported() -> None:
    """T0: JAW_* + JAW_X_BULGE_ALLOW_M importable; defaults match freezes."""
    assert pytest.approx(0.85) == JAW_RX_FRAC_HEAD_RX
    assert pytest.approx(0.55) == JAW_RY_FRAC_HEAD_RY
    assert pytest.approx(0.15) == JAW_RZ_FRAC_H
    assert pytest.approx(0.13) == JAW_Z_CENTER_FRAC_H
    assert pytest.approx(0.05) == JAW_Y_BIAS_FRAC_RY
    assert pytest.approx(0.015) == JAW_X_BULGE_ALLOW_M
    for name in (
        "JAW_RX_FRAC_HEAD_RX",
        "JAW_RY_FRAC_HEAD_RY",
        "JAW_RZ_FRAC_H",
        "JAW_Z_CENTER_FRAC_H",
        "JAW_Y_BIAS_FRAC_RY",
        "JAW_X_BULGE_ALLOW_M",
    ):
        assert name in face_recipe_mod.__all__


def test_jaw_soft__t1_kind_ellipsoid() -> None:
    """T1: face=True → RECIPE_jaw.kind == ellipsoid."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    jaw = _jaw_part(pkg.parts)
    assert jaw.kind == "ellipsoid"
    assert jaw.role == "jaw"


def test_jaw_soft__t1b_bpy_ensure_ellipsoid() -> None:
    """T1b: bpy script routes RECIPE_jaw via ensure_ellipsoid, not ensure_box."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    script = emit_bpy_script(pkg)
    assert "def ensure_ellipsoid(" in script
    assert "elif kind == 'ellipsoid'" in script
    idx = script.index("'name': 'RECIPE_jaw'")
    chunk = script[idx : idx + 280]
    assert "'kind': 'ellipsoid'" in chunk
    assert "'kind': 'box'" not in chunk


def test_jaw_soft__t2_ellipsoid_fields_not_box() -> None:
    """T2: rx/ry/rz set; box fields null."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    jaw = _jaw_part(pkg.parts)
    assert jaw.rx_m is not None and jaw.rx_m > 0
    assert jaw.ry_m is not None and jaw.ry_m > 0
    assert jaw.rz_m is not None and jaw.rz_m > 0
    assert jaw.top_half_width_m is None
    assert jaw.bottom_half_width_m is None
    assert jaw.half_depth_m is None
    assert jaw.z_bottom_m is None
    assert jaw.z_top_m is None


def test_jaw_soft__t3_extents_match_fracs() -> None:
    """T3: synthetic HeadBounds extents match fracs ±1e-6."""
    bounds = _synthetic_bounds(h=0.20, rx=0.10, ry=0.12, rz=0.11)
    report = _full_torso_report()
    msgs: list[str] = []
    parts = build_face_parts(report, bounds, face=True, messages=msgs)
    jaw = _jaw_part(parts)
    assert jaw.rx_m == pytest.approx(JAW_RX_FRAC_HEAD_RX * bounds.rx, abs=1e-6)
    assert jaw.ry_m == pytest.approx(JAW_RY_FRAC_HEAD_RY * bounds.ry, abs=1e-6)
    assert jaw.rz_m == pytest.approx(JAW_RZ_FRAC_H * bounds.H, abs=1e-6)


def test_jaw_soft__t4_center_z_and_chin_extent() -> None:
    """T4: center_z = z_chin + 0.13*H; center_z - rz ~ z_chin - 0.02*H."""
    bounds = _synthetic_bounds(z_chin=1.50, h=0.20, rx=0.09, ry=0.10)
    report = _full_torso_report()
    parts = build_face_parts(report, bounds, face=True, messages=[])
    jaw = _jaw_part(parts)
    assert jaw.center is not None and jaw.rz_m is not None
    expected_z = bounds.z_chin + JAW_Z_CENTER_FRAC_H * bounds.H
    assert jaw.center[2] == pytest.approx(expected_z, abs=1e-6)
    bottom = float(jaw.center[2]) - float(jaw.rz_m)
    expected_bottom = bounds.z_chin - 0.02 * bounds.H
    assert bottom == pytest.approx(expected_bottom, abs=1e-6)


def test_jaw_soft__t5_center_y_face_plane() -> None:
    """T5: center Y = face_y + 0.05*ry with face_y = head.y - 0.40*ry."""
    bounds = _synthetic_bounds(y=-0.03, ry=0.10)
    report = _full_torso_report()
    parts = build_face_parts(report, bounds, face=True, messages=[])
    jaw = _jaw_part(parts)
    assert jaw.center is not None
    face_y = bounds.y - 0.40 * bounds.ry
    expected_y = face_y + JAW_Y_BIAS_FRAC_RY * bounds.ry
    assert jaw.center[1] == pytest.approx(expected_y, abs=1e-6)


def test_jaw_soft__t6_parent_joint_chin_or_head() -> None:
    """T6: parent joint chin/head when skeleton present."""
    report = _full_torso_report()
    skel = _head_skeleton()
    pkg = build_blockout_recipe(report, limbs=False, face=True, skeleton=skel)
    jaw = _jaw_part(pkg.parts)
    assert jaw.parent_joint in ("chin", "head")


def test_jaw_soft__t7_without_face_no_jaw() -> None:
    """T7: without face flag, no jaw role."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=False)
    assert not any(p.role == "jaw" for p in pkg.parts)
    assert not any(p.name == "RECIPE_jaw" for p in pkg.parts)


def test_jaw_soft__t8_rest_of_face_kit() -> None:
    """T8: rest of face kit still emits (eyes≥2, nose, ears≥2, lip)."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    by_role = pkg.counts.get("by_role") or {}
    assert by_role.get("eye_soft", 0) >= 2
    assert by_role.get("nose_soft", 0) >= 1
    assert by_role.get("ear_soft", 0) >= 2
    assert by_role.get("lip_soft", 0) >= 1
    assert by_role.get("jaw", 0) >= 1


def test_jaw_soft__t9_messages_soft_mass_and_bulge() -> None:
    """T9: message contains soft-mass note + bulge metric when head present."""
    bounds = _synthetic_bounds(h=0.20, rx=0.09, ry=0.10, rz=0.11)
    report = _full_torso_report()
    msgs: list[str] = []
    build_face_parts(report, bounds, face=True, messages=msgs)
    soft = [m for m in msgs if "jaw soft mass ellipsoid" in m]
    assert soft, msgs
    assert "rx=" in soft[0] and "ry=" in soft[0] and "rz=" in soft[0]
    bulge_msgs = [m for m in msgs if "jaw_vs_head_x_bulge_m=" in m]
    assert bulge_msgs, msgs
    assert (
        f"allow={JAW_X_BULGE_ALLOW_M}" in bulge_msgs[0]
        or f"allow={JAW_X_BULGE_ALLOW_M:g}" in bulge_msgs[0]
    )


def test_jaw_soft__t10_legacy_box_jaw_loads(tmp_path: Path) -> None:
    """T10: load legacy recipe JSON with box jaw still loads (1.2.0 / 1.4.0)."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False)
    data = pkg.model_dump(mode="json")
    data["schema_version"] = "1.2.0"
    data["parts"].append(
        {
            "name": "RECIPE_jaw",
            "role": "jaw",
            "kind": "box",
            "center": [0.0, -0.03, 1.53],
            "p0": None,
            "p1": None,
            "top_half_width_m": 0.075,
            "bottom_half_width_m": 0.064,
            "half_depth_m": 0.05,
            "z_bottom_m": 1.505,
            "z_top_m": 1.568,
            "rx_m": None,
            "ry_m": None,
            "rz_m": None,
            "radius_m": None,
            "placement": "full3d",
            "label": "RECIPE_jaw",
            "notes": None,
            "parent_joint": "chin",
            "rotation_euler_deg": None,
        }
    )
    p12 = tmp_path / "legacy_box_jaw_1_2.json"
    p12.write_text(json.dumps(data, indent=2), encoding="utf-8")
    loaded12 = load_blockout_recipe(p12)
    assert loaded12.schema_version == "1.2.0"
    jaw12 = _jaw_part(loaded12.parts)
    assert jaw12.kind == "box"
    assert jaw12.top_half_width_m == pytest.approx(0.075)

    data14 = dict(data)
    data14["schema_version"] = "1.4.0"
    p14 = tmp_path / "legacy_box_jaw_1_4.json"
    p14.write_text(json.dumps(data14, indent=2), encoding="utf-8")
    loaded14 = load_blockout_recipe(p14)
    assert loaded14.schema_version == "1.4.0"
    jaw14 = _jaw_part(loaded14.parts)
    assert jaw14.kind == "box"


def test_jaw_soft__t11_classifier_and_axial() -> None:
    """T11: classifier RECIPE_jaw → head; axial rule does not fail on jaw Y."""
    role, _side = classify_part_name("RECIPE_jaw")
    assert role == "head"
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    axial = by_id["C_axial_depth_plane"]
    assert axial.status != "fail", axial.message
    assert "RECIPE_jaw" not in (axial.message or "") or "fail" not in (axial.message or "")


def test_jaw_soft__t11b_axial_exemption_covers_jaw() -> None:
    """T11b: B18 marks RECIPE_jaw axial-exempt; C_axial_depth_plane not fail with face."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    axial = by_id["C_axial_depth_plane"]
    assert axial.status != "fail", axial.message
    metrics = axial.metrics or {}
    # B18 locks face softs via name-token exemption metrics
    assert metrics.get("RECIPE_jaw_axial_exempt") is True


def test_jaw_soft__t12_neck_comove_jaw_present() -> None:
    """T12: neck co-move path with face=True still finds jaw (and shifts faceward)."""
    report = _full_torso_report()
    # Ensure chin Y for head center so co-move has a base
    lms = dict(report.landmarks_xyz)
    lms["chin"] = _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50)
    lms["chest_mid"] = _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25)
    report = report.model_copy(update={"landmarks_xyz": lms})
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    jaw = _jaw_part(pkg.parts)
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert jaw.center is not None and head.center is not None
    # Jaw remains present after neck column priors; faceward of head center Y
    assert jaw.kind == "ellipsoid"
    assert float(jaw.center[1]) < float(head.center[1])


def test_jaw_soft__t13_product_class_bounds_and_bulge() -> None:
    """T13: product-class HeadBounds extents + Z chin + bulge ≤ allow."""
    h = 0.2102
    z_chin = 1.5098
    rx = 0.0883
    ry = 0.0908
    rz = 0.1103  # scaled head surface (not H/2)
    bounds = HeadBounds(
        z_chin=z_chin,
        z_top=z_chin + h,
        z_c=z_chin + h / 2.0,
        H=h,
        rx=rx,
        ry=ry,
        rz=rz,
        y=0.0,
        placement="front_plane",
        has_y=True,
        top_from_landmark=True,
    )
    report = _full_torso_report()
    msgs: list[str] = []
    parts = build_face_parts(report, bounds, face=True, messages=msgs)
    jaw = _jaw_part(parts)
    assert jaw.kind == "ellipsoid"
    assert jaw.rx_m == pytest.approx(0.85 * rx, abs=1e-4)  # ~0.0750
    assert jaw.ry_m == pytest.approx(0.55 * ry, abs=1e-4)  # ~0.0499
    assert jaw.rz_m == pytest.approx(0.15 * h, abs=1e-4)  # ~0.0315
    assert jaw.center is not None
    expected_z = z_chin + 0.13 * h
    assert jaw.center[2] == pytest.approx(expected_z, abs=1e-4)
    bottom = float(jaw.center[2]) - float(jaw.rz_m or 0.0)
    assert bottom == pytest.approx(z_chin - 0.02 * h, abs=1e-4)

    jaw_rx = float(jaw.rx_m or 0.0)
    t = (float(jaw.center[2]) - bounds.z_c) / bounds.rz
    head_x = bounds.rx * math.sqrt(max(0.0, 1.0 - t * t))
    bulge = jaw_rx - head_x
    assert bulge == pytest.approx(0.0124, abs=2e-3)
    assert bulge <= JAW_X_BULGE_ALLOW_M
    assert any("jaw_vs_head_x_bulge_m=" in m for m in msgs)


def test_jaw_soft__t14_box_helper_absent() -> None:
    """T14: _box absent from face_recipe module (B14)."""
    assert not hasattr(face_recipe_mod, "_box")
    src = inspect.getsource(face_recipe_mod)
    assert "def _box(" not in src
    assert "_JAW_HALF_WIDTH_FRAC_RX" not in src
