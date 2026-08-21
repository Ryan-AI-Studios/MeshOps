"""Track 0078 — face jaw / lip pad refine (offline; no Blender / no network)."""

from __future__ import annotations

import math

import pytest

import meshops.proportion.face_recipe as face_recipe_mod
from meshops.proportion.blockout_recipe import RecipePart, build_blockout_recipe
from meshops.proportion.face_recipe import (
    FEATURE_FACE_Y_FRAC_RY,
    JAW_RX_FRAC_HEAD_RX,
    JAW_RY_FRAC_HEAD_RY,
    JAW_RZ_FRAC_H,
    JAW_X_BULGE_ALLOW_M,
    JAW_Y_BIAS_FRAC_RY,
    JAW_Z_CENTER_FRAC_H,
    LIP_RY_FRAC_H,
    LIP_RZ_FRAC_H,
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


def _product_class_bounds() -> HeadBounds:
    """Product-class HeadBounds (nofuse_0081-class inventory)."""
    h = 0.2102
    z_chin = 1.5098
    return HeadBounds(
        z_chin=z_chin,
        z_top=z_chin + h,
        z_c=z_chin + h / 2.0,
        H=h,
        rx=0.0883,
        ry=0.0908,
        rz=0.1103,
        y=-0.0269,
        placement="front_plane",
        has_y=True,
        top_from_landmark=True,
    )


def _jaw_part(parts: list[RecipePart]) -> RecipePart:
    jaw = next((p for p in parts if p.name == "RECIPE_jaw"), None)
    assert jaw is not None, "RECIPE_jaw missing"
    return jaw


def _lip_part(parts: list[RecipePart]) -> RecipePart:
    lip = next((p for p in parts if p.name == "RECIPE_lip_soft"), None)
    assert lip is not None, "RECIPE_lip_soft missing"
    return lip


def test_jaw_lip_pad__t_jaw_ry_thin() -> None:
    """T_jaw_ry_thin: jaw.ry / head.ry ≈ 0.42 (±0.02)."""
    bounds = _product_class_bounds()
    report = _full_torso_report()
    parts = build_face_parts(report, bounds, face=True, messages=[])
    jaw = _jaw_part(parts)
    assert jaw.ry_m is not None
    ratio = float(jaw.ry_m) / bounds.ry
    assert ratio == pytest.approx(JAW_RY_FRAC_HEAD_RY, abs=0.02)
    assert ratio == pytest.approx(0.42, abs=0.02)


def test_jaw_lip_pad__t_jaw_rz_chin_flush() -> None:
    """T_jaw_rz_chin_flush: jaw.center_z - jaw.rz ≈ z_chin (±2 mm)."""
    bounds = _product_class_bounds()
    report = _full_torso_report()
    parts = build_face_parts(report, bounds, face=True, messages=[])
    jaw = _jaw_part(parts)
    assert jaw.center is not None and jaw.rz_m is not None
    bottom = float(jaw.center[2]) - float(jaw.rz_m)
    assert bottom == pytest.approx(bounds.z_chin, abs=0.002)


def test_jaw_lip_pad__t_lip_thinner() -> None:
    """T_lip_thinner: lip ry/rz/rx under 0078 ceilings vs H."""
    bounds = _product_class_bounds()
    report = _full_torso_report()
    parts = build_face_parts(report, bounds, face=True, messages=[])
    lip = _lip_part(parts)
    h = bounds.H
    assert lip.ry_m is not None and lip.rz_m is not None and lip.rx_m is not None
    assert float(lip.ry_m) <= 0.032 * h + 1e-9
    assert float(lip.rz_m) <= 0.024 * h + 1e-9
    assert float(lip.rx_m) <= 0.105 * h + 1e-9
    assert float(lip.ry_m) == pytest.approx(LIP_RY_FRAC_H * h, abs=1e-6)
    assert float(lip.rz_m) == pytest.approx(LIP_RZ_FRAC_H * h, abs=1e-6)
    assert float(lip.rx_m) == pytest.approx(0.10 * h, abs=1e-6)


def test_jaw_lip_pad__t_planes_fence() -> None:
    """T_planes_fence: FEATURE_FACE_Y 0.90; jaw Y uses 0.40 plane + bias."""
    assert pytest.approx(0.90) == FEATURE_FACE_Y_FRAC_RY
    assert pytest.approx(0.40) == face_recipe_mod._JAW_FACE_Y_FRAC_RY
    bounds = _product_class_bounds()
    report = _full_torso_report()
    parts = build_face_parts(report, bounds, face=True, messages=[])
    jaw = _jaw_part(parts)
    lip = _lip_part(parts)
    assert jaw.center is not None and lip.center is not None
    jaw_face_y = bounds.y - face_recipe_mod._JAW_FACE_Y_FRAC_RY * bounds.ry
    expected_jaw_y = jaw_face_y + JAW_Y_BIAS_FRAC_RY * bounds.ry
    assert float(jaw.center[1]) == pytest.approx(expected_jaw_y, abs=1e-6)
    feature_face_y = bounds.y - FEATURE_FACE_Y_FRAC_RY * bounds.ry
    assert float(lip.center[1]) == pytest.approx(feature_face_y, abs=1e-6)


def test_jaw_lip_pad__t_no_chin_soft() -> None:
    """T_no_chin_soft: no chin_soft / RECIPE_chin names."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    names = [p.name for p in pkg.parts]
    blob = " ".join(names).lower()
    assert "chin_soft" not in blob
    assert not any(n.startswith("RECIPE_chin") for n in names)
    assert any(p.name == "RECIPE_jaw" for p in pkg.parts)
    assert any(p.name == "RECIPE_lip_soft" for p in pkg.parts)


def test_jaw_lip_pad__t_fence_0058() -> None:
    """T_fence_0058: eye, nose, cheek L/R still present with face=True."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    by_name = {p.name: p for p in pkg.parts}
    assert "RECIPE_eye_soft_l" in by_name
    assert "RECIPE_eye_soft_r" in by_name
    assert "RECIPE_nose_soft" in by_name
    assert by_name["RECIPE_nose_soft"].kind == "ellipsoid"
    assert "RECIPE_cheek_soft_l" in by_name
    assert "RECIPE_cheek_soft_r" in by_name
    eyes = [p for p in pkg.parts if p.role == "eye_soft"]
    assert len(eyes) == 2
    for eye in eyes:
        assert eye.ry_m is not None and eye.rz_m is not None
        assert float(eye.ry_m) > 0 and float(eye.rz_m) > 0


def test_jaw_lip_pad__t_bulge_cap() -> None:
    """T_bulge_cap: bulge ≤ JAW_X_BULGE_ALLOW_M (rx / Z_CENTER unchanged)."""
    bounds = _product_class_bounds()
    report = _full_torso_report()
    msgs: list[str] = []
    parts = build_face_parts(report, bounds, face=True, messages=msgs)
    jaw = _jaw_part(parts)
    assert jaw.center is not None and jaw.rx_m is not None
    jaw_rx = float(jaw.rx_m)
    t = (float(jaw.center[2]) - bounds.z_c) / bounds.rz
    head_x = bounds.rx * math.sqrt(max(0.0, 1.0 - t * t))
    bulge = jaw_rx - head_x
    assert bulge <= JAW_X_BULGE_ALLOW_M
    assert jaw.rx_m == pytest.approx(JAW_RX_FRAC_HEAD_RX * bounds.rx, abs=1e-4)
    assert jaw.rz_m == pytest.approx(JAW_RZ_FRAC_H * bounds.H, abs=1e-4)
    assert jaw.center[2] == pytest.approx(bounds.z_chin + JAW_Z_CENTER_FRAC_H * bounds.H, abs=1e-4)


def test_jaw_lip_pad__t_lip_message_rx() -> None:
    """B12 soft: lip message includes rx=."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    blob = " ".join(pkg.messages)
    assert "lip soft axes" in blob
    assert "rx=" in blob
    lip_msgs = [m for m in pkg.messages if "lip soft axes" in m]
    assert lip_msgs
    assert "rx=" in lip_msgs[0]
    assert "ry=" in lip_msgs[0]
    assert "rz=" in lip_msgs[0]
