"""Track 0085 — head / face hierarchy (orbital scale + lip Z + pitch).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 46 stay. Not photoreal / MediaPipe / chin_soft / 0086 nape.
"""

from __future__ import annotations

import math

import pytest

import meshops.proportion.blockout_recipe as blockout_recipe_mod
import meshops.proportion.face_recipe as face_recipe_mod
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    HEAD_PITCH_DEG,
    NECK_FORWARD_TILT_DEG,
    RECIPE_SCHEMA_VERSION,
    RecipePart,
    _apply_head_pitch,
    _rotate_yz_about_x,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.face_recipe import (
    BROW_R_FRAC_H,
    CHEEK_RX_FRAC_HEAD_RX,
    EYE_RADIUS_FRAC_H,
    EYE_RX_FRAC_R,
    EYE_RY_FRAC_R,
    EYE_RZ_FRAC_R,
    FEATURE_FACE_Y_FRAC_RY,
    JAW_RX_FRAC_HEAD_RX,
    JAW_RY_FRAC_HEAD_RY,
    JAW_RZ_FRAC_H,
    JAW_X_BULGE_ALLOW_M,
    JAW_Y_BIAS_FRAC_RY,
    JAW_Z_CENTER_FRAC_H,
    NOSE_RX_FRAC_H,
    NOSE_RY_FRAC_H,
    NOSE_RZ_FRAC_H,
    NOSE_TIP_Y_FRAC_RY,
    HeadBounds,
    build_face_parts,
)
from meshops.proportion.models import (
    CrossSection,
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import build_blockout_skeleton

_MID_R = 0.0613
_CALF_HW = 0.04379
_THIN_HW_M = 0.0263
_HIER_PREFIX = "head face hierarchy:"


def _lm(
    id_: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=id_, x_m=x_m, y_m=y_m, z_m=z_m)


def _diam(band_id: str, *, half_width_m: float | None = 0.05) -> DiameterMeasure:
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
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=head_unit_frac,
        landmarks_xyz=lms,
        diameters=[
            _diam("bust", half_width_m=0.16),
            _diam("waist", half_width_m=0.13),
            _diam("neck", half_width_m=0.05),
            _diam("upper_arm_l", half_width_m=0.05),
            _diam("upper_arm_r", half_width_m=0.05),
            _diam("thigh_l", half_width_m=0.05),
            _diam("thigh_r", half_width_m=0.05),
            _diam("head", half_width_m=0.08),
        ],
        depth_bands=[
            _depth_band("chest", depth_m=0.24, z_frac=0.72),
            _depth_band("hip", depth_m=0.26, z_frac=0.55),
            _depth_band("cranial", depth_m=0.20, z_frac=0.92, y_mid=-0.01),
        ],
        quality=QualityFlags(),
    )


def _product_class_report(*, height_m: float = 1.72) -> ProportionReport:
    h = height_m
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=1.72),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.38),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=None, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=None, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=None, z_m=0.90),
        "hand_l": _lm("hand_l", x_m=-0.33, y_m=None, z_m=0.85),
        "hand_r": _lm("hand_r", x_m=0.33, y_m=None, z_m=0.85),
        "fingertip_l": _lm("fingertip_l", x_m=-0.36, y_m=None, z_m=0.72),
        "fingertip_r": _lm("fingertip_r", x_m=0.36, y_m=None, z_m=0.72),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.1303, z_m=1.25),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
        "heel_l": _lm("heel_l", x_m=-0.10, y_m=0.06, z_m=0.02),
        "heel_r": _lm("heel_r", x_m=0.10, y_m=0.06, z_m=0.02),
        "toe_l": _lm("toe_l", x_m=-0.10, y_m=-0.12, z_m=0.02),
        "toe_r": _lm("toe_r", x_m=0.10, y_m=-0.12, z_m=0.02),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.0438),
        _diam("upper_arm_r", half_width_m=0.0438),
        _diam("forearm_l", half_width_m=0.0350),
        _diam("forearm_r", half_width_m=0.0350),
        _diam("thigh_l", half_width_m=_MID_R),
        _diam("thigh_r", half_width_m=_MID_R),
        _diam("calf_l", half_width_m=_CALF_HW),
        _diam("calf_r", half_width_m=_CALF_HW),
        _diam("ank_foot_l", half_width_m=_THIN_HW_M),
        _diam("ank_foot_r", half_width_m=_THIN_HW_M),
    ]
    bands = [
        _depth_band("chest", depth_m=0.2606, z_frac=0.72),
        _depth_band("breast", depth_m=0.18),
        _depth_band("hip", depth_m=0.26),
        _depth_band("glute", depth_m=0.22),
    ]
    return ProportionReport(
        schema_version="1.2.0",
        height_m=h,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        depth_bands=bands,
        diameters=diams,
        cross_sections=[
            CrossSection(
                level_id="bust",
                z_frac=0.72,
                rx_frac=0.10,
                ry_frac=0.08,
                sources=["test"],
            ),
        ],
        quality=QualityFlags(),
    )


def _product_flags(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "limbs": True,
        "torso": "ovals",
        "glute": "two_spheres",
        "nofuse": True,
        "face": True,
        "hair": "short",
        "hands": True,
        "feet": True,
        "fingers": "full",
        "toes": "full",
        "profile": load_anatomy_profile("torso_limb_f_athletic_v1"),
    }
    base.update(overrides)
    return base


def _template(*, taper: float = 0.22, thigh_tilt_deg: float = 10.0) -> TemplateAppliedPackage:
    constants = AppliedConstants(
        breast_mode="dual_tilted",
        glute_mode_default="two_spheres",
        torso_mode_default="ovals",
        torso_waist_taper=taper,
        thigh_tilt_deg=thigh_tilt_deg,
    )
    return TemplateAppliedPackage(
        template_id="female_adult_athletic",
        sex="female",  # type: ignore[arg-type]
        archetype="adult_athletic",
        source_report="mem",
        height_m=1.72,
        constants=constants,
    )


def _product_pkg(**flag_overrides: object):
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    return build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(**flag_overrides),  # type: ignore[arg-type]
    )


def _product_class_bounds() -> HeadBounds:
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


def _head_h(head: RecipePart) -> float:
    assert head.rz_m is not None
    return 2.0 * float(head.rz_m)


def _neck_len(neck: RecipePart) -> float:
    assert neck.p0 is not None and neck.p1 is not None
    return math.dist(neck.p0, neck.p1)


def _neck_tilt_deg(neck: RecipePart) -> float:
    assert neck.p0 is not None and neck.p1 is not None
    dy = float(neck.p1[1]) - float(neck.p0[1])
    dz = float(neck.p1[2]) - float(neck.p0[2])
    return math.degrees(math.atan2(-dy, dz))


def _ellip(
    name: str,
    role: str,
    center: list[float],
    *,
    rx: float = 0.02,
    ry: float = 0.02,
    rz: float = 0.02,
) -> RecipePart:
    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind="ellipsoid",
        center=list(center),
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
        placement="full3d",
        label=name,
    )


def _cyl(name: str, p0: list[float], p1: list[float], *, r: float = 0.03) -> RecipePart:
    return RecipePart(
        name=name,
        role="neck",
        kind="cylinder",
        p0=list(p0),
        p1=list(p1),
        radius_m=r,
        placement="full3d",
        label=name,
    )


def _cap(name: str, role: str, p0: list[float], p1: list[float], *, r: float = 0.01) -> RecipePart:
    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind="capsule",
        p0=list(p0),
        p1=list(p1),
        radius_m=r,
        placement="full3d",
        label=name,
    )


def test_t0_const_freezes() -> None:
    """T0: B1-B4 lifts; hold 0058/0078/0050 fences; invert leftover 0.08/0.45/0.20."""
    assert EYE_RADIUS_FRAC_H == 0.11
    assert EYE_RZ_FRAC_R == 0.58
    assert EYE_RX_FRAC_R == 1.00
    assert EYE_RY_FRAC_R == 0.95
    assert face_recipe_mod._LIP_Z_FRAC == 0.24
    assert HEAD_PITCH_DEG == 6.0
    assert FEATURE_FACE_Y_FRAC_RY == 0.90
    assert JAW_RY_FRAC_HEAD_RY == 0.42
    assert JAW_RZ_FRAC_H == 0.13
    assert JAW_Z_CENTER_FRAC_H == 0.13
    assert NECK_FORWARD_TILT_DEG == 12.0
    assert EYE_RADIUS_FRAC_H != 0.08
    assert EYE_RZ_FRAC_R != 0.45
    assert face_recipe_mod._LIP_Z_FRAC != 0.20
    assert HEAD_PITCH_DEG > 0.0
    assert 0.10 <= EYE_RADIUS_FRAC_H <= 0.12
    assert 0.52 <= EYE_RZ_FRAC_R <= 0.62
    assert 0.22 <= face_recipe_mod._LIP_Z_FRAC <= 0.26
    assert 4.0 <= HEAD_PITCH_DEG <= 8.0
    assert HEAD_PITCH_DEG < 12.0


def test_t1_product_class_parts_present() -> None:
    """T1: both eyes + jaw + lip + nose + cheeks + brows; no new names; n=131."""
    pkg = _product_pkg()
    names = {p.name for p in pkg.parts}
    assert "RECIPE_eye_soft_l" in names
    assert "RECIPE_eye_soft_r" in names
    assert "RECIPE_jaw" in names
    assert "RECIPE_lip_soft" in names
    assert "RECIPE_nose_soft" in names
    assert "RECIPE_cheek_soft_l" in names
    assert "RECIPE_cheek_soft_r" in names
    assert "RECIPE_brow_soft_l" in names
    assert "RECIPE_brow_soft_r" in names
    joined = " ".join(names).lower()
    assert "orbital_socket" not in joined
    assert "chin_soft" not in joined
    assert "lid_soft" not in joined
    assert len(pkg.parts) == 131


def test_t2_orbital_scale_and_outer() -> None:
    """T2: eye.rx == 0.11*H; ry/rz >= 1.2; outer X inside head.rx."""
    pkg = _product_pkg()
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    eye = next(p for p in pkg.parts if p.name == "RECIPE_eye_soft_l")
    assert head.rx_m is not None and head.center is not None
    assert eye.rx_m is not None and eye.ry_m is not None and eye.rz_m is not None
    assert eye.center is not None
    h = _head_h(head)
    assert float(eye.rx_m) == pytest.approx(0.11 * h, abs=1e-4)
    assert float(eye.ry_m) / float(eye.rz_m) >= 1.2
    assert abs(float(eye.center[0])) + float(eye.rx_m) < float(head.rx_m)


def test_t3_lip_shelf_vs_chin_and_nose() -> None:
    """T3: lip Z = 0.24*H shelf; above leftover 0.20; below nose_base; jaw flush."""
    bounds = _product_class_bounds()
    parts = build_face_parts(_full_torso_report(), bounds, face=True, messages=[])
    lip = next(p for p in parts if p.name == "RECIPE_lip_soft")
    jaw = next(p for p in parts if p.name == "RECIPE_jaw")
    assert lip.center is not None and jaw.center is not None and jaw.rz_m is not None
    h = bounds.H
    z_chin = bounds.z_chin
    assert float(lip.center[2]) == pytest.approx(z_chin + 0.24 * h, abs=2e-3)
    assert float(lip.center[2]) > z_chin + 0.20 * h
    assert float(lip.center[2]) < z_chin + 0.33 * h
    assert float(jaw.center[2]) - float(jaw.rz_m) == pytest.approx(z_chin, abs=0.002)


def test_t3b_product_lip_shelf_survives_pitch() -> None:
    """T3b: product emit unpitched lip Z still sits on the 0.24 shelf."""
    pkg = _product_pkg()
    lip = next(p for p in pkg.parts if p.name == "RECIPE_lip_soft")
    jaw = next(p for p in pkg.parts if p.name == "RECIPE_jaw")
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    assert lip.center is not None and jaw.center is not None and jaw.rz_m is not None
    assert neck.p1 is not None
    pivot = [float(neck.p1[0]), float(neck.p1[1]), float(neck.p1[2])]
    th = -math.radians(HEAD_PITCH_DEG)
    lip_pre = _rotate_yz_about_x(list(lip.center), pivot, th)
    jaw_pre = _rotate_yz_about_x(list(jaw.center), pivot, th)
    z_chin = float(jaw_pre[2]) - float(jaw.rz_m)
    h = float(jaw.rz_m) / JAW_RZ_FRAC_H
    assert float(lip_pre[2]) == pytest.approx(z_chin + 0.24 * h, abs=2e-3)
    assert float(lip_pre[2]) > z_chin + 0.20 * h
    assert float(lip_pre[2]) < z_chin + 0.33 * h


def test_t4_head_pitched_neck_hold() -> None:
    """T4: head more -Y than neck tip; pitch 6; neck 12 deg + L hold."""
    pkg = _product_pkg()
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    assert head.center is not None and neck.p0 is not None and neck.p1 is not None
    assert float(head.center[1]) < float(neck.p1[1])
    assert abs(HEAD_PITCH_DEG - 6.0) < 1e-9
    assert NECK_FORWARD_TILT_DEG == 12.0
    assert any("neck_forward_tilt_deg=12" in m for m in pkg.messages)
    assert _neck_tilt_deg(neck) == pytest.approx(12.0, abs=0.2)
    assert _neck_len(neck) > 0.0


def test_t5_unit_pitch_zero_and_six(monkeypatch: pytest.MonkeyPatch) -> None:
    """T5: pitch 0 is no-op; pitch 6 rotates head about neck tip +X."""
    pivot = [0.0, -0.0269, 1.507]
    head_c = [0.0, -0.0269, 1.6149]
    dz = head_c[2] - pivot[2]
    parts0 = [
        _cyl("RECIPE_neck", [0.0, 0.0, 1.380], pivot),
        _ellip("RECIPE_head", "head", head_c, rx=0.0883, ry=0.0908, rz=0.1103),
        _ellip("RECIPE_jaw", "jaw", [0.0, -0.0560, 1.5371], rx=0.0653, ry=0.0381, rz=0.0273),
    ]
    msgs0: list[str] = []
    monkeypatch.setattr(blockout_recipe_mod, "HEAD_PITCH_DEG", 0.0)
    _apply_head_pitch(parts0, msgs0)
    assert parts0[1].center is not None
    assert float(parts0[1].center[1]) == pytest.approx(head_c[1], abs=1e-12)

    parts6 = [
        _cyl("RECIPE_neck", [0.0, 0.0, 1.380], pivot),
        _ellip("RECIPE_head", "head", head_c, rx=0.0883, ry=0.0908, rz=0.1103),
        _ellip("RECIPE_jaw", "jaw", [0.0, -0.0560, 1.5371], rx=0.0653, ry=0.0381, rz=0.0273),
    ]
    msgs6: list[str] = []
    monkeypatch.setattr(blockout_recipe_mod, "HEAD_PITCH_DEG", 6.0)
    _apply_head_pitch(parts6, msgs6)
    assert parts6[1].center is not None
    expect_dy = -dz * math.sin(math.radians(6.0))
    assert float(parts6[1].center[1]) - head_c[1] == pytest.approx(expect_dy, abs=1e-6)


def test_t6_fence_0078_0058_0050() -> None:
    """T6: 0078 jaw / 0058 features / 0050 12 deg hold."""
    assert JAW_RY_FRAC_HEAD_RY == 0.42
    assert JAW_RZ_FRAC_H == 0.13
    assert JAW_Z_CENTER_FRAC_H == 0.13
    assert JAW_RX_FRAC_HEAD_RX == 0.74
    assert JAW_Y_BIAS_FRAC_RY == 0.08
    assert JAW_X_BULGE_ALLOW_M == 0.006
    assert FEATURE_FACE_Y_FRAC_RY == 0.90
    assert NOSE_RX_FRAC_H == 0.045
    assert NOSE_RY_FRAC_H == 0.055
    assert NOSE_RZ_FRAC_H == 0.040
    assert NOSE_TIP_Y_FRAC_RY == 0.98
    assert CHEEK_RX_FRAC_HEAD_RX == 0.28
    assert BROW_R_FRAC_H == 0.028
    assert NECK_FORWARD_TILT_DEG == 12.0


def test_t7_sibling_message_once_const_driven() -> None:
    """T7: exactly one const-driven head face hierarchy sibling after eye/lip."""
    pkg = _product_pkg()
    hits = [m for m in pkg.messages if _HIER_PREFIX in m]
    assert len(hits) == 1
    line = hits[0]
    assert f"r={EYE_RADIUS_FRAC_H}" in line
    assert f"rz={EYE_RZ_FRAC_R}" in line
    assert f"lip_z={face_recipe_mod._LIP_Z_FRAC}" in line
    assert f"pitch={HEAD_PITCH_DEG}" in line
    eye_i = next(i for i, m in enumerate(pkg.messages) if "eye soft axes" in m)
    lip_i = next(i for i, m in enumerate(pkg.messages) if "lip soft axes" in m)
    hier_i = next(i for i, m in enumerate(pkg.messages) if _HIER_PREFIX in m)
    assert hier_i > eye_i
    assert hier_i > lip_i


def test_t8_n_parts_schema_mcp() -> None:
    """T8: n_parts 131; schema 1.4.0; MCP 46."""
    pkg = _product_pkg()
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert pkg.schema_version == "1.4.0"
    assert len(TOOL_NAMES) == 47


def test_t9_jaw_bulge_and_chin_flush() -> None:
    """T9: jaw bulge <= 0.006; chin flush on unpitched face kit."""
    bounds = _product_class_bounds()
    msgs: list[str] = []
    parts = build_face_parts(_full_torso_report(), bounds, face=True, messages=msgs)
    jaw = next(p for p in parts if p.name == "RECIPE_jaw")
    assert jaw.center is not None and jaw.rz_m is not None
    assert float(jaw.center[2]) - float(jaw.rz_m) == pytest.approx(bounds.z_chin, abs=0.002)
    bulge_line = next(m for m in msgs if "jaw_vs_head_x_bulge_m=" in m)
    bulge = float(bulge_line.split("jaw_vs_head_x_bulge_m=", 1)[1].split()[0])
    assert bulge <= 0.006 + 1e-9


def test_t10_lr_equalize_and_exports() -> None:
    """T10: L/R eyes equalize; public exports include radius + pitch."""
    pkg = _product_pkg()
    left = next(p for p in pkg.parts if p.name == "RECIPE_eye_soft_l")
    right = next(p for p in pkg.parts if p.name == "RECIPE_eye_soft_r")
    assert left.rx_m == right.rx_m
    assert left.ry_m == right.ry_m
    assert left.rz_m == right.rz_m
    assert left.center is not None and right.center is not None
    assert abs(float(left.center[0])) == pytest.approx(abs(float(right.center[0])), abs=1e-9)
    assert "EYE_RADIUS_FRAC_H" in face_recipe_mod.__all__
    assert "HEAD_PITCH_DEG" in blockout_recipe_mod.__all__


def test_t11_compact_still_culls_eye() -> None:
    """T11: compact still culls eye_soft (0082 fence); jaw stays."""
    pkg = _product_pkg(soft_density="compact")
    names = {p.name for p in pkg.parts}
    assert not any("eye_soft" in n for n in names)
    assert "RECIPE_jaw" in names


def test_t12_pitch_tokens_fuse_p1_scm_p1() -> None:
    """T12: head/face rotate; neck + base skip; SCM p1 only; fuse p1 only."""
    pivot = [0.0, -0.0269, 1.507]
    head_c = [0.0, -0.0269, 1.6149]
    jaw_c = [0.0, -0.0560, 1.5371]
    eye_c = [-0.0462, -0.1087, 1.6149]
    neck_p0 = [0.0, 0.0, 1.380]
    base_c = [0.0, 0.0, 1.370]
    scm_p0 = [-0.01, 0.0, 1.380]
    scm_p1 = [-0.04, -0.03, 1.560]
    fuse_p0 = [0.0, -0.0269, 1.507]
    fuse_p1 = [0.0, -0.0269, 1.5046]
    parts = [
        _cyl("RECIPE_neck", neck_p0, pivot),
        _ellip("RECIPE_head", "head", head_c, rx=0.0883, ry=0.0908, rz=0.1103),
        _ellip("RECIPE_jaw", "jaw", jaw_c, rx=0.0653, ry=0.0381, rz=0.0273),
        _ellip("RECIPE_eye_soft_l", "eye_soft", eye_c),
        _ellip("RECIPE_brow_soft_l", "brow_soft", [-0.0462, -0.1087, 1.650]),
        _ellip("RECIPE_nose_soft", "nose_soft", [0.0, -0.10, 1.577]),
        _ellip("RECIPE_lip_soft", "lip_soft", [0.0, -0.1087, 1.560]),
        _ellip("RECIPE_cheek_soft_l", "cheek_soft", [-0.048, -0.09, 1.57]),
        _ellip("RECIPE_ear_soft_l", "ear_soft", [-0.088, -0.03, 1.58]),
        _ellip("RECIPE_hair_mass_short", "hair_mass", [0.0, 0.02, 1.68]),
        _ellip("RECIPE_neck_base_soft", "neck", base_c, rx=0.04, ry=0.03, rz=0.02),
        _cap("RECIPE_sternomastoid_soft_l", "sternomastoid_soft", scm_p0, scm_p1),
        _cap("RECIPE_neck_head_fuse", "neck", fuse_p0, fuse_p1),
    ]
    _apply_head_pitch(parts, [])
    by = {p.name: p for p in parts}
    assert by["RECIPE_neck"].p0 == neck_p0
    assert by["RECIPE_neck"].p1 == pivot
    assert by["RECIPE_neck_base_soft"].center == base_c
    assert by["RECIPE_head"].center is not None
    assert float(by["RECIPE_head"].center[1]) < head_c[1]
    assert by["RECIPE_jaw"].center is not None
    assert float(by["RECIPE_jaw"].center[1]) < jaw_c[1]
    assert by["RECIPE_eye_soft_l"].center is not None
    assert float(by["RECIPE_eye_soft_l"].center[1]) < eye_c[1]
    scm = by["RECIPE_sternomastoid_soft_l"]
    assert scm.p0 == scm_p0
    assert scm.p1 is not None
    assert scm.p1 != scm_p1
    fuse = by["RECIPE_neck_head_fuse"]
    assert fuse.p0 == fuse_p0
    assert fuse.p1 is not None
    assert fuse.p1 != fuse_p1


def test_t13_noface_still_pitches_head() -> None:
    """T13: no --face still nods RECIPE_head; hierarchy message may fire."""
    pkg = _product_pkg(face=False)
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    assert head.center is not None and neck.p1 is not None
    assert float(head.center[1]) < float(neck.p1[1])
    assert not any(p.name.startswith("RECIPE_eye_soft") for p in pkg.parts)
    hits = [m for m in pkg.messages if _HIER_PREFIX in m]
    assert len(hits) <= 1


def test_t14_b20_b24_no_new_parts_pitch_cap() -> None:
    """T14: HEAD_PITCH_DEG < 12; no chin_soft / orbital part names."""
    assert HEAD_PITCH_DEG < 12.0
    pkg = _product_pkg()
    joined = " ".join(p.name for p in pkg.parts).lower()
    assert "chin_soft" not in joined
    assert "orbital" not in joined
