"""Track 0091 — sit dual breast_soft proudly in front of 0090 chest plate.

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 46 stay. Not hang-Z / tear / tilt / extra pad / 0083 delt.
"""

from __future__ import annotations

import math

import pytest

import meshops.proportion.blockout_recipe as blockout_recipe_mod
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    BREAST_ATHLETIC_RX_MAX_FRAC_H,
    BREAST_ATTACH_Y_SCALE,
    BREAST_HANG_Z_DROP_FRAC_RZ,
    BREAST_HANG_Z_MIN_DROP_FRAC_RZ,
    BREAST_SIT_PROUD_OF_CHEST_FRONT_M,
    BREAST_STERNUM_CLEARANCE_M,
    BREAST_TEAR_RY_FRAC_RX,
    BREAST_TEAR_RZ_FRAC_RX,
    COMPACT_CULL_NAME_EXACT,
    COMPACT_CULL_NAME_PREFIXES,
    COMPACT_CULL_ROLES,
    RECIPE_SCHEMA_VERSION,
    TORSO_CHEST_Y_REAR_BIAS_FRAC_RY,
    TORSO_OVAL_RY_CHEST_FRAC,
    RecipePart,
    _apply_breast_sit_on_chest,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
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
_SIT_PREFIX = "breast sit-on chest:"


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


def _template(
    *,
    taper: float = 0.22,
    thigh_tilt_deg: float = 10.0,
    breast_tilt_x_deg: float = 20.0,
    intermammary_gap_m: float = 0.029,
) -> TemplateAppliedPackage:
    constants = AppliedConstants(
        breast_mode="dual_tilted",
        glute_mode_default="two_spheres",
        torso_mode_default="ovals",
        torso_waist_taper=taper,
        thigh_tilt_deg=thigh_tilt_deg,
        breast_tilt_x_deg=breast_tilt_x_deg,
        intermammary_gap_frac=0.18,
        intermammary_gap_m=intermammary_gap_m,
        breast_y_m=-0.10,
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


def _msg_value(messages: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for m in messages:
        if m.startswith(prefix):
            return m[len(prefix) :]
        if m == key:
            return ""
    return None


def _ellip(
    name: str,
    role: str,
    center: list[float],
    *,
    rx: float = 0.07224,
    ry: float = 0.05635,
    rz: float = 0.07585,
    rot: list[float] | None = None,
) -> RecipePart:
    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind="ellipsoid",
        center=list(center),
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
        rotation_euler_deg=rot,
        placement="full3d",
        label=name,
    )


def _dual_breast_and_chest(
    *,
    breast_cy: float = -0.10033,
    chest_cy: float = 0.04785,
    chest_ry: float = 0.09382,
    rot: list[float] | None = None,
) -> list[RecipePart]:
    return [
        _ellip(
            "RECIPE_breast_soft_l",
            "breast_soft",
            [-0.08674, breast_cy, 1.22814],
            rot=rot,
        ),
        _ellip(
            "RECIPE_breast_soft_r",
            "breast_soft",
            [0.08674, breast_cy, 1.22814],
            rot=rot,
        ),
        _ellip(
            "RECIPE_torso_oval_chest",
            "torso",
            [0.0, chest_cy, 1.29413],
            rx=0.2216,
            ry=chest_ry,
            rz=0.13389,
        ),
        _ellip(
            "RECIPE_deltoid_soft_l",
            "deltoid_soft",
            [-0.2622, 0.0, 1.35845],
            rx=0.0591,
            ry=0.04256,
            rz=0.0461,
        ),
        _ellip(
            "RECIPE_deltoid_soft_r",
            "deltoid_soft",
            [0.2622, 0.0, 1.35845],
            rx=0.0591,
            ry=0.04256,
            rz=0.0461,
        ),
    ]


def _breasts(parts: list[RecipePart]) -> list[RecipePart]:
    return [p for p in parts if p.role == "breast_soft"]


def test_t0_const_freezes() -> None:
    """T0: B1 sit; hold hang / athletic / attach / 0090 chest plate."""
    assert BREAST_SIT_PROUD_OF_CHEST_FRONT_M == 0.016
    assert BREAST_HANG_Z_DROP_FRAC_RZ == 0.55
    assert BREAST_HANG_Z_MIN_DROP_FRAC_RZ == 0.40
    assert BREAST_ATHLETIC_RX_MAX_FRAC_H == 0.042
    assert BREAST_TEAR_RY_FRAC_RX == 0.78
    assert BREAST_TEAR_RZ_FRAC_RX == 1.05
    assert BREAST_STERNUM_CLEARANCE_M == 0.010
    assert BREAST_ATTACH_Y_SCALE == 1.0
    assert TORSO_OVAL_RY_CHEST_FRAC == 0.72
    assert TORSO_CHEST_Y_REAR_BIAS_FRAC_RY == 0.51
    assert BREAST_SIT_PROUD_OF_CHEST_FRONT_M > 0.0
    assert BREAST_SIT_PROUD_OF_CHEST_FRONT_M != 0.0
    assert 0.012 <= BREAST_SIT_PROUD_OF_CHEST_FRONT_M <= 0.020
    assert BREAST_SIT_PROUD_OF_CHEST_FRONT_M < 0.028
    assert BREAST_SIT_PROUD_OF_CHEST_FRONT_M > 0.006


def test_t1_product_class_parts_present() -> None:
    """T1: dual breast + chest oval + delts; no new names; n=131."""
    pkg = _product_pkg()
    names = {p.name for p in pkg.parts}
    breasts = _breasts(pkg.parts)
    assert len(breasts) == 2
    assert "RECIPE_torso_oval_chest" in names
    assert any("deltoid_soft" in n for n in names)
    joined = " ".join(names).lower()
    assert "breast_pad" not in joined
    assert "lower_pole_soft" not in joined
    assert len(pkg.parts) == 131


def test_t2_product_sit_law() -> None:
    """T2: rear == chest_front - SIT; cy ~ -0.1184; chest front ~ -0.046."""
    pkg = _product_pkg()
    chest = next(p for p in pkg.parts if p.name == "RECIPE_torso_oval_chest")
    assert chest.center is not None and chest.ry_m is not None
    chest_front = float(chest.center[1]) - float(chest.ry_m)
    assert chest_front == pytest.approx(-0.046, abs=2e-3)
    sit = BREAST_SIT_PROUD_OF_CHEST_FRONT_M
    breasts = _breasts(pkg.parts)
    assert len(breasts) == 2
    for b in breasts:
        assert b.center is not None and b.ry_m is not None
        rear = float(b.center[1]) + float(b.ry_m)
        assert rear == pytest.approx(chest_front - sit, abs=1e-4)
        assert float(b.center[1]) == pytest.approx(-0.1184, abs=2e-3)


def test_t3_sit_does_not_change_xz_axes_rot(monkeypatch: pytest.MonkeyPatch) -> None:
    """T3: sit-on is Y-only vs sit=0 sibling; hang drop still ~0.55*rz."""
    pkg = _product_pkg()
    breasts = _breasts(pkg.parts)
    assert len(breasts) == 2
    monkeypatch.setattr(blockout_recipe_mod, "BREAST_SIT_PROUD_OF_CHEST_FRONT_M", 0.0)
    pkg_off = _product_pkg()
    off = {p.name: p for p in _breasts(pkg_off.parts)}
    assert any(m == "breast_hang_z_applied: true" for m in pkg.messages)
    drop_s = _msg_value(pkg.messages, "breast_hang_z_drop_m")
    assert drop_s is not None
    assert math.isfinite(float(drop_s))
    assert BREAST_HANG_Z_DROP_FRAC_RZ == 0.55
    for b in breasts:
        assert b.center is not None
        assert b.rx_m is not None and b.ry_m is not None and b.rz_m is not None
        other = off[b.name]
        assert other.center is not None
        assert float(b.center[0]) == pytest.approx(float(other.center[0]), abs=1e-9)
        assert float(b.center[2]) == pytest.approx(float(other.center[2]), abs=1e-9)
        assert float(b.rx_m) == pytest.approx(float(other.rx_m or 0.0), abs=1e-9)
        assert float(b.ry_m) == pytest.approx(float(other.ry_m or 0.0), abs=1e-9)
        assert float(b.rz_m) == pytest.approx(float(other.rz_m or 0.0), abs=1e-9)
        assert b.rotation_euler_deg == other.rotation_euler_deg
        assert float(b.rx_m) == pytest.approx(0.07224, abs=2e-3)


def test_t4_tilt_athletic_gap_hold() -> None:
    """T4: tilt 20 applied; 0067 athletic applied; gap ≈0.029."""
    pkg = _product_pkg()
    assert any(m == "breast_tilt_applied: true" for m in pkg.messages)
    assert any("breast_tilt_deg=20" in m for m in pkg.messages)
    assert any(m == "breast_lower_pole_athletic_applied: true" for m in pkg.messages)
    gap_s = _msg_value(pkg.messages, "breast_sternum_gap_m")
    assert gap_s is not None
    assert float(gap_s) == pytest.approx(0.029, abs=2e-3)


def test_t5_unit_sit_zero_and_016(monkeypatch: pytest.MonkeyPatch) -> None:
    """T5: sit 0 is no-op; 0.016 lands rear at chest_front - 0.016; X/Z hold."""
    parts0 = _dual_breast_and_chest()
    pre_y = [float(p.center[1]) for p in _breasts(parts0) if p.center is not None]
    msgs0: list[str] = []
    monkeypatch.setattr(blockout_recipe_mod, "BREAST_SIT_PROUD_OF_CHEST_FRONT_M", 0.0)
    _apply_breast_sit_on_chest(parts0, msgs0)
    for p, y0 in zip(_breasts(parts0), pre_y, strict=True):
        assert p.center is not None
        assert float(p.center[1]) == pytest.approx(y0, abs=1e-12)
        assert float(p.center[0]) != 0.0
        assert float(p.center[2]) == pytest.approx(1.22814, abs=1e-12)

    parts = _dual_breast_and_chest()
    chest = next(p for p in parts if p.name == "RECIPE_torso_oval_chest")
    assert chest.center is not None and chest.ry_m is not None
    chest_front = float(chest.center[1]) - float(chest.ry_m)
    msgs: list[str] = []
    monkeypatch.setattr(blockout_recipe_mod, "BREAST_SIT_PROUD_OF_CHEST_FRONT_M", 0.016)
    _apply_breast_sit_on_chest(parts, msgs)
    breasts = _breasts(parts)
    mean_ry = sum(float(b.ry_m) for b in breasts if b.ry_m is not None) / float(len(breasts))
    target_rear = chest_front - 0.016
    target_cy = target_rear - mean_ry
    for b in breasts:
        assert b.center is not None and b.ry_m is not None
        rear = float(b.center[1]) + float(b.ry_m)
        assert rear == pytest.approx(target_rear, abs=1e-9)
        assert float(b.center[1]) == pytest.approx(target_cy, abs=1e-9)
        assert float(b.center[2]) == pytest.approx(1.22814, abs=1e-12)


def test_t6_fence_hang_athletic_tilt_delt_chest() -> None:
    """T6: 0049 hang / 0067 athletic / 0033 tilt / 0083 delt / 0090 chest front."""
    assert BREAST_HANG_Z_DROP_FRAC_RZ == 0.55
    assert BREAST_HANG_Z_MIN_DROP_FRAC_RZ == 0.40
    assert BREAST_ATHLETIC_RX_MAX_FRAC_H == 0.042
    assert BREAST_TEAR_RY_FRAC_RX == 0.78
    assert BREAST_TEAR_RZ_FRAC_RX == 1.05
    assert BREAST_STERNUM_CLEARANCE_M == 0.010
    pkg = _product_pkg()
    assert any(m == "breast_tilt_applied: true" for m in pkg.messages)
    delts = [p for p in pkg.parts if "deltoid_soft" in p.name]
    assert delts
    for d in delts:
        assert d.center is not None
        assert abs(float(d.center[1])) <= 0.02
    chest = next(p for p in pkg.parts if p.name == "RECIPE_torso_oval_chest")
    assert chest.center is not None and chest.ry_m is not None
    chest_front = float(chest.center[1]) - float(chest.ry_m)
    assert chest_front == pytest.approx(-0.046, abs=2e-3)


def test_t7_sibling_message_once_const_driven() -> None:
    """T7: exactly one const-driven sit-on sibling after 0067, before 0049 hang."""
    pkg = _product_pkg()
    hits = [m for m in pkg.messages if _SIT_PREFIX in m]
    assert len(hits) == 1
    assert f"proud={BREAST_SIT_PROUD_OF_CHEST_FRONT_M}" in hits[0]
    athletic_i = next(
        i for i, m in enumerate(pkg.messages) if m == "breast_lower_pole_athletic_applied: true"
    )
    hang_i = next(i for i, m in enumerate(pkg.messages) if m.startswith("breast_hang_z_drop_m="))
    sit_i = next(i for i, m in enumerate(pkg.messages) if _SIT_PREFIX in m)
    assert sit_i > athletic_i
    assert sit_i < hang_i
    assert any(m == "breast_sit_on_chest_applied: true" for m in pkg.messages)


def test_t8_n_parts_schema_mcp() -> None:
    """T8: n_parts 131; schema 1.4.0; MCP 46."""
    pkg = _product_pkg()
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert pkg.schema_version == "1.4.0"
    assert len(TOOL_NAMES) == 46


def test_t9_dual_y_equal_neighbors_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """T9: L/R same Y; chest / delt / clavicle Y hold vs sit=0 sibling."""
    pkg_on = _product_pkg()
    breasts = _breasts(pkg_on.parts)
    ys = [float(b.center[1]) for b in breasts if b.center is not None]
    assert len(ys) == 2
    assert ys[0] == pytest.approx(ys[1], abs=1e-12)

    def _y(pkg, name_token: str) -> list[float]:
        out: list[float] = []
        for p in pkg.parts:
            if name_token not in p.name:
                continue
            if p.center is not None:
                out.append(float(p.center[1]))
            elif p.p0 is not None:
                out.append(float(p.p0[1]))
        return out

    chest_on = _y(pkg_on, "torso_oval_chest")
    delt_on = _y(pkg_on, "deltoid_soft")
    clav_on = _y(pkg_on, "clavicle")
    monkeypatch.setattr(blockout_recipe_mod, "BREAST_SIT_PROUD_OF_CHEST_FRONT_M", 0.0)
    pkg_off = _product_pkg()
    assert _y(pkg_off, "torso_oval_chest") == pytest.approx(chest_on, abs=1e-12)
    assert _y(pkg_off, "deltoid_soft") == pytest.approx(delt_on, abs=1e-12)
    assert _y(pkg_off, "clavicle") == pytest.approx(clav_on, abs=1e-12)


def test_t10_const_is_module_attribute_not_in_all() -> None:
    """T10: import as module attribute (0067 pattern); do not add a lone BREAST_* to __all__."""
    assert hasattr(blockout_recipe_mod, "BREAST_SIT_PROUD_OF_CHEST_FRONT_M")
    assert "BREAST_SIT_PROUD_OF_CHEST_FRONT_M" not in blockout_recipe_mod.__all__
    breast_exports = [n for n in blockout_recipe_mod.__all__ if n.startswith("BREAST_")]
    assert breast_exports == []


def test_t11_compact_still_sits() -> None:
    """T11: compact still applies sit-on; cull set unchanged; n_parts 94 (hair) / 93 helper."""
    assert "breast_soft" not in COMPACT_CULL_ROLES
    assert "RECIPE_breast_soft_l" not in COMPACT_CULL_NAME_EXACT
    assert not any(p.startswith("RECIPE_breast") for p in COMPACT_CULL_NAME_PREFIXES)
    pkg = _product_pkg(soft_density="compact")
    breasts = _breasts(pkg.parts)
    assert len(breasts) == 2
    chest = next(p for p in pkg.parts if p.name == "RECIPE_torso_oval_chest")
    assert chest.center is not None and chest.ry_m is not None
    chest_front = float(chest.center[1]) - float(chest.ry_m)
    for b in breasts:
        assert b.center is not None and b.ry_m is not None
        rear = float(b.center[1]) + float(b.ry_m)
        assert rear == pytest.approx(chest_front - BREAST_SIT_PROUD_OF_CHEST_FRONT_M, abs=1e-4)
    # 0085-class + hair=short compact is 94; 0082 helper (no hair) stays 93.
    assert len(pkg.parts) == 94


def test_t12_no_chest_oval_quiet_skip() -> None:
    """T12: missing RECIPE_torso_oval_chest → applied false; breast Y unchanged."""
    parts = _dual_breast_and_chest()
    parts = [p for p in parts if p.name != "RECIPE_torso_oval_chest"]
    pre = [(list(p.center) if p.center else None) for p in _breasts(parts)]
    msgs: list[str] = []
    _apply_breast_sit_on_chest(parts, msgs)
    assert any(m == "breast_sit_on_chest_applied: false" for m in msgs)
    assert not any(_SIT_PREFIX in m for m in msgs)
    for p, c0 in zip(_breasts(parts), pre, strict=True):
        assert p.center == c0


def test_t13_male_pec_skip() -> None:
    """T13: male pec — sit-on applied false; no Y thrash; hang applied false."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(profile=load_anatomy_profile("torso_limb_m_athletic_v1")),  # type: ignore[arg-type]
    )
    assert not any(p.role == "breast_soft" for p in pkg.parts)
    assert any(p.role == "pec_soft" for p in pkg.parts)
    pecs = [p for p in pkg.parts if p.role == "pec_soft"]
    pec_ys = [float(p.center[1]) for p in pecs if p.center is not None]
    assert pec_ys
    assert any(m == "breast_sit_on_chest_applied: false" for m in pkg.messages)
    assert any(m == "breast_hang_z_applied: false" for m in pkg.messages)
    assert not any(_SIT_PREFIX in m for m in pkg.messages)


def test_t14_b20_b24_band_and_no_extra_parts() -> None:
    """T14: sit band; no breast_pad / lower_pole_soft names."""
    assert BREAST_SIT_PROUD_OF_CHEST_FRONT_M < 0.028
    assert BREAST_SIT_PROUD_OF_CHEST_FRONT_M > 0.006
    pkg = _product_pkg()
    joined = " ".join(p.name for p in pkg.parts).lower()
    assert "breast_pad" not in joined
    assert "lower_pole_soft" not in joined
    assert not any("breast_pad" in (p.role or "") for p in pkg.parts)
    assert not any("lower_pole_soft" in (p.role or "") for p in pkg.parts)
