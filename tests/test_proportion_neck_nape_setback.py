"""Track 0086 — neck nape setback (+Y after 0085 pitch).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 46 stay. Not giraffe / nape_soft / 0050 12° / 0085 6° / 0091 hang.
"""

from __future__ import annotations

import math

import pytest

import meshops.proportion.blockout_recipe as blockout_recipe_mod
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    HEAD_PITCH_DEG,
    NECK_BASE_RX_FRAC_R,
    NECK_BASE_RY_FRAC_R,
    NECK_BASE_RZ_FRAC_R,
    NECK_FORWARD_TILT_DEG,
    NECK_NAPE_CLEARANCE_M,
    NECK_NAPE_SETBACK_M,
    NECK_R_MAX_FRAC_HEAD_RX,
    RECIPE_SCHEMA_VERSION,
    TRAP_NAPE_Z_BIAS_FRAC_H,
    RecipePart,
    _apply_neck_nape_setback,
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
_SETBACK_PREFIX = "neck nape setback:"


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
    """T0: B1 setback; hold 0050/0085/0059/0019/0061; SETBACK != CLEARANCE."""
    assert NECK_NAPE_SETBACK_M == 0.018
    assert NECK_FORWARD_TILT_DEG == 12.0
    assert HEAD_PITCH_DEG == 6.0
    assert NECK_R_MAX_FRAC_HEAD_RX == 0.40
    assert blockout_recipe_mod._GIRAFFE_FRAC == 0.20
    assert NECK_NAPE_CLEARANCE_M == 0.005
    assert NECK_NAPE_SETBACK_M != NECK_NAPE_CLEARANCE_M
    assert NECK_NAPE_SETBACK_M > 0.0
    assert NECK_NAPE_SETBACK_M != 0.0
    assert 0.014 <= NECK_NAPE_SETBACK_M <= 0.022
    assert NECK_NAPE_SETBACK_M < 0.028


def test_t1_product_class_parts_present() -> None:
    """T1: neck + head + base + SCM + traps + delts; no new names; n=131."""
    pkg = _product_pkg()
    names = {p.name for p in pkg.parts}
    assert "RECIPE_neck" in names
    assert "RECIPE_head" in names
    assert "RECIPE_neck_base_soft" in names
    assert "RECIPE_sternomastoid_soft_l" in names
    assert "RECIPE_sternomastoid_soft_r" in names
    assert any("trap_soft" in n for n in names)
    assert any("deltoid_soft" in n for n in names)
    joined = " ".join(names).lower()
    assert "nape_soft" not in joined
    assert "cervical" not in joined
    assert "c7_" not in joined
    assert len(pkg.parts) == 131


def test_t2_product_p0_y_and_length_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """T2: p0 y == 0.018; p1 still -Y of p0; L holds vs setback=0 sibling."""
    pkg_on = _product_pkg()
    neck_on = next(p for p in pkg_on.parts if p.name == "RECIPE_neck")
    assert neck_on.p0 is not None and neck_on.p1 is not None
    assert float(neck_on.p0[1]) == pytest.approx(0.018, abs=1e-4)
    assert float(neck_on.p1[1]) < float(neck_on.p0[1])
    len_on = _neck_len(neck_on)
    monkeypatch.setattr(blockout_recipe_mod, "NECK_NAPE_SETBACK_M", 0.0)
    pkg_off = _product_pkg()
    neck_off = next(p for p in pkg_off.parts if p.name == "RECIPE_neck")
    assert neck_off.p0 is not None and neck_off.p1 is not None
    assert float(neck_off.p0[1]) == pytest.approx(0.0, abs=1e-4)
    assert _neck_len(neck_off) == pytest.approx(len_on, abs=1e-4)
    dy = float(neck_on.p1[1]) - float(neck_on.p0[1])
    assert dy == pytest.approx(-len_on * math.sin(math.radians(12.0)), abs=1e-4)


def test_t3_pitch_nod_survives_setback(monkeypatch: pytest.MonkeyPatch) -> None:
    """T3: 0085 nod survives — head_y < tip_y; relative Δ matches setback=0."""
    pkg = _product_pkg()
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    assert head.center is not None and neck.p1 is not None
    assert float(head.center[1]) < float(neck.p1[1])
    delta_on = float(neck.p1[1]) - float(head.center[1])
    assert delta_on > 0.005
    monkeypatch.setattr(blockout_recipe_mod, "NECK_NAPE_SETBACK_M", 0.0)
    pkg_off = _product_pkg()
    head_off = next(p for p in pkg_off.parts if p.name == "RECIPE_head")
    neck_off = next(p for p in pkg_off.parts if p.name == "RECIPE_neck")
    assert head_off.center is not None and neck_off.p1 is not None
    delta_off = float(neck_off.p1[1]) - float(head_off.center[1])
    assert delta_on == pytest.approx(delta_off, abs=1e-6)


def test_t4_tilt_pitch_radius_hold() -> None:
    """T4: tilt 12; pitch 6; r ≈ 0.03531; L not giraffe."""
    pkg = _product_pkg()
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    assert neck.radius_m is not None
    assert any("neck_forward_tilt_deg=12" in m for m in pkg.messages)
    assert any("pitch=6.0" in m for m in pkg.messages)
    assert _neck_tilt_deg(neck) == pytest.approx(12.0, abs=0.2)
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    assert head.rx_m is not None
    assert float(neck.radius_m) == pytest.approx(
        NECK_R_MAX_FRAC_HEAD_RX * float(head.rx_m), abs=2e-4
    )
    assert float(neck.radius_m) == pytest.approx(0.03531, abs=2e-3)
    assert _neck_len(neck) == pytest.approx(0.130, abs=0.01)
    assert blockout_recipe_mod._GIRAFFE_FRAC == 0.20


def test_t5_unit_setback_zero_and_018(monkeypatch: pytest.MonkeyPatch) -> None:
    """T5: setback 0 is no-op; 0.018 translates neck p0/p1 + head Y only."""
    neck_p0 = [0.0, 0.0, 1.380]
    neck_p1 = [0.0, -0.0269, 1.507]
    head_c = [0.0, -0.0382, 1.614]
    parts0 = [
        _cyl("RECIPE_neck", neck_p0, neck_p1),
        _ellip("RECIPE_head", "head", head_c, rx=0.0883, ry=0.0908, rz=0.1103),
    ]
    msgs0: list[str] = []
    monkeypatch.setattr(blockout_recipe_mod, "NECK_NAPE_SETBACK_M", 0.0)
    _apply_neck_nape_setback(parts0, msgs0)
    assert parts0[0].p0 is not None and parts0[0].p1 is not None
    assert parts0[1].center is not None
    assert float(parts0[0].p0[1]) == pytest.approx(0.0, abs=1e-12)
    assert float(parts0[0].p1[1]) == pytest.approx(-0.0269, abs=1e-12)
    assert float(parts0[1].center[1]) == pytest.approx(-0.0382, abs=1e-12)
    assert float(parts0[0].p0[2]) == pytest.approx(1.380, abs=1e-12)
    assert float(parts0[1].center[0]) == pytest.approx(0.0, abs=1e-12)

    parts = [
        _cyl("RECIPE_neck", neck_p0, neck_p1),
        _ellip("RECIPE_head", "head", head_c, rx=0.0883, ry=0.0908, rz=0.1103),
    ]
    msgs: list[str] = []
    monkeypatch.setattr(blockout_recipe_mod, "NECK_NAPE_SETBACK_M", 0.018)
    _apply_neck_nape_setback(parts, msgs)
    assert parts[0].p0 is not None and parts[0].p1 is not None
    assert parts[1].center is not None
    assert float(parts[0].p0[1]) == pytest.approx(0.018, abs=1e-9)
    assert float(parts[0].p1[1]) == pytest.approx(-0.0089, abs=1e-9)
    assert float(parts[1].center[1]) == pytest.approx(-0.0202, abs=1e-9)
    assert float(parts[0].p0[2]) == pytest.approx(1.380, abs=1e-12)
    assert float(parts[0].p1[2]) == pytest.approx(1.507, abs=1e-12)
    assert float(parts[1].center[2]) == pytest.approx(1.614, abs=1e-12)
    assert float(parts[1].center[0]) == pytest.approx(0.0, abs=1e-12)


def test_t6_fence_0050_0085_0059_0061_0083() -> None:
    """T6: 0050 12° / 0085 6° / 0059 r+base / 0061 nape Z / 0083 delt plane."""
    assert NECK_FORWARD_TILT_DEG == 12.0
    assert HEAD_PITCH_DEG == 6.0
    assert NECK_R_MAX_FRAC_HEAD_RX == 0.40
    assert NECK_BASE_RX_FRAC_R == 1.25
    assert NECK_BASE_RY_FRAC_R == 0.90
    assert NECK_BASE_RZ_FRAC_R == 0.55
    assert TRAP_NAPE_Z_BIAS_FRAC_H == 0.010
    pkg = _product_pkg()
    delts = [p for p in pkg.parts if "deltoid_soft" in p.name]
    assert delts
    for d in delts:
        assert d.center is not None
        assert abs(float(d.center[1])) <= 0.02


def test_t7_sibling_message_once_const_driven() -> None:
    """T7: exactly one const-driven neck nape setback sibling after tilt/pitch."""
    pkg = _product_pkg()
    hits = [m for m in pkg.messages if _SETBACK_PREFIX in m]
    assert len(hits) == 1
    assert f"dy={NECK_NAPE_SETBACK_M}" in hits[0]
    tilt_i = next(i for i, m in enumerate(pkg.messages) if "neck_forward_tilt_deg=" in m)
    pitch_i = next(i for i, m in enumerate(pkg.messages) if "head face hierarchy:" in m)
    set_i = next(i for i, m in enumerate(pkg.messages) if _SETBACK_PREFIX in m)
    assert set_i > tilt_i
    assert set_i > pitch_i


def test_t8_n_parts_schema_mcp() -> None:
    """T8: n_parts 131; schema 1.4.0; MCP 46."""
    pkg = _product_pkg()
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert pkg.schema_version == "1.4.0"
    assert len(TOOL_NAMES) == 46


def test_t9_base_and_scm_follow_p0_neckline_stays() -> None:
    """T9: base cy == p0 y; SCM both ends at p0 y; neckline stays near 0."""
    pkg = _product_pkg()
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    base = next(p for p in pkg.parts if p.name == "RECIPE_neck_base_soft")
    assert neck.p0 is not None and base.center is not None
    p0_y = float(neck.p0[1])
    assert float(base.center[1]) == pytest.approx(p0_y, abs=1e-6)
    for name in ("RECIPE_sternomastoid_soft_l", "RECIPE_sternomastoid_soft_r"):
        scm = next(p for p in pkg.parts if p.name == name)
        assert scm.p0 is not None
        assert float(scm.p0[1]) == pytest.approx(p0_y, abs=1e-4)
    necklines = [p for p in pkg.parts if "neckline" in p.name.lower()]
    for nl in necklines:
        cy = nl.center[1] if nl.center is not None else None
        if cy is not None:
            assert abs(float(cy)) <= 0.05


def test_t10_lr_equalize_and_export() -> None:
    """T10: L/R SCM / trap / delt equalize on |x|; __all__ exports setback."""
    pkg = _product_pkg()

    def _pair(token: str) -> tuple[RecipePart, RecipePart]:
        left = next(p for p in pkg.parts if token in p.name and p.name.endswith("_l"))
        right = next(p for p in pkg.parts if token in p.name and p.name.endswith("_r"))
        return left, right

    for token in ("sternomastoid_soft", "trap_soft", "deltoid_soft"):
        left, right = _pair(token)
        lx = left.center[0] if left.center is not None else (left.p0[0] if left.p0 else 0.0)
        rx = right.center[0] if right.center is not None else (right.p0[0] if right.p0 else 0.0)
        assert abs(float(lx)) == pytest.approx(abs(float(rx)), abs=1e-6)
    assert "NECK_NAPE_SETBACK_M" in blockout_recipe_mod.__all__


def test_t11_compact_still_setbacks() -> None:
    """T11: compact still applies setback; n_parts 93 hold; cull set unchanged."""
    pkg = _product_pkg(soft_density="compact")
    neck = next(p for p in pkg.parts if p.name == "RECIPE_neck")
    assert neck.p0 is not None
    assert float(neck.p0[1]) == pytest.approx(NECK_NAPE_SETBACK_M, abs=1e-4)
    # 0085-class + hair=short compact is 94; 0082 helper (no hair) stays 93.
    assert len(pkg.parts) == 94


def test_t12_translate_tokens_scm_both_fuse_full() -> None:
    """T12: head/face + neck p0/p1 + SCM both + fuse full; girdle stays."""
    neck_p0 = [0.0, 0.0, 1.380]
    neck_p1 = [0.0, -0.0269, 1.507]
    head_c = [0.0, -0.0382, 1.614]
    jaw_c = [0.0, -0.0560, 1.537]
    eye_c = [-0.0462, -0.1087, 1.615]
    scm_p0 = [-0.01, 0.0, 1.380]
    scm_p1 = [-0.04, -0.03, 1.560]
    fuse_p0 = [0.0, -0.0269, 1.507]
    fuse_p1 = [0.0, -0.0269, 1.5046]
    delt_c = [0.16, 0.0, 1.38]
    trap_c = [0.14, 0.0172, 1.397]
    scap_c = [0.10, 0.1248, 1.286]
    clav_p0 = [0.02, -0.08, 1.36]
    clav_p1 = [0.16, 0.0, 1.38]
    chest_c = [0.0, 0.0478, 1.29]
    parts = [
        _cyl("RECIPE_neck", neck_p0, neck_p1),
        _ellip("RECIPE_head", "head", head_c, rx=0.0883, ry=0.0908, rz=0.1103),
        _ellip("RECIPE_jaw", "jaw", jaw_c),
        _ellip("RECIPE_eye_soft_l", "eye_soft", eye_c),
        _ellip("RECIPE_brow_soft_l", "brow_soft", [-0.0462, -0.1087, 1.650]),
        _ellip("RECIPE_nose_soft", "nose_soft", [0.0, -0.10, 1.577]),
        _ellip("RECIPE_lip_soft", "lip_soft", [0.0, -0.1087, 1.560]),
        _ellip("RECIPE_cheek_soft_l", "cheek_soft", [-0.048, -0.09, 1.57]),
        _ellip("RECIPE_ear_soft_l", "ear_soft", [-0.088, -0.03, 1.58]),
        _ellip("RECIPE_hair_mass_short", "hair_mass", [0.0, 0.02, 1.68]),
        _cap("RECIPE_sternomastoid_soft_l", "sternomastoid_soft", scm_p0, scm_p1),
        _cap("RECIPE_neck_head_fuse", "neck", fuse_p0, fuse_p1),
        _ellip("RECIPE_deltoid_soft_l", "deltoid_soft", delt_c),
        _ellip("RECIPE_trap_soft_l", "trap_soft", trap_c),
        _ellip("RECIPE_scap_soft_l", "scap_soft", scap_c),
        _cap("RECIPE_clavicle_l", "clavicle", clav_p0, clav_p1),
        _ellip("RECIPE_torso_oval_chest", "torso", chest_c, rx=0.13, ry=0.09, rz=0.13),
    ]
    _apply_neck_nape_setback(parts, [])
    by = {p.name: p for p in parts}
    dy = NECK_NAPE_SETBACK_M
    assert by["RECIPE_neck"].p0 is not None and by["RECIPE_neck"].p1 is not None
    assert float(by["RECIPE_neck"].p0[1]) == pytest.approx(neck_p0[1] + dy, abs=1e-9)
    assert float(by["RECIPE_neck"].p1[1]) == pytest.approx(neck_p1[1] + dy, abs=1e-9)
    assert by["RECIPE_head"].center is not None
    assert float(by["RECIPE_head"].center[1]) == pytest.approx(head_c[1] + dy, abs=1e-9)
    assert by["RECIPE_jaw"].center is not None
    assert float(by["RECIPE_jaw"].center[1]) == pytest.approx(jaw_c[1] + dy, abs=1e-9)
    assert by["RECIPE_eye_soft_l"].center is not None
    assert float(by["RECIPE_eye_soft_l"].center[1]) == pytest.approx(eye_c[1] + dy, abs=1e-9)
    scm = by["RECIPE_sternomastoid_soft_l"]
    assert scm.p0 is not None and scm.p1 is not None
    assert float(scm.p0[1]) == pytest.approx(scm_p0[1] + dy, abs=1e-9)
    assert float(scm.p1[1]) == pytest.approx(scm_p1[1] + dy, abs=1e-9)
    fuse = by["RECIPE_neck_head_fuse"]
    assert fuse.p0 is not None and fuse.p1 is not None
    assert float(fuse.p0[1]) == pytest.approx(fuse_p0[1] + dy, abs=1e-9)
    assert float(fuse.p1[1]) == pytest.approx(fuse_p1[1] + dy, abs=1e-9)
    assert by["RECIPE_deltoid_soft_l"].center == delt_c
    assert by["RECIPE_trap_soft_l"].center == trap_c
    assert by["RECIPE_scap_soft_l"].center == scap_c
    assert by["RECIPE_clavicle_l"].p0 == clav_p0
    assert by["RECIPE_clavicle_l"].p1 == clav_p1
    assert by["RECIPE_torso_oval_chest"].center == chest_c


def test_t13_no_neck_quiet_skip() -> None:
    """T13: no RECIPE_neck → no message / no crash; head Y unchanged."""
    head_c = [0.0, -0.0382, 1.614]
    parts = [_ellip("RECIPE_head", "head", head_c, rx=0.0883, ry=0.0908, rz=0.1103)]
    msgs: list[str] = []
    _apply_neck_nape_setback(parts, msgs)
    assert parts[0].center is not None
    assert float(parts[0].center[1]) == pytest.approx(-0.0382, abs=1e-12)
    assert not any(_SETBACK_PREFIX in m for m in msgs)


def test_t14_b20_b24_no_new_parts_setback_cap() -> None:
    """T14: SETBACK < 0.028 and >0; no nape_soft / cervical names."""
    assert NECK_NAPE_SETBACK_M < 0.028
    assert NECK_NAPE_SETBACK_M > 0.0
    pkg = _product_pkg()
    joined = " ".join(p.name for p in pkg.parts).lower()
    assert "nape_soft" not in joined
    assert "cervical" not in joined
