"""Track 0071 - Knee widen (seam) + calf belly p0-only bias (T0-T12)."""

from __future__ import annotations

from typing import Any

import pytest

from meshops.proportion.blockout_recipe import (
    CALF_BELLY_LAT_FRAC,
    CALF_BELLY_REAR_FRAC,
    CALF_BELLY_SCALE,
    CALF_DIST_END_SCALE,
    CALF_PROX_END_SCALE,
    KNEE_SOFT_FRAC,
    KNEE_SOFT_MIN_FRAC_H,
    KNEE_SOFT_OUTER_FRAC_RX,
    KNEE_SOFT_REAR_FRAC_RY,
    KNEE_SOFT_RY_FRAC,
    KNEE_SOFT_RZ_FRAC,
    THIGH_DIST_SHAFT_SCALE,
    _knee_seam_radius_m,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.constraints import (
    classify_part_name,
    validate_constraints,
)
from meshops.proportion.models import (
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


def _limb_mass_report(
    *,
    height_m: float = 1.72,
    thigh_hw: float = 0.06,
    calf_hw: float = 0.05,
    arm_hw: float = 0.04,
    include_knees: bool = True,
    knee_y: float | None = 0.04,
    include_feet_lms: bool = False,
) -> ProportionReport:
    """Synthetic full-limb report for 0071 knee/calf shape tests."""
    lms: dict[str, LandmarkXYZ] = {
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=1.72),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=0.0, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=0.0, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=0.0, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=0.0, z_m=0.90),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
    }
    if include_knees:
        lms["knee_l"] = _lm("knee_l", x_m=-0.12, y_m=knee_y, z_m=0.50)
        lms["knee_r"] = _lm("knee_r", x_m=0.12, y_m=knee_y, z_m=0.50)
    if include_feet_lms:
        lms["heel_l"] = _lm("heel_l", x_m=-0.10, y_m=0.06, z_m=0.02)
        lms["heel_r"] = _lm("heel_r", x_m=0.10, y_m=0.06, z_m=0.02)
        lms["toe_l"] = _lm("toe_l", x_m=-0.10, y_m=-0.12, z_m=0.02)
        lms["toe_r"] = _lm("toe_r", x_m=0.10, y_m=-0.12, z_m=0.02)
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=arm_hw),
        _diam("upper_arm_r", half_width_m=arm_hw),
        _diam("forearm_l", half_width_m=arm_hw),
        _diam("forearm_r", half_width_m=arm_hw),
        _diam("thigh_l", half_width_m=thigh_hw),
        _diam("thigh_r", half_width_m=thigh_hw),
        _diam("calf_l", half_width_m=calf_hw),
        _diam("calf_r", half_width_m=calf_hw),
    ]
    if include_feet_lms:
        diams.append(_diam("ank_foot_l", half_width_m=0.035))
        diams.append(_diam("ank_foot_r", half_width_m=0.035))
    return ProportionReport(
        schema_version="1.0.0",
        height_m=height_m,
        landmarks_xyz=lms,
        diameters=diams,
        quality=QualityFlags(),
    )


def _template_applied_tilt(*, thigh_tilt_deg: float = 10.0, height_m: float = 1.72) -> object:
    return TemplateAppliedPackage(
        template_id="female_adult_athletic",
        sex="female",
        archetype="adult_athletic",
        source_report="test",
        height_m=height_m,
        constants=AppliedConstants(
            breast_mode="dual_tilted",
            glute_mode_default="oval",
            torso_mode_default="trap",
            thigh_tilt_deg=thigh_tilt_deg,
        ),
    )


def _part(
    name: str,
    *,
    kind: str = "capsule",
    role: str = "limb_segment",
    center: list[float] | None = None,
    rx_m: float | None = None,
    ry_m: float | None = None,
    rz_m: float | None = None,
    radius_m: float | None = None,
    p0: list[float] | None = None,
    p1: list[float] | None = None,
) -> Any:
    from meshops.proportion.blockout_recipe import RecipePart

    kwargs: dict[str, Any] = {
        "name": name,
        "role": role,
        "kind": kind,
        "center": center,
        "rx_m": rx_m,
        "ry_m": ry_m,
        "rz_m": rz_m,
        "radius_m": radius_m,
        "p0": p0,
        "p1": p1,
    }
    clean = {k: v for k, v in kwargs.items() if v is not None or k in ("name", "role", "kind")}
    return RecipePart.model_validate(clean)


# ---------------------------------------------------------------------------
# T0-T12
# ---------------------------------------------------------------------------


def test_t0_const_freezes() -> None:
    """T0: knee seam freezes + calf belly lat/rear + 0045 scale fence."""
    assert KNEE_SOFT_FRAC == 1.10
    assert KNEE_SOFT_MIN_FRAC_H == 0.018
    assert KNEE_SOFT_RY_FRAC == 0.90
    assert KNEE_SOFT_RZ_FRAC == 0.75
    assert KNEE_SOFT_OUTER_FRAC_RX == 0.06
    assert KNEE_SOFT_REAR_FRAC_RY == 0.10
    assert CALF_BELLY_LAT_FRAC == 0.22
    assert CALF_BELLY_REAR_FRAC == 0.28
    assert CALF_BELLY_SCALE == 1.08
    assert CALF_PROX_END_SCALE == 0.88
    assert CALF_DIST_END_SCALE == 0.72


def test_t1_knee_rx_above_calf_a_near_seam() -> None:
    """T1: knee rx > calf_a.rx and rx >= 0.95*seam; not require rx >= thigh prox."""
    height_m = 1.72
    thigh_hw = 0.08
    calf_hw = 0.04
    report = _limb_mass_report(height_m=height_m, thigh_hw=thigh_hw, calf_hw=calf_hw)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        knee = by[f"RECIPE_knee_soft_{side}"]
        thigh = by[f"RECIPE_limb_thigh_{side}"]
        dist = by[f"RECIPE_thigh_taper_dist_{side}"]
        calf_a = by[f"RECIPE_calf_a_{side}"]
        assert knee.rx_m is not None and calf_a.rx_m is not None
        assert dist.radius_m is not None and thigh.radius_m is not None
        seam = max(float(dist.radius_m), float(calf_a.rx_m))
        assert float(knee.rx_m) > float(calf_a.rx_m)
        assert float(knee.rx_m) >= 0.95 * seam - 1e-9
        # Not leg-wide: rx may be < thigh prox
        assert float(thigh.radius_m) > float(calf_a.rx_m)


def test_t2_knee_anisotropy() -> None:
    """T2: rx >= ry >= rz with 0.90 / 0.75 fracs."""
    report = _limb_mass_report(thigh_hw=0.08, calf_hw=0.04)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        knee = by[f"RECIPE_knee_soft_{side}"]
        assert knee.rx_m is not None and knee.ry_m is not None and knee.rz_m is not None
        rx, ry, rz = float(knee.rx_m), float(knee.ry_m), float(knee.rz_m)
        assert rx >= ry - 1e-12
        assert ry >= rz - 1e-12
        assert ry == pytest.approx(rx * KNEE_SOFT_RY_FRAC, abs=1e-9)
        assert rz == pytest.approx(rx * KNEE_SOFT_RZ_FRAC, abs=1e-9)


def test_t3_knee_bias_vs_co_moved_calf_a() -> None:
    """T3: post-adduction outer/rear bias vs co-moved calf_a (not raw landmark)."""
    report = _limb_mass_report(thigh_hw=0.08, calf_hw=0.04)
    tpl = _template_applied_tilt(thigh_tilt_deg=10.0)
    pkg = build_blockout_recipe(report, limbs=True, template_applied=tpl)  # type: ignore[arg-type]
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        knee = by[f"RECIPE_knee_soft_{side}"]
        calf_a = by[f"RECIPE_calf_a_{side}"]
        assert knee.center is not None and calf_a.center is not None
        sign = 1.0 if side == "r" else -1.0
        # Outer: knee further out than co-moved calf_a
        assert sign * (float(knee.center[0]) - float(calf_a.center[0])) > 0
        # Rear: knee +Y vs calf_a
        assert float(knee.center[1]) > float(calf_a.center[1])


def test_t4_knee_message_rx() -> None:
    """T4: message knee_soft_{side}: rx= present."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    for side in ("l", "r"):
        assert any(f"knee_soft_{side}: rx=" in m for m in pkg.messages)
        assert not any(
            m.startswith(f"knee_soft_{side}: r=") and "rx=" not in m for m in pkg.messages
        )


def test_t5_seam_adj_axes_message() -> None:
    """T5: seam adj x 1.10 + axes + message rx= (pin spirit)."""
    height_m = 1.72
    thigh_hw = 0.08
    calf_hw = 0.04
    report = _limb_mass_report(height_m=height_m, thigh_hw=thigh_hw, calf_hw=calf_hw)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        knee = by[f"RECIPE_knee_soft_{side}"]
        dist = by[f"RECIPE_thigh_taper_dist_{side}"]
        calf_a = by[f"RECIPE_calf_a_{side}"]
        seam = max(float(dist.radius_m), float(calf_a.rx_m))  # type: ignore[arg-type]
        base = max(KNEE_SOFT_FRAC * seam, KNEE_SOFT_MIN_FRAC_H * height_m)
        assert knee.rx_m == pytest.approx(base, abs=1e-9)
        assert knee.ry_m == pytest.approx(base * KNEE_SOFT_RY_FRAC, abs=1e-9)
        assert knee.rz_m == pytest.approx(base * KNEE_SOFT_RZ_FRAC, abs=1e-9)
        helper = _knee_seam_radius_m(pkg.parts, side, report)
        assert helper == pytest.approx(seam, abs=1e-9)
        assert float(dist.radius_m) == pytest.approx(  # type: ignore[arg-type]
            thigh_hw * THIGH_DIST_SHAFT_SCALE, abs=1e-9
        )
        assert any(f"knee_soft_{side}: rx=" in m for m in pkg.messages)


def test_t5b_no_knee_skips_knee_soft() -> None:
    """T5b: no knee joint → no knee_soft."""
    report = _limb_mass_report(include_knees=False)
    pkg = build_blockout_recipe(report, limbs=True)
    assert not any("knee_soft" in p.name for p in pkg.parts)


def test_t6_calf_split_names() -> None:
    """T6: calf_a/cyl/b present; no limb_calf."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        assert f"RECIPE_calf_a_{side}" in by
        assert f"RECIPE_calf_cyl_{side}" in by
        assert f"RECIPE_calf_b_{side}" in by
        assert f"RECIPE_limb_calf_{side}" not in by
    assert not any("limb_calf" in p.name.lower() for p in pkg.parts)


def test_t7_calf_p0_belly_bias_post_b6() -> None:
    """T7: post-B6 (+feet): cyl.p0 lat+rear vs calf_a; p1 no rear belly offset."""
    report = _limb_mass_report(include_feet_lms=True, calf_hw=0.05, knee_y=0.04)
    pkg = build_blockout_recipe(report, limbs=True, feet=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        a = by[f"RECIPE_calf_a_{side}"]
        cyl = by[f"RECIPE_calf_cyl_{side}"]
        assert a.center is not None and cyl.p0 is not None and cyl.p1 is not None
        assert cyl.radius_m is not None
        sign = 1.0 if side == "r" else -1.0
        cyl_r = float(cyl.radius_m)
        dx = sign * CALF_BELLY_LAT_FRAC * cyl_r
        dy = CALF_BELLY_REAR_FRAC * cyl_r
        # p0 biased vs a.center (a on joint axis)
        assert float(cyl.p0[0]) == pytest.approx(float(a.center[0]) + dx, abs=1e-6)
        assert float(cyl.p0[1]) == pytest.approx(float(a.center[1]) + dy, abs=1e-6)
        assert float(cyl.p0[2]) == pytest.approx(float(a.center[2]), abs=1e-6)
        # p1: B6 may set Y to ank_foot; no lat/rear belly bias on distal end
        b = by[f"RECIPE_calf_b_{side}"]
        assert b.center is not None
        assert float(cyl.p1[0]) == pytest.approx(float(b.center[0]), abs=1e-6)
        assert float(cyl.p1[1]) == pytest.approx(float(b.center[1]), abs=1e-6)
        # Distal Y is not proximal+rear (top-heavy gastroc only on p0)
        assert abs(float(cyl.p1[1]) - (float(a.center[1]) + dy)) > 1e-3
        assert any(f"calf_{side}: belly bias p0 lat=" in m for m in pkg.messages)


def test_t8_calf_ab_centers_not_biased() -> None:
    """T8: calf_a / calf_b centers stay on joint axis (not cyl belly offset)."""
    report = _limb_mass_report(calf_hw=0.05, knee_y=0.04)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    lms = report.landmarks_xyz
    for side in ("l", "r"):
        a = by[f"RECIPE_calf_a_{side}"]
        b = by[f"RECIPE_calf_b_{side}"]
        cyl = by[f"RECIPE_calf_cyl_{side}"]
        knee = lms[f"knee_{side}"]
        ankle = lms[f"ankle_{side}"]
        assert a.center is not None and b.center is not None
        assert cyl.p0 is not None
        # a near knee joint XZ/Y
        assert float(a.center[0]) == pytest.approx(float(knee.x_m), abs=1e-6)  # type: ignore[arg-type]
        assert float(a.center[2]) == pytest.approx(float(knee.z_m), abs=1e-6)  # type: ignore[arg-type]
        if knee.y_m is not None:
            assert float(a.center[1]) == pytest.approx(float(knee.y_m), abs=1e-6)
        # b near ankle
        assert float(b.center[0]) == pytest.approx(float(ankle.x_m), abs=1e-6)  # type: ignore[arg-type]
        assert float(b.center[2]) == pytest.approx(float(ankle.z_m), abs=1e-6)  # type: ignore[arg-type]
        # a is not co-located with biased p0 in X/Y
        assert abs(float(a.center[0]) - float(cyl.p0[0])) > 1e-4
        assert abs(float(a.center[1]) - float(cyl.p0[1])) > 1e-4


def test_t9_calf_radius_order() -> None:
    """T9: b.rx < cyl.r and a.rx <= cyl.r."""
    report = _limb_mass_report(calf_hw=0.05)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        a = by[f"RECIPE_calf_a_{side}"]
        b = by[f"RECIPE_calf_b_{side}"]
        cyl = by[f"RECIPE_calf_cyl_{side}"]
        assert a.rx_m is not None and b.rx_m is not None and cyl.radius_m is not None
        assert float(b.rx_m) < float(cyl.radius_m)
        assert float(a.rx_m) <= float(cyl.radius_m) + 1e-12


def test_t10_join_ready_preserves_calf_order() -> None:
    """T10: post join_ready, b.rx < cyl.r order preserved."""
    report = _limb_mass_report(include_feet_lms=True, calf_hw=0.05)
    pkg = build_blockout_recipe(report, limbs=True, feet=True, join_ready=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        b = by[f"RECIPE_calf_b_{side}"]
        cyl = by[f"RECIPE_calf_cyl_{side}"]
        assert b.rx_m is not None and cyl.radius_m is not None
        assert float(b.rx_m) < float(cyl.radius_m)


def test_t11_calf_slant_pass() -> None:
    """T11: C_calf_slant pass via validate_constraints on product-like limbs+feet."""
    report = _limb_mass_report(include_feet_lms=True, calf_hw=0.05, knee_y=0.04)
    pkg = build_blockout_recipe(report, limbs=True, feet=True)
    result = validate_constraints(pkg)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_calf_slant"].status == "pass", by_id["C_calf_slant"].message


def test_t12_classifier_and_no_dup() -> None:
    """T12: classify knee_soft → unknown; C_no_dup_limb pass on product package."""
    assert classify_part_name("RECIPE_knee_soft_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_knee_soft_r") == ("unknown", "r")
    report = _limb_mass_report(include_feet_lms=True)
    pkg = build_blockout_recipe(report, limbs=True, feet=True)
    result = validate_constraints(pkg)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_no_dup_limb"].status == "pass", by_id["C_no_dup_limb"].message
