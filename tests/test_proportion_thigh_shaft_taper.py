"""Track 0070 - Thigh shaft prox > distal taper (T0-T14)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from meshops.proportion.blockout_recipe import (
    HIP_SOFT_RX_SCALE,
    THIGH_ADDUCTION_MAX_MEDIAL_M,
    THIGH_DIST_SHAFT_SCALE,
    THIGH_PROX_SHAFT_SCALE,
    THIGH_PROX_SOFT_SCALE,
    THIGH_SPLIT_T,
    _apply_join_ready_overlaps,
    _apply_thigh_adduction,
    _build_thigh_tapered,
    _knee_adj_radius_m,
    build_blockout_recipe,
)
from meshops.proportion.body_template import AppliedConstants, TemplateAppliedPackage
from meshops.proportion.connection_metrics import _hip_pair
from meshops.proportion.constraints import (
    OUTER_X_TOL_M,
    _free_parts,
    _project_hard_constraints,
    _thigh_chain_outer_x,
    classify_part,
    classify_part_name,
    optimize_package,
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
) -> ProportionReport:
    """Synthetic full-limb report matching 0045/0046 limb mass fixture layout."""
    lms = {
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
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
    }
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
# T0-T14
# ---------------------------------------------------------------------------


def test_t0_const_freezes() -> None:
    """T0: shaft taper const freezes exported / importable."""
    assert THIGH_PROX_SHAFT_SCALE == 1.0
    assert THIGH_DIST_SHAFT_SCALE == 0.72
    assert THIGH_SPLIT_T == 0.5
    assert THIGH_DIST_SHAFT_SCALE < THIGH_PROX_SHAFT_SCALE
    # Fence: prox soft + arms-only dist soft still present
    assert THIGH_PROX_SOFT_SCALE == 1.18
    from meshops.proportion.blockout_recipe import _LIMB_DIST_SOFT_BANDS

    assert "thigh_l" not in _LIMB_DIST_SOFT_BANDS
    assert "thigh_r" not in _LIMB_DIST_SOFT_BANDS


def test_t1_emit_both_segments() -> None:
    """T1: both RECIPE_limb_thigh + RECIPE_thigh_taper_dist present (l+r)."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        assert prox.kind == "capsule"
        assert dist.kind == "capsule"
        assert prox.role == "limb_segment"
        assert dist.role == "limb_segment"
        assert prox.radius_m is not None
        assert dist.radius_m is not None
        assert float(prox.radius_m) > float(dist.radius_m)


def test_t2_radii_scales() -> None:
    """T2: prox_r = mid*1.0; dist_r = mid*0.72 on synthetic thigh_hw."""
    mid_r = 0.0613
    report = _limb_mass_report(thigh_hw=mid_r)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        assert float(prox.radius_m) == pytest.approx(  # type: ignore[arg-type]
            mid_r * THIGH_PROX_SHAFT_SCALE, abs=1e-9
        )
        assert float(dist.radius_m) == pytest.approx(  # type: ignore[arg-type]
            mid_r * THIGH_DIST_SHAFT_SCALE, abs=1e-9
        )
        ratio = float(dist.radius_m) / float(prox.radius_m)  # type: ignore[arg-type]
        assert 0.68 <= ratio <= 0.74
        assert ratio == pytest.approx(0.72, abs=1e-9)


def test_t3_classifier() -> None:
    """T3: limb_thigh→thigh; thigh_taper_dist→unknown; prox_soft→unknown."""
    assert classify_part_name("RECIPE_limb_thigh_l") == ("thigh", "l")
    assert classify_part_name("RECIPE_limb_thigh_r") == ("thigh", "r")
    assert classify_part_name("RECIPE_thigh_taper_dist_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_thigh_taper_dist_r") == ("unknown", "r")
    assert classify_part_name("RECIPE_prox_soft_thigh_l") == ("unknown", "l")


def test_t4_no_dist_soft_thigh() -> None:
    """T4: no dist_soft thigh (0045 B13 fence)."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    soft_names = [p.name for p in pkg.parts if "dist_soft" in p.name.lower()]
    assert not any("thigh" in n for n in soft_names)
    by_name = {p.name: p for p in pkg.parts}
    assert "RECIPE_dist_soft_thigh_l" not in by_name
    assert "RECIPE_dist_soft_thigh_r" not in by_name
    # 0069: product path emits hip_soft (not prox_soft)
    assert "RECIPE_hip_soft_l" in by_name
    assert "RECIPE_prox_soft_thigh_l" not in by_name


def test_t5_no_dup_with_both_segments() -> None:
    """T5: both segs + prox_soft + knee + calf → C_no_dup_limb pass."""
    from meshops.proportion.blockout_recipe import BlockoutRecipePackage

    parts = [
        _part(
            "RECIPE_limb_thigh_l",
            radius_m=0.0613,
            p0=[0.12, 0.0, 0.95],
            p1=[0.12, 0.0, 0.725],
        ),
        _part(
            "RECIPE_thigh_taper_dist_l",
            radius_m=0.0490,
            p0=[0.12, 0.0, 0.725],
            p1=[0.12, 0.0, 0.50],
        ),
        _part(
            "RECIPE_prox_soft_thigh_l",
            kind="ellipsoid",
            center=[0.12, 0.0, 0.95],
            rx_m=0.0723,
            ry_m=0.0723,
            rz_m=0.0723,
        ),
        _part(
            "RECIPE_knee_soft_l",
            kind="ellipsoid",
            center=[0.12, 0.0, 0.50],
            rx_m=0.0337,
            ry_m=0.0337,
            rz_m=0.0337,
        ),
        _part(
            "RECIPE_calf_a_l",
            kind="ellipsoid",
            center=[0.12, 0.0, 0.50],
            rx_m=0.0385,
            ry_m=0.0385,
            rz_m=0.0385,
        ),
        _part(
            "RECIPE_calf_cyl_l",
            radius_m=0.0473,
            p0=[0.12, 0.0, 0.50],
            p1=[0.12, 0.04, 0.12],
        ),
        _part(
            "RECIPE_calf_b_l",
            kind="ellipsoid",
            center=[0.12, 0.04, 0.12],
            rx_m=0.0315,
            ry_m=0.0315,
            rz_m=0.0315,
        ),
    ]
    pkg = BlockoutRecipePackage(parts=parts, counts={"parts": len(parts)})
    result = validate_constraints(pkg)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_no_dup_limb"].status == "pass", by_id["C_no_dup_limb"].message


def test_t6_adduction_chain_knee_delta() -> None:
    """T6: medial on taper_dist.p1; co-move Δ from dist.p1; engagement vs dist.p1."""
    report = _limb_mass_report()
    tpl = _template_applied_tilt(thigh_tilt_deg=10.0)
    pkg0 = build_blockout_recipe(report, limbs=True)
    pkg = build_blockout_recipe(report, limbs=True, template_applied=tpl)  # type: ignore[arg-type]
    by0 = {p.name: p for p in pkg0.parts}
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox0 = by0[f"RECIPE_limb_thigh_{side}"]
        prox1 = by[f"RECIPE_limb_thigh_{side}"]
        d0 = by0[f"RECIPE_thigh_taper_dist_{side}"]
        d1 = by[f"RECIPE_thigh_taper_dist_{side}"]
        assert prox0.p0 is not None and d0.p1 is not None
        assert prox1.p0 is not None and d1.p1 is not None
        # Hip fixed
        assert float(prox1.p0[0]) == pytest.approx(float(prox0.p0[0]), abs=1e-9)
        # Chain length preserved
        len0 = math.dist(prox0.p0, d0.p1)
        len1 = math.dist(prox1.p0, d1.p1)
        assert len1 == pytest.approx(len0, abs=1e-6)
        # Medial on chain knee (dist.p1)
        if side == "r":
            assert float(d1.p1[0]) < float(d0.p1[0]) - 1e-6
        else:
            assert float(d1.p1[0]) > float(d0.p1[0]) + 1e-6
        medial = abs(float(d1.p1[0]) - float(d0.p1[0]))
        assert medial <= THIGH_ADDUCTION_MAX_MEDIAL_M + 1e-5
        # limb_thigh.p1 ≈ mid after tilt
        assert prox1.p1 is not None and d1.p0 is not None
        for i in range(3):
            assert float(prox1.p1[i]) == pytest.approx(float(d1.p0[i]), abs=1e-6)
            expected_mid = float(prox1.p0[i]) + THIGH_SPLIT_T * (
                float(d1.p1[i]) - float(prox1.p0[i])
            )
            assert float(prox1.p1[i]) == pytest.approx(expected_mid, abs=1e-6)
        # Engagement vs dist.p1
        knee = by[f"RECIPE_knee_soft_{side}"]
        assert knee.center is not None and knee.rx_m is not None
        assert math.dist(d1.p1, knee.center) <= float(knee.rx_m) + 1e-5
        # Co-move Δ from chain knee (not prox mid)
        delta = [float(d1.p1[i]) - float(d0.p1[i]) for i in range(3)]
        assert abs(delta[0]) > 1e-6
        knee0 = by0[f"RECIPE_knee_soft_{side}"]
        assert knee0.center is not None
        for i in range(3):
            assert float(knee.center[i]) == pytest.approx(
                float(knee0.center[i]) + delta[i], abs=1e-5
            )
        calf_a0 = by0[f"RECIPE_calf_a_{side}"]
        calf_a1 = by[f"RECIPE_calf_a_{side}"]
        assert calf_a0.center is not None and calf_a1.center is not None
        for i in range(3):
            assert float(calf_a1.center[i]) == pytest.approx(
                float(calf_a0.center[i]) + delta[i], abs=1e-5
            )
        cyl0 = by0[f"RECIPE_calf_cyl_{side}"]
        cyl1 = by[f"RECIPE_calf_cyl_{side}"]
        assert cyl0.p0 is not None and cyl1.p0 is not None
        assert cyl0.p1 is not None and cyl1.p1 is not None
        for i in range(3):
            assert float(cyl1.p0[i]) == pytest.approx(float(cyl0.p0[i]) + delta[i], abs=1e-5)
            assert float(cyl1.p1[i]) == pytest.approx(float(cyl0.p1[i]), abs=1e-5)
        calf_b0 = by0[f"RECIPE_calf_b_{side}"]
        calf_b1 = by[f"RECIPE_calf_b_{side}"]
        assert calf_b0.center is not None and calf_b1.center is not None
        for i in range(3):
            assert float(calf_b1.center[i]) == pytest.approx(float(calf_b0.center[i]), abs=1e-5)
        assert any(f"thigh_{side}: adduction_tilt_deg=" in m for m in pkg.messages)


def test_t7_knee_soft_max_path() -> None:
    """T7 (0071/0081): knee_soft scale uses SEAM adj, not full-leg max(prox, dist, calf_a)."""
    from meshops.proportion.blockout_recipe import (
        KNEE_SOFT_FRAC,
        KNEE_SOFT_MAX_VS_THIGH_PROX,
        KNEE_SOFT_MIN_FRAC_H,
        _knee_seam_radius_m,
    )

    height_m = 1.72
    thigh_hw = 0.08
    calf_hw = 0.04
    report = _limb_mass_report(height_m=height_m, thigh_hw=thigh_hw, calf_hw=calf_hw)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        calf_a = by_name[f"RECIPE_calf_a_{side}"]
        # Full-leg fence helper still includes prox
        full_adj = max(
            float(prox.radius_m),  # type: ignore[arg-type]
            float(dist.radius_m),  # type: ignore[arg-type]
            float(calf_a.rx_m),  # type: ignore[arg-type]
        )
        # Seam: prefer taper_dist; max(dist, calf_a) — typically dist when prox=0.08
        seam = max(float(dist.radius_m), float(calf_a.rx_m))  # type: ignore[arg-type]
        assert float(prox.radius_m) == pytest.approx(thigh_hw, abs=1e-9)  # type: ignore[arg-type]
        assert float(dist.radius_m) < float(prox.radius_m)  # type: ignore[arg-type]
        assert full_adj == pytest.approx(float(prox.radius_m), abs=1e-9)  # type: ignore[arg-type]
        assert seam == pytest.approx(float(dist.radius_m), abs=1e-9)  # type: ignore[arg-type]
        expected = max(KNEE_SOFT_FRAC * seam, KNEE_SOFT_MIN_FRAC_H * height_m)
        expected = min(expected, KNEE_SOFT_MAX_VS_THIGH_PROX * float(prox.radius_m))  # type: ignore[arg-type]
        assert knee.rx_m == pytest.approx(expected, abs=1e-9)
        # Helpers: full max for fence; seam for scale path
        helper_adj = _knee_adj_radius_m(pkg.parts, side, report)
        assert helper_adj == pytest.approx(full_adj, abs=1e-9)
        helper_seam = _knee_seam_radius_m(pkg.parts, side, report)
        assert helper_seam == pytest.approx(seam, abs=1e-9)


def test_t8_hip_pair_child_is_limb_thigh() -> None:
    """T8: _hip_pair child is RECIPE_limb_thigh_* when both segs present."""
    parts = [
        _part(
            "RECIPE_thigh_taper_dist_l",
            radius_m=0.049,
            p0=[0.12, 0.0, 0.725],
            p1=[0.12, 0.0, 0.50],
        ),
        _part(
            "RECIPE_limb_thigh_l",
            radius_m=0.0613,
            p0=[0.12, 0.0, 0.95],
            p1=[0.12, 0.0, 0.725],
        ),
        _part(
            "RECIPE_pelvis_oval",
            role="pelvis",
            kind="ellipsoid",
            center=[0.0, 0.0, 0.90],
            rx_m=0.12,
            ry_m=0.08,
            rz_m=0.06,
        ),
    ]
    by_name = {p.name: p for p in parts}
    pair = _hip_pair(parts, by_name, "l")
    assert pair is not None
    child, parent = pair
    assert child.name == "RECIPE_limb_thigh_l"
    assert parent.name == "RECIPE_pelvis_oval"
    assert "thigh_taper" not in child.name


def test_t8b_hip_pair_fallback_excludes_taper() -> None:
    """T8b / B10: fallback loop excludes thigh_taper when primary name missing."""
    # Non-canonical thigh so _find_named(RECIPE_limb_thigh_l) misses.
    thigh = _part(
        "RECIPE_custom_thigh_l",
        radius_m=0.06,
        p0=[0.12, 0.0, 0.95],
        p1=[0.12, 0.0, 0.725],
    )
    decoy = _part(
        "RECIPE_thigh_taper_dist_l",
        radius_m=0.048,
        p0=[0.12, 0.0, 0.725],
        p1=[0.12, 0.0, 0.50],
    )
    pelvis = _part(
        "RECIPE_pelvis_oval",
        role="pelvis",
        kind="ellipsoid",
        center=[0.0, 0.0, 0.90],
        rx_m=0.12,
        ry_m=0.08,
        rz_m=0.06,
    )
    parts = [decoy, thigh, pelvis]
    by_name = {p.name: p for p in parts}
    pair = _hip_pair(parts, by_name, "l")
    assert pair is not None
    child, _parent = pair
    assert child.name == "RECIPE_custom_thigh_l"


def test_t8c_hip_bridge_primary_thigh_still_joined() -> None:
    """T8c / B10: hip_bridge stays primary; limb_thigh still in join rows for B14.

    Codex misread B10 as requiring limb_thigh over hip_bridge. Product still has
    both; resolve_join_connections dual-rows thigh when bridge wins hip_pair.
    """
    from meshops.proportion.connection_metrics import resolve_join_connections

    parts = [
        _part(
            "RECIPE_hip_bridge_l",
            role="hip_bridge",
            kind="ellipsoid",
            center=[-0.16, 0.03, 0.90],
            rx_m=0.03,
            ry_m=0.03,
            rz_m=0.03,
        ),
        _part(
            "RECIPE_limb_thigh_l",
            radius_m=0.0613,
            p0=[-0.22, 0.0, 0.95],
            p1=[-0.15, 0.0, 0.725],
        ),
        _part(
            "RECIPE_thigh_taper_dist_l",
            radius_m=0.049,
            p0=[-0.15, 0.0, 0.725],
            p1=[-0.08, 0.0, 0.50],
        ),
        _part(
            "RECIPE_pelvis_oval",
            role="pelvis",
            kind="ellipsoid",
            center=[0.0, 0.0, 0.90],
            rx_m=0.12,
            ry_m=0.08,
            rz_m=0.06,
        ),
    ]
    by_name = {p.name: p for p in parts}
    pair = _hip_pair(parts, by_name, "l")
    assert pair is not None
    child, _parent = pair
    assert child.name == "RECIPE_hip_bridge_l"
    rows = resolve_join_connections(parts)
    hip_children = {c.name for cid, c, _p, _ax in rows if cid == "hip_l"}
    assert "RECIPE_hip_bridge_l" in hip_children
    assert "RECIPE_limb_thigh_l" in hip_children
    assert "RECIPE_thigh_taper_dist_l" not in hip_children


def test_t9_thigh_outer_binds_on_limb_thigh() -> None:
    """T9: C_thigh_outer uses full hip->knee chain mid when split (not prox mid)."""
    from meshops.proportion.blockout_recipe import BlockoutRecipePackage
    from meshops.proportion.constraints import _outer_x

    # Adducted chain: hip lateral, knee medial — prox-half mid is more lateral.
    hip = [-0.2224, 0.0, 0.90]
    knee = [-0.0821, 0.0, 0.57]
    mid = [
        0.5 * (hip[0] + knee[0]),
        0.5 * (hip[1] + knee[1]),
        0.5 * (hip[2] + knee[2]),
    ]
    r = 0.0613
    # Hip outer tuned so chain-mid outer passes (same geometry intent as product).
    chain_mid_x = 0.5 * (hip[0] + knee[0])
    chain_outer = chain_mid_x - r
    hip_half = 0.03
    hip_cx = chain_outer + hip_half  # left: outer = cx - half
    parts = [
        _part(
            "RECIPE_hip_bridge_l",
            role="hip_bridge",
            kind="ellipsoid",
            center=[hip_cx, 0.03, 0.90],
            rx_m=hip_half,
            ry_m=0.03,
            rz_m=0.03,
        ),
        _part(
            "RECIPE_limb_thigh_l",
            radius_m=r,
            p0=list(hip),
            p1=list(mid),
        ),
        _part(
            "RECIPE_thigh_taper_dist_l",
            radius_m=r * 0.8,
            p0=list(mid),
            p1=list(knee),
        ),
    ]
    pkg = BlockoutRecipePackage(parts=parts, counts={"parts": len(parts)})
    indexed = [(p, *classify_part(p)) for p in parts]
    thigh = parts[1]
    outer_chain = _thigh_chain_outer_x(indexed, thigh, "l")
    outer_prox = _outer_x(thigh, "l")
    assert outer_chain == pytest.approx(chain_outer, abs=1e-9)
    # Prox-half outer is strictly more lateral (more negative on left) than chain.
    assert outer_prox is not None and outer_chain is not None
    assert outer_prox < outer_chain - 1e-6
    result = validate_constraints(pkg)
    by_id = {r.id: r for r in result.rules}
    assert "C_thigh_outer" in by_id
    assert by_id["C_thigh_outer"].status == "pass", by_id["C_thigh_outer"].message
    assert by_id["C_thigh_outer"].metrics is not None
    assert float(by_id["C_thigh_outer"].metrics["delta_l"]) <= OUTER_X_TOL_M + 1e-9
    # Metrics must record chain outer, not prox-half outer.
    assert by_id["C_thigh_outer"].metrics["thigh_outer_x_l"] == pytest.approx(outer_chain, abs=1e-9)


def test_t10_product_width_mid() -> None:
    """T10: product-like mid_r=0.0613 → dist = mid * THIGH_DIST_SHAFT_SCALE."""
    mid_r = 0.0613
    report = _limb_mass_report(thigh_hw=mid_r)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    dist = by_name["RECIPE_thigh_taper_dist_l"]
    assert float(dist.radius_m) == pytest.approx(  # type: ignore[arg-type]
        mid_r * THIGH_DIST_SHAFT_SCALE, abs=1e-5
    )
    prox = by_name["RECIPE_limb_thigh_l"]
    assert float(prox.radius_m) == pytest.approx(0.0613, abs=1e-5)  # type: ignore[arg-type]
    soft = by_name["RECIPE_hip_soft_l"]
    assert float(soft.rx_m) == pytest.approx(  # type: ignore[arg-type]
        mid_r * HIP_SOFT_RX_SCALE, abs=1e-5
    )


def test_t11_message_shaft_taper() -> None:
    """T11: messages contain shaft_taper prox= token."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    assert any("shaft_taper prox=" in m for m in pkg.messages)
    assert any("thigh_l: shaft_taper" in m for m in pkg.messages)
    assert any("thigh_r: shaft_taper" in m for m in pkg.messages)


def test_t12_split_mid_geometry() -> None:
    """T12: mid point coords = lerp(p0,p1,0.5) pre-adduction."""
    p0 = [0.12, 0.04, 0.95]
    p1 = [0.12, 0.04, 0.50]
    messages: list[str] = []
    parts = _build_thigh_tapered(
        side="l",
        p0=p0,
        p1=p1,
        radius=0.06,
        placement="full3d",
        messages=messages,
    )
    prox, dist = parts[0], parts[1]
    assert prox.p0 is not None and prox.p1 is not None
    assert dist.p0 is not None and dist.p1 is not None
    t = THIGH_SPLIT_T
    for i in range(3):
        mid = float(p0[i]) + t * (float(p1[i]) - float(p0[i]))
        assert float(prox.p1[i]) == pytest.approx(mid, abs=1e-12)
        assert float(dist.p0[i]) == pytest.approx(mid, abs=1e-12)
        assert float(prox.p0[i]) == pytest.approx(float(p0[i]), abs=1e-12)
        assert float(dist.p1[i]) == pytest.approx(float(p1[i]), abs=1e-12)
    # Product path pre-adduction
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        assert prox.p0 is not None and prox.p1 is not None
        assert dist.p0 is not None and dist.p1 is not None
        for i in range(3):
            assert float(prox.p1[i]) == pytest.approx(float(dist.p0[i]), abs=1e-9)


def test_t13_chain_length() -> None:
    """T13: ||taper_dist.p1 - limb_thigh.p0|| ~ full hip-knee; preserved under adduction."""
    report = _limb_mass_report()
    pkg0 = build_blockout_recipe(report, limbs=True)
    tpl = _template_applied_tilt(thigh_tilt_deg=10.0)
    pkg = build_blockout_recipe(report, limbs=True, template_applied=tpl)  # type: ignore[arg-type]
    by0 = {p.name: p for p in pkg0.parts}
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox0 = by0[f"RECIPE_limb_thigh_{side}"]
        dist0 = by0[f"RECIPE_thigh_taper_dist_{side}"]
        prox1 = by[f"RECIPE_limb_thigh_{side}"]
        dist1 = by[f"RECIPE_thigh_taper_dist_{side}"]
        assert prox0.p0 is not None and dist0.p1 is not None
        assert prox1.p0 is not None and dist1.p1 is not None
        chain0 = math.dist(prox0.p0, dist0.p1)
        chain1 = math.dist(prox1.p0, dist1.p1)
        # Full hip→knee (report landmarks: hip_z≈0.95, knee_z=0.50 → ~0.45+xy)
        assert chain0 > 0.40
        assert chain1 == pytest.approx(chain0, abs=1e-6)
        # Prox alone is half-ish
        assert math.dist(prox0.p0, prox0.p1) == pytest.approx(  # type: ignore[arg-type]
            chain0 * THIGH_SPLIT_T, abs=1e-5
        )


def test_t14_split_closed_optimize_and_join_ready() -> None:
    """T14: split closed under optimize fast + join-ready hip pull (B13/B14)."""
    from meshops.proportion.blockout_recipe import BlockoutRecipePackage

    # Build product-like package with both segs + hip_bridge + anchors for opt.
    hip = [0.12, 0.0, 0.95]
    mid = [0.12, 0.0, 0.725]
    knee = [0.12, 0.0, 0.50]
    parts = [
        _part(
            "RECIPE_limb_thigh_l",
            radius_m=0.06,
            p0=list(hip),
            p1=list(mid),
        ),
        _part(
            "RECIPE_thigh_taper_dist_l",
            radius_m=0.048,
            p0=list(mid),
            p1=list(knee),
        ),
        _part(
            "RECIPE_hip_bridge",
            role="hip_bridge",
            kind="ellipsoid",
            center=[0.0, 0.0, 0.95],
            rx_m=0.15,
            ry_m=0.06,
            rz_m=0.05,
        ),
        _part(
            "RECIPE_ank_foot_l",
            kind="ellipsoid",
            center=[0.12, 0.04, 0.06],
            rx_m=0.03,
            ry_m=0.03,
            rz_m=0.03,
        ),
        _part(
            "RECIPE_pelvis_oval",
            role="pelvis",
            kind="ellipsoid",
            center=[0.0, 0.0, 0.90],
            rx_m=0.12,
            ry_m=0.08,
            rz_m=0.06,
        ),
    ]
    pkg = BlockoutRecipePackage(parts=parts, counts={"parts": len(parts)})

    # B13: free set excludes thigh when taper_dist sibling present.
    free = _free_parts(pkg, freeze_feet=True)
    free_names = {p.name for p, _, _ in free}
    assert "RECIPE_limb_thigh_l" not in free_names

    # Outer-X projection must not move prox alone (strand dist).
    from meshops.proportion.constraints import part_x

    x0 = part_x(pkg.parts[0])
    _project_hard_constraints(pkg, freeze_feet=True)
    assert part_x(pkg.parts[0]) == pytest.approx(x0)  # type: ignore[arg-type]
    # Split still closed after project
    prox = next(p for p in pkg.parts if p.name == "RECIPE_limb_thigh_l")
    dist = next(p for p in pkg.parts if p.name == "RECIPE_thigh_taper_dist_l")
    assert prox.p1 is not None and dist.p0 is not None
    assert math.dist(prox.p1, dist.p0) == pytest.approx(0.0, abs=1e-9)

    # Optimize fast: either refuses (no free dofs) or keeps split closed.
    try:
        optimized, _result = optimize_package(pkg, mode="fast", freeze_feet=True)
        by_name = {p.name: p for p in optimized.parts}
        p = by_name["RECIPE_limb_thigh_l"]
        d = by_name["RECIPE_thigh_taper_dist_l"]
        assert p.p1 is not None and d.p0 is not None
        assert math.dist(p.p1, d.p0) == pytest.approx(0.0, abs=1e-6)
    except Exception as ei:  # ProportionError optimize_no_free_dofs is OK
        from meshops.proportion.errors import ProportionError

        assert isinstance(ei, ProportionError)
        assert ei.code == "optimize_no_free_dofs"

    # B14: join-ready hip pull co-shifts taper_dist.
    # Hip join is axis=1 (Y). Place thigh far on +Y so gap drives a real pull;
    # X-only offset would already overlap on Y and skip B14 (internal P3-1).
    pull_parts = [
        _part(
            "RECIPE_limb_thigh_l",
            radius_m=0.06,
            p0=[0.12, 0.40, 0.95],
            p1=[0.12, 0.40, 0.725],
        ),
        _part(
            "RECIPE_thigh_taper_dist_l",
            radius_m=0.048,
            p0=[0.12, 0.40, 0.725],
            p1=[0.12, 0.40, 0.50],
        ),
        _part(
            "RECIPE_pelvis_oval",
            role="pelvis",
            kind="ellipsoid",
            center=[0.0, 0.0, 0.90],
            rx_m=0.12,
            ry_m=0.08,
            rz_m=0.06,
        ),
    ]
    dist_before = next(p for p in pull_parts if p.name == "RECIPE_thigh_taper_dist_l")
    assert dist_before.p0 is not None and dist_before.p1 is not None
    dist_p0_y0 = float(dist_before.p0[1])
    dist_p1_y0 = float(dist_before.p1[1])
    prox_p0_y0 = 0.40
    messages: list[str] = []
    _apply_join_ready_overlaps(pull_parts, messages)
    prox2 = next(p for p in pull_parts if p.name == "RECIPE_limb_thigh_l")
    dist2 = next(p for p in pull_parts if p.name == "RECIPE_thigh_taper_dist_l")
    assert prox2.p0 is not None and prox2.p1 is not None
    assert dist2.p0 is not None and dist2.p1 is not None
    # Non-zero Y pull must have run (else B14 untested).
    dy_prox = float(prox2.p0[1]) - prox_p0_y0
    assert abs(dy_prox) > 1e-6
    # Dist co-shifted by same world Δ on both ends; mid join stays closed.
    assert float(dist2.p0[1]) - dist_p0_y0 == pytest.approx(dy_prox, abs=1e-9)
    assert float(dist2.p1[1]) - dist_p1_y0 == pytest.approx(dy_prox, abs=1e-9)
    assert math.dist(prox2.p1, dist2.p0) == pytest.approx(0.0, abs=1e-6)


def test_legacy_single_capsule_adduction_byte_identical() -> None:
    """B8 / AI2 P3-1: no dist seg → legacy single-capsule adduction path."""
    parts = [
        _part(
            "RECIPE_limb_thigh_r",
            radius_m=0.05,
            p0=[0.10, 0.0, 0.95],
            p1=[0.10, 0.0, 0.50],
        ),
        _part(
            "RECIPE_knee_soft_r",
            kind="ellipsoid",
            center=[0.10, 0.0, 0.50],
            rx_m=0.033,
            ry_m=0.033,
            rz_m=0.033,
        ),
    ]
    tpl = _template_applied_tilt(thigh_tilt_deg=10.0)
    messages: list[str] = []
    _apply_thigh_adduction(parts, tpl, messages)  # type: ignore[arg-type]
    thigh = parts[0]
    assert thigh.p1 is not None
    # Right medial: p1.x decreases; full length retained on single capsule
    assert float(thigh.p1[0]) < 0.10 - 1e-6
    assert math.dist(thigh.p0, thigh.p1) == pytest.approx(  # type: ignore[arg-type]
        0.45, abs=1e-6
    )
    assert any("adduction_tilt_deg=" in m for m in messages)
