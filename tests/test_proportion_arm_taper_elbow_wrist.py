"""Track 0062 - Arm shaft taper + elbow soft + wrist continuity (T0-T14)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from meshops.proportion.blockout_recipe import (
    ELBOW_SOFT_MAX_SCALE,
    ELBOW_SOFT_MIN_FRAC_H,
    ELBOW_SOFT_RY_FRAC,
    ELBOW_SOFT_RZ_FRAC,
    ELBOW_SOFT_SCALE,
    FA_DIST_SHAFT_SCALE,
    FA_PROX_SHAFT_SCALE,
    FA_SPLIT_T,
    LIMB_DISTAL_SOFT_SCALE,
    UA_DIST_SHAFT_SCALE,
    UA_PROX_SHAFT_SCALE,
    UA_SPLIT_T,
    WRIST_SOFT_FA_DIST_SCALE,
    WRIST_SOFT_PALM_RX_FRAC,
    _apply_join_ready_overlaps,
    _build_arm_tapered,
    build_blockout_recipe,
)
from meshops.proportion.constraints import (
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
    ua_hw: float = 0.04,
    fa_hw: float | None = None,
    include_hand_lms: bool = False,
) -> ProportionReport:
    """Synthetic full-limb report; optional separate FA half-width + hand LMs."""
    fa = fa_hw if fa_hw is not None else ua_hw
    lms = {
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.05, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.05, z_m=1.38),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=1.72),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=0.05, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=0.05, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=0.05, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=0.05, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
    }
    if include_hand_lms:
        lms["hand_l"] = _lm("hand_l", x_m=-0.33, y_m=0.05, z_m=0.85)
        lms["hand_r"] = _lm("hand_r", x_m=0.33, y_m=0.05, z_m=0.85)
        lms["fingertip_l"] = _lm("fingertip_l", x_m=-0.36, y_m=0.05, z_m=0.72)
        lms["fingertip_r"] = _lm("fingertip_r", x_m=0.36, y_m=0.05, z_m=0.72)
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=ua_hw),
        _diam("upper_arm_r", half_width_m=ua_hw),
        _diam("forearm_l", half_width_m=fa),
        _diam("forearm_r", half_width_m=fa),
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
    """T0: scales/split; ELBOW_SOFT_SCALE==1.22; WRIST floors; bands forearm-only."""
    assert UA_PROX_SHAFT_SCALE == 1.0
    assert UA_DIST_SHAFT_SCALE == 0.84
    assert UA_SPLIT_T == 0.5
    assert FA_PROX_SHAFT_SCALE == 1.0
    assert FA_DIST_SHAFT_SCALE == 0.70
    assert FA_SPLIT_T == 0.5
    assert UA_DIST_SHAFT_SCALE < UA_PROX_SHAFT_SCALE
    assert FA_DIST_SHAFT_SCALE < FA_PROX_SHAFT_SCALE
    assert ELBOW_SOFT_SCALE == 1.22
    assert ELBOW_SOFT_MIN_FRAC_H == 0.016
    assert ELBOW_SOFT_RY_FRAC == 0.90
    assert ELBOW_SOFT_RZ_FRAC == 0.78
    assert ELBOW_SOFT_MAX_SCALE == 1.28
    assert WRIST_SOFT_PALM_RX_FRAC == 0.95
    assert WRIST_SOFT_FA_DIST_SCALE == 1.20
    assert LIMB_DISTAL_SOFT_SCALE == 0.78
    from meshops.proportion.blockout_recipe import _LIMB_DIST_SOFT_BANDS

    assert frozenset({"forearm_l", "forearm_r"}) == _LIMB_DIST_SOFT_BANDS
    assert "upper_arm_l" not in _LIMB_DIST_SOFT_BANDS
    assert "upper_arm_r" not in _LIMB_DIST_SOFT_BANDS
    assert "thigh_l" not in _LIMB_DIST_SOFT_BANDS


def test_t1_emit_both_arm_segments() -> None:
    """T1: limb_upper_arm + arm_taper_dist_ua + forearm pair present."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        ua = by_name[f"RECIPE_limb_upper_arm_{side}"]
        ua_dist = by_name[f"RECIPE_arm_taper_dist_ua_{side}"]
        fa = by_name[f"RECIPE_limb_forearm_{side}"]
        fa_dist = by_name[f"RECIPE_arm_taper_dist_fa_{side}"]
        for p in (ua, ua_dist, fa, fa_dist):
            assert p.kind == "capsule"
            assert p.role == "limb_segment"
            assert p.radius_m is not None
        assert float(ua.radius_m) > float(ua_dist.radius_m)  # type: ignore[arg-type]
        assert float(fa.radius_m) > float(fa_dist.radius_m)  # type: ignore[arg-type]


def test_t2_radii_scales() -> None:
    """T2: UA/FA prox/dist scales on synthetic mid."""
    ua_mid = 0.0438
    fa_mid = 0.0350
    report = _limb_mass_report(ua_hw=ua_mid, fa_hw=fa_mid)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        ua = by_name[f"RECIPE_limb_upper_arm_{side}"]
        ua_dist = by_name[f"RECIPE_arm_taper_dist_ua_{side}"]
        fa = by_name[f"RECIPE_limb_forearm_{side}"]
        fa_dist = by_name[f"RECIPE_arm_taper_dist_fa_{side}"]
        assert float(ua.radius_m) == pytest.approx(  # type: ignore[arg-type]
            ua_mid * UA_PROX_SHAFT_SCALE, abs=1e-9
        )
        assert float(ua_dist.radius_m) == pytest.approx(  # type: ignore[arg-type]
            ua_mid * UA_DIST_SHAFT_SCALE, abs=1e-9
        )
        assert float(fa.radius_m) == pytest.approx(  # type: ignore[arg-type]
            fa_mid * FA_PROX_SHAFT_SCALE, abs=1e-9
        )
        assert float(fa_dist.radius_m) == pytest.approx(  # type: ignore[arg-type]
            fa_mid * FA_DIST_SHAFT_SCALE, abs=1e-9
        )
        ua_ratio = float(ua_dist.radius_m) / float(ua.radius_m)  # type: ignore[arg-type]
        fa_ratio = float(fa_dist.radius_m) / float(fa.radius_m)  # type: ignore[arg-type]
        assert 0.82 <= ua_ratio <= 0.88
        assert 0.66 <= fa_ratio <= 0.74


def test_t3_classifier() -> None:
    """T3: classify prox roles; arm_taper + elbow_soft → unknown."""
    assert classify_part_name("RECIPE_limb_upper_arm_l") == ("upper_arm", "l")
    assert classify_part_name("RECIPE_limb_forearm_r") == ("forearm", "r")
    assert classify_part_name("RECIPE_arm_taper_dist_ua_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_arm_taper_dist_fa_r") == ("unknown", "r")
    assert classify_part_name("RECIPE_elbow_soft_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_elbow_soft_r") == ("unknown", "r")
    assert classify_part_name("RECIPE_dist_soft_forearm_l") == ("unknown", "l")


def test_t4_no_ua_dist_soft() -> None:
    """T4: no dist_soft_upper_arm; forearm dist_soft present."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        assert f"RECIPE_dist_soft_upper_arm_{side}" not in by_name
        assert f"RECIPE_dist_soft_forearm_{side}" in by_name


def test_t5_no_dup_with_split_elbow_wrist() -> None:
    """T5: C_no_dup pass with split + elbow + wrist soft."""
    from meshops.proportion.blockout_recipe import BlockoutRecipePackage

    parts = [
        _part(
            "RECIPE_limb_upper_arm_l",
            radius_m=0.0438,
            p0=[-0.20, 0.05, 1.38],
            p1=[-0.225, 0.05, 1.24],
        ),
        _part(
            "RECIPE_arm_taper_dist_ua_l",
            radius_m=0.0385,
            p0=[-0.225, 0.05, 1.24],
            p1=[-0.25, 0.05, 1.10],
        ),
        _part(
            "RECIPE_limb_forearm_l",
            radius_m=0.0350,
            p0=[-0.25, 0.05, 1.10],
            p1=[-0.275, 0.05, 1.00],
        ),
        _part(
            "RECIPE_arm_taper_dist_fa_l",
            radius_m=0.0273,
            p0=[-0.275, 0.05, 1.00],
            p1=[-0.30, 0.05, 0.90],
        ),
        _part(
            "RECIPE_elbow_soft_l",
            kind="ellipsoid",
            center=[-0.25, 0.05, 1.10],
            rx_m=0.0470,
            ry_m=0.0423,
            rz_m=0.0367,
        ),
        _part(
            "RECIPE_dist_soft_forearm_l",
            kind="ellipsoid",
            center=[-0.30, 0.05, 0.90],
            rx_m=0.0273,
            ry_m=0.0273,
            rz_m=0.0273,
        ),
    ]
    pkg = BlockoutRecipePackage(parts=parts, counts={"parts": len(parts)})
    result = validate_constraints(pkg)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_no_dup_limb"].status == "pass", by_id["C_no_dup_limb"].message


def test_t6_wrist_soft_center_and_scale() -> None:
    """T6: soft.center == fa taper p1; soft == max(mid*0.78, fa_dist*1.20) no hands."""
    arm_hw = 0.04
    report = _limb_mass_report(ua_hw=arm_hw, fa_hw=arm_hw)
    pkg = build_blockout_recipe(report, limbs=True, hands=False)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        fa_dist = by_name[f"RECIPE_arm_taper_dist_fa_{side}"]
        soft = by_name[f"RECIPE_dist_soft_forearm_{side}"]
        assert soft.center is not None and fa_dist.p1 is not None
        for i in range(3):
            assert float(soft.center[i]) == pytest.approx(float(fa_dist.p1[i]), abs=1e-9)
        mid_emit = max(arm_hw * LIMB_DISTAL_SOFT_SCALE, 1e-4)
        fa_floor = float(fa_dist.radius_m) * WRIST_SOFT_FA_DIST_SCALE  # type: ignore[arg-type]
        expected = max(mid_emit, fa_floor)
        assert soft.rx_m == pytest.approx(expected, abs=1e-9)
        # Product-class pin: FA dist * 1.20 (0.70*0.04*1.20 = 0.0336), not mid*0.78 alone (0.0312)
        assert expected == pytest.approx(0.0336, abs=1e-5)


def test_t7_elbow_soft_readable() -> None:
    """T7: elbow rx == min(max(scale*adj, floor_H), MAX*adj); ry/rz aniso; > shafts."""
    height_m = 1.72
    ua_hw = 0.0438
    fa_hw = 0.0350
    report = _limb_mass_report(height_m=height_m, ua_hw=ua_hw, fa_hw=fa_hw)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        elbow = by_name[f"RECIPE_elbow_soft_{side}"]
        ua_dist = by_name[f"RECIPE_arm_taper_dist_ua_{side}"]
        fa = by_name[f"RECIPE_limb_forearm_{side}"]
        adj = max(float(ua_dist.radius_m), float(fa.radius_m))  # type: ignore[arg-type]
        expected = ELBOW_SOFT_SCALE * adj
        expected = max(expected, ELBOW_SOFT_MIN_FRAC_H * height_m)
        expected = min(expected, ELBOW_SOFT_MAX_SCALE * adj)
        assert elbow.rx_m == pytest.approx(expected, abs=1e-9)
        assert elbow.ry_m == pytest.approx(expected * ELBOW_SOFT_RY_FRAC, abs=1e-9)
        assert elbow.rz_m == pytest.approx(expected * ELBOW_SOFT_RZ_FRAC, abs=1e-9)
        assert float(elbow.rx_m) > float(ua_dist.radius_m)  # type: ignore[arg-type]
        assert float(elbow.rx_m) > float(fa.radius_m)  # type: ignore[arg-type]
        # Seam center = ua_dist.p1
        assert elbow.center is not None and ua_dist.p1 is not None
        for i in range(3):
            assert float(elbow.center[i]) == pytest.approx(float(ua_dist.p1[i]), abs=1e-9)


def test_t8_no_seam_or_no_limbs_skips_elbow() -> None:
    """T8: no seam / no limbs → no elbow_soft."""
    report = _limb_mass_report()
    pkg_no = build_blockout_recipe(report, limbs=False)
    assert not any("elbow_soft" in p.name for p in pkg_no.parts)

    # Only torso diams — no arm segments → no elbow soft
    lms = report.landmarks_xyz
    sparse = ProportionReport(
        schema_version="1.0.0",
        height_m=1.72,
        landmarks_xyz=lms,
        diameters=[
            _diam("bust", half_width_m=0.16),
            _diam("waist", half_width_m=0.13),
            _diam("neck", half_width_m=0.05),
        ],
        quality=QualityFlags(),
    )
    pkg_sparse = build_blockout_recipe(sparse, limbs=True)
    assert not any("elbow_soft" in p.name for p in pkg_sparse.parts)


def test_t9_wrist_palm_floor() -> None:
    """T9: with palm (hands=True), wrist soft >= 0.95*palm.rx + honesty message."""
    report = _limb_mass_report(include_hand_lms=True, fa_hw=0.0350)
    pkg = build_blockout_recipe(report, limbs=True, hands=True, fingers="mitten")
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        palm = by_name.get(f"RECIPE_palm_{side}")
        soft = by_name[f"RECIPE_dist_soft_forearm_{side}"]
        assert palm is not None and palm.rx_m is not None
        floor = WRIST_SOFT_PALM_RX_FRAC * float(palm.rx_m)
        assert soft.rx_m is not None
        assert float(soft.rx_m) >= floor - 1e-9
        # Soft re-pinned at fa taper wrist
        fa_dist = by_name[f"RECIPE_arm_taper_dist_fa_{side}"]
        assert soft.center is not None and fa_dist.p1 is not None
        for i in range(3):
            assert float(soft.center[i]) == pytest.approx(float(fa_dist.p1[i]), abs=1e-9)
        # R9/B12: palm-floor honesty message uses const frac (not hard-coded 0.85)
        palm_msgs = [m for m in pkg.messages if f"wrist_soft_{side}: palm_floor" in m]
        assert palm_msgs
        assert f"({WRIST_SOFT_PALM_RX_FRAC:.2f}*palm.rx)" in palm_msgs[0]
        assert "0.85*palm" not in palm_msgs[0]


def test_t10_product_like_mids() -> None:
    """T10: product-like mids 0.0438/0.0350 → dist ≈ 0.0368/0.0245; elbow ≈ 0.0449."""
    ua_mid = 0.0438
    fa_mid = 0.0350
    height_m = 1.72
    report = _limb_mass_report(height_m=height_m, ua_hw=ua_mid, fa_hw=fa_mid)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        ua = by_name[f"RECIPE_limb_upper_arm_{side}"]
        ua_dist = by_name[f"RECIPE_arm_taper_dist_ua_{side}"]
        fa = by_name[f"RECIPE_limb_forearm_{side}"]
        fa_dist = by_name[f"RECIPE_arm_taper_dist_fa_{side}"]
        elbow = by_name[f"RECIPE_elbow_soft_{side}"]
        assert float(ua.radius_m) == pytest.approx(0.0438, abs=1e-5)  # type: ignore[arg-type]
        assert float(ua_dist.radius_m) == pytest.approx(0.036792, abs=1e-5)  # type: ignore[arg-type]
        assert float(fa.radius_m) == pytest.approx(0.0350, abs=1e-5)  # type: ignore[arg-type]
        assert float(fa_dist.radius_m) == pytest.approx(0.0245, abs=1e-5)  # type: ignore[arg-type]
        # 1.22 * max(0.036792, 0.0350) = 0.04488624
        adj = max(float(ua_dist.radius_m), float(fa.radius_m))  # type: ignore[arg-type]
        expected = ELBOW_SOFT_SCALE * adj
        assert float(elbow.rx_m) == pytest.approx(expected, abs=1e-4)  # type: ignore[arg-type]
        assert float(elbow.rx_m) == pytest.approx(0.0449, abs=1e-4)  # type: ignore[arg-type]


def test_t11_messages_shaft_taper_and_elbow() -> None:
    """T11: messages shaft_taper + elbow_soft rx= + wrist palm_floor (R9/B12)."""
    report = _limb_mass_report(include_hand_lms=True, fa_hw=0.0350)
    pkg = build_blockout_recipe(report, limbs=True, hands=True, fingers="mitten")
    assert any("upper_arm_l: shaft_taper" in m for m in pkg.messages)
    assert any("upper_arm_r: shaft_taper" in m for m in pkg.messages)
    assert any("forearm_l: shaft_taper" in m for m in pkg.messages)
    assert any("forearm_r: shaft_taper" in m for m in pkg.messages)
    assert any("elbow_soft_l: rx=" in m for m in pkg.messages)
    assert any("elbow_soft_r: rx=" in m for m in pkg.messages)
    assert not any(m.startswith("elbow_soft_l: r=") and "rx=" not in m for m in pkg.messages)
    assert any("wrist_soft_l: palm_floor" in m for m in pkg.messages)
    assert any("wrist_soft_r: palm_floor" in m for m in pkg.messages)


def test_t12_split_mid_geometry() -> None:
    """T12: mid = lerp(p0,p1,0.5); split closed ||prox.p1-dist.p0||~0."""
    p0 = [-0.20, 0.05, 1.38]
    p1 = [-0.25, 0.05, 1.10]
    messages: list[str] = []
    parts = _build_arm_tapered(
        side="l",
        band="ua",
        p0=p0,
        p1=p1,
        radius=0.04,
        placement="full3d",
        messages=messages,
    )
    prox, dist = parts[0], parts[1]
    assert prox.p0 is not None and prox.p1 is not None
    assert dist.p0 is not None and dist.p1 is not None
    t = UA_SPLIT_T
    for i in range(3):
        mid = float(p0[i]) + t * (float(p1[i]) - float(p0[i]))
        assert float(prox.p1[i]) == pytest.approx(mid, abs=1e-12)
        assert float(dist.p0[i]) == pytest.approx(mid, abs=1e-12)
    assert math.dist(prox.p1, dist.p0) == pytest.approx(0.0, abs=1e-12)

    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        for prefix, dist_name in (
            ("limb_upper_arm", "arm_taper_dist_ua"),
            ("limb_forearm", "arm_taper_dist_fa"),
        ):
            prox = by_name[f"RECIPE_{prefix}_{side}"]
            dist = by_name[f"RECIPE_{dist_name}_{side}"]
            assert prox.p1 is not None and dist.p0 is not None
            assert math.dist(prox.p1, dist.p0) == pytest.approx(0.0, abs=1e-9)


def test_t13_ua_chain_length() -> None:
    """T13: ||ua_taper.p1 - limb_ua.p0|| ~ full shoulder-elbow length."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_upper_arm_{side}"]
        dist = by_name[f"RECIPE_arm_taper_dist_ua_{side}"]
        assert prox.p0 is not None and dist.p1 is not None and prox.p1 is not None
        chain = math.dist(prox.p0, dist.p1)
        # shoulder_z 1.38 → elbow_z 1.10 + small xy → ~0.28+
        assert chain > 0.25
        assert math.dist(prox.p0, prox.p1) == pytest.approx(chain * UA_SPLIT_T, abs=1e-5)
        # FA chain elbow→wrist
        fa = by_name[f"RECIPE_limb_forearm_{side}"]
        fa_dist = by_name[f"RECIPE_arm_taper_dist_fa_{side}"]
        assert fa.p0 is not None and fa_dist.p1 is not None
        fa_chain = math.dist(fa.p0, fa_dist.p1)
        assert fa_chain > 0.15


def test_t14_join_ready_leaves_arm_coords_unchanged() -> None:
    """T14: join-ready leaves arm limb + taper_dist world coords unchanged (B14)."""
    from meshops.proportion.blockout_recipe import BlockoutRecipePackage

    # Product-shaped: shoulder join is deltoid/bridge only — arms not free-set.
    sh = [-0.20, 0.40, 1.38]
    mid_ua = [-0.225, 0.40, 1.24]
    elbow = [-0.25, 0.40, 1.10]
    mid_fa = [-0.275, 0.40, 1.00]
    wrist = [-0.30, 0.40, 0.90]
    parts = [
        _part(
            "RECIPE_limb_upper_arm_l",
            radius_m=0.0438,
            p0=list(sh),
            p1=list(mid_ua),
        ),
        _part(
            "RECIPE_arm_taper_dist_ua_l",
            radius_m=0.0385,
            p0=list(mid_ua),
            p1=list(elbow),
        ),
        _part(
            "RECIPE_limb_forearm_l",
            radius_m=0.0350,
            p0=list(elbow),
            p1=list(mid_fa),
        ),
        _part(
            "RECIPE_arm_taper_dist_fa_l",
            radius_m=0.0273,
            p0=list(mid_fa),
            p1=list(wrist),
        ),
        _part(
            "RECIPE_deltoid_soft_l",
            role="deltoid_soft",
            kind="ellipsoid",
            center=[-0.20, 0.0, 1.38],
            rx_m=0.06,
            ry_m=0.05,
            rz_m=0.05,
        ),
        _part(
            "RECIPE_torso_trap",
            role="torso",
            kind="capsule",
            p0=[0.0, 0.0, 1.10],
            p1=[0.0, 0.0, 1.40],
            radius_m=0.12,
        ),
    ]
    before = {
        p.name: (
            list(p.p0) if p.p0 is not None else None,
            list(p.p1) if p.p1 is not None else None,
            float(p.radius_m) if p.radius_m is not None else None,
        )
        for p in parts
        if "arm" in p.name.lower() or "forearm" in p.name.lower()
    }
    messages: list[str] = []
    _apply_join_ready_overlaps(parts, messages)
    for p in parts:
        if p.name not in before:
            continue
        p0_b, p1_b, r_b = before[p.name]
        assert p.p0 is not None and p1_b is not None and p.p1 is not None
        assert p0_b is not None
        for i in range(3):
            assert float(p.p0[i]) == pytest.approx(float(p0_b[i]), abs=1e-9)
            assert float(p.p1[i]) == pytest.approx(float(p1_b[i]), abs=1e-9)
        assert float(p.radius_m) == pytest.approx(r_b, abs=1e-9)  # type: ignore[arg-type]

    # Optimize fast: arms not free → either refuse or leave coords unchanged
    pkg = BlockoutRecipePackage(parts=list(parts), counts={"parts": len(parts)})
    try:
        optimized, _result = optimize_package(pkg, mode="fast", freeze_feet=True)
        by_name = {p.name: p for p in optimized.parts}
        for name, (p0_b, p1_b, r_b) in before.items():
            p = by_name[name]
            assert p.p0 is not None and p.p1 is not None and p0_b is not None and p1_b is not None
            for i in range(3):
                assert float(p.p0[i]) == pytest.approx(float(p0_b[i]), abs=1e-6)
                assert float(p.p1[i]) == pytest.approx(float(p1_b[i]), abs=1e-6)
            assert float(p.radius_m) == pytest.approx(r_b, abs=1e-6)  # type: ignore[arg-type]
    except Exception as ei:
        from meshops.proportion.errors import ProportionError

        assert isinstance(ei, ProportionError)
        assert ei.code == "optimize_no_free_dofs"


def test_no_dead_co_shift_helper() -> None:
    """B14 fence: do not ship unused _co_shift_arm_taper_dist."""
    import meshops.proportion.blockout_recipe as br

    assert not hasattr(br, "_co_shift_arm_taper_dist")
