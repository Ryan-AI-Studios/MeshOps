"""Track 0063 - Arm muscle softs (bicep scale+front past + triceps append) T0-T14."""

from __future__ import annotations

from typing import Any

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    BICEP_ALONG_T,
    BICEP_ARM_RX_SCALE,
    BICEP_FRONT_PAST_M,
    BICEP_RY_FRAC,
    BICEP_RZ_FRAC,
    DELT_ARM_RADIUS_SCALE,
    ELBOW_SOFT_SCALE,
    TRICEP_ALONG_T,
    TRICEP_ARM_RX_SCALE,
    TRICEP_REAR_PAST_M,
    TRICEP_RY_FRAC,
    TRICEP_RZ_FRAC,
    UA_DIST_SHAFT_SCALE,
    UA_PROX_SHAFT_SCALE,
    BlockoutRecipePackage,
    RecipePart,
    _apply_arm_muscle_softs,
    _ua_shaft_metrics,
    build_blockout_recipe,
)
from meshops.proportion.constraints import classify_part_name, validate_constraints
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
    ua_hw: float = 0.0438,
    fa_hw: float | None = None,
    shoulder_y: float = 0.05,
) -> ProportionReport:
    """Synthetic full-limb report for 0063 arm muscle softs tests."""
    fa = fa_hw if fa_hw is not None else 0.0350
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=shoulder_y, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=shoulder_y, z_m=1.38),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=height_m),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=shoulder_y, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=shoulder_y, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=shoulder_y, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=shoulder_y, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=ua_hw),
        _diam("upper_arm_r", half_width_m=ua_hw),
        _diam("forearm_l", half_width_m=fa),
        _diam("forearm_r", half_width_m=fa),
        _diam("thigh_l", half_width_m=0.06),
        _diam("thigh_r", half_width_m=0.06),
        _diam("calf_l", half_width_m=0.05),
        _diam("calf_r", half_width_m=0.05),
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
) -> RecipePart:
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


def _build_limbs_profile(
    *,
    ua_hw: float = 0.0438,
    fa_hw: float = 0.0350,
    profile_id: str = "torso_limb_f_athletic_v1",
) -> Any:
    report = _limb_mass_report(ua_hw=ua_hw, fa_hw=fa_hw)
    profile = load_anatomy_profile(profile_id)
    return build_blockout_recipe(
        report,
        limbs=True,
        torso="ovals",
        profile=profile,
    )


# ---------------------------------------------------------------------------
# T0-T14
# ---------------------------------------------------------------------------


def test_t0_const_freezes() -> None:
    """T0: B1-B8 const pins."""
    assert BICEP_ARM_RX_SCALE == 0.78
    assert BICEP_RY_FRAC == 0.90
    assert BICEP_RZ_FRAC == 0.95
    assert BICEP_FRONT_PAST_M == 0.010
    assert BICEP_ALONG_T == 0.50
    assert TRICEP_ARM_RX_SCALE == 0.82
    assert TRICEP_RY_FRAC == 0.88
    assert TRICEP_RZ_FRAC == 0.92
    assert TRICEP_REAR_PAST_M == 0.010
    assert TRICEP_ALONG_T == 0.50


def test_t1_bicep_rx_not_055() -> None:
    """T1: limbs+profile -> bicep rx == ua_r * 0.78; rejects 0.55x."""
    ua_hw = 0.0438
    pkg = _build_limbs_profile(ua_hw=ua_hw)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        ua = by[f"RECIPE_limb_upper_arm_{side}"]
        bicep = by[f"RECIPE_bicep_soft_{side}"]
        ua_r = float(ua.radius_m)  # type: ignore[arg-type]
        assert bicep.rx_m is not None
        assert float(bicep.rx_m) == pytest.approx(ua_r * BICEP_ARM_RX_SCALE, abs=1e-6)
        assert float(bicep.rx_m) != pytest.approx(ua_r * 0.55, abs=1e-4)
        assert float(bicep.rx_m) > ua_r * 0.55 + 1e-4


def test_t2_bicep_front_past() -> None:
    """T2: front past >= BICEP_FRONT_PAST_M - eps."""
    pkg = _build_limbs_profile()
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        metrics = _ua_shaft_metrics(list(pkg.parts), side, along_t=BICEP_ALONG_T)
        assert metrics is not None
        ua_r, mid = metrics
        bicep = by[f"RECIPE_bicep_soft_{side}"]
        assert bicep.center is not None and bicep.ry_m is not None
        shaft_front = float(mid[1]) - ua_r
        cy = float(bicep.center[1])
        ry = float(bicep.ry_m)
        past = shaft_front - (cy - ry)
        assert past >= BICEP_FRONT_PAST_M - 1e-6


def test_t3_triceps_present_when_profile() -> None:
    """T3: RECIPE_triceps_soft_l/r present when profile+limbs."""
    pkg = _build_limbs_profile()
    names = {p.name for p in pkg.parts}
    assert "RECIPE_triceps_soft_l" in names
    assert "RECIPE_triceps_soft_r" in names
    for side in ("l", "r"):
        tri = next(p for p in pkg.parts if p.name == f"RECIPE_triceps_soft_{side}")
        assert tri.role == "limb_segment"
        assert tri.kind == "ellipsoid"
        assert tri.center is not None
        assert tri.rx_m is not None


def test_t4_triceps_rear_past() -> None:
    """T4: triceps rear past >= TRICEP_REAR_PAST_M - eps."""
    pkg = _build_limbs_profile()
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        metrics = _ua_shaft_metrics(list(pkg.parts), side, along_t=TRICEP_ALONG_T)
        assert metrics is not None
        ua_r, tmid = metrics
        tri = by[f"RECIPE_triceps_soft_{side}"]
        assert tri.center is not None and tri.ry_m is not None
        shaft_rear = float(tmid[1]) + ua_r
        tcy = float(tri.center[1])
        try_ = float(tri.ry_m)
        rear_past = (tcy + try_) - shaft_rear
        assert rear_past >= TRICEP_REAR_PAST_M - 1e-6


def test_t5_c_no_dup_with_arms_elbow_bicep_triceps() -> None:
    """T5: C_no_dup pass with split arms + elbow + bicep + triceps."""
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
            rx_m=0.0424,
            ry_m=0.0424,
            rz_m=0.0424,
        ),
        _part(
            "RECIPE_bicep_soft_l",
            kind="ellipsoid",
            role="bicep_soft",
            center=[-0.225, -0.08, 1.24],
            rx_m=0.0342,
            ry_m=0.0308,
            rz_m=0.0325,
        ),
        _part(
            "RECIPE_triceps_soft_l",
            kind="ellipsoid",
            center=[-0.225, 0.12, 1.24],
            rx_m=0.0359,
            ry_m=0.0316,
            rz_m=0.0330,
        ),
    ]
    pkg = BlockoutRecipePackage(parts=parts, counts={"parts": len(parts)})
    result = validate_constraints(pkg)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_no_dup_limb"].status == "pass", by_id["C_no_dup_limb"].message


def test_t6_classifier_triceps_unknown_bicep_upper_arm() -> None:
    """T6: classify triceps→unknown; bicep→upper_arm."""
    assert classify_part_name("RECIPE_triceps_soft_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_triceps_soft_r") == ("unknown", "r")
    assert classify_part_name("RECIPE_bicep_soft_l") == ("upper_arm", "l")
    assert classify_part_name("RECIPE_bicep_soft_r") == ("upper_arm", "r")


def test_t7_fence_0062_taper_elbow() -> None:
    """T7: 0062 fence — UA dist/prox scales; elbow r > max adj."""
    ua_mid = 0.0438
    fa_mid = 0.0350
    height_m = 1.72
    report = _limb_mass_report(height_m=height_m, ua_hw=ua_mid, fa_hw=fa_mid)
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        ua = by[f"RECIPE_limb_upper_arm_{side}"]
        ua_dist = by[f"RECIPE_arm_taper_dist_ua_{side}"]
        elbow = by[f"RECIPE_elbow_soft_{side}"]
        fa = by[f"RECIPE_limb_forearm_{side}"]
        assert float(ua.radius_m) == pytest.approx(  # type: ignore[arg-type]
            ua_mid * UA_PROX_SHAFT_SCALE, abs=1e-9
        )
        assert float(ua_dist.radius_m) == pytest.approx(  # type: ignore[arg-type]
            ua_mid * UA_DIST_SHAFT_SCALE, abs=1e-9
        )
        adj = max(float(ua_dist.radius_m), float(fa.radius_m))  # type: ignore[arg-type]
        assert float(elbow.rx_m) > adj  # type: ignore[arg-type]
        assert float(elbow.rx_m) == pytest.approx(  # type: ignore[arg-type]
            ELBOW_SOFT_SCALE * adj, abs=1e-6
        )


def test_t8_fence_0060_delt_not_055() -> None:
    """T8: 0060 fence - delt still DELT scale path (not x0.55)."""
    arm_hw = 0.0438
    report = _limb_mass_report(ua_hw=arm_hw)
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(report, limbs=True, profile=profile)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    expected = arm_hw * DELT_ARM_RADIUS_SCALE
    for d in delts:
        assert d.rx_m is not None
        assert float(d.rx_m) >= expected - 1e-6
        assert float(d.rx_m) != pytest.approx(arm_hw * 0.55, abs=1e-3)


def test_t9_messages_bicep_and_triceps() -> None:
    """T9: messages contain bicep_soft + triceps_soft."""
    pkg = _build_limbs_profile()
    assert any("bicep_soft_l:" in m for m in pkg.messages)
    assert any("bicep_soft_r:" in m for m in pkg.messages)
    assert any("triceps_soft_l:" in m for m in pkg.messages)
    assert any("triceps_soft_r:" in m for m in pkg.messages)
    assert any("front_past=" in m for m in pkg.messages)
    assert any("rear_past=" in m for m in pkg.messages)


def test_t10_product_like_scales() -> None:
    """T10: product-like ua_r=0.0438 → bicep≈0.0342, triceps≈0.0359."""
    ua_hw = 0.0438
    pkg = _build_limbs_profile(ua_hw=ua_hw)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        bicep = by[f"RECIPE_bicep_soft_{side}"]
        tri = by[f"RECIPE_triceps_soft_{side}"]
        assert float(bicep.rx_m) == pytest.approx(0.034164, abs=1e-4)  # type: ignore[arg-type]
        assert float(tri.rx_m) == pytest.approx(0.035916, abs=1e-4)  # type: ignore[arg-type]
        # Convenience pins from plan
        assert float(bicep.rx_m) == pytest.approx(0.0342, abs=1e-3)  # type: ignore[arg-type]
        assert float(tri.rx_m) == pytest.approx(0.0359, abs=1e-3)  # type: ignore[arg-type]


def test_t11_full_chain_mid_xz_and_y_past() -> None:
    """T11: limbs+profile full-chain mid X/Z; Y from past law."""
    pkg = _build_limbs_profile()
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        metrics = _ua_shaft_metrics(list(pkg.parts), side, along_t=BICEP_ALONG_T)
        assert metrics is not None
        ua_r, mid = metrics
        bicep = by[f"RECIPE_bicep_soft_{side}"]
        assert bicep.center is not None and bicep.ry_m is not None
        assert float(bicep.center[0]) == pytest.approx(float(mid[0]), abs=1e-6)
        assert float(bicep.center[2]) == pytest.approx(float(mid[2]), abs=1e-6)
        # Y is front-past law, not mid bone Y
        shaft_front = float(mid[1]) - ua_r
        expected_cy = shaft_front - BICEP_FRONT_PAST_M + float(bicep.ry_m)
        assert float(bicep.center[1]) == pytest.approx(expected_cy, abs=1e-6)
        assert float(bicep.center[1]) != pytest.approx(float(mid[1]), abs=1e-3)


def test_t12_no_ua_skips_softs() -> None:
    """T12: no limb_upper_arm → no crash; softs skipped."""
    # Profile without limbs: bicep pea present; post-pass skips (no UA).
    report = _limb_mass_report()
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(report, limbs=False, profile=profile)
    assert not any(p.name.startswith("RECIPE_triceps_soft_") for p in pkg.parts)
    assert not any("bicep_soft_l: rx=" in m for m in pkg.messages)
    assert not any("triceps_soft_" in m for m in pkg.messages)

    # Direct helper: empty parts is quiet
    messages: list[str] = []
    parts: list[RecipePart] = []
    _apply_arm_muscle_softs(parts, messages)
    assert parts == []
    assert messages == []
    assert _ua_shaft_metrics([], "l", along_t=0.5) is None


def test_t13_full_chain_mid_ne_prox_only() -> None:
    """T13: full-chain mid ≠ prox-only mid when taper present (B15)."""
    # Asymmetric UA: shoulder → elbow with large X/Z change; split at mid.
    # Prox capsule ends at seam; full chain goes to elbow.
    parts = [
        _part(
            "RECIPE_limb_upper_arm_l",
            radius_m=0.04,
            p0=[-0.20, 0.05, 1.38],
            p1=[-0.225, 0.05, 1.24],  # seam @ t=0.5 of full chain
        ),
        _part(
            "RECIPE_arm_taper_dist_ua_l",
            radius_m=0.035,
            p0=[-0.225, 0.05, 1.24],
            p1=[-0.25, 0.05, 1.10],  # elbow
        ),
    ]
    metrics = _ua_shaft_metrics(parts, "l", along_t=0.50)
    assert metrics is not None
    _, full_mid = metrics
    # Prox-only mid = lerp(prox.p0, prox.p1, 0.5)
    prox = parts[0]
    assert prox.p0 is not None and prox.p1 is not None
    prox_mid = [
        float(prox.p0[0]) + 0.5 * (float(prox.p1[0]) - float(prox.p0[0])),
        float(prox.p0[1]) + 0.5 * (float(prox.p1[1]) - float(prox.p0[1])),
        float(prox.p0[2]) + 0.5 * (float(prox.p1[2]) - float(prox.p0[2])),
    ]
    # Full-chain mid at t=0.5 == seam (prox.p1); prox-only mid is t=0.25 class
    assert full_mid[0] != pytest.approx(prox_mid[0], abs=1e-9)
    assert full_mid[2] != pytest.approx(prox_mid[2], abs=1e-9)
    # Full mid X/Z should match lerp(shoulder, elbow, 0.5)
    shoulder = [-0.20, 0.05, 1.38]
    elbow = [-0.25, 0.05, 1.10]
    expected = [
        shoulder[0] + 0.5 * (elbow[0] - shoulder[0]),
        shoulder[1] + 0.5 * (elbow[1] - shoulder[1]),
        shoulder[2] + 0.5 * (elbow[2] - shoulder[2]),
    ]
    for i in range(3):
        assert float(full_mid[i]) == pytest.approx(expected[i], abs=1e-9)


def test_t14_no_profile_no_triceps() -> None:
    """T14: limbs=True, no profile → no RECIPE_triceps_soft_* (B16)."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True, torso="ovals")
    assert not any(p.name.startswith("RECIPE_triceps_soft_") for p in pkg.parts)
    assert not any(p.role == "bicep_soft" for p in pkg.parts)
    assert not any("triceps_soft_" in m for m in pkg.messages)
    # UA still present
    assert any(p.name.startswith("RECIPE_limb_upper_arm_") for p in pkg.parts)
