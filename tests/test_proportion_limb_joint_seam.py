"""Track 0081 — Limb joint seam soften (elbow/knee/wrist retune pins)."""

from __future__ import annotations

import pytest

from meshops.proportion.blockout_recipe import (
    ELBOW_SOFT_RY_FRAC,
    ELBOW_SOFT_RZ_FRAC,
    ELBOW_SOFT_SCALE,
    FA_DIST_SHAFT_SCALE,
    FA_PROX_SHAFT_SCALE,
    KNEE_SOFT_RY_FRAC,
    KNEE_SOFT_RZ_FRAC,
    THIGH_DIST_SHAFT_SCALE,
    THIGH_PROX_SHAFT_SCALE,
    UA_DIST_SHAFT_SCALE,
    UA_PROX_SHAFT_SCALE,
    WRIST_SOFT_FA_DIST_SCALE,
    WRIST_SOFT_PALM_RX_FRAC,
    build_blockout_recipe,
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


def _product_class_report(
    *,
    height_m: float = 1.72,
    ua_hw: float = 0.0438,
    fa_hw: float = 0.0350,
    thigh_hw: float = 0.0613,
    calf_hw: float = 0.0438,
    include_hand_lms: bool = False,
) -> ProportionReport:
    """Product_0072up-class mids for 0081 joint-seam pins."""
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
        _diam("forearm_l", half_width_m=fa_hw),
        _diam("forearm_r", half_width_m=fa_hw),
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


def test_t_elbow_past_ua_prox() -> None:
    """R1: product-class elbow.rx > limb_upper_arm.radius (soft owns the step)."""
    report = _product_class_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        elbow = by[f"RECIPE_elbow_soft_{side}"]
        ua = by[f"RECIPE_limb_upper_arm_{side}"]
        assert elbow.rx_m is not None and ua.radius_m is not None
        assert float(elbow.rx_m) > float(ua.radius_m)
        # Live product pin class: ~0.0470 > ~0.0438
        assert float(elbow.rx_m) == pytest.approx(ELBOW_SOFT_SCALE * 0.038544, abs=1e-4)


def test_t_elbow_aniso() -> None:
    """R2: elbow ry/rx ≈ 0.90; rz/rx ≈ 0.78 (±0.02)."""
    report = _product_class_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        elbow = by[f"RECIPE_elbow_soft_{side}"]
        assert elbow.rx_m is not None and elbow.ry_m is not None and elbow.rz_m is not None
        rx = float(elbow.rx_m)
        assert rx > 0.0
        assert float(elbow.ry_m) / rx == pytest.approx(ELBOW_SOFT_RY_FRAC, abs=0.02)
        assert float(elbow.rz_m) / rx == pytest.approx(ELBOW_SOFT_RZ_FRAC, abs=0.02)
        # Not pure sphere
        assert abs(float(elbow.ry_m) - rx) > 1e-6 or abs(float(elbow.rz_m) - rx) > 1e-6


def test_t_knee_sleeve() -> None:
    """R3: knee sleeve rz/rx ≥ 0.90 and rz ≥ ry."""
    report = _product_class_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        knee = by[f"RECIPE_knee_soft_{side}"]
        assert knee.rx_m is not None and knee.ry_m is not None and knee.rz_m is not None
        rx, ry, rz = float(knee.rx_m), float(knee.ry_m), float(knee.rz_m)
        assert rx > 0.0
        assert rz / rx >= 0.90 - 1e-9
        assert rz >= ry - 1e-12
        assert ry / rx == pytest.approx(KNEE_SOFT_RY_FRAC, abs=0.02)
        assert rz / rx == pytest.approx(KNEE_SOFT_RZ_FRAC, abs=0.02)


def test_t_wrist_palm() -> None:
    """R4 palm: wrist.rx / palm.rx ≥ 0.94 when both present."""
    report = _product_class_report(include_hand_lms=True)
    pkg = build_blockout_recipe(report, limbs=True, hands=True, fingers="mitten")
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by[f"RECIPE_dist_soft_forearm_{side}"]
        palm = by.get(f"RECIPE_palm_{side}")
        assert palm is not None and palm.rx_m is not None and soft.rx_m is not None
        ratio = float(soft.rx_m) / float(palm.rx_m)
        assert ratio >= 0.94 - 1e-9
        assert float(soft.rx_m) >= WRIST_SOFT_PALM_RX_FRAC * float(palm.rx_m) - 1e-9


def test_t_wrist_fa() -> None:
    """R4 FA safety net: hands=False -> wrist.rx >= 1.15 * fa_dist."""
    report = _product_class_report()
    pkg = build_blockout_recipe(report, limbs=True, hands=False)
    by = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by[f"RECIPE_dist_soft_forearm_{side}"]
        fa_dist = by[f"RECIPE_arm_taper_dist_fa_{side}"]
        assert soft.rx_m is not None and fa_dist.radius_m is not None
        assert float(soft.rx_m) >= 1.15 * float(fa_dist.radius_m) - 1e-9
        assert float(soft.rx_m) >= WRIST_SOFT_FA_DIST_SCALE * float(fa_dist.radius_m) - 1e-9
        assert any(f"wrist_soft_{side}: fa_floor" in m for m in pkg.messages)


def test_t_no_new_sleeve_names() -> None:
    """R5: no joint_sleeve / condyle RECIPE names (retune existing softs only)."""
    report = _product_class_report(include_hand_lms=True)
    pkg = build_blockout_recipe(report, limbs=True, hands=True, fingers="mitten")
    names = [p.name.lower() for p in pkg.parts]
    assert not any("joint_sleeve" in n for n in names)
    assert not any("condyle" in n for n in names)
    # Existing joint softs still present
    assert any("elbow_soft" in n for n in names)
    assert any("knee_soft" in n for n in names)
    assert any("dist_soft_forearm" in n for n in names)


def test_t_fence_shafts() -> None:
    """R6: UA/FA/thigh shaft scales stay 1.00/0.88/0.78/0.80 class."""
    assert UA_PROX_SHAFT_SCALE == 1.00
    assert UA_DIST_SHAFT_SCALE == 0.88
    assert FA_PROX_SHAFT_SCALE == 1.00
    assert FA_DIST_SHAFT_SCALE == 0.78
    assert THIGH_PROX_SHAFT_SCALE == 1.00
    assert THIGH_DIST_SHAFT_SCALE == 0.80
    report = _product_class_report()
    pkg = build_blockout_recipe(report, limbs=True)
    by = {p.name: p for p in pkg.parts}
    ua_mid = 0.0438
    fa_mid = 0.0350
    thigh_mid = 0.0613
    for side in ("l", "r"):
        assert float(by[f"RECIPE_limb_upper_arm_{side}"].radius_m) == pytest.approx(  # type: ignore[arg-type]
            ua_mid * UA_PROX_SHAFT_SCALE, abs=1e-5
        )
        assert float(by[f"RECIPE_arm_taper_dist_ua_{side}"].radius_m) == pytest.approx(  # type: ignore[arg-type]
            ua_mid * UA_DIST_SHAFT_SCALE, abs=1e-5
        )
        assert float(by[f"RECIPE_limb_forearm_{side}"].radius_m) == pytest.approx(  # type: ignore[arg-type]
            fa_mid * FA_PROX_SHAFT_SCALE, abs=1e-5
        )
        assert float(by[f"RECIPE_arm_taper_dist_fa_{side}"].radius_m) == pytest.approx(  # type: ignore[arg-type]
            fa_mid * FA_DIST_SHAFT_SCALE, abs=1e-5
        )
        assert float(by[f"RECIPE_limb_thigh_{side}"].radius_m) == pytest.approx(  # type: ignore[arg-type]
            thigh_mid * THIGH_PROX_SHAFT_SCALE, abs=1e-5
        )
        assert float(by[f"RECIPE_thigh_taper_dist_{side}"].radius_m) == pytest.approx(  # type: ignore[arg-type]
            thigh_mid * THIGH_DIST_SHAFT_SCALE, abs=1e-5
        )
