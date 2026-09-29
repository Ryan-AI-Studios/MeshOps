"""Track 0127 — soft new-id calf_cyl p0 Y + arch Z (no 0108/0096/0071/0042/0056 retune).

Does not weaken 0108 T* / 0096 calf / 0042 C_* asserts.
Authoring only — not mesh or print success.
"""

from __future__ import annotations

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    CALF_BELLY_LAT_FRAC,
    CALF_BELLY_REAR_FRAC,
    CALF_BELLY_SCALE,
    CALF_DIST_SHAFT_SCALE,
    CALF_SPLIT_T,
    _apply_measured_calf_cyl_y,
    build_blockout_recipe,
)
from meshops.proportion.extremity_recipe import (
    ANK_RY_FRAC_HALF_W,
    ANK_RZ_FRAC_HALF_W,
    ARCH_SOFT_RY_FRAC_HALF_DEPTH,
    BALL_SOFT_RY_FRAC_HALF_DEPTH,
    TOE_BALL_NEST_FRAC,
    TOE_TIP_PAD_RY_FRAC,
)
from meshops.proportion.models import LandmarkXYZ
from meshops.proportion.skeleton import build_blockout_skeleton
from test_proportion_torso_anti_tire_plus import (
    _product_class_report,
    _product_flags,
    _template,
)


def _lm(
    lid: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=lid, x_m=x_m, y_m=y_m, z_m=z_m)


def _emit(report, **flag_overrides: object):
    skel = build_blockout_skeleton(report)
    return build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(**flag_overrides),  # type: ignore[arg-type]
    )


def test_t6_no_taper_measured_y_keeps_p1_ankle() -> None:
    """T6 / B3: no-taper measured Y updates p0 only; p1 stays ankle, not 42% stub."""
    report = _product_class_report()
    pkg = _emit(report)
    taper = next(p for p in pkg.parts if p.name == "RECIPE_calf_taper_dist_l")
    assert taper.p1 is not None
    ank_y = float(taper.p1[1])
    parts = [p for p in pkg.parts if not p.name.startswith("RECIPE_calf_taper_dist_")]
    cyl = next(p for p in parts if p.name == "RECIPE_calf_cyl_l")
    assert cyl.p0 is not None and cyl.p1 is not None
    cyl.p1 = [float(cyl.p1[0]), ank_y, float(cyl.p1[2])]
    measured_y = -0.04
    report.landmarks_xyz["gastroc_med_l"] = _lm(
        "gastroc_med_l", x_m=-0.10, y_m=measured_y, z_m=0.57
    )
    messages: list[str] = []
    _apply_measured_calf_cyl_y(parts, report, messages)
    assert float(cyl.p0[1]) == pytest.approx(measured_y, abs=1e-6)
    assert float(cyl.p0[1]) < 0.0
    assert float(cyl.p1[1]) == pytest.approx(ank_y, abs=1e-6)
    stub_y = measured_y + CALF_SPLIT_T * (ank_y - measured_y)
    assert float(cyl.p1[1]) != pytest.approx(stub_y, abs=1e-4)
    assert any("measured calf_cyl y=" in m for m in messages)


def test_e1_measured_gastroc_y_after_b6() -> None:
    """E1: measured gastroc_med_l Y → calf_cyl_l.p0[1] tracks measured after B6.

    B32/B33: write after B6 call then _calf_split_mid into cyl.p1 and taper.p0.
    Negative Y catches erroneous abs(); mid must sit at CALF_SPLIT_T.
    """
    report = _product_class_report()
    measured_y = -0.04
    report.landmarks_xyz["gastroc_med_l"] = _lm(
        "gastroc_med_l", x_m=-0.10, y_m=measured_y, z_m=0.57
    )
    pkg = _emit(report)
    cyl = next(p for p in pkg.parts if p.name == "RECIPE_calf_cyl_l")
    taper = next(p for p in pkg.parts if p.name == "RECIPE_calf_taper_dist_l")
    assert cyl.p0 is not None and cyl.p1 is not None
    assert taper.p0 is not None and taper.p1 is not None
    assert float(cyl.p0[1]) == pytest.approx(measured_y, abs=1e-6)
    assert float(cyl.p0[1]) < 0.0
    assert float(cyl.p1[1]) == pytest.approx(float(taper.p0[1]), abs=1e-6)
    expected_mid_y = float(cyl.p0[1]) + CALF_SPLIT_T * (float(taper.p1[1]) - float(cyl.p0[1]))
    assert float(cyl.p1[1]) == pytest.approx(expected_mid_y, abs=1e-6)
    assert CALF_SPLIT_T == 0.42
    assert any("measured calf_cyl y=" in m for m in pkg.messages)


def test_e2_absent_ids_prior_path() -> None:
    """E2: absent gastroc/arch_apex ids → calf_cyl p0 Y / arch Z match 0096/0108 prior."""
    report = _product_class_report()
    pkg = _emit(report)
    cyl = next(p for p in pkg.parts if p.name == "RECIPE_calf_cyl_l")
    arch = next(p for p in pkg.parts if p.name == "RECIPE_arch_soft_l")
    assert cyl.p0 is not None
    prior_y = float(cyl.p0[1])
    assert prior_y != pytest.approx(0.12, abs=1e-3)
    assert arch.center is not None
    prior_z = float(arch.center[2])
    assert prior_z != pytest.approx(0.09, abs=1e-3)
    assert not any("measured calf_cyl y=" in m for m in pkg.messages)
    assert not any("measured arch z=" in m for m in pkg.messages)


def test_e3_const_hold() -> None:
    """E3: 0108/0096/0071 hold — B13 exact-equality."""
    assert ANK_RY_FRAC_HALF_W == 0.78
    assert TOE_BALL_NEST_FRAC == 0.52
    assert TOE_TIP_PAD_RY_FRAC == 0.55
    assert ARCH_SOFT_RY_FRAC_HALF_DEPTH == 0.18
    assert BALL_SOFT_RY_FRAC_HALF_DEPTH == 0.16
    assert CALF_BELLY_SCALE == 1.18
    assert CALF_DIST_SHAFT_SCALE == 0.80
    assert CALF_SPLIT_T == 0.42
    assert CALF_BELLY_LAT_FRAC == 0.30
    assert CALF_BELLY_REAR_FRAC == 0.42
    assert ANK_RZ_FRAC_HALF_W == 1.80


def test_e4_front_only_malleolus_no_calf_y() -> None:
    """E4: front-only measured malleolus → no Y write on calf_cyl."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["malleolus_med_l"] = _lm("malleolus_med_l", x_m=-0.10, z_m=0.13)
    moved = _emit(report)
    base_c = next(p for p in baseline.parts if p.name == "RECIPE_calf_cyl_l")
    new_c = next(p for p in moved.parts if p.name == "RECIPE_calf_cyl_l")
    assert base_c.p0 is not None and new_c.p0 is not None
    assert float(new_c.p0[1]) == pytest.approx(float(base_c.p0[1]), abs=1e-9)


def test_e5_have_ids_do_not_change_calf_or_ank() -> None:
    """E5: calf_front/ankle_l/heel_l finite must not change ANK_RY / CALF_BELLY emit."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["calf_front"] = _lm("calf_front", x_m=-0.09, y_m=-0.05, z_m=0.45)
    report.landmarks_xyz["ankle_l"] = _lm("ankle_l", x_m=-0.09, y_m=0.07, z_m=0.13)
    report.landmarks_xyz["heel_l"] = _lm("heel_l", x_m=-0.09, y_m=0.09, z_m=0.05)
    moved = _emit(report)
    assert ANK_RY_FRAC_HALF_W == 0.78
    assert CALF_BELLY_SCALE == 1.18
    base_c = next(p for p in baseline.parts if p.name == "RECIPE_calf_cyl_l")
    new_c = next(p for p in moved.parts if p.name == "RECIPE_calf_cyl_l")
    assert base_c.p0 is not None and new_c.p0 is not None
    assert float(new_c.p0[1]) == pytest.approx(float(base_c.p0[1]), abs=1e-6)
    assert float(new_c.radius_m or 0.0) == pytest.approx(float(base_c.radius_m or 0.0), abs=1e-6)
    base_a = next(p for p in baseline.parts if p.name == "RECIPE_ank_foot_l")
    new_a = next(p for p in moved.parts if p.name == "RECIPE_ank_foot_l")
    assert base_a.rx_m is not None and new_a.rx_m is not None
    assert float(new_a.rx_m) == pytest.approx(float(base_a.rx_m), abs=1e-9)


def test_e6_measured_malleolus_x_does_not_change_ank_rx() -> None:
    """E6: measured malleolus_med_l X must not change ank_foot rx vs C_foot_width."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["malleolus_med_l"] = _lm("malleolus_med_l", x_m=-0.20, y_m=0.07, z_m=0.13)
    moved = _emit(report)
    base_a = next(p for p in baseline.parts if p.name == "RECIPE_ank_foot_l")
    new_a = next(p for p in moved.parts if p.name == "RECIPE_ank_foot_l")
    assert base_a.rx_m is not None and new_a.rx_m is not None
    assert float(new_a.rx_m) == pytest.approx(float(base_a.rx_m), abs=1e-9)
    assert base_a.center is not None and new_a.center is not None
    assert float(new_a.center[0]) == pytest.approx(float(base_a.center[0]), abs=1e-9)


def test_e7_one_side_measured_y_after_b6() -> None:
    """E7: measured Y on one side different from contralateral prior still tracks."""
    report = _product_class_report()
    baseline = _emit(report)
    base_r = next(p for p in baseline.parts if p.name == "RECIPE_calf_cyl_r")
    assert base_r.p0 is not None
    prior_r = float(base_r.p0[1])
    measured_y = 0.18
    assert measured_y != pytest.approx(prior_r, abs=1e-3)
    report.landmarks_xyz["gastroc_med_l"] = _lm(
        "gastroc_med_l", x_m=-0.10, y_m=measured_y, z_m=0.57
    )
    pkg = _emit(report)
    cyl_l = next(p for p in pkg.parts if p.name == "RECIPE_calf_cyl_l")
    cyl_r = next(p for p in pkg.parts if p.name == "RECIPE_calf_cyl_r")
    assert cyl_l.p0 is not None and cyl_r.p0 is not None
    assert float(cyl_l.p0[1]) == pytest.approx(measured_y, abs=1e-6)
    assert float(cyl_r.p0[1]) == pytest.approx(prior_r, abs=1e-6)


def test_e8_measured_arch_z_after_append() -> None:
    """E8: measured arch_apex_l Z → arch_soft_l.center[2] tracks after arch write."""
    report = _product_class_report()
    measured_z = 0.09
    report.landmarks_xyz["arch_apex_l"] = _lm("arch_apex_l", x_m=-0.09, y_m=-0.01, z_m=measured_z)
    pkg = _emit(report)
    arch = next(p for p in pkg.parts if p.name == "RECIPE_arch_soft_l")
    assert arch.center is not None
    assert float(arch.center[2]) == pytest.approx(measured_z, abs=1e-6)
    assert any("measured arch z=" in m for m in pkg.messages)


def test_e_mcp_catalog_51() -> None:
    assert len(TOOL_NAMES) == 54
    assert "mesh_proportion_blockout_leg_foot_compare" in TOOL_NAMES
