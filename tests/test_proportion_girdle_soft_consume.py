"""Track 0128 — soft new-id trap Y/Z (no 0061/0059/0050/0086/0103 retune).

Does not weaken 0061 T* / 0059 neck / 0086 nape / 0103 delt asserts.
Authoring only — not mesh or print success.
"""

from __future__ import annotations

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    CLAVICLE_LATERAL_INSET_FRAC,
    CLAVICLE_MEDIAL_Z_DROP_FRAC_H,
    CLAVICLE_RADIUS_FRAC_H,
    DELT_DISTAL_BURY_T,
    DELT_RY_FRAC,
    DELT_RZ_FRAC,
    NECK_FORWARD_TILT_DEG,
    NECK_NAPE_CLEARANCE_M,
    NECK_NAPE_SETBACK_M,
    NECK_R_MAX_FRAC_HEAD_RX,
    SCM_R_FRAC_NECK_R,
    TRAP_LAT_FRAC,
    TRAP_NAPE_Z_BIAS_FRAC_H,
    TRAP_RX_FLOOR_FRAC_H,
    TRAP_RY_FLOOR_FRAC_H,
    TRAP_RZ_FLOOR_FRAC_H,
    TRAP_Y_BACK_FRAC_RY,
    build_blockout_recipe,
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


def test_e1_measured_trap_y_after_girdle_call() -> None:
    """E1: measured trap_apex_l Y → trap_soft_l.center[1] tracks measured after L4738.

    B32/B33: write as next statement after girdle call (after L1814-1815).
    Negative Y catches erroneous abs().
    """
    report = _product_class_report()
    measured_y = -0.04
    report.landmarks_xyz["trap_apex_l"] = _lm("trap_apex_l", x_m=-0.14, y_m=measured_y, z_m=1.40)
    pkg = _emit(report)
    trap = next(p for p in pkg.parts if p.name == "RECIPE_trap_soft_l")
    assert trap.center is not None
    assert float(trap.center[1]) == pytest.approx(measured_y, abs=1e-6)
    assert float(trap.center[1]) < 0.0
    assert any("measured trap y=" in m for m in pkg.messages)


def test_e2_absent_ids_prior_path() -> None:
    """E2: absent trap_apex ids → trap Y/Z match 0061 prior path."""
    report = _product_class_report()
    pkg = _emit(report)
    trap = next(p for p in pkg.parts if p.name == "RECIPE_trap_soft_l")
    assert trap.center is not None
    prior_y = float(trap.center[1])
    prior_z = float(trap.center[2])
    assert prior_y != pytest.approx(-0.04, abs=1e-3)
    assert prior_z != pytest.approx(1.35, abs=1e-3)
    assert not any("measured trap y=" in m for m in pkg.messages)
    assert not any("measured trap z=" in m for m in pkg.messages)


def test_e3_const_hold() -> None:
    """E3: 0061/0059/0086/0103 hold — B13 exact-equality."""
    assert TRAP_Y_BACK_FRAC_RY == 0.4
    assert TRAP_LAT_FRAC == 0.55
    assert TRAP_NAPE_Z_BIAS_FRAC_H == 0.010
    assert TRAP_RX_FLOOR_FRAC_H == 0.042
    assert TRAP_RY_FLOOR_FRAC_H == 0.022
    assert TRAP_RZ_FLOOR_FRAC_H == 0.038
    assert CLAVICLE_RADIUS_FRAC_H == 0.012
    assert CLAVICLE_MEDIAL_Z_DROP_FRAC_H == 0.025
    assert CLAVICLE_LATERAL_INSET_FRAC == 0.06
    assert NECK_NAPE_CLEARANCE_M == 0.005
    assert NECK_NAPE_SETBACK_M == 0.018
    assert NECK_R_MAX_FRAC_HEAD_RX == 0.40
    assert NECK_FORWARD_TILT_DEG == 12.0
    assert SCM_R_FRAC_NECK_R == 0.38
    assert DELT_RY_FRAC == 0.62
    assert DELT_RZ_FRAC == 1.08
    assert DELT_DISTAL_BURY_T == 0.36


def test_e4_front_only_clav_med_no_p1_y() -> None:
    """E4: front-only measured clav_med → no Y write on clavicle p1."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["clav_med_l"] = _lm("clav_med_l", x_m=0.0, z_m=1.34)
    moved = _emit(report)
    base_c = next(p for p in baseline.parts if p.name == "RECIPE_clavicle_l")
    new_c = next(p for p in moved.parts if p.name == "RECIPE_clavicle_l")
    assert base_c.p1 is not None and new_c.p1 is not None
    assert float(new_c.p1[1]) == pytest.approx(float(base_c.p1[1]), abs=1e-9)


def test_e5_have_ids_do_not_change_trap_or_neck() -> None:
    """E5: shoulder_l/neck finite must not change TRAP_* / NECK_R_MAX emit vs absent.

    Product-class reports already carry shoulder_l (X/Z). Do not retarget that X
    (0061 trap lat may read it). Adding HAVE ``neck`` + finite shoulder Y must
    not fire 0128 trap overlay.
    """
    report = _product_class_report()
    baseline = _emit(report)
    # HAVE ids: product report already has shoulder_l. Add neck X only — do not
    # set neck z_m (0059/0061 already read neck Z for nape clamp).
    report.landmarks_xyz["neck"] = _lm("neck", x_m=0.0)
    moved = _emit(report)
    assert TRAP_Y_BACK_FRAC_RY == 0.4
    assert NECK_R_MAX_FRAC_HEAD_RX == 0.40
    base_t = next(p for p in baseline.parts if p.name == "RECIPE_trap_soft_l")
    new_t = next(p for p in moved.parts if p.name == "RECIPE_trap_soft_l")
    assert base_t.center is not None and new_t.center is not None
    assert float(new_t.center[1]) == pytest.approx(float(base_t.center[1]), abs=1e-6)
    assert float(new_t.center[2]) == pytest.approx(float(base_t.center[2]), abs=1e-6)
    assert not any("measured trap y=" in m for m in moved.messages)
    assert not any("measured trap z=" in m for m in moved.messages)
    base_n = next(p for p in baseline.parts if p.name == "RECIPE_neck")
    new_n = next(p for p in moved.parts if p.name == "RECIPE_neck")
    assert float(new_n.radius_m or 0.0) == pytest.approx(float(base_n.radius_m or 0.0), abs=1e-9)


def test_e6_measured_clav_med_y_does_not_change_p1() -> None:
    """E6: measured clav_med_l Y finite must not change clavicle p1 vs 0061 shelf."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["clav_med_l"] = _lm("clav_med_l", x_m=0.0, y_m=-0.12, z_m=1.34)
    moved = _emit(report)
    base_c = next(p for p in baseline.parts if p.name == "RECIPE_clavicle_l")
    new_c = next(p for p in moved.parts if p.name == "RECIPE_clavicle_l")
    assert base_c.p1 is not None and new_c.p1 is not None
    assert float(new_c.p1[1]) == pytest.approx(float(base_c.p1[1]), abs=1e-9)
    assert float(new_c.p1[0]) == pytest.approx(float(base_c.p1[0]), abs=1e-9)


def test_e7_one_side_measured_y_after_girdle() -> None:
    """E7: measured Y on one side different from contralateral prior still tracks after L4738."""
    report = _product_class_report()
    baseline = _emit(report)
    base_r = next(p for p in baseline.parts if p.name == "RECIPE_trap_soft_r")
    assert base_r.center is not None
    prior_r = float(base_r.center[1])
    measured_y = 0.08
    assert measured_y != pytest.approx(prior_r, abs=1e-3)
    report.landmarks_xyz["trap_apex_l"] = _lm("trap_apex_l", x_m=-0.14, y_m=measured_y, z_m=1.40)
    pkg = _emit(report)
    trap_l = next(p for p in pkg.parts if p.name == "RECIPE_trap_soft_l")
    trap_r = next(p for p in pkg.parts if p.name == "RECIPE_trap_soft_r")
    assert trap_l.center is not None and trap_r.center is not None
    assert float(trap_l.center[1]) == pytest.approx(measured_y, abs=1e-6)
    assert float(trap_r.center[1]) == pytest.approx(prior_r, abs=1e-6)


def test_e8_measured_trap_z_after_nape() -> None:
    """E8: measured trap_apex_l Z -> trap_soft_l.center[2] tracks after Z nape L1804-1806."""
    report = _product_class_report()
    measured_z = 1.35
    report.landmarks_xyz["trap_apex_l"] = _lm("trap_apex_l", x_m=-0.14, y_m=0.02, z_m=measured_z)
    pkg = _emit(report)
    trap = next(p for p in pkg.parts if p.name == "RECIPE_trap_soft_l")
    assert trap.center is not None
    assert float(trap.center[2]) == pytest.approx(measured_z, abs=1e-6)
    assert any("measured trap z=" in m for m in pkg.messages)


def test_e_mcp_catalog_53() -> None:
    assert len(TOOL_NAMES) == 57
    assert "mesh_proportion_blockout_girdle_compare" in TOOL_NAMES
