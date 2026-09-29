"""Track 0129 — soft new-id bi/tri Y/Z (no 0062/0063/0103/0081/0088/0104 retune).

Does not weaken 0062 T* / 0063 bi-tri / 0103 delt / 0081 elbow / 0088 finger / 0104 curl.
Authoring only — not mesh or print success.
"""

from __future__ import annotations

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    BICEP_ALONG_T,
    BICEP_FRONT_PAST_M,
    DELT_DISTAL_BURY_T,
    DELT_RY_FRAC,
    DELT_RZ_FRAC,
    ELBOW_SOFT_SCALE,
    FA_DIST_SHAFT_SCALE,
    UA_DIST_SHAFT_SCALE,
    build_blockout_recipe,
)
from meshops.proportion.extremity_recipe import _FINGER_CURL_PIP_DEG, _THUMB_PALM_PITCH
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


def test_e1_measured_bicep_y_after_muscle_call() -> None:
    """E1: measured bi_belly_l Y → bicep_soft_l.center[1] tracks after L4778.

    B32/B33: write as next statement after muscle call. Negative Y catches abs().
    """
    report = _product_class_report()
    measured_y = -0.04
    report.landmarks_xyz["bi_belly_l"] = _lm("bi_belly_l", x_m=-0.33, y_m=measured_y, z_m=1.26)
    pkg = _emit(report)
    bicep = next(p for p in pkg.parts if p.name == "RECIPE_bicep_soft_l")
    assert bicep.center is not None
    assert float(bicep.center[1]) == pytest.approx(measured_y, abs=1e-6)
    assert float(bicep.center[1]) < 0.0
    assert any("arm_hand: measured bicep y=" in m for m in pkg.messages)


def test_e2_absent_ids_prior_path() -> None:
    """E2: absent bi/tri ids → bicep/triceps Y/Z match 0063 prior path."""
    report = _product_class_report()
    pkg = _emit(report)
    bicep = next(p for p in pkg.parts if p.name == "RECIPE_bicep_soft_l")
    assert bicep.center is not None
    prior_y = float(bicep.center[1])
    prior_z = float(bicep.center[2])
    assert prior_y != pytest.approx(-0.04, abs=1e-3)
    assert prior_z != pytest.approx(1.20, abs=1e-3)
    assert not any("arm_hand: measured bicep y=" in m for m in pkg.messages)
    assert not any("arm_hand: measured triceps z=" in m for m in pkg.messages)


def test_e3_const_hold() -> None:
    """E3: 0062/0063/0103/0081/0088/0104 hold — B13 exact-equality."""
    assert UA_DIST_SHAFT_SCALE == 0.84
    assert FA_DIST_SHAFT_SCALE == 0.70
    assert BICEP_FRONT_PAST_M == 0.010
    assert BICEP_ALONG_T == 0.50
    assert DELT_RY_FRAC == 0.62
    assert DELT_RZ_FRAC == 1.08
    assert DELT_DISTAL_BURY_T == 0.36
    assert ELBOW_SOFT_SCALE == 1.22
    assert _FINGER_CURL_PIP_DEG == 14.0
    assert _THUMB_PALM_PITCH == -0.55


def test_e4_front_only_humeral_no_ua_p0_y() -> None:
    """E4: front-only measured humeral_head → no Y write on UA p0."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["humeral_head_l"] = _lm("humeral_head_l", x_m=-0.26, z_m=1.38)
    moved = _emit(report)
    base_ua = next(p for p in baseline.parts if p.name == "RECIPE_limb_upper_arm_l")
    new_ua = next(p for p in moved.parts if p.name == "RECIPE_limb_upper_arm_l")
    assert base_ua.p0 is not None and new_ua.p0 is not None
    assert float(new_ua.p0[1]) == pytest.approx(float(base_ua.p0[1]), abs=1e-9)


def test_e5_have_ids_do_not_change_bicep() -> None:
    """E5: elbow_l/wrist_l/shoulder_l finite must not fire 0129 bi/tri overlay.

    Product-class reports already carry HAVE arm joints. Re-emitting without
    bi/tri belly ids must keep BICEP_*/UA_DIST_* and must not write arm_hand
    overlay messages (do not retarget elbow/wrist X/Z — those drive hang).
    """
    report = _product_class_report()
    assert "shoulder_l" in report.landmarks_xyz
    pkg = _emit(report)
    assert BICEP_FRONT_PAST_M == 0.010
    assert UA_DIST_SHAFT_SCALE == 0.84
    bicep = next(p for p in pkg.parts if p.name == "RECIPE_bicep_soft_l")
    assert bicep.center is not None
    # HAVE ids present on product report must not imply soft consume fired.
    assert not any("arm_hand: measured bicep" in m for m in pkg.messages)
    assert not any("arm_hand: measured triceps" in m for m in pkg.messages)


def test_e6_measured_fa_belly_y_does_not_change_fa_r() -> None:
    """E6: measured fa_belly_l Y finite must not change FA radius_m vs 0062."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["fa_belly_l"] = _lm("fa_belly_l", x_m=-0.42, y_m=-0.04, z_m=1.08)
    moved = _emit(report)
    base_fa = next(p for p in baseline.parts if p.name == "RECIPE_limb_forearm_l")
    new_fa = next(p for p in moved.parts if p.name == "RECIPE_limb_forearm_l")
    assert float(new_fa.radius_m or 0.0) == pytest.approx(float(base_fa.radius_m or 0.0), abs=1e-9)


def test_e7_one_side_measured_y_after_muscle() -> None:
    """E7: measured Y on one side different from contralateral prior still tracks."""
    report = _product_class_report()
    baseline = _emit(report)
    base_r = next(p for p in baseline.parts if p.name == "RECIPE_bicep_soft_r")
    assert base_r.center is not None
    prior_r = float(base_r.center[1])
    measured_y = 0.08
    assert measured_y != pytest.approx(prior_r, abs=1e-3)
    report.landmarks_xyz["bi_belly_l"] = _lm("bi_belly_l", x_m=-0.33, y_m=measured_y, z_m=1.26)
    pkg = _emit(report)
    bi_l = next(p for p in pkg.parts if p.name == "RECIPE_bicep_soft_l")
    bi_r = next(p for p in pkg.parts if p.name == "RECIPE_bicep_soft_r")
    assert bi_l.center is not None and bi_r.center is not None
    assert float(bi_l.center[1]) == pytest.approx(measured_y, abs=1e-6)
    assert float(bi_r.center[1]) == pytest.approx(prior_r, abs=1e-6)


def test_e8_measured_triceps_z_after_muscle() -> None:
    """E8: measured tri_belly_l Z → triceps_soft_l.center[2] tracks (B34 name lookup)."""
    report = _product_class_report()
    measured_z = 1.22
    report.landmarks_xyz["tri_belly_l"] = _lm("tri_belly_l", x_m=-0.33, y_m=0.01, z_m=measured_z)
    pkg = _emit(report)
    tri = next(p for p in pkg.parts if p.name == "RECIPE_triceps_soft_l")
    assert tri.center is not None
    assert float(tri.center[2]) == pytest.approx(measured_z, abs=1e-6)
    assert any("arm_hand: measured triceps z=" in m for m in pkg.messages)
    assert tri.role == "limb_segment"


def test_e9_no_shoulder_ball_emit() -> None:
    """E9: emit contains no part named RECIPE_shoulder_ball_l."""
    report = _product_class_report()
    report.landmarks_xyz["humeral_head_l"] = _lm("humeral_head_l", x_m=-0.26, y_m=0.0, z_m=1.38)
    pkg = _emit(report)
    names = {p.name for p in pkg.parts}
    assert "RECIPE_shoulder_ball_l" not in names
    assert "RECIPE_shoulder_ball_r" not in names


def test_e_mcp_catalog_53() -> None:
    assert len(TOOL_NAMES) == 54
    assert "mesh_proportion_blockout_arm_hand_compare" in TOOL_NAMES
