"""Track 0125 — soft new-id placement consume (no 0105/0090/0066/0074/0118 retune).

Does not weaken 0105 T4/T5 overlap asserts. Authoring only — not mesh or print success.
"""

from __future__ import annotations

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    BREAST_SIT_CHEST_BURY_M,
    MID_BACK_REAR_PAST_M,
    MID_BACK_Z_BELOW_WAIST_M,
    SCAP_REAR_PAST_M,
    SCAP_RY_FRAC_RX,
    TORSO_OVAL_OVERLAP_FLOOR_M,
    TORSO_OVAL_RY_CHEST_FRAC,
    TORSO_OVAL_Z_NORM_CHEST,
    TORSO_OVAL_Z_NORM_HIP,
    TORSO_OVAL_Z_NORM_WAIST,
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


def test_e1_measured_scap_y_survives_abs() -> None:
    """E1: measured scap_inferior_l Y → emitted scap center[1] tracks measured.

    B32: measured value must survive abs(cy) at L1912.
    """
    report = _product_class_report()
    measured_y = 0.18
    report.landmarks_xyz["scap_inferior_l"] = _lm(
        "scap_inferior_l", x_m=-0.12, y_m=measured_y, z_m=1.28
    )
    pkg = _emit(report)
    scap_l = next(p for p in pkg.parts if p.name == "RECIPE_scap_soft_l")
    assert scap_l.center is not None
    assert float(scap_l.center[1]) == pytest.approx(measured_y, abs=1e-6)
    assert any("measured scap y=" in m for m in pkg.messages)


def test_e2_absent_scap_ids_prior_path() -> None:
    """E2: absent scap ids → scap cy matches 0066 prior path."""
    report = _product_class_report()
    pkg = _emit(report)
    scap_l = next(p for p in pkg.parts if p.name == "RECIPE_scap_soft_l")
    assert scap_l.center is not None
    prior_y = float(scap_l.center[1])
    assert prior_y != pytest.approx(0.18, abs=1e-3)
    assert not any("measured scap y=" in m for m in pkg.messages)


def test_e3_const_hold() -> None:
    """E3: 0105/0090/0066/0074/0118 hold — B31 chest ry is _FRAC."""
    assert TORSO_OVAL_Z_NORM_CHEST == 0.20
    assert TORSO_OVAL_Z_NORM_WAIST == 0.50
    assert TORSO_OVAL_Z_NORM_HIP == 0.78
    assert TORSO_OVAL_OVERLAP_FLOOR_M == 0.080
    assert TORSO_OVAL_RY_CHEST_FRAC == 0.72
    assert SCAP_REAR_PAST_M == 0.012
    assert SCAP_RY_FRAC_RX == 0.42
    assert MID_BACK_REAR_PAST_M == 0.032
    assert MID_BACK_Z_BELOW_WAIST_M == 0.035
    assert BREAST_SIT_CHEST_BURY_M == 0.004


def test_e4_front_only_sternum_no_scap_y() -> None:
    """E4: front-only measured sternum → no Y write on scap/mid_back."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["sternum_mid"] = _lm("sternum_mid", x_m=0.0, z_m=1.28)
    moved = _emit(report)
    base_scap = next(p for p in baseline.parts if p.name == "RECIPE_scap_soft_l")
    new_scap = next(p for p in moved.parts if p.name == "RECIPE_scap_soft_l")
    assert base_scap.center is not None and new_scap.center is not None
    assert float(new_scap.center[1]) == pytest.approx(float(base_scap.center[1]), abs=1e-9)
    base_mb = next(p for p in baseline.parts if p.name == "RECIPE_mid_back_soft_l")
    new_mb = next(p for p in moved.parts if p.name == "RECIPE_mid_back_soft_l")
    assert base_mb.center is not None and new_mb.center is not None
    assert float(new_mb.center[1]) == pytest.approx(float(base_mb.center[1]), abs=1e-9)


def test_e5_depth_pairs_do_not_change_bury() -> None:
    """E5: existing chest_front/breast_back finite must not change 0118 bury emit."""
    report = _product_class_report()
    baseline = _emit(report)
    report.landmarks_xyz["chest_front"] = _lm("chest_front", x_m=0.0, y_m=-0.20, z_m=1.25)
    report.landmarks_xyz["breast_back"] = _lm("breast_back", x_m=0.0, y_m=0.05, z_m=1.22)
    moved = _emit(report)
    assert BREAST_SIT_CHEST_BURY_M == 0.004
    base_b = next(p for p in baseline.parts if p.name == "RECIPE_breast_soft_l")
    new_b = next(p for p in moved.parts if p.name == "RECIPE_breast_soft_l")
    assert base_b.center is not None and new_b.center is not None
    assert float(new_b.center[1]) == pytest.approx(float(base_b.center[1]), abs=1e-6)


def test_e_left_y_consumed_on_mid_back() -> None:
    """Left-finite mid_back Y moves plate Y off the 0074 prior."""
    report = _product_class_report()
    measured_y = 0.16
    report.landmarks_xyz["mid_back_l"] = _lm("mid_back_l", x_m=-0.12, y_m=measured_y, z_m=1.10)
    pkg = _emit(report)
    mid = next(p for p in pkg.parts if p.name == "RECIPE_mid_back_soft_l")
    assert mid.center is not None
    assert float(mid.center[1]) == pytest.approx(measured_y, abs=1e-6)
    assert any("measured mid_back y=" in m for m in pkg.messages)


def test_e_mcp_catalog_49() -> None:
    assert len(TOOL_NAMES) == 49
