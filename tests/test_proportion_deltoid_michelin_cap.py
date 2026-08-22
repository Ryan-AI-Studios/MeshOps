"""Track 0119 — deltoid Michelin cap after tall rz (aniso then uniform).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 47 stay. Not mesh/print success.
Does not reopen 0103 ry 0.62 / rz 1.08 / t 0.36, 0046 scale 1.35,
0060 outer 0.08, 0083 Y=0, 0081 knee, or generic profile Michelin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    _MICHELIN_FRAC,
    COMPACT_CULL_ROLES,
    DELT_ARM_RADIUS_SCALE,
    DELT_DISTAL_BURY_T,
    DELT_OUTER_X_FRAC,
    DELT_RY_FRAC,
    DELT_RZ_FRAC,
    RECIPE_SCHEMA_VERSION,
    _michelin_cap_aniso_axes,
    build_blockout_recipe,
)
from meshops.proportion.skeleton import build_blockout_skeleton
from test_proportion_deltoid_socket import _limb_mass_report
from test_proportion_torso_anti_tire_plus import (
    _product_class_report,
    _product_flags,
)

_REPO = Path(__file__).resolve().parents[1]
_CLI_PY = _REPO / "src" / "meshops" / "cli.py"

_PRODUCT_RX = 0.0591
_PRODUCT_RY = 0.0367
_PRODUCT_RZ = 0.0638
_PRODUCT_CAP = 0.045 * 1.72  # female pack; does not bind vs live rz


def _product_pkg(**flag_overrides: object):
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    return build_blockout_recipe(
        report,
        skeleton=skel,
        **_product_flags(**flag_overrides),  # type: ignore[arg-type]
    )


def test_t0_const_hold() -> None:
    """T0: 0103 DELT axes + bury; keep 0046 scale and 0060 outer."""
    assert DELT_RY_FRAC == 0.62
    assert DELT_RZ_FRAC == 1.08
    assert DELT_DISTAL_BURY_T == 0.36
    assert DELT_ARM_RADIUS_SCALE == 1.35
    assert DELT_OUTER_X_FRAC == 0.08


def test_t1_helper_identity_when_max_le_cap() -> None:
    """T1: helper identity when max(rx, ry, rz) <= cap (product-class numbers)."""
    rx, ry, rz = _PRODUCT_RX, _PRODUCT_RY, _PRODUCT_RZ
    out_rx, out_ry, out_rz, clamped = _michelin_cap_aniso_axes(rx, ry, rz, _PRODUCT_CAP)
    assert clamped is False
    assert (out_rx, out_ry, out_rz) == (rx, ry, rz)
    assert max(rx, ry, rz) <= _PRODUCT_CAP


def test_t2_helper_uniform_when_rz_gt_cap_gt_rx() -> None:
    """T2: helper uniform-scale: max==cap, rz/rx==1.08, ry/rx==0.62."""
    rx = _PRODUCT_RX
    ry = rx * DELT_RY_FRAC
    rz = rx * DELT_RZ_FRAC
    cap = 0.061  # rz > cap > rx (product-class rz ~0.0638)
    assert rz > cap > rx
    out_rx, out_ry, out_rz, clamped = _michelin_cap_aniso_axes(rx, ry, rz, cap)
    assert clamped is True
    assert max(out_rx, out_ry, out_rz) == pytest.approx(cap, abs=1e-12)
    assert out_rz == pytest.approx(out_rx * DELT_RZ_FRAC, abs=1e-12)
    assert out_ry == pytest.approx(out_rx * DELT_RY_FRAC, abs=1e-12)


def test_t3_helper_none_nonfinite_nonpositive_identity() -> None:
    """T3: cap None / <=0 / non-finite → identity (no ZeroDivision / isfinite(None))."""
    rx, ry, rz = _PRODUCT_RX, _PRODUCT_RY, _PRODUCT_RZ
    for cap in (None, 0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        out_rx, out_ry, out_rz, clamped = _michelin_cap_aniso_axes(rx, ry, rz, cap)
        assert clamped is False
        assert (out_rx, out_ry, out_rz) == (rx, ry, rz)


def test_t4_base_fat_arm_max_and_ratio() -> None:
    """T4: base fat-arm: max<=0.45*0.18 AND rz/rx==1.08; B15 clamp-to is helper rx."""
    shoulder_x = 0.18
    report = _limb_mass_report(arm_hw=0.20, shoulder_x=shoulder_x)
    pkg = build_blockout_recipe(report, limbs=False)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    cap = _MICHELIN_FRAC * shoulder_x
    helper_rx = cap / DELT_RZ_FRAC  # 0.081/1.08 = 0.075 (not the cap)
    for d in delts:
        assert d.rx_m is not None and d.ry_m is not None and d.rz_m is not None
        rx = float(d.rx_m)
        ry = float(d.ry_m)
        rz = float(d.rz_m)
        assert max(rx, ry, rz) <= cap + 1e-9
        assert rz == pytest.approx(rx * DELT_RZ_FRAC, abs=1e-9)
        assert ry == pytest.approx(rx * DELT_RY_FRAC, abs=1e-9)
        assert rx == pytest.approx(helper_rx, abs=1e-9)
        assert d.center is not None
        expected_abs = shoulder_x + DELT_OUTER_X_FRAC * rx
        assert abs(float(d.center[0])) == pytest.approx(expected_abs, abs=1e-9)
    assert any("Michelin guard" in m and "clamped to 0.075m" in m for m in pkg.messages)
    assert not any("clamped to 0.081m" in m for m in pkg.messages)


def test_t5_profile_female_fat_arm_not_sphere() -> None:
    """T5: profile female fat-arm: not sphere; rz/rx==1.08; max<=0.045*H."""
    report = _limb_mass_report(arm_hw=0.20)
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(report, limbs=False, profile=profile)
    h = report.height_m or 1.72
    cap = 0.045 * h
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    for d in delts:
        assert d.rx_m is not None and d.ry_m is not None and d.rz_m is not None
        rx = float(d.rx_m)
        rz = float(d.rz_m)
        assert rx != rz
        assert rz == pytest.approx(rx * DELT_RZ_FRAC, abs=1e-9)
        assert max(rx, float(d.ry_m), rz) <= cap + 1e-9
    assert any("michelin_cap_frac_h" in m for m in pkg.messages)


def test_t6_product_class_unclamped_meters() -> None:
    """T6: product-class unclamped ~0.0591/0.0367/0.0638; ratios 0.62/1.08; Y 0.0."""
    pkg = _product_pkg()
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    for d in delts:
        assert d.rx_m is not None and d.ry_m is not None and d.rz_m is not None
        rx = float(d.rx_m)
        ry = float(d.ry_m)
        rz = float(d.rz_m)
        assert rx == pytest.approx(_PRODUCT_RX, abs=5e-4)
        assert ry == pytest.approx(_PRODUCT_RY, abs=5e-4)
        assert rz == pytest.approx(_PRODUCT_RZ, abs=5e-4)
        assert ry == pytest.approx(rx * DELT_RY_FRAC, abs=1e-6)
        assert rz == pytest.approx(rx * DELT_RZ_FRAC, abs=1e-6)
        assert max(rx, ry, rz) < _PRODUCT_CAP
        assert d.center is not None
        assert float(d.center[1]) == pytest.approx(0.0, abs=1e-3)


def test_t7_mcp47_schema_140() -> None:
    """T7: MCP catalog 47; recipe schema 1.4.0."""
    assert len(TOOL_NAMES) == 47
    assert RECIPE_SCHEMA_VERSION == "1.4.0"


def test_t8_helper_private_no_cli() -> None:
    """T8: helper not in __all__; no blockout-delt-michelin / def delt_michelin CLI."""
    from meshops.proportion import blockout_recipe as br

    assert "_michelin_cap_aniso_axes" not in br.__all__
    cli = _CLI_PY.read_text(encoding="utf-8")
    assert "blockout-delt-michelin" not in cli
    assert "def delt_michelin" not in cli


def test_t9_t2_class_unclamped_ratios() -> None:
    """T9: 0103 T2-class (limbs=False, arm_hw 0.04) still unclamped ratios."""
    arm_hw = 0.04
    report = _limb_mass_report(arm_hw=arm_hw)
    pkg = build_blockout_recipe(report, limbs=False)
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
    expected_rx = arm_hw * DELT_ARM_RADIUS_SCALE
    for d in delts:
        assert d.rx_m is not None and d.ry_m is not None and d.rz_m is not None
        rx = float(d.rx_m)
        assert rx == pytest.approx(expected_rx, abs=1e-9)
        assert d.ry_m == pytest.approx(rx * DELT_RY_FRAC, abs=1e-9)
        assert d.rz_m == pytest.approx(rx * DELT_RZ_FRAC, abs=1e-9)


def test_t10_compact_still_emits_deltoid_soft() -> None:
    """T10: compact soft_density still emits both deltoid_soft."""
    assert "deltoid_soft" not in COMPACT_CULL_ROLES
    pkg = _product_pkg(soft_density="compact")
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
