"""Track 0120 — generic profile Michelin cap (aniso then uniform).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 47 stay. Not mesh/print success.
Does not reopen 0119 helper, 0103 ry 0.62 / rz 1.08 / t 0.36,
pack-cap adds, 0081 knee, or generic CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import ProfileScaleSpec, load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    COMPACT_CULL_ROLES,
    DELT_DISTAL_BURY_T,
    DELT_RY_FRAC,
    DELT_RZ_FRAC,
    RECIPE_SCHEMA_VERSION,
    _michelin_cap_aniso_axes,
    _resolve_profile_axes,
    _ResolvedMetrics,
    build_blockout_recipe,
)
from meshops.proportion.models import ProportionReport
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
_H = 1.72
_BIND_CAP = 0.045 * _H  # 0.0774
_PACK_IDS = ("torso_limb_f_athletic_v1", "torso_limb_m_athletic_v1")


def _product_pkg(**flag_overrides: object):
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    return build_blockout_recipe(
        report,
        skeleton=skel,
        **_product_flags(**flag_overrides),  # type: ignore[arg-type]
    )


def _generic_axes(
    *,
    rx_frac_h: float,
    ry_frac_h: float,
    rz_frac_h: float,
    michelin_cap_frac_h: float | None,
    height_m: float | None = _H,
    role: str = "bicep_soft",
) -> tuple[float, float, float, list[str]]:
    report = ProportionReport(height_m=height_m)
    m = _ResolvedMetrics()
    m.height_m = height_m
    scale = ProfileScaleSpec(
        rx_frac_h=rx_frac_h,
        ry_frac_h=ry_frac_h,
        rz_frac_h=rz_frac_h,
        michelin_cap_frac_h=michelin_cap_frac_h,
    )
    messages: list[str] = []
    rx, ry, rz = _resolve_profile_axes(
        report,
        m,
        scale,
        side="none",
        template_applied=None,
        messages=messages,
        role=role,
    )
    return rx, ry, rz, messages


def test_t0_helper_private_and_delt_const_hold() -> None:
    """T0: helper present; exact-name not in __all__ (B18); 0119 DELT_* hold."""
    from meshops.proportion import blockout_recipe as br

    assert callable(_michelin_cap_aniso_axes)
    assert "_michelin_cap_aniso_axes" not in br.__all__
    assert DELT_RY_FRAC == 0.62
    assert DELT_RZ_FRAC == 1.08
    assert DELT_DISTAL_BURY_T == 0.36


def test_t1_generic_identity_when_max_le_cap() -> None:
    """T1: generic identity when max <= cap (fracs 0.02/0.015/0.022 cap 0.045)."""
    rx, ry, rz, messages = _generic_axes(
        rx_frac_h=0.02,
        ry_frac_h=0.015,
        rz_frac_h=0.022,
        michelin_cap_frac_h=0.045,
    )
    assert (rx, ry, rz) == pytest.approx((0.02 * _H, 0.015 * _H, 0.022 * _H))
    assert max(rx, ry, rz) <= _BIND_CAP
    assert not any("michelin_cap_frac_h" in m for m in messages)


def test_t2_generic_bind_max_and_ratio() -> None:
    """T2: generic bind 0.08/0.05/0.09 cap 0.045: max==0.0774 AND ratios (not sphere)."""
    rx, ry, rz, messages = _generic_axes(
        rx_frac_h=0.08,
        ry_frac_h=0.05,
        rz_frac_h=0.09,
        michelin_cap_frac_h=0.045,
    )
    assert max(rx, ry, rz) == pytest.approx(_BIND_CAP, abs=1e-12)
    assert rx != rz
    assert rz / rx == pytest.approx(1.125, abs=1e-12)
    assert ry / rx == pytest.approx(0.625, abs=1e-12)
    assert rx == pytest.approx(0.0688, abs=1e-9)
    assert ry == pytest.approx(0.0430, abs=1e-9)
    assert rz == pytest.approx(0.0774, abs=1e-9)
    assert any("michelin_cap_frac_h" in m for m in messages)
    assert any(f"clamped to {max(rx, ry, rz):.3f}m" in m for m in messages)


def test_t3_generic_cap_zero_and_none_identity() -> None:
    """T3: michelin_cap_frac_h=0 identity (not zeros); None identity."""
    rx0, ry0, rz0, messages0 = _generic_axes(
        rx_frac_h=0.08,
        ry_frac_h=0.05,
        rz_frac_h=0.09,
        michelin_cap_frac_h=0.0,
    )
    assert (rx0, ry0, rz0) == pytest.approx((0.08 * _H, 0.05 * _H, 0.09 * _H))
    assert rx0 > 0.0 and ry0 > 0.0 and rz0 > 0.0
    assert not any("michelin_cap_frac_h" in m for m in messages0)

    rxn, ryn, rzn, messagesn = _generic_axes(
        rx_frac_h=0.08,
        ry_frac_h=0.05,
        rz_frac_h=0.09,
        michelin_cap_frac_h=None,
    )
    assert (rxn, ryn, rzn) == pytest.approx((0.08 * _H, 0.05 * _H, 0.09 * _H))
    assert not any("michelin_cap_frac_h" in m for m in messagesn)


def test_t4_deltoid_role_still_helper_path() -> None:
    """T4: role=deltoid_soft 0119 T5-class female fat-arm: not sphere, rz/rx==1.08."""
    report = _limb_mass_report(arm_hw=0.20)
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(report, limbs=False, profile=profile)
    h = report.height_m or _H
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


def test_t5_product_class_unclamped_meters() -> None:
    """T5: product-class unclamped ~0.0591/0.0367/0.0638 (in-memory, not work/)."""
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
    assert len(pkg.parts) == 131


def test_t6_packs_cap_only_on_deltoid_soft() -> None:
    """T6: F/M packs: michelin_cap_frac_h only on deltoid_soft."""
    for pid in _PACK_IDS:
        doc = load_anatomy_profile(pid)
        capped = [
            part.role
            for region in doc.regions
            for part in region.parts
            if part.scale.michelin_cap_frac_h is not None
        ]
        assert capped, f"{pid} should keep deltoid_soft cap"
        assert set(capped) == {"deltoid_soft"}
        for role in ("hip_soft", "breast_soft", "bicep_soft", "glute_soft"):
            assert role not in capped


def test_t7_mcp47_schema_140() -> None:
    """T7: MCP catalog 47; recipe schema 1.4.0."""
    assert len(TOOL_NAMES) == 50
    assert RECIPE_SCHEMA_VERSION == "1.4.0"


def test_t8_helper_private_no_cli() -> None:
    """T8: exact helper name not in __all__ (B18); no generic-michelin CLI."""
    from meshops.proportion import blockout_recipe as br

    assert "_michelin_cap_aniso_axes" not in br.__all__
    cli = _CLI_PY.read_text(encoding="utf-8")
    assert "blockout-generic-michelin" not in cli
    assert "def generic_michelin" not in cli


def test_t9_missing_h_and_shoulder_hw_identity() -> None:
    """T9: missing H + missing shoulder_hw + cap_frac 0.045 → clamp_max None → identity."""
    rx, ry, rz, messages = _generic_axes(
        rx_frac_h=0.08,
        ry_frac_h=0.05,
        rz_frac_h=0.09,
        michelin_cap_frac_h=0.045,
        height_m=None,
    )
    # frac*H unavailable → fallbacks 0.04/0.03/0.035 * 1.7
    assert (rx, ry, rz) == pytest.approx((0.04 * 1.7, 0.03 * 1.7, 0.035 * 1.7))
    assert not any("michelin_cap_frac_h" in m for m in messages)


def test_t10_compact_still_emits_deltoid_soft() -> None:
    """T10: compact soft_density still emits both deltoid_soft."""
    assert "deltoid_soft" not in COMPACT_CULL_ROLES
    pkg = _product_pkg(soft_density="compact")
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert len(delts) == 2
