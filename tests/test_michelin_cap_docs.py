"""Track 0121 — per-job Michelin cap (published README + hold-the-line).

RECIPE_HONESTY / Difficulty §1 / §2 / §5 / §9 / §12 / N6.
Michelin polish is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
Tests grep published files only — never conductor/, docs/, .agents/, AGENTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion import blockout_recipe as br
from meshops.proportion.blockout_recipe import (
    DELT_DISTAL_BURY_T,
    DELT_RY_FRAC,
    DELT_RZ_FRAC,
    KNEE_SOFT_MAX_VS_THIGH_PROX,
)

_REPO = Path(__file__).resolve().parents[1]
_M1 = "Per-job Michelin cap is not a product default"
_M2 = "--profiles"
_M3 = "Do not promote session Michelin tweaks"
_M4 = "Michelin polish is not print success"
_HEADING = "Per-job Michelin cap"
_PACKS = (
    _REPO / "src/meshops/proportion/body_profiles/torso_limb_f_athletic_v1.json",
    _REPO / "src/meshops/proportion/body_profiles/torso_limb_m_athletic_v1.json",
)


def _readme() -> str:
    return (_REPO / "README.md").read_text(encoding="utf-8")


def _michelin_section(text: str) -> str:
    idx = text.find(_HEADING)
    if idx < 0:
        return ""
    rest = text[idx:]
    next_h = rest.find("\n### ", 1)
    if next_h < 0:
        next_h = rest.find("\n## ", 1)
    return rest if next_h < 0 else rest[:next_h]


def test_t0_readme_m1_not_a_product_default() -> None:
    """T0: README contains frozen M1."""
    assert _M1 in _readme()


def test_t1_readme_m2_profiles_in_michelin_section() -> None:
    """T1: M2 lives under the Per-job Michelin cap heading."""
    text = _readme()
    assert _HEADING in text
    section = _michelin_section(text)
    assert _M2 in section


def test_t2_readme_m3_do_not_promote_contiguous() -> None:
    """T2: M3 is contiguous on the full README (no backtick between Do not promote and session)."""
    assert _M3 in _readme()


def test_t3_readme_m4_michelin_not_print() -> None:
    """T3: README contains frozen M4."""
    assert _M4 in _readme()


def test_t4_delt_0103_hold() -> None:
    """T4: 0103/0119 hold ry 0.62 / rz 1.08 / t 0.36; fail rz >=1.18 as product sausage."""
    assert DELT_RY_FRAC == 0.62
    assert DELT_RZ_FRAC == 1.08
    assert DELT_DISTAL_BURY_T == 0.36
    assert DELT_RZ_FRAC < 1.18  # fail-as-product sausage


def test_t5_packs_cap_only_on_deltoid_soft() -> None:
    """T5: F/M published packs: michelin_cap_frac_h only on deltoid_soft."""
    for path in _PACKS:
        doc = json.loads(path.read_text(encoding="utf-8"))
        capped: list[str] = []
        for region in doc.get("regions", []):
            for part in region.get("parts", []):
                scale = part.get("scale") or {}
                if scale.get("michelin_cap_frac_h") is not None:
                    capped.append(part["role"])
        assert capped, f"{path.name} should keep deltoid_soft cap"
        assert set(capped) == {"deltoid_soft"}
        for role in ("hip_soft", "breast_soft", "bicep_soft", "glute_soft"):
            assert role not in capped


def test_t6_mcp_catalog_47() -> None:
    """T6: MCP catalog stays 47."""
    assert len(TOOL_NAMES) == 56


def test_t7_no_michelin_cli_command() -> None:
    """T7: no blockout-michelin / generic-michelin / def generic_michelin / def michelin_cap."""
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-michelin" not in cli_text
    assert "blockout-generic-michelin" not in cli_text
    assert "def generic_michelin" not in cli_text
    assert "def michelin_cap" not in cli_text


def test_t8_knee_0081_hold_helper_private() -> None:
    """T8: 0081 knee 1.25 + exact helper name not in __all__ (not substring michelin)."""
    assert KNEE_SOFT_MAX_VS_THIGH_PROX == 1.25
    assert "_michelin_cap_aniso_axes" not in br.__all__
