"""Track 0117 — per-job breast costume/sculpt (published README + hold-the-line).

RECIPE_HONESTY / Difficulty §1 / §2 / §5 / §9 / §12 / N6.
Breast costume polish is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
Tests grep published files only — never conductor/, docs/, .agents/, AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    BREAST_ATHLETIC_RX_MAX_FRAC_H,
    BREAST_HANG_Z_DROP_FRAC_RZ,
    BREAST_SIT_CHEST_BURY_M,
    BREAST_TEAR_RY_FRAC_RX,
    BREAST_TEAR_RZ_FRAC_RX,
)

_REPO = Path(__file__).resolve().parents[1]
_C1 = "Breast costume/sculpt is not a product default"
_C2 = "--breast-tilt-deg"
_C3 = "Do not promote session breast tweaks"
_C4 = "Breast costume polish is not print success"
_HEADING = "Breast costume / sculpt"


def _readme() -> str:
    return (_REPO / "README.md").read_text(encoding="utf-8")


def _breast_section(text: str) -> str:
    idx = text.find(_HEADING)
    if idx < 0:
        return ""
    rest = text[idx:]
    next_h = rest.find("\n### ", 1)
    if next_h < 0:
        next_h = rest.find("\n## ", 1)
    return rest if next_h < 0 else rest[:next_h]


def test_t0_readme_c1_not_a_product_default() -> None:
    """T0: README contains frozen C1."""
    assert _C1 in _readme()


def test_t1_readme_c2_breast_tilt_in_costume_section() -> None:
    """T1: C2 lives under the Breast costume / sculpt heading."""
    text = _readme()
    assert _HEADING in text
    section = _breast_section(text)
    assert _C2 in section


def test_t2_readme_c3_do_not_promote_contiguous() -> None:
    """T2: C3 is contiguous on the full README (no backtick between Do not promote and session)."""
    assert _C3 in _readme()


def test_t3_readme_c4_breast_costume_not_print() -> None:
    """T3: README contains frozen C4."""
    assert _C4 in _readme()


def test_t4_breast_sit_bury_hold() -> None:
    """T4: 0118 bury 0.004 hold; fail >=0.016 as product (0091 proud restore)."""
    assert BREAST_SIT_CHEST_BURY_M == 0.004
    assert BREAST_SIT_CHEST_BURY_M < 0.016  # 0091 proud 0.016 is not product contact
    assert BREAST_SIT_CHEST_BURY_M > 0.0  # negative bury is not product


def test_t5_breast_tear_0067_hold() -> None:
    """T5: 0067 tear hold — ry/rx 0.78 / rz/rx 1.05."""
    assert BREAST_TEAR_RY_FRAC_RX == 0.78
    assert BREAST_TEAR_RZ_FRAC_RX == 1.05


def test_t6_mcp_catalog_47() -> None:
    """T6: MCP catalog stays 47."""
    assert len(TOOL_NAMES) == 47


def test_t7_no_breast_costume_cli_command() -> None:
    """T7: no blockout-breast-costume / blockout-breast-sculpt / def breast_costume command."""
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-breast-costume" not in cli_text
    assert "blockout-breast-sculpt" not in cli_text
    assert "def breast_costume" not in cli_text


def test_t8_hang_and_athletic_hold() -> None:
    """T8: 0049 hang 0.55 + 0067 athletic rx 0.042."""
    assert BREAST_HANG_Z_DROP_FRAC_RZ == 0.55
    assert BREAST_ATHLETIC_RX_MAX_FRAC_H == 0.042
    assert BREAST_ATHLETIC_RX_MAX_FRAC_H < 0.055  # fail-as-product cup bump
