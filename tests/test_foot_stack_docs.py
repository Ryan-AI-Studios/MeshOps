"""Track 0123 — per-job foot stack (published README + hold-the-line).

RECIPE_HONESTY / Difficulty §1 / §2 / §5 / §9 / §12 / N6.
Foot polish is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
Tests grep published files only — never conductor/, docs/, .agents/, AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import COMPACT_CULL_NAME_PREFIXES
from meshops.proportion.extremity_recipe import (
    ANK_RY_FRAC_HALF_W,
    ARCH_SOFT_RY_FRAC_HALF_DEPTH,
    BALL_SOFT_RY_FRAC_HALF_DEPTH,
    TOE_BALL_NEST_FRAC,
    TOE_TIP_PAD_RY_FRAC,
    TOE_TIP_PAD_SCALE,
)

_REPO = Path(__file__).resolve().parents[1]
_E1 = "Per-job foot stack is not a product default"
_E2 = "--feet"
_E3 = "Do not promote session foot tweaks"
_E4 = "Foot polish is not print success"
_HEADING = "Per-job foot stack"


def _readme() -> str:
    return (_REPO / "README.md").read_text(encoding="utf-8")


def _foot_section(text: str) -> str:
    idx = text.find(_HEADING)
    if idx < 0:
        return ""
    rest = text[idx:]
    next_h = rest.find("\n### ", 1)
    if next_h < 0:
        next_h = rest.find("\n## ", 1)
    return rest if next_h < 0 else rest[:next_h]


def test_t0_readme_e1_not_a_product_default() -> None:
    """T0: README contains frozen E1."""
    assert _E1 in _readme()


def test_t1_readme_e2_feet_in_foot_section() -> None:
    """T1: E2 lives under the Per-job foot stack heading."""
    text = _readme()
    assert _HEADING in text
    section = _foot_section(text)
    assert _E2 in section


def test_t2_readme_e3_do_not_promote_contiguous() -> None:
    """T2: E3 is contiguous on the full README (no backtick between Do not promote and session)."""
    assert _E3 in _readme()


def test_t3_readme_e4_foot_not_print() -> None:
    """T3: README contains frozen E4."""
    assert _E4 in _readme()


def test_t4_foot_0108_ank_arch_ball_hold() -> None:
    """T4: 0108 hold ank 0.78 / arch 0.18 / ball 0.16; fail ank >=0.92 as product bead."""
    assert ANK_RY_FRAC_HALF_W == 0.78
    assert ARCH_SOFT_RY_FRAC_HALF_DEPTH == 0.18
    assert BALL_SOFT_RY_FRAC_HALF_DEPTH == 0.16
    assert ANK_RY_FRAC_HALF_W < 0.92  # fail-as-product bead


def test_t5_foot_0108_nest_tip_hold() -> None:
    """T5: 0108 hold nest 0.52 / tip_ry 0.55 / tip scale 0.78."""
    assert TOE_BALL_NEST_FRAC == 0.52
    assert TOE_TIP_PAD_RY_FRAC == 0.55
    assert TOE_TIP_PAD_SCALE == 0.78


def test_t6_mcp_catalog_47() -> None:
    """T6: MCP catalog stays 47."""
    assert len(TOOL_NAMES) == 56


def test_t7_no_foot_cli_command() -> None:
    """T7: no blockout-foot / def foot_stack_scale / def foot_scale / def foot_polish."""
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-foot" not in cli_text
    assert "def foot_stack_scale" not in cli_text
    assert "def foot_scale" not in cli_text
    assert "def foot_polish" not in cli_text


def test_t8_ank_structural_keep_and_compact_culls_tip_arch() -> None:
    """T8: 0108 ank 0.78 + name-prefix keep/cull (not fake role tokens)."""
    assert ANK_RY_FRAC_HALF_W == 0.78
    assert "RECIPE_ank_foot_" not in COMPACT_CULL_NAME_PREFIXES
    assert "RECIPE_toe_tip_" in COMPACT_CULL_NAME_PREFIXES
    assert "RECIPE_arch_soft_" in COMPACT_CULL_NAME_PREFIXES
