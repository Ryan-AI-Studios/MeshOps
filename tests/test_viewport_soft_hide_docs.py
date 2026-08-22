"""Track 0112 — viewport soft hide / compact QA (published README + hold-the-line).

RECIPE_HONESTY / Difficulty §5 / §9 / §12 / N6.
Viewport hide is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
Tests grep published files only — never conductor/, docs/, .agents/, AGENTS.md.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    COMPACT_CULL_NAME_EXACT,
    COMPACT_CULL_NAME_PREFIXES,
    COMPACT_CULL_ROLES,
    build_blockout_recipe,
)

_REPO = Path(__file__).resolve().parents[1]
_H1 = "Hide is not cull"
_H2 = "--soft-density"
_H3 = "Keep full for region polish"
_H4 = "Viewport hide is not print success"
_HEADING = "Viewport soft hide"


def _readme() -> str:
    return (_REPO / "README.md").read_text(encoding="utf-8")


def _hide_section(text: str) -> str:
    idx = text.find(_HEADING)
    if idx < 0:
        return ""
    rest = text[idx:]
    next_h = rest.find("\n### ", 1)
    if next_h < 0:
        next_h = rest.find("\n## ", 1)
    return rest if next_h < 0 else rest[:next_h]


def test_t0_readme_h1_hide_is_not_cull() -> None:
    """T0: README contains frozen H1."""
    assert _H1 in _readme()


def test_t1_readme_h2_soft_density_in_hide_section() -> None:
    """T1: H2 lives under the Viewport soft hide heading."""
    text = _readme()
    assert _HEADING in text
    section = _hide_section(text)
    assert _H2 in section


def test_t2_readme_h3_keep_full_contiguous() -> None:
    """T2: H3 is contiguous (no backtick between Keep full and for)."""
    assert _H3 in _readme()


def test_t3_readme_h4_hide_not_print() -> None:
    """T3: README contains frozen H4."""
    assert _H4 in _readme()


def test_t4_compact_cull_roles_hold() -> None:
    """T4: 0082 compact role cull still includes secondaries, not structural."""
    assert "brow_soft" in COMPACT_CULL_ROLES
    assert "eye_soft" in COMPACT_CULL_ROLES
    assert "bicep_soft" in COMPACT_CULL_ROLES
    assert "mid_back_soft" in COMPACT_CULL_ROLES
    assert "trap_soft" in COMPACT_CULL_ROLES
    assert "clavicle" in COMPACT_CULL_ROLES
    assert "jaw" not in COMPACT_CULL_ROLES
    assert "deltoid_soft" not in COMPACT_CULL_ROLES
    assert "hip_soft" not in COMPACT_CULL_ROLES


def test_t5_compact_cull_name_sets_hold() -> None:
    """T5: 0082 compact name prefixes and exact set stay frozen."""
    assert COMPACT_CULL_NAME_PREFIXES == (
        "RECIPE_triceps_soft_",
        "RECIPE_dist_soft_forearm_",
        "RECIPE_toe_tip_",
        "RECIPE_arch_soft_",
    )
    assert {"RECIPE_neck_base_soft"} == COMPACT_CULL_NAME_EXACT


def test_t6_mcp_catalog_47() -> None:
    """T6: MCP catalog stays 47."""
    assert len(TOOL_NAMES) == 47


def test_t7_no_hide_cli_command() -> None:
    """T7: no blockout-hide / blockout-viewport-hide / def hide_soft command."""
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-hide" not in cli_text
    assert "blockout-viewport-hide" not in cli_text
    assert "def hide_soft" not in cli_text


def test_t8_soft_density_default_full() -> None:
    """T8: build_blockout_recipe soft_density default stays full."""
    param = inspect.signature(build_blockout_recipe).parameters["soft_density"]
    assert param.default == "full"
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    idx = cli_text.find("--soft-density")
    assert idx >= 0
    window = cli_text[max(0, idx - 80) : idx]
    assert '"full"' in window
