"""Track 0113 — one-character Loomis polish (published README + hold-the-line).

RECIPE_HONESTY / Difficulty §1 / §2 / §5 / §9 / §12 / N6.
Loomis polish is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
Tests grep published files only — never conductor/, docs/, .agents/, AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import HEAD_PITCH_DEG, build_blockout_recipe
from meshops.proportion.errors import ProportionError
from meshops.proportion.face_recipe import (
    CHEEK_Z_MIX,
    EYE_RADIUS_FRAC_H,
    EYE_RY_FRAC_R,
    EYE_RZ_FRAC_R,
    JAW_RY_FRAC_HEAD_RY,
    LIP_RY_FRAC_H,
    LIP_RZ_FRAC_H,
)
from meshops.proportion.models import ProportionReport

_REPO = Path(__file__).resolve().parents[1]
_L1 = "One-character Loomis is not a product default"
_L2 = "--face"
_L3 = "Do not promote session face tweaks"
_L4 = "Loomis polish is not print success"
_HEADING = "One-character Loomis"


def _readme() -> str:
    return (_REPO / "README.md").read_text(encoding="utf-8")


def _loomis_section(text: str) -> str:
    idx = text.find(_HEADING)
    if idx < 0:
        return ""
    rest = text[idx:]
    next_h = rest.find("\n### ", 1)
    if next_h < 0:
        next_h = rest.find("\n## ", 1)
    return rest if next_h < 0 else rest[:next_h]


def test_t0_readme_l1_not_a_product_default() -> None:
    """T0: README contains frozen L1."""
    assert _L1 in _readme()


def test_t1_readme_l2_face_in_loomis_section() -> None:
    """T1: L2 lives under the One-character Loomis heading."""
    text = _readme()
    assert _HEADING in text
    section = _loomis_section(text)
    assert _L2 in section


def test_t2_readme_l3_do_not_promote_contiguous() -> None:
    """T2: L3 is contiguous (no backtick between Do not promote and session)."""
    assert _L3 in _readme()


def test_t3_readme_l4_loomis_not_print() -> None:
    """T3: README contains frozen L4."""
    assert _L4 in _readme()


def test_t4_eye_0102_hold() -> None:
    """T4: 0102 orbital hold — eye ry 0.62 / radius 0.11 / rz 0.58."""
    assert EYE_RY_FRAC_R == 0.62
    assert EYE_RADIUS_FRAC_H == 0.11
    assert EYE_RZ_FRAC_R == 0.58


def test_t5_lip_cheek_0102_hold() -> None:
    """T5: 0102 lip/cheek hold — lip 0.028/0.020 / cheek mix 0.30."""
    assert LIP_RY_FRAC_H == 0.028
    assert LIP_RZ_FRAC_H == 0.020
    assert CHEEK_Z_MIX == 0.30


def test_t6_mcp_catalog_47() -> None:
    """T6: MCP catalog stays 47."""
    assert len(TOOL_NAMES) == 47


def test_t7_no_loomis_cli_command() -> None:
    """T7: no blockout-loomis / blockout-face-polish / def loomis_polish command."""
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-loomis" not in cli_text
    assert "blockout-face-polish" not in cli_text
    assert "def loomis_polish" not in cli_text


def test_t8_pitch_jaw_and_empty_report() -> None:
    """T8: HEAD_PITCH_DEG 6.0 + JAW_RY 0.42; empty report is recipe_empty."""
    assert HEAD_PITCH_DEG == 6.0
    assert JAW_RY_FRAC_HEAD_RY == 0.42
    report = ProportionReport(schema_version="1.1.0")
    with pytest.raises(ProportionError) as ei:
        build_blockout_recipe(report)
    assert ei.value.code == "recipe_empty"
