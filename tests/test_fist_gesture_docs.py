"""Track 0115 — fist/gesture hands (published README + hold-the-line).

RECIPE_HONESTY / Difficulty §1 / §2 / §5 / §9 / §12 / N6.
Fist polish is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
Tests grep published files only — never conductor/, docs/, .agents/, AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.extremity_recipe import (
    _FINGER_CURL_DIP_DEG,
    _FINGER_CURL_PIP_DEG,
    _FINGER_R_SCALES_SEG,
    _FINGER_SEG_FRACS_HAND,
    _THUMB_CURL_IP_DEG,
    _THUMB_PALM_PITCH,
)

_REPO = Path(__file__).resolve().parents[1]
_F1 = "Fist/gesture is not a product default"
_F2 = "--fingers"
_F3 = "Do not promote session fist tweaks"
_F4 = "Fist polish is not print success"
_HEADING = "Fist / gesture hands"


def _readme() -> str:
    return (_REPO / "README.md").read_text(encoding="utf-8")


def _fist_section(text: str) -> str:
    idx = text.find(_HEADING)
    if idx < 0:
        return ""
    rest = text[idx:]
    next_h = rest.find("\n### ", 1)
    if next_h < 0:
        next_h = rest.find("\n## ", 1)
    return rest if next_h < 0 else rest[:next_h]


def test_t0_readme_f1_not_a_product_default() -> None:
    """T0: README contains frozen F1."""
    assert _F1 in _readme()


def test_t1_readme_f2_fingers_in_fist_section() -> None:
    """T1: F2 lives under the Fist / gesture hands heading."""
    text = _readme()
    assert _HEADING in text
    section = _fist_section(text)
    assert _F2 in section


def test_t2_readme_f3_do_not_promote_contiguous() -> None:
    """T2: F3 is contiguous (no backtick between Do not promote and session)."""
    assert _F3 in _readme()


def test_t3_readme_f4_fist_not_print() -> None:
    """T3: README contains frozen F4."""
    assert _F4 in _readme()


def test_t4_curl_0104_hold() -> None:
    """T4: 0104 hang-flex hold — PIP 14 / DIP 20 / thumb IP 12; fail >=30 / >=35 as fist."""
    assert _FINGER_CURL_PIP_DEG == 14.0
    assert _FINGER_CURL_DIP_DEG == 20.0
    assert _THUMB_CURL_IP_DEG == 12.0
    assert _FINGER_CURL_PIP_DEG < 30.0  # 0104 B1: >=30 is a fist, not product hang-flex
    assert _FINGER_CURL_DIP_DEG < 35.0  # 0104 B2: >=35 is a fist, not product hang-flex


def test_t5_r_scales_0088_hold() -> None:
    """T5: 0088 taper hold — r scales 1.00 / 0.86 / 0.72."""
    assert _FINGER_R_SCALES_SEG == (1.00, 0.86, 0.72)


def test_t6_mcp_catalog_47() -> None:
    """T6: MCP catalog stays 47."""
    assert len(TOOL_NAMES) == 47


def test_t7_no_fist_cli_command() -> None:
    """T7: no blockout-fist / blockout-gesture / def fist_pose command."""
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-fist" not in cli_text
    assert "blockout-gesture" not in cli_text
    assert "def fist_pose" not in cli_text


def test_t8_thumb_pitch_and_seg_fracs() -> None:
    """T8: 0084 hang pitch -0.55 + 0088 segs 0.27/0.18/0.10."""
    assert _THUMB_PALM_PITCH == -0.55
    assert _FINGER_SEG_FRACS_HAND == (0.27, 0.18, 0.10)
