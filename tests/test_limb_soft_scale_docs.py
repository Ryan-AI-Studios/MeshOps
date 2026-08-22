"""Track 0116 — per-job limb soft scale (published README + hold-the-line).

RECIPE_HONESTY / Difficulty §1 / §2 / §5 / §9 / §12 / N6.
Limb scale polish is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
Tests grep published files only — never conductor/, docs/, .agents/, AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    CALF_DIST_SHAFT_SCALE,
    ELBOW_SOFT_SCALE,
    FA_DIST_SHAFT_SCALE,
    KNEE_SOFT_FRAC,
    KNEE_SOFT_RZ_FRAC,
    THIGH_DIST_SHAFT_SCALE,
    UA_DIST_SHAFT_SCALE,
)

_REPO = Path(__file__).resolve().parents[1]
_J1 = "Limb soft scale is not a product default"
_J2 = "--limbs"
_J3 = "Do not promote session limb-scale tweaks"
_J4 = "Limb scale polish is not print success"
_HEADING = "Per-job limb soft scale"


def _readme() -> str:
    return (_REPO / "README.md").read_text(encoding="utf-8")


def _limb_section(text: str) -> str:
    idx = text.find(_HEADING)
    if idx < 0:
        return ""
    rest = text[idx:]
    next_h = rest.find("\n### ", 1)
    if next_h < 0:
        next_h = rest.find("\n## ", 1)
    return rest if next_h < 0 else rest[:next_h]


def test_t0_readme_j1_not_a_product_default() -> None:
    """T0: README contains frozen J1."""
    assert _J1 in _readme()


def test_t1_readme_j2_limbs_in_limb_section() -> None:
    """T1: J2 lives under the Per-job limb soft scale heading."""
    text = _readme()
    assert _HEADING in text
    section = _limb_section(text)
    assert _J2 in section


def test_t2_readme_j3_do_not_promote_contiguous() -> None:
    """T2: J3 is contiguous on the full README (no backtick between Do not promote and session)."""
    assert _J3 in _readme()


def test_t3_readme_j4_limb_scale_not_print() -> None:
    """T3: README contains frozen J4."""
    assert _J4 in _readme()


def test_t4_elbow_knee_seam_hold() -> None:
    """T4: 0081 elbow 1.22 + 0095 knee 1.08 hold; fail >=1.28 / >=1.18 as product."""
    assert ELBOW_SOFT_SCALE == 1.22
    assert KNEE_SOFT_FRAC == 1.08
    assert ELBOW_SOFT_SCALE < 1.28  # 0081 cap: >=1.28 is not product bead
    assert KNEE_SOFT_FRAC < 1.18  # 0095 inverted from 1.18; do not restore


def test_t5_shaft_0107_hold() -> None:
    """T5: 0107 shaft hold — UA 0.84 / FA 0.70 / calf 0.80."""
    assert UA_DIST_SHAFT_SCALE == 0.84
    assert FA_DIST_SHAFT_SCALE == 0.70
    assert CALF_DIST_SHAFT_SCALE == 0.80


def test_t6_mcp_catalog_47() -> None:
    """T6: MCP catalog stays 47."""
    assert len(TOOL_NAMES) == 47


def test_t7_no_limb_scale_cli_command() -> None:
    """T7: no blockout-limb-scale / blockout-soft-scale / def limb_soft_scale command."""
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-limb-scale" not in cli_text
    assert "blockout-soft-scale" not in cli_text
    assert "def limb_soft_scale" not in cli_text


def test_t8_thigh_dist_and_knee_rz() -> None:
    """T8: 0094 thigh dist 0.72 + 0095 knee rz 1.15."""
    assert THIGH_DIST_SHAFT_SCALE == 0.72
    assert KNEE_SOFT_RZ_FRAC == 1.15
