"""Track 0122 — per-job hip_soft (published README + hold-the-line).

RECIPE_HONESTY / Difficulty §1 / §2 / §5 / §9 / §12 / N6.
Hip-soft polish is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
Tests grep published files only — never conductor/, docs/, .agents/, AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    COMPACT_CULL_NAME_PREFIXES,
    HIP_SOFT_RX_SCALE,
    HIP_SOFT_RY_FRAC_RX,
    HIP_SOFT_RZ_FRAC_RX,
    HIP_SOFT_Y_REAR_FRAC_RX,
    HIP_SOFT_Z_DROP_FRAC_H,
)

_REPO = Path(__file__).resolve().parents[1]
_K1 = "Per-job hip_soft is not a product default"
_K2 = "--torso"
_K3 = "Do not promote session hip-soft tweaks"
_K4 = "Hip-soft polish is not print success"
_HEADING = "Per-job hip_soft"


def _readme() -> str:
    return (_REPO / "README.md").read_text(encoding="utf-8")


def _hip_section(text: str) -> str:
    idx = text.find(_HEADING)
    if idx < 0:
        return ""
    rest = text[idx:]
    next_h = rest.find("\n### ", 1)
    if next_h < 0:
        next_h = rest.find("\n## ", 1)
    return rest if next_h < 0 else rest[:next_h]


def test_t0_readme_k1_not_a_product_default() -> None:
    """T0: README contains frozen K1."""
    assert _K1 in _readme()


def test_t1_readme_k2_torso_in_hip_section() -> None:
    """T1: K2 lives under the Per-job hip_soft heading."""
    text = _readme()
    assert _HEADING in text
    section = _hip_section(text)
    assert _K2 in section


def test_t2_readme_k3_do_not_promote_contiguous() -> None:
    """T2: K3 is contiguous on the full README (no backtick between Do not promote and session)."""
    assert _K3 in _readme()


def test_t3_readme_k4_hip_soft_not_print() -> None:
    """T3: README contains frozen K4."""
    assert _K4 in _readme()


def test_t4_hip_0106_hold() -> None:
    """T4: 0106 hold ry 0.62 / rz 1.00 / drop 0.022; fail rz >=1.18 as product sausage."""
    assert HIP_SOFT_RY_FRAC_RX == 0.62
    assert HIP_SOFT_RZ_FRAC_RX == 1.00
    assert HIP_SOFT_Z_DROP_FRAC_H == 0.022
    assert HIP_SOFT_RZ_FRAC_RX < 1.18  # fail-as-product sausage


def test_t5_hip_rx_scale_hold() -> None:
    """T5: 0069/0106 hold HIP_SOFT_RX_SCALE 1.15."""
    assert HIP_SOFT_RX_SCALE == 1.15


def test_t6_mcp_catalog_47() -> None:
    """T6: MCP catalog stays 47."""
    assert len(TOOL_NAMES) == 51


def test_t7_no_hip_cli_command() -> None:
    """T7: no hip_soft skill CLI (0126 hip-glute-compare is a different verb)."""
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-hip-soft" not in cli_text
    assert 'command("blockout-hip")' not in cli_text
    assert "def hip_soft_scale" not in cli_text
    assert "def hip_scale" not in cli_text
    assert "def hip_michelin" not in cli_text


def test_t8_hip_structural_keep_y_rear() -> None:
    """T8: RECIPE_hip_soft_ not in compact name prefixes + Y rear 0.12."""
    assert "RECIPE_hip_soft_" not in COMPACT_CULL_NAME_PREFIXES
    assert HIP_SOFT_Y_REAR_FRAC_RX == 0.12
