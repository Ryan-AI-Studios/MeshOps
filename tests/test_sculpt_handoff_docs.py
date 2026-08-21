"""Track 0114 — post-nofuse sculpt handoff (published README + hold-the-line).

RECIPE_HONESTY / FUSE_HONESTY / Difficulty §8 / §9 / §12 / §13 / N1 / N6.
Sculpt handoff is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
Tests grep published files only — never conductor/, docs/, .agents/, AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import build_blockout_recipe
from meshops.proportion.errors import ProportionError
from meshops.proportion.fuse_plan import DEFAULT_VOXEL_COARSE_M, DEFAULT_VOXEL_FINE_M
from meshops.proportion.honesty import FUSE_HONESTY
from meshops.proportion.models import ProportionReport, QualityFlags

_REPO = Path(__file__).resolve().parents[1]
_S1 = "Stop RECIPE oval thrash after form lock"
_S2 = "blockout-fuse-plan"
_S3 = "Voxel remesh is opt-in authoring weld"
_S4 = "1 island is not print success"
_HEADING = "Post-nofuse sculpt"


def _readme() -> str:
    return (_REPO / "README.md").read_text(encoding="utf-8")


def _sculpt_section(text: str) -> str:
    idx = text.find(_HEADING)
    if idx < 0:
        return ""
    rest = text[idx:]
    next_h = rest.find("\n### ", 1)
    if next_h < 0:
        next_h = rest.find("\n## ", 1)
    return rest if next_h < 0 else rest[:next_h]


def test_t0_readme_s1_stop_oval_thrash() -> None:
    """T0: README contains frozen S1."""
    assert _S1 in _readme()


def test_t1_readme_s2_join_ready_in_sculpt_section() -> None:
    """T1: S2 and --join-ready live under the Post-nofuse sculpt heading."""
    text = _readme()
    assert _HEADING in text
    section = _sculpt_section(text)
    assert _S2 in section
    assert "--join-ready" in section


def test_t2_readme_s3_voxel_opt_in() -> None:
    """T2: S3 is contiguous (no backtick between Voxel remesh and is)."""
    assert _S3 in _readme()


def test_t3_readme_s4_island_not_print() -> None:
    """T3: README contains frozen S4."""
    assert _S4 in _readme()


def test_t4_voxel_defaults_hold() -> None:
    """T4: 0039 voxel defaults stay 0.02 / 0.014."""
    assert DEFAULT_VOXEL_COARSE_M == 0.02
    assert DEFAULT_VOXEL_FINE_M == 0.014


def test_t5_fuse_honesty_token() -> None:
    """T5: FUSE_HONESTY token hold."""
    assert FUSE_HONESTY == "proportion_blockout_fuse_not_mesh_or_print_success"


def test_t6_mcp_catalog_47() -> None:
    """T6: MCP catalog stays 47."""
    assert len(TOOL_NAMES) == 47


def test_t7_no_sculpt_cli_command() -> None:
    """T7: no blockout-sculpt / blockout-remesh / def sculpt command."""
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-sculpt" not in cli_text
    assert "blockout-remesh" not in cli_text
    assert "def sculpt" not in cli_text


def test_t8_nofuse_join_ready_mutual_exclusion() -> None:
    """T8: nofuse+join_ready still recipe_failed at build_blockout_recipe."""
    report = ProportionReport(
        schema_version="1.2.0",
        honesty="proportion_measurement_not_mesh_or_print_success",
        quality=QualityFlags(),
    )
    with pytest.raises(ProportionError) as ei:
        build_blockout_recipe(report, nofuse=True, join_ready=True)
    assert ei.value.code == "recipe_failed"
    assert "mutually exclusive" in str(ei.value)
