"""Track 0111 — remake only when emit moved (published README + hold-the-line).

RECIPE_HONESTY / SETUP_LAUNCH_HONESTY / Difficulty §2 / §4 / §9 / §12 / §13 / N6.
Remake policy is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
Tests grep published files only — never conductor/, docs/, .agents/, AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import BPY_BASENAME
from meshops.proportion.errors import ProportionError
from meshops.proportion.setup_launch import run_blockout_open_setup

_REPO = Path(__file__).resolve().parents[1]
_P1 = "Remake only when emit moved"
_P2 = "setup_blockout_recipe.py"
_P3 = "build_and_render.py is work/-only"
_P4 = "Hygiene / CLI / docs / skill tracks are tests"
_HEADING = "Product remake"


def _readme() -> str:
    return (_REPO / "README.md").read_text(encoding="utf-8")


def _remake_section(text: str) -> str:
    idx = text.find(_HEADING)
    if idx < 0:
        return ""
    rest = text[idx:]
    next_h = rest.find("\n### ", 1)
    if next_h < 0:
        next_h = rest.find("\n## ", 1)
    return rest if next_h < 0 else rest[:next_h]


@pytest.fixture
def fake_blender(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    exe = tmp_path / "blender.exe"
    exe.write_bytes(b"")
    resolved = exe.resolve()

    def _find(*, require: bool = True) -> tuple[Path, str]:
        return resolved, "env"

    monkeypatch.setattr(
        "meshops.proportion.setup_launch.find_blender_with_source",
        _find,
    )
    return resolved


def test_t0_readme_p1_emit_moved() -> None:
    """T0: README contains frozen P1."""
    assert _P1 in _readme()


def test_t1_readme_p2_in_remake_section() -> None:
    """T1: P2 lives under the Product remake heading."""
    text = _readme()
    assert _HEADING in text
    section = _remake_section(text)
    assert _P2 in section


def test_t2_readme_p3_work_only() -> None:
    """T2: P3 is contiguous (no backtick between .py and is)."""
    assert _P3 in _readme()


def test_t3_readme_p4_hygiene_tests() -> None:
    """T3: README contains frozen P4."""
    assert _P4 in _readme()


def test_t4_setup_launch_work_only_0111() -> None:
    """T4: refuse helper still cites work/-only and 0111."""
    launch = (_REPO / "src/meshops/proportion/setup_launch.py").read_text(encoding="utf-8")
    assert "work/-only" in launch
    assert "0111" in launch


def test_t5_refuse_build_and_render(tmp_path: Path, fake_blender: Path) -> None:
    """T5: refuse build_and_render.py; setup_not_found; message cites bpy + 0111."""
    bad = tmp_path / "build_and_render.py"
    bad.write_text("# not product\n", encoding="utf-8")
    with pytest.raises(ProportionError) as ei:
        run_blockout_open_setup(bad)
    assert ei.value.code == "setup_not_found"
    msg = str(ei.value)
    assert _P2 in msg
    assert "0111" in msg


def test_t6_mcp_catalog_47() -> None:
    """T6: MCP catalog stays 47."""
    assert len(TOOL_NAMES) == 47


def test_t7_no_remake_cli_command() -> None:
    """T7: no blockout-remake / remake-product / build_and_render command."""
    cli_text = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-remake" not in cli_text
    assert "remake-product" not in cli_text
    assert "def remake" not in cli_text
    assert "def build_and_render" not in cli_text
    assert 'name="build_and_render"' not in cli_text
    assert "name='build_and_render'" not in cli_text


def test_t8_bpy_basename_unchanged() -> None:
    """T8: product bpy basename stays setup_blockout_recipe.py."""
    assert BPY_BASENAME == "setup_blockout_recipe.py"
