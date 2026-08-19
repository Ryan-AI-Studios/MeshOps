"""Track 0110 — blockout-open-setup prints abs Blender --python.

SETUP_LAUNCH_HONESTY / RECIPE_HONESTY / Difficulty §4 / §12 / §13 / N6.
Print or spawn is not mesh/print success. MCP catalog 47. Schema 1.4.0 stay.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.escalate.errors import EscalateError
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import SETUP_LAUNCH_HONESTY
from meshops.proportion.setup_launch import run_blockout_open_setup

_BPY = "setup_blockout_recipe.py"
_STUB = "# setup_blockout_recipe.py — MeshOps track 0019\n"
_REPO = Path(__file__).resolve().parents[1]


def _write_setup(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_STUB, encoding="utf-8")
    return path


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


def test_t0_hygiene() -> None:
    """T0: honesty token, CLI verb, MCP 47, refuse string, no bpy emit."""
    honesty = (_REPO / "src/meshops/proportion/honesty.py").read_text(encoding="utf-8")
    assert "SETUP_LAUNCH_HONESTY" in honesty
    assert "proportion_blockout_setup_launch_not_mesh_or_print_success" in honesty
    cli = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-open-setup" in cli
    server = (_REPO / "src/meshops/mcp/server.py").read_text(encoding="utf-8")
    assert "mesh_proportion_blockout_open_setup" in server
    mcp_test = (_REPO / "tests/test_mcp_server.py").read_text(encoding="utf-8")
    assert "len(TOOL_NAMES) == 47" in mcp_test
    launch = (_REPO / "src/meshops/proportion/setup_launch.py").read_text(encoding="utf-8")
    assert "build_and_render" in launch
    assert "emit_bpy_script" not in launch
    assert "PARTS =" not in launch
    assert "mesh_proportion_blockout_open_setup" in TOOL_NAMES
    assert len(TOOL_NAMES) == 47


def test_t1_file_abs_print(tmp_path: Path, fake_blender: Path) -> None:
    """T1: abs setup file → print-only argv with abs --python."""
    setup_py = _write_setup(tmp_path / "job" / _BPY)
    payload = run_blockout_open_setup(setup_py)
    setup_abs = str(setup_py.resolve())
    blender_abs = str(fake_blender)
    assert payload["ok"] is True
    assert payload["spawned"] is False
    assert payload["background"] is False
    assert payload["setup"] == setup_abs
    assert payload["blender"] == blender_abs
    assert payload["argv"] == [blender_abs, "--python", setup_abs]
    assert payload["honesty"] == SETUP_LAUNCH_HONESTY
    assert blender_abs in payload["command"]
    assert setup_abs in payload["command"]


def test_t2_relative_file_from_tmp_cwd(
    tmp_path: Path, fake_blender: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2: relative basename from tmp cwd resolves to abs in argv."""
    setup_py = _write_setup(tmp_path / _BPY)
    monkeypatch.chdir(tmp_path)
    payload = run_blockout_open_setup(_BPY)
    setup_abs = str(setup_py.resolve())
    assert payload["setup"] == setup_abs
    assert payload["argv"][1] == "--python"
    assert payload["argv"][2] == setup_abs
    assert payload["argv"][2] != _BPY


def test_t3_directory_class(tmp_path: Path, fake_blender: Path) -> None:
    """T3: --setup a dir resolves to dir/setup_blockout_recipe.py (no parent hop)."""
    nofuse = tmp_path / "nofuse"
    setup_py = _write_setup(nofuse / _BPY)
    payload = run_blockout_open_setup(nofuse)
    assert payload["setup"] == str(setup_py.resolve())


def test_t3b_ghost_dir_no_parent_hop(tmp_path: Path, fake_blender: Path) -> None:
    """T3b: missing dir-class path tries only dir/setup_blockout_recipe.py."""
    ghost = tmp_path / "ghost"
    with pytest.raises(ProportionError) as ei:
        run_blockout_open_setup(ghost)
    assert ei.value.code == "setup_not_found"
    msg = str(ei.value)
    tried = str((ghost / _BPY).resolve())
    assert tried in msg
    assert str((tmp_path / _BPY).resolve()) not in msg or tried == str((tmp_path / _BPY).resolve())


def test_t4_json_sibling(tmp_path: Path, fake_blender: Path) -> None:
    """T4: recipe JSON + sibling py → py. JSON without sibling → emit-setup hint."""
    setup_py = _write_setup(tmp_path / _BPY)
    recipe = tmp_path / "blockout_recipe.json"
    recipe.write_text("{}\n", encoding="utf-8")
    payload = run_blockout_open_setup(recipe)
    assert payload["setup"] == str(setup_py.resolve())

    empty = tmp_path / "empty"
    empty.mkdir()
    lonely = empty / "blockout_recipe.json"
    lonely.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProportionError) as ei:
        run_blockout_open_setup(lonely)
    assert ei.value.code == "setup_not_found"
    msg = str(ei.value)
    assert "blockout-emit-setup" in msg
    assert str(lonely.resolve()) in msg


def test_t5_missing_py_file_class(tmp_path: Path, fake_blender: Path) -> None:
    """T5: missing .py file-class → setup_not_found with abs; no dir hop."""
    missing = tmp_path / _BPY
    with pytest.raises(ProportionError) as ei:
        run_blockout_open_setup(missing)
    assert ei.value.code == "setup_not_found"
    assert str(missing.resolve()) in str(ei.value)


def test_t6_relative_cwd_under_fake_windows_root(
    tmp_path: Path, fake_blender: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T6: relative --setup + cwd under patched Windows root → setup_cwd_unsafe."""
    fake_root = tmp_path / "Windows"
    sys32 = fake_root / "System32"
    sys32.mkdir(parents=True)
    monkeypatch.setattr(
        "meshops.proportion.setup_launch._windows_root",
        lambda: fake_root.resolve(),
    )
    monkeypatch.chdir(sys32)
    with pytest.raises(ProportionError) as ei:
        run_blockout_open_setup(r"work\job\setup_blockout_recipe.py")
    assert ei.value.code == "setup_cwd_unsafe"
    cwd_abs = str(sys32.resolve())
    assert cwd_abs in str(ei.value)
    assert ei.value.details.get("cwd") == cwd_abs


def test_t7_resolved_under_windows_root(
    tmp_path: Path, fake_blender: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7: existing abs setup under patched Windows root still unsafe."""
    fake_root = tmp_path / "Windows"
    setup_py = _write_setup(fake_root / "System32" / _BPY)
    monkeypatch.setattr(
        "meshops.proportion.setup_launch._windows_root",
        lambda: fake_root.resolve(),
    )
    with pytest.raises(ProportionError) as ei:
        run_blockout_open_setup(setup_py)
    assert ei.value.code == "setup_cwd_unsafe"
    assert str(setup_py.resolve()) in str(ei.value)


def test_t8_refuse_build_and_render(tmp_path: Path, fake_blender: Path) -> None:
    """T8: refuse build_and_render.py; code setup_not_found; message cites 0111."""
    bad = tmp_path / "build_and_render.py"
    bad.write_text("# not product\n", encoding="utf-8")
    with pytest.raises(ProportionError) as ei:
        run_blockout_open_setup(bad)
    assert ei.value.code == "setup_not_found"
    msg = str(ei.value)
    assert _BPY in msg
    assert "0111" in msg
    assert f"not found {bad.resolve()}" not in msg
    assert ei.value.details.get("setup") == str(bad.resolve())


def test_t9_refuse_guides_setup(tmp_path: Path, fake_blender: Path) -> None:
    """T9: refuse setup_proportion_guides.py; code setup_not_found; message names bpy."""
    bad = tmp_path / "setup_proportion_guides.py"
    bad.write_text("# guides\n", encoding="utf-8")
    with pytest.raises(ProportionError) as ei:
        run_blockout_open_setup(bad)
    assert ei.value.code == "setup_not_found"
    assert _BPY in str(ei.value)


def test_t10_blender_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T10: EscalateError blender_missing wraps as ProportionError blender_missing."""
    setup_py = _write_setup(tmp_path / _BPY)

    def _missing(*, require: bool = True) -> tuple[Path | None, str]:
        raise EscalateError(
            "Blender 5.2 LTS not found. Set MESHOPS_BLENDER to blender.exe.",
            code="blender_missing",
            details={"env": None},
        )

    monkeypatch.setattr(
        "meshops.proportion.setup_launch.find_blender_with_source",
        _missing,
    )
    with pytest.raises(ProportionError) as ei:
        run_blockout_open_setup(setup_py)
    assert ei.value.code == "blender_missing"
    assert "MESHOPS_BLENDER" in str(ei.value)


def test_t11_background(tmp_path: Path, fake_blender: Path) -> None:
    """T11: --background inserts -b after blender and --python-exit-code 1."""
    setup_py = _write_setup(tmp_path / _BPY)
    payload = run_blockout_open_setup(setup_py, background=True)
    argv: list[str] = payload["argv"]
    assert argv[0] == str(fake_blender)
    assert argv[1] == "-b"
    assert "--python" in argv
    i = argv.index("--python-exit-code")
    assert argv[i + 1] == "1"
    assert payload["background"] is True


def test_t12_spawn_mocked(
    tmp_path: Path, fake_blender: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T12: --spawn Popen detached, cwd=setup.parent, never wait."""
    setup_py = _write_setup(tmp_path / "job" / _BPY)
    seen: dict[str, Any] = {}

    class FakePopen:
        def __init__(self, args: list[str], **kwargs: Any) -> None:
            seen["args"] = args
            seen["kwargs"] = kwargs
            seen["waited"] = False

        def wait(self, *args: Any, **kwargs: Any) -> int:
            seen["waited"] = True
            return 0

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            seen["waited"] = True
            return (b"", b"")

    monkeypatch.setattr("meshops.proportion.setup_launch.subprocess.Popen", FakePopen)
    payload = run_blockout_open_setup(setup_py, spawn=True)
    assert payload["spawned"] is True
    assert seen["args"] == payload["argv"]
    assert Path(seen["kwargs"]["cwd"]) == setup_py.parent.resolve()
    assert seen["waited"] is False
    assert seen["kwargs"]["stdin"] is subprocess.DEVNULL
    assert seen["kwargs"]["stdout"] is subprocess.DEVNULL
    assert seen["kwargs"]["stderr"] is subprocess.DEVNULL
    if os.name == "nt":
        flags = seen["kwargs"]["creationflags"]
        expected = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        assert flags == expected
        assert "CREATE_NO_WINDOW" not in str(flags)
    else:
        assert seen["kwargs"].get("start_new_session") is True


def test_t12b_spawn_oserror(
    tmp_path: Path, fake_blender: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T12b: Popen OSError → setup_spawn_failed."""
    setup_py = _write_setup(tmp_path / _BPY)

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated spawn fail")

    monkeypatch.setattr("meshops.proportion.setup_launch.subprocess.Popen", _boom)
    with pytest.raises(ProportionError) as ei:
        run_blockout_open_setup(setup_py, spawn=True)
    assert ei.value.code == "setup_spawn_failed"
    assert ei.value.details.get("setup") == str(setup_py.resolve())


def test_t13_cli_json(tmp_path: Path, fake_blender: Path) -> None:
    """T13: CLI --json exit 0; payload ok=true."""
    setup_py = _write_setup(tmp_path / _BPY)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-open-setup",
            "--setup",
            str(setup_py),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["ok"] is True
    assert data["setup"] == str(setup_py.resolve())
    assert data["honesty"] == SETUP_LAUNCH_HONESTY


def test_t14_mcp_catalog_47() -> None:
    """T14: mesh_proportion_blockout_open_setup in TOOL_NAMES; len == 47."""
    assert "mesh_proportion_blockout_open_setup" in TOOL_NAMES
    assert len(TOOL_NAMES) == 47


def test_t14b_mcp_wrapper(tmp_path: Path, fake_blender: Path) -> None:
    """T14b: MCP wrapper resolves --setup and returns print-only payload."""
    from meshops.mcp.tools import mesh_proportion_blockout_open_setup

    setup_py = _write_setup(tmp_path / _BPY)
    payload = mesh_proportion_blockout_open_setup(
        tmp_path,
        setup=str(setup_py),
        spawn=False,
        background=False,
    )
    assert payload["ok"] is True
    assert payload["spawned"] is False
    assert payload["setup"] == str(setup_py.resolve())
    assert payload["honesty"] == SETUP_LAUNCH_HONESTY
