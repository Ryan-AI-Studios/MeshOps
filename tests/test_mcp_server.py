"""In-memory MCP adapter tests (track 0008). Marker: mcp."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp import Client

from meshops.ingest.pipeline import ingest_stl
from meshops.mcp import TOOL_NAMES, build_server, resolve_work_root
from meshops.mcp.server import SERVER_INSTRUCTIONS

pytestmark = pytest.mark.mcp

FORBIDDEN_TOOLS = frozenset({"design_organic_api", "design_organic_agent"})


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_resolve_work_root_override(tmp_path: Path) -> None:
    wr = resolve_work_root(tmp_path)
    assert wr == tmp_path.resolve()


def test_resolve_work_root_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESHOPS_WORK", str(tmp_path / "jobs"))
    wr = resolve_work_root()
    assert wr == (tmp_path / "jobs").resolve()


def test_tool_catalog_complete_and_no_forbidden() -> None:
    async def _body() -> None:
        server = build_server()
        async with Client(server) as client:
            listed = await client.list_tools()
            names = {t.name for t in listed.tools}
            assert names >= TOOL_NAMES
            assert names & FORBIDDEN_TOOLS == set()
            assert "mesh_list_recipes" in names
            assert "mesh_accept_revision" in names
            assert "mesh_accept_candidate" in names
            assert "mesh_promote_working" in names
            assert "mesh_import_sculpt" in names

    _run(_body())


def test_server_identity() -> None:
    async def _body() -> None:
        server = build_server()
        async with Client(server) as client:
            assert client.server_info is not None
            assert client.server_info.name == "meshops"
            assert client.protocol_version == "2026-07-28"
            assert SERVER_INSTRUCTIONS
            assert "mesh_triage" in SERVER_INSTRUCTIONS

    _run(_body())


def test_mesh_version_success() -> None:
    async def _body() -> None:
        server = build_server()
        async with Client(server) as client:
            result = await client.call_tool("mesh_version", {})
            assert result.is_error is False
            assert result.structured_content is not None
            assert "version" in result.structured_content

    _run(_body())


def test_mesh_list_recipes_known() -> None:
    async def _body() -> None:
        server = build_server()
        async with Client(server) as client:
            result = await client.call_tool("mesh_list_recipes", {})
            assert result.is_error is False
            assert result.structured_content is not None
            recipes = result.structured_content["recipes"]
            assert "t1_clean" in recipes
            assert "t2_smooth_spikes" in recipes
            assert "t2_close_small_holes" in recipes

    _run(_body())


def test_ingest_and_triage_under_tmp(
    solid_cylinder_stl: Path,
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()

    async def _body() -> None:
        server = build_server(work_root=work)
        async with Client(server) as client:
            ing = await client.call_tool("mesh_ingest", {"path": str(solid_cylinder_stl)})
            assert ing.is_error is False
            assert ing.structured_content is not None
            mesh_id = ing.structured_content["mesh_id"]
            assert isinstance(mesh_id, str) and mesh_id
            assert (work / mesh_id).is_dir()

            tri = await client.call_tool("mesh_triage", {"mesh_id": mesh_id})
            assert tri.is_error is False
            assert tri.structured_content is not None
            assert "diagnostics" in tri.structured_content
            assert (work / mesh_id / "diagnostics.json").is_file()

    _run(_body())


def test_repair_unknown_recipe_is_error(
    solid_cylinder_stl: Path,
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    # Need a job first
    mesh_id = ingest_stl(solid_cylinder_stl, work_root=work).mesh_id

    async def _body() -> None:
        server = build_server(work_root=work)
        async with Client(server) as client:
            result = await client.call_tool(
                "mesh_repair",
                {"mesh_id": mesh_id, "recipe": "nope_not_a_recipe"},
            )
            assert result.is_error is True
            assert result.structured_content is None
            # Model-readable text content
            assert result.content

    _run(_body())


def test_import_sculpt_approve_required_in_schema() -> None:
    async def _body() -> None:
        server = build_server()
        async with Client(server) as client:
            listed = await client.list_tools()
            tool = next(t for t in listed.tools if t.name == "mesh_import_sculpt")
            schema = tool.input_schema
            assert schema is not None
            required = schema.get("required") or []
            assert "approve" in required
            # No default on approve: property may exist but must be required
            props = schema.get("properties") or {}
            assert "approve" in props
            # default must not be present for required consent param (R8)
            approve_prop = props["approve"]
            assert "default" not in approve_prop

    _run(_body())


def test_parity_ingest_api_vs_mcp(
    solid_cylinder_stl: Path,
    tmp_path: Path,
) -> None:
    """Same STL via engine API and MCP mesh_ingest → same mesh_id (content hash)."""
    work_api = tmp_path / "work_api"
    work_mcp = tmp_path / "work_mcp"
    work_api.mkdir()
    work_mcp.mkdir()

    api_result = ingest_stl(solid_cylinder_stl, work_root=work_api)

    async def _body() -> str:
        server = build_server(work_root=work_mcp)
        async with Client(server) as client:
            r = await client.call_tool("mesh_ingest", {"path": str(solid_cylinder_stl)})
            assert r.is_error is False
            assert r.structured_content is not None
            return str(r.structured_content["mesh_id"])

    mcp_id = _run(_body())
    assert mcp_id == api_result.mesh_id


def test_entrypoint_missing_mcp_hint() -> None:
    """Friendly ImportError path: simulate missing mcp via stripped sys.path."""
    # Run a small Python snippet that forces ImportError before main import
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "# Prepend a fake empty mcp package block by shadowing via builtins\n"
        "import builtins\n"
        "_real = builtins.__import__\n"
        "def _block(name, *a, **k):\n"
        "    if name == 'mcp' or name.startswith('mcp.'):\n"
        "        raise ImportError('blocked')\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = _block\n"
        "sys.path.insert(0, str(Path(r'"
        + str(Path(__file__).resolve().parents[1] / "src")
        + "')))\n"
        "from meshops.mcp.__main__ import main\n"
        "try:\n"
        "    main()\n"
        "except SystemExit as e:\n"
        "    raise SystemExit(e.code)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "uv sync --extra mcp" in proc.stderr
    assert proc.stdout == ""
