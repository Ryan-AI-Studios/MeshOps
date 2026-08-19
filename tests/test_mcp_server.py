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

from meshops.ingest.pipeline import ingest_stl  # used for unknown-recipe pre-seed
from meshops.mcp import TOOL_NAMES, build_server, resolve_work_root
from meshops.mcp.server import SERVER_INSTRUCTIONS

pytestmark = pytest.mark.mcp

# design_organic_api is registered by 0007 (post-plateau hosted fallback).
FORBIDDEN_TOOLS = frozenset({"design_organic_agent"})


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
            assert "design_organic_api" in names
            assert "mesh_proportion_template" in names
            assert "mesh_proportion_templates" in names
            assert "mesh_proportion_apply_template" in names
            assert "mesh_proportion_analyze" in names
            assert "mesh_proportion_show" in names
            assert "mesh_proportion_scaffold" in names
            assert "mesh_proportion_guides" in names
            assert "mesh_proportion_capture" in names
            assert "mesh_proportion_depth_samples" in names
            assert "mesh_proportion_blockout_recipe" in names
            assert "mesh_proportion_anatomy_profiles" in names
            assert "mesh_proportion_blockout_validate_constraints" in names
            assert "mesh_proportion_blockout_optimize" in names
            assert "mesh_proportion_blockout_emit_setup" in names
            assert "mesh_proportion_blockout_fuse_plan" in names
            assert "mesh_proportion_skeleton_build" in names
            assert "mesh_proportion_depth_heatmap" in names
            assert "mesh_proportion_depth_hint" in names
            assert "mesh_proportion_silhouette_compare" in names
            assert "mesh_proportion_blockout_feedback" in names
            assert "mesh_proportion_blockout_open_setup" in names
            assert len(names) == 47

    _run(_body())


def test_mcp__proportion_tools_in_catalog() -> None:
    """Explicit 0110 catalog freeze: proportion tools + open-setup; len == 47."""

    async def _body() -> None:
        server = build_server()
        async with Client(server) as client:
            listed = await client.list_tools()
            names = {t.name for t in listed.tools}
            for n in (
                "mesh_proportion_template",
                "mesh_proportion_templates",
                "mesh_proportion_apply_template",
                "mesh_proportion_analyze",
                "mesh_proportion_show",
                "mesh_proportion_scaffold",
                "mesh_proportion_guides",
                "mesh_proportion_capture",
                "mesh_proportion_depth_samples",
                "mesh_proportion_blockout_recipe",
                "mesh_proportion_anatomy_profiles",
                "mesh_proportion_blockout_validate_constraints",
                "mesh_proportion_blockout_optimize",
                "mesh_proportion_blockout_emit_setup",
                "mesh_proportion_blockout_fuse_plan",
                "mesh_proportion_skeleton_build",
                "mesh_proportion_depth_heatmap",
                "mesh_proportion_depth_hint",
                "mesh_proportion_silhouette_compare",
                "mesh_proportion_blockout_feedback",
                "mesh_proportion_blockout_open_setup",
            ):
                assert n in names
            assert len(names) == 47
            assert names >= TOOL_NAMES
            assert len(TOOL_NAMES) == 47

    _run(_body())


def test_mcp__t10_t11_join_ready_and_catalog_47() -> None:
    """T10/T11: catalog 47; emit-setup/fuse-plan/open-setup; recipe join_ready; feedback tool."""

    async def _body() -> None:
        server = build_server()
        async with Client(server) as client:
            listed = await client.list_tools()
            by_name = {t.name: t for t in listed.tools}
            assert len(by_name) == 47
            assert "mesh_proportion_blockout_emit_setup" in by_name
            assert "mesh_proportion_blockout_fuse_plan" in by_name
            assert "mesh_proportion_blockout_feedback" in by_name
            assert "mesh_proportion_blockout_open_setup" in by_name
            recipe_tool = by_name["mesh_proportion_blockout_recipe"]
            schema = (
                getattr(recipe_tool, "input_schema", None)
                or getattr(recipe_tool, "inputSchema", None)
                or {}
            )
            props = schema.get("properties") or {}
            assert "join_ready" in props

    _run(_body())


def test_mcp__proportion_template_smoke(tmp_path: Path) -> None:
    async def _body() -> None:
        server = build_server(tmp_path)
        async with Client(server) as client:
            result = await client.call_tool(
                "mesh_proportion_template",
                {"out": "landmarks_assist.json"},
            )
            assert result.is_error is False
            assert result.structured_content is not None
            assert result.structured_content.get("ok") is True
            path = Path(result.structured_content["path"])
            assert path.is_file()
            # honesty / N6 not a mesh success claim
            assert "assist" in result.structured_content

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


def test_parity_cli_vs_mcp_ingest_triage(
    solid_cylinder_stl: Path,
    tmp_path: Path,
) -> None:
    """Same STL via CLI and MCP → same mesh_id + matching diagnostics (DoD 10)."""
    import json

    from typer.testing import CliRunner

    from meshops.cli import app

    work_cli = tmp_path / "work_cli"
    work_mcp = tmp_path / "work_mcp"
    work_cli.mkdir()
    work_mcp.mkdir()

    runner = CliRunner()
    r_ing = runner.invoke(
        app,
        [
            "ingest",
            "--path",
            str(solid_cylinder_stl),
            "--work-root",
            str(work_cli),
            "--json",
        ],
    )
    assert r_ing.exit_code == 0, r_ing.stdout + r_ing.stderr
    cli_ingest = json.loads(r_ing.stdout)
    cli_mesh_id = cli_ingest["mesh_id"]

    r_tri = runner.invoke(
        app,
        [
            "triage",
            "--mesh-id",
            cli_mesh_id,
            "--work-root",
            str(work_cli),
            "--json",
        ],
    )
    assert r_tri.exit_code == 0, r_tri.stdout + r_tri.stderr
    cli_triage = json.loads(r_tri.stdout)
    cli_diag = cli_triage["diagnostics"]

    async def _body() -> tuple[str, dict[str, Any]]:
        server = build_server(work_root=work_mcp)
        async with Client(server) as client:
            ing = await client.call_tool("mesh_ingest", {"path": str(solid_cylinder_stl)})
            assert ing.is_error is False
            assert ing.structured_content is not None
            mesh_id = str(ing.structured_content["mesh_id"])
            tri = await client.call_tool("mesh_triage", {"mesh_id": mesh_id})
            assert tri.is_error is False
            assert tri.structured_content is not None
            diag = tri.structured_content["diagnostics"]
            assert isinstance(diag, dict)
            return mesh_id, diag

    mcp_id, mcp_diag = _run(_body())
    assert mcp_id == cli_mesh_id
    assert mcp_diag["schema_version"] == cli_diag["schema_version"]
    assert mcp_diag["mesh_id"] == cli_diag["mesh_id"]
    assert mcp_diag["sheet_score"]["score"] == cli_diag["sheet_score"]["score"]
    assert (work_cli / cli_mesh_id / "diagnostics.json").is_file()
    assert (work_mcp / mcp_id / "diagnostics.json").is_file()


def test_accept_candidate_require_slice_attaches_hook(
    solid_cylinder_stl: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """require_slice=True must attach make_orca_hook (Codex P1), not leave slice_hook=None."""
    from meshops.mcp import tools as mcp_tools

    work = tmp_path / "work"
    work.mkdir()
    # Fake Orca present
    fake_orca = tmp_path / "orca-slicer.exe"
    fake_orca.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "meshops.slice.find_orca",
        lambda require=False: fake_orca,
    )
    hooks: list[Any] = []

    def _capture_hook(**kwargs: Any) -> Any:
        hooks.append(kwargs)

        # Return a callable-compatible stub that always fails closed without invoking Orca
        def _hook(**_kw: Any) -> Any:
            from meshops.acceptance.models import SliceAcceptResult

            return SliceAcceptResult(
                status="fail",
                error_code="test_stub",
                messages=["stub"],
                metrics={},
            )

        return _hook

    monkeypatch.setattr("meshops.slice.make_orca_hook", _capture_hook)

    # accept will raise because hook returns fail — we only care that hook was built
    with pytest.raises(RuntimeError):
        mcp_tools.mesh_accept_candidate(
            work,
            baseline_path=str(solid_cylinder_stl),
            candidate_path=str(solid_cylinder_stl),
            require_views=False,
            require_slice=True,
        )
    assert hooks, "make_orca_hook must be called when require_slice=True and Orca present"
    assert "work_root" in hooks[0]


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


def test_design_organic_api_gate_closed_is_error(tmp_path: Path) -> None:
    """Closed plateau raises → MCP is_error path (R1)."""
    import json

    work = tmp_path / "work"
    work.mkdir()
    closed = {
        "schema_version": "1.0.0",
        "session_id": "o9ea985cbc14",
        "reason": "agent metaball loop plateaued without hero quality structure",
        "pass_count": 1,
        "max_passes": 8,
        "criteria_met": ["min_one_pass"],
        "created_at": "2026-08-01T00:00:00+00:00",
        "allows_hosted_fallback": False,
    }
    plateau = tmp_path / "plateau.json"
    plateau.write_text(json.dumps(closed), encoding="utf-8")
    fixtures = Path(__file__).resolve().parent / "fixtures" / "hosted" / "views"
    views = [str(fixtures / "front.png"), str(fixtures / "left.png")]

    async def _body() -> None:
        server = build_server(work_root=work)
        async with Client(server) as client:
            result = await client.call_tool(
                "design_organic_api",
                {
                    "justify": "Agent plateaued; request multi-view hosted structure regen.",
                    "plateau": str(plateau),
                    "provider": "mock",
                    "views": views,
                    "views_from": "explicit",
                },
            )
            assert result.is_error is True

    _run(_body())


def test_design_organic_api_mock_success(tmp_path: Path) -> None:
    """Mock provider e2e via MCP (offline)."""
    import json

    work = tmp_path / "work"
    work.mkdir()
    session_id = "o9ea985cbc14"
    organic = tmp_path / "sessions" / session_id / "organic"
    views_dir = organic / "passes" / "p001_simple_bust" / "views"
    views_dir.mkdir(parents=True)
    mini = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    for key in ("front", "left", "three_quarter"):
        (views_dir / f"{key}.png").write_bytes(mini)
    open_plateau = {
        "schema_version": "1.0.0",
        "session_id": session_id,
        "reason": "agent metaball loop plateaued without hero quality structure",
        "pass_count": 1,
        "max_passes": 8,
        "criteria_met": [
            "min_one_pass",
            "max_passes_or_reason",
            "all_passes_have_views",
            "status_plateau",
        ],
        "created_at": "2026-08-01T00:00:00+00:00",
        "allows_hosted_fallback": True,
    }
    plateau = organic / "plateau.json"
    plateau.write_text(json.dumps(open_plateau), encoding="utf-8")
    (organic / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "session_id": session_id,
                "prompt": "test figurine",
                "status": "plateau",
                "passes": ["p001_simple_bust"],
                "max_passes": 8,
            }
        ),
        encoding="utf-8",
    )

    async def _body() -> None:
        server = build_server(work_root=work)
        async with Client(server) as client:
            result = await client.call_tool(
                "design_organic_api",
                {
                    "justify": "Agent plateaued; request multi-view hosted structure regen.",
                    "plateau": str(plateau),
                    "provider": "mock",
                },
            )
            assert result.is_error is False, result
            assert result.structured_content is not None
            assert result.structured_content.get("ok") is True
            assert result.structured_content.get("mesh_id")
            assert result.structured_content.get("provider") == "mock"

    _run(_body())
