"""Organic pass runner unit tests with mocked Blender (track 0006)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meshops.organic import OrganicError, create_session, load_session
from meshops.organic.evidence import write_stub_pass_views
from meshops.organic.models import REQUIRED_VIEW_KEYS as MODEL_KEYS
from meshops.organic.pass_runner import (
    extract_trace_messages,
    resolve_source_stl,
    run_pass,
)
from meshops.organic.session import save_manifest


def _mini_stl(path: Path) -> None:
    """Write a tiny ASCII STL that trimesh can load if needed."""
    path.write_text(
        "solid meshops\n"
        "  facet normal 0 0 1\n"
        "    outer loop\n"
        "      vertex 0 0 0\n"
        "      vertex 1 0 0\n"
        "      vertex 0 1 0\n"
        "    endloop\n"
        "  endfacet\n"
        "endsolid meshops\n",
        encoding="utf-8",
    )


def test_extract_trace_messages() -> None:
    stderr = "Traceback...\nRuntimeError: boom\nAttributeError: no SPHERE\n"
    msgs = extract_trace_messages("", stderr)
    assert any("RuntimeError" in m for m in msgs)
    assert any("AttributeError" in m for m in msgs)


def test_required_view_keys() -> None:
    assert set(MODEL_KEYS) == {
        "front",
        "left",
        "three_quarter",
        "three_quarter_depth",
    }


def test_from_mesh_source_token_and_abs(tmp_path: Path) -> None:
    m = create_session("source resolve test prompt", work_root=tmp_path, session_id="o50000ce0001")
    paths, manifest = load_session(m.session_id, work_root=tmp_path)
    pass_id = "p001_simple_bust"
    pdir = paths.pass_dir(pass_id)
    pdir.mkdir(parents=True)
    mesh = pdir / "mesh.stl"
    _mini_stl(mesh)
    manifest.passes.append(pass_id)
    save_manifest(paths, manifest)

    resolved = resolve_source_stl("p001", paths=paths, pass_names=manifest.passes)
    assert resolved == mesh.resolve()

    resolved2 = resolve_source_stl("p001_simple_bust", paths=paths, pass_names=manifest.passes)
    assert resolved2 == mesh.resolve()

    abs_resolved = resolve_source_stl(str(mesh.resolve()), paths=paths, pass_names=manifest.passes)
    assert abs_resolved == mesh.resolve()

    with pytest.raises(OrganicError) as ei:
        resolve_source_stl("passes/p001/mesh.stl", paths=paths, pass_names=manifest.passes)
    assert ei.value.code == "invalid_params"

    with pytest.raises(OrganicError) as ei:
        resolve_source_stl("p099", paths=paths, pass_names=manifest.passes)
    assert ei.value.code == "invalid_params"


def test_from_mesh_finalized_before_resolve(tmp_path: Path) -> None:
    m = create_session(
        "finalized blocks from_mesh",
        work_root=tmp_path,
        session_id="of1a11ced001",
    )
    paths, manifest = load_session(m.session_id, work_root=tmp_path)
    manifest.status = "finalized"
    save_manifest(paths, manifest)

    with pytest.raises(OrganicError) as ei:
        run_pass(
            m.session_id,
            recipe="from_mesh",
            params={"source_stl": "p001"},
            work_root=tmp_path,
        )
    assert ei.value.code == "session_finalized"


def test_max_passes_exceeded(tmp_path: Path) -> None:
    m = create_session(
        "max passes test",
        work_root=tmp_path,
        session_id="oa0a55e50001",
        max_passes=2,
    )
    paths, manifest = load_session(m.session_id, work_root=tmp_path)
    manifest.passes = ["p001_simple_bust", "p002_simple_bust"]
    save_manifest(paths, manifest)

    with pytest.raises(OrganicError) as ei:
        run_pass(m.session_id, work_root=tmp_path)
    assert ei.value.code == "max_passes_exceeded"


def test_failed_pass_not_in_manifest(tmp_path: Path) -> None:
    m = create_session(
        "failed pass rename test",
        work_root=tmp_path,
        session_id="ofa11ed0a501",
    )

    def _boom(*_a, **_k):
        from meshops.escalate.errors import EscalateError

        raise EscalateError("no blender", code="blender_missing")

    with patch("meshops.organic.pass_runner.find_blender", side_effect=_boom):
        with pytest.raises(OrganicError) as ei:
            run_pass(m.session_id, work_root=tmp_path)
        assert ei.value.code == "blender_not_found"

    paths, manifest = load_session(m.session_id, work_root=tmp_path)
    assert manifest.passes == []
    failed = list(paths.organic_dir.glob("failed_p*"))
    assert len(failed) == 1
    # failed_* is sibling of passes/, not nested under it
    assert failed[0].parent == paths.organic_dir
    assert not (paths.passes_dir / failed[0].name).exists()


def test_unexpected_exception_renames_to_failed(tmp_path: Path) -> None:
    """Non-OrganicError inside run_pass still renames to failed_* (no orphan success dir)."""
    m = create_session(
        "unexpected exception rename test",
        work_root=tmp_path,
        session_id="o0e0ce000001",
    )

    with (
        patch("meshops.organic.pass_runner.find_blender", return_value=Path("blender.exe")),
        patch("meshops.organic.pass_runner.require_blender_52", return_value="5.2.0"),
        patch(
            "meshops.organic.pass_runner.subprocess.run",
            side_effect=RuntimeError("disk full mid-recipe"),
        ),
    ):
        with pytest.raises(OrganicError) as ei:
            run_pass(m.session_id, work_root=tmp_path)
        assert ei.value.code == "blender_failed"
        assert "unexpected" in str(ei.value).lower() or "disk full" in str(ei.value).lower()

    paths, manifest = load_session(m.session_id, work_root=tmp_path)
    assert manifest.passes == []
    failed = list(paths.organic_dir.glob("failed_p*"))
    assert len(failed) == 1
    assert failed[0].parent == paths.organic_dir
    # No in-progress pass dir left under passes/
    leftovers = (
        [p for p in paths.passes_dir.iterdir() if p.is_dir()] if paths.passes_dir.is_dir() else []
    )
    assert leftovers == []


def test_pass_success_mock_blender(tmp_path: Path) -> None:
    m = create_session(
        "mock blender success pass",
        work_root=tmp_path,
        session_id="ob0c0b1e0d01",
    )

    def fake_run(cmd, **kwargs):
        # --out DIR is after --
        out_dir = None
        if "--" in cmd:
            idx = cmd.index("--")
            rest = cmd[idx + 1 :]
            if "--out" in rest:
                out_dir = Path(rest[rest.index("--out") + 1])
        assert out_dir is not None
        partial = out_dir / "mesh.stl.partial"
        _mini_stl(partial)
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = f"meshops_organic_ok path={out_dir / 'mesh.stl'}\n"
        proc.stderr = ""
        return proc

    with (
        patch("meshops.organic.pass_runner.find_blender", return_value=Path("blender.exe")),
        patch("meshops.organic.pass_runner.require_blender_52", return_value="5.2.0"),
        patch("meshops.organic.pass_runner.subprocess.run", side_effect=fake_run),
        patch(
            "meshops.organic.pass_runner.render_pass_views",
            side_effect=lambda mesh, views, force_stub=False: (
                write_stub_pass_views(Path(views)),
                "stub",
                ["views_stub_used"],
            ),
        ),
    ):
        result = run_pass(m.session_id, work_root=tmp_path)

    assert result.ok is True
    assert result.returncode == 0
    assert result.duration_s is not None
    assert result.blender_version == "5.2.0"
    assert result.view_kind == "stub"
    assert result.mesh_path is not None
    assert result.mesh_path.is_file()
    pass_json = result.mesh_path.parent / "pass.json"
    data = json.loads(pass_json.read_text(encoding="utf-8"))
    assert "returncode" in data
    assert "duration_s" in data
    assert data["returncode"] == 0

    paths, manifest = load_session(m.session_id, work_root=tmp_path)
    assert len(manifest.passes) == 1
    assert paths.session_report_md.is_file()


def test_pass_no_views_fails(tmp_path: Path) -> None:
    m = create_session(
        "no views fail test",
        work_root=tmp_path,
        session_id="o0f1e0000001",
    )

    def fake_run(cmd, **kwargs):
        out_dir = Path(cmd[cmd.index("--out") + 1])
        _mini_stl(out_dir / "mesh.stl.partial")
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "meshops_organic_ok path=x\n"
        proc.stderr = ""
        return proc

    with (
        patch("meshops.organic.pass_runner.find_blender", return_value=Path("blender.exe")),
        patch("meshops.organic.pass_runner.require_blender_52", return_value="5.2.0"),
        patch("meshops.organic.pass_runner.subprocess.run", side_effect=fake_run),
        patch(
            "meshops.organic.pass_runner.render_pass_views",
            return_value=({}, "f3d", []),
        ),
    ):
        with pytest.raises(OrganicError) as ei:
            run_pass(m.session_id, work_root=tmp_path)
        assert ei.value.code == "pass_no_views"

    _, manifest = load_session(m.session_id, work_root=tmp_path)
    assert manifest.passes == []


def test_atomic_rename_path(tmp_path: Path) -> None:
    from meshops.organic.pass_runner import _atomic_promote_mesh

    partial = tmp_path / "mesh.stl.partial"
    final = tmp_path / "mesh.stl"
    _mini_stl(partial)
    _atomic_promote_mesh(partial, final)
    assert final.is_file()
    assert not partial.exists()

    with pytest.raises(OrganicError) as ei:
        _atomic_promote_mesh(tmp_path / "missing.partial", final)
    assert ei.value.code == "pass_no_mesh"


def test_no_network_imports_under_organic() -> None:
    """Grep-style: organic package must not import network clients."""
    root = Path("src/meshops/organic")
    forbidden = ("requests", "httpx", "urllib.request", "aiohttp", "openai", "anthropic")
    for py in root.rglob("*.py"):
        if "scripts" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, f"{py} contains {tok}"
