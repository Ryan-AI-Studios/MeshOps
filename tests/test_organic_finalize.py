"""Finalize always triage + job views on new mesh_id (B12)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import trimesh

from meshops.jobstore.paths import JobPaths
from meshops.organic import OrganicError, create_session, finalize_session, load_session
from meshops.organic.models import REQUIRED_VIEW_KEYS
from meshops.organic.session import save_manifest


def _write_real_stl(path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=20.0)
    mesh.export(path)


def _seed_successful_pass(tmp_path: Path, session_id: str = "of1a11ce0001") -> str:
    m = create_session(
        "finalize pipeline test prompt",
        work_root=tmp_path,
        session_id=session_id,
    )
    paths, manifest = load_session(m.session_id, work_root=tmp_path)
    pass_id = "p001_simple_bust"
    pdir = paths.pass_dir(pass_id)
    views = pdir / "views"
    views.mkdir(parents=True)
    _write_real_stl(pdir / "mesh.stl")
    mini = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    for key in REQUIRED_VIEW_KEYS:
        (views / f"{key}.png").write_bytes(mini)
    # Session pass views exist — finalize must still create JOB views
    manifest.passes.append(pass_id)
    save_manifest(paths, manifest)
    return m.session_id


def test_finalize_no_pass(tmp_path: Path) -> None:
    m = create_session("no pass finalize", work_root=tmp_path, session_id="oa0000000001")
    with pytest.raises(OrganicError) as ei:
        finalize_session(m.session_id, work_root=tmp_path)
    assert ei.value.code == "finalize_no_pass"


def test_finalize_creates_job_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESHOPS_STUB_DIFF", "1")
    sid = _seed_successful_pass(tmp_path)

    result = finalize_session(sid, work_root=tmp_path, accept=False)
    assert result.ok is True
    assert result.mesh_id is not None
    assert result.job_dir is not None
    assert result.honesty_message is not None
    assert (
        "authored" in result.honesty_message.lower() or "organic" in result.honesty_message.lower()
    )

    job = JobPaths(work_root=tmp_path, mesh_id=result.mesh_id)
    assert job.original_stl.is_file()
    assert job.views_dir.is_dir()
    job_views = list(job.views_dir.glob("*.png"))
    assert len(job_views) >= 1, "B12: job views required (not session pass views alone)"

    # Session pass views must not be the only evidence
    paths, manifest = load_session(sid, work_root=tmp_path)
    assert manifest.status == "finalized"
    assert manifest.final_mesh_id == result.mesh_id
    assert paths.final_stl.is_file()
    assert paths.finalize_json.is_file()

    # Job views dir is under mesh_id, not session organic
    assert result.mesh_id not in str(paths.organic_dir) or result.mesh_id != sid
    assert str(job.views_dir).startswith(str(tmp_path / result.mesh_id))


def test_finalize_accept_for_sculpt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESHOPS_STUB_DIFF", "1")
    sid = _seed_successful_pass(tmp_path, session_id="of1a11ceacc1")

    with patch("meshops.organic.finalize.accept_candidate") as mock_accept:
        from meshops.acceptance.models import AcceptanceResult
        from meshops.guards.models import GuardResult

        mock_accept.return_value = AcceptanceResult(
            ok=True,
            failed=[],
            messages=[],
            guard=GuardResult(
                ok=True,
                failed=[],
                messages=[],
                metrics={},
                policy_tier="sculpt",
            ),
            view_paths=["x.png"],
            views_ok=True,
            view_kind="stub",
            honesty="guards_and_stub_views",
            honesty_message="mechanical only",
            policy_tier="sculpt",
            metrics={},
        )
        result = finalize_session(sid, work_root=tmp_path, accept=True)

    assert result.ok is True
    mock_accept.assert_called_once()
    kwargs = mock_accept.call_args.kwargs
    assert kwargs["policy"].tier == "sculpt"
    assert kwargs["require_views"] is True


def test_render_mesh_to_dir_exists() -> None:
    from meshops.render.f3d_renderer import F3DRenderer

    assert hasattr(F3DRenderer, "render_mesh_to_dir")
    import inspect

    sig = inspect.signature(F3DRenderer.render_mesh_to_dir)
    assert "include_depth_for" in sig.parameters
    assert "camera_names" in sig.parameters
