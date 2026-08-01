"""Optional live Blender 5.2 organic recipe integration (track 0006)."""

from __future__ import annotations

from pathlib import Path

import pytest

from meshops.escalate.discover import find_blender
from meshops.escalate.errors import EscalateError
from meshops.organic import create_session, finalize_session, load_session, run_pass


def _blender_available() -> bool:
    try:
        return find_blender(require=False) is not None
    except Exception:
        return False


pytestmark = pytest.mark.blender


@pytest.fixture
def require_blender() -> Path:
    try:
        b = find_blender(require=True)
    except EscalateError as exc:
        pytest.skip(f"Blender not available: {exc}")
    assert b is not None
    return b


def test_simple_bust_nonempty_mesh(
    tmp_path: Path,
    require_blender: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MESHOPS_STUB_DIFF", "1")  # views may stub without F3D
    m = create_session(
        "live blender simple bust",
        work_root=tmp_path,
        session_id="ob1e00de0001",
        default_recipe="simple_bust",
    )
    result = run_pass(
        m.session_id,
        recipe="simple_bust",
        params={"resolution": 0.6, "scale_mm": 100.0, "threshold": 0.6},
        work_root=tmp_path,
        force_stub_views=True,
    )
    assert result.ok is True
    assert result.mesh_path is not None
    assert result.mesh_path.is_file()
    assert result.mesh_path.stat().st_size > 100
    assert result.returncode == 0
    assert result.view_paths
    for key in ("front", "left", "three_quarter", "three_quarter_depth"):
        assert key in result.view_paths
        assert Path(result.view_paths[key]).is_file()

    _paths, manifest = load_session(m.session_id, work_root=tmp_path)
    assert len(manifest.passes) == 1

    fin = finalize_session(m.session_id, work_root=tmp_path, accept=False)
    assert fin.ok is True
    assert fin.mesh_id is not None
    job_original = tmp_path / fin.mesh_id / "original.stl"
    assert job_original.is_file()
    assert (tmp_path / fin.mesh_id / "views").is_dir()
