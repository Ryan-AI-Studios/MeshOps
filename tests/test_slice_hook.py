"""Orca SliceAcceptHook + accept_candidate wiring (DoD-8)."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from meshops.acceptance import accept_candidate
from meshops.acceptance.models import SliceAcceptResult
from meshops.ingest.stats import compute_stats, load_mesh
from meshops.slice.hook import make_orca_hook

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "slice"


def _ok_3mf_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    xml = (FIXTURES / "slice_info_ok.config").read_text(encoding="utf-8")
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Metadata/slice_info.config", xml)
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
    return buf.getvalue()


def test_hook_orca_not_found_no_raise(
    monkeypatch: pytest.MonkeyPatch, solid_cylinder_stl: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr("meshops.slice.hook.find_orca", lambda require=False: None)
    hook = make_orca_hook(mesh_id="x", work_root=tmp_path)
    result = hook(candidate_path=str(solid_cylinder_stl), slice_profile="default")
    assert isinstance(result, SliceAcceptResult)
    assert result.status == "fail"
    assert result.error_code == "orca_not_found"


def test_hook_missing_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "orca.exe"
    fake.write_bytes(b"x")
    monkeypatch.setattr("meshops.slice.hook.find_orca", lambda require=False: fake)
    hook = make_orca_hook(work_root=tmp_path)
    result = hook(candidate_path=None)
    assert result.status == "fail"
    assert result.error_code == "missing_candidate"


def test_hook_mock_slice_into_accept(
    tmp_path: Path, solid_cylinder_stl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "orca.exe"
    fake.write_bytes(b"x")
    monkeypatch.setattr("meshops.slice.hook.find_orca", lambda require=False: fake)
    monkeypatch.setattr("meshops.slice.runner.find_orca", lambda require=False: fake)

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        out_idx = argv.index("--export-3mf")
        Path(argv[out_idx + 1]).write_bytes(_ok_3mf_bytes())
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    hook = make_orca_hook(
        mesh_id="hookmesh0001",
        work_root=tmp_path / "work",
        run_orca_fn=fake_run,
        orca_path=fake,
    )
    from meshops.jobstore.paths import content_sha256

    mesh = load_mesh(solid_cylinder_stl)
    stats = compute_stats(
        mesh,
        content_sha256_hex=content_sha256(solid_cylinder_stl),
        file_size_bytes=solid_cylinder_stl.stat().st_size,
        mesh_id="hookmesh0001",
    )
    pack = accept_candidate(
        stats,
        solid_cylinder_stl,
        require_views=False,
        require_slice=True,
        slice_hook=hook,
    )
    assert pack.slice is not None
    assert pack.slice.status == "pass"
    assert pack.ok is True
    assert "slice_fail" not in pack.failed


def test_hook_rechecks_find_orca_each_call(
    tmp_path: Path, solid_cylinder_stl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def flaky_find(*, require: bool = False) -> Path | None:
        calls["n"] += 1
        return None

    monkeypatch.setattr("meshops.slice.hook.find_orca", flaky_find)
    hook = make_orca_hook(work_root=tmp_path)
    r1 = hook(candidate_path=str(solid_cylinder_stl))
    r2 = hook(candidate_path=str(solid_cylinder_stl))
    assert r1.error_code == "orca_not_found"
    assert r2.error_code == "orca_not_found"
    assert calls["n"] >= 2


def test_require_slice_without_hook_fails() -> None:
    from meshops.models.diagnostics import MeshStats

    s = MeshStats(
        faces=1000,
        vertices=500,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(10.0, 10.0, 10.0),
        bbox_diagonal=17.32,
        components=1,
        is_watertight=True,
        is_volume=True,
        is_manifold=True,
        file_size_bytes=1000,
        content_sha256="b" * 64,
        mesh_id="t",
    )
    pack = accept_candidate(s, s, require_views=False, require_slice=True)
    assert pack.ok is False
    assert "slice_not_configured" in pack.failed
