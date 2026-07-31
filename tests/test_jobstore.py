"""Job store + ingest tests (DoD-1)."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest
import trimesh

from meshops.ingest.pipeline import ingest_stl
from meshops.jobstore.paths import (
    PROXY_FACE_THRESHOLD,
    JobPaths,
    content_sha256,
    mesh_id_from_path,
    set_readonly,
)


def test_mesh_id_is_first_12_hex_of_sha256(solid_cylinder_stl: Path) -> None:
    data = solid_cylinder_stl.read_bytes()
    expected = hashlib.sha256(data).hexdigest()[:12]
    assert mesh_id_from_path(solid_cylinder_stl) == expected
    assert content_sha256(solid_cylinder_stl) == hashlib.sha256(data).hexdigest()


def test_ingest_creates_layout(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)

    assert paths.job_dir.is_dir()
    assert paths.original_stl.is_file()
    assert paths.working_ply.is_file()
    assert paths.views_dir.is_dir()
    assert paths.rois_dir.is_dir()
    assert paths.revs_dir.is_dir()
    assert paths.design_dir.is_dir()
    assert paths.handoff_dir.is_dir()
    assert paths.previews_dir.is_dir()
    assert paths.report_md.is_file()
    assert result.stats.faces > 0
    assert result.stats.mesh_id == result.mesh_id
    assert result.stats.content_sha256.startswith(result.mesh_id)


def test_reingest_same_mesh_id(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    r1 = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    r2 = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    assert r1.mesh_id == r2.mesh_id
    assert r2.reused is True


def test_source_never_overwritten(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    before = solid_cylinder_stl.read_bytes()
    mtime_before = solid_cylinder_stl.stat().st_mtime_ns
    ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    after = solid_cylinder_stl.read_bytes()
    assert before == after
    # Source path identity preserved
    assert solid_cylinder_stl.is_file()
    assert solid_cylinder_stl.stat().st_mtime_ns == mtime_before


def test_original_readonly_best_effort(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    # Assert ingest left original read-only (do not re-apply set_readonly here).
    assert result.original_path.is_file()
    mode = result.original_path.stat().st_mode
    if os.name == "nt":
        import ctypes

        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(result.original_path))  # type: ignore[attr-defined]
        # FILE_ATTRIBUTE_READONLY = 0x01; INVALID_FILE_ATTRIBUTES = -1
        assert attrs != -1
        assert attrs & 0x01, "Windows readonly attribute not set on original.stl"
    else:
        assert not (mode & stat.S_IWRITE), "POSIX user-write bit still set on original.stl"
    # Helper remains available for re-ingest / repair paths
    set_readonly(result.original_path)


def test_proxy_created_when_over_face_threshold(
    solid_cylinder_stl: Path,
    tmp_work: Path,
) -> None:
    """Proxy policy without requiring Rogue2 (DoD-1)."""
    mesh = trimesh.load(solid_cylinder_stl, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    if len(mesh.faces) <= 10:
        pytest.skip("synthetic too small even for lowered threshold")
    # Pass threshold explicitly — default is bound at def time from PROXY_FACE_THRESHOLD.
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work, proxy_face_threshold=10)
    assert result.proxy_path is not None
    assert Path(result.proxy_path).is_file()
    assert result.stats.faces > 10
    assert PROXY_FACE_THRESHOLD == 100_000


def test_stats_topology_fields_present(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    s = result.stats
    assert s.bbox_diagonal > 0
    assert len(s.bbox_min) == 3
    assert s.vertices > 0
    # Topology is best-effort; fields may be null but keys exist via model
    dumped = s.model_dump()
    for key in (
        "is_watertight",
        "is_manifold",
        "boundary_edge_count",
        "euler_characteristic",
    ):
        assert key in dumped
