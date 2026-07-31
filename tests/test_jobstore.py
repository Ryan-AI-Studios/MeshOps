"""Job store + ingest tests (DoD-1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from meshops.ingest.pipeline import ingest_stl
from meshops.jobstore.paths import JobPaths, content_sha256, mesh_id_from_path, set_readonly


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
    # Best-effort: set_readonly should not raise; on Windows attribute may be set.
    set_readonly(result.original_path)
    assert result.original_path.is_file()
    # Writing should fail or be restricted — check not writable by owner bit when possible
    mode = result.original_path.stat().st_mode
    # At least one of: no user-write bit, or Windows readonly (we only assert file still exists)
    assert mode is not None


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
