"""Atomic revision store tests."""

from __future__ import annotations

from pathlib import Path

from meshops.guards.models import GuardResult
from meshops.ingest.pipeline import ingest_stl
from meshops.jobstore.paths import JobPaths, content_sha256, ensure_job_layout
from meshops.revs.models import RevManifest
from meshops.revs.store import allocate_rev, fail_rev, promote_rev, write_manifest


def _dummy_guard(ok: bool = True) -> GuardResult:
    return GuardResult(ok=ok, failed=[] if ok else ["face_floor"], messages=["test"])


def test_revs_dir_created_by_layout(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)
    assert paths.revs_dir.is_dir()


def test_revs__atomic_fail__failed_prefix(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)
    ensure_job_layout(paths)
    alloc = allocate_rev(paths, "t1_clean")
    assert alloc.tmp_dir.is_dir()
    assert alloc.tmp_dir.name.startswith(".tmp_")
    # Write a fake mesh
    alloc.mesh_path.write_bytes(b"solid fake")
    man = RevManifest(
        rev_id=alloc.rev_id,
        parent_rev=None,
        recipe_id="t1_clean",
        created_at="2026-01-01T00:00:00+00:00",
        ok=False,
        guard_result=_dummy_guard(ok=False),
        triage_class="T1_topology",
        mesh_path=f"revs/failed_{alloc.rev_id}/mesh.stl",
        error="guard fail",
    )
    failed = fail_rev(alloc, man)
    assert failed.name.startswith("failed_")
    assert failed.is_dir()
    assert not alloc.success_dir.exists()
    assert not alloc.tmp_dir.exists()
    assert (failed / "meta.json").is_file()


def test_revs__promote_success(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)
    alloc = allocate_rev(paths, "t1_clean")
    alloc.mesh_path.write_bytes(b"solid ok")
    man = RevManifest(
        rev_id=alloc.rev_id,
        parent_rev=None,
        recipe_id="t1_clean",
        created_at="2026-01-01T00:00:00+00:00",
        ok=True,
        guard_result=_dummy_guard(ok=True),
        triage_class="T1_topology",
        mesh_path=f"revs/{alloc.rev_id}/mesh.stl",
        n_faces=10,
        n_vertices=10,
        file_size_bytes=8,
    )
    write_manifest(alloc, man)
    dest = promote_rev(alloc)
    assert dest == alloc.success_dir
    assert dest.is_dir()
    assert not alloc.tmp_dir.exists()
    assert (dest / "mesh.stl").is_file()
    assert (dest / "meta.json").is_file()


def test_revs__original_hash__unchanged(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)
    before = content_sha256(paths.original_stl)
    alloc = allocate_rev(paths, "t1_clean")
    alloc.mesh_path.write_bytes(b"x")
    man = RevManifest(
        rev_id=alloc.rev_id,
        parent_rev=None,
        recipe_id="t1_clean",
        created_at="2026-01-01T00:00:00+00:00",
        ok=True,
        guard_result=_dummy_guard(ok=True),
        triage_class="none",
        mesh_path=f"revs/{alloc.rev_id}/mesh.stl",
    )
    write_manifest(alloc, man)
    promote_rev(alloc)
    after = content_sha256(paths.original_stl)
    assert before == after
