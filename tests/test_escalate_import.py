"""import-sculpt approve gate + sculpt policy accept (DoD-7, DoD-8)."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from meshops.escalate.errors import EscalateError
from meshops.escalate.import_sculpt import RECIPE_ID, import_sculpt
from meshops.ingest.pipeline import ingest_stl


def test_import_requires_approve(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    with pytest.raises(EscalateError) as ei:
        import_sculpt(
            ing.mesh_id,
            solid_cylinder_stl,
            approve=False,
            work_root=tmp_work,
            no_diff=True,
        )
    assert ei.value.code == "approve_required"


def test_import_happy_path_synthetic(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    # Use same mesh as "sculpt" — should pass export-like floors vs original
    result = import_sculpt(
        ing.mesh_id,
        solid_cylinder_stl,
        approve=True,
        work_root=tmp_work,
        no_diff=True,
    )
    assert result.ok is True
    assert result.rev_id is not None
    assert result.recipe_id == RECIPE_ID
    assert result.acceptance is not None
    assert result.acceptance.ok is True
    assert result.acceptance.policy_tier == "sculpt"
    assert "not_autonomous_hero_fixed" in result.notes
    assert result.extra.get("promoted_to_working") is False

    rev_dir = Path(result.rev_dir)  # type: ignore[arg-type]
    assert rev_dir.is_dir()
    assert (rev_dir / "mesh.stl").is_file()
    assert (rev_dir / "meta.json").is_file()
    # Views required
    assert result.acceptance.view_paths
    # working.ply must still be original ingest working (not auto-promoted sculpt)
    working = tmp_work / ing.mesh_id / "working.ply"
    assert working.is_file()


def test_import_wipeout_fails(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    """Tiny wipeout sculpt fails face/size floors vs original."""
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)

    # Tiny tetrahedron — massive face/size collapse vs cylinder
    tiny = trimesh.creation.icosphere(subdivisions=0, radius=0.01)
    wipe_path = tmp_work / "wipeout_sculpt.stl"
    tiny.export(wipe_path)

    result = import_sculpt(
        ing.mesh_id,
        wipe_path,
        approve=True,
        work_root=tmp_work,
        no_diff=True,
    )
    assert result.ok is False
    assert result.acceptance is not None
    assert result.acceptance.ok is False
    assert result.rev_dir is not None
    assert "failed_" in Path(result.rev_dir).name or result.acceptance.failed


def test_import_preview_path_refused(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    from meshops.escalate.preview_t3 import preview_t3

    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    prev = preview_t3(ing.mesh_id, work_root=tmp_work)
    preview_mesh = prev.preview_dir / "mesh.stl"
    with pytest.raises(EscalateError) as ei:
        import_sculpt(
            ing.mesh_id,
            preview_mesh,
            approve=True,
            work_root=tmp_work,
            no_diff=True,
        )
    assert ei.value.code == "preview_refuse_promote"


def test_import_malformed_mesh_fails_atomically(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    """Malformed sculpt must leave failed_* rev, never .tmp_* orphan (Codex P1)."""
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    bad = tmp_work / "not_a_mesh.stl"
    bad.write_bytes(b"this is not an stl mesh at all!!!")

    with pytest.raises(EscalateError) as ei:
        import_sculpt(
            ing.mesh_id,
            bad,
            approve=True,
            work_root=tmp_work,
            no_diff=True,
        )
    assert ei.value.code == "import_failed"
    revs = tmp_work / ing.mesh_id / "revs"
    assert revs.is_dir()
    # No leftover staging dirs
    tmp_orphans = [p for p in revs.iterdir() if p.name.startswith(".tmp_")]
    assert tmp_orphans == []
    failed = [p for p in revs.iterdir() if p.name.startswith("failed_")]
    assert len(failed) >= 1
    assert (failed[0] / "meta.json").is_file()


def test_import_parent_rev_uses_parent_mesh_baseline(
    solid_cylinder_stl: Path, tmp_work: Path
) -> None:
    """When parent_rev is set, guard baseline is parent mesh not diagnostics original."""
    import shutil
    from datetime import UTC, datetime

    from meshops.guards.models import GuardResult
    from meshops.jobstore.paths import JobPaths
    from meshops.revs.models import RevManifest
    from meshops.revs.store import allocate_rev, promote_rev, write_manifest

    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=ing.mesh_id)

    # Parent rev: copy original as a prior "sculpt" baseline
    alloc = allocate_rev(paths, "prior_sculpt")
    shutil.copy2(solid_cylinder_stl, alloc.mesh_path)
    man = RevManifest(
        rev_id=alloc.rev_id,
        parent_rev=None,
        recipe_id="blender_sculpt_import",
        created_at=datetime.now(UTC).isoformat(),
        ok=True,
        guard_result=GuardResult(ok=True, failed=[], metrics={}, messages=[], policy_tier="sculpt"),
        triage_class="T3_sheet",
        mesh_path=f"revs/{alloc.rev_id}/mesh.stl",
        notes=["test_parent"],
    )
    write_manifest(alloc, man)
    parent_dir = promote_rev(alloc)

    result = import_sculpt(
        ing.mesh_id,
        solid_cylinder_stl,
        approve=True,
        work_root=tmp_work,
        no_diff=True,
        parent_rev=alloc.rev_id,
    )
    assert result.ok is True
    # Manifest records parent_rev
    from meshops.revs.store import load_manifest, resolve_rev_dir

    rev_dir = resolve_rev_dir(paths, result.rev_id)  # type: ignore[arg-type]
    loaded = load_manifest(rev_dir)
    assert loaded.parent_rev == alloc.rev_id
    assert parent_dir.is_dir()
