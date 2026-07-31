"""T1/T2 repair orchestrator tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fixtures.synthetic.t1_t2 import (
    build_open_box_large_hole,
    build_solid_box_stl,
    build_t1_nonmanifold_stl,
)

from meshops.ingest.pipeline import ingest_stl
from meshops.jobstore.paths import JobPaths, content_sha256
from meshops.models.diagnostics import (
    DefectClass,
    DefectHypothesis,
    Diagnostics,
    LateralityStatus,
    SheetScoreResult,
)
from meshops.recipes.orchestrate import RepairError, RepairRefuseError, run_repair
from meshops.recipes.pymeshlab_io import RecipeEngineError
from meshops.triage.orchestrate import mesh_triage


def _write_diag(
    paths: JobPaths,
    stats_from_ingest: object,
    *,
    primary: DefectClass | None,
) -> None:
    from meshops.models.diagnostics import MeshStats

    assert isinstance(stats_from_ingest, MeshStats)
    hyps: list[DefectHypothesis] = []
    if primary is not None:
        hyps.append(
            DefectHypothesis(
                defect_class=primary,
                confidence=0.9,
                notes=f"forced {primary} for test",
            )
        )
    diag = Diagnostics(
        mesh_id=paths.mesh_id,
        stats=stats_from_ingest,
        defect_hypotheses=hyps,
        sheet_score=SheetScoreResult(score=0.1, confidence=0.5),
        laterality_status=LateralityStatus.NOT_APPLICABLE,
    )
    paths.diagnostics_json.write_text(diag.model_dump_json(indent=2), encoding="utf-8")


def test_remap_view_paths_after_promote(tmp_path: Path) -> None:
    """Absolute .tmp_* view paths rewrite to promoted rev dir (P2 fix)."""
    from meshops.recipes.orchestrate import _remap_paths_after_promote

    # Use real tmp_path so Path.resolve() is portable (Windows + Linux CI).
    revs = tmp_path / "revs"
    tmp = revs / ".tmp_r001_t1_clean"
    success = revs / "r001_t1_clean"
    (tmp / "views").mkdir(parents=True)
    views = [
        str(tmp / "views" / "front_before.png"),
        str(tmp / "views" / "front_after.png"),
    ]
    fixed = _remap_paths_after_promote(views, from_root=tmp, to_root=success)
    assert fixed == [
        str((success / "views" / "front_before.png").resolve()),
        str((success / "views" / "front_after.png").resolve()),
    ]
    assert ".tmp_" not in fixed[0]


def test_repair__t1_unify_off_synthetic__guards_pass(tmp_path: Path, tmp_work: Path) -> None:
    """T1 on unify-off STL path (honest non-manifold; recipe loads unify_vertices=False)."""
    stl = build_t1_nonmanifold_stl(tmp_path)
    result = ingest_stl(stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)
    _write_diag(paths, result.stats, primary=DefectClass.T1_TOPOLOGY)
    before = content_sha256(result.original_path)

    out = run_repair(result.mesh_id, "t1_clean", work_root=tmp_work, no_diff=True)
    assert out.ok is True
    assert out.rev_id is not None
    assert out.rev_id.startswith("r")
    assert out.manifest is not None
    assert out.manifest.ok is True
    assert out.manifest.mesh_format == "stl_binary"
    assert out.manifest.guard_result.ok is True
    # DoD-4/8: success always carries on-disk view_paths (stubs OK under --no-diff)
    assert out.manifest.view_paths
    assert all(Path(p).is_file() for p in out.manifest.view_paths)
    assert not any(".tmp_" in p for p in out.manifest.view_paths)
    rev_dir = Path(out.rev_dir or "")
    assert rev_dir.is_dir()
    assert not rev_dir.name.startswith("failed_")
    assert (rev_dir / "mesh.stl").is_file()
    assert content_sha256(result.original_path) == before


def test_repair__t3_primary__refuse(arm_sheet_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(arm_sheet_stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)
    _write_diag(paths, result.stats, primary=DefectClass.T3_SHEET)
    with pytest.raises(RepairRefuseError) as ei:
        run_repair(result.mesh_id, "t1_clean", work_root=tmp_work, no_diff=True)
    assert ei.value.code == "refused_class"
    # Refuse happens before allocate — no promoted r00N_* success dir
    if paths.revs_dir.is_dir():
        for child in paths.revs_dir.iterdir():
            if (
                child.is_dir()
                and child.name.startswith("r")
                and not child.name.startswith(("failed_", ".tmp_"))
            ):
                pytest.fail(f"unexpected promoted rev on refuse: {child}")


def test_repair__unknown_recipe__refuse(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    mesh_triage(result.mesh_id, work_root=tmp_work)
    with pytest.raises(RepairRefuseError) as ei:
        run_repair(result.mesh_id, "voxel_remesh_all", work_root=tmp_work, no_diff=True)
    assert ei.value.code in {"unknown_recipe", "never_recipe"}


def test_repair__pymeshlab_exception__failed_rev(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    mesh_triage(result.mesh_id, work_root=tmp_work)

    def boom(*_a: object, **_k: object) -> dict[str, object]:
        raise RecipeEngineError("simulated PyMeshLabException")

    with (
        patch("meshops.recipes.orchestrate.get_recipe", return_value=boom),
        pytest.raises(RepairError) as ei,
    ):
        run_repair(result.mesh_id, "t1_clean", work_root=tmp_work, no_diff=True)
    assert ei.value.rev_dir is not None
    failed = Path(ei.value.rev_dir)
    assert failed.name.startswith("failed_")
    assert (failed / "meta.json").is_file()


def test_repair__large_hole__not_false_success(tmp_path: Path, tmp_work: Path) -> None:
    stl = build_open_box_large_hole(tmp_path)
    result = ingest_stl(stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)
    _write_diag(paths, result.stats, primary=DefectClass.T2_PRINTABILITY)
    with pytest.raises(RepairError) as ei:
        run_repair(
            result.mesh_id,
            "t2_close_small_holes",
            work_root=tmp_work,
            no_diff=True,
        )
    assert ei.value.rev_dir is not None
    assert Path(ei.value.rev_dir).name.startswith("failed_")
    assert "hole" in str(ei.value).lower() or "boundary" in str(ei.value).lower()


def test_repair__missing_diagnostics__refuse(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    with pytest.raises(RepairRefuseError) as ei:
        run_repair(result.mesh_id, "t1_clean", work_root=tmp_work, no_diff=True)
    assert ei.value.code == "missing_diagnostics"


def test_repair__t2_smooth_ok(tmp_path: Path, tmp_work: Path) -> None:
    stl = build_solid_box_stl(tmp_path)
    result = ingest_stl(stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)
    _write_diag(paths, result.stats, primary=DefectClass.T2_PRINTABILITY)
    out = run_repair(result.mesh_id, "t2_smooth_spikes", work_root=tmp_work, no_diff=True)
    assert out.ok is True
    assert out.manifest is not None
    assert out.manifest.recipe_id == "t2_smooth_spikes"
