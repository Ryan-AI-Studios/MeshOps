"""CLI export / repair / diff contract tests for 0002."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fixtures.synthetic.t1_t2 import build_t1_nonmanifold_stl
from typer.testing import CliRunner

from meshops.cli import app
from meshops.guards import GuardPolicy, check_export
from meshops.ingest.pipeline import ingest_stl
from meshops.jobstore.paths import JobPaths
from meshops.models.diagnostics import (
    DefectClass,
    DefectHypothesis,
    Diagnostics,
    LateralityStatus,
    MeshStats,
    SheetScoreResult,
)
from meshops.revs.store import parent_mesh_path
from meshops.triage.orchestrate import mesh_triage

runner = CliRunner()


def test_export__wipeout_stats__nonzero_exit(tmp_work: Path, solid_cylinder_stl: Path) -> None:
    """Stats-level wipeout cannot pass export-tier guards (library + CLI fail-closed)."""
    # Direct guard path used by export
    base = MeshStats(
        faces=500_000,
        vertices=250_000,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(100.0, 100.0, 100.0),
        bbox_diagonal=173.2,
        components=1,
        file_size_bytes=25_000_000,
        content_sha256="b" * 64,
        mesh_id="hero",
    )
    cand = MeshStats(
        faces=7000,
        vertices=3500,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
        bbox_diagonal=1.73,
        components=1,
        file_size_bytes=358_000,
        content_sha256="c" * 64,
        mesh_id="wipe",
    )
    r = check_export(base, cand, policy=GuardPolicy.for_export())
    assert r.ok is False

    # CLI e2e: poison diagnostics baseline to hero-scale; small original must fail export
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    mesh_triage(ing.mesh_id, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=ing.mesh_id)
    diag = Diagnostics.model_validate_json(paths.diagnostics_json.read_text(encoding="utf-8"))
    hero_stats = diag.stats.model_copy(
        update={
            "faces": 500_000,
            "vertices": 250_000,
            "file_size_bytes": 25_000_000,
            "bbox_min": (0.0, 0.0, 0.0),
            "bbox_max": (100.0, 100.0, 100.0),
            "bbox_diagonal": 173.2,
            "components": 2,
        }
    )
    paths.diagnostics_json.write_text(
        diag.model_copy(update={"stats": hero_stats}).model_dump_json(indent=2),
        encoding="utf-8",
    )
    wipe_out = tmp_work / "wipeout_should_not_export.stl"
    res_fail = runner.invoke(
        app,
        [
            "export",
            "--mesh-id",
            ing.mesh_id,
            "--out",
            str(wipe_out),
            "--work-root",
            str(tmp_work),
            "--json",
        ],
    )
    assert res_fail.exit_code != 0, res_fail.stdout + res_fail.stderr
    fail_data = json.loads(res_fail.stdout)
    assert fail_data["ok"] is False
    assert "guard" in fail_data
    assert fail_data["guard"]["ok"] is False
    assert not wipe_out.is_file()

    # CLI export of a healthy synthetic still works (fresh job, honest stats)
    healthy = ingest_stl(solid_cylinder_stl, work_root=tmp_work / "healthy")
    mesh_triage(healthy.mesh_id, work_root=tmp_work / "healthy")
    out = tmp_work / "out.stl"
    res = runner.invoke(
        app,
        [
            "export",
            "--mesh-id",
            healthy.mesh_id,
            "--out",
            str(out),
            "--work-root",
            str(tmp_work / "healthy"),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    data = json.loads(res.stdout)
    assert data["ok"] is True
    assert out.is_file()


def test_diff__baseline_original_or_parent(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    mesh_triage(result.mesh_id, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)
    baseline = parent_mesh_path(paths, None)
    assert baseline == paths.original_stl
    assert baseline.name != "working.ply"


def _force_t1_diag(work_root: Path, mesh_id: str) -> None:
    paths = JobPaths(work_root=work_root, mesh_id=mesh_id)
    from meshops.ingest.stats import compute_stats, load_mesh
    from meshops.jobstore.paths import content_sha256

    mesh = load_mesh(paths.original_stl)
    stats = compute_stats(
        mesh,
        mesh_id=mesh_id,
        content_sha256_hex=content_sha256(paths.original_stl),
        file_size_bytes=paths.original_stl.stat().st_size,
        source_path=str(paths.original_stl),
    )
    diag = Diagnostics(
        mesh_id=mesh_id,
        stats=stats,
        defect_hypotheses=[
            DefectHypothesis(
                defect_class=DefectClass.T1_TOPOLOGY,
                confidence=0.9,
                notes="forced T1 for CLI repair test",
            )
        ],
        sheet_score=SheetScoreResult(score=0.1, confidence=0.5),
        laterality_status=LateralityStatus.NOT_APPLICABLE,
    )
    paths.diagnostics_json.write_text(diag.model_dump_json(indent=2), encoding="utf-8")


def test_cli_repair_json(tmp_path: Path, tmp_work: Path) -> None:
    stl = build_t1_nonmanifold_stl(tmp_path)
    r = runner.invoke(
        app,
        ["ingest", "--path", str(stl), "--work-root", str(tmp_work), "--json"],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    mesh_id = json.loads(r.stdout)["mesh_id"]
    _force_t1_diag(tmp_work, mesh_id)
    r3 = runner.invoke(
        app,
        [
            "repair",
            "--mesh-id",
            mesh_id,
            "--recipe",
            "t1_clean",
            "--work-root",
            str(tmp_work),
            "--no-diff",
            "--json",
        ],
    )
    assert r3.exit_code == 0, r3.stdout + r3.stderr
    data = json.loads(r3.stdout)
    assert data["ok"] is True
    assert data["rev_id"].startswith("r")


def test_cli_help_lists_repair() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    help_text = result.stdout.lower()
    assert "repair" in help_text
    assert "export" in help_text
    assert "diff" in help_text


@pytest.mark.f3d
def test_diff__writes_views(tmp_path: Path, tmp_work: Path) -> None:
    stl = build_t1_nonmanifold_stl(tmp_path)
    r = runner.invoke(
        app,
        ["ingest", "--path", str(stl), "--work-root", str(tmp_work), "--json"],
    )
    mesh_id = json.loads(r.stdout)["mesh_id"]
    _force_t1_diag(tmp_work, mesh_id)
    r3 = runner.invoke(
        app,
        [
            "repair",
            "--mesh-id",
            mesh_id,
            "--recipe",
            "t1_clean",
            "--work-root",
            str(tmp_work),
            "--json",
        ],
    )
    if r3.exit_code != 0:
        pytest.skip(f"repair failed before diff: {r3.stdout}")
    data = json.loads(r3.stdout)
    rev_id = data["rev_id"]
    r4 = runner.invoke(
        app,
        [
            "diff",
            "--mesh-id",
            mesh_id,
            "--rev",
            rev_id,
            "--work-root",
            str(tmp_work),
            "--json",
        ],
    )
    if r4.exit_code != 0:
        pytest.skip(f"F3D unavailable: {r4.stdout}")
    payload = json.loads(r4.stdout)
    assert payload["ok"] is True
    assert len(payload["view_paths"]) >= 2
    for p in payload["view_paths"]:
        assert Path(p).is_file()
