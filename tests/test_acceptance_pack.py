"""0011 Acceptance Pack — composition, honesty, slice, promote, Rogue2."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from tests.acceptance_helpers import assert_accepted
from typer.testing import CliRunner

from meshops.acceptance import (
    HONESTY_MESSAGE,
    AcceptanceResult,
    GuardPolicy,
    accept_candidate,
    accept_revision,
    check_export,
    promote_working,
)
from meshops.acceptance.models import SliceAcceptResult
from meshops.acceptance.numeric import DEGENERATE_FACE_RATIO_MAX, count_degenerate_faces
from meshops.acceptance.promote import PromoteError
from meshops.cli import app
from meshops.guards.models import GuardResult
from meshops.ingest.pipeline import ingest_stl
from meshops.ingest.stats import compute_stats, load_mesh
from meshops.jobstore.paths import JobPaths, content_sha256
from meshops.models.diagnostics import (
    DefectClass,
    DefectHypothesis,
    Diagnostics,
    LateralityStatus,
    MeshStats,
    SheetScoreResult,
)
from meshops.revs.models import RevManifest
from meshops.revs.store import allocate_rev, fail_rev, promote_rev, write_manifest
from meshops.triage.orchestrate import mesh_triage

runner = CliRunner()

# Minimal valid 1x1 PNG (same pattern as recipes)
_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _stats(
    *,
    faces: int,
    bytes_: int,
    vertices: int | None = None,
    components: int = 1,
    bbox_min: tuple[float, float, float] = (10.0, 20.0, 30.0),
    bbox_max: tuple[float, float, float] = (110.0, 120.0, 130.0),
    is_volume: bool | None = True,
    mesh_id: str = "t",
) -> MeshStats:
    bmin = bbox_min
    bmax = bbox_max
    diag = ((bmax[0] - bmin[0]) ** 2 + (bmax[1] - bmin[1]) ** 2 + (bmax[2] - bmin[2]) ** 2) ** 0.5
    return MeshStats(
        faces=faces,
        vertices=vertices if vertices is not None else faces // 2,
        bbox_min=bmin,
        bbox_max=bmax,
        bbox_diagonal=diag,
        components=components,
        is_watertight=is_volume,
        is_volume=is_volume,
        is_manifold=True,
        file_size_bytes=bytes_,
        content_sha256="a" * 64,
        mesh_id=mesh_id,
    )


def _write_views(dir_path: Path, names: tuple[str, ...] = ("front", "top")) -> list[str]:
    dir_path.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name in names:
        p = dir_path / f"{name}.png"
        p.write_bytes(_MIN_PNG)
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------
# Core accept_candidate
# ---------------------------------------------------------------------------


def test_accept_candidate__pass_guards_only() -> None:
    s = _stats(faces=100_000, bytes_=5_000_000)
    r = accept_candidate(s, s, require_views=False)
    assert_accepted(r, ok=True, honesty="guards_only")
    assert r.honesty_message == HONESTY_MESSAGE
    assert r.schema_version == "1.0.0"
    assert r.guard is not None and r.guard.ok is True


def test_accept_candidate__wipeout_hero_poison() -> None:
    """Hero wipeout cannot ok=True (Difficulty §6)."""
    base = _stats(faces=500_000, bytes_=25_000_000)
    cand = _stats(
        faces=7000,
        bytes_=358_000,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
    )
    r = accept_candidate(base, cand, require_views=False)
    assert_accepted(r, ok=False, honesty="not_accepted")
    assert any(c in r.failed for c in ("wipeout_class", "face_collapse", "face_floor"))
    assert r.guard is not None and r.guard.ok is False


def test_accept_candidate__missing_views(tmp_path: Path) -> None:
    s = _stats(faces=1000, bytes_=50_000)
    r = accept_candidate(s, s, require_views=True, view_paths=[])
    assert_accepted(r, ok=False, failed_contains=["missing_views"])


def test_accept_candidate__empty_view_file(tmp_path: Path) -> None:
    s = _stats(faces=1000, bytes_=50_000)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    r = accept_candidate(s, s, require_views=True, view_paths=[str(empty)])
    assert_accepted(r, ok=False, failed_contains=["empty_views"])


def test_accept_candidate__stub_views_honesty(tmp_path: Path) -> None:
    s = _stats(faces=1000, bytes_=50_000)
    views = _write_views(tmp_path / "views")
    r = accept_candidate(
        s,
        s,
        require_views=True,
        view_paths=views,
        view_notes=["diff_stub_no_diff_flag"],
        allow_stubs=True,
    )
    assert_accepted(r, ok=True, honesty="guards_and_stub_views", view_kind="stub")


def test_accept_candidate__real_views_honesty(tmp_path: Path) -> None:
    s = _stats(faces=1000, bytes_=50_000)
    views = _write_views(tmp_path / "views")
    r = accept_candidate(
        s,
        s,
        require_views=True,
        view_paths=views,
        view_kind="f3d",
    )
    assert_accepted(r, ok=True, honesty="guards_and_views", view_kind="f3d")


def test_accept_candidate__f3d_fallback_notes_are_stub(tmp_path: Path) -> None:
    """F3D-unavailable / empty-result stub notes must not claim guards_and_views (§12)."""
    s = _stats(faces=1000, bytes_=50_000)
    # Minimal valid 1x1 PNG (same bytes as recipe stubs)
    min_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    png = tmp_path / "front_after.png"
    png.write_bytes(min_png)
    for note in (
        "diff_views_empty_result_used_stub",
        "diff_views_unavailable_used_stub: RuntimeError: boom",
    ):
        r = accept_candidate(
            s,
            s,
            require_views=True,
            view_paths=[str(png)],
            view_notes=[note],
            allow_stubs=True,
        )
        assert_accepted(r, ok=True, honesty="guards_and_stub_views", view_kind="stub")


def test_accept_candidate__allow_stubs_false(tmp_path: Path) -> None:
    s = _stats(faces=1000, bytes_=50_000)
    views = _write_views(tmp_path / "views")
    r = accept_candidate(
        s,
        s,
        require_views=True,
        view_paths=views,
        view_kind="stub",
        allow_stubs=False,
    )
    assert_accepted(r, ok=False, failed_contains=["stub_views_disallowed"])


def test_accept_candidate__expected_view_names(tmp_path: Path) -> None:
    s = _stats(faces=1000, bytes_=50_000)
    views = _write_views(tmp_path / "views", names=("front",))
    r = accept_candidate(
        s,
        s,
        require_views=True,
        view_paths=views,
        view_kind="f3d",
        expected_view_names=["front", "three_quarter"],
    )
    assert_accepted(r, ok=False, failed_contains=["missing_camera_angle"])


def test_accept_candidate__require_slice_without_hook() -> None:
    s = _stats(faces=1000, bytes_=50_000)
    r = accept_candidate(s, s, require_views=False, require_slice=True)
    assert_accepted(r, ok=False, failed_contains=["slice_not_configured"])


def test_accept_candidate__mock_hook_fail_even_if_not_required() -> None:
    s = _stats(faces=1000, bytes_=50_000)

    def bad_hook(**_kwargs: object) -> SliceAcceptResult:
        return SliceAcceptResult(status="fail", error_code="filament_zero", messages=["filament 0"])

    r = accept_candidate(
        s,
        s,
        require_views=False,
        require_slice=False,
        slice_hook=bad_hook,
    )
    assert_accepted(r, ok=False, failed_contains=["slice_fail"])
    assert r.slice is not None
    assert r.slice.status == "fail"


def test_accept_candidate__volume_inverted_opt_in() -> None:
    """Volume inverted fails when check_volume_ratio and cand_volume < 0."""
    from meshops.acceptance.numeric import evaluate_volume_ratio

    failed, _msgs, metrics = evaluate_volume_ratio(base_volume=10.0, cand_volume=-1.0)
    assert "volume_inverted" in failed
    assert metrics["pack.volume_ratio"] is None

    s = _stats(faces=1000, bytes_=50_000)
    # Stats-only path: volumes unavailable → no hard-fail even when opted in
    r = accept_candidate(s, s, require_views=False, check_volume_ratio=True)
    assert r.ok is True
    assert r.metrics.get("pack.volume_ratio") is None


def test_accept_candidate__volume_ratio_with_paths(
    solid_cylinder_stl: Path, tmp_path: Path
) -> None:
    """Opt-in volume ratio on real mesh paths loads mesh and reports ratio."""
    # Same mesh against itself → ratio ~1.0
    r = accept_candidate(
        solid_cylinder_stl,
        solid_cylinder_stl,
        require_views=False,
        check_volume_ratio=True,
        policy=GuardPolicy.for_export(),
    )
    assert r.ok is True
    ratio = r.metrics.get("pack.volume_ratio")
    assert ratio is not None
    assert abs(float(ratio) - 1.0) < 0.05


def test_accept_candidate__topology_opt_in_real_api(solid_cylinder_stl: Path) -> None:
    mesh = load_mesh(solid_cylinder_stl)
    n = count_degenerate_faces(mesh)
    assert isinstance(n, int)
    assert n >= 0
    assert DEGENERATE_FACE_RATIO_MAX == 0.05

    r = accept_candidate(
        solid_cylinder_stl,
        solid_cylinder_stl,
        require_views=False,
        check_topology=True,
        policy=GuardPolicy.for_export(),
    )
    assert r.ok is True
    assert "pack.degenerate_faces" in r.metrics
    assert r.metrics["pack.degenerate_faces"] == n


def test_accept_candidate__stats_only_no_load_mesh() -> None:
    """Defaults must not force mesh load on MeshStats path (A2-B2 / DoD-16)."""
    s = _stats(faces=1000, bytes_=50_000)
    with patch("meshops.ingest.stats.load_mesh") as mock_load:
        r = accept_candidate(s, s, require_views=False)
        assert r.ok is True
        mock_load.assert_not_called()


def test_accept_candidate__clothing_cape_unmodified(
    clothing_cape_stl: Path,
) -> None:
    """Unmodified clothing-cape synthetic accepts (N8 — pack introduces no delete)."""
    r = accept_candidate(
        clothing_cape_stl,
        clothing_cape_stl,
        require_views=False,
        policy=GuardPolicy.for_export(),
    )
    assert_accepted(r, ok=True, honesty="guards_only")


def test_import_from_meshops_acceptance() -> None:
    from meshops.acceptance import (
        AcceptanceResult as AR,
    )
    from meshops.acceptance import (
        GuardPolicy as GP,
    )
    from meshops.acceptance import (
        GuardResult as GR,
    )
    from meshops.acceptance import (
        accept_candidate as ac,
    )
    from meshops.acceptance import (
        accept_revision as ar,
    )
    from meshops.acceptance import (
        check_export as ce,
    )

    assert callable(ac) and callable(ar) and callable(ce)
    assert AR is AcceptanceResult
    assert GP is GuardPolicy
    assert GR is GuardResult


# ---------------------------------------------------------------------------
# accept_revision + promote
# ---------------------------------------------------------------------------


def _seed_job_with_rev(
    stl: Path,
    tmp_work: Path,
    *,
    ok: bool = True,
    failed_dir: bool = False,
    notes: list[str] | None = None,
) -> tuple[str, str, JobPaths]:
    """Ingest + triage + synthetic rev (success or failed)."""
    ing = ingest_stl(stl, work_root=tmp_work)
    mesh_triage(ing.mesh_id, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=ing.mesh_id)
    alloc = allocate_rev(paths, "t1_clean")
    # Copy original as candidate mesh
    alloc.mesh_path.write_bytes(paths.original_stl.read_bytes())
    mesh = load_mesh(alloc.mesh_path)
    stats = compute_stats(
        mesh,
        mesh_id=ing.mesh_id,
        content_sha256_hex=content_sha256(alloc.mesh_path),
        file_size_bytes=alloc.mesh_path.stat().st_size,
        source_path=str(alloc.mesh_path),
    )
    guard = check_export(ing.stats, stats, policy=GuardPolicy.for_recipe("t1_clean"))
    note_list = list(notes or ["diff_stub_test"])
    # Placeholder paths; rewritten after promote/fail so absolute paths match final dir.
    final_dir = alloc.failed_dir if (failed_dir or not ok) else alloc.success_dir
    views = [str(final_dir / "views" / f"{name}.png") for name in ("front", "top")]
    manifest = RevManifest(
        rev_id=alloc.rev_id,
        recipe_id="t1_clean",
        created_at="2026-01-01T00:00:00+00:00",
        ok=ok and guard.ok,
        guard_result=(
            guard
            if (ok and guard.ok)
            else GuardResult(
                ok=False,
                failed=["simulated_fail"] if not ok else list(guard.failed),
                metrics=dict(guard.metrics),
                messages=["simulated"] if not ok else list(guard.messages),
                policy_tier="recipe",
            )
        ),
        triage_class="T1_topology",
        mesh_path=f"revs/{alloc.rev_id}/mesh.stl",
        n_faces=stats.faces,
        n_vertices=stats.vertices,
        file_size_bytes=stats.file_size_bytes,
        view_paths=views,
        view_kind="stub",
        notes=note_list,
    )
    write_manifest(alloc, manifest)
    # Materialize views under tmp so rename carries them
    _write_views(alloc.views_dir)
    if failed_dir or not ok:
        fail_rev(alloc, manifest)
        rev_id = f"failed_{alloc.rev_id}"
    else:
        promote_rev(alloc)
        rev_id = alloc.rev_id
        # Rewrite meta with paths under final success dir (already set above)
        (final_dir / "meta.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
    return ing.mesh_id, rev_id, paths


def test_accept_revision__failed_rev(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    mesh_id, rev_id, _paths = _seed_job_with_rev(
        solid_cylinder_stl, tmp_work, ok=False, failed_dir=True
    )
    r = accept_revision(mesh_id, rev_id, work_root=tmp_work, require_views=False)
    assert_accepted(r, ok=False, failed_contains=["failed_rev"])


def test_accept_revision__pass_with_stubs(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    mesh_id, rev_id, _paths = _seed_job_with_rev(solid_cylinder_stl, tmp_work, ok=True)
    r = accept_revision(mesh_id, rev_id, work_root=tmp_work, require_views=True, allow_stubs=True)
    assert_accepted(r, ok=True, honesty="guards_and_stub_views")


def test_promote_working__only_on_ok(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    mesh_id, rev_id, paths = _seed_job_with_rev(solid_cylinder_stl, tmp_work, ok=True)
    original_hash = content_sha256(paths.original_stl)

    out = promote_working(
        mesh_id,
        rev_id,
        work_root=tmp_work,
        require_views=True,
        allow_stubs=True,
    )
    assert out["ok"] is True
    assert paths.working_ply.is_file()
    # Must be real PLY content (not STL bytes under .ply name) — loadable by extension.
    promoted = load_mesh(paths.working_ply)
    assert len(promoted.faces) > 0
    # Header check: PLY magic
    head = paths.working_ply.read_bytes()[:3]
    assert head in {b"ply", b"PLY"}
    manifest_path = paths.job_dir / "working_manifest.json"
    assert manifest_path.is_file()
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert meta["promoted_from_rev"] == rev_id
    assert "content_sha256" in meta
    assert meta["acceptance"]["ok"] is True
    # original never touched
    assert content_sha256(paths.original_stl) == original_hash
    # diagnostics still original identity
    assert paths.diagnostics_json.is_file()


def test_promote_working__refuses_failed(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    mesh_id, rev_id, paths = _seed_job_with_rev(
        solid_cylinder_stl, tmp_work, ok=False, failed_dir=True
    )
    with pytest.raises(PromoteError) as ei:
        promote_working(mesh_id, rev_id, work_root=tmp_work, require_views=False)
    assert ei.value.code in {"not_accepted", "failed_rev"}
    # working_manifest must not exist
    assert not (paths.job_dir / "working_manifest.json").is_file()


def test_promote_working__refuses_preview_only_notes(
    solid_cylinder_stl: Path, tmp_work: Path
) -> None:
    """0004: promote_working hard-refuses revs tagged preview_only (N6)."""
    mesh_id, rev_id, paths = _seed_job_with_rev(
        solid_cylinder_stl,
        tmp_work,
        ok=True,
        notes=["preview_only", "diff_stub_test"],
    )
    with pytest.raises(PromoteError) as ei:
        promote_working(
            mesh_id,
            rev_id,
            work_root=tmp_work,
            require_views=True,
            allow_stubs=True,
        )
    assert ei.value.code == "preview_refuse_promote"
    assert not (paths.job_dir / "working_manifest.json").is_file()


def test_promote_working__refuses_t3_preview_recipe(
    solid_cylinder_stl: Path, tmp_work: Path
) -> None:
    """0004: promote_working hard-refuses recipe_id t3_preview* (N6)."""
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    mesh_triage(ing.mesh_id, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=ing.mesh_id)
    alloc = allocate_rev(paths, "t3_preview")
    alloc.mesh_path.write_bytes(paths.original_stl.read_bytes())
    mesh = load_mesh(alloc.mesh_path)
    stats = compute_stats(
        mesh,
        mesh_id=ing.mesh_id,
        content_sha256_hex=content_sha256(alloc.mesh_path),
        file_size_bytes=alloc.mesh_path.stat().st_size,
        source_path=str(alloc.mesh_path),
    )
    guard = check_export(ing.stats, stats, policy=GuardPolicy.for_export())
    views = [str(alloc.success_dir / "views" / f"{name}.png") for name in ("front", "top")]
    manifest = RevManifest(
        rev_id=alloc.rev_id,
        recipe_id="t3_preview_local",
        created_at="2026-01-01T00:00:00+00:00",
        ok=True,
        guard_result=guard,
        triage_class="T3_escalate",
        mesh_path=f"revs/{alloc.rev_id}/mesh.stl",
        n_faces=stats.faces,
        n_vertices=stats.vertices,
        file_size_bytes=stats.file_size_bytes,
        view_paths=views,
        view_kind="stub",
        notes=["diff_stub_test"],
    )
    write_manifest(alloc, manifest)
    _write_views(alloc.views_dir)
    promote_rev(alloc)
    (alloc.success_dir / "meta.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    with pytest.raises(PromoteError) as ei:
        promote_working(
            ing.mesh_id,
            alloc.rev_id,
            work_root=tmp_work,
            require_views=True,
            allow_stubs=True,
        )
    assert ei.value.code == "preview_refuse_promote"
    assert not (paths.job_dir / "working_manifest.json").is_file()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_accept__json_exit_nonzero_on_wipeout(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    """CLI accept exit != 0 on wipeout/fail (A2-B10)."""
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    mesh_triage(ing.mesh_id, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=ing.mesh_id)
    diag = Diagnostics.model_validate_json(paths.diagnostics_json.read_text(encoding="utf-8"))
    # Poison baseline to hero; self-accept original will wipeout vs tiny mesh
    hero = diag.stats.model_copy(
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
        diag.model_copy(update={"stats": hero}).model_dump_json(indent=2),
        encoding="utf-8",
    )
    res = runner.invoke(
        app,
        [
            "accept",
            "--mesh-id",
            ing.mesh_id,
            "--work-root",
            str(tmp_work),
            "--no-require-views",
            "--json",
        ],
    )
    assert res.exit_code != 0, res.stdout + res.stderr
    data = json.loads(res.stdout)
    assert data["ok"] is False
    assert "acceptance" in data
    assert data["acceptance"]["ok"] is False


def test_cli_accept__pass_and_help(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    help_r = runner.invoke(app, ["--help"])
    assert "accept" in help_r.stdout.lower()

    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    mesh_triage(ing.mesh_id, work_root=tmp_work)
    res = runner.invoke(
        app,
        [
            "accept",
            "--mesh-id",
            ing.mesh_id,
            "--work-root",
            str(tmp_work),
            "--no-require-views",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    data = json.loads(res.stdout)
    assert data["ok"] is True
    assert data["acceptance"]["honesty"] == "guards_only"
    assert data["acceptance"]["honesty_message"] == HONESTY_MESSAGE


def test_cli_accept__promote_working(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    mesh_id, rev_id, paths = _seed_job_with_rev(solid_cylinder_stl, tmp_work, ok=True)
    res = runner.invoke(
        app,
        [
            "accept",
            "--mesh-id",
            mesh_id,
            "--rev",
            rev_id,
            "--work-root",
            str(tmp_work),
            "--allow-stubs",
            "--promote-working",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    data = json.loads(res.stdout)
    assert data["ok"] is True
    assert "promote" in data
    assert paths.working_ply.is_file()
    assert (paths.job_dir / "working_manifest.json").is_file()


# ---------------------------------------------------------------------------
# Repair / export acceptance field
# ---------------------------------------------------------------------------


def test_repair_returns_acceptance(tmp_path: Path, tmp_work: Path) -> None:
    from fixtures.synthetic.t1_t2 import build_t1_nonmanifold_stl

    from meshops.recipes.orchestrate import run_repair

    stl = build_t1_nonmanifold_stl(tmp_path)
    ing = ingest_stl(stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=ing.mesh_id)
    diag = Diagnostics(
        mesh_id=ing.mesh_id,
        stats=ing.stats,
        defect_hypotheses=[
            DefectHypothesis(
                defect_class=DefectClass.T1_TOPOLOGY,
                confidence=0.9,
                notes="forced T1",
            )
        ],
        sheet_score=SheetScoreResult(score=0.1, confidence=0.5),
        laterality_status=LateralityStatus.NOT_APPLICABLE,
    )
    paths.diagnostics_json.write_text(diag.model_dump_json(indent=2), encoding="utf-8")
    out = run_repair(ing.mesh_id, "t1_clean", work_root=tmp_work, no_diff=True)
    assert out.ok is True
    assert out.acceptance is not None
    assert isinstance(out.acceptance, AcceptanceResult)
    assert out.acceptance.ok is True
    assert out.acceptance.honesty in {"guards_and_stub_views", "guards_and_views"}


def test_export_payload_includes_acceptance(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    from meshops.export_guarded import guarded_export

    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    mesh_triage(ing.mesh_id, work_root=tmp_work)
    out = tmp_work / "export_out.stl"
    payload = guarded_export(ing.mesh_id, out, work_root=tmp_work)
    assert payload["ok"] is True
    assert "acceptance" in payload
    acc = payload["acceptance"]
    assert isinstance(acc, dict)
    assert acc["ok"] is True
    assert acc["honesty"] == "guards_only"


# ---------------------------------------------------------------------------
# Rogue2
# ---------------------------------------------------------------------------


@pytest.mark.rogue2
def test_rogue2__self_accept_and_poison(rogue2_stl: Path) -> None:
    """Rogue2 self-accept without F3D tax (DoD-7); poison still fails.

    Single resolve_stats load — avoid multi-load hang/pressure on 676k faces.
    """
    from meshops.guards import resolve_stats

    base = resolve_stats(rogue2_stl, mesh_id="rogue2")
    # Self-accept: same MeshStats both sides — no second disk load
    r = accept_candidate(
        base,
        base,
        require_views=False,
        policy=GuardPolicy.for_export(),
    )
    assert_accepted(r, ok=True, honesty="guards_only")

    poison = _stats(
        faces=7000,
        bytes_=358_000,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
    )
    r2 = accept_candidate(base, poison, require_views=False, policy=GuardPolicy.for_export())
    assert_accepted(r2, ok=False)
    assert any(
        c in r2.failed for c in ("wipeout_class", "face_collapse", "face_floor", "size_floor")
    )
