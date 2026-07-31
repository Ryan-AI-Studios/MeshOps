"""Recipe orchestration: triage gate → copy → recipe → guards → atomic rev.

Hard rules:
  - diagnostics.json required
  - refuse T3/T4 primary and unknown recipes
  - never overwrite original.stl
  - recipe-tier check_export; fail → failed_r00N
  - catch PyMeshLabException / RecipeEngineError → failed rev
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meshops.guards import GuardPolicy, check_export
from meshops.ingest.stats import compute_stats, load_mesh
from meshops.jobstore.paths import JobPaths, content_sha256, ensure_job_layout
from meshops.models.diagnostics import Diagnostics
from meshops.recipes.pymeshlab_io import RecipeEngineError
from meshops.recipes.registry import (
    ALLOWED_PRIMARY_CLASSES,
    NEVER_RECIPE_IDS,
    REFUSED_PRIMARY_CLASSES,
    get_recipe,
    list_recipes,
)
from meshops.revs.models import RecipeResult, RevManifest
from meshops.revs.store import (
    allocate_rev,
    fail_rev,
    parent_mesh_path,
    promote_rev,
    write_manifest,
)


class RepairRefuseError(RuntimeError):
    """Structured refusal (no mutation performed)."""

    def __init__(self, message: str, *, code: str = "refuse") -> None:
        super().__init__(message)
        self.code = code


class RepairError(RuntimeError):
    """Recipe/guard failure after staging (failed rev may exist)."""

    def __init__(
        self,
        message: str,
        *,
        rev_id: str | None = None,
        rev_dir: Path | None = None,
        result: RecipeResult | None = None,
    ) -> None:
        super().__init__(message)
        self.rev_id = rev_id
        self.rev_dir = rev_dir
        self.result = result


def _load_diagnostics(paths: JobPaths) -> Diagnostics:
    if not paths.diagnostics_json.is_file():
        raise RepairRefuseError(
            f"diagnostics.json required before repair (run triage first): {paths.diagnostics_json}",
            code="missing_diagnostics",
        )
    return Diagnostics.model_validate_json(paths.diagnostics_json.read_text(encoding="utf-8"))


def _primary_class(diag: Diagnostics) -> str:
    if not diag.defect_hypotheses:
        return "none"
    # Highest confidence hypothesis
    top = max(diag.defect_hypotheses, key=lambda h: h.confidence)
    return str(top.defect_class.value if hasattr(top.defect_class, "value") else top.defect_class)


def _assert_recipe_allowed(recipe_id: str, primary: str) -> None:
    if recipe_id in NEVER_RECIPE_IDS:
        raise RepairRefuseError(
            f"recipe {recipe_id!r} is permanently refused (Never track)",
            code="never_recipe",
        )
    try:
        get_recipe(recipe_id)
    except KeyError as exc:
        raise RepairRefuseError(str(exc), code="unknown_recipe") from exc

    if primary in REFUSED_PRIMARY_CLASSES:
        raise RepairRefuseError(
            f"primary triage class {primary!r} refused for T1/T2 recipes "
            f"(escalate to later tracks); allowlist primaries="
            f"{sorted(ALLOWED_PRIMARY_CLASSES)}",
            code="refused_class",
        )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# Minimal valid 1x1 PNG (black) for stub evidence when F3D is unavailable.
_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

_STUB_CAMERAS = ("front", "three_quarter", "top")


def _write_stub_views(views_dir: Path) -> list[str]:
    """Write minimal before/after PNGs so mutating success always has view_paths.

    Difficulty §12 / DoD-4/8: success requires evidence paths on disk. Stubs are
    honest placeholders when F3D is unavailable or --no-diff is set; notes mark them.
    """
    views_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name in _STUB_CAMERAS:
        for suffix in ("before", "after"):
            dest = views_dir / f"{name}_{suffix}.png"
            dest.write_bytes(_MIN_PNG)
            paths.append(str(dest))
    return paths


def _prefer_stub_diff() -> bool:
    """True when F3D in-process is unsafe (CI runners) or explicitly requested."""
    import os

    if no_diff_env := os.environ.get("MESHOPS_STUB_DIFF", "").strip().lower():
        return no_diff_env in {"1", "true", "yes", "on"}
    # GitHub Actions / common CI: F3D Engine.create(offscreen) has segfaulted (exit 139).
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def _try_diff_views(
    paths: JobPaths,
    *,
    baseline_mesh: Path,
    candidate_mesh: Path,
    views_dir: Path,
    no_diff: bool,
) -> tuple[list[str], list[str]]:
    """Produce rev view evidence. Prefer F3D; always return non-empty paths on success path.

    --no-diff skips F3D and writes stub PNGs (explicit headless opt-out).
    CI defaults to stubs (F3D offscreen has crashed GH runners).
    F3D failure also falls back to stubs with a note (never silent empty success).
    """
    notes: list[str] = []
    if no_diff:
        notes.append("diff_stub_no_diff_flag")
        return _write_stub_views(views_dir), notes
    if _prefer_stub_diff():
        notes.append("diff_stub_ci_or_MESHOPS_STUB_DIFF")
        return _write_stub_views(views_dir), notes
    try:
        from meshops.recipes.diff_views import render_diff_views

        view_paths = render_diff_views(
            baseline_mesh=baseline_mesh,
            candidate_mesh=candidate_mesh,
            views_dir=views_dir,
        )
        if not view_paths:
            notes.append("diff_views_empty_result_used_stub")
            return _write_stub_views(views_dir), notes
        return view_paths, notes
    except Exception as exc:
        notes.append(f"diff_views_unavailable_used_stub: {type(exc).__name__}: {exc}")
        return _write_stub_views(views_dir), notes


def run_repair(
    mesh_id: str,
    recipe_id: str,
    *,
    work_root: Path | str = "work",
    parent_rev: str | None = None,
    no_diff: bool = False,
    unify_vertices: bool | None = None,
) -> RecipeResult:
    """Run allowlisted T1/T2 recipe into atomic rev with recipe-tier guards."""
    work_root_p = Path(work_root)
    paths = JobPaths(work_root=work_root_p, mesh_id=mesh_id)
    if not paths.job_dir.is_dir():
        raise RepairRefuseError(
            f"Job directory not found: {paths.job_dir}",
            code="job_not_found",
        )
    ensure_job_layout(paths)

    # Hash pin — never mutate original
    original_hash_before: str | None = None
    if paths.original_stl.is_file():
        original_hash_before = content_sha256(paths.original_stl)

    diag = _load_diagnostics(paths)
    primary = _primary_class(diag)
    _assert_recipe_allowed(recipe_id, primary)

    # Input mesh: parent rev or original.stl (never working.ply as mutation source default)
    try:
        input_mesh = parent_mesh_path(paths, parent_rev)
    except FileNotFoundError as exc:
        raise RepairRefuseError(str(exc), code="missing_parent") from exc

    baseline_stats = diag.stats

    alloc = allocate_rev(paths, recipe_id)
    filter_metrics: dict[str, Any] = {}
    notes: list[str] = []
    error: str | None = None

    # unify_vertices: False for t1_clean (dups matter), True otherwise unless forced
    if unify_vertices is None:
        unify_vertices = recipe_id != "t1_clean"

    try:
        filter_metrics = get_recipe(recipe_id)(
            input_mesh,
            alloc.mesh_path,
            unify_vertices=unify_vertices,
        )
    except RecipeEngineError as exc:
        error = str(exc)
        from meshops.guards.models import GuardResult

        guard = GuardResult(
            ok=False,
            failed=["recipe_engine"],
            metrics={"recipe_error": error},
            messages=[error],
            policy_tier="recipe",
        )
        manifest = RevManifest(
            rev_id=alloc.rev_id,
            parent_rev=parent_rev,
            recipe_id=recipe_id,
            created_at=_now_iso(),
            ok=False,
            guard_result=guard,
            triage_class=primary,
            mesh_path=f"revs/failed_{alloc.rev_id}/mesh.stl",
            mesh_format="stl_binary",
            error=error,
            filter_metrics=filter_metrics,
            notes=[*notes, "recipe_engine_error"],
        )
        failed_dir = fail_rev(alloc, manifest)
        result = RecipeResult(
            ok=False,
            recipe_id=recipe_id,
            rev_id=alloc.rev_id,
            rev_dir=str(failed_dir),
            manifest=manifest,
            error=error,
            error_type="RecipeEngineError",
        )
        raise RepairError(error, rev_id=alloc.rev_id, rev_dir=failed_dir, result=result) from exc
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        from meshops.guards.models import GuardResult

        guard = GuardResult(
            ok=False,
            failed=["recipe_exception"],
            metrics={},
            messages=[error],
            policy_tier="recipe",
        )
        manifest = RevManifest(
            rev_id=alloc.rev_id,
            parent_rev=parent_rev,
            recipe_id=recipe_id,
            created_at=_now_iso(),
            ok=False,
            guard_result=guard,
            triage_class=primary,
            mesh_path=f"revs/failed_{alloc.rev_id}/mesh.stl",
            mesh_format="stl_binary",
            error=error,
            filter_metrics=filter_metrics,
            notes=[*notes, "unexpected_exception"],
        )
        failed_dir = fail_rev(alloc, manifest)
        result = RecipeResult(
            ok=False,
            recipe_id=recipe_id,
            rev_id=alloc.rev_id,
            rev_dir=str(failed_dir),
            manifest=manifest,
            error=error,
            error_type=type(exc).__name__,
        )
        raise RepairError(error, rev_id=alloc.rev_id, rev_dir=failed_dir, result=result) from exc

    # Candidate stats from written mesh
    if not alloc.mesh_path.is_file():
        error = "recipe produced no mesh.stl"
        from meshops.guards.models import GuardResult

        guard = GuardResult(
            ok=False,
            failed=["missing_mesh"],
            metrics={},
            messages=[error],
            policy_tier="recipe",
        )
        manifest = RevManifest(
            rev_id=alloc.rev_id,
            parent_rev=parent_rev,
            recipe_id=recipe_id,
            created_at=_now_iso(),
            ok=False,
            guard_result=guard,
            triage_class=primary,
            mesh_path=f"revs/failed_{alloc.rev_id}/mesh.stl",
            error=error,
            filter_metrics=filter_metrics,
        )
        failed_dir = fail_rev(alloc, manifest)
        result = RecipeResult(
            ok=False,
            recipe_id=recipe_id,
            rev_id=alloc.rev_id,
            rev_dir=str(failed_dir),
            manifest=manifest,
            error=error,
            error_type="MissingMesh",
        )
        raise RepairError(error, rev_id=alloc.rev_id, rev_dir=failed_dir, result=result)

    cand_mesh = load_mesh(alloc.mesh_path)
    cand_digest = content_sha256(alloc.mesh_path)
    cand_stats = compute_stats(
        cand_mesh,
        mesh_id=mesh_id,
        content_sha256_hex=cand_digest,
        file_size_bytes=alloc.mesh_path.stat().st_size,
        source_path=str(alloc.mesh_path),
    )

    policy = GuardPolicy.for_recipe(recipe_id)
    guard = check_export(baseline_stats, cand_stats, policy=policy)

    view_paths: list[str] = []
    if guard.ok:
        view_paths, view_notes = _try_diff_views(
            paths,
            baseline_mesh=input_mesh,
            candidate_mesh=alloc.mesh_path,
            views_dir=alloc.views_dir,
            no_diff=no_diff,
        )
        notes.extend(view_notes)
        # DoD-4/8 / Difficulty §12: never promote success without on-disk view evidence.
        if not view_paths or not all(Path(p).is_file() for p in view_paths):
            error = "mutating success requires view_paths on disk (DoD-4/8)"
            notes.append(error)
            from meshops.guards.models import GuardResult

            visual_guard = GuardResult(
                ok=False,
                failed=["missing_views"],
                metrics=dict(guard.metrics),
                messages=[error],
                policy_tier=guard.policy_tier,
            )
            manifest = RevManifest(
                rev_id=alloc.rev_id,
                parent_rev=parent_rev,
                recipe_id=recipe_id,
                created_at=_now_iso(),
                ok=False,
                guard_result=visual_guard,
                triage_class=primary,
                mesh_path=f"revs/failed_{alloc.rev_id}/mesh.stl",
                mesh_format="stl_binary",
                n_faces=cand_stats.faces,
                n_vertices=cand_stats.vertices,
                file_size_bytes=cand_stats.file_size_bytes,
                view_paths=view_paths,
                error=error,
                filter_metrics=filter_metrics,
                notes=notes,
            )
            write_manifest(alloc, manifest)
            failed_dir = fail_rev(alloc, manifest)
            result = RecipeResult(
                ok=False,
                recipe_id=recipe_id,
                rev_id=alloc.rev_id,
                rev_dir=str(failed_dir),
                manifest=manifest,
                error=error,
                error_type="MissingViews",
                notes=notes,
            )
            raise RepairError(error, rev_id=alloc.rev_id, rev_dir=failed_dir, result=result)

    rel_mesh = f"revs/{alloc.rev_id}/mesh.stl"
    manifest = RevManifest(
        rev_id=alloc.rev_id,
        parent_rev=parent_rev,
        recipe_id=recipe_id,
        created_at=_now_iso(),
        ok=guard.ok,
        guard_result=guard,
        triage_class=primary,
        mesh_path=rel_mesh if guard.ok else f"revs/failed_{alloc.rev_id}/mesh.stl",
        mesh_format="stl_binary",
        n_faces=cand_stats.faces,
        n_vertices=cand_stats.vertices,
        file_size_bytes=cand_stats.file_size_bytes,
        view_paths=view_paths,
        error=None if guard.ok else "; ".join(guard.messages),
        filter_metrics=filter_metrics,
        notes=notes,
    )

    if not guard.ok:
        write_manifest(alloc, manifest)
        failed_dir = fail_rev(alloc, manifest)
        result = RecipeResult(
            ok=False,
            recipe_id=recipe_id,
            rev_id=alloc.rev_id,
            rev_dir=str(failed_dir),
            manifest=manifest,
            error=manifest.error,
            error_type="GuardFail",
        )
        raise RepairError(
            f"guards failed: {manifest.error}",
            rev_id=alloc.rev_id,
            rev_dir=failed_dir,
            result=result,
        )

    write_manifest(alloc, manifest)
    success_dir = promote_rev(alloc)

    # Remap absolute view paths from .tmp_* → promoted success dir (files moved with rename).
    if view_paths:
        fixed_views = _remap_paths_after_promote(
            view_paths,
            from_root=alloc.tmp_dir,
            to_root=success_dir,
        )
        if fixed_views != view_paths:
            manifest = manifest.model_copy(update={"view_paths": fixed_views})
            (success_dir / "meta.json").write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
            view_paths = fixed_views

    if original_hash_before is not None:
        after = content_sha256(paths.original_stl)
        if after != original_hash_before:
            raise RuntimeError("FATAL: original.stl content hash changed during repair — abort")

    return RecipeResult(
        ok=True,
        recipe_id=recipe_id,
        rev_id=alloc.rev_id,
        rev_dir=str(success_dir),
        manifest=manifest,
        notes=notes,
    )


def _remap_paths_after_promote(
    paths_in: list[str],
    *,
    from_root: Path,
    to_root: Path,
) -> list[str]:
    """Rewrite absolute paths that lived under tmp_dir to the promoted rev dir."""
    from_resolved = from_root.resolve()
    to_resolved = to_root.resolve()
    out: list[str] = []
    for p in paths_in:
        path = Path(p)
        try:
            rel = path.resolve().relative_to(from_resolved)
            out.append(str(to_resolved / rel))
        except ValueError:
            # Path was not under tmp — keep as-is
            out.append(p)
    return out


def available_recipes() -> list[str]:
    return list_recipes()
