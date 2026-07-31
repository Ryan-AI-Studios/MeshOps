"""Import human/agent sculpt STL as atomic rev + sculpt-tier accept.

Requires explicit approve (CLI --approve). Never claims autonomous hero fixed (N6).
Does not auto-promote to working.ply.
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from meshops.acceptance.models import AcceptanceResult, ViewKind
from meshops.acceptance.pack import build_acceptance_from_guard
from meshops.escalate.errors import EscalateError
from meshops.escalate.models import ImportSculptResult
from meshops.escalate.preview_t3 import refuse_promote_preview
from meshops.guards import GuardPolicy, check_export, resolve_stats
from meshops.guards.models import GuardResult
from meshops.ingest.stats import compute_stats, load_mesh
from meshops.jobstore.paths import JobPaths, content_sha256, ensure_job_layout
from meshops.revs.models import RevManifest
from meshops.revs.store import (
    allocate_rev,
    fail_rev,
    promote_rev,
    write_manifest,
)

RECIPE_ID = "blender_sculpt_import"

_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
_STUB_CAMERAS = ("front", "three_quarter", "top")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _write_stub_views(views_dir: Path) -> list[str]:
    views_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name in _STUB_CAMERAS:
        for suffix in ("before", "after"):
            dest = views_dir / f"{name}_{suffix}.png"
            dest.write_bytes(_MIN_PNG)
            paths.append(str(dest))
    return paths


def _prefer_stub_diff() -> bool:
    if stub := os.environ.get("MESHOPS_STUB_DIFF", "").strip().lower():
        return stub in {"1", "true", "yes", "on"}
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def _try_diff_views(
    *,
    baseline_mesh: Path,
    candidate_mesh: Path,
    views_dir: Path,
    no_diff: bool,
) -> tuple[list[str], list[str], ViewKind]:
    notes: list[str] = []
    if no_diff or _prefer_stub_diff():
        notes.append("diff_stub_no_diff_flag" if no_diff else "diff_stub_ci_or_MESHOPS_STUB_DIFF")
        return _write_stub_views(views_dir), notes, "stub"
    try:
        from meshops.recipes.diff_views import render_diff_views

        view_paths = render_diff_views(
            baseline_mesh=baseline_mesh,
            candidate_mesh=candidate_mesh,
            views_dir=views_dir,
        )
        if not view_paths:
            notes.append("diff_views_empty_result_used_stub")
            return _write_stub_views(views_dir), notes, "stub"
        return view_paths, notes, "f3d"
    except Exception as exc:
        notes.append(f"diff_views_unavailable_used_stub: {type(exc).__name__}: {exc}")
        return _write_stub_views(views_dir), notes, "stub"


def _remap_paths_after_promote(
    paths_in: list[str],
    *,
    from_root: Path,
    to_root: Path,
) -> list[str]:
    from_resolved = from_root.resolve()
    to_resolved = to_root.resolve()
    out: list[str] = []
    for p in paths_in:
        path = Path(p)
        try:
            rel = path.resolve().relative_to(from_resolved)
            out.append(str(to_resolved / rel))
        except ValueError:
            out.append(p)
    return out


def import_sculpt(
    mesh_id: str,
    path: Path | str,
    *,
    approve: bool,
    work_root: Path | str = "work",
    no_diff: bool = False,
    parent_rev: str | None = None,
) -> ImportSculptResult:
    """Import sculpted STL as ``blender_sculpt_import`` rev with sculpt policy accept.

    *approve* must be True (CLI ``--approve``). Does **not** promote to working.ply.
    """
    if not approve:
        raise EscalateError(
            "import-sculpt requires explicit --approve "
            "(acknowledges human/agent sculpt responsibility; N6)",
            code="approve_required",
        )

    sculpt_path = Path(path)
    if not sculpt_path.is_file():
        raise EscalateError(
            f"sculpt mesh not found: {sculpt_path}",
            code="missing_mesh",
            details={"path": str(sculpt_path)},
        )

    # Refuse importing from preview tree as if it were a finished sculpt
    parts = {p.lower() for p in sculpt_path.parts}
    if "previews" in parts:
        refuse_promote_preview(preview_id=sculpt_path.parent.name)
        raise EscalateError(
            "refusing import of preview artifact as sculpt "
            "(export a real Blender sculpt STL instead)",
            code="preview_refuse_promote",
        )

    work_root_p = Path(work_root)
    paths = JobPaths(work_root=work_root_p, mesh_id=mesh_id)
    if not paths.job_dir.is_dir():
        raise EscalateError(
            f"Job directory not found: {paths.job_dir}",
            code="job_not_found",
        )
    ensure_job_layout(paths)

    if not paths.original_stl.is_file():
        raise EscalateError(
            f"original.stl missing: {paths.original_stl}",
            code="missing_mesh",
        )

    # Baseline: parent rev mesh or original.stl (never working.ply).
    # When parent_rev is set, guards MUST use that mesh's stats — do not replace
    # with diagnostics (which cache original.stl only).
    from meshops.revs.store import parent_mesh_path

    try:
        baseline_mesh = parent_mesh_path(paths, parent_rev)
    except FileNotFoundError as exc:
        raise EscalateError(str(exc), code="missing_mesh") from exc

    baseline_stats = resolve_stats(baseline_mesh, mesh_id=mesh_id)
    if parent_rev is None and paths.diagnostics_json.is_file():
        try:
            from meshops.models.diagnostics import Diagnostics

            diag = Diagnostics.model_validate_json(
                paths.diagnostics_json.read_text(encoding="utf-8")
            )
            # Diagnostics stats are for original.stl — only safe when baseline is original
            baseline_stats = diag.stats
        except Exception:
            pass

    alloc = allocate_rev(paths, RECIPE_ID)
    notes: list[str] = [
        "blender_sculpt_import",
        "approve=true",
        "not_autonomous_hero_fixed",
        "no_auto_promote_working",
    ]
    view_paths: list[str] = []
    view_kind: ViewKind = "none"

    def _fail_import(
        *,
        error: str,
        code: str,
        failed_codes: list[str],
        extra_notes: list[str] | None = None,
    ) -> NoReturn:
        """Write failed rev and raise EscalateError (atomic cleanup)."""
        guard_f = GuardResult(
            ok=False,
            failed=failed_codes,
            metrics={},
            messages=[error],
            policy_tier="sculpt",
        )
        man = RevManifest(
            rev_id=alloc.rev_id,
            parent_rev=parent_rev,
            recipe_id=RECIPE_ID,
            created_at=_now_iso(),
            ok=False,
            guard_result=guard_f,
            triage_class="T3_sheet",
            mesh_path=f"revs/failed_{alloc.rev_id}/mesh.stl",
            error=error,
            notes=[*notes, *(extra_notes or [])],
        )
        failed = fail_rev(alloc, man)
        raise EscalateError(
            error,
            code=code,
            details={"rev_dir": str(failed)},
        )

    try:
        shutil.copy2(sculpt_path, alloc.mesh_path)
    except OSError as exc:
        _fail_import(
            error=f"failed to copy sculpt mesh: {exc}",
            code="import_failed",
            failed_codes=["import_copy"],
            extra_notes=["copy_failed"],
        )

    if not alloc.mesh_path.is_file() or alloc.mesh_path.stat().st_size <= 0:
        _fail_import(
            error="sculpt mesh empty after copy",
            code="import_failed",
            failed_codes=["missing_mesh"],
        )

    try:
        cand_mesh = load_mesh(alloc.mesh_path)
        cand_digest = content_sha256(alloc.mesh_path)
        cand_stats = compute_stats(
            cand_mesh,
            mesh_id=mesh_id,
            content_sha256_hex=cand_digest,
            file_size_bytes=alloc.mesh_path.stat().st_size,
            source_path=str(alloc.mesh_path),
        )
    except Exception as exc:
        _fail_import(
            error=f"sculpt mesh unreadable/malformed: {type(exc).__name__}: {exc}",
            code="import_failed",
            failed_codes=["malformed_mesh"],
            extra_notes=["load_mesh_failed"],
        )

    policy = GuardPolicy.for_sculpt()
    guard = check_export(baseline_stats, cand_stats, policy=policy)

    if guard.ok:
        view_paths, view_notes, view_kind = _try_diff_views(
            baseline_mesh=baseline_mesh,
            candidate_mesh=alloc.mesh_path,
            views_dir=alloc.views_dir,
            no_diff=no_diff,
        )
        notes.extend(view_notes)
        if not view_paths or not all(Path(p).is_file() for p in view_paths):
            guard = GuardResult(
                ok=False,
                failed=["missing_views"],
                metrics=dict(guard.metrics),
                messages=["sculpt import success requires view_paths (Difficulty §12)"],
                policy_tier="sculpt",
            )
            notes.append("missing_views")

    acceptance: AcceptanceResult = build_acceptance_from_guard(
        guard,
        baseline_stats=baseline_stats,
        cand_stats=cand_stats,
        view_paths=view_paths,
        require_views=True,
        view_kind=view_kind if view_paths else "none",
        view_notes=notes,
        allow_stubs=True,
        policy_tier="sculpt",
    )

    ok = bool(guard.ok and acceptance.ok)
    rel_mesh = f"revs/{alloc.rev_id}/mesh.stl" if ok else f"revs/failed_{alloc.rev_id}/mesh.stl"
    manifest = RevManifest(
        rev_id=alloc.rev_id,
        parent_rev=parent_rev,
        recipe_id=RECIPE_ID,
        created_at=_now_iso(),
        ok=ok,
        guard_result=guard,
        triage_class="T3_sheet",
        mesh_path=rel_mesh,
        mesh_format="stl_binary",
        n_faces=cand_stats.faces,
        n_vertices=cand_stats.vertices,
        file_size_bytes=cand_stats.file_size_bytes,
        view_paths=view_paths,
        view_kind=view_kind if view_paths else None,
        error=None if ok else "; ".join(guard.messages + acceptance.messages),
        filter_metrics={"source_sculpt": str(sculpt_path)},
        notes=notes,
    )
    write_manifest(alloc, manifest)

    if not ok:
        failed_dir = fail_rev(alloc, manifest)
        # Remap view paths if any were written under tmp
        if view_paths:
            remapped = _remap_paths_after_promote(
                view_paths, from_root=alloc.tmp_dir, to_root=failed_dir
            )
            manifest = manifest.model_copy(update={"view_paths": remapped})
            (failed_dir / "meta.json").write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
        return ImportSculptResult(
            ok=False,
            mesh_id=mesh_id,
            rev_id=alloc.rev_id,
            rev_dir=str(failed_dir),
            recipe_id=RECIPE_ID,
            acceptance=acceptance,
            notes=notes,
            paths={"rev_dir": str(failed_dir), "mesh": str(failed_dir / "mesh.stl")},
            extra={"failed": list(acceptance.failed)},
        )

    # Promote success rev (atomic rename) — still NOT working.ply
    success_dir = promote_rev(alloc)
    remapped_views = _remap_paths_after_promote(
        view_paths, from_root=alloc.tmp_dir, to_root=success_dir
    )
    manifest = manifest.model_copy(update={"view_paths": remapped_views})
    (success_dir / "meta.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Rebuild acceptance with remapped view paths for caller honesty
    acceptance = build_acceptance_from_guard(
        guard,
        baseline_stats=baseline_stats,
        cand_stats=cand_stats,
        view_paths=remapped_views,
        require_views=True,
        view_kind=view_kind,
        view_notes=notes,
        allow_stubs=True,
        policy_tier="sculpt",
    )

    result_paths: dict[str, str] = {
        "rev_dir": str(success_dir),
        "mesh": str(success_dir / "mesh.stl"),
        "meta": str(success_dir / "meta.json"),
    }
    extra: dict[str, Any] = {
        "policy_tier": "sculpt",
        "promoted_to_working": False,
    }

    return ImportSculptResult(
        ok=bool(acceptance.ok),
        mesh_id=mesh_id,
        rev_id=alloc.rev_id,
        rev_dir=str(success_dir),
        recipe_id=RECIPE_ID,
        acceptance=acceptance,
        notes=notes,
        paths=result_paths,
        extra=extra,
    )
