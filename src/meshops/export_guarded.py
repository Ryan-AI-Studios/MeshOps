"""Guarded export of original or rev mesh (fail-closed)."""

from __future__ import annotations

import shutil
from pathlib import Path

from meshops.guards import GuardPolicy, check_export
from meshops.guards.models import GuardResult
from meshops.ingest.stats import compute_stats, load_mesh
from meshops.jobstore.paths import JobPaths, content_sha256
from meshops.models.diagnostics import Diagnostics, MeshStats
from meshops.revs.store import load_manifest, resolve_rev_dir, rev_mesh_path


class ExportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        guard: GuardResult | None = None,
        code: str = "export_error",
    ) -> None:
        super().__init__(message)
        self.guard = guard
        self.code = code


def _baseline_stats(paths: JobPaths) -> MeshStats:
    if paths.diagnostics_json.is_file():
        diag = Diagnostics.model_validate_json(paths.diagnostics_json.read_text(encoding="utf-8"))
        return diag.stats
    if not paths.original_stl.is_file():
        raise ExportError("no baseline (missing diagnostics and original.stl)")
    mesh = load_mesh(paths.original_stl)
    digest = content_sha256(paths.original_stl)
    return compute_stats(
        mesh,
        mesh_id=paths.mesh_id,
        content_sha256_hex=digest,
        file_size_bytes=paths.original_stl.stat().st_size,
        source_path=str(paths.original_stl),
    )


def resolve_export_source(
    paths: JobPaths,
    rev: str | None,
) -> Path:
    if rev is None:
        if paths.original_stl.is_file():
            return paths.original_stl
        raise ExportError("no original.stl to export")
    rev_dir = resolve_rev_dir(paths, rev)
    man = load_manifest(rev_dir)
    if not man.ok:
        raise ExportError(
            f"refusing to export failed rev {rev!r}",
            code="failed_rev",
        )
    return rev_mesh_path(rev_dir)


def guarded_export(
    mesh_id: str,
    out: Path | str,
    *,
    work_root: Path | str = "work",
    rev: str | None = None,
) -> dict[str, object]:
    """Export-tier check_export then copy to out. Fail-closed."""
    paths = JobPaths(work_root=Path(work_root), mesh_id=mesh_id)
    if not paths.job_dir.is_dir():
        raise ExportError(f"job not found: {paths.job_dir}", code="job_not_found")

    src = resolve_export_source(paths, rev)
    baseline = _baseline_stats(paths)
    policy = GuardPolicy.for_export()
    guard = check_export(baseline, src, policy=policy)
    if not guard.ok:
        raise ExportError(
            f"export guards failed: {'; '.join(guard.messages)}",
            guard=guard,
            code="guard_fail",
        )

    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return {
        "ok": True,
        "mesh_id": mesh_id,
        "rev": rev,
        "source": str(src),
        "out": str(dest),
        "guard": guard.model_dump(mode="json"),
    }
