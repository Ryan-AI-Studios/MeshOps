"""Ingest STL into work/<mesh_id>/ without mutating the source."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import trimesh

from meshops.ingest.stats import compute_stats, load_mesh
from meshops.jobstore.paths import (
    PROXY_FACE_THRESHOLD,
    JobPaths,
    content_sha256,
    ensure_job_layout,
    set_readonly,
)
from meshops.models.diagnostics import MeshStats


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of a non-destructive ingest."""

    mesh_id: str
    job_dir: Path
    stats: MeshStats
    original_path: Path
    working_path: Path
    proxy_path: Path | None
    reused: bool


def _make_proxy(mesh: trimesh.Trimesh, target_faces: int = 50_000) -> trimesh.Trimesh:
    """Coarse decimation for triage compute when faces > threshold.

    Uses trimesh simplify_quadric_decimation when available; falls back to
    vertex clustering via simplify_quadratic if needed, else returns a
    randomly face-subsampled mesh as last resort (documented quality limit).
    """
    n = len(mesh.faces)
    if n <= target_faces:
        return mesh.copy()

    # Prefer quadric decimation (OpenGL/osmesa not required).
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        if simplified is not None and len(simplified.faces) > 0:
            return simplified
    except Exception:
        pass

    # Fallback: uniform face subsample (quality limit - topology not preserved).
    import numpy as np

    face_idx = np.random.default_rng(0).choice(n, size=min(target_faces, n), replace=False)
    sub = mesh.submesh([face_idx], append=True)
    if isinstance(sub, list):
        sub = trimesh.util.concatenate(sub)
    return sub


def ingest_stl(
    path: Path | str,
    *,
    work_root: Path | str = "work",
    proxy_face_threshold: int = PROXY_FACE_THRESHOLD,
) -> IngestResult:
    """Copy STL into job store, write working.ply / optional proxy.ply, return stats.

    Never overwrites the source path. Re-ingest of the same content reuses the
    same mesh_id directory (idempotent layout).
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"STL not found: {source}")

    work_root_p = Path(work_root)
    work_root_p.mkdir(parents=True, exist_ok=True)

    digest = content_sha256(source)
    mesh_id = digest[:12]
    paths = JobPaths(work_root=work_root_p, mesh_id=mesh_id)
    reused = paths.original_stl.is_file()

    ensure_job_layout(paths)

    # Copy original if missing; never write to source.
    if not paths.original_stl.is_file():
        shutil.copy2(source, paths.original_stl)
        set_readonly(paths.original_stl)
    else:
        # Ensure read-only even on re-ingest.
        set_readonly(paths.original_stl)

    # Load from job copy (source remains untouched).
    mesh = load_mesh(paths.original_stl)
    file_size = paths.original_stl.stat().st_size

    # Working PLY (indexed).
    if not paths.working_ply.is_file():
        mesh.export(paths.working_ply)

    proxy_path: Path | None = None
    if len(mesh.faces) > proxy_face_threshold:
        if not paths.proxy_ply.is_file():
            proxy = _make_proxy(mesh)
            proxy.export(paths.proxy_ply)
        proxy_path = paths.proxy_ply

    # Stub report until Phase 6 fills it.
    if not paths.report_md.is_file():
        paths.report_md.write_text(
            f"# MeshOps report — {mesh_id}\n\n_Stub — run `meshops report` after triage/render._\n",
            encoding="utf-8",
        )

    stats = compute_stats(
        mesh,
        mesh_id=mesh_id,
        content_sha256_hex=digest,
        file_size_bytes=file_size,
        source_path=str(source),
    )

    return IngestResult(
        mesh_id=mesh_id,
        job_dir=paths.job_dir,
        stats=stats,
        original_path=paths.original_stl,
        working_path=paths.working_ply,
        proxy_path=proxy_path,
        reused=reused,
    )
