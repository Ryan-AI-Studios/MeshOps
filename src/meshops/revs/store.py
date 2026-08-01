"""Atomic revision store under work/<mesh_id>/revs/.

Protocol:
  1. allocate next r00N + slug → create .tmp_r00N_<slug>/
  2. write mesh + meta draft into temp
  3. success → rename to r00N_<slug>/
  4. fail → rename to failed_r00N_<slug>/

Never write original.stl. Never leave half-written successful rev names.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from meshops.jobstore.paths import JobPaths, ensure_job_layout

# NOTE: RevManifest is imported lazily at runtime (TYPE_CHECKING only here).
# Top-level runtime import creates a cycle: revs.models → acceptance → revs.store → revs.models.

if TYPE_CHECKING:
    from meshops.revs.models import RevManifest

_REV_DIR_RE = re.compile(r"^(?:\.tmp_|failed_)?r(\d{3,})_(.+)$")
_META_NAME = "meta.json"
_MESH_NAME = "mesh.stl"


@dataclass(frozen=True, slots=True)
class RevAllocation:
    """Staging paths for an in-progress revision write."""

    rev_id: str  # r00N_slug
    rev_num: int
    slug: str
    tmp_dir: Path
    success_dir: Path
    failed_dir: Path
    mesh_path: Path  # under tmp
    meta_path: Path  # under tmp
    views_dir: Path  # under tmp


def _scan_max_rev_num(revs_dir: Path) -> int:
    """Highest r00N index among tmp/success/failed dirs."""
    if not revs_dir.is_dir():
        return 0
    max_n = 0
    for child in revs_dir.iterdir():
        if not child.is_dir():
            continue
        m = _REV_DIR_RE.match(child.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n


def next_rev_num(paths: JobPaths) -> int:
    ensure_job_layout(paths)
    return _scan_max_rev_num(paths.revs_dir) + 1


def allocate_rev(paths: JobPaths, slug: str) -> RevAllocation:
    """Create `.tmp_r00N_<slug>/` staging directory and return allocation."""
    ensure_job_layout(paths)
    # Sanitize slug: alnum + underscore only
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", slug).strip("_").lower() or "rev"
    n = next_rev_num(paths)
    rev_id = f"r{n:03d}_{clean}"
    tmp_dir = paths.revs_dir / f".tmp_{rev_id}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    views = tmp_dir / "views"
    views.mkdir()
    return RevAllocation(
        rev_id=rev_id,
        rev_num=n,
        slug=clean,
        tmp_dir=tmp_dir,
        success_dir=paths.revs_dir / rev_id,
        failed_dir=paths.revs_dir / f"failed_{rev_id}",
        mesh_path=tmp_dir / _MESH_NAME,
        meta_path=tmp_dir / _META_NAME,
        views_dir=views,
    )


def write_manifest(alloc: RevAllocation, manifest: RevManifest) -> None:
    """Write meta.json into the staging (tmp) directory."""
    alloc.meta_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def promote_rev(alloc: RevAllocation) -> Path:
    """Atomic rename tmp → r00N_slug. Fails if success dir already exists."""
    if alloc.success_dir.exists():
        raise FileExistsError(f"rev already exists: {alloc.success_dir}")
    if alloc.failed_dir.exists():
        raise FileExistsError(f"failed rev already exists: {alloc.failed_dir}")
    alloc.tmp_dir.rename(alloc.success_dir)
    return alloc.success_dir


def fail_rev(alloc: RevAllocation, manifest: RevManifest | None = None) -> Path:
    """Rename tmp → failed_r00N_slug (keep for debug)."""
    if manifest is not None:
        # Force ok=false
        if manifest.ok:
            manifest = manifest.model_copy(update={"ok": False})
        write_manifest(alloc, manifest)
    if alloc.failed_dir.exists():
        shutil.rmtree(alloc.failed_dir)
    if alloc.tmp_dir.exists():
        alloc.tmp_dir.rename(alloc.failed_dir)
    else:
        alloc.failed_dir.mkdir(parents=True, exist_ok=True)
        if manifest is not None:
            (alloc.failed_dir / _META_NAME).write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
    return alloc.failed_dir


def resolve_rev_dir(paths: JobPaths, rev_id: str) -> Path:
    """Resolve a promoted rev directory by id (not failed/tmp)."""
    # Accept bare r001_slug or with failed_ prefix for explicit lookup
    candidate = paths.revs_dir / rev_id
    if not candidate.is_dir():
        raise FileNotFoundError(f"Revision not found: {rev_id} under {paths.revs_dir}")
    return candidate


def rev_mesh_path(rev_dir: Path) -> Path:
    mesh = rev_dir / _MESH_NAME
    if not mesh.is_file():
        raise FileNotFoundError(f"mesh.stl missing in rev: {rev_dir}")
    return mesh


def load_manifest(rev_dir: Path) -> RevManifest:
    from meshops.revs.models import RevManifest as _RevManifest

    meta = rev_dir / _META_NAME
    if not meta.is_file():
        raise FileNotFoundError(f"meta.json missing in rev: {rev_dir}")
    return _RevManifest.model_validate_json(meta.read_text(encoding="utf-8"))


def parent_mesh_path(paths: JobPaths, parent_rev: str | None) -> Path:
    """Diff/export baseline: parent rev mesh or original.stl — never working.ply."""
    if parent_rev is None:
        if not paths.original_stl.is_file():
            raise FileNotFoundError(f"original.stl missing for mesh_id={paths.mesh_id}")
        return paths.original_stl
    rev_dir = resolve_rev_dir(paths, parent_rev)
    return rev_mesh_path(rev_dir)
