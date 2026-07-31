"""Promote accepted revision mesh to JobPaths.working_ply + working_manifest.json.

Never touch original.stl or rewrite diagnostics.json as working identity.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meshops.acceptance.models import AcceptanceResult
from meshops.acceptance.pack import accept_revision
from meshops.jobstore.paths import JobPaths, content_sha256
from meshops.revs.store import load_manifest, resolve_rev_dir, rev_mesh_path


class PromoteError(RuntimeError):
    """Promote refused (acceptance not ok or missing artifacts)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "promote_error",
        acceptance: AcceptanceResult | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.acceptance = acceptance


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def promote_working(
    mesh_id: str,
    rev: str,
    *,
    work_root: Path | str = "work",
    acceptance: AcceptanceResult | None = None,
    **accept_kwargs: Any,
) -> dict[str, Any]:
    """Copy accepted rev mesh → working_ply; write working_manifest.json.

    Runs accept_revision first unless a precomputed AcceptanceResult is provided.
    Aborts when acceptance.ok is False. Never overwrites original.stl.
    """
    paths = JobPaths(work_root=Path(work_root), mesh_id=mesh_id)
    if not paths.job_dir.is_dir():
        raise PromoteError(f"job not found: {paths.job_dir}", code="job_not_found")

    result = acceptance
    if result is None:
        result = accept_revision(mesh_id, rev, work_root=work_root, **accept_kwargs)

    if not result.ok:
        raise PromoteError(
            f"refuse promote: acceptance not ok (failed={result.failed})",
            code="not_accepted",
            acceptance=result,
        )

    rev_dir = resolve_rev_dir(paths, rev)
    if rev_dir.name.startswith("failed_"):
        raise PromoteError(
            f"refuse promote of failed rev dir: {rev_dir.name}",
            code="failed_rev",
            acceptance=result,
        )
    man = load_manifest(rev_dir)
    if not man.ok:
        raise PromoteError(
            f"refuse promote: manifest.ok=False for {rev!r}",
            code="failed_rev",
            acceptance=result,
        )

    # 0004: never promote T3 preview artifacts to working.ply (N6)
    if man.recipe_id.startswith("t3_preview") or man.recipe_id == "preview":
        raise PromoteError(
            f"refuse promote of preview recipe {man.recipe_id!r}",
            code="preview_refuse_promote",
            acceptance=result,
        )
    if any("preview_only" in n for n in man.notes):
        raise PromoteError(
            "refuse promote: revision notes include preview_only",
            code="preview_refuse_promote",
            acceptance=result,
        )

    src = rev_mesh_path(rev_dir)
    # Destination is JobPaths.working_ply — must be real PLY (ingest contract).
    # Never copy STL bytes under a .ply name (trimesh/F3D load by extension).
    dest = paths.working_ply
    dest.parent.mkdir(parents=True, exist_ok=True)
    from meshops.ingest.stats import load_mesh

    mesh = load_mesh(src)
    # Overwrite existing working.ply from a prior ingest with the accepted rev.
    if dest.is_file():
        dest.unlink()
    mesh.export(dest, file_type="ply")

    digest = content_sha256(dest)
    size = dest.stat().st_size
    manifest_path = paths.job_dir / "working_manifest.json"
    working_manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "mesh_id": mesh_id,
        "promoted_from_rev": man.rev_id,
        "promoted_at": _now_iso(),
        "working_path": str(dest),
        "source_mesh": str(src),
        "content_sha256": digest,
        "file_size_bytes": size,
        "n_faces": man.n_faces,
        "n_vertices": man.n_vertices,
        "recipe_id": man.recipe_id,
        "acceptance": {
            "ok": result.ok,
            "honesty": result.honesty,
            "honesty_message": result.honesty_message,
            "failed": list(result.failed),
            "view_kind": result.view_kind,
            "policy_tier": result.policy_tier,
        },
        "pack.promoted": True,
    }
    manifest_path.write_text(
        json.dumps(working_manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # Annotate acceptance metrics for callers (immutably via model_copy)
    updated = result.model_copy(
        update={
            "metrics": {**result.metrics, "pack.promoted": True},
        }
    )

    return {
        "ok": True,
        "mesh_id": mesh_id,
        "rev": man.rev_id,
        "working_ply": str(dest),
        "working_manifest": str(manifest_path),
        "content_sha256": digest,
        "acceptance": updated,
    }
