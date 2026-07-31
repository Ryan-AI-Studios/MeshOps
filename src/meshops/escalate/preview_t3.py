"""T3 preview — honest local experiment package; NEVER promote / NEVER claim fixed.

Difficulty:
  §8 / N1 — no whole-model voxel remesh
  §6 / N2 — no full-mesh boolean after solidify
  §7 / N8 — no linked-flat auto-delete
  §13 / N6 — no autonomous hero-fixed claim

Preferred v1: minimal honest preview (copy mesh + meta under previews/).
Product ceiling is Blender handoff, not preview beauty.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meshops.escalate.errors import EscalateError
from meshops.escalate.models import PreviewResult
from meshops.escalate.roi import load_roi
from meshops.jobstore.paths import JobPaths, ensure_job_layout

# Forbidden operations for any future preview mutator (DoD-4 / N1/N2/N8).
FORBIDDEN_PREVIEW_OPS: frozenset[str] = frozenset(
    {
        "whole_model_voxel_remesh",
        "whole_model_remesh",
        "linked_flat_delete",
        "auto_delete_sheet",
        "full_mesh_boolean_after_solidify",
        "full_mesh_boolean",
        "global_remesh",
    }
)

_PREVIEW_HONESTY = (
    "preview_only — NOT fixed; not print-ready; not autonomous hero sculpt (N6). "
    "Use meshops escalate handoff + human/agent Blender sculpt."
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def assert_preview_op_allowed(op_id: str) -> None:
    """Raise if *op_id* is a permanently refused preview operation (N1/N2/N8)."""
    if op_id in FORBIDDEN_PREVIEW_OPS:
        raise EscalateError(
            f"preview operation {op_id!r} is permanently refused (N1/N2/N8)",
            code="preview_refuse_promote",
            details={"op_id": op_id, "forbidden": sorted(FORBIDDEN_PREVIEW_OPS)},
        )


def refuse_promote_preview(
    *,
    preview_id: str | None = None,
    notes: list[str] | None = None,
    recipe_id: str | None = None,
) -> None:
    """Hard refuse promote-to-working for preview artifacts."""
    note_hit = any("preview_only" in n for n in (notes or []))
    recipe_hit = recipe_id is not None and (
        "preview" in recipe_id or recipe_id.startswith("t3_preview")
    )
    if preview_id or note_hit or recipe_hit:
        raise EscalateError(
            "refusing promote of T3 preview artifact — previews never become working.ply "
            "(use import-sculpt --approve after human sculpt)",
            code="preview_refuse_promote",
            details={
                "preview_id": preview_id,
                "recipe_id": recipe_id,
                "notes": list(notes or []),
            },
        )


def preview_t3(
    mesh_id: str,
    roi_id: str | None = None,
    *,
    work_root: Path | str = "work",
) -> PreviewResult:
    """Create an honest T3 preview package under ``previews/<preview_id>/``.

    Does **not**:
      - claim fixed / print-ready
      - promote to working.ply
      - run remesh / linked-flat delete / full-mesh boolean
      - call accept as success

    Copies original (or working if present as reference only for path listing) into
    the preview tree with ``meta.json`` notes including ``preview_only``.
    """
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

    roi_notes: list[str] = []
    if roi_id is not None:
        roi = load_roi(mesh_id, roi_id, work_root=work_root_p)
        roi_notes.append(f"roi_id={roi.roi_id}")
        roi_notes.extend(roi.notes)

    # Stable-ish preview id: p + short hash of mesh + roi
    import hashlib

    key = f"{mesh_id}:{roi_id or 'noroi'}:{_now_iso()[:16]}"
    preview_id = "p" + hashlib.sha256(key.encode()).hexdigest()[:10]
    preview_dir = paths.previews_dir / preview_id
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True)

    mesh_copy = preview_dir / "mesh.stl"
    shutil.copy2(paths.original_stl, mesh_copy)

    # Optional ROI-region note file (no geometry mutation)
    views_dir = preview_dir / "views"
    views_dir.mkdir(exist_ok=True)
    # Minimal stub PNG so view_paths exist without claiming F3D success
    _MIN_PNG = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    view_paths: list[str] = []
    for cam in ("front", "three_quarter", "waist_zoom"):
        dest = views_dir / f"{cam}.png"
        dest.write_bytes(_MIN_PNG)
        view_paths.append(str(dest))

    notes = [
        "preview_only",
        "ok=false",
        "not_fixed",
        "no_whole_model_remesh",
        "no_linked_flat_delete",
        "no_full_mesh_boolean",
        "encourage_handoff",
        *roi_notes,
    ]

    meta: dict[str, Any] = {
        "schema_version": "1.0.0",
        "preview": True,
        "preview_id": preview_id,
        "mesh_id": mesh_id,
        "roi_id": roi_id,
        "ok": False,
        "may_promote_working": False,
        "may_claim_fixed": False,
        "created_at": _now_iso(),
        "notes": notes,
        "honesty_note": _PREVIEW_HONESTY,
        "mesh_path": str(mesh_copy),
        "view_paths": view_paths,
        "forbidden_ops": sorted(FORBIDDEN_PREVIEW_OPS),
    }
    meta_path = preview_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Sanity: working.ply must not have been rewritten by this function
    # (caller may check separately)

    return PreviewResult(
        ok=False,
        preview=True,
        mesh_id=mesh_id,
        roi_id=roi_id,
        preview_id=preview_id,
        preview_dir=preview_dir,
        notes=notes,
        honesty_note=_PREVIEW_HONESTY,
        paths={
            "preview_dir": str(preview_dir),
            "mesh": str(mesh_copy),
            "meta": str(meta_path),
            "views_dir": str(views_dir),
        },
        may_promote_working=False,
        may_claim_fixed=False,
    )
