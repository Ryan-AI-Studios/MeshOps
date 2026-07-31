"""ROI AABB packages under ``rois/<roi_id>/`` (manual bbox always; heuristic optional).

Difficulty §1: no stdin; laterality / needs_user_input recorded as notes only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from meshops.escalate.errors import EscalateError
from meshops.escalate.models import RoiManifest
from meshops.jobstore.paths import JobPaths, ensure_job_layout

Vec3 = tuple[float, float, float]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _as_vec3(v: Sequence[float], *, name: str) -> Vec3:
    if len(v) != 3:
        raise EscalateError(
            f"{name} must have 3 components, got {len(v)}",
            code="invalid_bbox",
        )
    return (float(v[0]), float(v[1]), float(v[2]))


def _normalize_bbox(bbox_min: Vec3, bbox_max: Vec3) -> tuple[Vec3, Vec3]:
    lo = (
        min(bbox_min[0], bbox_max[0]),
        min(bbox_min[1], bbox_max[1]),
        min(bbox_min[2], bbox_max[2]),
    )
    hi = (
        max(bbox_min[0], bbox_max[0]),
        max(bbox_min[1], bbox_max[1]),
        max(bbox_min[2], bbox_max[2]),
    )
    if lo == hi:
        raise EscalateError(
            "bbox_min and bbox_max collapse to a point (zero volume AABB)",
            code="invalid_bbox",
            details={"bbox_min": list(lo), "bbox_max": list(hi)},
        )
    # Degenerate flat AABB (zero extent on one axis) is allowed for sheet ROIs
    # but reject completely inverted already handled by min/max.
    return lo, hi


def roi_id_for(mesh_id: str, bbox_min: Vec3, bbox_max: Vec3) -> str:
    """Deterministic short roi_id from mesh_id + bbox."""
    payload = (
        f"{mesh_id}:"
        f"{bbox_min[0]:.6g},{bbox_min[1]:.6g},{bbox_min[2]:.6g}:"
        f"{bbox_max[0]:.6g},{bbox_max[1]:.6g},{bbox_max[2]:.6g}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"r{digest}"


def _job_paths(mesh_id: str, work_root: Path | str) -> JobPaths:
    paths = JobPaths(work_root=Path(work_root), mesh_id=mesh_id)
    if not paths.job_dir.is_dir():
        raise EscalateError(
            f"Job directory not found: {paths.job_dir}",
            code="job_not_found",
            details={"mesh_id": mesh_id, "job_dir": str(paths.job_dir)},
        )
    ensure_job_layout(paths)
    return paths


def _laterality_notes(paths: JobPaths) -> list[str]:
    """Append needs_user_input reminder when diagnostics flag multi-figure laterality."""
    notes: list[str] = []
    if not paths.diagnostics_json.is_file():
        return notes
    try:
        from meshops.models.diagnostics import Diagnostics

        diag = Diagnostics.model_validate_json(paths.diagnostics_json.read_text(encoding="utf-8"))
        if diag.needs_user_input:
            notes.append(
                "needs_user_input: multi-figure / laterality uncertain — "
                "confirm anatomical L/R before sculpt (Difficulty §1); no stdin prompt"
            )
        lat = getattr(diag.laterality_status, "value", str(diag.laterality_status))
        if lat in ("unknown", "left", "right", "bilateral"):
            notes.append(f"laterality_status={lat}")
    except Exception as exc:
        notes.append(f"diagnostics_read_soft_fail: {type(exc).__name__}")
    return notes


def _try_extract_roi_ply(
    mesh_path: Path,
    bbox_min: Vec3,
    bbox_max: Vec3,
    dest: Path,
) -> str | None:
    """Best-effort face subset intersecting AABB → roi.ply. Returns path or None."""
    try:
        import numpy as np
        import trimesh

        from meshops.ingest.stats import load_mesh

        mesh = load_mesh(mesh_path)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            return None
        centroids = mesh.triangles_center
        lo = np.asarray(bbox_min, dtype=np.float64)
        hi = np.asarray(bbox_max, dtype=np.float64)
        inside = np.all((centroids >= lo) & (centroids <= hi), axis=1)
        face_idx = np.nonzero(inside)[0]
        if face_idx.size == 0:
            return None
        sub = mesh.submesh([face_idx], append=True)
        if isinstance(sub, list):
            if not sub:
                return None
            sub = trimesh.util.concatenate(sub)
        if sub is None or len(sub.faces) == 0:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        sub.export(dest, file_type="ply")
        if dest.is_file() and dest.stat().st_size > 0:
            return str(dest)
    except Exception:
        return None
    return None


def create_roi_bbox(
    mesh_id: str,
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
    *,
    work_root: Path | str = "work",
    notes: list[str] | None = None,
    source: str = "manual",
    extract_ply: bool = True,
) -> RoiManifest:
    """Write ``rois/<roi_id>/mask.json`` (+ optional ``roi.ply``) and return manifest.

    Manual bbox is always allowed. Heuristic callers must pass ``source="heuristic"``.
    """
    paths = _job_paths(mesh_id, work_root)
    lo = _as_vec3(bbox_min, name="bbox_min")
    hi = _as_vec3(bbox_max, name="bbox_max")
    lo, hi = _normalize_bbox(lo, hi)

    if source not in ("manual", "heuristic"):
        raise EscalateError(
            f"roi source must be 'manual' or 'heuristic', got {source!r}",
            code="invalid_bbox",
        )

    roi_id = roi_id_for(mesh_id, lo, hi)
    roi_dir = paths.rois_dir / roi_id
    roi_dir.mkdir(parents=True, exist_ok=True)

    extra_notes: list[str] = list(notes or [])
    extra_notes.extend(_laterality_notes(paths))
    if source == "heuristic":
        extra_notes.append(
            "source=heuristic — suggested only; override with manual bbox (never sole path)"
        )

    roi_ply_rel: str | None = None
    if extract_ply and paths.original_stl.is_file():
        ply_path = roi_dir / "roi.ply"
        written = _try_extract_roi_ply(paths.original_stl, lo, hi, ply_path)
        if written:
            roi_ply_rel = f"rois/{roi_id}/roi.ply"
            extra_notes.append("roi_ply_extracted")
        else:
            extra_notes.append("roi_ply_extract_skipped_or_empty")

    manifest = RoiManifest(
        roi_id=roi_id,
        mesh_id=mesh_id,
        kind="aabb",
        bbox_min=lo,
        bbox_max=hi,
        source=source,  # type: ignore[arg-type]
        notes=extra_notes,
        created_at=_now_iso(),
        roi_ply=roi_ply_rel,
        mask_path=f"rois/{roi_id}/mask.json",
    )
    mask_path = roi_dir / "mask.json"
    mask_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    # Optional free-form notes sidecar
    notes_path = roi_dir / "notes.json"
    notes_path.write_text(
        json.dumps({"roi_id": roi_id, "notes": extra_notes}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_roi(
    mesh_id: str,
    roi_id: str,
    *,
    work_root: Path | str = "work",
) -> RoiManifest:
    """Load ``rois/<roi_id>/mask.json``."""
    paths = _job_paths(mesh_id, work_root)
    mask = paths.rois_dir / roi_id / "mask.json"
    if not mask.is_file():
        raise EscalateError(
            f"ROI not found: {mask}",
            code="roi_not_found",
            details={"mesh_id": mesh_id, "roi_id": roi_id},
        )
    return RoiManifest.model_validate_json(mask.read_text(encoding="utf-8"))


def create_roi_from_sheet_heuristic(
    mesh_id: str,
    *,
    work_root: Path | str = "work",
    notes: list[str] | None = None,
) -> RoiManifest:
    """Suggest an AABB from mesh stats / flat-region heuristic.

    **Never** the sole path — records ``source=heuristic`` and allows override.
    Prefer ``create_roi_bbox`` with a confirmed manual bbox for production sculpt.
    """
    paths = _job_paths(mesh_id, work_root)
    if not paths.original_stl.is_file():
        raise EscalateError(
            f"original.stl missing for heuristic ROI: {paths.original_stl}",
            code="missing_mesh",
        )

    extra = list(notes or [])
    bbox_min: Vec3
    bbox_max: Vec3

    # Prefer diagnostics stats bbox (always present after triage/ingest stats)
    if paths.diagnostics_json.is_file():
        try:
            from meshops.models.diagnostics import Diagnostics

            diag = Diagnostics.model_validate_json(
                paths.diagnostics_json.read_text(encoding="utf-8")
            )
            bbox_min = tuple(diag.stats.bbox_min)  # type: ignore[assignment]
            bbox_max = tuple(diag.stats.bbox_max)  # type: ignore[assignment]
            # Shrink toward center band (middle 60%) as a coarse sheet-region hint
            cx = 0.5 * (bbox_min[0] + bbox_max[0])
            cy = 0.5 * (bbox_min[1] + bbox_max[1])
            cz = 0.5 * (bbox_min[2] + bbox_max[2])
            hx = 0.3 * (bbox_max[0] - bbox_min[0])
            hy = 0.3 * (bbox_max[1] - bbox_min[1])
            hz = 0.3 * (bbox_max[2] - bbox_min[2])
            # Prefer mid-height band (waist-ish for multi-figure) — Difficulty §1/§10
            bbox_min = (cx - hx, cy - hy, cz - hz * 0.5)
            bbox_max = (cx + hx, cy + hy, cz + hz * 0.5)
            extra.append(f"heuristic_from_diagnostics sheet_score={diag.sheet_score.score:.3f}")
        except Exception as exc:
            extra.append(f"heuristic_diag_fail:{type(exc).__name__}")
            bbox_min, bbox_max = _bbox_from_mesh(paths.original_stl)
            extra.append("heuristic_fallback_mesh_bbox_center")
    else:
        bbox_min, bbox_max = _bbox_from_mesh(paths.original_stl)
        extra.append("heuristic_from_mesh_bbox_center_no_diagnostics")

    return create_roi_bbox(
        mesh_id,
        bbox_min,
        bbox_max,
        work_root=work_root,
        notes=extra,
        source="heuristic",
    )


def _bbox_from_mesh(mesh_path: Path) -> tuple[Vec3, Vec3]:
    from meshops.ingest.stats import load_mesh

    mesh = load_mesh(mesh_path)
    bmin = mesh.bounds[0]
    bmax = mesh.bounds[1]
    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    cz = 0.5 * (bmin[2] + bmax[2])
    hx = 0.3 * (bmax[0] - bmin[0])
    hy = 0.3 * (bmax[1] - bmin[1])
    hz = 0.3 * (bmax[2] - bmin[2])
    return (
        (float(cx - hx), float(cy - hy), float(cz - hz * 0.5)),
        (float(cx + hx), float(cy + hy), float(cz + hz * 0.5)),
    )
