"""Absolute design validation floors (PRIMARY design gate).

Self-baseline accept_candidate relative floors are a hero wipeout safety net only.
Primary gate = these absolute floors + topology + views.
"""

from __future__ import annotations

from pathlib import Path

from meshops.design.errors import DesignError
from meshops.models.diagnostics import MeshStats

# Absolute floors (DoD / plan Phase 4.2)
MIN_FACES = 50
BBOX_DIAGONAL_MIN_MM = 5.0
BBOX_DIAGONAL_MAX_MM = 1000.0


def _volume_ok_recheck(mesh_path: Path) -> bool:
    """Binary STL may be an unmerged vertex soup under process=False.

    Recheck with process=True (vertex weld) — CAD tessellations are typically
    closed solids once vertices are merged.
    """
    try:
        import trimesh

        loaded = trimesh.load(str(mesh_path), force="mesh", process=True)
        if isinstance(loaded, trimesh.Trimesh):
            return bool(loaded.is_volume)
    except Exception:
        return False
    return False


def validate_design_mesh(
    stats: MeshStats,
    *,
    mesh_path: Path | str | None = None,
) -> None:
    """Fail closed if design mesh fails absolute floors / scale band.

    Raises DesignError with codes:
      - validation_failed (faces / volume)
      - unreasonable_cad_scale (bbox diagonal outside [5, 1000] mm)
    """
    if stats.faces < MIN_FACES:
        raise DesignError(
            f"design mesh face count {stats.faces} < min_faces={MIN_FACES}",
            code="validation_failed",
            details={"faces": stats.faces, "min_faces": MIN_FACES},
        )

    # require_volume when is_volume known (False). None = unknown → skip hard fail.
    if stats.is_volume is False:
        path = Path(mesh_path) if mesh_path is not None else None
        if path is not None and path.is_file() and _volume_ok_recheck(path):
            pass  # welded recheck confirms solid volume
        else:
            raise DesignError(
                "design mesh is_volume=False (require volume solid)",
                code="validation_failed",
                details={"is_volume": stats.is_volume, "mesh_path": str(path) if path else None},
            )

    diag = float(stats.bbox_diagonal)
    if diag < BBOX_DIAGONAL_MIN_MM or diag > BBOX_DIAGONAL_MAX_MM:
        raise DesignError(
            f"bbox_diagonal {diag:.4f} mm outside "
            f"[{BBOX_DIAGONAL_MIN_MM}, {BBOX_DIAGONAL_MAX_MM}] mm "
            f"(unreasonable CAD scale; units must be mm)",
            code="unreasonable_cad_scale",
            details={
                "bbox_diagonal": diag,
                "min_mm": BBOX_DIAGONAL_MIN_MM,
                "max_mm": BBOX_DIAGONAL_MAX_MM,
            },
        )
