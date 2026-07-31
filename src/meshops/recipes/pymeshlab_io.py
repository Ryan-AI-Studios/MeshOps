"""Lazy PyMeshLab I/O + apply_filter helpers.

Code rules (pytest filterwarnings=error):
  - Never call len(ms), ms.number_meshes(), ms.number_rasters().
  - Use ms.mesh_number() / ms.raster_number() if needed.
  - Fresh MeshSet() per recipe (or clear() after).
  - Prefer ms.apply_filter(name, **kwargs) for metrics dicts.
  - Catch PyMeshLabException → structured fail (never bare crash).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any


class RecipeEngineError(RuntimeError):
    """Structured PyMeshLab failure for recipe orchestration."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


def _import_pymeshlab() -> Any:
    """Lazy import — avoid hard import at package load for non-recipe paths."""
    try:
        import pymeshlab  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RecipeEngineError(
            "pymeshlab is not installed or failed to import",
            cause=exc,
        ) from exc
    return pymeshlab


def pymeshlab_exception_type() -> type[BaseException]:
    pml = _import_pymeshlab()
    return pml.PyMeshLabException  # type: ignore[no-any-return]


def new_mesh_set() -> Any:
    """Create a fresh MeshSet (required per recipe)."""
    pml = _import_pymeshlab()
    return pml.MeshSet()


def load_mesh_into(
    ms: Any,
    path: Path | str,
    *,
    unify_vertices: bool = False,
) -> None:
    """Load mesh into MeshSet.

    Prefer ``load_new_mesh`` when format plugins work. On CI/Linux wheels that
    raise ``Unknown format for load: stl|ply``, build a ``pymeshlab.Mesh`` from
    numpy arrays via trimesh (no file-format plugin required).

    ``unify_vertices`` only applies to the native load path; the numpy bridge
    preserves the mesh as loaded by trimesh with ``process=False`` (dups kept).
    """
    pml = _import_pymeshlab()
    src = Path(path).resolve()
    if not src.is_file():
        raise RecipeEngineError(f"load failed: file not found: {src}")

    # 1) Native file load (preserves unify_vertices for honest T1 when available)
    try:
        try:
            ms.load_new_mesh(str(src), unify_vertices=unify_vertices)
        except TypeError:
            ms.load_new_mesh(str(src))
        return
    except Exception:
        pass  # fall through to numpy bridge

    # 2) Numpy bridge — works when IO plugins are missing/broken
    try:
        from meshops.ingest.stats import load_mesh as trimesh_load

        mesh = trimesh_load(src)
        verts = mesh.vertices.astype("float64", copy=False)
        faces = mesh.faces.astype("int32", copy=False)
        pml_mesh = pml.Mesh(verts, faces)
        # add_mesh may take mesh and optional name
        try:
            ms.add_mesh(pml_mesh, src.stem)
        except TypeError:
            ms.add_mesh(pml_mesh)
        # Ensure current mesh is the one we added
        with contextlib.suppress(Exception):
            if hasattr(ms, "set_current_mesh") and ms.mesh_number() > 0:
                ms.set_current_mesh(ms.mesh_number() - 1)
    except pml.PyMeshLabException as exc:
        raise RecipeEngineError(f"load failed: {exc}", cause=exc) from exc
    except Exception as exc:
        raise RecipeEngineError(f"load failed: {exc}", cause=exc) from exc


def apply_filter(ms: Any, name: str, **kwargs: Any) -> dict[str, Any]:
    """Apply named filter; return metrics dict (may be empty)."""
    pml = _import_pymeshlab()
    try:
        result = ms.apply_filter(name, **kwargs)
    except pml.PyMeshLabException as exc:
        raise RecipeEngineError(
            f"filter {name!r} failed: {exc}",
            cause=exc,
        ) from exc
    if result is None:
        return {}
    if isinstance(result, dict):
        return dict(result)
    return {"result": result}


def save_current_mesh(ms: Any, path: Path | str) -> None:
    """Save current mesh as binary STL (MeshLab default for .stl).

    Falls back to trimesh export of vertex/face arrays when MeshLab IO plugins
    cannot write STL (same class of failure as load on some Linux wheels).
    """
    pml = _import_pymeshlab()
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        ms.save_current_mesh(str(dest))
        if dest.is_file() and dest.stat().st_size > 0:
            return
    except Exception:
        pass

    try:
        import numpy as np
        import trimesh

        mesh = ms.current_mesh()
        verts = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
        faces = np.asarray(mesh.face_matrix(), dtype=np.int64)
        tmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        tmesh.export(dest, file_type="stl")
    except Exception as exc:
        raise RecipeEngineError(f"save failed: {exc}", cause=exc) from exc


def current_face_vertex_counts(ms: Any) -> tuple[int, int]:
    """Best-effort face/vertex counts without deprecated MeshSet APIs."""
    try:
        mesh = ms.current_mesh()
        faces = int(mesh.face_number())
        verts = int(mesh.vertex_number())
        return faces, verts
    except Exception:
        return 0, 0


def run_filter_chain(
    input_path: Path | str,
    output_path: Path | str,
    steps: list[tuple[str, dict[str, Any]]],
    *,
    unify_vertices: bool = False,
) -> dict[str, Any]:
    """Load → apply ordered filters → save binary STL. Fresh MeshSet.

    Returns aggregated filter_metrics keyed by filter name (with index suffix
    when the same name appears more than once).
    """
    ms = new_mesh_set()
    load_mesh_into(ms, input_path, unify_vertices=unify_vertices)
    metrics: dict[str, Any] = {}
    for i, (name, kwargs) in enumerate(steps):
        m = apply_filter(ms, name, **kwargs)
        key = name if name not in metrics else f"{name}#{i}"
        metrics[key] = m
    save_current_mesh(ms, output_path)
    # Explicit clear for memory hygiene (fresh set already per call).
    with contextlib.suppress(Exception):
        ms.clear()
    return metrics
