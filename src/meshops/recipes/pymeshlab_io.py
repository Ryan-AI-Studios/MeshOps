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

    STL default in MeshLab unifies verts; pass unify_vertices=False when
    duplicate-vertex repair must still see dups. Prefer PLY fixtures for T1.

    Some Linux pymeshlab wheels fail to load binary STL with
    ``Unknown format for load: stl``; fall back to a temporary PLY bridge
    via trimesh so CI and Windows both succeed.
    """
    pml = _import_pymeshlab()
    src = Path(path).resolve()
    if not src.is_file():
        raise RecipeEngineError(f"load failed: file not found: {src}")

    def _try_load(load_path: Path) -> None:
        # Prefer absolute path; kwargs order varies slightly across builds.
        try:
            ms.load_new_mesh(str(load_path), unify_vertices=unify_vertices)
        except TypeError:
            # Older/newer signature without unify_vertices kw.
            ms.load_new_mesh(str(load_path))

    try:
        _try_load(src)
        return
    except pml.PyMeshLabException as first_exc:
        # Bridge via PLY when STL format plugins misbehave (common on Linux CI).
        if src.suffix.lower() not in {".stl", ".stla", ".stlb"}:
            raise RecipeEngineError(f"load failed: {first_exc}", cause=first_exc) from first_exc
    except Exception as first_exc:
        if src.suffix.lower() not in {".stl", ".stla", ".stlb"}:
            raise RecipeEngineError(f"load failed: {first_exc}", cause=first_exc) from first_exc

    try:
        import tempfile

        from meshops.ingest.stats import load_mesh as trimesh_load

        mesh = trimesh_load(src)
        with tempfile.TemporaryDirectory(prefix="meshops_pml_") as tmp:
            ply_path = Path(tmp) / "bridge.ply"
            mesh.export(ply_path)
            _try_load(ply_path.resolve())
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
    """Save current mesh as binary STL (MeshLab default for .stl)."""
    pml = _import_pymeshlab()
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        ms.save_current_mesh(str(dest))
    except pml.PyMeshLabException as exc:
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
