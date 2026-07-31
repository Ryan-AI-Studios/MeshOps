"""Non-mutating mesh statistics and topology signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from meshops.models.diagnostics import MeshStats


def _as_trimesh(mesh: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    """Coerce Scene or Trimesh to a single Trimesh (concatenate geometry)."""
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("Scene contains no Trimesh geometry")
        if len(geoms) == 1:
            return geoms[0]
        return trimesh.util.concatenate(geoms)
    raise TypeError(f"Unsupported mesh type: {type(mesh)!r}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load an STL/PLY mesh file as a single Trimesh."""
    loaded = trimesh.load(path, force="mesh", process=False)
    return _as_trimesh(loaded)  # type: ignore[arg-type]


def compute_topology(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Best-effort non-mutating topology fields; missing → None + notes."""
    notes: list[str] = []
    result: dict[str, Any] = {
        "is_watertight": None,
        "is_volume": None,
        "is_manifold": None,
        "non_manifold_edge_count": None,
        "boundary_edge_count": None,
        "euler_characteristic": None,
        "topology_notes": notes,
    }

    try:
        result["is_watertight"] = bool(mesh.is_watertight)
    except Exception as exc:
        notes.append(f"is_watertight unavailable: {exc}")

    try:
        result["is_volume"] = bool(mesh.is_volume)
    except Exception as exc:
        notes.append(f"is_volume unavailable: {exc}")

    # Edge / manifold analysis via face adjacency (cheap for typical meshes).
    try:
        edges = mesh.edges_unique
        edge_count = len(edges)
        faces = len(mesh.faces)
        verts = len(mesh.vertices)
        result["euler_characteristic"] = verts - edge_count + faces
    except Exception as exc:
        notes.append(f"euler_characteristic unavailable: {exc}")

    try:
        # Non-manifold edges: edges shared by != 1 (boundary) or 2 (manifold) faces.
        # trimesh.edges_unique_length paired with face_adjacency covers 2-face edges;
        # edges_face counts faces per unique edge when available.
        if hasattr(mesh, "faces"):
            boundary = getattr(mesh, "edges_boundary", None)
            if boundary is not None:
                result["boundary_edge_count"] = len(boundary)
            # Non-manifold: edges with face degree > 2 (boundary deg=1 is OK).
            edges_all = np.asarray(mesh.edges_sorted)
            if len(edges_all) > 0:
                _, counts = np.unique(edges_all, axis=0, return_counts=True)
                non_manifold = int(np.sum(counts > 2))
                result["non_manifold_edge_count"] = non_manifold
                result["is_manifold"] = non_manifold == 0
            else:
                result["non_manifold_edge_count"] = 0
                result["is_manifold"] = True
        else:
            notes.append("edge degree analysis unavailable")
    except Exception as exc:
        notes.append(f"manifold analysis unavailable: {exc}")

    return result


def compute_stats(
    mesh: trimesh.Trimesh,
    *,
    mesh_id: str,
    content_sha256_hex: str,
    file_size_bytes: int,
    source_path: str | None = None,
) -> MeshStats:
    """Build MeshStats from a loaded mesh and file metadata."""
    bounds = mesh.bounds  # (2, 3)
    bbox_min = (float(bounds[0, 0]), float(bounds[0, 1]), float(bounds[0, 2]))
    bbox_max = (float(bounds[1, 0]), float(bounds[1, 1]), float(bounds[1, 2]))
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))

    try:
        components = int(mesh.body_count) if hasattr(mesh, "body_count") else 1
        # Prefer split count for multi-body (connected components by face adjacency).
        parts = mesh.split(only_watertight=False)
        components = max(1, len(parts))
    except Exception:
        components = 1

    topo = compute_topology(mesh)

    return MeshStats(
        faces=len(mesh.faces),
        vertices=len(mesh.vertices),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_diagonal=diagonal,
        components=components,
        is_watertight=topo["is_watertight"],
        is_volume=topo["is_volume"],
        is_manifold=topo["is_manifold"],
        non_manifold_edge_count=topo["non_manifold_edge_count"],
        boundary_edge_count=topo["boundary_edge_count"],
        euler_characteristic=topo["euler_characteristic"],
        file_size_bytes=file_size_bytes,
        content_sha256=content_sha256_hex,
        mesh_id=mesh_id,
        source_path=source_path,
        topology_notes=list(topo["topology_notes"]),
    )
