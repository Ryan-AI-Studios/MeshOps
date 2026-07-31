"""t2_close_small_holes — close small boundary loops only.

meshing_close_holes(maxholesize=N) uses edge-count bound and **silently skips**
holes larger than N. Wrapper post-checks remaining large boundary loops and
refuses success when large holes remain (Difficulty honesty / T2 bound).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from meshops.ingest.stats import load_mesh
from meshops.recipes.pymeshlab_io import (
    RecipeEngineError,
    apply_filter,
    load_mesh_into,
    new_mesh_set,
    save_current_mesh,
)

RECIPE_ID = "t2_close_small_holes"

# Edge count max for a "small" hole (PyMeshLab default is 30)
DEFAULT_MAX_HOLE_SIZE = 30


def _boundary_edges(mesh: object) -> object:
    """Return (n, 2) boundary edge array; works across trimesh versions."""
    import numpy as np

    # Prefer degree-1 unique edges (portable; trimesh 4.x lacks edges_boundary).
    edges = np.asarray(mesh.edges_sorted)  # type: ignore[attr-defined]
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return uniq[counts == 1]


def _mesh_for_topology(mesh_path: Path) -> object:
    """Load mesh with vertices merged so boundary-edge degree is meaningful.

    Binary STL is a triangle soup (duplicated verts). Without merge, every edge
    looks unique/boundary and loop walks are nonsense.
    """
    mesh = load_mesh(mesh_path)
    # Merge coincident vertices for topology queries only.
    with contextlib.suppress(Exception):
        mesh.merge_vertices()  # type: ignore[attr-defined]
    with contextlib.suppress(Exception):
        mesh.update_faces(mesh.unique_faces())  # type: ignore[attr-defined]
    return mesh


def _max_boundary_loop_edges(mesh_path: Path) -> int:
    """Estimate largest boundary loop edge count via trimesh."""
    from collections import defaultdict

    import numpy as np

    mesh = _mesh_for_topology(mesh_path)
    edges = np.asarray(_boundary_edges(mesh))
    if len(edges) == 0:
        return 0

    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        ai, bi = int(a), int(b)
        adj[ai].append(bi)
        adj[bi].append(ai)

    visited_edges: set[tuple[int, int]] = set()
    max_loop = 0

    def edge_key(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    for start in list(adj.keys()):
        for neigh in adj[start]:
            ek0 = edge_key(start, neigh)
            if ek0 in visited_edges:
                continue
            length = 0
            cur = neigh
            visited_edges.add(ek0)
            length += 1
            guard = 0
            while cur != start and guard < len(edges) + 2:
                guard += 1
                nxts = [n for n in adj[cur] if edge_key(cur, n) not in visited_edges]
                if not nxts:
                    break
                nxt = nxts[0]
                visited_edges.add(edge_key(cur, nxt))
                length += 1
                cur = nxt
                if cur == start:
                    break
            max_loop = max(max_loop, length)
    return max_loop


def run_t2_close_holes(
    input_path: Path | str,
    output_path: Path | str,
    *,
    maxholesize: int = DEFAULT_MAX_HOLE_SIZE,
    unify_vertices: bool = True,
) -> dict[str, Any]:
    """Close small holes; raise RecipeEngineError if large holes remain unfilled.

    When input already has a boundary loop larger than maxholesize, the filter
    will skip it — we detect remaining large loops and fail closed.
    """
    inp = Path(input_path)
    out = Path(output_path)

    large_before = _max_boundary_loop_edges(inp)

    ms = new_mesh_set()
    load_mesh_into(ms, inp, unify_vertices=unify_vertices)
    hole_metrics = apply_filter(ms, "meshing_close_holes", maxholesize=maxholesize)
    save_current_mesh(ms, out)
    with contextlib.suppress(Exception):
        ms.clear()

    large_after = _max_boundary_loop_edges(out)
    metrics: dict[str, Any] = {
        "meshing_close_holes": hole_metrics,
        "max_boundary_loop_before": large_before,
        "max_boundary_loop_after": large_after,
        "maxholesize": maxholesize,
    }

    # If a hole larger than maxholesize existed and still exists → not false success
    if large_before > maxholesize and large_after > maxholesize:
        raise RecipeEngineError(
            f"large hole remains unfilled: max_boundary_loop={large_after} "
            f"> maxholesize={maxholesize} (filter silently skips large holes)",
        )

    # If we expected close (closed_holes==0) but large hole still present, fail
    closed = 0
    if isinstance(hole_metrics, dict):
        closed = int(hole_metrics.get("closed_holes", 0) or 0)
    if closed == 0 and large_after > maxholesize:
        raise RecipeEngineError(
            f"no holes closed and large boundary remains "
            f"(max_boundary_loop={large_after} > {maxholesize})",
        )

    return metrics
