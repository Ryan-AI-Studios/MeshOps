"""t1_clean — topology heal (duplicate verts + non-manifold edges/vertices).

Filter order (PyMeshLab 2025.7):
  1. meshing_remove_duplicate_vertices
  2. meshing_repair_non_manifold_edges(method='Remove Faces')
  3. meshing_repair_non_manifold_vertices

Load STL with unify_vertices=False when dups must remain visible; prefer PLY
for honest synthetic fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meshops.recipes.pymeshlab_io import run_filter_chain

RECIPE_ID = "t1_clean"

T1_STEPS: list[tuple[str, dict[str, Any]]] = [
    ("meshing_remove_duplicate_vertices", {}),
    ("meshing_repair_non_manifold_edges", {"method": "Remove Faces"}),
    ("meshing_repair_non_manifold_vertices", {}),
]


def run_t1_clean(
    input_path: Path | str,
    output_path: Path | str,
    *,
    unify_vertices: bool = False,
) -> dict[str, Any]:
    """Run T1 topology clean chain → binary STL at output_path."""
    return run_filter_chain(
        input_path,
        output_path,
        T1_STEPS,
        unify_vertices=unify_vertices,
    )
