"""Synthetic T1/T2 fixtures for repair tests (PLY preferred for honest dups)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def _t1_nonmanifold_mesh() -> trimesh.Trimesh:
    """Solid-ish body + a small non-manifold flap so T1 can remove few faces.

    Recipe-tier face floor is ~0.90 — a tiny 4-face mesh would fail after
    ``Remove Faces`` on a non-manifold edge. Use a subdivided box body so
    removing a handful of flap faces still retains ≥90% faces.
    """
    body = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    body = body.subdivide().subdivide()  # denser surface

    # Non-manifold flap: three triangles share the same edge (a,b).
    # Place near body surface so bbox stays similar.
    a = np.array([5.0, 0.0, 0.0])
    b = np.array([5.0, 1.0, 0.0])
    c1 = np.array([6.0, 0.5, 0.5])
    c2 = np.array([6.0, 0.5, -0.5])
    c3 = np.array([6.5, 0.5, 0.0])
    # Near-duplicate verts for remove_duplicate_vertices
    a2 = a + np.array([1e-8, 0.0, 0.0])
    b2 = b + np.array([1e-8, 0.0, 0.0])
    c4 = c1 + np.array([0.0, 1e-8, 0.0])

    flap_verts = np.vstack([a, b, c1, c2, c3, a2, b2, c4])
    flap_faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],  # edge 0-1 has 3 faces → non-manifold
            [5, 6, 7],  # near-dup triangle
        ],
        dtype=np.int64,
    )
    flap = trimesh.Trimesh(vertices=flap_verts, faces=flap_faces, process=False)
    return trimesh.util.concatenate([body, flap])


def build_t1_nonmanifold_ply(out_dir: Path) -> Path:
    """PLY with dups + non-manifold edge (honest load; no STL auto-unify).

    Also writes companion ``t1_nonmanifold.stl`` for job-store ingest
    (original.stl naming). Prefer the PLY path when feeding PyMeshLab directly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = _t1_nonmanifold_mesh()
    ply_path = out_dir / "t1_nonmanifold.ply"
    stl_path = out_dir / "t1_nonmanifold.stl"
    mesh.export(ply_path)
    mesh.export(stl_path)
    return ply_path


def build_t1_nonmanifold_stl(out_dir: Path) -> Path:
    """STL companion for ingest → original.stl job layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "t1_nonmanifold.stl"
    _t1_nonmanifold_mesh().export(path)
    return path


def build_open_box_large_hole(out_dir: Path, *, hole_scale: float = 0.9) -> Path:
    """Open cylinder (no top) — large circular boundary for hole-close post-check.

    Boundary edge count ≈ sections (default 64) >> maxholesize=30 so
    ``meshing_close_holes`` silently skips the hole; recipe must refuse success.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "open_box_large_hole.stl"

    # High section count → large boundary loop edge count
    cyl = trimesh.creation.cylinder(radius=10.0, height=20.0, sections=64)
    normals = cyl.face_normals
    # Drop near-horizontal top cap faces (+Z)
    keep = normals[:, 2] < 0.85
    sub = cyl.submesh([np.where(keep)[0]], append=True)
    if isinstance(sub, list):
        sub = trimesh.util.concatenate(sub)
    assert isinstance(sub, trimesh.Trimesh)
    sub.export(path)
    return path


def build_solid_box_stl(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "solid_box.stl"
    mesh = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    mesh.export(path)
    return path
