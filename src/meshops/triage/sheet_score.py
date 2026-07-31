"""2-stage multi-feature sheet/ribbon score (Difficulty §2, §7; N8).

Neighborhood radius is scale-invariant: r = k * bbox_diagonal with k=0.02.

Stage 1 (fast): merge vertices → sample face centroids; batched cKDTree local
  PCA planar cues. Optional thickness rays if a ray backend is available
  (rtree/embree); algorithm remains correct without them.

Stage 2 (targeted / topology): face_adjacency_angles flat clustering,
  AABB thinness of large coplanar regions, section_multiplane, principal
  inertia, clothing discrimination via large smooth planar fraction.

auto_action is never "delete" (N8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[attr-defined]

from meshops.models.diagnostics import (
    AutoAction,
    SheetScoreFeatures,
    SheetScoreResult,
)

if TYPE_CHECKING:
    import trimesh

# Scale-invariant neighborhood: r = k * bbox_diagonal
NEIGHBORHOOD_K = 0.02
THINNESS_CANDIDATE_THRESHOLD = 0.55
REL_THICKNESS_SHEET = 0.04
MAX_STAGE1_SAMPLES = 40_000
KNN_K = 16
MIN_NEIGHBORS = 6
MAX_THICKNESS_RAYS = 4_000


def _bbox_diagonal(mesh: trimesh.Trimesh) -> float:
    bounds = mesh.bounds
    return float(np.linalg.norm(bounds[1] - bounds[0]))


def _prepare_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Copy + merge vertices so face adjacency works on raw STL imports."""
    m = mesh.copy()
    try:
        m.merge_vertices()
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
    except Exception:
        pass
    return m


def _sample_face_indices(n_faces: int, max_samples: int) -> np.ndarray:
    if n_faces <= max_samples:
        return np.arange(n_faces, dtype=np.int64)
    rng = np.random.default_rng(42)
    return rng.choice(n_faces, size=max_samples, replace=False)


def _pca_planar_cue(pts: np.ndarray) -> float:
    """Planar cue ∈ [0,1] from local covariance (high = flat areal neighborhood)."""
    if len(pts) < MIN_NEIGHBORS:
        return 0.0
    centered = pts - pts.mean(axis=0)
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    try:
        eigvals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 0.0
    eigvals = np.clip(eigvals, 0.0, None)
    lam0, lam1, lam2 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    if lam2 <= 1e-18:
        return 0.0
    flatness = 1.0 - (lam0 / lam2)
    areal = lam1 / lam2
    return float(np.clip(flatness * (0.5 + 0.5 * areal), 0.0, 1.0))


def _local_planar_cues(
    points: np.ndarray,
    tree: Any,
    radius: float,
) -> np.ndarray:
    """Per-point planar cue via radius ball with kNN fallback."""
    n = len(points)
    cues = np.zeros(n, dtype=np.float64)
    knn_dists, knn_idx = tree.query(points, k=min(KNN_K + 1, n))
    neighbors_list = tree.query_ball_point(points, r=radius)

    for i, nbrs in enumerate(neighbors_list):
        if len(nbrs) >= MIN_NEIGHBORS:
            pts = points[nbrs]
        else:
            idxs = np.atleast_1d(knn_idx[i])
            dists = np.atleast_1d(knn_dists[i])
            mask = dists > 1e-15
            idxs, dists = idxs[mask], dists[mask]
            within = dists <= max(3.0 * radius, 1e-12)
            if np.sum(within) >= MIN_NEIGHBORS:
                idxs = idxs[within]
            if len(idxs) < MIN_NEIGHBORS:
                continue
            pts = points[idxs]
        cues[i] = _pca_planar_cue(pts)
    return cues


def _optional_thickness_sheet_frac(
    mesh: trimesh.Trimesh,
    face_idx: np.ndarray,
    diagonal: float,
) -> float | None:
    """Optional ray thickness sheet fraction; None if ray backend unavailable."""
    if len(face_idx) == 0 or diagonal <= 1e-12:
        return None
    if len(face_idx) > MAX_THICKNESS_RAYS:
        rng = np.random.default_rng(0)
        face_idx = rng.choice(face_idx, size=MAX_THICKNESS_RAYS, replace=False)

    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)[face_idx]
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_idx]
    eps = max(1e-6 * diagonal, 1e-9)
    origins_in = centroids - normals * eps

    try:
        locations, index_ray, _index_tri = mesh.ray.intersects_location(
            ray_origins=origins_in,
            ray_directions=-normals,
            multiple_hits=False,
        )
    except Exception:
        return None

    if len(index_ray) == 0:
        return 0.0

    thicknesses = np.full(len(face_idx), diagonal, dtype=np.float64)
    for hit_i, ray_i in enumerate(index_ray):
        ri = int(ray_i)
        if 0 <= ri < len(thicknesses):
            dist = float(np.linalg.norm(locations[hit_i] - origins_in[ri]))
            if dist > eps:
                thicknesses[ri] = min(thicknesses[ri], dist)
    rel = thicknesses / diagonal
    return float(np.mean(rel < REL_THICKNESS_SHEET))


def _flat_region_features(mesh: trimesh.Trimesh) -> dict[str, float]:
    """Stage-2: coplanar face clustering + section cues + clothing penalty."""
    out = {
        "planarity": 0.0,
        "section_thinness": 0.0,
        "dihedral_crease": 0.0,
        "normal_smoothness": 0.0,
        "clothing_penalty": 0.0,
        "max_flat_area_fraction": 0.0,
        "flat_aabb_thinness": 0.0,
    }
    try:
        angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
        adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    except Exception:
        return out

    if len(angles) == 0 or len(adjacency) == 0:
        return out

    flat_thresh = 0.15
    is_flat = angles < flat_thresh
    is_sharp = angles > 0.8
    out["normal_smoothness"] = float(np.mean(is_flat))
    out["dihedral_crease"] = float(np.mean(is_sharp))

    n_faces = len(mesh.faces)
    parent = np.arange(n_faces, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (fa, fb), flat in zip(adjacency, is_flat, strict=False):
        if flat:
            union(int(fa), int(fb))

    roots = np.array([find(i) for i in range(n_faces)], dtype=np.int64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total_area = float(areas.sum()) + 1e-12
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)

    best_score = -1.0
    best_frac = 0.0
    best_thin = 0.0
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) + 1e-12
    ref_area = diag * diag  # scale-free area reference

    for root in np.unique(roots):
        mask = roots == root
        cluster_area = float(areas[mask].sum())
        frac = cluster_area / total_area
        if cluster_area < 1e-12:
            continue
        face_ids = np.nonzero(mask)[0]
        cluster_verts = verts[faces[face_ids].ravel()]
        extents = np.sort(cluster_verts.max(axis=0) - cluster_verts.min(axis=0))
        min_ext, mid_ext, max_ext = float(extents[0]), float(extents[1]), float(extents[2])
        if max_ext <= 1e-12:
            continue
        thin = 1.0 - (min_ext / max_ext)
        areal = mid_ext / max_ext
        # Require thin AABB (sheet/slab), skip bulky regions
        if thin < 0.75:
            continue
        # Absolute size matter: small caps on large solids should not dominate.
        # Use max(frac, area/ref) so a large sheet on a multi-body mesh still scores.
        size_term = max(frac, min(1.0, 4.0 * cluster_area / ref_area))
        if size_term < 0.03:
            continue
        score_key = size_term * thin * (0.5 + 0.5 * areal)
        if score_key > best_score:
            best_score = score_key
            best_frac = frac
            best_thin = thin

    out["max_flat_area_fraction"] = best_frac
    out["flat_aabb_thinness"] = best_thin
    out["planarity"] = float(np.clip(max(best_score, 0.0) if best_score > 0 else 0.0, 0.0, 1.0))

    try:
        vectors = np.asarray(mesh.principal_inertia_vectors, dtype=np.float64)
        if vectors.shape == (3, 3):
            thin_axis = vectors[:, 0]
            thin_axis = thin_axis / (np.linalg.norm(thin_axis) + 1e-12)
            sections = mesh.section_multiplane(
                plane_origin=mesh.centroid,
                plane_normal=thin_axis,
                heights=[0.0],
            )
            if sections and sections[0] is not None:
                sec = sections[0]
                if hasattr(sec, "extents"):
                    ext = np.asarray(sec.extents, dtype=np.float64)
                    if ext.size >= 2 and float(ext.max()) > 1e-12:
                        out["section_thinness"] = float(1.0 - float(ext.min() / ext.max()))
                elif hasattr(sec, "vertices") and len(sec.vertices) > 1:
                    vv = np.asarray(sec.vertices, dtype=np.float64)
                    span = np.sort(vv.max(axis=0) - vv.min(axis=0))
                    if float(span[-1]) > 1e-12:
                        out["section_thinness"] = float(1.0 - float(span[0] / span[-1]))
    except Exception:
        pass

    # Clothing / cape: large smooth planar region (Difficulty §7).
    # Capes show high planarity + smoothness; arm sheets are more localized.
    # Penalty reduces score; auto_action still never delete (N8).
    if out["planarity"] > 0.55 and out["normal_smoothness"] > 0.55 and best_thin > 0.9:
        out["clothing_penalty"] = 0.4
    if out["planarity"] > 0.75 and out["normal_smoothness"] > 0.6:
        out["clothing_penalty"] = 0.55

    return out


def compute_sheet_score(
    mesh: trimesh.Trimesh,
    *,
    neighborhood_k: float = NEIGHBORHOOD_K,
    max_samples: int = MAX_STAGE1_SAMPLES,
) -> SheetScoreResult:
    """Compute multi-feature sheet_score ∈ [0,1]. auto_action never delete."""
    notes: list[str] = []
    work = _prepare_mesh(mesh)
    diag = _bbox_diagonal(work)
    if diag <= 1e-12 or len(work.faces) == 0:
        return SheetScoreResult(
            score=0.0,
            confidence=0.0,
            features=SheetScoreFeatures(neighborhood_k=neighborhood_k),
            auto_action=AutoAction.NONE,
            notes=["degenerate mesh"],
        )

    radius = neighborhood_k * diag
    face_idx = _sample_face_indices(len(work.faces), max_samples)
    centroids = np.asarray(work.triangles_center, dtype=np.float64)[face_idx]
    tree = cKDTree(centroids)

    planar = _local_planar_cues(centroids, tree, radius)
    thin_mean = float(np.mean(planar))
    thin_p95 = float(np.percentile(planar, 95))
    cand_mask = planar >= THINNESS_CANDIDATE_THRESHOLD
    n_cand = int(np.sum(cand_mask))
    cand_frac = float(n_cand / max(len(planar), 1))

    ray_faces = face_idx[cand_mask] if n_cand >= 8 else face_idx
    sheet_frac_opt = _optional_thickness_sheet_frac(work, ray_faces, diag)
    if sheet_frac_opt is None:
        notes.append("thickness_rays_unavailable_no_rtree")
        thickness_boost = 0.0
    else:
        notes.append(f"thickness_sheet_frac={sheet_frac_opt:.4f}")
        thickness_boost = float(sheet_frac_opt)

    s2 = _flat_region_features(work)
    stage2_used = True
    notes.append("stage2_flat_clusters_and_sections")
    if n_cand >= 3:
        notes.append("stage1_candidates_present")

    # Primary discriminator: large thin coplanar clusters (Stage 2).
    # Stage-1 planar cues fire on any surface (including curved solids) so they
    # are down-weighted. Optional ray thickness boosts when available.
    slab = s2["planarity"]
    slab_area = min(1.0, s2["max_flat_area_fraction"] / 0.15)
    slab_thin = s2["flat_aabb_thinness"]

    raw = (
        0.45 * slab
        + 0.20 * slab_thin * slab_area
        + 0.10 * s2["section_thinness"] * min(1.0, s2["max_flat_area_fraction"] / 0.1)
        + 0.10 * thickness_boost
        + 0.08 * max(0.0, thin_p95 - 0.55)
        + 0.05 * max(0.0, thin_mean - 0.5)
        + 0.02 * min(1.0, cand_frac)
    )
    raw = float(np.clip(raw - 0.55 * s2["clothing_penalty"], 0.0, 1.0))

    conf = 0.5 + 0.2 * min(1.0, len(face_idx) / 2000) + 0.15
    if s2["max_flat_area_fraction"] > 0.05:
        conf += 0.1
    conf = float(np.clip(conf - 0.1 * s2["clothing_penalty"], 0.0, 1.0))

    features = SheetScoreFeatures(
        thinness_mean=thin_mean,
        thinness_p95=thin_p95,
        candidate_fraction=cand_frac,
        planarity=s2["planarity"],
        section_thinness=s2["section_thinness"],
        dihedral_crease=s2["dihedral_crease"],
        normal_smoothness=s2["normal_smoothness"],
        clothing_penalty=s2["clothing_penalty"],
        neighborhood_k=neighborhood_k,
        neighborhood_radius=radius,
        n_samples=len(face_idx),
        n_candidates=n_cand,
        stage2_used=stage2_used,
    )

    if raw >= 0.55:
        action = AutoAction.REVIEW
        notes.append("elevated_sheet_score_review_only")
    elif raw >= 0.35:
        action = AutoAction.REVIEW
    else:
        action = AutoAction.NONE

    if s2["clothing_penalty"] > 0.3:
        notes.append("clothing_like_plane_penalty_applied")
        action = AutoAction.REVIEW if raw >= 0.35 else AutoAction.NONE

    return SheetScoreResult(
        score=raw,
        confidence=conf,
        features=features,
        auto_action=action,
        notes=notes,
    )
