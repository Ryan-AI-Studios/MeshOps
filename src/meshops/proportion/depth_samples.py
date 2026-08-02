"""Sparse metric depth samples + optional mesh ray deltas (track 0017).

Export depth_at_landmarks.json from ProportionReport (depth_bands + fused y),
and optional depth_mesh_deltas.json via trimesh Y-axis rays against a blockout.

Authoring measurement aids only — not mesh or print success (Difficulty §12 / N6).

Sign contract (mesh deltas):
  delta_y_m     = ref_y_m - mesh_y_m       # positive → mesh shallower (less +Y)
  delta_depth_m = ref_depth_m - mesh_depth_m  # positive → mesh thinner than ref
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.analyze import load_report
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import DEPTH_HONESTY
from meshops.proportion.models import LandmarkXYZ, ProportionReport

DEPTH_SAMPLES_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
DEPTH_DELTAS_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"

SAMPLES_BASENAME: Final[str] = "depth_at_landmarks.json"
DELTAS_BASENAME: Final[str] = "depth_mesh_deltas.json"

AXIS_NOTES: Final[str] = (
    "Z-up soles=0; +X camera-right; body +Y = front depth "
    "(y_front > y_back after orientation fix; +Y toward camera on camera_left); "
    "y_m is body depth not image y; blockout face -Y is placement (toes -Y) "
    "separate from depth y_m sign"
)

DepthSampleRole = Literal[
    "landmark",
    "band_front",
    "band_back",
    "band_mid",
    "band_span",
]
DepthSampleSource = Literal["fused_xyz", "depth_band"]
DepthMethod = Literal["trimesh_ray_y"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DepthSample(BaseModel):
    """One sparse depth / landmark sample (meters + fracs)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    role: DepthSampleRole
    y_m: float | None = None
    depth_m: float | None = None
    depth_frac: float | None = None
    x_m: float | None = None
    z_m: float | None = None
    x_frac: float | None = None
    y_frac: float | None = None
    z_frac: float | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: DepthSampleSource
    view: str | None = None
    orientation_swapped: bool = False
    band_id: str | None = None


class DepthSamplesPackage(BaseModel):
    """depth_at_landmarks.json package (schema 1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = DEPTH_SAMPLES_SCHEMA_VERSION
    honesty: str = DEPTH_HONESTY
    source_report_schema: str | None = None
    height_m: float | None = None
    axis_notes: str = AXIS_NOTES
    samples: list[DepthSample] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class DepthDelta(BaseModel):
    """One ref-mesh depth delta at a sample (meters)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    ref_y_m: float | None = None
    mesh_y_m: float | None = None
    delta_y_m: float | None = None
    ref_depth_m: float | None = None
    mesh_depth_m: float | None = None
    delta_depth_m: float | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    method: DepthMethod = "trimesh_ray_y"


class DepthDeltasPackage(BaseModel):
    """depth_mesh_deltas.json package (schema 1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = DEPTH_DELTAS_SCHEMA_VERSION
    honesty: str = DEPTH_HONESTY
    mesh_path: str
    method: DepthMethod = "trimesh_ray_y"
    deltas: list[DepthDelta] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


def extract_depth_samples(report: ProportionReport) -> DepthSamplesPackage:
    """Build DepthSamplesPackage from report depth_bands + landmarks_xyz (R3).

    Raises ProportionError(depth_empty) when zero samples after extraction.
    """
    messages: list[str] = []
    samples: list[DepthSample] = []
    height_m = report.height_m

    if height_m is None:
        messages.append("height_m unset — meters null")

    if report.quality.multi_figure:
        messages.append(
            "quality.multi_figure: depth samples still emitted — confirm primary figure"
        )
    if report.quality.needs_user_input:
        messages.append(
            "quality.needs_user_input: depth samples still emitted — confirm before acting"
        )

    for band in report.depth_bands:
        bid = band.band_id
        conf = float(band.confidence)
        swapped = bool(band.orientation_swapped)
        z_frac = band.z_frac
        y_front_m = band.y_front * height_m if height_m is not None else None
        y_back_m = band.y_back * height_m if height_m is not None else None
        y_mid_m = band.y_mid * height_m if height_m is not None else None

        samples.append(
            DepthSample(
                id=f"band_{bid}_front",
                role="band_front",
                y_m=y_front_m,
                y_frac=band.y_front,
                z_frac=z_frac,
                confidence=conf,
                source="depth_band",
                view="left",
                orientation_swapped=swapped,
                band_id=bid,
            )
        )
        samples.append(
            DepthSample(
                id=f"band_{bid}_back",
                role="band_back",
                y_m=y_back_m,
                y_frac=band.y_back,
                z_frac=z_frac,
                confidence=conf,
                source="depth_band",
                view="left",
                orientation_swapped=swapped,
                band_id=bid,
            )
        )
        samples.append(
            DepthSample(
                id=f"band_{bid}_mid",
                role="band_mid",
                y_m=y_mid_m,
                y_frac=band.y_mid,
                z_frac=z_frac,
                confidence=conf,
                source="depth_band",
                view="left",
                orientation_swapped=swapped,
                band_id=bid,
            )
        )
        samples.append(
            DepthSample(
                id=f"band_{bid}_span",
                role="band_span",
                y_m=y_mid_m,
                depth_m=band.depth_m,
                depth_frac=band.depth_frac,
                y_frac=band.y_mid,
                z_frac=z_frac,
                confidence=conf,
                source="depth_band",
                view="left",
                orientation_swapped=swapped,
                band_id=bid,
            )
        )

    skipped_lm = 0
    for key, lm in report.landmarks_xyz.items():
        if lm.y_m is None and lm.y is None:
            skipped_lm += 1
            continue
        samples.append(
            DepthSample(
                id=key,
                role="landmark",
                y_m=lm.y_m,
                x_m=lm.x_m,
                z_m=lm.z_m,
                x_frac=lm.x,
                y_frac=lm.y,
                z_frac=lm.z,
                confidence=float(lm.confidence),
                source="fused_xyz",
                view=None,
                orientation_swapped=False,
                band_id=None,
            )
        )

    if skipped_lm > 0:
        messages.append(f"{skipped_lm} landmarks skipped (no depth y)")

    if not samples:
        raise ProportionError(
            "no depth samples: empty depth_bands and no landmarks with depth y",
            code="depth_empty",
        )

    return DepthSamplesPackage(
        schema_version=DEPTH_SAMPLES_SCHEMA_VERSION,
        honesty=DEPTH_HONESTY,
        source_report_schema=report.schema_version,
        height_m=height_m,
        axis_notes=AXIS_NOTES,
        samples=samples,
        messages=messages,
        counts={"samples": len(samples)},
    )


# ---------------------------------------------------------------------------
# Mesh rays
# ---------------------------------------------------------------------------


def _y_far(mesh: trimesh.Trimesh, height_m: float | None) -> float:
    """Dynamic ray origin offset outside mesh AABB (R4 AI1)."""
    ym_min = float(mesh.bounds[0, 1])
    ym_max = float(mesh.bounds[1, 1])
    extent_y = abs(ym_max - ym_min)
    h = height_m if height_m is not None else 0.0
    return max(10.0, 2.0 * max(abs(ym_min), abs(ym_max), extent_y, h, 1.0))


def _resolve_sample_xz(
    sample: DepthSample,
    *,
    landmarks_xyz: dict[str, LandmarkXYZ],
    samples_by_id: dict[str, DepthSample],
) -> tuple[float | None, float | None]:
    """Return (x_m, z_m) for ray casting; pair band_* to fused when needed."""
    if sample.x_m is not None and sample.z_m is not None:
        return sample.x_m, sample.z_m

    # band_chest_front → chest_front (strip leading "band_")
    if sample.id.startswith("band_"):
        pair_key = sample.id[len("band_") :]
        lm = landmarks_xyz.get(pair_key)
        if lm is not None and lm.x_m is not None and lm.z_m is not None:
            return float(lm.x_m), float(lm.z_m)
        other = samples_by_id.get(pair_key)
        if other is not None and other.x_m is not None and other.z_m is not None:
            return float(other.x_m), float(other.z_m)

    return None, None


def _ray_hits_trimesh(
    mesh: trimesh.Trimesh,
    origins: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (locations, index_ray) via trimesh ray API; None if unavailable."""
    try:
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=True,
        )
    except Exception:
        return None
    if locations is None or len(locations) == 0:
        return None
    return np.asarray(locations, dtype=np.float64), np.asarray(index_ray, dtype=np.int64)


def _ray_hits_bruteforce(
    mesh: trimesh.Trimesh,
    origins: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Brute-force Moller-Trumbore multi-hit rays (no rtree required).

    Used when trimesh ray backend cannot build a triangle bounds tree (missing
    optional rtree). Acceptable for sparse landmark samples on blockout meshes.
    """
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.size == 0:
        return None

    # v0, v1, v2 for each triangle
    v0 = triangles[:, 0, :]
    edge1 = triangles[:, 1, :] - v0
    edge2 = triangles[:, 2, :] - v0
    eps = 1e-12

    hit_locs: list[list[float]] = []
    hit_rays: list[int] = []

    for ri in range(len(origins)):
        origin = origins[ri]
        direction = directions[ri]
        # pvec = dir cross edge2
        pvec = np.cross(direction, edge2)
        det = np.einsum("ij,ij->i", edge1, pvec)
        abs_det = np.abs(det)
        valid = abs_det > eps
        if not np.any(valid):
            continue
        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]
        tvec = origin - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        valid &= (u >= 0.0) & (u <= 1.0)
        if not np.any(valid):
            continue
        qvec = np.cross(tvec, edge1)
        v = np.einsum("ij,ij->i", np.broadcast_to(direction, edge2.shape), qvec) * inv_det
        valid &= (v >= 0.0) & ((u + v) <= 1.0)
        if not np.any(valid):
            continue
        t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
        valid &= t > eps
        if not np.any(valid):
            continue
        hits = origin + direction * t[valid, None]
        for row in hits:
            hit_locs.append([float(row[0]), float(row[1]), float(row[2])])
            hit_rays.append(ri)

    if not hit_locs:
        return None
    return np.asarray(hit_locs, dtype=np.float64), np.asarray(hit_rays, dtype=np.int64)


def _ray_front_back(
    mesh: trimesh.Trimesh,
    *,
    x_m: float,
    z_m: float,
    y_far: float,
) -> tuple[float, float] | None:
    """Cast +Y and -Y rays; return (mesh_y_front, mesh_y_back) or None on miss.

    Front: origin (x, +Y_FAR, z) dir (0,-1,0) → max hit.y
    Back:  origin (x, -Y_FAR, z) dir (0,+1,0) → min hit.y

    Prefers mesh.ray.intersects_location (sheet_score path); falls back to a
    pure-Python multi-hit caster when rtree is unavailable.
    """
    origins = np.array(
        [
            [x_m, y_far, z_m],
            [x_m, -y_far, z_m],
        ],
        dtype=np.float64,
    )
    directions = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    hit = _ray_hits_trimesh(mesh, origins, directions)
    if hit is None:
        hit = _ray_hits_bruteforce(mesh, origins, directions)
    if hit is None:
        return None

    locations, index_ray = hit
    front_ys: list[float] = []
    back_ys: list[float] = []
    for hit_i, ray_i in enumerate(index_ray):
        y = float(locations[hit_i][1])
        if int(ray_i) == 0:
            front_ys.append(y)
        elif int(ray_i) == 1:
            back_ys.append(y)

    if not front_ys or not back_ys:
        return None

    mesh_y_front = max(front_ys)
    mesh_y_back = min(back_ys)
    return mesh_y_front, mesh_y_back


def load_mesh_for_deltas(path: Path | str) -> trimesh.Trimesh:
    """Load mesh for ray sampling; raise mesh_load_failed on failure."""
    p = Path(path)
    try:
        loaded = trimesh.load(str(p), force="mesh", process=True)
    except Exception as exc:
        raise ProportionError(
            f"cannot load mesh for depth deltas: {p}: {exc}",
            code="mesh_load_failed",
            details={"path": str(p)},
        ) from exc

    if not isinstance(loaded, trimesh.Trimesh):
        raise ProportionError(
            f"mesh load did not return a Trimesh: {p} ({type(loaded).__name__})",
            code="mesh_load_failed",
            details={"path": str(p)},
        )
    if loaded.is_empty or len(loaded.vertices) == 0:
        raise ProportionError(
            f"mesh is empty: {p}",
            code="mesh_load_failed",
            details={"path": str(p)},
        )
    return loaded


def compute_mesh_deltas(
    samples: list[DepthSample],
    mesh: trimesh.Trimesh,
    *,
    mesh_path: str,
    height_m: float | None,
    landmarks_xyz: dict[str, LandmarkXYZ] | None = None,
) -> DepthDeltasPackage:
    """Ray-sample mesh on Y axis; build DepthDeltasPackage (R4).

    If height_m is None: empty deltas + message (not an error).
    skipped_mesh counts ray skips only (missing xz / miss).
    """
    messages: list[str] = []
    landmarks_xyz = landmarks_xyz or {}

    if height_m is None:
        return DepthDeltasPackage(
            schema_version=DEPTH_DELTAS_SCHEMA_VERSION,
            honesty=DEPTH_HONESTY,
            mesh_path=mesh_path,
            method="trimesh_ray_y",
            deltas=[],
            messages=["height_m unset — mesh deltas empty (need stature meters)"],
            counts={"deltas": 0, "skipped": 0},
        )

    z_floor = -0.05 * float(height_m if height_m is not None else 1.0)
    min_z = float(mesh.bounds[0, 2])
    if min_z < z_floor:
        messages.append(
            f"mesh min Z={min_z} below soles plane — ensure soles at Z=0 for accurate deltas"
        )

    samples_by_id = {s.id: s for s in samples}
    y_far = _y_far(mesh, height_m)
    deltas: list[DepthDelta] = []
    skipped = 0

    for sample in samples:
        x_m, z_m = _resolve_sample_xz(
            sample,
            landmarks_xyz=landmarks_xyz,
            samples_by_id=samples_by_id,
        )
        if x_m is None or z_m is None:
            skipped += 1
            continue

        hit = _ray_front_back(mesh, x_m=float(x_m), z_m=float(z_m), y_far=y_far)
        if hit is None:
            skipped += 1
            continue

        mesh_y_front, mesh_y_back = hit
        mesh_depth_m = mesh_y_front - mesh_y_back
        mesh_y_mid = (mesh_y_front + mesh_y_back) / 2.0
        ref_y_m = sample.y_m
        ref_depth_m = sample.depth_m
        delta_y_m = (ref_y_m - mesh_y_mid) if ref_y_m is not None else None
        delta_depth_m = (ref_depth_m - mesh_depth_m) if ref_depth_m is not None else None

        deltas.append(
            DepthDelta(
                id=sample.id,
                ref_y_m=ref_y_m,
                mesh_y_m=mesh_y_mid,
                delta_y_m=delta_y_m,
                ref_depth_m=ref_depth_m,
                mesh_depth_m=mesh_depth_m,
                delta_depth_m=delta_depth_m,
                confidence=float(sample.confidence),
                method="trimesh_ray_y",
            )
        )

    return DepthDeltasPackage(
        schema_version=DEPTH_DELTAS_SCHEMA_VERSION,
        honesty=DEPTH_HONESTY,
        mesh_path=mesh_path,
        method="trimesh_ray_y",
        deltas=deltas,
        messages=messages,
        counts={"deltas": len(deltas), "skipped": skipped},
    )


# ---------------------------------------------------------------------------
# Write paths
# ---------------------------------------------------------------------------


def _resolve_out_paths(
    out: Path,
    *,
    with_mesh: bool,
) -> tuple[Path, Path | None]:
    """Resolve samples path and optional deltas path from --out (R1).

    Returns (samples_path, deltas_path|None).
    """
    s = str(out)
    ends_sep = s.endswith(("/", "\\"))
    if (out.exists() and out.is_dir()) or ends_sep:
        is_dir = True
    elif out.suffix.lower() == ".json":
        is_dir = False
    else:
        # file not ending .json and not a directory → depth_failed
        raise ProportionError(
            "--out file must end with .json or be a directory",
            code="depth_failed",
            details={"out": str(out)},
        )

    if is_dir:
        samples_path = out / SAMPLES_BASENAME
        deltas_path = (out / DELTAS_BASENAME) if with_mesh else None
        return samples_path, deltas_path

    # file .json
    samples_path = out
    deltas_path = (out.parent / DELTAS_BASENAME) if with_mesh else None
    return samples_path, deltas_path


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            raise ProportionError(
                f"output already exists (use --force): {path}",
                code="write_failed",
                details={"path": str(path)},
            )
        path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    except ProportionError:
        raise
    except OSError as exc:
        raise ProportionError(
            f"failed to write depth samples: {exc}",
            code="write_failed",
            details={"path": str(path)},
        ) from exc


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def run_depth_samples(
    report: Path | ProportionReport,
    out: Path,
    *,
    mesh: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Extract depth samples (+ optional mesh deltas) and write JSON packages.

    Returns the CLI/MCP success payload shape. Raises ProportionError.
    Does not call CLI emit helpers.
    """
    rep = report if isinstance(report, ProportionReport) else load_report(report)

    package = extract_depth_samples(rep)
    out_path = Path(out)
    samples_path, deltas_path = _resolve_out_paths(out_path, with_mesh=mesh is not None)

    paths: list[str] = []
    messages = list(package.messages)
    skipped_mesh = 0
    deltas_count = 0

    _write_json(
        samples_path,
        package.model_dump(mode="json"),
        force=force,
    )
    paths.append(str(samples_path))

    if mesh is not None:
        mesh_path = Path(mesh)
        tri = load_mesh_for_deltas(mesh_path)
        deltas_pkg = compute_mesh_deltas(
            package.samples,
            tri,
            mesh_path=str(mesh_path),
            height_m=rep.height_m,
            landmarks_xyz=rep.landmarks_xyz,
        )
        messages.extend(deltas_pkg.messages)
        skipped_mesh = int(deltas_pkg.counts.get("skipped", 0))
        deltas_count = int(deltas_pkg.counts.get("deltas", 0))
        assert deltas_path is not None
        _write_json(
            deltas_path,
            deltas_pkg.model_dump(mode="json"),
            force=force,
        )
        paths.append(str(deltas_path))

    return {
        "ok": True,
        "paths": paths,
        "counts": {
            "samples": int(package.counts.get("samples", len(package.samples))),
            "deltas": deltas_count,
            "skipped_mesh": skipped_mesh,
        },
        "messages": messages,
    }


__all__ = [
    "AXIS_NOTES",
    "DELTAS_BASENAME",
    "DEPTH_DELTAS_SCHEMA_VERSION",
    "DEPTH_SAMPLES_SCHEMA_VERSION",
    "SAMPLES_BASENAME",
    "DepthDelta",
    "DepthDeltasPackage",
    "DepthSample",
    "DepthSamplesPackage",
    "compute_mesh_deltas",
    "extract_depth_samples",
    "load_mesh_for_deltas",
    "run_depth_samples",
]
