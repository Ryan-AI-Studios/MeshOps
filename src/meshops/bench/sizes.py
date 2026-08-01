"""Deterministic synthetic size ladder (S/M/L/XL).

Face targets ±15% (spec §3.2). Prefer UV-sphere / exceed-then-trim — not bare
icosphere power-of-4 only. Fixed seed for any RNG path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import trimesh

SizeLabel = Literal["S", "M", "L", "XL"]

FACE_TARGETS: Final[dict[SizeLabel, int]] = {
    "S": 100_000,
    "M": 500_000,
    "L": 1_000_000,
    "XL": 2_000_000,
}

FACE_TOLERANCE_FRAC: Final[float] = 0.15
DEFAULT_SEED: Final[int] = 0x4D455348  # 'MESH'


@dataclass(frozen=True, slots=True)
class LadderMesh:
    """Generated mesh plus face accounting."""

    label: str
    target_faces: int
    mesh: trimesh.Trimesh
    path: Path | None = None

    @property
    def actual_faces(self) -> int:
        return len(self.mesh.faces)

    @property
    def verts(self) -> int:
        return len(self.mesh.vertices)

    def within_tolerance(self, frac: float = FACE_TOLERANCE_FRAC) -> bool:
        lo = self.target_faces * (1.0 - frac)
        hi = self.target_faces * (1.0 + frac)
        return lo <= self.actual_faces <= hi


def face_band(target: int, frac: float = FACE_TOLERANCE_FRAC) -> tuple[int, int]:
    """Inclusive (lo, hi) face band for target ± frac."""
    lo = int(target * (1.0 - frac))
    hi = int(target * (1.0 + frac))
    return lo, hi


def _uv_sphere_faces(count_u: int, count_v: int) -> int:
    """Create a throwaway UV sphere and return face count (source of truth)."""
    m = trimesh.creation.uv_sphere(count=[count_u, count_v])
    return len(m.faces)


def _count_for_target(target: int) -> tuple[int, int]:
    """Pick UV-sphere count≈ so faces land at or just above target.

    Empirical: faces ≈ 4 * count^2 for square count on trimesh UV spheres.
    One mesh create + optional single upward nudge — avoids multi-XL mesh builds.
    """
    # Closed-form with slight overshoot so we usually exceed target on first try.
    est = max(8, int(np.ceil(np.sqrt(max(target, 1) / 4.0) * 1.05)))
    faces = _uv_sphere_faces(est, est)
    if faces >= target:
        return est, est
    # Under-shot: scale once by sqrt ratio
    ratio = float(np.sqrt(target / max(faces, 1)))
    count = max(est + 1, int(np.ceil(est * ratio * 1.03)))
    return count, count


def generate_ladder_mesh(
    label: str,
    target_faces: int | None = None,
    *,
    seed: int = DEFAULT_SEED,
) -> LadderMesh:
    """Build a deterministic mesh whose face count is within ±15% of target.

    Strategy:
    1. UV-sphere with count chosen so faces ≥ target (parametric, no RNG).
    2. If still over the +15% band, deterministically subsample faces to
       ``target_faces`` with a fixed seed (exceed-then-trim).
    3. Record actual face count on the result.
    """
    if target_faces is None:
        key = label.upper()
        if key not in FACE_TARGETS:
            raise ValueError(f"unknown size label {label!r}; expected one of {list(FACE_TARGETS)}")
        target_faces = FACE_TARGETS[key]  # type: ignore[index]
        label = key

    if target_faces < 100:
        raise ValueError(f"target_faces too small: {target_faces}")

    cu, cv = _count_for_target(target_faces)
    mesh = trimesh.creation.uv_sphere(count=[cu, cv])
    # Ensure origin-centered for stable framing
    mesh.apply_translation(-mesh.centroid)

    lo, hi = face_band(target_faces)
    n = len(mesh.faces)
    if n > hi:
        # Exceed-then-trim: fixed-seed face subsample to exact target.
        rng = np.random.default_rng(seed ^ (target_faces & 0xFFFFFFFF))
        idx = rng.choice(n, size=target_faces, replace=False)
        idx.sort()
        sub = mesh.submesh([idx], append=True)
        if isinstance(sub, list):
            sub = trimesh.util.concatenate(sub)
        mesh = sub

    # Final sanity: still within band (exact target after trim, or raw UV).
    if not (lo <= len(mesh.faces) <= hi):
        # Last resort: if UV under-shot (shouldn't), subdivide once then trim.
        while len(mesh.faces) < lo:
            mesh = mesh.subdivide()
        n2 = len(mesh.faces)
        if n2 > hi:
            rng = np.random.default_rng(seed ^ (target_faces & 0xFFFFFFFF) ^ 0xA5A5)
            idx = rng.choice(n2, size=target_faces, replace=False)
            idx.sort()
            sub = mesh.submesh([idx], append=True)
            if isinstance(sub, list):
                sub = trimesh.util.concatenate(sub)
            mesh = sub

    return LadderMesh(label=label, target_faces=target_faces, mesh=mesh)


def write_ladder_stl(
    label: str,
    dest: Path | str,
    *,
    target_faces: int | None = None,
    seed: int = DEFAULT_SEED,
) -> LadderMesh:
    """Generate ladder mesh and export binary STL to ``dest``."""
    ladder = generate_ladder_mesh(label, target_faces, seed=seed)
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    ladder.mesh.export(path)
    return LadderMesh(
        label=ladder.label,
        target_faces=ladder.target_faces,
        mesh=ladder.mesh,
        path=path,
    )


def parse_sizes(spec: str | None) -> list[SizeLabel]:
    """Parse comma-separated size labels (case-insensitive). Default S,M,L,XL."""
    if not spec or not str(spec).strip():
        return ["S", "M", "L", "XL"]
    out: list[SizeLabel] = []
    for part in str(spec).split(","):
        token = part.strip().upper()
        if not token:
            continue
        if token not in FACE_TARGETS:
            raise ValueError(f"unknown size {part!r}; expected comma-separated S,M,L,XL")
        out.append(token)  # type: ignore[arg-type]
    if not out:
        raise ValueError("no sizes parsed from size list")
    return out
