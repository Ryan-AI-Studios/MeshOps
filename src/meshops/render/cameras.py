"""Bbox-relative orthographic camera poses (scale/origin-invariant).

View names are camera names (front/back/left/right/top/bottom/three_quarter),
not anatomical left/right (Difficulty §1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Simple look-at camera for F3D framing."""

    name: str
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    # Ortho half-extent roughly proportional to bbox diagonal
    ortho_scale: float


def bbox_cameras(
    bbox_min: tuple[float, float, float] | np.ndarray,
    bbox_max: tuple[float, float, float] | np.ndarray,
    *,
    distance_factor: float = 1.5,
) -> list[CameraPose]:
    """Six principal orthographic views + one three-quarter.

    Cameras sit on a sphere scaled by bbox diagonal around the center.
    """
    bmin = np.asarray(bbox_min, dtype=np.float64)
    bmax = np.asarray(bbox_max, dtype=np.float64)
    center = 0.5 * (bmin + bmax)
    diagonal = float(np.linalg.norm(bmax - bmin))
    if diagonal <= 1e-12:
        diagonal = 1.0
    dist = distance_factor * diagonal
    ortho = diagonal * 0.6  # half-height-ish framing

    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    focal = (cx, cy, cz)

    # Convention: +Z up when possible; for top/bottom adjust view_up.
    specs: list[tuple[str, np.ndarray, tuple[float, float, float]]] = [
        ("front", np.array([0.0, -1.0, 0.0]), (0.0, 0.0, 1.0)),
        ("back", np.array([0.0, 1.0, 0.0]), (0.0, 0.0, 1.0)),
        ("left", np.array([-1.0, 0.0, 0.0]), (0.0, 0.0, 1.0)),
        ("right", np.array([1.0, 0.0, 0.0]), (0.0, 0.0, 1.0)),
        ("top", np.array([0.0, 0.0, 1.0]), (0.0, 1.0, 0.0)),
        ("bottom", np.array([0.0, 0.0, -1.0]), (0.0, 1.0, 0.0)),
        ("three_quarter", np.array([0.7, -0.7, 0.4]), (0.0, 0.0, 1.0)),
    ]

    poses: list[CameraPose] = []
    for name, direction, view_up in specs:
        d = direction / (np.linalg.norm(direction) + 1e-12)
        pos = center + d * dist
        poses.append(
            CameraPose(
                name=name,
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                focal_point=focal,
                view_up=view_up,
                ortho_scale=ortho,
            )
        )
    return poses
