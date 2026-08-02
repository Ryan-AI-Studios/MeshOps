"""Blank landmarks_assist.json emitter for meshops proportion template."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meshops.proportion.models import CANONICAL_VIEW_KEYS, PROPORTION_SCHEMA_VERSION

# Vertical ladder + key horizontals / limbs / depths (nulls for fill-in).
_FRONT_LANDMARK_KEYS: tuple[str, ...] = (
    "hair_crown",
    "cranial_vertex",
    "chin",
    "shoulder_l",
    "shoulder_r",
    "nipple_bust",
    "underbust",
    "navel",
    "belt_hip",
    "crotch_pubic",
    "greater_trochanter",
    "knee",
    "ankle",
    "sole",
    "midline_x",
    "hip_l",
    "hip_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "fingertip_l",
    "fingertip_r",
    "stance_l",
    "stance_r",
)

_LEFT_LANDMARK_KEYS: tuple[str, ...] = (
    "hair_crown",
    "cranial_vertex",
    "chin",
    "sole",
    "chest_front",
    "chest_back",
    "hip_front",
    "hip_back",
    "spine_hint",
)

_TQ_LANDMARK_KEYS: tuple[str, ...] = (
    "hair_crown",
    "cranial_vertex",
    "chin",
    "sole",
    "shoulder_l",
    "shoulder_r",
)

_BACK_LANDMARK_KEYS: tuple[str, ...] = (
    "hair_crown",
    "cranial_vertex",
    "chin",
    "sole",
    "midline_x",
)

_FACING: dict[str, str] = {
    "front": "camera_front",
    "left": "camera_left",
    "three_quarter": "camera_front",
    "back": "camera_back",
}

_KEYS_BY_VIEW: dict[str, tuple[str, ...]] = {
    "front": _FRONT_LANDMARK_KEYS,
    "left": _LEFT_LANDMARK_KEYS,
    "three_quarter": _TQ_LANDMARK_KEYS,
    "back": _BACK_LANDMARK_KEYS,
}


def blank_assist_document() -> dict[str, Any]:
    """Return a blank assist dict with all canonical keys (null coords)."""
    views: dict[str, Any] = {}
    for key in CANONICAL_VIEW_KEYS:
        landmarks = {lid: None for lid in _KEYS_BY_VIEW[key]}
        views[key] = {
            "facing_direction": _FACING[key],
            "landmarks": landmarks,
        }
    return {
        "schema_version": PROPORTION_SCHEMA_VERSION,
        "pose": "unknown",
        "multi_figure": False,
        "views": views,
    }


def write_template(out_path: Path | str) -> Path:
    """Write blank assist JSON; create parent dirs. Returns path written."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = blank_assist_document()
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path
