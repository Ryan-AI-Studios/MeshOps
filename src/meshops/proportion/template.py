"""Blank landmarks_assist.json emitter for meshops proportion template."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meshops.proportion.models import CANONICAL_VIEW_KEYS, PROPORTION_SCHEMA_VERSION

# Vertical ladder + horizontals / limbs / diameters (nulls for fill-in).
# Aligns with spec section 3.1 B / assist.KNOWN_LANDMARK_IDS primary vocabulary.
_FRONT_LANDMARK_KEYS: tuple[str, ...] = (
    "hair_crown",
    "cranial_vertex",
    "chin",
    "shoulder",
    "shoulder_l",
    "shoulder_r",
    "nipple_bust",
    "underbust",
    "navel",
    "belt_hip",
    "crotch_pubic",
    "greater_trochanter",
    "knee",
    "knee_l",
    "knee_r",
    "ankle",
    "ankle_l",
    "ankle_r",
    "sole",
    "heel",
    "heel_l",
    "heel_r",
    "midline",
    "midline_x",
    "bust_l",
    "bust_r",
    "waist_l",
    "waist_r",
    "hip_l",
    "hip_r",
    "stance_l",
    "stance_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "fingertip_l",
    "fingertip_r",
    "upper_arm_l",
    "upper_arm_r",
    "forearm_l",
    "forearm_r",
    "thigh_l",
    "thigh_r",
    "calf_l",
    "calf_r",
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
    "breast_front",
    "breast_back",
    "glute_front",
    "glute_back",
    "thigh_front",
    "thigh_back",
    "calf_front",
    "calf_back",
    "spine_hint",
)

# Top-level edge_pairs stubs (sibling of views) — fill [[x0,y0],[x1,y1]].
_EDGE_PAIR_BANDS: tuple[str, ...] = (
    "upper_arm_l",
    "upper_arm_r",
    "forearm_l",
    "forearm_r",
    "wrist_l",
    "wrist_r",
    "thigh_l",
    "thigh_r",
    "calf_l",
    "calf_r",
    "ankle_l",
    "ankle_r",
    "bust",
    "waist",
    "neck",
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
    # Top-level edge_pairs (R4 / R15) — null stubs per front band
    edge_pairs: dict[str, Any] = {
        "front": {band: None for band in _EDGE_PAIR_BANDS},
    }
    return {
        "schema_version": PROPORTION_SCHEMA_VERSION,
        "pose": "unknown",
        "multi_figure": False,
        "edge_pairs": edge_pairs,
        "views": views,
    }


def write_template(out_path: Path | str) -> Path:
    """Write blank assist JSON; create parent dirs. Returns path written."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = blank_assist_document()
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path
