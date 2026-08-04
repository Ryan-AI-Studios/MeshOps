"""Load optional landmarks_assist.json into per-view Landmark2D records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meshops.proportion.errors import ProportionError
from meshops.proportion.load_views import ViewImage
from meshops.proportion.models import (
    Landmark2D,
    PoseKind,
    ViewLandmarks,
)

# Common landmark ids (loose validation — unknown ids allowed with note).
# Edge suffixes {band}_edge0/_edge1 are allowed without spam (R5) via is_edge_landmark_id.
KNOWN_LANDMARK_IDS: frozenset[str] = frozenset(
    {
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
        "upper_arm_l",
        "upper_arm_r",
        "forearm_l",
        "forearm_r",
        "thigh_l",
        "thigh_r",
        "calf_l",
        "calf_r",
        "heel",
        "heel_l",
        "heel_r",
        "toe_l",
        "toe_r",
        "foot_front",
        "foot_back",
        "cranial_front",
        "cranial_back",
        "breast_lower",
        "breast_lower_l",
        "breast_lower_r",
        "breast_upper",
        # 0030 top-primary soft-spacing vocabulary
        "breast_center_l",
        "breast_center_r",
        "breast_medial_l",
        "breast_medial_r",
        "breast_lateral_l",
        "breast_lateral_r",
        "glute_peak_l",
        "glute_peak_r",
        "glute_cleft",
        "glute_medial_l",
        "glute_medial_r",
        "bust",
        "waist",
        "neck",
    }
)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def point_to_landmark2d(
    landmark_id: str,
    x_px: float,
    y_px: float,
    *,
    width_px: int,
    height_px: int,
    method: str = "assist",
    confidence: float = 1.0,
) -> Landmark2D:
    """Build Landmark2D with px + image-normalized fracs."""
    w = max(width_px, 1)
    h = max(height_px, 1)
    return Landmark2D(
        id=landmark_id,
        x_px=float(x_px),
        y_px=float(y_px),
        x_frac=_clamp01(float(x_px) / w),
        y_frac=_clamp01(float(y_px) / h),
        method=method,
        confidence=confidence,
    )


def _parse_landmark_value(
    landmark_id: str,
    value: Any,
    *,
    width_px: int,
    height_px: int,
) -> Landmark2D | None:
    """Accept [x,y], {x,y}, or scalar (midline_x → point on midline at mid-height)."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        # Scalar landmarks: store as (value, height/2) so x is usable.
        x_px = float(value)
        y_px = height_px / 2.0
        return point_to_landmark2d(landmark_id, x_px, y_px, width_px=width_px, height_px=height_px)

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return point_to_landmark2d(
            landmark_id,
            float(value[0]),
            float(value[1]),
            width_px=width_px,
            height_px=height_px,
        )

    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return point_to_landmark2d(
                landmark_id,
                float(value["x"]),
                float(value["y"]),
                width_px=width_px,
                height_px=height_px,
                confidence=float(value.get("confidence", 1.0)),
            )
        if "x_px" in value and "y_px" in value:
            return point_to_landmark2d(
                landmark_id,
                float(value["x_px"]),
                float(value["y_px"]),
                width_px=width_px,
                height_px=height_px,
                confidence=float(value.get("confidence", 1.0)),
            )

    raise ProportionError(
        f"invalid assist landmark {landmark_id!r}: expected [x,y], scalar, or object",
        code="invalid_assist",
        details={"landmark": landmark_id},
    )


def load_assist_json(path: Path | str) -> dict[str, Any]:
    """Load raw assist document."""
    p = Path(path)
    if not p.is_file():
        raise ProportionError(
            f"assist file not found: {p}",
            code="invalid_assist",
            details={"path": str(p)},
        )
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot parse assist JSON: {p}: {exc}",
            code="invalid_assist",
            details={"path": str(p)},
        ) from exc
    if not isinstance(raw, dict):
        raise ProportionError(
            "assist JSON root must be an object",
            code="invalid_assist",
            details={"path": str(p)},
        )
    return raw


def apply_assist(
    assist: dict[str, Any],
    view_images: dict[str, ViewImage],
) -> tuple[dict[str, ViewLandmarks], PoseKind | str, bool, list[str], dict[str, Any]]:
    """Merge assist landmarks into ViewLandmarks keyed by view.

    Returns (views, pose, multi_figure, notes, edge_pairs).
    edge_pairs is the top-level assist key (sibling of views), or {}.
    """
    from meshops.proportion.diameters import is_edge_landmark_id, parse_edge_pairs

    notes: list[str] = []
    pose: PoseKind | str = assist.get("pose", "unknown") or "unknown"
    multi_figure = bool(assist.get("multi_figure", False))
    edge_pairs = parse_edge_pairs(assist)
    views_raw = assist.get("views")
    if views_raw is None:
        views_raw = {}
    if not isinstance(views_raw, dict):
        raise ProportionError(
            "assist.views must be an object",
            code="invalid_assist",
        )

    result: dict[str, ViewLandmarks] = {}

    # Seed empty ViewLandmarks from discovered images.
    for key, img in view_images.items():
        result[key] = ViewLandmarks(
            view=key,
            width_px=img.width_px,
            height_px=img.height_px,
            path=str(img.path),
        )

    for view_key, view_data in views_raw.items():
        if not isinstance(view_data, dict):
            notes.append(f"assist view {view_key!r} is not an object; skipped")
            continue

        img = view_images.get(str(view_key))
        if img is None:
            notes.append(f"assist has view {view_key!r} but no image; skipped")
            continue

        vl = result[str(view_key)]
        facing = view_data.get("facing_direction")
        if facing:
            vl.facing_direction = str(facing)

        lm_raw = view_data.get("landmarks") or {}
        if not isinstance(lm_raw, dict):
            raise ProportionError(
                f"assist.views.{view_key}.landmarks must be an object",
                code="invalid_assist",
            )

        for lid, val in lm_raw.items():
            sid = str(lid)
            # R5: edge0/edge1 suffixes allowed without spam notes
            if sid not in KNOWN_LANDMARK_IDS and not is_edge_landmark_id(sid):
                notes.append(f"unknown landmark id {sid!r} in {view_key} (allowed)")
            lm = _parse_landmark_value(sid, val, width_px=img.width_px, height_px=img.height_px)
            if lm is not None:
                vl.landmarks[sid] = lm

        # Optional figure_span / bbox from assist
        if "figure_span_px" in view_data and view_data["figure_span_px"] is not None:
            vl.figure_span_px = float(view_data["figure_span_px"])
        if "subject_bbox" in view_data and isinstance(view_data["subject_bbox"], dict):
            bb = view_data["subject_bbox"]
            from meshops.proportion.models import SubjectBBox

            vl.subject_bbox = SubjectBBox(
                x0=float(bb["x0"]),
                y0=float(bb["y0"]),
                x1=float(bb["x1"]),
                y1=float(bb["y1"]),
            )

    return result, pose, multi_figure, notes, edge_pairs


def find_default_assist(views_dir: Path | str) -> Path | None:
    """Look for landmarks_assist.json beside views."""
    p = Path(views_dir) / "landmarks_assist.json"
    return p if p.is_file() else None
