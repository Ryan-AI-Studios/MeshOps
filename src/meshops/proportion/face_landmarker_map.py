"""Face Landmarker 478 → 0124 form-read id map (track 0130).

MeshOps 3.13 owns the frozen table. The optional Python 3.12 sidecar reads the
same JSON; this module never imports mediapipe.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Final

from meshops.proportion.assist import (
    FACE_FRONT_LANDMARK_IDS,
    FACE_LEFT_LANDMARK_IDS,
    FACE_TQ_LANDMARK_IDS,
    KNOWN_LANDMARK_IDS,
)
from meshops.proportion.honesty import LANDMARKER_HONESTY

MAP_SCHEMA_VERSION: Final[str] = "1.0.0"
MAP_RESOURCE_NAME: Final[str] = "face_landmarker_map.json"
MESH_POINT_COUNT: Final[int] = 478
LANDMARKER_SKIP_REASONS: Final[frozenset[str]] = frozenset(
    {"tool_missing", "model_missing", "no_face", "multi_face", "bad_image"}
)

_VIEW_IDS: Final[dict[str, tuple[str, ...]]] = {
    "front": FACE_FRONT_LANDMARK_IDS,
    "left": FACE_LEFT_LANDMARK_IDS,
    "three_quarter": FACE_TQ_LANDMARK_IDS,
}


def default_map_path() -> Path:
    """Path to the checked-in map JSON (package data)."""
    root = resources.files("meshops.proportion")
    return Path(str(root.joinpath(MAP_RESOURCE_NAME)))


def load_face_landmarker_map(path: Path | str | None = None) -> dict[str, Any]:
    """Load and lightly validate the frozen map."""
    p = Path(path) if path is not None else default_map_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("face landmarker map must be an object")
    if data.get("schema_version") != MAP_SCHEMA_VERSION:
        raise ValueError(f"unsupported face landmarker map schema {data.get('schema_version')!r}")
    if int(data.get("mesh_point_count", -1)) != MESH_POINT_COUNT:
        raise ValueError("face landmarker map mesh_point_count must be 478")
    landmarks = data.get("landmarks")
    if not isinstance(landmarks, dict) or not landmarks:
        raise ValueError("face landmarker map landmarks missing")
    return data


def skip_reason_for_face_count(n_faces: int) -> str | None:
    """Return skip reason for detector face count (§1 / B9). None = proceed."""
    if n_faces <= 0:
        return "no_face"
    if n_faces >= 2:
        return "multi_face"
    return None


def _as_xy(pt: Any) -> tuple[float, float] | None:
    if pt is None:
        return None
    x: Any
    y: Any
    if isinstance(pt, Mapping):
        x = pt.get("x", pt.get(0))
        y = pt.get("y", pt.get(1))
    elif isinstance(pt, Sequence) and not isinstance(pt, (str, bytes)):
        if len(pt) < 2:
            return None
        x, y = pt[0], pt[1]
    else:
        x = getattr(pt, "x", None)
        y = getattr(pt, "y", None)
    if x is None or y is None:
        return None
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(xf) or not math.isfinite(yf):
        return None
    return xf, yf


def _index_spec_to_xy(
    pts: Sequence[Any],
    spec: Any,
) -> tuple[float, float] | None:
    """Resolve an index or {mean:[i,j,...]} against a landmark list."""
    if isinstance(spec, bool) or spec is None:
        return None
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        idx = int(spec)
        if idx < 0 or idx >= len(pts):
            return None
        return _as_xy(pts[idx])
    if isinstance(spec, Mapping) and "mean" in spec:
        idxs = spec.get("mean")
        if not isinstance(idxs, Sequence) or isinstance(idxs, (str, bytes)):
            return None
        xs: list[float] = []
        ys: list[float] = []
        for raw in idxs:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                return None
            if idx < 0 or idx >= len(pts):
                return None
            xy = _as_xy(pts[idx])
            if xy is None:
                return None
            xs.append(xy[0])
            ys.append(xy[1])
        if not xs:
            return None
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return None


def resolve_entry_xy(
    pts: Sequence[Any],
    entry: Mapping[str, Any],
    *,
    prefer_iris: bool = True,
) -> tuple[float, float] | None:
    """Resolve primary (or fallback) normalized XY for one map entry."""
    primary = entry.get("primary")
    # Iris indices 468/473 require len==478; else use contour mean fallbacks (B39).
    if prefer_iris and isinstance(primary, int) and primary >= 468 and len(pts) < 478:
        fb = entry.get("fallback")
        return _index_spec_to_xy(pts, fb) if fb is not None else None
    xy = _index_spec_to_xy(pts, primary)
    if xy is not None:
        return xy
    fb = entry.get("fallback")
    if fb is None:
        return None
    return _index_spec_to_xy(pts, fb)


def normalized_to_px(
    x: float,
    y: float,
    width_px: int,
    height_px: int,
) -> tuple[float, float] | None:
    """Convert normalized landmarker XY to assist pixels; clamp; skip non-finite."""
    if width_px < 1 or height_px < 1:
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    x_px = x * float(width_px)
    y_px = y * float(height_px)
    if not math.isfinite(x_px) or not math.isfinite(y_px):
        return None
    x_px = min(max(x_px, 0.0), float(width_px - 1))
    y_px = min(max(y_px, 0.0), float(height_px - 1))
    return x_px, y_px


def map_face_landmarks_px(
    pts: Sequence[Any],
    *,
    view: str,
    width_px: int,
    height_px: int,
    map_doc: Mapping[str, Any] | None = None,
    include_optional: bool = True,
) -> dict[str, list[float]]:
    """Map detector points → {id: [x_px, y_px]} for one view (no Z)."""
    doc = map_doc if map_doc is not None else load_face_landmarker_map()
    landmarks = doc.get("landmarks")
    if not isinstance(landmarks, dict):
        return {}
    allowed = set(_VIEW_IDS.get(view, ()))
    out: dict[str, list[float]] = {}
    for lid, entry in landmarks.items():
        if not isinstance(entry, Mapping):
            continue
        if lid == "cranial_vertex":
            continue
        views = entry.get("views") or []
        if view not in views:
            continue
        optional = bool(entry.get("optional", False))
        if optional and not include_optional:
            continue
        # Required form-read ids must appear in the frozen view set.
        if not optional and allowed and lid not in allowed:
            continue
        xy = resolve_entry_xy(pts, entry)
        if xy is None:
            continue
        px = normalized_to_px(xy[0], xy[1], width_px, height_px)
        if px is None:
            continue
        out[str(lid)] = [float(px[0]), float(px[1])]
    return out


def build_assist_pixel_dump(
    *,
    view: str,
    width_px: int,
    height_px: int,
    landmarks_px: Mapping[str, Sequence[float]],
    pose: str = "unknown",
    facing_direction: str | None = None,
    method: str = "pose_model",
    with_method: bool = True,
    messages: list[str] | None = None,
) -> dict[str, Any]:
    """Build assist_pixel_capture dump (pixels only; no landmarker Z)."""
    facing = facing_direction
    if facing is None:
        facing = {
            "front": "camera_front",
            "left": "camera_left",
            "three_quarter": "camera_three_quarter",
        }.get(view, "unknown")
    lm_out: dict[str, Any] = {}
    for lid, xy in landmarks_px.items():
        if len(xy) < 2:
            continue
        x_px, y_px = float(xy[0]), float(xy[1])
        if with_method:
            lm_out[str(lid)] = {
                "x": x_px,
                "y": y_px,
                "method": method,
            }
        else:
            lm_out[str(lid)] = [x_px, y_px]
    msgs = list(messages or [])
    if LANDMARKER_HONESTY not in msgs:
        msgs.append(LANDMARKER_HONESTY)
    return {
        "schema_version": "1.0.0",
        "kind": "assist_pixel_capture",
        "honesty": "proportion_capture_not_mesh_or_print_success",
        "pose": pose,
        "multi_figure": False,
        "detector": "face_landmarker",
        "views": {
            view: {
                "width_px": int(width_px),
                "height_px": int(height_px),
                "facing_direction": facing,
                "landmarks": lm_out,
            }
        },
        "messages": msgs,
    }


def assert_map_covers_face_front(map_doc: Mapping[str, Any] | None = None) -> None:
    """Raise AssertionError if FACE_FRONT ids lack map entries / valid indices."""
    doc = map_doc if map_doc is not None else load_face_landmarker_map()
    landmarks = doc.get("landmarks")
    assert isinstance(landmarks, dict)
    for lid in FACE_FRONT_LANDMARK_IDS:
        assert lid in landmarks, f"missing map entry for {lid}"
        entry = landmarks[lid]
        assert isinstance(entry, Mapping)
        primary = entry.get("primary")
        idxs = _collect_indices(primary)
        fb = entry.get("fallback")
        if fb is not None:
            idxs.extend(_collect_indices(fb))
        assert idxs, f"no indices for {lid}"
        for i in idxs:
            assert 0 <= i <= 477, f"index {i} out of range for {lid}"
        assert lid in KNOWN_LANDMARK_IDS, f"{lid} not in KNOWN_LANDMARK_IDS"


def _collect_indices(spec: Any) -> list[int]:
    if isinstance(spec, bool) or spec is None:
        return []
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return [int(spec)]
    if isinstance(spec, Mapping) and "mean" in spec:
        raw = spec.get("mean")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            out: list[int] = []
            for v in raw:
                try:
                    out.append(int(v))
                except (TypeError, ValueError):
                    continue
            return out
    return []


def mediapipe_import_forbidden_in_src() -> list[str]:
    """Return src/meshops relative paths with a real mediapipe import (should be empty)."""
    # Match only statement-form imports, not string literals that mention the token.
    pat = re.compile(r"^(?:import\s+mediapipe\b|from\s+mediapipe\b)")
    root = Path(__file__).resolve().parents[1]  # meshops/
    hits: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pat.match(stripped):
                hits.append(str(path.relative_to(root.parent)))
                break
    return hits


__all__ = [
    "LANDMARKER_SKIP_REASONS",
    "MAP_RESOURCE_NAME",
    "MAP_SCHEMA_VERSION",
    "MESH_POINT_COUNT",
    "assert_map_covers_face_front",
    "build_assist_pixel_dump",
    "default_map_path",
    "load_face_landmarker_map",
    "map_face_landmarks_px",
    "mediapipe_import_forbidden_in_src",
    "normalized_to_px",
    "resolve_entry_xy",
    "skip_reason_for_face_count",
]
