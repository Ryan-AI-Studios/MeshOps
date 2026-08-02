"""Edge-pair diameters: top-level edge_pairs + {band}_edge0/_edge1 (0013).

Width math (R11):
  width_eucl = hypot(dx, dy)
  theta_deg = degrees(atan2(|dy|, |dx|))  # 0 = pure horizontal
  width_ortho = width_eucl * cos(theta) when theta > 15°, else width_eucl
  width_frac = width_ortho / figure_h
"""

from __future__ import annotations

import math
from typing import Any

from meshops.proportion.assist import point_to_landmark2d
from meshops.proportion.frame import figure_span_from_landmarks
from meshops.proportion.models import DiameterMeasure, ViewLandmarks

# Canonical band ids (front primary). Optional ankle_* allowed.
CANONICAL_BAND_IDS: frozenset[str] = frozenset(
    {
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
    }
)

_ORTHO_THETA_DEG = 15.0
_EDGE0_SUFFIX = "_edge0"
_EDGE1_SUFFIX = "_edge1"


def is_edge_landmark_id(landmark_id: str) -> bool:
    """True for {band}_edge0 / {band}_edge1 suffix form."""
    return landmark_id.endswith(_EDGE0_SUFFIX) or landmark_id.endswith(_EDGE1_SUFFIX)


def edge_band_id(landmark_id: str) -> str | None:
    """Return band id if landmark is an edge suffix, else None."""
    if landmark_id.endswith(_EDGE0_SUFFIX):
        return landmark_id[: -len(_EDGE0_SUFFIX)]
    if landmark_id.endswith(_EDGE1_SUFFIX):
        return landmark_id[: -len(_EDGE1_SUFFIX)]
    return None


def ortho_width(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float]:
    """Return (width_ortho_px, width_eucl_px, theta_deg)."""
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    width_eucl = math.hypot(dx, dy)
    theta_deg = math.degrees(math.atan2(abs(dy), abs(dx))) if width_eucl > 0 else 0.0
    if theta_deg > _ORTHO_THETA_DEG:
        width_ortho = width_eucl * math.cos(math.radians(theta_deg))
    else:
        width_ortho = width_eucl
    return width_ortho, width_eucl, theta_deg


def _parse_pair_coords(value: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Accept [[x0,y0],[x1,y1]] or equivalent nested lists."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    p0, p1 = value[0], value[1]
    if not isinstance(p0, (list, tuple)) or len(p0) < 2:
        return None
    if not isinstance(p1, (list, tuple)) or len(p1) < 2:
        return None
    return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))


def _figure_h(view: ViewLandmarks) -> float | None:
    span = view.figure_span_px or figure_span_from_landmarks(view)
    if span is not None and span > 0:
        return float(span)
    lm = view.landmarks
    sole = lm.get("sole")
    top = lm.get("cranial_vertex") or lm.get("hair_crown")
    if sole is None or top is None:
        return None
    fh = sole.y_px - top.y_px
    return float(fh) if fh > 0 else None


def _midline_x(view: ViewLandmarks) -> float | None:
    lm = view.landmarks
    if "midline_x" in lm:
        return lm["midline_x"].x_px
    if "midline" in lm:
        return lm["midline"].x_px
    return None


def _z_frac(view: ViewLandmarks, y_px: float, figure_h: float) -> float | None:
    sole = view.landmarks.get("sole")
    top = view.landmarks.get("cranial_vertex") or view.landmarks.get("hair_crown")
    if sole is None or top is None or figure_h <= 0:
        return None
    return (sole.y_px - y_px) / figure_h


def _inject_edge_landmarks(
    view: ViewLandmarks,
    band_id: str,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    overwrite: bool,
) -> None:
    """Ensure {band}_edge0 / {band}_edge1 exist on the view for overlays + XYZ."""
    e0, e1 = f"{band_id}_edge0", f"{band_id}_edge1"
    for lid, pt in ((e0, p0), (e1, p1)):
        if lid in view.landmarks and not overwrite:
            continue
        view.landmarks[lid] = point_to_landmark2d(
            lid,
            pt[0],
            pt[1],
            width_px=view.width_px,
            height_px=view.height_px,
            method="assist",
        )


def _measure_from_pair(
    band_id: str,
    view_key: str,
    view: ViewLandmarks,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    method: str,
    height_m: float | None,
    source_tag: str,
) -> DiameterMeasure | None:
    figure_h = _figure_h(view)
    if figure_h is None or figure_h <= 0:
        return None
    width_px, width_eucl, theta_deg = ortho_width(p0[0], p0[1], p1[0], p1[1])
    if width_px <= 0:
        return None
    mid_x = (p0[0] + p1[0]) / 2.0
    mid_y = (p0[1] + p1[1]) / 2.0
    width_frac = width_px / figure_h
    half = width_frac / 2.0
    midline = _midline_x(view)
    x_frac = (mid_x - midline) / figure_h if midline is not None else None
    z_frac = _z_frac(view, mid_y, figure_h)
    conf = 1.0
    e0 = view.landmarks.get(f"{band_id}_edge0")
    e1 = view.landmarks.get(f"{band_id}_edge1")
    if e0 is not None and e1 is not None:
        conf = min(e0.confidence, e1.confidence)
    width_m = width_frac * height_m if height_m is not None else None
    half_m = half * height_m if height_m is not None else None
    return DiameterMeasure(
        band_id=band_id,
        view=view_key,
        width_px=width_px,
        width_eucl_px=width_eucl,
        theta_deg=theta_deg,
        width_frac=width_frac,
        width_m=width_m,
        half_width_frac=half,
        half_width_m=half_m,
        mid_x_px=mid_x,
        mid_y_px=mid_y,
        z_frac=z_frac,
        x_frac=x_frac,
        confidence=conf,
        method=method,
        sources=[source_tag],
    )


def discover_suffix_pairs(
    view: ViewLandmarks,
) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Find {band}_edge0 + {band}_edge1 pairs on a view."""
    lm = view.landmarks
    found: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    edge0_ids = [k for k in lm if k.endswith(_EDGE0_SUFFIX)]
    for e0_id in edge0_ids:
        band = e0_id[: -len(_EDGE0_SUFFIX)]
        e1_id = f"{band}{_EDGE1_SUFFIX}"
        if e1_id not in lm:
            continue
        a, b = lm[e0_id], lm[e1_id]
        found[band] = ((a.x_px, a.y_px), (b.x_px, b.y_px))
    return found


def compute_diameters(
    edge_pairs: dict[str, Any] | None,
    views: dict[str, ViewLandmarks],
    *,
    height_m: float | None = None,
) -> tuple[list[DiameterMeasure], list[str]]:
    """Build diameters from structured edge_pairs (wins) + suffix landmarks.

    Mutates views by injecting {band}_edge0/_edge1 when structured pairs present.
    Returns (diameters, messages).
    """
    messages: list[str] = []
    out: list[DiameterMeasure] = []
    # band_id -> (view_key, measure) — structured overwrites suffix for same band+view
    by_key: dict[tuple[str, str], DiameterMeasure] = {}

    # 1) Structured top-level edge_pairs (sibling of views)
    ep = edge_pairs if isinstance(edge_pairs, dict) else {}
    for view_key, bands in ep.items():
        vk = str(view_key)
        view = views.get(vk)
        if view is None:
            messages.append(f"edge_pairs view {vk!r} has no image/view; skipped")
            continue
        if not isinstance(bands, dict):
            messages.append(f"edge_pairs.{vk} is not an object; skipped")
            continue
        for band_id, raw in bands.items():
            bid = str(band_id)
            pair = _parse_pair_coords(raw)
            if pair is None:
                messages.append(f"edge_pairs.{vk}.{bid} invalid pair; skipped")
                continue
            p0, p1 = pair
            # Structured wins: inject/overwrite landmarks for overlays
            _inject_edge_landmarks(view, bid, p0, p1, overwrite=True)
            m = _measure_from_pair(
                bid,
                vk,
                view,
                p0,
                p1,
                method="edge_pairs",
                height_m=height_m,
                source_tag=f"edge_pairs:{vk}",
            )
            if m is not None:
                by_key[(vk, bid)] = m

    # 2) Suffix form — only if structured did not already set that band on the view
    for vk, view in views.items():
        suffix = discover_suffix_pairs(view)
        for bid, (p0, p1) in suffix.items():
            if (vk, bid) in by_key:
                continue  # structured wins
            m = _measure_from_pair(
                bid,
                vk,
                view,
                p0,
                p1,
                method="edge_suffix",
                height_m=height_m,
                source_tag=f"landmarks:{vk}",
            )
            if m is not None:
                by_key[(vk, bid)] = m

    # Stable order: view order front/left/… then band name
    view_order = {k: i for i, k in enumerate(("front", "left", "three_quarter", "back"))}
    ordered = sorted(by_key.items(), key=lambda kv: (view_order.get(kv[0][0], 99), kv[0][1]))
    out = [m for _, m in ordered]
    if out:
        messages.append(f"diameters: {len(out)} band(s) measured")
    return out, messages


def parse_edge_pairs(assist: dict[str, Any] | None) -> dict[str, Any]:
    """Extract top-level edge_pairs from assist document (or empty)."""
    if not assist or not isinstance(assist, dict):
        return {}
    raw = assist.get("edge_pairs")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw
