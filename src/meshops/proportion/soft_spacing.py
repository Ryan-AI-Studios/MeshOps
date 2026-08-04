"""Soft-tissue spacing metrics from top-view assist + fused meters (0030).

Top plan axes (B1 — matches F3D top view_up=(0,1,0)):
  image-right → body +X (camera-right)
  image-up (smaller y_px) → body +Y (back)
  image-down → body -Y (face)

Scale (never height_m / top.figure_span_px — top span is body-depth pixels, AI fold B2):
  1) fused landmarks_xyz meters when endpoints have finite x_m (y_m when needed)
  2) (a) bust/shoulder width_m / top lateral pixel span
  3) (b) height_m / front.figure_span_px + note ``soft_spacing scale: stature mpp fallback``
  4) fail → note ``soft_spacing scale: unresolved``

Authoring aids only (Difficulty §12 / N6) — not clinical biometrics / bra sizing.
"""

from __future__ import annotations

import math
from typing import Final

from meshops.proportion.frame import figure_span_from_landmarks
from meshops.proportion.models import (
    BreastMetrics,
    BreastSideMetrics,
    DepthBand,
    DiameterMeasure,
    Landmark2D,
    LandmarkXYZ,
    SoftSpacing,
    ViewLandmarks,
)

# Exact note / message literals (plan §7 / C6) — tests match verbatim.
MSG_TOP_ABSENT: Final[str] = "top view absent — breast separation / glute cleft weak"
NOTE_STATURE_MPP: Final[str] = "soft_spacing scale: stature mpp fallback"
NOTE_SCALE_UNRESOLVED: Final[str] = "soft_spacing scale: unresolved"
NOTE_BUST_CENTER_UNRESOLVED: Final[str] = (
    "soft_spacing: bust edges present — center span unresolved"
)
NOTE_PEAKS_ONLY: Final[str] = "peaks only — cleft edges missing"
NOTE_NO_CONTRA_BREAST: Final[str] = "no contralateral breast — gap null"
NOTE_NO_CONTRA_GLUTE: Final[str] = "no contralateral glute — span null"
NOTE_VOLUME_PROXY: Final[str] = (
    "volume_proxy is full-ellipsoid authoring aid, not scanned breast volume"
)
NOTE_CIRC_PROXY: Final[str] = (
    "circumference_proxy is rough ellipse perimeter, not clinical tape measure"
)
NOTE_RX_ASYMMETRY: Final[str] = "breast rx asymmetry > 15%"
NOTE_HANG_DEFERRED: Final[str] = "joint soft_spacing: left-view hang_tilt deferred to 0027"

_SOURCE_VIEW_ORDER: Final[tuple[str, ...]] = (
    "front",
    "left",
    "three_quarter",
    "back",
    "top",
)

_SPHERE_RATIO: Final[float] = 1.15
_TEARDROP_RZ_OVER_RY: Final[float] = 1.15
_RX_ASYMMETRY_FRAC: Final[float] = 0.15

_BREAST_LOWER_IDS: Final[frozenset[str]] = frozenset(
    {"breast_lower", "breast_lower_l", "breast_lower_r"}
)


def _finite(v: float | None) -> bool:
    return v is not None and math.isfinite(float(v))


def _f(v: float | None) -> float | None:
    if v is None:
        return None
    fv = float(v)
    return fv if math.isfinite(fv) else None


def _abs_diff(a: float | None, b: float | None) -> float | None:
    if not _finite(a) or not _finite(b):
        return None
    return abs(float(a) - float(b))  # type: ignore[arg-type]


def _lm2(
    views: dict[str, ViewLandmarks],
    view: str,
    lid: str,
) -> Landmark2D | None:
    vl = views.get(view)
    if vl is None:
        return None
    return vl.landmarks.get(lid)


def _xyz(landmarks_xyz: dict[str, LandmarkXYZ], lid: str) -> LandmarkXYZ | None:
    return landmarks_xyz.get(lid)


def _x_m_from_xyz(landmarks_xyz: dict[str, LandmarkXYZ], lid: str) -> float | None:
    item = _xyz(landmarks_xyz, lid)
    if item is None:
        return None
    return _f(item.x_m)


def _y_m_from_xyz(landmarks_xyz: dict[str, LandmarkXYZ], lid: str) -> float | None:
    item = _xyz(landmarks_xyz, lid)
    if item is None:
        return None
    return _f(item.y_m)


def _z_m_from_xyz(landmarks_xyz: dict[str, LandmarkXYZ], lid: str) -> float | None:
    item = _xyz(landmarks_xyz, lid)
    if item is None:
        return None
    return _f(item.z_m)


def _record_source(contrib: set[str], views: dict[str, ViewLandmarks], view: str, lid: str) -> None:
    if _lm2(views, view, lid) is not None:
        contrib.add(view)


def _ordered_sources(contrib: set[str]) -> list[str]:
    return [k for k in _SOURCE_VIEW_ORDER if k in contrib]


def _diameter_width_m(
    diameters: list[DiameterMeasure] | None,
    band_id: str,
) -> float | None:
    if not diameters:
        return None
    for d in diameters:
        if d.band_id != band_id:
            continue
        if _finite(d.width_m):
            return float(d.width_m)  # type: ignore[arg-type]
        if _finite(d.half_width_m):
            return float(d.half_width_m) * 2.0  # type: ignore[arg-type]
    return None


def _depth_band(
    depth_bands: list[DepthBand] | None,
    band_id: str,
) -> DepthBand | None:
    if not depth_bands:
        return None
    for b in depth_bands:
        if b.band_id == band_id:
            return b
    return None


def _top_pixel_span_x(
    views: dict[str, ViewLandmarks],
    id_a: str,
    id_b: str,
) -> float | None:
    a = _lm2(views, "top", id_a)
    b = _lm2(views, "top", id_b)
    if a is None or b is None:
        return None
    span = abs(float(a.x_px) - float(b.x_px))
    return span if span > 0.0 else None


def _resolve_mpp(
    views: dict[str, ViewLandmarks],
    landmarks_xyz: dict[str, LandmarkXYZ],
    *,
    height_m: float | None,
    diameters: list[DiameterMeasure] | None,
    notes: list[str],
) -> float | None:
    """Meters per top-view pixel. Never uses top.figure_span_px as stature."""
    # Prefer fused meters for known lateral pairs that also exist on top.
    for id_l, id_r in (
        ("bust_l", "bust_r"),
        ("shoulder_l", "shoulder_r"),
        ("breast_lateral_l", "breast_lateral_r"),
        ("breast_medial_l", "breast_medial_r"),
        ("breast_center_l", "breast_center_r"),
        ("hip_l", "hip_r"),
    ):
        xm_l = _x_m_from_xyz(landmarks_xyz, id_l)
        xm_r = _x_m_from_xyz(landmarks_xyz, id_r)
        px = _top_pixel_span_x(views, id_l, id_r)
        if _finite(xm_l) and _finite(xm_r) and px is not None and px > 0:
            meters = abs(float(xm_r) - float(xm_l))  # type: ignore[arg-type]
            if meters > 0:
                return meters / px

    # (a) diameter width meters / top lateral pixel span of same band marks
    for band, id_l, id_r in (
        ("bust", "bust_l", "bust_r"),
        ("shoulder", "shoulder_l", "shoulder_r"),
    ):
        width_m = _diameter_width_m(diameters, band)
        if width_m is None and band == "shoulder":
            # shoulder may only exist as landmarks; try half-widths sum from paired arms N/A
            width_m = _diameter_width_m(diameters, "bust")
            id_l, id_r = "bust_l", "bust_r"
        px = _top_pixel_span_x(views, id_l, id_r)
        if width_m is not None and width_m > 0 and px is not None and px > 0:
            return float(width_m) / px

    # Also allow breast lateral pair with bust diameter when present
    bust_w = _diameter_width_m(diameters, "bust")
    lat_px = _top_pixel_span_x(views, "breast_lateral_l", "breast_lateral_r")
    if bust_w is not None and bust_w > 0 and lat_px is not None and lat_px > 0:
        return float(bust_w) / lat_px

    # (b) stature / front figure span — never top.figure_span_px
    if _finite(height_m) and float(height_m) > 0:  # type: ignore[arg-type]
        front = views.get("front")
        if front is not None:
            span = front.figure_span_px
            if span is None:
                span = figure_span_from_landmarks(front)
            if span is not None and float(span) > 0:
                notes.append(NOTE_STATURE_MPP)
                return float(height_m) / float(span)  # type: ignore[arg-type]

    notes.append(NOTE_SCALE_UNRESOLVED)
    return None


def _plan_xy_m(
    lm: Landmark2D,
    *,
    mpp: float,
    ref_x_px: float,
    ref_y_px: float,
) -> tuple[float, float]:
    """Top pixel → body plan meters (B1)."""
    x_m = (float(lm.x_px) - ref_x_px) * mpp
    # smaller y_px → larger +Y (back)
    y_m = (ref_y_px - float(lm.y_px)) * mpp
    return x_m, y_m


def _endpoint_x_m(
    lid: str,
    *,
    views: dict[str, ViewLandmarks],
    landmarks_xyz: dict[str, LandmarkXYZ],
    mpp: float | None,
    ref_x_px: float,
    ref_y_px: float,
) -> float | None:
    """Prefer fused x_m; else top-view pixel * mpp."""
    xm = _x_m_from_xyz(landmarks_xyz, lid)
    if _finite(xm):
        return float(xm)  # type: ignore[arg-type]
    if mpp is None:
        return None
    top_lm = _lm2(views, "top", lid)
    if top_lm is None:
        return None
    x_m, _y = _plan_xy_m(top_lm, mpp=mpp, ref_x_px=ref_x_px, ref_y_px=ref_y_px)
    return x_m


def _endpoint_y_m(
    lid: str,
    *,
    views: dict[str, ViewLandmarks],
    landmarks_xyz: dict[str, LandmarkXYZ],
    mpp: float | None,
    ref_x_px: float,
    ref_y_px: float,
) -> float | None:
    ym = _y_m_from_xyz(landmarks_xyz, lid)
    if _finite(ym):
        return float(ym)  # type: ignore[arg-type]
    if mpp is None:
        return None
    top_lm = _lm2(views, "top", lid)
    if top_lm is None:
        return None
    _x, y_m = _plan_xy_m(top_lm, mpp=mpp, ref_x_px=ref_x_px, ref_y_px=ref_y_px)
    return y_m


def _pair_span_m(
    id_l: str,
    id_r: str,
    *,
    views: dict[str, ViewLandmarks],
    landmarks_xyz: dict[str, LandmarkXYZ],
    mpp: float | None,
    ref_x_px: float,
    ref_y_px: float,
    contrib: set[str],
) -> float | None:
    xl = _endpoint_x_m(
        id_l,
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
    )
    xr = _endpoint_x_m(
        id_r,
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
    )
    span = _abs_diff(xl, xr)
    if span is None:
        return None
    for view in _SOURCE_VIEW_ORDER:
        _record_source(contrib, views, view, id_l)
        _record_source(contrib, views, view, id_r)
    # fused without view mark still counts as sources from landmarks_xyz
    for lid in (id_l, id_r):
        item = _xyz(landmarks_xyz, lid)
        if item is not None:
            for s in item.sources:
                if s in _SOURCE_VIEW_ORDER:
                    contrib.add(s)
    return span


def _has_any_breast_side_mark(views: dict[str, ViewLandmarks], side: str) -> bool:
    ids = (
        f"breast_center_{side}",
        f"breast_medial_{side}",
        f"breast_lateral_{side}",
        f"breast_lower_{side}",
    )
    for vl in views.values():
        for lid in ids:
            if lid in vl.landmarks:
                return True
        if side == "l" and "breast_lower" in vl.landmarks:
            return True
    return False


def _has_any_glute_side_mark(views: dict[str, ViewLandmarks], side: str) -> bool:
    ids = (f"glute_peak_{side}", f"glute_medial_{side}")
    for vl in views.values():
        for lid in ids:
            if lid in vl.landmarks:
                return True
    return False


def _breast_lower_present(views: dict[str, ViewLandmarks], side: str) -> bool:
    for vl in views.values():
        if f"breast_lower_{side}" in vl.landmarks:
            return True
        if "breast_lower" in vl.landmarks:
            return True
    return False


def _rx_for_side(
    side: str,
    *,
    views: dict[str, ViewLandmarks],
    landmarks_xyz: dict[str, LandmarkXYZ],
    mpp: float | None,
    ref_x_px: float,
    ref_y_px: float,
    contrib: set[str],
) -> float | None:
    medial = f"breast_medial_{side}"
    lateral = f"breast_lateral_{side}"
    center = f"breast_center_{side}"
    # half |lateral - medial|
    span = _pair_span_m(
        medial,
        lateral,
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
        contrib=contrib,
    )
    if span is not None:
        return span / 2.0
    # half center → lateral
    span2 = _pair_span_m(
        center,
        lateral,
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
        contrib=contrib,
    )
    if span2 is not None:
        return span2 / 2.0
    return None


def _ry_for_side(
    *,
    views: dict[str, ViewLandmarks],
    landmarks_xyz: dict[str, LandmarkXYZ],
    mpp: float | None,
    ref_x_px: float,
    ref_y_px: float,
    height_m: float | None,
    depth_bands: list[DepthBand] | None,
    contrib: set[str],
) -> float | None:
    # Prefer half top plan depth when anterior/posterior marks on top exist.
    # Use breast_front / breast_back fused y when present (left-view depth).
    yf = _y_m_from_xyz(landmarks_xyz, "breast_front")
    yb = _y_m_from_xyz(landmarks_xyz, "breast_back")
    if _finite(yf) and _finite(yb):
        contrib.add("left")
        return abs(float(yf) - float(yb)) / 2.0  # type: ignore[arg-type]

    # Top: half Y span of any breast medial/lateral/center pair as weak plan depth
    if mpp is not None:
        top = views.get("top")
        if top is not None:
            ys: list[float] = []
            for lid in (
                "breast_medial_l",
                "breast_medial_r",
                "breast_lateral_l",
                "breast_lateral_r",
                "breast_center_l",
                "breast_center_r",
            ):
                lm = top.landmarks.get(lid)
                if lm is None:
                    continue
                _x, y_m = _plan_xy_m(lm, mpp=mpp, ref_x_px=ref_x_px, ref_y_px=ref_y_px)
                ys.append(y_m)
            if len(ys) >= 2:
                span_y = max(ys) - min(ys)
                if span_y > 1e-9:
                    contrib.add("top")
                    return span_y / 2.0

    band = _depth_band(depth_bands, "breast") or _depth_band(depth_bands, "chest")
    if band is not None:
        if _finite(band.depth_m):
            contrib.add("left")
            return float(band.depth_m) / 2.0  # type: ignore[arg-type]
        if _finite(band.depth_frac) and _finite(height_m):
            contrib.add("left")
            return float(band.depth_frac) * float(height_m) / 2.0  # type: ignore[arg-type]
    return None


def _rz_for_side(
    side: str,
    *,
    views: dict[str, ViewLandmarks],
    landmarks_xyz: dict[str, LandmarkXYZ],
    contrib: set[str],
) -> float | None:
    """Half |upper - lower*| vertical extent (breast_lower* partial promote - rz only)."""
    upper_ids = ("breast_upper",)
    lower_ids = (
        f"breast_lower_{side}",
        "breast_lower",
        f"breast_lower_{'r' if side == 'l' else 'l'}",
    )
    # Prefer same-side lower, then shared breast_lower
    zu = None
    for uid in upper_ids:
        zu = _z_m_from_xyz(landmarks_xyz, uid)
        if _finite(zu):
            item = _xyz(landmarks_xyz, uid)
            if item is not None:
                for s in item.sources:
                    if s in _SOURCE_VIEW_ORDER:
                        contrib.add(s)
            break
        for view in _SOURCE_VIEW_ORDER:
            if _lm2(views, view, uid) is not None:
                # no meters without fuse z
                break

    zl = None
    for lid in lower_ids:
        zl = _z_m_from_xyz(landmarks_xyz, lid)
        if _finite(zl):
            item = _xyz(landmarks_xyz, lid)
            if item is not None:
                for s in item.sources:
                    if s in _SOURCE_VIEW_ORDER:
                        contrib.add(s)
            break

    span = _abs_diff(zu, zl)
    if span is not None:
        return span / 2.0
    return None


def _slant_deg(
    side: str,
    *,
    views: dict[str, ViewLandmarks],
    landmarks_xyz: dict[str, LandmarkXYZ],
    mpp: float | None,
    ref_x_px: float,
    ref_y_px: float,
    contrib: set[str],
) -> float | None:
    """atan2(dy, dx) lateral-medial body XY; 0=+X; + = lateral more +Y/back; (-180,180]."""
    medial = f"breast_medial_{side}"
    lateral = f"breast_lateral_{side}"
    xm = _endpoint_x_m(
        medial,
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
    )
    xl = _endpoint_x_m(
        lateral,
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
    )
    ym = _endpoint_y_m(
        medial,
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
    )
    yl = _endpoint_y_m(
        lateral,
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
    )
    if not all(_finite(v) for v in (xm, xl, ym, yl)):
        return None
    dx = float(xl) - float(xm)  # type: ignore[arg-type]
    dy = float(yl) - float(ym)  # type: ignore[arg-type]
    if dx == 0.0 and dy == 0.0:
        return None
    ang = math.degrees(math.atan2(dy, dx))
    # range (-180, 180]
    if ang <= -180.0:
        ang = 180.0
    for lid in (medial, lateral):
        for view in _SOURCE_VIEW_ORDER:
            _record_source(contrib, views, view, lid)
    return ang


def _classify_shape(
    rx: float | None,
    ry: float | None,
    rz: float | None,
    *,
    breast_lower_present: bool,
) -> str | None:
    if (
        breast_lower_present
        and _finite(rz)
        and _finite(ry)
        and float(rz) > float(ry) * _TEARDROP_RZ_OVER_RY  # type: ignore[arg-type]
    ):
        return "teardrop_proxy"
    finite_vals = [float(v) for v in (rx, ry, rz) if _finite(v) and float(v) > 0]  # type: ignore[arg-type]
    if len(finite_vals) < 2:
        return None
    mx = max(finite_vals)
    mn = min(finite_vals)
    if mn > 0 and mx / mn <= _SPHERE_RATIO:
        return "sphere"
    if _finite(rz):
        rz_f = float(rz)  # type: ignore[arg-type]
        if all(rz_f >= v for v in finite_vals):
            return "prolate"
        if all(rz_f <= v for v in finite_vals):
            return "oblate"
    return None


def _side_metrics(
    side: str,
    *,
    views: dict[str, ViewLandmarks],
    landmarks_xyz: dict[str, LandmarkXYZ],
    mpp: float | None,
    ref_x_px: float,
    ref_y_px: float,
    height_m: float | None,
    depth_bands: list[DepthBand] | None,
    notes: list[str],
    contrib: set[str],
) -> BreastSideMetrics | None:
    if not _has_any_breast_side_mark(views, side) and not any(
        f"breast_{k}_{side}" in landmarks_xyz for k in ("center", "medial", "lateral", "lower")
    ):
        # also allow shared breast_front/back only if side marks exist elsewhere
        return None

    rx = _rx_for_side(
        side,
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
        contrib=contrib,
    )
    ry = _ry_for_side(
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
        height_m=height_m,
        depth_bands=depth_bands,
        contrib=contrib,
    )
    rz = _rz_for_side(
        side,
        views=views,
        landmarks_xyz=landmarks_xyz,
        contrib=contrib,
    )
    slant = _slant_deg(
        side,
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
        contrib=contrib,
    )
    lower_ok = _breast_lower_present(views, side)
    shape = _classify_shape(rx, ry, rz, breast_lower_present=lower_ok)

    volume: float | None = None
    circ: float | None = None
    if _finite(rx) and _finite(ry) and _finite(rz):
        volume = (4.0 / 3.0) * math.pi * float(rx) * float(ry) * float(rz)  # type: ignore[arg-type]
        if NOTE_VOLUME_PROXY not in notes:
            notes.append(NOTE_VOLUME_PROXY)
    if _finite(rx) and _finite(ry):
        circ = math.pi * (float(rx) + float(ry))  # type: ignore[arg-type]
        if NOTE_CIRC_PROXY not in notes:
            notes.append(NOTE_CIRC_PROXY)

    # hang_tilt always null in 0030 (C5)
    if all(v is None for v in (rx, ry, rz, slant, volume, circ, shape)):
        return None
    return BreastSideMetrics(
        circumference_proxy_m=circ,
        volume_proxy_m3=volume,
        shape=shape,  # type: ignore[arg-type]
        slant_deg=slant,
        hang_tilt_deg=None,
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
    )


def compute_soft_spacing(
    views: dict[str, ViewLandmarks],
    landmarks_xyz: dict[str, LandmarkXYZ],
    *,
    height_m: float | None = None,
    diameters: list[DiameterMeasure] | None = None,
    depth_bands: list[DepthBand] | None = None,
) -> tuple[SoftSpacing | None, BreastMetrics | None, list[str]]:
    """Compute soft_spacing + breast_metrics for ProportionReport.

    Returns (soft_spacing, breast_metrics, extra_messages).
    extra_messages include B3 when top is absent.
    """
    extra_messages: list[str] = []
    notes: list[str] = []
    contrib: set[str] = set()

    has_top = "top" in views
    if not has_top:
        extra_messages.append(MSG_TOP_ABSENT)

    # Reference origin for top plan: image center of top view when present
    ref_x_px = 0.0
    ref_y_px = 0.0
    top = views.get("top")
    if top is not None:
        ref_x_px = float(top.width_px) / 2.0
        ref_y_px = float(top.height_px) / 2.0

    mpp = _resolve_mpp(
        views,
        landmarks_xyz,
        height_m=height_m,
        diameters=diameters,
        notes=notes,
    )

    # --- breast center span (centers only — AI fold B3) ---
    breast_center_span_m = _pair_span_m(
        "breast_center_l",
        "breast_center_r",
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
        contrib=contrib,
    )
    bust_l = _lm2(views, "top", "bust_l") or _xyz(landmarks_xyz, "bust_l")
    bust_r = _lm2(views, "top", "bust_r") or _xyz(landmarks_xyz, "bust_r")
    if breast_center_span_m is None and bust_l is not None and bust_r is not None:
        notes.append(NOTE_BUST_CENTER_UNRESOLVED)

    # --- intermammary gap ---
    intermammary_gap_m: float | None = None
    has_med_l = (
        _endpoint_x_m(
            "breast_medial_l",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
        )
        is not None
    )
    has_med_r = (
        _endpoint_x_m(
            "breast_medial_r",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
        )
        is not None
    )
    has_ctr_l = (
        _endpoint_x_m(
            "breast_center_l",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
        )
        is not None
    )
    has_ctr_r = (
        _endpoint_x_m(
            "breast_center_r",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
        )
        is not None
    )

    breast_side_l = _has_any_breast_side_mark(views, "l") or has_med_l or has_ctr_l
    breast_side_r = _has_any_breast_side_mark(views, "r") or has_med_r or has_ctr_r

    if has_med_l and has_med_r:
        intermammary_gap_m = _pair_span_m(
            "breast_medial_l",
            "breast_medial_r",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
            contrib=contrib,
        )
    elif has_ctr_l and has_ctr_r:
        # max(0, span - rx_l - rx_r) when radii known
        span = breast_center_span_m
        if span is None:
            span = _pair_span_m(
                "breast_center_l",
                "breast_center_r",
                views=views,
                landmarks_xyz=landmarks_xyz,
                mpp=mpp,
                ref_x_px=ref_x_px,
                ref_y_px=ref_y_px,
                contrib=contrib,
            )
        rx_l = _rx_for_side(
            "l",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
            contrib=contrib,
        )
        rx_r = _rx_for_side(
            "r",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
            contrib=contrib,
        )
        if span is not None and _finite(rx_l) and _finite(rx_r):
            intermammary_gap_m = max(0.0, float(span) - float(rx_l) - float(rx_r))  # type: ignore[arg-type]
    elif breast_side_l ^ breast_side_r:
        # B4 single side
        notes.append(NOTE_NO_CONTRA_BREAST)

    # --- glute ---
    glute_cleft_gap_m: float | None = None
    glute_peak_span_m: float | None = None

    has_gm_l = (
        _endpoint_x_m(
            "glute_medial_l",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
        )
        is not None
    )
    has_gm_r = (
        _endpoint_x_m(
            "glute_medial_r",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
        )
        is not None
    )
    has_gp_l = (
        _endpoint_x_m(
            "glute_peak_l",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
        )
        is not None
    )
    has_gp_r = (
        _endpoint_x_m(
            "glute_peak_r",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
        )
        is not None
    )

    if has_gm_l and has_gm_r:
        glute_cleft_gap_m = _pair_span_m(
            "glute_medial_l",
            "glute_medial_r",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
            contrib=contrib,
        )
    if has_gp_l and has_gp_r:
        glute_peak_span_m = _pair_span_m(
            "glute_peak_l",
            "glute_peak_r",
            views=views,
            landmarks_xyz=landmarks_xyz,
            mpp=mpp,
            ref_x_px=ref_x_px,
            ref_y_px=ref_y_px,
            contrib=contrib,
        )
        if glute_cleft_gap_m is None and not (has_gm_l and has_gm_r):
            notes.append(NOTE_PEAKS_ONLY)
    elif (
        (has_gp_l ^ has_gp_r)
        or (has_gm_l ^ has_gm_r)
        or (
            (_has_any_glute_side_mark(views, "l") ^ _has_any_glute_side_mark(views, "r"))
            and not (has_gm_l and has_gm_r)
            and not (has_gp_l and has_gp_r)
        )
    ):
        notes.append(NOTE_NO_CONTRA_GLUTE)

    # glute_cleft single id: midline sanity only (AI fold B9)
    cleft_lm = _lm2(views, "top", "glute_cleft")
    if cleft_lm is not None and mpp is not None:
        cx, _cy = _plan_xy_m(cleft_lm, mpp=mpp, ref_x_px=ref_x_px, ref_y_px=ref_y_px)
        # large |x| relative to typical half-hip (~0.15 m) → optional note
        if abs(cx) > 0.08:
            notes.append("glute_cleft mark off-midline — sanity check only")
        contrib.add("top")

    # --- breast metrics ---
    breast_notes: list[str] = []
    left_m = _side_metrics(
        "l",
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
        height_m=height_m,
        depth_bands=depth_bands,
        notes=breast_notes,
        contrib=contrib,
    )
    right_m = _side_metrics(
        "r",
        views=views,
        landmarks_xyz=landmarks_xyz,
        mpp=mpp,
        ref_x_px=ref_x_px,
        ref_y_px=ref_y_px,
        height_m=height_m,
        depth_bands=depth_bands,
        notes=breast_notes,
        contrib=contrib,
    )
    # Merge side honesty notes into soft notes (volume/circ once)
    for n in breast_notes:
        if n not in notes:
            notes.append(n)

    symmetry_notes: list[str] = []
    if (
        left_m is not None
        and right_m is not None
        and _finite(left_m.rx_m)
        and _finite(right_m.rx_m)
    ):
        rxl = float(left_m.rx_m)  # type: ignore[arg-type]
        rxr = float(right_m.rx_m)  # type: ignore[arg-type]
        denom = max(rxl, rxr)
        if denom > 0 and abs(rxl - rxr) / denom > _RX_ASYMMETRY_FRAC:
            symmetry_notes.append(NOTE_RX_ASYMMETRY)

    # hang_tilt always null — optional joint note once when breast sides exist
    if (left_m is not None or right_m is not None) and NOTE_HANG_DEFERRED not in notes:
        notes.append(NOTE_HANG_DEFERRED)

    soft = SoftSpacing(
        intermammary_gap_m=intermammary_gap_m,
        breast_center_span_m=breast_center_span_m,
        glute_cleft_gap_m=glute_cleft_gap_m,
        glute_peak_span_m=glute_peak_span_m,
        source_views=_ordered_sources(contrib),
        notes=notes,
    )
    # Emit soft_spacing even when all null if we have notes / top absent context
    breast_metrics: BreastMetrics | None = None
    if left_m is not None or right_m is not None:
        breast_metrics = BreastMetrics(
            left=left_m,
            right=right_m,
            symmetry_notes=symmetry_notes,
        )

    # If completely empty (no marks, no notes beyond scale unresolved alone without any soft marks)
    any_soft_value = any(
        _finite(v)
        for v in (
            intermammary_gap_m,
            breast_center_span_m,
            glute_cleft_gap_m,
            glute_peak_span_m,
        )
    )
    if not any_soft_value and breast_metrics is None and not notes and not extra_messages:
        return None, None, extra_messages

    return soft, breast_metrics, extra_messages
