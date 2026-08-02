"""Cross-view fusion → blockout-grade landmarks_xyz (frozen axis signs).

Axes (freeze):
  Z up, soles = 0
  +X = camera-right in front view
  +Y toward camera when facing_direction == camera_left (invert for camera_right / right)
"""

from __future__ import annotations

from meshops.proportion.frame import figure_span_from_landmarks
from meshops.proportion.models import (
    Landmark2D,
    LandmarkXYZ,
    QualityFlags,
    ViewLandmarks,
)


def _stature_top(lm: dict[str, Landmark2D]) -> tuple[Landmark2D | None, bool]:
    """Return (top landmark, used_hair_fallback)."""
    if "cranial_vertex" in lm:
        return lm["cranial_vertex"], False
    if "hair_crown" in lm:
        return lm["hair_crown"], True
    return None, False


def head_unit_frac_from_front(front: ViewLandmarks) -> tuple[float | None, bool, list[str]]:
    """Cranial-preferred head height as fraction of figure stature.

    Returns (hu_frac, hair_volume_margin, messages).
    """
    msgs: list[str] = []
    lm = front.landmarks
    chin = lm.get("chin")
    sole = lm.get("sole")
    top, hair = _stature_top(lm)
    if top is None or chin is None or sole is None:
        msgs.append("incomplete stature for head unit (need top+chin+sole)")
        return None, hair, msgs
    figure_h = sole.y_px - top.y_px
    head_h = chin.y_px - top.y_px
    if figure_h <= 0 or head_h <= 0:
        msgs.append("non-positive head or figure height")
        return None, hair, msgs
    return head_h / figure_h, hair, msgs


def vertical_span_discrepancy(
    front: ViewLandmarks | None,
    left: ViewLandmarks | None,
) -> float | None:
    """|span_f - span_l| / max(span_f, span_l); None if either span unknown."""
    if front is None or left is None:
        return None
    sf = front.figure_span_px or figure_span_from_landmarks(front)
    sl = left.figure_span_px or figure_span_from_landmarks(left)
    if sf is None or sl is None or max(sf, sl) <= 0:
        return None
    return abs(sf - sl) / max(sf, sl)


def _depth_sign(facing: str | None) -> float:
    """+1 when +Y toward camera for camera_left (MeshOps left default)."""
    if facing is None or facing in ("camera_left", "unknown", ""):
        return 1.0
    if facing in ("camera_right", "right"):
        return -1.0
    return 1.0


def fuse_xyz(
    views: dict[str, ViewLandmarks],
    *,
    height_m: float | None = None,
    foreshortening_risk: bool = False,
) -> tuple[dict[str, LandmarkXYZ], QualityFlags, list[str]]:
    """Fuse per-view 2D landmarks into figure-normalized XYZ."""
    messages: list[str] = []
    quality = QualityFlags()
    front = views.get("front")
    left = views.get("left")

    if front is None:
        quality.incomplete_stature = True
        messages.append("no front view — cannot fuse stature XYZ")
        return {}, quality, messages

    lm = front.landmarks
    top, hair = _stature_top(lm)
    sole = lm.get("sole")
    if hair:
        quality.hair_volume_margin = True
        messages.append("head unit uses hair_crown→chin (hair_volume_margin)")

    if top is None or sole is None:
        quality.incomplete_stature = True
        messages.append("incomplete stature (need cranial_vertex|hair_crown + sole on front)")
        # Still emit partial X if midline known — but Z unknown
        return {}, quality, messages

    figure_h = sole.y_px - top.y_px
    if figure_h <= 0:
        quality.incomplete_stature = True
        messages.append("non-positive figure height on front")
        return {}, quality, messages

    # Cache span on front
    if front.figure_span_px is None:
        front.figure_span_px = figure_h

    midline_x = _midline_x(front)

    conf_scale = 0.7 if foreshortening_risk else 1.0
    if foreshortening_risk:
        quality.foreshortening_risk = True

    out: dict[str, LandmarkXYZ] = {}

    # All front landmarks → X/Z
    for lid, L in lm.items():
        if lid == "midline_x":
            continue
        z = (sole.y_px - L.y_px) / figure_h
        x = (L.x_px - midline_x) / figure_h if midline_x is not None else None
        conf = min(1.0, L.confidence * conf_scale)
        xyz = LandmarkXYZ(
            id=lid,
            x=x,
            y=None,
            z=z,
            confidence=conf,
            sources=["front"],
        )
        if height_m is not None:
            if x is not None:
                xyz.x_m = x * height_m
            xyz.z_m = z * height_m
        out[lid] = xyz

    # Depth from left view
    if left is not None and left.landmarks:
        left_span = left.figure_span_px or figure_span_from_landmarks(left)
        if left_span is None or left_span <= 0:
            # fallback: use front figure_h (imperfect)
            left_span = figure_h
            messages.append("left figure span unknown; using front span for depth scale")
        else:
            if left.figure_span_px is None:
                left.figure_span_px = left_span

        facing = left.facing_direction or "camera_left"
        sign = _depth_sign(str(facing) if facing else "camera_left")
        torso_cx = _torso_center_x(left)

        depth_pairs = (
            ("chest_front", "chest_back", "chest_mid"),
            ("hip_front", "hip_back", "hip_mid"),
        )
        for front_id, back_id, mid_id in depth_pairs:
            lf = left.landmarks.get(front_id)
            lb = left.landmarks.get(back_id)
            if lf is None and lb is None:
                continue
            for src_lm in (lf, lb):
                if src_lm is None:
                    continue
                y_body = sign * (src_lm.x_px - torso_cx) / left_span
                conf = min(1.0, src_lm.confidence * conf_scale * 0.9)
                existing = out.get(src_lm.id)
                if existing is None:
                    z_left = _z_from_view(left, src_lm, left_span)
                    item = LandmarkXYZ(
                        id=src_lm.id,
                        x=None,
                        y=y_body,
                        z=z_left,
                        confidence=conf,
                        sources=["left"],
                    )
                    if height_m is not None:
                        item.y_m = y_body * height_m
                        if z_left is not None:
                            item.z_m = z_left * height_m
                    out[src_lm.id] = item
                else:
                    existing.y = y_body
                    if height_m is not None:
                        existing.y_m = y_body * height_m
                    if "left" not in existing.sources:
                        existing.sources.append("left")
                    existing.confidence = min(existing.confidence, conf)

            if lf is not None and lb is not None:
                y_mid = sign * ((lf.x_px + lb.x_px) / 2.0 - torso_cx) / left_span
                z_ref: float | None = None
                x_ref: float | None = None
                if mid_id == "chest_mid":
                    for ref in ("nipple_bust", "belt_hip", "shoulder"):
                        if ref in out:
                            z_ref = out[ref].z
                            x_ref = out[ref].x
                            break
                else:
                    for ref in ("crotch_pubic", "belt_hip", "hip_l"):
                        if ref in out:
                            z_ref = out[ref].z
                            x_ref = out[ref].x
                            break
                mid = LandmarkXYZ(
                    id=mid_id,
                    x=x_ref,
                    y=y_mid,
                    z=z_ref,
                    confidence=0.6 * conf_scale,
                    sources=["left"],
                )
                if height_m is not None:
                    mid.y_m = y_mid * height_m
                    if z_ref is not None:
                        mid.z_m = z_ref * height_m
                    if x_ref is not None:
                        mid.x_m = x_ref * height_m
                out[mid_id] = mid

    return out, quality, messages


def _midline_x(front: ViewLandmarks) -> float | None:
    lm = front.landmarks
    if "midline_x" in lm:
        return lm["midline_x"].x_px
    if "midline" in lm:
        return lm["midline"].x_px
    # Estimate from shoulder pair
    sl, sr = lm.get("shoulder_l"), lm.get("shoulder_r")
    if sl is not None and sr is not None:
        return (sl.x_px + sr.x_px) / 2.0
    hl, hr = lm.get("hip_l"), lm.get("hip_r")
    if hl is not None and hr is not None:
        return (hl.x_px + hr.x_px) / 2.0
    if "sole" in lm:
        return lm["sole"].x_px
    return front.width_px / 2.0


def _torso_center_x(left: ViewLandmarks) -> float:
    lm = left.landmarks
    cf, cb = lm.get("chest_front"), lm.get("chest_back")
    if cf is not None and cb is not None:
        return (cf.x_px + cb.x_px) / 2.0
    hf, hb = lm.get("hip_front"), lm.get("hip_back")
    if hf is not None and hb is not None:
        return (hf.x_px + hb.x_px) / 2.0
    if "spine_hint" in lm:
        return lm["spine_hint"].x_px
    return left.width_px / 2.0


def _z_from_view(view: ViewLandmarks, L: Landmark2D, span: float) -> float | None:
    lm = view.landmarks
    sole = lm.get("sole")
    top, _ = _stature_top(lm)
    if sole is None or top is None or span <= 0:
        return None
    return (sole.y_px - L.y_px) / span


def compute_package_score(views: dict[str, ViewLandmarks]) -> tuple[float, dict[str, float]]:
    """Weighted completeness score 0-100 (spec section 3.1 F)."""
    from meshops.proportion.models import (
        REQUIRED_VIEW_KEYS,
        SCORE_DEPTH,
        SCORE_PER_REQUIRED_VIEW,
        SCORE_STATURE,
        SCORE_WIDTH_PAIR,
    )

    breakdown: dict[str, float] = {
        "views": 0.0,
        "stature": 0.0,
        "width_pair": 0.0,
        "depth": 0.0,
    }

    for key in REQUIRED_VIEW_KEYS:
        if key in views:
            breakdown["views"] += SCORE_PER_REQUIRED_VIEW

    front = views.get("front")
    if front is not None:
        lm = front.landmarks
        top, _ = _stature_top(lm)
        if top is not None and "sole" in lm:
            breakdown["stature"] = SCORE_STATURE

        width_ok = (
            ("shoulder_l" in lm and "shoulder_r" in lm)
            or ("hip_l" in lm and "hip_r" in lm)
            or ("bust_l" in lm and "bust_r" in lm)
            or ("waist_l" in lm and "waist_r" in lm)
        )
        if width_ok:
            breakdown["width_pair"] = SCORE_WIDTH_PAIR

    left = views.get("left")
    if left is not None:
        llm = left.landmarks
        depth_ok = ("chest_front" in llm and "chest_back" in llm) or (
            "hip_front" in llm and "hip_back" in llm
        )
        if depth_ok:
            breakdown["depth"] = SCORE_DEPTH

    total = sum(breakdown.values())
    # Clamp float dust
    total = min(100.0, round(total, 4))
    return total, breakdown
