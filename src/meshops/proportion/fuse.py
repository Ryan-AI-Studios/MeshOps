"""Cross-view fusion → blockout-grade landmarks_xyz (frozen axis signs).

Axes (freeze):
  Z up, soles = 0
  +X = camera-right in front view
  +Y toward camera when facing_direction == camera_left (invert for camera_right / right)

0013: extended depth pairs + DepthBand list + optional CrossSection correlation.
0024: cranial/foot depth pairs; torso_cx skips cranial/foot bands (B4).
package_score depth gate remains chest/hip only (R8).

Consumers of foot/cranial depth_bands must use depth_frac / y_mid, not raw
y_front as "toe Y" — global auto-swap may invert y_front/y_back labels (C4).
breast_lower* is assist vocabulary only until 0027 (no fuse pair here).
"""

from __future__ import annotations

from meshops.proportion.frame import figure_span_from_landmarks
from meshops.proportion.models import (
    CheckResult,
    CrossSection,
    DepthBand,
    DiameterMeasure,
    Landmark2D,
    LandmarkXYZ,
    QualityFlags,
    ViewLandmarks,
)

# Left-view front/back/mid depth pairs (0012 chest/hip + 0013 extras + 0024 cranial/foot).
DEPTH_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("chest_front", "chest_back", "chest_mid"),
    ("hip_front", "hip_back", "hip_mid"),
    ("breast_front", "breast_back", "breast_mid"),
    ("glute_front", "glute_back", "glute_mid"),
    ("thigh_front", "thigh_back", "thigh_mid"),
    ("calf_front", "calf_back", "calf_mid"),
    ("cranial_front", "cranial_back", "cranial_mid"),
    ("foot_front", "foot_back", "foot_mid"),
)

# Bands never used as torso center X fallback (B4 — head/foot are not torso).
_TORSO_CX_SKIP_BANDS: frozenset[str] = frozenset({"cranial", "foot"})

# package_score depth gate (R8) — never include glute/thigh/breast.
_SCORE_DEPTH_BANDS: frozenset[str] = frozenset({"chest", "hip"})

_CROSS_Z_TOL = 0.03


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


def _band_id_from_pair(front_id: str) -> str:
    """chest_front → chest."""
    if front_id.endswith("_front"):
        return front_id[: -len("_front")]
    return front_id


def _mid_z_refs(mid_id: str) -> tuple[str, ...]:
    if mid_id == "chest_mid":
        return ("nipple_bust", "belt_hip", "shoulder")
    if mid_id == "hip_mid":
        return ("crotch_pubic", "belt_hip", "hip_l")
    if mid_id == "breast_mid":
        return ("nipple_bust", "chest_mid")
    if mid_id == "glute_mid":
        return ("crotch_pubic", "hip_mid", "belt_hip")
    if mid_id == "thigh_mid":
        return ("greater_trochanter", "knee", "crotch_pubic")
    if mid_id == "calf_mid":
        return ("knee", "ankle", "sole")
    # C3: empty → pair-mean Z fallback (do not prefer chin as first ref).
    if mid_id == "cranial_mid":
        return ()
    if mid_id == "foot_mid":
        return ("ankle", "sole", "heel")
    return ()


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

    # Depth from left view (extended pair table)
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

        for front_id, back_id, mid_id in DEPTH_PAIRS:
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
                for ref in _mid_z_refs(mid_id):
                    if ref in out:
                        z_ref = out[ref].z
                        x_ref = out[ref].x
                        break
                if z_ref is None:
                    # Fall back to average left-view z of the pair
                    zf = _z_from_view(left, lf, left_span)
                    zb = _z_from_view(left, lb, left_span)
                    if zf is not None and zb is not None:
                        z_ref = (zf + zb) / 2.0
                    elif zf is not None:
                        z_ref = zf
                    else:
                        z_ref = zb
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

    # Soft foot length from front toe/heel Z when meters present (R4).
    messages.extend(_foot_len_messages(out))

    return out, quality, messages


def _foot_len_messages(xyz: dict[str, LandmarkXYZ]) -> list[str]:
    """Emit foot_len_{l,r}_m when toe+heel both have z_m (R4; no schema bump)."""
    msgs: list[str] = []
    for side in ("l", "r"):
        toe = xyz.get(f"toe_{side}")
        heel = xyz.get(f"heel_{side}")
        if toe is None or heel is None:
            continue
        if toe.z_m is None or heel.z_m is None:
            continue
        length_m = abs(float(toe.z_m) - float(heel.z_m))
        msgs.append(f"foot_len_{side}_m~={length_m:.6f}")
    return msgs


def build_depth_bands(
    views: dict[str, ViewLandmarks],
    *,
    height_m: float | None = None,
    foreshortening_risk: bool = False,
) -> tuple[list[DepthBand], list[CheckResult], list[str]]:
    """Build DepthBand list from left front/back pairs with orientation auto-swap (R12).

    torso_cx is reused for all bands (approximation for non-torso — R9).
    """
    messages: list[str] = []
    checks: list[CheckResult] = []
    bands: list[DepthBand] = []

    left = views.get("left")
    if left is None or not left.landmarks:
        return bands, checks, messages

    front = views.get("front")
    left_span = left.figure_span_px or figure_span_from_landmarks(left)
    if left_span is None or left_span <= 0:
        if front is not None:
            left_span = front.figure_span_px or figure_span_from_landmarks(front)
        if left_span is None or left_span <= 0:
            messages.append("cannot build depth_bands (no left/front span)")
            return bands, checks, messages
        messages.append("depth_bands: left span unknown; using front span")

    facing = left.facing_direction or "camera_left"
    sign = _depth_sign(str(facing) if facing else "camera_left")
    torso_cx = _torso_center_x(left)
    conf_scale = 0.7 if foreshortening_risk else 1.0

    non_torso_used = False
    for front_id, back_id, mid_id in DEPTH_PAIRS:
        lf = left.landmarks.get(front_id)
        lb = left.landmarks.get(back_id)
        if lf is None or lb is None:
            continue

        band_id = _band_id_from_pair(front_id)
        if band_id not in ("chest", "hip", "breast"):
            non_torso_used = True

        y_front = sign * (lf.x_px - torso_cx) / left_span
        y_back = sign * (lb.x_px - torso_cx) / left_span
        swapped = False
        # Require y_front > y_back (+Y toward camera/front). If inverted → swap.
        if y_front < y_back:
            y_front, y_back = y_back, y_front
            swapped = True
            checks.append(
                CheckResult(
                    name="depth_band_orientation",
                    ok=True,
                    severity="info",
                    message=(
                        f"{band_id}: y_front < y_back after body-depth map — "
                        "auto-swapped endpoints (orientation_swapped)"
                    ),
                    measured={"band_id": band_id, "swapped": True},
                    expected="y_front > y_back (+Y toward camera)",
                )
            )

        depth_px = abs(lf.x_px - lb.x_px)
        depth_frac = abs(y_front - y_back)  # == depth_px / left_span
        y_mid = (y_front + y_back) / 2.0
        z_front = _z_from_view(left, lf, left_span)
        z_back = _z_from_view(left, lb, left_span)
        z_frac: float | None
        if z_front is not None and z_back is not None:
            z_frac = (z_front + z_back) / 2.0
        else:
            z_frac = z_front if z_front is not None else z_back

        conf = min(1.0, min(lf.confidence, lb.confidence) * conf_scale * 0.9)
        depth_m = depth_frac * height_m if height_m is not None else None
        bands.append(
            DepthBand(
                band_id=band_id,
                depth_px=depth_px,
                depth_frac=depth_frac,
                depth_m=depth_m,
                y_front=y_front,
                y_back=y_back,
                y_mid=y_mid,
                z_frac=z_frac,
                confidence=conf,
                sources=["left", front_id, back_id, mid_id],
                orientation_swapped=swapped,
            )
        )

    if non_torso_used:
        messages.append(
            "depth_bands: torso_cx reused for non-torso bands "
            "(thigh/calf/glute/cranial/foot) — approximation (R9)"
        )
    if bands:
        messages.append(f"depth_bands: {len(bands)} band(s)")
    return bands, checks, messages


def level_id_from_band(band_id: str) -> str:
    """Normalize diameter/depth band id to a shared level_id for cross-sections."""
    bid = band_id
    if bid.endswith("_l") or bid.endswith("_r"):
        bid = bid[:-2]
    # Front diameter "bust" correlates with left depth "breast"
    if bid == "bust":
        return "breast"
    return bid


def build_cross_sections(
    diameters: list[DiameterMeasure],
    depth_bands: list[DepthBand],
    *,
    z_tol: float = _CROSS_Z_TOL,
) -> list[CrossSection]:
    """Emit CrossSection when diameter + depth share level and |z_frac| within tol (R13)."""
    if not diameters or not depth_bands:
        return []

    # Prefer first depth band per level_id
    depth_by_level: dict[str, DepthBand] = {}
    for db in depth_bands:
        lid = level_id_from_band(db.band_id)
        if lid not in depth_by_level:
            depth_by_level[lid] = db

    out: list[CrossSection] = []
    seen: set[str] = set()
    for d in diameters:
        if d.z_frac is None:
            continue
        level = level_id_from_band(d.band_id)
        db = depth_by_level.get(level)
        if db is None or db.z_frac is None:
            continue
        if abs(d.z_frac - db.z_frac) > z_tol:
            continue
        if level in seen:
            continue
        seen.add(level)
        half_w = d.half_width_frac if d.half_width_frac is not None else d.width_frac / 2.0
        half_d = db.depth_frac / 2.0
        out.append(
            CrossSection(
                level_id=level,
                z_frac=(d.z_frac + db.z_frac) / 2.0,
                rx_frac=half_w,
                ry_frac=half_d,
                sources=[f"diameter:{d.band_id}", f"depth:{db.band_id}"],
            )
        )
    return out


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
    """Torso center X from left depth pairs (chest → hip → other torso only).

    B4 load-bearing: never use cranial/foot pairs as torso center — head/foot
    midpoints are not torso. Fall through to spine_hint or width/2 instead.
    """
    lm = left.landmarks
    cf, cb = lm.get("chest_front"), lm.get("chest_back")
    if cf is not None and cb is not None:
        return (cf.x_px + cb.x_px) / 2.0
    hf, hb = lm.get("hip_front"), lm.get("hip_back")
    if hf is not None and hb is not None:
        return (hf.x_px + hb.x_px) / 2.0
    # Fall back to other torso depth pairs if chest/hip missing (skip head/foot).
    for front_id, back_id, _mid in DEPTH_PAIRS:
        band_id = _band_id_from_pair(front_id)
        if band_id in _TORSO_CX_SKIP_BANDS:
            continue
        a, b = lm.get(front_id), lm.get(back_id)
        if a is not None and b is not None:
            return (a.x_px + b.x_px) / 2.0
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
    """Weighted completeness score 0-100 (spec section 3.1 F).

    Depth points only for chest_front+back OR hip_front+back (R8 — unchanged by 0013).
    """
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
        # R8: chest/hip only — glute/thigh/breast never score-weight depth
        depth_ok = ("chest_front" in llm and "chest_back" in llm) or (
            "hip_front" in llm and "hip_back" in llm
        )
        if depth_ok:
            breakdown["depth"] = SCORE_DEPTH
        # Silence unused set (documents freeze)
        _ = _SCORE_DEPTH_BANDS

    total = sum(breakdown.values())
    # Clamp float dust
    total = min(100.0, round(total, 4))
    return total, breakdown
