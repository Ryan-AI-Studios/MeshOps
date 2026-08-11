"""Head / face / hair / neckline RECIPE primitives (track 0028).

Opt-in blockout-grade face kit parented to head/neck skeleton or Loomis placement.
Authoring only - not identity biometrics, not print success (Difficulty §12 / N6).

All names stay RECIPE_* (never FACE_*/HAIR_*/NECKLINE_* prefixes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

from meshops.proportion.models import DepthBand, LandmarkXYZ

if TYPE_CHECKING:
    from meshops.proportion.blockout_recipe import RecipePart
    from meshops.proportion.models import ProportionReport
    from meshops.proportion.skeleton import BlockoutSkeleton

HairTier = Literal["none", "short", "bun", "long_proxy"]
NecklineTier = Literal["none", "crew", "v_proxy"]
HeadPlacement = Literal["full3d", "front_plane"]

HAIR_TIERS: Final[frozenset[str]] = frozenset({"none", "short", "bun", "long_proxy"})
NECKLINE_TIERS: Final[frozenset[str]] = frozenset({"none", "crew", "v_proxy"})

# B11 exact skip message (em-dash per freeze)
FACE_KIT_SKIP_BOUNDS: Final[str] = "face kit skipped — chin/top bounds unresolved"

# Loomis / authoring freezes (plan §7)
_EYE_Z_FRAC: Final[float] = 0.50
_BROW_Z_FRAC: Final[float] = 0.67
_NOSE_BASE_Z_FRAC: Final[float] = 0.33
_LIP_Z_FRAC: Final[float] = 0.20  # authoring choice, not Loomis third
_EYE_RADIUS_FRAC_H: Final[float] = 0.08
# 0055 jaw soft mass ellipsoid; 0057 chin-strap polish (public; use HeadBounds.H for Z fracs)
JAW_RX_FRAC_HEAD_RX: Final[float] = 0.74  # was 0.85 — 0057
JAW_RY_FRAC_HEAD_RY: Final[float] = 0.42  # was 0.55 — 0078 left chin shelf
JAW_RZ_FRAC_H: Final[float] = 0.13  # was 0.15 — 0078 face U-band height
JAW_Z_CENTER_FRAC_H: Final[float] = 0.13
JAW_Y_BIAS_FRAC_RY: Final[float] = 0.08  # was 0.05 — 0057
JAW_X_BULGE_ALLOW_M: Final[float] = 0.006  # was 0.015 — 0057
# Jaw / legacy face plane (keep jaw 0057 placement; features use FEATURE_* plane)
_JAW_FACE_Y_FRAC_RY: Final[float] = 0.40
# 0058 face feature softs (public — export in __all__, P2-2)
# FEATURE_FACE_Y: D7 Codex P1 — was 0.40 (features buried inside head shell).
# Near-surface plane so eye/brow/lip/cheek read on solid workbench multi-view.
FEATURE_FACE_Y_FRAC_RY: Final[float] = 0.90
EYE_RX_FRAC_R: Final[float] = 1.00  # stay
EYE_RY_FRAC_R: Final[float] = 0.95  # was 0.85 (B15 D7 left pad) / was 0.25
EYE_RZ_FRAC_R: Final[float] = 0.45  # was 0.70

NOSE_RX_FRAC_H: Final[float] = 0.045
NOSE_RY_FRAC_H: Final[float] = 0.055
NOSE_RZ_FRAC_H: Final[float] = 0.040
# D7 surface-readable tip: was 0.15 (deep embed; invisible on solid head).
# tip_y = head.y - NOSE_TIP_Y_FRAC_RY * head.ry; center_y = tip_y + nose_ry
# front surface = center_y - nose_ry = tip_y; center_z ≈ nose_base_z - 0.01*H
NOSE_TIP_Y_FRAC_RY: Final[float] = 0.98

LIP_RX_FRAC_H: Final[float] = 0.10  # was 0.12 — 0078 pad thin
LIP_RY_FRAC_H: Final[float] = 0.022  # was 0.035 — 0078 pad thin
LIP_RZ_FRAC_H: Final[float] = 0.016  # was 0.025 — 0078 pad thin

BROW_R_FRAC_H: Final[float] = 0.028  # was 0.015
BROW_HALF_LEN_FRAC_EYE_R: Final[float] = 1.1  # stay

CHEEK_RX_FRAC_HEAD_RX: Final[float] = 0.28
CHEEK_RY_FRAC_HEAD_RY: Final[float] = 0.22
CHEEK_RZ_FRAC_H: Final[float] = 0.06
CHEEK_X_FRAC_HEAD_RX: Final[float] = 0.55
CHEEK_Z_MIX: Final[float] = 0.50  # blend eye_z ↔ nose_base_z
CHEEK_Y_BIAS_FRAC_RY: Final[float] = 0.05
_HAIR_SHORT_RZ_FRAC: Final[float] = 0.25
_BUN_R_FRAC_H: Final[float] = 0.12
_LONG_PROXY_LEN_FRAC_H: Final[float] = 0.45
_CREW_LEN_SHOULDER_FRAC: Final[float] = 0.4
_CREW_LEN_H_FALLBACK: Final[float] = 0.25
_NECKLINE_R_FRAC_H: Final[float] = 0.012
_V_X_SHOULDER_FRAC: Final[float] = 0.05
_V_DOWN_Z_FRAC_H: Final[float] = 0.08
_V_ANGLE_DEG: Final[float] = 15.0
_FUSE_EPS_FRAC: Final[float] = 0.02
_FUSE_R_NECK_FRAC: Final[float] = 0.8
_SCM_R_FRAC_H: Final[float] = 0.012


@dataclass(frozen=True)
class HeadBounds:
    """Shared head geometry for RECIPE_head + face kit (B17)."""

    z_chin: float
    z_top: float
    z_c: float
    H: float  # z_top - z_chin
    rx: float
    ry: float
    rz: float
    y: float
    placement: HeadPlacement
    has_y: bool
    # True only when cranial_vertex or hair_crown provided z_top (B11 face kit).
    top_from_landmark: bool = True


def scale_head_bounds(
    bounds: HeadBounds, *, rx_scale: float, ry_scale: float, rz_scale: float
) -> HeadBounds:
    """Apply template head radius/depth scales to shared bounds (same path as RECIPE_head)."""
    return replace(
        bounds,
        rx=float(bounds.rx) * float(rx_scale),
        ry=float(bounds.ry) * float(ry_scale),
        rz=float(bounds.rz) * float(rz_scale),
    )


def _top_landmark_with_z(lms: dict[str, LandmarkXYZ]) -> LandmarkXYZ | None:
    """First of cranial_vertex / hair_crown with finite z_m (do not mask crown)."""
    for key in ("cranial_vertex", "hair_crown"):
        lm = lms.get(key)
        if lm is not None and lm.z_m is not None:
            return lm
    return None


def resolve_head_bounds(
    report: ProportionReport,
    *,
    head_unit_m: float | None,
    height_m: float | None,
    messages: list[str],
    chest_y: float | None = None,
) -> HeadBounds | None:
    """Resolve HeadBounds from landmarks (same ladder as pre-0028 RECIPE_head).

    When top z is invented from head_unit/stature, *top_from_landmark* is False so
    the face kit can refuse to invent H (B11) while RECIPE_head still emits.

    When chin/top lack y_m, *chest_y* (axial mid, B2 ladder) is the no-Y fallback —
    not a hardcode of 0.0 alone and never chest_front (0032 B5).
    """
    from meshops.proportion.blockout_recipe import (
        _half_width_from_diameter,
        _resolve_diameter,
    )

    lms = report.landmarks_xyz
    chin = lms.get("chin")
    top = _top_landmark_with_z(lms)
    if chin is None or chin.z_m is None:
        messages.append("RECIPE_head skipped: need chin z_m")
        return None
    z_chin = float(chin.z_m)
    top_from_landmark = False
    if top is not None and top.z_m is not None:
        z_top = float(top.z_m)
        top_from_landmark = True
    elif head_unit_m is not None:
        z_top = z_chin + 0.75 * float(head_unit_m)
    elif height_m is not None:
        z_top = z_chin + 0.10 * float(height_m)
    else:
        messages.append("RECIPE_head skipped: insufficient z for head")
        return None
    if z_top <= z_chin:
        messages.append("RECIPE_head skipped: top z <= chin z")
        return None
    z_c = (z_chin + z_top) / 2.0
    h = z_top - z_chin
    rz = h / 2.0
    diam = _resolve_diameter(report.diameters, "head")
    rx: float | None = None
    if diam is not None:
        rx = _half_width_from_diameter(diam)
    if rx is None:
        if head_unit_m is not None:
            rx = 0.40 * float(head_unit_m)
        elif height_m is not None:
            rx = 0.06 * float(height_m)
        else:
            messages.append(
                "RECIPE_head skipped: no head diameter, head_unit_m, or height_m for radius"
            )
            return None
    # ry prefer: cranial depth_m (meters) / 2 → landmark front/back span / 2 → 0.9*rx
    # Unit freeze: depth_m and landmark y_m are meters; never band y_mid/y_front fractions.
    ry: float
    cranial_band: DepthBand | None = None
    for band in report.depth_bands or []:
        if band.band_id == "cranial":
            cranial_band = band
            break
    if (
        cranial_band is not None
        and cranial_band.depth_m is not None
        and math.isfinite(cranial_band.depth_m)
    ):
        ry = float(cranial_band.depth_m) / 2.0
        messages.append("head ry from cranial depth_m")
    else:
        cf = lms.get("cranial_front")
        cb = lms.get("cranial_back")
        if (
            cf is not None
            and cb is not None
            and cf.y_m is not None
            and cb.y_m is not None
            and math.isfinite(cf.y_m)
            and math.isfinite(cb.y_m)
        ):
            ry = abs(float(cb.y_m) - float(cf.y_m)) / 2.0
            messages.append("head ry from cranial_front/back span")
        else:
            ry = float(rx) * 0.9
            messages.append("head ry fallback 0.9*rx (no cranial depth)")

    # Prefer chin/top landmark Y; else cranial_mid / pair mean (meters); else chest_y / 0.0
    y = 0.0
    has_y = False
    if chin.y_m is not None:
        y = float(chin.y_m)
        has_y = True
    elif top is not None and top.y_m is not None:
        y = float(top.y_m)
        has_y = True
    else:
        cm = lms.get("cranial_mid")
        if cm is not None and cm.y_m is not None and math.isfinite(cm.y_m):
            y = float(cm.y_m)
            has_y = True
            messages.append("head center y from cranial_mid")
        else:
            cf = lms.get("cranial_front")
            cb = lms.get("cranial_back")
            if (
                cf is not None
                and cb is not None
                and cf.y_m is not None
                and cb.y_m is not None
                and math.isfinite(cf.y_m)
                and math.isfinite(cb.y_m)
            ):
                y = (float(cf.y_m) + float(cb.y_m)) / 2.0
                has_y = True
                messages.append("head center y from cranial_front/back pair mean")
            else:
                y = float(chest_y) if chest_y is not None else 0.0
                has_y = False
    placement: HeadPlacement = "full3d" if has_y else "front_plane"
    return HeadBounds(
        z_chin=z_chin,
        z_top=z_top,
        z_c=z_c,
        H=h,
        rx=float(rx),
        ry=ry,
        rz=rz,
        y=y,
        placement=placement,
        has_y=has_y,
        top_from_landmark=top_from_landmark,
    )


def head_part_from_bounds(bounds: HeadBounds) -> RecipePart:
    """Emit RECIPE_head ellipsoid from shared HeadBounds."""
    from meshops.proportion.blockout_recipe import RecipePart

    return RecipePart(
        name="RECIPE_head",
        role="head",
        kind="ellipsoid",
        center=[0.0, bounds.y, bounds.z_c],
        rx_m=bounds.rx,
        ry_m=bounds.ry,
        rz_m=bounds.rz,
        placement=bounds.placement,
        label="RECIPE_head",
    )


def _parent_joint(
    preferred: str,
    fallbacks: list[str],
    skeleton: BlockoutSkeleton | None,
    *,
    role: str,
    messages: list[str],
) -> str | None:
    """Resolve parent_joint id; null + message if unresolved (B10)."""
    if skeleton is None:
        return None
    from meshops.proportion.blockout_recipe import _joints_map, _resolve_parent_joint_id

    joints = _joints_map(skeleton)
    pid = _resolve_parent_joint_id(preferred, fallbacks, joints, side="none")
    if pid is None:
        messages.append(f"parent_joint {role} unresolved — using landmark/Loomis placement")
    return pid


def _ellipsoid(
    name: str,
    role: str,
    center: list[float],
    rx: float,
    ry: float,
    rz: float,
    *,
    placement: HeadPlacement,
    parent_joint: str | None = None,
    notes: str | None = None,
) -> RecipePart:
    from meshops.proportion.blockout_recipe import RecipePart

    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind="ellipsoid",
        center=center,
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
        placement=placement,
        label=name,
        parent_joint=parent_joint,
        notes=notes,
    )


def _capsule(
    name: str,
    role: str,
    p0: list[float],
    p1: list[float],
    radius: float,
    *,
    placement: HeadPlacement,
    parent_joint: str | None = None,
    notes: str | None = None,
) -> RecipePart:
    from meshops.proportion.blockout_recipe import RecipePart

    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind="capsule",
        p0=p0,
        p1=p1,
        radius_m=radius,
        placement=placement,
        label=name,
        parent_joint=parent_joint,
        notes=notes,
    )


def _build_face_features(
    bounds: HeadBounds,
    *,
    skeleton: BlockoutSkeleton | None,
    messages: list[str],
) -> list[RecipePart]:
    """Jaw + brows + eyes + nose + ears + lip (R1 / B7)."""
    h = bounds.H
    z_chin = bounds.z_chin
    y = bounds.y
    rx = bounds.rx
    ry = bounds.ry
    placement = bounds.placement

    eye_r = _EYE_RADIUS_FRAC_H * h
    eye_z = z_chin + _EYE_Z_FRAC * h
    brow_z = z_chin + _BROW_Z_FRAC * h
    nose_base_z = z_chin + _NOSE_BASE_Z_FRAC * h
    lip_z = z_chin + _LIP_Z_FRAC * h
    # Jaw keeps legacy embed plane (0057 fence). Feature softs use near-surface plane (D7).
    jaw_face_y = y - _JAW_FACE_Y_FRAC_RY * ry
    feature_face_y = y - FEATURE_FACE_Y_FRAC_RY * ry
    # Inter-eye gap ~ one eye width; half-sep = 2 * eye_r
    eye_half_sep = 2.0 * eye_r
    nose_tip_y = y - NOSE_TIP_Y_FRAC_RY * ry

    pj_head = _parent_joint("head", ["chin", "crown"], skeleton, role="eye_soft", messages=messages)
    # Prefer head for features; jaw prefers chin
    pj_jaw = _parent_joint("chin", ["head"], skeleton, role="jaw", messages=messages)
    # Reuse head parent for remaining features without spamming messages
    pj_feat = pj_head
    if skeleton is not None and pj_feat is None:
        # one message already emitted for eye_soft; still try silent resolve for others
        from meshops.proportion.blockout_recipe import _joints_map, _resolve_parent_joint_id

        pj_feat = _resolve_parent_joint_id(
            "head", ["chin", "crown"], _joints_map(skeleton), side="none"
        )

    parts: list[RecipePart] = []

    # 0055: jaw soft mass ellipsoid (not world-axis box); Z fracs use HeadBounds.H
    jaw_rx = JAW_RX_FRAC_HEAD_RX * rx
    jaw_ry = JAW_RY_FRAC_HEAD_RY * ry
    jaw_rz = JAW_RZ_FRAC_H * h
    jaw_z_c = z_chin + JAW_Z_CENTER_FRAC_H * h
    jaw_y = jaw_face_y + JAW_Y_BIAS_FRAC_RY * ry
    parts.append(
        _ellipsoid(
            "RECIPE_jaw",
            "jaw",
            [0.0, jaw_y, jaw_z_c],
            jaw_rx,
            jaw_ry,
            jaw_rz,
            placement=placement,
            parent_joint=pj_jaw,
        )
    )
    messages.append(
        f"face: jaw soft mass ellipsoid rx={jaw_rx:.4f} ry={jaw_ry:.4f} rz={jaw_rz:.4f}"
    )
    # Product X-bulge vs head ellipsoid surface at jaw center Z (AI2 P2-1 / B13)
    if math.isfinite(rx) and math.isfinite(bounds.rz) and abs(float(bounds.rz)) > 1e-12:
        t = (jaw_z_c - bounds.z_c) / float(bounds.rz)
        head_x = float(rx) * math.sqrt(max(0.0, 1.0 - t * t))
        bulge = float(jaw_rx) - head_x
        messages.append(f"face: jaw_vs_head_x_bulge_m={bulge:.4f} (allow={JAW_X_BULGE_ALLOW_M})")

    # Brows L/R - horizontal capsules (0058: BROW_R_FRAC_H floor)
    brow_half_len = BROW_HALF_LEN_FRAC_EYE_R * eye_r
    brow_r = BROW_R_FRAC_H * h
    for side, sx in (("l", -1.0), ("r", 1.0)):
        cx = sx * eye_half_sep
        parts.append(
            _capsule(
                f"RECIPE_brow_soft_{side}",
                "brow_soft",
                [cx - brow_half_len, feature_face_y, brow_z],
                [cx + brow_half_len, feature_face_y, brow_z],
                brow_r,
                placement=placement,
                parent_joint=pj_feat,
            )
        )
    messages.append(f"face: brow soft capsule r={brow_r:.4f} (floor frac={BROW_R_FRAC_H})")

    # Eyes L/R - product-like orbital pads (0058: depth ≥ height; near-surface plane)
    eye_rx = EYE_RX_FRAC_R * eye_r
    eye_ry = EYE_RY_FRAC_R * eye_r
    eye_rz = EYE_RZ_FRAC_R * eye_r
    for side, sx in (("l", -1.0), ("r", 1.0)):
        parts.append(
            _ellipsoid(
                f"RECIPE_eye_soft_{side}",
                "eye_soft",
                [sx * eye_half_sep, feature_face_y, eye_z],
                eye_rx,
                eye_ry,
                eye_rz,
                placement=placement,
                parent_joint=pj_feat,
            )
        )
    messages.append(f"face: eye soft axes rx={eye_rx:.4f} ry={eye_ry:.4f} rz={eye_rz:.4f}")

    # Nose - short wedge ellipsoid (0058; front surface Y = tip_y near head front)
    nose_ry = NOSE_RY_FRAC_H * h
    nose_rx = NOSE_RX_FRAC_H * h
    nose_rz = NOSE_RZ_FRAC_H * h
    nose_center_y = nose_tip_y + nose_ry
    nose_center_z = nose_base_z - 0.01 * h
    parts.append(
        _ellipsoid(
            "RECIPE_nose_soft",
            "nose_soft",
            [0.0, nose_center_y, nose_center_z],
            nose_rx,
            nose_ry,
            nose_rz,
            placement=placement,
            parent_joint=pj_feat,
        )
    )
    messages.append(f"face: nose soft ellipsoid rx={nose_rx:.4f} ry={nose_ry:.4f} rz={nose_rz:.4f}")

    # Ears L/R - span brow→nose base Z, lateral ±rx (0028 stay)
    ear_z = (brow_z + nose_base_z) / 2.0
    ear_rz = max((brow_z - nose_base_z) / 2.0, 0.04 * h)
    ear_rx = 0.08 * h
    ear_ry = 0.04 * h
    for side, sx in (("l", -1.0), ("r", 1.0)):
        parts.append(
            _ellipsoid(
                f"RECIPE_ear_soft_{side}",
                "ear_soft",
                [sx * rx, y, ear_z],
                ear_rx,
                ear_ry,
                ear_rz,
                placement=placement,
                parent_joint=pj_feat,
            )
        )

    # Lip - closed-mouth readable bar (0058 floors; near-surface plane)
    lip_rx = LIP_RX_FRAC_H * h
    lip_ry = LIP_RY_FRAC_H * h
    lip_rz = LIP_RZ_FRAC_H * h
    parts.append(
        _ellipsoid(
            "RECIPE_lip_soft",
            "lip_soft",
            [0.0, feature_face_y, lip_z],
            lip_rx,
            lip_ry,
            lip_rz,
            placement=placement,
            parent_joint=pj_feat,
        )
    )
    messages.append(f"face: lip soft axes rx={lip_rx:.4f} ry={lip_ry:.4f} rz={lip_rz:.4f}")

    # Cheek - mild single pad per side (0058; not multi-pad photoreal)
    cheek_rx = CHEEK_RX_FRAC_HEAD_RX * rx
    cheek_ry = CHEEK_RY_FRAC_HEAD_RY * ry
    cheek_rz = CHEEK_RZ_FRAC_H * h
    cheek_z = CHEEK_Z_MIX * eye_z + (1.0 - CHEEK_Z_MIX) * nose_base_z
    cheek_y = feature_face_y + CHEEK_Y_BIAS_FRAC_RY * ry
    for side, sx in (("l", -1.0), ("r", 1.0)):
        parts.append(
            _ellipsoid(
                f"RECIPE_cheek_soft_{side}",
                "cheek_soft",
                [sx * CHEEK_X_FRAC_HEAD_RX * rx, cheek_y, cheek_z],
                cheek_rx,
                cheek_ry,
                cheek_rz,
                placement=placement,
                parent_joint=pj_feat,
            )
        )
    messages.append(
        f"face: cheek soft pads present L/R rx={cheek_rx:.4f} ry={cheek_ry:.4f} rz={cheek_rz:.4f}"
    )

    return parts


def _build_scm(
    bounds: HeadBounds,
    *,
    skeleton: BlockoutSkeleton | None,
    shoulder_z: float | None,
    chest_y: float | None,
    neck_len_m: float | None,
    messages: list[str],
) -> list[RecipePart]:
    """Sternomastoid L/R capsules neck_base to near ear base (B14). With --face when neck ok."""
    h = bounds.H
    placement = bounds.placement
    y = bounds.y if bounds.has_y else (chest_y if chest_y is not None else 0.0)

    neck_base_z: float | None = None
    neck_base_y = y

    if skeleton is not None:
        from meshops.proportion.blockout_recipe import _joint_xyz, _joints_map

        jmap = _joints_map(skeleton)
        nb = _joint_xyz(jmap.get("neck_base"))
        if nb is not None:
            neck_base_z = nb[2]
            neck_base_y = nb[1]
    if neck_base_z is None:
        if shoulder_z is not None:
            neck_base_z = float(shoulder_z)
            if chest_y is not None:
                neck_base_y = float(chest_y)
        elif neck_len_m is not None:
            # Fallback: chin - neck_len ≈ shoulder
            neck_base_z = bounds.z_chin - float(neck_len_m)
        else:
            messages.append("sternomastoid skipped — neck_base / shoulder unavailable")
            return []

    pj = _parent_joint(
        "neck_base", ["neck_top", "head"], skeleton, role="sternomastoid_soft", messages=messages
    )

    # Ear base approx lateral at mid ear Z
    brow_z = bounds.z_chin + _BROW_Z_FRAC * h
    nose_base_z = bounds.z_chin + _NOSE_BASE_Z_FRAC * h
    ear_z = (brow_z + nose_base_z) / 2.0
    ear_base_z = ear_z - 0.3 * ((brow_z - nose_base_z) / 2.0)
    r = _SCM_R_FRAC_H * h
    parts: list[RecipePart] = []
    for side, sx in (("l", -1.0), ("r", 1.0)):
        p0 = [sx * 0.03 * h, neck_base_y, float(neck_base_z)]
        p1 = [sx * bounds.rx * 0.85, y, ear_base_z]
        if math.dist(p0, p1) <= 1e-9:
            continue
        parts.append(
            _capsule(
                f"RECIPE_sternomastoid_soft_{side}",
                "sternomastoid_soft",
                p0,
                p1,
                r,
                placement=placement,
                parent_joint=pj,
            )
        )
    return parts


def _build_fuse(
    bounds: HeadBounds,
    *,
    skeleton: BlockoutSkeleton | None,
    neck_top_z: float | None,
    neck_radius: float | None,
    head_unit_m: float | None,
    messages: list[str],
) -> list[RecipePart]:
    """Optional RECIPE_neck_head_fuse when gap large (B15)."""
    head_bottom = bounds.z_c - bounds.rz  # == z_chin for ellipsoid head
    if neck_top_z is None:
        return []
    gap = float(head_bottom) - float(neck_top_z)
    eps_ref = float(head_unit_m) if head_unit_m is not None else bounds.H
    eps = _FUSE_EPS_FRAC * eps_ref
    if gap <= eps:
        return []
    r_neck = float(neck_radius) if neck_radius is not None else 0.05 * bounds.H
    r = _FUSE_R_NECK_FRAC * r_neck
    pj = _parent_joint("neck_top", ["neck_base", "head"], skeleton, role="neck", messages=messages)
    y = bounds.y
    return [
        _capsule(
            "RECIPE_neck_head_fuse",
            "neck",
            [0.0, y, float(neck_top_z)],
            [0.0, y, float(head_bottom)],
            r,
            placement=bounds.placement,
            parent_joint=pj,
            notes=f"fuse gap={gap:.4f}m",
        )
    ]


def _build_hair(
    bounds: HeadBounds,
    hair: str,
    *,
    skeleton: BlockoutSkeleton | None,
    messages: list[str],
) -> list[RecipePart]:
    """Hair mass tiers: short | bun | long_proxy (B8)."""
    if hair == "none":
        return []
    h = bounds.H
    placement = bounds.placement
    y = bounds.y
    crown_z = bounds.z_top
    pj = _parent_joint("crown", ["head"], skeleton, role="hair_mass", messages=messages)
    parts: list[RecipePart] = []

    if hair in ("short", "bun"):
        # Shallow cap ellipsoid on crown
        cap_rz = _HAIR_SHORT_RZ_FRAC * bounds.rz
        parts.append(
            _ellipsoid(
                "RECIPE_hair_mass",
                "hair_mass",
                [0.0, y, crown_z - 0.15 * bounds.rz],
                bounds.rx * 1.05,
                bounds.ry * 1.05,
                cap_rz,
                placement=placement,
                parent_joint=pj,
            )
        )
        if hair == "bun":
            bun_r = _BUN_R_FRAC_H * h
            parts.append(
                _ellipsoid(
                    "RECIPE_hair_mass_bun",
                    "hair_mass",
                    [0.0, y + 0.05 * bounds.ry, crown_z + bun_r * 0.6],
                    bun_r,
                    bun_r,
                    bun_r,
                    placement=placement,
                    parent_joint=pj,
                )
            )
    elif hair == "long_proxy":
        # Elongated ellipsoid rear (+Y) from crown
        length = _LONG_PROXY_LEN_FRAC_H * h
        parts.append(
            _ellipsoid(
                "RECIPE_hair_mass",
                "hair_mass",
                [0.0, y + 0.55 * length, crown_z - 0.35 * length],
                bounds.rx * 0.85,
                0.45 * length,
                0.55 * length,
                placement=placement,
                parent_joint=pj,
            )
        )
    else:
        messages.append(f"unknown hair tier {hair!r} - no hair mass")
    return parts


def _build_neckline(
    bounds: HeadBounds,
    neckline: str,
    *,
    skeleton: BlockoutSkeleton | None,
    shoulder_hw: float | None,
    shoulder_z: float | None,
    chest_y: float | None,
    messages: list[str],
) -> list[RecipePart]:
    """Crew or v_proxy neckline (B9)."""
    if neckline == "none":
        return []
    h = bounds.H
    placement = bounds.placement
    r = _NECKLINE_R_FRAC_H * h

    neck_z: float | None = None
    neck_y = bounds.y - 0.05 * h
    if chest_y is not None:
        neck_y = float(chest_y) - 0.05 * h
    # chest_y is axial mid-depth (B2 ladder / 0032), not chest_front

    pj = _parent_joint("neck_base", ["neck_top"], skeleton, role="neckline", messages=messages)
    if skeleton is not None:
        from meshops.proportion.blockout_recipe import _joint_xyz, _joints_map

        nb = _joint_xyz(_joints_map(skeleton).get("neck_base"))
        if nb is not None:
            neck_z = nb[2]
            neck_y = nb[1] - 0.02 * h
    if neck_z is None:
        if shoulder_z is not None:
            neck_z = float(shoulder_z)
        else:
            neck_z = bounds.z_chin - 0.5 * (bounds.z_chin - (shoulder_z or bounds.z_chin - 0.1 * h))
            if shoulder_z is None:
                messages.append("neckline: neck_base/shoulder z fallback from head")

    shw = float(shoulder_hw) if shoulder_hw is not None else None
    parts: list[RecipePart] = []

    if neckline == "crew":
        half_len = 0.5 * (
            _CREW_LEN_SHOULDER_FRAC * shw if shw is not None else _CREW_LEN_H_FALLBACK * h
        )
        parts.append(
            _capsule(
                "RECIPE_neckline_crew",
                "neckline",
                [-half_len, neck_y, float(neck_z)],
                [half_len, neck_y, float(neck_z)],
                r,
                placement=placement,
                parent_joint=pj,
            )
        )
    elif neckline == "v_proxy":
        dx = _V_X_SHOULDER_FRAC * shw if shw is not None else 0.05 * h
        down = _V_DOWN_Z_FRAC_H * h
        ang = math.radians(_V_ANGLE_DEG)
        for side, sx in (("l", -1.0), ("r", 1.0)):
            # From neck_base ±dx X, extend down-Z with slight outward X (~15° from vertical)
            p0 = [sx * dx, neck_y, float(neck_z)]
            p1 = [
                sx * (dx + down * math.sin(ang)),
                neck_y,
                float(neck_z) - down * math.cos(ang),
            ]
            parts.append(
                _capsule(
                    f"RECIPE_neckline_v_{side}",
                    "neckline",
                    p0,
                    p1,
                    r,
                    placement=placement,
                    parent_joint=pj,
                )
            )
    else:
        messages.append(f"unknown neckline tier {neckline!r} - no neckline")
    return parts


def build_face_parts(
    report: ProportionReport,
    head_bounds: HeadBounds | None,
    *,
    face: bool = False,
    hair: str = "none",
    neckline: str = "none",
    skeleton: BlockoutSkeleton | None = None,
    shoulder_hw: float | None = None,
    neck_len_m: float | None = None,
    shoulder_z: float | None = None,
    chest_y: float | None = None,
    neck_top_z: float | None = None,
    neck_radius: float | None = None,
    head_unit_m: float | None = None,
    messages: list[str] | None = None,
) -> list[RecipePart]:
    """Emit face/hair/neckline RECIPE parts from shared HeadBounds (B7-B15).

    *report* reserved for future measured face diameters; placement uses *head_bounds*.
    """
    del report  # placement is Loomis/bounds-driven in v1
    msgs = messages if messages is not None else []
    if not face and hair == "none" and neckline == "none":
        return []
    # B11: never invent H for face/hair/neckline — require landmark top z.
    if head_bounds is None or not head_bounds.top_from_landmark:
        msgs.append(FACE_KIT_SKIP_BOUNDS)
        return []

    parts: list[RecipePart] = []
    if face:
        parts.extend(_build_face_features(head_bounds, skeleton=skeleton, messages=msgs))
        parts.extend(
            _build_scm(
                head_bounds,
                skeleton=skeleton,
                shoulder_z=shoulder_z,
                chest_y=chest_y,
                neck_len_m=neck_len_m,
                messages=msgs,
            )
        )
        parts.extend(
            _build_fuse(
                head_bounds,
                skeleton=skeleton,
                neck_top_z=neck_top_z,
                neck_radius=neck_radius,
                head_unit_m=head_unit_m,
                messages=msgs,
            )
        )
    parts.extend(_build_hair(head_bounds, hair, skeleton=skeleton, messages=msgs))
    parts.extend(
        _build_neckline(
            head_bounds,
            neckline,
            skeleton=skeleton,
            shoulder_hw=shoulder_hw,
            shoulder_z=shoulder_z,
            chest_y=chest_y,
            messages=msgs,
        )
    )
    return parts


__all__ = [
    "BROW_HALF_LEN_FRAC_EYE_R",
    "BROW_R_FRAC_H",
    "CHEEK_RX_FRAC_HEAD_RX",
    "CHEEK_RY_FRAC_HEAD_RY",
    "CHEEK_RZ_FRAC_H",
    "CHEEK_X_FRAC_HEAD_RX",
    "CHEEK_Y_BIAS_FRAC_RY",
    "CHEEK_Z_MIX",
    "EYE_RX_FRAC_R",
    "EYE_RY_FRAC_R",
    "EYE_RZ_FRAC_R",
    "FACE_KIT_SKIP_BOUNDS",
    "FEATURE_FACE_Y_FRAC_RY",
    "HAIR_TIERS",
    "JAW_RX_FRAC_HEAD_RX",
    "JAW_RY_FRAC_HEAD_RY",
    "JAW_RZ_FRAC_H",
    "JAW_X_BULGE_ALLOW_M",
    "JAW_Y_BIAS_FRAC_RY",
    "JAW_Z_CENTER_FRAC_H",
    "LIP_RX_FRAC_H",
    "LIP_RY_FRAC_H",
    "LIP_RZ_FRAC_H",
    "NECKLINE_TIERS",
    "NOSE_RX_FRAC_H",
    "NOSE_RY_FRAC_H",
    "NOSE_RZ_FRAC_H",
    "NOSE_TIP_Y_FRAC_RY",
    "HeadBounds",
    "build_face_parts",
    "head_part_from_bounds",
    "resolve_head_bounds",
    "scale_head_bounds",
]
