"""Hand / foot / digit RECIPE primitives (track 0029).

Opt-in blockout-grade extremities parented to wrist/hand/ankle/heel/toe joints
or landmarks. Authoring only — not print-ready articulated digits, not boots
as law (Difficulty §12 / N6).

All names stay RECIPE_* (never HAND_*/FOOT_*/DIGIT_* prefixes).
Ankle mass labels must contain ank_foot (classifier -> ankle_bridge).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from meshops.proportion.blockout_recipe import RecipePart
    from meshops.proportion.body_template import TemplateAppliedPackage
    from meshops.proportion.models import ProportionReport
    from meshops.proportion.skeleton import BlockoutSkeleton

FingerTier = Literal["none", "mitten", "full"]
ToeTier = Literal["none", "wedge", "full"]

FINGER_TIERS: Final[frozenset[str]] = frozenset({"none", "mitten", "full"})
TOE_TIERS: Final[frozenset[str]] = frozenset({"none", "wedge", "full"})

# B11 / plan §7 freezes
FOOT_LEN_BASE_FRAC_H: Final[float] = 1.0 / 7.5
_FOOT_LEN_NO_TEMPLATE_FRAC: Final[float] = 0.15
_FOOT_WIDTH_FALLBACK_FRAC_H: Final[float] = 0.02
# 0054 sole thickness freezes (B1-B4)
SOLE_THICKNESS_FRAC_H: Final[float] = 0.025  # was _PLATE_THICKNESS_FRAC_H 0.02
SOLE_RZ_FRAC_OF_THICKNESS: Final[float] = 0.70  # was bare 0.55
SOLE_RZ_FLOOR_M: Final[float] = 0.016  # was 0.012
_HAND_LEN_FALLBACK_FRAC_H: Final[float] = 0.11
# Hand constants — 0048 bulk + 0064 palm pad / 0079 digit sausage (mitten fence unchanged)
_PALM_WIDTH_FRAC_HAND: Final[float] = 0.62
_PALM_THICKNESS_FRAC_HAND: Final[float] = 0.36  # 0064 B1 (was 0.30)
_PALM_PAD_RY_FRAC_TH: Final[float] = 0.78  # 0064 B2 (was 0.65)
_PALM_LEN_FRAC_HAND: Final[float] = 0.48
_MITTEN_LEN_FRAC_HAND: Final[float] = 0.50
_MITTEN_R_FRAC_PALM: Final[float] = 0.72  # fat mitt, not thin stick — DO NOT CHANGE
# 0079 hand digit sausage freezes (B1-B5, B16)
_FINGER_SEG_FRACS_HAND: Final[tuple[float, float, float]] = (0.25, 0.18, 0.12)  # sum 0.55
_FINGER_R_SCALES_SEG: Final[tuple[float, float, float]] = (1.00, 0.90, 0.82)
_FINGER_DIGIT_L_SCALE: Final[dict[str, float]] = {
    "index": 0.96,
    "middle": 1.00,
    "ring": 0.96,
    "pinky": 0.88,
}
_FINGER_DIGIT_R_SCALE: Final[dict[str, float]] = {
    "index": 0.94,
    "middle": 1.00,
    "ring": 0.96,
    "pinky": 0.86,
}
_THUMB_DISTAL_L_SCALE: Final[float] = 0.78
_THUMB_DISTAL_R_SCALE: Final[float] = 0.88
_FINGER_R_FRAC_PALM: Final[float] = 0.16
_FINGER_R_FLOOR_M: Final[float] = 0.006
_FINGER_R_CAP_VS_HALF_W: Final[float] = 0.55
_FINGER_SPLAY_FRAC_HALF_W: Final[float] = 1.95
_FINGER_MIN_CENTER_SPACING_VS_R: Final[float] = 2.0  # T12 groove law docs
_THUMB_SEG_FRAC_HAND: Final[float] = 1.0 / 6.0
_THUMB_R_SCALE_VS_FINGER: Final[float] = 1.25
_THUMB_BASE_LATERAL_FRAC_HALF_W: Final[float] = 0.95
_THUMB_OPPOSE_LATERAL: Final[float] = 0.45
_THUMB_PALM_PITCH: Final[float] = -0.55  # 0084 B6: face -Y (was +0.55 rear rake)
# Foot-friendly toe / sole fracs (0044 Phase 0 organic sole)
_TOE_WEDGE_LEN_FRAC: Final[float] = 0.42  # elongated front mass (not a ball on plate)
# 0054 full-toe bulk freezes (B5-B10, B15, B11)
TOE_R_FRAC_HALF_W: Final[float] = 0.36  # AI2 P2-1 must win on product hw~0.0263
TOE_R_FLOOR_M: Final[float] = 0.009
TOE_R_CAP_FRAC_HALF_W: Final[float] = 0.45
TOE_BIG_SCALE: Final[float] = 1.20
TOE_FULL_LEN_FRAC: Final[float] = 0.16  # 0072 B5 (was 0.26 stick-class)
TOE_BASE_NEST_FRAC: Final[float] = 0.35  # 0072 B6 nest INTO plate (+Y from front)
TOE_BALL_NEST_FRAC: Final[float] = 0.40  # 0075 B1 nest INTO ball (+Y from ball front)
TOE_TIP_PAST_FRAC: Final[float] = 0.55  # 0075 B3 (was 0.90 stick-class past plate)
TOE_TIP_MAX_PAST_M: Final[float] = 0.024  # 0075 B3 plate absolute tip budget (was 0.038)
TOE_TIP_MAX_PAST_FRAC: Final[float] = 0.12  # 0075 B3 plate proportional tip budget (was 0.15)
TOE_TIP_MAX_PAST_BALL_M: Final[float] = 0.028  # 0075 B2 ball absolute tip budget
TOE_TIP_MAX_PAST_BALL_FRAC: Final[float] = 0.12  # 0075 B2 ball proportional tip budget
TOE_TIP_PAD_SCALE: Final[float] = 1.15  # 0075 B5 tip pad mass vs digit r
TOE_SPLAY_FRAC_HALF_W: Final[float] = 1.25
TOE_MIN_CENTER_SPACING_VS_R: Final[float] = 1.0  # soft B15
TOE_WEDGE_RZ_FRAC_SOLE: Final[float] = 0.85
_BALL_SOFT_R_FRAC_FOOT: Final[float] = 0.14
BALL_SOFT_RY_FRAC_HALF_DEPTH: Final[float] = 0.32  # 0072 B8 (was bare 0.28)
# Rounded sole: ellipsoid foot_plate (not world-axis square box).
# Heel min-floor inside max(...) — rear pad primary (0044 B6-B8), not tower.
_HEEL_R_FRAC_FOOT: Final[float] = 0.18
_HEEL_BRIDGE_OVERLAP_FRAC: Final[float] = 0.35  # 0040 reuse (reach overlap concept)
# 0044 B1-B3 / B6-B8 visual mass freezes (+ 0072 B1-B3 / B10-B11 + 0080 full-figure)
FOOT_LEN_VISUAL_MIN_FRAC_H: Final[float] = 0.145  # 0080 B1 (was 0.13 / 0072)
FOOT_LEN_VISUAL_MAX_FRAC_H: Final[float] = 0.155  # 0080 B15 floor-induced anti-boat cap only
FOOT_LEN_MIN_VS_ANK_HW: Final[float] = 4.8
FOOT_LEN_MIN_VS_CALF_DIAM: Final[float] = 4.0  # 0080 B2 (was 1.55)
# 0080 half-width visual floors (never shrink; calf_b distal only)
FOOT_HW_MIN_FRAC_LEN: Final[float] = 0.16
FOOT_HW_MIN_VS_CALF_R: Final[float] = 1.20  # calf_b distal only
FOOT_HW_MIN_FRAC_H: Final[float] = 0.022
HEEL_REAR_Y_BIAS_FRAC_DEPTH: Final[float] = 0.10  # 0076 B3 (was 0.06 / 0072)
HEEL_REAR_OVERHANG_M: Final[float] = 0.012  # 0072 B3 rear tip clamp budget
HEEL_Z_FRAC_ANK: Final[float] = 0.42
HEEL_RZ_CAP_FRAC_ANK: Final[float] = 0.48
HEEL_RY_MIN_FRAC_DEPTH: Final[float] = 0.30  # 0072 B1 (was 0.42)
HEEL_RY_MIN_VS_RZ_FRAC: Final[float] = 0.70  # 0072 B1c (was bare 0.70)
HEEL_RY_MAX_FRAC_HALF_DEPTH: Final[float] = 0.34  # 0072 B11 composition accept
# 0056 ank/heel contact mass freezes (B1-B7, B13) + 0076 anti-ball / mild column
ANK_RY_FRAC_HALF_W: Final[float] = 1.22  # 0076 B1 anti-ball (was 1.45)
ANK_RY_FLOOR_M: Final[float] = 0.030  # 0076 B1 (was 0.036; product frac wins)
ANK_RZ_FRAC_HALF_W: Final[float] = 1.80  # 0076 B2 mild column (was 2.00)
ANK_RZ_FLOOR_M: Final[float] = 0.044  # 0076 B2 (was 0.048; product frac wins)
ANK_RZ_MIN_VS_CALF_B: Final[float] = 1.35
ANK_RZ_MAX_FRAC_ANK_Z: Final[float] = 0.60  # AI2 P2-2 ceiling
HEEL_CONTACT_OVERLAP_TARGET_M: Final[float] = 0.005  # B7 all three sites (0056 fence)
_FINGER_Y_BIAS_FRAC: Final[float] = 0.10  # hang and no-tip slight -Y (never primary -Y)
# Mirror constraints.AXIAL_DEPTH_MARGIN_M / skeleton._GLENOID_PLANE_MARGIN_M (0084 B21).
_FINGERTIP_PLANE_MARGIN_M: Final[float] = 0.02
_NEAR_ZERO: Final[float] = 1e-9
_FINGER_NAMES: Final[tuple[str, ...]] = ("index", "middle", "ring", "pinky")

PARENT_UNRESOLVED_MSG: Final[str] = (
    "parent_joint {role} unresolved — using landmark/template placement"
)


# ---------------------------------------------------------------------------
# Helpers (lazy blockout_recipe imports — B17)
# ---------------------------------------------------------------------------


def _assert_ank_foot_name(name: str, role: str) -> None:
    """B2 / AI2: ankle_bridge emit must contain ank_foot in the name."""
    if role == "ankle_bridge" and "ank_foot" not in name.lower():
        msg = f"ankle_bridge name must contain 'ank_foot' (got {name!r})"
        raise ValueError(msg)


def _parent_joint(
    preferred: str,
    fallbacks: list[str],
    skeleton: BlockoutSkeleton | None,
    *,
    side: str,
    role: str,
    messages: list[str],
) -> str | None:
    """Resolve parent_joint id; null + message if unresolved (B12)."""
    if skeleton is None:
        return None
    from meshops.proportion.blockout_recipe import _joints_map, _resolve_parent_joint_id

    joints = _joints_map(skeleton)
    pid = _resolve_parent_joint_id(preferred, fallbacks, joints, side=side)
    if pid is None:
        messages.append(PARENT_UNRESOLVED_MSG.format(role=role))
    return pid


def _lm_xyz3(
    report: ProportionReport,
    lm_id: str,
) -> list[float] | None:
    """Landmark [x,y,z] when x and z finite; y falls back to 0."""
    lm = report.landmarks_xyz.get(lm_id)
    if lm is None or lm.x_m is None or lm.z_m is None:
        return None
    y = float(lm.y_m) if lm.y_m is not None else 0.0
    return [float(lm.x_m), y, float(lm.z_m)]


def _joint_or_lm(
    report: ProportionReport,
    skeleton: BlockoutSkeleton | None,
    joint_id: str,
    lm_id: str | None = None,
) -> list[float] | None:
    """Prefer skeleton joint full xyz; else landmark (y may fallback 0)."""
    if skeleton is not None:
        from meshops.proportion.blockout_recipe import _joint_xyz, _joints_map

        j = _joints_map(skeleton).get(joint_id)
        xyz = _joint_xyz(j)
        if xyz is not None:
            return xyz
    return _lm_xyz3(report, lm_id or joint_id)


def _finite3(p: list[float] | None) -> bool:
    if p is None or len(p) < 3:
        return False
    return all(v == v for v in p[:3])


def _dist3(a: list[float], b: list[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n <= _NEAR_ZERO:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def _add(p: list[float], d: tuple[float, float, float], scale: float) -> list[float]:
    return [p[0] + d[0] * scale, p[1] + d[1] * scale, p[2] + d[2] * scale]


def _ellipsoid(
    name: str,
    role: str,
    center: list[float],
    rx: float,
    ry: float,
    rz: float,
    *,
    parent_joint: str | None = None,
    notes: str | None = None,
    half_depth_m: float | None = None,
    top_half_width_m: float | None = None,
    bottom_half_width_m: float | None = None,
    z_bottom_m: float | None = None,
    z_top_m: float | None = None,
) -> RecipePart:
    from meshops.proportion.blockout_recipe import RecipePart

    _assert_ank_foot_name(name, role)
    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind="ellipsoid",
        center=center,
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
        half_depth_m=half_depth_m,
        top_half_width_m=top_half_width_m,
        bottom_half_width_m=bottom_half_width_m,
        z_bottom_m=z_bottom_m,
        z_top_m=z_top_m,
        placement="full3d",
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
    parent_joint: str | None = None,
    notes: str | None = None,
) -> RecipePart:
    from meshops.proportion.blockout_recipe import RecipePart

    _assert_ank_foot_name(name, role)
    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind="capsule",
        p0=p0,
        p1=p1,
        radius_m=radius,
        placement="full3d",
        label=name,
        parent_joint=parent_joint,
        notes=notes,
    )


def _box(
    name: str,
    role: str,
    center: list[float],
    *,
    half_width: float,
    half_depth: float,
    z_bottom: float,
    z_top: float,
    parent_joint: str | None = None,
    notes: str | None = None,
) -> RecipePart:
    from meshops.proportion.blockout_recipe import RecipePart

    _assert_ank_foot_name(name, role)
    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind="box",
        center=center,
        top_half_width_m=half_width,
        bottom_half_width_m=half_width,
        half_depth_m=half_depth,
        z_bottom_m=z_bottom,
        z_top_m=z_top,
        placement="full3d",
        label=name,
        parent_joint=parent_joint,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Foot length / width ladders (B11)
# ---------------------------------------------------------------------------


def resolve_foot_length_m(
    *,
    heel: list[float] | None,
    toe: list[float] | None,
    template_applied: TemplateAppliedPackage | None,
    height_m: float | None,
    messages: list[str],
    side: str,
) -> float | None:
    """B11 foot length ladder — never invent stature_foot_prior."""
    if (
        heel is not None
        and toe is not None
        and _finite3(heel)
        and _finite3(toe)
        and _dist3(heel, toe) > _NEAR_ZERO
    ):
        return _dist3(heel, toe)

    if template_applied is not None and height_m is not None and height_m > 0:
        scale = float(template_applied.constants.foot_len_scale)
        fl = scale * FOOT_LEN_BASE_FRAC_H * float(height_m)
        messages.append(
            f"foot_{side}: length from template foot_len_scale={scale} "
            f"* {FOOT_LEN_BASE_FRAC_H:.4f} * H"
        )
        return fl

    if height_m is not None and height_m > 0:
        fl = _FOOT_LEN_NO_TEMPLATE_FRAC * float(height_m)
        messages.append(
            f"foot_{side}: length fallback {_FOOT_LEN_NO_TEMPLATE_FRAC}*H (no heel↔toe / template)"
        )
        return fl

    messages.append(f"foot_{side}: length unresolved — side skipped")
    return None


def resolve_foot_half_width_m(
    report: ProportionReport,
    *,
    template_applied: TemplateAppliedPackage | None,
    height_m: float | None,
    side: str,
    messages: list[str],
) -> float:
    """Foot half-width = ankle half-width ladder (B11)."""
    from meshops.proportion.blockout_recipe import (
        _half_width_from_diameter,
        _resolve_diameter,
    )

    # Ankle diameter from report diameters when present
    for band in (f"ank_foot_{side}", f"ankle_{side}", "ank_foot", "ankle"):
        diam = _resolve_diameter(report.diameters, band)
        if diam is not None:
            hw = _half_width_from_diameter(diam)
            if hw is not None and hw > 0:
                return float(hw)

    if template_applied is not None:
        c = template_applied.constants
        if c.ank_foot_r_m is not None and float(c.ank_foot_r_m) > 0:
            return float(c.ank_foot_r_m)
        if c.ank_foot_r_frac is not None and height_m is not None:
            return float(c.ank_foot_r_frac) * float(height_m)

    if height_m is not None and height_m > 0:
        return _FOOT_WIDTH_FALLBACK_FRAC_H * float(height_m)

    messages.append(f"foot_{side}: width fallback 0.03 m (no H/template/diameter)")
    return 0.03


def apply_foot_length_visual_floor(
    foot_len: float,
    *,
    height_m: float | None,
    half_width: float,
    calf_distal_r: float | None,
    messages: list[str],
    side: str,
) -> float:
    """0044/0080: mannequin visual length floor; never shrinks measured/template.

    Floor sources (skip when unavailable): stature FOOT_LEN_VISUAL_MIN_FRAC_H·H
    (0.145·H), ank half-width 4.8·hw, calf distal diam 4.0·(2·calf_r). When H is
    present, each floor candidate is capped at FOOT_LEN_VISUAL_MAX_FRAC_H·H
    (0.155·H, B15 anti-boat) *before* max — does not shrink measured/template
    already above the cap. Template length is not a floor when measured exists
    (B18).
    """
    floors: list[tuple[str, float]] = []
    max_cap: float | None = None
    if height_m is not None and height_m > 0:
        max_cap = FOOT_LEN_VISUAL_MAX_FRAC_H * float(height_m)

    def _cap(raw: float) -> float:
        if max_cap is None:
            return raw
        return min(raw, max_cap)

    if height_m is not None and height_m > 0:
        floors.append(("stature", _cap(FOOT_LEN_VISUAL_MIN_FRAC_H * float(height_m))))
    if half_width > 0:
        floors.append(("ank_hw", _cap(FOOT_LEN_MIN_VS_ANK_HW * float(half_width))))
    if calf_distal_r is not None and calf_distal_r > 0:
        diam = 2.0 * float(calf_distal_r)
        floors.append(("calf_diam", _cap(FOOT_LEN_MIN_VS_CALF_DIAM * diam)))
    if not floors:
        return foot_len
    floor_val = max(v for _, v in floors)
    winner = max(floors, key=lambda t: t[1])[0]
    if foot_len + 1e-12 < floor_val:
        messages.append(
            f"foot_{side}: length visual floor {foot_len:.4f}->{floor_val:.4f} m ({winner})"
        )
        return floor_val
    return foot_len


def apply_foot_half_width_visual_floor(
    half_width: float,
    *,
    foot_len: float,
    height_m: float | None,
    calf_distal_r: float | None,
    messages: list[str],
    side: str,
) -> float:
    """0080 B4/B5: mannequin half-width visual floor; never shrinks measured/template.

    Floor sources (skip when unavailable): foot_len * FOOT_HW_MIN_FRAC_LEN (0.16),
    calf_b distal r * FOOT_HW_MIN_VS_CALF_R (1.20), stature FOOT_HW_MIN_FRAC_H*H
    (0.022*H). Skips foot_len floor when foot_len <= 0. Applied *after* length
    floor only — never re-runs length.
    """
    floors: list[tuple[str, float]] = []
    if foot_len > 0:
        floors.append(("foot_len", FOOT_HW_MIN_FRAC_LEN * float(foot_len)))
    if calf_distal_r is not None and calf_distal_r > 0:
        floors.append(("calf_r", FOOT_HW_MIN_VS_CALF_R * float(calf_distal_r)))
    if height_m is not None and height_m > 0:
        floors.append(("stature", FOOT_HW_MIN_FRAC_H * float(height_m)))
    if not floors:
        return half_width
    floor_val = max(v for _, v in floors)
    winner = max(floors, key=lambda t: t[1])[0]
    if half_width + 1e-12 < floor_val:
        messages.append(
            f"foot_{side}: width visual floor {half_width:.4f}->{floor_val:.4f} m ({winner})"
        )
        return floor_val
    return half_width


def _calf_distal_r_from_parts(
    existing_parts: list[RecipePart] | None,
    side: str,
) -> float | None:
    """B5: calf distal r from RECIPE_calf_b_{side}; prefer rx else mean of finite radii."""
    if not existing_parts:
        return None
    name = f"RECIPE_calf_b_{side}"
    part = next((p for p in existing_parts if p.name == name), None)
    if part is None:
        return None
    if part.rx_m is not None and float(part.rx_m) > 0:
        return float(part.rx_m)
    vals = [
        float(v)
        for v in (part.rx_m, part.ry_m, part.rz_m)
        if v is not None and float(v) == float(v) and float(v) > 0
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Finger axis (B7) — exported for tests
# ---------------------------------------------------------------------------


def _fingertip_y_trusted(wrist_y: float | None, tip_y: float | None) -> bool:
    """0084 B19: True only for a face-forward, off-plane measured tip.

    False when either Y is missing/non-finite, tip is plane-class
    (``abs(y) <= 0.02`` / invent 0), or wrist→tip would rear-rake (ΔY >= 0).
    """
    if wrist_y is None or tip_y is None:
        return False
    if not (math.isfinite(wrist_y) and math.isfinite(tip_y)):
        return False
    return tip_y < -_FINGERTIP_PLANE_MARGIN_M and (tip_y - wrist_y) < 0.0


def finger_primary_axis(
    wrist: list[float] | None,
    fingertip: list[float] | None,
    *,
    hand_len: float,
) -> tuple[float, float, float]:
    """Primary finger/mitten axis: wrist->tip when tip Y trusted; else hang -Z.

    Hang and no-tip both use ``_FINGER_Y_BIAS_FRAC`` as slight -Y on
    ``(0, -bias, -1)``. Never returns ``axis[1] > 0``.
    """
    if (
        wrist is not None
        and fingertip is not None
        and _finite3(wrist)
        and _finite3(fingertip)
        and _fingertip_y_trusted(wrist[1], fingertip[1])
    ):
        raw = (
            fingertip[0] - wrist[0],
            fingertip[1] - wrist[1],
            fingertip[2] - wrist[2],
        )
        n = _normalize(raw)
        if n is not None:
            if n[1] > 0.0:
                n2 = _normalize((n[0], 0.0, n[2]))
                return n2 if n2 is not None else (0.0, 0.0, -1.0)
            return n

    # Hang / no-tip: primary -Z; slight -Y (B2 / B7, <=10% hand_len)
    bias = _FINGER_Y_BIAS_FRAC * max(hand_len, 0.01)
    raw_def = (0.0, -bias, -1.0)
    n = _normalize(raw_def)
    return n if n is not None else (0.0, 0.0, -1.0)


# ---------------------------------------------------------------------------
# Feet
# ---------------------------------------------------------------------------


def build_foot_parts(
    report: ProportionReport,
    *,
    skeleton: BlockoutSkeleton | None = None,
    template_applied: TemplateAppliedPackage | None = None,
    height_m: float | None = None,
    toes: ToeTier = "wedge",
    messages: list[str] | None = None,
    existing_parts: list[RecipePart] | None = None,
) -> list[RecipePart]:
    """Emit L/R foot plate + heel + ank_foot (+ toes per tier). RECIPE_* only.

    ``existing_parts`` supplies already-emitted limb parts (e.g. calf_b) for the
    0044 visual length floor (B5).
    """
    msgs = messages if messages is not None else []
    if toes not in TOE_TIERS:
        msg = f"toes must be one of {sorted(TOE_TIERS)} (got {toes!r})"
        raise ValueError(msg)

    h = (
        float(height_m)
        if height_m is not None
        else (float(report.height_m) if report.height_m is not None else None)
    )
    parts: list[RecipePart] = []
    for side in ("l", "r"):
        parts.extend(
            _build_foot_side(
                report,
                side=side,
                skeleton=skeleton,
                template_applied=template_applied,
                height_m=h,
                toes=toes,
                messages=msgs,
                existing_parts=existing_parts,
            )
        )
    return parts


def _build_foot_side(
    report: ProportionReport,
    *,
    side: str,
    skeleton: BlockoutSkeleton | None,
    template_applied: TemplateAppliedPackage | None,
    height_m: float | None,
    toes: ToeTier,
    messages: list[str],
    existing_parts: list[RecipePart] | None = None,
) -> list[RecipePart]:
    ankle = _joint_or_lm(report, skeleton, f"ankle_{side}")
    heel = _joint_or_lm(report, skeleton, f"heel_{side}")
    toe = _joint_or_lm(report, skeleton, f"toe_{side}")

    # R1: skip side if missing ankle AND heel AND toe (no sole_* id)
    if ankle is None and heel is None and toe is None:
        messages.append(f"foot_{side} skipped: missing ankle, heel, and toe joints/landmarks")
        return []

    foot_len = resolve_foot_length_m(
        heel=heel,
        toe=toe,
        template_applied=template_applied,
        height_m=height_m,
        messages=messages,
        side=side,
    )
    if foot_len is None or foot_len <= _NEAR_ZERO:
        return []

    half_width = resolve_foot_half_width_m(
        report,
        template_applied=template_applied,
        height_m=height_m,
        side=side,
        messages=messages,
    )

    # 0044/0080: length floor then width floor after resolve (never shrinks; no re-length)
    calf_r = _calf_distal_r_from_parts(existing_parts, side)
    foot_len = apply_foot_length_visual_floor(
        foot_len,
        height_m=height_m,
        half_width=half_width,
        calf_distal_r=calf_r,
        messages=messages,
        side=side,
    )
    half_width = apply_foot_half_width_visual_floor(
        half_width,
        foot_len=foot_len,
        height_m=height_m,
        calf_distal_r=calf_r,
        messages=messages,
        side=side,
    )

    # B17: start from floored length; max with measured heel↔toe half-span
    half_depth = foot_len / 2.0
    thickness = (
        SOLE_THICKNESS_FRAC_H * float(height_m)
        if height_m is not None and height_m > 0
        else 0.035  # slight bump vs old 0.03 no-H fallback
    )

    # Plate center X from ankle (else mean heel/toe, else 0 with side offset)
    if ankle is not None:
        plate_x = float(ankle[0])
    elif heel is not None and toe is not None:
        plate_x = 0.5 * (float(heel[0]) + float(toe[0]))
    elif heel is not None:
        plate_x = float(heel[0])
    elif toe is not None:
        plate_x = float(toe[0])
    else:
        plate_x = -0.08 if side == "l" else 0.08

    # Plate Y: mid heel↔toe when both known (frame: heel +Y, toe -Y); else ankle Y; else 0
    if heel is not None and toe is not None and abs(float(heel[1]) - float(toe[1])) > _NEAR_ZERO:
        hy = float(heel[1])
        ty = float(toe[1])
        # Ensure heel is the more +Y landmark for frame signs
        if hy < ty:
            hy, ty = ty, hy
        plate_y = 0.5 * (hy + ty)
        measured_half = abs(hy - ty) / 2.0
        if measured_half > _NEAR_ZERO:
            # Never overwrite floor with measured alone (P1-1)
            half_depth = max(measured_half, half_depth)
            foot_len = max(foot_len, 2.0 * half_depth)
    elif ankle is not None:
        plate_y = float(ankle[1])
    else:
        plate_y = 0.0

    # Organic sole ellipsoid: center_z = sole_rz, z_bottom=0 -> z_top = 2xsole_rz (R5a2 / B4)
    sole_rz = max(thickness * SOLE_RZ_FRAC_OF_THICKNESS, SOLE_RZ_FLOOR_M)
    z_bottom = 0.0
    z_top = 2.0 * sole_rz
    sole_cz = sole_rz
    messages.append(
        f"foot_{side}: sole thickness scale sole_rz={sole_rz:.4f} "
        f"z_top={z_top:.4f} (frac_h={SOLE_THICKNESS_FRAC_H})"
    )

    # Heel mass + ankle: rear-third stack; heel rear-biased pad (0044 B6)
    # Mid of rear third: plate_y + (2/3)*half_depth (toward heel +Y).
    stack_y = plate_y + (2.0 / 3.0) * half_depth
    ank_y = stack_y
    heel_y = stack_y + HEEL_REAR_Y_BIAS_FRAC_DEPTH * half_depth

    # Ankle Z: use joint when clearly above the plate (real ankle height).
    if ankle is not None and ankle[2] == ankle[2] and float(ankle[2]) > z_top:
        ank_z = float(ankle[2])
    else:
        ank_z = z_top + half_width * 2.2  # rest above plate top

    # 0056: ank contact mass — rx stays half_width (C_foot_width);
    # ry/rz freezes + calf floor + ceiling
    ank_rx = half_width
    ank_ry = max(half_width * ANK_RY_FRAC_HALF_W, ANK_RY_FLOOR_M)
    ank_rz = max(half_width * ANK_RZ_FRAC_HALF_W, ANK_RZ_FLOOR_M)
    if calf_r is not None and calf_r > 0:
        ank_rz = max(ank_rz, float(calf_r) * ANK_RZ_MIN_VS_CALF_B)
    ank_rz = min(ank_rz, ANK_RZ_MAX_FRAC_ANK_Z * float(ank_z))  # B13 after floors
    messages.append(
        f"foot_{side}: ank contact mass ry={ank_ry:.4f} rz={ank_rz:.4f} "
        f"(frac_rz={ANK_RZ_FRAC_HALF_W})"
    )

    # Heel rear pad (0044 B6-B8 + 0056 B7 contact): emit overlap on ALL three sites
    ov = HEEL_CONTACT_OVERLAP_TARGET_M
    heel_z = HEEL_Z_FRAC_ANK * ank_z
    reach_need = (ank_z - ank_rz) + ov - heel_z
    heel_rz = max(reach_need, _HEEL_R_FRAC_FOOT * foot_len * 0.55, half_width * 0.9)
    rz_cap = HEEL_RZ_CAP_FRAC_ANK * ank_z
    if heel_rz > rz_cap + _NEAR_ZERO:
        heel_z = min(heel_z, (ank_z - ank_rz) + ov - rz_cap)
        heel_z = max(heel_z, z_top * 0.35)
        heel_rz = min(heel_rz, rz_cap)
        still_need = (ank_z - ank_rz) + ov - heel_z
        if still_need > heel_rz + _NEAR_ZERO:
            heel_rz = still_need
            messages.append(
                f"foot_{side}: heel_rz cap lose for contact (rz={heel_rz:.4f} > cap={rz_cap:.4f})"
            )
    # 0072 B1/B1b/B1c: min floors only — no half_depth*0.38 (defeats B1)
    heel_ry = max(
        HEEL_RY_MIN_FRAC_DEPTH * half_depth,
        heel_rz * HEEL_RY_MIN_VS_RZ_FRAC,
    )
    heel_rx = max(half_width * 1.05, ank_rx * 0.95)
    # Strict heel_z < ank_z
    if heel_z >= ank_z - _NEAR_ZERO:
        heel_z = min(ank_z * 0.55, ank_z - _NEAR_ZERO)
        if heel_z < z_top * 0.35:
            heel_z = min(max(z_top * 0.35, ank_z * 0.35), ank_z - _NEAR_ZERO)

    # 0072 B3: clamp heel rear tip so pad does not hang past plate rear + overhang
    plate_rear_y = plate_y + half_depth
    heel_y = min(heel_y, plate_rear_y + HEEL_REAR_OVERHANG_M - heel_ry)
    heel_rear_tip = heel_y + heel_ry
    messages.append(
        f"foot_{side}: heel proportion ry={heel_ry:.4f} rear_tip={heel_rear_tip:.4f} "
        f"(plate_rear={plate_rear_y:.4f} overhang={HEEL_REAR_OVERHANG_M})"
    )
    # 0076 B9: separate heel/ank proportion telemetry (do not merge into heel line)
    messages.append(
        f"foot_{side}: heel/ank proportion dy={heel_y - ank_y:.4f} ank_ry_rx={ank_ry / ank_rx:.3f}"
    )

    # Toe wedge: **in front of** the plate (-Y past front edge), elongated + flat on sole.
    # Skeleton heel/toe often inherit ankle Z (estimated) — never use that for sole masses.
    toe_ry = max(_TOE_WEDGE_LEN_FRAC * foot_len * 0.5, half_depth * 0.45)
    toe_rx = half_width * 1.00  # was 0.95
    toe_rz = max(sole_rz * TOE_WEDGE_RZ_FRAC_SOLE, half_width * 0.32, 0.012)
    toe_z = sole_cz + sole_rz * 0.1  # sole-class Z (not perched mid-ball)
    plate_front_y = plate_y - half_depth  # toes -Y edge of foot_plate
    # Center past the front edge so the bulk of the ellipsoid is *ahead* of the plate.
    toe_y = plate_front_y - toe_ry * 0.55

    # parent_joint
    pj_plate = _parent_joint(
        f"ankle_{side}",
        [f"heel_{side}"],
        skeleton,
        side=side,
        role="foot_plate",
        messages=messages,
    )
    pj_heel = _parent_joint(
        f"heel_{side}",
        [f"ankle_{side}"],
        skeleton,
        side=side,
        role="heel",
        messages=messages,
    )
    pj_ank = _parent_joint(
        f"ankle_{side}",
        [],
        skeleton,
        side=side,
        role="ankle_bridge",
        messages=messages,
    )
    pj_toe = _parent_joint(
        f"toe_{side}",
        [f"ankle_{side}"],
        skeleton,
        side=side,
        role="toe_soft",
        messages=messages,
    )

    out: list[RecipePart] = []

    # foot_plate = rounded sole ellipsoid (not square box — canvas read as brick).
    # Keep half_depth_m / z_top_m / half_width for C_ankle_over_heel / C_foot_width / sole-Z.
    plate_name = f"RECIPE_foot_plate_{side}"
    out.append(
        _ellipsoid(
            plate_name,
            "foot_plate",
            [plate_x, plate_y, sole_cz],
            half_width * 1.05,  # rx width
            half_depth,  # ry heel->toe
            sole_rz,  # rz thin sole
            parent_joint=pj_plate,
            half_depth_m=half_depth,
            top_half_width_m=half_width,
            bottom_half_width_m=half_width,
            z_bottom_m=z_bottom,
            z_top_m=z_top,
            notes="organic sole ellipsoid (not box)",
        )
    )

    # Heel rear pad (+Y of stack) — reaches ank_foot without tower (0044)
    heel_name = f"RECIPE_heel_{side}"
    out.append(
        _ellipsoid(
            heel_name,
            "heel",
            [plate_x, heel_y, heel_z],
            heel_rx,
            heel_ry,
            heel_rz,
            parent_joint=pj_heel,
        )
    )

    # Arch always with --feet (R5b2), including toes=none. Role stays ball_soft.
    arch_y = plate_y + half_depth * 0.08
    arch_rx = half_width * 0.92
    arch_ry = half_depth * 0.38  # elongated along foot
    arch_rz = max(sole_rz * 1.15, half_width * 0.28)  # low pad, not a perched sphere
    arch_z = sole_cz + sole_rz * 0.25  # mostly embedded; slight instep rise only
    out.append(
        _ellipsoid(
            f"RECIPE_arch_soft_{side}",
            "ball_soft",
            [plate_x, arch_y, arch_z],
            arch_rx,
            arch_ry,
            arch_rz,
            parent_joint=pj_ank,
        )
    )

    # ank_foot — name MUST contain ank_foot (B2)
    ank_name = f"RECIPE_ank_foot_{side}"
    _assert_ank_foot_name(ank_name, "ankle_bridge")
    out.append(
        _ellipsoid(
            ank_name,
            "ankle_bridge",
            [plate_x, ank_y, ank_z],
            ank_rx,
            ank_ry,
            ank_rz,
            parent_joint=pj_ank,
        )
    )

    if toes == "none":
        return out

    # Ball of foot (forefoot pad) — toes ≠ none; in sole, not stacked on top
    ball_y = plate_y - (1.0 / 3.0) * half_depth
    ball_rx = half_width * 0.95
    ball_ry = max(
        _BALL_SOFT_R_FRAC_FOOT * foot_len * 1.1,
        BALL_SOFT_RY_FRAC_HALF_DEPTH * half_depth,
    )
    ball_rz = max(sole_rz * 1.1, half_width * 0.22)
    ball_z = sole_cz + sole_rz * 0.2
    out.append(
        _ellipsoid(
            f"RECIPE_ball_soft_{side}",
            "ball_soft",
            [plate_x, ball_y, ball_z],
            ball_rx,
            ball_ry,
            ball_rz,
            parent_joint=pj_ank,
        )
    )

    if toes == "wedge":
        # Elongated toe mass past plate front (-Y), flat sole wedge
        out.append(
            _ellipsoid(
                f"RECIPE_toe_soft_{side}",
                "toe_soft",
                [plate_x, toe_y, toe_z],
                toe_rx,
                toe_ry,
                toe_rz,
                parent_joint=pj_toe,
            )
        )
        return out

    # toes == full: 5 toe capsules + tip pads (0054 r + 0072 len + 0075 nest/tip mass)
    toe_len = TOE_FULL_LEN_FRAC * foot_len
    base_r = min(
        max(TOE_R_FRAC_HALF_W * half_width, TOE_R_FLOOR_M),
        TOE_R_CAP_FRAC_HALF_W * half_width,
    )
    splay = half_width * TOE_SPLAY_FRAC_HALF_W
    # B1: ball-relative base nest wins over plate-only nest
    ball_front_y = ball_y - ball_ry
    plate_nest_y = plate_front_y + TOE_BASE_NEST_FRAC * toe_len
    ball_nest_y = ball_front_y + TOE_BALL_NEST_FRAC * toe_len
    base_y = max(plate_nest_y, ball_nest_y)
    # B3 raw tip + B2 dual rear-only clamp past ball/plate + inversion guard
    tip_y = plate_front_y - TOE_TIP_PAST_FRAC * toe_len
    ball_budget = min(TOE_TIP_MAX_PAST_BALL_M, TOE_TIP_MAX_PAST_BALL_FRAC * foot_len)
    plate_budget = min(TOE_TIP_MAX_PAST_M, TOE_TIP_MAX_PAST_FRAC * foot_len)
    tip_y = max(tip_y, ball_front_y - ball_budget, plate_front_y - plate_budget)
    tip_y = min(tip_y, base_y - 1e-6)
    tip_past_ball = ball_front_y - tip_y
    tip_past_plate = plate_front_y - tip_y
    messages.append(
        f"foot_{side}: toe bulk full r={base_r:.4f} "
        f"(frac_hw={TOE_R_FRAC_HALF_W} floor={TOE_R_FLOOR_M})"
    )
    messages.append(
        f"foot_{side}: toe tip mass tip_past_ball={tip_past_ball:.4f} "
        f"tip_past_plate={tip_past_plate:.4f} base_y={base_y:.4f} "
        f"ball_front={ball_front_y:.4f}"
    )
    tip_z = toe_z * 0.85
    for i in range(1, 6):
        # 1=medial ... 5=lateral; sign by side
        t = (i - 3) / 3.0  # -2/3 ... +2/3
        dx = t * splay
        if side == "l":
            dx = -dx  # mirror splay
        r_i = base_r * (TOE_BIG_SCALE if i == 1 else 1.0)
        r_i = min(r_i, TOE_R_CAP_FRAC_HALF_W * half_width)  # cap big toe too
        base = [plate_x + dx, base_y, toe_z]
        tip = [plate_x + dx, tip_y, tip_z]
        out.append(
            _capsule(
                f"RECIPE_toe_{i}_{side}",
                "toe_soft",
                base,
                tip,
                r_i,
                parent_joint=pj_toe,
            )
        )
        # B5: tip pad mass — sole-class ellipsoid at capsule tip (no dual-radius schema)
        r_pad = TOE_TIP_PAD_SCALE * r_i
        r_pad = min(r_pad, TOE_R_CAP_FRAC_HALF_W * half_width)  # AI2 P3-6
        rz_pad = min(r_pad, max(sole_rz * 0.85, r_i))
        out.append(
            _ellipsoid(
                f"RECIPE_toe_tip_{i}_{side}",
                "toe_soft",
                [plate_x + dx, tip_y, tip_z],
                r_pad,
                r_pad,
                rz_pad,
                parent_joint=pj_toe,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Hands
# ---------------------------------------------------------------------------


def build_hand_parts(
    report: ProportionReport,
    *,
    skeleton: BlockoutSkeleton | None = None,
    height_m: float | None = None,
    fingers: FingerTier = "mitten",
    messages: list[str] | None = None,
) -> list[RecipePart]:
    """Emit L/R palm + finger kit per tier. RECIPE_* only."""
    msgs = messages if messages is not None else []
    if fingers not in FINGER_TIERS:
        msg = f"fingers must be one of {sorted(FINGER_TIERS)} (got {fingers!r})"
        raise ValueError(msg)

    h = (
        float(height_m)
        if height_m is not None
        else (float(report.height_m) if report.height_m is not None else None)
    )
    parts: list[RecipePart] = []
    for side in ("l", "r"):
        parts.extend(
            _build_hand_side(
                report,
                side=side,
                skeleton=skeleton,
                height_m=h,
                fingers=fingers,
                messages=msgs,
            )
        )
    # B10 / 0079: bulk + taper messages once when full digits emitted
    if fingers == "full" and parts:
        msgs.append(
            f"hand bulk: full digits r_frac={_FINGER_R_FRAC_PALM} "
            f"palm_w={_PALM_WIDTH_FRAC_HAND} splay={_FINGER_SPLAY_FRAC_HALF_W} "
            f"palm_th={_PALM_THICKNESS_FRAC_HAND} pad_ry={_PALM_PAD_RY_FRAC_TH} "
            f"segs={_FINGER_SEG_FRACS_HAND} anti-stick"
        )
        msgs.append(
            f"hand taper: r_scales={_FINGER_R_SCALES_SEG} "
            f"digit_L={dict(_FINGER_DIGIT_L_SCALE)} digit_R={dict(_FINGER_DIGIT_R_SCALE)} "
            f"thumb_distal={_THUMB_DISTAL_L_SCALE}/{_THUMB_DISTAL_R_SCALE} anti-sausage"
        )
    return parts


def _build_hand_side(
    report: ProportionReport,
    *,
    side: str,
    skeleton: BlockoutSkeleton | None,
    height_m: float | None,
    fingers: FingerTier,
    messages: list[str],
) -> list[RecipePart]:
    wrist = _joint_or_lm(report, skeleton, f"wrist_{side}")
    hand = _joint_or_lm(report, skeleton, f"hand_{side}")
    tip = _joint_or_lm(report, skeleton, f"fingertip_{side}")

    # Hand estimate: hand joint or mid wrist->tip or tip alone
    hand_est = hand
    if hand_est is None and wrist is not None and tip is not None:
        hand_est = [
            0.5 * (wrist[0] + tip[0]),
            0.5 * (wrist[1] + tip[1]),
            0.5 * (wrist[2] + tip[2]),
        ]
    elif hand_est is None and tip is not None:
        hand_est = list(tip)

    # R1: skip if missing both wrist and any hand estimate
    if wrist is None and hand_est is None:
        messages.append(
            f"hand_{side} skipped: missing wrist and hand estimate (need wrist or fingertip/hand)"
        )
        return []

    # Hand length
    if wrist is not None and tip is not None and _dist3(wrist, tip) > _NEAR_ZERO:
        hand_len = _dist3(wrist, tip)
    elif height_m is not None and height_m > 0:
        hand_len = _HAND_LEN_FALLBACK_FRAC_H * float(height_m)
        messages.append(f"hand_{side}: length fallback {_HAND_LEN_FALLBACK_FRAC_H}*H")
    else:
        hand_len = 0.18
        messages.append(f"hand_{side}: length fallback 0.18 m (no H)")

    axis = finger_primary_axis(wrist, tip, hand_len=hand_len)
    tip_trusted = (
        wrist is not None
        and tip is not None
        and _finite3(wrist)
        and _finite3(tip)
        and _fingertip_y_trusted(wrist[1], tip[1])
    )
    axis_mode = "wrist_tip" if tip_trusted else "hang"

    # Palm center: hand joint or mid wrist->tip or wrist + 0.35*hand_len along axis
    if hand is not None:
        palm_c = list(hand)
    elif wrist is not None and tip is not None:
        palm_c = [
            0.5 * (wrist[0] + tip[0]),
            0.5 * (wrist[1] + tip[1]),
            0.5 * (wrist[2] + tip[2]),
        ]
    elif wrist is not None:
        palm_c = _add(wrist, axis, 0.35 * hand_len)
    else:
        assert hand_est is not None
        palm_c = list(hand_est)

    # 0084 B4: after the whole ladder, untrusted tip → palm Y = wrist Y (keep X/Z).
    if wrist is not None and _finite3(wrist) and not tip_trusted:
        palm_c[1] = wrist[1]

    palm_w = _PALM_WIDTH_FRAC_HAND * hand_len
    palm_th = _PALM_THICKNESS_FRAC_HAND * hand_len
    palm_len = _PALM_LEN_FRAC_HAND * hand_len  # along finger axis (full extent)

    pj_palm = _parent_joint(
        f"hand_{side}",
        [f"wrist_{side}"],
        skeleton,
        side=side,
        role="palm",
        messages=messages,
    )
    pj_digit = _parent_joint(
        f"hand_{side}",
        [f"wrist_{side}"],
        skeleton,
        side=side,
        role="finger_soft",
        messages=messages,
    )

    out: list[RecipePart] = []

    # Ellipsoid palm (never world-axis box — that read as "cube with a stick").
    # Width on X; thickness on Y; length along hang Z when axis is primarily -Z.
    palm_name = f"RECIPE_palm_{side}"
    half_w = palm_w / 2.0
    if abs(axis[2]) >= abs(axis[1]):
        # A-pose hang: flattened hand pad
        palm_rx = max(half_w, 0.02)
        palm_ry = max(palm_th * _PALM_PAD_RY_FRAC_TH, 0.012)
        palm_rz = max(palm_len * 0.5, 0.025)
    else:
        # Tip-directed (often -Y): length along Y; thickness on Z
        palm_rx = max(half_w, 0.02)
        palm_ry = max(palm_len * 0.5, 0.025)
        palm_rz = max(palm_th * _PALM_PAD_RY_FRAC_TH, 0.012)
    out.append(
        _ellipsoid(
            palm_name,
            "palm",
            palm_c,
            palm_rx,
            palm_ry,
            palm_rz,
            parent_joint=pj_palm,
        )
    )

    if fingers == "none":
        return out

    if fingers in ("full", "mitten"):
        messages.append(f"hand hang: axis={axis_mode} pitch={_THUMB_PALM_PITCH} anti-rake")

    if fingers == "mitten":
        # Fat mitten ellipsoid overlapping palm (not a thin stick capsule)
        mitten_len = _MITTEN_LEN_FRAC_HAND * hand_len
        mitt_c = _add(palm_c, axis, 0.28 * hand_len)
        mitt_r = max(half_w * _MITTEN_R_FRAC_PALM, 0.014)
        if abs(axis[2]) >= abs(axis[1]):
            mitt_rx, mitt_ry, mitt_rz = mitt_r, mitt_r * 0.75, mitten_len * 0.5
        else:
            mitt_rx, mitt_ry, mitt_rz = mitt_r, mitten_len * 0.5, mitt_r * 0.75
        out.append(
            _ellipsoid(
                f"RECIPE_finger_mitten_{side}",
                "finger_soft",
                mitt_c,
                mitt_rx,
                mitt_ry,
                mitt_rz,
                parent_joint=pj_digit,
            )
        )
        return out

    # fingers == full: 4 fingers x 3 capsules + thumb x 2 (0048 bulk + 0079 sausage)
    fr = min(
        max(_FINGER_R_FRAC_PALM * palm_w, _FINGER_R_FLOOR_M),
        _FINGER_R_CAP_VS_HALF_W * half_w,
    )
    # Lateral splay in X (B12: scale with bulk so grooves stay visible)
    splay = half_w * _FINGER_SPLAY_FRAC_HALF_W
    # Per-digit L/R scales + per-seg r taper; B16 post-scale r floor
    for fi, fname in enumerate(_FINGER_NAMES):
        digit_L = _FINGER_DIGIT_L_SCALE[fname]
        digit_R = _FINGER_DIGIT_R_SCALE[fname]
        t = (fi - 1.5) / 1.5  # -1 ... +1-ish
        # Mirror lateral splay: left flips sign so fingers fan correctly in +X world
        dx = (-t if side == "l" else t) * splay * 0.5
        base = _add([palm_c[0] + dx, palm_c[1], palm_c[2]], axis, 0.12 * hand_len)
        along = 0.0
        for si in range(3):
            seg_l = _FINGER_SEG_FRACS_HAND[si] * hand_len * digit_L
            r = fr * _FINGER_R_SCALES_SEG[si] * digit_R
            r = max(r, _FINGER_R_FLOOR_M)  # B16 post-scale floor
            p0 = _add(base, axis, along)
            p1 = _add(base, axis, along + seg_l)
            along += seg_l
            out.append(
                _capsule(
                    f"RECIPE_finger_{fname}_{si}_{side}",
                    "finger_soft",
                    p0,
                    p1,
                    r,
                    parent_joint=pj_digit,
                )
            )

    # Thumb: 2 segs, bulk + distal taper (B5) + lateral base + palm-plane pitch
    thumb_seg0 = _THUMB_SEG_FRAC_HAND * hand_len
    thumb_seg1 = thumb_seg0 * _THUMB_DISTAL_L_SCALE
    thumb_r0 = max(fr * _THUMB_R_SCALE_VS_FINGER, _FINGER_R_FLOOR_M)
    thumb_r1 = max(thumb_r0 * _THUMB_DISTAL_R_SCALE, _FINGER_R_FLOOR_M)
    oppose = _THUMB_OPPOSE_LATERAL if side == "l" else -_THUMB_OPPOSE_LATERAL
    # Palm-plane pitch: face -Y (0084 B6) so thumb swings across palm front, not rear.
    pitch = _THUMB_PALM_PITCH
    thumb_dir_raw = (
        axis[0] + oppose * 0.5,
        axis[1] + pitch,  # primary oppose read (AI1 P2)
        axis[2],
    )
    thumb_dir = _normalize(thumb_dir_raw) or axis
    # B5 pin floor 0.95*half_w; with B12 splay, outer finger sits ~0.975*half_w so
    # pure pin is slightly inboard of index -> raise lat to clear T14 soft notch
    # (open decision: geometry raise, not threshold-only harden).
    lat = half_w * _THUMB_BASE_LATERAL_FRAC_HALF_W
    outer_dx = splay * 0.5
    lat = max(lat, outer_dx + 0.35 * (fr + thumb_r0))
    thumb_base = _add(
        [palm_c[0] + (lat if side == "l" else -lat), palm_c[1], palm_c[2]],
        thumb_dir,
        0.05 * hand_len,
    )
    thumb_lens = (thumb_seg0, thumb_seg1)
    thumb_rs = (thumb_r0, thumb_r1)
    along_t = 0.0
    for si in range(2):
        seg_l = thumb_lens[si]
        p0 = _add(thumb_base, thumb_dir, along_t)
        p1 = _add(thumb_base, thumb_dir, along_t + seg_l)
        along_t += seg_l
        out.append(
            _capsule(
                f"RECIPE_thumb_soft_{si}_{side}",
                "thumb_soft",
                p0,
                p1,
                thumb_rs[si],
                parent_joint=pj_digit,
            )
        )
    return out


__all__ = [
    "ANK_RY_FLOOR_M",
    "ANK_RY_FRAC_HALF_W",
    "ANK_RZ_FLOOR_M",
    "ANK_RZ_FRAC_HALF_W",
    "ANK_RZ_MAX_FRAC_ANK_Z",
    "ANK_RZ_MIN_VS_CALF_B",
    "BALL_SOFT_RY_FRAC_HALF_DEPTH",
    "FINGER_TIERS",
    "FOOT_HW_MIN_FRAC_H",
    "FOOT_HW_MIN_FRAC_LEN",
    "FOOT_HW_MIN_VS_CALF_R",
    "FOOT_LEN_BASE_FRAC_H",
    "FOOT_LEN_MIN_VS_ANK_HW",
    "FOOT_LEN_MIN_VS_CALF_DIAM",
    "FOOT_LEN_VISUAL_MAX_FRAC_H",
    "FOOT_LEN_VISUAL_MIN_FRAC_H",
    "HEEL_CONTACT_OVERLAP_TARGET_M",
    "HEEL_REAR_OVERHANG_M",
    "HEEL_REAR_Y_BIAS_FRAC_DEPTH",
    "HEEL_RY_MAX_FRAC_HALF_DEPTH",
    "HEEL_RY_MIN_FRAC_DEPTH",
    "HEEL_RY_MIN_VS_RZ_FRAC",
    "HEEL_RZ_CAP_FRAC_ANK",
    "HEEL_Z_FRAC_ANK",
    "PARENT_UNRESOLVED_MSG",
    "SOLE_RZ_FLOOR_M",
    "SOLE_RZ_FRAC_OF_THICKNESS",
    "SOLE_THICKNESS_FRAC_H",
    "TOE_BALL_NEST_FRAC",
    "TOE_BASE_NEST_FRAC",
    "TOE_BIG_SCALE",
    "TOE_FULL_LEN_FRAC",
    "TOE_MIN_CENTER_SPACING_VS_R",
    "TOE_R_CAP_FRAC_HALF_W",
    "TOE_R_FLOOR_M",
    "TOE_R_FRAC_HALF_W",
    "TOE_SPLAY_FRAC_HALF_W",
    "TOE_TIERS",
    "TOE_TIP_MAX_PAST_BALL_FRAC",
    "TOE_TIP_MAX_PAST_BALL_M",
    "TOE_TIP_MAX_PAST_FRAC",
    "TOE_TIP_MAX_PAST_M",
    "TOE_TIP_PAD_SCALE",
    "TOE_TIP_PAST_FRAC",
    "TOE_WEDGE_RZ_FRAC_SOLE",
    "_FINGERTIP_PLANE_MARGIN_M",
    "_FINGER_DIGIT_L_SCALE",
    "_FINGER_DIGIT_R_SCALE",
    "_FINGER_R_SCALES_SEG",
    "_FINGER_SEG_FRACS_HAND",
    "_PALM_PAD_RY_FRAC_TH",
    "_PALM_THICKNESS_FRAC_HAND",
    "_THUMB_DISTAL_L_SCALE",
    "_THUMB_DISTAL_R_SCALE",
    "_THUMB_PALM_PITCH",
    "FingerTier",
    "ToeTier",
    "_fingertip_y_trusted",
    "apply_foot_half_width_visual_floor",
    "apply_foot_length_visual_floor",
    "build_foot_parts",
    "build_hand_parts",
    "finger_primary_axis",
    "resolve_foot_half_width_m",
    "resolve_foot_length_m",
]
