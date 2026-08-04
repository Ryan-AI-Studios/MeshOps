"""Hand / foot / digit RECIPE primitives (track 0029).

Opt-in blockout-grade extremities parented to wrist/hand/ankle/heel/toe joints
or landmarks. Authoring only — not print-ready articulated digits, not boots
as law (Difficulty §12 / N6).

All names stay RECIPE_* (never HAND_*/FOOT_*/DIGIT_* prefixes).
Ankle mass labels must contain ank_foot (classifier → ankle_bridge).
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
_PLATE_THICKNESS_FRAC_H: Final[float] = 0.02
_HAND_LEN_FALLBACK_FRAC_H: Final[float] = 0.11
_PALM_WIDTH_FRAC_HAND: Final[float] = 0.45
_PALM_THICKNESS_FRAC_HAND: Final[float] = 0.12
_MITTEN_LEN_FRAC_HAND: Final[float] = 0.55
_FINGER_SEG_FRAC_HAND: Final[float] = 1.0 / 5.0
_FINGER_R_FRAC_PALM: Final[float] = 0.08
_THUMB_SEG_FRAC_HAND: Final[float] = 1.0 / 6.0
_TOE_WEDGE_LEN_FRAC: Final[float] = 0.25
_TOE_FULL_LEN_FRAC: Final[float] = 0.20
_BALL_SOFT_R_FRAC_FOOT: Final[float] = 0.12
_HEEL_R_FRAC_FOOT: Final[float] = 0.12
_FINGER_Y_BIAS_FRAC: Final[float] = 0.10  # secondary only when no tip
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


# ---------------------------------------------------------------------------
# Finger axis (B7) — exported for tests
# ---------------------------------------------------------------------------


def finger_primary_axis(
    wrist: list[float] | None,
    fingertip: list[float] | None,
    *,
    hand_len: float,
) -> tuple[float, float, float]:
    """Primary finger/mitten axis: wrist→tip when both finite; else -Z (B7).

    Optional slight -Y bias (<=10% hand_len) is applied only as secondary nudge
    when no tip — returned as unit vector of (-small_Y, -Z) renormalized.
    """
    if wrist is not None and fingertip is not None and _finite3(wrist) and _finite3(fingertip):
        raw = (
            fingertip[0] - wrist[0],
            fingertip[1] - wrist[1],
            fingertip[2] - wrist[2],
        )
        n = _normalize(raw)
        if n is not None:
            return n

    # Default A-pose hang: primary -Z; slight -Y secondary only (B7, ≤10% hand_len)
    bias = _FINGER_Y_BIAS_FRAC * max(hand_len, 0.01)
    # Direction components before normalize: (0, -bias, -1) so -Z dominates
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
) -> list[RecipePart]:
    """Emit L/R foot plate + heel + ank_foot (+ toes per tier). RECIPE_* only."""
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

    half_depth = foot_len / 2.0
    half_width = resolve_foot_half_width_m(
        report,
        template_applied=template_applied,
        height_m=height_m,
        side=side,
        messages=messages,
    )
    thickness = (
        _PLATE_THICKNESS_FRAC_H * float(height_m) if height_m is not None and height_m > 0 else 0.03
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
        # Prefer measured span for plate half-depth when it is usable
        measured_half = abs(hy - ty) / 2.0
        if measured_half > _NEAR_ZERO:
            half_depth = measured_half
    elif ankle is not None:
        plate_y = float(ankle[1])
    else:
        plate_y = 0.0

    # Z floor: plate bottom ≈ 0 (B8)
    z_bottom = 0.0
    z_top = thickness
    plate_z = thickness / 2.0

    # Heel mass + ankle share rear-third Y so C_ankle_over_heel passes on heel-tol
    # path when both roles are present, and rear-third path when only plate.
    # Mid of rear third: plate_y + (2/3)*half_depth (toward heel +Y).
    stack_y = plate_y + (2.0 / 3.0) * half_depth
    heel_y = stack_y
    ank_y = stack_y
    toe_y = plate_y - (2.0 / 3.0) * half_depth  # front third for toe wedge

    # Ankle / heel Z
    if ankle is not None and ankle[2] == ankle[2] and float(ankle[2]) > z_top:
        ank_z = float(ankle[2])
    else:
        ank_z = z_top + half_width  # rest above plate top

    heel_r = max(_HEEL_R_FRAC_FOOT * foot_len, half_width * 0.8)
    if heel is not None and heel[2] == heel[2] and float(heel[2]) >= 0:
        heel_z = max(float(heel[2]), heel_r)
    else:
        heel_z = z_top + heel_r * 0.5

    toe_r = max(0.08 * foot_len, half_width * 0.6)
    if toe is not None and toe[2] == toe[2] and float(toe[2]) >= 0:
        toe_z = max(float(toe[2]), toe_r * 0.5)
    else:
        toe_z = z_top + toe_r * 0.4

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

    # foot_plate (B8.1)
    plate_name = f"RECIPE_foot_plate_{side}"
    out.append(
        _box(
            plate_name,
            "foot_plate",
            [plate_x, plate_y, plate_z],
            half_width=half_width,
            half_depth=half_depth,
            z_bottom=z_bottom,
            z_top=z_top,
            parent_joint=pj_plate,
        )
    )

    # heel mass at rear stack Y (+Y of plate mid) — same Y as ank_foot (0023)
    heel_name = f"RECIPE_heel_{side}"
    out.append(
        _ellipsoid(
            heel_name,
            "heel",
            [plate_x, heel_y, heel_z],
            half_width * 0.95,
            heel_r,
            heel_r * 0.85,
            parent_joint=pj_heel,
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
            half_width,  # rx = ankle half-width (same as plate half-width)
            half_width * 0.9,
            half_width,
            parent_joint=pj_ank,
        )
    )

    if toes == "none":
        return out

    if toes == "wedge":
        # Single toe wedge at front third (-Y)
        wedge_len = _TOE_WEDGE_LEN_FRAC * foot_len
        out.append(
            _ellipsoid(
                f"RECIPE_toe_soft_{side}",
                "toe_soft",
                [plate_x, toe_y, toe_z],
                half_width * 0.85,
                wedge_len / 2.0,
                toe_r,
                parent_joint=pj_toe,
            )
        )
        return out

    # toes == full: ball_soft + 5 toe capsules (no "foot" substring in soft names)
    ball_r = _BALL_SOFT_R_FRAC_FOOT * foot_len
    ball_y = plate_y - (1.0 / 3.0) * half_depth
    out.append(
        _ellipsoid(
            f"RECIPE_ball_soft_{side}",
            "ball_soft",
            [plate_x, ball_y, z_top + ball_r * 0.5],
            half_width * 0.9,
            ball_r,
            ball_r * 0.7,
            parent_joint=pj_ank,
        )
    )
    toe_len = _TOE_FULL_LEN_FRAC * foot_len
    toe_radius = max(half_width * 0.18, 0.005)
    splay = half_width * 0.7
    for i in range(1, 6):
        # 1=medial ... 5=lateral; sign by side
        t = (i - 3) / 3.0  # -2/3 ... +2/3
        dx = t * splay
        if side == "l":
            dx = -dx  # mirror splay
        base = [plate_x + dx, plate_y - half_depth + toe_len * 0.15, toe_z]
        tip = [plate_x + dx, plate_y - half_depth - toe_len * 0.35, toe_z * 0.9]
        out.append(
            _capsule(
                f"RECIPE_toe_{i}_{side}",
                "toe_soft",
                base,
                tip,
                toe_radius,
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

    # Hand estimate: hand joint or mid wrist→tip or tip alone
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

    # Palm center: hand joint or mid wrist→tip or wrist + 0.35*hand_len along axis
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

    palm_w = _PALM_WIDTH_FRAC_HAND * hand_len
    palm_th = _PALM_THICKNESS_FRAC_HAND * hand_len
    palm_len = 0.40 * hand_len  # along finger axis half-extent as box depth proxy

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

    # Palm as flat box: width X, thickness along Y-ish, length along axis approximated
    # Use box with half_depth ≈ palm_len/2 in Y for A-pose readability when axis is -Z
    # Prefer ellipsoid flat in palm plane when hanging (-Z primary)
    palm_name = f"RECIPE_palm_{side}"
    # Box oriented world axes (authoring): half_width = palm_w/2, half_depth small in Y,
    # z span = palm_len (hand hangs in Z when no tip)
    half_w = palm_w / 2.0
    if abs(axis[2]) >= abs(axis[1]):
        # Primary -Z hang: palm extent mainly in Z
        z_half = palm_len / 2.0
        out.append(
            _box(
                palm_name,
                "palm",
                palm_c,
                half_width=half_w,
                half_depth=max(palm_th, 0.01),
                z_bottom=palm_c[2] - z_half,
                z_top=palm_c[2] + z_half,
                parent_joint=pj_palm,
            )
        )
    else:
        # Tip-directed (often -Y-ish): use ellipsoid
        out.append(
            _ellipsoid(
                palm_name,
                "palm",
                palm_c,
                half_w,
                palm_len / 2.0,
                palm_th,
                parent_joint=pj_palm,
            )
        )

    if fingers == "none":
        return out

    if fingers == "mitten":
        mitten_len = _MITTEN_LEN_FRAC_HAND * hand_len
        p0 = _add(palm_c, axis, 0.15 * hand_len)
        p1 = _add(palm_c, axis, 0.15 * hand_len + mitten_len)
        out.append(
            _capsule(
                f"RECIPE_finger_mitten_{side}",
                "finger_soft",
                p0,
                p1,
                max(half_w * 0.55, 0.008),
                parent_joint=pj_digit,
            )
        )
        return out

    # fingers == full: 4 fingers x 3 capsules + thumb x 2
    seg = _FINGER_SEG_FRAC_HAND * hand_len
    fr = max(_FINGER_R_FRAC_PALM * palm_w, 0.004)
    # Lateral splay in X
    splay = half_w * 0.9
    # Perpendicular-ish offset in X for finger rows
    for fi, fname in enumerate(_FINGER_NAMES):
        t = (fi - 1.5) / 1.5  # -1 ... +1-ish
        # Mirror lateral splay: left flips sign so fingers fan correctly in +X world
        dx = (-t if side == "l" else t) * splay * 0.5
        base = _add([palm_c[0] + dx, palm_c[1], palm_c[2]], axis, 0.12 * hand_len)
        for si in range(3):
            p0 = _add(base, axis, si * seg)
            p1 = _add(base, axis, (si + 1) * seg)
            out.append(
                _capsule(
                    f"RECIPE_finger_{fname}_{si}_{side}",
                    "finger_soft",
                    p0,
                    p1,
                    fr,
                    parent_joint=pj_digit,
                )
            )

    # Thumb: 2 segs, slight oppose (+X for L, -X for R anatomical palm)
    thumb_seg = _THUMB_SEG_FRAC_HAND * hand_len
    oppose = 0.35 if side == "l" else -0.35
    # Combine axis with lateral oppose
    thumb_dir_raw = (axis[0] + oppose * 0.5, axis[1], axis[2])
    thumb_dir = _normalize(thumb_dir_raw) or axis
    thumb_base = _add(
        [palm_c[0] + (half_w * 0.6 if side == "l" else -half_w * 0.6), palm_c[1], palm_c[2]],
        thumb_dir,
        0.05 * hand_len,
    )
    for si in range(2):
        p0 = _add(thumb_base, thumb_dir, si * thumb_seg)
        p1 = _add(thumb_base, thumb_dir, (si + 1) * thumb_seg)
        out.append(
            _capsule(
                f"RECIPE_thumb_soft_{si}_{side}",
                "thumb_soft",
                p0,
                p1,
                fr * 1.15,
                parent_joint=pj_digit,
            )
        )
    return out


__all__ = [
    "FINGER_TIERS",
    "FOOT_LEN_BASE_FRAC_H",
    "PARENT_UNRESOLVED_MSG",
    "TOE_TIERS",
    "FingerTier",
    "ToeTier",
    "build_foot_parts",
    "build_hand_parts",
    "finger_primary_axis",
    "resolve_foot_half_width_m",
    "resolve_foot_length_m",
]
