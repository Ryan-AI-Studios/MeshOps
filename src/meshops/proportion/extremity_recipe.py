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
# Hand constants — 0048 bulk priors (anti-stick full digits; mitten fence unchanged)
_PALM_WIDTH_FRAC_HAND: Final[float] = 0.62
_PALM_THICKNESS_FRAC_HAND: Final[float] = 0.30
_PALM_PAD_RY_FRAC_TH: Final[float] = 0.65  # pad mult on thickness axis (was bare 0.55)
_PALM_LEN_FRAC_HAND: Final[float] = 0.48
_MITTEN_LEN_FRAC_HAND: Final[float] = 0.50
_MITTEN_R_FRAC_PALM: Final[float] = 0.72  # fat mitt, not thin stick — DO NOT CHANGE
_FINGER_SEG_FRAC_HAND: Final[float] = 1.0 / 5.0
_FINGER_R_FRAC_PALM: Final[float] = 0.16
_FINGER_R_FLOOR_M: Final[float] = 0.006
_FINGER_R_CAP_VS_HALF_W: Final[float] = 0.55
_FINGER_SPLAY_FRAC_HALF_W: Final[float] = 1.95
_FINGER_MIN_CENTER_SPACING_VS_R: Final[float] = 2.0  # T12 groove law docs
_THUMB_SEG_FRAC_HAND: Final[float] = 1.0 / 6.0
_THUMB_R_SCALE_VS_FINGER: Final[float] = 1.25
_THUMB_BASE_LATERAL_FRAC_HALF_W: Final[float] = 0.95
_THUMB_OPPOSE_LATERAL: Final[float] = 0.45
_THUMB_PALM_PITCH: Final[float] = 0.55
# Foot-friendly toe / sole fracs (0044 Phase 0 organic sole)
_TOE_WEDGE_LEN_FRAC: Final[float] = 0.42  # elongated front mass (not a ball on plate)
# 0054 full-toe bulk freezes (B5-B10, B15, B11)
TOE_R_FRAC_HALF_W: Final[float] = 0.36  # AI2 P2-1 must win on product hw~0.0263
TOE_R_FLOOR_M: Final[float] = 0.009
TOE_R_CAP_FRAC_HALF_W: Final[float] = 0.45
TOE_BIG_SCALE: Final[float] = 1.20
TOE_FULL_LEN_FRAC: Final[float] = 0.26  # was _TOE_FULL_LEN_FRAC 0.22
TOE_SPLAY_FRAC_HALF_W: Final[float] = 1.25
TOE_MIN_CENTER_SPACING_VS_R: Final[float] = 1.0  # soft B15
TOE_WEDGE_RZ_FRAC_SOLE: Final[float] = 0.85
_BALL_SOFT_R_FRAC_FOOT: Final[float] = 0.14
# Rounded sole: ellipsoid foot_plate (not world-axis square box).
# Heel min-floor inside max(...) — rear pad primary (0044 B6-B8), not tower.
_HEEL_R_FRAC_FOOT: Final[float] = 0.18
_HEEL_BRIDGE_OVERLAP_FRAC: Final[float] = 0.35  # 0040 reuse (reach overlap concept)
# 0044 B1-B3 / B6-B8 visual mass freezes
FOOT_LEN_VISUAL_MIN_FRAC_H: Final[float] = 0.12
FOOT_LEN_MIN_VS_ANK_HW: Final[float] = 4.8
FOOT_LEN_MIN_VS_CALF_DIAM: Final[float] = 1.55
HEEL_REAR_Y_BIAS_FRAC_DEPTH: Final[float] = 0.12
HEEL_Z_FRAC_ANK: Final[float] = 0.42
HEEL_RZ_CAP_FRAC_ANK: Final[float] = 0.48
HEEL_RY_MIN_FRAC_DEPTH: Final[float] = 0.42
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
    """0044 B1-B5: mannequin visual length floor; never shrinks measured/template.

    Floor sources (skip when unavailable): stature 0.12·H, ank half-width 4.8·hw,
    calf distal diam 1.55·(2·calf_r). Template length is not a floor when measured
    exists (B18).
    """
    floors: list[tuple[str, float]] = []
    if height_m is not None and height_m > 0:
        floors.append(("stature", FOOT_LEN_VISUAL_MIN_FRAC_H * float(height_m)))
    if half_width > 0:
        floors.append(("ank_hw", FOOT_LEN_MIN_VS_ANK_HW * float(half_width)))
    if calf_distal_r is not None and calf_distal_r > 0:
        diam = 2.0 * float(calf_distal_r)
        floors.append(("calf_diam", FOOT_LEN_MIN_VS_CALF_DIAM * diam))
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


def finger_primary_axis(
    wrist: list[float] | None,
    fingertip: list[float] | None,
    *,
    hand_len: float,
) -> tuple[float, float, float]:
    """Primary finger/mitten axis: wrist->tip when both finite; else -Z (B7).

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
    from meshops.proportion.constraints import HEEL_REACH_GAP_TOL_M

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

    # 0044 B1-B5: visual floor after resolve (never shrinks)
    calf_r = _calf_distal_r_from_parts(existing_parts, side)
    foot_len = apply_foot_length_visual_floor(
        foot_len,
        height_m=height_m,
        half_width=half_width,
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

    # Ankle mass: keep rx = half_width for C_foot_width; grow ry/rz so it reads as a
    # joint and meets the heel pad + calf distal (not a pea on the plate).
    ank_rx = half_width
    ank_ry = max(half_width * 1.05, 0.028)
    ank_rz = max(half_width * 1.35, 0.034)

    # Heel rear pad (B6-B8): lower center, +Y bias, rz capped — still reaches ank_foot.
    heel_z = HEEL_Z_FRAC_ANK * ank_z
    reach_need = (ank_z - ank_rz) - HEEL_REACH_GAP_TOL_M - heel_z
    heel_rz = max(reach_need, _HEEL_R_FRAC_FOOT * foot_len * 0.55, half_width * 0.9)
    rz_cap = HEEL_RZ_CAP_FRAC_ANK * ank_z
    if heel_rz > rz_cap + _NEAR_ZERO:
        # Prefer lower center so min rz can still meet reach under the cap
        heel_z = min(heel_z, (ank_z - ank_rz) - HEEL_REACH_GAP_TOL_M - rz_cap)
        heel_z = max(heel_z, z_top * 0.35)
        heel_rz = min(heel_rz, rz_cap)
        still_need = (ank_z - ank_rz) - HEEL_REACH_GAP_TOL_M - heel_z
        if still_need > heel_rz + _NEAR_ZERO:
            # Rare: cap loses to reach — grow rz and message (R4c fail-safe)
            heel_rz = still_need
            messages.append(
                f"foot_{side}: heel_rz cap lose for C_heel_reaches "
                f"(rz={heel_rz:.4f} > cap={rz_cap:.4f})"
            )
    heel_ry = max(
        HEEL_RY_MIN_FRAC_DEPTH * half_depth,
        half_depth * 0.38,
        heel_rz * 0.70,
    )
    heel_rx = max(half_width * 1.05, ank_rx * 0.95)
    # Strict heel_z < ank_z
    if heel_z >= ank_z - _NEAR_ZERO:
        heel_z = min(ank_z * 0.55, ank_z - _NEAR_ZERO)
        if heel_z < z_top * 0.35:
            heel_z = min(max(z_top * 0.35, ank_z * 0.35), ank_z - _NEAR_ZERO)

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
    ball_ry = max(_BALL_SOFT_R_FRAC_FOOT * foot_len * 1.1, half_depth * 0.28)
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

    # toes == full: 5 toe capsules past front edge (0054 B5-B10 / B15 bulk freezes)
    toe_len = TOE_FULL_LEN_FRAC * foot_len
    base_r = min(
        max(TOE_R_FRAC_HALF_W * half_width, TOE_R_FLOOR_M),
        TOE_R_CAP_FRAC_HALF_W * half_width,
    )
    splay = half_width * TOE_SPLAY_FRAC_HALF_W
    messages.append(
        f"foot_{side}: toe bulk full r={base_r:.4f} "
        f"(frac_hw={TOE_R_FRAC_HALF_W} floor={TOE_R_FLOOR_M})"
    )
    for i in range(1, 6):
        # 1=medial ... 5=lateral; sign by side
        t = (i - 3) / 3.0  # -2/3 ... +2/3
        dx = t * splay
        if side == "l":
            dx = -dx  # mirror splay
        r_i = base_r * (TOE_BIG_SCALE if i == 1 else 1.0)
        r_i = min(r_i, TOE_R_CAP_FRAC_HALF_W * half_width)  # cap big toe too
        # Past plate front (-Y)
        base = [plate_x + dx, plate_front_y - toe_len * 0.05, toe_z]
        tip = [plate_x + dx, plate_front_y - toe_len * 0.95, toe_z * 0.85]
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
    # B11: bulk message once when full digits emitted
    if fingers == "full" and parts:
        msgs.append(
            f"hand bulk: full digits r_frac={_FINGER_R_FRAC_PALM} "
            f"palm_w={_PALM_WIDTH_FRAC_HAND} splay={_FINGER_SPLAY_FRAC_HALF_W} (anti-stick)"
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

    # fingers == full: 4 fingers x 3 capsules + thumb x 2 (0048 bulk + splay)
    seg = _FINGER_SEG_FRAC_HAND * hand_len
    fr = min(
        max(_FINGER_R_FRAC_PALM * palm_w, _FINGER_R_FLOOR_M),
        _FINGER_R_CAP_VS_HALF_W * half_w,
    )
    # Lateral splay in X (B12: scale with bulk so grooves stay visible)
    splay = half_w * _FINGER_SPLAY_FRAC_HALF_W
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

    # Thumb: 2 segs, bulk + lateral base + palm-plane pitch (B5 / AI1 P2)
    thumb_seg = _THUMB_SEG_FRAC_HAND * hand_len
    thumb_r = fr * _THUMB_R_SCALE_VS_FINGER
    oppose = _THUMB_OPPOSE_LATERAL if side == "l" else -_THUMB_OPPOSE_LATERAL
    # Palm-plane pitch: +Y when axis primarily -Z (A-pose hang) so thumb swings across palm front
    pitch = _THUMB_PALM_PITCH  # positive Y component
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
    lat = max(lat, outer_dx + 0.35 * (fr + thumb_r))
    thumb_base = _add(
        [palm_c[0] + (lat if side == "l" else -lat), palm_c[1], palm_c[2]],
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
                thumb_r,
                parent_joint=pj_digit,
            )
        )
    return out


__all__ = [
    "FINGER_TIERS",
    "FOOT_LEN_BASE_FRAC_H",
    "FOOT_LEN_MIN_VS_ANK_HW",
    "FOOT_LEN_MIN_VS_CALF_DIAM",
    "FOOT_LEN_VISUAL_MIN_FRAC_H",
    "HEEL_REAR_Y_BIAS_FRAC_DEPTH",
    "HEEL_RY_MIN_FRAC_DEPTH",
    "HEEL_RZ_CAP_FRAC_ANK",
    "HEEL_Z_FRAC_ANK",
    "PARENT_UNRESOLVED_MSG",
    "SOLE_RZ_FLOOR_M",
    "SOLE_RZ_FRAC_OF_THICKNESS",
    "SOLE_THICKNESS_FRAC_H",
    "TOE_BIG_SCALE",
    "TOE_FULL_LEN_FRAC",
    "TOE_MIN_CENTER_SPACING_VS_R",
    "TOE_R_CAP_FRAC_HALF_W",
    "TOE_R_FLOOR_M",
    "TOE_R_FRAC_HALF_W",
    "TOE_SPLAY_FRAC_HALF_W",
    "TOE_TIERS",
    "TOE_WEDGE_RZ_FRAC_SOLE",
    "FingerTier",
    "ToeTier",
    "apply_foot_length_visual_floor",
    "build_foot_parts",
    "build_hand_parts",
    "finger_primary_axis",
    "resolve_foot_half_width_m",
    "resolve_foot_length_m",
]
