"""Pure-Python connection gap metrics for blockout fuse readiness (0039).

gap_m < 0 → overlapping (good for fuse)
gap_m == 0 → touching
gap_m > 0 → separated (island risk)

Required keys: shoulder_l, shoulder_r, hip_l, hip_r, neck, ankle_l, ankle_r.
Ankle uses stack-min of calf_b↔ank_foot and ank_foot↔foot_plate.
No bpy. Sphere proxy for ellipsoids; capsule for cylinders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from meshops.proportion.blockout_recipe import BlockoutRecipePackage, RecipePart

REQUIRED_GAP_KEYS: Final[tuple[str, ...]] = (
    "shoulder_l",
    "shoulder_r",
    "hip_l",
    "hip_r",
    "neck",
    "ankle_l",
    "ankle_r",
)

# Missing-part sentinel (always > 0 so nofuse-style fixtures document island risk).
_MISSING_GAP: Final[float] = 1e9

Axis = Literal[0, 1, 2]


def _strip_blender_suffix(name: str) -> str:
    base = name
    if "." in base:
        head, tail = base.rsplit(".", 1)
        if tail.isdigit():
            base = head
    return base


def is_toe_part(name: str) -> bool:
    """True for any RECIPE_toe_* (wedge or digit) — never grow/pull in join-ready."""
    return _strip_blender_suffix(name).startswith("RECIPE_toe_")


def _parts_by_name(parts: list[RecipePart]) -> dict[str, RecipePart]:
    return {_strip_blender_suffix(p.name): p for p in parts}


def _find_named(by_name: dict[str, RecipePart], *candidates: str) -> RecipePart | None:
    for c in candidates:
        p = by_name.get(c)
        if p is not None:
            return p
    return None


def _find_role(
    parts: list[RecipePart],
    role: str,
    *,
    side: Literal["l", "r"] | None = None,
    name_contains: str | None = None,
) -> RecipePart | None:
    for p in parts:
        if p.role != role:
            continue
        if side is not None:
            base = _strip_blender_suffix(p.name).lower()
            if not base.endswith(f"_{side}"):
                continue
        if name_contains is not None and name_contains not in p.name:
            continue
        return p
    return None


def part_center(part: RecipePart) -> list[float] | None:
    """Geometric center: center, else mid(p0, p1), else trap/box z-mid."""
    if part.center is not None and len(part.center) >= 3:
        c = [float(part.center[0]), float(part.center[1]), float(part.center[2])]
        if (
            part.z_bottom_m is not None
            and part.z_top_m is not None
            and part.kind
            in (
                "trap_box",
                "box",
            )
        ):
            # Prefer mid-height for vertical extent when Z still at package center.
            c[2] = 0.5 * (float(part.z_bottom_m) + float(part.z_top_m))
        return c
    if part.p0 is not None and part.p1 is not None and len(part.p0) >= 3 and len(part.p1) >= 3:
        return [
            0.5 * (float(part.p0[0]) + float(part.p1[0])),
            0.5 * (float(part.p0[1]) + float(part.p1[1])),
            0.5 * (float(part.p0[2]) + float(part.p1[2])),
        ]
    if part.z_bottom_m is not None and part.z_top_m is not None and part.half_depth_m is not None:
        # trap/box without center — unusual; invent midline center.
        return [
            0.0,
            0.0,
            0.5 * (float(part.z_bottom_m) + float(part.z_top_m)),
        ]
    return None


def part_sphere_radius(part: RecipePart) -> float | None:
    """Sphere-proxy radius: mean half-extents for ellipsoids; radius for capsules."""
    if part.kind == "ellipsoid":
        rs: list[float] = []
        for v in (part.rx_m, part.ry_m, part.rz_m):
            if v is not None:
                rs.append(float(v))
        if rs:
            return sum(rs) / len(rs)
        if part.radius_m is not None:
            return float(part.radius_m)
        return None
    if part.kind in ("cylinder", "capsule"):
        if part.radius_m is not None:
            return float(part.radius_m)
        return None
    if part.kind in ("trap_box", "box"):
        extents: list[float] = []
        if part.top_half_width_m is not None and part.bottom_half_width_m is not None:
            extents.append(max(float(part.top_half_width_m), float(part.bottom_half_width_m)))
        elif part.top_half_width_m is not None:
            extents.append(float(part.top_half_width_m))
        elif part.bottom_half_width_m is not None:
            extents.append(float(part.bottom_half_width_m))
        if part.half_depth_m is not None:
            extents.append(float(part.half_depth_m))
        if part.z_bottom_m is not None and part.z_top_m is not None:
            extents.append(0.5 * abs(float(part.z_top_m) - float(part.z_bottom_m)))
        if extents:
            return sum(extents) / len(extents)
        return None
    if part.radius_m is not None:
        return float(part.radius_m)
    return None


def sphere_proxy(part: RecipePart) -> tuple[list[float], float] | None:
    """Return (center, radius) sphere proxy, or None if under-specified."""
    c = part_center(part)
    r = part_sphere_radius(part)
    if c is None or r is None or r <= 0.0:
        return None
    return c, float(r)


def gap_along_axis(
    child: RecipePart,
    parent: RecipePart,
    axis: Axis,
) -> float | None:
    """Signed surface gap along *axis* only: dist_axis - r_child - r_parent.

    Negative = overlap. Not full 3D centroid distance (preserves joint anchors).
    """
    pc = sphere_proxy(child)
    pp = sphere_proxy(parent)
    if pc is None or pp is None:
        return None
    c0, r0 = pc
    c1, r1 = pp
    dist = abs(float(c0[axis]) - float(c1[axis]))
    return dist - r0 - r1


def socket_overlap_m(r_child: float, r_parent: float) -> float:
    """Default socket overlap target (meters)."""
    return max(0.008, 0.5 * min(float(r_child), float(r_parent)))


def _shoulder_pair(
    parts: list[RecipePart],
    by_name: dict[str, RecipePart],
    side: Literal["l", "r"],
) -> tuple[RecipePart, RecipePart] | None:
    child = _find_named(
        by_name,
        f"RECIPE_deltoid_soft_{side}",
        f"RECIPE_shoulder_bridge_{side}",
    )
    if child is None:
        child = _find_role(parts, "deltoid_soft", side=side)
    if child is None:
        child = _find_role(parts, "shoulder_bridge", side=side)
    parent = _find_named(
        by_name,
        "RECIPE_torso_oval_chest",
        "RECIPE_torso_trap",
    )
    if parent is None:
        parent = _find_role(parts, "torso")
    if parent is None:
        parent = _find_role(parts, "trap_soft", name_contains="chest")
    if child is None or parent is None:
        return None
    return child, parent


def _hip_pair(
    parts: list[RecipePart],
    by_name: dict[str, RecipePart],
    side: Literal["l", "r"],
) -> tuple[RecipePart, RecipePart] | None:
    child = _find_named(
        by_name,
        f"RECIPE_hip_bridge_{side}",
        f"RECIPE_limb_thigh_{side}",
    )
    if child is None:
        child = _find_role(parts, "hip_bridge", side=side)
    if child is None:
        # Proximal thigh capsule fallback (0045 B11/P3-2: exclude dist_soft decoys)
        for p in parts:
            if p.role == "limb_segment" and f"thigh_{side}" in p.name and "dist_soft" not in p.name:
                child = p
                break
    parent = _find_named(
        by_name,
        "RECIPE_torso_oval_hip",
        "RECIPE_pelvis_oval",
        "RECIPE_pelvis_bucket",
        "RECIPE_torso_trap",
    )
    if parent is None:
        parent = _find_role(parts, "pelvis")
    if parent is None:
        parent = _find_role(parts, "torso")
    if child is None or parent is None:
        return None
    return child, parent


def _neck_pairs(
    parts: list[RecipePart],
    by_name: dict[str, RecipePart],
) -> list[tuple[RecipePart, RecipePart]]:
    neck = _find_named(by_name, "RECIPE_neck")
    if neck is None:
        neck = _find_role(parts, "neck")
    if neck is None:
        return []
    pairs: list[tuple[RecipePart, RecipePart]] = []
    head = _find_named(by_name, "RECIPE_head")
    if head is None:
        head = _find_role(parts, "head")
    if head is not None:
        pairs.append((neck, head))
    chest = _find_named(
        by_name,
        "RECIPE_torso_oval_chest",
        "RECIPE_torso_trap",
    )
    if chest is None:
        chest = _find_role(parts, "torso")
    if chest is not None:
        pairs.append((neck, chest))
    return pairs


def _ankle_stack_gaps(
    by_name: dict[str, RecipePart],
    side: Literal["l", "r"],
) -> list[float]:
    """Gaps for stack links (Z). Heel is secondary parent only — not in stack-min."""
    calf_b = by_name.get(f"RECIPE_calf_b_{side}")
    ank = by_name.get(f"RECIPE_ank_foot_{side}")
    plate = by_name.get(f"RECIPE_foot_plate_{side}")
    gaps: list[float] = []
    if calf_b is not None and ank is not None:
        g = gap_along_axis(calf_b, ank, 2)
        if g is not None:
            gaps.append(g)
    if ank is not None and plate is not None:
        g = gap_along_axis(ank, plate, 2)
        if g is not None:
            gaps.append(g)
    return gaps


def connection_gap_metrics(package: BlockoutRecipePackage) -> dict[str, float]:
    """Per-class min gap_m (negative = overlap). Always returns REQUIRED_GAP_KEYS."""
    parts = list(package.parts)
    by_name = _parts_by_name(parts)
    out: dict[str, float] = {}

    for side in ("l", "r"):
        pair = _shoulder_pair(parts, by_name, side)  # type: ignore[arg-type]
        if pair is None:
            out[f"shoulder_{side}"] = _MISSING_GAP
        else:
            g = gap_along_axis(pair[0], pair[1], 1)  # Y depth
            out[f"shoulder_{side}"] = float(g) if g is not None else _MISSING_GAP

        pair_h = _hip_pair(parts, by_name, side)  # type: ignore[arg-type]
        if pair_h is None:
            out[f"hip_{side}"] = _MISSING_GAP
        else:
            g = gap_along_axis(pair_h[0], pair_h[1], 1)  # Y depth
            out[f"hip_{side}"] = float(g) if g is not None else _MISSING_GAP

        stack = _ankle_stack_gaps(by_name, side)  # type: ignore[arg-type]
        if not stack:
            out[f"ankle_{side}"] = _MISSING_GAP
        else:
            out[f"ankle_{side}"] = min(stack)

    neck_pairs = _neck_pairs(parts, by_name)
    if not neck_pairs:
        out["neck"] = _MISSING_GAP
    else:
        ngaps: list[float] = []
        for child, parent in neck_pairs:
            g = gap_along_axis(child, parent, 2)  # Z
            if g is not None:
                ngaps.append(float(g))
        out["neck"] = min(ngaps) if ngaps else _MISSING_GAP

    # Ensure all required keys present
    for k in REQUIRED_GAP_KEYS:
        out.setdefault(k, _MISSING_GAP)
    return out


def resolve_join_connections(
    parts: list[RecipePart],
) -> list[tuple[str, RecipePart, RecipePart, Axis]]:
    """List of (class_id, child, parent, axis) for join-ready post-pass.

    Ankle emits both stack links as separate rows (class ankle_l / ankle_r),
    distal-first (ank→plate before calf→ank) so multi-link pull does not reopen
    the proximal gap. Heel is secondary parent when plate is missing.
    Toes never appear as children.
    """
    by_name = _parts_by_name(parts)
    rows: list[tuple[str, RecipePart, RecipePart, Axis]] = []

    for side in ("l", "r"):
        # Shoulder: nudge deltoid and bridge independently when both present
        parent_s = None
        pair = _shoulder_pair(parts, by_name, side)  # type: ignore[arg-type]
        if pair is not None:
            parent_s = pair[1]
        if parent_s is not None:
            for cname in (
                f"RECIPE_deltoid_soft_{side}",
                f"RECIPE_shoulder_bridge_{side}",
            ):
                child = by_name.get(cname)
                if child is not None and not is_toe_part(child.name):
                    rows.append((f"shoulder_{side}", child, parent_s, 1))

        pair_h = _hip_pair(parts, by_name, side)  # type: ignore[arg-type]
        if pair_h is not None:
            child_h, parent_h = pair_h
            if not is_toe_part(child_h.name):
                rows.append((f"hip_{side}", child_h, parent_h, 1))
            # Also proximal thigh when bridge was primary
            thigh = by_name.get(f"RECIPE_limb_thigh_{side}")
            if thigh is not None and thigh is not child_h and not is_toe_part(thigh.name):
                rows.append((f"hip_{side}", thigh, parent_h, 1))

        # Ankle stack (Z): distal-first so calf pull sees ank at final position.
        # Order: ank_foot→foot_plate (heel secondary parent), then calf_b→ank_foot.
        calf_b = by_name.get(f"RECIPE_calf_b_{side}")
        ank = by_name.get(f"RECIPE_ank_foot_{side}")
        plate = by_name.get(f"RECIPE_foot_plate_{side}")
        heel = by_name.get(f"RECIPE_heel_{side}")
        if ank is not None and plate is not None and not is_toe_part(ank.name):
            rows.append((f"ankle_{side}", ank, plate, 2))
        elif ank is not None and heel is not None and not is_toe_part(ank.name):
            rows.append((f"ankle_{side}", ank, heel, 2))
        if calf_b is not None and ank is not None and not is_toe_part(calf_b.name):
            rows.append((f"ankle_{side}", calf_b, ank, 2))

    for child, parent in _neck_pairs(parts, by_name):
        if not is_toe_part(child.name):
            rows.append(("neck", child, parent, 2))

    return rows


__all__ = [
    "REQUIRED_GAP_KEYS",
    "connection_gap_metrics",
    "gap_along_axis",
    "is_toe_part",
    "part_center",
    "part_sphere_radius",
    "resolve_join_connections",
    "socket_overlap_m",
    "sphere_proxy",
]
