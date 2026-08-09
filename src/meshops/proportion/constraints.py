"""Blockout role graph + hard constraints + constrained optimize (track 0023).

Named roles replace free-name substring optimizers. Validate and optimize are
authoring QA only — not mesh or print success (Difficulty §12 / N6).

Freezes (session bugs — do not re-learn):
- ank_foot classified before foot → ankle_bridge not foot_plate
- whole RECIPE_limb_calf → calf (not calf_proximal)
- freeze-feet default freezes foot_plate / heel / ankle_bridge / calf_distal
- no scipy.optimize; no package_score reweight; no recipe schema bump
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.blockout_recipe import (
    BlockoutRecipePackage,
    RecipePart,
    load_blockout_recipe,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import CONSTRAINT_HONESTY, OPTIMIZE_HONESTY

# ---------------------------------------------------------------------------
# Constants (v1 freeze — testable)
# ---------------------------------------------------------------------------

CONSTRAINTS_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
OPTIMIZE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"

CONSTRAINTS_REPORT_BASENAME: Final[str] = "constraints_report.json"
OPTIMIZE_RESULT_BASENAME: Final[str] = "optimize_result.json"
OPTIMIZED_RECIPE_BASENAME: Final[str] = "blockout_recipe.json"

ANKLE_OVER_HEEL_TOL_M: Final[float] = 0.03
FOOT_WIDTH_TOL_M: Final[float] = 0.015
OUTER_X_TOL_M: Final[float] = 0.02
SOFT_GAP_FRAC: Final[float] = 0.9
AXIAL_DEPTH_MARGIN_M: Final[float] = 0.02

# 0042 foot-stack connectivity (hard-when-present; RECIPE-only)
TOE_FORWARD_EPS_M: Final[float] = 0.005
HEEL_REACH_GAP_TOL_M: Final[float] = 0.015
TOE_SOLE_ABS_M: Final[float] = 0.04
TOE_SOLE_RZ_FRAC: Final[float] = 1.5
TOE_SOLE_CEIL_M: Final[float] = 0.06

BAND_W_BREAST: Final[float] = 1.5
BAND_W_GLUTE: Final[float] = 1.5
BAND_W_THIGH: Final[float] = 1.0
BAND_W_CALF: Final[float] = 0.5
BAND_W_FOOT: Final[float] = 0.0

OPTIMIZE_FAST_SEED: Final[int] = 11
OPTIMIZE_SLOW_SEED: Final[int] = 13

_FAST_N_TRIALS: Final[int] = 24
_SLOW_N_TRIALS: Final[int] = 16
_FAST_STEP_M: Final[float] = 0.008
_SLOW_STEP_M: Final[float] = 0.006

ConstraintRole = Literal[
    "torso",
    "pelvis",
    "neck",
    "head",
    "shoulder_bridge",
    "hip_bridge",
    "breast",
    "glute",
    "deltoid",
    "iliac",
    "thigh",
    "calf",
    "calf_proximal",
    "calf_distal",
    "upper_arm",
    "forearm",
    "foot_plate",
    "heel",
    "ankle_bridge",
    "unknown",
]
Side = Literal["l", "r", "none"]
RuleStatus = Literal["pass", "fail", "skip"]
OptimizeMode = Literal["fast", "slow"]

# Foot-stack roles frozen by --freeze-feet (whole calf is NOT auto-frozen).
FREEZE_FEET_ROLES: Final[frozenset[str]] = frozenset(
    {"foot_plate", "heel", "ankle_bridge", "calf_distal"}
)

# Limb / foot roles subject to (role, side) uniqueness in C_no_dup_limb.
_LIMB_FOOT_ROLES: Final[frozenset[str]] = frozenset(
    {
        "thigh",
        "calf",
        "calf_proximal",
        "calf_distal",
        "upper_arm",
        "forearm",
        "foot_plate",
        "heel",
        "ankle_bridge",
        "breast",
        "glute",
        "deltoid",
        "iliac",
    }
)

_BLENDER_SUFFIX_RE = re.compile(r"\.\d{3}$")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ConstraintRuleResult(BaseModel):
    """One hard-constraint rule outcome."""

    model_config = ConfigDict(extra="forbid")

    id: str
    status: RuleStatus
    message: str
    metrics: dict[str, Any] | None = None


class ConstraintsReport(BaseModel):
    """constraints_report.json (schema 1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = CONSTRAINTS_SCHEMA_VERSION
    honesty: str = CONSTRAINT_HONESTY
    ok: bool
    rules: list[ConstraintRuleResult] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    classified: list[dict[str, str]] = Field(default_factory=list)


class OptimizeResult(BaseModel):
    """optimize_result.json (schema 1.0.0) — B3 freeze."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = OPTIMIZE_SCHEMA_VERSION
    honesty: str = OPTIMIZE_HONESTY
    mode: OptimizeMode
    freeze_feet: bool
    score_before: float
    score_after: float
    moved_roles: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    n_trials: int | None = None
    n_kept: int | None = None


# ---------------------------------------------------------------------------
# Classifier (B1 / B2)
# ---------------------------------------------------------------------------


def strip_blender_suffix(name: str) -> str:
    """Strip trailing Blender ``.001`` / ``.###`` numeric suffixes."""
    return _BLENDER_SUFFIX_RE.sub("", name)


def side_from_name(name: str) -> Side:
    """Side from ``_l`` / ``_r`` suffix after Blender suffix strip; else ``none``."""
    base = strip_blender_suffix(name).lower()
    if base.endswith("_l"):
        return "l"
    if base.endswith("_r"):
        return "r"
    return "none"


def classify_part_name(name: str) -> tuple[ConstraintRole, Side]:
    """Ordered role classifier — more specific first (name lowercased).

    Order freeze (B1):
    1. ank_foot → ankle_bridge (NOT foot_plate)
    2. heel → heel
    3. foot → foot_plate (only if not ank_foot)
    4. RECIPE_limb_calf / limb_calf → calf (whole product capsule)
    5. calf + ends _b → calf_distal
    6. calf + ends _a → calf_proximal
    7. bare calf → calf (NOT proximal)
    8. thigh / limb_thigh, arms, bridges, softs, axial, unknown
    """
    raw = strip_blender_suffix(name)
    lower = raw.lower()
    side = side_from_name(name)
    # Stem without side suffix for _a/_b end checks.
    stem = lower
    if stem.endswith("_l") or stem.endswith("_r"):
        stem = stem[:-2]

    # 1 — ank_foot before any "foot" substring match
    if "ank_foot" in lower:
        return "ankle_bridge", side
    # 2
    if "heel" in lower:
        return "heel", side
    # 3
    if "foot" in lower:
        return "foot_plate", side
    # 4 — whole product calf capsule (before _a/_b split rules)
    if "recipe_limb_calf" in lower or "limb_calf" in lower:
        return "calf", side
    # 5-7 calf split / generic
    if "calf" in lower:
        if stem.endswith("_b"):
            return "calf_distal", side
        if stem.endswith("_a"):
            return "calf_proximal", side
        return "calf", side
    # 8 — limbs, bridges, softs, axial
    # 0027 profile softs: ordered before generic fallthrough (AI2 B2).
    # Use trap_soft (not bare "trap") so RECIPE_torso_trap stays torso (0019 default).
    # bicep before upper_arm limb; clavicle before shoulder_bridge; scap before torso.
    if "trap_soft" in lower:
        return "neck", side
    if "bicep" in lower:
        return "upper_arm", side  # soft bead; exempt from limb no-dup vs limb_segment
    if "clavicle" in lower:
        return "shoulder_bridge", side
    if "scap" in lower:
        return "torso", side
    # 0028 face kit tokens (B5): after 0027 softs, before generic fallthrough.
    # Prefer *_soft / multi-token forms (trap_soft lesson). Exact substrings.
    if "jaw" in lower:
        return "head", side
    if "brow_soft" in lower:
        return "head", side
    if "eye_soft" in lower:
        return "head", side
    if "nose_soft" in lower:
        return "head", side
    if "ear_soft" in lower:
        return "head", side
    if "lip_soft" in lower:
        return "head", side
    if "hair_mass" in lower:
        return "head", side
    if "sternomastoid" in lower:
        return "neck", side
    if "neckline" in lower:
        return "neck", side
    # 0029 extremity softs (B5): must not map to foot_plate / limb no-dup.
    # After face kit; before generic limb/foot fallthrough. Soft digits must not
    # contain substring "foot". Explicit unknown so C_role_classified is quiet.
    if "toe_soft" in lower:
        return "unknown", side
    if "ball_soft" in lower:
        return "unknown", side
    if "palm" in lower:
        return "unknown", side
    if "finger" in lower or "mitten" in lower:
        return "unknown", side
    if "thumb_soft" in lower:
        return "unknown", side
    # RECIPE_toe_{1..5}_* (full toes) — no "foot" substring → unknown (not foot_plate)
    if "recipe_toe_" in lower or (
        "toe_" in lower and "ank_foot" not in lower and "foot" not in lower
    ):
        return "unknown", side
    # 0045 limb visual mass softs: before generic thigh/upper_arm/forearm (B6).
    # dist_soft must not label as arm ConstraintRole; knee_soft explicit pin.
    # 0046 B7: prox_soft before generic thigh (C_no_dup safe).
    # 0069 B9: hip_soft → unknown (defensive; free-set skip); keep legacy prox_soft.
    # 0070 B5: thigh_taper (dist shaft seg) → unknown before generic thigh (C_no_dup).
    # 0062 B8: arm_taper + elbow_soft → unknown before generic upper_arm/forearm.
    if "knee_soft" in lower:
        return "unknown", side
    if "dist_soft" in lower:
        return "unknown", side
    if "hip_soft" in lower:
        return "unknown", side
    if "prox_soft" in lower:
        return "unknown", side
    if "thigh_taper" in lower:
        return "unknown", side
    if "arm_taper" in lower:
        return "unknown", side
    if "elbow_soft" in lower:
        return "unknown", side
    if "thigh" in lower or "limb_thigh" in lower:
        return "thigh", side
    if "upper_arm" in lower or "limb_upper_arm" in lower:
        return "upper_arm", side
    if "forearm" in lower or "limb_forearm" in lower:
        return "forearm", side
    if "hip_bridge" in lower:
        return "hip_bridge", side
    if "shoulder_bridge" in lower:
        return "shoulder_bridge", side
    if "breast" in lower or "pec" in lower:
        return "breast", side
    if "glute" in lower:
        return "glute", side
    if "deltoid" in lower:
        return "deltoid", side
    if "iliac" in lower:
        return "iliac", side
    if "torso" in lower or "chest" in lower:
        return "torso", side
    if "pelvis" in lower:
        return "pelvis", side
    if "neck" in lower:
        return "neck", side
    if "head" in lower or "cranium" in lower or "skull" in lower:
        return "head", side
    return "unknown", side


def classify_part(part: RecipePart) -> tuple[ConstraintRole, Side]:
    """Classify a recipe part by name."""
    return classify_part_name(part.name)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def part_center_xyz(part: RecipePart) -> list[float] | None:
    """World center as [x, y, z]; from center or midpoint of p0/p1."""
    if part.center is not None and len(part.center) >= 3:
        return [float(part.center[0]), float(part.center[1]), float(part.center[2])]
    if part.p0 is not None and part.p1 is not None and len(part.p0) >= 3 and len(part.p1) >= 3:
        return [
            (float(part.p0[0]) + float(part.p1[0])) / 2.0,
            (float(part.p0[1]) + float(part.p1[1])) / 2.0,
            (float(part.p0[2]) + float(part.p1[2])) / 2.0,
        ]
    return None


def part_y(part: RecipePart) -> float | None:
    c = part_center_xyz(part)
    return None if c is None else c[1]


def part_x(part: RecipePart) -> float | None:
    c = part_center_xyz(part)
    return None if c is None else c[0]


def part_z(part: RecipePart) -> float | None:
    """World center Z from center or p0/p1 midpoint."""
    c = part_center_xyz(part)
    return None if c is None else c[2]


def part_rz(part: RecipePart) -> float | None:
    """Vertical half-extent from ``rz_m`` only (no invent from radius)."""
    if part.rz_m is None:
        return None
    return float(part.rz_m)


def set_part_y(part: RecipePart, y: float) -> None:
    """Move part primarily in world Y (center and/or p0/p1)."""
    if part.center is not None and len(part.center) >= 3:
        old = float(part.center[1])
        part.center = [float(part.center[0]), float(y), float(part.center[2])]
        dy = float(y) - old
        if part.p0 is not None and len(part.p0) >= 3:
            part.p0 = [float(part.p0[0]), float(part.p0[1]) + dy, float(part.p0[2])]
        if part.p1 is not None and len(part.p1) >= 3:
            part.p1 = [float(part.p1[0]), float(part.p1[1]) + dy, float(part.p1[2])]
        return
    if part.p0 is not None and part.p1 is not None:
        mid = (float(part.p0[1]) + float(part.p1[1])) / 2.0
        dy = float(y) - mid
        part.p0 = [float(part.p0[0]), float(part.p0[1]) + dy, float(part.p0[2])]
        part.p1 = [float(part.p1[0]), float(part.p1[1]) + dy, float(part.p1[2])]


def set_part_x(part: RecipePart, x: float) -> None:
    """Limited X move for outer=hip_bridge clamps."""
    if part.center is not None and len(part.center) >= 3:
        old = float(part.center[0])
        part.center = [float(x), float(part.center[1]), float(part.center[2])]
        dx = float(x) - old
        if part.p0 is not None and len(part.p0) >= 3:
            part.p0 = [float(part.p0[0]) + dx, float(part.p0[1]), float(part.p0[2])]
        if part.p1 is not None and len(part.p1) >= 3:
            part.p1 = [float(part.p1[0]) + dx, float(part.p1[1]), float(part.p1[2])]
        return
    if part.p0 is not None and part.p1 is not None:
        mid = (float(part.p0[0]) + float(part.p1[0])) / 2.0
        dx = float(x) - mid
        part.p0 = [float(part.p0[0]) + dx, float(part.p0[1]), float(part.p0[2])]
        part.p1 = [float(part.p1[0]) + dx, float(part.p1[1]), float(part.p1[2])]


def _half_extent_x(part: RecipePart) -> float | None:
    if part.rx_m is not None:
        return float(part.rx_m)
    if part.radius_m is not None:
        return float(part.radius_m)
    if part.top_half_width_m is not None and part.bottom_half_width_m is not None:
        return max(float(part.top_half_width_m), float(part.bottom_half_width_m))
    if part.top_half_width_m is not None:
        return float(part.top_half_width_m)
    if part.bottom_half_width_m is not None:
        return float(part.bottom_half_width_m)
    return None


def _width_x(part: RecipePart) -> float | None:
    half = _half_extent_x(part)
    return None if half is None else 2.0 * half


def _depth_extent_y(part: RecipePart) -> float | None:
    """Half-depth / ry toward +Y heel for foot plate rear-third (B4)."""
    if part.ry_m is not None:
        return float(part.ry_m)
    if part.half_depth_m is not None:
        return float(part.half_depth_m)
    return None


def _outer_x(part: RecipePart, side: Side) -> float | None:
    """Outer tip X: +X for right, -X for left."""
    cx = part_x(part)
    if cx is None:
        return None
    half = _half_extent_x(part) or 0.0
    if side == "r":
        return cx + half
    if side == "l":
        return cx - half
    return cx


def _thigh_chain_outer_x(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
    thigh: RecipePart,
    side: Side,
) -> float | None:
    """0070: C_thigh_outer uses full hip->knee chain mid when shaft is split.

    Prox segment mid sits ~0.25x along an adducted chain and falsely reads more
    lateral than the pre-0070 full capsule mid. Prefer chain ends:
    limb_thigh.p0 + taper_dist.p1, half-extent from prox shaft radius.
    """
    if side not in ("l", "r"):
        return _outer_x(thigh, side)
    by_name = {p.name: p for p, _, _ in indexed}
    dist = by_name.get(f"RECIPE_thigh_taper_dist_{side}")
    if dist is None or thigh.p0 is None or dist.p1 is None or len(thigh.p0) < 1 or len(dist.p1) < 1:
        return _outer_x(thigh, side)
    cx = 0.5 * (float(thigh.p0[0]) + float(dist.p1[0]))
    half = _half_extent_x(thigh) or 0.0
    if side == "r":
        return cx + half
    return cx - half


def _index_parts(
    package: BlockoutRecipePackage,
) -> list[tuple[RecipePart, ConstraintRole, Side]]:
    out: list[tuple[RecipePart, ConstraintRole, Side]] = []
    for part in package.parts:
        role, side = classify_part(part)
        out.append((part, role, side))
    return out


def _find(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
    role: ConstraintRole,
    side: Side | None = None,
) -> list[RecipePart]:
    found: list[RecipePart] = []
    for part, r, s in indexed:
        if r != role:
            continue
        if side is not None and s != side:
            continue
        found.append(part)
    return found


def _find_name_contains(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
    token: str,
    side: Side | None = None,
) -> list[RecipePart]:
    """Parts whose stripped name contains ``token`` (case-insensitive).

    ``side=None`` matches any side (used for palm global rule).
    """
    needle = token.lower()
    found: list[RecipePart] = []
    for part, _role, s in indexed:
        if side is not None and s != side:
            continue
        base = strip_blender_suffix(part.name).lower()
        if needle in base:
            found.append(part)
    return found


def _find_toe_mass(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
    side: Side,
) -> RecipePart | None:
    """Toe mass for side: ``toe_soft`` preferred, else min-Y ``recipe_toe_*``."""
    softs = _find_name_contains(indexed, "toe_soft", side)
    if softs:
        best: RecipePart | None = None
        best_y: float | None = None
        for p in softs:
            y = part_y(p)
            if y is None:
                if best is None:
                    best = p
                continue
            if best_y is None or y < best_y:
                best_y = y
                best = p
        return best

    candidates: list[RecipePart] = []
    for part, _role, s in indexed:
        if s != side:
            continue
        lower = strip_blender_suffix(part.name).lower()
        if "recipe_toe_" in lower:
            candidates.append(part)
    if not candidates:
        return None
    best_c: RecipePart | None = None
    best_cy = float("inf")
    for p in candidates:
        y = part_y(p)
        if y is None:
            continue
        if y < best_cy:
            best_cy = y
            best_c = p
    return best_c


def _plate_z_top(plate: RecipePart) -> float | None:
    """Foot plate top Z from ``z_top_m`` only (no invent)."""
    if plate.z_top_m is None:
        return None
    return float(plate.z_top_m)


def _toe_sole_slack_m(toe_rz_eff: float) -> float:
    """Sole-class slack above plate top: min(max(abs, rz*frac), ceil)."""
    return min(
        max(TOE_SOLE_ABS_M, float(toe_rz_eff) * TOE_SOLE_RZ_FRAC),
        TOE_SOLE_CEIL_M,
    )


# ---------------------------------------------------------------------------
# Report / template optional loads
# ---------------------------------------------------------------------------


def _optional_report(path: Path | str | None) -> Any | None:
    if path is None:
        return None
    from meshops.proportion.analyze import load_report

    return load_report(path)


def _optional_template(path: Path | str | None) -> Any | None:
    if path is None:
        return None
    from meshops.proportion.body_template import load_template_applied

    return load_template_applied(path)


def _lm_y(report: Any | None, landmark_id: str) -> float | None:
    if report is None:
        return None
    lms = getattr(report, "landmarks_xyz", None) or {}
    lm = lms.get(landmark_id) if isinstance(lms, dict) else None
    if lm is None:
        return None
    y = getattr(lm, "y_m", None)
    return None if y is None else float(y)


def _template_gap_m(tpl: Any | None, field: str) -> float | None:
    if tpl is None:
        return None
    constants = getattr(tpl, "constants", None)
    if constants is None:
        return None
    val = getattr(constants, field, None)
    return None if val is None else float(val)


def _bust_half_width_m(report: Any | None) -> float | None:
    """Bust half-width from report diameters (band_id=bust) when present."""
    if report is None:
        return None
    diameters = getattr(report, "diameters", None) or []
    for d in diameters:
        if getattr(d, "band_id", None) != "bust":
            continue
        hw = getattr(d, "half_width_m", None)
        if hw is not None:
            return float(hw)
        width = getattr(d, "width_m", None)
        if width is not None:
            return float(width) / 2.0
    return None


def _measured_soft_gap_m(report: Any | None, field: str) -> float | None:
    """B7: measured soft_spacing field when finite (first prefer)."""
    if report is None:
        return None
    soft = getattr(report, "soft_spacing", None)
    if soft is None:
        return None
    val = getattr(soft, field, None)
    if val is None:
        return None
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return None
    if fv != fv:  # NaN
        return None
    return fv


def _hip_half_width_m(report: Any | None) -> float | None:
    """Hip half-width from report diameters (band_id=hip) when present."""
    if report is None:
        return None
    diameters = getattr(report, "diameters", None) or []
    for d in diameters:
        if getattr(d, "band_id", None) != "hip":
            continue
        hw = getattr(d, "half_width_m", None)
        if hw is not None:
            return float(hw)
        width = getattr(d, "width_m", None)
        if width is not None:
            return float(width) / 2.0
    return None


def _intermammary_gap_m(
    template_applied: Any | None,
    report: Any | None = None,
) -> float | None:
    """B7 ladder: measured soft_spacing → template gap_m → frac * bust_hw → None."""
    measured = _measured_soft_gap_m(report, "intermammary_gap_m")
    if measured is not None:
        return measured
    gap = _template_gap_m(template_applied, "intermammary_gap_m")
    if gap is not None:
        return gap
    frac = _template_gap_m(template_applied, "intermammary_gap_frac")
    if frac is None:
        return None
    bust_hw = _bust_half_width_m(report)
    if bust_hw is None or bust_hw <= 0.0:
        return None
    return float(frac) * float(bust_hw)


def _glute_cleft_gap_m(
    template_applied: Any | None,
    report: Any | None = None,
) -> float | None:
    """B7 ladder: measured soft_spacing → template glute_cleft_m → frac * hip_hw → None."""
    measured = _measured_soft_gap_m(report, "glute_cleft_gap_m")
    if measured is not None:
        return measured
    gap = _template_gap_m(template_applied, "glute_cleft_m")
    if gap is not None:
        return gap
    frac = _template_gap_m(template_applied, "glute_cleft_frac")
    if frac is None:
        return None
    hip_hw = _hip_half_width_m(report)
    if hip_hw is None or hip_hw <= 0.0:
        return None
    return float(frac) * float(hip_hw)


# ---------------------------------------------------------------------------
# Hard rules
# ---------------------------------------------------------------------------


def _rule(
    rule_id: str,
    status: RuleStatus,
    message: str,
    metrics: dict[str, Any] | None = None,
) -> ConstraintRuleResult:
    return ConstraintRuleResult(
        id=rule_id,
        status=status,
        message=message,
        metrics=metrics,
    )


def _check_ankle_over_heel(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
) -> ConstraintRuleResult:
    """B4: ankle over heel (heel tol OR foot_plate rear-third +Y)."""
    sides: list[Side] = ["l", "r"]
    metrics: dict[str, Any] = {}
    statuses: list[RuleStatus] = []
    messages: list[str] = []

    for side in sides:
        ankles = _find(indexed, "ankle_bridge", side)
        if not ankles:
            continue
        ankle = ankles[0]
        ay = part_y(ankle)
        if ay is None:
            statuses.append("skip")
            messages.append(f"ankle_{side}: no Y")
            continue

        heels = _find(indexed, "heel", side)
        plates = _find(indexed, "foot_plate", side)
        heel_y = part_y(heels[0]) if heels else None
        plate = plates[0] if plates else None
        plate_y = part_y(plate) if plate is not None else None
        extent = _depth_extent_y(plate) if plate is not None else None

        metrics[f"ankle_y_{side}"] = ay
        if heel_y is not None:
            metrics[f"heel_y_{side}"] = heel_y
            delta = abs(ay - heel_y)
            metrics[f"ankle_heel_delta_{side}"] = delta
            if delta <= ANKLE_OVER_HEEL_TOL_M:
                statuses.append("pass")
                messages.append(f"ankle_{side}: |ΔY|={delta:.4f} ≤ {ANKLE_OVER_HEEL_TOL_M}")
            else:
                statuses.append("fail")
                messages.append(f"ankle_{side}: |ΔY heel|={delta:.4f} > {ANKLE_OVER_HEEL_TOL_M}")
            continue

        if plate_y is not None and extent is not None and extent > 0:
            # B4(2): heel direction +Y — rear third only = [cy+(1/3)*ext, cy+ext]
            rear0 = plate_y + (1.0 / 3.0) * extent
            rear1 = plate_y + extent
            front0 = plate_y - extent
            front1 = plate_y - (1.0 / 3.0) * extent
            metrics[f"foot_plate_y_{side}"] = plate_y
            metrics[f"extent_y_{side}"] = extent
            metrics[f"rear_third_{side}"] = [rear0, rear1]
            in_rear = rear0 <= ay <= rear1 or abs(ay - rear1) <= 1e-9
            # Fail if closer to toe-side front third than rear when both ends known
            mid_rear = 0.5 * (rear0 + rear1)
            mid_front = 0.5 * (front0 + front1)
            closer_front = abs(ay - mid_front) + 1e-12 < abs(ay - mid_rear)
            if closer_front and (front0 <= ay <= front1 or ay < plate_y):
                statuses.append("fail")
                messages.append(
                    f"ankle_{side}: closer to toe front-third than heel rear "
                    f"(y={ay:.4f}, plate_y={plate_y:.4f})"
                )
            elif in_rear:
                statuses.append("pass")
                messages.append(
                    f"ankle_{side}: in rear third of foot plate "
                    f"(y={ay:.4f}, rear=[{rear0:.4f},{rear1:.4f}])"
                )
            else:
                statuses.append("fail")
                messages.append(
                    f"ankle_{side}: not in rear third (y={ay:.4f}, rear=[{rear0:.4f},{rear1:.4f}])"
                )
            continue

        statuses.append("skip")
        messages.append(f"ankle_{side}: no heel and no usable foot extent")

    if not statuses:
        return _rule(
            "C_ankle_over_heel",
            "skip",
            "no ankle_bridge parts",
            metrics or None,
        )
    if any(s == "fail" for s in statuses):
        return _rule(
            "C_ankle_over_heel",
            "fail",
            "; ".join(messages),
            metrics,
        )
    if all(s == "skip" for s in statuses):
        return _rule(
            "C_ankle_over_heel",
            "skip",
            "; ".join(messages),
            metrics or None,
        )
    return _rule(
        "C_ankle_over_heel",
        "pass",
        "; ".join(messages),
        metrics,
    )


def _check_foot_width(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
) -> ConstraintRuleResult:
    metrics: dict[str, Any] = {}
    statuses: list[RuleStatus] = []
    messages: list[str] = []
    for side in ("l", "r"):
        plates = _find(indexed, "foot_plate", side)
        ankles = _find(indexed, "ankle_bridge", side)
        if not plates or not ankles:
            continue
        fw = _width_x(plates[0])
        ad = _width_x(ankles[0])
        if fw is None or ad is None:
            statuses.append("skip")
            messages.append(f"foot_{side}: missing width/diam fields")
            continue
        delta = abs(fw - ad)
        metrics[f"foot_width_{side}"] = fw
        metrics[f"ankle_diam_{side}"] = ad
        metrics[f"delta_{side}"] = delta
        if delta <= FOOT_WIDTH_TOL_M:
            statuses.append("pass")
            messages.append(f"foot_{side}: |Δ|={delta:.4f} ≤ {FOOT_WIDTH_TOL_M}")
        else:
            statuses.append("fail")
            messages.append(f"foot_{side}: |Δ|={delta:.4f} > {FOOT_WIDTH_TOL_M}")
    if not statuses:
        return _rule("C_foot_width", "skip", "no dual foot/ankle pairs", None)
    if any(s == "fail" for s in statuses):
        return _rule("C_foot_width", "fail", "; ".join(messages), metrics)
    if all(s == "skip" for s in statuses):
        return _rule("C_foot_width", "skip", "; ".join(messages), metrics or None)
    return _rule("C_foot_width", "pass", "; ".join(messages), metrics)


def _hip_outer_x(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]], side: Side
) -> float | None:
    bridges = _find(indexed, "hip_bridge", "none") + _find(indexed, "hip_bridge", side)
    if not bridges:
        # any hip_bridge
        bridges = _find(indexed, "hip_bridge")
    if not bridges:
        return None
    bridge = bridges[0]
    cx = part_x(bridge)
    if cx is None:
        return None
    half = _half_extent_x(bridge)
    if half is None:
        # Try top/bottom half width fields (trap/box bridges)
        half = _half_extent_x(bridge)
    if half is None:
        return cx  # midline bridge without width — weak
    if side == "r":
        return cx + half
    if side == "l":
        return cx - half
    return cx


def _check_outer(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
    *,
    role: ConstraintRole,
    rule_id: str,
) -> ConstraintRuleResult:
    metrics: dict[str, Any] = {}
    statuses: list[RuleStatus] = []
    messages: list[str] = []
    for side in ("l", "r"):
        parts = _find(indexed, role, side)
        if not parts:
            continue
        # 0070: thigh outer measured on full hip→knee chain when split present.
        if role == "thigh":
            outer = _thigh_chain_outer_x(indexed, parts[0], side)  # type: ignore[arg-type]
        else:
            outer = _outer_x(parts[0], side)  # type: ignore[arg-type]
        hip = _hip_outer_x(indexed, side)  # type: ignore[arg-type]
        if outer is None or hip is None:
            statuses.append("skip")
            messages.append(f"{role}_{side}: missing outer/hip X")
            continue
        delta = abs(outer - hip)
        metrics[f"{role}_outer_x_{side}"] = outer
        metrics[f"hip_outer_x_{side}"] = hip
        metrics[f"delta_{side}"] = delta
        if delta <= OUTER_X_TOL_M:
            statuses.append("pass")
            messages.append(f"{role}_{side}: |ΔX|={delta:.4f} ≤ {OUTER_X_TOL_M}")
        else:
            statuses.append("fail")
            messages.append(f"{role}_{side}: |ΔX|={delta:.4f} > {OUTER_X_TOL_M}")
    if not statuses:
        return _rule(rule_id, "skip", f"no {role} sides present", None)
    if any(s == "fail" for s in statuses):
        return _rule(rule_id, "fail", "; ".join(messages), metrics)
    if all(s == "skip" for s in statuses):
        return _rule(rule_id, "skip", "; ".join(messages), metrics or None)
    return _rule(rule_id, "pass", "; ".join(messages), metrics)


def _check_calf_slant(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
) -> ConstraintRuleResult:
    metrics: dict[str, Any] = {}
    statuses: list[RuleStatus] = []
    messages: list[str] = []
    for side in ("l", "r"):
        prox = _find(indexed, "calf_proximal", side)
        dist = _find(indexed, "calf_distal", side)
        whole = _find(indexed, "calf", side)
        if not prox or not dist:
            if whole:
                statuses.append("skip")
                messages.append(f"calf_{side}: whole calf only — slant skipped (B1)")
            continue
        py = part_y(prox[0])
        dy = part_y(dist[0])
        ankles = _find(indexed, "ankle_bridge", side)
        ay = part_y(ankles[0]) if ankles else None
        if py is None or dy is None:
            statuses.append("skip")
            messages.append(f"calf_{side}: missing Y on proximal/distal")
            continue
        metrics[f"prox_y_{side}"] = py
        metrics[f"dist_y_{side}"] = dy
        if ay is not None:
            metrics[f"ankle_y_{side}"] = ay
        mid = 0.5 * (py + (ay if ay is not None else py))
        # Distal should be toward ankle Y, not parked at mid between prox and ...
        # Prefer distal closer to ankle than proximal is (when ankle known).
        if ay is not None:
            if abs(dy - ay) <= abs(py - ay) + 1e-6:
                # distal at least as close to ankle as proximal — good
                # Also refuse if distal is essentially at mid of prox-ankle and
                # not nearer ankle.
                if abs(dy - mid) < abs(dy - ay) and abs(dy - ay) > 0.02:
                    statuses.append("fail")
                    messages.append(
                        f"calf_{side}: distal Y parked mid "
                        f"(dist={dy:.4f}, mid={mid:.4f}, ankle={ay:.4f})"
                    )
                else:
                    statuses.append("pass")
                    messages.append(
                        f"calf_{side}: distal toward ankle (dist={dy:.4f}, ankle={ay:.4f})"
                    )
            else:
                statuses.append("fail")
                messages.append(
                    f"calf_{side}: distal farther from ankle than proximal "
                    f"(dist={dy:.4f}, prox={py:.4f}, ankle={ay:.4f})"
                )
        else:
            # No ankle: just require distal != proximal mid collapse without target
            statuses.append("pass")
            messages.append(f"calf_{side}: split present; no ankle target — soft pass")

    if not statuses:
        return _rule(
            "C_calf_slant",
            "skip",
            "no proximal+distal calf pair (whole calf skips)",
            None,
        )
    if any(s == "fail" for s in statuses):
        return _rule("C_calf_slant", "fail", "; ".join(messages), metrics)
    if all(s == "skip" for s in statuses):
        return _rule("C_calf_slant", "skip", "; ".join(messages), metrics or None)
    return _rule("C_calf_slant", "pass", "; ".join(messages), metrics)


# B18: face/hair/neckline/SCM/fuse softs must not fail C_axial_depth_plane
# (forward of chest_mid by design). Core RECIPE_head / RECIPE_neck still checked.
_AXIAL_EXEMPT_NAME_TOKENS: Final[tuple[str, ...]] = (
    "jaw",
    "brow_soft",
    "eye_soft",
    "nose_soft",
    "ear_soft",
    "lip_soft",
    "hair_mass",
    "neckline",
    "sternomastoid",
    "neck_head_fuse",
)


def _axial_name_exempt(name: str) -> bool:
    lower = name.lower()
    return any(tok in lower for tok in _AXIAL_EXEMPT_NAME_TOKENS)


def _check_axial_depth_plane(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
    report: Any | None,
) -> ConstraintRuleResult:
    chest_mid = _lm_y(report, "chest_mid")
    chest_front = _lm_y(report, "chest_front")
    # Fallback: mean Y of torso parts as mid proxy when no report mid
    if chest_mid is None:
        torsos = _find(indexed, "torso")
        if torsos:
            ys = [part_y(t) for t in torsos if part_y(t) is not None]
            if ys:
                chest_mid = sum(ys) / len(ys)  # type: ignore[assignment]

    metrics: dict[str, Any] = {}
    if chest_mid is not None:
        metrics["chest_mid_y"] = chest_mid
    if chest_front is not None:
        metrics["chest_front_y"] = chest_front

    axial_roles: tuple[ConstraintRole, ...] = (
        "neck",
        "head",
        "torso",
        "shoulder_bridge",
    )
    statuses: list[RuleStatus] = []
    messages: list[str] = []
    for role in axial_roles:
        parts = _find(indexed, role)
        for part in parts:
            # 0028 B18: skip face-kit softs classified as head/neck
            if _axial_name_exempt(part.name):
                metrics[f"{part.name}_axial_exempt"] = True
                continue
            y = part_y(part)
            if y is None:
                continue
            metrics[f"{part.name}_y"] = y
            if chest_mid is None:
                statuses.append("skip")
                messages.append(f"{part.name}: no chest_mid reference")
                continue
            if chest_front is not None:
                # Prefer mid over front: fail if closer to front than mid by margin
                d_mid = abs(y - chest_mid)
                d_front = abs(y - chest_front)
                if d_front + AXIAL_DEPTH_MARGIN_M < d_mid:
                    statuses.append("fail")
                    messages.append(
                        f"{part.name}: Y closer to chest_front "
                        f"({y:.4f}) than chest_mid ({chest_mid:.4f})"
                    )
                else:
                    statuses.append("pass")
                    messages.append(f"{part.name}: Y near chest_mid ({y:.4f} vs {chest_mid:.4f})")
            else:
                if abs(y - chest_mid) <= AXIAL_DEPTH_MARGIN_M * 5:
                    statuses.append("pass")
                    messages.append(f"{part.name}: Y within loose band of mid ({y:.4f})")
                else:
                    # Without front reference, only soft-check
                    statuses.append("pass")
                    messages.append(f"{part.name}: no chest_front; mid-only soft pass")

    if not statuses:
        return _rule(
            "C_axial_depth_plane",
            "skip",
            "no axial parts with Y",
            metrics or None,
        )
    if any(s == "fail" for s in statuses):
        return _rule(
            "C_axial_depth_plane",
            "fail",
            "; ".join(messages),
            metrics,
        )
    if all(s == "skip" for s in statuses):
        return _rule(
            "C_axial_depth_plane",
            "skip",
            "; ".join(messages),
            metrics or None,
        )
    return _rule(
        "C_axial_depth_plane",
        "pass",
        "; ".join(messages),
        metrics,
    )


def _check_soft_gap(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
    *,
    role: ConstraintRole,
    rule_id: str,
    gap_m: float | None,
    gap_name: str,
) -> ConstraintRuleResult:
    left = _find(indexed, role, "l")
    right = _find(indexed, role, "r")
    if not left or not right:
        return _rule(
            rule_id,
            "skip",
            f"missing dual {role} parts",
            None,
        )
    if gap_m is None:
        return _rule(
            rule_id,
            "skip",
            f"missing template {gap_name}",
            None,
        )
    lx = part_x(left[0])
    rx = part_x(right[0])
    if lx is None or rx is None:
        return _rule(rule_id, "skip", f"{role}: missing center X", None)
    gap = abs(rx - lx)
    min_gap = SOFT_GAP_FRAC * float(gap_m)
    metrics = {
        "gap_m": gap,
        "min_gap_m": min_gap,
        "template_gap_m": float(gap_m),
        "left_x": lx,
        "right_x": rx,
    }
    if gap + 1e-12 >= min_gap:
        return _rule(
            rule_id,
            "pass",
            f"{role} gap {gap:.4f} >= {SOFT_GAP_FRAC}x{gap_m:.4f}={min_gap:.4f}",
            metrics,
        )
    return _rule(
        rule_id,
        "fail",
        f"{role} gap {gap:.4f} < {SOFT_GAP_FRAC}x{gap_m:.4f}={min_gap:.4f}",
        metrics,
    )


def _check_no_dup_limb(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
) -> ConstraintRuleResult:
    metrics: dict[str, Any] = {"dup_names": [], "dup_role_side": []}
    messages: list[str] = []
    fail = False

    # *.001 names
    for part, _role, _side in indexed:
        if _BLENDER_SUFFIX_RE.search(part.name):
            fail = True
            metrics["dup_names"].append(part.name)
            messages.append(f"blender suffix name: {part.name}")

    # duplicate RECIPE_limb_* basenames (strip suffix)
    limb_bases: dict[str, list[str]] = {}
    for part, _role, _side in indexed:
        lower = part.name.lower()
        if "recipe_limb_" in lower or part.name.startswith("RECIPE_limb_"):
            base = strip_blender_suffix(part.name)
            limb_bases.setdefault(base, []).append(part.name)
    for base, names in limb_bases.items():
        if len(names) > 1:
            fail = True
            metrics["dup_names"].extend(names)
            messages.append(f"duplicate RECIPE_limb base {base}: {names}")

    # Also catch base + .001 as duplicate of base
    seen_bases: dict[str, str] = {}
    for part, _role, _side in indexed:
        if not ("recipe_limb_" in part.name.lower() or part.name.startswith("RECIPE_limb_")):
            continue
        base = strip_blender_suffix(part.name)
        if base in seen_bases and seen_bases[base] != part.name:
            fail = True
            metrics["dup_names"].extend([seen_bases[base], part.name])
            messages.append(f"duplicate RECIPE_limb {base}: {seen_bases[base]} vs {part.name}")
        else:
            seen_bases[base] = part.name

    # (role, side) uniqueness for limb/foot stack
    # 0027: bicep_soft beads share upper_arm ConstraintRole with limb_segment — exempt.
    rs_map: dict[tuple[str, str], list[str]] = {}
    for part, role, side in indexed:
        if role not in _LIMB_FOOT_ROLES:
            continue
        lower_name = part.name.lower()
        if "bicep" in lower_name and "limb_" not in lower_name:
            continue  # soft bead exempt from no-dup vs limb_segment
        if side == "none" and role not in ("foot_plate", "heel", "ankle_bridge"):
            continue
        key = (role, side)
        rs_map.setdefault(key, []).append(part.name)
    for key, names in rs_map.items():
        if len(names) > 1:
            fail = True
            metrics["dup_role_side"].append({"role": key[0], "side": key[1], "names": names})
            messages.append(f"duplicate (role,side)={key}: {names}")

    if fail:
        return _rule("C_no_dup_limb", "fail", "; ".join(messages) or "duplicates", metrics)
    return _rule("C_no_dup_limb", "pass", "no duplicate limb/foot names", metrics)


def _check_role_classified(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
) -> ConstraintRuleResult:
    """Critical foot-stack name patterns must not remain unknown.

    Fail when a part is still ``unknown`` but its name contains a critical
    foot-stack token (``ank_foot`` / ``heel`` / ``foot`` / ``ankle``). Note
    that normal names like ``RECIPE_foot_plate_*`` classify as ``foot_plate``
    before this rule runs; unknowns that still contain those tokens are
    fail-closed (e.g. garbled names that never matched the ordered classifier).
    """
    critical_substrings = ("ank_foot", "heel", "foot", "ankle")
    bad: list[str] = []
    for part, role, _side in indexed:
        if role != "unknown":
            continue
        lower = part.name.lower()
        if any(s in lower for s in critical_substrings):
            bad.append(part.name)
    if bad:
        return _rule(
            "C_role_classified",
            "fail",
            f"critical foot-stack names left unknown: {bad}",
            {"unknown_critical": bad},
        )
    return _rule(
        "C_role_classified",
        "pass",
        "critical foot-stack names classified",
        None,
    )


def _check_toe_forward_of_heel(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
) -> ConstraintRuleResult:
    """B3: toe mass center Y strictly forward (-Y) of heel on same side."""
    metrics: dict[str, Any] = {}
    statuses: list[RuleStatus] = []
    messages: list[str] = []

    for side in ("l", "r"):
        heels = _find(indexed, "heel", side)
        toe = _find_toe_mass(indexed, side)  # type: ignore[arg-type]
        if not heels or toe is None:
            continue
        heel_y = part_y(heels[0])
        toe_y = part_y(toe)
        if heel_y is None or toe_y is None:
            statuses.append("skip")
            messages.append(f"toe_{side}: missing Y (toe_y={toe_y}, heel_y={heel_y})")
            continue
        threshold = heel_y - TOE_FORWARD_EPS_M
        delta = heel_y - toe_y
        metrics[f"toe_y_{side}"] = toe_y
        metrics[f"heel_y_{side}"] = heel_y
        metrics[f"delta_{side}"] = delta
        metrics[f"threshold_{side}"] = threshold
        if toe_y <= threshold:
            statuses.append("pass")
            messages.append(
                f"toe_{side}: y={toe_y:.4f} heel_y={heel_y:.4f} "
                f"Δ={delta:.4f} ≥ eps={TOE_FORWARD_EPS_M}"
            )
        else:
            statuses.append("fail")
            messages.append(
                f"toe_{side}: y={toe_y:.4f} not forward of heel_y={heel_y:.4f} "
                f"(need ≤ {threshold:.4f}, eps={TOE_FORWARD_EPS_M})"
            )

    if not statuses:
        return _rule(
            "C_toe_forward_of_heel",
            "skip",
            "no heel+toe pairs",
            metrics or None,
        )
    if any(s == "fail" for s in statuses):
        return _rule(
            "C_toe_forward_of_heel",
            "fail",
            "; ".join(messages),
            metrics,
        )
    if all(s == "skip" for s in statuses):
        return _rule(
            "C_toe_forward_of_heel",
            "skip",
            "; ".join(messages),
            metrics or None,
        )
    return _rule(
        "C_toe_forward_of_heel",
        "pass",
        "; ".join(messages),
        metrics,
    )


def _check_heel_reaches_ank_foot(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
) -> ConstraintRuleResult:
    """B4: heel top reaches ankle bottom; heel center strictly below ankle."""
    metrics: dict[str, Any] = {}
    statuses: list[RuleStatus] = []
    messages: list[str] = []

    for side in ("l", "r"):
        heels = _find(indexed, "heel", side)
        ankles = _find(indexed, "ankle_bridge", side)
        if not heels or not ankles:
            continue
        heel = heels[0]
        ank = ankles[0]
        hz = part_z(heel)
        az = part_z(ank)
        heel_rz = part_rz(heel)
        ank_rz = part_rz(ank)
        if hz is None or az is None or heel_rz is None or ank_rz is None:
            statuses.append("skip")
            messages.append(
                f"heel_{side}: missing center Z or rz_m "
                f"(heel_z={hz}, ank_z={az}, heel_rz={heel_rz}, ank_rz={ank_rz})"
            )
            continue
        heel_top = hz + heel_rz
        ank_bottom = az - ank_rz
        gap = heel_top - ank_bottom  # positive = overlap/reach
        metrics[f"heel_z_{side}"] = hz
        metrics[f"ank_z_{side}"] = az
        metrics[f"heel_top_{side}"] = heel_top
        metrics[f"ank_bottom_{side}"] = ank_bottom
        metrics[f"reach_gap_{side}"] = gap
        reach_ok = heel_top >= ank_bottom - HEEL_REACH_GAP_TOL_M
        below_ok = hz < az  # strict — equality is fail (clone class)
        if reach_ok and below_ok:
            statuses.append("pass")
            messages.append(
                f"heel_{side}: top={heel_top:.4f} ank_bottom={ank_bottom:.4f} "
                f"gap={gap:.4f} (tol={HEEL_REACH_GAP_TOL_M}); "
                f"heel.z={hz:.4f} < ank.z={az:.4f}"
            )
        else:
            statuses.append("fail")
            reasons: list[str] = []
            if not reach_ok:
                reasons.append(
                    f"top={heel_top:.4f} below ank_bottom={ank_bottom:.4f} "
                    f"- tol={HEEL_REACH_GAP_TOL_M} (gap={gap:.4f})"
                )
            if not below_ok:
                reasons.append(f"heel.z={hz:.4f} not < ank.z={az:.4f} (clone/equal class)")
            messages.append(f"heel_{side}: " + "; ".join(reasons))

    if not statuses:
        return _rule(
            "C_heel_reaches_ank_foot",
            "skip",
            "no heel+ankle_bridge pairs",
            metrics or None,
        )
    if any(s == "fail" for s in statuses):
        return _rule(
            "C_heel_reaches_ank_foot",
            "fail",
            "; ".join(messages),
            metrics,
        )
    if all(s == "skip" for s in statuses):
        return _rule(
            "C_heel_reaches_ank_foot",
            "skip",
            "; ".join(messages),
            metrics or None,
        )
    return _rule(
        "C_heel_reaches_ank_foot",
        "pass",
        "; ".join(messages),
        metrics,
    )


def _check_toe_sole_z(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
) -> ConstraintRuleResult:
    """B5: toe center is sole-class relative to foot_plate z_top + slack."""
    metrics: dict[str, Any] = {}
    statuses: list[RuleStatus] = []
    messages: list[str] = []

    for side in ("l", "r"):
        plates = _find(indexed, "foot_plate", side)
        toe = _find_toe_mass(indexed, side)  # type: ignore[arg-type]
        if not plates or toe is None:
            continue
        plate_top = _plate_z_top(plates[0])
        tz = part_z(toe)
        if plate_top is None or tz is None:
            statuses.append("skip")
            messages.append(
                f"toe_{side}: missing plate z_top_m or toe Z (plate_top={plate_top}, toe_z={tz})"
            )
            continue
        toe_rz_eff = float(toe.rz_m) if toe.rz_m is not None else 0.0
        slack = _toe_sole_slack_m(toe_rz_eff)
        bound = plate_top + slack
        metrics[f"toe_z_{side}"] = tz
        metrics[f"plate_z_top_{side}"] = plate_top
        metrics[f"toe_rz_eff_{side}"] = toe_rz_eff
        metrics[f"slack_{side}"] = slack
        metrics[f"bound_{side}"] = bound
        if tz <= bound:
            statuses.append("pass")
            messages.append(
                f"toe_{side}: z={tz:.4f} ≤ plate_top={plate_top:.4f} "
                f"+ slack={slack:.4f} (ceil={TOE_SOLE_CEIL_M})"
            )
        else:
            statuses.append("fail")
            messages.append(
                f"toe_{side}: z={tz:.4f} plate_top={plate_top:.4f} "
                f"slack={slack:.4f} → fail sole (bound={bound:.4f})"
            )

    if not statuses:
        return _rule(
            "C_toe_sole_z",
            "skip",
            "no foot_plate+toe pairs",
            metrics or None,
        )
    if any(s == "fail" for s in statuses):
        return _rule("C_toe_sole_z", "fail", "; ".join(messages), metrics)
    if all(s == "skip" for s in statuses):
        return _rule(
            "C_toe_sole_z",
            "skip",
            "; ".join(messages),
            metrics or None,
        )
    return _rule("C_toe_sole_z", "pass", "; ".join(messages), metrics)


def _check_palm_ellipsoid(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
) -> ConstraintRuleResult:
    """B6: when palm parts exist, every palm kind must be ellipsoid."""
    palms = _find_name_contains(indexed, "palm", side=None)
    if not palms:
        return _rule("C_palm_ellipsoid", "skip", "no palm parts", None)
    metrics: dict[str, Any] = {
        "palms": [p.name for p in palms],
        "kinds": {p.name: p.kind for p in palms},
    }
    bad = [p for p in palms if p.kind != "ellipsoid"]
    if bad:
        msgs = [f"palm {p.name}: kind={p.kind} expected=ellipsoid" for p in bad]
        return _rule(
            "C_palm_ellipsoid",
            "fail",
            "; ".join(msgs),
            metrics,
        )
    names = ", ".join(p.name for p in palms)
    return _rule(
        "C_palm_ellipsoid",
        "pass",
        f"all palms ellipsoid ({names})",
        metrics,
    )


def validate_constraints(
    package: BlockoutRecipePackage,
    *,
    report: Any | None = None,
    template_applied: Any | None = None,
) -> ConstraintsReport:
    """Run hard constraint rules; ok=false if any rule fails (skips OK)."""
    indexed = _index_parts(package)
    classified = [{"name": p.name, "role": r, "side": s} for p, r, s in indexed]
    breast_gap = _intermammary_gap_m(template_applied, report)
    glute_gap = _glute_cleft_gap_m(template_applied, report)

    rules = [
        _check_ankle_over_heel(indexed),
        _check_foot_width(indexed),
        _check_outer(indexed, role="thigh", rule_id="C_thigh_outer"),
        _check_outer(indexed, role="glute", rule_id="C_glute_outer"),
        _check_calf_slant(indexed),
        _check_axial_depth_plane(indexed, report),
        _check_soft_gap(
            indexed,
            role="breast",
            rule_id="C_breast_gap",
            gap_m=breast_gap,
            gap_name="intermammary_gap_m",
        ),
        _check_soft_gap(
            indexed,
            role="glute",
            rule_id="C_glute_cleft",
            gap_m=glute_gap,
            gap_name="glute_cleft_m",
        ),
        _check_no_dup_limb(indexed),
        _check_role_classified(indexed),
        _check_toe_forward_of_heel(indexed),
        _check_heel_reaches_ank_foot(indexed),
        _check_toe_sole_z(indexed),
        _check_palm_ellipsoid(indexed),
    ]
    ok = not any(r.status == "fail" for r in rules)
    messages: list[str] = []
    if not ok:
        messages.append("one or more hard constraints failed")
    else:
        messages.append("all hard constraints pass or skip")
    return ConstraintsReport(
        schema_version=CONSTRAINTS_SCHEMA_VERSION,
        honesty=CONSTRAINT_HONESTY,
        ok=ok,
        rules=rules,
        messages=messages,
        classified=classified,
    )


# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------


def _is_frozen(role: ConstraintRole, freeze_feet: bool) -> bool:
    return freeze_feet and role in FREEZE_FEET_ROLES


def _role_weight(role: ConstraintRole, freeze_feet: bool) -> float:
    if _is_frozen(role, freeze_feet):
        return BAND_W_FOOT
    if role == "breast":
        return BAND_W_BREAST
    if role == "glute":
        return BAND_W_GLUTE
    if role == "thigh":
        return BAND_W_THIGH
    if role in ("calf", "calf_proximal", "calf_distal"):
        if role == "calf_distal" and freeze_feet:
            return BAND_W_FOOT
        return BAND_W_CALF
    if role in FREEZE_FEET_ROLES:
        return BAND_W_FOOT
    if role in ("upper_arm", "forearm", "deltoid"):
        return 0.5
    if role in ("torso", "pelvis", "neck", "head", "shoulder_bridge", "hip_bridge"):
        return 0.3
    return 0.2


def _side_or_none(side: Side) -> Side | None:
    """Pass side into ``_find``; ``none`` means any-side lookup."""
    return None if side == "none" else side


def _role_target_y(
    role: ConstraintRole,
    side: Side,
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
    report: Any | None,
    template_applied: Any | None,
) -> float | None:
    """Mild role Y targets for fast optimize (authoring pull, not IK).

    Unanchored roles (hip_bridge, pelvis, arms, …) intentionally return None so
    they stay out of the free set and cannot score-neutral-walk (P1 / 0023).
    """
    side_q = _side_or_none(side)
    if role in ("ankle_bridge", "heel"):
        heels = _find(indexed, "heel", side_q)
        if heels:
            return part_y(heels[0])
        plates = _find(indexed, "foot_plate", side_q)
        if plates:
            py = part_y(plates[0])
            ext = _depth_extent_y(plates[0])
            if py is not None and ext is not None:
                return py + 0.5 * ext  # heel-side
            return py
        return None
    if role == "calf_distal":
        ankles = _find(indexed, "ankle_bridge", side_q)
        if ankles:
            return part_y(ankles[0])
        return None
    if role == "calf_proximal":
        thighs = _find(indexed, "thigh", side_q)
        if thighs:
            return part_y(thighs[0])
        return None
    if role == "thigh":
        # Mid Y of hip_bridge (or pelvis) and ankle_bridge (or foot_plate).
        upper_y: float | None = None
        hips = _find(indexed, "hip_bridge")
        if hips:
            upper_y = part_y(hips[0])
        if upper_y is None:
            pelvis = _find(indexed, "pelvis")
            if pelvis:
                upper_y = part_y(pelvis[0])
        lower_y: float | None = None
        ankles = _find(indexed, "ankle_bridge", side_q)
        if ankles:
            lower_y = part_y(ankles[0])
        if lower_y is None:
            plates = _find(indexed, "foot_plate", side_q)
            if plates:
                lower_y = part_y(plates[0])
        if upper_y is not None and lower_y is not None:
            return 0.5 * (upper_y + lower_y)
        return None
    if role == "calf":
        # Whole calf: mid Y of thigh and ankle_bridge on same side.
        thighs = _find(indexed, "thigh", side_q)
        ankles = _find(indexed, "ankle_bridge", side_q)
        ty = part_y(thighs[0]) if thighs else None
        ay = part_y(ankles[0]) if ankles else None
        if ty is not None and ay is not None:
            return 0.5 * (ty + ay)
        return None
    if role == "breast":
        y = _template_gap_m(template_applied, "breast_y_m")  # reuse helper
        if y is not None:
            return -abs(y)  # breast -Y (B1 0022)
        return _lm_y(report, "chest_front")
    if role == "glute":
        # 0052 B11: sticky seat Y = max(template |y|, dual mean |y|) so optimize
        # does not re-bury emit seat toward bare template glute_y_m.
        template_y: float | None = None
        if template_applied is not None:
            constants = getattr(template_applied, "constants", None)
            if constants is not None:
                gy = getattr(constants, "glute_y_m", None)
                if gy is not None:
                    try:
                        gy_f = abs(float(gy))
                    except (TypeError, ValueError):
                        gy_f = float("nan")
                    if math.isfinite(gy_f):
                        template_y = gy_f
        dual_ys: list[float] = []
        for part, r, _s in indexed:
            if r != "glute":
                continue
            py = part_y(part)
            if py is not None and math.isfinite(float(py)):
                dual_ys.append(abs(float(py)))
        mean_y = sum(dual_ys) / float(len(dual_ys)) if dual_ys else None
        candidates = [x for x in (template_y, mean_y) if x is not None]
        if not candidates:
            return None
        return max(candidates)
    if role in ("neck", "head", "torso", "shoulder_bridge"):
        # B9 (0032): bare `or` treats mid y_m=0.0 as falsy → wrong front target
        mid = _lm_y(report, "chest_mid")
        return mid if mid is not None else _lm_y(report, "chest_front")
    return None


def _band_weighted_free_dof_score(
    package: BlockoutRecipePackage,
    *,
    freeze_feet: bool,
    report: Any | None,
    template_applied: Any | None,
) -> float:
    """C4 band-weighted residual of free recipe part Y/X vs role targets.

    Primary trial ranking driver for both fast and slow optimize. Mesh depth
    deltas are not used here — they are static across recipe-center trials.
    """
    indexed = _index_parts(package)
    total = 0.0
    for part, role, side in indexed:
        w = _role_weight(role, freeze_feet)
        if w <= 0.0:
            continue
        y = part_y(part)
        if y is None:
            continue
        target = _role_target_y(role, side, indexed, report, template_applied)
        if target is None:
            continue
        total += w * abs(y - target)
    # Soft gap penalties (duals) — B7 measured-first for breast and glute
    for role, _field, weight in (
        ("breast", "intermammary_gap_m", BAND_W_BREAST),
        ("glute", "glute_cleft_m", BAND_W_GLUTE),
    ):
        if role == "breast":
            gap_m = _intermammary_gap_m(template_applied, report)
        else:
            gap_m = _glute_cleft_gap_m(template_applied, report)
        left = _find(indexed, role, "l")  # type: ignore[arg-type]
        right = _find(indexed, role, "r")  # type: ignore[arg-type]
        if gap_m is None or not left or not right:
            continue
        lx, rx = part_x(left[0]), part_x(right[0])
        if lx is None or rx is None:
            continue
        min_gap = SOFT_GAP_FRAC * gap_m
        actual = abs(rx - lx)
        if actual < min_gap:
            total += weight * (min_gap - actual)
    return total


def _geometric_score(
    package: BlockoutRecipePackage,
    *,
    freeze_feet: bool,
    report: Any | None,
    template_applied: Any | None,
) -> float:
    """Alias of band-weighted free-DOF score (fast/slow share the same ranking)."""
    return _band_weighted_free_dof_score(
        package,
        freeze_feet=freeze_feet,
        report=report,
        template_applied=template_applied,
    )


def _depth_samples_score(
    report_path: Path | str,
    mesh_path: Path | str,
    *,
    freeze_feet: bool,
    force: bool = True,
) -> float:
    """One-shot mesh-vs-report depth residual (baseline only — not trial ranking).

    Recipe optimize mutates part centers only and does not re-bake mesh, so this
    value is constant across trials. Prefer ``_band_weighted_free_dof_score`` for
    keep/reject ranking.
    """
    from meshops.proportion.analyze import load_report
    from meshops.proportion.depth_samples import (
        compute_mesh_deltas,
        extract_depth_samples,
        load_mesh_for_deltas,
    )

    _ = force  # reserved for future write-through of trial samples
    rep = load_report(report_path)
    package = extract_depth_samples(rep)
    tri = load_mesh_for_deltas(Path(mesh_path))
    deltas_pkg = compute_mesh_deltas(
        package.samples,
        tri,
        mesh_path=str(mesh_path),
        height_m=rep.height_m,
        landmarks_xyz=rep.landmarks_xyz,
    )

    total = 0.0
    for d in deltas_pkg.deltas:
        bid = d.id.lower()
        w = 0.3
        if "breast" in bid:
            w = BAND_W_BREAST
        elif "glute" in bid:
            w = BAND_W_GLUTE
        elif "thigh" in bid:
            w = BAND_W_THIGH
        elif "calf" in bid:
            w = BAND_W_CALF
        elif any(k in bid for k in ("foot", "heel", "ankle")):
            w = BAND_W_FOOT if freeze_feet else 0.2
        dy = d.delta_y_m
        dd = d.delta_depth_m
        if dy is not None:
            total += w * abs(float(dy))
        if dd is not None:
            total += 0.5 * w * abs(float(dd))
    return total


def _has_dual_sides(
    indexed: list[tuple[RecipePart, ConstraintRole, Side]],
    role: ConstraintRole,
) -> bool:
    return bool(_find(indexed, role, "l")) and bool(_find(indexed, role, "r"))


def _free_parts(
    package: BlockoutRecipePackage,
    *,
    freeze_feet: bool,
    report: Any | None = None,
    template_applied: Any | None = None,
) -> list[tuple[RecipePart, ConstraintRole, Side]]:
    """Parts allowed in random Y trials.

    Only include roles that can change score or soft-gap residual:
    - ``_role_target_y`` is not None, or
    - breast/glute with dual L/R present (soft-gap residual path).

    Unscored roles (hip_bridge, pelvis, arms, thigh without anchors, …) are
    excluded so score-neutral random walks cannot drift them (P1 / 0023).
    Outer-X projection still runs on thigh/glute independently of free set.
    """
    free: list[tuple[RecipePart, ConstraintRole, Side]] = []
    indexed = _index_parts(package)
    breast_dual = _has_dual_sides(indexed, "breast")
    glute_dual = _has_dual_sides(indexed, "glute")
    # 0070 B13: freeze split thigh chain when taper_dist sibling present.
    by_name = {p.name: p for p in package.parts}
    for part, role, side in indexed:
        if role == "unknown":
            continue
        if _is_frozen(role, freeze_feet):
            continue
        if part_y(part) is None and part_x(part) is None:
            continue
        if role == "thigh" and side in ("l", "r") and f"RECIPE_thigh_taper_dist_{side}" in by_name:
            continue  # freeze split chain - do not free-DOF Y walk
        target = _role_target_y(role, side, indexed, report, template_applied)
        if target is not None:
            free.append((part, role, side))
            continue
        if role == "breast" and breast_dual:
            free.append((part, role, side))
            continue
        if role == "glute" and glute_dual:
            free.append((part, role, side))
            continue
    return free


def _project_hard_constraints(
    package: BlockoutRecipePackage,
    *,
    freeze_feet: bool,
) -> None:
    """Clamp projection after a step (ankle over heel; outer X; soft gaps)."""
    indexed = _index_parts(package)
    # Ankle → heel / rear foot when not frozen (if frozen, leave alone)
    for side in ("l", "r"):
        ankles = _find(indexed, "ankle_bridge", side)
        if not ankles:
            continue
        if freeze_feet:
            continue
        heels = _find(indexed, "heel", side)
        if heels:
            hy = part_y(heels[0])
            if hy is not None:
                set_part_y(ankles[0], hy)
                continue
        plates = _find(indexed, "foot_plate", side)
        if plates:
            py = part_y(plates[0])
            ext = _depth_extent_y(plates[0])
            if py is not None and ext is not None:
                set_part_y(ankles[0], py + 0.5 * ext)

    # Thigh / glute outer X → hip_bridge outer
    # 0070 B13: skip outer-X for thigh when taper_dist sibling exists (freeze chain).
    by_name = {p.name: p for p, _, _ in indexed}
    for role in ("thigh", "glute"):
        for side in ("l", "r"):
            if (
                role == "thigh"
                and side in ("l", "r")
                and f"RECIPE_thigh_taper_dist_{side}" in by_name
            ):
                continue
            parts = _find(indexed, role, side)  # type: ignore[arg-type]
            if not parts:
                continue
            hip = _hip_outer_x(indexed, side)  # type: ignore[arg-type]
            if hip is None:
                continue
            half = _half_extent_x(parts[0]) or 0.0
            # center so outer tip ≈ hip outer
            if side == "r":
                set_part_x(parts[0], hip - half)
            else:
                set_part_x(parts[0], hip + half)

    # Calf distal toward ankle when split and distal not frozen
    for side in ("l", "r"):
        dist = _find(indexed, "calf_distal", side)
        ankles = _find(indexed, "ankle_bridge", side)
        if not dist or not ankles:
            continue
        if freeze_feet:
            continue
        ay = part_y(ankles[0])
        if ay is not None:
            set_part_y(dist[0], ay)


def optimize_package(
    package: BlockoutRecipePackage,
    *,
    mode: OptimizeMode = "fast",
    freeze_feet: bool = True,
    mesh: Path | str | None = None,
    report: Path | str | None = None,
    template_applied: Any | None = None,
    report_obj: Any | None = None,
) -> tuple[BlockoutRecipePackage, OptimizeResult]:
    """Constrained optimize; mutates a deep copy. Raises ProportionError on refuse."""
    if mode == "slow" and mesh is None:
        raise ProportionError(
            "slow optimize requires --mesh (depth-samples band score)",
            code="optimize_slow_needs_mesh",
        )

    rep = report_obj
    if rep is None and report is not None:
        rep = _optional_report(report)

    work = package.model_copy(deep=True)
    free = _free_parts(
        work,
        freeze_feet=freeze_feet,
        report=rep,
        template_applied=template_applied,
    )
    if not free:
        raise ProportionError(
            "no free DOFs to optimize (all movable parts frozen, missing, or unscored)",
            code="optimize_no_free_dofs",
            details={"freeze_feet": freeze_feet},
        )

    def score_fn(pkg: BlockoutRecipePackage) -> float:
        # Trial ranking is always band-weighted free-DOF residual (C4 weights).
        # Mesh depth deltas are static across recipe-center moves — not used for
        # keep/reject (optionally logged once as baseline below).
        return _band_weighted_free_dof_score(
            pkg,
            freeze_feet=freeze_feet,
            report=rep,
            template_applied=template_applied,
        )

    score_before = float(score_fn(work))
    seed = OPTIMIZE_FAST_SEED if mode == "fast" else OPTIMIZE_SLOW_SEED
    rng = random.Random(seed)
    n_trials = _FAST_N_TRIALS if mode == "fast" else _SLOW_N_TRIALS
    step = _FAST_STEP_M if mode == "fast" else _SLOW_STEP_M
    n_kept = 0
    moved: set[str] = set()
    messages: list[str] = [
        f"mode={mode}",
        f"freeze_feet={freeze_feet}",
        f"free_parts={len(free)}",
        f"seed={seed}",
        "score=band_weighted_free_dof",
    ]
    if mode == "slow":
        # Product contract: mesh required (already enforced). Ranking is free-DOF;
        # optional one-shot mesh baseline is informational only.
        messages.append(
            "score=band_weighted_free_dof (mesh static baseline not used for trial ranking)"
        )
        if report is not None and mesh is not None:
            try:
                baseline = _depth_samples_score(
                    report,
                    mesh,
                    freeze_feet=freeze_feet,
                )
                messages.append(f"mesh_depth_baseline={baseline:.6f} (info only)")
            except Exception as exc:
                messages.append(f"mesh_depth_baseline_skipped: {exc}")

    # Initial projection + mild target pull
    for part, role, side in free:
        target = _role_target_y(role, side, _index_parts(work), rep, template_applied)
        y = part_y(part)
        if target is not None and y is not None:
            # Pull 40% toward target
            new_y = y + 0.4 * (target - y)
            set_part_y(part, new_y)
            moved.add(part.name)
    _project_hard_constraints(work, freeze_feet=freeze_feet)

    best = work.model_copy(deep=True)
    best_score = float(score_fn(best))

    for _ in range(n_trials):
        trial = best.model_copy(deep=True)
        trial_free = _free_parts(
            trial,
            freeze_feet=freeze_feet,
            report=rep,
            template_applied=template_applied,
        )
        if not trial_free:
            break
        part, role, side = trial_free[rng.randrange(len(trial_free))]
        y = part_y(part)
        if y is None:
            continue
        delta = rng.uniform(-step, step)
        # Mild bias toward target
        target = _role_target_y(role, side, _index_parts(trial), rep, template_applied)
        if target is not None:
            delta += 0.25 * max(-step, min(step, target - y))
        set_part_y(part, y + delta)
        _project_hard_constraints(trial, freeze_feet=freeze_feet)
        # Reject if freeze-feet violated (defense in depth)
        if freeze_feet:
            violated = False
            for bp, br, _bs in _index_parts(best):
                if br not in FREEZE_FEET_ROLES:
                    continue
                by = part_y(bp)
                for tp, _tr, _ts in _index_parts(trial):
                    if tp.name == bp.name:
                        ty = part_y(tp)
                        if by is not None and ty is not None and abs(by - ty) > 1e-9:
                            violated = True
                        break
                if violated:
                    break
            if violated:
                continue
        s = float(score_fn(trial))
        # Strict improvement only — refuse score-neutral random walks (P1).
        if s < best_score - 1e-12:
            best = trial
            best_score = s
            n_kept += 1
            moved.add(part.name)

    # Final freeze-feet restore (hard guarantee)
    if freeze_feet:
        orig_by_name = {p.name: p for p in package.parts}
        for part in best.parts:
            role, _ = classify_part_name(part.name)
            if role in FREEZE_FEET_ROLES and part.name in orig_by_name:
                op = orig_by_name[part.name]
                if op.center is not None:
                    part.center = list(op.center)
                if op.p0 is not None:
                    part.p0 = list(op.p0)
                if op.p1 is not None:
                    part.p1 = list(op.p1)

    score_after = float(score_fn(best))
    result = OptimizeResult(
        schema_version=OPTIMIZE_SCHEMA_VERSION,
        honesty=OPTIMIZE_HONESTY,
        mode=mode,
        freeze_feet=freeze_feet,
        score_before=score_before,
        score_after=score_after,
        moved_roles=sorted(moved),
        messages=messages,
        n_trials=n_trials,
        n_kept=n_kept,
    )
    return best, result


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _resolve_json_out(out: Path | str, basename: str) -> Path:
    raw = str(out)
    ends_sep = raw.endswith(("/", "\\"))
    path = Path(raw.rstrip("/\\") if ends_sep else raw)
    if ends_sep or (path.exists() and path.is_dir()) or path.suffix.lower() != ".json":
        return path / basename
    return path


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            raise ProportionError(
                f"output already exists (use --force): {path}",
                code="write_failed",
                details={"path": str(path)},
            )
        path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    except ProportionError:
        raise
    except OSError as exc:
        raise ProportionError(
            f"failed to write constraint/optimize output: {exc}",
            code="constraint_report_failed",
            details={"path": str(path)},
        ) from exc


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def run_blockout_validate_constraints(
    recipe: Path | str,
    out: Path | str,
    *,
    report: Path | str | None = None,
    template_applied: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Validate blockout hard constraints; write constraints_report.json."""
    try:
        package = load_blockout_recipe(recipe)
        rep = _optional_report(report)
        tpl = _optional_template(template_applied)
        constraints = validate_constraints(package, report=rep, template_applied=tpl)
        out_path = _resolve_json_out(out, CONSTRAINTS_REPORT_BASENAME)
        _write_json(
            out_path,
            constraints.model_dump(mode="json"),
            force=force,
        )
    except ProportionError:
        raise
    except Exception as exc:
        raise ProportionError(
            f"constraint report failed: {exc}",
            code="constraint_report_failed",
            details={"recipe": str(recipe)},
        ) from exc

    return {
        "ok": True,
        "constraints_ok": constraints.ok,
        "paths": [str(out_path)],
        "honesty": CONSTRAINT_HONESTY,
        "messages": list(constraints.messages),
        "rules": [
            {"id": r.id, "status": r.status, "message": r.message} for r in constraints.rules
        ],
    }


def run_blockout_optimize(
    recipe: Path | str,
    out: Path | str,
    *,
    mode: str = "fast",
    freeze_feet: bool = True,
    mesh: Path | str | None = None,
    report: Path | str | None = None,
    template_applied: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Constrained optimize; write adjusted recipe + optimize_result.json."""
    mode_n = (mode or "fast").strip().lower()
    if mode_n not in ("fast", "slow"):
        raise ProportionError(
            f"optimize mode must be fast|slow (got {mode!r})",
            code="optimize_failed",
            details={"mode": mode},
        )
    try:
        package = load_blockout_recipe(recipe)
        tpl = _optional_template(template_applied)
        optimized, result = optimize_package(
            package,
            mode=mode_n,  # type: ignore[arg-type]
            freeze_feet=freeze_feet,
            mesh=mesh,
            report=report,
            template_applied=tpl,
        )
        raw = str(out)
        ends_sep = raw.endswith(("/", "\\"))
        out_base = Path(raw.rstrip("/\\") if ends_sep else raw)
        # Directory-style out: write both files under it
        if (
            ends_sep
            or out_base.suffix.lower() != ".json"
            or (out_base.exists() and out_base.is_dir())
        ):
            recipe_path = out_base / OPTIMIZED_RECIPE_BASENAME
            result_path = out_base / OPTIMIZE_RESULT_BASENAME
        else:
            # Single .json → treat as optimize_result; recipe alongside
            result_path = out_base
            recipe_path = out_base.parent / OPTIMIZED_RECIPE_BASENAME

        _write_json(
            recipe_path,
            optimized.model_dump(mode="json"),
            force=force,
        )
        _write_json(
            result_path,
            result.model_dump(mode="json"),
            force=force,
        )
    except ProportionError:
        raise
    except Exception as exc:
        raise ProportionError(
            f"optimize failed: {exc}",
            code="optimize_failed",
            details={"recipe": str(recipe)},
        ) from exc

    return {
        "ok": True,
        "paths": [str(recipe_path), str(result_path)],
        "honesty": OPTIMIZE_HONESTY,
        "mode": result.mode,
        "freeze_feet": result.freeze_feet,
        "score_before": result.score_before,
        "score_after": result.score_after,
        "moved_roles": list(result.moved_roles),
        "messages": list(result.messages),
        "n_trials": result.n_trials,
        "n_kept": result.n_kept,
    }


__all__ = [
    "ANKLE_OVER_HEEL_TOL_M",
    "AXIAL_DEPTH_MARGIN_M",
    "BAND_W_BREAST",
    "BAND_W_CALF",
    "BAND_W_FOOT",
    "BAND_W_GLUTE",
    "BAND_W_THIGH",
    "CONSTRAINTS_REPORT_BASENAME",
    "CONSTRAINTS_SCHEMA_VERSION",
    "FOOT_WIDTH_TOL_M",
    "FREEZE_FEET_ROLES",
    "HEEL_REACH_GAP_TOL_M",
    "OPTIMIZE_FAST_SEED",
    "OPTIMIZE_RESULT_BASENAME",
    "OPTIMIZE_SCHEMA_VERSION",
    "OPTIMIZE_SLOW_SEED",
    "OUTER_X_TOL_M",
    "SOFT_GAP_FRAC",
    "TOE_FORWARD_EPS_M",
    "TOE_SOLE_ABS_M",
    "TOE_SOLE_CEIL_M",
    "TOE_SOLE_RZ_FRAC",
    "ConstraintRole",
    "ConstraintRuleResult",
    "ConstraintsReport",
    "OptimizeResult",
    "Side",
    "classify_part",
    "classify_part_name",
    "optimize_package",
    "part_center_xyz",
    "part_y",
    "part_z",
    "run_blockout_optimize",
    "run_blockout_validate_constraints",
    "side_from_name",
    "strip_blender_suffix",
    "validate_constraints",
]
