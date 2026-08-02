"""Proportion → blockout primitive recipes (track 0019).

Build BlockoutRecipePackage from ProportionReport; emit JSON + Blender 5.2 bpy script.
Authoring layout only — not mesh or print success (Difficulty §12 / N6).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from meshops.proportion.analyze import load_report
from meshops.proportion.errors import ProportionError
from meshops.proportion.guides import AXIS_NOTES, SEED_SEGMENT_MAP
from meshops.proportion.honesty import RECIPE_HONESTY
from meshops.proportion.models import (
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
)

if TYPE_CHECKING:
    from meshops.proportion.depth_samples import DepthSamplesPackage

RECIPE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
RECIPE_ID: Final[Literal["humanoid_a_pose_v1"]] = "humanoid_a_pose_v1"

JSON_BASENAME: Final[str] = "blockout_recipe.json"
BPY_BASENAME: Final[str] = "setup_blockout_recipe.py"

MIDLINE_X_TOL_M: Final[float] = 0.05
CROTCH_Z_FRAC_FALLBACK: Final[float] = 0.5
_NEAR_ZERO_LEN: Final[float] = 1e-9
_GIRAFFE_FRAC: Final[float] = 0.20
_GIRAFFE_ABS_NO_H: Final[float] = 0.35
_MICHELIN_FRAC: Final[float] = 0.45
_CHEST_HALF_DEPTH_FALLBACK_FRAC: Final[float] = 0.12
_HIP_HALF_DEPTH_FALLBACK_FRAC: Final[float] = 0.13

RecipeFormat = Literal["bpy", "json", "both"]
RecipeRole = Literal[
    "torso",
    "pelvis",
    "neck",
    "head",
    "shoulder_bridge",
    "hip_bridge",
    "deltoid_soft",
    "breast_soft",
    "glute_soft",
    "iliac_soft",
    "limb_segment",
]
RecipeKind = Literal["trap_box", "box", "cylinder", "ellipsoid", "capsule"]

_MIDLINE_EXEMPT_ROLES: Final[frozenset[str]] = frozenset({"torso", "pelvis", "neck", "head"})

Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RecipePart(BaseModel):
    """One RECIPE_* primitive (meters)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: RecipeRole
    kind: RecipeKind
    center: list[float] | None = None
    p0: list[float] | None = None
    p1: list[float] | None = None
    top_half_width_m: float | None = None
    bottom_half_width_m: float | None = None
    half_depth_m: float | None = None
    z_bottom_m: float | None = None
    z_top_m: float | None = None
    rx_m: float | None = None
    ry_m: float | None = None
    rz_m: float | None = None
    radius_m: float | None = None
    placement: Literal["full3d", "front_plane"] = "full3d"
    label: str = ""
    notes: str | None = None

    @model_validator(mode="after")
    def _label_recipe_prefix(self) -> RecipePart:
        if not self.label:
            object.__setattr__(self, "label", self.name)
        if not self.label.startswith("RECIPE_"):
            msg = f"recipe label must start with RECIPE_: {self.label!r}"
            raise ValueError(msg)
        if not self.name.startswith("RECIPE_"):
            msg = f"recipe name must start with RECIPE_: {self.name!r}"
            raise ValueError(msg)
        return self


class RecipeMetrics(BaseModel):
    """Resolved input metrics after R4/R5 (audit trail; parts are emit truth)."""

    model_config = ConfigDict(extra="forbid")

    neck_len_m: float | None = None
    shoulder_half_width_m: float | None = None
    hip_half_width_m: float | None = None
    chest_depth_m: float | None = None
    hip_depth_m: float | None = None


class BlockoutRecipePackage(BaseModel):
    """blockout_recipe.json package (schema 1.0.0 only)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = RECIPE_SCHEMA_VERSION
    honesty: str = RECIPE_HONESTY
    source_report_schema: str | None = None
    height_m: float | None = None
    head_unit_m: float | None = None
    axis_notes: str = AXIS_NOTES
    recipe_id: Literal["humanoid_a_pose_v1"] = RECIPE_ID
    parts: list[RecipePart] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    counts: dict[str, Any] = Field(default_factory=dict)
    metrics: RecipeMetrics = Field(default_factory=RecipeMetrics)


# ---------------------------------------------------------------------------
# Diameter / landmark helpers
# ---------------------------------------------------------------------------


def _resolve_diameter(diameters: list[DiameterMeasure], band_id: str) -> DiameterMeasure | None:
    """Prefer view front; first match among remaining."""
    matches = [d for d in diameters if d.band_id == band_id]
    if not matches:
        return None
    front = [d for d in matches if d.view == "front"]
    return front[0] if front else matches[0]


def _half_width_from_diameter(d: DiameterMeasure) -> float | None:
    if d.half_width_m is not None:
        return float(d.half_width_m)
    if d.width_m is not None:
        return float(d.width_m) / 2.0
    return None


def _mean_abs_x_pair(lms: dict[str, LandmarkXYZ], left_id: str, right_id: str) -> float | None:
    left = lms.get(left_id)
    right = lms.get(right_id)
    if left is None or right is None or left.x_m is None or right.x_m is None:
        return None
    return (abs(float(left.x_m)) + abs(float(right.x_m))) / 2.0


def _mean_z(lms: dict[str, LandmarkXYZ], ids: tuple[str, ...]) -> float | None:
    zs: list[float] = []
    for i in ids:
        lm = lms.get(i)
        if lm is not None and lm.z_m is not None:
            zs.append(float(lm.z_m))
    if not zs:
        return None
    return sum(zs) / len(zs)


def _mean_y(lms: dict[str, LandmarkXYZ], ids: tuple[str, ...]) -> float | None:
    ys: list[float] = []
    for i in ids:
        lm = lms.get(i)
        if lm is not None and lm.y_m is not None:
            ys.append(float(lm.y_m))
    if not ys:
        return None
    return sum(ys) / len(ys)


def _segment_length(p0: Vec3, p1: Vec3) -> float:
    return math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2)


def _lm_xyz(lm: LandmarkXYZ, *, y_fallback: float = 0.0) -> tuple[Vec3, bool]:
    """Return (x,y,z) with null y→fallback; front_plane True if y was null."""
    x = 0.0 if lm.x_m is None else float(lm.x_m)
    z = 0.0 if lm.z_m is None else float(lm.z_m)
    if lm.y_m is None:
        return (x, y_fallback, z), True
    return (x, float(lm.y_m), z), False


def _depth_band(report: ProportionReport, band_id: str) -> Any | None:
    for band in report.depth_bands:
        if band.band_id == band_id:
            return band
    return None


def _depth_from_package(
    depth_package: DepthSamplesPackage | None, band_id: str
) -> tuple[float | None, str | None]:
    """Prefer band_{id}_span sample depth_m; else role=band_span + band_id."""
    if depth_package is None:
        return None, None
    want_id = f"band_{band_id}_span"
    for s in depth_package.samples:
        if s.id == want_id and s.depth_m is not None:
            return float(s.depth_m), s.id
    for s in depth_package.samples:
        if s.role == "band_span" and s.band_id == band_id and s.depth_m is not None:
            return float(s.depth_m), s.id
    return None, None


# ---------------------------------------------------------------------------
# Per-kind required fields
# ---------------------------------------------------------------------------


def _validate_part_fields(part: RecipePart) -> None:
    """Raise recipe_failed if kind-required fields are missing."""
    missing: list[str] = []
    kind = part.kind
    if kind == "trap_box":
        for f in (
            "center",
            "top_half_width_m",
            "bottom_half_width_m",
            "half_depth_m",
            "z_bottom_m",
            "z_top_m",
        ):
            if getattr(part, f) is None:
                missing.append(f)
    elif kind == "box":
        for f in ("center", "half_depth_m", "z_bottom_m", "z_top_m"):
            if getattr(part, f) is None:
                missing.append(f)
        if part.top_half_width_m is None and part.bottom_half_width_m is None:
            missing.append("top_half_width_m|bottom_half_width_m")
    elif kind in ("cylinder", "capsule"):
        for f in ("p0", "p1", "radius_m"):
            if getattr(part, f) is None:
                missing.append(f)
    elif kind == "ellipsoid":
        for f in ("center", "rx_m", "ry_m", "rz_m"):
            if getattr(part, f) is None:
                missing.append(f)
    if missing:
        raise ProportionError(
            f"recipe part {part.name!r} kind={kind} missing required fields: {', '.join(missing)}",
            code="recipe_failed",
            details={"name": part.name, "kind": kind, "missing": missing},
        )


# ---------------------------------------------------------------------------
# Metrics resolution (R4-R5)
# ---------------------------------------------------------------------------


class _ResolvedMetrics:
    """Internal resolved dims for part build."""

    def __init__(self) -> None:
        self.height_m: float | None = None
        self.head_unit_m: float | None = None
        self.shoulder_hw: float | None = None
        self.hip_hw: float | None = None
        self.chest_depth_m: float | None = None  # full depth when known
        self.hip_depth_m: float | None = None
        self.chest_half_depth: float | None = None
        self.hip_half_depth: float | None = None
        self.chest_z: float | None = None
        self.shoulder_z: float | None = None
        self.hip_z: float | None = None
        self.neck_len_m: float | None = None
        self.chest_y: float | None = None
        self.hip_y: float | None = None


def _resolve_metrics(
    report: ProportionReport,
    *,
    depth_package: DepthSamplesPackage | None,
    messages: list[str],
) -> _ResolvedMetrics:
    m = _ResolvedMetrics()
    lms = report.landmarks_xyz
    h = report.height_m
    m.height_m = float(h) if h is not None else None
    hu_frac = report.head_unit_frac
    if m.height_m is not None and hu_frac is not None and hu_frac > 0.0:
        m.head_unit_m = float(m.height_m) * float(hu_frac)

    # shoulder_hw
    sh_hw = _mean_abs_x_pair(lms, "shoulder_l", "shoulder_r")
    if sh_hw is not None:
        m.shoulder_hw = sh_hw
    else:
        bust = _resolve_diameter(report.diameters, "bust")
        if bust is not None:
            hw = _half_width_from_diameter(bust)
            if hw is not None:
                m.shoulder_hw = float(hw) * 1.05
                messages.append("shoulder_hw from bust*1.05 (no shoulder_l/r x_m)")
        if m.shoulder_hw is None:
            messages.append(
                "shoulder_hw unavailable (no shoulder x_m, no bust) — RECIPE_torso_trap skipped"
            )

    # hip_hw
    hip_hw = _mean_abs_x_pair(lms, "hip_l", "hip_r")
    if hip_hw is not None:
        m.hip_hw = hip_hw
    else:
        waist = _resolve_diameter(report.diameters, "waist")
        if waist is not None:
            hw = _half_width_from_diameter(waist)
            if hw is not None:
                m.hip_hw = float(hw)
                messages.append("hip_hw from waist")
        if m.hip_hw is None:
            messages.append("hip_hw unavailable (no hip x_m, no waist) — trap/pelvis may skip")

    # chest_depth
    depth_m, sample_id = _depth_from_package(depth_package, "chest")
    if depth_m is not None and sample_id is not None:
        m.chest_depth_m = depth_m
        m.chest_half_depth = depth_m / 2.0
        messages.append(f"depth from depth-at-landmarks:{sample_id}")
    else:
        band = _depth_band(report, "chest")
        if band is not None and band.depth_m is not None:
            m.chest_depth_m = float(band.depth_m)
            m.chest_half_depth = float(band.depth_m) / 2.0
        elif m.height_m is not None:
            m.chest_half_depth = _CHEST_HALF_DEPTH_FALLBACK_FRAC * m.height_m
            m.chest_depth_m = 2.0 * m.chest_half_depth
            messages.append("chest_depth fallback 0.12*H")
        else:
            messages.append("chest_depth unavailable (no depth, no H) — trap may skip")

    # hip_depth
    depth_m, sample_id = _depth_from_package(depth_package, "hip")
    if depth_m is not None and sample_id is not None:
        m.hip_depth_m = depth_m
        m.hip_half_depth = depth_m / 2.0
        messages.append(f"depth from depth-at-landmarks:{sample_id}")
    else:
        band = _depth_band(report, "hip")
        if band is not None and band.depth_m is not None:
            m.hip_depth_m = float(band.depth_m)
            m.hip_half_depth = float(band.depth_m) / 2.0
        elif m.height_m is not None:
            m.hip_half_depth = _HIP_HALF_DEPTH_FALLBACK_FRAC * m.height_m
            m.hip_depth_m = 2.0 * m.hip_half_depth
            messages.append("hip_depth fallback 0.13*H")
        else:
            messages.append("hip_depth unavailable (no depth, no H)")

    m.shoulder_z = _mean_z(lms, ("shoulder_l", "shoulder_r"))
    m.hip_z = _mean_z(lms, ("hip_l", "hip_r"))
    m.chest_y = _mean_y(lms, ("chest_front", "shoulder_l", "shoulder_r"))
    m.hip_y = _mean_y(lms, ("hip_l", "hip_r"))

    # chest_z
    chest_band = _depth_band(report, "chest")
    if chest_band is not None and chest_band.z_frac is not None and m.height_m is not None:
        m.chest_z = float(chest_band.z_frac) * m.height_m
    else:
        cf = lms.get("chest_front")
        if cf is not None and cf.z_m is not None:
            m.chest_z = float(cf.z_m)
        elif m.shoulder_z is not None:
            m.chest_z = m.shoulder_z
            messages.append("chest z unknown — trap top at shoulder z")
        else:
            messages.append("chest_z unavailable")

    # neck length R5
    m.neck_len_m = _resolve_neck_len(lms, m.height_m, messages)

    return m


def _resolve_neck_len(
    lms: dict[str, LandmarkXYZ],
    height_m: float | None,
    messages: list[str],
) -> float | None:
    chin = lms.get("chin")
    z_sh = _mean_z(lms, ("shoulder_l", "shoulder_r"))
    if chin is None or chin.z_m is None or z_sh is None:
        messages.append("neck skipped: missing chin or shoulder z_m")
        return None
    raw = float(chin.z_m) - float(z_sh)
    if raw <= 0.0:
        messages.append("neck_len non-positive — neck skipped")
        return None
    if height_m is not None:
        cap = _GIRAFFE_FRAC * float(height_m)
        if raw > cap:
            messages.append(f"neck_len {raw:.3f}m clamped to {cap:.3f}m (giraffe guard)")
            return cap
        return raw
    # H null: allow up to absolute cap
    if raw > _GIRAFFE_ABS_NO_H:
        messages.append(
            f"neck_len {raw:.3f}m clamped to {_GIRAFFE_ABS_NO_H:.3f}m (giraffe guard, no H)"
        )
        return _GIRAFFE_ABS_NO_H
    return raw


# ---------------------------------------------------------------------------
# Midline filter (R7)
# ---------------------------------------------------------------------------


def _crotch_z(
    report: ProportionReport, height_m: float | None, messages: list[str]
) -> float | None:
    cp = report.landmarks_xyz.get("crotch_pubic")
    if cp is not None and cp.z_m is not None:
        return float(cp.z_m)
    if height_m is not None:
        messages.append("crotch z from 0.5*H fallback (no crotch_pubic)")
        return CROTCH_Z_FRAC_FALLBACK * float(height_m)
    return None


def _midline_blocked(
    center: list[float],
    role: RecipeRole,
    crotch_z: float | None,
) -> bool:
    if role in _MIDLINE_EXEMPT_ROLES:
        return False
    if crotch_z is None:
        return False
    return abs(center[0]) < MIDLINE_X_TOL_M and center[2] < crotch_z


# ---------------------------------------------------------------------------
# Part builders (R6)
# ---------------------------------------------------------------------------


def _append_part(parts: list[RecipePart], part: RecipePart) -> None:
    _validate_part_fields(part)
    parts.append(part)


def _build_torso_trap(m: _ResolvedMetrics, messages: list[str]) -> RecipePart | None:
    if m.shoulder_hw is None or m.hip_hw is None:
        messages.append("RECIPE_torso_trap skipped: need shoulder_hw and hip_hw")
        return None
    if m.hip_z is None:
        messages.append("RECIPE_torso_trap skipped: need hip_z")
        return None
    z_bottom = m.hip_z
    z_candidates = [z for z in (m.shoulder_z, m.chest_z) if z is not None]
    if not z_candidates:
        messages.append("RECIPE_torso_trap skipped: need shoulder_z or chest_z")
        return None
    z_top = max(z_candidates)
    if z_top <= z_bottom:
        messages.append("RECIPE_torso_trap skipped: z_top <= z_bottom")
        return None
    if m.chest_half_depth is None:
        messages.append("RECIPE_torso_trap skipped: need chest half_depth")
        return None
    y = m.chest_y if m.chest_y is not None else 0.0
    placement: Literal["full3d", "front_plane"] = (
        "full3d" if m.chest_y is not None else "front_plane"
    )
    if placement == "front_plane":
        messages.append("RECIPE_torso_trap: y null — front_plane placement")
    z_mid = (z_bottom + z_top) / 2.0
    return RecipePart(
        name="RECIPE_torso_trap",
        role="torso",
        kind="trap_box",
        center=[0.0, y, z_mid],
        top_half_width_m=m.shoulder_hw,
        bottom_half_width_m=m.hip_hw,
        half_depth_m=m.chest_half_depth,
        z_bottom_m=z_bottom,
        z_top_m=z_top,
        placement=placement,
        label="RECIPE_torso_trap",
    )


def _build_pelvis(m: _ResolvedMetrics, messages: list[str]) -> RecipePart | None:
    if m.hip_hw is None:
        messages.append("RECIPE_pelvis_bucket skipped: need hip_hw")
        return None
    if m.hip_z is None:
        messages.append("RECIPE_pelvis_bucket skipped: need hip_z")
        return None
    half_depth = m.hip_half_depth
    if half_depth is None:
        if m.height_m is not None:
            half_depth = _HIP_HALF_DEPTH_FALLBACK_FRAC * m.height_m
        else:
            messages.append("RECIPE_pelvis_bucket skipped: need hip depth or H")
            return None
    # Bucket: from ~crotch area up through hips
    h = m.height_m if m.height_m is not None else 1.7
    z_top = m.hip_z + 0.03 * h
    z_bottom = m.hip_z - 0.12 * h
    if z_bottom < 0.0:
        z_bottom = 0.0
    y = m.hip_y if m.hip_y is not None else 0.0
    placement: Literal["full3d", "front_plane"] = "full3d" if m.hip_y is not None else "front_plane"
    z_mid = (z_bottom + z_top) / 2.0
    return RecipePart(
        name="RECIPE_pelvis_bucket",
        role="pelvis",
        kind="box",
        center=[0.0, y, z_mid],
        top_half_width_m=m.hip_hw * 1.05,
        bottom_half_width_m=m.hip_hw * 1.05,
        half_depth_m=half_depth,
        z_bottom_m=z_bottom,
        z_top_m=z_top,
        placement=placement,
        label="RECIPE_pelvis_bucket",
    )


def _build_neck(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
) -> RecipePart | None:
    if m.neck_len_m is None:
        return None
    z_sh = m.shoulder_z
    if z_sh is None:
        return None
    y = m.chest_y if m.chest_y is not None else 0.0
    diam = _resolve_diameter(report.diameters, "neck")
    radius: float | None = None
    if diam is not None:
        radius = _half_width_from_diameter(diam)
    if radius is None:
        if m.head_unit_m is not None:
            radius = 0.05 * m.head_unit_m
        elif m.height_m is not None:
            radius = 0.04 * m.height_m
        else:
            radius = 0.04
    p0 = [0.0, y, z_sh]
    p1 = [0.0, y, z_sh + m.neck_len_m]
    placement: Literal["full3d", "front_plane"] = (
        "full3d" if m.chest_y is not None else "front_plane"
    )
    return RecipePart(
        name="RECIPE_neck",
        role="neck",
        kind="cylinder",
        p0=p0,
        p1=p1,
        radius_m=radius,
        placement=placement,
        label="RECIPE_neck",
    )


def _build_head(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
) -> RecipePart | None:
    lms = report.landmarks_xyz
    chin = lms.get("chin")
    top = lms.get("cranial_vertex") or lms.get("hair_crown")
    if chin is None or chin.z_m is None:
        messages.append("RECIPE_head skipped: need chin z_m")
        return None
    z_chin = float(chin.z_m)
    if top is not None and top.z_m is not None:
        z_top = float(top.z_m)
    elif m.head_unit_m is not None:
        z_top = z_chin + 0.75 * m.head_unit_m
    elif m.height_m is not None:
        z_top = z_chin + 0.10 * m.height_m
    else:
        messages.append("RECIPE_head skipped: insufficient z for head")
        return None
    if z_top <= z_chin:
        messages.append("RECIPE_head skipped: top z <= chin z")
        return None
    z_c = (z_chin + z_top) / 2.0
    rz = (z_top - z_chin) / 2.0
    # Lateral radius from head diameter or head unit
    diam = _resolve_diameter(report.diameters, "head")
    rx: float | None = None
    if diam is not None:
        rx = _half_width_from_diameter(diam)
    if rx is None:
        if m.head_unit_m is not None:
            rx = 0.40 * m.head_unit_m
        elif m.height_m is not None:
            rx = 0.06 * m.height_m
        else:
            rx = 0.08
    ry = rx * 0.9
    y = 0.0
    if chin.y_m is not None:
        y = float(chin.y_m)
    elif top is not None and top.y_m is not None:
        y = float(top.y_m)
    has_y = chin.y_m is not None or (top is not None and top.y_m is not None)
    placement: Literal["full3d", "front_plane"] = "full3d" if has_y else "front_plane"
    return RecipePart(
        name="RECIPE_head",
        role="head",
        kind="ellipsoid",
        center=[0.0, y, z_c],
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
        placement=placement,
        label="RECIPE_head",
    )


def _build_shoulder_bridges(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
) -> list[RecipePart]:
    parts: list[RecipePart] = []
    lms = report.landmarks_xyz
    if m.shoulder_hw is None or m.shoulder_z is None:
        messages.append("shoulder bridges skipped: need shoulder_hw and shoulder_z")
        return parts
    # Bridge radius
    ua_l = _resolve_diameter(report.diameters, "upper_arm_l")
    ua_r = _resolve_diameter(report.diameters, "upper_arm_r")
    ua_hw_l = _half_width_from_diameter(ua_l) if ua_l else None
    ua_hw_r = _half_width_from_diameter(ua_r) if ua_r else None
    default_r = 0.04 * m.height_m if m.height_m is not None else 0.04
    y_torso = m.chest_y if m.chest_y is not None else 0.0

    for side, lm_id, ua_hw in (
        ("l", "shoulder_l", ua_hw_l),
        ("r", "shoulder_r", ua_hw_r),
    ):
        lm = lms.get(lm_id)
        if lm is None or lm.x_m is None or lm.z_m is None:
            messages.append(f"RECIPE_shoulder_bridge_{side} skipped: missing joint")
            continue
        sx = float(lm.x_m)
        sz = float(lm.z_m)
        sy = float(lm.y_m) if lm.y_m is not None else y_torso
        # Torso side attachment at shoulder_hw * 0.85 toward shoulder
        torso_x = (
            math.copysign(m.shoulder_hw * 0.85, sx)
            if sx != 0.0
            else (m.shoulder_hw * 0.85 if side == "r" else -m.shoulder_hw * 0.85)
        )
        p0 = [torso_x, y_torso, m.shoulder_z]
        p1 = [sx, sy, sz]
        if _segment_length((p0[0], p0[1], p0[2]), (p1[0], p1[1], p1[2])) <= _NEAR_ZERO_LEN:
            messages.append(f"RECIPE_shoulder_bridge_{side} skipped: zero length")
            continue
        # Spec: radius <= 0.55 * upper_arm half-width or 0.04 * H
        radius = min(0.55 * ua_hw, default_r) if ua_hw is not None else default_r
        placement: Literal["full3d", "front_plane"] = (
            "full3d" if lm.y_m is not None and m.chest_y is not None else "front_plane"
        )
        parts.append(
            RecipePart(
                name=f"RECIPE_shoulder_bridge_{side}",
                role="shoulder_bridge",
                kind="cylinder",
                p0=p0,
                p1=p1,
                radius_m=radius,
                placement=placement,
                label=f"RECIPE_shoulder_bridge_{side}",
            )
        )
    return parts


def _build_hip_bridges(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
) -> list[RecipePart]:
    parts: list[RecipePart] = []
    lms = report.landmarks_xyz
    if m.hip_hw is None or m.hip_z is None:
        messages.append("hip bridges skipped: need hip_hw and hip_z")
        return parts
    y_pelvis = m.hip_y if m.hip_y is not None else 0.0
    for side, lm_id, thigh_band in (
        ("l", "hip_l", "thigh_l"),
        ("r", "hip_r", "thigh_r"),
    ):
        lm = lms.get(lm_id)
        if lm is None or lm.x_m is None or lm.z_m is None:
            messages.append(f"RECIPE_hip_bridge_{side} skipped: missing joint")
            continue
        diam = _resolve_diameter(report.diameters, thigh_band)
        thigh_hw = _half_width_from_diameter(diam) if diam else None
        if thigh_hw is None:
            radius = 0.03 * m.height_m if m.height_m is not None else 0.03
        else:
            radius = 0.5 * thigh_hw
        hx = float(lm.x_m)
        hz = float(lm.z_m)
        hy = float(lm.y_m) if lm.y_m is not None else y_pelvis
        pelvis_x = (
            math.copysign(m.hip_hw * 0.7, hx)
            if hx != 0.0
            else (m.hip_hw * 0.7 if side == "r" else -m.hip_hw * 0.7)
        )
        p0 = [pelvis_x, y_pelvis, m.hip_z]
        p1 = [hx, hy, hz]
        if _segment_length((p0[0], p0[1], p0[2]), (p1[0], p1[1], p1[2])) <= _NEAR_ZERO_LEN:
            messages.append(f"RECIPE_hip_bridge_{side} skipped: zero length")
            continue
        placement: Literal["full3d", "front_plane"] = (
            "full3d" if lm.y_m is not None and m.hip_y is not None else "front_plane"
        )
        parts.append(
            RecipePart(
                name=f"RECIPE_hip_bridge_{side}",
                role="hip_bridge",
                kind="cylinder",
                p0=p0,
                p1=p1,
                radius_m=radius,
                placement=placement,
                label=f"RECIPE_hip_bridge_{side}",
            )
        )
    return parts


def _build_deltoids(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
    crotch_z: float | None,
) -> list[RecipePart]:
    parts: list[RecipePart] = []
    if m.shoulder_hw is None:
        messages.append("deltoid softs skipped: shoulder_hw null")
        return parts
    lms = report.landmarks_xyz
    clamp_max = _MICHELIN_FRAC * m.shoulder_hw
    for side, lm_id, band in (
        ("l", "shoulder_l", "upper_arm_l"),
        ("r", "shoulder_r", "upper_arm_r"),
    ):
        lm = lms.get(lm_id)
        if lm is None or lm.x_m is None or lm.z_m is None:
            messages.append(f"RECIPE_deltoid_soft_{side} skipped: missing joint")
            continue
        diam = _resolve_diameter(report.diameters, band)
        base_r = _half_width_from_diameter(diam) if diam else None
        if base_r is None:
            base_r = 0.08 * m.shoulder_hw if m.shoulder_hw else 0.04
        # Soft deltoid slightly larger than arm radius
        measured = base_r * 1.15
        clamped = measured
        if measured >= clamp_max:
            clamped = clamp_max
            messages.append(
                f"deltoid radius {measured:.3f}m clamped to {clamped:.3f}m (Michelin guard)"
            )
        y = float(lm.y_m) if lm.y_m is not None else 0.0
        placement: Literal["full3d", "front_plane"] = (
            "full3d" if lm.y_m is not None else "front_plane"
        )
        center = [float(lm.x_m), y, float(lm.z_m)]
        name = f"RECIPE_deltoid_soft_{side}"
        if _midline_blocked(center, "deltoid_soft", crotch_z):
            messages.append(f"midline below crotch skipped: {name}")
            continue
        parts.append(
            RecipePart(
                name=name,
                role="deltoid_soft",
                kind="ellipsoid",
                center=center,
                rx_m=clamped,
                ry_m=clamped * 0.9,
                rz_m=clamped * 0.85,
                placement=placement,
                label=name,
            )
        )
    return parts


def _build_soft_ovals(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
    crotch_z: float | None,
) -> list[RecipePart]:
    """Breast / glute / iliac soft ellipsoids when CS or depth available."""
    parts: list[RecipePart] = []
    h = m.height_m

    def _cs_level(level_id: str) -> Any | None:
        for cs in report.cross_sections:
            if cs.level_id == level_id:
                return cs
        return None

    # Breast soft (single midline or paired via bust depth)
    breast_cs = _cs_level("bust") or _cs_level("chest") or _cs_level("breast")
    breast_band = _depth_band(report, "breast") or _depth_band(report, "chest")
    bust_diam = _resolve_diameter(report.diameters, "bust")
    if breast_cs is not None and h is not None:
        z_m = float(breast_cs.z_frac) * h
        rx = float(breast_cs.rx_frac) * h * 0.35
        ry = float(breast_cs.ry_frac) * h * 0.5
        rz = max(0.02, (m.head_unit_m or 0.2) * 0.12)
        # paired softs offset from midline
        offset = (m.shoulder_hw or 0.15) * 0.35
        for side, sign in (("l", -1.0), ("r", 1.0)):
            center = [sign * offset, ry * 0.3, z_m]
            name = f"RECIPE_breast_soft_{side}"
            if _midline_blocked(center, "breast_soft", crotch_z):
                messages.append(f"midline below crotch skipped: {name}")
                continue
            parts.append(
                RecipePart(
                    name=name,
                    role="breast_soft",
                    kind="ellipsoid",
                    center=center,
                    rx_m=rx,
                    ry_m=ry,
                    rz_m=rz,
                    placement="full3d",
                    label=name,
                )
            )
    elif breast_band is not None and bust_diam is not None and h is not None:
        hw = _half_width_from_diameter(bust_diam)
        if hw is not None and breast_band.z_frac is not None:
            z_m = float(breast_band.z_frac) * h
            depth = float(breast_band.depth_m) if breast_band.depth_m is not None else 0.08 * h
            offset = hw * 0.45
            rx = hw * 0.25
            ry = depth * 0.35
            rz = 0.04 * h
            for side, sign in (("l", -1.0), ("r", 1.0)):
                center = [sign * offset, float(breast_band.y_mid) * h * 0.1, z_m]
                name = f"RECIPE_breast_soft_{side}"
                if _midline_blocked(center, "breast_soft", crotch_z):
                    messages.append(f"midline below crotch skipped: {name}")
                    continue
                parts.append(
                    RecipePart(
                        name=name,
                        role="breast_soft",
                        kind="ellipsoid",
                        center=center,
                        rx_m=rx,
                        ry_m=ry,
                        rz_m=rz,
                        placement="full3d",
                        label=name,
                    )
                )
    else:
        messages.append("breast softs skipped: no CS/depth/bust")

    # Glute softs
    glute_cs = _cs_level("glute") or _cs_level("hip")
    glute_band = _depth_band(report, "glute") or _depth_band(report, "hip")
    if glute_cs is not None and h is not None:
        z_m = float(glute_cs.z_frac) * h
        rx = float(glute_cs.rx_frac) * h * 0.3
        ry = float(glute_cs.ry_frac) * h * 0.45
        rz = max(0.02, (m.head_unit_m or 0.2) * 0.15)
        offset = (m.hip_hw or 0.12) * 0.45
        for side, sign in (("l", -1.0), ("r", 1.0)):
            center = [sign * offset, -abs(ry) * 0.4, z_m]
            name = f"RECIPE_glute_soft_{side}"
            if _midline_blocked(center, "glute_soft", crotch_z):
                messages.append(f"midline below crotch skipped: {name}")
                continue
            parts.append(
                RecipePart(
                    name=name,
                    role="glute_soft",
                    kind="ellipsoid",
                    center=center,
                    rx_m=rx,
                    ry_m=ry,
                    rz_m=rz,
                    placement="full3d",
                    label=name,
                )
            )
    elif glute_band is not None and m.hip_hw is not None and h is not None:
        if glute_band.z_frac is not None:
            z_m = float(glute_band.z_frac) * h
            depth = float(glute_band.depth_m) if glute_band.depth_m is not None else 0.10 * h
            offset = m.hip_hw * 0.5
            for side, sign in (("l", -1.0), ("r", 1.0)):
                center = [sign * offset, -depth * 0.25, z_m]
                name = f"RECIPE_glute_soft_{side}"
                if _midline_blocked(center, "glute_soft", crotch_z):
                    messages.append(f"midline below crotch skipped: {name}")
                    continue
                parts.append(
                    RecipePart(
                        name=name,
                        role="glute_soft",
                        kind="ellipsoid",
                        center=center,
                        rx_m=m.hip_hw * 0.35,
                        ry_m=depth * 0.3,
                        rz_m=0.05 * h,
                        placement="full3d",
                        label=name,
                    )
                )
    else:
        messages.append("glute softs skipped: no CS/depth/hip_hw")

    # Iliac soft optional
    if m.hip_hw is not None and m.hip_z is not None:
        h_or = h if h is not None else 1.7
        for side, sign in (("l", -1.0), ("r", 1.0)):
            center = [
                sign * m.hip_hw * 0.9,
                m.hip_y if m.hip_y is not None else 0.0,
                m.hip_z + 0.02 * h_or,
            ]
            name = f"RECIPE_iliac_soft_{side}"
            if _midline_blocked(center, "iliac_soft", crotch_z):
                messages.append(f"midline below crotch skipped: {name}")
                continue
            r = m.hip_hw * 0.18
            parts.append(
                RecipePart(
                    name=name,
                    role="iliac_soft",
                    kind="ellipsoid",
                    center=center,
                    rx_m=r,
                    ry_m=r * 0.7,
                    rz_m=r * 0.6,
                    placement="full3d" if m.hip_y is not None else "front_plane",
                    label=name,
                )
            )

    return parts


def _build_limbs(
    report: ProportionReport,
    messages: list[str],
) -> list[RecipePart]:
    """Limb capsules only on SEED_SEGMENT_MAP bands (C2)."""
    parts: list[RecipePart] = []
    lms = report.landmarks_xyz
    skip_count = 0

    for band_id, (p0_id, p1_id) in SEED_SEGMENT_MAP.items():
        if p0_id not in lms or p1_id not in lms:
            messages.append(f"{band_id}: missing joint — limb skipped")
            skip_count += 1
            continue
        lm0 = lms[p0_id]
        lm1 = lms[p1_id]
        if lm0.x_m is None or lm0.z_m is None or lm1.x_m is None or lm1.z_m is None:
            messages.append(f"{band_id}: joint missing meters — limb skipped")
            skip_count += 1
            continue
        diam = _resolve_diameter(report.diameters, band_id)
        radius = _half_width_from_diameter(diam) if diam else None
        if radius is None:
            messages.append(f"{band_id}: no usable radius — limb skipped")
            skip_count += 1
            continue
        y0_null = lm0.y_m is None
        y1_null = lm1.y_m is None
        if y0_null or y1_null:
            ys = [y for y in (lm0.y_m, lm1.y_m) if y is not None]
            y_plane = (sum(ys) / len(ys)) if ys else 0.0
            p0 = [float(lm0.x_m), y_plane, float(lm0.z_m)]
            p1 = [float(lm1.x_m), y_plane, float(lm1.z_m)]
            placement: Literal["full3d", "front_plane"] = "front_plane"
            messages.append(f"{band_id}: y_m null — front_plane limb capsule")
        else:
            p0 = [float(lm0.x_m), float(lm0.y_m), float(lm0.z_m)]  # type: ignore[arg-type]
            p1 = [float(lm1.x_m), float(lm1.y_m), float(lm1.z_m)]  # type: ignore[arg-type]
            placement = "full3d"
        if _segment_length((p0[0], p0[1], p0[2]), (p1[0], p1[1], p1[2])) <= _NEAR_ZERO_LEN:
            messages.append(f"{band_id}: zero-length segment — limb skipped")
            skip_count += 1
            continue
        name = f"RECIPE_limb_{band_id}"
        parts.append(
            RecipePart(
                name=name,
                role="limb_segment",
                kind="capsule",
                p0=p0,
                p1=p1,
                radius_m=radius,
                placement=placement,
                label=name,
            )
        )

    # Cap skip flood: at most 8 segment skip messages already one-per-band
    _ = skip_count  # ≤8 by construction (SEED map size)
    return parts


# ---------------------------------------------------------------------------
# Build package
# ---------------------------------------------------------------------------


def build_blockout_recipe(
    report: ProportionReport,
    *,
    depth_package: DepthSamplesPackage | None = None,
    limbs: bool = True,
) -> BlockoutRecipePackage:
    """Build BlockoutRecipePackage from a loaded ProportionReport.

    Raises ProportionError(code=recipe_empty) when zero parts.
    """
    messages: list[str] = []

    if report.quality.needs_user_input:
        messages.append("quality.needs_user_input: recipe still emitted — confirm primary figure")
    if report.quality.multi_figure:
        messages.append("quality.multi_figure: recipe still emitted — confirm primary figure")

    resolved = _resolve_metrics(report, depth_package=depth_package, messages=messages)
    crotch_z = _crotch_z(report, resolved.height_m, messages)

    parts: list[RecipePart] = []

    # 1 torso
    torso = _build_torso_trap(resolved, messages)
    if torso is not None:
        _append_part(parts, torso)

    # 2 pelvis
    pelvis = _build_pelvis(resolved, messages)
    if pelvis is not None:
        _append_part(parts, pelvis)

    # 3 neck
    neck = _build_neck(report, resolved, messages)
    if neck is not None:
        _append_part(parts, neck)

    # 4 head
    head = _build_head(report, resolved, messages)
    if head is not None:
        _append_part(parts, head)

    # 5-6 shoulder bridges
    for p in _build_shoulder_bridges(report, resolved, messages):
        _append_part(parts, p)

    # 7-8 hip bridges
    for p in _build_hip_bridges(report, resolved, messages):
        _append_part(parts, p)

    # 9-10 deltoids
    for p in _build_deltoids(report, resolved, messages, crotch_z):
        _append_part(parts, p)

    # 11+ breast/glute/iliac
    for p in _build_soft_ovals(report, resolved, messages, crotch_z):
        _append_part(parts, p)

    # 13+ limbs
    if limbs:
        for p in _build_limbs(report, messages):
            _append_part(parts, p)
    else:
        messages.append("--no-limbs: limb_segment parts omitted")

    if not parts:
        raise ProportionError(
            "nothing to emit: zero recipe parts after resolution",
            code="recipe_empty",
        )

    by_role: dict[str, int] = {}
    for p in parts:
        by_role[p.role] = by_role.get(p.role, 0) + 1

    metrics = RecipeMetrics(
        neck_len_m=resolved.neck_len_m,
        shoulder_half_width_m=resolved.shoulder_hw,
        hip_half_width_m=resolved.hip_hw,
        chest_depth_m=resolved.chest_depth_m,
        hip_depth_m=resolved.hip_depth_m,
    )

    return BlockoutRecipePackage(
        schema_version=RECIPE_SCHEMA_VERSION,
        honesty=RECIPE_HONESTY,
        source_report_schema=report.schema_version,
        height_m=resolved.height_m,
        head_unit_m=resolved.head_unit_m,
        axis_notes=AXIS_NOTES,
        recipe_id=RECIPE_ID,
        parts=parts,
        messages=messages,
        counts={"parts": len(parts), "by_role": by_role},
        metrics=metrics,
    )


def load_blockout_recipe(path: Path | str) -> BlockoutRecipePackage:
    """Load blockout_recipe.json; accepts schema 1.0.0 only."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot load blockout recipe: {p}: {exc}",
            code="recipe_failed",
            details={"path": str(p)},
        ) from exc
    ver = data.get("schema_version") if isinstance(data, dict) else None
    if ver != "1.0.0":
        raise ProportionError(
            f"blockout recipe schema_version must be 1.0.0 (got {ver!r})",
            code="recipe_failed",
            details={"path": str(p), "schema_version": ver},
        )
    try:
        return BlockoutRecipePackage.model_validate(data)
    except Exception as exc:
        raise ProportionError(
            f"invalid blockout recipe: {p}: {exc}",
            code="recipe_failed",
            details={"path": str(p)},
        ) from exc


def _load_depth_at_landmarks(path: Path | str) -> DepthSamplesPackage:
    """Lazy-load DepthSamplesPackage from depth_at_landmarks.json (F1)."""
    from meshops.proportion.depth_samples import DepthSamplesPackage

    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return DepthSamplesPackage.model_validate(data)
    except FileNotFoundError as exc:
        raise ProportionError(
            f"depth-at-landmarks file not found: {p}",
            code="recipe_failed",
            details={"path": str(p)},
        ) from exc
    except Exception as exc:
        raise ProportionError(
            f"invalid depth-at-landmarks package: {p}: {exc}",
            code="recipe_failed",
            details={"path": str(p)},
        ) from exc


# ---------------------------------------------------------------------------
# bpy emit (R8)
# ---------------------------------------------------------------------------


def _py_repr(obj: Any) -> str:
    return repr(obj)


def emit_bpy_script(package: BlockoutRecipePackage) -> str:
    """Emit self-contained Blender 5.2 Python script (no meshops imports)."""
    parts_data: list[dict[str, Any]] = []
    for p in package.parts:
        entry: dict[str, Any] = {
            "name": p.name,
            "role": p.role,
            "kind": p.kind,
            "placement": p.placement,
            "label": p.label,
        }
        if p.center is not None:
            entry["center"] = list(p.center)
        if p.p0 is not None:
            entry["p0"] = list(p.p0)
        if p.p1 is not None:
            entry["p1"] = list(p.p1)
        if p.top_half_width_m is not None:
            entry["top_half_width_m"] = p.top_half_width_m
        if p.bottom_half_width_m is not None:
            entry["bottom_half_width_m"] = p.bottom_half_width_m
        if p.half_depth_m is not None:
            entry["half_depth_m"] = p.half_depth_m
        if p.z_bottom_m is not None:
            entry["z_bottom_m"] = p.z_bottom_m
        if p.z_top_m is not None:
            entry["z_top_m"] = p.z_top_m
        if p.rx_m is not None:
            entry["rx_m"] = p.rx_m
        if p.ry_m is not None:
            entry["ry_m"] = p.ry_m
        if p.rz_m is not None:
            entry["rz_m"] = p.rz_m
        if p.radius_m is not None:
            entry["radius_m"] = p.radius_m
        parts_data.append(entry)

    lines: list[str] = [
        "# setup_blockout_recipe.py — MeshOps track 0019",
        f"# honesty: {RECIPE_HONESTY}",
        "# N6 / Difficulty §12: RECIPE primitives are authoring layout only —",
        "# not mesh reconstruction, not print-ready, not hero sculpt success.",
        f"# axis_notes: {AXIS_NOTES}",
        f"# recipe schema_version: {RECIPE_SCHEMA_VERSION}",
        f"# recipe_id: {package.recipe_id}",
        "# MeshOps face -Y: toes -Y, heels +Y. RECIPE only — not final mesh.",
        "# N1: do not whole-model voxel remesh from this script.",
        "",
        "import math",
        "import bpy",
        "from mathutils import Matrix, Vector",
        "",
        "PARTS = " + _py_repr(parts_data),
        f"HONESTY = {_py_repr(RECIPE_HONESTY)}",
        "",
        "# mode safety",
        'if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":',
        '    bpy.ops.object.mode_set(mode="OBJECT")',
        "",
        "# units: only if NONE (never assign scale_length)",
        'if bpy.context.scene.unit_settings.system == "NONE":',
        '    bpy.context.scene.unit_settings.system = "METRIC"',
        '    print("scene units were NONE → set METRIC")',
        "",
        'scale_len = getattr(bpy.context.scene.unit_settings, "scale_length", 1.0) or 1.0',
        "",
        "",
        "def to_bu(x_m, y_m=None, z_m=None):",
        "    if y_m is None and z_m is None:",
        "        # scalar meters → BU",
        "        return x_m / scale_len",
        "    return (x_m / scale_len, y_m / scale_len, z_m / scale_len)",
        "",
        "",
        "def ensure_collection(name):",
        "    col = bpy.data.collections.get(name)",
        "    if col is None:",
        "        col = bpy.data.collections.new(name)",
        "    scene_col = bpy.context.scene.collection",
        "    if col.name not in {c.name for c in scene_col.children}:",
        "        try:",
        "            scene_col.children.link(col)",
        "        except RuntimeError as exc:",
        '            print(f"warning: could not link collection {name!r} under scene: {exc}")',
        "    return col",
        "",
        "",
        "def _link_obj(obj, collection):",
        "    for col in list(obj.users_collection):",
        "        col.objects.unlink(obj)",
        "    if obj.name not in collection.objects:",
        "        try:",
        "            collection.objects.link(obj)",
        "        except RuntimeError as exc:",
        '            print(f"warning: could not link {obj.name!r}: {exc}")',
        "",
        "",
        "def ensure_trap_box(",
        "    name, center_m, top_hw, bottom_hw, half_depth, z_bottom, z_top, collection",
        "):",
        "    # trap_box: 8 corner verts (meters local before BU)",
        "    # Vertex order pin (local m):",
        "    #   0-3 bottom rect at z_bottom (+-bot_hw X, +-hd Y)",
        "    #   4-7 top rect at z_top (+-top_hw X, +-hd Y)",
        "    # faces = 4 sides + bottom + top (6 quads)",
        "    cx, cy, _cz = center_m",
        "    bot = bottom_hw",
        "    top = top_hw",
        "    hd = half_depth",
        "    zb = z_bottom",
        "    zt = z_top",
        "    verts_m = [",
        "        (cx - bot, cy - hd, zb),",
        "        (cx + bot, cy - hd, zb),",
        "        (cx + bot, cy + hd, zb),",
        "        (cx - bot, cy + hd, zb),",
        "        (cx - top, cy - hd, zt),",
        "        (cx + top, cy - hd, zt),",
        "        (cx + top, cy + hd, zt),",
        "        (cx - top, cy + hd, zt),",
        "    ]",
        "    verts = [to_bu(*v) for v in verts_m]",
        "    faces = [",
        "        (0, 1, 2, 3),  # bottom",
        "        (4, 7, 6, 5),  # top",
        "        (0, 4, 5, 1),  # -Y side",
        "        (1, 5, 6, 2),  # +X side",
        "        (2, 6, 7, 3),  # +Y side",
        "        (3, 7, 4, 0),  # -X side",
        "    ]",
        "    obj = bpy.data.objects.get(name)",
        '    if obj is not None and obj.type == "MESH":',
        "        mesh = obj.data",
        "        mesh.clear_geometry()",
        "        mesh.from_pydata(verts, [], faces)",
        "        mesh.update()",
        "        _link_obj(obj, collection)",
        "    else:",
        "        if obj is not None:",
        "            bpy.data.objects.remove(obj, do_unlink=True)",
        "        mesh = bpy.data.meshes.new(name + '_mesh')",
        "        mesh.from_pydata(verts, [], faces)",
        "        mesh.update()",
        "        obj = bpy.data.objects.new(name, mesh)",
        "        collection.objects.link(obj)",
        '    obj["meshops_role"] = "recipe"',
        "    return obj",
        "",
        "",
        "def ensure_box(name, center_m, half_width, half_depth, z_bottom, z_top, collection):",
        "    cx, cy, _ = center_m",
        "    z_mid = (z_bottom + z_top) / 2.0",
        "    sx = max(half_width * 2.0, 1e-6) / scale_len",
        "    sy = max(half_depth * 2.0, 1e-6) / scale_len",
        "    sz = max((z_top - z_bottom), 1e-6) / scale_len",
        "    loc = to_bu(cx, cy, z_mid)",
        "    obj = bpy.data.objects.get(name)",
        '    if obj is None or obj.type != "MESH":',
        "        if obj is not None:",
        "            bpy.data.objects.remove(obj, do_unlink=True)",
        "        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)",
        "        obj = bpy.context.active_object",
        "        obj.name = name",
        "        _link_obj(obj, collection)",
        "    else:",
        "        _link_obj(obj, collection)",
        "        obj.location = loc",
        "    obj.scale = (sx, sy, sz)",
        '    obj["meshops_role"] = "recipe"',
        "    return obj",
        "",
        "",
        "def ensure_cylinder(name, p0_m, p1_m, radius_m, collection):",
        "    p0 = Vector(to_bu(*p0_m))",
        "    p1 = Vector(to_bu(*p1_m))",
        "    v = p1 - p0",
        "    length = v.length",
        "    if length <= 1e-12:",
        "        return None",
        "    midpoint = (p0 + p1) / 2.0",
        "    radius = radius_m / scale_len",
        "    rot = Vector((0.0, 0.0, 1.0)).rotation_difference(v.normalized()).to_4x4()",
        "    mat = (",
        "        Matrix.Translation(midpoint)",
        "        @ rot",
        "        @ Matrix.Scale(radius, 4, (1, 0, 0))",
        "        @ Matrix.Scale(radius, 4, (0, 1, 0))",
        "        @ Matrix.Scale(length / 2.0, 4, (0, 0, 1))",
        "    )",
        "    obj = bpy.data.objects.get(name)",
        '    if obj is None or obj.type != "MESH":',
        "        if obj is not None:",
        "            bpy.data.objects.remove(obj, do_unlink=True)",
        "        bpy.ops.mesh.primitive_cylinder_add(",
        "            radius=1.0, depth=2.0, location=(0.0, 0.0, 0.0)",
        "        )",
        "        obj = bpy.context.active_object",
        "        obj.name = name",
        "        _link_obj(obj, collection)",
        "    else:",
        "        _link_obj(obj, collection)",
        "    obj.matrix_world = mat",
        '    obj["meshops_role"] = "recipe"',
        "    return obj",
        "",
        "",
        "def ensure_ellipsoid(name, center_m, rx_m, ry_m, rz_m, collection):",
        "    cx, cy, cz = to_bu(*center_m)",
        "    sx = rx_m / scale_len",
        "    sy = ry_m / scale_len",
        "    sz = rz_m / scale_len",
        "    mat = (",
        "        Matrix.Translation(Vector((cx, cy, cz)))",
        "        @ Matrix.Scale(sx, 4, (1, 0, 0))",
        "        @ Matrix.Scale(sy, 4, (0, 1, 0))",
        "        @ Matrix.Scale(sz, 4, (0, 0, 1))",
        "    )",
        "    obj = bpy.data.objects.get(name)",
        '    if obj is None or obj.type != "MESH":',
        "        if obj is not None:",
        "            bpy.data.objects.remove(obj, do_unlink=True)",
        "        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 0.0))",
        "        obj = bpy.context.active_object",
        "        obj.name = name",
        "        _link_obj(obj, collection)",
        "    else:",
        "        _link_obj(obj, collection)",
        "    obj.matrix_world = mat",
        '    obj["meshops_role"] = "recipe"',
        "    return obj",
        "",
        "",
        "recipes_col = ensure_collection('Proportion_Recipes')",
        "",
        "n_parts = 0",
        "for p in PARTS:",
        "    kind = p['kind']",
        "    name = p['name']",
        "    if kind == 'trap_box':",
        "        ensure_trap_box(",
        "            name, p['center'], p['top_half_width_m'], p['bottom_half_width_m'],",
        "            p['half_depth_m'], p['z_bottom_m'], p['z_top_m'], recipes_col,",
        "        )",
        "        n_parts += 1",
        "    elif kind == 'box':",
        "        hw = p.get('top_half_width_m') or p.get('bottom_half_width_m') or 0.1",
        "        ensure_box(",
        "            name, p['center'], hw, p['half_depth_m'],",
        "            p['z_bottom_m'], p['z_top_m'], recipes_col,",
        "        )",
        "        n_parts += 1",
        "    elif kind in ('cylinder', 'capsule') and p.get('p0') and p.get('p1'):",
        "        if p.get('radius_m') is not None:",
        "            ensure_cylinder(name, p['p0'], p['p1'], p['radius_m'], recipes_col)",
        "            n_parts += 1",
        "    elif kind == 'ellipsoid' and p.get('center') is not None:",
        "        ensure_ellipsoid(",
        "            name, p['center'], p['rx_m'], p['ry_m'], p['rz_m'], recipes_col",
        "        )",
        "        n_parts += 1",
        "",
        "print(",
        "    f'MeshOps blockout recipe: parts={n_parts} honesty={HONESTY}'",
        ")",
        "print('blockout-recipe only — not mesh or print success')",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Write paths (R1 — copy 0015 table)
# ---------------------------------------------------------------------------


def _is_directory_out(out: Path) -> bool:
    """R1: existing dir OR ends with / or \\ OR no .py/.json suffix → directory."""
    s = str(out)
    if s.endswith(("/", "\\")):
        return True
    if out.exists() and out.is_dir():
        return True
    return out.suffix.lower() not in (".py", ".json")


def write_blockout_recipe(
    out: Path | str,
    package: BlockoutRecipePackage,
    *,
    format: RecipeFormat = "both",
    force: bool = False,
) -> list[Path]:
    """Write recipe JSON and/or bpy script per R1 path resolution.

    May append warn messages to package.messages for single-file + both.
    """
    out_path = Path(out)
    fmt: RecipeFormat = format
    written: list[Path] = []

    is_dir = _is_directory_out(out_path)
    suffix = out_path.suffix.lower()

    if not is_dir:
        if suffix == ".py" and fmt == "json":
            raise ProportionError(
                "--out .py conflicts with --format json",
                code="recipe_failed",
                details={"out": str(out_path), "format": fmt},
            )
        if suffix == ".json" and fmt == "bpy":
            raise ProportionError(
                "--out .json conflicts with --format bpy",
                code="recipe_failed",
                details={"out": str(out_path), "format": fmt},
            )
        if fmt == "both":
            if suffix == ".py":
                package.messages.append("format both with single-file .py — emitting bpy only")
                fmt = "bpy"
            elif suffix == ".json":
                package.messages.append("format both with single-file .json — emitting json only")
                fmt = "json"

    targets: list[tuple[Path, Literal["json", "bpy"]]] = []
    if is_dir:
        directory = out_path
        if fmt in ("json", "both"):
            targets.append((directory / JSON_BASENAME, "json"))
        if fmt in ("bpy", "both"):
            targets.append((directory / BPY_BASENAME, "bpy"))
    else:
        if fmt == "json" or (fmt == "both" and suffix == ".json"):
            targets.append((out_path, "json"))
        elif fmt == "bpy" or (fmt == "both" and suffix == ".py"):
            targets.append((out_path, "bpy"))
        else:
            if suffix == ".json":
                targets.append((out_path, "json"))
            else:
                targets.append((out_path, "bpy"))

    try:
        for path, kind in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not force:
                raise ProportionError(
                    f"output already exists (use --force): {path}",
                    code="write_failed",
                    details={"path": str(path)},
                )
            if kind == "json":
                path.write_text(
                    json.dumps(package.model_dump(mode="json"), indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                path.write_text(emit_bpy_script(package), encoding="utf-8")
            written.append(path)
    except ProportionError:
        raise
    except OSError as exc:
        raise ProportionError(
            f"failed to write blockout recipe: {exc}",
            code="write_failed",
            details={"out": str(out_path)},
        ) from exc

    return written


def run_blockout_recipe(
    report_path: Path | str,
    out: Path | str,
    *,
    format: RecipeFormat = "both",
    depth_at_landmarks: Path | str | None = None,
    limbs: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """CLI helper: load report → build → write; return success payload."""
    report = load_report(report_path)
    depth_pkg: DepthSamplesPackage | None = None
    if depth_at_landmarks is not None:
        depth_pkg = _load_depth_at_landmarks(depth_at_landmarks)
    package = build_blockout_recipe(
        report,
        depth_package=depth_pkg,
        limbs=limbs,
    )
    paths = write_blockout_recipe(out, package, format=format, force=force)
    return {
        "ok": True,
        "format": format,
        "paths": [str(p) for p in paths],
        "counts": dict(package.counts),
        "messages": list(package.messages),
        "neck_len_m": package.metrics.neck_len_m,
    }


__all__ = [
    "AXIS_NOTES",
    "BPY_BASENAME",
    "CROTCH_Z_FRAC_FALLBACK",
    "JSON_BASENAME",
    "MIDLINE_X_TOL_M",
    "RECIPE_HONESTY",
    "RECIPE_ID",
    "RECIPE_SCHEMA_VERSION",
    "BlockoutRecipePackage",
    "RecipeMetrics",
    "RecipePart",
    "build_blockout_recipe",
    "emit_bpy_script",
    "load_blockout_recipe",
    "run_blockout_recipe",
    "write_blockout_recipe",
]
