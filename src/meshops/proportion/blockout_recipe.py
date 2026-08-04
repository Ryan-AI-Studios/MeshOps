"""Proportion → blockout primitive recipes (track 0019 + 0027 profiles).

Build BlockoutRecipePackage from ProportionReport; emit JSON + Blender 5.2 bpy script.
Authoring layout only — not mesh or print success (Difficulty §12 / N6).

0024 soft note: measured cranial/foot depth_bands and foot_len_*_m messages may
override template head/foot scales when present — not wired in v1 (document-only).
neck diameter already preferred when available.
breast_lower* used for rz in 0030; lateral fuse/diameter still 0027.
0030: soft offsets prefer report.soft_spacing measured gaps (B7) over template fracs.
0027: anatomy profiles (--profiles) skip_roles merge + parent_joint; schema 1.1.0.
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
    from meshops.proportion.anatomy_profile import (
        AnatomyProfileDocument,
        ProfilePartSpec,
        ProfileScaleSpec,
    )
    from meshops.proportion.body_template import TemplateAppliedPackage
    from meshops.proportion.depth_samples import DepthSamplesPackage
    from meshops.proportion.skeleton import BlockoutSkeleton, SkeletonJoint

RECIPE_SCHEMA_VERSION: Final[Literal["1.0.0", "1.1.0"]] = "1.1.0"
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
_DEFAULT_WAIST_TAPER: Final[float] = 0.14
_COLUMNAR_WIDTH_RATIO: Final[float] = 0.1

RecipeFormat = Literal["bpy", "json", "both"]
TorsoMode = Literal["trap", "ovals"]
GluteMode = Literal["oval", "two_spheres"]
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
    "trap_soft",
    "pec_soft",
    "scap_soft",
    "bicep_soft",
    "clavicle",
]
RecipeKind = Literal["trap_box", "box", "cylinder", "ellipsoid", "capsule"]

_MIDLINE_EXEMPT_ROLES: Final[frozenset[str]] = frozenset({"torso", "pelvis", "neck", "head"})
# Pre-0027 role set snapshot for B11 (no profile → no trap/pec/scap/bicep/clavicle).
_BASELINE_ROLES_NO_PROFILE: Final[frozenset[str]] = frozenset(
    {
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
    }
)
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
    parent_joint: str | None = None  # 1.1.0 additive; null on 1.0.0 loads

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
    """blockout_recipe.json package (schema 1.0.0 | 1.1.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0", "1.1.0"] = RECIPE_SCHEMA_VERSION
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
    # Bucket: from ~crotch area up through hips (needs H for span — no invent 1.7m)
    if m.height_m is None:
        messages.append("RECIPE_pelvis_bucket skipped: need height_m for pelvis span")
        return None
    h = m.height_m
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
            messages.append(
                "RECIPE_neck skipped: no neck diameter, head_unit_m, or height_m for radius"
            )
            return None
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
            messages.append(
                "RECIPE_head skipped: no head diameter, head_unit_m, or height_m for radius"
            )
            return None
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
    # Spec: radius <= 0.55 * upper_arm half-width or 0.04 * H — no absolute invent
    default_r = 0.04 * m.height_m if m.height_m is not None else None
    y_torso = m.chest_y if m.chest_y is not None else 0.0

    for side, lm_id, ua_hw in (
        ("l", "shoulder_l", ua_hw_l),
        ("r", "shoulder_r", ua_hw_r),
    ):
        lm = lms.get(lm_id)
        if lm is None or lm.x_m is None or lm.z_m is None:
            messages.append(f"RECIPE_shoulder_bridge_{side} skipped: missing joint")
            continue
        if ua_hw is not None and default_r is not None:
            radius = min(0.55 * ua_hw, default_r)
        elif ua_hw is not None:
            radius = 0.55 * ua_hw
        elif default_r is not None:
            radius = default_r
        else:
            messages.append(
                f"RECIPE_shoulder_bridge_{side} skipped: "
                "no upper_arm diameter or height_m for radius"
            )
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
        if thigh_hw is not None:
            radius = 0.5 * thigh_hw
        elif m.height_m is not None:
            radius = 0.03 * m.height_m
        else:
            messages.append(
                f"RECIPE_hip_bridge_{side} skipped: no thigh diameter or height_m for radius"
            )
            continue
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


def _michelin_clamp_max(
    m: _ResolvedMetrics,
    *,
    michelin_cap_frac_h: float | None = None,
) -> float | None:
    """Return Michelin radius cap (meters). Per-part frac_h*H overrides global."""
    if michelin_cap_frac_h is not None and m.height_m is not None:
        return float(michelin_cap_frac_h) * float(m.height_m)
    if michelin_cap_frac_h is not None and m.shoulder_hw is not None:
        return float(michelin_cap_frac_h) * float(m.shoulder_hw)
    if m.shoulder_hw is not None:
        return _MICHELIN_FRAC * float(m.shoulder_hw)
    return None


def _joint_xyz(joint: SkeletonJoint | None) -> list[float] | None:
    """Return [x,y,z] when all three finite; else None."""
    if joint is None:
        return None
    if joint.x_m is None or joint.y_m is None or joint.z_m is None:
        return None
    x, y, z = float(joint.x_m), float(joint.y_m), float(joint.z_m)
    if any(v != v for v in (x, y, z)):  # NaN
        return None
    return [x, y, z]


def _joints_map(skeleton: BlockoutSkeleton | None) -> dict[str, SkeletonJoint]:
    if skeleton is None:
        return {}
    return {j.id: j for j in skeleton.joints}


def _midpoint_of_joints(
    joints: dict[str, SkeletonJoint],
    id_a: str,
    id_b: str,
) -> list[float] | None:
    """Midpoint of two skeleton joints when both have finite xyz (AI2 B4)."""
    pa = _joint_xyz(joints.get(id_a))
    pb = _joint_xyz(joints.get(id_b))
    if pa is None or pb is None:
        return None
    return [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, (pa[2] + pb[2]) / 2.0]


def _lm_midpoint(
    lms: dict[str, LandmarkXYZ],
    id_a: str,
    id_b: str,
) -> list[float] | None:
    """Landmark midpoint when both have finite x,z (y falls back 0)."""
    la = lms.get(id_a)
    lb = lms.get(id_b)
    if la is None or lb is None:
        return None
    if la.x_m is None or la.z_m is None or lb.x_m is None or lb.z_m is None:
        return None
    ya = float(la.y_m) if la.y_m is not None else 0.0
    yb = float(lb.y_m) if lb.y_m is not None else 0.0
    return [
        (float(la.x_m) + float(lb.x_m)) / 2.0,
        (ya + yb) / 2.0,
        (float(la.z_m) + float(lb.z_m)) / 2.0,
    ]


def _side_match_joint_id(cid: str, side: str) -> str:
    """Rewrite L/R joint suffix to match emit side (shoulder_l + side=r → shoulder_r)."""
    if side not in ("l", "r"):
        return cid
    if cid.endswith("_l") or cid.endswith("_r"):
        base = cid[:-2]
        return f"{base}_{side}"
    if cid in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "toe"):
        return f"{cid}_{side}"
    return cid


def _resolve_parent_joint_id(
    preferred: str | None,
    fallbacks: list[str],
    joints: dict[str, SkeletonJoint],
    *,
    side: str,
) -> str | None:
    """Pick first joint id with finite xyz; side-specific ids use emit side.

    Packs may store ``shoulder_l`` on ``side: both``; the emit side rewrites
    wrong-side suffixes so R parts parent to ``shoulder_r`` (D4 / B6).
    """
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(fallbacks)
    expanded: list[str] = []
    for c in candidates:
        if c.endswith("_*") or c.endswith("*"):
            continue
        if side in ("l", "r"):
            sided = _side_match_joint_id(c, side)
            if sided != c:
                expanded.append(sided)
            # Prefer side-suffixed form when base joint missing (shoulder → shoulder_l).
            if not c.endswith(("_l", "_r")):
                maybe = f"{c}_{side}"
                if maybe not in expanded:
                    expanded.append(maybe)
            expanded.append(c)
        else:
            expanded.append(c)
    seen: set[str] = set()
    for cid in expanded:
        if side in ("l", "r"):
            cid = _side_match_joint_id(cid, side)
        if cid in seen:
            continue
        seen.add(cid)
        if _joint_xyz(joints.get(cid)) is not None:
            return cid
    return None


def _build_deltoids(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
    crotch_z: float | None,
    *,
    michelin_cap_frac_h: float | None = None,
) -> list[RecipePart]:
    parts: list[RecipePart] = []
    if m.shoulder_hw is None:
        messages.append("deltoid softs skipped: shoulder_hw null")
        return parts
    lms = report.landmarks_xyz
    clamp_max = _michelin_clamp_max(m, michelin_cap_frac_h=michelin_cap_frac_h)
    if clamp_max is None:
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
    *,
    glute_mode: GluteMode = "oval",
    template_applied: TemplateAppliedPackage | None = None,
    skip_roles: frozenset[str] | set[str] | None = None,
) -> list[RecipePart]:
    """Breast / glute / iliac soft ellipsoids when CS or depth available.

    Body frame (B1): breast center y = -abs(offset) (front -Y);
    glute center y = +abs(offset) (back +Y).
    skip_roles (0027): omit breast_soft / glute_soft when profile owns them.
    """
    parts: list[RecipePart] = []
    skip = frozenset(skip_roles or ())
    h = m.height_m
    soft_y_msg = "soft_y_frame: face=-Y glute=+Y breast=-Y"
    if soft_y_msg not in messages:
        messages.append(soft_y_msg)

    def _cs_level(level_id: str) -> Any | None:
        for cs in report.cross_sections:
            if cs.level_id == level_id:
                return cs
        return None

    # Optional template scales for soft bulk
    breast_ry_scale = 1.0
    breast_rz_scale = 1.0
    breast_y_override: float | None = None
    gap_frac: float | None = None
    template_intermammary_m: float | None = None
    glute_r_override: float | None = None
    glute_y_override: float | None = None
    glute_z_override: float | None = None
    cleft_frac: float | None = None
    template_glute_cleft_m: float | None = None
    if template_applied is not None:
        tc = template_applied.constants
        breast_ry_scale = float(tc.breast_ry_scale)
        breast_rz_scale = float(tc.breast_rz_scale)
        breast_y_override = tc.breast_y_m
        gap_frac = tc.intermammary_gap_frac
        template_intermammary_m = getattr(tc, "intermammary_gap_m", None)
        glute_r_override = tc.glute_r_m
        glute_y_override = tc.glute_y_m
        glute_z_override = tc.glute_z_m
        cleft_frac = tc.glute_cleft_frac
        template_glute_cleft_m = getattr(tc, "glute_cleft_m", None)

    # B7: measured soft_spacing first, then template gap_m, then frac * hw
    measured_breast_gap: float | None = None
    measured_glute_gap: float | None = None
    soft = getattr(report, "soft_spacing", None)
    if soft is not None:
        mg = getattr(soft, "intermammary_gap_m", None)
        if mg is not None:
            try:
                mf = float(mg)
                if mf == mf:  # not NaN
                    measured_breast_gap = mf
            except (TypeError, ValueError):
                pass
        gg = getattr(soft, "glute_cleft_gap_m", None)
        if gg is not None:
            try:
                gf = float(gg)
                if gf == gf:
                    measured_glute_gap = gf
            except (TypeError, ValueError):
                pass

    def _breast_half_gap(bust_hw: float | None, rx: float) -> float | None:
        if measured_breast_gap is not None:
            return measured_breast_gap / 2.0
        if template_intermammary_m is not None:
            return float(template_intermammary_m) / 2.0
        if bust_hw is not None and gap_frac is not None:
            return (gap_frac * bust_hw) / 2.0
        return None

    def _glute_half_gap(hip_hw: float) -> float | None:
        if measured_glute_gap is not None:
            return measured_glute_gap / 2.0
        if template_glute_cleft_m is not None:
            return float(template_glute_cleft_m) / 2.0
        if cleft_frac is not None:
            return (cleft_frac * hip_hw) / 2.0
        return None

    # Breast soft (single midline or paired via bust depth)
    breast_cs = _cs_level("bust") or _cs_level("chest") or _cs_level("breast")
    breast_band = _depth_band(report, "breast") or _depth_band(report, "chest")
    bust_diam = _resolve_diameter(report.diameters, "bust")
    if "breast_soft" in skip:
        messages.append("base breast_soft skipped (profile owns chest)")
    elif breast_cs is not None and h is not None:
        z_m = float(breast_cs.z_frac) * h
        rx = float(breast_cs.rx_frac) * h * 0.35
        ry = float(breast_cs.ry_frac) * h * 0.5 * breast_ry_scale
        rz = max(0.02, (m.head_unit_m or 0.2) * 0.12) * breast_rz_scale
        # paired softs offset from midline; B7 measured gap first
        bust_hw = _half_width_from_diameter(bust_diam) if bust_diam else None
        half_gap = _breast_half_gap(bust_hw, rx)
        if half_gap is not None:
            offset = max(half_gap + rx * 0.5, (m.shoulder_hw or 0.15) * 0.25)
        else:
            offset = (m.shoulder_hw or 0.15) * 0.35
        breast_y = (
            -abs(float(breast_y_override)) if breast_y_override is not None else -abs(ry * 0.3)
        )
        for side, sign in (("l", -1.0), ("r", 1.0)):
            center = [sign * offset, breast_y, z_m]
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
            half_gap = _breast_half_gap(hw, hw * 0.25)
            offset = max(half_gap + hw * 0.12, hw * 0.35) if half_gap is not None else hw * 0.45
            rx = hw * 0.25
            ry = depth * 0.35 * breast_ry_scale
            rz = 0.04 * h * breast_rz_scale
            # Force front -Y (B1); ignore unsigned band.y_mid sign
            breast_y = (
                -abs(float(breast_y_override))
                if breast_y_override is not None
                else -abs(float(breast_band.y_mid) * h * 0.1 if breast_band.y_mid else ry * 0.3)
            )
            if breast_y == 0.0:
                breast_y = -abs(ry * 0.3)
            for side, sign in (("l", -1.0), ("r", 1.0)):
                center = [sign * offset, breast_y, z_m]
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
    elif "breast_soft" not in skip:
        messages.append("breast softs skipped: no CS/depth/bust")

    # Glute softs — oval ellipsoids or two equal-axis spheres
    glute_cs = _cs_level("glute") or _cs_level("hip")
    glute_band = _depth_band(report, "glute") or _depth_band(report, "hip")
    glute_parts_built = False

    if "glute_soft" in skip:
        messages.append("base glute_soft skipped (profile owns glutes)")
    elif glute_mode == "two_spheres":
        # Equal-axis ellipsoids RECIPE_glute_sphere_l/r; centers y > 0 (back).
        # Measured hip_hw / CS radii win over template glute_r (binding scale rule).
        r: float | None = None
        if m.hip_hw is not None and m.hip_hw > 0:
            r = float(m.hip_hw) * 0.55
            if glute_r_override is not None:
                messages.append(
                    "glute two_spheres: measured hip_hw preferred over template glute_r"
                )
        elif glute_cs is not None and h is not None:
            r = float(glute_cs.rx_frac) * h * 0.3
            if glute_r_override is not None:
                messages.append("glute two_spheres: measured CS preferred over template glute_r")
        elif glute_r_override is not None:
            r = float(glute_r_override)
        elif h is not None:
            r = 0.0718 * h  # female seed scale; not inventing absolute meters without H
        if r is not None and (m.hip_z is not None or glute_z_override is not None or h is not None):
            z_m = (
                float(glute_z_override)
                if glute_z_override is not None
                else (
                    float(glute_cs.z_frac) * h
                    if glute_cs is not None and h is not None
                    else (
                        float(glute_band.z_frac) * h
                        if glute_band is not None
                        and glute_band.z_frac is not None
                        and h is not None
                        else (m.hip_z if m.hip_z is not None else 0.5 * (h or 1.0))
                    )
                )
            )
            hip_hw = m.hip_hw or (r * 1.6)
            # Outer toward hip_bridge: center so outer tip ≈ hip_hw; B7 measured first
            half_gap = _glute_half_gap(float(hip_hw))
            offset = half_gap + r if half_gap is not None else max(hip_hw - r, r * 0.9)
            glute_y = abs(float(glute_y_override)) if glute_y_override is not None else abs(r * 0.4)
            for side, sign in (("l", -1.0), ("r", 1.0)):
                center = [sign * offset, glute_y, z_m]
                name = f"RECIPE_glute_sphere_{side}"
                if _midline_blocked(center, "glute_soft", crotch_z):
                    messages.append(f"midline below crotch skipped: {name}")
                    continue
                parts.append(
                    RecipePart(
                        name=name,
                        role="glute_soft",
                        kind="ellipsoid",
                        center=center,
                        rx_m=r,
                        ry_m=r,
                        rz_m=r,
                        placement="full3d",
                        label=name,
                    )
                )
            glute_parts_built = True
        else:
            messages.append("glute two_spheres skipped: need radius and z")

    if "glute_soft" not in skip and not glute_parts_built and glute_mode == "oval":
        if glute_cs is not None and h is not None:
            z_m = (
                float(glute_z_override)
                if glute_z_override is not None
                else float(glute_cs.z_frac) * h
            )
            # Prefer measured CS radii; do not replace with template glute_r.
            rx = float(glute_cs.rx_frac) * h * 0.3
            ry = float(glute_cs.ry_frac) * h * 0.45
            rz = max(0.02, (m.head_unit_m or 0.2) * 0.15)
            if glute_r_override is not None:
                messages.append("glute oval: measured CS radii preferred over template glute_r")
            offset = (m.hip_hw or 0.12) * 0.45
            glute_y = (
                abs(float(glute_y_override)) if glute_y_override is not None else abs(ry) * 0.4
            )
            for side, sign in (("l", -1.0), ("r", 1.0)):
                center = [sign * offset, glute_y, z_m]
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
            glute_parts_built = True
        elif glute_band is not None and m.hip_hw is not None and h is not None:
            if glute_band.z_frac is not None:
                z_m = (
                    float(glute_z_override)
                    if glute_z_override is not None
                    else float(glute_band.z_frac) * h
                )
                depth = float(glute_band.depth_m) if glute_band.depth_m is not None else 0.10 * h
                offset = m.hip_hw * 0.5
                glute_y = (
                    abs(float(glute_y_override))
                    if glute_y_override is not None
                    else abs(depth) * 0.25
                )
                for side, sign in (("l", -1.0), ("r", 1.0)):
                    center = [sign * offset, glute_y, z_m]
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
                glute_parts_built = True

    if "glute_soft" not in skip and not glute_parts_built and glute_mode == "oval":
        # Template prior only when no measured CS/depth/hip path.
        if glute_r_override is not None and h is not None:
            r = float(glute_r_override)
            z_m = (
                float(glute_z_override)
                if glute_z_override is not None
                else (m.hip_z if m.hip_z is not None else 0.5 * h)
            )
            offset = (m.hip_hw or r * 1.6) * 0.45
            glute_y = abs(float(glute_y_override)) if glute_y_override is not None else abs(r) * 0.4
            for side, sign in (("l", -1.0), ("r", 1.0)):
                center = [sign * offset, glute_y, z_m]
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
                        rx_m=r,
                        ry_m=r,
                        rz_m=r,
                        placement="full3d",
                        label=name,
                        notes="template glute_r prior (no measured CS/depth)",
                    )
                )
            glute_parts_built = True
            messages.append("glute oval: template glute_r prior (no measured CS/depth/hip)")
        else:
            messages.append("glute softs skipped: no CS/depth/hip_hw")

    # Iliac soft optional — needs H for z offset (no invent 1.7m)
    if m.hip_hw is not None and m.hip_z is not None and h is not None:
        for side, sign in (("l", -1.0), ("r", 1.0)):
            center = [
                sign * m.hip_hw * 0.9,
                m.hip_y if m.hip_y is not None else 0.0,
                m.hip_z + 0.02 * h,
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
    elif m.hip_hw is not None and m.hip_z is not None and h is None:
        messages.append("iliac softs skipped: need height_m for z offset")

    return parts


def _waist_width_at(
    z_norm: float,
    w_shoulder: float,
    w_hip: float,
    taper: float,
) -> float:
    """W(z) = lerp(W_s, W_h, z) * (1 - taper * sin(π·z)) for z∈[0,1] shoulder→hip."""
    z = max(0.0, min(1.0, z_norm))
    lerp = w_shoulder * (1.0 - z) + w_hip * z
    return lerp * (1.0 - taper * math.sin(math.pi * z))


def _build_torso_ovals(
    m: _ResolvedMetrics,
    messages: list[str],
    *,
    taper: float,
) -> list[RecipePart]:
    """RECIPE_torso_oval_{chest,waist,hip} + RECIPE_pelvis_oval (D6)."""
    parts: list[RecipePart] = []
    if m.shoulder_hw is None or m.hip_hw is None:
        messages.append("torso ovals skipped: need shoulder_hw and hip_hw")
        return parts
    if m.hip_z is None:
        messages.append("torso ovals skipped: need hip_z")
        return parts
    z_candidates = [z for z in (m.shoulder_z, m.chest_z) if z is not None]
    if not z_candidates:
        messages.append("torso ovals skipped: need shoulder_z or chest_z")
        return parts
    z_top = max(z_candidates)
    z_bottom = m.hip_z
    if z_top <= z_bottom:
        messages.append("torso ovals skipped: z_top <= z_bottom")
        return parts
    half_depth = m.chest_half_depth
    if half_depth is None:
        messages.append("torso ovals skipped: need chest half_depth")
        return parts
    y = m.chest_y if m.chest_y is not None else 0.0
    placement: Literal["full3d", "front_plane"] = (
        "full3d" if m.chest_y is not None else "front_plane"
    )
    w_s = m.shoulder_hw
    w_h = m.hip_hw
    # F5 near-columnar
    max_w = max(abs(w_s), abs(w_h), 1e-9)
    if abs(w_s - w_h) / max_w < _COLUMNAR_WIDTH_RATIO:
        messages.append("torso ovals: shoulder≈hip near-columnar")

    span = z_top - z_bottom
    # z_norm 0 at shoulder (top), 1 at hip (bottom)
    layers: list[tuple[str, float]] = [
        ("RECIPE_torso_oval_chest", 0.15),
        ("RECIPE_torso_oval_waist", 0.50),
        ("RECIPE_torso_oval_hip", 0.85),
    ]
    for name, z_norm in layers:
        z_m = z_top - z_norm * span
        hw = _waist_width_at(z_norm, w_s, w_h, taper)
        # Vertical radius ~ 1/6 of span; depth slightly less than half_depth
        rz = max(0.02, span * 0.12)
        ry = half_depth * 0.9
        parts.append(
            RecipePart(
                name=name,
                role="torso",
                kind="ellipsoid",
                center=[0.0, y, z_m],
                rx_m=hw,
                ry_m=ry,
                rz_m=rz,
                placement=placement,
                label=name,
            )
        )

    # Pelvis oval below hip
    hip_half = m.hip_half_depth if m.hip_half_depth is not None else half_depth
    if m.height_m is not None:
        h = m.height_m
        z_pelvis = m.hip_z - 0.04 * h
        if z_pelvis < 0.0:
            z_pelvis = max(0.02, m.hip_z * 0.5)
        y_pelvis = m.hip_y if m.hip_y is not None else y
        p_place: Literal["full3d", "front_plane"] = "full3d" if m.hip_y is not None else placement
        parts.append(
            RecipePart(
                name="RECIPE_pelvis_oval",
                role="pelvis",
                kind="ellipsoid",
                center=[0.0, y_pelvis, z_pelvis],
                rx_m=w_h * 1.05,
                ry_m=hip_half * 0.85,
                rz_m=max(0.03, 0.06 * h),
                placement=p_place,
                label="RECIPE_pelvis_oval",
            )
        )
    else:
        messages.append("RECIPE_pelvis_oval skipped: need height_m")
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
# Profile emit (0027)
# ---------------------------------------------------------------------------


def _profile_skip_roles(profile: AnatomyProfileDocument) -> set[str]:
    """R6.1: roles base emit must not append when profile owns them."""
    from meshops.proportion.anatomy_profile import region_enabled

    skip: set[str] = set()
    if region_enabled(profile, "delts"):
        skip.add("deltoid_soft")
    if region_enabled(profile, "chest"):
        # Clear base mid-chest / dual breast so profile dual breast or pec owns chest.
        skip.add("breast_soft")
    if region_enabled(profile, "glutes"):
        skip.add("glute_soft")
    # hips: do NOT skip iliac_soft (base owns iliac — C7)
    return skip


def _scale_from_frac_h(frac: float | None, height_m: float | None) -> float | None:
    if frac is None or height_m is None:
        return None
    return float(frac) * float(height_m)


def _breast_metrics_rx_ry_rz(
    report: ProportionReport,
) -> tuple[float | None, float | None, float | None]:
    bm = getattr(report, "breast_metrics", None)
    if bm is None:
        return None, None, None
    rxs: list[float] = []
    rys: list[float] = []
    rzs: list[float] = []
    for side_attr in ("left", "right"):
        side = getattr(bm, side_attr, None)
        if side is None:
            continue
        if side.rx_m is not None and side.rx_m == side.rx_m:
            rxs.append(float(side.rx_m))
        if side.ry_m is not None and side.ry_m == side.ry_m:
            rys.append(float(side.ry_m))
        if side.rz_m is not None and side.rz_m == side.rz_m:
            rzs.append(float(side.rz_m))
    rx = sum(rxs) / len(rxs) if rxs else None
    ry = sum(rys) / len(rys) if rys else None
    rz = sum(rzs) / len(rzs) if rzs else None
    return rx, ry, rz


def _breast_slant_deg(
    report: ProportionReport,
    template_applied: TemplateAppliedPackage | None,
) -> float | None:
    """Plan slant only (B9) — never hang."""
    bm = getattr(report, "breast_metrics", None)
    if bm is not None:
        for side_attr in ("left", "right"):
            side = getattr(bm, side_attr, None)
            if side is not None and side.slant_deg is not None:
                return float(side.slant_deg)
    if template_applied is not None:
        return float(getattr(template_applied.constants, "breast_slant_deg", 0.0) or 0.0)
    return None


def _resolve_profile_axes(
    report: ProportionReport,
    m: _ResolvedMetrics,
    scale: ProfileScaleSpec,
    *,
    side: str,
    template_applied: TemplateAppliedPackage | None,
    messages: list[str],
) -> tuple[float, float, float]:
    """B8 scale precedence → (rx, ry, rz) meters."""
    h = m.height_m
    rx: float | None = None
    ry: float | None = None
    rz: float | None = None

    # 1) breast_metrics
    if scale.use_breast_metrics:
        brx, bry, brz = _breast_metrics_rx_ry_rz(report)
        if brx is not None:
            rx = brx
        if bry is not None:
            ry = bry
        if brz is not None:
            rz = brz

    # 2) diameter
    if scale.use_diameter:
        band = scale.use_diameter
        if band == "upper_arm":
            band = f"upper_arm_{side}" if side in ("l", "r") else band
        elif band == "hip":
            band = "waist"  # hip half-width often via waist when no hip diam
        diam = _resolve_diameter(report.diameters, band)
        if diam is None and scale.use_diameter == "hip":
            diam = _resolve_diameter(report.diameters, "hip")
        if diam is None and scale.use_diameter == "bust":
            diam = _resolve_diameter(report.diameters, "bust")
        hw = _half_width_from_diameter(diam) if diam else None
        if hw is not None:
            if rx is None:
                rx = float(hw) * 0.55
            if ry is None:
                ry = float(hw) * 0.45
            if rz is None:
                rz = float(hw) * 0.5

    # 3) depth_band
    if scale.use_depth_band:
        band = _depth_band(report, scale.use_depth_band)
        if band is not None and band.depth_m is not None and ry is None:
            ry = float(band.depth_m) * 0.35

    # 5) *_frac_h * H (soft_spacing gaps applied at placement, not axes)
    if rx is None:
        rx = _scale_from_frac_h(scale.rx_frac_h, h)
    if ry is None:
        ry = _scale_from_frac_h(scale.ry_frac_h, h)
    if rz is None:
        rz = _scale_from_frac_h(scale.rz_frac_h, h)

    # 6) template / hard fallbacks
    if rx is None:
        rx = 0.04 * (h or 1.7)
        messages.append(f"profile scale rx fallback {rx:.4f}m")
    if ry is None:
        ry = 0.03 * (h or 1.7)
    if rz is None:
        rz = 0.035 * (h or 1.7)

    # Michelin cap when set
    if scale.michelin_cap_frac_h is not None:
        cap = _michelin_clamp_max(m, michelin_cap_frac_h=scale.michelin_cap_frac_h)
        if cap is not None:
            for axis_name, val in (("rx", rx), ("ry", ry), ("rz", rz)):
                if val > cap:
                    messages.append(
                        f"profile {axis_name} {val:.3f}m clamped to {cap:.3f}m "
                        f"(michelin_cap_frac_h={scale.michelin_cap_frac_h})"
                    )
            rx = min(rx, cap)
            ry = min(ry, cap)
            rz = min(rz, cap)

    _ = template_applied  # reserved for future template soft scales
    return float(rx), float(ry), float(rz)


def _half_gap_intermammary(
    report: ProportionReport,
    m: _ResolvedMetrics,
    template_applied: TemplateAppliedPackage | None,
) -> float:
    soft = getattr(report, "soft_spacing", None)
    if soft is not None and soft.intermammary_gap_m is not None:
        g = float(soft.intermammary_gap_m)
        if g == g:
            return g / 2.0
    if template_applied is not None:
        tm = getattr(template_applied.constants, "intermammary_gap_m", None)
        if tm is not None:
            return float(tm) / 2.0
        gf = getattr(template_applied.constants, "intermammary_gap_frac", None)
        bust = _resolve_diameter(report.diameters, "bust")
        hw = _half_width_from_diameter(bust) if bust else m.shoulder_hw
        if gf is not None and hw is not None:
            return (float(gf) * float(hw)) / 2.0
    return (m.shoulder_hw or 0.15) * 0.18


def _half_gap_glute(
    report: ProportionReport,
    m: _ResolvedMetrics,
    template_applied: TemplateAppliedPackage | None,
) -> float:
    soft = getattr(report, "soft_spacing", None)
    if soft is not None and soft.glute_cleft_gap_m is not None:
        g = float(soft.glute_cleft_gap_m)
        if g == g:
            return g / 2.0
    if template_applied is not None:
        tm = getattr(template_applied.constants, "glute_cleft_m", None)
        if tm is not None:
            return float(tm) / 2.0
        cf = getattr(template_applied.constants, "glute_cleft_frac", None)
        hip = m.hip_hw or 0.12
        if cf is not None:
            return (float(cf) * float(hip)) / 2.0
    return (m.hip_hw or 0.12) * 0.08


def _emit_profile_parts(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
    crotch_z: float | None,
    *,
    profile: AnatomyProfileDocument,
    skeleton: BlockoutSkeleton | None,
    template_applied: TemplateAppliedPackage | None,
) -> list[RecipePart]:
    """Emit dual softs + trap/pec/scap/bicep/clavicle from profile regions."""
    parts: list[RecipePart] = []
    joints = _joints_map(skeleton)
    lms = report.landmarks_xyz
    h = m.height_m

    messages.append(f"anatomy_profile: id={profile.id}")

    # Hang (B9) — CLI already applied upstream; note slant separately.
    slant = _breast_slant_deg(report, template_applied)
    if slant is not None:
        messages.append(f"breast_slant_deg={slant} (plan only; not hang)")

    for reg in profile.regions:
        if not reg.enabled:
            continue
        if reg.id in ("torso", "hips", "neck"):
            # torso: preferred_torso_mode only; hips: no iliac re-emit; neck: base owns
            continue
        for spec in reg.parts:
            sides: list[str]
            if spec.side == "both":
                sides = ["l", "r"]
            elif spec.side in ("l", "r"):
                sides = [spec.side]
            else:
                sides = ["none"]

            for side in sides:
                emitted = _emit_one_profile_part(
                    report,
                    m,
                    messages,
                    crotch_z,
                    spec=spec,
                    side=side,
                    joints=joints,
                    lms=lms,
                    template_applied=template_applied,
                    height_m=h,
                )
                if emitted is not None:
                    parts.append(emitted)
    return parts


def _emit_one_profile_part(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
    crotch_z: float | None,
    *,
    spec: ProfilePartSpec,
    side: str,
    joints: dict[str, SkeletonJoint],
    lms: dict[str, LandmarkXYZ],
    template_applied: TemplateAppliedPackage | None,
    height_m: float | None,
) -> RecipePart | None:
    role = spec.role  # type: ignore[assignment]
    kind = spec.kind  # type: ignore[assignment]
    rules = set(spec.placement_rules)
    side_tag = side if side in ("l", "r") else "none"
    name = f"RECIPE_{role}" if side_tag == "none" else f"RECIPE_{role}_{side_tag}"

    parent_id = _resolve_parent_joint_id(
        spec.parent_joint_id,
        list(spec.parent_joint_fallback),
        joints,
        side=side_tag if side_tag in ("l", "r") else "none",
    )
    parent_joint: str | None = parent_id
    center: list[float] | None = None
    p0: list[float] | None = None
    p1: list[float] | None = None

    # --- Placement ---
    if role == "trap_soft" and side_tag in ("l", "r"):
        sh_id = f"shoulder_{side_tag}"
        mid = _midpoint_of_joints(joints, "neck_base", sh_id)
        if mid is None:
            # landmark fallback: estimate neck_base at chin/shoulder mid z
            chin = lms.get("chin")
            sh = lms.get(sh_id)
            if chin is not None and sh is not None and sh.x_m is not None and sh.z_m is not None:
                neck_z = (
                    (float(chin.z_m) + float(sh.z_m)) / 2.0
                    if chin.z_m is not None
                    else float(sh.z_m) + 0.02
                )
                mid = [
                    float(sh.x_m) * 0.5,
                    float(sh.y_m) if sh.y_m is not None else 0.0,
                    neck_z,
                ]
            else:
                mid = _lm_midpoint(lms, "chin", sh_id)
        if mid is not None:
            center = list(mid)
            if "y_back_pos" in rules:
                center[1] = abs(center[1]) if center[1] != 0.0 else 0.01 * (height_m or 1.7)
        if parent_joint is None:
            parent_joint = None
            messages.append(f"parent_joint {role} unresolved — using landmark placement")

    elif role == "bicep_soft" and side_tag in ("l", "r"):
        sh_id = f"shoulder_{side_tag}"
        el_id = f"elbow_{side_tag}"
        mid = _midpoint_of_joints(joints, sh_id, el_id)
        if mid is None:
            mid = _lm_midpoint(lms, sh_id, el_id)
        if mid is not None:
            center = list(mid)
        # Prefer side-correct shoulder as parent SoT (D4); mid placement is independent.
        if _joint_xyz(joints.get(sh_id)) is not None:
            parent_joint = sh_id
        elif parent_joint is None:
            messages.append(f"parent_joint {role} unresolved — using landmark placement")

    elif role == "clavicle" and side_tag in ("l", "r"):
        sh_id = f"shoulder_{side_tag}"
        sh_j = _joint_xyz(joints.get(sh_id))
        sp_j = _joint_xyz(joints.get("spine_high")) or _joint_xyz(joints.get("spine_mid"))
        if sh_j is not None and sp_j is not None:
            p0 = list(sh_j)
            p1 = list(sp_j)
        else:
            sh_lm = lms.get(sh_id)
            if sh_lm is not None and sh_lm.x_m is not None and sh_lm.z_m is not None:
                y = float(sh_lm.y_m) if sh_lm.y_m is not None else 0.0
                p0 = [float(sh_lm.x_m), y, float(sh_lm.z_m)]
                # mid-chest toward spine
                p1 = [0.0, y, float(sh_lm.z_m) - 0.02 * (height_m or 1.7)]
                messages.append(f"parent_joint {role} unresolved — using landmark placement")
            else:
                messages.append(f"RECIPE_clavicle_{side_tag} skipped: missing shoulder")
                return None
        # Side-correct shoulder wins over pack left-default parent_joint_id.
        parent_joint = sh_id if sh_j is not None else parent_id

    elif role in ("deltoid_soft",) and side_tag in ("l", "r"):
        sh_id = f"shoulder_{side_tag}"
        jxyz = _joint_xyz(joints.get(sh_id))
        if jxyz is not None:
            center = list(jxyz)
            parent_joint = sh_id
        else:
            lm = lms.get(sh_id)
            if lm is not None and lm.x_m is not None and lm.z_m is not None:
                center = [
                    float(lm.x_m),
                    float(lm.y_m) if lm.y_m is not None else 0.0,
                    float(lm.z_m),
                ]
                messages.append(f"parent_joint {role} unresolved — using landmark placement")
            else:
                messages.append(f"{name} skipped: missing joint")
                return None

    elif role in ("breast_soft", "pec_soft", "scap_soft", "glute_soft"):
        # Anchor at spine_high / pelvis then offset L/R + Y rule
        if role == "glute_soft":
            anchor = _joint_xyz(joints.get("pelvis"))
            if anchor is None and m.hip_z is not None:
                anchor = [0.0, m.hip_y or 0.0, m.hip_z]
        else:
            anchor = _joint_xyz(joints.get("spine_high")) or _joint_xyz(joints.get("spine_mid"))
            if anchor is None and m.chest_z is not None:
                anchor = [0.0, m.chest_y or 0.0, m.chest_z]
            elif anchor is None and m.shoulder_z is not None:
                anchor = [0.0, m.chest_y or 0.0, m.shoulder_z]
        if anchor is None:
            messages.append(f"parent_joint {role} unresolved — using landmark placement")
            z0 = m.hip_z if role == "glute_soft" else (m.chest_z or m.shoulder_z or 1.2)
            if z0 is None:
                messages.append(f"{name} skipped: no z anchor")
                return None
            anchor = [0.0, 0.0, float(z0)]
        center = list(anchor)
        if parent_joint is None:
            messages.append(f"parent_joint {role} unresolved — using landmark placement")
    else:
        if parent_id is not None:
            jxyz = _joint_xyz(joints.get(parent_id))
            if jxyz is not None:
                center = list(jxyz)
        if center is None and kind != "capsule":
            messages.append(f"parent_joint {role} unresolved — using landmark placement")
            messages.append(f"{name} skipped: no center")
            return None

    # Axes / radius
    rx = ry = rz = 0.0
    radius_m: float | None = None
    if kind == "capsule" or kind == "cylinder":
        rfrac = spec.scale.radius_frac_h
        if rfrac is not None and height_m is not None:
            radius_m = float(rfrac) * float(height_m)
        else:
            radius_m = 0.006 * (height_m or 1.7)
    else:
        rx, ry, rz = _resolve_profile_axes(
            report,
            m,
            spec.scale,
            side=side_tag if side_tag in ("l", "r") else "none",
            template_applied=template_applied,
            messages=messages,
        )

    # Lateral dual gap offsets
    if center is not None and side_tag in ("l", "r") and "dual_lr" in rules:
        sign = -1.0 if side_tag == "l" else 1.0
        if "gap_intermammary" in rules or role in ("breast_soft", "pec_soft"):
            half = _half_gap_intermammary(report, m, template_applied)
            offset = half + max(rx, 0.02) * 0.55
            center[0] = sign * offset
        elif "gap_glute_cleft" in rules or role == "glute_soft":
            half = _half_gap_glute(report, m, template_applied)
            offset = half + max(rx, 0.02) * 0.55
            center[0] = sign * offset
        elif role in ("trap_soft", "scap_soft", "deltoid_soft", "bicep_soft"):
            # keep mid/joint x; mild lateral bias for traps/scap if near zero
            if abs(center[0]) < 1e-6:
                lat = (m.shoulder_hw or 0.15) * (0.35 if role == "trap_soft" else 0.45)
                center[0] = sign * lat
        else:
            if abs(center[0]) < 1e-6:
                center[0] = sign * (m.shoulder_hw or 0.15) * 0.3

    # Soft Y frame (B8)
    if center is not None:
        if "y_front_neg" in rules or role in ("breast_soft", "pec_soft"):
            mag = abs(center[1]) if center[1] != 0.0 else abs(ry) * 0.35
            if template_applied is not None and role == "breast_soft":
                by = template_applied.constants.breast_y_m
                if by is not None:
                    mag = abs(float(by))
            center[1] = -abs(mag) if mag != 0.0 else -abs(ry) * 0.3
        if "y_back_pos" in rules or role in ("glute_soft", "scap_soft", "trap_soft"):
            mag = abs(center[1]) if center[1] != 0.0 else abs(ry) * 0.4
            if template_applied is not None and role == "glute_soft":
                gy = template_applied.constants.glute_y_m
                if gy is not None:
                    mag = abs(float(gy))
            center[1] = abs(mag) if mag != 0.0 else abs(ry) * 0.35

    if (
        center is not None
        and role
        in (
            "deltoid_soft",
            "breast_soft",
            "glute_soft",
            "trap_soft",
            "pec_soft",
            "scap_soft",
            "bicep_soft",
        )
        and role not in _MIDLINE_EXEMPT_ROLES
        and _midline_blocked(center, "deltoid_soft", crotch_z)
    ):
        messages.append(f"midline below crotch skipped: {name}")
        return None

    if kind in ("capsule", "cylinder"):
        if p0 is None or p1 is None or radius_m is None:
            messages.append(f"{name} skipped: missing capsule endpoints/radius")
            return None
        if _segment_length((p0[0], p0[1], p0[2]), (p1[0], p1[1], p1[2])) < _NEAR_ZERO_LEN:
            messages.append(f"{name} skipped: zero length")
            return None
        return RecipePart(
            name=name,
            role=role,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            p0=p0,
            p1=p1,
            radius_m=radius_m,
            placement="full3d",
            label=name,
            parent_joint=parent_joint,
            notes=spec.notes,
        )

    if center is None:
        messages.append(f"{name} skipped: no center")
        return None
    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        center=center,
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
        placement="full3d",
        label=name,
        parent_joint=parent_joint,
        notes=spec.notes,
    )


# ---------------------------------------------------------------------------
# Build package
# ---------------------------------------------------------------------------


def build_blockout_recipe(
    report: ProportionReport,
    *,
    depth_package: DepthSamplesPackage | None = None,
    limbs: bool = True,
    torso: TorsoMode = "trap",
    glute: GluteMode = "oval",
    nofuse: bool = False,
    breast_tilt_deg: float | None = None,
    template_applied: TemplateAppliedPackage | None = None,
    profile: AnatomyProfileDocument | None = None,
    skeleton: BlockoutSkeleton | None = None,
) -> BlockoutRecipePackage:
    """Build BlockoutRecipePackage from a loaded ProportionReport.

    Raises ProportionError(code=recipe_empty) when zero parts.
    Topology modes only in messages (C1) — not in counts.
    When *profile* set: skip_roles merge (R6.1) + profile dual softs / new roles.
    """
    messages: list[str] = []

    if report.quality.needs_user_input:
        messages.append("quality.needs_user_input: recipe still emitted — confirm primary figure")
    if report.quality.multi_figure:
        messages.append("quality.multi_figure: recipe still emitted — confirm primary figure")

    skip_roles: set[str] = set()
    torso_mode: TorsoMode = torso
    glute_mode: GluteMode = glute
    if profile is not None:
        from meshops.proportion.anatomy_profile import region_by_id

        skip_roles = _profile_skip_roles(profile)
        torso_reg = region_by_id(profile, "torso")
        if (
            torso_reg is not None
            and torso_reg.enabled
            and torso_reg.preferred_torso_mode is not None
        ):
            torso_mode = torso_reg.preferred_torso_mode
            messages.append(f"preferred_torso_mode from profile: {torso_mode}")
        # Profile glutes enabled → dual wins over CLI --glute oval (AI2 B7)
        glute_reg = region_by_id(profile, "glutes")
        if glute_reg is not None and glute_reg.enabled:
            glute_mode = "two_spheres"
            messages.append("profile glutes enabled: dual glute (wins over CLI glute mode)")

    # Mode messages (C1)
    messages.append(f"torso_mode={torso_mode}")
    messages.append(f"glute_mode={glute_mode}")
    messages.append(f"nofuse={str(bool(nofuse)).lower()}")
    if nofuse:
        messages.append("nofuse=true: no join/boolean (emit layout only)")
    if skip_roles:
        messages.append(f"skip_roles={sorted(skip_roles)}")

    if template_applied is not None:
        messages.append(f"template_applied: id={template_applied.template_id}")

    # Breast hang (B9): CLI → template tilt — never slant (C2 message-only hang)
    tilt_val: float | None = breast_tilt_deg
    if tilt_val is None and template_applied is not None:
        tilt_val = float(template_applied.constants.breast_tilt_x_deg)
    if tilt_val is not None:
        messages.append(f"breast_tilt_deg={tilt_val}")
        messages.append("breast_tilt_applied: false")

    resolved = _resolve_metrics(report, depth_package=depth_package, messages=messages)
    crotch_z = _crotch_z(report, resolved.height_m, messages)

    # Waist taper from template or default
    taper = _DEFAULT_WAIST_TAPER
    if template_applied is not None:
        taper = float(template_applied.constants.torso_waist_taper)

    parts: list[RecipePart] = []

    # 1-2 torso + pelvis
    if torso_mode == "ovals":
        for p in _build_torso_ovals(resolved, messages, taper=taper):
            _append_part(parts, p)
    else:
        torso_part = _build_torso_trap(resolved, messages)
        if torso_part is not None:
            _append_part(parts, torso_part)
        pelvis = _build_pelvis(resolved, messages)
        if pelvis is not None:
            _append_part(parts, pelvis)

    # 3 neck
    neck = _build_neck(report, resolved, messages)
    if neck is not None:
        # Apply template neck thickness scale when present
        if template_applied is not None and neck.radius_m is not None:
            scale = float(template_applied.constants.neck_thickness_scale)
            if scale != 1.0:
                neck = neck.model_copy(update={"radius_m": float(neck.radius_m) * scale})
                messages.append(f"neck_thickness_scale={scale} applied to RECIPE_neck")
        _append_part(parts, neck)

    # 4 head
    head = _build_head(report, resolved, messages)
    if head is not None:
        if template_applied is not None:
            hd = float(template_applied.constants.head_depth_scale)
            hr = float(template_applied.constants.head_radius_scale)
            updates: dict[str, Any] = {}
            if head.rx_m is not None and hr != 1.0:
                updates["rx_m"] = float(head.rx_m) * hr
            if head.ry_m is not None and hd != 1.0:
                updates["ry_m"] = float(head.ry_m) * hd
            if head.rz_m is not None and hr != 1.0:
                updates["rz_m"] = float(head.rz_m) * hr
            if updates:
                head = head.model_copy(update=updates)
                messages.append(f"head scales depth={hd} radius={hr} applied to RECIPE_head")
        _append_part(parts, head)

    # 5-6 shoulder bridges
    for p in _build_shoulder_bridges(report, resolved, messages):
        _append_part(parts, p)

    # 7-8 hip bridges
    for p in _build_hip_bridges(report, resolved, messages):
        _append_part(parts, p)

    # 9-10 deltoids (skip when profile owns delts)
    if "deltoid_soft" not in skip_roles:
        for p in _build_deltoids(report, resolved, messages, crotch_z):
            _append_part(parts, p)
    else:
        messages.append("base deltoid_soft skipped (profile owns delts)")

    # 11+ breast/glute/iliac
    for p in _build_soft_ovals(
        report,
        resolved,
        messages,
        crotch_z,
        glute_mode=glute_mode,
        template_applied=template_applied,
        skip_roles=skip_roles,
    ):
        _append_part(parts, p)

    # 13+ limbs
    if limbs:
        for p in _build_limbs(report, messages):
            _append_part(parts, p)
    else:
        messages.append("--no-limbs: limb_segment parts omitted")

    # 0027 profile emit after base (skip_roles already applied)
    if profile is not None:
        for p in _emit_profile_parts(
            report,
            resolved,
            messages,
            crotch_z,
            profile=profile,
            skeleton=skeleton,
            template_applied=template_applied,
        ):
            _append_part(parts, p)

        # Coincident trap L/R guard
        traps = [p for p in parts if p.role == "trap_soft" and p.center is not None]
        if len(traps) >= 2:
            c0, c1 = traps[0].center, traps[1].center
            if (
                c0 is not None
                and c1 is not None
                and abs(c0[0] - c1[0]) < 1e-6
                and abs(c0[1] - c1[1]) < 1e-6
                and abs(c0[2] - c1[2]) < 1e-6
            ):
                messages.append("trap_soft L/R coincident — check neck_base/shoulder joints")

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

    # counts: numeric only — no mode strings (C1)
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
    """Load blockout_recipe.json; accepts schema 1.0.0 | 1.1.0."""
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
    if ver not in ("1.0.0", "1.1.0"):
        raise ProportionError(
            f"blockout recipe schema_version must be 1.0.0 or 1.1.0 (got {ver!r})",
            code="recipe_failed",
            details={"path": str(p), "schema_version": ver},
        )
    try:
        package = BlockoutRecipePackage.model_validate(data)
    except Exception as exc:
        raise ProportionError(
            f"invalid blockout recipe: {p}: {exc}",
            code="recipe_failed",
            details={"path": str(p)},
        ) from exc
    # R2: per-kind required fields on load (emit path validates via _append_part)
    for part in package.parts:
        _validate_part_fields(part)
    return package


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
        if p.parent_joint is not None:
            entry["parent_joint"] = p.parent_joint
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
        "# hard-delete startup Cube + prior RECIPE_/OVAL_/SOFT_ (0022 C3/C4)",
        "# keep LM_/SEED_/HU_/LM_HEIGHT/cameras/lights",
        "for o in list(bpy.data.objects):",
        "    n = o.name",
        '    if n == "Cube" or n.startswith(("Cube.", "RECIPE_", "OVAL_", "SOFT_")):',
        "        bpy.data.objects.remove(o, do_unlink=True)",
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


def _is_directory_out(out: Path | str) -> bool:
    """R1: existing dir OR ends with / or \\ OR no .py/.json suffix → directory.

    Accept str so trailing separators survive (Path normalizes them away).
    """
    raw = str(out)
    if raw.endswith(("/", "\\")):
        return True
    path = Path(raw.rstrip("/\\"))
    if path.exists() and path.is_dir():
        return True
    return path.suffix.lower() not in (".py", ".json")


def write_blockout_recipe(
    out: Path | str,
    package: BlockoutRecipePackage,
    *,
    format: RecipeFormat = "both",
    force: bool = False,
) -> list[Path]:
    """Write recipe JSON and/or bpy script per R1 path resolution.

    May append warn messages to package.messages for single-file + both.
    *out* may be str so trailing directory separators (R1) are preserved.
    """
    raw = str(out)
    ends_sep = raw.endswith(("/", "\\"))
    out_path = Path(raw.rstrip("/\\") if ends_sep else raw)
    fmt: RecipeFormat = format
    written: list[Path] = []

    is_dir = ends_sep or _is_directory_out(raw if ends_sep else out_path)
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
    torso: TorsoMode = "trap",
    glute: GluteMode = "oval",
    nofuse: bool = False,
    breast_tilt_deg: float | None = None,
    template_applied: Path | str | None = None,
    profiles: str | None = None,
    skeleton: Path | str | None = None,
) -> dict[str, Any]:
    """CLI helper: load report → build → write; return success payload."""
    report = load_report(report_path)
    depth_pkg: DepthSamplesPackage | None = None
    if depth_at_landmarks is not None:
        depth_pkg = _load_depth_at_landmarks(depth_at_landmarks)

    tpl: TemplateAppliedPackage | None = None
    if template_applied is not None:
        from meshops.proportion.body_template import load_template_applied

        tpl = load_template_applied(template_applied)

    profile_doc: AnatomyProfileDocument | None = None
    if profiles is not None and str(profiles).strip():
        from meshops.proportion.anatomy_profile import load_anatomy_profile

        profile_doc = load_anatomy_profile(str(profiles).strip())

    skel: BlockoutSkeleton | None = None
    if skeleton is not None:
        from meshops.proportion.skeleton import load_blockout_skeleton

        try:
            skel = load_blockout_skeleton(skeleton)
        except ProportionError as exc:
            # Soft: continue without parent_joint (B6)
            # Note: load errors still message; parts emit via landmarks.
            # Re-raise only if code is not skeleton_failed/empty? Spec: message + continue.
            if exc.code in ("skeleton_failed", "skeleton_empty"):
                # Defer message into package via a temp list — build will not see it;
                # attach after build.
                skel = None
                skel_err = f"skeleton unreadable: {exc} — parent_joint disabled"
            else:
                raise
        else:
            skel_err = None
    else:
        skel_err = None

    package = build_blockout_recipe(
        report,
        depth_package=depth_pkg,
        limbs=limbs,
        torso=torso,
        glute=glute,
        nofuse=nofuse,
        breast_tilt_deg=breast_tilt_deg,
        template_applied=tpl,
        profile=profile_doc,
        skeleton=skel,
    )
    if skel_err is not None:
        package.messages.append(skel_err)
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
    "_BASELINE_ROLES_NO_PROFILE",
    "_MICHELIN_FRAC",
    "BlockoutRecipePackage",
    "RecipeMetrics",
    "RecipePart",
    "_midpoint_of_joints",
    "build_blockout_recipe",
    "emit_bpy_script",
    "load_blockout_recipe",
    "run_blockout_recipe",
    "write_blockout_recipe",
]
