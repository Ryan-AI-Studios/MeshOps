"""Proportion → blockout primitive recipes (0019 + 0027 + 0028 face + 0029 extremities).

Build BlockoutRecipePackage from ProportionReport; emit JSON + Blender 5.2 bpy script.
Authoring layout only — not mesh or print success (Difficulty §12 / N6).

0024 soft note: measured cranial/foot depth_bands and foot_len_*_m messages may
override template head/foot scales when present — not wired in v1 (document-only).
neck diameter already preferred when available.
breast_lower* used for rz in 0030; lateral fuse/diameter still 0027.
0030: soft offsets prefer report.soft_spacing measured gaps (B7) over template fracs.
0027: anatomy profiles (--profiles) skip_roles merge + parent_joint; schema 1.1.0.
0028: face/hair/neckline RECIPE kit (opt-in); schema write 1.2.0; load 1.0|1.1|1.2.
0029: hands/feet RECIPE kit (opt-in); schema write 1.3.0; load 1.0|1.1|1.2|1.3.
0033: breast hang tilt on breast_soft (rotation_euler_deg + bpy TRS); schema write 1.4.0;
load 1.0|1.1|1.2|1.3|1.4.
0034: product split calf RECIPE_calf_{a,cyl,b}_{side} (not limb_calf) so C_calf_slant
can pass; B6 distal/cyl p1 Y sync to ank_foot after feet emit.
0036: post-pass aligns glute_soft outer X to hip_bridge outer (pre-optimize C_glute_outer).
0039: opt-in --join-ready socket overlaps (shoulder/hip/neck/ankle); mutually exclusive with
--nofuse; setup re-emit via run_blockout_emit_setup; schema stay 1.4.0 + join_ready bool.
0047: torso oval ry depth taper (anti-snowman chest/waist/hip); schema stay 1.4.0.
0049: breast_soft center Z hang drop (before 0033 tilt); B1 floor 0.55*rz; schema stay 1.4.0.
0050: neck column forward tilt (p0/p1) + head/face co-move + radius ceiling vs head.rx;
schema stay 1.4.0.
0052: glute_soft seat mass (ry floor + rear +Y) before 0036 outer align; schema stay 1.4.0.
0053: pelvis bucket shelf scale (oval ry/rx/rz + trap half_depth/z-span); schema stay 1.4.0.
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
from meshops.proportion.skeleton import (
    _arm_forward_y,
    _chest_half_depth_for_arm_prior,
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

RECIPE_SCHEMA_VERSION: Final[Literal["1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0"]] = "1.4.0"
RECIPE_ID: Final[Literal["humanoid_a_pose_v1"]] = "humanoid_a_pose_v1"

JSON_BASENAME: Final[str] = "blockout_recipe.json"
BPY_BASENAME: Final[str] = "setup_blockout_recipe.py"

MIDLINE_X_TOL_M: Final[float] = 0.05
CROTCH_Z_FRAC_FALLBACK: Final[float] = 0.5
_NEAR_ZERO_LEN: Final[float] = 1e-9
# 0045 B1: calf belly + asymmetric ends (replaces single _CALF_END_SCALE=0.95).
_CALF_END_R_FLOOR: Final[float] = 1e-4
CALF_BELLY_SCALE: Final[float] = 1.08
CALF_PROX_END_SCALE: Final[float] = 0.88
CALF_DIST_END_SCALE: Final[float] = 0.72
# 0045 B3: arm distal soft beads only (not thigh — P2-1 / B13).
LIMB_DISTAL_SOFT_SCALE: Final[float] = 0.78
_LIMB_DIST_SOFT_BANDS: Final[frozenset[str]] = frozenset(
    {
        "upper_arm_l",
        "upper_arm_r",
        "forearm_l",
        "forearm_r",
    }
)
# 0045 B5: knee soft joint mass.
KNEE_SOFT_FRAC: Final[float] = 0.55
KNEE_SOFT_MIN_FRAC_H: Final[float] = 0.018
# 0046 B1: deltoid scale vs upper_arm half-width (profile + base).
DELT_ARM_RADIUS_SCALE: Final[float] = 1.35
DELT_RY_FRAC: Final[float] = 0.90
DELT_RZ_FRAC: Final[float] = 0.85
DELT_OUTER_X_FRAC: Final[float] = 0.25  # * rx, sign by side (r:+, l:-); skip if crosses midline
# 0046 B6: thigh proximal soft at hip (no dist_soft - 0045 B13).
THIGH_PROX_SOFT_SCALE: Final[float] = 1.18
_THIGH_PROX_R_FLOOR: Final[float] = 1e-4
# 0046 B9: template thigh_tilt adduction (medial-shift cap + knee-cluster co-move).
THIGH_TILT_DEG_CAP: Final[float] = 15.0
THIGH_ADDUCTION_MAX_MEDIAL_M: Final[float] = 0.030
# 0047 B1: torso oval anteroposterior depth taper (anti-snowman); keep 0040 rz law.
TORSO_OVAL_RY_CHEST_FRAC: Final[float] = 0.95
TORSO_OVAL_RY_WAIST_FRAC: Final[float] = 0.72
TORSO_OVAL_RY_HIP_FRAC: Final[float] = 0.80  # ≠ PELVIS_OVAL_RY_FRAC_HALF_HIP (0.60)
TORSO_OVAL_RZ_SPAN_FRAC: Final[float] = 0.22
TORSO_OVAL_RZ_FLOOR_M: Final[float] = 0.025
# 0049 B1: breast_soft vertical hang floor (center Z drop as fraction of rz).
BREAST_HANG_Z_DROP_FRAC_RZ: Final[float] = 0.55
# 0049 D2: unit min hang drop vs pre-anchor (softer than B1; waist soft-clamp threshold).
BREAST_HANG_Z_MIN_DROP_FRAC_RZ: Final[float] = 0.40
# 0050 B1/B5: neck column forward tilt about +X (tip -Y) + radius ceiling vs head.rx.
NECK_FORWARD_TILT_DEG: Final[float] = 12.0
NECK_R_MAX_FRAC_HEAD_RX: Final[float] = 0.55
_NECK_HEAD_ATTACHED_TOKENS: Final[tuple[str, ...]] = (
    "jaw",
    "brow_soft",
    "eye_soft",
    "nose_soft",
    "ear_soft",
    "lip_soft",
    "hair_mass",
    "neck_head_fuse",
)
# 0052 B1-B13: glute_soft seat depth (ry) + rear projection (+Y) before 0036 outer.
GLUTE_SEAT_RY_FRAC_HALF_DEPTH: Final[float] = 0.90
GLUTE_SEAT_RY_FROM_RX: Final[float] = 1.05
GLUTE_SEAT_BEYOND_REF_Y: Final[float] = 0.020  # meters beyond pelvis/hip ref rear
GLUTE_SEAT_RY_CAP_FRAC_H: Final[float] = 0.10
GLUTE_SEAT_Y_CAP_FRAC_H: Final[float] = 0.15
GLUTE_SEAT_RY_ANISOTROPY_MAX: Final[float] = 2.0  # ry/rx after seat
# 0053: pelvis bucket scale (shelf, not mid-blob)
PELVIS_OVAL_RY_FRAC_HALF_HIP: Final[float] = 0.60
PELVIS_OVAL_RX_FRAC_HIP_HW: Final[float] = 1.00
PELVIS_OVAL_RZ_FRAC_H: Final[float] = 0.042
PELVIS_OVAL_RZ_FLOOR_M: Final[float] = 0.028
PELVIS_OVAL_RY_OVER_RX_MAX: Final[float] = 0.45  # B15 unit ceiling
PELVIS_OVAL_RZ_OVER_H_MAX: Final[float] = 0.05  # B15 unit ceiling
PELVIS_BUCKET_HALF_DEPTH_FRAC: Final[float] = 0.60
PELVIS_BUCKET_HW_FRAC: Final[float] = 1.00
PELVIS_BUCKET_Z_TOP_FRAC_H: Final[float] = 0.02
PELVIS_BUCKET_Z_BOTTOM_FRAC_H: Final[float] = 0.08

_GIRAFFE_FRAC: Final[float] = 0.20
_GIRAFFE_ABS_NO_H: Final[float] = 0.35
_MICHELIN_FRAC: Final[float] = 0.45
_CHEST_HALF_DEPTH_FALLBACK_FRAC: Final[float] = 0.12
_HIP_HALF_DEPTH_FALLBACK_FRAC: Final[float] = 0.13
_DEFAULT_WAIST_TAPER: Final[float] = 0.14
_COLUMNAR_WIDTH_RATIO: Final[float] = 0.1
# Mirror constraints.AXIAL_DEPTH_MARGIN_M — local to avoid import cycle (B12).
_AXIAL_DEPTH_MARGIN_M: Final[float] = 0.02

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
    # 0028 face / hair / neckline
    "jaw",
    "brow_soft",
    "eye_soft",
    "nose_soft",
    "ear_soft",
    "lip_soft",
    "hair_mass",
    "neckline",
    "sternomastoid_soft",
    # 0029 hand / foot / digit
    "palm",
    "finger_soft",
    "thumb_soft",
    "foot_plate",
    "heel",
    "ankle_bridge",
    "toe_soft",
    "ball_soft",
]
RecipeKind = Literal["trap_box", "box", "cylinder", "ellipsoid", "capsule"]
HairTier = Literal["none", "short", "bun", "long_proxy"]
NecklineTier = Literal["none", "crew", "v_proxy"]
FingerTier = Literal["none", "mitten", "full"]
ToeTier = Literal["none", "wedge", "full"]

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
    # 1.4.0 additive: Euler XYZ degrees [rx, ry, rz]; null = identity R in bpy emit
    rotation_euler_deg: list[float] | None = Field(default=None, min_length=3, max_length=3)

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
    """blockout_recipe.json package (schema 1.0.0 | 1.1.0 | 1.2.0 | 1.3.0 | 1.4.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0"] = RECIPE_SCHEMA_VERSION
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
    # 0039 additive optional (schema stay 1.4.0); old JSON → False
    join_ready: bool = False


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


def _finite_m(v: float | None) -> float | None:
    """Return float meters when finite; else None."""
    if v is None:
        return None
    fv = float(v)
    return fv if math.isfinite(fv) else None


def _resolve_axial_chest_y(report: ProportionReport, messages: list[str]) -> float:
    """Axial mid-depth chest_y (B2). Never chest_front alone. Always float.

    Ladder:
      1. landmarks_xyz[chest_mid].y_m (meters) if finite
      2. depth_bands chest y_mid (fraction) * height_m if both finite
      3. 0.0 + source=fallback0 mid_plane
    """
    lms = report.landmarks_xyz
    mid = lms.get("chest_mid")
    if mid is not None:
        y = _finite_m(mid.y_m)
        if y is not None:
            messages.append(f"chest_y={y:.6g} source=chest_mid")
            return y
    band = _depth_band(report, "chest")
    h = _finite_m(report.height_m)
    if band is not None and h is not None:
        y_mid_frac = _finite_m(getattr(band, "y_mid", None))
        if y_mid_frac is not None:
            chest_y = float(y_mid_frac) * float(h)
            messages.append(f"chest_y={chest_y:.6g} source=band")
            return chest_y
    messages.append("chest_y=0 source=fallback0 mid_plane")
    return 0.0


def _resolve_hip_y(report: ProportionReport, messages: list[str]) -> float:
    """Hip Y ladder (B6): hip_mid → mean hip_l/r y → 0.0. Always float."""
    lms = report.landmarks_xyz
    mid = lms.get("hip_mid")
    if mid is not None:
        y = _finite_m(mid.y_m)
        if y is not None:
            messages.append(f"hip_y={y:.6g} source=hip_mid")
            return y
    mean = _mean_y(lms, ("hip_l", "hip_r"))
    if mean is not None and math.isfinite(mean):
        messages.append(f"hip_y={mean:.6g} source=hip_l_r")
        return float(mean)
    messages.append("hip_y=0 source=fallback0")
    return 0.0


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
    # B2 / B6: axial mid-depth plane — never chest_front alone (0032)
    m.chest_y = _resolve_axial_chest_y(report, messages)
    m.hip_y = _resolve_hip_y(report, messages)

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
    # 0053 shelf z-span. P3-4: oval z-center ~ hip_z-0.04H; bucket mid ~ hip_z-0.03H
    # under new span (top +0.02H / bottom -0.08H) -- intentional.
    z_top = m.hip_z + PELVIS_BUCKET_Z_TOP_FRAC_H * h
    z_bottom = max(0.0, m.hip_z - PELVIS_BUCKET_Z_BOTTOM_FRAC_H * h)
    half_depth = half_depth * PELVIS_BUCKET_HALF_DEPTH_FRAC
    y = m.hip_y if m.hip_y is not None else 0.0
    placement: Literal["full3d", "front_plane"] = "full3d" if m.hip_y is not None else "front_plane"
    z_mid = (z_bottom + z_top) / 2.0
    return RecipePart(
        name="RECIPE_pelvis_bucket",
        role="pelvis",
        kind="box",
        center=[0.0, y, z_mid],
        top_half_width_m=m.hip_hw * PELVIS_BUCKET_HW_FRAC,
        bottom_half_width_m=m.hip_hw * PELVIS_BUCKET_HW_FRAC,
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
) -> tuple[RecipePart | None, Any]:
    """Emit RECIPE_head from shared HeadBounds (0028 B17). Returns (part, bounds)."""
    from meshops.proportion.face_recipe import head_part_from_bounds, resolve_head_bounds

    bounds = resolve_head_bounds(
        report,
        head_unit_m=m.head_unit_m,
        height_m=m.height_m,
        messages=messages,
        chest_y=m.chest_y,
    )
    if bounds is None:
        return None, None
    return head_part_from_bounds(bounds), bounds


def _build_shoulder_bridges(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
    skeleton: BlockoutSkeleton | None = None,
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
    # Landmark refs for B12 clamp (prefer true mid/front when present)
    mid_lm = lms.get("chest_mid")
    front_lm = lms.get("chest_front")
    axial_ref = (
        float(mid_lm.y_m)
        if mid_lm is not None and mid_lm.y_m is not None and math.isfinite(float(mid_lm.y_m))
        else y_torso
    )
    chest_front_y = (
        float(front_lm.y_m)
        if front_lm is not None and front_lm.y_m is not None and math.isfinite(float(front_lm.y_m))
        else None
    )
    skel_joints = _joints_map(skeleton)

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
        # B13: p1 Y prefers skeleton shoulder when finite; else landmark; else y_torso.
        # B12: p0 stays axial mid (y_torso).
        sk_sh = skel_joints.get(lm_id)
        if sk_sh is not None and sk_sh.y_m is not None and math.isfinite(float(sk_sh.y_m)):
            sy = float(sk_sh.y_m)
        elif lm.y_m is not None and math.isfinite(float(lm.y_m)):
            sy = float(lm.y_m)
        else:
            sy = y_torso
        # Torso side attachment at shoulder_hw * 0.85 toward shoulder
        torso_x = (
            math.copysign(m.shoulder_hw * 0.85, sx)
            if sx != 0.0
            else (m.shoulder_hw * 0.85 if side == "r" else -m.shoulder_hw * 0.85)
        )
        p0 = [torso_x, y_torso, m.shoulder_z]
        p1 = [sx, sy, sz]
        # B12 clamp: if midpoint closer to chest_front than mid by margin → both mid
        if chest_front_y is not None:
            midpt_y = 0.5 * (float(p0[1]) + float(p1[1]))
            d_mid = abs(midpt_y - axial_ref)
            d_front = abs(midpt_y - chest_front_y)
            if d_front + _AXIAL_DEPTH_MARGIN_M < d_mid:
                p0[1] = axial_ref
                p1[1] = axial_ref
                messages.append(f"RECIPE_shoulder_bridge_{side}: Y clamped to axial mid (B12)")
        if _segment_length((p0[0], p0[1], p0[2]), (p1[0], p1[1], p1[2])) <= _NEAR_ZERO_LEN:
            messages.append(f"RECIPE_shoulder_bridge_{side} skipped: zero length")
            continue
        placement: Literal["full3d", "front_plane"] = (
            "full3d" if m.chest_y is not None else "front_plane"
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


def _apply_delt_outer_x_bias(
    center: list[float],
    side: str,
    rx: float,
    messages: list[str] | None = None,
) -> None:
    """0046 B4: modest outer X bias on deltoid center; skip if would cross midline."""
    if side not in ("l", "r") or rx <= 0.0:
        return
    sign = 1.0 if side == "r" else -1.0
    old_x = float(center[0])
    new_x = old_x + sign * DELT_OUTER_X_FRAC * float(rx)
    # Cross midline if sign of x flips (or lands exactly on 0 from non-zero).
    if old_x != 0.0 and new_x * old_x <= 0.0:
        if messages is not None:
            messages.append(f"deltoid_{side}: outer_x bias skipped (midline cross)")
        return
    center[0] = new_x


def _build_deltoids(
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
    crotch_z: float | None,
    *,
    michelin_cap_frac_h: float | None = None,
    skeleton: BlockoutSkeleton | None = None,
) -> list[RecipePart]:
    parts: list[RecipePart] = []
    if m.shoulder_hw is None:
        messages.append("deltoid softs skipped: shoulder_hw null")
        return parts
    lms = report.landmarks_xyz
    clamp_max = _michelin_clamp_max(m, michelin_cap_frac_h=michelin_cap_frac_h)
    if clamp_max is None:
        clamp_max = _MICHELIN_FRAC * m.shoulder_hw
    skel_joints = _joints_map(skeleton)
    half_depth = _chest_half_depth_for_arm_prior(lms, report.depth_bands)
    front_lm = lms.get("chest_front")
    chest_front_y = (
        float(front_lm.y_m)
        if front_lm is not None and front_lm.y_m is not None and math.isfinite(float(front_lm.y_m))
        else None
    )
    mid_lm = lms.get("chest_mid")
    chest_mid_y = (
        float(mid_lm.y_m)
        if mid_lm is not None and mid_lm.y_m is not None and math.isfinite(float(mid_lm.y_m))
        else None
    )
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
        # 0046 B2: soft deltoid larger than arm half-width (DELT scale).
        measured = base_r * DELT_ARM_RADIUS_SCALE
        clamped = measured
        if measured >= clamp_max:
            clamped = clamp_max
            messages.append(
                f"deltoid radius {measured:.3f}m clamped to {clamped:.3f}m (Michelin guard)"
            )
        # 0051 B8: skeleton shoulder Y → landmark Y → arm forward prior.
        sk_sh = skel_joints.get(lm_id)
        if sk_sh is not None and sk_sh.y_m is not None and math.isfinite(float(sk_sh.y_m)):
            y = float(sk_sh.y_m)
            placement: Literal["full3d", "front_plane"] = "full3d"
        elif lm.y_m is not None and math.isfinite(float(lm.y_m)):
            y = float(lm.y_m)
            placement = "full3d"
        else:
            y_plane = float(chest_mid_y) if chest_mid_y is not None else 0.0
            y = _arm_forward_y(
                y_plane,
                half_depth=half_depth,
                height_m=m.height_m,
                chest_front_y=chest_front_y,
            )
            placement = "full3d"
            messages.append(f"RECIPE_deltoid_soft_{side}: y_m from arm forward prior")
        center = [float(lm.x_m), y, float(lm.z_m)]
        _apply_delt_outer_x_bias(center, side, clamped, messages)
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
                ry_m=clamped * DELT_RY_FRAC,
                rz_m=clamped * DELT_RZ_FRAC,
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
    half_chest = m.chest_half_depth
    if half_chest is None:
        messages.append("torso ovals skipped: need chest half_depth")
        return parts
    half_hip = m.hip_half_depth if m.hip_half_depth is not None else half_chest
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
    ry_chest: float | None = None
    ry_waist: float | None = None
    ry_hip: float | None = None
    for name, z_norm in layers:
        z_m = z_top - z_norm * span
        hw = _waist_width_at(z_norm, w_s, w_h, taper)
        # Vertical radius must exceed half layer spacing (0.35*span/2) so chest/waist/hip
        # ovals overlap. 0.12 left a belly gap; 0.18 barely kissed; 0.22 ≈ 4cm overlap.
        rz = max(TORSO_OVAL_RZ_FLOOR_M, span * TORSO_OVAL_RZ_SPAN_FRAC)
        if name.endswith("_chest"):
            ry = half_chest * TORSO_OVAL_RY_CHEST_FRAC
            ry_chest = ry
        elif name.endswith("_waist"):
            ry = half_chest * TORSO_OVAL_RY_WAIST_FRAC
            ry_waist = ry
        else:  # hip
            ry = half_hip * TORSO_OVAL_RY_HIP_FRAC
            ry_hip = ry
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
    if ry_chest is not None and ry_waist is not None and ry_hip is not None:
        messages.append(
            "torso depth taper: chest/waist/hip "
            f"ry={ry_chest:.4f}/{ry_waist:.4f}/{ry_hip:.4f} (anti-snowman)"
        )

    # 0053 pelvis shelf freezes (was B10: ry = hip_half * 0.85)
    if m.height_m is not None:
        h = m.height_m
        z_pelvis = m.hip_z - 0.04 * h
        if z_pelvis < 0.0:
            z_pelvis = max(0.02, m.hip_z * 0.5)
        y_pelvis = m.hip_y if m.hip_y is not None else y
        p_place: Literal["full3d", "front_plane"] = "full3d" if m.hip_y is not None else placement
        # P3-7: after B2, rx_pelvis may exceed rx of RECIPE_torso_oval_hip
        # (bicristal shelf — intentional).
        rx_p = w_h * PELVIS_OVAL_RX_FRAC_HIP_HW
        ry_p = half_hip * PELVIS_OVAL_RY_FRAC_HALF_HIP
        rz_p = max(PELVIS_OVAL_RZ_FLOOR_M, PELVIS_OVAL_RZ_FRAC_H * h)
        parts.append(
            RecipePart(
                name="RECIPE_pelvis_oval",
                role="pelvis",
                kind="ellipsoid",
                center=[0.0, y_pelvis, z_pelvis],
                rx_m=rx_p,
                ry_m=ry_p,
                rz_m=rz_p,
                placement=p_place,
                label="RECIPE_pelvis_oval",
            )
        )
        messages.append(
            f"pelvis bucket scale: rx={rx_p:.4f} ry={ry_p:.4f} rz={rz_p:.4f} "
            f"(fracs {PELVIS_OVAL_RX_FRAC_HIP_HW:.2f}/"
            f"{PELVIS_OVAL_RY_FRAC_HALF_HIP:.2f}/{PELVIS_OVAL_RZ_FRAC_H}H)"
        )
    else:
        messages.append("RECIPE_pelvis_oval skipped: need height_m")
    return parts


def _build_calf_split(
    *,
    side: str,
    p0: list[float],
    p1: list[float],
    radius: float,
    placement: Literal["full3d", "front_plane"],
    messages: list[str],
) -> list[RecipePart]:
    """0034 names + 0045 B1-B2 belly/taper: calf_a / calf_cyl / calf_b.

    Proximal ellipsoid @ knee (p0) with prox end scale, distal @ ankle (p1)
    with dist end scale, shaft capsule with belly scale vs mid measured r.
    """
    mid_r = float(radius)
    cyl_r = max(mid_r * CALF_BELLY_SCALE, _CALF_END_R_FLOOR)
    prox_r = max(mid_r * CALF_PROX_END_SCALE, _CALF_END_R_FLOOR)
    dist_r = max(mid_r * CALF_DIST_END_SCALE, _CALF_END_R_FLOOR)
    name_a = f"RECIPE_calf_a_{side}"
    name_cyl = f"RECIPE_calf_cyl_{side}"
    name_b = f"RECIPE_calf_b_{side}"
    parts = [
        RecipePart(
            name=name_a,
            role="limb_segment",
            kind="ellipsoid",
            center=[float(p0[0]), float(p0[1]), float(p0[2])],
            rx_m=prox_r,
            ry_m=prox_r,
            rz_m=prox_r,
            placement=placement,
            label=name_a,
        ),
        RecipePart(
            name=name_cyl,
            role="limb_segment",
            kind="capsule",
            p0=[float(p0[0]), float(p0[1]), float(p0[2])],
            p1=[float(p1[0]), float(p1[1]), float(p1[2])],
            radius_m=cyl_r,
            placement=placement,
            label=name_cyl,
        ),
        RecipePart(
            name=name_b,
            role="limb_segment",
            kind="ellipsoid",
            center=[float(p1[0]), float(p1[1]), float(p1[2])],
            rx_m=dist_r,
            ry_m=dist_r,
            rz_m=dist_r,
            placement=placement,
            label=name_b,
        ),
    ]
    note = f"calf_{side}: belly/taper a={prox_r:.4f} cyl={cyl_r:.4f} b={dist_r:.4f}"
    if placement == "front_plane":
        note += " (front_plane)"
    messages.append(note)
    return parts


def _sync_calf_distal_to_ankle(
    parts: list[RecipePart],
    messages: list[str],
) -> None:
    """0034 B6: set calf_distal center Y and cyl p1[1] to ank_foot Y when present.

    Idempotent absolute set. No ankle → leave landmark Y. Name-matched to avoid
    import cycle with constraints.classify_part_name.

    When Y is rewritten from ank_foot, also set placement=full3d so front_plane
    calves (null joint y_m at emit) do not keep stale plane metadata after sync.
    """
    by_name = {p.name: p for p in parts}
    for side in ("l", "r"):
        ank = by_name.get(f"RECIPE_ank_foot_{side}")
        if ank is None or ank.center is None or len(ank.center) < 3:
            continue
        ay = float(ank.center[1])
        updated = False
        dist = by_name.get(f"RECIPE_calf_b_{side}")
        if dist is not None and dist.center is not None and len(dist.center) >= 3:
            dist.center = [float(dist.center[0]), ay, float(dist.center[2])]
            dist.placement = "full3d"
            updated = True
        cyl = by_name.get(f"RECIPE_calf_cyl_{side}")
        if cyl is not None and cyl.p1 is not None and len(cyl.p1) >= 3:
            # Keep p0[1] = proximal Y; only p1 Y tracks ankle.
            cyl.p1 = [float(cyl.p1[0]), ay, float(cyl.p1[2])]
            cyl.placement = "full3d"
            updated = True
        # Honesty: only claim sync when distal and/or cyl were actually written
        # (feet without limbs leave ank_foot present but no calf parts).
        if updated:
            messages.append(f"calf_{side}: distal/cyl p1 Y synced to ank_foot ({ay:.4f})")


def _build_limbs(
    report: ProportionReport,
    messages: list[str],
    skeleton: BlockoutSkeleton | None = None,
) -> list[RecipePart]:
    """Limb capsules on SEED_SEGMENT_MAP; calf → a/cyl/b split (0034).

    0037 R2: upper_arm/forearm prefer skeleton joint endpoints when both finite XYZ.
    Thigh/calf stay report-only for DoD (arm-only free ride).
    """
    parts: list[RecipePart] = []
    lms = report.landmarks_xyz
    skip_count = 0
    joints = _joints_map(skeleton)
    # Arm segments only — not thigh/calf (0037 AI2 B4).
    _ARM_SKELETON_BANDS = frozenset({"upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r"})

    for band_id, (p0_id, p1_id) in SEED_SEGMENT_MAP.items():
        diam = _resolve_diameter(report.diameters, band_id)
        radius = _half_width_from_diameter(diam) if diam else None
        if radius is None:
            messages.append(f"{band_id}: no usable radius — limb skipped")
            skip_count += 1
            continue

        p0: list[float] | None = None
        p1: list[float] | None = None
        placement: Literal["full3d", "front_plane"] = "front_plane"
        used_skeleton = False

        if band_id in _ARM_SKELETON_BANDS:
            sk0 = _joint_xyz(joints.get(p0_id))
            sk1 = _joint_xyz(joints.get(p1_id))
            if sk0 is not None and sk1 is not None:
                # _joint_xyz requires finite XYZ → both Y finite → full3d
                p0 = list(sk0)
                p1 = list(sk1)
                placement = "full3d"
                used_skeleton = True
                messages.append(f"{band_id}: endpoints from skeleton joints")

        if not used_skeleton:
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
            y0_null = lm0.y_m is None
            y1_null = lm1.y_m is None
            is_arm = band_id in _ARM_SKELETON_BANDS
            if y0_null or y1_null:
                # 0051 B7: arm both-null → arm forward prior (full3d); mixed → mean/front_plane.
                # Thigh/calf unchanged: any-null → front_plane.
                if is_arm and y0_null and y1_null:
                    half_depth = _chest_half_depth_for_arm_prior(lms, report.depth_bands)
                    front_lm = lms.get("chest_front")
                    chest_front_y = (
                        float(front_lm.y_m)
                        if front_lm is not None
                        and front_lm.y_m is not None
                        and math.isfinite(float(front_lm.y_m))
                        else None
                    )
                    mid_lm = lms.get("chest_mid")
                    chest_mid_y = (
                        float(mid_lm.y_m)
                        if mid_lm is not None
                        and mid_lm.y_m is not None
                        and math.isfinite(float(mid_lm.y_m))
                        else None
                    )
                    y_plane = float(chest_mid_y) if chest_mid_y is not None else 0.0
                    y_prior = _arm_forward_y(
                        y_plane,
                        half_depth=half_depth,
                        height_m=report.height_m,
                        chest_front_y=chest_front_y,
                    )
                    p0 = [float(lm0.x_m), y_prior, float(lm0.z_m)]
                    p1 = [float(lm1.x_m), y_prior, float(lm1.z_m)]
                    placement = "full3d"
                    messages.append(f"{band_id}: y_m null — arm forward prior")
                else:
                    ys = [y for y in (lm0.y_m, lm1.y_m) if y is not None]
                    y_plane = (sum(ys) / len(ys)) if ys else 0.0
                    p0 = [float(lm0.x_m), y_plane, float(lm0.z_m)]
                    p1 = [float(lm1.x_m), y_plane, float(lm1.z_m)]
                    placement = "front_plane"
                    if band_id not in ("calf_l", "calf_r"):
                        messages.append(f"{band_id}: y_m null — front_plane limb capsule")
            else:
                p0 = [float(lm0.x_m), float(lm0.y_m), float(lm0.z_m)]  # type: ignore[arg-type]
                p1 = [float(lm1.x_m), float(lm1.y_m), float(lm1.z_m)]  # type: ignore[arg-type]
                placement = "full3d"

        assert p0 is not None and p1 is not None
        if _segment_length((p0[0], p0[1], p0[2]), (p1[0], p1[1], p1[2])) <= _NEAR_ZERO_LEN:
            messages.append(f"{band_id}: zero-length segment — limb skipped")
            skip_count += 1
            continue
        if band_id in ("calf_l", "calf_r"):
            side = "l" if band_id.endswith("_l") else "r"
            parts.extend(
                _build_calf_split(
                    side=side,
                    p0=p0,
                    p1=p1,
                    radius=float(radius),
                    placement=placement,
                    messages=messages,
                )
            )
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
        # 0046 B6: thigh proximal soft at hip only (no dist_soft — 0045 B13).
        if band_id in ("thigh_l", "thigh_r"):
            side = "l" if band_id.endswith("_l") else "r"
            soft_r = max(float(radius) * THIGH_PROX_SOFT_SCALE, _THIGH_PROX_R_FLOOR)
            soft_name = f"RECIPE_prox_soft_thigh_{side}"
            parts.append(
                RecipePart(
                    name=soft_name,
                    role="limb_segment",
                    kind="ellipsoid",
                    center=[float(p0[0]), float(p0[1]), float(p0[2])],
                    rx_m=soft_r,
                    ry_m=soft_r,
                    rz_m=soft_r,
                    placement=placement,
                    label=soft_name,
                )
            )
            messages.append(f"thigh_{side}: prox_soft r={soft_r:.4f}")
        # 0045 B3-B4: arm distal soft only (not thigh — P2-1 / B13).
        if band_id in _LIMB_DIST_SOFT_BANDS:
            soft_r = max(float(radius) * LIMB_DISTAL_SOFT_SCALE, 1e-4)
            soft_name = f"RECIPE_dist_soft_{band_id}"
            parts.append(
                RecipePart(
                    name=soft_name,
                    role="limb_segment",
                    kind="ellipsoid",
                    center=[float(p1[0]), float(p1[1]), float(p1[2])],
                    rx_m=soft_r,
                    ry_m=soft_r,
                    rz_m=soft_r,
                    placement=placement,
                    label=soft_name,
                )
            )

    # Cap skip flood: at most 8 segment skip messages already one-per-band
    _ = skip_count  # ≤8 by construction (SEED map size)
    return parts


def _knee_adj_radius_m(
    parts: list[RecipePart],
    side: str,
    report: ProportionReport,
) -> float | None:
    """0045 B5: adjacent shaft radius for knee soft (name-keyed primary)."""
    by = {p.name: p for p in parts}
    candidates: list[float] = []
    thigh = by.get(f"RECIPE_limb_thigh_{side}")
    if thigh is not None and thigh.radius_m is not None:
        candidates.append(float(thigh.radius_m))
    calf_a = by.get(f"RECIPE_calf_a_{side}")
    if calf_a is not None and calf_a.rx_m is not None:
        candidates.append(float(calf_a.rx_m))
    if candidates:
        return max(candidates)
    # Fallback only when both parts absent: diameter ladder half-widths
    th = _resolve_diameter(report.diameters, f"thigh_{side}")
    ca = _resolve_diameter(report.diameters, f"calf_{side}")
    for d in (th, ca):
        if d is not None:
            hw = _half_width_from_diameter(d)
            if hw is not None:
                candidates.append(float(hw))
    return max(candidates) if candidates else None


def _knee_center_and_placement(
    report: ProportionReport,
    side: str,
    skeleton: BlockoutSkeleton | None,
) -> tuple[list[float], Literal["full3d", "front_plane"]] | None:
    """Knee landmark first (finite XZ); skeleton knee_{side} fallback.

    Placement full3d when Y known; else front_plane with y_plane policy.
    """
    lms = report.landmarks_xyz
    kid = f"knee_{side}"
    if kid in lms:
        lm = lms[kid]
        if lm.x_m is not None and lm.z_m is not None:
            if lm.y_m is not None:
                return (
                    [float(lm.x_m), float(lm.y_m), float(lm.z_m)],
                    "full3d",
                )
            return (
                [float(lm.x_m), 0.0, float(lm.z_m)],
                "front_plane",
            )
    sk = _joint_xyz(_joints_map(skeleton).get(kid))
    if sk is not None:
        return (list(sk), "full3d")
    return None


def _append_knee_softs(
    parts: list[RecipePart],
    report: ProportionReport,
    skeleton: BlockoutSkeleton | None,
    height_m: float | None,
    messages: list[str],
) -> None:
    """0045 B5: post-pass knee soft ellipsoids after limbs emit."""
    for side in ("l", "r"):
        center_place = _knee_center_and_placement(report, side, skeleton)
        if center_place is None:
            continue  # skip missing joint — not fail
        center, placement = center_place
        adj = _knee_adj_radius_m(parts, side, report)
        r: float | None = KNEE_SOFT_FRAC * adj if adj is not None else None
        if height_m is not None and height_m > 0:
            floor = KNEE_SOFT_MIN_FRAC_H * float(height_m)
            r = max(r, floor) if r is not None else floor
        if r is None:
            continue
        name = f"RECIPE_knee_soft_{side}"
        _append_part(
            parts,
            RecipePart(
                name=name,
                role="limb_segment",
                kind="ellipsoid",
                center=center,
                rx_m=r,
                ry_m=r,
                rz_m=r,
                placement=placement,
                label=name,
            ),
        )
        messages.append(f"knee_soft_{side}: r={r:.4f}")


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
    role: str | None = None,
) -> tuple[float, float, float]:
    """B8 scale precedence -> (rx, ry, rz) meters.

    0046 B3: role==deltoid_soft uses arm diameter as *boost* (DELT scale), not x0.55.
    Breast/glute diameter multipliers stay on the generic path.
    """
    h = m.height_m

    # --- 0046 B3: deltoid_soft dedicated scale (primary product path) ---
    if role == "deltoid_soft":
        rx: float | None = None
        if scale.use_diameter:
            band = scale.use_diameter
            if band == "upper_arm":
                band = f"upper_arm_{side}" if side in ("l", "r") else band
            diam = _resolve_diameter(report.diameters, band)
            hw = _half_width_from_diameter(diam) if diam else None
            if hw is not None:
                rx = float(hw) * DELT_ARM_RADIUS_SCALE
        if rx is None:
            rx = _scale_from_frac_h(scale.rx_frac_h, h)
        if rx is None:
            rx = 0.04 * (h or 1.7)
            messages.append(f"profile deltoid rx fallback {rx:.4f}m")
        ry = float(rx) * DELT_RY_FRAC
        rz = float(rx) * DELT_RZ_FRAC
        if scale.michelin_cap_frac_h is not None:
            cap = _michelin_clamp_max(m, michelin_cap_frac_h=scale.michelin_cap_frac_h)
            if cap is not None:
                for axis_name, val in (("rx", rx), ("ry", ry), ("rz", rz)):
                    if val > cap:
                        messages.append(
                            f"profile {axis_name} {val:.3f}m clamped to {cap:.3f}m "
                            f"(michelin_cap_frac_h={scale.michelin_cap_frac_h})"
                        )
                rx = min(float(rx), cap)
                ry = min(float(ry), cap)
                rz = min(float(rz), cap)
        _ = template_applied
        return float(rx), float(ry), float(rz)

    rx = None
    ry = None
    rz = None

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
                # 0051 B8: null landmark Y → arm forward prior (not invent-0 mid-plane).
                if lm.y_m is not None and math.isfinite(float(lm.y_m)):
                    dy = float(lm.y_m)
                else:
                    half_depth = _chest_half_depth_for_arm_prior(lms, report.depth_bands)
                    front_lm = lms.get("chest_front")
                    chest_front_y = (
                        float(front_lm.y_m)
                        if front_lm is not None
                        and front_lm.y_m is not None
                        and math.isfinite(float(front_lm.y_m))
                        else None
                    )
                    mid_lm = lms.get("chest_mid")
                    chest_mid_y = (
                        float(mid_lm.y_m)
                        if mid_lm is not None
                        and mid_lm.y_m is not None
                        and math.isfinite(float(mid_lm.y_m))
                        else None
                    )
                    y_plane = float(chest_mid_y) if chest_mid_y is not None else 0.0
                    dy = _arm_forward_y(
                        y_plane,
                        half_depth=half_depth,
                        height_m=height_m,
                        chest_front_y=chest_front_y,
                    )
                    messages.append(
                        f"RECIPE_deltoid_soft_{side_tag}: y_m from arm forward prior (profile)"
                    )
                center = [float(lm.x_m), dy, float(lm.z_m)]
                messages.append(f"parent_joint {role} unresolved — using landmark placement")
            else:
                messages.append(f"{name} skipped: missing joint")
                return None

    elif role in ("breast_soft", "pec_soft", "scap_soft", "glute_soft"):
        # Anchor at spine_high / pelvis then offset L/R + Y rule
        if role == "glute_soft":
            anchor = _joint_xyz(joints.get("pelvis"))
            if anchor is None and m.hip_z is not None:
                hip_y_anchor = m.hip_y if m.hip_y is not None else 0.0
                anchor = [0.0, hip_y_anchor, m.hip_z]
        else:
            anchor = _joint_xyz(joints.get("spine_high")) or _joint_xyz(joints.get("spine_mid"))
            chest_y_anchor = m.chest_y if m.chest_y is not None else 0.0
            if anchor is None and m.chest_z is not None:
                anchor = [0.0, chest_y_anchor, m.chest_z]
            elif anchor is None and m.shoulder_z is not None:
                anchor = [0.0, chest_y_anchor, m.shoulder_z]
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
            role=str(role) if role is not None else None,
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

    # 0046 B4: deltoid outer X bias after placement + axes (uses post-cap rx).
    if center is not None and role == "deltoid_soft" and side_tag in ("l", "r"):
        _apply_delt_outer_x_bias(center, side_tag, float(rx), messages)

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
    join_ready: bool = False,
    breast_tilt_deg: float | None = None,
    template_applied: TemplateAppliedPackage | None = None,
    profile: AnatomyProfileDocument | None = None,
    skeleton: BlockoutSkeleton | None = None,
    face: bool = False,
    hair: HairTier = "none",
    neckline: NecklineTier = "none",
    hands: bool = False,
    feet: bool = False,
    fingers: FingerTier = "mitten",
    toes: ToeTier = "wedge",
) -> BlockoutRecipePackage:
    """Build BlockoutRecipePackage from a loaded ProportionReport.

    Raises ProportionError(code=recipe_empty) when zero parts.
    Topology modes only in messages (C1) — not in counts.
    When *profile* set: skip_roles merge (R6.1) + profile dual softs / new roles.
    Opt-in face/hair/neckline (0028): default flags preserve pre-0028 role set (B6).
    Opt-in hands/feet (0029): default flags preserve pre-0029 role set (B6).
    Opt-in join_ready (0039): socket overlaps after glute align; mutually exclusive with nofuse.
    """
    if nofuse and join_ready:
        raise ProportionError(
            "nofuse and join-ready are mutually exclusive",
            code="recipe_failed",
        )

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
    if face or hair != "none" or neckline != "none":
        messages.append(f"face={str(bool(face)).lower()} hair={hair} neckline={neckline}")
    if hands or feet:
        messages.append(
            f"hands={str(bool(hands)).lower()} feet={str(bool(feet)).lower()} "
            f"fingers={fingers} toes={toes}"
        )

    if template_applied is not None:
        messages.append(f"template_applied: id={template_applied.template_id}")

    # Breast hang ladder (B1 / 0027 B9): CLI → template tilt — never slant as hang.
    # Applied true/false + rotation attach happen in post-pass after profile emit (0033 B8).
    tilt_val: float | None = breast_tilt_deg
    if tilt_val is None and template_applied is not None:
        tilt_val = float(template_applied.constants.breast_tilt_x_deg)

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

    # 4 head (shared HeadBounds for face kit — B17)
    head, head_bounds = _build_head(report, resolved, messages)
    if head is not None:
        if template_applied is not None and head_bounds is not None:
            from meshops.proportion.face_recipe import scale_head_bounds

            hd = float(template_applied.constants.head_depth_scale)
            hr = float(template_applied.constants.head_radius_scale)
            rx_s = hr if hr != 1.0 else 1.0
            ry_s = hd if hd != 1.0 else 1.0
            rz_s = hr if hr != 1.0 else 1.0
            if rx_s != 1.0 or ry_s != 1.0 or rz_s != 1.0:
                head_bounds = scale_head_bounds(
                    head_bounds, rx_scale=rx_s, ry_scale=ry_s, rz_scale=rz_s
                )
                head = head.model_copy(
                    update={
                        "rx_m": head_bounds.rx,
                        "ry_m": head_bounds.ry,
                        "rz_m": head_bounds.rz,
                    }
                )
                messages.append(f"head scales depth={hd} radius={hr} applied to RECIPE_head")
        _append_part(parts, head)

    # 4b face / hair / neckline kit (0028) — same HeadBounds as RECIPE_head
    if face or hair != "none" or neckline != "none":
        from meshops.proportion.face_recipe import build_face_parts

        neck_top_z: float | None = None
        neck_radius: float | None = None
        if neck is not None:
            if neck.p1 is not None and len(neck.p1) >= 3:
                neck_top_z = float(neck.p1[2])
            neck_radius = float(neck.radius_m) if neck.radius_m is not None else None
        for p in build_face_parts(
            report,
            head_bounds,
            face=face,
            hair=hair,
            neckline=neckline,
            skeleton=skeleton,
            shoulder_hw=resolved.shoulder_hw,
            neck_len_m=resolved.neck_len_m,
            shoulder_z=resolved.shoulder_z,
            chest_y=resolved.chest_y,
            neck_top_z=neck_top_z,
            neck_radius=neck_radius,
            head_unit_m=resolved.head_unit_m,
            messages=messages,
        ):
            _append_part(parts, p)

    # 0050: mild forward neck tilt + head co-move + radius ceiling (after face kit if any)
    _apply_neck_column_priors(parts, messages)

    # 5-6 shoulder bridges
    for p in _build_shoulder_bridges(report, resolved, messages, skeleton=skeleton):
        _append_part(parts, p)

    # 7-8 hip bridges
    for p in _build_hip_bridges(report, resolved, messages):
        _append_part(parts, p)

    # 9-10 deltoids (skip when profile owns delts)
    if "deltoid_soft" not in skip_roles:
        for p in _build_deltoids(report, resolved, messages, crotch_z, skeleton=skeleton):
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
        for p in _build_limbs(report, messages, skeleton=skeleton):
            _append_part(parts, p)
        # 0045 B5: knee soft post-pass (independent of feet)
        _append_knee_softs(
            parts,
            report,
            skeleton,
            resolved.height_m,
            messages,
        )
    else:
        messages.append("--no-limbs: limb_segment parts omitted")

    # 14 extremities (0029) — append after limbs; does not remove forearm/calf (B15)
    if hands:
        from meshops.proportion.extremity_recipe import build_hand_parts

        for p in build_hand_parts(
            report,
            skeleton=skeleton,
            height_m=resolved.height_m,
            fingers=fingers,
            messages=messages,
        ):
            _append_part(parts, p)
    if feet:
        from meshops.proportion.extremity_recipe import build_foot_parts

        for p in build_foot_parts(
            report,
            skeleton=skeleton,
            template_applied=template_applied,
            height_m=resolved.height_m,
            toes=toes,
            messages=messages,
            existing_parts=parts,
        ):
            _append_part(parts, p)

    # 0034 B6: distal/cyl p1 Y ← ank_foot after feet, before profile/breast tilt
    _sync_calf_distal_to_ankle(parts, messages)

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

    # 0049 B4: drop breast_soft center Z for readable hang (before 0033 tilt)
    _apply_breast_hang_z(parts, report, resolved, messages)

    # 0033 B3/B8: attach breast hang tilt to breast_soft ellipsoids after all emitters
    _apply_breast_tilt(parts, tilt_val=tilt_val, messages=messages)

    # 0046 B9: thigh adduction after limbs+knee+calf exist; before glute outer + join_ready
    _apply_thigh_adduction(parts, template_applied, messages)

    # 0052: glute seat ry floor + rear +Y (before 0036 outer so rx stays outer-correct)
    _apply_glute_seat_mass(parts, report, resolved, messages)

    # 0036: glute outer tip X = hip_bridge outer X (same formula as constraints opt clamp)
    _align_glute_outer_to_hip_bridge(parts, messages)

    # 0039: join-ready socket overlaps AFTER breast tilt, calf sync, glute outer (B10)
    if join_ready:
        _apply_join_ready_overlaps(parts, messages)
        messages.append("join_ready=true")

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
        join_ready=bool(join_ready),
    )


# ---------------------------------------------------------------------------
# 0036 — glute outer X = hip_bridge outer at emit
# Parity with constraints._hip_outer_x / set_part_x / outer clamp (recipe-local;
# do not import constraints — avoid cycles). Does not call optimize_package.
# ---------------------------------------------------------------------------


def _side_from_recipe_name(name: str) -> Literal["l", "r"] | None:
    """Simple _l / _r suffix (after optional Blender .###); else None."""
    base = name.lower()
    if "." in base:
        # Strip trailing .001 style suffixes only when last segment is digits.
        head, tail = base.rsplit(".", 1)
        if tail.isdigit():
            base = head
    if base.endswith("_l"):
        return "l"
    if base.endswith("_r"):
        return "r"
    return None


def _part_center_x(part: RecipePart) -> float | None:
    """Center X from center, else mid(p0, p1)."""
    if part.center is not None and len(part.center) >= 1:
        return float(part.center[0])
    if part.p0 is not None and part.p1 is not None and len(part.p0) >= 1 and len(part.p1) >= 1:
        return 0.5 * (float(part.p0[0]) + float(part.p1[0]))
    return None


def _half_extent_x_local(part: RecipePart) -> float | None:
    """Half-extent X: prefer rx_m, else radius_m, else top/bottom half-width."""
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


def _hip_bridge_outer_x(parts: list[RecipePart], side: Literal["l", "r"]) -> float | None:
    """Hip bridge outer tip X for *side*. Prefer side-tagged, then any hip_bridge."""
    side_bridges = [
        p for p in parts if p.role == "hip_bridge" and _side_from_recipe_name(p.name) == side
    ]
    bridges = side_bridges
    if not bridges:
        bridges = [p for p in parts if p.role == "hip_bridge"]
    if not bridges:
        return None
    bridge = bridges[0]
    cx = _part_center_x(bridge)
    if cx is None:
        return None
    half = _half_extent_x_local(bridge)
    if half is None:
        return cx  # midline bridge without width — weak (parity with constraints)
    if side == "r":
        return cx + half
    return cx - half


def _set_part_x_local(part: RecipePart, x: float) -> None:
    """Mirror constraints.set_part_x: set center X and shift p0/p1 by same Δ."""
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


# ---------------------------------------------------------------------------
# 0039 — join-ready socket overlaps (local center-pull + radius grow ≤1.08)
# ---------------------------------------------------------------------------

_JOIN_READY_MAX_SCALE: Final[float] = 1.08
_JOIN_READY_EPS: Final[float] = 1e-6


def _shift_part_along_axis(part: RecipePart, axis: int, delta: float) -> None:
    """Translate part center / p0 / p1 by *delta* along axis 0|1|2."""
    if abs(delta) <= 1e-15:
        return
    if part.center is not None and len(part.center) >= 3:
        c = [float(part.center[0]), float(part.center[1]), float(part.center[2])]
        c[axis] += float(delta)
        part.center = c
    if part.p0 is not None and len(part.p0) >= 3:
        p0 = [float(part.p0[0]), float(part.p0[1]), float(part.p0[2])]
        p0[axis] += float(delta)
        part.p0 = p0
    if part.p1 is not None and len(part.p1) >= 3:
        p1 = [float(part.p1[0]), float(part.p1[1]), float(part.p1[2])]
        p1[axis] += float(delta)
        part.p1 = p1


def _apply_neck_column_priors(parts: list[RecipePart], messages: list[str]) -> None:
    """0050: mild forward cervical tilt of RECIPE_neck + head stack co-move + radius ceiling.

    Rewrites neck p0/p1 only (bpy cylinder ignores Euler). Base p0 fixed; length L preserved;
    tip leans -Y by NECK_FORWARD_TILT_DEG about +X. Head-attached parts full dY; SCM tip only.
    Radius ceiling after template thickness_scale already applied.
    """
    idxs = [
        i
        for i, p in enumerate(parts)
        if p.name == "RECIPE_neck"
        and p.kind == "cylinder"
        and p.p0 is not None
        and p.p1 is not None
        and len(p.p0) >= 3
        and len(p.p1) >= 3
    ]
    if not idxs:
        messages.append("neck_column_tilt_applied: false")
        return

    i = idxs[0]
    neck = parts[i]
    assert neck.p0 is not None and neck.p1 is not None
    p0 = [float(c) for c in neck.p0]
    p1 = [float(c) for c in neck.p1]
    length = math.dist(p0, p1)
    if length <= 0.0 or not math.isfinite(length):
        messages.append("neck_column_tilt_applied: false")
        messages.append("neck_column_tilt_reason=nonpositive_length")
        return

    theta = math.radians(NECK_FORWARD_TILT_DEG)
    if not math.isfinite(theta) or abs(theta) <= 1e-15:
        messages.append("neck_column_tilt_applied: false")
        return

    y0 = p0[1]
    z0 = p0[2]
    p1_new = [p0[0], y0 - length * math.sin(theta), z0 + length * math.cos(theta)]
    dy_tip = p1_new[1] - y0
    parts[i] = neck.model_copy(update={"p0": list(p0), "p1": p1_new})

    # B5 radius ceiling vs head (after template scale already applied upstream)
    head = next((p for p in parts if p.name == "RECIPE_head"), None)
    neck_r = parts[i].radius_m
    head_rx = head.rx_m if head is not None else None
    if (
        neck_r is not None
        and head_rx is not None
        and math.isfinite(float(head_rx))
        and float(head_rx) > 0.0
        and math.isfinite(float(neck_r))
        and float(neck_r) > 0.0
    ):
        cap = NECK_R_MAX_FRAC_HEAD_RX * float(head_rx)
        r0 = float(neck_r)
        if r0 > cap:
            parts[i] = parts[i].model_copy(update={"radius_m": cap})
            messages.append(f"neck_radius_clamped_head_frac={NECK_R_MAX_FRAC_HEAD_RX}")

    # B4a head-attached full translate by dy_tip
    moved_head = False
    for j, p in enumerate(parts):
        if j == i:
            continue
        name_l = p.name.lower()
        if p.name == "RECIPE_head" or any(t in name_l for t in _NECK_HEAD_ATTACHED_TOKENS):
            _shift_part_along_axis(p, 1, dy_tip)
            if p.name == "RECIPE_head":
                moved_head = True
            if "neck_head_fuse" in name_l and p.p0 is not None and len(p.p0) >= 3:
                # AI2 P3: fuse p0 Z meets post-tilt neck tip
                p.p0 = [float(p.p0[0]), float(p.p0[1]), float(p1_new[2])]

    # B4b SCM: head-end (p1) only; p0 base fixed
    for p in parts:
        if "sternomastoid" not in p.name.lower():
            continue
        if p.p1 is not None and len(p.p1) >= 3:
            p.p1 = [float(p.p1[0]), float(p.p1[1]) + dy_tip, float(p.p1[2])]

    messages.append(f"neck_forward_tilt_deg={NECK_FORWARD_TILT_DEG}")
    messages.append("neck_column_tilt_applied: true")
    messages.append(f"neck_column_tip_dy_m={dy_tip}")
    messages.append(f"neck_column_head_comove: {str(moved_head).lower()}")


def _scale_part_radii(part: RecipePart, factor: float) -> None:
    """Uniform linear scale of radius / half-extents (B2 max 1.08 from original)."""
    if factor <= 1.0 + 1e-15:
        return
    f = float(factor)
    if part.rx_m is not None:
        part.rx_m = float(part.rx_m) * f
    if part.ry_m is not None:
        part.ry_m = float(part.ry_m) * f
    if part.rz_m is not None:
        part.rz_m = float(part.rz_m) * f
    if part.radius_m is not None:
        part.radius_m = float(part.radius_m) * f
    if part.top_half_width_m is not None:
        part.top_half_width_m = float(part.top_half_width_m) * f
    if part.bottom_half_width_m is not None:
        part.bottom_half_width_m = float(part.bottom_half_width_m) * f
    if part.half_depth_m is not None:
        part.half_depth_m = float(part.half_depth_m) * f


def _nudge_connection(
    child: RecipePart,
    parent: RecipePart,
    axis: int,
    *,
    original_scale_cap: dict[str, float],
) -> str:
    """Apply B2 algorithm to one child↔parent pair. Returns status token."""
    from meshops.proportion.connection_metrics import (
        gap_along_axis,
        is_toe_part,
        socket_overlap_m,
        sphere_proxy,
    )

    if is_toe_part(child.name):
        return "skipped"

    pc = sphere_proxy(child)
    pp = sphere_proxy(parent)
    if pc is None or pp is None:
        return "skipped"
    _c0, r_child = pc
    _c1, r_parent = pp
    if r_child <= 0.0 or r_parent <= 0.0:
        return "skipped"

    overlap = socket_overlap_m(r_child, r_parent)
    max_pull = 0.5 * (r_child + r_parent)

    gap = gap_along_axis(child, parent, axis)  # type: ignore[arg-type]
    if gap is None:
        return "skipped"
    if gap <= _JOIN_READY_EPS:
        return "overlapped"

    # 1) center-pull child along attach axis toward parent
    cc = sphere_proxy(child)
    cp = sphere_proxy(parent)
    if cc is None or cp is None:
        return "skipped"
    child_c, _ = cc
    parent_c, _ = cp
    direction = float(parent_c[axis]) - float(child_c[axis])
    if abs(direction) < 1e-15:
        # Same coord on axis but still gap from radii math — skip pull, try grow
        pull = 0.0
    else:
        sign = 1.0 if direction > 0.0 else -1.0
        pull = min(float(gap) + overlap / 2.0, max_pull) * sign
        _shift_part_along_axis(child, axis, pull)

    gap2 = gap_along_axis(child, parent, axis)  # type: ignore[arg-type]
    if gap2 is not None and gap2 <= _JOIN_READY_EPS:
        return "overlapped"

    # 2) grow child radius up to 1.08x original
    used = original_scale_cap.get(child.name, 1.0)
    remaining = _JOIN_READY_MAX_SCALE / used
    if remaining <= 1.0 + 1e-12:
        return "partial"

    gap_now = gap_along_axis(child, parent, axis)  # type: ignore[arg-type]
    pc2 = sphere_proxy(child)
    if gap_now is None or pc2 is None:
        return "partial"
    _, r_now = pc2
    # want dist - r_now*s - r_parent <= 0 → s >= (dist - r_parent) / r_now
    # dist = gap_now + r_now + r_parent
    dist = float(gap_now) + r_now + r_parent
    if r_now <= 1e-15:
        return "partial"
    needed = (dist - r_parent + overlap / 2.0) / r_now
    if needed <= 1.0:
        return "overlapped"
    scale = min(float(needed), float(remaining))
    if scale > 1.0 + 1e-12:
        _scale_part_radii(child, scale)
        original_scale_cap[child.name] = used * scale

    gap3 = gap_along_axis(child, parent, axis)  # type: ignore[arg-type]
    if gap3 is not None and gap3 <= _JOIN_READY_EPS:
        return "overlapped"
    return "partial"


def _apply_join_ready_overlaps(
    parts: list[RecipePart],
    messages: list[str],
) -> None:
    """0039 B2: nudge RECIPE_* centers/radii for connection classes (no SOCKET_* parts)."""
    from meshops.proportion.connection_metrics import resolve_join_connections

    scale_cap: dict[str, float] = {}
    class_status: dict[str, str] = {}

    for class_id, child, parent, axis in resolve_join_connections(parts):
        status = _nudge_connection(child, parent, int(axis), original_scale_cap=scale_cap)
        # Keep worst status per class: partial > overlapped > skipped
        prev = class_status.get(class_id)
        if prev == "partial" or status == "partial":
            class_status[class_id] = "partial"
        elif prev == "overlapped" or status == "overlapped":
            class_status[class_id] = "overlapped"
        else:
            class_status[class_id] = status

    for class_id in (
        "shoulder_l",
        "shoulder_r",
        "hip_l",
        "hip_r",
        "neck",
        "ankle_l",
        "ankle_r",
    ):
        st = class_status.get(class_id, "skipped")
        messages.append(f"join_ready.{class_id}: {st}")


def _apply_thigh_adduction(
    parts: list[RecipePart],
    template_applied: TemplateAppliedPackage | None,
    messages: list[str],
) -> None:
    """0046 B9: apply template thigh_tilt_deg with medial-shift cap + knee co-move.

    Keep hip p0 fixed; rotate thigh p1 in frontal plane (X-Z) toward midline;
    co-move knee_soft / calf_a / calf_cyl.p0 by the same world delta. Does not move
    ankle / foot / calf_b.
    """
    if template_applied is None:
        return
    raw_tilt = getattr(template_applied.constants, "thigh_tilt_deg", 0.0)
    try:
        tilt_f = float(raw_tilt)
    except (TypeError, ValueError):
        return
    if tilt_f != tilt_f or abs(tilt_f) <= 1e-6:  # NaN or ~0
        return
    tilt_req = max(-THIGH_TILT_DEG_CAP, min(THIGH_TILT_DEG_CAP, tilt_f))
    by_name = {p.name: p for p in parts}

    for side in ("l", "r"):
        thigh = by_name.get(f"RECIPE_limb_thigh_{side}")
        if (
            thigh is None
            or thigh.p0 is None
            or thigh.p1 is None
            or len(thigh.p0) < 3
            or len(thigh.p1) < 3
        ):
            continue
        p0 = [float(thigh.p0[0]), float(thigh.p0[1]), float(thigh.p0[2])]
        p1_old = [float(thigh.p1[0]), float(thigh.p1[1]), float(thigh.p1[2])]
        vx = p1_old[0] - p0[0]
        vy = p1_old[1] - p0[1]
        vz = p1_old[2] - p0[2]
        length = math.sqrt(vx * vx + vy * vy + vz * vz)
        if length <= _NEAR_ZERO_LEN:
            continue

        # Frontal-plane (X-Z) rotation toward midline: +tilt for left, -tilt for right
        # so p1.x moves medial (l: +, r: -) when the segment points roughly down.
        alpha = math.radians(tilt_req if side == "l" else -tilt_req)
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        vx_i = vx * ca - vz * sa
        vz_i = vx * sa + vz * ca
        # Preserve full 3d length after XZ rotation.
        scale_len = length / math.sqrt(vx_i * vx_i + vy * vy + vz_i * vz_i)
        p1_ideal = [
            p0[0] + vx_i * scale_len,
            p0[1] + vy * scale_len,
            p0[2] + vz_i * scale_len,
        ]
        delta = [p1_ideal[i] - p1_old[i] for i in range(3)]
        # Medial component of delta-x (positive = toward midline).
        medial_x = -delta[0] if side == "r" else delta[0]
        capped = False
        if abs(medial_x) > THIGH_ADDUCTION_MAX_MEDIAL_M + 1e-12 and abs(medial_x) > 1e-12:
            s = THIGH_ADDUCTION_MAX_MEDIAL_M / abs(medial_x)
            delta = [d * s for d in delta]
            capped = True
        p1_new = [p1_old[i] + delta[i] for i in range(3)]
        # Reproject onto length sphere about p0 (may slightly adjust medial).
        nvx = p1_new[0] - p0[0]
        nvy = p1_new[1] - p0[1]
        nvz = p1_new[2] - p0[2]
        nlen = math.sqrt(nvx * nvx + nvy * nvy + nvz * nvz)
        if nlen > _NEAR_ZERO_LEN:
            s_len = length / nlen
            p1_new = [p0[0] + nvx * s_len, p0[1] + nvy * s_len, p0[2] + nvz * s_len]
        delta_capped = [p1_new[i] - p1_old[i] for i in range(3)]
        medial_shift = abs(delta_capped[0])
        # Hard medial cap wins over tiny reproject overshoot (scale full world delta).
        if medial_shift > THIGH_ADDUCTION_MAX_MEDIAL_M + 1e-12:
            s = THIGH_ADDUCTION_MAX_MEDIAL_M / medial_shift
            delta_capped = [d * s for d in delta_capped]
            p1_new = [p1_old[i] + delta_capped[i] for i in range(3)]
            # Re-length after scale (length first); if X still exceeds, scale again once.
            nvx = p1_new[0] - p0[0]
            nvy = p1_new[1] - p0[1]
            nvz = p1_new[2] - p0[2]
            nlen = math.sqrt(nvx * nvx + nvy * nvy + nvz * nvz)
            if nlen > _NEAR_ZERO_LEN:
                s_len = length / nlen
                p1_new = [p0[0] + nvx * s_len, p0[1] + nvy * s_len, p0[2] + nvz * s_len]
            delta_capped = [p1_new[i] - p1_old[i] for i in range(3)]
            medial_shift = abs(delta_capped[0])
            if medial_shift > THIGH_ADDUCTION_MAX_MEDIAL_M + 1e-12:
                s = THIGH_ADDUCTION_MAX_MEDIAL_M / medial_shift
                delta_capped = [d * s for d in delta_capped]
                p1_new = [p1_old[i] + delta_capped[i] for i in range(3)]
                medial_shift = abs(delta_capped[0])
            capped = True

        thigh.p1 = p1_new

        # Knee-cluster co-move: same world delta_capped.
        for soft_name in (f"RECIPE_knee_soft_{side}", f"RECIPE_calf_a_{side}"):
            soft = by_name.get(soft_name)
            if soft is not None and soft.center is not None and len(soft.center) >= 3:
                soft.center = [
                    float(soft.center[0]) + delta_capped[0],
                    float(soft.center[1]) + delta_capped[1],
                    float(soft.center[2]) + delta_capped[2],
                ]
        cyl = by_name.get(f"RECIPE_calf_cyl_{side}")
        if cyl is not None and cyl.p0 is not None and len(cyl.p0) >= 3:
            cyl.p0 = [
                float(cyl.p0[0]) + delta_capped[0],
                float(cyl.p0[1]) + delta_capped[1],
                float(cyl.p0[2]) + delta_capped[2],
            ]

        msg = f"thigh_{side}: adduction_tilt_deg={tilt_req:.1f} medial_shift_m={medial_shift:.4f}"
        if capped:
            msg += " (capped from ideal)"
        messages.append(msg)


def _is_glute_soft_ellipsoid(part: RecipePart) -> bool:
    """0052 B5: seat pass gate — role glute_soft + ellipsoid + center len ≥ 3."""
    return (
        part.role == "glute_soft"
        and part.kind == "ellipsoid"
        and part.center is not None
        and len(part.center) >= 3
    )


def _glute_or_hip_half_depth_m(report: ProportionReport) -> float | None:
    """0052 local half soft-depth ladder (do not import body_template._soft_half_depth_m).

    1. depth_bands ``glute`` then ``hip`` → depth_m/2 when finite > 0
    2. measured landmark half-extent (hip_front/back or glute peak pair) when both finite
    3. else None (B2-only path)
    """
    for band_id in ("glute", "hip"):
        band = _depth_band(report, band_id)
        if band is None:
            continue
        depth = _finite_m(getattr(band, "depth_m", None))
        if depth is not None and depth > 0.0:
            return float(depth) / 2.0

    lms = report.landmarks_xyz

    def _lm_y(lm_id: str) -> float | None:
        lm = lms.get(lm_id)
        if lm is None:
            return None
        return _finite_m(lm.y_m)

    # Prefer hip_front / hip_back body-frame Y half-extent.
    y_f = _lm_y("hip_front")
    y_b = _lm_y("hip_back")
    if y_f is not None and y_b is not None:
        half = abs(float(y_b) - float(y_f)) / 2.0
        if half > 0.0:
            return half
    # Named glute front/back pairs when present (product may not emit these).
    for front_id, back_id in (
        ("glute_front", "glute_back"),
        ("glute_peak_front", "glute_peak_back"),
    ):
        y_front = _lm_y(front_id)
        y_back = _lm_y(back_id)
        if y_front is not None and y_back is not None:
            half = abs(float(y_back) - float(y_front)) / 2.0
            if half > 0.0:
                return half
    return None


def _part_rear_y_m(part: RecipePart) -> float | None:
    """Whole-part rear tip Y = center_y + depth half-extent when both finite.

    Prefer ``ry_m`` (ellipsoid). Fall back to ``half_depth_m`` for trap_box /
    box pelvis (0053-ready; AI1 Blind Spot 1) so RECIPE_pelvis_bucket contributes
    to B3 ref without needing a role-ellipsoid sibling.
    """
    if part.center is None or len(part.center) < 3:
        return None
    cy = _finite_m(float(part.center[1]))
    if cy is None:
        return None
    depth_extent = _finite_m(part.ry_m)
    if depth_extent is None:
        depth_extent = _finite_m(part.half_depth_m)
    if depth_extent is None:
        return None
    return float(cy) + float(depth_extent)


def _pelvis_ref_rear_y(parts: list[RecipePart]) -> tuple[float | None, str | None]:
    """0052 B3 ref rear ladder: max rear over pelvis oval/bucket/role/hip oval.

    Returns (ref_rear_y, short_name) for messaging. whole-part rear, not z-slice.
    """
    candidates: list[tuple[float, str]] = []

    def _add_named(name: str, short: str) -> None:
        for p in parts:
            if p.name != name:
                continue
            rear = _part_rear_y_m(p)
            if rear is not None:
                candidates.append((float(rear), short))
            return

    _add_named("RECIPE_pelvis_oval", "pelvis_oval")
    _add_named("RECIPE_pelvis_bucket", "pelvis_bucket")
    for p in parts:
        if p.role != "pelvis":
            continue
        rear = _part_rear_y_m(p)
        if rear is not None:
            candidates.append((float(rear), "pelvis"))
    _add_named("RECIPE_torso_oval_hip", "torso_oval_hip")

    if not candidates:
        return None, None
    best = max(candidates, key=lambda t: t[0])
    return best[0], best[1]


def _apply_glute_seat_mass(
    parts: list[RecipePart],
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
) -> None:
    """0052: floor glute_soft seat ry + push +Y so rear tip clears pelvis/hip ref.

    Mutates *parts* in place. Order: B1/B2 floors → B7 ry cap → B13 anisotropy →
    B3 beyond-ref Y → B12 y cap → B4 dual lock. Quiet when no glute_soft candidates.
    Never invents ry=0 when floors are missing (B5 skip). Leaves rx unchanged (B8).
    """
    idxs = [i for i, p in enumerate(parts) if _is_glute_soft_ellipsoid(p)]
    if not idxs:
        return  # P3-7 quiet — no spam on glute-less builds

    half = _glute_or_hip_half_depth_m(report)
    ref_rear, ref_name = _pelvis_ref_rear_y(parts)
    h_raw = m.height_m
    h_f: float | None = None
    if h_raw is not None:
        try:
            h_cand = float(h_raw)
        except (TypeError, ValueError):
            h_cand = float("nan")
        if math.isfinite(h_cand) and h_cand > 0.0:
            h_f = h_cand

    applied_depth_floor = False
    applied_rx_floor = False
    any_seated = False
    anisotropy_capped = False
    ry_stature_capped = False
    y_capped = False
    beyond_applied = False
    beyond_dy = 0.0
    target_rear_msg: float | None = None
    seated_idxs: list[int] = []

    for i in idxs:
        p = parts[i]
        ry_opt: float | None = None
        if p.ry_m is not None:
            try:
                ry_cand = float(p.ry_m)
            except (TypeError, ValueError):
                ry_cand = float("nan")
            if math.isfinite(ry_cand):
                ry_opt = ry_cand
        rx: float | None = None
        if p.rx_m is not None:
            try:
                rx_cand = float(p.rx_m)
            except (TypeError, ValueError):
                rx_cand = float("nan")
            if math.isfinite(rx_cand) and rx_cand > 0.0:
                rx = rx_cand

        floored = False
        ry_work = ry_opt if ry_opt is not None else 0.0
        if half is not None and half > 0.0:
            floor_d = GLUTE_SEAT_RY_FRAC_HALF_DEPTH * float(half)
            ry_work = max(ry_work, floor_d)
            floored = True
            applied_depth_floor = True
        if rx is not None:
            floor_rx = float(rx) * GLUTE_SEAT_RY_FROM_RX
            ry_work = max(ry_work, floor_rx)
            floored = True
            applied_rx_floor = True

        if not floored or ry_work <= 0.0:
            messages.append(f"glute_seat: skip {p.name} (no ry floor source)")
            continue

        ry_final = float(ry_work)
        if h_f is not None:
            ry_cap = GLUTE_SEAT_RY_CAP_FRAC_H * h_f
            if ry_final > ry_cap:
                ry_final = ry_cap
                ry_stature_capped = True

        if rx is not None and ry_final / float(rx) > GLUTE_SEAT_RY_ANISOTROPY_MAX:
            ry_final = GLUTE_SEAT_RY_ANISOTROPY_MAX * float(rx)
            anisotropy_capped = True

        p.ry_m = ry_final

        # B3: rear tip beyond ref (+Y only; never face-ward).
        center = p.center
        if center is None or len(center) < 3:
            continue
        cy = float(center[1])
        if ref_rear is not None:
            target_rear = float(ref_rear) + GLUTE_SEAT_BEYOND_REF_Y
            need_y = target_rear - ry_final
            if need_y > cy:
                beyond_dy = max(beyond_dy, need_y - cy)
                cy = need_y
                beyond_applied = True
                target_rear_msg = target_rear

        # B12: center_y soft stature cap.
        if h_f is not None and cy > GLUTE_SEAT_Y_CAP_FRAC_H * h_f:
            cy = GLUTE_SEAT_Y_CAP_FRAC_H * h_f
            y_capped = True

        p.center = [float(center[0]), float(cy), float(center[2])]
        any_seated = True
        seated_idxs.append(i)

    if applied_depth_floor and half is not None:
        floor_d = GLUTE_SEAT_RY_FRAC_HALF_DEPTH * float(half)
        messages.append(
            f"glute_seat: ry_floor_depth={floor_d:.4f} "
            f"(half={float(half):.4f} x {GLUTE_SEAT_RY_FRAC_HALF_DEPTH:.2f}) applied l/r"
        )
    elif applied_rx_floor and half is None:
        messages.append("glute_seat: depth missing; ry_from_rx only")

    if ry_stature_capped and h_f is not None:
        ry_cap_m = GLUTE_SEAT_RY_CAP_FRAC_H * h_f
        messages.append(
            f"glute_seat: ry_cap ry_m<={ry_cap_m:.4f} ({GLUTE_SEAT_RY_CAP_FRAC_H:.2f}xH)"
        )

    if anisotropy_capped:
        messages.append("glute_seat: ry_anisotropy_cap")

    if beyond_applied and ref_rear is not None and target_rear_msg is not None:
        messages.append(
            f"glute_seat: beyond_ref dy={beyond_dy:+.4f} "
            f"target_rear={target_rear_msg:.4f} ref={ref_name} "
            f"(whole-part max; not z-slice)"
        )
    elif ref_rear is None and any_seated:
        messages.append("glute_seat: beyond_ref skipped (no pelvis/hip ref)")

    if y_capped and h_f is not None:
        y_cap = GLUTE_SEAT_Y_CAP_FRAC_H * h_f
        messages.append(
            f"glute_seat: y_cap center_y<={y_cap:.4f} ({GLUTE_SEAT_Y_CAP_FRAC_H:.2f}xH)"
        )

    # B4 dual lock: same ry + center_y on all seated L/R (max of pair).
    if len(seated_idxs) >= 2:
        rys: list[float] = []
        cys: list[float] = []
        for i in seated_idxs:
            sp = parts[i]
            if sp.ry_m is not None:
                rys.append(float(sp.ry_m))
            sc = sp.center
            if sc is not None and len(sc) >= 3:
                cys.append(float(sc[1]))
        if rys and cys:
            lock_ry = max(rys)
            lock_y = max(cys)
            for i in seated_idxs:
                p = parts[i]
                p.ry_m = lock_ry
                sc = p.center
                if sc is not None and len(sc) >= 3:
                    p.center = [float(sc[0]), lock_y, float(sc[2])]
            messages.append(f"glute_seat: dual lock ry={lock_ry:.4f} y={lock_y:.4f}")


def _align_glute_outer_to_hip_bridge(
    parts: list[RecipePart],
    messages: list[str],
) -> None:
    """0036: for each side, set all glute_soft outer tips to hip_bridge outer X.

    right: center_x = hip_outer - half; left: center_x = hip_outer + half.
    Selects by role==glute_soft only (never name prefix — two_spheres uses
    RECIPE_glute_sphere_*). Aligns all matching on that side (safety net).
    """
    for side in ("l", "r"):
        glutes = [
            p for p in parts if p.role == "glute_soft" and _side_from_recipe_name(p.name) == side
        ]
        if not glutes:
            continue
        hip_outer = _hip_bridge_outer_x(parts, side)
        if hip_outer is None:
            messages.append(f"glute_{side}: outer X align skipped (no hip_bridge outer)")
            continue
        aligned = False
        missing_half = False
        for g in glutes:
            half = _half_extent_x_local(g)
            if half is None:
                missing_half = True
                continue
            target_x = hip_outer - half if side == "r" else hip_outer + half
            _set_part_x_local(g, target_x)
            aligned = True
        if aligned:
            messages.append(f"glute_{side}: outer X aligned to hip_bridge (|ΔX| target 0)")
        if missing_half:
            # Emit even when some siblings aligned (observability; product paths always set rx)
            messages.append(f"glute_{side}: outer X align skipped (no glute half-extent)")


def _resolve_breast_lower_z_m(report: ProportionReport) -> float | None:
    """0049 B2: measured lower-pole Z — prefer breast_lower, else L/R mean or single."""
    lms = report.landmarks_xyz
    bl = lms.get("breast_lower")
    if bl is not None and bl.z_m is not None:
        z = float(bl.z_m)
        if math.isfinite(z):
            return z
    zs: list[float] = []
    for key in ("breast_lower_l", "breast_lower_r"):
        lm = lms.get(key)
        if lm is not None and lm.z_m is not None:
            z = float(lm.z_m)
            if math.isfinite(z):
                zs.append(z)
    if not zs:
        return None
    return sum(zs) / float(len(zs))


def _chest_ref_z_for_hang(parts: list[RecipePart], m: _ResolvedMetrics) -> float | None:
    """0049 B6: prefer RECIPE_torso_oval_chest.center[2]; else m.chest_z."""
    for p in parts:
        if p.name == "RECIPE_torso_oval_chest" and p.center is not None and len(p.center) >= 3:
            z = float(p.center[2])
            if math.isfinite(z):
                return z
    if m.chest_z is not None:
        z = float(m.chest_z)
        if math.isfinite(z):
            return z
    return None


def _crotch_z_quiet(report: ProportionReport, height_m: float | None) -> float | None:
    """Crotch Z for breast hang floor without re-emitting fallback messages."""
    cp = report.landmarks_xyz.get("crotch_pubic")
    if cp is not None and cp.z_m is not None:
        z = float(cp.z_m)
        if math.isfinite(z):
            return z
    if height_m is not None:
        h = float(height_m)
        if math.isfinite(h) and h > 0.0:
            return CROTCH_Z_FRAC_FALLBACK * h
    return None


def _apply_breast_hang_z(
    parts: list[RecipePart],
    report: ProportionReport,
    m: _ResolvedMetrics,
    messages: list[str],
) -> None:
    """0049: drop breast_soft center Z for readable lower-pole hang (before 0033 tilt).

    Mutates *parts* in place. B3 gate: role breast_soft + ellipsoid + finite rz>0 only.
    Never pec_soft. B1 floor 0.55*rz; B2 measured lower deepen-only; dual L/R same Z.
    """
    eps = _NEAR_ZERO_LEN
    idxs = [
        i
        for i, p in enumerate(parts)
        if p.role == "breast_soft"
        and p.kind == "ellipsoid"
        and p.center is not None
        and len(p.center) >= 3
        and p.rz_m is not None
        and math.isfinite(float(p.rz_m))
        and float(p.rz_m) > 0.0
    ]
    if not idxs:
        messages.append("breast_hang_z_applied: false")
        return

    pre_zs: list[float] = []
    rzs: list[float] = []
    for i in idxs:
        p = parts[i]
        c = p.center
        rz = p.rz_m
        assert c is not None and rz is not None
        pre_zs.append(float(c[2]))
        rzs.append(float(rz))
    anchor_z = sum(pre_zs) / float(len(pre_zs))
    mean_rz = sum(rzs) / float(len(rzs))
    b1_drop_m = BREAST_HANG_Z_DROP_FRAC_RZ * mean_rz
    b1_center_z = anchor_z - b1_drop_m

    chest_ref_z = _chest_ref_z_for_hang(parts, m)

    source = "frac_rz"
    reason: str | None = None
    candidate_z = b1_center_z

    z_lower = _resolve_breast_lower_z_m(report)
    if z_lower is not None:
        in_band = True
        h = m.height_m
        if chest_ref_z is not None and h is not None and math.isfinite(float(h)) and float(h) > 0.0:
            lo = chest_ref_z - 0.12 * float(h)
            hi = chest_ref_z + 0.02 * float(h)
            if not (lo <= z_lower <= hi):
                in_band = False
                reason = "lower_out_of_band"
        if in_band:
            measured_center_z = z_lower + mean_rz
            if measured_center_z <= b1_center_z + eps:
                candidate_z = measured_center_z
                source = "breast_lower"
            else:
                candidate_z = b1_center_z
                source = "frac_rz"
                reason = "measured_shallow_using_frac"

    # B5 clamps: B1 floor re-assert + no raise vs pre, then waist/crotch floors.
    center_z = min(candidate_z, b1_center_z)
    pre_min = min(pre_zs)
    center_z = min(center_z, pre_min)

    waist: RecipePart | None = None
    for p in parts:
        if (
            p.name == "RECIPE_torso_oval_waist"
            and p.center is not None
            and len(p.center) >= 3
            and p.rz_m is not None
            and math.isfinite(float(p.rz_m))
        ):
            waist = p
            break
    if waist is not None and waist.center is not None:
        # Axis-aligned lower pole ≥ waist center: center_z - rz ≥ waist.z
        waist_floor = float(waist.center[2]) + mean_rz
        if center_z < waist_floor - eps:
            raised = min(waist_floor, pre_min)
            d2_min = BREAST_HANG_Z_MIN_DROP_FRAC_RZ * mean_rz
            # Preserve prior B2 reason (source ladder more informative than waist clamp).
            if reason is None:
                reason = (
                    "clamped_floor_soft" if (anchor_z - raised) + eps < d2_min else "clamped_floor"
                )
            center_z = raised

    crotch_z = _crotch_z_quiet(report, m.height_m)
    if crotch_z is not None and center_z < crotch_z - eps:
        # Preserve prior reason; emit crotch only when no earlier ladder/clamp reason.
        if reason is None:
            reason = "clamped_crotch"
        center_z = min(max(center_z, crotch_z), pre_min)

    # Dual lock: identical final Z on all gated breast_soft parts.
    final_z = float(center_z)
    for i in idxs:
        p = parts[i]
        assert p.center is not None
        new_center = [float(p.center[0]), float(p.center[1]), final_z]
        parts[i] = p.model_copy(update={"center": new_center})

    drop_m = anchor_z - final_z
    messages.append(f"breast_hang_z_drop_m={drop_m}")
    messages.append("breast_hang_z_applied: true")
    messages.append(f"breast_hang_z_source={source}")
    messages.append(f"breast_hang_z_anchor_m={anchor_z}")
    if chest_ref_z is not None:
        messages.append(f"breast_hang_z_chest_ref_m={chest_ref_z}")
    if reason is not None:
        messages.append(f"breast_hang_z_reason={reason}")


def _apply_breast_tilt(
    parts: list[RecipePart],
    *,
    tilt_val: float | None,
    messages: list[str],
) -> None:
    """0033: set rotation_euler_deg on breast_soft ellipsoids; emit applied messages.

    Mutates *parts* in place (replaces matching entries). Role-required — never
    pec_soft / glute_soft / kind-only. Hang source already resolved (B1 ladder).
    """
    if tilt_val is None:
        return

    messages.append(f"breast_tilt_deg={tilt_val}")

    finite = math.isfinite(float(tilt_val))
    apply = finite and abs(float(tilt_val)) >= _NEAR_ZERO_LEN
    breast_idxs = [
        i for i, p in enumerate(parts) if p.role == "breast_soft" and p.kind == "ellipsoid"
    ]

    if apply and breast_idxs:
        rot = [float(tilt_val), 0.0, 0.0]
        for i in breast_idxs:
            parts[i] = parts[i].model_copy(update={"rotation_euler_deg": rot})
        messages.append("breast_tilt_applied: true")
        messages.append("breast_tilt_axis=X sign=+tip_down_face_negY")
        return

    messages.append("breast_tilt_applied: false")
    if not finite:
        messages.append("breast_tilt_reason=nonfinite")
    elif not apply:
        messages.append("breast_tilt_reason=zero")
    elif not breast_idxs:
        messages.append("breast_tilt_reason=no_breast_soft")


def load_blockout_recipe(path: Path | str) -> BlockoutRecipePackage:
    """Load blockout_recipe.json; accepts schema 1.0.0 | 1.1.0 | 1.2.0 | 1.3.0 | 1.4.0."""
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
    if ver not in ("1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0"):
        raise ProportionError(
            "blockout recipe schema_version must be 1.0.0, 1.1.0, 1.2.0, "
            f"1.3.0, or 1.4.0 (got {ver!r})",
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
        if p.rotation_euler_deg is not None:
            entry["rotation_euler_deg"] = list(p.rotation_euler_deg)
        parts_data.append(entry)

    join_ready_flag = bool(getattr(package, "join_ready", False))
    lines: list[str] = [
        "# setup_blockout_recipe.py — MeshOps track 0019",
        f"# honesty: {RECIPE_HONESTY}",
        "# N6 / Difficulty §12: RECIPE primitives are authoring layout only —",
        "# not mesh reconstruction, not print-ready, not hero sculpt success.",
        f"# axis_notes: {AXIS_NOTES}",
        f"# recipe schema_version: {RECIPE_SCHEMA_VERSION}",
        f"# recipe_id: {package.recipe_id}",
        f"# join_ready: {join_ready_flag}",
        "# MeshOps face -Y: toes -Y, heels +Y. RECIPE only — not final mesh.",
        "# N1: do not whole-model voxel remesh from this script.",
        "",
        "import math",
        "import bpy",
        "from mathutils import Matrix, Vector, Euler",
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
        (
            "    rot = Vector((0.0, 0.0, 1.0)).rotation_difference(v.normalized())"
            ".to_matrix().to_4x4()"
        ),
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
        "def ensure_ellipsoid(",
        "    name, center_m, rx_m, ry_m, rz_m, collection, rotation_euler_deg=None",
        "):",
        "    # T @ R @ S — R from Euler XYZ degrees when rotation_euler_deg length 3",
        "    cx, cy, cz = to_bu(*center_m)",
        "    sx = rx_m / scale_len",
        "    sy = ry_m / scale_len",
        "    sz = rz_m / scale_len",
        "    T = Matrix.Translation(Vector((cx, cy, cz)))",
        "    R = Matrix.Identity(4)",
        "    if rotation_euler_deg is not None and len(rotation_euler_deg) == 3:",
        "        rx, ry, rz = rotation_euler_deg",
        "        R = Euler(",
        "            (math.radians(rx), math.radians(ry), math.radians(rz)), 'XYZ'",
        "        ).to_matrix().to_4x4()",
        "    S = (",
        "        Matrix.Scale(sx, 4, (1, 0, 0))",
        "        @ Matrix.Scale(sy, 4, (0, 1, 0))",
        "        @ Matrix.Scale(sz, 4, (0, 0, 1))",
        "    )",
        "    mat = T @ R @ S",
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
        "            name, p['center'], p['rx_m'], p['ry_m'], p['rz_m'], recipes_col,",
        "            p.get('rotation_euler_deg'),",
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
    join_ready: bool = False,
    breast_tilt_deg: float | None = None,
    template_applied: Path | str | None = None,
    profiles: str | None = None,
    skeleton: Path | str | None = None,
    face: bool = False,
    hair: HairTier = "none",
    neckline: NecklineTier = "none",
    hands: bool = False,
    feet: bool = False,
    fingers: FingerTier = "mitten",
    toes: ToeTier = "wedge",
) -> dict[str, Any]:
    """CLI helper: load report → build → write; return success payload."""
    if nofuse and join_ready:
        raise ProportionError(
            "nofuse and join-ready are mutually exclusive",
            code="recipe_failed",
        )

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
        join_ready=join_ready,
        breast_tilt_deg=breast_tilt_deg,
        template_applied=tpl,
        profile=profile_doc,
        skeleton=skel,
        face=face,
        hair=hair,
        neckline=neckline,
        hands=hands,
        feet=feet,
        fingers=fingers,
        toes=toes,
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
        "join_ready": bool(package.join_ready),
    }


def run_blockout_emit_setup(
    recipe_path: Path | str,
    out: Path | str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """0039: re-emit setup_blockout_recipe.py from existing recipe JSON (load→write only).

    Uses write_blockout_recipe(format=\"bpy\") — never bare emit_bpy_script from CLI.
    Does not re-run optimize or join-ready post-pass.
    """
    package = load_blockout_recipe(recipe_path)
    paths = write_blockout_recipe(out, package, format="bpy", force=force)
    return {
        "ok": True,
        "format": "bpy",
        "paths": [str(p) for p in paths],
        "counts": dict(package.counts),
        "messages": list(package.messages),
        "join_ready": bool(package.join_ready),
        "honesty": RECIPE_HONESTY,
    }


__all__ = [
    "AXIS_NOTES",
    "BPY_BASENAME",
    "CALF_BELLY_SCALE",
    "CALF_DIST_END_SCALE",
    "CALF_PROX_END_SCALE",
    "CROTCH_Z_FRAC_FALLBACK",
    "DELT_ARM_RADIUS_SCALE",
    "DELT_OUTER_X_FRAC",
    "DELT_RY_FRAC",
    "DELT_RZ_FRAC",
    "GLUTE_SEAT_BEYOND_REF_Y",
    "GLUTE_SEAT_RY_ANISOTROPY_MAX",
    "GLUTE_SEAT_RY_CAP_FRAC_H",
    "GLUTE_SEAT_RY_FRAC_HALF_DEPTH",
    "GLUTE_SEAT_RY_FROM_RX",
    "GLUTE_SEAT_Y_CAP_FRAC_H",
    "JSON_BASENAME",
    "KNEE_SOFT_FRAC",
    "KNEE_SOFT_MIN_FRAC_H",
    "LIMB_DISTAL_SOFT_SCALE",
    "MIDLINE_X_TOL_M",
    "NECK_FORWARD_TILT_DEG",
    "NECK_R_MAX_FRAC_HEAD_RX",
    "PELVIS_BUCKET_HALF_DEPTH_FRAC",
    "PELVIS_BUCKET_HW_FRAC",
    "PELVIS_BUCKET_Z_BOTTOM_FRAC_H",
    "PELVIS_BUCKET_Z_TOP_FRAC_H",
    "PELVIS_OVAL_RX_FRAC_HIP_HW",
    "PELVIS_OVAL_RY_FRAC_HALF_HIP",
    "PELVIS_OVAL_RY_OVER_RX_MAX",
    "PELVIS_OVAL_RZ_FLOOR_M",
    "PELVIS_OVAL_RZ_FRAC_H",
    "PELVIS_OVAL_RZ_OVER_H_MAX",
    "RECIPE_HONESTY",
    "RECIPE_ID",
    "RECIPE_SCHEMA_VERSION",
    "THIGH_ADDUCTION_MAX_MEDIAL_M",
    "THIGH_PROX_SOFT_SCALE",
    "THIGH_TILT_DEG_CAP",
    "_BASELINE_ROLES_NO_PROFILE",
    "_MICHELIN_FRAC",
    "BlockoutRecipePackage",
    "RecipeMetrics",
    "RecipePart",
    "_apply_glute_seat_mass",
    "_apply_join_ready_overlaps",
    "_apply_neck_column_priors",
    "_apply_thigh_adduction",
    "_glute_or_hip_half_depth_m",
    "_midpoint_of_joints",
    "_pelvis_ref_rear_y",
    "build_blockout_recipe",
    "emit_bpy_script",
    "load_blockout_recipe",
    "run_blockout_emit_setup",
    "run_blockout_recipe",
    "write_blockout_recipe",
]
