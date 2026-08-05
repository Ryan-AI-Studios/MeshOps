"""Proportion → skeleton-first joint/bone graph (track 0026).

Build BlockoutSkeleton from ProportionReport; emit JSON + Blender 5.2 bpy script.
Authoring scaffold only — not mesh, print, or animation rig success (Difficulty §12 / N6).

Freezes (do not re-learn):
- B1: joint meters = landmarks_xyz as-is (NO Y flip); axis_notes = guides.AXIS_NOTES
- B2: root=(0,0,0) scene marker; pelvis anatomical root; NO root→pelvis bone
- B3: height_m/head_unit_m from report; template_id only from --template-applied
- B4: no parent_bone field in schema v1
- B5: omit hand without fingertip; never zero-length hand bone
- B6: head←cranial_vertex; crown←hair_crown; head_bone=neck_top→head
- B7: spine ladder from navel/chest/belt_hip before stature fracs
- AI1 A/B: measured only full XYZ; length_m only when all 6 finite
- C1-C8: honesty literal, codes, string bpy, out=dir, format, force, counts, no strict-limbs
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final, Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.analyze import load_report
from meshops.proportion.depth_samples import DepthSamplesPackage
from meshops.proportion.errors import ProportionError
from meshops.proportion.guides import AXIS_NOTES
from meshops.proportion.honesty import SKELETON_HONESTY
from meshops.proportion.models import DepthBand, LandmarkXYZ, ProportionReport

SKELETON_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"

JSON_BASENAME: Final[str] = "blockout_skeleton.json"
BPY_BASENAME: Final[str] = "setup_skeleton.py"

# Omit bone when length would be below this (meters).
_LENGTH_EPS_M: Final[float] = 1e-3

# Hand joint offset along wrist->fingertip as fraction of forearm length (B5).
_HAND_FOREARM_FRAC: Final[float] = 0.08

# Default lateral half-widths as frac of height when estimating sides (retune with fixtures).
_SHOULDER_HALF_WIDTH_FRAC: Final[float] = 0.12
_HIP_HALF_WIDTH_FRAC: Final[float] = 0.09
# Mild A-pose arm depth offset (meters scale via height) - only when inventing missing arm Y.
_A_POSE_ARM_Y_FRAC: Final[float] = -0.05

SkeletonFormat = Literal["json", "bpy", "both"]
JointSource = Literal["measured", "estimated", "template", "missing"]
JointSide = Literal["l", "r", "none"]

# Stature Z fracs (soles=0). Cite Proportions.md / Grok-Human-Research; retune only with fixtures.
# Spec section 7 freeze.
STATURE_Z_FRAC: Final[dict[str, float]] = {
    "pelvis": 0.50,
    "spine_low": 0.55,
    "spine_mid": 0.62,
    "spine_high": 0.72,
    "shoulder": 0.82,
    "neck_base": 0.85,
    "neck_top": 0.88,
    "chin": 0.92,
    "head": 0.96,
    "crown": 1.00,
    "hip": 0.50,
    "knee": 0.28,
    "ankle": 0.04,
    "heel": 0.02,
    "toe": 0.02,
    "elbow": 0.62,
    "wrist": 0.48,
}


# ---------------------------------------------------------------------------
# Models (schema 1.0.0 — extra=forbid; no parent_bone B4)
# ---------------------------------------------------------------------------


class SkeletonJoint(BaseModel):
    """One joint node in the blockout skeleton graph (meters)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    parent: str | None
    side: JointSide
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    source: JointSource
    landmark_id: str | None = None


class SkeletonBone(BaseModel):
    """One bone edge between two joints (meters). No parent_bone in v1 (B4)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    joint_a: str
    joint_b: str
    length_m: float | None = None


class SkeletonCounts(BaseModel):
    """Joint-source counts only (C7); bones not source-bucketed."""

    model_config = ConfigDict(extra="forbid")

    joints: int = 0
    bones: int = 0
    measured: int = 0
    estimated: int = 0
    missing: int = 0


class BlockoutSkeleton(BaseModel):
    """blockout_skeleton.json package (schema 1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SKELETON_SCHEMA_VERSION
    honesty: str = SKELETON_HONESTY
    pose: Literal["a_pose"] = "a_pose"
    height_m: float | None = None
    head_unit_m: float | None = None
    axis_notes: str = AXIS_NOTES
    joints: list[SkeletonJoint] = Field(default_factory=list)
    bones: list[SkeletonBone] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    counts: SkeletonCounts = Field(default_factory=SkeletonCounts)
    template_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite(v: float | None) -> TypeGuard[float]:
    return v is not None and math.isfinite(v)


def _any_finite_xyz(x: float | None, y: float | None, z: float | None) -> bool:
    return _finite(x) or _finite(y) or _finite(z)


def _all_finite_xyz(x: float | None, y: float | None, z: float | None) -> bool:
    return _finite(x) and _finite(y) and _finite(z)


def _dist3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def _mid3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def _joint_xyz(j: SkeletonJoint) -> tuple[float, float, float] | None:
    if _finite(j.x_m) and _finite(j.y_m) and _finite(j.z_m):
        return (j.x_m, j.y_m, j.z_m)
    return None


def _head_unit_m(report: ProportionReport) -> float | None:
    h = report.height_m
    hu = report.head_unit_frac
    if h is not None and hu is not None and hu > 0.0:
        return float(h) * float(hu)
    return None


def _lm_map(report: ProportionReport) -> dict[str, LandmarkXYZ]:
    return dict(report.landmarks_xyz or {})


def _pick_lm(lms: dict[str, LandmarkXYZ], *ids: str) -> tuple[LandmarkXYZ | None, str | None]:
    for i in ids:
        lm = lms.get(i)
        if lm is not None:
            return lm, i
    return None, None


# Joint → (mid landmark ids priority, band ids priority). Arms omitted → None.
_DEPTH_FAMILY_MAP: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "pelvis": (("hip_mid",), ("hip",)),
    "hip_l": (("hip_mid",), ("hip",)),
    "hip_r": (("hip_mid",), ("hip",)),
    "spine_low": (("hip_mid", "glute_mid"), ("hip", "glute")),
    "spine_mid": (("breast_mid", "navel", "hip_mid", "chest_mid"), ("breast", "chest", "hip")),
    "spine_high": (("chest_mid",), ("chest",)),
    "shoulder_l": (("chest_mid",), ("chest",)),
    "shoulder_r": (("chest_mid",), ("chest",)),
    "neck_base": (("chest_mid",), ("chest",)),
    "head": (("cranial_mid",), ("cranial",)),
    "chin": (("cranial_mid",), ("cranial",)),
    "crown": (("cranial_mid",), ("cranial",)),
    "neck_top": (("cranial_mid",), ("cranial",)),
    "knee_l": (("thigh_mid",), ("thigh",)),
    "knee_r": (("thigh_mid",), ("thigh",)),
    "ankle_l": (("calf_mid", "foot_mid"), ("calf", "foot")),
    "ankle_r": (("calf_mid", "foot_mid"), ("calf", "foot")),
    "heel_l": (("foot_mid",), ("foot",)),
    "heel_r": (("foot_mid",), ("foot",)),
    "toe_l": (("foot_mid",), ("foot",)),
    "toe_r": (("foot_mid",), ("foot",)),
}


def _depth_family_for_joint(
    joint_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return (mid_ids priority, band_ids priority) or None for arm joints with no band."""
    if joint_id.startswith(("elbow_", "wrist_", "hand_")):
        return None
    return _DEPTH_FAMILY_MAP.get(joint_id)


def _resolve_depth_y_m(
    joint_id: str,
    lms: dict[str, LandmarkXYZ],
    bands: list[DepthBand] | None,
    samples: DepthSamplesPackage | None,
    height_m: float | None,
) -> tuple[float, str] | None:
    """(y_m, source_id) or None. Order: mid → band*H → samples (R1)."""
    family = _depth_family_for_joint(joint_id)
    if family is None:
        return None
    mid_ids, band_ids = family

    # 2. Named mid landmark from report landmarks_xyz
    for mid_id in mid_ids:
        lm = lms.get(mid_id)
        if lm is not None and _finite(lm.y_m):
            return float(lm.y_m), mid_id

    # 3. depth_bands: y_m = y_mid * height_m when height finite nonzero
    if bands and height_m is not None and math.isfinite(height_m) and height_m != 0.0:
        by_id = {b.band_id: b for b in bands}
        for bid in band_ids:
            band = by_id.get(bid)
            if band is not None and _finite(band.y_mid):
                y_m = float(band.y_mid) * float(height_m)
                if math.isfinite(y_m):
                    return y_m, bid

    # 4. Optional depth_at_landmarks.json samples
    if samples is not None:
        sample_list = samples.samples
        # Prefer role=landmark matching mid id
        for mid_id in mid_ids:
            for s in sample_list:
                if s.role == "landmark" and s.id == mid_id and _finite(s.y_m):
                    return float(s.y_m), mid_id
        # Else band_mid / band_id family
        for bid in band_ids:
            for s in sample_list:
                if not _finite(s.y_m):
                    continue
                if s.role == "band_mid" and (s.band_id == bid or s.id == f"band_{bid}_mid"):
                    return float(s.y_m), bid
                if s.band_id == bid and s.role in ("band_mid", "band_span"):
                    return float(s.y_m), bid
                if s.id == bid:
                    return float(s.y_m), bid

    return None


def _try_depth_y(
    joint_id: str,
    y: float | None,
    y_from_lm: bool,
    *,
    lms: dict[str, LandmarkXYZ],
    bands: list[DepthBand] | None,
    samples: DepthSamplesPackage | None,
    height_m: float | None,
    messages: list[str],
) -> tuple[float | None, bool, bool]:
    """If y missing, try depth ladder before invent. Returns (y, y_from_lm, y_from_depth)."""
    if y is not None:
        return y, y_from_lm, False
    got = _resolve_depth_y_m(joint_id, lms, bands, samples, height_m)
    if got is not None:
        y_m, src_id = got
        messages.append(f"joint {joint_id}: y_m from {src_id} (depth)")
        return y_m, False, True
    return None, False, False


def _joint_source(
    x: float | None,
    y: float | None,
    z: float | None,
    *,
    x_from_lm: bool,
    y_from_lm: bool,
    y_from_depth: bool,
    z_from_lm: bool,
) -> JointSource:
    """R3: measured only when full XYZ + real X/Z + direct/depth Y."""
    if not _any_finite_xyz(x, y, z):
        return "missing"
    if _all_finite_xyz(x, y, z) and x_from_lm and z_from_lm and (y_from_lm or y_from_depth):
        return "measured"
    return "estimated"


def _raw_coords_from_lm(
    lm: LandmarkXYZ | None,
) -> tuple[float | None, float | None, float | None, str | None, bool, bool, bool]:
    """Landmark meters as-is (B1 no flip). No invent. x/y/z_from flags."""
    if lm is None:
        return None, None, None, None, False, False, False
    xm: float | None = lm.x_m if _finite(lm.x_m) else None
    ym: float | None = lm.y_m if _finite(lm.y_m) else None
    zm: float | None = lm.z_m if _finite(lm.z_m) else None
    return xm, ym, zm, lm.id, xm is not None, ym is not None, zm is not None


def _raw_mean_pair_xyz(
    lms: dict[str, LandmarkXYZ],
    left_id: str,
    right_id: str,
) -> tuple[float | None, float | None, float | None, str | None, bool, bool, bool]:
    """Mean of left/right landmarks without inventing axes.

    Coordinates may come from one side alone, but R3 provenance flags are True
    only when **both** sides contribute that axis (pre-0035 pair measured gate).
    One-sided or mixed-axis stitch → coords usable, flags false → estimated.
    """
    left = lms.get(left_id)
    right = lms.get(right_id)
    if left is None and right is None:
        return None, None, None, None, False, False, False

    def _parts(lm: LandmarkXYZ | None) -> tuple[float | None, float | None, float | None]:
        if lm is None:
            return None, None, None
        return (
            lm.x_m if _finite(lm.x_m) else None,
            lm.y_m if _finite(lm.y_m) else None,
            lm.z_m if _finite(lm.z_m) else None,
        )

    lx, ly, lz = _parts(left)
    rx, ry, rz = _parts(right)

    def _mean(a: float | None, b: float | None) -> float | None:
        if a is not None and b is not None:
            return (a + b) / 2.0
        return a if a is not None else b

    xm = _mean(lx, rx)
    ym = _mean(ly, ry)
    zm = _mean(lz, rz)
    lid = (
        f"{left_id}+{right_id}"
        if left is not None and right is not None
        else (left_id if left is not None else right_id)
    )
    # R3 / pre-0035: pair mean is landmark-backed only when both sides contribute.
    x_from = lx is not None and rx is not None
    y_from = ly is not None and ry is not None
    z_from = lz is not None and rz is not None
    return (xm, ym, zm, lid, x_from, y_from, z_from)


def _stature_z(height_m: float | None, key: str) -> float | None:
    if height_m is None:
        return None
    frac = STATURE_Z_FRAC.get(key)
    if frac is None:
        return None
    return float(height_m) * frac


def _fill_missing_axis(
    x: float | None,
    y: float | None,
    z: float | None,
    *,
    default_x: float = 0.0,
    default_y: float = 0.0,
    default_z: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Fill null axes when at least one axis known or default_z provided."""
    out_x = x if x is not None else default_x
    out_y = y if y is not None else default_y
    out_z = z if z is not None else default_z
    return out_x, out_y, out_z


def _set_joint(
    joints: dict[str, SkeletonJoint],
    *,
    id: str,
    parent: str | None,
    side: JointSide,
    x_m: float | None,
    y_m: float | None,
    z_m: float | None,
    source: JointSource,
    landmark_id: str | None,
) -> SkeletonJoint:
    j = SkeletonJoint(
        id=id,
        parent=parent,
        side=side,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        source=source if _any_finite_xyz(x_m, y_m, z_m) else "missing",
        landmark_id=landmark_id,
    )
    if not _any_finite_xyz(j.x_m, j.y_m, j.z_m):
        j = j.model_copy(update={"source": "missing"})
    joints[id] = j
    return j


# ---------------------------------------------------------------------------
# Resolve joints
# ---------------------------------------------------------------------------


def _resolve_root(joints: dict[str, SkeletonJoint]) -> None:
    """B2: scene origin marker at (0,0,0), estimated, parent=null."""
    _set_joint(
        joints,
        id="root",
        parent=None,
        side="none",
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        source="estimated",
        landmark_id=None,
    )


def _resolve_pelvis(
    joints: dict[str, SkeletonJoint],
    lms: dict[str, LandmarkXYZ],
    height_m: float | None,
    messages: list[str],
    *,
    bands: list[DepthBand] | None = None,
    samples: DepthSamplesPackage | None = None,
) -> None:
    # Prefer mean hips, else crotch_pubic / belt_hip.
    xm, ym, zm, lid, x_from, y_from, z_from = _raw_mean_pair_xyz(lms, "hip_l", "hip_r")
    if not _any_finite_xyz(xm, ym, zm):
        lm, lid2 = _pick_lm(lms, "crotch_pubic", "belt_hip")
        xm, ym, zm, lid, x_from, y_from, z_from = _raw_coords_from_lm(lm)
        if lid is None:
            lid = lid2

    ym, y_from, y_depth = _try_depth_y(
        "pelvis",
        ym,
        y_from,
        lms=lms,
        bands=bands,
        samples=samples,
        height_m=height_m,
        messages=messages,
    )

    if zm is None:
        sz = _stature_z(height_m, "pelvis")
        if sz is not None:
            if not _any_finite_xyz(xm, ym, None):
                messages.append("joint pelvis: estimated from stature frac ~0.50")
            zm = sz
            z_from = False
    if xm is None:
        # Synthesized midline — not landmark X (R3).
        xm = 0.0
        x_from = False
    if ym is None and _finite(zm):
        ym = 0.0
        y_from = False
        y_depth = False
        messages.append("joint pelvis: front-plane placement (y_m estimated)")

    src = _joint_source(
        xm, ym, zm, x_from_lm=x_from, y_from_lm=y_from, y_from_depth=y_depth, z_from_lm=z_from
    )
    _set_joint(
        joints,
        id="pelvis",
        parent=None,
        side="none",
        x_m=xm,
        y_m=ym,
        z_m=zm,
        source=src,
        landmark_id=lid,
    )
    if joints["pelvis"].source == "missing":
        messages.append("joint pelvis: missing (no hips/crotch and no height_m)")


def _resolve_spine(
    joints: dict[str, SkeletonJoint],
    lms: dict[str, LandmarkXYZ],
    height_m: float | None,
    messages: list[str],
    *,
    bands: list[DepthBand] | None = None,
    samples: DepthSamplesPackage | None = None,
) -> None:
    """B7 spine ladder: mid<-navel; high<-chest_mid/mean z; low<-belt_hip or mid pelvis->mid."""
    pelvis = joints.get("pelvis")
    py = pelvis.y_m if pelvis is not None and _finite(pelvis.y_m) else 0.0
    pz = pelvis.z_m if pelvis is not None and _finite(pelvis.z_m) else None

    # --- spine_mid ← navel z @ x=0; depth Y from breast_mid etc. ---
    navel, navel_id = _pick_lm(lms, "navel")
    mx, my, mz, mlid, mx_from, my_from, mz_from = _raw_coords_from_lm(navel)
    if navel is not None and mlid is None:
        mlid = navel_id
    my, my_from, my_depth = _try_depth_y(
        "spine_mid",
        my,
        my_from,
        lms=lms,
        bands=bands,
        samples=samples,
        height_m=height_m,
        messages=messages,
    )
    if mz is None:
        mz = _stature_z(height_m, "spine_mid")
        if mz is not None:
            messages.append("joint spine_mid: estimated from stature frac ~0.62")
            mz_from = False
    if mx is None:
        mx = 0.0
        mx_from = False
    if my is None and mz is not None:
        my = py
        my_from = False
        my_depth = False
        messages.append("joint spine_mid: front-plane placement (y_m estimated)")
    msrc = _joint_source(
        mx, my, mz, x_from_lm=mx_from, y_from_lm=my_from, y_from_depth=my_depth, z_from_lm=mz_from
    )
    _set_joint(
        joints,
        id="spine_mid",
        parent="spine_low",  # parent chain set after spine_low exists — fix below
        side="none",
        x_m=mx,
        y_m=my,
        z_m=mz,
        source=msrc,
        landmark_id=mlid or navel_id,
    )

    # --- spine_high: prefer chest_mid (R4); else mean z chest front/back ---
    cm = lms.get("chest_mid")
    hx: float | None
    hy: float | None
    hz: float | None
    hlid: str | None
    hx_from: bool
    hy_from: bool
    hz_from: bool
    hy_depth: bool
    if cm is not None and _any_finite_xyz(cm.x_m, cm.y_m, cm.z_m):
        hx, hy, hz, hlid, hx_from, hy_from, hz_from = _raw_coords_from_lm(cm)
        hy, hy_from, hy_depth = _try_depth_y(
            "spine_high",
            hy,
            hy_from,
            lms=lms,
            bands=bands,
            samples=samples,
            height_m=height_m,
            messages=messages,
        )
        if hx is None:
            hx = 0.0
            hx_from = False
        if hz is None:
            hz = _stature_z(height_m, "spine_high")
            if hz is not None:
                messages.append("joint spine_high: estimated from stature frac ~0.72")
                hz_from = False
        if hy is None and hz is not None:
            hy = py
            hy_from = False
            hy_depth = False
            messages.append("joint spine_high: front-plane placement (y_m estimated)")
    else:
        chest_ids = ("chest_front", "chest_back", "underbust")
        zs: list[float] = []
        ys: list[float] = []
        used: list[str] = []
        for cid in chest_ids:
            lm = lms.get(cid)
            if lm is None:
                continue
            if _finite(lm.z_m):
                zs.append(lm.z_m)
                used.append(cid)
            if _finite(lm.y_m):
                ys.append(lm.y_m)
        hx = 0.0
        hx_from = False  # synthesized midline x=0 (R3)
        hy = (sum(ys) / len(ys)) if ys else None
        hy_from = hy is not None
        hy_depth = False
        hz = (sum(zs) / len(zs)) if zs else None
        hz_from = hz is not None
        hlid = "+".join(used) if used else None
        if hy is None:
            hy, hy_from, hy_depth = _try_depth_y(
                "spine_high",
                None,
                False,
                lms=lms,
                bands=bands,
                samples=samples,
                height_m=height_m,
                messages=messages,
            )
        if zs:
            if hy is None:
                hy = py
                hy_from = False
                hy_depth = False
                messages.append("joint spine_high: front-plane placement (y_m estimated)")
        else:
            hz = _stature_z(height_m, "spine_high")
            if hz is not None:
                messages.append("joint spine_high: estimated from stature frac ~0.72")
                hz_from = False
                if hy is None:
                    hy = py
                    hy_from = False
                    hy_depth = False
    hsrc = _joint_source(
        hx, hy, hz, x_from_lm=hx_from, y_from_lm=hy_from, y_from_depth=hy_depth, z_from_lm=hz_from
    )
    _set_joint(
        joints,
        id="spine_high",
        parent="spine_mid",
        side="none",
        x_m=hx,
        y_m=hy,
        z_m=hz,
        source=hsrc,
        landmark_id=hlid,
    )

    # --- spine_low ← belt_hip or between pelvis and spine_mid ---
    belt, belt_id = _pick_lm(lms, "belt_hip")
    lx, ly, lz, llid, lx_from, ly_from, lz_from = _raw_coords_from_lm(belt)
    ly, ly_from, ly_depth = _try_depth_y(
        "spine_low",
        ly,
        ly_from,
        lms=lms,
        bands=bands,
        samples=samples,
        height_m=height_m,
        messages=messages,
    )
    if not _any_finite_xyz(lx, None, lz):
        # No belt XZ — mid pelvis→spine_mid or stature (X/Z not landmark-backed).
        sm = joints["spine_mid"]
        if pz is not None and _finite(sm.z_m):
            lx = 0.0
            lx_from = False
            if ly is None:
                ly = py
                ly_from = False
                ly_depth = False
            lz = (pz + sm.z_m) / 2.0
            lz_from = False
            llid = None
            messages.append("joint spine_low: estimated mid pelvis->spine_mid")
        else:
            lz = _stature_z(height_m, "spine_low")
            if lz is not None:
                lx = 0.0
                lx_from = False
                if ly is None:
                    ly = py
                    ly_from = False
                    ly_depth = False
                lz_from = False
                messages.append("joint spine_low: estimated from stature frac ~0.55")
    else:
        if lx is None:
            lx = 0.0
            lx_from = False
        if lz is None:
            sm = joints["spine_mid"]
            if pz is not None and _finite(sm.z_m):
                lz = (pz + sm.z_m) / 2.0
                lz_from = False
                messages.append("joint spine_low: estimated mid pelvis->spine_mid")
            else:
                lz = _stature_z(height_m, "spine_low")
                if lz is not None:
                    lz_from = False
                    messages.append("joint spine_low: estimated from stature frac ~0.55")
        if ly is None and lz is not None:
            ly = py
            ly_from = False
            ly_depth = False
            messages.append("joint spine_low: front-plane placement (y_m estimated)")
    lsrc = _joint_source(
        lx, ly, lz, x_from_lm=lx_from, y_from_lm=ly_from, y_from_depth=ly_depth, z_from_lm=lz_from
    )
    _set_joint(
        joints,
        id="spine_low",
        parent="pelvis",
        side="none",
        x_m=lx,
        y_m=ly,
        z_m=lz,
        source=lsrc,
        landmark_id=llid or belt_id,
    )
    # Fix spine_mid parent now that spine_low exists.
    joints["spine_mid"] = joints["spine_mid"].model_copy(update={"parent": "spine_low"})


def _resolve_neck_head(
    joints: dict[str, SkeletonJoint],
    lms: dict[str, LandmarkXYZ],
    height_m: float | None,
    messages: list[str],
    *,
    bands: list[DepthBand] | None = None,
    samples: DepthSamplesPackage | None = None,
) -> None:
    sh = joints.get("spine_high")
    shy = sh.y_m if sh is not None and _finite(sh.y_m) else 0.0

    # neck_base: neck or mean shoulder @ x=0
    # Depth Y must not suppress X/Z fills (0035 P2 — limb-style independent axis fill).
    neck_lm, neck_lid = _pick_lm(lms, "neck")
    nx, ny, nz, nlid, nx_from, ny_from, nz_from = _raw_coords_from_lm(neck_lm)
    ny, ny_from, ny_depth = _try_depth_y(
        "neck_base",
        ny,
        ny_from,
        lms=lms,
        bands=bands,
        samples=samples,
        height_m=height_m,
        messages=messages,
    )
    if nz is None:
        _sx, sy, sz, slid, _sx_from, sy_from, sz_from = _raw_mean_pair_xyz(
            lms, "shoulder_l", "shoulder_r"
        )
        if sz is not None:
            if nx is None:
                nx, nx_from = 0.0, False
            if ny is None:
                ny = sy if sy is not None else shy
                ny_from = sy_from if sy is not None else False
                ny_depth = False
            nz, nz_from = sz, sz_from
            if nlid is None:
                nlid = slid
            messages.append("joint neck_base: estimated from mean shoulder z @ x=0")
        else:
            nz = _stature_z(height_m, "neck_base")
            if nz is not None:
                if nx is None:
                    nx, nx_from = 0.0, False
                if ny is None:
                    ny, ny_from, ny_depth = shy, False, False
                nz_from = False
                messages.append("joint neck_base: estimated from stature frac ~0.85")
    if nx is None:
        nx, nx_from = 0.0, False
    if ny is None and nz is not None:
        ny, ny_from, ny_depth = shy, False, False
        messages.append("joint neck_base: front-plane placement (y_m estimated)")
    nsrc = _joint_source(
        nx, ny, nz, x_from_lm=nx_from, y_from_lm=ny_from, y_from_depth=ny_depth, z_from_lm=nz_from
    )
    _set_joint(
        joints,
        id="neck_base",
        parent="spine_high",
        side="none",
        x_m=nx,
        y_m=ny,
        z_m=nz,
        source=nsrc,
        landmark_id=nlid or neck_lid,
    )

    # head ← cranial_vertex → hair_crown (B6); depth Y does not suppress Z fill
    head_lm, head_lid = _pick_lm(lms, "cranial_vertex", "hair_crown")
    hx, hy, hz, hlid, hx_from, hy_from, hz_from = _raw_coords_from_lm(head_lm)
    hy, hy_from, hy_depth = _try_depth_y(
        "head",
        hy,
        hy_from,
        lms=lms,
        bands=bands,
        samples=samples,
        height_m=height_m,
        messages=messages,
    )
    if hz is None:
        hz = _stature_z(height_m, "head")
        if hz is not None:
            if hx is None:
                hx, hx_from = 0.0, False
            if hy is None:
                hy, hy_from, hy_depth = shy, False, False
            hz_from = False
            messages.append("joint head: estimated from stature frac ~0.96")
    if hx is None:
        hx, hx_from = 0.0, False
    if hy is None and hz is not None:
        hy, hy_from, hy_depth = 0.0, False, False
        messages.append("joint head: front-plane placement (y_m estimated)")
    hsrc = _joint_source(
        hx, hy, hz, x_from_lm=hx_from, y_from_lm=hy_from, y_from_depth=hy_depth, z_from_lm=hz_from
    )
    _set_joint(
        joints,
        id="head",
        parent="neck_top",
        side="none",
        x_m=hx,
        y_m=hy,
        z_m=hz,
        source=hsrc,
        landmark_id=hlid or head_lid,
    )

    # neck_top: estimate between neck_base and head if no landmark; depth Y independent
    ntop_lm, ntop_lid = _pick_lm(lms, "neck_top")
    tx, ty, tz, tlid, tx_from, ty_from, tz_from = _raw_coords_from_lm(ntop_lm)
    ty, ty_from, ty_depth = _try_depth_y(
        "neck_top",
        ty,
        ty_from,
        lms=lms,
        bands=bands,
        samples=samples,
        height_m=height_m,
        messages=messages,
    )
    if tz is None and not _any_finite_xyz(tx, None, tz):
        # No landmark XZ — prefer mid neck_base→head when both full; else stature Z.
        nb = joints["neck_base"]
        hd = joints["head"]
        if _joint_xyz(nb) and _joint_xyz(hd):
            mid = _mid3(_joint_xyz(nb), _joint_xyz(hd))  # type: ignore[arg-type]
            if tx is None:
                tx, tx_from = mid[0], False
            if ty is None:
                ty, ty_from, ty_depth = mid[1], False, False
            tz, tz_from = mid[2], False
            messages.append("joint neck_top: estimated mid neck_base→head")
        else:
            tz = _stature_z(height_m, "neck_top")
            if tz is not None:
                if tx is None:
                    tx, tx_from = 0.0, False
                if ty is None:
                    ty, ty_from, ty_depth = shy, False, False
                tz_from = False
                messages.append("joint neck_top: estimated from stature frac ~0.88")
    elif tz is None:
        tz = _stature_z(height_m, "neck_top")
        if tz is not None:
            tz_from = False
            messages.append("joint neck_top: estimated from stature frac ~0.88")
    if tx is None:
        tx, tx_from = 0.0, False
    if ty is None and tz is not None:
        ty, ty_from, ty_depth = shy, False, False
    tsrc = _joint_source(
        tx, ty, tz, x_from_lm=tx_from, y_from_lm=ty_from, y_from_depth=ty_depth, z_from_lm=tz_from
    )
    _set_joint(
        joints,
        id="neck_top",
        parent="neck_base",
        side="none",
        x_m=tx,
        y_m=ty,
        z_m=tz,
        source=tsrc,
        landmark_id=tlid or ntop_lid,
    )
    # head parent already neck_top
    joints["head"] = joints["head"].model_copy(update={"parent": "neck_top"})

    # chin ← chin parent=head; depth Y does not suppress Z fill
    chin_lm, chin_lid = _pick_lm(lms, "chin")
    cx, cy, cz, clid, cx_from, cy_from, cz_from = _raw_coords_from_lm(chin_lm)
    cy, cy_from, cy_depth = _try_depth_y(
        "chin",
        cy,
        cy_from,
        lms=lms,
        bands=bands,
        samples=samples,
        height_m=height_m,
        messages=messages,
    )
    if cz is None:
        cz = _stature_z(height_m, "chin")
        if cz is not None:
            head_y = joints["head"].y_m
            if cx is None:
                cx, cx_from = 0.0, False
            if cy is None:
                cy = head_y if _finite(head_y) else 0.0
                cy_from, cy_depth = False, False
            cz_from = False
            messages.append("joint chin: estimated from stature frac ~0.92")
    if cx is None:
        cx, cx_from = 0.0, False
    if cy is None and cz is not None:
        cy, cy_from, cy_depth = 0.0, False, False
        messages.append("joint chin: front-plane placement (y_m estimated)")
    csrc = _joint_source(
        cx, cy, cz, x_from_lm=cx_from, y_from_lm=cy_from, y_from_depth=cy_depth, z_from_lm=cz_from
    )
    _set_joint(
        joints,
        id="chin",
        parent="head",
        side="none",
        x_m=cx,
        y_m=cy,
        z_m=cz,
        source=csrc,
        landmark_id=clid or chin_lid,
    )

    # crown ← hair_crown else copy head + message (B6); depth Y independent of Z fill
    crown_lm, crown_lid = _pick_lm(lms, "hair_crown")
    crx, cry, crz, crlid, crx_from, cry_from, crz_from = _raw_coords_from_lm(crown_lm)
    cry, cry_from, cry_depth = _try_depth_y(
        "crown",
        cry,
        cry_from,
        lms=lms,
        bands=bands,
        samples=samples,
        height_m=height_m,
        messages=messages,
    )
    if crz is None and not _any_finite_xyz(crx, None, crz):
        hd = joints["head"]
        if _any_finite_xyz(hd.x_m, hd.y_m, hd.z_m):
            if crx is None:
                crx, crx_from = hd.x_m, False
            if cry is None:
                cry, cry_from, cry_depth = hd.y_m, False, False
            crz, crz_from = hd.z_m, False
            crlid = hd.landmark_id
            messages.append("joint crown: copied from head (no hair_crown landmark)")
        else:
            crz = _stature_z(height_m, "crown")
            if crz is not None:
                if crx is None:
                    crx = 0.0
                if cry is None:
                    cry = 0.0
                crx_from, cry_from, cry_depth, crz_from = False, False, False, False
                messages.append("joint crown: estimated from stature frac ~1.00")
    elif crz is None:
        crz = _stature_z(height_m, "crown")
        if crz is not None:
            crz_from = False
            messages.append("joint crown: estimated from stature frac ~1.00")
    if crx is None:
        crx, crx_from = 0.0, False
    if cry is None and crz is not None:
        cry, cry_from, cry_depth = 0.0, False, False
        messages.append("joint crown: front-plane placement (y_m estimated)")
    crsrc = _joint_source(
        crx,
        cry,
        crz,
        x_from_lm=crx_from,
        y_from_lm=cry_from,
        y_from_depth=cry_depth,
        z_from_lm=crz_from,
    )
    _set_joint(
        joints,
        id="crown",
        parent="head",
        side="none",
        x_m=crx,
        y_m=cry,
        z_m=crz,
        source=crsrc,
        landmark_id=crlid or crown_lid,
    )


def _resolve_limb_side(
    joints: dict[str, SkeletonJoint],
    lms: dict[str, LandmarkXYZ],
    height_m: float | None,
    messages: list[str],
    *,
    side: Literal["l", "r"],
    sign: float,
    bands: list[DepthBand] | None = None,
    samples: DepthSamplesPackage | None = None,
) -> None:
    """Shoulder→elbow→wrist→hand and hip→knee→ankle→heel/toe for one side."""
    s = side
    sh_id = f"shoulder_{s}"
    el_id = f"elbow_{s}"
    wr_id = f"wrist_{s}"
    hand_id = f"hand_{s}"
    hip_id = f"hip_{s}"
    kn_id = f"knee_{s}"
    an_id = f"ankle_{s}"
    heel_id = f"heel_{s}"
    toe_id = f"toe_{s}"
    tip_id = f"fingertip_{s}"

    default_sh_x = None
    default_hip_x = None
    if height_m is not None:
        default_sh_x = sign * _SHOULDER_HALF_WIDTH_FRAC * float(height_m)
        default_hip_x = sign * _HIP_HALF_WIDTH_FRAC * float(height_m)
    default_arm_y = _A_POSE_ARM_Y_FRAC * float(height_m) if height_m is not None else 0.0

    def _depth_y(jid: str, y: float | None, y_from: bool) -> tuple[float | None, bool, bool]:
        return _try_depth_y(
            jid,
            y,
            y_from,
            lms=lms,
            bands=bands,
            samples=samples,
            height_m=height_m,
            messages=messages,
        )

    # --- shoulder ---
    lm, lid = _pick_lm(lms, sh_id)
    x, y, z, lid2, x_from, y_from, z_from = _raw_coords_from_lm(lm)
    lid = lid2 or lid
    y, y_from, y_depth = _depth_y(sh_id, y, y_from)
    if not _all_finite_xyz(x, y, z):
        if z is None:
            z = _stature_z(height_m, "shoulder")
            if z is not None:
                z_from = False
        if x is None and default_sh_x is not None:
            x = default_sh_x
            x_from = False
        if y is None and z is not None:
            y = default_arm_y
            y_from = False
            y_depth = False
            messages.append(f"joint {sh_id}: front-plane placement (y_m estimated)")
        elif y is None and z is None and x is None and default_sh_x is not None:
            messages.append(f"joint {sh_id}: estimated from stature/lateral defaults")
    src = _joint_source(
        x, y, z, x_from_lm=x_from, y_from_lm=y_from, y_from_depth=y_depth, z_from_lm=z_from
    )
    if (
        src == "estimated"
        and not _any_finite_xyz(
            lm.x_m if lm else None, lm.y_m if lm else None, lm.z_m if lm else None
        )
        and _any_finite_xyz(x, y, z)
        and z is not None
        and (not z_from or not x_from)
        and not any(f"joint {sh_id}: estimated from stature" in m for m in messages)
    ):
        messages.append(f"joint {sh_id}: estimated from stature/lateral defaults")
    _set_joint(
        joints,
        id=sh_id,
        parent="spine_high",
        side=s,
        x_m=x,
        y_m=y,
        z_m=z,
        source=src,
        landmark_id=lid,
    )
    if joints[sh_id].source == "missing":
        messages.append(f"joint {sh_id}: missing")

    # --- wrist (needed before elbow mid); arms have NO depth band (R2) ---
    lm, lid = _pick_lm(lms, wr_id)
    x, y, z, lid2, x_from, y_from, z_from = _raw_coords_from_lm(lm)
    lid = lid2 or lid
    y_depth = False
    if not _all_finite_xyz(x, y, z):
        if z is None:
            z = _stature_z(height_m, "wrist")
            if z is not None:
                z_from = False
        shj = joints[sh_id]
        if x is None:
            x = shj.x_m if _finite(shj.x_m) else default_sh_x
            x_from = False
        if y is None and z is not None:
            y = shj.y_m if _finite(shj.y_m) else default_arm_y
            y_from = False
            y_depth = False
            messages.append(f"joint {wr_id}: front-plane placement (y_m estimated)")
        if not _any_finite_xyz(
            lm.x_m if lm else None, lm.y_m if lm else None, lm.z_m if lm else None
        ) and _any_finite_xyz(x, y, z):
            messages.append(f"joint {wr_id}: estimated from stature/chain")
    src = _joint_source(
        x, y, z, x_from_lm=x_from, y_from_lm=y_from, y_from_depth=y_depth, z_from_lm=z_from
    )
    _set_joint(
        joints,
        id=wr_id,
        parent=el_id,
        side=s,
        x_m=x,
        y_m=y,
        z_m=z,
        source=src,
        landmark_id=lid,
    )

    # --- elbow: landmark or mid shoulder-wrist (chain inherit Y allowed; no band) ---
    lm, lid = _pick_lm(lms, el_id)
    x, y, z, lid2, x_from, y_from, z_from = _raw_coords_from_lm(lm)
    lid = lid2 or lid
    y_depth = False
    if not _all_finite_xyz(x, y, z):
        sh_xyz = _joint_xyz(joints[sh_id])
        wr_xyz = _joint_xyz(joints[wr_id])
        if sh_xyz is not None and wr_xyz is not None and not _any_finite_xyz(x, y, z):
            mid = _mid3(sh_xyz, wr_xyz)
            x, y, z = mid
            x_from, y_from, y_depth, z_from = False, False, False, False
            messages.append(f"joint {el_id}: estimated mid shoulder->wrist")
        else:
            if z is None:
                z = _stature_z(height_m, "elbow")
                if z is not None:
                    z_from = False
            shx = joints[sh_id].x_m
            shy2 = joints[sh_id].y_m
            if x is None:
                x = shx if _finite(shx) else default_sh_x
                x_from = False
            if y is None and z is not None:
                y = shy2 if _finite(shy2) else default_arm_y
                y_from = False
                y_depth = False
                messages.append(f"joint {el_id}: front-plane placement (y_m estimated)")
            if sh_xyz is not None and wr_xyz is not None:
                mid = _mid3(sh_xyz, wr_xyz)
                if x is None:
                    x, x_from = mid[0], False
                if y is None:
                    y, y_from, y_depth = mid[1], False, False
                if z is None:
                    z, z_from = mid[2], False
            if (
                not _any_finite_xyz(
                    lm.x_m if lm else None, lm.y_m if lm else None, lm.z_m if lm else None
                )
                and _any_finite_xyz(x, y, z)
                and not any(f"joint {el_id}:" in m for m in messages)
            ):
                messages.append(f"joint {el_id}: estimated from stature/chain")
    src = _joint_source(
        x, y, z, x_from_lm=x_from, y_from_lm=y_from, y_from_depth=y_depth, z_from_lm=z_from
    )
    _set_joint(
        joints,
        id=el_id,
        parent=sh_id,
        side=s,
        x_m=x,
        y_m=y,
        z_m=z,
        source=src,
        landmark_id=lid,
    )
    # wrist parent already elbow
    joints[wr_id] = joints[wr_id].model_copy(update={"parent": el_id})
    if joints[el_id].source == "missing":
        messages.append(f"joint {el_id}: missing")
    if joints[wr_id].source == "missing":
        messages.append(f"joint {wr_id}: missing")

    # --- hand only if fingertip known (B5); no depth band ---
    tip_lm = lms.get(tip_id)
    if tip_lm is not None and (_finite(tip_lm.x_m) or _finite(tip_lm.y_m) or _finite(tip_lm.z_m)):
        tip_x: float | None = tip_lm.x_m if _finite(tip_lm.x_m) else None
        tip_y: float | None = tip_lm.y_m if _finite(tip_lm.y_m) else None
        tip_z: float | None = tip_lm.z_m if _finite(tip_lm.z_m) else None
        if tip_x is not None and tip_z is not None and tip_y is None:
            tip_y = 0.0
            messages.append(f"joint {hand_id}: front-plane placement (y_m estimated)")
        wr_xyz = _joint_xyz(joints[wr_id])
        el_xyz = _joint_xyz(joints[el_id])
        forearm_len: float | None = None
        if wr_xyz is not None and el_xyz is not None:
            forearm_len = _dist3(el_xyz, wr_xyz)

        hand_xyz: tuple[float, float, float] | None = None
        if wr_xyz is not None and tip_x is not None and tip_y is not None and tip_z is not None:
            tip = (tip_x, tip_y, tip_z)
            if forearm_len is not None and forearm_len >= _LENGTH_EPS_M:
                dx = tip[0] - wr_xyz[0]
                dy = tip[1] - wr_xyz[1]
                dz = tip[2] - wr_xyz[2]
                dlen = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dlen >= _LENGTH_EPS_M:
                    scale = (_HAND_FOREARM_FRAC * forearm_len) / dlen
                    hand_xyz = (
                        wr_xyz[0] + dx * scale,
                        wr_xyz[1] + dy * scale,
                        wr_xyz[2] + dz * scale,
                    )
                else:
                    # Degenerate direction — mid wrist→tip
                    hand_xyz = _mid3(wr_xyz, tip)
                    messages.append(f"joint {hand_id}: mid wrist→fingertip (degenerate direction)")
            else:
                hand_xyz = _mid3(wr_xyz, tip)
                messages.append(f"joint {hand_id}: mid wrist→fingertip (forearm length null)")
        if hand_xyz is not None:
            # Guard zero-length vs wrist
            if wr_xyz is not None and _dist3(wr_xyz, hand_xyz) < _LENGTH_EPS_M:
                messages.append(f"joint {hand_id}: omitted (would be zero-length vs wrist)")
            else:
                _set_joint(
                    joints,
                    id=hand_id,
                    parent=wr_id,
                    side=s,
                    x_m=hand_xyz[0],
                    y_m=hand_xyz[1],
                    z_m=hand_xyz[2],
                    source="estimated",
                    landmark_id=tip_id,
                )
        else:
            messages.append(f"joint {hand_id}: omitted (incomplete fingertip/wrist)")
    else:
        messages.append(f"joint {hand_id}: omitted (no fingertip_{s} landmark)")

    # --- hip ---
    lm, lid = _pick_lm(lms, hip_id, f"greater_trochanter_{s}")
    x, y, z, lid2, x_from, y_from, z_from = _raw_coords_from_lm(lm)
    lid = lid2 or lid
    y, y_from, y_depth = _depth_y(hip_id, y, y_from)
    if not _all_finite_xyz(x, y, z):
        if z is None:
            z = _stature_z(height_m, "hip")
            if z is not None:
                z_from = False
        if x is None and default_hip_x is not None:
            x = default_hip_x
            x_from = False
        if y is None and z is not None:
            y = 0.0
            y_from = False
            y_depth = False
            messages.append(f"joint {hip_id}: front-plane placement (y_m estimated)")
        if not _any_finite_xyz(
            lm.x_m if lm else None, lm.y_m if lm else None, lm.z_m if lm else None
        ) and _any_finite_xyz(x, y, z):
            messages.append(f"joint {hip_id}: estimated from stature/lateral defaults")
    src = _joint_source(
        x, y, z, x_from_lm=x_from, y_from_lm=y_from, y_from_depth=y_depth, z_from_lm=z_from
    )
    _set_joint(
        joints,
        id=hip_id,
        parent="pelvis",
        side=s,
        x_m=x,
        y_m=y,
        z_m=z,
        source=src,
        landmark_id=lid,
    )
    if joints[hip_id].source == "missing":
        messages.append(f"joint {hip_id}: missing")

    # --- ankle before knee mid ---
    lm, lid = _pick_lm(lms, an_id)
    x, y, z, lid2, x_from, y_from, z_from = _raw_coords_from_lm(lm)
    lid = lid2 or lid
    y, y_from, y_depth = _depth_y(an_id, y, y_from)
    if not _all_finite_xyz(x, y, z):
        if z is None:
            z = _stature_z(height_m, "ankle")
            if z is not None:
                z_from = False
        hipj = joints[hip_id]
        if x is None:
            x = hipj.x_m if _finite(hipj.x_m) else default_hip_x
            x_from = False
        if y is None and z is not None:
            y = hipj.y_m if _finite(hipj.y_m) else 0.0
            y_from = False
            y_depth = False
            messages.append(f"joint {an_id}: front-plane placement (y_m estimated)")
        if not _any_finite_xyz(
            lm.x_m if lm else None, lm.y_m if lm else None, lm.z_m if lm else None
        ) and _any_finite_xyz(x, y, z):
            messages.append(f"joint {an_id}: estimated from stature/chain")
    src = _joint_source(
        x, y, z, x_from_lm=x_from, y_from_lm=y_from, y_from_depth=y_depth, z_from_lm=z_from
    )
    _set_joint(
        joints,
        id=an_id,
        parent=kn_id,
        side=s,
        x_m=x,
        y_m=y,
        z_m=z,
        source=src,
        landmark_id=lid,
    )

    # --- knee: landmark or mid hip-ankle ---
    lm, lid = _pick_lm(lms, kn_id)
    x, y, z, lid2, x_from, y_from, z_from = _raw_coords_from_lm(lm)
    lid = lid2 or lid
    y, y_from, y_depth = _depth_y(kn_id, y, y_from)
    if not _all_finite_xyz(x, y, z):
        hip_xyz = _joint_xyz(joints[hip_id])
        an_xyz = _joint_xyz(joints[an_id])
        if hip_xyz is not None and an_xyz is not None and not _any_finite_xyz(x, y, z):
            mid = _mid3(hip_xyz, an_xyz)
            x, y, z = mid
            x_from, y_from, y_depth, z_from = False, False, False, False
            messages.append(f"joint {kn_id}: estimated mid hip->ankle")
        else:
            if z is None:
                z = _stature_z(height_m, "knee")
                if z is not None:
                    z_from = False
            hpx = joints[hip_id].x_m
            hpy = joints[hip_id].y_m
            if x is None:
                x = hpx if _finite(hpx) else default_hip_x
                x_from = False
            if y is None and z is not None:
                y = hpy if _finite(hpy) else 0.0
                y_from = False
                y_depth = False
                messages.append(f"joint {kn_id}: front-plane placement (y_m estimated)")
            if hip_xyz is not None and an_xyz is not None:
                mid = _mid3(hip_xyz, an_xyz)
                if x is None:
                    x, x_from = mid[0], False
                if y is None:
                    y, y_from, y_depth = mid[1], False, False
                if z is None:
                    z, z_from = mid[2], False
            if (
                not _any_finite_xyz(
                    lm.x_m if lm else None, lm.y_m if lm else None, lm.z_m if lm else None
                )
                and _any_finite_xyz(x, y, z)
                and not any(f"joint {kn_id}:" in m for m in messages)
            ):
                messages.append(f"joint {kn_id}: estimated from stature/chain")
    src = _joint_source(
        x, y, z, x_from_lm=x_from, y_from_lm=y_from, y_from_depth=y_depth, z_from_lm=z_from
    )
    _set_joint(
        joints,
        id=kn_id,
        parent=hip_id,
        side=s,
        x_m=x,
        y_m=y,
        z_m=z,
        source=src,
        landmark_id=lid,
    )
    joints[an_id] = joints[an_id].model_copy(update={"parent": kn_id})
    if joints[kn_id].source == "missing":
        messages.append(f"joint {kn_id}: missing")
    if joints[an_id].source == "missing":
        messages.append(f"joint {an_id}: missing")

    # --- heel / toe parent ankle ---
    for jid, keys, zkey in (
        (heel_id, (heel_id,), "heel"),
        (toe_id, (toe_id,), "toe"),
    ):
        lm, lid = _pick_lm(lms, *keys)
        x, y, z, lid2, x_from, y_from, z_from = _raw_coords_from_lm(lm)
        lid = lid2 or lid
        y, y_from, y_depth = _depth_y(jid, y, y_from)
        if not _all_finite_xyz(x, y, z):
            anj = joints[an_id]
            if z is None:
                z = anj.z_m if _finite(anj.z_m) else _stature_z(height_m, zkey)
                # chain from ankle Z is not direct landmark on this joint
                if z is not None and not (lm is not None and _finite(lm.z_m)):
                    z_from = False
            if x is None:
                x = anj.x_m if _finite(anj.x_m) else default_hip_x
                x_from = False
            # Foot: toes -Y, heels +Y per AXIS_NOTES when inventing depth
            if y is None and z is not None:
                if jid.startswith("toe"):
                    y = -0.08 if height_m is None else -0.05 * height_m
                else:
                    y = 0.04 if height_m is None else 0.03 * height_m
                y_from = False
                y_depth = False
                if lm is not None and _finite(lm.x_m) and _finite(lm.z_m):
                    messages.append(f"joint {jid}: front-plane placement (y_m estimated)")
                else:
                    messages.append(f"joint {jid}: estimated near ankle (foot depth default)")
        src = _joint_source(
            x, y, z, x_from_lm=x_from, y_from_lm=y_from, y_from_depth=y_depth, z_from_lm=z_from
        )
        _set_joint(
            joints,
            id=jid,
            parent=an_id,
            side=s,
            x_m=x,
            y_m=y,
            z_m=z,
            source=src,
            landmark_id=lid,
        )


# ---------------------------------------------------------------------------
# Bones
# ---------------------------------------------------------------------------


def _bone_length(ja: SkeletonJoint, jb: SkeletonJoint) -> float | None:
    """length_m only when all six endpoint coords finite (AI1 B)."""
    a = _joint_xyz(ja)
    b = _joint_xyz(jb)
    if a is None or b is None:
        return None
    return _dist3(a, b)


def _maybe_bone(
    bones: list[SkeletonBone],
    joints: dict[str, SkeletonJoint],
    bone_id: str,
    joint_a: str,
    joint_b: str,
) -> None:
    ja = joints.get(joint_a)
    jb = joints.get(joint_b)
    if ja is None or jb is None:
        return
    length = _bone_length(ja, jb)
    if length is not None and length < _LENGTH_EPS_M:
        return  # omit zero/near-zero bones
    bones.append(SkeletonBone(id=bone_id, joint_a=joint_a, joint_b=joint_b, length_m=length))


def _build_bones(joints: dict[str, SkeletonJoint]) -> list[SkeletonBone]:
    bones: list[SkeletonBone] = []
    # Axial — no root→pelvis (B2)
    _maybe_bone(bones, joints, "spine_low_bone", "pelvis", "spine_low")
    _maybe_bone(bones, joints, "spine_mid_bone", "spine_low", "spine_mid")
    _maybe_bone(bones, joints, "spine_high_bone", "spine_mid", "spine_high")
    _maybe_bone(bones, joints, "neck_bone", "neck_base", "neck_top")
    _maybe_bone(bones, joints, "head_bone", "neck_top", "head")

    for s in ("l", "r"):
        _maybe_bone(bones, joints, f"upper_arm_{s}", f"shoulder_{s}", f"elbow_{s}")
        _maybe_bone(bones, joints, f"forearm_{s}", f"elbow_{s}", f"wrist_{s}")
        if f"hand_{s}" in joints:
            _maybe_bone(bones, joints, f"hand_{s}", f"wrist_{s}", f"hand_{s}")
        _maybe_bone(bones, joints, f"thigh_{s}", f"hip_{s}", f"knee_{s}")
        _maybe_bone(bones, joints, f"calf_{s}", f"knee_{s}", f"ankle_{s}")
        # foot = heel→toe else ankle→toe if heel missing
        if f"heel_{s}" in joints and _any_finite_xyz(
            joints[f"heel_{s}"].x_m,
            joints[f"heel_{s}"].y_m,
            joints[f"heel_{s}"].z_m,
        ):
            _maybe_bone(bones, joints, f"foot_{s}", f"heel_{s}", f"toe_{s}")
        else:
            _maybe_bone(bones, joints, f"foot_{s}", f"ankle_{s}", f"toe_{s}")

    return bones


def _counts(joints: list[SkeletonJoint], bones: list[SkeletonBone]) -> SkeletonCounts:
    measured = sum(1 for j in joints if j.source == "measured")
    estimated = sum(1 for j in joints if j.source in ("estimated", "template"))
    missing = sum(1 for j in joints if j.source == "missing")
    return SkeletonCounts(
        joints=len(joints),
        bones=len(bones),
        measured=measured,
        estimated=estimated,
        missing=missing,
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_blockout_skeleton(
    report: ProportionReport,
    *,
    template_id: str | None = None,
    depth_samples: DepthSamplesPackage | None = None,
) -> BlockoutSkeleton:
    """Resolve joints + bones from report. Raises skeleton_empty when D7 applies."""
    messages: list[str] = []
    height_m = float(report.height_m) if report.height_m is not None else None
    head_unit = _head_unit_m(report)
    lms = _lm_map(report)
    bands: list[DepthBand] = list(report.depth_bands or [])

    if report.quality.needs_user_input:
        messages.append("quality.needs_user_input: skeleton still emitted — confirm primary figure")
    if report.quality.multi_figure:
        messages.append("quality.multi_figure: skeleton still emitted — confirm primary figure")

    # D7 / skeleton_empty: no usable landmark meters and no height to estimate.
    has_lm_meters = any(_any_finite_xyz(lm.x_m, lm.y_m, lm.z_m) for lm in lms.values())
    if not has_lm_meters and height_m is None:
        raise ProportionError(
            "skeleton empty: no landmark meters and no height_m to estimate",
            code="skeleton_empty",
            details={"landmarks": len(lms), "height_m": height_m},
        )

    joints: dict[str, SkeletonJoint] = {}
    try:
        _resolve_root(joints)
        _resolve_pelvis(joints, lms, height_m, messages, bands=bands, samples=depth_samples)
        _resolve_spine(joints, lms, height_m, messages, bands=bands, samples=depth_samples)
        _resolve_neck_head(joints, lms, height_m, messages, bands=bands, samples=depth_samples)
        _resolve_limb_side(
            joints,
            lms,
            height_m,
            messages,
            side="l",
            sign=-1.0,
            bands=bands,
            samples=depth_samples,
        )
        _resolve_limb_side(
            joints,
            lms,
            height_m,
            messages,
            side="r",
            sign=1.0,
            bands=bands,
            samples=depth_samples,
        )
    except ProportionError:
        raise
    except Exception as exc:
        raise ProportionError(
            f"skeleton resolve failed: {exc}",
            code="skeleton_failed",
            details={},
        ) from exc

    # Stable emit order (core freeze)
    order = [
        "root",
        "pelvis",
        "spine_low",
        "spine_mid",
        "spine_high",
        "neck_base",
        "neck_top",
        "head",
        "chin",
        "crown",
        "shoulder_l",
        "shoulder_r",
        "elbow_l",
        "elbow_r",
        "wrist_l",
        "wrist_r",
        "hand_l",
        "hand_r",
        "hip_l",
        "hip_r",
        "knee_l",
        "knee_r",
        "ankle_l",
        "ankle_r",
        "heel_l",
        "heel_r",
        "toe_l",
        "toe_r",
    ]
    joint_list = [joints[i] for i in order if i in joints]
    bones = _build_bones(joints)
    counts = _counts(joint_list, bones)

    # Safety: only root finite and no height → empty (D7 residual)
    anatomical_finite = [
        j for j in joint_list if j.id != "root" and _any_finite_xyz(j.x_m, j.y_m, j.z_m)
    ]
    if not anatomical_finite and height_m is None:
        raise ProportionError(
            "skeleton empty: no finite anatomical joints and no height_m",
            code="skeleton_empty",
            details={"counts": counts.model_dump()},
        )

    return BlockoutSkeleton(
        schema_version=SKELETON_SCHEMA_VERSION,
        honesty=SKELETON_HONESTY,
        pose="a_pose",
        height_m=height_m,
        head_unit_m=head_unit,
        axis_notes=AXIS_NOTES,
        joints=joint_list,
        bones=bones,
        messages=messages,
        counts=counts,
        template_id=template_id,
    )


# ---------------------------------------------------------------------------
# bpy emit (string only — C3)
# ---------------------------------------------------------------------------


def _py_repr(obj: Any) -> str:
    return repr(obj)


def emit_bpy_script(package: BlockoutSkeleton) -> str:
    """Emit self-contained Blender 5.2 Python script (no meshops imports)."""
    joints_data: list[dict[str, Any]] = []
    for j in package.joints:
        joints_data.append(
            {
                "id": j.id,
                "parent": j.parent,
                "side": j.side,
                "x_m": j.x_m,
                "y_m": j.y_m,
                "z_m": j.z_m,
                "source": j.source,
            }
        )
    bones_data: list[dict[str, Any]] = []
    for b in package.bones:
        bones_data.append(
            {
                "id": b.id,
                "joint_a": b.joint_a,
                "joint_b": b.joint_b,
                "length_m": b.length_m,
            }
        )

    lines: list[str] = [
        "# setup_skeleton.py — MeshOps track 0026",
        f"# honesty: {SKELETON_HONESTY}",
        "# N6 / Difficulty §12: skeleton is authoring scaffold only —",
        "# not mesh reconstruction, not print-ready, not animation rig success.",
        f"# axis_notes: {AXIS_NOTES}",
        f"# skeleton schema_version: {SKELETON_SCHEMA_VERSION}",
        "# MeshOps face -Y: toes -Y, heels +Y. SKEL only — not final mesh.",
        "# N1: do not whole-model voxel remesh from this script.",
        "",
        "import math",
        "import bpy",
        "from mathutils import Matrix, Vector",
        "",
        "JOINTS = " + _py_repr(joints_data),
        "BONES = " + _py_repr(bones_data),
        f"HONESTY = {_py_repr(SKELETON_HONESTY)}",
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
        "def ensure_empty(name, loc_m, collection, display_size_m=0.03):",
        "    loc = to_bu(*loc_m)",
        "    obj = bpy.data.objects.get(name)",
        '    if obj is None or obj.type != "EMPTY":',
        "        if obj is not None:",
        "            bpy.data.objects.remove(obj, do_unlink=True)",
        "        obj = bpy.data.objects.new(name, None)",
        '        obj.empty_display_type = "PLAIN_AXES"',
        "        collection.objects.link(obj)",
        "    else:",
        "        _link_obj(obj, collection)",
        "    obj.location = loc",
        "    obj.empty_display_size = display_size_m / scale_len",
        '    obj["meshops_role"] = "skeleton_joint"',
        "    return obj",
        "",
        "",
        "def ensure_bone_stick(name, p0_m, p1_m, collection, radius_m=0.008):",
        "    p0 = Vector(to_bu(*p0_m))",
        "    p1 = Vector(to_bu(*p1_m))",
        "    v = p1 - p0",
        "    length = v.length",
        "    if length <= 1e-9:",
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
        '    obj["meshops_role"] = "skeleton_bone"',
        "    return obj",
        "",
        "",
        "skel_col = ensure_collection('Proportion_Skeleton')",
        "",
        "# hard-delete startup Cube + prior SKEL_* (AI1 C)",
        "for o in list(bpy.data.objects):",
        "    n = o.name",
        '    if n == "Cube" or n.startswith(("Cube.", "SKEL_")):',
        "        bpy.data.objects.remove(o, do_unlink=True)",
        "",
        "joint_xyz = {}",
        "n_joints = 0",
        "for j in JOINTS:",
        "    x, y, z = j.get('x_m'), j.get('y_m'), j.get('z_m')",
        "    if x is None or y is None or z is None:",
        "        continue",
        "    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):",
        "        continue",
        "    name = 'SKEL_' + j['id']",
        "    ensure_empty(name, (x, y, z), skel_col)",
        "    joint_xyz[j['id']] = (x, y, z)",
        "    n_joints += 1",
        "",
        "n_bones = 0",
        "for b in BONES:",
        "    a = joint_xyz.get(b['joint_a'])",
        "    c = joint_xyz.get(b['joint_b'])",
        "    if a is None or c is None:",
        "        continue",
        "    name = 'SKEL_' + b['id']",
        "    if ensure_bone_stick(name, a, c, skel_col) is not None:",
        "        n_bones += 1",
        "",
        "print(",
        "    f'MeshOps blockout skeleton: joints={n_joints} bones={n_bones} honesty={HONESTY}'",
        ")",
        "print('skeleton-build only — not mesh or print success')",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Write / run (C4 out=directory; C5 format; C6 force basenames only)
# ---------------------------------------------------------------------------


def write_blockout_skeleton(
    out_dir: Path | str,
    package: BlockoutSkeleton,
    *,
    format: SkeletonFormat = "json",
    force: bool = False,
) -> list[Path]:
    """Write blockout_skeleton.json and/or setup_skeleton.py under *out_dir*."""
    directory = Path(str(out_dir).rstrip("/\\"))
    fmt: SkeletonFormat = format
    written: list[Path] = []
    targets: list[tuple[Path, Literal["json", "bpy"]]] = []
    if fmt in ("json", "both"):
        targets.append((directory / JSON_BASENAME, "json"))
    if fmt in ("bpy", "both"):
        targets.append((directory / BPY_BASENAME, "bpy"))

    try:
        directory.mkdir(parents=True, exist_ok=True)
        for path, kind in targets:
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
            f"failed to write blockout skeleton: {exc}",
            code="write_failed",
            details={"out": str(directory)},
        ) from exc

    return written


def load_blockout_skeleton(path: Path | str) -> BlockoutSkeleton:
    """Load blockout_skeleton.json from file or directory."""
    p = Path(path)
    if p.is_dir():
        p = p / JSON_BASENAME
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot load blockout skeleton: {p}: {exc}",
            code="skeleton_failed",
            details={"path": str(p)},
        ) from exc
    try:
        return BlockoutSkeleton.model_validate(data)
    except Exception as exc:
        raise ProportionError(
            f"invalid blockout skeleton: {p}: {exc}",
            code="skeleton_failed",
            details={"path": str(p)},
        ) from exc


def _load_depth_at_landmarks_file(path: Path | str) -> DepthSamplesPackage:
    """File-only load of depth_at_landmarks.json (match recipe; code=skeleton_failed)."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return DepthSamplesPackage.model_validate(data)
    except FileNotFoundError as exc:
        raise ProportionError(
            f"depth-at-landmarks file not found: {p}",
            code="skeleton_failed",
            details={"path": str(p)},
        ) from exc
    except Exception as exc:
        raise ProportionError(
            f"invalid depth-at-landmarks package: {p}: {exc}",
            code="skeleton_failed",
            details={"path": str(p)},
        ) from exc


def run_skeleton_build(
    report_path: Path | str,
    out: Path | str,
    *,
    format: SkeletonFormat = "json",
    force: bool = False,
    template_applied: Path | str | None = None,
    depth_at_landmarks: Path | str | None = None,
) -> dict[str, Any]:
    """CLI helper: load report → build → write; return success payload."""
    report = load_report(report_path)

    template_id: str | None = None
    if template_applied is not None:
        from meshops.proportion.body_template import load_template_applied

        try:
            tpl = load_template_applied(template_applied)
        except ProportionError as exc:
            # C2: reuse template_unknown for bad template ids; else skeleton_failed.
            details = dict(exc.details or {})
            msg_l = str(exc).lower()
            if (
                exc.code == "template_unknown"
                or "unknown template" in msg_l
                or ("template_id" in details and "unknown" in msg_l)
            ):
                raise ProportionError(
                    str(exc),
                    code="template_unknown",
                    details=details,
                ) from exc
            raise ProportionError(
                str(exc),
                code="skeleton_failed",
                details=details,
            ) from exc
        template_id = tpl.template_id

    depth_pkg: DepthSamplesPackage | None = None
    if depth_at_landmarks is not None:
        depth_pkg = _load_depth_at_landmarks_file(depth_at_landmarks)

    package = build_blockout_skeleton(report, template_id=template_id, depth_samples=depth_pkg)
    paths = write_blockout_skeleton(out, package, format=format, force=force)
    return {
        "ok": True,
        "format": format,
        "paths": [str(p) for p in paths],
        "counts": package.counts.model_dump(mode="json"),
        "messages": list(package.messages),
        "honesty": SKELETON_HONESTY,
        "schema_version": SKELETON_SCHEMA_VERSION,
        "template_id": template_id,
        "height_m": package.height_m,
        "head_unit_m": package.head_unit_m,
    }


__all__ = [
    "AXIS_NOTES",
    "BPY_BASENAME",
    "JSON_BASENAME",
    "SKELETON_HONESTY",
    "SKELETON_SCHEMA_VERSION",
    "STATURE_Z_FRAC",
    "BlockoutSkeleton",
    "SkeletonBone",
    "SkeletonCounts",
    "SkeletonFormat",
    "SkeletonJoint",
    "_depth_family_for_joint",
    "build_blockout_skeleton",
    "emit_bpy_script",
    "load_blockout_skeleton",
    "run_skeleton_build",
    "write_blockout_skeleton",
]
