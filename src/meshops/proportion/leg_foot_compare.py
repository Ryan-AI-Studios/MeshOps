"""Leg/ankle/foot landmark compare: photo vs RECIPE vs optional live scene (track 0127).

Authoring QA only — LEG_FOOT_COMPARE_HONESTY. Not mesh or print success.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.analyze import load_report
from meshops.proportion.blockout_recipe import load_blockout_recipe
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import LEG_FOOT_COMPARE_HONESTY
from meshops.proportion.models import LandmarkXYZ, ProportionReport

LEG_FOOT_COMPARE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
LEG_FOOT_COMPARE_JSON: Final[str] = "leg_foot_compare.json"
LEG_FOOT_METRICS_JSON: Final[str] = "leg_foot_metrics.json"
LEG_FOOT_COMPARE_ROLES: Final[tuple[str, ...]] = (
    "gastroc_med_l",
    "gastroc_med_r",
    "gastroc_lat_l",
    "gastroc_lat_r",
    "malleolus_med_l",
    "malleolus_med_r",
    "malleolus_lat_l",
    "malleolus_lat_r",
    "arch_apex_l",
    "arch_apex_r",
    "achilles_l",
    "achilles_r",
    "ball_l",
    "ball_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "calf_front",
    "calf_back",
    "heel_l",
    "heel_r",
    "toe_l",
    "toe_r",
    "foot_front",
    "foot_back",
)
SUGGESTED_ACTIONS: Final[frozenset[str]] = frozenset(
    {"skip", "hold_priors", "soft_adjust", "session_leg_foot", "remake_0111"}
)
FORM_READ_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "ank_float_above_heel",
        "plate_past_toe_tips",
        "calf_pedestal",
        "arch_flat",
        "ball_narrow_of_toes",
        "missing_id",
    }
)
ANK_FLOAT_ABOVE_HEEL_M: Final[float] = 0.015
PLATE_PAST_TOE_TIPS_M: Final[float] = 0.015
CALF_PEDESTAL_M: Final[float] = 0.010
ARCH_FLAT_M: Final[float] = 0.004
BALL_NARROW_FRAC: Final[float] = 0.80

_ROLE_PART: dict[str, str] = {
    "gastroc_med_l": "RECIPE_calf_cyl_l",
    "gastroc_med_r": "RECIPE_calf_cyl_r",
    "gastroc_lat_l": "RECIPE_calf_cyl_l",
    "gastroc_lat_r": "RECIPE_calf_cyl_r",
    "malleolus_med_l": "RECIPE_ank_foot_l",
    "malleolus_med_r": "RECIPE_ank_foot_r",
    "malleolus_lat_l": "RECIPE_ank_foot_l",
    "malleolus_lat_r": "RECIPE_ank_foot_r",
    "arch_apex_l": "RECIPE_arch_soft_l",
    "arch_apex_r": "RECIPE_arch_soft_r",
    "achilles_l": "RECIPE_heel_l",
    "achilles_r": "RECIPE_heel_r",
    "ball_l": "RECIPE_ball_soft_l",
    "ball_r": "RECIPE_ball_soft_r",
    "knee_l": "RECIPE_knee_soft_l",
    "knee_r": "RECIPE_knee_soft_r",
    "ankle_l": "RECIPE_ank_foot_l",
    "ankle_r": "RECIPE_ank_foot_r",
    "calf_front": "RECIPE_calf_cyl_l",
    "calf_back": "RECIPE_calf_cyl_l",
    "heel_l": "RECIPE_heel_l",
    "heel_r": "RECIPE_heel_r",
    "toe_l": "RECIPE_toe_1_l",
    "toe_r": "RECIPE_toe_1_r",
    "foot_front": "RECIPE_foot_plate_l",
    "foot_back": "RECIPE_foot_plate_l",
}

# B4: soft_adjust only for new-id gastroc Y + arch Z.
# malleolus/ball/achilles/HAVE ids / DEPTH_PAIRS stay hold_priors.
_SOFT_ADJUST_IDS: Final[frozenset[str]] = frozenset(
    {
        "gastroc_med_l",
        "gastroc_med_r",
        "arch_apex_l",
        "arch_apex_r",
    }
)
_CAPSULE_ROLES: Final[frozenset[str]] = frozenset(
    {
        "gastroc_med_l",
        "gastroc_med_r",
        "gastroc_lat_l",
        "gastroc_lat_r",
        "calf_front",
        "calf_back",
        "toe_l",
        "toe_r",
    }
)

_KNOB: dict[str, str] = {
    "gastroc_med_l": "CALF_BELLY_REAR_FRAC",
    "gastroc_med_r": "CALF_BELLY_REAR_FRAC",
    "gastroc_lat_l": "CALF_BELLY_LAT_FRAC",
    "gastroc_lat_r": "CALF_BELLY_LAT_FRAC",
    "malleolus_med_l": "ANK_RY_FRAC_HALF_W",
    "malleolus_med_r": "ANK_RY_FRAC_HALF_W",
    "malleolus_lat_l": "ANK_RY_FRAC_HALF_W",
    "malleolus_lat_r": "ANK_RY_FRAC_HALF_W",
    "arch_apex_l": "ARCH_SOFT_RY_FRAC_HALF_DEPTH",
    "arch_apex_r": "ARCH_SOFT_RY_FRAC_HALF_DEPTH",
    "achilles_l": "ANK_RZ_FRAC_HALF_W",
    "achilles_r": "ANK_RZ_FRAC_HALF_W",
    "ball_l": "BALL_SOFT_RY_FRAC_HALF_DEPTH",
    "ball_r": "BALL_SOFT_RY_FRAC_HALF_DEPTH",
    "knee_l": "KNEE_SOFT_FRAC",
    "knee_r": "KNEE_SOFT_FRAC",
    "ankle_l": "ANK_RY_FRAC_HALF_W",
    "ankle_r": "ANK_RY_FRAC_HALF_W",
    "calf_front": "CALF_BELLY_REAR_FRAC",
    "calf_back": "CALF_BELLY_REAR_FRAC",
    "heel_l": "ANK_RY_FRAC_HALF_W",
    "heel_r": "ANK_RY_FRAC_HALF_W",
    "toe_l": "TOE_TIP_PAD_RY_FRAC",
    "toe_r": "TOE_TIP_PAD_RY_FRAC",
    "foot_front": "FOOT_LEN_VISUAL_MIN_FRAC_H",
    "foot_back": "FOOT_LEN_VISUAL_MIN_FRAC_H",
}

_KNOWN_DUMP_NAMES: Final[frozenset[str]] = frozenset(
    {
        "RECIPE_calf_a_l",
        "RECIPE_calf_a_r",
        "RECIPE_calf_cyl_l",
        "RECIPE_calf_cyl_r",
        "RECIPE_calf_taper_dist_l",
        "RECIPE_calf_taper_dist_r",
        "RECIPE_calf_b_l",
        "RECIPE_calf_b_r",
        "RECIPE_ank_foot_l",
        "RECIPE_ank_foot_r",
        "RECIPE_heel_l",
        "RECIPE_heel_r",
        "RECIPE_foot_plate_l",
        "RECIPE_foot_plate_r",
        "RECIPE_arch_soft_l",
        "RECIPE_arch_soft_r",
        "RECIPE_ball_soft_l",
        "RECIPE_ball_soft_r",
        "RECIPE_toe_1_l",
        "RECIPE_toe_1_r",
        "RECIPE_toe_tip_1_l",
        "RECIPE_toe_tip_1_r",
        "RECIPE_knee_soft_l",
        "RECIPE_knee_soft_r",
    }
)


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _as_vec3(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    coords: list[float] = []
    for i in range(3):
        item = _as_float(value[i])
        if item is None:
            return None
        coords.append(item)
    return coords


def _as_part_dict(part: Any) -> dict[str, Any]:
    if isinstance(part, dict):
        return part
    dump = getattr(part, "model_dump", None)
    if callable(dump):
        dumped = dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return {}


def _part_name(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("name") or "")
    name = getattr(part, "name", "")
    return str(name)


def _midpoint(p0: object, p1: object) -> list[float] | None:
    a = _as_vec3(p0)
    b = _as_vec3(p1)
    if a is None or b is None:
        return None
    return [(a[i] + b[i]) / 2.0 for i in range(3)]


def _center_from_part(part: dict[str, Any], *, prefer_p0: bool = False) -> list[float] | None:
    if prefer_p0:
        p0 = _as_vec3(part.get("p0"))
        if p0 is not None:
            return p0
    center = _as_vec3(part.get("center"))
    if center is not None:
        return center
    return _midpoint(part.get("p0"), part.get("p1"))


def extract_recipe_leg_foot_part(parts: list[Any], role_id: str) -> dict[str, Any] | None:
    """Extract one leg/foot role from RECIPE parts (B26/B39 capsule p0)."""
    want = _ROLE_PART.get(role_id)
    if want is None:
        return None
    match: dict[str, Any] | None = None
    for part in parts:
        if _part_name(part) == want:
            match = _as_part_dict(part)
            break
    if match is None:
        return None
    kind = str(match.get("kind") or "")
    prefer_p0 = role_id in _CAPSULE_ROLES or kind == "capsule"
    center = _center_from_part(match, prefer_p0=prefer_p0)
    if kind == "capsule" and center is None:
        return None
    ry = _as_float(match.get("ry_m"))
    if role_id == "foot_front" and center is not None and ry is not None:
        center = [center[0], center[1] - ry, center[2]]
    if role_id == "foot_back" and center is not None and ry is not None:
        center = [center[0], center[1] + ry, center[2]]
    return {
        "name": match.get("name"),
        "role": match.get("role"),
        "kind": kind or None,
        "center": center,
        "rx_m": match.get("rx_m"),
        "ry_m": match.get("ry_m"),
        "rz_m": match.get("rz_m"),
        "radius_m": match.get("radius_m"),
        "p0": match.get("p0"),
        "p1": match.get("p1"),
    }


class LegFootMetrics(BaseModel):
    """Sidecar leg/foot meters (not a ProportionReport field — stay 1.2.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    honesty: str = LEG_FOOT_COMPARE_HONESTY
    ank_bottom_vs_heel_bottom_m: float | None = None
    plate_front_vs_toe_tip_m: float | None = None
    calf_belly_r_m: float | None = None
    calf_taper_r_m: float | None = None
    ank_ry_over_rx: float | None = None
    arch_rise_m: float | None = None
    ball_span_m: float | None = None
    y_m: dict[str, float | None] = Field(default_factory=dict)


class LegFootCompareCoord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    confidence: float | None = None
    sources: list[str] = Field(default_factory=list)


class LegFootCompareDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float | None = None
    y: float | None = None
    z: float | None = None


class LegFootCompareRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    measured: LegFootCompareCoord | None = None
    recipe: dict[str, Any] | None = None
    live: dict[str, Any] | None = None
    delta_mm: LegFootCompareDelta | None = None
    knob: str | None = None
    form_read: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    suggested: Literal["skip", "hold_priors", "soft_adjust", "session_leg_foot", "remake_0111"] = (
        "skip"
    )


class LegFootComparePackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = LEG_FOOT_COMPARE_SCHEMA_VERSION
    honesty: str = LEG_FOOT_COMPARE_HONESTY
    region: Literal["leg_foot"] = "leg_foot"
    ok: bool = False
    roles: list[LegFootCompareRole] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    leg_foot_metrics: LegFootMetrics | None = None
    package_path: str | None = None


def _recipe_parts(recipe: Any | None) -> list[Any]:
    if recipe is None:
        return []
    if isinstance(recipe, dict) and isinstance(recipe.get("parts"), list):
        return list(recipe["parts"])
    raw_parts = getattr(recipe, "parts", None)
    if isinstance(raw_parts, list):
        return list(raw_parts)
    return []


def _find_part(parts: list[Any], name: str) -> dict[str, Any] | None:
    for part in parts:
        if _part_name(part) == name:
            return _as_part_dict(part)
    return None


def _surface_y(part: dict[str, Any] | None, *, rear: bool) -> float | None:
    if part is None:
        return None
    center = _as_vec3(part.get("center"))
    ry = _as_float(part.get("ry_m"))
    if center is None or ry is None:
        return None
    return center[1] + ry if rear else center[1] - ry


def _bottom_z(part: dict[str, Any] | None) -> float | None:
    if part is None:
        return None
    center = _as_vec3(part.get("center"))
    rz = _as_float(part.get("rz_m"))
    if center is None or rz is None:
        return None
    return center[2] - rz


def _ry_over_rx(part: dict[str, Any] | None) -> float | None:
    if part is None:
        return None
    rx = _as_float(part.get("rx_m"))
    ry = _as_float(part.get("ry_m"))
    if rx is None or ry is None or rx <= 0:
        return None
    return ry / rx


def _p0_z(part: dict[str, Any] | None) -> float | None:
    if part is None:
        return None
    p0 = _as_vec3(part.get("p0"))
    if p0 is None:
        return None
    return p0[2]


def _toe_splay_m(parts: list[Any]) -> float | None:
    left = _find_part(parts, "RECIPE_toe_1_l")
    right = _find_part(parts, "RECIPE_toe_1_r")
    if left is None or right is None:
        return None
    lp = _as_vec3(left.get("p0")) or _as_vec3(left.get("p1"))
    rp = _as_vec3(right.get("p0")) or _as_vec3(right.get("p1"))
    if lp is None or rp is None:
        return None
    return abs(rp[0] - lp[0])


def build_leg_foot_metrics(
    report: ProportionReport,
    recipe: Any | None = None,
) -> LegFootMetrics:
    """Compute leg_foot_metrics from landmarks_xyz (+ optional RECIPE parts)."""
    parts = _recipe_parts(recipe)
    ank = _find_part(parts, "RECIPE_ank_foot_l") or _find_part(parts, "RECIPE_ank_foot_r")
    heel = _find_part(parts, "RECIPE_heel_l") or _find_part(parts, "RECIPE_heel_r")
    plate = _find_part(parts, "RECIPE_foot_plate_l") or _find_part(parts, "RECIPE_foot_plate_r")
    tip = _find_part(parts, "RECIPE_toe_tip_1_l") or _find_part(parts, "RECIPE_toe_tip_1_r")
    cyl = _find_part(parts, "RECIPE_calf_cyl_l") or _find_part(parts, "RECIPE_calf_cyl_r")
    taper = _find_part(parts, "RECIPE_calf_taper_dist_l") or _find_part(
        parts, "RECIPE_calf_taper_dist_r"
    )
    arch = _find_part(parts, "RECIPE_arch_soft_l") or _find_part(parts, "RECIPE_arch_soft_r")

    ank_bottom = _bottom_z(ank)
    heel_bottom = _bottom_z(heel)
    ank_vs_heel: float | None = None
    if ank_bottom is not None and heel_bottom is not None:
        ank_vs_heel = ank_bottom - heel_bottom

    plate_front = _surface_y(plate, rear=False)
    tip_c = _as_vec3(tip.get("center")) if tip is not None else None
    tip_y = tip_c[1] if tip_c is not None else None
    plate_vs_tip: float | None = None
    if plate_front is not None and tip_y is not None:
        plate_vs_tip = plate_front - tip_y

    arch_c = _as_vec3(arch.get("center")) if arch is not None else None
    plate_c = _as_vec3(plate.get("center")) if plate is not None else None
    arch_rise: float | None = None
    if arch_c is not None and plate_c is not None:
        arch_rise = arch_c[2] - plate_c[2]

    lms = report.landmarks_xyz
    y_fields: dict[str, float | None] = {}
    for lid in LEG_FOOT_COMPARE_ROLES:
        lm = lms.get(lid)
        y_fields[lid] = _as_float(lm.y_m) if lm is not None else None

    ball_l = lms.get("ball_l")
    ball_r = lms.get("ball_r")
    xl = _as_float(ball_l.x_m) if ball_l is not None else None
    xr = _as_float(ball_r.x_m) if ball_r is not None else None
    ball_span: float | None = None
    if xl is not None and xr is not None:
        ball_span = abs(xr - xl)

    return LegFootMetrics(
        ank_bottom_vs_heel_bottom_m=ank_vs_heel,
        plate_front_vs_toe_tip_m=plate_vs_tip,
        calf_belly_r_m=_as_float(cyl.get("radius_m")) if cyl is not None else None,
        calf_taper_r_m=_as_float(taper.get("radius_m")) if taper is not None else None,
        ank_ry_over_rx=_ry_over_rx(ank),
        arch_rise_m=arch_rise,
        ball_span_m=ball_span,
        y_m=y_fields,
    )


def _measured_coord(lm: LandmarkXYZ | None) -> LegFootCompareCoord | None:
    if lm is None:
        return None
    x_m = _as_float(lm.x_m)
    y_m = _as_float(lm.y_m)
    z_m = _as_float(lm.z_m)
    if x_m is None and y_m is None and z_m is None:
        return None
    return LegFootCompareCoord(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        confidence=float(lm.confidence),
        sources=list(lm.sources),
    )


def _anchor_xyz(recipe: dict[str, Any] | None) -> list[float] | None:
    if recipe is None:
        return None
    kind = str(recipe.get("kind") or "")
    if kind == "capsule":
        p0 = _as_vec3(recipe.get("p0"))
        if p0 is not None:
            return p0
    center = _as_vec3(recipe.get("center"))
    if center is not None:
        return center
    return _as_vec3(recipe.get("p0"))


def _delta_mm(
    measured: LegFootCompareCoord | None,
    recipe: dict[str, Any] | None,
) -> LegFootCompareDelta | None:
    if measured is None or recipe is None:
        return None
    # B39: capsule roles use p0 (live calf_cyl center=null).
    anchor = _anchor_xyz(recipe)
    if anchor is None:
        return None
    mx = _as_float(measured.x_m)
    my = _as_float(measured.y_m)
    mz = _as_float(measured.z_m)
    dx = (mx - anchor[0]) * 1000.0 if mx is not None else None
    dy = (my - anchor[1]) * 1000.0 if my is not None else None
    dz = (mz - anchor[2]) * 1000.0 if mz is not None else None
    if dx is None and dy is None and dz is None:
        return None
    return LegFootCompareDelta(x=dx, y=dy, z=dz)


def _suggest(
    measured: LegFootCompareCoord | None,
    recipe: dict[str, Any] | None,
    delta: LegFootCompareDelta | None,
    role_id: str,
) -> Literal["skip", "hold_priors", "soft_adjust", "session_leg_foot", "remake_0111"]:
    if measured is None:
        return "skip"
    if recipe is None or _anchor_xyz(recipe) is None:
        return "skip"
    if role_id not in _SOFT_ADJUST_IDS:
        return "hold_priors"
    if delta is None:
        return "hold_priors"
    # B36: Y/Z-only consume — ignore delta.x.
    vals = [abs(v) for v in (delta.y, delta.z) if v is not None]
    if not vals:
        return "hold_priors"
    if max(vals) < 1.0:
        return "hold_priors"
    return "soft_adjust"


def _load_scene_dump(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot load scene dump: {path}: {exc}",
            code="leg_foot_compare_failed",
            details={"path": str(path)},
        ) from exc
    if not isinstance(raw, dict) or "parts" not in raw:
        raise ProportionError(
            f"scene dump must be an object with parts: {path}",
            code="leg_foot_compare_failed",
            details={"path": str(path)},
        )
    parts = raw.get("parts")
    if not isinstance(parts, list):
        raise ProportionError(
            f"scene dump parts must be a list: {path}",
            code="leg_foot_compare_failed",
            details={"path": str(path)},
        )
    out: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        if not isinstance(part, dict) or "name" not in part:
            raise ProportionError(
                f"scene dump part {i} must be an object with name",
                code="leg_foot_compare_failed",
                details={"path": str(path), "index": i},
            )
        name = str(part.get("name") or "")
        if name in _KNOWN_DUMP_NAMES:
            has_center = _as_vec3(part.get("center")) is not None
            has_caps = _as_vec3(part.get("p0")) is not None and _as_vec3(part.get("p1")) is not None
            if not has_center and not has_caps:
                raise ProportionError(
                    f"scene dump part {name!r} missing center or p0/p1",
                    code="leg_foot_compare_failed",
                    details={"path": str(path), "index": i, "name": name},
                )
        out.append(part)
    return out


def _has_finite_leg_foot(lms: dict[str, LandmarkXYZ]) -> bool:
    for lid in LEG_FOOT_COMPARE_ROLES:
        lm = lms.get(lid)
        if lm is None:
            continue
        if any(_as_float(v) is not None for v in (lm.x_m, lm.y_m, lm.z_m)):
            return True
    return False


def run_blockout_leg_foot_compare(
    report: Path | str,
    recipe: Path | str,
    out: Path | str,
    *,
    scene_dump: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Compare photo landmarks vs RECIPE vs optional live scene dump."""
    report_path = Path(report)
    recipe_path = Path(recipe)
    out_dir = Path(out)
    dest = out_dir / LEG_FOOT_COMPARE_JSON
    if dest.exists() and not force:
        raise ProportionError(
            f"leg/foot compare exists: {dest} (pass --force)",
            code="leg_foot_compare_failed",
            details={"path": str(dest)},
        )

    rep = load_report(report_path)
    pkg = load_blockout_recipe(recipe_path)
    parts = list(pkg.parts)
    live_parts: list[dict[str, Any]] | None = None
    if scene_dump is not None:
        live_parts = _load_scene_dump(Path(scene_dump))

    messages: list[str] = [
        "LEG_FOOT_COMPARE_HONESTY — authoring QA only; not mesh or print success",
    ]
    metrics = build_leg_foot_metrics(rep, recipe=pkg)
    ank_float = (
        metrics.ank_bottom_vs_heel_bottom_m is not None
        and metrics.ank_bottom_vs_heel_bottom_m >= ANK_FLOAT_ABOVE_HEEL_M
    )
    if ank_float:
        messages.append(f"ank_float_above_heel={metrics.ank_bottom_vs_heel_bottom_m:.4f}")
    plate_past = (
        metrics.plate_front_vs_toe_tip_m is not None
        and metrics.plate_front_vs_toe_tip_m <= -PLATE_PAST_TOE_TIPS_M
    )
    if plate_past:
        messages.append(f"plate_past_toe_tips_m={metrics.plate_front_vs_toe_tip_m:.4f}")

    calf_a = _find_part(parts, "RECIPE_calf_a_l") or _find_part(parts, "RECIPE_calf_a_r")
    cyl = _find_part(parts, "RECIPE_calf_cyl_l") or _find_part(parts, "RECIPE_calf_cyl_r")
    calf_a_c = _as_vec3(calf_a.get("center")) if calf_a is not None else None
    cyl_p0z = _p0_z(cyl)
    pedestal = False
    if calf_a_c is not None and cyl_p0z is not None:
        pedestal = (calf_a_c[2] - cyl_p0z) >= CALF_PEDESTAL_M
    if pedestal:
        messages.append("calf_pedestal")

    arch_flat = metrics.arch_rise_m is not None and metrics.arch_rise_m < ARCH_FLAT_M
    if arch_flat:
        messages.append(f"arch_flat_m={metrics.arch_rise_m:.4f}")

    splay = _toe_splay_m(parts)
    ball_narrow = False
    if metrics.ball_span_m is not None and splay is not None and splay > 0:
        ball_narrow = metrics.ball_span_m < BALL_NARROW_FRAC * splay
    if ball_narrow:
        messages.append(f"ball_narrow_of_toes span={metrics.ball_span_m:.4f}")

    lms = rep.landmarks_xyz
    roles: list[LegFootCompareRole] = []
    for lid in LEG_FOOT_COMPARE_ROLES:
        measured = _measured_coord(lms.get(lid))
        rec = extract_recipe_leg_foot_part(parts, lid)
        live = extract_recipe_leg_foot_part(live_parts, lid) if live_parts is not None else None
        delta = _delta_mm(measured, rec)
        form: list[str] = []
        if measured is None:
            form.append("missing_id")
        if ank_float and lid.startswith(("ankle_", "heel_", "malleolus_")):
            form.append("ank_float_above_heel")
        if plate_past and lid in ("foot_front", "toe_l", "toe_r"):
            form.append("plate_past_toe_tips")
        if pedestal and lid.startswith("gastroc_"):
            form.append("calf_pedestal")
        if arch_flat and lid.startswith("arch_apex_"):
            form.append("arch_flat")
        if ball_narrow and lid.startswith("ball_"):
            form.append("ball_narrow_of_toes")
        conf = 0.0
        if measured is not None and measured.confidence is not None:
            conf = float(measured.confidence)
        elif rec is not None:
            conf = 0.5
        roles.append(
            LegFootCompareRole(
                id=lid,
                measured=measured,
                recipe=rec,
                live=live,
                delta_mm=delta,
                knob=_KNOB.get(lid),
                form_read=form,
                confidence=conf,
                suggested=_suggest(measured, rec, delta, lid),
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    package = LegFootComparePackage(
        ok=True,
        roles=roles,
        messages=messages,
        leg_foot_metrics=metrics,
        package_path=str(dest),
    )
    dest.write_text(package.model_dump_json(indent=2) + "\n", encoding="utf-8")
    sidecar = out_dir / LEG_FOOT_METRICS_JSON
    if _has_finite_leg_foot(lms):
        sidecar.write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
    elif sidecar.exists():
        sidecar.unlink()
    return package.model_dump(mode="json")
