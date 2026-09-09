"""Hip/glute/groin landmark compare: photo vs RECIPE vs optional live scene (track 0126).

Authoring QA only — HIP_GLUTE_COMPARE_HONESTY. Not mesh or print success.
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
from meshops.proportion.honesty import HIP_GLUTE_COMPARE_HONESTY
from meshops.proportion.models import LandmarkXYZ, ProportionReport

HIP_GLUTE_COMPARE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
HIP_GLUTE_COMPARE_JSON: Final[str] = "hip_glute_compare.json"
HIP_GLUTE_METRICS_JSON: Final[str] = "hip_glute_metrics.json"
HIP_GLUTE_COMPARE_ROLES: Final[tuple[str, ...]] = (
    "asis_l",
    "asis_r",
    "psis_l",
    "psis_r",
    "glute_outer_l",
    "glute_outer_r",
    "glute_bottom_l",
    "glute_bottom_r",
    "glute_top_seam",
    "groin_fold_l",
    "groin_fold_r",
    "thigh_medial_l",
    "thigh_medial_r",
    "crotch_pubic",
    "greater_trochanter",
    "glute_peak_l",
    "glute_peak_r",
    "glute_cleft",
    "hip_front",
    "hip_back",
    "glute_front",
    "glute_back",
)
SUGGESTED_ACTIONS: Final[frozenset[str]] = frozenset(
    {"skip", "hold_priors", "soft_adjust", "session_hip_glute", "remake_0111"}
)
FORM_READ_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "glute_short_of_hip_soft_outer",
        "pelvis_front_behind_glute",
        "thigh_gap_wide",
        "glute_midline_gap",
        "groin_disconnected",
        "missing_id",
    }
)
GLUTE_SHORT_OF_HIP_SOFT_M: Final[float] = 0.040
PELVIS_BEHIND_GLUTE_M: Final[float] = 0.010
THIGH_GAP_WIDE_M: Final[float] = 0.070
GROIN_DISCONNECT_M: Final[float] = 0.040
GLUTE_MIDLINE_GAP_M: Final[float] = 0.060

_ROLE_PART: dict[str, str] = {
    "asis_l": "RECIPE_pelvis_oval",
    "asis_r": "RECIPE_pelvis_oval",
    "psis_l": "RECIPE_pelvis_oval",
    "psis_r": "RECIPE_pelvis_oval",
    "glute_outer_l": "RECIPE_glute_soft_l",
    "glute_outer_r": "RECIPE_glute_soft_r",
    "glute_bottom_l": "RECIPE_glute_soft_l",
    "glute_bottom_r": "RECIPE_glute_soft_r",
    "glute_top_seam": "RECIPE_glute_soft_l",
    "groin_fold_l": "RECIPE_pelvis_oval",
    "groin_fold_r": "RECIPE_pelvis_oval",
    "thigh_medial_l": "RECIPE_limb_thigh_l",
    "thigh_medial_r": "RECIPE_limb_thigh_r",
    "crotch_pubic": "RECIPE_pelvis_oval",
    "greater_trochanter": "RECIPE_hip_soft_l",
    "glute_peak_l": "RECIPE_glute_soft_l",
    "glute_peak_r": "RECIPE_glute_soft_r",
    "glute_cleft": "RECIPE_glute_soft_l",
    "hip_front": "RECIPE_torso_oval_hip",
    "hip_back": "RECIPE_torso_oval_hip",
    "glute_front": "RECIPE_glute_soft_l",
    "glute_back": "RECIPE_glute_soft_l",
}

# B4: soft_adjust only for new-id glute Y/Z consume (bottom/top_seam).
# glute_outer X / HAVE ids / DEPTH_PAIRS stay hold_priors (no 0036/0106/0092/0068 retune).
_SOFT_ADJUST_IDS: Final[frozenset[str]] = frozenset(
    {
        "glute_bottom_l",
        "glute_bottom_r",
        "glute_top_seam",
    }
)

_KNOB: dict[str, str] = {
    "asis_l": "PELVIS_OVAL_RY_FRAC_HALF_HIP",
    "asis_r": "PELVIS_OVAL_RY_FRAC_HALF_HIP",
    "psis_l": "PELVIS_OVAL_RY_FRAC_HALF_HIP",
    "psis_r": "PELVIS_OVAL_RY_FRAC_HALF_HIP",
    "glute_outer_l": "C_glute_outer",
    "glute_outer_r": "C_glute_outer",
    "glute_bottom_l": "GLUTE_SEAT_Y_FLOOR_M",
    "glute_bottom_r": "GLUTE_SEAT_Y_FLOOR_M",
    "glute_top_seam": "GLUTE_SEAT_Y_FLOOR_M",
    "groin_fold_l": "GLUTE_SEAT_Y_FLOOR_M",
    "groin_fold_r": "GLUTE_SEAT_Y_FLOOR_M",
    "thigh_medial_l": "THIGH_DIST_SHAFT_SCALE",
    "thigh_medial_r": "THIGH_DIST_SHAFT_SCALE",
    "crotch_pubic": "PELVIS_OVAL_RY_FRAC_HALF_HIP",
    "greater_trochanter": "HIP_SOFT_RY_FRAC_RX",
    "glute_peak_l": "GLUTE_SEAT_Y_FLOOR_M",
    "glute_peak_r": "GLUTE_SEAT_Y_FLOOR_M",
    "glute_cleft": "GLUTE_SEAT_Y_FLOOR_M",
    "hip_front": "TORSO_OVAL_RY_HIP_FRAC",
    "hip_back": "TORSO_OVAL_RY_HIP_FRAC",
    "glute_front": "GLUTE_SEAT_Y_FLOOR_M",
    "glute_back": "GLUTE_SEAT_Y_FLOOR_M",
}

_KNOWN_DUMP_NAMES: Final[frozenset[str]] = frozenset(
    {
        "RECIPE_torso_oval_hip",
        "RECIPE_pelvis_oval",
        "RECIPE_glute_soft_l",
        "RECIPE_glute_soft_r",
        "RECIPE_hip_soft_l",
        "RECIPE_hip_soft_r",
        "RECIPE_limb_thigh_l",
        "RECIPE_limb_thigh_r",
        "RECIPE_thigh_taper_dist_l",
        "RECIPE_thigh_taper_dist_r",
    }
)

_THIGH_ROLES: Final[frozenset[str]] = frozenset({"thigh_medial_l", "thigh_medial_r"})


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


def extract_recipe_hip_glute_part(parts: list[Any], role_id: str) -> dict[str, Any] | None:
    """Extract one hip/glute role from RECIPE parts (B26 thigh capsule p0)."""
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
    prefer_p0 = role_id in _THIGH_ROLES
    center = _center_from_part(match, prefer_p0=prefer_p0)
    if kind == "capsule" and center is None:
        return None
    ry = _as_float(match.get("ry_m"))
    if role_id in ("hip_front", "glute_front") and center is not None and ry is not None:
        center = [center[0], center[1] - ry, center[2]]
    if role_id in ("hip_back", "glute_back") and center is not None and ry is not None:
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


class HipGluteMetrics(BaseModel):
    """Sidecar hip/glute meters (not a ProportionReport field — stay 1.2.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    honesty: str = HIP_GLUTE_COMPARE_HONESTY
    glute_outer_vs_hip_soft_outer_m: float | None = None
    glute_rear_past_hip_oval_m: float | None = None
    pelvis_front_vs_glute_front_m: float | None = None
    glute_seat_y_m: float | None = None
    glute_ry_over_rx: float | None = None
    hip_soft_ry_over_rx: float | None = None
    thigh_gap_m: float | None = None
    y_m: dict[str, float | None] = Field(default_factory=dict)


class HipGluteCompareCoord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    confidence: float | None = None
    sources: list[str] = Field(default_factory=list)


class HipGluteCompareDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float | None = None
    y: float | None = None
    z: float | None = None


class HipGluteCompareRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    measured: HipGluteCompareCoord | None = None
    recipe: dict[str, Any] | None = None
    live: dict[str, Any] | None = None
    delta_mm: HipGluteCompareDelta | None = None
    knob: str | None = None
    form_read: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    suggested: Literal["skip", "hold_priors", "soft_adjust", "session_hip_glute", "remake_0111"] = (
        "skip"
    )


class HipGluteComparePackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = HIP_GLUTE_COMPARE_SCHEMA_VERSION
    honesty: str = HIP_GLUTE_COMPARE_HONESTY
    region: Literal["hip_glute"] = "hip_glute"
    ok: bool = False
    roles: list[HipGluteCompareRole] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    hip_glute_metrics: HipGluteMetrics | None = None
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


def _outer_x(part: dict[str, Any] | None) -> float | None:
    if part is None:
        return None
    center = _as_vec3(part.get("center"))
    rx = _as_float(part.get("rx_m"))
    if center is None or rx is None:
        return None
    return abs(center[0]) + rx


def _inner_x(part: dict[str, Any] | None) -> float | None:
    if part is None:
        return None
    center = _as_vec3(part.get("center"))
    rx = _as_float(part.get("rx_m"))
    if center is None or rx is None:
        return None
    return abs(center[0]) - rx


def _ry_over_rx(part: dict[str, Any] | None) -> float | None:
    if part is None:
        return None
    rx = _as_float(part.get("rx_m"))
    ry = _as_float(part.get("ry_m"))
    if rx is None or ry is None or rx <= 0:
        return None
    return ry / rx


def build_hip_glute_metrics(
    report: ProportionReport,
    recipe: Any | None = None,
) -> HipGluteMetrics:
    """Compute hip_glute_metrics from landmarks_xyz (+ optional RECIPE ovals)."""
    parts = _recipe_parts(recipe)
    hip = _find_part(parts, "RECIPE_torso_oval_hip")
    pelvis = _find_part(parts, "RECIPE_pelvis_oval")
    glute = _find_part(parts, "RECIPE_glute_soft_l") or _find_part(parts, "RECIPE_glute_soft_r")
    hip_soft = _find_part(parts, "RECIPE_hip_soft_l") or _find_part(parts, "RECIPE_hip_soft_r")

    glute_outer = _outer_x(glute)
    hip_soft_outer = _outer_x(hip_soft)
    outer_delta: float | None = None
    if glute_outer is not None and hip_soft_outer is not None:
        outer_delta = hip_soft_outer - glute_outer

    hip_rear = _surface_y(hip, rear=True)
    glute_rear = _surface_y(glute, rear=True)
    pelvis_front = _surface_y(pelvis, rear=False)
    glute_front = _surface_y(glute, rear=False)

    glute_cy: float | None = None
    glute_ry = _as_float(glute.get("ry_m")) if glute is not None else None
    if glute is not None:
        gc = _as_vec3(glute.get("center"))
        if gc is not None:
            glute_cy = gc[1]

    lms = report.landmarks_xyz
    y_fields: dict[str, float | None] = {}
    for lid in HIP_GLUTE_COMPARE_ROLES:
        lm = lms.get(lid)
        y_fields[lid] = _as_float(lm.y_m) if lm is not None else None

    # Overlay measured center Y first, then recompute surface pride/past with +ry
    # (plan slice B / 0133 lesson — do not leave RECIPE-derived past stale).
    measured_seat = y_fields.get("glute_bottom_l")
    if measured_seat is None:
        measured_seat = y_fields.get("glute_bottom_r")
    if measured_seat is None:
        measured_seat = y_fields.get("glute_top_seam")
    if measured_seat is not None:
        glute_cy = measured_seat
        if glute_ry is not None:
            glute_front = glute_cy - glute_ry
            glute_rear = glute_cy + glute_ry

    rear_past: float | None = None
    if glute_rear is not None and hip_rear is not None:
        rear_past = glute_rear - hip_rear
    pelvis_vs_glute: float | None = None
    if pelvis_front is not None and glute_front is not None:
        pelvis_vs_glute = pelvis_front - glute_front

    thigh_l = lms.get("thigh_medial_l")
    thigh_r = lms.get("thigh_medial_r")
    xl = _as_float(thigh_l.x_m) if thigh_l is not None else None
    xr = _as_float(thigh_r.x_m) if thigh_r is not None else None
    thigh_gap: float | None = None
    if xl is not None and xr is not None:
        thigh_gap = abs(xr - xl)

    return HipGluteMetrics(
        glute_outer_vs_hip_soft_outer_m=outer_delta,
        glute_rear_past_hip_oval_m=rear_past,
        pelvis_front_vs_glute_front_m=pelvis_vs_glute,
        glute_seat_y_m=glute_cy,
        glute_ry_over_rx=_ry_over_rx(glute),
        hip_soft_ry_over_rx=_ry_over_rx(hip_soft),
        thigh_gap_m=thigh_gap,
        y_m=y_fields,
    )


def _measured_coord(lm: LandmarkXYZ | None) -> HipGluteCompareCoord | None:
    if lm is None:
        return None
    x_m = _as_float(lm.x_m)
    y_m = _as_float(lm.y_m)
    z_m = _as_float(lm.z_m)
    if x_m is None and y_m is None and z_m is None:
        return None
    return HipGluteCompareCoord(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        confidence=float(lm.confidence),
        sources=list(lm.sources),
    )


def _delta_mm(
    measured: HipGluteCompareCoord | None,
    recipe: dict[str, Any] | None,
) -> HipGluteCompareDelta | None:
    if measured is None or recipe is None:
        return None
    center = _as_vec3(recipe.get("center"))
    if center is None:
        return None
    mx = _as_float(measured.x_m)
    my = _as_float(measured.y_m)
    mz = _as_float(measured.z_m)
    dx = (mx - center[0]) * 1000.0 if mx is not None else None
    dy = (my - center[1]) * 1000.0 if my is not None else None
    dz = (mz - center[2]) * 1000.0 if mz is not None else None
    if dx is None and dy is None and dz is None:
        return None
    return HipGluteCompareDelta(x=dx, y=dy, z=dz)


def _suggest(
    measured: HipGluteCompareCoord | None,
    recipe: dict[str, Any] | None,
    delta: HipGluteCompareDelta | None,
    role_id: str,
) -> Literal["skip", "hold_priors", "soft_adjust", "session_hip_glute", "remake_0111"]:
    if measured is None:
        return "skip"
    if recipe is None or recipe.get("center") is None:
        return "skip"
    if role_id not in _SOFT_ADJUST_IDS:
        return "hold_priors"
    if delta is None:
        return "hold_priors"
    # B36 analog (0127): Y/Z-only consume — ignore delta.x.
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
            code="hip_glute_compare_failed",
            details={"path": str(path)},
        ) from exc
    if not isinstance(raw, dict) or "parts" not in raw:
        raise ProportionError(
            f"scene dump must be an object with parts: {path}",
            code="hip_glute_compare_failed",
            details={"path": str(path)},
        )
    parts = raw.get("parts")
    if not isinstance(parts, list):
        raise ProportionError(
            f"scene dump parts must be a list: {path}",
            code="hip_glute_compare_failed",
            details={"path": str(path)},
        )
    out: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        if not isinstance(part, dict) or "name" not in part:
            raise ProportionError(
                f"scene dump part {i} must be an object with name",
                code="hip_glute_compare_failed",
                details={"path": str(path), "index": i},
            )
        name = str(part.get("name") or "")
        if name in _KNOWN_DUMP_NAMES:
            has_center = _as_vec3(part.get("center")) is not None
            has_caps = _as_vec3(part.get("p0")) is not None and _as_vec3(part.get("p1")) is not None
            if not has_center and not has_caps:
                raise ProportionError(
                    f"scene dump part {name!r} missing center or p0/p1",
                    code="hip_glute_compare_failed",
                    details={"path": str(path), "index": i, "name": name},
                )
        out.append(part)
    return out


def _has_finite_hip_glute(lms: dict[str, LandmarkXYZ]) -> bool:
    for lid in HIP_GLUTE_COMPARE_ROLES:
        lm = lms.get(lid)
        if lm is None:
            continue
        if any(_as_float(v) is not None for v in (lm.x_m, lm.y_m, lm.z_m)):
            return True
    return False


def run_blockout_hip_glute_compare(
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
    dest = out_dir / HIP_GLUTE_COMPARE_JSON
    if dest.exists() and not force:
        raise ProportionError(
            f"hip/glute compare exists: {dest} (pass --force)",
            code="hip_glute_compare_failed",
            details={"path": str(dest)},
        )

    rep = load_report(report_path)
    pkg = load_blockout_recipe(recipe_path)
    parts = list(pkg.parts)
    live_parts: list[dict[str, Any]] | None = None
    if scene_dump is not None:
        live_parts = _load_scene_dump(Path(scene_dump))

    messages: list[str] = [
        "HIP_GLUTE_COMPARE_HONESTY — authoring QA only; not mesh or print success",
    ]
    metrics = build_hip_glute_metrics(rep, recipe=pkg)
    short_outer = (
        metrics.glute_outer_vs_hip_soft_outer_m is not None
        and metrics.glute_outer_vs_hip_soft_outer_m >= GLUTE_SHORT_OF_HIP_SOFT_M
    )
    if short_outer:
        messages.append(
            f"glute_short_of_hip_soft_outer={metrics.glute_outer_vs_hip_soft_outer_m:.4f}"
        )
    pelvis_behind = (
        metrics.pelvis_front_vs_glute_front_m is not None
        and metrics.pelvis_front_vs_glute_front_m >= PELVIS_BEHIND_GLUTE_M
    )
    if pelvis_behind:
        messages.append(f"pelvis_front_behind_glute_m={metrics.pelvis_front_vs_glute_front_m:.4f}")
    thigh_wide = metrics.thigh_gap_m is not None and metrics.thigh_gap_m > THIGH_GAP_WIDE_M
    if thigh_wide:
        messages.append(f"thigh_gap_wide_m={metrics.thigh_gap_m:.4f}")

    glute_l = _find_part(parts, "RECIPE_glute_soft_l")
    glute_r = _find_part(parts, "RECIPE_glute_soft_r")
    inner_l = _inner_x(glute_l)
    inner_r = _inner_x(glute_r)
    gap: float | None = None
    if inner_l is not None and inner_r is not None:
        gap = min(inner_l, inner_r)
    elif inner_l is not None:
        gap = inner_l
    elif inner_r is not None:
        gap = inner_r
    midline_gap = gap is not None and gap >= GLUTE_MIDLINE_GAP_M
    if midline_gap:
        messages.append(f"glute_midline_gap_m={gap:.4f}")

    lms = rep.landmarks_xyz
    crotch = lms.get("crotch_pubic")
    crotch_z = _as_float(crotch.z_m) if crotch is not None else None

    roles: list[HipGluteCompareRole] = []
    for lid in HIP_GLUTE_COMPARE_ROLES:
        measured = _measured_coord(lms.get(lid))
        rec = extract_recipe_hip_glute_part(parts, lid)
        live = extract_recipe_hip_glute_part(live_parts, lid) if live_parts is not None else None
        delta = _delta_mm(measured, rec)
        form: list[str] = []
        if measured is None:
            form.append("missing_id")
        if short_outer and lid.startswith("glute_outer_"):
            form.append("glute_short_of_hip_soft_outer")
        if pelvis_behind and lid in (
            "asis_l",
            "asis_r",
            "crotch_pubic",
            "glute_front",
            "pelvis",
        ):
            form.append("pelvis_front_behind_glute")
        if thigh_wide and lid.startswith("thigh_medial_"):
            form.append("thigh_gap_wide")
        if midline_gap and lid in ("glute_cleft", "glute_peak_l", "glute_peak_r"):
            form.append("glute_midline_gap")
        if lid.startswith("groin_fold_") and measured is not None and crotch_z is not None:
            gz = _as_float(measured.z_m)
            if gz is not None and abs(gz - crotch_z) >= GROIN_DISCONNECT_M:
                form.append("groin_disconnected")
        conf = 0.0
        if measured is not None and measured.confidence is not None:
            conf = float(measured.confidence)
        elif rec is not None:
            conf = 0.5
        roles.append(
            HipGluteCompareRole(
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
    package = HipGluteComparePackage(
        ok=True,
        roles=roles,
        messages=messages,
        hip_glute_metrics=metrics,
        package_path=str(dest),
    )
    dest.write_text(package.model_dump_json(indent=2) + "\n", encoding="utf-8")
    sidecar = out_dir / HIP_GLUTE_METRICS_JSON
    if _has_finite_hip_glute(lms):
        sidecar.write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
    elif sidecar.exists():
        sidecar.unlink()
    return package.model_dump(mode="json")
