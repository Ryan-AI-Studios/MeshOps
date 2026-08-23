"""Shoulder girdle landmark compare: photo vs RECIPE vs optional live scene (track 0128).

Authoring QA only — GIRDLE_COMPARE_HONESTY. Not mesh or print success.
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
from meshops.proportion.honesty import GIRDLE_COMPARE_HONESTY
from meshops.proportion.models import LandmarkXYZ, ProportionReport

GIRDLE_COMPARE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
GIRDLE_COMPARE_JSON: Final[str] = "girdle_compare.json"
GIRDLE_METRICS_JSON: Final[str] = "girdle_metrics.json"
GIRDLE_COMPARE_ROLES: Final[tuple[str, ...]] = (
    "trap_apex_l",
    "trap_apex_r",
    "trap_med_l",
    "trap_med_r",
    "trap_lat_l",
    "trap_lat_r",
    "clav_med_l",
    "clav_med_r",
    "clav_lat_l",
    "clav_lat_r",
    "scm_origin_l",
    "scm_origin_r",
    "scm_insert_l",
    "scm_insert_r",
    "acromion_l",
    "acromion_r",
    "nape",
    "shoulder_l",
    "shoulder_r",
    "neck",
)
SUGGESTED_ACTIONS: Final[frozenset[str]] = frozenset(
    {"skip", "hold_priors", "soft_adjust", "session_girdle", "remake_0111"}
)
FORM_READ_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "trap_medial_gap",
        "trap_towers",
        "scm_front_plane",
        "missing_id",
    }
)
TRAP_MEDIAL_GAP_M: Final[float] = 0.015
TRAP_TOWERS_RZ_M: Final[float] = 0.060

_ROLE_PART: dict[str, str] = {
    "trap_apex_l": "RECIPE_trap_soft_l",
    "trap_apex_r": "RECIPE_trap_soft_r",
    "trap_med_l": "RECIPE_trap_soft_l",
    "trap_med_r": "RECIPE_trap_soft_r",
    "trap_lat_l": "RECIPE_trap_soft_l",
    "trap_lat_r": "RECIPE_trap_soft_r",
    "clav_med_l": "RECIPE_clavicle_l",
    "clav_med_r": "RECIPE_clavicle_r",
    "clav_lat_l": "RECIPE_clavicle_l",
    "clav_lat_r": "RECIPE_clavicle_r",
    "scm_origin_l": "RECIPE_sternomastoid_soft_l",
    "scm_origin_r": "RECIPE_sternomastoid_soft_r",
    "scm_insert_l": "RECIPE_sternomastoid_soft_l",
    "scm_insert_r": "RECIPE_sternomastoid_soft_r",
    "acromion_l": "RECIPE_deltoid_soft_l",
    "acromion_r": "RECIPE_deltoid_soft_r",
    "nape": "RECIPE_neck",
    "shoulder_l": "RECIPE_deltoid_soft_l",
    "shoulder_r": "RECIPE_deltoid_soft_r",
    "neck": "RECIPE_neck",
}

# B4: soft_adjust only for new-id trap_apex Y and Z.
# clav/scm/acromion/nape/trap_med/trap_lat/HAVE ids stay hold_priors.
_SOFT_ADJUST_IDS: Final[frozenset[str]] = frozenset({"trap_apex_l", "trap_apex_r"})

# B39: role-mapped capsule/cylinder endpoints (not 0127 always-p0).
_ENDPOINT: dict[str, Literal["p0", "p1"]] = {
    "clav_lat_l": "p0",
    "clav_lat_r": "p0",
    "clav_med_l": "p1",
    "clav_med_r": "p1",
    "scm_origin_l": "p0",
    "scm_origin_r": "p0",
    "scm_insert_l": "p1",
    "scm_insert_r": "p1",
    "nape": "p0",
}

_KNOB: dict[str, str] = {
    "trap_apex_l": "TRAP_Y_BACK_FRAC_RY",
    "trap_apex_r": "TRAP_Y_BACK_FRAC_RY",
    "trap_med_l": "TRAP_LAT_FRAC",
    "trap_med_r": "TRAP_LAT_FRAC",
    "trap_lat_l": "TRAP_LAT_FRAC",
    "trap_lat_r": "TRAP_LAT_FRAC",
    "clav_med_l": "CLAVICLE_RADIUS_FRAC_H",
    "clav_med_r": "CLAVICLE_RADIUS_FRAC_H",
    "clav_lat_l": "CLAVICLE_RADIUS_FRAC_H",
    "clav_lat_r": "CLAVICLE_RADIUS_FRAC_H",
    "scm_origin_l": "SCM_R_FRAC_NECK_R",
    "scm_origin_r": "SCM_R_FRAC_NECK_R",
    "scm_insert_l": "SCM_R_FRAC_NECK_R",
    "scm_insert_r": "SCM_R_FRAC_NECK_R",
    "acromion_l": "DELT_RY_FRAC",
    "acromion_r": "DELT_RY_FRAC",
    "nape": "NECK_NAPE_SETBACK_M",
    "shoulder_l": "DELT_RY_FRAC",
    "shoulder_r": "DELT_RY_FRAC",
    "neck": "NECK_R_MAX_FRAC_HEAD_RX",
}

_KNOWN_DUMP_NAMES: Final[frozenset[str]] = frozenset(
    {
        "RECIPE_trap_soft_l",
        "RECIPE_trap_soft_r",
        "RECIPE_clavicle_l",
        "RECIPE_clavicle_r",
        "RECIPE_sternomastoid_soft_l",
        "RECIPE_sternomastoid_soft_r",
        "RECIPE_neck",
        "RECIPE_neck_base_soft",
        "RECIPE_deltoid_soft_l",
        "RECIPE_deltoid_soft_r",
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


def _mapped_endpoint(part: dict[str, Any], role_id: str) -> list[float] | None:
    """B39: clav/SCM/nape use role-mapped p0/p1 — never 0127 always-p0."""
    which = _ENDPOINT.get(role_id)
    if which == "p0":
        return _as_vec3(part.get("p0"))
    if which == "p1":
        return _as_vec3(part.get("p1"))
    return None


def extract_recipe_girdle_part(parts: list[Any], role_id: str) -> dict[str, Any] | None:
    """Extract one girdle role from RECIPE parts (B26/B39 mapped capsule endpoints)."""
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
    mapped = _mapped_endpoint(match, role_id)
    center = mapped
    if center is None:
        center = _as_vec3(match.get("center"))
    if center is None:
        center = _midpoint(match.get("p0"), match.get("p1"))
    if kind in ("capsule", "cylinder") and center is None:
        return None
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
        "placement": match.get("placement"),
    }


class GirdleMetrics(BaseModel):
    """Sidecar girdle meters (not a ProportionReport field — stay 1.2.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    honesty: str = GIRDLE_COMPARE_HONESTY
    trap_medial_gap_m: float | None = None
    trap_ry_m: float | None = None
    trap_rz_m: float | None = None
    clav_r_m: float | None = None
    scm_r_m: float | None = None
    neck_r_m: float | None = None
    neck_base_rx_m: float | None = None
    neck_base_ry_m: float | None = None
    neck_base_rz_m: float | None = None
    nape_y_m: float | None = None
    y_m: dict[str, float | None] = Field(default_factory=dict)


class GirdleCompareCoord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    confidence: float | None = None
    sources: list[str] = Field(default_factory=list)


class GirdleCompareDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float | None = None
    y: float | None = None
    z: float | None = None


class GirdleCompareRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    measured: GirdleCompareCoord | None = None
    recipe: dict[str, Any] | None = None
    live: dict[str, Any] | None = None
    delta_mm: GirdleCompareDelta | None = None
    knob: str | None = None
    form_read: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    suggested: Literal["skip", "hold_priors", "soft_adjust", "session_girdle", "remake_0111"] = (
        "skip"
    )


class GirdleComparePackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = GIRDLE_COMPARE_SCHEMA_VERSION
    honesty: str = GIRDLE_COMPARE_HONESTY
    region: Literal["girdle"] = "girdle"
    ok: bool = False
    roles: list[GirdleCompareRole] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    girdle_metrics: GirdleMetrics | None = None
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


def _trap_surface_gap(trap: dict[str, Any] | None) -> float | None:
    if trap is None:
        return None
    center = _as_vec3(trap.get("center"))
    rx = _as_float(trap.get("rx_m"))
    if center is None or rx is None:
        return None
    return abs(center[0]) - rx


def build_girdle_metrics(
    report: ProportionReport,
    recipe: Any | None = None,
) -> GirdleMetrics:
    """Compute girdle_metrics from landmarks_xyz (+ optional RECIPE parts)."""
    parts = _recipe_parts(recipe)
    trap_l = _find_part(parts, "RECIPE_trap_soft_l")
    trap_r = _find_part(parts, "RECIPE_trap_soft_r")
    neck = _find_part(parts, "RECIPE_neck")
    clav = _find_part(parts, "RECIPE_clavicle_l") or _find_part(parts, "RECIPE_clavicle_r")
    scm = _find_part(parts, "RECIPE_sternomastoid_soft_l") or _find_part(
        parts, "RECIPE_sternomastoid_soft_r"
    )
    neck_base = _find_part(parts, "RECIPE_neck_base_soft")
    trap = trap_l or trap_r
    neck_r = _as_float(neck.get("radius_m")) if neck is not None else None
    gaps: list[float] = []
    for candidate in (trap_l, trap_r):
        gap = _trap_surface_gap(candidate)
        if gap is not None:
            gaps.append(gap)
    trap_medial_gap: float | None = None
    if gaps and neck_r is not None:
        trap_medial_gap = min(gaps) - neck_r

    nape_y: float | None = None
    if neck is not None:
        p0 = _as_vec3(neck.get("p0"))
        if p0 is not None:
            nape_y = p0[1]

    lms = report.landmarks_xyz
    y_fields: dict[str, float | None] = {}
    for lid in GIRDLE_COMPARE_ROLES:
        lm = lms.get(lid)
        y_fields[lid] = _as_float(lm.y_m) if lm is not None else None

    return GirdleMetrics(
        trap_medial_gap_m=trap_medial_gap,
        trap_ry_m=_as_float(trap.get("ry_m")) if trap is not None else None,
        trap_rz_m=_as_float(trap.get("rz_m")) if trap is not None else None,
        clav_r_m=_as_float(clav.get("radius_m")) if clav is not None else None,
        scm_r_m=_as_float(scm.get("radius_m")) if scm is not None else None,
        neck_r_m=neck_r,
        neck_base_rx_m=_as_float(neck_base.get("rx_m")) if neck_base is not None else None,
        neck_base_ry_m=_as_float(neck_base.get("ry_m")) if neck_base is not None else None,
        neck_base_rz_m=_as_float(neck_base.get("rz_m")) if neck_base is not None else None,
        nape_y_m=nape_y,
        y_m=y_fields,
    )


def _measured_coord(lm: LandmarkXYZ | None) -> GirdleCompareCoord | None:
    if lm is None:
        return None
    x_m = _as_float(lm.x_m)
    y_m = _as_float(lm.y_m)
    z_m = _as_float(lm.z_m)
    if x_m is None and y_m is None and z_m is None:
        return None
    return GirdleCompareCoord(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        confidence=float(lm.confidence),
        sources=list(lm.sources),
    )


def _anchor_xyz(recipe: dict[str, Any] | None, role_id: str) -> list[float] | None:
    if recipe is None:
        return None
    mapped = _mapped_endpoint(recipe, role_id)
    if mapped is not None:
        return mapped
    center = _as_vec3(recipe.get("center"))
    if center is not None:
        return center
    return _midpoint(recipe.get("p0"), recipe.get("p1"))


def _delta_mm(
    measured: GirdleCompareCoord | None,
    recipe: dict[str, Any] | None,
    role_id: str,
) -> GirdleCompareDelta | None:
    if measured is None or recipe is None:
        return None
    # B39: capsule/cylinder roles use mapped p0/p1 (live clav/SCM center=null).
    anchor = _anchor_xyz(recipe, role_id)
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
    return GirdleCompareDelta(x=dx, y=dy, z=dz)


def _suggest(
    measured: GirdleCompareCoord | None,
    recipe: dict[str, Any] | None,
    delta: GirdleCompareDelta | None,
    role_id: str,
) -> Literal["skip", "hold_priors", "soft_adjust", "session_girdle", "remake_0111"]:
    if measured is None:
        return "skip"
    if recipe is None or _anchor_xyz(recipe, role_id) is None:
        return "skip"
    if role_id not in _SOFT_ADJUST_IDS:
        return "hold_priors"
    if delta is None:
        return "hold_priors"
    # B36/B40: Y/Z consume — ignore delta.x; fire on max(|dy|, |dz|).
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
            code="girdle_compare_failed",
            details={"path": str(path)},
        ) from exc
    if not isinstance(raw, dict) or "parts" not in raw:
        raise ProportionError(
            f"scene dump must be an object with parts: {path}",
            code="girdle_compare_failed",
            details={"path": str(path)},
        )
    parts = raw.get("parts")
    if not isinstance(parts, list):
        raise ProportionError(
            f"scene dump parts must be a list: {path}",
            code="girdle_compare_failed",
            details={"path": str(path)},
        )
    out: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        if not isinstance(part, dict) or "name" not in part:
            raise ProportionError(
                f"scene dump part {i} must be an object with name",
                code="girdle_compare_failed",
                details={"path": str(path), "index": i},
            )
        name = str(part.get("name") or "")
        if name in _KNOWN_DUMP_NAMES:
            has_center = _as_vec3(part.get("center")) is not None
            has_caps = _as_vec3(part.get("p0")) is not None and _as_vec3(part.get("p1")) is not None
            if not has_center and not has_caps:
                raise ProportionError(
                    f"scene dump part {name!r} missing center or p0/p1",
                    code="girdle_compare_failed",
                    details={"path": str(path), "index": i, "name": name},
                )
        out.append(part)
    return out


def _has_finite_girdle(lms: dict[str, LandmarkXYZ]) -> bool:
    for lid in GIRDLE_COMPARE_ROLES:
        lm = lms.get(lid)
        if lm is None:
            continue
        if any(_as_float(v) is not None for v in (lm.x_m, lm.y_m, lm.z_m)):
            return True
    return False


def _scm_front_plane(parts: list[Any]) -> bool:
    for name in ("RECIPE_sternomastoid_soft_l", "RECIPE_sternomastoid_soft_r"):
        part = _find_part(parts, name)
        if part is None:
            continue
        placement = str(part.get("placement") or "full3d")
        if placement != "full3d":
            return True
    return False


def run_blockout_girdle_compare(
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
    dest = out_dir / GIRDLE_COMPARE_JSON
    if dest.exists() and not force:
        raise ProportionError(
            f"girdle compare exists: {dest} (pass --force)",
            code="girdle_compare_failed",
            details={"path": str(dest)},
        )

    rep = load_report(report_path)
    pkg = load_blockout_recipe(recipe_path)
    parts = list(pkg.parts)
    live_parts: list[dict[str, Any]] | None = None
    if scene_dump is not None:
        live_parts = _load_scene_dump(Path(scene_dump))

    messages: list[str] = [
        "GIRDLE_COMPARE_HONESTY — authoring QA only; not mesh or print success",
    ]
    metrics = build_girdle_metrics(rep, recipe=pkg)
    gap_flag = (
        metrics.trap_medial_gap_m is not None and metrics.trap_medial_gap_m >= TRAP_MEDIAL_GAP_M
    )
    if gap_flag:
        messages.append(f"trap_medial_gap_m={metrics.trap_medial_gap_m:.4f}")
    towers = metrics.trap_rz_m is not None and metrics.trap_rz_m >= TRAP_TOWERS_RZ_M
    if towers:
        messages.append(f"trap_towers_rz_m={metrics.trap_rz_m:.4f}")
    scm_front = _scm_front_plane(parts)
    if scm_front:
        messages.append("scm_front_plane")

    lms = rep.landmarks_xyz
    roles: list[GirdleCompareRole] = []
    for lid in GIRDLE_COMPARE_ROLES:
        measured = _measured_coord(lms.get(lid))
        rec = extract_recipe_girdle_part(parts, lid)
        live = extract_recipe_girdle_part(live_parts, lid) if live_parts is not None else None
        delta = _delta_mm(measured, rec, lid)
        form: list[str] = []
        if measured is None:
            form.append("missing_id")
        if gap_flag and lid.startswith("trap_"):
            form.append("trap_medial_gap")
        if towers and lid.startswith("trap_"):
            form.append("trap_towers")
        if scm_front and lid.startswith("scm_"):
            form.append("scm_front_plane")
        conf = 0.0
        if measured is not None and measured.confidence is not None:
            conf = float(measured.confidence)
        elif rec is not None:
            conf = 0.5
        roles.append(
            GirdleCompareRole(
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
    package = GirdleComparePackage(
        ok=True,
        roles=roles,
        messages=messages,
        girdle_metrics=metrics,
        package_path=str(dest),
    )
    dest.write_text(package.model_dump_json(indent=2) + "\n", encoding="utf-8")
    sidecar = out_dir / GIRDLE_METRICS_JSON
    if _has_finite_girdle(lms):
        sidecar.write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
    elif sidecar.exists():
        sidecar.unlink()
    return package.model_dump(mode="json")
