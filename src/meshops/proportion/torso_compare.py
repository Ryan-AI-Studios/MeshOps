"""Torso landmark compare: photo vs RECIPE vs optional live scene (track 0125).

Authoring QA only — TORSO_COMPARE_HONESTY. Not mesh or print success.
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
from meshops.proportion.honesty import TORSO_COMPARE_HONESTY
from meshops.proportion.models import LandmarkXYZ, ProportionReport

TORSO_COMPARE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
TORSO_COMPARE_JSON: Final[str] = "torso_compare.json"
TORSO_METRICS_JSON: Final[str] = "torso_metrics.json"
TORSO_COMPARE_ROLES: Final[tuple[str, ...]] = (
    "sternum_mid",
    "nipple_bust",
    "underbust",
    "navel",
    "costal_l",
    "costal_r",
    "scap_inferior_l",
    "scap_inferior_r",
    "scap_medial_l",
    "scap_medial_r",
    "mid_back_l",
    "mid_back_r",
    "chest_front",
    "chest_back",
    "breast_front",
    "breast_back",
)
SUGGESTED_ACTIONS: Final[frozenset[str]] = frozenset(
    {"skip", "hold_priors", "soft_adjust", "session_torso", "remake_0111"}
)
FORM_READ_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "three_tire",
        "saucer_scap",
        "saucer_mid_back",
        "breast_disconnected",
        "chest_clav_pride",
        "missing_id",
    }
)
OVERLAP_FLOOR_M: Final[float] = 0.080
SAUCER_SCAP_RY_RX: Final[float] = 0.30
SAUCER_MID_BACK_RY_RX: Final[float] = 0.26
CLAV_PRIDE_M: Final[float] = 0.008

_ROLE_PART: dict[str, str] = {
    "sternum_mid": "RECIPE_torso_oval_chest",
    "nipple_bust": "RECIPE_breast_soft_l",
    "underbust": "RECIPE_torso_oval_chest",
    "navel": "RECIPE_torso_oval_waist",
    "costal_l": "RECIPE_torso_oval_chest",
    "costal_r": "RECIPE_torso_oval_chest",
    "scap_inferior_l": "RECIPE_scap_soft_l",
    "scap_inferior_r": "RECIPE_scap_soft_r",
    "scap_medial_l": "RECIPE_scap_soft_l",
    "scap_medial_r": "RECIPE_scap_soft_r",
    "mid_back_l": "RECIPE_mid_back_soft_l",
    "mid_back_r": "RECIPE_mid_back_soft_r",
    "chest_front": "RECIPE_torso_oval_chest",
    "chest_back": "RECIPE_torso_oval_chest",
    "breast_front": "RECIPE_breast_soft_l",
    "breast_back": "RECIPE_breast_soft_l",
    "clavicle_l": "RECIPE_clavicle_l",
    "clavicle_r": "RECIPE_clavicle_r",
}

# B4: soft_adjust only for new-id placement consume (scap/mid_back Y/Z).
# Existing DEPTH_PAIRS / oval-const knobs stay hold_priors (no 0105/0090/0118 retune).
_SOFT_ADJUST_IDS: Final[frozenset[str]] = frozenset(
    {
        "scap_inferior_l",
        "scap_inferior_r",
        "mid_back_l",
        "mid_back_r",
    }
)

_KNOB: dict[str, str] = {
    "sternum_mid": "TORSO_OVAL_Z_NORM_CHEST",
    "nipple_bust": "TORSO_OVAL_Z_NORM_CHEST",
    "underbust": "TORSO_OVAL_Z_NORM_CHEST",
    "navel": "TORSO_OVAL_Z_NORM_WAIST",
    "costal_l": "TORSO_OVAL_RY_CHEST_FRAC",
    "costal_r": "TORSO_OVAL_RY_CHEST_FRAC",
    "scap_inferior_l": "SCAP_REAR_PAST_M",
    "scap_inferior_r": "SCAP_REAR_PAST_M",
    "scap_medial_l": "SCAP_LAT_FRAC",
    "scap_medial_r": "SCAP_LAT_FRAC",
    "mid_back_l": "MID_BACK_REAR_PAST_M",
    "mid_back_r": "MID_BACK_REAR_PAST_M",
    "chest_front": "TORSO_OVAL_RY_CHEST_FRAC",
    "chest_back": "TORSO_OVAL_RY_CHEST_FRAC",
    "breast_front": "BREAST_SIT_CHEST_BURY_M",
    "breast_back": "BREAST_SIT_CHEST_BURY_M",
}

_KNOWN_DUMP_NAMES: Final[frozenset[str]] = frozenset(
    {
        "RECIPE_torso_oval_chest",
        "RECIPE_torso_oval_waist",
        "RECIPE_torso_oval_hip",
        "RECIPE_scap_soft_l",
        "RECIPE_scap_soft_r",
        "RECIPE_mid_back_soft_l",
        "RECIPE_mid_back_soft_r",
        "RECIPE_breast_soft_l",
        "RECIPE_breast_soft_r",
        "RECIPE_pelvis_oval",
        "RECIPE_clavicle_l",
        "RECIPE_clavicle_r",
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


def _center_from_part(part: dict[str, Any], *, prefer_p1: bool = False) -> list[float] | None:
    if prefer_p1:
        p1 = _as_vec3(part.get("p1"))
        if p1 is not None:
            return p1
    center = _as_vec3(part.get("center"))
    if center is not None:
        return center
    return _midpoint(part.get("p0"), part.get("p1"))


def extract_recipe_torso_part(parts: list[Any], role_id: str) -> dict[str, Any] | None:
    """Extract one torso role from RECIPE parts (B26 clavicle capsule p1)."""
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
    prefer_p1 = role_id.startswith("clavicle")
    center = _center_from_part(match, prefer_p1=prefer_p1)
    if kind == "capsule" and center is None:
        return None
    ry = _as_float(match.get("ry_m"))
    if role_id in ("chest_front", "breast_front") and center is not None and ry is not None:
        center = [center[0], center[1] - ry, center[2]]
    if role_id in ("chest_back", "breast_back") and center is not None and ry is not None:
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


class TorsoMetrics(BaseModel):
    """Sidecar torso meters (not a ProportionReport field — stay 1.2.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    honesty: str = TORSO_COMPARE_HONESTY
    oval_overlap_chest_waist_m: float | None = None
    oval_overlap_waist_hip_m: float | None = None
    chest_front_y_m: float | None = None
    chest_rear_y_m: float | None = None
    breast_rear_vs_chest_front_m: float | None = None
    scap_rear_past_m: float | None = None
    scap_ry_over_rx: float | None = None
    mid_back_rear_past_m: float | None = None
    chest_clav_pride_m: float | None = None
    y_m: dict[str, float | None] = Field(default_factory=dict)


class TorsoCompareCoord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    confidence: float | None = None
    sources: list[str] = Field(default_factory=list)


class TorsoCompareDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float | None = None
    y: float | None = None
    z: float | None = None


class TorsoCompareRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    measured: TorsoCompareCoord | None = None
    recipe: dict[str, Any] | None = None
    live: dict[str, Any] | None = None
    delta_mm: TorsoCompareDelta | None = None
    knob: str | None = None
    form_read: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    suggested: Literal["skip", "hold_priors", "soft_adjust", "session_torso", "remake_0111"] = (
        "skip"
    )


class TorsoComparePackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = TORSO_COMPARE_SCHEMA_VERSION
    honesty: str = TORSO_COMPARE_HONESTY
    region: Literal["torso"] = "torso"
    ok: bool = False
    roles: list[TorsoCompareRole] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    torso_metrics: TorsoMetrics | None = None
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


def _z_overlap(upper: dict[str, Any] | None, lower: dict[str, Any] | None) -> float | None:
    if upper is None or lower is None:
        return None
    uc = _as_vec3(upper.get("center"))
    lc = _as_vec3(lower.get("center"))
    urz = _as_float(upper.get("rz_m"))
    lrz = _as_float(lower.get("rz_m"))
    if uc is None or lc is None or urz is None or lrz is None:
        return None
    upper_bottom = uc[2] - urz
    lower_top = lc[2] + lrz
    return lower_top - upper_bottom


def _ry_over_rx(part: dict[str, Any] | None) -> float | None:
    if part is None:
        return None
    rx = _as_float(part.get("rx_m"))
    ry = _as_float(part.get("ry_m"))
    if rx is None or ry is None or rx <= 0:
        return None
    return ry / rx


def _clav_y(parts: list[Any]) -> float | None:
    clav = extract_recipe_torso_part(parts, "clavicle_l")
    if clav is None:
        clav = extract_recipe_torso_part(parts, "clavicle_r")
    if clav is None:
        return None
    center = _as_vec3(clav.get("center"))
    if center is None:
        return None
    return center[1]


def build_torso_metrics(
    report: ProportionReport,
    recipe: Any | None = None,
) -> TorsoMetrics:
    """Compute torso_metrics from landmarks_xyz (+ optional RECIPE ovals)."""
    parts = _recipe_parts(recipe)
    chest = _find_part(parts, "RECIPE_torso_oval_chest")
    waist = _find_part(parts, "RECIPE_torso_oval_waist")
    hip = _find_part(parts, "RECIPE_torso_oval_hip")
    scap = _find_part(parts, "RECIPE_scap_soft_l") or _find_part(parts, "RECIPE_scap_soft_r")
    mid = _find_part(parts, "RECIPE_mid_back_soft_l") or _find_part(parts, "RECIPE_mid_back_soft_r")
    breast = _find_part(parts, "RECIPE_breast_soft_l") or _find_part(parts, "RECIPE_breast_soft_r")

    chest_front = _surface_y(chest, rear=False)
    chest_rear = _surface_y(chest, rear=True)
    breast_rear = _surface_y(breast, rear=True)

    lms = report.landmarks_xyz
    y_fields: dict[str, float | None] = {}
    for lid in (
        "sternum_mid",
        "costal_l",
        "costal_r",
        "scap_inferior_l",
        "scap_inferior_r",
        "scap_medial_l",
        "scap_medial_r",
        "mid_back_l",
        "mid_back_r",
        "chest_front",
        "chest_back",
        "breast_front",
        "breast_back",
    ):
        lm = lms.get(lid)
        y_fields[lid] = _as_float(lm.y_m) if lm is not None else None

    # B1/B21: overlay measured surface Ys before bury / pride / past (never `or`).
    if y_fields.get("chest_front") is not None:
        chest_front = y_fields["chest_front"]
    if y_fields.get("chest_back") is not None:
        chest_rear = y_fields["chest_back"]
    if y_fields.get("breast_back") is not None:
        breast_rear = y_fields["breast_back"]

    bury: float | None = None
    if breast_rear is not None and chest_front is not None:
        bury = breast_rear - chest_front

    clav_y = _clav_y(parts)
    pride: float | None = None
    if chest_front is not None and clav_y is not None:
        pride = chest_front - clav_y

    scap_rear = _surface_y(scap, rear=True)
    scap_past: float | None = (
        scap_rear - chest_rear if scap_rear is not None and chest_rear is not None else None
    )

    # B5/B6/B21: measured scap Y is center-class; rear = y + ry when ry finite.
    measured_scap_y = y_fields.get("scap_inferior_l")
    if measured_scap_y is None:
        measured_scap_y = y_fields.get("scap_inferior_r")
    ry_scap = _as_float(scap.get("ry_m")) if scap is not None else None
    if measured_scap_y is not None and chest_rear is not None and ry_scap is not None:
        scap_past = (measured_scap_y + ry_scap) - chest_rear

    waist_rear = _surface_y(waist, rear=True)
    mid_rear = _surface_y(mid, rear=True)
    mid_past: float | None = (
        mid_rear - waist_rear if mid_rear is not None and waist_rear is not None else None
    )

    measured_mid_y = y_fields.get("mid_back_l")
    if measured_mid_y is None:
        measured_mid_y = y_fields.get("mid_back_r")
    ry_mid = _as_float(mid.get("ry_m")) if mid is not None else None
    if measured_mid_y is not None and waist_rear is not None and ry_mid is not None:
        mid_past = (measured_mid_y + ry_mid) - waist_rear

    return TorsoMetrics(
        oval_overlap_chest_waist_m=_z_overlap(chest, waist),
        oval_overlap_waist_hip_m=_z_overlap(waist, hip),
        chest_front_y_m=chest_front,
        chest_rear_y_m=chest_rear,
        breast_rear_vs_chest_front_m=bury,
        scap_rear_past_m=scap_past,
        scap_ry_over_rx=_ry_over_rx(scap),
        mid_back_rear_past_m=mid_past,
        chest_clav_pride_m=pride,
        y_m=y_fields,
    )


def _measured_coord(lm: LandmarkXYZ | None) -> TorsoCompareCoord | None:
    if lm is None:
        return None
    x_m = _as_float(lm.x_m)
    y_m = _as_float(lm.y_m)
    z_m = _as_float(lm.z_m)
    if x_m is None and y_m is None and z_m is None:
        return None
    return TorsoCompareCoord(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        confidence=float(lm.confidence),
        sources=list(lm.sources),
    )


def _delta_mm(
    measured: TorsoCompareCoord | None,
    recipe: dict[str, Any] | None,
) -> TorsoCompareDelta | None:
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
    return TorsoCompareDelta(x=dx, y=dy, z=dz)


def _suggest(
    measured: TorsoCompareCoord | None,
    recipe: dict[str, Any] | None,
    delta: TorsoCompareDelta | None,
    role_id: str,
) -> Literal["skip", "hold_priors", "soft_adjust", "session_torso", "remake_0111"]:
    if measured is None:
        return "skip"
    if recipe is None or recipe.get("center") is None:
        return "skip"
    if role_id not in _SOFT_ADJUST_IDS:
        return "hold_priors"
    if delta is None:
        return "hold_priors"
    vals = [abs(v) for v in (delta.x, delta.y, delta.z) if v is not None]
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
            code="torso_compare_failed",
            details={"path": str(path)},
        ) from exc
    if not isinstance(raw, dict) or "parts" not in raw:
        raise ProportionError(
            f"scene dump must be an object with parts: {path}",
            code="torso_compare_failed",
            details={"path": str(path)},
        )
    parts = raw.get("parts")
    if not isinstance(parts, list):
        raise ProportionError(
            f"scene dump parts must be a list: {path}",
            code="torso_compare_failed",
            details={"path": str(path)},
        )
    out: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        if not isinstance(part, dict) or "name" not in part:
            raise ProportionError(
                f"scene dump part {i} must be an object with name",
                code="torso_compare_failed",
                details={"path": str(path), "index": i},
            )
        name = str(part.get("name") or "")
        if name in _KNOWN_DUMP_NAMES:
            has_center = _as_vec3(part.get("center")) is not None
            has_caps = _as_vec3(part.get("p0")) is not None and _as_vec3(part.get("p1")) is not None
            if not has_center and not has_caps:
                raise ProportionError(
                    f"scene dump part {name!r} missing center or p0/p1",
                    code="torso_compare_failed",
                    details={"path": str(path), "index": i, "name": name},
                )
        out.append(part)
    return out


def _has_finite_torso(lms: dict[str, LandmarkXYZ]) -> bool:
    for lid in TORSO_COMPARE_ROLES:
        lm = lms.get(lid)
        if lm is None:
            continue
        if any(_as_float(v) is not None for v in (lm.x_m, lm.y_m, lm.z_m)):
            return True
    return False


def run_blockout_torso_compare(
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
    dest = out_dir / TORSO_COMPARE_JSON
    if dest.exists() and not force:
        raise ProportionError(
            f"torso compare exists: {dest} (pass --force)",
            code="torso_compare_failed",
            details={"path": str(dest)},
        )

    rep = load_report(report_path)
    pkg = load_blockout_recipe(recipe_path)
    parts = list(pkg.parts)
    live_parts: list[dict[str, Any]] | None = None
    if scene_dump is not None:
        live_parts = _load_scene_dump(Path(scene_dump))

    messages: list[str] = [
        "TORSO_COMPARE_HONESTY — authoring QA only; not mesh or print success",
    ]
    metrics = build_torso_metrics(rep, recipe=pkg)
    ov_cw = metrics.oval_overlap_chest_waist_m
    ov_wh = metrics.oval_overlap_waist_hip_m
    three_tire = False
    if ov_cw is not None and ov_wh is not None and min(ov_cw, ov_wh) < OVERLAP_FLOOR_M:
        three_tire = True
        messages.append(
            f"three_tire overlap_cw={ov_cw:.4f} overlap_wh={ov_wh:.4f} < {OVERLAP_FLOOR_M}"
        )
    bury = metrics.breast_rear_vs_chest_front_m
    breast_gap = bury is not None and bury < 0.0
    if breast_gap:
        messages.append(f"breast_disconnected rear_vs_chest_front_m={bury:.4f}")
    saucer_scap = (
        metrics.scap_ry_over_rx is not None and metrics.scap_ry_over_rx < SAUCER_SCAP_RY_RX
    )
    mid_ratio = None
    mid_part = _find_part(parts, "RECIPE_mid_back_soft_l") or _find_part(
        parts, "RECIPE_mid_back_soft_r"
    )
    mid_ratio = _ry_over_rx(mid_part)
    saucer_mid = mid_ratio is not None and mid_ratio < SAUCER_MID_BACK_RY_RX
    pride = metrics.chest_clav_pride_m
    clav_pride = pride is not None and abs(pride) >= CLAV_PRIDE_M
    if clav_pride:
        messages.append(f"chest_clav_pride_m={pride:.4f}")

    roles: list[TorsoCompareRole] = []
    lms = rep.landmarks_xyz
    for lid in TORSO_COMPARE_ROLES:
        measured = _measured_coord(lms.get(lid))
        rec = extract_recipe_torso_part(parts, lid)
        live = extract_recipe_torso_part(live_parts, lid) if live_parts is not None else None
        delta = _delta_mm(measured, rec)
        form: list[str] = []
        if measured is None:
            form.append("missing_id")
        if three_tire and lid in (
            "sternum_mid",
            "navel",
            "chest_front",
            "chest_back",
        ):
            form.append("three_tire")
        if saucer_scap and lid.startswith("scap_"):
            form.append("saucer_scap")
        if saucer_mid and lid.startswith("mid_back_"):
            form.append("saucer_mid_back")
        if breast_gap and lid in ("breast_front", "breast_back"):
            form.append("breast_disconnected")
        if clav_pride and lid in ("sternum_mid", "chest_front"):
            form.append("chest_clav_pride")
        conf = 0.0
        if measured is not None and measured.confidence is not None:
            conf = float(measured.confidence)
        elif rec is not None:
            conf = 0.5
        roles.append(
            TorsoCompareRole(
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
    package = TorsoComparePackage(
        ok=True,
        roles=roles,
        messages=messages,
        torso_metrics=metrics,
        package_path=str(dest),
    )
    dest.write_text(package.model_dump_json(indent=2) + "\n", encoding="utf-8")
    sidecar = out_dir / TORSO_METRICS_JSON
    if _has_finite_torso(lms):
        sidecar.write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
    elif sidecar.exists():
        sidecar.unlink()
    return package.model_dump(mode="json")
