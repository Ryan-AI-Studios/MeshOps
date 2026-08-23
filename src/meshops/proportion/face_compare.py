"""Face landmark compare: photo vs RECIPE vs optional live scene (track 0124).

Authoring QA only — FACE_COMPARE_HONESTY. Not mesh or print success.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.analyze import load_report
from meshops.proportion.assist import FACE_FRONT_LANDMARK_IDS
from meshops.proportion.blockout_recipe import load_blockout_recipe
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import FACE_COMPARE_HONESTY
from meshops.proportion.models import LandmarkXYZ, ProportionReport

FACE_COMPARE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
FACE_COMPARE_JSON: Final[str] = "face_compare.json"
FACE_METRICS_JSON: Final[str] = "face_metrics.json"
FACE_COMPARE_ROLES: Final[tuple[str, ...]] = FACE_FRONT_LANDMARK_IDS
SUGGESTED_ACTIONS: Final[frozenset[str]] = frozenset(
    {"skip", "hold_priors", "soft_adjust", "session_0113", "remake_0111"}
)
FORM_READ_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "eyes_float",
        "eyes_wide_set",
        "nose_speck",
        "lip_buried",
        "midline_blob",
        "ear_cup_handle",
        "hat_hair_mass",
        "brow_posts",
        "missing_id",
    }
)
WIDE_SET_HALF_IPD_RX: Final[float] = 0.45
_LOOMIS_EYE_Z: Final[float] = 0.50
_LOOMIS_BROW_Z: Final[float] = 0.67
_LOOMIS_NOSE_Z: Final[float] = 0.33
_LOOMIS_LIP_Z: Final[float] = 0.28  # 0102, not a Loomis third

_ROLE_PART: dict[str, str] = {
    "eye_l": "RECIPE_eye_soft_l",
    "eye_r": "RECIPE_eye_soft_r",
    "brow_l": "RECIPE_brow_soft_l",
    "brow_r": "RECIPE_brow_soft_r",
    "nose_tip": "RECIPE_nose_soft",
    "lip_mid": "RECIPE_lip_soft",
    "mouth_corner_l": "RECIPE_lip_soft",
    "mouth_corner_r": "RECIPE_lip_soft",
    "cheek_l": "RECIPE_cheek_soft_l",
    "cheek_r": "RECIPE_cheek_soft_r",
    "ear_l": "RECIPE_ear_soft_l",
    "ear_r": "RECIPE_ear_soft_r",
}

_KNOB: dict[str, str] = {
    "eye_l": "eye_half_sep",
    "eye_r": "eye_half_sep",
    "brow_l": "_BROW_Z_FRAC",
    "brow_r": "_BROW_Z_FRAC",
    "nose_tip": "_NOSE_BASE_Z_FRAC",
    "lip_mid": "_LIP_Z_FRAC",
    "mouth_corner_l": "mouth_width",
    "mouth_corner_r": "mouth_width",
    "cheek_l": "CHEEK_X_FRAC_HEAD_RX",
    "cheek_r": "CHEEK_X_FRAC_HEAD_RX",
    "ear_l": "ear_cx_head_rx",
    "ear_r": "ear_cx_head_rx",
}


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


def _center_from_part(part: dict[str, Any]) -> list[float] | None:
    center = _as_vec3(part.get("center"))
    if center is not None:
        return center
    return _midpoint(part.get("p0"), part.get("p1"))


def extract_recipe_face_part(parts: list[Any], role_id: str) -> dict[str, Any] | None:
    """Extract one face role from RECIPE parts (B26 capsule midpoint)."""
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
    center = _center_from_part(match)
    if kind == "capsule" and center is None:
        return None
    if role_id in ("mouth_corner_l", "mouth_corner_r") and center is not None:
        rx = _as_float(match.get("rx_m"))
        if rx is not None:
            sx = -1.0 if role_id.endswith("_l") else 1.0
            center = [center[0] + sx * rx, center[1], center[2]]
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


class FaceMetrics(BaseModel):
    """Sidecar face meters (not a ProportionReport field — stay 1.2.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    honesty: str = FACE_COMPARE_HONESTY
    ipd_m: float | None = None
    half_ipd_over_head_rx: float | None = None
    eye_z_frac_h: float | None = None
    brow_z_frac_h: float | None = None
    nose_z_frac_h: float | None = None
    lip_z_frac_h: float | None = None
    mouth_width_m: float | None = None
    loomis_residuals: dict[str, float | None] = Field(default_factory=dict)
    y_m: dict[str, float | None] = Field(default_factory=dict)


class FaceCompareCoord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    confidence: float | None = None
    sources: list[str] = Field(default_factory=list)


class FaceCompareDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float | None = None
    y: float | None = None
    z: float | None = None


class FaceCompareRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    measured: FaceCompareCoord | None = None
    recipe: dict[str, Any] | None = None
    live: dict[str, Any] | None = None
    delta_mm: FaceCompareDelta | None = None
    knob: str | None = None
    form_read: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    suggested: Literal["skip", "hold_priors", "soft_adjust", "session_0113", "remake_0111"] = "skip"


class FaceComparePackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = FACE_COMPARE_SCHEMA_VERSION
    honesty: str = FACE_COMPARE_HONESTY
    region: Literal["face"] = "face"
    ok: bool = False
    roles: list[FaceCompareRole] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    face_metrics: FaceMetrics | None = None
    package_path: str | None = None


def _z_frac(z_m: float | None, z_chin: float | None, h: float | None) -> float | None:
    zm = _as_float(z_m)
    zc = _as_float(z_chin)
    hh = _as_float(h)
    if zm is None or zc is None or hh is None or hh <= 0:
        return None
    return (zm - zc) / hh


def _head_rx_from_parts(parts: list[Any] | None) -> float | None:
    if not parts:
        return None
    for part in parts:
        if _part_name(part) == "RECIPE_head":
            return _as_float(_as_part_dict(part).get("rx_m"))
    return None


def build_face_metrics(
    report: ProportionReport,
    recipe: Any | None = None,
) -> FaceMetrics:
    """Compute face_metrics from landmarks_xyz (+ optional RECIPE head.rx)."""
    lms = report.landmarks_xyz
    eye_l = lms.get("eye_l")
    eye_r = lms.get("eye_r")
    ipd_m: float | None = None
    xl = _as_float(eye_l.x_m) if eye_l is not None else None
    xr = _as_float(eye_r.x_m) if eye_r is not None else None
    if xl is not None and xr is not None:
        ipd_m = abs(xr - xl)

    parts: list[Any] | None = None
    if recipe is not None:
        if isinstance(recipe, dict) and isinstance(recipe.get("parts"), list):
            parts = recipe["parts"]
        else:
            raw_parts = getattr(recipe, "parts", None)
            if isinstance(raw_parts, list):
                parts = raw_parts
    head_rx = _head_rx_from_parts(parts)
    half_over: float | None = None
    if ipd_m is not None and head_rx is not None and head_rx > 0:
        half_over = (ipd_m / 2.0) / head_rx

    chin = lms.get("chin")
    top = lms.get("cranial_vertex") or lms.get("hair_crown")
    z_chin = _as_float(chin.z_m) if chin is not None else None
    z_top = _as_float(top.z_m) if top is not None else None
    h = (z_top - z_chin) if z_chin is not None and z_top is not None else None
    if h is not None and h <= 0:
        h = None

    def _feat_z(lid: str) -> float | None:
        lm = lms.get(lid)
        return _as_float(lm.z_m) if lm is not None else None

    eye_z = _feat_z("eye_l")
    if eye_z is None:
        eye_z = _feat_z("eye_r")
    brow_z = _feat_z("brow_l")
    if brow_z is None:
        brow_z = _feat_z("brow_r")
    nose_z = _feat_z("nose_tip")
    lip_z = _feat_z("lip_mid")

    eye_frac = _z_frac(eye_z, z_chin, h)
    brow_frac = _z_frac(brow_z, z_chin, h)
    nose_frac = _z_frac(nose_z, z_chin, h)
    lip_frac = _z_frac(lip_z, z_chin, h)

    mouth_w: float | None = None
    mc_l = lms.get("mouth_corner_l")
    mc_r = lms.get("mouth_corner_r")
    mxl = _as_float(mc_l.x_m) if mc_l is not None else None
    mxr = _as_float(mc_r.x_m) if mc_r is not None else None
    if mxl is not None and mxr is not None:
        mouth_w = abs(mxr - mxl)

    y_fields: dict[str, float | None] = {}
    for lid in FACE_COMPARE_ROLES:
        lm = lms.get(lid)
        y_fields[lid] = _as_float(lm.y_m) if lm is not None else None

    residuals: dict[str, float | None] = {
        "eye_z": (eye_frac - _LOOMIS_EYE_Z) if eye_frac is not None else None,
        "brow_z": (brow_frac - _LOOMIS_BROW_Z) if brow_frac is not None else None,
        "nose_z": (nose_frac - _LOOMIS_NOSE_Z) if nose_frac is not None else None,
        "lip_z": (lip_frac - _LOOMIS_LIP_Z) if lip_frac is not None else None,
    }
    return FaceMetrics(
        ipd_m=ipd_m,
        half_ipd_over_head_rx=half_over,
        eye_z_frac_h=eye_frac,
        brow_z_frac_h=brow_frac,
        nose_z_frac_h=nose_frac,
        lip_z_frac_h=lip_frac,
        mouth_width_m=mouth_w,
        loomis_residuals=residuals,
        y_m=y_fields,
    )


def _measured_coord(lm: LandmarkXYZ | None) -> FaceCompareCoord | None:
    if lm is None:
        return None
    x_m = _as_float(lm.x_m)
    y_m = _as_float(lm.y_m)
    z_m = _as_float(lm.z_m)
    if x_m is None and y_m is None and z_m is None:
        return None
    return FaceCompareCoord(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        confidence=float(lm.confidence),
        sources=list(lm.sources),
    )


def _delta_mm(
    measured: FaceCompareCoord | None,
    recipe: dict[str, Any] | None,
) -> FaceCompareDelta | None:
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
    return FaceCompareDelta(x=dx, y=dy, z=dz)


def _suggest(
    measured: FaceCompareCoord | None,
    recipe: dict[str, Any] | None,
    delta: FaceCompareDelta | None,
) -> Literal["skip", "hold_priors", "soft_adjust", "session_0113", "remake_0111"]:
    if measured is None:
        return "skip"
    if recipe is None or recipe.get("center") is None:
        return "skip"
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
            code="face_compare_failed",
            details={"path": str(path)},
        ) from exc
    if not isinstance(raw, dict) or "parts" not in raw:
        raise ProportionError(
            f"scene dump must be an object with parts: {path}",
            code="face_compare_failed",
            details={"path": str(path)},
        )
    parts = raw.get("parts")
    if not isinstance(parts, list):
        raise ProportionError(
            f"scene dump parts must be a list: {path}",
            code="face_compare_failed",
            details={"path": str(path)},
        )
    known_names = set(_ROLE_PART.values()) | {"RECIPE_head", "RECIPE_jaw", "RECIPE_hair_mass"}
    out: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        if not isinstance(part, dict) or "name" not in part:
            raise ProportionError(
                f"scene dump part {i} must be an object with name",
                code="face_compare_failed",
                details={"path": str(path), "index": i},
            )
        name = str(part.get("name") or "")
        if name in known_names:
            has_center = _as_vec3(part.get("center")) is not None
            has_caps = _as_vec3(part.get("p0")) is not None and _as_vec3(part.get("p1")) is not None
            if not has_center and not has_caps:
                raise ProportionError(
                    f"scene dump part {name!r} missing center or p0/p1",
                    code="face_compare_failed",
                    details={"path": str(path), "index": i, "name": name},
                )
        out.append(part)
    return out


def _recipe_half_ipd_over_rx(parts: list[Any]) -> float | None:
    left = extract_recipe_face_part(parts, "eye_l")
    right = extract_recipe_face_part(parts, "eye_r")
    head_rx = _head_rx_from_parts(parts)
    if left is None or right is None or head_rx is None or head_rx <= 0:
        return None
    c0 = _as_vec3(left.get("center"))
    c1 = _as_vec3(right.get("center"))
    if c0 is None or c1 is None:
        return None
    half = abs(c1[0] - c0[0]) / 2.0
    return half / head_rx


def _nose_lip_overlap_m(parts: list[Any]) -> float | None:
    nose = extract_recipe_face_part(parts, "nose_tip")
    lip = extract_recipe_face_part(parts, "lip_mid")
    if nose is None or lip is None:
        return None
    nc = _as_vec3(nose.get("center"))
    lc = _as_vec3(lip.get("center"))
    nr = _as_float(nose.get("rz_m"))
    lr = _as_float(lip.get("rz_m"))
    if nc is None or lc is None or nr is None or lr is None:
        return None
    nose_bottom = nc[2] - nr
    lip_top = lc[2] + lr
    return lip_top - nose_bottom


def _has_finite_face(lms: dict[str, LandmarkXYZ]) -> bool:
    for lid in FACE_COMPARE_ROLES:
        lm = lms.get(lid)
        if lm is None:
            continue
        if any(_as_float(v) is not None for v in (lm.x_m, lm.y_m, lm.z_m)):
            return True
    return False


def run_blockout_face_compare(
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
    dest = out_dir / FACE_COMPARE_JSON
    if dest.exists() and not force:
        raise ProportionError(
            f"face compare exists: {dest} (pass --force)",
            code="face_compare_failed",
            details={"path": str(dest)},
        )

    rep = load_report(report_path)
    pkg = load_blockout_recipe(recipe_path)
    parts = list(pkg.parts)
    live_parts: list[dict[str, Any]] | None = None
    if scene_dump is not None:
        live_parts = _load_scene_dump(Path(scene_dump))

    messages: list[str] = [
        "FACE_COMPARE_HONESTY — authoring QA only; not mesh or print success",
    ]
    wide = _recipe_half_ipd_over_rx(parts)
    if wide is not None and wide >= WIDE_SET_HALF_IPD_RX:
        messages.append(f"eyes_wide_set half_ipd/rx={wide:.3f} >= {WIDE_SET_HALF_IPD_RX}")
    overlap = _nose_lip_overlap_m(parts)
    if overlap is not None and overlap > 0:
        messages.append(f"midline_blob nose/lip overlap_m={overlap:.4f}")
    if any(_part_name(p) == "RECIPE_hair_mass" for p in parts):
        messages.append("hat_hair_mass present — hide for face QA (0113)")

    roles: list[FaceCompareRole] = []
    lms = rep.landmarks_xyz
    for lid in FACE_COMPARE_ROLES:
        measured = _measured_coord(lms.get(lid))
        rec = extract_recipe_face_part(parts, lid)
        live = extract_recipe_face_part(live_parts, lid) if live_parts is not None else None
        delta = _delta_mm(measured, rec)
        form: list[str] = []
        if measured is None:
            form.append("missing_id")
        if lid in ("eye_l", "eye_r") and wide is not None and wide >= WIDE_SET_HALF_IPD_RX:
            form.append("eyes_wide_set")
        if lid in ("nose_tip", "lip_mid") and overlap is not None and overlap > 0:
            form.append("midline_blob")
        if lid in ("ear_l", "ear_r") and rec is not None:
            form.append("ear_cup_handle")
        if any(_part_name(p) == "RECIPE_hair_mass" for p in parts) and lid in ("eye_l", "eye_r"):
            form.append("hat_hair_mass")
        conf = 0.0
        if measured is not None and measured.confidence is not None:
            conf = float(measured.confidence)
        elif rec is not None:
            conf = 0.5
        roles.append(
            FaceCompareRole(
                id=lid,
                measured=measured,
                recipe=rec,
                live=live,
                delta_mm=delta,
                knob=_KNOB.get(lid),
                form_read=form,
                confidence=conf,
                suggested=_suggest(measured, rec, delta),
            )
        )

    metrics = build_face_metrics(rep, recipe=pkg)
    out_dir.mkdir(parents=True, exist_ok=True)
    package = FaceComparePackage(
        ok=True,
        roles=roles,
        messages=messages,
        face_metrics=metrics,
        package_path=str(dest),
    )
    dest.write_text(package.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if _has_finite_face(lms):
        (out_dir / FACE_METRICS_JSON).write_text(
            metrics.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    payload = package.model_dump(mode="json")
    return payload
