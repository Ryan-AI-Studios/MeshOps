"""Arm/hand landmark compare: photo vs RECIPE vs optional live scene (track 0129).

Authoring QA only — ARM_HAND_COMPARE_HONESTY. Not mesh or print success.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.analyze import load_report
from meshops.proportion.blockout_recipe import (
    BICEP_ALONG_T,
    TRICEP_ALONG_T,
    load_blockout_recipe,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import ARM_HAND_COMPARE_HONESTY
from meshops.proportion.models import LandmarkXYZ, ProportionReport

ARM_HAND_COMPARE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
ARM_HAND_COMPARE_JSON: Final[str] = "arm_hand_compare.json"
ARM_HAND_METRICS_JSON: Final[str] = "arm_hand_metrics.json"
ARM_HAND_COMPARE_ROLES: Final[tuple[str, ...]] = (
    "humeral_head_l",
    "humeral_head_r",
    "bi_belly_l",
    "bi_belly_r",
    "tri_belly_l",
    "tri_belly_r",
    "olecranon_l",
    "olecranon_r",
    "fa_belly_l",
    "fa_belly_r",
    "palm_center_l",
    "palm_center_r",
    "thumb_cmc_l",
    "thumb_cmc_r",
    "thumb_tip_l",
    "thumb_tip_r",
    "mcp_index_l",
    "mcp_index_r",
    "mcp_pinky_l",
    "mcp_pinky_r",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
)
SUGGESTED_ACTIONS: Final[frozenset[str]] = frozenset(
    {"skip", "hold_priors", "soft_adjust", "session_arm_hand", "remake_0111"}
)
FORM_READ_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "bi_front_past",
        "tri_rear_past",
        "fa_stepped",
        "thumb_not_forward",
        "missing_id",
    }
)
BI_FRONT_PAST_FLAG_M: Final[float] = 0.006
TRI_REAR_PAST_FLAG_M: Final[float] = 0.006
_CHAIN_NEAR_ZERO_LEN: Final[float] = 1e-9  # emit _NEAR_ZERO_LEN
FA_STEPPED_RATIO_MAX: Final[float] = 0.78
THUMB_FORWARD_Y_MAX: Final[float] = -0.40

_ROLE_PART: dict[str, str] = {
    "humeral_head_l": "RECIPE_limb_upper_arm_l",
    "humeral_head_r": "RECIPE_limb_upper_arm_r",
    "bi_belly_l": "RECIPE_bicep_soft_l",
    "bi_belly_r": "RECIPE_bicep_soft_r",
    "tri_belly_l": "RECIPE_triceps_soft_l",
    "tri_belly_r": "RECIPE_triceps_soft_r",
    "olecranon_l": "RECIPE_elbow_soft_l",
    "olecranon_r": "RECIPE_elbow_soft_r",
    "fa_belly_l": "RECIPE_limb_forearm_l",
    "fa_belly_r": "RECIPE_limb_forearm_r",
    "palm_center_l": "RECIPE_palm_l",
    "palm_center_r": "RECIPE_palm_r",
    "thumb_cmc_l": "RECIPE_thumb_soft_0_l",
    "thumb_cmc_r": "RECIPE_thumb_soft_0_r",
    "thumb_tip_l": "RECIPE_thumb_soft_1_l",
    "thumb_tip_r": "RECIPE_thumb_soft_1_r",
    "mcp_index_l": "RECIPE_finger_index_0_l",
    "mcp_index_r": "RECIPE_finger_index_0_r",
    "mcp_pinky_l": "RECIPE_finger_pinky_0_l",
    "mcp_pinky_r": "RECIPE_finger_pinky_0_r",
    "shoulder_l": "RECIPE_deltoid_soft_l",
    "shoulder_r": "RECIPE_deltoid_soft_r",
    "elbow_l": "RECIPE_elbow_soft_l",
    "elbow_r": "RECIPE_elbow_soft_r",
    "wrist_l": "RECIPE_dist_soft_forearm_l",
    "wrist_r": "RECIPE_dist_soft_forearm_r",
}

# B36/B40: soft_adjust only for new-id bi/tri belly Y and Z.
_SOFT_ADJUST_IDS: Final[frozenset[str]] = frozenset(
    {"bi_belly_l", "bi_belly_r", "tri_belly_l", "tri_belly_r"}
)

# B39: role-mapped capsule endpoints (not 0127 always-p0; not 0126 center-skip).
_ENDPOINT: dict[str, Literal["p0", "p1", "midpoint"]] = {
    "humeral_head_l": "p0",
    "humeral_head_r": "p0",
    "fa_belly_l": "midpoint",
    "fa_belly_r": "midpoint",
    "thumb_cmc_l": "p0",
    "thumb_cmc_r": "p0",
    "thumb_tip_l": "p1",
    "thumb_tip_r": "p1",
    "mcp_index_l": "p0",
    "mcp_index_r": "p0",
    "mcp_pinky_l": "p0",
    "mcp_pinky_r": "p0",
}

_KNOB: dict[str, str] = {
    "humeral_head_l": "UA_PROX_SHAFT_SCALE",
    "humeral_head_r": "UA_PROX_SHAFT_SCALE",
    "bi_belly_l": "BICEP_FRONT_PAST_M",
    "bi_belly_r": "BICEP_FRONT_PAST_M",
    "tri_belly_l": "TRICEP_REAR_PAST_M",
    "tri_belly_r": "TRICEP_REAR_PAST_M",
    "olecranon_l": "ELBOW_SOFT_SCALE",
    "olecranon_r": "ELBOW_SOFT_SCALE",
    "fa_belly_l": "FA_PROX_SHAFT_SCALE",
    "fa_belly_r": "FA_PROX_SHAFT_SCALE",
    "palm_center_l": "_PALM_WIDTH_FRAC_HAND",
    "palm_center_r": "_PALM_WIDTH_FRAC_HAND",
    "thumb_cmc_l": "_THUMB_PALM_PITCH",
    "thumb_cmc_r": "_THUMB_PALM_PITCH",
    "thumb_tip_l": "_THUMB_PALM_PITCH",
    "thumb_tip_r": "_THUMB_PALM_PITCH",
    "mcp_index_l": "_FINGER_R_SCALES_SEG",
    "mcp_index_r": "_FINGER_R_SCALES_SEG",
    "mcp_pinky_l": "_FINGER_R_SCALES_SEG",
    "mcp_pinky_r": "_FINGER_R_SCALES_SEG",
    "shoulder_l": "DELT_RY_FRAC",
    "shoulder_r": "DELT_RY_FRAC",
    "elbow_l": "ELBOW_SOFT_SCALE",
    "elbow_r": "ELBOW_SOFT_SCALE",
    "wrist_l": "FA_DIST_SHAFT_SCALE",
    "wrist_r": "FA_DIST_SHAFT_SCALE",
}

_KNOWN_DUMP_NAMES: Final[frozenset[str]] = frozenset(
    {
        "RECIPE_limb_upper_arm_l",
        "RECIPE_limb_upper_arm_r",
        "RECIPE_arm_taper_dist_ua_l",
        "RECIPE_arm_taper_dist_ua_r",
        "RECIPE_limb_forearm_l",
        "RECIPE_limb_forearm_r",
        "RECIPE_arm_taper_dist_fa_l",
        "RECIPE_arm_taper_dist_fa_r",
        "RECIPE_bicep_soft_l",
        "RECIPE_bicep_soft_r",
        "RECIPE_triceps_soft_l",
        "RECIPE_triceps_soft_r",
        "RECIPE_elbow_soft_l",
        "RECIPE_elbow_soft_r",
        "RECIPE_dist_soft_forearm_l",
        "RECIPE_dist_soft_forearm_r",
        "RECIPE_palm_l",
        "RECIPE_palm_r",
        "RECIPE_thumb_soft_0_l",
        "RECIPE_thumb_soft_0_r",
        "RECIPE_thumb_soft_1_l",
        "RECIPE_thumb_soft_1_r",
        "RECIPE_finger_index_0_l",
        "RECIPE_finger_index_0_r",
        "RECIPE_finger_pinky_0_l",
        "RECIPE_finger_pinky_0_r",
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
    """B39: UA/FA/thumb/finger use role-mapped p0/p1/midpoint — never 0127 always-p0."""
    which = _ENDPOINT.get(role_id)
    if which == "p0":
        return _as_vec3(part.get("p0"))
    if which == "p1":
        return _as_vec3(part.get("p1"))
    if which == "midpoint":
        return _midpoint(part.get("p0"), part.get("p1"))
    return None


def extract_recipe_arm_hand_part(parts: list[Any], role_id: str) -> dict[str, Any] | None:
    """Extract one arm/hand role from RECIPE parts (B26/B39 mapped capsule endpoints)."""
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


class ArmHandMetrics(BaseModel):
    """Sidecar arm/hand meters (not a ProportionReport field — stay 1.2.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    honesty: str = ARM_HAND_COMPARE_HONESTY
    bicep_front_past_m: float | None = None
    triceps_rear_past_m: float | None = None
    ua_prox_r_m: float | None = None
    ua_dist_r_m: float | None = None
    fa_prox_r_m: float | None = None
    fa_dist_r_m: float | None = None
    bicep_rx_m: float | None = None
    triceps_rx_m: float | None = None
    palm_rx_m: float | None = None
    palm_ry_m: float | None = None
    y_m: dict[str, float | None] = Field(default_factory=dict)


class ArmHandCompareCoord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    confidence: float | None = None
    sources: list[str] = Field(default_factory=list)


class ArmHandCompareDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float | None = None
    y: float | None = None
    z: float | None = None


class ArmHandCompareRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    measured: ArmHandCompareCoord | None = None
    recipe: dict[str, Any] | None = None
    live: dict[str, Any] | None = None
    delta_mm: ArmHandCompareDelta | None = None
    knob: str | None = None
    form_read: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    suggested: Literal["skip", "hold_priors", "soft_adjust", "session_arm_hand", "remake_0111"] = (
        "skip"
    )


class ArmHandComparePackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = ARM_HAND_COMPARE_SCHEMA_VERSION
    honesty: str = ARM_HAND_COMPARE_HONESTY
    region: Literal["arm_hand"] = "arm_hand"
    ok: bool = False
    roles: list[ArmHandCompareRole] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    arm_hand_metrics: ArmHandMetrics | None = None
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


def _ua_chain_mid_y_r(
    ua: dict[str, Any] | None,
    ua_dist: dict[str, Any] | None,
    *,
    along_t: float,
) -> tuple[float | None, float | None]:
    """0063 parity: p0=ua.p0; p1=taper.p1 else ua.p1; None if invalid / near-zero."""
    if ua is None:
        return None, None
    ua_r = _as_float(ua.get("radius_m"))
    if ua_r is None or ua_r <= 0.0:
        return None, None
    p0 = _as_vec3(ua.get("p0"))
    if p0 is None:
        return None, None
    p1 = _as_vec3(ua_dist.get("p1")) if ua_dist is not None else None
    if p1 is None:
        p1 = _as_vec3(ua.get("p1"))
    if p1 is None:
        return None, None
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dz = p1[2] - p0[2]
    if math.sqrt(dx * dx + dy * dy + dz * dz) <= _CHAIN_NEAR_ZERO_LEN:
        return None, None
    t = float(along_t)
    mid_y = p0[1] + t * (p1[1] - p0[1])
    return mid_y, ua_r


def _bicep_front_past(
    bicep: dict[str, Any] | None,
    ua: dict[str, Any] | None,
    ua_dist: dict[str, Any] | None,
) -> float | None:
    """B35: None unless bicep center/ry and UA r are finite."""
    if bicep is None or ua is None:
        return None
    center = _as_vec3(bicep.get("center"))
    ry = _as_float(bicep.get("ry_m"))
    mid_y, ua_r = _ua_chain_mid_y_r(ua, ua_dist, along_t=BICEP_ALONG_T)
    if center is None or ry is None or mid_y is None or ua_r is None:
        return None
    shaft_front = mid_y - ua_r
    return shaft_front - (center[1] - ry)


def _triceps_rear_past(
    triceps: dict[str, Any] | None,
    ua: dict[str, Any] | None,
    ua_dist: dict[str, Any] | None,
) -> float | None:
    if triceps is None or ua is None:
        return None
    center = _as_vec3(triceps.get("center"))
    ry = _as_float(triceps.get("ry_m"))
    mid_y, ua_r = _ua_chain_mid_y_r(ua, ua_dist, along_t=TRICEP_ALONG_T)
    if center is None or ry is None or mid_y is None or ua_r is None:
        return None
    shaft_rear = mid_y + ua_r
    return (center[1] + ry) - shaft_rear


def build_arm_hand_metrics(
    report: ProportionReport,
    recipe: Any | None = None,
) -> ArmHandMetrics:
    """Compute arm_hand_metrics from landmarks_xyz (+ optional RECIPE parts)."""
    parts = _recipe_parts(recipe)
    side = (
        "l"
        if _find_part(parts, "RECIPE_limb_upper_arm_l") is not None
        else "r"
        if _find_part(parts, "RECIPE_limb_upper_arm_r") is not None
        else None
    )
    ua = _find_part(parts, f"RECIPE_limb_upper_arm_{side}") if side else None
    ua_dist = _find_part(parts, f"RECIPE_arm_taper_dist_ua_{side}") if side else None
    fa = _find_part(parts, "RECIPE_limb_forearm_l") or _find_part(parts, "RECIPE_limb_forearm_r")
    fa_dist = _find_part(parts, "RECIPE_arm_taper_dist_fa_l") or _find_part(
        parts, "RECIPE_arm_taper_dist_fa_r"
    )
    bicep = (
        _find_part(parts, f"RECIPE_bicep_soft_{side}")
        if side
        else _find_part(parts, "RECIPE_bicep_soft_l") or _find_part(parts, "RECIPE_bicep_soft_r")
    )
    triceps = (
        _find_part(parts, f"RECIPE_triceps_soft_{side}")
        if side
        else _find_part(parts, "RECIPE_triceps_soft_l")
        or _find_part(parts, "RECIPE_triceps_soft_r")
    )
    palm = _find_part(parts, "RECIPE_palm_l") or _find_part(parts, "RECIPE_palm_r")

    lms = report.landmarks_xyz
    y_fields: dict[str, float | None] = {}
    for lid in ARM_HAND_COMPARE_ROLES:
        lm = lms.get(lid)
        y_fields[lid] = _as_float(lm.y_m) if lm is not None else None

    return ArmHandMetrics(
        bicep_front_past_m=_bicep_front_past(bicep, ua, ua_dist),
        triceps_rear_past_m=_triceps_rear_past(triceps, ua, ua_dist),
        ua_prox_r_m=_as_float(ua.get("radius_m")) if ua is not None else None,
        ua_dist_r_m=_as_float(ua_dist.get("radius_m")) if ua_dist is not None else None,
        fa_prox_r_m=_as_float(fa.get("radius_m")) if fa is not None else None,
        fa_dist_r_m=_as_float(fa_dist.get("radius_m")) if fa_dist is not None else None,
        bicep_rx_m=_as_float(bicep.get("rx_m")) if bicep is not None else None,
        triceps_rx_m=_as_float(triceps.get("rx_m")) if triceps is not None else None,
        palm_rx_m=_as_float(palm.get("rx_m")) if palm is not None else None,
        palm_ry_m=_as_float(palm.get("ry_m")) if palm is not None else None,
        y_m=y_fields,
    )


def _measured_coord(lm: LandmarkXYZ | None) -> ArmHandCompareCoord | None:
    if lm is None:
        return None
    x_m = _as_float(lm.x_m)
    y_m = _as_float(lm.y_m)
    z_m = _as_float(lm.z_m)
    if x_m is None and y_m is None and z_m is None:
        return None
    return ArmHandCompareCoord(
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
    measured: ArmHandCompareCoord | None,
    recipe: dict[str, Any] | None,
    role_id: str,
) -> ArmHandCompareDelta | None:
    if measured is None or recipe is None:
        return None
    # B39: capsule/cylinder roles use mapped p0/p1/midpoint (live center=null).
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
    return ArmHandCompareDelta(x=dx, y=dy, z=dz)


def _suggest(
    measured: ArmHandCompareCoord | None,
    recipe: dict[str, Any] | None,
    delta: ArmHandCompareDelta | None,
    role_id: str,
) -> Literal["skip", "hold_priors", "soft_adjust", "session_arm_hand", "remake_0111"]:
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
            code="arm_hand_compare_failed",
            details={"path": str(path)},
        ) from exc
    if not isinstance(raw, dict) or "parts" not in raw:
        raise ProportionError(
            f"scene dump must be an object with parts: {path}",
            code="arm_hand_compare_failed",
            details={"path": str(path)},
        )
    parts = raw.get("parts")
    if not isinstance(parts, list):
        raise ProportionError(
            f"scene dump parts must be a list: {path}",
            code="arm_hand_compare_failed",
            details={"path": str(path)},
        )
    out: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        if not isinstance(part, dict) or "name" not in part:
            raise ProportionError(
                f"scene dump part {i} must be an object with name",
                code="arm_hand_compare_failed",
                details={"path": str(path), "index": i},
            )
        name = str(part.get("name") or "")
        if name in _KNOWN_DUMP_NAMES:
            has_center = _as_vec3(part.get("center")) is not None
            has_caps = _as_vec3(part.get("p0")) is not None and _as_vec3(part.get("p1")) is not None
            if not has_center and not has_caps:
                raise ProportionError(
                    f"scene dump part {name!r} missing center or p0/p1",
                    code="arm_hand_compare_failed",
                    details={"path": str(path), "index": i, "name": name},
                )
        out.append(part)
    return out


def _has_finite_arm_hand(lms: dict[str, LandmarkXYZ]) -> bool:
    for lid in ARM_HAND_COMPARE_ROLES:
        lm = lms.get(lid)
        if lm is None:
            continue
        if any(_as_float(v) is not None for v in (lm.x_m, lm.y_m, lm.z_m)):
            return True
    return False


def _thumb_axis_y(parts: list[Any]) -> float | None:
    for name in ("RECIPE_thumb_soft_0_l", "RECIPE_thumb_soft_0_r"):
        part = _find_part(parts, name)
        if part is None:
            continue
        p0 = _as_vec3(part.get("p0"))
        p1 = _as_vec3(part.get("p1"))
        if p0 is None or p1 is None:
            continue
        dy = p1[1] - p0[1]
        dx = p1[0] - p0[0]
        dz = p1[2] - p0[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-12:
            continue
        return dy / length
    return None


def run_blockout_arm_hand_compare(
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
    dest = out_dir / ARM_HAND_COMPARE_JSON
    if dest.exists() and not force:
        raise ProportionError(
            f"arm/hand compare exists: {dest} (pass --force)",
            code="arm_hand_compare_failed",
            details={"path": str(dest)},
        )

    rep = load_report(report_path)
    pkg = load_blockout_recipe(recipe_path)
    parts = list(pkg.parts)
    live_parts: list[dict[str, Any]] | None = None
    if scene_dump is not None:
        live_parts = _load_scene_dump(Path(scene_dump))

    messages: list[str] = [
        "ARM_HAND_COMPARE_HONESTY — authoring QA only; not mesh or print success",
    ]
    metrics = build_arm_hand_metrics(rep, recipe=pkg)
    bi_flag = (
        metrics.bicep_front_past_m is not None and metrics.bicep_front_past_m < BI_FRONT_PAST_FLAG_M
    )
    if bi_flag:
        messages.append(f"bicep_front_past_m={metrics.bicep_front_past_m:.4f}")
    tri_flag = (
        metrics.triceps_rear_past_m is not None
        and metrics.triceps_rear_past_m < TRI_REAR_PAST_FLAG_M
    )
    if tri_flag:
        messages.append(f"triceps_rear_past_m={metrics.triceps_rear_past_m:.4f}")
    fa_flag = False
    if (
        metrics.fa_prox_r_m is not None
        and metrics.fa_dist_r_m is not None
        and metrics.fa_prox_r_m > 0.0
    ):
        ratio = metrics.fa_dist_r_m / metrics.fa_prox_r_m
        fa_flag = ratio <= FA_STEPPED_RATIO_MAX
        if fa_flag:
            messages.append(f"fa_stepped_ratio={ratio:.4f}")
    thumb_y = _thumb_axis_y(parts)
    thumb_flag = thumb_y is not None and thumb_y >= THUMB_FORWARD_Y_MAX
    if thumb_flag:
        messages.append(f"thumb_axis_y={thumb_y:.4f}")

    lms = rep.landmarks_xyz
    roles: list[ArmHandCompareRole] = []
    for lid in ARM_HAND_COMPARE_ROLES:
        measured = _measured_coord(lms.get(lid))
        rec = extract_recipe_arm_hand_part(parts, lid)
        live = extract_recipe_arm_hand_part(live_parts, lid) if live_parts is not None else None
        delta = _delta_mm(measured, rec, lid)
        form: list[str] = []
        if measured is None:
            form.append("missing_id")
        if bi_flag and lid.startswith("bi_belly_"):
            form.append("bi_front_past")
        if tri_flag and lid.startswith("tri_belly_"):
            form.append("tri_rear_past")
        if fa_flag and lid.startswith("fa_belly_"):
            form.append("fa_stepped")
        if thumb_flag and lid.startswith("thumb_"):
            form.append("thumb_not_forward")
        conf = 0.0
        if measured is not None and measured.confidence is not None:
            conf = float(measured.confidence)
        elif rec is not None:
            conf = 0.5
        roles.append(
            ArmHandCompareRole(
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
    package = ArmHandComparePackage(
        ok=True,
        roles=roles,
        messages=messages,
        arm_hand_metrics=metrics,
        package_path=str(dest),
    )
    dest.write_text(package.model_dump_json(indent=2) + "\n", encoding="utf-8")
    sidecar = out_dir / ARM_HAND_METRICS_JSON
    if _has_finite_arm_hand(lms):
        sidecar.write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
    elif sidecar.exists():
        sidecar.unlink()
    return package.model_dump(mode="json")
