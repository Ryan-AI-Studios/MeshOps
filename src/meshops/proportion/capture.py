"""Proportion assist capture (track 0016).

Build landmarks_assist.json from pixel capture dumps, Blender ASSIST_* empty
dumps, or inverse-fuse reproject from guides/report + merge anchors.

Authoring aid only — not mesh or print success (Difficulty §12 / N6).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.assist import KNOWN_LANDMARK_IDS, load_assist_json
from meshops.proportion.errors import ProportionError
from meshops.proportion.fuse import DEPTH_PAIRS
from meshops.proportion.guides import GuidePackage
from meshops.proportion.honesty import CAPTURE_HONESTY
from meshops.proportion.load_views import load_views
from meshops.proportion.models import (
    CANONICAL_VIEW_KEYS,
    PROPORTION_SCHEMA_VERSION,
    ProportionReport,
)
from meshops.proportion.template import blank_assist_document

CAPTURE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"

CaptureSource = Literal["px", "dump", "reproject"]

# View prefix match longest-first (three_quarter before front).
_VIEW_PREFIX_ORDER: Final[tuple[str, ...]] = (
    "three_quarter",
    "front",
    "left",
    "back",
    "top",
)

# Depth-pair landmark ids → left view on reproject.
_LEFT_DEPTH_IDS: Final[frozenset[str]] = frozenset(
    lid for triple in DEPTH_PAIRS for lid in (triple[0], triple[1], triple[2])
) | frozenset({"spine_hint"})

# Prefixes / names to skip in dump (guides / seeds).
_SKIP_DUMP_PREFIXES: Final[tuple[str, ...]] = ("LM_", "SEED_")
_SKIP_DUMP_EXACT: Final[frozenset[str]] = frozenset({"LM_HEIGHT"})

_DEFAULT_CONF_PX_DUMP: Final[float] = 1.0
_DEFAULT_CONF_REPROJECT: Final[float] = 0.75

NOTE_PREFIX_ASSIST: Final[str] = "proportion_assist="
NOTE_PREFIX_REPORT: Final[str] = "proportion_report="


# ---------------------------------------------------------------------------
# Intermediate models (0016-owned schemas)
# ---------------------------------------------------------------------------


class CaptureViewSize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class AssistEmptyEntry(BaseModel):
    """One ASSIST_* empty from a Blender dump."""

    model_config = ConfigDict(extra="allow")

    name: str
    x_px: float
    y_px: float
    view: str | None = None
    landmark_id: str | None = None
    confidence: float | None = None
    px_source: str | None = None


class AssistEmptyDump(BaseModel):
    """Kind assist_empty_dump (schema 1.0.0)."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = CAPTURE_SCHEMA_VERSION
    kind: Literal["assist_empty_dump"] = "assist_empty_dump"
    honesty: str = CAPTURE_HONESTY
    view_sizes: dict[str, CaptureViewSize | dict[str, Any]] = Field(default_factory=dict)
    empties: list[AssistEmptyEntry] = Field(default_factory=list)
    pose: str | None = None
    multi_figure: bool | None = None
    messages: list[str] = Field(default_factory=list)


class AssistPixelCapture(BaseModel):
    """Kind assist_pixel_capture (schema 1.0.0)."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = CAPTURE_SCHEMA_VERSION
    kind: Literal["assist_pixel_capture"] = "assist_pixel_capture"
    honesty: str = CAPTURE_HONESTY
    pose: str = "unknown"
    multi_figure: bool = False
    views: dict[str, Any] = Field(default_factory=dict)
    edge_pairs: dict[str, Any] | None = None
    messages: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Name parse / helpers
# ---------------------------------------------------------------------------


def strip_blender_dup_suffix(name: str) -> str:
    """Strip Blender ``.001`` duplication suffixes (digits-only last segment)."""
    if "." not in name:
        return name
    base, last = name.rsplit(".", 1)
    if last.isdigit():
        return base
    return name


def parse_assist_empty_name(name: str) -> tuple[str, str] | None:
    """Parse ``ASSIST_{view}_{id}`` → (view, landmark_id) or None.

    Longest-first view match so three_quarter is not truncated to ``three``.
    """
    clean = strip_blender_dup_suffix(name.strip())
    if not clean.startswith("ASSIST_"):
        return None
    for vk in _VIEW_PREFIX_ORDER:
        prefix = f"ASSIST_{vk}_"
        if clean.startswith(prefix):
            lid = clean[len(prefix) :]
            if lid:
                return vk, lid
            return None
    return None


def should_skip_dump_name(name: str) -> str | None:
    """Return skip reason for LM_*/SEED_* (or None if not skipped by prefix)."""
    clean = strip_blender_dup_suffix(name.strip())
    if clean in _SKIP_DUMP_EXACT or clean.startswith("LM_HU_"):
        return f"skipped guide empty {name!r} (LM_*/HU)"
    for pref in _SKIP_DUMP_PREFIXES:
        if clean.startswith(pref):
            return f"skipped {pref.rstrip('_')} object {name!r}"
    return None


def _replace_prefixed_note(notes: list[str], prefix: str, new_line: str) -> list[str]:
    """Idempotent prefix replace (R9) — not exact-match append."""
    filtered = [n for n in notes if not n.startswith(prefix)]
    filtered.append(new_line)
    return filtered


def _coord_value(x: float, y: float, confidence: float | None) -> Any:
    """Serialize landmark coord for assist JSON."""
    if confidence is not None and confidence != 1.0:
        return {"x": float(x), "y": float(y), "confidence": float(confidence)}
    return [float(x), float(y)]


def _parse_coord_value(value: Any) -> tuple[float, float, float | None] | None:
    """Parse null | [x,y] | {x,y} | {x_px,y_px,confidence?} → (x,y,conf|None)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1]), None
    if isinstance(value, dict):
        if "x_px" in value and "y_px" in value:
            conf = value.get("confidence")
            return (
                float(value["x_px"]),
                float(value["y_px"]),
                float(conf) if conf is not None else None,
            )
        if "x" in value and "y" in value:
            conf = value.get("confidence")
            return (
                float(value["x"]),
                float(value["y"]),
                float(conf) if conf is not None else None,
            )
    if isinstance(value, (int, float)):
        # Scalar midline-style — x only; y unknown → store as (x, 0) not ideal;
        # treat as x at y=None skipped: use as midline x with y mid later.
        return float(value), 0.0, None
    return None


def _as_dict(val: Any) -> dict[str, Any]:
    """Narrow Any to dict for basedpyright."""
    return val if isinstance(val, dict) else {}


def _count_non_null_landmarks(doc: dict[str, Any]) -> tuple[int, int]:
    """Return (landmark_count, view_count_with_any)."""
    views = _as_dict(doc.get("views"))
    n_lm = 0
    n_views = 0
    for vdata in views.values():
        if not isinstance(vdata, dict):
            continue
        lms = _as_dict(vdata.get("landmarks"))
        view_has = False
        for val in lms.values():
            if val is not None:
                n_lm += 1
                view_has = True
        if view_has:
            n_views += 1
    return n_lm, n_views


def _set_landmark(
    doc: dict[str, Any],
    view: str,
    landmark_id: str,
    value: Any,
    *,
    messages: list[str],
) -> None:
    views = doc.setdefault("views", {})
    if view not in views or not isinstance(views[view], dict):
        views[view] = {
            "facing_direction": blank_assist_document()["views"]
            .get(view, {})
            .get("facing_direction", "unknown"),
            "landmarks": {},
        }
    lms = views[view].setdefault("landmarks", {})
    if not isinstance(lms, dict):
        views[view]["landmarks"] = {}
        lms = views[view]["landmarks"]
    if landmark_id not in KNOWN_LANDMARK_IDS and landmark_id not in lms:
        messages.append(f"unknown landmark id {landmark_id!r} on view {view} (allowed)")
    lms[landmark_id] = value


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot parse capture input JSON: {path}: {exc}",
            code="capture_failed",
            details={"path": str(path)},
        ) from exc
    if not isinstance(raw, dict):
        raise ProportionError(
            f"capture input root must be an object: {path}",
            code="capture_failed",
            details={"path": str(path)},
        )
    return raw


def _view_sizes_from_dir(views_dir: Path | str | None) -> dict[str, dict[str, int]]:
    if views_dir is None:
        return {}
    root = Path(views_dir)
    if not root.is_dir():
        raise ProportionError(
            f"views directory not found: {root}",
            code="capture_failed",
            details={"path": str(root)},
        )
    images = load_views(root, partial_ok=True, required=())
    return {
        key: {"width_px": img.width_px, "height_px": img.height_px} for key, img in images.items()
    }


def _normalize_view_sizes(
    raw: dict[str, Any],
    *,
    from_dir: dict[str, dict[str, int]] | None = None,
) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for key, val in raw.items():
        if key not in CANONICAL_VIEW_KEYS:
            continue
        if isinstance(val, dict):
            w = val.get("width_px")
            h = val.get("height_px")
            if w is not None and h is not None and int(w) > 0 and int(h) > 0:
                out[str(key)] = (int(w), int(h))
    if from_dir:
        for key, sz in from_dir.items():
            if key not in out:
                out[key] = (int(sz["width_px"]), int(sz["height_px"]))
    return out


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_assist_docs(
    base: dict[str, Any],
    new: dict[str, Any],
    *,
    prefer_merge: bool = False,
) -> dict[str, Any]:
    """Merge landmark coords (R5).

    Per landmark: merge non-null + new null → keep old; both non-null → new wins
    unless prefer_merge (then old wins).
    """
    out = blank_assist_document()
    # pose: prefer new if set; multi_figure: prefer new when present so
    # explicit --no-multi-figure (False) is not sticky-ORed with base True.
    out["pose"] = new.get("pose") or base.get("pose") or out["pose"]
    if "multi_figure" in new:
        out["multi_figure"] = bool(new["multi_figure"])
    else:
        out["multi_figure"] = bool(base.get("multi_figure", False))

    # edge_pairs
    base_ep = _as_dict(base.get("edge_pairs"))
    new_ep = _as_dict(new.get("edge_pairs"))
    merged_ep: dict[str, Any] = {}
    for vk in set(base_ep) | set(new_ep):
        b_bands = _as_dict(base_ep.get(vk))
        n_bands = _as_dict(new_ep.get(vk))
        band_out: dict[str, Any] = {}
        for band in set(b_bands) | set(n_bands):
            b_val = b_bands.get(band)
            n_val = n_bands.get(band)
            if n_val is None and b_val is not None:
                band_out[band] = b_val
            elif b_val is None and n_val is not None:
                band_out[band] = n_val
            elif b_val is not None and n_val is not None:
                band_out[band] = b_val if prefer_merge else n_val
            else:
                band_out[band] = None
        # Keep blank stubs for bands only in blank template when both missing
        out_ep = _as_dict(out.get("edge_pairs"))
        front_stubs = _as_dict(out_ep.get("front"))
        if vk == "front" and front_stubs:
            for stub_band, stub_val in front_stubs.items():
                band_out.setdefault(stub_band, stub_val)
        merged_ep[vk] = band_out
    if merged_ep:
        out["edge_pairs"] = merged_ep

    base_views = _as_dict(base.get("views"))
    new_views = _as_dict(new.get("views"))
    out_views = _as_dict(out.get("views"))
    all_views = set(out_views) | set(base_views) | set(new_views)

    for vk in all_views:
        b_v = _as_dict(base_views.get(vk))
        n_v = _as_dict(new_views.get(vk))
        if vk not in out_views:
            out_views[vk] = {
                "facing_direction": n_v.get("facing_direction")
                or b_v.get("facing_direction")
                or "unknown",
                "landmarks": {},
            }
        facing = n_v.get("facing_direction") or b_v.get("facing_direction")
        if facing:
            out_views[vk]["facing_direction"] = facing

        b_lm = _as_dict(b_v.get("landmarks"))
        n_lm = _as_dict(n_v.get("landmarks"))
        lms_out = _as_dict(out_views[vk].get("landmarks"))
        out_views[vk]["landmarks"] = lms_out
        for lid in set(b_lm) | set(n_lm) | set(lms_out):
            b_val = b_lm.get(lid)
            n_val = n_lm.get(lid)
            if n_val is None and b_val is not None:
                lms_out[lid] = b_val
            elif b_val is None and n_val is not None:
                lms_out[lid] = n_val
            elif b_val is not None and n_val is not None:
                lms_out[lid] = b_val if prefer_merge else n_val
            else:
                lms_out.setdefault(lid, None)
    out["views"] = out_views

    return out


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_assist_from_px(
    capture: dict[str, Any] | AssistPixelCapture,
    *,
    pose: str | None = None,
    multi_figure: bool | None = None,
    default_confidence: float = _DEFAULT_CONF_PX_DUMP,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize assist_pixel_capture → blank_assist_document shape (no width/height)."""
    data = capture.model_dump(mode="json") if isinstance(capture, AssistPixelCapture) else capture

    kind = data.get("kind")
    if kind != "assist_pixel_capture":
        raise ProportionError(
            f"expected kind assist_pixel_capture, got {kind!r}",
            code="capture_failed",
            details={"kind": kind},
        )

    messages: list[str] = list(data.get("messages") or [])
    doc = blank_assist_document()
    doc["pose"] = pose or data.get("pose") or "unknown"
    mf = multi_figure if multi_figure is not None else bool(data.get("multi_figure", False))
    doc["multi_figure"] = mf
    if mf:
        messages.append(
            "multi_figure=true — capture does not invent primary figure (Difficulty §1)"
        )

    views_raw = data.get("views") or {}
    if not isinstance(views_raw, dict):
        raise ProportionError(
            "pixel capture views must be an object",
            code="capture_failed",
        )

    for vk, vdata in views_raw.items():
        if vk not in CANONICAL_VIEW_KEYS:
            messages.append(f"unknown view key {vk!r} skipped")
            continue
        if not isinstance(vdata, dict):
            messages.append(f"view {vk!r} is not an object; skipped")
            continue
        facing = vdata.get("facing_direction")
        if facing:
            doc["views"][vk]["facing_direction"] = str(facing)
        lm_raw = vdata.get("landmarks") or {}
        if not isinstance(lm_raw, dict):
            messages.append(f"view {vk}.landmarks is not an object; skipped")
            continue
        for lid, val in lm_raw.items():
            parsed = _parse_coord_value(val)
            if parsed is None:
                continue
            x, y, conf = parsed
            c = conf if conf is not None else default_confidence
            _set_landmark(
                doc,
                str(vk),
                str(lid),
                _coord_value(x, y, c if c != 1.0 else None),
                messages=messages,
            )

    # edge_pairs pass-through (pixel pairs or null)
    if isinstance(data.get("edge_pairs"), dict):
        ep_in = data["edge_pairs"]
        ep_out = doc.get("edge_pairs") or {"front": {}}
        for vk, bands in ep_in.items():
            if not isinstance(bands, dict):
                continue
            target = ep_out.setdefault(str(vk), {})
            if not isinstance(target, dict):
                ep_out[str(vk)] = {}
                target = ep_out[str(vk)]
            for band, pair in bands.items():
                target[str(band)] = pair
        doc["edge_pairs"] = ep_out

    return doc, messages


def build_assist_from_dump(
    dump: dict[str, Any] | AssistEmptyDump,
    *,
    pose: str | None = None,
    multi_figure: bool | None = None,
    default_confidence: float = _DEFAULT_CONF_PX_DUMP,
    views_dir: Path | str | None = None,
) -> tuple[dict[str, Any], list[str], int]:
    """Build assist from assist_empty_dump. Returns (doc, messages, skipped_count)."""
    data = dump.model_dump(mode="json") if isinstance(dump, AssistEmptyDump) else dump

    kind = data.get("kind")
    if kind != "assist_empty_dump":
        raise ProportionError(
            f"expected kind assist_empty_dump, got {kind!r}",
            code="capture_failed",
            details={"kind": kind},
        )

    messages: list[str] = list(data.get("messages") or [])
    dir_sizes = _view_sizes_from_dir(views_dir)
    view_sizes = _normalize_view_sizes(
        _as_dict(data.get("view_sizes")),
        from_dir=dir_sizes,
    )

    doc = blank_assist_document()
    doc["pose"] = pose or data.get("pose") or "unknown"
    mf = multi_figure if multi_figure is not None else bool(data.get("multi_figure", False))
    doc["multi_figure"] = mf
    if mf:
        messages.append(
            "multi_figure=true — capture does not invent primary figure (Difficulty §1)"
        )

    empties = data.get("empties") or []
    if not isinstance(empties, list):
        raise ProportionError(
            "dump empties must be a list",
            code="capture_failed",
        )

    skipped = 0
    needed_views: set[str] = set()
    resolved: list[tuple[str, str, float, float, float | None, str]] = []

    for entry in empties:
        if not isinstance(entry, dict):
            messages.append("skipped non-object empty entry")
            skipped += 1
            continue
        name = str(entry.get("name") or "")
        skip_reason = should_skip_dump_name(name)
        if skip_reason:
            messages.append(skip_reason)
            skipped += 1
            continue

        view = entry.get("view")
        lid = entry.get("landmark_id")
        if view is None or lid is None:
            parsed = parse_assist_empty_name(name)
            if parsed is None:
                messages.append(f"unparseable empty name {name!r}; skipped")
                skipped += 1
                continue
            view, lid = parsed
        view_s = str(view)
        lid_s = str(lid)
        if view_s not in CANONICAL_VIEW_KEYS:
            messages.append(f"empty {name!r}: unknown view {view_s!r}; skipped")
            skipped += 1
            continue

        try:
            x_px = float(entry["x_px"])
            y_px = float(entry["y_px"])
        except (KeyError, TypeError, ValueError):
            messages.append(f"empty {name!r}: missing/invalid x_px/y_px; skipped")
            skipped += 1
            continue

        conf = entry.get("confidence")
        conf_f = float(conf) if conf is not None else default_confidence
        px_source = entry.get("px_source")
        if px_source == "location_fallback":
            messages.append(
                f"empty {name!r}: px_source=location_fallback (fragile; prefer meshops_x_px/y_px)"
            )
        needed_views.add(view_s)
        resolved.append((view_s, lid_s, x_px, y_px, conf_f, name))

    # Fail closed if any resolved view lacks sizes
    for vk in sorted(needed_views):
        if vk not in view_sizes:
            raise ProportionError(
                f"dump view {vk} missing from view_sizes — cannot compute coords",
                code="capture_failed",
                details={"view": vk},
            )

    for view_s, lid_s, x_px, y_px, conf_f, _name in resolved:
        _set_landmark(
            doc,
            view_s,
            lid_s,
            _coord_value(x_px, y_px, conf_f if conf_f != 1.0 else None),
            messages=messages,
        )

    return doc, messages, skipped


def _xy_from_assist_landmark(val: Any) -> tuple[float, float] | None:
    parsed = _parse_coord_value(val)
    if parsed is None:
        return None
    return parsed[0], parsed[1]


def _front_anchors_from_merge(
    merge_doc: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float], float | None]:
    """Return (sole_xy, top_xy, midline_x|None). Raises capture_failed if incomplete."""
    views = _as_dict(merge_doc.get("views"))
    front = _as_dict(views.get("front"))
    lms = _as_dict(front.get("landmarks"))

    sole = _xy_from_assist_landmark(lms.get("sole"))
    top = _xy_from_assist_landmark(lms.get("cranial_vertex"))
    if top is None:
        top = _xy_from_assist_landmark(lms.get("hair_crown"))

    if sole is None or top is None:
        raise ProportionError(
            "reproject requires --merge with stature anchors "
            "(front sole + cranial_vertex or hair_crown)",
            code="capture_failed",
        )

    midline: float | None = None
    mid = _xy_from_assist_landmark(lms.get("midline_x"))
    if mid is not None:
        midline = mid[0]
    else:
        mid2 = _xy_from_assist_landmark(lms.get("midline"))
        if mid2 is not None:
            midline = mid2[0]
        else:
            sl = _xy_from_assist_landmark(lms.get("shoulder_l"))
            sr = _xy_from_assist_landmark(lms.get("shoulder_r"))
            if sl is not None and sr is not None:
                midline = (sl[0] + sr[0]) / 2.0

    return sole, top, midline


def _left_anchors_from_merge(
    merge_doc: dict[str, Any],
) -> tuple[float, float, float | None, float | None]:
    """Return (torso_cx, left_span_hint, sole_y|None, top_y|None) or raise if missing pairs.

    left_span_hint is horizontal half-span proxy; reproject uses figure_h for scale.
    """
    views = _as_dict(merge_doc.get("views"))
    left = _as_dict(views.get("left"))
    lms = _as_dict(left.get("landmarks"))

    cf = _xy_from_assist_landmark(lms.get("chest_front"))
    cb = _xy_from_assist_landmark(lms.get("chest_back"))
    hf = _xy_from_assist_landmark(lms.get("hip_front"))
    hb = _xy_from_assist_landmark(lms.get("hip_back"))

    torso_cx: float | None = None
    if cf is not None and cb is not None:
        torso_cx = (cf[0] + cb[0]) / 2.0
    elif hf is not None and hb is not None:
        torso_cx = (hf[0] + hb[0]) / 2.0
    else:
        raise ProportionError(
            "reproject left anchors missing: need chest_front+chest_back or "
            "hip_front+hip_back in --merge assist left view",
            code="capture_failed",
        )

    sole = _xy_from_assist_landmark(lms.get("sole"))
    top = _xy_from_assist_landmark(lms.get("cranial_vertex"))
    if top is None:
        top = _xy_from_assist_landmark(lms.get("hair_crown"))
    sole_y = sole[1] if sole is not None else None
    top_y = top[1] if top is not None else None
    return float(torso_cx), 0.0, sole_y, top_y


def _meters_from_guides_or_report(
    data: dict[str, Any],
) -> tuple[dict[str, tuple[float | None, float | None, float | None]], float, list[str]]:
    """Extract landmark meters + height_m from guides or report JSON."""
    messages: list[str] = []

    # Prefer guides when empties present (0015 package); else report.
    if "empties" in data and "landmarks_xyz" not in data:
        try:
            pkg = GuidePackage.model_validate(data)
        except Exception as exc:
            raise ProportionError(
                f"reproject input is not a valid guides package: {exc}",
                code="capture_failed",
            ) from exc
        height_m = pkg.height_m
        if height_m is None or height_m <= 0:
            raise ProportionError(
                "reproject guides/report requires positive height_m",
                code="capture_failed",
            )
        out: dict[str, tuple[float | None, float | None, float | None]] = {}
        for e in pkg.empties:
            if e.kind != "landmark":
                continue
            sid = e.source_id or (e.name[3:] if e.name.startswith("LM_") else e.name)
            if not sid or sid.startswith("HU_"):
                continue
            out[sid] = (e.x_m, e.y_m, e.z_m)
        return out, float(height_m), messages

    try:
        report = ProportionReport.model_validate(data)
    except Exception as exc:
        raise ProportionError(
            f"reproject input is neither valid guides nor report: {exc}",
            code="capture_failed",
        ) from exc
    height_m = report.height_m
    if height_m is None or height_m <= 0:
        raise ProportionError(
            "reproject guides/report requires positive height_m",
            code="capture_failed",
        )
    out = {key: (lm.x_m, lm.y_m, lm.z_m) for key, lm in report.landmarks_xyz.items()}
    return out, float(height_m), messages


def _reproject_view_for_id(landmark_id: str) -> str | None:
    """Assign reprojected id to front or left; None → skip."""
    if landmark_id.startswith("SEED_") or landmark_id.startswith("LM_"):
        return None
    if landmark_id.endswith(("_edge0", "_edge1")):
        return None
    if landmark_id in _LEFT_DEPTH_IDS:
        return "left"
    if landmark_id.endswith(("_front", "_back")):
        return "left"
    return "front"


def build_assist_from_reproject(
    source: dict[str, Any],
    merge_doc: dict[str, Any],
    *,
    pose: str | None = None,
    multi_figure: bool | None = None,
    default_confidence: float = _DEFAULT_CONF_REPROJECT,
    views_dir: Path | str | None = None,
) -> tuple[dict[str, Any], list[str], int]:
    """Inverse-fuse meters → px using merge stature anchors."""
    messages: list[str] = []
    meters, height_m, extra = _meters_from_guides_or_report(source)
    messages.extend(extra)

    sole, top, midline_x = _front_anchors_from_merge(merge_doc)
    figure_h_px = sole[1] - top[1]
    if figure_h_px <= 0:
        raise ProportionError(
            "reproject figure_h_px non-positive (sole.y must be below top.y in image space)",
            code="capture_failed",
            details={"figure_h_px": figure_h_px},
        )

    if midline_x is None:
        messages.append("midline missing in merge assist — reproject X skipped (Z still filled)")

    # Determine if any left landmarks will be filled
    left_ids = [
        lid
        for lid in meters
        if _reproject_view_for_id(lid) == "left" and any(c is not None for c in meters[lid])
    ]
    torso_cx: float | None = None
    left_sole_y: float | None = None
    left_top_y: float | None = None
    if left_ids:
        torso_cx, _, left_sole_y, left_top_y = _left_anchors_from_merge(merge_doc)

    _ = views_dir  # reserved: optional view size hints (anchors already supply stature)

    doc = blank_assist_document()
    # Start from merge so unfilled keys preserve via later merge; builder only new coords
    doc["pose"] = pose or merge_doc.get("pose") or "unknown"
    mf = multi_figure if multi_figure is not None else bool(merge_doc.get("multi_figure", False))
    doc["multi_figure"] = mf
    if mf:
        messages.append(
            "multi_figure=true — capture does not invent primary figure (Difficulty §1)"
        )

    skipped = 0
    conf = default_confidence
    sign = 1.0  # camera_left

    for lid, (x_m, y_m, z_m) in meters.items():
        if lid.startswith("SEED_"):
            messages.append(f"skipped SEED_* id {lid!r}")
            skipped += 1
            continue
        target = _reproject_view_for_id(lid)
        if target is None:
            messages.append(f"skipped reproject id {lid!r} (not a normal assist key)")
            skipped += 1
            continue
        if x_m is None and y_m is None and z_m is None:
            skipped += 1
            continue

        if target == "front":
            if z_m is None:
                messages.append(f"{lid}: missing z_m — front reproject skipped")
                skipped += 1
                continue
            z = float(z_m) / height_m
            y_px = sole[1] - z * figure_h_px
            if x_m is not None and midline_x is not None:
                x = float(x_m) / height_m
                x_px = midline_x + x * figure_h_px
            elif midline_x is not None:
                x_px = midline_x
                messages.append(f"{lid}: missing x_m — used midline_x only")
            else:
                # no midline: still write y with sole.x as weak x
                x_px = sole[0]
                messages.append(f"{lid}: no midline — x set to sole.x")
            _set_landmark(
                doc,
                "front",
                lid,
                _coord_value(x_px, y_px, conf),
                messages=messages,
            )
        else:
            # left depth
            assert torso_cx is not None
            # figure span for left
            if left_sole_y is not None and left_top_y is not None:
                left_h = left_sole_y - left_top_y
                if left_h <= 0:
                    left_h = figure_h_px
                    messages.append("left sole/top invalid span; using front figure_h")
            else:
                left_h = figure_h_px
                messages.append("left sole/top missing; using front figure_h for left span")

            if y_m is None:
                messages.append(f"{lid}: missing y_m — left reproject skipped")
                skipped += 1
                continue
            y_body = float(y_m) / height_m
            x_px = float(torso_cx) + sign * y_body * left_h

            if z_m is not None:
                z = float(z_m) / height_m
                sole_y_use = left_sole_y if left_sole_y is not None else sole[1]
                y_px = sole_y_use - z * left_h
            else:
                # place mid-height of figure
                sole_y_use = left_sole_y if left_sole_y is not None else sole[1]
                y_px = sole_y_use - 0.5 * left_h
                messages.append(f"{lid}: missing z_m — y_px mid-span fallback")

            _set_landmark(
                doc,
                "left",
                lid,
                _coord_value(x_px, y_px, conf),
                messages=messages,
            )

    return doc, messages, skipped


# ---------------------------------------------------------------------------
# Write / dump script / attach
# ---------------------------------------------------------------------------


def write_assist(
    out_path: Path | str,
    doc: dict[str, Any],
    *,
    force: bool = False,
) -> Path:
    """Write landmarks_assist.json (blank_assist shape)."""
    path = Path(out_path)
    if path.exists() and not force:
        raise ProportionError(
            f"output already exists (use --force): {path}",
            code="write_failed",
            details={"path": str(path)},
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure schema_version is proportion assist version
        out_doc = dict(doc)
        out_doc["schema_version"] = PROPORTION_SCHEMA_VERSION
        # Strip any accidental per-view width/height
        views = out_doc.get("views")
        if isinstance(views, dict):
            for vdata in views.values():
                if isinstance(vdata, dict):
                    vdata.pop("width_px", None)
                    vdata.pop("height_px", None)
        path.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    except ProportionError:
        raise
    except OSError as exc:
        raise ProportionError(
            f"failed to write assist: {exc}",
            code="write_failed",
            details={"path": str(path)},
        ) from exc
    return path


def emit_dump_script() -> str:
    """Self-contained Blender 5.2 dump script (no meshops imports)."""
    honesty = CAPTURE_HONESTY
    lines = [
        "# assist_empty_dump.py — MeshOps track 0016",
        f"# honesty: {honesty}",
        "# N6 / Difficulty §12: capture is an authoring aid only —",
        "# not mesh reconstruction, not print-ready, not hero sculpt success.",
        "# Scans ASSIST_* empties; prefers custom props meshops_x_px / meshops_y_px.",
        "# location.x/y fallback is fragile (BU ≠ pixels).",
        "# Writes assist_empty_dump.json beside the .blend (or cwd if unsaved).",
        "",
        "import json",
        "import os",
        "import bpy",
        "",
        f"HONESTY = {honesty!r}",
        'KIND = "assist_empty_dump"',
        'SCHEMA = "1.0.0"',
        "",
        "# mode safety",
        'if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":',
        '    bpy.ops.object.mode_set(mode="OBJECT")',
        "",
        "",
        "def strip_dup(name):",
        '    if "." not in name:',
        "        return name",
        '    base, last = name.rsplit(".", 1)',
        "    if last.isdigit():",
        "        return base",
        "    return name",
        "",
        "",
        "def is_assist(name):",
        '    return strip_dup(name).startswith("ASSIST_")',
        "",
        "",
        "empties = []",
        "messages = [",
        '    "view_sizes is empty — fill width_px/height_px per view before capture, "',
        '    "or pass --views-dir to meshops proportion capture --source dump.",',
        "]",
        "fallback_count = 0",
        "",
        "for obj in bpy.data.objects:",
        "    if not is_assist(obj.name):",
        "        continue",
        "    name = obj.name",
        '    if "meshops_x_px" in obj and "meshops_y_px" in obj:',
        '        x_px = float(obj["meshops_x_px"])',
        '        y_px = float(obj["meshops_y_px"])',
        '        px_source = "custom_prop"',
        "    else:",
        "        x_px = float(obj.location.x)",
        "        y_px = float(obj.location.y)",
        '        px_source = "location_fallback"',
        "        fallback_count += 1",
        "        messages.append(",
        '            f"{name!r}: using location as px (fragile; set meshops_x_px/meshops_y_px)"',
        "        )",
        "    empties.append(",
        "        {",
        '            "name": name,',
        '            "x_px": x_px,',
        '            "y_px": y_px,',
        '            "px_source": px_source,',
        "        }",
        "    )",
        "",
        "if fallback_count:",
        '    print(f"WARNING: {fallback_count} ASSIST_* empties used location fallback")',
        "",
        "doc = {",
        '    "schema_version": SCHEMA,',
        '    "kind": KIND,',
        '    "honesty": HONESTY,',
        '    "view_sizes": {},',
        '    "empties": empties,',
        '    "pose": "unknown",',
        '    "multi_figure": False,',
        '    "messages": messages,',
        "}",
        "",
        "blend = bpy.data.filepath",
        "out_dir = os.path.dirname(blend) if blend else os.getcwd()",
        'out_path = os.path.join(out_dir, "assist_empty_dump.json")',
        'with open(out_path, "w", encoding="utf-8") as f:',
        "    json.dump(doc, f, indent=2)",
        '    f.write("\\n")',
        'print("wrote", os.path.abspath(out_path))',
        'print("honesty:", HONESTY)',
        'print("ASSIST empties:", len(empties))',
        "",
    ]
    return "\n".join(lines)


def write_dump_script(path: Path | str) -> Path:
    """Write the self-contained dump script to *path*."""
    out = Path(path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(emit_dump_script(), encoding="utf-8")
    except OSError as exc:
        raise ProportionError(
            f"failed to write dump script: {exc}",
            code="write_failed",
            details={"path": str(out)},
        ) from exc
    return out


def attach_to_organic_session(
    session_id: str,
    artifact_path: Path | str,
    *,
    work_root: Path | str = "work",
    note_prefix: str,
    dest_basename: str,
) -> Path:
    """Copy artifact under session organic/proportion/ and update notes (R9).

    Uses live SessionPaths layout: work_root/<session_id>/organic/.
    """
    from meshops.organic.errors import OrganicError
    from meshops.organic.session import load_session, save_manifest

    try:
        paths, manifest = load_session(session_id, work_root=work_root)
    except OrganicError as exc:
        raise ProportionError(
            f"organic session not found: {session_id}",
            code="capture_failed",
            details={"session_id": session_id, "organic_error": str(exc)},
        ) from exc

    src = Path(artifact_path)
    if not src.is_file():
        raise ProportionError(
            f"attach artifact not found: {src}",
            code="capture_failed",
            details={"path": str(src)},
        )

    prop_dir = paths.organic_dir / "proportion"
    prop_dir.mkdir(parents=True, exist_ok=True)
    dest = prop_dir / dest_basename
    try:
        shutil.copy2(src, dest)
    except OSError as exc:
        raise ProportionError(
            f"failed to copy artifact into organic session: {exc}",
            code="write_failed",
            details={"dest": str(dest)},
        ) from exc

    note_line = f"{note_prefix}{dest}"
    manifest.notes = _replace_prefixed_note(list(manifest.notes), note_prefix, note_line)
    save_manifest(paths, manifest)
    return dest


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_capture(
    *,
    source: CaptureSource | None = None,
    in_path: Path | str | None = None,
    out_path: Path | str | None = None,
    views_dir: Path | str | None = None,
    pose: str | None = None,
    multi_figure: bool | None = None,
    merge_path: Path | str | None = None,
    prefer_merge: bool = False,
    default_confidence: float | None = None,
    force: bool = False,
    emit_dump_script_path: Path | str | None = None,
    attach_session: str | None = None,
    work_root: Path | str | None = None,
) -> dict[str, Any]:
    """CLI/MCP entry: build assist from source, write, optional attach.

    ``emit_dump_script_path`` alone is allowed (writes script and returns early).
    """
    messages: list[str] = []
    paths_written: list[str] = []

    if emit_dump_script_path is not None:
        sp = write_dump_script(emit_dump_script_path)
        paths_written.append(str(sp))
        if source is None and in_path is None and out_path is None:
            return {
                "ok": True,
                "source": None,
                "out": str(sp),
                "counts": {"landmarks": 0, "views": 0, "skipped": 0},
                "messages": [f"wrote dump script: {sp}", CAPTURE_HONESTY],
                "honesty": CAPTURE_HONESTY,
                "paths": paths_written,
            }

    if source is None:
        raise ProportionError(
            "--source is required (px|dump|reproject); v1 does not silent-infer",
            code="capture_failed",
        )
    if source not in ("px", "dump", "reproject"):
        raise ProportionError(
            f"invalid --source {source!r}; expected px|dump|reproject",
            code="capture_failed",
        )
    if in_path is None:
        raise ProportionError("--in is required for capture", code="capture_failed")
    if out_path is None:
        raise ProportionError("--out is required for capture", code="capture_failed")

    conf = default_confidence
    if conf is None:
        conf = _DEFAULT_CONF_REPROJECT if source == "reproject" else _DEFAULT_CONF_PX_DUMP

    raw = _load_json_object(Path(in_path))
    skipped = 0
    new_doc: dict[str, Any]

    if source == "px":
        new_doc, msgs = build_assist_from_px(
            raw,
            pose=pose,
            multi_figure=multi_figure,
            default_confidence=conf,
        )
        messages.extend(msgs)
    elif source == "dump":
        new_doc, msgs, skipped = build_assist_from_dump(
            raw,
            pose=pose,
            multi_figure=multi_figure,
            default_confidence=conf,
            views_dir=views_dir,
        )
        messages.extend(msgs)
    else:
        if merge_path is None:
            raise ProportionError(
                "reproject requires --merge with stature anchors (front sole + cranial/hair)",
                code="capture_failed",
            )
        merge_doc = load_assist_json(merge_path)
        new_doc, msgs, skipped = build_assist_from_reproject(
            raw,
            merge_doc,
            pose=pose,
            multi_figure=multi_figure,
            default_confidence=conf,
            views_dir=views_dir,
        )
        messages.extend(msgs)
        # Merge reproject result into existing assist
        new_doc = merge_assist_docs(merge_doc, new_doc, prefer_merge=prefer_merge)
        merge_path = None  # already applied

    if merge_path is not None and source != "reproject":
        base = load_assist_json(merge_path)
        new_doc = merge_assist_docs(base, new_doc, prefer_merge=prefer_merge)

    n_lm, n_views = _count_non_null_landmarks(new_doc)
    if n_lm == 0:
        raise ProportionError(
            "no landmarks written (capture empty)",
            code="capture_empty",
            details={"source": source, "in": str(in_path)},
        )

    written = write_assist(out_path, new_doc, force=force)
    paths_written.append(str(written))

    attach_dest: str | None = None
    wr = work_root if work_root is not None else os.environ.get("MESHOPS_WORK", "work")
    if attach_session:
        dest = attach_to_organic_session(
            attach_session,
            written,
            work_root=wr,
            note_prefix=NOTE_PREFIX_ASSIST,
            dest_basename="landmarks_assist.json",
        )
        attach_dest = str(dest)
        messages.append(f"attached assist to organic session {attach_session}: {dest}")

    messages.append(CAPTURE_HONESTY)
    return {
        "ok": True,
        "source": source,
        "out": str(written),
        "counts": {"landmarks": n_lm, "views": n_views, "skipped": skipped},
        "messages": messages,
        "honesty": CAPTURE_HONESTY,
        "paths": paths_written,
        "attach_dest": attach_dest,
    }


def attach_report_to_organic_session(
    session_id: str,
    report_path: Path | str,
    *,
    work_root: Path | str = "work",
) -> Path:
    """Attach proportion_report.json to organic session (analyze path)."""
    return attach_to_organic_session(
        session_id,
        report_path,
        work_root=work_root,
        note_prefix=NOTE_PREFIX_REPORT,
        dest_basename="proportion_report.json",
    )


__all__ = [
    "CAPTURE_SCHEMA_VERSION",
    "NOTE_PREFIX_ASSIST",
    "NOTE_PREFIX_REPORT",
    "AssistEmptyDump",
    "AssistPixelCapture",
    "attach_report_to_organic_session",
    "attach_to_organic_session",
    "build_assist_from_dump",
    "build_assist_from_px",
    "build_assist_from_reproject",
    "emit_dump_script",
    "merge_assist_docs",
    "parse_assist_empty_name",
    "run_capture",
    "strip_blender_dup_suffix",
    "write_assist",
    "write_dump_script",
]
