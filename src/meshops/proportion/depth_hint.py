"""Depth-channel assist hints from external disparity maps (track 0020).

Side document (landmarks_assist.hint.json) is authoring-only and is never
auto-loaded by analyze. Use --merge-into for a canonical assist loadable by
apply_assist. Not mesh or print success (Difficulty §12 / N6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.errors import ProportionError
from meshops.proportion.fuse import DEPTH_PAIRS
from meshops.proportion.honesty import HINT_HONESTY
from meshops.proportion.load_views import png_size

HINT_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
HINT_BASENAME: Final[str] = "landmarks_assist.hint.json"
HINT_KIND: Final[str] = "landmarks_assist_hint"

CONF_PROTECT: Final[float] = 0.99
CONF_HINT_DEFAULT: Final[float] = 0.35
BODY_MASK_FRAC: Final[float] = 0.05

# Anatomical prior (~8-head-ish stature fractions; not measured law) — C1
DEFAULT_Z_FRAC: Final[dict[str, float]] = {
    "chest": 0.72,
    "breast": 0.70,
    "hip": 0.53,
    "glute": 0.50,
    "thigh": 0.35,
    "calf": 0.18,
}

MONOCULAR_UNAVAILABLE_MSG: Final[str] = (
    "monocular backend not available (no torch/onnx pin; research option: "
    "Depth Anything V2 / YOLO26-depth — install separately and use "
    "--backend external with the generated depth map)"
)

HintBackend = Literal["external", "monocular"]
HintMethod = Literal["depth_channel_external", "monocular_unavailable"]


class HintPoint(BaseModel):
    """One hinted landmark in pixel coords."""

    model_config = ConfigDict(extra="forbid")

    x_px: float
    y_px: float
    confidence: float = Field(default=CONF_HINT_DEFAULT, ge=0.0, le=1.0)


class DepthHintPackage(BaseModel):
    """Side-document landmarks_assist.hint.json (schema 1.0.0) — B1."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = HINT_SCHEMA_VERSION
    kind: Literal["landmarks_assist_hint"] = HINT_KIND
    honesty: str = HINT_HONESTY
    method: HintMethod = "depth_channel_external"
    backend: HintBackend = "external"
    source_depth_map: str | None = None
    source_left: str | None = None
    messages: list[str] = Field(default_factory=list)
    hints: dict[str, dict[str, HintPoint]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pillow_required(what: str = "depth-map pixel decode") -> ProportionError:
    return ProportionError(
        f"{what} require Pillow (install meshops[proportion] / pillow)",
        code="pillow_required",
        details={"hint": "uv sync --extra proportion"},
    )


def _band_from_front_id(front_id: str) -> str:
    """chest_front → chest."""
    if front_id.endswith("_front"):
        return front_id[: -len("_front")]
    return front_id


def _resolve_hint_out(out: Path | str) -> Path:
    """Resolve side-doc path from --out (file or directory)."""
    raw = str(out)
    ends_sep = raw.endswith(("/", "\\"))
    path = Path(raw.rstrip("/\\") if ends_sep else raw)
    if (path.exists() and path.is_dir()) or ends_sep:
        return path / HINT_BASENAME
    return path


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            raise ProportionError(
                f"output already exists (use --force): {path}",
                code="write_failed",
                details={"path": str(path)},
            )
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except ProportionError:
        raise
    except OSError as exc:
        raise ProportionError(
            f"failed to write depth hint: {exc}",
            code="write_failed",
            details={"path": str(path)},
        ) from exc


def _z_frac_from_report(
    report_path: Path | None,
    band: str,
    messages: list[str],
) -> float:
    """Band z_frac from report depth_bands, else DEFAULT_Z_FRAC + message (C1)."""
    if report_path is not None and report_path.is_file():
        try:
            from meshops.proportion.analyze import load_report

            rep = load_report(report_path)
            for db in rep.depth_bands:
                if db.band_id == band and db.z_frac is not None:
                    return float(db.z_frac)
        except ProportionError:
            messages.append(f"report unreadable for z_frac ({report_path}); using defaults")

    if band in DEFAULT_Z_FRAC:
        messages.append(f"z_frac for {band} from DEFAULT_Z_FRAC (no report / band missing)")
        return float(DEFAULT_Z_FRAC[band])

    messages.append(f"z_frac for {band} from DEFAULT_Z_FRAC (no report / band missing)")
    return 0.5


def _load_depth_array(path: Path) -> np.ndarray:
    """Load depth-map pixels as 2D float array (Pillow required — D4)."""
    try:
        from PIL import Image  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:
        raise _pillow_required("depth-map pixel decode") from exc

    try:
        with Image.open(path) as im:
            # Prefer single-channel; convert RGB via luminance if needed
            if im.mode in ("I", "I;16", "F", "L"):
                arr = np.asarray(im, dtype=np.float64)
            else:
                gray = im.convert("L")
                arr = np.asarray(gray, dtype=np.float64)
    except OSError as exc:
        raise ProportionError(
            f"cannot open depth map: {path}: {exc}",
            code="hint_failed",
            details={"path": str(path)},
        ) from exc

    if arr.ndim != 2:
        arr = np.asarray(arr[..., 0], dtype=np.float64)
    return arr


def _load_left_alpha(path: Path, w: int, h: int) -> np.ndarray | None:
    """Optional alpha channel from left PNG (Pillow). None if unavailable."""
    try:
        from PIL import Image  # type: ignore[import-untyped,import-not-found]
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            if im.mode in ("RGBA", "LA"):
                im = im.resize((w, h), Image.Resampling.NEAREST)
                alpha = np.asarray(im.split()[-1], dtype=np.float64)
                return alpha
            if im.mode == "P" and "transparency" in im.info:
                rgba = im.convert("RGBA").resize((w, h), Image.Resampling.NEAREST)
                return np.asarray(rgba.split()[-1], dtype=np.float64)
    except OSError:
        return None
    return None


def _resample_nearest(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbor resize to (H, W)."""
    src_h, src_w = arr.shape[:2]
    if src_w == w and src_h == h:
        return arr.astype(np.float64, copy=False)
    # Map output pixels to source
    ys = np.clip(np.round(np.linspace(0, src_h - 1, h)).astype(np.intp), 0, src_h - 1)
    xs = np.clip(np.round(np.linspace(0, src_w - 1, w)).astype(np.intp), 0, src_w - 1)
    return arr[np.ix_(ys, xs)].astype(np.float64, copy=False)


def extract_depth_hints(
    depth_map: Path | str,
    left: Path | str,
    *,
    report: Path | str | None = None,
) -> tuple[dict[str, HintPoint], list[str], int]:
    """Extract front/back pixel hints for DEPTH_PAIRS bands.

    Depth convention (C3): larger pixel value = closer (disparity-style).
    Returns (hints_left, messages, pairs_count).
    """
    left_path = Path(left)
    depth_path = Path(depth_map)
    report_path = Path(report) if report is not None else None
    messages: list[str] = []

    if not left_path.is_file():
        raise ProportionError(
            f"left view not found: {left_path}",
            code="hint_failed",
            details={"path": str(left_path)},
        )
    if not depth_path.is_file():
        raise ProportionError(
            f"depth map not found: {depth_path}",
            code="hint_failed",
            details={"path": str(depth_path)},
        )

    # Left size via IHDR (no Pillow) for PNG
    try:
        w, h = png_size(left_path) if left_path.suffix.lower() == ".png" else (0, 0)
    except ProportionError:
        w, h = 0, 0
    if w <= 0 or h <= 0:
        # Fallback: Pillow size
        try:
            from PIL import Image  # type: ignore[import-untyped,import-not-found]
        except ImportError as exc:
            raise _pillow_required("left view sizing") from exc
        try:
            with Image.open(left_path) as im:
                w, h = im.size
        except OSError as exc:
            raise ProportionError(
                f"cannot size left view: {left_path}: {exc}",
                code="hint_failed",
                details={"path": str(left_path)},
            ) from exc

    depth = _load_depth_array(depth_path)
    depth = _resample_nearest(depth, w, h)
    alpha = _load_left_alpha(left_path, w, h)

    hints: dict[str, HintPoint] = {}
    pairs_ok = 0

    for front_id, back_id, _mid in DEPTH_PAIRS:
        band = _band_from_front_id(front_id)
        z_frac = _z_frac_from_report(report_path, band, messages)
        y_px = (1.0 - float(z_frac)) * (h - 1)
        row_i = round(y_px)
        row_i = max(0, min(h - 1, row_i))
        row = depth[row_i, :]

        # Background: depth <= 0; optional AND left alpha == 0 (C2)
        mask = row > 0.0
        if alpha is not None:
            mask = mask & (alpha[row_i, :] > 0.0)

        if not np.any(mask):
            messages.append(f"band {band}: no body pixels on row (all background)")
            continue

        remaining = row[mask]
        row_max = float(np.max(remaining))
        if row_max <= 0:
            messages.append(f"band {band}: row_max <= 0; skipped")
            continue

        body_mask = mask & (row > BODY_MASK_FRAC * row_max)
        body_count = int(np.count_nonzero(body_mask))
        if body_count < 3:
            messages.append(f"band {band}: body width {body_count} < 3; skipped")
            continue

        xs = np.flatnonzero(body_mask)
        vals = row[xs]
        # Front = argmax depth (closer); back = argmin (C3 disparity)
        front_x = float(xs[int(np.argmax(vals))])
        back_x = float(xs[int(np.argmin(vals))])
        y_out = float(y_px)

        hints[front_id] = HintPoint(x_px=front_x, y_px=y_out, confidence=CONF_HINT_DEFAULT)
        hints[back_id] = HintPoint(x_px=back_x, y_px=y_out, confidence=CONF_HINT_DEFAULT)
        pairs_ok += 1

    if not hints:
        raise ProportionError(
            "no depth-pair hints extracted (empty body mask / all bands skipped)",
            code="hint_empty",
            details={"depth_map": str(depth_path), "left": str(left_path)},
        )

    return hints, messages, pairs_ok


def _existing_confidence(value: Any) -> float | None:
    """Read conf from assist landmark value if present."""
    if isinstance(value, dict) and "confidence" in value:
        try:
            return float(value["confidence"])
        except (TypeError, ValueError):
            return None
    return None


def merge_hints_into_assist(
    hints_left: dict[str, HintPoint],
    *,
    assist: dict[str, Any] | None = None,
    force_hint: bool = False,
) -> tuple[dict[str, Any], int]:
    """Merge hint points into canonical assist (views.left.landmarks).

    Conf floor R6:
      missing → insert
      conf >= CONF_PROTECT without force_hint → skip
      conf >= CONF_PROTECT with force_hint → replace
      conf < CONF_PROTECT → replace
    Returns (assist_doc, protected_skipped).
    """
    if assist is None:
        doc: dict[str, Any] = {
            "schema_version": "1.0.0",
            "multi_figure": False,
            "views": {"left": {"landmarks": {}}},
        }
    else:
        doc = json.loads(json.dumps(assist))  # deep copy via JSON

    views = doc.setdefault("views", {})
    if not isinstance(views, dict):
        views = {}
        doc["views"] = views
    left = views.setdefault("left", {})
    if not isinstance(left, dict):
        left = {}
        views["left"] = left
    landmarks = left.setdefault("landmarks", {})
    if not isinstance(landmarks, dict):
        landmarks = {}
        left["landmarks"] = landmarks

    if "schema_version" not in doc:
        doc["schema_version"] = "1.0.0"
    if "multi_figure" not in doc:
        doc["multi_figure"] = False

    protected_skipped = 0
    for lid, hp in hints_left.items():
        existing = landmarks.get(lid)
        if existing is not None:
            conf = _existing_confidence(existing)
            if conf is not None and conf >= CONF_PROTECT and not force_hint:
                protected_skipped += 1
                continue
        landmarks[lid] = {
            "x_px": float(hp.x_px),
            "y_px": float(hp.y_px),
            "confidence": float(hp.confidence),
        }
    return doc, protected_skipped


def build_hint_package(
    hints_left: dict[str, HintPoint],
    *,
    depth_map: str | None,
    left: str | None,
    messages: list[str],
    backend: HintBackend = "external",
) -> DepthHintPackage:
    """Build side-document package (B1)."""
    method: HintMethod = (
        "depth_channel_external" if backend == "external" else "monocular_unavailable"
    )
    return DepthHintPackage(
        schema_version=HINT_SCHEMA_VERSION,
        kind=HINT_KIND,
        honesty=HINT_HONESTY,
        method=method,
        backend=backend,
        source_depth_map=depth_map,
        source_left=left,
        messages=list(messages),
        hints={"left": hints_left},
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def run_depth_hint(
    depth_map: Path | str | None,
    left: Path | str | None,
    out: Path | str,
    *,
    assist: Path | str | None = None,
    report: Path | str | None = None,
    backend: HintBackend | str = "external",
    force: bool = False,
    force_hint: bool = False,
    merge_into: Path | str | None = None,
) -> dict[str, Any]:
    """Extract depth hints → side .hint.json; optional merge-into assist.

    Raises ProportionError. Returns CLI/MCP success payload.
    """
    be = (backend or "external").strip().lower()
    if be not in ("external", "monocular"):
        raise ProportionError(
            f"unknown depth-hint backend: {backend!r} (use external|monocular)",
            code="hint_failed",
            details={"backend": backend},
        )

    if be == "monocular":
        raise ProportionError(
            MONOCULAR_UNAVAILABLE_MSG,
            code="monocular_unavailable",
            details={"backend": "monocular"},
        )

    if depth_map is None or left is None:
        raise ProportionError(
            "backend=external requires --depth-map and --left",
            code="hint_failed",
            details={},
        )

    hints_left, messages, pairs = extract_depth_hints(
        depth_map,
        left,
        report=report,
    )

    pkg = build_hint_package(
        hints_left,
        depth_map=str(depth_map),
        left=str(left),
        messages=messages,
        backend="external",
    )

    hint_path = _resolve_hint_out(out)
    _write_json(hint_path, pkg.model_dump(mode="json"), force=force)

    paths: list[str] = [str(hint_path)]
    protected_skipped = 0

    if merge_into is not None:
        merge_path = Path(merge_into)
        base_assist: dict[str, Any] | None = None
        if assist is not None:
            ap = Path(assist)
            if ap.is_file():
                try:
                    from meshops.proportion.assist import load_assist_json

                    base_assist = load_assist_json(ap)
                except ProportionError as exc:
                    raise ProportionError(
                        f"cannot load --assist for merge: {exc}",
                        code="invalid_assist",
                        details={"path": str(ap)},
                    ) from exc
        elif merge_path.is_file():
            # Start from existing merge target when no --assist
            try:
                from meshops.proportion.assist import load_assist_json

                base_assist = load_assist_json(merge_path)
            except ProportionError:
                base_assist = None

        merged, protected_skipped = merge_hints_into_assist(
            hints_left,
            assist=base_assist,
            force_hint=force_hint,
        )
        _write_json(merge_path, merged, force=force)
        paths.append(str(merge_path))

    return {
        "ok": True,
        "paths": paths,
        "counts": {
            "hints": len(hints_left),
            "protected_skipped": protected_skipped,
            "pairs": pairs,
        },
        "messages": messages,
    }


__all__ = [
    "BODY_MASK_FRAC",
    "CONF_HINT_DEFAULT",
    "CONF_PROTECT",
    "DEFAULT_Z_FRAC",
    "HINT_BASENAME",
    "HINT_KIND",
    "HINT_SCHEMA_VERSION",
    "MONOCULAR_UNAVAILABLE_MSG",
    "DepthHintPackage",
    "HintPoint",
    "build_hint_package",
    "extract_depth_hints",
    "merge_hints_into_assist",
    "run_depth_hint",
]
