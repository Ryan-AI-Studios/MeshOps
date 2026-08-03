"""Front-only binary silhouette IoU/Dice compare (track 0021 + 0025 trust).

Authoring QA score between Package A front ref and mesh front view.
Not mesh or print success (Difficulty §12 / N6). Front-vs-front only.

0025: studio-gray recovery cascade + silhouette_trusted / trust_reasons.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.errors import ProportionError
from meshops.proportion.frame import (
    _LARGE_BLOB_FRAC,
    _load_rgba_array,
    connected_large_blobs,
    silhouette_mask,
)
from meshops.proportion.honesty import SILHOUETTE_HONESTY

SILHOUETTE_SCHEMA_VERSION: Final[Literal["1.1.0"]] = "1.1.0"
SILHOUETTE_JSON_BASENAME: Final[str] = "silhouette_compare.json"
SILHOUETTE_OVERLAY_BASENAME: Final[str] = "silhouette_overlay.png"
GRID_PX: Final[int] = 256
LUMA_THR: Final[int] = 18
METHOD: Final[Literal["binary_mask_iou_dice"]] = "binary_mask_iou_dice"
ALIGNMENT_MODE: Final[Literal["content_bbox"]] = "content_bbox"
RESIZE_LABEL: Final[Literal["nearest_256"]] = "nearest_256"

# Trust band for final post-recovery coverage (inclusive).
_COV_LO: Final[float] = 0.02
_COV_HI: Final[float] = 0.90
# Otsu between-class variance floor (AI2 A) — flat/unimodal hist fails.
_OTSU_MIN_SIGMA_B2: Final[float] = 10.0
# Corner std above this → bg_uncertain when still out of band.
_BG_STD_UNCERTAIN: Final[float] = 25.0
# Alpha matte usefulness thresholds.
_ALPHA_FRAC_TRANS: Final[float] = 0.01
_ALPHA_STD_MIN: Final[float] = 5.0
_ALPHA_OPAQUE_THR: Final[int] = 16

MaskMethod = Literal["primary", "alpha_matte", "corner_median", "otsu_luma", "empty"]

TRUST_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "coverage_high",
        "coverage_low",
        "coverage_high_after_recovery",
        "recovery_failed",
        "bg_uncertain",
        "otsu_low_histogram_bimodality",
        "ref_untrusted",
        "mesh_untrusted",
    }
)

_NON_FRONT_STEM_TOKENS: Final[tuple[str, ...]] = (
    "left",
    "right",
    "side",
    "profile",
    "back",
    "rear",
    "camera_left",
    "camera_right",
    "camera_back",
)

_BASENAME_ADVISORY: Final[str] = (
    "Advisory: --{side} filename appears non-front; ensure comparison is "
    "front-vs-front per MeshOps QA law."
)

_IDENTICAL_MSG: Final[str] = "ref and mesh-view are identical — score is trivially 1.0"


class SilhouetteScores(BaseModel):
    """IoU + Dice on the 256^2 aligned grid."""

    model_config = ConfigDict(extra="forbid")

    iou: float
    dice: float


class SilhouetteCounts(BaseModel):
    """Foreground pixel counts on grid and original masks."""

    model_config = ConfigDict(extra="forbid")

    ref_fg_grid_px: int
    mesh_fg_grid_px: int
    intersection_grid_px: int
    union_grid_px: int
    ref_fg_orig_px: int
    mesh_fg_orig_px: int


class SilhouetteAlignment(BaseModel):
    """Content-bbox normalize freezes (no rotation)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["content_bbox"] = ALIGNMENT_MODE
    rotation: bool = False
    resize: Literal["nearest_256"] = RESIZE_LABEL
    ref_bbox: list[int] | None = None
    mesh_bbox: list[int] | None = None


class SilhouetteComparePackage(BaseModel):
    """silhouette_compare.json package (schema 1.1.0 — write-only)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.1.0"] = SILHOUETTE_SCHEMA_VERSION
    honesty: str = SILHOUETTE_HONESTY
    view_role: Literal["front"] = "front"
    ref_path: str
    mesh_path: str | None = None
    mesh_view_path: str | None = None
    method: Literal["binary_mask_iou_dice"] = METHOD
    grid_px: int = GRID_PX
    scores: SilhouetteScores
    counts: SilhouetteCounts
    alignment: SilhouetteAlignment
    messages: list[str] = Field(default_factory=list)
    overlay_path: str | None = None
    # 0025 trust fields (R1)
    silhouette_trusted: bool = False
    trust_reasons: list[str] = Field(default_factory=list)
    mask_method_ref: MaskMethod = "primary"
    mask_method_mesh: MaskMethod = "primary"
    ref_coverage_frac: float = 0.0
    mesh_coverage_frac: float = 0.0


class _MaskExtractResult(NamedTuple):
    """Per-side mask extraction outcome (post-recovery when triggered)."""

    mask: np.ndarray
    messages: list[str]
    method: MaskMethod
    trust_reasons: list[str]
    coverage_frac: float
    recovery_ran: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pillow_required() -> ProportionError:
    return ProportionError(
        "silhouette compare requires Pillow (install meshops[proportion] / pillow)",
        code="pillow_required",
        details={"hint": "uv sync --extra proportion"},
    )


def _require_pillow() -> Any:
    try:
        from PIL import Image  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:
        raise _pillow_required() from exc
    return Image


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _stem_looks_non_front(path: Path) -> bool:
    stem = path.stem.lower()
    return any(tok in stem for tok in _NON_FRONT_STEM_TOKENS)


def _resolve_out_paths(out: Path | str) -> tuple[Path, Path]:
    """Resolve (json_path, overlay_path) from --out (C7 / depth_heatmap style).

    ``.json`` suffix → that file; overlay sibling ``silhouette_overlay.png``.
    Else treat as directory. A ``.json`` path that exists as a directory fails.
    """
    raw = str(out)
    ends_sep = raw.endswith(("/", "\\"))
    path = Path(raw.rstrip("/\\") if ends_sep else raw)

    if path.suffix.lower() == ".json":
        if path.exists() and path.is_dir():
            raise ProportionError(
                f"--out .json path exists as a directory: {path}",
                code="silhouette_failed",
                details={"out": raw},
            )
        return path, path.parent / SILHOUETTE_OVERLAY_BASENAME

    # Directory (existing, trailing sep, or path without .json suffix)
    if ends_sep or (path.exists() and path.is_dir()) or path.suffix.lower() != ".json":
        return path / SILHOUETTE_JSON_BASENAME, path / SILHOUETTE_OVERLAY_BASENAME

    raise ProportionError(
        f"cannot resolve --out: {raw}",
        code="silhouette_failed",
        details={"out": raw},
    )


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
            f"failed to write silhouette compare: {exc}",
            code="write_failed",
            details={"path": str(path)},
        ) from exc


def _write_png(path: Path, image: Any, *, force: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            raise ProportionError(
                f"output already exists (use --force): {path}",
                code="write_failed",
                details={"path": str(path)},
            )
        image.save(path)
    except ProportionError:
        raise
    except OSError as exc:
        raise ProportionError(
            f"failed to write silhouette overlay: {exc}",
            code="write_failed",
            details={"path": str(path)},
        ) from exc


def _luma(rgba: np.ndarray) -> np.ndarray:
    """Rec. 601 luminance from HxWxC uint8."""
    rgb = rgba[:, :, :3].astype(np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _coverage_frac(mask: np.ndarray) -> float:
    h, w = mask.shape
    return float(mask.sum()) / float(max(h * w, 1))


def _in_trust_band(coverage: float) -> bool:
    return _COV_LO <= coverage <= _COV_HI


def _alpha_channel(rgba: np.ndarray) -> np.ndarray:
    if rgba.shape[2] >= 4:
        return rgba[:, :, 3]
    return np.full(rgba.shape[:2], 255, dtype=np.uint8)


def _alpha_useful(rgba: np.ndarray) -> bool:
    """Prefer alpha matte when channel has real transparency / variance."""
    alpha = _alpha_channel(rgba)
    frac_trans = float(np.mean(alpha.astype(np.float64) < 250.0))
    std = float(np.std(alpha.astype(np.float64)))
    return frac_trans >= _ALPHA_FRAC_TRANS or std >= _ALPHA_STD_MIN


def _alpha_matte_mask(rgba: np.ndarray) -> np.ndarray:
    """FG from alpha; exclude near-white RGB when present (optional and-not)."""
    alpha = _alpha_channel(rgba)
    opaque = alpha > _ALPHA_OPAQUE_THR
    rgb = rgba[:, :, :3].astype(np.int16)
    near_white = (rgb[:, :, 0] >= 250) & (rgb[:, :, 1] >= 250) & (rgb[:, :, 2] >= 250)
    return opaque & ~near_white


def _corner_median_luma(luma: np.ndarray) -> tuple[float, float, list[float]]:
    """Return (bg_median, std_of_corners, [tl, tr, bl, br])."""
    h, w = luma.shape
    patch = max(1, min(3, h, w))

    def _med(y0: int, x0: int) -> float:
        y1 = min(h, y0 + patch)
        x1 = min(w, x0 + patch)
        return float(np.median(luma[y0:y1, x0:x1]))

    tl = _med(0, 0)
    tr = _med(0, max(0, w - patch))
    bl = _med(max(0, h - patch), 0)
    br = _med(max(0, h - patch), max(0, w - patch))
    corners = [tl, tr, bl, br]
    bg = float(np.median(np.asarray(corners, dtype=np.float64)))
    std = float(np.std(np.asarray(corners, dtype=np.float64)))
    return bg, std, corners


def _corner_median_mask(
    rgba: np.ndarray,
) -> tuple[np.ndarray, float, float, list[str]]:
    """Corner-median luma FG mask; returns (mask, bg_median, std, messages)."""
    messages: list[str] = []
    luma = _luma(rgba)
    bg, std, _corners = _corner_median_luma(luma)
    mask = np.abs(luma - bg) > float(LUMA_THR)
    messages.append("primary silhouette_mask empty — used corner-median fallback")
    if std > _BG_STD_UNCERTAIN:
        messages.append("background estimate uncertain — busy frame")
    return mask, bg, std, messages


def _otsu_threshold(hist: np.ndarray) -> tuple[int, float]:
    """Classic Otsu on 256-bin histogram → (threshold, max between-class variance).

    Uses class probabilities so sigma_b2 is scale-free vs image size (AI2 A floor 10.0).
    """
    total = float(hist.sum())
    if total <= 0.0:
        return 0, 0.0

    levels = np.arange(256, dtype=np.float64)
    sum_total = float(np.dot(levels, hist.astype(np.float64)))

    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    best_t = 0

    for t in range(256):
        w_b += float(hist[t])
        if w_b <= 0.0:
            continue
        w_f = total - w_b
        if w_f <= 0.0:
            break
        sum_b += float(t) * float(hist[t])
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        # Probability-weighted between-class variance
        omega0 = w_b / total
        omega1 = w_f / total
        var = omega0 * omega1 * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var
            best_t = t

    if max_var < 0.0:
        return 0, 0.0
    return best_t, float(max_var)


def _otsu_fg_mask(
    luma_u8: np.ndarray,
    t: int,
    *,
    bg_median: float | None,
    bg_std: float | None,
) -> tuple[np.ndarray, list[str]]:
    """Pick FG side of Otsu threshold (B2: corner BG sign preferred).

    Classic Otsu ``t`` is the last level of the low class (levels 0..t). Light-BG
    FG therefore uses ``luma <= t`` (equivalent to freeze ``luma < t'`` with
    t' = t+1) so mode pixels at exactly ``t`` stay in the low class.
    """
    messages: list[str] = []
    use_bg_sign = bg_median is not None and bg_std is not None and bg_std <= _BG_STD_UNCERTAIN
    if use_bg_sign:
        assert bg_median is not None
        if bg_median >= 128.0:
            mask = luma_u8 <= t
            messages.append(f"otsu FG = luma <= {t} (bg_median={bg_median:.1f})")
        else:
            mask = luma_u8 > t
            messages.append(f"otsu FG = luma > {t} (bg_median={bg_median:.1f})")
        return mask, messages

    # Ambiguous / no corner: coverage closest to 0.25 in band, else max <=0.90
    low = luma_u8 <= t
    high = luma_u8 > t
    candidates: list[tuple[np.ndarray, float]] = [
        (low, _coverage_frac(low)),
        (high, _coverage_frac(high)),
    ]
    in_band = [(m, c) for m, c in candidates if _in_trust_band(c)]
    if in_band:
        chosen, cov = min(in_band, key=lambda x: abs(x[1] - 0.25))
        messages.append(f"otsu FG side by coverage~0.25 (cov={cov:.4f})")
        return chosen, messages

    under = [(m, c) for m, c in candidates if c <= _COV_HI]
    if under:
        chosen, cov = max(under, key=lambda x: x[1])
        messages.append(f"otsu FG side max cov<=0.90 (cov={cov:.4f})")
        return chosen, messages

    # Both sides > 0.90 — best-effort lower coverage
    chosen, cov = min(candidates, key=lambda x: x[1])
    messages.append(f"otsu FG side best-effort high cov (cov={cov:.4f})")
    return chosen, messages


def _keep_large_blob_union(mask: np.ndarray) -> tuple[np.ndarray, int, list[str]]:
    """Keep union of ALL components with area >= _LARGE_BLOB_FRAC (AI2 B).

    Never keeps only the single largest when multiple large exist.
    """
    from scipy import ndimage

    messages: list[str] = []
    if not mask.any():
        return mask, 0, messages

    label_out: tuple[np.ndarray, int] = ndimage.label(mask)  # type: ignore[assignment]
    labeled, n_components = label_out
    if n_components == 0:
        return mask, 0, messages

    area_min = max(1, int(mask.size * _LARGE_BLOB_FRAC))
    keep = np.zeros_like(mask, dtype=bool)
    count = 0
    for idx in range(1, n_components + 1):
        component = labeled == idx
        if int(component.sum()) >= area_min:
            keep |= component
            count += 1

    if count > 1:
        messages.append(
            f"multi_figure: kept union of {count} large blobs (area>={_LARGE_BLOB_FRAC:.0%} each)"
        )
    if count == 0:
        # No component met the large-blob floor — leave original for best-effort.
        return mask.astype(bool), 0, messages
    return keep, count, messages


def extract_silhouette_mask(
    rgba: np.ndarray,
    *,
    side: str,
) -> _MaskExtractResult:
    """Primary ``silhouette_mask`` + recovery cascade (0025 R2/R4).

    Cascade (each side always):
      0 primary → 1 alpha_matte if useful → if empty or coverage>0.90:
      3 corner-median → 4 Otsu luma → 5 large-blob union → 6 empty raises
      → 7 out-of-band best-effort + trust reason codes

    Raises ``silhouette_empty`` if still empty after cascade.
    """
    messages: list[str] = []
    reasons: list[str] = []
    recovery_ran = False
    bg_median: float | None = None
    bg_std: float | None = None

    # 0 Primary
    primary = silhouette_mask(rgba)
    mask = primary
    method: MaskMethod = "primary"

    # 1 Alpha prefer when useful
    if _alpha_useful(rgba):
        alpha_mask = _alpha_matte_mask(rgba)
        if alpha_mask.any():
            mask = alpha_mask
            method = "alpha_matte"
            messages.append("alpha matte preferred (useful transparency/variance)")

    cov = _coverage_frac(mask)
    need_recovery = (not mask.any()) or cov > _COV_HI

    if need_recovery:
        recovery_ran = True
        was_empty = not mask.any()

        # 3 Corner-median
        corner_mask, bg_median, bg_std, corner_msgs = _corner_median_mask(rgba)
        if was_empty:
            messages.extend(corner_msgs)
        else:
            messages.append(f"primary/alpha coverage {cov * 100.0:.1f}% high — recovery cascade")
            if bg_std is not None and bg_std > _BG_STD_UNCERTAIN:
                messages.append("background estimate uncertain — busy frame")
        mask = corner_mask
        method = "corner_median"
        cov = _coverage_frac(mask)

        # 4 Otsu if still out-of-band or empty
        if (not mask.any()) or (not _in_trust_band(cov)):
            luma_u8 = _luma(rgba).astype(np.uint8)
            hist, _ = np.histogram(luma_u8, bins=256, range=(0, 256))
            t, sigma_b2 = _otsu_threshold(hist)
            messages.append(f"otsu_luma threshold={t} sigma_b2={sigma_b2:.4f}")

            if sigma_b2 < _OTSU_MIN_SIGMA_B2:
                reasons.append("otsu_low_histogram_bimodality")
                messages.append(f"Otsu rejected: sigma_b2={sigma_b2:.4f} < {_OTSU_MIN_SIGMA_B2}")
                # Keep corner mask (may be empty/high); do not invent 50% FG
            else:
                otsu_mask, otsu_msgs = _otsu_fg_mask(
                    luma_u8,
                    t,
                    bg_median=bg_median,
                    bg_std=bg_std,
                )
                messages.extend(otsu_msgs)
                if otsu_mask.any():
                    mask = otsu_mask
                    method = "otsu_luma"
                    cov = _coverage_frac(mask)
                else:
                    messages.append("otsu FG empty — keeping prior recovery mask as best-effort")

        # 5 Large-blob keep on recovery mask
        if mask.any():
            mask, n_large, blob_msgs = _keep_large_blob_union(mask)
            messages.extend(blob_msgs)
            cov = _coverage_frac(mask)
            if n_large == 0 and not _in_trust_band(cov):
                # Noise-only recovery mask
                messages.append("no large blobs after recovery cleanup")

    # 6 Still empty → hard fail
    if not mask.any():
        reasons.append("recovery_failed")
        raise ProportionError(
            f"empty silhouette mask for {side}",
            code="silhouette_empty",
            details={"side": side, "trust_reasons": list(reasons)},
        )

    # 7 Final trust reasons for this side
    cov = _coverage_frac(mask)
    if cov > _COV_HI:
        reasons.append("coverage_high")
        if recovery_ran:
            reasons.append("coverage_high_after_recovery")
            reasons.append("recovery_failed")
        messages.append(
            f"Foreground coverage ({cov * 100.0:.1f}%) is unusually high "
            "— check background lighting or provide RGBA"
        )
    elif cov < _COV_LO:
        reasons.append("coverage_low")
        messages.append(
            f"Foreground coverage ({cov * 100.0:.1f}%) is unusually low "
            "— check background lighting or provide RGBA"
        )

    if (
        recovery_ran
        and bg_std is not None
        and bg_std > _BG_STD_UNCERTAIN
        and not _in_trust_band(cov)
    ):
        reasons.append("bg_uncertain")

    # Deduplicate reasons preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for r in reasons:
        if r in TRUST_REASON_CODES and r not in seen:
            seen.add(r)
            ordered.append(r)

    return _MaskExtractResult(
        mask=mask.astype(bool),
        messages=messages,
        method=method,
        trust_reasons=ordered,
        coverage_frac=cov,
        recovery_ran=recovery_ran,
    )


def _content_bbox(
    mask: np.ndarray,
    *,
    prefer_large_blobs: bool = False,
) -> tuple[int, int, int, int]:
    """Tight AABB of foreground with pad; optional large-blob union (F5).

    Returns ``(x0, y0, x1, y1)`` inclusive in original image pixels.
    """
    h, w = mask.shape
    y0 = x0 = y1 = x1 = 0
    used = False

    if prefer_large_blobs:
        _count, bboxes = connected_large_blobs(mask)
        if bboxes:
            # bboxes are (y0, x0, y1, x1)
            y0 = min(b[0] for b in bboxes)
            x0 = min(b[1] for b in bboxes)
            y1 = max(b[2] for b in bboxes)
            x1 = max(b[3] for b in bboxes)
            used = True

    if not used:
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

    pad = max(2, min(h, w) // 50)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h - 1, y1 + pad)
    x1 = min(w - 1, x1 + pad)
    return x0, y0, x1, y1


def _crop_and_resize_mask(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    grid_px: int = GRID_PX,
) -> np.ndarray:
    """Crop mask to bbox, nearest-resize to grid_px^2, re-binarize."""
    x0, y0, x1, y1 = bbox
    crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return np.zeros((grid_px, grid_px), dtype=bool)

    Image = _require_pillow()
    # PIL expects HxW mode "L"
    arr = crop.astype(np.uint8) * 255
    im = Image.fromarray(arr, mode="L")
    resized = im.resize((grid_px, grid_px), Image.Resampling.NEAREST)
    out = np.asarray(resized, dtype=np.uint8) > 0
    return out


def _iou_dice(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int, int, int]:
    """Return (iou, dice, a_fg, b_fg, intersection, union)."""
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    inter = int(np.logical_and(a_b, b_b).sum())
    a_fg = int(a_b.sum())
    b_fg = int(b_b.sum())
    union = a_fg + b_fg - inter
    iou = float(inter) / float(union) if union > 0 else 0.0
    denom = a_fg + b_fg
    dice = (2.0 * float(inter) / float(denom)) if denom > 0 else 0.0
    return iou, dice, a_fg, b_fg, inter, union


def _aspect_ratio(bbox: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    h = max(1, y1 - y0 + 1)
    w = max(1, x1 - x0 + 1)
    return float(max(h, w)) / float(max(1, min(h, w)))


def _build_overlay(ref_grid: np.ndarray, mesh_grid: np.ndarray) -> Any:
    """256x256 overlay: ref red, mesh cyan, intersection yellow (R6)."""
    Image = _require_pillow()
    g = GRID_PX
    canvas = np.zeros((g, g, 4), dtype=np.uint8)
    # dark bg
    canvas[:, :, 3] = 255
    canvas[:, :, 0:3] = 20

    ref_only = ref_grid & ~mesh_grid
    mesh_only = mesh_grid & ~ref_grid
    both = ref_grid & mesh_grid

    # ref red alpha ~120 (R6)
    canvas[ref_only, 0] = 220
    canvas[ref_only, 1] = 40
    canvas[ref_only, 2] = 40
    canvas[ref_only, 3] = 120

    # mesh cyan alpha ~120 (R6)
    canvas[mesh_only, 0] = 40
    canvas[mesh_only, 1] = 200
    canvas[mesh_only, 2] = 220
    canvas[mesh_only, 3] = 120

    # intersection yellow/white (slightly more opaque for glance)
    canvas[both, 0] = 240
    canvas[both, 1] = 230
    canvas[both, 2] = 60
    canvas[both, 3] = 180

    return Image.fromarray(canvas, mode="RGBA")


def _render_mesh_front(mesh_path: Path) -> Path:
    """Render mesh front view via F3D; return path to front.png (R5 / B2).

    Forces white background so ``frame.silhouette_mask`` (near-white >= 250)
    classifies the backdrop as background, not full-frame foreground.

    Caller must delete the returned path when finished (temp copy).
    """
    from meshops.render.f3d_renderer import F3DRenderer, RenderUnavailableError

    with tempfile.TemporaryDirectory(prefix="meshops_sil_") as td:
        tmp = Path(td)
        try:
            renderer = F3DRenderer()
            result = renderer.render_mesh_to_dir(
                mesh_path,
                tmp,
                camera_names=("front",),
                include_depth_for=(),
                background_color=(1.0, 1.0, 1.0),
            )
        except RenderUnavailableError as exc:
            raise ProportionError(
                f"mesh front render unavailable: {exc}",
                code="silhouette_failed",
                details={
                    "code": "render_unavailable",
                    "mesh": str(mesh_path),
                    "cause": str(exc),
                },
            ) from exc

        front: Path | None = None
        for p in result.view_paths:
            cand = Path(p)
            if cand.stem.lower() == "front" and cand.is_file():
                front = cand
                break
        if front is None:
            cand = tmp / "front.png"
            if cand.is_file():
                front = cand
        if front is None or not front.is_file():
            raise ProportionError(
                "mesh front render produced no front.png",
                code="silhouette_failed",
                details={"code": "render_unavailable", "mesh": str(mesh_path)},
            )
        # Copy out of TemporaryDirectory before it is removed.
        out_fd, out_name = tempfile.mkstemp(prefix="meshops_sil_front_", suffix=".png")
        os.close(out_fd)
        out_path = Path(out_name)
        try:
            shutil.copy2(front, out_path)
        except OSError as exc:
            out_path.unlink(missing_ok=True)
            raise ProportionError(
                f"failed to stage mesh front render: {exc}",
                code="silhouette_failed",
                details={"code": "render_unavailable", "mesh": str(mesh_path)},
            ) from exc
        return out_path


def _normalize_view_role(view_role: str) -> None:
    """Accept only case-insensitive 'front' (C5)."""
    if view_role.strip().lower() != "front":
        raise ProportionError(
            f"--view-role must be 'front' (front-only silhouette law); got {view_role!r}",
            code="silhouette_failed",
            details={"view_role": view_role},
        )


def _aggregate_trust(
    ref: _MaskExtractResult,
    mesh: _MaskExtractResult,
) -> tuple[bool, list[str]]:
    """Combine per-side outcomes into package silhouette_trusted + trust_reasons.

    Trusted iff both sides final coverage in [0.02, 0.90] (R1). When trusted,
    trust_reasons is always empty.
    """
    ref_ok = _in_trust_band(ref.coverage_frac)
    mesh_ok = _in_trust_band(mesh.coverage_frac)
    if ref_ok and mesh_ok:
        return True, []

    reasons: list[str] = []
    for ext in (ref, mesh):
        for r in ext.trust_reasons:
            if r in TRUST_REASON_CODES and r not in reasons:
                reasons.append(r)
    if not ref_ok and "ref_untrusted" not in reasons:
        reasons.append("ref_untrusted")
    if not mesh_ok and "mesh_untrusted" not in reasons:
        reasons.append("mesh_untrusted")
    if not reasons:
        reasons = ["recovery_failed"]
    return False, reasons


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def run_silhouette_compare(
    ref: Path | str,
    out: Path | str,
    *,
    mesh: Path | str | None = None,
    mesh_view: Path | str | None = None,
    view_role: str = "front",
    overlay: bool = True,
    force: bool = False,
    require_trusted: bool = False,
) -> dict[str, Any]:
    """Compare front silhouettes: Package A ref vs mesh front view.

    Returns CLI/MCP success payload with ok/paths/counts/score_iou/score_dice/
    silhouette_trusted/trust_reasons/messages.
    Raises :class:`ProportionError` on failure.
    When ``require_trusted`` and result is untrusted, raises
    ``ProportionError(code="silhouette_untrusted")``.
    """
    _require_pillow()
    _normalize_view_role(view_role)

    mesh_p = Path(mesh) if mesh is not None else None
    mesh_view_p = Path(mesh_view) if mesh_view is not None else None
    ref_p = Path(ref)

    if mesh_p is not None and mesh_view_p is not None:
        raise ProportionError(
            "pass only one of --mesh or --mesh-view",
            code="silhouette_failed",
            details={"mesh": str(mesh_p), "mesh_view": str(mesh_view_p)},
        )
    if mesh_p is None and mesh_view_p is None:
        raise ProportionError(
            "exactly one of --mesh or --mesh-view is required",
            code="silhouette_failed",
            details={},
        )

    if not ref_p.is_file():
        raise ProportionError(
            f"ref image not found: {ref_p}",
            code="silhouette_failed",
            details={"ref": str(ref_p)},
        )

    messages: list[str] = []

    # Basename advisories (A3 / D3) — soft
    if _stem_looks_non_front(ref_p):
        messages.append(_BASENAME_ADVISORY.format(side="ref"))
    if mesh_view_p is not None and _stem_looks_non_front(mesh_view_p):
        messages.append(_BASENAME_ADVISORY.format(side="mesh-view"))

    # Resolve mesh-view path (render if needed)
    rendered_mesh_view: Path | None = None
    if mesh_p is not None:
        if not mesh_p.is_file():
            raise ProportionError(
                f"mesh not found: {mesh_p}",
                code="silhouette_failed",
                details={"mesh": str(mesh_p)},
            )
        rendered_mesh_view = _render_mesh_front(mesh_p)
        mesh_view_for_mask = rendered_mesh_view
    else:
        assert mesh_view_p is not None
        if not mesh_view_p.is_file():
            raise ProportionError(
                f"mesh-view image not found: {mesh_view_p}",
                code="silhouette_failed",
                details={"mesh_view": str(mesh_view_p)},
            )
        mesh_view_for_mask = mesh_view_p

    # Identical inputs (F2)
    identical = False
    if mesh_view_p is not None:
        try:
            same_path = ref_p.resolve() == mesh_view_p.resolve()
        except OSError:
            same_path = False
        if same_path:
            identical = True
        else:
            try:
                if _sha256_file(ref_p) == _sha256_file(mesh_view_p):
                    identical = True
            except OSError:
                pass
    if identical:
        messages.append(_IDENTICAL_MSG)

    # Load + mask (clean staged --mesh render after load)
    try:
        ref_rgba = _load_rgba_array(ref_p)
        mesh_rgba = _load_rgba_array(mesh_view_for_mask)
    except ProportionError:
        if rendered_mesh_view is not None:
            rendered_mesh_view.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if rendered_mesh_view is not None:
            rendered_mesh_view.unlink(missing_ok=True)
        raise ProportionError(
            f"failed to load silhouette images: {exc}",
            code="silhouette_failed",
            details={"ref": str(ref_p), "mesh_view": str(mesh_view_for_mask)},
        ) from exc
    finally:
        if rendered_mesh_view is not None:
            rendered_mesh_view.unlink(missing_ok=True)

    # Full cascade both sides (C6)
    ref_ext = extract_silhouette_mask(ref_rgba, side="ref")
    messages.extend(ref_ext.messages)
    mesh_ext = extract_silhouette_mask(mesh_rgba, side="mesh_view")
    messages.extend(mesh_ext.messages)

    ref_mask = ref_ext.mask
    mesh_mask = mesh_ext.mask

    trusted, trust_reasons = _aggregate_trust(ref_ext, mesh_ext)

    # Multi-figure on ref (F5)
    large_count, _ = connected_large_blobs(ref_mask)
    if large_count > 1:
        messages.append(
            "ref has multiple figures; score compares mesh against union bbox of all ref figures"
        )

    ref_bbox = _content_bbox(ref_mask, prefer_large_blobs=True)
    mesh_bbox = _content_bbox(mesh_mask, prefer_large_blobs=False)

    # Aspect warn (C4)
    ref_ar = _aspect_ratio(ref_bbox)
    mesh_ar = _aspect_ratio(mesh_bbox)
    ar_ratio = max(ref_ar, mesh_ar) / max(1e-9, min(ref_ar, mesh_ar))
    if ar_ratio > 2.0:
        messages.append("aspect mismatch >2x — scores may be weak")

    ref_grid = _crop_and_resize_mask(ref_mask, ref_bbox)
    mesh_grid = _crop_and_resize_mask(mesh_mask, mesh_bbox)

    iou, dice, ref_fg, mesh_fg, inter, union = _iou_dice(ref_grid, mesh_grid)

    json_path, overlay_path = _resolve_out_paths(out)

    overlay_written: str | None = None
    if overlay:
        img = _build_overlay(ref_grid, mesh_grid)
        _write_png(overlay_path, img, force=force)
        overlay_written = str(overlay_path)

    package = SilhouetteComparePackage(
        schema_version=SILHOUETTE_SCHEMA_VERSION,
        honesty=SILHOUETTE_HONESTY,
        view_role="front",
        ref_path=str(ref_p),
        mesh_path=str(mesh_p) if mesh_p is not None else None,
        mesh_view_path=str(mesh_view_p) if mesh_view_p is not None else None,
        method=METHOD,
        grid_px=GRID_PX,
        scores=SilhouetteScores(iou=iou, dice=dice),
        counts=SilhouetteCounts(
            ref_fg_grid_px=ref_fg,
            mesh_fg_grid_px=mesh_fg,
            intersection_grid_px=inter,
            union_grid_px=union,
            ref_fg_orig_px=int(ref_mask.sum()),
            mesh_fg_orig_px=int(mesh_mask.sum()),
        ),
        alignment=SilhouetteAlignment(
            mode=ALIGNMENT_MODE,
            rotation=False,
            resize=RESIZE_LABEL,
            ref_bbox=[ref_bbox[0], ref_bbox[1], ref_bbox[2], ref_bbox[3]],
            mesh_bbox=[mesh_bbox[0], mesh_bbox[1], mesh_bbox[2], mesh_bbox[3]],
        ),
        messages=messages,
        overlay_path=overlay_written,
        silhouette_trusted=trusted,
        trust_reasons=trust_reasons,
        mask_method_ref=ref_ext.method,
        mask_method_mesh=mesh_ext.method,
        ref_coverage_frac=ref_ext.coverage_frac,
        mesh_coverage_frac=mesh_ext.coverage_frac,
    )

    payload_pkg = package.model_dump(mode="json")
    _write_json(json_path, payload_pkg, force=force)

    paths = [str(json_path)]
    if overlay_written is not None:
        paths.append(overlay_written)

    payload: dict[str, Any] = {
        "ok": True,
        "paths": paths,
        "counts": {
            "ref_fg_grid_px": ref_fg,
            "mesh_fg_grid_px": mesh_fg,
        },
        "score_iou": iou,
        "score_dice": dice,
        "messages": messages,
        "silhouette_trusted": trusted,
        "trust_reasons": trust_reasons,
        "ref_coverage_frac": ref_ext.coverage_frac,
        "mesh_coverage_frac": mesh_ext.coverage_frac,
        "mask_method_ref": ref_ext.method,
        "mask_method_mesh": mesh_ext.method,
    }

    if require_trusted and not trusted:
        raise ProportionError(
            "silhouette compare result is untrusted "
            f"(reasons={','.join(trust_reasons) or 'unknown'})",
            code="silhouette_untrusted",
            details={
                "trust_reasons": trust_reasons,
                "score_iou": iou,
                "score_dice": dice,
                "ref_coverage_frac": ref_ext.coverage_frac,
                "mesh_coverage_frac": mesh_ext.coverage_frac,
                "mask_method_ref": ref_ext.method,
                "mask_method_mesh": mesh_ext.method,
                "paths": paths,
            },
        )

    return payload


__all__ = [
    "GRID_PX",
    "LUMA_THR",
    "METHOD",
    "SILHOUETTE_JSON_BASENAME",
    "SILHOUETTE_OVERLAY_BASENAME",
    "SILHOUETTE_SCHEMA_VERSION",
    "TRUST_REASON_CODES",
    "MaskMethod",
    "SilhouetteAlignment",
    "SilhouetteComparePackage",
    "SilhouetteCounts",
    "SilhouetteScores",
    "extract_silhouette_mask",
    "run_silhouette_compare",
]
