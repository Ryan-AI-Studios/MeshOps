"""Front-only binary silhouette IoU/Dice compare (track 0021).

Authoring QA score between Package A front ref and mesh front view.
Not mesh or print success (Difficulty §12 / N6). Front-vs-front only.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.errors import ProportionError
from meshops.proportion.frame import (
    _load_rgba_array,
    connected_large_blobs,
    silhouette_mask,
)
from meshops.proportion.honesty import SILHOUETTE_HONESTY

SILHOUETTE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
SILHOUETTE_JSON_BASENAME: Final[str] = "silhouette_compare.json"
SILHOUETTE_OVERLAY_BASENAME: Final[str] = "silhouette_overlay.png"
GRID_PX: Final[int] = 256
LUMA_THR: Final[int] = 18
METHOD: Final[Literal["binary_mask_iou_dice"]] = "binary_mask_iou_dice"
ALIGNMENT_MODE: Final[Literal["content_bbox"]] = "content_bbox"
RESIZE_LABEL: Final[Literal["nearest_256"]] = "nearest_256"

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
    """IoU + Dice on the 256² aligned grid."""

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
    """silhouette_compare.json package (schema 1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SILHOUETTE_SCHEMA_VERSION
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


def _corner_median_mask(rgba: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Fallback FG mask via corner-median luma (C1). Only when primary empty."""
    messages: list[str] = []
    luma = _luma(rgba)
    bg, std, _corners = _corner_median_luma(luma)
    mask = np.abs(luma - bg) > float(LUMA_THR)
    messages.append("primary silhouette_mask empty — used corner-median fallback")
    if std > 25.0:
        messages.append("background estimate uncertain — busy frame")
    return mask, messages


def extract_silhouette_mask(
    rgba: np.ndarray,
    *,
    side: str,
) -> tuple[np.ndarray, list[str]]:
    """Primary ``silhouette_mask`` + corner-median fallback (B1 / R3).

    Raises ``silhouette_empty`` if still empty after fallback.
    """
    messages: list[str] = []
    mask = silhouette_mask(rgba)
    if not mask.any():
        mask, fb_msgs = _corner_median_mask(rgba)
        messages.extend(fb_msgs)

    if not mask.any():
        raise ProportionError(
            f"empty silhouette mask for {side}",
            code="silhouette_empty",
            details={"side": side},
        )

    h, w = mask.shape
    coverage = float(mask.sum()) / float(max(h * w, 1))
    if coverage < 0.02 or coverage > 0.90:
        kind = "low" if coverage < 0.02 else "high"
        messages.append(
            f"Foreground coverage ({coverage * 100.0:.1f}%) is unusually {kind} "
            "— check background lighting or provide RGBA"
        )
    return mask, messages


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
    """Crop mask to bbox, nearest-resize to grid_px², re-binarize."""
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

    Forces white background so ``frame.silhouette_mask`` (near-white ≥ 250)
    classifies the backdrop as background, not full-frame foreground.
    """
    from meshops.render.f3d_renderer import F3DRenderer, RenderUnavailableError

    tmp = Path(tempfile.mkdtemp(prefix="meshops_sil_"))
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

    # Prefer explicit front.png from result
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
    return front


def _normalize_view_role(view_role: str) -> None:
    """Accept only case-insensitive 'front' (C5)."""
    if view_role.strip().lower() != "front":
        raise ProportionError(
            f"--view-role must be 'front' (front-only silhouette law); got {view_role!r}",
            code="silhouette_failed",
            details={"view_role": view_role},
        )


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
) -> dict[str, Any]:
    """Compare front silhouettes: Package A ref vs mesh front view.

    Returns CLI/MCP success payload with ok/paths/counts/score_iou/score_dice/messages.
    Raises :class:`ProportionError` on failure.
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

    # Load + mask
    try:
        ref_rgba = _load_rgba_array(ref_p)
        mesh_rgba = _load_rgba_array(mesh_view_for_mask)
    except ProportionError:
        raise
    except Exception as exc:
        raise ProportionError(
            f"failed to load silhouette images: {exc}",
            code="silhouette_failed",
            details={"ref": str(ref_p), "mesh_view": str(mesh_view_for_mask)},
        ) from exc

    ref_mask, ref_msgs = extract_silhouette_mask(ref_rgba, side="ref")
    messages.extend(ref_msgs)
    mesh_mask, mesh_msgs = extract_silhouette_mask(mesh_rgba, side="mesh_view")
    messages.extend(mesh_msgs)

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
    )

    payload_pkg = package.model_dump(mode="json")
    _write_json(json_path, payload_pkg, force=force)

    paths = [str(json_path)]
    if overlay_written is not None:
        paths.append(overlay_written)

    return {
        "ok": True,
        "paths": paths,
        "counts": {
            "ref_fg_grid_px": ref_fg,
            "mesh_fg_grid_px": mesh_fg,
        },
        "score_iou": iou,
        "score_dice": dice,
        "messages": messages,
    }


__all__ = [
    "GRID_PX",
    "LUMA_THR",
    "METHOD",
    "SILHOUETTE_JSON_BASENAME",
    "SILHOUETTE_OVERLAY_BASENAME",
    "SILHOUETTE_SCHEMA_VERSION",
    "SilhouetteAlignment",
    "SilhouetteComparePackage",
    "SilhouetteCounts",
    "SilhouetteScores",
    "extract_silhouette_mask",
    "run_silhouette_compare",
]
