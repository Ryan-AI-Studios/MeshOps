"""Heuristic frame: subject bbox / figure span / multi-blob count.

Never invents joint landmarks — bbox and span only (method=heuristic_frame).
Uses scipy.ndimage for connected components when available (core dep).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from meshops.proportion.errors import ProportionError
from meshops.proportion.load_views import ViewImage
from meshops.proportion.models import SubjectBBox, ViewLandmarks

# Minimum blob area as fraction of image area to count as "large"
_LARGE_BLOB_FRAC = 0.02
# Background: pure white / near-white treated as empty for synthetic fixtures
_BG_THRESHOLD = 250


def _load_rgba_array(path: Path) -> np.ndarray:
    """Load image as HxWx4 uint8. PNG via pure path when possible; else Pillow."""
    suffix = path.suffix.lower()
    if suffix == ".png":
        try:
            return _png_to_rgba(path)
        except Exception:
            pass  # fall through to Pillow

    try:
        from PIL import Image  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:
        raise ProportionError(
            f"cannot load frame pixels for {path.name}; need valid PNG or Pillow",
            code="unreadable_image",
            details={"path": str(path)},
        ) from exc

    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        return np.asarray(rgba, dtype=np.uint8)


def _png_to_rgba(path: Path) -> np.ndarray:
    """Decode common PNG types via Pillow if present, else simple RGBA8 path.

    For offline tests we primarily generate RGBA8 or RGB8 PNGs. Prefer Pillow
    when installed; otherwise use a minimal zlib decode for 8-bit RGB/RGBA only.
    """
    try:
        from PIL import Image  # type: ignore[import-untyped,import-not-found]

        with Image.open(path) as img:
            return np.asarray(img.convert("RGBA"), dtype=np.uint8)
    except ImportError:
        return _png_stdlib_rgba(path)


def _png_stdlib_rgba(path: Path) -> np.ndarray:
    """Minimal PNG decoder for 8-bit RGB/RGBA (no filter complexity beyond basic)."""
    import struct
    import zlib

    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ProportionError("not PNG", code="unreadable_image")

    offset = 8
    width = height = None
    bit_depth = color_type = None
    idat = bytearray()

    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        ctype = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset = offset + 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[:10])
        elif ctype == b"IDAT":
            idat.extend(payload)
        elif ctype == b"IEND":
            break

    if width is None or height is None or bit_depth != 8:
        raise ProportionError(
            "stdlib PNG frame decode supports 8-bit RGB/RGBA only",
            code="unreadable_image",
        )
    if color_type not in (2, 6):  # RGB, RGBA
        raise ProportionError(
            f"stdlib PNG frame decode unsupported color_type={color_type}",
            code="unreadable_image",
        )

    raw = zlib.decompress(bytes(idat))
    channels = 3 if color_type == 2 else 4
    stride = 1 + width * channels
    rows = []
    for y in range(height):
        row = raw[y * stride : (y + 1) * stride]
        filt = row[0]
        scan = bytearray(row[1:])
        if filt == 1:  # Sub
            for i in range(channels, len(scan)):
                scan[i] = (scan[i] + scan[i - channels]) & 0xFF
        elif filt == 2:  # Up
            if y > 0:
                prev = rows[y - 1]
                for i in range(len(scan)):
                    scan[i] = (scan[i] + prev[i]) & 0xFF
        elif filt == 0:
            pass
        else:
            # Paeth / Average — fall back by requiring filter 0/1/2 only
            raise ProportionError(
                f"stdlib PNG unsupported filter type {filt}",
                code="unreadable_image",
            )
        rows.append(bytes(scan))

    arr = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, channels)
    if channels == 3:
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)
    return arr


def silhouette_mask(rgba: np.ndarray) -> np.ndarray:
    """Boolean mask of non-background pixels (non-white / non-transparent)."""
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError("expected HxWxC image")
    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3] if rgba.shape[2] >= 4 else np.full(rgb.shape[:2], 255, dtype=np.uint8)
    near_white = (
        (rgb[:, :, 0] >= _BG_THRESHOLD)
        & (rgb[:, :, 1] >= _BG_THRESHOLD)
        & (rgb[:, :, 2] >= _BG_THRESHOLD)
    )
    opaque = alpha > 16
    return opaque & ~near_white


def connected_large_blobs(mask: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Return (count of large components, list of bboxes y0,x0,y1,x1)."""
    from scipy import ndimage

    if not mask.any():
        return 0, []

    # scipy.ndimage.label stubs are overloaded; cast to concrete tuple form.
    label_out: tuple[np.ndarray, int] = ndimage.label(mask)  # type: ignore[assignment]
    labeled, n_components = label_out
    if n_components == 0:
        return 0, []

    area_min = max(1, int(mask.size * _LARGE_BLOB_FRAC))
    bboxes: list[tuple[int, int, int, int]] = []
    large = 0
    for idx in range(1, n_components + 1):
        ys, xs = np.where(labeled == idx)
        if ys.size < area_min:
            continue
        large += 1
        bboxes.append((int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())))
    return large, bboxes


def frame_from_image(path: Path) -> dict[str, Any]:
    """Compute bbox, figure_span_px, large_blob_count for one image."""
    rgba = _load_rgba_array(path)
    mask = silhouette_mask(rgba)
    h, w = mask.shape
    if not mask.any():
        return {
            "subject_bbox": None,
            "figure_span_px": None,
            "large_blob_count": 0,
            "width_px": w,
            "height_px": h,
        }

    large_count, bboxes = connected_large_blobs(mask)
    # Union bbox of all large blobs (or whole mask if none pass threshold)
    if bboxes:
        y0 = min(b[0] for b in bboxes)
        x0 = min(b[1] for b in bboxes)
        y1 = max(b[2] for b in bboxes)
        x1 = max(b[3] for b in bboxes)
    else:
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        large_count = 1

    span = float(y1 - y0)
    return {
        "subject_bbox": SubjectBBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)),
        "figure_span_px": span if span > 0 else None,
        "large_blob_count": large_count,
        "width_px": w,
        "height_px": h,
    }


def apply_heuristic_frame(
    view_images: dict[str, ViewImage],
    views: dict[str, ViewLandmarks] | None = None,
) -> tuple[dict[str, ViewLandmarks], bool, list[str]]:
    """Fill bbox/span/blob counts. Never invents joint landmarks.

    Returns (views, multi_figure_detected, notes).
    """
    notes: list[str] = []
    result = dict(views or {})
    multi = False

    for key, img in view_images.items():
        try:
            frame = frame_from_image(img.path)
        except ProportionError as exc:
            notes.append(f"heuristic_frame skip {key}: {exc}")
            if key not in result:
                result[key] = ViewLandmarks(
                    view=key,
                    width_px=img.width_px,
                    height_px=img.height_px,
                    path=str(img.path),
                )
            continue

        if key not in result:
            result[key] = ViewLandmarks(
                view=key,
                width_px=img.width_px,
                height_px=img.height_px,
                path=str(img.path),
            )
        vl = result[key]
        if vl.subject_bbox is None and frame["subject_bbox"] is not None:
            vl.subject_bbox = frame["subject_bbox"]
        if vl.figure_span_px is None and frame["figure_span_px"] is not None:
            # Prefer assist crown/sole span when landmarks later set it; heuristic is fallback
            vl.figure_span_px = frame["figure_span_px"]
        vl.large_blob_count = frame["large_blob_count"]
        if (frame["large_blob_count"] or 0) >= 2:
            multi = True
            notes.append(
                f"view {key}: {frame['large_blob_count']} large silhouette blobs → multi_figure"
            )

    return result, multi, notes


def figure_span_from_landmarks(vl: ViewLandmarks) -> float | None:
    """Vertical figure span from sole and stature top when landmarks present."""
    lm = vl.landmarks
    sole = lm.get("sole")
    top = lm.get("cranial_vertex") or lm.get("hair_crown")
    if sole is None or top is None:
        return None
    span = float(sole.y_px - top.y_px)
    return span if span > 0 else None
