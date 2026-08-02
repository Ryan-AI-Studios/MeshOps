"""Discover canonical multi-view images and read pixel sizes.

PNG size without Pillow: robust IHDR chunk scan (not fixed offset 16).
JPG / JPEG / WebP require Pillow (meshops[proportion]).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from meshops.proportion.errors import ProportionError
from meshops.proportion.models import CANONICAL_VIEW_KEYS, IMAGE_EXTENSIONS

# PNG signature + chunk layout
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_IHDR = b"IHDR"


@dataclass(frozen=True, slots=True)
class ViewImage:
    """One discovered view file with pixel dimensions."""

    view: str
    path: Path
    width_px: int
    height_px: int


def _pillow_available() -> bool:
    try:
        import PIL  # noqa: F401  # type: ignore[import-untyped,import-not-found]

        return True
    except ImportError:
        return False


def png_size_from_bytes(data: bytes) -> tuple[int, int]:
    """Return (width, height) by scanning for the IHDR chunk.

    Does not assume IHDR starts at a fixed byte offset after the signature.
    """
    if len(data) < 24 or not data.startswith(_PNG_SIG):
        raise ProportionError(
            "not a PNG (missing signature)",
            code="unreadable_image",
        )
    # Walk chunks: after signature, each chunk is len(4) + type(4) + data + crc(4)
    offset = 8
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        if data_end + 4 > len(data):
            break
        if chunk_type == _IHDR:
            if length < 8:
                raise ProportionError(
                    "PNG IHDR chunk too short",
                    code="unreadable_image",
                )
            width, height = struct.unpack(">II", data[data_start : data_start + 8])
            if width <= 0 or height <= 0:
                raise ProportionError(
                    f"invalid PNG dimensions {width}x{height}",
                    code="unreadable_image",
                )
            return int(width), int(height)
        offset = data_end + 4  # skip CRC
    raise ProportionError(
        "PNG has no IHDR chunk",
        code="unreadable_image",
    )


def png_size(path: Path) -> tuple[int, int]:
    """Read PNG dimensions via robust IHDR scan (stdlib only)."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ProportionError(
            f"cannot read image: {path}: {exc}",
            code="unreadable_image",
            details={"path": str(path)},
        ) from exc
    try:
        return png_size_from_bytes(data)
    except ProportionError as exc:
        raise ProportionError(
            f"{exc} ({path})",
            code=exc.code,
            details={"path": str(path)},
        ) from exc


def image_size(path: Path, *, allow_pillow: bool = True) -> tuple[int, int]:
    """Return (width, height) for a view image.

    PNG: always via IHDR scan (no Pillow required).
    Other formats: require Pillow when allow_pillow is True.
    """
    suffix = path.suffix.lower()
    if suffix == ".png":
        # Prefer pure IHDR path so PNG works offline without the extra.
        return png_size(path)

    if suffix not in IMAGE_EXTENSIONS:
        raise ProportionError(
            f"unsupported image extension {suffix!r} for {path.name}",
            code="unreadable_image",
            details={"path": str(path)},
        )

    if not allow_pillow or not _pillow_available():
        raise ProportionError(
            f"JPG/WebP require Pillow (install meshops[proportion] / pillow). "
            f"Cannot size {path.name} without Pillow.",
            code="pillow_required",
            details={"path": str(path), "hint": "uv sync --extra proportion"},
        )

    try:
        from PIL import Image  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:  # pragma: no cover - guarded above
        raise ProportionError(
            f"Pillow required for {path.name}",
            code="pillow_required",
            details={"path": str(path)},
        ) from exc

    try:
        with Image.open(path) as img:
            w, h = img.size
    except OSError as exc:
        raise ProportionError(
            f"cannot open image {path}: {exc}",
            code="unreadable_image",
            details={"path": str(path)},
        ) from exc

    if w <= 0 or h <= 0:
        raise ProportionError(
            f"invalid image dimensions {w}x{h} for {path}",
            code="unreadable_image",
            details={"path": str(path)},
        )
    return int(w), int(h)


def find_view_path(views_dir: Path, view: str) -> Path | None:
    """Return first existing file for canonical basename + supported extension."""
    for ext in IMAGE_EXTENSIONS:
        candidate = views_dir / f"{view}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_views(
    views_dir: Path | str,
    *,
    allow_pillow: bool = True,
    required: tuple[str, ...] | None = None,
    partial_ok: bool = False,
) -> dict[str, ViewImage]:
    """Scan *views_dir* for canonical multi-view basenames.

    Raises ``missing_views`` when a required key is absent and partial_ok is False.
    """
    root = Path(views_dir)
    if not root.is_dir():
        raise ProportionError(
            f"views directory not found: {root}",
            code="missing_views",
            details={"path": str(root)},
        )

    req = required if required is not None else ("front", "left", "three_quarter")
    found: dict[str, ViewImage] = {}
    missing: list[str] = []

    for key in CANONICAL_VIEW_KEYS:
        path = find_view_path(root, key)
        if path is None:
            if key in req:
                missing.append(key)
            continue
        w, h = image_size(path, allow_pillow=allow_pillow)
        found[key] = ViewImage(view=key, path=path, width_px=w, height_px=h)

    if missing and not partial_ok:
        raise ProportionError(
            f"missing required views: {', '.join(missing)} under {root}",
            code="missing_views",
            details={"missing": missing, "path": str(root)},
        )

    if "front" not in found and not partial_ok:
        raise ProportionError(
            f"missing required view: front under {root}",
            code="missing_views",
            details={"missing": ["front"], "path": str(root)},
        )

    return found
