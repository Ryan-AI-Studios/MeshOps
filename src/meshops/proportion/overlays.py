"""Draw landmark overlays on view images (lazy Pillow import)."""

from __future__ import annotations

from pathlib import Path

from meshops.proportion.errors import ProportionError
from meshops.proportion.models import ProportionReport, ViewLandmarks


def draw_view_overlay(
    image_path: Path | str,
    view: ViewLandmarks,
    out_path: Path | str,
) -> Path:
    """Draw landmark dots + ids onto a copy of the view image."""
    try:
        from PIL import Image, ImageDraw  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:
        raise ProportionError(
            "overlays require Pillow (install meshops[proportion] / pillow)",
            code="pillow_required",
            details={"hint": "uv sync --extra proportion"},
        ) from exc

    src = Path(image_path)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        canvas = im.convert("RGBA").copy()
    draw = ImageDraw.Draw(canvas)

    r = max(3, min(canvas.size) // 120)
    for lid, lm in view.landmarks.items():
        x, y = float(lm.x_px), float(lm.y_px)
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=(255, 64, 64, 220),
            outline=(0, 0, 0, 255),
        )
        draw.text((x + r + 2, y - r), lid, fill=(20, 20, 20, 255))

    if view.subject_bbox is not None:
        bb = view.subject_bbox
        draw.rectangle(
            (bb.x0, bb.y0, bb.x1, bb.y1),
            outline=(0, 180, 255, 200),
            width=2,
        )

    canvas.save(dest)
    return dest


def write_overlays(
    report: ProportionReport,
    views_dir: Path | str,
    out_dir: Path | str,
) -> list[Path]:
    """Write overlay PNGs for each view that has a source path."""
    root = Path(views_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for key, vl in report.views.items():
        src: Path | None = Path(vl.path) if vl.path else None
        if src is None or not src.is_file():
            # try resolve from views_dir
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                cand = root / f"{key}{ext}"
                if cand.is_file():
                    src = cand
                    break
        if src is None or not src.is_file():
            continue
        dest = out / f"{key}_overlay.png"
        written.append(draw_view_overlay(src, vl, dest))
    return written
