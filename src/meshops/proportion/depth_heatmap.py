"""Depth sample / delta heatmap glance PNG + meta (track 0020).

Authoring visualization only — numbers in depth_at_landmarks.json remain source
of truth. Not mesh or print success (Difficulty §12 / N6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from meshops.proportion.depth_samples import (
    AXIS_NOTES as SAMPLES_AXIS_NOTES,
)
from meshops.proportion.depth_samples import (
    DepthSample,
    DepthSamplesPackage,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import HEATMAP_HONESTY

HEATMAP_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
HEATMAP_PNG_BASENAME: Final[str] = "depth_heatmap.png"
HEATMAP_JSON_BASENAME: Final[str] = "depth_heatmap.json"

# Dual-panel geometry freeze (F3).
PANEL_GAP_PX: Final[int] = 12
DEFAULT_PANEL_W: Final[int] = 640
DEFAULT_PANEL_H: Final[int] = 400
COLORBAR_H: Final[int] = 16
MARGIN: Final[int] = 40
FOOTER_H: Final[int] = 28

AXIS_NOTES: Final[str] = (
    f"{SAMPLES_AXIS_NOTES}; heatmap plot: front (+Y) = RIGHT, "
    "vertical = body-up z_frac (0 sole → 1 crown)"
)

_COLOR_NOTE: Final[str] = "relative glance only; numbers in depth_at_landmarks are SoT"

ColorScalePanel = Literal["samples", "deltas"]
ColorScaleMode = Literal["y_m", "depth_m", "delta_y_m", "delta_depth_m"]


class ColorScale(BaseModel):
    """One color-bar description per panel."""

    model_config = ConfigDict(extra="forbid")

    panel: ColorScalePanel
    mode: ColorScaleMode
    vmin: float | None = None
    vmax: float | None = None
    note: str = _COLOR_NOTE


class DepthHeatmapPackage(BaseModel):
    """depth_heatmap.json meta package (schema 1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = HEATMAP_SCHEMA_VERSION
    honesty: str = HEATMAP_HONESTY
    source_samples: str | None = None
    source_deltas: str | None = None
    underlay: str | None = None
    axis_notes: str = AXIS_NOTES
    color_scales: list[ColorScale] = Field(default_factory=list)
    plotted_sample_ids: list[str] = Field(default_factory=list)
    plotted_delta_ids: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_depth_samples_package(path: Path | str) -> DepthSamplesPackage:
    """Load and validate depth_at_landmarks.json."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot load depth samples: {p}: {exc}",
            code="invalid_depth_samples",
            details={"path": str(p)},
        ) from exc
    try:
        return DepthSamplesPackage.model_validate(raw)
    except ValidationError as exc:
        raise ProportionError(
            f"invalid depth samples package: {exc}",
            code="invalid_depth_samples",
            details={"path": str(p)},
        ) from exc


def load_depth_deltas_package(path: Path | str) -> Any:
    """Load depth_mesh_deltas.json (lazy DepthDeltasPackage import — F1)."""
    # Lazy import so heatmap-only callers do not need the deltas type path first.
    from meshops.proportion.depth_samples import DepthDeltasPackage

    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot load depth deltas: {p}: {exc}",
            code="invalid_depth_deltas",
            details={"path": str(p)},
        ) from exc
    try:
        return DepthDeltasPackage.model_validate(raw)
    except ValidationError as exc:
        raise ProportionError(
            f"invalid depth deltas package: {exc}",
            code="invalid_depth_deltas",
            details={"path": str(p)},
        ) from exc


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _sample_z_frac(sample: DepthSample, *, height_m: float | None) -> float | None:
    """Body-up z_frac (0 sole → 1 crown). Never invert."""
    if sample.z_frac is not None:
        return float(sample.z_frac)
    if sample.z_m is not None and height_m is not None and height_m > 0:
        return float(sample.z_m) / float(height_m)
    return None


def _sample_y_value(sample: DepthSample) -> tuple[float | None, str | None]:
    """Horizontal body depth: prefer y_m, else y_frac. Returns (value, mode)."""
    if sample.y_m is not None:
        return float(sample.y_m), "y_m"
    if sample.y_frac is not None:
        return float(sample.y_frac), "y_frac"
    return None, None


def _sample_color_value(sample: DepthSample) -> tuple[float | None, ColorScaleMode | None]:
    """Color samples by y_m, else depth_m."""
    if sample.y_m is not None:
        return float(sample.y_m), "y_m"
    if sample.depth_m is not None:
        return float(sample.depth_m), "depth_m"
    if sample.y_frac is not None:
        return float(sample.y_frac), "y_m"  # treat as relative y scale
    return None, None


def _blue_green_red(t: float) -> tuple[int, int, int]:
    """Map t in [0,1] → RGB via np.interp blue→green→red (A2)."""
    t = float(np.clip(t, 0.0, 1.0))
    # knots: 0=blue, 0.5=green, 1=red
    r = int(np.interp(t, [0.0, 0.5, 1.0], [0.0, 0.0, 255.0]))
    g = int(np.interp(t, [0.0, 0.5, 1.0], [0.0, 200.0, 0.0]))
    b = int(np.interp(t, [0.0, 0.5, 1.0], [255.0, 0.0, 0.0]))
    return r, g, b


def _norm_color(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    t = 0.5 if vmax <= vmin else (value - vmin) / (vmax - vmin)
    return _blue_green_red(t)


def _pillow_required() -> ProportionError:
    return ProportionError(
        "heatmap PNG require Pillow (install meshops[proportion] / pillow)",
        code="pillow_required",
        details={"hint": "uv sync --extra proportion"},
    )


def _import_pillow() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageDraw  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:
        raise _pillow_required() from exc
    return Image, ImageDraw


def _resolve_heatmap_paths(out: Path | str) -> tuple[Path, Path]:
    """Resolve (png_path, meta_path) from --out (R1.1).

    Accepts str so trailing directory separators survive.
    """
    raw = str(out)
    ends_sep = raw.endswith(("/", "\\"))
    path = Path(raw.rstrip("/\\") if ends_sep else raw)

    if (path.exists() and path.is_dir()) or ends_sep:
        return path / HEATMAP_PNG_BASENAME, path / HEATMAP_JSON_BASENAME

    suffix = path.suffix.lower()
    if suffix == ".png":
        return path, path.parent / HEATMAP_JSON_BASENAME
    if suffix == ".json":
        return path.parent / HEATMAP_PNG_BASENAME, path
    raise ProportionError(
        "--out file must end with .png or .json or be a directory",
        code="heatmap_failed",
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
            f"failed to write heatmap meta: {exc}",
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
            f"failed to write heatmap PNG: {exc}",
            code="write_failed",
            details={"path": str(path)},
        ) from exc


def _usable_sample_points(
    samples: list[DepthSample],
    *,
    height_m: float | None,
) -> list[dict[str, Any]]:
    """Collect plottable sample points (body-up z + horizontal y)."""
    points: list[dict[str, Any]] = []
    for s in samples:
        zf = _sample_z_frac(s, height_m=height_m)
        y_val, _ = _sample_y_value(s)
        if zf is None or y_val is None:
            continue
        c_val, c_mode = _sample_color_value(s)
        points.append(
            {
                "id": s.id,
                "role": s.role,
                "z_frac": zf,
                "y_val": y_val,
                "color_val": c_val if c_val is not None else y_val,
                "color_mode": c_mode or "y_m",
            }
        )
    return points


def _usable_delta_points(
    deltas: list[Any],
    samples_by_id: dict[str, DepthSample],
    *,
    height_m: float | None,
) -> list[dict[str, Any]]:
    """Join deltas to sample z/y for plot positions."""
    points: list[dict[str, Any]] = []
    for d in deltas:
        sample = samples_by_id.get(d.id)
        if sample is None:
            continue
        zf = _sample_z_frac(sample, height_m=height_m)
        y_val, _ = _sample_y_value(sample)
        if zf is None or y_val is None:
            continue
        # Color by delta_y_m prefer, else delta_depth_m
        if d.delta_y_m is not None:
            c_val, c_mode = float(d.delta_y_m), "delta_y_m"
        elif d.delta_depth_m is not None:
            c_val, c_mode = float(d.delta_depth_m), "delta_depth_m"
        else:
            continue
        points.append(
            {
                "id": d.id,
                "role": sample.role,
                "z_frac": zf,
                "y_val": y_val,
                "color_val": c_val,
                "color_mode": c_mode,
            }
        )
    return points


def _panel_xy(
    z_frac: float,
    y_val: float,
    *,
    y_min: float,
    y_max: float,
    plot_x0: int,
    plot_y0: int,
    plot_w: int,
    plot_h: int,
) -> tuple[int, int]:
    """Map body-up z + body y → pixel (front +Y = right)."""
    # Vertical: z_frac 0 (sole) at bottom, 1 (crown) at top
    py = plot_y0 + round((1.0 - float(z_frac)) * (plot_h - 1))
    t = 0.5 if y_max <= y_min else (float(y_val) - y_min) / (y_max - y_min)
    t = float(np.clip(t, 0.0, 1.0))
    # front (+Y larger) = right
    px = plot_x0 + round(t * (plot_w - 1))
    return px, py


def _draw_marker(
    draw: Any,
    role: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    r: int = 6,
) -> None:
    """Marker shape by role (R3)."""
    fill = (*color, 230)
    outline = (20, 20, 20, 255)
    if role == "landmark":
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline)
    elif role == "band_front":
        # diamond
        draw.polygon(
            [(x, y - r), (x + r, y), (x, y + r), (x - r, y)],
            fill=fill,
            outline=outline,
        )
    elif role == "band_back":
        draw.rectangle((x - r, y - r, x + r, y + r), fill=fill, outline=outline)
    elif role == "band_mid":
        draw.ellipse((x - r // 2, y - r // 2, x + r // 2, y + r // 2), fill=fill, outline=outline)
        draw.line((x - r, y, x + r, y), fill=outline, width=1)
        draw.line((x, y - r, x, y + r), fill=outline, width=1)
    else:
        # band_span / other: X
        draw.line((x - r, y - r, x + r, y + r), fill=fill, width=2)
        draw.line((x - r, y + r, x + r, y - r), fill=fill, width=2)


def _draw_colorbar(
    draw: Any,
    *,
    x0: int,
    y0: int,
    w: int,
    h: int,
    vmin: float,
    vmax: float,
) -> None:
    """Horizontal color bar (R3.3: at bottom of each panel). Left=vmin, right=vmax."""
    for i in range(w):
        t = i / max(w - 1, 1)  # left = vmin, right = vmax
        color = _blue_green_red(t)
        draw.line([(x0 + i, y0), (x0 + i, y0 + h - 1)], fill=(*color, 255))
    draw.rectangle((x0, y0, x0 + w - 1, y0 + h - 1), outline=(40, 40, 40, 255))
    # labels under bar ends
    draw.text((x0, y0 + h + 1), f"{vmin:.3g}", fill=(30, 30, 30, 255))
    vmax_s = f"{vmax:.3g}"
    draw.text((x0 + w - max(len(vmax_s) * 6, 24), y0 + h + 1), vmax_s, fill=(30, 30, 30, 255))


def _draw_panel(
    canvas: Any,
    draw: Any,
    points: list[dict[str, Any]],
    *,
    panel_x0: int,
    panel_y0: int,
    panel_w: int,
    panel_h: int,
    title: str,
) -> ColorScale:
    """Draw one abstract panel; return ColorScale for meta.

    Layout (R3.3): plot area, then horizontal color bar at the **bottom** of the panel.
    """
    # Reserve bottom strip for horizontal colorbar + value labels (R3.3 bottom of panel).
    bottom_reserve = COLORBAR_H + 18
    plot_x0 = panel_x0 + MARGIN
    plot_y0 = panel_y0 + MARGIN
    plot_w = panel_w - MARGIN * 2
    plot_h = panel_h - MARGIN * 2 - bottom_reserve
    plot_w = max(plot_w, 40)
    plot_h = max(plot_h, 40)

    # axes box
    draw.rectangle(
        (plot_x0, plot_y0, plot_x0 + plot_w - 1, plot_y0 + plot_h - 1),
        outline=(80, 80, 80, 255),
        fill=(248, 248, 252, 255),
    )
    draw.text((plot_x0, panel_y0 + 4), title, fill=(20, 20, 20, 255))
    draw.text(
        (plot_x0, plot_y0 + plot_h + 2), "back ← y → front (+Y right)", fill=(60, 60, 60, 255)
    )
    draw.text((panel_x0 + 4, plot_y0), "crown", fill=(60, 60, 60, 255))
    draw.text((panel_x0 + 4, plot_y0 + plot_h - 12), "sole", fill=(60, 60, 60, 255))

    if not points:
        mode: ColorScaleMode = "y_m"
        return ColorScale(panel="samples" if "sample" in title.lower() else "deltas", mode=mode)

    y_vals = [float(p["y_val"]) for p in points]
    c_vals = [float(p["color_val"]) for p in points]
    y_min, y_max = min(y_vals), max(y_vals)
    c_min, c_max = min(c_vals), max(c_vals)
    modes = {p["color_mode"] for p in points}
    mode = (
        next(iter(modes))
        if len(modes) == 1
        else ("delta_y_m" if any(str(m).startswith("delta") for m in modes) else "y_m")
    )
    # normalize mode to allowed
    if mode not in ("y_m", "depth_m", "delta_y_m", "delta_depth_m"):
        mode = "y_m"

    for p in points:
        px, py = _panel_xy(
            float(p["z_frac"]),
            float(p["y_val"]),
            y_min=y_min,
            y_max=y_max,
            plot_x0=plot_x0,
            plot_y0=plot_y0,
            plot_w=plot_w,
            plot_h=plot_h,
        )
        color = _norm_color(float(p["color_val"]), c_min, c_max)
        _draw_marker(draw, str(p["role"]), px, py, color)

    # R3.3: color bar at bottom of each panel (horizontal).
    cb_y = panel_y0 + panel_h - COLORBAR_H - 14
    _draw_colorbar(draw, x0=plot_x0, y0=cb_y, w=plot_w, h=COLORBAR_H, vmin=c_min, vmax=c_max)

    panel_name: ColorScalePanel = (
        "deltas" if mode.startswith("delta") or "delta" in title.lower() else "samples"
    )
    return ColorScale(
        panel=panel_name,
        mode=mode,  # type: ignore[arg-type]
        vmin=c_min,
        vmax=c_max,
        note=_COLOR_NOTE,
    )


def _plot_vertical_positions(
    points: list[dict[str, Any]],
    *,
    panel_h: int = DEFAULT_PANEL_H,
) -> dict[str, int]:
    """Public helper for tests: body-up z → same vertical plot row for same z_frac."""
    plot_h = panel_h - MARGIN * 2 - 16
    plot_y0 = MARGIN
    out: dict[str, int] = {}
    for p in points:
        py = plot_y0 + round((1.0 - float(p["z_frac"])) * (plot_h - 1))
        out[str(p["id"])] = py
    return out


def render_heatmap_image(
    sample_points: list[dict[str, Any]],
    delta_points: list[dict[str, Any]] | None = None,
    *,
    underlay_path: Path | None = None,
    messages: list[str] | None = None,
) -> tuple[Any, list[ColorScale], list[str]]:
    """Render heatmap PNG (Pillow Image). Returns (image, color_scales, messages)."""
    Image, ImageDraw = _import_pillow()
    msgs = list(messages or [])
    has_deltas = bool(delta_points)

    panel_w = DEFAULT_PANEL_W
    panel_h = DEFAULT_PANEL_H

    if has_deltas:
        canvas_h = 2 * panel_h + PANEL_GAP_PX + FOOTER_H
        canvas_w = panel_w
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        cs_samples = _draw_panel(
            canvas,
            draw,
            sample_points,
            panel_x0=0,
            panel_y0=0,
            panel_w=panel_w,
            panel_h=panel_h,
            title="samples (y / depth glance)",
        )
        cs_samples = cs_samples.model_copy(update={"panel": "samples"})
        cs_deltas = _draw_panel(
            canvas,
            draw,
            delta_points or [],
            panel_x0=0,
            panel_y0=panel_h + PANEL_GAP_PX,
            panel_w=panel_w,
            panel_h=panel_h,
            title="deltas (mesh shallower/thinner = +)",
        )
        cs_deltas = cs_deltas.model_copy(update={"panel": "deltas"})
        color_scales = [cs_samples, cs_deltas]
    else:
        # Optional underlay: markers on image when fracs allow
        if underlay_path is not None and underlay_path.is_file():
            try:
                with Image.open(underlay_path) as im:
                    base = im.convert("RGBA").copy()
                # Draw markers at glance y_px; horizontal mid if no x frac
                draw = ImageDraw.Draw(base)
                w, h = base.size
                for p in sample_points:
                    y_px = (1.0 - float(p["z_frac"])) * (h - 1)
                    # Map y_val across image width as glance (front right)
                    y_vals = [float(q["y_val"]) for q in sample_points]
                    y_min, y_max = min(y_vals), max(y_vals)
                    t = 0.5 if y_max <= y_min else (float(p["y_val"]) - y_min) / (y_max - y_min)
                    x_px = t * (w - 1)
                    c_vals = [float(q["color_val"]) for q in sample_points]
                    color = _norm_color(float(p["color_val"]), min(c_vals), max(c_vals))
                    _draw_marker(
                        draw,
                        str(p["role"]),
                        round(x_px),
                        round(y_px),
                        color,
                        r=max(4, min(w, h) // 80),
                    )
                # Footer strip for honesty
                footer = Image.new("RGBA", (w, FOOTER_H), (255, 255, 255, 255))
                fdraw = ImageDraw.Draw(footer)
                fdraw.text(
                    (4, 6),
                    f"{HEATMAP_HONESTY} — glance only; numbers SoT",
                    fill=(40, 40, 40, 255),
                )
                canvas = Image.new("RGBA", (w, h + FOOTER_H), (255, 255, 255, 255))
                canvas.paste(base, (0, 0))
                canvas.paste(footer, (0, h))
                c_vals = [float(q["color_val"]) for q in sample_points] or [0.0]
                modes = {p["color_mode"] for p in sample_points}
                mode = next(iter(modes)) if len(modes) == 1 else "y_m"
                if mode not in ("y_m", "depth_m", "delta_y_m", "delta_depth_m"):
                    mode = "y_m"
                color_scales = [
                    ColorScale(
                        panel="samples",
                        mode=mode,  # type: ignore[arg-type]
                        vmin=min(c_vals),
                        vmax=max(c_vals),
                        note=_COLOR_NOTE,
                    )
                ]
                return canvas, color_scales, msgs
            except OSError:
                msgs.append(f"underlay ignored (unreadable): {underlay_path}")
        elif underlay_path is not None:
            msgs.append(f"underlay ignored (not a file): {underlay_path}")

        canvas_h = panel_h + FOOTER_H
        canvas = Image.new("RGBA", (panel_w, canvas_h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        cs = _draw_panel(
            canvas,
            draw,
            sample_points,
            panel_x0=0,
            panel_y0=0,
            panel_w=panel_w,
            panel_h=panel_h,
            title="samples (y / depth glance)",
        )
        cs = cs.model_copy(update={"panel": "samples"})
        color_scales = [cs]

    # Footer honesty
    draw = ImageDraw.Draw(canvas)
    fh = canvas.size[1]
    draw.text(
        (4, fh - FOOTER_H + 6),
        f"{HEATMAP_HONESTY} — glance only; numbers SoT",
        fill=(40, 40, 40, 255),
    )
    return canvas, color_scales, msgs


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def run_depth_heatmap(
    samples: Path | str,
    out: Path | str,
    *,
    deltas: Path | str | None = None,
    underlay: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build depth heatmap PNG + meta from samples (+ optional deltas).

    Returns CLI/MCP success payload. Raises ProportionError.
    *out* may be str so trailing directory separators are preserved.
    """
    samples_path = Path(samples)
    package = load_depth_samples_package(samples_path)
    height_m = package.height_m

    sample_points = _usable_sample_points(package.samples, height_m=height_m)
    messages = list(package.messages)

    delta_points: list[dict[str, Any]] | None = None
    deltas_path_str: str | None = None
    if deltas is not None:
        dpath = Path(deltas)
        deltas_pkg = load_depth_deltas_package(dpath)
        deltas_path_str = str(dpath)
        by_id = {s.id: s for s in package.samples}
        delta_points = _usable_delta_points(
            list(deltas_pkg.deltas),
            by_id,
            height_m=height_m,
        )
        messages.extend(list(deltas_pkg.messages))

    if not sample_points and not (delta_points or []):
        raise ProportionError(
            "no usable depth samples/deltas to plot (all missing z/y meters or fracs)",
            code="heatmap_empty",
            details={"samples": str(samples_path)},
        )

    # Need at least one sample point for primary; allow deltas-only with empty samples
    # but heatmap_empty already covers both empty.
    if not sample_points and delta_points:
        # Still plot deltas dual layout? Spec: zero usable samples → heatmap_empty
        # "Zero usable samples (all null meters/fracs with nothing to plot)"
        # If only deltas plottable via sample join, sample_points empty means join failed
        raise ProportionError(
            "no usable depth samples to plot",
            code="heatmap_empty",
            details={"samples": str(samples_path)},
        )

    png_path, meta_path = _resolve_heatmap_paths(out)
    underlay_p = Path(underlay) if underlay is not None else None

    image, color_scales, messages = render_heatmap_image(
        sample_points,
        delta_points if deltas is not None else None,
        underlay_path=underlay_p,
        messages=messages,
    )

    plotted_sample_ids = [str(p["id"]) for p in sample_points]
    plotted_delta_ids = [str(p["id"]) for p in (delta_points or [])]

    meta = DepthHeatmapPackage(
        schema_version=HEATMAP_SCHEMA_VERSION,
        honesty=HEATMAP_HONESTY,
        source_samples=str(samples_path),
        source_deltas=deltas_path_str,
        underlay=str(underlay_p) if underlay_p is not None else None,
        axis_notes=AXIS_NOTES,
        color_scales=color_scales,
        plotted_sample_ids=plotted_sample_ids,
        plotted_delta_ids=plotted_delta_ids,
        messages=messages,
        counts={
            "samples_plotted": len(plotted_sample_ids),
            "deltas_plotted": len(plotted_delta_ids),
        },
    )

    _write_png(png_path, image, force=force)
    _write_json(meta_path, meta.model_dump(mode="json"), force=force)

    return {
        "ok": True,
        "paths": [str(png_path), str(meta_path)],
        "counts": {
            "samples_plotted": len(plotted_sample_ids),
            "deltas_plotted": len(plotted_delta_ids),
        },
        "messages": messages,
    }


__all__ = [
    "AXIS_NOTES",
    "HEATMAP_JSON_BASENAME",
    "HEATMAP_PNG_BASENAME",
    "HEATMAP_SCHEMA_VERSION",
    "PANEL_GAP_PX",
    "ColorScale",
    "DepthHeatmapPackage",
    "_panel_xy",
    "_plot_vertical_positions",
    "_usable_sample_points",
    "load_depth_deltas_package",
    "load_depth_samples_package",
    "render_heatmap_image",
    "run_depth_heatmap",
]
