"""Collect multi-view reference images for hosted submit (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from meshops.hosted.errors import HostedError

# Preferred RGB keys for multi-view conditioning (depth not required).
PREFERRED_VIEW_KEYS: tuple[str, ...] = ("front", "left", "three_quarter", "back")
MIN_VIEWS = 2

ViewsFrom = Literal["pass", "latest", "explicit"]


def _is_complete_rgb(views_dir: Path) -> bool:
    """True when front + left + three_quarter exist as non-empty files."""
    for key in ("front", "left", "three_quarter"):
        p = views_dir / f"{key}.png"
        if not p.is_file() or p.stat().st_size <= 0:
            return False
    return True


def _collect_from_views_dir(views_dir: Path) -> list[Path]:
    """Ordered preferred keys that exist under views_dir."""
    out: list[Path] = []
    for key in PREFERRED_VIEW_KEYS:
        p = views_dir / f"{key}.png"
        if p.is_file() and p.stat().st_size > 0:
            out.append(p.resolve())
    return out


def _latest_pass_with_views(organic_dir: Path) -> list[Path]:
    """Find latest successful pass under organic/passes with complete RGB views.

    Pass order: prefer manifest.passes if present; else lexicographic dir names.
    """
    passes_dir = organic_dir / "passes"
    if not passes_dir.is_dir():
        return []

    pass_names: list[str] = []
    manifest_path = organic_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw = data.get("passes") or []
            if isinstance(raw, list):
                pass_names = [str(x) for x in raw]
        except (OSError, json.JSONDecodeError, TypeError):
            pass_names = []

    if not pass_names:
        pass_names = sorted(
            d.name for d in passes_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    # Walk newest-first
    for name in reversed(pass_names):
        views_dir = passes_dir / name / "views"
        if _is_complete_rgb(views_dir):
            return _collect_from_views_dir(views_dir)
    return []


def collect_view_paths(
    *,
    plateau_path: Path,
    views_from: ViewsFrom = "latest",
    explicit_views: list[Path] | None = None,
) -> list[Path]:
    """Resolve multi-view image paths relative to plateau parent (out-of-tree OK).

    Raises HostedError multiview_required when fewer than MIN_VIEWS images.
    """
    organic_dir = Path(plateau_path).resolve().parent
    paths: list[Path] = []

    if views_from == "explicit":
        if not explicit_views:
            raise HostedError(
                "explicit views required when --views-from=explicit",
                code="multiview_required",
                details={"views_from": views_from},
            )
        for v in explicit_views:
            p = Path(v)
            if not p.is_file():
                # Resolve relative to plateau parent for out-of-tree sessions
                alt = organic_dir / v
                if alt.is_file():
                    p = alt
            if not p.is_file() or p.stat().st_size <= 0:
                raise HostedError(
                    f"view path missing or empty: {v}",
                    code="multiview_required",
                    details={"path": str(v)},
                )
            paths.append(p.resolve())
    else:
        # pass | latest — both map to latest successful pass with complete views
        paths = _latest_pass_with_views(organic_dir)

    if len(paths) < MIN_VIEWS:
        raise HostedError(
            f"multi-view required: need ≥{MIN_VIEWS} images, got {len(paths)}",
            code="multiview_required",
            details={
                "count": len(paths),
                "paths": [str(p) for p in paths],
                "organic_dir": str(organic_dir),
                "views_from": views_from,
            },
        )
    return paths
