"""Pass multi-view evidence via F3D render_mesh_to_dir (B3/B4).

Required: front.png, left.png, three_quarter.png, three_quarter_depth.png.
Depth is F3D display_depth MODE on three_quarter pose — not a camera name.
Do NOT use render_diff_views for pass evidence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from meshops.organic.models import REQUIRED_VIEW_KEYS

# Minimal valid 1x1 PNG (black) — same spirit as design/orchestrate.
_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

REQUIRED_RGB = ("front", "left", "three_quarter")
ViewKind = Literal["f3d", "stub"]


def _prefer_stub_views() -> bool:
    if stub := os.environ.get("MESHOPS_STUB_DIFF", "").strip().lower():
        return stub in {"1", "true", "yes", "on"}
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def write_stub_pass_views(views_dir: Path) -> dict[str, str]:
    """Write honest stub PNGs for required organic pass views."""
    views_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for key in REQUIRED_VIEW_KEYS:
        dest = views_dir / f"{key}.png"
        dest.write_bytes(_MIN_PNG)
        out[key] = str(dest)
    return out


def collect_view_paths(views_dir: Path) -> dict[str, str]:
    """Collect required view file paths if present and non-empty."""
    out: dict[str, str] = {}
    for key in REQUIRED_VIEW_KEYS:
        p = views_dir / f"{key}.png"
        if p.is_file() and p.stat().st_size > 0:
            out[key] = str(p)
    return out


def views_complete(view_paths: dict[str, str]) -> bool:
    return all(k in view_paths for k in REQUIRED_VIEW_KEYS)


def render_pass_views(
    mesh_path: Path | str,
    views_dir: Path | str,
    *,
    force_stub: bool = False,
) -> tuple[dict[str, str], ViewKind, list[str]]:
    """Render required pass evidence; honest stub fallback when F3D unavailable.

    Returns (view_paths dict, view_kind, notes).
    """
    mesh = Path(mesh_path)
    vdir = Path(views_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    if force_stub or _prefer_stub_views():
        notes.append("views_stub_used")
        if force_stub:
            notes.append("views_stub_force")
        else:
            notes.append("views_stub_ci_or_MESHOPS_STUB_DIFF")
        return write_stub_pass_views(vdir), "stub", notes

    try:
        from meshops.render.f3d_renderer import F3DRenderer

        result = F3DRenderer().render_mesh_to_dir(
            mesh,
            vdir,
            camera_names=REQUIRED_RGB,
            include_depth_for=("three_quarter",),
            mesh_id="",
            rendered_from="organic_pass",
        )
        # Map result paths into required keys
        view_paths: dict[str, str] = {}
        for p in result.view_paths:
            stem = Path(p).stem
            if stem in REQUIRED_RGB:
                view_paths[stem] = p
        for p in result.depth_paths:
            name = Path(p).name
            if name == "three_quarter_depth.png" or Path(p).stem == "three_quarter_depth":
                view_paths["three_quarter_depth"] = p
        # Also accept {pose}_depth.png naming from renderer
        tqd = vdir / "three_quarter_depth.png"
        if tqd.is_file() and tqd.stat().st_size > 0:
            view_paths["three_quarter_depth"] = str(tqd)

        if not views_complete(view_paths):
            notes.append("views_incomplete_used_stub")
            notes.append("views_stub_used")
            return write_stub_pass_views(vdir), "stub", notes
        return view_paths, "f3d", notes
    except Exception as exc:
        notes.append(f"views_unavailable_used_stub: {type(exc).__name__}: {exc}")
        notes.append("views_stub_used")
        return write_stub_pass_views(vdir), "stub", notes
