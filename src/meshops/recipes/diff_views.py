"""Diff views: parent/original vs rev with same cameras (Difficulty §12).

Baseline pin: parent rev mesh or original.stl — NEVER working.ply.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from meshops.ingest.stats import load_mesh
from meshops.render.cameras import bbox_cameras
from meshops.render.f3d_renderer import RenderUnavailableError


def render_diff_views(
    *,
    baseline_mesh: Path,
    candidate_mesh: Path,
    views_dir: Path,
    width: int = 512,
    height: int = 512,
    camera_names: list[str] | None = None,
) -> list[str]:
    """Render same-camera before/after PNGs under views_dir.

    Writes ``{name}_before.png`` and ``{name}_after.png``.
    Returns list of written paths (strings).
    Raises RenderUnavailableError if F3D cannot run.
    """
    from meshops.render.f3d_renderer import (
        _apply_camera,
        _import_f3d,
        _require_option,
    )

    views_dir.mkdir(parents=True, exist_ok=True)

    # Frame cameras from union of bboxes (stable framing)
    base = load_mesh(baseline_mesh)
    cand = load_mesh(candidate_mesh)
    b0 = base.bounds
    b1 = cand.bounds
    bmin = (
        float(min(b0[0, 0], b1[0, 0])),
        float(min(b0[0, 1], b1[0, 1])),
        float(min(b0[0, 2], b1[0, 2])),
    )
    bmax = (
        float(max(b0[1, 0], b1[1, 0])),
        float(max(b0[1, 1], b1[1, 1])),
        float(max(b0[1, 2], b1[1, 2])),
    )
    poses = bbox_cameras(bmin, bmax)
    if camera_names is not None:
        allow = set(camera_names)
        poses = [p for p in poses if p.name in allow]
    # Default subset for repair speed: front + three_quarter
    if camera_names is None:
        prefer = {"front", "three_quarter", "top"}
        poses = [p for p in poses if p.name in prefer] or poses[:3]

    f3d = _import_f3d()
    try:
        engine = f3d.Engine.create(offscreen=True)
    except Exception as exc:
        raise RenderUnavailableError(
            "F3D Engine.create(offscreen=True) failed",
            cause=exc,
        ) from exc

    view_paths: list[str] = []
    options = engine.options
    _require_option(options, "scene.camera.orthographic", True)
    window = engine.window
    window.size = (width, height)
    camera = window.camera

    def _render_mesh(mesh_path: Path, dest: Path) -> None:
        scene = engine.scene
        # Clear previous geometry if API allows
        if hasattr(scene, "clear"):
            with contextlib.suppress(Exception):
                scene.clear()
        scene.add(str(mesh_path))
        _require_option(options, "render.effect.display_depth", False)
        if hasattr(window, "render_to_image"):
            img = window.render_to_image()
        elif hasattr(window, "renderToImage"):
            img = window.renderToImage()
        else:
            raise RenderUnavailableError("Window has no render_to_image API")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(img, "save"):
            img.save(str(dest))
        else:
            raise RenderUnavailableError("F3D Image has no save() method")

    try:
        for pose in poses:
            _apply_camera(camera, pose)
            before = views_dir / f"{pose.name}_before.png"
            after = views_dir / f"{pose.name}_after.png"
            _render_mesh(baseline_mesh, before)
            view_paths.append(str(before))
            _render_mesh(candidate_mesh, after)
            view_paths.append(str(after))
    except RenderUnavailableError:
        raise
    except Exception as exc:
        raise RenderUnavailableError("diff render failed", cause=exc) from exc

    return view_paths


def render_rev_diff(
    mesh_id: str,
    rev_id: str,
    *,
    work_root: Path | str = "work",
) -> dict[str, Any]:
    """Standalone diff for an existing rev; updates meta view_paths best-effort."""
    from meshops.jobstore.paths import JobPaths
    from meshops.revs.store import (
        load_manifest,
        parent_mesh_path,
        resolve_rev_dir,
        rev_mesh_path,
    )

    paths = JobPaths(work_root=Path(work_root), mesh_id=mesh_id)
    rev_dir = resolve_rev_dir(paths, rev_id)
    manifest = load_manifest(rev_dir)
    baseline = parent_mesh_path(paths, manifest.parent_rev)
    candidate = rev_mesh_path(rev_dir)
    views_dir = rev_dir / "views"
    view_paths = render_diff_views(
        baseline_mesh=baseline,
        candidate_mesh=candidate,
        views_dir=views_dir,
    )
    # Update meta
    updated = manifest.model_copy(update={"view_paths": view_paths})
    (rev_dir / "meta.json").write_text(
        updated.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "mesh_id": mesh_id,
        "rev_id": rev_id,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "view_paths": view_paths,
    }
