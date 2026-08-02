"""F3DRenderer — lazy f3d import, offscreen RGB + visual depth (Difficulty §9).

Depth via render.effect.display_depth is an 8-bit colormapped preview for agents,
NOT a metric rangefinder. Numeric thickness uses mesh queries.

F3D 3.5 Python bindings (snake_case): Engine.create(offscreen=True), engine.scene,
engine.window, engine.options, window.camera, window.render_to_image(), image.save().

Track 0006: ``render_mesh_to_dir`` is the path-based API; ``render_job`` delegates.
Depth is a *mode* (display_depth), not a camera name — e.g. three_quarter_depth.png.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meshops.ingest.stats import load_mesh
from meshops.jobstore.paths import JobPaths
from meshops.render.cameras import CameraPose, bbox_cameras


class RenderUnavailableError(RuntimeError):
    """Structured failure when F3D offscreen cannot run."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


@dataclass
class RenderResult:
    """Paths written under views/ plus metadata."""

    mesh_id: str
    rendered_from: str
    view_paths: list[str] = field(default_factory=list)
    depth_paths: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)


def _import_f3d() -> Any:
    """Lazy import f3d — never at module top level for ingest/triage/report."""
    try:
        import f3d  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RenderUnavailableError(
            "f3d package is not installed or failed to import",
            cause=exc,
        ) from exc
    return f3d


def _select_mesh_file(paths: JobPaths) -> tuple[Path, str]:
    """Prefer working, then original, then proxy."""
    if paths.working_ply.is_file():
        return paths.working_ply, "working"
    if paths.original_stl.is_file():
        return paths.original_stl, "original"
    if paths.proxy_ply.is_file():
        return paths.proxy_ply, "proxy"
    raise RenderUnavailableError(f"No mesh file found for mesh_id={paths.mesh_id!r}")


def _apply_camera(camera: Any, pose: CameraPose) -> None:
    """Set libf3d 3.5 camera from CameraPose (position + look-at + up)."""
    camera.position = list(pose.position)
    camera.focal_point = list(pose.focal_point)
    camera.view_up = list(pose.view_up)
    # Parallel-projection framing: zoom / parallel_scale when available.
    for attr, value in (
        ("zoom", pose.ortho_scale),
        ("parallel_scale", pose.ortho_scale),
        ("view_angle", 30.0),  # unused when orthographic; harmless fallback
    ):
        if hasattr(camera, attr):
            try:
                setattr(camera, attr, value)
                break
            except Exception:
                continue


def _set_option(options: Any, key: str, value: Any) -> bool:
    """Set an F3D option. Returns True if the assignment succeeded."""
    try:
        options[key] = value
        return True
    except Exception:
        try:
            if hasattr(options, "update"):
                options.update({key: value})
                return True
        except Exception:
            return False
    return False


def _require_option(options: Any, key: str, value: Any) -> None:
    if not _set_option(options, key, value):
        raise RenderUnavailableError(f"Failed to set F3D option {key!r}={value!r}")


def _merge_render_into_diagnostics(paths: JobPaths, result: RenderResult) -> None:
    """Update diagnostics.json view evidence when triage already ran (DoD-2 fields)."""
    if not paths.diagnostics_json.is_file():
        return
    try:
        from meshops.models.diagnostics import Diagnostics

        diag = Diagnostics.model_validate_json(paths.diagnostics_json.read_text(encoding="utf-8"))
        diag.rendered_from = result.rendered_from
        diag.view_paths = list(result.view_paths) + list(result.depth_paths)
        paths.diagnostics_json.write_text(
            diag.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except Exception:
        # Non-fatal: report.md still lists views from filesystem.
        pass


class F3DRenderer:
    """Offscreen F3D renderer for triage evidence views."""

    def __init__(self, width: int = 512, height: int = 512) -> None:
        self.width = width
        self.height = height

    def render_mesh_to_dir(
        self,
        mesh_path: Path | str,
        views_dir: Path | str,
        *,
        camera_names: Sequence[str] = ("front", "left", "three_quarter"),
        include_depth_for: Sequence[str] = ("three_quarter",),
        width: int | None = None,
        height: int | None = None,
        mesh_id: str = "",
        rendered_from: str = "mesh",
        background_color: Sequence[float] | None = None,
    ) -> RenderResult:
        """Render RGB (+ selective visual depth) for named cameras into views_dir.

        Depth is F3D ``display_depth`` mode on the listed pose names (not a camera).
        When ``include_depth_for`` is non-empty, at least one depth map must be
        written or ``RenderUnavailableError`` is raised.

        ``background_color`` is optional RGB in 0..1 (F3D ``render.background.color``).
        Silhouette compare (0021) passes white so ``frame.silhouette_mask`` treats
        background as near-white rather than full-frame foreground.
        """
        f3d = _import_f3d()
        mesh_file = Path(mesh_path)
        out_dir = Path(views_dir)
        if not mesh_file.is_file():
            raise RenderUnavailableError(f"Mesh file not found: {mesh_file}")
        out_dir.mkdir(parents=True, exist_ok=True)

        w = width if width is not None else self.width
        h = height if height is not None else self.height
        depth_set = set(include_depth_for)
        name_set = set(camera_names)

        try:
            mesh = load_mesh(mesh_file)
            bounds = mesh.bounds
            bmin = (float(bounds[0, 0]), float(bounds[0, 1]), float(bounds[0, 2]))
            bmax = (float(bounds[1, 0]), float(bounds[1, 1]), float(bounds[1, 2]))
            poses = [p for p in bbox_cameras(bmin, bmax) if p.name in name_set]
            # Preserve caller order when possible
            order = {n: i for i, n in enumerate(camera_names)}
            poses.sort(key=lambda p: order.get(p.name, 999))
        except Exception as exc:
            raise RenderUnavailableError(
                "Failed to load mesh for camera framing",
                cause=exc,
            ) from exc

        if not poses:
            raise RenderUnavailableError(
                f"No camera poses matched camera_names={list(camera_names)!r}"
            )

        try:
            engine = f3d.Engine.create(offscreen=True)
        except Exception as exc:
            raise RenderUnavailableError(
                "F3D Engine.create(offscreen=True) failed",
                cause=exc,
            ) from exc

        try:
            scene = engine.scene
            scene.add(str(mesh_file))

            window = engine.window
            window.size = (w, h)

            options = engine.options
            _require_option(options, "scene.camera.orthographic", True)
            if background_color is not None:
                # Soft set: fail closed only when caller requires white (0021).
                bg = [float(c) for c in background_color]
                if not _set_option(options, "render.background.color", bg):
                    raise RenderUnavailableError(
                        "Failed to set F3D option 'render.background.color'="
                        f"{bg!r} (required for silhouette-safe mesh renders)"
                    )

            camera = window.camera
            result = RenderResult(mesh_id=mesh_id, rendered_from=rendered_from)

            for pose in poses:
                _apply_camera(camera, pose)

                _require_option(options, "render.effect.display_depth", False)
                rgb_path = out_dir / f"{pose.name}.png"
                self._render_to(window, rgb_path)
                result.view_paths.append(str(rgb_path))
                result.cameras.append(pose.name)

                if pose.name in depth_set:
                    _require_option(options, "render.effect.display_depth", True)
                    depth_path = out_dir / f"{pose.name}_depth.png"
                    self._render_to(window, depth_path)
                    if not depth_path.is_file() or depth_path.stat().st_size <= 0:
                        raise RenderUnavailableError(
                            f"Depth map missing or empty: {depth_path.name}"
                        )
                    result.depth_paths.append(str(depth_path))
                    _require_option(options, "render.effect.display_depth", False)

            if depth_set and not result.depth_paths:
                raise RenderUnavailableError(
                    "DoD-6 requires ≥1 visual depth map; none were written"
                )

            return result
        except RenderUnavailableError:
            raise
        except Exception as exc:
            raise RenderUnavailableError("F3D render failed", cause=exc) from exc

    def render_job(
        self,
        mesh_id: str,
        *,
        work_root: Path | str = "work",
        include_depth: bool = True,
    ) -> RenderResult:
        """Render RGB (+ visual depth) for all bbox cameras into job views/.

        When ``include_depth`` is True, depth is written for **all** poses
        (preserves pre-0006 job behavior). Delegates to ``render_mesh_to_dir``.
        """
        paths = JobPaths(work_root=Path(work_root), mesh_id=mesh_id)
        if not paths.job_dir.is_dir():
            raise RenderUnavailableError(f"Job directory not found: {paths.job_dir}")

        mesh_file, rendered_from = _select_mesh_file(paths)
        paths.views_dir.mkdir(exist_ok=True)

        # All standard bbox camera names
        all_names = (
            "front",
            "back",
            "left",
            "right",
            "top",
            "bottom",
            "three_quarter",
        )
        depth_for: Sequence[str] = all_names if include_depth else ()

        result = self.render_mesh_to_dir(
            mesh_file,
            paths.views_dir,
            camera_names=all_names,
            include_depth_for=depth_for,
            mesh_id=mesh_id,
            rendered_from=rendered_from,
        )
        _merge_render_into_diagnostics(paths, result)
        return result

    def _render_to(self, window: Any, dest: Path) -> None:
        try:
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
        except RenderUnavailableError:
            raise
        except Exception as exc:
            raise RenderUnavailableError(
                f"render_to_image failed for {dest.name}",
                cause=exc,
            ) from exc
