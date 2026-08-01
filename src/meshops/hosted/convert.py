"""GLB → STL conversion via trimesh Scene.to_mesh bake (when needed)."""

from __future__ import annotations

from pathlib import Path

import trimesh

from meshops.hosted.errors import HostedError


def glb_to_stl(glb_path: Path | str, stl_path: Path | str) -> Path:
    """Load GLB (force scene), bake transforms with to_mesh(), export STL."""
    src = Path(glb_path)
    dest = Path(stl_path)
    if not src.is_file():
        raise HostedError(
            f"GLB not found for convert: {src}",
            code="convert_failed",
            details={"glb_path": str(src)},
        )
    try:
        loaded = trimesh.load(str(src), force="scene")
        mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
        if mesh is None or not hasattr(mesh, "export"):
            raise HostedError(
                "GLB load produced no exportable mesh",
                code="convert_failed",
                details={"glb_path": str(src), "type": type(loaded).__name__},
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(dest))
    except HostedError:
        raise
    except Exception as exc:
        raise HostedError(
            f"GLB→STL convert failed: {exc}",
            code="convert_failed",
            details={"glb_path": str(src), "error": str(exc), "cause": type(exc).__name__},
        ) from exc

    if not dest.is_file() or dest.stat().st_size <= 0:
        raise HostedError(
            f"convert produced empty STL: {dest}",
            code="convert_failed",
            details={"stl_path": str(dest)},
        )
    return dest
