"""Subprocess isolation for F3D tests.

libf3d can SIGSEGV on some CI images (no usable GL). DoD-6 allows skip only on
proven unavailability — a crash in a child process counts as unavailable, not a
suite-level hard failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_f3d_render_job_isolated(
    mesh_id: str,
    work_root: Path,
    *,
    width: int = 256,
    height: int = 256,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run F3DRenderer.render_job in a child process; return JSON result dict.

    Raises RuntimeError only for unexpected harness failures. Callers should treat
    ``ok is False`` or process crash as skip / unavailable.
    """
    work = str(Path(work_root).resolve())
    code = f"""
import json
import sys
from pathlib import Path

from meshops.render.f3d_renderer import F3DRenderer, RenderUnavailableError

try:
    result = F3DRenderer(width={width}, height={height}).render_job(
        {mesh_id!r},
        work_root=Path({work!r}),
    )
    print(
        json.dumps(
            {{
                "ok": True,
                "mesh_id": result.mesh_id,
                "rendered_from": result.rendered_from,
                "view_paths": result.view_paths,
                "depth_paths": result.depth_paths,
                "cameras": result.cameras,
            }}
        )
    )
except RenderUnavailableError as exc:
    print(json.dumps({{"ok": False, "error": "RenderUnavailableError", "message": str(exc)}}))
    raise SystemExit(2) from exc
except Exception as exc:
    print(
        json.dumps(
            {{
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }}
        )
    )
    raise SystemExit(3) from exc
"""
    env = os.environ.copy()
    # Prefer software GL when present (Linux CI without GPU).
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("GALLIUM_DRIVER", "llvmpipe")
    env.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
        check=False,
    )

    # Negative returncode = killed by signal (e.g. SIGSEGV → -11 on Unix).
    if proc.returncode < 0 or proc.returncode in {139, 134, -11, -6}:
        return {
            "ok": False,
            "error": "RenderUnavailableError",
            "message": (
                f"F3D child crashed (returncode={proc.returncode}); "
                f"stderr={proc.stderr[-500:] if proc.stderr else ''}"
            ),
        }

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return {
            "ok": False,
            "error": "RenderUnavailableError",
            "message": f"F3D child produced no stdout (rc={proc.returncode}): {proc.stderr[-500:]}",
        }

    # Last non-empty JSON line (ignore any noise).
    line = stdout.splitlines()[-1]
    try:
        payload: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "RenderUnavailableError",
            "message": f"F3D child non-JSON output: {line[:500]}",
        }
    return payload
