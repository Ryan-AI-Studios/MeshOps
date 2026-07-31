"""Blender binary discovery (env → PATH → well-known Windows 5.2 path).

Install mirrors / doctor ritual belong to track 0010 — not reimplemented here.
Difficulty §4: discovery + env override only.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from meshops.escalate.errors import EscalateError

# TechStack pin: Blender 5.2 LTS only (not 4.2 EOL).
WELL_KNOWN_WINDOWS_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")

ENV_MESHOPS_BLENDER = "MESHOPS_BLENDER"


def find_blender(*, require: bool = True) -> Path | None:
    """Locate Blender executable.

    Order:
      1. ``MESHOPS_BLENDER`` env (file path)
      2. ``shutil.which("blender")``
      3. Windows well-known 5.2 LTS install path

    When *require* is True (default), missing binary raises
    ``EscalateError(code="blender_missing")``.
    """
    candidates: list[Path] = []

    env = os.environ.get(ENV_MESHOPS_BLENDER, "").strip()
    if env:
        candidates.append(Path(env))

    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))

    if os.name == "nt":
        candidates.append(WELL_KNOWN_WINDOWS_BLENDER)

    for cand in candidates:
        try:
            p = cand.expanduser().resolve(strict=False)
        except OSError:
            p = cand.expanduser()
        if p.is_file():
            return p

    if require:
        hint = (
            f"Set {ENV_MESHOPS_BLENDER} to blender.exe, install Blender 5.2 LTS, "
            "or complete track 0010 doctor/mirror bootstrap. "
            f"Well-known path checked: {WELL_KNOWN_WINDOWS_BLENDER}"
        )
        raise EscalateError(
            f"Blender 5.2 LTS not found. {hint}",
            code="blender_missing",
            details={
                "env": env or None,
                "which": which,
                "well_known": str(WELL_KNOWN_WINDOWS_BLENDER),
            },
        )
    return None
