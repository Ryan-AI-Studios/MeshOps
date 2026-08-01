"""Blender binary discovery (env → PATH → well-known → portable MeshOps tools).

Install mirrors / doctor ritual belong to track 0010 — discovery only here.
Difficulty §4: discovery + env override + portable well-known path (R4).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

from meshops.escalate.errors import EscalateError
from meshops.ops.mirrors import PORTABLE_BLENDER_DIR_NAME, PORTABLE_BLENDER_EXE_NAME

# TechStack pin: Blender 5.2 LTS only (not 4.2 EOL).
WELL_KNOWN_WINDOWS_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")

ENV_MESHOPS_BLENDER = "MESHOPS_BLENDER"

BlenderSource = Literal["env", "path", "well_known", "portable", "missing"]


def portable_blender_path(*, localappdata: str | None = None) -> Path:
    """Well-known portable layout after ``scripts/bootstrap-tools.ps1``.

    ``%LOCALAPPDATA%\\MeshOps\\tools\\blender-5.2.0\\blender.exe``
    """
    base = (
        localappdata if localappdata is not None else os.environ.get("LOCALAPPDATA", "")
    ).strip()
    if not base:
        # Fallback when LOCALAPPDATA unset (unusual on Windows)
        base = str(Path.home() / "AppData" / "Local")
    return Path(base) / "MeshOps" / "tools" / PORTABLE_BLENDER_DIR_NAME / PORTABLE_BLENDER_EXE_NAME


# Import-time snapshot + public constant. Tests monkeypatch
# WELL_KNOWN_PORTABLE_BLENDER; production always prefers live LOCALAPPDATA recompute.
_PORTABLE_AT_IMPORT = portable_blender_path()
WELL_KNOWN_PORTABLE_BLENDER = _PORTABLE_AT_IMPORT


def _portable_candidate() -> Path:
    """Portable Blender path: live LOCALAPPDATA unless tests override the constant.

    Production always recomputes from current ``LOCALAPPDATA``. Tests that
    ``monkeypatch.setattr(WELL_KNOWN_PORTABLE_BLENDER, …)`` get the override
    because the bound value no longer equals the import-time snapshot.
    """
    if WELL_KNOWN_PORTABLE_BLENDER != _PORTABLE_AT_IMPORT:
        return WELL_KNOWN_PORTABLE_BLENDER
    return portable_blender_path()


def find_blender_with_source(*, require: bool = True) -> tuple[Path | None, BlenderSource]:
    """Locate Blender and report which candidate tier hit.

    Order:
      1. ``MESHOPS_BLENDER`` env (file path) → source ``env``
      2. ``shutil.which("blender")`` → source ``path``
      3. Windows Program Files well-known 5.2 LTS → source ``well_known``
      4. Portable MeshOps tools path (bootstrap) → source ``portable``
    """
    candidates: list[tuple[Path, BlenderSource]] = []

    env = os.environ.get(ENV_MESHOPS_BLENDER, "").strip()
    if env:
        candidates.append((Path(env), "env"))

    which = shutil.which("blender")
    if which:
        candidates.append((Path(which), "path"))

    portable = _portable_candidate()

    if os.name == "nt":
        candidates.append((WELL_KNOWN_WINDOWS_BLENDER, "well_known"))
        candidates.append((portable, "portable"))
    else:
        # Non-Windows: still check portable / test override
        if portable.as_posix() not in ("", "."):
            candidates.append((portable, "portable"))

    for cand, source in candidates:
        try:
            p = cand.expanduser().resolve(strict=False)
        except OSError:
            p = cand.expanduser()
        if p.is_file():
            return p, source

    if require:
        portable_hint = str(portable)
        hint = (
            f"Set {ENV_MESHOPS_BLENDER} to blender.exe, install Blender 5.2 LTS, "
            "run .\\scripts\\bootstrap-tools.ps1 (portable under "
            f"{portable_hint}), or complete track 0010 doctor/mirror bootstrap. "
            f"Well-known path checked: {WELL_KNOWN_WINDOWS_BLENDER}"
        )
        raise EscalateError(
            f"Blender 5.2 LTS not found. {hint}",
            code="blender_missing",
            details={
                "env": env or None,
                "which": which,
                "well_known": str(WELL_KNOWN_WINDOWS_BLENDER),
                "portable": portable_hint,
            },
        )
    return None, "missing"


def find_blender(*, require: bool = True) -> Path | None:
    """Locate Blender executable (stable API — see ``find_blender_with_source``)."""
    path, _source = find_blender_with_source(require=require)
    return path
