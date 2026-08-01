"""OrcaSlicer binary discovery (env → PATH → well-known Windows).

Install / doctor ritual belongs to track 0010 — not reimplemented here.
Version probe: AppData conf only pre-slice — never ``--help`` (silent on Windows).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from meshops.slice.errors import SliceError

# TechStack pin: OrcaSlicer 2.4.2 (subprocess only).
WELL_KNOWN_WINDOWS_ORCA = Path(r"C:\Program Files\OrcaSlicer\orca-slicer.exe")

ENV_MESHOPS_ORCA = "MESHOPS_ORCA"
ENV_MESHOPS_ORCASLICER = "MESHOPS_ORCASLICER"

# Soft pin major.minor — warn if older; hard fail only when require_version.
SOFT_MAJOR = 2
SOFT_MINOR = 4

_CONF_VERSION_RE = re.compile(r'"version"\s*:\s*"([^"]+)"')


def find_orca(*, require: bool = False) -> Path | None:
    """Locate OrcaSlicer executable.

    Order:
      1. ``MESHOPS_ORCA`` env (file path)
      2. ``MESHOPS_ORCASLICER`` env (alias)
      3. ``shutil.which("orca-slicer")`` / ``which("orcaslicer")``
      4. Windows well-known: ``C:\\Program Files\\OrcaSlicer\\orca-slicer.exe``

    When *require* is True, missing binary raises
    ``SliceError(code="orca_not_found")``. Default False returns None.
    """
    candidates: list[Path] = []

    for env_name in (ENV_MESHOPS_ORCA, ENV_MESHOPS_ORCASLICER):
        env = os.environ.get(env_name, "").strip()
        if env:
            candidates.append(Path(env))

    for name in ("orca-slicer", "orcaslicer"):
        which = shutil.which(name)
        if which:
            candidates.append(Path(which))

    if os.name == "nt":
        candidates.append(WELL_KNOWN_WINDOWS_ORCA)

    seen: set[str] = set()
    for cand in candidates:
        try:
            p = cand.expanduser().resolve(strict=False)
        except OSError:
            p = cand.expanduser()
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p

    if require:
        hint = (
            f"Set {ENV_MESHOPS_ORCA} to orca-slicer.exe, install OrcaSlicer 2.4.x, "
            "or complete track 0010 doctor/mirror bootstrap. "
            f"Well-known path checked: {WELL_KNOWN_WINDOWS_ORCA}"
        )
        raise SliceError(
            f"OrcaSlicer not found. {hint}",
            code="orca_not_found",
            details={
                "env_orca": os.environ.get(ENV_MESHOPS_ORCA) or None,
                "env_orcaslicer": os.environ.get(ENV_MESHOPS_ORCASLICER) or None,
                "well_known": str(WELL_KNOWN_WINDOWS_ORCA),
            },
        )
    return None


def orca_appdata_conf_path() -> Path | None:
    """Return ``%AppData%\\Roaming\\OrcaSlicer\\OrcaSlicer.conf`` if it exists."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            conf = Path(appdata) / "OrcaSlicer" / "OrcaSlicer.conf"
            if conf.is_file():
                return conf
    # Linux / macOS common locations
    home = Path.home()
    for rel in (
        Path(".config") / "OrcaSlicer" / "OrcaSlicer.conf",
        Path("Library") / "Application Support" / "OrcaSlicer" / "OrcaSlicer.conf",
    ):
        conf = home / rel
        if conf.is_file():
            return conf
    return None


def read_orca_version_from_appdata() -> str | None:
    """Pre-slice version probe from OrcaSlicer.conf ``"version"`` key.

    Never uses ``--help`` (host-verified silent on Windows).
    """
    conf = orca_appdata_conf_path()
    if conf is None:
        return None
    try:
        text = conf.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # Prefer JSON parse when the file is valid JSON (or JSON-like object).
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            ver = data.get("version")
            if isinstance(ver, str) and ver.strip():
                return ver.strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    m = _CONF_VERSION_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def soft_version_ok(version: str | None) -> bool:
    """Return True if version is missing or major.minor >= 2.4 (soft pin)."""
    if not version:
        return True
    m = re.match(r"(\d+)\.(\d+)", version.strip())
    if not m:
        return True
    major, minor = int(m.group(1)), int(m.group(2))
    if major > SOFT_MAJOR:
        return True
    if major < SOFT_MAJOR:
        return False
    return minor >= SOFT_MINOR
