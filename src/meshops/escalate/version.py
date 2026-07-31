"""Blender version probe — require major.minor == 5.2 (TechStack pin)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from meshops.escalate.errors import EscalateError

# TechStack: Blender 5.2 LTS only. 4.2 is EOL Jul 2026 — fail closed.
REQUIRED_MAJOR = 5
REQUIRED_MINOR = 2

_VERSION_RE = re.compile(
    r"Blender\s+(\d+)\.(\d+)(?:\.(\d+))?",
    re.IGNORECASE,
)


def parse_blender_version(text: str) -> tuple[int, int, int, str]:
    """Parse ``blender --version`` stdout into (major, minor, patch, raw_line)."""
    for line in text.splitlines():
        m = _VERSION_RE.search(line)
        if m:
            major = int(m.group(1))
            minor = int(m.group(2))
            patch = int(m.group(3) or 0)
            return major, minor, patch, line.strip()
    raise EscalateError(
        f"could not parse Blender version from output: {text[:200]!r}",
        code="blender_version",
        details={"output_head": text[:500]},
    )


def require_blender_52(blender: Path, *, timeout_s: float = 30.0) -> str:
    """Run ``blender --version``; require 5.2.x; return version string (e.g. '5.2.0').

    Fails with ``blender_version`` on 4.2 or any non-5.2 major.minor.
    """
    try:
        proc = subprocess.run(
            [str(blender), "--version"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise EscalateError(
            f"Blender binary not executable: {blender}",
            code="blender_missing",
            details={"path": str(blender)},
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EscalateError(
            f"Blender --version timed out after {timeout_s}s",
            code="timeout",
            details={"path": str(blender)},
        ) from exc

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise EscalateError(
            f"Blender --version exited {proc.returncode}: {out[:300]!r}",
            code="blender_version",
            details={
                "path": str(blender),
                "returncode": proc.returncode,
                "output_head": out[:500],
            },
        )

    major, minor, patch, raw = parse_blender_version(out)
    version_str = f"{major}.{minor}.{patch}"

    if major != REQUIRED_MAJOR or minor != REQUIRED_MINOR:
        raise EscalateError(
            f"Blender {version_str} found; MeshOps requires 5.2.x LTS "
            f"(got major.minor={major}.{minor}; 4.2 is EOL). Path={blender}",
            code="blender_version",
            details={
                "path": str(blender),
                "version": version_str,
                "raw": raw,
                "required": f"{REQUIRED_MAJOR}.{REQUIRED_MINOR}.x",
            },
        )
    return version_str


def blender_version_string(blender: Path, *, timeout_s: float = 30.0) -> str:
    """Alias for require_blender_52 (version pin enforced)."""
    return require_blender_52(blender, timeout_s=timeout_s)
