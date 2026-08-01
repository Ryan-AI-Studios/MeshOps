"""MESHOPS_* environment variable catalog — single source of truth (R12/R15).

Covers every env var read via ``os.environ`` under ``src/`` plus bootstrap-only
keys. Does **not** catalog stdout protocol tokens (MESHOPS_DESIGN_OK / ERR*).
"""

from __future__ import annotations

from typing import Final, TypedDict


class EnvCatalogEntry(TypedDict):
    name: str
    description: str
    example: str
    consumer: str


ENV_CATALOG: Final[tuple[EnvCatalogEntry, ...]] = (
    {
        "name": "MESHOPS_BLENDER",
        "description": "Absolute path to blender.exe (5.2 LTS pin)",
        "example": r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
        "consumer": "meshops.escalate.discover",
    },
    {
        "name": "MESHOPS_ORCA",
        "description": "Absolute path to orca-slicer.exe",
        "example": r"C:\Program Files\OrcaSlicer\orca-slicer.exe",
        "consumer": "meshops.slice.discover",
    },
    {
        "name": "MESHOPS_ORCASLICER",
        "description": "Alias for MESHOPS_ORCA (absolute path to orca-slicer.exe)",
        "example": r"C:\Program Files\OrcaSlicer\orca-slicer.exe",
        "consumer": "meshops.slice.discover",
    },
    {
        "name": "MESHOPS_ORCA_PROFILES",
        "description": "Optional root directory of named Orca profile folders",
        "example": r"C:\Users\you\orca-profiles",
        "consumer": "meshops.slice.profiles",
    },
    {
        "name": "MESHOPS_ORCA_DATADIR",
        "description": "Optional Orca system datadir (vendor preset trees)",
        "example": r"C:\Program Files\OrcaSlicer\resources",
        "consumer": "meshops.slice.profiles",
    },
    {
        "name": "MESHOPS_ORCA_MACHINE",
        "description": "Absolute path override for machine.json profile",
        "example": r"C:\profiles\machine.json",
        "consumer": "meshops.slice.profiles",
    },
    {
        "name": "MESHOPS_ORCA_PROCESS",
        "description": "Absolute path override for process.json profile",
        "example": r"C:\profiles\process.json",
        "consumer": "meshops.slice.profiles",
    },
    {
        "name": "MESHOPS_ORCA_FILAMENT",
        "description": "Absolute path override for filament.json profile",
        "example": r"C:\profiles\filament.json",
        "consumer": "meshops.slice.profiles",
    },
    {
        "name": "MESHOPS_ORCA_TIMEOUT_S",
        "description": "OrcaSlicer subprocess timeout seconds (slice runner)",
        "example": "600",
        "consumer": "meshops.slice.runner",
    },
    {
        "name": "MESHOPS_ROGUE2_PATH",
        "description": "Absolute path to Rogue2.stl hero fixture (tests)",
        "example": r"C:\data\Rogue2.stl",
        "consumer": "tests / fixtures",
    },
    {
        "name": "MESHOPS_STUB_DIFF",
        "description": "When set (truthy), stub multi-view diffs for CI (1/true/yes)",
        "example": "1",
        "consumer": "recipes / design / escalate / organic",
    },
    {
        "name": "MESHOPS_WORK",
        "description": (
            "Job/session store root for MCP server (default ./work; expanduser+resolve)"
        ),
        "example": r"C:\dev\stl\work",
        "consumer": "meshops.mcp.server (server-bound work_root)",
    },
    {
        "name": "MESHOPS_ORGANIC_TIMEOUT_S",
        "description": "Organic Blender pass timeout seconds (default 300; 600+ high-res)",
        "example": "600",
        "consumer": "meshops.organic.pass_runner",
    },
    {
        "name": "MESHOPS_BLENDER_MIRROR",
        "description": (
            "Preferred Blender zip URL for bootstrap only — never read by find_blender"
        ),
        "example": (
            "https://ftp.halifax.rwth-aachen.de/blender/release/Blender5.2/"
            "blender-5.2.0-windows-x64.zip"
        ),
        "consumer": "scripts/bootstrap-tools.ps1 (bootstrap-only)",
    },
    {
        "name": "MESHOPS_BOOTSTRAP_DIR",
        "description": "Optional tools cache root for portable Blender extract",
        "example": r"%LOCALAPPDATA%\MeshOps\tools",
        "consumer": "scripts/bootstrap-tools.ps1 (bootstrap-only)",
    },
)

ENV_CATALOG_BY_NAME: Final[dict[str, EnvCatalogEntry]] = {e["name"]: e for e in ENV_CATALOG}

# Names that appear as MESHOPS_* strings in src but are stdout protocol tokens, not env.
STDOUT_PROTOCOL_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "MESHOPS_DESIGN_OK",
        "MESHOPS_DESIGN_ERR",
        "MESHOPS_DESIGN_STL_BYTES",
        "MESHOPS_DESIGN_STEP_BYTES",
    }
)


def catalog_names() -> frozenset[str]:
    """All cataloged MESHOPS_* env var names."""
    return frozenset(ENV_CATALOG_BY_NAME)


def env_presence_map() -> dict[str, bool]:
    """Return whether each cataloged key is set (non-empty) in the process env."""
    import os

    out: dict[str, bool] = {}
    for name in ENV_CATALOG_BY_NAME:
        out[name] = bool(os.environ.get(name, "").strip())
    return out


def catalog_as_list() -> list[EnvCatalogEntry]:
    """JSON-serializable list of catalog entries (copy)."""
    return [dict(e) for e in ENV_CATALOG]  # type: ignore[misc]
