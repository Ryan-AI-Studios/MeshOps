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
        "name": "MESHOPS_HOSTED_API_KEY",
        "description": (
            "Primary API key for hosted multi-view generator (Meshy v1 default; "
            "presence-only — never remote-validated by doctor)"
        ),
        "example": "(secret — never commit)",
        "consumer": "meshops.hosted.orchestrate",
    },
    {
        "name": "MESHOPS_MESHY_API_KEY",
        "description": (
            "Optional Meshy-specific key override (takes precedence over "
            "MESHOPS_HOSTED_API_KEY when provider=meshy)"
        ),
        "example": "(secret — never commit)",
        "consumer": "meshops.hosted.orchestrate",
    },
    {
        "name": "MESHOPS_TRIPO_API_KEY",
        "description": "Optional Tripo key for future adapter (not v1 primary)",
        "example": "(secret — never commit)",
        "consumer": "meshops.hosted.orchestrate",
    },
    {
        "name": "MESHOPS_HOSTED_POLL_INTERVAL_S",
        "description": "Seconds between hosted provider status polls (default 5)",
        "example": "5",
        "consumer": "meshops.hosted.orchestrate",
    },
    {
        "name": "MESHOPS_HOSTED_TIMEOUT_S",
        "description": "Overall hosted poll deadline seconds (default 300)",
        "example": "300",
        "consumer": "meshops.hosted.orchestrate",
    },
    {
        "name": "MESHOPS_HOSTED_MAX_HTTP_RETRIES",
        "description": "Max HTTP retries for 429/503 hosted provider calls (default 3)",
        "example": "3",
        "consumer": "meshops.hosted.providers.meshy",
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
    {
        "name": "MESHOPS_BENCH_SOAK",
        "description": (
            "Truthiness gate for soak tests (1/true/yes/on). Marker @pytest.mark.slow "
            "alone is insufficient — default pytest never long-soaks (0009 C2)."
        ),
        "example": "1",
        "consumer": "tests/test_bench_soak.py / meshops.bench",
    },
    {
        "name": "MESHOPS_BENCH_SIZES",
        "description": "Optional default size ladder override for meshops bench run (e.g. S,M)",
        "example": "S,M",
        "consumer": "meshops.bench.runner",
    },
    {
        "name": "MESHOPS_BENCH_WORK_ROOT",
        "description": "Optional results/jobs root for meshops bench (default work/bench)",
        "example": r"C:\dev\stl\work\bench",
        "consumer": "meshops.bench.runner / meshops.cli bench",
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
