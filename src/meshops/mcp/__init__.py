"""MeshOps MCP adapter — thin stdio server over engine package APIs (track 0008).

Install: ``uv sync --extra mcp`` (pins ``mcp==2.0.0``).
Run: ``meshops-mcp`` or ``python -m meshops.mcp`` (cwd = repo; MESHOPS_WORK optional).
"""

from __future__ import annotations

from meshops.mcp.server import (
    SERVER_INSTRUCTIONS,
    TOOL_NAMES,
    build_server,
    resolve_work_root,
)

__all__ = [
    "SERVER_INSTRUCTIONS",
    "TOOL_NAMES",
    "build_server",
    "resolve_work_root",
]
