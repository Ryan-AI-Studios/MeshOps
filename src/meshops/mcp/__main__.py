"""stdio MCP entrypoint: ``python -m meshops.mcp`` / ``meshops-mcp``.

Import gate (R4): missing mcp → stderr install hint + exit 1.
Stdout law (R2): logging to stderr only; never print to stdout.
"""

from __future__ import annotations

import logging
import sys


def main() -> None:
    """Launch MeshOps MCP server on stdio."""
    try:
        import mcp  # noqa: F401
        from mcp.server import MCPServer  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "Error: MCP package not installed.\n"
            "Run: uv sync --extra mcp  (meshops[mcp] → mcp==2.0.0)\n"
        )
        raise SystemExit(1) from None

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    from meshops.mcp.server import build_server

    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
