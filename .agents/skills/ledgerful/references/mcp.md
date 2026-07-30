# Ledgerful MCP Server

Ledgerful provides a Model Context Protocol (MCP) server that exposes its intelligence as read-only tools for AI coding agents.

## Registration

Prefer a released binary or npm wrapper when possible:

```bash
# PATH binary (default build includes mcp)
ledgerful mcp

# or npm wrapper (downloads pinned engine)
npx @ledgerful/mcp-server
```

### Claude Code
```bash
mcp add ledgerful ledgerful mcp
# from a source checkout instead:
# mcp add ledgerful cargo run --manifest-path /path/to/Ledgerful/Cargo.toml --features mcp -- mcp
```

### Cursor / Windsurf / Cline / Continue
Point the host at either `ledgerful` + args `["mcp"]` or `npx` +
`["@ledgerful/mcp-server"]`. For a source checkout, use `cargo` with
`--manifest-path <repo>/Cargo.toml`, `--features mcp`, and trailing `-- mcp`.

### Aider
```bash
aider --mcp-server "ledgerful mcp"
```

## Tools

1. `scan`: Run impact scan on current repo.
2. `search`: BM25/regex code search.
3. `ask`: Semantic Q&A with context assembly.
4. `ledger_status`: Current pending/unaudited state.
5. `ledger_search`: Full-text search transactions.
6. `hotspots`: Current hotspot rankings.
7. `endpoints_changed`: API endpoints affected by current diff.
8. `security_boundaries`: Security policy graph summary.
9. `dead_code`: Confidence-ranked dead code candidates in the repo.
10. `verify_plan`: Predicted test list for the current diff, without running tests.

## Known Limitations

- No streaming.
- No mutations (read-only v1).

The MCP tool set is a **subset** of the full CLI (no `doctor`, `gate mode`, `config view`, etc. as
MCP tools). Prefer the CLI for those; use MCP when the host only exposes MCP.

## Runtime discovery

```bash
ledgerful mcp --help
# MCP itself is stdio JSON-RPC; list tools via your host's tool-list UI after connect
```

Default builds include the `mcp` feature. Source builds without `--features mcp` (or without
defaults) will not expose the `mcp` subcommand.

## Troubleshooting

- **agent can't find ledgerful on PATH**: Install from [install.md](install.md) or use
  `npx @ledgerful/mcp-server` / `cargo run --features mcp -- mcp`.
- **MCP feature missing**: Rebuild with default features or explicit `--features mcp`.
