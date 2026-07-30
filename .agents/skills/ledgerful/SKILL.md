---
name: ledgerful
description: Use this skill when making code edits, reviews, impact/risk analysis, verification planning, drift handling, ledger provenance, or deciding what tests to run. Before meaningful edits, run ledgerful doctor then scan/impact; after edits, run verification and report unresolved drift or ledger state. Prefer --json for machine-readable agent output.
---

# Ledgerful

Use Ledgerful as the local safety layer and engineering intelligence engine for code changes. The
canonical CLI is **`ledgerful`** (short alias **`ldg`** may be present from the shell installer). It
provides impact analysis, hotspot and temporal-coupling signals, verification planning, and
transactional provenance.

## Core Capabilities

- **Search & Discovery**: High-performance regex (Tantivy), optional SCIP edge augment (`index --auto-scip`, off by default), and conceptual semantic search (local embeddings) with parallel HNSW retrieval.
- **Code Symbol Index**: Tree-sitter parsing of Rust, TypeScript, and Python — extracts every public function, struct, enum, trait, module, and HTTP route into the Knowledge Graph. Queryable via `ledgerful search` and `ledgerful ask`.
- **Gemini Token Budgeting**: Automatically calculates character limits based on `config.gemini.context_window`. Appends `[Packet truncated for Gemini submission]` when limits are hit to ensure predictable LLM behavior.
- **Route Extraction**: Detects HTTP routes from Axum, Express, and other frameworks. Stores `method`, `path_pattern`, `handler_name`, `framework`, and confidence score.
- **Call Graph**: Tracks function call relationships (`Direct`, `MethodCall`, `TraitDispatch`, `Dynamic`, `External`) so you can answer "what calls this function?" and "what does this function depend on?".
- **Knowledge Graph**: Durable, billion-edge relational and vector storage (CozoDB-redux/Sled) with native code-aware tokenization (Tree-Sitter). Stores symbols in `project_symbol` table.
- **AI-Brains Bridge**: Exports hotspots, ledger entries, and MADR data to AI-Brains via `ledgerful bridge export --hotspots --ledger [--madr] [--stdout]`. AI-Brains nightly pipeline ingests this output as code symbols into recall (T70). Inbound recall uses `ledgerful bridge query "<text>"` (IPC with CLI fallback).
- **Impact Analysis**: Deep "blast radius" analysis across 20+ specialized providers (Infra, Contracts, Observability, Temporal).
- **Cryptographic Provenance**: Mathematical proof of intent via Ed25519 signing of every ledger entry. Offline verification via `verify --signatures`.
- **Intent Capture TUI**: Interactive terminal UI for auditing and refining LLM-drafted intent payloads during the git commit process.
- **Real-time Sync (watch)**: Incremental Knowledge Graph updates, AST re-parsing, and code-aware symbol indexing via the `watch` command — **not** team ledger sync.
- **Predictable Verification**: Bayesian test reordering and CI failure prediction.
- **Documentation Generation**: Export Knowledge Graph data to Markdown/Mermaid passive documentation (`index --export-docs`).
- **Dead Code Detection**: Confidence-based dead code detection blending graph reachability, git activity, and test history (`dead-code` command). Use `dead-code --prune` for interactive opt-in removal wrapped in a pending ledger transaction.
- **Scoped Verification**: `ledgerful verify --scope fast` uses the `test_mapping` index to run only the tests covering changed files (nextest filtersets), falling back to the full suite when shared infrastructure is touched. The pre-push hook uses `--scope fast`; CI uses `--scope full`. See `docs/testing.md`.
- **Nightly Scheduler**: Cross-platform nightly indexing via `ledgerful schedule setup-nightly` (Windows schtasks / Unix crontab), with `--dry-run` and `--uninstall`. Runs `git fetch` + `index --analyze-graph` sequentially, logging to `.ledgerful/logs/nightly.log`.
- **Live Visualization**: WebSocket-based Arc Diagram for real-time Knowledge Graph updates (`viz-server`, `viz-server --stop`).
- **Endpoints**: Indexed endpoint graph with auth, schemas, consumers, and owner links. `ledgerful endpoints --json` / `--changed` for direct review.
- **Services Diff**: Declared service map with queue/topic/RPC edges and PR-style boundary diff. `ledgerful services diff`.
- **Data Models**: Durable data model, table, migration, and compatibility-class relations with impact rules for destructive changes. `ledgerful data-models impact --changed`.
- **Config Schema & Diff**: Explicit env var schema metadata (required/secret/owner/provider) and change diff. `ledgerful config schema` / `ledgerful config diff`.
- **Dependency & Advisory Graph**: Cargo/npm/Python lockfile ingestion with cargo-audit/osv advisory matching. Impact rules for vulnerable dependency introduction.
- **Test Mapping**: Durable test nodes linked to endpoints, symbols, services, and data models. `ledgerful verify --explain --entity <path>` for entity-scoped test explanation.
- **Observability Graph**: SLO, metric, alert, and signal nodes from OpenSLO YAML. Source-file-backed diff matching. `ledgerful observability diff` / `observability coverage`.
- **Hotspot Trends**: Persistent hotspot and temporal coupling snapshots with trend deltas. `ledgerful hotspots trend` / `hotspots explain`.
- **Ledger Graph**: Per-transaction entity neighborhood view linking ledger entries to symbols, endpoints, services, ADRs, config keys, and deploy surfaces. `ledgerful ledger graph <tx-id>`.
- **Ledger Validator Lifecycle**: Full validator lifecycle with `ledger validator list`, `disable`, `enable`, `remove`, `doctor`, and hook-repair rollback for sidecar/pending mismatches.
- **Security Boundaries**: Cedar policy parsing with cross-surface links (policy→endpoint/service/config_key/deploy_surface/ADR). `ledgerful security boundaries` / `security impact --changed`.
- **Team Sync [Experimental]**: Opt-in encrypted ledger entry bundles via `ledgerful sync` (default feature; `[sync].enabled = false` forever until you opt in). Pairing is real (`LF-PAIR-1` invite + `sync pair` accept/list/revoke; never auto-enables). Multi-device apply polish / Available land in 0112–0113. See `docs/team-sync.md`. Not the same as watch “Real-time Sync”.



## Philosophy: CLI-First Intelligence

Ledgerful is a **CLI-first** tool. Prefer the `ledgerful` binary (or `ldg` if that alias is on
`PATH`) for discovery and safety. MCP (`ledgerful mcp`) and the local dashboard
(`ledgerful web start`) are optional surfaces on default builds.

**Machine-readable output:** many commands accept **`--json`** (and related format flags). Agents
should prefer `--json` when parsing stdout so human progress lines do not break consumers. Pair with
quiet/machine expectations documented in `docs/operator-surface-policy.md` when present.

**Index freshness (three tiers):** full policy in `docs/index-freshness-policy.md`.

- **Light on-demand (shared path):** `--auto-index` on **`search`, `ask`, `hotspots`,
  `dead-code`**. Triggers on time-stale, never-indexed (full bootstrap), or content-hash
  **drift-stale** (same-day dirty worktree). Never SCIP / never silent `--analyze-graph`.
- **`verify --auto-index` is different:** only refreshes stale/empty `test_mapping` for
  `--scope fast` — not the shared drift/bootstrap path. Use search/ask/hotspots/dead-code or
  `index` for a general symbol floor.
- **`scan` / `scan --impact` have no `--auto-index`** — run `doctor` / `index --check` /
  `index --incremental` (or a prior auto-index command) first when freshness matters.
- **Light continuous:** `ledgerful watch` (foreground). Mega-batch ≥1000 unique paths skips
  unbounded incremental, marks STALE, asks for `index --full`.
- **Heavy:** `schedule setup-nightly` / `index --full` / explicit `--auto-scip`. Default nightly
  argv is `index --analyze-graph` only (no `--auto-scip`).
- **Never idle SCIP.** `init` installs no watcher/schedule. **Doctor green ≠ index fresh.**

## Git worktrees

Linked worktrees share ledger state with the primary worktree's `.ledgerful` (same pending TX and `ledger.db`). Run `ledgerful` commands from the worktree cwd; do not copy state into the linked tree. Submodules keep their own `.ledgerful`. Set absolute `LEDGERFUL_STATE_DIR` to override.

## Default Workflow

1. Confirm the tool is healthy (first step — same pre-condition as `AGENTS.md`):

   ```bash
   # Prefer --json when parsing; branch on readyForPublish (zero block findings).
   ledgerful doctor --json
   # Human: ledgerful doctor
   ```

   **`readyForPublish == true`** means the publish-environment path is fit to enter
   (no lifecycle/tool **block** findings). It does **not** mean `verify` passed,
   tests are green, or CI is green. Optional backends (embedding/completion/SCIP/
   sccache/gemini) never set `readyForPublish=false`.

2. Check provenance / drift before editing:

   ```bash
   ledgerful audit
   ledgerful ledger status --compact
   # or machine-readable:
   ledgerful ledger status --json
   ```

3. Before meaningful code edits, assess impact:

   ```bash
   ledgerful scan --impact
   # CI / fixed base (not working tree): ledgerful scan --base-ref origin/main --impact
   # Machine payload: ledgerful scan --impact --json
   ```

4. Read `.ledgerful/reports/latest-impact.json` when it exists, but treat it
   as a cached artifact rather than ground truth. Validate that its
   `headHash`, `treeClean`, and `timestampUtc` still match the current repo
   state before relying on it for risk level, hotspots, temporal couplings,
   affected symbols, runtime dependencies, or verification hints.

5. Make the smallest scoped change that satisfies the task.

6. After edits, run:

   ```bash
   ledgerful verify
   # scoped local gate (pre-push style): ledgerful verify --scope fast
   ```

   Also run any repo-specific tests needed for the touched files.

7. For final gates, avoid overlapping `cargo`, `nextest`, or `ledgerful
   verify` jobs. Parallel read-only inspection is fine, but final verification
   should run sequentially to avoid Windows file-lock and linker contention.

8. Report the outcome: impact/risk signals used, verification run, and any
   unresolved pending transactions, drift, or unavailable Ledgerful command.

## Code Symbol Queries — Use These First

Before searching the web or reading files manually, query Ledgerful's symbol index. It knows every public function, struct, route, and call edge in the codebase.

```bash
# Refresh the index (incremental), or pass --auto-index on the query command below
ledgerful index --incremental

# Optional SCIP edge augment (OFF by default). Use when you need higher-precision
# call edges for impact / blast-radius prep, not as a universal quality KPI.
# Requires a capable indexer (capability probe). Adds structural_edges with
# evidence=scip:ref onto native symbols only — does not replace the native index.
# Under --json, read scip.status, scip.edges_added, scip.edges_updated.
# ledgerful index --auto-scip --json
# ledgerful index --scip path/to/index.scip --json

# Find a function, struct, or type by name
ledgerful search "handleGetUser" --auto-index
ledgerful search "AuthMiddleware"

# Find HTTP routes
ledgerful search "POST /auth"
ledgerful ask "list all HTTP GET route handlers"

# Find what calls a function
ledgerful ask "what calls validateToken"
ledgerful ask "show callers of UserRepository::find_by_id"

# Find all public endpoints
ledgerful ask "find all Axum route handlers"
ledgerful ask "what API endpoints are defined in src/routes"

# Dead code
ledgerful dead-code --threshold 0.75

# Dead code — show everything including standard traits (Eq, Clone, Debug, …)
# By default, standard trait symbols are EXCLUDED because they are used implicitly
# via derive macros or blanket impls and almost always produce false positives.
ledgerful dead-code --include-traits
```

> **Heuristic note**: Dead code analysis blends graph reachability, git inactivity, and
> test coverage. Results are probabilistic, not definitive. Common false-positive patterns:
> - Traits derived via `#[derive(...)]` (Eq, Ord, Clone, Debug, Serialize, …) — suppressed by default.
> - Types ending in `Provider`, `Chunk`, `Record`, `Result` — receive a -0.20 confidence penalty
>   (they are often dispatched dynamically or through serde).
> Use `--include-traits` to restore unfiltered output for auditing purposes.

These queries work because Ledgerful indexes:
- Every `pub fn`, `pub struct`, `pub enum`, `pub trait` via tree-sitter
- HTTP route registrations (Axum `Router::route`, Express `app.get`, etc.)
- Function call edges via static analysis (native call graph)
- Optional SCIP reference edges (`evidence=scip:ref`) when you opt in with `--auto-scip` / `--scip`

Symbols ingested by the bridge become AI-Brains memories (T70) and are returned
by `ai-brains recall "<topic>"` alongside session memories. To verify the
bridge is alive end-to-end, run `ai-brains preflight --summary` and confirm
hotspots and decisions are listed.

## Audit Smoke Tests

When reviewing CLI/config behavior, supplement unit tests with command-level
smoke tests against the current build output, usually `target\debug\ledgerful.exe`
on Windows. Prefer focused temporary repositories and verify failure cases as
well as success cases.

Useful checks include:

- JSON mode remains parseable on failure paths (`config verify --json`, invalid
  `config.toml`, invalid `rules.toml`, unknown `--section`).
- Dry-run commands do not create persistent state or perform external probes
  unless that is explicitly part of the dry-run contract.
- Requested vs effective config values are visible when runtime clamping or
  defaults change the final behavior.
- Internal callsites that construct CLI argument structs still populate new
  fields explicitly.

## Repository Configuration

Ledgerful's `.ledgerful/rules.toml` and `.ledgerful/config.toml` are
repo-local policy, not portable defaults. When installing or copying this skill
into another repository, review and update:

- `required_verifications`: use commands that actually exist in that repo
  rather than aliases such as `lint`, `test`, or `build` unless the repo defines
  those commands.
- `verify.default_timeout_secs`: set a timeout that fits the repo's slowest
  expected verification command.
- `protected_paths`: keep enforcement scoped to paths that make sense for the
  repository.

If `ledgerful verify` fails with "Command not found" or times out while the
same command passes manually, fix the repo-local config before treating it as a
code failure.

`ledgerful init` sanitizes every starter template before creating
`.ledgerful/config.toml`. Secret-bearing keys and credentialed connection
URLs are omitted, including values from `LEDGERFUL_DEFAULT_CONFIG` and
`~/.ledgerful/default-config.toml`. Keep credentials in the environment or
an ignored repo-local `.env` (`GEMINI_API_KEY`, `OLLAMA_CLOUD_API_KEY`, or the
legacy `OLLAMA_API_KEY`); Ledgerful does not interpolate `${VAR}` expressions
inside TOML.

## Dependency Alert Workflow

For Dependabot or audit findings:

- Identify whether the vulnerable crate is direct or transitive with
  `cargo tree -i <crate>@<version>`.
- If the vulnerable crate is transitive through a direct dependency, prefer
  upgrading the direct dependency over adding a downstream patch.
- If the vulnerable path enters through a git dependency, verify whether the
  upstream fix is visible to downstream consumers. Workspace-level
  `[patch.crates-io]` entries in the dependency repository are not transitive.
- Record external remediation handoffs in a conductor track when another repo
  owns the durable fix.
- After dependency changes, run focused dependency checks plus `ledgerful
  verify`.

## When To Skip

Skip Ledgerful only for trivial formatting, simple dependency lockfile updates,
binary/media changes, temporary scratch files, or when the user explicitly says
to bypass it.

## If Commands Fail

- If `ledgerful` is unavailable, continue with normal repo tools and tell the
  user Ledgerful signals were unavailable.
- If `ledger status` shows unaudited drift, reconcile or adopt before continuing
  unless the user directs otherwise.
- If `scan --impact` cannot complete, continue cautiously and include the error
  in the final report.
- If a command reports that the index is `[STALE]`, append `--auto-index` to **`search`, `ask`,
  `hotspots`, `dead-code`** (prefer proactively). Time-stale and content-hash drift both refresh;
  never-indexed runs full bootstrap. Use `index --check --json` / `doctor --json` for readiness.
- **`verify --auto-index` only repairs `test_mapping` for `--scope fast`** — not general index
  bootstrap. Before **`scan --impact`**, refresh first — that command has no `--auto-index`.
- Prefer **`--json`** when an agent must parse command output.
- Do not edit `.ledgerful/` state files directly.
- Doctor / `readyForPublish` green does **not** mean the search/graph index is fresh.

## Ledger Provenance

For tracked manual edits:

```bash
ledgerful ledger start <entity> --category <CAT> --message "Intent"
# edit files
ledgerful ledger commit <tx-id> --summary "Done" --reason "Why"
```

For surgical one-command provenance:

```bash
ledgerful ledger atomic <entity> --category <CAT> --summary "Task" --reason "Goal"
```

For lightweight notes or lessons learned:

```bash
# Both positional and --message formats are supported
ledgerful ledger note <entity> "Note content"
ledgerful ledger note <entity> --message "Note content"
```

### Git Hook Lifecycle (Milestone O)

Ledgerful uses a two-phase commit lifecycle to ensure zero phantom records:
1. **`commit-msg`**: Launches the TUI to capture intent. Creates a `PENDING` transaction and a sidecar file.
2. **`post-commit`**: Automatically promotes the `PENDING` transaction to `COMMITTED` once the Git commit is finalized. If the Git commit fails, the record remains pending or is safely rolled back on the next attempt.

### Cryptographic Security

If `intent.require_signing = true` is set in `.ledgerful/config.toml`, all ledger entries must be signed by the developer's local Ed25519 key (generated during `init`).

To verify the integrity of the entire ledger:
```bash
ledgerful verify --signatures
```
This performs an offline mathematical validation of every record against its signature and public key.

## Publish Hygiene

**Dual green (do not collapse):**

| Signal | Means | Does **not** mean |
|---|---|---|
| `doctor` / `readyForPublish` | Zero **block** doctor findings; env fit to enter publish path | Verify/tests/CI green |
| Pre-push hook | `verify --scope fast` + ledger cleanliness | Full fmt/clippy/nextest/CI |
| `verify --scope full` / CI | Repo full gate | Doctor readiness |

When asked to push, catch up `main`, or prune branches:

1. Fetch current remote state first:

   ```powershell
   git fetch --all --prune
   git rev-list --left-right --count origin/main...HEAD
   ```

2. If `origin/main` moved, reconcile before staging or pushing. Do not rebase or
   reset over user work without explicit direction.

3. Confirm publish-env readiness (optional parse):

   ```bash
   ledgerful doctor --json   # branch on .readyForPublish
   ```

4. Stage only the intended scope, commit, then push:

   ```powershell
   git push origin main
   ```

   The pre-push hook runs `ledgerful verify --scope fast` (scoped test
   selection via `test_mapping`) plus `ledgerful ledger status`; treat
   that as the authoritative publish gate and report its result. For the
   full suite, run `ledgerful verify --scope full` manually or in CI.
   Doctor green ≠ pre-push green ≠ full CI.

5. Prune conservatively:

   ```powershell
   git remote prune origin --dry-run
   git branch --merged main
   ```

   Delete local branches only when they are listed as merged into `main` and are
   not the active branch. Branch pruning can legitimately be a no-op.

## Reasoning Rules

- If temporal coupling is above 70% for an unchanged file, inspect that file.
- If hotspots are reported, bias verification toward those files first.
- If KG reachability identifies downstream nodes, inspect them before finalizing.
- Treat hooks and CI gates as enforcement. Treat this skill as guidance.

## Maintenance & Upgrades

To keep your Ledgerful environment synchronized with the latest engine features:

```bash
# Safely migrate repository state (clears indices, preserves ledger)
ledgerful update --migrate --force

# Rebuild indices after migration
ledgerful index --semantic
```

## Working On Ledgerful Itself

After changing Ledgerful source code, you can use the built-in update command to reinstall the global binary:

```bash
ledgerful update --binary
```

Alternatively, run manually from the source root:

```bash
cargo install --path .
```

Treat the install step as part of done criteria after Ledgerful source edits,
before publishing or handing the work back.

## Cross-Model Review Notes

For high-risk diffs, a read-only `codex exec` review can be useful before final
verification. In non-interactive Windows/PowerShell runs, redirect stdin from
`NUL` so the process does not wait for input:

```powershell
cmd /c "codex exec -C ""C:\dev\Ledgerful"" -s read-only -m gpt-5.4 -o output\review.md ""Review the current diff for regressions. Do not modify files."" < NUL"
```

If the command appears stuck, inspect the output file before waiting longer; the
review may already have written useful findings.

## References

- Command details: `references/commands.md` (agent-oriented subset + categories; use `ledgerful --help` for full surface)
- Install fallback: `references/install.md`
- Architecture/internal notes: `references/internals.md`
- MCP server: `references/mcp.md`
- Local web dashboard: `references/web.md`
