# Ledgerful Command Reference

Agent-oriented command sheet (not a complete man page). For the full surface run
`ledgerful --help` / `ledgerful <cmd> --help`. Prefer **`--json`** when an agent must parse output.

## Health & setup

```bash
ledgerful doctor                         # First step: environment / index / config health
ledgerful doctor --json                  # Pure schema-v1 JSON: readyForPublish + findings
ledgerful setup                          # Onboarding wizard
ledgerful update --binary                # Reinstall global binary after engine source edits
ledgerful update --migrate --force       # Migrate state (clears indices, keeps ledger)
```

**`doctor --json`:** branch on **`readyForPublish`** (true iff zero **block** findings).
Optional backends (embed/completion/SCIP/sccache/gemini) never block publish readiness.
`readyForPublish` ≠ verify green — still run `verify --scope fast` (pre-push) / full CI.
Dashboard `doctor-results.json` `failures` = block + non-optional warn (optional excluded).

## Core Commands

### Impact & Scan

```bash
ledgerful scan --impact                  # Before edits: full change intelligence
ledgerful scan --impact --json           # Machine-readable impact packet
ledgerful scan --base-ref origin/main    # Diff vs a git ref (CI), not working-tree status
ledgerful scan --base-ref origin/main --impact
ledgerful scan --pr main...HEAD --format json   # PR-style range (mutually exclusive with --impact)
ledgerful scan --out path/to/report.json # Write JSON to file when supported with --json/--impact
ledgerful impact --all-parents           # Include side-branch commits in coupling analysis
ledgerful impact --summary               # One-line triage: RISK | N changed | N couplings
ledgerful impact --dead-code             # Include dead-code confidence analysis
ledgerful impact --telemetry             # Telemetry coverage analysis
ledgerful impact --json                  # Machine-readable impact
ledgerful impact --out path/to/out.json  # Write output file when supported
```

### Verification

```bash
ledgerful verify                         # Run configured or predicted verification (full scope)
ledgerful verify --scope fast            # Scoped: only tests covering changed files via test_mapping
ledgerful verify --scope full            # Full suite (default; CI should always use this)
ledgerful verify -c "cargo clippy -- -D warnings"   # Manual single command
ledgerful verify --no-predict            # Skip predictive suggestions
ledgerful verify --dry-run               # Show the plan without executing
ledgerful verify --signatures            # Offline Ed25519 verification of ledger records
ledgerful verify --json                  # Machine-readable report (when supported)
ledgerful verify --explain --entity PATH # Entity-scoped test explanation
```

**`--scope fast`** uses the `test_mapping` index to run only the test modules
that cover the changed files, emitting a nextest filterset command (e.g.
`cargo nextest run -E 'test(cli_scan) + test(dead_code_prune)'`). Falls back
to the full suite when shared infrastructure is touched (Cargo.toml,
cli/args.rs, config/**, migrations/**) or when `test_mapping` is empty.
The pre-push hook uses `--scope fast`. See `docs/testing.md` for the full
layered strategy.

### Reset

```bash
ledgerful reset                          # Preserves config, rules, and ledger.db
ledgerful reset --remove-config          # Remove .ledgerful/config.toml
ledgerful reset --remove-rules           # Remove .ledgerful/rules.toml
ledgerful reset --include-ledger --yes   # Destructive: wipe ledger.db
ledgerful reset --all --yes              # Destructive: wipe the entire .ledgerful tree
```

### Intent & Capture (Milestone O)

```bash
ledgerful intent demo                    # Launch the interactive intent capture TUI demo
ledgerful verify --signatures            # Mathematical verification of the entire ledger
```

### Audit & Search

```bash
ledgerful audit [--entity PATH] [--include-unaudited]  # Holistic provenance view
ledgerful ledger audit [--entity PATH]                 # Same as above (legacy alias)
ledgerful ledger search QUERY [--category CAT] [--days N] [--breaking] [--limit N] # FTS5 search
```

## Ledger Subcommands (Provenance)

```bash
ledgerful ledger start PATH --category CAT [--message TEXT] [--issue REF]
ledgerful ledger commit TX_ID --summary TEXT --reason TEXT [--change-type TYPE] [--breaking] [--auto-reconcile | --no-auto-reconcile]
ledgerful ledger rollback TX_ID --reason TEXT
ledgerful ledger atomic PATH --summary TEXT --reason TEXT [--category CAT]
ledgerful ledger status [--entity PATH] [--compact] [--json] [--exit-code] [--verify-signatures]
ledgerful ledger status --global [--repo NAME] [--reindex] [--opt-in|--opt-out]  # multi-repo rollup
ledgerful ledger reconcile [--tx-id ID] [--pattern GLOB] [--all] [--reason TEXT]
ledgerful ledger adopt [--pattern GLOB] [--all] --category CAT --summary TEXT --reason TEXT
ledgerful ledger stack [CAT]                              # Show tech stack and validators
ledgerful ledger register rule TERM --category CAT --reason REASON
ledgerful ledger register validator NAME --command CMD --category CAT [--timeout SEC]
ledgerful ledger adr [--output-dir DIR]                   # Export decisions to MADR
ledgerful ledger graph <tx-id>                            # Entity neighborhood for a transaction
```

## Gate, policy, config (agent-critical)

```bash
ledgerful gate mode                      # Show observe/enforce posture
ledgerful policy check                   # Evaluate declared CI policy
ledgerful config view | verify | schema | diff | set | unset
```

## Topology & security (common review commands)

```bash
ledgerful endpoints [--json] [--changed]
ledgerful services diff
ledgerful data-models impact --changed
ledgerful security boundaries
ledgerful security impact --changed
ledgerful dependencies list | audit
ledgerful observability diff | coverage
ledgerful tests                          # test mapping lookup
ledgerful ci diff | deploy impact
```

## Dead Code Detection

```bash
ledgerful impact --dead-code                         # Include dead-code analysis in impact
ledgerful dead-code [--threshold 0.75] [--limit 50] [--auto-index]
ledgerful dead-code --prune [--threshold 0.75]       # Interactively prune high-confidence dead code
```

`dead-code --prune` iterates through high-confidence findings and prompts
`[Y/n]` per symbol via `inquire`. Approved removals are written to disk and
documented in a `PENDING` ledger transaction with `DELETED` token provenance,
so tests must pass before `ledger commit` finalizes the deletion.

## Live Visualization (feature: viz-server)

```bash
ledgerful viz-server [--port 8765] [--bind 127.0.0.1] [--open]   # Start WebSocket Arc Diagram server
ledgerful viz-server --stop                                       # Stop a running viz server
```

## Watch

```bash
ledgerful watch [--interval 1000] [--json]          # Watch repository for changes
ledgerful watch --no-graph-sync                     # Disable live KG updates during watch
```

## Hotspots & Federation

```bash
ledgerful hotspots --limit 20 --commits 500 [--auto-index]
ledgerful hotspots --json
ledgerful federate status
```

### Indexing & Search

```bash
ledgerful index --docs              # Index markdown documentation
ledgerful index --contracts         # Index OpenAPI/Swagger contracts
ledgerful index --export-docs       # Export KG data to Markdown/Mermaid docs
ledgerful index --export-docs --doc-type module_map --doc-type symbol_index  # Export specific doc types
ledgerful index --full              # Full re-index
ledgerful index --incremental       # Fast refresh
ledgerful search "symbolOrQuery" [--json] [--auto-index]
```

## Gemini-Assisted Reporting

```bash
ledgerful ask "What should I verify next?" [--auto-index]
ledgerful ask --mode suggest "What checks should I run?"
ledgerful ask --mode review-patch "Review the current diff."
ledgerful ask --narrative
```

## Nightly Graph Indexing Scheduler

```bash
ledgerful schedule setup-nightly                # Install nightly `git fetch` + `index --analyze-graph`
ledgerful schedule setup-nightly --dry-run      # Print the generated scheduler syntax without registering it
ledgerful schedule setup-nightly --uninstall   # Remove the scheduled task
ledgerful schedule run-nightly                   # Run the sequence directly (git fetch, then index --analyze-graph)
```

- On **Windows** the command registers a `schtasks` daily task at 02:00 named `LedgerfulNightlyIndex`.
- On **macOS/Linux** it installs a crontab line at `0 2 * * *` that runs `ledgerful schedule run-nightly`.
- Output is appended to `.ledgerful/logs/nightly.log` with RFC3339 timestamps.

## Categories

Use with `ledgerful ledger start|atomic|… --category <CAT>`. Matches engine `Category` (`types.rs`):

| Category | Covers |
|---|---|
| `ARCHITECTURE` | High-level system design, multi-module contracts |
| `FEATURE` | New user-facing or internal functionality |
| `BUGFIX` | Defect repairs |
| `REFACTOR` | Structural improvement without behavior change |
| `INFRA` | CI, git hooks, Docker, build system |
| `SECURITY` | Auth, authz, crypto, disclosure, supply-chain security work |
| `TOOLING` | Internal scripts, dev tooling |
| `DOCS` | Documentation, README, ADRs |
| `CHORE` | Dependencies, formatting, minor cleanup |

## Not exhaustively listed here

Feature-gated or less agent-critical surfaces — use `--help` / product docs:

- `sync` **[Experimental]** — team ledger bundles; opt-in `[sync].enabled=false` default; pairing real (`LF-PAIR-1` invite + accept/list/revoke; never auto-enables). Apply polish 0112 / Available 0113. See `docs/team-sync.md`. Not `watch` Real-time Sync / not `federate`.
- `web`, `usage`, `openapi`, `export evidence`, `bridge`, `mcp`, `demo`, `timings`, `viz-server`
- Full `ledger` advanced subcommands (`re-sign`, `gc`, `export-public`, validators, ADR subcommands, …)
