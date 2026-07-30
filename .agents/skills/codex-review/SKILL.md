---
name: codex-review
description: Audit an orchestrator-specified Ledgerful conductor track after implementation and internal fixes. Verify every requirement and Definition of Done, find placeholders, incomplete wiring, regressions, and weak tests, and return evidence-based findings for fix and re-review.
----------------------------------------------

# Track Completion Review

Read-only audit. The orchestrator selects the track, implements, fixes, runs
gates, manages deferrals, and decides completion.

## Handoff

Required:

```text
TRACK: <####-Name or absolute track directory>
```

Optional:

```text
REPOS: <execution repo paths>
SCOPE: <base/commit range/working tree>
IMPLEMENTED: <brief summary>
KNOWN GATES: <commands/results or external gates>
FOCUS: <extra risks>
```

Never guess or switch tracks. If only a number/name is supplied, resolve it
from `C:\dev\coordinated\conductor\conductor.md`, then read that track's
`spec.md` and `plan.md`. Use their execution repos when `REPOS` is omitted.

```text
ROOT=C:\dev\coordinated
DEFERRED=C:\dev\coordinated\conductor\deferred.md
COORDINATION=C:\dev\coordinated\coordination.md
```

Raw output: `<track>\review.<reviewer>.md`.
The orchestrator writes canonical `<track>\review.md`.

## Rules

* Never modify files, governance, Git state, or `deferred.md`.
* Review the supplied track, all named repos, and resulting behavior.
* Read every requirement, plan phase, risk, and DoD item.
* Include supplied committed, staged, unstaged, untracked, and already-merged work.
* Inspect affected callers, consumers, tests, migrations, feature gates,
  generated files, contracts, docs, and configuration.
* Do not claim a command passed unless its result was observed.
* No invented, style-only, keyword-only, or unsupported speculative findings.
* A clean diff or compiling symbol is not proof of completion.

## Audit

### 1. Requirements / DoD

One row per independently verifiable item:

```text
Requirement | Met/Partial/Unmet/Not verifiable | Evidence | Tests | Gap
```

Verify every required phase and DoD; unauthorized omissions or scope cuts;
accurate implementation/conductor claims; and separation of external gates
from incomplete engineering.

### 2. Completeness Sweep

Search relevant code for:

```text
TODO FIXME XXX HACK TEMP placeholder stub mock fake sample
not implemented unsupported todo! unimplemented! panic!("TODO")
hardcoded zero/empty/null/None silent fallback skip ignore allow(dead_code)
```

Investigate:

* no-op/false-success paths and fake `0/false/null/None/[]/{}/—/unknown`;
* dead, unreachable, commented-out, or never-called implementation;
* unregistered CLI/routes/UI and disconnected controls;
* writes never read; displays never populated;
* unregistered migrations or stale schemas/generated contracts;
* required behavior missing under supported features/platforms;
* mocks/stale data shown as live;
* skipped tests, weakened assertions, swallowed errors;
* temporary paths with no correct final path;
* placeholders the track exists to remove.

A placeholder satisfying promised behavior is blocking.

### 3. Wiring

Trace every core path:

```text
entry -> implementation -> persistence/boundary -> reader ->
API/CLI/UI -> contract/client -> visible result
```

Verify production reachability, callers, consumers, and supported variants.

### 4. Correctness / Regression

Check relevant invariants, states, failures, atomicity, idempotency, retries,
timeouts, concurrency, paths/symlinks, subprocesses, auth/secrets, bounds,
encoding/serialization, legacy/null data, migrations/upgrades, renames,
binaries, platforms/features, determinism, signing/provenance/hashing, and
unintended changes outside scope.

### 5. Tests / Evidence

Tests should execute production paths; prove acceptance criteria; cover material
failures/boundaries; fail against old behavior; verify persisted/external
results; cover relevant legacy/migrated/feature/platform cases; and avoid mocks
when integration evidence is required.

Separate:

```text
Observed now | Reported by orchestrator | Recommended | Not verifiable
```

The orchestrator runs artifact-writing gates unless explicitly authorized.

### 6. Contracts / Docs

Check affected CLI help, APIs/OpenAPI, generated types, migrations/schema docs,
`coordination.md`, architecture/install docs, trust/security and
pricing/availability claims, changelog, errors, feature labels, and conductor
summary. They must match actual behavior.

## Findings

```text
P0 catastrophic security/corruption/destructive migration/invalid signing
P1 unmet core DoD or serious correctness/security regression
P2 substantive edge/integration/contract/compatibility/test defect
P3 real limited non-blocking issue
```

Format:

```text
[P0-P3] Title
Confidence: High|Medium|Low
Requirement:
Location: repo\path:line
Problem:
Evidence:
Failure scenario:
Correction:
Verification:
Deferrable: Yes|No
```

Inspect surrounding code, invariants, and tests before reporting. Exclude generic
“more tests,” duplicates, unreachable theories, preferences, and unrelated debt
unless this track worsens or depends on it.

## Deferral

Reviewer proposes only; orchestrator validates and edits `deferred.md`.

Deferrable only if validated P3; not a DoD/promise/placeholder gap; no security,
integrity, auth, signing, provenance, migration, compatibility, or core-workflow
impact; current behavior remains correct/honest; fix is materially difficult,
risky, or outside bounded scope; and not already recorded.

Difficulty never makes P0-P2 deferrable. Fix easy P3s.

## Output

```text
# Track Completion Audit — <TRACK>
## Verdict: PASS | PASS WITH DEFERRED P3 | FAIL
## Scope Reviewed
## Requirement and DoD Matrix
## Findings
## Completeness Sweep
## Wiring and Regression Review
## Verification Evidence
## Deferred Candidates
## Completion Decision
```

`PASS`: all engineering DoD met, no findings.
`PASS WITH DEFERRED P3`: only qualifying lows.
Otherwise: `FAIL`.

## Reviewer Prompt

```text
You are the independent completion reviewer for Ledgerful track <TRACK>.

Track directory: <TRACK_DIR>
Execution repos: <REPOS>
Scope: <SCOPE>
Implemented: <IMPLEMENTED>
Known gates: <KNOWN_GATES>
Extra focus: <FOCUS>

Read all of spec.md and plan.md. Audit every requirement and Definition of Done
against the resulting implementation. READ-ONLY; never modify files or Git.

Check:
1. Every requirement/DoD is actually implemented.
2. No placeholders, stubs, fake values, no-op paths, silent fallbacks,
   skipped tests, or incomplete implementation remains.
3. Core behavior is wired end to end and reachable in production.
4. Correctness, failures, edge cases, compatibility, migrations, determinism,
   security, signing/provenance boundaries, and regressions.
5. Tests prove required behavior and would catch regression.
6. APIs, schemas, generated types, docs, claims, and governance agree.
7. No required work was omitted, narrowed, or improperly deferred.

Use the required P0-P3 format and output sections. Do not invent findings.
Only difficult, non-blocking P3 items may be proposed for deferred.md.
```

## Reviewers

Flow:

```text
internal review -> fix -> internal re-review ->
cross-model review -> validate/fix -> cross-model re-review
```

### Codex Primary

Default full audit: `gpt-5.6-luna`, high.
Use Luna/high only for narrow rechecks; Sol/high for security-critical work.

```powershell
# Current CLI (verified 0.144.x): NO -a / --approval-policy flag on `codex exec`.
# Read-only sandbox is -s read-only only. Do not invent -a never (fails to parse).
codex exec -C $PrimaryRepo -s read-only `
  -m $CodexModel -c 'model_reasoning_effort="high"' `
  --add-dir "C:\dev\coordinated" --ephemeral `
  -o "$TrackDir\review.codex.md" $Prompt
```

Repeat `--add-dir <repo>` for sibling repos.

**CLI hygiene (do not regress):**

* **No `-a never`** — not a valid `codex exec` option; agents must not re-add it.
* **No `$null |`**, **no `--yolo`**, **no** `--dangerously-bypass-approvals-and-sandbox`.
* Prefer this skill's `codex exec … -s read-only` path over native `codex review`
  for track DoD audits (full prompt + multi-repo `--add-dir`).
* If PowerShell mangles a long `$Prompt`, pipe via stdin: `$Prompt | codex exec … -`
  (trailing `-` reads the prompt from stdin).
* On Windows non-interactive hangs, redirect stdin from NUL only when not piping
  the prompt: `cmd /c "codex exec … < NUL"`.

### Claude Fallback / Second

Use Sonnet 5/high. Restrict tools; `allowedTools` alone is not restriction.

**Two invocation patterns** — the positional `$Prompt` form can fail on Windows
PowerShell when the prompt contains characters that parse as CLI flags. Piping
via stdin is the reliable fallback.

**Pattern A — positional argument (may fail on long/complex prompts):**

```powershell
Push-Location $PrimaryRepo
try {
  claude -p --model claude-sonnet-5 --effort high `
    --permission-mode dontAsk --no-session-persistence --no-chrome `
    --disable-slash-commands --strict-mcp-config `
    --tools "Read,Glob,Grep,Bash" `
    --allowedTools "Read" "Glob" "Grep" `
      "Bash(git status *)" "Bash(git branch *)" "Bash(git diff *)" `
      "Bash(git log *)" "Bash(git show *)" "Bash(git rev-parse *)" `
      "Bash(git merge-base *)" "Bash(git ls-files *)" "Bash(git grep *)" `
      "Bash(git -C * status *)" "Bash(git -C * diff *)" `
      "Bash(git -C * log *)" "Bash(git -C * show *)" `
    --disallowedTools "Edit" "Write" "NotebookEdit" "mcp__*" `
    --add-dir "C:\dev\coordinated" $Prompt |
    Set-Content "$TrackDir\review.claude.md" -Encoding utf8
} finally { Pop-Location }
```

**Pattern B — stdin pipe (reliable fallback):**

```powershell
Push-Location $PrimaryRepo
try {
  $Prompt | claude -p --model claude-sonnet-5 --effort high `
    --permission-mode dontAsk --no-session-persistence --no-chrome `
    --disable-slash-commands --strict-mcp-config `
    --tools "Read,Glob,Grep,Bash" `
    --allowedTools "Read" "Glob" "Grep" `
      "Bash(git status *)" "Bash(git branch *)" "Bash(git diff *)" `
      "Bash(git log *)" "Bash(git show *)" "Bash(git rev-parse *)" `
      "Bash(git merge-base *)" "Bash(git ls-files *)" "Bash(git grep *)" `
      "Bash(git -C * status *)" "Bash(git -C * diff *)" `
      "Bash(git -C * log *)" "Bash(git -C * show *)" `
    --disallowedTools "Edit" "Write" "NotebookEdit" "mcp__*" `
    --add-dir "C:\dev\coordinated" |
    Set-Content "$TrackDir\review.claude.md" -Encoding utf8
} finally { Pop-Location }
```

Add sibling repos after `--add-dir`. Never bypass permissions.

### OpenCode Fallback / Second

Use configured Ollama Cloud `glm-5.2:cloud`, variant `high`. Resolve exact
`provider/model` with `opencode models --refresh`; never guess/substitute.

Require a `track-review` primary agent allowing read/glob/grep and read-only
Git only; deny edit and all other shell commands.

```powershell
opencode run --dir $PrimaryRepo --agent track-review `
  --model $OpenCodeModel --variant high --format default $Prompt |
  Set-Content "$TrackDir\review.opencode.md" -Encoding utf8
```

Omit `--auto`.

## Orchestrator Loop

Classify findings as `Validated`, `Partly valid`, `False positive`,
`Already fixed`, or `Out-of-scope real`.

1. Fix validated P0-P2 and easy P3.
2. Record only qualifying difficult P3 in `deferred.md`.
3. Run normal repo gates.
4. Reinvoke with same track plus fix summary.
5. Require prior-finding verification and a fresh regression sweep.
6. Repeat until `PASS` or `PASS WITH DEFERRED P3`.

Reviewer never fixes code or marks completion.

## Completion Handoff

Orchestrator writes `<track>\review.md` with scope, reviewers/rounds, final DoD
matrix, findings/dispositions, exact gate results, recorded deferrals, residual
external gates, and completion decision.

Mark `Completed` only after engineering DoD, verification, review, governance,
and canonical `review.md` are complete.
