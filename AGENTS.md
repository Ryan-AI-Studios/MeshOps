powershell{
  forbid:"&& | [[ | ]] | then | fi | done | echo -e"
  prefer:"Get-ChildItem | Get-Content | Test-Path | Join-Path | Copy-Item | Remove-Item"
  rules:
    - use $_ and object properties for pipelines
    - use backslashes for shell-level Windows paths
    - avoid Bash shims for complex logic
    - chain commands with ; or separate lines
}

meshops{
  product:"MeshOps"
  binary:"meshops"
  repo:"C:\\dev\\stl"
  docs:"C:\\dev\\stl\\docs"
  conductor:"C:\\dev\\stl\\conductor"
  work_dir:"work/"
  fixtures:"fixtures/"
  before:
    - load .agents/skills/onboarding/SKILL.md
    - read docs/Difficulty.md (binding law)
    - read docs/TechStack.md (binding pins)
    - read conductor/conductor.md
    - skim conductor/deferred.md (Never/Parked — not a work queue)
  edit:
    - never overwrite original.stl / fixture originals
    - never whole-model voxel remesh hero meshes without explicit user opt-in
    - never full-mesh boolean after solidify
    - classify before mutate — no recipe without triage class
    - never implement deferred.md items without promoting a track
  after:
    - multi-view renders required before claiming success
    - export guards must pass (face floor, size floor, bbox, components)
    - ledgerful verify (or ruff + format + basedpyright + pytest)
    - Rogue2 must never export wipeout-class "success"
  skip_for:
    - docs-only wording with no acceptance/behavior change
    - explicit user bypass
  fail:
    numeric_only_success:"FAIL — visual acceptance missing"
    wipeout_export:"FAIL — abort and archive as do-not-use"
}

difficulty_law{
  source:"docs/Difficulty.md"
  rule:"every guardrail and acceptance test cites a Difficulty lesson number"
  permanent_fixture:"Rogue2.stl (path/policy in fixtures/rogue2/)"
  never:
    - claim fixed on watertight/face count alone
    - auto-delete linked-flat organic sheets without tight mask score
    - trust origin-space marching-cubes without world transform check
}

stack{
  python:">=3.13,<3.14"
  primary_cad:"build123d"
  alt_cad:"cadquery"
  blender:"5.2 LTS"
  open3d:"optional"
  meshops_io_rust:"optional"
  pins:"docs/TechStack.md only — do not invent versions from memory"
}

ledger{
  note:"When ledgerful is available on this tree, use ChangeGuard/Ledgerful discipline"
  start:"ledgerful ledger start meshops --category <CAT> --message <intent>"
  commit:"ledgerful ledger commit <tx-id> --summary <what> --reason <why>"
  categories:"ARCHITECTURE | FEATURE | BUGFIX | REFACTOR | INFRA | SECURITY | TOOLING | DOCS | CHORE"
  if_unavailable:"branch + clear commits + update conductor status; do not block on missing ledger"
}

verify{
  scope:"targeted during work; full before finalizing a track"
  required_when_code_exists:
    - ruff check .
    - ruff format --check .
    - basedpyright
    - pytest (unit + fixture gates)
    - Rogue2 wipeout guard test green (when fixtures exist)
  how:
    - preferred:"ledgerful verify  # uses .ledgerful/config.toml steps"
    - ci:".github/workflows/ci.yml mirrors the same four gates"
    - pins:"docs/TechStack.md §7"
  never:
    - declare track done on numeric mesh stats alone
}

git_hygiene{
  branches:
    - never push directly to main without explicit user OK
    - feature/<track-name> preferred
  commits:
    - imperative messages
    - no secrets, no huge binary dumps without LFS/policy
}

agent_surfaces{
  primary:"CLI"
  secondary:"localhost FastAPI"
  later:"MCP adapter (same tools as CLI) — not the product"
}
