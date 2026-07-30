---
name: onboarding
description: Load at session start, before meshops code/docs edits, before conductor work, and before using implement. Establishes MeshOps repo context. To execute a track, load implement.
---

# MeshOps Onboarding

identity{
  product:"MeshOps"
  binary:"meshops"
  repo:"C:\\dev\\stl"
  wsl_repo:"/mnt/c/dev/STL"
  docs:"C:\\dev\\stl\\docs"
  conductor:"C:\\dev\\stl\\conductor"
  work_dir:"work/"
  fixtures:"fixtures/"
  os:"Windows (primary) + WSL agents OK for docs/git"
  language:"Python 3.13 (pin >=3.13,<3.14)"
  rule:"do not rename product to STL casually in user-facing strings; folder may be stl/STL"
  monetization:"personal/internal first; prefer MIT/Apache/BSD deps; GPL only subprocess or accepted local filter (PyMeshLab)"
}

start{
  read AGENTS.md
  read .agents/skills/meshops/SKILL.md
  read docs/Difficulty.md
  read docs/MeshOps.md
  read docs/TechStack.md
  read conductor/conductor.md
}

mission{
  summary:"local agent-driven OS for 3D-printable meshes — repair broken STLs, design mechanical parts from specs, design organics agent-first against render evidence, never lie about success"
  jobs:
    - repair (T1–T5) with classify → evidence → guarded mutate → validate
    - design mechanical T7 (build123d primary)
    - design organic T6 (Blender agent loop primary; hosted multi-view API fallback only)
  non_goals:
    - one-click hero sculpt marketing
    - whole-model remesh as default fix
    - FreeCAD as pipeline center
    - MCP-first product (CLI first)
}

authority_order:
  - user/run prompt
  - docs/Difficulty.md
  - docs/TechStack.md
  - docs/MeshOps.md
  - conductor/conductor.md
  - conductor/<track>/spec.md
  - conductor/<track>/plan.md
  - .agents/skills/implement/SKILL.md
  - this onboarding skill
  - AGENTS.md
  - docs/PRD.md
  - docs/Tool-Reference.md
  - research reports (Grok/CGPT/Gemini) as background only
  - external docs

session_start{
  steps:
    - read conductor/conductor.md — active/ready tracks
    - if code exists: meshops doctor (or pytest --collect-only) when available
    - confirm fixture policy for Rogue2 (path or LFS)
    - identify assigned track; refuse to mutate meshes without triage design in-scope
  then:
    - load implement only when executing a track end-to-end
}

conductor_system{
  root:"C:\\dev\\stl\\conductor\\"
  track_naming:"####-Description (e.g. 0001-TriageCore)"
  files:
    registry:"conductor/conductor.md"
    spec:"conductor/<track>/spec.md"
    plan:"conductor/<track>/plan.md"
    review:"conductor/<track>/review.md"
  requirements:
    - each active track MUST have spec.md + plan.md before implementation
    - DoD in spec MUST cite Difficulty.md lesson numbers where guards apply
    - sketches may be thin; deepen at track start — do not invent multi-month padding
  implement:"use implement skill for plan -> tests-first gates -> implement -> visual/numeric proof -> conductor update"
  active_tracks:"from conductor registry, not hard-coded skills"
}

architecture_target{
  engine:"Python 3.13 meshops package"
  interfaces:"typer CLI primary; FastAPI localhost optional; MCP adapter last"
  job_store:"filesystem work/<mesh_id>/{original,working,views,rois,revs,diagnostics.json,report.md}"
  geometry:"trimesh glue; PyMeshLab T1/T2; manifold3d local booleans; F3D eyes; Blender 5.2 LTS escalate"
  design:"build123d T7; CadQuery alternate; organic agent Blender loop T6"
  optional:"Open3D metrics/ROI; Rust meshops-io parser — not blockers for triage"
}

module_boundaries_target{
  ingest:"parse/index/hash only"
  triage:"classify + sheet_score + diagnostics only — no mutation"
  render:"camera sets + depth — no mutation"
  recipes:"T1/T2 only; refuse T3/T4"
  guards:"export acceptance shared library — every mutator calls it"
  design_mechanical:"emits code + mesh into job store then triage"
  design_organic:"authoring jobs produce untrusted meshes into triage"
  slice:"oracle only — not a repair strategy"
}

python_standards{
  requires_python:">=3.13,<3.14"
  typing:"modern type hints"
  schemas:"pydantic 2.x for diagnostics/reports"
  errors:"explicit failure modes; never silent wipeout"
  pins:"docs/TechStack.md"
}

test_conventions{
  framework:"pytest"
  fixtures:
    - fixtures/rogue2/ — permanent hero regression
    - fixtures/synthetic/ — sheet + clothing false-positive
  naming:"feature__condition__expected"
  require:
    - wipeout export must fail hard (Difficulty §6)
    - bbox/component/face-floor guards
    - no success without render artifact paths in report for mutating paths
  never:
    - network calls in unit tests
    - overwrite fixture originals
}

anti_overengineering{
  do_not:
    - block M1 on Rust parser or Open3D
    - build hosted generator before agent organic loop exists
    - center pipeline on FreeCAD
    - add photoreal render tier for triage
    - invent plugin frameworks before second backend exists
}

retrieval_precedence:
  - active file/spec
  - Difficulty.md
  - TechStack.md
  - conductor track
  - MeshOps.md
  - Tool-Reference.md
  - official Blender/build123d/trimesh docs
  - web search for pin/API currency

external_research{
  stale_training:true
  use_when:
    - version pin matters
    - Blender bpy API matters
    - wheel/platform install matters
  preferred:
    - docs/TechStack.md first
    - official docs / PyPI / GitHub releases
}

standard_track_summary{
  phases:
    - plan (spec/plan present; Difficulty citations)
    - failing acceptance tests / fixture gates
    - implement
    - targeted pytest + manual render proof when visual
    - full gate
    - conductor status update
    - ledger commit if ledgerful available
}

key_docs:
  - docs/Difficulty.md
  - docs/MeshOps.md
  - docs/TechStack.md
  - docs/PRD.md
  - docs/Tool-Reference.md
  - conductor/conductor.md
  - AGENTS.md
  - .agents/skills/implement/SKILL.md
  - .agents/skills/meshops/SKILL.md
