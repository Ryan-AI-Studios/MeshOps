---
name: implement
description: Use when implementing one assigned MeshOps conductor track end-to-end. Load with onboarding. Conductor under C:\dev\stl\conductor is source of truth for tracks.
---

# Implement MeshOps Conductor Track

identity{
  repo:"C:\\dev\\stl"
  load_with:"onboarding"
  source_of_truth:"conductor/conductor.md + conductor/<track>/spec.md + conductor/<track>/plan.md"
  law:"docs/Difficulty.md"
  pins:"docs/TechStack.md"
  do_not:
    - clear done with open wipeout-class export path
    - clear done without multi-view proof on mutating tracks
    - whole-model hero remesh without explicit user opt-in
}

mode{
  default:"serial single-agent"
  clearance:"one track/slice at a time"
  velocity:"prefer thin vertical slices; hours not months — still no skip of guards"
}

loop:
  - plan
  - start_tx_if_ledger
  - implement_with_gates
  - targeted_checks
  - visual_proof_if_mutating
  - full_gate
  - conductor_update
  - ledger_commit_if_available

plan{
  before_edit:
    - read conductor/conductor.md
    - read conductor/<track>/spec.md
    - read conductor/<track>/plan.md
    - read cited Difficulty.md sections
    - confirm TechStack pins for any new dep
  output_plan:
    - modules/files touched
    - defect classes affected (T1–T7)
    - acceptance tests + fixtures
    - render/camera requirements
    - export guard interactions
  if_missing_spec_or_plan:
    create thin sketch:
      - objective
      - Difficulty citations
      - DoD
      - phased checklist
    conductor_status:"Planning"
  spec_vs_reality:
    rule:"follow actual code layout over aspirational paths"
    action:"note drift in plan.md; do not create fake packages only to match sketch"
}

implement{
  tests_first_for_guards:true
  red:"write failing guard/fixture tests where behavior is specified"
  green:"implement until gates pass"
  recipes:
    - T1/T2 only in mesh_repair
    - T3 = preview or blender_handoff only
  design:
    - T7 via build123d primary
    - T6 agent path before any hosted API
  intermediate_commits:"allowed"
}

research{
  stale_knowledge:true
  required_when:
    - bpy / Blender version behavior
    - wheel install (Open3D, PyMeshLab, F3D)
    - build123d/CadQuery API
  precedence:
    - TechStack.md
    - Difficulty.md
    - track spec
    - MeshOps.md
    - upstream docs
}

targeted_checks{
  when_code_exists:
    - ruff check
    - pytest -q path/to/tests
    - manual: inspect views/ PNGs for T3/visual tracks
}

visual_proof_if_mutating{
  required_cameras_minimum:"front + 3/4 + relevant zoom"
  t3_extra:"front ortho silhouette + waist/limb zoom"
  artifacts:"paths recorded in job report.md / diagnostics"
  rule:"no done without artifacts on disk"
}

full_gate{
  required:
    - pytest (guards + fixtures in scope)
    - Rogue2 never exports < configured size/face floor as success
  hygiene:
    - no secrets
    - originals/fixtures untouched
    - work/ outputs gitignored unless golden fixtures
}

conductor_update{
  - set track status accurately (Ready / In Progress / Done)
  - note proof location (test names, sample work/<id>/)
  - leave next-track dependencies explicit
}

ledger{
  if_ledgerful_available:
    start:"ledgerful ledger start meshops --category <CAT> --message <track intent>"
    commit:"ledgerful ledger commit <tx> --summary <what> --reason <why>"
  else:"skip without drama; git history + conductor is enough"
}

parallelism{
  default:"serial"
  note:"mesh jobs are filesystem-heavy; do not parallelize two mutators on same mesh_id"
}
