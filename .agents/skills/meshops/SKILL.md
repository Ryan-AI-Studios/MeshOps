---
name: meshops
description: Use when working on MeshOps mesh repair, triage, design-from-spec, organic authoring, Blender handoff, slicing validation, or export guards. Load onboarding first. Prefer CLI-shaped tools; never claim mesh success without multi-view evidence.
---

# MeshOps

Local **agent-driven OS for 3D-printable meshes**. Primary interface is a future `meshops` CLI; MCP is a later adapter over the same tools.

## Core jobs

| Job | Input | Output |
|-----|--------|--------|
| Repair | Broken STL/scan | Validated printable mesh + evidence |
| Mechanical design (T7) | Text/dimensional spec | build123d code → STEP/STL → slice |
| Organic design (T6) | Prompt + refs | Agent Blender loop → triage (hosted API fallback only) |

Spine for all jobs: **ingest → triage → evidence → guarded mutation → validation**.

## Binding law

Read `docs/Difficulty.md`. Highlights:

1. Classify before mutate
2. Visual acceptance is first-class
3. No whole-model hero remesh without opt-in
4. Export guards (face/size/bbox/components)
5. Non-destructive originals + revs
6. Ask on laterality / multi-figure ambiguity
7. Local volume booleans only — never full-mesh boolean after solidify

Rogue2.stl is the permanent regression fixture: never export wipeout-class "success".

## Defect classes (route before tools)

| Class | Route |
|-------|--------|
| T1 Topology | PyMeshLab auto |
| T2 Printability | PyMeshLab + slicer report |
| T3 Sheet limb | Preview only → Blender escalate |
| T4 Missing volume | Regen / sculpt |
| T5 Mechanical feature | FreeCAD specialist |
| T6 Organic from scratch | Agent Blender → triage |
| T7 Mechanical from scratch | build123d → triage |

## Tool surface (target)

See `docs/MeshOps.md` §5. Mutating tools require prior triage. `mesh_repair` refuses T3/T4.

## Stack pins

**Only** `docs/TechStack.md`. Do not invent versions.

- Python `>=3.13,<3.14`
- build123d primary; CadQuery alternate
- Blender **5.2 LTS** (mirror install — blender.org CDN may 403 automation)
- Open3D optional; Rust parser optional
- F3D for ortho+depth eyes; OrcaSlicer as printability oracle (subprocess)

## Job store

```text
work/<mesh_id>/original.stl   # hashed, never overwrite
work/<mesh_id>/diagnostics.json
work/<mesh_id>/views/
work/<mesh_id>/revs/
work/<mesh_id>/report.md
```

## Agent behavior

- Prefer depth maps for sheet detection evidence
- Confirm limb laterality with user when multi-figure
- Hosted generators: multi-view only + justify vs free agent path
- Prefer MIT/Apache/BSD deps when choosing new libraries
- Personal tool first — still keep productization-friendly licenses

## References

- `references/difficulty-map.md` — lesson → guard mapping
- `references/commands.md` — CLI verb cheat sheet (target)
- `docs/Tool-Reference.md` — tool appendix
- `docs/MeshOps.md` — full spec
