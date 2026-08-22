# MeshOps

Local agent-driven OS for 3D-printable meshes (repair + mechanical design + organic authoring).

| Doc | Role |
|-----|------|
| [docs/archive/Difficulty.md](docs/archive/Difficulty.md) | **Binding law** (Rogue2 lessons) |
| [docs/MeshOps.md](docs/MeshOps.md) | Product spec |
| [docs/TechStack.md](docs/TechStack.md) | Dependency pins |
| [docs/PRD.md](docs/PRD.md) | Short PRD |
| [conductor/conductor.md](conductor/conductor.md) | Tracks |
| [conductor/deferred.md](conductor/deferred.md) | Parking lot (not a work queue) |
| [conductor/planner-handoff.md](conductor/planner-handoff.md) | Living cold-start for planners |
Local only (gitignored): `conductor/`, `docs/`, `.agents/` (skills including onboarding), `AGENTS.md` (agent rules — Ledgerful-style compact policy).

**Status:** Design + local conductor/skills + quality gates. Product code tracks start at 0001.

### Setup (this machine)

```powershell
# 1. Python env (requires Python >=3.13,<3.14)
uv sync --extra dev
# optional T7 design stack:
# uv sync --extra design
# optional MCP stdio adapter (Claude Desktop / Cursor / Inspector):
# uv sync --extra mcp
# uv run meshops-mcp   # or: python -m meshops.mcp  (set cwd=repo; MESHOPS_WORK optional)
# docs: docs/mcp/README.md
# optional multi-view proportion overlays + JPG/WebP (track 0012):
# uv sync --extra proportion
# uv run meshops proportion template|analyze|show

# 2. Diagnose tools (default: core env only — missing Blender/Orca is a warning)
uv run meshops doctor
uv run meshops doctor --json
# print/organic-ready box:
# uv run meshops doctor --strict

# 3. Bootstrap portable Blender 5.2 LTS if missing (Difficulty §4 mirrors)
#    Requires a **repo clone** — scripts/ is not shipped in the pip/uv wheel.
.\scripts\bootstrap-tools.ps1
# dry-run (no network):
.\scripts\bootstrap-tools.ps1 -WhatIf

# 4. OrcaSlicer 2.4.2 (manual): GitHub Releases or Microsoft Store, then:
# $env:MESHOPS_ORCA = 'C:\Program Files\OrcaSlicer\orca-slicer.exe'
```

`meshops doctor` composes existing Blender/Orca discoverers; bootstrap sets `MESHOPS_BLENDER` (User + current session) and installs under `%LOCALAPPDATA%\MeshOps\tools\blender-5.2.0\`.

### Quality gates (local = CI)

```powershell
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest
# or all four via:
ledgerful verify
```

CI: `.github/workflows/ci.yml` (Python 3.13, Node 24, uv 0.12.0). Pins: `docs/TechStack.md` §7.

### Product remake (blockout)

Remake only when emit moved. Hygiene / CLI / docs / skill tracks are tests (and local skills) only — do not mint `product_####up`.

MeshOps SoT is the CLI recipe loop → `setup_blockout_recipe.py` → `meshops proportion blockout-open-setup`. build_and_render.py is work/-only — not a product verb (track 0111). Job-local `render_d7.py` is OK for isolated D7 under that remake. Do not copy `template_applied.json` into a remake so validate will run (0109 skips missing). RECIPE ≠ print success (N6).

### Post-nofuse sculpt (blockout)

Stop RECIPE oval thrash after form lock.

Handoff is `--nofuse` lock then `--join-ready` then `blockout-fuse-plan`. Voxel remesh is opt-in authoring weld — not a repair default (N1). 1 island is not print success (N6).

### Viewport soft hide (blockout)

Hide is not cull. Keep full for region polish.

Overview QA hides 0082-class secondaries in Blender; `--soft-density compact` culls them from the recipe. Viewport hide is not print success (N6).

Blender manual epub under `docs/blender_manual_v520_en.epub` is reference only.
