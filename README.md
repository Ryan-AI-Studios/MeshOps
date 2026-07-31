# MeshOps

Local agent-driven OS for 3D-printable meshes (repair + mechanical design + organic authoring).

| Doc | Role |
|-----|------|
| [docs/Difficulty.md](docs/Difficulty.md) | **Binding law** (Rogue2 lessons) |
| [docs/MeshOps.md](docs/MeshOps.md) | Product spec |
| [docs/TechStack.md](docs/TechStack.md) | Dependency pins |
| [docs/PRD.md](docs/PRD.md) | Short PRD |
| [conductor/conductor.md](conductor/conductor.md) | Tracks |
| [conductor/deferred.md](conductor/deferred.md) | Parking lot (not a work queue) |
| [conductor/planner-handoff.md](conductor/planner-handoff.md) | Living cold-start for planners |
| [AGENTS.md](AGENTS.md) | Agent rules (published) |

Local only (gitignored): `conductor/`, `docs/`, `.agents/` (skills including onboarding).

**Status:** Design + local conductor/skills + quality gates. Product code tracks start at 0001.

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

Blender manual epub under `docs/blender_manual_v520_en.epub` is reference only.
