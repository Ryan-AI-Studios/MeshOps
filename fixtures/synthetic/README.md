# Synthetic fixtures

Code-generated meshes for tests and the **0009 size ladder**. Rebuildable — not sacred originals (never overwrite fixture originals elsewhere).

## Existing four (sheet_score / clothing FP)

Built by `fixtures/synthetic/build.py` → `cache/*.stl` (generated on demand):

| Name | Builder | Intent |
|---|---|---|
| `solid_cylinder` | `build_solid_cylinder` | Control — low sheet_score |
| `cylinder_arm_sheet` | `build_cylinder_with_arm_sheet` | T3-like coplanar arm sheet |
| `two_body_gap_sheet` | `build_two_body_gap_sheet` | Gap bridge sheet |
| `clothing_cape` | `build_clothing_cape_plane` | Clothing FP — auto_action must stay non-delete |

Also: `t1_t2.py` for T1/T2 repair recipe fixtures.

```powershell
uv run python -c "from fixtures.synthetic.build import build_all; print(build_all())"
```

## Size ladder (S / M / L / XL)

Owned by `meshops.bench.sizes` (track **0009-Hardening**). Deterministic UV-sphere / exceed-then-trim generators — **not** bare icosphere power-of-4 only.

| Label | Target faces | Tolerance |
|---|---:|---|
| **S** | 100_000 | ±15% actual |
| **M** | 500_000 | ±15% |
| **L** | 1_000_000 | ±15%; may skip if available RAM &lt; ~4 GiB |
| **XL** | 2_000_000 | ±15%; may skip if available RAM &lt; ~4 GiB |

```powershell
uv run --extra bench meshops bench run --sizes S,M --json
uv run meshops bench envelope --json
```

Meshes are written under `work/bench/` (gitignored). Record both `target_faces` and `actual_faces` in envelope JSON.

## Policy

- Do **not** treat synthetic ladder STLs as wipeout regression heroes — that role is **Rogue2** (`fixtures/rogue2/`, `MESHOPS_ROGUE2_PATH`).
- Benchmarks never redefine wipeout success as speed (Difficulty §6).
