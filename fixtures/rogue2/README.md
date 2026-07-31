# Rogue2 fixture policy

**Permanent binding fixture** for MeshOps (Difficulty report subject).

## Source

- Default path (developer machine): `C:\Users\RyanB\Documents\3D\Elizabeth\Rogue2.stl`
- Override: environment variable **`MESHOPS_ROGUE2_PATH`** pointing at a readable STL.

## Scale

- ~676k faces / ~34 MB binary STL (Difficulty §3)
- Multi-component statue; laterality is ambiguous without user confirmation (Difficulty §1)

## Rules

1. **Never overwrite** the user original path. Ingest copies into `work/<mesh_id>/original.stl` and marks it read-only.
2. Fixture originals under user Documents are **sacred** — no in-place mutation.
3. CI / default pytest **skips** Rogue2 tests when the path is absent (`@pytest.mark.rogue2`).
4. Synthetics under `fixtures/synthetic/` always run; Rogue2 is optional evidence for DoD-5.
5. Triage only on this track (0001) — no repair recipes, no claim of “fixed.”

## Hash

Record content SHA-256 after first successful ingest if desired (optional). Deterministic `mesh_id` = first 12 hex of SHA-256 of original bytes.
