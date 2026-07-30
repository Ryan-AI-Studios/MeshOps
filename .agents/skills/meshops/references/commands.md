# meshops CLI (target surface)

Not all commands exist until tracks land. Agents should design toward this shape.

```text
meshops doctor                     # tools on PATH, blender path, f3d, python pins
meshops ingest <path>              # -> mesh_id
meshops triage <mesh_id>           # diagnostics.json + hypotheses
meshops render <mesh_id> [--cameras ...]
meshops roi <mesh_id> --bbox|--mask
meshops repair <mesh_id> --recipe  # T1/T2 only
meshops preview-t3 <mesh_id> --roi
meshops blender-handoff <mesh_id> --roi
meshops diff-views <a> <b>
meshops slice <mesh_id> --profile
meshops design spec "..."          # T7 build123d
meshops design organic-agent ...   # T6 primary
meshops design organic-api ...     # T6 fallback (multi-view + justify)
meshops report <mesh_id>
meshops export <mesh_id> --rev     # runs guards or aborts
```

Rules:

- mutate ⇒ triage first
- export ⇒ guards + views
- organic-api ⇒ multi-view + non-default justification
