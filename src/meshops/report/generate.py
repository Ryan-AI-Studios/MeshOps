"""report.md from diagnostics template (spec §3.1)."""

from __future__ import annotations

import json
from pathlib import Path

from meshops.jobstore.paths import JobPaths
from meshops.models.diagnostics import Diagnostics


def _load_diagnostics(paths: JobPaths) -> Diagnostics:
    if not paths.diagnostics_json.is_file():
        raise FileNotFoundError(f"diagnostics.json missing for {paths.mesh_id!r}; run triage first")
    data = json.loads(paths.diagnostics_json.read_text(encoding="utf-8"))
    return Diagnostics.model_validate(data)


def _list_views(views_dir: Path) -> list[str]:
    if not views_dir.is_dir():
        return []
    return sorted(str(p.relative_to(views_dir.parent)) for p in views_dir.glob("*.png"))


def generate_report(
    mesh_id: str,
    *,
    work_root: Path | str = "work",
) -> Path:
    """Write report.md and return its path."""
    paths = JobPaths(work_root=Path(work_root), mesh_id=mesh_id)
    if not paths.job_dir.is_dir():
        raise FileNotFoundError(f"Job directory not found: {paths.job_dir}")

    diag = _load_diagnostics(paths)
    views = _list_views(paths.views_dir)
    s = diag.stats
    sheet = diag.sheet_score

    hyp_lines = []
    for h in diag.defect_hypotheses:
        hyp_lines.append(
            f"- **{h.defect_class.value}** / confidence={h.confidence:.3f} / {h.notes}"
        )
    if not hyp_lines:
        hyp_lines.append("- (none)")

    view_lines = (
        [f"- `{v}`" for v in views] if views else ["- (no views yet — run `meshops render`)"]
    )

    feat = sheet.features
    md = f"""# MeshOps report — {mesh_id}

## Stats
| field | value |
|---|---|
| mesh_id | {s.mesh_id} |
| faces | {s.faces} |
| vertices | {s.vertices} |
| components | {s.components} |
| bbox_min | {s.bbox_min} |
| bbox_max | {s.bbox_max} |
| bbox_diagonal | {s.bbox_diagonal:.6g} |
| is_watertight | {s.is_watertight} |
| is_volume | {s.is_volume} |
| is_manifold | {s.is_manifold} |
| non_manifold_edge_count | {s.non_manifold_edge_count} |
| boundary_edge_count | {s.boundary_edge_count} |
| euler_characteristic | {s.euler_characteristic} |
| file_size_bytes | {s.file_size_bytes} |
| content_sha256 | {s.content_sha256} |

## Defect hypotheses
{chr(10).join(hyp_lines)}

## Sheet score
- score: **{sheet.score:.4f}**
- confidence: **{sheet.confidence:.4f}**
- auto_action: **{sheet.auto_action.value}** (never delete — N8 / Difficulty §7)
- features:
  - thinness_mean={feat.thinness_mean:.4f}
  - thinness_p95={feat.thinness_p95:.4f}
  - candidate_fraction={feat.candidate_fraction:.4f}
  - planarity={feat.planarity:.4f}
  - section_thinness={feat.section_thinness:.4f}
  - dihedral_crease={feat.dihedral_crease:.4f}
  - normal_smoothness={feat.normal_smoothness:.4f}
  - clothing_penalty={feat.clothing_penalty:.4f}
  - neighborhood: k={feat.neighborhood_k}, r={feat.neighborhood_radius:.6g}
  - samples={feat.n_samples}, candidates={feat.n_candidates}, stage2={feat.stage2_used}
- notes: {", ".join(sheet.notes) if sheet.notes else "(none)"}

## Evidence views
{chr(10).join(view_lines)}

## Camera convention
- bbox-relative orthographic cameras (front/back/left/right/top/bottom/three_quarter)
- view names are **camera** names, not anatomical L/R (Difficulty §1)
- depth maps (`*_depth.png`) are **visual** 8-bit colormapped previews for agents —
  **not** metric depth / rangefinder (Difficulty §9). Numeric thickness uses mesh queries.

## User input
- needs_user_input: **{diag.needs_user_input}**
- laterality_status: **{diag.laterality_status.value}**

## Honesty
- Triage only — no repair / print-ready claims (MeshOps §9; Difficulty §12).
- schema_version: {diag.schema_version}
"""
    paths.report_md.write_text(md, encoding="utf-8")
    return paths.report_md
