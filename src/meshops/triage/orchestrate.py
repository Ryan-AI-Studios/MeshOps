"""Triage orchestration: stats + sheet_score + hypotheses → diagnostics.json."""

from __future__ import annotations

from pathlib import Path

from meshops.ingest.stats import compute_stats, load_mesh
from meshops.jobstore.paths import JobPaths, content_sha256
from meshops.models.diagnostics import (
    AutoAction,
    DefectClass,
    DefectHypothesis,
    Diagnostics,
    LateralityStatus,
    SheetScoreResult,
)
from meshops.triage.sheet_score import compute_sheet_score

# Face threshold above which triage compute prefers proxy.ply
PROXY_TRIAGE_FACES = 100_000


class JobNotFoundError(FileNotFoundError):
    """Raised when mesh_id has no job directory / original.stl."""


def _resolve_triage_mesh(paths: JobPaths) -> tuple[Path, str]:
    """Pick working or proxy for compute; return (path, label)."""
    if not paths.original_stl.is_file() and not paths.working_ply.is_file():
        raise JobNotFoundError(f"No job for mesh_id={paths.mesh_id!r} under {paths.work_root}")

    # Prefer working.ply; fall back to original.stl
    if paths.working_ply.is_file():
        mesh_path = paths.working_ply
        label = "working"
    else:
        mesh_path = paths.original_stl
        label = "original"

    mesh = load_mesh(mesh_path)
    if len(mesh.faces) > PROXY_TRIAGE_FACES and paths.proxy_ply.is_file():
        return paths.proxy_ply, "proxy"
    return mesh_path, label


def _build_hypotheses(
    *,
    sheet: SheetScoreResult,
    is_watertight: bool | None,
    is_manifold: bool | None,
    boundary_edge_count: int | None,
    non_manifold_edge_count: int | None,
) -> list[DefectHypothesis]:
    hyps: list[DefectHypothesis] = []

    if sheet.score >= 0.45:
        hyps.append(
            DefectHypothesis(
                defect_class=DefectClass.T3_SHEET,
                confidence=min(1.0, sheet.confidence * (0.5 + 0.5 * sheet.score)),
                notes="Elevated multi-feature sheet_score (Difficulty §2)",
                evidence={"sheet_score": sheet.score, "features": sheet.features.model_dump()},
            )
        )

    if is_manifold is False or (
        non_manifold_edge_count is not None and non_manifold_edge_count > 0
    ):
        conf = 0.7 if non_manifold_edge_count and non_manifold_edge_count > 10 else 0.5
        hyps.append(
            DefectHypothesis(
                defect_class=DefectClass.T1_NONMANIFOLD,
                confidence=conf,
                notes="Non-manifold edges detected (topology best-effort)",
                evidence={"non_manifold_edge_count": non_manifold_edge_count},
            )
        )

    if is_watertight is False and boundary_edge_count is not None and boundary_edge_count > 0:
        hyps.append(
            DefectHypothesis(
                defect_class=DefectClass.T2_HOLES,
                confidence=0.5,
                notes="Open boundary edges; possible holes (detect-only)",
                evidence={"boundary_edge_count": boundary_edge_count},
            )
        )

    if not hyps:
        hyps.append(
            DefectHypothesis(
                defect_class=DefectClass.T5_OTHER,
                confidence=0.3,
                notes="No strong defect signal; review evidence views",
                evidence={},
            )
        )

    return hyps


def mesh_triage(
    mesh_id: str,
    *,
    work_root: Path | str = "work",
) -> Diagnostics:
    """Classify-only triage: write diagnostics.json and return Diagnostics."""
    work_root_p = Path(work_root)
    paths = JobPaths(work_root=work_root_p, mesh_id=mesh_id)

    if not paths.job_dir.is_dir():
        raise JobNotFoundError(f"Job directory not found: {paths.job_dir}")
    if not paths.original_stl.is_file() and not paths.working_ply.is_file():
        raise JobNotFoundError(f"No mesh files for mesh_id={mesh_id!r}")

    triage_path, triage_label = _resolve_triage_mesh(paths)
    mesh = load_mesh(triage_path)

    # Stats prefer original for hash/size fidelity
    if paths.original_stl.is_file():
        digest = content_sha256(paths.original_stl)
        file_size = paths.original_stl.stat().st_size
        # Full-mesh stats when proxy is used for sheet score compute.
        stats_mesh = mesh if triage_label != "proxy" else load_mesh(paths.original_stl)
    else:
        digest = content_sha256(triage_path)
        file_size = triage_path.stat().st_size
        stats_mesh = mesh

    stats = compute_stats(
        stats_mesh,
        mesh_id=mesh_id,
        content_sha256_hex=digest,
        file_size_bytes=file_size,
        source_path=str(paths.original_stl) if paths.original_stl.is_file() else None,
    )

    # Sheet score on triage mesh (proxy for large meshes)
    sheet = compute_sheet_score(mesh)

    # Laterality: multi-component → unknown + needs_user_input (Difficulty §1)
    notes: list[str] = [f"triage_mesh={triage_label}"]
    if stats.components > 1:
        laterality = LateralityStatus.UNKNOWN
        needs_user = True
        notes.append("multi_component_laterality_unknown")
    else:
        laterality = LateralityStatus.NOT_APPLICABLE
        needs_user = False

    # Clothing-like: never escalate to delete; keep needs_user if ambiguous
    if sheet.features.clothing_penalty > 0.3 and sheet.score >= 0.35:
        needs_user = True
        notes.append("clothing_like_sheet_needs_review")

    hyps = _build_hypotheses(
        sheet=sheet,
        is_watertight=stats.is_watertight,
        is_manifold=stats.is_manifold,
        boundary_edge_count=stats.boundary_edge_count,
        non_manifold_edge_count=stats.non_manifold_edge_count,
    )

    # Ensure auto_action never delete (enum has no delete)
    assert sheet.auto_action in {AutoAction.NONE, AutoAction.REVIEW, AutoAction.ESCALATE}

    diagnostics = Diagnostics(
        mesh_id=mesh_id,
        stats=stats,
        defect_hypotheses=hyps,
        sheet_score=sheet,
        laterality_status=laterality,
        needs_user_input=needs_user,
        notes=notes,
    )

    paths.diagnostics_json.write_text(
        diagnostics.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return diagnostics
