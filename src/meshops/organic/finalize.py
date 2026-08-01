"""Finalize organic session → untrusted ingest + triage + job views (B12)."""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meshops.acceptance.honesty import HONESTY_MESSAGE
from meshops.acceptance.pack import accept_candidate
from meshops.guards.policy import GuardPolicy
from meshops.ingest.pipeline import ingest_stl
from meshops.jobstore.paths import JobPaths
from meshops.organic.errors import OrganicError
from meshops.organic.models import FinalizeResult
from meshops.organic.report import write_session_report
from meshops.organic.session import load_session, save_manifest
from meshops.triage.orchestrate import mesh_triage

# Organic honesty beyond pack message
ORGANIC_HONESTY = (
    "Authored organic (agent-first Blender recipes) — not a print-ready hero sculpt "
    "(N6). Ingested as untrusted mesh; triage + job views required."
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _prefer_stub_views() -> bool:
    if stub := os.environ.get("MESHOPS_STUB_DIFF", "").strip().lower():
        return stub in {"1", "true", "yes", "on"}
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _render_job_views(
    mesh_id: str,
    *,
    work_root: Path,
) -> tuple[list[str], list[str]]:
    """Always render job views into work/<mesh_id>/views/ (B12).

    Session pass views must NOT satisfy this step.
    Returns (view_paths, notes).
    """
    job = JobPaths(work_root=work_root, mesh_id=mesh_id)
    job.views_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    if _prefer_stub_views():
        notes.append("finalize_views_stub_ci_or_MESHOPS_STUB_DIFF")
        paths: list[str] = []
        for name in ("front", "left", "three_quarter", "three_quarter_depth"):
            dest = job.views_dir / f"{name}.png"
            dest.write_bytes(_MIN_PNG)
            paths.append(str(dest))
        return paths, notes

    try:
        from meshops.render.f3d_renderer import F3DRenderer

        result = F3DRenderer().render_job(mesh_id, work_root=work_root, include_depth=True)
        paths = list(result.view_paths) + list(result.depth_paths)
        if not paths:
            raise RuntimeError("render_job returned no view paths")
        return paths, notes
    except Exception as exc:
        notes.append(f"finalize_views_stub: {type(exc).__name__}: {exc}")
        paths = []
        for name in ("front", "left", "three_quarter", "three_quarter_depth"):
            dest = job.views_dir / f"{name}.png"
            dest.write_bytes(_MIN_PNG)
            paths.append(str(dest))
        return paths, notes


def finalize_session(
    session_id: str,
    *,
    work_root: Path | str = "work",
    accept: bool = False,
) -> FinalizeResult:
    """B12 binding sequence: final.stl → ingest → triage → job views → optional accept."""
    work_root_p = Path(work_root)
    paths, manifest = load_session(session_id, work_root=work_root_p)

    if not manifest.passes:
        raise OrganicError(
            "finalize requires at least one successful pass",
            code="finalize_no_pass",
            details={"session_id": manifest.session_id},
        )

    last_pass = manifest.passes[-1]
    src_mesh = paths.pass_dir(last_pass) / "mesh.stl"
    if not src_mesh.is_file() or src_mesh.stat().st_size <= 0:
        raise OrganicError(
            f"last pass mesh missing: {src_mesh}",
            code="finalize_no_pass",
            details={"pass_id": last_pass},
        )

    # 1) Copy last successful mesh → organic/final.stl
    shutil.copy2(src_mesh, paths.final_stl)

    # 2) ingest_stl → content-hash mesh_id
    try:
        ing = ingest_stl(paths.final_stl, work_root=work_root_p)
    except Exception as exc:
        raise OrganicError(
            f"ingest failed: {exc}",
            code="ingest_failed",
            details={"error": str(exc)},
        ) from exc

    mesh_id = ing.mesh_id
    job = JobPaths(work_root=work_root_p, mesh_id=mesh_id)
    messages: list[str] = []

    # 3) ALWAYS triage on new mesh_id
    triage_summary: dict[str, Any] | None = None
    try:
        diag = mesh_triage(mesh_id, work_root=work_root_p)
        triage_summary = {
            "mesh_id": mesh_id,
            "stats": diag.stats.model_dump(mode="json"),
            "defect_hypotheses": [h.model_dump(mode="json") for h in diag.defect_hypotheses],
            "notes": list(diag.notes),
        }
    except Exception as exc:
        raise OrganicError(
            f"triage failed on finalized mesh: {exc}",
            code="ingest_failed",
            details={"mesh_id": mesh_id, "stage": "triage", "error": str(exc)},
        ) from exc

    # 4) ALWAYS render job views (not session pass views)
    view_paths, view_notes = _render_job_views(mesh_id, work_root=work_root_p)
    messages.extend(view_notes)
    if not any(Path(p).is_file() for p in view_paths):
        raise OrganicError(
            "finalize: job views missing after render",
            code="pass_no_views",
            details={"mesh_id": mesh_id, "views_dir": str(job.views_dir)},
        )

    acceptance = None
    honesty = f"{ORGANIC_HONESTY} {HONESTY_MESSAGE}"

    # 5) Optional accept with for_sculpt + require_views on JOB views
    if accept:
        try:
            acceptance = accept_candidate(
                job.original_stl,
                job.original_stl,  # self-baseline for authored mesh
                policy=GuardPolicy.for_sculpt(),
                view_paths=view_paths,
                require_views=True,
                allow_stubs=True,
                view_notes=[*view_notes, "organic_finalize", "authored_organic_not_print_hero"],
            )
            if acceptance.honesty_message:
                honesty = f"{ORGANIC_HONESTY} {acceptance.honesty_message}"
        except Exception as exc:
            messages.append(f"accept_candidate failed: {exc}")
            return FinalizeResult(
                ok=False,
                session_id=manifest.session_id,
                mesh_id=mesh_id,
                job_dir=job.job_dir,
                triage_summary=triage_summary,
                acceptance=None,
                honesty_message=honesty,
                error_code="ingest_failed",
                messages=messages,
            )

    # 6) finalize.json + status
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "session_id": manifest.session_id,
        "mesh_id": mesh_id,
        "job_dir": str(job.job_dir),
        "final_stl": str(paths.final_stl),
        "source_pass": last_pass,
        "created_at": _now_iso(),
        "accepted": bool(accept and acceptance is not None and acceptance.ok),
        "honesty_message": honesty,
        "view_paths": view_paths,
        "triage_summary": triage_summary,
    }
    if acceptance is not None:
        payload["acceptance"] = acceptance.model_dump(mode="json")

    paths.finalize_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest.status = "finalized"
    manifest.final_mesh_id = mesh_id
    if "authored_organic_not_print_hero" not in manifest.notes:
        manifest.notes.append("authored_organic_not_print_hero")
    save_manifest(paths, manifest)
    write_session_report(
        paths,
        manifest,
        extra_lines=[
            "## Finalize",
            "",
            f"- mesh_id: `{mesh_id}`",
            f"- job_dir: `{job.job_dir}`",
            f"- accept: {accept}",
            f"- honesty: {honesty}",
        ],
    )

    ok = True
    if accept and acceptance is not None:
        ok = bool(acceptance.ok)

    return FinalizeResult(
        ok=ok,
        session_id=manifest.session_id,
        mesh_id=mesh_id,
        job_dir=job.job_dir,
        triage_summary=triage_summary,
        acceptance=acceptance,
        honesty_message=honesty,
        error_code=None if ok else "ingest_failed",
        messages=messages,
    )
