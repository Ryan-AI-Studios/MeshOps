"""Thin MCP tool wrappers over meshops engine APIs.

No geometry / guards reimplementation. ``work_root`` is bound by the caller
(server closures). Never expose test seams (``run_orca_fn``, ``force_stub_views``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meshops import __version__
from meshops.mcp.errors import raise_if_not_ok, raise_if_slice_not_pass


def _dump_model(obj: Any) -> Any:
    """JSON-serializable dump for pydantic models; passthrough otherwise."""
    if obj is None:
        return None
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return obj


def mesh_version() -> dict[str, str]:
    return {"version": __version__}


def mesh_doctor(
    work_root: Path,
    *,
    require: list[str] | None = None,
) -> dict[str, Any]:
    from meshops.ops.doctor import expand_require, run_doctor

    req = expand_require(require)
    report = run_doctor(require=req, cwd=Path.cwd(), work_root=work_root)
    return report.model_dump(mode="json")


def mesh_list_recipes() -> dict[str, list[str]]:
    from meshops.recipes.registry import list_recipes

    return {"recipes": list_recipes()}


def mesh_ingest(work_root: Path, *, path: str) -> dict[str, Any]:
    from meshops.ingest.pipeline import ingest_stl

    result = ingest_stl(path, work_root=work_root)
    return {
        "mesh_id": result.mesh_id,
        "job_dir": str(result.job_dir),
        "original": str(result.original_path),
        "working": str(result.working_path),
        "proxy": str(result.proxy_path) if result.proxy_path else None,
        "reused": result.reused,
        "stats": _dump_model(result.stats),
    }


def mesh_triage(work_root: Path, *, mesh_id: str) -> dict[str, Any]:
    from meshops.triage.orchestrate import mesh_triage as _triage

    diag = _triage(mesh_id, work_root=work_root)
    return {"mesh_id": mesh_id, "diagnostics": _dump_model(diag)}


def mesh_render(work_root: Path, *, mesh_id: str) -> dict[str, Any]:
    from meshops.render.f3d_renderer import F3DRenderer

    result = F3DRenderer().render_job(mesh_id, work_root=work_root)
    return {
        "mesh_id": result.mesh_id,
        "rendered_from": result.rendered_from,
        "view_paths": result.view_paths,
        "depth_paths": result.depth_paths,
        "cameras": result.cameras,
        "depth_semantics": "visual_colormap_not_metric",
    }


def mesh_report(work_root: Path, *, mesh_id: str) -> dict[str, Any]:
    from meshops.report.generate import generate_report

    report_path = generate_report(mesh_id, work_root=work_root)
    return {"mesh_id": mesh_id, "report_path": str(report_path)}


def mesh_repair(
    work_root: Path,
    *,
    mesh_id: str,
    recipe: str,
    parent_rev: str | None = None,
    no_diff: bool = False,
) -> dict[str, Any]:
    from meshops.recipes.orchestrate import run_repair

    result = run_repair(
        mesh_id,
        recipe,
        work_root=work_root,
        parent_rev=parent_rev,
        no_diff=no_diff,
    )
    return {
        "mesh_id": mesh_id,
        "recipe_id": result.recipe_id,
        "rev_id": result.rev_id,
        "rev_dir": str(result.rev_dir) if result.rev_dir is not None else None,
        "manifest": _dump_model(result.manifest),
        "notes": result.notes,
        "acceptance": _dump_model(result.acceptance),
    }


def mesh_export(
    work_root: Path,
    *,
    mesh_id: str,
    out: str,
    rev: str | None = None,
) -> dict[str, Any]:
    from meshops.export_guarded import guarded_export

    payload = guarded_export(mesh_id, out, work_root=work_root, rev=rev)
    # Ensure paths are strings
    out_payload: dict[str, Any] = {}
    for k, v in payload.items():
        out_payload[k] = str(v) if isinstance(v, Path) else v
    return out_payload


def mesh_accept_revision(
    work_root: Path,
    *,
    mesh_id: str,
    rev: str,
    require_views: bool = True,
    allow_stubs: bool = True,
    require_slice: bool = False,
) -> dict[str, Any]:
    from meshops.acceptance import accept_revision

    slice_hook = None
    if require_slice:
        from meshops.slice import find_orca, make_orca_hook

        if find_orca(require=False) is not None:
            slice_hook = make_orca_hook(mesh_id=mesh_id, work_root=work_root)

    result = accept_revision(
        mesh_id,
        rev,
        work_root=work_root,
        require_views=require_views,
        allow_stubs=allow_stubs,
        require_slice=require_slice,
        slice_hook=slice_hook,
    )
    raise_if_not_ok(result, what=f"accept_revision mesh_id={mesh_id!r} rev={rev!r}")
    return {
        "ok": result.ok,
        "mesh_id": mesh_id,
        "rev": rev,
        "acceptance": result.model_dump(mode="json"),
    }


def mesh_accept_candidate(
    work_root: Path,
    *,
    baseline_path: str,
    candidate_path: str,
    require_views: bool = True,
    allow_stubs: bool = True,
    require_slice: bool = False,
) -> dict[str, Any]:
    from meshops.acceptance import accept_candidate

    # Attach live Orca hook when require_slice (CLI parity). mesh_id optional —
    # make_orca_hook accepts mesh_id=None and still runs on candidate_path.
    slice_hook = None
    if require_slice:
        from meshops.slice import find_orca, make_orca_hook

        if find_orca(require=False) is not None:
            slice_hook = make_orca_hook(work_root=work_root)

    result = accept_candidate(
        Path(baseline_path),
        Path(candidate_path),
        require_views=require_views,
        allow_stubs=allow_stubs,
        require_slice=require_slice,
        slice_hook=slice_hook,
    )
    raise_if_not_ok(
        result,
        what=f"accept_candidate baseline={baseline_path!r} candidate={candidate_path!r}",
    )
    return {
        "ok": result.ok,
        "baseline_path": baseline_path,
        "candidate_path": candidate_path,
        "acceptance": result.model_dump(mode="json"),
    }


def mesh_promote_working(
    work_root: Path,
    *,
    mesh_id: str,
    rev: str,
) -> dict[str, Any]:
    from meshops.acceptance import promote_working

    promo = promote_working(mesh_id, rev, work_root=work_root)
    acceptance = promo.get("acceptance")
    return {
        "mesh_id": mesh_id,
        "rev": rev,
        "working_ply": str(promo.get("working_ply", "")),
        "working_manifest": str(promo.get("working_manifest", "")),
        "content_sha256": promo.get("content_sha256"),
        "acceptance": _dump_model(acceptance),
    }


def mesh_diff_views(work_root: Path, *, mesh_id: str, rev: str) -> dict[str, Any]:
    from meshops.recipes.diff_views import render_rev_diff

    payload = render_rev_diff(mesh_id, rev, work_root=work_root)
    # stringify paths
    out: dict[str, Any] = dict(payload)
    for key in ("baseline", "candidate"):
        if key in out and out[key] is not None:
            out[key] = str(out[key])
    return out


def mesh_slice(
    work_root: Path,
    *,
    mesh_id: str,
    rev: str | None = None,
    profile: str = "default",
    orient: bool = False,
    arrange: bool = False,
    allow_reorient_retry: bool = False,
) -> dict[str, Any]:
    """Resolve candidate like CLI, then ``run_slice`` (no ``run_orca_fn`` — R7)."""
    from meshops.jobstore.paths import JobPaths
    from meshops.revs.store import resolve_rev_dir
    from meshops.slice import find_orca, run_slice
    from meshops.slice.errors import SliceError

    if find_orca(require=False) is None:
        raise SliceError(
            "OrcaSlicer not found (set MESHOPS_ORCA or install 2.4.x)",
            code="orca_not_found",
        )
    paths = JobPaths(work_root=work_root, mesh_id=mesh_id)
    if not paths.job_dir.is_dir():
        raise SliceError(
            f"job not found: {paths.job_dir}",
            code="job_not_found",
            details={"mesh_id": mesh_id},
        )

    cand: Path | None
    if rev is not None:
        rev_dir = resolve_rev_dir(paths, rev)
        cand = None
        for name in ("mesh.stl", "mesh.ply", "result.stl"):
            p = rev_dir / name
            if p.is_file():
                cand = p
                break
        if cand is None:
            stls = sorted(rev_dir.glob("*.stl")) + sorted(rev_dir.glob("*.ply"))
            if stls:
                cand = stls[0]
        if cand is None:
            raise SliceError(
                f"no mesh in rev {rev}: {rev_dir}",
                code="missing_candidate",
            )
    elif paths.working_ply.is_file():
        cand = paths.working_ply
    elif paths.original_stl.is_file():
        cand = paths.original_stl
    else:
        raise SliceError(
            f"no candidate mesh under {paths.job_dir}",
            code="missing_candidate",
        )

    result = run_slice(
        cand,
        mesh_id=mesh_id,
        work_root=work_root,
        slice_profile=profile,
        orient=1 if orient else 0,
        arrange=1 if arrange else 0,
        allow_reorient_retry=allow_reorient_retry,
    )
    raise_if_slice_not_pass(result)
    return {
        "ok": True,
        "mesh_id": mesh_id,
        "slice": result.model_dump(mode="json"),
    }


def mesh_extract_roi(
    work_root: Path,
    *,
    mesh_id: str,
    bbox: list[float] | None = None,
    from_sheet_heuristic: bool = False,
) -> dict[str, Any]:
    from meshops.escalate.errors import EscalateError
    from meshops.escalate.roi import create_roi_bbox, create_roi_from_sheet_heuristic

    if from_sheet_heuristic and bbox is None:
        manifest = create_roi_from_sheet_heuristic(mesh_id, work_root=work_root)
    elif bbox is not None:
        if len(bbox) != 6:
            raise EscalateError(
                "bbox must be [xmin,ymin,zmin,xmax,ymax,zmax] (6 floats)",
                code="invalid_bbox",
            )
        manifest = create_roi_bbox(
            mesh_id,
            bbox[0:3],
            bbox[3:6],
            work_root=work_root,
            source="manual",
        )
    else:
        raise EscalateError(
            "provide bbox [xmin,ymin,zmin,xmax,ymax,zmax] or from_sheet_heuristic=true",
            code="invalid_bbox",
        )
    return {
        "mesh_id": mesh_id,
        "roi": manifest.model_dump(mode="json"),
    }


def mesh_preview_t3(
    work_root: Path,
    *,
    mesh_id: str,
    roi: str | None = None,
) -> dict[str, Any]:
    """Honest T3 preview package. ``ok=False`` is intentional honesty — return payload."""
    from meshops.escalate.preview_t3 import preview_t3

    result = preview_t3(mesh_id, roi, work_root=work_root)
    return {
        "ok": result.ok,
        "preview": True,
        "mesh_id": result.mesh_id,
        "preview_id": result.preview_id,
        "roi_id": result.roi_id,
        "preview_dir": str(result.preview_dir),
        "notes": result.notes,
        "honesty_note": result.honesty_note,
        "may_promote_working": False,
        "may_claim_fixed": False,
        "paths": result.paths,
    }


def mesh_blender_handoff(
    work_root: Path,
    *,
    mesh_id: str,
    roi: str,
    timeout: float = 300.0,
) -> dict[str, Any]:
    from meshops.escalate.handoff import build_handoff

    manifest = build_handoff(
        mesh_id,
        roi,
        work_root=work_root,
        timeout_s=timeout,
    )
    return {
        "mesh_id": mesh_id,
        "handoff": manifest.model_dump(mode="json"),
        "honesty_note": "handoff package ready — not autonomous hero fixed (N6)",
    }


def mesh_import_sculpt(
    work_root: Path,
    *,
    mesh_id: str,
    path: str,
    approve: bool,
    no_diff: bool = False,
) -> dict[str, Any]:
    """Import sculpt STL. ``approve`` is required (no default at tool schema — R8)."""
    from meshops.escalate.import_sculpt import import_sculpt

    result = import_sculpt(
        mesh_id,
        path,
        approve=approve,
        work_root=work_root,
        no_diff=no_diff,
    )
    raise_if_not_ok(result, what=f"import_sculpt mesh_id={mesh_id!r}")
    return {
        "ok": result.ok,
        "mesh_id": result.mesh_id,
        "rev_id": result.rev_id,
        "rev_dir": str(result.rev_dir) if result.rev_dir is not None else None,
        "recipe_id": result.recipe_id,
        "notes": result.notes,
        "honesty_note": result.honesty_note,
        "paths": result.paths,
        "acceptance": _dump_model(result.acceptance),
        "promoted_to_working": False,
    }


def design_from_spec(
    work_root: Path,
    *,
    template: str = "bracket_m4",
    hole_spacing: float = 40.0,
    wall: float = 3.0,
    thickness: float = 4.0,
    hole_diameter: float = 4.2,
    timeout: float = 60.0,
    no_diff: bool = False,
) -> dict[str, Any]:
    try:
        from meshops.design import BracketParams, design_from_template
    except ImportError as exc:
        raise RuntimeError(
            f"design package import failed (install meshops[design]): {exc}"
        ) from exc

    params = BracketParams(
        hole_spacing_mm=hole_spacing,
        wall_mm=wall,
        thickness_mm=thickness,
        hole_diameter_mm=hole_diameter,
    )
    result = design_from_template(
        template,
        params=params,
        work_root=work_root,
        timeout_s=timeout,
        no_diff=no_diff,
    )
    raise_if_not_ok(result, what=f"design_from_spec template={template!r}")
    return {
        "ok": result.ok,
        "mesh_id": result.mesh_id,
        "job_dir": str(result.job_dir) if result.job_dir is not None else None,
        "template": template,
        "params": params.model_dump(mode="json"),
        "paths": result.paths,
        "manifest": _dump_model(result.manifest),
        "notes": result.notes,
        "acceptance": _dump_model(result.acceptance),
        "slice": "skipped",
        "honesty_note": "absolute validate is primary gate; self-baseline safety net only",
    }


def design_run(
    work_root: Path,
    *,
    source: str,
    timeout: float = 60.0,
    no_diff: bool = False,
) -> dict[str, Any]:
    try:
        from meshops.design import run_design_pipeline
    except ImportError as exc:
        raise RuntimeError(
            f"design package import failed (install meshops[design]): {exc}"
        ) from exc

    result = run_design_pipeline(
        Path(source),
        work_root=work_root,
        timeout_s=timeout,
        no_diff=no_diff,
    )
    raise_if_not_ok(result, what=f"design_run source={source!r}")
    return {
        "ok": result.ok,
        "mesh_id": result.mesh_id,
        "job_dir": str(result.job_dir) if result.job_dir is not None else None,
        "paths": result.paths,
        "manifest": _dump_model(result.manifest),
        "notes": result.notes,
        "acceptance": _dump_model(result.acceptance),
        "slice": "skipped",
        "honesty_note": "absolute validate is primary gate; self-baseline safety net only",
    }


def organic_create(
    work_root: Path,
    *,
    prompt: str,
    style: str = "",
    refs: list[str] | None = None,
    recipe: str = "simple_bust",
    session_id: str | None = None,
) -> dict[str, Any]:
    from meshops.organic import create_session

    manifest = create_session(
        prompt,
        style_notes=style,
        refs=list(refs) if refs else None,
        default_recipe=recipe,
        session_id=session_id,
        work_root=work_root,
    )
    return {
        "session_id": manifest.session_id,
        "status": manifest.status,
        "default_recipe": manifest.default_recipe,
        "notes": manifest.notes,
        "manifest": manifest.model_dump(mode="json"),
    }


def organic_pass(
    work_root: Path,
    *,
    session_id: str,
    recipe: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one organic pass. No ``force_stub_views`` in schema (R7)."""
    from meshops.organic import run_pass

    result = run_pass(
        session_id,
        recipe=recipe,
        params=params,
        work_root=work_root,
    )
    # PassResult.ok False should surface as error for agent clarity
    raise_if_not_ok(result, what=f"organic_pass session_id={session_id!r}")
    return {
        "ok": result.ok,
        "pass_id": result.pass_id,
        "recipe": result.recipe,
        "mesh_path": str(result.mesh_path) if result.mesh_path else None,
        "view_paths": result.view_paths,
        "view_kind": result.view_kind,
        "returncode": result.returncode,
        "duration_s": result.duration_s,
        "blender_version": result.blender_version,
        "messages": result.messages,
        "error_code": result.error_code,
    }


def organic_status(work_root: Path, *, session_id: str) -> dict[str, Any]:
    from meshops.organic import load_session

    paths, manifest = load_session(session_id, work_root=work_root)
    return {
        "session_id": manifest.session_id,
        "status": manifest.status,
        "passes": manifest.passes,
        "pass_count": len(manifest.passes),
        "final_mesh_id": manifest.final_mesh_id,
        "blender_version": manifest.blender_version,
        "notes": manifest.notes,
        "organic_dir": str(paths.organic_dir),
        "manifest": manifest.model_dump(mode="json"),
    }


def organic_plateau(
    work_root: Path,
    *,
    session_id: str,
    reason: str,
) -> dict[str, Any]:
    from meshops.organic import mark_plateau

    record = mark_plateau(session_id, reason, work_root=work_root)
    return {
        "session_id": record.session_id,
        "allows_hosted_fallback": record.allows_hosted_fallback,
        "criteria_met": record.criteria_met,
        "pass_count": record.pass_count,
        "reason": record.reason,
        "plateau": record.model_dump(mode="json"),
    }


def organic_finalize(
    work_root: Path,
    *,
    session_id: str,
    accept: bool = False,
) -> dict[str, Any]:
    from meshops.organic import finalize_session

    result = finalize_session(session_id, work_root=work_root, accept=accept)
    raise_if_not_ok(result, what=f"organic_finalize session_id={session_id!r}")
    return {
        "ok": result.ok,
        "session_id": result.session_id,
        "mesh_id": result.mesh_id,
        "job_dir": str(result.job_dir) if result.job_dir else None,
        "triage_summary": result.triage_summary,
        "honesty_message": result.honesty_message,
        "messages": result.messages,
        "error_code": result.error_code,
        "acceptance": _dump_model(result.acceptance),
    }
