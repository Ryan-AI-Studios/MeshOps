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


def design_organic_api(
    work_root: Path,
    *,
    justify: str,
    session_id: str | None = None,
    plateau: str | None = None,
    prompt: str = "",
    provider: str = "meshy",
    views: list[str] | None = None,
    views_from: str = "latest",
    accept: bool = False,
    accept_policy: str = "export",
) -> dict[str, Any]:
    """Hosted multi-view fallback. Raises HostedError on gate/provider fail (R1)."""
    from meshops.hosted import HostedError, run_hosted_fallback
    from meshops.hosted.views import ViewsFrom

    vf_raw = (views_from or "latest").strip().lower()
    if vf_raw not in ("latest", "pass", "explicit"):
        raise HostedError(
            f"invalid views_from: {views_from!r}",
            code="multiview_required",
        )
    vf: ViewsFrom = vf_raw  # type: ignore[assignment]

    result = run_hosted_fallback(
        session_id=session_id,
        work_root=work_root,
        plateau=Path(plateau) if plateau else None,
        views_from=vf,
        view_paths=[Path(v) for v in views] if views else None,
        prompt=prompt,
        justify=justify,
        provider=provider,
        accept=accept,
        accept_policy=accept_policy,
    )
    raise_if_not_ok(result, what="design_organic_api")
    return {
        "ok": result.ok,
        "session_id": result.session_id,
        "mesh_id": result.mesh_id,
        "job_dir": result.job_dir,
        "provider": result.provider,
        "provider_task_id": result.provider_task_id,
        "justification": _dump_model(result.justification),
        "view_paths": result.view_paths,
        "diagnostics": result.diagnostics,
        "honesty": result.honesty,
        "messages": result.messages,
        "error_code": result.error_code,
        "acceptance": _dump_model(result.acceptance),
        "schema_version": result.schema_version,
    }


# ---------------------------------------------------------------------------
# Proportion (0012-0016) — measurement / authoring only (N6)
# ---------------------------------------------------------------------------


def _resolve_tool_path(path: str | Path, work_root: Path) -> Path:
    """Resolve relative paths against work_root, else cwd (R8 / 0008 R3)."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    wr_cand = (work_root / p).resolve()
    if wr_cand.exists() or wr_cand.parent.exists():
        return wr_cand
    return (Path.cwd() / p).resolve()


def mesh_proportion_template(
    work_root: Path, *, out: str = "landmarks_assist.json"
) -> dict[str, Any]:
    """Write blank landmarks_assist.json. Authoring only — not mesh/print success (N6)."""
    from meshops.proportion.template import blank_assist_document, write_template

    path = write_template(_resolve_tool_path(out, work_root))
    return {"ok": True, "path": str(path), "assist": blank_assist_document()}


def mesh_proportion_analyze(
    work_root: Path,
    *,
    views_dir: str,
    landmarks: str | None = None,
    height_m: float | None = None,
    out: str | None = None,
    overlays: bool = False,
    partial_ok: bool = False,
    attach_session: str | None = None,
) -> dict[str, Any]:
    """Analyze multi-view package → proportion_report. Measurement only (N6)."""
    from meshops.proportion.analyze import analyze_proportion
    from meshops.proportion.capture import attach_report_to_organic_session
    from meshops.proportion.errors import ProportionError

    views = _resolve_tool_path(views_dir, work_root)
    lm = _resolve_tool_path(landmarks, work_root) if landmarks else None
    out_dir = _resolve_tool_path(out, work_root) if out else None
    if attach_session and out_dir is None:
        raise ProportionError(
            "attach_session requires out (report must be written)",
            code="capture_failed",
            details={"attach_session": attach_session},
        )
    report = analyze_proportion(
        views,
        landmarks_path=lm,
        height_m=height_m,
        out_dir=out_dir,
        partial_ok=partial_ok,
        overlays=overlays,
    )
    payload = report.model_dump(mode="json")
    payload["ok"] = True
    if out_dir is not None:
        report_path = out_dir / "proportion_report.json"
        payload["report_path"] = str(report_path)
        if attach_session:
            dest = attach_report_to_organic_session(
                attach_session,
                report_path,
                work_root=work_root,
            )
            payload["attach_dest"] = str(dest)
    return payload


def mesh_proportion_show(work_root: Path, *, report: str) -> dict[str, Any]:
    """Load proportion_report.json → {report, markdown}. Measurement only (N6)."""
    from meshops.proportion.analyze import load_report, report_to_markdown

    path = _resolve_tool_path(report, work_root)
    rep = load_report(path)
    return {
        "report": rep.model_dump(mode="json"),
        "markdown": report_to_markdown(rep),
    }


def mesh_proportion_scaffold(
    work_root: Path,
    *,
    out: str,
    dual: bool = False,
    mode: str | None = None,
    height_m: float | None = None,
    subject: str | None = None,
    pose: str = "a_pose",
    with_template: bool = False,
    stub_images: bool = False,
    include_back_stub: bool = False,
    include_top_stub: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Create multi-view package layout + checklist. Layout only — not mesh/print (N6)."""
    from meshops.proportion.scaffold import scaffold_package

    out_path = _resolve_tool_path(out, work_root)
    resolved_mode = "dual" if dual else (mode.strip().lower() if mode else "single")
    result = scaffold_package(
        out_path,
        dual=dual or resolved_mode == "dual",
        mode=resolved_mode if not dual else "dual",  # type: ignore[arg-type]
        height_m=height_m,
        subject=subject,
        pose=pose,
        with_template=with_template,
        stub_images=stub_images,
        include_back_stub=include_back_stub,
        include_top_stub=include_top_stub,
        force=force,
    )
    return {
        "ok": True,
        "mode": result.mode,
        "paths": [str(p) for p in result.paths],
        "analyze_hint": str(result.analyze_hint) if result.analyze_hint is not None else None,
    }


def mesh_proportion_guides(
    work_root: Path,
    *,
    report: str,
    out: str,
    format: str = "both",
    seeds: bool = False,
    front_plane_seeds: bool = False,
    quiet_null_y: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Emit LM_* guides / optional SEED_* from report. Authoring aids only (N6)."""
    from meshops.proportion.guides import run_guides

    fmt = (format or "both").strip().lower()
    if fmt not in ("bpy", "json", "both"):
        raise ValueError("--format must be bpy, json, or both")
    return run_guides(
        _resolve_tool_path(report, work_root),
        _resolve_tool_path(out, work_root),
        format=fmt,  # type: ignore[arg-type]
        seeds=seeds,
        front_plane_seeds=front_plane_seeds,
        quiet_null_y=quiet_null_y,
        force=force,
    )


def mesh_proportion_capture(
    work_root: Path,
    *,
    source: str | None = None,
    in_path: str | None = None,
    out: str | None = None,
    views_dir: str | None = None,
    pose: str | None = None,
    multi_figure: bool | None = None,
    merge: str | None = None,
    prefer_merge: bool = False,
    default_confidence: float | None = None,
    force: bool = False,
    emit_dump_script: str | None = None,
    attach_session: str | None = None,
) -> dict[str, Any]:
    """Fill landmarks_assist from px/dump/reproject. Authoring only — CAPTURE_HONESTY / N6."""
    from meshops.proportion.capture import run_capture

    return run_capture(
        source=source,  # type: ignore[arg-type]
        in_path=_resolve_tool_path(in_path, work_root) if in_path else None,
        out_path=_resolve_tool_path(out, work_root) if out else None,
        views_dir=_resolve_tool_path(views_dir, work_root) if views_dir else None,
        pose=pose,
        multi_figure=multi_figure,
        merge_path=_resolve_tool_path(merge, work_root) if merge else None,
        prefer_merge=prefer_merge,
        default_confidence=default_confidence,
        force=force,
        emit_dump_script_path=(
            _resolve_tool_path(emit_dump_script, work_root) if emit_dump_script else None
        ),
        attach_session=attach_session,
        work_root=work_root,
    )


def mesh_proportion_depth_samples(
    work_root: Path,
    *,
    report: str,
    out: str,
    mesh: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Export depth_at_landmarks (+ optional mesh deltas). Authoring only — DEPTH_HONESTY / N6."""
    from meshops.proportion.depth_samples import run_depth_samples

    # Preserve trailing directory separator intent (R1); Path resolve strips it.
    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )

    return run_depth_samples(
        _resolve_tool_path(report, work_root),
        out_arg,
        mesh=_resolve_tool_path(mesh, work_root) if mesh else None,
        force=force,
    )


def mesh_proportion_templates(work_root: Path) -> dict[str, Any]:
    """List body template ids. Authoring only — TEMPLATE_HONESTY / N6."""
    from meshops.proportion.body_template import list_body_templates
    from meshops.proportion.honesty import TEMPLATE_HONESTY

    _ = work_root  # catalog tools always receive work_root
    return {
        "ok": True,
        "templates": list_body_templates(),
        "honesty": TEMPLATE_HONESTY,
    }


def mesh_proportion_apply_template(
    work_root: Path,
    *,
    report: str,
    template: str,
    out: str,
    force: bool = False,
) -> dict[str, Any]:
    """Apply body template. Authoring only — TEMPLATE_HONESTY / N6."""
    from meshops.proportion.body_template import apply_body_template

    return apply_body_template(
        _resolve_tool_path(report, work_root),
        template,
        _resolve_tool_path(out, work_root),
        force=force,
    )


def mesh_proportion_anatomy_profiles(work_root: Path) -> dict[str, Any]:
    """List anatomy profile packs. Authoring only — ANATOMY_PROFILE_HONESTY / N6."""
    from meshops.proportion.anatomy_profile import list_anatomy_profiles
    from meshops.proportion.honesty import ANATOMY_PROFILE_HONESTY

    _ = work_root
    return {
        "ok": True,
        "profiles": list_anatomy_profiles(),
        "honesty": ANATOMY_PROFILE_HONESTY,
    }


def mesh_proportion_blockout_recipe(
    work_root: Path,
    *,
    report: str,
    out: str,
    format: str = "both",
    depth_at_landmarks: str | None = None,
    limbs: bool = True,
    force: bool = False,
    torso: str = "trap",
    glute: str = "oval",
    nofuse: bool = False,
    join_ready: bool = False,
    soft_density: str = "full",
    breast_tilt_deg: float | None = None,
    template_applied: str | None = None,
    profiles: str | None = None,
    skeleton: str | None = None,
    face: bool = False,
    hair: str = "none",
    neckline: str = "none",
    hands: bool = False,
    feet: bool = False,
    fingers: str = "mitten",
    toes: str = "wedge",
) -> dict[str, Any]:
    """Emit RECIPE_* blockout primitives. Authoring only — RECIPE_HONESTY / N6."""
    from meshops.proportion.blockout_recipe import run_blockout_recipe

    fmt = (format or "both").strip().lower()
    if fmt not in ("bpy", "json", "both"):
        raise ValueError("--format must be bpy, json, or both")
    torso_mode = (torso or "trap").strip().lower()
    if torso_mode not in ("trap", "ovals"):
        raise ValueError("--torso must be trap or ovals")
    glute_mode = (glute or "oval").strip().lower()
    if glute_mode not in ("oval", "two_spheres"):
        raise ValueError("--glute must be oval or two_spheres")
    hair_tier = (hair or "none").strip().lower()
    if hair_tier not in ("none", "short", "bun", "long_proxy"):
        raise ValueError("--hair must be none, short, bun, or long_proxy")
    neckline_tier = (neckline or "none").strip().lower()
    if neckline_tier not in ("none", "crew", "v_proxy"):
        raise ValueError("--neckline must be none, crew, or v_proxy")
    fingers_tier = (fingers or "mitten").strip().lower()
    if fingers_tier not in ("none", "mitten", "full"):
        raise ValueError("--fingers must be none, mitten, or full")
    toes_tier = (toes or "wedge").strip().lower()
    if toes_tier not in ("none", "wedge", "full"):
        raise ValueError("--toes must be none, wedge, or full")
    soft_density_mode = (soft_density or "full").strip().lower()
    if soft_density_mode not in ("full", "compact"):
        raise ValueError("--soft-density must be full or compact")
    # Preserve trailing directory separator intent (R1); Path resolve strips it.
    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )
    return run_blockout_recipe(
        _resolve_tool_path(report, work_root),
        out_arg,
        format=fmt,  # type: ignore[arg-type]
        depth_at_landmarks=(
            _resolve_tool_path(depth_at_landmarks, work_root) if depth_at_landmarks else None
        ),
        limbs=limbs,
        force=force,
        torso=torso_mode,  # type: ignore[arg-type]
        glute=glute_mode,  # type: ignore[arg-type]
        nofuse=nofuse,
        join_ready=bool(join_ready),
        soft_density=soft_density_mode,  # type: ignore[arg-type]
        breast_tilt_deg=breast_tilt_deg,
        template_applied=(
            _resolve_tool_path(template_applied, work_root) if template_applied else None
        ),
        profiles=profiles,
        skeleton=(_resolve_tool_path(skeleton, work_root) if skeleton else None),
        face=bool(face),
        hair=hair_tier,  # type: ignore[arg-type]
        neckline=neckline_tier,  # type: ignore[arg-type]
        hands=bool(hands),
        feet=bool(feet),
        fingers=fingers_tier,  # type: ignore[arg-type]
        toes=toes_tier,  # type: ignore[arg-type]
    )


def mesh_proportion_blockout_emit_setup(
    work_root: Path,
    *,
    recipe: str,
    out: str,
    force: bool = False,
) -> dict[str, Any]:
    """Re-emit setup_blockout_recipe.py from recipe JSON. Authoring only — RECIPE_HONESTY / N6."""
    from meshops.proportion.blockout_recipe import run_blockout_emit_setup

    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )
    return run_blockout_emit_setup(
        _resolve_tool_path(recipe, work_root),
        out_arg,
        force=force,
    )


def mesh_proportion_blockout_open_setup(
    work_root: Path,
    *,
    setup: str,
    spawn: bool = False,
    background: bool = False,
) -> dict[str, Any]:
    """Print (or spawn) abs Blender --python setup. Authoring only — SETUP_LAUNCH_HONESTY / N6."""
    from meshops.proportion.setup_launch import run_blockout_open_setup

    return run_blockout_open_setup(
        _resolve_tool_path(setup, work_root),
        spawn=bool(spawn),
        background=bool(background),
    )


def mesh_proportion_blockout_fuse_plan(
    work_root: Path,
    *,
    recipe: str,
    out: str,
    force: bool = False,
) -> dict[str, Any]:
    """Write fuse_plan.json procedure. Authoring weld only — FUSE_HONESTY / N6."""
    from meshops.proportion.fuse_plan import run_blockout_fuse_plan

    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )
    return run_blockout_fuse_plan(
        _resolve_tool_path(recipe, work_root),
        out_arg,
        force=force,
    )


def mesh_proportion_blockout_validate_constraints(
    work_root: Path,
    *,
    recipe: str,
    out: str,
    report: str | None = None,
    template_applied: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Validate named-role hard constraints. Authoring only — CONSTRAINT_HONESTY / N6.

    template_applied is optional: missing file skips soft template rules (hard C_*
    still run). A directory without the file also tries parent/template_applied.json.
    """
    from meshops.proportion.constraints import run_blockout_validate_constraints

    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )
    return run_blockout_validate_constraints(
        _resolve_tool_path(recipe, work_root),
        out_arg,
        report=_resolve_tool_path(report, work_root) if report else None,
        template_applied=(
            _resolve_tool_path(template_applied, work_root) if template_applied else None
        ),
        force=force,
    )


def mesh_proportion_blockout_optimize(
    work_root: Path,
    *,
    recipe: str,
    out: str,
    mode: str = "fast",
    freeze_feet: bool = True,
    mesh: str | None = None,
    report: str | None = None,
    template_applied: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Constrained free-DOF optimize. Authoring only — OPTIMIZE_HONESTY / N6.

    template_applied is optional: missing file skips soft template rules (hard C_*
    still run). A directory without the file also tries parent/template_applied.json.
    """
    from meshops.proportion.constraints import run_blockout_optimize

    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )
    return run_blockout_optimize(
        _resolve_tool_path(recipe, work_root),
        out_arg,
        mode=mode,
        freeze_feet=freeze_feet,
        mesh=_resolve_tool_path(mesh, work_root) if mesh else None,
        report=_resolve_tool_path(report, work_root) if report else None,
        template_applied=(
            _resolve_tool_path(template_applied, work_root) if template_applied else None
        ),
        force=force,
    )


def mesh_proportion_skeleton_build(
    work_root: Path,
    *,
    report: str,
    out: str,
    format: str = "json",
    template_applied: str | None = None,
    depth_at_landmarks: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Emit joint/bone skeleton graph. Authoring only — SKELETON_HONESTY / N6."""
    from meshops.proportion.skeleton import run_skeleton_build

    fmt = (format or "json").strip().lower()
    if fmt not in ("json", "bpy", "both"):
        raise ValueError("--format must be json, bpy, or both")
    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )
    return run_skeleton_build(
        _resolve_tool_path(report, work_root),
        out_arg,
        format=fmt,  # type: ignore[arg-type]
        force=force,
        template_applied=(
            _resolve_tool_path(template_applied, work_root) if template_applied else None
        ),
        depth_at_landmarks=(
            _resolve_tool_path(depth_at_landmarks, work_root) if depth_at_landmarks else None
        ),
    )


def mesh_proportion_depth_heatmap(
    work_root: Path,
    *,
    samples: str,
    out: str,
    deltas: str | None = None,
    underlay: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Glance heatmap PNG + meta from depth samples. Authoring only — HEATMAP_HONESTY / N6."""
    from meshops.proportion.depth_heatmap import run_depth_heatmap

    # Preserve trailing directory separator intent (R1); Path resolve strips it.
    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )
    return run_depth_heatmap(
        _resolve_tool_path(samples, work_root),
        out_arg,
        deltas=_resolve_tool_path(deltas, work_root) if deltas else None,
        underlay=_resolve_tool_path(underlay, work_root) if underlay else None,
        force=force,
    )


def mesh_proportion_depth_hint(
    work_root: Path,
    *,
    depth_map: str | None = None,
    left: str | None = None,
    out: str = "landmarks_assist.hint.json",
    assist: str | None = None,
    report: str | None = None,
    backend: str = "external",
    force: bool = False,
    force_hint: bool = False,
    merge_into: str | None = None,
) -> dict[str, Any]:
    """Depth-channel assist hints. Authoring only — HINT_HONESTY / N6; monocular refuses."""
    from meshops.proportion.depth_hint import run_depth_hint

    # Preserve trailing directory separator intent (R1); Path resolve strips it.
    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )
    return run_depth_hint(
        _resolve_tool_path(depth_map, work_root) if depth_map else None,
        _resolve_tool_path(left, work_root) if left else None,
        out_arg,
        assist=_resolve_tool_path(assist, work_root) if assist else None,
        report=_resolve_tool_path(report, work_root) if report else None,
        backend=backend,
        force=force,
        force_hint=force_hint,
        merge_into=(_resolve_tool_path(merge_into, work_root) if merge_into else None),
    )


def mesh_proportion_silhouette_compare(
    work_root: Path,
    *,
    ref: str,
    out: str,
    mesh: str | None = None,
    mesh_view: str | None = None,
    view_role: str = "front",
    overlay: bool = True,
    force: bool = False,
    require_trusted: bool = False,
) -> dict[str, Any]:
    """Same-role silhouette IoU/Dice (front|left). Authoring only — SILHOUETTE_HONESTY / N6.

    Payload includes silhouette_trusted, trust_reasons, coverage fracs, mask methods
    (schema 1.2.0). require_trusted raises silhouette_untrusted when untrusted.
    """
    from meshops.proportion.silhouette import run_silhouette_compare

    # Preserve trailing directory separator intent (R1); Path resolve strips it.
    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )
    return run_silhouette_compare(
        _resolve_tool_path(ref, work_root),
        out_arg,
        mesh=_resolve_tool_path(mesh, work_root) if mesh else None,
        mesh_view=_resolve_tool_path(mesh_view, work_root) if mesh_view else None,
        view_role=view_role,
        overlay=overlay,
        force=force,
        require_trusted=require_trusted,
    )


def mesh_proportion_blockout_feedback(
    work_root: Path,
    *,
    report: str,
    out: str,
    mesh: str | None = None,
    ref_front: str | None = None,
    ref_left: str | None = None,
    mesh_view_front: str | None = None,
    mesh_view_left: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Sticky post-export checklist. Authoring only — FEEDBACK_HONESTY / N6."""
    from meshops.proportion.blockout_feedback import run_blockout_feedback

    ends_sep = out.endswith(("/", "\\"))
    out_base = out.rstrip("/\\") if ends_sep else out
    out_resolved = _resolve_tool_path(out_base, work_root)
    out_arg: str | Path = (
        str(out_resolved) + ("\\" if ends_sep else "") if ends_sep else out_resolved
    )
    return run_blockout_feedback(
        _resolve_tool_path(report, work_root),
        out_arg,
        mesh=_resolve_tool_path(mesh, work_root) if mesh else None,
        ref_front=_resolve_tool_path(ref_front, work_root) if ref_front else None,
        ref_left=_resolve_tool_path(ref_left, work_root) if ref_left else None,
        mesh_view_front=(
            _resolve_tool_path(mesh_view_front, work_root) if mesh_view_front else None
        ),
        mesh_view_left=(_resolve_tool_path(mesh_view_left, work_root) if mesh_view_left else None),
        force=force,
    )
