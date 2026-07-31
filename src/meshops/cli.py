"""Typer CLI — ingest / triage / render / report / repair / export / diff /
accept / design / escalate.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import typer

from meshops import __version__

app = typer.Typer(
    name="meshops",
    help=(
        "MeshOps — triage + guarded T1/T2 repair + T7 design + T3 escalate "
        "(ingest / triage / render / report / repair / export / diff / accept / design / escalate)."
    ),
    add_completion=False,
    no_args_is_help=True,
)

design_app = typer.Typer(
    name="design",
    help="T7 mechanical design-from-code (build123d harness + absolute validate).",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(design_app, name="design")

escalate_app = typer.Typer(
    name="escalate",
    help=(
        "T3 escalation: ROI + preview + Blender 5.2 LTS handoff + import-sculpt. "
        "Set MESHOPS_BLENDER or install 5.2 LTS (track 0010 for mirror/doctor)."
    ),
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(escalate_app, name="escalate")


def _emit_json(payload: dict[str, Any]) -> None:
    """Pure JSON on stdout (no progress noise)."""
    sys.stdout.write(json.dumps(payload, indent=2, default=str))
    sys.stdout.write("\n")


def _emit_error(exc: BaseException, *, json_mode: bool, code: int = 1) -> NoReturn:
    payload: dict[str, Any] = {
        "ok": False,
        "error": type(exc).__name__,
        "message": str(exc),
    }
    # Attach structured fields when present
    for attr in ("code", "rev_id", "rev_dir"):
        if hasattr(exc, attr):
            val = getattr(exc, attr)
            if val is not None:
                payload[attr] = str(val) if not isinstance(val, (str, int, bool)) else val
    if hasattr(exc, "result") and exc.result is not None:  # type: ignore[attr-defined]
        with contextlib.suppress(Exception):
            payload["result"] = exc.result.model_dump(mode="json")  # type: ignore[attr-defined]
    if hasattr(exc, "guard") and exc.guard is not None:  # type: ignore[attr-defined]
        with contextlib.suppress(Exception):
            payload["guard"] = exc.guard.model_dump(mode="json")  # type: ignore[attr-defined]
    if json_mode:
        _emit_json(payload)
    else:
        typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code) from exc


@app.callback()
def main() -> None:
    """MeshOps CLI."""


@app.command("version")
def version_cmd(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Print package version."""
    if json_out:
        _emit_json({"ok": True, "version": __version__})
    else:
        typer.echo(__version__)


@app.command("ingest")
def ingest_cmd(
    path: Path = typer.Option(..., "--path", help="Path to source STL (never overwritten)"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Ingest STL into work/<mesh_id>/ (non-destructive)."""
    from meshops.ingest.pipeline import ingest_stl

    try:
        result = ingest_stl(path, work_root=work_root)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
        "ok": True,
        "mesh_id": result.mesh_id,
        "job_dir": str(result.job_dir),
        "original": str(result.original_path),
        "working": str(result.working_path),
        "proxy": str(result.proxy_path) if result.proxy_path else None,
        "reused": result.reused,
        "stats": result.stats.model_dump(mode="json"),
    }
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(f"mesh_id={result.mesh_id} job_dir={result.job_dir}")


@app.command("triage")
def triage_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Classify-only triage → diagnostics.json."""
    from meshops.triage.orchestrate import JobNotFoundError, mesh_triage

    try:
        diag = mesh_triage(mesh_id, work_root=work_root)
    except JobNotFoundError as exc:
        _emit_error(exc, json_mode=json_out, code=2)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    if json_out:
        _emit_json({"ok": True, "diagnostics": diag.model_dump(mode="json")})
    else:
        typer.echo(
            f"mesh_id={diag.mesh_id} sheet_score={diag.sheet_score.score:.3f} "
            f"needs_user_input={diag.needs_user_input}"
        )


@app.command("render")
def render_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """F3D offscreen RGB + visual depth views."""
    from meshops.render.f3d_renderer import F3DRenderer, RenderUnavailableError

    try:
        result = F3DRenderer().render_job(mesh_id, work_root=work_root)
    except RenderUnavailableError as exc:
        _emit_error(exc, json_mode=json_out, code=3)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
        "ok": True,
        "mesh_id": result.mesh_id,
        "rendered_from": result.rendered_from,
        "view_paths": result.view_paths,
        "depth_paths": result.depth_paths,
        "cameras": result.cameras,
        "depth_semantics": "visual_colormap_not_metric",
    }
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(f"rendered {len(result.view_paths)} views from {result.rendered_from}")


@app.command("report")
def report_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Generate report.md from diagnostics + views."""
    from meshops.report.generate import generate_report

    try:
        report_path = generate_report(mesh_id, work_root=work_root)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    if json_out:
        _emit_json({"ok": True, "mesh_id": mesh_id, "report_path": str(report_path)})
    else:
        typer.echo(str(report_path))


@app.command("repair")
def repair_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    recipe: str = typer.Option(
        ...,
        "--recipe",
        help="Allowlisted recipe: t1_clean | t2_smooth_spikes | t2_close_small_holes",
    ),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    parent_rev: str | None = typer.Option(
        None, "--parent-rev", help="Parent revision id (default: original.stl)"
    ),
    no_diff: bool = typer.Option(
        False,
        "--no-diff",
        help="Skip F3D; write stub PNG view_paths (still required for success)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Run allowlisted T1/T2 recipe → atomic rev + guards (+ optional diff)."""
    from meshops.recipes.orchestrate import RepairError, RepairRefuseError, run_repair

    try:
        result = run_repair(
            mesh_id,
            recipe,
            work_root=work_root,
            parent_rev=parent_rev,
            no_diff=no_diff,
        )
    except RepairRefuseError as exc:
        _emit_error(exc, json_mode=json_out, code=2)
    except RepairError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload: dict[str, Any] = {
        "ok": True,
        "mesh_id": mesh_id,
        "recipe_id": result.recipe_id,
        "rev_id": result.rev_id,
        "rev_dir": result.rev_dir,
        "manifest": result.manifest.model_dump(mode="json") if result.manifest else None,
        "notes": result.notes,
        "acceptance": (
            result.acceptance.model_dump(mode="json") if result.acceptance is not None else None
        ),
    }
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(f"repair ok rev_id={result.rev_id} rev_dir={result.rev_dir}")


@app.command("export")
def export_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    out: Path = typer.Option(..., "--out", help="Destination STL path"),
    rev: str | None = typer.Option(None, "--rev", help="Revision id (default: original.stl)"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Guarded export of original or rev; fail-closed on wipeout."""
    from meshops.export_guarded import ExportError, guarded_export

    try:
        payload = guarded_export(mesh_id, out, work_root=work_root, rev=rev)
    except ExportError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    if json_out:
        _emit_json(dict(payload))
    else:
        typer.echo(f"exported {payload['out']}")


@app.command("accept")
def accept_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    rev: str | None = typer.Option(
        None,
        "--rev",
        help="Revision id to accept (default: original.stl self-accept)",
    ),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    require_views: bool = typer.Option(
        True,
        "--require-views/--no-require-views",
        help="Require non-empty on-disk view paths (default: true)",
    ),
    allow_stubs: bool = typer.Option(
        True,
        "--allow-stubs/--no-allow-stubs",
        help="Allow stub view_kind (default: true; CI reality)",
    ),
    require_slice: bool = typer.Option(
        False,
        "--require-slice",
        help="Fail if no slice hook configured (default: false)",
    ),
    promote: bool = typer.Option(
        False,
        "--promote-working",
        help="On ok, promote rev mesh to working.ply + working_manifest.json",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Run acceptance pack on a rev (or original self-check). Exit 0 only if ok.

    Without --rev: original self-check is export-style (require_views=False,
    honesty=guards_only) — original has no mutator view_paths. With --rev:
    require_views defaults True (mutator honesty).
    """
    from meshops.acceptance import accept_candidate, accept_revision, promote_working
    from meshops.acceptance.promote import PromoteError
    from meshops.export_guarded import _baseline_stats
    from meshops.jobstore.paths import JobPaths

    try:
        if rev is not None:
            result = accept_revision(
                mesh_id,
                rev,
                work_root=work_root,
                require_views=require_views,
                allow_stubs=allow_stubs,
                require_slice=require_slice,
            )
        else:
            # Original self-check: always guards_only (no mutator views on original).
            paths = JobPaths(work_root=work_root, mesh_id=mesh_id)
            if not paths.job_dir.is_dir():
                raise FileNotFoundError(f"job not found: {paths.job_dir}")
            baseline = _baseline_stats(paths)
            if not paths.original_stl.is_file():
                raise FileNotFoundError(f"original.stl missing: {paths.original_stl}")
            result = accept_candidate(
                baseline,
                paths.original_stl,
                require_views=False,
                allow_stubs=allow_stubs,
                require_slice=require_slice,
            )

        promote_info: dict[str, Any] | None = None
        if promote:
            if rev is None:
                raise PromoteError(
                    "--promote-working requires --rev",
                    code="promote_needs_rev",
                    acceptance=result,
                )
            if result.ok:
                promo = promote_working(
                    mesh_id,
                    rev,
                    work_root=work_root,
                    acceptance=result,
                )
                promote_info = {
                    "working_ply": promo["working_ply"],
                    "working_manifest": promo["working_manifest"],
                    "content_sha256": promo["content_sha256"],
                }
                if hasattr(promo.get("acceptance"), "model_dump"):
                    result = promo["acceptance"]  # type: ignore[assignment]
            # if not ok, skip promote; still report fail exit

    except PromoteError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload: dict[str, Any] = {
        "ok": result.ok,
        "mesh_id": mesh_id,
        "rev": rev,
        "acceptance": result.model_dump(mode="json"),
    }
    if promote_info is not None:
        payload["promote"] = promote_info

    if json_out:
        _emit_json(payload)
    else:
        status = "ok" if result.ok else "FAIL"
        typer.echo(
            f"accept {status} honesty={result.honesty} failed={result.failed}",
            err=not result.ok,
        )

    if not result.ok:
        raise typer.Exit(1)


@design_app.command("run")
def design_run_cmd(
    source: Path = typer.Option(..., "--source", help="Path to build123d geometry source (.py)"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    timeout: float = typer.Option(60.0, "--timeout", help="Runner timeout seconds"),
    no_diff: bool = typer.Option(
        False,
        "--no-diff",
        help="Skip F3D; write stub PNG view_paths (still required for success)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Run geometry source through harness → design job + accept_candidate."""
    try:
        from meshops.design import run_design_pipeline
    except ImportError as exc:
        _emit_error(
            RuntimeError(f"design package import failed (install meshops[design]): {exc}"),
            json_mode=json_out,
            code=1,
        )

    try:
        result = run_design_pipeline(
            source,
            work_root=work_root,
            timeout_s=timeout,
            no_diff=no_diff,
        )
    except Exception as exc:
        _emit_error(exc, json_mode=json_out, code=1)

    payload: dict[str, Any] = {
        "ok": result.ok,
        "mesh_id": result.mesh_id,
        "job_dir": str(result.job_dir),
        "paths": result.paths,
        "manifest": result.manifest.model_dump(mode="json"),
        "notes": result.notes,
        "acceptance": (
            result.acceptance.model_dump(mode="json") if result.acceptance is not None else None
        ),
        "slice": "skipped",
        "honesty_note": "absolute validate is primary gate; self-baseline safety net only",
    }
    if json_out:
        _emit_json(payload)
    else:
        status = "ok" if result.ok else "FAIL"
        typer.echo(
            f"design run {status} mesh_id={result.mesh_id} job_dir={result.job_dir}",
            err=not result.ok,
        )
    if not result.ok:
        raise typer.Exit(1)


@design_app.command("from-spec")
def design_from_spec_cmd(
    template: str = typer.Option(
        "bracket_m4",
        "--template",
        help="Template id (default: bracket_m4)",
    ),
    hole_spacing: float = typer.Option(40.0, "--hole-spacing", help="Hole center spacing mm"),
    wall: float = typer.Option(3.0, "--wall", help="Wall thickness mm"),
    thickness: float = typer.Option(4.0, "--thickness", help="Plate thickness mm"),
    hole_diameter: float = typer.Option(4.2, "--hole-diameter", help="Hole diameter mm"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    timeout: float = typer.Option(60.0, "--timeout", help="Runner timeout seconds"),
    no_diff: bool = typer.Option(
        False,
        "--no-diff",
        help="Skip F3D; write stub PNG view_paths (still required for success)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Parametric template → design job + accept_candidate (unattended)."""
    try:
        from meshops.design import BracketParams, design_from_template
    except ImportError as exc:
        _emit_error(
            RuntimeError(f"design package import failed (install meshops[design]): {exc}"),
            json_mode=json_out,
            code=1,
        )

    try:
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
    except Exception as exc:
        _emit_error(exc, json_mode=json_out, code=1)

    payload = {
        "ok": result.ok,
        "mesh_id": result.mesh_id,
        "job_dir": str(result.job_dir),
        "template": template,
        "params": params.model_dump(mode="json"),
        "paths": result.paths,
        "manifest": result.manifest.model_dump(mode="json"),
        "notes": result.notes,
        "acceptance": (
            result.acceptance.model_dump(mode="json") if result.acceptance is not None else None
        ),
        "slice": "skipped",
        "honesty_note": "absolute validate is primary gate; self-baseline safety net only",
    }
    if json_out:
        _emit_json(payload)
    else:
        status = "ok" if result.ok else "FAIL"
        typer.echo(
            f"design from-spec {status} template={template} mesh_id={result.mesh_id}",
            err=not result.ok,
        )
    if not result.ok:
        raise typer.Exit(1)


@escalate_app.command("roi")
def escalate_roi_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    bbox: str | None = typer.Option(
        None,
        "--bbox",
        help="AABB as xmin,ymin,zmin,xmax,ymax,zmax (world coords)",
    ),
    from_sheet_heuristic: bool = typer.Option(
        False,
        "--from-sheet-heuristic",
        help="Suggest ROI from mesh/diagnostics heuristic (never sole path)",
    ),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Create ROI package under rois/<roi_id>/ (manual bbox preferred)."""
    from meshops.escalate.errors import EscalateError
    from meshops.escalate.roi import create_roi_bbox, create_roi_from_sheet_heuristic

    try:
        if from_sheet_heuristic and bbox is None:
            manifest = create_roi_from_sheet_heuristic(mesh_id, work_root=work_root)
        elif bbox is not None:
            parts = [p.strip() for p in bbox.split(",")]
            if len(parts) != 6:
                raise EscalateError(
                    "--bbox must be xmin,ymin,zmin,xmax,ymax,zmax (6 floats)",
                    code="invalid_bbox",
                )
            vals = [float(p) for p in parts]
            manifest = create_roi_bbox(
                mesh_id,
                vals[0:3],
                vals[3:6],
                work_root=work_root,
                source="manual",
            )
        else:
            raise EscalateError(
                "provide --bbox xmin,ymin,zmin,xmax,ymax,zmax or --from-sheet-heuristic",
                code="invalid_bbox",
            )
    except EscalateError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
        "ok": True,
        "mesh_id": mesh_id,
        "roi": manifest.model_dump(mode="json"),
    }
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(f"roi ok roi_id={manifest.roi_id} source={manifest.source}")


@escalate_app.command("preview-t3")
def escalate_preview_t3_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    roi: str | None = typer.Option(None, "--roi", help="ROI id from escalate roi"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Honest T3 preview package (never fixed / never auto-promote)."""
    from meshops.escalate.errors import EscalateError
    from meshops.escalate.preview_t3 import preview_t3

    try:
        result = preview_t3(mesh_id, roi, work_root=work_root)
    except EscalateError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
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
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(
            f"preview-t3 preview_only preview_id={result.preview_id} "
            f"(NOT fixed — handoff recommended)",
            err=False,
        )
    # Exit 0 for successful package write even though ok=False (preview semantics)
    # Callers must not treat as fixed.


@escalate_app.command("handoff")
def escalate_handoff_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    roi: str = typer.Option(..., "--roi", help="ROI id from escalate roi"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    timeout: float = typer.Option(300.0, "--timeout", help="Blender timeout seconds"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Build Blender 5.2 LTS handoff .blend + instructions (MESHOPS_BLENDER / 0010)."""
    from meshops.escalate.errors import EscalateError
    from meshops.escalate.handoff import build_handoff

    try:
        manifest = build_handoff(
            mesh_id,
            roi,
            work_root=work_root,
            timeout_s=timeout,
        )
    except EscalateError as exc:
        code = 2 if exc.code == "blender_missing" else 1
        _emit_error(exc, json_mode=json_out, code=code)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
        "ok": True,
        "mesh_id": mesh_id,
        "handoff": manifest.model_dump(mode="json"),
        "honesty_note": "handoff package ready — not autonomous hero fixed (N6)",
    }
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(f"handoff ok blend={manifest.blend_path} blender={manifest.blender_version}")


@escalate_app.command("import-sculpt")
def escalate_import_sculpt_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    path: Path = typer.Option(..., "--path", help="Path to sculpted STL from Blender"),
    approve: bool = typer.Option(
        False,
        "--approve",
        help="Required: acknowledge human/agent sculpt responsibility",
    ),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    no_diff: bool = typer.Option(
        False,
        "--no-diff",
        help="Skip F3D; write stub PNG view_paths (still required for success)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Import sculpt STL as rev + sculpt-tier accept (requires --approve)."""
    from meshops.escalate.errors import EscalateError
    from meshops.escalate.import_sculpt import import_sculpt

    try:
        result = import_sculpt(
            mesh_id,
            path,
            approve=approve,
            work_root=work_root,
            no_diff=no_diff,
        )
    except EscalateError as exc:
        code = 2 if exc.code == "approve_required" else 1
        _emit_error(exc, json_mode=json_out, code=code)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload: dict[str, Any] = {
        "ok": result.ok,
        "mesh_id": result.mesh_id,
        "rev_id": result.rev_id,
        "rev_dir": result.rev_dir,
        "recipe_id": result.recipe_id,
        "notes": result.notes,
        "honesty_note": result.honesty_note,
        "paths": result.paths,
        "acceptance": (
            result.acceptance.model_dump(mode="json") if result.acceptance is not None else None
        ),
        "promoted_to_working": False,
    }
    if json_out:
        _emit_json(payload)
    else:
        status = "ok" if result.ok else "FAIL"
        typer.echo(
            f"import-sculpt {status} rev_id={result.rev_id} (not auto-promoted to working)",
            err=not result.ok,
        )
    if not result.ok:
        raise typer.Exit(1)


@app.command("diff")
def diff_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    rev: str = typer.Option(..., "--rev", help="Revision id to compare"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Compare views: parent/original baseline vs rev (never working.ply)."""
    from meshops.jobstore.paths import JobPaths
    from meshops.recipes.diff_views import render_rev_diff
    from meshops.render.f3d_renderer import RenderUnavailableError
    from meshops.revs.store import load_manifest, parent_mesh_path, resolve_rev_dir

    try:
        payload = render_rev_diff(mesh_id, rev, work_root=work_root)
    except RenderUnavailableError as exc:
        # Still report baseline pin for unit assertions when F3D unavailable
        paths = JobPaths(work_root=work_root, mesh_id=mesh_id)
        try:
            rev_dir = resolve_rev_dir(paths, rev)
            man = load_manifest(rev_dir)
            baseline = parent_mesh_path(paths, man.parent_rev)
            partial = {
                "ok": False,
                "error": "RenderUnavailableError",
                "message": str(exc),
                "mesh_id": mesh_id,
                "rev_id": rev,
                "baseline": str(baseline),
                "baseline_is_working_ply": baseline.name == "working.ply",
            }
            if json_out:
                _emit_json(partial)
            else:
                typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(3) from exc
        except typer.Exit:
            raise
        except Exception:
            _emit_error(exc, json_mode=json_out, code=3)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    if json_out:
        _emit_json(payload)
    else:
        typer.echo(
            f"diff ok views={len(payload.get('view_paths', []))} baseline={payload.get('baseline')}"
        )


if __name__ == "__main__":
    app()
