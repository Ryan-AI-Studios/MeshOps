"""Typer CLI — ingest / triage / render / report / repair / export / diff /
accept / design / escalate / organic / hosted / slice / doctor / bench /
proportion.
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
        "MeshOps — triage + guarded T1/T2 repair + T7 design + T3 escalate + T6 organic + "
        "hosted multi-view fallback + Orca slice + doctor + bench + proportion "
        "(ingest / triage / render / report / repair / export / diff / accept / design / "
        "escalate / organic / hosted / slice / doctor / bench / proportion)."
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

organic_app = typer.Typer(
    name="organic",
    help=(
        "T6 organic agent-first: SessionPaths session + Blender metaball recipes + "
        "F3D pass evidence + plateau + untrusted finalize. "
        "MESHOPS_ORGANIC_TIMEOUT_S default 300 (use 600+ for high-res). "
        "Hosted multi-view fallback is `meshops hosted` after plateau gate."
    ),
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(organic_app, name="organic")

hosted_app = typer.Typer(
    name="hosted",
    help=(
        "Hosted multi-view image-to-3D fallback (post-plateau only). "
        "Requires plateau.json with allows_hosted_fallback=true. Never default organic path. "
        "Env: MESHOPS_HOSTED_API_KEY (or MESHOPS_MESHY_API_KEY); mock provider for offline."
    ),
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(hosted_app, name="hosted")

bench_app = typer.Typer(
    name="bench",
    help=(
        "Read-only size-ladder benchmarks (ingest / triage / optional F3D). "
        "No --approve / GuardPolicy. Writes work/bench/bench_results.json. "
        "Env: MESHOPS_BENCH_SIZES, MESHOPS_BENCH_WORK_ROOT; soak tests need MESHOPS_BENCH_SOAK=1."
    ),
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(bench_app, name="bench")

proportion_app = typer.Typer(
    name="proportion",
    help=(
        "Pixel proportion analysis from multi-view RGB (tracks 0012-0014). "
        "Verbs: template | analyze | show | scaffold. "
        "Assist-first landmarks + head-unit checks + blockout-grade XYZ; "
        "schema 1.1.0 diameters (edge pairs) + left depth bands + cross-sections; "
        "scaffold creates package layout + package_checklist.json only (not mesh/print success). "
        "Optional: meshops[proportion] (Pillow)."
    ),
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(proportion_app, name="proportion")


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


@app.command("mcp")
def mcp_cmd() -> None:
    """Launch MeshOps MCP stdio server (requires meshops[mcp] / mcp==2.0.0).

    Prefer console script ``meshops-mcp`` or ``python -m meshops.mcp``.
    Host must set process cwd to the repo; work_root from MESHOPS_WORK or ./work.
    """
    from meshops.mcp.__main__ import main as mcp_main

    mcp_main()


@app.command("version")
def version_cmd(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Print package version."""
    if json_out:
        _emit_json({"ok": True, "version": __version__})
    else:
        typer.echo(__version__)


@app.command("doctor")
def doctor_cmd(
    json_out: bool = typer.Option(False, "--json", help="Emit DoctorReport JSON on stdout"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Require core + Blender 5.2 + Orca path (print/organic-ready box)",
    ),
    require: list[str] | None = typer.Option(
        None,
        "--require",
        help=("Repeatable required check: core|blender|orca|f3d|design|all (default: core only)"),
    ),
) -> None:
    """Diagnose Python env, core packages, Blender, Orca, F3D (ops health).

    Default exit 0 when core is healthy even if Blender/Orca are missing.
    Cold-start may take multiple seconds (native imports: pymeshlab, f3d, …).
    """
    from meshops.ops.doctor import expand_require, run_doctor

    tokens: list[str] = list(require or [])
    if strict:
        tokens.extend(["core", "blender", "orca"])
    try:
        req = expand_require(tokens if tokens else None)
    except ValueError as exc:
        _emit_error(exc, json_mode=json_out, code=2)

    report = run_doctor(require=req, cwd=Path.cwd())
    payload = report.model_dump(mode="json")

    if json_out:
        _emit_json(payload)
    else:
        status = "OK" if report.ok else "FAIL"
        typer.echo(f"meshops doctor [{status}]  required={','.join(report.required)}")
        typer.echo(
            f"  python {report.python.version}  pin_ok={report.python.pin_ok}  "
            f"exe={report.python.executable}"
        )
        typer.echo("  packages:")
        for name, pkg in sorted(report.packages.items()):
            flag = "ok" if pkg.import_ok else "MISSING"
            opt = " (optional)" if pkg.optional else ""
            ver = pkg.version or "?"
            typer.echo(f"    {name}: {flag} {ver}{opt}")
        b = report.tools.blender
        typer.echo(
            f"  blender: status={b.status} pin_ok={b.pin_ok} source={b.source} "
            f"version={b.version or '-'} path={b.path or '-'}"
        )
        o = report.tools.orca
        typer.echo(
            f"  orca: status={o.status} source={o.source} "
            f"version_source={o.version_source} version={o.version or '-'} "
            f"path={o.path or '-'}"
        )
        f3 = report.tools.f3d
        typer.echo(f"  f3d: import_ok={f3.import_ok} version={f3.version or '-'}")
        uv = report.tooling
        typer.echo(f"  uv: version={uv.version or '-'} uv_lock_present={uv.uv_lock_present}")
        disk = report.disk
        mb = disk.pymeshlab_approx_mb
        mb_s = f"{mb} MB" if mb is not None else "unknown"
        typer.echo(f"  disk: pymeshlab ~ {mb_s} ({disk.note})")
        typer.echo("  licenses:")
        for line in report.licenses:
            typer.echo(f"    - {line}")
        n = report.vram.nvidia
        typer.echo(
            f"  vram: nvidia status={n.status} free_mib={n.free_mib} total_mib={n.total_mib}"
        )
        typer.echo(f"         {report.vram.ritual}")
        if report.hints:
            typer.echo("  hints:")
            for h in report.hints:
                typer.echo(f"    - {h}")
        if report.notes:
            typer.echo("  notes:")
            for note in report.notes:
                typer.echo(f"    - {note}")

    raise typer.Exit(0 if report.ok else 1)


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
        help="Require Orca printability oracle (fail closed if Orca missing)",
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

    With --require-slice: attach live Orca hook when binary is present at invoke
    time; if missing, fail closed (slice_not_configured / orca_not_found) — never
    skip-as-pass.
    """
    from meshops.acceptance import accept_candidate, accept_revision, promote_working
    from meshops.acceptance.promote import PromoteError
    from meshops.export_guarded import _baseline_stats
    from meshops.jobstore.paths import JobPaths

    slice_hook = None
    if require_slice:
        from meshops.slice import find_orca, make_orca_hook

        # Fail closed at attach if Orca absent — pack still gets require_slice=True
        # with no hook → slice_not_configured. Prefer live hook when present;
        # hook re-checks find_orca on every call.
        if find_orca(require=False) is not None:
            slice_hook = make_orca_hook(mesh_id=mesh_id, work_root=work_root)

    try:
        if rev is not None:
            result = accept_revision(
                mesh_id,
                rev,
                work_root=work_root,
                require_views=require_views,
                allow_stubs=allow_stubs,
                require_slice=require_slice,
                slice_hook=slice_hook,
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
                slice_hook=slice_hook,
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


@app.command("slice")
def slice_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id from ingest"),
    rev: str | None = typer.Option(
        None,
        "--rev",
        help="Revision id mesh to slice (default: working.ply if present, else original.stl)",
    ),
    profile: str = typer.Option(
        "default",
        "--profile",
        help="Slice profile name or absolute dir with machine/process/filament.json",
    ),
    orient: bool = typer.Option(
        False,
        "--orient/--no-orient",
        help="Pass --orient 1 to Orca (default: false → 0)",
    ),
    arrange: bool = typer.Option(
        False,
        "--arrange/--no-arrange",
        help="Pass --arrange 1 to Orca (default: false → 0 for determinism)",
    ),
    allow_reorient_retry: bool = typer.Option(
        False,
        "--allow-reorient-retry",
        help="On filament_anomaly_high, retry once with --orient 1",
    ),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Run OrcaSlicer printability oracle on a job candidate. Exit 0 only on pass.

    Candidate priority: --rev mesh → working.ply if present → original.stl.
    Always writes slice_report.md under work/<id>/slice/<run_id>/ (even on fail).
    """
    from meshops.jobstore.paths import JobPaths
    from meshops.revs.store import resolve_rev_dir
    from meshops.slice import find_orca, run_slice
    from meshops.slice.errors import SliceError

    try:
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

        if rev is not None:
            rev_dir = resolve_rev_dir(paths, rev)
            cand: Path | None = None
            for name in ("mesh.stl", "mesh.ply", "result.stl"):
                p = rev_dir / name
                if p.is_file():
                    cand = p
                    break
            if cand is None:
                # Fall back to any .stl/.ply in rev dir
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
    except SliceError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload: dict[str, Any] = {
        "ok": result.status == "pass"
        and result.accept is not None
        and result.accept.status == "pass",
        "mesh_id": mesh_id,
        "slice": result.model_dump(mode="json"),
    }
    if json_out:
        _emit_json(payload)
    else:
        status = "ok" if payload["ok"] else "FAIL"
        typer.echo(
            f"slice {status} run_id={result.run_id} "
            f"accept={result.accept.status if result.accept else None} "
            f"error={result.error_code}",
            err=not payload["ok"],
        )
    if not payload["ok"]:
        raise typer.Exit(1)


# --- organic (T6 / track 0006) -------------------------------------------------


@organic_app.command("create")
def organic_create_cmd(
    prompt: str = typer.Option(..., "--prompt", help="Authoring prompt (required, non-empty)"),
    style: str = typer.Option("", "--style", help="Style notes"),
    ref: list[Path] | None = typer.Option(
        None,
        "--ref",
        help="Reference image path (repeatable)",
    ),
    recipe: str = typer.Option("simple_bust", "--recipe", help="Default recipe id"),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Optional o + 11 hex session id",
    ),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Session store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Create organic authoring session under work/<session_id>/organic/."""
    from meshops.organic import OrganicError, create_session

    try:
        manifest = create_session(
            prompt,
            style_notes=style,
            refs=list(ref) if ref else None,
            default_recipe=recipe,
            session_id=session_id,
            work_root=work_root,
        )
    except OrganicError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
        "ok": True,
        "session_id": manifest.session_id,
        "status": manifest.status,
        "default_recipe": manifest.default_recipe,
        "notes": manifest.notes,
        "manifest": manifest.model_dump(mode="json"),
    }
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(f"organic create ok session_id={manifest.session_id}")


@organic_app.command("pass")
def organic_pass_cmd(
    session_id: str = typer.Option(..., "--session-id", help="Session id (o + 11 hex)"),
    recipe: str | None = typer.Option(None, "--recipe", help="Recipe override"),
    params: str | None = typer.Option(
        None,
        "--params",
        help="JSON object of recipe params, e.g. '{\"resolution\": 0.5}'",
    ),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Session store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Run one Blender organic recipe pass + required multi-view evidence."""
    from meshops.organic import OrganicError, run_pass

    raw_params: dict[str, Any] | None = None
    if params:
        try:
            loaded = json.loads(params)
        except json.JSONDecodeError as exc:
            _emit_error(
                OrganicError(f"invalid --params JSON: {exc}", code="invalid_params"),
                json_mode=json_out,
                code=1,
            )
            return  # pragma: no cover — _emit_error never returns
        if not isinstance(loaded, dict):
            _emit_error(
                OrganicError("--params must be a JSON object", code="invalid_params"),
                json_mode=json_out,
                code=1,
            )
            return  # pragma: no cover
        raw_params = loaded

    try:
        result = run_pass(
            session_id,
            recipe=recipe,
            params=raw_params,
            work_root=work_root,
        )
    except OrganicError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
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
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(
            f"organic pass ok pass_id={result.pass_id} views={len(result.view_paths)} "
            f"kind={result.view_kind}"
        )


@organic_app.command("status")
def organic_status_cmd(
    session_id: str = typer.Option(..., "--session-id", help="Session id (o + 11 hex)"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Session store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Show organic session manifest status."""
    from meshops.organic import OrganicError, load_session

    try:
        paths, manifest = load_session(session_id, work_root=work_root)
    except OrganicError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
        "ok": True,
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
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(
            f"organic status session_id={manifest.session_id} status={manifest.status} "
            f"passes={len(manifest.passes)}"
        )


@organic_app.command("plateau")
def organic_plateau_cmd(
    session_id: str = typer.Option(..., "--session-id", help="Session id (o + 11 hex)"),
    reason: str = typer.Option(
        ...,
        "--reason",
        help="Quality reason (≥15 chars; not filler) for 0007 hosted-fallback gate",
    ),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Session store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Mark session plateau (machine-readable criteria_met for 0007)."""
    from meshops.organic import OrganicError, mark_plateau

    try:
        record = mark_plateau(session_id, reason, work_root=work_root)
    except OrganicError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
        "ok": True,
        "session_id": record.session_id,
        "allows_hosted_fallback": record.allows_hosted_fallback,
        "criteria_met": record.criteria_met,
        "pass_count": record.pass_count,
        "reason": record.reason,
        "plateau": record.model_dump(mode="json"),
    }
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(
            f"organic plateau ok allows_hosted_fallback={record.allows_hosted_fallback} "
            f"criteria={record.criteria_met}"
        )


@organic_app.command("finalize")
def organic_finalize_cmd(
    session_id: str = typer.Option(..., "--session-id", help="Session id (o + 11 hex)"),
    accept: bool = typer.Option(
        False,
        "--accept/--no-accept",
        help="Run accept_candidate with GuardPolicy.for_sculpt() on job views",
    ),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Session store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Finalize: final.stl → ingest → triage → job views → optional sculpt accept (B12)."""
    from meshops.organic import OrganicError, finalize_session

    try:
        result = finalize_session(session_id, work_root=work_root, accept=accept)
    except OrganicError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
        "ok": result.ok,
        "session_id": result.session_id,
        "mesh_id": result.mesh_id,
        "job_dir": str(result.job_dir) if result.job_dir else None,
        "triage_summary": result.triage_summary,
        "honesty_message": result.honesty_message,
        "messages": result.messages,
        "error_code": result.error_code,
        "acceptance": (
            result.acceptance.model_dump(mode="json") if result.acceptance is not None else None
        ),
    }
    if json_out:
        _emit_json(payload)
    else:
        status = "ok" if result.ok else "FAIL"
        typer.echo(
            f"organic finalize {status} mesh_id={result.mesh_id} job_dir={result.job_dir}",
            err=not result.ok,
        )
    if not result.ok:
        raise typer.Exit(1)


# --- hosted (track 0007 multi-view fallback) ----------------------------------


@hosted_app.command("providers")
def hosted_providers_cmd(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """List hosted provider adapters (no secrets)."""
    from meshops.hosted.providers import DEFAULT_PROVIDER_NAME, list_providers

    rows = list_providers()
    payload = {
        "ok": True,
        "default": DEFAULT_PROVIDER_NAME,
        "providers": rows,
    }
    if json_out:
        _emit_json(payload)
    else:
        for row in rows:
            mark = " (default)" if row.get("default") else ""
            offline = " offline" if row.get("offline") else ""
            typer.echo(f"{row['name']}{mark}{offline}")


@hosted_app.command("run")
def hosted_run_cmd(
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Organic session id (o + 11 hex); used with --work-root for plateau path",
    ),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Session/job store root"),
    plateau: Path | None = typer.Option(
        None,
        "--plateau",
        help="Absolute path to plateau.json (out-of-tree sessions OK)",
    ),
    views_from: str = typer.Option(
        "latest",
        "--views-from",
        help="View source: latest|pass|explicit (default latest successful pass)",
    ),
    view: list[Path] | None = typer.Option(
        None,
        "--view",
        help="Explicit view image path (repeatable; ≥2 required if used)",
    ),
    prompt: str = typer.Option("", "--prompt", help="Optional prompt (else session manifest)"),
    justify: str = typer.Option(
        ...,
        "--justify",
        help="Operator justification (≥15 chars; not filler; same rules as plateau reason)",
    ),
    provider: str = typer.Option(
        "meshy",
        "--provider",
        help="Provider adapter (default meshy; use mock for offline)",
    ),
    accept: bool = typer.Option(
        False,
        "--accept/--no-accept",
        help="Optional 0011 accept_candidate compose after triage",
    ),
    accept_policy: str = typer.Option(
        "export",
        "--accept-policy",
        help="When --accept: export|sculpt|design (default export; not hard-wired sculpt)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Run hosted multi-view fallback after a valid 0006 plateau gate."""
    from meshops.hosted import HostedError, run_hosted_fallback
    from meshops.hosted.views import ViewsFrom

    vf: ViewsFrom
    if views_from not in ("latest", "pass", "explicit"):
        _emit_error(
            HostedError(
                f"invalid --views-from: {views_from!r}",
                code="multiview_required",
            ),
            json_mode=json_out,
            code=1,
        )
        return  # pragma: no cover
    vf = views_from  # type: ignore[assignment]

    try:
        result = run_hosted_fallback(
            session_id=session_id,
            work_root=work_root,
            plateau=plateau,
            views_from=vf,
            view_paths=list(view) if view else None,
            prompt=prompt,
            justify=justify,
            provider=provider,
            accept=accept,
            accept_policy=accept_policy,
        )
    except HostedError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = result.model_dump(mode="json")
    if json_out:
        _emit_json(payload)
    else:
        status = "ok" if result.ok else "FAIL"
        typer.echo(
            f"hosted run {status} mesh_id={result.mesh_id} provider={result.provider} "
            f"task={result.provider_task_id}",
            err=not result.ok,
        )
    if not result.ok:
        raise typer.Exit(1)


@hosted_app.command("status")
def hosted_status_cmd(
    mesh_id: str = typer.Option(..., "--mesh-id", help="Job mesh_id with hosted/ artifacts"),
    work_root: Path = typer.Option(Path("work"), "--work-root", help="Job store root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Show last hosted run_manifest under work/<mesh_id>/hosted/ (if present)."""
    from meshops.jobstore.paths import JobPaths

    job = JobPaths(work_root=work_root, mesh_id=mesh_id)
    manifest = job.hosted_dir / "run_manifest.json"
    if not manifest.is_file():
        payload = {
            "ok": False,
            "mesh_id": mesh_id,
            "error": "hosted_manifest_missing",
            "path": str(manifest),
        }
        if json_out:
            _emit_json(payload)
        else:
            typer.echo(f"no hosted run_manifest at {manifest}", err=True)
        raise typer.Exit(1)

    data = json.loads(manifest.read_text(encoding="utf-8"))
    payload = {"ok": True, "mesh_id": mesh_id, "manifest": data, "path": str(manifest)}
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(
            f"hosted status mesh_id={mesh_id} provider={data.get('provider')} "
            f"task={data.get('provider_task_id')}"
        )


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


_BENCH_SIZES_DEFAULT = "S,M,L,XL"


@bench_app.command("run")
def bench_run_cmd(
    sizes: str = typer.Option(
        _BENCH_SIZES_DEFAULT,
        "--sizes",
        help=(
            "Comma-separated size labels (S,M,L,XL). "
            "Env MESHOPS_BENCH_SIZES overrides when this flag is left at default."
        ),
    ),
    work_root: Path | None = typer.Option(
        None,
        "--work-root",
        help="Results/jobs root (default: MESHOPS_BENCH_WORK_ROOT or work/bench)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit Envelope JSON on stdout"),
    no_render: bool = typer.Option(
        False,
        "--no-render",
        help="Skip F3D timing (ingest+triage only)",
    ),
) -> None:
    """Run deterministic size ladder; write bench_results.json + .md under work root.

    Read-only: no mutators, no --approve, no GuardPolicy. Per-case failures continue
    the ladder. L/XL may skip with skipped_insufficient_ram when available RAM < ~4 GiB.
    """
    import os

    from meshops.bench.report import resolve_work_root, write_results
    from meshops.bench.runner import run_ladder
    from meshops.bench.sizes import parse_sizes

    # Prefer MESHOPS_BENCH_SIZES when --sizes left at typer default.
    size_spec = sizes
    if sizes.strip().upper() == _BENCH_SIZES_DEFAULT:
        env_sizes = os.environ.get("MESHOPS_BENCH_SIZES", "").strip()
        if env_sizes:
            size_spec = env_sizes

    try:
        label_list = parse_sizes(size_spec)
    except ValueError as exc:
        _emit_error(exc, json_mode=json_out, code=2)

    # Single resolved root for ladder cases + results JSON (env when flag omitted).
    resolved_root = resolve_work_root(work_root)

    try:
        envelope = run_ladder(
            label_list,
            work_root=resolved_root,
            include_render=not no_render,
        )
        json_path, md_path = write_results(envelope, work_root=resolved_root)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = envelope.model_dump(mode="json")
    payload["ok"] = all(c.status != "failed" for c in envelope.cases)
    payload["results_json"] = str(json_path)
    payload["results_md"] = str(md_path)
    payload["work_root"] = str(resolved_root)

    if json_out:
        _emit_json(payload)
    else:
        typer.echo(f"bench run wrote {json_path}")
        typer.echo(f"  markdown: {md_path}")
        for c in envelope.cases:
            extra = ""
            if c.skipped_reason:
                extra = f" ({c.skipped_reason})"
            elif c.error_message:
                extra = f" ({c.error_code}: {c.error_message[:80]})"
            typer.echo(
                f"  {c.label}: {c.status} faces={c.actual_faces}/{c.target_faces} "
                f"ingest={c.ingest_s} triage={c.triage_s} render={c.render_s}{extra}"
            )
    raise typer.Exit(0 if payload["ok"] else 1)


@bench_app.command("envelope")
def bench_envelope_cmd(
    work_root: Path | None = typer.Option(
        None,
        "--work-root",
        help=(
            "Search root for newest bench_results.json "
            "(default: MESHOPS_BENCH_WORK_ROOT or work/bench)"
        ),
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit Envelope JSON on stdout"),
) -> None:
    """Print latest bench envelope (newest bench_results.json by mtime)."""
    from meshops.bench.report import (
        default_results_dir,
        envelope_to_markdown,
        find_latest_results,
        load_envelope,
        resolve_work_root,
    )

    resolved = resolve_work_root(work_root)
    search = default_results_dir(resolved)
    path = find_latest_results(search)
    if path is None:
        msg = f"no bench_results.json found under {search} (run: meshops bench run)"
        if json_out:
            _emit_json({"ok": False, "error": "bench_results_missing", "message": msg})
        else:
            typer.echo(msg, err=True)
        raise typer.Exit(1)

    try:
        envelope = load_envelope(path)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    if json_out:
        payload = envelope.model_dump(mode="json")
        payload["ok"] = True
        payload["source_path"] = str(path)
        payload["work_root"] = str(resolved)
        _emit_json(payload)
    else:
        typer.echo(f"source: {path}")
        typer.echo(envelope_to_markdown(envelope))
    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# proportion (0012-0014) — template | analyze | show | scaffold  (no check verb)
# ---------------------------------------------------------------------------


@proportion_app.command("template")
def proportion_template_cmd(
    out: Path = typer.Option(
        Path("landmarks_assist.json"),
        "--out",
        help="Path for blank landmarks_assist.json",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
) -> None:
    """Emit blank assist JSON with all canonical view keys (null landmarks)."""
    from meshops.proportion.errors import ProportionError
    from meshops.proportion.template import blank_assist_document, write_template

    try:
        path = write_template(out)
        doc = blank_assist_document()
    except ProportionError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except OSError as exc:
        _emit_error(exc, json_mode=json_out, code=1)

    if json_out:
        _emit_json({"ok": True, "path": str(path), "assist": doc})
    else:
        typer.echo(f"wrote blank assist template: {path}")
    raise typer.Exit(0)


@proportion_app.command("analyze")
def proportion_analyze_cmd(
    views_dir: Path = typer.Option(
        ...,
        "--views-dir",
        help="Directory with front/left/three_quarter[.png|.jpg] (optional back)",
    ),
    landmarks: Path | None = typer.Option(
        None,
        "--landmarks",
        help="landmarks_assist.json (default: <views-dir>/landmarks_assist.json)",
    ),
    height_m: float | None = typer.Option(
        None,
        "--height-m",
        help="Optional stature in meters; scales landmarks_xyz *_m fields",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output directory for proportion_report.json (+ .md / overlays)",
    ),
    overlays: bool = typer.Option(
        False,
        "--overlays",
        help="Write landmark overlay PNGs (requires Pillow / meshops[proportion])",
    ),
    partial_ok: bool = typer.Option(
        False,
        "--partial-ok",
        help="Allow missing required views; sets partial_package + lower package_score",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit ProportionReport JSON on stdout"),
) -> None:
    """Analyze multi-view package → proportion_report (not mesh/print success)."""
    from meshops.proportion.analyze import analyze_proportion
    from meshops.proportion.errors import ProportionError

    try:
        report = analyze_proportion(
            views_dir,
            landmarks_path=landmarks,
            height_m=height_m,
            out_dir=out,
            partial_ok=partial_ok,
            overlays=overlays,
        )
    except ProportionError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = report.model_dump(mode="json")
    payload["ok"] = True
    if out is not None:
        payload["report_path"] = str(Path(out) / "proportion_report.json")

    if json_out:
        _emit_json(payload)
    else:
        typer.echo(
            f"proportion package_score={report.package_score:.1f} "
            f"pose={report.pose} honesty={report.honesty}"
        )
        if report.head_unit_frac is not None:
            typer.echo(
                f"  head_unit_frac={report.head_unit_frac:.4f} "
                f"(~{1.0 / report.head_unit_frac:.2f} heads)"
            )
        q = report.quality
        flags = [
            name
            for name, val in (
                ("hair_volume_margin", q.hair_volume_margin),
                ("foreshortening_risk", q.foreshortening_risk),
                ("multi_figure", q.multi_figure),
                ("needs_user_input", q.needs_user_input),
                ("incomplete_stature", q.incomplete_stature),
                ("partial_package", q.partial_package),
            )
            if val
        ]
        if flags:
            typer.echo(f"  quality: {', '.join(flags)}")
        for c in report.checks:
            if not c.ok or c.severity != "info":
                mark = "ok" if c.ok else "flag"
                typer.echo(f"  [{mark}] {c.name}: {c.message}")
        if out is not None:
            typer.echo(f"  wrote {Path(out) / 'proportion_report.json'}")
    raise typer.Exit(0)


@proportion_app.command("show")
def proportion_show_cmd(
    report: Path = typer.Option(
        ...,
        "--report",
        help="Path to proportion_report.json",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit report JSON on stdout"),
) -> None:
    """Re-display a saved proportion_report.json (checks included; no separate check verb)."""
    from meshops.proportion.analyze import load_report, report_to_markdown
    from meshops.proportion.errors import ProportionError

    try:
        rep = load_report(report)
    except ProportionError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    if json_out:
        payload = rep.model_dump(mode="json")
        payload["ok"] = True
        payload["source_path"] = str(report)
        _emit_json(payload)
    else:
        typer.echo(report_to_markdown(rep))
    raise typer.Exit(0)


@proportion_app.command("scaffold")
def proportion_scaffold_cmd(
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output directory for package layout (created if missing)",
    ),
    dual: bool = typer.Option(
        False,
        "--dual",
        help="Dual package: proportion/ + character/ under --out (Package A + B)",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Package mode: single | dual (conflict with --dual + single → error)",
    ),
    height_m: float | None = typer.Option(
        None,
        "--height-m",
        help="Optional stature meters stored in package_checklist.json",
    ),
    subject: str | None = typer.Option(
        None,
        "--subject",
        help="Subject label for checklist / SOURCE.txt",
    ),
    pose: str = typer.Option(
        "a_pose",
        "--pose",
        help="Pose kind default for checklist / optional assist template (default a_pose)",
    ),
    heroic_vs_realistic: str = typer.Option(
        "unknown",
        "--heroic-vs-realistic",
        help="heroic | realistic | stylized | unknown",
    ),
    figures: str | None = typer.Option(
        None,
        "--figures",
        help="Comma-separated in-scope figure labels (strip/drop empties)",
    ),
    source: str = typer.Option(
        "unknown",
        "--source",
        help="source_kind: imagen|chatgpt|photo|f3d|blender|other|unknown",
    ),
    wardrobe_tier: str | None = typer.Option(
        None,
        "--wardrobe-tier",
        help="two_piece_midriff|unitard|tank_leggings|costume|unknown",
    ),
    with_template: bool = typer.Option(
        False,
        "--with-template",
        help="Write landmarks_assist.json (pose post-processed from checklist)",
    ),
    stub_images: bool = typer.Option(
        False,
        "--stub-images",
        help="Write 1x1 PNG stubs for required views (off by default; layout only)",
    ),
    include_back_stub: bool = typer.Option(
        False,
        "--include-back-stub",
        help="Also write back.png stub when --stub-images",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing package_checklist.json (and stubs if --stub-images)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit scaffold result JSON"),
) -> None:
    """Create multi-view package layout + checklist (layout only — not mesh/print success)."""
    from meshops.proportion.checklist import parse_figures
    from meshops.proportion.errors import ProportionError
    from meshops.proportion.scaffold import scaffold_package

    # R3: --dual + --mode single is illegal
    if dual and mode is not None and mode.strip().lower() == "single":
        raise typer.BadParameter(
            "cannot combine --dual with --mode single (use --dual alone or --mode dual)"
        )
    if mode is not None and mode.strip().lower() not in ("single", "dual"):
        raise typer.BadParameter("--mode must be 'single' or 'dual'")

    resolved_mode = "dual" if dual else (mode.strip().lower() if mode else "single")
    figure_list = parse_figures(figures)

    try:
        result = scaffold_package(
            out,
            dual=dual or resolved_mode == "dual",
            mode=resolved_mode if not dual else "dual",  # type: ignore[arg-type]
            height_m=height_m,
            subject=subject,
            pose=pose,
            heroic_vs_realistic=heroic_vs_realistic,
            figures=figure_list,
            source_kind=source,  # type: ignore[arg-type]
            wardrobe_tier=wardrobe_tier,  # type: ignore[arg-type]
            with_template=with_template,
            stub_images=stub_images,
            include_back_stub=include_back_stub,
            force=force,
        )
    except ProportionError as exc:
        _emit_error(exc, json_mode=json_out, code=1)
    except Exception as exc:
        _emit_error(exc, json_mode=json_out)

    payload = {
        "ok": True,
        "mode": result.mode,
        "paths": [str(p) for p in result.paths],
        "analyze_hint": str(result.analyze_hint) if result.analyze_hint is not None else None,
    }
    if json_out:
        _emit_json(payload)
    else:
        typer.echo(
            f"scaffold mode={result.mode} paths={len(result.paths)} "
            "(layout only — not mesh or print success)"
        )
        if result.analyze_hint is not None:
            typer.echo(f"  analyze_hint: {result.analyze_hint}")
        for p in result.paths:
            typer.echo(f"  {p}")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
