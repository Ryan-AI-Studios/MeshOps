"""Typer CLI — ingest / triage / render / report with --json on each verb."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import typer

from meshops import __version__

app = typer.Typer(
    name="meshops",
    help="MeshOps — classify-only triage core (ingest / triage / render / report).",
    add_completion=False,
    no_args_is_help=True,
)


def _emit_json(payload: dict[str, Any]) -> None:
    """Pure JSON on stdout (no progress noise)."""
    sys.stdout.write(json.dumps(payload, indent=2, default=str))
    sys.stdout.write("\n")


def _emit_error(exc: BaseException, *, json_mode: bool, code: int = 1) -> NoReturn:
    if json_mode:
        _emit_json({"ok": False, "error": type(exc).__name__, "message": str(exc)})
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


if __name__ == "__main__":
    app()
