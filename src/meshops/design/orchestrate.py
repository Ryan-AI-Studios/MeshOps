"""Full design pipeline: harness export → ingest → design/ → triage → views → accept."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from meshops.acceptance import accept_candidate
from meshops.acceptance.models import AcceptanceResult, ViewKind
from meshops.design.errors import DesignError
from meshops.design.models import BracketParams, DesignManifest, DesignResult
from meshops.design.runner import DEFAULT_TIMEOUT_S, run_geometry_source
from meshops.design.templates.registry import render_template
from meshops.design.validate import validate_design_mesh
from meshops.guards.policy import GuardPolicy
from meshops.ingest.pipeline import ingest_stl
from meshops.jobstore.paths import JobPaths, content_sha256, ensure_job_layout
from meshops.triage.orchestrate import mesh_triage

# Minimal valid 1x1 PNG for stub views (same as recipes/orchestrate).
_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
_STUB_CAMERAS = ("front", "three_quarter", "top")


def _write_stub_views(views_dir: Path) -> list[str]:
    views_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name in _STUB_CAMERAS:
        dest = views_dir / f"{name}.png"
        dest.write_bytes(_MIN_PNG)
        paths.append(str(dest))
    return paths


def _prefer_stub_views() -> bool:
    if stub := os.environ.get("MESHOPS_STUB_DIFF", "").strip().lower():
        return stub in {"1", "true", "yes", "on"}
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def _render_design_views(
    mesh_id: str,
    *,
    work_root: Path,
    no_diff: bool = False,
) -> tuple[list[str], ViewKind, list[str]]:
    """Produce on-disk views; honest stub when F3D unavailable / CI."""
    paths = JobPaths(work_root=work_root, mesh_id=mesh_id)
    notes: list[str] = []
    if no_diff or _prefer_stub_views():
        kind: ViewKind = "stub"
        notes.append(
            "design_stub_ci_or_no_diff" if no_diff else "design_stub_ci_or_MESHOPS_STUB_DIFF"
        )
        return _write_stub_views(paths.views_dir), kind, notes
    try:
        from meshops.render.f3d_renderer import F3DRenderer

        result = F3DRenderer().render_job(mesh_id, work_root=work_root, include_depth=False)
        if not result.view_paths:
            notes.append("design_views_empty_used_stub")
            return _write_stub_views(paths.views_dir), "stub", notes
        return list(result.view_paths), "f3d", notes
    except Exception as exc:
        notes.append(f"design_views_unavailable_used_stub: {type(exc).__name__}: {exc}")
        return _write_stub_views(paths.views_dir), "stub", notes


def _stage_and_ingest(
    *,
    source_text: str,
    work_root: Path,
    timeout_s: float,
    template_id: str | None,
    params: dict[str, Any],
    source_label: str | None,
    no_diff: bool,
) -> DesignResult:
    """Shared pipeline after geometry source text is ready."""
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    # 1) Stage export via AST + subprocess harness
    import tempfile

    with tempfile.TemporaryDirectory(prefix="meshops_design_stage_") as tmp:
        stage = Path(tmp)
        exported = run_geometry_source(source_text, staging_dir=stage, timeout_s=timeout_s)

        # 2) Ingest staging STL → original.stl canonical
        try:
            ing = ingest_stl(exported.stl_path, work_root=work_root)
        except Exception as exc:
            raise DesignError(
                f"ingest after design export failed: {exc}",
                code="ingest_failed",
            ) from exc

        mesh_id = ing.mesh_id
        job_paths = JobPaths(work_root=work_root, mesh_id=mesh_id)
        ensure_job_layout(job_paths)

        # 3) design/ artifacts: same STL bytes, STEP, source, spec, manifest
        design_dir = job_paths.design_dir
        design_dir.mkdir(parents=True, exist_ok=True)
        part_stl = design_dir / "part.stl"
        part_step = design_dir / "part.step"
        source_py = design_dir / "source.py"
        spec_json = design_dir / "spec.json"
        manifest_path = design_dir / "manifest.json"

        shutil.copy2(exported.stl_path, part_stl)
        shutil.copy2(exported.step_path, part_step)
        source_py.write_text(source_text, encoding="utf-8")
        spec_json.write_text(
            json.dumps(
                {
                    "template_id": template_id,
                    "source": source_label,
                    "params": params,
                    "units": "mm",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        orig_hash = content_sha256(job_paths.original_stl)
        design_hash = content_sha256(part_stl)
        if orig_hash != design_hash:
            raise DesignError(
                "original.stl and design/part.stl content_sha256 mismatch",
                code="hash_mismatch",
                details={"original": orig_hash, "design": design_hash},
            )

        notes: list[str] = [
            "primary_gate=validate_design_mesh absolute floors",
            "self_baseline_accept=safety_net_only",
            "accept_path=accept_candidate (no design rev v1)",
            "slice_skipped_until_0005",
        ]

        # 4) Absolute validate (PRIMARY gate) — before accept
        # Pass original.stl so volume can recheck with process=True (STL vertex soup).
        validate_design_mesh(ing.stats, mesh_path=job_paths.original_stl)

        # 5) Triage
        try:
            mesh_triage(mesh_id, work_root=work_root)
        except Exception as exc:
            notes.append(f"triage_warning: {type(exc).__name__}: {exc}")

        # 6) Views (honest stubs OK)
        view_paths, view_kind, view_notes = _render_design_views(
            mesh_id, work_root=work_root, no_diff=no_diff
        )
        notes.extend(view_notes)

        # 7) accept_candidate only — for_design + topology; no slice
        acceptance: AcceptanceResult = accept_candidate(
            ing.stats,
            ing.stats,
            policy=GuardPolicy.for_design(),
            require_views=True,
            view_paths=view_paths,
            view_kind=view_kind,
            view_notes=view_notes,
            allow_stubs=True,
            check_topology=True,
            require_slice=False,
        )

        manifest = DesignManifest(
            template_id=template_id,
            source=source_label,
            params=params,
            units="mm",
            export_stl=dict(exported.export_stl),
            export_step=dict(exported.export_step),
            content_sha256=orig_hash,
            notes=list(notes),
            runner=dict(exported.runner_meta),
        )
        manifest_path.write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        ok = bool(acceptance.ok)
        return DesignResult(
            ok=ok,
            mesh_id=mesh_id,
            job_dir=job_paths.job_dir,
            paths={
                "original_stl": str(job_paths.original_stl),
                "design_stl": str(part_stl),
                "design_step": str(part_step),
                "source": str(source_py),
                "spec": str(spec_json),
                "manifest": str(manifest_path),
                "views_dir": str(job_paths.views_dir),
            },
            acceptance=acceptance,
            manifest=manifest,
            notes=notes,
        )


def design_from_template(
    template_id: str = "bracket_m4",
    *,
    params: BracketParams | dict[str, Any] | None = None,
    work_root: Path | str = "work",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    no_diff: bool = False,
) -> DesignResult:
    """Render template → harness export → job + accept_candidate."""
    if template_id == "bracket_m4":
        if params is None:
            bp = BracketParams()
        elif isinstance(params, BracketParams):
            bp = params
        else:
            try:
                bp = BracketParams.model_validate(params)
            except Exception as exc:
                raise DesignError(
                    f"invalid BracketParams: {exc}",
                    code="template_error",
                ) from exc
        source_text = render_template(template_id, bp)
        param_dict = bp.model_dump(mode="json")
    else:
        source_text = render_template(template_id, params)
        if isinstance(params, BracketParams):
            param_dict = params.model_dump(mode="json")
        elif isinstance(params, dict):
            param_dict = dict(params)
        else:
            param_dict = {}

    return _stage_and_ingest(
        source_text=source_text,
        work_root=Path(work_root),
        timeout_s=timeout_s,
        template_id=template_id,
        params=param_dict,
        source_label=None,
        no_diff=no_diff,
    )


def run_design_pipeline(
    source: str | Path,
    *,
    work_root: Path | str = "work",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    no_diff: bool = False,
) -> DesignResult:
    """Run agent/template geometry source end-to-end into a design job."""
    if isinstance(source, Path):
        source_text = source.read_text(encoding="utf-8")
        source_label = str(source)
    else:
        source_text = source
        source_label = "<inline>"

    return _stage_and_ingest(
        source_text=source_text,
        work_root=Path(work_root),
        timeout_s=timeout_s,
        template_id=None,
        params={},
        source_label=source_label,
        no_diff=no_diff,
    )
