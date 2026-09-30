"""Hosted mesh print qualification (track 0141).

Archives a GLB or STL and writes a report. Does not repair, slice-accept, or
claim a print-ready mesh.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Literal, NoReturn

import trimesh
from pydantic import BaseModel, ConfigDict, Field

from meshops.hosted.convert import glb_to_stl
from meshops.hosted.errors import HostedError
from meshops.hosted.honesty import QUALIFY_HONESTY
from meshops.ingest.stats import compute_stats, load_mesh
from meshops.jobstore.paths import content_sha256
from meshops.models.diagnostics import SheetScoreResult
from meshops.proportion.benchmark_multiview import MultiviewBenchmark, load_benchmark
from meshops.proportion.errors import ProportionError
from meshops.recipes.registry import (
    ALLOWED_PRIMARY_CLASSES,
    NEVER_RECIPE_IDS,
    REFUSED_PRIMARY_CLASSES,
)
from meshops.slice.errors import SliceError
from meshops.slice.runner import RunOrcaFn, run_slice
from meshops.triage.sheet_score import compute_sheet_score

QUALIFY_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
_PROPOSED_RECIPE_IDS: tuple[str, ...] = (
    "t1_clean",
    "t2_close_small_holes",
    "t2_smooth_spikes",
)
_NOT_A_PRINT_GATE = "check_" + "export is not a print gate."
_ACCEPTANCE_SENTENCE = "Acceptance is not a print-ready mesh."
_SLICE_SENTENCE = "slice pass is not watertight proof."
_CONFLICT_SENTENCE = "GLB and STL are both archived; qualification does not pick a print mesh"
_REFUSED_SENTENCE = "T3/T4/T5 is not a T1/T2 repair; escalate is a human next step"
_BENCHMARK_SENTENCE = "benchmark ok is not qualification success"

QualifyStatus = Literal[
    "verdict_pending",
    "rejected",
    "accepted",
    "baseline_only",
    "source_conflict",
    "slice_absent",
    "slice_failed",
    "topology_blocked",
    "refused",
]
RepairStatus = Literal["not_attempted", "refused"]
SliceQualifyStatus = Literal["not_run", "pass", "fail", "orca_not_found"]
UnitsGuess = Literal["metre_scale", "millimetre_scale"]
ArtifactRole = Literal["glb", "stl", "bake"]


class QualificationArtifact(BaseModel):
    """One archived GLB, STL, or bake inside the qualification bundle."""

    model_config = ConfigDict(extra="forbid")

    role: ArtifactRole
    path: str
    sha256: str
    material_names: list[str] = Field(default_factory=list)
    faces: int | None = None
    vertices: int | None = None
    components_raw: int | None = None
    components_welded: int | None = None
    is_watertight: bool | None = None
    is_manifold: bool | None = None
    non_manifold_edge_count: int | None = None
    boundary_edge_count: int | None = None
    extent_max: float | None = None
    units_guess: UnitsGuess | None = None


class QualificationReport(BaseModel):
    """qualification_report.json schema 1.0.0. A report is not print success."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = QUALIFY_SCHEMA_VERSION
    honesty: Literal["hosted_mesh_qualification_not_print_success"]
    ok: bool
    status: QualifyStatus
    visual_verdict: Literal["accept", "reject"] | None = None
    needs_user_input: bool
    rank: None = None
    compare_status: Literal["baseline_only", "context"]
    repair_status: RepairStatus
    proposed_recipes: list[str]
    primary_class: str
    slice_status: SliceQualifyStatus
    print_height_mm: float | None = None
    scale_factor: float | None = None
    benchmark_status: str | None = None
    figure: str | None = None
    source_conflict: bool
    topology_blocks_print: bool
    appearance_in_iou: Literal[False] = False
    artifacts: list[QualificationArtifact]
    messages: list[str]
    paths: list[str]


def _fail(message: str, **details: Any) -> NoReturn:
    raise HostedError(message, code="qualify_failed", details=details or None)


def _hypotheses(
    *,
    sheet: SheetScoreResult,
    is_watertight: bool | None,
    is_manifold: bool | None,
    boundary_edge_count: int | None,
    non_manifold_edge_count: int | None,
) -> list[tuple[str, float]]:
    """Sheet >= 0.45, else non-manifold, else an open boundary. Silence is not T2/T4/T5."""
    found: list[tuple[str, float]] = []
    if sheet.score >= 0.45:
        confidence = min(1.0, sheet.confidence * (0.5 + 0.5 * sheet.score))
        found.append(("T3_sheet", confidence))
    if is_manifold is False or (
        non_manifold_edge_count is not None and non_manifold_edge_count > 0
    ):
        confidence = 0.7 if non_manifold_edge_count and non_manifold_edge_count > 10 else 0.5
        found.append(("T1_topology", confidence))
    elif is_watertight is False and boundary_edge_count is not None and boundary_edge_count > 0:
        found.append(("T1_topology", 0.5))
    return found


def _primary_class(found: list[tuple[str, float]]) -> str:
    if not found:
        return "none"
    return max(found, key=lambda item: item[1])[0]


def classify_repair(
    *,
    sheet: SheetScoreResult,
    is_watertight: bool | None,
    is_manifold: bool | None,
    boundary_edge_count: int | None,
    non_manifold_edge_count: int | None,
) -> tuple[str, RepairStatus, list[str]]:
    """Propose the three T1/T2 ids, or refuse T3/T4/T5. Does not run a recipe."""
    primary = _primary_class(
        _hypotheses(
            sheet=sheet,
            is_watertight=is_watertight,
            is_manifold=is_manifold,
            boundary_edge_count=boundary_edge_count,
            non_manifold_edge_count=non_manifold_edge_count,
        )
    )
    if primary in REFUSED_PRIMARY_CLASSES:
        return primary, "refused", []
    if primary not in ALLOWED_PRIMARY_CLASSES:
        _fail(f"unclassified primary {primary!r}", primary_class=primary)
    proposed = [
        recipe_id for recipe_id in _PROPOSED_RECIPE_IDS if recipe_id not in NEVER_RECIPE_IDS
    ]
    return primary, "not_attempted", proposed


def _require_height(value: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("print height must be a finite number > 0", print_height_mm=value)
    height = float(value)
    if not math.isfinite(height) or height <= 0:
        _fail("print height must be a finite number > 0", print_height_mm=value)
    return height


def _visual(verdict: str | None) -> Literal["accept", "reject"] | None:
    if verdict is None:
        return None
    if verdict == "accept":
        return "accept"
    if verdict == "reject":
        return "reject"
    _fail(f"invalid verdict: {verdict!r}", verdict=verdict)


def _load_checked(path: Path) -> MultiviewBenchmark:
    if not path.is_file():
        _fail(f"benchmark not found: {path}", path=str(path))
    try:
        document = load_benchmark(path)
    except ProportionError as exc:
        raise HostedError(
            f"benchmark failed: {exc}",
            code="qualify_failed",
            details={"path": str(path)},
        ) from exc
    for role, image in document.reference.images.items():
        file = Path(image.path)
        if not file.is_absolute():
            file = path.parent / file
        if not file.is_file() or content_sha256(file) != image.sha256:
            _fail(f"benchmark reference sha mismatch: {role}", role=role, path=str(file))
    return document


def _material_names(glb_path: Path) -> list[str]:
    try:
        loaded = trimesh.load(str(glb_path), file_type="glb", force="scene")
    except HostedError:
        raise
    except Exception as exc:
        _fail(f"GLB load failed: {exc}", path=str(glb_path))
    names: list[str] = []
    geometries = getattr(loaded, "geometry", None)
    if isinstance(geometries, dict):
        for geom in geometries.values():
            visual = getattr(geom, "visual", None)
            material = getattr(visual, "material", None)
            name = getattr(material, "name", None)
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _archive(src: Path, dest_dir: Path, *, role: str) -> Path:
    """Copy *src* to ``<dest>/<role>-<filename>`` so GLB and STL cannot overwrite."""
    if not src.is_file():
        _fail(f"mesh not found: {src}", path=str(src))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{role}-{src.name}"
    if dest.exists():
        _fail(f"archive path already exists: {dest.name}", path=str(dest))
    try:
        shutil.copy2(src, dest)
    except OSError as exc:
        _fail(f"archive copy failed: {exc}", path=str(src))
    return dest


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    """Count degree-1 edges. trimesh 4.x has no edges_boundary attribute."""
    import numpy as np

    edges = np.asarray(mesh.edges_sorted)
    if len(edges) == 0:
        return 0
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


def _measure(path: Path) -> tuple[dict[str, Any], trimesh.Trimesh, SheetScoreResult]:
    try:
        raw_mesh = load_mesh(path)
        digest = content_sha256(path)
        size = path.stat().st_size
        raw_stats = compute_stats(
            raw_mesh,
            mesh_id="qualify",
            content_sha256_hex=digest,
            file_size_bytes=size,
        )
        welded = raw_mesh.copy()
        welded.merge_vertices(digits_vertex=6)
        welded_stats = compute_stats(
            welded,
            mesh_id="qualify",
            content_sha256_hex=digest,
            file_size_bytes=size,
        )
        sheet = compute_sheet_score(welded)
        boundary = welded_stats.boundary_edge_count
        if boundary is None:
            boundary = _boundary_edge_count(welded)
        extent = max(float(welded_stats.bbox_max[i] - welded_stats.bbox_min[i]) for i in range(3))
    except HostedError:
        raise
    except Exception as exc:
        _fail(f"mesh measure failed: {exc}", path=str(path))
    units: UnitsGuess = "millimetre_scale" if extent >= 10 else "metre_scale"
    fields: dict[str, Any] = {
        "faces": welded_stats.faces,
        "vertices": welded_stats.vertices,
        "components_raw": raw_stats.components,
        "components_welded": welded_stats.components,
        "is_watertight": welded_stats.is_watertight,
        "is_manifold": welded_stats.is_manifold,
        "non_manifold_edge_count": welded_stats.non_manifold_edge_count,
        "boundary_edge_count": boundary,
        "extent_max": extent,
        "units_guess": units,
    }
    return fields, welded, sheet


def _artifact(
    *,
    role: ArtifactRole,
    path: Path,
    root: Path,
    material_names: list[str] | None = None,
    fields: dict[str, Any] | None = None,
) -> QualificationArtifact:
    payload: dict[str, Any] = {
        "role": role,
        "path": _rel(path, root),
        "sha256": content_sha256(path),
        "material_names": list(material_names or []),
    }
    if fields is not None:
        payload.update(fields)
    return QualificationArtifact.model_validate(payload)


def _decide(
    *,
    verdict: str | None,
    source_conflict: bool,
    has_benchmark: bool,
    slice_status: SliceQualifyStatus,
    topology_blocks_print: bool,
    repair_status: RepairStatus,
    needs_user_input: bool,
) -> tuple[bool, QualifyStatus]:
    if verdict is None:
        return False, "verdict_pending"
    if verdict == "reject":
        return False, "rejected"
    if source_conflict:
        return False, "source_conflict"
    if not has_benchmark:
        return False, "baseline_only"
    if slice_status != "pass":
        return False, "slice_absent" if slice_status == "not_run" else "slice_failed"
    if topology_blocks_print:
        return False, "topology_blocked"
    if repair_status == "refused":
        return False, "refused"
    if needs_user_input:
        return False, "verdict_pending"
    return True, "accepted"


def _markdown(report: QualificationReport) -> str:
    lines = [
        "# Hosted mesh qualification",
        "",
        f"status: {report.status}",
        f"ok: {str(report.ok).lower()}",
        f"honesty: {report.honesty}",
        "",
        *report.messages,
        "",
    ]
    return "\n".join(lines)


def run_hosted_qualify(
    *,
    out: Path | str,
    glb: Path | str | None = None,
    stl: Path | str | None = None,
    benchmark: Path | str | None = None,
    qualify_slice: bool = False,
    print_height_mm: float | None = None,
    figure: str | None = None,
    verdict: str | None = None,
    run_orca_fn: RunOrcaFn | None = None,
    orca_path: Path | str | None = None,
) -> dict[str, Any]:
    """Write qualification_report.json and .md under *out*. Return JSON-mode dict."""
    visual = _visual(verdict)
    glb_path = Path(glb) if glb is not None else None
    stl_path = Path(stl) if stl is not None else None
    benchmark_path = Path(benchmark) if benchmark is not None else None
    if glb_path is None and stl_path is None:
        _fail("at least one of --glb or --stl is required")
    height: float | None = None
    if qualify_slice:
        height = _require_height(print_height_mm)
    if figure is not None and benchmark_path is None:
        _fail("figure requires a benchmark", figure=figure)

    document: MultiviewBenchmark | None = None
    if benchmark_path is not None:
        document = _load_checked(benchmark_path)
        figures = list(document.reference.in_scope_figures)
        if figure is not None and figure not in figures:
            _fail(f"figure {figure!r} is outside the benchmark", figure=figure)

    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = out_dir / "source"
    written: list[Path] = []
    artifacts: list[QualificationArtifact] = []

    archived_glb: Path | None = None
    archived_stl: Path | None = None
    material_names: list[str] = []
    if glb_path is not None:
        archived_glb = _archive(glb_path, source_dir, role="glb")
        written.append(archived_glb)
        material_names = _material_names(archived_glb)
    if stl_path is not None:
        archived_stl = _archive(stl_path, source_dir, role="stl")
        written.append(archived_stl)

    source_conflict = archived_glb is not None and archived_stl is not None
    chosen: Path | None = None
    if archived_glb is not None and not source_conflict:
        bake = out_dir / "classified" / "baked.stl"
        try:
            glb_to_stl(archived_glb, bake)
        except HostedError as exc:
            raise HostedError(str(exc), code="qualify_failed", details=exc.details) from exc
        except Exception as exc:
            _fail(f"GLB bake failed: {exc}", path=str(archived_glb))
        written.append(bake)
        chosen = bake
    elif archived_stl is not None and not source_conflict:
        chosen = archived_stl

    if archived_glb is not None:
        artifacts.append(
            _artifact(
                role="glb",
                path=archived_glb,
                root=out_dir,
                material_names=material_names,
            )
        )

    fields: dict[str, Any] | None = None
    welded_mesh: trimesh.Trimesh | None = None
    sheet: SheetScoreResult | None = None
    primary = "unresolved"
    repair_status: RepairStatus = "not_attempted"
    proposed: list[str] = []
    if archived_stl is not None and source_conflict:
        stl_fields, _, _ = _measure(archived_stl)
        artifacts.append(_artifact(role="stl", path=archived_stl, root=out_dir, fields=stl_fields))
    elif chosen is not None:
        fields, welded_mesh, sheet = _measure(chosen)
        primary, repair_status, proposed = classify_repair(
            sheet=sheet,
            is_watertight=fields["is_watertight"],
            is_manifold=fields["is_manifold"],
            boundary_edge_count=fields["boundary_edge_count"],
            non_manifold_edge_count=fields["non_manifold_edge_count"],
        )
        chosen_role: ArtifactRole = "stl" if chosen == archived_stl else "bake"
        artifacts.append(_artifact(role=chosen_role, path=chosen, root=out_dir, fields=fields))

    extent_max = None if fields is None else float(fields["extent_max"])
    slice_status: SliceQualifyStatus = "not_run"
    scale_factor: float | None = None
    if qualify_slice and not source_conflict and chosen is not None and welded_mesh is not None:
        if extent_max is None or extent_max <= 0:
            _fail("extent_max must be > 0 to scale a print mesh", extent_max=extent_max)
        assert height is not None
        scale_factor = height / extent_max
        scaled_path = out_dir / "slice" / "print_scaled.stl"
        scaled_path.parent.mkdir(parents=True, exist_ok=True)
        scaled = welded_mesh.copy()
        scaled.apply_scale(scale_factor)
        try:
            scaled.export(scaled_path)
        except HostedError:
            raise
        except Exception as exc:
            _fail(f"scaled export failed: {exc}", path=str(scaled_path))
        written.append(scaled_path)
        try:
            sliced = run_slice(
                scaled_path,
                mesh_id=None,
                work_root=out_dir,
                run_orca_fn=run_orca_fn,
                orca_path=orca_path,
            )
        except SliceError as exc:
            slice_status = "orca_not_found" if exc.code == "orca_not_found" else "fail"
        else:
            if sliced.status == "pass":
                slice_status = "pass"
            elif sliced.error_code == "orca_not_found":
                slice_status = "orca_not_found"
            else:
                slice_status = "fail"

    components = None if fields is None else int(fields["components_welded"])
    needs_user_input = False
    if figure is None and document is not None and len(document.reference.in_scope_figures) >= 2:
        needs_user_input = True
    if components is not None and components > 1:
        needs_user_input = True
    if sheet is not None and sheet.features.clothing_penalty > 0.3 and sheet.score >= 0.35:
        needs_user_input = True

    topology_blocks = False
    if fields is not None:
        topology_blocks = fields["is_watertight"] is not True or fields["is_manifold"] is not True

    ok, status = _decide(
        verdict=verdict,
        source_conflict=source_conflict,
        has_benchmark=document is not None,
        slice_status=slice_status,
        topology_blocks_print=topology_blocks,
        repair_status=repair_status,
        needs_user_input=needs_user_input,
    )
    messages = [_ACCEPTANCE_SENTENCE, _NOT_A_PRINT_GATE]
    if source_conflict:
        messages.append(_CONFLICT_SENTENCE)
    if document is not None:
        messages.append(_BENCHMARK_SENTENCE)
    if repair_status == "refused":
        messages.append(_REFUSED_SENTENCE)
    if slice_status != "not_run":
        messages.append(_SLICE_SENTENCE)

    report_json = out_dir / "qualification_report.json"
    report_md = out_dir / "qualification_report.md"
    paths = [_rel(path, out_dir) for path in written]
    paths.extend([_rel(report_json, out_dir), _rel(report_md, out_dir)])
    report = QualificationReport(
        honesty=QUALIFY_HONESTY,
        ok=ok,
        status=status,
        visual_verdict=visual,
        needs_user_input=needs_user_input,
        compare_status="context" if document is not None else "baseline_only",
        repair_status=repair_status,
        proposed_recipes=proposed,
        primary_class=primary,
        slice_status=slice_status,
        print_height_mm=height,
        scale_factor=scale_factor,
        benchmark_status=None if document is None else document.status,
        figure=figure,
        source_conflict=source_conflict,
        topology_blocks_print=topology_blocks,
        artifacts=artifacts,
        messages=messages,
        paths=paths,
    )
    report_json.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    report_md.write_text(_markdown(report), encoding="utf-8", newline="\n")
    return report.model_dump(mode="json")
