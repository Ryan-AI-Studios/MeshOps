"""One named junction weld plan (0139). Authoring only — not mesh or print success.

Welds the two members of one cluster. Does not join every RECIPE_* part.
Voxel sizes are the 0114 fuse-plan constants. The archive suffix here is
``_pre_surface``; ``DEFAULT_ARCHIVE_SUFFIX`` stays ``_pre_fuse``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.escalate.discover import find_blender
from meshops.escalate.errors import EscalateError
from meshops.escalate.version import require_blender_52
from meshops.guards.check import check_export
from meshops.guards.policy import GuardPolicy
from meshops.ingest.stats import compute_stats, load_mesh
from meshops.jobstore.paths import content_sha256
from meshops.proportion.benchmark_multiview import load_benchmark, score_aligned_masks
from meshops.proportion.blockout_recipe import (
    BlockoutRecipePackage,
    RecipePart,
    load_blockout_recipe,
)
from meshops.proportion.connection_metrics import (
    _MISSING_GAP,
    connection_gap_metrics,
    is_toe_part,
    part_center,
    resolve_join_connections,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.fuse_plan import (
    DEFAULT_ARCHIVE_SUFFIX,
    DEFAULT_FORBID,
    DEFAULT_VOXEL_COARSE_M,
    DEFAULT_VOXEL_FINE_M,
)
from meshops.proportion.honesty import SURFACE_HONESTY

SURFACE_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
SURFACE_ARCHIVE_SUFFIX: Literal["_pre_surface"] = "_pre_surface"
_BPY = Path(__file__).with_name("surface_weld_bpy.py")
_ROLES = ("front", "left", "three_quarter", "back")
_WELDABLE_IDS = ("shoulder_l", "shoulder_r", "hip_l", "hip_r", "neck_torso")


class ClusterRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    members: list[str]
    weldable: bool
    gap_m: float
    reason: str


class SurfaceScore(BaseModel):
    """IoU evidence only. No threshold and no pass bit."""

    model_config = ConfigDict(extra="forbid")

    role: str
    iou: float
    dice: float
    delta_iou: float | None = None


class SurfacePlan(BaseModel):
    """Write-only surface_plan.json. Schema literal 1.0.0."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SURFACE_SCHEMA_VERSION
    honesty: str = SURFACE_HONESTY
    ok: bool
    status: Literal["verdict_pending", "rejected", "accepted", "plan_only"]
    visual_verdict: Literal["accept", "reject"] | None = None
    needs_user_input: bool = False
    rank: None = None
    n_parts: int
    apply_status: Literal["plan_only", "applied", "guard_failed"]
    clusters: list[ClusterRecord]
    gaps: dict[str, float]
    benchmark_status: str | None = None
    scores: list[SurfaceScore] = Field(default_factory=list)
    messages: list[str]
    paths: list[str]
    target_islands_max: int = 1
    archive_suffix: str = SURFACE_ARCHIVE_SUFFIX
    forbid: list[str] = Field(default_factory=list)
    voxel_coarse_m: float = DEFAULT_VOXEL_COARSE_M
    voxel_fine_m: float = DEFAULT_VOXEL_FINE_M


def _guard_stats(path: Path) -> Any:
    """MeshStats for an STL whose triangles do not share vertices on disk.

    ``load_mesh`` keeps that soup, so a face count would be reported as a
    component count. Weld exact duplicates before the sculpt guard.
    """
    mesh = load_mesh(path)
    mesh.merge_vertices()
    return compute_stats(
        mesh,
        mesh_id=path.stem,
        content_sha256_hex=content_sha256(path),
        file_size_bytes=path.stat().st_size,
        source_path=str(path),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_name(name: str) -> str:
    if "." in name:
        head, tail = name.rsplit(".", 1)
        if tail.isdigit():
            return head
    return name


def _forbidden(name: str) -> bool:
    base = _base_name(name)
    return is_toe_part(name) or base.startswith("RECIPE_finger_") or base.startswith("RECIPE_palm_")


def _load_recipe(path: Path) -> BlockoutRecipePackage:
    if not path.is_file():
        raise ProportionError(
            f"recipe not found: {path}",
            code="surface_failed",
            details={"path": str(path)},
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"recipe is not valid JSON: {path}",
            code="surface_failed",
            details={"path": str(path)},
        ) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("parts"), list):
        raise ProportionError(
            f"recipe needs a parts list: {path}",
            code="surface_failed",
            details={"path": str(path)},
        )
    try:
        return load_blockout_recipe(path)
    except ProportionError as exc:
        raise ProportionError(
            f"recipe failed: {exc}",
            code="surface_failed",
            details={"path": str(path)},
        ) from exc


def _classify(
    package: BlockoutRecipePackage,
) -> tuple[list[ClusterRecord], dict[str, float]]:
    parts = list(package.parts)
    gaps = connection_gap_metrics(package)
    rows = resolve_join_connections(parts)
    first: dict[str, tuple[RecipePart, RecipePart]] = {}
    ankle_members: dict[str, list[str]] = {"ankle_l": [], "ankle_r": []}
    neck_head: tuple[RecipePart, RecipePart] | None = None
    neck_chest: tuple[RecipePart, RecipePart] | None = None
    for class_id, child, parent, _axis in rows:
        if class_id in ankle_members:
            for part in (child, parent):
                if part.name not in ankle_members[class_id]:
                    ankle_members[class_id].append(part.name)
            continue
        if class_id == "neck":
            parent_name = _base_name(parent.name)
            if parent.role == "head" or parent_name.startswith("RECIPE_head"):
                neck_head = (child, parent)
            elif neck_chest is None:
                neck_chest = (child, parent)
            continue
        if class_id not in first:
            first[class_id] = (child, parent)

    clusters: list[ClusterRecord] = []
    for side in ("l", "r"):
        clusters.append(
            _junction(
                f"shoulder_{side}",
                first.get(f"shoulder_{side}"),
                float(gaps[f"shoulder_{side}"]),
                allow=True,
            )
        )
        clusters.append(
            _junction(
                f"hip_{side}",
                first.get(f"hip_{side}"),
                float(gaps[f"hip_{side}"]),
                allow=True,
            )
        )
    neck_gap = float(gaps["neck"])
    clusters.append(_junction("neck_torso", neck_chest, neck_gap, allow=True))
    clusters.append(
        _junction("neck_head", neck_head, neck_gap, allow=False, refuse_reason="neck_head_refused")
    )
    for side in ("l", "r"):
        names = ankle_members[f"ankle_{side}"]
        clusters.append(
            ClusterRecord(
                cluster_id=f"ankle_{side}",
                members=names,
                weldable=False,
                gap_m=float(gaps[f"ankle_{side}"]),
                reason="ankle_refused" if names else "missing_member",
            )
        )
    seen = {item.cluster_id for item in clusters}
    for part in parts:
        if not _forbidden(part.name) or part.name in seen:
            continue
        clusters.append(
            ClusterRecord(
                cluster_id=part.name,
                members=[part.name],
                weldable=False,
                gap_m=float(_MISSING_GAP),
                reason="forbid_face_hand_foot",
            )
        )
        seen.add(part.name)
    return clusters, gaps


def _junction(
    cluster_id: str,
    pair: tuple[RecipePart, RecipePart] | None,
    gap_m: float,
    *,
    allow: bool,
    refuse_reason: str = "refused",
) -> ClusterRecord:
    if pair is None:
        return ClusterRecord(
            cluster_id=cluster_id,
            members=[],
            weldable=False,
            gap_m=float(_MISSING_GAP),
            reason="missing_member",
        )
    names = [pair[0].name, pair[1].name]
    if any(_forbidden(name) for name in names):
        return ClusterRecord(
            cluster_id=cluster_id,
            members=names,
            weldable=False,
            gap_m=gap_m,
            reason="forbid_member",
        )
    if not allow:
        return ClusterRecord(
            cluster_id=cluster_id,
            members=names,
            weldable=False,
            gap_m=gap_m,
            reason=refuse_reason,
        )
    return ClusterRecord(
        cluster_id=cluster_id,
        members=names,
        weldable=True,
        gap_m=gap_m,
        reason="named_junction",
    )


def _checked_blender() -> Path:
    """Return a Blender 5.2 executable or raise surface_weld_unavailable."""
    blender = find_blender(require=False)
    if blender is None:
        raise ProportionError(
            "Blender 5.2 is required for a region weld",
            code="surface_weld_unavailable",
        )
    try:
        require_blender_52(blender)
    except EscalateError as exc:
        raise ProportionError(
            f"Blender 5.2 is required for a region weld: {exc}",
            code="surface_weld_unavailable",
            details={"blender": str(blender)},
        ) from exc
    return blender


def _verdict_state(
    verdict: str | None,
    apply_status: str,
    needs_user_input: bool,
) -> tuple[bool, Literal["verdict_pending", "rejected", "accepted", "plan_only"], str | None]:
    if verdict not in (None, "accept", "reject"):
        raise ProportionError(
            f"invalid verdict {verdict!r}",
            code="surface_failed",
            details={"verdict": verdict},
        )
    if verdict is None:
        return False, "verdict_pending", None
    if verdict == "reject":
        return False, "rejected", "reject"
    if apply_status == "applied" and not needs_user_input:
        return True, "accepted", "accept"
    if apply_status == "plan_only":
        return False, "plan_only", "accept"
    return False, "verdict_pending", "accept"


def _member_payload(part: RecipePart) -> dict[str, Any]:
    """Recipe emit fields for one part. ``center`` is the geometric center.

    Kind, endpoints, and box dimensions stay intact. A cylinder is not stored
    as a sphere radius.
    """
    payload = part.model_dump(mode="json")
    payload["center"] = part_center(part)
    return payload


def _write_markdown(plan: SurfacePlan, path: Path) -> None:
    lines = [
        "# surface plan",
        "",
        f"honesty: {plan.honesty}",
        "",
        "Plan acceptance is not a welded mesh and not print success.",
        "",
        f"n_parts: {plan.n_parts}",
        f"status: {plan.status}",
        f"ok: {str(plan.ok).lower()}",
        f"apply_status: {plan.apply_status}",
        "rank: null",
        f"target_islands_max: {plan.target_islands_max} (context only, not success)",
        f"voxel_coarse_m: {plan.voxel_coarse_m}",
        f"voxel_fine_m: {plan.voxel_fine_m}",
        "",
        "## clusters",
        "",
    ]
    for cluster in plan.clusters:
        members = ", ".join(cluster.members) if cluster.members else "(none)"
        lines.append(
            f"- {cluster.cluster_id} weldable={str(cluster.weldable).lower()} "
            f"gap_m={cluster.gap_m} members={members} reason={cluster.reason}"
        )
    lines.extend(["", "## notes", ""])
    lines.extend(f"- {message}" for message in plan.messages)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _snapshot_views(out: Path) -> dict[str, str | None]:
    """SHA-256 of each view PNG before this run writes anything."""
    found: dict[str, str | None] = {}
    for role in _ROLES:
        path = out / "views" / f"{role}.png"
        found[role] = _sha256(path) if path.is_file() else None
    return found


def _scores(
    out: Path,
    benchmark_path: Path | None,
    *,
    apply_status: str,
    prior_views: dict[str, str | None],
) -> list[SurfaceScore]:
    """Score views only after a guarded apply that replaced every view PNG.

    Plan-only runs and reused ``out/views`` files keep their pre-run hash and
    record no scores. IoU is evidence, not acceptance.
    """
    if benchmark_path is None or apply_status != "applied":
        return []
    view_dir = out / "views"
    for role in _ROLES:
        path = view_dir / f"{role}.png"
        if not path.is_file():
            return []
        if _sha256(path) == prior_views.get(role):
            return []
    document = load_benchmark(benchmark_path)
    scores: list[SurfaceScore] = []
    for role in _ROLES:
        image = document.reference.images.get(role)
        if image is None:
            return []
        ref = Path(image.path)
        if not ref.is_file():
            return []
        scored = score_aligned_masks(ref, view_dir / f"{role}.png")
        delta: float | None = None
        role_views = document.views.get(role)
        if role_views is not None:
            delta = float(scored["iou"]) - float(role_views.meshops.iou)
        scores.append(
            SurfaceScore(
                role=role,
                iou=float(scored["iou"]),
                dice=float(scored["dice"]),
                delta_iou=delta,
            )
        )
    return scores


def _verify_benchmark(path: Path) -> Any:
    try:
        document = load_benchmark(path)
    except ProportionError as exc:
        raise ProportionError(
            f"benchmark failed: {exc}",
            code="surface_failed",
            details={"path": str(path)},
        ) from exc
    for role, image in document.reference.images.items():
        file = Path(image.path)
        if not file.is_file() or _sha256(file) != image.sha256:
            raise ProportionError(
                f"benchmark reference sha mismatch: {role}",
                code="surface_failed",
                details={"role": role, "path": str(file)},
            )
    return document


def _run_weld(
    out: Path,
    recipe: Path,
    package: BlockoutRecipePackage,
    cluster: ClusterRecord,
) -> tuple[Literal["applied", "guard_failed"], list[str]]:
    by_name = {part.name: part for part in package.parts}
    members = [_member_payload(by_name[name]) for name in cluster.members]
    frozen = out / "frozen" / recipe.name
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(recipe, frozen)
    archive = out / "archive" / f"{cluster.cluster_id}{SURFACE_ARCHIVE_SUFFIX}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(
        json.dumps(
            {"cluster_id": cluster.cluster_id, "members": members},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    blender = _checked_blender()
    messages = [
        f"frozen recipe sha256={_sha256(frozen)} bytes={frozen.stat().st_size} path={frozen}",
    ]
    with tempfile.TemporaryDirectory(prefix="surface-weld-") as tmp:
        before = Path(tmp) / "joined.stl"
        after = Path(tmp) / "remeshed.stl"
        job_path = out / "frozen" / "surface_job.json"
        job = {
            "cluster_id": cluster.cluster_id,
            "members": members,
            "voxel_coarse_m": DEFAULT_VOXEL_COARSE_M,
            "voxel_fine_m": DEFAULT_VOXEL_FINE_M,
            "before_stl": str(before),
            "after_stl": str(after),
        }
        job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [str(blender), "-b", "-P", str(_BPY), "--", str(job_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if "surface_weld_unavailable" in combined:
            raise ProportionError(
                "region weld operator is unavailable",
                code="surface_weld_unavailable",
                details={"tail": combined[-2000:]},
            )
        if proc.returncode != 0 or "SURFACE_WELD_OK" not in (proc.stdout or ""):
            raise ProportionError(
                f"region weld failed: {combined[-2000:]}",
                code="surface_failed",
            )
        if not before.is_file() or not after.is_file():
            raise ProportionError(
                "region weld did not write both STL files",
                code="surface_failed",
            )
        guard = check_export(
            _guard_stats(before),
            _guard_stats(after),
            policy=GuardPolicy.for_sculpt(),
        )
        weld_path = out / "weld" / f"{cluster.cluster_id}.stl"
        if not guard.ok:
            if weld_path.is_file():
                weld_path.unlink()
            messages.extend(guard.messages)
            messages.append("export guard failed; weld STL was not kept")
            return "guard_failed", messages
        weld_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(after, weld_path)
        messages.append(f"weld stl={weld_path}")
        return "applied", messages


def run_surface_plan(
    recipe: Path | str,
    out: Path | str,
    *,
    benchmark: Path | str | None = None,
    cluster: str | None = None,
    apply: bool = False,
    allow_region_weld: bool = False,
    figure: str | None = None,
    verdict: str | None = None,
) -> dict[str, Any]:
    """Write surface_plan.json and surface_plan.md. Default is plan-only."""
    recipe_path = Path(recipe)
    out_dir = Path(out)
    prior_views = _snapshot_views(out_dir)
    package = _load_recipe(recipe_path)
    clusters, gaps = _classify(package)
    messages = [
        "blockout-surface authoring weld only - not mesh or print success",
        f"fuse voxel coarse {DEFAULT_VOXEL_COARSE_M} fine {DEFAULT_VOXEL_FINE_M}",
        f"fuse archive suffix stays {DEFAULT_ARCHIVE_SUFFIX}",
        "target_islands_max is context only and does not set ok",
        "one island is not mesh or print success",
    ]
    benchmark_path = Path(benchmark) if benchmark is not None else None
    benchmark_status: str | None = None
    needs_user_input = False
    if benchmark_path is not None:
        document = _verify_benchmark(benchmark_path)
        benchmark_status = str(document.status)
        messages.append("benchmark ok is not surface success")
        figures = list(document.reference.in_scope_figures)
        if len(figures) >= 2 and figure is None:
            needs_user_input = True
            messages.append("multi-figure benchmark needs --figure")
        elif figure is not None and figure not in figures:
            raise ProportionError(
                f"figure {figure!r} is outside in_scope_figures",
                code="surface_failed",
                details={"figure": figure, "in_scope_figures": figures},
            )
    if verdict not in (None, "accept", "reject"):
        raise ProportionError(
            f"invalid verdict {verdict!r}",
            code="surface_failed",
            details={"verdict": verdict},
        )
    if allow_region_weld and not apply:
        messages.append("--allow-region-weld without --apply does not weld")
    if apply and not allow_region_weld:
        raise ProportionError(
            "--apply requires --allow-region-weld",
            code="surface_failed",
        )
    selected: ClusterRecord | None = None
    if apply:
        if not cluster:
            raise ProportionError("--apply requires --cluster", code="surface_failed")
        selected = next((item for item in clusters if item.cluster_id == cluster), None)
        if selected is None:
            raise ProportionError(
                f"unknown cluster {cluster}",
                code="surface_failed",
                details={"cluster": cluster},
            )
        if not selected.weldable:
            raise ProportionError(
                f"cluster {cluster} is not weldable",
                code="surface_failed",
                details={"cluster": cluster, "reason": selected.reason},
            )
        if selected.cluster_id not in _WELDABLE_IDS:
            raise ProportionError(
                f"cluster {cluster} is not a named junction",
                code="surface_failed",
            )

    apply_status: Literal["plan_only", "applied", "guard_failed"] = "plan_only"
    if selected is not None:
        apply_status, weld_messages = _run_weld(out_dir, recipe_path, package, selected)
        messages.extend(weld_messages)

    ok, status, visual = _verdict_state(verdict, apply_status, needs_user_input)
    if needs_user_input:
        ok = False
        if status == "accepted":
            status = "plan_only"
    scores = _scores(
        out_dir,
        benchmark_path,
        apply_status=apply_status,
        prior_views=prior_views,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "surface_plan.json"
    md_path = out_dir / "surface_plan.md"
    paths = [str(json_path), str(md_path)]
    if selected is not None:
        frozen = out_dir / "frozen" / recipe_path.name
        if frozen.is_file():
            paths.append(str(frozen))
        archive = out_dir / "archive" / f"{selected.cluster_id}{SURFACE_ARCHIVE_SUFFIX}.json"
        if archive.is_file():
            paths.append(str(archive))
        weld_path = out_dir / "weld" / f"{selected.cluster_id}.stl"
        if weld_path.is_file():
            paths.append(str(weld_path))
    plan = SurfacePlan(
        honesty=SURFACE_HONESTY,
        ok=ok,
        status=status,
        visual_verdict=visual,  # type: ignore[arg-type]
        needs_user_input=needs_user_input,
        n_parts=len(package.parts),
        apply_status=apply_status,
        clusters=clusters,
        gaps=gaps,
        benchmark_status=benchmark_status,
        scores=scores,
        messages=messages,
        paths=paths,
        target_islands_max=1,
        archive_suffix=SURFACE_ARCHIVE_SUFFIX,
        forbid=list(DEFAULT_FORBID),
        voxel_coarse_m=DEFAULT_VOXEL_COARSE_M,
        voxel_fine_m=DEFAULT_VOXEL_FINE_M,
    )
    json_path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _write_markdown(plan, md_path)
    if apply_status == "guard_failed":
        raise ProportionError(
            "region weld failed the sculpt export guard",
            code="surface_failed",
            details={"apply_status": apply_status},
        )
    return plan.model_dump(mode="json")
