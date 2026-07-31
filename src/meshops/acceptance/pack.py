"""accept_candidate / accept_revision — compose guards + views + optional slice.

No double-gate: repair/export use build_acceptance_from_guard with in-hand GuardResult.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from meshops.acceptance.honesty import HONESTY_MESSAGE
from meshops.acceptance.hooks import SliceAcceptHook, null_slice_result
from meshops.acceptance.models import (
    AcceptanceResult,
    HonestyKind,
    SliceAcceptResult,
    ViewKind,
)
from meshops.acceptance.numeric import (
    evaluate_topology,
    evaluate_volume_ratio,
    mesh_volume,
    resolve_mesh_for_numeric,
)
from meshops.guards.check import check_export, resolve_stats
from meshops.guards.models import GuardResult
from meshops.guards.policy import GuardPolicy
from meshops.jobstore.paths import JobPaths
from meshops.models.diagnostics import Diagnostics, MeshStats
from meshops.revs.store import load_manifest, resolve_rev_dir, rev_mesh_path

# Known T1/T2 recipe ids eligible for GuardPolicy.for_recipe default.
# Design/sculpt revs (0003/0004) must pass explicit policy=.
_KNOWN_RECIPE_IDS: frozenset[str] = frozenset(
    {
        "t1_clean",
        "t2_smooth_spikes",
        "t2_close_small_holes",
    }
)


def _dedupe(codes: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _infer_view_kind(
    *,
    view_kind: ViewKind | None,
    view_notes: Sequence[str] | None,
    view_paths: Sequence[str],
) -> ViewKind:
    if view_kind is not None:
        return view_kind
    notes = list(view_notes or ())
    # Stub markers include CI/--no-diff (diff_stub*) and F3D-fallback (*used_stub*).
    # Must win over generic "has paths → f3d" so honesty is not overclaimed (§12).
    stub_tokens = ("diff_stub", "used_stub", "_stub")
    if any(any(tok in n for tok in stub_tokens) for n in notes):
        return "stub"
    if view_paths:
        return "f3d"
    return "none"


def _check_views(
    view_paths: Sequence[str],
    *,
    require_views: bool,
    allow_stubs: bool,
    view_kind: ViewKind,
    expected_view_names: Sequence[str] | None,
) -> tuple[bool | None, list[str], list[str]]:
    """Return (views_ok, failed, messages)."""
    failed: list[str] = []
    messages: list[str] = []

    if not require_views:
        return None, failed, messages

    if not view_paths:
        failed.append("missing_views")
        messages.append("require_views=True but view_paths is empty")
        return False, failed, messages

    missing_files: list[str] = []
    empty_files: list[str] = []
    for p in view_paths:
        path = Path(p)
        if not path.is_file():
            missing_files.append(p)
            continue
        if path.stat().st_size <= 0:
            empty_files.append(p)

    if missing_files:
        failed.append("missing_views")
        messages.append(f"view path(s) missing on disk: {missing_files[:5]}")
    if empty_files:
        failed.append("empty_views")
        messages.append(f"view path(s) empty (st_size==0): {empty_files[:5]}")

    if view_kind == "stub" and not allow_stubs:
        failed.append("stub_views_disallowed")
        messages.append("allow_stubs=False but view_kind is stub")

    if expected_view_names:
        basenames = {Path(p).name for p in view_paths}
        stems = {Path(p).stem for p in view_paths}
        for name in expected_view_names:
            if name in basenames or name in stems:
                continue
            # also allow substring match on stem (camera token)
            if any(name in s for s in stems):
                continue
            failed.append("missing_camera_angle")
            messages.append(f"expected view name not found: {name}")

    views_ok = len(failed) == 0
    return views_ok, failed, messages


def _run_slice(
    *,
    slice_hook: SliceAcceptHook | None,
    require_slice: bool,
    slice_profile: str | None,
    candidate_path: str | None,
) -> tuple[SliceAcceptResult, list[str], list[str]]:
    """Return (slice_result, failed_codes, messages). Hook fail always fails pack."""
    failed: list[str] = []
    messages: list[str] = []

    if slice_hook is None:
        result = null_slice_result()
        if require_slice:
            failed.append("slice_not_configured")
            messages.append("require_slice=True but no slice_hook provided")
            result = SliceAcceptResult(
                status="skipped",
                messages=["slice_not_configured"],
                error_code="slice_not_configured",
            )
        return result, failed, messages

    try:
        result = slice_hook(
            candidate_path=candidate_path,
            slice_profile=slice_profile,
        )
    except Exception as exc:
        result = SliceAcceptResult(
            status="fail",
            error_code="slice_hook_exception",
            messages=[f"{type(exc).__name__}: {exc}"],
        )

    if result.status == "fail":
        failed.append("slice_fail")
        messages.extend(result.messages or ["slice hook returned fail"])
    elif result.status == "skipped" and require_slice:
        failed.append("slice_not_configured")
        messages.append("require_slice=True but hook returned skipped")

    return result, failed, messages


def _compose_honesty(
    *,
    ok: bool,
    require_views: bool,
    view_kind: ViewKind,
) -> HonestyKind:
    if not ok:
        return "not_accepted"
    if not require_views:
        return "guards_only"
    if view_kind == "stub":
        return "guards_and_stub_views"
    if view_kind in ("f3d", "workbench", "mixed"):
        return "guards_and_views"
    # require_views but kind none/unknown — still not artistic proof
    return "guards_and_views" if view_kind != "none" else "guards_only"


def _cheap_stats_metrics(
    baseline_stats: MeshStats | None,
    cand_stats: MeshStats | None,
) -> dict[str, Any]:
    """Pack-namespaced flags from MeshStats (no mesh load)."""
    metrics: dict[str, Any] = {}
    if baseline_stats is not None:
        metrics["pack.base_is_watertight"] = baseline_stats.is_watertight
        metrics["pack.base_is_volume"] = baseline_stats.is_volume
        metrics["pack.base_is_manifold"] = baseline_stats.is_manifold
    if cand_stats is not None:
        metrics["pack.cand_is_watertight"] = cand_stats.is_watertight
        metrics["pack.cand_is_volume"] = cand_stats.is_volume
        metrics["pack.cand_is_manifold"] = cand_stats.is_manifold
    return metrics


def build_acceptance_from_guard(
    guard: GuardResult,
    *,
    baseline_stats: MeshStats | None = None,
    cand_stats: MeshStats | None = None,
    baseline: MeshStats | Path | None = None,
    candidate: MeshStats | Path | None = None,
    view_paths: Sequence[str | Path] | None = None,
    require_views: bool = True,
    view_kind: ViewKind | None = None,
    view_notes: Sequence[str] | None = None,
    allow_stubs: bool = True,
    expected_view_names: Sequence[str] | None = None,
    slice_hook: SliceAcceptHook | None = None,
    require_slice: bool = False,
    slice_profile: str | None = None,
    check_volume_ratio: bool = False,
    volume_ratio_min: float = 0.50,
    check_topology: bool = False,
    policy_tier: str | None = None,
) -> AcceptanceResult:
    """Shape AcceptanceResult from a precomputed GuardResult (no second check_export).

    Used by repair/export wiring and by accept_candidate after a single guard call.
    """
    paths_str = [str(p) for p in (view_paths or ())]
    resolved_kind = _infer_view_kind(
        view_kind=view_kind,
        view_notes=view_notes,
        view_paths=paths_str,
    )

    failed: list[str] = list(guard.failed)
    messages: list[str] = list(guard.messages)

    metrics: dict[str, Any] = dict(guard.metrics)
    metrics.update(_cheap_stats_metrics(baseline_stats, cand_stats))

    # --- Optional volume / topology (may load mesh when Path) ---
    if check_volume_ratio or check_topology:
        base_src: MeshStats | Path | None = baseline if baseline is not None else baseline_stats
        cand_src: MeshStats | Path | None = candidate if candidate is not None else cand_stats
        base_mesh = resolve_mesh_for_numeric(base_src) if check_volume_ratio else None
        cand_mesh = resolve_mesh_for_numeric(cand_src)

        if check_volume_ratio:
            base_vol = mesh_volume(base_mesh) if base_mesh is not None else None
            cand_vol = mesh_volume(cand_mesh) if cand_mesh is not None else None
            v_failed, v_msgs, v_metrics = evaluate_volume_ratio(
                base_volume=base_vol,
                cand_volume=cand_vol,
                volume_ratio_min=volume_ratio_min,
            )
            failed.extend(v_failed)
            messages.extend(v_msgs)
            metrics.update(v_metrics)
        else:
            metrics.setdefault("pack.volume_ratio", None)

        if check_topology:
            if cand_mesh is None:
                # topology opted in but no mesh to load — honest gap, no hard-fail
                metrics["pack.degenerate_faces"] = None
                metrics["pack.degenerate_face_ratio"] = None
            else:
                t_failed, t_msgs, t_metrics = evaluate_topology(cand_mesh)
                failed.extend(t_failed)
                messages.extend(t_msgs)
                metrics.update(t_metrics)

    # --- Views ---
    views_ok, v_failed, v_msgs = _check_views(
        paths_str,
        require_views=require_views,
        allow_stubs=allow_stubs,
        view_kind=resolved_kind,
        expected_view_names=expected_view_names,
    )
    failed.extend(v_failed)
    messages.extend(v_msgs)
    metrics["pack.views_ok"] = views_ok
    metrics["pack.view_kind"] = resolved_kind

    # --- Slice ---
    cand_path_str: str | None = None
    if isinstance(candidate, Path):
        cand_path_str = str(candidate)
    elif cand_stats is not None and cand_stats.source_path:
        cand_path_str = cand_stats.source_path

    slice_result, s_failed, s_msgs = _run_slice(
        slice_hook=slice_hook,
        require_slice=require_slice,
        slice_profile=slice_profile,
        candidate_path=cand_path_str,
    )
    failed.extend(s_failed)
    messages.extend(s_msgs)
    metrics["pack.slice_status"] = slice_result.status

    failed = _dedupe(failed)
    # ok requires guard ok AND no pack-level fails (guard.failed already in failed list)
    ok = guard.ok and len(failed) == 0

    honesty = _compose_honesty(ok=ok, require_views=require_views, view_kind=resolved_kind)
    tier = policy_tier if policy_tier is not None else guard.policy_tier

    return AcceptanceResult(
        ok=ok,
        failed=failed,
        messages=messages,
        guard=guard,
        view_paths=paths_str,
        views_ok=views_ok,
        view_kind=resolved_kind,
        slice=slice_result,
        metrics=metrics,
        policy_tier=tier,
        honesty=honesty,
        honesty_message=HONESTY_MESSAGE,
    )


def accept_candidate(
    baseline: MeshStats | Path,
    candidate: MeshStats | Path,
    *,
    policy: GuardPolicy | None = None,
    view_paths: Sequence[Path | str] | None = None,
    require_views: bool = True,
    view_kind: ViewKind | None = None,
    view_notes: Sequence[str] | None = None,
    allow_stubs: bool = True,
    expected_view_names: Sequence[str] | None = None,
    slice_hook: SliceAcceptHook | None = None,
    require_slice: bool = False,
    slice_profile: str | None = None,
    check_volume_ratio: bool = False,
    volume_ratio_min: float = 0.50,
    check_topology: bool = False,
) -> AcceptanceResult:
    """Full acceptance: single check_export + views/slice/honesty composition.

    Prefer precomputed MeshStats from repair/export to avoid reloading large meshes.
    Defaults keep volume/topology off (O(1) stats path).
    """
    # Prefer MeshStats; resolve Path only when needed for guards
    base_stats = resolve_stats(baseline, mesh_id="baseline")
    cand_stats = resolve_stats(candidate, mesh_id="candidate")
    guard = check_export(base_stats, cand_stats, policy=policy)
    return build_acceptance_from_guard(
        guard,
        baseline_stats=base_stats,
        cand_stats=cand_stats,
        baseline=baseline if isinstance(baseline, Path) else base_stats,
        candidate=candidate if isinstance(candidate, Path) else cand_stats,
        view_paths=view_paths,
        require_views=require_views,
        view_kind=view_kind,
        view_notes=view_notes,
        allow_stubs=allow_stubs,
        expected_view_names=expected_view_names,
        slice_hook=slice_hook,
        require_slice=require_slice,
        slice_profile=slice_profile,
        check_volume_ratio=check_volume_ratio,
        volume_ratio_min=volume_ratio_min,
        check_topology=check_topology,
        policy_tier=guard.policy_tier,
    )


def _baseline_from_job(paths: JobPaths) -> MeshStats:
    if paths.diagnostics_json.is_file():
        diag = Diagnostics.model_validate_json(paths.diagnostics_json.read_text(encoding="utf-8"))
        return diag.stats
    if not paths.original_stl.is_file():
        raise FileNotFoundError(
            f"no baseline for mesh_id={paths.mesh_id} (missing diagnostics and original.stl)"
        )
    return resolve_stats(paths.original_stl, mesh_id=paths.mesh_id)


def _default_policy_for_recipe(recipe_id: str) -> GuardPolicy:
    """Map recipe_id → default GuardPolicy when accept_revision policy= is omitted.

    - Known T1/T2 → for_recipe (tight floors)
    - blender_sculpt_import → for_sculpt (export-like + wipeout; 0004)
    - else → for_export

    Design revs (0003) should still pass explicit policy= when possible.
    """
    if recipe_id in _KNOWN_RECIPE_IDS:
        return GuardPolicy.for_recipe(recipe_id)
    if recipe_id == "blender_sculpt_import":
        return GuardPolicy.for_sculpt()
    return GuardPolicy.for_export()


def accept_revision(
    mesh_id: str,
    rev: str,
    *,
    work_root: Path | str = "work",
    policy: GuardPolicy | None = None,
    require_views: bool = True,
    view_kind: ViewKind | None = None,
    allow_stubs: bool = True,
    expected_view_names: Sequence[str] | None = None,
    slice_hook: SliceAcceptHook | None = None,
    require_slice: bool = False,
    slice_profile: str | None = None,
    check_volume_ratio: bool = False,
    volume_ratio_min: float = 0.50,
    check_topology: bool = False,
) -> AcceptanceResult:
    """Accept a job revision by mesh_id + rev id (or failed_* dir name).

    Failed revs (manifest.ok=False or directory name starts with failed_) always
    return ok=False with failed code ``failed_rev``.

    Default policy: known T1/T2 → for_recipe; blender_sculpt_import → for_sculpt;
    else for_export. Design revs (0003) should still pass explicit policy=.
    """
    paths = JobPaths(work_root=Path(work_root), mesh_id=mesh_id)
    if not paths.job_dir.is_dir():
        raise FileNotFoundError(f"job not found: {paths.job_dir}")

    rev_dir = resolve_rev_dir(paths, rev)
    man = load_manifest(rev_dir)

    is_failed_dir = rev_dir.name.startswith("failed_")
    if (not man.ok) or is_failed_dir:
        # Fail-closed: never pretend views fix a failed rev
        empty_guard = GuardResult(
            ok=False,
            failed=["failed_rev"],
            metrics={"rev_id": man.rev_id, "recipe_id": man.recipe_id},
            messages=[
                f"refusing failed revision {rev!r} (manifest.ok={man.ok}, dir={rev_dir.name})"
            ],
            policy_tier="recipe" if man.recipe_id in _KNOWN_RECIPE_IDS else "export",
        )
        return AcceptanceResult(
            ok=False,
            failed=["failed_rev"],
            messages=list(empty_guard.messages),
            guard=empty_guard,
            view_paths=list(man.view_paths),
            views_ok=False,
            view_kind="none",
            slice=null_slice_result(reason="skipped for failed_rev"),
            metrics={
                **empty_guard.metrics,
                "pack.views_ok": False,
                "pack.view_kind": "none",
                "pack.slice_status": "skipped",
            },
            policy_tier=empty_guard.policy_tier,
            honesty="not_accepted",
            honesty_message=HONESTY_MESSAGE,
        )

    baseline = _baseline_from_job(paths)
    cand_path = rev_mesh_path(rev_dir)
    cand_stats = resolve_stats(cand_path, mesh_id=mesh_id)

    pol = policy if policy is not None else _default_policy_for_recipe(man.recipe_id)
    guard = check_export(baseline, cand_stats, policy=pol)

    # Prefer explicit arg, then manifest.view_kind, then notes inference
    man_kind: ViewKind | None = None
    if getattr(man, "view_kind", None) is not None:
        raw = man.view_kind
        if raw in ("f3d", "workbench", "stub", "mixed", "none"):
            man_kind = raw  # type: ignore[assignment]
    resolved_kind_arg = view_kind if view_kind is not None else man_kind

    return build_acceptance_from_guard(
        guard,
        baseline_stats=baseline,
        cand_stats=cand_stats,
        baseline=baseline,
        candidate=cand_path,
        view_paths=man.view_paths,
        require_views=require_views,
        view_kind=resolved_kind_arg,
        view_notes=man.notes,
        allow_stubs=allow_stubs,
        expected_view_names=expected_view_names,
        slice_hook=slice_hook,
        require_slice=require_slice,
        slice_profile=slice_profile,
        check_volume_ratio=check_volume_ratio,
        volume_ratio_min=volume_ratio_min,
        check_topology=check_topology,
        policy_tier=guard.policy_tier,
    )


# Re-export Literal helpers for type checkers using pack module
__all__ = [
    "accept_candidate",
    "accept_revision",
    "build_acceptance_from_guard",
]
