"""Multi-signal export / recipe guards (Difficulty §5, §6).

check_export never claims success when any wipeout or floor signal fires.
Baseline prefers MeshStats (from diagnostics) to avoid reloading large meshes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meshops.guards.models import GuardResult
from meshops.guards.policy import HERO_SIZE_ABS_FLOOR_BYTES, GuardPolicy
from meshops.ingest.stats import compute_stats, load_mesh
from meshops.jobstore.paths import content_sha256
from meshops.models.diagnostics import MeshStats


def _centroid(stats: MeshStats) -> tuple[float, float, float]:
    return (
        0.5 * (stats.bbox_min[0] + stats.bbox_max[0]),
        0.5 * (stats.bbox_min[1] + stats.bbox_max[1]),
        0.5 * (stats.bbox_min[2] + stats.bbox_max[2]),
    )


def _centroid_norm(c: tuple[float, float, float]) -> float:
    return float((c[0] ** 2 + c[1] ** 2 + c[2] ** 2) ** 0.5)


def _is_hero_scale(stats: MeshStats, policy: GuardPolicy) -> bool:
    return (
        stats.file_size_bytes >= policy.hero_bytes_threshold
        or stats.faces >= policy.hero_faces_threshold
    )


def resolve_stats(source: MeshStats | Path, *, mesh_id: str = "candidate") -> MeshStats:
    """Normalize MeshStats | Path into MeshStats."""
    if isinstance(source, MeshStats):
        return source
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh path not found for guards: {path}")
    mesh = load_mesh(path)
    digest = content_sha256(path)
    return compute_stats(
        mesh,
        mesh_id=mesh_id,
        content_sha256_hex=digest,
        file_size_bytes=path.stat().st_size,
        source_path=str(path),
    )


def check_export(
    baseline: MeshStats | Path,
    candidate: MeshStats | Path,
    *,
    policy: GuardPolicy | None = None,
) -> GuardResult:
    """Compare candidate against baseline with multi-signal wipeout + floors.

    On ok=False, export / rev promote must abort (fail-closed).
    """
    pol = policy if policy is not None else GuardPolicy.for_export()
    base = resolve_stats(baseline, mesh_id="baseline")
    cand = resolve_stats(candidate, mesh_id="candidate")

    failed: list[str] = []
    messages: list[str] = []
    metrics: dict[str, Any] = {
        "faces_in": base.faces,
        "faces_out": cand.faces,
        "bytes_in": base.file_size_bytes,
        "bytes_out": cand.file_size_bytes,
        "components_in": base.components,
        "components_out": cand.components,
        "bbox_diagonal_in": base.bbox_diagonal,
        "bbox_diagonal_out": cand.bbox_diagonal,
        "face_ratio": (cand.faces / base.faces) if base.faces > 0 else None,
        "size_ratio": (
            (cand.file_size_bytes / base.file_size_bytes) if base.file_size_bytes > 0 else None
        ),
        "policy_tier": pol.tier,
        "recipe_id": pol.recipe_id,
        "hero_scale": _is_hero_scale(base, pol),
    }

    face_ratio = metrics["face_ratio"]
    size_ratio = metrics["size_ratio"]

    # --- Face floor ---
    if base.faces > 0 and face_ratio is not None and face_ratio < pol.face_floor_ratio:
        failed.append("face_floor")
        messages.append(
            f"face floor fail: ratio={face_ratio:.4f} < {pol.face_floor_ratio} "
            f"(faces {cand.faces}/{base.faces}, tier={pol.tier})"
        )

    # --- Size floor (ratio always; abs for hero) ---
    if base.file_size_bytes > 0 and size_ratio is not None and size_ratio < pol.size_floor_ratio:
        failed.append("size_floor")
        messages.append(
            f"size floor fail: ratio={size_ratio:.4f} < {pol.size_floor_ratio} "
            f"(bytes {cand.file_size_bytes}/{base.file_size_bytes})"
        )

    hero = _is_hero_scale(base, pol)
    abs_floor = pol.size_abs_floor_bytes
    if abs_floor is None and hero:
        abs_floor = HERO_SIZE_ABS_FLOOR_BYTES
    if abs_floor is not None and hero and cand.file_size_bytes < abs_floor:
        if "size_floor" not in failed:
            failed.append("size_floor")
        messages.append(f"hero abs size floor fail: bytes_out={cand.file_size_bytes} < {abs_floor}")
        metrics["size_abs_floor_bytes"] = abs_floor

    # --- Multi-signal wipeout (hero-scale only; non-negotiable) ---
    if hero and pol.enforce_global_wipeout:
        # Mass loss: tiny output AND tiny faces, OR >90% face collapse
        mass_loss = (
            cand.file_size_bytes < pol.wipeout_bytes_out and cand.faces < pol.wipeout_faces_out
        )
        face_collapse = base.faces > 0 and cand.faces < pol.face_collapse_out_ratio * base.faces
        if mass_loss:
            failed.append("wipeout_class")
            messages.append(
                f"wipeout_class: bytes_out={cand.file_size_bytes} < {pol.wipeout_bytes_out} "
                f"and faces_out={cand.faces} < {pol.wipeout_faces_out} "
                f"(hero baseline faces={base.faces} bytes={base.file_size_bytes})"
            )
        if face_collapse:
            failed.append("face_collapse")
            messages.append(
                f"face_collapse: faces_out={cand.faces} < "
                f"{pol.face_collapse_out_ratio} * faces_in={base.faces}"
            )

        # Component disintegration >75% drop
        if base.components > 0:
            drop = 1.0 - (cand.components / base.components)
            metrics["component_drop"] = drop
            if drop > pol.component_collapse_drop:
                failed.append("component_collapse")
                messages.append(
                    f"component_collapse: components {base.components}→{cand.components} "
                    f"drop={drop:.2%} > {pol.component_collapse_drop:.0%}"
                )

        # Origin orphan / bbox drift (Difficulty §5)
        c_base = _centroid(base)
        c_cand = _centroid(cand)
        n_base = _centroid_norm(c_base)
        n_cand = _centroid_norm(c_cand)
        metrics["centroid_norm_in"] = n_base
        metrics["centroid_norm_out"] = n_cand

        origin_orphan = n_cand <= pol.origin_centroid_eps and n_base >= pol.baseline_offset_min
        extents_collapse = (
            base.bbox_diagonal > pol.near_zero_diagonal
            and cand.bbox_diagonal < pol.extents_collapse_ratio * base.bbox_diagonal
        )
        near_zero_orphan = base.bbox_diagonal > 1.0 and cand.bbox_diagonal <= pol.near_zero_diagonal
        if origin_orphan:
            failed.append("bbox_origin_orphan")
            messages.append(
                f"bbox_origin_orphan: candidate centroid |c|={n_cand:.4f} near origin "
                f"while baseline |c|={n_base:.4f}"
            )
        if extents_collapse or near_zero_orphan:
            failed.append("bbox_drift")
            messages.append(
                f"bbox_drift: diagonal {base.bbox_diagonal:.6g}→{cand.bbox_diagonal:.6g}"
            )

    # --- Component explosion (both tiers) ---
    if pol.allow_component_growth and base.components > 0:
        max_allowed = max(
            base.components + pol.component_growth_k,
            int(pol.component_growth_factor * base.components),
        )
        metrics["components_max_allowed"] = max_allowed
        if cand.components > max_allowed:
            failed.append("components")
            messages.append(
                f"component explosion: {cand.components} > max_allowed={max_allowed} "
                f"(in={base.components})"
            )

    # --- Volume sanity ---
    if pol.check_volume and base.is_volume is True:
        if cand.is_volume is False:
            failed.append("volume")
            messages.append("volume: baseline is_volume=True but candidate is not a volume")
        # Near-zero volume when computable would need mesh load; use diagonal proxy if tiny
        if cand.bbox_diagonal <= pol.near_zero_diagonal and base.bbox_diagonal > 1.0:
            if "volume" not in failed:
                failed.append("volume")
            messages.append("volume: near-zero bbox diagonal on solid baseline")

    # Deduplicate failed codes preserving order
    seen: set[str] = set()
    uniq_failed: list[str] = []
    for code in failed:
        if code not in seen:
            seen.add(code)
            uniq_failed.append(code)

    ok = len(uniq_failed) == 0
    if ok:
        messages.append("all guards passed")

    return GuardResult(
        ok=ok,
        failed=uniq_failed,
        metrics=metrics,
        messages=messages,
        policy_tier=pol.tier,
    )
