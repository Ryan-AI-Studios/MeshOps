"""Guarded local booleans via direct manifold3d API (Difficulty §5, §6).

Binding pattern:
  1. Pre-check trimesh is_watertight / is_volume on both operands
  2. Call direct manifold3d (NOT trimesh.boolean as safety path)
  3. Inspect status() — fail on anything other than NoError
  4. Reject empty / zero-triangle / near-zero volume
  5. check_export vs larger input (or combined baseline stats)

No full-Rogue2 / full-mesh-after-solidify product recipe (N2).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import numpy as np
import trimesh

from meshops.guards import GuardPolicy, check_export
from meshops.guards.models import GuardResult
from meshops.ingest.stats import compute_stats
from meshops.models.diagnostics import MeshStats


class BooleanError(RuntimeError):
    """Structured boolean refusal / failure."""

    def __init__(self, message: str, *, code: str = "boolean_error") -> None:
        super().__init__(message)
        self.code = code


class BooleanOp(StrEnum):
    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"


def _import_manifold3d() -> Any:
    try:
        import manifold3d
    except ImportError as exc:
        raise BooleanError(
            "manifold3d is not installed",
            code="missing_manifold3d",
        ) from exc
    return manifold3d


def _precheck_volume(mesh: trimesh.Trimesh, label: str) -> None:
    try:
        wt = bool(mesh.is_watertight)
    except Exception as exc:
        raise BooleanError(
            f"{label}: is_watertight unavailable: {exc}",
            code="precheck",
        ) from exc
    try:
        vol = bool(mesh.is_volume)
    except Exception as exc:
        raise BooleanError(
            f"{label}: is_volume unavailable: {exc}",
            code="precheck",
        ) from exc
    if not wt or not vol:
        raise BooleanError(
            f"{label}: operands must be watertight volumes "
            f"(is_watertight={wt}, is_volume={vol}); run T1 first",
            code="not_volume",
        )


def _trimesh_to_manifold(mesh: trimesh.Trimesh, m3d: Any) -> Any:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    if verts.ndim != 2 or verts.shape[1] < 3:
        raise BooleanError("invalid vertex array", code="invalid_mesh")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise BooleanError("faces must be triangles", code="invalid_mesh")
    # vert_properties: (n, >=3); use first 3 coords
    props = np.ascontiguousarray(verts[:, :3], dtype=np.float32)
    tris = np.ascontiguousarray(faces, dtype=np.uint32)
    raw = m3d.Mesh(vert_properties=props, tri_verts=tris)
    man = m3d.Manifold(raw)
    return man


def _manifold_to_trimesh(man: Any) -> trimesh.Trimesh:
    mesh = man.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)
    if verts.ndim == 2 and verts.shape[1] > 3:
        verts = verts[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _stats_from_trimesh(mesh: trimesh.Trimesh, *, mesh_id: str) -> MeshStats:
    # Synthetic file size estimate from face count (binary STL ~ 84 + 50*faces)
    est_bytes = 84 + 50 * len(mesh.faces)
    return compute_stats(
        mesh,
        mesh_id=mesh_id,
        content_sha256_hex="0" * 64,
        file_size_bytes=est_bytes,
        source_path=None,
    )


def boolean_meshes(
    a: trimesh.Trimesh,
    b: trimesh.Trimesh,
    op: BooleanOp | Literal["union", "difference", "intersection"] = BooleanOp.UNION,
    *,
    run_guards: bool = True,
) -> tuple[trimesh.Trimesh, GuardResult | None]:
    """Local guarded boolean. Returns (result_mesh, guard_result|None).

    Raises BooleanError on pre-check fail, status fail, empty result, or guard fail.
    """
    op_e = BooleanOp(op) if not isinstance(op, BooleanOp) else op
    _precheck_volume(a, "operand_a")
    _precheck_volume(b, "operand_b")

    m3d = _import_manifold3d()
    ma = _trimesh_to_manifold(a, m3d)
    mb = _trimesh_to_manifold(b, m3d)

    st_a = ma.status()
    st_b = mb.status()
    no_error = m3d.Error.NoError
    if st_a != no_error:
        raise BooleanError(f"operand_a manifold status={st_a}", code="status")
    if st_b != no_error:
        raise BooleanError(f"operand_b manifold status={st_b}", code="status")

    if op_e is BooleanOp.UNION:
        result = ma + mb
    elif op_e is BooleanOp.DIFFERENCE:
        result = ma - mb
    elif op_e is BooleanOp.INTERSECTION:
        result = ma ^ mb  # manifold3d: __xor__ == Intersect
    else:
        raise BooleanError(f"unsupported op: {op_e}", code="bad_op")

    status = result.status()
    if status != no_error:
        raise BooleanError(
            f"boolean status={status} (NotManifold/NonFiniteVertex/ResultTooLarge/Cancelled/…)",
            code="status",
        )
    if result.is_empty() or result.num_tri() <= 0:
        raise BooleanError("boolean produced empty / zero-triangle mesh", code="empty")

    try:
        vol = float(result.volume())
    except Exception:
        vol = None
    if vol is not None and abs(vol) < 1e-12:
        raise BooleanError("boolean produced near-zero volume", code="empty")

    out = _manifold_to_trimesh(result)
    if len(out.faces) == 0:
        raise BooleanError("converted result has zero faces", code="empty")

    guard: GuardResult | None = None
    if run_guards:
        # Baseline = larger operand by face count
        base_mesh = a if len(a.faces) >= len(b.faces) else b
        base_stats = _stats_from_trimesh(base_mesh, mesh_id="bool_base")
        out_stats = _stats_from_trimesh(out, mesh_id="bool_out")
        # Union may grow faces — use export tier with relaxed face floor for union
        policy = GuardPolicy.for_export()
        if op_e is BooleanOp.UNION:
            policy = policy.model_copy(update={"face_floor_ratio": 0.25})
        guard = check_export(base_stats, out_stats, policy=policy)
        if not guard.ok:
            raise BooleanError(
                f"boolean guard fail: {guard.messages}",
                code="guard_fail",
            )

    return out, guard
