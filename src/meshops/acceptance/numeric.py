"""Optional volume-ratio and topology helpers (opt-in; real trimesh API only).

Defaults keep the stats-only path free of forced mesh loads (spec §3.1 / A2-B2).

Degenerate faces: trimesh has no ``mesh.degenerate_faces`` attribute. Use
``int((~mesh.nondegenerate_faces()).sum())`` (method call) or area_faces == 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from meshops.ingest.stats import load_mesh
from meshops.models.diagnostics import MeshStats

# Hard-fail topology gate when check_topology=True and ratio exceeds this.
DEGENERATE_FACE_RATIO_MAX = 0.05


def count_degenerate_faces(mesh: trimesh.Trimesh) -> int:
    """Count degenerate faces via real trimesh API (never mesh.degenerate_faces).

    ``nondegenerate_faces`` is a *method* on trimesh.Trimesh returning a bool mask.
    """
    mask = np.asarray(mesh.nondegenerate_faces(), dtype=bool)
    return int((~mask).sum())


def degenerate_face_ratio(mesh: trimesh.Trimesh) -> float | None:
    """degenerate / total faces; None if zero faces."""
    n = len(mesh.faces)
    if n <= 0:
        return None
    return count_degenerate_faces(mesh) / float(n)


def mesh_volume(mesh: trimesh.Trimesh) -> float | None:
    """Best-effort signed volume; None if unavailable."""
    try:
        vol = float(mesh.volume)
    except Exception:
        return None
    if vol != vol:  # NaN
        return None
    return vol


def resolve_mesh_for_numeric(
    source: MeshStats | Path | trimesh.Trimesh | None,
) -> trimesh.Trimesh | None:
    """Load mesh only when Path or already-loaded Trimesh; MeshStats alone → None."""
    if source is None:
        return None
    if isinstance(source, trimesh.Trimesh):
        return source
    if isinstance(source, Path):
        if not source.is_file():
            return None
        return load_mesh(source)
    # MeshStats — no free volume/topology without path
    return None


def evaluate_volume_ratio(
    *,
    base_volume: float | None,
    cand_volume: float | None,
    volume_ratio_min: float = 0.50,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return (failed, messages, pack metrics) for volume ratio gate.

    - cand_volume < 0 → fail volume_inverted
    - both known and base > 0 → abs(cand/base) must be >= volume_ratio_min
    - volumes unavailable → metrics None, no hard-fail
    """
    failed: list[str] = []
    messages: list[str] = []
    metrics: dict[str, Any] = {
        "pack.volume_base": base_volume,
        "pack.volume_cand": cand_volume,
        "pack.volume_ratio": None,
    }

    if cand_volume is not None and cand_volume < 0:
        failed.append("volume_inverted")
        messages.append(f"volume_inverted: cand_volume={cand_volume}")
        metrics["pack.volume_ratio"] = None
        return failed, messages, metrics

    if base_volume is None or cand_volume is None:
        return failed, messages, metrics
    if base_volume <= 0:
        metrics["pack.volume_ratio"] = None
        return failed, messages, metrics

    ratio = abs(cand_volume / base_volume)
    metrics["pack.volume_ratio"] = ratio
    if ratio < volume_ratio_min:
        failed.append("volume_ratio")
        messages.append(
            f"volume_ratio fail: abs(cand/base)={ratio:.4f} < {volume_ratio_min} "
            f"(base={base_volume}, cand={cand_volume})"
        )
    return failed, messages, metrics


def evaluate_topology(
    mesh: trimesh.Trimesh,
    *,
    max_ratio: float = DEGENERATE_FACE_RATIO_MAX,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return (failed, messages, pack metrics) for degenerate-face topology gate."""
    failed: list[str] = []
    messages: list[str] = []
    n_deg = count_degenerate_faces(mesh)
    n_faces = len(mesh.faces)
    ratio = (n_deg / float(n_faces)) if n_faces > 0 else None
    metrics: dict[str, Any] = {
        "pack.degenerate_faces": n_deg,
        "pack.degenerate_face_ratio": ratio,
    }
    if ratio is not None and ratio > max_ratio:
        failed.append("degenerate_faces")
        messages.append(
            f"degenerate_face_ratio={ratio:.4f} > {max_ratio} (degenerate={n_deg}/{n_faces})"
        )
    return failed, messages, metrics
