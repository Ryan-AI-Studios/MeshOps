"""Sheet score synthetic tests (DoD-3, DoD-4)."""

from __future__ import annotations

from pathlib import Path

import trimesh

from meshops.models.diagnostics import AutoAction
from meshops.triage.sheet_score import NEIGHBORHOOD_K, compute_sheet_score


def _load(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    assert isinstance(m, trimesh.Trimesh)
    return m


def test_arm_sheet_scores_high(arm_sheet_stl: Path) -> None:
    mesh = _load(arm_sheet_stl)
    result = compute_sheet_score(mesh)
    solid = compute_sheet_score(_load(arm_sheet_stl.parent / "solid_cylinder.stl"))
    assert result.score > solid.score
    assert result.score >= 0.35
    assert result.features.neighborhood_k == NEIGHBORHOOD_K
    assert result.features.neighborhood_radius > 0
    assert "delete" not in result.auto_action.value


def test_gap_sheet_elevated(gap_sheet_stl: Path, solid_cylinder_stl: Path) -> None:
    gap = compute_sheet_score(_load(gap_sheet_stl))
    solid = compute_sheet_score(_load(solid_cylinder_stl))
    assert gap.score > solid.score


def test_solid_cylinder_low(solid_cylinder_stl: Path) -> None:
    result = compute_sheet_score(_load(solid_cylinder_stl))
    assert result.score < 0.45
    assert result.auto_action in {AutoAction.NONE, AutoAction.REVIEW}


def test_clothing_never_auto_delete(clothing_cape_stl: Path) -> None:
    """Difficulty §7 / N8 — clothing FP must not force delete."""
    result = compute_sheet_score(_load(clothing_cape_stl))
    assert result.auto_action != "delete"
    assert result.auto_action in {AutoAction.NONE, AutoAction.REVIEW, AutoAction.ESCALATE}
    # Enum has no delete; also ensure value string
    assert result.auto_action.value in {"none", "review", "escalate"}
    # Multi-feature clothing discrimination (not only stage2_used which is often always true)
    assert result.features is not None
    multi = (
        result.features.clothing_penalty > 0
        or result.features.dihedral_crease > 0
        or result.features.planarity > 0
        or result.features.normal_smoothness > 0
    )
    assert multi, "clothing fixture should exercise non-PCA multi-feature sheet path"


def test_scale_invariance_smoke(arm_sheet_stl: Path) -> None:
    """Uniform scale should keep score roughly similar (r = k * diagonal)."""
    mesh = _load(arm_sheet_stl)
    r1 = compute_sheet_score(mesh)
    mesh2 = mesh.copy()
    mesh2.apply_scale(10.0)
    r2 = compute_sheet_score(mesh2)
    # Same order of magnitude / relative classification
    assert abs(r1.score - r2.score) < 0.35
