"""GuardPolicy.for_sculpt tier (DoD-8 policy)."""

from __future__ import annotations

from meshops.acceptance.pack import _default_policy_for_recipe
from meshops.guards import GuardPolicy


def test_for_sculpt_tier_present() -> None:
    p = GuardPolicy.for_sculpt()
    assert p.tier == "sculpt"
    assert p.face_floor_ratio == 0.50
    assert p.size_floor_ratio == 0.40
    assert p.enforce_global_wipeout is True
    assert "sculpt_tier" in p.notes
    assert p.recipe_id == "blender_sculpt_import"


def test_for_sculpt_export_like_not_recipe_tight() -> None:
    sculpt = GuardPolicy.for_sculpt()
    recipe = GuardPolicy.for_recipe("t1_clean")
    assert sculpt.face_floor_ratio < recipe.face_floor_ratio
    assert sculpt.size_floor_ratio < recipe.size_floor_ratio


def test_default_policy_maps_blender_sculpt_import_to_sculpt() -> None:
    p = _default_policy_for_recipe("blender_sculpt_import")
    assert p.tier == "sculpt"
    assert p.recipe_id == "blender_sculpt_import"
    assert p.face_floor_ratio == GuardPolicy.for_sculpt().face_floor_ratio
