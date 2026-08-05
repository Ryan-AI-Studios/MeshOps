"""0039 emit-setup + fuse_plan (T7-T9).

Authoring only - not mesh/print success (N6 / FUSE_HONESTY).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshops.proportion.blockout_recipe import (
    BlockoutRecipePackage,
    RecipePart,
    emit_bpy_script,
    load_blockout_recipe,
    run_blockout_emit_setup,
    write_blockout_recipe,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.fuse_plan import (
    DEFAULT_MAX_LOCAL_GROW,
    DEFAULT_VOXEL_COARSE_M,
    DEFAULT_VOXEL_FINE_M,
    FUSE_HONESTY,
    FUSE_PLAN_SCHEMA_VERSION,
    build_fuse_plan,
    run_blockout_fuse_plan,
)
from meshops.proportion.honesty import RECIPE_HONESTY


def _minimal_join_ready_package() -> BlockoutRecipePackage:
    return BlockoutRecipePackage(
        join_ready=True,
        parts=[
            RecipePart(
                name="RECIPE_head",
                role="head",
                kind="ellipsoid",
                center=[0.0, 0.0, 1.6],
                rx_m=0.08,
                ry_m=0.09,
                rz_m=0.1,
                placement="full3d",
            ),
            RecipePart(
                name="RECIPE_deltoid_soft_l",
                role="deltoid_soft",
                kind="ellipsoid",
                center=[0.2, 0.0, 1.4],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
                placement="full3d",
            ),
        ],
        messages=["join_ready=true"],
    )


def test_t7_emit_setup_roundtrip_join_ready(tmp_path: Path) -> None:
    """T7: emit-setup round-trip join_ready True + bpy honesty + # join_ready."""
    recipe_path = tmp_path / "blockout_recipe.json"
    pkg = _minimal_join_ready_package()
    write_blockout_recipe(recipe_path, pkg, format="json", force=True)

    out_py = tmp_path / "setup_blockout_recipe.py"
    payload = run_blockout_emit_setup(recipe_path, out_py, force=True)
    assert payload["ok"] is True
    assert payload["join_ready"] is True
    assert out_py.is_file()

    text = out_py.read_text(encoding="utf-8")
    assert RECIPE_HONESTY in text
    assert "# join_ready: True" in text
    assert "RECIPE_head" in text
    assert "RECIPE_deltoid_soft_l" in text

    # Reload JSON still join_ready; re-write preserves field
    loaded = load_blockout_recipe(recipe_path)
    assert loaded.join_ready is True
    script = emit_bpy_script(loaded)
    assert "# join_ready: True" in script


def test_t8_emit_setup_exists_without_force(tmp_path: Path) -> None:
    """T8: emit-setup exists without force → write_failed."""
    recipe_path = tmp_path / "blockout_recipe.json"
    pkg = _minimal_join_ready_package()
    write_blockout_recipe(recipe_path, pkg, format="json", force=True)

    out_py = tmp_path / "setup_blockout_recipe.py"
    run_blockout_emit_setup(recipe_path, out_py, force=True)
    with pytest.raises(ProportionError) as ei:
        run_blockout_emit_setup(recipe_path, out_py, force=False)
    assert ei.value.code == "write_failed"


def test_t9_fuse_plan_defaults_and_honesty(tmp_path: Path) -> None:
    """T9: fuse_plan defaults + FUSE_HONESTY + light_smooth in procedure."""
    recipe_path = tmp_path / "blockout_recipe.json"
    pkg = _minimal_join_ready_package()
    write_blockout_recipe(recipe_path, pkg, format="json", force=True)

    plan = build_fuse_plan(pkg)
    assert plan.schema_version == FUSE_PLAN_SCHEMA_VERSION
    assert plan.honesty == FUSE_HONESTY
    assert plan.honesty == "proportion_blockout_fuse_not_mesh_or_print_success"
    assert plan.voxel_m["coarse"] == DEFAULT_VOXEL_COARSE_M
    assert plan.voxel_m["fine"] == DEFAULT_VOXEL_FINE_M
    assert plan.max_local_grow == DEFAULT_MAX_LOCAL_GROW
    assert plan.archive_suffix == "_pre_fuse"
    assert any("light_smooth" in step for step in plan.procedure)
    # No separate free-form smooth field on model
    assert not hasattr(plan, "smooth") or "smooth" not in plan.model_fields

    out = tmp_path / "fuse_plan.json"
    payload = run_blockout_fuse_plan(recipe_path, out, force=True)
    assert payload["ok"] is True
    assert payload["honesty"] == FUSE_HONESTY
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["voxel_m"]["coarse"] == 0.02
    assert data["voxel_m"]["fine"] == 0.014
    assert data["max_local_grow"] == 1.08
    assert any("light_smooth" in s for s in data["procedure"])
    assert "smooth" not in data or isinstance(data.get("smooth"), type(None))
