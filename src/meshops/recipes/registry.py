"""Allowlisted T1/T2 recipes only. Unknown / T3 / remesh → refuse."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from meshops.recipes.t1_clean import RECIPE_ID as T1_CLEAN
from meshops.recipes.t1_clean import run_t1_clean
from meshops.recipes.t2_holes import RECIPE_ID as T2_HOLES
from meshops.recipes.t2_holes import run_t2_close_holes
from meshops.recipes.t2_smooth import RECIPE_ID as T2_SMOOTH
from meshops.recipes.t2_smooth import run_t2_smooth

RecipeFn = Callable[..., dict[str, Any]]

# Primary triage classes allowed for mutation in 0002
ALLOWED_PRIMARY_CLASSES = frozenset(
    {
        "T1_topology",
        "T2_printability",
        # Also allow clean meshes with no defect hyp — still T1/T2 recipes only
        "none",
        "unknown",
    }
)

REFUSED_PRIMARY_CLASSES = frozenset(
    {
        "T3_sheet",
        "T4_missing_volume",
        "T5_mechanical",
    }
)

# Never recipes (N1/N2/N8) — not registered
NEVER_RECIPE_IDS = frozenset(
    {
        "remesh_all",
        "voxel_remesh",
        "full_boolean_after_solidify",
        "delete_sheet",
        "linked_flat_delete",
        "decimate_default",
    }
)

REGISTRY: dict[str, RecipeFn] = {
    T1_CLEAN: run_t1_clean,
    T2_SMOOTH: run_t2_smooth,
    T2_HOLES: run_t2_close_holes,
}


def list_recipes() -> list[str]:
    return sorted(REGISTRY.keys())


def get_recipe(recipe_id: str) -> RecipeFn:
    if recipe_id in NEVER_RECIPE_IDS:
        raise KeyError(f"recipe {recipe_id!r} is permanently refused (Never)")
    if recipe_id not in REGISTRY:
        raise KeyError(f"unknown recipe: {recipe_id!r}; allowlist={list_recipes()}")
    return REGISTRY[recipe_id]


def run_recipe(
    recipe_id: str,
    input_path: Path | str,
    output_path: Path | str,
    **kwargs: Any,
) -> dict[str, Any]:
    fn = get_recipe(recipe_id)
    return fn(input_path, output_path, **kwargs)
