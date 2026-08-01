"""Organic recipe registry + param validation (track 0006).

Meta types (B1): BALL, CAPSULE, PLANE, ELLIPSOID, CUBE — no SPHERE/TORUS.
Meta fields (B2): co, radius, size_x/y/z, rotation, stiffness, use_negative, type.
Both resolution + render_resolution must be set (B5); resolution ∈ [0.05, 2.0].
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from meshops.organic.errors import OrganicError

RecipeId = Literal["simple_bust", "simple_figurine", "from_mesh"]

RECIPE_IDS: frozenset[str] = frozenset({"simple_bust", "simple_figurine", "from_mesh"})

# Host-verified Blender 5.2 MetaElement types (B1) — scripts must use only these.
META_TYPES: frozenset[str] = frozenset({"BALL", "CAPSULE", "PLANE", "ELLIPSOID", "CUBE"})

# Host-verified MetaElement field names (B2).
META_FIELDS: frozenset[str] = frozenset(
    {
        "co",
        "radius",
        "size_x",
        "size_y",
        "size_z",
        "rotation",
        "stiffness",
        "use_negative",
        "type",
    }
)

# N1/N2/N8 refuse keywords in freeform params / notes.
_REFUSED_OPS: frozenset[str] = frozenset(
    {
        "voxel_remesh",
        "whole_model_voxel",
        "voxel remesh",
        "boolean_after_solidify",
        "full_mesh_boolean",
        "linked_flat_delete",
        "linked-flat",
        "auto_delete_sheet",
    }
)

RESOLUTION_MIN = 0.05
RESOLUTION_MAX = 2.0
THRESHOLD_MIN = 0.2
THRESHOLD_MAX = 2.0
SMOOTH_ITERS_MAX = 5


class RecipeParams(BaseModel):
    """Validated recipe parameters passed to Blender via params.json."""

    model_config = ConfigDict(extra="forbid")

    recipe: RecipeId = "simple_bust"
    scale_mm: float = Field(default=100.0, gt=10.0, lt=500.0)
    resolution: float = Field(default=0.5, ge=RESOLUTION_MIN, le=RESOLUTION_MAX)
    threshold: float = Field(default=0.6, ge=THRESHOLD_MIN, le=THRESHOLD_MAX)
    smooth_iterations: int = Field(default=0, ge=0, le=SMOOTH_ITERS_MAX)
    # from_mesh only — absolute path after runner resolve, or token before resolve
    source_stl: str | None = None

    @field_validator("resolution")
    @classmethod
    def _cap_resolution(cls, v: float) -> float:
        if v < RESOLUTION_MIN or v > RESOLUTION_MAX:
            raise ValueError(f"resolution must be in [{RESOLUTION_MIN}, {RESOLUTION_MAX}], got {v}")
        return v


def validate_recipe_id(recipe: str) -> RecipeId:
    if recipe not in RECIPE_IDS:
        raise OrganicError(
            f"unknown recipe {recipe!r}; known: {sorted(RECIPE_IDS)}",
            code="recipe_unknown",
            details={"recipe": recipe, "known": sorted(RECIPE_IDS)},
        )
    return recipe  # type: ignore[return-value]


def refuse_n1_n2_params(params: dict[str, Any]) -> None:
    """Hard refuse N1 whole-voxel remesh / N2 boolean-after-solidify / N8 sheet delete."""
    blob = " ".join(f"{k}={v}" for k, v in params.items()).lower()
    for token in _REFUSED_OPS:
        if token.replace("_", " ") in blob or token in blob:
            raise OrganicError(
                f"recipe refused: operation {token!r} is Never (N1/N2/N8)",
                code="recipe_refused",
                details={"token": token},
            )
    # Explicit flags
    for key in ("voxel_remesh", "boolean_after_solidify", "linked_flat_delete"):
        if params.get(key) is True or params.get(key) == 1:
            raise OrganicError(
                f"recipe refused: {key} is Never",
                code="recipe_refused",
                details={"key": key},
            )


def parse_params(
    recipe: str,
    raw: dict[str, Any] | None = None,
) -> RecipeParams:
    """Validate recipe id + params; refuse N1/N2/N8."""
    rid = validate_recipe_id(recipe)
    data = dict(raw or {})
    refuse_n1_n2_params(data)
    data["recipe"] = rid
    if rid == "from_mesh" and not data.get("source_stl"):
        raise OrganicError(
            "from_mesh requires source_stl (absolute path or pass token p001)",
            code="invalid_params",
            details={"field": "source_stl"},
        )
    try:
        return RecipeParams.model_validate(data)
    except Exception as exc:
        raise OrganicError(
            f"invalid recipe params: {exc}",
            code="invalid_params",
            details={"recipe": rid, "error": str(exc)},
        ) from exc
