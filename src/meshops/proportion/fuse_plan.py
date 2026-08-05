"""Blockout fuse plan artifact (0039) — pure data, no Blender execution.

Authoring weld assist only — FUSE_HONESTY / N6. Not mesh or print success.
Schema 1.0.0 for fuse_plan only (not a recipe schema bump).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import FUSE_HONESTY

FUSE_PLAN_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
FUSE_PLAN_BASENAME: Final[str] = "fuse_plan.json"

DEFAULT_VOXEL_COARSE_M: Final[float] = 0.02
DEFAULT_VOXEL_FINE_M: Final[float] = 0.014
DEFAULT_MAX_LOCAL_GROW: Final[float] = 1.08
DEFAULT_ARCHIVE_SUFFIX: Final[str] = "_pre_fuse"

DEFAULT_PROCEDURE: Final[tuple[str, ...]] = (
    "archive *_pre_fuse",
    "join RECIPE_*",
    "voxel_coarse 0.02",
    "voxel_fine 0.014",
    "light_smooth (sculpt.mesh_filter, 1-2 passes, not 8x global)",
)

DEFAULT_FORBID: Final[tuple[str, ...]] = (
    "global_uniform_scale > 1.05 body",
    "solidify_multi_island_bridge",
    "full_mesh_boolean_after_solidify",
)


class BlockoutFusePlan(BaseModel):
    """fuse_plan.json — agent/skill procedure only; no bpy in product."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = FUSE_PLAN_SCHEMA_VERSION
    honesty: str = FUSE_HONESTY
    procedure: list[str] = Field(default_factory=lambda: list(DEFAULT_PROCEDURE))
    voxel_m: dict[str, float] = Field(
        default_factory=lambda: {
            "coarse": DEFAULT_VOXEL_COARSE_M,
            "fine": DEFAULT_VOXEL_FINE_M,
        }
    )
    max_local_grow: float = DEFAULT_MAX_LOCAL_GROW
    archive_suffix: str = DEFAULT_ARCHIVE_SUFFIX
    forbid: list[str] = Field(default_factory=lambda: list(DEFAULT_FORBID))
    qa: dict[str, Any] = Field(
        default_factory=lambda: {
            "target_islands_max": 1,
            "boundary_edges": 0,
            "note": "N6 — island/boundary QA is authoring weld only, not print success",
        }
    )
    source_recipe_id: str | None = None
    messages: list[str] = Field(default_factory=list)


def build_fuse_plan(package: Any) -> BlockoutFusePlan:
    """Build default fuse plan from a loaded recipe package (no mutate)."""
    recipe_id = getattr(package, "recipe_id", None)
    join_ready = bool(getattr(package, "join_ready", False))
    msgs: list[str] = [
        "fuse_plan authoring weld assist only — not mesh or print success",
        f"source recipe_id={recipe_id}",
        f"join_ready={str(join_ready).lower()}",
    ]
    return BlockoutFusePlan(
        honesty=FUSE_HONESTY,
        procedure=list(DEFAULT_PROCEDURE),
        voxel_m={"coarse": DEFAULT_VOXEL_COARSE_M, "fine": DEFAULT_VOXEL_FINE_M},
        max_local_grow=DEFAULT_MAX_LOCAL_GROW,
        archive_suffix=DEFAULT_ARCHIVE_SUFFIX,
        forbid=list(DEFAULT_FORBID),
        source_recipe_id=str(recipe_id) if recipe_id is not None else None,
        messages=msgs,
    )


def write_fuse_plan(
    out: Path | str,
    plan: BlockoutFusePlan,
    *,
    force: bool = False,
) -> Path:
    """Write fuse_plan.json to *out* (file or directory)."""
    raw = str(out)
    ends_sep = raw.endswith(("/", "\\"))
    base = Path(raw.rstrip("/\\") if ends_sep else raw)
    if ends_sep or (base.exists() and base.is_dir()) or base.suffix.lower() != ".json":
        path = base / FUSE_PLAN_BASENAME
    else:
        path = base

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            raise ProportionError(
                f"output already exists (use --force): {path}",
                code="write_failed",
                details={"path": str(path)},
            )
        path.write_text(
            json.dumps(plan.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
    except ProportionError:
        raise
    except OSError as exc:
        raise ProportionError(
            f"failed to write fuse plan: {exc}",
            code="write_failed",
            details={"out": str(path)},
        ) from exc
    return path


def run_blockout_fuse_plan(
    recipe_path: Path | str,
    out: Path | str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """CLI helper: load recipe → build plan → write JSON."""
    from meshops.proportion.blockout_recipe import load_blockout_recipe

    package = load_blockout_recipe(recipe_path)
    plan = build_fuse_plan(package)
    path = write_fuse_plan(out, plan, force=force)
    return {
        "ok": True,
        "paths": [str(path)],
        "honesty": FUSE_HONESTY,
        "schema_version": plan.schema_version,
        "voxel_m": dict(plan.voxel_m),
        "max_local_grow": plan.max_local_grow,
        "procedure": list(plan.procedure),
        "messages": list(plan.messages),
    }


__all__ = [
    "DEFAULT_ARCHIVE_SUFFIX",
    "DEFAULT_MAX_LOCAL_GROW",
    "DEFAULT_PROCEDURE",
    "DEFAULT_VOXEL_COARSE_M",
    "DEFAULT_VOXEL_FINE_M",
    "FUSE_HONESTY",
    "FUSE_PLAN_BASENAME",
    "FUSE_PLAN_SCHEMA_VERSION",
    "BlockoutFusePlan",
    "build_fuse_plan",
    "run_blockout_fuse_plan",
    "write_fuse_plan",
]
