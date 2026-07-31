"""Revision history models — RevManifest + RecipeResult."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.acceptance.models import AcceptanceResult
from meshops.guards.models import GuardResult

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

ViewKindOpt = Literal["f3d", "workbench", "stub", "mixed", "none"]


class RevManifest(BaseModel):
    """Pinned meta.json schema for a promoted or failed revision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    rev_id: str
    parent_rev: str | None = None
    recipe_id: str
    created_at: str
    ok: bool
    guard_result: GuardResult
    triage_class: str
    mesh_path: str
    mesh_format: str = "stl_binary"
    n_faces: int = 0
    n_vertices: int = 0
    file_size_bytes: int = 0
    view_paths: list[str] = Field(default_factory=list)
    # Optional additive (0011): explicit view_kind; default None → infer from notes
    view_kind: ViewKindOpt | None = None
    error: str | None = None
    filter_metrics: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RecipeResult(BaseModel):
    """Outcome of a single recipe run before/around atomic promote."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    recipe_id: str
    rev_id: str | None = None
    rev_dir: str | None = None
    manifest: RevManifest | None = None
    error: str | None = None
    error_type: str | None = None
    refused: bool = False
    notes: list[str] = Field(default_factory=list)
    # 0011: pack result from in-hand stats (no second check_export)
    acceptance: AcceptanceResult | None = None
