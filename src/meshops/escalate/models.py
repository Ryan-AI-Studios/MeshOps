"""Escalation ROI / handoff / preview / import-sculpt models (schema 1.0.0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.acceptance.models import AcceptanceResult

ESCALATE_SCHEMA: Literal["1.0.0"] = "1.0.0"

RoiSource = Literal["manual", "heuristic"]
RoiKind = Literal["aabb"]


class RoiManifest(BaseModel):
    """On-disk ROI package under ``rois/<roi_id>/mask.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = ESCALATE_SCHEMA
    roi_id: str
    mesh_id: str
    kind: RoiKind = "aabb"
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    source: RoiSource = "manual"
    notes: list[str] = Field(default_factory=list)
    created_at: str = ""
    roi_ply: str | None = None
    # Relative path to mask.json from job root (optional convenience)
    mask_path: str | None = None


class HandoffManifest(BaseModel):
    """On-disk handoff package meta under ``handoff/meta.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = ESCALATE_SCHEMA
    mesh_id: str
    roi_id: str
    blender_path: str
    blender_version: str
    blend_path: str
    instructions_path: str
    created_at: str
    vertex_group: str = "meshops_roi"
    cameras: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    timeout_s: float = 300.0
    # Parsed from build_handoff.py stdout (roi_verts=N); None if unparsed
    roi_vert_count: int | None = None


class PreviewResult(BaseModel):
    """T3 preview outcome — never success-as-fixed (N6 / Difficulty §13)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ok: bool = False  # previews never claim fixed success
    preview: Literal[True] = True
    mesh_id: str
    roi_id: str | None = None
    preview_id: str
    preview_dir: Path
    notes: list[str] = Field(default_factory=list)
    honesty_note: str = (
        "preview_only — NOT fixed; not print-ready; handoff + human sculpt required (N6)"
    )
    paths: dict[str, str] = Field(default_factory=dict)
    # Explicit refuse flags for promote/export consumers
    may_promote_working: Literal[False] = False
    may_claim_fixed: Literal[False] = False


class ImportSculptResult(BaseModel):
    """Sculpt STL import as atomic rev + accept with sculpt policy."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ok: bool
    mesh_id: str
    rev_id: str | None = None
    rev_dir: str | None = None
    recipe_id: str = "blender_sculpt_import"
    acceptance: AcceptanceResult | None = None
    notes: list[str] = Field(default_factory=list)
    honesty_note: str = "mechanical sculpt import package only — not autonomous hero fixed (N6)"
    paths: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
