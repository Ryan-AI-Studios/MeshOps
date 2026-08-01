"""Organic session models — schema_version \"1.0.0\" is **0006-owned**.

Independence (B10): OrganicManifest / PassResult / PlateauRecord schema_version
strings do **not** share versioning, freeze rules, or compatibility with:

- AcceptanceResult.schema_version (0011)
- SliceAcceptResult / SliceRunResult.schema_version (0005 / 0011 hook)
- DesignManifest.schema_version (0003)

Bump organic schemas only when 0006 contracts change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.acceptance.models import AcceptanceResult

ORGANIC_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

SessionStatus = Literal["active", "plateau", "finalized", "failed"]
ViewKindOrganic = Literal["f3d", "stub"]

# Honesty token seeded on create (N6 / C6).
HONESTY_NOTE = "authored_organic_not_print_hero"

REQUIRED_VIEW_KEYS: tuple[str, ...] = (
    "front",
    "left",
    "three_quarter",
    "three_quarter_depth",
)


class OrganicManifest(BaseModel):
    """On-disk session metadata under organic/manifest.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = ORGANIC_SCHEMA_VERSION
    session_id: str
    prompt: str
    style_notes: str = ""
    ref_paths: list[str] = Field(default_factory=list)
    default_recipe: str = "simple_bust"
    status: SessionStatus = "active"
    passes: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    blender_version: str | None = None
    final_mesh_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    max_passes: int = 8


class PassResult(BaseModel):
    """Outcome of one organic pass (also written as pass.json)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    schema_version: Literal["1.0.0"] = ORGANIC_SCHEMA_VERSION
    ok: bool
    pass_id: str
    recipe: str
    mesh_path: Path | None = None
    view_paths: dict[str, str] = Field(default_factory=dict)
    view_kind: ViewKindOrganic | None = None
    blender_version: str | None = None
    returncode: int | None = None
    duration_s: float | None = None
    error_code: str | None = None
    messages: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    scale_mm: float | None = None


class PlateauRecord(BaseModel):
    """Machine-readable plateau for 0007 hosted fallback gate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = ORGANIC_SCHEMA_VERSION
    session_id: str
    reason: str
    pass_count: int
    max_passes: int = 8
    criteria_met: list[str] = Field(default_factory=list)
    created_at: str = ""
    allows_hosted_fallback: bool = False


class FinalizeResult(BaseModel):
    """Outcome of finalize_session."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ok: bool
    session_id: str
    mesh_id: str | None = None
    job_dir: Path | None = None
    triage_summary: dict[str, Any] | None = None
    acceptance: AcceptanceResult | None = None
    honesty_message: str | None = None
    error_code: str | None = None
    messages: list[str] = Field(default_factory=list)
