"""Acceptance pack result models (schema_version 1.0.0 frozen for 0011).

Composes 0002 GuardResult; never forks wipeout logic.
Difficulty §12 / N6 — honesty_message always set; no "limb fixed" success language.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.acceptance.honesty import HONESTY_MESSAGE
from meshops.guards.models import GuardResult

# Frozen for 0011 — bump requires a new track.
SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

HonestyKind = Literal[
    "guards_and_views",
    "guards_and_stub_views",
    "guards_only",
    "not_accepted",
]

ViewKind = Literal["f3d", "workbench", "stub", "mixed", "none"]

SliceStatus = Literal["pass", "fail", "skipped"]


class SliceAcceptResult(BaseModel):
    """Optional printability / slicer gate outcome (hook body is 0005)."""

    model_config = ConfigDict(extra="forbid")

    status: SliceStatus
    filament_used_cm3: float | None = None
    print_time_s: float | None = None
    bed_overflow: bool = False
    error_code: str | None = None
    messages: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class AcceptanceResult(BaseModel):
    """Shared mutator / export acceptance outcome — fail-closed when ok=False."""

    model_config = ConfigDict(extra="forbid")

    # schema_version frozen for 0011
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    ok: bool
    failed: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    guard: GuardResult | None = None
    view_paths: list[str] = Field(default_factory=list)
    views_ok: bool | None = None
    view_kind: ViewKind = "none"
    slice: SliceAcceptResult | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    policy_tier: str | None = None
    honesty: HonestyKind = "not_accepted"
    honesty_message: str = HONESTY_MESSAGE
