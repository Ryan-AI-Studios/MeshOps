"""Slice oracle models (0005-owned schema_version 1.0.0).

Independent of 0011 AcceptanceResult freeze — bump only when SliceRunResult shape changes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.acceptance.models import SliceAcceptResult

SLICE_SCHEMA: Literal["1.0.0"] = "1.0.0"

SliceRunStatus = Literal["pass", "fail", "error"]
ParseSource = Literal["slice_info", "gcode_comments", "failed"]


class ProfilePaths(BaseModel):
    """Resolved absolute profile triad for Orca CLI."""

    model_config = ConfigDict(extra="forbid")

    machine: str
    process: str
    filament: str
    profile_name: str = "default"
    datadir: str | None = None


class FilamentSlot(BaseModel):
    """Per-slot filament usage from slice_info attributes."""

    model_config = ConfigDict(extra="forbid")

    id: str
    used_g: float = 0.0
    used_m: float = 0.0
    type: str | None = None
    color: str | None = None


class SliceWarning(BaseModel):
    """Parsed plate warning (level int; error_code optional)."""

    model_config = ConfigDict(extra="forbid")

    msg: str = ""
    level: int = 0
    error_code: str | None = None


class ParsedSliceStats(BaseModel):
    """Internal dual-source parse product (not the public accept schema)."""

    model_config = ConfigDict(extra="forbid")

    parse_source: ParseSource = "failed"
    orca_version: str | None = None
    plate_count: int = 0
    plate_index: int | None = None
    print_time_s: float | None = None
    weight_g: float | None = None
    filament_used_g: float = 0.0
    filament_used_m: float = 0.0
    filament_used_cm3: float | None = None
    bed_overflow: bool = False
    support_used: bool = False
    filaments: list[FilamentSlot] = Field(default_factory=list)
    warnings: list[SliceWarning] = Field(default_factory=list)
    warning_max_level: int = 0
    messages: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class SliceRunResult(BaseModel):
    """Full oracle run outcome under work/<mesh_id>/slice/<run_id>/."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SLICE_SCHEMA
    run_id: str
    status: SliceRunStatus
    mesh_id: str | None = None
    candidate_path: str
    output_3mf: str | None = None
    run_dir: str | None = None
    profile_paths: ProfilePaths | None = None
    orca_path: str | None = None
    orca_version: str | None = None
    plate_count: int = 0
    accept: SliceAcceptResult | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    messages: list[str] = Field(default_factory=list)
    error_code: str | None = None
    started_at: str = ""
    finished_at: str = ""
    report_path: str | None = None
    argv: list[str] = Field(default_factory=list)
    returncode: int | None = None
