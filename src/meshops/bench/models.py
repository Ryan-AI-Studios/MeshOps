"""Benchmark envelope schema (schema_version 1.0.0).

Binding fields from conductor/0009-Hardening/spec.md §3.1.1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

CaseStatus = Literal["ok", "skipped", "failed"]

# Documented skip reason for RAM gate (R2).
SKIPPED_INSUFFICIENT_RAM: Literal["skipped_insufficient_ram"] = "skipped_insufficient_ram"


class BenchCaseResult(BaseModel):
    """One size-ladder case (S/M/L/XL or custom label)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    label: str
    target_faces: int
    actual_faces: int
    verts: int
    ingest_s: float | None = None
    triage_s: float | None = None
    render_s: float | None = None
    ingest_samples_s: list[float] = Field(default_factory=list)
    triage_samples_s: list[float] = Field(default_factory=list)
    render_samples_s: list[float] = Field(default_factory=list)
    rss_peak_mb: float | None = None
    status: CaseStatus = "ok"
    skipped_reason: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    host_os: str
    python_version: str
    deps: dict[str, str] = Field(default_factory=dict)
    created_at: str


class MethodBlock(BaseModel):
    """Timing method metadata (warmup + median-of-N)."""

    model_config = ConfigDict(extra="forbid")

    warmup: int = 1
    timed_iters: int = 3
    aggregate: Literal["median"] = "median"
    gc_collect_before_timed: bool = True
    cameras: list[str] = Field(default_factory=lambda: ["front", "left", "three_quarter"])
    face_tolerance_frac: float = 0.15
    ram_gate_bytes: int = 4 * 1024**3
    notes: str = (
        "gc.collect before each timed series; 1 untimed warmup; median of 3 timed iters. "
        "Ingest/triage samples are warm re-ingest after one seed write (idempotent job dir); "
        "not cold first-touch wall time — see profile_load_vs_ingest for cold load ratio. "
        "rss_peak_mb prefers OS peak counters (Win PeakWorkingSet / Unix ru_maxrss); "
        "psutil current RSS only as last-resort fallback. "
        "F3D RenderUnavailableError → render_s null (case may still be ok). "
        "L/XL skip when available RAM < ~4 GiB → skipped_insufficient_ram."
    )


class HostBlock(BaseModel):
    """Host environment captured at envelope build time."""

    model_config = ConfigDict(extra="forbid")

    os: str
    python_version: str
    cpu: str | None = None
    total_ram_mb: float | None = None
    available_ram_mb: float | None = None
    deps: dict[str, str] = Field(default_factory=dict)


class Envelope(BaseModel):
    """Full bench envelope: method + host + cases."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    created_at: str
    method: MethodBlock = Field(default_factory=MethodBlock)
    host: HostBlock
    cases: list[BenchCaseResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
