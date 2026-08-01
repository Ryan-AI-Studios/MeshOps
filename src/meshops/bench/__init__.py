"""MeshOps bench harness — size ladder, envelope, optional accelerator probes.

Read-only metric surface (no GuardPolicy / --approve). Core path never requires
Open3D, Rust, or psutil (standing rule 8 / C1).
"""

from __future__ import annotations

from meshops.bench.models import (
    SCHEMA_VERSION,
    SKIPPED_INSUFFICIENT_RAM,
    BenchCaseResult,
    Envelope,
    HostBlock,
    MethodBlock,
)
from meshops.bench.report import (
    envelope_to_markdown,
    find_latest_results,
    load_envelope,
    resolve_work_root,
    write_results,
)
from meshops.bench.rss import get_available_ram_bytes, get_peak_rss_mb
from meshops.bench.runner import collect_deps, profile_load_vs_ingest, run_ladder, time_median
from meshops.bench.sizes import FACE_TARGETS, FACE_TOLERANCE_FRAC, generate_ladder_mesh, parse_sizes

__all__ = [
    "FACE_TARGETS",
    "FACE_TOLERANCE_FRAC",
    "SCHEMA_VERSION",
    "SKIPPED_INSUFFICIENT_RAM",
    "BenchCaseResult",
    "Envelope",
    "HostBlock",
    "MethodBlock",
    "collect_deps",
    "envelope_to_markdown",
    "find_latest_results",
    "generate_ladder_mesh",
    "get_available_ram_bytes",
    "get_peak_rss_mb",
    "load_envelope",
    "parse_sizes",
    "profile_load_vs_ingest",
    "resolve_work_root",
    "run_ladder",
    "time_median",
    "write_results",
]
