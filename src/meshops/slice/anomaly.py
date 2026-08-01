"""Two-sided filament anomaly + hard-fail printability mapping (track 0005).

Difficulty N6 / §6: slice pass ≠ artistic fixed. PLA density 1.24 is approximate.
Non-watertight meshes skip ratio (trimesh ``.volume`` is 0.0, not trustworthy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trimesh

from meshops.acceptance.models import SliceAcceptResult
from meshops.slice.models import ParsedSliceStats

# Default bands (spec §3.6)
DEFAULT_LOW_RATIO = 0.05
DEFAULT_WARN_RATIO = 3.0
DEFAULT_FAIL_HIGH = 8.0

REORIENT_SUGGESTION = (
    "Filament usage high vs mesh volume (ratio={ratio:.3f}). "
    "Consider re-slice with --orient 1 or lower support settings."
)


@dataclass(frozen=True, slots=True)
class AnomalyThresholds:
    """Two-sided filament ratio bands."""

    low_ratio: float = DEFAULT_LOW_RATIO
    warn_ratio: float = DEFAULT_WARN_RATIO
    fail_high: float = DEFAULT_FAIL_HIGH
    # When True, warn band still passes; only fail_high/low hard-fail.
    warn_is_pass: bool = True


def mesh_volume_cm3_from_path(path: Path | str) -> float | None:
    """Load mesh; return volume cm³ only when watertight/volume is trustworthy.

    Non-watertight → None (never treat trimesh 0.0 volume as real).
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        loaded = trimesh.load(str(p), force="mesh")
    except Exception:
        return None
    if not isinstance(loaded, trimesh.Trimesh):
        return None
    return mesh_volume_cm3_from_mesh(loaded)


def mesh_volume_cm3_from_mesh(mesh: trimesh.Trimesh) -> float | None:
    """Volume cm³ when ``is_watertight`` (or ``is_volume``) — else None."""
    is_wt = bool(getattr(mesh, "is_watertight", False))
    is_vol = bool(getattr(mesh, "is_volume", False)) if hasattr(mesh, "is_volume") else False
    if not (is_wt or is_vol):
        return None
    try:
        vol_mm3 = float(mesh.volume)
    except Exception:
        return None
    if vol_mm3 <= 0.0:
        return None
    return vol_mm3 / 1000.0


def evaluate_printability(
    stats: ParsedSliceStats,
    mesh_volume_cm3: float | None = None,
    *,
    thresholds: AnomalyThresholds | None = None,
    subprocess_ok: bool = True,
    missing_3mf: bool = False,
) -> SliceAcceptResult:
    """Map parse stats + optional mesh volume → SliceAcceptResult.

    Hard fail:
      - missing 3mf / parse failed / subprocess failure
      - bed_overflow
      - filament zero
      - warning level >= 2
      - anomaly high (>= fail_high) / low (<= low_ratio) when volume available
    """
    thr = thresholds or AnomalyThresholds()
    messages: list[str] = list(stats.messages)
    metrics: dict[str, Any] = dict(stats.metrics)
    metrics["slice.mesh_volume_available"] = mesh_volume_cm3 is not None and mesh_volume_cm3 > 0.0
    if mesh_volume_cm3 is not None and mesh_volume_cm3 > 0.0:
        metrics["slice.mesh_volume_cm3"] = mesh_volume_cm3

    filament_cm3 = stats.filament_used_cm3
    bed = stats.bed_overflow
    print_time = stats.print_time_s

    # Surface warnings in messages
    for w in stats.warnings:
        tag = f"slice warning L{w.level}"
        if w.error_code:
            tag += f" [{w.error_code}]"
        if w.msg:
            tag += f": {w.msg}"
        messages.append(tag)

    metrics["slice.warning_max_level"] = stats.warning_max_level
    metrics["slice.bed_overflow"] = bed
    metrics["slice.support_used"] = stats.support_used
    metrics["slice.plate_count"] = stats.plate_count
    if stats.orca_version:
        metrics["slice.orca_version"] = stats.orca_version

    def _fail(code: str, msg: str) -> SliceAcceptResult:
        messages.append(msg)
        return SliceAcceptResult(
            status="fail",
            filament_used_cm3=filament_cm3,
            print_time_s=print_time,
            bed_overflow=bed,
            error_code=code,
            messages=messages,
            metrics=metrics,
        )

    if missing_3mf:
        return _fail("missing_3mf", "Orca did not produce a non-empty .gcode.3mf")

    if not subprocess_ok:
        return _fail("slice_failed", "Orca subprocess failed")

    if stats.parse_source == "failed":
        return _fail("parse_failed", "slice parse failed (no usable slice_info or gcode stats)")

    if bed:
        return _fail("bed_overflow", "model outside build plate (outside=true)")

    # Zero filament: cm3 == 0.0 or both g and m zero
    zero_cm3 = filament_cm3 is not None and filament_cm3 == 0.0
    zero_raw = stats.filament_used_g == 0.0 and stats.filament_used_m == 0.0
    if zero_cm3 or zero_raw:
        code = "filament_zero"
        # Unsliceable hint if warnings indicate empty
        if any(
            (w.error_code or "").lower().find("unslice") >= 0
            or "unsliceable" in (w.msg or "").lower()
            for w in stats.warnings
        ):
            code = "unsliceable_geometry"
        return _fail(code, "filament used is zero — empty or unsliceable geometry signal")

    if stats.warning_max_level >= 2:
        # Prefer warning error_code when present
        err = next(
            (w.error_code for w in stats.warnings if w.level >= 2 and w.error_code),
            "slice_warning_error",
        )
        lvl = stats.warning_max_level
        return _fail(err or "slice_warning_error", f"slice warning level {lvl} >= 2")

    # Anomaly ratio (only when volume trustworthy)
    ratio: float | None = None
    suggest_reorient = False
    if mesh_volume_cm3 is not None and mesh_volume_cm3 > 0.0 and filament_cm3 is not None:
        ratio = filament_cm3 / mesh_volume_cm3
        metrics["slice.filament_ratio"] = ratio
        if ratio >= thr.fail_high:
            suggest_reorient = True
            metrics["slice.suggest_reorient"] = True
            messages.append(REORIENT_SUGGESTION.format(ratio=ratio))
            return _fail(
                "filament_anomaly_high",
                f"filament/mesh volume ratio {ratio:.3f} >= {thr.fail_high}",
            )
        if ratio <= thr.low_ratio:
            metrics["slice.suggest_reorient"] = False
            return _fail(
                "filament_anomaly_low",
                f"filament/mesh volume ratio {ratio:.3f} <= {thr.low_ratio} "
                "(wall collapse / empty interior signal)",
            )
        if ratio > thr.warn_ratio:
            suggest_reorient = True
            metrics["slice.suggest_reorient"] = True
            messages.append(REORIENT_SUGGESTION.format(ratio=ratio))
            if not thr.warn_is_pass:
                return _fail(
                    "filament_anomaly_high",
                    f"filament/mesh volume ratio {ratio:.3f} > warn {thr.warn_ratio}",
                )
        else:
            metrics["slice.suggest_reorient"] = False
    else:
        metrics["slice.suggest_reorient"] = False
        metrics.setdefault("slice.mesh_volume_available", False)

    if suggest_reorient:
        metrics["slice.suggest_reorient"] = True

    messages.append("slice printability pass")
    return SliceAcceptResult(
        status="pass",
        filament_used_cm3=filament_cm3,
        print_time_s=print_time,
        bed_overflow=False,
        error_code=None,
        messages=messages,
        metrics=metrics,
    )
