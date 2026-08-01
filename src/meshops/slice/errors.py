"""Structured slice / Orca oracle failures (track 0005)."""

from __future__ import annotations

from typing import Any, Literal

SliceErrorCode = Literal[
    "orca_not_found",
    "slice_timeout",
    "slice_failed",
    "missing_3mf",
    "parse_failed",
    "filament_zero",
    "bed_overflow",
    "filament_anomaly_high",
    "filament_anomaly_low",
    "unsliceable_geometry",
    "slice_warning_error",
    "profile_not_found",
    "missing_candidate",
    "job_not_found",
]


class SliceError(RuntimeError):
    """Fail-closed slicing oracle error with stable machine code."""

    def __init__(
        self,
        message: str,
        *,
        code: SliceErrorCode | str = "slice_failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
