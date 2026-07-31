"""Structured escalation failures (track 0004)."""

from __future__ import annotations

from typing import Any, Literal

EscalateErrorCode = Literal[
    "blender_missing",
    "blender_failed",
    "blender_version",
    "approve_required",
    "job_not_found",
    "roi_not_found",
    "preview_refuse_promote",
    "missing_mesh",
    "timeout",
    "invalid_bbox",
    "import_failed",
    "wipeout_refuse",
]


class EscalateError(RuntimeError):
    """Fail-closed escalation / handoff error with stable machine code."""

    def __init__(
        self,
        message: str,
        *,
        code: EscalateErrorCode | str = "blender_failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
