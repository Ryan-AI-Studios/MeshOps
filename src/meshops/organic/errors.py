"""Structured organic session failures (track 0006)."""

from __future__ import annotations

from typing import Any, Literal

OrganicErrorCode = Literal[
    "blender_not_found",
    "blender_version",
    "blender_failed",
    "blender_timeout",
    "session_not_found",
    "session_finalized",
    "invalid_params",
    "recipe_unknown",
    "recipe_refused",
    "pass_no_mesh",
    "pass_no_views",
    "plateau_reason_required",
    "plateau_reason_weak",
    "finalize_no_pass",
    "ingest_failed",
    "max_passes_exceeded",
]


class OrganicError(RuntimeError):
    """Fail-closed organic authoring error with stable machine code."""

    def __init__(
        self,
        message: str,
        *,
        code: OrganicErrorCode | str = "blender_failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
