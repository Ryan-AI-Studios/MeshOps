"""Structured design failures (track 0003)."""

from __future__ import annotations

from typing import Any, Literal

DesignErrorCode = Literal[
    "cad_kernel_failure",
    "runner_crash",
    "ast_denied",
    "validation_failed",
    "multi_solid",
    "missing_result",
    "export_failed",
    "missing_dependency",
    "unreasonable_cad_scale",
    "timeout",
    "template_error",
    "hash_mismatch",
    "unknown_template",
    "ingest_failed",
]


class DesignError(RuntimeError):
    """Fail-closed design pipeline error with stable machine code."""

    def __init__(
        self,
        message: str,
        *,
        code: DesignErrorCode | str = "runner_crash",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
