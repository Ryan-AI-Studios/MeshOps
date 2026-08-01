"""Structured hosted-generator failures (track 0007)."""

from __future__ import annotations

from typing import Any, Literal

HostedErrorCode = Literal[
    "plateau_missing",
    "plateau_invalid",
    "plateau_gate_closed",
    "multiview_required",
    "justify_invalid",
    "api_key_missing",
    "provider_http",
    "provider_timeout",
    "provider_failed",
    "download_failed",
    "convert_failed",
    "ingest_failed",
]


class HostedError(RuntimeError):
    """Fail-closed hosted fallback error with stable machine code."""

    def __init__(
        self,
        message: str,
        *,
        code: HostedErrorCode | str = "provider_failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
