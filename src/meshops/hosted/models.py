"""Hosted fallback models — schema_version "1.0.0" is **0007-owned**.

Independence: HostedRunResult.schema_version does not share versioning with
organic / acceptance / design schemas.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.acceptance.models import AcceptanceResult
from meshops.hosted.honesty import HOSTED_HONESTY

HOSTED_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

ProviderStatusName = Literal[
    "PENDING",
    "IN_PROGRESS",
    "SUCCEEDED",
    "FAILED",
    "CANCELED",
]


class Justification(BaseModel):
    """Plateau reason + operator --justify (both required for a successful run)."""

    model_config = ConfigDict(extra="forbid")

    plateau_reason: str
    operator_justify: str


class ProviderJobStatus(BaseModel):
    """Normalized provider poll status."""

    model_config = ConfigDict(extra="forbid")

    status: ProviderStatusName
    progress: float | None = None
    message: str | None = None
    task_error: dict[str, Any] | None = None
    model_urls: dict[str, str] | None = None
    thumbnail_urls: dict[str, str] | list[str] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class HostedRunResult(BaseModel):
    """Outcome of run_hosted_fallback."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    schema_version: Literal["1.0.0"] = HOSTED_SCHEMA_VERSION
    ok: bool
    session_id: str
    mesh_id: str | None = None
    job_dir: str | None = None
    provider: str
    provider_task_id: str | None = None
    justification: Justification | None = None
    view_paths: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] | None = None
    acceptance: AcceptanceResult | None = None
    honesty: str = HOSTED_HONESTY
    error_code: str | None = None
    messages: list[str] = Field(default_factory=list)
