"""Guard result models (export / recipe acceptance).

Difficulty §6 — multi-signal wipeout hard-fail.
Difficulty §5 — bbox / origin drift after transforms.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"


class GuardResult(BaseModel):
    """Outcome of check_export — never success when ok=False."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    ok: bool
    failed: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    messages: list[str] = Field(default_factory=list)
    policy_tier: str | None = None
