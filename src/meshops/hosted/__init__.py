"""Hosted multi-view generator fallback (track 0007) — post-plateau only.

Never the default organic path. Gate first always.
"""

from __future__ import annotations

from meshops.hosted.errors import HostedError, HostedErrorCode
from meshops.hosted.gate import assert_hosted_fallback_allowed, validate_plateau_gate
from meshops.hosted.honesty import HOSTED_HONESTY
from meshops.hosted.models import (
    HOSTED_SCHEMA_VERSION,
    HostedRunResult,
    Justification,
    ProviderJobStatus,
)
from meshops.hosted.orchestrate import run_hosted_fallback

__all__ = [
    "HOSTED_HONESTY",
    "HOSTED_SCHEMA_VERSION",
    "HostedError",
    "HostedErrorCode",
    "HostedRunResult",
    "Justification",
    "ProviderJobStatus",
    "assert_hosted_fallback_allowed",
    "run_hosted_fallback",
    "validate_plateau_gate",
]
