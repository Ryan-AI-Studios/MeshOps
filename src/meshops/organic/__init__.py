"""Organic design agent-first (T6 primary / track 0006).

Harness-driven Blender metaball recipes + F3D pass evidence + plateau + untrusted ingest.
Hosted multi-view fallback is meshops.hosted after plateau gate (track 0007).
"""

from __future__ import annotations

from meshops.organic.errors import OrganicError, OrganicErrorCode
from meshops.organic.finalize import finalize_session
from meshops.organic.models import (
    ORGANIC_SCHEMA_VERSION,
    FinalizeResult,
    OrganicManifest,
    PassResult,
    PlateauRecord,
)
from meshops.organic.pass_runner import run_pass
from meshops.organic.paths import SessionPaths
from meshops.organic.plateau import mark_plateau
from meshops.organic.session import create_session, load_session, save_manifest

__all__ = [
    "ORGANIC_SCHEMA_VERSION",
    "FinalizeResult",
    "OrganicError",
    "OrganicErrorCode",
    "OrganicManifest",
    "PassResult",
    "PlateauRecord",
    "SessionPaths",
    "create_session",
    "finalize_session",
    "load_session",
    "mark_plateau",
    "run_pass",
    "save_manifest",
]
