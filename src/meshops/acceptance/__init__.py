"""MeshOps acceptance pack — shared mutator/export validation moat (track 0011).

Composes ``meshops.guards.check_export`` (never forks wipeout). Honesty ceiling:
stubs → ``guards_and_stub_views``; mechanical only — never claim artistic sculpt fix.

schema_version ``1.0.0`` is frozen for 0011.
"""

from __future__ import annotations

from meshops.acceptance.honesty import HONESTY_MESSAGE
from meshops.acceptance.hooks import SliceAcceptHook, null_slice_result
from meshops.acceptance.models import AcceptanceResult, SliceAcceptResult
from meshops.acceptance.numeric import DEGENERATE_FACE_RATIO_MAX, count_degenerate_faces
from meshops.acceptance.pack import (
    accept_candidate,
    accept_revision,
    build_acceptance_from_guard,
)
from meshops.acceptance.promote import PromoteError, promote_working
from meshops.guards import GuardPolicy, GuardResult, check_export

__all__ = [
    "DEGENERATE_FACE_RATIO_MAX",
    "HONESTY_MESSAGE",
    "AcceptanceResult",
    "GuardPolicy",
    "GuardResult",
    "PromoteError",
    "SliceAcceptHook",
    "SliceAcceptResult",
    "accept_candidate",
    "accept_revision",
    "build_acceptance_from_guard",
    "check_export",
    "count_degenerate_faces",
    "null_slice_result",
    "promote_working",
]
