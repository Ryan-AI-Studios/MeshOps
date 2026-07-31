"""MeshOps T3 escalation — ROI, preview, Blender 5.2 handoff, sculpt import (0004).

Public API prepares the operating room; it does not claim autonomous hero fixed (N6).
"""

from __future__ import annotations

from meshops.escalate.discover import find_blender
from meshops.escalate.errors import EscalateError
from meshops.escalate.handoff import build_handoff
from meshops.escalate.import_sculpt import import_sculpt
from meshops.escalate.models import HandoffManifest, ImportSculptResult, PreviewResult, RoiManifest
from meshops.escalate.preview_t3 import preview_t3
from meshops.escalate.roi import create_roi_bbox

__all__ = [
    "EscalateError",
    "HandoffManifest",
    "ImportSculptResult",
    "PreviewResult",
    "RoiManifest",
    "build_handoff",
    "create_roi_bbox",
    "find_blender",
    "import_sculpt",
    "preview_t3",
]
