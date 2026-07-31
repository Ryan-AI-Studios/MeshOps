"""MeshOps T7 mechanical design-from-code (track 0003).

Public API: harness-wrapper geometry run, templates, validate, orchestrate.
Primary design gate = validate.py absolute floors; accept_candidate self-baseline
is a hero wipeout safety net only. No design rev history in v1.
"""

from __future__ import annotations

from meshops.design.errors import DesignError
from meshops.design.models import BracketParams, DesignManifest, DesignResult
from meshops.design.orchestrate import design_from_template, run_design_pipeline
from meshops.design.runner import run_geometry_source

__all__ = [
    "BracketParams",
    "DesignError",
    "DesignManifest",
    "DesignResult",
    "design_from_template",
    "run_design_pipeline",
    "run_geometry_source",
]
