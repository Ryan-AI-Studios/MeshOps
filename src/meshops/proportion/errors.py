"""Structured proportion analysis failures (track 0012)."""

from __future__ import annotations

from typing import Any, Literal

ProportionErrorCode = Literal[
    "missing_views",
    "unreadable_image",
    "invalid_assist",
    "pillow_required",
    "incomplete_package",
    "incomplete_stature",
    "invalid_report",
    "write_failed",
    "checklist_exists",
    "scaffold_failed",
    "invalid_checklist",
    "guides_empty",
    "guides_failed",
    "capture_empty",
    "capture_failed",
    "depth_empty",
    "depth_failed",
    "mesh_load_failed",
    "recipe_empty",
    "recipe_failed",
    "heatmap_empty",
    "heatmap_failed",
    "hint_empty",
    "hint_failed",
    "monocular_unavailable",
    "invalid_depth_samples",
    "invalid_depth_deltas",
    "silhouette_empty",
    "silhouette_failed",
    "silhouette_untrusted",
    "template_unknown",
    "template_empty",
    "template_failed",
    "optimize_slow_needs_mesh",
    "optimize_no_free_dofs",
    "optimize_failed",
    "constraint_report_failed",
    "skeleton_empty",
    "skeleton_failed",
    "profile_unknown",
    "unknown",
]


class ProportionError(RuntimeError):
    """Fail-closed proportion error with stable machine code."""

    def __init__(
        self,
        message: str,
        *,
        code: ProportionErrorCode | str = "unknown",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
