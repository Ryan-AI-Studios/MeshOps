"""Two-tier GuardPolicy — export wipeout floor vs recipe-tight acceptance.

Default constants (record in review.md if tuned):

Export tier:
  face_floor_ratio = 0.50
  size_floor_ratio = 0.40
Recipe tier (T1/T2):
  face_floor_ratio = 0.90
  size_floor_ratio = 0.80

Global wipeout floors (non-negotiable under either tier when hero-scale):
  hero_bytes_threshold = 10_000_000
  hero_faces_threshold = 200_000
  wipeout_bytes_out = 500_000
  wipeout_faces_out = 20_000
  face_collapse_out_ratio = 0.10  # >90% face loss
  component_collapse_drop = 0.75  # >75% component drop
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Default thresholds (Difficulty §5, §6) ---

EXPORT_FACE_FLOOR_RATIO = 0.50
EXPORT_SIZE_FLOOR_RATIO = 0.40
RECIPE_FACE_FLOOR_RATIO = 0.90
RECIPE_SIZE_FLOOR_RATIO = 0.80

HERO_BYTES_THRESHOLD = 10_000_000
HERO_FACES_THRESHOLD = 200_000
WIPEOUT_BYTES_OUT = 500_000
WIPEOUT_FACES_OUT = 20_000
FACE_COLLAPSE_OUT_RATIO = 0.10  # fail if faces_out < 0.10 * faces_in
COMPONENT_COLLAPSE_DROP = 0.75  # fail if drop fraction > 0.75

# Component explosion: max(components_in + k, 2 * components_in)
COMPONENT_GROWTH_K = 8
COMPONENT_GROWTH_FACTOR = 2.0

# Bbox / origin orphan (Difficulty §5)
ORIGIN_CENTROID_EPS = 1.0  # candidate |centroid| near origin (absolute units)
BASELINE_OFFSET_MIN = 10.0  # baseline |centroid| must exceed this to flag orphan
EXTENTS_COLLAPSE_RATIO = 0.05  # candidate diagonal < 5% of baseline
NEAR_ZERO_DIAGONAL = 1e-3

# Absolute size floor only applied when baseline is hero-scale
HERO_SIZE_ABS_FLOOR_BYTES = 500_000


class GuardPolicy(BaseModel):
    """Thresholds for check_export. Global wipeout floors cannot be loosened."""

    model_config = ConfigDict(extra="forbid")

    tier: Literal["export", "recipe"] = "export"
    recipe_id: str | None = None

    face_floor_ratio: float = EXPORT_FACE_FLOOR_RATIO
    size_floor_ratio: float = EXPORT_SIZE_FLOOR_RATIO
    size_abs_floor_bytes: int | None = None  # set for hero-class when applicable

    # Hero-scale wipeout (global, non-negotiable)
    hero_bytes_threshold: int = HERO_BYTES_THRESHOLD
    hero_faces_threshold: int = HERO_FACES_THRESHOLD
    wipeout_bytes_out: int = WIPEOUT_BYTES_OUT
    wipeout_faces_out: int = WIPEOUT_FACES_OUT
    face_collapse_out_ratio: float = FACE_COLLAPSE_OUT_RATIO
    component_collapse_drop: float = COMPONENT_COLLAPSE_DROP

    component_growth_k: int = COMPONENT_GROWTH_K
    component_growth_factor: float = COMPONENT_GROWTH_FACTOR

    origin_centroid_eps: float = ORIGIN_CENTROID_EPS
    baseline_offset_min: float = BASELINE_OFFSET_MIN
    extents_collapse_ratio: float = EXTENTS_COLLAPSE_RATIO
    near_zero_diagonal: float = NEAR_ZERO_DIAGONAL

    allow_component_growth: bool = True
    check_volume: bool = True
    volume_near_zero: float = 1e-9

    # Recipe policies still subject to global wipeout floors
    enforce_global_wipeout: bool = True

    notes: list[str] = Field(default_factory=list)

    @classmethod
    def for_export(cls) -> GuardPolicy:
        """User-facing export / hard safety floor."""
        return cls(
            tier="export",
            face_floor_ratio=EXPORT_FACE_FLOOR_RATIO,
            size_floor_ratio=EXPORT_SIZE_FLOOR_RATIO,
            size_abs_floor_bytes=None,  # applied dynamically for hero baselines
            notes=["export_tier"],
        )

    @classmethod
    def for_recipe(cls, recipe_id: str) -> GuardPolicy:
        """Post-recipe acceptance — tighter mass retention; wipeout floors still bind."""
        return cls(
            tier="recipe",
            recipe_id=recipe_id,
            face_floor_ratio=RECIPE_FACE_FLOOR_RATIO,
            size_floor_ratio=RECIPE_SIZE_FLOOR_RATIO,
            size_abs_floor_bytes=None,
            notes=[f"recipe_tier:{recipe_id}"],
        )
