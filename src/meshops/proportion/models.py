"""Proportion report models — schema_version "1.0.0" is **0012-owned**.

Independence: ProportionReport schema_version does **not** share versioning,
freeze rules, or compatibility with:

- OrganicManifest / PassResult / PlateauRecord (0006)
- AcceptanceResult.schema_version (0011)
- SliceAcceptResult / SliceRunResult.schema_version (0005 / 0011 hook)
- DesignManifest.schema_version (0003)
- HostedReport schema (0007)

Bump proportion schemas only when 0012 contracts change.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.honesty import PROPORTION_HONESTY

PROPORTION_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

ViewKey = Literal["front", "left", "three_quarter", "back"]
PoseKind = Literal["a_pose", "hanging", "unknown", "t_pose", "other"]
FacingDirection = Literal[
    "camera_front",
    "camera_left",
    "camera_right",
    "camera_back",
    "unknown",
]
LandmarkMethod = Literal["assist", "heuristic_frame", "fixture_known", "pose_model"]
CheckSeverity = Literal["info", "warn", "error"]

REQUIRED_VIEW_KEYS: tuple[str, ...] = ("front", "left", "three_quarter")
CANONICAL_VIEW_KEYS: tuple[str, ...] = ("front", "left", "three_quarter", "back")
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")

# package_score weights (spec §3.1 F) — completeness only.
SCORE_VIEWS_TOTAL = 40.0
SCORE_PER_REQUIRED_VIEW = SCORE_VIEWS_TOTAL / 3.0  # ≈13.333
SCORE_STATURE = 25.0
SCORE_WIDTH_PAIR = 15.0
SCORE_DEPTH = 20.0


class Landmark2D(BaseModel):
    """Single 2D landmark in one view (px + image-normalized fracs)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    x_px: float
    y_px: float
    x_frac: float = Field(ge=0.0, le=1.0)
    y_frac: float = Field(ge=0.0, le=1.0)
    method: LandmarkMethod | str = "assist"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SubjectBBox(BaseModel):
    """Axis-aligned subject bounding box in pixels (top-left origin)."""

    model_config = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float


class ViewLandmarks(BaseModel):
    """Per-view landmarks and frame metadata."""

    model_config = ConfigDict(extra="forbid")

    view: str
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    path: str | None = None
    facing_direction: FacingDirection | str | None = None
    subject_bbox: SubjectBBox | None = None
    figure_span_px: float | None = None
    landmarks: dict[str, Landmark2D] = Field(default_factory=dict)
    large_blob_count: int | None = None


class LandmarkXYZ(BaseModel):
    """Fused blockout-grade landmark in figure-normalized XYZ (Z up, soles=0)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    x: float | None = None  # body lateral frac of figure_h; + = camera-right
    y: float | None = None  # depth frac; + toward camera when facing camera_left
    z: float | None = None  # stature frac; soles = 0
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)


class CheckResult(BaseModel):
    """One comparative-measurement check (flag, do not force-fit)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    ok: bool
    severity: CheckSeverity = "info"
    message: str
    measured: float | dict[str, Any] | None = None
    expected: float | dict[str, Any] | str | None = None


class QualityFlags(BaseModel):
    """Report-level quality / policy flags."""

    model_config = ConfigDict(extra="forbid")

    hair_volume_margin: bool = False
    foreshortening_risk: bool = False
    multi_figure: bool = False
    needs_user_input: bool = False
    incomplete_stature: bool = False
    partial_package: bool = False
    notes: list[str] = Field(default_factory=list)


class ProportionReport(BaseModel):
    """Versioned proportion contract for agents and Blender blockout."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = PROPORTION_SCHEMA_VERSION
    honesty: str = PROPORTION_HONESTY
    package_score: float = Field(default=0.0, ge=0.0, le=100.0)
    pose: PoseKind | str = "unknown"
    height_m: float | None = None
    head_unit_frac: float | None = None
    figure_height_frac: float | None = Field(
        default=None,
        description="Always 1.0 when stature known; kept for consumers",
    )
    vertical_span_discrepancy: float | None = None
    views: dict[str, ViewLandmarks] = Field(default_factory=dict)
    landmarks_xyz: dict[str, LandmarkXYZ] = Field(default_factory=dict)
    checks: list[CheckResult] = Field(default_factory=list)
    quality: QualityFlags = Field(default_factory=QualityFlags)
    messages: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
