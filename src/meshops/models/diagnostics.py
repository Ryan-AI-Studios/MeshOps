"""Typed diagnostics models (schema_version 1.0.0).

Difficulty §1 — laterality uncertainty via needs_user_input (never stdin).
Difficulty §2 — sheet/ribbon is a first-class hypothesis via sheet_score.
Difficulty §7 / N8 — auto_action never delete.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"


class DefectClass(StrEnum):
    """Primary defect hypothesis labels T1-T5 (MeshOps taxonomy; detect-only in 0001).

    Matches docs/MeshOps.md section 2:
    T1 Topology | T2 Printability | T3 Sheet/ribbon | T4 Missing volume | T5 Mechanical feature
    """

    T1_TOPOLOGY = "T1_topology"
    T2_PRINTABILITY = "T2_printability"
    T3_SHEET = "T3_sheet"
    T4_MISSING_VOLUME = "T4_missing_volume"
    T5_MECHANICAL = "T5_mechanical"


class AutoAction(StrEnum):
    """Recommended follow-up - never auto-delete (N8 / Difficulty section 7)."""

    NONE = "none"
    REVIEW = "review"
    ESCALATE = "escalate"
    # Explicitly no "delete" value in 0001.


class LateralityStatus(StrEnum):
    """Anatomical laterality of multi-figure defects (Difficulty section 1)."""

    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"


class MeshStats(BaseModel):
    """Non-mutating mesh statistics and best-effort topology signals."""

    model_config = ConfigDict(extra="forbid")

    faces: int
    vertices: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_diagonal: float
    components: int
    is_watertight: bool | None = None
    is_volume: bool | None = None
    is_manifold: bool | None = None
    non_manifold_edge_count: int | None = None
    boundary_edge_count: int | None = None
    euler_characteristic: int | None = None
    file_size_bytes: int
    content_sha256: str
    mesh_id: str
    source_path: str | None = None
    topology_notes: list[str] = Field(default_factory=list)


class SheetScoreFeatures(BaseModel):
    """Multi-feature breakdown for sheet discrimination (Difficulty §2, §7)."""

    model_config = ConfigDict(extra="forbid")

    thinness_mean: float = 0.0
    thinness_p95: float = 0.0
    candidate_fraction: float = 0.0
    planarity: float = 0.0
    section_thinness: float = 0.0
    dihedral_crease: float = 0.0
    normal_smoothness: float = 0.0
    clothing_penalty: float = 0.0
    neighborhood_k: float = 0.02
    neighborhood_radius: float = 0.0
    n_samples: int = 0
    n_candidates: int = 0
    stage2_used: bool = False


class SheetScoreResult(BaseModel):
    """Sheet/ribbon score ∈ [0,1] with features and non-destructive auto_action."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    features: SheetScoreFeatures = Field(default_factory=SheetScoreFeatures)
    auto_action: AutoAction = AutoAction.NONE
    notes: list[str] = Field(default_factory=list)


class DefectHypothesis(BaseModel):
    """A single defect hypothesis with confidence."""

    model_config = ConfigDict(extra="forbid")

    defect_class: DefectClass
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class Diagnostics(BaseModel):
    """Root triage payload written to diagnostics.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    mesh_id: str
    stats: MeshStats
    defect_hypotheses: list[DefectHypothesis] = Field(default_factory=list)
    sheet_score: SheetScoreResult
    laterality_status: LateralityStatus = LateralityStatus.NOT_APPLICABLE
    needs_user_input: bool = False
    rendered_from: str | None = None
    view_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
