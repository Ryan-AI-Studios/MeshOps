"""Design pipeline models — manifest, result, bracket params (track 0003)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from meshops.acceptance.models import AcceptanceResult

DESIGN_MANIFEST_SCHEMA: Literal["1.0.0"] = "1.0.0"

# Default export kwargs recorded in DesignManifest (build123d 0.11.1).
DEFAULT_EXPORT_STL_KWARGS: dict[str, Any] = {
    "tolerance": 0.001,
    "angular_tolerance": 0.1,
    "ascii_format": False,
}
DEFAULT_EXPORT_STEP_KWARGS: dict[str, Any] = {
    "unit": "MM",
    "write_pcurves": True,
    "precision_mode": "AVERAGE",
}


class BracketParams(BaseModel):
    """Parametric M4-class mounting bracket (algebra template).

    Units: millimetres only.
    Clearance rule: hole_spacing_mm > hole_diameter_mm + 2 * wall_mm.
    """

    model_config = ConfigDict(extra="forbid")

    hole_spacing_mm: float = Field(default=40.0, gt=10.0, lt=500.0)
    wall_mm: float = Field(default=3.0, gt=1.0, lt=50.0)
    thickness_mm: float = Field(default=4.0, gt=1.0, lt=50.0)
    hole_diameter_mm: float = Field(default=4.2, gt=1.0, lt=20.0)

    @model_validator(mode="after")
    def _clearance(self) -> BracketParams:
        min_spacing = self.hole_diameter_mm + 2.0 * self.wall_mm
        if self.hole_spacing_mm <= min_spacing:
            raise ValueError(
                f"hole_spacing_mm ({self.hole_spacing_mm}) must be > "
                f"hole_diameter_mm + 2*wall_mm ({min_spacing})"
            )
        return self

    def plate_extents_mm(self) -> tuple[float, float, float]:
        """Expected axis-aligned plate size (X length, Y width, Z thickness)."""
        length_x = self.hole_spacing_mm + self.hole_diameter_mm + 2.0 * self.wall_mm
        width_y = self.hole_diameter_mm + 2.0 * self.wall_mm
        return (length_x, width_y, self.thickness_mm)


class DesignManifest(BaseModel):
    """On-disk design job metadata under design/manifest.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = DESIGN_MANIFEST_SCHEMA
    template_id: str | None = None
    source: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    units: Literal["mm"] = "mm"
    export_stl: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_EXPORT_STL_KWARGS))
    export_step: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_EXPORT_STEP_KWARGS))
    content_sha256: str = ""
    notes: list[str] = Field(default_factory=list)
    runner: dict[str, Any] = Field(default_factory=dict)


class DesignResult(BaseModel):
    """Outcome of design_from_template / run_geometry_source pipeline."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ok: bool
    mesh_id: str
    job_dir: Path
    paths: dict[str, str] = Field(default_factory=dict)
    acceptance: AcceptanceResult | None = None
    manifest: DesignManifest
    notes: list[str] = Field(default_factory=list)
