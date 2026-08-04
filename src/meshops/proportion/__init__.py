"""Pixel proportion analysis (track 0012+0013+0030) — multi-view landmarks + widths/depths.

Offline, assist-first. Not mesh reconstruction or print success.
Schema write: 1.2.0; load accepts 1.0.0|1.1.0|1.2.0 (new fields default null/empty).
"""

from __future__ import annotations

from meshops.proportion.honesty import PROPORTION_HONESTY
from meshops.proportion.models import (
    PROPORTION_SCHEMA_VERSION,
    BreastMetrics,
    CrossSection,
    DepthBand,
    DiameterMeasure,
    ProportionReport,
    SoftSpacing,
)

__all__ = [
    "PROPORTION_HONESTY",
    "PROPORTION_SCHEMA_VERSION",
    "BreastMetrics",
    "CrossSection",
    "DepthBand",
    "DiameterMeasure",
    "ProportionReport",
    "SoftSpacing",
]
