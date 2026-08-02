"""Pixel proportion analysis (track 0012+0013) — multi-view landmarks + widths/depths.

Offline, assist-first. Not mesh reconstruction or print success.
Schema write: 1.1.0; load accepts 1.0.0 (new fields default empty).
"""

from __future__ import annotations

from meshops.proportion.honesty import PROPORTION_HONESTY
from meshops.proportion.models import (
    PROPORTION_SCHEMA_VERSION,
    CrossSection,
    DepthBand,
    DiameterMeasure,
    ProportionReport,
)

__all__ = [
    "PROPORTION_HONESTY",
    "PROPORTION_SCHEMA_VERSION",
    "CrossSection",
    "DepthBand",
    "DiameterMeasure",
    "ProportionReport",
]
