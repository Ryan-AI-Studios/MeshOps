"""Pixel proportion analysis (track 0012) — multi-view landmarks + head-unit checks.

Offline, assist-first. Not mesh reconstruction or print success.
"""

from __future__ import annotations

from meshops.proportion.honesty import PROPORTION_HONESTY
from meshops.proportion.models import PROPORTION_SCHEMA_VERSION, ProportionReport

__all__ = [
    "PROPORTION_HONESTY",
    "PROPORTION_SCHEMA_VERSION",
    "ProportionReport",
]
