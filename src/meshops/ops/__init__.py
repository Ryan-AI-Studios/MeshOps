"""Ops surface: doctor, env catalog, Blender mirrors (track 0010)."""

from __future__ import annotations

from meshops.ops.doctor import expand_require, run_doctor
from meshops.ops.env_catalog import ENV_CATALOG, catalog_names
from meshops.ops.mirrors import (
    BLENDER_MIRROR_URLS,
    BLENDER_WINDOWS_X64_SHA256,
    BLENDER_ZIP_NAME,
)
from meshops.ops.models import SCHEMA_VERSION, DoctorReport

__all__ = [
    "BLENDER_MIRROR_URLS",
    "BLENDER_WINDOWS_X64_SHA256",
    "BLENDER_ZIP_NAME",
    "ENV_CATALOG",
    "SCHEMA_VERSION",
    "DoctorReport",
    "catalog_names",
    "expand_require",
    "run_doctor",
]
