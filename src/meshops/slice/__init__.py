"""Slicing printability oracle — OrcaSlicer subprocess only (track 0005).

Public API: ``find_orca``, ``run_slice``, ``make_orca_hook``, ``SliceRunResult``.
Never mutates mesh geometry. Slice pass ≠ artistic fixed (N6 / Difficulty §13).
"""

from __future__ import annotations

from meshops.slice.discover import find_orca, read_orca_version_from_appdata
from meshops.slice.errors import SliceError
from meshops.slice.hook import make_orca_hook
from meshops.slice.models import ProfilePaths, SliceRunResult
from meshops.slice.runner import build_orca_argv, make_run_id, run_slice

__all__ = [
    "ProfilePaths",
    "SliceError",
    "SliceRunResult",
    "build_orca_argv",
    "find_orca",
    "make_orca_hook",
    "make_run_id",
    "read_orca_version_from_appdata",
    "run_slice",
]
