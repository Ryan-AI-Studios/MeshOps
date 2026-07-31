"""Ingest original.stl integrity on re-use (Codex 0003 P2)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from meshops.ingest.pipeline import ingest_stl
from meshops.jobstore.paths import JobPaths


def _clear_readonly(path: Path) -> None:
    """Clear POSIX write bits and Windows FILE_ATTRIBUTE_READONLY."""
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IWRITE | stat.S_IWUSR)
    if os.name == "nt":
        import ctypes

        # FILE_ATTRIBUTE_NORMAL = 0x80 clears readonly among others for this path.
        ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x80)  # type: ignore[attr-defined]


def test_ingest__reuse_mismatch_raises(tmp_path: Path, solid_cylinder_stl: Path) -> None:
    """If original.stl bytes differ from source under same mesh_id path, refuse."""
    work = tmp_path / "work"
    ing = ingest_stl(solid_cylinder_stl, work_root=work)
    paths = JobPaths(work_root=work, mesh_id=ing.mesh_id)
    # Corrupt canonical original while keeping same path/mesh_id layout.
    _clear_readonly(paths.original_stl)
    paths.original_stl.write_bytes(b"corrupted-not-stl-bytes-xxxxxx")
    with pytest.raises(ValueError, match="integrity"):
        # Re-ingest same source (same mesh_id) against corrupted original
        ingest_stl(solid_cylinder_stl, work_root=work)
