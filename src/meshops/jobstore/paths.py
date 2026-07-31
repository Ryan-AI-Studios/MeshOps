"""Job directory layout and content-hash mesh_id helpers."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

# mesh_id = first 12 lowercase hex of SHA-256 of original bytes
MESH_ID_HEX_LEN = 12

# Generate proxy.ply when face count exceeds this (triage compute, not F3D).
PROXY_FACE_THRESHOLD = 100_000

CHUNK_SIZE = 1024 * 1024


def content_sha256(path: Path) -> str:
    """Return full hex SHA-256 of file bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def mesh_id_from_bytes(data: bytes) -> str:
    """Deterministic mesh_id from raw STL bytes."""
    return hashlib.sha256(data).hexdigest()[:MESH_ID_HEX_LEN]


def mesh_id_from_path(path: Path) -> str:
    """Deterministic mesh_id from on-disk STL content."""
    return content_sha256(path)[:MESH_ID_HEX_LEN]


def set_readonly(path: Path) -> None:
    """Best-effort mark path read-only (POSIX 0o444 / Windows FILE_ATTRIBUTE_READONLY)."""
    try:
        mode = path.stat().st_mode
        # Clear write bits for user/group/other.
        path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    except OSError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            # FILE_ATTRIBUTE_READONLY = 0x01
            ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x01)  # type: ignore[attr-defined]
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class JobPaths:
    """Resolved paths under work/<mesh_id>/."""

    work_root: Path
    mesh_id: str

    @property
    def job_dir(self) -> Path:
        return self.work_root / self.mesh_id

    @property
    def original_stl(self) -> Path:
        return self.job_dir / "original.stl"

    @property
    def working_ply(self) -> Path:
        return self.job_dir / "working.ply"

    @property
    def proxy_ply(self) -> Path:
        return self.job_dir / "proxy.ply"

    @property
    def diagnostics_json(self) -> Path:
        return self.job_dir / "diagnostics.json"

    @property
    def report_md(self) -> Path:
        return self.job_dir / "report.md"

    @property
    def views_dir(self) -> Path:
        return self.job_dir / "views"

    @property
    def rois_dir(self) -> Path:
        return self.job_dir / "rois"

    @property
    def revs_dir(self) -> Path:
        """Atomic revision history root (0002+)."""
        return self.job_dir / "revs"

    @property
    def design_dir(self) -> Path:
        """Mechanical design-from-code artifacts (0003+): source, STEP, STL copy."""
        return self.job_dir / "design"


def ensure_job_layout(paths: JobPaths) -> None:
    """Create job directory skeleton (views/, rois/, revs/, design/)."""
    paths.job_dir.mkdir(parents=True, exist_ok=True)
    paths.views_dir.mkdir(exist_ok=True)
    paths.rois_dir.mkdir(exist_ok=True)
    paths.revs_dir.mkdir(exist_ok=True)
    paths.design_dir.mkdir(exist_ok=True)
