"""Shared pytest fixtures and synthetic helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fixtures.synthetic import build as synth_build


@pytest.fixture
def tmp_work(tmp_path: Path) -> Path:
    """Isolated work root for job-store tests."""
    root = tmp_path / "work"
    root.mkdir()
    return root


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped code-generated synthetic STLs."""
    out = tmp_path_factory.mktemp("synthetic_stl")
    synth_build.build_all(out)
    return out


@pytest.fixture
def solid_cylinder_stl(synthetic_dir: Path) -> Path:
    return synthetic_dir / "solid_cylinder.stl"


@pytest.fixture
def arm_sheet_stl(synthetic_dir: Path) -> Path:
    return synthetic_dir / "cylinder_arm_sheet.stl"


@pytest.fixture
def gap_sheet_stl(synthetic_dir: Path) -> Path:
    return synthetic_dir / "two_body_gap_sheet.stl"


@pytest.fixture
def clothing_cape_stl(synthetic_dir: Path) -> Path:
    return synthetic_dir / "clothing_cape.stl"


def rogue2_path() -> Path | None:
    """Resolve Rogue2 STL if present."""
    env = os.environ.get("MESHOPS_ROGUE2_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path(r"C:\Users\RyanB\Documents\3D\Elizabeth\Rogue2.stl"))
    repo_fixture = Path(__file__).resolve().parents[1] / "fixtures" / "rogue2" / "Rogue2.stl"
    candidates.append(repo_fixture)
    for p in candidates:
        if p.is_file():
            return p
    return None


@pytest.fixture
def rogue2_stl() -> Path:
    path = rogue2_path()
    if path is None:
        pytest.skip("Rogue2.stl not found (set MESHOPS_ROGUE2_PATH)")
    return path
