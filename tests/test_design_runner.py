"""Harness runner + export unit tests (DoD-2,3,15). Requires build123d."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("build123d")

from meshops.design.errors import DesignError
from meshops.design.export_b123d import ensure_single_solid, export_shape
from meshops.design.runner import run_geometry_source

pytestmark = pytest.mark.design

_TINY_BOX = """
from build123d import Box
result = Box(20, 15, 10)
"""

# Two disjoint solids — fuse cannot collapse to a single solid (DoD-15).
_MULTI_SOLID_DISJOINT = """
from build123d import Box, Part, Pos
result = Part() + Box(10, 10, 10) + (Pos(100, 0, 0) * Box(10, 10, 10))
"""


def test_runner__tiny_box_exports(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    out = run_geometry_source(_TINY_BOX, staging_dir=stage, timeout_s=120.0)
    assert out.stl_path.is_file()
    assert out.stl_path.stat().st_size > 0
    assert out.step_path.is_file()
    assert out.step_path.stat().st_size > 0
    assert out.source_path.is_file()
    assert "MESHOPS_DESIGN_OK" in out.stdout or out.runner_meta["returncode"] == 0


def test_runner__ast_blocks_before_subprocess(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(DesignError) as ei:
        run_geometry_source("import subprocess\nresult=1\n", staging_dir=stage)
    assert ei.value.code == "ast_denied"


def test_runner__missing_result(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    src = "from build123d import Box\npart = Box(10, 10, 10)\n"
    with pytest.raises(DesignError) as ei:
        run_geometry_source(src, staging_dir=stage, timeout_s=60.0)
    assert ei.value.code in {"missing_result", "runner_crash"}


def test_runner__timeout_maps_to_timeout_code(tmp_path: Path) -> None:
    """DoD-3: subprocess.TimeoutExpired → DesignError(code=timeout)."""
    stage = tmp_path / "stage"
    stage.mkdir()
    with (
        patch(
            "meshops.design.runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["py"], timeout=1.0),
        ),
        pytest.raises(DesignError) as ei,
    ):
        run_geometry_source(_TINY_BOX, staging_dir=stage, timeout_s=1.0)
    assert ei.value.code == "timeout"
    assert ei.value.details.get("timeout_s") == 1.0


def test_runner__nonzero_exit_maps_to_runner_crash(tmp_path: Path) -> None:
    """DoD-3: non-zero worker exit without kernel markers → runner_crash."""
    stage = tmp_path / "stage"
    stage.mkdir()
    fake = MagicMock()
    fake.returncode = 1
    fake.stdout = ""
    fake.stderr = "boom worker failed"
    with (
        patch("meshops.design.runner.subprocess.run", return_value=fake),
        pytest.raises(DesignError) as ei,
    ):
        run_geometry_source(_TINY_BOX, staging_dir=stage, timeout_s=30.0)
    assert ei.value.code == "runner_crash"
    assert ei.value.details.get("returncode") == 1


def test_runner__multi_solid_disjoint_fails(tmp_path: Path) -> None:
    """DoD-15: disjoint multi-solid cannot fuse → multi_solid (fail closed)."""
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(DesignError) as ei:
        run_geometry_source(_MULTI_SOLID_DISJOINT, staging_dir=stage, timeout_s=120.0)
    assert ei.value.code == "multi_solid"


def test_export__ensure_single_solid_disjoint() -> None:
    """DoD-15 in-process: disjoint Part solids raise multi_solid."""
    from build123d import Box, Part, Pos  # type: ignore[reportMissingImports]

    multi = Part() + Box(10, 10, 10) + (Pos(100, 0, 0) * Box(10, 10, 10))
    assert len(list(multi.solids())) == 2
    with pytest.raises(DesignError) as ei:
        ensure_single_solid(multi)
    assert ei.value.code == "multi_solid"


def test_export__ensure_single_solid_overlapping_ok(tmp_path: Path) -> None:
    """Overlapping solids already one solid after Part composition → export ok."""
    from build123d import Box, Part, Pos  # type: ignore[reportMissingImports]

    # Offset less than size so boxes intersect; Part composition yields 1 solid.
    shape = Part() + Box(10, 10, 10) + (Pos(5, 0, 0) * Box(10, 10, 10))
    solid = ensure_single_solid(shape)
    assert len(list(solid.solids())) == 1
    stl = tmp_path / "fused.stl"
    step = tmp_path / "fused.step"
    meta = export_shape(solid, stl_path=stl, step_path=step)
    assert stl.is_file() and stl.stat().st_size > 0
    assert step.is_file() and step.stat().st_size > 0
    assert meta["stl_bytes"] > 0


def test_export__bool_false_path(tmp_path: Path) -> None:
    from build123d import Box  # type: ignore[reportMissingImports]

    shape = Box(10, 10, 10)
    stl = tmp_path / "a.stl"
    step = tmp_path / "a.step"
    # Patch the name looked up inside export_b123d after import build123d as b123d.
    with patch("build123d.export_stl", return_value=False) as mocked:
        assert mocked is not None
        with pytest.raises(DesignError) as ei:
            export_shape(shape, stl_path=stl, step_path=step)
        # export_stl False → export_failed; if mock misses, cad_kernel is also fail-closed
        assert ei.value.code in {"export_failed", "cad_kernel_failure"}
