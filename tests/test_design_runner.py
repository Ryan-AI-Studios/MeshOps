"""Harness runner + export unit tests (DoD-2,3,15). Requires build123d."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("build123d")

from meshops.design.errors import DesignError
from meshops.design.export_b123d import export_shape
from meshops.design.runner import run_geometry_source

pytestmark = pytest.mark.design

_TINY_BOX = """
from build123d import Box
result = Box(20, 15, 10)
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


def test_export__bool_false_path(tmp_path: Path) -> None:
    from build123d import Box

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
