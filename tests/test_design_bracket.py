"""Bracket template e2e + golden dimensions (DoD-4,5,6,7,8,10). Requires build123d."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

pytest.importorskip("build123d")

from meshops.design.models import BracketParams
from meshops.design.orchestrate import design_from_template
from meshops.design.templates.bracket_m4 import expected_dimensions
from meshops.jobstore.paths import JobPaths, content_sha256

pytestmark = pytest.mark.design

TOL_MM = 0.1


def test_bracket__from_spec_e2e(tmp_work: Path) -> None:
    result = design_from_template(
        "bracket_m4",
        params=BracketParams(),
        work_root=tmp_work,
        timeout_s=120.0,
        no_diff=True,
    )
    assert result.mesh_id
    paths = JobPaths(work_root=tmp_work, mesh_id=result.mesh_id)
    assert paths.original_stl.is_file()
    assert paths.design_dir.is_dir()
    part_stl = paths.design_dir / "part.stl"
    part_step = paths.design_dir / "part.step"
    source_py = paths.design_dir / "source.py"
    spec = paths.design_dir / "spec.json"
    man = paths.design_dir / "manifest.json"
    assert part_stl.is_file() and part_stl.stat().st_size > 0
    assert part_step.is_file() and part_step.stat().st_size > 0
    assert source_py.is_file()
    assert "result" in source_py.read_text(encoding="utf-8")
    assert spec.is_file()
    assert man.is_file()
    assert content_sha256(paths.original_stl) == content_sha256(part_stl)
    assert result.manifest.units == "mm"
    assert result.manifest.content_sha256 == content_sha256(paths.original_stl)
    assert paths.diagnostics_json.is_file()
    assert result.acceptance is not None
    assert result.acceptance.ok is True
    assert result.acceptance.policy_tier == "design"
    assert result.acceptance.view_paths
    # Slice skipped honesty
    assert result.acceptance.slice is not None
    assert result.acceptance.slice.status == "skipped"
    assert any("slice_skipped" in n for n in result.notes)
    assert result.ok is True


def test_bracket__golden_dimensions(tmp_work: Path) -> None:
    params = BracketParams(
        hole_spacing_mm=40.0,
        wall_mm=3.0,
        thickness_mm=4.0,
        hole_diameter_mm=4.2,
    )
    result = design_from_template(
        "bracket_m4",
        params=params,
        work_root=tmp_work,
        timeout_s=120.0,
        no_diff=True,
    )
    mesh = trimesh.load(result.paths["design_stl"], force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    extents = mesh.bounding_box.extents  # (dx, dy, dz)
    expected = expected_dimensions(params)
    assert extents[0] == pytest.approx(expected["extent_x_mm"], abs=TOL_MM)
    assert extents[1] == pytest.approx(expected["extent_y_mm"], abs=TOL_MM)
    assert extents[2] == pytest.approx(expected["extent_z_mm"], abs=TOL_MM)
    # Thickness / wall encoded in extents
    assert extents[2] == pytest.approx(params.thickness_mm, abs=TOL_MM)
    assert extents[1] == pytest.approx(params.hole_diameter_mm + 2 * params.wall_mm, abs=TOL_MM)
    # Hole spacing reflected in plate length construction
    assert extents[0] == pytest.approx(
        params.hole_spacing_mm + params.hole_diameter_mm + 2 * params.wall_mm,
        abs=TOL_MM,
    )
    # Source documents spacing for review
    src = Path(result.paths["source"]).read_text(encoding="utf-8")
    assert f"spacing={params.hole_spacing_mm}" in src or "_half_span" in src
