"""F3D render + camera math tests (DoD-6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meshops.ingest.pipeline import ingest_stl
from meshops.render.cameras import bbox_cameras
from meshops.render.f3d_renderer import F3DRenderer, RenderUnavailableError


def test_bbox_cameras_seven_views() -> None:
    poses = bbox_cameras((0.0, 0.0, 0.0), (10.0, 20.0, 30.0))
    names = [p.name for p in poses]
    assert names == [
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
        "three_quarter",
    ]
    # Focal at center
    for p in poses:
        assert p.focal_point == (5.0, 10.0, 15.0)
        assert p.ortho_scale > 0


def test_bbox_cameras_scale_with_diagonal() -> None:
    small = bbox_cameras((0, 0, 0), (1, 1, 1))
    large = bbox_cameras((0, 0, 0), (100, 100, 100))
    assert large[0].ortho_scale > small[0].ortho_scale
    # Positions farther for larger mesh
    d_small = np.linalg.norm(np.array(small[0].position) - np.array(small[0].focal_point))
    d_large = np.linalg.norm(np.array(large[0].position) - np.array(large[0].focal_point))
    assert d_large > d_small


def test_camera_names_not_anatomical() -> None:
    poses = bbox_cameras((0, 0, 0), (1, 1, 1))
    names = {p.name for p in poses}
    assert "anatomical_left" not in names
    assert "anatomical_right" not in names


@pytest.mark.f3d
def test_f3d_render_offscreen(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    """Attempt offscreen render; skip only on proven RenderUnavailableError."""
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    try:
        rendered = F3DRenderer(width=256, height=256).render_job(result.mesh_id, work_root=tmp_work)
    except RenderUnavailableError as exc:
        pytest.skip(f"F3D unavailable: {exc}")

    assert len(rendered.view_paths) >= 1
    for p in rendered.view_paths:
        assert Path(p).is_file()
        assert Path(p).stat().st_size > 0
    # DoD-6: ≥1 visual depth map on successful render
    assert len(rendered.depth_paths) >= 1
    for p in rendered.depth_paths:
        assert Path(p).is_file()
        assert Path(p).stat().st_size > 0
    assert rendered.rendered_from in {"working", "original", "proxy"}


@pytest.mark.f3d
def test_render_cli_json(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    from typer.testing import CliRunner

    from meshops.cli import app

    runner = CliRunner()
    r = runner.invoke(
        app,
        ["ingest", "--path", str(solid_cylinder_stl), "--work-root", str(tmp_work), "--json"],
    )
    assert r.exit_code == 0
    import json

    mesh_id = json.loads(r.stdout)["mesh_id"]
    r2 = runner.invoke(
        app,
        ["render", "--mesh-id", mesh_id, "--work-root", str(tmp_work), "--json"],
    )
    if r2.exit_code != 0:
        data = json.loads(r2.stdout)
        if data.get("error") == "RenderUnavailableError":
            pytest.skip(f"F3D unavailable: {data.get('message')}")
    assert r2.exit_code == 0, r2.stdout + r2.stderr
    payload = json.loads(r2.stdout)
    assert payload["ok"] is True
    assert payload["depth_semantics"] == "visual_colormap_not_metric"
    assert len(payload["depth_paths"]) >= 1
