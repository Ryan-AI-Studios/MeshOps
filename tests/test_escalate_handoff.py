"""Blender handoff integration (DoD-5, DoD-6) — skip when Blender absent."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from meshops.escalate.discover import find_blender
from meshops.escalate.errors import EscalateError
from meshops.escalate.handoff import build_handoff, probe_handoff_blend
from meshops.escalate.roi import create_roi_bbox
from meshops.escalate.version import require_blender_52
from meshops.ingest.pipeline import ingest_stl


def _blender_or_skip() -> Path:
    try:
        b = find_blender(require=True)
    except EscalateError:
        pytest.skip("Blender 5.2 LTS not found")
    assert b is not None
    try:
        require_blender_52(b)
    except EscalateError as exc:
        pytest.skip(f"Blender version not 5.2: {exc}")
    return b


@pytest.mark.blender
def test_handoff_produces_blend(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    blender = _blender_or_skip()
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    # ROI covering typical cylinder extent (synthetic is ~unit scale)
    roi = create_roi_bbox(
        ing.mesh_id,
        (-10.0, -10.0, -10.0),
        (10.0, 10.0, 20.0),
        work_root=tmp_work,
    )
    man = build_handoff(ing.mesh_id, roi.roi_id, work_root=tmp_work, timeout_s=120.0)
    assert man.schema_version == "1.0.0"
    blend = Path(man.blend_path)
    assert blend.is_file()
    assert blend.stat().st_size > 0
    assert Path(man.instructions_path).is_file()
    instructions = Path(man.instructions_path).read_text(encoding="utf-8")
    assert "Inflate" in instructions
    assert "Grab" in instructions
    assert "Clay Strips" in instructions
    assert "Smooth" in instructions
    assert "voxel remesh" in instructions.lower() or "remesh" in instructions.lower()
    assert "boolean" in instructions.lower()
    assert man.vertex_group == "meshops_roi"
    assert "front" in man.cameras
    assert "waist_zoom" in man.cameras
    # build_handoff already probes; re-probe here to assert VG/cameras inside .blend
    probe_handoff_blend(
        blend,
        blender=blender,
        vertex_group=man.vertex_group,
        cameras=man.cameras,
        timeout_s=120.0,
    )
    # ROI covering the cylinder should assign some verts (parse from build stdout)
    assert man.roi_vert_count is not None
    assert man.roi_vert_count >= 0
    if man.roi_vert_count == 0:
        assert "empty_roi_vertex_group" in man.notes
    else:
        assert "empty_roi_vertex_group" not in man.notes
    meta = tmp_work / ing.mesh_id / "handoff" / "meta.json"
    assert meta.is_file()
    raw = json.loads(meta.read_text(encoding="utf-8"))
    assert raw["blender_version"].startswith("5.2")
    assert "roi_vert_count" in raw


@pytest.mark.blender
def test_handoff_missing_roi(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    _blender_or_skip()
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    with pytest.raises(EscalateError) as ei:
        build_handoff(ing.mesh_id, "r_does_not_exist", work_root=tmp_work)
    assert ei.value.code == "roi_not_found"


def test_handoff_job_not_found(tmp_work: Path) -> None:
    with pytest.raises(EscalateError) as ei:
        build_handoff("deadbeef0001", "r00000000", work_root=tmp_work)
    assert ei.value.code == "job_not_found"


@pytest.mark.blender
def test_cli_handoff_json(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    from typer.testing import CliRunner

    from meshops.cli import app

    _blender_or_skip()
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    roi = create_roi_bbox(
        ing.mesh_id,
        (-10.0, -10.0, -10.0),
        (10.0, 10.0, 20.0),
        work_root=tmp_work,
    )
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "escalate",
            "handoff",
            "--mesh-id",
            ing.mesh_id,
            "--roi",
            roi.roi_id,
            "--work-root",
            str(tmp_work),
            "--json",
        ],
        env={**os.environ},
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert Path(payload["handoff"]["blend_path"]).is_file()
