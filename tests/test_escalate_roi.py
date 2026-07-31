"""ROI bbox → mask.json (DoD-2, partial DoD-14)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshops.escalate.errors import EscalateError
from meshops.escalate.models import RoiManifest
from meshops.escalate.roi import create_roi_bbox, create_roi_from_sheet_heuristic, load_roi
from meshops.ingest.pipeline import ingest_stl


def test_create_roi_bbox_writes_mask(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    man = create_roi_bbox(
        ing.mesh_id,
        (-1.0, -1.0, 0.0),
        (1.0, 1.0, 5.0),
        work_root=tmp_work,
        notes=["unit_test"],
    )
    assert man.schema_version == "1.0.0"
    assert man.kind == "aabb"
    assert man.source == "manual"
    assert man.roi_id.startswith("r")
    assert man.bbox_min == (-1.0, -1.0, 0.0)
    assert man.bbox_max == (1.0, 1.0, 5.0)

    mask = tmp_work / ing.mesh_id / "rois" / man.roi_id / "mask.json"
    assert mask.is_file()
    raw = json.loads(mask.read_text(encoding="utf-8"))
    assert raw["schema_version"] == "1.0.0"
    assert raw["kind"] == "aabb"
    reloaded = load_roi(ing.mesh_id, man.roi_id, work_root=tmp_work)
    assert isinstance(reloaded, RoiManifest)
    assert reloaded.roi_id == man.roi_id


def test_roi_deterministic_id(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    a = create_roi_bbox(ing.mesh_id, (0, 0, 0), (1, 1, 1), work_root=tmp_work)
    b = create_roi_bbox(ing.mesh_id, (0, 0, 0), (1, 1, 1), work_root=tmp_work)
    assert a.roi_id == b.roi_id


def test_roi_invalid_bbox(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    with pytest.raises(EscalateError) as ei:
        create_roi_bbox(ing.mesh_id, (0, 0, 0), (0, 0, 0), work_root=tmp_work)
    assert ei.value.code == "invalid_bbox"


def test_roi_job_not_found(tmp_work: Path) -> None:
    with pytest.raises(EscalateError) as ei:
        create_roi_bbox("deadbeef0001", (0, 0, 0), (1, 1, 1), work_root=tmp_work)
    assert ei.value.code == "job_not_found"


def test_roi_heuristic_records_source(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    man = create_roi_from_sheet_heuristic(ing.mesh_id, work_root=tmp_work)
    assert man.source == "heuristic"
    assert any("heuristic" in n.lower() or "source=heuristic" in n for n in man.notes)
    # mask still valid schema
    loaded = load_roi(ing.mesh_id, man.roi_id, work_root=tmp_work)
    assert loaded.source == "heuristic"


def test_roi_no_stdin_on_needs_user_input(
    arm_sheet_stl: Path,
    tmp_work: Path,
) -> None:
    """DoD-14: laterality / needs_user_input is note-only, never blocks."""
    from meshops.triage.orchestrate import mesh_triage

    ing = ingest_stl(arm_sheet_stl, work_root=tmp_work)
    mesh_triage(ing.mesh_id, work_root=tmp_work)
    man = create_roi_bbox(
        ing.mesh_id,
        (-10, -10, -10),
        (10, 10, 10),
        work_root=tmp_work,
    )
    assert man.roi_id
    # Function returns without prompting; notes may include needs_user_input reminder
