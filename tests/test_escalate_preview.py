"""T3 preview never promotes / never claims fixed (DoD-3, DoD-4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshops.escalate.errors import EscalateError
from meshops.escalate.preview_t3 import (
    FORBIDDEN_PREVIEW_OPS,
    assert_preview_op_allowed,
    preview_t3,
    refuse_promote_preview,
)
from meshops.escalate.roi import create_roi_bbox
from meshops.ingest.pipeline import ingest_stl


def test_preview_never_claims_fixed(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    roi = create_roi_bbox(
        ing.mesh_id,
        (-5, -5, -5),
        (5, 5, 5),
        work_root=tmp_work,
    )
    result = preview_t3(ing.mesh_id, roi.roi_id, work_root=tmp_work)
    assert result.preview is True
    assert result.ok is False
    assert result.may_promote_working is False
    assert result.may_claim_fixed is False
    assert "preview_only" in result.notes
    assert "NOT fixed" in result.honesty_note or "not fixed" in result.honesty_note.lower()

    meta = json.loads((result.preview_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["ok"] is False
    assert meta["preview"] is True
    assert meta["may_promote_working"] is False


def test_preview_does_not_touch_working_ply(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    working = tmp_work / ing.mesh_id / "working.ply"
    assert working.is_file()
    before = working.read_bytes()
    mtime = working.stat().st_mtime_ns
    preview_t3(ing.mesh_id, work_root=tmp_work)
    assert working.read_bytes() == before
    assert working.stat().st_mtime_ns == mtime


def test_preview_under_previews_dir(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    result = preview_t3(ing.mesh_id, work_root=tmp_work)
    assert "previews" in result.preview_dir.parts
    assert (result.preview_dir / "mesh.stl").is_file()
    assert (result.preview_dir / "meta.json").is_file()


def test_forbidden_preview_ops() -> None:
    for op in (
        "whole_model_voxel_remesh",
        "linked_flat_delete",
        "full_mesh_boolean_after_solidify",
        "global_remesh",
    ):
        assert op in FORBIDDEN_PREVIEW_OPS
        with pytest.raises(EscalateError) as ei:
            assert_preview_op_allowed(op)
        assert ei.value.code == "preview_refuse_promote"


def test_refuse_promote_preview() -> None:
    with pytest.raises(EscalateError) as ei:
        refuse_promote_preview(preview_id="pabc123")
    assert ei.value.code == "preview_refuse_promote"

    with pytest.raises(EscalateError):
        refuse_promote_preview(notes=["preview_only", "experiment"])

    with pytest.raises(EscalateError):
        refuse_promote_preview(recipe_id="t3_preview_local")
