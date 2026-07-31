"""Triage orchestration tests (DoD-2, partial DoD-5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from meshops.ingest.pipeline import ingest_stl
from meshops.models.diagnostics import Diagnostics, LateralityStatus
from meshops.triage.orchestrate import JobNotFoundError, mesh_triage


def test_triage_without_ingest_fails(tmp_work: Path) -> None:
    with pytest.raises(JobNotFoundError):
        mesh_triage("nonexistent12", work_root=tmp_work)


def test_triage_writes_diagnostics(arm_sheet_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(arm_sheet_stl, work_root=tmp_work)
    diag = mesh_triage(result.mesh_id, work_root=tmp_work)
    assert isinstance(diag, Diagnostics)
    assert diag.schema_version == "1.0.0"
    assert diag.mesh_id == result.mesh_id
    path = tmp_work / result.mesh_id / "diagnostics.json"
    assert path.is_file()
    restored = Diagnostics.model_validate_json(path.read_text(encoding="utf-8"))
    assert restored.sheet_score.score == diag.sheet_score.score


def test_triage_idempotent_overwrite(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    result = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    d1 = mesh_triage(result.mesh_id, work_root=tmp_work)
    d2 = mesh_triage(result.mesh_id, work_root=tmp_work)
    assert d1.mesh_id == d2.mesh_id
    assert abs(d1.sheet_score.score - d2.sheet_score.score) < 1e-9


def test_multi_component_needs_user_input(gap_sheet_stl: Path, tmp_work: Path) -> None:
    """Difficulty §1 — multi-component → laterality unknown, needs_user_input."""
    result = ingest_stl(gap_sheet_stl, work_root=tmp_work)
    diag = mesh_triage(result.mesh_id, work_root=tmp_work)
    if diag.stats.components > 1:
        assert diag.laterality_status == LateralityStatus.UNKNOWN
        assert diag.needs_user_input is True
