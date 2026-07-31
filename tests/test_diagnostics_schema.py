"""Pydantic diagnostics schema tests (DoD-2)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from meshops.models.diagnostics import (
    SCHEMA_VERSION,
    AutoAction,
    DefectClass,
    DefectHypothesis,
    Diagnostics,
    LateralityStatus,
    MeshStats,
    SheetScoreFeatures,
    SheetScoreResult,
)


def _minimal_stats() -> MeshStats:
    return MeshStats(
        faces=10,
        vertices=8,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
        bbox_diagonal=1.732,
        components=1,
        file_size_bytes=100,
        content_sha256="a" * 64,
        mesh_id="a" * 12,
    )


def test_schema_version_literal() -> None:
    assert SCHEMA_VERSION == "1.0.0"
    diag = Diagnostics(
        mesh_id="a" * 12,
        stats=_minimal_stats(),
        sheet_score=SheetScoreResult(score=0.1, confidence=0.5),
    )
    assert diag.schema_version == "1.0.0"
    data = json.loads(diag.model_dump_json())
    assert data["schema_version"] == "1.0.0"


def test_round_trip_json() -> None:
    diag = Diagnostics(
        mesh_id="deadbeefcafe",
        stats=_minimal_stats(),
        defect_hypotheses=[
            DefectHypothesis(
                defect_class=DefectClass.T3_SHEET,
                confidence=0.8,
                notes="test",
            )
        ],
        sheet_score=SheetScoreResult(
            score=0.7,
            confidence=0.6,
            features=SheetScoreFeatures(thinness_mean=0.1, stage2_used=True),
            auto_action=AutoAction.REVIEW,
        ),
        laterality_status=LateralityStatus.UNKNOWN,
        needs_user_input=True,
    )
    raw = diag.model_dump_json()
    restored = Diagnostics.model_validate_json(raw)
    assert restored.mesh_id == diag.mesh_id
    assert restored.sheet_score.score == 0.7
    assert restored.needs_user_input is True
    assert restored.defect_hypotheses[0].defect_class == DefectClass.T3_SHEET


def test_reject_invalid_defect_class() -> None:
    with pytest.raises(ValidationError):
        DefectHypothesis(defect_class="T9_bogus", confidence=0.5)  # type: ignore[arg-type]


def test_auto_action_has_no_delete() -> None:
    names = {a.value for a in AutoAction}
    assert "delete" not in names
    assert AutoAction.NONE.value == "none"


def test_score_bounds() -> None:
    with pytest.raises(ValidationError):
        SheetScoreResult(score=1.5, confidence=0.5)
    with pytest.raises(ValidationError):
        SheetScoreResult(score=0.5, confidence=-0.1)
