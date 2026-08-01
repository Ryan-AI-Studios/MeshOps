"""Plateau reason quality + criteria_met (track 0006)."""

from __future__ import annotations

from pathlib import Path

import pytest

from meshops.organic import OrganicError, create_session, mark_plateau
from meshops.organic.models import HONESTY_NOTE, REQUIRED_VIEW_KEYS
from meshops.organic.plateau import FILLER_REASONS, validate_plateau_reason
from meshops.organic.session import load_session, save_manifest


def test_weak_reason_length() -> None:
    with pytest.raises(OrganicError) as ei:
        validate_plateau_reason("too short")
    assert ei.value.code == "plateau_reason_weak"


def test_filler_reasons() -> None:
    for filler in ("done", "ok", "n/a", "plateau", "whatever", "finished"):
        with pytest.raises(OrganicError) as ei:
            validate_plateau_reason(filler)
        assert ei.value.code == "plateau_reason_weak"
    assert "done" in FILLER_REASONS


def test_empty_reason() -> None:
    with pytest.raises(OrganicError) as ei:
        validate_plateau_reason("   ")
    assert ei.value.code == "plateau_reason_required"


def test_plateau_criteria_with_pass(tmp_path: Path) -> None:
    m = create_session(
        "figurine test for plateau protocol",
        work_root=tmp_path,
        session_id="oa1ea0000001",
    )
    # Fake a successful pass with required views
    paths, manifest = load_session(m.session_id, work_root=tmp_path)
    pass_id = "p001_simple_bust"
    pdir = paths.pass_dir(pass_id)
    views = pdir / "views"
    views.mkdir(parents=True)
    (pdir / "mesh.stl").write_bytes(b"solid fake\nendsolid fake\n")
    mini = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    for key in REQUIRED_VIEW_KEYS:
        (views / f"{key}.png").write_bytes(mini)
    manifest.passes.append(pass_id)
    save_manifest(paths, manifest)

    rec = mark_plateau(
        m.session_id,
        "agent exhausted simple bust iterations without hero quality",
        work_root=tmp_path,
    )
    assert "min_one_pass" in rec.criteria_met
    assert "max_passes_or_reason" in rec.criteria_met
    assert "all_passes_have_views" in rec.criteria_met
    assert "status_plateau" in rec.criteria_met
    assert rec.allows_hosted_fallback is True
    assert paths.plateau_json.is_file()

    _, m2 = load_session(m.session_id, work_root=tmp_path)
    assert m2.status == "plateau"
    assert HONESTY_NOTE in m2.notes
