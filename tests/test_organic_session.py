"""Organic session create / SessionPaths / honesty (track 0006)."""

from __future__ import annotations

from pathlib import Path

import pytest

from meshops.jobstore.paths import JobPaths, ensure_job_layout
from meshops.organic import (
    ORGANIC_SCHEMA_VERSION,
    OrganicError,
    SessionPaths,
    create_session,
    load_session,
)
from meshops.organic.models import HONESTY_NOTE


def test_session_paths_layout(tmp_path: Path) -> None:
    sp = SessionPaths(work_root=tmp_path, session_id="oabcdef01234")
    assert sp.session_dir == tmp_path / "oabcdef01234"
    assert sp.organic_dir == tmp_path / "oabcdef01234" / "organic"
    assert sp.manifest_path.name == "manifest.json"
    assert sp.pass_dir("p001_simple_bust").name == "p001_simple_bust"
    assert sp.failed_pass_dir("p001_simple_bust").name == "failed_p001_simple_bust"
    assert "organic" in str(sp.final_stl)
    sp.ensure_layout()
    assert sp.refs_dir.is_dir()
    assert sp.passes_dir.is_dir()


def test_ensure_job_layout_has_no_organic(tmp_path: Path) -> None:
    """B6: JobPaths must not grow organic/ for every triage job."""
    jp = JobPaths(work_root=tmp_path, mesh_id="abc123def456")
    ensure_job_layout(jp)
    assert not (jp.job_dir / "organic").exists()
    assert not hasattr(jp, "organic_dir")


def test_create_session_and_honesty(tmp_path: Path) -> None:
    m = create_session(
        "simple clay bust of a warrior",
        style_notes="matte clay",
        work_root=tmp_path,
        session_id="o0123456789a",
    )
    assert m.session_id == "o0123456789a"
    assert m.status == "active"
    assert m.passes == []
    assert HONESTY_NOTE in m.notes
    assert m.schema_version == ORGANIC_SCHEMA_VERSION
    sp = SessionPaths(work_root=tmp_path, session_id=m.session_id)
    assert sp.manifest_path.is_file()
    assert sp.prompt_md.read_text(encoding="utf-8").startswith("simple clay")
    assert sp.session_report_md.is_file()


def test_create_empty_prompt_reject(tmp_path: Path) -> None:
    with pytest.raises(OrganicError) as ei:
        create_session("   ", work_root=tmp_path)
    assert ei.value.code == "invalid_params"


def test_load_session_missing(tmp_path: Path) -> None:
    with pytest.raises(OrganicError) as ei:
        load_session("oaaaaaaaaaaa", work_root=tmp_path)
    assert ei.value.code == "session_not_found"


def test_schema_independence() -> None:
    """B10: organic 1.0.0 is 0006-owned — not acceptance/slice/design freeze."""
    from meshops.acceptance.models import AcceptanceResult
    from meshops.design.models import DESIGN_MANIFEST_SCHEMA

    assert ORGANIC_SCHEMA_VERSION == "1.0.0"
    # Document independence: same string value is OK but owners differ
    assert DESIGN_MANIFEST_SCHEMA == "1.0.0"
    # AcceptanceResult has its own schema field default
    assert "schema_version" in AcceptanceResult.model_fields


def test_session_id_pattern(tmp_path: Path) -> None:
    with pytest.raises(OrganicError) as ei:
        create_session("hello world prompt", session_id="bad-id", work_root=tmp_path)
    assert ei.value.code == "invalid_params"

    m = create_session("hello world prompt enough length", work_root=tmp_path)
    assert m.session_id.startswith("o")
    assert len(m.session_id) == 12
