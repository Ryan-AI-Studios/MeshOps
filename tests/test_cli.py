"""CLI contract tests — four verbs + --json (DoD-7)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from meshops.cli import app

runner = CliRunner()


def test_version_json() -> None:
    result = runner.invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["ok"] is True
    assert "version" in data


def test_ingest_triage_report_json(arm_sheet_stl: Path, tmp_work: Path) -> None:
    r = runner.invoke(
        app,
        ["ingest", "--path", str(arm_sheet_stl), "--work-root", str(tmp_work), "--json"],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    ingest = json.loads(r.stdout)
    assert ingest["ok"] is True
    mesh_id = ingest["mesh_id"]

    r2 = runner.invoke(
        app,
        ["triage", "--mesh-id", mesh_id, "--work-root", str(tmp_work), "--json"],
    )
    assert r2.exit_code == 0, r2.stdout + r2.stderr
    triage = json.loads(r2.stdout)
    assert triage["ok"] is True
    assert triage["diagnostics"]["schema_version"] == "1.0.0"
    assert "sheet_score" in triage["diagnostics"]

    r3 = runner.invoke(
        app,
        ["report", "--mesh-id", mesh_id, "--work-root", str(tmp_work), "--json"],
    )
    assert r3.exit_code == 0, r3.stdout + r3.stderr
    report = json.loads(r3.stdout)
    assert report["ok"] is True
    assert Path(report["report_path"]).is_file()


def test_triage_missing_job_json(tmp_work: Path) -> None:
    r = runner.invoke(
        app,
        ["triage", "--mesh-id", "missing000001", "--work-root", str(tmp_work), "--json"],
    )
    assert r.exit_code != 0
    data = json.loads(r.stdout)
    assert data["ok"] is False


def test_core_verbs_present() -> None:
    """Core CLI verbs present (0002 adds repair/export/diff; 0010 doctor)."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    help_text = result.stdout.lower()
    assert "ingest" in help_text
    assert "triage" in help_text
    assert "render" in help_text
    assert "report" in help_text
    # 0002 guarded mutation verbs
    assert "repair" in help_text
    assert "export" in help_text
    assert "diff" in help_text
    # 0010 ops
    assert "doctor" in help_text
    for banned in ("delete-sheet", "remesh", "mutate"):
        assert banned not in help_text
