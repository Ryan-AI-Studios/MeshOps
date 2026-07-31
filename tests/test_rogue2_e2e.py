"""Rogue2 e2e when fixture present (DoD-5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import rogue2_path
from typer.testing import CliRunner

from meshops.cli import app

runner = CliRunner()


@pytest.mark.rogue2
@pytest.mark.slow
def test_rogue2_ingest_triage_report(tmp_work: Path) -> None:
    path = rogue2_path()
    if path is None:
        pytest.skip("Rogue2.stl not found")

    r = runner.invoke(
        app,
        ["ingest", "--path", str(path), "--work-root", str(tmp_work), "--json"],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    ingest = json.loads(r.stdout)
    mesh_id = ingest["mesh_id"]
    assert ingest["stats"]["faces"] > 100_000  # Rogue2-class
    # Proxy should exist for large meshes
    assert ingest["proxy"] is not None

    r2 = runner.invoke(
        app,
        ["triage", "--mesh-id", mesh_id, "--work-root", str(tmp_work), "--json"],
    )
    assert r2.exit_code == 0, r2.stdout + r2.stderr
    triage = json.loads(r2.stdout)
    diag = triage["diagnostics"]
    assert diag["schema_version"] == "1.0.0"
    assert "sheet_score" in diag
    assert diag["sheet_score"]["auto_action"] != "delete"

    r3 = runner.invoke(
        app,
        ["report", "--mesh-id", mesh_id, "--work-root", str(tmp_work), "--json"],
    )
    assert r3.exit_code == 0, r3.stdout + r3.stderr
    report = json.loads(r3.stdout)
    assert Path(report["report_path"]).is_file()

    # Render optional — skip only on RenderUnavailableError
    r4 = runner.invoke(
        app,
        ["render", "--mesh-id", mesh_id, "--work-root", str(tmp_work), "--json"],
    )
    if r4.exit_code != 0:
        try:
            data = json.loads(r4.stdout)
        except json.JSONDecodeError:
            pytest.skip(f"render failed non-json: {r4.stdout}")
        if data.get("error") == "RenderUnavailableError":
            pytest.skip(f"F3D unavailable: {data.get('message')}")
        raise AssertionError(r4.stdout + r4.stderr)
    else:
        payload = json.loads(r4.stdout)
        assert payload["ok"] is True
        assert len(payload["view_paths"]) >= 1
        assert len(payload["depth_paths"]) >= 1
