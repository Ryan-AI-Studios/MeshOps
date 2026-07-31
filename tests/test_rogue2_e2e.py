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

    # Render optional — isolate F3D so CI SIGSEGV does not kill the suite
    from tests.f3d_helpers import run_f3d_render_job_isolated

    payload = run_f3d_render_job_isolated(mesh_id, tmp_work, width=256, height=256)
    if not payload.get("ok"):
        pytest.skip(f"F3D unavailable: {payload.get('error')}: {payload.get('message')}")
    assert len(payload["view_paths"]) >= 1
    assert len(payload["depth_paths"]) >= 1
