"""CLI organic verbs (track 0006)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from meshops.cli import app
from meshops.organic.models import FinalizeResult, PassResult

runner = CliRunner()


def test_organic_create_status_json(tmp_path: Path) -> None:
    r = runner.invoke(
        app,
        [
            "organic",
            "create",
            "--prompt",
            "cli organic create test",
            "--session-id",
            "oc1100000001",
            "--work-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["session_id"] == "oc1100000001"
    assert any("authored_organic" in n for n in payload["notes"])

    r2 = runner.invoke(
        app,
        [
            "organic",
            "status",
            "--session-id",
            "oc1100000001",
            "--work-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert r2.exit_code == 0, r2.output
    st = json.loads(r2.stdout)
    assert st["status"] == "active"
    assert st["pass_count"] == 0


def test_organic_create_empty_prompt_fails(tmp_path: Path) -> None:
    r = runner.invoke(
        app,
        [
            "organic",
            "create",
            "--prompt",
            "  ",
            "--work-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert r.exit_code != 0
    payload = json.loads(r.stdout)
    assert payload["ok"] is False


def test_organic_plateau_weak_reason(tmp_path: Path) -> None:
    runner.invoke(
        app,
        [
            "organic",
            "create",
            "--prompt",
            "plateau cli test",
            "--session-id",
            "oc1101aea001",
            "--work-root",
            str(tmp_path),
            "--json",
        ],
    )
    r = runner.invoke(
        app,
        [
            "organic",
            "plateau",
            "--session-id",
            "oc1101aea001",
            "--reason",
            "done",
            "--work-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert r.exit_code != 0
    payload = json.loads(r.stdout)
    assert (
        payload.get("code") == "plateau_reason_weak" or "weak" in payload.get("message", "").lower()
    )


def test_organic_help_lists_verbs() -> None:
    r = runner.invoke(app, ["organic", "--help"])
    assert r.exit_code == 0
    out = r.stdout.lower()
    assert "create" in out
    assert "pass" in out
    assert "status" in out
    assert "plateau" in out
    assert "finalize" in out


def test_organic_pass_json_smoke(tmp_path: Path) -> None:
    """Thin CLI wiring: organic pass --json with mocked run_pass."""
    mock_result = PassResult(
        ok=True,
        pass_id="p001_simple_bust",
        recipe="simple_bust",
        mesh_path=tmp_path / "mesh.stl",
        view_paths={"front": str(tmp_path / "front.png")},
        view_kind="stub",
        blender_version="5.2.0",
        returncode=0,
        duration_s=1.25,
        error_code=None,
        messages=[],
        params={},
        scale_mm=180.0,
    )
    with patch("meshops.organic.run_pass", return_value=mock_result):
        r = runner.invoke(
            app,
            [
                "organic",
                "pass",
                "--session-id",
                "oc1100pass001",
                "--work-root",
                str(tmp_path),
                "--json",
            ],
        )
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["pass_id"] == "p001_simple_bust"
    assert payload["recipe"] == "simple_bust"
    assert payload["returncode"] == 0


def test_organic_finalize_json_smoke(tmp_path: Path) -> None:
    """Thin CLI wiring: organic finalize --json with mocked finalize_session."""
    mock_result = FinalizeResult(
        ok=True,
        session_id="oc1100f1a001",
        mesh_id="abc123meshid",
        job_dir=tmp_path / "abc123meshid",
        triage_summary={"mesh_id": "abc123meshid"},
        acceptance=None,
        honesty_message="authored organic — not a print-ready hero",
        error_code=None,
        messages=["finalize_views_stub_ci_or_MESHOPS_STUB_DIFF"],
    )
    with patch("meshops.organic.finalize_session", return_value=mock_result):
        r = runner.invoke(
            app,
            [
                "organic",
                "finalize",
                "--session-id",
                "oc1100f1a001",
                "--work-root",
                str(tmp_path),
                "--json",
            ],
        )
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["session_id"] == "oc1100f1a001"
    assert payload["mesh_id"] == "abc123meshid"
    assert payload["job_dir"] is not None
