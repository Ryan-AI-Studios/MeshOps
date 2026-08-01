"""CLI organic verbs (track 0006)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from meshops.cli import app

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
