"""CLI escalate verbs + --json exit codes (DoD-7, DoD-9)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from typer.testing import CliRunner

from meshops.cli import app
from meshops.ingest.pipeline import ingest_stl

runner = CliRunner()


def test_cli_escalate_help() -> None:
    r = runner.invoke(app, ["escalate", "--help"])
    assert r.exit_code == 0
    assert "roi" in r.stdout
    assert "preview-t3" in r.stdout
    assert "handoff" in r.stdout
    assert "import-sculpt" in r.stdout
    assert "MESHOPS_BLENDER" in r.stdout or "5.2" in r.stdout or "Blender" in r.stdout


def test_cli_root_help_mentions_escalate() -> None:
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "escalate" in r.stdout


def test_cli_roi_json(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    r = runner.invoke(
        app,
        [
            "escalate",
            "roi",
            "--mesh-id",
            ing.mesh_id,
            "--bbox",
            "-1,-1,0,1,1,5",
            "--work-root",
            str(tmp_work),
            "--json",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["roi"]["roi_id"]
    assert payload["roi"]["kind"] == "aabb"


def test_cli_preview_t3_json(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    r = runner.invoke(
        app,
        [
            "escalate",
            "preview-t3",
            "--mesh-id",
            ing.mesh_id,
            "--work-root",
            str(tmp_work),
            "--json",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    assert payload["preview"] is True
    assert payload["ok"] is False
    assert payload["may_promote_working"] is False
    assert payload["may_claim_fixed"] is False


def test_cli_import_sculpt_requires_approve(
    solid_cylinder_stl: Path,
    tmp_work: Path,
) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    r = runner.invoke(
        app,
        [
            "escalate",
            "import-sculpt",
            "--mesh-id",
            ing.mesh_id,
            "--path",
            str(solid_cylinder_stl),
            "--work-root",
            str(tmp_work),
            "--json",
        ],
    )
    assert r.exit_code != 0
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert payload.get("code") == "approve_required"


def test_cli_import_sculpt_happy(
    solid_cylinder_stl: Path,
    tmp_work: Path,
) -> None:
    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    env = {**os.environ, "MESHOPS_STUB_DIFF": "1"}
    r = runner.invoke(
        app,
        [
            "escalate",
            "import-sculpt",
            "--mesh-id",
            ing.mesh_id,
            "--path",
            str(solid_cylinder_stl),
            "--approve",
            "--no-diff",
            "--work-root",
            str(tmp_work),
            "--json",
        ],
        env=env,
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["recipe_id"] == "blender_sculpt_import"
    assert payload["acceptance"]["policy_tier"] == "sculpt"
    assert payload["promoted_to_working"] is False


def test_cli_roi_missing_job_exit(tmp_work: Path) -> None:
    r = runner.invoke(
        app,
        [
            "escalate",
            "roi",
            "--mesh-id",
            "deadbeef0001",
            "--bbox",
            "0,0,0,1,1,1",
            "--work-root",
            str(tmp_work),
            "--json",
        ],
    )
    assert r.exit_code != 0
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
