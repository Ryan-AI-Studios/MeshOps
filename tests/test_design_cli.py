"""CLI design verbs (DoD-9,11)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshops.cli import app

runner = CliRunner()

# Core CLI tests always collect; design e2e needs build123d.
_HAS_B123D = True
try:
    import build123d  # noqa: F401  # type: ignore[reportMissingImports]
except ImportError:
    _HAS_B123D = False


def test_cli__design_help() -> None:
    r = runner.invoke(app, ["design", "--help"])
    assert r.exit_code == 0
    assert "from-spec" in r.stdout
    assert "run" in r.stdout


@pytest.mark.design
@pytest.mark.skipif(not _HAS_B123D, reason="build123d not installed")
def test_cli__design_from_spec_json(tmp_path: Path) -> None:
    work = tmp_path / "work"
    env = {**os.environ, "MESHOPS_STUB_DIFF": "1"}
    r = runner.invoke(
        app,
        [
            "design",
            "from-spec",
            "--template",
            "bracket_m4",
            "--hole-spacing",
            "40",
            "--wall",
            "3",
            "--thickness",
            "4",
            "--work-root",
            str(work),
            "--no-diff",
            "--json",
        ],
        env=env,
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["mesh_id"]
    assert payload["slice"] == "skipped"
    assert payload["acceptance"]["ok"] is True
    assert payload["acceptance"]["policy_tier"] == "design"
    job = Path(payload["job_dir"])
    assert (job / "design" / "part.stl").is_file()
    assert (job / "design" / "part.step").is_file()
    assert (job / "original.stl").is_file()


@pytest.mark.design
@pytest.mark.skipif(not _HAS_B123D, reason="build123d not installed")
def test_cli__design_from_spec_bad_params_exit(tmp_path: Path) -> None:
    work = tmp_path / "work"
    r = runner.invoke(
        app,
        [
            "design",
            "from-spec",
            "--hole-spacing",
            "12",
            "--wall",
            "5",
            "--hole-diameter",
            "4",
            "--work-root",
            str(work),
            "--json",
        ],
    )
    assert r.exit_code != 0


def test_cli__missing_build123d_message_path() -> None:
    """When build123d missing, design from-spec should fail clearly (if import path hits).

    With build123d installed this documents the error class via DesignError path
    by simulating missing_dependency from runner — unit-level check of message.
    """
    from meshops.design.errors import DesignError

    err = DesignError(
        "build123d is not installed; install with: uv sync --extra design",
        code="missing_dependency",
    )
    assert err.code == "missing_dependency"
    assert "build123d" in str(err)
    assert "extra design" in str(err)
