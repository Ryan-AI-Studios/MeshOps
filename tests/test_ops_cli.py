"""CLI tests for meshops doctor."""

from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from meshops.cli import app
from meshops.ops.models import (
    BlenderToolStatus,
    DiskInfo,
    DoctorReport,
    F3dToolStatus,
    NvidiaProbe,
    OrcaToolStatus,
    PackageStatus,
    PythonInfo,
    ToolsBlock,
    UvTooling,
    VramInfo,
)

runner = CliRunner()


def _sample_report(*, ok: bool = True, required: list[str] | None = None) -> DoctorReport:
    return DoctorReport(
        ok=ok,
        python=PythonInfo(version="3.13.0", executable="/py", pin_ok=True),
        packages={
            "trimesh": PackageStatus(import_ok=True, version="4.12.2"),
            "f3d": PackageStatus(import_ok=True, version="3.5.0"),
        },
        tools=ToolsBlock(
            blender=BlenderToolStatus(status="missing", source="missing"),
            orca=OrcaToolStatus(status="missing", source="missing", version_source="missing"),
            f3d=F3dToolStatus(import_ok=True, version="3.5.0"),
        ),
        tooling=UvTooling(version="0.12.0", uv_lock_present=True),
        disk=DiskInfo(pymeshlab_approx_mb=140.0),
        licenses=["pymeshlab: GPL-3.0 linked"],
        env={"MESHOPS_BLENDER": False},
        env_catalog=[],
        hints=["hint one"],
        vram=VramInfo(
            ritual="ritual text",
            nvidia=NvidiaProbe(status="no_nvidia_gpu"),
        ),
        required=required or ["core"],
    )


def test_doctor_help_mentions_cold_start() -> None:
    r = runner.invoke(app, ["doctor", "--help"])
    assert r.exit_code == 0
    text = (r.stdout + r.stderr).lower()
    assert "cold" in text or "second" in text or "native" in text


def test_doctor_in_app_help() -> None:
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "doctor" in r.stdout.lower()


def test_doctor_json_fields_and_exit_0() -> None:
    report = _sample_report(ok=True)
    # doctor_cmd does `from meshops.ops.doctor import run_doctor` — patch module attr.
    import meshops.ops.doctor as doctor_mod

    with patch.object(doctor_mod, "run_doctor", return_value=report):
        r = runner.invoke(app, ["doctor", "--json"])
    assert r.exit_code == 0, r.stdout + r.stderr
    data = json.loads(r.stdout)
    assert data["ok"] is True
    assert data["schema_version"] == "1.0.0"
    assert "tools" in data
    assert "env_catalog" in data
    assert "hints" in data
    assert data["tools"]["blender"]["status"] == "missing"


def test_doctor_exit_1_when_not_ok() -> None:
    report = _sample_report(ok=False, required=["core", "blender"])
    import meshops.ops.doctor as doctor_mod

    with patch.object(doctor_mod, "run_doctor", return_value=report):
        r = runner.invoke(app, ["doctor", "--json", "--require", "blender"])
    assert r.exit_code == 1
    data = json.loads(r.stdout)
    assert data["ok"] is False


def test_doctor_strict_passes_require_set() -> None:
    report = _sample_report(ok=False, required=["blender", "core", "orca"])
    import meshops.ops.doctor as doctor_mod

    captured: dict[str, object] = {}

    def _capture(*, require=None, work_root=None, cwd=None):  # type: ignore[no-untyped-def]
        captured["require"] = set(require) if require is not None else None
        return report

    with patch.object(doctor_mod, "run_doctor", side_effect=_capture):
        r = runner.invoke(app, ["doctor", "--json", "--strict"])
    assert r.exit_code == 1
    req = captured.get("require")
    assert isinstance(req, set)
    assert {"core", "blender", "orca"} <= req


def test_doctor_require_repeatable() -> None:
    report = _sample_report(ok=True, required=["core", "design"])
    import meshops.ops.doctor as doctor_mod

    captured: dict[str, object] = {}

    def _capture(*, require=None, work_root=None, cwd=None):  # type: ignore[no-untyped-def]
        captured["require"] = set(require) if require is not None else None
        return report

    with patch.object(doctor_mod, "run_doctor", side_effect=_capture):
        r = runner.invoke(app, ["doctor", "--json", "--require", "design", "--require", "core"])
    assert r.exit_code == 0
    req = captured.get("require")
    assert isinstance(req, set)
    assert "design" in req
    assert "core" in req
