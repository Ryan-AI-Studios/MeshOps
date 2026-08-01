"""Doctor unit tests with monkeypatched finders (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from meshops.ops.doctor import expand_require, run_doctor
from meshops.ops.models import DoctorReport


def _fake_blender_missing() -> tuple[None, str]:
    return None, "missing"


def _fake_blender_ok(path: Path) -> Any:
    def _find(*, require: bool = True) -> tuple[Path | None, str]:
        return path, "env"

    return _find


def test_expand_require_defaults() -> None:
    assert expand_require(None) == {"core"}
    assert expand_require([]) == {"core"}
    with pytest.raises(ValueError):
        expand_require(["strict"])
    assert expand_require(["blender"]) == {"core", "blender"}
    assert expand_require(["all"]) >= {"core", "blender", "orca", "design"}


def test_default_ok_when_blender_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "meshops.ops.doctor._probe_blender",
        lambda: __import__("meshops.ops.models", fromlist=["BlenderToolStatus"]).BlenderToolStatus(
            status="missing", source="missing"
        ),
    )
    monkeypatch.setattr(
        "meshops.ops.doctor._probe_orca",
        lambda: __import__("meshops.ops.models", fromlist=["OrcaToolStatus"]).OrcaToolStatus(
            status="missing", source="missing", version_source="missing"
        ),
    )
    monkeypatch.setattr(
        "meshops.ops.doctor._probe_nvidia",
        lambda: __import__("meshops.ops.models", fromlist=["NvidiaProbe"]).NvidiaProbe(
            status="no_nvidia_gpu"
        ),
    )
    report = run_doctor(require={"core"})
    assert isinstance(report, DoctorReport)
    assert report.schema_version == "1.0.0"
    # Core should pass on a real dev env; if not, still assert structure
    assert report.tools.blender.status == "missing"
    # Default require core: ok depends on packages
    if report.python.pin_ok and report.tools.f3d.import_ok:
        assert report.ok is True


def test_require_blender_fails_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from meshops.ops.models import BlenderToolStatus, NvidiaProbe, OrcaToolStatus

    monkeypatch.setattr(
        "meshops.ops.doctor._probe_blender",
        lambda: BlenderToolStatus(status="missing", source="missing"),
    )
    monkeypatch.setattr(
        "meshops.ops.doctor._probe_orca",
        lambda: OrcaToolStatus(status="missing", source="missing", version_source="missing"),
    )
    monkeypatch.setattr(
        "meshops.ops.doctor._probe_nvidia",
        lambda: NvidiaProbe(status="no_nvidia_gpu"),
    )
    report = run_doctor(require={"core", "blender"})
    assert report.ok is False
    assert "blender" in report.required


def test_strict_requires_blender_and_orca(monkeypatch: pytest.MonkeyPatch) -> None:
    from meshops.ops.models import BlenderToolStatus, NvidiaProbe, OrcaToolStatus

    monkeypatch.setattr(
        "meshops.ops.doctor._probe_blender",
        lambda: BlenderToolStatus(
            path=r"C:\fake\blender.exe",
            version="5.2.0",
            pin_ok=True,
            status="ok",
            source="env",
        ),
    )
    monkeypatch.setattr(
        "meshops.ops.doctor._probe_orca",
        lambda: OrcaToolStatus(
            path=r"C:\fake\orca-slicer.exe",
            version="2.4.2",
            soft_pin_ok=True,
            status="ok",
            source="well_known",
            version_source="appdata",
        ),
    )
    monkeypatch.setattr(
        "meshops.ops.doctor._probe_nvidia",
        lambda: NvidiaProbe(status="no_nvidia_gpu"),
    )
    report = run_doctor(require=expand_require(["core", "blender", "orca"]))
    assert "blender" in report.required
    assert "orca" in report.required
    if report.python.pin_ok and report.tools.f3d.import_ok:
        assert report.ok is True


def test_json_required_fields_present() -> None:
    with (
        patch("meshops.ops.doctor._probe_blender") as pb,
        patch("meshops.ops.doctor._probe_orca") as po,
        patch("meshops.ops.doctor._probe_nvidia") as pn,
    ):
        from meshops.ops.models import BlenderToolStatus, NvidiaProbe, OrcaToolStatus

        pb.return_value = BlenderToolStatus(status="missing", source="missing")
        po.return_value = OrcaToolStatus(
            status="missing", source="missing", version_source="missing"
        )
        pn.return_value = NvidiaProbe(status="no_nvidia_gpu")
        report = run_doctor(require={"core"})
        data = report.model_dump(mode="json")

    required_top = {
        "schema_version",
        "ok",
        "python",
        "packages",
        "tools",
        "tooling",
        "disk",
        "licenses",
        "env",
        "env_catalog",
        "hints",
        "vram",
        "required",
    }
    assert required_top <= set(data.keys())
    assert data["schema_version"] == "1.0.0"
    assert "version" in data["python"]
    assert "pin_ok" in data["python"]
    assert "blender" in data["tools"]
    assert "orca" in data["tools"]
    assert "f3d" in data["tools"]
    assert "version_source" in data["tools"]["orca"]
    assert "source" in data["tools"]["blender"]
    assert "uv_lock_present" in data["tooling"]
    assert "pymeshlab_approx_mb" in data["disk"]
    assert isinstance(data["env_catalog"], list)
    assert len(data["env_catalog"]) >= 1
    assert "name" in data["env_catalog"][0]
    assert isinstance(data["hints"], list)
    assert "ritual" in data["vram"]
    assert "nvidia" in data["vram"]


def test_orca_version_source_path_only_ok_for_require(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meshops.ops.models import BlenderToolStatus, NvidiaProbe, OrcaToolStatus

    monkeypatch.setattr(
        "meshops.ops.doctor._probe_blender",
        lambda: BlenderToolStatus(status="missing", source="missing"),
    )
    monkeypatch.setattr(
        "meshops.ops.doctor._probe_orca",
        lambda: OrcaToolStatus(
            path=r"C:\Program Files\OrcaSlicer\orca-slicer.exe",
            version=None,
            soft_pin_ok=True,
            status="warn",
            source="well_known",
            version_source="path_only",
        ),
    )
    monkeypatch.setattr(
        "meshops.ops.doctor._probe_nvidia",
        lambda: NvidiaProbe(status="no_nvidia_gpu"),
    )
    report = run_doctor(require={"core", "orca"})
    # Path present → orca requirement passes even if version unknown
    if report.python.pin_ok and report.tools.f3d.import_ok:
        assert report.ok is True
    assert report.tools.orca.version_source == "path_only"
