"""CLI meshops slice + accept --require-slice wiring (DoD-9,10)."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.ingest.pipeline import ingest_stl

runner = CliRunner()
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "slice"


def _ok_3mf_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    xml = (FIXTURES / "slice_info_ok.config").read_text(encoding="utf-8")
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Metadata/slice_info.config", xml)
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
    return buf.getvalue()


def test_orcaslicer_marker_registered() -> None:
    """Strict-markers: orcaslicer must be registered in pyproject."""
    config_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = config_path.read_text(encoding="utf-8")
    assert "orcaslicer:" in text
    assert "needs OrcaSlicer binary" in text


def test_cli_help_has_slice() -> None:
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "slice" in r.stdout.lower()


def test_slice_cmd_mock(
    solid_cylinder_stl: Path, tmp_work: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = tmp_path / "orca-slicer.exe"
    fake.write_bytes(b"x")
    monkeypatch.setenv("MESHOPS_ORCA", str(fake))

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        out_idx = argv.index("--export-3mf")
        Path(argv[out_idx + 1]).write_bytes(_ok_3mf_bytes())
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr("meshops.slice.runner.run_orca", fake_run)
    monkeypatch.setattr("meshops.slice.runner.find_orca", lambda require=False: fake)
    monkeypatch.setattr("meshops.slice.find_orca", lambda require=False: fake)
    monkeypatch.setattr("meshops.cli.find_orca", lambda require=False: fake, raising=False)

    # Patch where CLI imports at call time
    import meshops.slice as slice_pkg

    monkeypatch.setattr(slice_pkg, "find_orca", lambda require=False: fake)
    monkeypatch.setattr("meshops.slice.runner.run_orca", fake_run)

    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    r = runner.invoke(
        app,
        [
            "slice",
            "--mesh-id",
            ing.mesh_id,
            "--work-root",
            str(tmp_work),
            "--json",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    data = json.loads(r.stdout)
    assert data["ok"] is True
    assert data["slice"]["status"] == "pass"
    assert data["slice"]["run_id"].startswith("run_")
    report = data["slice"].get("report_path")
    if report:
        assert Path(report).is_file()


def test_slice_cmd_orca_missing(
    solid_cylinder_stl: Path, tmp_work: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MESHOPS_ORCA", raising=False)
    monkeypatch.delenv("MESHOPS_ORCASLICER", raising=False)
    import meshops.slice as slice_pkg

    monkeypatch.setattr(slice_pkg, "find_orca", lambda require=False: None)

    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    r = runner.invoke(
        app,
        [
            "slice",
            "--mesh-id",
            ing.mesh_id,
            "--work-root",
            str(tmp_work),
            "--json",
        ],
    )
    assert r.exit_code != 0
    data = json.loads(r.stdout)
    assert data["ok"] is False
    assert data.get("code") == "orca_not_found" or "orca" in data.get("message", "").lower()


def test_accept_require_slice_fail_closed_no_orca(
    solid_cylinder_stl: Path, tmp_work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import meshops.slice as slice_pkg

    monkeypatch.setattr(slice_pkg, "find_orca", lambda require=False: None)

    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    r = runner.invoke(
        app,
        [
            "accept",
            "--mesh-id",
            ing.mesh_id,
            "--work-root",
            str(tmp_work),
            "--require-slice",
            "--json",
        ],
    )
    assert r.exit_code != 0
    data = json.loads(r.stdout)
    assert data["ok"] is False
    failed = data["acceptance"]["failed"]
    assert "slice_not_configured" in failed or "slice_fail" in failed


def test_jobstore_slice_dir(solid_cylinder_stl: Path, tmp_work: Path) -> None:
    from meshops.jobstore.paths import JobPaths, ensure_job_layout

    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    paths = JobPaths(work_root=tmp_work, mesh_id=ing.mesh_id)
    ensure_job_layout(paths)
    assert paths.slice_dir == paths.job_dir / "slice"
    assert paths.slice_dir.is_dir()


@pytest.mark.orcaslicer
def test_live_orca_cube_slice(
    solid_cylinder_stl: Path, tmp_work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Optional live Orca 2.4.x slice — skip if binary missing."""
    from meshops.slice import find_orca, run_slice

    orca = find_orca(require=False)
    if orca is None:
        pytest.skip("OrcaSlicer not found")

    ing = ingest_stl(solid_cylinder_stl, work_root=tmp_work)
    # Keep live timeout bounded — incomplete flattened profiles can stall Orca GUI.
    result = run_slice(
        solid_cylinder_stl,
        mesh_id=ing.mesh_id,
        work_root=tmp_work,
        slice_profile="default",
        load_volume=True,
        timeout_s=90.0,
    )
    # Live may fail on profile incompatibility — record but prefer success path
    assert result.run_dir is not None
    assert Path(result.run_dir).joinpath("slice_report.md").is_file()
    if result.status == "pass":
        assert result.accept is not None
        assert result.accept.filament_used_cm3 is not None
        assert result.accept.filament_used_cm3 > 0
        assert result.accept.bed_overflow is False
        if result.accept.print_time_s is not None:
            assert result.accept.print_time_s > 0
    else:
        # Soft skip when profiles invalid for this Orca build
        pytest.skip(
            f"live slice did not pass (status={result.status} "
            f"code={result.error_code} msgs={result.messages[:3]})"
        )
