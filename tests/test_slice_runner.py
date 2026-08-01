"""Argv builder + mockable run_slice (DoD-2,4,14)."""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from meshops.slice.errors import SliceError
from meshops.slice.models import ProfilePaths
from meshops.slice.profiles import resolve_profiles
from meshops.slice.runner import build_orca_argv, make_run_id, run_slice

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "slice"


def _ok_3mf_bytes() -> bytes:
    """Build in-memory 3mf with ok slice_info."""
    import io

    buf = io.BytesIO()
    xml = (FIXTURES / "slice_info_ok.config").read_text(encoding="utf-8")
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Metadata/slice_info.config", xml)
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
    return buf.getvalue()


def test_make_run_id_format() -> None:
    rid = make_run_id()
    assert re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{8}", rid)


def test_build_orca_argv_spaces_and_settings(tmp_path: Path) -> None:
    orca = tmp_path / "orca-slicer.exe"
    orca.write_bytes(b"x")
    space_dir = tmp_path / "path with spaces"
    space_dir.mkdir()
    machine = space_dir / "machine.json"
    process = space_dir / "process.json"
    filament = space_dir / "filament.json"
    for p in (machine, process, filament):
        p.write_text("{}", encoding="utf-8")
    stl = space_dir / "model part.stl"
    stl.write_bytes(b"solid x\nendsolid x\n")
    out = space_dir / "out.gcode.3mf"

    profiles = ProfilePaths(
        machine=str(machine),
        process=str(process),
        filament=str(filament),
        profile_name="test",
    )
    argv = build_orca_argv(
        orca=orca,
        input_stl=stl,
        output_3mf=out,
        profiles=profiles,
        orient=0,
        arrange=0,
        plate=1,
    )
    assert argv[0] == str(orca.resolve())
    assert "--load-settings" in argv
    idx = argv.index("--load-settings")
    settings = argv[idx + 1]
    # ONE element with semicolon between absolute paths
    assert ";" in settings
    assert settings.count(";") == 1
    assert str(machine.resolve()) in settings
    assert str(process.resolve()) in settings
    # Not two separate argv elements after the flag
    assert argv[idx + 2] == "--load-filaments"
    assert "--slice" in argv
    assert argv[argv.index("--slice") + 1] == "1"
    assert "--arrange" in argv
    assert argv[argv.index("--arrange") + 1] == "0"
    assert "--outputdir" not in argv
    assert argv[-1] == str(stl.resolve())
    assert "--export-3mf" in argv


def test_build_orca_argv_datadir(tmp_path: Path) -> None:
    orca = tmp_path / "orca.exe"
    orca.write_bytes(b"x")
    for name in ("machine.json", "process.json", "filament.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    stl = tmp_path / "a.stl"
    stl.write_bytes(b"solid\nendsolid\n")
    out = tmp_path / "o.gcode.3mf"
    dd = tmp_path / "datadir"
    dd.mkdir()
    profiles = ProfilePaths(
        machine=str(tmp_path / "machine.json"),
        process=str(tmp_path / "process.json"),
        filament=str(tmp_path / "filament.json"),
        datadir=str(dd),
    )
    argv = build_orca_argv(
        orca=orca,
        input_stl=stl,
        output_3mf=out,
        profiles=profiles,
    )
    assert "--datadir" in argv
    assert argv[argv.index("--datadir") + 1] == str(dd.resolve())


def test_run_slice_mock_success(
    tmp_path: Path, solid_cylinder_stl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_orca = tmp_path / "orca-slicer.exe"
    fake_orca.write_bytes(b"x")
    monkeypatch.setenv("MESHOPS_ORCA", str(fake_orca))

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        # Write 3mf at --export-3mf path
        out_idx = argv.index("--export-3mf")
        out_path = Path(argv[out_idx + 1])
        out_path.write_bytes(_ok_3mf_bytes())
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    result = run_slice(
        solid_cylinder_stl,
        mesh_id="abc123def456",
        work_root=tmp_path / "work",
        run_orca_fn=fake_run,
        orca_path=fake_orca,
        load_volume=True,
    )
    assert result.status == "pass"
    assert result.accept is not None
    assert result.accept.status == "pass"
    assert result.run_dir is not None
    run_dir = Path(result.run_dir)
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "slice_report.md").is_file()
    assert (run_dir / "output.gcode.3mf").is_file()
    assert (run_dir / "orca_stdout.log").is_file()
    report = (run_dir / "slice_report.md").read_text(encoding="utf-8")
    assert "Slice report" in report
    assert "1.24" in report
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["schema_version"] == "1.0.0"
    assert man["profile_paths"]["machine"]
    assert re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{8}", result.run_id)


def test_run_slice_nonzero_exit_with_3mf_fails(tmp_path: Path, solid_cylinder_stl: Path) -> None:
    """Codex P1-001: non-zero Orca exit must fail even if a parseable 3mf exists."""
    fake_orca = tmp_path / "orca.exe"
    fake_orca.write_bytes(b"x")

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        out_idx = argv.index("--export-3mf")
        Path(argv[out_idx + 1]).write_bytes(_ok_3mf_bytes())
        return subprocess.CompletedProcess(argv, 2, stdout="partial", stderr="warn")

    result = run_slice(
        solid_cylinder_stl,
        mesh_id="nonzeroexit001",
        work_root=tmp_path / "work",
        run_orca_fn=fake_run,
        orca_path=fake_orca,
        load_volume=False,
    )
    assert result.accept is not None
    assert result.accept.status == "fail"
    assert result.accept.error_code == "slice_failed"
    assert result.status in ("fail", "error")


def test_invalid_run_id_rejected(tmp_path: Path, solid_cylinder_stl: Path) -> None:
    """Codex P2-004: reject path traversal / non-canonical run_id."""
    fake_orca = tmp_path / "orca.exe"
    fake_orca.write_bytes(b"x")
    with pytest.raises(SliceError) as ei:
        run_slice(
            solid_cylinder_stl,
            mesh_id="runidbad00001",
            work_root=tmp_path / "work",
            run_orca_fn=lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""),
            orca_path=fake_orca,
            load_volume=False,
            run_id="../escape",
        )
    assert ei.value.code == "slice_failed"


def test_run_slice_missing_3mf_fail(tmp_path: Path, solid_cylinder_stl: Path) -> None:
    fake_orca = tmp_path / "orca.exe"
    fake_orca.write_bytes(b"x")

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="fail")

    result = run_slice(
        solid_cylinder_stl,
        mesh_id="deadbeef0001",
        work_root=tmp_path / "work",
        run_orca_fn=fake_run,
        orca_path=fake_orca,
        load_volume=False,
    )
    assert result.status == "fail"
    assert result.accept is not None
    assert result.accept.status == "fail"
    assert result.accept.error_code in ("missing_3mf", "slice_failed", "parse_failed")
    assert Path(result.run_dir or ".").joinpath("slice_report.md").is_file()


def test_unique_run_dirs(tmp_path: Path, solid_cylinder_stl: Path) -> None:
    fake_orca = tmp_path / "orca.exe"
    fake_orca.write_bytes(b"x")

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        out_idx = argv.index("--export-3mf")
        Path(argv[out_idx + 1]).write_bytes(_ok_3mf_bytes())
        return subprocess.CompletedProcess(argv, 0, "", "")

    r1 = run_slice(
        solid_cylinder_stl,
        mesh_id="sameid000001",
        work_root=tmp_path / "work",
        run_orca_fn=fake_run,
        orca_path=fake_orca,
        load_volume=False,
    )
    r2 = run_slice(
        solid_cylinder_stl,
        mesh_id="sameid000001",
        work_root=tmp_path / "work",
        run_orca_fn=fake_run,
        orca_path=fake_orca,
        load_volume=False,
    )
    assert r1.run_id != r2.run_id
    assert r1.run_dir != r2.run_dir
    assert Path(r1.run_dir or "").is_dir()
    assert Path(r2.run_dir or "").is_dir()


def test_run_slice_orca_not_found(
    tmp_path: Path, solid_cylinder_stl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MESHOPS_ORCA", raising=False)
    monkeypatch.delenv("MESHOPS_ORCASLICER", raising=False)
    monkeypatch.setattr(
        "meshops.slice.runner.find_orca",
        lambda require=False: None,
    )
    with pytest.raises(SliceError) as ei:
        run_slice(solid_cylinder_stl, work_root=tmp_path, load_volume=False)
    assert ei.value.code == "orca_not_found"


def test_default_profiles_usable_in_argv() -> None:
    pp = resolve_profiles("default")
    # Ensure paths work with build when files exist
    assert Path(pp.machine).is_file()


def test_timeout_raises(tmp_path: Path, solid_cylinder_stl: Path) -> None:
    fake_orca = tmp_path / "orca.exe"
    fake_orca.write_bytes(b"x")

    def boom(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise SliceError("timeout", code="slice_timeout")

    # run_slice catches via early_code when invoker raises? Currently invoker
    # exceptions from run_orca_fn are not wrapped — only run_orca raises SliceError.
    # Our fake raises SliceError; runner._one_pass catches SliceError.
    result = run_slice(
        solid_cylinder_stl,
        mesh_id="timeout000001",
        work_root=tmp_path / "work",
        run_orca_fn=boom,  # type: ignore[arg-type]
        orca_path=fake_orca,
        load_volume=False,
    )
    assert result.error_code == "slice_timeout" or (
        result.accept is not None and result.accept.error_code == "slice_timeout"
    )


def test_run_slice_preserves_ply_suffix(tmp_path: Path, solid_cylinder_stl: Path) -> None:
    """Post-promote working.ply must not be mislabeled as input.stl (P2-001)."""
    import trimesh

    fake_orca = tmp_path / "orca.exe"
    fake_orca.write_bytes(b"x")
    mesh = trimesh.load(str(solid_cylinder_stl), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    ply = tmp_path / "working.ply"
    mesh.export(str(ply))

    staged: list[str] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        model = argv[-1]
        staged.append(model)
        out_idx = argv.index("--export-3mf")
        Path(argv[out_idx + 1]).write_bytes(_ok_3mf_bytes())
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = run_slice(
        ply,
        mesh_id="plysuffix0001",
        work_root=tmp_path / "work",
        run_orca_fn=fake_run,
        orca_path=fake_orca,
        load_volume=False,
    )
    assert result.status == "pass"
    assert staged
    assert staged[0].lower().endswith(".ply")
    assert Path(staged[0]).is_file()


def test_allow_reorient_retry_once_max(tmp_path: Path, solid_cylinder_stl: Path) -> None:
    """allow_reorient_retry re-invokes at most once with --orient 1."""
    fake_orca = tmp_path / "orca.exe"
    fake_orca.write_bytes(b"x")
    calls: list[list[str]] = []

    high_xml = """<?xml version="1.0"?>
<config>
  <header><header_item key="OrcaSlicer-Version" value="2.4.2"/></header>
  <plate>
    <metadata key="index" value="1"/>
    <metadata key="prediction" value="100"/>
    <metadata key="weight" value="1000"/>
    <metadata key="support_used" value="true"/>
    <metadata key="outside" value="false"/>
    <filament id="1" used_m="100" used_g="1000" type="PLA"/>
  </plate>
</config>
"""
    ok_xml = (FIXTURES / "slice_info_ok.config").read_text(encoding="utf-8")

    def _zip_xml(xml: str) -> bytes:
        import io

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("Metadata/slice_info.config", xml)
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
        return buf.getvalue()

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        out_idx = argv.index("--export-3mf")
        out_path = Path(argv[out_idx + 1])
        # First call: high filament → anomaly high; second: ok
        xml = high_xml if len(calls) == 1 else ok_xml
        out_path.write_bytes(_zip_xml(xml))
        return subprocess.CompletedProcess(argv, 0, "", "")

    # Volume such that high-filament first pass fails high, ok second pass stays under fail_high.
    # high: (1000/1.24)/2 ≈ 403 >= 8; ok fixture: ~11.46/2 ≈ 5.7 (warn band, still pass).
    result = run_slice(
        solid_cylinder_stl,
        mesh_id="reorient00001",
        work_root=tmp_path / "work",
        run_orca_fn=fake_run,
        orca_path=fake_orca,
        load_volume=False,
        mesh_volume_cm3=2.0,
        allow_reorient_retry=True,
        orient=0,
    )
    assert len(calls) == 2
    assert "--orient" in calls[0] and calls[0][calls[0].index("--orient") + 1] == "0"
    assert "--orient" in calls[1] and calls[1][calls[1].index("--orient") + 1] == "1"
    assert result.metrics.get("slice.reorient_retry_used") is True
    first = result.metrics.get("slice.reorient_first_attempt")
    assert isinstance(first, dict)
    assert first.get("error_code") == "filament_anomaly_high"
    assert first.get("archive_dir")
    assert Path(str(first["archive_dir"])).joinpath("manifest.json").is_file()
    # Second pass uses ok fixture → pass
    assert result.accept is not None
    assert result.accept.status == "pass"


def test_mock_magic_unused() -> None:
    # keep import used if needed
    assert MagicMock is not None
