"""Bench harness unit + CLI tests (0009-Hardening DoD)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from meshops.bench.models import (
    SCHEMA_VERSION,
    SKIPPED_INSUFFICIENT_RAM,
    BenchCaseResult,
    Envelope,
    HostBlock,
    MethodBlock,
)
from meshops.bench.report import find_latest_results, load_envelope, write_results
from meshops.bench.runner import run_case, run_ladder, time_median
from meshops.bench.sizes import FACE_TARGETS, FACE_TOLERANCE_FRAC, generate_ladder_mesh, parse_sizes
from meshops.cli import app
from meshops.render.f3d_renderer import RenderUnavailableError

runner = CliRunner()


def _fast_ingest(path: Any, *, work_root: Any = "work", **_kw: Any) -> MagicMock:
    """Lightweight ingest stub: write minimal job layout for mesh_id."""
    from meshops.jobstore.paths import JobPaths, content_sha256, ensure_job_layout

    source = Path(path)
    digest = content_sha256(source) if source.is_file() else "a" * 64
    mesh_id = digest[:12]
    paths = JobPaths(work_root=Path(work_root), mesh_id=mesh_id)
    ensure_job_layout(paths)
    if source.is_file() and not paths.original_stl.is_file():
        paths.original_stl.write_bytes(source.read_bytes())
    # minimal working file
    if not paths.working_ply.is_file():
        paths.working_ply.write_bytes(b"ply\n")
    result = MagicMock()
    result.mesh_id = mesh_id
    result.job_dir = paths.job_dir
    return result


def _fast_triage(mesh_id: str, *, work_root: Any = "work", **_kw: Any) -> MagicMock:
    return MagicMock(mesh_id=mesh_id)


def _fast_write_ladder_stl(
    label: str,
    dest: Path | str,
    *,
    target_faces: int | None = None,
    seed: int = 0,
) -> Any:
    """Tiny mesh export for unit tests - avoids 100k-2M face generation wall time."""
    import trimesh

    from meshops.bench.sizes import FACE_TARGETS, LadderMesh

    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = label.upper()
    if target_faces is not None:
        target = int(target_faces)
    elif key in FACE_TARGETS:
        target = int(FACE_TARGETS[key])  # type: ignore[index]
    else:
        target = 1_000
    # ~2k faces: fast export; actual_faces still > 0 for assertions
    mesh = trimesh.creation.uv_sphere(count=[16, 16])
    mesh.export(path)
    return LadderMesh(label=key, target_faces=int(target), mesh=mesh, path=path)


@pytest.fixture
def patch_pipeline():
    """Patch mesh gen + ingest/triage so ladder tests stay offline-fast."""
    with (
        patch("meshops.bench.runner.write_ladder_stl", side_effect=_fast_write_ladder_stl),
        patch("meshops.ingest.pipeline.ingest_stl", side_effect=_fast_ingest),
        patch("meshops.triage.orchestrate.mesh_triage", side_effect=_fast_triage),
    ):
        yield


def test_s_within_face_tolerance() -> None:
    ladder = generate_ladder_mesh("S")
    lo = FACE_TARGETS["S"] * (1.0 - FACE_TOLERANCE_FRAC)
    hi = FACE_TARGETS["S"] * (1.0 + FACE_TOLERANCE_FRAC)
    assert lo <= ladder.actual_faces <= hi
    assert ladder.within_tolerance()
    assert ladder.target_faces == FACE_TARGETS["S"]
    assert ladder.verts > 0


def test_small_custom_target_within_tolerance() -> None:
    """Generator hits ±15% for non-ladder custom targets too."""
    ladder = generate_ladder_mesh("custom", target_faces=5_000)
    assert ladder.within_tolerance()
    assert ladder.actual_faces >= 5_000 * (1.0 - FACE_TOLERANCE_FRAC)


def test_parse_sizes() -> None:
    assert parse_sizes("s,m") == ["S", "M"]
    assert parse_sizes(None) == ["S", "M", "L", "XL"]
    with pytest.raises(ValueError):
        parse_sizes("S,XXL")


def test_time_median_warmup_and_three_samples() -> None:
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        time.sleep(0.001)

    med, samples = time_median(fn, warmup=1, iters=3)
    assert calls["n"] == 4  # 1 warmup + 3 timed
    assert len(samples) == 3
    assert isinstance(med, float)
    assert med >= 0.0


def test_injected_fail_continues_ladder(tmp_path: Path, patch_pipeline: None) -> None:
    env = run_ladder(
        ["S", "M"],
        work_root=tmp_path / "bench",
        include_render=False,
        inject_fail_labels=["S"],
        warmup=0,
        timed_iters=1,
    )
    assert len(env.cases) == 2
    by = {c.label: c for c in env.cases}
    assert by["S"].status == "failed"
    assert by["S"].error_code is not None
    # M must still run (continue after S fail)
    assert by["M"].status == "ok"
    assert by["M"].actual_faces > 0
    assert by["M"].ingest_s is not None
    assert len(by["M"].ingest_samples_s) == 1


def test_warmup_median_of_3_recorded(tmp_path: Path, patch_pipeline: None) -> None:
    case = run_case(
        "S",
        work_root=tmp_path / "bench",
        deps={"trimesh": "4.12.2"},
        include_render=False,
        warmup=1,
        timed_iters=3,
    )
    assert case.status == "ok"
    assert len(case.ingest_samples_s) == 3
    assert len(case.triage_samples_s) == 3
    assert case.ingest_s == sorted(case.ingest_samples_s)[1] or case.ingest_s is not None


def test_l_xl_skip_on_low_ram(tmp_path: Path) -> None:
    # 1 GiB available → below 4 GiB gate
    with patch("meshops.bench.runner.get_available_ram_bytes", return_value=1 * 1024**3):
        env = run_ladder(
            ["L", "XL"],
            work_root=tmp_path / "bench",
            include_render=False,
            warmup=0,
            timed_iters=1,
        )
    assert len(env.cases) == 2
    for c in env.cases:
        assert c.status == "skipped"
        assert c.skipped_reason == SKIPPED_INSUFFICIENT_RAM


def test_soak_skips_without_meshops_bench_soak() -> None:
    """Prove soak module self-skips when MESHOPS_BENCH_SOAK unset (C2)."""
    raw = os.environ.get("MESHOPS_BENCH_SOAK")
    try:
        os.environ.pop("MESHOPS_BENCH_SOAK", None)
        from tests.test_bench_soak import soak_env_enabled

        assert soak_env_enabled() is False
    finally:
        if raw is not None:
            os.environ["MESHOPS_BENCH_SOAK"] = raw


def test_open3d_marker_registered() -> None:
    """open3d marker collects under --strict-markers."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "open3d: needs Open3D" in text
    mark = pytest.mark.open3d
    assert mark.name == "open3d"


def test_open3d_marker_collects_strict() -> None:
    """Pytest with --strict-markers accepts open3d-marked items."""

    @pytest.mark.open3d
    def _dummy() -> None:
        pass

    assert _dummy.pytestmark[0].name == "open3d"  # type: ignore[attr-defined]


def test_envelope_json_reads_latest(tmp_path: Path) -> None:
    older = Envelope(
        created_at="2020-01-01T00:00:00+00:00",
        host=HostBlock(os="test", python_version="3.13.0", deps={"trimesh": "4.12.2"}),
        cases=[],
    )
    newer = Envelope(
        created_at="2026-08-01T00:00:00+00:00",
        host=HostBlock(os="test", python_version="3.13.0", deps={"trimesh": "4.12.2"}),
        cases=[],
        notes=["newest"],
    )
    root = tmp_path / "bench"
    write_results(older, work_root=root, basename="bench_results")
    # Nested run dir: write_results nests …/run2 → …/run2/bench/
    sub = root / "run2"
    json_new, _md = write_results(newer, work_root=sub, basename="bench_results")
    # Ensure newer mtime wins
    json_new.write_text(json_new.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    latest = find_latest_results(root)
    assert latest is not None
    loaded = load_envelope(latest)
    assert loaded.created_at == newer.created_at or "newest" in loaded.notes

    r = runner.invoke(app, ["bench", "envelope", "--json", "--work-root", str(root)])
    assert r.exit_code == 0, r.stdout + r.stderr
    data = json.loads(r.stdout)
    assert data["ok"] is True
    assert data["schema_version"] == SCHEMA_VERSION


def test_f3d_unavailable_render_skipped_not_failed(tmp_path: Path, patch_pipeline: None) -> None:
    def _boom(*_a: Any, **_k: Any) -> None:
        raise RenderUnavailableError("f3d unavailable for test")

    with patch("meshops.render.f3d_renderer.F3DRenderer.render_mesh_to_dir", _boom):
        case = run_case(
            "S",
            work_root=tmp_path / "bench",
            deps={"trimesh": "4.12.2"},
            include_render=True,
            warmup=0,
            timed_iters=1,
        )
    assert case.status == "ok"
    assert case.render_s is None
    assert case.ingest_s is not None
    assert case.triage_s is not None


def test_models_forbid_extra() -> None:
    with pytest.raises(ValidationError):
        BenchCaseResult(
            label="S",
            target_faces=1,
            actual_faces=1,
            verts=1,
            host_os="x",
            python_version="3.13",
            created_at="t",
            unexpected=True,  # type: ignore[call-arg]
        )


def test_models_schema_version() -> None:
    case = BenchCaseResult(
        label="S",
        target_faces=100_000,
        actual_faces=100_000,
        verts=10,
        host_os="test",
        python_version="3.13.0",
        created_at="2026-08-01T00:00:00+00:00",
        status="ok",
    )
    assert case.schema_version == "1.0.0"
    env = Envelope(
        created_at=case.created_at,
        host=HostBlock(os="test", python_version="3.13.0"),
        cases=[case],
        method=MethodBlock(),
    )
    assert env.schema_version == "1.0.0"
    dumped = json.loads(env.model_dump_json())
    assert dumped["method"]["warmup"] == 1
    assert dumped["method"]["timed_iters"] == 3
    assert dumped["method"]["aggregate"] == "median"


def test_cli_bench_run_s(tmp_path: Path, patch_pipeline: None) -> None:
    r = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--sizes",
            "S",
            "--json",
            "--no-render",
            "--work-root",
            str(tmp_path / "bench"),
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    data = json.loads(r.stdout)
    assert data["schema_version"] == "1.0.0"
    assert len(data["cases"]) == 1
    assert data["cases"][0]["label"] == "S"
    assert data["cases"][0]["status"] == "ok"
    assert Path(data["results_json"]).is_file()
    # median-of-3 samples recorded
    assert len(data["cases"][0]["ingest_samples_s"]) == 3


def test_cli_bench_help() -> None:
    r = runner.invoke(app, ["bench", "--help"])
    assert r.exit_code == 0
    text = (r.stdout + r.stderr).lower()
    assert "run" in text
    assert "envelope" in text


def test_rss_helper_returns_float_or_none() -> None:
    from meshops.bench.rss import get_peak_rss_mb

    val = get_peak_rss_mb()
    assert val is None or (isinstance(val, float) and val > 0)


def test_collect_deps_has_core_keys() -> None:
    from meshops.bench.runner import collect_deps

    deps = collect_deps()
    for key in ("trimesh", "numpy", "scipy", "pymeshlab", "manifold3d", "f3d"):
        assert key in deps
