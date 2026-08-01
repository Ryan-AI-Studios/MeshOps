"""Hosted multi-view fallback tests (track 0007) — offline, no network."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from meshops.hosted import (
    HOSTED_HONESTY,
    HostedError,
    run_hosted_fallback,
    validate_plateau_gate,
)
from meshops.hosted.encode import path_to_data_uri
from meshops.hosted.gate import REQUIRED_CRITERIA, load_plateau
from meshops.hosted.orchestrate import resolve_api_key, validate_operator_justify
from meshops.hosted.providers import get_provider, list_providers
from meshops.hosted.providers.mock import MockProvider
from meshops.hosted.views import collect_view_paths
from meshops.jobstore.paths import JobPaths, ensure_job_layout

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hosted"
OPEN_PLATEAU = FIXTURES / "plateau_open.json"
CLOSED_PLATEAU = FIXTURES / "plateau_closed.json"
MOCK_STL = FIXTURES / "mock_mesh.stl"
VIEW_DIR = FIXTURES / "views"

_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

JUSTIFY_OK = "Agent metaball loop plateaued; requesting multi-view hosted regen."


def skip_if_no_hosted_key() -> None:
    """Skip live hosted tests when no API key is configured."""
    key = (
        os.environ.get("MESHOPS_HOSTED_API_KEY", "").strip()
        or os.environ.get("MESHOPS_MESHY_API_KEY", "").strip()
    )
    if not key:
        pytest.skip("MESHOPS_HOSTED_API_KEY / MESHOPS_MESHY_API_KEY not set")


# --- gate --------------------------------------------------------------------


def test_gate_open_fixture() -> None:
    rec, msgs = validate_plateau_gate(OPEN_PLATEAU)
    assert rec.allows_hosted_fallback is True
    assert REQUIRED_CRITERIA.issubset(set(rec.criteria_met))
    assert msgs == []


def test_gate_closed_fixture() -> None:
    with pytest.raises(HostedError) as ei:
        validate_plateau_gate(CLOSED_PLATEAU)
    assert ei.value.code == "plateau_gate_closed"


def test_gate_missing() -> None:
    with pytest.raises(HostedError) as ei:
        validate_plateau_gate(Path("no/such/plateau.json"))
    assert ei.value.code == "plateau_missing"


def test_gate_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "plateau.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(HostedError) as ei:
        load_plateau(bad)
    assert ei.value.code == "plateau_invalid"


def test_gate_invalid_schema(tmp_path: Path) -> None:
    bad = tmp_path / "plateau.json"
    bad.write_text(json.dumps({"schema_version": "1.0.0", "extra_field": 1}), encoding="utf-8")
    with pytest.raises(HostedError) as ei:
        load_plateau(bad)
    assert ei.value.code == "plateau_invalid"


def test_gate_session_id_mismatch_warn_only() -> None:
    rec, msgs = validate_plateau_gate(OPEN_PLATEAU, session_id="oother000001")
    assert rec.allows_hosted_fallback is True
    assert any("session_id mismatch" in m for m in msgs)


def test_gate_incomplete_criteria_even_if_allows_true(tmp_path: Path) -> None:
    data = json.loads(OPEN_PLATEAU.read_text(encoding="utf-8"))
    data["criteria_met"] = ["min_one_pass"]  # incomplete but allows true in file
    data["allows_hosted_fallback"] = True
    p = tmp_path / "plateau.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(HostedError) as ei:
        validate_plateau_gate(p)
    assert ei.value.code == "plateau_gate_closed"


# --- views / encode / justify ------------------------------------------------


def _seed_organic_tree(root: Path, *, session_id: str = "o9ea985cbc14") -> Path:
    """Create out-of-tree organic session with plateau + pass views."""
    organic = root / session_id / "organic"
    passes = organic / "passes" / "p001_simple_bust" / "views"
    passes.mkdir(parents=True)
    for key in ("front", "left", "three_quarter", "three_quarter_depth"):
        (passes / f"{key}.png").write_bytes(_MIN_PNG)
    (organic / "passes" / "p001_simple_bust" / "mesh.stl").write_bytes(MOCK_STL.read_bytes())
    plateau = json.loads(OPEN_PLATEAU.read_text(encoding="utf-8"))
    plateau["session_id"] = session_id
    (organic / "plateau.json").write_text(json.dumps(plateau, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0",
        "session_id": session_id,
        "prompt": "test figurine for hosted fallback",
        "style_notes": "",
        "ref_paths": [],
        "default_recipe": "simple_bust",
        "status": "plateau",
        "passes": ["p001_simple_bust"],
        "created_at": "2026-08-01T00:00:00+00:00",
        "updated_at": "2026-08-01T00:00:00+00:00",
        "notes": ["authored_organic_not_print_hero"],
        "max_passes": 8,
    }
    (organic / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return organic / "plateau.json"


def test_multiview_required_single_view(tmp_path: Path) -> None:
    plateau = _seed_organic_tree(tmp_path)
    single = VIEW_DIR / "front.png"
    with pytest.raises(HostedError) as ei:
        collect_view_paths(
            plateau_path=plateau,
            views_from="explicit",
            explicit_views=[single],
        )
    assert ei.value.code == "multiview_required"


def test_out_of_tree_views_resolve(tmp_path: Path) -> None:
    plateau = _seed_organic_tree(tmp_path)
    paths = collect_view_paths(plateau_path=plateau, views_from="latest")
    assert len(paths) >= 2
    for p in paths:
        assert p.is_file()
        assert str(tmp_path) in str(p)


def test_encode_data_uri_roundtrip() -> None:
    src = VIEW_DIR / "front.png"
    uri = path_to_data_uri(src)
    assert uri.startswith("data:image/png;base64,")
    import base64

    b64 = uri.split(",", 1)[1]
    assert base64.b64decode(b64) == src.read_bytes()


def test_justify_short_and_filler() -> None:
    with pytest.raises(HostedError) as ei:
        validate_operator_justify("too short")
    assert ei.value.code == "justify_invalid"
    with pytest.raises(HostedError) as ei2:
        validate_operator_justify("done")
    assert ei2.value.code == "justify_invalid"
    assert validate_operator_justify(JUSTIFY_OK) == JUSTIFY_OK


# --- mock provider -----------------------------------------------------------


def test_mock_submit_poll_download(tmp_path: Path) -> None:
    prov = MockProvider(fixture_stl=MOCK_STL)
    uris = [path_to_data_uri(VIEW_DIR / n) for n in ("front.png", "left.png")]
    job_id = prov.submit_multiview(uris, "prompt")
    st = prov.poll(job_id)
    assert st.status == "SUCCEEDED"
    out = prov.download(job_id, tmp_path / "dl")
    assert out.is_file()
    assert out.stat().st_size > 0
    assert out.read_bytes() == MOCK_STL.read_bytes()


def test_list_providers_marks_default() -> None:
    rows = list_providers()
    names = {r["name"] for r in rows}
    assert "meshy" in names
    assert "mock" in names
    default = [r for r in rows if r.get("default")]
    assert len(default) == 1
    assert default[0]["name"] == "meshy"


# --- e2e mock ----------------------------------------------------------------


def test_mock_e2e_open_plateau(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    plateau = _seed_organic_tree(tmp_path / "sessions")
    result = run_hosted_fallback(
        plateau=plateau,
        work_root=work,
        justify=JUSTIFY_OK,
        provider="mock",
        fixture_stl=MOCK_STL,
    )
    assert result.ok is True
    assert result.mesh_id
    assert result.diagnostics is not None
    assert result.honesty
    assert (
        HOSTED_HONESTY.split("—")[0].strip() in result.honesty
        or "Hosted multi-view" in result.honesty
    )
    assert result.justification is not None
    assert result.justification.operator_justify == JUSTIFY_OK
    job = JobPaths(work_root=work, mesh_id=result.mesh_id)
    assert (job.hosted_dir / "run_manifest.json").is_file()
    assert (job.hosted_dir / "hosted_report.md").is_file()
    report = (job.hosted_dir / "hosted_report.md").read_text(encoding="utf-8")
    assert "Hosted multi-view" in report
    assert JUSTIFY_OK in report


def test_closed_plateau_never_calls_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    plateau_path = tmp_path / "plateau.json"
    shutil.copy2(CLOSED_PLATEAU, plateau_path)
    # also need views for if gate somehow passed
    views = [VIEW_DIR / "front.png", VIEW_DIR / "left.png"]

    called: list[str] = []

    class SpyProvider(MockProvider):
        def submit_multiview(self, image_uris, prompt, **opts):  # type: ignore[no-untyped-def]
            called.append("submit")
            return super().submit_multiview(image_uris, prompt, **opts)

    with pytest.raises(HostedError) as ei:
        run_hosted_fallback(
            plateau=plateau_path,
            work_root=work,
            justify=JUSTIFY_OK,
            provider="mock",
            view_paths=views,
            provider_instance=SpyProvider(fixture_stl=MOCK_STL),
        )
    assert ei.value.code == "plateau_gate_closed"
    assert called == []


def test_missing_api_key_real_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MESHOPS_HOSTED_API_KEY", raising=False)
    monkeypatch.delenv("MESHOPS_MESHY_API_KEY", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    plateau = _seed_organic_tree(tmp_path / "sessions")
    with pytest.raises(HostedError) as ei:
        run_hosted_fallback(
            plateau=plateau,
            work_root=work,
            justify=JUSTIFY_OK,
            provider="meshy",
        )
    assert ei.value.code == "api_key_missing"


def test_api_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESHOPS_HOSTED_API_KEY", "host-key")
    monkeypatch.setenv("MESHOPS_MESHY_API_KEY", "meshy-key")
    assert resolve_api_key("meshy") == "meshy-key"
    monkeypatch.delenv("MESHOPS_MESHY_API_KEY")
    assert resolve_api_key("meshy") == "host-key"


# --- network-import guard ----------------------------------------------------


def test_no_network_imports_under_hosted_gate() -> None:
    """gate/views/models/errors/encode/honesty must not import urllib/httpx/requests."""
    hosted = Path(__file__).resolve().parents[1] / "src" / "meshops" / "hosted"
    files = [
        hosted / "gate.py",
        hosted / "views.py",
        hosted / "models.py",
        hosted / "errors.py",
        hosted / "encode.py",
        hosted / "honesty.py",
    ]
    banned = ("urllib", "httpx", "requests")
    for path in files:
        text = path.read_text(encoding="utf-8")
        for ban in banned:
            # allow mentions in comments/docstrings is ok for honesty; ban imports
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if stripped.startswith("import ") or stripped.startswith("from "):
                    assert ban not in stripped, f"{path.name} imports {ban}: {stripped}"


# --- jobstore / marker / honesty ---------------------------------------------


def test_jobpaths_hosted_dir(tmp_path: Path) -> None:
    job = JobPaths(work_root=tmp_path, mesh_id="abc123def456")
    ensure_job_layout(job)
    assert job.hosted_dir == job.job_dir / "hosted"
    assert job.hosted_dir.is_dir()


def test_hosted_marker_registered() -> None:
    """Strict-markers: hosted must be registered in pyproject."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("hosted:") for m in markers)


def test_optional_extra_hosted_empty() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["optional-dependencies"]["hosted"] == []


def test_hosted_honesty_constant() -> None:
    assert "not a print-ready hero sculpt" in HOSTED_HONESTY
    assert "ToS" in HOSTED_HONESTY or "commercial" in HOSTED_HONESTY


# --- CLI smoke ---------------------------------------------------------------


def test_cli_hosted_providers() -> None:
    from typer.testing import CliRunner

    from meshops.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["hosted", "providers", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["default"] == "meshy"
    names = {p["name"] for p in payload["providers"]}
    assert "mock" in names


def test_cli_hosted_run_mock(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from meshops.cli import app

    work = tmp_path / "work"
    work.mkdir()
    plateau = _seed_organic_tree(tmp_path / "sessions")
    runner = CliRunner()
    # CLI uses provider mock; fixture path is package default (tests/fixtures)
    result = runner.invoke(
        app,
        [
            "hosted",
            "run",
            "--plateau",
            str(plateau),
            "--work-root",
            str(work),
            "--justify",
            JUSTIFY_OK,
            "--provider",
            "mock",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["mesh_id"]
    assert payload["provider"] == "mock"


def test_cli_organic_help_no_hosted_claim() -> None:
    from typer.testing import CliRunner

    from meshops.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["organic", "--help"])
    assert result.exit_code == 0
    assert "No hosted API." not in result.output


def test_cli_hosted_run_closed_nonzero(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from meshops.cli import app

    plateau = tmp_path / "plateau.json"
    shutil.copy2(CLOSED_PLATEAU, plateau)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hosted",
            "run",
            "--plateau",
            str(plateau),
            "--work-root",
            str(tmp_path / "work"),
            "--justify",
            JUSTIFY_OK,
            "--provider",
            "mock",
            "--view",
            str(VIEW_DIR / "front.png"),
            "--view",
            str(VIEW_DIR / "left.png"),
            "--json",
        ],
    )
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload.get("code") == "plateau_gate_closed"


# --- convert -----------------------------------------------------------------


def test_convert_glb_to_stl_if_cheap(tmp_path: Path) -> None:
    """GLB convert path via trimesh export — skip if export unavailable."""
    import trimesh

    from meshops.hosted.convert import glb_to_stl

    box = trimesh.creation.box(extents=(5.0, 5.0, 5.0))
    glb = tmp_path / "box.glb"
    try:
        box.export(str(glb))
    except Exception:
        pytest.skip("trimesh cannot export glb in this env")
    if not glb.is_file() or glb.stat().st_size <= 0:
        pytest.skip("empty glb export")
    stl = tmp_path / "box.stl"
    out = glb_to_stl(glb, stl)
    assert out.is_file()
    assert out.stat().st_size > 0


# --- live marker collect (skipped without key) -------------------------------


@pytest.mark.hosted
def test_live_hosted_marker_skip_without_key() -> None:
    skip_if_no_hosted_key()
    # If key is present we only assert key resolution — no live network burn in default CI.
    key = resolve_api_key("meshy")
    assert key


# --- registry ----------------------------------------------------------------


def test_get_provider_unknown() -> None:
    with pytest.raises(HostedError):
        get_provider("not-a-real-provider")


def test_accept_policy_selectable_not_hard_sculpt(tmp_path: Path) -> None:
    """R25: accept uses operator policy (default export), not hard-wired for_sculpt."""
    from meshops.hosted.orchestrate import resolve_accept_policy

    assert resolve_accept_policy("export").tier == "export"
    assert resolve_accept_policy("sculpt").tier == "sculpt"
    assert resolve_accept_policy("design").tier == "design"
    with pytest.raises(HostedError) as ei:
        resolve_accept_policy("recipe")
    assert ei.value.code == "accept_policy_invalid"

    work = tmp_path / "work"
    work.mkdir()
    plateau = _seed_organic_tree(tmp_path / "sessions")
    result = run_hosted_fallback(
        plateau=plateau,
        work_root=work,
        justify=JUSTIFY_OK,
        provider="mock",
        fixture_stl=MOCK_STL,
        accept=True,
        accept_policy="export",
    )
    assert result.ok is True
    assert result.mesh_id is not None
    assert result.acceptance is not None
    assert result.acceptance.policy_tier == "export"
    job = JobPaths(work_root=work, mesh_id=result.mesh_id)
    man = json.loads((job.hosted_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert man.get("accept_policy") == "export"
    assert man.get("accept_requested") is True
