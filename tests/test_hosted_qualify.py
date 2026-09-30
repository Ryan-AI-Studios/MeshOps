"""Track 0141 — hosted qualify report. Not print success.

Synthetic meshes under tmp_path. No Blender and no live Orca.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any, get_args

import pytest
import trimesh
from pydantic import ValidationError
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
from typer.testing import CliRunner

from meshops.cli import app
from meshops.hosted.errors import HostedError, HostedErrorCode
from meshops.hosted.honesty import QUALIFY_HONESTY
from meshops.hosted.qualify import QualificationReport, classify_repair, run_hosted_qualify
from meshops.mcp.server import TOOL_NAMES
from meshops.models.diagnostics import SheetScoreFeatures, SheetScoreResult
from meshops.proportion.benchmark_multiview import (
    CandidateProvenance,
    CandidateRecord,
    Candidates,
    CaptureContract,
    FileProvenance,
    MultiviewBenchmark,
    ReferenceRecord,
    TopologyContext,
    TopologyPair,
)
from meshops.proportion.honesty import MULTIVIEW_BENCHMARK_HONESTY
from meshops.recipes.registry import NEVER_RECIPE_IDS

_RUNNER = CliRunner()
_PROPOSED = ["t1_clean", "t2_close_small_holes", "t2_smooth_spikes"]
_FORBIDDEN = (
    "input(",
    "import bpy",
    "find_blender",
    "run_hosted_fallback",
    "ingest_stl",
    "mesh_triage",
    "accept_candidate",
    "check_export",
)


def _slice_info_xml(used_g: float) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<config>\n"
        "  <header>\n"
        '    <header_item key="OrcaSlicer-Version" value="2.4.2"/>\n'
        "  </header>\n"
        "  <plate>\n"
        '    <metadata key="index" value="1"/>\n'
        '    <metadata key="prediction" value="120"/>\n'
        f'    <metadata key="weight" value="{used_g:.4f}"/>\n'
        '    <metadata key="support_used" value="false"/>\n'
        '    <metadata key="outside" value="false"/>\n'
        f'    <filament id="1" used_m="1.0" used_g="{used_g:.4f}" type="PLA" color="#FF0000"/>\n'
        "  </plate>\n"
        "</config>\n"
    )


def _ok_3mf_bytes(used_g: float = 14.21) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("Metadata/slice_info.config", _slice_info_xml(used_g))
        archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
    return buf.getvalue()


def _fake_orca(tmp_path: Path) -> tuple[Path, Any]:
    exe = tmp_path / "orca.exe"
    exe.write_bytes(b"x")

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        out = Path(argv[argv.index("--export-3mf") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        used_g = 14.21
        candidate = Path(argv[-1])
        if candidate.is_file():
            loaded = trimesh.load(candidate, force="mesh")
            if isinstance(loaded, trimesh.Trimesh) and loaded.is_watertight and loaded.volume > 0:
                # PLA 1.24 g/cm³ at a mid-band filament/mesh ratio so the oracle can pass.
                used_g = 0.4 * (float(loaded.volume) / 1000.0) * 1.24
        out.write_bytes(_ok_3mf_bytes(used_g))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    return exe, fake_run


def _box(path: Path, *, kind: str = "closed") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "closed":
        trimesh.creation.box().export(path)
    elif kind == "open":
        mesh = trimesh.creation.box()
        mesh.faces = mesh.faces[:-1]
        mesh.export(path)
    elif kind == "pair":
        left = trimesh.creation.box()
        right = trimesh.creation.box()
        right.apply_translation((3.0, 0.0, 0.0))
        trimesh.util.concatenate([left, right]).export(path)
    elif kind == "glb":
        mesh = trimesh.creation.box()
        mesh.visual = TextureVisuals(material=PBRMaterial(name="qualify-clay"))
        trimesh.Scene(mesh).export(path)
    else:
        raise AssertionError(kind)
    return path


def _solid(path: Path) -> Path:
    """Watertight manifold solid whose sheet score stays under 0.45.

    A unit cube scores about 0.65 here, so the hypothesis rule calls it T3.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=2, radius=0.5).export(path)
    return path


def _unindexed_box(path: Path) -> Path:
    corners = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    )
    faces = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    )
    lines = ["solid unindexed"]
    for tri in faces:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for index in tri:
            x, y, z = corners[index]
            lines.append(f"      vertex {x:.1f} {y:.1f} {z:.1f}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid unindexed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _benchmark(
    path: Path,
    *,
    figures: list[str] | None = None,
    ok: bool = False,
    sha: str | None = None,
) -> Path:
    figures = figures or ["rogue"]
    path.parent.mkdir(parents=True, exist_ok=True)
    image = path.parent / "ref.bin"
    image.write_bytes(b"x")
    digest = hashlib.sha256(image.read_bytes()).hexdigest() if sha is None else sha
    prov = FileProvenance(path=str(image.resolve()), byte_size=image.stat().st_size, sha256=digest)
    doc = MultiviewBenchmark(
        honesty=MULTIVIEW_BENCHMARK_HONESTY,
        ok=ok,
        status="accepted" if ok else "verdict_pending",
        visual_verdict="accept" if ok else None,
        needs_user_input=False,
        height_m=1.72,
        yaw_deg=0.0,
        capture=CaptureContract(),
        reference=ReferenceRecord(
            checklist=prov,
            images={"front": prov},
            height_m=1.72,
            figure=figures[0],
            multi_figure=len(figures) >= 2,
            in_scope_figures=figures,
        ),
        candidates=Candidates(
            meshops=CandidateRecord(provenance=CandidateProvenance()),
            meshy=CandidateRecord(provenance=CandidateProvenance()),
        ),
        views={},
        crops=[],
        topology_context=TopologyPair(
            meshops=TopologyContext(status="not_supplied", note="context only"),
            meshy=TopologyContext(status="not_supplied", note="context only"),
        ),
        messages=[],
        paths=[],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    return path


def _load(out: Path) -> dict[str, Any]:
    return json.loads((out / "qualification_report.json").read_text(encoding="utf-8"))


def _ready(
    tmp_path: Path,
    *,
    kind: str = "closed",
    figures: list[str] | None = None,
    benchmark: bool = True,
    verdict: str | None = "accept",
    qualify_slice: bool = True,
    height: float | None = 180.0,
    figure: str | None = None,
    ok: bool = False,
) -> dict[str, Any]:
    if kind == "closed":
        stl = _solid(tmp_path / "mesh.stl")
    else:
        stl = _box(tmp_path / "mesh.stl", kind=kind)
    exe, fake = _fake_orca(tmp_path)
    bench = _benchmark(tmp_path / "bench" / "multiview_benchmark.json", figures=figures, ok=ok)
    return run_hosted_qualify(
        out=tmp_path / "out",
        stl=stl,
        benchmark=bench if benchmark else None,
        qualify_slice=qualify_slice,
        print_height_mm=height,
        figure=figure,
        verdict=verdict,
        run_orca_fn=fake,
        orca_path=exe,
    )


def test_t0_source_has_no_forbidden_calls() -> None:
    import meshops.hosted.qualify as qualify_mod

    text = Path(qualify_mod.__file__).read_text(encoding="utf-8")
    for name in _FORBIDDEN:
        assert name not in text, name
    help_result = _RUNNER.invoke(app, ["hosted", "qualify", "--help"])
    assert help_result.exit_code == 0
    assert "hosted qualify archives a GLB or STL" in help_result.stdout
    assert "QUALIFY_HONESTY" in help_result.stdout
    root = _RUNNER.invoke(app, ["--help"])
    assert "qualify" in root.stdout


def test_t1_schema_honesty_markdown_and_null_rank(tmp_path: Path) -> None:
    stl = _box(tmp_path / "mesh.stl")
    out = tmp_path / "out"
    payload = run_hosted_qualify(out=out, stl=stl)
    report = QualificationReport.model_validate(payload)
    assert report.schema_version == "1.0.0"
    assert report.honesty == QUALIFY_HONESTY
    assert report.rank is None
    assert report.appearance_in_iou is False
    with pytest.raises(ValidationError):
        QualificationReport.model_validate({**payload, "print_score": 1})
    markdown = (out / "qualification_report.md").read_text(encoding="utf-8")
    assert "Acceptance is not a print-ready mesh." in markdown
    assert "check_export is not a print gate." in markdown
    assert payload["ok"] is False
    assert payload["status"] == "verdict_pending"


def test_t2_requires_a_mesh(tmp_path: Path) -> None:
    with pytest.raises(HostedError) as exc:
        run_hosted_qualify(out=tmp_path / "out")
    assert exc.value.code == "qualify_failed"
    result = _RUNNER.invoke(
        app,
        ["hosted", "qualify", "--out", str(tmp_path / "cli"), "--json"],
    )
    assert result.exit_code == 1
    body = json.loads(result.stdout)
    assert body["code"] == "qualify_failed"


def test_t3_both_inputs_are_a_source_conflict(tmp_path: Path) -> None:
    stl = _box(tmp_path / "mesh.stl")
    glb = _box(tmp_path / "mesh.glb", kind="glb")
    before_stl = stl.read_bytes()
    before_glb = glb.read_bytes()
    out = tmp_path / "out"
    payload = run_hosted_qualify(
        out=out,
        stl=stl,
        glb=glb,
        verdict="accept",
        qualify_slice=True,
        print_height_mm=180.0,
    )
    assert payload["source_conflict"] is True
    assert payload["ok"] is False
    assert payload["status"] == "source_conflict"
    assert payload["primary_class"] == "unresolved"
    assert payload["proposed_recipes"] == []
    assert payload["repair_status"] == "not_attempted"
    assert payload["slice_status"] == "not_run"
    assert payload["topology_blocks_print"] is False
    assert not (out / "classified").exists()
    assert not (out / "slice").exists()
    assert stl.read_bytes() == before_stl
    assert glb.read_bytes() == before_glb
    by_role = {item["role"]: item for item in payload["artifacts"]}
    assert set(by_role) == {"glb", "stl"}
    assert by_role["stl"]["sha256"] == hashlib.sha256(before_stl).hexdigest()
    assert by_role["glb"]["sha256"] == hashlib.sha256(before_glb).hexdigest()
    assert by_role["glb"]["faces"] is None
    assert "qualify-clay" in by_role["glb"]["material_names"]
    markdown = (out / "qualification_report.md").read_text(encoding="utf-8")
    assert "GLB and STL are both archived; qualification does not pick a print mesh" in markdown


def test_t4_unindexed_stl_welds_components_without_touching_source(tmp_path: Path) -> None:
    stl = _unindexed_box(tmp_path / "unindexed.stl")
    before = stl.read_bytes()
    out = tmp_path / "out"
    payload = run_hosted_qualify(out=out, stl=stl)
    artifact = payload["artifacts"][0]
    assert artifact["components_raw"] > artifact["components_welded"]
    assert artifact["components_welded"] == 1
    assert artifact["components_raw"] == 12
    assert stl.read_bytes() == before
    archived = out / "source" / f"stl-{stl.name}"
    assert archived.read_bytes() == before


def test_t5_sheet_refuses_repair_and_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sheet = SheetScoreResult(score=0.9, confidence=1.0, features=SheetScoreFeatures())
    primary, repair, proposed = classify_repair(
        sheet=sheet,
        is_watertight=True,
        is_manifold=True,
        boundary_edge_count=0,
        non_manifold_edge_count=0,
    )
    assert primary == "T3_sheet"
    assert repair == "refused"
    assert proposed == []

    def _high(*_args: object, **_kwargs: object) -> SheetScoreResult:
        return SheetScoreResult(score=0.9, confidence=1.0, features=SheetScoreFeatures())

    monkeypatch.setattr("meshops.hosted.qualify.compute_sheet_score", _high)
    payload = _ready(tmp_path)
    assert payload["repair_status"] == "refused"
    assert payload["proposed_recipes"] == []
    assert payload["status"] == "refused"
    assert payload["ok"] is False
    assert "T3/T4/T5 is not a T1/T2 repair; escalate is a human next step" in payload["messages"]


def test_t6_watertight_box_proposes_three_recipes_and_archives_once(tmp_path: Path) -> None:
    stl = _solid(tmp_path / "mesh.stl")
    out = tmp_path / "stl-only"
    payload = run_hosted_qualify(out=out, stl=stl)
    assert payload["repair_status"] == "not_attempted"
    assert payload["proposed_recipes"] == _PROPOSED
    assert payload["primary_class"] == "none"
    stls = list(out.rglob("*.stl"))
    assert len(stls) == 1
    assert stls[0].parent.name == "source"
    assert not (out / "classified").exists()

    glb = _solid(tmp_path / "solid.glb")
    glb_out = tmp_path / "glb-only"
    glb_payload = run_hosted_qualify(out=glb_out, glb=glb)
    assert (glb_out / "classified" / "baked.stl").is_file()
    assert (glb_out / "source" / f"glb-{glb.name}").is_file()
    roles = {item["role"] for item in glb_payload["artifacts"]}
    assert roles == {"glb", "bake"}
    assert glb_payload["proposed_recipes"] == _PROPOSED

    # A unit cube's sheet score is about 0.65, so the same rule refuses it.
    cube = _box(tmp_path / "cube.stl")
    cube_payload = run_hosted_qualify(out=tmp_path / "cube-out", stl=cube)
    assert cube_payload["primary_class"] == "T3_sheet"
    assert cube_payload["repair_status"] == "refused"
    assert cube_payload["proposed_recipes"] == []


def test_t7_benchmark_sha_mismatch_and_ok_is_not_success(tmp_path: Path) -> None:
    stl = _box(tmp_path / "mesh.stl")
    bad = _benchmark(tmp_path / "bad" / "multiview_benchmark.json", sha="0" * 64)
    with pytest.raises(HostedError) as exc:
        run_hosted_qualify(out=tmp_path / "out-bad", stl=stl, benchmark=bad)
    assert exc.value.code == "qualify_failed"
    good = _benchmark(tmp_path / "good" / "multiview_benchmark.json", ok=True)
    payload = run_hosted_qualify(out=tmp_path / "out-good", stl=stl, benchmark=good)
    assert payload["ok"] is False
    assert "benchmark ok is not qualification success" in payload["messages"]
    assert payload["benchmark_status"] == "accepted"
    assert payload["compare_status"] == "context"


def test_t8_accept_without_benchmark_is_baseline_only(tmp_path: Path) -> None:
    payload = _ready(tmp_path, benchmark=False)
    assert payload["status"] == "baseline_only"
    assert payload["ok"] is False
    assert payload["compare_status"] == "baseline_only"
    assert payload["slice_status"] == "pass"


def test_t9_accept_without_slice_is_slice_absent(tmp_path: Path) -> None:
    payload = _ready(tmp_path, qualify_slice=False, height=None)
    assert payload["status"] == "slice_absent"
    assert payload["ok"] is False
    assert payload["slice_status"] == "not_run"


def test_t10_slice_pass_does_not_clear_open_topology(tmp_path: Path) -> None:
    payload = _ready(tmp_path, kind="open")
    out = tmp_path / "out"
    assert payload["status"] == "topology_blocked"
    assert payload["ok"] is False
    assert payload["slice_status"] == "pass"
    assert payload["topology_blocks_print"] is True
    markdown = (out / "qualification_report.md").read_text(encoding="utf-8")
    assert "slice pass is not watertight proof." in markdown


def test_open_boundary_is_t1_when_the_sheet_score_is_low(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    mesh.faces = mesh.faces[:-1]
    path = tmp_path / "open.stl"
    mesh.export(path)
    payload = run_hosted_qualify(out=tmp_path / "out", stl=path)
    artifact = payload["artifacts"][0]
    assert artifact["is_watertight"] is False
    assert artifact["boundary_edge_count"] > 0
    assert payload["primary_class"] == "T1_topology"
    assert payload["repair_status"] == "not_attempted"
    assert payload["proposed_recipes"] == _PROPOSED
    assert payload["topology_blocks_print"] is True

    closed = run_hosted_qualify(out=tmp_path / "closed", stl=_solid(tmp_path / "closed.stl"))
    assert closed["artifacts"][0]["boundary_edge_count"] == 0
    assert closed["primary_class"] == "none"


def test_t10_missing_orca_still_writes_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("meshops.slice.runner.find_orca", lambda **_kwargs: None)
    stl = _box(tmp_path / "mesh.stl")
    bench = _benchmark(tmp_path / "bench" / "multiview_benchmark.json")
    out = tmp_path / "out"
    payload = run_hosted_qualify(
        out=out,
        stl=stl,
        benchmark=bench,
        verdict="accept",
        qualify_slice=True,
        print_height_mm=180.0,
    )
    assert payload["slice_status"] == "orca_not_found"
    assert payload["status"] == "slice_failed"
    assert payload["ok"] is False
    assert (out / "qualification_report.json").is_file()
    assert (out / "slice" / "print_scaled.stl").is_file()
    assert "slice pass is not watertight proof." in payload["messages"]


def test_t11_accept_scales_the_longest_axis_to_180(tmp_path: Path) -> None:
    payload = _ready(tmp_path, figures=["rogue"])
    out = tmp_path / "out"
    assert payload["status"] == "accepted"
    assert payload["ok"] is True
    assert payload["honesty"] == QUALIFY_HONESTY
    assert payload["needs_user_input"] is False
    assert payload["scale_factor"] == pytest.approx(180.0)
    assert payload["print_height_mm"] == pytest.approx(180.0)
    assert payload["appearance_in_iou"] is False
    scaled = trimesh.load(out / "slice" / "print_scaled.stl", force="mesh", process=False)
    assert float(max(scaled.extents)) == pytest.approx(180.0, abs=0.05)
    artifact = payload["artifacts"][0]
    assert artifact["units_guess"] == "metre_scale"
    assert (out / "_ad_hoc_slice").is_dir()
    markdown = (out / "qualification_report.md").read_text(encoding="utf-8")
    assert "Acceptance is not a print-ready mesh." in markdown
    assert payload["honesty"] == QUALIFY_HONESTY


def test_t12_slice_requires_a_finite_positive_height(tmp_path: Path) -> None:
    stl = _box(tmp_path / "mesh.stl")
    for height in (None, 0.0, -5.0, float("nan"), float("inf")):
        out = tmp_path / f"out-{height}"
        with pytest.raises(HostedError) as exc:
            run_hosted_qualify(
                out=out,
                stl=stl,
                qualify_slice=True,
                print_height_mm=height,
            )
        assert exc.value.code == "qualify_failed"
        assert not (out / "qualification_report.json").exists()


def test_t13_require_verdict_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stl = _box(tmp_path / "mesh.stl")
    pending = tmp_path / "pending"
    pending_result = _RUNNER.invoke(
        app,
        [
            "hosted",
            "qualify",
            "--stl",
            str(stl),
            "--out",
            str(pending),
            "--json",
            "--require-verdict",
        ],
    )
    assert pending_result.exit_code == 2
    assert (pending / "qualification_report.json").is_file()
    pending_body = json.loads((pending / "qualification_report.json").read_text(encoding="utf-8"))
    assert pending_body["status"] == "verdict_pending"

    exe, fake = _fake_orca(tmp_path)
    monkeypatch.setenv("MESHOPS_ORCA", str(exe))
    monkeypatch.setattr("meshops.slice.runner.run_orca", fake)
    bench = _benchmark(tmp_path / "bench" / "multiview_benchmark.json")
    solid = _solid(tmp_path / "solid.stl")
    accepted = tmp_path / "accepted"
    accepted_result = _RUNNER.invoke(
        app,
        [
            "hosted",
            "qualify",
            "--stl",
            str(solid),
            "--benchmark",
            str(bench),
            "--out",
            str(accepted),
            "--slice",
            "--print-height-mm",
            "180",
            "--verdict",
            "accept",
            "--json",
            "--require-verdict",
        ],
    )
    assert accepted_result.exit_code == 0
    body = json.loads(accepted_result.stdout)
    assert body["status"] == "accepted"
    assert body["ok"] is True


def test_t14_figure_laterality(tmp_path: Path) -> None:
    stl = _box(tmp_path / "mesh.stl")
    bench = _benchmark(tmp_path / "bench" / "multiview_benchmark.json", figures=["a", "b"])
    pending = run_hosted_qualify(out=tmp_path / "pending", stl=stl, benchmark=bench)
    assert pending["needs_user_input"] is True

    exe, fake = _fake_orca(tmp_path)
    solid = _solid(tmp_path / "solid.stl")
    accepted = run_hosted_qualify(
        out=tmp_path / "accepted",
        stl=solid,
        benchmark=bench,
        figure="a",
        verdict="accept",
        qualify_slice=True,
        print_height_mm=180.0,
        run_orca_fn=fake,
        orca_path=exe,
    )
    assert accepted["needs_user_input"] is False
    assert accepted["status"] == "accepted"
    assert accepted["ok"] is True

    with pytest.raises(HostedError) as missing:
        run_hosted_qualify(out=tmp_path / "no-bench", stl=stl, figure="a")
    assert missing.value.code == "qualify_failed"
    with pytest.raises(HostedError) as outside:
        run_hosted_qualify(out=tmp_path / "outside", stl=stl, benchmark=bench, figure="c")
    assert outside.value.code == "qualify_failed"


def test_t15_components_and_clothing_need_a_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _box(tmp_path / "pair.stl", kind="pair")
    bench = _benchmark(tmp_path / "bench" / "multiview_benchmark.json", figures=["a", "b"])
    payload = run_hosted_qualify(
        out=tmp_path / "pair-out",
        stl=pair,
        benchmark=bench,
        figure="a",
        verdict="accept",
    )
    assert payload["needs_user_input"] is True
    assert payload["artifacts"][0]["components_welded"] > 1

    def _clothing(*_args: object, **_kwargs: object) -> SheetScoreResult:
        return SheetScoreResult(
            score=0.4,
            confidence=1.0,
            features=SheetScoreFeatures(clothing_penalty=0.55),
        )

    monkeypatch.setattr("meshops.hosted.qualify.compute_sheet_score", _clothing)
    clothed = _ready(
        tmp_path / "cloth", figures=["solo"], figure="solo", qualify_slice=False, height=None
    )
    assert clothed["needs_user_input"] is True
    assert clothed["repair_status"] == "not_attempted"
    assert clothed["status"] == "slice_absent"


def test_t16_catalog_error_literal_and_never_recipes(tmp_path: Path) -> None:
    assert len(TOOL_NAMES) == 57
    assert "mesh_hosted_qualify" in TOOL_NAMES
    assert "qualify_failed" in get_args(HostedErrorCode)
    payload = run_hosted_qualify(out=tmp_path / "out", stl=_solid(tmp_path / "mesh.stl"))
    assert NEVER_RECIPE_IDS.isdisjoint(payload["proposed_recipes"])
    assert set(payload["proposed_recipes"]) == set(_PROPOSED)


def test_t17_no_rogue_path_and_addopts_unchanged() -> None:
    text = Path(__file__).read_text(encoding="utf-8")
    assert "work/" + "rogue-v3" not in text
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    raw = pyproject.read_text(encoding="utf-8")
    assert 'addopts = ["-q", "--strict-markers", "--strict-config"]' in raw


def test_same_filename_sources_keep_distinct_archives(tmp_path: Path) -> None:
    glb_bytes = _box(tmp_path / "clay.glb", kind="glb").read_bytes()
    stl_bytes = _box(tmp_path / "solid.stl").read_bytes()
    name = "twin.stl"
    glb = tmp_path / "from-glb" / name
    stl = tmp_path / "from-stl" / name
    glb.parent.mkdir()
    stl.parent.mkdir()
    glb.write_bytes(glb_bytes)
    stl.write_bytes(stl_bytes)
    out = tmp_path / "out"
    payload = run_hosted_qualify(out=out, glb=glb, stl=stl, verdict="accept")
    by_role = {item["role"]: item for item in payload["artifacts"]}
    assert by_role["glb"]["path"] != by_role["stl"]["path"]
    assert by_role["glb"]["path"] == f"source/glb-{name}"
    assert by_role["stl"]["path"] == f"source/stl-{name}"
    assert by_role["glb"]["sha256"] == hashlib.sha256(glb.read_bytes()).hexdigest()
    assert by_role["stl"]["sha256"] == hashlib.sha256(stl.read_bytes()).hexdigest()
    assert by_role["glb"]["sha256"] != by_role["stl"]["sha256"]
    assert (out / by_role["glb"]["path"]).read_bytes() == glb.read_bytes()
    assert (out / by_role["stl"]["path"]).read_bytes() == stl.read_bytes()
    assert payload["source_conflict"] is True
    assert not (out / "classified").exists()
    assert not (out / "slice").exists()


def test_malformed_mesh_is_qualify_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.stl"
    bad.write_bytes(b"not a mesh")

    def _boom(_path: object) -> object:
        raise ValueError("unsupported mesh")

    monkeypatch.setattr("meshops.hosted.qualify.load_mesh", _boom)
    with pytest.raises(HostedError) as exc:
        run_hosted_qualify(out=tmp_path / "out", stl=bad)
    assert exc.value.code == "qualify_failed"

    result = _RUNNER.invoke(
        app,
        ["hosted", "qualify", "--stl", str(bad), "--out", str(tmp_path / "cli"), "--json"],
    )
    assert result.exit_code == 1
    body = json.loads(result.stdout)
    assert body["code"] == "qualify_failed"
    assert body["error"] == "HostedError"


def test_invalid_verdict_fails_before_a_bundle(tmp_path: Path) -> None:
    stl = _box(tmp_path / "mesh.stl")
    out = tmp_path / "out"
    with pytest.raises(HostedError) as exc:
        run_hosted_qualify(out=out, stl=stl, verdict="maybe")
    assert exc.value.code == "qualify_failed"
    assert not out.exists()
