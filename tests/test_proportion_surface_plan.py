"""Track 0139 — one named junction weld plan. Authoring only, not print success.

Default path uses synthetic recipes. No Blender and no work/rogue-v3.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import trimesh
from pydantic import ValidationError
from typer.testing import CliRunner

from meshops.cli import app
from meshops.guards.check import check_export
from meshops.guards.policy import GuardPolicy
from meshops.mcp.server import TOOL_NAMES
from meshops.models.diagnostics import MeshStats
from meshops.proportion.benchmark_multiview import (
    MULTIVIEW_BENCHMARK_HONESTY,
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
from meshops.proportion.connection_metrics import connection_gap_metrics
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import SURFACE_HONESTY
from meshops.proportion.silhouette import SILHOUETTE_SCHEMA_VERSION, run_silhouette_compare
from meshops.proportion.surface_plan import (
    SURFACE_SCHEMA_VERSION,
    SurfacePlan,
    SurfaceScore,
    run_surface_plan,
)

_REPO = Path(__file__).resolve().parents[1]
_RUNNER = CliRunner()
_RECIPE_HONESTY = "proportion_blockout_recipe_not_mesh_or_print_success"


def _ellipsoid(name: str, role: str, center: tuple[float, float, float]) -> dict[str, Any]:
    return {
        "name": name,
        "label": name,
        "role": role,
        "kind": "ellipsoid",
        "center": [center[0], center[1], center[2]],
        "rx_m": 0.05,
        "ry_m": 0.04,
        "rz_m": 0.06,
    }


def _recipe(parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "1.4.0",
        "honesty": _RECIPE_HONESTY,
        "recipe_id": "humanoid_a_pose_v1",
        "parts": parts,
    }


def _write_recipe(path: Path, parts: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_recipe(parts)), encoding="utf-8")
    return path


def _shoulder_recipe(path: Path) -> Path:
    return _write_recipe(
        path,
        [
            _ellipsoid("RECIPE_deltoid_soft_l", "deltoid_soft", (-0.18, -0.02, 1.35)),
            _ellipsoid("RECIPE_torso_oval_chest", "torso", (0.0, 0.0, 1.25)),
            _ellipsoid("RECIPE_toe_1_l", "toe_soft", (-0.08, -0.12, 0.02)),
        ],
    )


def _forbid_recipe(path: Path) -> Path:
    return _write_recipe(
        path,
        [
            _ellipsoid("RECIPE_deltoid_soft_l", "deltoid_soft", (-0.18, -0.02, 1.35)),
            _ellipsoid("RECIPE_torso_oval_chest", "torso", (0.0, 0.0, 1.25)),
            _ellipsoid("RECIPE_toe_1_l", "toe_soft", (-0.08, -0.12, 0.02)),
            _ellipsoid("RECIPE_finger_index_0_l", "finger_soft", (-0.28, -0.04, 0.9)),
            _ellipsoid("RECIPE_palm_l", "palm", (-0.24, -0.02, 0.95)),
            _ellipsoid("RECIPE_neck", "neck", (0.0, 0.0, 1.5)),
            _ellipsoid("RECIPE_head", "head", (0.0, 0.0, 1.62)),
        ],
    )


def _explode_blender(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("find_blender called")


def _invoke(recipe: Path, out: Path, *extra: str):
    args = [
        "proportion",
        "blockout-surface",
        "--recipe",
        str(recipe),
        "--out",
        str(out),
        *extra,
    ]
    return _RUNNER.invoke(app, args)


def _write_benchmark(
    path: Path,
    image: Path,
    *,
    ok: bool,
    figures: list[str],
    sha: str | None = None,
) -> None:
    data = image.read_bytes()
    digest = hashlib.sha256(data).hexdigest() if sha is None else sha
    prov = FileProvenance(path=str(image), byte_size=len(data), sha256=digest)
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
    path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")


def _stats(faces: int, size: int) -> MeshStats:
    return MeshStats(
        faces=faces,
        vertices=faces // 2,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
        bbox_diagonal=1.732,
        components=1,
        file_size_bytes=size,
        content_sha256="a" * 64,
        mesh_id="t12",
    )


def test_t0_honesty_cli_mcp_and_no_bpy_import() -> None:
    """T0: honesty, CLI verb, MCP name, no stdin, surface module does not import bpy."""
    honesty = (_REPO / "src/meshops/proportion/honesty.py").read_text(encoding="utf-8")
    cli = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    server = (_REPO / "src/meshops/mcp/server.py").read_text(encoding="utf-8")
    tools = (_REPO / "src/meshops/mcp/tools.py").read_text(encoding="utf-8")
    surface = (_REPO / "src/meshops/proportion/surface_plan.py").read_text(encoding="utf-8")
    assert SURFACE_HONESTY == "proportion_continuous_surface_not_mesh_or_print_success"
    assert "SURFACE_HONESTY" in honesty
    assert 'command("blockout-surface")' in cli
    assert "input(" not in surface
    assert "import bpy" not in surface
    assert "mesh_proportion_blockout_surface" in server
    assert "def mesh_proportion_blockout_surface" in tools


def test_t1_schema_rejects_unknown_field_and_old_version() -> None:
    """T1: extra fields and schema 0.9.0 are rejected."""
    body = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "honesty": SURFACE_HONESTY,
        "ok": False,
        "status": "verdict_pending",
        "needs_user_input": False,
        "rank": None,
        "n_parts": 0,
        "apply_status": "plan_only",
        "clusters": [],
        "gaps": {},
        "messages": [],
        "paths": [],
        "forbid": [],
    }
    SurfacePlan.model_validate(body)
    with pytest.raises(ValidationError):
        SurfacePlan.model_validate({**body, "not_a_field": 1})
    with pytest.raises(ValidationError):
        SurfacePlan.model_validate({**body, "schema_version": "0.9.0"})


def test_t2_omitted_verdict_writes_pending_bundle(tmp_path: Path) -> None:
    """T2: omitted verdict writes the bundle, ok false, exit 0, plan_only."""
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    out = tmp_path / "out"
    result = _invoke(recipe, out)
    assert result.exit_code == 0
    payload = json.loads((out / "surface_plan.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "verdict_pending"
    assert payload["apply_status"] == "plan_only"
    assert payload["rank"] is None
    assert (out / "surface_plan.md").is_file()


def test_t3_accept_without_apply_stays_plan_only(tmp_path: Path) -> None:
    """T3: --verdict accept without apply stays ok false and plan_only."""
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    out = tmp_path / "out"
    result = _invoke(recipe, out, "--verdict", "accept")
    assert result.exit_code == 0
    payload = json.loads((out / "surface_plan.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "plan_only"


def test_t4_gaps_and_toe_cluster(tmp_path: Path) -> None:
    """T4: gap keys come from connection_gap_metrics; toe cluster is unweldable."""
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    out = tmp_path / "out"
    payload = run_surface_plan(recipe, out)
    assert payload["n_parts"] == 3
    from meshops.proportion.blockout_recipe import load_blockout_recipe

    package = load_blockout_recipe(recipe)
    assert payload["gaps"] == connection_gap_metrics(package)
    toe = next(c for c in payload["clusters"] if c["cluster_id"] == "RECIPE_toe_1_l")
    assert toe["weldable"] is False
    shoulder = next(c for c in payload["clusters"] if c["cluster_id"] == "shoulder_l")
    assert shoulder["weldable"] is True
    assert shoulder["members"] == ["RECIPE_deltoid_soft_l", "RECIPE_torso_oval_chest"]


def test_t5_bad_recipe_does_not_probe_blender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T5: non-JSON and a missing parts list raise surface_failed before Blender."""
    monkeypatch.setattr("meshops.proportion.surface_plan.find_blender", _explode_blender)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProportionError) as exc:
        run_surface_plan(bad, tmp_path / "out-a")
    assert exc.value.code == "surface_failed"
    missing = tmp_path / "noparts.json"
    missing.write_text(
        json.dumps({"schema_version": "1.4.0", "honesty": _RECIPE_HONESTY}),
        encoding="utf-8",
    )
    with pytest.raises(ProportionError) as exc2:
        run_surface_plan(missing, tmp_path / "out-b")
    assert exc2.value.code == "surface_failed"


def test_t6_apply_without_allow_before_blender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T6: --apply without --allow-region-weld raises before find_blender."""
    monkeypatch.setattr("meshops.proportion.surface_plan.find_blender", _explode_blender)
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    with pytest.raises(ProportionError) as exc:
        run_surface_plan(
            recipe,
            tmp_path / "out",
            cluster="shoulder_l",
            apply=True,
            allow_region_weld=False,
        )
    assert exc.value.code == "surface_failed"


def test_t7_refused_clusters_before_blender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7: toe, finger, palm, and neck_head refuse before find_blender."""
    monkeypatch.setattr("meshops.proportion.surface_plan.find_blender", _explode_blender)
    recipe = _forbid_recipe(tmp_path / "blockout_recipe.json")
    for cluster in ("RECIPE_toe_1_l", "RECIPE_finger_index_0_l", "RECIPE_palm_l", "neck_head"):
        with pytest.raises(ProportionError) as exc:
            run_surface_plan(
                recipe,
                tmp_path / cluster,
                cluster=cluster,
                apply=True,
                allow_region_weld=True,
            )
        assert exc.value.code == "surface_failed"


def test_invalid_verdict_before_blender(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown verdict raises before find_blender, including on an apply request."""
    monkeypatch.setattr("meshops.proportion.surface_plan.find_blender", _explode_blender)
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    with pytest.raises(ProportionError) as exc:
        run_surface_plan(
            recipe,
            tmp_path / "out",
            cluster="shoulder_l",
            apply=True,
            allow_region_weld=True,
            verdict="maybe",
        )
    assert exc.value.code == "surface_failed"


def test_require_verdict_exits_2_after_write(tmp_path: Path) -> None:
    """--require-verdict writes the bundle, then exits 2 while status is pending."""
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    out = tmp_path / "out"
    result = _invoke(recipe, out, "--require-verdict")
    assert result.exit_code == 2
    assert (out / "surface_plan.json").is_file()


def test_t8_unknown_cluster_before_blender(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T8: unknown cluster id raises surface_failed before find_blender."""
    monkeypatch.setattr("meshops.proportion.surface_plan.find_blender", _explode_blender)
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    with pytest.raises(ProportionError) as exc:
        run_surface_plan(
            recipe,
            tmp_path / "out",
            cluster="image_left_hand",
            apply=True,
            allow_region_weld=True,
        )
    assert exc.value.code == "surface_failed"


def test_t9_benchmark_sha_and_ok_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T9: reference SHA mismatch raises; benchmark ok true does not set surface ok."""
    monkeypatch.setattr("meshops.proportion.surface_plan.find_blender", _explode_blender)
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    image = tmp_path / "front.png"
    image.write_bytes(b"png-bytes")
    bad = tmp_path / "bad_benchmark.json"
    _write_benchmark(bad, image, ok=True, figures=["woman"], sha="0" * 64)
    with pytest.raises(ProportionError) as exc:
        run_surface_plan(recipe, tmp_path / "out-bad", benchmark=bad)
    assert exc.value.code == "surface_failed"
    good = tmp_path / "good_benchmark.json"
    _write_benchmark(good, image, ok=True, figures=["woman"])
    payload = run_surface_plan(recipe, tmp_path / "out-good", benchmark=good)
    assert payload["ok"] is False
    assert payload["benchmark_status"] == "accepted"
    assert "benchmark ok is not surface success" in payload["messages"]


def test_t10_multi_figure_needs_user(tmp_path: Path) -> None:
    """T10: two in-scope figures and no --figure keep needs_user_input and ok false."""
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    image = tmp_path / "front.png"
    image.write_bytes(b"png-bytes")
    bench = tmp_path / "bench.json"
    _write_benchmark(bench, image, ok=True, figures=["woman", "man"])
    payload = run_surface_plan(
        recipe,
        tmp_path / "out",
        benchmark=bench,
        verdict="accept",
    )
    assert payload["needs_user_input"] is True
    assert payload["ok"] is False
    with pytest.raises(ProportionError) as exc:
        run_surface_plan(
            recipe,
            tmp_path / "out2",
            benchmark=bench,
            figure="neither",
        )
    assert exc.value.code == "surface_failed"


def test_t11_silhouette_schema_and_back_role_stay(tmp_path: Path) -> None:
    """T11: silhouette schema stays 1.2.0 and back is still rejected."""
    assert SILHOUETTE_SCHEMA_VERSION == "1.2.0"
    image = tmp_path / "front.png"
    image.write_bytes(b"not-a-real-png")
    with pytest.raises(ProportionError):
        run_silhouette_compare(image, tmp_path / "sil", mesh_view=image, view_role="back")


def test_t12_sculpt_guard_and_islands_do_not_accept() -> None:
    """T12: collapsed candidate fails for_sculpt; target_islands_max does not set ok."""
    result = check_export(
        _stats(1000, 4000),
        _stats(10, 4000),
        policy=GuardPolicy.for_sculpt(),
    )
    assert result.ok is False
    assert "face_floor" in result.failed
    body = {
        "schema_version": "1.0.0",
        "honesty": SURFACE_HONESTY,
        "ok": False,
        "status": "verdict_pending",
        "needs_user_input": False,
        "rank": None,
        "n_parts": 3,
        "apply_status": "plan_only",
        "clusters": [],
        "gaps": {},
        "messages": [],
        "paths": [],
        "forbid": [],
        "target_islands_max": 1,
    }
    plan = SurfacePlan.model_validate(body)
    assert plan.target_islands_max == 1
    assert plan.ok is False


def test_t13_catalog_is_55() -> None:
    """T13: live catalog length and the setup-launch source pin are 55."""
    assert len(TOOL_NAMES) == 57
    assert "mesh_proportion_blockout_surface" in TOOL_NAMES
    mcp_test = (_REPO / "tests/test_mcp_server.py").read_text(encoding="utf-8")
    launch = (_REPO / "tests/test_proportion_setup_launch.py").read_text(encoding="utf-8")
    assert "len(TOOL_NAMES) == 57" in mcp_test
    assert "len(TOOL_NAMES) == 57" in launch
    assert "len(TOOL_NAMES) == 53" not in mcp_test
    assert "len(TOOL_NAMES) == 53" not in launch


def test_t14_markdown_names_clusters_and_writes_no_html(tmp_path: Path) -> None:
    """T14: markdown carries honesty and cluster ids, and no HTML is written."""
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    out = tmp_path / "out"
    run_surface_plan(recipe, out)
    text = (out / "surface_plan.md").read_text(encoding="utf-8")
    assert SURFACE_HONESTY in text
    for cluster_id in ("shoulder_l", "hip_l", "neck_torso", "neck_head", "ankle_l"):
        assert cluster_id in text
    assert "not print success" in text
    assert list(out.rglob("*.html")) == []


def test_t15_refused_apply_keeps_recipe_bytes(tmp_path: Path) -> None:
    """T15: a refused apply leaves the caller recipe unchanged and writes no weld STL."""
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    before = recipe.read_bytes()
    out = tmp_path / "out"
    with pytest.raises(ProportionError):
        run_surface_plan(recipe, out, cluster="shoulder_l", apply=True)
    assert recipe.read_bytes() == before
    assert list(out.rglob("*.stl")) == []


def test_t16_delta_iou_has_no_threshold(tmp_path: Path) -> None:
    """T16: recorded delta_iou has no threshold field that sets ok."""
    assert "threshold" not in SurfaceScore.model_fields
    assert "iou_pass" not in SurfaceScore.model_fields
    score = SurfaceScore(role="front", iou=0.25, dice=0.4, delta_iou=-0.05)
    dumped = score.model_dump()
    assert dumped["delta_iou"] == pytest.approx(-0.05)
    assert "threshold" not in dumped
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    out = tmp_path / "out"
    payload = run_surface_plan(recipe, out)
    raw = (out / "surface_plan.json").read_text(encoding="utf-8")
    assert payload["ok"] is False
    assert "threshold" not in raw
    assert "iou_pass" not in raw


def _write_sphere(path: Path, subdivisions: int) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def test_apply_moves_stl_only_after_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarded apply copies the remesh STL; a collapsed candidate leaves no weld STL."""
    monkeypatch.setattr(
        "meshops.proportion.surface_plan.find_blender",
        lambda require=False: Path("blender"),
    )
    monkeypatch.setattr(
        "meshops.proportion.surface_plan.require_blender_52",
        lambda _blender: "5.2.0",
    )

    def _pass(cmd: list[str], **_kwargs: object) -> object:
        job = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        _write_sphere(Path(job["before_stl"]), 3)
        _write_sphere(Path(job["after_stl"]), 3)

        class _Proc:
            returncode = 0
            stdout = "SURFACE_WELD_OK\n"
            stderr = ""

        return _Proc()

    monkeypatch.setattr("meshops.proportion.surface_plan.subprocess.run", _pass)
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    before = recipe.read_bytes()
    out = tmp_path / "pass"
    payload = run_surface_plan(
        recipe,
        out,
        cluster="shoulder_l",
        apply=True,
        allow_region_weld=True,
        verdict="accept",
    )
    assert payload["apply_status"] == "applied"
    assert payload["ok"] is True
    assert payload["status"] == "accepted"
    assert (out / "weld" / "shoulder_l.stl").is_file()
    assert (out / "archive" / "shoulder_l_pre_surface.json").is_file()
    assert (out / "frozen" / "blockout_recipe.json").is_file()
    assert recipe.read_bytes() == before

    def _fail(cmd: list[str], **_kwargs: object) -> object:
        job = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        _write_sphere(Path(job["before_stl"]), 3)
        _write_sphere(Path(job["after_stl"]), 1)

        class _Proc:
            returncode = 0
            stdout = "SURFACE_WELD_OK\n"
            stderr = ""

        return _Proc()

    monkeypatch.setattr("meshops.proportion.surface_plan.subprocess.run", _fail)
    out_fail = tmp_path / "fail"
    with pytest.raises(ProportionError) as exc:
        run_surface_plan(
            recipe,
            out_fail,
            cluster="shoulder_l",
            apply=True,
            allow_region_weld=True,
            verdict="accept",
        )
    assert exc.value.code == "surface_failed"
    written = json.loads((out_fail / "surface_plan.json").read_text(encoding="utf-8"))
    assert written["apply_status"] == "guard_failed"
    assert written["ok"] is False
    assert list(out_fail.rglob("*.stl")) == []
    assert recipe.read_bytes() == before


def _cylinder(
    name: str,
    role: str,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    *,
    kind: str = "cylinder",
) -> dict[str, Any]:
    return {
        "name": name,
        "label": name,
        "role": role,
        "kind": kind,
        "p0": [p0[0], p0[1], p0[2]],
        "p1": [p1[0], p1[1], p1[2]],
        "radius_m": 0.04,
    }


def _trap_box(name: str, role: str, center: tuple[float, float, float]) -> dict[str, Any]:
    return {
        "name": name,
        "label": name,
        "role": role,
        "kind": "trap_box",
        "center": [center[0], center[1], center[2]],
        "top_half_width_m": 0.16,
        "bottom_half_width_m": 0.14,
        "half_depth_m": 0.08,
        "z_bottom_m": center[2] - 0.08,
        "z_top_m": center[2] + 0.08,
    }


def _mock_blender(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "meshops.proportion.surface_plan.find_blender",
        lambda require=False: Path("blender"),
    )
    monkeypatch.setattr(
        "meshops.proportion.surface_plan.require_blender_52",
        lambda _blender: "5.2.0",
    )

    def _pass(cmd: list[str], **_kwargs: object) -> object:
        job = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        _write_sphere(Path(job["before_stl"]), 3)
        _write_sphere(Path(job["after_stl"]), 3)

        class _Proc:
            returncode = 0
            stdout = "SURFACE_WELD_OK\n"
            stderr = ""

        return _Proc()

    monkeypatch.setattr("meshops.proportion.surface_plan.subprocess.run", _pass)


def _solid_png(path: Path, rgb: tuple[int, int, int]) -> None:
    pytest.importorskip("PIL")
    from PIL import Image  # type: ignore[import-untyped,import-not-found]

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), rgb).save(path)


def _write_four_views(directory: Path, rgb: tuple[int, int, int]) -> None:
    for role in ("front", "left", "three_quarter", "back"):
        _solid_png(directory / f"{role}.png", rgb)


def _write_four_role_benchmark(path: Path, image: Path) -> None:
    data = image.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    prov = FileProvenance(path=str(image), byte_size=len(data), sha256=digest)
    images = {role: prov for role in ("front", "left", "three_quarter", "back")}
    doc = MultiviewBenchmark(
        honesty=MULTIVIEW_BENCHMARK_HONESTY,
        ok=False,
        status="verdict_pending",
        visual_verdict=None,
        needs_user_input=False,
        height_m=1.72,
        yaw_deg=0.0,
        capture=CaptureContract(),
        reference=ReferenceRecord(
            checklist=prov,
            images=images,
            height_m=1.72,
            figure="rogue",
            multi_figure=False,
            in_scope_figures=["rogue"],
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
    path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")


def test_job_keeps_cylinder_capsule_and_trap_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply job keeps recipe kind. Cylinder and capsule are not sphere payloads."""
    from meshops.proportion.surface_weld_bpy import emission_kind, trap_box_corners_m

    _mock_blender(monkeypatch)
    recipe = _write_recipe(
        tmp_path / "blockout_recipe.json",
        [
            _cylinder(
                "RECIPE_hip_bridge_l",
                "hip_bridge",
                (-0.12, 0.0, 0.9),
                (-0.2, 0.0, 0.9),
            ),
            _trap_box("RECIPE_torso_oval_hip", "torso", (0.0, 0.0, 0.9)),
            _cylinder(
                "RECIPE_neck",
                "neck",
                (0.0, 0.0, 1.45),
                (0.0, 0.0, 1.55),
                kind="capsule",
            ),
            _ellipsoid("RECIPE_torso_oval_chest", "torso", (0.0, 0.0, 1.25)),
        ],
    )
    out = tmp_path / "hip"
    run_surface_plan(
        recipe,
        out,
        cluster="hip_l",
        apply=True,
        allow_region_weld=True,
    )
    job = json.loads((out / "frozen" / "surface_job.json").read_text(encoding="utf-8"))
    by_name = {member["name"]: member for member in job["members"]}
    bridge = by_name["RECIPE_hip_bridge_l"]
    hip = by_name["RECIPE_torso_oval_hip"]
    assert bridge["kind"] == "cylinder"
    assert bridge["p0"] == [-0.12, 0.0, 0.9]
    assert bridge["p1"] == [-0.2, 0.0, 0.9]
    assert bridge["radius_m"] == pytest.approx(0.04)
    assert emission_kind(bridge) == "cylinder"
    assert hip["kind"] == "trap_box"
    assert len(trap_box_corners_m(hip)) == 8
    assert hip["top_half_width_m"] == pytest.approx(0.16)
    assert emission_kind(hip) == "trap_box"

    out_neck = tmp_path / "neck"
    run_surface_plan(
        recipe,
        out_neck,
        cluster="neck_torso",
        apply=True,
        allow_region_weld=True,
    )
    neck_job = json.loads((out_neck / "frozen" / "surface_job.json").read_text(encoding="utf-8"))
    neck = next(member for member in neck_job["members"] if member["name"] == "RECIPE_neck")
    assert neck["kind"] == "capsule"
    assert neck["p0"] is not None and neck["p1"] is not None
    assert emission_kind(neck) == "cylinder"
    chest = next(
        member for member in neck_job["members"] if member["name"] == "RECIPE_torso_oval_chest"
    )
    assert chest["kind"] == "ellipsoid"
    assert chest["rx_m"] == pytest.approx(0.05)
    assert emission_kind(chest) == "ellipsoid"


def test_plan_only_ignores_stale_views(tmp_path: Path) -> None:
    """Stale out/views on a plan-only run record no scores."""
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    out = tmp_path / "out"
    _write_four_views(out / "views", (240, 240, 240))
    ref = tmp_path / "ref.png"
    _solid_png(ref, (240, 240, 240))
    benchmark = tmp_path / "benchmark.json"
    _write_four_role_benchmark(benchmark, ref)
    payload = run_surface_plan(recipe, out, benchmark=benchmark)
    assert payload["apply_status"] == "plan_only"
    assert payload["scores"] == []


def test_applied_scores_only_views_written_this_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged view hashes are not scored. Views rewritten during apply are."""
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    ref = tmp_path / "ref.png"
    _solid_png(ref, (240, 240, 240))
    benchmark = tmp_path / "benchmark.json"
    _write_four_role_benchmark(benchmark, ref)

    stale = tmp_path / "stale"
    _write_four_views(stale / "views", (240, 240, 240))
    _mock_blender(monkeypatch)
    stale_payload = run_surface_plan(
        recipe,
        stale,
        benchmark=benchmark,
        cluster="shoulder_l",
        apply=True,
        allow_region_weld=True,
    )
    assert stale_payload["apply_status"] == "applied"
    assert stale_payload["scores"] == []

    fresh = tmp_path / "fresh"
    _write_four_views(fresh / "views", (240, 240, 240))

    def _rewrite_all(cmd: list[str], **_kwargs: object) -> object:
        job = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        _write_sphere(Path(job["before_stl"]), 3)
        _write_sphere(Path(job["after_stl"]), 3)
        _write_four_views(fresh / "views", (10, 10, 10))

        class _Proc:
            returncode = 0
            stdout = "SURFACE_WELD_OK\n"
            stderr = ""

        return _Proc()

    monkeypatch.setattr("meshops.proportion.surface_plan.subprocess.run", _rewrite_all)
    fresh_payload = run_surface_plan(
        recipe,
        fresh,
        benchmark=benchmark,
        cluster="shoulder_l",
        apply=True,
        allow_region_weld=True,
    )
    assert fresh_payload["apply_status"] == "applied"
    assert len(fresh_payload["scores"]) == 4
    assert {row["role"] for row in fresh_payload["scores"]} == {
        "front",
        "left",
        "three_quarter",
        "back",
    }
    for row in fresh_payload["scores"]:
        assert "threshold" not in row
        assert "iou_pass" not in row
        assert "iou" in row and "dice" in row


def test_guard_welds_stl_triangle_soup(tmp_path: Path) -> None:
    """Binary STL soup is judged as shells, not one component per triangle."""
    from meshops.proportion.surface_plan import _guard_stats

    box = trimesh.creation.box(extents=(0.2, 0.1, 0.16))
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for face in box.faces:
        start = len(vertices)
        for index in face:
            vertices.append([float(item) for item in box.vertices[index]])
        faces.append([start, start + 1, start + 2])
    soup = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    path = tmp_path / "soup.stl"
    soup.export(path)
    loaded = trimesh.load(path, force="mesh", process=False)
    assert isinstance(loaded, trimesh.Trimesh)
    assert len(loaded.split(only_watertight=False)) == len(loaded.faces)
    stats = _guard_stats(path)
    assert stats.components == 1
    assert stats.faces == len(box.faces)


def test_plan_only_omits_stale_frozen_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A later plan-only run does not report a frozen recipe from an earlier apply."""
    _mock_blender(monkeypatch)
    recipe = _shoulder_recipe(tmp_path / "blockout_recipe.json")
    out = tmp_path / "out"
    applied = run_surface_plan(
        recipe,
        out,
        cluster="shoulder_l",
        apply=True,
        allow_region_weld=True,
    )
    assert any("frozen" in path for path in applied["paths"])
    body = json.loads(recipe.read_text(encoding="utf-8"))
    body["parts"][0]["rx_m"] = 0.07
    recipe.write_text(json.dumps(body), encoding="utf-8")
    planned = run_surface_plan(recipe, out)
    assert planned["apply_status"] == "plan_only"
    assert all(
        "frozen" not in path and "archive" not in path and "weld" not in path
        for path in planned["paths"]
    )
    assert (out / "frozen" / "blockout_recipe.json").is_file()


class _Vec:
    def __init__(self, data: object) -> None:
        values = list(data)  # type: ignore[arg-type]
        self.v = [float(item) for item in values]

    def __sub__(self, other: _Vec) -> _Vec:
        return _Vec([a - b for a, b in zip(self.v, other.v, strict=True)])

    def __add__(self, other: _Vec) -> _Vec:
        return _Vec([a + b for a, b in zip(self.v, other.v, strict=True)])

    def __truediv__(self, scalar: float) -> _Vec:
        return _Vec([item / float(scalar) for item in self.v])

    @property
    def length(self) -> float:
        return math.sqrt(sum(item * item for item in self.v))

    def normalized(self) -> _Vec:
        span = self.length or 1.0
        return _Vec([item / span for item in self.v])

    def rotation_difference(self, _other: _Vec) -> _Vec:
        return self

    def to_matrix(self) -> _Mat:
        return _Mat("R", self.v)

    def to_4x4(self) -> _Mat:
        return _Mat("R4", self.v)


class _Mat:
    def __init__(self, kind: str, data: object = None) -> None:
        self.kind = kind
        self.data = data

    def to_4x4(self) -> _Mat:
        return self

    def __matmul__(self, other: _Mat) -> _Mat:
        return _Mat("mul", (self, other))

    @staticmethod
    def Translation(vec: object) -> _Mat:
        return _Mat("T", vec)

    @staticmethod
    def Scale(factor: float, size: int, axis: tuple[int, int, int]) -> _Mat:
        return _Mat("S", (factor, size, axis))

    @staticmethod
    def Identity(size: int) -> _Mat:
        return _Mat("I", size)


class _Euler:
    def __init__(self, angles: tuple[float, float, float], order: str = "XYZ") -> None:
        self.angles = angles
        self.order = order

    def to_matrix(self) -> _Mat:
        return _Mat("E", self.angles)


class _Mesh:
    def __init__(self, name: str) -> None:
        self.name = name
        self.verts: list[object] | None = None
        self.faces: list[object] | None = None
        self.remesh_voxel_size: float | None = None
        self.use_remesh_preserve_volume: bool | None = None
        self.remesh_voxel_adaptivity: float | None = None

    def from_pydata(self, verts: list[object], _edges: list[object], faces: list[object]) -> None:
        self.verts = verts
        self.faces = faces

    def update(self) -> None:
        return None


class _Obj:
    def __init__(self, name: str) -> None:
        self.name = name
        self.data = _Mesh(name + "_mesh")
        self.scale = (1.0, 1.0, 1.0)
        self.matrix_world: object = None
        self.mode = "OBJECT"
        self.selected = False
        self.props: dict[str, str] = {}

    def select_set(self, value: bool) -> None:
        self.selected = value

    def __setitem__(self, key: str, value: str) -> None:
        self.props[key] = value


class _BpyHarness:
    def __init__(self) -> None:
        self.log: list[object] = []
        self.meshes: list[_Mesh] = []
        self.active: _Obj | None = None
        self.object: _Obj | None = None


def _install_bpy(monkeypatch: pytest.MonkeyPatch) -> _BpyHarness:
    """In-process bpy stand-in so the driver dispatch runs without Blender."""
    harness = _BpyHarness()

    class _ObjectOps:
        def select_all(self, action: str = "SELECT") -> None:
            harness.log.append(("select_all", action))

        def delete(self) -> None:
            harness.log.append(("delete",))

        def mode_set(self, mode: str) -> None:
            harness.log.append(("mode_set", mode))

        def transform_apply(
            self, location: bool = False, rotation: bool = False, scale: bool = False
        ) -> None:
            harness.log.append(("transform_apply", location, rotation, scale))

        def join(self) -> None:
            harness.log.append(("join",))

        def voxel_remesh(self) -> set[str]:
            active = harness.active
            assert active is not None
            harness.log.append(
                (
                    "voxel_remesh",
                    active.data.remesh_voxel_size,
                    active.data.use_remesh_preserve_volume,
                    active.data.remesh_voxel_adaptivity,
                )
            )
            return {"FINISHED"}

    class _MeshOps:
        def _spawn(self, op: str, kwargs: dict[str, object]) -> None:
            harness.log.append((op, kwargs))
            harness.active = _Obj(op)
            harness.object = harness.active

        def primitive_cylinder_add(self, **kwargs: object) -> None:
            self._spawn("primitive_cylinder_add", kwargs)

        def primitive_cube_add(self, **kwargs: object) -> None:
            self._spawn("primitive_cube_add", kwargs)

        def primitive_uv_sphere_add(self, **kwargs: object) -> None:
            self._spawn("primitive_uv_sphere_add", kwargs)

    class _WmOps:
        def stl_export(self, filepath: str) -> set[str]:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).write_bytes(b"stl")
            harness.log.append(("stl_export", filepath))
            return {"FINISHED"}

    class _Meshes:
        def new(self, name: str) -> _Mesh:
            mesh = _Mesh(name)
            harness.meshes.append(mesh)
            return mesh

    class _Objects:
        def new(self, name: str, mesh: _Mesh) -> _Obj:
            obj = _Obj(name)
            obj.data = mesh
            return obj

    class _Link:
        def link(self, obj: _Obj) -> None:
            harness.log.append(("link", obj.name))

    class _Collection:
        objects = _Link()

    class _Units:
        system = "METRIC"
        scale_length = 1.0

    class _Scene:
        unit_settings = _Units()
        collection = _Collection()

    class _ActiveSlot:
        def __get__(self, _instance: object, _owner: type) -> _Obj | None:
            return harness.active

        def __set__(self, _instance: object, value: _Obj | None) -> None:
            harness.active = value

    class _ViewObjects:
        active = _ActiveSlot()

    class _ViewLayer:
        objects = _ViewObjects()

    class _Context:
        scene = _Scene()
        view_layer = _ViewLayer()

        @property
        def object(self) -> _Obj | None:
            return harness.object

        @property
        def active_object(self) -> _Obj | None:
            return harness.active

    class _Ops:
        object = _ObjectOps()
        mesh = _MeshOps()
        wm = _WmOps()

    bpy_mod = types.ModuleType("bpy")
    bpy_mod.ops = _Ops()  # type: ignore[attr-defined]
    bpy_mod.context = _Context()  # type: ignore[attr-defined]
    bpy_mod.data = types.SimpleNamespace(meshes=_Meshes(), objects=_Objects())  # type: ignore[attr-defined]

    class _BMesh:
        def __init__(self) -> None:
            self.verts: list[object] = []

        def from_mesh(self, _mesh: object) -> None:
            return None

        def to_mesh(self, _mesh: object) -> None:
            return None

        def free(self) -> None:
            return None

    def _remove_doubles(_working: object, verts: list[object], dist: float) -> None:
        harness.log.append(("remove_doubles", dist, len(verts)))

    bmesh_mod = types.ModuleType("bmesh")
    bmesh_ops = types.ModuleType("bmesh.ops")
    bmesh_ops.remove_doubles = _remove_doubles  # type: ignore[attr-defined]
    bmesh_mod.ops = bmesh_ops  # type: ignore[attr-defined]
    bmesh_mod.new = lambda: _BMesh()  # type: ignore[attr-defined]
    mathutils_mod = types.ModuleType("mathutils")
    mathutils_mod.Vector = _Vec  # type: ignore[attr-defined]
    mathutils_mod.Matrix = _Mat  # type: ignore[attr-defined]
    mathutils_mod.Euler = _Euler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bpy", bpy_mod)
    monkeypatch.setitem(sys.modules, "bmesh", bmesh_mod)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils_mod)
    return harness


def _run_driver(
    tmp_path: Path, members: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> _BpyHarness:
    harness = _install_bpy(monkeypatch)
    tmp_path.mkdir(parents=True, exist_ok=True)
    job = tmp_path / "job.json"
    before = tmp_path / "before.stl"
    after = tmp_path / "after.stl"
    job.write_text(
        json.dumps(
            {
                "members": members,
                "voxel_coarse_m": 0.02,
                "voxel_fine_m": 0.014,
                "before_stl": str(before),
                "after_stl": str(after),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["blender", "--", str(job)],
    )
    from meshops.proportion.surface_weld_bpy import main

    main()
    return harness


def _ops(harness: _BpyHarness) -> list[str]:
    names: list[str] = []
    for item in harness.log:
        if isinstance(item, tuple):
            names.append(str(item[0]))
    return names


def test_driver_builds_cylinder_capsule_trap_box_and_ellipsoid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver entry builds each recipe kind. A cylinder is not a UV sphere."""
    cylinder = _cylinder(
        "RECIPE_hip_bridge_l",
        "hip_bridge",
        (-0.12, 0.0, 0.9),
        (-0.2, 0.0, 0.9),
    )
    trap = _trap_box("RECIPE_torso_oval_hip", "torso", (0.0, 0.0, 0.9))
    hip = _run_driver(tmp_path / "hip", [cylinder, trap], monkeypatch)
    hip_ops = _ops(hip)
    assert "primitive_cylinder_add" in hip_ops
    assert "primitive_uv_sphere_add" not in hip_ops
    assert any(mesh.verts is not None and len(mesh.verts) == 8 for mesh in hip.meshes)
    assert ("voxel_remesh", 0.02, True, 0.0) in hip.log
    assert ("voxel_remesh", 0.014, True, 0.0) in hip.log
    assert hip_ops.count("remove_doubles") == 2

    capsule = _cylinder(
        "RECIPE_neck",
        "neck",
        (0.0, 0.0, 1.45),
        (0.0, 0.0, 1.55),
        kind="capsule",
    )
    box = {
        "name": "RECIPE_torso_oval_chest",
        "kind": "box",
        "center": [0.0, 0.0, 1.25],
        "top_half_width_m": 0.12,
        "half_depth_m": 0.08,
        "z_bottom_m": 1.15,
        "z_top_m": 1.35,
    }
    neck = _run_driver(tmp_path / "neck", [capsule, box], monkeypatch)
    neck_ops = _ops(neck)
    assert "primitive_cylinder_add" in neck_ops
    assert "primitive_cube_add" in neck_ops
    assert "primitive_uv_sphere_add" not in neck_ops

    ellipsoid = _ellipsoid("RECIPE_deltoid_soft_l", "deltoid_soft", (-0.08, 0.0, 0.2))
    ellipsoid["rotation_euler_deg"] = [12.0, 0.0, 0.0]
    chest = _ellipsoid("RECIPE_torso_oval_chest", "torso", (0.08, 0.0, 0.2))
    balls = _run_driver(tmp_path / "balls", [ellipsoid, chest], monkeypatch)
    ball_ops = _ops(balls)
    assert ball_ops.count("primitive_uv_sphere_add") == 2
    assert "primitive_cylinder_add" not in ball_ops


@pytest.mark.blender
def test_surface_weld_opt_in_blender(tmp_path: Path) -> None:
    """Real Blender 5.2 weld. Skipped unless MESHOPS_SURFACE_BLENDER=1."""
    if os.environ.get("MESHOPS_SURFACE_BLENDER") != "1":
        pytest.skip("not part of default pytest; set MESHOPS_SURFACE_BLENDER=1")
    from meshops.proportion.surface_plan import _checked_blender

    blender = _checked_blender()
    assert blender.is_file()
    recipe = _write_recipe(
        tmp_path / "blockout_recipe.json",
        [
            _ellipsoid("RECIPE_deltoid_soft_l", "deltoid_soft", (-0.08, 0.0, 0.2)),
            _ellipsoid("RECIPE_torso_oval_chest", "torso", (0.08, 0.0, 0.2)),
        ],
    )
    out = tmp_path / "out"
    payload = run_surface_plan(
        recipe,
        out,
        cluster="shoulder_l",
        apply=True,
        allow_region_weld=True,
    )
    assert payload["apply_status"] == "applied"
    assert payload["ok"] is False
    assert (out / "weld" / "shoulder_l.stl").is_file()
    hip_recipe = _write_recipe(
        tmp_path / "hip_recipe.json",
        [
            _cylinder(
                "RECIPE_hip_bridge_l",
                "hip_bridge",
                (-0.04, 0.0, 0.2),
                (0.04, 0.0, 0.2),
            ),
            _trap_box("RECIPE_torso_oval_hip", "torso", (0.0, 0.0, 0.2)),
        ],
    )
    hip_out = tmp_path / "hip"
    hip_payload = run_surface_plan(
        hip_recipe,
        hip_out,
        cluster="hip_l",
        apply=True,
        allow_region_weld=True,
    )
    assert hip_payload["apply_status"] == "applied"
    assert hip_payload["ok"] is False
    assert (hip_out / "weld" / "hip_l.stl").is_file()
