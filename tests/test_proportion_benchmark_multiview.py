"""Track 0138 — four-view visual benchmark. Authoring QA only, not print success.

Default path uses synthetic PNGs. No Blender and no work/rogue-v3.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw
from pydantic import ValidationError
from typer.testing import CliRunner

from meshops.cli import app
from meshops.escalate.discover import find_blender
from meshops.escalate.errors import EscalateError
from meshops.hosted.convert import glb_to_stl
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.benchmark_multiview import (
    ALIGNMENT_MESSAGE,
    CROP_FRACTIONS,
    MULTIVIEW_BENCHMARK_HONESTY,
    ROLES,
    MultiviewBenchmark,
    crop_box_px,
    load_benchmark,
    normalize_mesh,
    run_benchmark_multiview,
    score_aligned_masks,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.silhouette import (
    SILHOUETTE_SCHEMA_VERSION,
    run_silhouette_compare,
)

_REPO = Path(__file__).resolve().parents[1]
_RUNNER = CliRunner()


def _write_gray_pixels(path: Path, pixels: list[tuple[int, int]]) -> None:
    image = Image.new("RGB", (80, 120), (180, 180, 180))
    draw = ImageDraw.Draw(image)
    for x, y in pixels:
        draw.point((x, y), fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_png(path: Path, *, foreground: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (80, 120), (255, 255, 255))
    if foreground:
        draw = ImageDraw.Draw(image)
        draw.rectangle((24, 16, 56, 100), fill=(30, 30, 30))
    image.save(path)


def _checklist(path: Path, *, figures: list[str], multi: bool) -> None:
    payload = {
        "schema_version": "1.0.0",
        "height_m": 1.72,
        "in_scope_figures": figures,
        "multi_figure": multi,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reference(root: Path, *, multi: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        _write_png(root / f"{role}.png", foreground=True)
    figures = ["woman", "man"] if multi else ["woman"]
    _checklist(root / "package_checklist.json", figures=figures, multi=multi)
    return root


def _views(root: Path, *, foreground: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        _write_png(root / f"{role}.png", foreground=foreground)
    return root


def _invoke(reference: Path, out: Path, views: Path, *extra: str):
    args = [
        "proportion",
        "benchmark-multiview",
        "--reference",
        str(reference),
        "--out",
        str(out),
        "--meshops-views",
        str(views),
        "--meshy-views",
        str(views),
        *extra,
    ]
    return _RUNNER.invoke(app, args)


def test_t0_honesty_cli_mcp_and_forbidden_calls() -> None:
    """T0: honesty, CLI verb, MCP name, no stdin, no remesh/repair strings."""
    honesty = (_REPO / "src/meshops/proportion/honesty.py").read_text(encoding="utf-8")
    assert "MULTIVIEW_BENCHMARK_HONESTY" in honesty
    assert "proportion_multiview_benchmark_not_mesh_or_print_success" in honesty
    assert MULTIVIEW_BENCHMARK_HONESTY == (
        "proportion_multiview_benchmark_not_mesh_or_print_success"
    )
    cli = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    assert "benchmark-multiview" in cli
    assert "mesh_proportion_benchmark_multiview" in TOOL_NAMES
    server = (_REPO / "src/meshops/mcp/server.py").read_text(encoding="utf-8")
    assert "mesh_proportion_benchmark_multiview" in server
    module = (_REPO / "src/meshops/proportion/benchmark_multiview.py").read_text(encoding="utf-8")
    assert "input(" not in module
    assert "voxel_remesh" not in module
    assert "meshing_repair" not in module


def test_t1_model_rejects_unknown_and_loader_rejects_old_schema(tmp_path: Path) -> None:
    """T1: extra field forbidden; schema other than 1.0.0 raises before validate."""
    with pytest.raises(ValidationError) as exc:
        MultiviewBenchmark.model_validate({"schema_version": "1.0.0", "nope": 1})
    assert any(err["type"] == "extra_forbidden" for err in exc.value.errors())
    path = tmp_path / "multiview_benchmark.json"
    path.write_text('{"schema_version": "0.9.0", "ok": true}\n', encoding="utf-8")
    with pytest.raises(ProportionError) as err:
        load_benchmark(path)
    assert err.value.code == "benchmark_failed"


def test_t2_omitted_verdict_pending_exit_0(tmp_path: Path) -> None:
    """T2: bundle written, ok false, verdict_pending, exit 0."""
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    out = tmp_path / "out"
    result = _invoke(reference, out, views)
    assert result.exit_code == 0, result.output
    payload = load_benchmark(out / "multiview_benchmark.json")
    assert payload.ok is False
    assert payload.status == "verdict_pending"
    assert payload.visual_verdict is None
    assert payload.rank is None


def test_t3_accept_ok_rank_null(tmp_path: Path) -> None:
    """T3: --verdict accept with four roles → ok true, rank null, honesty present."""
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    out = tmp_path / "out"
    result = _invoke(reference, out, views, "--verdict", "accept")
    assert result.exit_code == 0, result.output
    payload = load_benchmark(out / "multiview_benchmark.json")
    assert payload.ok is True
    assert payload.status == "accepted"
    assert payload.rank is None
    assert payload.honesty == MULTIVIEW_BENCHMARK_HONESTY
    assert ALIGNMENT_MESSAGE in payload.messages


def test_t4_aligned_iou_and_back_role_still_rejected(tmp_path: Path) -> None:
    """T4: identical IoU ~1; empty vs dark IoU 0; silhouette back still raises."""
    same = tmp_path / "same.png"
    blank = tmp_path / "blank.png"
    _write_png(same, foreground=True)
    _write_png(blank, foreground=False)
    identical = score_aligned_masks(same, same)
    assert identical["iou"] == pytest.approx(1.0, abs=1e-6)
    disjoint = score_aligned_masks(blank, same)
    assert disjoint["iou"] == 0.0
    assert disjoint["dice"] == 0.0
    gray_ell = tmp_path / "gray_ell.png"
    gray_bar = tmp_path / "gray_bar.png"
    ell = [(x, y) for x in range(20, 36) for y in range(20, 100)]
    ell += [(x, y) for x in range(36, 60) for y in range(80, 100)]
    bar = [(x, y) for x in range(20, 50) for y in range(20, 100)]
    _write_gray_pixels(gray_ell, ell)
    _write_gray_pixels(gray_bar, bar)
    gray = score_aligned_masks(gray_ell, gray_bar)
    assert gray["mask_method"] == "corner_median"
    assert gray["iou"] < 0.95
    with pytest.raises(ProportionError):
        run_silhouette_compare(same, tmp_path / "sil", mesh_view=same, view_role="back")


def test_t5_crop_boxes_match_frozen_fractions(tmp_path: Path) -> None:
    """T5: boxes recompute from CROP_FRACTIONS; crop_source is content_bbox_fraction."""
    content = (10, 20, 109, 119)
    width = content[2] - content[0] + 1
    height = content[3] - content[1] + 1
    assert width == 100 and height == 100
    for name, fractions in CROP_FRACTIONS.items():
        fx0, fy0, fx1, fy1 = fractions
        expected = (
            content[0] + round(fx0 * width),
            content[1] + round(fy0 * height),
            content[0] + round(fx1 * width) - 1,
            content[1] + round(fy1 * height) - 1,
        )
        assert crop_box_px(content, fractions) == expected
        assert name in {
            "face",
            "torso",
            "image_left_hand",
            "image_right_hand",
            "feet",
        }
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    out = tmp_path / "out"
    run_benchmark_multiview(reference, out, meshops_views=views, meshy_views=views)
    payload = load_benchmark(out / "multiview_benchmark.json")
    assert payload.crops
    assert {crop.crop_source for crop in payload.crops} == {"content_bbox_fraction"}
    assert {crop.name for crop in payload.crops} == set(CROP_FRACTIONS)


def test_t6_y_up_vertex_grounds_then_yaw_180() -> None:
    """T6: +Z maps to -Y, height 1.72, zmin 0; yaw 180 sends that point to +Y."""
    vertices = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    grounded, _record = normalize_mesh(mesh, up="y", height_m=1.72, yaw_deg=0.0)
    assert grounded.vertices[0, 1] < 0.0
    z_extent = float(grounded.vertices[:, 2].max() - grounded.vertices[:, 2].min())
    assert z_extent == pytest.approx(1.72)
    assert float(grounded.vertices[:, 2].min()) == pytest.approx(0.0, abs=1e-9)
    spun, _record2 = normalize_mesh(mesh, up="y", height_m=1.72, yaw_deg=180.0)
    assert spun.vertices[0, 1] > 0.0


def test_t7_glb_bake_does_not_normalize(tmp_path: Path) -> None:
    """T7: Y-up box stays Y-long through glb_to_stl; Z-long only after normalize."""
    source = (_REPO / "src/meshops/hosted/convert.py").read_text(encoding="utf-8")
    assert "normalize_mesh" not in source
    box = trimesh.creation.box(extents=(0.4, 2.0, 0.3))
    glb = tmp_path / "box.glb"
    box.export(glb)
    baked = tmp_path / "box.stl"
    glb_to_stl(glb, baked)
    loaded = trimesh.load(baked, force="mesh")
    assert isinstance(loaded, trimesh.Trimesh)
    assert loaded.extents[1] > loaded.extents[0]
    assert loaded.extents[1] > loaded.extents[2]
    normalized, _record = normalize_mesh(loaded, up="y", height_m=1.72, yaw_deg=0.0)
    assert float(normalized.extents[2]) == pytest.approx(1.72, abs=1e-4)
    assert normalized.extents[2] > normalized.extents[1]


def test_t8_topology_does_not_change_ok(tmp_path: Path) -> None:
    """T8: box stats stay context; recipe candidate mesh_stats stays null; ok holds."""
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    glb = tmp_path / "box.glb"
    trimesh.creation.box(extents=(0.2, 0.4, 0.2)).export(glb)
    out = tmp_path / "out"
    payload = run_benchmark_multiview(
        reference,
        out,
        meshops_views=views,
        meshy_views=views,
        meshy_glb=glb,
        verdict="accept",
    )
    assert payload["ok"] is True
    assert payload["rank"] is None
    meshops = payload["topology_context"]["meshops"]
    meshy = payload["topology_context"]["meshy"]
    assert meshops["representation"] == "recipe_primitives"
    assert meshops["mesh_stats"] is None
    assert meshy["mesh_stats"]["faces"] >= 1
    assert meshy["status"] == "context_only"
    assert "0141 owns print" in meshy["note"]


def test_t9_multi_figure_needs_user_even_on_accept(tmp_path: Path) -> None:
    """T9: multi-figure without --figure stays needs_user_input and ok false."""
    reference = _reference(tmp_path / "ref", multi=True)
    views = _views(tmp_path / "views")
    out = tmp_path / "out"
    result = _invoke(reference, out, views, "--verdict", "accept")
    assert result.exit_code == 0, result.output
    payload = load_benchmark(out / "multiview_benchmark.json")
    assert payload.needs_user_input is True
    assert payload.ok is False
    assert payload.status == "verdict_pending"
    with pytest.raises(ProportionError):
        run_benchmark_multiview(
            reference,
            tmp_path / "out2",
            meshops_views=views,
            meshy_views=views,
            figure="nobody",
            force=True,
        )


def test_t10_role_filenames_ignore_depth(tmp_path: Path) -> None:
    """T10: front/left/three_quarter/back score; depth PNG is ignored."""
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    _write_png(views / "three_quarter_depth.png", foreground=True)
    out = tmp_path / "out"
    payload = run_benchmark_multiview(reference, out, meshops_views=views, meshy_views=views)
    assert set(payload["views"]) == set(ROLES)
    joined = "\n".join(payload["messages"])
    assert "stem" not in joined
    assert "_stem_conflicts_with_role" not in joined
    for role in ROLES:
        assert (out / "views" / "meshops" / f"{role}.png").is_file()
        assert (out / "views" / "meshy" / f"{role}.png").is_file()
        assert not (out / "views" / "meshops" / "three_quarter_depth.png").exists()


def test_t11_require_verdict_exits_2_after_write(tmp_path: Path) -> None:
    """T11: --require-verdict exits 2 when the bundle is still pending."""
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    out = tmp_path / "out"
    result = _invoke(reference, out, views, "--require-verdict")
    assert result.exit_code == 2, result.output
    payload = load_benchmark(out / "multiview_benchmark.json")
    assert payload.status == "verdict_pending"
    assert payload.ok is False


def test_t12_catalog_is_54() -> None:
    """T12: live catalog length and the setup-launch source pin are 54."""
    assert len(TOOL_NAMES) == 54
    mcp_test = (_REPO / "tests/test_mcp_server.py").read_text(encoding="utf-8")
    launch = (_REPO / "tests/test_proportion_setup_launch.py").read_text(encoding="utf-8")
    assert "len(TOOL_NAMES) == 54" in mcp_test
    assert "len(TOOL_NAMES) == 54" in launch
    assert "len(TOOL_NAMES) == 53" not in mcp_test
    assert "len(TOOL_NAMES) == 53" not in launch


def test_t13_markdown_contact_sheet_has_no_html_writer(tmp_path: Path) -> None:
    """T13: markdown names both candidates, four roles, honesty; writer has no .html."""
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    out = tmp_path / "out"
    run_benchmark_multiview(reference, out, meshops_views=views, meshy_views=views)
    text = (out / "multiview_benchmark.md").read_text(encoding="utf-8")
    for role in ROLES:
        assert role in text
    assert "meshops" in text
    assert "meshy" in text
    assert MULTIVIEW_BENCHMARK_HONESTY in text
    module = (_REPO / "src/meshops/proportion/benchmark_multiview.py").read_text(encoding="utf-8")
    assert ".html" not in module
    assert ".html" not in text


def test_t14_silhouette_schema_stays() -> None:
    """T14: silhouette schema constant is still 1.2.0."""
    assert SILHOUETTE_SCHEMA_VERSION == "1.2.0"


def test_render_rejects_non_52_blender(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A present Blender that reports 4.2 fails closed before any render."""
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    fake = tmp_path / "blender-4.2.exe"
    fake.write_bytes(b"")

    def _found(**_kwargs: object) -> Path:
        return fake

    def _reject(blender: Path, **_kwargs: object) -> str:
        raise EscalateError(
            f"Blender 4.2.0 found; MeshOps requires 5.2.x LTS. Path={blender}",
            code="blender_version",
        )

    monkeypatch.setattr("meshops.proportion.benchmark_multiview.find_blender", _found)
    monkeypatch.setattr(
        "meshops.proportion.benchmark_multiview.require_blender_52",
        _reject,
    )
    with pytest.raises(ProportionError) as err:
        run_benchmark_multiview(
            reference,
            tmp_path / "out",
            meshops_views=views,
            meshy_glb=tmp_path / "absent.glb",
        )
    assert err.value.code == "benchmark_render_unavailable"
    assert "4.2.0" in str(err.value)
    assert not (tmp_path / "out" / "multiview_benchmark.json").exists()


def test_view_dir_does_not_probe_blender(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-rendered views never locate Blender."""

    def _boom(**_kwargs: object) -> Path:
        raise AssertionError("view-dir mode must not locate Blender")

    monkeypatch.setattr("meshops.proportion.benchmark_multiview.find_blender", _boom)
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    payload = run_benchmark_multiview(
        reference,
        tmp_path / "out",
        meshops_views=views,
        meshy_views=views,
    )
    assert payload["status"] == "verdict_pending"


def _setup_with_recipe(directory: Path, recipe_text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    setup = directory / "setup_blockout_recipe.py"
    setup.write_text("import bpy\n", encoding="utf-8")
    (directory / "blockout_recipe.json").write_text(recipe_text, encoding="utf-8")
    return setup


def test_malformed_recipe_rejected_in_view_dir(tmp_path: Path) -> None:
    """A present sibling recipe that is not JSON fails closed in view-dir mode."""
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    setup = _setup_with_recipe(tmp_path / "setup", "{")
    with pytest.raises(ProportionError) as err:
        run_benchmark_multiview(
            reference,
            tmp_path / "out",
            meshops_setup=setup,
            meshops_views=views,
            meshy_views=views,
            verdict="accept",
        )
    assert err.value.code == "benchmark_failed"
    assert "not valid JSON" in str(err.value)
    assert not (tmp_path / "out" / "multiview_benchmark.json").exists()


def test_recipe_without_parts_rejected_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present recipe without a parts list fails before Blender is located."""

    def _boom(**_kwargs: object) -> Path:
        raise AssertionError("malformed recipe must fail before Blender")

    monkeypatch.setattr("meshops.proportion.benchmark_multiview.find_blender", _boom)
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    setup = _setup_with_recipe(tmp_path / "setup", '{"parts": "nope"}')
    with pytest.raises(ProportionError) as err:
        run_benchmark_multiview(
            reference,
            tmp_path / "out",
            meshops_setup=setup,
            meshy_views=views,
            meshy_glb=tmp_path / "absent.glb",
        )
    assert err.value.code == "benchmark_failed"
    assert "parts list" in str(err.value)


def test_recipe_parts_count_recorded(tmp_path: Path) -> None:
    """len(parts) is recorded when the sibling recipe is valid JSON."""
    reference = _reference(tmp_path / "ref")
    views = _views(tmp_path / "views")
    setup = _setup_with_recipe(tmp_path / "setup", '{"parts": [{}, {}, {}]}')
    payload = run_benchmark_multiview(
        reference,
        tmp_path / "out",
        meshops_setup=setup,
        meshops_views=views,
        meshy_views=views,
    )
    assert payload["candidates"]["meshops"]["n_parts"] == 3


def _blender_or_skip() -> Path:
    try:
        blender = find_blender(require=True)
    except EscalateError:
        pytest.skip("Blender 5.2 LTS not found")
    assert blender is not None
    from meshops.escalate.version import require_blender_52

    try:
        require_blender_52(blender)
    except EscalateError as exc:
        pytest.skip(f"Blender version not 5.2: {exc}")
    return blender


@pytest.mark.blender
def test_blender_two_primitives_render_four_views(tmp_path: Path) -> None:
    """Opt-in Blender render: two RECIPE primitives and a tiny STL at 960x1280.

    Default ``uv run pytest`` skips this. The shared ``blender`` marker still
    runs older handoff tests. Set ``MESHOPS_BENCHMARK_BLENDER=1`` to run it.
    """
    if os.environ.get("MESHOPS_BENCHMARK_BLENDER") != "1":
        pytest.skip("not part of default pytest; set MESHOPS_BENCHMARK_BLENDER=1")
    _blender_or_skip()
    reference = _reference(tmp_path / "ref")
    setup = tmp_path / "setup_blockout_recipe.py"
    setup.write_text(
        "import bpy\n"
        "bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0.4), radius=0.15)\n"
        "bpy.context.active_object.name = 'RECIPE_a'\n"
        "bpy.ops.mesh.primitive_cube_add(location=(0, 0, 1.2), scale=(0.08, 0.08, 0.2))\n"
        "bpy.context.active_object.name = 'RECIPE_b'\n",
        encoding="utf-8",
    )
    glb = tmp_path / "box.glb"
    trimesh.creation.box(extents=(0.2, 0.3, 0.2)).export(glb)
    out = tmp_path / "out"
    payload = run_benchmark_multiview(
        reference,
        out,
        meshops_setup=setup,
        meshy_glb=glb,
    )
    assert payload["ok"] is False
    assert payload["status"] == "verdict_pending"
    for candidate in ("meshops", "meshy"):
        for role in ROLES:
            png = out / "views" / candidate / f"{role}.png"
            assert png.is_file()
            with Image.open(png) as image:
                assert image.size == (960, 1280)
