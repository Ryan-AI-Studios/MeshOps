"""Track 0140 — character-detail region report. Authoring only, not print success.

Synthetic benchmarks under tmp_path. No Blender and no work/rogue-v3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from meshops.cli import app
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.benchmark_multiview import (
    CROP_FRACTIONS,
    CROP_NAMES,
    MULTIVIEW_BENCHMARK_HONESTY,
    ROLES,
    CandidateProvenance,
    CandidateRecord,
    Candidates,
    CaptureContract,
    CropRecord,
    FileProvenance,
    MultiviewBenchmark,
    ReferenceRecord,
    TopologyContext,
    TopologyPair,
)
from meshops.proportion.character_detail import (
    DETAIL_SCHEMA_VERSION,
    CharacterDetail,
    run_character_detail,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import DETAIL_HONESTY, SURFACE_HONESTY
from meshops.proportion.silhouette import SILHOUETTE_SCHEMA_VERSION

_REPO = Path(__file__).resolve().parents[1]
_RUNNER = CliRunner()


def _explode_blender(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("find_blender called")


def _pil() -> Any:
    pytest.importorskip("PIL")
    from PIL import Image  # type: ignore[import-untyped,import-not-found]

    return Image


def _png(path: Path, kind: str) -> Path:
    """16x16 PNG. ``dark`` is a small square so content-bbox IoU can be 1."""
    image_cls = _pil()
    img = image_cls.new("RGB", (16, 16), (255, 255, 255))
    if kind == "dark":
        for x in range(4, 8):
            for y in range(4, 8):
                img.putpixel((x, y), (0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def _provenance(path: Path, *, sha: str | None = None) -> FileProvenance:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest() if sha is None else sha
    return FileProvenance(path=str(path.resolve()), byte_size=len(data), sha256=digest)


def _crop(name: str, role: str, subject: str, png: Path) -> CropRecord:
    return CropRecord(
        name=name,
        role=role,
        subject=subject,
        crop_source="content_bbox_fraction",
        fractions=CROP_FRACTIONS[name],
        png=str(png.resolve()),
    )


def _write_benchmark(
    path: Path,
    image: Path,
    *,
    ok: bool = False,
    figures: list[str] | None = None,
    sha: str | None = None,
    crops: list[CropRecord] | None = None,
    appearance: list[str] | None = None,
) -> Path:
    figures = figures or ["rogue"]
    prov = _provenance(image, sha=sha)
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
        crops=crops or [],
        topology_context=TopologyPair(
            meshops=TopologyContext(status="not_supplied", note="context only"),
            meshy=TopologyContext(status="not_supplied", note="context only"),
        ),
        messages=[],
        paths=[],
        appearance_preview=appearance or [],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return path


def _pair_crops(
    root: Path,
    *,
    face: str = "dark",
    other: str = "dark",
    tag: str = "before",
) -> list[CropRecord]:
    """Reference and meshops crops for every region and role. Meshy is ignored."""
    crops: list[CropRecord] = []
    for name in CROP_NAMES:
        kind = face if name == "face" else other
        for role in ROLES:
            ref = _png(root / tag / role / f"{name}_reference.png", "dark")
            mesh = _png(root / tag / role / f"{name}_meshops.png", kind)
            crops.append(_crop(name, role, "reference", ref))
            crops.append(_crop(name, role, "meshops", mesh))
            crops.append(
                _crop(name, role, "meshy", _png(root / tag / role / f"{name}_meshy.png", "empty"))
            )
    return crops


def _bundle(
    tmp_path: Path,
    *,
    ok: bool = False,
    figures: list[str] | None = None,
    sha: str | None = None,
    face: str = "dark",
    appearance: list[str] | None = None,
    name: str = "multiview_benchmark.json",
) -> Path:
    image = _png(tmp_path / "front.png", "dark")
    crops = _pair_crops(tmp_path / "src", face=face)
    return _write_benchmark(
        tmp_path / name,
        image,
        ok=ok,
        figures=figures,
        sha=sha,
        crops=crops,
        appearance=appearance,
    )


def _surface(path: Path, apply_status: str) -> Path:
    body = {
        "schema_version": "1.0.0",
        "honesty": SURFACE_HONESTY,
        "ok": False,
        "status": "verdict_pending",
        "needs_user_input": False,
        "rank": None,
        "n_parts": 131,
        "apply_status": apply_status,
        "clusters": [],
        "gaps": {},
        "messages": [],
        "paths": [],
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _invoke(benchmark: Path, out: Path, *extra: str):
    return _RUNNER.invoke(
        app,
        [
            "proportion",
            "character-detail",
            "--benchmark",
            str(benchmark),
            "--out",
            str(out),
            *extra,
        ],
    )


def _minimal_detail() -> dict[str, Any]:
    return {
        "schema_version": DETAIL_SCHEMA_VERSION,
        "honesty": DETAIL_HONESTY,
        "ok": False,
        "status": "verdict_pending",
        "needs_user_input": False,
        "rank": None,
        "compare_status": "baseline_only",
        "geometry_baseline": "recipe_primitives",
        "benchmark_status": "verdict_pending",
        "view_conflict": False,
        "appearance_preview": [],
        "appearance_preview_after": [],
        "appearance_in_iou": False,
        "remaining_gap": "after views absent; geometric gap not measured",
        "regions": [],
        "messages": [],
        "paths": [],
    }


def test_t0_honesty_cli_mcp_and_no_bpy() -> None:
    """T0: honesty, CLI verb, MCP name, no stdin, module does not name bpy."""
    honesty = (_REPO / "src/meshops/proportion/honesty.py").read_text(encoding="utf-8")
    cli = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    server = (_REPO / "src/meshops/mcp/server.py").read_text(encoding="utf-8")
    tools = (_REPO / "src/meshops/mcp/tools.py").read_text(encoding="utf-8")
    module = (_REPO / "src/meshops/proportion/character_detail.py").read_text(encoding="utf-8")
    assert DETAIL_HONESTY == "proportion_character_detail_not_mesh_or_print_success"
    assert "DETAIL_HONESTY" in honesty
    assert 'command("character-detail")' in cli
    assert "input(" not in module
    assert "import bpy" not in module
    assert "find_blender" not in module
    assert "mesh_proportion_character_detail" in server
    assert "def mesh_proportion_character_detail" in tools


def test_t1_schema_rejects_unknown_field_and_old_version() -> None:
    """T1: extra fields and schema 0.9.0 are rejected."""
    body = _minimal_detail()
    CharacterDetail.model_validate(body)
    with pytest.raises(ValidationError):
        CharacterDetail.model_validate({**body, "not_a_field": 1})
    with pytest.raises(ValidationError):
        CharacterDetail.model_validate({**body, "schema_version": "0.9.0"})


def test_t2_omitted_verdict_writes_pending_bundle(tmp_path: Path) -> None:
    """T2: omitted verdict writes the bundle, pending, exit 0, baseline_only."""
    benchmark = _bundle(tmp_path)
    out = tmp_path / "out"
    result = _invoke(benchmark, out)
    assert result.exit_code == 0, result.output
    payload = json.loads((out / "character_detail.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "verdict_pending"
    assert payload["compare_status"] == "baseline_only"
    assert (out / "character_detail.md").is_file()


def test_t3_accept_without_after_stays_baseline_only(tmp_path: Path) -> None:
    """T3: accept without --after stays ok false and baseline_only."""
    benchmark = _bundle(tmp_path)
    out = tmp_path / "out"
    result = _invoke(benchmark, out, "--verdict", "accept")
    assert result.exit_code == 0, result.output
    payload = json.loads((out / "character_detail.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "baseline_only"
    assert payload["compare_status"] == "baseline_only"


def test_t4_regions_sort_by_iou_and_rank_stays_null(tmp_path: Path) -> None:
    """T4: crop ids, ascending mean iou, null rank, no character_score."""
    benchmark = _bundle(tmp_path, face="empty")
    payload = run_character_detail(benchmark, tmp_path / "out")
    names = [row["name"] for row in payload["regions"]]
    assert names == list(CROP_NAMES)
    assert set(names) == set(CROP_FRACTIONS)
    assert payload["regions"][0]["mean_iou"] == 0.0
    assert payload["regions"][1]["mean_iou"] > 0.9
    assert payload["rank"] is None
    assert "character_score" not in payload


def test_t5_sha_mismatch_and_benchmark_ok_is_not_success(tmp_path: Path) -> None:
    """T5: reference SHA mismatch is detail_failed. Benchmark ok does not set ok."""
    image = _png(tmp_path / "front.png", "dark")
    bad = _write_benchmark(tmp_path / "bad.json", image, sha="0" * 64)
    with pytest.raises(ProportionError) as exc:
        run_character_detail(bad, tmp_path / "bad-out")
    assert exc.value.code == "detail_failed"
    benchmark = _bundle(tmp_path / "good", ok=True)
    payload = run_character_detail(benchmark, tmp_path / "good-out")
    assert payload["ok"] is False
    assert "benchmark ok is not detail success" in payload["messages"]


def test_t6_missing_benchmark_does_not_call_blender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T6: missing benchmark is detail_failed and does not call find_blender."""
    monkeypatch.setattr("meshops.escalate.discover.find_blender", _explode_blender)
    with pytest.raises(ProportionError) as exc:
        run_character_detail(tmp_path / "missing.json", tmp_path / "out")
    assert exc.value.code == "detail_failed"


def test_t7_plan_only_surface_is_not_a_body_mesh(tmp_path: Path) -> None:
    """T7: plan_only sets recipe_primitives and says it is not a body mesh."""
    benchmark = _bundle(tmp_path)
    surface = _surface(tmp_path / "surface_plan.json", "plan_only")
    payload = run_character_detail(benchmark, tmp_path / "out", surface=surface)
    assert payload["geometry_baseline"] == "recipe_primitives"
    assert payload["surface_apply_status"] == "plan_only"
    assert any("not a body mesh" in message for message in payload["messages"])
    assert payload["ok"] is False


def test_t8_applied_surface_is_one_junction(tmp_path: Path) -> None:
    """T8: applied weld is one_junction and still not accepted without a verdict."""
    benchmark = _bundle(tmp_path)
    surface = _surface(tmp_path / "surface_plan.json", "applied")
    payload = run_character_detail(benchmark, tmp_path / "out", surface=surface)
    assert payload["geometry_baseline"] == "one_junction"
    assert payload["ok"] is False
    assert any("one junction" in message for message in payload["messages"])


def test_t9_appearance_paths_stay_out_of_iou(tmp_path: Path) -> None:
    """T9: appearance_preview paths are copied and appearance_in_iou is false."""
    preview = str((tmp_path / "appearance" / "front.png").resolve())
    benchmark = _bundle(tmp_path, appearance=[preview])
    payload = run_character_detail(benchmark, tmp_path / "out")
    assert payload["appearance_preview"] == [preview]
    assert payload["appearance_in_iou"] is False


def test_t10_two_figures_need_a_choice_even_with_accept(tmp_path: Path) -> None:
    """T10: two figures and no --figure stay needs_user_input and ok false."""
    before = _bundle(tmp_path / "before", figures=["a", "b"])
    after = _bundle(tmp_path / "after", figures=["a", "b"], name="after.json")
    payload = run_character_detail(
        before,
        tmp_path / "out",
        after=after,
        verdict="accept",
    )
    assert payload["needs_user_input"] is True
    assert payload["ok"] is False
    assert payload["status"] == "verdict_pending"
    assert payload["compare_status"] == "compared"


def test_t11_silhouette_schema_stays_1_2_0() -> None:
    """T11: silhouette schema constant is still 1.2.0."""
    assert SILHOUETTE_SCHEMA_VERSION == "1.2.0"


def test_t12_opposite_sign_deltas_set_view_conflict(tmp_path: Path) -> None:
    """T12: opposite-sign role deltas set view_conflict and block accept."""
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    before_crops = _pair_crops(before_root / "src", face="dark")
    after_crops: list[CropRecord] = []
    for name in CROP_NAMES:
        for role in ROLES:
            ref = _png(after_root / "src" / role / f"{name}_reference.png", "dark")
            if name == "face" and role == "front":
                kind = "empty"
            elif name == "face" and role == "left":
                kind = "dark"
            else:
                kind = "dark"
            mesh = _png(after_root / "src" / role / f"{name}_meshops.png", kind)
            after_crops.append(_crop(name, role, "reference", ref))
            after_crops.append(_crop(name, role, "meshops", mesh))
    # front before matches (dark/dark). left before is empty meshops so delta can flip.
    flipped: list[CropRecord] = []
    for crop in before_crops:
        if crop.name == "face" and crop.role == "left" and crop.subject == "meshops":
            empty = _png(before_root / "src" / "left" / "face_meshops_empty.png", "empty")
            flipped.append(_crop(crop.name, crop.role, crop.subject, empty))
        else:
            flipped.append(crop)
    before_image = _png(before_root / "front.png", "dark")
    after_image = _png(after_root / "front.png", "dark")
    before = _write_benchmark(before_root / "bench.json", before_image, crops=flipped)
    after = _write_benchmark(after_root / "bench.json", after_image, crops=after_crops)
    payload = run_character_detail(before, tmp_path / "out", after=after, verdict="accept")
    face = next(row for row in payload["regions"] if row["name"] == "face")
    assert face["view_conflict"] is True
    assert payload["view_conflict"] is True
    assert payload["ok"] is False
    assert payload["status"] == "view_conflict"


def test_t13_catalog_is_56() -> None:
    """T13: live catalog length and the setup-launch source pin are 56."""
    assert len(TOOL_NAMES) == 56
    assert "mesh_proportion_character_detail" in TOOL_NAMES
    mcp_test = (_REPO / "tests/test_mcp_server.py").read_text(encoding="utf-8")
    launch = (_REPO / "tests/test_proportion_setup_launch.py").read_text(encoding="utf-8")
    assert "len(TOOL_NAMES) == 56" in mcp_test
    assert "len(TOOL_NAMES) == 56" in launch
    assert "len(TOOL_NAMES) == 53" not in mcp_test
    assert "len(TOOL_NAMES) == 53" not in launch


def test_t14_markdown_honesty_and_no_html(tmp_path: Path) -> None:
    """T14: markdown has honesty and image_left_hand, and the writer emits no HTML."""
    benchmark = _bundle(tmp_path)
    out = tmp_path / "out"
    run_character_detail(benchmark, out)
    text = (out / "character_detail.md").read_text(encoding="utf-8")
    assert DETAIL_HONESTY in text
    assert "image_left_hand" in text
    assert "not print success" in text
    assert "not anatomical left" in text
    assert list(out.rglob("*.html")) == []


def test_t15_image_left_hand_is_not_renamed(tmp_path: Path) -> None:
    """T15: region id stays image_left_hand."""
    benchmark = _bundle(tmp_path)
    payload = run_character_detail(benchmark, tmp_path / "out")
    names = [row["name"] for row in payload["regions"]]
    assert "image_left_hand" in names
    assert "hand_r" not in names
    module = (_REPO / "src/meshops/proportion/character_detail.py").read_text(encoding="utf-8")
    assert "hand_r" not in module


def test_t16_remaining_gap_names_absent_after_views(tmp_path: Path) -> None:
    """T16: remaining_gap is generated and a baseline-only run names absent after views."""
    benchmark = _bundle(tmp_path)
    payload = run_character_detail(benchmark, tmp_path / "out")
    assert payload["remaining_gap"]
    assert "after views absent" in payload["remaining_gap"]


def test_t17_no_breast_costume_verb_and_no_hair_tier() -> None:
    """T17: breast-costume stays absent and this module does not mention HairTier."""
    cli = (_REPO / "src/meshops/cli.py").read_text(encoding="utf-8")
    module = (_REPO / "src/meshops/proportion/character_detail.py").read_text(encoding="utf-8")
    assert "blockout-breast-costume" not in cli
    assert "HairTier" not in module


def test_t18_after_multi_figure_still_needs_a_choice(tmp_path: Path) -> None:
    """T18: a multi-figure --after needs --figure even when the baseline has one."""
    before = _bundle(tmp_path / "before", figures=["woman"])
    after = _bundle(tmp_path / "after", figures=["woman", "man"], name="after.json")
    pending = run_character_detail(before, tmp_path / "pending", after=after, verdict="accept")
    assert pending["needs_user_input"] is True
    assert pending["ok"] is False
    assert pending["status"] == "verdict_pending"
    assert pending["compare_status"] == "compared"
    assert pending["view_conflict"] is False
    resolved = run_character_detail(
        before,
        tmp_path / "resolved",
        after=after,
        figure="woman",
        verdict="accept",
    )
    assert resolved["needs_user_input"] is False
    assert resolved["ok"] is True
    assert resolved["status"] == "accepted"


def test_t19_require_verdict_exits_2_unless_accepted(tmp_path: Path) -> None:
    """T19: --require-verdict exits 2 unless status is accepted. The bundle is written."""
    benchmark = _bundle(tmp_path / "base")
    pending_out = tmp_path / "pending"
    pending = _invoke(benchmark, pending_out, "--require-verdict")
    assert pending.exit_code == 2, pending.output
    written = json.loads((pending_out / "character_detail.json").read_text(encoding="utf-8"))
    assert written["status"] == "verdict_pending"
    before = _bundle(tmp_path / "before")
    after = _bundle(tmp_path / "after", name="after.json")
    accepted_out = tmp_path / "accepted"
    accepted = _invoke(
        before,
        accepted_out,
        "--after",
        str(after),
        "--verdict",
        "accept",
        "--require-verdict",
    )
    assert accepted.exit_code == 0, accepted.output
    payload = json.loads((accepted_out / "character_detail.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["status"] == "accepted"
