"""Track 0014 — multi-view package scaffold & package_checklist.json (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.test_proportion import make_package, write_solid_png
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.analyze import analyze_proportion
from meshops.proportion.checklist import (
    PACKAGE_CHECKLIST_FILENAME,
    PACKAGE_CHECKLIST_SCHEMA_VERSION,
    PackageChecklist,
    find_package_checklist,
    load_package_checklist,
    parse_figures,
    resolve_checklist_pair,
    write_package_checklist,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.scaffold import PNG_1X1_BYTES, scaffold_package
from meshops.proportion.template import blank_assist_document

runner = CliRunner()


# ---------------------------------------------------------------------------
# Checklist model
# ---------------------------------------------------------------------------


def test_checklist_roundtrip__valid__ok(tmp_path: Path) -> None:
    cl = PackageChecklist(
        subject="Rogue",
        height_m=1.72,
        pose="a_pose",
        heroic_vs_realistic="stylized",
        package_mode="single",
        package_role="combined",
    )
    path = tmp_path / PACKAGE_CHECKLIST_FILENAME
    write_package_checklist(path, cl)
    back = load_package_checklist(path)
    assert back.schema_version == PACKAGE_CHECKLIST_SCHEMA_VERSION
    assert back.subject == "Rogue"
    assert back.height_m == pytest.approx(1.72)
    assert back.pose == "a_pose"
    raw = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        PackageChecklist.model_validate({**raw, "extra_field": True})


def test_checklist_normalize__two_figures__multi_true() -> None:
    cl = PackageChecklist(
        in_scope_figures=["female", "male"],
        multi_figure=False,
    )
    assert cl.multi_figure is True
    cl2 = PackageChecklist(in_scope_figures=["only_one"], multi_figure=False)
    assert cl2.multi_figure is False


# ---------------------------------------------------------------------------
# Scaffold trees
# ---------------------------------------------------------------------------


def test_scaffold_single__dirs__checklist_source(tmp_path: Path) -> None:
    out = tmp_path / "pkg"
    result = scaffold_package(out, subject="Solo", height_m=1.7, pose="a_pose")
    assert result.mode == "single"
    assert result.analyze_hint is None
    checklist = out / PACKAGE_CHECKLIST_FILENAME
    source = out / "SOURCE.txt"
    assert checklist.is_file()
    assert source.is_file()
    cl = load_package_checklist(checklist)
    assert cl.package_mode == "single"
    assert cl.package_role == "combined"
    text = source.read_text(encoding="utf-8")
    assert "# MeshOps multi-view package" in text
    assert "package_mode: single" in text
    assert "pose: a_pose" in text
    assert "# Honesty: layout only" in text


def test_scaffold_dual__tree__leaf_roles(tmp_path: Path) -> None:
    out = tmp_path / "dual"
    result = scaffold_package(
        out,
        dual=True,
        subject="Rogue",
        height_m=1.72,
        pose="a_pose",
        heroic_vs_realistic="stylized",
        source_kind="imagen",
    )
    assert result.mode == "dual"
    assert result.analyze_hint is not None
    assert result.analyze_hint.name == "proportion"
    assert (out / "proportion").is_dir()
    assert (out / "character").is_dir()
    root = load_package_checklist(out / PACKAGE_CHECKLIST_FILENAME)
    assert root.package_mode == "dual"
    assert root.package_role is None
    prop = load_package_checklist(out / "proportion" / PACKAGE_CHECKLIST_FILENAME)
    assert prop.package_role == "proportion"
    assert prop.wardrobe_tier == "two_piece_midriff"
    assert prop.height_m == pytest.approx(1.72)
    assert prop.subject == "Rogue"
    char = load_package_checklist(out / "character" / PACKAGE_CHECKLIST_FILENAME)
    assert char.package_role == "character"
    assert char.wardrobe_tier == "costume"
    # Leaf snapshot of shared fields
    assert char.height_m == pytest.approx(1.72)
    text = (out / "SOURCE.txt").read_text(encoding="utf-8")
    assert "analyze_hint" in text
    assert "package_mode: dual" in text


def test_scaffold_exists__no_force__error(tmp_path: Path) -> None:
    out = tmp_path / "exists"
    scaffold_package(out, subject="First")
    with pytest.raises(ProportionError) as ei:
        scaffold_package(out, subject="Second")
    assert ei.value.code == "checklist_exists"


def test_scaffold_force__overwrites_checklist(tmp_path: Path) -> None:
    out = tmp_path / "force"
    scaffold_package(out, subject="First")
    scaffold_package(out, subject="Second", force=True)
    cl = load_package_checklist(out / PACKAGE_CHECKLIST_FILENAME)
    assert cl.subject == "Second"


def test_scaffold_with_template__assist_pose(tmp_path: Path) -> None:
    out = tmp_path / "templ"
    scaffold_package(out, pose="a_pose", with_template=True)
    assist_path = out / "landmarks_assist.json"
    assert assist_path.is_file()
    doc = json.loads(assist_path.read_text(encoding="utf-8"))
    assert doc["pose"] == "a_pose"
    # R6: blank_assist_document still defaults unknown; scaffold post-processes
    blank = blank_assist_document()
    assert blank["pose"] == "unknown"


def test_scaffold_dual_mode_conflict__error() -> None:
    result = runner.invoke(
        app,
        ["proportion", "scaffold", "--out", "x", "--dual", "--mode", "single"],
    )
    assert result.exit_code != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "dual" in combined.lower() or "mode" in combined.lower() or result.exit_code == 2


def test_scaffold_json_shape__ok(tmp_path: Path) -> None:
    out = tmp_path / "json_pkg"
    result = runner.invoke(
        app,
        [
            "proportion",
            "scaffold",
            "--out",
            str(out),
            "--dual",
            "--subject",
            "Rogue",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["mode"] == "dual"
    assert isinstance(payload["paths"], list)
    assert len(payload["paths"]) >= 2
    assert payload["analyze_hint"] is not None
    assert payload["analyze_hint"].endswith("proportion") or "proportion" in payload[
        "analyze_hint"
    ].replace("\\", "/")


def test_stub_images__png_constant_valid(tmp_path: Path) -> None:
    import struct
    import zlib

    assert PNG_1X1_BYTES.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in PNG_1X1_BYTES
    assert b"IEND" in PNG_1X1_BYTES
    # Validate IHDR/IDAT CRCs (stdlib; no Pillow required)
    offset = 8
    while offset < len(PNG_1X1_BYTES):
        length = struct.unpack(">I", PNG_1X1_BYTES[offset : offset + 4])[0]
        tag = PNG_1X1_BYTES[offset + 4 : offset + 8]
        data = PNG_1X1_BYTES[offset + 8 : offset + 8 + length]
        crc_stored = struct.unpack(">I", PNG_1X1_BYTES[offset + 8 + length : offset + 12 + length])[
            0
        ]
        crc_calc = zlib.crc32(tag + data) & 0xFFFFFFFF
        assert crc_stored == crc_calc, f"bad CRC for chunk {tag!r}"
        offset += 12 + length
        if tag == b"IEND":
            break
    out = tmp_path / "stubs"
    scaffold_package(out, stub_images=True, include_back_stub=True)
    for name in ("front.png", "left.png", "three_quarter.png", "back.png"):
        data = (out / name).read_bytes()
        assert data == PNG_1X1_BYTES
    # Optional pillow 1x1 size check
    try:
        from PIL import Image  # type: ignore[import-untyped,import-not-found]
    except ImportError:
        return
    with Image.open(out / "front.png") as im:
        assert im.size == (1, 1)


def test_scaffold_invalid_fields__invalid_checklist(tmp_path: Path) -> None:
    from meshops.proportion.errors import ProportionError

    with pytest.raises(ProportionError) as ei:
        scaffold_package(tmp_path / "bad_h", height_m=0.0)
    assert ei.value.code == "invalid_checklist"

    with pytest.raises(ProportionError) as ei2:
        scaffold_package(tmp_path / "bad_src", source_kind="bogus")  # type: ignore[arg-type]
    assert ei2.value.code == "invalid_checklist"


# ---------------------------------------------------------------------------
# Analyze integration (R1 / R7)
# ---------------------------------------------------------------------------


def _write_minimal_views(d: Path) -> None:
    for key in ("front", "left", "three_quarter"):
        write_solid_png(d / f"{key}.png", 64, 64)


def test_analyze_height_leaf_over_parent(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    prop = root / "proportion"
    prop.mkdir(parents=True)
    _write_minimal_views(prop)
    write_package_checklist(
        root / PACKAGE_CHECKLIST_FILENAME,
        PackageChecklist(
            package_mode="dual",
            package_role=None,
            height_m=1.50,
            pose="a_pose",
        ),
    )
    write_package_checklist(
        prop / PACKAGE_CHECKLIST_FILENAME,
        PackageChecklist(
            package_mode="dual",
            package_role="proportion",
            height_m=1.80,
            pose="a_pose",
        ),
    )
    report = analyze_proportion(prop, partial_ok=True, run_heuristic_frame=False)
    assert report.height_m == pytest.approx(1.80)
    assert any("1.8" in m and "package_checklist" in m for m in report.messages)


def test_analyze_height_parent_when_leaf_null(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    prop = root / "proportion"
    prop.mkdir(parents=True)
    _write_minimal_views(prop)
    write_package_checklist(
        root / PACKAGE_CHECKLIST_FILENAME,
        PackageChecklist(
            package_mode="dual",
            package_role=None,
            height_m=1.65,
            pose="a_pose",
        ),
    )
    write_package_checklist(
        prop / PACKAGE_CHECKLIST_FILENAME,
        PackageChecklist(
            package_mode="dual",
            package_role="proportion",
            height_m=None,
            pose="a_pose",
        ),
    )
    report = analyze_proportion(prop, partial_ok=True, run_heuristic_frame=False)
    assert report.height_m == pytest.approx(1.65)
    assert any("1.65" in m and "package_checklist" in m for m in report.messages)


def test_analyze_height_cli_overrides(tmp_path: Path) -> None:
    d = make_package(tmp_path)
    write_package_checklist(
        d / PACKAGE_CHECKLIST_FILENAME,
        PackageChecklist(
            package_mode="single",
            package_role="combined",
            height_m=1.50,
            pose="a_pose",
        ),
    )
    report = analyze_proportion(d, height_m=1.90, partial_ok=False, run_heuristic_frame=False)
    assert report.height_m == pytest.approx(1.90)
    # CLI wins — no "from package_checklist" for height when CLI set
    assert not any("height_m=" in m and "package_checklist" in m for m in report.messages)


def test_analyze_pose_after_assist_unknown(tmp_path: Path) -> None:
    d = make_package(tmp_path)  # no assist → pose unknown
    write_package_checklist(
        d / PACKAGE_CHECKLIST_FILENAME,
        PackageChecklist(
            package_mode="single",
            package_role="combined",
            pose="a_pose",
            height_m=1.7,
        ),
    )
    report = analyze_proportion(d, run_heuristic_frame=False)
    assert report.pose == "a_pose"
    assert any("pose from package_checklist" in m for m in report.messages)


def test_analyze_multifigure_checklist__needs_user_input(tmp_path: Path) -> None:
    d = make_package(tmp_path)
    write_package_checklist(
        d / PACKAGE_CHECKLIST_FILENAME,
        PackageChecklist(
            package_mode="single",
            package_role="combined",
            in_scope_figures=["a", "b"],
            multi_figure=True,
            pose="a_pose",
        ),
    )
    report = analyze_proportion(d, run_heuristic_frame=False)
    assert report.quality.multi_figure is True
    assert report.quality.needs_user_input is True


def test_analyze_no_checklist__unchanged(tmp_path: Path) -> None:
    d = make_package(tmp_path)
    report = analyze_proportion(d, run_heuristic_frame=False)
    assert report.height_m is None
    assert report.pose == "unknown"
    assert not any("package_checklist" in m for m in report.messages)


# ---------------------------------------------------------------------------
# Discovery / figures
# ---------------------------------------------------------------------------


def test_find_checklist__dual_leaf_parent(tmp_path: Path) -> None:
    root = tmp_path / "views"
    prop = root / "proportion"
    prop.mkdir(parents=True)
    write_package_checklist(
        root / PACKAGE_CHECKLIST_FILENAME,
        PackageChecklist(package_mode="dual", package_role=None),
    )
    # Leaf has no checklist → find walks parent
    found = find_package_checklist(prop)
    assert found is not None
    assert found.resolve() == (root / PACKAGE_CHECKLIST_FILENAME).resolve()
    leaf, parent = resolve_checklist_pair(prop)
    assert leaf is None
    assert parent is not None
    assert parent.resolve() == (root / PACKAGE_CHECKLIST_FILENAME).resolve()


def test_find_checklist__non_dual_name__no_parent_lookup(tmp_path: Path) -> None:
    root = tmp_path / "views"
    other = root / "other_name"
    other.mkdir(parents=True)
    write_package_checklist(
        root / PACKAGE_CHECKLIST_FILENAME,
        PackageChecklist(package_mode="single", package_role="combined"),
    )
    found = find_package_checklist(other)
    assert found is None
    leaf, parent = resolve_checklist_pair(other)
    assert leaf is None
    assert parent is None


def test_figures_parse__strip_drop_empty() -> None:
    assert parse_figures("female, male") == ["female", "male"]
    assert parse_figures("a,, ,b ,") == ["a", "b"]
    assert parse_figures("") == []
    assert parse_figures(None) == []
    assert parse_figures("solo") == ["solo"]
    # single figure → multi false on model
    cl = PackageChecklist(in_scope_figures=parse_figures("solo"))
    assert cl.multi_figure is False
    cl2 = PackageChecklist(in_scope_figures=parse_figures("a,b"))
    assert cl2.multi_figure is True


def test_scaffold_dual_leaf_exists_no_force(tmp_path: Path) -> None:
    """Dual leaf checklist exists without force → checklist_exists (preflight, no partial write)."""
    out = tmp_path / "dual2"
    scaffold_package(out, dual=True, subject="First")
    # Full dual re-run without force should refuse before mutating
    with pytest.raises(ProportionError) as ei:
        scaffold_package(out, dual=True, subject="Second")
    assert ei.value.code == "checklist_exists"
    assert load_package_checklist(out / PACKAGE_CHECKLIST_FILENAME).subject == "First"

    # Root removed, leaf remains → still refuse without writing a new root
    (out / PACKAGE_CHECKLIST_FILENAME).unlink()
    source_before = (out / "SOURCE.txt").read_text(encoding="utf-8")
    with pytest.raises(ProportionError) as ei2:
        scaffold_package(out, dual=True, subject="Third")
    assert ei2.value.code == "checklist_exists"
    assert not (out / PACKAGE_CHECKLIST_FILENAME).is_file()
    assert (out / "SOURCE.txt").read_text(encoding="utf-8") == source_before
    leaf_subject = load_package_checklist(out / "proportion" / PACKAGE_CHECKLIST_FILENAME).subject
    assert leaf_subject == "First"
