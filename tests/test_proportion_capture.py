"""Track 0016 — proportion assist capture (offline; no Blender; no network)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.organic.session import create_session, load_session
from meshops.proportion.capture import (
    NOTE_PREFIX_ASSIST,
    attach_to_organic_session,
    build_assist_from_dump,
    build_assist_from_px,
    build_assist_from_reproject,
    emit_dump_script,
    merge_assist_docs,
    parse_assist_empty_name,
    run_capture,
    strip_blender_dup_suffix,
    write_assist,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import CAPTURE_HONESTY
from meshops.proportion.template import blank_assist_document

runner = CliRunner()


def _minimal_png(path: Path, w: int = 100, h: int = 200) -> Path:
    """Write a valid minimal PNG (IHDR only) for view sizing."""
    # Minimal 1x1 PNG, then we still report via IHDR — use real dimensions in IHDR.
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    # empty IDAT with filter+RGB for one scanline would be needed for decoders;
    # meshops only reads IHDR — still need valid chunk structure.
    raw = b"\x00" + b"\x00\x00\x00" * w  # filter + RGB
    # pad rows for h
    raw = (b"\x00" + b"\x00\x00\x00" * w) * h
    idat = zlib.compress(raw)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    path.write_bytes(data)
    return path


def _px_doc(
    *,
    chin: list[float] | None = None,
    sole: list[float] | None = None,
    pose: str = "a_pose",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "kind": "assist_pixel_capture",
        "honesty": CAPTURE_HONESTY,
        "pose": pose,
        "multi_figure": False,
        "views": {
            "front": {
                "width_px": 100,
                "height_px": 200,
                "landmarks": {
                    "chin": chin if chin is not None else [50, 40],
                    "sole": sole if sole is not None else [50, 190],
                },
            }
        },
    }


def _dump_doc(empties: list[dict], view_sizes: dict | None = None) -> dict:
    return {
        "schema_version": "1.0.0",
        "kind": "assist_empty_dump",
        "honesty": CAPTURE_HONESTY,
        "view_sizes": view_sizes
        if view_sizes is not None
        else {"front": {"width_px": 100, "height_px": 200}},
        "empties": empties,
        "pose": "unknown",
        "multi_figure": False,
    }


def _merge_with_anchors(
    *,
    sole: list[float] | None = None,
    top: list[float] | None = None,
    midline_x: float | None = 50.0,
    left_chest: bool = False,
) -> dict:
    doc = blank_assist_document()
    front = doc["views"]["front"]["landmarks"]
    front["sole"] = sole if sole is not None else [50, 190]
    front["cranial_vertex"] = top if top is not None else [50, 20]
    if midline_x is not None:
        front["midline_x"] = [midline_x, 100]
    if left_chest:
        left = doc["views"]["left"]["landmarks"]
        left["chest_front"] = [70, 100]
        left["chest_back"] = [30, 100]
        left["sole"] = [50, 190]
        left["cranial_vertex"] = [50, 20]
    return doc


# ---------------------------------------------------------------------------
# Name parse
# ---------------------------------------------------------------------------


def test_capture__dump_name_parse() -> None:
    assert parse_assist_empty_name("ASSIST_front_chin") == ("front", "chin")
    assert parse_assist_empty_name("ASSIST_front_shoulder_l") == ("front", "shoulder_l")


def test_capture__dump_name_parse_three_quarter() -> None:
    assert parse_assist_empty_name("ASSIST_three_quarter_hair_crown") == (
        "three_quarter",
        "hair_crown",
    )


def test_capture__dump_name_strips_blender_suffix() -> None:
    assert strip_blender_dup_suffix("ASSIST_front_chin.001") == "ASSIST_front_chin"
    assert parse_assist_empty_name("ASSIST_front_chin.001") == ("front", "chin")


# ---------------------------------------------------------------------------
# Pixel / dump builders
# ---------------------------------------------------------------------------


def test_capture__px__writes_assist(tmp_path: Path) -> None:
    src = tmp_path / "landmark_capture.json"
    out = tmp_path / "landmarks_assist.json"
    src.write_text(json.dumps(_px_doc()) + "\n", encoding="utf-8")
    payload = run_capture(source="px", in_path=src, out_path=out, force=True)
    assert payload["ok"] is True
    assert payload["source"] == "px"
    assert payload["honesty"] == CAPTURE_HONESTY
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert "width_px" not in doc["views"]["front"]
    assert "height_px" not in doc["views"]["front"]
    assert doc["views"]["front"]["landmarks"]["chin"] == [50.0, 40.0]
    assert doc["views"]["front"]["landmarks"]["sole"] == [50.0, 190.0]
    assert payload["counts"]["landmarks"] >= 2


def test_capture__dump_skips_lm_seed() -> None:
    dump = _dump_doc(
        [
            {"name": "ASSIST_front_chin", "x_px": 10, "y_px": 20},
            {"name": "LM_chin", "x_px": 1, "y_px": 2},
            {"name": "SEED_thigh_l", "x_px": 3, "y_px": 4},
            {"name": "LM_HEIGHT", "x_px": 0, "y_px": 0},
            {"name": "LM_HU_0", "x_px": 0, "y_px": 0},
        ]
    )
    doc, messages, skipped = build_assist_from_dump(dump)
    assert doc["views"]["front"]["landmarks"]["chin"] == [10.0, 20.0]
    assert skipped >= 4
    assert any("LM_" in m or "SEED" in m or "skipped" in m for m in messages)


def test_capture__dump_missing_view_sizes() -> None:
    dump = _dump_doc(
        [{"name": "ASSIST_front_chin", "x_px": 10, "y_px": 20}],
        view_sizes={},
    )
    with pytest.raises(ProportionError) as ei:
        build_assist_from_dump(dump)
    assert ei.value.code == "capture_failed"
    assert "view_sizes" in str(ei.value)


def test_capture__empty__capture_empty(tmp_path: Path) -> None:
    src = tmp_path / "empty.json"
    out = tmp_path / "out.json"
    src.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "kind": "assist_pixel_capture",
                "honesty": CAPTURE_HONESTY,
                "pose": "unknown",
                "multi_figure": False,
                "views": {"front": {"width_px": 10, "height_px": 10, "landmarks": {"chin": None}}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProportionError) as ei:
        run_capture(source="px", in_path=src, out_path=out, force=True)
    assert ei.value.code == "capture_empty"


# ---------------------------------------------------------------------------
# Reproject + merge
# ---------------------------------------------------------------------------


def test_capture__reproject_needs_merge(tmp_path: Path) -> None:
    src = tmp_path / "guides.json"
    out = tmp_path / "assist.json"
    src.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "honesty": "proportion_guides_not_mesh_or_print_success",
                "height_m": 1.7,
                "empties": [
                    {
                        "name": "LM_chin",
                        "x_m": 0.0,
                        "y_m": 0.0,
                        "z_m": 1.5,
                        "kind": "landmark",
                        "source_id": "chin",
                        "display_size_m": 0.05,
                    }
                ],
                "ladder": [],
                "seeds": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProportionError) as ei:
        run_capture(source="reproject", in_path=src, out_path=out, force=True)
    assert ei.value.code == "capture_failed"
    assert "merge" in str(ei.value).lower()


def test_capture__reproject_needs_front_anchors(tmp_path: Path) -> None:
    src = tmp_path / "guides.json"
    merge_p = tmp_path / "merge.json"
    out = tmp_path / "assist.json"
    src.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "honesty": "proportion_guides_not_mesh_or_print_success",
                "height_m": 1.7,
                "empties": [
                    {
                        "name": "LM_chin",
                        "x_m": 0.0,
                        "y_m": 0.0,
                        "z_m": 1.5,
                        "kind": "landmark",
                        "source_id": "chin",
                        "display_size_m": 0.05,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_assist(merge_p, blank_assist_document(), force=True)
    with pytest.raises(ProportionError) as ei:
        run_capture(
            source="reproject",
            in_path=src,
            out_path=out,
            merge_path=merge_p,
            force=True,
        )
    assert ei.value.code == "capture_failed"
    assert "stature" in str(ei.value).lower() or "sole" in str(ei.value).lower()


def test_capture__reproject_with_merge(tmp_path: Path) -> None:
    src = tmp_path / "guides.json"
    merge_p = tmp_path / "merge.json"
    out = tmp_path / "assist.json"
    # figure_h_px = 190-20 = 170; H=1.7 → chin z=1.53 → z_frac=0.9 → y=190-0.9*170=37
    src.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "honesty": "proportion_guides_not_mesh_or_print_success",
                "height_m": 1.7,
                "empties": [
                    {
                        "name": "LM_chin",
                        "x_m": 0.0,
                        "y_m": 0.0,
                        "z_m": 1.53,
                        "kind": "landmark",
                        "source_id": "chin",
                        "display_size_m": 0.05,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_assist(merge_p, _merge_with_anchors(), force=True)
    payload = run_capture(
        source="reproject",
        in_path=src,
        out_path=out,
        merge_path=merge_p,
        force=True,
    )
    assert payload["ok"] is True
    doc = json.loads(out.read_text(encoding="utf-8"))
    chin = doc["views"]["front"]["landmarks"]["chin"]
    if isinstance(chin, dict):
        x, y = chin["x"], chin["y"]
        assert chin.get("confidence") == 0.75
    else:
        x, y = chin[0], chin[1]
    assert x == pytest.approx(50.0, abs=0.5)
    assert y == pytest.approx(37.0, abs=0.5)
    # merge preserved sole
    assert doc["views"]["front"]["landmarks"]["sole"] == [50.0, 190.0] or doc["views"]["front"][
        "landmarks"
    ]["sole"] == [50, 190]


def test_capture__merge_preserves_missing() -> None:
    base = blank_assist_document()
    base["views"]["front"]["landmarks"]["sole"] = [50, 190]
    base["views"]["front"]["landmarks"]["chin"] = [50, 40]
    new = blank_assist_document()
    new["views"]["front"]["landmarks"]["chin"] = None  # missing in new
    new["views"]["front"]["landmarks"]["navel"] = [50, 100]
    merged = merge_assist_docs(base, new, prefer_merge=False)
    assert merged["views"]["front"]["landmarks"]["sole"] == [50, 190]
    assert merged["views"]["front"]["landmarks"]["chin"] == [50, 40]
    assert merged["views"]["front"]["landmarks"]["navel"] == [50, 100]


def test_capture__prefer_merge() -> None:
    base = blank_assist_document()
    base["views"]["front"]["landmarks"]["chin"] = [1, 1]
    new = blank_assist_document()
    new["views"]["front"]["landmarks"]["chin"] = [9, 9]
    merged_new = merge_assist_docs(base, new, prefer_merge=False)
    merged_old = merge_assist_docs(base, new, prefer_merge=True)
    assert merged_new["views"]["front"]["landmarks"]["chin"] == [9, 9]
    assert merged_old["views"]["front"]["landmarks"]["chin"] == [1, 1]


# ---------------------------------------------------------------------------
# Dump script + CLI
# ---------------------------------------------------------------------------


def test_capture__emit_dump_script_self_contained() -> None:
    script = emit_dump_script()
    assert "import meshops" not in script
    assert "from meshops" not in script
    assert "import bpy" in script
    assert "assist_empty_dump.json" in script
    assert "bpy.data.filepath" in script
    assert CAPTURE_HONESTY in script
    assert "meshops_x_px" in script
    assert "OBJECT" in script


def test_capture__cli_json_shape(tmp_path: Path) -> None:
    src = tmp_path / "cap.json"
    out = tmp_path / "assist.json"
    src.write_text(json.dumps(_px_doc()), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "proportion",
            "capture",
            "--source",
            "px",
            "--in",
            str(src),
            "--out",
            str(out),
            "--force",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["source"] == "px"
    assert "out" in payload
    assert "counts" in payload
    assert "landmarks" in payload["counts"]
    assert "messages" in payload
    assert payload.get("honesty") == CAPTURE_HONESTY


def test_capture__cli_emit_dump_script_alone(tmp_path: Path) -> None:
    script_path = tmp_path / "dump_assist.py"
    result = runner.invoke(
        app,
        ["proportion", "capture", "--emit-dump-script", str(script_path), "--json"],
    )
    assert result.exit_code == 0, result.output
    assert script_path.is_file()
    text = script_path.read_text(encoding="utf-8")
    assert "meshops" not in text.split("import")[0] or "import meshops" not in text
    assert "import meshops" not in text
    assert "from meshops" not in text


# ---------------------------------------------------------------------------
# Organic attach
# ---------------------------------------------------------------------------


def test_attach_session__prefix_idempotent(tmp_path: Path) -> None:
    work = tmp_path / "work"
    manifest = create_session("capture attach test", work_root=work, session_id="oabcdef01234")
    assert manifest.session_id == "oabcdef01234"

    assist = tmp_path / "landmarks_assist.json"
    write_assist(assist, blank_assist_document() | {"pose": "a_pose"}, force=True)
    # fill one landmark so re-attach of real capture path still works
    doc = blank_assist_document()
    doc["views"]["front"]["landmarks"]["chin"] = [1, 2]
    write_assist(assist, doc, force=True)

    dest1 = attach_to_organic_session(
        "oabcdef01234",
        assist,
        work_root=work,
        note_prefix=NOTE_PREFIX_ASSIST,
        dest_basename="landmarks_assist.json",
    )
    dest2 = attach_to_organic_session(
        "oabcdef01234",
        assist,
        work_root=work,
        note_prefix=NOTE_PREFIX_ASSIST,
        dest_basename="landmarks_assist.json",
    )
    assert dest1 == dest2
    _, man = load_session("oabcdef01234", work_root=work)
    notes = [n for n in man.notes if n.startswith(NOTE_PREFIX_ASSIST)]
    assert len(notes) == 1
    assert notes[0].startswith(NOTE_PREFIX_ASSIST)


def test_capture__attach_session_via_run(tmp_path: Path) -> None:
    work = tmp_path / "work"
    create_session("run capture attach", work_root=work, session_id="oabcdef01235")
    src = tmp_path / "cap.json"
    out = tmp_path / "assist.json"
    src.write_text(json.dumps(_px_doc()), encoding="utf-8")
    payload = run_capture(
        source="px",
        in_path=src,
        out_path=out,
        force=True,
        attach_session="oabcdef01235",
        work_root=work,
    )
    assert payload["attach_dest"]
    _, man = load_session("oabcdef01235", work_root=work)
    assert any(n.startswith(NOTE_PREFIX_ASSIST) for n in man.notes)


def test_capture__honesty_constant() -> None:
    assert CAPTURE_HONESTY == "proportion_capture_not_mesh_or_print_success"


def test_capture__px_build_engine() -> None:
    doc, _messages = build_assist_from_px(_px_doc(chin=[12, 34], sole=[56, 78]))
    assert doc["views"]["front"]["landmarks"]["chin"] == [12.0, 34.0]
    assert "width_px" not in doc["views"]["front"]


def test_capture__reproject_left_needs_anchors() -> None:
    merge = _merge_with_anchors(left_chest=False)
    guides = {
        "schema_version": "1.0.0",
        "honesty": "proportion_guides_not_mesh_or_print_success",
        "height_m": 1.7,
        "empties": [
            {
                "name": "LM_chest_front",
                "x_m": 0.0,
                "y_m": 0.1,
                "z_m": 1.2,
                "kind": "landmark",
                "source_id": "chest_front",
                "display_size_m": 0.05,
            }
        ],
    }
    with pytest.raises(ProportionError) as ei:
        build_assist_from_reproject(guides, merge)
    assert ei.value.code == "capture_failed"
    assert "left" in str(ei.value).lower()


def test_capture__dump_views_dir_fills_sizes(tmp_path: Path) -> None:
    views = tmp_path / "views"
    views.mkdir()
    _minimal_png(views / "front.png", 120, 240)
    dump = _dump_doc(
        [{"name": "ASSIST_front_chin", "x_px": 60, "y_px": 50}],
        view_sizes={},
    )
    doc, _msgs, _sk = build_assist_from_dump(dump, views_dir=views)
    assert doc["views"]["front"]["landmarks"]["chin"] == [60.0, 50.0]
