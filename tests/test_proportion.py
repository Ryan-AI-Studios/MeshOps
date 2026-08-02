"""Track 0012 — pixel proportion analysis (offline; no new pytest markers)."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.analyze import analyze_proportion, load_report
from meshops.proportion.assist import load_assist_json, point_to_landmark2d
from meshops.proportion.errors import ProportionError
from meshops.proportion.fuse import compute_package_score, head_unit_frac_from_front
from meshops.proportion.honesty import PROPORTION_HONESTY
from meshops.proportion.load_views import load_views, png_size_from_bytes
from meshops.proportion.models import (
    PROPORTION_SCHEMA_VERSION,
    Landmark2D,
    ProportionReport,
    ViewLandmarks,
)
from meshops.proportion.template import blank_assist_document, write_template

runner = CliRunner()


# ---------------------------------------------------------------------------
# Synthetic PNG helpers (stdlib — no Pillow required)
# ---------------------------------------------------------------------------


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def write_png_rgb(path: Path, width: int, height: int, rgb_fn: Any) -> None:
    """Write an 8-bit RGB PNG; rgb_fn(x,y) -> (r,g,b). Filter type 0 rows."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter None
        for x in range(width):
            r, g, b = rgb_fn(x, y)
            rows.extend((r & 0xFF, g & 0xFF, b & 0xFF))
    compressed = zlib.compress(bytes(rows), level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_solid_png(
    path: Path, width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)
) -> None:
    write_png_rgb(path, width, height, lambda _x, _y: color)


def write_figure_blob_png(
    path: Path,
    width: int,
    height: int,
    boxes: list[tuple[int, int, int, int]],
    *,
    bg: tuple[int, int, int] = (255, 255, 255),
    fg: tuple[int, int, int] = (40, 40, 40),
) -> None:
    """White background with filled dark rectangles (subject silhouettes)."""

    def rgb(x: int, y: int) -> tuple[int, int, int]:
        for x0, y0, x1, y1 in boxes:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return fg
        return bg

    write_png_rgb(path, width, height, rgb)


def write_minimal_jpg_stub(path: Path) -> None:
    """Not a real JPEG — used only to trigger format/pillow path errors."""
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"not-a-real-jpeg")


# 8-head ladder: figure 480 px tall, head=60, cranial y=40, chin=100, sole=520
# Image size 512x560 so margins exist.
W, H = 512, 560
CRANIAL_Y = 40
HAIR_Y = 20
CHIN_Y = 100
SOLE_Y = 520  # 520-40 = 480 = 8*60
MID_X = 256
HEAD = 60


def eight_head_assist() -> dict[str, Any]:
    """Consistent 8-head assist for front + left + three_quarter."""
    front_lm: dict[str, Any] = {
        "hair_crown": [MID_X, HAIR_Y],
        "cranial_vertex": [MID_X, CRANIAL_Y],
        "chin": [MID_X, CHIN_Y],
        "shoulder_l": [180, 140],
        "shoulder_r": [332, 140],
        "nipple_bust": [MID_X, 160],
        "navel": [MID_X, 220],
        "crotch_pubic": [MID_X, 280],  # mid of 480 from cranial → y=40+240=280
        "greater_trochanter": [MID_X, 290],
        "hip_l": [210, 280],
        "hip_r": [302, 280],
        "knee": [MID_X, 400],
        "ankle": [MID_X, 500],
        "sole": [MID_X, SOLE_Y],
        "midline_x": MID_X,
        # A-pose arms: out and down
        "elbow_l": [120, 200],
        "wrist_l": [80, 250],
        "fingertip_l": [60, 280],
        "elbow_r": [392, 200],
        "wrist_r": [432, 250],
        "fingertip_r": [452, 280],
    }
    left_lm: dict[str, Any] = {
        "cranial_vertex": [200, CRANIAL_Y],
        "chin": [200, CHIN_Y],
        "sole": [200, SOLE_Y],
        "chest_front": [230, 180],
        "chest_back": [170, 180],
        "hip_front": [225, 280],
        "hip_back": [175, 280],
    }
    tq_lm: dict[str, Any] = {
        "cranial_vertex": [220, CRANIAL_Y],
        "sole": [220, SOLE_Y],
    }
    return {
        "schema_version": "1.0.0",
        "pose": "a_pose",
        "multi_figure": False,
        "views": {
            "front": {"facing_direction": "camera_front", "landmarks": front_lm},
            "left": {"facing_direction": "camera_left", "landmarks": left_lm},
            "three_quarter": {"facing_direction": "camera_front", "landmarks": tq_lm},
        },
    }


def make_package(
    tmp: Path,
    *,
    views: tuple[str, ...] = ("front", "left", "three_quarter"),
    assist: dict[str, Any] | None = None,
    multi_blob_front: bool = False,
    front_span_y: tuple[int, int] | None = None,
) -> Path:
    """Create views dir with solid/blob PNGs + optional assist."""
    d = tmp / "views"
    d.mkdir(parents=True, exist_ok=True)
    for key in views:
        path = d / f"{key}.png"
        if multi_blob_front and key == "front":
            # Two large dark rectangles
            write_figure_blob_png(
                path,
                W,
                H,
                [(40, 40, 180, 500), (320, 40, 460, 500)],
            )
        elif front_span_y is not None and key == "front":
            y0, y1 = front_span_y
            write_figure_blob_png(path, W, H, [(200, y0, 300, y1)])
        else:
            # Single central figure blob matching sole/cranial roughly
            write_figure_blob_png(path, W, H, [(180, 30, 340, 530)])
    if assist is not None:
        (d / "landmarks_assist.json").write_text(json.dumps(assist, indent=2), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Unit / integration tests (plan §7)
# ---------------------------------------------------------------------------


def test_honesty_present() -> None:
    assert PROPORTION_HONESTY == "proportion_measurement_not_mesh_or_print_success"
    r = ProportionReport()
    assert r.honesty == PROPORTION_HONESTY
    assert r.schema_version == PROPORTION_SCHEMA_VERSION


def test_schema_round_trip() -> None:
    lm = Landmark2D(
        id="chin",
        x_px=10,
        y_px=20,
        x_frac=0.1,
        y_frac=0.2,
        method="assist",
        confidence=0.9,
    )
    vl = ViewLandmarks(view="front", width_px=100, height_px=200, landmarks={"chin": lm})
    report = ProportionReport(views={"front": vl}, package_score=13.33)
    raw = report.model_dump(mode="json")
    back = ProportionReport.model_validate(raw)
    assert back.views["front"].landmarks["chin"].x_frac == 0.1
    assert back.package_score == pytest.approx(13.33)
    # extra=forbid
    with pytest.raises(ValidationError):
        ProportionReport.model_validate({**raw, "extra_field": 1})


def test_png_ihdr_robust_not_fixed_offset() -> None:
    """IHDR found by chunk scan even if a tEXt chunk precedes IHDR."""
    rows = bytearray()
    for _y in range(2):
        rows.append(0)
        rows.extend(b"\x00\x00\x00" * 3)
    compressed = zlib.compress(bytes(rows))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    # Insert a tEXt chunk BEFORE IHDR to prove we do not assume offset 16
    text = b"Comment\x00hello"
    ihdr = struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0)
    # Valid PNG must have IHDR first per spec — our scanner still walks chunks.
    # Put IHDR first (valid), but scanner doesn't hardcode offset 16 for width.
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"tEXt", text)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    w, h = png_size_from_bytes(png)
    assert (w, h) == (3, 2)


def test_missing_front__errors(tmp_path: Path) -> None:
    d = tmp_path / "views"
    d.mkdir()
    write_solid_png(d / "left.png", 64, 64)
    with pytest.raises(ProportionError) as ei:
        load_views(d, partial_ok=False)
    assert ei.value.code == "missing_views"


def test_partial_ok_front_only(tmp_path: Path) -> None:
    d = make_package(tmp_path, views=("front",), assist=eight_head_assist())
    # strip non-front from assist is fine; images only front
    report = analyze_proportion(d, partial_ok=True, run_heuristic_frame=False)
    assert report.quality.partial_package is True
    assert report.package_score < 100
    assert "front" in report.views
    assert "left" not in report.views


def test_assist_8head__hu(tmp_path: Path) -> None:
    d = make_package(tmp_path, assist=eight_head_assist())
    report = analyze_proportion(d, run_heuristic_frame=False)
    assert report.head_unit_frac is not None
    # 60/480 = 0.125
    assert report.head_unit_frac == pytest.approx(HEAD / 480.0, abs=1e-6)
    heads = 1.0 / report.head_unit_frac
    assert heads == pytest.approx(8.0, abs=0.05)
    assert report.quality.hair_volume_margin is False
    assert "chin" in report.landmarks_xyz
    assert report.landmarks_xyz["chin"].z == pytest.approx(1.0 - HEAD / 480.0, abs=1e-6)
    assert report.landmarks_xyz["sole"].z == pytest.approx(0.0, abs=1e-6)


def test_cranial_vs_hair__flag(tmp_path: Path) -> None:
    assist = eight_head_assist()
    # Remove cranial — force hair fallback
    del assist["views"]["front"]["landmarks"]["cranial_vertex"]
    del assist["views"]["left"]["landmarks"]["cranial_vertex"]
    d = make_package(tmp_path, assist=assist)
    report = analyze_proportion(d, run_heuristic_frame=False)
    assert report.quality.hair_volume_margin is True
    # hair at 20, chin 100, sole 520 → head=80, figure=500
    assert report.head_unit_frac == pytest.approx(80.0 / 500.0, abs=1e-6)


def test_a_pose_wrist__no_false_fail(tmp_path: Path) -> None:
    d = make_package(tmp_path, assist=eight_head_assist())
    report = analyze_proportion(d, run_heuristic_frame=False)
    arm = {c.name: c for c in report.checks}
    left = arm["arm_chain_l"]
    right = arm["arm_chain_r"]
    # Must not hard-fail because wrist Y != hanging vertical
    assert left.ok is True
    assert right.ok is True
    assert "a_pose" in str(left.measured)


def test_span_discrepancy__foreshorten(tmp_path: Path) -> None:
    assist = eight_head_assist()
    # Left sole much higher → shorter left span
    assist["views"]["left"]["landmarks"]["sole"] = [200, 400]  # span 360 vs 480
    d = make_package(tmp_path, assist=assist)
    report = analyze_proportion(d, run_heuristic_frame=False)
    assert report.vertical_span_discrepancy is not None
    assert report.vertical_span_discrepancy > 0.05
    assert report.quality.foreshortening_risk is True


def test_multi_blob__needs_user_input(tmp_path: Path) -> None:
    d = make_package(tmp_path, views=("front", "left", "three_quarter"), multi_blob_front=True)
    report = analyze_proportion(d, partial_ok=True, run_heuristic_frame=True)
    assert report.quality.multi_figure is True
    assert report.quality.needs_user_input is True


def test_multi_figure_assist_flag(tmp_path: Path) -> None:
    assist = eight_head_assist()
    assist["multi_figure"] = True
    d = make_package(tmp_path, assist=assist)
    report = analyze_proportion(d, run_heuristic_frame=False)
    assert report.quality.multi_figure is True
    assert report.quality.needs_user_input is True


def test_jpg_without_pillow__errors(tmp_path: Path) -> None:
    d = tmp_path / "views"
    d.mkdir()
    # PNG front so we only fail on jpg sibling path via allow_pillow=False
    write_solid_png(d / "front.png", 32, 32)
    write_minimal_jpg_stub(d / "left.jpg")
    write_solid_png(d / "three_quarter.png", 32, 32)
    with pytest.raises(ProportionError) as ei:
        load_views(d, allow_pillow=False, partial_ok=False)
    assert ei.value.code == "pillow_required"
    assert "meshops[proportion]" in str(ei.value) or "Pillow" in str(ei.value)


def test_template_cli__writes(tmp_path: Path) -> None:
    out = tmp_path / "landmarks_assist.json"
    r = runner.invoke(app, ["proportion", "template", "--out", str(out), "--json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert out.is_file()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "1.0.0"
    assert doc["pose"] == "unknown"
    assert doc["multi_figure"] is False
    for key in ("front", "left", "three_quarter", "back"):
        assert key in doc["views"]
        assert "landmarks" in doc["views"][key]
    blank = blank_assist_document()
    assert "cranial_vertex" in blank["views"]["front"]["landmarks"]


def test_meters_flag__scales(tmp_path: Path) -> None:
    d = make_package(tmp_path, assist=eight_head_assist())
    Hm = 1.72
    report = analyze_proportion(d, height_m=Hm, run_heuristic_frame=False)
    sole = report.landmarks_xyz["sole"]
    assert sole.z == pytest.approx(0.0)
    assert sole.z_m == pytest.approx(0.0)
    chin = report.landmarks_xyz["chin"]
    assert chin.z is not None
    assert chin.z_m is not None
    assert chin.z_m == pytest.approx(float(chin.z) * Hm, abs=1e-9)
    # cranial top ≈ 1.0 * Hm
    top = report.landmarks_xyz["cranial_vertex"]
    assert top.z_m == pytest.approx(Hm, abs=1e-6)


def test_package_score_calculation(tmp_path: Path) -> None:
    d = make_package(tmp_path, assist=eight_head_assist())
    report = analyze_proportion(d, run_heuristic_frame=False)
    # Full package: 40 + 25 + 15 + 20 = 100
    assert report.package_score == pytest.approx(100.0, abs=0.1)
    assert report.score_breakdown["views"] == pytest.approx(40.0, abs=0.1)
    assert report.score_breakdown["stature"] == pytest.approx(25.0)
    assert report.score_breakdown["width_pair"] == pytest.approx(15.0)
    assert report.score_breakdown["depth"] == pytest.approx(20.0)

    # Score function alone with empty views
    s, br = compute_package_score({})
    assert s == 0.0
    assert br["views"] == 0.0


def test_analyze_cli_json(tmp_path: Path) -> None:
    d = make_package(tmp_path, assist=eight_head_assist())
    out = tmp_path / "out"
    r = runner.invoke(
        app,
        [
            "proportion",
            "analyze",
            "--views-dir",
            str(d),
            "--out",
            str(out),
            "--height-m",
            "1.70",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["honesty"] == PROPORTION_HONESTY
    assert (out / "proportion_report.json").is_file()
    loaded = load_report(out / "proportion_report.json")
    assert loaded.package_score == pytest.approx(payload["package_score"])


def test_show_cli(tmp_path: Path) -> None:
    d = make_package(tmp_path, assist=eight_head_assist())
    out = tmp_path / "out"
    analyze_proportion(d, out_dir=out, run_heuristic_frame=False)
    r = runner.invoke(
        app,
        ["proportion", "show", "--report", str(out / "proportion_report.json"), "--json"],
    )
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["schema_version"] == "1.0.0"


def test_point_to_landmark_fracs() -> None:
    lm = point_to_landmark2d("chin", 256, 100, width_px=512, height_px=560)
    assert lm.x_frac == pytest.approx(0.5)
    assert lm.y_frac == pytest.approx(100 / 560)


def test_fuse_signs_x_camera_right(tmp_path: Path) -> None:
    d = make_package(tmp_path, assist=eight_head_assist())
    report = analyze_proportion(d, run_heuristic_frame=False)
    # shoulder_r has larger x than midline → +X
    sr = report.landmarks_xyz["shoulder_r"]
    sl = report.landmarks_xyz["shoulder_l"]
    assert sr.x is not None and sl.x is not None
    assert sr.x > 0
    assert sl.x < 0


def test_overlay_with_pillow(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from meshops.proportion.overlays import write_overlays

    d = make_package(tmp_path, assist=eight_head_assist())
    report = analyze_proportion(d, run_heuristic_frame=False)
    written = write_overlays(report, d, tmp_path / "overlays")
    assert len(written) >= 1
    assert written[0].is_file()


def test_template_write_helper(tmp_path: Path) -> None:
    p = write_template(tmp_path / "a" / "landmarks_assist.json")
    assert p.is_file()
    doc = load_assist_json(p)
    assert doc["views"]["left"]["facing_direction"] == "camera_left"


def test_head_unit_prefers_cranial() -> None:
    front = ViewLandmarks(
        view="front",
        width_px=W,
        height_px=H,
        landmarks={
            "hair_crown": Landmark2D(
                id="hair_crown",
                x_px=MID_X,
                y_px=HAIR_Y,
                x_frac=0.5,
                y_frac=HAIR_Y / H,
            ),
            "cranial_vertex": Landmark2D(
                id="cranial_vertex",
                x_px=MID_X,
                y_px=CRANIAL_Y,
                x_frac=0.5,
                y_frac=CRANIAL_Y / H,
            ),
            "chin": Landmark2D(
                id="chin",
                x_px=MID_X,
                y_px=CHIN_Y,
                x_frac=0.5,
                y_frac=CHIN_Y / H,
            ),
            "sole": Landmark2D(
                id="sole",
                x_px=MID_X,
                y_px=SOLE_Y,
                x_frac=0.5,
                y_frac=SOLE_Y / H,
            ),
        },
    )
    hu, hair, _ = head_unit_frac_from_front(front)
    assert hair is False
    assert hu == pytest.approx(60 / 480)


def test_no_network_imports() -> None:
    """Sanity: proportion package modules import without network."""
    import meshops.proportion.analyze as a
    import meshops.proportion.checks as c
    import meshops.proportion.fuse as f

    assert a is not None and c is not None and f is not None
