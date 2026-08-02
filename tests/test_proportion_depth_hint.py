"""Track 0020 — depth-channel assist hints (offline; no torch/Blender/network)."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.assist import apply_assist, load_assist_json
from meshops.proportion.depth_hint import (
    CONF_HINT_DEFAULT,
    CONF_PROTECT,
    DEFAULT_Z_FRAC,
    MONOCULAR_UNAVAILABLE_MSG,
    DepthHintPackage,
    extract_depth_hints,
    merge_hints_into_assist,
    run_depth_hint,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import HINT_HONESTY
from meshops.proportion.load_views import ViewImage

runner = CliRunner()


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_gray_png(path: Path, arr: np.ndarray) -> None:
    """Write 8-bit grayscale PNG from 2D uint8 array (stdlib zlib)."""
    h, w = arr.shape
    raw = b""
    for y in range(h):
        raw += b"\x00" + bytes(arr[y, :].tolist())
    compressed = zlib.compress(raw, 9)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # color type 0 = gray
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_rgba_png(path: Path, rgba: np.ndarray) -> None:
    """Write 8-bit RGBA PNG from HxWx4 uint8 array."""
    h, w, _ = rgba.shape
    raw = b""
    for y in range(h):
        raw += b"\x00" + bytes(rgba[y].reshape(-1).tolist())
    compressed = zlib.compress(raw, 9)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # RGBA
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _synthetic_disparity(
    w: int = 100,
    h: int = 200,
    *,
    body_x0: int = 30,
    body_x1: int = 70,
    front_peak: int = 65,
    back_val: int = 40,
) -> np.ndarray:
    """Disparity map: larger = closer. Body strip with front peak on the right side."""
    arr = np.zeros((h, w), dtype=np.uint8)
    # background 0
    for x in range(body_x0, body_x1):
        # linear ramp: left = back (smaller), right toward front_peak = front (larger)
        t = (x - body_x0) / max(body_x1 - body_x0 - 1, 1)
        val = int(back_val + t * (front_peak - back_val))
        arr[:, x] = val
    return arr


def test_hint__external_disparity_pairs(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    depth = _synthetic_disparity()
    depth_path = tmp_path / "depth.png"
    left_path = tmp_path / "left.png"
    write_gray_png(depth_path, depth)
    # solid left (no alpha)
    write_gray_png(left_path, np.full_like(depth, 180))

    payload = run_depth_hint(
        depth_path,
        left_path,
        tmp_path / "landmarks_assist.hint.json",
        force=True,
    )
    assert payload["ok"] is True
    assert payload["counts"]["pairs"] >= 1
    assert payload["counts"]["hints"] >= 2

    raw = json.loads(Path(payload["paths"][0]).read_text(encoding="utf-8"))
    pkg = DepthHintPackage.model_validate(raw)
    assert pkg.honesty == HINT_HONESTY
    assert pkg.kind == "landmarks_assist_hint"
    assert "left" in pkg.hints
    left_hints = pkg.hints["left"]
    assert "chest_front" in left_hints
    assert "chest_back" in left_hints
    cf = left_hints["chest_front"]
    cb = left_hints["chest_back"]
    assert cf.confidence <= CONF_HINT_DEFAULT
    assert cb.confidence <= CONF_HINT_DEFAULT
    # front (argmax) should be to the right of back (argmin) in our ramp
    assert cf.x_px > cb.x_px
    assert cf.y_px == pytest.approx(cb.y_px)


def test_hint__merge_into_apply_assist(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    depth = _synthetic_disparity()
    depth_path = tmp_path / "depth.png"
    left_path = tmp_path / "left.png"
    write_gray_png(depth_path, depth)
    write_gray_png(left_path, np.full_like(depth, 180))
    merge_path = tmp_path / "landmarks_assist.json"
    hint_path = tmp_path / "landmarks_assist.hint.json"

    payload = run_depth_hint(
        depth_path,
        left_path,
        hint_path,
        merge_into=merge_path,
        force=True,
    )
    assert merge_path.is_file()
    assert str(merge_path) in payload["paths"]

    assist = load_assist_json(merge_path)
    assert "views" in assist
    assert "left" in assist["views"]
    lms = assist["views"]["left"]["landmarks"]
    assert "chest_front" in lms
    assert "x_px" in lms["chest_front"]

    views = {
        "left": ViewImage(
            view="left",
            path=left_path,
            width_px=100,
            height_px=200,
        )
    }
    result_views, _pose, _mf, _notes, _edges = apply_assist(assist, views)
    assert "left" in result_views
    assert "chest_front" in result_views["left"].landmarks
    lm = result_views["left"].landmarks["chest_front"]
    assert lm.x_px == pytest.approx(float(lms["chest_front"]["x_px"]))


def test_hint__protected_conf_skip_and_force(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    depth = _synthetic_disparity()
    depth_path = tmp_path / "depth.png"
    left_path = tmp_path / "left.png"
    write_gray_png(depth_path, depth)
    write_gray_png(left_path, np.full_like(depth, 180))

    assist_path = tmp_path / "assist.json"
    assist_doc = {
        "schema_version": "1.0.0",
        "multi_figure": False,
        "views": {
            "left": {
                "landmarks": {
                    "chest_front": {"x_px": 10.0, "y_px": 20.0, "confidence": 1.0},
                    "chest_back": {"x_px": 5.0, "y_px": 20.0, "confidence": 0.5},
                }
            }
        },
    }
    assist_path.write_text(json.dumps(assist_doc, indent=2) + "\n", encoding="utf-8")

    # Without force-hint: conf=1.0 protected skip
    merge1 = tmp_path / "merged1.json"
    p1 = run_depth_hint(
        depth_path,
        left_path,
        tmp_path / "h1.hint.json",
        assist=assist_path,
        merge_into=merge1,
        force=True,
        force_hint=False,
    )
    assert p1["counts"]["protected_skipped"] >= 1
    m1 = json.loads(merge1.read_text(encoding="utf-8"))
    assert m1["views"]["left"]["landmarks"]["chest_front"]["x_px"] == 10.0
    # low conf replaced
    assert m1["views"]["left"]["landmarks"]["chest_back"]["confidence"] <= CONF_HINT_DEFAULT

    # With force-hint: replace protected
    merge2 = tmp_path / "merged2.json"
    p2 = run_depth_hint(
        depth_path,
        left_path,
        tmp_path / "h2.hint.json",
        assist=assist_path,
        merge_into=merge2,
        force=True,
        force_hint=True,
    )
    assert p2["counts"]["protected_skipped"] == 0
    m2 = json.loads(merge2.read_text(encoding="utf-8"))
    assert m2["views"]["left"]["landmarks"]["chest_front"]["x_px"] != 10.0
    assert m2["views"]["left"]["landmarks"]["chest_front"]["confidence"] <= CONF_HINT_DEFAULT


def test_hint__default_z_frac_message(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    depth = _synthetic_disparity()
    depth_path = tmp_path / "depth.png"
    left_path = tmp_path / "left.png"
    write_gray_png(depth_path, depth)
    write_gray_png(left_path, np.full_like(depth, 180))

    hints, messages, _pairs = extract_depth_hints(depth_path, left_path, report=None)
    assert hints
    default_msgs = [m for m in messages if "DEFAULT_Z_FRAC" in m]
    assert default_msgs
    assert any("chest" in m for m in default_msgs)
    # sanity: DEFAULT_Z_FRAC has expected keys
    assert "chest" in DEFAULT_Z_FRAC


def test_hint__monocular_unavailable(tmp_path: Path) -> None:
    with pytest.raises(ProportionError) as ei:
        run_depth_hint(
            None,
            None,
            tmp_path / "out.hint.json",
            backend="monocular",
            force=True,
        )
    assert ei.value.code == "monocular_unavailable"
    msg = str(ei.value)
    assert "Depth Anything" in msg
    assert "external" in msg
    assert msg == MONOCULAR_UNAVAILABLE_MSG


def test_hint__pillow_required_for_depth_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    depth = _synthetic_disparity()
    depth_path = tmp_path / "depth.png"
    left_path = tmp_path / "left.png"
    write_gray_png(depth_path, depth)
    write_gray_png(left_path, np.full_like(depth, 180))

    import builtins

    real_import = builtins.__import__

    def _block_pil(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("No module named 'PIL'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pil)
    with pytest.raises(ProportionError) as ei:
        run_depth_hint(depth_path, left_path, tmp_path / "h.hint.json", force=True)
    assert ei.value.code == "pillow_required"
    assert "Pillow" in str(ei.value)
    assert ei.value.details.get("hint") == "uv sync --extra proportion"


def test_hint__empty_all_background(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    # all zeros → background
    depth = np.zeros((50, 50), dtype=np.uint8)
    depth_path = tmp_path / "depth.png"
    left_path = tmp_path / "left.png"
    write_gray_png(depth_path, depth)
    write_gray_png(left_path, np.full_like(depth, 100))
    with pytest.raises(ProportionError) as ei:
        run_depth_hint(depth_path, left_path, tmp_path / "h.hint.json", force=True)
    assert ei.value.code == "hint_empty"


def test_hint__merge_helpers_conf_floor() -> None:
    from meshops.proportion.depth_hint import HintPoint

    hints = {
        "chest_front": HintPoint(x_px=50.0, y_px=40.0, confidence=0.35),
        "new_pt": HintPoint(x_px=1.0, y_px=2.0, confidence=0.35),
    }
    assist = {
        "schema_version": "1.0.0",
        "views": {
            "left": {
                "landmarks": {
                    "chest_front": {"x_px": 9.0, "y_px": 9.0, "confidence": CONF_PROTECT},
                }
            }
        },
    }
    doc, skipped = merge_hints_into_assist(hints, assist=assist, force_hint=False)
    assert skipped == 1
    assert doc["views"]["left"]["landmarks"]["chest_front"]["x_px"] == 9.0
    assert "new_pt" in doc["views"]["left"]["landmarks"]


def test_hint__cli_json(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    depth = _synthetic_disparity()
    depth_path = tmp_path / "depth.png"
    left_path = tmp_path / "left.png"
    write_gray_png(depth_path, depth)
    write_gray_png(left_path, np.full_like(depth, 180))
    result = runner.invoke(
        app,
        [
            "proportion",
            "depth-hint",
            "--depth-map",
            str(depth_path),
            "--left",
            str(left_path),
            "--out",
            str(tmp_path / "out.hint.json"),
            "--force",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["counts"]["pairs"] >= 1
