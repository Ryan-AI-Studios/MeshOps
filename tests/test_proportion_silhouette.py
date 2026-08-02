"""Track 0021 — front silhouette compare (offline; no Blender/network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import SILHOUETTE_HONESTY
from meshops.proportion.silhouette import (
    GRID_PX,
    SilhouetteComparePackage,
    run_silhouette_compare,
)

runner = CliRunner()


def _write_rgba_png(path: Path, rgba: np.ndarray) -> Path:
    """Write HxWx4 uint8 PNG via Pillow."""
    from PIL import Image  # type: ignore[import-untyped,import-not-found]

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


def _blank_white(h: int = 128, w: int = 128) -> np.ndarray:
    arr = np.full((h, w, 4), 255, dtype=np.uint8)
    return arr


def _rect_silhouette(
    h: int = 128,
    w: int = 128,
    *,
    x0: int = 40,
    y0: int = 20,
    x1: int = 88,
    y1: int = 108,
    fill: tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """White background with a solid rectangle subject."""
    arr = _blank_white(h, w)
    arr[y0:y1, x0:x1, 0] = fill[0]
    arr[y0:y1, x0:x1, 1] = fill[1]
    arr[y0:y1, x0:x1, 2] = fill[2]
    arr[y0:y1, x0:x1, 3] = 255
    return arr


def test_silhouette__identical_iou_near_one(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    out = tmp_path / "out"
    payload = run_silhouette_compare(ref, out, mesh_view=mesh_view, force=True)
    assert payload["ok"] is True
    assert payload["score_iou"] == pytest.approx(1.0, abs=1e-6)
    assert payload["score_dice"] == pytest.approx(1.0, abs=1e-6)
    assert payload["counts"]["ref_fg_grid_px"] > 0


def test_silhouette__disjoint_iou_near_zero(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    # Filled solid vs hollow frame → low IoU after content-bbox normalize.
    solid = _rect_silhouette(x0=40, y0=40, x1=88, y1=88)
    hollow = _blank_white(128, 128)
    hollow[20:100, 20:24, :3] = 20
    hollow[20:100, 96:100, :3] = 20
    hollow[20:24, 20:100, :3] = 20
    hollow[96:100, 20:100, :3] = 20
    ref = _write_rgba_png(tmp_path / "front.png", solid)
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", hollow)
    payload = run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert payload["score_iou"] < 0.35


def test_silhouette__partial_overlap(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    # Tall thin vs wide short → partial overlap after content-bbox resize.
    tall = _rect_silhouette(x0=55, y0=10, x1=75, y1=118)
    wide = _rect_silhouette(x0=10, y0=55, x1=118, y1=75)
    ref = _write_rgba_png(tmp_path / "front.png", tall)
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", wide)
    payload = run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert 0.0 < payload["score_iou"] < 1.0
    assert 0.0 < payload["score_dice"] < 1.0


def test_silhouette__empty_ref(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _blank_white())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    with pytest.raises(ProportionError) as ei:
        run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert ei.value.code == "silhouette_empty"
    assert ei.value.details.get("side") == "ref"


def test_silhouette__pillow_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Write PNGs first with real Pillow
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())

    import builtins

    real_import = builtins.__import__

    def _block_pil(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("No module named 'PIL'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pil)
    with pytest.raises(ProportionError) as ei:
        run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert ei.value.code == "pillow_required"
    assert "Pillow" in str(ei.value)
    assert "meshops[proportion]" in str(ei.value)
    assert ei.value.details.get("hint") == "uv sync --extra proportion"


def test_silhouette__both_mesh_and_mesh_view_fail(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    mesh = tmp_path / "dummy.stl"
    mesh.write_bytes(b"solid x\nendsolid x\n")
    with pytest.raises(ProportionError) as ei:
        run_silhouette_compare(
            ref,
            tmp_path / "out",
            mesh=mesh,
            mesh_view=mesh_view,
            force=True,
        )
    assert ei.value.code == "silhouette_failed"
    assert "only one" in str(ei.value).lower()


def test_silhouette__view_role_front_only(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    for bad in ("left", "Left", "side", "back"):
        with pytest.raises(ProportionError) as ei:
            run_silhouette_compare(
                ref,
                tmp_path / f"out_{bad}",
                mesh_view=mesh_view,
                view_role=bad,
                force=True,
            )
        assert ei.value.code == "silhouette_failed"
        assert "front" in str(ei.value).lower()

    for ok_role in ("front", "Front", "FRONT"):
        payload = run_silhouette_compare(
            ref,
            tmp_path / f"out_{ok_role}",
            mesh_view=mesh_view,
            view_role=ok_role,
            force=True,
        )
        assert payload["ok"] is True


def test_silhouette__basename_left_advisory(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "left.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    payload = run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert payload["ok"] is True
    assert payload["score_iou"] == pytest.approx(1.0, abs=1e-6)
    assert any("non-front" in m for m in payload["messages"])
    assert any("Advisory" in m for m in payload["messages"])


def test_silhouette__identical_path_trivial_message(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    shared = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    payload = run_silhouette_compare(
        shared,
        tmp_path / "out",
        mesh_view=shared,
        force=True,
    )
    assert payload["ok"] is True
    assert payload["score_iou"] == pytest.approx(1.0, abs=1e-6)
    assert any("identical" in m and "trivially" in m for m in payload["messages"])


def test_silhouette__multi_figure_message(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    # Two large blobs (>= 2% of 200x200 = 800 px each)
    arr = _blank_white(200, 200)
    arr[10:80, 10:80, :3] = 25  # 70*70 = 4900
    arr[120:190, 120:190, :3] = 25
    ref = _write_rgba_png(tmp_path / "front.png", arr)
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette(200, 200))
    payload = run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert payload["ok"] is True
    assert any("multiple figures" in m for m in payload["messages"])


def test_silhouette__one_px_shift_iou(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    # Same rect shifted by 1 px in the frame — content-bbox should keep IoU high
    a = _rect_silhouette(x0=40, y0=20, x1=88, y1=108)
    b = _rect_silhouette(x0=41, y0=21, x1=89, y1=109)
    ref = _write_rgba_png(tmp_path / "front.png", a)
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", b)
    payload = run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert payload["score_iou"] >= 0.95


def test_silhouette__thirty_px_frame_shift_iou(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    # Placement offset only — content-bbox hides pose (F4). Keep both fully in frame.
    a = _rect_silhouette(160, 160, x0=20, y0=20, x1=60, y1=100)
    b = _rect_silhouette(160, 160, x0=50, y0=50, x1=90, y1=130)
    ref = _write_rgba_png(tmp_path / "front.png", a)
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", b)
    payload = run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert payload["score_iou"] == pytest.approx(1.0, abs=0.02)


def test_silhouette__overlay_and_honesty_json(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    out_dir = tmp_path / "sil"
    payload = run_silhouette_compare(
        ref,
        out_dir,
        mesh_view=mesh_view,
        overlay=True,
        force=True,
    )
    assert payload["ok"] is True
    paths = [Path(p) for p in payload["paths"]]
    json_path = next(p for p in paths if p.suffix.lower() == ".json")
    overlay = next(p for p in paths if p.suffix.lower() == ".png")
    assert json_path.is_file()
    assert overlay.is_file()
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    pkg = SilhouetteComparePackage.model_validate(raw)
    assert pkg.honesty == SILHOUETTE_HONESTY
    assert pkg.honesty == "proportion_silhouette_compare_not_mesh_or_print_success"
    assert pkg.view_role == "front"
    assert pkg.grid_px == GRID_PX
    assert pkg.alignment.rotation is False
    assert pkg.alignment.resize == "nearest_256"
    assert pkg.overlay_path is not None


def test_silhouette__no_overlay(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    payload = run_silhouette_compare(
        ref,
        tmp_path / "out.json",
        mesh_view=mesh_view,
        overlay=False,
        force=True,
    )
    assert payload["ok"] is True
    assert all(not p.lower().endswith(".png") for p in payload["paths"])
    raw = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert raw["overlay_path"] is None


def test_silhouette__neither_mesh_nor_mesh_view(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    with pytest.raises(ProportionError) as ei:
        run_silhouette_compare(ref, tmp_path / "out", force=True)
    assert ei.value.code == "silhouette_failed"


def test_silhouette__cli_json_smoke(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    out = tmp_path / "sil_out"
    result = runner.invoke(
        app,
        [
            "proportion",
            "silhouette-compare",
            "--ref",
            str(ref),
            "--mesh-view",
            str(mesh_view),
            "--out",
            str(out),
            "--force",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert "score_iou" in payload
    assert "score_dice" in payload
