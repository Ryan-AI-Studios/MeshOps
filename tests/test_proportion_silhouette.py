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
    _OTSU_MIN_SIGMA_B2,
    GRID_PX,
    SILHOUETTE_SCHEMA_VERSION,
    TRUST_REASON_CODES,
    SilhouetteComparePackage,
    _keep_large_blob_union,
    _otsu_fg_mask,
    _otsu_threshold,
    extract_silhouette_mask,
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


def test_silhouette__b1_reuses_frame_silhouette_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: primary mask must call frame.silhouette_mask (not a parallel extractor)."""
    pytest.importorskip("PIL")
    import meshops.proportion.silhouette as sil

    calls: list[int] = []
    real = sil.silhouette_mask

    def _wrap(rgba: np.ndarray) -> np.ndarray:
        calls.append(1)
        return real(rgba)

    monkeypatch.setattr(sil, "silhouette_mask", _wrap)
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert len(calls) >= 2  # ref + mesh_view


def test_silhouette__corner_median_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When primary silhouette_mask is empty, corner-median fallback is used."""
    pytest.importorskip("PIL")
    import meshops.proportion.silhouette as mod

    arr = _blank_white(64, 64)
    arr[16:48, 16:48, :3] = 40  # dark center — fallback FG via luma
    path = _write_rgba_png(tmp_path / "ref.png", arr)
    mesh_view = _write_rgba_png(tmp_path / "mesh.png", _rect_silhouette(64, 64))

    def empty_primary(rgba: np.ndarray) -> np.ndarray:
        return np.zeros(rgba.shape[:2], dtype=bool)

    monkeypatch.setattr(mod, "silhouette_mask", empty_primary)
    payload = run_silhouette_compare(path, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert payload["ok"] is True
    assert any("corner-median fallback" in m for m in payload["messages"])


def test_silhouette__render_unavailable_maps_to_silhouette_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2: RenderUnavailableError → silhouette_failed + details.code=render_unavailable."""
    pytest.importorskip("PIL")
    from meshops.render.f3d_renderer import RenderUnavailableError

    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh = tmp_path / "unit.stl"
    mesh.write_text("solid x\nendsolid x\n", encoding="utf-8")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RenderUnavailableError("F3D Engine.create(offscreen=True) failed")

    monkeypatch.setattr(
        "meshops.render.f3d_renderer.F3DRenderer.render_mesh_to_dir",
        _boom,
    )
    with pytest.raises(ProportionError) as ei:
        run_silhouette_compare(ref, tmp_path / "out", mesh=mesh, force=True)
    assert ei.value.code == "silhouette_failed"
    assert ei.value.details.get("code") == "render_unavailable"


def test_silhouette__json_out_path_is_directory_fails(tmp_path: Path) -> None:
    """C7: .json-suffixed path that is an existing directory → silhouette_failed."""
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    bad = tmp_path / "already.json"
    bad.mkdir()
    with pytest.raises(ProportionError) as ei:
        run_silhouette_compare(ref, bad, mesh_view=mesh_view, force=True)
    assert ei.value.code == "silhouette_failed"


def test_silhouette__overlay_alpha_approx_120(tmp_path: Path) -> None:
    """R6: ref/mesh overlay tints use alpha ~120."""
    pytest.importorskip("PIL")
    from meshops.proportion.silhouette import _build_overlay

    ref_g = np.zeros((GRID_PX, GRID_PX), dtype=bool)
    mesh_g = np.zeros((GRID_PX, GRID_PX), dtype=bool)
    ref_g[10:50, 10:50] = True
    mesh_g[30:70, 30:70] = True
    img = _build_overlay(ref_g, mesh_g)
    arr = np.asarray(img)
    # ref-only pixel
    assert int(arr[15, 15, 3]) == 120
    # mesh-only pixel
    assert int(arr[60, 60, 3]) == 120


def test_silhouette__mesh_render_requests_white_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5: --mesh path must pass white background_color to F3DRenderer."""
    pytest.importorskip("PIL")
    from meshops.render.f3d_renderer import RenderResult

    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh = tmp_path / "unit.stl"
    mesh.write_text("solid x\nendsolid x\n", encoding="utf-8")
    seen: dict[str, Any] = {}

    def _fake_render(
        self: Any,
        mesh_path: Any,
        views_dir: Any,
        **kwargs: Any,
    ) -> RenderResult:
        seen.update(kwargs)
        out = Path(views_dir)
        out.mkdir(parents=True, exist_ok=True)
        front = out / "front.png"
        # white bg + dark subject so mask is non-empty
        _write_rgba_png(front, _rect_silhouette())
        return RenderResult(
            mesh_id="",
            rendered_from="mesh",
            view_paths=[str(front)],
            cameras=["front"],
        )

    monkeypatch.setattr(
        "meshops.render.f3d_renderer.F3DRenderer.render_mesh_to_dir",
        _fake_render,
    )
    payload = run_silhouette_compare(ref, tmp_path / "out", mesh=mesh, force=True)
    assert payload["ok"] is True
    assert seen.get("background_color") == (1.0, 1.0, 1.0)
    assert seen.get("camera_names") == ("front",)
    assert seen.get("include_depth_for") == ()


# ---------------------------------------------------------------------------
# Track 0025 — mask trust / recovery cascade
# ---------------------------------------------------------------------------


def _studio_gray_subject(
    h: int = 128,
    w: int = 128,
    *,
    rect_w: int = 30,
    rect_h: int = 40,
    bg: tuple[int, int, int] = (128, 128, 128),
    fg: tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """C2 fixture: mid-gray studio bg + centered dark rect (30x40 on 128)."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = bg[0]
    arr[:, :, 1] = bg[1]
    arr[:, :, 2] = bg[2]
    arr[:, :, 3] = 255
    x0 = (w - rect_w) // 2
    y0 = (h - rect_h) // 2
    arr[y0 : y0 + rect_h, x0 : x0 + rect_w, 0] = fg[0]
    arr[y0 : y0 + rect_h, x0 : x0 + rect_w, 1] = fg[1]
    arr[y0 : y0 + rect_h, x0 : x0 + rect_w, 2] = fg[2]
    return arr


def test_silhouette__0025_white_subject_trusted_primary(tmp_path: Path) -> None:
    """Pure white + black subject → trusted; IoU~1; method primary."""
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    payload = run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert payload["ok"] is True
    assert payload["silhouette_trusted"] is True
    assert payload["trust_reasons"] == []
    assert payload["score_iou"] == pytest.approx(1.0, abs=1e-6)
    assert payload["mask_method_ref"] == "primary"
    assert payload["mask_method_mesh"] == "primary"
    assert 0.02 <= payload["ref_coverage_frac"] <= 0.90
    assert 0.02 <= payload["mesh_coverage_frac"] <= 0.90


def test_silhouette__0025_studio_gray_c2_recovery_trusted(tmp_path: Path) -> None:
    """C2: studio gray 128 + 30x40 dark → recovery → trusted; cov~0.07."""
    pytest.importorskip("PIL")
    gray = _studio_gray_subject()
    # Mesh side: white + same rect so score is meaningful
    mesh_arr = _rect_silhouette(
        128,
        128,
        x0=(128 - 30) // 2,
        y0=(128 - 40) // 2,
        x1=(128 - 30) // 2 + 30,
        y1=(128 - 40) // 2 + 40,
    )
    ref = _write_rgba_png(tmp_path / "front.png", gray)
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", mesh_arr)
    payload = run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert payload["ok"] is True
    assert payload["silhouette_trusted"] is True
    assert payload["trust_reasons"] == []
    # Recovery methods only (not primary full-FG)
    assert payload["mask_method_ref"] in ("corner_median", "otsu_luma")
    expected_cov = (30 * 40) / (128 * 128)
    assert payload["ref_coverage_frac"] == pytest.approx(expected_cov, abs=0.02)
    assert 0.02 <= payload["ref_coverage_frac"] <= 0.90

    raw = json.loads((tmp_path / "out" / "silhouette_compare.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == "1.1.0"
    assert raw["mask_method_ref"] == payload["mask_method_ref"]
    assert raw["silhouette_trusted"] is True


def test_silhouette__0025_flat_mid_gray_not_silent_success(tmp_path: Path) -> None:
    """Full-frame mid-gray only → empty and/or untrusted; not silent success."""
    pytest.importorskip("PIL")
    flat = np.full((64, 64, 4), 128, dtype=np.uint8)
    flat[:, :, 3] = 255
    ref = _write_rgba_png(tmp_path / "front.png", flat)
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette(64, 64))
    with pytest.raises(ProportionError) as ei:
        run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    # Empty after cascade, or untrusted hard-fail path
    assert ei.value.code in ("silhouette_empty", "silhouette_untrusted", "silhouette_failed")
    if ei.value.code == "silhouette_empty":
        assert ei.value.details.get("side") == "ref"


def test_silhouette__0025_alpha_matte_preferred(tmp_path: Path) -> None:
    """Useful alpha matte → mask_method alpha_matte; trusted if in band."""
    pytest.importorskip("PIL")
    h, w = 128, 128
    arr = np.zeros((h, w, 4), dtype=np.uint8)  # transparent bg
    arr[30:100, 40:90, 0] = 40
    arr[30:100, 40:90, 1] = 40
    arr[30:100, 40:90, 2] = 40
    arr[30:100, 40:90, 3] = 255
    ref = _write_rgba_png(tmp_path / "front.png", arr)
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", arr.copy())
    payload = run_silhouette_compare(ref, tmp_path / "out", mesh_view=mesh_view, force=True)
    assert payload["ok"] is True
    assert payload["mask_method_ref"] == "alpha_matte"
    assert payload["mask_method_mesh"] == "alpha_matte"
    assert payload["silhouette_trusted"] is True
    assert payload["score_iou"] == pytest.approx(1.0, abs=1e-6)


def test_silhouette__0025_untrusted_high_cov_package_exit0(tmp_path: Path) -> None:
    """Untrusted high coverage: package written; default exit 0."""
    pytest.importorskip("PIL")
    # ~92% dark subject leaves white corners so recovery still yields high cov
    arr = _blank_white(100, 100)
    arr[2:98, 2:98, :3] = 30
    ref = _write_rgba_png(tmp_path / "front.png", arr)
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette(100, 100))
    out = tmp_path / "out"
    payload = run_silhouette_compare(ref, out, mesh_view=mesh_view, force=True)
    assert payload["ok"] is True
    assert payload["silhouette_trusted"] is False
    assert len(payload["trust_reasons"]) > 0
    assert all(r in TRUST_REASON_CODES for r in payload["trust_reasons"])
    assert (
        "coverage_high" in payload["trust_reasons"] or "ref_untrusted" in payload["trust_reasons"]
    )
    json_path = out / "silhouette_compare.json"
    assert json_path.is_file()
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == "1.1.0"
    assert raw["silhouette_trusted"] is False

    # CLI default: exit 0 + UNTRUSTED banner
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
            str(tmp_path / "cli_out"),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "UNTRUSTED" in result.output
    assert "Do not thrash mesh geometry" in result.output


def test_silhouette__0025_require_trusted_exit1(tmp_path: Path) -> None:
    """--require-trusted + untrusted → silhouette_untrusted exit 1."""
    pytest.importorskip("PIL")
    arr = _blank_white(100, 100)
    arr[2:98, 2:98, :3] = 30
    ref = _write_rgba_png(tmp_path / "front.png", arr)
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette(100, 100))

    with pytest.raises(ProportionError) as ei:
        run_silhouette_compare(
            ref,
            tmp_path / "out",
            mesh_view=mesh_view,
            force=True,
            require_trusted=True,
        )
    assert ei.value.code == "silhouette_untrusted"
    assert "trust_reasons" in ei.value.details
    # Package still written for diagnostics
    assert (tmp_path / "out" / "silhouette_compare.json").is_file()

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
            str(tmp_path / "cli_req"),
            "--force",
            "--require-trusted",
            "--json",
        ],
    )
    assert result.exit_code == 1, result.output
    err_payload = json.loads(result.stdout)
    assert (
        err_payload.get("code") == "silhouette_untrusted" or "untrusted" in str(err_payload).lower()
    )


def test_silhouette__0025_schema_1_1_0_fields(tmp_path: Path) -> None:
    """Schema write 1.1.0 + all R1 fields; trust_reasons codes only."""
    pytest.importorskip("PIL")
    ref = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_view = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    out = tmp_path / "sil"
    payload = run_silhouette_compare(ref, out, mesh_view=mesh_view, force=True)
    assert SILHOUETTE_SCHEMA_VERSION == "1.1.0"
    raw = json.loads((out / "silhouette_compare.json").read_text(encoding="utf-8"))
    pkg = SilhouetteComparePackage.model_validate(raw)
    assert pkg.schema_version == "1.1.0"
    assert pkg.honesty == SILHOUETTE_HONESTY
    assert isinstance(pkg.silhouette_trusted, bool)
    assert isinstance(pkg.trust_reasons, list)
    assert all(r in TRUST_REASON_CODES for r in pkg.trust_reasons)
    assert pkg.mask_method_ref in (
        "primary",
        "alpha_matte",
        "corner_median",
        "otsu_luma",
        "empty",
    )
    assert pkg.mask_method_mesh in (
        "primary",
        "alpha_matte",
        "corner_median",
        "otsu_luma",
        "empty",
    )
    assert isinstance(pkg.ref_coverage_frac, float)
    assert isinstance(pkg.mesh_coverage_frac, float)
    # Success payload mirrors package trust fields
    assert "silhouette_trusted" in payload
    assert "trust_reasons" in payload
    assert "ref_coverage_frac" in payload
    assert "mesh_coverage_frac" in payload
    assert "mask_method_ref" in payload
    assert "mask_method_mesh" in payload


def test_silhouette__0025_frame_bg_threshold_unchanged() -> None:
    """frame.silhouette_mask / _BG_THRESHOLD must stay 250 (no global change)."""
    from meshops.proportion import frame as frame_mod

    assert frame_mod._BG_THRESHOLD == 250
    # near-white 250 is BG; 249 is FG
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:, :, :3] = 250
    rgba[:, :, 3] = 255
    assert not frame_mod.silhouette_mask(rgba).any()
    rgba[:, :, :3] = 249
    assert frame_mod.silhouette_mask(rgba).all()


def test_silhouette__0025_otsu_flat_unimodal_low_bimodality() -> None:
    """Unimodal hist: Otsu fails floor; extract tags otsu_low_histogram_bimodality."""
    # Unimodal spike: between-class variance collapses
    hist = np.zeros(256, dtype=np.float64)
    hist[128] = 10_000.0
    _t, sigma_b2 = _otsu_threshold(hist)
    assert sigma_b2 < _OTSU_MIN_SIGMA_B2

    # Flat mid-gray: primary full-FG → recovery → corner empty → Otsu rejects unimodal
    h, w = 64, 64
    rgba = np.full((h, w, 4), 140, dtype=np.uint8)
    rgba[:, :, 3] = 255
    with pytest.raises(ProportionError) as ei:
        extract_silhouette_mask(rgba, side="ref")
    assert ei.value.code == "silhouette_empty"
    reasons = ei.value.details.get("trust_reasons", [])
    assert "otsu_low_histogram_bimodality" in reasons


def test_silhouette__0025_multi_blob_union_keeps_both_large() -> None:
    """Recovery large-blob union keeps ALL components ≥2% area, not only largest."""
    mask = np.zeros((100, 100), dtype=bool)
    # Two equal large blobs: 30*30 = 900 px = 9% each (>= 2%)
    mask[5:35, 5:35] = True
    mask[65:95, 65:95] = True
    # One noise speck below 2% floor (10*10 = 1%)
    mask[45:55, 45:55] = True

    kept, count, msgs = _keep_large_blob_union(mask)
    assert count == 2
    assert bool(kept[10, 10])  # first large blob
    assert bool(kept[70, 70])  # second large blob
    assert not bool(kept[50, 50])  # noise speck dropped
    assert any("multi_figure" in m or "union" in m for m in msgs)


def test_silhouette__0025_otsu_light_bg_dark_fg_side() -> None:
    """B2: light BG (high median) → dark subject FG via luma <= t."""
    luma = np.full((50, 50), 200, dtype=np.uint8)
    luma[10:40, 10:40] = 40
    t = 100
    mask, msgs = _otsu_fg_mask(luma, t, bg_median=200.0, bg_std=5.0)
    assert bool(mask[25, 25])  # dark subject is FG
    assert not bool(mask[0, 0])  # light studio is BG
    assert any("luma <=" in m for m in msgs)
    # Dark BG flips side
    mask_dark_bg, msgs_dark = _otsu_fg_mask(luma, t, bg_median=40.0, bg_std=5.0)
    assert not bool(mask_dark_bg[25, 25])  # dark region treated as BG when bg is dark
    assert bool(mask_dark_bg[0, 0])
    assert any("luma >" in m for m in msgs_dark)
