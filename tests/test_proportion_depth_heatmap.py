"""Track 0020 — depth heatmap glance PNG + meta (offline; no Blender/network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.depth_heatmap import (
    DEFAULT_PANEL_H,
    FOOTER_H,
    PANEL_GAP_PX,
    DepthHeatmapPackage,
    _panel_xy,
    _usable_sample_points,
    run_depth_heatmap,
)
from meshops.proportion.depth_samples import (
    DepthDelta,
    DepthDeltasPackage,
    DepthSample,
    DepthSamplesPackage,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import HEATMAP_HONESTY

runner = CliRunner()


def _sample(
    id_: str,
    *,
    role: str = "landmark",
    y_m: float | None = 0.1,
    z_frac: float | None = 0.72,
    depth_m: float | None = 0.2,
    y_frac: float | None = None,
    z_m: float | None = None,
) -> DepthSample:
    return DepthSample(
        id=id_,
        role=role,  # type: ignore[arg-type]
        y_m=y_m,
        y_frac=y_frac,
        z_frac=z_frac,
        z_m=z_m,
        depth_m=depth_m,
        source="fused_xyz" if role == "landmark" else "depth_band",
        confidence=0.8,
    )


def _write_samples(
    path: Path,
    samples: list[DepthSample],
    *,
    height_m: float | None = 1.7,
) -> Path:
    pkg = DepthSamplesPackage(
        height_m=height_m,
        samples=samples,
        counts={"samples": len(samples)},
    )
    path.write_text(json.dumps(pkg.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return path


def _write_deltas(path: Path, deltas: list[DepthDelta]) -> Path:
    pkg = DepthDeltasPackage(
        mesh_path="blockout.stl",
        deltas=deltas,
        counts={"deltas": len(deltas), "skipped": 0},
    )
    path.write_text(json.dumps(pkg.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return path


def test_heatmap__synthetic_png_and_meta(tmp_path: Path) -> None:
    samples_path = _write_samples(
        tmp_path / "depth_at_landmarks.json",
        [
            _sample("chest_front", y_m=0.12, z_frac=0.72),
            _sample("chest_back", y_m=-0.08, z_frac=0.72, role="landmark"),
            _sample(
                "band_chest_front",
                role="band_front",
                y_m=0.11,
                z_frac=0.72,
            ),
        ],
    )
    out_dir = tmp_path / "vis"
    out_dir.mkdir()
    payload = run_depth_heatmap(samples_path, str(out_dir) + "\\", force=True)
    assert payload["ok"] is True
    assert payload["counts"]["samples_plotted"] >= 1
    paths = [Path(p) for p in payload["paths"]]
    png = next(p for p in paths if p.suffix.lower() == ".png")
    meta = next(p for p in paths if p.suffix.lower() == ".json")
    assert png.is_file()
    assert meta.is_file()
    raw = json.loads(meta.read_text(encoding="utf-8"))
    pkg = DepthHeatmapPackage.model_validate(raw)
    assert pkg.honesty == HEATMAP_HONESTY
    assert len(pkg.plotted_sample_ids) >= 1
    assert pkg.counts["samples_plotted"] >= 1


def test_heatmap__pillow_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples_path = _write_samples(
        tmp_path / "depth_at_landmarks.json",
        [_sample("chest_front")],
    )
    import builtins

    real_import = builtins.__import__

    def _block_pil(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("No module named 'PIL'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pil)
    with pytest.raises(ProportionError) as ei:
        run_depth_heatmap(samples_path, tmp_path / "out.png", force=True)
    assert ei.value.code == "pillow_required"
    assert "Pillow" in str(ei.value)
    assert "meshops[proportion]" in str(ei.value)
    assert ei.value.details.get("hint") == "uv sync --extra proportion"


def test_heatmap__deltas_dual_panel(tmp_path: Path) -> None:
    samples_path = _write_samples(
        tmp_path / "depth_at_landmarks.json",
        [
            _sample("chest_front", y_m=0.12, z_frac=0.72),
            _sample("hip_front", y_m=0.10, z_frac=0.53),
        ],
    )
    deltas_path = _write_deltas(
        tmp_path / "depth_mesh_deltas.json",
        [
            DepthDelta(
                id="chest_front",
                ref_y_m=0.12,
                mesh_y_m=0.08,
                delta_y_m=0.04,
                ref_depth_m=0.2,
                mesh_depth_m=0.15,
                delta_depth_m=0.05,
            ),
            DepthDelta(
                id="hip_front",
                ref_y_m=0.10,
                mesh_y_m=0.07,
                delta_y_m=0.03,
                delta_depth_m=0.02,
            ),
        ],
    )
    out_png = tmp_path / "hm.png"
    payload = run_depth_heatmap(samples_path, out_png, deltas=deltas_path, force=True)
    assert payload["counts"]["deltas_plotted"] >= 1
    meta = json.loads((tmp_path / "depth_heatmap.json").read_text(encoding="utf-8"))
    panels = {c["panel"] for c in meta["color_scales"]}
    assert "samples" in panels
    assert "deltas" in panels
    # F3: canvas_h = 2 * panel_h + 12 + footer
    from PIL import Image

    with Image.open(out_png) as im:
        w, h = im.size
    expected_h = 2 * DEFAULT_PANEL_H + PANEL_GAP_PX + FOOTER_H
    assert h == expected_h
    assert w > 0


def test_heatmap__band_and_landmark_same_z_vertical() -> None:
    """B2 defensive: same body-up z_frac → same vertical plot position."""
    samples = [
        _sample("chest_front", role="landmark", y_m=0.12, z_frac=0.72),
        _sample("band_chest_front", role="band_front", y_m=0.11, z_frac=0.72),
    ]
    points = _usable_sample_points(samples, height_m=1.7)
    assert len(points) == 2
    # Same z → same py via _panel_xy vertical component
    _, py0 = _panel_xy(
        points[0]["z_frac"],
        points[0]["y_val"],
        y_min=-1.0,
        y_max=1.0,
        plot_x0=0,
        plot_y0=0,
        plot_w=100,
        plot_h=200,
    )
    _, py1 = _panel_xy(
        points[1]["z_frac"],
        points[1]["y_val"],
        y_min=-1.0,
        y_max=1.0,
        plot_x0=0,
        plot_y0=0,
        plot_w=100,
        plot_h=200,
    )
    assert py0 == py1


def test_heatmap__empty_samples(tmp_path: Path) -> None:
    samples_path = _write_samples(
        tmp_path / "depth_at_landmarks.json",
        [
            # no z / y → unusable
            _sample("bad", y_m=None, z_frac=None, depth_m=None, y_frac=None),
        ],
    )
    with pytest.raises(ProportionError) as ei:
        run_depth_heatmap(samples_path, tmp_path / "out.png", force=True)
    assert ei.value.code == "heatmap_empty"


def test_heatmap__front_right(tmp_path: Path) -> None:
    """D3: larger y_m plots further right."""
    front = _sample("front", y_m=0.20, z_frac=0.5)
    back = _sample("back", y_m=-0.10, z_frac=0.5)
    points = _usable_sample_points([front, back], height_m=1.7)
    y_vals = [p["y_val"] for p in points]
    y_min, y_max = min(y_vals), max(y_vals)
    px_front, _ = _panel_xy(
        0.5, 0.20, y_min=y_min, y_max=y_max, plot_x0=0, plot_y0=0, plot_w=100, plot_h=100
    )
    px_back, _ = _panel_xy(
        0.5, -0.10, y_min=y_min, y_max=y_max, plot_x0=0, plot_y0=0, plot_w=100, plot_h=100
    )
    assert px_front > px_back


def test_heatmap__invalid_samples(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "valid"}\n', encoding="utf-8")
    with pytest.raises(ProportionError) as ei:
        run_depth_heatmap(bad, tmp_path / "out.png", force=True)
    assert ei.value.code == "invalid_depth_samples"


def test_heatmap__cli_json(tmp_path: Path) -> None:
    samples_path = _write_samples(
        tmp_path / "depth_at_landmarks.json",
        [_sample("chest_front")],
    )
    out = tmp_path / "vis"
    out.mkdir()
    result = runner.invoke(
        app,
        [
            "proportion",
            "depth-heatmap",
            "--samples",
            str(samples_path),
            "--out",
            str(out) + "\\",
            "--force",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["counts"]["samples_plotted"] >= 1
