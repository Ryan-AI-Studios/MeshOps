"""Track 0043 — sticky blockout-feedback checklist (offline; no Blender/network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.blockout_feedback import (
    FEEDBACK_SCHEMA_VERSION,
    INCLUDED_BANDS,
    compute_soft_depth_summary,
    run_blockout_feedback,
)
from meshops.proportion.honesty import FEEDBACK_HONESTY
from meshops.proportion.models import (
    DepthBand,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

runner = CliRunner()


def _write_rgba_png(path: Path, rgba: np.ndarray) -> Path:
    from PIL import Image  # type: ignore[import-untyped,import-not-found]

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


def _blank_white(h: int = 128, w: int = 128) -> np.ndarray:
    return np.full((h, w, 4), 255, dtype=np.uint8)


def _rect_silhouette(
    h: int = 128,
    w: int = 128,
    *,
    x0: int = 40,
    y0: int = 20,
    x1: int = 88,
    y1: int = 108,
) -> np.ndarray:
    arr = _blank_white(h, w)
    arr[y0:y1, x0:x1, 0:3] = 30
    arr[y0:y1, x0:x1, 3] = 255
    return arr


def _lm(
    id_: str,
    *,
    x: float | None = 0.0,
    y: float | None = 0.05,
    z: float | None = 0.7,
    x_m: float | None = 0.0,
    y_m: float | None = 0.085,
    z_m: float | None = 1.19,
    confidence: float = 0.9,
) -> LandmarkXYZ:
    return LandmarkXYZ(
        id=id_,
        x=x,
        y=y,
        z=z,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        confidence=confidence,
    )


def _band(
    band_id: str,
    *,
    height_m: float = 1.7,
    depth_frac: float = 0.15,
    y_front: float = 0.08,
    y_back: float = -0.07,
    z_frac: float = 0.72,
) -> DepthBand:
    y_mid = (y_front + y_back) / 2.0
    return DepthBand(
        band_id=band_id,
        depth_px=30.0,
        depth_frac=depth_frac,
        depth_m=depth_frac * height_m,
        y_front=y_front,
        y_back=y_back,
        y_mid=y_mid,
        z_frac=z_frac,
        confidence=0.85,
        orientation_swapped=False,
    )


def _synthetic_report(
    *,
    bands: list[DepthBand] | None = None,
    height_m: float | None = 1.7,
) -> ProportionReport:
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        landmarks_xyz={
            "chest_front": _lm("chest_front"),
        },
        depth_bands=bands
        or [
            _band("chest", z_frac=0.72),
            _band("hip", z_frac=0.55, y_front=0.10, y_back=-0.08),
            _band("glute", z_frac=0.50, y_front=0.06, y_back=-0.12),
        ],
        quality=QualityFlags(),
    )


def _write_report(path: Path, report: ProportionReport) -> Path:
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_feedback__depth_only_skips_silhouettes(tmp_path: Path) -> None:
    """T5: report + mesh, no refs → silhouettes skipped; package ok per B5.1."""
    pytest.importorskip("PIL")
    report = _write_report(tmp_path / "report.json", _synthetic_report())
    box = trimesh.creation.box(extents=[0.3, 0.4, 1.7])
    mesh_path = tmp_path / "box.stl"
    box.export(mesh_path)
    out = tmp_path / "feedback"
    payload = run_blockout_feedback(
        report,
        out,
        mesh=mesh_path,
        force=True,
    )
    assert payload["ok"] is True
    steps = payload["steps"]
    assert steps["depth_samples"]["ok"] is True
    assert steps["depth_mesh_deltas"]["ok"] is True
    assert steps["depth_heatmap"]["ok"] is True
    assert steps["silhouette_front"]["skipped_reason"] == "no --ref-front"
    assert steps["silhouette_left"]["skipped_reason"] == "no --ref-left"
    assert steps["silhouette_front"]["ok"] is False
    assert steps["silhouette_left"]["ok"] is False
    pkg_path = Path(payload["package_path"])
    assert pkg_path.is_file()
    raw = json.loads(pkg_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == FEEDBACK_SCHEMA_VERSION
    assert raw["honesty"] == FEEDBACK_HONESTY
    assert raw["ok"] is True
    assert (out / "blockout_feedback.md").is_file()


def test_feedback__with_front_and_left_silhouettes(tmp_path: Path) -> None:
    """T6: ref-front + ref-left + mesh-views → both silhouette steps ok."""
    pytest.importorskip("PIL")
    report = _write_report(tmp_path / "report.json", _synthetic_report())
    ref_front = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    ref_left = _write_rgba_png(tmp_path / "left.png", _rect_silhouette())
    mesh_front = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    mesh_left = _write_rgba_png(tmp_path / "mesh_left.png", _rect_silhouette())
    out = tmp_path / "feedback"
    payload = run_blockout_feedback(
        report,
        out,
        ref_front=ref_front,
        ref_left=ref_left,
        mesh_view_front=mesh_front,
        mesh_view_left=mesh_left,
        force=True,
    )
    assert payload["ok"] is True
    steps = payload["steps"]
    assert steps["depth_samples"]["ok"] is True
    assert steps["depth_mesh_deltas"]["skipped_reason"] == "no --mesh"
    assert steps["silhouette_front"]["ok"] is True
    assert steps["silhouette_left"]["ok"] is True
    assert steps["silhouette_front"]["iou"] == pytest.approx(1.0, abs=1e-6)
    assert steps["silhouette_left"]["iou"] == pytest.approx(1.0, abs=1e-6)
    assert (out / "silhouette_front" / "silhouette_compare.json").is_file()
    assert (out / "silhouette_left" / "silhouette_compare.json").is_file()
    sil_left = json.loads(
        (out / "silhouette_left" / "silhouette_compare.json").read_text(encoding="utf-8")
    )
    assert sil_left["view_role"] == "left"
    assert sil_left["schema_version"] == "1.2.0"


def test_feedback__soft_depth_hip_weighted_foot_excluded() -> None:
    """T7: hip weighted; foot-family residual weight 0 so it does not dominate."""
    # Large foot residual + modest hip residual → score tracks hip, not foot.
    deltas = [
        {"id": "band_hip_front", "delta_y_m": 0.02},
        {"id": "band_hip_back", "delta_y_m": -0.02},
        {"id": "band_foot_front", "delta_y_m": 10.0},  # would dominate if weighted
        {"id": "band_toe_span", "delta_y_m": 10.0},
        {"id": "band_heel_mid", "delta_y_m": 10.0},
        {"id": "band_chest_front", "delta_y_m": 0.01},
    ]
    soft = compute_soft_depth_summary(deltas=deltas)
    assert "hip" in soft.included_bands
    assert "breast" in soft.included_bands
    for tok in ("foot", "heel", "toe", "ank", "ankle"):
        assert tok in soft.excluded_bands
    assert soft.score_m is not None
    # Foot excluded → per-band means hip=0.02, chest=0.01; equal weight → 0.015
    assert soft.score_m < 1.0
    assert soft.score_m == pytest.approx(0.015, abs=1e-6)
    assert "hip" in soft.per_band
    assert "foot" not in soft.per_band
    assert "toe" not in soft.per_band
    assert "heel" not in soft.per_band

    # With only foot-family residuals → score_m null
    foot_only = compute_soft_depth_summary(deltas=[{"id": "band_foot_front", "delta_y_m": 1.0}])
    assert foot_only.score_m is None


def test_feedback__never_calls_optimize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T8: feedback never imports/calls optimize."""
    pytest.importorskip("PIL")
    import meshops.proportion.blockout_feedback as fb

    # Source purity: orchestrator must not import/call optimize engines
    src = Path(fb.__file__).read_text(encoding="utf-8")
    assert "run_blockout_optimize" not in src
    assert "blockout_optimize" not in src
    assert "from meshops.proportion.constraints" not in src

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("optimize must not be called by blockout_feedback")

    # Guard both common entry points if somehow imported later
    import meshops.proportion.constraints as constraints_mod

    if hasattr(constraints_mod, "run_blockout_optimize"):
        monkeypatch.setattr(constraints_mod, "run_blockout_optimize", _boom)

    report = _write_report(tmp_path / "report.json", _synthetic_report())
    ref_front = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_front = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    payload = run_blockout_feedback(
        report,
        tmp_path / "out",
        ref_front=ref_front,
        mesh_view_front=mesh_front,
        force=True,
    )
    assert payload["ok"] is True
    assert payload["honesty"] == FEEDBACK_HONESTY


def test_feedback__cli_json_smoke(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    report = _write_report(tmp_path / "report.json", _synthetic_report())
    ref_front = _write_rgba_png(tmp_path / "front.png", _rect_silhouette())
    mesh_front = _write_rgba_png(tmp_path / "mesh_front.png", _rect_silhouette())
    out = tmp_path / "cli_out"
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-feedback",
            "--report",
            str(report),
            "--out",
            str(out),
            "--ref-front",
            str(ref_front),
            "--mesh-view-front",
            str(mesh_front),
            "--force",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["honesty"] == FEEDBACK_HONESTY
    assert "hip" in INCLUDED_BANDS
