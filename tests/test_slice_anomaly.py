"""Two-sided filament anomaly + watertight volume guard (DoD-6,7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from meshops.slice.anomaly import (
    DEFAULT_FAIL_HIGH,
    DEFAULT_LOW_RATIO,
    AnomalyThresholds,
    evaluate_printability,
    mesh_volume_cm3_from_mesh,
    mesh_volume_cm3_from_path,
)
from meshops.slice.models import ParsedSliceStats, SliceWarning


def _stats(
    *,
    cm3: float = 10.0,
    g: float | None = None,
    bed: bool = False,
    warnings: list[SliceWarning] | None = None,
    parse_source: str = "slice_info",
) -> ParsedSliceStats:
    used_g = g if g is not None else cm3 * 1.24
    return ParsedSliceStats(
        parse_source=parse_source,  # type: ignore[arg-type]
        plate_count=1,
        print_time_s=100.0,
        filament_used_g=used_g,
        filament_used_m=1.0 if used_g > 0 else 0.0,
        filament_used_cm3=cm3,
        bed_overflow=bed,
        warnings=warnings or [],
        warning_max_level=max((w.level for w in (warnings or [])), default=0),
        metrics={"slice.parse_source": parse_source},
    )


def test_normal_ratio_pass() -> None:
    # ratio = 10 / 10 = 1.0 → normal
    accept = evaluate_printability(_stats(cm3=10.0), mesh_volume_cm3=10.0)
    assert accept.status == "pass"
    assert accept.metrics.get("slice.filament_ratio") == pytest.approx(1.0)
    assert accept.metrics.get("slice.mesh_volume_available") is True


def test_high_anomaly_fail() -> None:
    # ratio = 90 / 10 = 9.0 >= 8.0
    accept = evaluate_printability(_stats(cm3=90.0), mesh_volume_cm3=10.0)
    assert accept.status == "fail"
    assert accept.error_code == "filament_anomaly_high"
    assert accept.metrics.get("slice.suggest_reorient") is True
    assert any("re-slice" in m.lower() or "orient" in m.lower() for m in accept.messages)


def test_low_anomaly_fail() -> None:
    # ratio = 0.01 / 10 = 0.001 <= 0.05
    accept = evaluate_printability(_stats(cm3=0.01), mesh_volume_cm3=10.0)
    assert accept.status == "fail"
    assert accept.error_code == "filament_anomaly_low"


def test_warn_band_still_pass() -> None:
    # ratio = 40 / 10 = 4.0 → warn but pass by default
    accept = evaluate_printability(_stats(cm3=40.0), mesh_volume_cm3=10.0)
    assert accept.status == "pass"
    assert accept.metrics.get("slice.suggest_reorient") is True


def test_non_watertight_skips_ratio() -> None:
    # Even absurd cm3 — no volume → skip ratio, still pass if other rules ok
    accept = evaluate_printability(_stats(cm3=9999.0), mesh_volume_cm3=None)
    assert accept.status == "pass"
    assert accept.metrics.get("slice.mesh_volume_available") is False
    assert "slice.filament_ratio" not in accept.metrics


def test_zero_filament_still_fails_without_volume() -> None:
    accept = evaluate_printability(_stats(cm3=0.0, g=0.0), mesh_volume_cm3=None)
    assert accept.status == "fail"
    assert accept.error_code == "filament_zero"


def test_mesh_volume_watertight_box() -> None:
    mesh = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    assert mesh.is_watertight
    vol = mesh_volume_cm3_from_mesh(mesh)
    assert vol is not None
    # 1000 mm³ = 1 cm³
    assert abs(vol - 1.0) < 1e-6


def test_mesh_volume_open_sheet_none() -> None:
    # Single triangle — not watertight
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    assert not mesh.is_watertight
    assert mesh_volume_cm3_from_mesh(mesh) is None


def test_mesh_volume_from_path_missing(tmp_path: Path) -> None:
    assert mesh_volume_cm3_from_path(tmp_path / "nope.stl") is None


def test_thresholds_custom() -> None:
    thr = AnomalyThresholds(fail_high=2.0, low_ratio=0.5)
    accept = evaluate_printability(_stats(cm3=25.0), mesh_volume_cm3=10.0, thresholds=thr)
    assert accept.status == "fail"
    assert accept.error_code == "filament_anomaly_high"


def test_defaults_match_spec() -> None:
    assert DEFAULT_FAIL_HIGH == 8.0
    assert DEFAULT_LOW_RATIO == 0.05
