"""Track 0017 — depth samples + optional mesh ray deltas (offline; no Blender/F3D)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.depth_samples import (
    AXIS_NOTES,
    DELTAS_BASENAME,
    SAMPLES_BASENAME,
    extract_depth_samples,
    run_depth_samples,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import DEPTH_HONESTY
from meshops.proportion.models import (
    DepthBand,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

runner = CliRunner()


def _lm(
    id_: str,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
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


def _chest_band(
    *,
    height_m: float | None = 1.7,
    depth_frac: float = 0.15,
) -> DepthBand:
    y_front = 0.08
    y_back = -0.07
    y_mid = (y_front + y_back) / 2.0
    depth_m = depth_frac * height_m if height_m is not None else None
    return DepthBand(
        band_id="chest",
        depth_px=30.0,
        depth_frac=depth_frac,
        depth_m=depth_m,
        y_front=y_front,
        y_back=y_back,
        y_mid=y_mid,
        z_frac=0.72,
        confidence=0.85,
        orientation_swapped=False,
    )


def _synthetic_report(
    *,
    landmarks_xyz: dict[str, LandmarkXYZ] | None = None,
    depth_bands: list[DepthBand] | None = None,
    height_m: float | None = 1.7,
    quality: QualityFlags | None = None,
) -> ProportionReport:
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        landmarks_xyz=landmarks_xyz or {},
        depth_bands=depth_bands or [],
        quality=quality or QualityFlags(),
    )


def _write_report(path: Path, report: ProportionReport) -> Path:
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# R8 tests
# ---------------------------------------------------------------------------


def test_depth__bands_and_fused_coexist() -> None:
    report = _synthetic_report(
        height_m=1.7,
        depth_bands=[_chest_band(height_m=1.7)],
        landmarks_xyz={
            "chest_front": _lm(
                "chest_front",
                x=0.01,
                y=0.08,
                z=0.72,
                x_m=0.017,
                y_m=0.136,
                z_m=1.224,
            ),
            "chest_back": _lm(
                "chest_back",
                x=0.01,
                y=-0.07,
                z=0.72,
                x_m=0.017,
                y_m=-0.119,
                z_m=1.224,
            ),
        },
    )
    pkg = extract_depth_samples(report)
    ids = {s.id for s in pkg.samples}
    assert "band_chest_front" in ids
    assert "band_chest_back" in ids
    assert "band_chest_mid" in ids
    assert "band_chest_span" in ids
    assert "chest_front" in ids
    assert "chest_back" in ids
    by_id = {s.id: s for s in pkg.samples}
    assert by_id["band_chest_front"].source == "depth_band"
    assert by_id["band_chest_front"].role == "band_front"
    assert by_id["band_chest_front"].view == "left"
    assert by_id["band_chest_front"].band_id == "chest"
    assert by_id["chest_front"].source == "fused_xyz"
    assert by_id["chest_front"].role == "landmark"
    assert by_id["band_chest_span"].depth_m is not None
    assert by_id["band_chest_span"].depth_frac == pytest.approx(0.15)


def test_depth__height_null_fracs_only(tmp_path: Path) -> None:
    report = _synthetic_report(
        height_m=None,
        depth_bands=[_chest_band(height_m=None)],
        landmarks_xyz={
            "chest_front": _lm(
                "chest_front",
                x=0.0,
                y=0.08,
                z=0.7,
                x_m=None,
                y_m=None,
                z_m=None,
            ),
        },
    )
    pkg = extract_depth_samples(report)
    assert any("height_m unset" in m for m in pkg.messages)
    for s in pkg.samples:
        assert s.y_m is None
        if s.role.startswith("band_"):
            assert s.y_frac is not None

    report_path = _write_report(tmp_path / "report.json", report)
    # box mesh; height null → empty deltas not error
    box = trimesh.creation.box(extents=[0.2, 0.3, 1.7])
    mesh_path = tmp_path / "box.stl"
    box.export(mesh_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    payload = run_depth_samples(report_path, out_dir, mesh=mesh_path)
    assert payload["ok"] is True
    assert payload["counts"]["deltas"] == 0
    assert any("mesh deltas empty" in m for m in payload["messages"])
    deltas_path = out_dir / DELTAS_BASENAME
    assert deltas_path.is_file()
    raw = json.loads(deltas_path.read_text(encoding="utf-8"))
    assert raw["deltas"] == []
    assert raw["honesty"] == DEPTH_HONESTY


def test_depth__empty_report__depth_empty() -> None:
    report = _synthetic_report(
        height_m=1.7,
        landmarks_xyz={
            # no y / y_m → skipped
            "sole": _lm("sole", x=0.0, z=0.0, x_m=0.0, z_m=0.0),
        },
        depth_bands=[],
    )
    with pytest.raises(ProportionError) as ei:
        extract_depth_samples(report)
    assert ei.value.code == "depth_empty"


def test_depth__mesh_box_delta_sign(tmp_path: Path) -> None:
    """Box with known Y extent; mesh_y_front > mesh_y_back; delta_depth sign."""
    # Box centered at origin: Y from -0.1 to +0.1 → depth 0.2 m
    # extents are full size along each axis
    box = trimesh.creation.box(extents=[0.3, 0.2, 1.7])
    # shift so Z soles ≈ 0 (min Z = 0)
    box.apply_translation([0.0, 0.0, 0.85])
    mesh_path = tmp_path / "box.stl"
    box.export(mesh_path)

    # ref depth thinner claim: ref_depth 0.25 > mesh 0.2 → delta_depth positive
    # landmark at center XZ with y_m at mesh mid (0)
    report = _synthetic_report(
        height_m=1.7,
        depth_bands=[
            DepthBand(
                band_id="chest",
                depth_px=40.0,
                depth_frac=0.25 / 1.7,
                depth_m=0.25,
                y_front=0.125 / 1.7,
                y_back=-0.125 / 1.7,
                y_mid=0.0,
                z_frac=0.5,
                confidence=0.9,
            )
        ],
        landmarks_xyz={
            "chest_front": _lm(
                "chest_front",
                x=0.0,
                y=0.125 / 1.7,
                z=0.5,
                x_m=0.0,
                y_m=0.125,
                z_m=0.85,
            ),
            "chest_span_proxy": _lm(
                "chest_span_proxy",
                x=0.0,
                y=0.0,
                z=0.5,
                x_m=0.0,
                y_m=0.0,
                z_m=0.85,
            ),
        },
    )
    report_path = _write_report(tmp_path / "report.json", report)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    payload = run_depth_samples(report_path, out_dir, mesh=mesh_path)
    assert payload["ok"] is True
    deltas_path = out_dir / DELTAS_BASENAME
    raw = json.loads(deltas_path.read_text(encoding="utf-8"))
    assert raw["method"] == "trimesh_ray_y"
    assert len(raw["deltas"]) >= 1

    # band_chest_span pairs via band_ → chest_span missing; use chest_front for front
    # At least chest_front should ray-hit
    by_id = {d["id"]: d for d in raw["deltas"]}
    assert "chest_front" in by_id
    d = by_id["chest_front"]
    assert d["mesh_depth_m"] is not None
    assert d["mesh_depth_m"] == pytest.approx(0.2, abs=1e-3)
    # mesh mid should be ~0 (box Y centered at 0)
    assert d["mesh_y_m"] == pytest.approx(0.0, abs=1e-3)

    # Also check band_span after pairing to chest_front? band_chest_front strips to chest_front
    if "band_chest_front" in by_id:
        bf = by_id["band_chest_front"]
        # We don't store mesh_y_front in delta; mesh_y_m is mid.
        # Verify mesh_depth positive (front > back implied)
        assert bf["mesh_depth_m"] is not None
        assert bf["mesh_depth_m"] > 0

    # span sample: pair key chest_span won't match; skip is OK
    # Build a sample that has depth_m and xz via fused landmark id band pairing on front
    # Use chest_front with a synthetic package that has span paired via mid coords:
    # For delta_depth sign: run rays on band_chest_span if we can give it xz
    # Pairing for band_chest_span → chest_span (missing). Add chest_mid with xz.
    report2 = _synthetic_report(
        height_m=1.7,
        depth_bands=[
            DepthBand(
                band_id="chest",
                depth_px=40.0,
                depth_frac=0.25 / 1.7,
                depth_m=0.25,
                y_front=0.125 / 1.7,
                y_back=-0.125 / 1.7,
                y_mid=0.0,
                z_frac=0.5,
                confidence=0.9,
            )
        ],
        landmarks_xyz={
            "chest_front": _lm(
                "chest_front",
                x=0.0,
                y=0.125 / 1.7,
                z=0.5,
                x_m=0.0,
                y_m=0.125,
                z_m=0.85,
            ),
            "chest_span": _lm(
                "chest_span",
                x=0.0,
                y=0.0,
                z=0.5,
                x_m=0.0,
                y_m=0.0,
                z_m=0.85,
            ),
        },
    )
    report2_path = _write_report(tmp_path / "report2.json", report2)
    out2 = tmp_path / "out2"
    out2.mkdir()
    payload2 = run_depth_samples(report2_path, out2, mesh=mesh_path)
    raw2 = json.loads((out2 / DELTAS_BASENAME).read_text(encoding="utf-8"))
    by_id2 = {d["id"]: d for d in raw2["deltas"]}
    assert "band_chest_span" in by_id2
    span = by_id2["band_chest_span"]
    assert span["mesh_depth_m"] == pytest.approx(0.2, abs=1e-3)
    # ref 0.25 - mesh 0.2 = +0.05 → mesh thinner than ref
    assert span["delta_depth_m"] == pytest.approx(0.05, abs=1e-3)
    assert span["delta_depth_m"] > 0
    assert payload2["ok"] is True


def test_depth__mesh_missing_file(tmp_path: Path) -> None:
    report = _synthetic_report(
        height_m=1.7,
        landmarks_xyz={
            "chest_front": _lm(
                "chest_front",
                y=0.1,
                y_m=0.17,
                x_m=0.0,
                z_m=1.0,
            ),
        },
    )
    report_path = _write_report(tmp_path / "report.json", report)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ProportionError) as ei:
        run_depth_samples(
            report_path,
            out_dir,
            mesh=tmp_path / "missing.stl",
        )
    assert ei.value.code == "mesh_load_failed"


def test_depth__out_non_json_file(tmp_path: Path) -> None:
    report = _synthetic_report(
        height_m=1.7,
        landmarks_xyz={
            "chest_front": _lm("chest_front", y=0.1, y_m=0.17),
        },
    )
    report_path = _write_report(tmp_path / "report.json", report)
    bad_out = tmp_path / "not_json.txt"
    bad_out.write_text("x", encoding="utf-8")
    with pytest.raises(ProportionError) as ei:
        run_depth_samples(report_path, bad_out)
    assert ei.value.code == "depth_failed"
    assert "must end with .json" in str(ei.value)


def test_depth__cli_json_shape(tmp_path: Path) -> None:
    report = _synthetic_report(
        height_m=1.7,
        depth_bands=[_chest_band()],
        landmarks_xyz={
            "chest_front": _lm(
                "chest_front",
                y=0.08,
                y_m=0.136,
                x_m=0.0,
                z_m=1.2,
            ),
        },
    )
    report_path = _write_report(tmp_path / "report.json", report)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = runner.invoke(
        app,
        [
            "proportion",
            "depth-samples",
            "--report",
            str(report_path),
            "--out",
            str(out_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert "paths" in payload
    assert "counts" in payload
    assert "samples" in payload["counts"]
    assert "deltas" in payload["counts"]
    assert "skipped_mesh" in payload["counts"]
    assert "messages" in payload
    assert (out_dir / SAMPLES_BASENAME).is_file()


def test_depth__honesty_token(tmp_path: Path) -> None:
    report = _synthetic_report(
        height_m=1.7,
        landmarks_xyz={
            "chest_front": _lm("chest_front", y=0.1, y_m=0.17),
        },
    )
    pkg = extract_depth_samples(report)
    assert pkg.honesty == DEPTH_HONESTY
    assert pkg.axis_notes == AXIS_NOTES
    assert DEPTH_HONESTY == "proportion_depth_samples_not_mesh_or_print_success"

    report_path = _write_report(tmp_path / "report.json", report)
    out = tmp_path / "depth_at_landmarks.json"
    run_depth_samples(report_path, out)
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["honesty"] == DEPTH_HONESTY
    assert raw["schema_version"] == "1.0.0"
    assert raw["axis_notes"] == AXIS_NOTES


def test_depth__skipped_landmarks_summary() -> None:
    report = _synthetic_report(
        height_m=1.7,
        landmarks_xyz={
            "sole": _lm("sole", x=0.0, z=0.0, x_m=0.0, z_m=0.0),  # no y
            "chin": _lm("chin", x=0.0, z=0.9, x_m=0.0, z_m=1.5),  # no y
            "chest_front": _lm("chest_front", y=0.1, y_m=0.17),
        },
    )
    pkg = extract_depth_samples(report)
    assert any("2 landmarks skipped (no depth y)" in m for m in pkg.messages)
    assert len([s for s in pkg.samples if s.role == "landmark"]) == 1


def test_depth__accepts_proportion_report_object(tmp_path: Path) -> None:
    report = _synthetic_report(
        height_m=1.7,
        landmarks_xyz={
            "chest_front": _lm("chest_front", y=0.1, y_m=0.17),
        },
    )
    out = tmp_path / "out"
    out.mkdir()
    payload = run_depth_samples(report, out)
    assert payload["ok"] is True
    assert payload["counts"]["samples"] >= 1
