"""Track 0022 — body template pack (offline; no Blender)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshops.proportion.body_template import (
    TEMPLATE_HONESTY,
    TEMPLATE_SCHEMA_VERSION,
    apply_body_template,
    list_body_templates,
    load_body_template,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import TEMPLATE_HONESTY as HONESTY_TOKEN
from meshops.proportion.models import (
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)


def _lm(
    id_: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=id_, x_m=x_m, y_m=y_m, z_m=z_m)


def _report(
    *,
    height_m: float | None = 1.72,
    head_unit_frac: float | None = 1.0 / 7.5,
    bust_hw: float = 0.16,
    hip_x: float = 0.14,
) -> ProportionReport:
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=head_unit_frac,
        landmarks_xyz={
            "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
            "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
            "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
            "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
            "hip_l": _lm("hip_l", x_m=-hip_x, y_m=0.0, z_m=0.95),
            "hip_r": _lm("hip_r", x_m=hip_x, y_m=0.0, z_m=0.95),
        },
        diameters=[
            DiameterMeasure(
                band_id="bust",
                view="front",
                width_px=40.0,
                width_eucl_px=40.0,
                theta_deg=90.0,
                width_frac=0.1,
                width_m=bust_hw * 2.0,
                half_width_m=bust_hw,
                mid_x_px=100.0,
                mid_y_px=200.0,
            )
        ],
        quality=QualityFlags(),
    )


def _write_report(tmp: Path, report: ProportionReport) -> Path:
    p = tmp / "proportion_report.json"
    p.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return p


def test_templates__list_both_ids() -> None:
    rows = list_body_templates()
    ids = {r["id"] for r in rows}
    assert ids == {"female_adult_athletic", "male_adult_athletic"}
    for r in rows:
        assert r["description"]
        assert len(r["description"]) > 8
        assert r["archetype"] == "adult_athletic"


def test_templates__honesty_exact() -> None:
    assert TEMPLATE_HONESTY == "proportion_body_template_not_mesh_or_print_success"
    assert HONESTY_TOKEN == TEMPLATE_HONESTY
    for tid in ("female_adult_athletic", "male_adult_athletic"):
        doc = load_body_template(tid)
        assert doc.honesty == TEMPLATE_HONESTY


def test_apply__female_h172(tmp_path: Path) -> None:
    report = _report(height_m=1.72)
    report_path = _write_report(tmp_path, report)
    out = tmp_path / "applied"
    payload = apply_body_template(report_path, "female_adult_athletic", out, force=True)
    assert payload["ok"] is True
    assert payload["honesty"] == TEMPLATE_HONESTY
    assert payload["schema_version"] == TEMPLATE_SCHEMA_VERSION
    assert (out / "template_applied.json").is_file()
    assert (out / "template_constants.py").is_file()
    data = json.loads((out / "template_applied.json").read_text(encoding="utf-8"))
    assert data["honesty"] == TEMPLATE_HONESTY
    assert data["schema_version"] == "1.0.0"
    assert data["template_id"] == "female_adult_athletic"
    breast_y_frac = data["constants"]["breast_y_frac"]
    assert breast_y_frac is not None
    assert breast_y_frac < 0
    assert data["constants"]["breast_mode"] == "dual_tilted"
    assert data["height_m"] == pytest.approx(1.72)
    assert data["head_unit_m"] == pytest.approx(1.72 / 7.5)
    py_text = (out / "template_constants.py").read_text(encoding="utf-8")
    assert "N6" in py_text or "not mesh" in py_text
    assert TEMPLATE_HONESTY in py_text


def test_apply__height_null_template_empty(tmp_path: Path) -> None:
    report = _report(height_m=None)
    report_path = _write_report(tmp_path, report)
    with pytest.raises(ProportionError) as ei:
        apply_body_template(report_path, "female_adult_athletic", tmp_path / "out")
    assert ei.value.code == "template_empty"


def test_apply__height_zero_template_empty(tmp_path: Path) -> None:
    report = _report(height_m=0.0)
    report_path = _write_report(tmp_path, report)
    with pytest.raises(ProportionError) as ei:
        apply_body_template(report_path, "female_adult_athletic", tmp_path / "out")
    assert ei.value.code == "template_empty"


def test_apply__male_pec_ovals_art_canon(tmp_path: Path) -> None:
    report = _report(height_m=1.80)
    report_path = _write_report(tmp_path, report)
    out = tmp_path / "male"
    payload = apply_body_template(report_path, "male_adult_athletic", out, force=True)
    assert payload["ok"] is True
    consts = payload["constants"]
    assert consts["breast_mode"] == "pec_ovals"
    assert consts["breast_mode"] != "dual_tilted"
    assert consts["glute_mode_default"] == "mild_oval"
    assert consts["torso_waist_taper"] <= 0.06
    assert consts["thigh_tilt_deg"] <= 5.0
    assert consts["neck_thickness_scale"] >= 1.0
    assert consts["shoulder_widest"] is True
    art = "male_adult_athletic v1 is art-canon prior, not measured — retune from report"
    assert any(art in m for m in payload["messages"])


def test_apply__unknown_id(tmp_path: Path) -> None:
    report = _report()
    report_path = _write_report(tmp_path, report)
    with pytest.raises(ProportionError) as ei:
        apply_body_template(report_path, "child_cartoon", tmp_path / "out")
    assert ei.value.code == "template_unknown"


def test_apply__intermammary_gap_uses_bust_hw(tmp_path: Path) -> None:
    """C6: intermammary_gap_m = gap_frac * bust_hw only."""
    bust_hw = 0.18
    report = _report(height_m=1.72, bust_hw=bust_hw)
    report_path = _write_report(tmp_path, report)
    out = tmp_path / "gap"
    payload = apply_body_template(report_path, "female_adult_athletic", out, force=True)
    gap_frac = payload["constants"]["intermammary_gap_frac"]
    gap_m = payload["constants"]["intermammary_gap_m"]
    assert gap_frac is not None
    assert gap_m is not None
    assert gap_m == pytest.approx(gap_frac * bust_hw)
    assert any("bust_hw" in n for n in payload["scale_notes"])


def test_female_seeds__neck_and_breast() -> None:
    doc = load_body_template("female_adult_athletic")
    assert doc.breast.tilt_x_deg == pytest.approx(20.0)
    assert doc.breast.y_frac == pytest.approx(-0.77)
    assert doc.breast.ry_scale == pytest.approx(1.4)
    assert doc.breast.rz_scale == pytest.approx(2.1)
    assert doc.neck_thickness_scale == pytest.approx(0.72675)
    assert doc.neck_thickness_notes.stages == [0.95, 0.9, 0.85]
    assert doc.torso_waist_taper == pytest.approx(0.14)
    assert doc.thigh_tilt_deg == pytest.approx(10.0)
    assert doc.glute.y_frac is not None
    assert doc.glute.y_frac > 0
    assert doc.breast_mode == "dual_tilted"
    assert doc.glute_mode_default == "two_spheres"


def test_apply__glute_r_prefers_measured_hip_hw(tmp_path: Path) -> None:
    """Binding scale rule: measured hip_hw wins over template r_frac * H."""
    hip_x = 0.25  # hip_hw_lm = 0.25
    report = _report(height_m=1.72, hip_x=hip_x)
    report_path = _write_report(tmp_path, report)
    out = tmp_path / "measured_glute"
    payload = apply_body_template(report_path, "female_adult_athletic", out, force=True)
    expected = 0.25 * 0.55
    assert payload["constants"]["glute_r_m"] == pytest.approx(expected)
    assert any("prefer measured" in n for n in payload["scale_notes"])
    # Pure template prior would be r_frac * 1.72 ≈ 0.1235 — must not win when hip measured
    doc = load_body_template("female_adult_athletic")
    pure_prior = float(doc.glute.r_frac) * 1.72
    assert abs(payload["constants"]["glute_r_m"] - pure_prior) > 1e-4
