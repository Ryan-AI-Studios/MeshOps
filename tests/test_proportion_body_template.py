"""Track 0022 - body template pack (offline; no Blender).

Track 0031 - soft Y meters: breast/glute y_frac * soft half-depth (not stature).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from meshops.proportion.body_template import (
    ABS_SOFT_Y_CLAMP_M,
    HIP_SOFT_HALF_FALLBACK_FRAC,
    TEMPLATE_HONESTY,
    TEMPLATE_SCHEMA_VERSION,
    _clamp_soft_y_m,
    _resolve_applied_constants,
    _soft_y_from_frac,
    apply_body_template,
    list_body_templates,
    load_body_template,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import TEMPLATE_HONESTY as HONESTY_TOKEN
from meshops.proportion.models import (
    DepthBand,
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


def _depth_band(
    band_id: str,
    *,
    depth_m: float = 0.26,
    z_frac: float = 0.72,
) -> DepthBand:
    return DepthBand(
        band_id=band_id,
        depth_px=50.0,
        depth_frac=0.12,
        depth_m=depth_m,
        y_front=0.1,
        y_back=-0.1,
        y_mid=0.0,
        z_frac=z_frac,
    )


def _report(
    *,
    height_m: float | None = 1.72,
    head_unit_frac: float | None = 1.0 / 7.5,
    bust_hw: float = 0.16,
    hip_x: float = 0.14,
    depth_bands: list[DepthBand] | None = None,
    extra_landmarks: dict[str, LandmarkXYZ] | None = None,
) -> ProportionReport:
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "hip_l": _lm("hip_l", x_m=-hip_x, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=hip_x, y_m=0.0, z_m=0.95),
    }
    if extra_landmarks:
        lms.update(extra_landmarks)
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=head_unit_frac,
        landmarks_xyz=lms,
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
        depth_bands=list(depth_bands) if depth_bands is not None else [],
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
    # F6: soft-depth resolve keeps |breast_y_m| well under stature product (~1.32)
    breast_y_m = data["constants"]["breast_y_m"]
    assert breast_y_m is not None
    assert abs(breast_y_m) < 0.30
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
    assert doc.torso_waist_taper == pytest.approx(0.22)
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


# ---------------------------------------------------------------------------
# 0031 — soft Y meters (half soft-depth, not stature)
# ---------------------------------------------------------------------------


def test_soft_y_from_frac__zero_is_positive_zero() -> None:
    """B9 / F7: frac==0.0 → +0.0 (IEEE positive zero, not -0.0)."""
    y = _soft_y_from_frac(0.0, 0.2)
    assert y == 0.0
    assert not (y < 0.0)  # reject -0.0
    # copysign(+1, y) is +1 only for +0.0; for -0.0 it is -1
    assert math.copysign(1.0, y) > 0.0


def test_clamp_soft_y_m__soft_depth_and_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1 / B5: |y| ≤ soft_depth and ≤ ABS_SOFT_Y_CLAMP_M; notes/messages; apply ok."""
    notes: list[str] = []
    msgs: list[str] = []
    # soft_depth limit wins when smaller than abs clamp
    out = _clamp_soft_y_m(1.0, 0.1, label="breast_y_m", scale_notes=notes, messages=msgs)
    assert out == pytest.approx(0.1)
    assert any("clamped" in n and "breast_y_m" in n for n in notes)
    assert any("clamped" in m for m in msgs)

    notes2: list[str] = []
    msgs2: list[str] = []
    # absolute safety when soft_depth is huge
    out_abs = _clamp_soft_y_m(1.0, 1.0, label="glute_y_m", scale_notes=notes2, messages=msgs2)
    assert out_abs == pytest.approx(ABS_SOFT_Y_CLAMP_M)
    assert ABS_SOFT_Y_CLAMP_M == 0.35
    assert any("clamped" in n and "glute_y_m" in n for n in notes2)

    # Within limits: pass-through
    notes3: list[str] = []
    msgs3: list[str] = []
    assert _clamp_soft_y_m(
        -0.05, 0.2, label="breast_y_m", scale_notes=notes3, messages=msgs3
    ) == pytest.approx(-0.05)
    assert notes3 == []
    assert msgs3 == []

    # Apply path: high |y_frac| forces clamp; payload ok stays True (B5)
    doc = load_body_template("female_adult_athletic")
    high_frac_doc = doc.model_copy(
        update={"breast": doc.breast.model_copy(update={"y_frac": -5.0})}
    )
    real_load = load_body_template

    def _load_high_frac(tid: str):
        if tid == "female_adult_athletic":
            return high_frac_doc
        return real_load(tid)

    monkeypatch.setattr("meshops.proportion.body_template.load_body_template", _load_high_frac)
    report = _report(height_m=1.72)
    report_path = _write_report(tmp_path, report)
    payload = apply_body_template(
        report_path, "female_adult_athletic", tmp_path / "clamp_ok", force=True
    )

    assert payload["ok"] is True
    breast_y = payload["constants"]["breast_y_m"]
    soft_fallback = 0.12 * 1.72
    assert breast_y is not None
    assert abs(breast_y) == pytest.approx(soft_fallback, abs=1e-6)
    assert any("clamped" in n for n in payload["scale_notes"])
    assert any("clamped" in m for m in payload["messages"])


def test_apply__breast_y_not_stature_product(tmp_path: Path) -> None:
    """Pin bug: female H=1.72 no depth -> |breast_y_m| < 0.30, typically ~-0.159."""
    h = 1.72
    report = _report(height_m=h)
    report_path = _write_report(tmp_path, report)
    payload = apply_body_template(
        report_path, "female_adult_athletic", tmp_path / "soft_y", force=True
    )
    breast_y = payload["constants"]["breast_y_m"]
    breast_y_frac = payload["constants"]["breast_y_frac"]
    assert breast_y is not None
    assert breast_y_frac == pytest.approx(-0.77)
    assert abs(breast_y) < 0.30
    # Forbidden pre-0031: y_frac * height ~-1.32
    stature_product = -0.77 * h
    assert abs(breast_y - stature_product) > 0.5
    # Fallback: 0.12 * H soft half-depth
    soft_depth = 0.12 * h
    expected = -0.77 * soft_depth
    assert breast_y == pytest.approx(expected, abs=1e-4)
    assert breast_y == pytest.approx(-0.159, abs=0.01)
    notes = " ".join(payload["scale_notes"])
    assert "soft_depth" in notes
    assert "source=fallback" in notes
    assert any(
        "breast_y_m=" in n and "y_frac=" in n and "soft_depth_m=" in n
        for n in payload["scale_notes"]
    )


def test_apply__breast_y_from_chest_depth_band(tmp_path: Path) -> None:
    """Depth band chest depth_m=0.26 -> soft_depth=0.13; breast_y ~-0.77*0.13."""
    report = _report(height_m=1.72, depth_bands=[_depth_band("chest", depth_m=0.26)])
    report_path = _write_report(tmp_path, report)
    payload = apply_body_template(
        report_path, "female_adult_athletic", tmp_path / "band", force=True
    )
    breast_y = payload["constants"]["breast_y_m"]
    assert breast_y == pytest.approx(-0.77 * 0.13, abs=1e-4)
    assert any("source=band" in n for n in payload["scale_notes"])
    assert any("soft_depth_m=" in n for n in payload["scale_notes"])


def test_apply__breast_y_from_breast_depth_band(tmp_path: Path) -> None:
    """F5: band_id=breast is accepted on chest soft-depth ladder (same as chest)."""
    report = _report(height_m=1.72, depth_bands=[_depth_band("breast", depth_m=0.26)])
    report_path = _write_report(tmp_path, report)
    payload = apply_body_template(
        report_path, "female_adult_athletic", tmp_path / "breast_band", force=True
    )
    breast_y = payload["constants"]["breast_y_m"]
    assert breast_y == pytest.approx(-0.77 * 0.13, abs=1e-4)
    assert any("source=band" in n and "breast_y_m=" in n for n in payload["scale_notes"])


def test_apply__breast_y_from_measured_chest_front_back(tmp_path: Path) -> None:
    """Rung1: chest_front/back y_m → source=measured; soft_depth = abs(diff)/2."""
    # front y=-0.10, back y=+0.16 → half depth = 0.13
    extra = {
        "chest_front": _lm("chest_front", y_m=-0.10, z_m=1.25),
        "chest_back": _lm("chest_back", y_m=0.16, z_m=1.25),
    }
    report = _report(height_m=1.72, extra_landmarks=extra)
    report_path = _write_report(tmp_path, report)
    payload = apply_body_template(
        report_path, "female_adult_athletic", tmp_path / "measured", force=True
    )
    breast_y = payload["constants"]["breast_y_m"]
    assert breast_y == pytest.approx(-0.77 * 0.13, abs=1e-4)
    assert any("source=measured" in n for n in payload["scale_notes"])


def test_apply__glute_y_from_measured_hip_front_back(tmp_path: Path) -> None:
    """F2: hip_front/back y_m → glute source=measured; y ≈ 0.224 * half_depth."""
    # front y=-0.10, back y=+0.16 → half depth = 0.13
    extra = {
        "hip_front": _lm("hip_front", y_m=-0.10, z_m=0.95),
        "hip_back": _lm("hip_back", y_m=0.16, z_m=0.95),
    }
    report = _report(height_m=1.72, extra_landmarks=extra)
    report_path = _write_report(tmp_path, report)
    payload = apply_body_template(
        report_path, "female_adult_athletic", tmp_path / "hip_meas", force=True
    )
    glute_y = payload["constants"]["glute_y_m"]
    glute_frac = payload["constants"]["glute_y_frac"]
    assert glute_frac == pytest.approx(0.224, abs=0.01)
    assert glute_y == pytest.approx(0.224 * 0.13, abs=1e-4)
    assert any("glute_y_m=" in n and "source=measured" in n for n in payload["scale_notes"])


def test_apply__glute_y_from_hip_depth_band(tmp_path: Path) -> None:
    """F2: depth_band hip depth_m → glute source=band; y ≈ 0.224 * (depth_m/2)."""
    report = _report(height_m=1.72, depth_bands=[_depth_band("hip", depth_m=0.26)])
    report_path = _write_report(tmp_path, report)
    payload = apply_body_template(
        report_path, "female_adult_athletic", tmp_path / "hip_band", force=True
    )
    glute_y = payload["constants"]["glute_y_m"]
    assert glute_y == pytest.approx(0.224 * 0.13, abs=1e-4)
    assert any("glute_y_m=" in n and "source=band" in n for n in payload["scale_notes"])


def test_apply__glute_band_id_ignored_for_soft_depth(tmp_path: Path) -> None:
    """F4: only band_id=glute (no hip) does not fuse — still hip fallback 0.13*H."""
    h = 1.72
    # Large glute band depth so a mistaken band path would diverge clearly from fallback
    report = _report(height_m=h, depth_bands=[_depth_band("glute", depth_m=1.0)])
    report_path = _write_report(tmp_path, report)
    payload = apply_body_template(
        report_path, "female_adult_athletic", tmp_path / "glute_band_only", force=True
    )
    glute_y = payload["constants"]["glute_y_m"]
    hip_soft = HIP_SOFT_HALF_FALLBACK_FRAC * h
    assert glute_y == pytest.approx(0.224 * hip_soft, abs=1e-4)
    # Must not use glute band half-depth 0.50
    assert abs(float(glute_y) - 0.224 * 0.50) > 0.05
    assert any("glute_y_m=" in n and "source=fallback" in n for n in payload["scale_notes"])


def test_resolve__glute_absolute_y_m_passthrough() -> None:
    """F3 / B4: absolute glute.y_m pass-through then clamp; frac stays raw."""
    doc = load_body_template("female_adult_athletic")
    doc = doc.model_copy(
        update={
            "glute": doc.glute.model_copy(update={"y_m": 0.05, "y_frac": 0.224}),
        }
    )
    report = _report(height_m=1.72)
    notes: list[str] = []
    msgs: list[str] = []
    consts = _resolve_applied_constants(doc, report, 1.72, scale_notes=notes, messages=msgs)
    assert consts.glute_y_m == pytest.approx(0.05)
    assert consts.glute_y_frac == pytest.approx(0.224)
    assert any("absolute template y_m" in n for n in notes)
    # Still clamped if absolute y_m exceeds soft_depth
    doc2 = doc.model_copy(
        update={"glute": doc.glute.model_copy(update={"y_m": 1.0, "y_frac": 0.224})}
    )
    notes2: list[str] = []
    msgs2: list[str] = []
    consts2 = _resolve_applied_constants(doc2, report, 1.72, scale_notes=notes2, messages=msgs2)
    hip_soft = HIP_SOFT_HALF_FALLBACK_FRAC * 1.72
    assert consts2.glute_y_m == pytest.approx(hip_soft, abs=1e-6)
    assert any("clamped" in n for n in notes2)
    assert any("clamped" in m for m in msgs2)


def test_apply__glute_y_retuned_fallback(tmp_path: Path) -> None:
    """Glute y_frac retune: ≈0.05 F / ≈0.043 M at H=1.72 hip fallback; not ~0.006."""
    h = 1.72
    hip_soft = 0.13 * h
    report = _report(height_m=h)
    report_path = _write_report(tmp_path, report)

    f_payload = apply_body_template(
        report_path, "female_adult_athletic", tmp_path / "gf", force=True
    )
    f_y = f_payload["constants"]["glute_y_m"]
    f_frac = f_payload["constants"]["glute_y_frac"]
    assert f_y is not None and f_y > 0
    assert f_y == pytest.approx(0.05, abs=0.005)
    assert abs(f_y - 0.006) > 0.02
    # Raw template prior, not y_m/h
    doc_f = load_body_template("female_adult_athletic")
    assert doc_f.glute.y_frac is not None
    assert f_frac == pytest.approx(float(doc_f.glute.y_frac))
    assert f_frac == pytest.approx(0.224, abs=0.01)
    assert abs(float(f_frac) - (f_y / h)) > 0.05  # must not be stature hybrid
    assert any("glute_y_m=" in n and "soft_depth_m=" in n for n in f_payload["scale_notes"])
    # Sanity: y_m ≈ frac * hip soft half
    assert f_y == pytest.approx(float(f_frac) * hip_soft, abs=1e-4)

    m_payload = apply_body_template(report_path, "male_adult_athletic", tmp_path / "gm", force=True)
    m_y = m_payload["constants"]["glute_y_m"]
    m_frac = m_payload["constants"]["glute_y_frac"]
    assert m_y is not None and m_y > 0
    assert m_y == pytest.approx(0.043, abs=0.005)
    assert abs(m_y - 0.006) > 0.02
    doc_m = load_body_template("male_adult_athletic")
    assert doc_m.glute.y_frac is not None
    assert m_frac == pytest.approx(float(doc_m.glute.y_frac))
    assert m_frac == pytest.approx(0.192, abs=0.01)


def test_apply__male_pec_breast_y_envelope(tmp_path: Path) -> None:
    """Male pec: |breast_y_m| < 0.30 (soft-depth, not stature)."""
    report = _report(height_m=1.72)
    report_path = _write_report(tmp_path, report)
    payload = apply_body_template(
        report_path, "male_adult_athletic", tmp_path / "male_pec", force=True
    )
    breast_y = payload["constants"]["breast_y_m"]
    assert breast_y is not None
    assert abs(breast_y) < 0.30
    assert breast_y < 0
    # -0.35 * 0.12 * 1.72
    assert breast_y == pytest.approx(-0.35 * 0.12 * 1.72, abs=1e-4)
    assert payload["constants"]["breast_y_frac"] == pytest.approx(-0.35)


def test_female_seeds__glute_y_frac_retuned() -> None:
    doc = load_body_template("female_adult_athletic")
    assert doc.glute.y_frac is not None
    assert doc.glute.y_frac == pytest.approx(0.224, abs=0.01)
    doc_m = load_body_template("male_adult_athletic")
    assert doc_m.glute.y_frac is not None
    assert doc_m.glute.y_frac == pytest.approx(0.192, abs=0.01)
