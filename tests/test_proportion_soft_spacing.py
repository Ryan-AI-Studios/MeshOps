"""Track 0030 — top view + soft_spacing / breast_metrics (offline)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.test_proportion import eight_head_assist, make_package

from meshops.proportion.analyze import analyze_proportion, load_report, report_to_markdown
from meshops.proportion.assist import KNOWN_LANDMARK_IDS, point_to_landmark2d
from meshops.proportion.blockout_recipe import build_blockout_recipe
from meshops.proportion.constraints import _glute_cleft_gap_m, _intermammary_gap_m
from meshops.proportion.honesty import PROPORTION_HONESTY
from meshops.proportion.models import (
    CANONICAL_VIEW_KEYS,
    PROPORTION_SCHEMA_VERSION,
    REQUIRED_VIEW_KEYS,
    BreastMetrics,
    BreastSideMetrics,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    SoftSpacing,
    ViewLandmarks,
)
from meshops.proportion.scaffold import PNG_1X1_BYTES, scaffold_package
from meshops.proportion.soft_spacing import (
    MSG_TOP_ABSENT,
    NOTE_BUST_CENTER_UNRESOLVED,
    NOTE_CIRC_PROXY,
    NOTE_NO_CONTRA_BREAST,
    NOTE_NO_CONTRA_GLUTE,
    NOTE_PEAKS_ONLY,
    NOTE_RX_ASYMMETRY,
    NOTE_SCALE_UNRESOLVED,
    NOTE_STATURE_MPP,
    NOTE_VOLUME_PROXY,
    _classify_shape,
    _resolve_mpp,
    compute_soft_spacing,
)
from meshops.proportion.template import blank_assist_document


def _vl(
    view: str,
    width: int = 200,
    height: int = 200,
    landmarks: dict[str, tuple[float, float]] | None = None,
    *,
    figure_span_px: float | None = None,
) -> ViewLandmarks:
    lms = {}
    for lid, (x, y) in (landmarks or {}).items():
        lms[lid] = point_to_landmark2d(lid, x, y, width_px=width, height_px=height)
    return ViewLandmarks(
        view=view,
        width_px=width,
        height_px=height,
        landmarks=lms,
        figure_span_px=figure_span_px,
    )


def _xyz(
    lid: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
    sources: list[str] | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(
        id=lid,
        x=x_m,
        y=y_m,
        z=z_m,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        sources=sources or [],
    )


# ---------------------------------------------------------------------------
# View policy
# ---------------------------------------------------------------------------


def test_canonical_includes_top_required_unchanged() -> None:
    assert "top" in CANONICAL_VIEW_KEYS
    assert CANONICAL_VIEW_KEYS == ("front", "left", "three_quarter", "back", "top")
    assert REQUIRED_VIEW_KEYS == ("front", "left", "three_quarter")
    assert "top" not in REQUIRED_VIEW_KEYS


def test_blank_assist_includes_top() -> None:
    doc = blank_assist_document()
    assert "top" in doc["views"]
    assert "breast_medial_l" in doc["views"]["top"]["landmarks"]
    assert "glute_peak_r" in doc["views"]["top"]["landmarks"]


def test_known_landmark_ids_soft_vocab() -> None:
    for lid in (
        "breast_center_l",
        "breast_center_r",
        "breast_medial_l",
        "breast_medial_r",
        "breast_lateral_l",
        "breast_lateral_r",
        "breast_upper",
        "glute_peak_l",
        "glute_peak_r",
        "glute_cleft",
        "glute_medial_l",
        "glute_medial_r",
    ):
        assert lid in KNOWN_LANDMARK_IDS


def test_scaffold_include_top_stub(tmp_path: Path) -> None:
    out = tmp_path / "pkg"
    scaffold_package(out, stub_images=True, include_top_stub=True, include_back_stub=True)
    assert (out / "top.png").is_file()
    assert (out / "top.png").read_bytes() == PNG_1X1_BYTES
    cl = json.loads((out / "package_checklist.json").read_text(encoding="utf-8"))
    assert cl["view_keys_optional"] == ["back", "top"]


# ---------------------------------------------------------------------------
# Scale + gaps
# ---------------------------------------------------------------------------


def test_two_medials_intermammary_gap() -> None:
    """Two medials + scale → intermammary_gap_m > 0."""
    views = {
        "front": _vl("front", figure_span_px=480.0),
        "top": _vl(
            "top",
            landmarks={
                "breast_medial_l": (90.0, 100.0),
                "breast_medial_r": (110.0, 100.0),
                "bust_l": (70.0, 100.0),
                "bust_r": (130.0, 100.0),
            },
        ),
    }
    xyz = {
        "bust_l": _xyz("bust_l", x_m=-0.15, sources=["front"]),
        "bust_r": _xyz("bust_r", x_m=0.15, sources=["front"]),
    }
    soft, _bm, msgs = compute_soft_spacing(views, xyz, height_m=1.72)
    assert soft is not None
    assert soft.intermammary_gap_m is not None
    assert soft.intermammary_gap_m > 0
    assert MSG_TOP_ABSENT not in msgs


def test_glute_medials_and_peaks() -> None:
    views = {
        "top": _vl(
            "top",
            landmarks={
                "glute_medial_l": (95.0, 120.0),
                "glute_medial_r": (105.0, 120.0),
                "glute_peak_l": (80.0, 130.0),
                "glute_peak_r": (120.0, 130.0),
                "bust_l": (70.0, 100.0),
                "bust_r": (130.0, 100.0),
            },
        ),
    }
    xyz = {
        "bust_l": _xyz("bust_l", x_m=-0.15),
        "bust_r": _xyz("bust_r", x_m=0.15),
    }
    soft, _, _ = compute_soft_spacing(views, xyz, height_m=1.7)
    assert soft is not None
    assert soft.glute_cleft_gap_m is not None and soft.glute_cleft_gap_m > 0
    assert soft.glute_peak_span_m is not None and soft.glute_peak_span_m > 0
    assert soft.glute_peak_span_m > soft.glute_cleft_gap_m


def test_peaks_only_note() -> None:
    views = {
        "top": _vl(
            "top",
            landmarks={
                "glute_peak_l": (80.0, 130.0),
                "glute_peak_r": (120.0, 130.0),
                "bust_l": (70.0, 100.0),
                "bust_r": (130.0, 100.0),
            },
        ),
    }
    xyz = {
        "bust_l": _xyz("bust_l", x_m=-0.15),
        "bust_r": _xyz("bust_r", x_m=0.15),
    }
    soft, _, _ = compute_soft_spacing(views, xyz)
    assert soft is not None
    assert soft.glute_cleft_gap_m is None
    assert soft.glute_peak_span_m is not None
    assert NOTE_PEAKS_ONLY in soft.notes


def test_missing_top_b3_no_partial_package(tmp_path: Path) -> None:
    """Missing top → B3 message; partial_package false; score unchanged."""
    d = make_package(tmp_path, assist=eight_head_assist())
    report = analyze_proportion(d, run_heuristic_frame=False)
    assert MSG_TOP_ABSENT in report.messages
    assert report.quality.partial_package is False
    # REQUIRED views present → full package score (top does not reweight)
    assert report.package_score == pytest.approx(100.0, abs=0.1)
    md = report_to_markdown(report)
    assert "## Soft spacing" in md


def test_single_breast_null_gap() -> None:
    views = {
        "top": _vl(
            "top",
            landmarks={
                "breast_center_l": (90.0, 100.0),
                "bust_l": (70.0, 100.0),
                "bust_r": (130.0, 100.0),
            },
        ),
    }
    xyz = {
        "bust_l": _xyz("bust_l", x_m=-0.15),
        "bust_r": _xyz("bust_r", x_m=0.15),
    }
    soft, bm, _ = compute_soft_spacing(views, xyz)
    assert soft is not None
    assert soft.intermammary_gap_m is None
    assert soft.breast_center_span_m is None
    assert NOTE_NO_CONTRA_BREAST in soft.notes
    # no invent right side span
    if bm is not None:
        assert bm.right is None or bm.right.rx_m is None


def test_single_glute_peak_null_span() -> None:
    views = {
        "top": _vl(
            "top",
            landmarks={
                "glute_peak_l": (80.0, 130.0),
                "bust_l": (70.0, 100.0),
                "bust_r": (130.0, 100.0),
            },
        ),
    }
    xyz = {
        "bust_l": _xyz("bust_l", x_m=-0.15),
        "bust_r": _xyz("bust_r", x_m=0.15),
    }
    soft, _, _ = compute_soft_spacing(views, xyz)
    assert soft is not None
    assert soft.glute_peak_span_m is None
    assert soft.glute_cleft_gap_m is None
    assert NOTE_NO_CONTRA_GLUTE in soft.notes


def test_classify_shape_teardrop_and_prolate() -> None:
    """Shape cascade: teardrop when lower present + rz>ry*1.15; else prolate/sphere."""
    assert _classify_shape(0.05, 0.05, 0.08, breast_lower_present=True) == "teardrop_proxy"
    # lower absent → no teardrop even if rz tall
    assert _classify_shape(0.05, 0.05, 0.08, breast_lower_present=False) == "prolate"
    assert _classify_shape(0.05, 0.05, 0.05, breast_lower_present=False) == "sphere"


def test_rx_asymmetry_note() -> None:
    """Dual-side rx differ >15% → NOTE_RX_ASYMMETRY in breast_metrics.symmetry_notes."""
    views = {
        "top": _vl(
            "top",
            landmarks={
                "breast_medial_l": (95.0, 100.0),
                "breast_lateral_l": (70.0, 100.0),  # wider left
                "breast_medial_r": (105.0, 100.0),
                "breast_lateral_r": (115.0, 100.0),  # narrow right
                "bust_l": (60.0, 100.0),
                "bust_r": (140.0, 100.0),
            },
        ),
        "front": _vl("front", figure_span_px=480.0),
    }
    xyz = {
        "bust_l": _xyz("bust_l", x_m=-0.18, sources=["front"]),
        "bust_r": _xyz("bust_r", x_m=0.18, sources=["front"]),
        "breast_medial_l": _xyz("breast_medial_l", x_m=-0.02, sources=["top"]),
        "breast_lateral_l": _xyz("breast_lateral_l", x_m=-0.12, sources=["top"]),
        "breast_medial_r": _xyz("breast_medial_r", x_m=0.02, sources=["top"]),
        "breast_lateral_r": _xyz("breast_lateral_r", x_m=0.05, sources=["top"]),
    }
    soft, bm, _ = compute_soft_spacing(views, xyz, height_m=1.72)
    assert bm is not None
    assert bm.left is not None and bm.right is not None
    assert bm.left.rx_m is not None and bm.right.rx_m is not None
    assert abs(bm.left.rx_m - bm.right.rx_m) / max(bm.left.rx_m, bm.right.rx_m) > 0.15
    assert NOTE_RX_ASYMMETRY in bm.symmetry_notes
    assert soft is not None


def test_volume_and_circ_honesty() -> None:
    """Fixed rx,ry,rz → volume 4/3π product + honesty notes."""
    # Build side metrics via compute with fused radii sources
    views = {
        "top": _vl(
            "top",
            landmarks={
                "breast_medial_l": (95.0, 100.0),
                "breast_lateral_l": (75.0, 100.0),
                "breast_medial_r": (105.0, 100.0),
                "breast_lateral_r": (125.0, 100.0),
                "bust_l": (70.0, 100.0),
                "bust_r": (130.0, 100.0),
            },
            figure_span_px=100.0,  # must not be used as stature
        ),
        "left": _vl(
            "left",
            landmarks={
                "breast_front": (80.0, 200.0),
                "breast_back": (120.0, 200.0),
                "breast_upper": (100.0, 180.0),
                "breast_lower_l": (100.0, 220.0),
                "breast_lower_r": (100.0, 220.0),
            },
        ),
        "front": _vl("front", figure_span_px=480.0),
    }
    # Direct meters for axes
    xyz = {
        "bust_l": _xyz("bust_l", x_m=-0.15, sources=["front"]),
        "bust_r": _xyz("bust_r", x_m=0.15, sources=["front"]),
        "breast_medial_l": _xyz("breast_medial_l", x_m=-0.02, y_m=-0.05, sources=["top"]),
        "breast_lateral_l": _xyz("breast_lateral_l", x_m=-0.10, y_m=-0.04, sources=["top"]),
        "breast_medial_r": _xyz("breast_medial_r", x_m=0.02, y_m=-0.05, sources=["top"]),
        "breast_lateral_r": _xyz("breast_lateral_r", x_m=0.10, y_m=-0.04, sources=["top"]),
        "breast_front": _xyz("breast_front", y_m=-0.08, sources=["left"]),
        "breast_back": _xyz("breast_back", y_m=-0.02, sources=["left"]),
        "breast_upper": _xyz("breast_upper", z_m=1.30, sources=["left"]),
        "breast_lower_l": _xyz("breast_lower_l", z_m=1.20, sources=["left"]),
        "breast_lower_r": _xyz("breast_lower_r", z_m=1.20, sources=["left"]),
    }
    soft, bm, _ = compute_soft_spacing(views, xyz, height_m=1.72)
    assert bm is not None
    assert bm.left is not None
    left = bm.left
    assert left.rx_m is not None and left.ry_m is not None and left.rz_m is not None
    expected_vol = (4.0 / 3.0) * math.pi * left.rx_m * left.ry_m * left.rz_m
    assert left.volume_proxy_m3 == pytest.approx(expected_vol, rel=1e-9)
    assert left.circumference_proxy_m == pytest.approx(math.pi * (left.rx_m + left.ry_m), rel=1e-9)
    assert soft is not None
    assert NOTE_VOLUME_PROXY in soft.notes
    assert NOTE_CIRC_PROXY in soft.notes
    assert left.hang_tilt_deg is None


def test_top_y_mapping_image_up_is_plus_y() -> None:
    """Image-top mark → larger body +Y than image-bottom mark (B1)."""
    views = {
        "top": _vl(
            "top",
            width=200,
            height=200,
            landmarks={
                # image-top (small y_px) and image-bottom (large y_px)
                "breast_center_l": (100.0, 40.0),  # image up → +Y
                "breast_center_r": (100.0, 160.0),  # image down -> -Y
                "bust_l": (70.0, 100.0),
                "bust_r": (130.0, 100.0),
            },
        ),
    }
    xyz = {
        "bust_l": _xyz("bust_l", x_m=-0.15),
        "bust_r": _xyz("bust_r", x_m=0.15),
    }
    soft, _, _ = compute_soft_spacing(views, xyz)
    assert soft is not None
    # Use internal plan mapping via recompute endpoints
    from meshops.proportion.soft_spacing import _endpoint_y_m, _resolve_mpp

    mpp = _resolve_mpp(views, xyz, height_m=None, diameters=None, notes=[])
    assert mpp is not None
    y_up = _endpoint_y_m(
        "breast_center_l",
        views=views,
        landmarks_xyz=xyz,
        mpp=mpp,
        ref_x_px=100.0,
        ref_y_px=100.0,
    )
    y_down = _endpoint_y_m(
        "breast_center_r",
        views=views,
        landmarks_xyz=xyz,
        mpp=mpp,
        ref_x_px=100.0,
        ref_y_px=100.0,
    )
    assert y_up is not None and y_down is not None
    assert y_up > y_down


def test_slant_sign_lateral_more_posterior() -> None:
    """Known plan slant: lateral more +Y than medial → positive slant_deg."""
    views = {
        "top": _vl(
            "top",
            landmarks={
                "breast_medial_l": (95.0, 110.0),  # more face (-Y in plan after map)
                "breast_lateral_l": (70.0, 80.0),  # more back (+Y: smaller y_px)
                "bust_l": (60.0, 100.0),
                "bust_r": (140.0, 100.0),
            },
        ),
    }
    xyz = {
        "bust_l": _xyz("bust_l", x_m=-0.18),
        "bust_r": _xyz("bust_r", x_m=0.18),
    }
    _, bm, _ = compute_soft_spacing(views, xyz)
    assert bm is not None and bm.left is not None
    assert bm.left.slant_deg is not None
    # lateral more posterior (+Y) → positive
    assert bm.left.slant_deg > 0


def test_center_span_never_bust_edges() -> None:
    views = {
        "top": _vl(
            "top",
            landmarks={
                "bust_l": (70.0, 100.0),
                "bust_r": (130.0, 100.0),
            },
        ),
    }
    xyz = {
        "bust_l": _xyz("bust_l", x_m=-0.15),
        "bust_r": _xyz("bust_r", x_m=0.15),
    }
    soft, _, _ = compute_soft_spacing(views, xyz)
    assert soft is not None
    assert soft.breast_center_span_m is None
    assert NOTE_BUST_CENTER_UNRESOLVED in soft.notes


def test_never_stature_from_top_figure_span() -> None:
    """mpp path refuses top.figure_span_px as stature (AI fold B2)."""
    views = {
        "top": _vl("top", figure_span_px=50.0),  # body-depth pixels — not stature
        # no front
    }
    notes: list[str] = []
    mpp = _resolve_mpp(views, {}, height_m=1.72, diameters=None, notes=notes)
    # Without front span or diameter, scale must be unresolved — not 1.72/50
    assert mpp is None
    assert NOTE_SCALE_UNRESOLVED in notes
    assert NOTE_STATURE_MPP not in notes

    # With front span, stature fallback uses front not top
    views2 = {
        "front": _vl("front", figure_span_px=480.0),
        "top": _vl("top", figure_span_px=50.0),
    }
    notes2: list[str] = []
    mpp2 = _resolve_mpp(views2, {}, height_m=1.72, diameters=None, notes=notes2)
    assert mpp2 == pytest.approx(1.72 / 480.0)
    assert NOTE_STATURE_MPP in notes2
    assert mpp2 != pytest.approx(1.72 / 50.0)


# ---------------------------------------------------------------------------
# Schema load / write
# ---------------------------------------------------------------------------


def test_load_1_0_0_and_1_1_0_no_error() -> None:
    for ver in ("1.0.0", "1.1.0"):
        r = ProportionReport.model_validate(
            {
                "schema_version": ver,
                "honesty": PROPORTION_HONESTY,
                "package_score": 40.0,
            }
        )
        assert r.schema_version == ver
        assert r.soft_spacing is None
        assert r.breast_metrics is None


def test_write_load_1_2_0_roundtrip(tmp_path: Path) -> None:
    assert PROPORTION_SCHEMA_VERSION == "1.2.0"
    report = ProportionReport(
        schema_version="1.2.0",
        soft_spacing=SoftSpacing(
            intermammary_gap_m=0.04,
            breast_center_span_m=0.16,
            glute_cleft_gap_m=0.03,
            glute_peak_span_m=0.18,
            source_views=["top"],
            notes=[NOTE_VOLUME_PROXY],
        ),
        breast_metrics=BreastMetrics(
            left=BreastSideMetrics(
                rx_m=0.06,
                ry_m=0.05,
                rz_m=0.07,
                volume_proxy_m3=(4.0 / 3.0) * math.pi * 0.06 * 0.05 * 0.07,
                circumference_proxy_m=math.pi * (0.06 + 0.05),
                shape="prolate",
                slant_deg=12.0,
                hang_tilt_deg=None,
            ),
            right=None,
            symmetry_notes=[],
        ),
    )
    path = tmp_path / "proportion_report.json"
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    loaded = load_report(path)
    assert loaded.schema_version == "1.2.0"
    assert loaded.soft_spacing is not None
    assert loaded.soft_spacing.intermammary_gap_m == pytest.approx(0.04)
    assert loaded.breast_metrics is not None
    assert loaded.breast_metrics.left is not None
    assert loaded.breast_metrics.left.hang_tilt_deg is None
    assert loaded.breast_metrics.left.shape == "prolate"


# ---------------------------------------------------------------------------
# Consumers B7
# ---------------------------------------------------------------------------


def test_constraints_measured_gap_beats_template() -> None:
    """Report gap 0.04 beats template 0.08 (B7)."""

    class _C:
        intermammary_gap_m = 0.08
        intermammary_gap_frac = 0.2
        glute_cleft_m = 0.10
        glute_cleft_frac = 0.12

    class _T:
        constants = _C()

    class _R:
        soft_spacing = SoftSpacing(intermammary_gap_m=0.04, glute_cleft_gap_m=0.025)

    gap = _intermammary_gap_m(_T(), _R())
    assert gap == pytest.approx(0.04)
    g2 = _glute_cleft_gap_m(_T(), _R())
    assert g2 == pytest.approx(0.025)


def test_constraints_template_when_no_measured() -> None:
    class _C:
        intermammary_gap_m = 0.08
        glute_cleft_m = 0.06

    class _T:
        constants = _C()

    class _R:
        soft_spacing = None

    assert _intermammary_gap_m(_T(), _R()) == pytest.approx(0.08)
    assert _glute_cleft_gap_m(_T(), _R()) == pytest.approx(0.06)


def _recipe_breast_center_gap(measured_gap_m: float) -> tuple[float, float, float]:
    """Build recipe with fixed fixture + measured gap; return (center_gap, rx, shoulder_hw).

    0067 B3 (post-pass SoT over emit provisional X):
    medial_half = max(0.010, measured_gap/2);
    offset = max(rx + medial_half, shoulder_hw*0.25);
    offset = min(offset, shoulder_hw*0.45);  # B3b
    centers at ±offset; center_span = 2*offset.
    rx is post-pass (after B1 cap + B2 tear equalize).
    Shoulders ±0.25 keep B3b from collapsing both gaps to the same offset.
    """
    from meshops.proportion.models import CrossSection

    shoulder_hw = 0.25
    report = ProportionReport(
        schema_version="1.2.0",
        height_m=1.72,
        head_unit_frac=1.0 / 8.0,
        soft_spacing=SoftSpacing(intermammary_gap_m=measured_gap_m),
        cross_sections=[
            CrossSection(
                level_id="bust", z_frac=0.75, rx_frac=0.12, ry_frac=0.08, sources=["front"]
            ),
        ],
        depth_bands=[],
        diameters=[],
        landmarks_xyz={
            "shoulder_l": _xyz("shoulder_l", x_m=-shoulder_hw, z_m=1.4),
            "shoulder_r": _xyz("shoulder_r", x_m=shoulder_hw, z_m=1.4),
            "crotch_pubic": _xyz("crotch_pubic", z_m=0.9),
        },
    )
    pkg = build_blockout_recipe(report)
    breasts = [p for p in pkg.parts if p.role == "breast_soft"]
    assert len(breasts) == 2, f"expected exactly 2 breast_soft parts, got {len(breasts)}"
    xs = sorted(p.center[0] for p in breasts if p.center is not None)
    assert len(xs) == 2
    center_gap = xs[1] - xs[0]
    rx_vals = [p.rx_m for p in breasts if p.rx_m is not None]
    assert len(rx_vals) == 2
    assert rx_vals[0] == pytest.approx(rx_vals[1], rel=1e-9)
    rx = float(rx_vals[0])
    return center_gap, rx, shoulder_hw


def _expected_b3_center_span(rx: float, measured_gap_m: float, shoulder_hw: float) -> float:
    """0067 B3/B3b: base=rx+max(0.010, gap/2); floor sh*0.25; cap never below base."""
    medial = max(0.010, measured_gap_m / 2.0)
    base = rx + medial
    offset = max(base, shoulder_hw * 0.25)
    offset = min(offset, max(shoulder_hw * 0.45, base))
    return 2.0 * offset


def test_recipe_measured_gap_smoke() -> None:
    """Recipe B7/0067: measured soft_spacing drives center gap via B3 sternum post-pass."""
    gap_04, rx_04, shoulder_hw = _recipe_breast_center_gap(0.04)
    expected_04 = _expected_b3_center_span(rx_04, 0.04, shoulder_hw)
    assert gap_04 == pytest.approx(expected_04, rel=1e-5)

    gap_20, rx_20, shoulder_hw_20 = _recipe_breast_center_gap(0.20)
    assert shoulder_hw_20 == pytest.approx(shoulder_hw, rel=1e-9)
    assert rx_20 == pytest.approx(rx_04, rel=1e-9)
    expected_20 = _expected_b3_center_span(rx_20, 0.20, shoulder_hw_20)
    assert gap_20 == pytest.approx(expected_20, rel=1e-5)

    # soft_spacing is consulted: larger measured gap → larger center separation
    assert gap_04 < gap_20
    # would fail if only pre-0030 shoulder*0.35 path were used (same gap for both)
    assert gap_04 != pytest.approx(gap_20, rel=1e-5)


def test_male_no_breast_null_metrics_no_error(tmp_path: Path) -> None:
    d = make_package(tmp_path)
    report = analyze_proportion(d, run_heuristic_frame=False)
    # No soft landmarks → metrics null or empty sides; no exception
    if report.breast_metrics is not None:
        assert report.breast_metrics.left is None or report.breast_metrics.left.rx_m is None
    # soft_spacing may exist with B3 note in messages
    assert MSG_TOP_ABSENT in report.messages


def test_extra_forbid_soft_models() -> None:
    with pytest.raises(ValidationError):
        SoftSpacing.model_validate({"intermammary_gap_m": 0.04, "extra": True})
    with pytest.raises(ValidationError):
        BreastSideMetrics.model_validate({"rx_m": 0.05, "bogus": 1})


def test_source_views_canonical_order() -> None:
    """C1: source_views = contrib views in front,left,three_quarter,back,top order."""
    from meshops.proportion.soft_spacing import _ordered_sources

    assert _ordered_sources({"top", "front", "back"}) == ["front", "back", "top"]
    assert _ordered_sources({"three_quarter", "left"}) == ["left", "three_quarter"]
    assert _ordered_sources(set()) == []


def test_diameter_scale_path() -> None:
    views = {
        "top": _vl(
            "top",
            landmarks={
                "bust_l": (70.0, 100.0),
                "bust_r": (130.0, 100.0),  # 60 px
                "breast_medial_l": (95.0, 100.0),
                "breast_medial_r": (105.0, 100.0),  # 10 px
            },
        ),
    }
    diameters = [
        DiameterMeasure(
            band_id="bust",
            view="front",
            width_px=60.0,
            width_eucl_px=60.0,
            theta_deg=0.0,
            width_frac=0.18,
            width_m=0.30,
            mid_x_px=256.0,
            mid_y_px=200.0,
        )
    ]
    soft, _, _ = compute_soft_spacing(views, {}, diameters=diameters)
    assert soft is not None
    assert soft.intermammary_gap_m == pytest.approx(0.30 * (10.0 / 60.0), rel=1e-6)
