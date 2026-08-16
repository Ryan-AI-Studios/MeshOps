"""Track 0053 — pelvis bucket shelf scale (oval + trap; authoring only)."""

from __future__ import annotations

import pytest

from meshops.proportion.blockout_recipe import (
    GLUTE_SEAT_BEYOND_REF_Y,
    GLUTE_SEAT_RY_ANISOTROPY_MAX,
    GLUTE_SEAT_RY_CAP_FRAC_H,
    GLUTE_SEAT_RY_FRAC_HALF_DEPTH,
    GLUTE_SEAT_RY_FROM_RX,
    GLUTE_SEAT_Y_CAP_FRAC_H,
    PELVIS_BUCKET_HALF_DEPTH_FRAC,
    PELVIS_BUCKET_HW_FRAC,
    PELVIS_BUCKET_Z_BOTTOM_FRAC_H,
    PELVIS_BUCKET_Z_TOP_FRAC_H,
    PELVIS_OVAL_RX_FRAC_HIP_HW,
    PELVIS_OVAL_RY_FRAC_HALF_HIP,
    PELVIS_OVAL_RY_OVER_RX_MAX,
    PELVIS_OVAL_RZ_FLOOR_M,
    PELVIS_OVAL_RZ_FRAC_H,
    PELVIS_OVAL_RZ_OVER_H_MAX,
    TORSO_OVAL_RY_CHEST_FRAC,
    TORSO_OVAL_RY_HIP_FRAC,
    TORSO_OVAL_RY_WAIST_FRAC,
    RecipePart,
    _part_rear_y_m,
    build_blockout_recipe,
)
from meshops.proportion.models import (
    CrossSection,
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


def _diam(band_id: str, *, half_width_m: float = 0.05) -> DiameterMeasure:
    return DiameterMeasure(
        band_id=band_id,
        view="front",
        width_px=40.0,
        width_eucl_px=40.0,
        theta_deg=90.0,
        width_frac=0.1,
        width_m=half_width_m * 2.0,
        half_width_m=half_width_m,
        mid_x_px=100.0,
        mid_y_px=200.0,
    )


def _band(
    band_id: str,
    *,
    depth_m: float = 0.26,
    z_frac: float = 0.55,
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


def _full_torso_report(
    *,
    height_m: float = 1.72,
    hip_x: float = 0.14,
    hip_z: float = 0.95,
    hip_depth_m: float = 0.26,
    with_glute_cs: bool = False,
) -> ProportionReport:
    """Minimal report so torso ovals + pelvis oval emit (chest/hip depth bands)."""
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "hip_l": _lm("hip_l", x_m=-hip_x, y_m=0.0, z_m=hip_z),
        "hip_r": _lm("hip_r", x_m=hip_x, y_m=0.0, z_m=hip_z),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
    ]
    bands = [
        _band("chest", depth_m=0.24, z_frac=0.72),
        _band("hip", depth_m=hip_depth_m, z_frac=0.55),
    ]
    cs: list[CrossSection] = []
    if with_glute_cs:
        cs = [
            CrossSection(
                level_id="glute",
                z_frac=0.50,
                rx_frac=0.11,
                ry_frac=0.09,
                sources=["test"],
            ),
        ]
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=bands,
        cross_sections=cs,
        quality=QualityFlags(),
    )


def _pelvis_oval(pkg_parts: list[RecipePart]) -> RecipePart:
    return next(p for p in pkg_parts if p.name == "RECIPE_pelvis_oval")


def _hip_half(depth_m: float = 0.26) -> float:
    return depth_m / 2.0


# ---------------------------------------------------------------------------
# Oval freezes (torso="ovals")
# ---------------------------------------------------------------------------


def test_t1_oval_ry_is_half_hip_times_0_60() -> None:
    """T1: oval ry = half_hip * PELVIS_OVAL_RY_FRAC_HALF_HIP (0.60)."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    p = _pelvis_oval(pkg.parts)
    half_hip = _hip_half()
    assert p.ry_m == pytest.approx(half_hip * PELVIS_OVAL_RY_FRAC_HALF_HIP, abs=1e-9)
    assert PELVIS_OVAL_RY_FRAC_HALF_HIP == 0.60


def test_t2_oval_rx_is_hip_hw_times_1_00() -> None:
    """T2: oval rx = hip_hw * PELVIS_OVAL_RX_FRAC_HIP_HW (1.00)."""
    hip_x = 0.14
    report = _full_torso_report(hip_x=hip_x)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    p = _pelvis_oval(pkg.parts)
    assert p.rx_m == pytest.approx(hip_x * PELVIS_OVAL_RX_FRAC_HIP_HW, abs=1e-9)
    assert PELVIS_OVAL_RX_FRAC_HIP_HW == 1.00


def test_t3_oval_rz_default_h() -> None:
    """T3: oval rz = max(0.028, 0.042*H) at default H=1.72."""
    h = 1.72
    report = _full_torso_report(height_m=h)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    p = _pelvis_oval(pkg.parts)
    expected = max(PELVIS_OVAL_RZ_FLOOR_M, PELVIS_OVAL_RZ_FRAC_H * h)
    assert p.rz_m == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(0.042 * h, abs=1e-9)
    assert expected > PELVIS_OVAL_RZ_FLOOR_M


def test_t3b_oval_rz_floor_short_h() -> None:
    """T3b: short H (< 0.667) locks rz to PELVIS_OVAL_RZ_FLOOR_M (0.028)."""
    # 0.028 / 0.042 ≈ 0.6667; use H well below so floor wins.
    h = 0.50
    # Scale landmarks so torso ovals still form (z_top > hip_z).
    report = _full_torso_report(height_m=h, hip_z=0.28)
    # Rebuild landmarks at short scale so shoulder stays above hip.
    report = ProportionReport(
        schema_version="1.1.0",
        height_m=h,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz={
            "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
            "chin": _lm("chin", x_m=0.0, y_m=-0.01, z_m=0.44),
            "shoulder_l": _lm("shoulder_l", x_m=-0.06, y_m=0.0, z_m=0.40),
            "shoulder_r": _lm("shoulder_r", x_m=0.06, y_m=0.0, z_m=0.40),
            "hip_l": _lm("hip_l", x_m=-0.04, y_m=0.0, z_m=0.28),
            "hip_r": _lm("hip_r", x_m=0.04, y_m=0.0, z_m=0.28),
            "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=0.48),
            "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.25),
            "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.02, z_m=0.36),
        },
        diameters=[
            _diam("bust", half_width_m=0.05),
            _diam("waist", half_width_m=0.04),
            _diam("neck", half_width_m=0.02),
        ],
        depth_bands=[
            _band("chest", depth_m=0.08, z_frac=0.72),
            _band("hip", depth_m=0.09, z_frac=0.55),
        ],
        quality=QualityFlags(),
    )
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    p = _pelvis_oval(pkg.parts)
    assert PELVIS_OVAL_RZ_FRAC_H * h < PELVIS_OVAL_RZ_FLOOR_M
    assert p.rz_m == pytest.approx(PELVIS_OVAL_RZ_FLOOR_M, abs=1e-9)
    assert p.rz_m == pytest.approx(0.028, abs=1e-9)


def test_t4_strict_ry_hip_gt_ry_pelvis() -> None:
    """T4: strict ry_hip > ry_pelvis on default full torso ovals."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    ry_h = float(by["RECIPE_torso_oval_hip"].ry_m or 0.0)
    ry_p = float(by["RECIPE_pelvis_oval"].ry_m or 0.0)
    assert ry_h > ry_p + 1e-9
    # Expected: 0.13*0.70 = 0.091; 0.13*0.60 = 0.078 (0073 hip ry)
    half_hip = _hip_half()
    assert ry_h == pytest.approx(half_hip * TORSO_OVAL_RY_HIP_FRAC, abs=1e-9)
    assert ry_p == pytest.approx(half_hip * PELVIS_OVAL_RY_FRAC_HALF_HIP, abs=1e-9)


def test_t5_b15_unit_ceilings() -> None:
    """T5: ry_p < half_hip; B15 ry_p/rx_p <= 0.45; rz_p <= 0.05*H.

    Product-like hip band (plan: half_hip~0.139, hip_hw~0.222 -> ry/rx~0.375).
    Narrow-hip synthetic (0.14 hw / 0.13 half) exceeds 0.45 by geometry alone.
    """
    h = 1.72
    hip_hw = 0.222
    half_hip = 0.139
    report = _full_torso_report(height_m=h, hip_x=hip_hw, hip_depth_m=half_hip * 2.0)
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    p = _pelvis_oval(pkg.parts)
    ry_p = float(p.ry_m or 0.0)
    rx_p = float(p.rx_m or 0.0)
    rz_p = float(p.rz_m or 0.0)
    assert ry_p == pytest.approx(half_hip * PELVIS_OVAL_RY_FRAC_HALF_HIP, abs=1e-9)
    assert rx_p == pytest.approx(hip_hw * PELVIS_OVAL_RX_FRAC_HIP_HW, abs=1e-9)
    assert ry_p < half_hip - 1e-9
    assert ry_p / rx_p <= PELVIS_OVAL_RY_OVER_RX_MAX + 1e-9
    assert ry_p / rx_p == pytest.approx(0.375, abs=5e-3)
    assert rz_p <= PELVIS_OVAL_RZ_OVER_H_MAX * h + 1e-9
    assert PELVIS_OVAL_RY_OVER_RX_MAX == 0.45
    assert PELVIS_OVAL_RZ_OVER_H_MAX == 0.05


def test_t6_message_pelvis_bucket_scale() -> None:
    """T6: message starts with 'pelvis bucket scale:' when pelvis oval emits."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    matches = [m for m in pkg.messages if m.startswith("pelvis bucket scale:")]
    assert len(matches) == 1
    msg = matches[0]
    p = _pelvis_oval(pkg.parts)
    assert f"rx={float(p.rx_m or 0.0):.4f}" in msg
    assert f"ry={float(p.ry_m or 0.0):.4f}" in msg
    assert f"rz={float(p.rz_m or 0.0):.4f}" in msg
    assert "fracs 1.00/0.60/0.042H" in msg


# ---------------------------------------------------------------------------
# Trap bucket (torso="trap", default)
# ---------------------------------------------------------------------------


def test_t7_trap_bucket_scale() -> None:
    """T7: trap pelvis_bucket half_depth*0.60; z span shortened; widths*1.00."""
    hip_x = 0.14
    hip_z = 0.95
    h = 1.72
    half_hip_depth = _hip_half()  # 0.13 from hip depth_m=0.26
    report = _full_torso_report(height_m=h, hip_x=hip_x, hip_z=hip_z)
    pkg = build_blockout_recipe(report, limbs=False, torso="trap")
    bucket = next(p for p in pkg.parts if p.name == "RECIPE_pelvis_bucket")
    assert bucket.kind == "box"
    assert bucket.half_depth_m == pytest.approx(
        half_hip_depth * PELVIS_BUCKET_HALF_DEPTH_FRAC, abs=1e-9
    )
    assert bucket.top_half_width_m == pytest.approx(hip_x * PELVIS_BUCKET_HW_FRAC, abs=1e-9)
    assert bucket.bottom_half_width_m == pytest.approx(hip_x * PELVIS_BUCKET_HW_FRAC, abs=1e-9)
    assert bucket.z_top_m == pytest.approx(hip_z + PELVIS_BUCKET_Z_TOP_FRAC_H * h, abs=1e-9)
    assert bucket.z_bottom_m == pytest.approx(
        max(0.0, hip_z - PELVIS_BUCKET_Z_BOTTOM_FRAC_H * h), abs=1e-9
    )
    # Shorter than pre-0053 0.03/0.12H span (0.15H → 0.10H).
    span = float(bucket.z_top_m or 0.0) - float(bucket.z_bottom_m or 0.0)
    assert span == pytest.approx(0.10 * h, abs=1e-9)
    assert span < 0.15 * h - 1e-9


def test_t8_0052_glute_seat_constants_still_exist() -> None:
    """T8: import smoke — 0052 glute seat freezes unchanged."""
    assert GLUTE_SEAT_RY_FRAC_HALF_DEPTH == 0.90
    assert GLUTE_SEAT_RY_FROM_RX == 1.05
    assert GLUTE_SEAT_BEYOND_REF_Y == 0.035
    assert GLUTE_SEAT_RY_CAP_FRAC_H == 0.10
    assert GLUTE_SEAT_Y_CAP_FRAC_H == 0.15
    assert GLUTE_SEAT_RY_ANISOTROPY_MAX == 2.0
    assert PELVIS_OVAL_RY_FRAC_HALF_HIP == 0.60
    assert PELVIS_BUCKET_HALF_DEPTH_FRAC == 0.60


def test_t9_torso_oval_fracs_unchanged_0047() -> None:
    """T9: torso chest/waist/hip ry fracs (0047 + 0065 + 0073 + 0090: 0.72/0.58/0.70)."""
    assert TORSO_OVAL_RY_CHEST_FRAC == 0.72
    assert TORSO_OVAL_RY_WAIST_FRAC == 0.58
    assert TORSO_OVAL_RY_HIP_FRAC == 0.70
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals")
    by = {p.name: p for p in pkg.parts}
    half_chest = 0.12
    half_hip = _hip_half()
    assert float(by["RECIPE_torso_oval_chest"].ry_m or 0.0) == pytest.approx(
        half_chest * TORSO_OVAL_RY_CHEST_FRAC, abs=1e-9
    )
    assert float(by["RECIPE_torso_oval_waist"].ry_m or 0.0) == pytest.approx(
        half_chest * TORSO_OVAL_RY_WAIST_FRAC, abs=1e-9
    )
    assert float(by["RECIPE_torso_oval_hip"].ry_m or 0.0) == pytest.approx(
        half_hip * TORSO_OVAL_RY_HIP_FRAC, abs=1e-9
    )


def test_t11_full_recipe_cascade_seat_beyond_pelvis() -> None:
    """T11: ovals + two_spheres -- ry_p=0.60*half; seat rear >= rear_p + BEYOND."""
    report = _full_torso_report(with_glute_cs=True)
    # Add glute depth so seat floor has a band.
    report = ProportionReport(
        schema_version=report.schema_version,
        height_m=report.height_m,
        head_unit_frac=report.head_unit_frac,
        landmarks_xyz=report.landmarks_xyz,
        diameters=report.diameters,
        depth_bands=[
            _band("chest", depth_m=0.24, z_frac=0.72),
            _band("hip", depth_m=0.26, z_frac=0.55),
            _band("glute", depth_m=0.27, z_frac=0.50),
        ],
        cross_sections=report.cross_sections,
        quality=report.quality,
    )
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals", glute="two_spheres")
    pelvis = _pelvis_oval(pkg.parts)
    half_hip = _hip_half()
    assert float(pelvis.ry_m or 0.0) == pytest.approx(
        half_hip * PELVIS_OVAL_RY_FRAC_HALF_HIP, abs=1e-9
    )
    rear_p = _part_rear_y_m(pelvis)
    assert rear_p is not None
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) >= 2
    for g in glutes:
        rear_g = _part_rear_y_m(g)
        assert rear_g is not None
        assert rear_g + 1e-9 >= float(rear_p) + GLUTE_SEAT_BEYOND_REF_Y
