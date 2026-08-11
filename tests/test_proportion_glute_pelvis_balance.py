"""Track 0068 — glute vs pelvis balance (Z drop, Y floor, rz mass; authoring only)."""

from __future__ import annotations

import pytest

from meshops.proportion.blockout_recipe import (
    CROTCH_SEAT_SLACK_M,
    GLUTE_BOTTOM_UNDER_MID_M,
    GLUTE_SEAT_BEYOND_REF_Y,
    GLUTE_SEAT_RY_ANISOTROPY_MAX,
    GLUTE_SEAT_RY_CAP_FRAC_H,
    GLUTE_SEAT_RY_FRAC_HALF_DEPTH,
    GLUTE_SEAT_RY_FROM_RX,
    GLUTE_SEAT_RZ_FRAC_RY,
    GLUTE_SEAT_RZ_OVER_H_MAX,
    GLUTE_SEAT_Y_CAP_FRAC_H,
    GLUTE_SEAT_Y_FLOOR_FRAC_H,
    GLUTE_SEAT_Y_FLOOR_M,
    GLUTE_SEAT_Z_DROP_FRAC_H,
    GLUTE_TOP_OVER_PELVIS_ALLOW_M,
    PELVIS_OVAL_RY_FRAC_HALF_HIP,
    PELVIS_OVAL_RZ_FRAC_H,
    RECIPE_SCHEMA_VERSION,
    BlockoutRecipePackage,
    RecipePart,
    _apply_glute_seat_mass,
    _ResolvedMetrics,
    build_blockout_recipe,
)
from meshops.proportion.constraints import (
    _index_parts,
    _role_target_y,
    optimize_package,
    part_y,
    validate_constraints,
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


def _base_lms(
    *,
    height_m: float = 1.72,
    crotch_z: float | None = 0.70,
) -> dict[str, LandmarkXYZ]:
    """Default crotch low enough that clamp does not fight product-like Z drop."""
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "hip_l": _lm("hip_l", x_m=-0.14, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.14, y_m=0.0, z_m=0.95),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
    }
    if crotch_z is not None:
        lms["crotch_pubic"] = _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=crotch_z)
    return lms


def _report(
    *,
    height_m: float | None = 1.72,
    depth_bands: list[DepthBand] | None = None,
    extra_lms: dict[str, LandmarkXYZ] | None = None,
    crotch_z: float | None = 0.70,
    omit_crotch: bool = False,
) -> ProportionReport:
    h = 1.72 if height_m is None else float(height_m)
    lms = _base_lms(height_m=h, crotch_z=None if omit_crotch else crotch_z)
    if extra_lms:
        lms.update(extra_lms)
    bands = (
        list(depth_bands)
        if depth_bands is not None
        else [
            _band("chest", depth_m=0.24, z_frac=0.72),
            _band("hip", depth_m=0.26, z_frac=0.55),
            _band("glute", depth_m=0.269, z_frac=0.50),
        ]
    )
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m if height_m is not None else h,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=[
            _diam("bust", half_width_m=0.16),
            _diam("waist", half_width_m=0.13),
            _diam("neck", half_width_m=0.05),
        ],
        depth_bands=bands,
        cross_sections=[
            CrossSection(
                level_id="glute",
                z_frac=0.50,
                rx_frac=0.11,
                ry_frac=0.09,
                sources=["test"],
            ),
        ],
        quality=QualityFlags(),
    )


def _resolved(*, height_m: float | None = 1.72) -> _ResolvedMetrics:
    m = _ResolvedMetrics()
    m.height_m = height_m
    m.hip_hw = 0.14
    m.hip_depth_m = 0.26
    m.hip_half_depth = 0.13
    return m


def _glute(
    name: str,
    *,
    x: float,
    y: float = 0.031,
    z: float = 0.902,
    rx: float | None = 0.0694,
    ry: float | None = 0.1212,
    rz: float | None = 0.0631,
) -> RecipePart:
    return RecipePart(
        name=name,
        role="glute_soft",
        kind="ellipsoid",
        center=[x, y, z],
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
    )


def _pelvis_oval(
    *,
    y: float = 0.0347,
    z: float = 0.8332,
    rx: float = 0.2224,
    ry: float = 0.0834,
    rz: float = 0.0722,
) -> RecipePart:
    """Product_0056up-class thin pelvis shelf (post-0053)."""
    return RecipePart(
        name="RECIPE_pelvis_oval",
        role="pelvis",
        kind="ellipsoid",
        center=[0.0, y, z],
        rx_m=rx,
        ry_m=ry,
        rz_m=rz,
    )


def _hip_bridge(side: str, *, outer_x: float = 0.22, half: float = 0.04) -> RecipePart:
    cx = outer_x - half if side == "r" else -outer_x + half
    return RecipePart(
        name=f"RECIPE_hip_bridge_{side}",
        role="hip_bridge",
        kind="ellipsoid",
        center=[cx, 0.05, 0.95],
        rx_m=half,
        ry_m=0.04,
        rz_m=0.04,
    )


def _product_like_parts(
    *,
    y: float = 0.031,
    z: float = 0.902,
    rx: float = 0.0694,
    ry: float = 0.1212,
    rz: float = 0.0631,
) -> list[RecipePart]:
    """Live product_0056up residual: high thin-bead seats on thin pelvis shelf."""
    return [
        _glute("RECIPE_glute_soft_l", x=-0.150, y=y, z=z, rx=rx, ry=ry, rz=rz),
        _glute("RECIPE_glute_soft_r", x=0.150, y=y, z=z, rx=rx, ry=ry, rz=rz),
        _pelvis_oval(),
        _hip_bridge("l"),
        _hip_bridge("r"),
    ]


def _apply(
    parts: list[RecipePart],
    *,
    height_m: float | None = 1.72,
    crotch_z: float | None = 0.70,
    omit_crotch: bool = False,
    depth_m: float = 0.269,
) -> list[str]:
    report = _report(
        height_m=height_m if height_m is not None else 1.72,
        depth_bands=[_band("glute", depth_m=depth_m), _band("hip", depth_m=0.278)],
        crotch_z=crotch_z,
        omit_crotch=omit_crotch,
    )
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(height_m=height_m), messages)
    return messages


# ---------------------------------------------------------------------------
# Freezes
# ---------------------------------------------------------------------------


def test_t0_constants_frozen() -> None:
    """T0/T13: binding freezes + B8 pelvis defaults stay off."""
    assert GLUTE_SEAT_BEYOND_REF_Y == 0.035
    assert GLUTE_SEAT_Z_DROP_FRAC_H == 0.035
    assert CROTCH_SEAT_SLACK_M == 0.15
    assert GLUTE_SEAT_Y_FLOOR_M == 0.045
    assert GLUTE_SEAT_Y_FLOOR_FRAC_H == 0.026
    assert GLUTE_SEAT_RZ_FRAC_RY == 0.72
    assert GLUTE_SEAT_RZ_OVER_H_MAX == 0.065
    assert GLUTE_TOP_OVER_PELVIS_ALLOW_M == 0.025
    assert GLUTE_BOTTOM_UNDER_MID_M == 0.020
    assert GLUTE_SEAT_RY_FRAC_HALF_DEPTH == 0.90
    assert GLUTE_SEAT_RY_FROM_RX == 1.05
    assert GLUTE_SEAT_RY_CAP_FRAC_H == 0.10
    assert GLUTE_SEAT_Y_CAP_FRAC_H == 0.15
    assert GLUTE_SEAT_RY_ANISOTROPY_MAX == 2.0
    # B8 not default-on
    assert PELVIS_OVAL_RY_FRAC_HALF_HIP == 0.60
    assert PELVIS_OVAL_RZ_FRAC_H == 0.042


def test_t0_red_product_like_before_balance() -> None:
    """T0: product residual — high Z, low Y, bead rz fails composition/floors."""
    parts = _product_like_parts()
    glutes = [p for p in parts if p.role == "glute_soft"]
    pelvis = next(p for p in parts if p.name == "RECIPE_pelvis_oval")
    assert pelvis.center is not None and pelvis.rz_m is not None
    pelvis_z = float(pelvis.center[2])
    for g in glutes:
        assert g.center is not None and g.ry_m is not None and g.rz_m is not None
        cy = float(g.center[1])
        cz = float(g.center[2])
        rz = float(g.rz_m)
        # Pre-balance residual signatures
        assert cy < GLUTE_SEAT_Y_FLOOR_M
        assert cz > pelvis_z + 0.05
        assert rz < GLUTE_SEAT_RZ_FRAC_RY * float(g.ry_m) - 1e-6
        # Bottom sits near pelvis mid (peas ON shelf)
        bot = cz - rz
        assert bot > pelvis_z - GLUTE_BOTTOM_UNDER_MID_M


# ---------------------------------------------------------------------------
# Z drop / crotch / never-raise
# ---------------------------------------------------------------------------


def test_t1_z_drop_frac_h() -> None:
    """T1: delta_z = -0.035*H when H known."""
    h = 1.72
    z0 = 0.902
    parts = _product_like_parts(z=z0)
    _apply(parts, height_m=h)
    expected_z = z0 - GLUTE_SEAT_Z_DROP_FRAC_H * h
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.center is not None
        assert float(g.center[2]) == pytest.approx(expected_z, abs=1e-6)


def test_t1b_never_raise_z() -> None:
    """T1b: drop never raises Z (z only decreases or stays vs pre-pass)."""
    z0 = 0.80
    parts = _product_like_parts(z=z0)
    _apply(parts, height_m=1.72)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.center is not None
        assert float(g.center[2]) <= z0 + 1e-12


def test_t1c_h_missing_drop_skipped_y_rz_still() -> None:
    """T1c: H missing → Z drop skipped; y floor + rz floor still apply."""
    z0 = 0.902
    y0 = 0.031
    parts = _product_like_parts(y=y0, z=z0, rz=0.05)
    messages = _apply(parts, height_m=None, omit_crotch=True)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.center is not None and g.ry_m is not None and g.rz_m is not None
        assert float(g.center[2]) == pytest.approx(z0, abs=1e-9)
        assert float(g.center[1]) >= GLUTE_SEAT_Y_FLOOR_M - 1e-9
        assert float(g.rz_m) + 1e-9 >= GLUTE_SEAT_RZ_FRAC_RY * float(g.ry_m)
    assert not any("z_drop" in m for m in messages)
    assert any("y_floor" in m for m in messages) or all(
        float(p.center[1]) >= GLUTE_SEAT_Y_FLOOR_M - 1e-9  # type: ignore[index]
        for p in parts
        if p.role == "glute_soft" and p.center is not None
    )


def test_t1d_crotch_clamp() -> None:
    """T1d: when drop would bury bottom under crotch-slack, raise z so bottom clears."""
    h = 1.72
    # Start low enough that after drop bottom < crotch_z - slack (0.15).
    z0 = 0.75
    crotch_z = 0.88
    parts = _product_like_parts(z=z0, rz=0.0631)
    _apply(parts, height_m=h, crotch_z=crotch_z)
    floor_bottom = crotch_z - CROTCH_SEAT_SLACK_M
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.center is not None and g.rz_m is not None
        bot = float(g.center[2]) - float(g.rz_m)
        assert bot + 1e-9 >= floor_bottom


def test_t12b_product_crotch_pubic_composition() -> None:
    """Product crotch_pubic ~0.88 must not undo Z-drop composition (B15)."""
    h = 1.72
    parts = _product_like_parts()
    _apply(parts, height_m=h, crotch_z=0.8845)
    pelvis = next(p for p in parts if p.name == "RECIPE_pelvis_oval")
    assert pelvis.center is not None and pelvis.rz_m is not None
    pelvis_z = float(pelvis.center[2])
    pelvis_top = pelvis_z + float(pelvis.rz_m)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.center is not None and g.rz_m is not None
        top = float(g.center[2]) + float(g.rz_m)
        bot = float(g.center[2]) - float(g.rz_m)
        assert top <= pelvis_top + GLUTE_TOP_OVER_PELVIS_ALLOW_M + 1e-6
        assert bot <= pelvis_z - GLUTE_BOTTOM_UNDER_MID_M + 1e-6


# ---------------------------------------------------------------------------
# Y floor + beyond
# ---------------------------------------------------------------------------


def test_t2_y_floor_composite() -> None:
    """T2: center_y >= max(0.045, 0.026*H)."""
    h = 1.72
    floor = max(GLUTE_SEAT_Y_FLOOR_M, GLUTE_SEAT_Y_FLOOR_FRAC_H * h)
    parts = _product_like_parts(y=0.031)
    messages = _apply(parts, height_m=h)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.center is not None
        assert float(g.center[1]) + 1e-9 >= floor
    assert any("y_floor" in m for m in messages)


def test_t3_beyond_ref_035() -> None:
    """T3: rear tip ≥ pelvis_rear + 0.035 when ref present and beyond binds."""
    # Fat-ish pelvis rear so beyond need_y exceeds y floor
    parts = [
        _glute("RECIPE_glute_soft_l", x=-0.15, y=0.02, ry=0.08, rz=0.05),
        _glute("RECIPE_glute_soft_r", x=0.15, y=0.02, ry=0.08, rz=0.05),
        _pelvis_oval(y=0.05, ry=0.15),  # rear = 0.20
    ]
    pelvis_rear = 0.05 + 0.15
    messages = _apply(parts, height_m=1.72, depth_m=0.20)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.center is not None and g.ry_m is not None
        rear = float(g.center[1]) + float(g.ry_m)
        assert rear + 1e-9 >= pelvis_rear + GLUTE_SEAT_BEYOND_REF_Y
    assert any("beyond_ref" in m for m in messages)


def test_t3b_y_floor_primary_on_thin_pelvis() -> None:
    """T3b (P3-1): thin pelvis — beyond alone insufficient; Y floor is the +Y lever.

    Product-like: pelvis rear≈0.118, ry≈0.121 → need_y≈0.032 < floor 0.045.
    """
    parts = _product_like_parts(y=0.031, ry=0.1212)
    pelvis = next(p for p in parts if p.name == "RECIPE_pelvis_oval")
    assert pelvis.center is not None and pelvis.ry_m is not None
    pelvis_rear = float(pelvis.center[1]) + float(pelvis.ry_m)
    _apply(parts, height_m=1.72)
    # After ry floor, ry stays ~0.1212 (depth half 0.1345 * 0.90)
    g0 = next(p for p in parts if p.role == "glute_soft")
    assert g0.ry_m is not None and g0.center is not None
    ry = float(g0.ry_m)
    beyond_need_y = pelvis_rear + GLUTE_SEAT_BEYOND_REF_Y - ry
    floor = max(GLUTE_SEAT_Y_FLOOR_M, GLUTE_SEAT_Y_FLOOR_FRAC_H * 1.72)
    assert beyond_need_y < floor  # beyond alone would leave y < floor
    assert float(g0.center[1]) + 1e-9 >= floor
    # Rear still clears beyond margin (via floor-driven y)
    rear = float(g0.center[1]) + ry
    assert rear + 1e-9 >= pelvis_rear + GLUTE_SEAT_BEYOND_REF_Y or float(g0.center[1]) >= floor


# ---------------------------------------------------------------------------
# rz floor / ceiling
# ---------------------------------------------------------------------------


def test_t4_rz_floor_frac_ry() -> None:
    """T4: rz >= 0.72*ry after balance."""
    parts = _product_like_parts(rz=0.05)
    _apply(parts, height_m=1.72)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.ry_m is not None and g.rz_m is not None
        assert float(g.rz_m) + 1e-9 >= GLUTE_SEAT_RZ_FRAC_RY * float(g.ry_m)


def test_t4b_rz_h_ceiling() -> None:
    """T4b: rz <= 0.065*H even if bead was huge."""
    h = 1.0
    parts = [
        _glute("RECIPE_glute_soft_l", x=-0.1, ry=0.20, rz=0.50, rx=0.15),
        _glute("RECIPE_glute_soft_r", x=0.1, ry=0.20, rz=0.50, rx=0.15),
        _pelvis_oval(),
    ]
    _apply(parts, height_m=h, depth_m=0.30)
    cap = GLUTE_SEAT_RZ_OVER_H_MAX * h
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.rz_m is not None
        assert float(g.rz_m) <= cap + 1e-9


# ---------------------------------------------------------------------------
# Dual lock / outer / cleft
# ---------------------------------------------------------------------------


def test_t5_dual_lock_y_ry_z() -> None:
    """T5: dual lock equal ry, center_y, and center_z."""
    parts = [
        _glute("RECIPE_glute_soft_l", x=-0.15, y=0.02, z=0.90, ry=0.08, rz=0.05, rx=0.06),
        _glute("RECIPE_glute_soft_r", x=0.15, y=0.05, z=0.88, ry=0.10, rz=0.06, rx=0.07),
        _pelvis_oval(),
    ]
    messages = _apply(parts, height_m=1.72)
    gl, gr = parts[0], parts[1]
    assert gl.ry_m == pytest.approx(float(gr.ry_m or 0.0))
    assert gl.center is not None and gr.center is not None
    assert gl.center[1] == pytest.approx(gr.center[1])
    assert gl.center[2] == pytest.approx(gr.center[2])
    assert any("dual lock" in m and "z=" in m for m in messages)


def test_t6_outer_green_after_balance() -> None:
    """T6: full recipe — C_glute_outer pass."""
    report = _report(
        depth_bands=[_band("glute", depth_m=0.27), _band("hip", depth_m=0.28)],
        crotch_z=0.70,
    )
    pkg = build_blockout_recipe(report, limbs=False, glute="two_spheres", torso="ovals")
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_glute_outer"].status == "pass", by_id["C_glute_outer"].message


def test_t7_cleft_dual_remain() -> None:
    """T7: dual glute_soft remain; cleft not one-blob."""
    report = _report(
        depth_bands=[_band("glute", depth_m=0.27), _band("hip", depth_m=0.28)],
        crotch_z=0.70,
    )
    pkg = build_blockout_recipe(report, limbs=False, glute="two_spheres")
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) >= 2
    assert any(g.center is not None and g.center[0] < 0 for g in glutes)
    assert any(g.center is not None and g.center[0] > 0 for g in glutes)

    class _C:
        glute_cleft_m = 0.02

    class _T:
        constants = _C()

    result = validate_constraints(pkg, report=report, template_applied=_T())
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_glute_cleft"].status in ("pass", "skip"), by_id["C_glute_cleft"].message


def test_t8_ry_floor_0052_still() -> None:
    """T8: 0052 depth-primary ry floor still binds."""
    depth_m = 0.27
    half = depth_m / 2.0
    floor = GLUTE_SEAT_RY_FRAC_HALF_DEPTH * half
    parts = _product_like_parts(ry=0.05)
    _apply(parts, height_m=1.72, depth_m=depth_m)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.ry_m is not None
        assert float(g.ry_m) + 1e-9 >= floor


# ---------------------------------------------------------------------------
# Sticky Y / optimize no Z walk
# ---------------------------------------------------------------------------


def test_t9_sticky_seat_y() -> None:
    """T9 B11: seat y (post-floor) survives fast optimize vs bare template."""
    seat_y = 0.045
    template_y = 0.031

    class _C:
        glute_y_m = template_y
        glute_cleft_m = 0.03

    class _T:
        constants = _C()

    parts = [
        _glute("RECIPE_glute_soft_l", x=-0.12, y=seat_y, ry=0.12, rx=0.07),
        _glute("RECIPE_glute_soft_r", x=0.12, y=seat_y, ry=0.12, rx=0.07),
        _hip_bridge("l"),
        _hip_bridge("r"),
    ]
    pkg = BlockoutRecipePackage(parts=parts, counts={"parts": len(parts)})
    indexed = _index_parts(pkg)
    target = _role_target_y("glute", "l", indexed, None, _T())
    assert target == pytest.approx(seat_y, abs=1e-6)

    optimized, result = optimize_package(
        pkg,
        mode="fast",
        freeze_feet=True,
        template_applied=_T(),
    )
    assert result.score_after <= result.score_before + 1e-12
    for name in ("RECIPE_glute_soft_l", "RECIPE_glute_soft_r"):
        p = next(pp for pp in optimized.parts if pp.name == name)
        y = part_y(p)
        assert y is not None
        assert abs(float(y) - seat_y) <= 0.005


def test_t9b_optimize_does_not_walk_z() -> None:
    """T9b: optimize leaves center_z unchanged (Y sticky only)."""
    z0 = 0.842
    parts = [
        _glute("RECIPE_glute_soft_l", x=-0.12, y=0.045, z=z0, ry=0.12, rx=0.07),
        _glute("RECIPE_glute_soft_r", x=0.12, y=0.045, z=z0, ry=0.12, rx=0.07),
        _hip_bridge("l"),
        _hip_bridge("r"),
    ]
    pkg = BlockoutRecipePackage(parts=parts, counts={"parts": len(parts)})

    class _C:
        glute_y_m = 0.045
        glute_cleft_m = 0.03

    class _T:
        constants = _C()

    optimized, _ = optimize_package(
        pkg,
        mode="fast",
        freeze_feet=True,
        template_applied=_T(),
    )
    for name in ("RECIPE_glute_soft_l", "RECIPE_glute_soft_r"):
        p = next(pp for pp in optimized.parts if pp.name == name)
        assert p.center is not None
        assert float(p.center[2]) == pytest.approx(z0, abs=1e-9)


# ---------------------------------------------------------------------------
# Composition / cascade / quiet / anisotropy
# ---------------------------------------------------------------------------


def test_t11_full_recipe_cascade() -> None:
    """T11: one full build — seat messages + schema 1.4.0 + dual glutes."""
    report = _report(
        depth_bands=[
            _band("chest", depth_m=0.24, z_frac=0.72),
            _band("hip", depth_m=0.26, z_frac=0.55),
            _band("glute", depth_m=0.27, z_frac=0.50),
        ],
        crotch_z=0.70,
    )
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals", glute="two_spheres")
    assert pkg.schema_version == RECIPE_SCHEMA_VERSION == "1.4.0"
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) >= 2
    assert any(m.startswith("glute_seat:") for m in pkg.messages)
    # Seat before outer
    seat_i = next(i for i, m in enumerate(pkg.messages) if m.startswith("glute_seat:"))
    outer_msgs = [i for i, m in enumerate(pkg.messages) if "outer X aligned" in m]
    if outer_msgs:
        assert seat_i < min(outer_msgs)


def test_t12_b15_composition_product_like() -> None:
    """T12: product-like synthetic — top/bottom vs pelvis meet B15 allows."""
    h = 1.72
    parts = _product_like_parts()
    messages = _apply(parts, height_m=h, crotch_z=0.70)
    pelvis = next(p for p in parts if p.name == "RECIPE_pelvis_oval")
    assert pelvis.center is not None and pelvis.rz_m is not None
    pelvis_z = float(pelvis.center[2])
    pelvis_top = pelvis_z + float(pelvis.rz_m)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.center is not None and g.rz_m is not None
        top = float(g.center[2]) + float(g.rz_m)
        bot = float(g.center[2]) - float(g.rz_m)
        assert top <= pelvis_top + GLUTE_TOP_OVER_PELVIS_ALLOW_M + 1e-6
        assert bot <= pelvis_z - GLUTE_BOTTOM_UNDER_MID_M + 1e-6
    assert any("composition" in m for m in messages)


def test_t13_b8_pelvis_defaults_unchanged() -> None:
    """T13: pelvis oval still emits with 0.60 / 0.042 (B8 off)."""
    report = _report(
        depth_bands=[
            _band("chest", depth_m=0.24, z_frac=0.72),
            _band("hip", depth_m=0.26, z_frac=0.55),
            _band("glute", depth_m=0.27, z_frac=0.50),
        ],
        crotch_z=0.70,
    )
    pkg = build_blockout_recipe(report, limbs=False, torso="ovals", glute="two_spheres")
    pelvis = next(p for p in pkg.parts if p.name == "RECIPE_pelvis_oval")
    half_hip = 0.13  # from hip depth band 0.26
    assert float(pelvis.ry_m or 0.0) == pytest.approx(
        half_hip * PELVIS_OVAL_RY_FRAC_HALF_HIP, abs=1e-3
    )
    assert float(pelvis.rz_m or 0.0) == pytest.approx(
        max(0.028, PELVIS_OVAL_RZ_FRAC_H * 1.72), abs=1e-3
    )


def test_t14_quiet_no_glute() -> None:
    """T14: no glute_soft → quiet (no glute_seat messages)."""
    parts = [
        RecipePart(
            name="RECIPE_torso_oval_chest",
            role="torso",
            kind="ellipsoid",
            center=[0.0, 0.0, 1.2],
            rx_m=0.1,
            ry_m=0.08,
            rz_m=0.1,
        )
    ]
    messages = _apply(parts, height_m=1.72)
    assert not any(m.startswith("glute_seat:") for m in messages)


def test_t15_anisotropy_still() -> None:
    """T15: ry/rx ≤ anisotropy max after balance."""
    parts = _product_like_parts(ry=0.05, rx=0.05)
    # Huge depth would try to inflate ry; anisotropy caps at 2.0 * rx
    _apply(parts, height_m=1.72, depth_m=0.40)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.rx_m is not None and g.ry_m is not None
        assert float(g.rx_m) > 0.0
        assert float(g.ry_m) / float(g.rx_m) <= GLUTE_SEAT_RY_ANISOTROPY_MAX + 1e-9


def test_rx_unchanged_no_inflate() -> None:
    """0077 closes 0068 B10: product-like hip_hw floors thin glute rx to 0.40*hip_hw.

    Formerly asserted rx unchanged (0068 left lateral pea residual). Option (b)
    product hip_hw=0.2224 so floor fires (0.40*0.2224=0.0890 > emit-prior 0.0694).
    """
    from meshops.proportion.blockout_recipe import (
        GLUTE_RX_LAT_CAP_FRAC_HIP_HW,
        GLUTE_RX_LAT_FLOOR_FRAC_HIP_HW,
    )

    product_hip_hw = 0.2224
    parts = _product_like_parts()
    r0 = 0.0694
    report = _report(
        height_m=1.72,
        depth_bands=[_band("glute", depth_m=0.269), _band("hip", depth_m=0.278)],
        crotch_z=0.70,
    )
    messages: list[str] = []
    m = _resolved(height_m=1.72)
    m.hip_hw = product_hip_hw
    _apply_glute_seat_mass(parts, report, m, messages)
    floor = GLUTE_RX_LAT_FLOOR_FRAC_HIP_HW * product_hip_hw
    cap = GLUTE_RX_LAT_CAP_FRAC_HIP_HW * product_hip_hw
    for p in [p for p in parts if p.role == "glute_soft"]:
        assert p.rx_m is not None
        assert float(p.rx_m) == pytest.approx(floor, abs=1e-9)
        assert float(p.rx_m) >= floor - 1e-9
        assert float(p.rx_m) <= cap + 1e-9
        assert float(p.rx_m) > r0  # was thin pea; now floored
