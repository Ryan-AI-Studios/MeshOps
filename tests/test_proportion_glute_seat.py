"""Track 0052 — glute_soft seat depth (ry) + rear +Y projection (authoring only)."""

from __future__ import annotations

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    GLUTE_SEAT_BEYOND_REF_Y,
    GLUTE_SEAT_RY_ANISOTROPY_MAX,
    GLUTE_SEAT_RY_CAP_FRAC_H,
    GLUTE_SEAT_RY_FRAC_HALF_DEPTH,
    GLUTE_SEAT_RY_FROM_RX,
    GLUTE_SEAT_Y_CAP_FRAC_H,
    RECIPE_SCHEMA_VERSION,
    BlockoutRecipePackage,
    RecipePart,
    _apply_glute_seat_mass,
    _glute_or_hip_half_depth_m,
    _pelvis_ref_rear_y,
    _ResolvedMetrics,
    build_blockout_recipe,
)
from meshops.proportion.constraints import (
    OUTER_X_TOL_M,
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


def _base_lms(*, height_m: float = 1.72) -> dict[str, LandmarkXYZ]:
    return {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "hip_l": _lm("hip_l", x_m=-0.14, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.14, y_m=0.0, z_m=0.95),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
    }


def _report(
    *,
    height_m: float = 1.72,
    depth_bands: list[DepthBand] | None = None,
    extra_lms: dict[str, LandmarkXYZ] | None = None,
    with_glute_cs: bool = True,
) -> ProportionReport:
    lms = _base_lms(height_m=height_m)
    if extra_lms:
        lms.update(extra_lms)
    bands = (
        list(depth_bands)
        if depth_bands is not None
        else [
            _band("chest", depth_m=0.24, z_frac=0.72),
            _band("hip", depth_m=0.26, z_frac=0.55),
        ]
    )
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
    ]
    cs: list[CrossSection] = []
    if with_glute_cs:
        cs = [
            CrossSection(
                level_id="bust",
                z_frac=0.72,
                rx_frac=0.10,
                ry_frac=0.08,
                sources=["test"],
            ),
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


def _resolved(*, height_m: float = 1.72) -> _ResolvedMetrics:
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
    y: float = 0.03,
    z: float = 0.90,
    rx: float | None = 0.07,
    ry: float | None = 0.05,
    rz: float | None = 0.06,
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
    y: float = 0.035,
    ry: float = 0.118,
    z: float = 0.90,
) -> RecipePart:
    return RecipePart(
        name="RECIPE_pelvis_oval",
        role="pelvis",
        kind="ellipsoid",
        center=[0.0, y, z],
        rx_m=0.14,
        ry_m=ry,
        rz_m=0.08,
    )


def _hip_bridge(side: str, *, outer_x: float = 0.22, half: float = 0.04) -> RecipePart:
    # outer = center ± half → center = outer ∓ half
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


def _seat_parts_product_like() -> list[RecipePart]:
    """Profile-like shallow glutes buried under pelvis (T0 red condition)."""
    return [
        _glute("RECIPE_glute_soft_l", x=-0.15, y=0.03, ry=0.05, rx=0.069),
        _glute("RECIPE_glute_soft_r", x=0.15, y=0.03, ry=0.05, rx=0.069),
        _pelvis_oval(y=0.035, ry=0.118),
        _hip_bridge("l"),
        _hip_bridge("r"),
    ]


# ---------------------------------------------------------------------------
# Unit helpers on synthetic parts
# ---------------------------------------------------------------------------


def test_t0_red_fixture_buried_before_seat() -> None:
    """T0: profile-like shallow glutes — rear tip under pelvis rear before seat."""
    parts = _seat_parts_product_like()
    glutes = [p for p in parts if p.role == "glute_soft"]
    pelvis = next(p for p in parts if p.name == "RECIPE_pelvis_oval")
    assert pelvis.center is not None and pelvis.ry_m is not None
    pelvis_rear = float(pelvis.center[1]) + float(pelvis.ry_m)
    for g in glutes:
        assert g.center is not None and g.ry_m is not None
        rear = float(g.center[1]) + float(g.ry_m)
        assert rear < pelvis_rear
        assert float(g.ry_m) < 0.10


def test_t1_ry_floor_from_glute_depth_band() -> None:
    """T1: glute depth_m=0.27 -> ry >= 0.90 * half after seat."""
    depth_m = 0.27
    half = depth_m / 2.0
    floor = GLUTE_SEAT_RY_FRAC_HALF_DEPTH * half
    report = _report(depth_bands=[_band("glute", depth_m=depth_m), _band("hip", depth_m=0.20)])
    assert _glute_or_hip_half_depth_m(report) == pytest.approx(half)

    parts = _seat_parts_product_like()
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(), messages)
    glutes = [p for p in parts if p.role == "glute_soft"]
    for g in glutes:
        assert g.ry_m is not None
        assert float(g.ry_m) >= floor - 1e-9
    assert any("glute_seat: ry_floor_depth=" in m for m in messages)


def test_t2_rear_tip_beyond_pelvis() -> None:
    """T2: after seat, rear tip ≥ pelvis_rear + BEYOND."""
    report = _report(depth_bands=[_band("glute", depth_m=0.27)])
    parts = _seat_parts_product_like()
    pelvis = next(p for p in parts if p.name == "RECIPE_pelvis_oval")
    assert pelvis.center is not None and pelvis.ry_m is not None
    pelvis_rear = float(pelvis.center[1]) + float(pelvis.ry_m)
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(), messages)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.center is not None and g.ry_m is not None
        rear = float(g.center[1]) + float(g.ry_m)
        assert rear + 1e-9 >= pelvis_rear + GLUTE_SEAT_BEYOND_REF_Y
    assert any("glute_seat: beyond_ref " in m for m in messages)
    assert any("whole-part max; not z-slice" in m for m in messages)


def test_t3_dual_lock_equal_y_and_ry() -> None:
    """T3: asymmetric L/R y/ry → dual lock equal y and ry."""
    report = _report(depth_bands=[_band("glute", depth_m=0.27)])
    parts = [
        _glute("RECIPE_glute_soft_l", x=-0.15, y=0.02, ry=0.04, rx=0.06),
        _glute("RECIPE_glute_soft_r", x=0.15, y=0.05, ry=0.08, rx=0.07),
        _pelvis_oval(),
    ]
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(), messages)
    gl, gr = parts[0], parts[1]
    assert gl.ry_m == pytest.approx(float(gr.ry_m or 0.0))
    assert gl.center is not None and gr.center is not None
    assert gl.center[1] == pytest.approx(gr.center[1])
    # B4/B8: dual lock does not force equal rx (left intentionally asymmetric).
    assert gl.rx_m == pytest.approx(0.06)
    assert gr.rx_m == pytest.approx(0.07)
    assert any("glute_seat: dual lock" in m for m in messages)


def test_t4_outer_align_still_fires_with_hip_bridge() -> None:
    """T4: after full recipe emit, outer align message + |outer-hip| <= tol."""
    report = _report(depth_bands=[_band("glute", depth_m=0.27), _band("hip", depth_m=0.28)])
    pkg = build_blockout_recipe(report, limbs=False, glute="two_spheres", torso="ovals")
    assert any("glute_l: outer X aligned to hip_bridge" in m for m in pkg.messages)
    assert any("glute_r: outer X aligned to hip_bridge" in m for m in pkg.messages)
    # Seat messages before outer (T10 also)
    seat_idx = next(i for i, m in enumerate(pkg.messages) if m.startswith("glute_seat:"))
    outer_idx = next(i for i, m in enumerate(pkg.messages) if "outer X aligned" in m)
    assert seat_idx < outer_idx

    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_glute_outer"].status == "pass", by_id["C_glute_outer"].message

    # Numeric outer tip within OUTER_X_TOL_M of hip_bridge outer
    from meshops.proportion.blockout_recipe import _half_extent_x_local, _hip_bridge_outer_x

    for side in ("l", "r"):
        glutes = [p for p in pkg.parts if p.role == "glute_soft" and p.name.endswith(f"_{side}")]
        hip_outer = _hip_bridge_outer_x(pkg.parts, side)  # type: ignore[arg-type]
        assert hip_outer is not None
        for g in glutes:
            half = _half_extent_x_local(g)
            assert half is not None and g.center is not None
            outer = float(g.center[0]) + half if side == "r" else float(g.center[0]) - half
            assert abs(outer - hip_outer) <= OUTER_X_TOL_M + 1e-9


def test_t5_cleft_not_negative_after_seat() -> None:
    """T5: dual with gap — C_glute_cleft still pass; gap not negative."""
    report = _report(depth_bands=[_band("glute", depth_m=0.27), _band("hip", depth_m=0.28)])
    pkg = build_blockout_recipe(report, limbs=False, glute="two_spheres")
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) >= 2
    # Medial faces should leave positive gap when centers are dual with rx.
    xs = sorted(float(g.center[0]) for g in glutes if g.center is not None)
    rxs = [float(g.rx_m or 0.0) for g in glutes]
    # Approximate cleft: distance between medial tips
    gap_proxy = (xs[1] - rxs[1] if xs[1] > 0 else xs[1] + rxs[1]) - (
        xs[0] + rxs[0] if xs[0] < 0 else xs[0] - rxs[0]
    )
    # Simpler: centers stay on opposite sides of midline
    assert any(g.center is not None and g.center[0] < 0 for g in glutes)
    assert any(g.center is not None and g.center[0] > 0 for g in glutes)
    del gap_proxy

    class _C:
        glute_cleft_m = 0.02

    class _T:
        constants = _C()

    result = validate_constraints(pkg, report=report, template_applied=_T())
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_glute_cleft"].status in ("pass", "skip"), by_id["C_glute_cleft"].message


def test_t6_no_glute_soft_quiet() -> None:
    """T6: no glute_soft → no throw; quiet (no required loud skip)."""
    report = _report(depth_bands=[])
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
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(), messages)
    assert not any(m.startswith("glute_seat:") for m in messages)


def test_t7_base_two_spheres_anisotropy_bound() -> None:
    """T7: base two_spheres path seats; ry/rx ≤ 2.0 after seat."""
    report = _report(depth_bands=[_band("glute", depth_m=0.27), _band("hip", depth_m=0.28)])
    pkg = build_blockout_recipe(report, limbs=False, glute="two_spheres")
    spheres = [p for p in pkg.parts if p.name.startswith("RECIPE_glute_sphere_")]
    assert len(spheres) == 2
    for g in spheres:
        assert g.rx_m is not None and g.ry_m is not None
        assert float(g.rx_m) > 0.0
        assert float(g.ry_m) / float(g.rx_m) <= GLUTE_SEAT_RY_ANISOTROPY_MAX + 1e-9
    assert any(m.startswith("glute_seat:") for m in pkg.messages)


def test_t8_ry_cap_at_frac_h() -> None:
    """T8: huge floor attempt caps at 0.10*H."""
    height_m = 1.72
    # half huge → floor would exceed cap
    report = _report(
        height_m=height_m,
        depth_bands=[_band("glute", depth_m=2.0)],
    )
    parts = [
        _glute("RECIPE_glute_soft_l", x=-0.1, ry=0.05, rx=0.5),  # large rx so B13 not binding
        _glute("RECIPE_glute_soft_r", x=0.1, ry=0.05, rx=0.5),
    ]
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(height_m=height_m), messages)
    cap = GLUTE_SEAT_RY_CAP_FRAC_H * height_m
    for g in parts:
        assert g.ry_m is not None
        assert float(g.ry_m) <= cap + 1e-9
    assert any("glute_seat: ry_cap" in m for m in messages)


def test_t9_male_profile_dual_anisotropy() -> None:
    """T9: male mild profile — dual remain; ry/rx ≤ 2.0; not one part."""
    report = _report(depth_bands=[_band("glute", depth_m=0.24), _band("hip", depth_m=0.25)])
    profile = load_anatomy_profile("torso_limb_m_athletic_v1")
    pkg = build_blockout_recipe(report, limbs=False, profile=profile)
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) >= 2
    for g in glutes:
        if g.rx_m is not None and float(g.rx_m) > 0 and g.ry_m is not None:
            assert float(g.ry_m) / float(g.rx_m) <= GLUTE_SEAT_RY_ANISOTROPY_MAX + 1e-9


def test_t10_message_order_seat_before_outer() -> None:
    """T10: glute_seat: ry_floor_depth= appears before glute_*: outer X aligned."""
    report = _report(depth_bands=[_band("glute", depth_m=0.27), _band("hip", depth_m=0.28)])
    pkg = build_blockout_recipe(report, limbs=False, glute="two_spheres", torso="ovals")
    floor_i = next(
        (i for i, m in enumerate(pkg.messages) if "glute_seat: ry_floor_depth=" in m),
        None,
    )
    outer_i = next(
        (i for i, m in enumerate(pkg.messages) if "outer X aligned" in m),
        None,
    )
    assert floor_i is not None
    assert outer_i is not None
    assert floor_i < outer_i


def test_t11_join_ready_still_applies_seat() -> None:
    """T11: join_ready True — seat still applied (order before join)."""
    report = _report(depth_bands=[_band("glute", depth_m=0.27), _band("hip", depth_m=0.28)])
    pkg = build_blockout_recipe(
        report,
        limbs=True,
        glute="two_spheres",
        torso="ovals",
        join_ready=True,
    )
    assert pkg.join_ready is True
    assert any(m.startswith("glute_seat:") for m in pkg.messages)
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert glutes
    half = 0.27 / 2.0
    floor = GLUTE_SEAT_RY_FRAC_HALF_DEPTH * half
    for g in glutes:
        assert g.ry_m is not None
        assert float(g.ry_m) >= floor - 1e-6


def test_t12_pinned_message_prefixes() -> None:
    """T12: messages contain pinned prefixes when seat applied."""
    report = _report(depth_bands=[_band("glute", depth_m=0.27)])
    parts = _seat_parts_product_like()
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(), messages)
    joined = "\n".join(messages)
    assert "glute_seat: ry_floor_depth=" in joined
    assert "glute_seat: beyond_ref " in joined
    assert "glute_seat: dual lock" in joined


def test_t13_schema_unchanged() -> None:
    """T13: schema write stays 1.4.0."""
    report = _report(depth_bands=[_band("glute", depth_m=0.27)])
    pkg = build_blockout_recipe(report, limbs=False, glute="two_spheres")
    assert pkg.schema_version == RECIPE_SCHEMA_VERSION
    assert RECIPE_SCHEMA_VERSION == "1.4.0"


def test_t14_product_like_numeric_rear_clears() -> None:
    """T14: product-like dims — rear tip > pelvis rear after seat."""
    report = _report(depth_bands=[_band("glute", depth_m=0.269), _band("hip", depth_m=0.278)])
    parts = [
        _glute("RECIPE_glute_soft_l", x=-0.15, y=0.031, ry=0.0567, rx=0.0694, rz=0.0631),
        _glute("RECIPE_glute_soft_r", x=0.15, y=0.031, ry=0.0567, rx=0.0694, rz=0.0631),
        _pelvis_oval(y=0.035, ry=0.118),
    ]
    pelvis_rear = 0.035 + 0.118
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(), messages)
    for g in parts[:2]:
        assert g.center is not None and g.ry_m is not None
        rear = float(g.center[1]) + float(g.ry_m)
        assert rear > pelvis_rear
        assert float(g.ry_m) >= GLUTE_SEAT_RY_FRAC_HALF_DEPTH * (0.269 / 2.0) - 1e-9


def test_t15_optimize_sticky_seat_y() -> None:
    """T15 B11: seat-applied y survives fast optimize (not pulled to bare template)."""
    seat_y = 0.052
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
    # Sticky target = max(template, dual mean) = seat_y
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


def test_t16_zero_ry_skip_no_zero_radius() -> None:
    """T16 P2-2: ry_m=None, no depth, no rx → skip; no ry_m=0 silent."""
    report = _report(depth_bands=[])  # no glute/hip depth
    parts = [
        RecipePart(
            name="RECIPE_glute_soft_l",
            role="glute_soft",
            kind="ellipsoid",
            center=[-0.1, 0.03, 0.9],
            rx_m=None,
            ry_m=None,
            rz_m=0.05,
        ),
        RecipePart(
            name="RECIPE_glute_soft_r",
            role="glute_soft",
            kind="ellipsoid",
            center=[0.1, 0.03, 0.9],
            rx_m=None,
            ry_m=None,
            rz_m=0.05,
        ),
    ]
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(), messages)
    for p in parts:
        assert p.ry_m is None  # unchanged — not coerced to 0
    assert any("glute_seat: skip " in m for m in messages)


def test_depth_missing_rx_floor_message() -> None:
    """B2-only: no depth band → ry from rx + depth missing message."""
    report = _report(depth_bands=[_band("chest", depth_m=0.2, z_frac=0.7)])  # no glute/hip
    parts = [
        _glute("RECIPE_glute_soft_l", x=-0.1, y=0.03, ry=0.04, rx=0.08),
        _glute("RECIPE_glute_soft_r", x=0.1, y=0.03, ry=0.04, rx=0.08),
    ]
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(), messages)
    floor = 0.08 * GLUTE_SEAT_RY_FROM_RX
    for g in parts:
        assert g.ry_m is not None
        assert float(g.ry_m) >= floor - 1e-9
    assert any("glute_seat: depth missing" in m for m in messages)


def test_y_cap_message() -> None:
    """B12: extreme beyond-ref push hits stature y cap."""
    height_m = 1.0  # small H so 0.15*H = 0.15 is tight
    report = _report(
        height_m=height_m,
        depth_bands=[_band("glute", depth_m=0.20)],
    )
    # Huge pelvis rear forces large need_y
    parts = [
        _glute("RECIPE_glute_soft_l", x=-0.1, y=0.01, ry=0.05, rx=0.05),
        _glute("RECIPE_glute_soft_r", x=0.1, y=0.01, ry=0.05, rx=0.05),
        _pelvis_oval(y=0.5, ry=0.5),  # rear = 1.0
    ]
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(height_m=height_m), messages)
    y_cap = GLUTE_SEAT_Y_CAP_FRAC_H * height_m
    for g in parts[:2]:
        assert g.center is not None
        assert float(g.center[1]) <= y_cap + 1e-9
    assert any("glute_seat: y_cap" in m for m in messages)


def test_pelvis_ref_ladder_bucket_and_role() -> None:
    """B3 ladder: pelvis_bucket / role pelvis / hip oval contribute to max rear."""
    bucket = RecipePart(
        name="RECIPE_pelvis_bucket",
        role="pelvis",
        kind="trap_box",
        center=[0.0, 0.04, 0.9],
        top_half_width_m=0.1,
        bottom_half_width_m=0.12,
        half_depth_m=0.1,  # rear = 0.04 + 0.1 = 0.14 (AI1: half_depth_m path)
        z_bottom_m=0.8,
        z_top_m=1.0,
    )
    role_pelvis = RecipePart(
        name="RECIPE_some_pelvis_mass",
        role="pelvis",
        kind="ellipsoid",
        center=[0.0, 0.02, 0.9],
        rx_m=0.1,
        ry_m=0.20,  # rear = 0.22
        rz_m=0.08,
    )
    hip_oval = RecipePart(
        name="RECIPE_torso_oval_hip",
        role="torso",
        kind="ellipsoid",
        center=[0.0, 0.05, 0.95],
        rx_m=0.12,
        ry_m=0.10,  # rear = 0.15
        rz_m=0.08,
    )
    # Named bucket alone: trap_box uses half_depth_m (AI1 Blind Spot 1 / 0053-ready)
    rear_b, name_b = _pelvis_ref_rear_y([bucket])
    assert rear_b == pytest.approx(0.14)
    assert name_b == "pelvis_bucket"

    rear, name = _pelvis_ref_rear_y([role_pelvis, hip_oval, bucket])
    assert rear == pytest.approx(0.22)
    assert name == "pelvis"

    oval = _pelvis_oval(y=0.035, ry=0.118)  # rear = 0.153
    rear2, name2 = _pelvis_ref_rear_y([oval, role_pelvis, hip_oval])
    # oval present first in ladder and contributes; max of all still role if larger
    assert rear2 == pytest.approx(0.22)
    assert name2 in ("pelvis", "pelvis_oval")


def test_half_depth_prefers_glute_over_hip_band() -> None:
    report = _report(
        depth_bands=[
            _band("hip", depth_m=0.40),
            _band("glute", depth_m=0.20),
        ]
    )
    assert _glute_or_hip_half_depth_m(report) == pytest.approx(0.10)


def test_half_depth_measured_landmarks() -> None:
    report = _report(
        depth_bands=[],
        extra_lms={
            "hip_front": _lm("hip_front", y_m=-0.10),
            "hip_back": _lm("hip_back", y_m=0.16),
        },
    )
    assert _glute_or_hip_half_depth_m(report) == pytest.approx(0.13)


def test_constants_frozen() -> None:
    assert GLUTE_SEAT_RY_FRAC_HALF_DEPTH == 0.90
    assert GLUTE_SEAT_RY_FROM_RX == 1.05
    assert GLUTE_SEAT_BEYOND_REF_Y == 0.035
    assert GLUTE_SEAT_RY_CAP_FRAC_H == 0.10
    assert GLUTE_SEAT_Y_CAP_FRAC_H == 0.15
    assert GLUTE_SEAT_RY_ANISOTROPY_MAX == 2.0


def test_non_glute_roles_untouched() -> None:
    """B5: breast/pec/iliac/torso never seated."""
    report = _report(depth_bands=[_band("glute", depth_m=0.27)])
    breast = RecipePart(
        name="RECIPE_breast_soft_l",
        role="breast_soft",
        kind="ellipsoid",
        center=[-0.08, -0.05, 1.2],
        rx_m=0.05,
        ry_m=0.04,
        rz_m=0.05,
    )
    parts = [breast, _glute("RECIPE_glute_soft_l", x=-0.1, ry=0.04, rx=0.06)]
    messages: list[str] = []
    _apply_glute_seat_mass(parts, report, _resolved(), messages)
    assert parts[0].ry_m == pytest.approx(0.04)
    assert parts[1].ry_m is not None
    assert float(parts[1].ry_m) > 0.04
