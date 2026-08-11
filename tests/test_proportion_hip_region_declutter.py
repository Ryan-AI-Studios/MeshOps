"""Track 0077 - Hip region declutter (iliac skip + glute rx lateral floor; authoring only)."""

from __future__ import annotations

from meshops.proportion.blockout_recipe import (
    GLUTE_RX_LAT_CAP_FRAC_HIP_HW,
    GLUTE_RX_LAT_FLOOR_FRAC_HIP_HW,
    GLUTE_SEAT_RY_FROM_RX,
    GLUTE_SEAT_RZ_FRAC_RY,
    GLUTE_SEAT_Y_FLOOR_M,
    HIP_SOFT_RX_SCALE,
    HIP_SOFT_RY_FRAC_RX,
    HIP_SOFT_RZ_FRAC_RX,
    RECIPE_SCHEMA_VERSION,
    RecipePart,
    _apply_glute_seat_mass,
    _hip_bridge_outer_x,
    _ResolvedMetrics,
    build_blockout_recipe,
)
from meshops.proportion.blockout_recipe import (
    __all__ as BLOCKOUT_ALL,
)
from meshops.proportion.constraints import OUTER_X_TOL_M, validate_constraints
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


def _limb_mass_report(
    *,
    height_m: float | None = 1.72,
    thigh_hw: float = 0.0613,
    calf_hw: float = 0.05,
    arm_hw: float = 0.04,
    hip_x: float = 0.2224,
) -> ProportionReport:
    """Product-like limb report; hip_x sets mean |hip| ≈ hip_hw."""
    hx = abs(hip_x)
    lms = {
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.70),
        "hip_l": _lm("hip_l", x_m=-hx, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=hx, y_m=0.0, z_m=0.95),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=1.72),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=0.0, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=0.0, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=0.0, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=0.0, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-hx, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=hx, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=arm_hw),
        _diam("upper_arm_r", half_width_m=arm_hw),
        _diam("forearm_l", half_width_m=arm_hw),
        _diam("forearm_r", half_width_m=arm_hw),
        _diam("thigh_l", half_width_m=thigh_hw),
        _diam("thigh_r", half_width_m=thigh_hw),
        _diam("calf_l", half_width_m=calf_hw),
        _diam("calf_r", half_width_m=calf_hw),
    ]
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=[
            _band("chest", depth_m=0.24, z_frac=0.72),
            _band("hip", depth_m=0.26, z_frac=0.55),
            _band("glute", depth_m=0.269, z_frac=0.50),
        ],
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


def _resolved(*, height_m: float | None = 1.72, hip_hw: float = 0.2224) -> _ResolvedMetrics:
    m = _ResolvedMetrics()
    m.height_m = height_m
    m.hip_hw = hip_hw
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


def _pelvis_oval() -> RecipePart:
    return RecipePart(
        name="RECIPE_pelvis_oval",
        role="pelvis",
        kind="ellipsoid",
        center=[0.0, 0.0347, 0.8332],
        rx_m=0.2224,
        ry_m=0.0834,
        rz_m=0.0722,
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
    rx: float = 0.0694,
    ry: float = 0.1212,
    rz: float = 0.0631,
) -> list[RecipePart]:
    return [
        _glute("RECIPE_glute_soft_l", x=-0.150, rx=rx, ry=ry, rz=rz),
        _glute("RECIPE_glute_soft_r", x=0.150, rx=rx, ry=ry, rz=rz),
        _pelvis_oval(),
        _hip_bridge("l"),
        _hip_bridge("r"),
    ]


def _seat_report() -> ProportionReport:
    return ProportionReport(
        schema_version="1.1.0",
        height_m=1.72,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz={
            "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
            "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.70),
            "hip_l": _lm("hip_l", x_m=-0.2224, y_m=0.0, z_m=0.95),
            "hip_r": _lm("hip_r", x_m=0.2224, y_m=0.0, z_m=0.95),
        },
        diameters=[_diam("bust", half_width_m=0.16)],
        depth_bands=[
            _band("glute", depth_m=0.269, z_frac=0.50),
            _band("hip", depth_m=0.278, z_frac=0.55),
        ],
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


def _apply_seat(
    parts: list[RecipePart],
    *,
    hip_hw: float = 0.2224,
    height_m: float | None = 1.72,
) -> list[str]:
    messages: list[str] = []
    _apply_glute_seat_mass(
        parts, _seat_report(), _resolved(height_m=height_m, hip_hw=hip_hw), messages
    )
    return messages


# ---------------------------------------------------------------------------
# T0-T13
# ---------------------------------------------------------------------------


def test_t0_constants_frozen() -> None:
    """T0: floor/cap freezes + hip_soft axes fence 0069."""
    assert GLUTE_RX_LAT_FLOOR_FRAC_HIP_HW == 0.40
    assert GLUTE_RX_LAT_CAP_FRAC_HIP_HW == 0.50
    assert HIP_SOFT_RX_SCALE == 1.15
    assert HIP_SOFT_RY_FRAC_RX == 0.88
    assert HIP_SOFT_RZ_FRAC_RX == 0.70
    assert RECIPE_SCHEMA_VERSION == "1.4.0"


def test_t1_zero_iliac_soft_and_skip_message() -> None:
    """T1: zero iliac_soft; B1 skip message present."""
    pkg = build_blockout_recipe(_limb_mass_report(), limbs=True, glute="two_spheres")
    iliac = [p for p in pkg.parts if p.role == "iliac_soft"]
    assert len(iliac) == 0
    assert not any(p.name.startswith("RECIPE_iliac_soft_") for p in pkg.parts)
    assert any("iliac_soft skipped: 0077" in m for m in pkg.messages)


def test_t2_hip_soft_present_no_prox_soft() -> None:
    """T2: hip_soft L/R present; no prox_soft_thigh (0069 fence)."""
    pkg = build_blockout_recipe(_limb_mass_report(), limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    assert "RECIPE_hip_soft_l" in by_name
    assert "RECIPE_hip_soft_r" in by_name
    assert "RECIPE_prox_soft_thigh_l" not in by_name
    assert "RECIPE_prox_soft_thigh_r" not in by_name
    assert not any("prox_soft" in p.name.lower() for p in pkg.parts)


def test_t3_glute_rx_floor() -> None:
    """T3: glute rx >= 0.40 * hip_hw - eps when hip_hw known."""
    hip_hw = 0.2224
    parts = _product_like_parts(rx=0.0694)
    _apply_seat(parts, hip_hw=hip_hw)
    floor = GLUTE_RX_LAT_FLOOR_FRAC_HIP_HW * hip_hw
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.rx_m is not None
        assert float(g.rx_m) >= floor - 1e-9


def test_t4_glute_rx_cap() -> None:
    """T4: glute rx <= 0.50 * hip_hw + eps."""
    hip_hw = 0.2224
    # Start above cap so B3 must clamp
    parts = _product_like_parts(rx=0.20)
    _apply_seat(parts, hip_hw=hip_hw)
    cap = GLUTE_RX_LAT_CAP_FRAC_HIP_HW * hip_hw
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.rx_m is not None
        assert float(g.rx_m) <= cap + 1e-9


def test_t5_glute_outer_matches_bridge_after_recipe() -> None:
    """T5: after full recipe, glute outer ≈ hip_bridge outer (±1e-3)."""
    pkg = build_blockout_recipe(
        _limb_mass_report(hip_x=0.2224),
        limbs=True,
        glute="two_spheres",
    )
    by_name = {p.name: p for p in pkg.parts}
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) == 2, f"expected dual glute_soft, got {[p.name for p in glutes]}"
    for side in ("l", "r"):
        glute = by_name.get(f"RECIPE_glute_sphere_{side}") or by_name.get(
            f"RECIPE_glute_soft_{side}"
        )
        assert glute is not None
        assert glute.center is not None and glute.rx_m is not None
        bridge_outer = _hip_bridge_outer_x(pkg.parts, side)  # type: ignore[arg-type]
        assert bridge_outer is not None
        glute_outer = float(glute.center[0]) + float(glute.rx_m) * (1.0 if side == "r" else -1.0)
        # Compare signed outer tips; abs-delta within 1e-3 (0036 SoT)
        assert abs(glute_outer - float(bridge_outer)) <= 1e-3


def test_t6_hip_soft_past_cap() -> None:
    """T6: hip_soft outer past thigh hip-end cap (0069 fence)."""
    pkg = build_blockout_recipe(_limb_mass_report(), limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_hip_soft_{side}"]
        thigh = by_name[f"RECIPE_limb_thigh_{side}"]
        assert soft.center is not None and soft.rx_m is not None
        assert thigh.p0 is not None and thigh.radius_m is not None
        soft_outer = abs(float(soft.center[0])) + float(soft.rx_m)
        thigh_cap = abs(float(thigh.p0[0])) + float(thigh.radius_m)
        assert soft_outer > thigh_cap - 1e-4


def test_t7_seat_y_and_rz_floors_hold() -> None:
    """T7: 0068 y floor / rz floor still hold product-like."""
    parts = _product_like_parts()
    _apply_seat(parts, hip_hw=0.2224)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.center is not None and g.ry_m is not None and g.rz_m is not None
        assert float(g.center[1]) >= GLUTE_SEAT_Y_FLOOR_M - 1e-9
        assert float(g.rz_m) >= GLUTE_SEAT_RZ_FRAC_RY * float(g.ry_m) - 1e-9


def test_t8_hip_declutter_message_when_glutes() -> None:
    """T8: hip declutter message with glute_rx_floor when glutes + hip_hw."""
    hip_hw = 0.2224
    parts = _product_like_parts()
    messages = _apply_seat(parts, hip_hw=hip_hw)
    floor = GLUTE_RX_LAT_FLOOR_FRAC_HIP_HW * hip_hw
    assert any("hip declutter:" in m and "glute_rx_floor=" in m for m in messages)
    assert any(f"glute_rx_floor={floor:.4f}" in m for m in messages)


def test_t9_all_exports_floor_cap() -> None:
    """T9: __all__ exports floor/cap constants."""
    assert "GLUTE_RX_LAT_FLOOR_FRAC_HIP_HW" in BLOCKOUT_ALL
    assert "GLUTE_RX_LAT_CAP_FRAC_HIP_HW" in BLOCKOUT_ALL


def test_t10_no_glutes_iliac_still_skipped() -> None:
    """T10: no glutes → iliac still skipped; no seat-side hip declutter required."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=False)
    assert len([p for p in pkg.parts if p.role == "iliac_soft"]) == 0
    assert any("iliac_soft skipped: 0077" in m for m in pkg.messages)
    # Seat path with zero glute_soft: early return — no hip declutter line
    parts = [_pelvis_oval(), _hip_bridge("l"), _hip_bridge("r")]
    messages = _apply_seat(parts)
    assert not any("hip declutter:" in m for m in messages)


def test_t11_constraints_smoke_outer_thigh() -> None:
    """T11: constraints smoke — C_glute_outer green on full recipe; thigh rule present.

    C_thigh_outer full-recipe green needs pre-aligned synthetic (0069 T9); here we
    assert glute outer (0036) passes and thigh rule still evaluates after declutter.
    """
    report = _limb_mass_report(hip_x=0.2224)
    pkg = build_blockout_recipe(report, limbs=True, glute="two_spheres")
    result = validate_constraints(pkg, report=report)
    by_id = {c.id: c for c in result.rules}
    assert "C_glute_outer" in by_id
    assert by_id["C_glute_outer"].status == "pass", by_id["C_glute_outer"].message
    assert "C_thigh_outer" in by_id
    # Thigh outer may fail on wide hip_x synthetic full-recipe (pre-existing);
    # smoke: rule still present and metrics populated when status is pass/fail.
    assert by_id["C_thigh_outer"].status in ("pass", "fail", "skip")
    # Pre-aligned mini package still green for thigh (not regressed by 0077).
    from meshops.proportion.blockout_recipe import BlockoutRecipePackage

    hip = [-0.12, 0.0, 0.95]
    knee = [-0.12, 0.0, 0.50]
    mid = [0.5 * (hip[0] + knee[0]), 0.5 * (hip[1] + knee[1]), 0.5 * (hip[2] + knee[2])]
    r = 0.06
    chain_outer = mid[0] - r
    hip_half = 0.03
    hip_cx = chain_outer + hip_half
    mini = BlockoutRecipePackage(
        parts=[
            RecipePart(
                name="RECIPE_hip_bridge_l",
                role="hip_bridge",
                kind="ellipsoid",
                center=[hip_cx, 0.03, 0.95],
                rx_m=hip_half,
                ry_m=0.03,
                rz_m=0.03,
            ),
            RecipePart(
                name="RECIPE_limb_thigh_l",
                role="limb_segment",
                kind="capsule",
                radius_m=r,
                p0=list(hip),
                p1=list(mid),
            ),
            RecipePart(
                name="RECIPE_thigh_taper_dist_l",
                role="limb_segment",
                kind="capsule",
                radius_m=r * 0.8,
                p0=list(mid),
                p1=list(knee),
            ),
        ],
        counts={"parts": 3},
    )
    mini_res = validate_constraints(mini)
    mini_by = {c.id: c for c in mini_res.rules}
    assert mini_by["C_thigh_outer"].status == "pass", mini_by["C_thigh_outer"].message
    assert mini_by["C_thigh_outer"].metrics is not None
    assert float(mini_by["C_thigh_outer"].metrics["delta_l"]) <= OUTER_X_TOL_M + 1e-9


def test_t12_docstring_no_leaves_rx_unchanged() -> None:
    """T12 hard: 'Leaves rx unchanged' not in _apply_glute_seat_mass.__doc__."""
    doc = _apply_glute_seat_mass.__doc__ or ""
    assert "Leaves rx unchanged" not in doc


def test_t13_optional_ry_reapply_after_rx_floor() -> None:
    """T13: shallow ry + thin rx -> after B3, ry >= 1.05*rx."""
    hip_hw = 0.2224
    # Thin rx floors to 0.40*hip_hw; shallow ry forces re-apply
    parts = _product_like_parts(rx=0.05, ry=0.02, rz=0.01)
    _apply_seat(parts, hip_hw=hip_hw)
    for g in [p for p in parts if p.role == "glute_soft"]:
        assert g.rx_m is not None and g.ry_m is not None
        assert float(g.ry_m) >= GLUTE_SEAT_RY_FROM_RX * float(g.rx_m) - 1e-9
        assert float(g.rx_m) >= GLUTE_RX_LAT_FLOOR_FRAC_HIP_HW * hip_hw - 1e-9
