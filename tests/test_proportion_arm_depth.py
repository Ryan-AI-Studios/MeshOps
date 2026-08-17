"""Track 0051 — Arm Depth Full3d (body-frame Y prior on torso-plane shoulders).

Authoring honesty only (Difficulty §12 / N6). Schema 1.4.0 / skeleton 1.0.0 stay.
"""

from __future__ import annotations

import pytest

from meshops.proportion.blockout_recipe import build_blockout_recipe
from meshops.proportion.models import DepthBand, LandmarkXYZ, ProportionReport, QualityFlags
from meshops.proportion.skeleton import (
    ARM_FORWARD_OF_HALF_DEPTH_FRAC,
    ELBOW_HANG_T,
    _arm_forward_y,
    _chest_half_depth_for_arm_prior,
    _depth_family_for_joint,
    _elbow_hang_y,
    build_blockout_skeleton,
)


def _lm(
    id_: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=id_, x_m=x_m, y_m=y_m, z_m=z_m)


def _band(
    band_id: str,
    *,
    y_mid: float = 0.0,
    depth_m: float | None = None,
    depth_frac: float = 0.06,
) -> DepthBand:
    return DepthBand(
        band_id=band_id,
        depth_px=20.0,
        depth_frac=depth_frac,
        depth_m=depth_m,
        y_front=(y_mid + depth_frac / 2.0),
        y_back=(y_mid - depth_frac / 2.0),
        y_mid=y_mid,
        z_frac=None,
        confidence=0.8,
        sources=["left"],
        orientation_swapped=False,
    )


def _report(
    lms: dict[str, LandmarkXYZ] | None = None,
    *,
    height_m: float | None = 1.72,
    depth_bands: list[DepthBand] | None = None,
) -> ProportionReport:
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms if lms is not None else {},
        depth_bands=list(depth_bands or []),
        quality=QualityFlags(),
    )


def _by_id(pkg):  # type: ignore[no-untyped-def]
    return {j.id: j for j in pkg.joints}


def _arm_lms(
    *,
    chest_y: float = 0.0,
    half_depth: float | None = 0.13,
    shoulder_y: float | None = None,
    with_front: bool = True,
) -> dict[str, LandmarkXYZ]:
    lms: dict[str, LandmarkXYZ] = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=shoulder_y, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=shoulder_y, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.28, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
        "wrist_r": _lm("wrist_r", x_m=0.32, y_m=None, z_m=0.85),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=chest_y, z_m=1.25),
    }
    if with_front and half_depth is not None:
        lms["chest_front"] = _lm("chest_front", x_m=0.0, y_m=chest_y - half_depth, z_m=1.25)
    return lms


# ---------------------------------------------------------------------------
# Helpers unit
# ---------------------------------------------------------------------------


def test_arm_depth__helpers_half_depth_and_prior_math() -> None:
    """Half-depth order: band depth_m/2 -> |front-mid|; prior frac path + clamp."""
    h = 1.72
    lms = {
        "chest_mid": _lm("chest_mid", y_m=0.0),
        "chest_front": _lm("chest_front", y_m=-0.13),
    }
    bands = [_band("chest", depth_m=0.26)]
    assert _chest_half_depth_for_arm_prior(lms, bands) == pytest.approx(0.13)
    # No band depth_m -> |front-mid|
    assert _chest_half_depth_for_arm_prior(lms, []) == pytest.approx(0.13)
    # No evidence -> None (never 0.12*H)
    assert _chest_half_depth_for_arm_prior({}, []) is None

    y = _arm_forward_y(0.0, half_depth=0.13, height_m=h, chest_front_y=-0.13)
    assert y == pytest.approx(-ARM_FORWARD_OF_HALF_DEPTH_FRAC * 0.13)
    # Clamp at front when prior would overshoot
    y_clamp = _arm_forward_y(0.0, half_depth=0.5, height_m=h, chest_front_y=-0.10)
    assert y_clamp == pytest.approx(-0.10)
    # Stature fallback
    y_st = _arm_forward_y(0.0, half_depth=None, height_m=h, chest_front_y=None)
    assert y_st == pytest.approx(-0.05 * h)


# ---------------------------------------------------------------------------
# T0 product-like torso plane + half-depth
# ---------------------------------------------------------------------------


def test_arm_depth__t0_product_like_chest_mid_prior() -> None:
    """T0 (0087): glenoid ~ plane; wrist distal + Y < mid-0.03; elbow hang lerp."""
    h = 1.72
    half = 0.13
    chest_y = 0.0
    lms = _arm_lms(chest_y=chest_y, half_depth=half)
    bands = [_band("chest", depth_m=half * 2.0)]
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=bands))
    j = _by_id(pkg)
    expected_distal = _arm_forward_y(
        chest_y,
        half_depth=half,
        height_m=h,
        chest_front_y=chest_y - half,
    )
    expected_hang = _elbow_hang_y(chest_y, expected_distal, t=ELBOW_HANG_T)
    for jid in ("shoulder_l", "shoulder_r"):
        y_m = j[jid].y_m
        assert y_m is not None, jid
        assert y_m == pytest.approx(chest_y, abs=1e-6), jid
        assert y_m != pytest.approx(expected_distal, abs=1e-6), jid
    for jid in ("wrist_l", "wrist_r"):
        y_m = j[jid].y_m
        assert y_m is not None, jid
        assert y_m == pytest.approx(expected_distal, abs=1e-6), jid
        assert float(y_m) < chest_y - 0.03, jid
    for jid in ("elbow_l", "elbow_r"):
        y_m = j[jid].y_m
        assert y_m is not None, jid
        assert y_m == pytest.approx(expected_hang, abs=1e-6), jid
    assert any("arm forward prior" in m and "distal" in m for m in pkg.messages)
    assert not any("shoulder_l" in m and "arm forward prior" in m for m in pkg.messages)
    assert not any("shoulder_r" in m and "arm forward prior" in m for m in pkg.messages)


# ---------------------------------------------------------------------------
# T1 landmark wins
# ---------------------------------------------------------------------------


def test_arm_depth__t1_direct_shoulder_landmark_preserved() -> None:
    """T1: direct shoulder y_m wins — no prior message / overwrite."""
    h = 1.72
    measured = -0.09
    lms = _arm_lms(chest_y=0.0, half_depth=0.13, shoulder_y=measured)
    bands = [_band("chest", depth_m=0.26)]
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=bands))
    j = _by_id(pkg)
    assert j["shoulder_l"].y_m == pytest.approx(measured)
    assert j["shoulder_r"].y_m == pytest.approx(measured)
    assert not any("arm forward prior" in m for m in pkg.messages)
    # Elbow/wrist inherit measured as (depth) — true off-plane arm Y
    assert any("elbow_l" in m and "inherited" in m and "(depth)" in m for m in pkg.messages)


def test_arm_depth__t1b_plane_class_zero_landmark_still_distal() -> None:
    """T1b (0087): landmark Y=0 plane-class — wrist distal; elbow hang message."""
    h = 1.72
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half, shoulder_y=0.0)
    bands = [_band("chest", depth_m=0.26)]
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=bands))
    j = _by_id(pkg)
    expected = _arm_forward_y(0.0, half_depth=half, height_m=h, chest_front_y=-half)
    hang = _elbow_hang_y(0.0, expected, t=ELBOW_HANG_T)
    assert j["shoulder_l"].y_m == pytest.approx(0.0)
    assert j["wrist_l"].y_m == pytest.approx(expected, abs=1e-6)
    assert j["elbow_l"].y_m == pytest.approx(hang, abs=1e-6)
    assert any("elbow_l" in m and "hang" in m and f"t={ELBOW_HANG_T}" in m for m in pkg.messages)
    assert not any("shoulder_l" in m and "arm forward prior" in m for m in pkg.messages)


# ---------------------------------------------------------------------------
# T5b mixed-null recipe
# ---------------------------------------------------------------------------


def test_arm_depth__t5b_mixed_null_arm_keeps_front_plane() -> None:
    """T5b: one arm endpoint has Y → mean/front_plane; measured end preserved."""
    from meshops.proportion.models import DiameterMeasure

    def _diam(band_id: str) -> DiameterMeasure:
        return DiameterMeasure(
            band_id=band_id,
            view="front",
            width_px=40.0,
            width_eucl_px=40.0,
            theta_deg=90.0,
            width_frac=0.1,
            width_m=0.1,
            half_width_m=0.05,
            mid_x_px=100.0,
            mid_y_px=200.0,
        )

    measured_sh_y = -0.04
    lms = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=measured_sh_y, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.40),
        "elbow_r": _lm("elbow_r", x_m=0.28, y_m=0.0, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=0.0, z_m=0.85),
        "wrist_r": _lm("wrist_r", x_m=0.32, y_m=0.0, z_m=0.85),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.0, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.0, z_m=0.50),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
    }
    report = ProportionReport(
        schema_version="1.1.0",
        height_m=1.72,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=[
            _diam("upper_arm_l"),
            _diam("upper_arm_r"),
            _diam("forearm_l"),
            _diam("forearm_r"),
            _diam("thigh_l"),
            _diam("thigh_r"),
            _diam("calf_l"),
            _diam("calf_r"),
            _diam("bust"),
            _diam("waist"),
            _diam("neck"),
        ],
        depth_bands=[_band("chest", depth_m=0.26)],
        quality=QualityFlags(),
    )
    pkg = build_blockout_recipe(report, limbs=True)
    ua_l = next(p for p in pkg.parts if p.name == "RECIPE_limb_upper_arm_l")
    assert ua_l.placement == "front_plane"
    assert ua_l.p0 is not None and ua_l.p1 is not None
    # Mean of available: only shoulder Y → both ends at measured_sh_y
    assert ua_l.p0[1] == pytest.approx(measured_sh_y)
    assert ua_l.p1[1] == pytest.approx(measured_sh_y)
    assert any("upper_arm_l: y_m null — front_plane limb capsule" in m for m in pkg.messages)
    assert not any("upper_arm_l: y_m null — arm forward prior" in m for m in pkg.messages)


# ---------------------------------------------------------------------------
# T6 deltoid / T6b bridge
# ---------------------------------------------------------------------------


def test_arm_depth__t6_deltoid_matches_skeleton_shoulder() -> None:
    """T6: deltoid center Y matches skeleton shoulder when present."""
    from meshops.proportion.models import DiameterMeasure

    def _diam(band_id: str, hw: float = 0.05) -> DiameterMeasure:
        return DiameterMeasure(
            band_id=band_id,
            view="front",
            width_px=40.0,
            width_eucl_px=40.0,
            theta_deg=90.0,
            width_frac=0.1,
            width_m=hw * 2,
            half_width_m=hw,
            mid_x_px=100.0,
            mid_y_px=200.0,
        )

    h = 1.72
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half)
    lms["hip_l"] = _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90)
    lms["hip_r"] = _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90)
    lms["chin"] = _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50)
    lms["cranial_vertex"] = _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68)
    report = ProportionReport(
        schema_version="1.1.0",
        height_m=h,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=[
            _diam("upper_arm_l", 0.05),
            _diam("upper_arm_r", 0.05),
            _diam("forearm_l"),
            _diam("forearm_r"),
            _diam("thigh_l"),
            _diam("thigh_r"),
            _diam("calf_l"),
            _diam("calf_r"),
            _diam("bust", 0.16),
            _diam("waist", 0.13),
            _diam("neck", 0.05),
        ],
        depth_bands=[_band("chest", depth_m=half * 2)],
        quality=QualityFlags(),
    )
    skel = build_blockout_skeleton(report)
    sh_y = next(j.y_m for j in skel.joints if j.id == "shoulder_l")
    assert sh_y is not None
    pkg = build_blockout_recipe(report, limbs=True, skeleton=skel)
    del_l = next(p for p in pkg.parts if p.name == "RECIPE_deltoid_soft_l")
    assert del_l.center is not None
    assert del_l.center[1] == pytest.approx(sh_y, abs=1e-6)
    assert del_l.placement == "full3d"


def test_arm_depth__t6b_bridge_p1_matches_skeleton_shoulder() -> None:
    """T6b: shoulder bridge p1 Y co-moves with skeleton shoulder (B13)."""
    from meshops.proportion.models import DiameterMeasure

    def _diam(band_id: str, hw: float = 0.05) -> DiameterMeasure:
        return DiameterMeasure(
            band_id=band_id,
            view="front",
            width_px=40.0,
            width_eucl_px=40.0,
            theta_deg=90.0,
            width_frac=0.1,
            width_m=hw * 2,
            half_width_m=hw,
            mid_x_px=100.0,
            mid_y_px=200.0,
        )

    h = 1.72
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half)
    lms["hip_l"] = _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90)
    lms["hip_r"] = _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90)
    lms["chin"] = _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50)
    lms["cranial_vertex"] = _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68)
    report = ProportionReport(
        schema_version="1.1.0",
        height_m=h,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=[
            _diam("upper_arm_l", 0.05),
            _diam("upper_arm_r", 0.05),
            _diam("forearm_l"),
            _diam("forearm_r"),
            _diam("thigh_l"),
            _diam("thigh_r"),
            _diam("calf_l"),
            _diam("calf_r"),
            _diam("bust", 0.16),
            _diam("waist", 0.13),
            _diam("neck", 0.05),
        ],
        depth_bands=[_band("chest", depth_m=half * 2)],
        quality=QualityFlags(),
    )
    skel = build_blockout_skeleton(report)
    sh_y = next(j.y_m for j in skel.joints if j.id == "shoulder_l")
    assert sh_y is not None
    pkg = build_blockout_recipe(report, limbs=True, skeleton=skel)
    br_l = next(p for p in pkg.parts if p.name == "RECIPE_shoulder_bridge_l")
    assert br_l.p1 is not None
    assert br_l.p1[1] == pytest.approx(sh_y, abs=1e-6)


# ---------------------------------------------------------------------------
# T7 clamp / T8 invent stable / T9 thigh / T11 no stale depth
# ---------------------------------------------------------------------------


def test_arm_depth__t7_clamp_not_past_chest_front() -> None:
    """T7 (0083): glenoid stays at plane; distal prior still clamps at chest_front."""
    h = 1.72
    chest_y = 0.0
    # Huge half-depth would push distal past front without clamp
    front_y = -0.05
    lms = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=chest_y, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=front_y, z_m=1.25),
    }
    bands = [_band("chest", depth_m=0.50)]  # half=0.25 → raw prior -0.1125 < front
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=bands))
    j = _by_id(pkg)
    assert j["shoulder_l"].y_m is not None
    assert j["shoulder_l"].y_m == pytest.approx(chest_y, abs=1e-6)
    assert float(j["shoulder_l"].y_m) >= front_y - 1e-9
    assert j["wrist_l"].y_m == pytest.approx(front_y)
    assert j["elbow_l"].y_m == pytest.approx(_elbow_hang_y(chest_y, front_y, t=ELBOW_HANG_T))


def test_arm_depth__t8_invent_no_half_depth_value_stable() -> None:
    """T8 (0083): invent glenoid = plane 0; distal still stature prior (-0.05)*H."""
    h = 1.72
    lms = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
    }
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=[]))
    j = _by_id(pkg)
    assert j["shoulder_l"].y_m == pytest.approx(0.0)
    assert j["wrist_l"].y_m == pytest.approx(-0.05 * h)
    assert j["elbow_l"].y_m == pytest.approx(_elbow_hang_y(0.0, -0.05 * h, t=ELBOW_HANG_T))
    assert any("arm forward prior" in m and "invent" in m and "distal" in m for m in pkg.messages)
    assert not any("shoulder_l" in m and "arm forward prior" in m for m in pkg.messages)
    assert not any("inherited" in m and "(depth)" in m for m in pkg.messages)


def test_arm_depth__t9_thigh_null_still_front_plane() -> None:
    """T9: thigh null Y still front_plane (arm-only prior)."""
    from meshops.proportion.models import DiameterMeasure

    def _diam(band_id: str) -> DiameterMeasure:
        return DiameterMeasure(
            band_id=band_id,
            view="front",
            width_px=40.0,
            width_eucl_px=40.0,
            theta_deg=90.0,
            width_frac=0.1,
            width_m=0.1,
            half_width_m=0.05,
            mid_x_px=100.0,
            mid_y_px=200.0,
        )

    lms = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.28, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
        "wrist_r": _lm("wrist_r", x_m=0.32, y_m=None, z_m=0.85),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=None, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=None, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=None, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=None, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=None, z_m=0.08),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
    }
    report = ProportionReport(
        schema_version="1.1.0",
        height_m=1.72,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=[
            _diam("upper_arm_l"),
            _diam("upper_arm_r"),
            _diam("forearm_l"),
            _diam("forearm_r"),
            _diam("thigh_l"),
            _diam("thigh_r"),
            _diam("calf_l"),
            _diam("calf_r"),
            _diam("bust"),
            _diam("waist"),
            _diam("neck"),
        ],
        depth_bands=[_band("chest", depth_m=0.26)],
        quality=QualityFlags(),
    )
    pkg = build_blockout_recipe(report, limbs=True)
    thigh = next(p for p in pkg.parts if p.name == "RECIPE_limb_thigh_l")
    assert thigh.placement == "front_plane"
    ua = next(p for p in pkg.parts if p.name == "RECIPE_limb_upper_arm_l")
    assert ua.placement == "full3d"


def test_arm_depth__t11_no_stale_lone_chest_mid_depth() -> None:
    """T11 (0083 / AI2 F4): distal claims prior; shoulder must not."""
    h = 1.72
    lms = _arm_lms(chest_y=0.0, half_depth=0.13)
    bands = [_band("chest", depth_m=0.26)]
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=bands))
    assert not any("shoulder_l" in m and "arm forward prior" in m for m in pkg.messages), (
        pkg.messages
    )
    assert any("elbow_l" in m and "hang" in m and f"t={ELBOW_HANG_T}" in m for m in pkg.messages), (
        pkg.messages
    )
    assert any(
        "wrist_l" in m and "arm forward prior" in m and "distal" in m for m in pkg.messages
    ), pkg.messages


def test_arm_depth__t10_family_none_still() -> None:
    """T10: 0037 T1 fence - elbow/wrist/hand family stays None."""
    for jid in (
        "elbow_l",
        "elbow_r",
        "wrist_l",
        "wrist_r",
        "hand_l",
        "hand_r",
    ):
        assert _depth_family_for_joint(jid) is None
