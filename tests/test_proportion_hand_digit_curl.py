"""Track 0104 — hand digit curl (PIP/DIP/thumb IP toward palm).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / skeleton 1.0.0 / MCP 47 stay. Not mesh/print success.
Does not reopen 0088 r/L, 0084 hang, mitten, knuckles, or 0115 fist.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    RECIPE_SCHEMA_VERSION,
    build_blockout_recipe,
)
from meshops.proportion.constraints import validate_constraints
from meshops.proportion.extremity_recipe import (
    _FINGER_CURL_DIP_DEG,
    _FINGER_CURL_PIP_DEG,
    _FINGER_NAMES,
    _FINGER_R_SCALES_SEG,
    _FINGER_SEG_FRACS_HAND,
    _THUMB_CURL_IP_DEG,
    _digit_seg_dirs,
    _fingertip_y_trusted,
    finger_primary_axis,
)
from meshops.proportion.models import (
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import (
    SKELETON_SCHEMA_VERSION,
    build_blockout_skeleton,
)

_CURL_PREFIX = "hand curl:"


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


def _report(
    lms: dict[str, LandmarkXYZ] | None = None,
    *,
    height_m: float | None = 1.72,
    depth_bands: list[DepthBand] | None = None,
    diameters: list[DiameterMeasure] | None = None,
) -> ProportionReport:
    return ProportionReport(
        schema_version="1.2.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms if lms is not None else {},
        depth_bands=list(depth_bands or []),
        diameters=list(diameters or []),
        quality=QualityFlags(),
    )


def _product_class_report(*, height_m: float = 1.72) -> ProportionReport:
    h = height_m
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=1.72),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.38),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=None, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=None, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=None, z_m=0.90),
        "hand_l": _lm("hand_l", x_m=-0.33, y_m=None, z_m=0.85),
        "hand_r": _lm("hand_r", x_m=0.33, y_m=None, z_m=0.85),
        "fingertip_l": _lm("fingertip_l", x_m=-0.36, y_m=None, z_m=0.72),
        "fingertip_r": _lm("fingertip_r", x_m=0.36, y_m=None, z_m=0.72),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.1303, z_m=1.25),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
        "heel_l": _lm("heel_l", x_m=-0.10, y_m=0.06, z_m=0.02),
        "heel_r": _lm("heel_r", x_m=0.10, y_m=0.06, z_m=0.02),
        "toe_l": _lm("toe_l", x_m=-0.10, y_m=-0.12, z_m=0.02),
        "toe_r": _lm("toe_r", x_m=0.10, y_m=-0.12, z_m=0.02),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.0438),
        _diam("upper_arm_r", half_width_m=0.0438),
        _diam("forearm_l", half_width_m=0.0350),
        _diam("forearm_r", half_width_m=0.0350),
        _diam("thigh_l", half_width_m=0.0613),
        _diam("thigh_r", half_width_m=0.0613),
        _diam("calf_l", half_width_m=0.0438),
        _diam("calf_r", half_width_m=0.0438),
        _diam("ank_foot_l", half_width_m=0.0263),
        _diam("ank_foot_r", half_width_m=0.0263),
    ]
    bands = [
        _band("chest", y_mid=0.0, depth_m=0.2606),
        _band("breast", depth_m=0.18),
        _band("hip", depth_m=0.26),
        _band("glute", depth_m=0.22),
    ]
    return _report(lms, height_m=h, depth_bands=bands, diameters=diams)


def _product_flags(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "limbs": True,
        "torso": "ovals",
        "glute": "two_spheres",
        "nofuse": True,
        "face": True,
        "hair": "short",
        "hands": True,
        "feet": True,
        "fingers": "full",
        "toes": "full",
        "profile": load_anatomy_profile("torso_limb_f_athletic_v1"),
    }
    base.update(overrides)
    return base


def _report_with_extremities(*, height_m: float = 1.72) -> ProportionReport:
    """Hang-path synthetic: plane-class fingertip Y (0084 untrusted)."""
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "hip_l": _lm("hip_l", x_m=-0.14, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.14, y_m=0.0, z_m=0.95),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.08, z_m=1.25),
        "wrist_l": _lm("wrist_l", x_m=-0.45, y_m=0.0, z_m=0.95),
        "wrist_r": _lm("wrist_r", x_m=0.45, y_m=0.0, z_m=0.95),
        "hand_l": _lm("hand_l", x_m=-0.48, y_m=0.0, z_m=0.88),
        "hand_r": _lm("hand_r", x_m=0.48, y_m=0.0, z_m=0.88),
        "fingertip_l": _lm("fingertip_l", x_m=-0.50, y_m=0.0, z_m=0.72),
        "fingertip_r": _lm("fingertip_r", x_m=0.50, y_m=0.0, z_m=0.72),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.02, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.02, z_m=0.08),
        "heel_l": _lm("heel_l", x_m=-0.10, y_m=0.06, z_m=0.02),
        "heel_r": _lm("heel_r", x_m=0.10, y_m=0.06, z_m=0.02),
        "toe_l": _lm("toe_l", x_m=-0.10, y_m=-0.12, z_m=0.02),
        "toe_r": _lm("toe_r", x_m=0.10, y_m=-0.12, z_m=0.02),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.05),
        _diam("upper_arm_r", half_width_m=0.05),
        _diam("thigh_l", half_width_m=0.05),
        _diam("thigh_r", half_width_m=0.05),
        _diam("ank_foot_l", half_width_m=0.035),
        _diam("ank_foot_r", half_width_m=0.035),
    ]
    bands = [
        _band("chest", depth_m=0.24),
        _band("hip", depth_m=0.26),
    ]
    return _report(lms, height_m=height_m, depth_bands=bands, diameters=diams)


def _trusted_tip_report(
    *,
    wrist_y: float = -0.0586,
    tip_y: float = -0.20,
    height_m: float = 1.72,
) -> ProportionReport:
    """Face-forward measured tip (0084 trusted path)."""
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "hip_l": _lm("hip_l", x_m=-0.14, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.14, y_m=0.0, z_m=0.95),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25),
        "wrist_l": _lm("wrist_l", x_m=-0.45, y_m=wrist_y, z_m=0.95),
        "wrist_r": _lm("wrist_r", x_m=0.45, y_m=wrist_y, z_m=0.95),
        "hand_l": _lm("hand_l", x_m=-0.48, y_m=wrist_y, z_m=0.88),
        "hand_r": _lm("hand_r", x_m=0.48, y_m=wrist_y, z_m=0.88),
        "fingertip_l": _lm("fingertip_l", x_m=-0.50, y_m=tip_y, z_m=0.72),
        "fingertip_r": _lm("fingertip_r", x_m=0.50, y_m=tip_y, z_m=0.72),
    }
    return _report(lms, height_m=height_m)


def _seg_length(part: object) -> float:
    p0 = getattr(part, "p0", None)
    p1 = getattr(part, "p1", None)
    assert p0 is not None and p1 is not None
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    dz = float(p1[2]) - float(p0[2])
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _require_radius_m(part: object) -> float:
    radius_m = getattr(part, "radius_m", None)
    assert radius_m is not None
    return float(radius_m)


def _seg_dir(part: object) -> tuple[float, float, float]:
    p0 = getattr(part, "p0", None)
    p1 = getattr(part, "p1", None)
    assert p0 is not None and p1 is not None
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    dz = float(p1[2]) - float(p0[2])
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    assert n > 1e-12
    return (dx / n, dy / n, dz / n)


def _hand_len_from_lms(lms: dict[str, LandmarkXYZ], side: str = "l") -> float:
    w = lms[f"wrist_{side}"]
    t = lms[f"fingertip_{side}"]
    assert w.x_m is not None and w.z_m is not None
    assert t.x_m is not None and t.z_m is not None
    wy = float(w.y_m) if w.y_m is not None else 0.0
    ty = float(t.y_m) if t.y_m is not None else 0.0
    dx = float(t.x_m) - float(w.x_m)
    dy = ty - wy
    dz = float(t.z_m) - float(w.z_m)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _finger(pkg: Any, fname: str, si: int, side: str) -> Any:
    return next(p for p in pkg.parts if p.name == f"RECIPE_finger_{fname}_{si}_{side}")


def _thumb(pkg: Any, si: int, side: str) -> Any:
    return next(p for p in pkg.parts if p.name == f"RECIPE_thumb_soft_{si}_{side}")


def _unit_len(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def test_t0_0104_constants() -> None:
    """T0: PIP 14 / DIP 20 / thumb IP 12 + bands; 0088 r/segs hold."""
    assert _FINGER_CURL_PIP_DEG == 14.0
    assert 10.0 <= _FINGER_CURL_PIP_DEG <= 18.0
    assert 0.0 <= _FINGER_CURL_PIP_DEG < 30.0
    assert _FINGER_CURL_DIP_DEG == 20.0
    assert 16.0 <= _FINGER_CURL_DIP_DEG <= 26.0
    assert 0.0 <= _FINGER_CURL_DIP_DEG < 35.0
    assert _THUMB_CURL_IP_DEG == 12.0
    assert 8.0 <= _THUMB_CURL_IP_DEG <= 16.0
    assert _THUMB_CURL_IP_DEG >= 0.0
    assert _FINGER_R_SCALES_SEG == (1.00, 0.86, 0.72)
    assert _FINGER_SEG_FRACS_HAND == (0.27, 0.18, 0.10)
    assert abs(sum(_FINGER_SEG_FRACS_HAND) - 0.55) < 1e-12


def test_t1_helper_toward_palm() -> None:
    """T1: hang (0,0,-1) + 14 deg -> dir.y < 0; never +Y."""
    d0, d1, d2 = _digit_seg_dirs(
        (0.0, 0.0, -1.0),
        pip_deg=_FINGER_CURL_PIP_DEG,
        dip_deg=_FINGER_CURL_DIP_DEG,
    )
    assert d0[1] <= 1e-9
    assert d1[1] < 0.0
    assert d2[1] < 0.0
    assert d2[1] < d1[1] < d0[1] + 1e-12
    assert d1[1] <= 0.0 and d2[1] <= 0.0


def test_t1b_degenerate_hinge() -> None:
    """T1b: axis parallel to palm normal still yields finite unit dirs (B27)."""
    dirs = _digit_seg_dirs((0.0, -1.0, 0.0), pip_deg=14.0, dip_deg=20.0)
    assert len(dirs) == 3
    for d in dirs:
        assert all(math.isfinite(c) for c in d)
        n = _unit_len(d)
        assert math.isfinite(n)
        assert n == pytest.approx(1.0, abs=1e-6)


def test_t2_not_colinear_hang_middle() -> None:
    """T2: hang path middle dir0 vs dir2 angle > 20 deg."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    cos20 = math.cos(math.radians(20.0))
    for side in ("l", "r"):
        d0 = _seg_dir(_finger(pkg, "middle", 0, side))
        d2 = _seg_dir(_finger(pkg, "middle", 2, side))
        dot = d0[0] * d2[0] + d0[1] * d2[1] + d0[2] * d2[2]
        assert dot < cos20, f"side={side} dot={dot} cos20={cos20}"


def test_t3_flexion_chain_more_neg_y() -> None:
    """T3: hang path tip.y < mid.p1.y < prox.p1.y; segs connected."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        prox = _finger(pkg, "middle", 0, side)
        mid = _finger(pkg, "middle", 1, side)
        dist = _finger(pkg, "middle", 2, side)
        assert prox.p1 is not None and mid.p0 is not None and mid.p1 is not None
        assert dist.p0 is not None and dist.p1 is not None
        assert float(mid.p0[0]) == pytest.approx(float(prox.p1[0]), abs=1e-9)
        assert float(mid.p0[1]) == pytest.approx(float(prox.p1[1]), abs=1e-9)
        assert float(mid.p0[2]) == pytest.approx(float(prox.p1[2]), abs=1e-9)
        assert float(dist.p0[0]) == pytest.approx(float(mid.p1[0]), abs=1e-9)
        assert float(dist.p0[1]) == pytest.approx(float(mid.p1[1]), abs=1e-9)
        assert float(dist.p0[2]) == pytest.approx(float(mid.p1[2]), abs=1e-9)
        assert float(dist.p1[1]) < float(mid.p1[1]) < float(prox.p1[1])


def test_t4_anti_rake() -> None:
    """T4: all four digits + thumb distal p1.y <= palm.cy."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert palm.center is not None
        palm_cy = float(palm.center[1])
        for fname in _FINGER_NAMES:
            for si in range(3):
                part = _finger(pkg, fname, si, side)
                assert part.p1 is not None
                assert float(part.p1[1]) <= palm_cy + 1e-6, (
                    f"{part.name} p1.y={part.p1[1]} palm={palm_cy}"
                )
        thumb_d = _thumb(pkg, 1, side)
        assert thumb_d.p1 is not None
        assert float(thumb_d.p1[1]) <= palm_cy + 1e-6


def test_t5_length_fence() -> None:
    """T5: middle L still segs * hand_len (0088 T2)."""
    report = _report_with_extremities()
    hand_len = _hand_len_from_lms(report.landmarks_xyz)
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        lengths = [_seg_length(_finger(pkg, "middle", si, side)) for si in range(3)]
        assert lengths[0] > lengths[1] > lengths[2]
        for si, frac in enumerate(_FINGER_SEG_FRACS_HAND):
            assert lengths[si] == pytest.approx(frac * hand_len, abs=1e-9)


def test_t6_radius_fence() -> None:
    """T6: middle r2/r0 == 0.72 (0088 T1)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        r0 = _require_radius_m(_finger(pkg, "middle", 0, side))
        r1 = _require_radius_m(_finger(pkg, "middle", 1, side))
        r2 = _require_radius_m(_finger(pkg, "middle", 2, side))
        assert r2 < r1 < r0
        assert r2 / r0 == pytest.approx(_FINGER_R_SCALES_SEG[2], abs=1e-9)


def test_t7_zero_deg_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """T7: PIP=DIP=0 -> dirs == axis; still emits (B25)."""
    import meshops.proportion.extremity_recipe as ext

    monkeypatch.setattr(ext, "_FINGER_CURL_PIP_DEG", 0.0)
    monkeypatch.setattr(ext, "_FINGER_CURL_DIP_DEG", 0.0)
    axis = (0.0, 0.0, -1.0)
    d0, d1, d2 = ext._digit_seg_dirs(axis, pip_deg=0.0, dip_deg=0.0)
    assert d0[0] == pytest.approx(axis[0], abs=1e-9)
    assert d0[1] == pytest.approx(axis[1], abs=1e-9)
    assert d0[2] == pytest.approx(axis[2], abs=1e-9)
    assert d1 == pytest.approx(d0, abs=1e-9)
    assert d2 == pytest.approx(d0, abs=1e-9)
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    assert any(p.name.startswith("RECIPE_finger_middle_") for p in pkg.parts)
    d_mid0 = _seg_dir(_finger(pkg, "middle", 0, "r"))
    d_mid2 = _seg_dir(_finger(pkg, "middle", 2, "r"))
    assert d_mid0[0] == pytest.approx(d_mid2[0], abs=1e-6)
    assert d_mid0[1] == pytest.approx(d_mid2[1], abs=1e-6)
    assert d_mid0[2] == pytest.approx(d_mid2[2], abs=1e-6)


def test_t8_nan_skip_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """T8: non-finite deg treats that joint as identity; no crash (B25)."""
    import meshops.proportion.extremity_recipe as ext

    axis = (0.0, 0.0, -1.0)
    d0, d1, d2 = ext._digit_seg_dirs(axis, pip_deg=float("nan"), dip_deg=20.0)
    assert d1 == pytest.approx(d0, abs=1e-9)
    assert d2 != pytest.approx(d0, abs=1e-3)
    monkeypatch.setattr(ext, "_FINGER_CURL_PIP_DEG", float("nan"))
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    assert any(p.name == "RECIPE_finger_middle_2_r" for p in pkg.parts)


def test_t9_mitten_no_curl_message() -> None:
    """T9: mitten is one ellipsoid; no hand curl line (B26). Hang may appear."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="mitten")
    names = {p.name for p in pkg.parts}
    assert "RECIPE_finger_mitten_l" in names
    assert "RECIPE_finger_mitten_r" in names
    for fname in _FINGER_NAMES:
        for si in range(3):
            assert f"RECIPE_finger_{fname}_{si}_l" not in names
    assert not any(_CURL_PREFIX in m for m in pkg.messages)


def test_t10_curl_message_once_const_driven() -> None:
    """T10: exactly one hand curl line on full path; const-driven (0094)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    curl = [m for m in pkg.messages if _CURL_PREFIX in m]
    assert len(curl) == 1, curl
    line = curl[0]
    assert f"pip={_FINGER_CURL_PIP_DEG:g}" in line
    assert f"dip={_FINGER_CURL_DIP_DEG:g}" in line
    assert f"thumb_ip={_THUMB_CURL_IP_DEG:g}" in line
    import meshops.proportion.extremity_recipe as ext

    src = ext.build_hand_parts.__code__.co_consts
    # Message is formatted from named consts, not a baked 'pip=14' string.
    assert not any(isinstance(c, str) and "pip=14" in c for c in src)


def test_t11_n_parts_mcp_palm() -> None:
    """T11: product-class n_parts 131; MCP 47; C_palm_ellipsoid."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    assert by_id["C_palm_ellipsoid"].status == "pass", by_id["C_palm_ellipsoid"].message
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert SKELETON_SCHEMA_VERSION == "1.0.0"
    assert len(TOOL_NAMES) == 47


def test_t12_four_digits_curl() -> None:
    """T12: index/ring/pinky also curl (same PIP/DIP)."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    cos20 = math.cos(math.radians(20.0))
    for side in ("l", "r"):
        for fname in ("index", "ring", "pinky"):
            d0 = _seg_dir(_finger(pkg, fname, 0, side))
            d2 = _seg_dir(_finger(pkg, fname, 2, side))
            dot = d0[0] * d2[0] + d0[1] * d2[1] + d0[2] * d2[2]
            assert dot < cos20, f"{fname}_{side} dot={dot}"
            dist = _finger(pkg, fname, 2, side)
            prox = _finger(pkg, fname, 0, side)
            assert dist.p1 is not None and prox.p1 is not None
            assert float(dist.p1[1]) < float(prox.p1[1])


def test_t13_thumb_ip_curl() -> None:
    """T13: distal thumb dir != prox; still p1.y <= palm.cy."""
    report = _report_with_extremities()
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    for side in ("l", "r"):
        palm = next(p for p in pkg.parts if p.name == f"RECIPE_palm_{side}")
        assert palm.center is not None
        palm_cy = float(palm.center[1])
        t0 = _thumb(pkg, 0, side)
        t1 = _thumb(pkg, 1, side)
        d0 = _seg_dir(t0)
        d1 = _seg_dir(t1)
        assert d0 != pytest.approx(d1, abs=1e-4)
        assert t1.p1 is not None
        assert float(t1.p1[1]) <= palm_cy + 1e-6
        assert t1.p0 is not None and t0.p1 is not None
        assert float(t1.p0[0]) == pytest.approx(float(t0.p1[0]), abs=1e-9)
        assert float(t1.p0[1]) == pytest.approx(float(t0.p1[1]), abs=1e-9)
        assert float(t1.p0[2]) == pytest.approx(float(t0.p1[2]), abs=1e-9)


def test_t14_trusted_tip_still_curls() -> None:
    """T14: measured face-forward tip curls relative to wrist->tip (keep hang helper)."""
    wrist_y = -0.0586
    tip_y = -0.20
    assert _fingertip_y_trusted(wrist_y, tip_y) is True
    wrist = [-0.45, wrist_y, 0.95]
    tip = [-0.50, tip_y, 0.72]
    axis = finger_primary_axis(wrist, tip, hand_len=0.3)
    raw = (tip[0] - wrist[0], tip[1] - wrist[1], tip[2] - wrist[2])
    n = math.sqrt(raw[0] ** 2 + raw[1] ** 2 + raw[2] ** 2)
    expected = (raw[0] / n, raw[1] / n, raw[2] / n)
    assert axis[0] == pytest.approx(expected[0], abs=1e-6)
    assert axis[1] == pytest.approx(expected[1], abs=1e-6)
    assert axis[2] == pytest.approx(expected[2], abs=1e-6)
    report = _trusted_tip_report(wrist_y=wrist_y, tip_y=tip_y)
    pkg = build_blockout_recipe(report, limbs=False, hands=True, fingers="full")
    d0 = _seg_dir(_finger(pkg, "middle", 0, "l"))
    d2 = _seg_dir(_finger(pkg, "middle", 2, "l"))
    assert d0[0] == pytest.approx(axis[0], abs=1e-5)
    assert d0[1] == pytest.approx(axis[1], abs=1e-5)
    assert d0[2] == pytest.approx(axis[2], abs=1e-5)
    assert d2[1] < d0[1]
    cos20 = math.cos(math.radians(20.0))
    dot = d0[0] * d2[0] + d0[1] * d2[1] + d0[2] * d2[2]
    assert dot < cos20


def test_t15_compact_still_curls() -> None:
    """T15: compact cull still emits fingers; curl still runs."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    names = {p.name for p in pkg.parts}
    assert "RECIPE_finger_middle_0_r" in names
    assert "RECIPE_finger_middle_2_r" in names
    assert any(_CURL_PREFIX in m for m in pkg.messages)
    d0 = _seg_dir(_finger(pkg, "middle", 0, "r"))
    d2 = _seg_dir(_finger(pkg, "middle", 2, "r"))
    cos20 = math.cos(math.radians(20.0))
    dot = d0[0] * d2[0] + d0[1] * d2[1] + d0[2] * d2[2]
    assert dot < cos20
