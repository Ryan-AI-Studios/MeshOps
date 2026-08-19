"""Track 0099 — Skeleton hang message honesty (silent hang-fallback Y).

Authoring honesty only (Difficulty §12 / N6 / SKELETON_HONESTY).
Not mesh/print success. Schema 1.0.0 / recipe 1.4.0 / MCP 46 stay.

Helpers copied from test_proportion_arm_elbow_hang.py (per-module; do not import).
"""

from __future__ import annotations

from pathlib import Path

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import RECIPE_SCHEMA_VERSION
from meshops.proportion.models import DepthBand, LandmarkXYZ, ProportionReport, QualityFlags
from meshops.proportion.skeleton import (
    ARM_FORWARD_OF_HALF_DEPTH_FRAC,
    ELBOW_HANG_T,
    GLENOID_ANTERIOR_FRAC,
    SKELETON_SCHEMA_VERSION,
    _arm_forward_y,
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
        schema_version="1.2.0",
        height_m=height_m,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms if lms is not None else {},
        depth_bands=list(depth_bands or []),
        diameters=[],
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
    elbow_xyz: bool = True,
) -> dict[str, LandmarkXYZ]:
    lms: dict[str, LandmarkXYZ] = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=shoulder_y, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=shoulder_y, z_m=1.40),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
        "wrist_r": _lm("wrist_r", x_m=0.32, y_m=None, z_m=0.85),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=chest_y, z_m=1.25),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
    }
    if elbow_xyz:
        lms["elbow_l"] = _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10)
        lms["elbow_r"] = _lm("elbow_r", x_m=0.28, y_m=None, z_m=1.10)
    if with_front and half_depth is not None:
        lms["chest_front"] = _lm("chest_front", x_m=0.0, y_m=chest_y - half_depth, z_m=1.25)
    return lms


def test_t0_const_freezes() -> None:
    """T0: hang T 0.50 in band; ARM_FORWARD 0.45; glenoid frac 0. Invert: T stays 0.50."""
    assert ELBOW_HANG_T == 0.50
    assert 0.45 <= ELBOW_HANG_T <= 0.55
    assert ARM_FORWARD_OF_HALF_DEPTH_FRAC == 0.45
    assert GLENOID_ANTERIOR_FRAC == 0.0


def test_t1_missing_wrist_hang_no_distal_on_elbow() -> None:
    """T1: missing wrist Y+Z + height_m=None: hang on elbow; no arm-forward on elbow_*."""
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half)
    lms["wrist_l"] = _lm("wrist_l", x_m=-0.32, y_m=None, z_m=None)
    lms["wrist_r"] = _lm("wrist_r", x_m=0.32, y_m=None, z_m=None)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=None, depth_bands=[_band("chest", depth_m=0.26)])
    )
    j = _by_id(pkg)
    expected_distal = _arm_forward_y(0.0, half_depth=half, height_m=None, chest_front_y=-half)
    expected_elbow = _elbow_hang_y(0.0, expected_distal)
    assert j["elbow_l"].y_m == expected_elbow
    assert j["elbow_r"].y_m == expected_elbow
    assert j["wrist_l"].y_m is None
    assert j["wrist_r"].y_m is None
    assert any("elbow_l" in m and "hang" in m and f"t={ELBOW_HANG_T}" in m for m in pkg.messages), (
        pkg.messages
    )
    assert any("elbow_r" in m and "hang" in m and f"t={ELBOW_HANG_T}" in m for m in pkg.messages), (
        pkg.messages
    )
    assert not any("elbow_l" in m and "arm forward prior" in m for m in pkg.messages), pkg.messages
    assert not any("elbow_r" in m and "arm forward prior" in m for m in pkg.messages), pkg.messages


def test_t2_product_like_single_source() -> None:
    """T2: finite wrist z: wrist distal + elbow hang; no arm-forward on elbow_* or shoulder_*."""
    h = 1.72
    lms = _arm_lms(chest_y=0.0, half_depth=0.13)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    assert any("elbow_l" in m and "hang" in m and f"t={ELBOW_HANG_T}" in m for m in pkg.messages), (
        pkg.messages
    )
    assert any(
        "wrist_l" in m and "arm forward prior" in m and "distal" in m for m in pkg.messages
    ), pkg.messages
    assert not any("elbow_l" in m and "arm forward prior" in m for m in pkg.messages)
    assert not any("elbow_r" in m and "arm forward prior" in m for m in pkg.messages)
    assert not any("shoulder_l" in m and "arm forward prior" in m for m in pkg.messages)
    assert not any("shoulder_r" in m and "arm forward prior" in m for m in pkg.messages)


def test_t3_y_hold_source_estimated() -> None:
    """T3: T2 Y hold; both source=estimated (do not claim measured)."""
    h = 1.72
    half = 0.13
    lms = _arm_lms(chest_y=0.0, half_depth=half)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    j = _by_id(pkg)
    expected_distal = _arm_forward_y(0.0, half_depth=half, height_m=h, chest_front_y=-half)
    assert j["wrist_l"].y_m == expected_distal
    assert j["wrist_r"].y_m == expected_distal
    assert j["elbow_l"].y_m == _elbow_hang_y(0.0, expected_distal)
    assert j["elbow_r"].y_m == _elbow_hang_y(0.0, expected_distal)
    assert j["elbow_l"].source == "estimated"
    assert j["wrist_l"].source == "estimated"
    assert j["elbow_r"].source == "estimated"
    assert j["wrist_r"].source == "estimated"


def test_t4_wrist_path_still_messages() -> None:
    """T4: T2 wrist_l message still contains arm forward prior and distal."""
    h = 1.72
    lms = _arm_lms(chest_y=0.0, half_depth=0.13)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    wr_msgs = [m for m in pkg.messages if "wrist_l" in m]
    assert any("arm forward prior" in m and "distal" in m for m in wr_msgs), wr_msgs


def test_t5_off_plane_inherit_no_hang() -> None:
    """T5: measured shoulder Y off plane -> no hang line on elbow."""
    h = 1.72
    measured = -0.09
    lms = _arm_lms(chest_y=0.0, half_depth=0.13, shoulder_y=measured)
    pkg = build_blockout_skeleton(
        _report(lms, height_m=h, depth_bands=[_band("chest", depth_m=0.26)])
    )
    j = _by_id(pkg)
    assert j["shoulder_l"].y_m == measured
    assert j["elbow_l"].y_m == measured
    assert j["wrist_l"].y_m == measured
    assert not any("elbow_l" in m and "elbow hang" in m for m in pkg.messages)
    assert not any("elbow_r" in m and "elbow hang" in m for m in pkg.messages)


def test_t6_schema_catalog() -> None:
    """T6: skeleton 1.0.0; recipe 1.4.0; MCP catalog 47."""
    assert SKELETON_SCHEMA_VERSION == "1.0.0"
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert len(TOOL_NAMES) == 47


def test_t7_all_hold_no_new_name() -> None:
    """T7: ELBOW_HANG_T still exported; no silent helper in skeleton.__all__."""
    from meshops.proportion import skeleton as skel_mod

    assert "ELBOW_HANG_T" in skel_mod.__all__
    assert "_distal_prior_y" not in skel_mod.__all__
    assert not any("silent" in name.lower() for name in skel_mod.__all__)


def test_t8_source_grep_no_distal_on_el_id() -> None:
    """T8: hang fallback must not call _apply_distal_prior(el_id); wrist wr_id stays."""
    src = Path("src/meshops/proportion/skeleton.py").read_text(encoding="utf-8")
    assert "_apply_distal_prior(el_id)" not in src
    assert "_apply_distal_prior(wr_id)" in src
