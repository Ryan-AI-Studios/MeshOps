"""Track 0127 — leg/ankle/foot landmark ids / template blanks / fuse XYZ (offline).

Authoring measurement only (LEG_FOOT_COMPARE_HONESTY / CAPTURE_HONESTY).
Not mesh or print success. Schema report stay 1.2.0. MCP 51 at ship.
"""

from __future__ import annotations

from meshops.proportion.assist import KNOWN_LANDMARK_IDS
from meshops.proportion.fuse import DEPTH_PAIRS, fuse_xyz
from meshops.proportion.models import Landmark2D, ViewLandmarks
from meshops.proportion.template import (
    _BACK_LANDMARK_KEYS,
    _FRONT_LANDMARK_KEYS,
    _LEFT_LANDMARK_KEYS,
)

_LEG_FOOT_FRONT_IDS: tuple[str, ...] = (
    "malleolus_med_l",
    "malleolus_med_r",
    "malleolus_lat_l",
    "malleolus_lat_r",
    "gastroc_med_l",
    "gastroc_med_r",
    "gastroc_lat_l",
    "gastroc_lat_r",
    "ball_l",
    "ball_r",
)

_LEG_FOOT_LEFT_IDS: tuple[str, ...] = (
    "gastroc_med_l",
    "arch_apex_l",
    "achilles_l",
    "malleolus_med_l",
)

_LEG_FOOT_BACK_IDS: tuple[str, ...] = (
    "gastroc_med_l",
    "gastroc_med_r",
    "gastroc_lat_l",
    "gastroc_lat_r",
    "achilles_l",
    "achilles_r",
)

_LEG_FOOT_NEW_IDS: tuple[str, ...] = (
    "gastroc_med_l",
    "gastroc_med_r",
    "gastroc_lat_l",
    "gastroc_lat_r",
    "malleolus_med_l",
    "malleolus_med_r",
    "malleolus_lat_l",
    "malleolus_lat_r",
    "arch_apex_l",
    "arch_apex_r",
    "achilles_l",
    "achilles_r",
    "ball_l",
    "ball_r",
)


def _lm2(lid: str, x_px: float, y_px: float, *, w: int = 100, h: int = 200) -> Landmark2D:
    return Landmark2D(
        id=lid,
        x_px=x_px,
        y_px=y_px,
        x_frac=x_px / w,
        y_frac=y_px / h,
        method="assist",
        confidence=1.0,
    )


def test_a1_leg_foot_ids_in_known() -> None:
    """A1: frozen v1 form-read ids are known (membership in, not ==)."""
    for lid in _LEG_FOOT_NEW_IDS:
        assert lid in KNOWN_LANDMARK_IDS


def test_a2_template_blanks_front_left_back() -> None:
    """A2: front blanks include v1 ids; left subset; back gastroc/achilles."""
    for lid in _LEG_FOOT_FRONT_IDS:
        assert lid in _FRONT_LANDMARK_KEYS
    for lid in _LEG_FOOT_LEFT_IDS:
        assert lid in _LEFT_LANDMARK_KEYS
    for lid in _LEG_FOOT_BACK_IDS:
        assert lid in _BACK_LANDMARK_KEYS


def test_a3_missing_id_no_invent_y() -> None:
    """A3: missing/unknown id skipped; front-only never invents Y."""
    front = ViewLandmarks(
        view="front",
        width_px=100,
        height_px=200,
        facing_direction="camera_front",
        landmarks={
            "cranial_vertex": _lm2("cranial_vertex", 50.0, 10.0),
            "sole": _lm2("sole", 50.0, 190.0),
            "chin": _lm2("chin", 50.0, 40.0),
            "gastroc_med_l": _lm2("gastroc_med_l", 35.0, 130.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front}, height_m=1.72)
    assert "gastroc_med_l" in out
    assert out["gastroc_med_l"].x_m is not None
    assert out["gastroc_med_l"].z_m is not None
    assert out["gastroc_med_l"].y is None
    assert out["gastroc_med_l"].y_m is None
    assert "arch_apex_l" not in out
    assert "achilles_l" not in out


def test_a4_leg_foot_have_ids_hold() -> None:
    """A4: 0013/0024 knee/ankle/calf/heel/toe/foot DEPTH stay known."""
    assert "knee_l" in KNOWN_LANDMARK_IDS
    assert "ankle_r" in KNOWN_LANDMARK_IDS
    assert "calf_front" in KNOWN_LANDMARK_IDS
    assert "heel_l" in KNOWN_LANDMARK_IDS
    assert "toe_r" in KNOWN_LANDMARK_IDS
    assert "foot_back" in KNOWN_LANDMARK_IDS
    assert "knee_l" in _FRONT_LANDMARK_KEYS
    assert "ankle_r" in _FRONT_LANDMARK_KEYS
    assert "calf_front" in _LEFT_LANDMARK_KEYS
    assert "heel_l" in _FRONT_LANDMARK_KEYS
    assert "toe_r" in _FRONT_LANDMARK_KEYS
    assert "foot_back" in _LEFT_LANDMARK_KEYS


def test_a5_no_gastroc_front_depth_pair() -> None:
    """A5: do not add gastroc_front/back DEPTH_PAIRS."""
    flat = {lid for triple in DEPTH_PAIRS for lid in triple}
    assert "gastroc_front" not in flat
    assert "gastroc_back" not in flat
    assert "gastroc_front" not in KNOWN_LANDMARK_IDS


def test_a6_asis_sternum_eye_still_known() -> None:
    """A6: 0126 asis_l, 0125 sternum_mid, 0124 eye_l still in KNOWN."""
    assert "asis_l" in KNOWN_LANDMARK_IDS
    assert "sternum_mid" in KNOWN_LANDMARK_IDS
    assert "eye_l" in KNOWN_LANDMARK_IDS
    assert "eye_l" in _FRONT_LANDMARK_KEYS


def test_a_back_view_sets_gastroc_xz() -> None:
    """Back view fills X/Z for gastroc; never invents Y (B8)."""
    front = ViewLandmarks(
        view="front",
        width_px=100,
        height_px=200,
        facing_direction="camera_front",
        landmarks={
            "cranial_vertex": _lm2("cranial_vertex", 50.0, 10.0),
            "sole": _lm2("sole", 50.0, 190.0),
            "chin": _lm2("chin", 50.0, 40.0),
        },
    )
    back = ViewLandmarks(
        view="back",
        width_px=100,
        height_px=200,
        facing_direction="camera_back",
        landmarks={
            "cranial_vertex": _lm2("cranial_vertex", 50.0, 10.0),
            "sole": _lm2("sole", 50.0, 190.0),
            "midline_x": _lm2("midline_x", 50.0, 100.0),
            "gastroc_med_l": _lm2("gastroc_med_l", 70.0, 140.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "back": back}, height_m=1.72)
    assert "gastroc_med_l" in out
    assert out["gastroc_med_l"].x is not None
    assert out["gastroc_med_l"].z is not None
    assert out["gastroc_med_l"].x_m is not None
    assert out["gastroc_med_l"].z_m is not None
    assert out["gastroc_med_l"].y is None
    assert out["gastroc_med_l"].y_m is None
    # camera_back inverts X so larger image-x (right of back photo) → MeshOps -X
    assert out["gastroc_med_l"].x < 0.0


def test_a_left_view_sets_leg_foot_y() -> None:
    """Left-view same-id overlay sets Y; never invents from front-only."""
    front = ViewLandmarks(
        view="front",
        width_px=100,
        height_px=200,
        facing_direction="camera_front",
        landmarks={
            "cranial_vertex": _lm2("cranial_vertex", 50.0, 10.0),
            "sole": _lm2("sole", 50.0, 190.0),
            "chin": _lm2("chin", 50.0, 40.0),
            "gastroc_med_l": _lm2("gastroc_med_l", 35.0, 130.0),
        },
    )
    left = ViewLandmarks(
        view="left",
        width_px=100,
        height_px=200,
        facing_direction="camera_left",
        landmarks={
            "calf_front": _lm2("calf_front", 70.0, 130.0),
            "calf_back": _lm2("calf_back", 30.0, 130.0),
            "gastroc_med_l": _lm2("gastroc_med_l", 25.0, 130.0),
            "arch_apex_l": _lm2("arch_apex_l", 40.0, 185.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "left": left}, height_m=1.72)
    assert out["gastroc_med_l"].y is not None
    assert out["gastroc_med_l"].y_m is not None
    assert out["arch_apex_l"].y is not None
    assert out["arch_apex_l"].y_m is not None
