"""Track 0129 — arm/hand landmark ids / template blanks / fuse XYZ (offline).

Authoring measurement only (ARM_HAND_COMPARE_HONESTY / CAPTURE_HONESTY).
Not mesh or print success. Schema report stay 1.2.0. MCP 53 at ship.
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

_ARM_HAND_FRONT_IDS: tuple[str, ...] = (
    "humeral_head_l",
    "humeral_head_r",
    "bi_belly_l",
    "bi_belly_r",
    "tri_belly_l",
    "tri_belly_r",
    "olecranon_l",
    "olecranon_r",
    "fa_belly_l",
    "fa_belly_r",
    "palm_center_l",
    "palm_center_r",
    "thumb_cmc_l",
    "thumb_cmc_r",
    "thumb_tip_l",
    "thumb_tip_r",
    "mcp_index_l",
    "mcp_index_r",
    "mcp_pinky_l",
    "mcp_pinky_r",
)

_ARM_HAND_LEFT_IDS: tuple[str, ...] = (
    "humeral_head_l",
    "bi_belly_l",
    "tri_belly_l",
    "olecranon_l",
    "fa_belly_l",
    "palm_center_l",
    "thumb_cmc_l",
    "thumb_tip_l",
)

_ARM_HAND_BACK_IDS: tuple[str, ...] = (
    "tri_belly_l",
    "tri_belly_r",
    "olecranon_l",
    "olecranon_r",
)

_ARM_HAND_NEW_IDS: tuple[str, ...] = _ARM_HAND_FRONT_IDS


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


def test_a1_arm_hand_ids_in_known() -> None:
    """A1: frozen v1 form-read ids are known (membership in, not ==)."""
    for lid in _ARM_HAND_NEW_IDS:
        assert lid in KNOWN_LANDMARK_IDS


def test_a2_template_blanks_front_left_back() -> None:
    """A2: front blanks include v1 ids; left subset; back tri/olecranon."""
    for lid in _ARM_HAND_FRONT_IDS:
        assert lid in _FRONT_LANDMARK_KEYS
    for lid in _ARM_HAND_LEFT_IDS:
        assert lid in _LEFT_LANDMARK_KEYS
    for lid in _ARM_HAND_BACK_IDS:
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
            "bi_belly_l": _lm2("bi_belly_l", 20.0, 70.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front}, height_m=1.72)
    assert "bi_belly_l" in out
    assert out["bi_belly_l"].x_m is not None
    assert out["bi_belly_l"].z_m is not None
    assert out["bi_belly_l"].y is None
    assert out["bi_belly_l"].y_m is None
    assert "tri_belly_l" not in out
    assert "humeral_head_l" not in out


def test_a4_arm_hand_have_ids_hold() -> None:
    """A4: 0013/0024 shoulder_l / elbow_l / wrist_l stay known."""
    assert "shoulder_l" in KNOWN_LANDMARK_IDS
    assert "elbow_l" in KNOWN_LANDMARK_IDS
    assert "wrist_l" in KNOWN_LANDMARK_IDS
    assert "shoulder_l" in _FRONT_LANDMARK_KEYS
    assert "elbow_l" in _FRONT_LANDMARK_KEYS
    assert "wrist_l" in _FRONT_LANDMARK_KEYS


def test_a5_no_arm_front_depth_pair() -> None:
    """A5: do not add arm_front/back DEPTH_PAIRS."""
    flat = {lid for triple in DEPTH_PAIRS for lid in triple}
    assert "arm_front" not in flat
    assert "arm_back" not in flat
    assert "arm_front" not in KNOWN_LANDMARK_IDS


def test_a6_siblings_still_known() -> None:
    """A6: 0128 trap_apex_l, 0127 gastroc_med_l, 0126 asis_l, 0125 sternum_mid, 0124 eye_l."""
    assert "trap_apex_l" in KNOWN_LANDMARK_IDS
    assert "gastroc_med_l" in KNOWN_LANDMARK_IDS
    assert "asis_l" in KNOWN_LANDMARK_IDS
    assert "sternum_mid" in KNOWN_LANDMARK_IDS
    assert "eye_l" in KNOWN_LANDMARK_IDS
    assert "eye_l" in _FRONT_LANDMARK_KEYS


def test_a_back_view_sets_tri_olecranon_xz() -> None:
    """Back view fills X/Z for tri/olecranon; never invents Y (B8)."""
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
            "tri_belly_l": _lm2("tri_belly_l", 70.0, 75.0),
            "olecranon_l": _lm2("olecranon_l", 75.0, 95.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "back": back}, height_m=1.72)
    assert "tri_belly_l" in out
    assert out["tri_belly_l"].x is not None
    assert out["tri_belly_l"].z is not None
    assert out["tri_belly_l"].x_m is not None
    assert out["tri_belly_l"].z_m is not None
    assert out["tri_belly_l"].y is None
    assert out["tri_belly_l"].y_m is None
    assert out["tri_belly_l"].x < 0.0
    assert "olecranon_l" in out
    assert out["olecranon_l"].z_m is not None
    assert out["olecranon_l"].y_m is None


def test_a_left_view_sets_arm_hand_y() -> None:
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
            "bi_belly_l": _lm2("bi_belly_l", 20.0, 70.0),
        },
    )
    left = ViewLandmarks(
        view="left",
        width_px=100,
        height_px=200,
        facing_direction="camera_left",
        landmarks={
            "chest_front": _lm2("chest_front", 70.0, 60.0),
            "chest_back": _lm2("chest_back", 30.0, 60.0),
            "bi_belly_l": _lm2("bi_belly_l", 25.0, 70.0),
            "tri_belly_l": _lm2("tri_belly_l", 55.0, 72.0),
            "palm_center_l": _lm2("palm_center_l", 20.0, 120.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "left": left}, height_m=1.72)
    assert out["bi_belly_l"].y is not None
    assert out["bi_belly_l"].y_m is not None
    assert out["tri_belly_l"].y is not None
    assert out["tri_belly_l"].y_m is not None
    assert out["palm_center_l"].y is not None
    assert out["palm_center_l"].y_m is not None
