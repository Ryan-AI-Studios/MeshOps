"""Track 0124 — face landmark ids / template blanks / fuse XYZ (offline).

Authoring measurement only (FACE_COMPARE_HONESTY / CAPTURE_HONESTY).
Not mesh or print success. Schema report stay 1.2.0. MCP 48 at ship.
"""

from __future__ import annotations

from meshops.proportion.assist import KNOWN_LANDMARK_IDS
from meshops.proportion.fuse import DEPTH_PAIRS, fuse_xyz
from meshops.proportion.models import Landmark2D, ViewLandmarks
from meshops.proportion.template import (
    _FRONT_LANDMARK_KEYS,
    _LEFT_LANDMARK_KEYS,
    _TQ_LANDMARK_KEYS,
)

_FACE_FRONT_IDS: tuple[str, ...] = (
    "eye_l",
    "eye_r",
    "brow_l",
    "brow_r",
    "nose_tip",
    "mouth_corner_l",
    "mouth_corner_r",
    "lip_mid",
    "cheek_l",
    "cheek_r",
    "ear_l",
    "ear_r",
)

_FACE_LEFT_IDS: tuple[str, ...] = (
    "eye_l",
    "brow_l",
    "nose_tip",
    "lip_mid",
    "ear_l",
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


def test_a1_face_ids_in_known() -> None:
    """A1: frozen v1 form-read ids are known."""
    for lid in _FACE_FRONT_IDS:
        assert lid in KNOWN_LANDMARK_IDS


def test_a2_template_blanks_front_and_left() -> None:
    """A2: front blanks include v1 ids; left subset present."""
    for lid in _FACE_FRONT_IDS:
        assert lid in _FRONT_LANDMARK_KEYS
    for lid in _FACE_LEFT_IDS:
        assert lid in _LEFT_LANDMARK_KEYS
    for lid in ("eye_l", "eye_r", "nose_tip"):
        assert lid in _TQ_LANDMARK_KEYS


def test_a3_missing_id_no_invent_y() -> None:
    """A3: missing face id skipped; front-only never invents Y."""
    front = ViewLandmarks(
        view="front",
        width_px=100,
        height_px=200,
        facing_direction="camera_front",
        landmarks={
            "cranial_vertex": _lm2("cranial_vertex", 50.0, 10.0),
            "sole": _lm2("sole", 50.0, 190.0),
            "chin": _lm2("chin", 50.0, 40.0),
            "eye_l": _lm2("eye_l", 40.0, 30.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front}, height_m=1.72)
    assert "eye_l" in out
    assert out["eye_l"].x_m is not None
    assert out["eye_l"].z_m is not None
    assert out["eye_l"].y is None
    assert out["eye_l"].y_m is None
    assert "eye_r" not in out
    assert "brow_l" not in out


def test_a4_chin_cranial_hold() -> None:
    """A4: 0024 chin / cranial_vertex stay known."""
    assert "chin" in KNOWN_LANDMARK_IDS
    assert "cranial_vertex" in KNOWN_LANDMARK_IDS
    assert "chin" in _FRONT_LANDMARK_KEYS
    assert "cranial_vertex" in _FRONT_LANDMARK_KEYS


def test_a_left_view_sets_face_y() -> None:
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
            "eye_l": _lm2("eye_l", 40.0, 30.0),
        },
    )
    left = ViewLandmarks(
        view="left",
        width_px=100,
        height_px=200,
        facing_direction="camera_left",
        landmarks={
            "chest_front": _lm2("chest_front", 70.0, 80.0),
            "chest_back": _lm2("chest_back", 30.0, 80.0),
            "eye_l": _lm2("eye_l", 65.0, 30.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "left": left}, height_m=1.72)
    assert out["eye_l"].y is not None
    assert out["eye_l"].y_m is not None


def test_a5_no_eye_front_depth_pair() -> None:
    """A5: do not add eye_front/eye_back DEPTH_PAIRS."""
    flat = {lid for triple in DEPTH_PAIRS for lid in triple}
    assert "eye_front" not in flat
    assert "eye_back" not in flat
    assert "eye_front" not in KNOWN_LANDMARK_IDS
