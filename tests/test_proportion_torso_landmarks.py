"""Track 0125 — torso landmark ids / template blanks / fuse XYZ (offline).

Authoring measurement only (TORSO_COMPARE_HONESTY / CAPTURE_HONESTY).
Not mesh or print success. Schema report stay 1.2.0. MCP 49 at ship.
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

_TORSO_FRONT_IDS: tuple[str, ...] = (
    "sternum_mid",
    "costal_l",
    "costal_r",
)

_TORSO_LEFT_IDS: tuple[str, ...] = (
    "sternum_mid",
    "scap_inferior_l",
    "mid_back_l",
)

_TORSO_BACK_IDS: tuple[str, ...] = (
    "scap_inferior_l",
    "scap_inferior_r",
    "scap_medial_l",
    "scap_medial_r",
    "mid_back_l",
    "mid_back_r",
)

_TORSO_NEW_IDS: tuple[str, ...] = (
    "sternum_mid",
    "costal_l",
    "costal_r",
    "scap_inferior_l",
    "scap_inferior_r",
    "scap_medial_l",
    "scap_medial_r",
    "mid_back_l",
    "mid_back_r",
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


def test_a1_torso_ids_in_known() -> None:
    """A1: frozen v1 form-read ids are known."""
    for lid in _TORSO_NEW_IDS:
        assert lid in KNOWN_LANDMARK_IDS


def test_a2_template_blanks_front_left_back() -> None:
    """A2: front blanks include v1 ids; left subset; back scap/mid_back."""
    for lid in _TORSO_FRONT_IDS:
        assert lid in _FRONT_LANDMARK_KEYS
    for lid in _TORSO_LEFT_IDS:
        assert lid in _LEFT_LANDMARK_KEYS
    for lid in _TORSO_BACK_IDS:
        assert lid in _BACK_LANDMARK_KEYS


def test_a3_missing_id_no_invent_y() -> None:
    """A3: missing torso id skipped; front-only never invents Y."""
    front = ViewLandmarks(
        view="front",
        width_px=100,
        height_px=200,
        facing_direction="camera_front",
        landmarks={
            "cranial_vertex": _lm2("cranial_vertex", 50.0, 10.0),
            "sole": _lm2("sole", 50.0, 190.0),
            "chin": _lm2("chin", 50.0, 40.0),
            "sternum_mid": _lm2("sternum_mid", 50.0, 70.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front}, height_m=1.72)
    assert "sternum_mid" in out
    assert out["sternum_mid"].x_m is not None
    assert out["sternum_mid"].z_m is not None
    assert out["sternum_mid"].y is None
    assert out["sternum_mid"].y_m is None
    assert "costal_l" not in out
    assert "scap_inferior_l" not in out


def test_a4_chest_nipple_navel_hold() -> None:
    """A4: 0013 chest/nipple/navel stay known."""
    assert "chest_front" in KNOWN_LANDMARK_IDS
    assert "nipple_bust" in KNOWN_LANDMARK_IDS
    assert "navel" in KNOWN_LANDMARK_IDS
    assert "chest_front" in _LEFT_LANDMARK_KEYS
    assert "nipple_bust" in _FRONT_LANDMARK_KEYS
    assert "navel" in _FRONT_LANDMARK_KEYS


def test_a5_no_scap_front_depth_pair() -> None:
    """A5: do not add scap_front/scap_back DEPTH_PAIRS."""
    flat = {lid for triple in DEPTH_PAIRS for lid in triple}
    assert "scap_front" not in flat
    assert "scap_back" not in flat
    assert "scap_front" not in KNOWN_LANDMARK_IDS


def test_a6_eye_l_still_known() -> None:
    """A6: 0124 eye_l still in KNOWN."""
    assert "eye_l" in KNOWN_LANDMARK_IDS
    assert "eye_l" in _FRONT_LANDMARK_KEYS


def test_a_back_view_sets_scap_xz() -> None:
    """Back view fills X/Z for scap; never invents Y (B8)."""
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
            "scap_inferior_l": _lm2("scap_inferior_l", 70.0, 60.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "back": back}, height_m=1.72)
    assert "scap_inferior_l" in out
    assert out["scap_inferior_l"].x is not None
    assert out["scap_inferior_l"].z is not None
    assert out["scap_inferior_l"].x_m is not None
    assert out["scap_inferior_l"].z_m is not None
    assert out["scap_inferior_l"].y is None
    assert out["scap_inferior_l"].y_m is None
    # camera_back inverts X so larger image-x (right of back photo) → MeshOps -X
    assert out["scap_inferior_l"].x < 0.0


def test_a_left_view_sets_torso_y() -> None:
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
            "sternum_mid": _lm2("sternum_mid", 50.0, 70.0),
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
            "sternum_mid": _lm2("sternum_mid", 65.0, 70.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "left": left}, height_m=1.72)
    assert out["sternum_mid"].y is not None
    assert out["sternum_mid"].y_m is not None
