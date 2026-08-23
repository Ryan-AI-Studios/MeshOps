"""Track 0128 — shoulder girdle landmark ids / template blanks / fuse XYZ (offline).

Authoring measurement only (GIRDLE_COMPARE_HONESTY / CAPTURE_HONESTY).
Not mesh or print success. Schema report stay 1.2.0. MCP 52 at ship.
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

_GIRDLE_FRONT_IDS: tuple[str, ...] = (
    "trap_apex_l",
    "trap_apex_r",
    "trap_med_l",
    "trap_med_r",
    "trap_lat_l",
    "trap_lat_r",
    "clav_med_l",
    "clav_med_r",
    "clav_lat_l",
    "clav_lat_r",
    "scm_origin_l",
    "scm_origin_r",
    "scm_insert_l",
    "scm_insert_r",
    "acromion_l",
    "acromion_r",
)

_GIRDLE_LEFT_IDS: tuple[str, ...] = (
    "trap_apex_l",
    "clav_med_l",
    "nape",
    "scm_origin_l",
)

_GIRDLE_BACK_IDS: tuple[str, ...] = (
    "trap_apex_l",
    "trap_apex_r",
    "trap_med_l",
    "trap_med_r",
    "trap_lat_l",
    "trap_lat_r",
    "nape",
)

_GIRDLE_NEW_IDS: tuple[str, ...] = (
    "trap_apex_l",
    "trap_apex_r",
    "trap_med_l",
    "trap_med_r",
    "trap_lat_l",
    "trap_lat_r",
    "clav_med_l",
    "clav_med_r",
    "clav_lat_l",
    "clav_lat_r",
    "scm_origin_l",
    "scm_origin_r",
    "scm_insert_l",
    "scm_insert_r",
    "acromion_l",
    "acromion_r",
    "nape",
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


def test_a1_girdle_ids_in_known() -> None:
    """A1: frozen v1 form-read ids are known (membership in, not ==)."""
    for lid in _GIRDLE_NEW_IDS:
        assert lid in KNOWN_LANDMARK_IDS


def test_a2_template_blanks_front_left_back() -> None:
    """A2: front blanks include v1 ids; left subset; back trap/nape."""
    for lid in _GIRDLE_FRONT_IDS:
        assert lid in _FRONT_LANDMARK_KEYS
    for lid in _GIRDLE_LEFT_IDS:
        assert lid in _LEFT_LANDMARK_KEYS
    for lid in _GIRDLE_BACK_IDS:
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
            "trap_apex_l": _lm2("trap_apex_l", 35.0, 45.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front}, height_m=1.72)
    assert "trap_apex_l" in out
    assert out["trap_apex_l"].x_m is not None
    assert out["trap_apex_l"].z_m is not None
    assert out["trap_apex_l"].y is None
    assert out["trap_apex_l"].y_m is None
    assert "nape" not in out
    assert "clav_med_l" not in out


def test_a4_girdle_have_ids_hold() -> None:
    """A4: 0013/0024 shoulder_l / neck stay known."""
    assert "shoulder_l" in KNOWN_LANDMARK_IDS
    assert "shoulder_r" in KNOWN_LANDMARK_IDS
    assert "neck" in KNOWN_LANDMARK_IDS
    assert "shoulder_l" in _FRONT_LANDMARK_KEYS
    assert "shoulder_r" in _FRONT_LANDMARK_KEYS


def test_a5_no_trap_front_depth_pair() -> None:
    """A5: do not add trap_front/back DEPTH_PAIRS."""
    flat = {lid for triple in DEPTH_PAIRS for lid in triple}
    assert "trap_front" not in flat
    assert "trap_back" not in flat
    assert "trap_front" not in KNOWN_LANDMARK_IDS


def test_a6_gastroc_asis_sternum_eye_still_known() -> None:
    """A6: 0127 gastroc_med_l, 0126 asis_l, 0125 sternum_mid, 0124 eye_l still in KNOWN."""
    assert "gastroc_med_l" in KNOWN_LANDMARK_IDS
    assert "asis_l" in KNOWN_LANDMARK_IDS
    assert "sternum_mid" in KNOWN_LANDMARK_IDS
    assert "eye_l" in KNOWN_LANDMARK_IDS
    assert "eye_l" in _FRONT_LANDMARK_KEYS


def test_a_back_view_sets_trap_xz() -> None:
    """Back view fills X/Z for trap; never invents Y (B8)."""
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
            "trap_apex_l": _lm2("trap_apex_l", 70.0, 40.0),
            "nape": _lm2("nape", 50.0, 35.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "back": back}, height_m=1.72)
    assert "trap_apex_l" in out
    assert out["trap_apex_l"].x is not None
    assert out["trap_apex_l"].z is not None
    assert out["trap_apex_l"].x_m is not None
    assert out["trap_apex_l"].z_m is not None
    assert out["trap_apex_l"].y is None
    assert out["trap_apex_l"].y_m is None
    # camera_back inverts X so larger image-x (right of back photo) → MeshOps -X
    assert out["trap_apex_l"].x < 0.0
    assert "nape" in out
    assert out["nape"].z_m is not None
    assert out["nape"].y_m is None


def test_a_left_view_sets_girdle_y() -> None:
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
            "trap_apex_l": _lm2("trap_apex_l", 35.0, 45.0),
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
            "trap_apex_l": _lm2("trap_apex_l", 25.0, 45.0),
            "clav_med_l": _lm2("clav_med_l", 55.0, 50.0),
            "nape": _lm2("nape", 35.0, 38.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "left": left}, height_m=1.72)
    assert out["trap_apex_l"].y is not None
    assert out["trap_apex_l"].y_m is not None
    assert out["clav_med_l"].y is not None
    assert out["clav_med_l"].y_m is not None
    assert out["nape"].y is not None
    assert out["nape"].y_m is not None
