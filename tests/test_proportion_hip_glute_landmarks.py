"""Track 0126 — hip/glute/groin landmark ids / template blanks / fuse XYZ (offline).

Authoring measurement only (HIP_GLUTE_COMPARE_HONESTY / CAPTURE_HONESTY).
Not mesh or print success. Schema report stay 1.2.0. MCP 50 at ship.
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

_HIP_GLUTE_FRONT_IDS: tuple[str, ...] = (
    "asis_l",
    "asis_r",
    "groin_fold_l",
    "groin_fold_r",
    "thigh_medial_l",
    "thigh_medial_r",
)

_HIP_GLUTE_LEFT_IDS: tuple[str, ...] = (
    "glute_bottom_l",
    "glute_top_seam",
    "psis_l",
)

_HIP_GLUTE_BACK_IDS: tuple[str, ...] = (
    "psis_l",
    "psis_r",
    "glute_outer_l",
    "glute_outer_r",
    "glute_bottom_l",
    "glute_bottom_r",
    "glute_top_seam",
)

_HIP_GLUTE_NEW_IDS: tuple[str, ...] = (
    "asis_l",
    "asis_r",
    "psis_l",
    "psis_r",
    "glute_outer_l",
    "glute_outer_r",
    "glute_bottom_l",
    "glute_bottom_r",
    "glute_top_seam",
    "groin_fold_l",
    "groin_fold_r",
    "thigh_medial_l",
    "thigh_medial_r",
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


def test_a1_hip_glute_ids_in_known() -> None:
    """A1: frozen v1 form-read ids are known (membership in, not ==)."""
    for lid in _HIP_GLUTE_NEW_IDS:
        assert lid in KNOWN_LANDMARK_IDS


def test_a2_template_blanks_front_left_back() -> None:
    """A2: front blanks include v1 ids; left subset; back outer/bottom/psis/top_seam."""
    for lid in _HIP_GLUTE_FRONT_IDS:
        assert lid in _FRONT_LANDMARK_KEYS
    for lid in _HIP_GLUTE_LEFT_IDS:
        assert lid in _LEFT_LANDMARK_KEYS
    for lid in _HIP_GLUTE_BACK_IDS:
        assert lid in _BACK_LANDMARK_KEYS


def test_a3_missing_id_no_invent_y() -> None:
    """A3: missing hip/glute id skipped; front-only never invents Y."""
    front = ViewLandmarks(
        view="front",
        width_px=100,
        height_px=200,
        facing_direction="camera_front",
        landmarks={
            "cranial_vertex": _lm2("cranial_vertex", 50.0, 10.0),
            "sole": _lm2("sole", 50.0, 190.0),
            "chin": _lm2("chin", 50.0, 40.0),
            "asis_l": _lm2("asis_l", 35.0, 110.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front}, height_m=1.72)
    assert "asis_l" in out
    assert out["asis_l"].x_m is not None
    assert out["asis_l"].z_m is not None
    assert out["asis_l"].y is None
    assert out["asis_l"].y_m is None
    assert "glute_bottom_l" not in out
    assert "psis_l" not in out


def test_a4_hip_glute_have_ids_hold() -> None:
    """A4: 0013/0030 hip/glute DEPTH + trochanter + peak/cleft stay known."""
    assert "hip_front" in KNOWN_LANDMARK_IDS
    assert "glute_back" in KNOWN_LANDMARK_IDS
    assert "greater_trochanter" in KNOWN_LANDMARK_IDS
    assert "glute_peak_l" in KNOWN_LANDMARK_IDS
    assert "crotch_pubic" in KNOWN_LANDMARK_IDS
    assert "hip_front" in _LEFT_LANDMARK_KEYS
    assert "glute_back" in _LEFT_LANDMARK_KEYS
    assert "greater_trochanter" in _FRONT_LANDMARK_KEYS
    assert "crotch_pubic" in _FRONT_LANDMARK_KEYS


def test_a5_no_glute_outer_front_depth_pair() -> None:
    """A5: do not add glute_outer_front/back DEPTH_PAIRS."""
    flat = {lid for triple in DEPTH_PAIRS for lid in triple}
    assert "glute_outer_front" not in flat
    assert "glute_outer_back" not in flat
    assert "glute_outer_front" not in KNOWN_LANDMARK_IDS


def test_a6_sternum_and_eye_still_known() -> None:
    """A6: 0125 sternum_mid and 0124 eye_l still in KNOWN."""
    assert "sternum_mid" in KNOWN_LANDMARK_IDS
    assert "eye_l" in KNOWN_LANDMARK_IDS
    assert "eye_l" in _FRONT_LANDMARK_KEYS


def test_a_back_view_sets_glute_outer_xz() -> None:
    """Back view fills X/Z for glute_outer; never invents Y (B8)."""
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
            "glute_outer_l": _lm2("glute_outer_l", 70.0, 120.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "back": back}, height_m=1.72)
    assert "glute_outer_l" in out
    assert out["glute_outer_l"].x is not None
    assert out["glute_outer_l"].z is not None
    assert out["glute_outer_l"].x_m is not None
    assert out["glute_outer_l"].z_m is not None
    assert out["glute_outer_l"].y is None
    assert out["glute_outer_l"].y_m is None
    # camera_back inverts X so larger image-x (right of back photo) → MeshOps -X
    assert out["glute_outer_l"].x < 0.0


def test_a_left_view_sets_hip_glute_y() -> None:
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
            "asis_l": _lm2("asis_l", 35.0, 110.0),
        },
    )
    left = ViewLandmarks(
        view="left",
        width_px=100,
        height_px=200,
        facing_direction="camera_left",
        landmarks={
            "hip_front": _lm2("hip_front", 70.0, 110.0),
            "hip_back": _lm2("hip_back", 30.0, 110.0),
            "glute_bottom_l": _lm2("glute_bottom_l", 25.0, 125.0),
        },
    )
    out, _quality, _msgs = fuse_xyz({"front": front, "left": left}, height_m=1.72)
    assert out["glute_bottom_l"].y is not None
    assert out["glute_bottom_l"].y_m is not None
    assert out["asis_l"].y is None
    assert out["asis_l"].y_m is None
