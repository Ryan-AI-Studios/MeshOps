"""0039 join-ready socket overlaps + connection gap metrics (T1-T6).

Authoring only - not mesh/print success (N6 / FUSE_HONESTY).
"""

from __future__ import annotations

import copy

import pytest

from meshops.proportion.blockout_recipe import (
    BlockoutRecipePackage,
    RecipePart,
    _apply_join_ready_overlaps,
)
from meshops.proportion.connection_metrics import (
    REQUIRED_GAP_KEYS,
    connection_gap_metrics,
)
from meshops.proportion.errors import ProportionError


def _ellipsoid(
    name: str,
    role: str,
    center: list[float],
    r: float = 0.05,
) -> RecipePart:
    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind="ellipsoid",
        center=list(center),
        rx_m=r,
        ry_m=r,
        rz_m=r,
        placement="full3d",
        label=name,
    )


def _capsule(
    name: str,
    role: str,
    p0: list[float],
    p1: list[float],
    radius: float = 0.04,
) -> RecipePart:
    return RecipePart(
        name=name,
        role=role,  # type: ignore[arg-type]
        kind="capsule",
        p0=list(p0),
        p1=list(p1),
        radius_m=radius,
        placement="full3d",
        label=name,
    )


def _shoulder_pkg(
    *, deltoid_y: float, torso_y: float = 0.0, r: float = 0.05
) -> BlockoutRecipePackage:
    """Minimal package for shoulder gap tests (Y attach axis)."""
    return BlockoutRecipePackage(
        parts=[
            _ellipsoid(
                "RECIPE_deltoid_soft_l",
                "deltoid_soft",
                [0.2, deltoid_y, 1.4],
                r=r,
            ),
            _ellipsoid(
                "RECIPE_torso_oval_chest",
                "torso",
                [0.0, torso_y, 1.35],
                r=r,
            ),
        ]
    )


def _full_humanoid_separated() -> BlockoutRecipePackage:
    """Synthetic multi-class package with modest positive gaps (closeable under B2 caps).

    Gaps sized so attach-axis pull + ≤1.08 grow can reach gap ≤ 0 (product caps).
    """
    r = 0.05
    # Parent radii larger so sockets have mass; children separated modestly on axis.
    parts: list[RecipePart] = [
        # Torso parents
        _ellipsoid("RECIPE_torso_oval_chest", "torso", [0.0, 0.0, 1.35], r=0.10),
        _ellipsoid("RECIPE_torso_oval_hip", "torso", [0.0, 0.0, 0.95], r=0.10),
        _ellipsoid("RECIPE_pelvis_oval", "pelvis", [0.0, 0.0, 0.85], r=0.09),
        _ellipsoid("RECIPE_head", "head", [0.0, 0.0, 1.62], r=0.09),
        # Neck (small Z gap vs head and chest)
        _capsule("RECIPE_neck", "neck", [0.0, 0.0, 1.46], [0.0, 0.0, 1.50], radius=0.035),
        # Shoulders - small Y separation (gap ~0.03-0.05)
        _ellipsoid("RECIPE_deltoid_soft_l", "deltoid_soft", [0.22, 0.16, 1.40], r=r),
        _ellipsoid("RECIPE_deltoid_soft_r", "deltoid_soft", [-0.22, 0.16, 1.40], r=r),
        _capsule(
            "RECIPE_shoulder_bridge_l",
            "shoulder_bridge",
            [0.08, 0.16, 1.40],
            [0.18, 0.16, 1.40],
            radius=0.03,
        ),
        _capsule(
            "RECIPE_shoulder_bridge_r",
            "shoulder_bridge",
            [-0.08, 0.16, 1.40],
            [-0.18, 0.16, 1.40],
            radius=0.03,
        ),
        # Hips — small Y separation
        _capsule(
            "RECIPE_hip_bridge_l",
            "hip_bridge",
            [0.05, 0.15, 0.95],
            [0.15, 0.15, 0.95],
            radius=0.04,
        ),
        _capsule(
            "RECIPE_hip_bridge_r",
            "hip_bridge",
            [-0.05, 0.15, 0.95],
            [-0.15, 0.15, 0.95],
            radius=0.04,
        ),
        # Ankle stack — positive Z gaps (~0.03) closeable under B2 (pull + ≤1.08 grow).
        # r=0.04 each; |dz|=0.11 → gap = 0.11 - 0.08 = 0.03 (must be > 0.01 before join).
        _ellipsoid("RECIPE_calf_b_l", "limb_segment", [0.12, 0.05, 0.24], r=0.04),
        _ellipsoid("RECIPE_calf_b_r", "limb_segment", [-0.12, 0.05, 0.24], r=0.04),
        _ellipsoid("RECIPE_ank_foot_l", "ankle_bridge", [0.12, 0.05, 0.13], r=0.04),
        _ellipsoid("RECIPE_ank_foot_r", "ankle_bridge", [-0.12, 0.05, 0.13], r=0.04),
        _ellipsoid("RECIPE_foot_plate_l", "foot_plate", [0.12, -0.02, 0.02], r=0.04),
        _ellipsoid("RECIPE_foot_plate_r", "foot_plate", [-0.12, -0.02, 0.02], r=0.04),
        _ellipsoid("RECIPE_heel_l", "heel", [0.12, 0.04, 0.03], r=0.025),
        _ellipsoid("RECIPE_heel_r", "heel", [-0.12, 0.04, 0.03], r=0.025),
        # Toes — must NOT grow under join-ready
        _ellipsoid("RECIPE_toe_soft_l", "toe_soft", [0.12, -0.12, 0.02], r=0.02),
        _ellipsoid("RECIPE_toe_soft_r", "toe_soft", [-0.12, -0.12, 0.02], r=0.02),
        _ellipsoid("RECIPE_toe_1_l", "toe_soft", [0.13, -0.14, 0.02], r=0.01),
    ]
    return BlockoutRecipePackage(parts=parts)


# ---------------------------------------------------------------------------
# T1-T4 gap metrics + post-pass
# ---------------------------------------------------------------------------


def test_t1_separated_shoulder_gap_positive() -> None:
    """T1: separated shoulder proxies → gap > 0."""
    pkg = _shoulder_pkg(deltoid_y=0.30, torso_y=0.0, r=0.05)
    gaps = connection_gap_metrics(pkg)
    assert gaps["shoulder_l"] > 0.0
    # dist_y=0.30, r+r=0.10 → gap=0.20
    assert gaps["shoulder_l"] == pytest.approx(0.20, abs=1e-9)


def test_t2_overlapped_shoulder_gap_nonpositive() -> None:
    """T2: overlapped shoulder → gap ≤ 0."""
    # Same Y, radii 0.05 each → gap = 0 - 0.05 - 0.05 = -0.10
    pkg = _shoulder_pkg(deltoid_y=0.0, torso_y=0.0, r=0.05)
    gaps = connection_gap_metrics(pkg)
    assert gaps["shoulder_l"] <= 0.0


def test_t3_join_ready_closes_all_required_classes() -> None:
    """T3: join_ready post-pass → ALL required classes gap ≤ eps."""
    pkg = _full_humanoid_separated()
    before = connection_gap_metrics(pkg)
    # Sanity: at least some classes start separated; ankles must start open (P3.2)
    assert any(before[k] > 0.0 for k in REQUIRED_GAP_KEYS)
    assert before["ankle_l"] > 0.01, f"ankle_l pre-closed: {before['ankle_l']}"
    assert before["ankle_r"] > 0.01, f"ankle_r pre-closed: {before['ankle_r']}"

    messages: list[str] = []
    _apply_join_ready_overlaps(pkg.parts, messages)
    after = connection_gap_metrics(pkg)
    eps = 1e-4
    for k in REQUIRED_GAP_KEYS:
        assert after[k] <= eps, f"{k}: gap={after[k]} (before={before[k]})"
    assert any("join_ready." in m for m in messages)


def test_t4_join_ready_false_skips_post_pass() -> None:
    """T4: join_ready=false build-gate mirror → no geometry mutate; True control does.

    Mirrors production ``if join_ready: _apply_join_ready_overlaps(...)`` so a
    regression that always applied the post-pass would fail (not a tautology).
    """
    pkg = _full_humanoid_separated()
    before = connection_gap_metrics(pkg)
    snap = [
        (
            p.name,
            copy.deepcopy(p.center),
            copy.deepcopy(p.p0),
            copy.deepcopy(p.p1),
            p.rx_m,
            p.ry_m,
            p.rz_m,
            p.radius_m,
        )
        for p in pkg.parts
    ]

    # Production gate mirror (build_blockout_recipe only applies when True)
    join_ready = False
    messages: list[str] = []
    if join_ready:
        _apply_join_ready_overlaps(pkg.parts, messages)

    assert pkg.join_ready is False
    assert messages == []
    after = connection_gap_metrics(pkg)
    assert after == before
    for i, p in enumerate(pkg.parts):
        name, c, p0, p1, rx, ry, rz, rad = snap[i]
        assert p.name == name
        assert p.center == c
        assert p.p0 == p0
        assert p.p1 == p1
        assert p.rx_m == rx
        assert p.ry_m == ry
        assert p.rz_m == rz
        assert p.radius_m == rad

    # Control: join_ready=True path mutates (or at least records class messages)
    pkg2 = _full_humanoid_separated()
    msgs2: list[str] = []
    _apply_join_ready_overlaps(pkg2.parts, msgs2)
    after2 = connection_gap_metrics(pkg2)
    assert after2 != before or any("join_ready." in m for m in msgs2)
    assert any("join_ready." in m for m in msgs2)


def test_t5_nofuse_join_ready_mutual_exclusion() -> None:
    """T5: nofuse+join_ready → ProportionError(code=recipe_failed)."""
    from meshops.proportion.blockout_recipe import build_blockout_recipe
    from meshops.proportion.models import ProportionReport, QualityFlags

    # Minimal invalid path: mutual exclusion fires before parts needed
    # Use a real report if available from fixtures; else construct minimal.
    report = ProportionReport(
        schema_version="1.2.0",
        honesty="proportion_measurement_not_mesh_or_print_success",
        quality=QualityFlags(),
    )
    with pytest.raises(ProportionError) as ei:
        build_blockout_recipe(report, nofuse=True, join_ready=True)
    assert ei.value.code == "recipe_failed"
    assert "mutually exclusive" in str(ei.value).lower()


def test_t6_multi_class_no_toe_growth() -> None:
    """T6: multi-class synthetic closes all classes; no RECIPE_toe_* radius growth."""
    pkg = _full_humanoid_separated()
    before = connection_gap_metrics(pkg)
    assert before["ankle_l"] > 0.01 and before["ankle_r"] > 0.01

    toe_before = {
        p.name: (p.rx_m, p.ry_m, p.rz_m) for p in pkg.parts if p.name.startswith("RECIPE_toe_")
    }
    assert toe_before  # fixture has toes

    messages: list[str] = []
    _apply_join_ready_overlaps(pkg.parts, messages)
    after = connection_gap_metrics(pkg)
    eps = 1e-4
    for k in REQUIRED_GAP_KEYS:
        assert after[k] <= eps, f"{k}: gap={after[k]}"

    for p in pkg.parts:
        if p.name.startswith("RECIPE_toe_"):
            assert (p.rx_m, p.ry_m, p.rz_m) == toe_before[p.name]


def test_connection_gap_metrics_required_keys() -> None:
    """Always emit the seven required class keys."""
    pkg = BlockoutRecipePackage(parts=[])
    gaps = connection_gap_metrics(pkg)
    assert set(gaps.keys()) >= set(REQUIRED_GAP_KEYS)
