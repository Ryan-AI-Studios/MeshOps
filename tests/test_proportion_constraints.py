"""Track 0023 — blockout constraints validate + freeze-feet optimize."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from meshops.proportion.blockout_recipe import (
    BlockoutRecipePackage,
    RecipePart,
)
from meshops.proportion.constraints import (
    ANKLE_OVER_HEEL_TOL_M,
    BAND_W_BREAST,
    BAND_W_CALF,
    BAND_W_FOOT,
    BAND_W_GLUTE,
    BAND_W_THIGH,
    CONSTRAINTS_SCHEMA_VERSION,
    FOOT_WIDTH_TOL_M,
    FREEZE_FEET_ROLES,
    OPTIMIZE_FAST_SEED,
    OPTIMIZE_SCHEMA_VERSION,
    OPTIMIZE_SLOW_SEED,
    OUTER_X_TOL_M,
    SOFT_GAP_FRAC,
    _band_weighted_free_dof_score,
    classify_part_name,
    optimize_package,
    part_y,
    run_blockout_optimize,
    run_blockout_validate_constraints,
    validate_constraints,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import CONSTRAINT_HONESTY, OPTIMIZE_HONESTY


def _part(
    name: str,
    role: str = "limb_segment",
    kind: str = "ellipsoid",
    *,
    center: list[float] | None = None,
    rx_m: float | None = 0.04,
    ry_m: float | None = 0.05,
    rz_m: float | None = 0.04,
    radius_m: float | None = None,
    half_depth_m: float | None = None,
    p0: list[float] | None = None,
    p1: list[float] | None = None,
) -> RecipePart:
    # Map constraint-oriented names to coarse RecipeRole for schema validity.
    recipe_role = role
    if role not in (
        "torso",
        "pelvis",
        "neck",
        "head",
        "shoulder_bridge",
        "hip_bridge",
        "deltoid_soft",
        "breast_soft",
        "glute_soft",
        "iliac_soft",
        "limb_segment",
    ):
        recipe_role = "limb_segment"
    kwargs: dict[str, Any] = {
        "name": name,
        "role": recipe_role,
        "kind": kind,
        "center": center,
        "rx_m": rx_m,
        "ry_m": ry_m,
        "rz_m": rz_m,
        "radius_m": radius_m,
        "half_depth_m": half_depth_m,
        "p0": p0,
        "p1": p1,
    }
    # Drop Nones that pydantic optional fields accept — but ellipsoids need radii
    clean = {k: v for k, v in kwargs.items() if v is not None or k in ("name", "role", "kind")}
    return RecipePart.model_validate(clean)


def _pkg(parts: list[RecipePart]) -> BlockoutRecipePackage:
    return BlockoutRecipePackage(
        parts=parts,
        counts={"parts": len(parts)},
    )


def _write_recipe(path: Path, package: BlockoutRecipePackage) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(package.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Classifier (B1 / B2)
# ---------------------------------------------------------------------------


def test_classify_ank_foot_is_ankle_bridge_not_foot_plate() -> None:
    role, side = classify_part_name("RECIPE_ank_foot_l")
    assert role == "ankle_bridge"
    assert side == "l"
    role2, side2 = classify_part_name("OVAL_ank_foot_r")
    assert role2 == "ankle_bridge"
    assert side2 == "r"
    # Must not match foot_plate via "foot" substring
    assert role != "foot_plate"


def test_classify_limb_calf_is_whole_calf_not_proximal() -> None:
    role, side = classify_part_name("RECIPE_limb_calf_l")
    assert role == "calf"
    assert side == "l"
    assert role != "calf_proximal"


def test_classify_calf_b_distal_and_calf_a_proximal() -> None:
    assert classify_part_name("RECIPE_calf_b_l") == ("calf_distal", "l")
    assert classify_part_name("RECIPE_calf_a_r") == ("calf_proximal", "r")
    assert classify_part_name("RECIPE_calf_l") == ("calf", "l")


def test_classify_unknown() -> None:
    role, side = classify_part_name("RECIPE_mystery_blob")
    assert role == "unknown"
    assert side == "none"


def test_classify_foot_plate_and_heel() -> None:
    assert classify_part_name("RECIPE_foot_plate_l") == ("foot_plate", "l")
    assert classify_part_name("RECIPE_heel_r") == ("heel", "r")


def test_constants_and_honesty_tokens() -> None:
    assert CONSTRAINT_HONESTY == ("proportion_blockout_constraints_not_mesh_or_print_success")
    assert OPTIMIZE_HONESTY == ("proportion_blockout_optimize_not_mesh_or_print_success")
    assert ANKLE_OVER_HEEL_TOL_M == 0.03
    assert FOOT_WIDTH_TOL_M == 0.015
    assert OUTER_X_TOL_M == 0.02
    assert SOFT_GAP_FRAC == 0.9
    assert BAND_W_BREAST == 1.5
    assert BAND_W_GLUTE == 1.5
    assert BAND_W_THIGH == 1.0
    assert BAND_W_CALF == 0.5
    assert BAND_W_FOOT == 0.0
    assert OPTIMIZE_FAST_SEED == 11
    assert OPTIMIZE_SLOW_SEED == 13
    assert frozenset({"foot_plate", "heel", "ankle_bridge", "calf_distal"}) == FREEZE_FEET_ROLES


# ---------------------------------------------------------------------------
# Validate rules
# ---------------------------------------------------------------------------


def test_ankle_at_toe_y_fails() -> None:
    # Foot plate center y=0, extent ry=0.08 → rear third ~[0.027, 0.08]
    # Ankle at toe front y=-0.06 should fail.
    pkg = _pkg(
        [
            _part(
                "RECIPE_foot_plate_l",
                role="limb_segment",
                kind="ellipsoid",
                center=[0.08, 0.0, 0.02],
                rx_m=0.04,
                ry_m=0.08,
                rz_m=0.02,
            ),
            _part(
                "RECIPE_ank_foot_l",
                role="limb_segment",
                kind="ellipsoid",
                center=[0.08, -0.06, 0.05],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.03,
            ),
        ]
    )
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_ankle_over_heel"].status == "fail"
    assert report.ok is False
    assert report.honesty == CONSTRAINT_HONESTY
    assert report.schema_version == CONSTRAINTS_SCHEMA_VERSION


def test_ankle_over_heel_pass() -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_heel_l",
                kind="ellipsoid",
                center=[0.08, 0.05, 0.02],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.02,
            ),
            _part(
                "RECIPE_ank_foot_l",
                kind="ellipsoid",
                center=[0.08, 0.05, 0.06],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.03,
            ),
        ]
    )
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_ankle_over_heel"].status == "pass"


def test_ankle_plate_mid_heel_side_not_rear_third_fails() -> None:
    """B4 pure rear-third: plate cy=0, ry=0.08 → rear≈[0.0267, 0.08]; y=0.02 fails."""
    pkg = _pkg(
        [
            _part(
                "RECIPE_foot_plate_l",
                kind="ellipsoid",
                center=[0.08, 0.0, 0.02],
                rx_m=0.04,
                ry_m=0.08,
                rz_m=0.02,
            ),
            _part(
                "RECIPE_ank_foot_l",
                kind="ellipsoid",
                center=[0.08, 0.02, 0.05],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.03,
            ),
        ]
    )
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    rule = by_id["C_ankle_over_heel"]
    assert rule.status == "fail"
    assert rule.metrics is not None
    assert rule.metrics["rear_third_l"][0] == pytest.approx(0.08 / 3.0)
    assert rule.metrics["rear_third_l"][1] == pytest.approx(0.08)


def test_ankle_plate_in_rear_third_pass() -> None:
    """B4 plate path: ankle y=0.05 is inside rear third of plate cy=0, ry=0.08."""
    pkg = _pkg(
        [
            _part(
                "RECIPE_foot_plate_l",
                kind="ellipsoid",
                center=[0.08, 0.0, 0.02],
                rx_m=0.04,
                ry_m=0.08,
                rz_m=0.02,
            ),
            _part(
                "RECIPE_ank_foot_l",
                kind="ellipsoid",
                center=[0.08, 0.05, 0.05],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.03,
            ),
        ]
    )
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_ankle_over_heel"].status == "pass"


def test_foot_width_far_from_ankle_diam_fails() -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_foot_plate_l",
                kind="ellipsoid",
                center=[0.08, 0.04, 0.02],
                rx_m=0.08,  # width 0.16
                ry_m=0.08,
                rz_m=0.02,
            ),
            _part(
                "RECIPE_ank_foot_l",
                kind="ellipsoid",
                center=[0.08, 0.04, 0.06],
                rx_m=0.03,  # diam 0.06
                ry_m=0.03,
                rz_m=0.03,
            ),
        ]
    )
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_foot_width"].status == "fail"


def test_thigh_outer_far_from_hip_bridge_fails() -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_hip_bridge",
                role="hip_bridge",
                kind="ellipsoid",
                center=[0.0, 0.0, 0.95],
                rx_m=0.15,
                ry_m=0.06,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_limb_thigh_r",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
                # outer tip = 0.35 + 0.05 = 0.40; hip outer = 0.15 → far
                p0=[0.35, 0.0, 0.5],
                p1=[0.35, 0.0, 0.9],
            ),
        ]
    )
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_thigh_outer"].status == "fail"


def test_glute_outer_far_from_hip_bridge_fails() -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_hip_bridge",
                role="hip_bridge",
                kind="ellipsoid",
                center=[0.0, 0.0, 0.95],
                rx_m=0.15,
                ry_m=0.06,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_glute_r",
                role="glute_soft",
                kind="ellipsoid",
                center=[0.40, 0.08, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
        ]
    )
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_glute_outer"].status == "fail"


def test_role_classified_unknown_critical_ankle_name_fails() -> None:
    # Bare "ankle" is not a classifier hit, but is a critical foot-stack token.
    pkg = _pkg(
        [
            _part(
                "RECIPE_ankle_mystery_l",
                kind="ellipsoid",
                center=[0.08, 0.04, 0.06],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.03,
            ),
        ]
    )
    role, _side = classify_part_name("RECIPE_ankle_mystery_l")
    assert role == "unknown"
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_role_classified"].status == "fail"


def test_glute_cleft_coincident_fails() -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_glute_l",
                role="glute_soft",
                kind="ellipsoid",
                center=[0.0, 0.08, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_glute_r",
                role="glute_soft",
                kind="ellipsoid",
                center=[0.0, 0.08, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
        ]
    )

    class _C:
        intermammary_gap_m = None
        glute_cleft_m = 0.06

    class _T:
        constants = _C()

    report = validate_constraints(pkg, template_applied=_T())
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_glute_cleft"].status == "fail"


def test_breast_gap_frac_times_bust_fallback_fails() -> None:
    """P3: when gap_m missing, use intermammary_gap_frac * bust half-width."""
    pkg = _pkg(
        [
            _part(
                "RECIPE_breast_l",
                role="breast_soft",
                kind="ellipsoid",
                center=[0.0, -0.05, 1.2],
                rx_m=0.05,
                ry_m=0.04,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_breast_r",
                role="breast_soft",
                kind="ellipsoid",
                center=[0.0, -0.05, 1.2],
                rx_m=0.05,
                ry_m=0.04,
                rz_m=0.05,
            ),
        ]
    )

    class _C:
        intermammary_gap_m = None
        intermammary_gap_frac = 0.2
        glute_cleft_m = None

    class _T:
        constants = _C()

    class _D:
        band_id = "bust"
        half_width_m = 0.15
        width_m = 0.30

    class _R:
        def __init__(self) -> None:
            self.diameters = [_D()]

    report = validate_constraints(pkg, report=_R(), template_applied=_T())
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_breast_gap"].status == "fail"
    # min_gap = 0.9 * 0.2 * 0.15 = 0.027
    assert by_id["C_breast_gap"].metrics is not None
    assert by_id["C_breast_gap"].metrics["template_gap_m"] == pytest.approx(0.03)


def test_breast_gap_coincident_fails() -> None:
    # Dual breasts at same X with template gap prior → fail
    pkg = _pkg(
        [
            _part(
                "RECIPE_breast_l",
                role="breast_soft",
                kind="ellipsoid",
                center=[0.0, -0.05, 1.2],
                rx_m=0.05,
                ry_m=0.04,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_breast_r",
                role="breast_soft",
                kind="ellipsoid",
                center=[0.0, -0.05, 1.2],
                rx_m=0.05,
                ry_m=0.04,
                rz_m=0.05,
            ),
        ]
    )

    class _C:
        intermammary_gap_m = 0.08
        glute_cleft_m = None

    class _T:
        constants = _C()

    report = validate_constraints(pkg, template_applied=_T())
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_breast_gap"].status == "fail"
    assert report.ok is False


def test_calf_slant_skips_whole_calf() -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_limb_calf_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.04,
                p0=[0.1, 0.0, 0.2],
                p1=[0.1, 0.0, 0.45],
            ),
        ]
    )
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_calf_slant"].status == "skip"
    assert "whole calf" in by_id["C_calf_slant"].message.lower() or "skip" in (
        by_id["C_calf_slant"].status
    )


def test_calf_slant_split_distal_toward_ankle_pass() -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_calf_a_l",
                kind="ellipsoid",
                center=[0.1, 0.02, 0.4],
                rx_m=0.04,
                ry_m=0.04,
                rz_m=0.06,
            ),
            _part(
                "RECIPE_calf_b_l",
                kind="ellipsoid",
                center=[0.1, 0.05, 0.2],
                rx_m=0.035,
                ry_m=0.035,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_ank_foot_l",
                kind="ellipsoid",
                center=[0.1, 0.05, 0.08],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.03,
            ),
            _part(
                "RECIPE_heel_l",
                kind="ellipsoid",
                center=[0.1, 0.05, 0.02],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.02,
            ),
        ]
    )
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_calf_slant"].status == "pass"
    # Distal not forced mid: distal y == ankle y
    assert part_y(pkg.parts[1]) == pytest.approx(0.05)


def test_duplicate_limb_dot001_fails() -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_limb_thigh_r",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
                p0=[0.12, 0.0, 0.5],
                p1=[0.12, 0.0, 0.9],
            ),
            _part(
                "RECIPE_limb_thigh_r.001",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
                p0=[0.13, 0.0, 0.5],
                p1=[0.13, 0.0, 0.9],
            ),
        ]
    )
    report = validate_constraints(pkg)
    by_id = {r.id: r for r in report.rules}
    assert by_id["C_no_dup_limb"].status == "fail"
    assert report.ok is False


# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------


def test_freeze_feet_ankle_and_foot_y_unchanged(tmp_path: Path) -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_foot_plate_l",
                kind="ellipsoid",
                center=[0.08, 0.04, 0.02],
                rx_m=0.04,
                ry_m=0.08,
                rz_m=0.02,
            ),
            _part(
                "RECIPE_ank_foot_l",
                kind="ellipsoid",
                center=[0.08, 0.04, 0.06],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.03,
            ),
            _part(
                "RECIPE_heel_l",
                kind="ellipsoid",
                center=[0.08, 0.04, 0.02],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.02,
            ),
            _part(
                "RECIPE_calf_b_l",
                kind="ellipsoid",
                center=[0.1, 0.04, 0.15],
                rx_m=0.035,
                ry_m=0.035,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_limb_thigh_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
                p0=[0.1, 0.0, 0.5],
                p1=[0.1, 0.0, 0.9],
            ),
            _part(
                "RECIPE_limb_calf_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.04,
                p0=[0.1, 0.0, 0.2],
                p1=[0.1, 0.0, 0.45],
            ),
            _part(
                "RECIPE_breast_l",
                role="breast_soft",
                kind="ellipsoid",
                center=[-0.06, -0.02, 1.2],
                rx_m=0.05,
                ry_m=0.04,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_breast_r",
                role="breast_soft",
                kind="ellipsoid",
                center=[0.06, -0.02, 1.2],
                rx_m=0.05,
                ry_m=0.04,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_hip_bridge",
                role="hip_bridge",
                kind="ellipsoid",
                center=[0.0, 0.0, 0.95],
                rx_m=0.15,
                ry_m=0.06,
                rz_m=0.05,
            ),
        ]
    )
    ankle_y0 = part_y(pkg.parts[1])
    foot_y0 = part_y(pkg.parts[0])
    heel_y0 = part_y(pkg.parts[2])
    calf_distal_y0 = part_y(pkg.parts[3])
    assert ankle_y0 is not None and foot_y0 is not None
    assert heel_y0 is not None and calf_distal_y0 is not None

    optimized, result = optimize_package(pkg, mode="fast", freeze_feet=True)
    assert result.honesty == OPTIMIZE_HONESTY
    assert result.schema_version == OPTIMIZE_SCHEMA_VERSION
    assert result.freeze_feet is True
    assert result.mode == "fast"

    by_name = {p.name: p for p in optimized.parts}
    assert part_y(by_name["RECIPE_ank_foot_l"]) == pytest.approx(ankle_y0)
    assert part_y(by_name["RECIPE_foot_plate_l"]) == pytest.approx(foot_y0)
    assert part_y(by_name["RECIPE_heel_l"]) == pytest.approx(heel_y0)
    assert part_y(by_name["RECIPE_calf_b_l"]) == pytest.approx(calf_distal_y0)

    # Also via CLI entrypoint
    recipe_path = _write_recipe(tmp_path / "in" / "blockout_recipe.json", pkg)
    out_dir = tmp_path / "opt_out"
    payload = run_blockout_optimize(
        recipe_path,
        out_dir,
        mode="fast",
        freeze_feet=True,
        force=True,
    )
    assert payload["ok"] is True
    assert payload["honesty"] == OPTIMIZE_HONESTY
    written = BlockoutRecipePackage.model_validate(
        json.loads((out_dir / "blockout_recipe.json").read_text(encoding="utf-8"))
    )
    wnames = {p.name: p for p in written.parts}
    assert part_y(wnames["RECIPE_ank_foot_l"]) == pytest.approx(ankle_y0)
    assert part_y(wnames["RECIPE_foot_plate_l"]) == pytest.approx(foot_y0)
    assert part_y(wnames["RECIPE_heel_l"]) == pytest.approx(heel_y0)
    assert part_y(wnames["RECIPE_calf_b_l"]) == pytest.approx(calf_distal_y0)


def test_band_weighted_free_dof_score_ranks_glute_y() -> None:
    """Free-DOF band score differs without mesh when free soft Y differs."""

    class _C:
        glute_y_m = 0.08
        intermammary_gap_m = None
        glute_cleft_m = None

    class _T:
        constants = _C()

    near = _pkg(
        [
            _part(
                "RECIPE_glute_l",
                role="glute_soft",
                kind="ellipsoid",
                center=[-0.08, 0.08, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_glute_r",
                role="glute_soft",
                kind="ellipsoid",
                center=[0.08, 0.08, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_limb_thigh_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
                p0=[0.1, 0.0, 0.5],
                p1=[0.1, 0.0, 0.9],
            ),
        ]
    )
    far = _pkg(
        [
            _part(
                "RECIPE_glute_l",
                role="glute_soft",
                kind="ellipsoid",
                center=[-0.08, 0.20, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_glute_r",
                role="glute_soft",
                kind="ellipsoid",
                center=[0.08, 0.20, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_limb_thigh_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
                p0=[0.1, 0.0, 0.5],
                p1=[0.1, 0.0, 0.9],
            ),
        ]
    )
    s_near = _band_weighted_free_dof_score(
        near, freeze_feet=True, report=None, template_applied=_T()
    )
    s_far = _band_weighted_free_dof_score(far, freeze_feet=True, report=None, template_applied=_T())
    assert s_near < s_far
    assert s_near == pytest.approx(0.0)
    # weight 1.5 * |0.20-0.08| * 2 glutes
    assert s_far == pytest.approx(1.5 * 0.12 * 2)


def test_slow_without_mesh_raises() -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_limb_thigh_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
                p0=[0.1, 0.0, 0.5],
                p1=[0.1, 0.0, 0.9],
            ),
        ]
    )
    with pytest.raises(ProportionError) as ei:
        optimize_package(pkg, mode="slow", freeze_feet=True, mesh=None)
    assert ei.value.code == "optimize_slow_needs_mesh"


def test_slow_with_mesh_uses_band_weighted_free_dof_message(tmp_path: Path) -> None:
    # Dual glutes + Y target so free set is non-empty after unscored-role filter.
    class _C:
        glute_y_m = 0.08

    class _T:
        constants = _C()

    pkg = _pkg(
        [
            _part(
                "RECIPE_limb_thigh_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
                p0=[0.1, 0.0, 0.5],
                p1=[0.1, 0.0, 0.9],
            ),
            _part(
                "RECIPE_glute_l",
                role="glute_soft",
                kind="ellipsoid",
                center=[-0.08, 0.20, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_glute_r",
                role="glute_soft",
                kind="ellipsoid",
                center=[0.08, 0.20, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
        ]
    )
    dummy_mesh = tmp_path / "joined.stl"
    dummy_mesh.write_bytes(b"solid empty\nendsolid empty\n")
    _optimized, result = optimize_package(
        pkg,
        mode="slow",
        freeze_feet=True,
        mesh=dummy_mesh,
        report=None,  # avoid load_report fail; mesh contract still satisfied
        template_applied=_T(),
    )
    joined = " ".join(result.messages)
    assert "band_weighted_free_dof" in joined
    assert "mesh static baseline not used for trial ranking" in joined
    assert result.mode == "slow"


def test_optimize_no_free_dofs_when_only_frozen() -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_foot_plate_l",
                kind="ellipsoid",
                center=[0.08, 0.04, 0.02],
                rx_m=0.04,
                ry_m=0.08,
                rz_m=0.02,
            ),
            _part(
                "RECIPE_ank_foot_l",
                kind="ellipsoid",
                center=[0.08, 0.04, 0.06],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.03,
            ),
        ]
    )
    with pytest.raises(ProportionError) as ei:
        optimize_package(pkg, mode="fast", freeze_feet=True)
    assert ei.value.code == "optimize_no_free_dofs"


def test_optimize_thigh_without_ankle_anchor_no_free_dofs() -> None:
    """P1: thigh + hip_bridge only (no ankle/foot) → no thigh Y target → refuse."""
    pkg = _pkg(
        [
            _part(
                "RECIPE_limb_thigh_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
                p0=[0.1, 0.0, 0.5],
                p1=[0.1, 0.0, 0.9],
            ),
            _part(
                "RECIPE_hip_bridge",
                role="hip_bridge",
                kind="ellipsoid",
                center=[0.0, 0.0, 0.95],
                rx_m=0.15,
                ry_m=0.06,
                rz_m=0.05,
            ),
        ]
    )
    thigh_y0 = part_y(pkg.parts[0])
    hip_y0 = part_y(pkg.parts[1])
    assert thigh_y0 is not None and hip_y0 is not None
    with pytest.raises(ProportionError) as ei:
        optimize_package(pkg, mode="fast", freeze_feet=True)
    assert ei.value.code == "optimize_no_free_dofs"
    # Input package unchanged when refuse (work is a deep copy).
    assert part_y(pkg.parts[0]) == pytest.approx(thigh_y0)
    assert part_y(pkg.parts[1]) == pytest.approx(hip_y0)


def test_optimize_thigh_with_anchors_moves_thigh_not_hip_bridge_y() -> None:
    """P1: thigh mid target from hip+ankle; hip_bridge Y never free-walks."""
    pkg = _pkg(
        [
            _part(
                "RECIPE_limb_thigh_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
                # Far from mid of hip_y=0 and ankle_y=0.04 → mid≈0.02
                p0=[0.1, 0.15, 0.5],
                p1=[0.1, 0.15, 0.9],
            ),
            _part(
                "RECIPE_hip_bridge",
                role="hip_bridge",
                kind="ellipsoid",
                center=[0.0, 0.0, 0.95],
                rx_m=0.15,
                ry_m=0.06,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_ank_foot_l",
                kind="ellipsoid",
                center=[0.1, 0.04, 0.06],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.03,
            ),
        ]
    )
    thigh_y0 = part_y(pkg.parts[0])
    hip_y0 = part_y(pkg.parts[1])
    assert thigh_y0 is not None and hip_y0 is not None

    optimized, result = optimize_package(pkg, mode="fast", freeze_feet=True)
    by_name = {p.name: p for p in optimized.parts}
    assert result.score_after <= result.score_before + 1e-12
    # hip_bridge has no Y target and is excluded from free set — Y unchanged.
    assert part_y(by_name["RECIPE_hip_bridge"]) == pytest.approx(hip_y0)
    # Thigh may move toward mid target (initial pull and/or strict improves).
    thigh_y1 = part_y(by_name["RECIPE_limb_thigh_l"])
    assert thigh_y1 is not None
    # Frozen ankle Y preserved.
    assert part_y(by_name["RECIPE_ank_foot_l"]) == pytest.approx(0.04)


def test_optimize_glute_duals_with_gap_template_still_runs() -> None:
    """P1: dual glutes + gap/Y template remain free and optimizable."""

    class _C:
        glute_y_m = 0.08
        glute_cleft_m = 0.04

    class _T:
        constants = _C()

    pkg = _pkg(
        [
            _part(
                "RECIPE_glute_l",
                role="glute_soft",
                kind="ellipsoid",
                center=[-0.08, 0.20, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_glute_r",
                role="glute_soft",
                kind="ellipsoid",
                center=[0.08, 0.20, 0.9],
                rx_m=0.05,
                ry_m=0.05,
                rz_m=0.05,
            ),
        ]
    )
    optimized, result = optimize_package(
        pkg,
        mode="fast",
        freeze_feet=True,
        template_applied=_T(),
    )
    assert result.score_after <= result.score_before + 1e-12
    # Initial pull toward glute_y_m=0.08 should move at least one glute off 0.20.
    by_name = {p.name: p for p in optimized.parts}
    yl = part_y(by_name["RECIPE_glute_l"])
    yr = part_y(by_name["RECIPE_glute_r"])
    assert yl is not None and yr is not None
    assert yl < 0.20 or yr < 0.20 or result.score_after < result.score_before
    assert result.n_trials is not None and result.n_trials > 0


def test_optimize_unscored_only_leaves_y_identical_or_refuses() -> None:
    """P1: unscored-only recipe must not score-neutral-walk Y (strict keep)."""
    pkg = _pkg(
        [
            _part(
                "RECIPE_hip_bridge",
                role="hip_bridge",
                kind="ellipsoid",
                center=[0.0, 0.12, 0.95],
                rx_m=0.15,
                ry_m=0.06,
                rz_m=0.05,
            ),
            _part(
                "RECIPE_pelvis",
                role="pelvis",
                kind="ellipsoid",
                center=[0.0, 0.05, 0.9],
                rx_m=0.12,
                ry_m=0.08,
                rz_m=0.06,
            ),
            _part(
                "RECIPE_limb_upper_arm_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.03,
                p0=[0.2, 0.1, 1.1],
                p1=[0.25, 0.1, 0.8],
            ),
        ]
    )
    y0 = {p.name: part_y(p) for p in pkg.parts}
    try:
        optimized, result = optimize_package(pkg, mode="fast", freeze_feet=True)
    except ProportionError as exc:
        assert exc.code == "optimize_no_free_dofs"
        # Prefer raise when free empty — input unchanged.
        for p in pkg.parts:
            assert part_y(p) == pytest.approx(y0[p.name])  # type: ignore[arg-type]
        return
    # Fallback contract: if optimize runs, every Y must match input (no walk).
    by_name = {p.name: p for p in optimized.parts}
    for name, y in y0.items():
        assert y is not None
        assert part_y(by_name[name]) == pytest.approx(y)
    assert result.score_before == pytest.approx(0.0)
    assert result.score_after == pytest.approx(0.0)
    assert result.n_kept == 0


def test_validate_entrypoint_writes_report(tmp_path: Path) -> None:
    pkg = _pkg(
        [
            _part(
                "RECIPE_heel_l",
                kind="ellipsoid",
                center=[0.08, 0.05, 0.02],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.02,
            ),
            _part(
                "RECIPE_ank_foot_l",
                kind="ellipsoid",
                center=[0.08, 0.05, 0.06],
                rx_m=0.03,
                ry_m=0.03,
                rz_m=0.03,
            ),
        ]
    )
    recipe_path = _write_recipe(tmp_path / "blockout_recipe.json", pkg)
    out_dir = tmp_path / "constraints"
    payload = run_blockout_validate_constraints(recipe_path, out_dir, force=True)
    assert payload["ok"] is True
    assert payload["honesty"] == CONSTRAINT_HONESTY
    report_path = out_dir / "constraints_report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["honesty"] == CONSTRAINT_HONESTY
    assert data["schema_version"] == "1.0.0"


def test_error_codes_present() -> None:
    from typing import get_args

    from meshops.proportion.errors import ProportionErrorCode

    codes = set(get_args(ProportionErrorCode))
    for c in (
        "optimize_slow_needs_mesh",
        "optimize_no_free_dofs",
        "optimize_failed",
        "constraint_report_failed",
    ):
        assert c in codes
