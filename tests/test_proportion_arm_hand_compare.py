"""Track 0129 — blockout-arm-hand-compare JSON / honesty / scene-dump (offline).

Authoring QA only — ARM_HAND_COMPARE_HONESTY. Not mesh or print success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.arm_hand_compare import (
    ARM_HAND_COMPARE_ROLES,
    ARM_HAND_COMPARE_SCHEMA_VERSION,
    BI_FRONT_PAST_FLAG_M,
    SUGGESTED_ACTIONS,
    TRI_REAR_PAST_FLAG_M,
    build_arm_hand_metrics,
    extract_recipe_arm_hand_part,
    run_blockout_arm_hand_compare,
)
from meshops.proportion.blockout_recipe import (
    BICEP_ALONG_T,
    BICEP_FRONT_PAST_M,
    RECIPE_ID,
    RECIPE_SCHEMA_VERSION,
    TRICEP_ALONG_T,
    TRICEP_REAR_PAST_M,
    build_blockout_recipe,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import ARM_HAND_COMPARE_HONESTY, PROPORTION_HONESTY
from meshops.proportion.models import (
    PROPORTION_SCHEMA_VERSION,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import build_blockout_skeleton
from test_proportion_torso_anti_tire_plus import (
    _product_class_report,
    _product_flags,
    _template,
)

runner = CliRunner()

_SUGGESTED = frozenset({"skip", "hold_priors", "soft_adjust", "session_arm_hand", "remake_0111"})


def _lm(
    lid: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
    confidence: float = 0.9,
) -> LandmarkXYZ:
    return LandmarkXYZ(
        id=lid,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        x=None,
        y=None,
        z=None,
        confidence=confidence,
        sources=["front"],
    )


def _report(
    landmarks: dict[str, LandmarkXYZ] | None = None,
    *,
    height_m: float = 1.72,
) -> ProportionReport:
    return ProportionReport(
        schema_version="1.2.0",
        honesty=PROPORTION_HONESTY,
        height_m=height_m,
        landmarks_xyz=landmarks or {},
        quality=QualityFlags(),
    )


def _write_report(path: Path, report: ProportionReport) -> Path:
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _ellipsoid(
    name: str,
    role: str,
    center: list[float],
    rx: float,
    ry: float,
    rz: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "kind": "ellipsoid",
        "center": center,
        "rx_m": rx,
        "ry_m": ry,
        "rz_m": rz,
        "label": name,
    }


def _capsule(
    name: str,
    role: str,
    p0: list[float],
    p1: list[float],
    radius: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "kind": "capsule",
        "center": None,
        "rx_m": None,
        "ry_m": None,
        "rz_m": None,
        "p0": p0,
        "p1": p1,
        "radius_m": radius,
        "label": name,
    }


def _recipe_doc(*, parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "honesty": "proportion_blockout_recipe_not_mesh_or_print_success",
        "recipe_id": RECIPE_ID,
        "axis_notes": "Z up, soles=0, +X camera-right, +Y toward camera_left",
        "height_m": 1.72,
        "head_unit_m": 0.21018,
        "parts": parts,
    }


def _write_recipe(path: Path, doc: dict[str, Any]) -> Path:
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _productish_recipe(
    *,
    include_ua_r: bool = True,
    bi_past_small: bool = False,
    thumb_forward: bool = True,
) -> dict[str, Any]:
    # Inventory-class: bicep front past ~0.010; FA dist/prox 0.70; UA r 0.04379.
    ua_r = 0.04379
    ua_mid_y = 0.0
    bi_ry = 0.03074
    bi_past = 0.003 if bi_past_small else 0.010
    tri_ry = 0.02945
    tri_past = 0.003 if bi_past_small else 0.010

    ua_p0 = [-0.2575, ua_mid_y, 1.3802]
    ua_p1 = [-0.3275, ua_mid_y, 1.2593]
    dist_p1 = [-0.3976, -0.0147, 1.1385]
    chain_mid_y = ua_p0[1] + BICEP_ALONG_T * (dist_p1[1] - ua_p0[1])
    bi_cy = chain_mid_y - ua_r - bi_past + bi_ry
    tri_cy = chain_mid_y + ua_r + tri_past - tri_ry

    thumb_p0 = [-0.45, -0.05, 0.95]
    thumb_p1 = [-0.46, -0.12, 0.90] if thumb_forward else [-0.46, 0.02, 0.90]

    parts: list[dict[str, Any]] = [
        _capsule(
            "RECIPE_limb_upper_arm_l",
            "limb_segment",
            ua_p0,
            ua_p1,
            ua_r,
        ),
        _capsule(
            "RECIPE_arm_taper_dist_ua_l",
            "limb_segment",
            [-0.3275, 0.0, 1.2593],
            dist_p1,
            0.03678,
        ),
        _capsule(
            "RECIPE_limb_forearm_l",
            "limb_segment",
            [-0.3976, -0.0293, 1.1385],
            [-0.4335, -0.0440, 1.0209],
            0.03503,
        ),
        _capsule(
            "RECIPE_arm_taper_dist_fa_l",
            "limb_segment",
            [-0.4335, -0.0440, 1.0209],
            [-0.4695, -0.0586, 0.9034],
            0.02452,
        ),
        _ellipsoid(
            "RECIPE_bicep_soft_l",
            "bicep_soft",
            [-0.3275, bi_cy, 1.2593],
            0.03415,
            bi_ry,
            0.03245,
        ),
        _ellipsoid(
            "RECIPE_triceps_soft_l",
            "limb_segment",
            [-0.3275, tri_cy, 1.2593],
            0.03591,
            tri_ry,
            0.03411,
        ),
        _ellipsoid(
            "RECIPE_elbow_soft_l",
            "limb_segment",
            [-0.3976, -0.0293, 1.1385],
            0.04487,
            0.04487,
            0.04487,
        ),
        _ellipsoid(
            "RECIPE_dist_soft_forearm_l",
            "limb_segment",
            [-0.4695, -0.0586, 0.9034],
            0.03562,
            0.03562,
            0.03562,
        ),
        _ellipsoid(
            "RECIPE_palm_l",
            "palm",
            [-0.4695, -0.0586, 0.9034],
            0.03750,
            0.03397,
            0.02903,
        ),
        _capsule(
            "RECIPE_thumb_soft_0_l",
            "thumb_soft",
            thumb_p0,
            thumb_p1,
            0.01500,
        ),
        _capsule(
            "RECIPE_thumb_soft_1_l",
            "thumb_soft",
            [-0.46, -0.12, 0.90],
            [-0.47, -0.16, 0.86],
            0.01200,
        ),
        _capsule(
            "RECIPE_finger_index_0_l",
            "finger_soft",
            [-0.48, -0.06, 0.88],
            [-0.49, -0.07, 0.84],
            0.01128,
        ),
        _capsule(
            "RECIPE_finger_pinky_0_l",
            "finger_soft",
            [-0.45, -0.05, 0.88],
            [-0.46, -0.06, 0.84],
            0.01000,
        ),
        _ellipsoid(
            "RECIPE_deltoid_soft_l",
            "deltoid_soft",
            [-0.2622, 0.0, 1.3367],
            0.05911,
            0.03665,
            0.06384,
        ),
    ]
    if not include_ua_r:
        parts[0]["radius_m"] = None
    return _recipe_doc(parts=parts)


def _emit_product(report, **flag_overrides: object):
    skel = build_blockout_skeleton(report)
    return build_blockout_recipe(
        report,
        skeleton=skel,
        template_applied=_template(),
        **_product_flags(**flag_overrides),  # type: ignore[arg-type]
    )


def test_t2_hanging_ua_chain_mid_vs_prox_mid() -> None:
    """T2: hanging UA — prox mid ≠ chain mid; past stays 0.010 on chain-placed bi/tri."""
    ua_r = 0.04379
    bi_ry = 0.03074
    tri_ry = 0.02945
    past = 0.010
    ua_p0 = [-0.2575, 0.05, 1.3802]
    ua_p1 = [-0.3275, 0.05, 1.2593]
    dist_p1 = [-0.3976, -0.05, 1.1385]
    prox_mid_y = (ua_p0[1] + ua_p1[1]) / 2.0
    chain_mid_y = ua_p0[1] + BICEP_ALONG_T * (dist_p1[1] - ua_p0[1])
    assert prox_mid_y != pytest.approx(chain_mid_y, abs=1e-4)
    bi_cy = chain_mid_y - ua_r - past + bi_ry
    tri_cy = chain_mid_y + ua_r + past - tri_ry
    recipe = _recipe_doc(
        parts=[
            _capsule("RECIPE_limb_upper_arm_l", "limb_segment", ua_p0, ua_p1, ua_r),
            _capsule(
                "RECIPE_arm_taper_dist_ua_l",
                "limb_segment",
                ua_p1,
                dist_p1,
                0.03678,
            ),
            _ellipsoid(
                "RECIPE_bicep_soft_l",
                "bicep_soft",
                [-0.3275, bi_cy, 1.2593],
                0.03415,
                bi_ry,
                0.03245,
            ),
            _ellipsoid(
                "RECIPE_triceps_soft_l",
                "limb_segment",
                [-0.3275, tri_cy, 1.2593],
                0.03591,
                tri_ry,
                0.03411,
            ),
        ]
    )
    metrics = build_arm_hand_metrics(_report(), recipe=recipe)
    assert metrics.bicep_front_past_m == pytest.approx(0.010, abs=1e-4)
    assert metrics.triceps_rear_past_m == pytest.approx(0.010, abs=1e-4)
    prox_bi_past = (prox_mid_y - ua_r) - (bi_cy - bi_ry)
    assert abs(prox_bi_past - 0.010) >= 0.005


def test_t3_no_same_side_taper_ignores_contralateral() -> None:
    """T3: missing same-side taper falls back to limb.p1; contralateral taper ignored."""
    ua_r = 0.04379
    bi_ry = 0.03074
    tri_ry = 0.02945
    past = 0.010
    ua_p0 = [-0.2575, 0.0, 1.3802]
    ua_p1 = [-0.3275, 0.0, 1.2593]
    limb_mid_y = (ua_p0[1] + ua_p1[1]) / 2.0
    bi_cy = limb_mid_y - ua_r - past + bi_ry
    tri_cy = limb_mid_y + ua_r + past - tri_ry
    recipe = _recipe_doc(
        parts=[
            _capsule("RECIPE_limb_upper_arm_l", "limb_segment", ua_p0, ua_p1, ua_r),
            _capsule(
                "RECIPE_arm_taper_dist_ua_r",
                "limb_segment",
                [0.3275, 0.0, 1.2593],
                [0.3976, -0.10, 1.1385],
                0.03678,
            ),
            _ellipsoid(
                "RECIPE_bicep_soft_l",
                "bicep_soft",
                [-0.3275, bi_cy, 1.2593],
                0.03415,
                bi_ry,
                0.03245,
            ),
            _ellipsoid(
                "RECIPE_triceps_soft_l",
                "limb_segment",
                [-0.3275, tri_cy, 1.2593],
                0.03591,
                tri_ry,
                0.03411,
            ),
        ]
    )
    metrics = build_arm_hand_metrics(_report(), recipe=recipe)
    assert metrics.bicep_front_past_m == pytest.approx(0.010, abs=1e-4)
    assert metrics.triceps_rear_past_m == pytest.approx(0.010, abs=1e-4)


def test_t4_malformed_dist_p1_falls_back_to_limb() -> None:
    """T4: same-side taper with missing p1 falls back to limb.p1."""
    ua_r = 0.04379
    bi_ry = 0.03074
    tri_ry = 0.02945
    past = 0.010
    ua_p0 = [-0.2575, 0.0, 1.3802]
    ua_p1 = [-0.3275, 0.0, 1.2593]
    limb_mid_y = (ua_p0[1] + ua_p1[1]) / 2.0
    bi_cy = limb_mid_y - ua_r - past + bi_ry
    tri_cy = limb_mid_y + ua_r + past - tri_ry
    taper = _capsule(
        "RECIPE_arm_taper_dist_ua_l",
        "limb_segment",
        ua_p1,
        [-0.3976, -0.10, 1.1385],
        0.03678,
    )
    taper["p1"] = None
    recipe = _recipe_doc(
        parts=[
            _capsule("RECIPE_limb_upper_arm_l", "limb_segment", ua_p0, ua_p1, ua_r),
            taper,
            _ellipsoid(
                "RECIPE_bicep_soft_l",
                "bicep_soft",
                [-0.3275, bi_cy, 1.2593],
                0.03415,
                bi_ry,
                0.03245,
            ),
            _ellipsoid(
                "RECIPE_triceps_soft_l",
                "limb_segment",
                [-0.3275, tri_cy, 1.2593],
                0.03591,
                tri_ry,
                0.03411,
            ),
        ]
    )
    metrics = build_arm_hand_metrics(_report(), recipe=recipe)
    assert metrics.bicep_front_past_m == pytest.approx(0.010, abs=1e-4)
    assert metrics.triceps_rear_past_m == pytest.approx(0.010, abs=1e-4)


def test_t5_product_emit_past_matches_0063(tmp_path: Path) -> None:
    """T5: product-class emit pasts ≈0.010; no bi_front_past / tri_rear_past tokens."""
    report = _product_class_report()
    pkg = _emit_product(report)
    metrics = build_arm_hand_metrics(report, recipe=pkg)
    assert metrics.bicep_front_past_m == pytest.approx(0.010, abs=1.5e-3)
    assert metrics.triceps_rear_past_m == pytest.approx(0.010, abs=1.5e-3)
    recipe = _write_recipe(tmp_path / "recipe.json", pkg.model_dump(mode="json"))
    payload = run_blockout_arm_hand_compare(
        _write_report(tmp_path / "report.json", report),
        recipe,
        tmp_path / "cmp",
        force=True,
    )
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "bi_front_past" not in tokens
    assert "tri_rear_past" not in tokens


def test_t6_degenerate_ua_past_none() -> None:
    """T6: near-zero UA segment → pasts None (never invent)."""
    ua_r = 0.04379
    p = [-0.2575, 0.0, 1.3802]
    recipe = _recipe_doc(
        parts=[
            _capsule("RECIPE_limb_upper_arm_l", "limb_segment", p, list(p), ua_r),
            _ellipsoid(
                "RECIPE_bicep_soft_l",
                "bicep_soft",
                [-0.3275, -0.02, 1.2593],
                0.03415,
                0.03074,
                0.03245,
            ),
            _ellipsoid(
                "RECIPE_triceps_soft_l",
                "limb_segment",
                [-0.3275, 0.02, 1.2593],
                0.03591,
                0.02945,
                0.03411,
            ),
        ]
    )
    metrics = build_arm_hand_metrics(_report(), recipe=recipe)
    assert metrics.bicep_front_past_m is None
    assert metrics.triceps_rear_past_m is None


def test_t9_const_hold() -> None:
    """T9: 0063 past consts / along_t / compare flags hold."""
    assert BICEP_FRONT_PAST_M == 0.010
    assert TRICEP_REAR_PAST_M == 0.010
    assert BICEP_ALONG_T == 0.50
    assert TRICEP_ALONG_T == 0.50
    assert BI_FRONT_PAST_FLAG_M == 0.006
    assert TRI_REAR_PAST_FLAG_M == 0.006


def test_b1_bicep_front_past_when_recipe() -> None:
    """B1: recipe present → bicep_front_past_m finite (~0.010 class)."""
    metrics = build_arm_hand_metrics(_report(), recipe=_productish_recipe())
    assert metrics is not None
    assert metrics.bicep_front_past_m == pytest.approx(0.010, abs=1e-4)
    assert metrics.triceps_rear_past_m == pytest.approx(0.010, abs=1e-4)
    assert metrics.ua_prox_r_m == pytest.approx(0.04379, abs=1e-5)
    assert metrics.ua_dist_r_m == pytest.approx(0.03678, abs=1e-5)
    assert metrics.fa_prox_r_m == pytest.approx(0.03503, abs=1e-5)
    assert metrics.fa_dist_r_m == pytest.approx(0.02452, abs=1e-5)
    assert metrics.bicep_rx_m == pytest.approx(0.03415, abs=1e-5)
    assert metrics.triceps_rx_m == pytest.approx(0.03591, abs=1e-5)
    assert metrics.palm_rx_m == pytest.approx(0.03750, abs=1e-5)
    assert metrics.palm_ry_m == pytest.approx(0.03397, abs=1e-5)


def test_b2_front_only_y_null() -> None:
    """B2: front-only measured → y_m null on new arm/hand ids (no invent)."""
    report = _report(
        {
            "bi_belly_l": _lm("bi_belly_l", x_m=-0.33, z_m=1.26),
            "humeral_head_l": _lm("humeral_head_l", x_m=-0.26, z_m=1.38),
        }
    )
    metrics = build_arm_hand_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("bi_belly_l") is None
    assert metrics.y_m.get("humeral_head_l") is None


def test_b3_missing_one_bi_belly_no_invent() -> None:
    """B3: missing one bi_belly → no invent contralateral."""
    report = _report({"bi_belly_l": _lm("bi_belly_l", x_m=-0.33, y_m=-0.04, z_m=1.26)})
    metrics = build_arm_hand_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("bi_belly_l") == pytest.approx(-0.04, abs=1e-9)
    assert metrics.y_m.get("bi_belly_r") is None


def test_b4_proportion_report_stay_1_2_0() -> None:
    """B4: proportion report schema stay 1.2.0 (sidecar, not a 1.3.0 bump)."""
    assert PROPORTION_SCHEMA_VERSION == "1.2.0"
    report = _report()
    assert report.schema_version == "1.2.0"
    dumped = report.model_dump(mode="json")
    assert "arm_hand_metrics" not in dumped


def test_b5_past_none_without_ua_r() -> None:
    """B5 / B35: UA r missing → bicep_front_past_m is None (never invent)."""
    report = _report(
        {
            "upper_arm_l": _lm("upper_arm_l", x_m=-0.30, z_m=1.30),
        }
    )
    metrics = build_arm_hand_metrics(report, recipe=_productish_recipe(include_ua_r=False))
    assert metrics is not None
    assert metrics.bicep_front_past_m is None
    assert metrics.triceps_rear_past_m is None
    assert metrics.ua_prox_r_m is None


def test_c1_recipe_bicep_and_ua_extract() -> None:
    """C1: recipe JSON extracts RECIPE_bicep_soft_l axes + RECIPE_limb_upper_arm_l p0/p1."""
    bi = extract_recipe_arm_hand_part(_productish_recipe()["parts"], "bi_belly_l")
    assert bi is not None
    assert bi["name"] == "RECIPE_bicep_soft_l"
    assert bi["kind"] == "ellipsoid"
    assert bi["center"] is not None
    assert bi["rx_m"] == pytest.approx(0.03415, abs=1e-6)
    assert bi["ry_m"] == pytest.approx(0.03074, abs=1e-6)
    ua = extract_recipe_arm_hand_part(_productish_recipe()["parts"], "humeral_head_l")
    assert ua is not None
    assert ua["name"] == "RECIPE_limb_upper_arm_l"
    assert ua["kind"] == "capsule"
    assert ua["p0"] is not None
    assert ua["p1"] is not None
    assert ua["radius_m"] == pytest.approx(0.04379, abs=1e-6)


def test_c2_missing_scene_dump_live_null(tmp_path: Path) -> None:
    """C2: missing --scene-dump → live=null (not fail)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    assert payload["ok"] is True
    for role in payload["roles"]:
        assert role["live"] is None


def test_c2b_valid_dump_live_populated(tmp_path: Path) -> None:
    """Valid --scene-dump overlays live recipe extract (not null)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = _write_recipe(
        tmp_path / "dump.json",
        _recipe_doc(
            parts=[
                _ellipsoid(
                    "RECIPE_bicep_soft_l",
                    "bicep_soft",
                    [-0.33, -0.05, 1.26],
                    0.034,
                    0.031,
                    0.032,
                )
            ]
        ),
    )
    payload = run_blockout_arm_hand_compare(
        report,
        recipe,
        tmp_path / "cmp",
        scene_dump=dump,
        force=True,
    )
    bi = next(r for r in payload["roles"] if r["id"] == "bi_belly_l")
    assert bi["live"] is not None
    assert bi["live"]["center"] is not None
    assert bi["live"]["center"][1] == pytest.approx(-0.05, abs=1e-9)


def test_c3_malformed_dump_fail_closed(tmp_path: Path) -> None:
    """C3: malformed dump → ProportionError (fail-closed)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = tmp_path / "dump.json"
    dump.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ProportionError):
        run_blockout_arm_hand_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "arm_hand_compare.json").is_file()


def test_c4_capsule_extract_mapped_endpoint() -> None:
    """C4 / B26: capsule UA extract uses mapped p0; rx_m stays null."""
    p0 = [-0.2575, 0.0, 1.3802]
    p1 = [-0.3275, 0.0, 1.2593]
    head = extract_recipe_arm_hand_part(
        [_capsule("RECIPE_limb_upper_arm_l", "limb_segment", p0, p1, 0.04379)],
        "humeral_head_l",
    )
    assert head is not None
    assert head["kind"] == "capsule"
    assert head["center"] is not None
    assert head["center"][1] == pytest.approx(p0[1], abs=1e-9)
    assert head["p0"] == p0
    assert head["p1"] == p1
    assert head["radius_m"] == pytest.approx(0.04379, abs=1e-9)
    assert head["rx_m"] is None


def test_c5_triceps_by_name_limb_segment() -> None:
    """C5 / B34: triceps role=limb_segment found by name RECIPE_triceps_soft_l."""
    tri = extract_recipe_arm_hand_part(_productish_recipe()["parts"], "tri_belly_l")
    assert tri is not None
    assert tri["name"] == "RECIPE_triceps_soft_l"
    assert tri["role"] == "limb_segment"
    assert tri["center"] is not None


def test_d1_honesty_schema_region(tmp_path: Path) -> None:
    """D1: payload honesty / schema 1.0.0 / region=arm_hand."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    assert payload["honesty"] == ARM_HAND_COMPARE_HONESTY
    assert payload["schema_version"] == ARM_HAND_COMPARE_SCHEMA_VERSION
    assert payload["schema_version"] == "1.0.0"
    assert payload["region"] == "arm_hand"
    raw = json.loads((tmp_path / "cmp" / "arm_hand_compare.json").read_text(encoding="utf-8"))
    assert raw["honesty"] == ARM_HAND_COMPARE_HONESTY
    assert raw["region"] == "arm_hand"


def test_d2_suggested_closed_set(tmp_path: Path) -> None:
    """D2: each role has suggested in the closed set."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    assert SUGGESTED_ACTIONS == _SUGGESTED
    ids = {r["id"] for r in payload["roles"]}
    for lid in ARM_HAND_COMPARE_ROLES:
        assert lid in ids
    for role in payload["roles"]:
        assert role["suggested"] in _SUGGESTED


def test_d3_signed_delta_mm(tmp_path: Path) -> None:
    """D3: signed delta_mm when both measured+recipe finite (ellipsoid center)."""
    doc = _productish_recipe()
    bi = next(p for p in doc["parts"] if p["name"] == "RECIPE_bicep_soft_l")
    assert bi["center"] is not None
    cy = float(bi["center"][1])
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "bi_belly_l": _lm("bi_belly_l", x_m=-0.3275, y_m=cy + 0.02, z_m=1.2593),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", doc)
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    role = next(r for r in payload["roles"] if r["id"] == "bi_belly_l")
    assert role["delta_mm"] is not None
    assert role["delta_mm"]["y"] == pytest.approx(20.0, abs=1e-3)
    assert role["suggested"] == "soft_adjust"


def test_d4_missing_id_skip_no_nan(tmp_path: Path) -> None:
    """D4: missing id → suggested skip / missing_id — no NaN."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    bi = next(r for r in payload["roles"] if r["id"] == "bi_belly_l")
    assert bi["measured"] is None
    assert bi["suggested"] == "skip"
    assert "missing_id" in bi["form_read"]
    dumped = json.dumps(payload)
    assert "NaN" not in dumped
    assert "Infinity" not in dumped


def test_d5_fa_stepped_token(tmp_path: Path) -> None:
    """D5: synthetic FA dist/prox ≤ 0.78 → fa_stepped in form_read."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "fa_stepped" in tokens


def test_d6_cli_help_json_ok(tmp_path: Path) -> None:
    """D6: CLI help / --json path; exit 0 on structural ok."""
    help_result = runner.invoke(app, ["proportion", "blockout-arm-hand-compare", "--help"])
    assert help_result.exit_code == 0
    assert "blockout-arm-hand-compare" in help_result.output
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-arm-hand-compare",
            "--report",
            str(report),
            "--recipe",
            str(recipe),
            "--out",
            str(tmp_path / "cmp"),
            "--force",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["region"] == "arm_hand"


def test_d7_stdout_honesty(tmp_path: Path) -> None:
    """D7: stdout honesty — compare is not mesh or print success."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-arm-hand-compare",
            "--report",
            str(report),
            "--recipe",
            str(recipe),
            "--out",
            str(tmp_path / "cmp"),
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert ARM_HAND_COMPARE_HONESTY in result.output
    assert "not mesh or print success" in result.output.lower()
    assert "Difficulty §N6" not in result.output


def test_d8_bi_front_past_and_thumb_not_forward(tmp_path: Path) -> None:
    """D8: synthetic bicep past <0.006 -> bi_front_past; thumb axis Y >= -0.40."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(
        tmp_path / "recipe.json",
        _productish_recipe(bi_past_small=True, thumb_forward=False),
    )
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "bi_front_past" in tokens
    assert "tri_rear_past" in tokens
    assert "thumb_not_forward" in tokens


def test_d9_compare_only_ids_hold_priors(tmp_path: Path) -> None:
    """D9 / B23: humeral/olecranon/fa/palm/thumb/mcp/HAVE stay hold_priors."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "humeral_head_l": _lm("humeral_head_l", x_m=-0.2575, y_m=0.02, z_m=1.3802),
                "olecranon_l": _lm("olecranon_l", x_m=-0.3976, y_m=-0.01, z_m=1.1385),
                "fa_belly_l": _lm("fa_belly_l", x_m=-0.415, y_m=-0.02, z_m=1.08),
                "palm_center_l": _lm("palm_center_l", x_m=-0.4695, y_m=-0.04, z_m=0.9034),
                "thumb_cmc_l": _lm("thumb_cmc_l", x_m=-0.45, y_m=-0.03, z_m=0.95),
                "mcp_index_l": _lm("mcp_index_l", x_m=-0.48, y_m=-0.04, z_m=0.88),
                "shoulder_l": _lm("shoulder_l", x_m=-0.2622, y_m=0.01, z_m=1.3367),
                "elbow_l": _lm("elbow_l", x_m=-0.3976, y_m=-0.01, z_m=1.1385),
                "wrist_l": _lm("wrist_l", x_m=-0.4695, y_m=-0.04, z_m=0.9034),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    for lid in (
        "humeral_head_l",
        "olecranon_l",
        "fa_belly_l",
        "palm_center_l",
        "thumb_cmc_l",
        "mcp_index_l",
        "shoulder_l",
        "elbow_l",
        "wrist_l",
    ):
        role = next(r for r in payload["roles"] if r["id"] == lid)
        assert role["measured"] is not None
        assert role["suggested"] == "hold_priors"


def test_d10_bi_belly_ignore_delta_x(tmp_path: Path) -> None:
    """D10 / B36: bi_belly_l large delta.x and dy=dz=0 → hold_priors (ignore X)."""
    doc = _productish_recipe()
    bi = next(p for p in doc["parts"] if p["name"] == "RECIPE_bicep_soft_l")
    cy = float(bi["center"][1])
    cz = float(bi["center"][2])
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "bi_belly_l": _lm("bi_belly_l", x_m=-0.60, y_m=cy, z_m=cz),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", doc)
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    role = next(r for r in payload["roles"] if r["id"] == "bi_belly_l")
    assert role["delta_mm"] is not None
    assert abs(role["delta_mm"]["x"]) >= 1.0
    assert role["delta_mm"]["y"] == pytest.approx(0.0, abs=1e-6)
    assert role["delta_mm"]["z"] == pytest.approx(0.0, abs=1e-6)
    assert role["suggested"] == "hold_priors"


def test_d11_capsule_p0_humeral_hold_priors(tmp_path: Path) -> None:
    """D11 / B39: capsule center=null + measured humeral_head_l uses p0; hold_priors."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "humeral_head_l": _lm("humeral_head_l", x_m=-0.2575, y_m=0.02, z_m=1.3802),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    head = next(r for r in payload["roles"] if r["id"] == "humeral_head_l")
    assert head["recipe"] is not None
    assert head["recipe"]["kind"] == "capsule"
    assert head["delta_mm"] is not None
    # p0 Y = 0.0; always-p0 or mid both 0 here — Δy = 20 mm from measured 0.02.
    assert head["delta_mm"]["y"] == pytest.approx(20.0, abs=1e-3)
    assert head["suggested"] == "hold_priors"


def test_d12_capsule_fa_belly_midpoint(tmp_path: Path) -> None:
    """D12 / B39: FA capsule + measured fa_belly_l uses midpoint (not p0); hold_priors."""
    p0 = [-0.3976, -0.0293, 1.1385]
    p1 = [-0.4335, -0.0440, 1.0209]
    mid_y = (p0[1] + p1[1]) / 2.0
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "fa_belly_l": _lm("fa_belly_l", x_m=-0.41555, y_m=mid_y + 0.02, z_m=1.0797),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    fa = next(r for r in payload["roles"] if r["id"] == "fa_belly_l")
    assert fa["recipe"] is not None
    assert fa["recipe"]["kind"] == "capsule"
    assert fa["delta_mm"] is not None
    # Midpoint Y; wrongly using p0 (-0.0293) would yield a different delta.
    assert fa["delta_mm"]["y"] == pytest.approx(20.0, abs=1e-3)
    # Confirm not p0: measured_y - p0_y would be mid_y+0.02 - (-0.0293) ≠ 0.02.
    assert fa["delta_mm"]["y"] != pytest.approx(((mid_y + 0.02) - p0[1]) * 1000.0, abs=1e-3)
    assert fa["suggested"] == "hold_priors"


def test_d13_thumb_tip_p1(tmp_path: Path) -> None:
    """D13 / B39: thumb_soft_1 capsule + measured thumb_tip_l uses p1 (not p0)."""
    tip_p1 = [-0.47, -0.16, 0.86]
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "thumb_tip_l": _lm("thumb_tip_l", x_m=-0.47, y_m=-0.14, z_m=0.86),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_arm_hand_compare(report, recipe, tmp_path / "cmp", force=True)
    tip = next(r for r in payload["roles"] if r["id"] == "thumb_tip_l")
    assert tip["recipe"] is not None
    assert tip["recipe"]["kind"] == "capsule"
    assert tip["delta_mm"] is not None
    # p1 Y = -0.16; wrongly using CMC/p0 (-0.12 on soft_1 p0) would differ.
    assert tip["delta_mm"]["y"] == pytest.approx((-0.14 - tip_p1[1]) * 1000.0, abs=1e-3)
    assert tip["suggested"] == "hold_priors"


def test_f1_mcp_catalog_53() -> None:
    """F1: TOOL_NAMES 53 and arm-hand-compare tool present."""
    assert "mesh_proportion_blockout_arm_hand_compare" in TOOL_NAMES
    assert len(TOOL_NAMES) == 54


def test_f2_cli_contains_verb() -> None:
    """F2: src/meshops/cli.py contains blockout-arm-hand-compare."""
    cli = Path("src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-arm-hand-compare" in cli


def test_sidecar_unlinked_when_no_finite_ids(tmp_path: Path) -> None:
    """--force with no finite arm/hand ids must not leave a stale sidecar."""
    report_hit = _write_report(
        tmp_path / "hit.json",
        _report({"bi_belly_l": _lm("bi_belly_l", x_m=-0.33, z_m=1.26)}),
    )
    report_miss = _write_report(tmp_path / "miss.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    run_blockout_arm_hand_compare(report_hit, recipe, out, force=True)
    assert (out / "arm_hand_metrics.json").is_file()
    run_blockout_arm_hand_compare(report_miss, recipe, out, force=True)
    assert not (out / "arm_hand_metrics.json").is_file()


def test_sidecar_arm_hand_metrics_when_ids_present(tmp_path: Path) -> None:
    """Compare writes arm_hand_metrics.json when at least one v1 id is finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report({"bi_belly_l": _lm("bi_belly_l", x_m=-0.33, z_m=1.26)}),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    payload = run_blockout_arm_hand_compare(report, recipe, out, force=True)
    sidecar = out / "arm_hand_metrics.json"
    assert sidecar.is_file()
    metrics = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metrics["bicep_front_past_m"] == pytest.approx(0.010, abs=1e-4)
    assert payload["arm_hand_metrics"]["bicep_front_past_m"] == pytest.approx(0.010, abs=1e-4)


def test_mcp_wrapper_calls_compare(tmp_path: Path) -> None:
    """MCP adapter reaches the same compare engine."""
    from meshops.mcp.tools import mesh_proportion_blockout_arm_hand_compare

    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = mesh_proportion_blockout_arm_hand_compare(
        tmp_path,
        report=str(report),
        recipe=str(recipe),
        out=str(tmp_path / "cmp"),
        force=True,
    )
    assert payload["ok"] is True
    assert payload["region"] == "arm_hand"
    assert (tmp_path / "cmp" / "arm_hand_compare.json").is_file()


def test_f4_honesty_token() -> None:
    """F4: ARM_HAND_COMPARE_HONESTY in honesty.py."""
    from meshops.proportion import honesty as honesty_mod

    assert ARM_HAND_COMPARE_HONESTY == "proportion_arm_hand_compare_not_mesh_or_print_success"
    assert hasattr(honesty_mod, "ARM_HAND_COMPARE_HONESTY")
