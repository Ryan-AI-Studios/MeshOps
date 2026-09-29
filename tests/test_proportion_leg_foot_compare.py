"""Track 0127 — blockout-leg-foot-compare JSON / honesty / scene-dump (offline).

Authoring QA only — LEG_FOOT_COMPARE_HONESTY. Not mesh or print success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import RECIPE_ID, RECIPE_SCHEMA_VERSION
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import LEG_FOOT_COMPARE_HONESTY, PROPORTION_HONESTY
from meshops.proportion.leg_foot_compare import (
    _SOFT_ADJUST_AXES,
    LEG_FOOT_COMPARE_ROLES,
    LEG_FOOT_COMPARE_SCHEMA_VERSION,
    SUGGESTED_ACTIONS,
    build_leg_foot_metrics,
    extract_recipe_leg_foot_part,
    run_blockout_leg_foot_compare,
)
from meshops.proportion.models import (
    PROPORTION_SCHEMA_VERSION,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

runner = CliRunner()

_SUGGESTED = frozenset({"skip", "hold_priors", "soft_adjust", "session_leg_foot", "remake_0111"})


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
    ank_float: bool = True,
    plate_past: bool = True,
    pedestal: bool = False,
) -> dict[str, Any]:
    ank_z = 0.1314 if ank_float else 0.0552
    plate_cy = -0.0172 if plate_past else -0.1105
    calf_a_z = 0.5817 if pedestal else 0.5717
    return _recipe_doc(
        parts=[
            _ellipsoid(
                "RECIPE_calf_a_l",
                "limb_segment",
                [-0.0821, 0.0, calf_a_z],
                0.03853,
                0.03853,
                0.03853,
            ),
            _ellipsoid(
                "RECIPE_calf_a_r",
                "limb_segment",
                [0.0821, 0.0, calf_a_z],
                0.03853,
                0.03853,
                0.03853,
            ),
            _capsule(
                "RECIPE_calf_cyl_l",
                "limb_segment",
                [-0.0976, 0.0217, 0.5717],
                [-0.0949, 0.0424, 0.3867],
                0.05167,
            ),
            _capsule(
                "RECIPE_calf_cyl_r",
                "limb_segment",
                [0.0976, 0.0217, 0.5717],
                [0.0949, 0.0424, 0.3867],
                0.05167,
            ),
            _capsule(
                "RECIPE_calf_taper_dist_l",
                "limb_segment",
                [-0.0949, 0.0424, 0.3867],
                [-0.0911, 0.0711, 0.1314],
                0.03503,
            ),
            _capsule(
                "RECIPE_calf_taper_dist_r",
                "limb_segment",
                [0.0949, 0.0424, 0.3867],
                [0.0911, 0.0711, 0.1314],
                0.03503,
            ),
            _ellipsoid(
                "RECIPE_calf_b_l",
                "limb_segment",
                [-0.0911, 0.0711, 0.1314],
                0.03153,
                0.03153,
                0.03153,
            ),
            _ellipsoid(
                "RECIPE_calf_b_r",
                "limb_segment",
                [0.0911, 0.0711, 0.1314],
                0.03153,
                0.03153,
                0.03153,
            ),
            _ellipsoid(
                "RECIPE_ank_foot_l",
                "ankle_bridge",
                [-0.0911, 0.0711, ank_z],
                0.04237,
                0.03305,
                0.07627,
            ),
            _ellipsoid(
                "RECIPE_ank_foot_r",
                "ankle_bridge",
                [0.0911, 0.0711, ank_z],
                0.04237,
                0.03305,
                0.07627,
            ),
            _ellipsoid(
                "RECIPE_heel_l",
                "heel",
                [-0.0911, 0.0875, 0.0552],
                0.0445,
                0.0397,
                0.0381,
            ),
            _ellipsoid(
                "RECIPE_heel_r",
                "heel",
                [0.0911, 0.0875, 0.0552],
                0.0445,
                0.0397,
                0.0381,
            ),
            _ellipsoid(
                "RECIPE_foot_plate_l",
                "foot_plate",
                [-0.0911, plate_cy, 0.0301],
                0.0445,
                0.1324,
                0.0301,
            ),
            _ellipsoid(
                "RECIPE_foot_plate_r",
                "foot_plate",
                [0.0911, plate_cy, 0.0301],
                0.0445,
                0.1324,
                0.0301,
            ),
            _ellipsoid(
                "RECIPE_arch_soft_l",
                "ball_soft",
                [-0.0911, -0.0066, 0.0376],
                0.0390,
                0.0238,
                0.0346,
            ),
            _ellipsoid(
                "RECIPE_arch_soft_r",
                "ball_soft",
                [0.0911, -0.0066, 0.0376],
                0.0390,
                0.0238,
                0.0346,
            ),
            _ellipsoid(
                "RECIPE_ball_soft_l",
                "ball_soft",
                [-0.0911, -0.0613, 0.0361],
                0.0403,
                0.0212,
                0.0331,
            ),
            _ellipsoid(
                "RECIPE_ball_soft_r",
                "ball_soft",
                [0.0911, -0.0613, 0.0361],
                0.0403,
                0.0212,
                0.0331,
            ),
            _capsule(
                "RECIPE_toe_1_l",
                "toe_soft",
                [-0.0911, -0.0900, 0.0300],
                [-0.0911, -0.1105, 0.0300],
                0.0183,
            ),
            _capsule(
                "RECIPE_toe_1_r",
                "toe_soft",
                [0.0911, -0.0900, 0.0300],
                [0.0911, -0.1105, 0.0300],
                0.0183,
            ),
            _ellipsoid(
                "RECIPE_toe_tip_1_l",
                "toe_soft",
                [-0.0911, -0.1105, 0.0300],
                0.0143,
                0.00785,
                0.0143,
            ),
            _ellipsoid(
                "RECIPE_toe_tip_1_r",
                "toe_soft",
                [0.0911, -0.1105, 0.0300],
                0.0143,
                0.00785,
                0.0143,
            ),
            _ellipsoid(
                "RECIPE_knee_soft_l",
                "limb_segment",
                [-0.0821, 0.0, 0.5717],
                0.04767,
                0.03909,
                0.05482,
            ),
            _ellipsoid(
                "RECIPE_knee_soft_r",
                "limb_segment",
                [0.0821, 0.0, 0.5717],
                0.04767,
                0.03909,
                0.05482,
            ),
        ],
    )


def test_b1_ank_vs_heel_when_recipe() -> None:
    """B1: recipe present → ank_bottom_vs_heel_bottom_m finite (~0.038 class)."""
    metrics = build_leg_foot_metrics(_report(), recipe=_productish_recipe())
    assert metrics is not None
    assert metrics.ank_bottom_vs_heel_bottom_m == pytest.approx(0.03803, abs=1e-4)


def test_b2_front_only_y_null() -> None:
    """B2: front-only measured → y_m null on new leg/foot ids (no invent)."""
    report = _report(
        {
            "gastroc_med_l": _lm("gastroc_med_l", x_m=-0.10, z_m=0.50),
            "arch_apex_l": _lm("arch_apex_l", x_m=-0.09, z_m=0.04),
        }
    )
    metrics = build_leg_foot_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("gastroc_med_l") is None
    assert metrics.y_m.get("arch_apex_l") is None


def test_b3_missing_one_gastroc_no_invent() -> None:
    """B3: missing one gastroc_med → no invent contralateral."""
    report = _report({"gastroc_med_l": _lm("gastroc_med_l", x_m=-0.10, y_m=0.08, z_m=0.50)})
    metrics = build_leg_foot_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("gastroc_med_l") == pytest.approx(0.08, abs=1e-9)
    assert metrics.y_m.get("gastroc_med_r") is None


def test_b4_proportion_report_stay_1_2_0() -> None:
    """B4: proportion report schema stay 1.2.0 (sidecar, not a 1.3.0 bump)."""
    assert PROPORTION_SCHEMA_VERSION == "1.2.0"
    report = _report()
    assert report.schema_version == "1.2.0"
    dumped = report.model_dump(mode="json")
    assert "leg_foot_metrics" not in dumped


def test_b5_ball_span_none_unless_both_finite() -> None:
    """B5 / B35: one of ball_l/r missing → ball_span_m is None (never invent from toe)."""
    report = _report(
        {
            "ball_l": _lm("ball_l", x_m=-0.04, z_m=0.03),
            "toe_l": _lm("toe_l", x_m=-0.05, z_m=0.03),
            "toe_r": _lm("toe_r", x_m=0.05, z_m=0.03),
        }
    )
    metrics = build_leg_foot_metrics(report, recipe=_productish_recipe())
    assert metrics is not None
    assert metrics.ball_span_m is None
    both = _report(
        {
            "ball_l": _lm("ball_l", x_m=-0.04, z_m=0.03),
            "ball_r": _lm("ball_r", x_m=0.05, z_m=0.03),
        }
    )
    both_metrics = build_leg_foot_metrics(both, recipe=None)
    assert both_metrics is not None
    assert both_metrics.ball_span_m == pytest.approx(0.09, abs=1e-9)


def test_c1_recipe_calf_and_ank_extract() -> None:
    """C1: recipe JSON extracts RECIPE_calf_cyl_l p0/radius + RECIPE_ank_foot_l axes."""
    cyl = extract_recipe_leg_foot_part(_productish_recipe()["parts"], "gastroc_med_l")
    assert cyl is not None
    assert cyl["name"] == "RECIPE_calf_cyl_l"
    assert cyl["kind"] == "capsule"
    assert cyl["p0"] is not None
    assert cyl["p0"][1] == pytest.approx(0.0217, abs=1e-6)
    assert cyl["radius_m"] == pytest.approx(0.05167, abs=1e-6)
    ank = extract_recipe_leg_foot_part(_productish_recipe()["parts"], "ankle_l")
    assert ank is not None
    assert ank["name"] == "RECIPE_ank_foot_l"
    assert ank["center"] is not None
    assert ank["rx_m"] == pytest.approx(0.04237, abs=1e-6)
    assert ank["ry_m"] == pytest.approx(0.03305, abs=1e-6)
    assert ank["rz_m"] == pytest.approx(0.07627, abs=1e-6)


def test_c2_missing_scene_dump_live_null(tmp_path: Path) -> None:
    """C2: missing --scene-dump → live=null (not fail)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
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
                _capsule(
                    "RECIPE_calf_cyl_l",
                    "limb_segment",
                    [-0.10, 0.05, 0.57],
                    [-0.09, 0.04, 0.38],
                    0.05,
                )
            ]
        ),
    )
    payload = run_blockout_leg_foot_compare(
        report,
        recipe,
        tmp_path / "cmp",
        scene_dump=dump,
        force=True,
    )
    gastroc = next(r for r in payload["roles"] if r["id"] == "gastroc_med_l")
    assert gastroc["live"] is not None
    assert gastroc["live"]["p0"] is not None
    assert gastroc["live"]["p0"][1] == pytest.approx(0.05, abs=1e-9)


def test_c3_malformed_dump_fail_closed(tmp_path: Path) -> None:
    """C3: malformed dump → ProportionError (fail-closed)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = tmp_path / "dump.json"
    dump.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ProportionError):
        run_blockout_leg_foot_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "leg_foot_compare.json").is_file()


def test_c3b_named_part_without_geometry_fail_closed(tmp_path: Path) -> None:
    """C3: named calf RECIPE dump part without center/p0/p1 fails closed."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = tmp_path / "dump.json"
    dump.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source": "scene",
                "parts": [{"name": "RECIPE_calf_cyl_l"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProportionError):
        run_blockout_leg_foot_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "leg_foot_compare.json").is_file()


def test_c4_calf_cyl_capsule_extract_p0() -> None:
    """C4 / B26: capsule calf_cyl extract uses p0; rx_m stays null."""
    p0 = [-0.0976, 0.0217, 0.5717]
    p1 = [-0.0949, 0.0424, 0.3867]
    snap = extract_recipe_leg_foot_part(
        [_capsule("RECIPE_calf_cyl_l", "limb_segment", p0, p1, 0.05167)],
        "gastroc_med_l",
    )
    assert snap is not None
    assert snap["kind"] == "capsule"
    assert snap["center"] is not None
    assert snap["center"][1] == pytest.approx(p0[1], abs=1e-9)
    assert snap["center"][2] == pytest.approx(p0[2], abs=1e-9)
    assert snap["p0"] == p0
    assert snap["radius_m"] == pytest.approx(0.05167, abs=1e-9)
    assert snap["rx_m"] is None


def test_d1_honesty_schema_region(tmp_path: Path) -> None:
    """D1: payload honesty / schema 1.0.0 / region=leg_foot."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    assert payload["honesty"] == LEG_FOOT_COMPARE_HONESTY
    assert payload["schema_version"] == LEG_FOOT_COMPARE_SCHEMA_VERSION
    assert payload["schema_version"] == "1.0.0"
    assert payload["region"] == "leg_foot"
    raw = json.loads((tmp_path / "cmp" / "leg_foot_compare.json").read_text(encoding="utf-8"))
    assert raw["honesty"] == LEG_FOOT_COMPARE_HONESTY
    assert raw["region"] == "leg_foot"


def test_d2_suggested_closed_set(tmp_path: Path) -> None:
    """D2: each role has suggested in the closed set."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    assert SUGGESTED_ACTIONS == _SUGGESTED
    ids = {r["id"] for r in payload["roles"]}
    for lid in LEG_FOOT_COMPARE_ROLES:
        assert lid in ids
    for role in payload["roles"]:
        assert role["suggested"] in _SUGGESTED


def test_d3_signed_delta_mm(tmp_path: Path) -> None:
    """D3: signed delta_mm when both measured+recipe finite (capsule p0)."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "gastroc_med_l": _lm("gastroc_med_l", x_m=-0.10, y_m=0.04, z_m=0.5717),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    gastroc = next(r for r in payload["roles"] if r["id"] == "gastroc_med_l")
    assert gastroc["delta_mm"] is not None
    assert gastroc["delta_mm"]["y"] == pytest.approx((0.04 - 0.0217) * 1000.0, abs=1e-3)


def test_d4_missing_id_skip_no_nan(tmp_path: Path) -> None:
    """D4: missing id → suggested skip / missing_id — no NaN."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    gastroc = next(r for r in payload["roles"] if r["id"] == "gastroc_med_l")
    assert gastroc["measured"] is None
    assert gastroc["suggested"] == "skip"
    assert "missing_id" in gastroc["form_read"]
    dumped = json.dumps(payload)
    assert "NaN" not in dumped
    assert "Infinity" not in dumped


def test_d5_ank_float_above_heel(tmp_path: Path) -> None:
    """D5: synthetic ank floor - heel floor >= 0.015 -> ank_float_above_heel."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe(ank_float=True))
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "ank_float_above_heel" in tokens


def test_d6_cli_help_json_ok(tmp_path: Path) -> None:
    """D6: CLI help / --json path; exit 0 on structural ok."""
    help_result = runner.invoke(app, ["proportion", "blockout-leg-foot-compare", "--help"])
    assert help_result.exit_code == 0
    assert "blockout-leg-foot-compare" in help_result.output
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-leg-foot-compare",
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
    assert payload["region"] == "leg_foot"


def test_d7_stdout_honesty(tmp_path: Path) -> None:
    """D7: stdout honesty — compare is not mesh or print success."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-leg-foot-compare",
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
    assert LEG_FOOT_COMPARE_HONESTY in result.output
    assert "not mesh or print success" in result.output.lower()
    assert "Difficulty §N6" not in result.output


def test_d8_plate_past_toe_tips(tmp_path: Path) -> None:
    """D8: synthetic plate front past toe_tip ≥0.015 → plate_past_toe_tips."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe(plate_past=True))
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "plate_past_toe_tips" in tokens


def test_d9_malleolus_ball_achilles_hold_priors(tmp_path: Path) -> None:
    """D9 / B23: malleolus/ball/achilles stay hold_priors even when measured X finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "malleolus_med_l": _lm("malleolus_med_l", x_m=-0.12, y_m=0.07, z_m=0.13),
                "ball_l": _lm("ball_l", x_m=-0.12, y_m=-0.06, z_m=0.04),
                "achilles_l": _lm("achilles_l", x_m=-0.09, y_m=0.10, z_m=0.10),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    for lid in ("malleolus_med_l", "ball_l", "achilles_l"):
        role = next(r for r in payload["roles"] if r["id"] == lid)
        assert role["measured"] is not None
        assert role["suggested"] == "hold_priors"


def test_d10_gastroc_ignore_delta_x(tmp_path: Path) -> None:
    """D10 / B36: gastroc_med_l large delta.x and dy=dz=0 → hold_priors (ignore X)."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "gastroc_med_l": _lm("gastroc_med_l", x_m=-0.40, y_m=0.0217, z_m=0.5717),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    gastroc = next(r for r in payload["roles"] if r["id"] == "gastroc_med_l")
    assert gastroc["delta_mm"] is not None
    assert abs(gastroc["delta_mm"]["x"]) >= 1.0
    assert gastroc["delta_mm"]["y"] == pytest.approx(0.0, abs=1e-6)
    assert gastroc["delta_mm"]["z"] == pytest.approx(0.0, abs=1e-6)
    assert gastroc["suggested"] == "hold_priors"


def test_t0_soft_adjust_axes_table() -> None:
    """T0 / B1: module Final maps all four soft-adjust ids to the consumed axis."""
    assert _SOFT_ADJUST_AXES == {
        "gastroc_med_l": "y",
        "gastroc_med_r": "y",
        "arch_apex_l": "z",
        "arch_apex_r": "z",
    }


def test_t1_gastroc_l_unused_z_hold_priors(tmp_path: Path) -> None:
    """T1: gastroc_med_l |dz| ≥ 1 mm and dy = 0 → hold_priors (Y-only consume)."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "gastroc_med_l": _lm("gastroc_med_l", x_m=-0.0976, y_m=0.0217, z_m=0.40),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    gastroc = next(r for r in payload["roles"] if r["id"] == "gastroc_med_l")
    assert gastroc["delta_mm"] is not None
    assert gastroc["delta_mm"]["y"] == pytest.approx(0.0, abs=1e-6)
    assert abs(gastroc["delta_mm"]["z"]) >= 1.0
    assert gastroc["suggested"] == "hold_priors"


def test_t2_gastroc_r_y_soft_adjust(tmp_path: Path) -> None:
    """T2: gastroc_med_r |dy| ≥ 1 mm (Z large OK) → soft_adjust."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "gastroc_med_r": _lm("gastroc_med_r", x_m=0.0976, y_m=0.08, z_m=0.40),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    gastroc = next(r for r in payload["roles"] if r["id"] == "gastroc_med_r")
    assert gastroc["delta_mm"] is not None
    assert abs(gastroc["delta_mm"]["y"]) >= 1.0
    assert abs(gastroc["delta_mm"]["z"]) >= 1.0
    assert gastroc["suggested"] == "soft_adjust"


def test_t3_arch_l_unused_y_hold_priors(tmp_path: Path) -> None:
    """T3: arch_apex_l |dy| ≥ 1 mm and dz = 0 → hold_priors (Z-only consume)."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "arch_apex_l": _lm("arch_apex_l", x_m=-0.0911, y_m=0.10, z_m=0.0376),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    arch = next(r for r in payload["roles"] if r["id"] == "arch_apex_l")
    assert arch["delta_mm"] is not None
    assert abs(arch["delta_mm"]["y"]) >= 1.0
    assert arch["delta_mm"]["z"] == pytest.approx(0.0, abs=1e-6)
    assert arch["suggested"] == "hold_priors"


def test_t4_arch_r_z_soft_adjust(tmp_path: Path) -> None:
    """T4: arch_apex_r |dz| ≥ 1 mm (Y large OK) → soft_adjust."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "arch_apex_r": _lm("arch_apex_r", x_m=0.0911, y_m=0.10, z_m=0.20),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    arch = next(r for r in payload["roles"] if r["id"] == "arch_apex_r")
    assert arch["delta_mm"] is not None
    assert abs(arch["delta_mm"]["y"]) >= 1.0
    assert abs(arch["delta_mm"]["z"]) >= 1.0
    assert arch["suggested"] == "soft_adjust"


def test_t5_consumed_axis_none_hold_priors(tmp_path: Path) -> None:
    """T5: gastroc consumed Y omitted (Z far) → hold_priors."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "gastroc_med_l": _lm("gastroc_med_l", x_m=-0.0976, z_m=0.40),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    gastroc = next(r for r in payload["roles"] if r["id"] == "gastroc_med_l")
    assert gastroc["measured"] is not None
    assert gastroc["delta_mm"] is not None
    assert gastroc["delta_mm"]["y"] is None
    assert abs(gastroc["delta_mm"]["z"]) >= 1.0
    assert gastroc["suggested"] == "hold_priors"


def test_d11_capsule_p0_soft_adjust(tmp_path: Path) -> None:
    """D11 / B39: capsule center=null + measured Y Δ ≥1 mm → soft_adjust (not skip)."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "gastroc_med_l": _lm("gastroc_med_l", x_m=-0.0976, y_m=0.08, z_m=0.5717),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_leg_foot_compare(report, recipe, tmp_path / "cmp", force=True)
    gastroc = next(r for r in payload["roles"] if r["id"] == "gastroc_med_l")
    assert gastroc["recipe"] is not None
    assert gastroc["recipe"]["kind"] == "capsule"
    assert gastroc["suggested"] == "soft_adjust"


def test_f1_mcp_catalog_51() -> None:
    """F1 / T9: TOOL_NAMES stay 53 and leg-foot-compare tool present."""
    assert "mesh_proportion_blockout_leg_foot_compare" in TOOL_NAMES
    assert len(TOOL_NAMES) == 54


def test_f2_cli_contains_verb() -> None:
    """F2: src/meshops/cli.py contains blockout-leg-foot-compare."""
    cli = Path("src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-leg-foot-compare" in cli


def test_sidecar_unlinked_when_no_finite_ids(tmp_path: Path) -> None:
    """--force with no finite leg/foot ids must not leave a stale sidecar."""
    report_hit = _write_report(
        tmp_path / "hit.json",
        _report({"gastroc_med_l": _lm("gastroc_med_l", x_m=-0.10, z_m=0.50)}),
    )
    report_miss = _write_report(tmp_path / "miss.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    run_blockout_leg_foot_compare(report_hit, recipe, out, force=True)
    assert (out / "leg_foot_metrics.json").is_file()
    run_blockout_leg_foot_compare(report_miss, recipe, out, force=True)
    assert not (out / "leg_foot_metrics.json").is_file()


def test_sidecar_leg_foot_metrics_when_ids_present(tmp_path: Path) -> None:
    """Compare writes leg_foot_metrics.json when at least one v1 id is finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report({"gastroc_med_l": _lm("gastroc_med_l", x_m=-0.10, z_m=0.50)}),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    payload = run_blockout_leg_foot_compare(report, recipe, out, force=True)
    sidecar = out / "leg_foot_metrics.json"
    assert sidecar.is_file()
    metrics = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metrics["ank_bottom_vs_heel_bottom_m"] == pytest.approx(0.03803, abs=1e-4)
    assert payload["leg_foot_metrics"]["ank_bottom_vs_heel_bottom_m"] == pytest.approx(
        0.03803, abs=1e-4
    )


def test_mcp_wrapper_calls_compare(tmp_path: Path) -> None:
    """MCP adapter reaches the same compare engine."""
    from meshops.mcp.tools import mesh_proportion_blockout_leg_foot_compare

    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = mesh_proportion_blockout_leg_foot_compare(
        tmp_path,
        report=str(report),
        recipe=str(recipe),
        out=str(tmp_path / "cmp"),
        force=True,
    )
    assert payload["ok"] is True
    assert payload["region"] == "leg_foot"
    assert (tmp_path / "cmp" / "leg_foot_compare.json").is_file()


def test_f4_honesty_token() -> None:
    """F4: LEG_FOOT_COMPARE_HONESTY in honesty.py."""
    from meshops.proportion import honesty as honesty_mod

    assert LEG_FOOT_COMPARE_HONESTY == "proportion_leg_foot_compare_not_mesh_or_print_success"
    assert hasattr(honesty_mod, "LEG_FOOT_COMPARE_HONESTY")
