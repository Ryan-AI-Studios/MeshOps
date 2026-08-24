"""Track 0128 — blockout-girdle-compare JSON / honesty / scene-dump (offline).

Authoring QA only — GIRDLE_COMPARE_HONESTY. Not mesh or print success.
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
from meshops.proportion.girdle_compare import (
    GIRDLE_COMPARE_ROLES,
    GIRDLE_COMPARE_SCHEMA_VERSION,
    SUGGESTED_ACTIONS,
    build_girdle_metrics,
    extract_recipe_girdle_part,
    run_blockout_girdle_compare,
)
from meshops.proportion.honesty import GIRDLE_COMPARE_HONESTY, PROPORTION_HONESTY
from meshops.proportion.models import (
    PROPORTION_SCHEMA_VERSION,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

runner = CliRunner()

_SUGGESTED = frozenset({"skip", "hold_priors", "soft_adjust", "session_girdle", "remake_0111"})


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
    *,
    placement: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
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
    if placement is not None:
        out["placement"] = placement
    return out


def _cylinder(
    name: str,
    role: str,
    p0: list[float],
    p1: list[float],
    radius: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "kind": "cylinder",
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


def _productish_recipe(*, include_neck: bool = True, scm_front: bool = True) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [
        _ellipsoid(
            "RECIPE_trap_soft_l",
            "trap_soft",
            [-0.1416, 0.0172, 1.3974],
            0.07224,
            0.03784,
            0.06536,
        ),
        _ellipsoid(
            "RECIPE_trap_soft_r",
            "trap_soft",
            [0.1416, 0.0172, 1.3974],
            0.07224,
            0.03784,
            0.06536,
        ),
        _capsule(
            "RECIPE_clavicle_l",
            "clavicle",
            [-0.2420, 0.0, 1.3802],
            [0.0, -0.0460, 1.3372],
            0.02064,
        ),
        _capsule(
            "RECIPE_clavicle_r",
            "clavicle",
            [0.2420, 0.0, 1.3802],
            [0.0, -0.0460, 1.3372],
            0.02064,
        ),
        _capsule(
            "RECIPE_sternomastoid_soft_l",
            "sternomastoid_soft",
            [0.0, 0.018, 1.3744],
            [0.02, -0.01, 1.50],
            0.01342,
            placement="front_plane" if scm_front else "full3d",
        ),
        _capsule(
            "RECIPE_sternomastoid_soft_r",
            "sternomastoid_soft",
            [0.0, 0.018, 1.3744],
            [-0.02, -0.01, 1.50],
            0.01342,
            placement="front_plane" if scm_front else "full3d",
        ),
        _ellipsoid(
            "RECIPE_neck_base_soft",
            "neck",
            [0.0, 0.018, 1.3744],
            0.04414,
            0.03178,
            0.01942,
        ),
        _ellipsoid(
            "RECIPE_deltoid_soft_l",
            "deltoid_soft",
            [-0.2622, 0.0, 1.3367],
            0.05911,
            0.03665,
            0.06384,
        ),
        _ellipsoid(
            "RECIPE_deltoid_soft_r",
            "deltoid_soft",
            [0.2622, 0.0, 1.3367],
            0.05911,
            0.03665,
            0.06384,
        ),
    ]
    if include_neck:
        parts.append(
            _cylinder(
                "RECIPE_neck",
                "neck",
                [0.0, 0.018, 1.3802],
                [0.0, -0.00895, 1.5070],
                0.03531,
            )
        )
    return _recipe_doc(parts=parts)


def test_b1_trap_medial_gap_when_recipe() -> None:
    """B1: recipe present → trap_medial_gap_m finite (~0.034 class)."""
    metrics = build_girdle_metrics(_report(), recipe=_productish_recipe())
    assert metrics is not None
    # (|cx| - rx) - neck.r = (0.1416 - 0.07224) - 0.03531 = 0.03405
    assert metrics.trap_medial_gap_m == pytest.approx(0.03405, abs=1e-4)
    assert metrics.trap_ry_m == pytest.approx(0.03784, abs=1e-5)
    assert metrics.trap_rz_m == pytest.approx(0.06536, abs=1e-5)
    assert metrics.clav_r_m == pytest.approx(0.02064, abs=1e-5)
    assert metrics.scm_r_m == pytest.approx(0.01342, abs=1e-5)
    assert metrics.neck_r_m == pytest.approx(0.03531, abs=1e-5)
    assert metrics.neck_base_rx_m == pytest.approx(0.04414, abs=1e-5)
    assert metrics.neck_base_ry_m == pytest.approx(0.03178, abs=1e-5)
    assert metrics.neck_base_rz_m == pytest.approx(0.01942, abs=1e-5)
    assert metrics.nape_y_m == pytest.approx(0.018, abs=1e-6)


def test_b2_front_only_y_null() -> None:
    """B2: front-only measured → y_m null on new girdle ids (no invent)."""
    report = _report(
        {
            "trap_apex_l": _lm("trap_apex_l", x_m=-0.14, z_m=1.40),
            "clav_med_l": _lm("clav_med_l", x_m=0.0, z_m=1.34),
        }
    )
    metrics = build_girdle_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("trap_apex_l") is None
    assert metrics.y_m.get("clav_med_l") is None


def test_b3_missing_one_trap_apex_no_invent() -> None:
    """B3: missing one trap_apex → no invent contralateral."""
    report = _report({"trap_apex_l": _lm("trap_apex_l", x_m=-0.14, y_m=0.02, z_m=1.40)})
    metrics = build_girdle_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("trap_apex_l") == pytest.approx(0.02, abs=1e-9)
    assert metrics.y_m.get("trap_apex_r") is None


def test_b4_proportion_report_stay_1_2_0() -> None:
    """B4: proportion report schema stay 1.2.0 (sidecar, not a 1.3.0 bump)."""
    assert PROPORTION_SCHEMA_VERSION == "1.2.0"
    report = _report()
    assert report.schema_version == "1.2.0"
    dumped = report.model_dump(mode="json")
    assert "girdle_metrics" not in dumped


def test_b5_gap_none_without_neck_r() -> None:
    """B5 / B35: neck r missing → trap_medial_gap_m is None (never invent from shoulder_*)."""
    report = _report(
        {
            "shoulder_l": _lm("shoulder_l", x_m=-0.24, z_m=1.38),
            "shoulder_r": _lm("shoulder_r", x_m=0.24, z_m=1.38),
        }
    )
    metrics = build_girdle_metrics(report, recipe=_productish_recipe(include_neck=False))
    assert metrics is not None
    assert metrics.trap_medial_gap_m is None
    assert metrics.neck_r_m is None


def test_c1_recipe_trap_and_clav_extract() -> None:
    """C1: recipe JSON extracts RECIPE_trap_soft_l axes + RECIPE_clavicle_l p0/p1/radius."""
    trap = extract_recipe_girdle_part(_productish_recipe()["parts"], "trap_apex_l")
    assert trap is not None
    assert trap["name"] == "RECIPE_trap_soft_l"
    assert trap["kind"] == "ellipsoid"
    assert trap["center"] is not None
    assert trap["center"][1] == pytest.approx(0.0172, abs=1e-6)
    assert trap["rx_m"] == pytest.approx(0.07224, abs=1e-6)
    assert trap["ry_m"] == pytest.approx(0.03784, abs=1e-6)
    assert trap["rz_m"] == pytest.approx(0.06536, abs=1e-6)
    clav = extract_recipe_girdle_part(_productish_recipe()["parts"], "clav_lat_l")
    assert clav is not None
    assert clav["name"] == "RECIPE_clavicle_l"
    assert clav["kind"] == "capsule"
    assert clav["p0"] is not None
    assert clav["p1"] is not None
    assert clav["p0"][1] == pytest.approx(0.0, abs=1e-6)
    assert clav["p1"][1] == pytest.approx(-0.0460, abs=1e-6)
    assert clav["radius_m"] == pytest.approx(0.02064, abs=1e-6)


def test_c2_missing_scene_dump_live_null(tmp_path: Path) -> None:
    """C2: missing --scene-dump → live=null (not fail)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
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
                    "RECIPE_trap_soft_l",
                    "trap_soft",
                    [-0.14, 0.05, 1.40],
                    0.07,
                    0.04,
                    0.06,
                )
            ]
        ),
    )
    payload = run_blockout_girdle_compare(
        report,
        recipe,
        tmp_path / "cmp",
        scene_dump=dump,
        force=True,
    )
    trap = next(r for r in payload["roles"] if r["id"] == "trap_apex_l")
    assert trap["live"] is not None
    assert trap["live"]["center"] is not None
    assert trap["live"]["center"][1] == pytest.approx(0.05, abs=1e-9)


def test_c3_malformed_dump_fail_closed(tmp_path: Path) -> None:
    """C3: malformed dump → ProportionError (fail-closed)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = tmp_path / "dump.json"
    dump.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ProportionError):
        run_blockout_girdle_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "girdle_compare.json").is_file()


def test_c3b_named_part_without_geometry_fail_closed(tmp_path: Path) -> None:
    """C3: named trap RECIPE dump part without center/p0/p1 fails closed."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = tmp_path / "dump.json"
    dump.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source": "scene",
                "parts": [{"name": "RECIPE_trap_soft_l"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProportionError):
        run_blockout_girdle_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "girdle_compare.json").is_file()


def test_c4_clavicle_capsule_extract_mapped_endpoint() -> None:
    """C4 / B26: capsule clavicle extract uses mapped endpoint; rx_m stays null."""
    p0 = [-0.2420, 0.0, 1.3802]
    p1 = [0.0, -0.0460, 1.3372]
    lat = extract_recipe_girdle_part(
        [_capsule("RECIPE_clavicle_l", "clavicle", p0, p1, 0.02064)],
        "clav_lat_l",
    )
    med = extract_recipe_girdle_part(
        [_capsule("RECIPE_clavicle_l", "clavicle", p0, p1, 0.02064)],
        "clav_med_l",
    )
    assert lat is not None and med is not None
    assert lat["kind"] == "capsule"
    assert lat["center"] is not None
    assert lat["center"][1] == pytest.approx(p0[1], abs=1e-9)
    assert med["center"] is not None
    assert med["center"][1] == pytest.approx(p1[1], abs=1e-9)
    assert lat["p0"] == p0
    assert lat["p1"] == p1
    assert lat["radius_m"] == pytest.approx(0.02064, abs=1e-9)
    assert lat["rx_m"] is None
    assert med["rx_m"] is None


def test_d1_honesty_schema_region(tmp_path: Path) -> None:
    """D1: payload honesty / schema 1.0.0 / region=girdle."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    assert payload["honesty"] == GIRDLE_COMPARE_HONESTY
    assert payload["schema_version"] == GIRDLE_COMPARE_SCHEMA_VERSION
    assert payload["schema_version"] == "1.0.0"
    assert payload["region"] == "girdle"
    raw = json.loads((tmp_path / "cmp" / "girdle_compare.json").read_text(encoding="utf-8"))
    assert raw["honesty"] == GIRDLE_COMPARE_HONESTY
    assert raw["region"] == "girdle"


def test_d2_suggested_closed_set(tmp_path: Path) -> None:
    """D2: each role has suggested in the closed set."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    assert SUGGESTED_ACTIONS == _SUGGESTED
    ids = {r["id"] for r in payload["roles"]}
    for lid in GIRDLE_COMPARE_ROLES:
        assert lid in ids
    for role in payload["roles"]:
        assert role["suggested"] in _SUGGESTED


def test_d3_signed_delta_mm(tmp_path: Path) -> None:
    """D3: signed delta_mm when both measured+recipe finite (ellipsoid center)."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "trap_apex_l": _lm("trap_apex_l", x_m=-0.1416, y_m=0.04, z_m=1.3974),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    trap = next(r for r in payload["roles"] if r["id"] == "trap_apex_l")
    assert trap["delta_mm"] is not None
    assert trap["delta_mm"]["y"] == pytest.approx((0.04 - 0.0172) * 1000.0, abs=1e-3)
    assert trap["suggested"] == "soft_adjust"


def test_d4_missing_id_skip_no_nan(tmp_path: Path) -> None:
    """D4: missing id → suggested skip / missing_id — no NaN."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    trap = next(r for r in payload["roles"] if r["id"] == "trap_apex_l")
    assert trap["measured"] is None
    assert trap["suggested"] == "skip"
    assert "missing_id" in trap["form_read"]
    dumped = json.dumps(payload)
    assert "NaN" not in dumped
    assert "Infinity" not in dumped


def test_d5_trap_medial_gap_token(tmp_path: Path) -> None:
    """D5: synthetic trap medial gap ≥ 0.015 → trap_medial_gap in form_read."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "trap_medial_gap" in tokens


def test_d6_cli_help_json_ok(tmp_path: Path) -> None:
    """D6: CLI help / --json path; exit 0 on structural ok."""
    help_result = runner.invoke(app, ["proportion", "blockout-girdle-compare", "--help"])
    assert help_result.exit_code == 0
    assert "blockout-girdle-compare" in help_result.output
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-girdle-compare",
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
    assert payload["region"] == "girdle"


def test_d7_stdout_honesty(tmp_path: Path) -> None:
    """D7: stdout honesty — compare is not mesh or print success."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-girdle-compare",
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
    assert GIRDLE_COMPARE_HONESTY in result.output
    assert "not mesh or print success" in result.output.lower()
    assert "Difficulty §N6" not in result.output


def test_d8_trap_towers_and_scm_front_plane(tmp_path: Path) -> None:
    """D8: synthetic trap rz ≥0.060 → trap_towers; SCM front_plane → scm_front_plane."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe(scm_front=True))
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "trap_towers" in tokens
    assert "scm_front_plane" in tokens


def test_d9_clav_scm_acromion_nape_hold_priors(tmp_path: Path) -> None:
    """D9 / B23: clav/scm/acromion/nape/trap_med/shoulder stay hold_priors."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "clav_med_l": _lm("clav_med_l", x_m=0.0, y_m=-0.02, z_m=1.34),
                "scm_origin_l": _lm("scm_origin_l", x_m=0.01, y_m=0.02, z_m=1.37),
                "acromion_l": _lm("acromion_l", x_m=-0.26, y_m=0.0, z_m=1.34),
                "nape": _lm("nape", x_m=0.0, y_m=0.04, z_m=1.38),
                "trap_med_l": _lm("trap_med_l", x_m=-0.07, y_m=0.02, z_m=1.40),
                "shoulder_l": _lm("shoulder_l", x_m=-0.24, y_m=0.0, z_m=1.38),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    for lid in (
        "clav_med_l",
        "scm_origin_l",
        "acromion_l",
        "nape",
        "trap_med_l",
        "shoulder_l",
    ):
        role = next(r for r in payload["roles"] if r["id"] == lid)
        assert role["measured"] is not None
        assert role["suggested"] == "hold_priors"


def test_d10_trap_apex_ignore_delta_x(tmp_path: Path) -> None:
    """D10 / B36: trap_apex_l large delta.x and dy=dz=0 → hold_priors (ignore X)."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "trap_apex_l": _lm("trap_apex_l", x_m=-0.40, y_m=0.0172, z_m=1.3974),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    trap = next(r for r in payload["roles"] if r["id"] == "trap_apex_l")
    assert trap["delta_mm"] is not None
    assert abs(trap["delta_mm"]["x"]) >= 1.0
    assert trap["delta_mm"]["y"] == pytest.approx(0.0, abs=1e-6)
    assert trap["delta_mm"]["z"] == pytest.approx(0.0, abs=1e-6)
    assert trap["suggested"] == "hold_priors"


def test_d11_capsule_p1_hold_priors(tmp_path: Path) -> None:
    """D11 / B39: capsule center=null + measured clav_med_l Y Δ ≥1 mm uses p1; hold_priors."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "clav_med_l": _lm("clav_med_l", x_m=0.0, y_m=-0.03, z_m=1.3372),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    clav = next(r for r in payload["roles"] if r["id"] == "clav_med_l")
    assert clav["recipe"] is not None
    assert clav["recipe"]["kind"] == "capsule"
    assert clav["delta_mm"] is not None
    # p1 Y = -0.046; always-p0 would use 0.0 and yield -30 mm.
    assert clav["delta_mm"]["y"] == pytest.approx((-0.03 - (-0.046)) * 1000.0, abs=1e-3)
    assert clav["suggested"] == "hold_priors"


def test_d12_capsule_p0_hold_priors(tmp_path: Path) -> None:
    """D12 / B39: same capsule + measured clav_lat_l Y Δ ≥1 mm uses p0 (not p1); hold_priors."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "clav_lat_l": _lm("clav_lat_l", x_m=-0.2420, y_m=0.02, z_m=1.3802),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    clav = next(r for r in payload["roles"] if r["id"] == "clav_lat_l")
    assert clav["recipe"] is not None
    assert clav["recipe"]["kind"] == "capsule"
    assert clav["delta_mm"] is not None
    # p0 Y = 0.0; wrongly using p1 (-0.046) would yield 66 mm.
    assert clav["delta_mm"]["y"] == pytest.approx((0.02 - 0.0) * 1000.0, abs=1e-3)
    assert clav["suggested"] == "hold_priors"


def test_d13_scm_and_nape_mapped_endpoints(tmp_path: Path) -> None:
    """B39: scm_origin uses p0, scm_insert uses p1, nape uses neck p0."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "scm_origin_l": _lm("scm_origin_l", x_m=0.0, y_m=0.04, z_m=1.3744),
                "scm_insert_l": _lm("scm_insert_l", x_m=0.02, y_m=0.01, z_m=1.50),
                "nape": _lm("nape", x_m=0.0, y_m=0.04, z_m=1.3802),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_girdle_compare(report, recipe, tmp_path / "cmp", force=True)
    origin = next(r for r in payload["roles"] if r["id"] == "scm_origin_l")
    insert = next(r for r in payload["roles"] if r["id"] == "scm_insert_l")
    nape = next(r for r in payload["roles"] if r["id"] == "nape")
    assert origin["delta_mm"] is not None
    assert origin["delta_mm"]["y"] == pytest.approx((0.04 - 0.018) * 1000.0, abs=1e-3)
    assert insert["delta_mm"] is not None
    assert insert["delta_mm"]["y"] == pytest.approx((0.01 - (-0.01)) * 1000.0, abs=1e-3)
    assert nape["delta_mm"] is not None
    assert nape["delta_mm"]["y"] == pytest.approx((0.04 - 0.018) * 1000.0, abs=1e-3)
    assert origin["suggested"] == "hold_priors"
    assert insert["suggested"] == "hold_priors"
    assert nape["suggested"] == "hold_priors"


def test_f1_mcp_catalog_53() -> None:
    """F1: TOOL_NAMES 53 and girdle-compare tool present."""
    assert "mesh_proportion_blockout_girdle_compare" in TOOL_NAMES
    assert len(TOOL_NAMES) == 53


def test_f2_cli_contains_verb() -> None:
    """F2: src/meshops/cli.py contains blockout-girdle-compare."""
    cli = Path("src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-girdle-compare" in cli


def test_sidecar_unlinked_when_no_finite_ids(tmp_path: Path) -> None:
    """--force with no finite girdle ids must not leave a stale sidecar."""
    report_hit = _write_report(
        tmp_path / "hit.json",
        _report({"trap_apex_l": _lm("trap_apex_l", x_m=-0.14, z_m=1.40)}),
    )
    report_miss = _write_report(tmp_path / "miss.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    run_blockout_girdle_compare(report_hit, recipe, out, force=True)
    assert (out / "girdle_metrics.json").is_file()
    run_blockout_girdle_compare(report_miss, recipe, out, force=True)
    assert not (out / "girdle_metrics.json").is_file()


def test_sidecar_girdle_metrics_when_ids_present(tmp_path: Path) -> None:
    """Compare writes girdle_metrics.json when at least one v1 id is finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report({"trap_apex_l": _lm("trap_apex_l", x_m=-0.14, z_m=1.40)}),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    payload = run_blockout_girdle_compare(report, recipe, out, force=True)
    sidecar = out / "girdle_metrics.json"
    assert sidecar.is_file()
    metrics = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metrics["trap_medial_gap_m"] == pytest.approx(0.03405, abs=1e-4)
    assert payload["girdle_metrics"]["trap_medial_gap_m"] == pytest.approx(0.03405, abs=1e-4)


def test_mcp_wrapper_calls_compare(tmp_path: Path) -> None:
    """MCP adapter reaches the same compare engine."""
    from meshops.mcp.tools import mesh_proportion_blockout_girdle_compare

    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = mesh_proportion_blockout_girdle_compare(
        tmp_path,
        report=str(report),
        recipe=str(recipe),
        out=str(tmp_path / "cmp"),
        force=True,
    )
    assert payload["ok"] is True
    assert payload["region"] == "girdle"
    assert (tmp_path / "cmp" / "girdle_compare.json").is_file()


def test_f4_honesty_token() -> None:
    """F4: GIRDLE_COMPARE_HONESTY in honesty.py."""
    from meshops.proportion import honesty as honesty_mod

    assert GIRDLE_COMPARE_HONESTY == "proportion_girdle_compare_not_mesh_or_print_success"
    assert hasattr(honesty_mod, "GIRDLE_COMPARE_HONESTY")
