"""Track 0126 — blockout-hip-glute-compare JSON / honesty / scene-dump (offline).

Authoring QA only — HIP_GLUTE_COMPARE_HONESTY. Not mesh or print success.
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
from meshops.proportion.hip_glute_compare import (
    HIP_GLUTE_COMPARE_ROLES,
    HIP_GLUTE_COMPARE_SCHEMA_VERSION,
    SUGGESTED_ACTIONS,
    build_hip_glute_metrics,
    extract_recipe_hip_glute_part,
    run_blockout_hip_glute_compare,
)
from meshops.proportion.honesty import HIP_GLUTE_COMPARE_HONESTY, PROPORTION_HONESTY
from meshops.proportion.models import (
    PROPORTION_SCHEMA_VERSION,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

runner = CliRunner()

_SUGGESTED = frozenset({"skip", "hold_priors", "soft_adjust", "session_hip_glute", "remake_0111"})


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


def _productish_recipe(*, short_outer: bool = True, pelvis_behind: bool = True) -> dict[str, Any]:
    hip_soft_rx = 0.07050 if short_outer else 0.050
    pelvis_cy = 0.03475 if pelvis_behind else -0.05
    return _recipe_doc(
        parts=[
            _ellipsoid(
                "RECIPE_torso_oval_hip",
                "torso",
                [0.0, 0.02935, 1.00723],
                0.1979,
                0.08895,
                0.11476,
            ),
            _ellipsoid(
                "RECIPE_pelvis_oval",
                "pelvis",
                [0.0, pelvis_cy, 0.83324],
                0.2224,
                0.08339,
                0.07224,
            ),
            _ellipsoid(
                "RECIPE_glute_soft_l",
                "glute_soft",
                [-0.13075, 0.045, 0.84184],
                0.0890,
                0.1212,
                0.08725,
            ),
            _ellipsoid(
                "RECIPE_glute_soft_r",
                "glute_soft",
                [0.13075, 0.045, 0.84184],
                0.0890,
                0.1212,
                0.08725,
            ),
            _ellipsoid(
                "RECIPE_hip_soft_l",
                "limb_segment",
                [-0.22244, 0.00846, 0.86420],
                hip_soft_rx,
                0.04371,
                0.07050,
            ),
            _ellipsoid(
                "RECIPE_hip_soft_r",
                "limb_segment",
                [0.22244, 0.00846, 0.86420],
                hip_soft_rx,
                0.04371,
                0.07050,
            ),
            _capsule(
                "RECIPE_limb_thigh_l",
                "limb_segment",
                [-0.13, 0.0, 0.84],
                [-0.13, 0.0, 0.50],
                0.0613,
            ),
            _capsule(
                "RECIPE_limb_thigh_r",
                "limb_segment",
                [0.13, 0.0, 0.84],
                [0.13, 0.0, 0.50],
                0.0613,
            ),
            _capsule(
                "RECIPE_thigh_taper_dist_l",
                "limb_segment",
                [-0.13, 0.0, 0.50],
                [-0.13, 0.0, 0.42],
                0.04414,
            ),
            _capsule(
                "RECIPE_thigh_taper_dist_r",
                "limb_segment",
                [0.13, 0.0, 0.50],
                [0.13, 0.0, 0.42],
                0.04414,
            ),
        ],
    )


def test_b1_outer_delta_when_recipe() -> None:
    """B1: recipe present → glute vs hip_soft outer finite (inventory-class ~0.073)."""
    metrics = build_hip_glute_metrics(_report(), recipe=_productish_recipe())
    assert metrics is not None
    assert metrics.glute_outer_vs_hip_soft_outer_m == pytest.approx(0.07319, abs=1e-4)


def test_b2_front_only_y_null() -> None:
    """B2: front-only measured → y_m null on new hip/glute ids (no invent)."""
    report = _report(
        {
            "asis_l": _lm("asis_l", x_m=-0.10, z_m=0.95),
            "glute_bottom_l": _lm("glute_bottom_l", x_m=-0.13, z_m=0.84),
        }
    )
    metrics = build_hip_glute_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("asis_l") is None
    assert metrics.y_m.get("glute_bottom_l") is None


def test_b3_missing_one_glute_bottom_no_invent() -> None:
    """B3: missing one glute_bottom → no invent contralateral."""
    report = _report({"glute_bottom_l": _lm("glute_bottom_l", x_m=-0.13, y_m=0.08, z_m=0.84)})
    metrics = build_hip_glute_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("glute_bottom_l") == pytest.approx(0.08, abs=1e-9)
    assert metrics.y_m.get("glute_bottom_r") is None


def test_b4_proportion_report_stay_1_2_0() -> None:
    """B4: proportion report schema stay 1.2.0 (sidecar, not a 1.3.0 bump)."""
    assert PROPORTION_SCHEMA_VERSION == "1.2.0"
    report = _report()
    assert report.schema_version == "1.2.0"
    dumped = report.model_dump(mode="json")
    assert "hip_glute_metrics" not in dumped


def test_b5_thigh_gap_none_unless_both_finite() -> None:
    """B5 / B35: one of thigh_medial_l/r missing → thigh_gap_m is None."""
    report = _report({"thigh_medial_l": _lm("thigh_medial_l", x_m=-0.04, z_m=0.80)})
    metrics = build_hip_glute_metrics(report, recipe=_productish_recipe())
    assert metrics is not None
    assert metrics.thigh_gap_m is None
    both = _report(
        {
            "thigh_medial_l": _lm("thigh_medial_l", x_m=-0.04, z_m=0.80),
            "thigh_medial_r": _lm("thigh_medial_r", x_m=0.05, z_m=0.80),
        }
    )
    both_metrics = build_hip_glute_metrics(both, recipe=None)
    assert both_metrics is not None
    assert both_metrics.thigh_gap_m == pytest.approx(0.09, abs=1e-9)


def test_c1_recipe_glute_extract() -> None:
    """C1: recipe JSON extracts RECIPE_glute_soft_l center/rx/ry/rz."""
    snap = extract_recipe_hip_glute_part(_productish_recipe()["parts"], "glute_bottom_l")
    assert snap is not None
    assert snap["name"] == "RECIPE_glute_soft_l"
    assert snap["center"] is not None
    assert snap["center"][1] == pytest.approx(0.045, abs=1e-6)
    assert snap["rx_m"] == pytest.approx(0.0890, abs=1e-6)
    assert snap["ry_m"] == pytest.approx(0.1212, abs=1e-6)
    assert snap["rz_m"] == pytest.approx(0.08725, abs=1e-6)


def test_b_measured_seat_y_recomputes_surface_past() -> None:
    """Measured glute_bottom Y overlays seat then recomputes surface past (+ry)."""
    report = _report({"glute_bottom_l": _lm("glute_bottom_l", x_m=-0.13, y_m=0.18, z_m=0.84)})
    metrics = build_hip_glute_metrics(report, recipe=_productish_recipe())
    assert metrics is not None
    assert metrics.glute_seat_y_m == pytest.approx(0.18, abs=1e-9)
    hip_rear = 0.02935 + 0.08895
    glute_rear = 0.18 + 0.1212
    assert metrics.glute_rear_past_hip_oval_m == pytest.approx(glute_rear - hip_rear, abs=1e-4)
    pelvis_front = 0.03475 - 0.08339
    glute_front = 0.18 - 0.1212
    assert metrics.pelvis_front_vs_glute_front_m == pytest.approx(
        pelvis_front - glute_front, abs=1e-4
    )


def test_c2_missing_scene_dump_live_null(tmp_path: Path) -> None:
    """C2: missing --scene-dump → live=null (not fail)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_hip_glute_compare(report, recipe, tmp_path / "cmp", force=True)
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
                    "RECIPE_glute_soft_l",
                    "glute_soft",
                    [-0.12, 0.06, 0.85],
                    0.08,
                    0.11,
                    0.08,
                )
            ]
        ),
    )
    payload = run_blockout_hip_glute_compare(
        report,
        recipe,
        tmp_path / "cmp",
        scene_dump=dump,
        force=True,
    )
    glute = next(r for r in payload["roles"] if r["id"] == "glute_bottom_l")
    assert glute["live"] is not None
    assert glute["live"]["center"] is not None
    assert glute["live"]["center"][1] == pytest.approx(0.06, abs=1e-9)


def test_c3_malformed_dump_fail_closed(tmp_path: Path) -> None:
    """C3: malformed dump → ProportionError (fail-closed)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = tmp_path / "dump.json"
    dump.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ProportionError):
        run_blockout_hip_glute_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "hip_glute_compare.json").is_file()


def test_c3b_named_part_without_geometry_fail_closed(tmp_path: Path) -> None:
    """C3: named hip/glute RECIPE dump part without center/p0/p1 fails closed."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = tmp_path / "dump.json"
    dump.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source": "scene",
                "parts": [{"name": "RECIPE_glute_soft_l"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProportionError):
        run_blockout_hip_glute_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "hip_glute_compare.json").is_file()


def test_c4_thigh_capsule_extract_p0() -> None:
    """C4 / B26: capsule thigh extract uses p0; rx_m stays null."""
    p0 = [-0.13, 0.0, 0.84]
    p1 = [-0.13, 0.0, 0.50]
    snap = extract_recipe_hip_glute_part(
        [_capsule("RECIPE_limb_thigh_l", "limb_segment", p0, p1, 0.0613)],
        "thigh_medial_l",
    )
    assert snap is not None
    assert snap["kind"] == "capsule"
    assert snap["center"] is not None
    assert snap["center"][1] == pytest.approx(p0[1], abs=1e-9)
    assert snap["center"][2] == pytest.approx(p0[2], abs=1e-9)
    assert snap["radius_m"] == pytest.approx(0.0613, abs=1e-9)
    assert snap["rx_m"] is None


def test_d1_honesty_schema_region(tmp_path: Path) -> None:
    """D1: payload honesty / schema 1.0.0 / region=hip_glute."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_hip_glute_compare(report, recipe, tmp_path / "cmp", force=True)
    assert payload["honesty"] == HIP_GLUTE_COMPARE_HONESTY
    assert payload["schema_version"] == HIP_GLUTE_COMPARE_SCHEMA_VERSION
    assert payload["schema_version"] == "1.0.0"
    assert payload["region"] == "hip_glute"
    raw = json.loads((tmp_path / "cmp" / "hip_glute_compare.json").read_text(encoding="utf-8"))
    assert raw["honesty"] == HIP_GLUTE_COMPARE_HONESTY
    assert raw["region"] == "hip_glute"


def test_d2_suggested_closed_set(tmp_path: Path) -> None:
    """D2: each role has suggested in the closed set."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_hip_glute_compare(report, recipe, tmp_path / "cmp", force=True)
    assert SUGGESTED_ACTIONS == _SUGGESTED
    ids = {r["id"] for r in payload["roles"]}
    for lid in HIP_GLUTE_COMPARE_ROLES:
        assert lid in ids
    for role in payload["roles"]:
        assert role["suggested"] in _SUGGESTED


def test_d3_signed_delta_mm(tmp_path: Path) -> None:
    """D3: signed delta_mm when both measured+recipe finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "glute_bottom_l": _lm("glute_bottom_l", x_m=-0.12, y_m=0.08, z_m=0.84184),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_hip_glute_compare(report, recipe, tmp_path / "cmp", force=True)
    glute = next(r for r in payload["roles"] if r["id"] == "glute_bottom_l")
    assert glute["delta_mm"] is not None
    assert glute["delta_mm"]["x"] == pytest.approx((-0.12 - (-0.13075)) * 1000.0, abs=1e-3)


def test_d3b_glute_outer_hold_priors_not_soft_adjust(tmp_path: Path) -> None:
    """D9 / B23: glute_outer_* stay hold_priors even when measured X finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "glute_outer_l": _lm("glute_outer_l", x_m=-0.30, y_m=0.05, z_m=0.84),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_hip_glute_compare(report, recipe, tmp_path / "cmp", force=True)
    outer = next(r for r in payload["roles"] if r["id"] == "glute_outer_l")
    assert outer["measured"] is not None
    assert outer["delta_mm"] is not None
    assert outer["suggested"] == "hold_priors"


def test_d4_missing_id_skip_no_nan(tmp_path: Path) -> None:
    """D4: missing id → suggested skip / missing_id — no NaN."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_hip_glute_compare(report, recipe, tmp_path / "cmp", force=True)
    bottom = next(r for r in payload["roles"] if r["id"] == "glute_bottom_l")
    assert bottom["measured"] is None
    assert bottom["suggested"] == "skip"
    assert "missing_id" in bottom["form_read"]
    dumped = json.dumps(payload)
    assert "NaN" not in dumped
    assert "Infinity" not in dumped


def test_d5_glute_short_of_hip_soft_outer(tmp_path: Path) -> None:
    """D5: synthetic hip_soft-glute outer >= 0.040 -> glute_short_of_hip_soft_outer."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe(short_outer=True))
    payload = run_blockout_hip_glute_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "glute_short_of_hip_soft_outer" in tokens


def test_d6_cli_help_json_ok(tmp_path: Path) -> None:
    """D6: CLI help / --json path; exit 0 on structural ok."""
    help_result = runner.invoke(app, ["proportion", "blockout-hip-glute-compare", "--help"])
    assert help_result.exit_code == 0
    assert "blockout-hip-glute-compare" in help_result.output
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-hip-glute-compare",
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
    assert payload["region"] == "hip_glute"


def test_d7_stdout_honesty(tmp_path: Path) -> None:
    """D7: stdout honesty — compare is not mesh or print success."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-hip-glute-compare",
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
    assert HIP_GLUTE_COMPARE_HONESTY in result.output
    assert "not mesh or print success" in result.output.lower()
    assert "Difficulty §N6" not in result.output


def test_d8_pelvis_front_behind_glute(tmp_path: Path) -> None:
    """D8: synthetic pelvis front behind glute front by ≥0.010 → token."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(
        tmp_path / "recipe.json",
        _productish_recipe(pelvis_behind=True),
    )
    payload = run_blockout_hip_glute_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "pelvis_front_behind_glute" in tokens


def test_d9_glute_outer_hold_priors_alias() -> None:
    """D9 lives as test_d3b (hold_priors when measured X finite)."""
    assert "glute_outer_l" in HIP_GLUTE_COMPARE_ROLES


def test_f1_mcp_catalog_50() -> None:
    """F1: TOOL_NAMES 50 and hip-glute-compare tool present."""
    assert "mesh_proportion_blockout_hip_glute_compare" in TOOL_NAMES
    assert len(TOOL_NAMES) == 50


def test_f2_cli_contains_verb() -> None:
    """F2: src/meshops/cli.py contains blockout-hip-glute-compare."""
    cli = Path("src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-hip-glute-compare" in cli


def test_sidecar_unlinked_when_no_finite_ids(tmp_path: Path) -> None:
    """--force with no finite hip/glute ids must not leave a stale sidecar."""
    report_hit = _write_report(
        tmp_path / "hit.json",
        _report({"asis_l": _lm("asis_l", x_m=-0.10, z_m=0.95)}),
    )
    report_miss = _write_report(tmp_path / "miss.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    run_blockout_hip_glute_compare(report_hit, recipe, out, force=True)
    assert (out / "hip_glute_metrics.json").is_file()
    run_blockout_hip_glute_compare(report_miss, recipe, out, force=True)
    assert not (out / "hip_glute_metrics.json").is_file()


def test_sidecar_hip_glute_metrics_when_ids_present(tmp_path: Path) -> None:
    """Compare writes hip_glute_metrics.json when at least one v1 id is finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report({"asis_l": _lm("asis_l", x_m=-0.10, z_m=0.95)}),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    payload = run_blockout_hip_glute_compare(report, recipe, out, force=True)
    sidecar = out / "hip_glute_metrics.json"
    assert sidecar.is_file()
    metrics = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metrics["glute_outer_vs_hip_soft_outer_m"] == pytest.approx(0.07319, abs=1e-4)
    assert payload["hip_glute_metrics"]["glute_outer_vs_hip_soft_outer_m"] == pytest.approx(
        0.07319, abs=1e-4
    )


def test_mcp_wrapper_calls_compare(tmp_path: Path) -> None:
    """MCP adapter reaches the same compare engine."""
    from meshops.mcp.tools import mesh_proportion_blockout_hip_glute_compare

    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = mesh_proportion_blockout_hip_glute_compare(
        tmp_path,
        report=str(report),
        recipe=str(recipe),
        out=str(tmp_path / "cmp"),
        force=True,
    )
    assert payload["ok"] is True
    assert payload["region"] == "hip_glute"
    assert (tmp_path / "cmp" / "hip_glute_compare.json").is_file()


def test_f4_honesty_token() -> None:
    """F4: HIP_GLUTE_COMPARE_HONESTY in honesty.py."""
    from meshops.proportion import honesty as honesty_mod

    assert HIP_GLUTE_COMPARE_HONESTY == "proportion_hip_glute_compare_not_mesh_or_print_success"
    assert hasattr(honesty_mod, "HIP_GLUTE_COMPARE_HONESTY")
