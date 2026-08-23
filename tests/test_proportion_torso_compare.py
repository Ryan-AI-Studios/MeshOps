"""Track 0125 — blockout-torso-compare JSON / honesty / scene-dump (offline).

Authoring QA only — TORSO_COMPARE_HONESTY. Not mesh or print success.
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
from meshops.proportion.honesty import PROPORTION_HONESTY, TORSO_COMPARE_HONESTY
from meshops.proportion.models import (
    PROPORTION_SCHEMA_VERSION,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.torso_compare import (
    SUGGESTED_ACTIONS,
    TORSO_COMPARE_ROLES,
    TORSO_COMPARE_SCHEMA_VERSION,
    build_torso_metrics,
    extract_recipe_torso_part,
    run_blockout_torso_compare,
)

runner = CliRunner()

_SUGGESTED = frozenset({"skip", "hold_priors", "soft_adjust", "session_torso", "remake_0111"})


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


def _productish_recipe(*, overlap_ok: bool = True) -> dict[str, Any]:
    chest_rz = 0.13389 if overlap_ok else 0.040
    waist_rz = 0.09913 if overlap_ok else 0.030
    hip_rz = 0.11476 if overlap_ok else 0.040
    return _recipe_doc(
        parts=[
            _ellipsoid(
                "RECIPE_torso_oval_chest",
                "torso",
                [0.0, 0.04785, 1.28457],
                0.2181,
                0.09382,
                chest_rz,
            ),
            _ellipsoid(
                "RECIPE_torso_oval_waist",
                "torso",
                [0.0, 0.03174, 1.14112],
                0.1745,
                0.07558,
                waist_rz,
            ),
            _ellipsoid(
                "RECIPE_torso_oval_hip",
                "torso",
                [0.0, 0.02935, 1.00723],
                0.1979,
                0.08895,
                hip_rz,
            ),
            _ellipsoid(
                "RECIPE_scap_soft_l",
                "scap_soft",
                [-0.11586, 0.12477, 1.2856],
                0.0688,
                0.02890,
                0.07912,
            ),
            _ellipsoid(
                "RECIPE_scap_soft_r",
                "scap_soft",
                [0.11586, 0.12477, 1.2856],
                0.0688,
                0.02890,
                0.07912,
            ),
            _ellipsoid(
                "RECIPE_mid_back_soft_l",
                "mid_back_soft",
                [-0.12359, 0.11448, 1.10612],
                0.06536,
                0.02484,
                0.0860,
            ),
            _ellipsoid(
                "RECIPE_mid_back_soft_r",
                "mid_back_soft",
                [0.12359, 0.11448, 1.10612],
                0.06536,
                0.02484,
                0.0860,
            ),
            _ellipsoid(
                "RECIPE_breast_soft_l",
                "breast_soft",
                [-0.08674, -0.09832, 1.22814],
                0.07224,
                0.05635,
                0.07585,
            ),
            _ellipsoid(
                "RECIPE_breast_soft_r",
                "breast_soft",
                [0.08674, -0.09832, 1.22814],
                0.07224,
                0.05635,
                0.07585,
            ),
            _ellipsoid(
                "RECIPE_pelvis_oval",
                "pelvis",
                [0.0, 0.02, 0.8332],
                0.22,
                0.0834,
                0.0722,
            ),
            _capsule(
                "RECIPE_clavicle_l",
                "clavicle",
                [-0.02, -0.04597, 1.33],
                [-0.16, -0.04597, 1.36],
                0.02064,
            ),
            _capsule(
                "RECIPE_clavicle_r",
                "clavicle",
                [0.02, -0.04597, 1.33],
                [0.16, -0.04597, 1.36],
                0.02064,
            ),
        ],
    )


def test_b1_oval_overlaps_when_recipe() -> None:
    """B1: recipe present → oval overlaps finite (inventory-class ~0.0896/0.0800)."""
    metrics = build_torso_metrics(_report(), recipe=_productish_recipe())
    assert metrics is not None
    assert metrics.oval_overlap_chest_waist_m == pytest.approx(0.08957, abs=1e-4)
    assert metrics.oval_overlap_waist_hip_m == pytest.approx(0.08000, abs=1e-4)


def test_b2_front_only_y_null() -> None:
    """B2: front-only measured → y_m null on new torso ids (no invent)."""
    report = _report(
        {
            "sternum_mid": _lm("sternum_mid", x_m=0.0, z_m=1.28),
            "scap_inferior_l": _lm("scap_inferior_l", x_m=-0.11, z_m=1.28),
        }
    )
    metrics = build_torso_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("sternum_mid") is None
    assert metrics.y_m.get("scap_inferior_l") is None


def test_b3_missing_one_scap_no_invent() -> None:
    """B3: missing one scap → no invent contralateral."""
    report = _report({"scap_inferior_l": _lm("scap_inferior_l", x_m=-0.11, y_m=0.15, z_m=1.28)})
    metrics = build_torso_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("scap_inferior_l") == pytest.approx(0.15, abs=1e-9)
    assert metrics.y_m.get("scap_inferior_r") is None


def test_b4_proportion_report_stay_1_2_0() -> None:
    """B4: proportion report schema stay 1.2.0 (sidecar, not a 1.3.0 bump)."""
    assert PROPORTION_SCHEMA_VERSION == "1.2.0"
    report = _report()
    assert report.schema_version == "1.2.0"
    dumped = report.model_dump(mode="json")
    assert "torso_metrics" not in dumped


def test_c1_recipe_chest_extract() -> None:
    """C1: recipe JSON extracts RECIPE_torso_oval_chest center/rx/ry/rz."""
    snap = extract_recipe_torso_part(_productish_recipe()["parts"], "sternum_mid")
    assert snap is not None
    assert snap["name"] == "RECIPE_torso_oval_chest"
    assert snap["center"] is not None
    assert snap["center"][2] == pytest.approx(1.28457, abs=1e-6)
    assert snap["rx_m"] == pytest.approx(0.2181, abs=1e-6)
    assert snap["ry_m"] == pytest.approx(0.09382, abs=1e-6)
    assert snap["rz_m"] == pytest.approx(0.13389, abs=1e-6)


def test_c2_missing_scene_dump_live_null(tmp_path: Path) -> None:
    """C2: missing --scene-dump → live=null (not fail)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_torso_compare(report, recipe, tmp_path / "cmp", force=True)
    assert payload["ok"] is True
    for role in payload["roles"]:
        assert role["live"] is None


def test_c3_malformed_dump_fail_closed(tmp_path: Path) -> None:
    """C3: malformed dump → ProportionError (fail-closed)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = tmp_path / "dump.json"
    dump.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ProportionError):
        run_blockout_torso_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "torso_compare.json").is_file()


def test_c3b_named_part_without_geometry_fail_closed(tmp_path: Path) -> None:
    """C3: named torso RECIPE dump part without center/p0/p1 fails closed."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = tmp_path / "dump.json"
    dump.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source": "scene",
                "parts": [{"name": "RECIPE_torso_oval_chest"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProportionError):
        run_blockout_torso_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "torso_compare.json").is_file()


def test_c4_clavicle_capsule_pride() -> None:
    """C4 / B26: capsule clavicle pride uses p1 Y; rx_m stays null."""
    p0 = [-0.02, -0.04597, 1.33]
    p1 = [-0.16, -0.04597, 1.36]
    snap = extract_recipe_torso_part(
        [_capsule("RECIPE_clavicle_l", "clavicle", p0, p1, 0.02064)],
        "clavicle_l",
    )
    assert snap is not None
    assert snap["kind"] == "capsule"
    assert snap["center"] is not None
    assert snap["center"][1] == pytest.approx(p1[1], abs=1e-9)
    assert snap["radius_m"] == pytest.approx(0.02064, abs=1e-9)
    assert snap["rx_m"] is None


def test_d1_honesty_schema_region(tmp_path: Path) -> None:
    """D1: payload honesty / schema 1.0.0 / region=torso."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_torso_compare(report, recipe, tmp_path / "cmp", force=True)
    assert payload["honesty"] == TORSO_COMPARE_HONESTY
    assert payload["schema_version"] == TORSO_COMPARE_SCHEMA_VERSION
    assert payload["schema_version"] == "1.0.0"
    assert payload["region"] == "torso"
    raw = json.loads((tmp_path / "cmp" / "torso_compare.json").read_text(encoding="utf-8"))
    assert raw["honesty"] == TORSO_COMPARE_HONESTY
    assert raw["region"] == "torso"


def test_d2_suggested_closed_set(tmp_path: Path) -> None:
    """D2: each role has suggested in the closed set."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_torso_compare(report, recipe, tmp_path / "cmp", force=True)
    assert SUGGESTED_ACTIONS == _SUGGESTED
    ids = {r["id"] for r in payload["roles"]}
    for lid in TORSO_COMPARE_ROLES:
        assert lid in ids
    for role in payload["roles"]:
        assert role["suggested"] in _SUGGESTED


def test_d3_signed_delta_mm(tmp_path: Path) -> None:
    """D3: signed delta_mm when both measured+recipe finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "scap_inferior_l": _lm("scap_inferior_l", x_m=-0.10, y_m=0.12, z_m=1.2856),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_torso_compare(report, recipe, tmp_path / "cmp", force=True)
    scap = next(r for r in payload["roles"] if r["id"] == "scap_inferior_l")
    assert scap["delta_mm"] is not None
    assert scap["delta_mm"]["x"] == pytest.approx((-0.10 - (-0.11586)) * 1000.0, abs=1e-3)


def test_d3b_chest_front_hold_priors_not_soft_adjust(tmp_path: Path) -> None:
    """Existing DEPTH_PAIRS roles never suggest retuning frozen oval consts (B4)."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.20, z_m=1.28),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_torso_compare(report, recipe, tmp_path / "cmp", force=True)
    chest = next(r for r in payload["roles"] if r["id"] == "chest_front")
    assert chest["measured"] is not None
    assert chest["delta_mm"] is not None
    assert chest["suggested"] == "hold_priors"
    assert chest["knob"] == "TORSO_OVAL_RY_CHEST_FRAC"


def test_d4_missing_id_skip_no_nan(tmp_path: Path) -> None:
    """D4: missing id → suggested skip / missing_id — no NaN."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_torso_compare(report, recipe, tmp_path / "cmp", force=True)
    scap = next(r for r in payload["roles"] if r["id"] == "scap_inferior_l")
    assert scap["measured"] is None
    assert scap["suggested"] == "skip"
    assert "missing_id" in scap["form_read"]
    dumped = json.dumps(payload)
    assert "NaN" not in dumped
    assert "Infinity" not in dumped


def test_d5_three_tire_when_overlap_below_floor(tmp_path: Path) -> None:
    """D5: synthetic recipe overlap < 0.080 → three_tire in form_read."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(
        tmp_path / "recipe.json",
        _productish_recipe(overlap_ok=False),
    )
    payload = run_blockout_torso_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "three_tire" in tokens


def test_d6_cli_help_json_ok(tmp_path: Path) -> None:
    """D6: CLI help / --json path; exit 0 on structural ok."""
    help_result = runner.invoke(app, ["proportion", "blockout-torso-compare", "--help"])
    assert help_result.exit_code == 0
    assert "blockout-torso-compare" in help_result.output
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-torso-compare",
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
    assert payload["region"] == "torso"


def test_d7_stdout_honesty(tmp_path: Path) -> None:
    """D7: stdout honesty — compare is not mesh or print success."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-torso-compare",
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
    assert TORSO_COMPARE_HONESTY in result.output
    assert "not mesh or print success" in result.output.lower()
    assert "Difficulty §N6" not in result.output


def test_d8_breast_disconnected(tmp_path: Path) -> None:
    """D8: synthetic breast rear proud of chest front → breast_disconnected."""
    doc = _productish_recipe()
    for part in doc["parts"]:
        if str(part.get("name") or "").startswith("RECIPE_breast_soft_"):
            part["center"] = [part["center"][0], -0.20, part["center"][2]]
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", doc)
    payload = run_blockout_torso_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "breast_disconnected" in tokens


def test_f1_mcp_catalog_49() -> None:
    """F1: TOOL_NAMES 49 and torso-compare tool present."""
    assert "mesh_proportion_blockout_torso_compare" in TOOL_NAMES
    assert len(TOOL_NAMES) == 52


def test_f2_cli_contains_verb() -> None:
    """F2: src/meshops/cli.py contains blockout-torso-compare."""
    cli = Path("src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-torso-compare" in cli


def test_b_measured_chest_y_in_named_metrics() -> None:
    """Named metrics use measured chest_front y_m when left-finite."""
    report = _report({"chest_front": _lm("chest_front", x_m=0.0, y_m=-0.08, z_m=1.28)})
    metrics = build_torso_metrics(report, recipe=_productish_recipe())
    assert metrics is not None
    assert metrics.chest_front_y_m == pytest.approx(-0.08, abs=1e-9)
    assert metrics.y_m.get("chest_front") == pytest.approx(-0.08, abs=1e-9)


def test_sidecar_unlinked_when_no_finite_ids(tmp_path: Path) -> None:
    """--force with no finite torso ids must not leave a stale sidecar."""
    report_hit = _write_report(
        tmp_path / "hit.json",
        _report({"sternum_mid": _lm("sternum_mid", x_m=0.0, z_m=1.28)}),
    )
    report_miss = _write_report(tmp_path / "miss.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    run_blockout_torso_compare(report_hit, recipe, out, force=True)
    assert (out / "torso_metrics.json").is_file()
    run_blockout_torso_compare(report_miss, recipe, out, force=True)
    assert not (out / "torso_metrics.json").is_file()


def test_sidecar_torso_metrics_when_ids_present(tmp_path: Path) -> None:
    """Compare writes torso_metrics.json when at least one v1 torso id is finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report({"sternum_mid": _lm("sternum_mid", x_m=0.0, z_m=1.28)}),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    payload = run_blockout_torso_compare(report, recipe, out, force=True)
    sidecar = out / "torso_metrics.json"
    assert sidecar.is_file()
    metrics = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metrics["oval_overlap_waist_hip_m"] == pytest.approx(0.08000, abs=1e-4)
    assert payload["torso_metrics"]["oval_overlap_waist_hip_m"] == pytest.approx(0.08000, abs=1e-4)


def test_mcp_wrapper_calls_compare(tmp_path: Path) -> None:
    """MCP adapter reaches the same compare engine."""
    from meshops.mcp.tools import mesh_proportion_blockout_torso_compare

    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = mesh_proportion_blockout_torso_compare(
        tmp_path,
        report=str(report),
        recipe=str(recipe),
        out=str(tmp_path / "cmp"),
        force=True,
    )
    assert payload["ok"] is True
    assert payload["region"] == "torso"
    assert (tmp_path / "cmp" / "torso_compare.json").is_file()


def test_f4_honesty_token() -> None:
    """F4: TORSO_COMPARE_HONESTY in honesty.py."""
    from meshops.proportion import honesty as honesty_mod

    assert TORSO_COMPARE_HONESTY == "proportion_torso_compare_not_mesh_or_print_success"
    assert hasattr(honesty_mod, "TORSO_COMPARE_HONESTY")
