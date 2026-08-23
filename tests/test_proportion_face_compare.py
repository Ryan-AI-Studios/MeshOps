"""Track 0124 — blockout-face-compare JSON / honesty / scene-dump (offline).

Authoring QA only — FACE_COMPARE_HONESTY. Not mesh or print success.
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
from meshops.proportion.face_compare import (
    FACE_COMPARE_ROLES,
    FACE_COMPARE_SCHEMA_VERSION,
    SUGGESTED_ACTIONS,
    build_face_metrics,
    extract_recipe_face_part,
    run_blockout_face_compare,
)
from meshops.proportion.honesty import FACE_COMPARE_HONESTY, PROPORTION_HONESTY
from meshops.proportion.models import (
    PROPORTION_SCHEMA_VERSION,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

runner = CliRunner()

_SUGGESTED = frozenset({"skip", "hold_priors", "soft_adjust", "session_0113", "remake_0111"})


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


def _recipe_doc(*, parts: list[dict[str, Any]], head_rx: float = 0.08828) -> dict[str, Any]:
    return {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "honesty": "proportion_blockout_recipe_not_mesh_or_print_success",
        "recipe_id": RECIPE_ID,
        "axis_notes": "Z up, soles=0, +X camera-right, +Y toward camera_left",
        "height_m": 1.72,
        "head_unit_m": 0.21018,
        "parts": [
            *parts,
            _ellipsoid("RECIPE_head", "head", [0.0, -0.02023, 1.6143], head_rx, 0.09080, 0.11035),
        ],
    }


def _write_recipe(path: Path, doc: dict[str, Any]) -> Path:
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _productish_recipe(*, head_rx: float = 0.08828) -> dict[str, Any]:
    eye_cx = 0.04624
    return _recipe_doc(
        head_rx=head_rx,
        parts=[
            _ellipsoid(
                "RECIPE_eye_soft_l",
                "eye_soft",
                [-eye_cx, -0.10150, 1.6058],
                0.02312,
                0.01433,
                0.01341,
            ),
            _ellipsoid(
                "RECIPE_eye_soft_r",
                "eye_soft",
                [eye_cx, -0.10150, 1.6058],
                0.02312,
                0.01433,
                0.01341,
            ),
            _capsule(
                "RECIPE_brow_soft_l",
                "brow_soft",
                [-0.07167, -0.10524, 1.64131],
                [-0.02081, -0.10524, 1.64131],
                0.005885,
            ),
            _capsule(
                "RECIPE_brow_soft_r",
                "brow_soft",
                [0.02081, -0.10524, 1.64131],
                [0.07167, -0.10524, 1.64131],
                0.005885,
            ),
            _ellipsoid(
                "RECIPE_nose_soft",
                "nose_soft",
                [0.0, -0.09327, 1.5686],
                0.00946,
                0.01156,
                0.00841,
            ),
            _ellipsoid(
                "RECIPE_lip_soft",
                "lip_soft",
                [0.0, -0.09667, 1.5598],
                0.02102,
                0.00589,
                0.00420,
            ),
            _ellipsoid(
                "RECIPE_ear_soft_l",
                "ear_soft",
                [-head_rx, -0.02023, 1.605],
                0.0168,
                0.0084,
                0.036,
            ),
            _ellipsoid(
                "RECIPE_ear_soft_r",
                "ear_soft",
                [head_rx, -0.02023, 1.605],
                0.0168,
                0.0084,
                0.036,
            ),
            _ellipsoid(
                "RECIPE_cheek_soft_l",
                "cheek_soft",
                [-0.04855, -0.096, 1.59],
                0.0247,
                0.0127,
                0.0095,
            ),
            _ellipsoid(
                "RECIPE_cheek_soft_r",
                "cheek_soft",
                [0.04855, -0.096, 1.59],
                0.0247,
                0.0127,
                0.0095,
            ),
            _ellipsoid(
                "RECIPE_jaw",
                "jaw",
                [0.0, -0.056, 1.537],
                0.06532,
                0.03814,
                0.02732,
            ),
            _ellipsoid(
                "RECIPE_hair_mass",
                "hair_mass",
                [0.0, -0.02, 1.68],
                0.09,
                0.09,
                0.04,
            ),
        ],
    )


def test_b1_ipd_both_eyes_or_none() -> None:
    """B1: both eyes → ipd_m; missing one eye → no invent IPD."""
    both = _report(
        {
            "eye_l": _lm("eye_l", x_m=-0.04, z_m=1.60),
            "eye_r": _lm("eye_r", x_m=0.04, z_m=1.60),
        }
    )
    metrics = build_face_metrics(both, recipe=None)
    assert metrics is not None
    assert metrics.ipd_m == pytest.approx(0.08, abs=1e-9)
    one = _report({"eye_l": _lm("eye_l", x_m=-0.04, z_m=1.60)})
    metrics_one = build_face_metrics(one, recipe=None)
    assert metrics_one is not None
    assert metrics_one.ipd_m is None


def test_b2_z_fracs_when_chin_and_feature() -> None:
    """B2: Z fracs when chin + cranial + feature present."""
    chin_z = 1.50
    top_z = 1.71
    h = top_z - chin_z
    eye_z = chin_z + 0.50 * h
    report = _report(
        {
            "chin": _lm("chin", z_m=chin_z),
            "cranial_vertex": _lm("cranial_vertex", z_m=top_z),
            "eye_l": _lm("eye_l", x_m=-0.04, z_m=eye_z),
            "eye_r": _lm("eye_r", x_m=0.04, z_m=eye_z),
        }
    )
    metrics = build_face_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.eye_z_frac_h == pytest.approx(0.50, abs=1e-6)


def test_b3_front_only_y_null() -> None:
    """B3: front-only measured → y_m null on face features (no invent)."""
    report = _report(
        {
            "eye_l": _lm("eye_l", x_m=-0.04, z_m=1.60),
            "eye_r": _lm("eye_r", x_m=0.04, z_m=1.60),
        }
    )
    metrics = build_face_metrics(report, recipe=None)
    assert metrics is not None
    assert metrics.y_m.get("eye_l") is None
    assert metrics.y_m.get("eye_r") is None


def test_b4_proportion_report_stay_1_2_0() -> None:
    """B4: proportion report schema stay 1.2.0 (sidecar, not a 1.3.0 bump)."""
    assert PROPORTION_SCHEMA_VERSION == "1.2.0"
    report = _report()
    assert report.schema_version == "1.2.0"
    dumped = report.model_dump(mode="json")
    assert "face_metrics" not in dumped


def test_c1_recipe_eye_extract() -> None:
    """C1: recipe JSON extracts RECIPE_eye_soft_l center/rx/ry/rz."""
    snap = extract_recipe_face_part(_productish_recipe()["parts"], "eye_l")
    assert snap is not None
    assert snap["name"] == "RECIPE_eye_soft_l"
    assert snap["center"] is not None
    assert snap["center"][0] == pytest.approx(-0.04624, abs=1e-6)
    assert snap["rx_m"] == pytest.approx(0.02312, abs=1e-6)
    assert snap["ry_m"] == pytest.approx(0.01433, abs=1e-6)
    assert snap["rz_m"] == pytest.approx(0.01341, abs=1e-6)


def test_c2_missing_scene_dump_live_null(tmp_path: Path) -> None:
    """C2: missing --scene-dump → live=null (not fail)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    payload = run_blockout_face_compare(report, recipe, out, force=True)
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
        run_blockout_face_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "face_compare.json").is_file()


def test_c3b_named_part_without_geometry_fail_closed(tmp_path: Path) -> None:
    """C3: named face RECIPE dump part without center/p0/p1 fails closed."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    dump = tmp_path / "dump.json"
    dump.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source": "scene",
                "parts": [{"name": "RECIPE_eye_soft_l"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProportionError):
        run_blockout_face_compare(
            report,
            recipe,
            tmp_path / "cmp",
            scene_dump=dump,
            force=True,
        )
    assert not (tmp_path / "cmp" / "face_compare.json").is_file()


def test_c4_brow_capsule_midpoint() -> None:
    """C4 / B26: capsule brow center = mid(p0,p1); radius_m finite; rx_m stays null."""
    p0 = [-0.07167, -0.10524, 1.64131]
    p1 = [-0.02081, -0.10524, 1.64131]
    snap = extract_recipe_face_part(
        [_capsule("RECIPE_brow_soft_l", "brow_soft", p0, p1, 0.005885)],
        "brow_l",
    )
    assert snap is not None
    assert snap["kind"] == "capsule"
    assert snap["center"] is not None
    assert snap["center"][0] == pytest.approx((p0[0] + p1[0]) / 2.0, abs=1e-9)
    assert snap["center"][1] == pytest.approx(p0[1], abs=1e-9)
    assert snap["center"][2] == pytest.approx(p0[2], abs=1e-9)
    assert snap["radius_m"] == pytest.approx(0.005885, abs=1e-9)
    assert snap["rx_m"] is None


def test_d1_honesty_schema_region(tmp_path: Path) -> None:
    """D1: payload honesty / schema 1.0.0 / region=face."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_face_compare(report, recipe, tmp_path / "cmp", force=True)
    assert payload["honesty"] == FACE_COMPARE_HONESTY
    assert payload["schema_version"] == FACE_COMPARE_SCHEMA_VERSION
    assert payload["schema_version"] == "1.0.0"
    assert payload["region"] == "face"
    raw = json.loads((tmp_path / "cmp" / "face_compare.json").read_text(encoding="utf-8"))
    assert raw["honesty"] == FACE_COMPARE_HONESTY
    assert raw["region"] == "face"


def test_d2_suggested_closed_set(tmp_path: Path) -> None:
    """D2: each role has suggested in the closed set."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_face_compare(report, recipe, tmp_path / "cmp", force=True)
    assert SUGGESTED_ACTIONS == _SUGGESTED
    ids = {r["id"] for r in payload["roles"]}
    for lid in FACE_COMPARE_ROLES:
        assert lid in ids
    for role in payload["roles"]:
        assert role["suggested"] in _SUGGESTED


def test_d3_signed_delta_mm(tmp_path: Path) -> None:
    """D3: signed delta_mm when both measured+recipe finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "eye_l": _lm("eye_l", x_m=-0.036, y_m=-0.10, z_m=1.6058),
                "eye_r": _lm("eye_r", x_m=0.036, y_m=-0.10, z_m=1.6058),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_face_compare(report, recipe, tmp_path / "cmp", force=True)
    eye_l = next(r for r in payload["roles"] if r["id"] == "eye_l")
    assert eye_l["delta_mm"] is not None
    assert eye_l["delta_mm"]["x"] == pytest.approx((-0.036 - (-0.04624)) * 1000.0, abs=1e-3)


def test_d4_missing_id_skip_no_nan(tmp_path: Path) -> None:
    """D4: missing id → suggested skip / missing_id — no NaN."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = run_blockout_face_compare(report, recipe, tmp_path / "cmp", force=True)
    eye_l = next(r for r in payload["roles"] if r["id"] == "eye_l")
    assert eye_l["measured"] is None
    assert eye_l["suggested"] == "skip"
    assert "missing_id" in eye_l["form_read"]
    dumped = json.dumps(payload)
    assert "NaN" not in dumped
    assert "Infinity" not in dumped


def test_d5_eyes_wide_set_flag(tmp_path: Path) -> None:
    """D5: recipe half-IPD/rx ≥ 0.45 → eyes_wide_set (inventory 0.524)."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe(head_rx=0.08828))
    payload = run_blockout_face_compare(report, recipe, tmp_path / "cmp", force=True)
    tokens: list[str] = []
    for role in payload["roles"]:
        tokens.extend(role.get("form_read") or [])
    assert "eyes_wide_set" in tokens


def test_d6_cli_help_json_ok(tmp_path: Path) -> None:
    """D6: CLI help / --json path; exit 0 on structural ok."""
    help_result = runner.invoke(app, ["proportion", "blockout-face-compare", "--help"])
    assert help_result.exit_code == 0
    assert "blockout-face-compare" in help_result.output
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-face-compare",
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
    assert payload["region"] == "face"


def test_d7_stdout_honesty(tmp_path: Path) -> None:
    """D7: stdout honesty — compare is not mesh or print success."""
    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    result = runner.invoke(
        app,
        [
            "proportion",
            "blockout-face-compare",
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
    assert FACE_COMPARE_HONESTY in result.output
    assert "not mesh or print success" in result.output.lower()
    assert "Difficulty §N6" not in result.output


def test_f1_mcp_catalog_48() -> None:
    """F1: TOOL_NAMES 48 and face-compare tool present."""
    assert "mesh_proportion_blockout_face_compare" in TOOL_NAMES
    assert len(TOOL_NAMES) == 52


def test_f2_cli_contains_verb() -> None:
    """F2: src/meshops/cli.py contains blockout-face-compare."""
    cli = Path("src/meshops/cli.py").read_text(encoding="utf-8")
    assert "blockout-face-compare" in cli


def test_sidecar_face_metrics_when_ids_present(tmp_path: Path) -> None:
    """Compare writes face_metrics.json when at least one v1 face id is finite."""
    report = _write_report(
        tmp_path / "report.json",
        _report(
            {
                "eye_l": _lm("eye_l", x_m=-0.04, z_m=1.60),
                "eye_r": _lm("eye_r", x_m=0.04, z_m=1.60),
            }
        ),
    )
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    out = tmp_path / "cmp"
    payload = run_blockout_face_compare(report, recipe, out, force=True)
    sidecar = out / "face_metrics.json"
    assert sidecar.is_file()
    metrics = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metrics["ipd_m"] == pytest.approx(0.08, abs=1e-9)
    assert payload["face_metrics"]["ipd_m"] == pytest.approx(0.08, abs=1e-9)


def test_mcp_wrapper_calls_compare(tmp_path: Path) -> None:
    """MCP adapter reaches the same compare engine."""
    from meshops.mcp.tools import mesh_proportion_blockout_face_compare

    report = _write_report(tmp_path / "report.json", _report())
    recipe = _write_recipe(tmp_path / "recipe.json", _productish_recipe())
    payload = mesh_proportion_blockout_face_compare(
        tmp_path,
        report=str(report),
        recipe=str(recipe),
        out=str(tmp_path / "cmp"),
        force=True,
    )
    assert payload["ok"] is True
    assert payload["region"] == "face"
    assert (tmp_path / "cmp" / "face_compare.json").is_file()


def test_f4_honesty_token() -> None:
    """F4: FACE_COMPARE_HONESTY in honesty.py."""
    from meshops.proportion import honesty as honesty_mod

    assert FACE_COMPARE_HONESTY == "proportion_face_compare_not_mesh_or_print_success"
    assert hasattr(honesty_mod, "FACE_COMPARE_HONESTY")
