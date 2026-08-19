"""Track 0109 — validate/optimize --template-applied is honestly optional.

CONSTRAINT_HONESTY / OPTIMIZE_HONESTY / TEMPLATE_HONESTY / Difficulty §12 / §13 / N6.
Missing file skips soft template rules; hard C_* still run. Skip ≠ invent a
template. Validate ok is not mesh/print success. Schema 1.4.0 / constraints 1.0.0
/ MCP 46 stay. Recipe/skeleton load_template_applied is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    RECIPE_SCHEMA_VERSION,
    BlockoutRecipePackage,
    RecipePart,
)
from meshops.proportion.body_template import (
    APPLIED_JSON_BASENAME,
    apply_body_template,
    load_template_applied,
)
from meshops.proportion.constraints import (
    CONSTRAINTS_SCHEMA_VERSION,
    run_blockout_optimize,
    run_blockout_validate_constraints,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import CONSTRAINT_HONESTY
from meshops.proportion.models import LandmarkXYZ, ProportionReport, QualityFlags

_SKIP_PREFIX = "template_applied skipped: not found"
_PARENT_NOTE = "template_applied: resolved from parent"
_NOT_FOUND = "template_applied not found"


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
    z_top_m: float | None = None,
    z_bottom_m: float | None = None,
    top_half_width_m: float | None = None,
    bottom_half_width_m: float | None = None,
    p0: list[float] | None = None,
    p1: list[float] | None = None,
) -> RecipePart:
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
        "foot_plate",
        "palm",
        "toe_soft",
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
        "z_top_m": z_top_m,
        "z_bottom_m": z_bottom_m,
        "top_half_width_m": top_half_width_m,
        "bottom_half_width_m": bottom_half_width_m,
        "p0": p0,
        "p1": p1,
    }
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


def _heel_ank_pkg() -> BlockoutRecipePackage:
    return _pkg(
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


def _palm_box_pkg() -> BlockoutRecipePackage:
    return _pkg(
        [
            _part(
                "RECIPE_palm_r",
                role="palm",
                kind="box",
                center=[0.3, -0.1, 0.9],
                rx_m=None,
                ry_m=None,
                rz_m=None,
                half_depth_m=0.03,
                z_top_m=0.95,
                z_bottom_m=0.85,
                top_half_width_m=0.04,
                bottom_half_width_m=0.04,
            ),
        ]
    )


def _free_dof_pkg() -> BlockoutRecipePackage:
    """Thigh + hip + ank so freeze-feet optimize has a free DOF (not T9b)."""
    return _pkg(
        [
            _part(
                "RECIPE_limb_thigh_l",
                kind="capsule",
                center=None,
                rx_m=None,
                ry_m=None,
                rz_m=None,
                radius_m=0.05,
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


def _write_applied(tmp: Path) -> Path:
    report = ProportionReport(
        schema_version="1.1.0",
        height_m=1.72,
        landmarks_xyz={
            "sole": LandmarkXYZ(id="sole", x_m=0.0, y_m=0.0, z_m=0.0),
        },
        quality=QualityFlags(),
    )
    report_path = tmp / "proportion_report.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    apply_body_template(report_path, "female_adult_athletic", tmp, force=True)
    applied = tmp / APPLIED_JSON_BASENAME
    assert applied.is_file()
    return applied


def _unknown_id_json(tmp: Path) -> Path:
    path = tmp / APPLIED_JSON_BASENAME
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "honesty": "proportion_body_template_not_mesh_or_print_success",
                "template_id": "not_a_real_template",
                "sex": "female",
                "archetype": "adult_athletic",
                "source_report": str(tmp / "proportion_report.json"),
                "height_m": 1.72,
                "constants": {
                    "breast_mode": "dual_tilted",
                    "glute_mode_default": "two_spheres",
                    "torso_mode_default": "ovals",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_t0_missing_dir_no_longer_raises(tmp_path: Path) -> None:
    """T0 invert: empty dir used to raise recipe_failed; now skip + hard C_* run."""
    recipe_path = _write_recipe(tmp_path / "blockout_recipe.json", _heel_ank_pkg())
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_dir = tmp_path / "constraints"
    payload = run_blockout_validate_constraints(
        recipe_path,
        out_dir,
        template_applied=empty_dir,
        force=True,
    )
    assert payload["ok"] is True
    assert payload["honesty"] == CONSTRAINT_HONESTY
    joined = " ".join(payload["messages"])
    assert _SKIP_PREFIX in joined
    assert _NOT_FOUND not in joined
    report_path = out_dir / "constraints_report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert any(_SKIP_PREFIX in m for m in data["messages"])
    assert data["honesty"] == CONSTRAINT_HONESTY


def test_t1_missing_named_file_skips(tmp_path: Path) -> None:
    """T1: missing .json file skips that abs path only (B4, no parent hop)."""
    recipe_path = _write_recipe(tmp_path / "blockout_recipe.json", _heel_ank_pkg())
    missing = tmp_path / "ghost" / "template_applied.json"
    out_dir = tmp_path / "constraints"
    payload = run_blockout_validate_constraints(
        recipe_path,
        out_dir,
        template_applied=missing,
        force=True,
    )
    assert payload["ok"] is True
    joined = " ".join(payload["messages"])
    assert _SKIP_PREFIX in joined
    assert str(missing.resolve()) in joined
    assert _PARENT_NOTE not in joined
    parent_try = missing.parent.parent / APPLIED_JSON_BASENAME
    assert str(parent_try.resolve()) not in joined
    rule_ids = {r["id"] for r in payload["rules"]}
    assert "C_palm_ellipsoid" in rule_ids
    assert len(payload["rules"]) >= 14


def test_t2_missing_nofuse_dir_lists_both_abs(tmp_path: Path) -> None:
    """T2: nofuse-class dir + ghost path are directory-class; skip lists both tries."""
    job = tmp_path / "job"
    nofuse = job / "nofuse"
    nofuse.mkdir(parents=True)
    recipe_path = _write_recipe(nofuse / "blockout_recipe.json", _heel_ank_pkg())
    out_dir = tmp_path / "constraints"
    payload = run_blockout_validate_constraints(
        recipe_path,
        out_dir,
        template_applied=nofuse,
        force=True,
    )
    joined = " ".join(payload["messages"])
    assert _SKIP_PREFIX in joined
    assert str((nofuse / APPLIED_JSON_BASENAME).resolve()) in joined
    assert str((job / APPLIED_JSON_BASENAME).resolve()) in joined
    assert _PARENT_NOTE not in joined

    ghost = tmp_path / "ghost"
    payload_g = run_blockout_validate_constraints(
        recipe_path,
        tmp_path / "constraints_ghost",
        template_applied=ghost,
        force=True,
    )
    joined_g = " ".join(payload_g["messages"])
    assert _SKIP_PREFIX in joined_g
    assert str((ghost / APPLIED_JSON_BASENAME).resolve()) in joined_g
    assert str((tmp_path / APPLIED_JSON_BASENAME).resolve()) in joined_g


def test_t3_parent_resolve(tmp_path: Path) -> None:
    """T3: job-root template_applied.json resolved from nofuse dir (B3)."""
    job = tmp_path / "job"
    nofuse = job / "nofuse"
    nofuse.mkdir(parents=True)
    applied = _write_applied(job)
    recipe_path = _write_recipe(nofuse / "blockout_recipe.json", _heel_ank_pkg())
    payload = run_blockout_validate_constraints(
        recipe_path,
        tmp_path / "constraints",
        template_applied=nofuse,
        force=True,
    )
    assert payload["ok"] is True
    joined = " ".join(payload["messages"])
    assert _PARENT_NOTE in joined
    assert str(applied.resolve()) in joined
    assert _SKIP_PREFIX not in joined
    report_path = tmp_path / "constraints" / "constraints_report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert any(_PARENT_NOTE in m for m in data["messages"])


def test_t4_invalid_json_still_fails(tmp_path: Path) -> None:
    """T4: existing bad JSON still raises; do not skip."""
    recipe_path = _write_recipe(tmp_path / "blockout_recipe.json", _heel_ank_pkg())
    bad = tmp_path / APPLIED_JSON_BASENAME
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProportionError) as ei:
        run_blockout_validate_constraints(
            recipe_path,
            tmp_path / "constraints",
            template_applied=bad,
            force=True,
        )
    assert ei.value.code == "recipe_failed"
    assert "cannot load template_applied" in str(ei.value)
    assert _SKIP_PREFIX not in str(ei.value)


def test_t5_unknown_template_id_still_fails(tmp_path: Path) -> None:
    """T5: unknown template_id still raises via load_template_applied."""
    recipe_path = _write_recipe(tmp_path / "blockout_recipe.json", _heel_ank_pkg())
    bad = _unknown_id_json(tmp_path)
    with pytest.raises(ProportionError) as ei:
        run_blockout_validate_constraints(
            recipe_path,
            tmp_path / "constraints",
            template_applied=bad,
            force=True,
        )
    assert ei.value.code == "recipe_failed"
    assert "unknown template_id" in str(ei.value)
    assert _SKIP_PREFIX not in str(ei.value)


def test_t6_hard_c_star_still_fail_when_template_skipped(tmp_path: Path) -> None:
    """T6: skip does not swallow hard C_palm_ellipsoid fail."""
    recipe_path = _write_recipe(tmp_path / "blockout_recipe.json", _palm_box_pkg())
    empty_dir = tmp_path / "nofuse"
    empty_dir.mkdir()
    payload = run_blockout_validate_constraints(
        recipe_path,
        tmp_path / "constraints",
        template_applied=empty_dir,
        force=True,
    )
    assert payload["ok"] is True
    assert payload["constraints_ok"] is False
    by_id = {r["id"]: r for r in payload["rules"]}
    assert by_id["C_palm_ellipsoid"]["status"] == "fail"
    assert "ellipsoid" in by_id["C_palm_ellipsoid"]["message"]
    joined = " ".join(payload["messages"])
    assert _SKIP_PREFIX in joined


def test_t7_omit_flag_unchanged(tmp_path: Path) -> None:
    """T7: omit --template-applied stays None path; no skip note required."""
    recipe_path = _write_recipe(tmp_path / "blockout_recipe.json", _heel_ank_pkg())
    payload = run_blockout_validate_constraints(
        recipe_path,
        tmp_path / "constraints",
        force=True,
    )
    assert payload["ok"] is True
    joined = " ".join(payload["messages"])
    assert _SKIP_PREFIX not in joined
    report_path = tmp_path / "constraints" / "constraints_report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == CONSTRAINTS_SCHEMA_VERSION
    assert RECIPE_SCHEMA_VERSION == "1.4.0"


def test_t8_load_template_applied_still_fails(tmp_path: Path) -> None:
    """T8: B6 — shared loader still recipe_failed on missing path."""
    missing = tmp_path / "no_such_dir"
    with pytest.raises(ProportionError) as ei:
        load_template_applied(missing)
    assert ei.value.code == "recipe_failed"
    assert _NOT_FOUND in str(ei.value)


def test_t9_optimize_shares_helper(tmp_path: Path) -> None:
    """T9: run_blockout_optimize missing dir skips; not recipe_failed."""
    recipe_path = _write_recipe(tmp_path / "blockout_recipe.json", _free_dof_pkg())
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    payload = run_blockout_optimize(
        recipe_path,
        tmp_path / "opt",
        template_applied=empty_dir,
        force=True,
    )
    assert payload["ok"] is True
    joined = " ".join(payload["messages"])
    assert _SKIP_PREFIX in joined
    assert _NOT_FOUND not in joined
    result_path = tmp_path / "opt" / "optimize_result.json"
    assert result_path.is_file()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert any(_SKIP_PREFIX in m for m in data["messages"])


def test_t10_mcp_catalog_hold() -> None:
    """T10: MCP catalog stay 46."""
    assert len(TOOL_NAMES) == 46


def test_t11_cli_help_skip_and_parent() -> None:
    """T11: validate + optimize --help mention skip and parent (B9)."""
    runner = CliRunner()
    r_val = runner.invoke(app, ["proportion", "blockout-validate-constraints", "--help"])
    assert r_val.exit_code == 0, r_val.stdout
    help_val = r_val.stdout.lower()
    assert "optional" in help_val
    assert "skip" in help_val
    assert "parent" in help_val
    r_opt = runner.invoke(app, ["proportion", "blockout-optimize", "--help"])
    assert r_opt.exit_code == 0, r_opt.stdout
    help_opt = r_opt.stdout.lower()
    assert "optional" in help_opt
    assert "skip" in help_opt
    assert "parent" in help_opt
