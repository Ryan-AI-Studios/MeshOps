"""Track 0015 — proportion guides / blockout helpers (offline; no Blender)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.errors import ProportionError
from meshops.proportion.guides import (
    AXIS_NOTES,
    GUIDE_SCHEMA_VERSION,
    GuidePackage,
    build_guide_package,
    display_size_m,
    emit_bpy_script,
    run_guides,
    write_guides,
)
from meshops.proportion.honesty import GUIDE_HONESTY
from meshops.proportion.models import (
    CrossSection,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)

runner = CliRunner()


def _lm(
    id_: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=id_, x=x, y=y, z=z, x_m=x_m, y_m=y_m, z_m=z_m)


def _synthetic_report(
    *,
    landmarks_xyz: dict[str, LandmarkXYZ] | None = None,
    height_m: float | None = None,
    head_unit_frac: float | None = None,
    diameters: list[DiameterMeasure] | None = None,
    cross_sections: list[CrossSection] | None = None,
    schema_version: str = "1.1.0",
    quality: QualityFlags | None = None,
) -> ProportionReport:
    """Build ProportionReport directly (not via analyze_proportion) — R12 B8."""
    return ProportionReport(
        schema_version=schema_version,  # type: ignore[arg-type]
        height_m=height_m,
        head_unit_frac=head_unit_frac,
        landmarks_xyz=landmarks_xyz or {},
        diameters=diameters or [],
        cross_sections=cross_sections or [],
        quality=quality or QualityFlags(),
    )


def _thigh_diameter(
    *,
    band_id: str = "thigh_l",
    view: str = "front",
    width_m: float | None = 0.12,
    half_width_m: float | None = 0.06,
) -> DiameterMeasure:
    return DiameterMeasure(
        band_id=band_id,
        view=view,
        width_px=40.0,
        width_eucl_px=40.0,
        theta_deg=90.0,
        width_frac=0.1,
        width_m=width_m,
        half_width_m=half_width_m,
        mid_x_px=100.0,
        mid_y_px=200.0,
    )


# ---------------------------------------------------------------------------
# R12 tests
# ---------------------------------------------------------------------------


def test_guides__minimal_xyz__json_empties() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
            "chin": _lm("chin", x_m=0.01, y_m=-0.02, z_m=1.55),
        }
    )
    pkg = build_guide_package(report)
    names = {e.name for e in pkg.empties}
    assert "LM_sole" in names
    assert "LM_chin" in names
    by_name = {e.name: e for e in pkg.empties}
    assert by_name["LM_sole"].source_id == "sole"
    assert by_name["LM_chin"].z_m == pytest.approx(1.55)
    assert by_name["LM_chin"].x_m == pytest.approx(0.01)
    assert pkg.honesty == GUIDE_HONESTY
    assert pkg.axis_notes == AXIS_NOTES
    assert pkg.schema_version == GUIDE_SCHEMA_VERSION
    assert pkg.schema_version == "1.1.0"
    assert pkg.counts.get("seeds_front_plane") == 0


def test_guides__height_and_hu__ladder() -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
        height_m=1.70,
        head_unit_frac=1.0 / 7.5,  # ~0.1333 → n = floor(7.5)=7
    )
    pkg = build_guide_package(report)
    height_empties = [e for e in pkg.empties if e.kind == "height"]
    assert len(height_empties) == 1
    assert height_empties[0].name == "LM_HEIGHT"
    assert height_empties[0].z_m == pytest.approx(1.70)
    assert height_empties[0].source_id is None
    assert all(e.kind == "hu_rung" for e in pkg.ladder)
    assert pkg.ladder[0].name == "LM_HU_0"
    n = min(12, max(1, int((1.0 / (1.0 / 7.5) + 1e-9) // 1)))
    assert len(pkg.ladder) == n + 1
    assert pkg.ladder[-1].name == f"LM_HU_{n}"
    assert pkg.head_unit_m == pytest.approx(1.70 * (1.0 / 7.5))


def test_guides__hu_cap_index_12() -> None:
    """head_unit_frac=0.08 → max index 12 → 13 rungs (0..12)."""
    report = _synthetic_report(
        height_m=1.70,
        head_unit_frac=0.08,
    )
    pkg = build_guide_package(report)
    assert len(pkg.ladder) == 13
    names = [e.name for e in pkg.ladder]
    assert names[0] == "LM_HU_0"
    assert names[-1] == "LM_HU_12"
    assert all(e.source_id is None for e in pkg.ladder)


def test_guides__display_size_clamp() -> None:
    assert display_size_m(0.5) == pytest.approx(0.02)
    assert display_size_m(3.0) == pytest.approx(0.08)
    assert display_size_m(None) == pytest.approx(0.051)
    assert display_size_m(1.7) == pytest.approx(0.051)

    r_low = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
        height_m=0.5,
    )
    pkg_low = build_guide_package(r_low)
    assert pkg_low.empties[0].display_size_m == pytest.approx(0.02)

    r_hi = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
        height_m=3.0,
    )
    pkg_hi = build_guide_package(r_hi)
    assert pkg_hi.empties[0].display_size_m == pytest.approx(0.08)


def test_guides__source_report_schema_echo() -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
        schema_version="1.0.0",
    )
    pkg = build_guide_package(report)
    assert pkg.source_report_schema == "1.0.0"


def test_guides__bpy_contains_honesty_and_collection() -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
        height_m=1.7,
    )
    script = emit_bpy_script(build_guide_package(report))
    assert GUIDE_HONESTY in script
    assert "Proportion_Guides" in script
    assert "to_bu" in script
    assert "scale_length" in script
    assert "Proportion_Seeds" in script
    assert AXIS_NOTES in script or "face -Y" in script


def test_guides__bpy_self_contained() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=0.0, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.1, y_m=0.0, z_m=0.5),
        },
        diameters=[_thigh_diameter()],
        height_m=1.7,
    )
    pkg = build_guide_package(report, seeds=True)
    script = emit_bpy_script(pkg)
    assert "import meshops" not in script
    assert "from meshops" not in script
    assert "meshops." not in script or 'obj["meshops_role"]' in script
    # no meshops module import (allow meshops_role custom prop)
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            assert "meshops" not in stripped
    assert "rotation_difference" in script


def test_guides__bpy_mode_and_none_units() -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
    )
    script = emit_bpy_script(build_guide_package(report))
    assert 'mode != "OBJECT"' in script or "mode != 'OBJECT'" in script
    assert 'mode_set(mode="OBJECT")' in script or "mode_set(mode='OBJECT')" in script
    assert 'system == "NONE"' in script or "system == 'NONE'" in script
    assert 'system = "METRIC"' in script or "system = 'METRIC'" in script
    # never assign scale_length
    assert "scale_length =" not in script.replace(
        'getattr(bpy.context.scene.unit_settings, "scale_length"', ""
    )


def test_guides__seeds_opt_in_capsule() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=0.0, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.01, z_m=0.5),
        },
        diameters=[_thigh_diameter()],
        height_m=1.7,
    )
    pkg = build_guide_package(report, seeds=True)
    names = [s.name for s in pkg.seeds]
    assert "SEED_thigh_l" in names
    seed = next(s for s in pkg.seeds if s.name == "SEED_thigh_l")
    assert seed.kind == "capsule"
    assert seed.band_id == "thigh_l"
    assert seed.radius_m == pytest.approx(0.06)
    assert seed.p0 is not None and seed.p1 is not None
    assert seed.placement == "full3d"
    assert pkg.counts["seeds_front_plane"] == 0


def test_guides__seeds_missing_joint_message() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=0.0, z_m=0.9),
            # knee_l missing
        },
        diameters=[_thigh_diameter()],
    )
    pkg = build_guide_package(report, seeds=True)
    assert not any(s.name == "SEED_thigh_l" for s in pkg.seeds)
    assert any("thigh_l: missing joint knee_l — seed skipped" in m for m in pkg.messages)


def test_guides__seeds_cs_no_height_message() -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
        height_m=None,
        cross_sections=[CrossSection(level_id="waist", z_frac=0.55, rx_frac=0.08, ry_frac=0.06)],
    )
    pkg = build_guide_package(report, seeds=True)
    assert not any(s.kind == "ellipsoid" for s in pkg.seeds)
    assert any("SEED_CS_waist: height_m unset — ellipsoid seed skipped" in m for m in pkg.messages)


def test_guides__seeds_default_off() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=0.0, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.1, y_m=0.0, z_m=0.5),
        },
        diameters=[_thigh_diameter()],
    )
    pkg = build_guide_package(report, seeds=False)
    assert pkg.seeds == []
    assert pkg.counts["seeds"] == 0


def test_guides__empty_report__guides_empty() -> None:
    report = _synthetic_report()
    with pytest.raises(ProportionError) as ei:
        build_guide_package(report)
    assert ei.value.code == "guides_empty"


def test_guides__idempotent_bpy_snippet() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=0.0, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.1, y_m=0.0, z_m=0.5),
        },
        diameters=[_thigh_diameter()],
        height_m=1.7,
    )
    script = emit_bpy_script(build_guide_package(report, seeds=True))
    # empties: update location / empty_display_size
    assert "empty_display_size" in script
    assert "obj.location" in script
    # seeds: full matrix_world not location-only
    assert "matrix_world" in script
    assert "rotation_difference" in script
    assert 'meshops_role"] = "seed"' in script or 'meshops_role"] = "seed"' in script


def test_guides__out_format_conflict(tmp_path: Path) -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
    )
    pkg = build_guide_package(report)
    with pytest.raises(ProportionError) as ei:
        write_guides(tmp_path / "out.py", pkg, format="json")
    assert ei.value.code == "guides_failed"
    assert "conflicts" in str(ei.value).lower()

    with pytest.raises(ProportionError) as ei2:
        write_guides(tmp_path / "out.json", pkg, format="bpy")
    assert ei2.value.code == "guides_failed"


def test_guides__cli_json_shape(tmp_path: Path) -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
            "chin": _lm("chin", x_m=0.0, y_m=0.0, z_m=1.5),
        },
        height_m=1.7,
        head_unit_frac=0.125,
    )
    report_path = tmp_path / "proportion_report.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    out_dir = tmp_path / "guides_out"
    result = runner.invoke(
        app,
        [
            "proportion",
            "guides",
            "--report",
            str(report_path),
            "--out",
            str(out_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["format"] == "both"
    assert isinstance(payload["paths"], list)
    assert len(payload["paths"]) == 2
    assert set(payload["counts"].keys()) >= {
        "empties",
        "ladder",
        "seeds",
        "seeds_front_plane",
    }
    assert payload["counts"]["seeds_front_plane"] == 0
    assert isinstance(payload["messages"], list)
    # disk files
    assert (out_dir / "proportion_guides.json").is_file()
    assert (out_dir / "setup_proportion_guides.py").is_file()
    guides_json = json.loads((out_dir / "proportion_guides.json").read_text(encoding="utf-8"))
    assert guides_json["schema_version"] == "1.1.0"


def test_guides__cli_non_json_honesty(tmp_path: Path) -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
    )
    report_path = tmp_path / "proportion_report.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json")),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "proportion",
            "guides",
            "--report",
            str(report_path),
            "--out",
            str(tmp_path / "g"),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "guides only — not mesh or print success" in result.output


def test_guides__run_guides_helper(tmp_path: Path) -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
    )
    report_path = tmp_path / "proportion_report.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json")),
        encoding="utf-8",
    )
    payload = run_guides(report_path, tmp_path / "out_dir", format="both")
    assert payload["ok"] is True
    assert payload["format"] == "both"
    assert payload["counts"]["empties"] >= 1


def test_guides__sanitize_keeps_underscores() -> None:
    from meshops.proportion.guides import sanitize_landmark_key

    assert sanitize_landmark_key("_foo_") == "_foo_"
    assert sanitize_landmark_key("hip-l") == "hip_l"
    assert sanitize_landmark_key("a__b") == "a_b"
    assert sanitize_landmark_key("@@@") == "_"  # collapsed non-empty underscores kept
    assert sanitize_landmark_key("") == "unnamed"


def test_guides__hu_omitted_message_without_height() -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
        height_m=None,
        head_unit_frac=None,
    )
    pkg = build_guide_package(report)
    assert pkg.ladder == []
    assert any("HU ladder omitted" in m for m in pkg.messages)


def test_guides__bpy_link_failures_not_silent() -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
    )
    script = emit_bpy_script(build_guide_package(report))
    assert "except RuntimeError as exc:" in script
    assert (
        'print(f"warning: could not link' in script or "print(f'warning: could not link" in script
    )
    # no bare pass after RuntimeError
    assert "except RuntimeError:\n            pass" not in script
    assert "except RuntimeError:\n        pass" not in script


def test_guides__seeds_missing_meters_message() -> None:
    """Joint key present but meters incomplete → missing meters (not missing joint)."""
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=0.0, z_m=0.9),
            "knee_l": _lm("knee_l", x=0.1, y=0.0, z=0.3),  # fracs only; meters null
        },
        diameters=[_thigh_diameter()],
    )
    pkg = build_guide_package(report, seeds=True)
    assert not any(s.name == "SEED_thigh_l" for s in pkg.seeds)
    assert any("thigh_l: joint knee_l missing meters — seed skipped" in m for m in pkg.messages)
    assert not any("missing joint knee_l" in m for m in pkg.messages)


def test_guides__format_both_single_file_warn(tmp_path: Path) -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
    )
    pkg = build_guide_package(report)
    out_py = tmp_path / "only.py"
    paths = write_guides(out_py, pkg, format="both")
    assert len(paths) == 1
    assert paths[0] == out_py
    assert any("emitting bpy only" in m for m in pkg.messages)
    assert out_py.is_file()
    assert not (tmp_path / "proportion_guides.json").exists()


def test_guides__exists_without_force__write_failed(tmp_path: Path) -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
    )
    pkg = build_guide_package(report)
    out = tmp_path / "guides"
    write_guides(out, pkg, format="json", force=False)
    with pytest.raises(ProportionError) as ei:
        write_guides(out, pkg, format="json", force=False)
    assert ei.value.code == "write_failed"
    assert "already exists" in str(ei.value).lower()
    # force overwrites
    paths = write_guides(out, pkg, format="json", force=True)
    assert len(paths) == 1


def test_guides__cli_invalid_report(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_report.json"
    bad.write_text("{not valid json", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "proportion",
            "guides",
            "--report",
            str(bad),
            "--out",
            str(tmp_path / "g"),
            "--json",
        ],
    )
    assert result.exit_code != 0
    # structured error payload when --json
    try:
        payload = json.loads(result.output)
        assert payload.get("ok") is False or "error" in payload or "code" in payload
        code = payload.get("code") or (payload.get("error") or {}).get("code")
        if code is not None:
            assert code == "invalid_report"
    except json.JSONDecodeError:
        assert "invalid" in result.output.lower() or "report" in result.output.lower()


# ---------------------------------------------------------------------------
# 0018 R8 — class F/M, front-plane seeds, schema 1.1.0
# ---------------------------------------------------------------------------

_CLASS_F = "front-plane only (y_m null; depth unknown — not missing lateral meters)"
_CLASS_M = "missing *_m components zero-filled (need height_m on analyze for meters)"


def test_guides__empty_only_y_null__class_f() -> None:
    report = _synthetic_report(
        landmarks_xyz={"shoulder_l": _lm("shoulder_l", x_m=-0.2, y_m=None, z_m=1.4)},
    )
    pkg = build_guide_package(report)
    by_name = {e.name: e for e in pkg.empties}
    assert by_name["LM_shoulder_l"].y_m == 0.0
    assert by_name["LM_shoulder_l"].x_m == pytest.approx(-0.2)
    assert by_name["LM_shoulder_l"].z_m == pytest.approx(1.4)
    f_msg = f"shoulder_l: {_CLASS_F}"
    assert f_msg in pkg.messages
    assert not any(_CLASS_M in m for m in pkg.messages if m.startswith("shoulder_l:"))


def test_guides__empty_x_null__class_m() -> None:
    report = _synthetic_report(
        landmarks_xyz={"chin": _lm("chin", x_m=None, y_m=0.0, z_m=1.5)},
    )
    pkg = build_guide_package(report)
    m_msg = f"chin: {_CLASS_M}"
    assert m_msg in pkg.messages
    assert not any(_CLASS_F in m for m in pkg.messages if m.startswith("chin:"))


def test_guides__quiet_null_y_filters_f_keeps_m() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "shoulder_l": _lm("shoulder_l", x_m=-0.2, y_m=None, z_m=1.4),
            "chin": _lm("chin", x_m=None, y_m=0.0, z_m=1.5),
        },
    )
    pkg = build_guide_package(report, quiet_null_y=True)
    assert not any("front-plane only (y_m null" in m for m in pkg.messages)
    assert f"chin: {_CLASS_M}" in pkg.messages


def test_guides__seeds_full_xyz__placement_full3d() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=0.02, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.01, z_m=0.5),
        },
        diameters=[_thigh_diameter()],
    )
    pkg = build_guide_package(report, seeds=True)
    seed = next(s for s in pkg.seeds if s.name == "SEED_thigh_l")
    assert seed.placement == "full3d"
    assert pkg.counts["seeds_front_plane"] == 0


def test_guides__seeds_y_null_no_front_plane__message_cites_first_joint() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=None, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.5),
        },
        diameters=[_thigh_diameter()],
    )
    pkg = build_guide_package(report, seeds=True, front_plane_seeds=False)
    assert not any(s.name == "SEED_thigh_l" for s in pkg.seeds)
    assert any(
        "thigh_l: joint hip_l needs y_m for full3d seed "
        "(use --front-plane-seeds for front-plane capsule)" in m
        for m in pkg.messages
    )


def test_guides__seeds_p1_y_null_cites_p1() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=0.0, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.5),
        },
        diameters=[_thigh_diameter()],
    )
    pkg = build_guide_package(report, seeds=True, front_plane_seeds=False)
    assert not any(s.name == "SEED_thigh_l" for s in pkg.seeds)
    assert any(
        "thigh_l: joint knee_l needs y_m for full3d seed "
        "(use --front-plane-seeds for front-plane capsule)" in m
        for m in pkg.messages
    )


def test_guides__seeds_front_plane_y_null_with_diameter() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=None, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.5),
        },
        diameters=[_thigh_diameter()],
    )
    pkg = build_guide_package(report, seeds=True, front_plane_seeds=True)
    seed = next(s for s in pkg.seeds if s.name == "SEED_thigh_l")
    assert seed.placement == "front_plane"
    assert seed.p0 is not None and seed.p1 is not None
    assert seed.p0[1] == seed.p1[1] == pytest.approx(0.0)
    assert pkg.counts["seeds_front_plane"] == 1


def test_guides__front_plane_no_diameter__no_seed() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=None, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.5),
        },
        diameters=[],
    )
    pkg = build_guide_package(report, seeds=True, front_plane_seeds=True)
    assert not any(s.name == "SEED_thigh_l" for s in pkg.seeds)
    assert any("thigh_l: no usable radius — seed skipped" in m for m in pkg.messages)


def test_guides__front_plane_partial_y_mean() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=0.05, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.5),
        },
        diameters=[_thigh_diameter()],
    )
    pkg = build_guide_package(report, seeds=True, front_plane_seeds=True)
    seed = next(s for s in pkg.seeds if s.name == "SEED_thigh_l")
    assert seed.placement == "front_plane"
    assert seed.p0 is not None and seed.p1 is not None
    assert seed.p0[1] == seed.p1[1] == pytest.approx(0.05)


def test_guides__front_plane_seeds_without_seeds_ignored() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=None, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.5),
            "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        },
        diameters=[_thigh_diameter()],
    )
    pkg = build_guide_package(report, seeds=False, front_plane_seeds=True)
    assert pkg.counts["seeds"] == 0
    assert pkg.counts["seeds_front_plane"] == 0
    assert any("--front-plane-seeds ignored without --seeds" in m for m in pkg.messages)


def test_guides__cs_seeds_with_front_plane_flag() -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
        height_m=1.7,
        cross_sections=[CrossSection(level_id="waist", z_frac=0.55, rx_frac=0.08, ry_frac=0.06)],
    )
    pkg = build_guide_package(report, seeds=True, front_plane_seeds=True)
    cs = next(s for s in pkg.seeds if s.name == "SEED_CS_waist")
    assert cs.kind == "ellipsoid"
    assert cs.placement == "full3d"
    assert pkg.counts["seeds_front_plane"] == 0


def test_guides__schema_write_1_1_0(tmp_path: Path) -> None:
    report = _synthetic_report(
        landmarks_xyz={"sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0)},
    )
    pkg = build_guide_package(report)
    assert pkg.schema_version == "1.1.0"
    assert GUIDE_SCHEMA_VERSION == "1.1.0"
    out = tmp_path / "proportion_guides.json"
    write_guides(out, pkg, format="json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.1.0"
    assert "seeds_front_plane" in data["counts"]


def test_guides__load_1_0_0_fixture_no_placement() -> None:
    """Capture compat: GuidePackage 1.0.0 without placement loads (dual Literal)."""
    raw = {
        "schema_version": "1.0.0",
        "honesty": GUIDE_HONESTY,
        "axis_notes": AXIS_NOTES,
        "empties": [
            {
                "name": "LM_sole",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "kind": "landmark",
                "source_id": "sole",
                "display_size_m": 0.05,
            }
        ],
        "ladder": [],
        "seeds": [
            {
                "name": "SEED_thigh_l",
                "kind": "capsule",
                "band_id": "thigh_l",
                "p0": [-0.1, 0.0, 0.9],
                "p1": [-0.1, 0.0, 0.5],
                "radius_m": 0.06,
                "label": "SEED_thigh_l",
            }
        ],
        "messages": [],
        "counts": {"empties": 1, "ladder": 0, "seeds": 1},
    }
    pkg = GuidePackage.model_validate(raw)
    assert pkg.schema_version == "1.0.0"
    assert pkg.seeds[0].placement == "full3d"


def test_guides__bpy_schema_1_1_0_and_front_plane_comment() -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=None, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.5),
        },
        diameters=[_thigh_diameter()],
    )
    pkg = build_guide_package(report, seeds=True, front_plane_seeds=True)
    script = emit_bpy_script(pkg)
    assert f"# guide schema_version: {GUIDE_SCHEMA_VERSION}" in script
    assert "# guide schema_version: 1.1.0" in script
    assert (
        "# SEED_thigh_l (placement=front_plane, Y-plane=0.000m) "
        "— front-view lateral guide only (N6)" in script
    )
    assert "'placement': 'front_plane'" in script or '"placement": "front_plane"' in script


def test_guides__cli_non_json_echoes_seeds_front_plane(tmp_path: Path) -> None:
    report = _synthetic_report(
        landmarks_xyz={
            "hip_l": _lm("hip_l", x_m=-0.1, y_m=None, z_m=0.9),
            "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.5),
            "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        },
        diameters=[_thigh_diameter()],
    )
    report_path = tmp_path / "proportion_report.json"
    report_path.write_text(json.dumps(report.model_dump(mode="json")), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "proportion",
            "guides",
            "--report",
            str(report_path),
            "--out",
            str(tmp_path / "g"),
            "--format",
            "json",
            "--seeds",
            "--front-plane-seeds",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "seeds_front_plane=1" in result.output
