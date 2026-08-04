"""Track 0028 - head / face / hair / neckline RECIPE kit (offline; no Blender)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.blockout_recipe import (
    RECIPE_SCHEMA_VERSION,
    RecipePart,
    build_blockout_recipe,
    load_blockout_recipe,
    write_blockout_recipe,
)
from meshops.proportion.constraints import classify_part_name, validate_constraints
from meshops.proportion.face_recipe import (
    FACE_KIT_SKIP_BOUNDS,
    HeadBounds,
    build_face_parts,
    resolve_head_bounds,
)
from meshops.proportion.models import (
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)
from meshops.proportion.skeleton import BlockoutSkeleton, SkeletonBone, SkeletonJoint

runner = CliRunner()


def _lm(
    id_: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=id_, x_m=x_m, y_m=y_m, z_m=z_m)


def _diam(
    band_id: str,
    *,
    half_width_m: float | None = 0.05,
) -> DiameterMeasure:
    w = half_width_m * 2.0 if half_width_m is not None else 0.1
    return DiameterMeasure(
        band_id=band_id,
        view="front",
        width_px=40.0,
        width_eucl_px=40.0,
        theta_deg=90.0,
        width_frac=0.1,
        width_m=w,
        half_width_m=half_width_m,
        mid_x_px=100.0,
        mid_y_px=200.0,
    )


def _depth_band(
    band_id: str,
    *,
    depth_m: float = 0.22,
    z_frac: float = 0.72,
    y_mid: float = 0.0,
) -> DepthBand:
    return DepthBand(
        band_id=band_id,
        depth_px=50.0,
        depth_frac=0.12,
        depth_m=depth_m,
        y_front=0.1,
        y_back=-0.1,
        y_mid=y_mid,
        z_frac=z_frac,
    )


def _full_torso_report(
    *,
    height_m: float = 1.72,
    chin_z: float = 1.50,
    shoulder_z: float = 1.38,
    hip_z: float = 0.95,
    shoulder_x: float = 0.20,
    hip_x: float = 0.14,
    include_chin: bool = True,
    head_unit_frac: float = 1.0 / 7.5,
    extra_lms: dict[str, LandmarkXYZ] | None = None,
) -> ProportionReport:
    lms: dict[str, LandmarkXYZ] = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
    }
    if include_chin:
        lms["chin"] = _lm("chin", x_m=0.0, y_m=-0.02, z_m=chin_z)
    lms["shoulder_l"] = _lm("shoulder_l", x_m=-shoulder_x, y_m=0.0, z_m=shoulder_z)
    lms["shoulder_r"] = _lm("shoulder_r", x_m=shoulder_x, y_m=0.0, z_m=shoulder_z)
    lms["hip_l"] = _lm("hip_l", x_m=-hip_x, y_m=0.0, z_m=hip_z)
    lms["hip_r"] = _lm("hip_r", x_m=hip_x, y_m=0.0, z_m=hip_z)
    lms["cranial_vertex"] = _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=chin_z + 0.18)
    lms["crotch_pubic"] = _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86)
    lms["chest_mid"] = _lm("chest_mid", x_m=0.0, y_m=0.0, z_m=1.25)
    lms["chest_front"] = _lm("chest_front", x_m=0.0, y_m=-0.08, z_m=1.25)
    if extra_lms:
        lms.update(extra_lms)

    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.05),
        _diam("upper_arm_r", half_width_m=0.05),
        _diam("thigh_l", half_width_m=0.05),
        _diam("thigh_r", half_width_m=0.05),
    ]
    bands = [
        _depth_band("chest", depth_m=0.24, z_frac=0.72),
        _depth_band("hip", depth_m=0.26, z_frac=0.55),
    ]
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=head_unit_frac,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=bands,
        quality=QualityFlags(),
    )


def _head_skeleton() -> BlockoutSkeleton:
    def j(
        id_: str,
        *,
        x: float,
        y: float,
        z: float,
        side: str = "none",
        parent: str | None = None,
    ) -> SkeletonJoint:
        return SkeletonJoint(
            id=id_,
            parent=parent,
            side=side,  # type: ignore[arg-type]
            x_m=x,
            y_m=y,
            z_m=z,
            source="estimated",
        )

    joints = [
        j("root", x=0.0, y=0.0, z=0.0),
        j("pelvis", x=0.0, y=0.0, z=0.95, parent="root"),
        j("spine_high", x=0.0, y=0.0, z=1.25, parent="pelvis"),
        j("neck_base", x=0.0, y=0.0, z=1.40, parent="spine_high"),
        j("neck_top", x=0.0, y=0.0, z=1.48, parent="neck_base"),
        j("chin", x=0.0, y=-0.02, z=1.50, parent="neck_top"),
        j("head", x=0.0, y=-0.01, z=1.59, parent="chin"),
        j("crown", x=0.0, y=-0.01, z=1.68, parent="head"),
    ]
    bones = [
        SkeletonBone(id="spine", joint_a="pelvis", joint_b="spine_high", length_m=0.3),
        SkeletonBone(id="neck", joint_a="neck_base", joint_b="neck_top", length_m=0.08),
        SkeletonBone(id="head_bone", joint_a="neck_top", joint_b="head", length_m=0.11),
    ]
    return BlockoutSkeleton(
        schema_version="1.0.0",
        honesty="proportion_blockout_skeleton_not_mesh_or_print_success",
        joints=joints,
        bones=bones,
        messages=[],
    )


# ---------------------------------------------------------------------------
# R7 cases
# ---------------------------------------------------------------------------


def test_face__default_no_face_roles() -> None:
    """B6 / R7: without flags, no jaw/eye/hair/neckline roles."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False)
    roles = {p.role for p in pkg.parts}
    face_roles = {
        "jaw",
        "brow_soft",
        "eye_soft",
        "nose_soft",
        "ear_soft",
        "lip_soft",
        "hair_mass",
        "neckline",
        "sternomastoid_soft",
    }
    assert roles.isdisjoint(face_roles)
    assert "RECIPE_neck_head_fuse" not in {p.name for p in pkg.parts}
    assert pkg.schema_version == RECIPE_SCHEMA_VERSION == "1.2.0"


def test_face__face_flag_core_features() -> None:
    """R7: --face emits jaw + 2 eyes + nose + 2 ears + lip; all RECIPE_*."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    by_role = pkg.counts.get("by_role") or {}
    assert by_role.get("jaw", 0) >= 1
    assert by_role.get("eye_soft", 0) >= 2
    assert by_role.get("nose_soft", 0) >= 1
    assert by_role.get("ear_soft", 0) >= 2
    assert by_role.get("lip_soft", 0) >= 1
    names = {p.name for p in pkg.parts}
    assert "RECIPE_jaw" in names
    assert "RECIPE_eye_soft_l" in names and "RECIPE_eye_soft_r" in names
    assert "RECIPE_nose_soft" in names
    assert "RECIPE_ear_soft_l" in names and "RECIPE_ear_soft_r" in names
    assert "RECIPE_lip_soft" in names
    for p in pkg.parts:
        assert p.name.startswith("RECIPE_")
        assert p.label.startswith("RECIPE_")


def test_face__hair_bun_vs_none() -> None:
    report = _full_torso_report()
    pkg_none = build_blockout_recipe(report, limbs=False, hair="none")
    pkg_bun = build_blockout_recipe(report, limbs=False, hair="bun")
    assert pkg_bun.counts["parts"] > pkg_none.counts["parts"]
    roles_bun = {p.role for p in pkg_bun.parts}
    assert "hair_mass" in roles_bun
    assert not any(p.role == "hair_mass" for p in pkg_none.parts)
    bun_names = {p.name for p in pkg_bun.parts if p.role == "hair_mass"}
    assert "RECIPE_hair_mass" in bun_names
    assert "RECIPE_hair_mass_bun" in bun_names


def test_face__neckline_crew() -> None:
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, neckline="crew")
    assert any(p.role == "neckline" for p in pkg.parts)
    assert any(p.name == "RECIPE_neckline_crew" for p in pkg.parts)


def test_face__skeleton_parent_joint() -> None:
    report = _full_torso_report()
    skel = _head_skeleton()
    pkg = build_blockout_recipe(report, limbs=False, face=True, skeleton=skel)
    by_name = {p.name: p for p in pkg.parts}
    jaw = by_name["RECIPE_jaw"]
    assert jaw.parent_joint in ("chin", "head")
    eye = by_name["RECIPE_eye_soft_l"]
    assert eye.parent_joint == "head"


def test_face__schema_load_1_1_and_write_1_2(tmp_path: Path) -> None:
    """Load 1.1.0 without face roles OK; write 1.2.0; load 1.2.0 round-trip."""
    report = _full_torso_report()
    # Craft a 1.1.0 package without face roles
    pkg11 = build_blockout_recipe(report, limbs=False)
    data = pkg11.model_dump(mode="json")
    data["schema_version"] = "1.1.0"
    # Strip any 1.2-only roles just in case
    data["parts"] = [p for p in data["parts"] if p["role"] not in ("jaw", "hair_mass")]
    p11 = tmp_path / "recipe_1_1.json"
    p11.write_text(json.dumps(data, indent=2), encoding="utf-8")
    loaded11 = load_blockout_recipe(p11)
    assert loaded11.schema_version == "1.1.0"

    pkg12 = build_blockout_recipe(report, limbs=False, face=True)
    assert pkg12.schema_version == "1.2.0"
    paths = write_blockout_recipe(tmp_path / "out12", pkg12, format="json", force=True)
    loaded12 = load_blockout_recipe(paths[0])
    assert loaded12.schema_version == "1.2.0"
    assert any(p.role == "jaw" for p in loaded12.parts)


def test_face__classifier_b5_tokens() -> None:
    cases = [
        ("RECIPE_jaw", "head"),
        ("RECIPE_brow_soft_l", "head"),
        ("RECIPE_eye_soft_r", "head"),
        ("RECIPE_nose_soft", "head"),
        ("RECIPE_ear_soft_l", "head"),
        ("RECIPE_lip_soft", "head"),
        ("RECIPE_hair_mass", "head"),
        ("RECIPE_hair_mass_bun", "head"),
        ("RECIPE_sternomastoid_soft_l", "neck"),
        ("RECIPE_neckline_crew", "neck"),
        ("RECIPE_neckline_v_l", "neck"),
        ("RECIPE_neck_head_fuse", "neck"),
        ("RECIPE_head", "head"),
        ("RECIPE_neck", "neck"),
    ]
    for name, expected in cases:
        role, _side = classify_part_name(name)
        assert role == expected, f"{name} → {role}, expected {expected}"


def test_face__forbidden_face_prefix_and_recipe_jaw_ok() -> None:
    with pytest.raises(ValueError, match="RECIPE_"):
        RecipePart(
            name="FACE_jaw",
            role="jaw",
            kind="box",
            center=[0.0, 0.0, 1.5],
            top_half_width_m=0.05,
            bottom_half_width_m=0.04,
            half_depth_m=0.03,
            z_bottom_m=1.45,
            z_top_m=1.55,
        )
    ok = RecipePart(
        name="RECIPE_jaw",
        role="jaw",
        kind="box",
        center=[0.0, 0.0, 1.5],
        top_half_width_m=0.05,
        bottom_half_width_m=0.04,
        half_depth_m=0.03,
        z_bottom_m=1.45,
        z_top_m=1.55,
    )
    assert ok.role == "jaw"
    assert ok.name == "RECIPE_jaw"


def test_face__headbounds_shared_with_head_and_eyes() -> None:
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    eye = next(p for p in pkg.parts if p.name == "RECIPE_eye_soft_l")
    assert head.center is not None and eye.center is not None
    # Rebuild HeadBounds independently and check consistency
    msgs: list[str] = []
    bounds = resolve_head_bounds(
        report,
        head_unit_m=pkg.head_unit_m,
        height_m=pkg.height_m,
        messages=msgs,
    )
    assert bounds is not None
    assert head.center[2] == pytest.approx(bounds.z_c)
    assert head.rx_m == pytest.approx(bounds.rx)
    assert head.rz_m == pytest.approx(bounds.rz)
    expected_eye_z = bounds.z_chin + 0.50 * bounds.H
    assert eye.center[2] == pytest.approx(expected_eye_z)


def test_face__axial_exemption_with_face() -> None:
    """B18: --face recipe must not fail C_axial_depth_plane from face softs."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    axial = by_id["C_axial_depth_plane"]
    assert axial.status != "fail", axial.message
    # Nose is forward (-Y) but exempt; should not appear as fail message
    assert "nose_soft" not in (axial.message or "")
    assert "eye_soft" not in (axial.message or "")


def test_face__scm_and_fuse() -> None:
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    scm = [p for p in pkg.parts if p.role == "sternomastoid_soft"]
    assert len(scm) == 2
    # Normal neck reaches chin → small gap → no fuse
    assert not any(p.name == "RECIPE_neck_head_fuse" for p in pkg.parts)

    # Direct fuse: large gap emits; small gap does not
    msgs: list[str] = []
    bounds = resolve_head_bounds(
        report,
        head_unit_m=1.72 / 7.5,
        height_m=1.72,
        messages=msgs,
    )
    assert bounds is not None
    head_bottom = bounds.z_c - bounds.rz
    large = build_face_parts(
        report,
        bounds,
        face=True,
        neck_top_z=head_bottom - 0.10,
        neck_radius=0.05,
        head_unit_m=1.72 / 7.5,
        shoulder_z=1.38,
        messages=msgs,
    )
    assert any(p.name == "RECIPE_neck_head_fuse" for p in large)
    small = build_face_parts(
        report,
        bounds,
        face=True,
        neck_top_z=head_bottom - 0.001,
        neck_radius=0.05,
        head_unit_m=1.72 / 7.5,
        shoulder_z=1.38,
        messages=[],
    )
    assert not any(p.name == "RECIPE_neck_head_fuse" for p in small)


def test_face__scm_absent_message() -> None:
    """B14: no shoulder_z / neck metrics → 0 SCM + skip message."""
    report = _full_torso_report()
    bounds = resolve_head_bounds(report, head_unit_m=1.72 / 7.5, height_m=1.72, messages=[])
    assert bounds is not None
    msgs: list[str] = []
    parts = build_face_parts(
        report,
        bounds,
        face=True,
        neck_top_z=None,
        neck_radius=None,
        head_unit_m=1.72 / 7.5,
        shoulder_z=None,
        messages=msgs,
    )
    assert not any(p.role == "sternomastoid_soft" for p in parts)
    assert any("sternomastoid skipped" in m for m in msgs)


def test_face__hair_short_and_long_proxy() -> None:
    report = _full_torso_report()
    short = build_blockout_recipe(report, limbs=False, hair="short")
    longp = build_blockout_recipe(report, limbs=False, hair="long_proxy")
    none = build_blockout_recipe(report, limbs=False, hair="none")
    assert any(p.role == "hair_mass" for p in short.parts)
    assert any(p.role == "hair_mass" for p in longp.parts)
    assert not any(p.role == "hair_mass" for p in none.parts)
    # long_proxy rear (+Y) of head center
    head = next(p for p in longp.parts if p.name == "RECIPE_head")
    hair = next(p for p in longp.parts if p.role == "hair_mass")
    assert head.center is not None and hair.center is not None
    assert hair.center[1] > head.center[1]


def test_face__nose_tip_y_freeze() -> None:
    """B7: nose tip Y = head_center_y - 0.15 * head_ry."""
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, face=True)
    head = next(p for p in pkg.parts if p.name == "RECIPE_head")
    nose = next(p for p in pkg.parts if p.name == "RECIPE_nose_soft")
    assert head.center is not None and head.ry_m is not None
    assert nose.p1 is not None
    expected_tip_y = float(head.center[1]) - 0.15 * float(head.ry_m)
    assert nose.p1[1] == pytest.approx(expected_tip_y, abs=1e-6)


def test_face__neckline_v_proxy() -> None:
    report = _full_torso_report()
    pkg = build_blockout_recipe(report, limbs=False, neckline="v_proxy")
    v_parts = [p for p in pkg.parts if p.role == "neckline"]
    assert len(v_parts) == 2
    assert all(p.name.startswith("RECIPE_neckline") for p in v_parts)


def test_face__mcp_schema_properties_and_tool_count() -> None:
    """B12: catalog stay 43; inputSchema properties include face/hair/neckline."""
    import asyncio

    pytest.importorskip("mcp")
    from mcp import Client

    from meshops.mcp import TOOL_NAMES
    from meshops.mcp.server import build_server

    assert len(TOOL_NAMES) == 43

    async def _body() -> None:
        server = build_server()
        async with Client(server) as client:
            listed = await client.list_tools()
            names = {t.name for t in listed.tools}
            assert len(names) >= 43
            assert names >= TOOL_NAMES
            tool = next(t for t in listed.tools if t.name == "mesh_proportion_blockout_recipe")
            schema = tool.input_schema
            assert schema is not None
            props = schema.get("properties") or {}
            assert "face" in props
            assert "hair" in props
            assert "neckline" in props

    asyncio.run(_body())


def test_face__invalid_hair_neckline_cli() -> None:
    # Invalid enum fails before report load
    r = runner.invoke(
        app,
        [
            "proportion",
            "blockout-recipe",
            "--report",
            "nonexistent.json",
            "--out",
            "out",
            "--hair",
            "mohawk",
        ],
    )
    assert r.exit_code != 0
    r2 = runner.invoke(
        app,
        [
            "proportion",
            "blockout-recipe",
            "--report",
            "nonexistent.json",
            "--out",
            "out",
            "--neckline",
            "turtleneck",
        ],
    )
    assert r2.exit_code != 0


def test_face__mcp_invalid_enums() -> None:
    from meshops.mcp.tools import mesh_proportion_blockout_recipe

    with pytest.raises(ValueError, match="hair"):
        mesh_proportion_blockout_recipe(
            Path("."),
            report="r.json",
            out="o",
            hair="mohawk",
        )
    with pytest.raises(ValueError, match="neckline"):
        mesh_proportion_blockout_recipe(
            Path("."),
            report="r.json",
            out="o",
            neckline="turtleneck",
        )


def test_face__skip_when_chin_unresolved() -> None:
    report = _full_torso_report(include_chin=False)
    pkg = build_blockout_recipe(report, limbs=False, face=True, hair="short")
    assert FACE_KIT_SKIP_BOUNDS in pkg.messages
    assert not any(p.role == "jaw" for p in pkg.parts)
    assert not any(p.role == "hair_mass" for p in pkg.parts)


def test_face__headbounds_type() -> None:
    b = HeadBounds(
        z_chin=1.5,
        z_top=1.68,
        z_c=1.59,
        H=0.18,
        rx=0.08,
        ry=0.07,
        rz=0.09,
        y=-0.02,
        placement="full3d",
        has_y=True,
    )
    assert pytest.approx(0.18) == b.H
