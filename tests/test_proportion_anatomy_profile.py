"""Track 0027 — torso/limb anatomy profiles (offline; no Blender)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.anatomy_profile import (
    ANATOMY_PROFILE_HONESTY,
    list_anatomy_profiles,
    load_anatomy_profile,
)
from meshops.proportion.blockout_recipe import (
    _BASELINE_ROLES_NO_PROFILE,
    _MICHELIN_FRAC,
    RECIPE_SCHEMA_VERSION,
    _midpoint_of_joints,
    build_blockout_recipe,
    load_blockout_recipe,
    write_blockout_recipe,
)
from meshops.proportion.constraints import classify_part_name
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import ANATOMY_PROFILE_HONESTY as HONESTY_TOKEN
from meshops.proportion.models import (
    CrossSection,
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


def _diam(band_id: str, *, half_width_m: float = 0.05) -> DiameterMeasure:
    return DiameterMeasure(
        band_id=band_id,
        view="front",
        width_px=40.0,
        width_eucl_px=40.0,
        theta_deg=90.0,
        width_frac=0.1,
        width_m=half_width_m * 2.0,
        half_width_m=half_width_m,
        mid_x_px=100.0,
        mid_y_px=200.0,
    )


def _rich_report(*, height_m: float = 1.72) -> ProportionReport:
    h = height_m
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.38),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=-0.05, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.28, y_m=-0.05, z_m=1.10),
        "hip_l": _lm("hip_l", x_m=-0.14, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.14, y_m=0.0, z_m=0.95),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.05),
        _diam("upper_arm_r", half_width_m=0.05),
    ]
    bands = [
        DepthBand(
            band_id="chest",
            depth_px=50.0,
            depth_frac=0.12,
            depth_m=0.24,
            y_front=0.1,
            y_back=-0.1,
            y_mid=0.0,
            z_frac=0.72,
        ),
        DepthBand(
            band_id="breast",
            depth_px=40.0,
            depth_frac=0.10,
            depth_m=0.18,
            y_front=0.08,
            y_back=-0.05,
            y_mid=0.0,
            z_frac=0.70,
        ),
        DepthBand(
            band_id="hip",
            depth_px=55.0,
            depth_frac=0.13,
            depth_m=0.26,
            y_front=0.1,
            y_back=-0.1,
            y_mid=0.0,
            z_frac=0.55,
        ),
        DepthBand(
            band_id="glute",
            depth_px=50.0,
            depth_frac=0.12,
            depth_m=0.22,
            y_front=0.05,
            y_back=-0.12,
            y_mid=0.0,
            z_frac=0.52,
        ),
    ]
    cs = [
        CrossSection(
            level_id="bust",
            z_frac=0.70,
            rx_frac=0.09,
            ry_frac=0.06,
        ),
        CrossSection(
            level_id="glute",
            z_frac=0.50,
            rx_frac=0.08,
            ry_frac=0.07,
        ),
    ]
    return ProportionReport(
        schema_version="1.2.0",
        height_m=h,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=bands,
        cross_sections=cs,
        quality=QualityFlags(),
    )


def _skeleton_with_arms() -> BlockoutSkeleton:
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
        j("neck_base", x=0.0, y=0.0, z=1.42, parent="spine_high"),
        j("shoulder_l", x=-0.20, y=0.0, z=1.38, side="l", parent="spine_high"),
        j("shoulder_r", x=0.20, y=0.0, z=1.38, side="r", parent="spine_high"),
        j("elbow_l", x=-0.28, y=-0.05, z=1.10, side="l", parent="shoulder_l"),
        j("elbow_r", x=0.28, y=-0.05, z=1.10, side="r", parent="shoulder_r"),
    ]
    bones = [
        SkeletonBone(id="spine", joint_a="pelvis", joint_b="spine_high", length_m=0.3),
        SkeletonBone(id="upper_arm_l", joint_a="shoulder_l", joint_b="elbow_l", length_m=0.3),
        SkeletonBone(id="upper_arm_r", joint_a="shoulder_r", joint_b="elbow_r", length_m=0.3),
    ]
    return BlockoutSkeleton(
        schema_version="1.0.0",
        honesty="proportion_blockout_skeleton_not_mesh_or_print_success",
        joints=joints,
        bones=bones,
        messages=[],
    )


# ---------------------------------------------------------------------------
# List / load
# ---------------------------------------------------------------------------


def test_profiles__list_both_ids() -> None:
    rows = list_anatomy_profiles()
    ids = {r["id"] for r in rows}
    assert ids == {"torso_limb_f_athletic_v1", "torso_limb_m_athletic_v1"}
    for r in rows:
        assert r["description"]
        assert r["sex"] in ("female", "male")
        assert r["archetype"] == "adult_athletic"
        assert "template_id_hint" in r


def test_profiles__honesty_exact() -> None:
    assert ANATOMY_PROFILE_HONESTY == "proportion_anatomy_profile_not_mesh_or_print_success"
    assert HONESTY_TOKEN == ANATOMY_PROFILE_HONESTY
    for pid in ("torso_limb_f_athletic_v1", "torso_limb_m_athletic_v1"):
        doc = load_anatomy_profile(pid)
        assert doc.honesty == ANATOMY_PROFILE_HONESTY
        assert doc.schema_version == "1.0.0"


def test_profiles__cli_list_and_json() -> None:
    r = runner.invoke(app, ["proportion", "anatomy-profiles"])
    assert r.exit_code == 0, r.output
    assert "torso_limb_f_athletic_v1" in r.output
    assert "torso_limb_m_athletic_v1" in r.output
    r2 = runner.invoke(app, ["proportion", "anatomy-profiles", "--json"])
    assert r2.exit_code == 0, r2.output
    data = json.loads(r2.output)
    assert data["ok"] is True
    assert data["honesty"] == ANATOMY_PROFILE_HONESTY
    ids = {p["id"] for p in data["profiles"]}
    assert ids == {"torso_limb_f_athletic_v1", "torso_limb_m_athletic_v1"}


def test_profiles__unknown() -> None:
    with pytest.raises(ProportionError) as ei:
        load_anatomy_profile("not_a_real_profile")
    assert ei.value.code == "profile_unknown"


# ---------------------------------------------------------------------------
# Emit — female / male
# ---------------------------------------------------------------------------


def test_profile__female_emit_dual_mass_and_traps() -> None:
    report = _rich_report()
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    skel = _skeleton_with_arms()
    pkg = build_blockout_recipe(
        report,
        limbs=False,
        glute="oval",
        profile=profile,
        skeleton=skel,
    )
    assert pkg.schema_version == "1.2.0"
    roles = {p.role for p in pkg.parts}
    names = {p.name for p in pkg.parts}

    assert "trap_soft" in roles
    assert "scap_soft" in roles
    assert "clavicle" in roles
    assert "RECIPE_trap_soft_l" in names and "RECIPE_trap_soft_r" in names
    assert "RECIPE_breast_soft_l" in names and "RECIPE_breast_soft_r" in names
    assert "RECIPE_glute_soft_l" in names and "RECIPE_glute_soft_r" in names
    assert "RECIPE_scap_soft_l" in names and "RECIPE_scap_soft_r" in names
    assert "RECIPE_clavicle_l" in names and "RECIPE_clavicle_r" in names
    # no mid-chest singleton
    mid_breast = [p for p in pkg.parts if p.name == "RECIPE_breast_soft"]
    assert mid_breast == []
    # exactly one L/R breast and deltoid
    assert sum(1 for p in pkg.parts if p.name == "RECIPE_breast_soft_l") == 1
    assert sum(1 for p in pkg.parts if p.name == "RECIPE_deltoid_soft_l") == 1

    breasts = [p for p in pkg.parts if p.role == "breast_soft" and p.center]
    assert len(breasts) == 2
    xs = sorted(p.center[0] for p in breasts if p.center)
    assert xs[0] < 0 < xs[1]
    assert abs(xs[1] - xs[0]) > 0.02  # not coincident when gap set
    for p in breasts:
        assert p.center is not None
        assert p.center[1] < 0.0  # breast -Y (B8)

    traps = [p for p in pkg.parts if p.role == "trap_soft" and p.center]
    assert len(traps) == 2
    txs = [p.center[0] for p in traps if p.center]
    assert min(txs) < max(txs)  # L ≠ R

    # D4: parent_joint side-correct (not left-default on R)
    by_name = {p.name: p for p in pkg.parts}
    assert by_name["RECIPE_bicep_soft_l"].parent_joint == "shoulder_l"
    assert by_name["RECIPE_bicep_soft_r"].parent_joint == "shoulder_r"
    assert by_name["RECIPE_clavicle_l"].parent_joint == "shoulder_l"
    assert by_name["RECIPE_clavicle_r"].parent_joint == "shoulder_r"
    assert by_name["RECIPE_deltoid_soft_r"].parent_joint == "shoulder_r"


def test_profile__male_pec_no_breast_dual_mild_glute() -> None:
    report = _rich_report()
    profile = load_anatomy_profile("torso_limb_m_athletic_v1")
    pkg = build_blockout_recipe(
        report,
        limbs=False,
        glute="oval",  # CLI oval — profile must still dual glute
        profile=profile,
    )
    names = {p.name for p in pkg.parts}
    roles = {p.role for p in pkg.parts}
    assert "pec_soft" in roles
    assert "RECIPE_pec_soft_l" in names and "RECIPE_pec_soft_r" in names
    assert not any(p.role == "breast_soft" for p in pkg.parts)
    glutes = [p for p in pkg.parts if p.role == "glute_soft"]
    assert len(glutes) >= 2
    assert any("dual glute" in m or "profile glutes" in m for m in pkg.messages)


def test_profile__glute_y_nonneg() -> None:
    report = _rich_report()
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    pkg = build_blockout_recipe(report, limbs=False, profile=profile)
    for p in pkg.parts:
        if p.role == "glute_soft" and p.center is not None:
            assert p.center[1] >= 0.0


def test_profile__bicep_mid_when_joints_finite() -> None:
    report = _rich_report()
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    skel = _skeleton_with_arms()
    pkg = build_blockout_recipe(report, limbs=False, profile=profile, skeleton=skel)
    joints = {j.id: j for j in skel.joints}
    expected = _midpoint_of_joints(joints, "shoulder_l", "elbow_l")
    assert expected is not None
    bicep_l = next(p for p in pkg.parts if p.name == "RECIPE_bicep_soft_l")
    assert bicep_l.center is not None
    assert bicep_l.center[0] == pytest.approx(expected[0], abs=1e-6)
    assert bicep_l.center[1] == pytest.approx(expected[1], abs=1e-6)
    assert bicep_l.center[2] == pytest.approx(expected[2], abs=1e-6)


def test_profile__trap_mid_l_ne_r() -> None:
    report = _rich_report()
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    skel = _skeleton_with_arms()
    pkg = build_blockout_recipe(report, limbs=False, profile=profile, skeleton=skel)
    joints = {j.id: j for j in skel.joints}
    mid_l = _midpoint_of_joints(joints, "neck_base", "shoulder_l")
    mid_r = _midpoint_of_joints(joints, "neck_base", "shoulder_r")
    assert mid_l is not None and mid_r is not None
    assert mid_l[0] != mid_r[0]
    trap_l = next(p for p in pkg.parts if p.name == "RECIPE_trap_soft_l")
    trap_r = next(p for p in pkg.parts if p.name == "RECIPE_trap_soft_r")
    assert trap_l.center is not None and trap_r.center is not None
    assert trap_l.center[0] == pytest.approx(mid_l[0], abs=1e-5)
    assert trap_r.center[0] == pytest.approx(mid_r[0], abs=1e-5)
    assert trap_l.center[0] != trap_r.center[0]


def test_profile__no_profiles_excludes_new_roles() -> None:
    report = _rich_report()
    pkg = build_blockout_recipe(report, limbs=True)
    roles = {p.role for p in pkg.parts}
    assert roles <= _BASELINE_ROLES_NO_PROFILE
    for r in ("trap_soft", "pec_soft", "scap_soft", "bicep_soft", "clavicle"):
        assert r not in roles


def test_profile__unknown_via_run(tmp_path: Path) -> None:
    from meshops.proportion.blockout_recipe import run_blockout_recipe

    report = _rich_report()
    rp = tmp_path / "proportion_report.json"
    rp.write_text(json.dumps(report.model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ProportionError) as ei:
        run_blockout_recipe(rp, tmp_path / "out", format="json", profiles="nope_v9")
    assert ei.value.code == "profile_unknown"


def test_recipe__write_load_1_1_0_parent_joint_roundtrip(tmp_path: Path) -> None:
    report = _rich_report()
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    skel = _skeleton_with_arms()
    pkg = build_blockout_recipe(report, limbs=False, profile=profile, skeleton=skel)
    assert pkg.schema_version == RECIPE_SCHEMA_VERSION == "1.2.0"
    # at least one part with parent_joint set
    with_pj = [p for p in pkg.parts if p.parent_joint]
    assert with_pj, "expected parent_joint filled from skeleton"
    paths = write_blockout_recipe(tmp_path / "r", pkg, format="json", force=True)
    loaded = load_blockout_recipe(paths[0])
    assert loaded.schema_version == "1.2.0"
    loaded_map = {p.name: p.parent_joint for p in loaded.parts}
    for p in with_pj:
        assert loaded_map.get(p.name) == p.parent_joint


def test_classifier__profile_roles() -> None:
    cases = [
        ("RECIPE_torso_trap", "torso"),  # bare trap must not steal 0019 default torso
        ("RECIPE_trap_soft_l", "neck"),
        ("RECIPE_scap_soft_r", "torso"),
        ("RECIPE_bicep_soft_l", "upper_arm"),
        ("RECIPE_clavicle_r", "shoulder_bridge"),
        ("RECIPE_pec_soft_l", "breast"),
        ("RECIPE_deltoid_soft_l", "deltoid"),
    ]
    for name, role in cases:
        got, side = classify_part_name(name)
        assert got == role, f"{name} → {got} (want {role})"
        assert side in ("l", "r", "none")


def test_michelin_override_per_part() -> None:
    report = _rich_report()
    profile = load_anatomy_profile("torso_limb_f_athletic_v1")
    # female pack delts michelin_cap_frac_h=0.045 → cap 0.045*1.72 ≈ 0.0774
    # global _MICHELIN_FRAC * shoulder_hw = 0.45 * 0.20 = 0.09
    # force large diameter so clamp engages at override
    report.diameters = [d for d in report.diameters if not d.band_id.startswith("upper_arm")] + [
        _diam("upper_arm_l", half_width_m=0.20),
        _diam("upper_arm_r", half_width_m=0.20),
    ]
    pkg = build_blockout_recipe(report, limbs=False, profile=profile)
    h = report.height_m or 1.72
    cap = 0.045 * h
    delts = [p for p in pkg.parts if p.role == "deltoid_soft"]
    assert delts
    for p in delts:
        assert p.rx_m is not None
        assert p.rx_m <= cap + 1e-6
    assert any("michelin_cap_frac_h" in m for m in pkg.messages) or all(
        (p.rx_m or 0) <= cap + 1e-6 for p in delts
    )
    # override differs from pure global*shoulder when both would clamp
    global_cap = _MICHELIN_FRAC * 0.20
    assert cap < global_cap


def test_mcp__anatomy_profiles_catalog_and_recipe_params() -> None:
    pytest.importorskip("mcp")
    import asyncio

    from mcp import Client

    from meshops.mcp.server import TOOL_NAMES, build_server

    assert "mesh_proportion_anatomy_profiles" in TOOL_NAMES
    assert len(TOOL_NAMES) >= 43

    async def _body() -> None:
        server = build_server()
        async with Client(server) as client:
            listed = await client.list_tools()
            names = {t.name for t in listed.tools}
            assert "mesh_proportion_anatomy_profiles" in names
            assert len(names) >= 43
            recipe = next(t for t in listed.tools if t.name == "mesh_proportion_blockout_recipe")
            raw_schema: object | None = getattr(recipe, "input_schema", None)
            if raw_schema is None:
                raw_schema = getattr(recipe, "inputSchema", None)
            schema: dict[str, object] = {}
            if isinstance(raw_schema, dict):
                schema = raw_schema
            props_obj = schema.get("properties")
            props = props_obj if isinstance(props_obj, dict) else {}
            assert "profiles" in props
            assert "skeleton" in props

    asyncio.run(_body())


def test_offline_no_network_imports() -> None:
    """Smoke: load packs without network."""
    assert load_anatomy_profile("torso_limb_f_athletic_v1").id.startswith("torso_limb")
    assert load_anatomy_profile("torso_limb_m_athletic_v1").sex == "male"
