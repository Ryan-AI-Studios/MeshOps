"""Track 0082 - soft_density compact cull (part clutter policy) T0-T16."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from meshops.proportion.anatomy_profile import load_anatomy_profile
from meshops.proportion.blockout_recipe import (
    COMPACT_CULL_NAME_EXACT,
    COMPACT_CULL_NAME_PREFIXES,
    COMPACT_CULL_ROLES,
    RECIPE_SCHEMA_VERSION,
    SoftDensity,
    _is_compact_cull,
    build_blockout_recipe,
    emit_bpy_script,
    load_blockout_recipe,
    run_blockout_emit_setup,
    run_blockout_recipe,
    write_blockout_recipe,
)
from meshops.proportion.constraints import strip_blender_suffix, validate_constraints
from meshops.proportion.models import (
    DepthBand,
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
    QualityFlags,
)


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


def _band(
    band_id: str,
    *,
    depth_m: float = 0.24,
    z_frac: float = 0.72,
) -> DepthBand:
    return DepthBand(
        band_id=band_id,
        depth_px=50.0,
        depth_frac=0.12,
        depth_m=depth_m,
        y_front=0.1,
        y_back=-0.1,
        y_mid=0.0,
        z_frac=z_frac,
    )


def _product_class_report(*, height_m: float = 1.72) -> ProportionReport:
    """Synthetic product_0072up-class report → 130 parts with full flags + profile."""
    h = height_m
    lms = {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.50),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.68),
        "head_top": _lm("head_top", x_m=0.0, y_m=0.0, z_m=1.72),
        "neck_base": _lm("neck_base", x_m=0.0, y_m=0.0, z_m=1.45),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.05, z_m=1.38),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.05, z_m=1.38),
        "elbow_l": _lm("elbow_l", x_m=-0.25, y_m=0.05, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.25, y_m=0.05, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.30, y_m=0.05, z_m=0.90),
        "wrist_r": _lm("wrist_r", x_m=0.30, y_m=0.05, z_m=0.90),
        "hand_l": _lm("hand_l", x_m=-0.33, y_m=0.05, z_m=0.85),
        "hand_r": _lm("hand_r", x_m=0.33, y_m=0.05, z_m=0.85),
        "fingertip_l": _lm("fingertip_l", x_m=-0.36, y_m=0.05, z_m=0.72),
        "fingertip_r": _lm("fingertip_r", x_m=0.36, y_m=0.05, z_m=0.72),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.95),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.95),
        "crotch": _lm("crotch", x_m=0.0, y_m=0.0, z_m=0.90),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.08, z_m=1.25),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.04, z_m=0.50),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.04, z_m=0.50),
        "ankle_l": _lm("ankle_l", x_m=-0.10, y_m=0.01, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.10, y_m=0.01, z_m=0.08),
        "heel_l": _lm("heel_l", x_m=-0.10, y_m=0.06, z_m=0.02),
        "heel_r": _lm("heel_r", x_m=0.10, y_m=0.06, z_m=0.02),
        "toe_l": _lm("toe_l", x_m=-0.10, y_m=-0.12, z_m=0.02),
        "toe_r": _lm("toe_r", x_m=0.10, y_m=-0.12, z_m=0.02),
    }
    diams = [
        _diam("bust", half_width_m=0.16),
        _diam("waist", half_width_m=0.13),
        _diam("neck", half_width_m=0.05),
        _diam("upper_arm_l", half_width_m=0.0438),
        _diam("upper_arm_r", half_width_m=0.0438),
        _diam("forearm_l", half_width_m=0.0350),
        _diam("forearm_r", half_width_m=0.0350),
        _diam("thigh_l", half_width_m=0.0613),
        _diam("thigh_r", half_width_m=0.0613),
        _diam("calf_l", half_width_m=0.0438),
        _diam("calf_r", half_width_m=0.0438),
        _diam("ank_foot_l", half_width_m=0.0263),
        _diam("ank_foot_r", half_width_m=0.0263),
    ]
    bands = [
        _band("chest", depth_m=0.24, z_frac=0.72),
        _band("breast", depth_m=0.18, z_frac=0.70),
        _band("hip", depth_m=0.26, z_frac=0.55),
        _band("glute", depth_m=0.22, z_frac=0.52),
    ]
    return ProportionReport(
        schema_version="1.2.0",
        height_m=h,
        head_unit_frac=1.0 / 7.5,
        landmarks_xyz=lms,
        diameters=diams,
        depth_bands=bands,
        quality=QualityFlags(),
    )


def _product_flags(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "limbs": True,
        "torso": "ovals",
        "glute": "two_spheres",
        "nofuse": True,
        "face": True,
        "hands": True,
        "feet": True,
        "fingers": "full",
        "toes": "full",
        "profile": load_anatomy_profile("torso_limb_f_athletic_v1"),
    }
    base.update(overrides)
    return base


def _stripped_names(parts: list) -> list[str]:
    return [strip_blender_suffix(p.name) for p in parts]


def _has_name_prefix(parts: list, prefix: str) -> bool:
    return any(n.startswith(prefix) for n in _stripped_names(parts))


def _has_name_exact(parts: list, name: str) -> bool:
    return name in _stripped_names(parts)


def _has_role(parts: list, role: str) -> bool:
    return any(p.role == role for p in parts)


# ---------------------------------------------------------------------------
# T0 freezes
# ---------------------------------------------------------------------------


def test_t0_compact_cull_freezes() -> None:
    """T0: B4 roles frozenset + B5 prefixes exact; no rstrip in match path."""
    assert (
        frozenset(
            {
                "brow_soft",
                "eye_soft",
                "ear_soft",
                "cheek_soft",
                "nose_soft",
                "lip_soft",
                "bicep_soft",
                "mid_back_soft",
                "sternomastoid_soft",
                "trap_soft",
                "clavicle",
            }
        )
        == COMPACT_CULL_ROLES
    )
    assert COMPACT_CULL_NAME_PREFIXES == (
        "RECIPE_triceps_soft_",
        "RECIPE_dist_soft_forearm_",
        "RECIPE_toe_tip_",
        "RECIPE_arch_soft_",
    )
    assert frozenset({"RECIPE_neck_base_soft"}) == COMPACT_CULL_NAME_EXACT
    src = inspect.getsource(_is_compact_cull)
    assert "rstrip" not in src
    assert "startswith" in src


# ---------------------------------------------------------------------------
# T1 default full
# ---------------------------------------------------------------------------


def test_t1_default_soft_density_full() -> None:
    """T1: default soft_density full; no compact cull message."""
    report = _product_class_report()
    pkg = build_blockout_recipe(report, **_product_flags())  # type: ignore[arg-type]
    assert pkg.soft_density == "full"
    assert any(m == "soft_density=full" for m in pkg.messages)
    assert not any("soft_density=compact: culled=" in m for m in pkg.messages)
    assert len(pkg.parts) == 130


# ---------------------------------------------------------------------------
# T2 product-class compact zero B4/B5
# ---------------------------------------------------------------------------


def test_t2_product_compact_zero_b4_b5() -> None:
    """T2: compact product-class has zero B4 roles and zero B5 names."""
    report = _product_class_report()
    pkg = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    for role in COMPACT_CULL_ROLES:
        assert not _has_role(pkg.parts, role), f"B4 role still present: {role}"
    names = _stripped_names(pkg.parts)
    for n in names:
        assert n not in COMPACT_CULL_NAME_EXACT
        for prefix in COMPACT_CULL_NAME_PREFIXES:
            assert not n.startswith(prefix), f"B5 prefix still present: {n}"


# ---------------------------------------------------------------------------
# T3 specific B5 absences
# ---------------------------------------------------------------------------


def test_t3_compact_zero_tip_neck_triceps_arch() -> None:
    """T3: compact zero toe_tip, neck_base_soft, triceps, arch prefixes."""
    report = _product_class_report()
    pkg = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    assert not _has_name_prefix(pkg.parts, "RECIPE_toe_tip_")
    assert not _has_name_exact(pkg.parts, "RECIPE_neck_base_soft")
    assert not _has_name_prefix(pkg.parts, "RECIPE_triceps_soft_")
    assert not _has_name_prefix(pkg.parts, "RECIPE_arch_soft_")
    assert not _has_name_prefix(pkg.parts, "RECIPE_dist_soft_forearm_")


# ---------------------------------------------------------------------------
# T4 jaw + head kept
# ---------------------------------------------------------------------------


def test_t4_compact_jaw_head_present() -> None:
    """T4: jaw + head present when face=True compact."""
    report = _product_class_report()
    pkg = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    assert _has_name_exact(pkg.parts, "RECIPE_jaw")
    assert _has_name_exact(pkg.parts, "RECIPE_head")
    assert _has_role(pkg.parts, "jaw")
    assert _has_role(pkg.parts, "head")


# ---------------------------------------------------------------------------
# T5 structural softs kept
# ---------------------------------------------------------------------------


def test_t5_compact_hip_glute_deltoid_scap() -> None:
    """T5: hip_soft + glute + deltoid + scap present on profile path compact."""
    report = _product_class_report()
    pkg = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    names = _stripped_names(pkg.parts)
    assert any(n.startswith("RECIPE_hip_soft_") for n in names)
    assert any(n.startswith("RECIPE_glute_soft_") for n in names)
    assert any(n.startswith("RECIPE_deltoid_soft_") for n in names)
    assert any(n.startswith("RECIPE_scap_soft_") for n in names)
    assert _has_role(pkg.parts, "glute_soft")
    assert _has_role(pkg.parts, "deltoid_soft")
    assert _has_role(pkg.parts, "scap_soft")


# ---------------------------------------------------------------------------
# T6 elbow + knee kept
# ---------------------------------------------------------------------------


def test_t6_compact_elbow_knee_present() -> None:
    """T6: elbow_soft + knee_soft present when limbs compact."""
    report = _product_class_report()
    pkg = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    names = _stripped_names(pkg.parts)
    assert any(n.startswith("RECIPE_elbow_soft_") for n in names)
    assert any(n.startswith("RECIPE_knee_soft_") for n in names)


# ---------------------------------------------------------------------------
# T7 toes base kept; tips/arch gone; ball present
# ---------------------------------------------------------------------------


def test_t7_compact_toes_ball_arch() -> None:
    """T7: base toes present toes=full; tips absent; ball present; arch absent."""
    report = _product_class_report()
    pkg = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    names = _stripped_names(pkg.parts)
    for side in ("l", "r"):
        for i in range(1, 6):
            assert f"RECIPE_toe_{i}_{side}" in names
            assert f"RECIPE_toe_tip_{i}_{side}" not in names
        assert f"RECIPE_ball_soft_{side}" in names
        assert f"RECIPE_arch_soft_{side}" not in names


# ---------------------------------------------------------------------------
# T8 full path secondary softs present
# ---------------------------------------------------------------------------


def test_t8_full_mid_back_bicep_cheek() -> None:
    """T8: full path mid_back + bicep + cheek present."""
    report = _product_class_report()
    pkg = build_blockout_recipe(report, **_product_flags())  # type: ignore[arg-type]
    assert _has_role(pkg.parts, "mid_back_soft")
    assert _has_role(pkg.parts, "bicep_soft")
    assert _has_role(pkg.parts, "cheek_soft")


# ---------------------------------------------------------------------------
# T9 cull message 37
# ---------------------------------------------------------------------------


def test_t9_compact_cull_message_37() -> None:
    """T9: soft_density=compact: culled=37; names list present."""
    report = _product_class_report()
    pkg = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    cull_msgs = [m for m in pkg.messages if m.startswith("soft_density=compact: culled=")]
    assert len(cull_msgs) == 1
    assert cull_msgs[0] == "soft_density=compact: culled=37"
    name_msgs = [m for m in pkg.messages if m.startswith("soft_density_culled_names=")]
    assert len(name_msgs) == 1
    listed = name_msgs[0].removeprefix("soft_density_culled_names=").split(",")
    assert len(listed) == 37


# ---------------------------------------------------------------------------
# T10 package JSON soft_density round-trip
# ---------------------------------------------------------------------------


def test_t10_package_json_soft_density_roundtrip(tmp_path: Path) -> None:
    """T10: package JSON round-trip soft_density; old JSON without field → full."""
    report = _product_class_report()
    pkg = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    out = tmp_path / "recipe_out"
    paths = write_blockout_recipe(out, pkg, format="json", force=True)
    assert paths
    loaded = load_blockout_recipe(paths[0])
    assert loaded.soft_density == "compact"
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert loaded.schema_version == "1.4.0"

    # Old JSON without soft_density → default full
    raw = json.loads(paths[0].read_text(encoding="utf-8"))
    raw.pop("soft_density", None)
    old_path = tmp_path / "old_recipe.json"
    old_path.write_text(json.dumps(raw), encoding="utf-8")
    old_loaded = load_blockout_recipe(old_path)
    assert old_loaded.soft_density == "full"


# ---------------------------------------------------------------------------
# T11 n_parts == 93
# ---------------------------------------------------------------------------


def test_t11_product_compact_n_parts_93() -> None:
    """T11: product-class compact n_parts == 93 exact."""
    report = _product_class_report()
    full = build_blockout_recipe(report, **_product_flags())  # type: ignore[arg-type]
    compact = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    assert len(full.parts) == 130
    assert len(compact.parts) == 93
    assert compact.counts.get("parts") == 93


# ---------------------------------------------------------------------------
# T12 foot constraints still pass
# ---------------------------------------------------------------------------

_PRODUCT_COMPACT_RECIPE = Path(
    "work/rogue-v3/blockout/product_0072up/nofuse_0082_compact/blockout_recipe.json"
)


def test_t12_compact_foot_constraints() -> None:
    """T12: synthetic compact feet pass C_toe_forward_of_heel + C_toe_sole_z when present."""
    report = _product_class_report()
    pkg = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    result = validate_constraints(pkg, report=report)
    by_id = {r.id: r for r in result.rules}
    for cid in ("C_toe_forward_of_heel", "C_toe_sole_z"):
        assert cid in by_id, f"missing rule {cid}"
        assert by_id[cid].status in ("pass", "skip"), (
            f"{cid} status={by_id[cid].status} msg={by_id[cid].message}"
        )


def test_t12b_product_compact_constraints_if_present() -> None:
    """T12b: product compact recipe hard constraints all pass/skip; toe rules pass.

    Skips when product evidence under work/ is absent (CI without local re-emit).
    """
    path = _PRODUCT_COMPACT_RECIPE
    if not path.is_file():
        pytest.skip("product evidence not present")
    pkg = load_blockout_recipe(path)
    assert pkg.soft_density == "compact"
    assert len(pkg.parts) == 91
    result = validate_constraints(pkg)
    fails = [r for r in result.rules if r.status == "fail"]
    assert not fails, "hard constraint fails on product compact: " + "; ".join(
        f"{r.id}={r.message}" for r in fails
    )
    assert result.ok is True
    by_id = {r.id: r for r in result.rules}
    for cid in ("C_toe_forward_of_heel", "C_toe_sole_z"):
        assert cid in by_id, f"missing rule {cid}"
        assert by_id[cid].status == "pass", (
            f"{cid} status={by_id[cid].status} msg={by_id[cid].message}"
        )


# ---------------------------------------------------------------------------
# T13 fingers tier independent
# ---------------------------------------------------------------------------


def test_t13_compact_keeps_fingers_full() -> None:
    """T13: compact does not change fingers tier (full still emits finger segs)."""
    report = _product_class_report()
    pkg = build_blockout_recipe(
        report,
        **_product_flags(soft_density="compact", fingers="full"),  # type: ignore[arg-type]
    )
    finger_count = sum(1 for p in pkg.parts if p.role == "finger_soft")
    assert finger_count == 24
    assert any(p.role == "thumb_soft" for p in pkg.parts)


# ---------------------------------------------------------------------------
# T14 __all__ exports
# ---------------------------------------------------------------------------


def test_t14_all_exports() -> None:
    """T14: __all__ exports SoftDensity, COMPACT_CULL_ROLES, COMPACT_CULL_NAME_PREFIXES."""
    import meshops.proportion.blockout_recipe as br

    assert "SoftDensity" in br.__all__
    assert "COMPACT_CULL_ROLES" in br.__all__
    assert "COMPACT_CULL_NAME_PREFIXES" in br.__all__
    # type alias is a typing construct; runtime value is the Literal origin
    assert SoftDensity is br.SoftDensity
    assert COMPACT_CULL_ROLES is br.COMPACT_CULL_ROLES
    assert COMPACT_CULL_NAME_PREFIXES is br.COMPACT_CULL_NAME_PREFIXES


# ---------------------------------------------------------------------------
# T15 run payload soft_density
# ---------------------------------------------------------------------------


def test_t15_run_payload_soft_density(tmp_path: Path) -> None:
    """T15: run_blockout_recipe + emit-setup payload soft_density."""
    report = _product_class_report()
    # Write a minimal report JSON for run_blockout_recipe
    report_path = tmp_path / "proportion_report.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    out_dir = tmp_path / "recipe"
    payload = run_blockout_recipe(
        report_path,
        out_dir,
        format="json",
        limbs=True,
        torso="ovals",
        glute="two_spheres",
        nofuse=True,
        face=True,
        hands=True,
        feet=True,
        fingers="full",
        toes="full",
        profiles="torso_limb_f_athletic_v1",
        soft_density="compact",
        force=True,
    )
    assert payload["soft_density"] == "compact"
    assert payload["ok"] is True
    assert payload["counts"]["parts"] == 93

    recipe_json = next(Path(p) for p in payload["paths"] if p.endswith(".json"))
    setup_out = tmp_path / "setup_out"
    emit_payload = run_blockout_emit_setup(recipe_json, setup_out, force=True)
    assert emit_payload["soft_density"] == "compact"

    # bpy header includes soft_density
    pkg = load_blockout_recipe(recipe_json)
    script = emit_bpy_script(pkg)
    assert "# soft_density: compact" in script


# ---------------------------------------------------------------------------
# T16 startswith-only (no rstrip overmatch)
# ---------------------------------------------------------------------------


def test_t16_startswith_only_no_rstrip() -> None:
    """T16: incomplete prefix without trailing '_' does not match."""
    # RECIPE_arch_soft would match if we rstrip prefix trailing '_', but must NOT.
    fake = SimpleNamespace(role="ball_soft", name="RECIPE_arch_soft")
    assert _is_compact_cull(fake) is False  # type: ignore[arg-type]

    # True arch soft WITH trailing suffix after underscore does match.
    real = SimpleNamespace(role="ball_soft", name="RECIPE_arch_soft_l")
    assert _is_compact_cull(real) is True  # type: ignore[arg-type]

    # Incomplete toe tip name without underscore continuation
    tip_incomplete = SimpleNamespace(role="toe_soft", name="RECIPE_toe_tipX")
    assert _is_compact_cull(tip_incomplete) is False  # type: ignore[arg-type]

    tip_real = SimpleNamespace(role="toe_soft", name="RECIPE_toe_tip_1_l")
    assert _is_compact_cull(tip_real) is True  # type: ignore[arg-type]

    # Exact neck base still matches via EXACT set
    neck = SimpleNamespace(role="neck", name="RECIPE_neck_base_soft")
    assert _is_compact_cull(neck) is True  # type: ignore[arg-type]

    # Neck cylinder must not match
    neck_cyl = SimpleNamespace(role="neck", name="RECIPE_neck")
    assert _is_compact_cull(neck_cyl) is False  # type: ignore[arg-type]
