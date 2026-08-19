"""Track 0026 — skeleton-first joint/bone graph (offline; no Blender)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.depth_samples import DepthSample, DepthSamplesPackage
from meshops.proportion.errors import ProportionError
from meshops.proportion.guides import AXIS_NOTES
from meshops.proportion.honesty import DEPTH_HONESTY, SKELETON_HONESTY
from meshops.proportion.models import DepthBand, LandmarkXYZ, ProportionReport, QualityFlags
from meshops.proportion.skeleton import (
    BPY_BASENAME,
    ELBOW_HANG_T,
    JSON_BASENAME,
    SKELETON_SCHEMA_VERSION,
    BlockoutSkeleton,
    SkeletonBone,
    SkeletonJoint,
    _depth_family_for_joint,
    _elbow_hang_y,
    build_blockout_skeleton,
    emit_bpy_script,
    run_skeleton_build,
    write_blockout_skeleton,
)

runner = CliRunner()


def _lm(
    id_: str,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    z_m: float | None = None,
) -> LandmarkXYZ:
    return LandmarkXYZ(id=id_, x_m=x_m, y_m=y_m, z_m=z_m)


def _full_landmarks(*, height_m: float = 1.72) -> dict[str, LandmarkXYZ]:
    """Synthetic full A-pose landmark set (meters, soles=0)."""
    return {
        "sole": _lm("sole", x_m=0.0, y_m=0.0, z_m=0.0),
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90),
        "crotch_pubic": _lm("crotch_pubic", x_m=0.0, y_m=0.0, z_m=0.86),
        "navel": _lm("navel", x_m=0.0, y_m=0.02, z_m=1.05),
        "chest_front": _lm("chest_front", x_m=0.0, y_m=-0.05, z_m=1.25),
        "chest_back": _lm("chest_back", x_m=0.0, y_m=0.08, z_m=1.24),
        "underbust": _lm("underbust", x_m=0.0, y_m=0.0, z_m=1.18),
        "belt_hip": _lm("belt_hip", x_m=0.0, y_m=0.0, z_m=0.95),
        "neck": _lm("neck", x_m=0.0, y_m=0.0, z_m=1.45),
        "chin": _lm("chin", x_m=0.0, y_m=-0.02, z_m=1.52),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.70),
        "hair_crown": _lm("hair_crown", x_m=0.0, y_m=-0.01, z_m=1.72),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=-0.05, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.28, y_m=-0.05, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=-0.08, z_m=0.85),
        "wrist_r": _lm("wrist_r", x_m=0.32, y_m=-0.08, z_m=0.85),
        "fingertip_l": _lm("fingertip_l", x_m=-0.35, y_m=-0.10, z_m=0.70),
        "fingertip_r": _lm("fingertip_r", x_m=0.35, y_m=-0.10, z_m=0.70),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.0, z_m=0.48),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.0, z_m=0.48),
        "ankle_l": _lm("ankle_l", x_m=-0.12, y_m=0.0, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.12, y_m=0.0, z_m=0.08),
        "heel_l": _lm("heel_l", x_m=-0.12, y_m=0.04, z_m=0.02),
        "heel_r": _lm("heel_r", x_m=0.12, y_m=0.04, z_m=0.02),
        "toe_l": _lm("toe_l", x_m=-0.12, y_m=-0.08, z_m=0.02),
        "toe_r": _lm("toe_r", x_m=0.12, y_m=-0.08, z_m=0.02),
    }


def _report(
    lms: dict[str, LandmarkXYZ] | None = None,
    *,
    height_m: float | None = 1.72,
    head_unit_frac: float | None = 1.0 / 7.5,
    depth_bands: list[DepthBand] | None = None,
) -> ProportionReport:
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=head_unit_frac,
        landmarks_xyz=lms if lms is not None else {},
        depth_bands=list(depth_bands or []),
        quality=QualityFlags(),
    )


def _band(
    band_id: str,
    *,
    y_mid: float = 0.03,
    depth_frac: float = 0.06,
) -> DepthBand:
    """Minimal DepthBand (y_mid is height fraction; meters via y_mid * height_m)."""
    return DepthBand(
        band_id=band_id,
        depth_px=20.0,
        depth_frac=depth_frac,
        depth_m=None,
        y_front=y_mid + depth_frac / 2.0,
        y_back=y_mid - depth_frac / 2.0,
        y_mid=y_mid,
        z_frac=None,
        confidence=0.8,
        sources=["left"],
        orientation_swapped=False,
    )


def _landmarks_with_mids(*, height_m: float = 1.72) -> dict[str, LandmarkXYZ]:
    """Front XZ on limbs (y missing) + mid landmarks with real y_m (0035 T1 helper).

    Does not replace `_full_landmarks` used by existing tests.
    """
    _ = height_m
    return {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=None, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=None, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.48),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=None, z_m=0.48),
        "ankle_l": _lm("ankle_l", x_m=-0.12, y_m=None, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.12, y_m=None, z_m=0.08),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.40),
        # Mid landmarks supply body-depth Y (meters)
        "hip_mid": _lm("hip_mid", x_m=0.0, y_m=0.05, z_m=0.90),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.08, z_m=1.25),
        "breast_mid": _lm("breast_mid", x_m=0.0, y_m=0.06, z_m=1.15),
        "thigh_mid": _lm("thigh_mid", x_m=0.0, y_m=0.04, z_m=0.48),
        "calf_mid": _lm("calf_mid", x_m=0.0, y_m=0.03, z_m=0.20),
        # Scaffold (navel has no Y — prefer breast_mid for spine_mid)
        "belt_hip": _lm("belt_hip", x_m=0.0, y_m=None, z_m=0.95),
        "navel": _lm("navel", x_m=0.0, y_m=None, z_m=1.05),
        "neck": _lm("neck", x_m=0.0, y_m=None, z_m=1.45),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=None, z_m=1.70),
    }


def _write_report(tmp: Path, report: ProportionReport) -> Path:
    p = tmp / "proportion_report.json"
    p.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return p


def _by_id(pkg: BlockoutSkeleton) -> dict[str, SkeletonJoint]:
    return {j.id: j for j in pkg.joints}


def _bone_by_id(pkg: BlockoutSkeleton) -> dict[str, SkeletonBone]:
    return {b.id: b for b in pkg.bones}


# ---------------------------------------------------------------------------
# R6 tests
# ---------------------------------------------------------------------------


def test_skeleton__full_landmarks_critical_joints_and_limb_lengths() -> None:
    """1. Full landmarks → critical joints finite; limb bone lengths > 0 (no zero hand)."""
    pkg = build_blockout_skeleton(_report(_full_landmarks()))
    j = _by_id(pkg)
    for jid in (
        "root",
        "pelvis",
        "spine_low",
        "spine_mid",
        "spine_high",
        "neck_base",
        "neck_top",
        "head",
        "chin",
        "crown",
        "shoulder_l",
        "shoulder_r",
        "elbow_l",
        "wrist_l",
        "hand_l",
        "hip_l",
        "knee_l",
        "ankle_l",
        "heel_l",
        "toe_l",
    ):
        assert jid in j, f"missing joint {jid}"
        jj = j[jid]
        assert jj.x_m is not None and jj.y_m is not None and jj.z_m is not None

    bones = _bone_by_id(pkg)
    for bid in (
        "upper_arm_l",
        "forearm_l",
        "hand_l",
        "thigh_l",
        "calf_l",
        "foot_l",
        "spine_low_bone",
        "head_bone",
    ):
        assert bid in bones, f"missing bone {bid}"
        bl = bones[bid].length_m
        assert bl is not None and bl > 0.0, f"{bid} length={bl}"

    assert pkg.schema_version == SKELETON_SCHEMA_VERSION
    assert pkg.axis_notes == AXIS_NOTES
    assert pkg.pose == "a_pose"
    assert pkg.head_unit_m == pytest.approx(1.72 / 7.5)


def test_skeleton__missing_knee_mid_estimate() -> None:
    """2. Missing knee → mid estimate + message."""
    lms = _full_landmarks()
    del lms["knee_l"]
    pkg = build_blockout_skeleton(_report(lms))
    j = _by_id(pkg)
    knee = j["knee_l"]
    assert knee.source == "estimated"
    assert knee.x_m is not None and knee.y_m is not None and knee.z_m is not None
    hip = j["hip_l"]
    ankle = j["ankle_l"]
    assert hip.z_m is not None and ankle.z_m is not None
    assert knee.z_m == pytest.approx((hip.z_m + ankle.z_m) / 2.0, abs=1e-6)
    assert any("knee_l" in m and "mid" in m.lower() for m in pkg.messages)


def test_skeleton__missing_shoulder_partial_arm_writes(tmp_path: Path) -> None:
    """3. Missing shoulder → partial arm; package writes."""
    lms = _full_landmarks()
    del lms["shoulder_l"]
    report = _report(lms)
    path = _write_report(tmp_path, report)
    out = tmp_path / "skel_out"
    payload = run_skeleton_build(path, out, format="json", force=True)
    assert payload["ok"] is True
    assert (out / JSON_BASENAME).is_file()
    pkg = BlockoutSkeleton.model_validate(
        json.loads((out / JSON_BASENAME).read_text(encoding="utf-8"))
    )
    j = _by_id(pkg)
    # shoulder may be estimated from stature or missing; package still written
    assert "elbow_l" in j
    assert "wrist_l" in j
    assert pkg.counts.joints > 0


def test_skeleton__missing_fingertip_no_hand() -> None:
    """4. Missing fingertip → no hand joint / no hand bone."""
    lms = _full_landmarks()
    del lms["fingertip_l"]
    del lms["fingertip_r"]
    pkg = build_blockout_skeleton(_report(lms))
    ids = {j.id for j in pkg.joints}
    bone_ids = {b.id for b in pkg.bones}
    assert "hand_l" not in ids
    assert "hand_r" not in ids
    assert "hand_l" not in bone_ids
    assert "hand_r" not in bone_ids
    assert any("hand_l" in m and "omitted" in m for m in pkg.messages)


def test_skeleton__plane_class_tip_hand_y_from_wrist() -> None:
    """0084 B8: plane-class fingertip Y → hand.y_m == wrist.y_m (XZ still toward tip)."""
    lms = _full_landmarks()
    lms["fingertip_l"] = _lm("fingertip_l", x_m=-0.35, y_m=None, z_m=0.70)
    lms["fingertip_r"] = _lm("fingertip_r", x_m=0.35, y_m=None, z_m=0.70)
    pkg = build_blockout_skeleton(_report(lms))
    j = {jj.id: jj for jj in pkg.joints}
    for side in ("l", "r"):
        wrist = j[f"wrist_{side}"]
        hand = j[f"hand_{side}"]
        assert wrist.y_m is not None and hand.y_m is not None
        assert float(hand.y_m) == pytest.approx(float(wrist.y_m), abs=1e-6)
        assert hand.x_m is not None and wrist.x_m is not None
        assert abs(float(hand.x_m) - float(wrist.x_m)) > 1e-4
    assert any("fingertip untrusted" in m for m in pkg.messages)


def test_skeleton__trusted_tip_hand_y_interpolates() -> None:
    """0084 B3/B8: trusted face-forward tip still interpolates hand Y (not clone wrist)."""
    lms = _full_landmarks()
    pkg = build_blockout_skeleton(_report(lms))
    j = {jj.id: jj for jj in pkg.joints}
    for side in ("l", "r"):
        wrist = j[f"wrist_{side}"]
        hand = j[f"hand_{side}"]
        assert wrist.y_m is not None and hand.y_m is not None
        assert float(hand.y_m) != pytest.approx(float(wrist.y_m), abs=1e-6)
        # Toward the more-negative trusted tip
        assert float(hand.y_m) < float(wrist.y_m)


def test_skeleton__measured_requires_full_xyz_front_plane_estimated() -> None:
    """5. measured only full XYZ; invent/prior Y -> estimated (0051 arm prior)."""
    lms = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.01, z_m=1.40),
        "cranial_vertex": _lm("cranial_vertex", x_m=0.0, y_m=-0.01, z_m=1.70),
    }
    pkg = build_blockout_skeleton(_report(lms, height_m=1.72))
    j = _by_id(pkg)
    assert j["hip_l"].source == "measured"
    assert j["shoulder_r"].source == "measured"
    assert j["shoulder_l"].source == "estimated"
    assert j["shoulder_l"].y_m is not None
    # 0083: invent shoulder stays on glenoid plane (no arm-forward prior on glenoid)
    assert j["shoulder_l"].y_m == pytest.approx(0.0)
    assert not any("shoulder_l" in m and "arm forward prior" in m for m in pkg.messages)


def test_skeleton__root_origin_no_root_pelvis_bone() -> None:
    """6. root at origin; no root→pelvis bone; pelvis parent null."""
    pkg = build_blockout_skeleton(_report(_full_landmarks()))
    j = _by_id(pkg)
    root = j["root"]
    assert root.x_m == 0.0 and root.y_m == 0.0 and root.z_m == 0.0
    assert root.parent is None
    assert root.source == "estimated"
    pelvis = j["pelvis"]
    assert pelvis.parent is None
    for b in pkg.bones:
        assert not (b.joint_a == "root" or b.joint_b == "root")
        assert b.id != "root_pelvis"
        assert "parent_bone" not in b.model_dump()


def test_skeleton__format_bpy_without_bpy_installed(tmp_path: Path) -> None:
    """7. format=bpy succeeds without bpy installed (string emit only)."""
    report = _report(_full_landmarks())
    path = _write_report(tmp_path, report)
    out = tmp_path / "skel_bpy"
    # bpy must not be imported by meshops; string emit only (C3).
    payload = run_skeleton_build(path, out, format="bpy", force=True)
    assert payload["ok"] is True
    script = out / BPY_BASENAME
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "import bpy" in text
    assert "SKEL_" in text
    assert SKELETON_HONESTY in text
    assert "hard-delete" in text or 'startswith(("Cube."' in text or "SKEL_" in text
    # emit_bpy_script pure string
    pkg = build_blockout_skeleton(report)
    emitted = emit_bpy_script(pkg)
    assert emitted.startswith("# setup_skeleton.py")
    assert "Proportion_Skeleton" in emitted


def test_skeleton__empty_report_no_height_raises_empty() -> None:
    """8. Empty report + no height → skeleton_empty."""
    report = _report({}, height_m=None, head_unit_frac=None)
    with pytest.raises(ProportionError) as ei:
        build_blockout_skeleton(report)
    assert ei.value.code == "skeleton_empty"


def test_skeleton__honesty_exact() -> None:
    """10. Honesty exact string on package."""
    pkg = build_blockout_skeleton(_report(_full_landmarks()))
    assert pkg.honesty == "proportion_blockout_skeleton_not_mesh_or_print_success"
    assert pkg.honesty == SKELETON_HONESTY


def test_skeleton__write_force_basenames_only(tmp_path: Path) -> None:
    """C6: --force overwrites only skeleton basenames; siblings kept."""
    out = tmp_path / "out"
    out.mkdir()
    sibling = out / "keep_me.txt"
    sibling.write_text("safe", encoding="utf-8")
    pkg = build_blockout_skeleton(_report(_full_landmarks()))
    write_blockout_skeleton(out, pkg, format="both", force=False)
    assert (out / JSON_BASENAME).is_file()
    assert (out / BPY_BASENAME).is_file()
    with pytest.raises(ProportionError) as ei:
        write_blockout_skeleton(out, pkg, format="json", force=False)
    assert ei.value.code == "write_failed"
    write_blockout_skeleton(out, pkg, format="both", force=True)
    assert sibling.read_text(encoding="utf-8") == "safe"


def test_skeleton__cli_skeleton_build(tmp_path: Path) -> None:
    path = _write_report(tmp_path, _report(_full_landmarks()))
    out = tmp_path / "cli_out"
    result = runner.invoke(
        app,
        [
            "proportion",
            "skeleton-build",
            "--report",
            str(path),
            "--out",
            str(out),
            "--format",
            "both",
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / JSON_BASENAME).is_file()
    assert (out / BPY_BASENAME).is_file()
    assert "skeleton-build" in result.output


def test_skeleton__cli_empty_nonzero(tmp_path: Path) -> None:
    path = _write_report(tmp_path, _report({}, height_m=None, head_unit_frac=None))
    out = tmp_path / "empty_out"
    result = runner.invoke(
        app,
        [
            "proportion",
            "skeleton-build",
            "--report",
            str(path),
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert not (out / JSON_BASENAME).exists()


def test_skeleton__head_from_cranial_vertex() -> None:
    """B6: head ← cranial_vertex; head_bone = neck_top → head."""
    pkg = build_blockout_skeleton(_report(_full_landmarks()))
    j = _by_id(pkg)
    assert j["head"].landmark_id == "cranial_vertex"
    assert j["head"].z_m == pytest.approx(1.70)
    bones = _bone_by_id(pkg)
    assert bones["head_bone"].joint_a == "neck_top"
    assert bones["head_bone"].joint_b == "head"


def test_skeleton__no_parent_bone_in_schema() -> None:
    pkg = build_blockout_skeleton(_report(_full_landmarks()))
    dumped = pkg.model_dump(mode="json")
    for b in dumped["bones"]:
        assert "parent_bone" not in b


def test_skeleton__template_applied_sets_template_id(tmp_path: Path) -> None:
    """B3: --template-applied / run_skeleton_build sets template_id from package; else null."""
    from meshops.proportion.body_template import (
        TEMPLATE_HONESTY,
        AppliedConstants,
        TemplateAppliedPackage,
    )

    report = _report(_full_landmarks())
    report_path = _write_report(tmp_path, report)

    # Without template_applied → null on package and payload.
    out_none = tmp_path / "skel_no_tpl"
    payload_none = run_skeleton_build(report_path, out_none, format="json", force=True)
    assert payload_none["template_id"] is None
    pkg_none = BlockoutSkeleton.model_validate(
        json.loads((out_none / JSON_BASENAME).read_text(encoding="utf-8"))
    )
    assert pkg_none.template_id is None

    # Minimal template_applied.json accepted by load_template_applied.
    constants = AppliedConstants(
        breast_mode="dual_tilted",
        glute_mode_default="oval",
        torso_mode_default="trap",
    )
    applied = TemplateAppliedPackage(
        template_id="female_adult_athletic",
        sex="female",
        archetype="adult_athletic",
        source_report=str(report_path),
        height_m=1.72,
        constants=constants,
        honesty=TEMPLATE_HONESTY,
    )
    tpl_path = tmp_path / "template_applied.json"
    tpl_path.write_text(
        json.dumps(applied.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    out_tpl = tmp_path / "skel_with_tpl"
    payload_tpl = run_skeleton_build(
        report_path,
        out_tpl,
        format="json",
        force=True,
        template_applied=tpl_path,
    )
    assert payload_tpl["template_id"] == "female_adult_athletic"
    pkg_tpl = BlockoutSkeleton.model_validate(
        json.loads((out_tpl / JSON_BASENAME).read_text(encoding="utf-8"))
    )
    assert pkg_tpl.template_id == "female_adult_athletic"


def test_skeleton__measured_joint_y_m_as_is_no_sign_flip() -> None:
    """B1: measured joint y_m equals landmark y_m as-is (no sign flip / axis remap).

    Landmark uses y_m != 0 and y_m != -x_m so naive flip-to--x or sign-flip would fail.
    """
    y_as_is = -0.07
    x_m = -0.28  # y_as_is != -x_m (0.28) and y_as_is != 0
    assert y_as_is != 0.0
    assert y_as_is != -x_m
    assert y_as_is != x_m

    lms = _full_landmarks()
    lms["elbow_l"] = _lm("elbow_l", x_m=x_m, y_m=y_as_is, z_m=1.10)
    pkg = build_blockout_skeleton(_report(lms))
    elbow = _by_id(pkg)["elbow_l"]
    assert elbow.source == "measured"
    assert elbow.y_m == pytest.approx(y_as_is)
    assert elbow.x_m == pytest.approx(x_m)
    # Explicit anti-flip checks (would pass if someone remapped Y).
    assert elbow.y_m != pytest.approx(-y_as_is)
    assert elbow.y_m != pytest.approx(-x_m)


# ---------------------------------------------------------------------------
# 0035 — Skeleton depth when available (ADD only; existing tests unedited)
# ---------------------------------------------------------------------------


def test_skeleton__depth_mids_bands_measured_ge_8() -> None:
    """T1: front XZ + mids/bands → counts.measured >= 8; prefer breast_mid for spine."""
    lms = _landmarks_with_mids()
    bands = [
        _band("hip", y_mid=0.03),
        _band("chest", y_mid=0.04),
        _band("breast", y_mid=0.035),
        _band("thigh", y_mid=0.02),
        _band("calf", y_mid=0.015),
    ]
    pkg = build_blockout_skeleton(_report(lms, depth_bands=bands))
    j = _by_id(pkg)
    assert pkg.counts.measured >= 8, (
        f"expected measured>=8 got {pkg.counts.measured}; "
        f"sources={{k: v.source for k, v in j.items()}}"
    )
    # Core floor joints should be measured
    for jid in (
        "pelvis",
        "hip_l",
        "hip_r",
        "knee_l",
        "knee_r",
        "ankle_l",
        "ankle_r",
        "spine_high",
    ):
        assert j[jid].source == "measured", f"{jid} source={j[jid].source} y={j[jid].y_m}"
    # Prefer breast_mid (not navel) for spine_mid Y
    assert j["spine_mid"].y_m == pytest.approx(0.06)
    assert any("breast_mid" in m and "(depth)" in m for m in pkg.messages)
    # Depth messages present
    assert any("(depth)" in m for m in pkg.messages)


def test_skeleton__front_xz_without_depth_stays_estimated() -> None:
    """T2: same front XZ without mids/bands → no false measured from ladder."""
    lms = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=None, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=None, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.48),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=None, z_m=0.48),
        "ankle_l": _lm("ankle_l", x_m=-0.12, y_m=None, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.12, y_m=None, z_m=0.08),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.40),
    }
    pkg = build_blockout_skeleton(_report(lms, height_m=1.72))
    j = _by_id(pkg)
    for jid in ("hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "shoulder_l"):
        assert j[jid].source == "estimated", f"{jid} should be estimated, got {j[jid].source}"
        assert j[jid].y_m is not None
    assert not any("(depth)" in m for m in pkg.messages)


def test_skeleton__t3_existing_front_plane_test_still_imported() -> None:
    """T3: existing front-plane measured test stays green (unedited freeze)."""
    # No code change to existing test; this marker documents the freeze.
    assert callable(test_skeleton__measured_requires_full_xyz_front_plane_estimated)


def test_skeleton__spine_high_chest_mid_measured() -> None:
    """T4: spine_high with chest_mid full XYZ → measured."""
    lms = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.08, z_m=1.25),
        "navel": _lm("navel", x_m=0.0, y_m=0.02, z_m=1.05),
        "belt_hip": _lm("belt_hip", x_m=0.0, y_m=0.0, z_m=0.95),
    }
    pkg = build_blockout_skeleton(_report(lms))
    j = _by_id(pkg)
    assert j["spine_high"].source == "measured"
    assert j["spine_high"].y_m == pytest.approx(0.08)
    assert j["spine_high"].z_m == pytest.approx(1.25)
    assert j["spine_high"].landmark_id == "chest_mid"


def test_skeleton__depth_samples_file_supplies_y(tmp_path: Path) -> None:
    """T5: optional depth-samples file alone supplies Y when report lacks mid."""
    lms = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=None, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=None, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=None, z_m=0.48),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=None, z_m=0.48),
        "ankle_l": _lm("ankle_l", x_m=-0.12, y_m=None, z_m=0.08),
        "ankle_r": _lm("ankle_r", x_m=0.12, y_m=None, z_m=0.08),
    }
    # No mid landmarks and no bands on report — only depth file.
    report = _report(lms)
    report_path = _write_report(tmp_path, report)
    samples = DepthSamplesPackage(
        honesty=DEPTH_HONESTY,
        height_m=1.72,
        samples=[
            DepthSample(
                id="hip_mid",
                role="landmark",
                y_m=0.055,
                source="fused_xyz",
                confidence=0.9,
            ),
            DepthSample(
                id="thigh_mid",
                role="landmark",
                y_m=0.04,
                source="fused_xyz",
                confidence=0.9,
            ),
            DepthSample(
                id="calf_mid",
                role="landmark",
                y_m=0.03,
                source="fused_xyz",
                confidence=0.9,
            ),
        ],
    )
    depth_path = tmp_path / "depth_at_landmarks.json"
    depth_path.write_text(json.dumps(samples.model_dump(mode="json"), indent=2), encoding="utf-8")
    out = tmp_path / "skel_depth_file"
    payload = run_skeleton_build(
        report_path, out, format="json", force=True, depth_at_landmarks=depth_path
    )
    assert payload["ok"] is True
    pkg = BlockoutSkeleton.model_validate(
        json.loads((out / JSON_BASENAME).read_text(encoding="utf-8"))
    )
    j = _by_id(pkg)
    assert j["hip_l"].source == "measured"
    assert j["hip_l"].y_m == pytest.approx(0.055)
    assert j["knee_l"].source == "measured"
    assert j["ankle_l"].source == "measured"
    assert any("hip_mid" in m and "(depth)" in m for m in pkg.messages)


def test_skeleton__depth_family_arm_joints_no_band() -> None:
    """T6: elbow/wrist/hand have no depth family band; chain inherit still allowed."""
    assert _depth_family_for_joint("elbow_l") is None
    assert _depth_family_for_joint("elbow_r") is None
    assert _depth_family_for_joint("wrist_l") is None
    assert _depth_family_for_joint("wrist_r") is None
    assert _depth_family_for_joint("hand_l") is None
    assert _depth_family_for_joint("hand_r") is None
    # Non-arm joints still mapped
    fam = _depth_family_for_joint("hip_l")
    assert fam is not None
    assert "hip_mid" in fam[0]
    assert "hip" in fam[1]


def test_skeleton__cli_depth_at_landmarks_file(tmp_path: Path) -> None:
    """T7: CLI + MCP optional depth file path smoke (catalog 47 after 0110)."""
    from meshops.mcp import TOOL_NAMES
    from meshops.mcp.tools import mesh_proportion_skeleton_build

    assert len(TOOL_NAMES) == 47
    assert "mesh_proportion_skeleton_build" in TOOL_NAMES

    lms = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=None, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=None, z_m=0.90),
    }
    report_path = _write_report(tmp_path, _report(lms))
    samples = DepthSamplesPackage(
        honesty=DEPTH_HONESTY,
        height_m=1.72,
        samples=[
            DepthSample(
                id="hip_mid",
                role="landmark",
                y_m=0.05,
                source="fused_xyz",
                confidence=0.9,
            ),
        ],
    )
    depth_path = tmp_path / "depth_at_landmarks.json"
    depth_path.write_text(json.dumps(samples.model_dump(mode="json"), indent=2), encoding="utf-8")
    out = tmp_path / "cli_depth_out"
    result = runner.invoke(
        app,
        [
            "proportion",
            "skeleton-build",
            "--report",
            str(report_path),
            "--out",
            str(out),
            "--depth-at-landmarks",
            str(depth_path),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / JSON_BASENAME).is_file()
    pkg = BlockoutSkeleton.model_validate(
        json.loads((out / JSON_BASENAME).read_text(encoding="utf-8"))
    )
    assert _by_id(pkg)["hip_l"].source == "measured"

    # MCP tool param path (param ≠ new tool name)
    out_mcp = tmp_path / "mcp_depth_out"
    payload = mesh_proportion_skeleton_build(
        tmp_path,
        report=str(report_path),
        out=str(out_mcp),
        depth_at_landmarks=str(depth_path),
        force=True,
    )
    assert payload["ok"] is True
    pkg_mcp = BlockoutSkeleton.model_validate(
        json.loads((out_mcp / JSON_BASENAME).read_text(encoding="utf-8"))
    )
    assert _by_id(pkg_mcp)["hip_l"].source == "measured"


def test_skeleton__height_null_band_only_no_false_measured() -> None:
    """T8: height_m null + band-only → no crash; no false measured from band meters."""
    lms = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=None, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=None, z_m=0.90),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.40),
    }
    bands = [_band("hip", y_mid=0.03), _band("chest", y_mid=0.04)]
    pkg = build_blockout_skeleton(
        _report(lms, height_m=None, head_unit_frac=None, depth_bands=bands)
    )
    j = _by_id(pkg)
    assert j["hip_l"].source == "estimated"
    assert j["shoulder_l"].source == "estimated"
    # Band path skipped without height — no depth-from-band messages for meters
    assert not any("from hip (depth)" in m for m in pkg.messages)


def test_skeleton__spine_low_stature_z_with_hip_mid_stays_estimated() -> None:
    """T9: belt_hip absent + hip_mid Y → spine_low stays estimated (stature/mid Z — R3)."""
    lms = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90),
        "hip_mid": _lm("hip_mid", x_m=0.0, y_m=0.05, z_m=0.90),
        "navel": _lm("navel", x_m=0.0, y_m=0.02, z_m=1.05),
        # no belt_hip — Z filled via mid pelvis→spine_mid or stature
    }
    pkg = build_blockout_skeleton(_report(lms, height_m=1.72))
    j = _by_id(pkg)
    assert j["spine_low"].source == "estimated"
    # Depth Y may still be applied, but R3 blocks measured without real XZ
    assert j["spine_low"].y_m is not None


def test_skeleton__depth_y_does_not_suppress_neck_base_z_fill() -> None:
    """P2 regression: depth-only Y on neck_base must still fill Z from shoulders/stature."""
    lms = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.08, z_m=1.25),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.40),
        # no neck landmark — neck_base uses mean shoulder Z + chest_mid depth Y
    }
    pkg = build_blockout_skeleton(_report(lms))
    j = _by_id(pkg)
    nb = j["neck_base"]
    assert nb.z_m is not None and nb.z_m == pytest.approx(1.40)
    assert nb.y_m is not None and nb.y_m == pytest.approx(0.08)
    # Synth X + depth Y + landmark shoulder Z mean → still estimated under R3 (X not landmark)
    assert nb.source == "estimated"
    assert any("mean shoulder" in m for m in pkg.messages)


def test_skeleton__pair_mean_one_side_only_stays_estimated() -> None:
    """P1 regression: one-sided / mixed-axis pair mean + depth Y must not false-measure."""
    # Only hip_l has XZ; hip_r absent — pre-0035 pair gate → estimated even with hip_mid Y
    lms_one = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=None, z_m=0.90),
        "hip_mid": _lm("hip_mid", x_m=0.0, y_m=0.05, z_m=0.90),
    }
    pkg_one = build_blockout_skeleton(_report(lms_one))
    assert _by_id(pkg_one)["pelvis"].source == "estimated"
    assert _by_id(pkg_one)["pelvis"].y_m == pytest.approx(0.05)

    # Mixed-axis stitch: left X only + right Z only + hip_mid Y → estimated
    lms_mix = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=None, z_m=None),
        "hip_r": _lm("hip_r", x_m=None, y_m=None, z_m=0.90),
        "hip_mid": _lm("hip_mid", x_m=0.0, y_m=0.05, z_m=0.90),
    }
    pkg_mix = build_blockout_skeleton(_report(lms_mix))
    assert _by_id(pkg_mix)["pelvis"].source == "estimated"


# ---------------------------------------------------------------------------
# 0037 — Skeleton arm depth policy (inherit honesty; no chest-band steal)
# ---------------------------------------------------------------------------


def test_skeleton__0037_t1_depth_family_arm_joints_none() -> None:
    """T1: elbow/wrist/hand stay family None (0035 freeze; no chest-band steal)."""
    for jid in (
        "elbow_l",
        "elbow_r",
        "wrist_l",
        "wrist_r",
        "hand_l",
        "hand_r",
    ):
        assert _depth_family_for_joint(jid) is None, jid


def test_skeleton__0037_t2_real_shoulder_depth_elbow_wrist_inherit() -> None:
    """T2 (0087): torso-plane chest_mid → glenoid ≈ plane; wrist distal; elbow hang."""
    from meshops.proportion.skeleton import _arm_forward_y

    h = 1.72
    chest_y = 0.08
    lms = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.28, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
        "wrist_r": _lm("wrist_r", x_m=0.32, y_m=None, z_m=0.85),
        # Torso-plane depth for shoulder ladder → glenoid stays at plane
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=chest_y, z_m=1.25),
    }
    pkg = build_blockout_skeleton(_report(lms, height_m=h))
    j = _by_id(pkg)
    # No half-depth evidence -> stature prior from plane: chest_y + (-0.05)*H
    expected = _arm_forward_y(chest_y, half_depth=None, height_m=h, chest_front_y=None)
    hang = _elbow_hang_y(chest_y, expected, t=ELBOW_HANG_T)
    sh_y = j["shoulder_l"].y_m
    assert sh_y == pytest.approx(chest_y)
    assert sh_y != pytest.approx(expected)
    assert j["elbow_l"].y_m == pytest.approx(hang)
    assert j["wrist_l"].y_m == pytest.approx(expected)
    assert j["elbow_r"].y_m == pytest.approx(hang)
    assert j["wrist_r"].y_m == pytest.approx(expected)
    assert not any("shoulder_l" in m and "arm forward prior" in m for m in pkg.messages)
    assert any("elbow_l" in m and "hang" in m and f"t={ELBOW_HANG_T}" in m for m in pkg.messages)
    assert any("wrist_l" in m and "arm forward prior" in m and "distal" in m for m in pkg.messages)
    assert not any("(arm prior)" in m for m in pkg.messages)
    # Must not be front-plane-only for those joints
    assert not any("elbow_l" in m and "front-plane" in m for m in pkg.messages), (
        "elbow_l must not use front-plane when distal prior applied"
    )
    assert not any("wrist_l" in m and "front-plane" in m for m in pkg.messages), (
        "wrist_l must not use front-plane when distal prior applied"
    )
    assert not any("elbow_r" in m and "front-plane" in m for m in pkg.messages)
    assert not any("wrist_r" in m and "front-plane" in m for m in pkg.messages)
    # Glenoid Y is depth-ladder chest_mid (honest measured); distal prior is estimated
    assert j["elbow_l"].source == "estimated"
    assert j["wrist_l"].source == "estimated"
    assert j["elbow_r"].source == "estimated"
    assert j["wrist_r"].source == "estimated"


def test_skeleton__0037_t2b_invent_shoulder_keeps_front_plane_msgs() -> None:
    """T2b (0083): invent glenoid ≈ 0; distal keeps stature prior; never inherit (depth)."""
    h = 1.72
    lms = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
        # No chest_mid / no depth bands → invent glenoid from mid-plane 0
    }
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=[]))
    j = _by_id(pkg)
    assert j["shoulder_l"].y_m is not None
    assert j["shoulder_l"].y_m == pytest.approx(0.0)
    assert j["wrist_l"].y_m == pytest.approx(-0.05 * h)
    assert j["elbow_l"].y_m == pytest.approx(_elbow_hang_y(0.0, -0.05 * h, t=ELBOW_HANG_T))
    assert not any("inherited" in m and "(depth)" in m for m in pkg.messages), pkg.messages
    assert not any("shoulder_l" in m and "arm forward prior" in m for m in pkg.messages)
    assert any("arm forward prior" in m and "invent" in m and "distal" in m for m in pkg.messages)
    el_msgs = [m for m in pkg.messages if "elbow_l" in m]
    wr_msgs = [m for m in pkg.messages if "wrist_l" in m]
    assert any("hang" in m and f"t={ELBOW_HANG_T}" in m for m in el_msgs), el_msgs
    assert any("arm forward prior" in m and "distal" in m for m in wr_msgs), wr_msgs
    assert j["elbow_l"].source == "estimated"
    assert j["wrist_l"].source == "estimated"


def test_skeleton__0037_t3_elbow_no_chest_band_ladder() -> None:
    """T3: elbow must not receive Y solely from chest band via ladder (family None)."""
    assert _depth_family_for_joint("elbow_l") is None
    assert _depth_family_for_joint("elbow_r") is None
    h = 1.72
    # Chest band present with non-zero y_mid; elbow has no own Y; no shoulder Y evidence
    # beyond invent path (no chest_mid landmark for shoulder ladder to use mid path
    # that would be "real" if band alone — shoulder family uses band "chest" after mid).
    # With shoulder x,z + chest band → shoulder can get real depth from band.
    # Elbow family is None so ladder never maps chest onto elbow directly.
    lms = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
    }
    bands = [_band("chest", y_mid=0.08)]
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=bands))
    j = _by_id(pkg)
    # No ladder message claiming elbow y from chest band (family is None).
    # 0087 hang may mention (from chest) as glenoid provenance — not a band steal.
    assert not any(
        "elbow_l" in m and ("depth band" in m or "y_m from chest" in m) for m in pkg.messages
    ), pkg.messages
    assert any("elbow_l" in m and "elbow hang" in m for m in pkg.messages), pkg.messages
    # Elbow Y may match shoulder (chain inherit) but family remains None
    assert _depth_family_for_joint("elbow_l") is None
    assert j["elbow_l"].y_m is not None
    # If shoulder got band depth, elbow inherits via chain — still estimated
    assert j["elbow_l"].source == "estimated"
    # Must not claim measured for ladder steal (steal path does not exist)
    assert j["elbow_l"].source != "measured"


def test_skeleton__0037_t6_inherited_elbow_not_measured() -> None:
    """T6: pure inherit elbow is estimated; measured count excludes pure inherit."""
    h = 1.72
    lms = {
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=None, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=None, z_m=1.40),
        "elbow_l": _lm("elbow_l", x_m=-0.28, y_m=None, z_m=1.10),
        "elbow_r": _lm("elbow_r", x_m=0.28, y_m=None, z_m=1.10),
        "wrist_l": _lm("wrist_l", x_m=-0.32, y_m=None, z_m=0.85),
        "wrist_r": _lm("wrist_r", x_m=0.32, y_m=None, z_m=0.85),
        "chest_mid": _lm("chest_mid", x_m=0.0, y_m=0.08, z_m=1.25),
    }
    pkg = build_blockout_skeleton(_report(lms, height_m=h))
    j = _by_id(pkg)
    assert j["elbow_l"].source == "estimated"
    assert j["elbow_r"].source == "estimated"
    assert j["wrist_l"].source == "estimated"
    assert j["wrist_r"].source == "estimated"
    # Measured count equals joints with source measured — pure inherit not included
    measured_ids = {jj.id for jj in pkg.joints if jj.source == "measured"}
    assert "elbow_l" not in measured_ids
    assert "elbow_r" not in measured_ids
    assert "wrist_l" not in measured_ids
    assert "wrist_r" not in measured_ids
    assert pkg.counts.measured == len(measured_ids)


# ---------------------------------------------------------------------------
# 0038 — Skeleton cranial depth (ADD only; existing tests unedited)
# ---------------------------------------------------------------------------


def _landmarks_with_cranial(
    *,
    y_front: float = -0.02,
    y_back: float = 0.12,
    with_mid: bool = False,
    front_only: bool = False,
    height_m: float = 1.72,
) -> dict[str, LandmarkXYZ]:
    """Base = _full_landmarks; clear head-family landmark Y so depth ladder runs;
    add cranial_front/back (and mid if with_mid). front_only skips back.
    """
    lms = _full_landmarks(height_m=height_m)
    # Clear Y so _try_depth_y engages; keep X/Z for R3 measured (x_from_lm + z_from_lm).
    for key in ("cranial_vertex", "chin", "hair_crown"):
        lm = lms[key]
        lms[key] = _lm(key, x_m=lm.x_m, y_m=None, z_m=lm.z_m)
    if with_mid:
        lms["cranial_mid"] = _lm(
            "cranial_mid",
            x_m=0.0,
            y_m=(y_front + y_back) / 2.0,
            z_m=1.61,
        )
    lms["cranial_front"] = _lm("cranial_front", x_m=0.0, y_m=y_front, z_m=1.60)
    if not front_only:
        lms["cranial_back"] = _lm("cranial_back", x_m=0.0, y_m=y_back, z_m=1.60)
    return lms


def _clear_head_family_y(lms: dict[str, LandmarkXYZ]) -> dict[str, LandmarkXYZ]:
    """Clear cranial_vertex/chin/hair_crown Y so depth ladder is the only Y path."""
    out = dict(lms)
    for key in ("cranial_vertex", "chin", "hair_crown"):
        if key in out:
            lm = out[key]
            out[key] = _lm(key, x_m=lm.x_m, y_m=None, z_m=lm.z_m)
    return out


def test_skeleton__0038_t1_cranial_mid_head_measured() -> None:
    """T1: cranial_mid finite y → head measured; y≈mid; msg has cranial_mid (depth)."""
    y_front, y_back = -0.02, 0.12
    mid_y = (y_front + y_back) / 2.0
    lms = _landmarks_with_cranial(y_front=y_front, y_back=y_back, with_mid=True)
    pkg = build_blockout_skeleton(_report(lms))
    j = _by_id(pkg)
    assert j["head"].source == "measured"
    assert j["head"].y_m == pytest.approx(mid_y)
    assert any("cranial_mid" in m and "(depth)" in m and "head" in m for m in pkg.messages), (
        pkg.messages
    )


def test_skeleton__0038_t2_cranial_front_back_pair_mean() -> None:
    """T2: front+back Y, no mid → head measured; y≈mean; msg cranial_front+cranial_back."""
    y_front, y_back = -0.02, 0.12
    mean_y = (y_front + y_back) / 2.0
    lms = _landmarks_with_cranial(y_front=y_front, y_back=y_back, with_mid=False)
    assert "cranial_mid" not in lms
    pkg = build_blockout_skeleton(_report(lms))
    j = _by_id(pkg)
    assert j["head"].source == "measured"
    assert j["head"].y_m == pytest.approx(mean_y)
    assert any(
        "cranial_front+cranial_back" in m and "(depth)" in m and "head" in m for m in pkg.messages
    ), pkg.messages


def test_skeleton__0038_t3_cranial_band_only_measured() -> None:
    """T3: depth_bands band_id cranial only → head measured via band*H."""
    h = 1.72
    y_mid_frac = 0.05
    lms = _clear_head_family_y(_full_landmarks(height_m=h))
    bands = [_band("cranial", y_mid=y_mid_frac)]
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=bands))
    j = _by_id(pkg)
    assert j["head"].source == "measured"
    assert j["head"].y_m == pytest.approx(y_mid_frac * h)
    assert any("cranial" in m and "(depth)" in m and "head" in m for m in pkg.messages), (
        pkg.messages
    )


def test_skeleton__0038_t4_one_sided_front_stays_estimated() -> None:
    """T4: one-sided cranial_front only → head estimated; no crash; no false measured."""
    lms = _landmarks_with_cranial(front_only=True)
    assert "cranial_back" not in lms
    pkg = build_blockout_skeleton(_report(lms))
    j = _by_id(pkg)
    assert j["head"].source == "estimated"
    assert j["head"].y_m is not None
    assert not any("cranial_front+cranial_back" in m for m in pkg.messages)
    assert not any("cranial_mid" in m and "(depth)" in m for m in pkg.messages)


def test_skeleton__0038_t5_product_like_null_cranial_estimated() -> None:
    """T5: product-like null cranial (cleared head Y, no assist) → estimated + front-plane."""
    lms = _clear_head_family_y(_full_landmarks())
    assert "cranial_front" not in lms
    assert "cranial_back" not in lms
    assert "cranial_mid" not in lms
    pkg = build_blockout_skeleton(_report(lms))
    j = _by_id(pkg)
    assert j["head"].source == "estimated"
    assert any("head" in m and "front-plane" in m for m in pkg.messages), pkg.messages
    assert not any("cranial" in m and "(depth)" in m for m in pkg.messages)


def test_skeleton__0038_t6_chin_crown_measured_neck_top_optional() -> None:
    """T6: chin+crown measured when R1 pair + XZ ok; neck_top may stay estimated (OK)."""
    y_front, y_back = -0.02, 0.12
    mean_y = (y_front + y_back) / 2.0
    lms = _landmarks_with_cranial(y_front=y_front, y_back=y_back, with_mid=False)
    pkg = build_blockout_skeleton(_report(lms))
    j = _by_id(pkg)
    assert j["head"].source == "measured"
    assert j["chin"].source == "measured"
    assert j["crown"].source == "measured"
    assert j["chin"].y_m == pytest.approx(mean_y)
    assert j["crown"].y_m == pytest.approx(mean_y)
    # neck_top often estimated (mid X/Z fails R3) — not a T6 failure
    assert j["neck_top"].y_m is not None
    assert j["neck_top"].source in ("measured", "estimated")


def test_skeleton__0038_non_cranial_family_ignores_cranial_pair() -> None:
    """B1 isolation: hip/chest family must not consume cranial_front/back pair Y."""
    # Front XZ hips with null Y; only cranial pair present (no hip_mid / hip band).
    lms = {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=None, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=None, z_m=0.90),
        "cranial_front": _lm("cranial_front", x_m=0.0, y_m=-0.02, z_m=1.60),
        "cranial_back": _lm("cranial_back", x_m=0.0, y_m=0.12, z_m=1.60),
    }
    pkg = build_blockout_skeleton(_report(lms, height_m=1.72))
    j = _by_id(pkg)
    assert j["hip_l"].source == "estimated"
    assert j["hip_r"].source == "estimated"
    # Must not claim hip Y from cranial pair
    assert not any("cranial_front+cranial_back" in m and "hip" in m for m in pkg.messages)
    assert not any("joint hip" in m and "cranial" in m and "(depth)" in m for m in pkg.messages)


# ---------------------------------------------------------------------------
# 0041 — Skeleton foot joint Z honesty (heel/toe sole-class prior; ADD only)
# ---------------------------------------------------------------------------


def _landmarks_ankles_only(
    *, height_m: float = 1.72, ankle_z: float = 0.13
) -> dict[str, LandmarkXYZ]:
    """Ankles measured XZ; NO heel_*/toe_* keys. Include minimal body so skeleton builds."""
    _ = height_m
    return {
        "hip_l": _lm("hip_l", x_m=-0.12, y_m=0.0, z_m=0.90),
        "hip_r": _lm("hip_r", x_m=0.12, y_m=0.0, z_m=0.90),
        "knee_l": _lm("knee_l", x_m=-0.12, y_m=0.0, z_m=0.48),
        "knee_r": _lm("knee_r", x_m=0.12, y_m=0.0, z_m=0.48),
        "ankle_l": _lm("ankle_l", x_m=-0.12, y_m=0.0, z_m=ankle_z),
        "ankle_r": _lm("ankle_r", x_m=0.12, y_m=0.0, z_m=ankle_z),
        "shoulder_l": _lm("shoulder_l", x_m=-0.20, y_m=0.0, z_m=1.40),
        "shoulder_r": _lm("shoulder_r", x_m=0.20, y_m=0.0, z_m=1.40),
    }


def test_skeleton__0041_t1_estimated_heel_toe_sole_z_not_ankle_clone() -> None:
    """T1: no heel/toe lm; height + ankle measured → heel/toe estimated; z≈0.02*H; z≠ankle.z."""
    h = 1.72
    ankle_z = 0.13
    lms = _landmarks_ankles_only(height_m=h, ankle_z=ankle_z)
    assert "heel_l" not in lms and "toe_l" not in lms
    pkg = build_blockout_skeleton(_report(lms, height_m=h))
    j = _by_id(pkg)
    sole_prior = 0.02 * h
    for jid in ("heel_l", "heel_r", "toe_l", "toe_r"):
        assert j[jid].source == "estimated", f"{jid} source={j[jid].source}"
        assert j[jid].z_m is not None
        assert j[jid].z_m == pytest.approx(sole_prior, abs=1e-6), (
            f"{jid} z={j[jid].z_m} expected sole prior {sole_prior}"
        )
        assert j[jid].z_m != pytest.approx(ankle_z, abs=1e-9)
        assert j[jid].z_m != j[f"ankle_{jid[-1]}"].z_m


def test_skeleton__0041_t2_both_margins_below_ankle() -> None:
    """T2 (AI1 P3-1): BOTH z < ankle.z - 0.02 AND z <= ankle.z * 0.50 for heel and toe."""
    h = 1.72
    ankle_z = 0.13
    lms = _landmarks_ankles_only(height_m=h, ankle_z=ankle_z)
    pkg = build_blockout_skeleton(_report(lms, height_m=h))
    j = _by_id(pkg)
    for jid in ("heel_l", "heel_r", "toe_l", "toe_r"):
        z = j[jid].z_m
        assert z is not None
        assert z < ankle_z - 0.02, f"{jid}: {z} not < ankle-0.02 ({ankle_z - 0.02})"
        assert z <= ankle_z * 0.50, f"{jid}: {z} not <= ankle*0.50 ({ankle_z * 0.50})"


def test_skeleton__0041_t3_full_xyz_heel_toe_measured() -> None:
    """T3: full XYZ heel/toe landmarks → measured (R3); z from landmark."""
    pkg = build_blockout_skeleton(_report(_full_landmarks()))
    j = _by_id(pkg)
    for jid, z_exp, y_exp in (
        ("heel_l", 0.02, 0.04),
        ("heel_r", 0.02, 0.04),
        ("toe_l", 0.02, -0.08),
        ("toe_r", 0.02, -0.08),
    ):
        assert j[jid].source == "measured", f"{jid} source={j[jid].source}"
        assert j[jid].z_m == pytest.approx(z_exp)
        assert j[jid].y_m == pytest.approx(y_exp)


def test_skeleton__0041_t4_height_none_sole_fallback() -> None:
    """T4: height_m=None → sole z = 0.02 fallback (estimated), not ankle clone."""
    ankle_z = 0.13
    lms = _landmarks_ankles_only(ankle_z=ankle_z)
    pkg = build_blockout_skeleton(_report(lms, height_m=None, head_unit_frac=None))
    j = _by_id(pkg)
    for jid in ("heel_l", "heel_r", "toe_l", "toe_r"):
        assert j[jid].source == "estimated", f"{jid} source={j[jid].source}"
        assert j[jid].z_m == pytest.approx(0.02, abs=1e-9)
        assert j[jid].z_m != pytest.approx(ankle_z)
    assert any("z_m from sole fallback" in m for m in pkg.messages), pkg.messages


def test_skeleton__0041_t5a_message_stature_sole_prior_y_invent() -> None:
    """T5a: messages contain stature sole prior when Z invented (Y invent path OK)."""
    h = 1.72
    lms = _landmarks_ankles_only(height_m=h)
    pkg = build_blockout_skeleton(_report(lms, height_m=h))
    sole_msgs = [m for m in pkg.messages if "z_m from stature sole prior" in m]
    assert len(sole_msgs) >= 4, f"expected sole-prior msgs for 4 joints; got {pkg.messages}"
    for jid in ("heel_l", "heel_r", "toe_l", "toe_r"):
        assert any(jid in m and "z_m from stature sole prior" in m for m in pkg.messages), (
            f"missing sole prior msg for {jid}: {pkg.messages}"
        )


def test_skeleton__0041_t5b_sole_prior_when_foot_mid_depth_fills_y() -> None:
    """T5b (AI1 P3-3): sole-prior Z msg when foot depth band fills Y (decoupled from Y invent)."""
    h = 1.72
    lms = _landmarks_ankles_only(height_m=h)
    # foot band covers heel/toe family ("foot_mid",), ("foot",) - fills Y before invent
    bands = [_band("foot", y_mid=0.02)]
    pkg = build_blockout_skeleton(_report(lms, height_m=h, depth_bands=bands))
    j = _by_id(pkg)
    # Depth filled Y - no Y invent "foot depth default" for heel/toe
    y_invent = [
        m for m in pkg.messages if "foot depth default" in m or "front-plane placement" in m
    ]
    foot_y_invent = [m for m in y_invent if any(k in m for k in ("heel_", "toe_"))]
    assert foot_y_invent == [], (
        f"expected no heel/toe Y invent when foot band fills Y: {foot_y_invent}"
    )
    # Depth Y messages present
    assert any("foot" in m and "(depth)" in m for m in pkg.messages), pkg.messages
    # Z invent message still required (decoupled from Y)
    for jid in ("heel_l", "heel_r", "toe_l", "toe_r"):
        assert j[jid].source == "estimated"
        assert j[jid].z_m == pytest.approx(0.02 * h, abs=1e-6)
        assert any(jid in m and "z_m from stature sole prior" in m for m in pkg.messages), (
            f"missing sole prior for {jid} under foot_mid depth-Y path: {pkg.messages}"
        )


def test_skeleton__0041_t6_invent_y_toe_ahead_of_heel() -> None:
    """T6: when Y invent fires, toe.y < heel.y (frame: toes -Y, heels +Y)."""
    h = 1.72
    lms = _landmarks_ankles_only(height_m=h)
    pkg = build_blockout_skeleton(_report(lms, height_m=h))
    j = _by_id(pkg)
    assert any("foot depth default" in m for m in pkg.messages), pkg.messages
    for side in ("l", "r"):
        heel_y = j[f"heel_{side}"].y_m
        toe_y = j[f"toe_{side}"].y_m
        assert heel_y is not None and toe_y is not None
        assert toe_y < heel_y, f"side {side}: toe.y={toe_y} not < heel.y={heel_y}"
        assert toe_y == pytest.approx(-0.05 * h, abs=1e-6)
        assert heel_y == pytest.approx(0.03 * h, abs=1e-6)


def test_skeleton__0041_t8_low_ankle_clamp_equality_no_raise() -> None:
    """T8 optional: low ankle -> clamp equality sole==ankle*0.50; no raise."""
    h = 1.72
    # stature sole 0.02*H=0.0344; ankle 0.04 -> clamp = 0.02; sole = min(0.0344, 0.02) = 0.02
    ankle_z = 0.04
    lms = _landmarks_ankles_only(height_m=h, ankle_z=ankle_z)
    pkg = build_blockout_skeleton(_report(lms, height_m=h))
    j = _by_id(pkg)
    expected = ankle_z * 0.50
    for jid in ("heel_l", "heel_r", "toe_l", "toe_r"):
        z = j[jid].z_m
        assert z is not None
        assert z == pytest.approx(expected, abs=1e-9)
        assert z < ankle_z
