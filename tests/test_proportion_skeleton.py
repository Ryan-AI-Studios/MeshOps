"""Track 0026 — skeleton-first joint/bone graph (offline; no Blender)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshops.cli import app
from meshops.proportion.errors import ProportionError
from meshops.proportion.guides import AXIS_NOTES
from meshops.proportion.honesty import SKELETON_HONESTY
from meshops.proportion.models import LandmarkXYZ, ProportionReport, QualityFlags
from meshops.proportion.skeleton import (
    BPY_BASENAME,
    JSON_BASENAME,
    SKELETON_SCHEMA_VERSION,
    BlockoutSkeleton,
    SkeletonBone,
    SkeletonJoint,
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
) -> ProportionReport:
    return ProportionReport(
        schema_version="1.1.0",
        height_m=height_m,
        head_unit_frac=head_unit_frac,
        landmarks_xyz=lms if lms is not None else {},
        quality=QualityFlags(),
    )


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


def test_skeleton__measured_requires_full_xyz_front_plane_estimated() -> None:
    """5. measured only full XYZ; front-plane Y → estimated."""
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
    assert any("shoulder_l" in m and "front-plane" in m for m in pkg.messages)


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
