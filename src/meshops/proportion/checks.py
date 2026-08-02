"""Comparative-measurement checks (flag, do not force-fit).

Head unit, crotch vs half-body, pose-aware arm vectors, knee mid hip→heel,
vertical_span_discrepancy foreshortening.
"""

from __future__ import annotations

import math
from typing import Any

from meshops.proportion.fuse import head_unit_frac_from_front, vertical_span_discrepancy
from meshops.proportion.models import CheckResult, ViewLandmarks

# Tolerances (generous — art canons, not force-fit)
_HU_8HEAD = 1.0 / 8.0
_HU_TOL = 0.03  # ± ~0.24 head
_CROTCH_HALF_TOL = 0.08
_KNEE_MID_TOL = 0.08
_ARM_ALIGN_TOL = 0.12  # fraction of figure_h along arm path


def run_checks(
    views: dict[str, ViewLandmarks],
    *,
    pose: str = "unknown",
    head_unit_frac: float | None = None,
) -> list[CheckResult]:
    """Run macro checks against available landmarks."""
    checks: list[CheckResult] = []
    front = views.get("front")
    left = views.get("left")

    # Head unit
    hu = head_unit_frac
    if front is not None and hu is None:
        hu, _hair, _msgs = head_unit_frac_from_front(front)
    if hu is not None:
        # Info: report HU; warn only if wildly off realistic range
        ok = 1.0 / 10.0 <= hu <= 1.0 / 5.0  # 5-10 head figures
        checks.append(
            CheckResult(
                name="figure_head_units",
                ok=ok,
                severity="info" if ok else "warn",
                message=(
                    f"head unit ≈ {1.0 / hu:.2f} heads tall "
                    f"(hu_frac={hu:.4f}; 8-head canon={_HU_8HEAD:.4f})"
                ),
                measured=hu,
                expected={"eight_head": _HU_8HEAD, "seven_half": 1.0 / 7.5},
            )
        )
    else:
        checks.append(
            CheckResult(
                name="figure_head_units",
                ok=False,
                severity="warn",
                message="cannot compute head unit (need cranial|hair + chin + sole)",
                measured=None,
                expected="cranial_vertex→chin / figure_h",
            )
        )

    # Crotch vs half-body
    if front is not None:
        checks.append(_crotch_half_check(front))

    # Knee mid hip→heel
    if front is not None:
        checks.append(_knee_mid_check(front))

    # Pose-aware arms
    if front is not None:
        checks.extend(_arm_checks(front, pose=pose))

    # Shoulder vs pelvis width (info only)
    if front is not None:
        checks.append(_shoulder_pelvis_info(front))

    # Foreshortening span discrepancy
    disc = vertical_span_discrepancy(front, left)
    if disc is not None:
        ok = disc <= 0.05
        checks.append(
            CheckResult(
                name="vertical_span_discrepancy",
                ok=ok,
                severity="warn" if not ok else "info",
                message=(
                    f"front vs left vertical span discrepancy={disc:.4f} "
                    f"({'foreshortening_risk' if not ok else 'within 5%'})"
                ),
                measured=disc,
                expected=0.05,
            )
        )

    # three_quarter presence only (no IoU)
    tq = views.get("three_quarter")
    checks.append(
        CheckResult(
            name="three_quarter_present",
            ok=tq is not None,
            severity="info",
            message="three_quarter view present" if tq else "three_quarter view absent",
            measured=1.0 if tq else 0.0,
            expected="present for full package",
        )
    )

    return checks


def _figure_metrics(front: ViewLandmarks) -> tuple[float, float, float] | None:
    """Return (top_y, sole_y, figure_h) or None."""
    lm = front.landmarks
    sole = lm.get("sole")
    top = lm.get("cranial_vertex") or lm.get("hair_crown")
    if sole is None or top is None:
        return None
    fh = sole.y_px - top.y_px
    if fh <= 0:
        return None
    return top.y_px, sole.y_px, fh


def _crotch_half_check(front: ViewLandmarks) -> CheckResult:
    m = _figure_metrics(front)
    crotch = front.landmarks.get("crotch_pubic")
    if m is None or crotch is None:
        return CheckResult(
            name="crotch_vs_half_body",
            ok=True,
            severity="info",
            message="skipped (need stature + crotch_pubic)",
            measured=None,
            expected="~0.5 of stature (8-head) or slightly above (7.5)",
        )
    _top_y, sole_y, fh = m
    # z_frac of crotch from soles
    z_crotch = (sole_y - crotch.y_px) / fh
    # In image y-down, crotch should be near mid: z ≈ 0.5 from soles
    half_err = abs(z_crotch - 0.5)
    ok = half_err <= _CROTCH_HALF_TOL + 0.05  # loose
    return CheckResult(
        name="crotch_vs_half_body",
        ok=ok,
        severity="info",
        message=(
            f"crotch z_frac from soles={z_crotch:.3f} (8-head mid≈0.50; 7.5 slightly above mid)"
        ),
        measured=z_crotch,
        expected=0.5,
    )


def _knee_mid_check(front: ViewLandmarks) -> CheckResult:
    lm = front.landmarks
    knee = lm.get("knee") or lm.get("knee_l") or lm.get("knee_r")
    hip = lm.get("greater_trochanter") or lm.get("hip_l") or lm.get("crotch_pubic")
    heel = lm.get("heel") or lm.get("ankle") or lm.get("sole")
    if knee is None or hip is None or heel is None:
        return CheckResult(
            name="knee_mid_hip_heel",
            ok=True,
            severity="info",
            message="skipped (need knee + hip + heel/ankle/sole)",
            measured=None,
            expected="knee ≈ midpoint hip→heel",
        )
    mid_y = (hip.y_px + heel.y_px) / 2.0
    m = _figure_metrics(front)
    fh = m[2] if m else max(abs(heel.y_px - hip.y_px), 1.0)
    err = abs(knee.y_px - mid_y) / fh
    ok = err <= _KNEE_MID_TOL
    return CheckResult(
        name="knee_mid_hip_heel",
        ok=ok,
        severity="info" if ok else "warn",
        message=f"knee vs mid hip→heel error_frac={err:.3f}",
        measured=err,
        expected=_KNEE_MID_TOL,
    )


def _arm_checks(front: ViewLandmarks, *, pose: str) -> list[CheckResult]:
    out: list[CheckResult] = []
    for side in ("l", "r"):
        out.append(_one_arm_check(front, side=side, pose=pose))
    return out


def _one_arm_check(front: ViewLandmarks, *, side: str, pose: str) -> CheckResult:
    lm = front.landmarks
    shoulder = lm.get(f"shoulder_{side}")
    elbow = lm.get(f"elbow_{side}")
    wrist = lm.get(f"wrist_{side}")
    fingertip = lm.get(f"fingertip_{side}")
    name = f"arm_chain_{side}"

    if shoulder is None or elbow is None:
        return CheckResult(
            name=name,
            ok=True,
            severity="info",
            message=f"skipped arm {side} (need shoulder+elbow)",
            measured=None,
            expected="pose-aware arm vector",
        )

    pose_l = (pose or "unknown").lower()
    m = _figure_metrics(front)
    fh = m[2] if m else float(front.height_px)

    if pose_l == "a_pose":
        # Project along image-plane arm vector shoulder→elbow→wrist
        # Elbow should lie near the segment shoulder→wrist (or progressive chain)
        if wrist is None:
            return CheckResult(
                name=name,
                ok=True,
                severity="info",
                message=f"a_pose arm {side}: wrist missing; chain partial OK",
                measured=None,
                expected="shoulder→elbow→wrist colinear-ish in plane",
            )
        # Parameter t of elbow along shoulder→wrist
        err = _point_to_segment_frac(
            elbow.x_px,
            elbow.y_px,
            shoulder.x_px,
            shoulder.y_px,
            wrist.x_px,
            wrist.y_px,
            fh,
        )
        # Also check wrist not forced to match hanging Y
        ok = err <= _ARM_ALIGN_TOL
        measured: dict[str, Any] = {"elbow_off_segment_frac": err, "pose": "a_pose"}
        if fingertip is not None:
            measured["fingertip_present"] = True
        return CheckResult(
            name=name,
            ok=ok,
            severity="info" if ok else "warn",
            message=(
                f"a_pose arm {side}: elbow off shoulder→wrist segment by {err:.3f} "
                f"figure_h (no raw-Y hanging fail)"
            ),
            measured=measured,
            expected={"max_off_segment_frac": _ARM_ALIGN_TOL, "pose": "a_pose"},
        )

    if pose_l == "hanging":
        # Raw vertical: elbow.y > shoulder.y, wrist.y > elbow.y (image y down)
        issues: list[str] = []
        if elbow.y_px < shoulder.y_px - 1:
            issues.append("elbow above shoulder")
        if wrist is not None and wrist.y_px < elbow.y_px - 1:
            issues.append("wrist above elbow")
        ok = not issues
        return CheckResult(
            name=name,
            ok=ok,
            severity="info" if ok else "warn",
            message=(f"hanging arm {side}: {'ok vertical chain' if ok else ', '.join(issues)}"),
            measured={"pose": "hanging", "issues": issues},
            expected="elbow/wrist below shoulder in image Y",
        )

    # Unknown pose → lower confidence, info only, never hard fail on Y
    return CheckResult(
        name=name,
        ok=True,
        severity="info",
        message=(
            f"arm {side}: pose={pose_l!r} — arm chain present; "
            "no hanging-Y hard check (lower confidence)"
        ),
        measured={"pose": pose_l, "has_wrist": wrist is not None},
        expected="declare pose=a_pose|hanging for stricter checks",
    )


def _point_to_segment_frac(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    figure_h: float,
) -> float:
    """Distance from point to segment AB, normalized by figure_h."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-9:
        dist = math.hypot(apx, apy)
    else:
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        cx, cy = ax + t * abx, ay + t * aby
        dist = math.hypot(px - cx, py - cy)
    return dist / max(figure_h, 1.0)


def _shoulder_pelvis_info(front: ViewLandmarks) -> CheckResult:
    lm = front.landmarks
    sl, sr = lm.get("shoulder_l"), lm.get("shoulder_r")
    hl, hr = lm.get("hip_l"), lm.get("hip_r")
    if not all(v is not None for v in (sl, sr, hl, hr)):
        return CheckResult(
            name="shoulder_vs_pelvis_width",
            ok=True,
            severity="info",
            message="skipped (need shoulder_l/r + hip_l/r)",
            measured=None,
            expected="info flag only",
        )
    assert sl is not None and sr is not None and hl is not None and hr is not None
    m = _figure_metrics(front)
    fh = m[2] if m else float(front.width_px)
    sw = abs(sr.x_px - sl.x_px) / fh
    hw = abs(hr.x_px - hl.x_px) / fh
    return CheckResult(
        name="shoulder_vs_pelvis_width",
        ok=True,
        severity="info",
        message=f"shoulder_width_frac={sw:.3f} hip_width_frac={hw:.3f} (info only)",
        measured={"shoulder_frac": sw, "hip_frac": hw},
        expected="gender-leaning widths — no identity claim",
    )
