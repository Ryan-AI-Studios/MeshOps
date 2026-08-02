"""Orchestrate load → assist/frame → fuse → checks → package_score → report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meshops.proportion.assist import apply_assist, find_default_assist, load_assist_json
from meshops.proportion.checks import run_checks
from meshops.proportion.errors import ProportionError
from meshops.proportion.frame import apply_heuristic_frame, figure_span_from_landmarks
from meshops.proportion.fuse import (
    compute_package_score,
    fuse_xyz,
    head_unit_frac_from_front,
    vertical_span_discrepancy,
)
from meshops.proportion.honesty import PROPORTION_HONESTY
from meshops.proportion.load_views import load_views
from meshops.proportion.models import (
    PROPORTION_SCHEMA_VERSION,
    ProportionReport,
    QualityFlags,
    ViewLandmarks,
)


def analyze_proportion(
    views_dir: Path | str,
    *,
    landmarks_path: Path | str | None = None,
    height_m: float | None = None,
    out_dir: Path | str | None = None,
    partial_ok: bool = False,
    overlays: bool = False,
    allow_pillow: bool = True,
    run_heuristic_frame: bool = True,
) -> ProportionReport:
    """Analyze a multi-view package and optionally write proportion_report.json."""
    root = Path(views_dir)
    messages: list[str] = []

    view_images = load_views(root, allow_pillow=allow_pillow, partial_ok=partial_ok)
    if not view_images and partial_ok:
        raise ProportionError(
            f"no canonical views found under {root}",
            code="missing_views",
            details={"path": str(root)},
        )

    # Assist
    assist_path: Path | None = (
        Path(landmarks_path) if landmarks_path is not None else find_default_assist(root)
    )

    pose: str = "unknown"
    multi_from_assist = False
    views: dict[str, ViewLandmarks] = {}

    if assist_path is not None:
        assist = load_assist_json(assist_path)
        views, pose, multi_from_assist, assist_notes = apply_assist(assist, view_images)
        messages.extend(assist_notes)
        messages.append(f"assist loaded: {assist_path}")
    else:
        for key, img in view_images.items():
            views[key] = ViewLandmarks(
                view=key,
                width_px=img.width_px,
                height_px=img.height_px,
                path=str(img.path),
            )
        messages.append("no landmarks_assist.json — frame/heuristic only")

    # Heuristic frame (bbox / multi-blob; never invent joints)
    multi_from_frame = False
    if run_heuristic_frame:
        views, multi_from_frame, frame_notes = apply_heuristic_frame(view_images, views)
        messages.extend(frame_notes)

    # Refresh figure_span from landmarks when available
    for vl in views.values():
        span = figure_span_from_landmarks(vl)
        if span is not None:
            vl.figure_span_px = span

    quality = QualityFlags()
    multi = multi_from_assist or multi_from_frame
    if multi:
        quality.multi_figure = True
        quality.needs_user_input = True
        messages.append(
            "multi_figure signal → needs_user_input (Difficulty §1; never auto-pick primary)"
        )

    required = ("front", "left", "three_quarter")
    missing_req = [k for k in required if k not in views]
    if missing_req:
        quality.partial_package = True
        messages.append(f"partial package; missing required views: {', '.join(missing_req)}")

    # Foreshortening
    disc = vertical_span_discrepancy(views.get("front"), views.get("left"))
    foreshorten = disc is not None and disc > 0.05
    if foreshorten:
        quality.foreshortening_risk = True
        messages.append(f"vertical_span_discrepancy={disc:.4f} > 0.05 → foreshortening_risk")

    # HU
    head_unit: float | None = None
    front = views.get("front")
    if front is not None:
        head_unit, hair, hu_msgs = head_unit_frac_from_front(front)
        messages.extend(hu_msgs)
        if hair:
            quality.hair_volume_margin = True

    # Fuse XYZ
    landmarks_xyz, fuse_quality, fuse_msgs = fuse_xyz(
        views,
        height_m=height_m,
        foreshortening_risk=foreshorten,
    )
    messages.extend(fuse_msgs)
    # Merge quality flags from fuse
    quality.incomplete_stature = quality.incomplete_stature or fuse_quality.incomplete_stature
    quality.hair_volume_margin = quality.hair_volume_margin or fuse_quality.hair_volume_margin
    quality.foreshortening_risk = quality.foreshortening_risk or fuse_quality.foreshortening_risk
    if fuse_quality.notes:
        quality.notes.extend(fuse_quality.notes)

    # Checks
    checks = run_checks(views, pose=str(pose), head_unit_frac=head_unit)

    # package_score
    score, breakdown = compute_package_score(views)

    report = ProportionReport(
        schema_version=PROPORTION_SCHEMA_VERSION,
        honesty=PROPORTION_HONESTY,
        package_score=score,
        pose=pose,
        height_m=height_m,
        head_unit_frac=head_unit,
        figure_height_frac=1.0 if not quality.incomplete_stature and front else None,
        vertical_span_discrepancy=disc,
        views=views,
        landmarks_xyz=landmarks_xyz,
        checks=checks,
        quality=quality,
        messages=messages,
        score_breakdown=breakdown,
    )

    # Write outputs
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "proportion_report.json"
        try:
            json_path.write_text(
                json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ProportionError(
                f"failed to write {json_path}: {exc}",
                code="write_failed",
            ) from exc
        md_path = out / "proportion_report.md"
        md_path.write_text(report_to_markdown(report), encoding="utf-8")
        messages.append(f"wrote {json_path}")
        messages.append(f"wrote {md_path}")
        report.messages = list(messages)

        if overlays:
            from meshops.proportion.overlays import write_overlays

            written = write_overlays(report, root, out / "overlays")
            for p in written:
                report.messages.append(f"overlay: {p}")

    return report


def load_report(path: Path | str) -> ProportionReport:
    """Load a proportion_report.json from disk."""
    p = Path(path)
    try:
        raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot load report: {p}: {exc}",
            code="invalid_report",
            details={"path": str(p)},
        ) from exc
    try:
        return ProportionReport.model_validate(raw)
    except Exception as exc:
        raise ProportionError(
            f"invalid proportion report: {exc}",
            code="invalid_report",
            details={"path": str(p)},
        ) from exc


def report_to_markdown(report: ProportionReport) -> str:
    """Human-readable markdown summary (not a success claim)."""
    lines = [
        "# Proportion report",
        "",
        f"- **schema_version:** {report.schema_version}",
        f"- **honesty:** `{report.honesty}`",
        f"- **package_score:** {report.package_score:.2f} / 100",
        f"- **pose:** {report.pose}",
    ]
    if report.height_m is not None:
        lines.append(f"- **height_m:** {report.height_m}")
    if report.head_unit_frac is not None:
        lines.append(
            f"- **head_unit_frac:** {report.head_unit_frac:.4f} "
            f"(~{1.0 / report.head_unit_frac:.2f} heads)"
        )
    if report.vertical_span_discrepancy is not None:
        lines.append(f"- **vertical_span_discrepancy:** {report.vertical_span_discrepancy:.4f}")
    lines.extend(
        [
            "",
            "## Quality flags",
            "",
            f"- hair_volume_margin: {report.quality.hair_volume_margin}",
            f"- foreshortening_risk: {report.quality.foreshortening_risk}",
            f"- multi_figure: {report.quality.multi_figure}",
            f"- needs_user_input: {report.quality.needs_user_input}",
            f"- incomplete_stature: {report.quality.incomplete_stature}",
            f"- partial_package: {report.quality.partial_package}",
            "",
            "## Views",
            "",
        ]
    )
    for key, vl in report.views.items():
        lines.append(
            f"- **{key}:** {vl.width_px}x{vl.height_px}px, "
            f"{len(vl.landmarks)} landmarks, span={vl.figure_span_px}"
        )
    lines.extend(["", "## Checks", ""])
    for c in report.checks:
        mark = "OK" if c.ok else "FLAG"
        lines.append(f"- [{mark}] **{c.name}** ({c.severity}): {c.message}")
    lines.extend(["", "## Messages", ""])
    for m in report.messages:
        lines.append(f"- {m}")
    lines.extend(
        [
            "",
            "---",
            "",
            "_Proportion measurement only — not mesh or print success (Difficulty §12 / N6)._",
            "",
        ]
    )
    return "\n".join(lines)
