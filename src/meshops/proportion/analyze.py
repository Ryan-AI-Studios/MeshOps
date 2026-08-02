"""Orchestrate load → assist/frame → fuse → diameters/depth → checks → report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meshops.proportion.assist import apply_assist, find_default_assist, load_assist_json
from meshops.proportion.checks import diameter_info_checks, run_checks
from meshops.proportion.diameters import compute_diameters
from meshops.proportion.errors import ProportionError
from meshops.proportion.frame import apply_heuristic_frame, figure_span_from_landmarks
from meshops.proportion.fuse import (
    build_cross_sections,
    build_depth_bands,
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
    edge_pairs: dict[str, Any] = {}

    if assist_path is not None:
        assist = load_assist_json(assist_path)
        views, pose, multi_from_assist, assist_notes, edge_pairs = apply_assist(assist, view_images)
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

    # --- Package checklist defaults (0014 R1/R7): after assist pose known, before checks ---
    effective_height_m = height_m
    leaf_cl = None
    parent_cl = None
    try:
        from meshops.proportion.checklist import (
            field_from_pair,
            load_package_checklist,
            resolve_checklist_pair,
        )

        leaf_path, parent_path = resolve_checklist_pair(root)
        if leaf_path is not None:
            try:
                leaf_cl = load_package_checklist(leaf_path)
            except ProportionError as exc:
                messages.append(f"package checklist soft-warn (leaf): {leaf_path}: {exc}")
                leaf_cl = None
        if parent_path is not None:
            try:
                parent_cl = load_package_checklist(parent_path)
            except ProportionError as exc:
                messages.append(f"package checklist soft-warn (parent): {parent_path}: {exc}")
                parent_cl = None

        # height_m: CLI > leaf non-null > parent non-null > None (R1)
        if height_m is None:
            picked = field_from_pair("height_m", leaf_cl, parent_cl)
            if picked is not None:
                effective_height_m = float(picked)
                src = (
                    str(leaf_path)
                    if leaf_cl is not None and leaf_cl.height_m is not None
                    else str(parent_path)
                )
                messages.append(
                    f"height_m={effective_height_m} from package_checklist.json ({src})"
                )

        # Pose injection only when assist pose still unknown (R7)
        if pose == "unknown":
            cl_pose = field_from_pair("pose", leaf_cl, parent_cl)
            if cl_pose is not None and cl_pose != "unknown":
                pose = str(cl_pose)
                messages.append("pose from package_checklist.json")

        # Multi-figure union from checklist (Difficulty §1)
        multi_from_cl = False
        for cl in (leaf_cl, parent_cl):
            if cl is None:
                continue
            if cl.multi_figure or len(cl.in_scope_figures) >= 2:
                multi_from_cl = True
                break
        if multi_from_cl:
            quality.multi_figure = True
            quality.needs_user_input = True
            if not multi:
                messages.append(
                    "multi_figure from package_checklist → needs_user_input "
                    "(Difficulty §1; never auto-pick primary)"
                )
            multi = True
    except Exception as exc:  # pragma: no cover — defensive soft path
        messages.append(f"package checklist soft-warn: {exc}")

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

    # Diameters first so edge landmarks inject into views before XYZ fuse
    diameters, diam_msgs = compute_diameters(edge_pairs, views, height_m=effective_height_m)
    messages.extend(diam_msgs)

    # Fuse XYZ (includes injected edge landmarks)
    landmarks_xyz, fuse_quality, fuse_msgs = fuse_xyz(
        views,
        height_m=effective_height_m,
        foreshortening_risk=foreshorten,
    )
    messages.extend(fuse_msgs)
    # Merge quality flags from fuse
    quality.incomplete_stature = quality.incomplete_stature or fuse_quality.incomplete_stature
    quality.hair_volume_margin = quality.hair_volume_margin or fuse_quality.hair_volume_margin
    quality.foreshortening_risk = quality.foreshortening_risk or fuse_quality.foreshortening_risk
    if fuse_quality.notes:
        quality.notes.extend(fuse_quality.notes)

    # Depth bands + orientation info checks
    depth_bands, depth_checks, depth_msgs = build_depth_bands(
        views,
        height_m=effective_height_m,
        foreshortening_risk=foreshorten,
    )
    messages.extend(depth_msgs)

    # Cross-sections when Rx/Ry z match (R13)
    cross_sections = build_cross_sections(diameters, depth_bands)
    if cross_sections:
        messages.append(f"cross_sections: {len(cross_sections)} level(s)")

    # Checks (canon signature unchanged) + diameter info checks + depth orientation
    checks = run_checks(views, pose=str(pose), head_unit_frac=head_unit)
    checks.extend(diameter_info_checks(diameters))
    checks.extend(depth_checks)

    # package_score (R8 — chest/hip depth only; weights unchanged)
    score, breakdown = compute_package_score(views)

    report = ProportionReport(
        schema_version=PROPORTION_SCHEMA_VERSION,
        honesty=PROPORTION_HONESTY,
        package_score=score,
        pose=pose,
        height_m=effective_height_m,
        head_unit_frac=head_unit,
        figure_height_frac=1.0 if not quality.incomplete_stature and front else None,
        vertical_span_discrepancy=disc,
        views=views,
        landmarks_xyz=landmarks_xyz,
        diameters=diameters,
        depth_bands=depth_bands,
        cross_sections=cross_sections,
        thickness_band_count=len(diameters),
        depth_band_count=len(depth_bands),
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
    lines.append(f"- **thickness_band_count:** {report.thickness_band_count}")
    lines.append(f"- **depth_band_count:** {report.depth_band_count}")
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

    # R6: fixed section headers
    lines.extend(["", "## Diameters", ""])
    if report.diameters:
        for d in report.diameters:
            z_s = f"{d.z_frac:.3f}" if d.z_frac is not None else "—"
            lines.append(
                f"- **{d.band_id}** ({d.view}): width_frac={d.width_frac:.4f} "
                f"(width_px={d.width_px:.1f}, eucl={d.width_eucl_px:.1f}, "
                f"theta_deg={d.theta_deg:.1f}), z_frac={z_s}, conf={d.confidence:.2f}"
            )
    else:
        lines.append("- (none)")

    lines.extend(["", "## Depth bands", ""])
    if report.depth_bands:
        for b in report.depth_bands:
            swap = " [swapped]" if b.orientation_swapped else ""
            z_s = f"{b.z_frac:.3f}" if b.z_frac is not None else "—"
            lines.append(
                f"- **{b.band_id}**: depth_frac={b.depth_frac:.4f}, "
                f"y_mid={b.y_mid:.4f}, z_frac={z_s}, conf={b.confidence:.2f}{swap}"
            )
    else:
        lines.append("- (none)")

    lines.extend(["", "## Cross-sections", ""])
    if report.cross_sections:
        for cs in report.cross_sections:
            lines.append(
                f"- **{cs.level_id}**: rx_frac={cs.rx_frac:.4f}, "
                f"ry_frac={cs.ry_frac:.4f}, z_frac={cs.z_frac:.3f}"
            )
    else:
        lines.append("- (none)")

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
