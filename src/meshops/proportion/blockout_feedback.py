"""Sticky post-export blockout feedback checklist (track 0043).

Orchestrates depth-samples → heatmap → silhouette front/left into a single
authoring QA package. Never mutates mesh/recipe; never calls optimize/fuse/
repair/accept/promote (B7).

Honesty: FEEDBACK_HONESTY — not mesh or print success (Difficulty §12 / N6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import FEEDBACK_HONESTY

FEEDBACK_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
FEEDBACK_JSON_BASENAME: Final[str] = "blockout_feedback.json"
FEEDBACK_MD_BASENAME: Final[str] = "blockout_feedback.md"
SAMPLES_BASENAME: Final[str] = "depth_at_landmarks.json"
DELTAS_BASENAME: Final[str] = "depth_mesh_deltas.json"

SOFT_METHOD: Final[Literal["band_weighted_abs_delta_y"]] = "band_weighted_abs_delta_y"

INCLUDED_BANDS: Final[list[str]] = [
    "breast",
    "chest",
    "hip",
    "glute",
    "thigh",
    "calf",
]
EXCLUDED_BANDS: Final[list[str]] = [
    "foot",
    "heel",
    "toe",
    "ank",
    "ankle",
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FeedbackStepResult(BaseModel):
    """One checklist step outcome."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    path: str | None = None
    skipped_reason: str | None = None
    iou: float | None = None
    dice: float | None = None
    trusted: bool | None = None
    messages: list[str] = Field(default_factory=list)


class SoftDepthSummary(BaseModel):
    """Band-weighted |ΔY| soft rollup (feedback only — not optimizer)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["band_weighted_abs_delta_y"] = SOFT_METHOD
    included_bands: list[str] = Field(default_factory=lambda: list(INCLUDED_BANDS))
    excluded_bands: list[str] = Field(default_factory=lambda: list(EXCLUDED_BANDS))
    score_m: float | None = None
    per_band: dict[str, float] = Field(default_factory=dict)


class BlockoutFeedbackPackage(BaseModel):
    """blockout_feedback.json package (schema 1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = FEEDBACK_SCHEMA_VERSION
    honesty: str = FEEDBACK_HONESTY
    steps: dict[str, FeedbackStepResult] = Field(default_factory=dict)
    soft_depth_summary: SoftDepthSummary = Field(default_factory=SoftDepthSummary)
    messages: list[str] = Field(default_factory=list)
    ok: bool = False


# ---------------------------------------------------------------------------
# Soft depth weights (B6)
# ---------------------------------------------------------------------------


def _band_family_weight(band_token: str) -> float:
    """Weight for a band id / token (substring match, lowercased)."""
    bid = band_token.lower()
    for tok in EXCLUDED_BANDS:
        if tok in bid:
            return 0.0
    if "breast" in bid or "chest" in bid:
        return 1.0
    if "hip" in bid:
        return 1.0
    if "glute" in bid:
        return 1.0
    if "thigh" in bid:
        return 0.8
    if "calf" in bid:
        return 0.6
    return 0.3


def _band_token_from_delta_id(delta_id: str, band_id: str | None = None) -> str:
    """Prefer explicit band_id; else parse band_<id>_* sample ids."""
    if band_id:
        return band_id
    # band_chest_front → chest; landmark ids fall through as-is
    parts = delta_id.split("_")
    if len(parts) >= 2 and parts[0] == "band":
        # band_<id>_<role> — id may be multi-token (e.g. band_foot_sole_front)
        # Use everything between band_ and last role token when role is known.
        roles = {"front", "back", "mid", "span"}
        if parts[-1] in roles and len(parts) >= 3:
            return "_".join(parts[1:-1])
        return "_".join(parts[1:])
    return delta_id


def compute_soft_depth_summary(
    *,
    deltas: list[dict[str, Any]] | None = None,
    samples: list[dict[str, Any]] | None = None,
    sample_band_by_id: dict[str, str | None] | None = None,
) -> SoftDepthSummary:
    """Weighted mean of |ΔY| over non-zero-weight bands (B6).

    Prefer mesh deltas ``delta_y_m``. Without deltas, ``score_m`` is null.
    Foot-family tokens weight 0.0 (future-proof; never dominate score).
    """
    per_band_vals: dict[str, list[float]] = {}
    sample_band_by_id = sample_band_by_id or {}

    if deltas:
        for d in deltas:
            dy = d.get("delta_y_m")
            if dy is None:
                continue
            did = str(d.get("id", ""))
            band_tok = _band_token_from_delta_id(
                did,
                sample_band_by_id.get(did),
            )
            w = _band_family_weight(band_tok)
            if w <= 0.0:
                # Still record per_band for transparency? Plan: excluded from rollup.
                # Keep per_band only for weighted contributors + note excluded separately.
                continue
            per_band_vals.setdefault(band_tok, []).append(abs(float(dy)))
    elif samples:
        # Samples alone have no mesh ΔY — cannot form residual score.
        _ = samples

    per_band: dict[str, float] = {
        k: float(sum(vs) / len(vs)) for k, vs in per_band_vals.items() if vs
    }

    weighted_sum = 0.0
    weight_total = 0.0
    for band_tok, mean_abs in per_band.items():
        w = _band_family_weight(band_tok)
        if w <= 0.0:
            continue
        weighted_sum += w * mean_abs
        weight_total += w

    score_m = weighted_sum / weight_total if weight_total > 0.0 else None

    return SoftDepthSummary(
        method=SOFT_METHOD,
        included_bands=list(INCLUDED_BANDS),
        excluded_bands=list(EXCLUDED_BANDS),
        score_m=score_m,
        per_band=per_band,
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _step_ok_mark(step: FeedbackStepResult) -> str:
    if step.skipped_reason:
        return "—"
    return "✓" if step.ok else "✗"


def _render_markdown(package: BlockoutFeedbackPackage) -> str:
    lines = [
        "# Blockout feedback (authoring only — not mesh/print success)",
        "",
        f"honesty: `{package.honesty}`",
        f"package ok: **{'true' if package.ok else 'false'}**",
        "",
        "| Step | ok | notes |",
        "|------|----|-------|",
    ]
    for name, step in package.steps.items():
        notes: list[str] = []
        if step.skipped_reason:
            notes.append(f"skipped: {step.skipped_reason}")
        if step.iou is not None:
            notes.append(f"iou={step.iou:.4f}")
        if step.dice is not None:
            notes.append(f"dice={step.dice:.4f}")
        if step.trusted is not None:
            notes.append(f"trusted={step.trusted}")
        if step.path:
            notes.append(step.path)
        notes.extend(step.messages[:2])
        lines.append(f"| {name} | {_step_ok_mark(step)} | {'; '.join(notes) or '—'} |")

    soft = package.soft_depth_summary
    lines.extend(
        [
            "",
            "## Soft depth summary",
            "",
            f"- method: `{soft.method}`",
            f"- score_m: {soft.score_m if soft.score_m is not None else 'null'}",
            f"- included_bands: {', '.join(soft.included_bands)}",
            f"- excluded_bands (weight 0): {', '.join(soft.excluded_bands)}",
            f"- per_band: {json.dumps(soft.per_band, sort_keys=True)}",
            "",
            "Do not thrash untrusted silhouettes. Do not chase IoU 1.0.",
            "Authoring QA only — not mesh or print success (N6).",
            "",
        ]
    )
    if package.messages:
        lines.append("## Messages")
        lines.append("")
        for m in package.messages:
            lines.append(f"- {m}")
        lines.append("")
    return "\n".join(lines)


def _write_text(path: Path, text: str, *, force: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            raise ProportionError(
                f"output already exists (use --force): {path}",
                code="write_failed",
                details={"path": str(path)},
            )
        path.write_text(text, encoding="utf-8")
    except ProportionError:
        raise
    except OSError as exc:
        raise ProportionError(
            f"failed to write blockout feedback: {exc}",
            code="write_failed",
            details={"path": str(path)},
        ) from exc


def _package_ok(steps: dict[str, FeedbackStepResult]) -> bool:
    """B5.1: ok iff every non-skipped step has ok=true."""
    for step in steps.values():
        if step.skipped_reason is not None:
            continue
        if not step.ok:
            return False
    return True


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def run_blockout_feedback(
    report: Path | str,
    out: Path | str,
    *,
    mesh: Path | str | None = None,
    ref_front: Path | str | None = None,
    ref_left: Path | str | None = None,
    mesh_view_front: Path | str | None = None,
    mesh_view_left: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run sticky post-export checklist; write blockout_feedback.json + .md.

    Orchestrates existing engines only (B7). Returns CLI/MCP success payload.
    """
    from meshops.proportion.depth_heatmap import run_depth_heatmap
    from meshops.proportion.depth_samples import (
        DELTAS_BASENAME as DS_DELTAS,
    )
    from meshops.proportion.depth_samples import (
        SAMPLES_BASENAME as DS_SAMPLES,
    )
    from meshops.proportion.depth_samples import (
        run_depth_samples,
    )
    from meshops.proportion.silhouette import run_silhouette_compare

    report_p = Path(report)
    out_dir = Path(str(out).rstrip("/\\"))
    mesh_p = Path(mesh) if mesh is not None else None
    ref_front_p = Path(ref_front) if ref_front is not None else None
    ref_left_p = Path(ref_left) if ref_left is not None else None
    mesh_view_front_p = Path(mesh_view_front) if mesh_view_front is not None else None
    mesh_view_left_p = Path(mesh_view_left) if mesh_view_left is not None else None

    if not report_p.is_file():
        raise ProportionError(
            f"proportion report not found: {report_p}",
            code="feedback_failed",
            details={"report": str(report_p)},
        )

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProportionError(
            f"failed to create feedback out dir: {exc}",
            code="write_failed",
            details={"out": str(out_dir)},
        ) from exc

    steps: dict[str, FeedbackStepResult] = {}
    messages: list[str] = []
    paths: list[str] = []

    samples_path = out_dir / DS_SAMPLES
    deltas_path = out_dir / DS_DELTAS

    # --- 1-3 depth samples (+ optional mesh deltas) ---
    try:
        depth_payload = run_depth_samples(
            report_p,
            out_dir,
            mesh=mesh_p,
            force=force,
        )
        messages.extend(list(depth_payload.get("messages") or []))
        for p in depth_payload.get("paths") or []:
            paths.append(str(p))
        steps["depth_samples"] = FeedbackStepResult(
            ok=True,
            path=str(samples_path) if samples_path.is_file() else None,
        )
        if mesh_p is not None:
            steps["depth_mesh_deltas"] = FeedbackStepResult(
                ok=deltas_path.is_file(),
                path=str(deltas_path) if deltas_path.is_file() else None,
                messages=(
                    []
                    if deltas_path.is_file()
                    else ["mesh deltas path missing after depth-samples"]
                ),
            )
            if not deltas_path.is_file():
                steps["depth_mesh_deltas"].ok = False
        else:
            steps["depth_mesh_deltas"] = FeedbackStepResult(
                ok=False,
                path=None,
                skipped_reason="no --mesh",
            )
    except ProportionError as exc:
        messages.append(f"depth_samples failed: {exc}")
        steps["depth_samples"] = FeedbackStepResult(
            ok=False,
            path=None,
            messages=[str(exc)],
        )
        if mesh_p is not None:
            steps["depth_mesh_deltas"] = FeedbackStepResult(
                ok=False,
                path=None,
                messages=[f"skipped after depth_samples failure: {exc}"],
            )
        else:
            steps["depth_mesh_deltas"] = FeedbackStepResult(
                ok=False,
                path=None,
                skipped_reason="no --mesh",
            )

    # --- 4 depth heatmap ---
    if steps["depth_samples"].ok and samples_path.is_file():
        try:
            hm_payload = run_depth_heatmap(
                samples_path,
                out_dir,
                deltas=deltas_path if deltas_path.is_file() else None,
                force=force,
            )
            messages.extend(list(hm_payload.get("messages") or []))
            for p in hm_payload.get("paths") or []:
                paths.append(str(p))
            hm_paths = [str(p) for p in (hm_payload.get("paths") or [])]
            steps["depth_heatmap"] = FeedbackStepResult(
                ok=True,
                path=hm_paths[0] if hm_paths else str(out_dir / "depth_heatmap.png"),
            )
        except ProportionError as exc:
            messages.append(f"depth_heatmap failed: {exc}")
            steps["depth_heatmap"] = FeedbackStepResult(
                ok=False,
                path=None,
                messages=[str(exc)],
            )
    else:
        steps["depth_heatmap"] = FeedbackStepResult(
            ok=False,
            path=None,
            skipped_reason="depth_samples unavailable",
        )

    # --- 5 silhouette front ---
    if ref_front_p is None:
        steps["silhouette_front"] = FeedbackStepResult(
            ok=False,
            path=None,
            skipped_reason="no --ref-front",
        )
    else:
        sil_front_dir = out_dir / "silhouette_front"
        try:
            if mesh_view_front_p is not None:
                sil_payload = run_silhouette_compare(
                    ref_front_p,
                    sil_front_dir,
                    mesh_view=mesh_view_front_p,
                    view_role="front",
                    force=force,
                )
            elif mesh_p is not None:
                sil_payload = run_silhouette_compare(
                    ref_front_p,
                    sil_front_dir,
                    mesh=mesh_p,
                    view_role="front",
                    force=force,
                )
            else:
                raise ProportionError(
                    "silhouette front requires --mesh-view-front or --mesh",
                    code="feedback_failed",
                    details={},
                )
            sil_json = sil_front_dir / "silhouette_compare.json"
            paths.append(str(sil_json))
            steps["silhouette_front"] = FeedbackStepResult(
                ok=bool(sil_payload.get("ok", True)),
                path=str(sil_json),
                iou=float(sil_payload["score_iou"])
                if sil_payload.get("score_iou") is not None
                else None,
                dice=float(sil_payload["score_dice"])
                if sil_payload.get("score_dice") is not None
                else None,
                trusted=bool(sil_payload.get("silhouette_trusted"))
                if "silhouette_trusted" in sil_payload
                else None,
                messages=list(sil_payload.get("messages") or []),
            )
        except ProportionError as exc:
            messages.append(f"silhouette_front failed: {exc}")
            steps["silhouette_front"] = FeedbackStepResult(
                ok=False,
                path=None,
                messages=[str(exc)],
            )

    # --- 6 silhouette left ---
    if ref_left_p is None:
        steps["silhouette_left"] = FeedbackStepResult(
            ok=False,
            path=None,
            skipped_reason="no --ref-left",
        )
    else:
        sil_left_dir = out_dir / "silhouette_left"
        try:
            if mesh_view_left_p is not None:
                sil_payload = run_silhouette_compare(
                    ref_left_p,
                    sil_left_dir,
                    mesh_view=mesh_view_left_p,
                    view_role="left",
                    force=force,
                )
            elif mesh_p is not None:
                sil_payload = run_silhouette_compare(
                    ref_left_p,
                    sil_left_dir,
                    mesh=mesh_p,
                    view_role="left",
                    force=force,
                )
            else:
                raise ProportionError(
                    "silhouette left requires --mesh-view-left or --mesh",
                    code="feedback_failed",
                    details={},
                )
            sil_json = sil_left_dir / "silhouette_compare.json"
            paths.append(str(sil_json))
            steps["silhouette_left"] = FeedbackStepResult(
                ok=bool(sil_payload.get("ok", True)),
                path=str(sil_json),
                iou=float(sil_payload["score_iou"])
                if sil_payload.get("score_iou") is not None
                else None,
                dice=float(sil_payload["score_dice"])
                if sil_payload.get("score_dice") is not None
                else None,
                trusted=bool(sil_payload.get("silhouette_trusted"))
                if "silhouette_trusted" in sil_payload
                else None,
                messages=list(sil_payload.get("messages") or []),
            )
        except ProportionError as exc:
            messages.append(f"silhouette_left failed: {exc}")
            steps["silhouette_left"] = FeedbackStepResult(
                ok=False,
                path=None,
                messages=[str(exc)],
            )

    # --- 7 soft depth summary ---
    deltas_list: list[dict[str, Any]] | None = None
    sample_band_by_id: dict[str, str | None] = {}
    samples_list: list[dict[str, Any]] | None = None
    if samples_path.is_file():
        try:
            samples_raw = json.loads(samples_path.read_text(encoding="utf-8"))
            samples_list = list(samples_raw.get("samples") or [])
            for s in samples_list:
                sid = str(s.get("id", ""))
                sample_band_by_id[sid] = s.get("band_id")
        except (OSError, json.JSONDecodeError) as exc:
            messages.append(f"could not read depth samples for soft summary: {exc}")
    if deltas_path.is_file():
        try:
            deltas_raw = json.loads(deltas_path.read_text(encoding="utf-8"))
            deltas_list = list(deltas_raw.get("deltas") or [])
        except (OSError, json.JSONDecodeError) as exc:
            messages.append(f"could not read depth deltas for soft summary: {exc}")

    soft = compute_soft_depth_summary(
        deltas=deltas_list,
        samples=samples_list,
        sample_band_by_id=sample_band_by_id,
    )
    if soft.score_m is None:
        messages.append(
            "soft_depth_summary.score_m is null "
            "(no usable non-foot |ΔY| residuals — mesh deltas required)"
        )

    package = BlockoutFeedbackPackage(
        schema_version=FEEDBACK_SCHEMA_VERSION,
        honesty=FEEDBACK_HONESTY,
        steps=steps,
        soft_depth_summary=soft,
        messages=messages,
        ok=_package_ok(steps),
    )

    json_path = out_dir / FEEDBACK_JSON_BASENAME
    md_path = out_dir / FEEDBACK_MD_BASENAME
    payload_pkg = package.model_dump(mode="json")
    _write_text(
        json_path,
        json.dumps(payload_pkg, indent=2) + "\n",
        force=force,
    )
    _write_text(md_path, _render_markdown(package), force=force)
    paths.extend([str(json_path), str(md_path)])

    return {
        "ok": package.ok,
        "paths": paths,
        "honesty": FEEDBACK_HONESTY,
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "steps": {k: v.model_dump(mode="json") for k, v in steps.items()},
        "soft_depth_summary": soft.model_dump(mode="json"),
        "messages": messages,
        "package_path": str(json_path),
        "markdown_path": str(md_path),
    }


__all__ = [
    "DELTAS_BASENAME",
    "EXCLUDED_BANDS",
    "FEEDBACK_JSON_BASENAME",
    "FEEDBACK_MD_BASENAME",
    "FEEDBACK_SCHEMA_VERSION",
    "INCLUDED_BANDS",
    "SAMPLES_BASENAME",
    "BlockoutFeedbackPackage",
    "FeedbackStepResult",
    "SoftDepthSummary",
    "compute_soft_depth_summary",
    "run_blockout_feedback",
]
