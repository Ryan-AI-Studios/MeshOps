"""Plateau protocol — quality reason + machine-readable criteria_met (0007 gate)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from meshops.organic.errors import OrganicError
from meshops.organic.models import REQUIRED_VIEW_KEYS, PlateauRecord
from meshops.organic.paths import SessionPaths
from meshops.organic.report import write_session_report
from meshops.organic.session import load_session, require_not_finalized, save_manifest

# Case-insensitive filler denylist (A1-BS5)
FILLER_REASONS: frozenset[str] = frozenset(
    {
        "done",
        "plateau",
        "n/a",
        "na",
        "skip",
        "ok",
        "stop",
        "finish",
        "finished",
        "enough",
        "whatever",
    }
)

REASON_MIN_LEN = 15


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def validate_plateau_reason(reason: str) -> str:
    """Strip + length + filler check; raise plateau_reason_* on fail."""
    if reason is None:
        raise OrganicError(
            "plateau reason is required",
            code="plateau_reason_required",
        )
    cleaned = reason.strip()
    if not cleaned:
        raise OrganicError(
            "plateau reason is required (empty after strip)",
            code="plateau_reason_required",
        )
    if len(cleaned) < REASON_MIN_LEN:
        raise OrganicError(
            f"plateau reason must be ≥{REASON_MIN_LEN} chars after strip (got {len(cleaned)})",
            code="plateau_reason_weak",
            details={"reason": cleaned, "len": len(cleaned)},
        )
    if cleaned.lower() in FILLER_REASONS:
        raise OrganicError(
            f"plateau reason is a filler token: {cleaned!r}",
            code="plateau_reason_weak",
            details={"reason": cleaned},
        )
    return cleaned


def _all_passes_have_views(paths: SessionPaths, pass_names: list[str]) -> bool:
    for name in pass_names:
        views = paths.pass_dir(name) / "views"
        for key in REQUIRED_VIEW_KEYS:
            p = views / f"{key}.png"
            if not p.is_file() or p.stat().st_size <= 0:
                return False
    return True


def mark_plateau(
    session_id: str,
    reason: str,
    *,
    work_root: Path | str = "work",
    max_passes: int | None = None,
) -> PlateauRecord:
    """Record plateau; set allows_hosted_fallback only when all criteria met."""
    paths, manifest = load_session(session_id, work_root=work_root)
    require_not_finalized(manifest)

    cleaned = validate_plateau_reason(reason)
    mp = max_passes if max_passes is not None else manifest.max_passes
    pass_count = len(manifest.passes)

    criteria: list[str] = []
    if pass_count >= 1:
        criteria.append("min_one_pass")
    if pass_count >= mp or cleaned:
        # quality reason accepted OR hit max — reason already validated above
        criteria.append("max_passes_or_reason")
    if pass_count >= 1 and _all_passes_have_views(paths, list(manifest.passes)):
        criteria.append("all_passes_have_views")

    # status_plateau after we set status
    required = {
        "min_one_pass",
        "max_passes_or_reason",
        "all_passes_have_views",
        "status_plateau",
    }

    manifest.status = "plateau"
    criteria.append("status_plateau")
    # de-dupe preserve order
    seen: set[str] = set()
    criteria_met = []
    for c in criteria:
        if c not in seen:
            seen.add(c)
            criteria_met.append(c)

    allows = required.issubset(set(criteria_met))

    record = PlateauRecord(
        session_id=manifest.session_id,
        reason=cleaned,
        pass_count=pass_count,
        max_passes=mp,
        criteria_met=criteria_met,
        created_at=_now_iso(),
        allows_hosted_fallback=allows,
    )
    paths.plateau_json.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    save_manifest(paths, manifest)
    write_session_report(
        paths,
        manifest,
        extra_lines=[
            "## Plateau",
            "",
            f"- reason: {cleaned}",
            f"- allows_hosted_fallback: {allows}",
            f"- criteria_met: {', '.join(criteria_met)}",
        ],
    )
    return record
