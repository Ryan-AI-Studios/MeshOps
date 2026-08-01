"""Plateau gate for hosted fallback — no network imports (C1 / R19)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from meshops.hosted.errors import HostedError
from meshops.organic.models import PlateauRecord

# Criteria that must all be present for allows_hosted_fallback to be trusted open.
REQUIRED_CRITERIA: frozenset[str] = frozenset(
    {
        "min_one_pass",
        "max_passes_or_reason",
        "all_passes_have_views",
        "status_plateau",
    }
)


def load_plateau(path: Path | str) -> PlateauRecord:
    """Load and validate plateau.json as PlateauRecord.

    Raises:
        HostedError plateau_missing — path does not exist
        HostedError plateau_invalid — JSON / schema validation failure
    """
    p = Path(path)
    if not p.is_file():
        raise HostedError(
            f"plateau.json not found: {p}",
            code="plateau_missing",
            details={"path": str(p)},
        )
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HostedError(
            f"plateau.json unreadable or invalid JSON: {exc}",
            code="plateau_invalid",
            details={"path": str(p), "error": str(exc)},
        ) from exc
    try:
        return PlateauRecord.model_validate(data)
    except ValidationError as exc:
        raise HostedError(
            f"plateau.json failed PlateauRecord validation: {exc}",
            code="plateau_invalid",
            details={"path": str(p), "error": str(exc)},
        ) from exc


def assert_hosted_fallback_allowed(
    record: PlateauRecord,
    *,
    session_id: str | None = None,
) -> list[str]:
    """Assert plateau gate is open for hosted fallback.

    Returns warning messages (e.g. session_id mismatch). Does **not** hard-fail
    on session_id mismatch — warn only (R17).

    Raises:
        HostedError plateau_gate_closed — allows false or incomplete criteria_met
    """
    messages: list[str] = []

    if not record.allows_hosted_fallback:
        raise HostedError(
            "plateau does not allow hosted fallback (allows_hosted_fallback=false)",
            code="plateau_gate_closed",
            details={
                "session_id": record.session_id,
                "allows_hosted_fallback": False,
                "criteria_met": list(record.criteria_met),
            },
        )

    met = set(record.criteria_met or [])
    missing = sorted(REQUIRED_CRITERIA - met)
    if missing:
        raise HostedError(
            f"plateau criteria incomplete for hosted gate: missing {missing}",
            code="plateau_gate_closed",
            details={
                "session_id": record.session_id,
                "criteria_met": list(record.criteria_met),
                "missing": missing,
            },
        )

    if session_id and session_id != record.session_id:
        messages.append(f"session_id mismatch: flag={session_id!r} plateau={record.session_id!r}")

    return messages


def validate_plateau_gate(
    path: Path | str,
    *,
    session_id: str | None = None,
) -> tuple[PlateauRecord, list[str]]:
    """Load plateau + assert gate open. Returns (record, warning messages)."""
    record = load_plateau(path)
    messages = assert_hosted_fallback_allowed(record, session_id=session_id)
    return record, messages
