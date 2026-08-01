"""Write hosted_report.md + run_manifest.json (no secrets)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meshops.hosted.honesty import HOSTED_HONESTY


def write_run_manifest(hosted_dir: Path, payload: dict[str, Any]) -> Path:
    """Write run_manifest.json under hosted_dir (caller redacts secrets)."""
    hosted_dir.mkdir(parents=True, exist_ok=True)
    path = hosted_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_hosted_report(
    hosted_dir: Path,
    *,
    session_id: str,
    plateau_reason: str,
    operator_justify: str,
    view_paths: list[str],
    provider: str,
    provider_task_id: str | None,
    mesh_id: str | None,
    triage_summary: dict[str, Any] | None,
    honesty: str = HOSTED_HONESTY,
    extra_lines: list[str] | None = None,
) -> Path:
    """Write hosted_report.md with DoD-required fields."""
    hosted_dir.mkdir(parents=True, exist_ok=True)
    path = hosted_dir / "hosted_report.md"

    stats_bits: list[str] = []
    if triage_summary:
        stats = triage_summary.get("stats") or {}
        if isinstance(stats, dict):
            if "faces" in stats:
                stats_bits.append(f"faces={stats['faces']}")
            if "components" in stats:
                stats_bits.append(f"components={stats['components']}")
            bbox = stats.get("bbox_diagonal")
            if bbox is not None:
                stats_bits.append(f"bbox_diagonal={bbox}")

    lines = [
        "# Hosted multi-view fallback report",
        "",
        f"- session_id: `{session_id}`",
        f"- plateau reason: {plateau_reason}",
        f"- operator justify: {operator_justify}",
        f"- provider: `{provider}`",
        f"- provider_task_id: `{provider_task_id or 'n/a'}`",
        f"- mesh_id: `{mesh_id or 'n/a'}`",
        f"- triage: {', '.join(stats_bits) if stats_bits else 'n/a'}",
        "",
        "## Reference images",
        "",
    ]
    for vp in view_paths:
        lines.append(f"- `{vp}`")
    lines.extend(
        [
            "",
            "## Honesty",
            "",
            honesty,
            "",
        ]
    )
    if extra_lines:
        lines.extend(extra_lines)
        if not lines[-1].endswith("\n") and lines[-1] != "":
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
