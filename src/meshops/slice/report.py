"""Human/LLM ``slice_report.md`` writer (always on completed run, even fail)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meshops.slice.models import SliceRunResult


def write_slice_report(
    run_dir: Path | str,
    result: SliceRunResult,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``slice_report.md`` under *run_dir*. Returns path written."""
    dest = Path(run_dir) / "slice_report.md"
    dest.parent.mkdir(parents=True, exist_ok=True)

    accept = result.accept
    metrics = dict(result.metrics)
    if accept is not None:
        metrics = {**metrics, **accept.metrics}

    lines: list[str] = [
        "# Slice report",
        "",
        f"- **run_id:** `{result.run_id}`",
        f"- **status:** `{result.status}`",
        f"- **error_code:** `{result.error_code or '—'}`",
        f"- **candidate:** `{result.candidate_path}`",
        f"- **output_3mf:** `{result.output_3mf or '—'}`",
        f"- **orca_path:** `{result.orca_path or '—'}`",
        f"- **orca_version:** `{result.orca_version or '—'}`",
        f"- **plate_count:** {result.plate_count}",
        f"- **started_at:** {result.started_at or '—'}",
        f"- **finished_at:** {result.finished_at or '—'}",
        "",
        "## Printability",
        "",
    ]

    if accept is not None:
        lines.extend(
            [
                f"- **accept.status:** `{accept.status}`",
                (
                    f"- **print_time_s:** "
                    f"{accept.print_time_s if accept.print_time_s is not None else '—'}"
                ),
                f"- **filament_used_cm3 (PLA≈1.24):** "
                f"{accept.filament_used_cm3 if accept.filament_used_cm3 is not None else '—'}",
                f"- **bed_overflow:** {accept.bed_overflow}",
                f"- **accept.error_code:** `{accept.error_code or '—'}`",
            ]
        )
    else:
        lines.append("- accept result: *(none)*")

    dens_note = metrics.get("slice.density_note", "PLA v1 approximate 1.24 g/cm3")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- **support_used:** {metrics.get('slice.support_used', '—')}",
            f"- **filament_used_g:** {metrics.get('slice.filament_used_g', '—')}",
            f"- **filament_used_m:** {metrics.get('slice.filament_used_m', '—')}",
            f"- **filament_ratio:** {metrics.get('slice.filament_ratio', '—')}",
            f"- **mesh_volume_available:** {metrics.get('slice.mesh_volume_available', '—')}",
            f"- **mesh_volume_cm3:** {metrics.get('slice.mesh_volume_cm3', '—')}",
            f"- **suggest_reorient:** {metrics.get('slice.suggest_reorient', '—')}",
            f"- **warning_max_level:** {metrics.get('slice.warning_max_level', '—')}",
            f"- **parse_source:** {metrics.get('slice.parse_source', '—')}",
            f"- **density_note:** {dens_note}",
            "",
            "## Profiles",
            "",
        ]
    )

    pp = result.profile_paths
    if pp is not None:
        lines.extend(
            [
                f"- **profile:** `{pp.profile_name}`",
                f"- **machine:** `{pp.machine}`",
                f"- **process:** `{pp.process}`",
                f"- **filament:** `{pp.filament}`",
                f"- **datadir:** `{pp.datadir or '—'}`",
            ]
        )
    else:
        lines.append("- *(no profile paths)*")

    lines.extend(["", "## Messages", ""])
    msgs = list(result.messages)
    if accept is not None:
        for m in accept.messages:
            if m not in msgs:
                msgs.append(m)
    if msgs:
        for m in msgs:
            lines.append(f"- {m}")
    else:
        lines.append("- *(none)*")

    if result.argv:
        lines.extend(["", "## Argv", "", "```", " ".join(result.argv), "```"])

    if extra:
        lines.extend(["", "## Extra", ""])
        for k, v in extra.items():
            lines.append(f"- **{k}:** {v}")

    lines.extend(
        [
            "",
            "---",
            "",
            "_Honesty: slice pass means Orca produced G-code metadata, bed OK, "
            "filament non-zero, anomaly under threshold — **not** artistic quality (N6). "
            "`filament_used_cm3` is PLA-default approximate (1.24 g/cm³) in v1._",
            "",
        ]
    )

    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest
