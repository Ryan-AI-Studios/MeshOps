"""Write bench_results.json + markdown under work/bench (or --work-root)."""

from __future__ import annotations

from pathlib import Path

from meshops.bench.models import Envelope


def default_results_dir(work_root: Path | str | None = None) -> Path:
    """Resolve results directory (work/bench by default)."""
    if work_root is None:
        return Path("work") / "bench"
    root = Path(work_root)
    # If caller already pointed at …/bench, use it; else nest bench under it.
    if root.name.lower() == "bench":
        return root
    return root / "bench"


def write_results(
    envelope: Envelope,
    *,
    work_root: Path | str | None = None,
    basename: str = "bench_results",
) -> tuple[Path, Path]:
    """Write JSON + markdown; return (json_path, md_path)."""
    out_dir = default_results_dir(work_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{basename}.json"
    md_path = out_dir / f"{basename}.md"

    json_path.write_text(
        envelope.model_dump_json(indent=2),
        encoding="utf-8",
    )
    md_path.write_text(envelope_to_markdown(envelope), encoding="utf-8")
    return json_path, md_path


def find_latest_results(search_root: Path | str | None = None) -> Path | None:
    """Newest ``bench_results.json`` by mtime under search_root (default work/bench)."""
    root = Path(search_root) if search_root is not None else Path("work") / "bench"
    if not root.is_dir():
        # Also search work/ if caller passed work/
        alt = root / "bench" if root.name.lower() != "bench" else None
        if alt is not None and alt.is_dir():
            root = alt
        else:
            return None

    candidates = list(root.rglob("bench_results.json"))
    if not candidates:
        # Accept basename variants written in same tree
        candidates = [p for p in root.rglob("*.json") if p.name.startswith("bench_results")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_envelope(path: Path | str) -> Envelope:
    """Load Envelope from JSON path."""
    text = Path(path).read_text(encoding="utf-8")
    return Envelope.model_validate_json(text)


def envelope_to_markdown(envelope: Envelope) -> str:
    """Human-readable envelope table for docs/work."""
    lines: list[str] = [
        "# MeshOps Hardening Envelope",
        "",
        f"- **schema_version:** {envelope.schema_version}",
        f"- **created_at:** {envelope.created_at}",
        "",
        "## Method",
        "",
        f"- warmup: **{envelope.method.warmup}**",
        f"- timed_iters: **{envelope.method.timed_iters}**",
        f"- aggregate: **{envelope.method.aggregate}**",
        f"- cameras: `{', '.join(envelope.method.cameras)}`",
        f"- face_tolerance: ±{envelope.method.face_tolerance_frac * 100:.0f}%",
        f"- notes: {envelope.method.notes}",
        "",
        "## Host",
        "",
        f"- OS: {envelope.host.os}",
        f"- Python: {envelope.host.python_version}",
        f"- CPU: {envelope.host.cpu or 'n/a'}",
    ]
    total_ram = envelope.host.total_ram_mb
    avail_ram = envelope.host.available_ram_mb
    lines.append(f"- total_ram_mb: {total_ram if total_ram is not None else 'n/a'}")
    lines.append(f"- available_ram_mb: {avail_ram if avail_ram is not None else 'n/a'}")
    lines.extend(["", "### Dep versions", ""])
    if envelope.host.deps:
        for k, v in sorted(envelope.host.deps.items()):
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- _(none recorded)_")

    header = (
        "| label | status | target | actual | verts | "
        "ingest_s | triage_s | render_s | rss_mb | notes |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
    lines.extend(["", "## Cases", "", header, sep])
    for c in envelope.cases:
        notes = c.skipped_reason or c.error_message or c.error_code or ""
        notes = notes.replace("|", "/")[:60]
        row = (
            f"| {c.label} | {c.status} | {c.target_faces} | {c.actual_faces} | "
            f"{c.verts} | {_fmt(c.ingest_s)} | {_fmt(c.triage_s)} | "
            f"{_fmt(c.render_s)} | {_fmt(c.rss_peak_mb)} | {notes} |"
        )
        lines.append(row)

    if envelope.notes:
        lines.extend(["", "## Notes", ""])
        for n in envelope.notes:
            lines.append(f"- {n}")

    lines.append("")
    return "\n".join(lines)


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"
