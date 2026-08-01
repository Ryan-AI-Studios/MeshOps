"""session_report.md regeneration after successful passes / finalize."""

from __future__ import annotations

from meshops.organic.models import OrganicManifest
from meshops.organic.paths import SessionPaths


def write_session_report(
    paths: SessionPaths,
    manifest: OrganicManifest,
    *,
    extra_lines: list[str] | None = None,
) -> None:
    """Lightweight markdown report under organic/session_report.md."""
    lines = [
        f"# Organic session `{manifest.session_id}`",
        "",
        f"- **status:** {manifest.status}",
        f"- **recipe default:** {manifest.default_recipe}",
        f"- **created:** {manifest.created_at}",
        f"- **updated:** {manifest.updated_at}",
        f"- **passes:** {len(manifest.passes)}",
        f"- **blender:** {manifest.blender_version or 'n/a'}",
        f"- **final_mesh_id:** {manifest.final_mesh_id or 'n/a'}",
        "",
        "## Prompt",
        "",
        manifest.prompt.strip() or "_(empty)_",
        "",
        "## Style notes",
        "",
        (manifest.style_notes or "").strip() or "_(none)_",
        "",
        "## Successful passes",
        "",
    ]
    if manifest.passes:
        for p in manifest.passes:
            lines.append(f"- `{p}`")
    else:
        lines.append("_none yet_")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for n in manifest.notes:
        lines.append(f"- {n}")
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    lines.append("")
    lines.append("_Honesty: authored organic form — not a print-ready hero sculpt (N6)._")
    lines.append("")
    paths.session_report_md.write_text("\n".join(lines), encoding="utf-8")
