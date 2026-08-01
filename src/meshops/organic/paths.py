"""SessionPaths — organic authoring sessions are NOT content-hash jobs (B6).

Layout::

    work/<session_id>/organic/
      manifest.json, prompt.md, style_notes.md, session_report.md
      refs/, passes/, plateau.json, final.stl, finalize.json
      failed_<pass_id>/   # failed pass attempts (siblings of passes/, not under it)

Failed passes are renamed from ``passes/<pass_id>/`` to ``organic/failed_<pass_id>/``
(e.g. ``organic/failed_p001_simple_bust/``). They are siblings of ``passes/``, never
nested under it, so they cannot be mistaken for successful manifest.passes entries.

Never add organic/ via ensure_job_layout / JobPaths for every triage job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SessionPaths:
    """Resolved paths under work/<session_id>/organic/.

    Successful / in-progress pass dirs live at ``organic/passes/<pass_id>/``.
    Failed attempts are renamed to ``organic/failed_<pass_id>/`` (siblings of
    ``passes/``, not children of it) so orphan work dirs never pollute success layout.
    """

    work_root: Path
    session_id: str

    @property
    def session_dir(self) -> Path:
        return self.work_root / self.session_id

    @property
    def organic_dir(self) -> Path:
        return self.session_dir / "organic"

    @property
    def manifest_path(self) -> Path:
        return self.organic_dir / "manifest.json"

    @property
    def prompt_md(self) -> Path:
        return self.organic_dir / "prompt.md"

    @property
    def style_notes_md(self) -> Path:
        return self.organic_dir / "style_notes.md"

    @property
    def session_report_md(self) -> Path:
        return self.organic_dir / "session_report.md"

    @property
    def refs_dir(self) -> Path:
        return self.organic_dir / "refs"

    @property
    def passes_dir(self) -> Path:
        return self.organic_dir / "passes"

    @property
    def plateau_json(self) -> Path:
        return self.organic_dir / "plateau.json"

    @property
    def final_stl(self) -> Path:
        return self.organic_dir / "final.stl"

    @property
    def finalize_json(self) -> Path:
        return self.organic_dir / "finalize.json"

    def pass_dir(self, name: str) -> Path:
        """Successful (or in-progress) pass directory under passes/."""
        return self.passes_dir / name

    def failed_pass_dir(self, name: str) -> Path:
        """Failed attempt directory at organic/failed_<name>/ (sibling of passes/)."""
        return self.organic_dir / f"failed_{name}"

    def ensure_layout(self) -> None:
        """Create session organic skeleton (does not touch JobPaths)."""
        self.organic_dir.mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(exist_ok=True)
        self.passes_dir.mkdir(exist_ok=True)
