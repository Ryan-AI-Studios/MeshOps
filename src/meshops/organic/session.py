"""Organic session create / load (SessionPaths — not JobPaths)."""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from meshops.organic.errors import OrganicError
from meshops.organic.models import HONESTY_NOTE, OrganicManifest
from meshops.organic.paths import SessionPaths
from meshops.organic.report import write_session_report

SESSION_ID_RE = re.compile(r"^o[0-9a-f]{11}$")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def compute_session_id(
    prompt: str,
    *,
    created_at: str,
    seed: str | None = None,
) -> str:
    """``o`` + 11 hex from sha256(prompt + created_at + optional_seed)."""
    material = prompt.encode("utf-8") + b"\n" + created_at.encode("utf-8") + b"\n"
    if seed:
        material += seed.encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:11]
    return f"o{digest}"


def create_session(
    prompt: str,
    *,
    style_notes: str = "",
    refs: list[Path | str] | None = None,
    default_recipe: str = "simple_bust",
    session_id: str | None = None,
    work_root: Path | str = "work",
    max_passes: int = 8,
    seed: str | None = None,
) -> OrganicManifest:
    """Create a new organic authoring session under SessionPaths."""
    prompt_clean = (prompt or "").strip()
    if not prompt_clean:
        raise OrganicError(
            "prompt must be non-empty",
            code="invalid_params",
            details={"field": "prompt"},
        )

    work_root_p = Path(work_root)
    work_root_p.mkdir(parents=True, exist_ok=True)
    created_at = _now_iso()

    if session_id is not None:
        sid = session_id.strip().lower()
        if not SESSION_ID_RE.match(sid):
            raise OrganicError(
                f"session_id must match ^o[0-9a-f]{{11}}$ (got {session_id!r})",
                code="invalid_params",
                details={"session_id": session_id},
            )
    else:
        sid = compute_session_id(prompt_clean, created_at=created_at, seed=seed)

    paths = SessionPaths(work_root=work_root_p, session_id=sid)
    if paths.manifest_path.is_file():
        raise OrganicError(
            f"session already exists: {sid}",
            code="invalid_params",
            details={"session_id": sid, "path": str(paths.organic_dir)},
        )

    paths.ensure_layout()

    # Copy refs into organic/refs/
    ref_paths: list[str] = []
    for i, ref in enumerate(refs or []):
        src = Path(ref)
        if not src.is_file():
            raise OrganicError(
                f"ref not found: {src}",
                code="invalid_params",
                details={"ref": str(src)},
            )
        dest_name = f"ref_{i:02d}_{src.name}"
        dest = paths.refs_dir / dest_name
        shutil.copy2(src, dest)
        ref_paths.append(str(dest))

    paths.prompt_md.write_text(prompt_clean + "\n", encoding="utf-8")
    paths.style_notes_md.write_text((style_notes or "") + "\n", encoding="utf-8")

    manifest = OrganicManifest(
        session_id=sid,
        prompt=prompt_clean,
        style_notes=style_notes or "",
        ref_paths=ref_paths,
        default_recipe=default_recipe,
        status="active",
        passes=[],
        created_at=created_at,
        updated_at=created_at,
        blender_version=None,
        final_mesh_id=None,
        notes=[HONESTY_NOTE],
        max_passes=max_passes,
    )
    save_manifest(paths, manifest)
    write_session_report(paths, manifest)
    return manifest


def load_session(
    session_id: str,
    *,
    work_root: Path | str = "work",
) -> tuple[SessionPaths, OrganicManifest]:
    """Load session paths + manifest; raise session_not_found if missing."""
    work_root_p = Path(work_root)
    sid = session_id.strip().lower()
    paths = SessionPaths(work_root=work_root_p, session_id=sid)
    if not paths.manifest_path.is_file():
        raise OrganicError(
            f"session not found: {sid}",
            code="session_not_found",
            details={"session_id": sid, "path": str(paths.manifest_path)},
        )
    raw = paths.manifest_path.read_text(encoding="utf-8")
    manifest = OrganicManifest.model_validate_json(raw)
    return paths, manifest


def save_manifest(paths: SessionPaths, manifest: OrganicManifest) -> None:
    """Write manifest.json (UTF-8, indented)."""
    paths.organic_dir.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = _now_iso()
    paths.manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def require_not_finalized(manifest: OrganicManifest) -> None:
    """Raise session_finalized when status is finalized."""
    if manifest.status == "finalized":
        raise OrganicError(
            f"session {manifest.session_id} is already finalized",
            code="session_finalized",
            details={"session_id": manifest.session_id, "status": manifest.status},
        )
