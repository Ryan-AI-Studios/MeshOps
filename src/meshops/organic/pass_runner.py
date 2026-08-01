"""Organic pass runner — Blender recipe subprocess + evidence (track 0006).

shell=False; capture stdout/stderr; returncode + duration_s on pass.json (B8).
Only append manifest.passes on ok=True; failures → failed_pNNN_<recipe>/.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from meshops.escalate.discover import find_blender
from meshops.escalate.errors import EscalateError
from meshops.escalate.version import require_blender_52
from meshops.organic.errors import OrganicError
from meshops.organic.evidence import render_pass_views, views_complete
from meshops.organic.models import PassResult
from meshops.organic.paths import SessionPaths
from meshops.organic.recipes import parse_params
from meshops.organic.report import write_session_report
from meshops.organic.session import load_session, require_not_finalized, save_manifest

DEFAULT_TIMEOUT_S = 300.0
ENV_ORGANIC_TIMEOUT = "MESHOPS_ORGANIC_TIMEOUT_S"
RECIPE_SCRIPT = Path(__file__).resolve().parent / "scripts" / "run_recipe.py"

_TRACE_RE = re.compile(r"(Exception|Error|RuntimeError|AttributeError|TypeError)\b")
_PASS_TOKEN_RE = re.compile(r"^p\d{3}(_[a-z0-9_]+)?$", re.IGNORECASE)
_OK_RE = re.compile(r"meshops_organic_ok\s+path=(.+)")


def organic_timeout_s() -> float:
    """MESHOPS_ORGANIC_TIMEOUT_S default 300; high-res may need 600+ (B9)."""
    raw = os.environ.get(ENV_ORGANIC_TIMEOUT, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_S
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _map_escalate(exc: EscalateError) -> OrganicError:
    code = exc.code
    if code == "blender_missing":
        return OrganicError(str(exc), code="blender_not_found", details=exc.details)
    if code == "blender_version":
        return OrganicError(str(exc), code="blender_version", details=exc.details)
    if code == "timeout":
        return OrganicError(str(exc), code="blender_timeout", details=exc.details)
    return OrganicError(str(exc), code="blender_failed", details=exc.details)


def extract_trace_messages(stdout: str, stderr: str) -> list[str]:
    """Pull Exception/RuntimeError lines from Blender logs."""
    messages: list[str] = []
    for line in (stderr or "").splitlines() + (stdout or "").splitlines():
        if _TRACE_RE.search(line):
            cleaned = line.strip()
            if cleaned and cleaned not in messages:
                messages.append(cleaned)
    return messages[:40]


def resolve_source_stl(
    source: str,
    *,
    paths: SessionPaths,
    pass_names: list[str],
) -> Path:
    """Resolve from_mesh source_stl: absolute path or p001 / p001_recipe token (B7)."""
    token = source.strip()
    if _PASS_TOKEN_RE.match(token):
        # Prefer exact dir name among successful passes; bare p001 → unique prefix
        exact = [n for n in pass_names if n == token or n.lower() == token.lower()]
        if exact:
            mesh = paths.pass_dir(exact[0]) / "mesh.stl"
        else:
            prefix = token.split("_")[0].lower()  # p001
            matches = [n for n in pass_names if n.lower().startswith(prefix)]
            if len(matches) == 0:
                raise OrganicError(
                    f"from_mesh source token {token!r} matches no successful pass",
                    code="invalid_params",
                    details={"source_stl": token, "passes": pass_names},
                )
            if len(matches) > 1 and "_" not in token:
                # bare p001 with unique prefix
                prefixed = [n for n in matches if n.lower().startswith(prefix + "_")]
                if len(prefixed) == 1:
                    matches = prefixed
                elif len(matches) > 1:
                    raise OrganicError(
                        f"from_mesh source token {token!r} is ambiguous: {matches}",
                        code="invalid_params",
                        details={"source_stl": token, "matches": matches},
                    )
            mesh = paths.pass_dir(matches[0]) / "mesh.stl"
        if not mesh.is_file() or mesh.stat().st_size <= 0:
            raise OrganicError(
                f"pass mesh missing or empty for token {token!r}: {mesh}",
                code="pass_no_mesh",
                details={"source_stl": token, "path": str(mesh)},
            )
        return mesh.resolve()

    # Absolute path only (no session-relative)
    p = Path(token).expanduser()
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    if not resolved.is_absolute():
        raise OrganicError(
            "from_mesh source_stl must be absolute path or pass token (p001); "
            f"got relative {token!r}",
            code="invalid_params",
            details={"source_stl": token},
        )
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise OrganicError(
            f"from_mesh source_stl not found or empty: {resolved}",
            code="invalid_params",
            details={"source_stl": str(resolved)},
        )
    if resolved.suffix.lower() != ".stl":
        raise OrganicError(
            f"from_mesh source_stl must be a .stl file: {resolved}",
            code="invalid_params",
            details={"source_stl": str(resolved)},
        )
    return resolved


def _next_pass_index(manifest_passes: list[str], organic_dir: Path) -> int:
    """Allocate next pNNN index from successful + failed dir names."""
    idxs: list[int] = []
    for name in manifest_passes:
        m = re.match(r"^p(\d{3})", name)
        if m:
            idxs.append(int(m.group(1)))
    # Also scan failed_*
    for child in organic_dir.iterdir() if organic_dir.is_dir() else []:
        if child.name.startswith("failed_p"):
            m = re.match(r"^failed_p(\d{3})", child.name)
            if m:
                idxs.append(int(m.group(1)))
        if child.is_dir() and child.parent.name == "passes":
            m = re.match(r"^p(\d{3})", child.name)
            if m:
                idxs.append(int(m.group(1)))
    # Scan passes dir
    passes = organic_dir / "passes"
    if passes.is_dir():
        for child in passes.iterdir():
            m = re.match(r"^p(\d{3})", child.name)
            if m:
                idxs.append(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 1


def _atomic_promote_mesh(partial: Path, final: Path) -> None:
    """Rename mesh.stl.partial → mesh.stl only when complete (B9)."""
    if not partial.is_file() or partial.stat().st_size <= 0:
        raise OrganicError(
            f"Blender did not produce a non-empty mesh: {partial}",
            code="pass_no_mesh",
            details={"partial": str(partial)},
        )
    if final.exists():
        final.unlink()
    partial.replace(final)


def _optional_diff(
    *,
    prior_mesh: Path | None,
    cand_mesh: Path,
    diff_dir: Path,
) -> None:
    """Non-blocking pass-to-pass diff via render_diff_views."""
    if prior_mesh is None or not prior_mesh.is_file():
        return
    try:
        from meshops.recipes.diff_views import render_diff_views

        render_diff_views(
            baseline_mesh=prior_mesh,
            candidate_mesh=cand_mesh,
            views_dir=diff_dir,
            camera_names=["front", "three_quarter"],
        )
    except Exception:
        # Optional — never fail the pass
        pass


def run_pass(
    session_id: str,
    *,
    recipe: str | None = None,
    params: dict[str, Any] | None = None,
    work_root: Path | str = "work",
    force_stub_views: bool = False,
) -> PassResult:
    """Run one organic recipe pass; append to manifest only on full success."""
    paths, manifest = load_session(session_id, work_root=work_root)
    require_not_finalized(manifest)

    max_passes = manifest.max_passes
    if len(manifest.passes) >= max_passes:
        raise OrganicError(
            f"max_passes ({max_passes}) exceeded; mark plateau or finalize",
            code="max_passes_exceeded",
            details={
                "pass_count": len(manifest.passes),
                "max_passes": max_passes,
            },
        )

    recipe_id = recipe or manifest.default_recipe
    parsed = parse_params(recipe_id, params)

    # Resolve from_mesh after finalized check, before Blender (B7 / C5)
    blender_params = parsed.model_dump(mode="json")
    if parsed.recipe == "from_mesh":
        assert parsed.source_stl is not None
        # Prefer pass tokens over final.stl while active
        if parsed.source_stl.strip().lower().endswith("final.stl"):
            final_cand = Path(parsed.source_stl).expanduser().resolve()
            if final_cand == paths.final_stl.resolve():
                raise OrganicError(
                    "from_mesh: use a pass token (p001) rather than organic/final.stl",
                    code="invalid_params",
                    details={"source_stl": parsed.source_stl},
                )
        resolved = resolve_source_stl(
            parsed.source_stl,
            paths=paths,
            pass_names=list(manifest.passes),
        )
        blender_params["source_stl"] = str(resolved)

    idx = _next_pass_index(manifest.passes, paths.organic_dir)
    pass_id = f"p{idx:03d}_{parsed.recipe}"
    pass_dir = paths.pass_dir(pass_id)
    if pass_dir.exists():
        shutil.rmtree(pass_dir)
    pass_dir.mkdir(parents=True, exist_ok=True)
    views_dir = pass_dir / "views"
    views_dir.mkdir(exist_ok=True)

    params_json = pass_dir / "params.json"
    params_json.write_text(json.dumps(blender_params, indent=2), encoding="utf-8")

    messages: list[str] = []
    returncode: int | None = None
    duration_s: float | None = None
    blender_version: str | None = None
    mesh_final = pass_dir / "mesh.stl"
    mesh_partial = pass_dir / "mesh.stl.partial"

    try:
        try:
            blender = find_blender(require=True)
            assert blender is not None
            blender_version = require_blender_52(blender)
        except EscalateError as exc:
            raise _map_escalate(exc) from exc

        if not RECIPE_SCRIPT.is_file():
            raise OrganicError(
                f"recipe script missing: {RECIPE_SCRIPT}",
                code="blender_failed",
            )

        cmd = [
            str(blender),
            "-b",
            "-P",
            str(RECIPE_SCRIPT.resolve()),
            "--",
            "--out",
            str(pass_dir.resolve()),
            "--params-json",
            str(params_json.resolve()),
        ]
        timeout_s = organic_timeout_s()
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration_s = time.perf_counter() - t0
            # Do not promote partial
            if mesh_partial.is_file():
                mesh_partial.unlink(missing_ok=True)  # type: ignore[call-arg]
            stdout = (exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (exc.stderr or b"") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            (pass_dir / "blender_stdout.log").write_text(str(stdout), encoding="utf-8")
            (pass_dir / "blender_stderr.log").write_text(str(stderr), encoding="utf-8")
            raise OrganicError(
                f"Blender timed out after {timeout_s}s "
                f"(high-res may need MESHOPS_ORGANIC_TIMEOUT_S=600+)",
                code="blender_timeout",
                details={"timeout_s": timeout_s, "pass_id": pass_id},
            ) from exc

        duration_s = time.perf_counter() - t0
        returncode = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        (pass_dir / "blender_stdout.log").write_text(stdout, encoding="utf-8")
        (pass_dir / "blender_stderr.log").write_text(stderr, encoding="utf-8")
        messages.extend(extract_trace_messages(stdout, stderr))

        if returncode != 0:
            raise OrganicError(
                f"Blender recipe failed (exit {returncode}): {(stderr or stdout)[:400]}",
                code="blender_failed",
                details={"returncode": returncode, "pass_id": pass_id},
            )

        # Prefer explicit partial; also accept script writing mesh.stl.partial
        if mesh_partial.is_file():
            _atomic_promote_mesh(mesh_partial, mesh_final)
        elif mesh_final.is_file() and mesh_final.stat().st_size > 0:
            pass  # script may have already renamed
        else:
            # Check ok line path
            m = _OK_RE.search(stdout)
            if m:
                candidate = Path(m.group(1).strip())
                if candidate.is_file() and candidate.stat().st_size > 0:
                    if candidate != mesh_final:
                        shutil.copy2(candidate, mesh_final)
                else:
                    raise OrganicError(
                        "Blender reported ok but mesh path missing",
                        code="pass_no_mesh",
                    )
            else:
                raise OrganicError(
                    "Blender finished without mesh.stl",
                    code="pass_no_mesh",
                    details={"pass_dir": str(pass_dir)},
                )

        if not mesh_final.is_file() or mesh_final.stat().st_size <= 0:
            raise OrganicError(
                f"mesh.stl missing or empty after pass: {mesh_final}",
                code="pass_no_mesh",
            )

        # Evidence (required for success)
        view_paths, view_kind, view_notes = render_pass_views(
            mesh_final,
            views_dir,
            force_stub=force_stub_views,
        )
        messages.extend(view_notes)
        if not views_complete(view_paths):
            raise OrganicError(
                "required pass views missing (front, left, three_quarter, three_quarter_depth)",
                code="pass_no_views",
                details={"view_paths": view_paths},
            )

        # Optional non-blocking diff vs prior successful pass
        prior_mesh: Path | None = None
        if manifest.passes:
            prior = paths.pass_dir(manifest.passes[-1]) / "mesh.stl"
            if prior.is_file():
                prior_mesh = prior
        _optional_diff(
            prior_mesh=prior_mesh,
            cand_mesh=mesh_final,
            diff_dir=pass_dir / "diff",
        )

        result = PassResult(
            ok=True,
            pass_id=pass_id,
            recipe=parsed.recipe,
            mesh_path=mesh_final,
            view_paths=view_paths,
            view_kind=view_kind,
            blender_version=blender_version,
            returncode=returncode,
            duration_s=duration_s,
            error_code=None,
            messages=messages,
            params=blender_params,
            scale_mm=parsed.scale_mm,
        )
        _write_pass_json(pass_dir, result)

        manifest.passes.append(pass_id)
        manifest.blender_version = blender_version
        if view_kind == "stub" and "views_stub_used" not in manifest.notes:
            manifest.notes.append("views_stub_used")
        save_manifest(paths, manifest)
        write_session_report(paths, manifest)
        return result

    except OrganicError as exc:
        return _fail_pass(
            paths=paths,
            manifest=manifest,
            pass_id=pass_id,
            pass_dir=pass_dir,
            recipe=parsed.recipe,
            params=blender_params,
            error=exc,
            returncode=returncode,
            duration_s=duration_s,
            blender_version=blender_version,
            messages=messages,
            scale_mm=parsed.scale_mm,
        )
    except Exception as exc:
        # Unexpected failures must still rename work dir so it never looks successful
        wrapped = OrganicError(
            f"unexpected pass failure: {exc}",
            code="blender_failed",
            details={"cause": type(exc).__name__, "pass_id": pass_id},
        )
        try:
            _fail_pass(
                paths=paths,
                manifest=manifest,
                pass_id=pass_id,
                pass_dir=pass_dir,
                recipe=parsed.recipe,
                params=blender_params,
                error=wrapped,
                returncode=returncode,
                duration_s=duration_s,
                blender_version=blender_version,
                messages=messages,
                scale_mm=parsed.scale_mm,
            )
        except OrganicError as fail_exc:
            raise fail_exc from exc
        except Exception:
            # Best-effort rename/notes failed — still surface as OrganicError
            raise wrapped from exc
        raise wrapped from exc  # pragma: no cover — _fail_pass always raises


def _write_pass_json(pass_dir: Path, result: PassResult) -> None:
    data = result.model_dump(mode="json")
    # Ensure B8 keys present
    data.setdefault("returncode", result.returncode)
    data.setdefault("duration_s", result.duration_s)
    (pass_dir / "pass.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _fail_pass(
    *,
    paths: SessionPaths,
    manifest: Any,
    pass_id: str,
    pass_dir: Path,
    recipe: str,
    params: dict[str, Any],
    error: OrganicError,
    returncode: int | None,
    duration_s: float | None,
    blender_version: str | None,
    messages: list[str],
    scale_mm: float | None,
) -> PassResult:
    """Move pass dir to failed_*; do not append to manifest.passes; re-raise."""
    msgs = list(messages)
    msgs.append(str(error))
    result = PassResult(
        ok=False,
        pass_id=pass_id,
        recipe=recipe,
        mesh_path=None,
        view_paths={},
        view_kind=None,
        blender_version=blender_version,
        returncode=returncode,
        duration_s=duration_s,
        error_code=str(error.code),
        messages=msgs,
        params=params,
        scale_mm=scale_mm,
    )
    try:
        if pass_dir.is_dir():
            _write_pass_json(pass_dir, result)
            failed = paths.failed_pass_dir(pass_id)
            if failed.exists():
                shutil.rmtree(failed)
            pass_dir.rename(failed)
    except OSError:
        pass

    note = f"failed_{pass_id}: {error.code}: {error}"
    if note not in manifest.notes:
        manifest.notes.append(note)
    save_manifest(paths, manifest)
    raise error
