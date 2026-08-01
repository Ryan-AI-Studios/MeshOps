"""Hosted multi-view fallback orchestration (gate → encode → provider → ingest)."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meshops.acceptance.honesty import HONESTY_MESSAGE
from meshops.acceptance.pack import accept_candidate
from meshops.guards.policy import GuardPolicy
from meshops.hosted.convert import glb_to_stl
from meshops.hosted.encode import path_to_data_uri
from meshops.hosted.errors import HostedError
from meshops.hosted.gate import validate_plateau_gate
from meshops.hosted.honesty import HOSTED_HONESTY
from meshops.hosted.models import HOSTED_SCHEMA_VERSION, HostedRunResult, Justification
from meshops.hosted.providers import DEFAULT_PROVIDER_NAME, get_provider
from meshops.hosted.providers.base import HostedProvider
from meshops.hosted.report import write_hosted_report, write_run_manifest
from meshops.hosted.views import ViewsFrom, collect_view_paths
from meshops.ingest.pipeline import ingest_stl
from meshops.jobstore.paths import JobPaths, ensure_job_layout
from meshops.organic.plateau import FILLER_REASONS, REASON_MIN_LEN
from meshops.triage.orchestrate import mesh_triage

# Env defaults (binding)
DEFAULT_POLL_INTERVAL_S = 5.0
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_MAX_HTTP_RETRIES = 3

_TERMINAL_OK = "SUCCEEDED"
_TERMINAL_FAIL = frozenset({"FAILED", "CANCELED"})


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_plateau_path(
    *,
    plateau: Path | str | None = None,
    session_id: str | None = None,
    work_root: Path | str = "work",
) -> Path:
    """Resolve plateau.json from --plateau or session_id + work_root."""
    if plateau is not None:
        return Path(plateau)
    if session_id:
        return Path(work_root) / session_id / "organic" / "plateau.json"
    raise HostedError(
        "hosted run requires --plateau and/or --session-id",
        code="plateau_missing",
        details={},
    )


def validate_operator_justify(justify: str) -> str:
    """Same min-length + filler rules as plateau reason (import REASON_MIN_LEN)."""
    if justify is None:
        raise HostedError(
            "operator --justify is required",
            code="justify_invalid",
            details={},
        )
    cleaned = justify.strip()
    if not cleaned:
        raise HostedError(
            "operator --justify is required (empty after strip)",
            code="justify_invalid",
            details={},
        )
    if len(cleaned) < REASON_MIN_LEN:
        raise HostedError(
            f"operator --justify must be ≥{REASON_MIN_LEN} chars after strip (got {len(cleaned)})",
            code="justify_invalid",
            details={"len": len(cleaned), "justify": cleaned},
        )
    if cleaned.lower() in FILLER_REASONS:
        raise HostedError(
            f"operator --justify is a filler token: {cleaned!r}",
            code="justify_invalid",
            details={"justify": cleaned},
        )
    return cleaned


def resolve_api_key(provider_name: str) -> str | None:
    """Env precedence: provider-specific if set → MESHOPS_HOSTED_API_KEY.

    Never log full keys. Returns None when missing (caller raises for non-mock).
    """
    name = provider_name.strip().lower()
    if name == "meshy":
        specific = os.environ.get("MESHOPS_MESHY_API_KEY", "").strip()
        if specific:
            return specific
    elif name == "tripo":
        specific = os.environ.get("MESHOPS_TRIPO_API_KEY", "").strip()
        if specific:
            return specific
    primary = os.environ.get("MESHOPS_HOSTED_API_KEY", "").strip()
    return primary or None


def _build_provider(
    name: str,
    *,
    api_key: str | None,
    provider: HostedProvider | None = None,
    fixture_stl: Path | str | None = None,
    max_http_retries: int = DEFAULT_MAX_HTTP_RETRIES,
) -> HostedProvider:
    if provider is not None:
        return provider
    key = name.strip().lower()
    if key == "mock":
        kwargs: dict[str, Any] = {}
        if fixture_stl is not None:
            kwargs["fixture_stl"] = fixture_stl
        return get_provider("mock", **kwargs)
    if not api_key:
        raise HostedError(
            f"API key missing for provider {name!r} "
            f"(set MESHOPS_HOSTED_API_KEY or provider-specific override)",
            code="api_key_missing",
            details={"provider": name},
        )
    return get_provider(key, api_key=api_key, max_http_retries=max_http_retries)


def _poll_until_done(
    provider: HostedProvider,
    job_id: str,
    *,
    interval_s: float,
    timeout_s: float,
) -> Any:
    """Poll until SUCCEEDED / FAILED / CANCELED or MeshOps timeout."""
    deadline = time.monotonic() + timeout_s
    last_status = None
    while True:
        last_status = provider.poll(job_id)
        st = last_status.status
        if st == _TERMINAL_OK:
            return last_status
        if st in _TERMINAL_FAIL:
            te = last_status.task_error or {}
            raise HostedError(
                f"provider job {st}: {te.get('message') or last_status.message or st}",
                code="provider_failed",
                details={
                    "provider": getattr(provider, "name", None),
                    "job_id": job_id,
                    "status": st,
                    "type": te.get("type"),
                    "message": te.get("message") or last_status.message,
                    "doc_url": te.get("doc_url"),
                },
            )
        if time.monotonic() >= deadline:
            raise HostedError(
                f"hosted poll deadline exceeded ({timeout_s}s)",
                code="provider_timeout",
                details={
                    "provider": getattr(provider, "name", None),
                    "job_id": job_id,
                    "timeout_s": timeout_s,
                    "last_status": st,
                },
            )
        time.sleep(max(0.05, interval_s))


def run_hosted_fallback(
    *,
    session_id: str | None = None,
    work_root: Path | str = "work",
    plateau: Path | str | None = None,
    views_from: ViewsFrom = "latest",
    view_paths: Sequence[Path | str] | None = None,
    prompt: str = "",
    justify: str,
    provider: str = DEFAULT_PROVIDER_NAME,
    accept: bool = False,
    provider_instance: HostedProvider | None = None,
    fixture_stl: Path | str | None = None,
) -> HostedRunResult:
    """Gate → views → encode → submit → poll → download-now → ingest → triage.

    Hosted is **never** the default organic path — plateau gate first always.
    """
    work_root_p = Path(work_root)
    messages: list[str] = []
    provider_name = (provider or DEFAULT_PROVIDER_NAME).strip().lower()

    # 1) Resolve plateau path
    plateau_path = resolve_plateau_path(
        plateau=plateau,
        session_id=session_id,
        work_root=work_root_p,
    )

    # 2) Gate first (C1) — before any network / key check for closed gate tests
    record, gate_msgs = validate_plateau_gate(plateau_path, session_id=session_id)
    messages.extend(gate_msgs)
    sid = record.session_id

    # 3) Collect views + encode (explicit --view list wins when provided)
    explicit = [Path(p) for p in view_paths] if view_paths else None
    effective_from: ViewsFrom = "explicit" if explicit else views_from
    collected = collect_view_paths(
        plateau_path=plateau_path,
        views_from=effective_from,
        explicit_views=explicit,
    )
    view_path_strs = [str(p) for p in collected]
    data_uris = [path_to_data_uri(p) for p in collected]

    # 4) Justification
    op_justify = validate_operator_justify(justify)
    justification = Justification(
        plateau_reason=record.reason,
        operator_justify=op_justify,
    )

    # 5) API key for non-mock
    poll_interval = _env_float("MESHOPS_HOSTED_POLL_INTERVAL_S", DEFAULT_POLL_INTERVAL_S)
    poll_timeout = _env_float("MESHOPS_HOSTED_TIMEOUT_S", DEFAULT_TIMEOUT_S)
    max_retries = _env_int("MESHOPS_HOSTED_MAX_HTTP_RETRIES", DEFAULT_MAX_HTTP_RETRIES)

    api_key = resolve_api_key(provider_name) if provider_name != "mock" else None
    prov = _build_provider(
        provider_name,
        api_key=api_key,
        provider=provider_instance,
        fixture_stl=fixture_stl,
        max_http_retries=max_retries,
    )

    # Resolve prompt: explicit or inherit from session manifest
    run_prompt = prompt.strip() if prompt else ""
    if not run_prompt:
        manifest_path = Path(plateau_path).resolve().parent / "manifest.json"
        if manifest_path.is_file():
            try:
                import json

                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                run_prompt = str(data.get("prompt") or "")
            except (OSError, ValueError, TypeError):
                run_prompt = ""

    # 6) Submit → poll
    try:
        task_id = prov.submit_multiview(data_uris, run_prompt)
    except HostedError:
        raise
    except Exception as exc:
        raise HostedError(
            f"provider submit failed: {exc}",
            code="provider_failed",
            details={"provider": provider_name, "error": str(exc)},
        ) from exc

    try:
        _poll_until_done(
            prov,
            task_id,
            interval_s=poll_interval,
            timeout_s=poll_timeout,
        )
    except HostedError:
        raise

    # 7) Download immediately to **staging** (mesh_id unknown — C11)
    staging = Path(tempfile.mkdtemp(prefix="meshops-hosted-"))
    try:
        try:
            mesh_path = prov.download(task_id, staging)
        except HostedError:
            raise
        except Exception as exc:
            raise HostedError(
                f"provider download failed: {exc}",
                code="download_failed",
                details={"provider": provider_name, "task_id": task_id, "error": str(exc)},
            ) from exc

        # 8) Convert if GLB
        if mesh_path.suffix.lower() == ".glb":
            stl_out = staging / "model.stl"
            mesh_path = glb_to_stl(mesh_path, stl_out)
        elif mesh_path.suffix.lower() != ".stl":
            # Attempt convert path for other formats that trimesh might load
            if mesh_path.suffix.lower() in {".obj", ".ply", ".off"}:
                stl_out = staging / "model.stl"
                try:
                    mesh_path = glb_to_stl(mesh_path, stl_out)
                except HostedError:
                    raise HostedError(
                        f"unsupported mesh format from provider: {mesh_path.suffix}",
                        code="convert_failed",
                        details={"path": str(mesh_path)},
                    ) from None
            else:
                raise HostedError(
                    f"unsupported mesh format from provider: {mesh_path.suffix}",
                    code="convert_failed",
                    details={"path": str(mesh_path)},
                )

        # 9) Ingest as untrusted
        try:
            ing = ingest_stl(mesh_path, work_root=work_root_p)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HostedError(
                f"ingest failed: {exc}",
                code="ingest_failed",
                details={"error": str(exc), "stage": "ingest"},
            ) from exc
        except Exception as exc:
            raise HostedError(
                f"ingest failed: {exc}",
                code="ingest_failed",
                details={"error": str(exc), "stage": "ingest", "cause": type(exc).__name__},
            ) from exc

        mesh_id = ing.mesh_id
        job = JobPaths(work_root=work_root_p, mesh_id=mesh_id)
        ensure_job_layout(job)
        job.hosted_dir.mkdir(parents=True, exist_ok=True)

        # Copy provider_views from staging if present
        staging_thumbs = staging / "provider_views"
        if staging_thumbs.is_dir():
            dest_thumbs = job.hosted_dir / "provider_views"
            if dest_thumbs.exists():
                shutil.rmtree(dest_thumbs, ignore_errors=True)
            shutil.copytree(staging_thumbs, dest_thumbs)

        # 10) Triage always
        diagnostics: dict[str, Any] | None = None
        try:
            diag = mesh_triage(mesh_id, work_root=work_root_p)
            diagnostics = {
                "mesh_id": mesh_id,
                "stats": diag.stats.model_dump(mode="json"),
                "defect_hypotheses": [h.model_dump(mode="json") for h in diag.defect_hypotheses],
                "notes": list(diag.notes),
            }
        except Exception as exc:
            raise HostedError(
                f"triage failed on hosted mesh: {exc}",
                code="ingest_failed",
                details={"mesh_id": mesh_id, "stage": "triage", "error": str(exc)},
            ) from exc

        honesty = f"{HOSTED_HONESTY} {HONESTY_MESSAGE}"
        acceptance = None

        # 12) Optional 0011 accept
        if accept:
            try:
                acceptance = accept_candidate(
                    job.original_stl,
                    job.original_stl,
                    policy=GuardPolicy.for_sculpt(),
                    view_paths=[],
                    require_views=False,
                    allow_stubs=True,
                    view_notes=["hosted_fallback", "untrusted_generator"],
                )
                if acceptance.honesty_message:
                    honesty = f"{HOSTED_HONESTY} {acceptance.honesty_message}"
            except Exception as exc:
                messages.append(f"accept_candidate failed: {exc}")

        # 11) Artifacts under JobPaths.hosted_dir
        manifest_payload: dict[str, Any] = {
            "schema_version": HOSTED_SCHEMA_VERSION,
            "session_id": sid,
            "mesh_id": mesh_id,
            "job_dir": str(job.job_dir),
            "provider": provider_name,
            "provider_task_id": task_id,
            "justification": justification.model_dump(mode="json"),
            "view_paths": view_path_strs,
            "prompt": run_prompt,
            "created_at": _now_iso(),
            "honesty": honesty,
            "diagnostics": diagnostics,
            "accepted": bool(accept and acceptance is not None and acceptance.ok),
        }
        if acceptance is not None:
            manifest_payload["acceptance"] = acceptance.model_dump(mode="json")

        write_run_manifest(job.hosted_dir, manifest_payload)
        write_hosted_report(
            job.hosted_dir,
            session_id=sid,
            plateau_reason=justification.plateau_reason,
            operator_justify=justification.operator_justify,
            view_paths=view_path_strs,
            provider=provider_name,
            provider_task_id=task_id,
            mesh_id=mesh_id,
            triage_summary=diagnostics,
            honesty=honesty,
        )

        ok = True
        if accept and acceptance is not None:
            ok = bool(acceptance.ok)

        return HostedRunResult(
            ok=ok,
            session_id=sid,
            mesh_id=mesh_id,
            job_dir=str(job.job_dir),
            provider=provider_name,
            provider_task_id=task_id,
            justification=justification,
            view_paths=view_path_strs,
            diagnostics=diagnostics,
            acceptance=acceptance,
            honesty=honesty,
            error_code=None if ok else "ingest_failed",
            messages=messages,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
