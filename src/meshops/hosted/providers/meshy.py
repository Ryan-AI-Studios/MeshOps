"""Meshy multi-image-to-3d adapter (v1 primary — docs freeze 2026-08-01)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from meshops.hosted.errors import HostedError
from meshops.hosted.models import ProviderJobStatus

MESHY_API_BASE = "https://api.meshy.ai"
MESHY_CREATE_PATH = "/openapi/v1/multi-image-to-3d"
MESHY_STATUS_PATH = "/openapi/v1/multi-image-to-3d/{id}"

# Geometry-first defaults for triage (N6-flagged fallback).
DEFAULT_TARGET_FORMATS: list[str] = ["stl"]
DEFAULT_SHOULD_TEXTURE = False
DEFAULT_MULTI_VIEW_THUMBNAILS = True
DEFAULT_AI_MODEL = "latest"

_TERMINAL_OK = "SUCCEEDED"
_TERMINAL_FAIL = frozenset({"FAILED", "CANCELED"})


def _normalize_status(raw: str | None) -> str:
    s = (raw or "").strip().upper()
    if s in {"PENDING", "IN_PROGRESS", "SUCCEEDED", "FAILED", "CANCELED"}:
        return s
    # Meshy may use lowercase or alternate spellings
    mapping = {
        "SUCCESS": "SUCCEEDED",
        "COMPLETED": "SUCCEEDED",
        "COMPLETE": "SUCCEEDED",
        "PROCESSING": "IN_PROGRESS",
        "RUNNING": "IN_PROGRESS",
        "QUEUED": "PENDING",
        "CANCELLED": "CANCELED",
        "ERROR": "FAILED",
    }
    return mapping.get(s, s if s else "PENDING")


class MeshyProvider:
    """Live Meshy multi-image-to-3d client (stdlib urllib only)."""

    name: str = "meshy"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = MESHY_API_BASE,
        max_http_retries: int = 3,
        timeout_s: float = 60.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise HostedError(
                "Meshy API key missing",
                code="api_key_missing",
                details={"provider": self.name},
            )
        self._api_key = api_key.strip()
        self._base = base_url.rstrip("/")
        self._max_retries = max(1, max_http_retries)
        self._timeout_s = timeout_s
        # Last successful poll payload (for download URLs)
        self._last_status: dict[str, ProviderJobStatus] = {}

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        json_body: bool = False,
    ) -> dict[str, Any]:
        """HTTP with 429/503 backoff; raise provider_http on hard failure."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            req = urllib.request.Request(
                url,
                data=body,
                headers=self._headers(json_body=json_body),
                method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    raw = resp.read()
                    if not raw:
                        return {}
                    parsed: Any = json.loads(raw.decode("utf-8"))
                    if isinstance(parsed, dict):
                        return parsed
                    return {"result": parsed}
            except urllib.error.HTTPError as exc:
                last_exc = exc
                code = exc.code
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")[:2000]
                except Exception:
                    err_body = str(exc)
                if code in (429, 503) and attempt + 1 < self._max_retries:
                    time.sleep(min(2**attempt, 30))
                    continue
                raise HostedError(
                    f"Meshy HTTP {code}: {err_body or exc.reason}",
                    code="provider_http",
                    details={
                        "provider": self.name,
                        "http_status": code,
                        "url": url.split("?")[0],
                        "body": err_body,
                    },
                ) from exc
            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt + 1 < self._max_retries:
                    time.sleep(min(2**attempt, 30))
                    continue
                raise HostedError(
                    f"Meshy network error: {exc.reason}",
                    code="provider_http",
                    details={"provider": self.name, "error": str(exc.reason)},
                ) from exc
            except TimeoutError as exc:
                last_exc = exc
                if attempt + 1 < self._max_retries:
                    time.sleep(min(2**attempt, 30))
                    continue
                raise HostedError(
                    f"Meshy request timeout after {self._timeout_s}s",
                    code="provider_http",
                    details={"provider": self.name, "timeout_s": self._timeout_s},
                ) from exc
            except json.JSONDecodeError as exc:
                raise HostedError(
                    f"Meshy response not JSON: {exc}",
                    code="provider_http",
                    details={"provider": self.name},
                ) from exc
        raise HostedError(
            f"Meshy request failed after retries: {last_exc}",
            code="provider_http",
            details={"provider": self.name},
        )

    def submit_multiview(
        self,
        image_uris: list[str],
        prompt: str,
        **opts: Any,
    ) -> str:
        if len(image_uris) < 2:
            raise HostedError(
                "Meshy multi-image-to-3d requires ≥2 images",
                code="multiview_required",
                details={"count": len(image_uris)},
            )
        body: dict[str, Any] = {
            "image_urls": list(image_uris),
            "target_formats": list(opts.get("target_formats") or DEFAULT_TARGET_FORMATS),
            "should_texture": bool(opts.get("should_texture", DEFAULT_SHOULD_TEXTURE)),
            "multi_view_thumbnails": bool(
                opts.get("multi_view_thumbnails", DEFAULT_MULTI_VIEW_THUMBNAILS)
            ),
            "ai_model": str(opts.get("ai_model") or DEFAULT_AI_MODEL),
        }
        if prompt:
            body["prompt"] = prompt
        # Allow extra provider knobs without forking
        for k in ("topology", "symmetry_mode", "pose_mode"):
            if k in opts and opts[k] is not None:
                body[k] = opts[k]

        url = f"{self._base}{MESHY_CREATE_PATH}"
        payload = self._request(
            "POST",
            url,
            body=json.dumps(body).encode("utf-8"),
            json_body=True,
        )
        # Create returns {"result": "<task_id>"}
        task_id = payload.get("result") or payload.get("id") or payload.get("task_id")
        if not task_id or not isinstance(task_id, str):
            raise HostedError(
                "Meshy create response missing task id",
                code="provider_http",
                details={"provider": self.name, "payload_keys": list(payload.keys())},
            )
        return task_id

    def poll(self, job_id: str) -> ProviderJobStatus:
        path = MESHY_STATUS_PATH.format(id=job_id)
        url = f"{self._base}{path}"
        payload = self._request("GET", url)
        status = _normalize_status(str(payload.get("status") or ""))
        allowed = {"PENDING", "IN_PROGRESS", "SUCCEEDED", "FAILED", "CANCELED"}
        if status not in allowed:
            status = "PENDING"
        progress_raw = payload.get("progress")
        progress: float | None = None
        if isinstance(progress_raw, (int, float)):
            progress = float(progress_raw)

        task_error = payload.get("task_error")
        if task_error is not None and not isinstance(task_error, dict):
            task_error = {"message": str(task_error)}

        model_urls = payload.get("model_urls")
        if model_urls is not None and not isinstance(model_urls, dict):
            model_urls = None

        thumbs = payload.get("thumbnail_urls")

        result = ProviderJobStatus(
            status=status,  # type: ignore[arg-type]
            progress=progress,
            message=str(payload.get("message") or "") or None,
            task_error=task_error if isinstance(task_error, dict) else None,
            model_urls=model_urls,
            thumbnail_urls=thumbs if isinstance(thumbs, (dict, list)) else None,
            raw=payload,
        )
        self._last_status[job_id] = result
        return result

    def download(self, job_id: str, dest_dir: Path | str) -> Path:
        """Download model_urls.stl (or glb) immediately; optional thumbnails."""
        status = self._last_status.get(job_id)
        if status is None or not status.model_urls:
            # Re-poll once to get URLs
            status = self.poll(job_id)
        if status.status != _TERMINAL_OK:
            te = status.task_error or {}
            raise HostedError(
                f"Meshy job not SUCCEEDED (status={status.status})",
                code="provider_failed",
                details={
                    "provider": self.name,
                    "job_id": job_id,
                    "status": status.status,
                    "type": te.get("type"),
                    "message": te.get("message") or status.message,
                    "doc_url": te.get("doc_url"),
                },
            )
        urls = status.model_urls or {}
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        stl_url = urls.get("stl") or urls.get("STL")
        glb_url = urls.get("glb") or urls.get("GLB")
        if stl_url:
            out = dest / "model.stl"
            self._download_url(stl_url, out)
            self._maybe_thumbs(status, dest)
            return out
        if glb_url:
            out = dest / "model.glb"
            self._download_url(glb_url, out)
            self._maybe_thumbs(status, dest)
            return out
        raise HostedError(
            "Meshy SUCCEEDED but no stl/glb model_urls",
            code="download_failed",
            details={"provider": self.name, "job_id": job_id, "keys": list(urls.keys())},
        )

    def _maybe_thumbs(self, status: ProviderJobStatus, dest: Path) -> None:
        thumbs = status.thumbnail_urls
        if not thumbs:
            return
        thumb_dir = dest / "provider_views"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        items: list[tuple[str, str]] = []
        if isinstance(thumbs, dict):
            items = [(str(k), str(v)) for k, v in thumbs.items() if v]
        elif isinstance(thumbs, list):
            items = [(f"thumb_{i}", str(u)) for i, u in enumerate(thumbs) if u]
        for name, url in items:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:64]
            ext = ".png"
            if ".jpg" in url.lower() or ".jpeg" in url.lower():
                ext = ".jpg"
            try:
                self._download_url(url, thumb_dir / f"{safe}{ext}")
            except HostedError:
                # Thumbnails are optional proof — do not fail the run
                continue

    def _download_url(self, url: str, dest: Path) -> None:
        """Download a signed URL immediately (expires)."""
        req = urllib.request.Request(url, method="GET")
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    data = resp.read()
                if not data:
                    raise HostedError(
                        "empty download from provider URL",
                        code="download_failed",
                        details={"dest": str(dest)},
                    )
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return
            except HostedError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self._max_retries:
                    time.sleep(min(2**attempt, 10))
                    continue
                raise HostedError(
                    f"download failed: {exc}",
                    code="download_failed",
                    details={"dest": str(dest), "error": str(exc)},
                ) from exc
        raise HostedError(
            f"download failed after retries: {last_exc}",
            code="download_failed",
            details={"dest": str(dest)},
        )
