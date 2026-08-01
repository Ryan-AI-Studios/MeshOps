"""Deterministic offline mock provider (default CI path — no network)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from meshops.hosted.errors import HostedError
from meshops.hosted.models import ProviderJobStatus

# Package-adjacent fixture path (tests may override via fixture_stl kwarg).
_DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "hosted" / "mock_mesh.stl"
)


class MockProvider:
    """Offline multi-view provider: submit → SUCCEEDED → copy fixture STL."""

    name: str = "mock"

    def __init__(
        self,
        *,
        fixture_stl: Path | str | None = None,
        delay_s: float = 0.0,
        fail_on_submit: bool = False,
        fail_on_poll: bool = False,
    ) -> None:
        self._fixture = Path(fixture_stl) if fixture_stl else _DEFAULT_FIXTURE
        self._delay_s = delay_s
        self._fail_on_submit = fail_on_submit
        self._fail_on_poll = fail_on_poll
        self._jobs: dict[str, dict[str, Any]] = {}
        self._seq = 0

    def submit_multiview(
        self,
        image_uris: list[str],
        prompt: str,
        **opts: Any,
    ) -> str:
        if self._fail_on_submit:
            raise HostedError(
                "mock provider forced submit failure",
                code="provider_failed",
                details={"provider": self.name},
            )
        if len(image_uris) < 2:
            raise HostedError(
                "mock requires ≥2 image URIs",
                code="multiview_required",
                details={"count": len(image_uris)},
            )
        self._seq += 1
        job_id = f"mock-{self._seq:04d}"
        self._jobs[job_id] = {
            "prompt": prompt,
            "image_count": len(image_uris),
            "opts": dict(opts),
            "status": "SUCCEEDED",
        }
        return job_id

    def poll(self, job_id: str) -> ProviderJobStatus:
        if self._fail_on_poll:
            return ProviderJobStatus(
                status="FAILED",
                message="mock forced poll failure",
                task_error={"type": "mock_fail", "message": "forced"},
            )
        if job_id not in self._jobs:
            return ProviderJobStatus(
                status="FAILED",
                message=f"unknown mock job: {job_id}",
                task_error={"type": "unknown_job", "message": job_id},
            )
        # delay_s reserved for future timed simulation; default 0
        _ = self._delay_s
        return ProviderJobStatus(
            status="SUCCEEDED",
            progress=1.0,
            message="mock success",
            model_urls={"stl": str(self._fixture)},
        )

    def download(self, job_id: str, dest_dir: Path | str) -> Path:
        if job_id not in self._jobs:
            raise HostedError(
                f"mock download: unknown job {job_id}",
                code="download_failed",
                details={"job_id": job_id},
            )
        if not self._fixture.is_file():
            raise HostedError(
                f"mock fixture STL missing: {self._fixture}",
                code="download_failed",
                details={"fixture": str(self._fixture)},
            )
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / "model.stl"
        shutil.copy2(self._fixture, out)
        return out
