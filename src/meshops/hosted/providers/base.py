"""Provider Protocol for multi-view image-to-3D adapters (track 0007)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from meshops.hosted.models import ProviderJobStatus


@runtime_checkable
class HostedProvider(Protocol):
    """Multi-view hosted generator adapter."""

    name: str

    def submit_multiview(
        self,
        image_uris: list[str],
        prompt: str,
        **opts: Any,
    ) -> str:
        """Submit multi-view job; return provider task/job id."""
        ...

    def poll(self, job_id: str) -> ProviderJobStatus:
        """Poll job status (PENDING / IN_PROGRESS / SUCCEEDED / FAILED / CANCELED)."""
        ...

    def download(self, job_id: str, dest_dir: Path | str) -> Path:
        """Download mesh immediately after success into dest_dir; return mesh path.

        Prefer STL; may return GLB for convert path.
        """
        ...
