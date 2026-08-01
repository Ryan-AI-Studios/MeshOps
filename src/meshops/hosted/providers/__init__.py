"""Hosted provider registry — default live name: meshy; tests use mock."""

from __future__ import annotations

from typing import Any

from meshops.hosted.errors import HostedError
from meshops.hosted.providers.base import HostedProvider
from meshops.hosted.providers.meshy import MeshyProvider
from meshops.hosted.providers.mock import MockProvider

DEFAULT_PROVIDER_NAME = "meshy"

_REGISTRY: dict[str, type] = {
    "meshy": MeshyProvider,
    "mock": MockProvider,
}


def list_providers() -> list[dict[str, Any]]:
    """List registered adapters (no secrets)."""
    return [
        {
            "name": name,
            "default": name == DEFAULT_PROVIDER_NAME,
            "offline": name == "mock",
        }
        for name in sorted(_REGISTRY)
    ]


def get_provider(name: str, **kwargs: Any) -> HostedProvider:
    """Construct a provider adapter by name.

    kwargs forwarded to constructor (e.g. api_key, fixture_stl).
    """
    key = (name or DEFAULT_PROVIDER_NAME).strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise HostedError(
            f"unknown hosted provider: {name!r}",
            code="provider_failed",
            details={"known": sorted(_REGISTRY), "requested": name},
        )
    return cls(**kwargs)  # type: ignore[no-any-return]


__all__ = [
    "DEFAULT_PROVIDER_NAME",
    "HostedProvider",
    "MeshyProvider",
    "MockProvider",
    "get_provider",
    "list_providers",
]
