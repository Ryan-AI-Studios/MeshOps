"""Soak stability tests — env-gated (MESHOPS_BENCH_SOAK).

Default ``uv run pytest`` must never execute the long soak body (C2).
Marker ``@pytest.mark.slow`` alone is insufficient — first body line skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_SOAK_TRUTHY = frozenset({"1", "true", "yes", "on"})


def soak_env_enabled() -> bool:
    """Return True only when MESHOPS_BENCH_SOAK is truthy."""
    return os.environ.get("MESHOPS_BENCH_SOAK", "").strip().lower() in _SOAK_TRUTHY


@pytest.mark.slow
def test_soak_ingest_triage_s_repeated(tmp_path: Path) -> None:
    """N≥5 ingest+triage on size S; no exception.

    Enable with: ``$env:MESHOPS_BENCH_SOAK=1; uv run pytest -m slow tests/test_bench_soak.py``
    """
    if not soak_env_enabled():
        pytest.skip("MESHOPS_BENCH_SOAK not set (1/true/yes/on) — soak disabled for default CI")

    from meshops.bench.sizes import write_ladder_stl
    from meshops.ingest.pipeline import ingest_stl
    from meshops.triage.orchestrate import mesh_triage

    n = 5
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    stl = mesh_dir / "S.stl"
    write_ladder_stl("S", stl)

    for i in range(n):
        work = tmp_path / f"jobs_{i}"
        result = ingest_stl(stl, work_root=work)
        mesh_triage(result.mesh_id, work_root=work)


def test_soak_gate_helper_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit: gate is off without env (proves default pytest path)."""
    monkeypatch.delenv("MESHOPS_BENCH_SOAK", raising=False)
    assert soak_env_enabled() is False
    for val in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("MESHOPS_BENCH_SOAK", val)
        assert soak_env_enabled() is False


def test_soak_gate_helper_true_for_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MESHOPS_BENCH_SOAK", val)
        assert soak_env_enabled() is True
