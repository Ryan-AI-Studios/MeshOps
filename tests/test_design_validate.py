"""Absolute design validate floors (DoD-17 primary gate)."""

from __future__ import annotations

import pytest

from meshops.design.errors import DesignError
from meshops.design.validate import validate_design_mesh
from meshops.models.diagnostics import MeshStats


def _stats(
    *,
    faces: int = 100,
    bbox_diagonal: float = 50.0,
    is_volume: bool | None = True,
) -> MeshStats:
    half = bbox_diagonal / (3**0.5) / 2
    return MeshStats(
        faces=faces,
        vertices=faces // 2,
        bbox_min=(-half, -half, -half),
        bbox_max=(half, half, half),
        bbox_diagonal=bbox_diagonal,
        components=1,
        is_watertight=is_volume,
        is_volume=is_volume,
        is_manifold=True,
        file_size_bytes=10_000,
        content_sha256="b" * 64,
        mesh_id="designval",
    )


def test_validate__pass_normal() -> None:
    validate_design_mesh(_stats())  # no raise


def test_validate__min_faces() -> None:
    with pytest.raises(DesignError) as ei:
        validate_design_mesh(_stats(faces=10))
    assert ei.value.code == "validation_failed"


def test_validate__unreasonable_cad_scale_tiny() -> None:
    with pytest.raises(DesignError) as ei:
        validate_design_mesh(_stats(bbox_diagonal=0.5))
    assert ei.value.code == "unreasonable_cad_scale"


def test_validate__unreasonable_cad_scale_huge() -> None:
    with pytest.raises(DesignError) as ei:
        validate_design_mesh(_stats(bbox_diagonal=5000.0))
    assert ei.value.code == "unreasonable_cad_scale"


def test_validate__not_volume() -> None:
    with pytest.raises(DesignError) as ei:
        validate_design_mesh(_stats(is_volume=False))
    assert ei.value.code == "validation_failed"


def test_validate__unknown_volume_ok() -> None:
    validate_design_mesh(_stats(is_volume=None))
