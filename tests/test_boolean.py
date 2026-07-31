"""Direct manifold3d guarded boolean tests."""

from __future__ import annotations

import pytest
import trimesh

from meshops.boolean.manifold_guarded import BooleanError, BooleanOp, boolean_meshes


def test_boolean_local__two_boxes__ok() -> None:
    a = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    b = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    b.apply_translation([0.5, 0.5, 0.5])
    out, guard = boolean_meshes(a, b, BooleanOp.UNION)
    assert len(out.faces) > 0
    assert guard is not None
    assert guard.ok is True


def test_boolean_local__open_mesh__refuse() -> None:
    # Single triangle — not a volume
    open_mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    with pytest.raises(BooleanError) as ei:
        boolean_meshes(open_mesh, box, BooleanOp.UNION)
    assert ei.value.code == "not_volume"


def test_boolean_local__difference_ok() -> None:
    a = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    b = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    out, guard = boolean_meshes(a, b, BooleanOp.DIFFERENCE)
    assert len(out.faces) > 0
    assert guard is None or guard.ok is True
