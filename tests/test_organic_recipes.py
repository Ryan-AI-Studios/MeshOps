"""Organic recipe params, meta pins, N1/N2 refuse (track 0006)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from meshops.organic.errors import OrganicError
from meshops.organic.recipes import (
    META_FIELDS,
    META_TYPES,
    RECIPE_IDS,
    parse_params,
    refuse_n1_n2_params,
    validate_recipe_id,
)


def test_meta_types_exactly_five() -> None:
    """B1: only BALL CAPSULE PLANE ELLIPSOID CUBE — no SPHERE/TORUS."""
    assert {"BALL", "CAPSULE", "PLANE", "ELLIPSOID", "CUBE"} == META_TYPES
    assert "SPHERE" not in META_TYPES
    assert "TORUS" not in META_TYPES


def test_meta_fields_pinned() -> None:
    """B2: exact MetaElement field names."""
    required = {
        "co",
        "radius",
        "size_x",
        "size_y",
        "size_z",
        "rotation",
        "stiffness",
        "use_negative",
        "type",
    }
    assert required <= META_FIELDS
    assert "stiff" not in META_FIELDS
    assert "negative" not in META_FIELDS


def test_recipe_ids() -> None:
    assert {"simple_bust", "simple_figurine", "from_mesh"} == RECIPE_IDS


def test_unknown_recipe() -> None:
    with pytest.raises(OrganicError) as ei:
        validate_recipe_id("hero_sculpt")
    assert ei.value.code == "recipe_unknown"


def test_params_bounds() -> None:
    p = parse_params("simple_bust", {"resolution": 0.4, "threshold": 0.6})
    assert p.resolution == 0.4
    assert p.threshold == 0.6
    assert p.scale_mm == 100.0

    with pytest.raises(OrganicError) as ei:
        parse_params("simple_bust", {"resolution": 0.01})
    assert ei.value.code == "invalid_params"

    with pytest.raises(OrganicError):
        parse_params("simple_bust", {"resolution": 5.0})

    with pytest.raises(OrganicError):
        parse_params("simple_bust", {"threshold": 0.05})

    with pytest.raises(OrganicError):
        parse_params("simple_bust", {"smooth_iterations": 99})


def test_from_mesh_requires_source() -> None:
    with pytest.raises(OrganicError) as ei:
        parse_params("from_mesh", {})
    assert ei.value.code == "invalid_params"


def test_refuse_n1_n2() -> None:
    with pytest.raises(OrganicError) as ei:
        refuse_n1_n2_params({"voxel_remesh": True})
    assert ei.value.code == "recipe_refused"

    with pytest.raises(OrganicError) as ei:
        refuse_n1_n2_params({"boolean_after_solidify": True})
    assert ei.value.code == "recipe_refused"

    with pytest.raises(OrganicError) as ei:
        refuse_n1_n2_params({"linked_flat_delete": True})
    assert ei.value.code == "recipe_refused"


def test_script_pins_meta_and_export() -> None:
    """String pins in Blender script — B1/B2/B5/export name."""
    script = Path("src/meshops/organic/scripts/run_recipe.py")
    text = script.read_text(encoding="utf-8")
    assert "SPHERE" not in text or "no SPHERE" in text
    # Must not use SPHERE/TORUS as element types
    assert not re.search(r'el_type\s*=\s*"SPHERE"', text)
    assert not re.search(r'el_type\s*=\s*"TORUS"', text)
    assert "stiffness" in text
    assert "use_negative" in text
    assert "size_x" in text
    assert "render_resolution" in text
    assert "apply_modifiers=True" in text
    # Must not *use* the wrong RNA name as a kwarg (comment mention is OK)
    assert "use_mesh_modifiers=" not in text
    assert "mesh.stl.partial" in text or ".partial" in text
    assert "meshops_organic_ok" in text
