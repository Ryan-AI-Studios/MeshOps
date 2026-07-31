"""Code-generated synthetic STL fixtures for sheet_score tests.

Rebuildable — not sacred originals. Four cases:
1. cylinder + coplanar arm sheet (T3-like)
2. two-body gap sheet
3. clothing/cape plane (false-positive; auto_action must stay non-delete)
4. solid cylinder control (low sheet_score)
"""

from __future__ import annotations

from pathlib import Path

import trimesh


def _cache_dir() -> Path:
    return Path(__file__).resolve().parent / "cache"


def _export(mesh: trimesh.Trimesh, name: str, out_dir: Path | None = None) -> Path:
    dest_dir = out_dir if out_dir is not None else _cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{name}.stl"
    mesh.export(path)
    return path


def build_solid_cylinder(out_dir: Path | None = None) -> Path:
    """Control: solid cylinder — expect low sheet_score."""
    mesh = trimesh.creation.cylinder(radius=5.0, height=40.0, sections=32)
    mesh.apply_translation(-mesh.centroid)
    return _export(mesh, "solid_cylinder", out_dir)


def build_cylinder_with_arm_sheet(out_dir: Path | None = None) -> Path:
    """Cylinder torso + coplanar thin arm sheet glued at side."""
    body = trimesh.creation.cylinder(radius=6.0, height=50.0, sections=48)
    body.apply_translation(-body.centroid)

    # Thin sheet (arm): large planar slab; subdivide for surface samples
    sheet = trimesh.creation.box(extents=[18.0, 0.12, 6.0])
    sheet = sheet.subdivide().subdivide()
    sheet.apply_translation([6.0 + 9.0, 0.0, 5.0])

    combined = trimesh.util.concatenate([body, sheet])
    return _export(combined, "cylinder_arm_sheet", out_dir)


def build_two_body_gap_sheet(out_dir: Path | None = None) -> Path:
    """Two solid bodies with a coplanar sheet bridging the gap."""
    left = trimesh.creation.cylinder(radius=4.0, height=30.0, sections=24)
    left.apply_translation([-10.0, 0.0, 0.0])
    right = trimesh.creation.cylinder(radius=4.0, height=30.0, sections=24)
    right.apply_translation([10.0, 0.0, 0.0])

    # Large bridge sheet between them (thin in Y) — dominant flat area
    gap_sheet = trimesh.creation.box(extents=[16.0, 0.08, 12.0])
    gap_sheet = gap_sheet.subdivide().subdivide()
    gap_sheet.apply_translation([0.0, 0.0, 5.0])

    combined = trimesh.util.concatenate([left, right, gap_sheet])
    return _export(combined, "two_body_gap_sheet", out_dir)


def build_clothing_cape_plane(out_dir: Path | None = None) -> Path:
    """Body + large smooth cape plane (clothing FP - must not force delete)."""
    body = trimesh.creation.cylinder(radius=6.0, height=50.0, sections=48)
    body.apply_translation(-body.centroid)

    # Large flat cape behind body — smooth plane, clothing-like
    cape = trimesh.creation.box(extents=[30.0, 0.15, 40.0])
    cape = cape.subdivide().subdivide()
    cape.apply_translation([0.0, -8.0, 0.0])

    combined = trimesh.util.concatenate([body, cape])
    return _export(combined, "clothing_cape", out_dir)


def build_all(out_dir: Path | None = None) -> dict[str, Path]:
    """Build all synthetic fixtures; return name → path map."""
    return {
        "solid_cylinder": build_solid_cylinder(out_dir),
        "cylinder_arm_sheet": build_cylinder_with_arm_sheet(out_dir),
        "two_body_gap_sheet": build_two_body_gap_sheet(out_dir),
        "clothing_cape": build_clothing_cape_plane(out_dir),
    }


def build_scaled(
    name: str,
    scale: float,
    out_dir: Path | None = None,
) -> Path:
    """Uniform-scale a named synthetic for scale-invariance smoke tests."""
    builders = {
        "solid_cylinder": build_solid_cylinder,
        "cylinder_arm_sheet": build_cylinder_with_arm_sheet,
        "two_body_gap_sheet": build_two_body_gap_sheet,
        "clothing_cape": build_clothing_cape_plane,
    }
    if name not in builders:
        raise KeyError(name)
    # Build to temp then scale
    base = builders[name](out_dir)
    mesh = trimesh.load(base, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.apply_scale(scale)
    return _export(mesh, f"{name}_scale{scale:g}", out_dir)


if __name__ == "__main__":
    paths = build_all()
    for k, v in paths.items():
        print(f"{k}: {v} faces={len(trimesh.load(v, force='mesh').faces)}")  # type: ignore[union-attr]
