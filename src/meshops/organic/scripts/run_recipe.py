# SPDX-License-Identifier: MIT
# MeshOps organic recipe runner — runs INSIDE Blender 5.2 LTS only.
# Do not import this module from the meshops venv (no bpy).
# Invoked as: blender -b -P run_recipe.py -- --out DIR --params-json PATH
#
# Meta types (B1): BALL, CAPSULE, PLANE, ELLIPSOID, CUBE — no SPHERE/TORUS
# Meta fields (B2): co, radius, size_x/y/z, rotation, stiffness, use_negative, type
# Both meta.resolution and meta.render_resolution (B5)
# Export: wm.stl_export(..., apply_modifiers=True) — not use_mesh_modifiers
# Atomic write: mesh.stl.partial → rename mesh.stl
#
# Build in design units so MetaBall.resolution in [0.05, 2.0] stays sane;
# after convert scale so exported STL coordinates are millimetres (~50-150).

from __future__ import annotations

import json
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> dict[str, str]:
    argv = argv[argv.index("--") + 1 :] if "--" in argv else argv[1:]
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and i + 1 < len(argv):
            out[a[2:].replace("-", "_")] = argv[i + 1]
            i += 2
        else:
            i += 1
    return out


def _clear_scene() -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.metaballs):
        if block.users == 0:
            bpy.data.metaballs.remove(block)


# Design-space height (Blender units) for comfortable MetaBall density with
# resolution ∈ [0.05, 2.0] (higher resolution value = coarser mesh in Blender).
# ~10 BU tall works with default resolution 0.4-0.5; then scale to scale_mm.
_DESIGN_HEIGHT = 10.0


def _set_units() -> None:
    import bpy

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def _add_meta_element(
    mb, *, co, radius, el_type="BALL", size=None, stiffness=2.0, use_negative=False
):
    """Add MetaElement with host-verified field names only."""
    el = mb.elements.new(type=el_type)
    el.co = co
    el.radius = radius
    el.stiffness = stiffness
    el.use_negative = use_negative
    if size is not None:
        el.size_x, el.size_y, el.size_z = size
    return el


def _build_simple_bust(mb) -> None:
    """Head + neck + shoulders in design units (~2 BU tall)."""
    s = _DESIGN_HEIGHT / 100.0
    _add_meta_element(mb, co=(0.0, 0.0, 70.0 * s), radius=22.0 * s, el_type="BALL", stiffness=2.0)
    _add_meta_element(
        mb,
        co=(0.0, 0.0, 72.0 * s),
        radius=18.0 * s,
        el_type="ELLIPSOID",
        size=(18.0 * s, 16.0 * s, 20.0 * s),
        stiffness=2.0,
    )
    _add_meta_element(mb, co=(0.0, 0.0, 48.0 * s), radius=8.0 * s, el_type="BALL", stiffness=2.0)
    _add_meta_element(mb, co=(0.0, 0.0, 40.0 * s), radius=9.0 * s, el_type="BALL", stiffness=2.0)
    _add_meta_element(mb, co=(0.0, 0.0, 28.0 * s), radius=20.0 * s, el_type="BALL", stiffness=2.0)
    _add_meta_element(
        mb, co=(-22.0 * s, 0.0, 30.0 * s), radius=12.0 * s, el_type="BALL", stiffness=2.0
    )
    _add_meta_element(
        mb, co=(22.0 * s, 0.0, 30.0 * s), radius=12.0 * s, el_type="BALL", stiffness=2.0
    )
    _add_meta_element(
        mb,
        co=(0.0, 0.0, 18.0 * s),
        radius=16.0 * s,
        el_type="ELLIPSOID",
        size=(28.0 * s, 12.0 * s, 14.0 * s),
        stiffness=2.0,
    )


def _build_simple_figurine(mb) -> None:
    """Vertical metaball stack in design units (~2 BU tall)."""
    s = _DESIGN_HEIGHT / 100.0
    _add_meta_element(mb, co=(0.0, 0.0, 5.0 * s), radius=10.0 * s, el_type="BALL")
    _add_meta_element(mb, co=(0.0, 0.0, 22.0 * s), radius=9.0 * s, el_type="BALL")
    _add_meta_element(mb, co=(0.0, 0.0, 42.0 * s), radius=14.0 * s, el_type="BALL")
    _add_meta_element(mb, co=(-16.0 * s, 0.0, 48.0 * s), radius=5.0 * s, el_type="BALL")
    _add_meta_element(mb, co=(16.0 * s, 0.0, 48.0 * s), radius=5.0 * s, el_type="BALL")
    _add_meta_element(mb, co=(0.0, 0.0, 62.0 * s), radius=8.0 * s, el_type="BALL")
    _add_meta_element(mb, co=(0.0, 0.0, 78.0 * s), radius=12.0 * s, el_type="BALL")
    _add_meta_element(
        mb,
        co=(0.0, 0.0, 40.0 * s),
        radius=12.0 * s,
        el_type="ELLIPSOID",
        size=(12.0 * s, 8.0 * s, 18.0 * s),
    )


def _scale_object_to_mm(obj, scale_mm: float) -> None:
    """Map design units → millimetre coordinates (MeshOps STL convention)."""
    import bpy

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    factor = float(scale_mm) / _DESIGN_HEIGHT
    obj.scale = (factor, factor, factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def _import_stl(path: str):
    import bpy

    before = set(bpy.data.objects)
    result = bpy.ops.wm.stl_import(filepath=path)
    if result != {"FINISHED"}:
        result = bpy.ops.wm.stl_import(filepath=path)
    after = [o for o in bpy.data.objects if o not in before]
    mesh_objs = [o for o in after if o.type == "MESH"]
    if not mesh_objs:
        mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"STL import produced no mesh: {path}")
    return mesh_objs[0]


def _smooth_mesh(obj, iterations: int) -> None:
    import bpy

    if iterations <= 0:
        return
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mod = obj.modifiers.new(name="MeshOpsSmooth", type="SMOOTH")
    mod.iterations = int(iterations)
    # Prefer export-time apply_modifiers over headless modifier_apply


def _convert_meta_to_mesh(meta_obj):
    import bpy

    # Active + select before convert (A1-BS1)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = meta_obj
    meta_obj.select_set(True)
    result = bpy.ops.object.convert(target="MESH")
    if result != {"FINISHED"}:
        raise RuntimeError(f"object.convert(MESH) failed: {result}")
    # Re-fetch after convert (stale refs)
    active = bpy.context.view_layer.objects.active
    if active is None or active.type != "MESH":
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        if not meshes:
            raise RuntimeError("no mesh object after convert")
        active = meshes[0]
        bpy.context.view_layer.objects.active = active
        active.select_set(True)
    return active


def _export_stl(obj, dest: Path) -> None:
    import bpy

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: write .partial then rename
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    result = bpy.ops.wm.stl_export(
        filepath=str(partial),
        export_selected_objects=True,
        apply_modifiers=True,  # host RNA (not use_mesh_modifiers)
        ascii_format=False,
        # evaluation_mode omitted in v1 (C8) — default fine for metaball→mesh
    )
    if result != {"FINISHED"}:
        raise RuntimeError(f"wm.stl_export failed: {result}")
    if not partial.is_file() or partial.stat().st_size <= 0:
        raise RuntimeError(f"export produced empty file: {partial}")
    if dest.exists():
        dest.unlink()
    partial.rename(dest)


def main() -> int:
    import bpy

    # B13 optional defense-in-depth
    assert bpy.app.version >= (5, 2, 0), f"need Blender 5.2+, got {bpy.app.version}"

    args = _parse_args(sys.argv)
    out_dir = Path(args.get("out", ""))
    params_path = Path(args.get("params_json", ""))
    if not out_dir:
        print("meshops_organic_error: missing --out", file=sys.stderr)
        return 2
    if not params_path.is_file():
        print(f"meshops_organic_error: params missing {params_path}", file=sys.stderr)
        return 2

    params = json.loads(params_path.read_text(encoding="utf-8"))
    recipe = params.get("recipe", "simple_bust")
    scale_mm = float(params.get("scale_mm", 100.0))
    resolution = float(params.get("resolution", 0.5))
    threshold = float(params.get("threshold", 0.6))
    smooth_iterations = int(params.get("smooth_iterations", 0))

    # Cap resolution floor against OOM (B5)
    resolution = max(0.05, min(2.0, resolution))
    threshold = max(0.2, min(2.0, threshold))

    _clear_scene()
    _set_units()

    mesh_obj = None
    if recipe in ("simple_bust", "simple_figurine"):
        mb = bpy.data.metaballs.new("MeshOpsMeta")
        # B5: set BOTH resolution and render_resolution
        mb.resolution = resolution
        mb.render_resolution = resolution
        mb.threshold = threshold
        meta_obj = bpy.data.objects.new("MeshOpsMeta", mb)
        bpy.context.scene.collection.objects.link(meta_obj)
        if recipe == "simple_bust":
            _build_simple_bust(mb)
        else:
            _build_simple_figurine(mb)
        mesh_obj = _convert_meta_to_mesh(meta_obj)
        # Design units -> mm numeric coords (~50-150 mm tall)
        _scale_object_to_mm(mesh_obj, scale_mm)
    elif recipe == "from_mesh":
        source = params.get("source_stl")
        if not source:
            raise RuntimeError("from_mesh requires source_stl absolute path")
        mesh_obj = _import_stl(str(source))
    else:
        raise RuntimeError(f"unknown recipe: {recipe}")

    _smooth_mesh(mesh_obj, smooth_iterations)

    out_stl = out_dir / "mesh.stl"
    _export_stl(mesh_obj, out_stl)
    print(f"meshops_organic_ok path={out_stl}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"meshops_organic_error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
