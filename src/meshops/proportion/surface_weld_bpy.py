# SPDX-License-Identifier: MIT
# MeshOps 0139 region weld — runs INSIDE Blender 5.2 LTS only.
# Do not import this module from the meshops venv (no bpy).
# Invoked as: blender -b -P surface_weld_bpy.py -- <job.json>
# Rebuilds the two named parts with the blockout-recipe emit ops, joins them,
# then voxel-remeshes that one object. Prints SURFACE_WELD_OK on success.
# Capsule uses the recipe emitter's cylinder primitive, not a UV sphere.

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _job_path(argv: list[str]) -> Path:
    if "--" not in argv:
        raise SystemExit("surface weld driver expected -- <job.json>")
    tail = argv[argv.index("--") + 1 :]
    if not tail:
        raise SystemExit("surface weld driver missing job.json path")
    return Path(tail[0])


def _vec3(member: dict, key: str) -> tuple[float, float, float]:
    raw = member.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise SystemExit(f"surface weld member {member.get('name')} missing {key}")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _num(member: dict, key: str) -> float:
    raw = member.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SystemExit(f"surface weld member {member.get('name')} missing {key}")
    return float(raw)


def trap_box_corners_m(member: dict) -> list[tuple[float, float, float]]:
    """Eight corners in the recipe emitter's vertex order, meters."""
    cx, cy, _cz = _vec3(member, "center")
    bot = _num(member, "bottom_half_width_m")
    top = _num(member, "top_half_width_m")
    half_depth = _num(member, "half_depth_m")
    z_bottom = _num(member, "z_bottom_m")
    z_top = _num(member, "z_top_m")
    return [
        (cx - bot, cy - half_depth, z_bottom),
        (cx + bot, cy - half_depth, z_bottom),
        (cx + bot, cy + half_depth, z_bottom),
        (cx - bot, cy + half_depth, z_bottom),
        (cx - top, cy - half_depth, z_top),
        (cx + top, cy - half_depth, z_top),
        (cx + top, cy + half_depth, z_top),
        (cx - top, cy + half_depth, z_top),
    ]


def emission_kind(member: dict) -> str:
    """Recipe emit op for one member. Capsule shares the cylinder primitive."""
    kind = member.get("kind")
    if kind in ("cylinder", "capsule"):
        _vec3(member, "p0")
        _vec3(member, "p1")
        _num(member, "radius_m")
        return "cylinder"
    if kind == "trap_box":
        trap_box_corners_m(member)
        return "trap_box"
    if kind == "box":
        _vec3(member, "center")
        _num(member, "half_depth_m")
        _num(member, "z_bottom_m")
        _num(member, "z_top_m")
        return "box"
    if kind == "ellipsoid":
        _vec3(member, "center")
        _num(member, "rx_m")
        _num(member, "ry_m")
        _num(member, "rz_m")
        return "ellipsoid"
    raise SystemExit(f"surface weld unknown kind {kind!r}")


def _clear_scene() -> None:
    import bpy

    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _scene_scale() -> float:
    import bpy

    settings = bpy.context.scene.unit_settings
    if getattr(settings, "system", None) == "NONE":
        settings.system = "METRIC"
    scale = getattr(settings, "scale_length", 1.0) or 1.0
    return float(scale)


def _to_bu(scale: float, x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
    return (x_m / scale, y_m / scale, z_m / scale)


def _bake(obj: object) -> None:
    import bpy

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _add_trap_box(member: dict, scale: float) -> object:
    import bpy

    name = str(member["name"])
    verts = [_to_bu(scale, *corner) for corner in trap_box_corners_m(member)]
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj["meshops_role"] = "recipe"
    return obj


def _add_box(member: dict, scale: float) -> object:
    import bpy

    cx, cy, _cz = _vec3(member, "center")
    z_bottom = _num(member, "z_bottom_m")
    z_top = _num(member, "z_top_m")
    half_width = member.get("top_half_width_m") or member.get("bottom_half_width_m") or 0.1
    half_depth = _num(member, "half_depth_m")
    z_mid = (z_bottom + z_top) / 2.0
    loc = _to_bu(scale, cx, cy, z_mid)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.active_object
    if obj is None:
        raise SystemExit("surface weld box primitive missing")
    obj.name = str(member["name"])
    obj.scale = (
        max(float(half_width) * 2.0, 1e-6) / scale,
        max(half_depth * 2.0, 1e-6) / scale,
        max(z_top - z_bottom, 1e-6) / scale,
    )
    obj["meshops_role"] = "recipe"
    _bake(obj)
    return obj


def _add_cylinder(member: dict, scale: float) -> object:
    import bpy
    from mathutils import Matrix, Vector

    p0 = Vector(_to_bu(scale, *_vec3(member, "p0")))
    p1 = Vector(_to_bu(scale, *_vec3(member, "p1")))
    span = p1 - p0
    length = span.length
    if length <= 1e-12:
        raise SystemExit(f"surface weld member {member.get('name')} has zero length")
    radius = _num(member, "radius_m") / scale
    rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(span.normalized()).to_matrix().to_4x4()
    matrix = (
        Matrix.Translation((p0 + p1) / 2.0)
        @ rotation
        @ Matrix.Scale(radius, 4, (1, 0, 0))
        @ Matrix.Scale(radius, 4, (0, 1, 0))
        @ Matrix.Scale(length / 2.0, 4, (0, 0, 1))
    )
    bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    if obj is None:
        raise SystemExit("surface weld cylinder primitive missing")
    obj.name = str(member["name"])
    obj.matrix_world = matrix
    obj["meshops_role"] = "recipe"
    _bake(obj)
    return obj


def _add_ellipsoid(member: dict, scale: float) -> object:
    import bpy
    from mathutils import Euler, Matrix, Vector

    center = _to_bu(scale, *_vec3(member, "center"))
    sx = _num(member, "rx_m") / scale
    sy = _num(member, "ry_m") / scale
    sz = _num(member, "rz_m") / scale
    rotation = Matrix.Identity(4)
    euler = member.get("rotation_euler_deg")
    if isinstance(euler, (list, tuple)) and len(euler) == 3:
        rotation = (
            Euler(
                (
                    math.radians(float(euler[0])),
                    math.radians(float(euler[1])),
                    math.radians(float(euler[2])),
                ),
                "XYZ",
            )
            .to_matrix()
            .to_4x4()
        )
    scale_matrix = (
        Matrix.Scale(sx, 4, (1, 0, 0))
        @ Matrix.Scale(sy, 4, (0, 1, 0))
        @ Matrix.Scale(sz, 4, (0, 0, 1))
    )
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    if obj is None:
        raise SystemExit("surface weld ellipsoid primitive missing")
    obj.name = str(member["name"])
    obj.matrix_world = Matrix.Translation(Vector(center)) @ rotation @ scale_matrix
    obj["meshops_role"] = "recipe"
    _bake(obj)
    return obj


def _add_member(member: dict, scale: float) -> object:
    kind = emission_kind(member)
    if kind == "trap_box":
        return _add_trap_box(member, scale)
    if kind == "box":
        return _add_box(member, scale)
    if kind == "cylinder":
        return _add_cylinder(member, scale)
    return _add_ellipsoid(member, scale)


def _join(objects: list) -> object:
    import bpy

    if len(objects) == 1:
        return objects[0]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.active_object
    if joined is None:
        raise SystemExit("surface weld join produced no object")
    return joined


def _merge_doubles(obj: object) -> None:
    """Weld coincident vertices on the datablock before it is exported."""
    import bmesh

    mesh = obj.data
    working = bmesh.new()
    working.from_mesh(mesh)
    bmesh.ops.remove_doubles(working, verts=list(working.verts), dist=1e-6)
    working.to_mesh(mesh)
    working.free()
    mesh.update()


def _export_stl(path: str, obj: object) -> None:
    import bpy

    _merge_doubles(obj)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        result = bpy.ops.wm.stl_export(filepath=path)
        if result == {"FINISHED"}:
            return
    except (AttributeError, RuntimeError):
        pass
    bpy.ops.export_mesh.stl(filepath=path)


def _voxel_remesh(obj: object, size: float) -> None:
    import bpy

    obj_ops = getattr(bpy.ops, "object", None)
    op = getattr(obj_ops, "voxel_remesh", None) if obj_ops is not None else None
    if op is None:
        raise SystemExit("surface_weld_unavailable: bpy.ops.object.voxel_remesh missing")
    mesh = obj.data
    if hasattr(mesh, "remesh_voxel_size"):
        mesh.remesh_voxel_size = float(size)
    if hasattr(mesh, "use_remesh_preserve_volume"):
        mesh.use_remesh_preserve_volume = True
    if hasattr(mesh, "remesh_voxel_adaptivity"):
        mesh.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    result = op()
    if result != {"FINISHED"}:
        raise SystemExit(f"surface_weld_unavailable: voxel_remesh {result}")


def main() -> None:
    job = json.loads(_job_path(sys.argv).read_text(encoding="utf-8"))
    members = job.get("members") or []
    if len(members) != 2:
        raise SystemExit("surface weld driver expected exactly two members")
    _clear_scene()
    scale = _scene_scale()
    objects = [_add_member(member, scale) for member in members]
    joined = _join(objects)
    _export_stl(str(job["before_stl"]), joined)
    _voxel_remesh(joined, float(job.get("voxel_coarse_m", 0.02)))
    _voxel_remesh(joined, float(job.get("voxel_fine_m", 0.014)))
    _export_stl(str(job["after_stl"]), joined)
    print("SURFACE_WELD_OK")


if __name__ == "__main__":
    main()
