# SPDX-License-Identifier: MIT
# MeshOps handoff builder — runs INSIDE Blender 5.2 LTS only.
# Do not import this module from the meshops venv (no bpy).
# Invoked as: blender -b -P build_handoff.py -- --mesh PATH --roi-json PATH --out PATH
#
# Difficulty §10 tips are written to instructions.md by the parent process.
# This script: clear scene → import STL → ROI vertex group → cameras → save .blend

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> dict[str, str]:
    # Args after "--" from blender -b -P script -- ...
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


def _import_stl(path: str) -> object:
    import bpy

    before = set(bpy.data.objects)
    # Blender 5.2: wm.stl_import (not import_mesh.stl)
    result = bpy.ops.wm.stl_import(filepath=path)
    if result != {"FINISHED"}:
        # Retry with explicit keyword variants if needed
        result = bpy.ops.wm.stl_import(filepath=path)
    after = [o for o in bpy.data.objects if o not in before]
    mesh_objs = [o for o in after if o.type == "MESH"]
    if not mesh_objs:
        # Fallback: any mesh in scene
        mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"STL import produced no mesh object: {path}")
    return mesh_objs[0]


def _clear_scene() -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Purge orphans lightly
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.cameras):
        if block.users == 0:
            bpy.data.cameras.remove(block)


def _assign_roi_vg(obj: object, bbox_min: list[float], bbox_max: list[float], name: str) -> int:
    import bpy
    from mathutils import Vector

    assert obj.type == "MESH"  # type: ignore[attr-defined]
    bpy.context.view_layer.objects.active = obj  # type: ignore[assignment]
    obj.select_set(True)  # type: ignore[attr-defined]

    # Ensure object mode
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    mesh = obj.data  # type: ignore[attr-defined]
    # Remove existing group if re-run
    if name in obj.vertex_groups:  # type: ignore[attr-defined]
        obj.vertex_groups.remove(obj.vertex_groups[name])  # type: ignore[attr-defined]
    vg = obj.vertex_groups.new(name=name)  # type: ignore[attr-defined]

    lo = Vector(bbox_min)
    hi = Vector(bbox_max)
    # Expand slightly for float edge cases
    pad = 1e-6
    indices: list[int] = []
    for i, v in enumerate(mesh.vertices):
        # World-space (object may have transform)
        co = obj.matrix_world @ v.co  # type: ignore[attr-defined]
        if (
            lo.x - pad <= co.x <= hi.x + pad
            and lo.y - pad <= co.y <= hi.y + pad
            and lo.z - pad <= co.z <= hi.z + pad
        ):
            indices.append(i)
    if indices:
        vg.add(indices, 1.0, "REPLACE")
    return len(indices)


def _bbox_cameras(
    bbox_min: list[float],
    bbox_max: list[float],
    *,
    distance_factor: float = 1.5,
) -> list[dict[str, object]]:
    """Match meshops.render.cameras.bbox_cameras conventions (+ waist_zoom)."""
    cx = 0.5 * (bbox_min[0] + bbox_max[0])
    cy = 0.5 * (bbox_min[1] + bbox_max[1])
    cz = 0.5 * (bbox_min[2] + bbox_max[2])
    dx = bbox_max[0] - bbox_min[0]
    dy = bbox_max[1] - bbox_min[1]
    dz = bbox_max[2] - bbox_min[2]
    diagonal = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    dist = distance_factor * diagonal
    ortho = diagonal * 0.6
    focal = (cx, cy, cz)

    specs: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
        ("front", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        ("three_quarter", (0.7, -0.7, 0.4), (0.0, 0.0, 1.0)),
        ("top", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        # T3 waist zoom: closer front toward mid-height (Difficulty MeshOps §7.3)
        ("waist_zoom", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ]
    cams: list[dict[str, object]] = []
    for name, direction, view_up in specs:
        length = math.sqrt(sum(c * c for c in direction)) or 1.0
        d = tuple(c / length for c in direction)
        use_dist = dist * (0.55 if name == "waist_zoom" else 1.0)
        pos = (cx + d[0] * use_dist, cy + d[1] * use_dist, cz + d[2] * use_dist)
        use_ortho = ortho * (0.35 if name == "waist_zoom" else 1.0)
        use_focal = focal
        if name == "waist_zoom":
            # Focus slightly lower-mid for waist contact region
            use_focal = (cx, cy, cz - 0.05 * dz)
        cams.append(
            {
                "name": name,
                "position": pos,
                "focal_point": use_focal,
                "view_up": view_up,
                "ortho_scale": use_ortho,
            }
        )
    return cams


def _look_at(
    obj: object,
    position: tuple[float, float, float],
    focal: tuple[float, float, float],
    view_up: tuple[float, float, float],
) -> None:
    from mathutils import Matrix, Vector

    pos = Vector(position)
    target = Vector(focal)
    up = Vector(view_up).normalized()
    direction = (target - pos).normalized()
    # Build basis: -Z looks toward target in Blender cameras
    z_axis = -direction
    x_axis = up.cross(z_axis)
    if x_axis.length < 1e-8:
        x_axis = Vector((1.0, 0.0, 0.0)).cross(z_axis)
    x_axis.normalize()
    y_axis = z_axis.cross(x_axis)
    rot = Matrix((x_axis, y_axis, z_axis)).transposed().to_4x4()
    mat = Matrix.Translation(pos) @ rot
    obj.matrix_world = mat  # type: ignore[attr-defined]


def _create_cameras(cam_specs: list[dict[str, object]]) -> list[str]:
    import bpy

    names: list[str] = []
    for spec in cam_specs:
        name = str(spec["name"])
        cam_data = bpy.data.cameras.new(name=name)
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = float(spec["ortho_scale"])  # type: ignore[arg-type]
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        pos = tuple(float(x) for x in spec["position"])  # type: ignore[arg-type]
        focal = tuple(float(x) for x in spec["focal_point"])  # type: ignore[arg-type]
        up = tuple(float(x) for x in spec["view_up"])  # type: ignore[arg-type]
        _look_at(cam_obj, pos, focal, up)  # type: ignore[arg-type]
        names.append(name)
    return names


def main() -> int:
    import bpy

    args = _parse_args(sys.argv)
    mesh_path = args.get("mesh")
    roi_json = args.get("roi_json")
    out_path = args.get("out")
    if not mesh_path or not roi_json or not out_path:
        print("usage: --mesh PATH --roi-json PATH --out PATH", file=sys.stderr)
        return 2

    roi = json.loads(Path(roi_json).read_text(encoding="utf-8"))
    bbox_min = list(roi["bbox_min"])
    bbox_max = list(roi["bbox_max"])
    vg_name = str(roi.get("vertex_group") or "meshops_roi")

    _clear_scene()
    obj = _import_stl(mesh_path)
    n_roi = _assign_roi_vg(obj, bbox_min, bbox_max, vg_name)
    print(f"meshops_handoff: roi_verts={n_roi} vg={vg_name}")

    # Cameras: meshops bbox conventions + waist_zoom for T3
    mesh_bbox_min = list(roi.get("mesh_bbox_min") or bbox_min)
    mesh_bbox_max = list(roi.get("mesh_bbox_max") or bbox_max)
    cam_names = _create_cameras(_bbox_cameras(mesh_bbox_min, mesh_bbox_max))
    print(f"meshops_handoff: cameras={cam_names}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True)
    if not out.is_file() or out.stat().st_size <= 0:
        print(f"meshops_handoff: failed to write blend {out}", file=sys.stderr)
        return 1
    print(f"meshops_handoff: wrote {out} size={out.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
