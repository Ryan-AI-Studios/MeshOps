# SPDX-License-Identifier: MIT
# MeshOps handoff probe — runs INSIDE Blender 5.2 LTS only.
# Do not import this module from the meshops venv (no bpy).
# Invoked as: blender -b PATH/handoff.blend -P probe_handoff.py -- --vg NAME --cameras a,b,c
#
# Asserts: mesh object exists, vertex group present, expected cameras present.
# Prints meshops_probe: ok=1|0 and details; exit 0 on success, 1 on fail.

from __future__ import annotations

import sys


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


def main() -> int:
    import bpy

    args = _parse_args(sys.argv)
    vg_name = str(args.get("vg") or "meshops_roi")
    cams_raw = str(args.get("cameras") or "front,three_quarter,top,waist_zoom")
    expected_cams = [c.strip() for c in cams_raw.split(",") if c.strip()]

    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    cam_objs = [o for o in bpy.data.objects if o.type == "CAMERA"]
    cam_names = sorted(o.name for o in cam_objs)

    errors: list[str] = []
    if not mesh_objs:
        errors.append("no_mesh_object")

    vg_ok = False
    roi_verts = 0
    if mesh_objs:
        obj = mesh_objs[0]
        if vg_name not in obj.vertex_groups:
            errors.append(f"missing_vertex_group:{vg_name}")
        else:
            vg_ok = True
            vg = obj.vertex_groups[vg_name]
            # Count verts with weight > 0 in this group
            mesh = obj.data
            for v in mesh.vertices:
                for g in v.groups:
                    if g.group == vg.index and g.weight > 0.0:
                        roi_verts += 1
                        break

    missing_cams = [n for n in expected_cams if n not in {o.name for o in cam_objs}]
    if missing_cams:
        errors.append(f"missing_cameras:{','.join(missing_cams)}")

    if errors:
        print(
            f"meshops_probe: ok=0 mesh_count={len(mesh_objs)} vg_ok={int(vg_ok)} "
            f"roi_verts={roi_verts} cameras={cam_names} errors={errors}"
        )
        return 1

    print(
        f"meshops_probe: ok=1 mesh_count={len(mesh_objs)} vg_ok=1 "
        f"roi_verts={roi_verts} cameras={cam_names}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
