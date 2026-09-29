# SPDX-License-Identifier: MIT
# MeshOps 0138 multiview benchmark — runs INSIDE Blender 5.2 LTS only.
# Do not import this module from the meshops venv (no bpy).
# Invoked as: blender -b -P benchmark_multiview_bpy.py -- <job.json>
# Prints BENCHMARK_MULTIVIEW_OK on success. Camera numbers come from the job JSON.

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _job_path(argv: list[str]) -> Path:
    if "--" not in argv:
        raise SystemExit("benchmark bpy driver expected -- <job.json>")
    tail = argv[argv.index("--") + 1 :]
    if not tail:
        raise SystemExit("benchmark bpy driver missing job.json path")
    return Path(tail[0])


def _import_stl(path: str) -> None:
    import bpy

    try:
        result = bpy.ops.wm.stl_import(filepath=path)
        if result == {"FINISHED"}:
            return
    except (AttributeError, RuntimeError):
        pass
    bpy.ops.import_mesh.stl(filepath=path)


def _import_glb(path: str) -> None:
    import bpy

    wm = getattr(bpy.ops, "wm", None)
    op = getattr(wm, "gltf_import", None) if wm is not None else None
    if op is not None:
        try:
            result = op(filepath=path)
            if result == {"FINISHED"}:
                return
        except (AttributeError, RuntimeError):
            pass
    bpy.ops.import_scene.gltf(filepath=path)


def _drop_default_cubes() -> None:
    import bpy

    for obj in list(bpy.data.objects):
        if obj.name == "Cube" or obj.name.startswith("Cube."):
            bpy.data.objects.remove(obj, do_unlink=True)


def _apply_clay(rgb: list[float] | tuple[float, float, float]) -> None:
    import bpy

    color = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    material = bpy.data.materials.new("BenchmarkClay")
    material.diffuse_color = color
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        obj.color = color
        data = obj.data
        data.materials.clear()
        data.materials.append(material)


def _render_roles(capture: dict, out_dir: Path) -> None:
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    resolution = capture["resolution"]
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "MATERIAL"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = False
    world = capture["world_rgb"]
    scene.world.color = (float(world[0]), float(world[1]), float(world[2]))

    camera_data = bpy.data.cameras.new("BenchmarkCam")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(capture["ortho_scale"])
    camera = bpy.data.objects.new("BenchmarkCam", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    light_data = bpy.data.lights.new("BenchmarkSun", "SUN")
    light_data.energy = float(capture["sun_energy"])
    light = bpy.data.objects.new("BenchmarkSun", light_data)
    scene.collection.objects.link(light)
    euler = capture["sun_euler_deg"]
    light.rotation_euler = (
        math.radians(float(euler[0])),
        math.radians(float(euler[1])),
        math.radians(float(euler[2])),
    )

    look = Vector([float(v) for v in capture["look_at"]])
    track = capture["track"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for role in capture["roles"]:
        loc = [float(v) for v in capture["cameras"][role]]
        camera.location = loc
        direction = look - Vector(loc)
        camera.rotation_euler = direction.to_track_quat(str(track[0]), str(track[1])).to_euler()
        scene.render.filepath = str(out_dir / f"{role}.png")
        bpy.ops.render.render(write_still=True)


def _run_pass(item: dict, capture: dict) -> int:
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    kind = str(item["kind"])
    if kind == "setup":
        script = Path(item["script"])
        code = compile(script.read_text(encoding="utf-8"), str(script), "exec")
        exec(code, {"__name__": "__main__"})  # frozen setup script, Blender process only
        _drop_default_cubes()
        n_recipe = len([o for o in bpy.data.objects if o.name.startswith("RECIPE_")])
        print(f"BENCHMARK_RECIPE_OBJECTS {n_recipe}")
    elif kind == "stl":
        _import_stl(str(item["path"]))
    elif kind == "glb":
        _import_glb(str(item["path"]))
    else:
        raise RuntimeError(f"unknown benchmark pass kind: {kind}")
    if item.get("clay"):
        _apply_clay(capture["clay_rgb"])
    _render_roles(capture, Path(item["out_dir"]))
    return 0


def main() -> int:
    job = json.loads(_job_path(sys.argv).read_text(encoding="utf-8"))
    capture = job["capture"]
    for item in job["passes"]:
        _run_pass(item, capture)
    print("BENCHMARK_MULTIVIEW_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"BENCHMARK_MULTIVIEW_FAIL {exc}")
        raise SystemExit(1) from exc
