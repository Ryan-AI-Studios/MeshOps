"""Proportion → blockout guides (track 0015).

Build GuidePackage from ProportionReport; emit JSON + Blender 5.2 bpy script.
Authoring aids only — not mesh or print success (Difficulty §12 / N6).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from meshops.proportion.analyze import load_report
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import GUIDE_HONESTY
from meshops.proportion.models import (
    DiameterMeasure,
    LandmarkXYZ,
    ProportionReport,
)

GUIDE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"

AXIS_NOTES: Final[str] = (
    "Z-up soles=0; +X camera-right; +Y depth toward camera for camera_left; "
    "face -Y (toes -Y, heels +Y)"
)

JSON_BASENAME: Final[str] = "proportion_guides.json"
BPY_BASENAME: Final[str] = "setup_proportion_guides.py"

# Frozen limb segment map (R8.1)
SEED_SEGMENT_MAP: Final[dict[str, tuple[str, str]]] = {
    "upper_arm_l": ("shoulder_l", "elbow_l"),
    "upper_arm_r": ("shoulder_r", "elbow_r"),
    "forearm_l": ("elbow_l", "wrist_l"),
    "forearm_r": ("elbow_r", "wrist_r"),
    "thigh_l": ("hip_l", "knee_l"),
    "thigh_r": ("hip_r", "knee_r"),
    "calf_l": ("knee_l", "ankle_l"),
    "calf_r": ("knee_r", "ankle_r"),
}

_NEAR_ZERO_LEN: Final[float] = 1e-9
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]")
_COLLAPSE_RE = re.compile(r"_+")

GuideFormat = Literal["bpy", "json", "both"]
EmptyKind = Literal["landmark", "height", "hu_rung"]
SeedKind = Literal["capsule", "ellipsoid"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GuideEmpty(BaseModel):
    """Named empty / height / HU rung (meters)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    x_m: float
    y_m: float
    z_m: float
    kind: EmptyKind
    source_id: str | None = None
    display_size_m: float = Field(gt=0.0)


class GuideSeed(BaseModel):
    """Optional SEED_* capsule or ellipsoid primitive (meters)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: SeedKind
    band_id: str | None = None
    level_id: str | None = None
    p0: tuple[float, float, float] | None = None
    p1: tuple[float, float, float] | None = None
    center: tuple[float, float, float] | None = None
    rx_m: float | None = None
    ry_m: float | None = None
    rz_m: float | None = None
    radius_m: float | None = None
    label: str = ""

    @model_validator(mode="after")
    def _label_seed_prefix(self) -> GuideSeed:
        if not self.label:
            object.__setattr__(self, "label", self.name)
        if not self.label.startswith("SEED_"):
            msg = f"seed label must start with SEED_: {self.label!r}"
            raise ValueError(msg)
        if not self.name.startswith("SEED_"):
            msg = f"seed name must start with SEED_: {self.name!r}"
            raise ValueError(msg)
        return self


class GuidePackage(BaseModel):
    """Versioned guide document (0015-owned schema 1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = GUIDE_SCHEMA_VERSION
    honesty: str = GUIDE_HONESTY
    source_report_schema: str | None = None
    height_m: float | None = None
    head_unit_frac: float | None = None
    head_unit_m: float | None = None
    axis_notes: str = AXIS_NOTES
    empties: list[GuideEmpty] = Field(default_factory=list)
    ladder: list[GuideEmpty] = Field(default_factory=list)
    seeds: list[GuideSeed] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def sanitize_landmark_key(key: str) -> str:
    """Keep [A-Za-z0-9_]; other→_; collapse repeats; empty→unnamed.

    Leading/trailing underscores are retained (R5 — no strip).
    """
    s = _SANITIZE_RE.sub("_", key)
    s = _COLLAPSE_RE.sub("_", s)
    return s if s else "unnamed"


def display_size_m(height_m: float | None) -> float:
    """Clamp empty display size (R5)."""
    return max(0.02, min(0.08, (height_m if height_m is not None else 1.7) * 0.03))


def _meters_from_lm(lm: LandmarkXYZ) -> tuple[float, float, float, bool]:
    """Return (x,y,z) meters with null→0.0; bool True if any component was null."""
    any_null = lm.x_m is None or lm.y_m is None or lm.z_m is None
    return (
        0.0 if lm.x_m is None else float(lm.x_m),
        0.0 if lm.y_m is None else float(lm.y_m),
        0.0 if lm.z_m is None else float(lm.z_m),
        any_null,
    )


def _has_any_meter(lm: LandmarkXYZ) -> bool:
    return lm.x_m is not None or lm.y_m is not None or lm.z_m is not None


def _resolve_diameter(diameters: list[DiameterMeasure], band_id: str) -> DiameterMeasure | None:
    """Prefer view front; first match among remaining."""
    matches = [d for d in diameters if d.band_id == band_id]
    if not matches:
        return None
    front = [d for d in matches if d.view == "front"]
    return front[0] if front else matches[0]


def _radius_from_diameter(d: DiameterMeasure) -> float | None:
    if d.half_width_m is not None:
        return float(d.half_width_m)
    if d.width_m is not None:
        return float(d.width_m) / 2.0
    return None


def _build_seeds(
    report: ProportionReport,
    *,
    head_unit_m: float | None,
    messages: list[str],
) -> list[GuideSeed]:
    seeds: list[GuideSeed] = []
    lms = report.landmarks_xyz

    for band_id, (p0_id, p1_id) in SEED_SEGMENT_MAP.items():
        if p0_id not in lms:
            messages.append(f"{band_id}: missing joint {p0_id} — seed skipped")
            continue
        if p1_id not in lms:
            messages.append(f"{band_id}: missing joint {p1_id} — seed skipped")
            continue
        lm0 = lms[p0_id]
        lm1 = lms[p1_id]
        if lm0.x_m is None or lm0.y_m is None or lm0.z_m is None:
            messages.append(f"{band_id}: joint {p0_id} missing meters — seed skipped")
            continue
        if lm1.x_m is None or lm1.y_m is None or lm1.z_m is None:
            messages.append(f"{band_id}: joint {p1_id} missing meters — seed skipped")
            continue
        diam = _resolve_diameter(report.diameters, band_id)
        if diam is None:
            messages.append(f"{band_id}: no usable radius — seed skipped")
            continue
        radius = _radius_from_diameter(diam)
        if radius is None:
            messages.append(f"{band_id}: no usable radius — seed skipped")
            continue
        p0 = (float(lm0.x_m), float(lm0.y_m), float(lm0.z_m))
        p1 = (float(lm1.x_m), float(lm1.y_m), float(lm1.z_m))
        length = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2)
        if length <= _NEAR_ZERO_LEN:
            messages.append(f"{band_id}: zero-length segment — seed skipped")
            continue
        name = f"SEED_{band_id}"
        seeds.append(
            GuideSeed(
                name=name,
                kind="capsule",
                band_id=band_id,
                p0=p0,
                p1=p1,
                radius_m=float(radius),
                label=name,
            )
        )

    # Cross-section ellipsoids
    for cs in report.cross_sections:
        level_id = cs.level_id
        name = f"SEED_CS_{level_id}"
        if report.height_m is None:
            messages.append(f"{name}: height_m unset — ellipsoid seed skipped")
            continue
        h = float(report.height_m)
        z_m = float(cs.z_frac) * h
        rx_m = float(cs.rx_frac) * h
        ry_m = float(cs.ry_frac) * h
        rz_m = max(0.02, float(head_unit_m) * 0.15) if head_unit_m is not None else 0.03
        seeds.append(
            GuideSeed(
                name=name,
                kind="ellipsoid",
                level_id=level_id,
                center=(0.0, 0.0, z_m),
                rx_m=rx_m,
                ry_m=ry_m,
                rz_m=rz_m,
                label=name,
            )
        )

    return seeds


def build_guide_package(
    report: ProportionReport,
    *,
    seeds: bool = False,
) -> GuidePackage:
    """Build GuidePackage from a loaded ProportionReport.

    Raises ProportionError(code=guides_empty) when nothing to emit.
    """
    messages: list[str] = []
    empties: list[GuideEmpty] = []
    ladder: list[GuideEmpty] = []
    seed_list: list[GuideSeed] = []

    disp = display_size_m(report.height_m)
    height_m = report.height_m
    head_unit_frac = report.head_unit_frac
    head_unit_m: float | None = None
    if height_m is not None and head_unit_frac is not None and head_unit_frac > 0.0:
        head_unit_m = float(height_m) * float(head_unit_frac)

    # Landmarks → empties (canonical id = dict key)
    for key, lm in report.landmarks_xyz.items():
        if not _has_any_meter(lm):
            continue
        x_m, y_m, z_m, any_null = _meters_from_lm(lm)
        if any_null:
            messages.append(
                f"{key}: missing *_m components zero-filled (need height_m on analyze for meters)"
            )
        name = f"LM_{sanitize_landmark_key(key)}"
        empties.append(
            GuideEmpty(
                name=name,
                x_m=x_m,
                y_m=y_m,
                z_m=z_m,
                kind="landmark",
                source_id=key,
                display_size_m=disp,
            )
        )

    # Height marker in empties
    if height_m is not None:
        empties.append(
            GuideEmpty(
                name="LM_HEIGHT",
                x_m=0.0,
                y_m=0.0,
                z_m=float(height_m),
                kind="height",
                source_id=None,
                display_size_m=disp,
            )
        )

    # HU ladder (R7): message whenever ladder is not built
    if (
        height_m is not None
        and head_unit_frac is not None
        and head_unit_frac > 0.0
        and head_unit_m is not None
    ):
        n = min(12, max(1, math.floor(1.0 / float(head_unit_frac) + 1e-9)))
        for k in range(0, n + 1):
            ladder.append(
                GuideEmpty(
                    name=f"LM_HU_{k}",
                    x_m=0.0,
                    y_m=0.0,
                    z_m=float(k) * float(head_unit_m),
                    kind="hu_rung",
                    source_id=None,
                    display_size_m=disp,
                )
            )
    else:
        messages.append("HU ladder omitted: height_m and/or head_unit_frac missing or non-positive")

    # Quality warn (still emit)
    if report.quality.needs_user_input:
        messages.append("quality.needs_user_input: guides still emitted — confirm primary figure")
    if report.quality.multi_figure:
        messages.append("quality.multi_figure: guides still emitted — confirm primary figure")

    if seeds:
        seed_list = _build_seeds(report, head_unit_m=head_unit_m, messages=messages)

    if not empties and not ladder and not seed_list:
        raise ProportionError(
            "nothing to emit: no landmark meters, height, HU ladder, or seeds",
            code="guides_empty",
        )

    counts = {
        "empties": len(empties),
        "ladder": len(ladder),
        "seeds": len(seed_list),
    }
    return GuidePackage(
        schema_version=GUIDE_SCHEMA_VERSION,
        honesty=GUIDE_HONESTY,
        source_report_schema=report.schema_version,
        height_m=height_m,
        head_unit_frac=head_unit_frac,
        head_unit_m=head_unit_m,
        axis_notes=AXIS_NOTES,
        empties=empties,
        ladder=ladder,
        seeds=seed_list,
        messages=messages,
        counts=counts,
    )


# ---------------------------------------------------------------------------
# bpy emit
# ---------------------------------------------------------------------------


def _py_repr(obj: Any) -> str:
    """Stable Python literal for embedding (no meshops types)."""
    return repr(obj)


def emit_bpy_script(package: GuidePackage) -> str:
    """Emit self-contained Blender 5.2 Python script (no meshops imports)."""
    empties_data = [
        {
            "name": e.name,
            "x_m": e.x_m,
            "y_m": e.y_m,
            "z_m": e.z_m,
            "display_size_m": e.display_size_m,
            "kind": e.kind,
        }
        for e in package.empties
    ]
    ladder_data = [
        {
            "name": e.name,
            "x_m": e.x_m,
            "y_m": e.y_m,
            "z_m": e.z_m,
            "display_size_m": e.display_size_m,
            "kind": e.kind,
        }
        for e in package.ladder
    ]
    seeds_data: list[dict[str, Any]] = []
    for s in package.seeds:
        entry: dict[str, Any] = {
            "name": s.name,
            "kind": s.kind,
            "band_id": s.band_id,
            "level_id": s.level_id,
        }
        if s.kind == "capsule":
            entry["p0"] = list(s.p0) if s.p0 is not None else None
            entry["p1"] = list(s.p1) if s.p1 is not None else None
            entry["radius_m"] = s.radius_m
        else:
            entry["center"] = list(s.center) if s.center is not None else None
            entry["rx_m"] = s.rx_m
            entry["ry_m"] = s.ry_m
            entry["rz_m"] = s.rz_m
        seeds_data.append(entry)

    lines: list[str] = [
        "# setup_proportion_guides.py — MeshOps track 0015",
        f"# honesty: {GUIDE_HONESTY}",
        "# N6 / Difficulty §12: guides and seeds are authoring aids only —",
        "# not mesh reconstruction, not print-ready, not hero sculpt success.",
        f"# axis_notes: {AXIS_NOTES}",
        "# guide schema_version: 1.0.0",
        "# MeshOps face -Y: toes -Y, heels +Y. Do not place foot centers at +Y only.",
        "# SEED only — not final mesh",
        "",
        "import math",
        "import bpy",
        "from mathutils import Matrix, Vector",
        "",
        "EMPTIES = " + _py_repr(empties_data),
        "LADDER = " + _py_repr(ladder_data),
        "SEEDS = " + _py_repr(seeds_data),
        f"HONESTY = {_py_repr(GUIDE_HONESTY)}",
        "",
        "# mode safety",
        'if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":',
        '    bpy.ops.object.mode_set(mode="OBJECT")',
        "",
        "# units: only if NONE (never assign scale_length)",
        'if bpy.context.scene.unit_settings.system == "NONE":',
        '    bpy.context.scene.unit_settings.system = "METRIC"',
        '    print("scene units were NONE → set METRIC")',
        "",
        'scale_len = getattr(bpy.context.scene.unit_settings, "scale_length", 1.0) or 1.0',
        "",
        "",
        "def to_bu(x_m, y_m, z_m):",
        "    return (x_m / scale_len, y_m / scale_len, z_m / scale_len)",
        "",
        "",
        "def ensure_collection(name):",
        "    col = bpy.data.collections.get(name)",
        "    if col is None:",
        "        col = bpy.data.collections.new(name)",
        "    scene_col = bpy.context.scene.collection",
        "    if col.name not in {c.name for c in scene_col.children}:",
        "        try:",
        "            scene_col.children.link(col)",
        "        except RuntimeError as exc:",
        '            print(f"warning: could not link collection {name!r} under scene: {exc}")',
        "    return col",
        "",
        "",
        "def ensure_empty(name, x_m, y_m, z_m, display_size_m, collection):",
        "    loc = to_bu(x_m, y_m, z_m)",
        "    size_bu = display_size_m / scale_len",
        "    obj = bpy.data.objects.get(name)",
        "    if obj is None:",
        "        obj = bpy.data.objects.new(name, None)",
        '        obj.empty_display_type = "PLAIN_AXES"',
        "        collection.objects.link(obj)",
        "    else:",
        "        # idempotent: update location + empty_display_size; do not duplicate",
        "        if obj.name not in collection.objects:",
        "            try:",
        "                collection.objects.link(obj)",
        "            except RuntimeError as exc:",
        '                print(f"warning: could not link empty {name!r}: {exc}")',
        "    obj.location = loc",
        "    obj.empty_display_size = size_bu",
        "    return obj",
        "",
        "",
        "def ensure_capsule(name, p0_m, p1_m, radius_m, collection):",
        "    # SEED only — not final mesh",
        "    p0 = Vector(to_bu(*p0_m))",
        "    p1 = Vector(to_bu(*p1_m))",
        "    v = p1 - p0",
        "    length = v.length",
        "    if length <= 1e-12:",
        "        return None",
        "    midpoint = (p0 + p1) / 2.0",
        "    radius = radius_m / scale_len",
        "    rot = Vector((0.0, 0.0, 1.0)).rotation_difference(v.normalized()).to_4x4()",
        "    mat = (",
        "        Matrix.Translation(midpoint)",
        "        @ rot",
        "        @ Matrix.Scale(radius, 4, (1, 0, 0))",
        "        @ Matrix.Scale(radius, 4, (0, 1, 0))",
        "        @ Matrix.Scale(length / 2.0, 4, (0, 0, 1))",
        "    )",
        "    obj = bpy.data.objects.get(name)",
        '    if obj is None or obj.type != "MESH":',
        "        if obj is not None:",
        "            bpy.data.objects.remove(obj, do_unlink=True)",
        "        bpy.ops.mesh.primitive_cylinder_add(",
        "            radius=1.0, depth=2.0, location=(0.0, 0.0, 0.0)",
        "        )",
        "        obj = bpy.context.active_object",
        "        obj.name = name",
        "        for col in list(obj.users_collection):",
        "            col.objects.unlink(obj)",
        "        collection.objects.link(obj)",
        "    else:",
        "        if obj.name not in collection.objects:",
        "            try:",
        "                collection.objects.link(obj)",
        "            except RuntimeError as exc:",
        '                print(f"warning: could not link seed {name!r}: {exc}")',
        "    # idempotent: full matrix_world (location + rotation + scale)",
        "    obj.matrix_world = mat",
        '    obj["meshops_role"] = "seed"',
        "    return obj",
        "",
        "",
        "def ensure_ellipsoid(name, center_m, rx_m, ry_m, rz_m, collection):",
        "    # SEED only — not final mesh",
        "    cx, cy, cz = to_bu(*center_m)",
        "    sx = rx_m / scale_len",
        "    sy = ry_m / scale_len",
        "    sz = rz_m / scale_len",
        "    mat = (",
        "        Matrix.Translation(Vector((cx, cy, cz)))",
        "        @ Matrix.Scale(sx, 4, (1, 0, 0))",
        "        @ Matrix.Scale(sy, 4, (0, 1, 0))",
        "        @ Matrix.Scale(sz, 4, (0, 0, 1))",
        "    )",
        "    obj = bpy.data.objects.get(name)",
        '    if obj is None or obj.type != "MESH":',
        "        if obj is not None:",
        "            bpy.data.objects.remove(obj, do_unlink=True)",
        "        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 0.0))",
        "        obj = bpy.context.active_object",
        "        obj.name = name",
        "        for col in list(obj.users_collection):",
        "            col.objects.unlink(obj)",
        "        collection.objects.link(obj)",
        "    else:",
        "        if obj.name not in collection.objects:",
        "            try:",
        "                collection.objects.link(obj)",
        "            except RuntimeError as exc:",
        '                print(f"warning: could not link seed {name!r}: {exc}")',
        "    # idempotent: full matrix_world (location + rotation + scale)",
        "    obj.matrix_world = mat",
        '    obj["meshops_role"] = "seed"',
        "    return obj",
        "",
        "",
        "guides_col = ensure_collection('Proportion_Guides')",
        "seeds_col = ensure_collection('Proportion_Seeds')",
        "",
        "n_empties = 0",
        "for e in EMPTIES + LADDER:",
        "    ensure_empty(",
        "        e['name'], e['x_m'], e['y_m'], e['z_m'], e['display_size_m'], guides_col",
        "    )",
        "    n_empties += 1",
        "",
        "n_seeds = 0",
        "for s in SEEDS:",
        "    if s['kind'] == 'capsule' and s.get('p0') and s.get('p1'):",
        "        if s.get('radius_m') is not None:",
        "            ensure_capsule(",
        "                s['name'], s['p0'], s['p1'], s['radius_m'], seeds_col",
        "            )",
        "            n_seeds += 1",
        "    elif s['kind'] == 'ellipsoid' and s.get('center') is not None:",
        "        ensure_ellipsoid(",
        "            s['name'], s['center'], s['rx_m'], s['ry_m'], s['rz_m'], seeds_col",
        "        )",
        "        n_seeds += 1",
        "",
        "print(",
        "    f'MeshOps proportion guides: empties+ladder={n_empties} seeds={n_seeds} '",
        "    f'honesty={HONESTY}'",
        ")",
        "print('guides only — not mesh or print success')",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Write paths
# ---------------------------------------------------------------------------


def _is_directory_out(out: Path) -> bool:
    """R1: existing dir OR ends with / or \\ OR no .py/.json suffix → directory."""
    s = str(out)
    if s.endswith(("/", "\\")):
        return True
    if out.exists() and out.is_dir():
        return True
    return out.suffix.lower() not in (".py", ".json")


def write_guides(
    out: Path | str,
    package: GuidePackage,
    *,
    format: GuideFormat = "both",
    force: bool = False,
) -> list[Path]:
    """Write guide JSON and/or bpy script per R1 path resolution.

    May append warn messages to package.messages for single-file + both.
    """
    out_path = Path(out)
    fmt: GuideFormat = format
    written: list[Path] = []

    is_dir = _is_directory_out(out_path)
    suffix = out_path.suffix.lower()

    if not is_dir:
        if suffix == ".py" and fmt == "json":
            raise ProportionError(
                "--out .py conflicts with --format json",
                code="guides_failed",
                details={"out": str(out_path), "format": fmt},
            )
        if suffix == ".json" and fmt == "bpy":
            raise ProportionError(
                "--out .json conflicts with --format bpy",
                code="guides_failed",
                details={"out": str(out_path), "format": fmt},
            )
        if fmt == "both":
            if suffix == ".py":
                package.messages.append("format both with single-file .py — emitting bpy only")
                fmt = "bpy"
            elif suffix == ".json":
                package.messages.append("format both with single-file .json — emitting json only")
                fmt = "json"

    targets: list[tuple[Path, Literal["json", "bpy"]]] = []
    if is_dir:
        directory = out_path
        if fmt in ("json", "both"):
            targets.append((directory / JSON_BASENAME, "json"))
        if fmt in ("bpy", "both"):
            targets.append((directory / BPY_BASENAME, "bpy"))
    else:
        if fmt == "json" or (fmt == "both" and suffix == ".json"):
            targets.append((out_path, "json"))
        elif fmt == "bpy" or (fmt == "both" and suffix == ".py"):
            targets.append((out_path, "bpy"))
        else:
            # single-file with matching format already narrowed above
            if suffix == ".json":
                targets.append((out_path, "json"))
            else:
                targets.append((out_path, "bpy"))

    try:
        for path, kind in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not force:
                raise ProportionError(
                    f"output already exists (use --force): {path}",
                    code="write_failed",
                    details={"path": str(path)},
                )
            if kind == "json":
                path.write_text(
                    json.dumps(package.model_dump(mode="json"), indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                path.write_text(emit_bpy_script(package), encoding="utf-8")
            written.append(path)
    except ProportionError:
        raise
    except OSError as exc:
        raise ProportionError(
            f"failed to write guides: {exc}",
            code="write_failed",
            details={"out": str(out_path)},
        ) from exc

    return written


def run_guides(
    report_path: Path | str,
    out: Path | str,
    *,
    format: GuideFormat = "both",
    seeds: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """CLI helper: load report → build → write; return success payload."""
    report = load_report(report_path)
    package = build_guide_package(report, seeds=seeds)
    paths = write_guides(out, package, format=format, force=force)
    return {
        "ok": True,
        "format": format,
        "paths": [str(p) for p in paths],
        "counts": dict(package.counts),
        "messages": list(package.messages),
    }


__all__ = [
    "AXIS_NOTES",
    "GUIDE_SCHEMA_VERSION",
    "SEED_SEGMENT_MAP",
    "GuideEmpty",
    "GuidePackage",
    "GuideSeed",
    "build_guide_package",
    "display_size_m",
    "emit_bpy_script",
    "run_guides",
    "sanitize_landmark_key",
    "write_guides",
]
