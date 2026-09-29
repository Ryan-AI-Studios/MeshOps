"""Four-view visual benchmark (track 0138).

Package A stills versus a MeshOps recipe setup and a Meshy GLB.
Authoring QA only. A numeric score never accepts the package.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from meshops.escalate.discover import find_blender
from meshops.escalate.errors import EscalateError
from meshops.escalate.version import require_blender_52
from meshops.hosted.convert import glb_to_stl
from meshops.ingest.stats import compute_stats
from meshops.proportion.checklist import load_package_checklist
from meshops.proportion.errors import ProportionError
from meshops.proportion.frame import _load_rgba_array, silhouette_mask
from meshops.proportion.honesty import MULTIVIEW_BENCHMARK_HONESTY
from meshops.proportion.silhouette import (
    _COV_HI,
    GRID_PX,
    _content_bbox,
    _corner_median_mask,
    _crop_and_resize_mask,
    _iou_dice,
)

BENCHMARK_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
JSON_BASENAME = "multiview_benchmark.json"
MARKDOWN_BASENAME = "multiview_benchmark.md"
CONTRACT_BASENAME = "capture_contract.json"
DEFAULT_HEIGHT_M = 1.72
ROLES: tuple[str, ...] = ("front", "left", "three_quarter", "back")
CROP_NAMES: tuple[str, ...] = (
    "face",
    "torso",
    "image_left_hand",
    "image_right_hand",
    "feet",
)
CROP_FRACTIONS: dict[str, tuple[float, float, float, float]] = {
    "face": (0.30, 0.00, 0.70, 0.22),
    "torso": (0.22, 0.18, 0.78, 0.55),
    "image_left_hand": (0.00, 0.28, 0.22, 0.62),
    "image_right_hand": (0.78, 0.28, 1.00, 0.62),
    "feet": (0.28, 0.88, 0.72, 1.00),
}
ALIGNMENT_MESSAGE = (
    "reference images are Package A stills; both candidates share the ortho rig; "
    "IoU is content-bbox aligned and is not matched-lens proof"
)
TOPOLOGY_NOTE = "context only — 0141 owns print"
_BPY = Path(__file__).with_name("benchmark_multiview_bpy.py")
_RESOLUTION = (960, 1280)
_WORLD_RGB = (0.12, 0.12, 0.14)
_ORTHO_SCALE = 2.05
_TRACK = ("-Z", "Y")
_LOOK_AT = (0.0, 0.0, 0.9)
_CAMERAS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, -2.6, 0.9),
    "left": (-2.6, 0.0, 0.9),
    "three_quarter": (-1.9, -1.9, 1.05),
    "back": (0.0, 2.6, 0.9),
}
_CLAY_RGB = (0.62, 0.56, 0.50)
_SUN_ENERGY = 3.0
_SUN_EULER_DEG = (45.0, 0.0, 30.0)


class FileProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    byte_size: int
    sha256: str


class CaptureContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Literal["BLENDER_WORKBENCH"] = "BLENDER_WORKBENCH"
    resolution: tuple[int, int] = _RESOLUTION
    world_rgb: tuple[float, float, float] = _WORLD_RGB
    camera_type: Literal["ORTHO"] = "ORTHO"
    ortho_scale: float = _ORTHO_SCALE
    track: tuple[str, str] = _TRACK
    look_at: tuple[float, float, float] = _LOOK_AT
    cameras: dict[str, tuple[float, float, float]] = Field(default_factory=lambda: dict(_CAMERAS))
    clay_rgb: tuple[float, float, float] = _CLAY_RGB
    sun_energy: float = _SUN_ENERGY
    sun_euler_deg: tuple[float, float, float] = _SUN_EULER_DEG
    roles: tuple[str, ...] = ROLES


class MaskScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iou: float
    dice: float
    ref_fg: int
    cand_fg: int
    intersection: int
    union: int
    ref_coverage_frac: float
    cand_coverage_frac: float
    ref_content_bbox: tuple[int, int, int, int] | None = None
    cand_content_bbox: tuple[int, int, int, int] | None = None
    residual_mean: float
    residual_png: str
    mask_method: Literal["primary", "corner_median"] = "primary"


class RoleViews(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meshops: MaskScore
    meshy: MaskScore


class CropRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    subject: str
    crop_source: Literal["content_bbox_fraction"]
    fractions: tuple[float, float, float, float]
    content_bbox: tuple[int, int, int, int] | None = None
    box_px: tuple[int, int, int, int] | None = None
    png: str | None = None


class NormalizeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    up: Literal["y", "z"]
    scale: float
    translation_xyz: tuple[float, float, float]
    yaw_deg: float
    height_m: float


class CandidateProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    setup_script: FileProvenance | None = None
    blockout_recipe: FileProvenance | None = None
    blend: FileProvenance | None = None
    glb: FileProvenance | None = None
    sibling_stl: FileProvenance | None = None
    bake_stl: FileProvenance | None = None
    normalized_stl: FileProvenance | None = None


class CandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    representation: str | None = None
    n_parts: int | None = None
    provenance: CandidateProvenance
    normalize: NormalizeRecord | None = None


class Candidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meshops: CandidateRecord
    meshy: CandidateRecord


class MeshStatsContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faces: int
    vertices: int
    components: int
    is_watertight: bool | None = None
    non_manifold_edge_count: int | None = None
    boundary_edge_count: int | None = None


class TopologyContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_supplied", "context_only"]
    representation: str | None = None
    mesh_stats: MeshStatsContext | None = None
    note: str


class TopologyPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meshops: TopologyContext
    meshy: TopologyContext


class ReferenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checklist: FileProvenance
    images: dict[str, FileProvenance]
    height_m: float
    figure: str | None = None
    multi_figure: bool
    in_scope_figures: list[str]


class MultiviewBenchmark(BaseModel):
    """Write-only bundle. Schema literal 1.0.0."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = BENCHMARK_SCHEMA_VERSION
    honesty: str
    ok: bool
    status: Literal["verdict_pending", "rejected", "accepted"]
    visual_verdict: Literal["accept", "reject"] | None = None
    needs_user_input: bool
    rank: None = None
    height_m: float
    yaw_deg: float
    capture: CaptureContract
    reference: ReferenceRecord
    candidates: Candidates
    views: dict[str, RoleViews]
    crops: list[CropRecord]
    topology_context: TopologyPair
    messages: list[str]
    paths: list[str]
    appearance_preview: list[str] = Field(default_factory=list)


def load_benchmark(path: Path | str) -> MultiviewBenchmark:
    """Load schema 1.0.0. Any other schema_version raises before model validate."""
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot load multiview benchmark: {file}: {exc}",
            code="benchmark_failed",
            details={"path": str(file)},
        ) from exc
    version = raw.get("schema_version") if isinstance(raw, dict) else None
    if version != BENCHMARK_SCHEMA_VERSION:
        raise ProportionError(
            f"multiview benchmark schema must be {BENCHMARK_SCHEMA_VERSION}, got {version!r}",
            code="benchmark_failed",
            details={"path": str(file), "schema_version": version},
        )
    try:
        return MultiviewBenchmark.model_validate(raw)
    except ValidationError as exc:
        raise ProportionError(
            f"invalid multiview benchmark: {exc}",
            code="benchmark_failed",
            details={"path": str(file)},
        ) from exc


def crop_box_px(
    content_bbox: tuple[int, int, int, int],
    fractions: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """Inclusive pixel box from content-bbox fractions. Y grows down."""
    x0, y0, x1, y1 = content_bbox
    fx0, fy0, fx1, fy1 = fractions
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    px0 = x0 + round(fx0 * width)
    py0 = y0 + round(fy0 * height)
    px1 = x0 + round(fx1 * width) - 1
    py1 = y0 + round(fy1 * height) - 1
    px0 = min(max(px0, x0), x1)
    py0 = min(max(py0, y0), y1)
    px1 = min(max(px1, x0), x1)
    py1 = min(max(py1, y0), y1)
    if px1 < px0:
        px1 = px0
    if py1 < py0:
        py1 = py0
    return px0, py0, px1, py1


def normalize_mesh(
    mesh: trimesh.Trimesh,
    *,
    up: Literal["y", "z"],
    height_m: float,
    yaw_deg: float,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Map to MeshOps Z-up, scale to height_m, ground zmin, center X/Y, then yaw +Z.

    ``up='y'`` sends ``(x, y, z)`` to ``(x, -z, y)`` so glTF +Z lands on face -Y.
    Vertex order is preserved (``process=False``).
    """
    if up not in ("y", "z"):
        raise ProportionError(
            f"normalize up must be 'y' or 'z', got {up!r}",
            code="benchmark_failed",
        )
    if height_m <= 0:
        raise ProportionError("height_m must be > 0", code="benchmark_failed")
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    mapped = np.column_stack((verts[:, 0], -verts[:, 2], verts[:, 1])) if up == "y" else verts
    z_extent = float(mapped[:, 2].max() - mapped[:, 2].min())
    if z_extent <= 0.0:
        raise ProportionError(
            "mesh has zero Z extent after the up-axis map",
            code="benchmark_failed",
        )
    scale = float(height_m) / z_extent
    mapped *= scale
    center_x = float((mapped[:, 0].max() + mapped[:, 0].min()) * 0.5)
    center_y = float((mapped[:, 1].max() + mapped[:, 1].min()) * 0.5)
    zmin = float(mapped[:, 2].min())
    mapped[:, 0] -= center_x
    mapped[:, 1] -= center_y
    mapped[:, 2] -= zmin
    yaw = float(yaw_deg)
    if yaw != 0.0:
        radians = math.radians(yaw)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        x_col = mapped[:, 0].copy()
        y_col = mapped[:, 1].copy()
        mapped[:, 0] = cosine * x_col - sine * y_col
        mapped[:, 1] = sine * x_col + cosine * y_col
    out = trimesh.Trimesh(vertices=mapped, faces=faces, process=False)
    record = {
        "up": up,
        "scale": scale,
        "translation_xyz": (-center_x, -center_y, -zmin),
        "yaw_deg": yaw,
        "height_m": float(height_m),
    }
    return out, record


def _mask_for_score(rgba: np.ndarray) -> tuple[np.ndarray, str]:
    """Near-white mask, then corner-median when that mask covers the frame.

    Package A studio gray and the Workbench ground are not near-white, so the
    primary mask marks the whole picture. Corner-median is the existing
    studio-gray recovery. An empty result stays empty.
    """
    primary = silhouette_mask(rgba)
    if primary.size and float(primary.mean()) > _COV_HI:
        corner, _bg, _std, _msgs = _corner_median_mask(rgba)
        return corner, "corner_median"
    return primary, "primary"


def score_aligned_masks(ref_image: Path, cand_image: Path) -> dict[str, Any]:
    """Content-bbox 256 IoU/Dice. No view role. Not print success.

    An empty foreground returns IoU and Dice 0. Grids stay out of the JSON.
    """
    ref_rgba = _load_rgba_array(ref_image)
    cand_rgba = _load_rgba_array(cand_image)
    ref_mask, ref_method = _mask_for_score(ref_rgba)
    cand_mask, cand_method = _mask_for_score(cand_rgba)
    method = "corner_median" if "corner_median" in (ref_method, cand_method) else "primary"
    ref_cov = float(ref_mask.mean()) if ref_mask.size else 0.0
    cand_cov = float(cand_mask.mean()) if cand_mask.size else 0.0
    ref_fg_raw = int(ref_mask.sum())
    cand_fg_raw = int(cand_mask.sum())
    if ref_fg_raw == 0 or cand_fg_raw == 0:
        return {
            "iou": 0.0,
            "dice": 0.0,
            "ref_fg": ref_fg_raw,
            "cand_fg": cand_fg_raw,
            "intersection": 0,
            "union": ref_fg_raw + cand_fg_raw,
            "ref_coverage_frac": ref_cov,
            "cand_coverage_frac": cand_cov,
            "ref_content_bbox": None,
            "cand_content_bbox": None,
            "residual_mean": 0.0,
            "mask_method": method,
            "_ref_grid": None,
            "_cand_grid": None,
        }
    ref_bbox = _content_bbox(ref_mask)
    cand_bbox = _content_bbox(cand_mask)
    ref_grid = _crop_and_resize_mask(ref_mask, ref_bbox, grid_px=GRID_PX)
    cand_grid = _crop_and_resize_mask(cand_mask, cand_bbox, grid_px=GRID_PX)
    iou, dice, ref_fg, cand_fg, inter, union = _iou_dice(ref_grid, cand_grid)
    residual = np.abs(ref_grid.astype(np.float64) - cand_grid.astype(np.float64))
    return {
        "iou": float(iou),
        "dice": float(dice),
        "ref_fg": int(ref_fg),
        "cand_fg": int(cand_fg),
        "intersection": int(inter),
        "union": int(union),
        "ref_coverage_frac": ref_cov,
        "cand_coverage_frac": cand_cov,
        "ref_content_bbox": ref_bbox,
        "cand_content_bbox": cand_bbox,
        "residual_mean": float(residual.mean()),
        "mask_method": method,
        "_ref_grid": ref_grid,
        "_cand_grid": cand_grid,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _provenance(path: Path) -> FileProvenance:
    resolved = path.resolve()
    return FileProvenance(
        path=str(resolved),
        byte_size=resolved.stat().st_size,
        sha256=_sha256(resolved),
    )


def _require_roles(directory: Path, label: str) -> None:
    missing = [role for role in ROLES if not (directory / f"{role}.png").is_file()]
    if missing:
        raise ProportionError(
            f"missing {label} PNG(s): {', '.join(missing)}",
            code="benchmark_failed",
            details={"missing": missing, "directory": str(directory)},
        )


def _copy_roles(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        src = source / f"{role}.png"
        dst = dest / f"{role}.png"
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)


def _pillow_image() -> Any:
    """Pillow ``Image`` class. The proportion extra is optional on the design job."""
    try:
        from PIL import Image  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:
        raise ProportionError(
            "Pillow is required to write benchmark images; install meshops[proportion]",
            code="benchmark_failed",
        ) from exc
    return Image


def _write_residual(path: Path, ref_grid: np.ndarray | None, cand_grid: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_cls = _pillow_image()
    if ref_grid is None or cand_grid is None:
        image = image_cls.new("L", (GRID_PX, GRID_PX), 0)
    else:
        diff = np.abs(ref_grid.astype(np.float64) - cand_grid.astype(np.float64))
        image = image_cls.fromarray((diff * 255.0).astype(np.uint8), mode="L")
    image.save(path)


def _score_payload(raw: dict[str, Any], residual_png: str) -> dict[str, Any]:
    body = {key: value for key, value in raw.items() if not key.startswith("_")}
    body["residual_png"] = residual_png
    return body


def _n_parts_from_recipe(path: Path) -> int:
    """Count ``parts`` in a sibling recipe that is present on disk.

    A missing file is the caller's concern. A present file that is not a JSON
    object with a ``parts`` list fails closed.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"blockout recipe is not valid JSON: {path}",
            code="benchmark_failed",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    parts = raw.get("parts") if isinstance(raw, dict) else None
    if not isinstance(parts, list):
        raise ProportionError(
            f"blockout recipe needs a parts list: {path}",
            code="benchmark_failed",
            details={"path": str(path)},
        )
    return len(parts)


def _checked_blender() -> Path | None:
    """Return a Blender 5.2 executable, or None when Blender is absent.

    A present binary that is not 5.2.x fails closed. View-dir mode does not
    call this.
    """
    blender = find_blender(require=False)
    if blender is None:
        return None
    try:
        require_blender_52(blender)
    except EscalateError as exc:
        raise ProportionError(
            f"Blender 5.2 is required to render benchmark views: {exc}",
            code="benchmark_render_unavailable",
            details={"blender": str(blender), "cause": str(exc.code)},
        ) from exc
    return blender


def _as_mesh(loaded: object) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Scene):
        baked = loaded.to_mesh()
        if isinstance(baked, trimesh.Trimesh):
            return baked
        raise ProportionError("GLB bake produced no mesh", code="benchmark_failed")
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise ProportionError(
        f"GLB bake produced {type(loaded).__name__}",
        code="benchmark_failed",
    )


def _capture_payload() -> dict[str, Any]:
    return CaptureContract().model_dump(mode="json")


def _render_with_blender(out: Path, passes: list[dict[str, Any]], blender: Path) -> str:
    if not blender.is_file():
        raise ProportionError(
            "Blender 5.2 is required to render benchmark views",
            code="benchmark_render_unavailable",
            details={"blender": str(blender)},
        )
    job_path = out / "frozen" / "benchmark_job.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job = {"capture": _capture_payload(), "passes": passes}
    job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [str(blender), "-b", "-P", str(_BPY), "--", str(job_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout or ""
    if proc.returncode != 0 or "BENCHMARK_MULTIVIEW_OK" not in stdout:
        tail = (proc.stderr or stdout)[-2000:]
        raise ProportionError(
            f"benchmark render failed: {tail}",
            code="benchmark_failed",
            details={"returncode": proc.returncode, "stderr_tail": tail},
        )
    return stdout


def _recipe_count(stdout: str) -> int | None:
    found: int | None = None
    for line in stdout.splitlines():
        if line.startswith("BENCHMARK_RECIPE_OBJECTS "):
            try:
                found = int(line.split()[-1])
            except ValueError:
                continue
    return found


def _verdict_state(
    verdict: str | None,
    *,
    needs_user_input: bool,
) -> tuple[bool, Literal["verdict_pending", "rejected", "accepted"]]:
    if needs_user_input or verdict is None:
        return False, "verdict_pending"
    if verdict == "reject":
        return False, "rejected"
    if verdict == "accept":
        return True, "accepted"
    raise ProportionError(
        f"--verdict must be accept or reject, got {verdict!r}",
        code="benchmark_failed",
    )


def _write_markdown(path: Path, doc: MultiviewBenchmark, out: Path) -> None:
    lines = [
        "# Multiview visual benchmark",
        "",
        f"honesty: {doc.honesty}",
        "",
        f"status: {doc.status}",
        "",
        "Authoring QA only. Not mesh or print success.",
        "",
        doc.messages[0] if doc.messages else ALIGNMENT_MESSAGE,
        "",
    ]
    for role in ROLES:
        pair = doc.views[role]
        lines.append(f"## {role}")
        lines.append("")
        lines.append("| | reference | meshops | meshy |")
        lines.append("|---|---|---|---|")
        ref_rel = f"views/reference/{role}.png"
        meshops_rel = f"views/meshops/{role}.png"
        meshy_rel = f"views/meshy/{role}.png"
        lines.append(f"| image | ![]({ref_rel}) | ![]({meshops_rel}) | ![]({meshy_rel}) |")
        lines.append(f"| iou | | {pair.meshops.iou:.4f} | {pair.meshy.iou:.4f} |")
        lines.append(f"| dice | | {pair.meshops.dice:.4f} | {pair.meshy.dice:.4f} |")
        lines.append("")
    lines.append("## crops")
    lines.append("")
    for name in CROP_NAMES:
        lines.append(f"- {name} (`content_bbox_fraction`)")
    lines.append("")
    lines.append("## topology")
    lines.append("")
    lines.append(TOPOLOGY_NOTE)
    lines.append("")
    lines.append("## provenance")
    lines.append("")
    lines.append(f"- reference checklist: `{doc.reference.checklist.path}`")
    for role, image in doc.reference.images.items():
        lines.append(f"- reference {role}: `{image.path}`")
    setup = doc.candidates.meshops.provenance.setup_script
    if setup is not None:
        lines.append(f"- meshops setup: `{setup.path}`")
    recipe = doc.candidates.meshops.provenance.blockout_recipe
    if recipe is not None:
        lines.append(f"- meshops recipe: `{recipe.path}`")
    blend = doc.candidates.meshops.provenance.blend
    if blend is not None:
        lines.append(f"- meshops blend hash: `{blend.path}`")
    glb = doc.candidates.meshy.provenance.glb
    if glb is not None:
        lines.append(f"- meshy glb: `{glb.path}`")
    sibling = doc.candidates.meshy.provenance.sibling_stl
    if sibling is not None:
        lines.append(f"- meshy sibling stl hash only: `{sibling.path}`")
    normalized = doc.candidates.meshy.provenance.normalized_stl
    if normalized is not None:
        lines.append(f"- meshy normalized: `{normalized.path}`")
    lines.append("")
    lines.append(f"bundle: `{out.resolve()}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _crop_subject(
    *,
    role: str,
    name: str,
    subject: str,
    image_path: Path,
    content_bbox: tuple[int, int, int, int] | None,
    out: Path,
) -> CropRecord:
    fractions = CROP_FRACTIONS[name]
    png_rel: str | None = None
    box: tuple[int, int, int, int] | None = None
    if content_bbox is not None:
        box = crop_box_px(content_bbox, fractions)
        dest = out / "crops" / role / name / f"{subject}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with _pillow_image().open(image_path) as image:
            px0, py0, px1, py1 = box
            image.crop((px0, py0, px1 + 1, py1 + 1)).save(dest)
        png_rel = dest.resolve().relative_to(out.resolve()).as_posix()
    return CropRecord(
        name=name,
        role=role,
        subject=subject,
        crop_source="content_bbox_fraction",
        fractions=fractions,
        content_bbox=content_bbox,
        box_px=box,
        png=png_rel,
    )


def run_benchmark_multiview(
    reference: Path | str,
    out: Path | str,
    *,
    meshops_setup: Path | str | None = None,
    meshy_glb: Path | str | None = None,
    meshops_views: Path | str | None = None,
    meshy_views: Path | str | None = None,
    verdict: str | None = None,
    figure: str | None = None,
    yaw_deg: float = 0.0,
    appearance: bool = False,
    meshy_sibling_stl: Path | str | None = None,
    meshops_blend: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write the multiview bundle and return the JSON payload.

    ``--verdict`` omitted keeps ``ok`` false. IoU never sets ``ok`` or ``rank``.
    """
    if verdict is not None and verdict not in ("accept", "reject"):
        raise ProportionError(
            f"--verdict must be accept or reject, got {verdict!r}",
            code="benchmark_failed",
        )
    ref_dir = Path(reference)
    out_dir = Path(out)
    setup_path = Path(meshops_setup) if meshops_setup else None
    glb_path = Path(meshy_glb) if meshy_glb else None
    meshops_view_dir = Path(meshops_views) if meshops_views else None
    meshy_view_dir = Path(meshy_views) if meshy_views else None
    sibling_path = Path(meshy_sibling_stl) if meshy_sibling_stl else None
    blend_path = Path(meshops_blend) if meshops_blend else None

    checklist_path = ref_dir / "package_checklist.json"
    if not checklist_path.is_file():
        raise ProportionError(
            f"reference directory needs package_checklist.json: {ref_dir}",
            code="benchmark_failed",
            details={"reference": str(ref_dir)},
        )
    _require_roles(ref_dir, "reference")
    checklist = load_package_checklist(checklist_path)
    if figure is not None and figure not in checklist.in_scope_figures:
        raise ProportionError(
            f"--figure {figure!r} is not in in_scope_figures",
            code="benchmark_failed",
            details={
                "figure": figure,
                "in_scope_figures": list(checklist.in_scope_figures),
            },
        )
    needs_user = bool(checklist.multi_figure) and figure is None
    ok, status = _verdict_state(verdict, needs_user_input=needs_user)
    height_m = float(checklist.height_m) if checklist.height_m is not None else DEFAULT_HEIGHT_M

    json_path = out_dir / JSON_BASENAME
    if json_path.exists() and not force:
        raise ProportionError(
            f"benchmark bundle exists (pass --force): {json_path}",
            code="benchmark_failed",
            details={"path": str(json_path)},
        )
    if meshops_view_dir is None and setup_path is None:
        raise ProportionError(
            "--meshops-setup is required unless --meshops-views is set",
            code="benchmark_failed",
        )
    if meshy_view_dir is None and glb_path is None:
        raise ProportionError(
            "--meshy-glb is required unless --meshy-views is set",
            code="benchmark_failed",
        )
    if meshops_view_dir is not None:
        _require_roles(meshops_view_dir, "meshops views")
    if meshy_view_dir is not None:
        _require_roles(meshy_view_dir, "meshy views")
    n_parts: int | None = None
    recipe_path: Path | None = None
    if setup_path is not None:
        if not setup_path.is_file():
            raise ProportionError(
                f"meshops setup script not found: {setup_path}",
                code="benchmark_failed",
            )
        recipe_path = setup_path.parent / "blockout_recipe.json"
        if recipe_path.is_file():
            n_parts = _n_parts_from_recipe(recipe_path)
    render_geometry = meshops_view_dir is None or meshy_view_dir is None
    wants_appearance = bool(appearance and glb_path is not None)
    blender_bin: Path | None = None
    if render_geometry or wants_appearance:
        blender_bin = _checked_blender()
        if render_geometry and blender_bin is None:
            raise ProportionError(
                "Blender 5.2 is required to render benchmark views",
                code="benchmark_render_unavailable",
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = out_dir / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    _copy_roles(ref_dir, out_dir / "views" / "reference")
    if meshops_view_dir is not None:
        _copy_roles(meshops_view_dir, out_dir / "views" / "meshops")
    if meshy_view_dir is not None:
        _copy_roles(meshy_view_dir, out_dir / "views" / "meshy")

    messages = [ALIGNMENT_MESSAGE]
    if needs_user:
        messages.append("checklist is multi-figure; pass --figure to choose one subject")

    meshops_prov = CandidateProvenance()
    if setup_path is not None:
        meshops_prov.setup_script = _provenance(setup_path)
        shutil.copy2(setup_path, frozen / setup_path.name)
        if recipe_path is not None and recipe_path.is_file():
            meshops_prov.blockout_recipe = _provenance(recipe_path)
            shutil.copy2(recipe_path, frozen / recipe_path.name)
    if blend_path is not None:
        if not blend_path.is_file():
            raise ProportionError(
                f"meshops blend not found: {blend_path}",
                code="benchmark_failed",
            )
        meshops_prov.blend = _provenance(blend_path)

    meshy_prov = CandidateProvenance()
    normalize_record: NormalizeRecord | None = None
    meshy_stats: MeshStatsContext | None = None
    norm_path: Path | None = None
    if sibling_path is not None:
        if not sibling_path.is_file():
            raise ProportionError(
                f"meshy sibling stl not found: {sibling_path}",
                code="benchmark_failed",
            )
        meshy_prov.sibling_stl = _provenance(sibling_path)
    if glb_path is not None:
        if not glb_path.is_file():
            raise ProportionError(
                f"meshy glb not found: {glb_path}",
                code="benchmark_failed",
            )
        meshy_prov.glb = _provenance(glb_path)
        bake_path = frozen / "meshy-bake.stl"
        glb_to_stl(glb_path, bake_path)
        meshy_prov.bake_stl = _provenance(bake_path)
        loaded = _as_mesh(trimesh.load(bake_path, force="mesh"))
        normalized, raw_record = normalize_mesh(
            loaded,
            up="y",
            height_m=height_m,
            yaw_deg=yaw_deg,
        )
        norm_path = frozen / "meshy-normalized.stl"
        normalized.export(norm_path)
        meshy_prov.normalized_stl = _provenance(norm_path)
        normalize_record = NormalizeRecord.model_validate(raw_record)
        stats = compute_stats(
            normalized,
            mesh_id="meshy-normalized",
            content_sha256_hex=meshy_prov.normalized_stl.sha256,
            file_size_bytes=meshy_prov.normalized_stl.byte_size,
            source_path=str(norm_path.resolve()),
        )
        meshy_stats = MeshStatsContext(
            faces=int(stats.faces),
            vertices=int(stats.vertices),
            components=int(stats.components),
            is_watertight=stats.is_watertight,
            non_manifold_edge_count=stats.non_manifold_edge_count,
            boundary_edge_count=stats.boundary_edge_count,
        )

    passes: list[dict[str, Any]] = []
    if meshops_view_dir is None and setup_path is not None:
        passes.append(
            {
                "kind": "setup",
                "script": str((frozen / setup_path.name).resolve()),
                "out_dir": str((out_dir / "views" / "meshops").resolve()),
                "clay": True,
            }
        )
    if meshy_view_dir is None and norm_path is not None:
        passes.append(
            {
                "kind": "stl",
                "path": str(norm_path.resolve()),
                "out_dir": str((out_dir / "views" / "meshy").resolve()),
                "clay": True,
            }
        )
    appearance_paths: list[str] = []
    if appearance and glb_path is not None:
        if blender_bin is None:
            messages.append("appearance preview skipped; Blender is absent")
        else:
            appearance_dir = out_dir / "appearance" / "meshy"
            passes.append(
                {
                    "kind": "glb",
                    "path": str(glb_path.resolve()),
                    "out_dir": str(appearance_dir.resolve()),
                    "clay": False,
                }
            )
            appearance_paths = [f"appearance/meshy/{role}.png" for role in ROLES]
    if passes:
        if blender_bin is None:
            raise ProportionError(
                "Blender 5.2 is required to render benchmark views",
                code="benchmark_render_unavailable",
            )
        stdout = _render_with_blender(out_dir, passes, blender_bin)
        counted = _recipe_count(stdout)
        if n_parts is None and counted is not None:
            n_parts = counted
        _require_roles(out_dir / "views" / "meshops", "rendered meshops")
        _require_roles(out_dir / "views" / "meshy", "rendered meshy")

    views: dict[str, RoleViews] = {}
    crops: list[CropRecord] = []
    for role in ROLES:
        ref_png = out_dir / "views" / "reference" / f"{role}.png"
        meshops_png = out_dir / "views" / "meshops" / f"{role}.png"
        meshy_png = out_dir / "views" / "meshy" / f"{role}.png"
        meshops_raw = score_aligned_masks(ref_png, meshops_png)
        meshy_raw = score_aligned_masks(ref_png, meshy_png)
        meshops_residual = out_dir / "residuals" / "meshops" / f"{role}.png"
        meshy_residual = out_dir / "residuals" / "meshy" / f"{role}.png"
        _write_residual(meshops_residual, meshops_raw["_ref_grid"], meshops_raw["_cand_grid"])
        _write_residual(meshy_residual, meshy_raw["_ref_grid"], meshy_raw["_cand_grid"])
        views[role] = RoleViews(
            meshops=MaskScore.model_validate(
                _score_payload(
                    meshops_raw,
                    meshops_residual.resolve().relative_to(out_dir.resolve()).as_posix(),
                )
            ),
            meshy=MaskScore.model_validate(
                _score_payload(
                    meshy_raw,
                    meshy_residual.resolve().relative_to(out_dir.resolve()).as_posix(),
                )
            ),
        )
        subjects = {
            "reference": (ref_png, meshops_raw["ref_content_bbox"]),
            "meshops": (meshops_png, meshops_raw["cand_content_bbox"]),
            "meshy": (meshy_png, meshy_raw["cand_content_bbox"]),
        }
        for name in CROP_NAMES:
            for subject, (image_path, bbox) in subjects.items():
                crops.append(
                    _crop_subject(
                        role=role,
                        name=name,
                        subject=subject,
                        image_path=image_path,
                        content_bbox=bbox,
                        out=out_dir,
                    )
                )

    if any(
        pair.meshops.mask_method == "corner_median" or pair.meshy.mask_method == "corner_median"
        for pair in views.values()
    ):
        messages.append(
            "near-white mask covered the frame; scored with the existing corner-median luma mask"
        )

    if meshy_stats is None:
        meshy_topology = TopologyContext(
            status="not_supplied",
            representation=None,
            mesh_stats=None,
            note=TOPOLOGY_NOTE,
        )
    else:
        meshy_topology = TopologyContext(
            status="context_only",
            representation="triangle_mesh",
            mesh_stats=meshy_stats,
            note=TOPOLOGY_NOTE,
        )
    capture = CaptureContract()
    ref_images = {role: _provenance(ref_dir / f"{role}.png") for role in ROLES}
    doc = MultiviewBenchmark(
        honesty=MULTIVIEW_BENCHMARK_HONESTY,
        ok=ok,
        status=status,
        visual_verdict=verdict if verdict in ("accept", "reject") else None,
        needs_user_input=needs_user,
        rank=None,
        height_m=height_m,
        yaw_deg=float(yaw_deg),
        capture=capture,
        reference=ReferenceRecord(
            checklist=_provenance(checklist_path),
            images=ref_images,
            height_m=height_m,
            figure=figure,
            multi_figure=bool(checklist.multi_figure),
            in_scope_figures=list(checklist.in_scope_figures),
        ),
        candidates=Candidates(
            meshops=CandidateRecord(
                representation="recipe_primitives",
                n_parts=n_parts,
                provenance=meshops_prov,
                normalize=None,
            ),
            meshy=CandidateRecord(
                representation="triangle_mesh" if meshy_stats is not None else None,
                n_parts=None,
                provenance=meshy_prov,
                normalize=normalize_record,
            ),
        ),
        views=views,
        crops=crops,
        topology_context=TopologyPair(
            meshops=TopologyContext(
                status="context_only",
                representation="recipe_primitives",
                mesh_stats=None,
                note=TOPOLOGY_NOTE,
            ),
            meshy=meshy_topology,
        ),
        messages=messages,
        paths=[],
        appearance_preview=appearance_paths,
    )
    contract_path = out_dir / CONTRACT_BASENAME
    md_path = out_dir / MARKDOWN_BASENAME
    contract_path.write_text(
        json.dumps(capture.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(md_path, doc, out_dir)
    written = [
        json_path.resolve(),
        md_path.resolve(),
        contract_path.resolve(),
    ]
    doc.paths = [str(item) for item in written]
    json_path.write_text(
        json.dumps(doc.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return doc.model_dump(mode="json")
