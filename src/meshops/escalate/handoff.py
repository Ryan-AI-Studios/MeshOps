"""Build Blender 5.2 LTS handoff package (.blend + instructions + meta).

Subprocess boundary only — GPL Blender; no bpy in meshops venv.
Difficulty §4 discovery; §10 sculpt tips; §1 laterality doc; N1/N2/N6.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from meshops.escalate.discover import find_blender
from meshops.escalate.errors import EscalateError
from meshops.escalate.models import HandoffManifest
from meshops.escalate.roi import load_roi
from meshops.escalate.version import require_blender_52
from meshops.jobstore.paths import JobPaths, ensure_job_layout

DEFAULT_TIMEOUT_S = 300.0
VERTEX_GROUP = "meshops_roi"
DEFAULT_CAMERAS: tuple[str, ...] = ("front", "three_quarter", "top", "waist_zoom")
TEMPLATE_SCRIPT = Path(__file__).resolve().parent / "scripts" / "build_handoff.py"
PROBE_SCRIPT = Path(__file__).resolve().parent / "scripts" / "probe_handoff.py"
_ROI_VERTS_RE = re.compile(r"roi_verts=(\d+)")

INSTRUCTIONS_TEMPLATE = """# MeshOps Blender Handoff — Sculpt Instructions

**Mesh ID:** `{mesh_id}`
**ROI ID:** `{roi_id}`
**Vertex group:** `{vertex_group}` (select in Edit/Sculpt to bootstrap ROI)
**Blender:** {blender_version}
**Generated:** {created_at}

## Honesty ceiling (Difficulty §13 / N6)

This package prepares the operating room. MeshOps does **not** claim the organic
hero limb is fixed or print-ready. After sculpt, export STL and run:

```text
meshops escalate import-sculpt --mesh-id {mesh_id} --path sculpt.stl --approve
```

## Difficulty §10 — recommended sculpt tools

1. Select vertex group **`{vertex_group}`** → Edit Mode selection, then Sculpt.
2. Prefer **local** operations on the ROI (Difficulty §3) — not whole-model filters.
3. Tools that usually work for sheet→volume limbs:
   - **Inflate** — push sheet toward thickness
   - **Grab** — re-route limb silhouette / hand placement
   - **Clay Strips** — build tubular mass
   - **Smooth** — blend into clothing/body join
4. Iterate front ortho + three-quarter + waist zoom cameras (already placed).

## Hard refuse (Never / N1 / N2 / N8)

| Refuse | Why |
|---|---|
| Whole-model **voxel remesh** as "fix" | N1 / Difficulty §8 — destroys hero detail |
| **Full-mesh boolean after solidify** | N2 / Difficulty §6 — wipeout on organics |
| **Linked-flat auto-delete** of organic sheets | N8 / Difficulty §7 — false positives |
| Claiming "autonomous hero arm fixed" from scripts alone | N6 |

## Laterality (Difficulty §1)

- Camera names **left/right** are **image** left/right, **not** anatomical L/R.
- Multi-figure: confirm character identity + which limb before destructive sculpt.
- MeshOps never prompts stdin for laterality; if uncertain, re-check renders.

## Cameras

Placed cameras match MeshOps bbox conventions (front, three_quarter, top,
waist_zoom). Up-axis assumption: **+Z up** (MeshOps F3D / bbox_cameras).

## Export back to MeshOps

1. File → Export → STL (binary preferred).
2. `meshops escalate import-sculpt --mesh-id {mesh_id} --path <export.stl> --approve`
3. Acceptance uses **sculpt** guard tier (export-like floors + wipeout); views required.
4. Do **not** auto-promote preview packages; only approved sculpt revs may be accepted.

## ROI AABB (world)

- min: `{bbox_min}`
- max: `{bbox_max}`
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_roi_vert_count(stdout: str) -> int | None:
    """Extract ``roi_verts=N`` from build_handoff.py stdout (last match wins)."""
    matches = _ROI_VERTS_RE.findall(stdout or "")
    if not matches:
        return None
    return int(matches[-1])


def probe_handoff_blend(
    blend_path: Path | str,
    *,
    blender: Path | str,
    vertex_group: str = VERTEX_GROUP,
    cameras: list[str] | tuple[str, ...] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> None:
    """Open ``handoff.blend`` in Blender and assert mesh + VG + cameras exist.

    Raises EscalateError(code=blender_failed) when the probe exits non-zero.
    """
    blend = Path(blend_path)
    if not blend.is_file():
        raise EscalateError(
            f"probe: blend missing: {blend}",
            code="blender_failed",
        )
    if not PROBE_SCRIPT.is_file():
        raise EscalateError(
            f"probe script missing: {PROBE_SCRIPT}",
            code="blender_failed",
        )
    cam_list = list(cameras) if cameras is not None else list(DEFAULT_CAMERAS)
    cmd = [
        str(blender),
        "-b",
        str(blend.resolve()),
        "-P",
        str(PROBE_SCRIPT.resolve()),
        "--",
        "--vg",
        vertex_group,
        "--cameras",
        ",".join(cam_list),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EscalateError(
            f"Blender handoff probe timed out after {timeout_s}s",
            code="timeout",
            details={"cmd": cmd, "timeout_s": timeout_s},
        ) from exc
    except FileNotFoundError as exc:
        raise EscalateError(
            f"Blender binary disappeared during probe: {blender}",
            code="blender_missing",
        ) from exc

    if proc.returncode != 0:
        raise EscalateError(
            f"Blender handoff probe failed (exit {proc.returncode})",
            code="blender_failed",
            details={
                "returncode": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-2000:],
                "stderr_tail": (proc.stderr or "")[-2000:],
            },
        )


def build_handoff(
    mesh_id: str,
    roi_id: str,
    *,
    work_root: Path | str = "work",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> HandoffManifest:
    """Run headless Blender to produce ``handoff/handoff.blend`` + instructions + meta."""
    work_root_p = Path(work_root)
    paths = JobPaths(work_root=work_root_p, mesh_id=mesh_id)
    if not paths.job_dir.is_dir():
        raise EscalateError(
            f"Job directory not found: {paths.job_dir}",
            code="job_not_found",
        )
    ensure_job_layout(paths)

    if not paths.original_stl.is_file():
        raise EscalateError(
            f"original.stl missing: {paths.original_stl}",
            code="missing_mesh",
        )

    roi = load_roi(mesh_id, roi_id, work_root=work_root_p)
    blender = find_blender(require=True)
    assert blender is not None
    version = require_blender_52(blender)

    if not TEMPLATE_SCRIPT.is_file():
        raise EscalateError(
            f"handoff template missing: {TEMPLATE_SCRIPT}",
            code="blender_failed",
        )

    handoff_dir = paths.handoff_dir
    handoff_dir.mkdir(parents=True, exist_ok=True)

    # Copy script into handoff for inspectability
    script_dest = handoff_dir / "build_handoff.py"
    shutil.copy2(TEMPLATE_SCRIPT, script_dest)

    # ROI JSON for Blender script (include mesh bbox for camera framing)
    roi_payload = {
        "roi_id": roi.roi_id,
        "mesh_id": mesh_id,
        "bbox_min": list(roi.bbox_min),
        "bbox_max": list(roi.bbox_max),
        "vertex_group": VERTEX_GROUP,
        "mesh_bbox_min": None,
        "mesh_bbox_max": None,
    }
    # Prefer diagnostics / original stats for full-mesh cameras
    if paths.diagnostics_json.is_file():
        try:
            from meshops.models.diagnostics import Diagnostics

            diag = Diagnostics.model_validate_json(
                paths.diagnostics_json.read_text(encoding="utf-8")
            )
            roi_payload["mesh_bbox_min"] = list(diag.stats.bbox_min)
            roi_payload["mesh_bbox_max"] = list(diag.stats.bbox_max)
        except Exception:
            pass
    if roi_payload["mesh_bbox_min"] is None:
        try:
            from meshops.ingest.stats import load_mesh

            m = load_mesh(paths.original_stl)
            roi_payload["mesh_bbox_min"] = [float(x) for x in m.bounds[0]]
            roi_payload["mesh_bbox_max"] = [float(x) for x in m.bounds[1]]
        except Exception:
            roi_payload["mesh_bbox_min"] = list(roi.bbox_min)
            roi_payload["mesh_bbox_max"] = list(roi.bbox_max)

    roi_json_path = handoff_dir / "roi.json"
    import json

    roi_json_path.write_text(json.dumps(roi_payload, indent=2) + "\n", encoding="utf-8")

    blend_path = handoff_dir / "handoff.blend"
    if blend_path.is_file():
        blend_path.unlink()

    cmd = [
        str(blender),
        "-b",
        "-P",
        str(script_dest),
        "--",
        "--mesh",
        str(paths.original_stl.resolve()),
        "--roi-json",
        str(roi_json_path.resolve()),
        "--out",
        str(blend_path.resolve()),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EscalateError(
            f"Blender handoff timed out after {timeout_s}s",
            code="timeout",
            details={"cmd": cmd, "timeout_s": timeout_s},
        ) from exc
    except FileNotFoundError as exc:
        raise EscalateError(
            f"Blender binary disappeared: {blender}",
            code="blender_missing",
        ) from exc

    if proc.returncode != 0:
        raise EscalateError(
            f"Blender handoff failed (exit {proc.returncode})",
            code="blender_failed",
            details={
                "returncode": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-2000:],
                "stderr_tail": (proc.stderr or "")[-2000:],
            },
        )

    if not blend_path.is_file() or blend_path.stat().st_size <= 0:
        raise EscalateError(
            f"handoff.blend missing or empty after Blender run: {blend_path}",
            code="blender_failed",
            details={
                "stdout_tail": (proc.stdout or "")[-2000:],
                "stderr_tail": (proc.stderr or "")[-2000:],
            },
        )

    roi_vert_count = _parse_roi_vert_count(proc.stdout or "")

    # Second pass: prove mesh + meshops_roi VG + cameras inside the .blend
    probe_handoff_blend(
        blend_path,
        blender=blender,
        vertex_group=VERTEX_GROUP,
        cameras=DEFAULT_CAMERAS,
        timeout_s=timeout_s,
    )

    created = _now_iso()
    instructions_path = handoff_dir / "instructions.md"
    instructions_path.write_text(
        INSTRUCTIONS_TEMPLATE.format(
            mesh_id=mesh_id,
            roi_id=roi_id,
            vertex_group=VERTEX_GROUP,
            blender_version=version,
            created_at=created,
            bbox_min=list(roi.bbox_min),
            bbox_max=list(roi.bbox_max),
        ),
        encoding="utf-8",
    )

    cameras = list(DEFAULT_CAMERAS)
    notes = [
        "handoff_package",
        "not_autonomous_fixed",
        f"blender={version}",
        f"python={sys.version.split()[0]}",
    ]
    if any("needs_user_input" in n for n in roi.notes):
        notes.append("roi_notes_include_needs_user_input")
    # Empty ROI is a usability warning — package still ok (sculpt can re-select)
    if roi_vert_count == 0:
        notes.append("empty_roi_vertex_group")
        notes.append("roi_verts=0_warning")

    manifest = HandoffManifest(
        mesh_id=mesh_id,
        roi_id=roi_id,
        blender_path=str(blender),
        blender_version=version,
        blend_path=str(blend_path),
        instructions_path=str(instructions_path),
        created_at=created,
        vertex_group=VERTEX_GROUP,
        cameras=cameras,
        notes=notes,
        timeout_s=timeout_s,
        roi_vert_count=roi_vert_count,
    )
    meta_path = handoff_dir / "meta.json"
    meta_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest
