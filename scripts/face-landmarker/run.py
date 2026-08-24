#!/usr/bin/env python3
"""MeshOps 0130 — Face Landmarker sidecar (Python 3.12 + mediapipe).

Maps 478 Face Landmarker points → 0124 form-read ids and writes
assist_pixel_capture JSON for ``meshops proportion capture --source px``.

This script is intentionally isolated from the meshops 3.13 package.
Do not ``import meshops`` here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HONESTY = "face_landmarker_sidecar_not_mesh_or_print_success"
CAPTURE_HONESTY = "proportion_capture_not_mesh_or_print_success"
EXIT_OK = 0
EXIT_SKIP = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_map_path() -> Path:
    return _repo_root() / "src" / "meshops" / "proportion" / "face_landmarker_map.json"


def _default_model_path() -> Path | None:
    env = os.environ.get("MESHOPS_FACE_LANDMARKER_MODEL")
    if env:
        return Path(env)
    sibling = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
    if sibling.is_file():
        return sibling
    local = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "MeshOps"
        / "face-landmarker"
        / "face_landmarker.task"
    )
    if local.is_file():
        return local
    return sibling  # preferred create path (may be missing)


def _write_skip(out: Path | None, reason: str, **extra: Any) -> int:
    payload = {"ok": False, "skip": reason, "honesty": HONESTY, **extra}
    text = json.dumps(payload, indent=2) + "\n"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return EXIT_SKIP


def _as_xy(pt: Any) -> tuple[float, float] | None:
    x = getattr(pt, "x", None)
    y = getattr(pt, "y", None)
    if x is None and isinstance(pt, Mapping):
        x, y = pt.get("x"), pt.get("y")
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(xf) or not math.isfinite(yf):
        return None
    return xf, yf


def _index_spec_to_xy(pts: Sequence[Any], spec: Any) -> tuple[float, float] | None:
    if isinstance(spec, bool) or spec is None:
        return None
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        idx = int(spec)
        if idx < 0 or idx >= len(pts):
            return None
        return _as_xy(pts[idx])
    if isinstance(spec, Mapping) and "mean" in spec:
        idxs = spec.get("mean")
        if not isinstance(idxs, Sequence) or isinstance(idxs, (str, bytes)):
            return None
        xs: list[float] = []
        ys: list[float] = []
        for raw in idxs:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                return None
            if idx < 0 or idx >= len(pts):
                return None
            xy = _as_xy(pts[idx])
            if xy is None:
                return None
            xs.append(xy[0])
            ys.append(xy[1])
        if not xs:
            return None
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return None


def _resolve_entry(pts: Sequence[Any], entry: Mapping[str, Any]) -> tuple[float, float] | None:
    primary = entry.get("primary")
    if isinstance(primary, int) and primary >= 468 and len(pts) < 478:
        fb = entry.get("fallback")
        return _index_spec_to_xy(pts, fb) if fb is not None else None
    xy = _index_spec_to_xy(pts, primary)
    if xy is not None:
        return xy
    fb = entry.get("fallback")
    if fb is None:
        return None
    return _index_spec_to_xy(pts, fb)


def _to_px(x: float, y: float, width_px: int, height_px: int) -> tuple[float, float] | None:
    if width_px < 1 or height_px < 1:
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    x_px = x * float(width_px)
    y_px = y * float(height_px)
    if not math.isfinite(x_px) or not math.isfinite(y_px):
        return None
    x_px = min(max(x_px, 0.0), float(width_px - 1))
    y_px = min(max(y_px, 0.0), float(height_px - 1))
    return x_px, y_px


def _map_landmarks(
    pts: Sequence[Any],
    *,
    view: str,
    width_px: int,
    height_px: int,
    map_doc: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    landmarks = map_doc.get("landmarks") or {}
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(landmarks, Mapping):
        return out
    for lid, entry in landmarks.items():
        if not isinstance(entry, Mapping):
            continue
        if lid == "cranial_vertex":
            continue
        views = entry.get("views") or []
        if view not in views:
            continue
        xy = _resolve_entry(pts, entry)
        if xy is None:
            continue
        px = _to_px(xy[0], xy[1], width_px, height_px)
        if px is None:
            continue
        out[str(lid)] = {"x": float(px[0]), "y": float(px[1]), "method": "pose_model"}
    return out


def _facing(view: str) -> str:
    return {
        "front": "camera_front",
        "left": "camera_left",
        "three_quarter": "camera_three_quarter",
    }.get(view, "unknown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MeshOps Face Landmarker sidecar → assist_pixel_capture dump",
    )
    parser.add_argument("--image", required=True, type=Path, help="Input PNG/JPG path")
    parser.add_argument(
        "--view",
        required=True,
        choices=("front", "left", "three_quarter"),
        help="Assist view role",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path")
    parser.add_argument(
        "--map",
        type=Path,
        default=None,
        help="Override face_landmarker_map.json path",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override face_landmarker.task path",
    )
    parser.add_argument("--pose", default="unknown", help="Pose label for dump")
    args = parser.parse_args(argv)

    out: Path = args.out
    map_path = args.map if args.map is not None else _default_map_path()
    if not map_path.is_file():
        return _write_skip(out, "tool_missing", detail=f"map missing: {map_path}")

    try:
        map_doc = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _write_skip(out, "tool_missing", detail=f"map read failed: {exc}")

    model_path = args.model if args.model is not None else _default_model_path()
    if model_path is None or not Path(model_path).is_file():
        return _write_skip(
            out,
            "model_missing",
            detail=str(model_path) if model_path else "no model path",
        )

    try:
        import mediapipe as mp  # noqa: PLC0415 — optional 3.12-only dep
    except ImportError as exc:
        return _write_skip(out, "tool_missing", detail=f"mediapipe import failed: {exc}")

    image_path: Path = args.image
    if not image_path.is_file():
        return _write_skip(out, "bad_image", detail=f"missing image: {image_path}")

    try:
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        # Re-verify on installed wheel: Image.create_from_file lives on mp.Image
        mp_image = mp.Image.create_from_file(str(image_path))
        with FaceLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
    except Exception as exc:  # noqa: BLE001 — sidecar must skip, not crash MeshOps
        return _write_skip(out, "bad_image", detail=str(exc))

    faces = list(result.face_landmarks or [])
    n = len(faces)
    if n <= 0:
        return _write_skip(out, "no_face", n_faces=0)
    if n >= 2:
        return _write_skip(out, "multi_face", n_faces=n, multi_figure=True)

    arr = mp_image.numpy_view()
    height_px = int(arr.shape[0])
    width_px = int(arr.shape[1])
    pts = faces[0]
    mapped = _map_landmarks(
        pts,
        view=args.view,
        width_px=width_px,
        height_px=height_px,
        map_doc=map_doc,
    )
    dump = {
        "schema_version": "1.0.0",
        "kind": "assist_pixel_capture",
        "honesty": CAPTURE_HONESTY,
        "pose": args.pose,
        "multi_figure": False,
        "detector": "face_landmarker",
        "views": {
            args.view: {
                "width_px": width_px,
                "height_px": height_px,
                "facing_direction": _facing(args.view),
                "landmarks": mapped,
            }
        },
        "messages": [
            HONESTY,
            f"mapped {len(mapped)}/478 → 0124 ids (view={args.view})",
            "ingest with: meshops proportion capture --source px --merge --prefer-merge",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"ok": True, "out": str(out), "n_mapped": len(mapped)}) + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
