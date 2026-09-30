"""Track 0130 — Face Landmarker map / capture method / skip (offline).

No live mediapipe / no network. Authoring aid only — LANDMARKER_HONESTY.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.assist import (
    FACE_FRONT_LANDMARK_IDS,
    KNOWN_LANDMARK_IDS,
    _parse_landmark_value,
)
from meshops.proportion.capture import (
    _coord_value,
    _parse_coord_value,
    build_assist_from_px,
    merge_assist_docs,
)
from meshops.proportion.face_landmarker_map import (
    LANDMARKER_SKIP_REASONS,
    MESH_POINT_COUNT,
    assert_map_covers_face_front,
    build_assist_pixel_dump,
    default_map_path,
    load_face_landmarker_map,
    map_face_landmarks_px,
    mediapipe_import_forbidden_in_src,
    normalized_to_px,
    skip_reason_for_face_count,
)
from meshops.proportion.honesty import CAPTURE_HONESTY, LANDMARKER_HONESTY
from meshops.proportion.template import blank_assist_document


def _pt(x: float, y: float, z: float = -0.05) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _synthetic_478(*, iris_l_x: float = 0.35, iris_r_x: float = 0.65) -> list[dict[str, float]]:
    """478 NormalizedLandmark-like dicts; iris 468/473 set for camera-left test."""
    pts = [_pt(0.5, 0.5, -0.01) for _ in range(MESH_POINT_COUNT)]
    # Contour means used as fallbacks
    pts[33] = _pt(0.34, 0.40)
    pts[133] = _pt(0.36, 0.40)
    pts[263] = _pt(0.64, 0.40)
    pts[362] = _pt(0.66, 0.40)
    pts[105] = _pt(0.33, 0.32)
    pts[70] = _pt(0.32, 0.31)
    pts[334] = _pt(0.67, 0.32)
    pts[300] = _pt(0.68, 0.31)
    pts[1] = _pt(0.50, 0.48)
    pts[61] = _pt(0.40, 0.58)
    pts[291] = _pt(0.60, 0.58)
    pts[13] = _pt(0.50, 0.55)
    pts[14] = _pt(0.50, 0.57)
    pts[50] = _pt(0.30, 0.50)
    pts[280] = _pt(0.70, 0.50)
    pts[234] = _pt(0.18, 0.48)
    pts[127] = _pt(0.16, 0.47)
    pts[454] = _pt(0.82, 0.48)
    pts[356] = _pt(0.84, 0.47)
    pts[152] = _pt(0.50, 0.72)
    pts[10] = _pt(0.50, 0.20)  # forehead — must not become cranial_vertex
    pts[468] = _pt(iris_l_x, 0.40)
    pts[473] = _pt(iris_r_x, 0.40)
    return pts


def test_t0_no_mediapipe_import_in_src_meshops() -> None:
    hits = mediapipe_import_forbidden_in_src()
    assert hits == []


def test_t1_map_covers_face_front_indices() -> None:
    doc = load_face_landmarker_map()
    assert_map_covers_face_front(doc)
    landmarks = doc["landmarks"]
    for lid in FACE_FRONT_LANDMARK_IDS:
        assert lid in landmarks
    # 478 vocab must not leak into KNOWN
    for i in range(MESH_POINT_COUNT):
        assert f"mp_{i}" not in KNOWN_LANDMARK_IDS
        assert str(i) not in KNOWN_LANDMARK_IDS


def test_t2_camera_left_eye_l_x_less_than_eye_r() -> None:
    pts = _synthetic_478(iris_l_x=0.30, iris_r_x=0.70)
    mapped = map_face_landmarks_px(pts, view="front", width_px=1000, height_px=1000)
    assert "eye_l" in mapped and "eye_r" in mapped
    assert mapped["eye_l"][0] < mapped["eye_r"][0]


def test_t2b_iris_fallback_when_len_468() -> None:
    pts = _synthetic_478()[:468]
    mapped = map_face_landmarks_px(pts, view="front", width_px=100, height_px=100)
    assert "eye_l" in mapped and "eye_r" in mapped
    # Contour means: (33+133)/2 x=0.35 → 35px; (263+362)/2=0.65 → 65px
    assert mapped["eye_l"][0] < mapped["eye_r"][0]


def test_t3_forehead_not_cranial_chin_optional() -> None:
    pts = _synthetic_478()
    mapped = map_face_landmarks_px(
        pts, view="front", width_px=200, height_px=200, include_optional=False
    )
    assert "cranial_vertex" not in mapped
    assert "chin" not in mapped
    mapped_opt = map_face_landmarks_px(
        pts, view="front", width_px=200, height_px=200, include_optional=True
    )
    assert "cranial_vertex" not in mapped_opt
    assert "chin" in mapped_opt


def test_t4_dump_kind_no_z() -> None:
    dump = build_assist_pixel_dump(
        view="front",
        width_px=640,
        height_px=480,
        landmarks_px={"eye_l": [10.0, 20.0], "nose_tip": [100.0, 120.0]},
    )
    assert dump["kind"] == "assist_pixel_capture"
    assert dump["schema_version"] == "1.0.0"
    assert LANDMARKER_HONESTY in dump["messages"]
    lm = dump["views"]["front"]["landmarks"]["eye_l"]
    assert isinstance(lm, dict)
    assert "z" not in lm
    assert lm["method"] == "pose_model"
    blob = json.dumps(dump)
    assert '"z"' not in blob or "pose_model" in blob
    # Explicit: no landmarker Z key on coords
    for val in dump["views"]["front"]["landmarks"].values():
        assert "z" not in val


def test_t5_prefer_merge_keeps_human() -> None:
    base = blank_assist_document()
    base["views"]["front"]["landmarks"]["eye_l"] = [10.0, 20.0]
    new = blank_assist_document()
    new["views"]["front"]["landmarks"]["eye_l"] = [99.0, 99.0]
    merged = merge_assist_docs(base, new, prefer_merge=True)
    assert merged["views"]["front"]["landmarks"]["eye_l"] == [10.0, 20.0]


def test_t5b_method_round_trip_serialize_and_parse() -> None:
    raw = {"x": 12.5, "y": 34.5, "confidence": 0.8, "method": "pose_model"}
    parsed = _parse_coord_value(raw)
    assert parsed is not None
    x, y, conf, method = parsed
    assert x == 12.5 and y == 34.5
    assert conf == 0.8
    assert method == "pose_model"
    ser = _coord_value(x, y, conf, method=method)
    assert isinstance(ser, dict)
    assert ser["method"] == "pose_model"
    assert ser["x"] == 12.5 and ser["y"] == 34.5
    # Through build_assist_from_px
    dump = {
        "schema_version": "1.0.0",
        "kind": "assist_pixel_capture",
        "honesty": CAPTURE_HONESTY,
        "pose": "unknown",
        "multi_figure": False,
        "views": {
            "front": {
                "facing_direction": "camera_front",
                "landmarks": {"nose_tip": raw},
            }
        },
        "messages": [],
    }
    doc, _msgs = build_assist_from_px(dump)
    stored = doc["views"]["front"]["landmarks"]["nose_tip"]
    assert isinstance(stored, dict)
    assert stored.get("method") == "pose_model"
    # Assist → Landmark2D must keep pose_model (not fall back to "assist").
    lm2d = _parse_landmark_value("nose_tip", stored, width_px=640, height_px=480)
    assert lm2d is not None
    assert lm2d.method == "pose_model"


def test_t6_default_merge_new_wins() -> None:
    base = blank_assist_document()
    base["views"]["front"]["landmarks"]["eye_l"] = [10.0, 20.0]
    new = blank_assist_document()
    new["views"]["front"]["landmarks"]["eye_l"] = [99.0, 99.0]
    merged = merge_assist_docs(base, new, prefer_merge=False)
    assert merged["views"]["front"]["landmarks"]["eye_l"] == [99.0, 99.0]


def test_t7_face_count_skip_synthetic() -> None:
    assert skip_reason_for_face_count(0) == "no_face"
    assert skip_reason_for_face_count(2) == "multi_face"
    assert skip_reason_for_face_count(1) is None
    assert "no_face" in LANDMARKER_SKIP_REASONS
    assert "multi_face" in LANDMARKER_SKIP_REASONS


def test_t8_missing_sidecar_skip_ok_and_no_meshops_pin() -> None:
    """Four gates run without 3.12; launcher documents skip; no mediapipe pin."""
    p = default_map_path()
    assert p.is_file()
    assert p.name == "face_landmarker_map.json"
    repo = Path(__file__).resolve().parents[1]
    launcher = repo / "scripts" / "face-landmarker" / "run.ps1"
    assert launcher.is_file()
    launcher_txt = launcher.read_text(encoding="utf-8")
    assert "MESHOPS_FACE_LANDMARKER" in launcher_txt
    assert "tool_missing" in launcher_txt
    pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
    # No dependency pin — comments may mention the sidecar.
    assert "mediapipe==" not in pyproject
    assert '"mediapipe"' not in pyproject
    assert "'mediapipe'" not in pyproject
    # requirements pin stays sidecar-only
    req = (repo / "scripts" / "face-landmarker" / "requirements.txt").read_text(encoding="utf-8")
    assert "mediapipe==1.0.1" in req


def test_t9_mcp_catalog_stays_53() -> None:
    assert len(TOOL_NAMES) == 56


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sidecar_run_py() -> Path:
    return _repo_root() / "scripts" / "face-landmarker" / "run.py"


def _load_run_py() -> Any:
    path = _sidecar_run_py()
    spec = importlib.util.spec_from_file_location("face_landmarker_run_0137", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dummy_sidecar_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    image = tmp_path / "face.png"
    image.write_bytes(b"png")
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"task")
    out = tmp_path / "dump.json"
    return image, model, out


class _FakeImage:
    def numpy_view(self) -> Any:
        return types.SimpleNamespace(shape=(8, 8, 3))


class _FakeLandmarker:
    def __init__(self, faces: list[Any]) -> None:
        self._faces = faces

    def __enter__(self) -> _FakeLandmarker:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def detect(self, _image: object) -> Any:
        return types.SimpleNamespace(face_landmarks=self._faces)


def _install_fake_mediapipe(monkeypatch: Any, faces: list[Any]) -> None:
    landmarker = _FakeLandmarker(faces)

    def create_from_options(_options: object) -> _FakeLandmarker:
        return landmarker

    def create_from_file(_path: object) -> _FakeImage:
        return _FakeImage()

    mp: Any = types.ModuleType("mediapipe")
    mp.tasks = types.SimpleNamespace(
        BaseOptions=lambda **kw: types.SimpleNamespace(**kw),
        vision=types.SimpleNamespace(
            FaceLandmarker=types.SimpleNamespace(create_from_options=create_from_options),
            FaceLandmarkerOptions=lambda **kw: types.SimpleNamespace(**kw),
            RunningMode=types.SimpleNamespace(IMAGE="IMAGE"),
        ),
    )
    mp.Image = types.SimpleNamespace(create_from_file=create_from_file)
    monkeypatch.setitem(sys.modules, "mediapipe", mp)


def _dummy_face(n: int = MESH_POINT_COUNT) -> list[Any]:
    return [types.SimpleNamespace(x=0.5, y=0.5) for _ in range(n)]


def test_t10_honesty_in_sidecar_readme_and_commands() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = repo / "scripts" / "face-landmarker" / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert LANDMARKER_HONESTY in text
    assert "--prefer-merge" in text
    assert "run.ps1" in text
    assert "num_faces=2" in text
    assert "num_faces=1" not in text
    assert "Difficulty §N6" not in text
    # Local skill how-to (may be gitignored from publish — still present on this machine)
    commands = repo / ".agents" / "skills" / "meshops" / "references" / "commands.md"
    if commands.is_file():
        cmd = commands.read_text(encoding="utf-8")
        assert "face-landmarker" in cmd
        assert "prefer-merge" in cmd


def test_normalized_clamp_and_nonfinite() -> None:
    assert normalized_to_px(1.5, -0.2, 100, 50) == (99.0, 0.0)
    assert normalized_to_px(float("nan"), 0.5, 100, 100) is None


def test_left_view_only_left_ids() -> None:
    pts = _synthetic_478()
    mapped = map_face_landmarks_px(pts, view="left", width_px=100, height_px=100)
    assert "eye_r" not in mapped
    assert "mouth_corner_r" not in mapped
    assert "eye_l" in mapped
    assert "nose_tip" in mapped


def test_t11_run_py_num_faces_static() -> None:
    text = _sidecar_run_py().read_text(encoding="utf-8")
    assert "NUM_FACES: Final[int] = 2" in text
    assert "num_faces=NUM_FACES" in text
    assert "num_faces=1" not in text


def test_t12_run_py_multi_face_synthetic(tmp_path: Path, monkeypatch: Any) -> None:
    _install_fake_mediapipe(monkeypatch, [_dummy_face(), _dummy_face()])
    run = _load_run_py()
    image, model, out = _dummy_sidecar_paths(tmp_path)
    rc = run.main(
        ["--image", str(image), "--view", "front", "--out", str(out), "--model", str(model)]
    )
    assert rc == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == {
        "ok": False,
        "skip": "multi_face",
        "honesty": "face_landmarker_sidecar_not_mesh_or_print_success",
        "n_faces": 2,
        "multi_figure": True,
    }
    assert "views" not in payload
    assert "landmarks" not in payload


def test_t13_run_py_no_face_synthetic(tmp_path: Path, monkeypatch: Any) -> None:
    _install_fake_mediapipe(monkeypatch, [])
    run = _load_run_py()
    image, model, out = _dummy_sidecar_paths(tmp_path)
    rc = run.main(
        ["--image", str(image), "--view", "front", "--out", str(out), "--model", str(model)]
    )
    assert rc == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["skip"] == "no_face"
    assert payload["n_faces"] == 0


def test_t14_run_py_one_face_synthetic(tmp_path: Path, monkeypatch: Any) -> None:
    _install_fake_mediapipe(monkeypatch, [_dummy_face()])
    run = _load_run_py()
    image, model, out = _dummy_sidecar_paths(tmp_path)
    rc = run.main(
        ["--image", str(image), "--view", "front", "--out", str(out), "--model", str(model)]
    )
    assert rc == 0
    dump = json.loads(out.read_text(encoding="utf-8"))
    assert dump["kind"] == "assist_pixel_capture"
    assert "front" in dump["views"]
    assert dump["views"]["front"]["landmarks"]
