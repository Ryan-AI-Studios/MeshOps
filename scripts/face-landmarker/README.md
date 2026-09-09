# Face Landmarker sidecar (MeshOps 0130)

Optional **Python 3.12** runner that maps Google AI Edge Face Landmarker **478**
points → MeshOps **0124** form-read ids, then dumps `assist_pixel_capture` JSON
for existing `meshops proportion capture`.

**Honesty:** `face_landmarker_sidecar_not_mesh_or_print_success` — detector
pixels are **not** mesh or print success. Green landmarker ≠ identity, FACS,
biometrics, or photoreal. Blendshapes are off.

MeshOps stays on Python **3.13** and never imports `mediapipe`.

## Setup (skip-ok if missing)

```powershell
# Requires a local Python 3.12 interpreter (host may only have 3.13 — that is OK).
py -3.12 -m venv scripts\face-landmarker\.venv
.\scripts\face-landmarker\.venv\Scripts\python.exe -m pip install -r scripts\face-landmarker\requirements.txt

# Download the float16 task once (not fetched by pytest / CI):
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# Place at scripts/face-landmarker/models/face_landmarker.task
#   or %LOCALAPPDATA%\MeshOps\face-landmarker\face_landmarker.task
#   or set MESHOPS_FACE_LANDMARKER_MODEL
```

Env (optional, not doctor-required):

- `MESHOPS_FACE_LANDMARKER` — path to the 3.12 python.exe used by **`run.ps1`** (discovery only; `run.py` does not read this)
- `MESHOPS_FACE_LANDMARKER_MODEL` — path to `face_landmarker.task` (read by `run.py`)

Preferred entrypoint when 3.12 may be missing:

```powershell
# Emits {ok:false, skip:tool_missing} exit 2 if no 3.12 interpreter is found.
.\scripts\face-landmarker\run.ps1 --image <png> --view front --out <dump.json>
```

Direct `py -3.12 scripts\face-landmarker\run.py …` is fine when 3.12 is already on PATH.
MeshOps itself does **not** Popen this sidecar in v1 — operators run it, then `capture --prefer-merge`.

## Photo (Package A)

```powershell
.\scripts\face-landmarker\run.ps1 `
  --image views\proportion\front.png --view front `
  --out work\<job>\face_lm_photo_front.json
# (or: py -3.12 scripts\face-landmarker\run.py … when 3.12 is present)

meshops proportion capture --source px `
  --in work\<job>\face_lm_photo_front.json `
  --merge landmarks_assist.json `
  --out landmarks_assist.json `
  --prefer-merge --force
```

**Always pass `--prefer-merge`** so human fills win when both non-null.
`--force` is **file overwrite only** — it does **not** clobber landmark merge rules.

## Mannequin / Blender stills (same ids)

Prefer F3D job views first:

```powershell
py -3.12 scripts\face-landmarker\run.py `
  --image work\<job>\views\front.png --view front `
  --out work\<job>\face_lm_blender_front.json
```

Then left: `views\left.png` / `--view left`. Blender VIEW_3D / `render_d7` face
crops are fine when F3D views are missing — same `--view` roles.

Compare remains **0124** `blockout-face-compare` (photo vs RECIPE vs optional scene).

## Skip exits

Exit **2** with JSON `{ok:false, skip:…}` for `tool_missing` / `model_missing` /
`no_face` / `multi_face` / `bad_image`. Exit **0** when one face maps.

`num_faces=2` — detector cap is a max; if it returns 2+ faces, skip (`multi_face`) and do not pick a primary (§1). A second face below detection confidence can still yield n=1.

## Map SoT

Frozen table: `src/meshops/proportion/face_landmarker_map.json` (tested in 3.13).
Pass `--map` to override. Camera-left `_l` = smaller front `x_px` (person-right
MediaPipe indices). Landmarker **Z is discarded**.
