"""``run_doctor`` — compose discovery + package probes (no mesh mutation).

Default required set is ``{core}`` only (R3/C4). Missing Blender/Orca are
warnings + hints unless ``--require`` / ``--strict`` expand the set.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from meshops.ops.env_catalog import catalog_as_list, env_presence_map
from meshops.ops.models import (
    BlenderToolStatus,
    DiskInfo,
    DoctorReport,
    EnvCatalogItem,
    F3dToolStatus,
    NvidiaProbe,
    OrcaToolStatus,
    PackageStatus,
    PythonInfo,
    ToolsBlock,
    UvTooling,
    VramInfo,
)
from meshops.ops.sizes import approx_package_dir_mb

# Core packages that must import for default doctor ok.
CORE_PACKAGES: Final[tuple[str, ...]] = (
    "trimesh",
    "numpy",
    "scipy",
    "pydantic",
    "typer",
    "manifold3d",
    "pymeshlab",
    "f3d",
)

OPTIONAL_PACKAGES: Final[tuple[str, ...]] = ("build123d",)

REQUIRE_CHOICES: Final[frozenset[str]] = frozenset(
    {"core", "blender", "orca", "f3d", "design", "all"}
)

VRAM_RITUAL: Final = (
    "VRAM coexistence ritual (MeshOps §8): before Blender-heavy / organic days, "
    "unload local LLM runtimes sharing the GPU. LexBase ports: Gemma 8081, "
    "nomic 8083. Confirm free VRAM (nvidia-smi), run MeshOps Blender work, then "
    "reload LLMs. MeshOps never auto-kills LLM processes."
)

LICENSE_LINES: Final[tuple[str, ...]] = (
    "pymeshlab: GPL-3.0 **linked** (in-process; P2 commercial review if productize)",
    "Blender: GPL-2.0-or-later **subprocess** (not linked into meshops)",
    "OrcaSlicer: AGPL-3.0 **subprocess** (not linked into meshops)",
    "f3d/libf3d: BSD-3-Clause (PyPI wheel)",
)


def expand_require(require: Iterable[str] | None) -> set[str]:
    """Normalize ``--require`` tokens into a concrete check set."""
    if require is None:
        return {"core"}
    out: set[str] = set()
    for raw in require:
        token = raw.strip().lower()
        if not token:
            continue
        if token not in REQUIRE_CHOICES:
            raise ValueError(
                f"unknown require token {raw!r}; expected one of {sorted(REQUIRE_CHOICES)}"
            )
        if token == "all":
            out.update({"core", "blender", "orca", "design", "f3d"})
        else:
            out.add(token)
    if not out:
        out.add("core")
    # f3d is part of core; keeping it explicit is fine
    if "f3d" in out or "design" in out or "blender" in out or "orca" in out:
        out.add("core")
    return out


def _package_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_package(import_name: str, *, dist_name: str | None = None) -> PackageStatus:
    dist = dist_name or import_name
    try:
        importlib.import_module(import_name)
    except Exception as exc:
        return PackageStatus(
            import_ok=False,
            version=_package_version(dist),
            optional=import_name in OPTIONAL_PACKAGES,
            error=f"{type(exc).__name__}: {exc}",
        )
    return PackageStatus(
        import_ok=True,
        version=_package_version(dist),
        optional=import_name in OPTIONAL_PACKAGES,
    )


def _python_info() -> PythonInfo:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    pin_ok = sys.version_info.major == 3 and sys.version_info.minor == 13
    return PythonInfo(version=ver, executable=sys.executable, pin_ok=pin_ok)


def _probe_blender() -> BlenderToolStatus:
    from meshops.escalate.discover import find_blender_with_source
    from meshops.escalate.errors import EscalateError
    from meshops.escalate.version import require_blender_52

    path, source = find_blender_with_source(require=False)
    if path is None:
        return BlenderToolStatus(status="missing", source="missing")

    try:
        version = require_blender_52(path)
    except EscalateError as exc:
        if exc.code == "blender_version":
            # Soft-parse version from message details when present
            ver = None
            if isinstance(exc.details, dict):
                raw = exc.details.get("version")
                if isinstance(raw, str):
                    ver = raw
            return BlenderToolStatus(
                path=str(path),
                version=ver,
                pin_ok=False,
                status="version_mismatch",
                source=source if source != "missing" else "path",
            )
        return BlenderToolStatus(
            path=str(path),
            version=None,
            pin_ok=False,
            status="error",
            source=source if source != "missing" else "path",
        )
    except Exception:
        return BlenderToolStatus(
            path=str(path),
            version=None,
            pin_ok=False,
            status="error",
            source=source if source != "missing" else "path",
        )

    return BlenderToolStatus(
        path=str(path),
        version=version,
        pin_ok=True,
        status="ok",
        source=source if source != "missing" else "path",
    )


def _probe_orca() -> OrcaToolStatus:
    from meshops.slice.discover import (
        read_orca_version_from_appdata,
        soft_version_ok,
    )

    # Track source similarly to blender
    path, source = _find_orca_with_source()
    if path is None:
        return OrcaToolStatus(
            status="missing",
            source="missing",
            version_source="missing",
            soft_pin_ok=None,
        )

    app_ver = read_orca_version_from_appdata()
    if app_ver:
        version_source: str = "appdata"
        version = app_ver
        soft_ok = soft_version_ok(app_ver)
        status = "ok" if soft_ok else "warn"
    else:
        version_source = "path_only"
        version = None
        soft_ok = True  # unknown version: path present counts for orca require
        status = "warn"

    return OrcaToolStatus(
        path=str(path),
        version=version,
        soft_pin_ok=soft_ok,
        status=status,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        version_source=version_source,  # type: ignore[arg-type]
    )


def _find_orca_with_source() -> tuple[Path | None, str]:
    """Locate Orca with source label without forking discovery candidates."""
    import os as _os

    from meshops.slice.discover import (
        ENV_MESHOPS_ORCA,
        ENV_MESHOPS_ORCASLICER,
        WELL_KNOWN_WINDOWS_ORCA,
        find_orca,
    )

    path = find_orca(require=False)
    if path is None:
        return None, "missing"

    resolved = path.resolve(strict=False)
    for env_name in (ENV_MESHOPS_ORCA, ENV_MESHOPS_ORCASLICER):
        env = _os.environ.get(env_name, "").strip()
        if env:
            try:
                if Path(env).expanduser().resolve(strict=False) == resolved:
                    return path, "env"
            except OSError:
                if Path(env).expanduser() == path:
                    return path, "env"

    for name in ("orca-slicer", "orcaslicer"):
        which = shutil.which(name)
        if which:
            try:
                if Path(which).resolve(strict=False) == resolved:
                    return path, "path"
            except OSError:
                pass

    try:
        if WELL_KNOWN_WINDOWS_ORCA.resolve(strict=False) == resolved:
            return path, "well_known"
    except OSError:
        pass

    # Found but source ambiguous (e.g. same path as env and well_known)
    return path, "path"


def _probe_f3d() -> F3dToolStatus:
    try:
        import f3d  # type: ignore[import-untyped]

        ver = getattr(f3d, "__version__", None) or _package_version("f3d")
        return F3dToolStatus(import_ok=True, version=str(ver) if ver else _package_version("f3d"))
    except Exception:
        return F3dToolStatus(import_ok=False, version=_package_version("f3d"))


def _probe_uv(work_root: Path | None) -> UvTooling:
    version: str | None = None
    which = shutil.which("uv")
    if which:
        try:
            proc = subprocess.run(
                [which, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                shell=False,
            )
            if proc.returncode == 0:
                text = (proc.stdout or proc.stderr or "").strip()
                # e.g. "uv 0.12.1"
                m = re.search(r"(\d+\.\d+\.\d+\S*)", text)
                version = m.group(1) if m else text.split()[-1] if text else None
        except (OSError, subprocess.TimeoutExpired):
            version = None

    lock_present = False
    if work_root is not None:
        lock_present = (work_root / "uv.lock").is_file()
    else:
        # walk up from cwd a few levels
        cur = Path.cwd()
        for _ in range(4):
            if (cur / "uv.lock").is_file():
                lock_present = True
                break
            if cur.parent == cur:
                break
            cur = cur.parent

    return UvTooling(version=version, uv_lock_present=lock_present)


def _probe_pymeshlab_size() -> float | None:
    spec = importlib.util.find_spec("pymeshlab")
    if spec is None or not spec.origin:
        return None
    # origin is .../pymeshlab/__init__.py → package dir
    pkg_dir = Path(spec.origin).resolve().parent
    return approx_package_dir_mb(pkg_dir)


def _probe_nvidia() -> NvidiaProbe:
    which = shutil.which("nvidia-smi")
    if not which:
        return NvidiaProbe(status="no_nvidia_gpu")

    try:
        proc = subprocess.run(
            [
                which,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NvidiaProbe(status="probe_error", error=str(exc))

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:300]
        return NvidiaProbe(status="probe_error", error=err or f"exit {proc.returncode}")

    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return NvidiaProbe(status="probe_error", error="empty nvidia-smi output")

    # First GPU only
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 3:
        return NvidiaProbe(status="probe_error", error=f"unexpected csv: {line[0]!r}")

    name = parts[0]
    try:
        total = float(parts[1])
        free = float(parts[2])
    except ValueError:
        return NvidiaProbe(status="probe_error", error=f"bad numbers: {line[0]!r}")

    return NvidiaProbe(status="ok", name=name, free_mib=free, total_mib=total)


def _build_hints(
    *,
    python: PythonInfo,
    packages: dict[str, PackageStatus],
    blender: BlenderToolStatus,
    orca: OrcaToolStatus,
    f3d: F3dToolStatus,
    design_ok: bool,
) -> list[str]:
    hints: list[str] = []
    if not python.pin_ok:
        hints.append(
            "Install Python 3.13.x then recreate the env: "
            "py -3.13 -m venv .venv; .\\.venv\\Scripts\\Activate.ps1; uv sync --extra dev"
        )
    failed_core = [
        name for name in CORE_PACKAGES if name in packages and not packages[name].import_ok
    ]
    if failed_core or not f3d.import_ok:
        hints.append("Restore core deps from lockfile: uv sync --extra dev")
    if not design_ok:
        hints.append("Optional T7 design stack: uv sync --extra design")
    if blender.status == "missing":
        hints.append(
            "Install portable Blender 5.2 LTS (Difficulty §4 mirrors): "
            ".\\scripts\\bootstrap-tools.ps1"
        )
        hints.append(
            "Or set path: $env:MESHOPS_BLENDER = "
            "'C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe'"
        )
    elif blender.status == "version_mismatch":
        hints.append(
            "Blender found but not 5.2.x — install 5.2 LTS via "
            ".\\scripts\\bootstrap-tools.ps1 or set MESHOPS_BLENDER to 5.2 blender.exe"
        )
    if orca.status == "missing":
        hints.append(
            "Install OrcaSlicer 2.4.2 from "
            "https://github.com/OrcaSlicer/OrcaSlicer/releases/tag/v2.4.2 "
            "or Microsoft Store; then: "
            "$env:MESHOPS_ORCA = 'C:\\Program Files\\OrcaSlicer\\orca-slicer.exe'"
        )
        hints.append("Linux note: Flathub org.orcaslicer.OrcaSlicer may also provide a binary")
    elif orca.version_source == "path_only":
        hints.append(
            "Orca path found but AppData version unknown — open Orca once to write "
            "OrcaSlicer.conf, or ignore if soft pin is acceptable"
        )
    elif orca.soft_pin_ok is False:
        hints.append("Orca version older than soft pin 2.4 — upgrade to 2.4.2 from GitHub Releases")
    if not hints:
        hints.append("Core env healthy. For print/organic readiness: meshops doctor --strict")
    return hints


def run_doctor(
    *,
    require: set[str] | Iterable[str] | None = None,
    work_root: Path | None = None,
    cwd: Path | None = None,
) -> DoctorReport:
    """Aggregate tooling health. *require* defaults to ``{\"core\"}``.

    Parameters
    ----------
    require:
        Check set. Tokens: core, blender, orca, f3d, design, all.
    work_root / cwd:
        Directory used to look for ``uv.lock`` (repo root). Defaults to cwd.
    """
    req = expand_require(require)
    root = work_root if work_root is not None else cwd

    python = _python_info()
    packages: dict[str, PackageStatus] = {}
    for name in CORE_PACKAGES:
        packages[name] = _probe_package(name)
    for name in OPTIONAL_PACKAGES:
        packages[name] = _probe_package(name)

    f3d = _probe_f3d()
    # Align packages["f3d"] with tools.f3d when core probe ran
    if "f3d" in packages:
        packages["f3d"] = PackageStatus(
            import_ok=f3d.import_ok,
            version=f3d.version,
            optional=False,
            error=None if f3d.import_ok else packages["f3d"].error,
        )

    blender = _probe_blender()
    orca = _probe_orca()

    design_ok = packages.get("build123d", PackageStatus(import_ok=False)).import_ok

    pymeshlab_mb = _probe_pymeshlab_size()
    disk = DiskInfo(pymeshlab_approx_mb=pymeshlab_mb)

    tooling = _probe_uv(root)
    nvidia = _probe_nvidia()
    vram = VramInfo(ritual=VRAM_RITUAL, nvidia=nvidia)

    env_map = env_presence_map()
    catalog_items = [EnvCatalogItem(**e) for e in catalog_as_list()]

    hints = _build_hints(
        python=python,
        packages=packages,
        blender=blender,
        orca=orca,
        f3d=f3d,
        design_ok=design_ok,
    )

    # Evaluate required set
    core_packages_ok = all(packages[n].import_ok for n in CORE_PACKAGES)
    core_ok = python.pin_ok and core_packages_ok and f3d.import_ok

    failures: list[str] = []
    if "core" in req and not core_ok:
        if not python.pin_ok:
            failures.append("python_pin")
        if not core_packages_ok:
            failures.append("core_packages")
        if not f3d.import_ok:
            failures.append("f3d")
    if "f3d" in req and not f3d.import_ok:
        failures.append("f3d")
    if "blender" in req and (blender.status != "ok" or not blender.pin_ok):
        failures.append("blender")
    # Path present is enough; version_source path_only → warn only (R9)
    if "orca" in req and (orca.status == "missing" or orca.path is None):
        failures.append("orca")
    if "design" in req and not design_ok:
        failures.append("design")

    ok = len(failures) == 0
    notes: list[str] = []
    if orca.path and orca.version_source == "path_only" and "orca" in req:
        notes.append(
            "orca_version_source=path_only under require/strict — path accepted; "
            "AppData version not verified"
        )

    return DoctorReport(
        ok=ok,
        python=python,
        packages=packages,
        tools=ToolsBlock(blender=blender, orca=orca, f3d=f3d),
        tooling=tooling,
        disk=disk,
        licenses=list(LICENSE_LINES),
        env=env_map,
        env_catalog=catalog_items,
        hints=hints,
        vram=vram,
        required=sorted(req),
        notes=notes,
    )
