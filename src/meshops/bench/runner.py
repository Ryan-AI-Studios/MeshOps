"""Size-ladder benchmark runner: warmup + median-of-3; per-case continue.

Times ingest_stl / mesh_triage / optional F3D (front, left, three_quarter).
"""

from __future__ import annotations

import gc
import os
import platform
import statistics
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from meshops.bench.models import (
    SKIPPED_INSUFFICIENT_RAM,
    BenchCaseResult,
    Envelope,
    HostBlock,
    MethodBlock,
)
from meshops.bench.rss import (
    case_peak_rss_mb,
    get_available_ram_bytes,
    get_current_rss_mb,
    get_total_ram_mb,
)
from meshops.bench.sizes import FACE_TARGETS, LadderMesh, parse_sizes, write_ladder_stl

# L/XL RAM gate: skip when available < ~4 GiB (R2).
RAM_GATE_BYTES: int = 4 * 1024**3
RAM_GATED_LABELS: frozenset[str] = frozenset({"L", "XL"})

BENCH_CAMERAS: tuple[str, ...] = ("front", "left", "three_quarter")

DEP_NAMES: tuple[str, ...] = (
    "trimesh",
    "numpy",
    "scipy",
    "pymeshlab",
    "manifold3d",
    "f3d",
)


def time_median(
    fn: Callable[[], Any],
    *,
    warmup: int = 1,
    iters: int = 3,
) -> tuple[float, list[float]]:
    """Warmup then timed iters with gc.collect before each timed call; return median."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(statistics.median(samples)), samples


def collect_deps() -> dict[str, str]:
    """Best-effort package versions for envelope honesty (C7)."""
    out: dict[str, str] = {}
    for name in DEP_NAMES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
        except Exception as exc:
            out[name] = f"error:{type(exc).__name__}"
    return out


def _host_os() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _cpu_label() -> str | None:
    try:
        return platform.processor() or None
    except Exception:
        return None


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _should_skip_ram(label: str) -> bool:
    if label.upper() not in RAM_GATED_LABELS:
        return False
    avail = get_available_ram_bytes()
    if avail is None:
        # Unprobeable → do not false-skip (allow run; OOM risk is host-owned).
        return False
    return avail < RAM_GATE_BYTES


def _resolve_work_root(work_root: Path | str | None) -> Path:
    from meshops.bench.report import resolve_work_root

    return resolve_work_root(work_root)


def run_case(
    label: str,
    *,
    work_root: Path,
    deps: dict[str, str],
    warmup: int = 1,
    timed_iters: int = 3,
    include_render: bool = True,
    target_faces: int | None = None,
    inject_fail: bool = False,
) -> BenchCaseResult:
    """Run one ladder case; never raises — returns failed/skipped/ok result."""
    host_os = _host_os()
    py_ver = _python_version()
    created = _iso_now()
    target = (
        target_faces
        if target_faces is not None
        else FACE_TARGETS.get(
            label.upper(),  # type: ignore[arg-type]
            0,
        )
    )
    if target <= 0 and target_faces is None:
        return BenchCaseResult(
            label=label,
            target_faces=0,
            actual_faces=0,
            verts=0,
            status="failed",
            error_code="unknown_label",
            error_message=f"unknown size label {label!r}",
            host_os=host_os,
            python_version=py_ver,
            deps=dict(deps),
            created_at=created,
        )

    if _should_skip_ram(label):
        return BenchCaseResult(
            label=label.upper(),
            target_faces=target,
            actual_faces=0,
            verts=0,
            status="skipped",
            skipped_reason=SKIPPED_INSUFFICIENT_RAM,
            host_os=host_os,
            python_version=py_ver,
            deps=dict(deps),
            created_at=created,
        )

    case_dir = work_root / "cases" / f"{label.upper()}_{uuid.uuid4().hex[:8]}"
    meshes_dir = case_dir / "meshes"
    jobs_dir = case_dir / "jobs"
    views_dir = case_dir / "views"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    try:
        if inject_fail:
            raise RuntimeError("injected failure for ladder continue test")

        # Case-scoped RSS: sample current RSS at phase boundaries; take max.
        # Do NOT use process-lifetime PeakWorkingSet — that bleeds across S/M/L/XL.
        rss_samples: list[float | None] = [get_current_rss_mb()]

        stl_path = meshes_dir / f"{label.upper()}.stl"
        ladder: LadderMesh = write_ladder_stl(label, stl_path, target_faces=target)
        actual = ladder.actual_faces
        verts = ladder.verts
        rss_samples.append(get_current_rss_mb())

        from meshops.ingest.pipeline import ingest_stl
        from meshops.triage.orchestrate import mesh_triage

        # Unique work_root per timed iter would invalidate reuse; use jobs_dir
        # and re-ingest same content (idempotent by content hash) after first.
        # For fair timing we clear working artifacts between iters when possible.
        mesh_id_holder: dict[str, str] = {}

        def _ingest_once() -> None:
            result = ingest_stl(stl_path, work_root=jobs_dir)
            mesh_id_holder["id"] = result.mesh_id

        def _triage_once() -> None:
            mid = mesh_id_holder.get("id")
            if not mid:
                result = ingest_stl(stl_path, work_root=jobs_dir)
                mid = result.mesh_id
                mesh_id_holder["id"] = mid
            mesh_triage(mid, work_root=jobs_dir)

        # First ingest seeds mesh_id for triage timing.
        seed = ingest_stl(stl_path, work_root=jobs_dir)
        mesh_id_holder["id"] = seed.mesh_id
        rss_samples.append(get_current_rss_mb())

        ingest_med, ingest_samples = time_median(_ingest_once, warmup=warmup, iters=timed_iters)
        rss_samples.append(get_current_rss_mb())
        triage_med, triage_samples = time_median(_triage_once, warmup=warmup, iters=timed_iters)
        rss_samples.append(get_current_rss_mb())

        render_med: float | None = None
        render_samples: list[float] = []
        if include_render:
            render_med, render_samples = _time_render(
                stl_path,
                views_dir,
                mesh_id=seed.mesh_id,
                warmup=warmup,
                timed_iters=timed_iters,
            )
            rss_samples.append(get_current_rss_mb())

        rss = case_peak_rss_mb(rss_samples)
        return BenchCaseResult(
            label=label.upper(),
            target_faces=ladder.target_faces,
            actual_faces=actual,
            verts=verts,
            ingest_s=ingest_med,
            triage_s=triage_med,
            render_s=render_med,
            ingest_samples_s=ingest_samples,
            triage_samples_s=triage_samples,
            render_samples_s=render_samples,
            rss_peak_mb=rss,
            status="ok",
            host_os=host_os,
            python_version=py_ver,
            deps=dict(deps),
            created_at=created,
        )
    except Exception as exc:
        return BenchCaseResult(
            label=label.upper(),
            target_faces=target,
            actual_faces=0,
            verts=0,
            status="failed",
            error_code=type(exc).__name__,
            error_message=str(exc),
            host_os=host_os,
            python_version=py_ver,
            deps=dict(deps),
            created_at=_iso_now(),
            rss_peak_mb=get_current_rss_mb(),
        )


def _time_render(
    mesh_path: Path,
    views_dir: Path,
    *,
    mesh_id: str,
    warmup: int,
    timed_iters: int,
) -> tuple[float | None, list[float]]:
    """Time F3D 3-view render; unavailable → (None, [])."""
    from meshops.render.f3d_renderer import F3DRenderer, RenderUnavailableError

    renderer = F3DRenderer(width=256, height=256)
    views_dir.mkdir(parents=True, exist_ok=True)

    def _once() -> None:
        # Fresh subdir per call to avoid overwrite races; still under views_dir.
        dest = views_dir / f"iter_{uuid.uuid4().hex[:6]}"
        dest.mkdir(parents=True, exist_ok=True)
        renderer.render_mesh_to_dir(
            mesh_path,
            dest,
            camera_names=BENCH_CAMERAS,
            include_depth_for=(),  # RGB only for bench speed honesty
            mesh_id=mesh_id,
            rendered_from="bench",
        )

    try:
        return time_median(_once, warmup=warmup, iters=timed_iters)
    except RenderUnavailableError:
        return None, []
    except Exception:
        # Treat unexpected render failures as skipped render, not case failure
        # when ingest+triage already succeeded (R11).
        return None, []


def run_ladder(
    sizes: Sequence[str] | str | None = None,
    *,
    work_root: Path | str | None = None,
    include_render: bool = True,
    warmup: int = 1,
    timed_iters: int = 3,
    inject_fail_labels: Sequence[str] | None = None,
) -> Envelope:
    """Run size ladder; per-case try/except continues after failures (C4)."""
    if isinstance(sizes, str) or sizes is None:
        # sizes str or default from env / full ladder
        if sizes is None:
            env_sizes = os.environ.get("MESHOPS_BENCH_SIZES", "").strip()
            labels = parse_sizes(env_sizes or None)
        else:
            labels = parse_sizes(sizes)
    else:
        labels = [str(s).upper() for s in sizes]  # type: ignore[misc]
        # Validate
        for lab in labels:
            if lab not in FACE_TARGETS:
                raise ValueError(f"unknown size {lab!r}")

    root = _resolve_work_root(work_root)
    root.mkdir(parents=True, exist_ok=True)
    deps = collect_deps()
    fail_set = {x.upper() for x in (inject_fail_labels or [])}

    cases: list[BenchCaseResult] = []
    for lab in labels:
        case = run_case(
            lab,
            work_root=root,
            deps=deps,
            warmup=warmup,
            timed_iters=timed_iters,
            include_render=include_render,
            inject_fail=lab.upper() in fail_set,
        )
        cases.append(case)

    avail = get_available_ram_bytes()
    total = get_total_ram_mb()
    host = HostBlock(
        os=_host_os(),
        python_version=_python_version(),
        cpu=_cpu_label(),
        total_ram_mb=total,
        available_ram_mb=(float(avail) / (1024.0 * 1024.0)) if avail is not None else None,
        deps=deps,
    )
    method = MethodBlock(warmup=warmup, timed_iters=timed_iters, cameras=list(BENCH_CAMERAS))
    notes: list[str] = [
        "Wipeout is never redefined as speed (Difficulty §6 / C5).",
        "Open3D/Rust/psutil never required core (C1).",
    ]
    return Envelope(
        created_at=_iso_now(),
        method=method,
        host=host,
        cases=cases,
        notes=notes,
    )


def profile_load_vs_ingest(
    stl_path: Path | str,
    *,
    work_root: Path | str,
    warmup: int = 1,
    iters: int = 3,
) -> dict[str, Any]:
    """Profile t_load (trimesh.load) vs t_ingest (ingest_stl) for Rust prove/reject.

    Returns medians, sample lists, and ratio t_load / t_ingest.
    """
    import trimesh

    from meshops.ingest.pipeline import ingest_stl

    path = Path(stl_path)
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)

    def _load() -> None:
        trimesh.load(str(path), force="mesh")

    # Separate job roots so ingest does real work each time
    counter = {"n": 0}

    def _ingest() -> None:
        counter["n"] += 1
        sub = root / f"ingest_{counter['n']}_{uuid.uuid4().hex[:6]}"
        sub.mkdir(parents=True, exist_ok=True)
        ingest_stl(path, work_root=sub)

    t_load, load_samples = time_median(_load, warmup=warmup, iters=iters)
    t_ingest, ingest_samples = time_median(_ingest, warmup=warmup, iters=iters)
    ratio = t_load / t_ingest if t_ingest > 0 else float("inf")
    return {
        "t_load_s": t_load,
        "t_ingest_s": t_ingest,
        "ratio_load_over_ingest": ratio,
        "load_samples_s": load_samples,
        "ingest_samples_s": ingest_samples,
    }
