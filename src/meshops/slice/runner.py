"""Orca subprocess runner — list-args, shell=False, injectable for tests.

Never imports Orca internals. Never mutates mesh geometry (oracle only).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meshops.jobstore.paths import JobPaths, ensure_job_layout
from meshops.slice.anomaly import (
    AnomalyThresholds,
    evaluate_printability,
    mesh_volume_cm3_from_path,
)
from meshops.slice.discover import find_orca, read_orca_version_from_appdata, soft_version_ok
from meshops.slice.errors import SliceError
from meshops.slice.models import ParsedSliceStats, ProfilePaths, SliceRunResult
from meshops.slice.parse_3mf import (
    extract_plate_gcode_to,
    extract_slice_info_to,
    parse_gcode_3mf,
)
from meshops.slice.profiles import resolve_profiles
from meshops.slice.report import write_slice_report

ENV_ORCA_TIMEOUT = "MESHOPS_ORCA_TIMEOUT_S"
DEFAULT_TIMEOUT_S = 600.0

RunOrcaFn = Callable[..., subprocess.CompletedProcess[str]]


_RUN_ID_RE = re.compile(r"^run_\d{8}_\d{6}_[0-9a-f]{8}$")


def make_run_id(*, when: datetime | None = None) -> str:
    """``run_<UTC yyyymmdd_HHMMSS>_<8 hex>`` — concurrent-safe."""
    ts = when or datetime.now(UTC)
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(4)
    return f"run_{stamp}_{suffix}"


def validate_run_id(run_id: str) -> str:
    """Reject path traversal / non-canonical run ids (Codex P2-004)."""
    rid = (run_id or "").strip()
    if not _RUN_ID_RE.fullmatch(rid):
        raise SliceError(
            f"invalid run_id {run_id!r}; expected run_YYYYMMDD_HHMMSS_<8hex>",
            code="slice_failed",
            details={"run_id": run_id},
        )
    return rid


def build_orca_argv(
    *,
    orca: Path,
    input_stl: Path,
    output_3mf: Path,
    profiles: ProfilePaths,
    orient: int = 0,
    arrange: int = 0,
    plate: int = 1,
    datadir: Path | str | None = None,
) -> list[str]:
    """Build list-form argv for headless Orca slice (model path last).

    ``--load-settings`` is **one** element: ``"abs/machine;abs/process"``.
    Absolute ``--export-3mf`` — do **not** also set ``--outputdir``.
    """
    mach = Path(profiles.machine).resolve()
    proc = Path(profiles.process).resolve()
    fila = Path(profiles.filament).resolve()
    stl = Path(input_stl).resolve()
    out = Path(output_3mf).resolve()

    for label, p in (
        ("machine", mach),
        ("process", proc),
        ("filament", fila),
        ("input", stl),
    ):
        if not p.is_file():
            raise SliceError(
                f"{label} path not a file: {p}",
                code="profile_not_found" if label != "input" else "missing_candidate",
                details={"path": str(p)},
            )

    load_settings = f"{mach};{proc}"
    argv: list[str] = [str(Path(orca).resolve())]

    dd = datadir if datadir is not None else profiles.datadir
    if dd:
        argv.extend(["--datadir", str(Path(dd).resolve())])

    argv.extend(
        [
            "--orient",
            str(int(orient)),
            "--arrange",
            str(int(arrange)),
            "--load-settings",
            load_settings,
            "--load-filaments",
            str(fila),
            "--slice",
            str(int(plate)),
            "--export-3mf",
            str(out),
            str(stl),
        ]
    )
    return argv


def run_orca(
    argv: list[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    cwd: Path | str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke Orca with list-args, ``shell=False``, capture stdout/stderr."""
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            shell=False,
            cwd=str(cwd) if cwd is not None else None,
        )
    except FileNotFoundError as exc:
        raise SliceError(
            f"Orca binary not executable: {argv[0] if argv else '?'}",
            code="orca_not_found",
            details={"argv0": argv[0] if argv else None},
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SliceError(
            f"Orca slice timed out after {timeout_s}s",
            code="slice_timeout",
            details={"timeout_s": timeout_s, "argv_head": argv[:6]},
        ) from exc


def _timeout_from_env() -> float:
    raw = os.environ.get(ENV_ORCA_TIMEOUT, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_slice(
    candidate_path: Path | str,
    *,
    mesh_id: str | None = None,
    work_root: Path | str = "work",
    slice_profile: str | None = "default",
    orient: int = 0,
    arrange: int = 0,
    plate: int = 1,
    allow_reorient_retry: bool = False,
    mesh_volume_cm3: float | None = None,
    load_volume: bool = True,
    thresholds: AnomalyThresholds | None = None,
    orca_path: Path | str | None = None,
    run_orca_fn: RunOrcaFn | None = None,
    timeout_s: float | None = None,
    run_id: str | None = None,
) -> SliceRunResult:
    """Full oracle: discover → profiles → subprocess → parse → anomaly → report.

    Writes under ``work/<mesh_id>/slice/<run_id>/`` when mesh_id given;
    otherwise under ``<work_root>/_ad_hoc_slice/<run_id>/``.
    """
    started = _utc_now_iso()
    candidate = Path(candidate_path).resolve()
    if not candidate.is_file():
        raise SliceError(
            f"candidate mesh not found: {candidate}",
            code="missing_candidate",
            details={"path": str(candidate)},
        )

    orca = Path(orca_path).resolve() if orca_path else find_orca(require=False)
    if orca is None:
        raise SliceError(
            "OrcaSlicer not found (set MESHOPS_ORCA or install 2.4.x)",
            code="orca_not_found",
        )

    pre_version = read_orca_version_from_appdata()
    messages: list[str] = []
    if pre_version and not soft_version_ok(pre_version):
        messages.append(f"warning: Orca version {pre_version} is older than soft pin 2.4.x")

    profiles = resolve_profiles(slice_profile)
    rid = validate_run_id(run_id) if run_id else make_run_id()
    work_root_p = Path(work_root)

    if mesh_id:
        paths = JobPaths(work_root=work_root_p, mesh_id=mesh_id)
        ensure_job_layout(paths)
        base_slice = paths.slice_dir.resolve()
    else:
        base_slice = (work_root_p / "_ad_hoc_slice").resolve()
        base_slice.mkdir(parents=True, exist_ok=True)

    run_dir = (base_slice / rid).resolve()
    try:
        run_dir.relative_to(base_slice)
    except ValueError as exc:
        raise SliceError(
            f"run_id escapes slice root: {rid}",
            code="slice_failed",
            details={"run_id": rid, "slice_dir": str(base_slice)},
        ) from exc
    # Reject any pre-existing path (empty dir, non-empty dir, or file) — Codex P2-004.
    if run_dir.exists():
        raise SliceError(
            f"slice run_id path already exists: {rid}",
            code="slice_failed",
            details={"run_dir": str(run_dir)},
        )
    run_dir.mkdir(parents=True, exist_ok=False)

    # Preserve candidate suffix so post-promote working.ply is not mislabeled as .stl.
    # Orca accepts common mesh formats by extension; never mutate the source file.
    suffix = candidate.suffix.lower() if candidate.suffix else ".stl"
    if suffix not in {".stl", ".ply", ".obj", ".3mf"}:
        suffix = ".stl"
    input_mesh = run_dir / f"input{suffix}"
    if candidate.resolve() != input_mesh.resolve():
        shutil.copy2(candidate, input_mesh)
    # Keep alias name used in logs/docs when path is STL
    input_stl = input_mesh
    output_3mf = run_dir / "output.gcode.3mf"

    vol = mesh_volume_cm3
    if vol is None and load_volume:
        vol = mesh_volume_cm3_from_path(input_stl)

    invoker = run_orca_fn or run_orca
    timeout = timeout_s if timeout_s is not None else _timeout_from_env()

    def _one_pass(
        orient_flag: int,
    ) -> tuple[list[str], subprocess.CompletedProcess[str] | None, str | None]:
        argv = build_orca_argv(
            orca=orca,
            input_stl=input_stl,
            output_3mf=output_3mf,
            profiles=profiles,
            orient=orient_flag,
            arrange=arrange,
            plate=plate,
        )
        try:
            proc = invoker(argv, timeout_s=timeout, cwd=run_dir)
            return argv, proc, None
        except SliceError as exc:
            return argv, None, exc.code

    argv, proc, early_code = _one_pass(orient)

    # Optional single reorient retry when opted in (after first complete eval fails high)
    # We always write logs for the last invoke.

    def _write_logs(p: subprocess.CompletedProcess[str] | None) -> None:
        if p is None:
            return
        (run_dir / "orca_stdout.log").write_text(p.stdout or "", encoding="utf-8")
        (run_dir / "orca_stderr.log").write_text(p.stderr or "", encoding="utf-8")

    _write_logs(proc)

    result = _finalize_run(
        run_id=rid,
        mesh_id=mesh_id,
        candidate=candidate,
        input_stl=input_stl,
        output_3mf=output_3mf,
        run_dir=run_dir,
        profiles=profiles,
        orca=orca,
        pre_version=pre_version,
        argv=argv,
        proc=proc,
        early_code=early_code,
        vol=vol,
        thresholds=thresholds,
        messages=messages,
        started=started,
    )

    if (
        allow_reorient_retry
        and orient == 0
        and result.accept is not None
        and result.accept.status == "fail"
        and result.accept.error_code == "filament_anomaly_high"
    ):
        messages.append("allow_reorient_retry: re-invoking with --orient 1")
        # Archive first-pass artifacts so both runs are recorded (Codex P2-001).
        first_dir = run_dir / "attempt_orient0"
        first_dir.mkdir(exist_ok=True)
        for name in (
            "output.gcode.3mf",
            "orca_stdout.log",
            "orca_stderr.log",
            "manifest.json",
            "slice_report.md",
            "slice_info.config",
            "plate_1.gcode",
        ):
            src = run_dir / name
            if src.is_file():
                shutil.copy2(src, first_dir / name)
        # Snapshot first accept summary into metrics
        first_accept = result.accept
        first_summary = {
            "status": first_accept.status,
            "error_code": first_accept.error_code,
            "filament_used_cm3": first_accept.filament_used_cm3,
            "print_time_s": first_accept.print_time_s,
            "bed_overflow": first_accept.bed_overflow,
            "filament_ratio": first_accept.metrics.get("slice.filament_ratio"),
            "argv": list(result.argv),
            "returncode": result.returncode,
            "archive_dir": str(first_dir),
        }
        if output_3mf.is_file():
            output_3mf.unlink()
        argv2, proc2, early2 = _one_pass(1)
        _write_logs(proc2)
        result = _finalize_run(
            run_id=rid,
            mesh_id=mesh_id,
            candidate=candidate,
            input_stl=input_stl,
            output_3mf=output_3mf,
            run_dir=run_dir,
            profiles=profiles,
            orca=orca,
            pre_version=pre_version,
            argv=argv2,
            proc=proc2,
            early_code=early2,
            vol=vol,
            thresholds=thresholds,
            messages=[*messages, "reorient_retry_used=true"],
            started=started,
            extra_metrics={
                "slice.reorient_retry_used": True,
                "slice.reorient_first_attempt": first_summary,
            },
        )

    return result


def _finalize_run(
    *,
    run_id: str,
    mesh_id: str | None,
    candidate: Path,
    input_stl: Path,
    output_3mf: Path,
    run_dir: Path,
    profiles: ProfilePaths,
    orca: Path,
    pre_version: str | None,
    argv: list[str],
    proc: subprocess.CompletedProcess[str] | None,
    early_code: str | None,
    vol: float | None,
    thresholds: AnomalyThresholds | None,
    messages: list[str],
    started: str,
    extra_metrics: dict[str, Any] | None = None,
) -> SliceRunResult:
    finished = _utc_now_iso()
    metrics: dict[str, Any] = dict(extra_metrics or {})
    rc = proc.returncode if proc is not None else None
    metrics["slice.returncode"] = rc

    missing_3mf = not (output_3mf.is_file() and output_3mf.stat().st_size > 0)
    subprocess_ok = early_code is None and proc is not None and proc.returncode == 0

    if early_code == "slice_timeout":
        accept = evaluate_printability(
            ParsedSliceStats(parse_source="failed", messages=["timeout"]),
            vol,
            thresholds=thresholds,
            subprocess_ok=False,
            missing_3mf=True,
        )
        accept.error_code = "slice_timeout"
        accept.messages = [*list(accept.messages), "Orca slice timed out"]
        result = _build_result(
            run_id=run_id,
            mesh_id=mesh_id,
            candidate=candidate,
            output_3mf=output_3mf if not missing_3mf else None,
            run_dir=run_dir,
            profiles=profiles,
            orca=orca,
            orca_version=pre_version,
            accept=accept,
            metrics=metrics,
            messages=[*messages, "slice_timeout"],
            error_code="slice_timeout",
            started=started,
            finished=finished,
            argv=argv,
            returncode=rc,
            status="error",
        )
        _persist(result, run_dir, output_3mf if not missing_3mf else None)
        return result

    if early_code == "orca_not_found":
        from meshops.acceptance.models import SliceAcceptResult

        accept = SliceAcceptResult(
            status="fail",
            error_code="orca_not_found",
            messages=["Orca binary not executable"],
        )
        result = _build_result(
            run_id=run_id,
            mesh_id=mesh_id,
            candidate=candidate,
            output_3mf=None,
            run_dir=run_dir,
            profiles=profiles,
            orca=orca,
            orca_version=pre_version,
            accept=accept,
            metrics=metrics,
            messages=[*messages, "orca_not_found"],
            error_code="orca_not_found",
            started=started,
            finished=finished,
            argv=argv,
            returncode=rc,
            status="error",
        )
        _persist(result, run_dir, None)
        return result

    stats: ParsedSliceStats
    if missing_3mf:
        stats = ParsedSliceStats(
            parse_source="failed",
            messages=["missing or empty output.gcode.3mf"],
        )
        if proc is not None and proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-500:]
            stats.messages.append(f"orca exit {proc.returncode}: {tail}")
    else:
        stats = parse_gcode_3mf(output_3mf)
        extract_slice_info_to(output_3mf, run_dir / "slice_info.config")
        extract_plate_gcode_to(output_3mf, run_dir / "plate_1.gcode")

    orca_version = stats.orca_version or pre_version
    metrics.update(stats.metrics)

    # Hard-fail non-zero Orca exit regardless of parseable 3mf (Codex P1-001).
    accept = evaluate_printability(
        stats,
        vol,
        thresholds=thresholds,
        subprocess_ok=subprocess_ok,
        missing_3mf=missing_3mf,
    )

    # Prefer structured codes
    error_code = accept.error_code
    if accept.status == "fail" and missing_3mf and proc is not None and proc.returncode != 0:
        error_code = error_code or "slice_failed"
    if accept.status == "pass":
        status: str = "pass"
        error_code = None
    elif early_code:
        status = "error"
        error_code = early_code
    else:
        status = "fail"

    result = _build_result(
        run_id=run_id,
        mesh_id=mesh_id,
        candidate=candidate,
        output_3mf=output_3mf if not missing_3mf else None,
        run_dir=run_dir,
        profiles=profiles,
        orca=orca,
        orca_version=orca_version,
        accept=accept,
        metrics=metrics,
        messages=messages + list(stats.messages),
        error_code=error_code,
        started=started,
        finished=finished,
        argv=argv,
        returncode=rc,
        status=status,  # type: ignore[arg-type]
        plate_count=stats.plate_count,
    )
    _persist(result, run_dir, output_3mf if not missing_3mf else None)
    return result


def _build_result(
    *,
    run_id: str,
    mesh_id: str | None,
    candidate: Path,
    output_3mf: Path | None,
    run_dir: Path,
    profiles: ProfilePaths,
    orca: Path,
    orca_version: str | None,
    accept: Any,
    metrics: dict[str, Any],
    messages: list[str],
    error_code: str | None,
    started: str,
    finished: str,
    argv: list[str],
    returncode: int | None,
    status: str,
    plate_count: int = 0,
) -> SliceRunResult:
    return SliceRunResult(
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        mesh_id=mesh_id,
        candidate_path=str(candidate),
        output_3mf=str(output_3mf) if output_3mf else None,
        run_dir=str(run_dir),
        profile_paths=profiles,
        orca_path=str(orca),
        orca_version=orca_version,
        plate_count=plate_count or (accept.metrics.get("slice.plate_count", 0) if accept else 0),
        accept=accept,
        metrics=metrics,
        messages=messages,
        error_code=error_code,
        started_at=started,
        finished_at=finished,
        argv=argv,
        returncode=returncode,
    )


def _persist(
    result: SliceRunResult,
    run_dir: Path,
    output_3mf: Path | None,
) -> None:
    report = write_slice_report(run_dir, result)
    result.report_path = str(report)
    manifest = run_dir / "manifest.json"
    manifest.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    # re-write report with path now set
    result.report_path = str(report)
    write_slice_report(run_dir, result)
    manifest.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
