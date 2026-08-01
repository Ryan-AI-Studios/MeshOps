"""0011 SliceAcceptHook adapter — Orca body (track 0005).

Every call re-checks ``find_orca()`` → structured fail ``orca_not_found`` (never raise).
Loads candidate for watertight volume when path present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meshops.acceptance.hooks import SliceAcceptHook
from meshops.acceptance.models import SliceAcceptResult
from meshops.slice.anomaly import AnomalyThresholds, mesh_volume_cm3_from_path
from meshops.slice.discover import find_orca
from meshops.slice.runner import RunOrcaFn, run_slice


def make_orca_hook(
    *,
    mesh_id: str | None = None,
    work_root: Path | str | None = None,
    default_profile: str = "default",
    allow_reorient_retry: bool = False,
    thresholds: AnomalyThresholds | None = None,
    run_orca_fn: RunOrcaFn | None = None,
    orca_path: Path | str | None = None,
) -> SliceAcceptHook:
    """Build a ``SliceAcceptHook`` that runs the Orca printability oracle.

    Prefer structured ``SliceAcceptResult(status="fail", …)`` over exceptions.
    """
    root = Path(work_root) if work_root is not None else Path("work")

    def _hook(
        *,
        candidate_path: str | None = None,
        slice_profile: str | None = None,
        **kwargs: object,
    ) -> SliceAcceptResult:
        # 1. Re-find Orca every call — fail closed, do not raise
        orca = Path(orca_path) if orca_path is not None else find_orca(require=False)
        if orca is None or not Path(orca).is_file():
            return SliceAcceptResult(
                status="fail",
                error_code="orca_not_found",
                messages=[
                    "OrcaSlicer not found at hook invoke (set MESHOPS_ORCA or install 2.4.x)"
                ],
                metrics={"slice.hook": "make_orca_hook"},
            )

        if not candidate_path:
            return SliceAcceptResult(
                status="fail",
                error_code="missing_candidate",
                messages=["slice hook requires candidate_path"],
                metrics={"slice.hook": "make_orca_hook"},
            )

        cand = Path(str(candidate_path))
        if not cand.is_file():
            return SliceAcceptResult(
                status="fail",
                error_code="missing_candidate",
                messages=[f"candidate not found: {cand}"],
                metrics={"slice.hook": "make_orca_hook"},
            )

        # mid may be overridden via kwargs
        mid: str | None = mesh_id
        kw_mid = kwargs.get("mesh_id")
        if isinstance(kw_mid, str) and kw_mid:
            mid = kw_mid

        profile = slice_profile or default_profile
        vol = mesh_volume_cm3_from_path(cand)

        try:
            result = run_slice(
                cand,
                mesh_id=mid,
                work_root=root,
                slice_profile=profile,
                allow_reorient_retry=allow_reorient_retry,
                mesh_volume_cm3=vol,
                load_volume=False,  # already loaded
                thresholds=thresholds,
                orca_path=orca,
                run_orca_fn=run_orca_fn,
            )
        except Exception as exc:
            code = getattr(exc, "code", None) or "slice_failed"
            return SliceAcceptResult(
                status="fail",
                error_code=str(code),
                messages=[f"{type(exc).__name__}: {exc}"],
                metrics={"slice.hook": "make_orca_hook"},
            )

        if result.accept is not None:
            accept = result.accept
            # Namespace pack-facing metrics
            metrics: dict[str, Any] = dict(accept.metrics)
            metrics["slice.run_id"] = result.run_id
            if result.run_dir:
                metrics["slice.run_dir"] = result.run_dir
            if result.report_path:
                metrics["slice.report_path"] = result.report_path
            if result.orca_version:
                metrics["slice.orca_version"] = result.orca_version
            return SliceAcceptResult(
                status=accept.status,
                filament_used_cm3=accept.filament_used_cm3,
                print_time_s=accept.print_time_s,
                bed_overflow=accept.bed_overflow,
                error_code=accept.error_code,
                messages=list(accept.messages),
                metrics=metrics,
            )

        return SliceAcceptResult(
            status="fail",
            error_code=result.error_code or "slice_failed",
            messages=list(result.messages) or ["slice run produced no accept result"],
            metrics={"slice.run_id": result.run_id, "slice.hook": "make_orca_hook"},
        )

    return _hook
