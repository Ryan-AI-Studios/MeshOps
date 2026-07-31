"""MeshOps-owned build123d export helpers (agent never writes files).

Isolates OCP/build123d imports for basedpyright and kernel failures.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from meshops.design.errors import DesignError
from meshops.design.models import DEFAULT_EXPORT_STEP_KWARGS, DEFAULT_EXPORT_STL_KWARGS


def _require_build123d() -> Any:
    try:
        import build123d as b123d
    except ImportError as exc:
        raise DesignError(
            "build123d is not installed; install with: uv sync --extra design "
            "(meshops[design] / build123d==0.11.1)",
            code="missing_dependency",
            details={"package": "build123d"},
        ) from exc
    return b123d


def ensure_single_solid(shape: Any) -> Any:
    """Fuse multi-solid compounds or fail multi_solid."""
    b123d = _require_build123d()

    solids: list[Any]
    try:
        if hasattr(shape, "solids"):
            solids = list(shape.solids())
        elif type(shape).__name__ in {"Solid", "Part"}:
            solids = [shape]
        else:
            solids = [shape]
    except Exception as exc:
        raise DesignError(
            f"failed to inspect solids: {exc}",
            code="cad_kernel_failure",
        ) from exc

    if len(solids) == 0:
        raise DesignError("shape has zero solids", code="multi_solid")
    if len(solids) == 1:
        return solids[0] if solids[0] is not None else shape

    # Fuse multi-solid compound (A1-BS5).
    try:
        fused = solids[0]
        for s in solids[1:]:
            fused = fused.fuse(s) if hasattr(fused, "fuse") else fused + s  # type: ignore[operator]
        # Prefer Part wrapper when available
        if hasattr(b123d, "Part") and not isinstance(fused, b123d.Part):
            with contextlib.suppress(Exception):
                fused = b123d.Part(fused)
        # Re-check solid count after fuse
        if hasattr(fused, "solids"):
            after = list(fused.solids())
            if len(after) != 1:
                raise DesignError(
                    f"fuse left {len(after)} solids (require 1)",
                    code="multi_solid",
                    details={"solid_count": len(after)},
                )
        return fused
    except DesignError:
        raise
    except Exception as exc:
        raise DesignError(
            f"multi-solid fuse failed: {exc}",
            code="multi_solid",
        ) from exc


def export_shape(
    shape: Any,
    *,
    stl_path: Path | str,
    step_path: Path | str,
    stl_kwargs: dict[str, Any] | None = None,
    step_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export solid to binary STL + STEP (Unit.MM). Require bool True and size > 0.

    Returns recorded export kwargs for DesignManifest.
    """
    b123d = _require_build123d()
    stl_p = Path(stl_path)
    step_p = Path(step_path)
    stl_p.parent.mkdir(parents=True, exist_ok=True)
    step_p.parent.mkdir(parents=True, exist_ok=True)

    solid = ensure_single_solid(shape)

    stl_kw = dict(DEFAULT_EXPORT_STL_KWARGS)
    if stl_kwargs:
        stl_kw.update(stl_kwargs)
    stl_kw["ascii_format"] = False  # binary STL only

    step_kw_raw = dict(DEFAULT_EXPORT_STEP_KWARGS)
    if step_kwargs:
        step_kw_raw.update(step_kwargs)

    # Map unit / precision_mode strings → build123d enums
    unit_val = step_kw_raw.get("unit", "MM")
    if isinstance(unit_val, str):
        unit_enum = getattr(b123d.Unit, unit_val, b123d.Unit.MM)
    else:
        unit_enum = unit_val

    pm_val = step_kw_raw.get("precision_mode", "AVERAGE")
    if isinstance(pm_val, str):
        pm_enum = getattr(b123d.PrecisionMode, pm_val, b123d.PrecisionMode.AVERAGE)
    else:
        pm_enum = pm_val

    step_call_kw: dict[str, Any] = {
        "unit": unit_enum,
        "write_pcurves": bool(step_kw_raw.get("write_pcurves", True)),
        "precision_mode": pm_enum,
    }

    try:
        ok_stl = b123d.export_stl(
            solid,
            str(stl_p),
            tolerance=float(stl_kw.get("tolerance", 0.001)),
            angular_tolerance=float(stl_kw.get("angular_tolerance", 0.1)),
            ascii_format=False,
        )
        ok_step = b123d.export_step(solid, str(step_p), **step_call_kw)
    except DesignError:
        raise
    except RuntimeError as exc:
        raise DesignError(
            f"CAD kernel export RuntimeError: {exc}",
            code="cad_kernel_failure",
        ) from exc
    except Exception as exc:
        # OCP may raise non-RuntimeError native failures
        raise DesignError(
            f"CAD kernel export failed: {type(exc).__name__}: {exc}",
            code="cad_kernel_failure",
        ) from exc

    if ok_stl is not True:
        raise DesignError(
            f"export_stl returned {ok_stl!r} (require True)",
            code="export_failed",
            details={"export": "stl", "return": ok_stl},
        )
    if ok_step is not True:
        raise DesignError(
            f"export_step returned {ok_step!r} (require True)",
            code="export_failed",
            details={"export": "step", "return": ok_step},
        )
    if not stl_p.is_file() or stl_p.stat().st_size <= 0:
        raise DesignError(
            f"STL missing or empty after export: {stl_p}",
            code="export_failed",
            details={"path": str(stl_p)},
        )
    if not step_p.is_file() or step_p.stat().st_size <= 0:
        raise DesignError(
            f"STEP missing or empty after export: {step_p}",
            code="export_failed",
            details={"path": str(step_p)},
        )

    return {
        "export_stl": {
            "tolerance": float(stl_kw.get("tolerance", 0.001)),
            "angular_tolerance": float(stl_kw.get("angular_tolerance", 0.1)),
            "ascii_format": False,
        },
        "export_step": {
            "unit": "MM",
            "write_pcurves": bool(step_call_kw.get("write_pcurves", True)),
            "precision_mode": (
                pm_val if isinstance(pm_val, str) else getattr(pm_val, "name", str(pm_val))
            ),
        },
        "stl_bytes": stl_p.stat().st_size,
        "step_bytes": step_p.stat().st_size,
    }
