"""Adapter-only error helpers — no engine error classes (R16).

Business failures raise ordinary exceptions / engine ``*Error`` subclasses.
Never use ``MCPError`` for business failures (R1 / C10).
"""

from __future__ import annotations

from typing import Any


def raise_if_not_ok(
    result: Any,
    *,
    what: str,
    ok_attr: str = "ok",
    failed_attr: str = "failed",
    codes_attr: str | None = None,
) -> None:
    """Raise ``RuntimeError`` when an engine result object reports failure.

    Used for accept / design / slice / finalize / import_sculpt success-path
    truth: if the engine returns without raising but ``.ok`` is False, surface
    that as a tool error so the MCP host sees ``is_error=True``.
    """
    ok = getattr(result, ok_attr, None)
    if ok is True:
        return
    if ok is None and isinstance(result, dict):
        if result.get(ok_attr, True) is True:
            return
        failed = result.get(failed_attr) or result.get("error_code") or result.get("failed")
        raise RuntimeError(f"{what} failed: ok=false failed={failed!r}")

    failed = getattr(result, failed_attr, None)
    codes: Any = None
    if codes_attr is not None:
        codes = getattr(result, codes_attr, None)
    parts = [f"{what} failed: ok={ok!r}"]
    if failed is not None:
        parts.append(f"failed={failed!r}")
    if codes is not None:
        parts.append(f"codes={codes!r}")
    err_code = getattr(result, "error_code", None)
    if err_code is not None:
        parts.append(f"error_code={err_code!r}")
    raise RuntimeError("; ".join(parts))


def raise_if_slice_not_pass(result: Any) -> None:
    """Raise when slice run is not a full pass (CLI exits 1)."""
    accept = getattr(result, "accept", None)
    status = getattr(result, "status", None)
    accept_status = getattr(accept, "status", None) if accept is not None else None
    if status == "pass" and accept is not None and accept_status == "pass":
        return
    err = getattr(result, "error_code", None)
    run_id = getattr(result, "run_id", None)
    raise RuntimeError(
        f"slice not pass: status={status!r} accept={accept_status!r} "
        f"error_code={err!r} run_id={run_id!r}"
    )
