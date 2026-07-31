"""Shared acceptance assertions for 0011+ (and downstream 0003/0004/0005 tracks)."""

from __future__ import annotations

from typing import Any

from meshops.acceptance import HONESTY_MESSAGE, AcceptanceResult
from meshops.acceptance.models import HonestyKind


def assert_accepted(
    result: AcceptanceResult,
    *,
    ok: bool = True,
    honesty: HonestyKind | None = None,
    require_honesty_message: bool = True,
    failed_contains: list[str] | None = None,
    failed_excludes: list[str] | None = None,
    view_kind: str | None = None,
    **metric_checks: Any,
) -> None:
    """Assert AcceptanceResult contract for pack / mutator tests.

    Parameters
    ----------
    ok:
        Expected result.ok.
    honesty:
        If set, assert result.honesty equals this value.
    require_honesty_message:
        When True (default), assert honesty_message is the canonical constant.
    failed_contains / failed_excludes:
        Codes that must / must not appear in result.failed.
    view_kind:
        If set, assert result.view_kind.
    metric_checks:
        Optional exact key→value checks against result.metrics.
    """
    assert result.ok is ok, f"expected ok={ok}, got failed={result.failed} msgs={result.messages}"
    if require_honesty_message:
        assert result.honesty_message == HONESTY_MESSAGE
        assert result.honesty_message  # always non-empty
    if honesty is not None:
        assert result.honesty == honesty, f"honesty={result.honesty!r} expected {honesty!r}"
    if ok:
        assert result.honesty != "not_accepted"
        assert result.failed == []
    else:
        assert result.honesty == "not_accepted" or result.failed
    if failed_contains:
        for code in failed_contains:
            assert code in result.failed, f"missing failed code {code!r} in {result.failed}"
    if failed_excludes:
        for code in failed_excludes:
            assert code not in result.failed, f"unexpected failed code {code!r}"
    if view_kind is not None:
        assert result.view_kind == view_kind
    for key, expected in metric_checks.items():
        assert key in result.metrics, f"metric {key!r} missing from {list(result.metrics)}"
        assert result.metrics[key] == expected, (
            f"metric {key!r}: got {result.metrics[key]!r} expected {expected!r}"
        )
