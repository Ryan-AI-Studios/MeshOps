"""Slice acceptance hook protocol (runtime param only — never store on pydantic).

Body implementation is track 0005 (Orca). 0011 owns the protocol + SliceAcceptResult.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from meshops.acceptance.models import SliceAcceptResult


@runtime_checkable
class SliceAcceptHook(Protocol):
    """Callable that evaluates optional printability / slice acceptance.

    Never attach instances to AcceptanceResult or other pydantic models —
    store only SliceAcceptResult. slice_profile is forwarded without pack validation.
    """

    def __call__(
        self,
        *,
        candidate_path: str | None = None,
        slice_profile: str | None = None,
        **kwargs: object,
    ) -> SliceAcceptResult: ...


def null_slice_result(*, reason: str = "slice hook not configured") -> SliceAcceptResult:
    """Default when no hook is provided — status skipped."""
    return SliceAcceptResult(
        status="skipped",
        messages=[reason],
        error_code=None,
    )
