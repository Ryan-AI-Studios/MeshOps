"""Template registry for design-from-spec."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from meshops.design.errors import DesignError
from meshops.design.models import BracketParams
from meshops.design.templates import bracket_m4

TemplateRenderer = Callable[..., str]

_REGISTRY: dict[str, TemplateRenderer] = {
    bracket_m4.TEMPLATE_ID: bracket_m4.render_source,
}


def list_templates() -> list[str]:
    return sorted(_REGISTRY)


def get_template(template_id: str) -> TemplateRenderer:
    try:
        return _REGISTRY[template_id]
    except KeyError as exc:
        raise DesignError(
            f"unknown design template {template_id!r}; known={list_templates()}",
            code="unknown_template",
            details={"template_id": template_id, "known": list_templates()},
        ) from exc


def render_template(template_id: str, params: dict[str, Any] | BracketParams | None = None) -> str:
    """Render geometry source for a registered template."""
    renderer = get_template(template_id)
    if template_id == bracket_m4.TEMPLATE_ID:
        if params is None:
            bp = BracketParams()
        elif isinstance(params, BracketParams):
            bp = params
        else:
            try:
                bp = BracketParams.model_validate(params)
            except Exception as exc:
                raise DesignError(
                    f"invalid BracketParams: {exc}",
                    code="template_error",
                    details={"params": params},
                ) from exc
        return renderer(bp)
    # Generic: pass through
    return renderer(params)
