"""BracketParams pydantic validators (DoD-4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from meshops.design.models import BracketParams


def test_bracket_params__defaults_ok() -> None:
    p = BracketParams()
    assert p.hole_spacing_mm == 40.0
    assert p.wall_mm == 3.0
    assert p.thickness_mm == 4.0
    assert p.hole_diameter_mm == 4.2
    ex = p.plate_extents_mm()
    assert ex[0] == pytest.approx(40.0 + 4.2 + 2 * 3.0)
    assert ex[1] == pytest.approx(4.2 + 2 * 3.0)
    assert ex[2] == pytest.approx(4.0)


def test_bracket_params__refuse_bad_clearance() -> None:
    with pytest.raises(ValidationError):
        BracketParams(hole_spacing_mm=12.0, wall_mm=5.0, hole_diameter_mm=4.0)
    # 12 <= 4 + 2*5 = 14


def test_bracket_params__borderline_clearance_fails() -> None:
    # spacing == diameter + 2*wall → must fail (need >)
    with pytest.raises(ValidationError):
        BracketParams(hole_spacing_mm=14.0, wall_mm=5.0, hole_diameter_mm=4.0)


def test_bracket_params__good_clearance() -> None:
    p = BracketParams(hole_spacing_mm=30.0, wall_mm=3.0, hole_diameter_mm=4.2)
    assert p.hole_spacing_mm == 30.0


def test_bracket_params__bounds() -> None:
    with pytest.raises(ValidationError):
        BracketParams(hole_spacing_mm=5.0)  # gt=10
    with pytest.raises(ValidationError):
        BracketParams(wall_mm=0.5)  # gt=1
