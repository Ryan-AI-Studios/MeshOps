"""GuardPolicy.for_design tier (DoD-7)."""

from __future__ import annotations

from meshops.guards import GuardPolicy


def test_for_design_tier_present() -> None:
    p = GuardPolicy.for_design()
    assert p.tier == "design"
    assert p.face_floor_ratio == 0.50
    assert p.size_floor_ratio == 0.40
    assert "design_tier" in p.notes
    assert "self_baseline_safety_net_only" in p.notes
