"""Track 0101 - heel/ankle product-hw fixture hygiene.

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Fixture retarget is not mesh/print success. Schema 1.4.0 / MCP 46 stay.
"""

from __future__ import annotations

from pathlib import Path

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import RECIPE_SCHEMA_VERSION
from meshops.proportion.extremity_recipe import (
    ANK_RY_FLOOR_M,
    ANK_RY_FRAC_HALF_W,
    ARCH_SOFT_RY_FRAC_HALF_DEPTH,
    BALL_SOFT_RY_FRAC_HALF_DEPTH,
    FOOT_HW_MIN_FRAC_LEN,
    FOOT_LEN_MIN_VS_CALF_DIAM,
    FOOT_LEN_VISUAL_MAX_FRAC_H,
    FOOT_LEN_VISUAL_MIN_FRAC_H,
    HEEL_REAR_Y_BIAS_FRAC_DEPTH,
    TOE_TIP_PAD_SCALE,
)

_TESTS = Path(__file__).resolve().parent
_THIS = Path(__file__).name
_PRODUCT_HW_0098_M = 0.04237
_THIN_HW_M = 0.0263
_STALE_TOKENS = ("product_hw_0080", "_PRODUCT_HW_0080_M")
_STALE_HW = "0.04035"
_RAISE_PHRASE = "width floor may raise hw"


def _fn_body(text: str, name: str) -> str:
    marker = f"def {name}"
    start = text.find(marker)
    assert start != -1, f"missing {name}"
    nxt = text.find("\ndef ", start + len(marker))
    return text[start:] if nxt == -1 else text[start:nxt]


def test_t0_no_stale_0080_current_pins() -> None:
    """T0: tests/ has no current-pin product_hw_0080 / _PRODUCT_HW_0080_M.

    Literal 0.04035 only allowed in historical comments (not assignments or
    asserts). heel_ankle T3 must not claim the width floor may raise this hw.
    """
    stale_hits: list[str] = []
    literal_hits: list[str] = []
    for path in sorted(_TESTS.glob("test_*.py")):
        if path.name == _THIS:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, start=1):
            for tok in _STALE_TOKENS:
                if tok in line:
                    stale_hits.append(f"{path.name}:{i}: {line.strip()}")
            if _STALE_HW not in line:
                continue
            code = line.split("#", 1)[0]
            if "=" in code or "assert" in code:
                literal_hits.append(f"{path.name}:{i}: {line.strip()}")
    heel = (_TESTS / "test_proportion_heel_ankle_proportion.py").read_text(encoding="utf-8")
    t3 = _fn_body(heel, "test_t3_product_ank_ry_frac_wins_floor")
    assert _RAISE_PHRASE not in t3, "T3 still claims width floor may raise hw"
    assert not stale_hits, "stale 0080 tokens:\n" + "\n".join(stale_hits)
    assert not literal_hits, "0.04035 still assigned/asserted:\n" + "\n".join(literal_hits)


def test_t1_heel_ankle_t3_fixture_is_0098() -> None:
    """T1: heel_ankle T3 feeds 0.04237; law and emit agree; L224 reworded."""
    text = (_TESTS / "test_proportion_heel_ankle_proportion.py").read_text(encoding="utf-8")
    t3 = _fn_body(text, "test_t3_product_ank_ry_frac_wins_floor")
    assert "product_hw_0080" not in t3
    assert _STALE_HW not in t3
    assert "_PRODUCT_HW_0098_M" in t3 or "product_hw_0098" in t3
    assert "0.04237" in t3
    assert _RAISE_PHRASE not in t3
    assert "0098 hw stays above width floor" in t3
    assert "ry uses emitted ank.rx" in t3
    assert ANK_RY_FRAC_HALF_W * _PRODUCT_HW_0098_M > ANK_RY_FLOOR_M


def test_t2_ankle_heel_t3_fixture_is_0098() -> None:
    """T2: ankle_heel T3 0098-class; frac_ry from emitted ank.rx."""
    text = (_TESTS / "test_proportion_ankle_heel_contact.py").read_text(encoding="utf-8")
    t3 = _fn_body(text, "test_t3_product_ank_ry_frac_wins")
    assert "0080-class" not in t3
    assert "0098-class" in t3
    assert "product_hw_0080" not in t3
    assert _STALE_HW not in t3
    assert "_PRODUCT_HW_0098_M" in t3 or "product_hw_0098" in t3
    assert "ANK_RY_FRAC_HALF_W * float(ank.rx_m)" in t3


def test_t3_foot_stack_symbol_renamed() -> None:
    """T3: foot_stack uses _PRODUCT_HW_0098_M; value stays 0.04237; 6 uses."""
    text = (_TESTS / "test_proportion_foot_stack_hierarchy.py").read_text(encoding="utf-8")
    assert "_PRODUCT_HW_0080_M" not in text
    assert "_PRODUCT_HW_0098_M = 0.04237" in text
    assert text.count("_PRODUCT_HW_0098_M") >= 7
    t2 = _fn_body(text, "test_t2_product_hw_ank_ry_equals_rx")
    assert "0098-class" in t2


def test_t4_0098_floors_held() -> None:
    """T4: 0098 stature / calf / hw-frac / cap hold."""
    assert FOOT_LEN_VISUAL_MIN_FRAC_H == 0.150
    assert FOOT_LEN_MIN_VS_CALF_DIAM == 4.2
    assert FOOT_HW_MIN_FRAC_LEN == 0.16
    assert FOOT_LEN_VISUAL_MAX_FRAC_H == 0.155
    assert FOOT_LEN_VISUAL_MIN_FRAC_H != 0.145
    assert FOOT_LEN_MIN_VS_CALF_DIAM != 4.0
    assert FOOT_HW_MIN_FRAC_LEN < 0.17


def test_t5_thin_product_hw_held() -> None:
    """T5: 0076/0056 thin PRODUCT_HW_M = 0.0263 still (floor-bind path)."""
    needle = f"PRODUCT_HW_M: float = {_THIN_HW_M}"
    for name in (
        "test_proportion_heel_ankle_proportion.py",
        "test_proportion_ankle_heel_contact.py",
    ):
        text = (_TESTS / name).read_text(encoding="utf-8")
        assert needle in text, f"{name} lost thin PRODUCT_HW_M"


def test_t6_0097_hierarchy_held() -> None:
    """T6: 0097 ank 1.00 / bias 0.14 / arch 0.26 / ball 0.24 / tip 1.00 hold."""
    assert ANK_RY_FRAC_HALF_W == 1.00
    assert HEEL_REAR_Y_BIAS_FRAC_DEPTH == 0.14
    assert ARCH_SOFT_RY_FRAC_HALF_DEPTH == 0.26
    assert BALL_SOFT_RY_FRAC_HALF_DEPTH == 0.24
    assert TOE_TIP_PAD_SCALE == 1.00


def test_t7_schema_mcp_held() -> None:
    """T7: schema 1.4.0 / MCP 46 / no src emit change in this track."""
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert len(TOOL_NAMES) == 47
