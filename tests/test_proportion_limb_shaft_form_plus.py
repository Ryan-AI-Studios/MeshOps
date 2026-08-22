"""Track 0107 — limb shaft form plus (UA 0.84 / FA 0.70 / calf dist 0.80).

Authoring honesty only (Difficulty §12 / N6 / RECIPE_HONESTY).
Schema 1.4.0 / MCP 47 stay. Not mesh/print success.
Does not reopen 0094 thigh 0.72, 0081 elbow 1.22, 0095 knee 1.08,
0096 belly 1.18 / split 0.42, 0087 hang T 0.50, or 0116 per-job scale.
"""

from __future__ import annotations

import pytest

from meshops.mcp.server import TOOL_NAMES
from meshops.proportion.blockout_recipe import (
    CALF_BELLY_SCALE,
    CALF_DIST_END_SCALE,
    CALF_DIST_SHAFT_SCALE,
    CALF_PROX_END_SCALE,
    CALF_SPLIT_T,
    COMPACT_CULL_NAME_PREFIXES,
    COMPACT_CULL_ROLES,
    ELBOW_SOFT_SCALE,
    FA_DIST_SHAFT_SCALE,
    FA_PROX_SHAFT_SCALE,
    FA_SPLIT_T,
    KNEE_SOFT_FRAC,
    RECIPE_SCHEMA_VERSION,
    THIGH_DIST_SHAFT_SCALE,
    THIGH_PROX_SHAFT_SCALE,
    THIGH_SPLIT_T,
    UA_DIST_SHAFT_SCALE,
    UA_PROX_SHAFT_SCALE,
    UA_SPLIT_T,
    WRIST_SOFT_PALM_RX_FRAC,
    BlockoutRecipePackage,
    build_blockout_recipe,
)
from meshops.proportion.constraints import classify_part_name, validate_constraints
from meshops.proportion.skeleton import ELBOW_HANG_T, build_blockout_skeleton
from test_proportion_arm_taper_elbow_wrist import _limb_mass_report
from test_proportion_thigh_distal_taper_plus import _part
from test_proportion_torso_anti_tire_plus import _product_class_report, _product_flags

_UA_MID = 0.0438
_FA_MID = 0.0350
_THIGH_MID = 0.0613
_CALF_HW = 0.0438
_EXPECT_UA_DIST = _UA_MID * 0.84
_EXPECT_FA_DIST = _FA_MID * 0.70
_EXPECT_CALF_TAPER = _CALF_HW * 0.80
_EXPECT_ELBOW_RX = 1.22 * _EXPECT_UA_DIST
_EXPECT_KNEE_RX = _THIGH_MID * 0.72 * 1.08


def test_t0_const_freezes() -> None:
    """T0: UA dist 0.84 / FA dist 0.70 / calf dist 0.80; hold prox/splits/thigh/belly."""
    assert UA_DIST_SHAFT_SCALE == 0.84
    assert FA_DIST_SHAFT_SCALE == 0.70
    assert CALF_DIST_SHAFT_SCALE == 0.80
    assert UA_PROX_SHAFT_SCALE == 1.00
    assert FA_PROX_SHAFT_SCALE == 1.00
    assert UA_SPLIT_T == 0.50
    assert FA_SPLIT_T == 0.50
    assert THIGH_DIST_SHAFT_SCALE == 0.72
    assert THIGH_PROX_SHAFT_SCALE == 1.00
    assert THIGH_SPLIT_T == 0.50
    assert CALF_BELLY_SCALE == 1.18
    assert CALF_SPLIT_T == 0.42
    assert CALF_DIST_END_SCALE == 0.72
    assert CALF_PROX_END_SCALE == 0.88
    assert 0.82 <= UA_DIST_SHAFT_SCALE <= 0.88
    assert 0.66 <= FA_DIST_SHAFT_SCALE <= 0.74
    assert 0.76 <= CALF_DIST_SHAFT_SCALE <= 0.86


def test_t1_dist_less_than_prox() -> None:
    """T1: each dist scale < matching prox/belly; calf_b < calf dist shaft; thigh 0.72."""
    assert UA_DIST_SHAFT_SCALE < UA_PROX_SHAFT_SCALE
    assert FA_DIST_SHAFT_SCALE < FA_PROX_SHAFT_SCALE
    assert CALF_DIST_SHAFT_SCALE < CALF_BELLY_SCALE
    assert CALF_DIST_END_SCALE < CALF_DIST_SHAFT_SCALE
    assert THIGH_DIST_SHAFT_SCALE == 0.72


def test_t2_arm_dist_radii_both_sides() -> None:
    """T2: limbs=True — ua dist == mid*0.84; fa dist == mid*0.70; both segs L+R."""
    report = _limb_mass_report(ua_hw=_UA_MID, fa_hw=_FA_MID)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        ua = by_name[f"RECIPE_limb_upper_arm_{side}"]
        ua_dist = by_name[f"RECIPE_arm_taper_dist_ua_{side}"]
        fa = by_name[f"RECIPE_limb_forearm_{side}"]
        fa_dist = by_name[f"RECIPE_arm_taper_dist_fa_{side}"]
        assert ua.kind == "capsule"
        assert ua_dist.kind == "capsule"
        assert fa.kind == "capsule"
        assert fa_dist.kind == "capsule"
        assert float(ua_dist.radius_m) == pytest.approx(  # type: ignore[arg-type]
            _UA_MID * UA_DIST_SHAFT_SCALE, abs=1e-9
        )
        assert float(fa_dist.radius_m) == pytest.approx(  # type: ignore[arg-type]
            _FA_MID * FA_DIST_SHAFT_SCALE, abs=1e-9
        )
        assert float(ua_dist.radius_m) == pytest.approx(_EXPECT_UA_DIST, abs=1e-9)  # type: ignore[arg-type]
        assert float(fa_dist.radius_m) == pytest.approx(_EXPECT_FA_DIST, abs=1e-9)  # type: ignore[arg-type]


def test_t3_calf_cyl_taper_ends() -> None:
    """T3: cyl == mid*1.18; taper == mid*0.80; a/b hold 0.88/0.72."""
    report = _limb_mass_report(calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        cyl = by_name[f"RECIPE_calf_cyl_{side}"]
        taper = by_name[f"RECIPE_calf_taper_dist_{side}"]
        a = by_name[f"RECIPE_calf_a_{side}"]
        b = by_name[f"RECIPE_calf_b_{side}"]
        assert float(cyl.radius_m) == pytest.approx(  # type: ignore[arg-type]
            _CALF_HW * CALF_BELLY_SCALE, abs=1e-9
        )
        assert float(taper.radius_m) == pytest.approx(  # type: ignore[arg-type]
            _CALF_HW * CALF_DIST_SHAFT_SCALE, abs=1e-9
        )
        assert float(taper.radius_m) == pytest.approx(_EXPECT_CALF_TAPER, abs=1e-9)  # type: ignore[arg-type]
        assert float(a.rx_m) == pytest.approx(_CALF_HW * 0.88, abs=1e-9)  # type: ignore[arg-type]
        assert float(b.rx_m) == pytest.approx(_CALF_HW * 0.72, abs=1e-9)  # type: ignore[arg-type]
        assert float(taper.radius_m) != pytest.approx(float(a.rx_m), abs=1e-6)  # type: ignore[arg-type]


def test_t4_thigh_fence() -> None:
    """T4: thigh prox/dist still 1.00/0.72 on product-like mid 0.0613."""
    report = _limb_mass_report(thigh_hw=_THIGH_MID)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        prox = by_name[f"RECIPE_limb_thigh_{side}"]
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        assert float(prox.radius_m) == pytest.approx(_THIGH_MID, abs=1e-9)  # type: ignore[arg-type]
        assert float(dist.radius_m) == pytest.approx(_THIGH_MID * 0.72, abs=1e-9)  # type: ignore[arg-type]
        assert float(dist.radius_m) == pytest.approx(0.04414, abs=2e-4)  # type: ignore[arg-type]


def test_t5_elbow_cascade_b20() -> None:
    """T5: elbow.rx >= ua.radius - 1e-4; elbow.rx == 1.22 * ua_dist."""
    report = _limb_mass_report(ua_hw=_UA_MID, fa_hw=_FA_MID)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        ua = by_name[f"RECIPE_limb_upper_arm_{side}"]
        ua_dist = by_name[f"RECIPE_arm_taper_dist_ua_{side}"]
        elbow = by_name[f"RECIPE_elbow_soft_{side}"]
        ua_r = float(ua.radius_m)  # type: ignore[arg-type]
        dist_r = float(ua_dist.radius_m)  # type: ignore[arg-type]
        assert elbow.rx_m is not None
        assert float(elbow.rx_m) >= ua_r - 1e-4
        assert float(elbow.rx_m) == pytest.approx(ELBOW_SOFT_SCALE * dist_r, abs=1e-4)
        assert float(elbow.rx_m) == pytest.approx(_EXPECT_ELBOW_RX, abs=1e-4)


def test_t6_wrist_palm_floor_wins() -> None:
    """T6: dist_soft.rx == 0.95 * palm.rx (palm still wins; FA*1.20 smaller)."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        soft = by_name[f"RECIPE_dist_soft_forearm_{side}"]
        palm = by_name[f"RECIPE_palm_{side}"]
        fa_dist = by_name[f"RECIPE_arm_taper_dist_fa_{side}"]
        assert palm.rx_m is not None and soft.rx_m is not None
        assert float(soft.rx_m) == pytest.approx(0.95 * float(palm.rx_m), abs=1e-4)
        fa_floor = float(fa_dist.radius_m) * 1.20  # type: ignore[arg-type]
        palm_floor = WRIST_SOFT_PALM_RX_FRAC * float(palm.rx_m)
        assert palm_floor > fa_floor


def test_t6b_sibling_limb_shaft_form_plus() -> None:
    """T6b: sibling contains limb shaft form plus: + interpolated 0.84 / 0.70 / 0.80."""
    report = _limb_mass_report(ua_hw=_UA_MID, fa_hw=_FA_MID, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    msgs = pkg.messages
    ua_l = [i for i, m in enumerate(msgs) if m.startswith("upper_arm_l: shaft_taper")]
    ua_r = [i for i, m in enumerate(msgs) if m.startswith("upper_arm_r: shaft_taper")]
    fa_l = [i for i, m in enumerate(msgs) if m.startswith("forearm_l: shaft_taper")]
    fa_r = [i for i, m in enumerate(msgs) if m.startswith("forearm_r: shaft_taper")]
    calf_l = [i for i, m in enumerate(msgs) if m.startswith("calf_l: belly/taper")]
    calf_r = [i for i, m in enumerate(msgs) if m.startswith("calf_r: belly/taper")]
    sib = [i for i, m in enumerate(msgs) if m.startswith("limb shaft form plus:")]
    assert len(ua_l) == 1 and len(ua_r) == 1
    assert len(fa_l) == 1 and len(fa_r) == 1
    assert len(calf_l) == 1 and len(calf_r) == 1
    assert len(sib) == 1
    line = msgs[sib[0]]
    assert "limb shaft form plus:" in line
    assert f"ua_dist={UA_DIST_SHAFT_SCALE}" in line
    assert f"fa_dist={FA_DIST_SHAFT_SCALE}" in line
    assert f"calf_dist={CALF_DIST_SHAFT_SCALE}" in line
    assert sib[0] > ua_l[0] and sib[0] > ua_r[0]
    assert sib[0] > fa_l[0] and sib[0] > fa_r[0]
    assert sib[0] > calf_l[0] and sib[0] > calf_r[0]


def test_t7_per_side_messages_interpolate() -> None:
    """T7: per-side shaft_taper / calf_* belly/taper still interpolate new r."""
    report = _limb_mass_report(ua_hw=_UA_MID, fa_hw=_FA_MID, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    ua_line = next(m for m in pkg.messages if m.startswith("upper_arm_l: shaft_taper"))
    fa_line = next(m for m in pkg.messages if m.startswith("forearm_l: shaft_taper"))
    calf_sib = next(m for m in pkg.messages if m.startswith("calf shaft form:"))
    assert f"dist={_EXPECT_UA_DIST:.4f}" in ua_line
    assert f"dist={_EXPECT_FA_DIST:.4f}" in fa_line
    assert f"dist_shaft={CALF_DIST_SHAFT_SCALE}" in calf_sib


def test_t8_invert_old_first_pass() -> None:
    """T8: invert — UA dist != 0.88; FA dist != 0.78; calf dist shaft != 0.88."""
    assert UA_DIST_SHAFT_SCALE != 0.88
    assert FA_DIST_SHAFT_SCALE != 0.78
    assert CALF_DIST_SHAFT_SCALE != 0.88
    assert UA_DIST_SHAFT_SCALE == 0.84
    assert FA_DIST_SHAFT_SCALE == 0.70
    assert CALF_DIST_SHAFT_SCALE == 0.80


def test_t9_n_parts_schema_mcp47() -> None:
    """T9: n_parts 131 via 0092-style product flags + profile; schema 1.4.0; MCP 47."""
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    assert len(pkg.parts) == 131
    assert RECIPE_SCHEMA_VERSION == "1.4.0"
    assert pkg.schema_version == "1.4.0"
    assert len(TOOL_NAMES) == 47


def test_t10_all_already_exports_dist_scales() -> None:
    """T10: __all__ already exports the three dist scales (no new names)."""
    from meshops.proportion import blockout_recipe as br

    names = set(br.__all__)
    assert "UA_DIST_SHAFT_SCALE" in names
    assert "FA_DIST_SHAFT_SCALE" in names
    assert "CALF_DIST_SHAFT_SCALE" in names


def test_t11_knee_and_calf_a_hold() -> None:
    """T11: knee rx still 1.08 * thigh_dist (0095 fence); calf_a unchanged."""
    report = _limb_mass_report(thigh_hw=_THIGH_MID, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    by_name = {p.name: p for p in pkg.parts}
    for side in ("l", "r"):
        dist = by_name[f"RECIPE_thigh_taper_dist_{side}"]
        knee = by_name[f"RECIPE_knee_soft_{side}"]
        calf_a = by_name[f"RECIPE_calf_a_{side}"]
        dist_r = float(dist.radius_m)  # type: ignore[arg-type]
        assert knee.rx_m is not None
        assert float(knee.rx_m) == pytest.approx(dist_r * KNEE_SOFT_FRAC, abs=1e-5)
        assert float(knee.rx_m) == pytest.approx(_EXPECT_KNEE_RX, abs=2e-4)
        assert float(calf_a.rx_m) == pytest.approx(_CALF_HW * CALF_PROX_END_SCALE, abs=1e-4)  # type: ignore[arg-type]
        assert float(calf_a.rx_m) == pytest.approx(0.0385, abs=1e-4)  # type: ignore[arg-type]


def test_t12_c_thigh_outer_and_taper_unknown() -> None:
    """T12: C_thigh_outer pass; classifier unknown on taper tokens."""
    assert classify_part_name("RECIPE_arm_taper_dist_ua_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_arm_taper_dist_fa_r") == ("unknown", "r")
    assert classify_part_name("RECIPE_calf_taper_dist_l") == ("unknown", "l")
    assert classify_part_name("RECIPE_thigh_taper_dist_r") == ("unknown", "r")
    report = _limb_mass_report(thigh_hw=_THIGH_MID, calf_hw=_CALF_HW)
    pkg = build_blockout_recipe(report, limbs=True)
    dup = validate_constraints(pkg, report=report)
    dup_by = {r.id: r for r in dup.rules}
    assert dup_by["C_no_dup_limb"].status == "pass", dup_by["C_no_dup_limb"].message
    # C_thigh_outer: 0069 T9 aligned synthetic (hip_soft unknown must not fail it).
    hip = [-0.12, 0.0, 0.95]
    knee = [-0.12, 0.0, 0.50]
    mid = [0.5 * (hip[0] + knee[0]), 0.5 * (hip[1] + knee[1]), 0.5 * (hip[2] + knee[2])]
    r = 0.06
    dist_r = r * THIGH_DIST_SHAFT_SCALE
    chain_outer = mid[0] - r
    hip_half = 0.03
    hip_cx = chain_outer + hip_half
    syn = BlockoutRecipePackage(
        parts=[
            _part(
                "RECIPE_hip_bridge_l",
                role="hip_bridge",
                kind="ellipsoid",
                center=[hip_cx, 0.03, 0.95],
                rx_m=hip_half,
                ry_m=0.03,
                rz_m=0.03,
            ),
            _part("RECIPE_limb_thigh_l", radius_m=r, p0=list(hip), p1=list(mid)),
            _part(
                "RECIPE_thigh_taper_dist_l",
                radius_m=dist_r,
                p0=list(mid),
                p1=list(knee),
            ),
        ],
        counts={"parts": 3},
    )
    syn_result = validate_constraints(syn)
    syn_by = {rule.id: rule for rule in syn_result.rules}
    assert syn_by["C_thigh_outer"].status == "pass", syn_by["C_thigh_outer"].message


def test_t13_compact_still_emits_taper_segs() -> None:
    """T13: compact still emits ua/fa/thigh/calf taper segs."""
    assert "limb_segment" not in COMPACT_CULL_ROLES
    for prefix in (
        "RECIPE_arm_taper_dist_ua_",
        "RECIPE_arm_taper_dist_fa_",
        "RECIPE_thigh_taper_dist_",
        "RECIPE_calf_taper_dist_",
    ):
        assert prefix not in COMPACT_CULL_NAME_PREFIXES
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(
        report,
        skeleton=skel,
        **_product_flags(soft_density="compact"),  # type: ignore[arg-type]
    )
    names = {p.name for p in pkg.parts}
    for side in ("l", "r"):
        assert f"RECIPE_arm_taper_dist_ua_{side}" in names
        assert f"RECIPE_arm_taper_dist_fa_{side}" in names
        assert f"RECIPE_thigh_taper_dist_{side}" in names
        assert f"RECIPE_calf_taper_dist_{side}" in names


def test_t14_hang_t_and_palm_y() -> None:
    """T14: hang T 0.50 hold (0087); palm Y = wrist Y."""
    assert ELBOW_HANG_T == 0.50
    report = _product_class_report()
    skel = build_blockout_skeleton(report)
    pkg = build_blockout_recipe(report, skeleton=skel, **_product_flags())  # type: ignore[arg-type]
    by_id = {j.id: j for j in skel.joints}
    wr = by_id["wrist_l"].y_m
    palm = next(p for p in pkg.parts if p.name == "RECIPE_palm_l")
    assert palm.center is not None and wr is not None
    assert palm.center[1] == pytest.approx(wr, abs=1e-6)


def test_t15_no_new_shaft_kind() -> None:
    """T15: no RECIPE_prox_soft_thigh; no new shaft kind; capsules stay kind=capsule."""
    report = _limb_mass_report()
    pkg = build_blockout_recipe(report, limbs=True)
    names = [p.name for p in pkg.parts]
    assert not any("prox_soft_thigh" in n for n in names)
    for p in pkg.parts:
        if p.name.startswith(
            (
                "RECIPE_limb_upper_arm_",
                "RECIPE_arm_taper_dist_",
                "RECIPE_limb_forearm_",
                "RECIPE_limb_thigh_",
                "RECIPE_thigh_taper_dist_",
                "RECIPE_calf_cyl_",
                "RECIPE_calf_taper_dist_",
            )
        ):
            assert p.kind == "capsule"
