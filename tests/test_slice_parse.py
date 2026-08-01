"""Dual-source 2.4.x slice_info + gcode fallback parse (DoD-5,6)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from meshops.slice.anomaly import evaluate_printability
from meshops.slice.parse_3mf import (
    PLA_DENSITY_G_CM3,
    filament_cm3_from_usage,
    parse_gcode_3mf,
    parse_gcode_comments,
    parse_slice_info_xml,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "slice"


def _xml(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _make_3mf(path: Path, *, slice_info: str | None = None, gcode: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        if slice_info is not None:
            zf.writestr("Metadata/slice_info.config", slice_info)
        if gcode is not None:
            zf.writestr("Metadata/plate_1.gcode", gcode)
        # Minimal 3mf content types so zip is non-empty/valid
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
    return path


def test_parse_slice_info_ok_attrs() -> None:
    stats = parse_slice_info_xml(_xml("slice_info_ok.config"))
    assert stats.parse_source == "slice_info"
    assert stats.orca_version == "2.4.2"
    assert stats.plate_count == 1
    assert stats.print_time_s == 4823.0
    assert stats.bed_overflow is False
    assert stats.support_used is False
    assert abs(stats.filament_used_g - 14.21) < 1e-6
    assert abs(stats.filament_used_m - 4.823) < 1e-6
    assert stats.filament_used_cm3 is not None
    assert abs(stats.filament_used_cm3 - (14.21 / PLA_DENSITY_G_CM3)) < 1e-6
    assert stats.warning_max_level == 1
    assert stats.metrics["slice.filament_1.used_g"] == 14.21


def test_parse_outside_true_string() -> None:
    stats = parse_slice_info_xml(_xml("slice_info_outside.config"))
    assert stats.bed_overflow is True
    # Python bool("false") is True — we must not use that
    assert _parse_bool_guard() is False


def _parse_bool_guard() -> bool:
    """Document anti-pattern: bool('false') is True."""
    return "false".strip().lower() == "true"


def test_multi_plate_first_only() -> None:
    stats = parse_slice_info_xml(_xml("slice_info_multi_plate.config"))
    assert stats.plate_count == 2
    assert stats.print_time_s == 1000.0
    assert stats.filament_used_g == 10.0
    assert stats.bed_overflow is False  # plate 2 outside must not win


def test_warning_level_2() -> None:
    stats = parse_slice_info_xml(_xml("slice_info_warning_l2.config"))
    assert stats.warning_max_level == 2
    assert any(w.level == 2 for w in stats.warnings)
    accept = evaluate_printability(stats, mesh_volume_cm3=None)
    assert accept.status == "fail"
    assert accept.error_code == "unsliceable_geometry"


def test_zero_filament_fail() -> None:
    stats = parse_slice_info_xml(_xml("slice_info_zero_filament.config"))
    assert stats.filament_used_g == 0.0
    assert stats.filament_used_cm3 == 0.0
    accept = evaluate_printability(stats)
    assert accept.status == "fail"
    assert accept.error_code == "filament_zero"


def test_bed_overflow_fail() -> None:
    stats = parse_slice_info_xml(_xml("slice_info_outside.config"))
    accept = evaluate_printability(stats)
    assert accept.status == "fail"
    assert accept.error_code == "bed_overflow"


def test_gcode_comments_fallback() -> None:
    text = (FIXTURES / "plate_1_fallback.gcode").read_text(encoding="utf-8")
    stats = parse_gcode_comments(text)
    assert stats.parse_source == "gcode_comments"
    assert stats.print_time_s == 1 * 3600 + 20 * 60 + 5
    assert stats.filament_used_g == 12.5
    assert stats.filament_used_m == 4.2


def test_parse_gcode_3mf_primary(tmp_path: Path) -> None:
    p = _make_3mf(tmp_path / "ok.gcode.3mf", slice_info=_xml("slice_info_ok.config"))
    stats = parse_gcode_3mf(p)
    assert stats.parse_source == "slice_info"
    assert stats.print_time_s == 4823.0


def test_parse_gcode_3mf_fallback(tmp_path: Path) -> None:
    gcode = (FIXTURES / "plate_1_fallback.gcode").read_text(encoding="utf-8")
    p = _make_3mf(tmp_path / "fb.gcode.3mf", gcode=gcode)
    stats = parse_gcode_3mf(p)
    assert stats.parse_source == "gcode_comments"
    assert stats.filament_used_g == 12.5


def test_parse_gcode_3mf_root_gcode_fallback(tmp_path: Path) -> None:
    """Codex P2-002: any *.gcode member works, not only Metadata/*."""
    gcode = (FIXTURES / "plate_1_fallback.gcode").read_text(encoding="utf-8")
    p = tmp_path / "root.gcode.3mf"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("plate.gcode", gcode)
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
    stats = parse_gcode_3mf(p)
    assert stats.parse_source == "gcode_comments"
    assert stats.filament_used_g == 12.5


def test_parse_3mf_application_version_fallback(tmp_path: Path) -> None:
    """Codex P2-003: Application metadata when slice_info header version absent."""
    xml = """<?xml version="1.0"?>
<config>
  <header></header>
  <plate>
    <metadata key="index" value="1"/>
    <metadata key="prediction" value="100"/>
    <metadata key="outside" value="false"/>
    <filament id="1" used_m="1" used_g="1.24" type="PLA"/>
  </plate>
</config>
"""
    p = tmp_path / "appver.gcode.3mf"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("Metadata/slice_info.config", xml)
        zf.writestr(
            "3D/3dmodel.model",
            '<?xml version="1.0"?><model Application="OrcaSlicer-2.4.2"/>',
        )
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
    stats = parse_gcode_3mf(p)
    assert stats.parse_source == "slice_info"
    assert stats.orca_version == "2.4.2"


def test_parse_gcode_3mf_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope.gcode.3mf"
    stats = parse_gcode_3mf(missing)
    assert stats.parse_source == "failed"


def test_filament_cm3_density() -> None:
    cm3_one = filament_cm3_from_usage(used_g=1.24, used_m=0.0)
    assert cm3_one is not None
    assert abs(cm3_one - 1.0) < 1e-9
    # grams preferred over meters
    cm3 = filament_cm3_from_usage(used_g=2.48, used_m=100.0)
    assert cm3 is not None
    assert abs(cm3 - 2.0) < 1e-9
