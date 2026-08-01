"""Dual-source parse of Orca 2.4.x ``.gcode.3mf`` (ZIP).

Primary: ``Metadata/slice_info.config`` XML (filament **attributes**, string bools).
Fallback: G-code header comments in ``Metadata/plate_*.gcode``.
"""

from __future__ import annotations

import contextlib
import math
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from meshops.slice.models import (
    FilamentSlot,
    ParsedSliceStats,
    ParseSource,
    SliceWarning,
)

# PLA default density g/cm³ (v1 approximate — multi-material → 0009)
PLA_DENSITY_G_CM3 = 1.24
# Fallback radius when only used_m available (typical 1.75 mm filament → r=0.875 mm)
FILAMENT_RADIUS_MM = 0.875

_TIME_RE = re.compile(
    r"(?:estimated\s+printing\s+time|total\s+estimated\s+time|"
    r"model\s+printing\s+time)\s*[:=]?\s*(.+)",
    re.IGNORECASE,
)
_FILAMENT_G_RE = re.compile(
    r"(?:total\s+filament\s+weight\s*\[g\]|filament\s+used\s*\[g\]|"
    r"total\s+filament\s+used\s*\[g\])\s*[:=]?\s*([\d.]+)",
    re.IGNORECASE,
)
_FILAMENT_M_RE = re.compile(
    r"(?:total\s+filament\s+used\s*\[m\]|filament\s+used\s*\[m\])\s*[:=]?\s*([\d.]+)",
    re.IGNORECASE,
)
_FILAMENT_MM3_RE = re.compile(
    r"(?:filament\s+used\s*\[mm3\]|total\s+filament\s+used\s*\[mm3\])\s*[:=]?\s*([\d.]+)",
    re.IGNORECASE,
)


def _parse_bool_str(value: str | None) -> bool:
    """Orca uses std::boolalpha → string ``true``/``false``."""
    if value is None:
        return False
    return value.strip().lower() == "true"


def _meta_map(plate: ET.Element) -> dict[str, str]:
    out: dict[str, str] = {}
    for meta in plate.findall("metadata"):
        key = meta.get("key")
        if key is None:
            continue
        out[key] = meta.get("value", "")
    return out


def _parse_header_version(root: ET.Element) -> str | None:
    header = root.find("header")
    if header is None:
        return None
    for item in header.findall("header_item"):
        key = (item.get("key") or "").strip()
        if key in ("OrcaSlicer-Version", "X-BBL-Client-Version"):
            val = (item.get("value") or "").strip()
            if val:
                return val
    return None


def _parse_warnings(plate: ET.Element) -> list[SliceWarning]:
    warnings: list[SliceWarning] = []
    for w in plate.findall("warning"):
        level_raw = w.get("level", "0")
        try:
            level = int(float(level_raw))
        except (TypeError, ValueError):
            level = 0
        err = w.get("error_code") or w.get("errorCode")
        warnings.append(
            SliceWarning(
                msg=(w.get("msg") or w.get("message") or "").strip(),
                level=level,
                error_code=err.strip() if err else None,
            )
        )
    return warnings


def _parse_filaments(plate: ET.Element) -> list[FilamentSlot]:
    slots: list[FilamentSlot] = []
    for fil in plate.findall("filament"):
        fid = fil.get("id") or str(len(slots) + 1)
        try:
            used_g = float(fil.get("used_g") or 0.0)
        except (TypeError, ValueError):
            used_g = 0.0
        try:
            used_m = float(fil.get("used_m") or 0.0)
        except (TypeError, ValueError):
            used_m = 0.0
        slots.append(
            FilamentSlot(
                id=str(fid),
                used_g=used_g,
                used_m=used_m,
                type=fil.get("type"),
                color=fil.get("color"),
            )
        )
    return slots


def filament_cm3_from_usage(
    *,
    used_g: float,
    used_m: float,
    density_g_cm3: float = PLA_DENSITY_G_CM3,
) -> float | None:
    """v1: ``used_g / 1.24`` PLA; fallback pi*r^2*sum(used_m) if grams missing."""
    if used_g > 0.0 and density_g_cm3 > 0.0:
        return used_g / density_g_cm3
    if used_m > 0.0:
        # used_m is meters; r in mm → volume mm³ → cm³
        r = FILAMENT_RADIUS_MM
        mm3 = math.pi * (r**2) * (used_m * 1000.0)
        return mm3 / 1000.0
    if used_g == 0.0 and used_m == 0.0:
        return 0.0
    return None


def select_plate(plates: list[ET.Element]) -> tuple[ET.Element | None, int | None]:
    """Pick plate index=1 if present, else first plate. Never sum multi-plate."""
    if not plates:
        return None, None
    for plate in plates:
        meta = _meta_map(plate)
        idx_s = meta.get("index", "").strip()
        if idx_s == "1":
            try:
                return plate, int(idx_s)
            except ValueError:
                return plate, 1
    first = plates[0]
    meta = _meta_map(first)
    idx_s = meta.get("index", "").strip()
    try:
        idx = int(idx_s) if idx_s else 1
    except ValueError:
        idx = 1
    return first, idx


def parse_slice_info_xml(xml_text: str) -> ParsedSliceStats:
    """Parse 2.4.x ``slice_info.config`` body into ParsedSliceStats."""
    stats = ParsedSliceStats(parse_source="slice_info")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        stats.parse_source = "failed"
        stats.messages.append(f"slice_info XML parse error: {exc}")
        return stats

    stats.orca_version = _parse_header_version(root)
    plates = list(root.findall("plate"))
    stats.plate_count = len(plates)
    plate, plate_index = select_plate(plates)
    if plate is None:
        stats.messages.append("slice_info has no <plate> elements")
        stats.parse_source = "failed"
        return stats

    stats.plate_index = plate_index
    meta = _meta_map(plate)

    pred = meta.get("prediction", "").strip()
    if pred:
        try:
            stats.print_time_s = float(pred)
        except ValueError:
            stats.messages.append(f"unparseable prediction: {pred!r}")

    weight = meta.get("weight", "").strip()
    if weight:
        try:
            stats.weight_g = float(weight)
        except ValueError:
            stats.messages.append(f"unparseable weight: {weight!r}")

    stats.bed_overflow = _parse_bool_str(meta.get("outside"))
    stats.support_used = _parse_bool_str(meta.get("support_used"))

    slots = _parse_filaments(plate)
    stats.filaments = slots
    stats.filament_used_g = sum(s.used_g for s in slots)
    stats.filament_used_m = sum(s.used_m for s in slots)
    # Prefer sum of used_g; fall back to weight metadata
    if stats.filament_used_g <= 0.0 and stats.weight_g is not None:
        stats.filament_used_g = stats.weight_g

    stats.filament_used_cm3 = filament_cm3_from_usage(
        used_g=stats.filament_used_g,
        used_m=stats.filament_used_m,
    )

    warnings = _parse_warnings(plate)
    stats.warnings = warnings
    stats.warning_max_level = max((w.level for w in warnings), default=0)

    stats.metrics = _slot_metrics(stats)
    stats.metrics["slice.parse_source"] = "slice_info"
    stats.metrics["slice.plate_count"] = stats.plate_count
    stats.metrics["slice.plate_index"] = stats.plate_index
    stats.metrics["slice.support_used"] = stats.support_used
    stats.metrics["slice.warning_max_level"] = stats.warning_max_level
    stats.metrics["slice.filament_used_g"] = stats.filament_used_g
    stats.metrics["slice.filament_used_m"] = stats.filament_used_m
    stats.metrics["slice.density_g_cm3"] = PLA_DENSITY_G_CM3
    stats.metrics["slice.density_note"] = "PLA v1 approximate 1.24 g/cm³"
    return stats


def _slot_metrics(stats: ParsedSliceStats) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for slot in stats.filaments:
        metrics[f"slice.filament_{slot.id}.used_g"] = slot.used_g
        metrics[f"slice.filament_{slot.id}.used_m"] = slot.used_m
        if slot.type:
            metrics[f"slice.filament_{slot.id}.type"] = slot.type
    return metrics


def _parse_time_token(token: str) -> float | None:
    """Parse ``1h 20m 5s`` / ``45m`` / bare seconds."""
    token = token.strip()
    if not token:
        return None
    # bare number → seconds
    try:
        return float(token)
    except ValueError:
        pass
    total = 0.0
    found = False
    for m in re.finditer(r"([\d.]+)\s*([dhms])", token, re.IGNORECASE):
        found = True
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "d":
            total += val * 86400.0
        elif unit == "h":
            total += val * 3600.0
        elif unit == "m":
            total += val * 60.0
        else:
            total += val
    return total if found else None


def parse_gcode_comments(gcode_text: str) -> ParsedSliceStats:
    """Regex Orca/Bambu-style G-code header comments for time / filament."""
    stats = ParsedSliceStats(parse_source="gcode_comments")
    stats.plate_count = 1
    stats.plate_index = 1

    for line in gcode_text.splitlines():
        stripped = line.lstrip(";").strip()
        if not stripped:
            continue
        tm = _TIME_RE.search(stripped)
        if tm and stats.print_time_s is None:
            parsed = _parse_time_token(tm.group(1))
            if parsed is not None:
                stats.print_time_s = parsed
        gm = _FILAMENT_G_RE.search(stripped)
        if gm:
            with contextlib.suppress(ValueError):
                stats.filament_used_g = float(gm.group(1))
                stats.weight_g = stats.filament_used_g
        mm = _FILAMENT_M_RE.search(stripped)
        if mm:
            with contextlib.suppress(ValueError):
                stats.filament_used_m = float(mm.group(1))
        mm3 = _FILAMENT_MM3_RE.search(stripped)
        if mm3 and stats.filament_used_cm3 is None:
            with contextlib.suppress(ValueError):
                stats.filament_used_cm3 = float(mm3.group(1)) / 1000.0

    if stats.filament_used_cm3 is None:
        stats.filament_used_cm3 = filament_cm3_from_usage(
            used_g=stats.filament_used_g,
            used_m=stats.filament_used_m,
        )

    usable = (
        stats.print_time_s is not None
        or stats.filament_used_g > 0.0
        or stats.filament_used_m > 0.0
        or (stats.filament_used_cm3 is not None and stats.filament_used_cm3 > 0.0)
    )
    if not usable:
        stats.parse_source = "failed"
        stats.messages.append("gcode comments had no usable time/filament stats")
        return stats

    stats.metrics = {
        "slice.parse_source": "gcode_comments",
        "slice.plate_count": 1,
        "slice.filament_used_g": stats.filament_used_g,
        "slice.filament_used_m": stats.filament_used_m,
        "slice.density_g_cm3": PLA_DENSITY_G_CM3,
        "slice.density_note": "PLA v1 approximate 1.24 g/cm³",
    }
    return stats


def _read_zip_member(zf: zipfile.ZipFile, names: list[str]) -> str | None:
    # Case-insensitive member lookup
    lower_map = {n.lower().replace("\\", "/"): n for n in zf.namelist()}
    for want in names:
        key = want.lower().replace("\\", "/")
        real = lower_map.get(key)
        if real is not None:
            try:
                return zf.read(real).decode("utf-8", errors="replace")
            except (KeyError, OSError):
                continue
    return None


def _gcode_members(zf: zipfile.ZipFile) -> list[str]:
    names = []
    for n in zf.namelist():
        norm = n.replace("\\", "/")
        if norm.lower().endswith(".gcode") and "metadata" in norm.lower():
            names.append(n)
    # Prefer plate_1
    names.sort(key=lambda n: (0 if "plate_1" in n.lower() else 1, n.lower()))
    return names


def parse_gcode_3mf(path: Path | str) -> ParsedSliceStats:
    """Open ``.gcode.3mf`` ZIP: slice_info primary, G-code comments fallback."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        stats = ParsedSliceStats(parse_source="failed")
        stats.messages.append(f"3mf missing or empty: {p}")
        return stats

    try:
        with zipfile.ZipFile(p, "r") as zf:
            xml = _read_zip_member(
                zf,
                [
                    "Metadata/slice_info.config",
                    "metadata/slice_info.config",
                ],
            )
            if xml:
                stats = parse_slice_info_xml(xml)
                if stats.parse_source == "slice_info":
                    return stats
                # fall through to gcode if XML unusable

            for member in _gcode_members(zf):
                try:
                    text = zf.read(member).decode("utf-8", errors="replace")
                except (KeyError, OSError):
                    continue
                gstats = parse_gcode_comments(text)
                if gstats.parse_source == "gcode_comments":
                    return gstats

            failed = ParsedSliceStats(parse_source="failed")
            failed.messages.append("no usable slice_info or gcode comments in 3mf")
            failed.metrics["slice.parse_source"] = "failed"
            return failed
    except zipfile.BadZipFile as exc:
        stats = ParsedSliceStats(parse_source="failed")
        stats.messages.append(f"bad 3mf zip: {exc}")
        return stats


def extract_slice_info_to(path: Path | str, dest: Path) -> bool:
    """Copy ``Metadata/slice_info.config`` out of 3mf for human debug. Return ok."""
    p = Path(path)
    try:
        with zipfile.ZipFile(p, "r") as zf:
            xml = _read_zip_member(
                zf,
                ["Metadata/slice_info.config", "metadata/slice_info.config"],
            )
            if xml is None:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(xml, encoding="utf-8")
            return True
    except (OSError, zipfile.BadZipFile):
        return False


def extract_plate_gcode_to(path: Path | str, dest: Path) -> bool:
    """Extract first/preferred plate gcode from 3mf. Return ok."""
    p = Path(path)
    try:
        with zipfile.ZipFile(p, "r") as zf:
            members = _gcode_members(zf)
            if not members:
                return False
            data = zf.read(members[0])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True
    except (OSError, zipfile.BadZipFile, KeyError):
        return False


def parse_source_label(stats: ParsedSliceStats) -> ParseSource:
    return stats.parse_source
