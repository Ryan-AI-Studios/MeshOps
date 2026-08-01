"""Peak RSS helper — optional psutil, platform fallbacks, or None (honesty).

Never required core (C1). Prefer ``meshops[bench]`` + psutil for cross-platform RSS.
"""

from __future__ import annotations

import sys
from typing import Any


def get_current_rss_mb() -> float | None:
    """Return **current** process RSS in MiB, or None if unavailable.

    Prefer current working set (not process-lifetime high-water) so callers can
    compute **case-scoped** peaks by sampling during a ladder case.
    """
    via_psutil = _rss_via_psutil_current()
    if via_psutil is not None:
        return via_psutil

    if sys.platform == "win32":
        via_win = _rss_via_windows_ctypes_current()
        if via_win is not None:
            return via_win
    else:
        # Unix ru_maxrss is lifetime peak — not suitable as "current".
        # Leave None when psutil missing on Unix.
        pass

    return None


def get_peak_rss_mb() -> float | None:
    """Backward-compatible alias: current RSS sample (not process-lifetime peak).

    Ladder cases should use :func:`case_peak_rss_mb` with multiple samples so
    multi-size runs do not publish a single process high-water for every row.
    """
    return get_current_rss_mb()


def case_peak_rss_mb(samples: list[float | None]) -> float | None:
    """Max of non-None current-RSS samples collected during one case."""
    vals = [float(s) for s in samples if s is not None]
    if not vals:
        return None
    return max(vals)


def get_available_ram_bytes() -> int | None:
    """Best-effort available system RAM in bytes, or None if unprobeable.

    Used for L/XL RAM gate. When None, runner skips the gate (conservative allow)
    rather than false-skipping on healthy hosts without psutil.
    """
    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return int(stat.ullAvailPhys)
        except Exception:
            return None

    # Linux: MemAvailable from /proc/meminfo
    try:
        avail_kb: int | None = None
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    avail_kb = int(parts[1])
                    break
        if avail_kb is not None:
            return avail_kb * 1024
    except Exception:
        pass

    return None


def get_total_ram_mb() -> float | None:
    """Total physical RAM in MiB when probeable."""
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.virtual_memory().total) / (1024.0 * 1024.0)
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return float(stat.ullTotalPhys) / (1024.0 * 1024.0)
        except Exception:
            return None

    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        pass

    return None


def _rss_via_psutil_current() -> float | None:
    """psutil current RSS (WorkingSet / resident set) in MiB."""
    try:
        import psutil  # type: ignore[import-untyped]

        rss = psutil.Process().memory_info().rss
        return float(rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _rss_via_windows_ctypes_current() -> float | None:
    """Windows current WorkingSetSize via PSAPI (not PeakWorkingSetSize)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None
        current: Any = counters.WorkingSetSize
        if not current:
            return None
        return float(current) / (1024.0 * 1024.0)
    except Exception:
        return None
