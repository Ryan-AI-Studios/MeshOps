"""Peak RSS helper — optional psutil, platform fallbacks, or None (honesty).

Never required core (C1). Prefer ``meshops[bench]`` + psutil for cross-platform RSS.
"""

from __future__ import annotations

import sys
from typing import Any


def get_peak_rss_mb() -> float | None:
    """Return process **peak** RSS in MiB when the OS exposes it, else None.

    Order (true peak first — DoD honesty for ``rss_peak_mb``):
    1. Windows: ``GetProcessMemoryInfo`` → **PeakWorkingSetSize**
    2. Unix: ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (peak high-water mark)
    3. Optional ``psutil``: ``memory_info().peak_wset`` (Windows) or ``rss`` only as
       last-resort approximate when no peak field exists (documented in return path)
    4. ``None``

    Prefer OS peak counters over psutil current RSS so envelope numbers match the
    field name ``rss_peak_mb``.
    """
    if sys.platform == "win32":
        via_win = _rss_via_windows_ctypes()
        if via_win is not None:
            return via_win
    else:
        via_res = _rss_via_resource()
        if via_res is not None:
            return via_res

    via_psutil = _rss_via_psutil_peak_or_current()
    if via_psutil is not None:
        return via_psutil

    return None


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


def _rss_via_psutil_peak_or_current() -> float | None:
    """psutil path: prefer peak_wset when present; else current rss as last resort."""
    try:
        import psutil  # type: ignore[import-untyped]

        info = psutil.Process().memory_info()
        # Windows pmem may expose peak_wset (bytes); not on all platforms.
        peak = getattr(info, "peak_wset", None)
        if peak is not None and int(peak) > 0:
            return float(peak) / (1024.0 * 1024.0)
        rss = getattr(info, "rss", None)
        if rss is not None and int(rss) > 0:
            return float(rss) / (1024.0 * 1024.0)
        return None
    except Exception:
        return None


def _rss_via_windows_ctypes() -> float | None:
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
        # Prefer peak working set when available.
        peak: Any = counters.PeakWorkingSetSize or counters.WorkingSetSize
        return float(peak) / (1024.0 * 1024.0)
    except Exception:
        return None


def _rss_via_resource() -> float | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: ru_maxrss in KiB; macOS: bytes.
        rss = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return None
