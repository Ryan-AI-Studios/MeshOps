"""Fast on-disk size helpers (R6 — scandir walk, not naive rglob)."""

from __future__ import annotations

import os
from pathlib import Path


def approx_package_dir_mb(path: Path | str) -> float | None:
    """Approximate directory size in MiB via ``os.scandir`` walk.

    Returns None if *path* is missing or not a directory.
    Package-dir only — may undercount natives outside the package tree.
    """
    root = Path(path)
    if not root.is_dir():
        return None

    total = 0
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue

    return round(total / (1024 * 1024), 1)
