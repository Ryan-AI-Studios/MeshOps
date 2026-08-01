"""Resolve Orca profile triad (machine / process / filament).

Default bundle is flattened (no ``inherits`` keys) so headless CLI does not
depend on vendor preset trees. Optional ``MESHOPS_ORCA_DATADIR`` for system trees.
"""

from __future__ import annotations

import os
from pathlib import Path

from meshops.slice.errors import SliceError
from meshops.slice.models import ProfilePaths

ENV_ORCA_MACHINE = "MESHOPS_ORCA_MACHINE"
ENV_ORCA_PROCESS = "MESHOPS_ORCA_PROCESS"
ENV_ORCA_FILAMENT = "MESHOPS_ORCA_FILAMENT"
ENV_ORCA_PROFILES = "MESHOPS_ORCA_PROFILES"
ENV_ORCA_DATADIR = "MESHOPS_ORCA_DATADIR"

_PROFILE_FILES = ("machine.json", "process.json", "filament.json")

# Package default triad
_PACKAGE_PROFILES = Path(__file__).resolve().parent / "profiles"


def package_profiles_dir() -> Path:
    """Return package ``meshops/slice/profiles`` directory."""
    return _PACKAGE_PROFILES


def default_profile_dir() -> Path:
    """Return package ``profiles/default`` directory."""
    return _PACKAGE_PROFILES / "default"


def _check_triad(directory: Path) -> tuple[Path, Path, Path] | None:
    machine = directory / "machine.json"
    process = directory / "process.json"
    filament = directory / "filament.json"
    if machine.is_file() and process.is_file() and filament.is_file():
        return machine, process, filament
    return None


def resolve_profiles(
    slice_profile: str | None = "default",
    *,
    machine: Path | str | None = None,
    process: Path | str | None = None,
    filament: Path | str | None = None,
    datadir: Path | str | None = None,
) -> ProfilePaths:
    """Resolve absolute machine/process/filament paths.

    Order:
      1. Explicit kwargs / env absolute overrides (all three must resolve)
      2. Absolute directory with the triad files
      3. Named profile under ``MESHOPS_ORCA_PROFILES/<name>/``
      4. Package ``profiles/<name>/`` (default: ``default``)
    """
    name = (slice_profile or "default").strip() or "default"

    env_machine = os.environ.get(ENV_ORCA_MACHINE, "").strip()
    env_process = os.environ.get(ENV_ORCA_PROCESS, "").strip()
    env_filament = os.environ.get(ENV_ORCA_FILAMENT, "").strip()

    m_raw = Path(machine) if machine else (Path(env_machine) if env_machine else None)
    p_raw = Path(process) if process else (Path(env_process) if env_process else None)
    f_raw = Path(filament) if filament else (Path(env_filament) if env_filament else None)

    # If any override is set, require all three (mix of kwargs + env OK).
    if m_raw is not None or p_raw is not None or f_raw is not None:
        if m_raw is None or p_raw is None or f_raw is None:
            raise SliceError(
                "partial profile override: set all three of machine/process/filament "
                f"(env {ENV_ORCA_MACHINE}/{ENV_ORCA_PROCESS}/{ENV_ORCA_FILAMENT})",
                code="profile_not_found",
                details={
                    "machine": str(m_raw) if m_raw else None,
                    "process": str(p_raw) if p_raw else None,
                    "filament": str(f_raw) if f_raw else None,
                },
            )
        return _finalize(m_raw, p_raw, f_raw, name=name, datadir=datadir)

    # Absolute path to a profile directory
    as_path = Path(name)
    if as_path.is_absolute() or (len(name) > 1 and name[1] == ":"):
        triad = _check_triad(as_path.expanduser().resolve(strict=False))
        if triad is None:
            raise SliceError(
                f"profile directory missing triad ({', '.join(_PROFILE_FILES)}): {as_path}",
                code="profile_not_found",
                details={"dir": str(as_path)},
            )
        return _finalize(*triad, name=str(as_path), datadir=datadir)

    # MESHOPS_ORCA_PROFILES / <name>
    profiles_root = os.environ.get(ENV_ORCA_PROFILES, "").strip()
    if profiles_root:
        root = Path(profiles_root).expanduser().resolve(strict=False)
        triad = _check_triad(root / name)
        if triad is not None:
            return _finalize(*triad, name=name, datadir=datadir)

    # Package profiles
    triad = _check_triad(_PACKAGE_PROFILES / name)
    if triad is not None:
        return _finalize(*triad, name=name, datadir=datadir)

    raise SliceError(
        f"slice profile not found: {name!r} (checked package profiles and {ENV_ORCA_PROFILES})",
        code="profile_not_found",
        details={
            "name": name,
            "package": str(_PACKAGE_PROFILES / name),
            "env_profiles": profiles_root or None,
        },
    )


def _finalize(
    machine: Path,
    process: Path,
    filament: Path,
    *,
    name: str,
    datadir: Path | str | None,
) -> ProfilePaths:
    paths = []
    for label, p in (("machine", machine), ("process", process), ("filament", filament)):
        try:
            resolved = p.expanduser().resolve(strict=False)
        except OSError:
            resolved = p.expanduser()
        if not resolved.is_file():
            raise SliceError(
                f"profile {label} file not found: {resolved}",
                code="profile_not_found",
                details={"label": label, "path": str(resolved)},
            )
        paths.append(resolved)

    dd: str | None = None
    if datadir is not None:
        dd = str(Path(datadir).expanduser().resolve(strict=False))
    else:
        env_dd = os.environ.get(ENV_ORCA_DATADIR, "").strip()
        if env_dd:
            dd = str(Path(env_dd).expanduser().resolve(strict=False))

    return ProfilePaths(
        machine=str(paths[0]),
        process=str(paths[1]),
        filament=str(paths[2]),
        profile_name=name,
        datadir=dd,
    )


def default_bundle_has_inherits() -> list[str]:
    """Return relative paths in default bundle that still contain ``inherits``.

    Used by tests / CI guard — default triad MUST be flattened.
    """
    import json

    bad: list[str] = []
    base = default_profile_dir()
    for fname in _PROFILE_FILES:
        path = base / fname
        if not path.is_file():
            bad.append(f"missing:{fname}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            bad.append(f"unreadable:{fname}")
            continue
        if isinstance(data, dict) and "inherits" in data:
            bad.append(fname)
    return bad
