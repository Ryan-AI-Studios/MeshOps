"""Abs-path Blender launch helper for RECIPE setup (track 0110).

SETUP_LAUNCH_HONESTY / RECIPE_HONESTY / Difficulty §4 / §12 / §13 / N6.
Print or spawn is not mesh or print success. Does not emit or mutate bpy.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from meshops.escalate.discover import find_blender_with_source
from meshops.escalate.errors import EscalateError
from meshops.proportion.blockout_recipe import BPY_BASENAME
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import SETUP_LAUNCH_HONESTY

REFUSED_BASENAMES: frozenset[str] = frozenset({"build_and_render.py", "setup_proportion_guides.py"})

__all__ = [
    "REFUSED_BASENAMES",
    "SETUP_LAUNCH_HONESTY",
    "resolve_setup_script",
    "run_blockout_open_setup",
]


def _windows_root() -> Path:
    return Path(os.environ.get("SYSTEMROOT", r"C:\Windows")).expanduser().resolve()


def _under_windows_root(path: Path, *, root: Path | None = None) -> bool:
    base = root if root is not None else _windows_root()
    try:
        path.expanduser().resolve().relative_to(base)
        return True
    except ValueError:
        return False


def _ps_quote(p: str) -> str:
    return "'" + p.replace("'", "''") + "'"


def resolve_setup_script(path: Path | str) -> Path:
    """Resolve --setup to an abs setup_blockout_recipe.py (B2-B7)."""
    raw = Path(path)
    cwd = Path.cwd().resolve()
    if not raw.is_absolute() and _under_windows_root(cwd):
        raise ProportionError(
            f"setup path is relative and meshops cwd is Windows system dir {cwd}; "
            "Blender --python would resolve under System32. Pass an absolute --setup path.",
            code="setup_cwd_unsafe",
            details={"cwd": str(cwd), "setup": str(raw)},
        )
    name = raw.name.lower()
    if name in REFUSED_BASENAMES:
        extra = ""
        if name == "build_and_render.py":
            extra = " (build_and_render is work/-only — see track 0111)"
        refused = (raw if raw.is_absolute() else cwd / raw).resolve()
        raise ProportionError(
            f"{raw.name} is not MeshOps RECIPE setup; use {BPY_BASENAME}.{extra}",
            code="setup_not_found",
            details={"setup": str(refused)},
        )
    suffix = raw.suffix.lower()
    file_class = raw.is_file() or (not raw.exists() and suffix == ".py")
    json_class = (raw.is_file() and suffix == ".json") or (not raw.exists() and suffix == ".json")
    json_abs: str | None = None
    if json_class:
        json_path = (raw if raw.is_absolute() else cwd / raw).resolve()
        setup = json_path.parent / BPY_BASENAME
        json_abs = str(json_path)
    elif file_class:
        setup = raw if raw.is_absolute() else (cwd / raw)
        setup = setup.resolve()
        if setup.name.lower() != BPY_BASENAME.lower():
            raise ProportionError(
                f"setup file must be named {BPY_BASENAME}, got {setup.name}",
                code="setup_not_found",
                details={"setup": str(setup)},
            )
    else:
        directory = (raw if raw.is_absolute() else cwd / raw).resolve()
        setup = directory / BPY_BASENAME
    if _under_windows_root(setup):
        raise ProportionError(
            f"resolved setup is under Windows system dir: {setup.resolve()}",
            code="setup_cwd_unsafe",
            details={"setup": str(setup.resolve())},
        )
    if not setup.is_file():
        hint = ""
        if json_class:
            hint = (
                f" Run: meshops proportion blockout-emit-setup --recipe {json_abs} "
                f"--out {setup.parent} --force"
            )
        raise ProportionError(
            f"setup_blockout_recipe not found {setup.resolve()}.{hint}",
            code="setup_not_found",
            details={"setup": str(setup.resolve())},
        )
    return setup.resolve()


def _build_argv(blender: Path, setup: Path, *, background: bool) -> list[str]:
    argv: list[str] = [str(blender)]
    if background:
        argv.append("-b")
    argv.extend(["--python", str(setup)])
    if background:
        argv.extend(["--python-exit-code", "1"])
    return argv


def _build_command(blender: Path, setup: Path, *, background: bool) -> str:
    command = "& " + _ps_quote(str(blender))
    if background:
        command += " -b"
    command += " --python " + _ps_quote(str(setup))
    if background:
        command += " --python-exit-code 1"
    return command


def run_blockout_open_setup(
    setup: Path | str,
    *,
    spawn: bool = False,
    background: bool = False,
) -> dict[str, Any]:
    """Print (or optionally spawn) abs Blender --python for RECIPE setup."""
    try:
        blender, source = find_blender_with_source(require=True)
    except EscalateError as exc:
        details = exc.details if isinstance(exc.details, dict) else {}
        raise ProportionError(
            str(exc),
            code="blender_missing",
            details=details,
        ) from exc
    if blender is None:
        raise ProportionError(
            "Blender 5.2 LTS not found. Set MESHOPS_BLENDER to blender.exe.",
            code="blender_missing",
        )
    blender_abs = blender.expanduser().resolve()
    setup_abs = resolve_setup_script(setup)
    argv = _build_argv(blender_abs, setup_abs, background=background)
    spawned = False
    if spawn:
        kwargs: dict[str, Any] = {
            "cwd": str(setup_abs.parent),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            kwargs["start_new_session"] = True
        try:
            subprocess.Popen(argv, **kwargs)
        except OSError as exc:
            raise ProportionError(
                f"failed to spawn Blender: {exc}",
                code="setup_spawn_failed",
                details={"setup": str(setup_abs), "blender": str(blender_abs)},
            ) from exc
        spawned = True
    return {
        "ok": True,
        "blender": str(blender_abs),
        "blender_source": source,
        "setup": str(setup_abs),
        "cwd": str(setup_abs.parent.resolve()),
        "argv": argv,
        "command": _build_command(blender_abs, setup_abs, background=background),
        "spawned": spawned,
        "background": bool(background),
        "honesty": SETUP_LAUNCH_HONESTY,
        "messages": ["setup launch only — not mesh or print success"],
    }
