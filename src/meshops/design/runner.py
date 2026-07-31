"""AST lint + scrubbed subprocess harness for geometry sources."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meshops.design.ast_guard import lint_geometry_source
from meshops.design.errors import DesignError
from meshops.design.models import DEFAULT_EXPORT_STEP_KWARGS, DEFAULT_EXPORT_STL_KWARGS

# Keep in sync with meshops.design.worker.SUCCESS_MARKER (avoid importing worker module
# so `python -m meshops.design.worker` does not hit runpy double-import warning).
SUCCESS_MARKER = "MESHOPS_DESIGN_OK"
DEFAULT_TIMEOUT_S = 60.0
_STDERR_TRUNCATE = 4000

# Env keys safe to pass through for starting Python on Windows/POSIX.
_SAFE_ENV_KEYS: frozenset[str] = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SystemRoot",
        "SYSTEMROOT",
        "WINDIR",
        "SystemDrive",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "USERNAME",
        "USER",
        "LOGNAME",
        "LOCALAPPDATA",
        "APPDATA",
        "VIRTUAL_ENV",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PYTHONPATH",
        "PYTHONHOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
    }
)


def scrubbed_environ() -> dict[str, str]:
    """Minimal env: no secrets; enough for venv Python + native OCP DLLs."""
    env: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Ensure package import works even if env was sparse
    try:
        import meshops

        pkg_parent = str(Path(meshops.__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH", "")
        parts = [pkg_parent] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
    except Exception:
        pass
    return env


@dataclass(frozen=True, slots=True)
class RunnerExportResult:
    """Staging export outcome from sandboxed geometry run."""

    stl_path: Path
    step_path: Path
    source_path: Path
    export_stl: dict[str, Any]
    export_step: dict[str, Any]
    runner_meta: dict[str, Any]
    stdout: str
    stderr: str


def run_geometry_source(
    source: str | Path,
    *,
    staging_dir: Path | str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    stl_name: str = "part.stl",
    step_name: str = "part.step",
) -> RunnerExportResult:
    """AST-lint then subprocess-run geometry; MeshOps exports STEP+STL under staging.

    Geometry script contract: top-level ``result`` is a build123d solid/part.
    Agents must not import os/subprocess or write files.
    """
    if isinstance(source, Path):
        source_text = source.read_text(encoding="utf-8")
        source_label = str(source)
    else:
        source_text = source
        source_label = "<geometry>"

    lint_geometry_source(source_text, filename=source_label)

    # Ensure build123d present in this env (clear error before subprocess)
    try:
        # Optional meshops[design] extra — ignore missing for core-CI basedpyright.
        import build123d  # noqa: F401  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise DesignError(
            "build123d is not installed; install with: uv sync --extra design "
            "(meshops[design] / build123d==0.11.1)",
            code="missing_dependency",
            details={"package": "build123d"},
        ) from exc

    own_tmp = staging_dir is None
    stage = (
        Path(staging_dir)
        if staging_dir is not None
        else Path(tempfile.mkdtemp(prefix="meshops_design_"))
    )
    stage.mkdir(parents=True, exist_ok=True)

    source_path = stage / "source.py"
    stl_path = stage / stl_name
    step_path = stage / step_name
    source_path.write_text(source_text, encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "meshops.design.worker",
        "--source",
        str(source_path),
        "--stl",
        str(stl_path),
        "--step",
        str(step_path),
    ]
    env = scrubbed_environ()

    try:
        proc = subprocess.run(
            cmd,
            shell=False,
            cwd=str(stage),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        raise DesignError(
            f"design runner timed out after {timeout_s}s",
            code="timeout",
            details={"timeout_s": timeout_s, "stderr": stderr[:_STDERR_TRUNCATE]},
        ) from exc
    except OSError as exc:
        raise DesignError(
            f"design runner failed to start: {exc}",
            code="runner_crash",
        ) from exc

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    truncated_err = stderr[-_STDERR_TRUNCATE:] if stderr else ""

    runner_meta: dict[str, Any] = {
        "returncode": proc.returncode,
        "timeout_s": timeout_s,
        "cmd": [*cmd[:3], "..."],  # avoid huge paths in manifest noise
        "own_tmp": own_tmp,
    }

    if proc.returncode != 0:
        code = "runner_crash"
        if "cad_kernel_failure" in stderr or "MESHOPS_DESIGN_ERR:cad_kernel" in stderr:
            code = "cad_kernel_failure"
        elif "missing_result" in stderr:
            code = "missing_result"
        elif "export_failed" in stderr or "MESHOPS_DESIGN_ERR:export_failed" in stderr:
            code = "export_failed"
        elif "multi_solid" in stderr:
            code = "multi_solid"
        elif "missing_dependency" in stderr:
            code = "missing_dependency"
        # Negative returncode on POSIX = signal
        if proc.returncode < 0:
            code = "cad_kernel_failure"
            runner_meta["signal"] = -proc.returncode
        raise DesignError(
            f"design runner exited {proc.returncode}: {truncated_err or stdout[:500]}",
            code=code,
            details={"returncode": proc.returncode, "stderr": truncated_err},
        )

    if SUCCESS_MARKER not in stdout:
        raise DesignError(
            "design runner completed without success marker",
            code="runner_crash",
            details={"stdout": stdout[:500], "stderr": truncated_err},
        )

    if not stl_path.is_file() or stl_path.stat().st_size <= 0:
        raise DesignError("runner reported ok but STL missing/empty", code="export_failed")
    if not step_path.is_file() or step_path.stat().st_size <= 0:
        raise DesignError("runner reported ok but STEP missing/empty", code="export_failed")

    return RunnerExportResult(
        stl_path=stl_path,
        step_path=step_path,
        source_path=source_path,
        export_stl=dict(DEFAULT_EXPORT_STL_KWARGS),
        export_step=dict(DEFAULT_EXPORT_STEP_KWARGS),
        runner_meta=runner_meta,
        stdout=stdout,
        stderr=stderr,
    )
