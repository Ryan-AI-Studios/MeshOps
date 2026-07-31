"""Subprocess worker: exec geometry source, extract ``result``, MeshOps-export.

Invoked as: ``python -m meshops.design.worker --source ... --stl ... --step ...``
Not agent-authored — export paths are CLI args owned by MeshOps runner.

Defense-in-depth beyond AST: restricted builtins + filtered ``build123d`` import
that strips export/file-write APIs so agent code cannot write files even if
AST misses an alias pattern.
"""

from __future__ import annotations

import argparse
import builtins
import importlib
import sys
import traceback
import types
from pathlib import Path
from typing import Any

SUCCESS_MARKER = "MESHOPS_DESIGN_OK"

# Mirrors ast_guard allowlist roots (keep in sync for worker import filter).
_ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "build123d",
        "math",
        "typing",
        "typing_extensions",
        "dataclasses",
        "enum",
        "itertools",
        "numbers",
        "decimal",
        "fractions",
        "abc",
    }
)

_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "Exception",
        "False",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "None",
        "object",
        "print",
        "property",
        "range",
        "reversed",
        "round",
        "RuntimeError",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "True",
        "tuple",
        "TypeError",
        "ValueError",
        "zip",
        "classmethod",
        "staticmethod",
        "filter",
        "pow",
        "divmod",
        "hash",
        "id",
        "callable",
    }
)


def _is_blocked_export_attr(name: str) -> bool:
    return (
        name.startswith("export_")
        or name.startswith("Export")
        or name
        in {
            "export",
            "exporters",
            "ExportDXF",
            "ExportSVG",
            "ExportBREP",
            "ExportGLTF",
            "ExportSTL",
            "ExportSTEP",
        }
    )


def _filter_build123d_module(real: types.ModuleType) -> types.ModuleType:
    """Return a module proxy that hides export / file-write APIs."""

    class _FilteredBuild123d(types.ModuleType):
        def __init__(self, wrapped: types.ModuleType) -> None:
            super().__init__(wrapped.__name__)
            object.__setattr__(self, "_wrapped", wrapped)

        def __getattr__(self, item: str) -> Any:
            if _is_blocked_export_attr(item):
                raise AttributeError(
                    f"{item!r} is blocked by meshops design harness "
                    "(MeshOps owns export_stl/export_step)"
                )
            return getattr(object.__getattribute__(self, "_wrapped"), item)

        def __dir__(self) -> list[str]:
            wrapped = object.__getattribute__(self, "_wrapped")
            return [n for n in dir(wrapped) if not _is_blocked_export_attr(n)]

    return _FilteredBuild123d(real)


def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Allowlisted import; filter build123d to strip export APIs."""
    if level and level > 0:
        raise ImportError("relative imports denied in design worker")
    root = name.split(".", 1)[0]
    if root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import of {name!r} denied in design worker")
    if fromlist and "*" in fromlist:
        raise ImportError("star-import denied in design worker")
    if fromlist:
        for item in fromlist:
            if item and _is_blocked_export_attr(item):
                raise ImportError(
                    f"import of {name}.{item} denied (export APIs blocked in design worker)"
                )

    real = importlib.import_module(name)
    if root == "build123d":
        filtered = _filter_build123d_module(real)
        # For ``from build123d import Box``, Python still getattr on returned module.
        return filtered
    return real


def _restricted_builtins() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            out[name] = getattr(builtins, name)
    # Bind literals that live as builtins names.
    out["True"] = True
    out["False"] = False
    out["None"] = None
    out["__import__"] = _safe_import
    out["__build_class__"] = builtins.__build_class__
    out["__name__"] = "builtins"
    return out


def _run(source_path: Path, stl_path: Path, step_path: Path) -> int:
    from meshops.design.errors import DesignError
    from meshops.design.export_b123d import export_shape

    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"MESHOPS_DESIGN_ERR:read_source:{exc}", file=sys.stderr)
        return 2

    ns: dict[str, Any] = {
        "__name__": "__meshops_design_source__",
        "__builtins__": _restricted_builtins(),
    }
    try:
        code = compile(text, str(source_path), "exec")
        # Intentional: AST-linted geometry source executed under restricted builtins.
        exec(code, ns, ns)
    except DesignError as exc:
        print(f"MESHOPS_DESIGN_ERR:{exc.code}:{exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"MESHOPS_DESIGN_ERR:cad_kernel_failure:{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 4

    if "result" not in ns:
        print(
            "MESHOPS_DESIGN_ERR:missing_result:top-level 'result' not defined",
            file=sys.stderr,
        )
        return 5

    shape = ns["result"]
    if shape is None:
        print("MESHOPS_DESIGN_ERR:missing_result:result is None", file=sys.stderr)
        return 5

    try:
        meta = export_shape(shape, stl_path=stl_path, step_path=step_path)
    except DesignError as exc:
        print(f"MESHOPS_DESIGN_ERR:{exc.code}:{exc}", file=sys.stderr)
        return 6
    except Exception as exc:
        print(f"MESHOPS_DESIGN_ERR:cad_kernel_failure:{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 7

    print(SUCCESS_MARKER)
    print(f"MESHOPS_DESIGN_STL_BYTES={meta.get('stl_bytes', 0)}")
    print(f"MESHOPS_DESIGN_STEP_BYTES={meta.get('step_bytes', 0)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meshops.design.worker")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--step", type=Path, required=True)
    args = parser.parse_args(argv)
    return _run(args.source, args.stl, args.step)


if __name__ == "__main__":
    raise SystemExit(main())
