"""AST lint for agent/template geometry sources (not an OS sandbox).

Deny dangerous modules/calls; allow build123d + pure geometry helpers.
Defense-in-depth: runner still uses scrubbed subprocess + timeout + worker
import/export stripping (see worker.py).
"""

from __future__ import annotations

import ast
from typing import Final

from meshops.design.errors import DesignError

# Root module names agents may import (minimal geometry surface).
# NOTE: operator/functools intentionally excluded (attrgetter / partial escapes).
_ALLOWED_IMPORT_ROOTS: Final[frozenset[str]] = frozenset(
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

# Explicit deny even if someone tries to smuggle via alias.
_DENIED_IMPORT_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "subprocess",
        "socket",
        "ctypes",
        "importlib",
        "pickle",
        "pathlib",
        "shutil",
        "tempfile",
        "http",
        "httpx",
        "urllib",
        "requests",
        "os",
        "sys",
        "pty",
        "multiprocessing",
        "threading",
        "concurrent",
        "asyncio",
        "signal",
        "code",
        "codeop",
        "inspect",
        "gc",
        "builtins",  # re-import of builtins blocked at import; Name calls still checked
        "io",
        "socketserver",
        "ssl",
        "ftplib",
        "telnetlib",
        "webbrowser",
        "runpy",
        "pkgutil",
        "zipimport",
        "mmap",
        "resource",
        "fcntl",
        "msvcrt",
        "winreg",
        "nt",
        "posix",
        "operator",  # attrgetter escapes
        "functools",  # partial / wraps
        "collections",  # keep surface small
        "copy",
    }
)

_DENIED_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "execfile",
        "help",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "memoryview",
        "bytearray",
        "bytes",
        "chr",
        "ord",
        "ascii",
        "format",
        "repr",
    }
)

# Names that must never appear unbound (sandbox escape surface).
_DENIED_NAME_IDS: Final[frozenset[str]] = frozenset(
    {
        "__builtins__",
        "__builtin__",
        "__import__",
        "__loader__",
        "__spec__",
        "__package__",
        "builtins",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__dict__",
        "__reduce__",
        "__reduce_ex__",
    }
)

# Dunder attributes used in classic sandbox escapes (geometry never needs these).
_DENIED_DUNDER_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "__builtins__",
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__dict__",
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__reduce__",
        "__reduce_ex__",
        "__import__",
        "__loader__",
        "__spec__",
        "__init_subclass__",
        "__subclasshook__",
    }
)

_DENIED_ATTR_CALLS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("os", "system"),
        ("os", "popen"),
        ("os", "exec"),
        ("os", "execl"),
        ("os", "execle"),
        ("os", "execlp"),
        ("os", "execlpe"),
        ("os", "execv"),
        ("os", "execve"),
        ("os", "execvp"),
        ("os", "execvpe"),
        ("os", "spawn"),
        ("os", "spawnl"),
        ("os", "spawnle"),
        ("os", "spawnlp"),
        ("os", "spawnlpe"),
        ("os", "spawnv"),
        ("os", "spawnve"),
        ("os", "spawnvp"),
        ("os", "spawnvpe"),
        ("os", "fork"),
        ("os", "forkpty"),
        ("os", "kill"),
        ("os", "remove"),
        ("os", "unlink"),
        ("os", "rmdir"),
        ("os", "removedirs"),
        ("os", "rename"),
        ("os", "replace"),
        ("os", "makedirs"),
        ("os", "mkdir"),
        ("os", "chmod"),
        ("os", "chown"),
        ("os", "environ"),
        ("subprocess", "run"),
        ("subprocess", "Popen"),
        ("subprocess", "call"),
        ("subprocess", "check_call"),
        ("subprocess", "check_output"),
        ("importlib", "import_module"),
        ("importlib", "__import__"),
    }
)

# Agent geometry must not write files — MeshOps owns export (harness-wrapper).
_DENIED_EXPORT_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "export_stl",
        "export_step",
        "export_brep",
        "export_gltf",
        "export_binary_stl",
        "export_svg",
        "export_dxf",
        "export_stl_ascii",
        "exporters",
        "export",
        "ExportDXF",
        "ExportSVG",
        "ExportBREP",
        "ExportGLTF",
        "ExportSTL",
        "ExportSTEP",
    }
)

# Attribute methods that write files (build123d Export* classes).
_DENIED_IO_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "write",
        "save",
        "dump",
        "to_file",
        "export",
    }
)


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def _is_export_name(name: str) -> bool:
    return (
        name in _DENIED_EXPORT_CALL_NAMES or name.startswith("export_") or name.startswith("Export")
    )


def _check_import_name(mod: str, *, node: ast.AST) -> None:
    root = _root_module(mod)
    if root in _DENIED_IMPORT_ROOTS or root.startswith("http"):
        raise DesignError(
            f"AST denied import of {mod!r}",
            code="ast_denied",
            details={"module": mod, "lineno": getattr(node, "lineno", None)},
        )
    if root not in _ALLOWED_IMPORT_ROOTS:
        raise DesignError(
            f"AST denied import of {mod!r} (not on geometry allowlist)",
            code="ast_denied",
            details={"module": mod, "lineno": getattr(node, "lineno", None)},
        )


def _deny(msg: str, *, node: ast.AST, **details: object) -> None:
    raise DesignError(
        msg,
        code="ast_denied",
        details={"lineno": getattr(node, "lineno", None), **details},
    )


class _AstGuardVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            _check_import_name(alias.name, node=node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            raise DesignError(
                "AST denied relative import",
                code="ast_denied",
                details={"lineno": node.lineno},
            )
        if node.module is None:
            raise DesignError(
                "AST denied import with empty module",
                code="ast_denied",
                details={"lineno": node.lineno},
            )
        _check_import_name(node.module, node=node)
        for alias in node.names:
            if alias.name == "*":
                _deny(
                    "AST denied star-import (from … import *)",
                    node=node,
                    name="*",
                )
            base = alias.name.split(".", 1)[0]
            if _is_export_name(base):
                _deny(
                    f"AST denied import of export API {alias.name!r} (MeshOps harness owns export)",
                    node=node,
                    name=alias.name,
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _DENIED_NAME_IDS or _is_export_name(node.id):
            _deny(f"AST denied name {node.id!r}", node=node, name=node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block dunder escape surfaces (__class__, __subclasses__, …).
        if node.attr in _DENIED_DUNDER_ATTRS or (
            node.attr.startswith("__") and node.attr.endswith("__") and node.attr != "__doc__"
        ):
            _deny(
                f"AST denied dunder attribute .{node.attr}",
                node=node,
                attr=node.attr,
            )
        # Deny export API references even when assigned to aliases
        # (e.g. writer = build123d.export_stl; writer(...)).
        if _is_export_name(node.attr):
            _deny(
                f"AST denied export attribute .{node.attr} (MeshOps harness owns export)",
                node=node,
                attr=node.attr,
            )
        # os.environ / os.system as value (not only call)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr
            in {
                "system",
                "popen",
                "environ",
                "execv",
                "execve",
                "execl",
                "execlp",
                "execle",
                "execvpe",
                "execvp",
            }
        ):
            _deny(f"AST denied attribute os.{node.attr}", node=node, attr=f"os.{node.attr}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # __builtins__["open"] / builtins["__import__"] style bypasses.
        if isinstance(node.value, ast.Name) and node.value.id in {
            "__builtins__",
            "__builtin__",
            "builtins",
        }:
            _deny(
                f"AST denied subscript of {node.value.id}",
                node=node,
                target=node.value.id,
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _DENIED_CALL_NAMES:
            _deny(f"AST denied call to {func.id!r}", node=node, call=func.id)
        if isinstance(func, ast.Name) and _is_export_name(func.id):
            _deny(
                f"AST denied export call {func.id!r} (MeshOps harness owns export)",
                node=node,
                call=func.id,
            )
        if isinstance(func, ast.Attribute):
            # module.attr(...)
            if isinstance(func.value, ast.Name):
                pair = (func.value.id, func.attr)
                if pair in _DENIED_ATTR_CALLS:
                    _deny(
                        f"AST denied call to {func.value.id}.{func.attr}",
                        node=node,
                        call=f"{func.value.id}.{func.attr}",
                    )
            if func.attr in {
                "system",
                "popen",
                "execv",
                "execve",
                "execl",
                "Popen",
                "open",
                "__import__",
            }:
                _deny(
                    f"AST denied attribute call .{func.attr}",
                    node=node,
                    attr=func.attr,
                )
            if _is_export_name(func.attr):
                _deny(
                    f"AST denied export attribute call .{func.attr} (MeshOps harness owns export)",
                    node=node,
                    attr=func.attr,
                )
            # ExportDXF().write(...) / .save(...)
            if func.attr in _DENIED_IO_ATTRS:
                _deny(
                    f"AST denied I/O attribute call .{func.attr} "
                    "(geometry sources must not write files)",
                    node=node,
                    attr=func.attr,
                )
        # Deny calls whose callee is a subscript (__builtins__["open"](...)).
        if isinstance(func, ast.Subscript):
            _deny("AST denied call via subscript (possible builtins bypass)", node=node)
        self.generic_visit(node)


def lint_geometry_source(source: str, *, filename: str = "<geometry>") -> None:
    """Parse and lint geometry source; raise DesignError(code=ast_denied) on fail."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise DesignError(
            f"geometry source syntax error: {exc}",
            code="ast_denied",
            details={"filename": filename, "lineno": exc.lineno},
        ) from exc
    _AstGuardVisitor().visit(tree)
