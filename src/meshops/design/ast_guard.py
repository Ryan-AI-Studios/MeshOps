"""AST lint for agent/template geometry sources (not an OS sandbox).

Deny dangerous modules/calls; allow build123d + pure geometry helpers.
Defense-in-depth: runner still uses scrubbed subprocess + timeout.
"""

from __future__ import annotations

import ast
from typing import Final

from meshops.design.errors import DesignError

# Root module names agents may import.
_ALLOWED_IMPORT_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "build123d",
        "math",
        "typing",
        "typing_extensions",
        "collections",
        "dataclasses",
        "enum",
        "functools",
        "itertools",
        "operator",
        "numbers",
        "decimal",
        "fractions",
        "copy",
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
        "help",  # can load modules interactively in some envs
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


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


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
        # from os import system style — root already denied for os
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _DENIED_CALL_NAMES:
            raise DesignError(
                f"AST denied call to {func.id!r}",
                code="ast_denied",
                details={"call": func.id, "lineno": node.lineno},
            )
        if isinstance(func, ast.Attribute):
            # module.attr(...)
            if isinstance(func.value, ast.Name):
                pair = (func.value.id, func.attr)
                if pair in _DENIED_ATTR_CALLS:
                    raise DesignError(
                        f"AST denied call to {func.value.id}.{func.attr}",
                        code="ast_denied",
                        details={"call": f"{func.value.id}.{func.attr}", "lineno": node.lineno},
                    )
            if func.attr in {"system", "popen", "execv", "execve", "execl", "Popen"}:
                raise DesignError(
                    f"AST denied attribute call .{func.attr}",
                    code="ast_denied",
                    details={"attr": func.attr, "lineno": node.lineno},
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
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
            raise DesignError(
                f"AST denied attribute os.{node.attr}",
                code="ast_denied",
                details={"attr": f"os.{node.attr}", "lineno": node.lineno},
            )
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
