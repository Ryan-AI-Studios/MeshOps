"""AST lint for design geometry sources (DoD-3)."""

from __future__ import annotations

import pytest

from meshops.design.ast_guard import lint_geometry_source
from meshops.design.errors import DesignError


def test_ast__allows_build123d_algebra() -> None:
    src = """
from build123d import Box
result = Box(10, 10, 10)
"""
    lint_geometry_source(src)  # no raise


def test_ast__forbid_subprocess_import() -> None:
    src = "import subprocess\nresult = None\n"
    with pytest.raises(DesignError) as ei:
        lint_geometry_source(src)
    assert ei.value.code == "ast_denied"
    assert "subprocess" in str(ei.value).lower()


def test_ast__forbid_socket_import() -> None:
    with pytest.raises(DesignError) as ei:
        lint_geometry_source("import socket\n")
    assert ei.value.code == "ast_denied"


def test_ast__forbid_os_system() -> None:
    src = "import os\nos.system('echo hi')\n"
    with pytest.raises(DesignError) as ei:
        lint_geometry_source(src)
    assert ei.value.code == "ast_denied"


def test_ast__forbid_eval() -> None:
    with pytest.raises(DesignError) as ei:
        lint_geometry_source("result = eval('1+1')\n")
    assert ei.value.code == "ast_denied"


def test_ast__forbid_open_write() -> None:
    with pytest.raises(DesignError) as ei:
        lint_geometry_source("open('/tmp/x', 'w').write('x')\n")
    assert ei.value.code == "ast_denied"


def test_ast__forbid_ctypes() -> None:
    with pytest.raises(DesignError) as ei:
        lint_geometry_source("import ctypes\n")
    assert ei.value.code == "ast_denied"


def test_ast__forbid_unknown_module() -> None:
    with pytest.raises(DesignError) as ei:
        lint_geometry_source("import requests\n")
    assert ei.value.code == "ast_denied"


def test_ast__forbid_builtins_subscript_open() -> None:
    """Codex P1: __builtins__['open'] must not bypass Name-level open deny."""
    src = 'result = __builtins__["open"]\n'
    with pytest.raises(DesignError) as ei:
        lint_geometry_source(src)
    assert ei.value.code == "ast_denied"


def test_ast__forbid_builtins_import_bypass() -> None:
    src = 'result = __builtins__["__import__"]("subprocess")\n'
    with pytest.raises(DesignError) as ei:
        lint_geometry_source(src)
    assert ei.value.code == "ast_denied"


def test_ast__forbid_getattr() -> None:
    with pytest.raises(DesignError) as ei:
        lint_geometry_source("result = getattr(object, '__class__')\n")
    assert ei.value.code == "ast_denied"


def test_ast__forbid_dunder_class_escape() -> None:
    src = "result = ().__class__.__bases__[0].__subclasses__()\n"
    with pytest.raises(DesignError) as ei:
        lint_geometry_source(src)
    assert ei.value.code == "ast_denied"


def test_ast__forbid_build123d_export_stl_import() -> None:
    """Harness-wrapper: agents must not import/call export_stl (Codex re-review P1)."""
    src = """
from build123d import Box, export_stl
result = Box(10, 10, 10)
export_stl(result, "C:\\\\outside.stl")
"""
    with pytest.raises(DesignError) as ei:
        lint_geometry_source(src)
    assert ei.value.code == "ast_denied"
    assert "export" in str(ei.value).lower()


def test_ast__forbid_build123d_export_attr_call() -> None:
    src = """
import build123d
result = build123d.Box(10, 10, 10)
build123d.export_step(result, "out.step")
"""
    with pytest.raises(DesignError) as ei:
        lint_geometry_source(src)
    assert ei.value.code == "ast_denied"
