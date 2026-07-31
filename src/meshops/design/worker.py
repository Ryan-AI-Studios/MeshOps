"""Subprocess worker: exec geometry source, extract ``result``, MeshOps-export.

Invoked as: ``python -m meshops.design.worker --source ... --stl ... --step ...``
Not agent-authored — export paths are CLI args owned by MeshOps runner.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

SUCCESS_MARKER = "MESHOPS_DESIGN_OK"


def _run(source_path: Path, stl_path: Path, step_path: Path) -> int:
    from meshops.design.errors import DesignError
    from meshops.design.export_b123d import export_shape

    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"MESHOPS_DESIGN_ERR:read_source:{exc}", file=sys.stderr)
        return 2

    ns: dict[str, Any] = {"__name__": "__meshops_design_source__"}
    try:
        code = compile(text, str(source_path), "exec")
        # Intentional: AST-linted geometry source executed in harness worker.
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
