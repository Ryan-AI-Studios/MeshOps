"""t2_smooth_spikes — mild Taubin smoothing.

Params (PyMeshLab 2025.7):
  apply_coord_taubin_smoothing(
      lambda_=0.5,   # trailing underscore — Python keyword
      mu=-0.53,
      stepsmoothnum=5,  # ≤ default 10; mild for spikes
  )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meshops.recipes.pymeshlab_io import run_filter_chain

RECIPE_ID = "t2_smooth_spikes"

# Documented defaults for this product (start mild, ≤10)
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53
TAUBIN_STEPS = 5

T2_SMOOTH_STEPS: list[tuple[str, dict[str, Any]]] = [
    (
        "apply_coord_taubin_smoothing",
        {
            "lambda_": TAUBIN_LAMBDA,
            "mu": TAUBIN_MU,
            "stepsmoothnum": TAUBIN_STEPS,
        },
    ),
]


def run_t2_smooth(
    input_path: Path | str,
    output_path: Path | str,
    *,
    unify_vertices: bool = True,
    stepsmoothnum: int = TAUBIN_STEPS,
) -> dict[str, Any]:
    """Run mild Taubin smoothing → binary STL at output_path."""
    steps: list[tuple[str, dict[str, Any]]] = [
        (
            "apply_coord_taubin_smoothing",
            {
                "lambda_": TAUBIN_LAMBDA,
                "mu": TAUBIN_MU,
                "stepsmoothnum": min(stepsmoothnum, 10),
            },
        ),
    ]
    return run_filter_chain(
        input_path,
        output_path,
        steps,
        unify_vertices=unify_vertices,
    )
