"""Honesty token for proportion reports (Difficulty §12 / N6).

Never claim mesh reconstruction, printability, or artistic success from
pixel proportion measurement alone.
"""

from __future__ import annotations

# Binding freeze — do not rephrase without a schema bump.
PROPORTION_HONESTY = "proportion_measurement_not_mesh_or_print_success"
GUIDE_HONESTY = "proportion_guides_not_mesh_or_print_success"
CAPTURE_HONESTY = "proportion_capture_not_mesh_or_print_success"
DEPTH_HONESTY = "proportion_depth_samples_not_mesh_or_print_success"
RECIPE_HONESTY = "proportion_blockout_recipe_not_mesh_or_print_success"
