"""Honesty token for proportion reports (Difficulty §12 / N6).

Never claim mesh reconstruction, printability, or artistic success from
pixel proportion measurement alone.

0024 consumer notes (measurement only):
- foot/cranial depth_bands: use depth_frac / y_mid, not raw y_front as toe Y (C4)
- breast_lower* assist ids are prep-only until 0027 (no fuse pair / diameter)
"""

from __future__ import annotations

# Binding freeze — do not rephrase without a schema bump.
PROPORTION_HONESTY = "proportion_measurement_not_mesh_or_print_success"
GUIDE_HONESTY = "proportion_guides_not_mesh_or_print_success"
CAPTURE_HONESTY = "proportion_capture_not_mesh_or_print_success"
DEPTH_HONESTY = "proportion_depth_samples_not_mesh_or_print_success"
RECIPE_HONESTY = "proportion_blockout_recipe_not_mesh_or_print_success"
HEATMAP_HONESTY = "proportion_depth_heatmap_not_mesh_or_print_success"
HINT_HONESTY = "proportion_depth_hint_not_mesh_or_print_success"
SILHOUETTE_HONESTY = "proportion_silhouette_compare_not_mesh_or_print_success"
TEMPLATE_HONESTY = "proportion_body_template_not_mesh_or_print_success"
CONSTRAINT_HONESTY = "proportion_blockout_constraints_not_mesh_or_print_success"
OPTIMIZE_HONESTY = "proportion_blockout_optimize_not_mesh_or_print_success"
