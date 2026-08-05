"""MCPServer factory + tool registration (thin closures over work_root)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from meshops import __version__
from meshops.mcp import tools as T

SERVER_INSTRUCTIONS = (
    "MeshOps local mesh OS (CLI-equivalent tools). "
    "ALWAYS mesh_triage before mesh_repair/mesh_export. "
    "Classify before mutate. Export guards enforce wipeout hard-fail (Difficulty §6). "
    "T3/T4 repair refused. T3 preview never auto-promotes (N6). "
    "Organic finalize is untrusted — re-triage. "
    "design_organic_api is post-plateau hosted multi-view fallback only — never default. "
    "Proportion tools (mesh_proportion_*) are measurement/authoring only — "
    "never mesh or print success (N6). "
    "work_root is server-bound (MESHOPS_WORK or ./work); host must set cwd to repo."
)

# Catalog of registered MCP tool names (tests + docs). No mega agent.
TOOL_NAMES: frozenset[str] = frozenset(
    {
        "mesh_version",
        "mesh_doctor",
        "mesh_list_recipes",
        "mesh_ingest",
        "mesh_triage",
        "mesh_render",
        "mesh_report",
        "mesh_repair",
        "mesh_export",
        "mesh_accept_revision",
        "mesh_accept_candidate",
        "mesh_promote_working",
        "mesh_diff_views",
        "mesh_slice",
        "mesh_extract_roi",
        "mesh_preview_t3",
        "mesh_blender_handoff",
        "mesh_import_sculpt",
        "design_from_spec",
        "design_run",
        "organic_create",
        "organic_pass",
        "organic_status",
        "organic_plateau",
        "organic_finalize",
        "design_organic_api",
        "mesh_proportion_template",
        "mesh_proportion_templates",
        "mesh_proportion_apply_template",
        "mesh_proportion_analyze",
        "mesh_proportion_show",
        "mesh_proportion_scaffold",
        "mesh_proportion_guides",
        "mesh_proportion_capture",
        "mesh_proportion_depth_samples",
        "mesh_proportion_blockout_recipe",
        "mesh_proportion_anatomy_profiles",
        "mesh_proportion_blockout_validate_constraints",
        "mesh_proportion_blockout_optimize",
        "mesh_proportion_skeleton_build",
        "mesh_proportion_depth_heatmap",
        "mesh_proportion_depth_hint",
        "mesh_proportion_silhouette_compare",
    }
)


def resolve_work_root(override: Path | None = None) -> Path:
    """Resolve server-level work_root (R3).

    Priority: explicit *override* → ``MESHOPS_WORK`` env → ``\"work\"``.
    Always ``expanduser().resolve()`` against process CWD.
    """
    if override is not None:
        return Path(override).expanduser().resolve()
    return Path(os.environ.get("MESHOPS_WORK", "work")).expanduser().resolve()


def build_server(work_root: Path | None = None) -> Any:
    """Build an ``MCPServer`` with all catalog tools bound to *work_root*.

    *work_root* resolves via :func:`resolve_work_root` (override or MESHOPS_WORK / ``work``).
    Tools do **not** take work_root in their schema (R3).
    """
    from mcp.server import MCPServer

    wr = resolve_work_root(work_root)
    mcp = MCPServer(
        "meshops",
        version=__version__,
        instructions=SERVER_INSTRUCTIONS,
    )

    @mcp.tool()
    def mesh_version() -> dict[str, str]:
        """Return MeshOps package version."""
        return T.mesh_version()

    @mcp.tool()
    def mesh_doctor(require: list[str] | None = None) -> dict[str, Any]:
        """Diagnose Python env, core packages, Blender, Orca, F3D (ops health).

        require tokens: core | blender | orca | f3d | design | all (default: core).
        """
        return T.mesh_doctor(wr, require=require)

    @mcp.tool()
    def mesh_list_recipes() -> dict[str, list[str]]:
        """List allowlisted T1/T2 repair recipe ids."""
        return T.mesh_list_recipes()

    @mcp.tool()
    def mesh_ingest(path: str) -> dict[str, Any]:
        """Ingest STL into work/<mesh_id>/ (non-destructive; never overwrites source).

        PRECONDITION: path must exist and be a readable STL.
        """
        return T.mesh_ingest(wr, path=path)

    @mcp.tool()
    def mesh_triage(mesh_id: str) -> dict[str, Any]:
        """Classify-only triage → diagnostics.json. No mutation.

        PRECONDITION: mesh_id from mesh_ingest.
        ALWAYS run this before mesh_repair / mesh_export / mutating tools.
        """
        return T.mesh_triage(wr, mesh_id=mesh_id)

    @mcp.tool()
    def mesh_render(mesh_id: str) -> dict[str, Any]:
        """F3D offscreen RGB + visual depth views (not metric rangefinder).

        PRECONDITION: mesh_id from mesh_ingest.
        """
        return T.mesh_render(wr, mesh_id=mesh_id)

    @mcp.tool()
    def mesh_report(mesh_id: str) -> dict[str, Any]:
        """Generate report.md from diagnostics + views.

        PRECONDITION: mesh_triage (and preferably mesh_render) already run.
        """
        return T.mesh_report(wr, mesh_id=mesh_id)

    @mcp.tool()
    def mesh_repair(
        mesh_id: str,
        recipe: str,
        parent_rev: str | None = None,
        no_diff: bool = False,
    ) -> dict[str, Any]:
        """Run allowlisted T1/T2 recipe → atomic rev + guards.

        PRECONDITION: mesh_triage first. T3/T4 classes refuse repair.
        recipe: t1_clean | t2_smooth_spikes | t2_close_small_holes.
        """
        return T.mesh_repair(
            wr,
            mesh_id=mesh_id,
            recipe=recipe,
            parent_rev=parent_rev,
            no_diff=no_diff,
        )

    @mcp.tool()
    def mesh_export(
        mesh_id: str,
        out: str,
        rev: str | None = None,
    ) -> dict[str, Any]:
        """Guarded export of original or rev; fail-closed on wipeout (Difficulty §6).

        PRECONDITION: mesh_triage preferred; export guards always run.
        """
        return T.mesh_export(wr, mesh_id=mesh_id, out=out, rev=rev)

    @mcp.tool()
    def mesh_accept_revision(
        mesh_id: str,
        rev: str,
        require_views: bool = True,
        allow_stubs: bool = True,
        require_slice: bool = False,
    ) -> dict[str, Any]:
        """Accept a job revision (raise if not ok).

        PRECONDITION: rev from mesh_repair / mesh_import_sculpt / design.
        """
        return T.mesh_accept_revision(
            wr,
            mesh_id=mesh_id,
            rev=rev,
            require_views=require_views,
            allow_stubs=allow_stubs,
            require_slice=require_slice,
        )

    @mcp.tool()
    def mesh_accept_candidate(
        baseline_path: str,
        candidate_path: str,
        require_views: bool = True,
        allow_stubs: bool = True,
        require_slice: bool = False,
    ) -> dict[str, Any]:
        """Accept candidate mesh vs baseline paths (raise if not ok).

        Used for design self-check and path-based acceptance.
        """
        return T.mesh_accept_candidate(
            wr,
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            require_views=require_views,
            allow_stubs=allow_stubs,
            require_slice=require_slice,
        )

    @mcp.tool()
    def mesh_promote_working(mesh_id: str, rev: str) -> dict[str, Any]:
        """Promote accepted rev mesh to working.ply (runs accept first; refuse if not ok).

        PRECONDITION: rev already acceptable; never overwrites original.stl.
        """
        return T.mesh_promote_working(wr, mesh_id=mesh_id, rev=rev)

    @mcp.tool()
    def mesh_diff_views(mesh_id: str, rev: str) -> dict[str, Any]:
        """Compare views: parent/original baseline vs rev (never working.ply as baseline).

        PRECONDITION: rev exists under the job.
        """
        return T.mesh_diff_views(wr, mesh_id=mesh_id, rev=rev)

    @mcp.tool()
    def mesh_slice(
        mesh_id: str,
        rev: str | None = None,
        profile: str = "default",
        orient: bool = False,
        arrange: bool = False,
        allow_reorient_retry: bool = False,
    ) -> dict[str, Any]:
        """OrcaSlicer printability oracle. Raises if not pass.

        PRECONDITION: mesh_id job exists; Orca 2.4.x installed (MESHOPS_ORCA).
        Candidate: rev mesh → working.ply → original.stl.
        """
        return T.mesh_slice(
            wr,
            mesh_id=mesh_id,
            rev=rev,
            profile=profile,
            orient=orient,
            arrange=arrange,
            allow_reorient_retry=allow_reorient_retry,
        )

    @mcp.tool()
    def mesh_extract_roi(
        mesh_id: str,
        bbox: list[float] | None = None,
        from_sheet_heuristic: bool = False,
    ) -> dict[str, Any]:
        """Create ROI package under rois/<roi_id>/ (manual bbox preferred).

        PRECONDITION: mesh_id from mesh_ingest. Confirm laterality before mutate.
        bbox: [xmin,ymin,zmin,xmax,ymax,zmax] world coords.
        """
        return T.mesh_extract_roi(
            wr,
            mesh_id=mesh_id,
            bbox=bbox,
            from_sheet_heuristic=from_sheet_heuristic,
        )

    @mcp.tool()
    def mesh_preview_t3(
        mesh_id: str,
        roi: str | None = None,
    ) -> dict[str, Any]:
        """Honest T3 preview package. NEVER auto-promotes (N6). ok=False is intentional.

        PRECONDITION: mesh_triage. Not fixed / not print-ready.
        """
        return T.mesh_preview_t3(wr, mesh_id=mesh_id, roi=roi)

    @mcp.tool()
    def mesh_blender_handoff(
        mesh_id: str,
        roi: str,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Build Blender 5.2 LTS handoff .blend + instructions (not Blender-MCP).

        PRECONDITION: mesh_extract_roi for roi. Human/agent sculpt in Blender GUI.
        """
        return T.mesh_blender_handoff(wr, mesh_id=mesh_id, roi=roi, timeout=timeout)

    @mcp.tool()
    def mesh_import_sculpt(
        mesh_id: str,
        path: str,
        approve: bool,
        no_diff: bool = False,
    ) -> dict[str, Any]:
        """Import sculpt STL as rev + sculpt-tier accept.

        PRECONDITION: human/agent sculpt responsibility acknowledged.
        approve=true is REQUIRED (no default) — must pass approve=true explicitly.
        Does NOT auto-promote to working.ply.
        """
        return T.mesh_import_sculpt(
            wr,
            mesh_id=mesh_id,
            path=path,
            approve=approve,
            no_diff=no_diff,
        )

    @mcp.tool()
    def design_from_spec(
        template: str = "bracket_m4",
        hole_spacing: float = 40.0,
        wall: float = 3.0,
        thickness: float = 4.0,
        hole_diameter: float = 4.2,
        timeout: float = 60.0,
        no_diff: bool = False,
    ) -> dict[str, Any]:
        """Parametric T7 template → design job (needs meshops[design]).

        Raises if design extra missing or pipeline not ok.
        """
        return T.design_from_spec(
            wr,
            template=template,
            hole_spacing=hole_spacing,
            wall=wall,
            thickness=thickness,
            hole_diameter=hole_diameter,
            timeout=timeout,
            no_diff=no_diff,
        )

    @mcp.tool()
    def design_run(
        source: str,
        timeout: float = 60.0,
        no_diff: bool = False,
    ) -> dict[str, Any]:
        """Run build123d geometry source through design harness (needs meshops[design]).

        PRECONDITION: source path is a valid .py geometry file.
        """
        return T.design_run(wr, source=source, timeout=timeout, no_diff=no_diff)

    @mcp.tool()
    def organic_create(
        prompt: str,
        style: str = "",
        refs: list[str] | None = None,
        recipe: str = "simple_bust",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Create T6 organic authoring session under work/<session_id>/organic/.

        Local Blender recipes only. Hosted multi-view is design_organic_api after plateau.
        """
        return T.organic_create(
            wr,
            prompt=prompt,
            style=style,
            refs=refs,
            recipe=recipe,
            session_id=session_id,
        )

    @mcp.tool()
    def organic_pass(
        session_id: str,
        recipe: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one Blender organic recipe pass + multi-view evidence.

        PRECONDITION: organic_create. Requires Blender 5.2 LTS.
        """
        return T.organic_pass(wr, session_id=session_id, recipe=recipe, params=params)

    @mcp.tool()
    def organic_status(session_id: str) -> dict[str, Any]:
        """Show organic session manifest status."""
        return T.organic_status(wr, session_id=session_id)

    @mcp.tool()
    def organic_plateau(session_id: str, reason: str) -> dict[str, Any]:
        """Mark session plateau (machine-readable criteria for 0007 hosted-fallback gate).

        PRECONDITION: successful passes with views. reason ≥15 chars, not filler.
        """
        return T.organic_plateau(wr, session_id=session_id, reason=reason)

    @mcp.tool()
    def organic_finalize(
        session_id: str,
        accept: bool = False,
    ) -> dict[str, Any]:
        """Finalize organic session → ingest + triage (untrusted — re-triage).

        PRECONDITION: at least one successful pass. Output is untrusted mesh job.
        """
        return T.organic_finalize(wr, session_id=session_id, accept=accept)

    @mcp.tool()
    def design_organic_api(
        justify: str,
        session_id: str | None = None,
        plateau: str | None = None,
        prompt: str = "",
        provider: str = "meshy",
        views: list[str] | None = None,
        views_from: str = "latest",
        accept: bool = False,
        accept_policy: str = "export",
    ) -> dict[str, Any]:
        """Hosted multi-view image-to-3D fallback after 0006 plateau gate.

        PRECONDITION: plateau.json with allows_hosted_fallback=true (or session_id
        with plateau). Requires ≥2 views + operator justify. Never default organic path.
        Optional accept uses accept_policy export|sculpt|design (default export).
        Raises on gate/provider fail (is_error path).
        """
        return T.design_organic_api(
            wr,
            justify=justify,
            session_id=session_id,
            plateau=plateau,
            prompt=prompt,
            provider=provider,
            views=views,
            views_from=views_from,
            accept=accept,
            accept_policy=accept_policy,
        )

    @mcp.tool()
    def mesh_proportion_template(out: str = "landmarks_assist.json") -> dict[str, Any]:
        """Write blank landmarks_assist.json template.

        Authoring aid only — not mesh reconstruction or print success (N6 /
        proportion_measurement_not_mesh_or_print_success).
        """
        return T.mesh_proportion_template(wr, out=out)

    @mcp.tool()
    def mesh_proportion_templates() -> dict[str, Any]:
        """List body template ids + descriptions.

        Authoring priors only — proportion_body_template_not_mesh_or_print_success (N6).
        Raises ProportionError on failure (never ok:false success).
        """
        return T.mesh_proportion_templates(wr)

    @mcp.tool()
    def mesh_proportion_apply_template(
        report: str,
        template: str,
        out: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Apply sex/archetype body template to a proportion report.

        Writes template_applied.json + template_constants.py. Authoring only —
        proportion_body_template_not_mesh_or_print_success (N6). Raises ProportionError
        on failure (never ok:false success).
        """
        return T.mesh_proportion_apply_template(
            wr,
            report=report,
            template=template,
            out=out,
            force=force,
        )

    @mcp.tool()
    def mesh_proportion_analyze(
        views_dir: str,
        landmarks: str | None = None,
        height_m: float | None = None,
        out: str | None = None,
        overlays: bool = False,
        partial_ok: bool = False,
        attach_session: str | None = None,
    ) -> dict[str, Any]:
        """Analyze multi-view package → proportion_report.json.

        Measurement only — not mesh or print success (N6). Optional attach_session
        copies report under organic session proportion/ note (soft path).
        """
        return T.mesh_proportion_analyze(
            wr,
            views_dir=views_dir,
            landmarks=landmarks,
            height_m=height_m,
            out=out,
            overlays=overlays,
            partial_ok=partial_ok,
            attach_session=attach_session,
        )

    @mcp.tool()
    def mesh_proportion_show(report: str) -> dict[str, Any]:
        """Load proportion_report.json → {report, markdown}. Measurement only (N6)."""
        return T.mesh_proportion_show(wr, report=report)

    @mcp.tool()
    def mesh_proportion_scaffold(
        out: str,
        dual: bool = False,
        mode: str | None = None,
        height_m: float | None = None,
        subject: str | None = None,
        pose: str = "a_pose",
        with_template: bool = False,
        stub_images: bool = False,
        include_back_stub: bool = False,
        include_top_stub: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Create multi-view package layout + checklist. Layout only — not mesh/print (N6)."""
        return T.mesh_proportion_scaffold(
            wr,
            out=out,
            dual=dual,
            mode=mode,
            height_m=height_m,
            subject=subject,
            pose=pose,
            with_template=with_template,
            stub_images=stub_images,
            include_back_stub=include_back_stub,
            include_top_stub=include_top_stub,
            force=force,
        )

    @mcp.tool()
    def mesh_proportion_guides(
        report: str,
        out: str,
        format: str = "both",
        seeds: bool = False,
        front_plane_seeds: bool = False,
        quiet_null_y: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Emit LM_* guide empties + optional SEED_* from report.

        Authoring aids only — not mesh or print success (N6).
        front_plane_seeds: with seeds, allow limb capsules when y_m null.
        quiet_null_y: suppress front-plane-only empty messages.
        """
        return T.mesh_proportion_guides(
            wr,
            report=report,
            out=out,
            format=format,
            seeds=seeds,
            front_plane_seeds=front_plane_seeds,
            quiet_null_y=quiet_null_y,
            force=force,
        )

    @mcp.tool()
    def mesh_proportion_capture(
        source: str | None = None,
        in_path: str | None = None,
        out: str | None = None,
        views_dir: str | None = None,
        pose: str | None = None,
        multi_figure: bool | None = None,
        merge: str | None = None,
        prefer_merge: bool = False,
        default_confidence: float | None = None,
        force: bool = False,
        emit_dump_script: str | None = None,
        attach_session: str | None = None,
    ) -> dict[str, Any]:
        """Capture/fill landmarks_assist.json from px, ASSIST_* dump, or reproject.

        Authoring aid only — proportion_capture_not_mesh_or_print_success (N6).
        Raises ProportionError on failure (never ok:false success).
        """
        return T.mesh_proportion_capture(
            wr,
            source=source,
            in_path=in_path,
            out=out,
            views_dir=views_dir,
            pose=pose,
            multi_figure=multi_figure,
            merge=merge,
            prefer_merge=prefer_merge,
            default_confidence=default_confidence,
            force=force,
            emit_dump_script=emit_dump_script,
            attach_session=attach_session,
        )

    @mcp.tool()
    def mesh_proportion_depth_samples(
        report: str,
        out: str,
        mesh: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Export sparse depth samples (+ optional mesh Y-ray deltas) from a report.

        Authoring measurement aid only — proportion_depth_samples_not_mesh_or_print_success (N6).
        Raises ProportionError on failure (never ok:false success).
        """
        return T.mesh_proportion_depth_samples(
            wr,
            report=report,
            out=out,
            mesh=mesh,
            force=force,
        )

    @mcp.tool()
    def mesh_proportion_anatomy_profiles() -> dict[str, Any]:
        """List torso/limb anatomy profile packs (authoring only — N6)."""
        return T.mesh_proportion_anatomy_profiles(wr)

    @mcp.tool()
    def mesh_proportion_blockout_recipe(
        report: str,
        out: str,
        format: str = "both",
        depth_at_landmarks: str | None = None,
        limbs: bool = True,
        force: bool = False,
        torso: str = "trap",
        glute: str = "oval",
        nofuse: bool = False,
        breast_tilt_deg: float | None = None,
        template_applied: str | None = None,
        profiles: str | None = None,
        skeleton: str | None = None,
        face: bool = False,
        hair: str = "none",
        neckline: str = "none",
        hands: bool = False,
        feet: bool = False,
        fingers: str = "mitten",
        toes: str = "wedge",
    ) -> dict[str, Any]:
        """Emit RECIPE_* blockout primitives (trap/ovals/softs) from a report.

        Authoring layout only — proportion_blockout_recipe_not_mesh_or_print_success (N6).
        Topology: torso trap|ovals, glute oval|two_spheres, optional template-applied.
        Optional profiles + skeleton (0027). Opt-in face/hair/neckline (0028).
        Opt-in hands/feet (0029). Raises ProportionError on failure.
        """
        return T.mesh_proportion_blockout_recipe(
            wr,
            report=report,
            out=out,
            format=format,
            depth_at_landmarks=depth_at_landmarks,
            limbs=limbs,
            force=force,
            torso=torso,
            glute=glute,
            nofuse=nofuse,
            breast_tilt_deg=breast_tilt_deg,
            template_applied=template_applied,
            profiles=profiles,
            skeleton=skeleton,
            face=face,
            hair=hair,
            neckline=neckline,
            hands=hands,
            feet=feet,
            fingers=fingers,
            toes=toes,
        )

    @mcp.tool()
    def mesh_proportion_blockout_validate_constraints(
        recipe: str,
        out: str,
        report: str | None = None,
        template_applied: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Validate named-role hard constraints on a blockout recipe.

        Authoring QA only — proportion_blockout_constraints_not_mesh_or_print_success (N6).
        Raises ProportionError on failure (never ok:false success). Report ok=false is data.
        """
        return T.mesh_proportion_blockout_validate_constraints(
            wr,
            recipe=recipe,
            out=out,
            report=report,
            template_applied=template_applied,
            force=force,
        )

    @mcp.tool()
    def mesh_proportion_blockout_optimize(
        recipe: str,
        out: str,
        mode: str = "fast",
        freeze_feet: bool = True,
        mesh: str | None = None,
        report: str | None = None,
        template_applied: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Constrained free-DOF blockout adjust (freeze-feet default true).

        Authoring QA only — proportion_blockout_optimize_not_mesh_or_print_success (N6).
        Free-name random optimizers are NOT product. Slow mode requires mesh.
        Raises ProportionError on failure (never ok:false success).
        """
        return T.mesh_proportion_blockout_optimize(
            wr,
            recipe=recipe,
            out=out,
            mode=mode,
            freeze_feet=freeze_feet,
            mesh=mesh,
            report=report,
            template_applied=template_applied,
            force=force,
        )

    @mcp.tool()
    def mesh_proportion_skeleton_build(
        report: str,
        out: str,
        format: str = "json",
        template_applied: str | None = None,
        depth_at_landmarks: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Emit joint/bone blockout skeleton graph from a proportion report.

        Authoring scaffold only — proportion_blockout_skeleton_not_mesh_or_print_success (N6).
        Not an animation rig. Writes blockout_skeleton.json (+ optional setup_skeleton.py)
        under out directory. Optional depth_at_landmarks file for depth Y ladder (0035).
        Raises ProportionError on failure (never ok:false success).
        """
        return T.mesh_proportion_skeleton_build(
            wr,
            report=report,
            out=out,
            format=format,
            template_applied=template_applied,
            depth_at_landmarks=depth_at_landmarks,
            force=force,
        )

    @mcp.tool()
    def mesh_proportion_depth_heatmap(
        samples: str,
        out: str,
        deltas: str | None = None,
        underlay: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Glance depth/delta heatmap PNG + meta from depth_at_landmarks samples.

        Authoring visualization only — proportion_depth_heatmap_not_mesh_or_print_success (N6).
        Numbers in depth_at_landmarks remain source of truth. Raises ProportionError
        on failure (never ok:false success).
        """
        return T.mesh_proportion_depth_heatmap(
            wr,
            samples=samples,
            out=out,
            deltas=deltas,
            underlay=underlay,
            force=force,
        )

    @mcp.tool()
    def mesh_proportion_depth_hint(
        depth_map: str | None = None,
        left: str | None = None,
        out: str = "landmarks_assist.hint.json",
        assist: str | None = None,
        report: str | None = None,
        backend: str = "external",
        force: bool = False,
        force_hint: bool = False,
        merge_into: str | None = None,
    ) -> dict[str, Any]:
        """Suggest left depth-pair assist points from an external depth channel.

        Side .hint.json is not analyze law; use merge_into for canonical assist.
        Monocular backend refuses (no torch/onnx pin). Authoring only —
        proportion_depth_hint_not_mesh_or_print_success (N6). Raises ProportionError
        on failure (never ok:false success).
        """
        return T.mesh_proportion_depth_hint(
            wr,
            depth_map=depth_map,
            left=left,
            out=out,
            assist=assist,
            report=report,
            backend=backend,
            force=force,
            force_hint=force_hint,
            merge_into=merge_into,
        )

    @mcp.tool()
    def mesh_proportion_silhouette_compare(
        ref: str,
        out: str,
        mesh: str | None = None,
        mesh_view: str | None = None,
        view_role: str = "front",
        overlay: bool = True,
        force: bool = False,
        require_trusted: bool = False,
    ) -> dict[str, Any]:
        """Front-only binary silhouette IoU/Dice (Package A front vs mesh front).

        Authoring QA only — proportion_silhouette_compare_not_mesh_or_print_success (N6).
        Not print success. Front-vs-front only. Raises ProportionError on failure
        (never ok:false success). Payload includes silhouette_trusted / trust_reasons
        (schema 1.1.0). require_trusted → silhouette_untrusted when untrusted.
        """
        return T.mesh_proportion_silhouette_compare(
            wr,
            ref=ref,
            out=out,
            mesh=mesh,
            mesh_view=mesh_view,
            view_role=view_role,
            overlay=overlay,
            force=force,
            require_trusted=require_trusted,
        )

    return mcp
