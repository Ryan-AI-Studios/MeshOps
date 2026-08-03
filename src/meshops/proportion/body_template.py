"""Sex/archetype body template pack (track 0022).

List and apply versioned authoring priors scaled by report stature.
Not mesh or print success (Difficulty §12 / N6 / TEMPLATE_HONESTY).
"""

from __future__ import annotations

import json
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.analyze import load_report
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import TEMPLATE_HONESTY
from meshops.proportion.models import DiameterMeasure, ProportionReport

TEMPLATE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
APPLIED_JSON_BASENAME: Final[str] = "template_applied.json"
CONSTANTS_PY_BASENAME: Final[str] = "template_constants.py"

BreastMode = Literal["dual_tilted", "pec_ovals", "none"]
GluteModeDefault = Literal["two_spheres", "oval", "mild_oval"]
TorsoModeDefault = Literal["ovals", "trap"]
SexLiteral = Literal["female", "male"]
ArchetypeLiteral = Literal["adult_athletic"]

_KNOWN_TEMPLATE_IDS: Final[tuple[str, ...]] = (
    "female_adult_athletic",
    "male_adult_athletic",
)

_MALE_ART_CANON_MSG: Final[str] = (
    "male_adult_athletic v1 is art-canon prior, not measured — retune from report"
)


# ---------------------------------------------------------------------------
# Document models (input templates — _frac XOR _m suffix law)
# ---------------------------------------------------------------------------


class BreastTemplate(BaseModel):
    """Breast / pec soft priors."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    tilt_x_deg: float = 0.0
    y_frac: float  # signed body-frame (front -Y)
    ry_scale: float = 1.0
    rz_scale: float = 1.0
    intermammary_gap_frac: float  # of bust_hw only (C6)
    slant_deg: float = 0.0


class GluteTemplate(BaseModel):
    """Glute soft priors."""

    model_config = ConfigDict(extra="forbid")

    r_frac: float
    y_frac: float | None = None  # signed + back
    y_m: float | None = None
    z_frac: float | None = None
    z_m: float | None = None
    cleft_frac: float = 0.1


class FootTemplate(BaseModel):
    """Foot / ankle priors."""

    model_config = ConfigDict(extra="forbid")

    foot_len_scale: float = 1.4
    side_ankle_y_m: float = 0.015
    heel_y_m: float = 0.03
    ank_foot_r_frac: float


class SoftSpacingDefaults(BaseModel):
    """0030 hook keys — defaults only; measure path stays 0030."""

    model_config = ConfigDict(extra="forbid")

    intermammary_gap_frac: float
    glute_cleft_frac: float


class NeckThicknessNotes(BaseModel):
    """Provenance for cumulative neck scale stages."""

    model_config = ConfigDict(extra="forbid")

    stages: list[float] = Field(default_factory=list)


class BodyTemplateDocument(BaseModel):
    """On-disk template document schema 1.0.0."""

    model_config = ConfigDict(extra="forbid")

    id: str
    schema_version: Literal["1.0.0"] = TEMPLATE_SCHEMA_VERSION
    sex: SexLiteral
    archetype: ArchetypeLiteral = "adult_athletic"
    description: str
    honesty: str = TEMPLATE_HONESTY
    breast_mode: BreastMode
    glute_mode_default: GluteModeDefault
    torso_mode_default: TorsoModeDefault
    shoulder_widest: bool = False
    breast: BreastTemplate
    glute: GluteTemplate
    pelvis_y_frac: float | None = None
    pelvis_y_m: float | None = None
    torso_waist_taper: float = 0.0
    thigh_tilt_deg: float = 0.0
    neck_thickness_scale: float = 1.0
    neck_thickness_notes: NeckThicknessNotes = Field(default_factory=NeckThicknessNotes)
    head_depth_scale: float = 1.0
    head_radius_scale: float = 1.0
    foot: FootTemplate
    soft_spacing_defaults: SoftSpacingDefaults
    frame_notes: str = "Z up soles~0, +X camera-right, face -Y, toes -Y, heels +Y"
    messages: list[str] = Field(default_factory=list)


class TemplateListEntry(BaseModel):
    """One row for `proportion templates`."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    sex: SexLiteral
    archetype: ArchetypeLiteral


# ---------------------------------------------------------------------------
# Applied package
# ---------------------------------------------------------------------------


class AppliedConstants(BaseModel):
    """Resolved meters/deg constants after apply."""

    model_config = ConfigDict(extra="forbid")

    breast_mode: BreastMode
    glute_mode_default: GluteModeDefault
    torso_mode_default: TorsoModeDefault
    shoulder_widest: bool = False
    breast_enabled: bool = True
    breast_tilt_x_deg: float = 0.0
    breast_y_m: float | None = None
    breast_y_frac: float | None = None
    breast_ry_scale: float = 1.0
    breast_rz_scale: float = 1.0
    intermammary_gap_frac: float | None = None
    intermammary_gap_m: float | None = None
    breast_slant_deg: float = 0.0
    glute_r_m: float | None = None
    glute_r_frac: float | None = None
    glute_y_m: float | None = None
    glute_y_frac: float | None = None
    glute_z_m: float | None = None
    glute_z_frac: float | None = None
    glute_cleft_frac: float | None = None
    glute_cleft_m: float | None = None
    pelvis_y_m: float | None = None
    torso_waist_taper: float = 0.0
    thigh_tilt_deg: float = 0.0
    neck_thickness_scale: float = 1.0
    neck_thickness_stages: list[float] = Field(default_factory=list)
    head_depth_scale: float = 1.0
    head_radius_scale: float = 1.0
    foot_len_scale: float = 1.4
    side_ankle_y_m: float = 0.015
    heel_y_m: float = 0.03
    ank_foot_r_m: float | None = None
    ank_foot_r_frac: float | None = None
    soft_spacing_defaults: SoftSpacingDefaults | None = None
    frame_notes: str = ""


class TemplateAppliedPackage(BaseModel):
    """template_applied.json (schema 1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = TEMPLATE_SCHEMA_VERSION
    honesty: str = TEMPLATE_HONESTY
    template_id: str
    sex: SexLiteral
    archetype: ArchetypeLiteral
    source_report: str
    source_report_schema: str | None = None
    height_m: float
    head_unit_m: float | None = None
    constants: AppliedConstants
    scale_notes: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Load / registry
# ---------------------------------------------------------------------------


def _templates_root() -> Any:
    return resource_files("meshops.proportion.body_templates")


def load_body_template(template_id: str) -> BodyTemplateDocument:
    """Load a template document by id; unknown → template_unknown."""
    tid = (template_id or "").strip()
    if tid not in _KNOWN_TEMPLATE_IDS:
        raise ProportionError(
            f"unknown body template id: {template_id!r} (known: {', '.join(_KNOWN_TEMPLATE_IDS)})",
            code="template_unknown",
            details={"template_id": template_id, "known": list(_KNOWN_TEMPLATE_IDS)},
        )
    root = _templates_root()
    try:
        data = json.loads((root / f"{tid}.json").read_text(encoding="utf-8"))
    except (OSError, FileNotFoundError, json.JSONDecodeError, TypeError) as exc:
        raise ProportionError(
            f"failed to load body template {tid!r}: {exc}",
            code="template_failed",
            details={"template_id": tid},
        ) from exc
    try:
        doc = BodyTemplateDocument.model_validate(data)
    except Exception as exc:
        raise ProportionError(
            f"invalid body template document {tid!r}: {exc}",
            code="template_failed",
            details={"template_id": tid},
        ) from exc
    if doc.id != tid:
        raise ProportionError(
            f"template id mismatch: file {tid!r} has id {doc.id!r}",
            code="template_failed",
            details={"template_id": tid, "document_id": doc.id},
        )
    if doc.honesty != TEMPLATE_HONESTY:
        raise ProportionError(
            f"template honesty mismatch for {tid!r}",
            code="template_failed",
            details={"template_id": tid, "honesty": doc.honesty},
        )
    return doc


def list_body_templates() -> list[dict[str, Any]]:
    """Return [{id, description, sex, archetype}, …] for known templates."""
    out: list[dict[str, Any]] = []
    for tid in _KNOWN_TEMPLATE_IDS:
        doc = load_body_template(tid)
        out.append(
            TemplateListEntry(
                id=doc.id,
                description=doc.description,
                sex=doc.sex,
                archetype=doc.archetype,
            ).model_dump(mode="json")
        )
    return out


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _resolve_diameter(diameters: list[DiameterMeasure], band_id: str) -> DiameterMeasure | None:
    matches = [d for d in diameters if d.band_id == band_id]
    if not matches:
        return None
    front = [d for d in matches if d.view == "front"]
    return front[0] if front else matches[0]


def _half_width(d: DiameterMeasure) -> float | None:
    if d.half_width_m is not None:
        return float(d.half_width_m)
    if d.width_m is not None:
        return float(d.width_m) / 2.0
    return None


def _resolve_applied_constants(
    doc: BodyTemplateDocument,
    report: ProportionReport,
    height_m: float,
    *,
    scale_notes: list[str],
    messages: list[str],
) -> AppliedConstants:
    """Resolve fracs → meters; prefer measured diameters/depth when present."""
    h = float(height_m)
    breast = doc.breast
    glute = doc.glute
    foot = doc.foot

    breast_y_m = float(breast.y_frac) * h
    breast_y_frac = float(breast.y_frac)

    # Template prior for glute radius (frac * H); may be replaced by measured hip_hw.
    glute_r_m = float(glute.r_frac) * h
    glute_r_frac = float(glute.r_frac)
    if glute.y_m is not None:
        glute_y_m = float(glute.y_m)
        glute_y_frac = glute_y_m / h if h > 0 else None
    elif glute.y_frac is not None:
        glute_y_frac = float(glute.y_frac)
        glute_y_m = glute_y_frac * h
    else:
        glute_y_m = None
        glute_y_frac = None

    if glute.z_m is not None:
        glute_z_m = float(glute.z_m)
        glute_z_frac = glute_z_m / h if h > 0 else None
    elif glute.z_frac is not None:
        glute_z_frac = float(glute.z_frac)
        glute_z_m = glute_z_frac * h
    else:
        glute_z_m = None
        glute_z_frac = None

    # Prefer measured bust half-width for gap (C6)
    bust = _resolve_diameter(report.diameters, "bust")
    bust_hw = _half_width(bust) if bust is not None else None
    gap_frac = float(breast.intermammary_gap_frac)
    intermammary_gap_m: float | None = None
    if bust_hw is not None and bust_hw > 0:
        intermammary_gap_m = gap_frac * bust_hw
        scale_notes.append(
            f"intermammary_gap_m = gap_frac*{gap_frac:.4f} * bust_hw={bust_hw:.4f}m (C6)"
        )
    else:
        scale_notes.append("intermammary_gap: no bust_hw — gap_m unresolved; gap_frac prior kept")

    # Prefer measured hip half-width for glute radius / cleft (binding scale rule).
    # hip_hw from landmarks wins; else hip/waist diameter; else template r_frac * H.
    _GLUTE_R_FROM_HIP_FRAC = 0.55
    hip = _resolve_diameter(report.diameters, "hip") or _resolve_diameter(report.diameters, "waist")
    hip_hw_lm: float | None = None
    lms = report.landmarks_xyz
    left = lms.get("hip_l")
    right = lms.get("hip_r")
    if left is not None and right is not None and left.x_m is not None and right.x_m is not None:
        hip_hw_lm = (abs(float(left.x_m)) + abs(float(right.x_m))) / 2.0
    if hip_hw_lm is not None and hip_hw_lm > 0:
        glute_r_m = float(hip_hw_lm) * _GLUTE_R_FROM_HIP_FRAC
        glute_r_frac = glute_r_m / h if h > 0 else glute_r_frac
        scale_notes.append(
            f"glute_r_m={glute_r_m:.4f} from measured hip_hw={hip_hw_lm:.4f}m "
            f"* {_GLUTE_R_FROM_HIP_FRAC} (prefer measured over template r_frac)"
        )
    elif hip is not None:
        hw = _half_width(hip)
        if hw is not None and hw > 0:
            glute_r_m = float(hw) * _GLUTE_R_FROM_HIP_FRAC
            glute_r_frac = glute_r_m / h if h > 0 else glute_r_frac
            scale_notes.append(
                f"glute_r_m={glute_r_m:.4f} from diameter hip/waist_hw={hw:.4f}m "
                f"* {_GLUTE_R_FROM_HIP_FRAC} (prefer measured)"
            )
        else:
            scale_notes.append(
                f"glute_r_m={glute_r_m:.4f} from template r_frac*{h:.4f} (no usable hip measure)"
            )
    else:
        scale_notes.append(
            f"glute_r_m={glute_r_m:.4f} from template r_frac*{h:.4f} (no hip measure)"
        )

    # Depth bands: prefer measured depth note for recipe soft bulk
    for band in report.depth_bands:
        if band.band_id in ("glute", "hip") and band.depth_m is not None:
            scale_notes.append(
                f"depth_band {band.band_id} depth_m={float(band.depth_m):.4f} "
                "(prefer measured depth for soft bulk when building recipe)"
            )
            break

    cleft_frac = float(glute.cleft_frac)
    glute_cleft_m: float | None = None
    if hip_hw_lm is not None:
        glute_cleft_m = cleft_frac * hip_hw_lm

    if doc.pelvis_y_m is not None:
        pelvis_y_m = float(doc.pelvis_y_m)
    elif doc.pelvis_y_frac is not None:
        pelvis_y_m = float(doc.pelvis_y_frac) * h
    else:
        pelvis_y_m = None

    ank_r_frac = float(foot.ank_foot_r_frac)
    ank_foot_r_m = ank_r_frac * h

    # Soft spacing defaults (0030 hooks)
    soft_spacing = SoftSpacingDefaults(
        intermammary_gap_frac=float(doc.soft_spacing_defaults.intermammary_gap_frac),
        glute_cleft_frac=float(doc.soft_spacing_defaults.glute_cleft_frac),
    )

    if doc.id == "male_adult_athletic" and _MALE_ART_CANON_MSG not in messages:
        messages.append(_MALE_ART_CANON_MSG)

    for m in doc.messages:
        if m not in messages:
            messages.append(m)

    stages = list(doc.neck_thickness_notes.stages)

    return AppliedConstants(
        breast_mode=doc.breast_mode,
        glute_mode_default=doc.glute_mode_default,
        torso_mode_default=doc.torso_mode_default,
        shoulder_widest=bool(doc.shoulder_widest),
        breast_enabled=bool(breast.enabled),
        breast_tilt_x_deg=float(breast.tilt_x_deg),
        breast_y_m=breast_y_m,
        breast_y_frac=breast_y_frac,
        breast_ry_scale=float(breast.ry_scale),
        breast_rz_scale=float(breast.rz_scale),
        intermammary_gap_frac=gap_frac,
        intermammary_gap_m=intermammary_gap_m,
        breast_slant_deg=float(breast.slant_deg),
        glute_r_m=glute_r_m,
        glute_r_frac=glute_r_frac,
        glute_y_m=glute_y_m,
        glute_y_frac=glute_y_frac,
        glute_z_m=glute_z_m,
        glute_z_frac=glute_z_frac,
        glute_cleft_frac=cleft_frac,
        glute_cleft_m=glute_cleft_m,
        pelvis_y_m=pelvis_y_m,
        torso_waist_taper=float(doc.torso_waist_taper),
        thigh_tilt_deg=float(doc.thigh_tilt_deg),
        neck_thickness_scale=float(doc.neck_thickness_scale),
        neck_thickness_stages=stages,
        head_depth_scale=float(doc.head_depth_scale),
        head_radius_scale=float(doc.head_radius_scale),
        foot_len_scale=float(foot.foot_len_scale),
        side_ankle_y_m=float(foot.side_ankle_y_m),
        heel_y_m=float(foot.heel_y_m),
        ank_foot_r_m=ank_foot_r_m,
        ank_foot_r_frac=ank_r_frac,
        soft_spacing_defaults=soft_spacing,
        frame_notes=doc.frame_notes,
    )


def _emit_constants_py(package: TemplateAppliedPackage) -> str:
    """Pure assignments + N6 header for template_constants.py."""
    c = package.constants
    lines = [
        "# template_constants.py — MeshOps track 0022 body template apply",
        f"# honesty: {TEMPLATE_HONESTY}",
        "# N6 / Difficulty §12: body template priors are authoring layout only —",
        "# not mesh reconstruction, not print-ready, not hero sculpt success.",
        f"# template_id: {package.template_id}",
        f"# height_m: {package.height_m}",
        f"# source_report: {package.source_report}",
        "",
        f"TEMPLATE_ID = {package.template_id!r}",
        f"HONESTY = {TEMPLATE_HONESTY!r}",
        f"HEIGHT_M = {package.height_m!r}",
        f"HEAD_UNIT_M = {package.head_unit_m!r}",
        f"SEX = {package.sex!r}",
        f"ARCHETYPE = {package.archetype!r}",
        "",
        f"BREAST_MODE = {c.breast_mode!r}",
        f"GLUTE_MODE_DEFAULT = {c.glute_mode_default!r}",
        f"TORSO_MODE_DEFAULT = {c.torso_mode_default!r}",
        f"SHOULDER_WIDEST = {c.shoulder_widest!r}",
        f"BREAST_ENABLED = {c.breast_enabled!r}",
        f"BREAST_TILT_X_DEG = {c.breast_tilt_x_deg!r}",
        f"BREAST_Y_M = {c.breast_y_m!r}",
        f"BREAST_Y_FRAC = {c.breast_y_frac!r}",
        f"BREAST_RY_SCALE = {c.breast_ry_scale!r}",
        f"BREAST_RZ_SCALE = {c.breast_rz_scale!r}",
        f"INTERMAMMARY_GAP_FRAC = {c.intermammary_gap_frac!r}",
        f"INTERMAMMARY_GAP_M = {c.intermammary_gap_m!r}",
        f"BREAST_SLANT_DEG = {c.breast_slant_deg!r}",
        f"GLUTE_R_M = {c.glute_r_m!r}",
        f"GLUTE_R_FRAC = {c.glute_r_frac!r}",
        f"GLUTE_Y_M = {c.glute_y_m!r}",
        f"GLUTE_Y_FRAC = {c.glute_y_frac!r}",
        f"GLUTE_Z_M = {c.glute_z_m!r}",
        f"GLUTE_CLEFT_FRAC = {c.glute_cleft_frac!r}",
        f"GLUTE_CLEFT_M = {c.glute_cleft_m!r}",
        f"PELVIS_Y_M = {c.pelvis_y_m!r}",
        f"TORSO_WAIST_TAPER = {c.torso_waist_taper!r}",
        f"THIGH_TILT_DEG = {c.thigh_tilt_deg!r}",
        f"NECK_THICKNESS_SCALE = {c.neck_thickness_scale!r}",
        f"NECK_THICKNESS_STAGES = {c.neck_thickness_stages!r}",
        f"HEAD_DEPTH_SCALE = {c.head_depth_scale!r}",
        f"HEAD_RADIUS_SCALE = {c.head_radius_scale!r}",
        f"FOOT_LEN_SCALE = {c.foot_len_scale!r}",
        f"SIDE_ANKLE_Y_M = {c.side_ankle_y_m!r}",
        f"HEEL_Y_M = {c.heel_y_m!r}",
        f"ANK_FOOT_R_M = {c.ank_foot_r_m!r}",
        "",
    ]
    return "\n".join(lines)


def apply_body_template(
    report_path: Path | str,
    template_id: str,
    out: Path | str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Apply template to report → write template_applied.json + template_constants.py.

    height_m null or ≤0 → template_empty (never invent stature).
    """
    report = load_report(report_path)
    doc = load_body_template(template_id)

    h = report.height_m
    if h is None or float(h) <= 0.0:
        raise ProportionError(
            "cannot resolve template fracs: report height_m is null or ≤ 0 (never invent stature)",
            code="template_empty",
            details={"height_m": h, "template_id": template_id},
        )
    height_m = float(h)

    out_dir = Path(out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProportionError(
            f"cannot create template out directory: {out_dir}: {exc}",
            code="template_failed",
            details={"out": str(out_dir)},
        ) from exc

    json_path = out_dir / APPLIED_JSON_BASENAME
    py_path = out_dir / CONSTANTS_PY_BASENAME
    for p in (json_path, py_path):
        if p.exists() and not force:
            raise ProportionError(
                f"output already exists (use --force): {p}",
                code="write_failed",
                details={"path": str(p)},
            )

    scale_notes: list[str] = []
    messages: list[str] = []

    if report.quality.multi_figure:
        messages.append("quality.multi_figure: template still applied — confirm primary figure")
    if report.quality.needs_user_input:
        messages.append("quality.needs_user_input: template still applied — confirm primary figure")

    scale_notes.append(f"resolved _frac with height_m={height_m:.6f}")

    head_unit_m: float | None = None
    hu_frac = report.head_unit_frac
    if hu_frac is not None and float(hu_frac) > 0.0:
        head_unit_m = height_m * float(hu_frac)
        scale_notes.append(
            f"head_unit_m = height_m * head_unit_frac = {height_m:.6f} * {float(hu_frac):.6f}"
        )
    else:
        scale_notes.append("head_unit_m unresolved (no positive head_unit_frac)")

    constants = _resolve_applied_constants(
        doc, report, height_m, scale_notes=scale_notes, messages=messages
    )

    package = TemplateAppliedPackage(
        schema_version=TEMPLATE_SCHEMA_VERSION,
        honesty=TEMPLATE_HONESTY,
        template_id=doc.id,
        sex=doc.sex,
        archetype=doc.archetype,
        source_report=str(Path(report_path)),
        source_report_schema=report.schema_version,
        height_m=height_m,
        head_unit_m=head_unit_m,
        constants=constants,
        scale_notes=scale_notes,
        messages=messages,
    )

    try:
        json_path.write_text(
            json.dumps(package.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        py_path.write_text(_emit_constants_py(package), encoding="utf-8")
    except OSError as exc:
        raise ProportionError(
            f"failed to write template apply outputs: {exc}",
            code="write_failed",
            details={"out": str(out_dir)},
        ) from exc

    return {
        "ok": True,
        "template_id": doc.id,
        "paths": [str(json_path), str(py_path)],
        "height_m": height_m,
        "head_unit_m": head_unit_m,
        "honesty": TEMPLATE_HONESTY,
        "schema_version": TEMPLATE_SCHEMA_VERSION,
        "scale_notes": list(scale_notes),
        "messages": list(messages),
        "constants": constants.model_dump(mode="json"),
    }


def load_template_applied(path: Path | str) -> TemplateAppliedPackage:
    """Load template_applied.json; file or directory (D5)."""
    p = Path(path)
    if p.is_dir():
        p = p / APPLIED_JSON_BASENAME
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProportionError(
            f"template_applied not found: {p}",
            code="recipe_failed",
            details={"path": str(p)},
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot load template_applied: {p}: {exc}",
            code="recipe_failed",
            details={"path": str(p)},
        ) from exc
    try:
        package = TemplateAppliedPackage.model_validate(data)
    except Exception as exc:
        raise ProportionError(
            f"invalid template_applied: {p}: {exc}",
            code="recipe_failed",
            details={"path": str(p)},
        ) from exc
    if not package.template_id:
        raise ProportionError(
            "template_applied missing template_id",
            code="recipe_failed",
            details={"path": str(p)},
        )
    if package.template_id not in _KNOWN_TEMPLATE_IDS:
        raise ProportionError(
            f"template_applied has unknown template_id: {package.template_id!r}",
            code="recipe_failed",
            details={"template_id": package.template_id, "path": str(p)},
        )
    return package


__all__ = [
    "APPLIED_JSON_BASENAME",
    "CONSTANTS_PY_BASENAME",
    "TEMPLATE_HONESTY",
    "TEMPLATE_SCHEMA_VERSION",
    "AppliedConstants",
    "BodyTemplateDocument",
    "TemplateAppliedPackage",
    "apply_body_template",
    "list_body_templates",
    "load_body_template",
    "load_template_applied",
]
