"""Character-region report on 0138 crops (0140).

Ranks five existing crop regions. Does not sculpt, weld, or launch a DCC.
Authoring report only - not mesh or print success.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from meshops.proportion.benchmark_multiview import (
    CROP_NAMES,
    ROLES,
    MultiviewBenchmark,
    load_benchmark,
    score_aligned_masks,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import DETAIL_HONESTY
from meshops.proportion.surface_plan import SURFACE_SCHEMA_VERSION, SurfacePlan

DETAIL_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
JSON_BASENAME = "character_detail.json"
MARKDOWN_BASENAME = "character_detail.md"


class DetailRoleScore(BaseModel):
    """One role cell. A missing crop stays null and notes ``crop missing``."""

    model_config = ConfigDict(extra="forbid")

    role: str
    iou: float | None = None
    dice: float | None = None
    delta_iou: float | None = None
    note: str | None = None


class DetailRegion(BaseModel):
    """One crop region. ``mean_iou`` is authoring order only."""

    model_config = ConfigDict(extra="forbid")

    name: str
    mean_iou: float | None
    view_conflict: bool
    roles: list[DetailRoleScore]


class CharacterDetail(BaseModel):
    """Write-only character_detail.json. Schema literal 1.0.0."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = DETAIL_SCHEMA_VERSION
    honesty: str = DETAIL_HONESTY
    ok: bool
    status: Literal[
        "verdict_pending",
        "rejected",
        "accepted",
        "baseline_only",
        "view_conflict",
    ]
    visual_verdict: Literal["accept", "reject"] | None = None
    needs_user_input: bool = False
    rank: None = None
    compare_status: Literal["baseline_only", "compared"]
    geometry_baseline: Literal["recipe_primitives", "one_junction"]
    surface_apply_status: Literal["plan_only", "applied", "guard_failed"] | None = None
    benchmark_status: str
    figure: str | None = None
    regions: list[DetailRegion]
    view_conflict: bool
    appearance_preview: list[str] = Field(default_factory=list)
    appearance_preview_after: list[str] = Field(default_factory=list)
    appearance_in_iou: Literal[False] = False
    remaining_gap: str
    messages: list[str]
    paths: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _fail(message: str, **details: object) -> None:
    raise ProportionError(message, code="detail_failed", details=details or None)


def _load_checked(path: Path) -> MultiviewBenchmark:
    if not path.is_file():
        _fail(f"benchmark not found: {path}", path=str(path))
    try:
        document = load_benchmark(path)
    except ProportionError as exc:
        raise ProportionError(
            f"benchmark failed: {exc}",
            code="detail_failed",
            details={"path": str(path)},
        ) from exc
    for role, image in document.reference.images.items():
        file = Path(image.path)
        if not file.is_absolute():
            file = path.parent / file
        if not file.is_file() or _sha256(file) != image.sha256:
            _fail(
                f"benchmark reference sha mismatch: {role}",
                role=role,
                path=str(file),
            )
    return document


def _load_surface(path: Path) -> SurfacePlan:
    if not path.is_file():
        _fail(f"surface plan not found: {path}", path=str(path))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"surface plan is not valid JSON: {path}",
            code="detail_failed",
            details={"path": str(path)},
        ) from exc
    version = raw.get("schema_version") if isinstance(raw, dict) else None
    if version != SURFACE_SCHEMA_VERSION:
        _fail(
            f"surface plan schema must be {SURFACE_SCHEMA_VERSION}, got {version!r}",
            path=str(path),
        )
    try:
        return SurfacePlan.model_validate(raw)
    except ValidationError as exc:
        raise ProportionError(
            f"invalid surface plan: {exc}",
            code="detail_failed",
            details={"path": str(path)},
        ) from exc


def _resolve(base: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path


def _crop_index(document: MultiviewBenchmark, base: Path) -> dict[tuple[str, str, str], Path]:
    found: dict[tuple[str, str, str], Path] = {}
    for crop in document.crops:
        resolved = _resolve(base, crop.png)
        if resolved is not None:
            found[(crop.role, crop.name, crop.subject)] = resolved
    return found


def _score_cell(
    index: dict[tuple[str, str, str], Path],
    role: str,
    name: str,
) -> tuple[float | None, float | None, str | None]:
    ref = index.get((role, name, "reference"))
    mesh = index.get((role, name, "meshops"))
    if ref is None or mesh is None or not ref.is_file() or not mesh.is_file():
        return None, None, "crop missing"
    scored = score_aligned_masks(ref, mesh)
    return float(scored["iou"]), float(scored["dice"]), None


def _copy_source(index: dict[tuple[str, str, str], Path], role: str, name: str) -> Path | None:
    for subject in ("meshops", "reference"):
        path = index.get((role, name, subject))
        if path is not None and path.is_file():
            return path
    return None


def _mean(scores: list[DetailRoleScore]) -> float | None:
    values = [row.iou for row in scores if row.iou is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _region_conflict(scores: list[DetailRoleScore]) -> bool:
    positive = False
    negative = False
    for row in scores:
        delta = row.delta_iou
        if delta is None or delta == 0:
            continue
        if delta > 0:
            positive = True
        else:
            negative = True
    return positive and negative


def _geometry(
    surface: Path | None,
) -> tuple[
    Literal["recipe_primitives", "one_junction"],
    Literal["plan_only", "applied", "guard_failed"] | None,
    str,
]:
    if surface is None:
        return "recipe_primitives", None, "0139 plan_only is not a body mesh"
    plan = _load_surface(surface)
    if plan.apply_status == "applied":
        return "one_junction", plan.apply_status, "0139 weld is one junction, not a character mesh"
    if plan.apply_status == "guard_failed":
        return "recipe_primitives", plan.apply_status, "0139 guard_failed left no weld"
    return "recipe_primitives", plan.apply_status, "0139 plan_only is not a body mesh"


def _verdict_status(
    verdict: str | None,
    *,
    compare_status: str,
    view_conflict: bool,
    needs_user_input: bool,
) -> tuple[
    bool,
    Literal["verdict_pending", "rejected", "accepted", "baseline_only", "view_conflict"],
]:
    if verdict is None:
        return False, "verdict_pending"
    if verdict == "reject":
        return False, "rejected"
    if compare_status == "baseline_only":
        return False, "baseline_only"
    if view_conflict:
        return False, "view_conflict"
    if needs_user_input:
        return False, "verdict_pending"
    return True, "accepted"


def _remaining_gap(regions: list[DetailRegion], compare_status: str) -> str:
    parts = ["appearance previews are not geometry"]
    names = [row.name for row in regions if row.view_conflict]
    if names:
        parts.append("view_conflict: " + ", ".join(names))
    if compare_status == "baseline_only":
        parts.append("after views absent; geometric gap not measured")
    return " ".join(parts)


def _write_markdown(detail: CharacterDetail, path: Path) -> None:
    lines = [
        "# character detail",
        "",
        f"honesty: {detail.honesty}",
        "",
        "Acceptance is not a sculpted mesh and not print success.",
        "",
        "image_left_hand is image-left, not anatomical left.",
        "",
        f"status: {detail.status}",
        f"ok: {str(detail.ok).lower()}",
        f"compare_status: {detail.compare_status}",
        f"geometry_baseline: {detail.geometry_baseline}",
        "rank: null",
        "appearance_in_iou: false",
        "",
        f"remaining_gap: {detail.remaining_gap}",
        "",
        "## regions",
        "",
    ]
    for region in detail.regions:
        mean = "null" if region.mean_iou is None else f"{region.mean_iou:.4f}"
        lines.append(
            f"- {region.name} mean_iou={mean} view_conflict={str(region.view_conflict).lower()}"
        )
    lines.extend(["", "## notes", ""])
    lines.extend(f"- {message}" for message in detail.messages)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _copy_crops(
    out: Path,
    before: dict[tuple[str, str, str], Path],
    after: dict[tuple[str, str, str], Path] | None,
) -> list[str]:
    copied: list[str] = []
    for role in ROLES:
        for name in CROP_NAMES:
            source = _copy_source(before, role, name)
            if source is not None:
                dest = out / "crops" / role / f"{name}_before.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                copied.append(str(dest))
            if after is None:
                continue
            after_source = _copy_source(after, role, name)
            if after_source is None:
                continue
            dest = out / "crops" / role / f"{name}_after.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(after_source, dest)
            copied.append(str(dest))
    return copied


def run_character_detail(
    benchmark: Path,
    out: Path,
    *,
    surface: Path | None = None,
    after: Path | None = None,
    figure: str | None = None,
    verdict: str | None = None,
) -> dict[str, Any]:
    """Write character_detail.json and .md. A score never sets ok."""
    benchmark = Path(benchmark)
    out = Path(out)
    if verdict not in (None, "accept", "reject"):
        _fail(f"verdict must be accept or reject, got {verdict!r}")
    document = _load_checked(benchmark)
    after_path = Path(after) if after is not None else None
    after_doc = _load_checked(after_path) if after_path is not None else None
    figures = list(document.reference.in_scope_figures)
    after_figures = list(after_doc.reference.in_scope_figures) if after_doc is not None else []
    if figure is not None and figure not in figures:
        _fail(f"figure {figure!r} is outside the benchmark", figure=figure)
    if after_doc is not None and figure is not None and figure not in after_figures:
        _fail(f"figure {figure!r} is not in both benchmarks", figure=figure)
    needs_user_input = figure is None and (len(figures) >= 2 or len(after_figures) >= 2)
    geometry, surface_status, geometry_message = _geometry(
        Path(surface) if surface is not None else None
    )
    before_index = _crop_index(document, benchmark.parent)
    after_index = (
        _crop_index(after_doc, after_path.parent)
        if after_doc is not None and after_path is not None
        else None
    )
    regions: list[DetailRegion] = []
    missing = False
    for name in CROP_NAMES:
        roles: list[DetailRoleScore] = []
        for role in ROLES:
            iou, dice, note = _score_cell(before_index, role, name)
            delta: float | None = None
            if after_index is not None:
                after_iou, _after_dice, after_note = _score_cell(after_index, role, name)
                if note is None and after_note is not None:
                    note = after_note
                if iou is not None and after_iou is not None:
                    delta = after_iou - iou
            if note == "crop missing":
                missing = True
            roles.append(DetailRoleScore(role=role, iou=iou, dice=dice, delta_iou=delta, note=note))
        regions.append(
            DetailRegion(
                name=name,
                mean_iou=_mean(roles),
                view_conflict=_region_conflict(roles),
                roles=roles,
            )
        )
    order = {name: index for index, name in enumerate(CROP_NAMES)}
    regions.sort(key=lambda row: (row.mean_iou is None, row.mean_iou or 0.0, order[row.name]))
    view_conflict = any(row.view_conflict for row in regions)
    compare_status: Literal["baseline_only", "compared"] = (
        "compared" if after_doc is not None else "baseline_only"
    )
    ok, status = _verdict_status(
        verdict,
        compare_status=compare_status,
        view_conflict=view_conflict,
        needs_user_input=needs_user_input,
    )
    messages = ["benchmark ok is not detail success", geometry_message]
    if missing:
        messages.append("crop missing")
    out.mkdir(parents=True, exist_ok=True)
    copied = _copy_crops(out, before_index, after_index)
    json_path = out / JSON_BASENAME
    md_path = out / MARKDOWN_BASENAME
    detail = CharacterDetail(
        ok=ok,
        status=status,
        visual_verdict=verdict if verdict in ("accept", "reject") else None,
        needs_user_input=needs_user_input,
        compare_status=compare_status,
        geometry_baseline=geometry,
        surface_apply_status=surface_status,
        benchmark_status=document.status,
        figure=figure,
        regions=regions,
        view_conflict=view_conflict,
        appearance_preview=list(document.appearance_preview),
        appearance_preview_after=(
            list(after_doc.appearance_preview) if after_doc is not None else []
        ),
        remaining_gap=_remaining_gap(regions, compare_status),
        messages=messages,
        paths=[str(json_path), str(md_path), *copied],
    )
    json_path.write_text(detail.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    _write_markdown(detail, md_path)
    return detail.model_dump(mode="json")
