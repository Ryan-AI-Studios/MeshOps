"""Multi-view package scaffold (track 0014) — layout + checklist + SOURCE.txt.

Creates authoring directories only. Not mesh reconstruction or print success
(Difficulty §12 / N6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from pydantic import ValidationError

from meshops.proportion.checklist import (
    PACKAGE_CHECKLIST_FILENAME,
    PackageChecklist,
    PackageMode,
    SourceKind,
    WardrobeTier,
    parse_figures,
    write_package_checklist,
)
from meshops.proportion.errors import ProportionError
from meshops.proportion.models import REQUIRED_VIEW_KEYS, PoseKind
from meshops.proportion.template import blank_assist_document

# Frozen 1x1 RGBA PNG (R4) — precomputed valid bytes (spec freeze had CRC typo
# `\x1f\x15c4` vs `\x1f\x15\xc4\x89`); do not invent packing at runtime.
PNG_1X1_BYTES: Final[bytes] = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0bIDATx\x9cc`\x00\x02\x00\x00\x05\x00\x01z^\xab?"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

ASSIST_FILENAME = "landmarks_assist.json"
SOURCE_FILENAME = "SOURCE.txt"
PROPORTION_SUBDIR = "proportion"
CHARACTER_SUBDIR = "character"


@dataclass
class ScaffoldResult:
    """Result of scaffold_package (layout only — not mesh/print success)."""

    mode: Literal["single", "dual"]
    paths: list[Path] = field(default_factory=list)
    analyze_hint: Path | None = None


def _fmt_or_unset(value: object | None) -> str:
    if value is None or value == "":
        return "(unset)"
    return str(value)


def write_source_txt(
    path: Path,
    *,
    subject: str | None,
    package_mode: PackageMode,
    pose: str,
    height_m: float | None,
    source_kind: str | None,
    analyze_hint: Path | None,
) -> Path:
    """Write plain UTF-8 SOURCE.txt (no parse model — R14)."""
    lines = [
        "# MeshOps multi-view package",
        f"subject: {_fmt_or_unset(subject)}",
        f"package_mode: {package_mode}",
        f"pose: {pose}",
        f"height_m: {_fmt_or_unset(height_m)}",
        f"source_kind: {_fmt_or_unset(source_kind) if source_kind else 'unknown'}",
    ]
    if package_mode == "dual" and analyze_hint is not None:
        lines.extend(
            [
                "# Dual: Package A = proportion; Package B = character",
                f"# Analyze master (dual): {analyze_hint}",
                f"# analyze_hint: meshops proportion analyze --views-dir {analyze_hint}",
            ]
        )
    lines.append("# Honesty: layout only — not mesh or print success")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_stub_images(
    directory: Path,
    *,
    include_back: bool,
    include_top: bool = False,
    force: bool,
) -> list[Path]:
    """Write frozen 1x1 PNG stubs only if missing (unless force)."""
    keys = list(REQUIRED_VIEW_KEYS)
    if include_back:
        keys.append("back")
    if include_top:
        keys.append("top")
    written: list[Path] = []
    for key in keys:
        dest = directory / f"{key}.png"
        if dest.is_file() and not force:
            continue
        dest.write_bytes(PNG_1X1_BYTES)
        written.append(dest)
    return written


def _write_template_with_pose(directory: Path, pose: str) -> Path:
    """blank_assist_document post-process pose — do not change template.py (R6)."""
    doc = blank_assist_document()
    doc["pose"] = pose
    path = directory / ASSIST_FILENAME
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _shared_checklist_kwargs(
    *,
    subject: str | None,
    height_m: float | None,
    pose: PoseKind | str,
    heroic_vs_realistic: str,
    in_scope_figures: list[str],
    multi_figure: bool,
    source_kind: SourceKind | None,
    package_mode: PackageMode,
) -> dict[str, object]:
    return {
        "subject": subject,
        "height_m": height_m,
        "pose": pose,
        "heroic_vs_realistic": heroic_vs_realistic,
        "in_scope_figures": list(in_scope_figures),
        "multi_figure": multi_figure or len(in_scope_figures) >= 2,
        "package_mode": package_mode,
        "source_kind": source_kind,
        "view_keys_required": list(REQUIRED_VIEW_KEYS),
        "view_keys_optional": ["back", "top"],
    }


def scaffold_package(
    out: Path | str,
    *,
    dual: bool = False,
    mode: PackageMode | None = None,
    height_m: float | None = None,
    subject: str | None = None,
    pose: PoseKind | str = "a_pose",
    heroic_vs_realistic: str = "unknown",
    figures: list[str] | str | None = None,
    source_kind: SourceKind | str | None = "unknown",
    wardrobe_tier: WardrobeTier | str | None = None,
    with_template: bool = False,
    stub_images: bool = False,
    include_back_stub: bool = False,
    include_top_stub: bool = False,
    force: bool = False,
) -> ScaffoldResult:
    """Create multi-view package layout + package_checklist.json (+ optional stubs).

    Honesty: layout only — not mesh or print success.
    """
    root = Path(out)
    # Resolve mode (R3): dual flag ⇒ dual; explicit mode dual OK; conflict checked by CLI
    if dual:
        package_mode: PackageMode = "dual"
    elif mode is not None:
        package_mode = mode
    else:
        package_mode = "single"

    if isinstance(figures, str):
        in_scope = parse_figures(figures)
    elif figures is None:
        in_scope = []
    else:
        in_scope = [f.strip() for f in figures if f and str(f).strip()]

    multi = len(in_scope) >= 2
    sk: SourceKind | None = "unknown" if source_kind is None else source_kind  # type: ignore[assignment]

    paths: list[Path] = []
    analyze_hint: Path | None = None

    try:
        root.mkdir(parents=True, exist_ok=True)

        if package_mode == "single":
            checklist_path = root / PACKAGE_CHECKLIST_FILENAME
            if checklist_path.is_file() and not force:
                raise ProportionError(
                    f"package checklist already exists: {checklist_path} (use --force)",
                    code="checklist_exists",
                    details={"path": str(checklist_path)},
                )
            checklist = PackageChecklist(
                **_shared_checklist_kwargs(  # type: ignore[arg-type]
                    subject=subject,
                    height_m=height_m,
                    pose=pose,
                    heroic_vs_realistic=heroic_vs_realistic,
                    in_scope_figures=in_scope,
                    multi_figure=multi,
                    source_kind=sk,
                    package_mode="single",
                ),
                package_role="combined",
                wardrobe_tier=wardrobe_tier,  # type: ignore[arg-type]
            )
            write_package_checklist(checklist_path, checklist)
            paths.append(checklist_path)

            source_path = write_source_txt(
                root / SOURCE_FILENAME,
                subject=subject,
                package_mode="single",
                pose=str(pose),
                height_m=height_m,
                source_kind=sk,
                analyze_hint=None,
            )
            paths.append(source_path)

            if with_template:
                tpath = _write_template_with_pose(root, str(pose))
                paths.append(tpath)

            if stub_images:
                paths.extend(
                    _write_stub_images(
                        root,
                        include_back=include_back_stub,
                        include_top=include_top_stub,
                        force=force,
                    )
                )

            return ScaffoldResult(mode="single", paths=paths, analyze_hint=None)

        # --- dual ---
        prop_dir = root / PROPORTION_SUBDIR
        char_dir = root / CHARACTER_SUBDIR
        root_checklist_path = root / PACKAGE_CHECKLIST_FILENAME
        prop_path = prop_dir / PACKAGE_CHECKLIST_FILENAME
        char_path = char_dir / PACKAGE_CHECKLIST_FILENAME

        # Preflight all checklist targets before any write (no partial dual tree).
        if not force:
            for existing in (root_checklist_path, prop_path, char_path):
                if existing.is_file():
                    raise ProportionError(
                        f"package checklist already exists: {existing} (use --force)",
                        code="checklist_exists",
                        details={"path": str(existing)},
                    )

        prop_dir.mkdir(parents=True, exist_ok=True)
        char_dir.mkdir(parents=True, exist_ok=True)
        analyze_hint = prop_dir.resolve()

        shared = _shared_checklist_kwargs(
            subject=subject,
            height_m=height_m,
            pose=pose,
            heroic_vs_realistic=heroic_vs_realistic,
            in_scope_figures=in_scope,
            multi_figure=multi,
            source_kind=sk,
            package_mode="dual",
        )

        root_checklist = PackageChecklist(
            **shared,  # type: ignore[arg-type]
            package_role=None,
            wardrobe_tier=None,
            proportion_subdir=PROPORTION_SUBDIR,
            character_subdir=CHARACTER_SUBDIR,
            notes=None,
        )
        write_package_checklist(root_checklist_path, root_checklist)
        paths.append(root_checklist_path)

        source_path = write_source_txt(
            root / SOURCE_FILENAME,
            subject=subject,
            package_mode="dual",
            pose=str(pose),
            height_m=height_m,
            source_kind=sk,
            analyze_hint=analyze_hint,
        )
        paths.append(source_path)

        # Leaf snapshots of shared fields (R13) — no auto-sync later
        prop_checklist = PackageChecklist(
            **shared,  # type: ignore[arg-type]
            package_role="proportion",
            wardrobe_tier="two_piece_midriff",
            notes="Package A master; snapshot of root shared fields at scaffold",
        )
        write_package_checklist(prop_path, prop_checklist)
        paths.append(prop_path)

        char_checklist = PackageChecklist(
            **shared,  # type: ignore[arg-type]
            package_role="character",
            wardrobe_tier="costume",
            notes="Package B character; snapshot of root shared fields at scaffold",
        )
        write_package_checklist(char_path, char_checklist)
        paths.append(char_path)

        if with_template:
            # Template on Package A (proportion master)
            tpath = _write_template_with_pose(prop_dir, str(pose))
            paths.append(tpath)

        if stub_images:
            paths.extend(
                _write_stub_images(
                    prop_dir,
                    include_back=include_back_stub,
                    include_top=include_top_stub,
                    force=force,
                )
            )
            paths.extend(
                _write_stub_images(
                    char_dir,
                    include_back=include_back_stub,
                    include_top=include_top_stub,
                    force=force,
                )
            )

        return ScaffoldResult(mode="dual", paths=paths, analyze_hint=analyze_hint)

    except ProportionError:
        raise
    except ValidationError as exc:
        raise ProportionError(
            f"invalid checklist fields for scaffold: {exc}",
            code="invalid_checklist",
            details={"path": str(root)},
        ) from exc
    except OSError as exc:
        raise ProportionError(
            f"scaffold failed under {root}: {exc}",
            code="scaffold_failed",
            details={"path": str(root)},
        ) from exc
