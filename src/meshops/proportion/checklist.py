"""Operator package checklist (track 0014) — package_checklist.json schema 1.0.0.

Independent of ProportionReport.schema_version. Authoring metadata only —
not mesh reconstruction or print success (Difficulty §12 / N6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from meshops.proportion.errors import ProportionError
from meshops.proportion.models import CANONICAL_VIEW_KEYS, REQUIRED_VIEW_KEYS, PoseKind

PACKAGE_CHECKLIST_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
PACKAGE_CHECKLIST_FILENAME = "package_checklist.json"

HeroicVsRealistic = Literal["heroic", "realistic", "stylized", "unknown"]
PackageMode = Literal["single", "dual"]
PackageRole = Literal["combined", "proportion", "character"]
WardrobeTier = Literal[
    "two_piece_midriff",
    "unitard",
    "tank_leggings",
    "costume",
    "unknown",
]
SourceKind = Literal[
    "imagen",
    "chatgpt",
    "photo",
    "f3d",
    "blender",
    "other",
    "unknown",
]

_DUAL_LEAF_NAMES = frozenset({"proportion", "character"})


class PackageChecklist(BaseModel):
    """Operator intent for a multi-view package (schema 1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = PACKAGE_CHECKLIST_SCHEMA_VERSION
    subject: str | None = None
    height_m: float | None = None
    pose: PoseKind | str = "a_pose"
    heroic_vs_realistic: HeroicVsRealistic = "unknown"
    in_scope_figures: list[str] = Field(default_factory=list)
    multi_figure: bool = False
    package_mode: PackageMode = "single"
    package_role: PackageRole | None = "combined"
    wardrobe_tier: WardrobeTier | None = None
    source_kind: SourceKind | None = "unknown"
    proportion_subdir: str = "proportion"
    character_subdir: str = "character"
    view_keys_required: list[str] = Field(default_factory=lambda: list(REQUIRED_VIEW_KEYS))
    view_keys_optional: list[str] = Field(default_factory=lambda: ["back"])
    notes: str | None = None

    @field_validator("height_m")
    @classmethod
    def _height_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("height_m must be > 0 when set")
        return v

    @model_validator(mode="after")
    def _normalize_multi_figure(self) -> PackageChecklist:
        if len(self.in_scope_figures) >= 2:
            self.multi_figure = True
        return self


def normalize_checklist(checklist: PackageChecklist) -> PackageChecklist:
    """Return checklist with multi_figure normalized (len(figures) ≥ 2 → True)."""
    data = checklist.model_dump()
    if len(data.get("in_scope_figures") or []) >= 2:
        data["multi_figure"] = True
    data["schema_version"] = PACKAGE_CHECKLIST_SCHEMA_VERSION
    return PackageChecklist.model_validate(data)


def load_package_checklist(path: Path | str) -> PackageChecklist:
    """Load and validate package_checklist.json."""
    p = Path(path)
    try:
        raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProportionError(
            f"cannot load package checklist: {p}: {exc}",
            code="invalid_checklist",
            details={"path": str(p)},
        ) from exc
    try:
        return normalize_checklist(PackageChecklist.model_validate(raw))
    except Exception as exc:
        raise ProportionError(
            f"invalid package checklist: {exc}",
            code="invalid_checklist",
            details={"path": str(p)},
        ) from exc


def write_package_checklist(path: Path | str, checklist: PackageChecklist) -> Path:
    """Write normalized checklist always as schema 1.0.0. Returns path written."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = normalize_checklist(checklist)
        # Always write schema 1.0.0
        payload = doc.model_dump(mode="json")
        payload["schema_version"] = PACKAGE_CHECKLIST_SCHEMA_VERSION
        p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except ProportionError:
        raise
    except OSError as exc:
        raise ProportionError(
            f"failed to write package checklist {p}: {exc}",
            code="scaffold_failed",
            details={"path": str(p)},
        ) from exc
    except Exception as exc:
        raise ProportionError(
            f"invalid package checklist on write: {exc}",
            code="invalid_checklist",
            details={"path": str(p)},
        ) from exc
    return p


def find_package_checklist(views_dir: Path | str) -> Path | None:
    """Resolve package_checklist.json for a views directory.

    1. Leaf file at views_dir if present.
    2. Else one parent only when views_dir.name ∈ {proportion, character}.
    Never walks more than one parent level (R2 / R16).
    """
    root = Path(views_dir).resolve()
    leaf = root / PACKAGE_CHECKLIST_FILENAME
    if leaf.is_file():
        return leaf
    if root.name in _DUAL_LEAF_NAMES:
        parent_file = root.parent / PACKAGE_CHECKLIST_FILENAME
        if parent_file.is_file():
            return parent_file
    return None


def resolve_checklist_pair(
    views_dir: Path | str,
) -> tuple[Path | None, Path | None]:
    """Return (leaf_path|None, parent_path|None) for R1 field fallback.

    Leaf = checklist inside views_dir when present.
    Parent = checklist one level up only when views_dir is proportion/ or character/.
    """
    root = Path(views_dir).resolve()
    leaf_path = root / PACKAGE_CHECKLIST_FILENAME
    leaf: Path | None = leaf_path if leaf_path.is_file() else None
    parent: Path | None = None
    if root.name in _DUAL_LEAF_NAMES:
        parent_path = root.parent / PACKAGE_CHECKLIST_FILENAME
        if parent_path.is_file():
            parent = parent_path
    return leaf, parent


def parse_figures(figures: str | None) -> list[str]:
    """Split comma-separated figures: strip whitespace, drop empties (R15)."""
    if not figures:
        return []
    return [part.strip() for part in figures.split(",") if part.strip()]


def validate_package_layout(views_dir: Path | str) -> list[str]:
    """Advisory layout notes using CANONICAL_VIEW_KEYS (R11). Not a hard gate."""
    root = Path(views_dir)
    notes: list[str] = []
    if not root.is_dir():
        return [f"views directory not found: {root}"]

    for key in REQUIRED_VIEW_KEYS:
        if not any((root / f"{key}{ext}").is_file() for ext in (".png", ".jpg", ".jpeg", ".webp")):
            notes.append(f"missing required view image: {key}")
    for key in CANONICAL_VIEW_KEYS:
        if key in REQUIRED_VIEW_KEYS:
            continue
        # optional keys — no hard note
    checklist = root / PACKAGE_CHECKLIST_FILENAME
    if not checklist.is_file():
        notes.append(f"missing {PACKAGE_CHECKLIST_FILENAME}")
    return notes


def field_from_pair(
    field: str,
    leaf: PackageChecklist | None,
    parent: PackageChecklist | None,
) -> Any:
    """R1 field pick: leaf non-null > parent non-null > None (CLI applied by caller)."""
    if leaf is not None:
        val = getattr(leaf, field, None)
        if val is not None:
            return val
    if parent is not None:
        val = getattr(parent, field, None)
        if val is not None:
            return val
    return None
