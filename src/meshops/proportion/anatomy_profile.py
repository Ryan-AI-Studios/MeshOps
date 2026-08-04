"""Torso/limb anatomy profile packs (track 0027).

List and load versioned authoring shape profiles for blockout-recipe --profiles.
Not mesh or print success (Difficulty §12 / N6 / ANATOMY_PROFILE_HONESTY).
"""

from __future__ import annotations

import json
from importlib.resources import files as resource_files
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from meshops.proportion.errors import ProportionError
from meshops.proportion.honesty import ANATOMY_PROFILE_HONESTY

PROFILE_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"

SexLiteral = Literal["female", "male", "neutral"]
ArchetypeLiteral = Literal["adult_athletic"]
SideLiteral = Literal["l", "r", "none", "both"]
TorsoModeHint = Literal["trap", "ovals"]

# RecipeRole-compatible set for profile parts (includes 0027 extensions).
ProfileRecipeRole = Literal[
    "torso",
    "pelvis",
    "neck",
    "head",
    "shoulder_bridge",
    "hip_bridge",
    "deltoid_soft",
    "breast_soft",
    "glute_soft",
    "iliac_soft",
    "limb_segment",
    "trap_soft",
    "pec_soft",
    "scap_soft",
    "bicep_soft",
    "clavicle",
]
ProfileRecipeKind = Literal["trap_box", "box", "cylinder", "ellipsoid", "capsule"]

_KNOWN_PROFILE_IDS: Final[tuple[str, ...]] = (
    "torso_limb_f_athletic_v1",
    "torso_limb_m_athletic_v1",
)


class ProfileScaleSpec(BaseModel):
    """Optional scale keys into report / template / frac*H (B8 precedence)."""

    model_config = ConfigDict(extra="forbid")

    rx_frac_h: float | None = None
    ry_frac_h: float | None = None
    rz_frac_h: float | None = None
    radius_frac_h: float | None = None
    use_breast_metrics: bool = False
    use_soft_spacing: bool = False
    use_diameter: str | None = None
    use_depth_band: str | None = None
    michelin_cap_frac_h: float | None = None


class ProfilePartSpec(BaseModel):
    """One primitive template inside a profile region."""

    model_config = ConfigDict(extra="forbid")

    role: ProfileRecipeRole
    kind: ProfileRecipeKind
    side: SideLiteral = "none"
    count: int = 1
    parent_joint_id: str | None = None
    parent_joint_fallback: list[str] = Field(default_factory=list)
    scale: ProfileScaleSpec = Field(default_factory=ProfileScaleSpec)
    placement_rules: list[str] = Field(default_factory=list)
    notes: str | None = None


class ProfileRegion(BaseModel):
    """Named anatomy region (traps, chest, glutes, …)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    enabled: bool = True
    parts: list[ProfilePartSpec] = Field(default_factory=list)
    notes: str | None = None
    # C6: torso region only — steers base torso mode; no second torso part.
    preferred_torso_mode: TorsoModeHint | None = None


class AnatomyProfileDocument(BaseModel):
    """On-disk anatomy profile document schema 1.0.0."""

    model_config = ConfigDict(extra="forbid")

    id: str
    schema_version: Literal["1.0.0"] = PROFILE_SCHEMA_VERSION
    honesty: str
    sex: SexLiteral
    archetype: ArchetypeLiteral = "adult_athletic"
    description: str
    template_id_hint: str | None = None
    regions: list[ProfileRegion] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


class ProfileListEntry(BaseModel):
    """One row for `proportion anatomy-profiles`."""

    model_config = ConfigDict(extra="forbid")

    id: str
    sex: SexLiteral
    archetype: ArchetypeLiteral
    description: str
    template_id_hint: str | None = None


# ---------------------------------------------------------------------------
# Load / list
# ---------------------------------------------------------------------------


def _profiles_root() -> Any:
    return resource_files("meshops.proportion.body_profiles")


def load_anatomy_profile(profile_id: str) -> AnatomyProfileDocument:
    """Load a profile document by id; unknown → profile_unknown."""
    pid = (profile_id or "").strip()
    if pid not in _KNOWN_PROFILE_IDS:
        raise ProportionError(
            f"unknown anatomy profile id: {profile_id!r} (known: {', '.join(_KNOWN_PROFILE_IDS)})",
            code="profile_unknown",
            details={"profile_id": profile_id, "known": list(_KNOWN_PROFILE_IDS)},
        )
    root = _profiles_root()
    try:
        data = json.loads((root / f"{pid}.json").read_text(encoding="utf-8"))
    except (OSError, FileNotFoundError, json.JSONDecodeError, TypeError) as exc:
        raise ProportionError(
            f"failed to load anatomy profile {pid!r}: {exc}",
            code="profile_unknown",
            details={"profile_id": pid},
        ) from exc
    try:
        doc = AnatomyProfileDocument.model_validate(data)
    except Exception as exc:
        raise ProportionError(
            f"invalid anatomy profile document {pid!r}: {exc}",
            code="profile_unknown",
            details={"profile_id": pid},
        ) from exc
    if doc.id != pid:
        raise ProportionError(
            f"profile id mismatch: file {pid!r} has id {doc.id!r}",
            code="profile_unknown",
            details={"profile_id": pid, "document_id": doc.id},
        )
    if doc.honesty != ANATOMY_PROFILE_HONESTY:
        raise ProportionError(
            f"profile honesty mismatch for {pid!r}",
            code="profile_unknown",
            details={"profile_id": pid, "honesty": doc.honesty},
        )
    return doc


def list_anatomy_profiles() -> list[dict[str, Any]]:
    """Return [{id, sex, archetype, description, template_id_hint}, …]."""
    out: list[dict[str, Any]] = []
    for pid in _KNOWN_PROFILE_IDS:
        doc = load_anatomy_profile(pid)
        out.append(
            ProfileListEntry(
                id=doc.id,
                sex=doc.sex,
                archetype=doc.archetype,
                description=doc.description,
                template_id_hint=doc.template_id_hint,
            ).model_dump(mode="json")
        )
    return out


def region_by_id(doc: AnatomyProfileDocument, region_id: str) -> ProfileRegion | None:
    """Return first region matching id, or None."""
    for reg in doc.regions:
        if reg.id == region_id:
            return reg
    return None


def region_enabled(doc: AnatomyProfileDocument, region_id: str) -> bool:
    """True when region exists and enabled."""
    reg = region_by_id(doc, region_id)
    return reg is not None and reg.enabled


__all__ = [
    "ANATOMY_PROFILE_HONESTY",
    "PROFILE_SCHEMA_VERSION",
    "AnatomyProfileDocument",
    "ProfileListEntry",
    "ProfilePartSpec",
    "ProfileRegion",
    "ProfileScaleSpec",
    "list_anatomy_profiles",
    "load_anatomy_profile",
    "region_by_id",
    "region_enabled",
]
