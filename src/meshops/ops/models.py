"""Doctor report schema (schema_version 1.0.0)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

BlenderSource = Literal["env", "path", "well_known", "portable", "missing"]
OrcaSource = Literal["env", "path", "well_known", "missing"]
OrcaVersionSource = Literal["appdata", "path_only", "missing"]
ToolStatus = Literal["ok", "missing", "version_mismatch", "warn", "error", "skipped"]
VramStatus = Literal["ok", "no_nvidia_gpu", "probe_error", "ritual_only"]


class PythonInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    executable: str
    pin_ok: bool


class PackageStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    import_ok: bool
    version: str | None = None
    optional: bool = False
    error: str | None = None


class BlenderToolStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    version: str | None = None
    pin_ok: bool | None = None
    status: ToolStatus = "missing"
    source: BlenderSource = "missing"


class OrcaToolStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    version: str | None = None
    soft_pin_ok: bool | None = None
    status: ToolStatus = "missing"
    source: OrcaSource = "missing"
    version_source: OrcaVersionSource = "missing"


class F3dToolStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    import_ok: bool
    version: str | None = None


class ToolsBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blender: BlenderToolStatus
    orca: OrcaToolStatus
    f3d: F3dToolStatus


class UvTooling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str | None = None
    uv_lock_present: bool = False


class DiskInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pymeshlab_approx_mb: float | None = None
    note: str = (
        "package-dir size only (os.scandir walk); may undercount natives outside the package tree"
    )


class NvidiaProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: VramStatus = "ritual_only"
    name: str | None = None
    free_mib: float | None = None
    total_mib: float | None = None
    error: str | None = None


class VramInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ritual: str
    nvidia: NvidiaProbe = Field(default_factory=NvidiaProbe)


class EnvCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    example: str
    consumer: str


class DoctorReport(BaseModel):
    """Structured ops health report for ``meshops doctor``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    ok: bool
    python: PythonInfo
    packages: dict[str, PackageStatus]
    tools: ToolsBlock
    tooling: UvTooling
    disk: DiskInfo
    licenses: list[str] = Field(default_factory=list)
    env: dict[str, bool] = Field(default_factory=dict)
    env_catalog: list[EnvCatalogItem] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    vram: VramInfo
    required: list[str] = Field(default_factory=list)
    # Reserved for forward-compat diagnostics without breaking extra=forbid clients
    notes: list[str] = Field(default_factory=list)

    def model_dump_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
