"""Profile resolution + flattened default bundle guard (DoD-13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshops.slice.errors import SliceError
from meshops.slice.profiles import (
    ENV_ORCA_FILAMENT,
    ENV_ORCA_MACHINE,
    ENV_ORCA_PROCESS,
    ENV_ORCA_PROFILES,
    default_bundle_has_inherits,
    default_profile_dir,
    resolve_profiles,
)


def test_default_profiles_resolve() -> None:
    pp = resolve_profiles("default")
    assert Path(pp.machine).is_file()
    assert Path(pp.process).is_file()
    assert Path(pp.filament).is_file()
    assert pp.profile_name == "default"
    assert "machine.json" in pp.machine
    assert "process.json" in pp.process
    assert "filament.json" in pp.filament


def test_default_bundle_no_inherits() -> None:
    bad = default_bundle_has_inherits()
    assert bad == [], f"default profiles must be flattened, found inherits in: {bad}"
    base = default_profile_dir()
    for name in ("machine.json", "process.json", "filament.json"):
        data = json.loads((base / name).read_text(encoding="utf-8"))
        assert "inherits" not in data


def test_missing_profile_name() -> None:
    with pytest.raises(SliceError) as ei:
        resolve_profiles("definitely_not_a_real_profile_xyz")
    assert ei.value.code == "profile_not_found"


def test_env_override_all_three(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = tmp_path / "m.json"
    p = tmp_path / "p.json"
    f = tmp_path / "f.json"
    for path in (m, p, f):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(ENV_ORCA_MACHINE, str(m))
    monkeypatch.setenv(ENV_ORCA_PROCESS, str(p))
    monkeypatch.setenv(ENV_ORCA_FILAMENT, str(f))
    pp = resolve_profiles("ignored")
    assert Path(pp.machine).resolve() == m.resolve()
    assert Path(pp.process).resolve() == p.resolve()
    assert Path(pp.filament).resolve() == f.resolve()


def test_partial_env_override_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = tmp_path / "m.json"
    m.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(ENV_ORCA_MACHINE, str(m))
    monkeypatch.delenv(ENV_ORCA_PROCESS, raising=False)
    monkeypatch.delenv(ENV_ORCA_FILAMENT, raising=False)
    with pytest.raises(SliceError) as ei:
        resolve_profiles("default")
    assert ei.value.code == "profile_not_found"


def test_abs_dir_profile(tmp_path: Path) -> None:
    for name in ("machine.json", "process.json", "filament.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    pp = resolve_profiles(str(tmp_path))
    assert Path(pp.machine).parent.resolve() == tmp_path.resolve()


def test_named_under_env_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    named = tmp_path / "custom"
    named.mkdir()
    for name in ("machine.json", "process.json", "filament.json"):
        (named / name).write_text("{}", encoding="utf-8")
    monkeypatch.setenv(ENV_ORCA_PROFILES, str(tmp_path))
    monkeypatch.delenv(ENV_ORCA_MACHINE, raising=False)
    monkeypatch.delenv(ENV_ORCA_PROCESS, raising=False)
    monkeypatch.delenv(ENV_ORCA_FILAMENT, raising=False)
    pp = resolve_profiles("custom")
    assert pp.profile_name == "custom"
    assert Path(pp.machine).is_file()
