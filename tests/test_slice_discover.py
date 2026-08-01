"""Orca discovery + AppData version probe (DoD-3). Never uses --help."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from meshops.slice.discover import (
    ENV_MESHOPS_ORCA,
    ENV_MESHOPS_ORCASLICER,
    WELL_KNOWN_WINDOWS_ORCA,
    find_orca,
    read_orca_version_from_appdata,
    soft_version_ok,
)
from meshops.slice.errors import SliceError


def test_find_orca_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "orca-slicer.exe"
    fake.write_bytes(b"x")
    monkeypatch.setenv(ENV_MESHOPS_ORCA, str(fake))
    monkeypatch.delenv(ENV_MESHOPS_ORCASLICER, raising=False)
    monkeypatch.setattr("meshops.slice.discover.shutil.which", lambda _: None)
    found = find_orca(require=False)
    assert found is not None
    assert found.resolve() == fake.resolve()


def test_find_orca_alias_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "orcaslicer.exe"
    fake.write_bytes(b"x")
    monkeypatch.delenv(ENV_MESHOPS_ORCA, raising=False)
    monkeypatch.setenv(ENV_MESHOPS_ORCASLICER, str(fake))
    monkeypatch.setattr("meshops.slice.discover.shutil.which", lambda _: None)
    found = find_orca(require=False)
    assert found is not None
    assert found.resolve() == fake.resolve()


def test_find_orca_missing_require_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_MESHOPS_ORCA, raising=False)
    monkeypatch.delenv(ENV_MESHOPS_ORCASLICER, raising=False)
    monkeypatch.setattr(
        "meshops.slice.discover.WELL_KNOWN_WINDOWS_ORCA",
        tmp_path / "nope" / "orca-slicer.exe",
    )
    monkeypatch.setattr("meshops.slice.discover.shutil.which", lambda _: None)
    assert find_orca(require=False) is None


def test_find_orca_missing_require_true(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_MESHOPS_ORCA, raising=False)
    monkeypatch.delenv(ENV_MESHOPS_ORCASLICER, raising=False)
    monkeypatch.setattr(
        "meshops.slice.discover.WELL_KNOWN_WINDOWS_ORCA",
        tmp_path / "nope" / "orca-slicer.exe",
    )
    monkeypatch.setattr("meshops.slice.discover.shutil.which", lambda _: None)
    with pytest.raises(SliceError) as ei:
        find_orca(require=True)
    assert ei.value.code == "orca_not_found"


def test_well_known_path_constant() -> None:
    assert "OrcaSlicer" in str(WELL_KNOWN_WINDOWS_ORCA)
    assert "orca-slicer" in str(WELL_KNOWN_WINDOWS_ORCA).lower()


def test_soft_version_ok() -> None:
    assert soft_version_ok("2.4.2") is True
    assert soft_version_ok("2.3.0") is False
    assert soft_version_ok("1.9.0") is False
    assert soft_version_ok("3.0.0") is True
    assert soft_version_ok(None) is True


def test_read_orca_version_from_appdata_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conf = tmp_path / "OrcaSlicer.conf"
    conf.write_text('{"version": "2.4.2", "other": 1}\n', encoding="utf-8")
    monkeypatch.setattr(
        "meshops.slice.discover.orca_appdata_conf_path",
        lambda: conf,
    )
    assert read_orca_version_from_appdata() == "2.4.2"


def test_read_orca_version_regex_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conf = tmp_path / "OrcaSlicer.conf"
    conf.write_text('not-json\n"version": "2.4.1"\n', encoding="utf-8")
    monkeypatch.setattr(
        "meshops.slice.discover.orca_appdata_conf_path",
        lambda: conf,
    )
    assert read_orca_version_from_appdata() == "2.4.1"


def test_version_probe_never_calls_help(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: never shell out to --help for version."""
    conf = tmp_path / "OrcaSlicer.conf"
    conf.write_text('{"version": "2.4.2"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "meshops.slice.discover.orca_appdata_conf_path",
        lambda: conf,
    )
    with patch("subprocess.run") as run_mock:
        ver = read_orca_version_from_appdata()
        assert ver == "2.4.2"
        run_mock.assert_not_called()
