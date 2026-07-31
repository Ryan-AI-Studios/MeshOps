"""Blender discovery + 5.2 version gate (DoD-1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meshops.escalate.discover import (
    ENV_MESHOPS_BLENDER,
    WELL_KNOWN_WINDOWS_BLENDER,
    find_blender,
)
from meshops.escalate.errors import EscalateError
from meshops.escalate.version import parse_blender_version, require_blender_52


def test_parse_blender_version_52() -> None:
    major, minor, patch, raw = parse_blender_version("Blender 5.2.0 LTS\nbuild date: ...\n")
    assert major == 5
    assert minor == 2
    assert patch == 0
    assert "5.2.0" in raw


def test_parse_blender_version_42() -> None:
    major, minor, patch, _ = parse_blender_version("Blender 4.2.9\n")
    assert (major, minor, patch) == (4, 2, 9)


def test_require_52_rejects_42(tmp_path: Path) -> None:
    fake = tmp_path / "blender.exe"
    fake.write_text("", encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.stdout = "Blender 4.2.9\n"
    mock_proc.stderr = ""
    mock_proc.returncode = 0

    with (
        patch("meshops.escalate.version.subprocess.run", return_value=mock_proc),
        pytest.raises(EscalateError) as ei,
    ):
        require_blender_52(fake)
    assert ei.value.code == "blender_version"
    assert "4.2" in str(ei.value) or "5.2" in str(ei.value)


def test_require_52_accepts_52(tmp_path: Path) -> None:
    fake = tmp_path / "blender.exe"
    fake.write_text("", encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.stdout = "Blender 5.2.0 LTS\nbuild date: 2026-07-14\n"
    mock_proc.stderr = ""
    mock_proc.returncode = 0

    with patch("meshops.escalate.version.subprocess.run", return_value=mock_proc):
        ver = require_blender_52(fake)
    assert ver == "5.2.0"


def test_require_52_rejects_nonzero_returncode(tmp_path: Path) -> None:
    """Fail-closed: nonzero --version exit must not accept a version string."""
    fake = tmp_path / "blender.exe"
    fake.write_text("", encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.stdout = "Blender 5.2.0 LTS\n"
    mock_proc.stderr = "fatal: crash\n"
    mock_proc.returncode = 1

    with (
        patch("meshops.escalate.version.subprocess.run", return_value=mock_proc),
        pytest.raises(EscalateError) as ei,
    ):
        require_blender_52(fake)
    assert ei.value.code == "blender_version"
    assert "exited" in str(ei.value).lower() or "1" in str(ei.value)


def test_find_blender_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "my_blender.exe"
    fake.write_bytes(b"x")
    monkeypatch.setenv(ENV_MESHOPS_BLENDER, str(fake))
    monkeypatch.delenv("PATH", raising=False)
    found = find_blender(require=True)
    assert found is not None
    assert found.resolve() == fake.resolve()


def test_find_blender_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_MESHOPS_BLENDER, raising=False)
    # Point well-known + which away from real installs
    monkeypatch.setattr(
        "meshops.escalate.discover.WELL_KNOWN_WINDOWS_BLENDER",
        tmp_path / "nope" / "blender.exe",
    )
    monkeypatch.setattr("meshops.escalate.discover.shutil.which", lambda _: None)
    with pytest.raises(EscalateError) as ei:
        find_blender(require=True)
    assert ei.value.code == "blender_missing"
    assert "MESHOPS_BLENDER" in str(ei.value) or "0010" in str(ei.value)


def test_find_blender_require_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_MESHOPS_BLENDER, raising=False)
    monkeypatch.setattr(
        "meshops.escalate.discover.WELL_KNOWN_WINDOWS_BLENDER",
        tmp_path / "nope" / "blender.exe",
    )
    monkeypatch.setattr("meshops.escalate.discover.shutil.which", lambda _: None)
    assert find_blender(require=False) is None


def test_well_known_path_constant() -> None:
    assert "Blender 5.2" in str(WELL_KNOWN_WINDOWS_BLENDER)
