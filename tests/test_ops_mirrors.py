"""Mirror list + SHA pin unit tests (no network)."""

from __future__ import annotations

import re
from pathlib import Path

from meshops.ops.mirrors import (
    BLENDER_MIRROR_URLS,
    BLENDER_WINDOWS_X64_SHA256,
    BLENDER_ZIP_NAME,
    OFFICIAL_BLENDER_DOWNLOAD_URL,
    PORTABLE_BLENDER_DIR_NAME,
    mirror_checksum_url,
)

_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap-tools.ps1"


def test_mirror_list_min_three() -> None:
    assert len(BLENDER_MIRROR_URLS) >= 3


def test_official_not_first() -> None:
    assert "download.blender.org" not in BLENDER_MIRROR_URLS[0]
    assert BLENDER_MIRROR_URLS[-1] == OFFICIAL_BLENDER_DOWNLOAD_URL
    assert "download.blender.org" in OFFICIAL_BLENDER_DOWNLOAD_URL


def test_sha256_pin_length_hex() -> None:
    assert len(BLENDER_WINDOWS_X64_SHA256) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", BLENDER_WINDOWS_X64_SHA256)


def test_zip_name_and_checksum_url() -> None:
    assert BLENDER_ZIP_NAME == "blender-5.2.0-windows-x64.zip"
    first = BLENDER_MIRROR_URLS[0]
    assert first.endswith(BLENDER_ZIP_NAME)
    cs = mirror_checksum_url(first)
    assert cs.endswith(".sha256")
    assert "download.blender.org" not in first  # RWTH first


def test_bootstrap_ps1_pins_match_mirrors_py() -> None:
    """Drift guard: PowerShell bootstrap pins must match mirrors.py (P3-1)."""
    assert _PS1.is_file(), f"missing bootstrap script: {_PS1}"
    text = _PS1.read_text(encoding="utf-8-sig")
    m_sha = re.search(
        r'\$PinnedSha256\s*=\s*"([0-9a-fA-F]{64})"',
        text,
    )
    assert m_sha is not None, "PinnedSha256 not found in bootstrap-tools.ps1"
    assert m_sha.group(1).lower() == BLENDER_WINDOWS_X64_SHA256

    m_zip = re.search(r'\$ZipName\s*=\s*"([^"]+)"', text)
    assert m_zip is not None
    assert m_zip.group(1) == BLENDER_ZIP_NAME

    m_dir = re.search(r'\$PortableDirName\s*=\s*"([^"]+)"', text)
    assert m_dir is not None
    assert m_dir.group(1) == PORTABLE_BLENDER_DIR_NAME

    # PS1 builds URLs from host + $ReleaseRel; order only inside $MirrorList = @(...)
    m_list = re.search(
        r"\$MirrorList\s*=\s*@\((.*?)\)",
        text,
        flags=re.DOTALL,
    )
    assert m_list is not None, "$MirrorList block missing from bootstrap-tools.ps1"
    mirror_block = m_list.group(1)
    host_order = [
        "ftp.halifax.rwth-aachen.de",
        "mirrors.dotsrc.org",
        "mirror.cicku.me",
        "mirrors.iu13.net",
        "download.blender.org",
    ]
    positions: list[int] = []
    for host in host_order:
        pos = mirror_block.find(host)
        assert pos >= 0, f"mirror host missing from $MirrorList: {host}"
        positions.append(pos)
    assert positions == sorted(positions), "mirror hosts not in expected order in $MirrorList"
    assert "release/Blender5.2" in text
    assert BLENDER_ZIP_NAME in text

    # No setx usage (comments may mention "never setx")
    assert not re.search(r"(?i)\bsetx\b(?![^\n]*(never|do not|don't))", text) or re.search(
        r"(?i)never\s+(uses?\s+)?setx",
        text,
    )

    # Surface markers for -WhatIf / SHA / env (P3-5)
    assert "-WhatIf" in text or "$WhatIf" in text
    assert "MESHOPS_BLENDER" in text
    assert "SetEnvironmentVariable" in text
