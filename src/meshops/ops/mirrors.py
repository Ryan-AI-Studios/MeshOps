"""Blender portable zip mirror list + integrity pin (Difficulty §4).

Bootstrap only — never read by find_blender. Official CDN is last (403 class).
"""

from __future__ import annotations

from typing import Final

# TechStack pin: Blender 5.2.0 LTS windows-x64 portable zip.
BLENDER_VERSION: Final = "5.2.0"
BLENDER_ZIP_NAME: Final = "blender-5.2.0-windows-x64.zip"
BLENDER_SHA256_FILENAME: Final = "blender-5.2.0.sha256"

# Pinned SHA-256 of blender-5.2.0-windows-x64.zip (fail closed on mismatch — R7).
BLENDER_WINDOWS_X64_SHA256: Final = (
    "2d184b626c001692c362291911293b6a297179d618d95e9e9192c3a80318adc4"
)

# Layout after bootstrap flatten: %LOCALAPPDATA%\MeshOps\tools\blender-5.2.0\blender.exe
PORTABLE_BLENDER_DIR_NAME: Final = "blender-5.2.0"
PORTABLE_TOOLS_REL: Final = "MeshOps/tools"
PORTABLE_BLENDER_EXE_NAME: Final = "blender.exe"

# Nested folder name inside the official zip (bootstrap flattens to PORTABLE_BLENDER_DIR_NAME).
ZIP_NESTED_DIR_NAME: Final = "blender-5.2.0-windows-x64"

# Relative path under each mirror root to the zip.
_RELEASE_PATH: Final = f"release/Blender5.2/{BLENDER_ZIP_NAME}"

# Ordered non-CDN-first list. Official download.blender.org MUST be last (Difficulty §4).
# MESHOPS_BLENDER_MIRROR / -MirrorUrl may prepend a single override at runtime.
BLENDER_MIRROR_URLS: Final[tuple[str, ...]] = (
    # 1. RWTH Aachen (working + checksums — preferred first)
    f"https://ftp.halifax.rwth-aachen.de/blender/{_RELEASE_PATH}",
    # 2. dotsrc (DK)
    f"https://mirrors.dotsrc.org/blender/{_RELEASE_PATH}",
    # 3-4. Best-effort short timeout mirrors (may 522)
    f"https://mirror.cicku.me/blender/{_RELEASE_PATH}",
    f"https://mirrors.iu13.net/blender/{_RELEASE_PATH}",
    # 5. Official CDN last (403 class for many automated clients)
    f"https://download.blender.org/{_RELEASE_PATH}",
)

OFFICIAL_BLENDER_DOWNLOAD_URL: Final = BLENDER_MIRROR_URLS[-1]


def mirror_checksum_url(zip_url: str) -> str:
    """Sibling ``blender-5.2.0.sha256`` URL for a zip mirror URL."""
    if zip_url.endswith(BLENDER_ZIP_NAME):
        return zip_url[: -len(BLENDER_ZIP_NAME)] + BLENDER_SHA256_FILENAME
    return zip_url.rsplit("/", 1)[0] + "/" + BLENDER_SHA256_FILENAME
