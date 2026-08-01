"""Local image path → base64 data URI (stdlib only — no network)."""

from __future__ import annotations

import base64
from pathlib import Path

from meshops.hosted.errors import HostedError

# Magic sniff
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


def path_to_data_uri(path: Path | str) -> str:
    """Encode a local image file as ``data:image/{png|jpeg};base64,...``.

    Never returns a local file path as a provider URL (C9).
    """
    p = Path(path)
    if not p.is_file():
        raise HostedError(
            f"view image not found: {p}",
            code="multiview_required",
            details={"path": str(p)},
        )
    data = p.read_bytes()
    if not data:
        raise HostedError(
            f"view image empty: {p}",
            code="multiview_required",
            details={"path": str(p)},
        )
    if data.startswith(_PNG_MAGIC):
        mime = "image/png"
    elif data.startswith(_JPEG_MAGIC):
        mime = "image/jpeg"
    else:
        # Default to png; providers still get a data URI rather than a path.
        mime = "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
