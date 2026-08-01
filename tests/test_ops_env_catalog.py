"""Env catalog completeness vs src/ MESHOPS_* reads (R12)."""

from __future__ import annotations

import re
from pathlib import Path

from meshops.ops.env_catalog import (
    ENV_CATALOG,
    ENV_CATALOG_BY_NAME,
    STDOUT_PROTOCOL_TOKENS,
    catalog_names,
)

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

# os.environ.get("MESHOPS_…") / os.environ["MESHOPS_…"]
_ENV_GET_RE = re.compile(r"""os\.environ\.(?:get|__getitem__)\(\s*["'](MESHOPS_[A-Z0-9_]+)["']""")
_ENV_INDEX_RE = re.compile(r"""os\.environ\[\s*["'](MESHOPS_[A-Z0-9_]+)["']\s*\]""")
# ENV_* = "MESHOPS_…" style constants used with os.environ.get(ENV_*)
_CONST_ASSIGN_RE = re.compile(r"""=\s*["'](MESHOPS_[A-Z0-9_]+)["']""")


def _meshops_env_names_in_src() -> set[str]:
    """Collect MESHOPS_* names that are env vars (not stdout protocol tokens)."""
    names: set[str] = set()
    for path in _SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for rx in (_ENV_GET_RE, _ENV_INDEX_RE):
            names.update(rx.findall(text))
        # Constants assigned to MESHOPS_* that are used as env keys
        if "os.environ" in text:
            for m in _CONST_ASSIGN_RE.findall(text):
                names.add(m)
    return names


def test_catalog_non_empty() -> None:
    assert len(ENV_CATALOG) >= 10
    assert catalog_names() == frozenset(ENV_CATALOG_BY_NAME)


def test_bootstrap_only_vars_cataloged() -> None:
    assert "MESHOPS_BLENDER_MIRROR" in ENV_CATALOG_BY_NAME
    assert "MESHOPS_BOOTSTRAP_DIR" in ENV_CATALOG_BY_NAME
    assert "bootstrap" in ENV_CATALOG_BY_NAME["MESHOPS_BLENDER_MIRROR"]["consumer"].lower()


def test_design_stdout_tokens_not_cataloged() -> None:
    for tok in ("MESHOPS_DESIGN_OK", "MESHOPS_DESIGN_ERR"):
        assert tok not in ENV_CATALOG_BY_NAME


def test_every_src_meshops_env_read_is_cataloged() -> None:
    """Every MESHOPS_* env read under src/ has a catalog entry (or is stdout token)."""
    found = _meshops_env_names_in_src()
    env_reads: set[str] = set()
    for name in found:
        if name in STDOUT_PROTOCOL_TOKENS:
            continue
        if name.startswith("MESHOPS_DESIGN_"):
            continue
        env_reads.add(name)

    missing = sorted(env_reads - set(ENV_CATALOG_BY_NAME))
    assert missing == [], f"MESHOPS_* env reads missing from catalog: {missing}"
