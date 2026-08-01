"""approx_package_dir_mb scandir walk tests."""

from __future__ import annotations

from pathlib import Path

from meshops.ops.sizes import approx_package_dir_mb


def test_approx_size_temp_tree(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"x" * 1024)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * (1024 * 100))

    mb = approx_package_dir_mb(tmp_path)
    assert mb is not None
    # ~101 KiB → ~0.1 MiB
    assert 0.0 < mb < 1.0


def test_approx_size_missing() -> None:
    assert approx_package_dir_mb(Path("/nonexistent/meshops/path/xyz")) is None


def test_approx_size_file_not_dir(tmp_path: Path) -> None:
    f = tmp_path / "only.txt"
    f.write_text("hi", encoding="utf-8")
    assert approx_package_dir_mb(f) is None
