"""Guard multi-signal + two-tier policy tests (Difficulty §5, §6)."""

from __future__ import annotations

from meshops.guards import GuardPolicy, check_export
from meshops.models.diagnostics import MeshStats


def _stats(
    *,
    faces: int,
    vertices: int | None = None,
    bytes_: int,
    components: int = 1,
    bbox_min: tuple[float, float, float] = (10.0, 20.0, 30.0),
    bbox_max: tuple[float, float, float] = (110.0, 120.0, 130.0),
    is_volume: bool | None = True,
    mesh_id: str = "t",
) -> MeshStats:
    bmin = bbox_min
    bmax = bbox_max
    diag = ((bmax[0] - bmin[0]) ** 2 + (bmax[1] - bmin[1]) ** 2 + (bmax[2] - bmin[2]) ** 2) ** 0.5
    return MeshStats(
        faces=faces,
        vertices=vertices if vertices is not None else faces // 2,
        bbox_min=bmin,
        bbox_max=bmax,
        bbox_diagonal=diag,
        components=components,
        is_watertight=is_volume,
        is_volume=is_volume,
        is_manifold=True,
        file_size_bytes=bytes_,
        content_sha256="a" * 64,
        mesh_id=mesh_id,
    )


def test_guards__same_mesh__pass() -> None:
    s = _stats(faces=100_000, bytes_=5_000_000)
    r = check_export(s, s, policy=GuardPolicy.for_export())
    assert r.ok is True
    assert r.failed == []


def test_guards__face_floor_export_fail() -> None:
    base = _stats(faces=1000, bytes_=50_000)
    cand = _stats(faces=400, bytes_=40_000)  # 0.40 < 0.50
    r = check_export(base, cand, policy=GuardPolicy.for_export())
    assert r.ok is False
    assert "face_floor" in r.failed


def test_guards__face_floor_export_pass() -> None:
    base = _stats(faces=1000, bytes_=50_000)
    cand = _stats(faces=600, bytes_=40_000)  # 0.60 >= 0.50
    r = check_export(base, cand, policy=GuardPolicy.for_export())
    assert r.ok is True


def test_guards__recipe_tier_face_floor__fail() -> None:
    base = _stats(faces=1000, bytes_=50_000)
    cand = _stats(faces=800, bytes_=45_000)  # 0.80 < 0.90 recipe
    r = check_export(base, cand, policy=GuardPolicy.for_recipe("t1_clean"))
    assert r.ok is False
    assert "face_floor" in r.failed
    assert r.policy_tier == "recipe"


def test_guards__recipe_tier_face_floor_pass() -> None:
    base = _stats(faces=1000, bytes_=50_000)
    cand = _stats(faces=950, bytes_=48_000)
    r = check_export(base, cand, policy=GuardPolicy.for_recipe("t1_clean"))
    assert r.ok is True


def test_guards__size_floor__fail() -> None:
    base = _stats(faces=1000, bytes_=100_000)
    cand = _stats(faces=1000, bytes_=30_000)  # 0.30 < 0.40 export
    r = check_export(base, cand, policy=GuardPolicy.for_export())
    assert r.ok is False
    assert "size_floor" in r.failed


def test_guards__size_floor__pass() -> None:
    base = _stats(faces=1000, bytes_=100_000)
    cand = _stats(faces=1000, bytes_=50_000)
    r = check_export(base, cand, policy=GuardPolicy.for_export())
    assert r.ok is True


def test_guards__wipeout_class__hard_fail() -> None:
    """Hero-scale baseline + ~358KB / ~7k faces → wipeout_class (Difficulty §6)."""
    base = _stats(faces=500_000, bytes_=25_000_000)  # hero
    cand = _stats(
        faces=7000,
        bytes_=358_000,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
    )
    r = check_export(base, cand, policy=GuardPolicy.for_export())
    assert r.ok is False
    assert "wipeout_class" in r.failed or "face_collapse" in r.failed


def test_guards__face_collapse__hard_fail() -> None:
    base = _stats(faces=300_000, bytes_=12_000_000)
    cand = _stats(faces=20_000, bytes_=2_000_000)  # <10% faces, not tiny bytes
    r = check_export(base, cand, policy=GuardPolicy.for_export())
    assert r.ok is False
    assert "face_collapse" in r.failed


def test_guards__component_collapse__hard_fail() -> None:
    base = _stats(faces=300_000, bytes_=12_000_000, components=20)
    cand = _stats(faces=280_000, bytes_=11_000_000, components=2)  # 90% drop
    r = check_export(base, cand, policy=GuardPolicy.for_export())
    assert r.ok is False
    assert "component_collapse" in r.failed


def test_guards__bbox_origin_orphan__hard_fail() -> None:
    """Candidate near origin while baseline offset (Difficulty §5)."""
    base = _stats(
        faces=300_000,
        bytes_=12_000_000,
        bbox_min=(100.0, 200.0, 300.0),
        bbox_max=(200.0, 300.0, 400.0),
    )
    cand = _stats(
        faces=290_000,
        bytes_=11_500_000,
        bbox_min=(-0.5, -0.5, -0.5),
        bbox_max=(0.5, 0.5, 0.5),
    )
    r = check_export(base, cand, policy=GuardPolicy.for_export())
    assert r.ok is False
    assert "bbox_origin_orphan" in r.failed or "bbox_drift" in r.failed


def test_guards__component_explosion__fail() -> None:
    base = _stats(faces=1000, bytes_=50_000, components=1)
    cand = _stats(faces=1000, bytes_=50_000, components=50)
    r = check_export(base, cand, policy=GuardPolicy.for_export())
    assert r.ok is False
    assert "components" in r.failed


def test_guards__recipe_still_enforces_global_wipeout() -> None:
    """Recipe policy cannot loosen wipeout floors on hero-scale."""
    base = _stats(faces=500_000, bytes_=25_000_000)
    cand = _stats(faces=7000, bytes_=358_000)
    r = check_export(base, cand, policy=GuardPolicy.for_recipe("t1_clean"))
    assert r.ok is False
    assert any(c in r.failed for c in ("wipeout_class", "face_collapse", "face_floor"))
