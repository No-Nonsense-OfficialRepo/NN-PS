from collections import namedtuple

import pytest


def test_preview_does_not_move_and_reuses_cached_metadata(ph, tmp_path, monkeypatch):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"photo")
    calls = []
    def read(*args, **kwargs):
        calls.append(True)
        return {ph.norm(path): (None, True, "", (100, 100))}
    monkeypatch.setattr(ph, "read_info", read)
    preview, _ = ph.sort_files([path], None, tmp_path, lambda *_: None, 0, 1, preview=True)
    assert path.exists()
    assert not (tmp_path / "Sorted").exists()
    assert any("DATE FALLBACK" in line for line in preview.preview_lines)
    assert len(calls) == 1
    monkeypatch.setattr(ph, "classify", lambda *a, **k: pytest.fail("classification should be cached"))
    result, _ = ph.sort_files([path], None, tmp_path, lambda *_: None, 0, 1)
    assert len(calls) == 1
    assert result[0].exists() and not path.exists()


def test_preview_reserves_conflicting_destinations(ph, tmp_path, monkeypatch):
    paths = []
    for name in ("one", "two"):
        parent = tmp_path / name
        parent.mkdir()
        path = parent / "photo.jpg"
        path.write_bytes(b"photo")
        paths.append(path)
    monkeypatch.setattr(ph, "read_info", lambda *a, **k: {ph.norm(p): ("2020-05", True, "", (100, 100)) for p in paths})
    result, _ = ph.sort_files(paths, None, tmp_path, lambda *_: None, 0, 1, preview=True)
    assert result[0] != result[1]
    assert all(p.exists() for p in paths)


def test_cross_volume_verified_move(ph, tmp_path, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.write_bytes(b"original")
    monkeypatch.setattr(ph, "same_volume", lambda *a: False)
    ph.verified_move(source, dest)
    assert dest.read_bytes() == b"original" and not source.exists()


def test_bad_copy_preserves_source(ph, tmp_path, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.write_bytes(b"original")
    monkeypatch.setattr(ph, "same_volume", lambda *a: False)
    monkeypatch.setattr(ph.shutil, "copy2", lambda src, dst: dst.write_bytes(b"corrupted"))
    with pytest.raises(OSError, match="verification"):
        ph.verified_move(source, dest)
    assert source.read_bytes() == b"original" and not dest.exists()
    assert not list(tmp_path.glob("*.moving-*"))


def test_insufficient_space_preserves_source(ph, tmp_path, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.write_bytes(b"original")
    monkeypatch.setattr(ph, "same_volume", lambda *a: False)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(ph.shutil, "disk_usage", lambda *a: usage(100, 100, 0))
    with pytest.raises(OSError, match="free space"):
        ph.verified_move(source, dest)
    assert source.exists() and not dest.exists()
