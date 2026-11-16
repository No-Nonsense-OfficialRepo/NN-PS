"""move_logged + undo_last_run round trip in a temporary folder."""
import pytest


def _need(ph, name):
    if not hasattr(ph, name):
        pytest.skip(f"{name} is not implemented yet (engine task D2)")


def test_move_then_undo_restores_file(ph, tmp_path):
    _need(ph, "move_logged")
    _need(ph, "undo_last_run")
    root = tmp_path / "photos"
    sub = root / "sorted"
    sub.mkdir(parents=True)
    src = root / "a.jpg"
    dst = sub / "a.jpg"
    src.write_bytes(b"AAA")

    ph.move_logged(str(src), str(dst))
    assert not src.exists()
    assert dst.read_bytes() == b"AAA"

    counts = ph.undo_last_run(root)
    assert src.read_bytes() == b"AAA"
    assert not dst.exists()
    assert counts is not None


def test_undo_skips_when_source_already_exists(ph, tmp_path):
    _need(ph, "move_logged")
    _need(ph, "undo_last_run")
    root = tmp_path / "photos"
    sub = root / "sorted"
    sub.mkdir(parents=True)
    src = root / "b.jpg"
    dst = sub / "b.jpg"
    src.write_bytes(b"ORIGINAL")
    ph.move_logged(str(src), str(dst))
    src.write_bytes(b"NEW")  # something else now sits at the original place
    ph.undo_last_run(root)
    assert src.read_bytes() == b"NEW", "undo overwrote a file that was already there"
    assert dst.read_bytes() == b"ORIGINAL"