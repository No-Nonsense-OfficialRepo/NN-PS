"""Regression checks for moving copies and recovering a run."""
import cv2

from imgutil import photo_like


def test_crowded_fingerprint_group_keeps_one(ph, tmp_path):
    image = photo_like(320, 240, seed=9)
    ok, encoded = cv2.imencode(".bmp", image)
    assert ok
    paths = []
    for i in range(12):
        path = tmp_path / f"picture_{i}.bmp"
        path.write_bytes(encoded.tobytes())
        paths.append(path)
    ph.start_move_run(tmp_path)
    assert ph.find_lookalikes(paths, tmp_path, lambda *_: None) == 11
    assert sum(p.exists() for p in paths) == 1
    assert len(list((tmp_path / "_Verify").glob("*.bmp"))) == 11


def test_one_exact_move_failure_does_not_stop_others(ph, tmp_path, monkeypatch):
    keeper = tmp_path / "keeper.jpg"
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    for path in (keeper, first, second):
        path.write_bytes(b"same bytes")
    ph.start_move_run(tmp_path)
    real_move = ph.move_logged

    def fail_first(src, dst, root=None):
        if src == first:
            raise OSError("locked")
        return real_move(src, dst, root)

    monkeypatch.setattr(ph, "move_logged", fail_first)
    result = ph._exact_pass(tmp_path, None, {keeper: [first, second]}, {keeper: keeper}, {},
                            lambda *_: None, ("copies",))
    assert result["dups"] == 1
    assert result["copy_failures"] == 1
    assert first.exists()
    assert not second.exists()


def test_new_empty_run_does_not_offer_previous_undo(ph, tmp_path):
    source = tmp_path / "source.jpg"
    dest = tmp_path / "_Duplicates" / "source.jpg"
    dest.parent.mkdir()
    source.write_bytes(b"photo")
    ph.start_move_run(tmp_path)
    ph.move_logged(source, dest, tmp_path)
    assert len(ph.pending_moves(tmp_path)) == 1
    ph.start_move_run(tmp_path)
    assert ph.pending_moves(tmp_path) == []
    assert ph.undo_last_run(tmp_path) == {"restored": 0, "skipped": 0}
    assert dest.exists()


def test_copies_only_runs_without_exiftool_and_can_undo(ph, tmp_path):
    source = tmp_path / "photo.jpg"
    duplicate = tmp_path / "photo (1).jpg"
    source.write_bytes(b"same file")
    duplicate.write_bytes(source.read_bytes())
    ph.start_move_run(tmp_path)
    prepared = ph.stage_prepare(tmp_path, None, lambda *_: None, ("copies",))
    result = ph.stage_deep(tmp_path, None, prepared["groups"], prepared["current"],
                           prepared["cache"], lambda *_: None, ("copies",))
    assert result["dups"] == 1
    assert source.exists()
    assert not duplicate.exists()
    assert ph.undo_last_run(tmp_path) == {"restored": 1, "skipped": 0}
    assert duplicate.read_bytes() == source.read_bytes()
