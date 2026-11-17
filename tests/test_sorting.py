import pytest


def metadata(ph, monkeypatch, files, month="2020-05"):
    monkeypatch.setattr(ph, "read_info", lambda *a, **k: {ph.norm(p): (month, True, "", (100, 100)) for p in files})


def test_invalid_primary_date_uses_valid_secondary(ph, tmp_path, monkeypatch):
    p = tmp_path / "a.jpg"
    monkeypatch.setattr(ph, "exif_json", lambda *a: [{"SourceFile": str(p), "DateTimeOriginal": "0000-00",
                                                      "CreateDate": "2020-05"}])
    assert ph.read_info(None, [p])[ph.norm(p)][0] == "2020-05"


def test_metadata_error_stops_before_moving(ph, tmp_path, monkeypatch):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"a")
    monkeypatch.setattr(ph, "run_exif", lambda *a, **k: "not JSON")
    with pytest.raises(RuntimeError, match="Metadata"):
        ph.sort_files([p], None, tmp_path, lambda *_: None, 0, 1)
    assert p.exists()
    assert not (tmp_path / "Sorted").exists()


def test_exiftool_exit_failure_is_reported(ph, monkeypatch):
    class FailedProcess:
        returncode = 1

        def communicate(self, timeout=None):
            return b"", b"failed"

        def poll(self):
            return self.returncode

    monkeypatch.setattr(ph.subprocess, "Popen", lambda *a, **k: FailedProcess())
    with pytest.raises(RuntimeError, match="ExifTool"):
        ph.run_exif("exiftool.exe", [])


def test_bundle_collision_sidecars_repeat_and_undo(ph, tmp_path, monkeypatch):
    raw, jpg = tmp_path / "photo.nef", tmp_path / "photo.jpg"
    xmp, editor = tmp_path / "photo.xmp", tmp_path / "photo.nef.pp3"
    files = [raw, jpg]
    for p in (raw, jpg, xmp, editor):
        p.write_bytes(p.name.encode())
    target = tmp_path / "Sorted" / "2020" / "2020-05"
    target.mkdir(parents=True)
    occupied = target / "photo.xmp"
    occupied.write_bytes(b"existing")
    metadata(ph, monkeypatch, files)
    ph.start_move_run(tmp_path)
    paths, _ = ph.sort_files(files, None, tmp_path, lambda *_: None, 0, 1)
    assert [p.name for p in paths] == ["photo_1.nef", "photo_1.jpg"]
    assert (target / "photo_1.xmp").read_bytes() == b"photo.xmp"
    assert (target / "photo_1.nef.pp3").exists()
    metadata(ph, monkeypatch, paths)
    again, _ = ph.sort_files(paths, None, tmp_path, lambda *_: None, 0, 1)
    assert again == paths
    assert ph.undo_last_run(tmp_path)["restored"] == 4
    assert all(p.exists() for p in (raw, jpg, xmp, editor))
    assert occupied.read_bytes() == b"existing"


def test_failed_bundle_rolls_back_and_next_group_continues(ph, tmp_path, monkeypatch):
    first, sidecar, second = tmp_path / "a.nef", tmp_path / "a.xmp", tmp_path / "b.jpg"
    for p in (first, sidecar, second):
        p.write_bytes(b"photo")
    metadata(ph, monkeypatch, [first, second])
    real_move = ph.move_logged
    def move(src, dst, root=None):
        if src == sidecar:
            raise OSError("locked")
        return real_move(src, dst, root)
    monkeypatch.setattr(ph, "move_logged", move)
    paths, _ = ph.sort_files([first, second], None, tmp_path, lambda *_: None, 0, 1)
    assert paths.failures == 1
    assert first.exists() and sidecar.exists()
    assert paths[1].exists() and not second.exists()
    assert "locked" in (tmp_path / ph.INFO_DIR / "Sort Report.txt").read_text()


def test_missing_metadata_leaves_file_in_place(ph, tmp_path, monkeypatch):
    p = tmp_path / "camera.png"
    p.write_bytes(b"photo")
    monkeypatch.setattr(ph, "read_info", lambda *a, **k: {})
    paths, memes = ph.sort_files([p], None, tmp_path, lambda *_: None, 0, 1)
    assert paths == [p] and paths.failures == 1 and not memes
    assert p.exists()


def test_folder_names_do_not_override_camera_or_output_classification(ph, tmp_path):
    root = tmp_path / "Screenshot Collection"
    camera = root / "screenshots" / "camera.jpg"
    old = root / ph.MEMES_DIR / "2020" / "2020-05" / "photo.jpg"
    video = root / "memes" / "video.mp4"
    info = {ph.norm(p): ("2020-05", True, "", (100, 100)) for p in (camera, old, video)}
    assert ph.classify([camera, old, video], root, info, "standard") == set()
    info[ph.norm(old)] = ("2020-05", False, "", (100, 100))
    assert old not in ph.classify([old], root, info, "standard")
