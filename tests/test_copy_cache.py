import os

import cv2
import numpy as np

from imgutil import photo_like


def test_cache_persists_hash_and_features(ph, tmp_path):
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), photo_like())
    cache = ph.CopyCache(tmp_path)
    expected_hash = cache.get(path, "hash", ph.quick_hash)
    expected = cache.get(path, "features", ph.look_features)
    cache.close()

    def unexpected(_):
        raise AssertionError("unchanged file was decoded again")

    cache = ph.CopyCache(tmp_path)
    try:
        assert cache.get(path, "hash", unexpected) == expected_hash
        actual = cache.get(path, "features", unexpected)
        assert actual["pix"] == expected["pix"]
        np.testing.assert_array_equal(actual["g64"], expected["g64"])
        np.testing.assert_array_equal(actual["pts"], expected["pts"])
    finally:
        cache.close()


def test_changed_file_is_recomputed(ph, tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"one")
    cache = ph.CopyCache(tmp_path)
    try:
        assert cache.get(path, "hash", lambda _: 1) == 1
        stamp = path.stat()
        path.write_bytes(b"different")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert cache.get(path, "hash", lambda _: 2) == 2
    finally:
        cache.close()


def test_corrupt_database_falls_back_to_computation(ph, tmp_path):
    folder = tmp_path / ph.INFO_DIR
    folder.mkdir()
    (folder / "copies.sqlite3").write_bytes(b"broken database")
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"photo")
    cache = ph.CopyCache(tmp_path)
    try:
        assert cache.get(path, "hash", lambda _: 123) == 123
    finally:
        cache.close()


def test_changed_during_read_is_not_cached(ph, tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"photo")
    cache = ph.CopyCache(tmp_path)
    def changing(_):
        path.write_bytes(b"changed photo")
        return 1
    try:
        assert cache.get(path, "hash", changing) is None
        assert cache.get(path, "hash", lambda _: 2) == 2
    finally:
        cache.close()
