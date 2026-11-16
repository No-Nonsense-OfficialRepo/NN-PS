import io
import os
from pathlib import Path

import cv2
import numpy as np
import pytest


def face(ph, path, index=0, box=(.1, .1, .2, .2)):
    return ph.Face(str(path), box, (100, 100), np.eye(128, dtype=np.float32)[index], 1., b"thumb")


def test_equal_size_and_timestamp_do_not_share_faces(ph, tmp_path, monkeypatch):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"AAAA"); b.write_bytes(b"BBBB")
    stamp = a.stat().st_mtime_ns
    os.utime(b, ns=(stamp, stamp))
    calls = []
    def scan(p):
        calls.append(p)
        f = face(ph, p, len(calls)-1)
        return [(f.box, f.orig, f.emb, f.quality, f.thumb)]
    monkeypatch.setattr(ph, "scan_one", scan)
    found = ph.find_faces([a, b], tmp_path, {}, lambda *_: None, 0, 1)
    assert len(calls) == 2 and not np.array_equal(found[0].emb, found[1].emb)
    loaded = ph.load_face_cache(tmp_path)
    assert len(loaded) == 2
    monkeypatch.setattr(ph, "scan_one", lambda _: (_ for _ in ()).throw(AssertionError("should be cached")))
    again = ph.find_faces([a, b], tmp_path, loaded, lambda *_: None, 0, 1)
    assert len(again) == 2 and again.failures == 0


def test_weak_and_ambiguous_faces_are_not_named(ph, tmp_path, monkeypatch):
    path = tmp_path / "a.jpg"; path.write_bytes(b"photo")
    target = face(ph, path, 1)
    monkeypatch.setattr(ph, "run_exif", lambda *a: (_ for _ in ()).throw(AssertionError("must not write")))
    data = {"faces": [target], "members": [[0]], "root": str(tmp_path)}
    result = ph.stage_match(data, {"Alice": [face(ph,path).emb]}, {0: "Alice"}, None, lambda *_: None)
    assert result["tagged"] == 0 and result["uncertain"] == 1
    result = ph.stage_match(data, {"Alice": [target.emb], "Bob": [target.emb]}, {}, None, lambda *_: None)
    assert result["tagged"] == 0 and result["uncertain"] == 1


def test_detection_errors_are_reported(ph, tmp_path, monkeypatch):
    path = tmp_path / "bad.jpg"; path.write_bytes(b"bad")
    monkeypatch.setattr(ph, "scan_one", lambda _: None)
    result = ph.find_faces([path], tmp_path, {}, lambda *_: None, 0, 1)
    assert result.failures == 1
    assert "bad.jpg" in (tmp_path / ph.INFO_DIR / "Face Report.txt").read_text()


def test_model_download_is_pinned_verified_and_timed(ph, tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(ph, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(ph, "MODEL_URLS", {"test.onnx": "https://example.invalid/pinned"})
    monkeypatch.setattr(ph, "MODEL_SHA256", {"test.onnx": hashlib.sha256(b"valid").hexdigest()})
    (tmp_path / "test.onnx").write_bytes(b"corrupt" * 20000)
    calls = []
    def download(url, timeout):
        calls.append(timeout)
        return io.BytesIO(b"valid")
    monkeypatch.setattr(ph.urllib.request, "urlopen", download)
    ph.ensure_models(lambda *_: None)
    assert calls == [30] and (tmp_path / "test.onnx").read_bytes() == b"valid"
    ph.ensure_models(lambda *_: None)
    assert calls == [30]


def test_real_exiftool_preserves_regions_and_backups(ph, tmp_path):
    exe = str(Path(ph.APP_DIR) / "Exiftool" / "exiftool.exe")
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), np.zeros((100, 100, 3), np.uint8))
    ph.run_exif(exe, ["-XMP-mwg-rs:RegionInfo={AppliedToDimensions={W=100,H=100,Unit=pixel},RegionList=[{Area={X=0.2,Y=0.2,W=0.2,H=0.2,Unit=normalized},Name=Manual,Type=Face}]}", str(path)])
    before = path.read_bytes()
    target = face(ph, path, box=(.6, .6, .2, .2))
    data = {"faces": [target], "members": [[0]], "root": str(tmp_path)}
    result = ph.stage_match(data, {"Alice": [target.emb]}, {}, exe, lambda *_: None)
    assert result["tagged"] == 1 and result["failed"] == 0
    current = ph.exif_json(exe, ["-struct", "-RegionInfo"], [path])[0]
    assert {r["Name"] for r in current["RegionInfo"]["RegionList"]} == {"Manual", "Alice"}
    assert any(p.read_bytes() == before for p in (tmp_path / ph.INFO_DIR / "name-backups").iterdir())
    # A second run must not duplicate regions or tags.
    ph.stage_match(data, {"Alice": [target.emb]}, {}, exe, lambda *_: None)
    current = ph.exif_json(exe, ["-struct", "-RegionInfo"], [path])[0]
    assert len(current["RegionInfo"]["RegionList"]) == 2


def test_bad_model_download_is_rejected(ph, tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(ph, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(ph, "MODEL_URLS", {"test.onnx": "https://example.invalid/pinned"})
    monkeypatch.setattr(ph, "MODEL_SHA256", {"test.onnx": hashlib.sha256(b"valid").hexdigest()})
    monkeypatch.setattr(ph.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    with pytest.raises(RuntimeError, match="verification"):
        ph.ensure_models(lambda *_: None)
    assert not (tmp_path / "test.onnx").exists()
    assert not (tmp_path / "test.onnx.part").exists()
