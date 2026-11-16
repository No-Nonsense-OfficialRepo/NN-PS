"""Remembered-faces cache: JSON lines round trip, and old pickle files are not read."""
import json
import pickle
from pathlib import Path


def test_round_trip(ph, tmp_path):
    fresh = {"1234:5678901": {"emb": [0.25, -0.5, 0.125], "thumb": "aGVsbG8=", "box": [1, 2, 3, 4]}}
    ph.append_face_cache(tmp_path, fresh)
    assert ph.load_face_cache(tmp_path) == fresh


def test_two_appends_are_merged_on_load(ph, tmp_path):
    a = {"1:1": {"emb": [0.1], "thumb": "QQ=="}}
    b = {"2:2": {"emb": [0.2], "thumb": "Qg=="}}
    ph.append_face_cache(tmp_path, a)
    ph.append_face_cache(tmp_path, b)
    assert ph.load_face_cache(tmp_path) == {**a, **b}


def test_cache_file_is_json_lines(ph, tmp_path):
    fresh = {"1:1": {"emb": [0.5], "thumb": "QQ=="}}
    ph.append_face_cache(tmp_path, fresh)
    path = Path(tmp_path) / ph.INFO_DIR / ph.face_cache_name()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines, "cache file is empty"
    header = json.loads(lines[0])  # raises on pickle bytes
    assert header == list(ph.face_cache_id())
    merged = {}
    for line in lines[1:]:
        merged.update(json.loads(line))
    assert merged == fresh


def test_old_pickle_cache_is_not_read(ph, tmp_path):
    d = Path(tmp_path) / ph.INFO_DIR
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ph.face_cache_name(), "wb") as f:
        pickle.dump(ph.face_cache_id(), f)
        pickle.dump({"1:1": {"emb": [0.1]}}, f)
    assert ph.load_face_cache(tmp_path) == {}