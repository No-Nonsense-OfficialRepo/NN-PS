import json

import numpy as np
import pytest


def embedding(axis, offset=0):
    value = np.zeros(128, dtype=np.float32)
    value[axis] = 1
    value[10] = offset
    return value / np.linalg.norm(value)


def test_approved_people_persist_and_corrections_replace_labels(ph, tmp_path):
    e = embedding(0)
    ph.remember_person(tmp_path, "Alice", e)
    loaded = ph.load_people(tmp_path)
    np.testing.assert_array_equal(loaded["Alice"][0], e)
    ph.remember_person(tmp_path, "Bob", e)
    assert set(ph.load_people(tmp_path)) == {"Bob"}
    ph.remember_person(tmp_path, "bob", e)
    assert len(ph.load_people(tmp_path)["Bob"]) == 1


def test_calibration_requires_independent_approved_examples(ph):
    with pytest.raises(ValueError, match="three"):
        ph.calibrate_people({"Alice": [embedding(0)]})
    refs = {"Alice": [embedding(0, d) for d in (0, .1, -.1)],
            "Bob": [embedding(1, d) for d in (0, .1, -.1)]}
    result = ph.calibrate_people(refs)
    assert result["same_person_min"] < 1  # self-match excluded
    assert result["samples"] == 6 and result["people"] == 2
    assert result["threshold"] >= ph.MATCH_T
    assert result["sample_accept_rate"] == 1


def test_calibration_expires_when_approved_people_change(ph, tmp_path):
    refs = {"Alice": [embedding(0, d) for d in (0, .1, -.1)],
            "Bob": [embedding(1, d) for d in (0, .1, -.1)]}
    result = ph.calibrate_people(refs)
    result["threshold"] = .8
    ph.atomic_json(tmp_path / ph.INFO_DIR / "face-calibration.json", result)
    assert ph.matching_threshold(tmp_path, refs) == .8
    refs["Alice"].append(embedding(0, .2))
    assert ph.matching_threshold(tmp_path, refs) == ph.MATCH_T


def test_rejected_face_is_not_automatically_named(ph, tmp_path, monkeypatch):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"photo")
    e = embedding(0)
    f = ph.Face(str(path), (.1,.1,.2,.2), (100,100), e, 1., b"thumb")
    monkeypatch.setattr(ph, "run_exif", lambda *a: pytest.fail("rejected face must not be saved"))
    result = ph.stage_match({"faces": [f], "members": [[0]], "root": str(tmp_path), "confirmed": {0: None}},
                            {"Alice": [e]}, {}, None, lambda *_: None)
    assert result["tagged"] == 0


def test_scan_timings_show_cache_reuse(ph, tmp_path, monkeypatch):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"photo")
    monkeypatch.setattr(ph, "scan_one", lambda _: [])
    ph.find_faces([path], tmp_path, {}, lambda *_: None, 0, 1)
    ph.find_faces([path], tmp_path, ph.load_face_cache(tmp_path), lambda *_: None, 0, 1)
    rows = [json.loads(line) for line in (tmp_path / ph.INFO_DIR / "face-timings.jsonl").read_text().splitlines()]
    assert rows[0]["scan_attempts"] == 1 and rows[0]["cache_hits"] == 0
    assert rows[1]["scan_attempts"] == 0 and rows[1]["cache_hits"] == 1
    assert all(row["hash_bytes"] == len(b"photo") and row["total_seconds"] >= row["hash_seconds"] for row in rows)
