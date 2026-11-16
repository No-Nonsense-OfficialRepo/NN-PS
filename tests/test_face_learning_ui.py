import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

_app = None


@pytest.fixture
def window(ph, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    global _app
    _app = QApplication.instance() or QApplication([])
    saved = dict(ph.OPTS)
    monkeypatch.setattr(ph, "CONFIG", tmp_path / "config.json")
    w = ph.Main()
    w.root_folder = str(tmp_path)
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"photo")
    e = np.eye(128, dtype=np.float32)[0]
    w.data = {"faces": [ph.Face(str(path), (.1,.1,.2,.2), (100,100), e, 1., b"thumb")], "confirmed": {}}
    w.refs = {"Alice": [np.eye(128, dtype=np.float32)[1]]}
    w.spelling = {"alice": "Alice"}
    yield w
    w.phase = "welcome"
    w.close()
    ph.apply_options(saved)


@pytest.mark.parametrize("decision", [1, 2, 0])
def test_review_accept_reject_and_skip(ph, window, tmp_path, monkeypatch, decision):
    monkeypatch.setattr(ph.QDialog, "exec", lambda _: decision)
    window.review_uncertain_faces()
    if decision == 1:
        assert window.data["confirmed"] == {0: "Alice"}
        assert "Alice" in ph.load_people(tmp_path)
    elif decision == 2:
        assert window.data["confirmed"] == {0: None}
        assert ph.load_people(tmp_path) == {}
    else:
        assert window.data["confirmed"] == {}


def test_face_options_save_and_reset(ph, window):
    window.face_option_boxes["review_faces"].setChecked(True)
    window.face_option_boxes["remember_people"].setChecked(False)
    saved = ph.load_config()["options"]
    assert saved["review_faces"] is True and saved["remember_people"] is False
    window.reset_options()
    assert ph.load_config()["options"]["review_faces"] is False
    assert ph.load_config()["options"]["remember_people"] is True


def test_face_removal_preview_can_cancel_without_starting(ph, window, tmp_path, monkeypatch):
    monkeypatch.setattr(ph.QMessageBox, "exec", lambda _: 0)
    monkeypatch.setattr(window, "run_worker", lambda *a, **k: pytest.fail("must not remove without confirmation"))
    window.on_face_removal_preview({"root": str(tmp_path), "entries": [], "unreadable": [], "notes": []})
    assert window.phase == "welcome"


def test_reset_rescan_asks_about_even_one_face(ph, window, monkeypatch):
    data = dict(window.data, photos=1, dups=0, members=[[0]], groups={}, current={}, cache={}, rename_all=True)
    window.tasks = {"faces"}
    monkeypatch.setattr(window, "next_person", lambda: None)
    window.on_prepared(data)
    assert window.ask == [0]
