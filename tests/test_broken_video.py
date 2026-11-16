"""Video checks and broken-file moves use disposable files only."""
import json
import subprocess


def test_ffprobe_checks_video_stream_and_reports_errors(ph, tmp_path, monkeypatch):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"test")
    calls = []

    def probe(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, json.dumps({"streams": [{"index": 0}]}).encode(), b"")

    monkeypatch.setattr(ph.subprocess, "run", probe)
    assert ph.check_video_file(video, "ffprobe.exe") == (None, None)
    assert calls[0][1]["timeout"] == 60
    assert calls[0][0][-1] == str(video)

    monkeypatch.setattr(ph.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, b'{"streams": []}', b""))
    assert ph.check_video_file(video, "ffprobe.exe")[0] == "no video stream was found"
    monkeypatch.setattr(ph.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 1, b"", b"invalid"))
    assert ph.check_video_file(video, "ffprobe.exe")[0] == "video could not be read by ffprobe"


def test_broken_file_move_can_be_undone(ph, tmp_path, monkeypatch):
    source = tmp_path / "damaged.jpg"
    source.write_bytes(b"broken")
    monkeypatch.setattr(ph, "walk_media", lambda root: [source])
    monkeypatch.setattr(ph, "exif_json", lambda *args, **kwargs: [])
    monkeypatch.setattr(ph, "find_ffprobe", lambda: None)
    ph.start_move_run(tmp_path)
    result = ph.stage_broken(tmp_path, None, lambda *_: None)
    assert result["found"] == 1
    assert not source.exists()
    assert ph.undo_last_run(tmp_path)["restored"] == 1
    assert source.read_bytes() == b"broken"
