"""Cancellation ends an in-flight ExifTool process and removes its argument file."""
import subprocess
from pathlib import Path

import pytest


def test_cancel_kills_exiftool(ph, tmp_path, monkeypatch):
    class PendingProcess:
        def __init__(self, args, **kwargs):
            self.args = args
            self.returncode = None
            self.killed = False

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            if timeout is not None:
                ph.CANCEL_EVENT.set()
                raise subprocess.TimeoutExpired(self.args, timeout)
            self.returncode = -1
            return b"", b""

        def kill(self):
            self.killed = True
            self.returncode = -1

    processes = []
    def start(*args, **kwargs):
        process = PendingProcess(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(ph.subprocess, "Popen", start)
    ph.CANCEL_EVENT.clear()
    try:
        with pytest.raises(ph.Cancelled):
            ph.run_exif(str(tmp_path / "exiftool.exe"), ["-ver"])
        assert processes[0].killed
        assert not Path(processes[0].args[2]).exists()
    finally:
        ph.CANCEL_EVENT.clear()
