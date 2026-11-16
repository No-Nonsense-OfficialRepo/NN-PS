"""ffmpeg_command: the command is an argument list with the expected shape."""


def test_command_is_argument_list(ph, tmp_path):
    src = tmp_path / "in.mov"
    dest = tmp_path / "out.mp4"
    cmd = ph.ffmpeg_command("ffmpeg.exe", src, dest, "mp4", "libx264", 23)
    assert isinstance(cmd, list)
    assert all(isinstance(a, str) for a in cmd)
    assert cmd[0] == "ffmpeg.exe"
    assert cmd[-1] == str(dest)
    assert cmd[cmd.index("-i") + 1] == str(src)
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-crf") + 1] == "23"
    assert cmd[cmd.index("-f") + 1] == "mp4"
    assert "-c:a" in cmd


def test_h265_adds_hvc1_tag(ph, tmp_path):
    cmd = ph.ffmpeg_command("ffmpeg.exe", tmp_path / "a.mov", tmp_path / "b.mp4", "mp4", "libx265", 28)
    assert cmd[cmd.index("-tag:v") + 1] == "hvc1"
    assert cmd[cmd.index("-c:v") + 1] == "libx265"