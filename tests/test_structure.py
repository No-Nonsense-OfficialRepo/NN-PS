"""structure_problem: empty, cut-off and damaged files versus fine files with trailing data."""
import cv2

from imgutil import photo_like, encode


def test_empty_file_is_flagged(ph, tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    reason = ph.structure_problem(p, 0, ".jpg")
    assert reason is not None and "empty" in reason.lower()


def test_valid_jpeg_is_not_flagged(ph, tmp_path):
    p = tmp_path / "ok.jpg"
    data = encode(photo_like(), ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90])
    p.write_bytes(data)
    assert ph.structure_problem(p, len(data), ".jpg") is None


def test_truncated_jpeg_is_flagged(ph, tmp_path):
    data = encode(photo_like(), ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90])
    cut = data[: len(data) // 2]
    p = tmp_path / "cut.jpg"
    p.write_bytes(cut)
    assert ph.structure_problem(p, len(cut), ".jpg") is not None


def test_jpeg_with_small_trailing_data_containing_eoi_is_not_flagged(ph, tmp_path):
    data = encode(photo_like(), ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90])
    full = data + b"TRAILER\xff\xd9END"
    p = tmp_path / "trail.jpg"
    p.write_bytes(full)
    assert ph.structure_problem(p, len(full), ".jpg") is None


def test_motion_photo_with_large_appended_data_is_not_flagged(ph, tmp_path):
    # A complete JPEG (ends FF D9) with more than 1 KB of video appended after it.
    data = encode(photo_like(), ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90])
    full = data + b"ftypmp42" + bytes(range(256)) * 16
    p = tmp_path / "motion.jpg"
    p.write_bytes(full)
    assert ph.structure_problem(p, len(full), ".jpg") is None


def test_complete_png_is_not_flagged(ph, tmp_path):
    data = encode(photo_like(), ".png")
    p = tmp_path / "ok.png"
    p.write_bytes(data)
    assert ph.structure_problem(p, len(data), ".png") is None


def test_png_without_iend_is_flagged(ph, tmp_path):
    data = encode(photo_like(), ".png")
    cut = data[:-12]  # the IEND chunk is the last 12 bytes
    p = tmp_path / "noend.png"
    p.write_bytes(cut)
    reason = ph.structure_problem(p, len(cut), ".png")
    assert reason is not None, "PNG without IEND was not flagged"