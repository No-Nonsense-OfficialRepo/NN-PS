"""verify_pair: same picture, resized copy, or different picture."""
import cv2
import numpy as np

from imgutil import photo_like, encode


def _feat(ph, path):
    f = ph.look_features(path)
    assert f is not None, f"look_features could not read {path.name}"
    return f


def test_identical_pixels_in_different_files(ph, tmp_path):
    img = photo_like(seed=5)
    a = tmp_path / "a.png"
    b = tmp_path / "b.bmp"
    cv2.imwrite(str(a), img)
    cv2.imwrite(str(b), img)
    assert a.read_bytes() != b.read_bytes()
    result = ph.verify_pair(_feat(ph, a), _feat(ph, b), named=False, web=True)
    assert result is not None and "identical" in result


def test_resized_copy_is_recognised(ph, tmp_path):
    img = photo_like(320, 240, seed=5)
    small = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
    a = tmp_path / "orig.jpg"
    b = tmp_path / "small.jpg"
    a.write_bytes(encode(img, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]))
    b.write_bytes(encode(small, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85]))
    result = ph.verify_pair(_feat(ph, a), _feat(ph, b), named=False, web=False)
    assert result == "resized copy", f"expected 'resized copy', got {result!r}"


def test_different_picture_is_not_matched(ph, tmp_path):
    img = photo_like(320, 240, seed=5)
    rng = np.random.RandomState(7)
    other = rng.randint(0, 256, (240, 320, 3)).astype(np.uint8)
    a = tmp_path / "a.jpg"
    b = tmp_path / "other.jpg"
    a.write_bytes(encode(img, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]))
    b.write_bytes(encode(other, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]))
    result = ph.verify_pair(_feat(ph, a), _feat(ph, b), named=False, web=False)
    assert result is None, f"different pictures matched as {result!r}"