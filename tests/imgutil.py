"""Test pictures made at run time (no binary fixtures are committed)."""
import numpy as np
import cv2


def photo_like(w=320, h=240, seed=1):
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        base[..., c] = 127 + 80 * np.sin(x / 37.0 + c) * np.cos(y / 29.0 - c) + 30 * rng.uniform(-1, 1)
    for _ in range(12):
        cx, cy = int(rng.randint(0, w)), int(rng.randint(0, h))
        r = int(rng.randint(10, 60))
        col = [int(v) for v in rng.randint(0, 255, 3)]
        cv2.circle(base, (cx, cy), r, col, -1)
    noise = rng.normal(0, 2, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def caption_like(w=400, h=400, seed=2):
    """A photo with a white band top and bottom and black caption lettering, like a meme."""
    img = photo_like(w, h, seed)
    img[:70, :] = 255
    cv2.putText(img, "WHEN THE CODE WORKS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    img[h - 70:, :] = 255
    cv2.putText(img, "FIRST TRY", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    return img


def encode(img, ext, params=None):
    ok, buf = cv2.imencode(ext, img, params or [])
    assert ok, f"could not encode {ext}"
    return buf.tobytes()