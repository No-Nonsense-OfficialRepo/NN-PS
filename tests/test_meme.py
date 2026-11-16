"""looks_like_meme on generated caption-style pictures."""
import numpy as np

from imgutil import caption_like, photo_like


def test_caption_picture_is_a_meme(ph):
    assert ph.looks_like_meme(caption_like()) is True


def test_plain_picture_is_not_a_meme(ph):
    assert ph.looks_like_meme(photo_like()) is False


def test_none_and_tiny_images_are_not_memes(ph):
    assert ph.looks_like_meme(None) is False
    assert ph.looks_like_meme(np.zeros((50, 50, 3), np.uint8)) is False