"""name_base / has_copy_suffix: which file names count as named copies."""


def test_photo_1_is_not_a_named_copy(ph):
    assert ph.has_copy_suffix("photo_1") is False


def test_photo_2_is_not_a_named_copy(ph):
    assert ph.has_copy_suffix("photo_2") is False


def test_photo_1_and_photo_2_have_different_bases(ph):
    assert ph.name_base("photo_1") != ph.name_base("photo_2")


def test_hyphen_number_alone_is_not_a_copy(ph):
    assert ph.has_copy_suffix("photo-1") is False


def test_paren_number_is_a_copy(ph):
    assert ph.has_copy_suffix("photo (1)") is True
    assert ph.name_base("photo (1)") == ph.name_base("photo")


def test_plain_name_is_not_a_copy(ph):
    assert ph.has_copy_suffix("photo") is False


def test_dash_copy_and_copy_of_are_copies(ph):
    assert ph.has_copy_suffix("photo - Copy") is True
    assert ph.name_base("photo - Copy") == "photo"
    assert ph.has_copy_suffix("Copy of photo") is True
    assert ph.name_base("Copy of photo") == "photo"