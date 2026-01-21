from vmem.utils import cosine_similarity, normalize_text


def test_normalize_text():
    assert normalize_text("  Hello   World ") == "hello world"


def test_cosine_similarity_basic():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_mismatch():
    assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0
